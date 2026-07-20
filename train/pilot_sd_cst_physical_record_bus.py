#!/usr/bin/env python3
"""Training-only physical-record write-bus falsifier."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    cosine_scale,
    sha256_file,
)
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from pilot_sd_cst_renderer_orbit import (
    CONSUMED_TRAIN_SHA256,
    GLOBAL_PARAMETER_CAP,
    OrbitPilotRow,
    _minimum_rate,
    _source_digest,
    evaluate,
    expand_orbit,
    load_consumed_train,
    loss_groups,
    partition_rows,
)
from sd_cst_physical_record_bus import (
    PhysicalRecordBusCompiler,
    freeze_to_physical_record_bus,
    physical_record_trainable_names,
)
from sd_cst_renderer_orbit import HELD_OUT_RENDERERS, TRAIN_RENDERERS


JOINT_CHECKPOINT_SHA256 = (
    "4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d"
)
ARM_ORDER = ("constrained", "independent")
ARCHITECTURE = {
    "record_width": 384,
    "record_heads": 6,
    "record_layers": 4,
    "record_set_layers": 2,
    "record_ff": 1536,
    "max_line_bytes": 144,
    "sinkhorn_steps": 8,
}


def _trainable_state(model: PhysicalRecordBusCompiler) -> dict[str, torch.Tensor]:
    names = physical_record_trainable_names(model)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in names
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_trainable_state(
    model: PhysicalRecordBusCompiler,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = physical_record_trainable_names(model)
    if set(state) != expected:
        raise ValueError("physical-record initialization key contract differs")
    current = model.state_dict()
    for name, tensor in state.items():
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise ValueError("physical-record initialization tensor contract differs")
        current[name].copy_(tensor)


def initialize_model(
    joint_checkpoint: Path,
    device: torch.device,
    *,
    constrained_assignment: bool,
    initial_record_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[PhysicalRecordBusCompiler, dict[str, object], str]:
    if sha256_file(joint_checkpoint) != JOINT_CHECKPOINT_SHA256:
        raise ValueError("physical-record parent checkpoint hash differs")
    checkpoint = torch.load(joint_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_sd_cst_renderer_native_joint_pilot_v1":
        raise ValueError("physical-record parent checkpoint schema differs")
    if (
        checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
    ):
        raise ValueError("physical-record parent has scored access")

    model = PhysicalRecordBusCompiler(
        constrained_assignment=constrained_assignment,
        **ARCHITECTURE,
    )
    expected_missing = physical_record_trainable_names(model)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    if set(missing) != expected_missing or unexpected:
        raise ValueError("physical-record parent state contract differs")
    if initial_record_state is not None:
        _load_trainable_state(model, initial_record_state)
    trainable = freeze_to_physical_record_bus(model)
    parent_digest = frozen_state_digest(model, expected_missing)
    model.to(device)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    if complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("physical-record complete system reaches global cap")
    parameters: dict[str, object] = {
        "base": BASE_PARAMETERS,
        "compiler": compiler,
        "motor": MOTOR_PARAMETERS,
        "reader": READER_PARAMETERS,
        "complete_system": complete,
        "headroom": GLOBAL_PARAMETER_CAP - complete,
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": list(trainable),
    }
    return model, parameters, parent_digest


def train_arm(
    *,
    name: str,
    model: PhysicalRecordBusCompiler,
    parameters: Mapping[str, object],
    parent_digest: str,
    fit_groups: Sequence[Sequence[OrbitPilotRow]],
    heldout_groups: Sequence[Sequence[OrbitPilotRow]],
    device: torch.device,
    seed: int,
    epochs: int,
    family_batch_size: int,
    eval_family_batch_size: int,
    lr: float,
    warmup: int,
    consistency_weight: float,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(fit_groups) / family_batch_size)
    total_updates = updates_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_scale(step, total_updates, warmup),
    )
    rng = random.Random(seed)
    history: list[dict[str, object]] = []
    update = 0
    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(fit_groups)))
        rng.shuffle(order)
        totals: Counter[str] = Counter()
        seen = 0
        for start in range(0, len(order), family_batch_size):
            groups = [
                fit_groups[index]
                for index in order[start : start + family_batch_size]
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = loss_groups(
                    model,
                    groups,
                    device,
                    consistency_weight,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(f"non-finite physical-record gradient in {name}")
            optimizer.step()
            scheduler.step()
            update += 1
            rows_seen = sum(len(group) for group in groups)
            seen += rows_seen
            for field, value in pieces.items():
                totals[field] += value * rows_seen
        heldout = evaluate(
            model,
            heldout_groups,
            eval_family_batch_size,
            device,
        )
        record = {
            "arm": name,
            "epoch": epoch + 1,
            "updates": update,
            "fit_losses": {
                field: value / seen for field, value in sorted(totals.items())
            },
            "heldout_min_rates": {
                field: _minimum_rate(heldout, field)
                for field in (
                    "packet",
                    "whole_tape",
                    "kind",
                    "identity",
                    "amount",
                    "line_pointer",
                    "event_pointer",
                )
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    fit = evaluate(model, fit_groups, eval_family_batch_size, device)
    heldout = evaluate(model, heldout_groups, eval_family_batch_size, device)
    trainable_names = physical_record_trainable_names(model)
    final_parent_digest = frozen_state_digest(model, trainable_names)
    result: dict[str, object] = {
        "name": name,
        "constrained_assignment": model.constrained_assignment,
        "training": {
            "epochs": epochs,
            "updates": update,
            "family_batch_size": family_batch_size,
            "lr": lr,
            "warmup": warmup,
            "consistency_weight": consistency_weight,
            "elapsed_seconds": time.time() - started,
        },
        "parameters": dict(parameters),
        "frozen_parent_digest": parent_digest,
        "final_frozen_parent_digest": final_parent_digest,
        "history": history,
        "fit": fit,
        "heldout": heldout,
    }
    return result, _trainable_state(model)


def _gates(arms: Mapping[str, Mapping[str, object]]) -> dict[str, bool]:
    treatment = arms["constrained"]
    control = arms["independent"]
    heldout = treatment["heldout"]
    fit = treatment["fit"]
    control_heldout = control["heldout"]
    return {
        "heldout_initial_at_least_95pct": _minimum_rate(heldout, "initial")
        >= 0.95,
        "heldout_query_at_least_99pct": _minimum_rate(heldout, "query") >= 0.99,
        "heldout_binding_pointer_at_least_99pct": (
            _minimum_rate(heldout, "binding_pointer") >= 0.99
        ),
        "heldout_initial_pointer_at_least_99pct": (
            _minimum_rate(heldout, "initial_pointer") >= 0.99
        ),
        "heldout_line_pointer_at_least_95pct": (
            _minimum_rate(heldout, "line_pointer") >= 0.95
        ),
        "heldout_event_pointer_at_least_90pct": (
            _minimum_rate(heldout, "event_pointer") >= 0.90
        ),
        "heldout_kind_at_least_95pct": _minimum_rate(heldout, "kind") >= 0.95,
        "heldout_identity_at_least_90pct": _minimum_rate(heldout, "identity")
        >= 0.90,
        "heldout_amount_at_least_95pct": _minimum_rate(heldout, "amount") >= 0.95,
        "heldout_packet_at_least_80pct": _minimum_rate(heldout, "packet") >= 0.80,
        "fit_line_pointer_at_least_99pct": (
            _minimum_rate(fit, "line_pointer") >= 0.99
        ),
        "fit_event_pointer_at_least_99pct": (
            _minimum_rate(fit, "event_pointer") >= 0.99
        ),
        "fit_packet_at_least_95pct": _minimum_rate(fit, "packet") >= 0.95,
        "constrained_packet_beats_independent_by_5pp": (
            _minimum_rate(heldout, "packet")
            >= _minimum_rate(control_heldout, "packet") + 0.05
        ),
        "constrained_line_pointer_beats_independent_by_5pp": (
            _minimum_rate(heldout, "line_pointer")
            >= _minimum_rate(control_heldout, "line_pointer") + 0.05
        ),
        "constrained_event_pointer_beats_independent_by_5pp": (
            _minimum_rate(heldout, "event_pointer")
            >= _minimum_rate(control_heldout, "event_pointer") + 0.05
        ),
        "frozen_parent_byte_identical_both_arms": all(
            arm["frozen_parent_digest"] == arm["final_frozen_parent_digest"]
            for arm in arms.values()
        ),
        "matched_parameter_counts": (
            treatment["parameters"] == control["parameters"]
        ),
        "complete_system_below_200m": (
            int(treatment["parameters"]["complete_system"])
            < GLOBAL_PARAMETER_CAP
        ),
        "scored_access_zero": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-semantics", type=int, default=12_000)
    parser.add_argument("--heldout-semantics", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--family-batch-size", type=int, default=8)
    parser.add_argument("--eval-family-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing physical-record output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("physical-record pilot requires bf16 CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    source_rows = load_consumed_train(args.train_jsonl)
    fit_source, heldout_source = partition_rows(
        source_rows,
        args.fit_semantics,
        args.heldout_semantics,
    )
    fit_groups = expand_orbit(fit_source, TRAIN_RENDERERS)
    heldout_groups = expand_orbit(heldout_source, HELD_OUT_RENDERERS)

    first_model, parameters, first_parent_digest = initialize_model(
        args.joint_checkpoint,
        device,
        constrained_assignment=True,
    )
    initial_record_state = _trainable_state(first_model)
    initial_record_state_sha256 = _state_digest(initial_record_state)
    arms: dict[str, dict[str, object]] = {}
    arm_states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ARM_ORDER:
        if name == "constrained":
            model = first_model
            arm_parameters = parameters
            parent_digest = first_parent_digest
        else:
            model, arm_parameters, parent_digest = initialize_model(
                args.joint_checkpoint,
                device,
                constrained_assignment=False,
                initial_record_state=initial_record_state,
            )
            if _state_digest(_trainable_state(model)) != initial_record_state_sha256:
                raise ValueError("physical-record matched initialization differs")
        result, state = train_arm(
            name=name,
            model=model,
            parameters=arm_parameters,
            parent_digest=parent_digest,
            fit_groups=fit_groups,
            heldout_groups=heldout_groups,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            family_batch_size=args.family_batch_size,
            eval_family_batch_size=args.eval_family_batch_size,
            lr=args.lr,
            warmup=args.warmup,
            consistency_weight=args.consistency_weight,
        )
        arms[name] = result
        arm_states[name] = state
        del model
        if name == "constrained":
            del first_model
        torch.cuda.empty_cache()

    gates = _gates(arms)
    absolute_names = (
        "heldout_initial_at_least_95pct",
        "heldout_query_at_least_99pct",
        "heldout_binding_pointer_at_least_99pct",
        "heldout_initial_pointer_at_least_99pct",
        "heldout_line_pointer_at_least_95pct",
        "heldout_event_pointer_at_least_90pct",
        "heldout_kind_at_least_95pct",
        "heldout_identity_at_least_90pct",
        "heldout_amount_at_least_95pct",
        "heldout_packet_at_least_80pct",
        "fit_line_pointer_at_least_99pct",
        "fit_event_pointer_at_least_99pct",
        "fit_packet_at_least_95pct",
        "frozen_parent_byte_identical_both_arms",
        "matched_parameter_counts",
        "complete_system_below_200m",
        "scored_access_zero",
    )
    differential_names = (
        "constrained_packet_beats_independent_by_5pp",
        "constrained_line_pointer_beats_independent_by_5pp",
        "constrained_event_pointer_beats_independent_by_5pp",
    )
    absolute_pass = all(gates[name] for name in absolute_names)
    differential_pass = all(gates[name] for name in differential_names)
    if absolute_pass and differential_pass:
        decision = "retain_physical_record_bus_and_one_to_one_assignment"
    elif absolute_pass:
        decision = "retain_physical_record_bus_reject_one_to_one_attribution"
    else:
        decision = "reject_or_revise_physical_record_bus"

    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save(
        {
            "schema": "r12_sd_cst_physical_record_bus_pilot_v1",
            "seed": args.seed,
            "parent_checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "architecture": ARCHITECTURE,
            "initial_record_state_sha256": initial_record_state_sha256,
            "initial_record_state": initial_record_state,
            "arms": arm_states,
            "development_accesses": 0,
            "confirmation_accesses": 0,
            "score_eligible": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": "r12_sd_cst_physical_record_bus_pilot_report_v1",
        "decision": decision,
        "seed": args.seed,
        "source": {
            "train_sha256": CONSUMED_TRAIN_SHA256,
            "joint_checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "fit_source_sha256": _source_digest(fit_source),
            "heldout_source_sha256": _source_digest(heldout_source),
        },
        "architecture": ARCHITECTURE,
        "initial_record_state_sha256": initial_record_state_sha256,
        "arm_order": list(ARM_ORDER),
        "arms": arms,
        "gates": gates,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": "Consumed training rows only; conventional compiler "
        "mechanics control, not native reasoning or a novelty result.",
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    treatment = arms["constrained"]["heldout"]
    control = arms["independent"]["heldout"]
    print(
        json.dumps(
            {
                "decision": decision,
                "treatment_min_packet": _minimum_rate(treatment, "packet"),
                "control_min_packet": _minimum_rate(control, "packet"),
                "treatment_min_event_pointer": _minimum_rate(
                    treatment,
                    "event_pointer",
                ),
                "control_min_event_pointer": _minimum_rate(
                    control,
                    "event_pointer",
                ),
                "parameters": arms["constrained"]["parameters"],
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
