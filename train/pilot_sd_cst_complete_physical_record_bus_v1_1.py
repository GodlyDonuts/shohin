#!/usr/bin/env python3
"""Training-only declaration-key repair for the complete physical record bus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    cosine_scale,
    sha256_file,
)
from pilot_sd_cst_complete_physical_record_bus import (
    PHYSICAL_CHECKPOINT_SHA256,
)
from pilot_sd_cst_physical_record_bus import (
    ARCHITECTURE,
    JOINT_CHECKPOINT_SHA256,
    _state_digest,
)
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from pilot_sd_cst_renderer_orbit import (
    CONSUMED_TRAIN_SHA256,
    GLOBAL_PARAMETER_CAP,
    _minimum_rate,
    _source_digest,
    evaluate,
    expand_orbit,
    load_consumed_train,
    loss_groups,
    partition_rows,
)
from sd_cst_complete_physical_record_bus import local_completion_trainable_names
from sd_cst_complete_physical_record_bus_v1_1 import (
    CompletePhysicalRecordBusCompilerV1_1,
    declaration_repair_trainable_names,
    freeze_to_declaration_repair,
)
from sd_cst_physical_record_bus import physical_record_trainable_names
from sd_cst_renderer_orbit import HELD_OUT_RENDERERS, TRAIN_RENDERERS


V1_CHECKPOINT_SHA256 = (
    "30b75305031b1e2f67a24f98b4907d2d65bc847310ea406d25f34c7b9611e1b4"
)


def _named_state(
    model: CompletePhysicalRecordBusCompilerV1_1,
    names: frozenset[str],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in names
    }


def _query_names(
    model: CompletePhysicalRecordBusCompilerV1_1,
) -> frozenset[str]:
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith("local_query_")
    )


def initialize_model(
    joint_checkpoint: Path,
    physical_checkpoint: Path,
    v1_checkpoint: Path,
    device: torch.device,
) -> tuple[
    CompletePhysicalRecordBusCompilerV1_1,
    dict[str, object],
    str,
    str,
    str,
]:
    if sha256_file(joint_checkpoint) != JOINT_CHECKPOINT_SHA256:
        raise ValueError("v1.1 joint checkpoint hash differs")
    if sha256_file(physical_checkpoint) != PHYSICAL_CHECKPOINT_SHA256:
        raise ValueError("v1.1 physical checkpoint hash differs")
    if sha256_file(v1_checkpoint) != V1_CHECKPOINT_SHA256:
        raise ValueError("v1.1 local parent checkpoint hash differs")
    joint = torch.load(joint_checkpoint, map_location="cpu", weights_only=False)
    physical = torch.load(
        physical_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    v1 = torch.load(v1_checkpoint, map_location="cpu", weights_only=False)
    if joint.get("schema") != "r12_sd_cst_renderer_native_joint_pilot_v1":
        raise ValueError("v1.1 joint schema differs")
    if physical.get("schema") != "r12_sd_cst_physical_record_bus_pilot_v1":
        raise ValueError("v1.1 physical schema differs")
    if v1.get("schema") != "r12_sd_cst_complete_physical_record_bus_pilot_v1":
        raise ValueError("v1.1 local parent schema differs")
    if physical.get("parent_checkpoint_sha256") != JOINT_CHECKPOINT_SHA256:
        raise ValueError("v1.1 physical parent receipt differs")
    for receipt in (joint, physical, v1):
        if (
            receipt.get("development_accesses") != 0
            or receipt.get("confirmation_accesses") != 0
        ):
            raise ValueError("v1.1 parent has scored access")

    model = CompletePhysicalRecordBusCompilerV1_1(**ARCHITECTURE)
    local_names = local_completion_trainable_names(model)
    expected_missing = physical_record_trainable_names(model) | local_names
    missing, unexpected = model.load_state_dict(joint["state"], strict=False)
    if set(missing) != expected_missing or unexpected:
        raise ValueError("v1.1 joint state contract differs")

    record_state = physical.get("arms", {}).get("independent")
    if not isinstance(record_state, dict):
        raise ValueError("v1.1 independent record state is absent")
    if set(record_state) != physical_record_trainable_names(model):
        raise ValueError("v1.1 record state keys differ")
    current = model.state_dict()
    for name, tensor in record_state.items():
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise ValueError("v1.1 record state tensor differs")
        current[name].copy_(tensor)

    v1_state = v1.get("local_state")
    v1_expected = local_names - {"local_declaration_key_projection.weight"}
    if not isinstance(v1_state, dict) or set(v1_state) != v1_expected:
        raise ValueError("v1.1 local parent state keys differ")
    query_names = _query_names(model)
    if len(query_names) != 8:
        raise ValueError("v1.1 query state contract differs")
    for name in query_names:
        tensor = v1_state[name]
        if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
            raise ValueError("v1.1 query state tensor differs")
        current[name].copy_(tensor)

    trainable = freeze_to_declaration_repair(model)
    repair_names = declaration_repair_trainable_names(model)
    excluded_digest = frozen_state_digest(model, repair_names)
    repair_initial_sha256 = _state_digest(_named_state(model, repair_names))
    query_state_sha256 = _state_digest(_named_state(model, query_names))
    model.to(device)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    if complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("v1.1 complete system reaches global cap")
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
    return (
        model,
        parameters,
        excluded_digest,
        repair_initial_sha256,
        query_state_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
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
        raise SystemExit(f"refusing existing v1.1 output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("v1.1 pilot requires bf16 CUDA")
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
    (
        model,
        parameters,
        excluded_digest,
        repair_initial_sha256,
        query_state_sha256,
    ) = initialize_model(
        args.joint_checkpoint,
        args.physical_checkpoint,
        args.v1_checkpoint,
        device,
    )
    initial_heldout = evaluate(
        model,
        heldout_groups,
        args.eval_family_batch_size,
        device,
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(fit_groups) / args.family_batch_size)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_scale(step, total_updates, args.warmup),
    )
    rng = random.Random(args.seed)
    history: list[dict[str, object]] = []
    update = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(fit_groups)))
        rng.shuffle(order)
        totals: Counter[str] = Counter()
        seen = 0
        for start in range(0, len(order), args.family_batch_size):
            groups = [
                fit_groups[index]
                for index in order[start : start + args.family_batch_size]
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = loss_groups(
                    model,
                    groups,
                    device,
                    args.consistency_weight,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite v1.1 gradient")
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
            args.eval_family_batch_size,
            device,
        )
        record = {
            "epoch": epoch + 1,
            "updates": update,
            "fit_losses": {
                field: value / seen for field, value in sorted(totals.items())
            },
            "heldout_min_rates": {
                field: _minimum_rate(heldout, field)
                for field in (
                    "packet",
                    "initial",
                    "query",
                    "binding_pointer",
                    "initial_pointer",
                    "event_pointer",
                )
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    fit = evaluate(model, fit_groups, args.eval_family_batch_size, device)
    heldout = evaluate(model, heldout_groups, args.eval_family_batch_size, device)
    repair_names = declaration_repair_trainable_names(model)
    final_excluded_digest = frozen_state_digest(model, repair_names)
    gates = {
        "fit_packet_at_least_99pct": _minimum_rate(fit, "packet") >= 0.99,
        "heldout_packet_at_least_95pct": _minimum_rate(heldout, "packet")
        >= 0.95,
        "heldout_initial_at_least_95pct": _minimum_rate(heldout, "initial")
        >= 0.95,
        "heldout_query_at_least_99pct": _minimum_rate(heldout, "query") >= 0.99,
        "heldout_query_pointer_at_least_99pct": (
            _minimum_rate(heldout, "query_pointer") >= 0.99
        ),
        "heldout_binding_pointer_at_least_99pct": (
            _minimum_rate(heldout, "binding_pointer") >= 0.99
        ),
        "heldout_initial_pointer_at_least_99pct": (
            _minimum_rate(heldout, "initial_pointer") >= 0.99
        ),
        "heldout_event_pointer_at_least_99pct": (
            _minimum_rate(heldout, "event_pointer") >= 0.99
        ),
        "heldout_kind_identity_amount_all_99pct": all(
            _minimum_rate(heldout, field) >= 0.99
            for field in ("kind", "identity", "amount")
        ),
        "excluded_state_byte_identical": excluded_digest
        == final_excluded_digest,
        "complete_system_below_200m": parameters["complete_system"]
        < GLOBAL_PARAMETER_CAP,
        "scored_access_zero": True,
    }
    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save(
        {
            "schema": "r12_sd_cst_complete_physical_record_bus_pilot_v1_1",
            "seed": args.seed,
            "joint_checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "physical_checkpoint_sha256": PHYSICAL_CHECKPOINT_SHA256,
            "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
            "repair_initial_sha256": repair_initial_sha256,
            "query_state_sha256": query_state_sha256,
            "declaration_state": _named_state(model, repair_names),
            "excluded_state_digest": final_excluded_digest,
            "development_accesses": 0,
            "confirmation_accesses": 0,
            "score_eligible": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": "r12_sd_cst_complete_physical_record_bus_pilot_report_v1_1",
        "decision": (
            "retain_declaration_key_repair_for_fresh_board"
            if all(gates.values())
            else "reject_declaration_key_repair"
        ),
        "seed": args.seed,
        "source": {
            "train_sha256": CONSUMED_TRAIN_SHA256,
            "joint_checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "physical_checkpoint_sha256": PHYSICAL_CHECKPOINT_SHA256,
            "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
            "fit_source_sha256": _source_digest(fit_source),
            "heldout_source_sha256": _source_digest(heldout_source),
        },
        "architecture": ARCHITECTURE,
        "repair_initial_sha256": repair_initial_sha256,
        "query_state_sha256": query_state_sha256,
        "parameters": parameters,
        "initial_heldout": initial_heldout,
        "history": history,
        "fit": fit,
        "heldout": heldout,
        "gates": gates,
        "excluded_state_digest": excluded_digest,
        "final_excluded_state_digest": final_excluded_digest,
        "training": {
            "epochs": args.epochs,
            "updates": update,
            "family_batch_size": args.family_batch_size,
            "lr": args.lr,
            "warmup": args.warmup,
            "consistency_weight": args.consistency_weight,
            "elapsed_seconds": time.time() - started,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": "Consumed training rows only; declaration-key repair "
        "gate, not native reasoning or fresh generalization.",
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "heldout_min_packet": _minimum_rate(heldout, "packet"),
                "heldout_min_initial": _minimum_rate(heldout, "initial"),
                "heldout_min_query": _minimum_rate(heldout, "query"),
                "parameters": parameters,
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
