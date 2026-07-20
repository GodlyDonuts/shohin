#!/usr/bin/env python3
"""Training-only renderer-native program decoder pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from sd_cst_renderer_native_program import (
    RendererNativeProgramCompiler,
    freeze_to_renderer_native_joint,
    freeze_to_renderer_native_program,
    renderer_native_joint_trainable_names,
    renderer_native_program_trainable_names,
)
from sd_cst_renderer_orbit import HELD_OUT_RENDERERS, TRAIN_RENDERERS


ORBIT_CHECKPOINT_SHA256 = (
    "2e019b81406bb90e539665271c9893a0e568e0177396243ac427f17d8ca51eca"
)


def frozen_state_digest(
    model: RendererNativeProgramCompiler,
    trainable_names: frozenset[str],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name in trainable_names:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize_model(
    orbit_checkpoint: Path,
    device: torch.device,
    *,
    train_shared_orbit: bool = False,
) -> tuple[RendererNativeProgramCompiler, dict[str, object], str]:
    if sha256_file(orbit_checkpoint) != ORBIT_CHECKPOINT_SHA256:
        raise ValueError("renderer-native orbit checkpoint hash differs")
    checkpoint = torch.load(orbit_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_sd_cst_renderer_orbit_training_pilot_v1_2":
        raise ValueError("renderer-native orbit checkpoint schema differs")
    if (
        checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
    ):
        raise ValueError("renderer-native parent has scored access")

    model = RendererNativeProgramCompiler()
    expected_missing = renderer_native_program_trainable_names(model)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    if set(missing) != expected_missing or unexpected:
        raise ValueError("renderer-native parent state contract differs")
    if train_shared_orbit:
        trainable_names = renderer_native_joint_trainable_names(model)
        trainable = freeze_to_renderer_native_joint(model)
    else:
        trainable_names = expected_missing
        trainable = freeze_to_renderer_native_program(model)
    parent_digest = frozen_state_digest(model, trainable_names)
    model.to(device)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    if complete >= GLOBAL_PARAMETER_CAP:
        raise ValueError("renderer-native complete system reaches global cap")
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
        "train_shared_orbit": train_shared_orbit,
    }
    return model, parameters, parent_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--orbit-checkpoint", type=Path, required=True)
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
    parser.add_argument("--train-shared-orbit", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing renderer-native output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("renderer-native pilot requires bf16 CUDA")
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
    model, parameters, parent_digest = initialize_model(
        args.orbit_checkpoint,
        device,
        train_shared_orbit=args.train_shared_orbit,
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
                raise RuntimeError("non-finite renderer-native gradient")
            optimizer.step()
            scheduler.step()
            update += 1
            rows_seen = sum(len(group) for group in groups)
            seen += rows_seen
            for name, value in pieces.items():
                totals[name] += value * rows_seen
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
                name: value / seen for name, value in sorted(totals.items())
            },
            "heldout_min_rates": {
                field: _minimum_rate(heldout, field)
                for field in (
                    "packet",
                    "whole_tape",
                    "kind",
                    "amount",
                    "line_pointer",
                    "event_pointer",
                )
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    fit = evaluate(model, fit_groups, args.eval_family_batch_size, device)
    heldout = evaluate(model, heldout_groups, args.eval_family_batch_size, device)
    trainable_names = (
        renderer_native_joint_trainable_names(model)
        if args.train_shared_orbit
        else renderer_native_program_trainable_names(model)
    )
    final_parent_digest = frozen_state_digest(model, trainable_names)
    gates = {
        "heldout_initial_at_least_95pct": _minimum_rate(heldout, "initial") >= 0.95,
        "heldout_kind_at_least_95pct": _minimum_rate(heldout, "kind") >= 0.95,
        "heldout_identity_at_least_90pct": _minimum_rate(heldout, "identity") >= 0.90,
        "heldout_amount_at_least_95pct": _minimum_rate(heldout, "amount") >= 0.95,
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
        "heldout_line_pointer_at_least_95pct": (
            _minimum_rate(heldout, "line_pointer") >= 0.95
        ),
        "heldout_event_pointer_at_least_90pct": (
            _minimum_rate(heldout, "event_pointer") >= 0.90
        ),
        "heldout_packet_at_least_80pct": _minimum_rate(heldout, "packet") >= 0.80,
        "fit_event_pointer_at_least_99pct": (
            _minimum_rate(fit, "event_pointer") >= 0.99
        ),
        "frozen_parent_byte_identical": parent_digest == final_parent_digest,
        "complete_system_below_200m": parameters["complete_system"]
        < GLOBAL_PARAMETER_CAP,
        "scored_access_zero": True,
    }

    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    checkpoint_schema = (
        "r12_sd_cst_renderer_native_joint_pilot_v1"
        if args.train_shared_orbit
        else "r12_sd_cst_renderer_native_program_pilot_v1"
    )
    report_schema = (
        "r12_sd_cst_renderer_native_joint_pilot_report_v1"
        if args.train_shared_orbit
        else "r12_sd_cst_renderer_native_program_pilot_report_v1"
    )
    torch.save(
        {
            "schema": checkpoint_schema,
            "seed": args.seed,
            "state": model.state_dict(),
            "trainable_names": parameters["trainable_names"],
            "frozen_parent_digest": final_parent_digest,
            "development_accesses": 0,
            "confirmation_accesses": 0,
            "score_eligible": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": report_schema,
        "decision": (
            (
                "retain_renderer_native_joint_as_conventional_compiler_control"
                if args.train_shared_orbit
                else "retain_renderer_native_program_as_conventional_compiler_control"
            )
            if all(gates.values())
            else (
                "reject_or_revise_renderer_native_joint_control"
                if args.train_shared_orbit
                else "reject_or_revise_renderer_native_program_control"
            )
        ),
        "seed": args.seed,
        "source": {
            "train_sha256": CONSUMED_TRAIN_SHA256,
            "orbit_checkpoint_sha256": ORBIT_CHECKPOINT_SHA256,
            "fit_source_sha256": _source_digest(fit_source),
            "heldout_source_sha256": _source_digest(heldout_source),
        },
        "training": {
            "epochs": args.epochs,
            "updates": update,
            "family_batch_size": args.family_batch_size,
            "lr": args.lr,
            "warmup": args.warmup,
            "consistency_weight": args.consistency_weight,
            "train_shared_orbit": args.train_shared_orbit,
            "elapsed_seconds": time.time() - started,
        },
        "parameters": parameters,
        "frozen_parent_digest": parent_digest,
        "history": history,
        "fit": fit,
        "heldout": heldout,
        "gates": gates,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": "Consumed training rows only; favorable conventional "
        "compiler control, not a reasoning or novelty result.",
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "heldout_min_packet": _minimum_rate(heldout, "packet"),
                "heldout_min_kind": _minimum_rate(heldout, "kind"),
                "heldout_min_event_pointer": _minimum_rate(
                    heldout,
                    "event_pointer",
                ),
                "parameters": parameters,
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
