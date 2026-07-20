#!/usr/bin/env python3
"""Training-only frozen-parent pilot for hierarchical SD-CST binding."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from pilot_sd_cst_binding_bus import (
    BindingPilotRow,
    evaluate,
    load_rows,
    partition,
    span_mask,
    uniform_span_loss,
)
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MAX_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    batches,
    byte_batch,
    cosine_scale,
    labels,
    sha256_file,
)
from sd_cst import STOP_KIND
from sd_cst_binding_bus import HierarchicalBindingBusCompiler


PARENT_SHA256 = "e5f87a1d5b22d24250a6aac6fb7c70b4a77dbdf01bd5f5c509020a3584dfa6f9"
TRAINABLE_NAMES = frozenset({
    "binding_queries",
    "initial_entity_queries",
    "event_entity_queries",
    "bigram_embedding.weight",
    "fingerprint_projection.weight",
    "logit_scale",
})


def load_parent_state(
    model: HierarchicalBindingBusCompiler,
    parent_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    result = model.load_state_dict(parent_state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected parent state keys: {result.unexpected_keys}")
    expected_missing = set(model.state_dict()) - set(parent_state)
    if set(result.missing_keys) != expected_missing:
        raise ValueError(
            f"parent missing-key mismatch: {result.missing_keys} != {sorted(expected_missing)}"
        )
    return tuple(sorted(result.missing_keys))


def freeze_parent(model: HierarchicalBindingBusCompiler) -> tuple[str, ...]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE_NAMES)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != TRAINABLE_NAMES:
        raise ValueError(f"trainable parameter mismatch: {sorted(actual)}")
    return tuple(sorted(actual))


def frozen_parameter_digest(model: HierarchicalBindingBusCompiler) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def binding_loss(
    model: HierarchicalBindingBusCompiler,
    rows: Sequence[BindingPilotRow],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    target = labels(rows, device)
    output = model.compile_program(program_ids, program_valid)
    tape = output.tape
    initial_loss = F.cross_entropy(tape.initial_state, target["initial"])
    active_events = target["kind"].ne(STOP_KIND).reshape(-1)
    identity_loss = F.cross_entropy(
        tape.event_identity.reshape(-1, 3)[active_events],
        target["identity"].reshape(-1)[active_events],
    )
    binding_mask, binding_active = span_mask(
        rows, "binding_ranges", 3, program_ids.shape[1], device,
    )
    initial_mask, initial_active = span_mask(
        rows, "initial_entity_ranges", 3, program_ids.shape[1], device,
    )
    event_mask, event_active = span_mask(
        rows, "event_entity_ranges", 8, program_ids.shape[1], device,
    )
    pieces = {
        "initial": initial_loss,
        "identity": identity_loss,
        "binding_address": uniform_span_loss(
            output.binding_pointer_logits, binding_mask, binding_active,
        ),
        "initial_entity_address": uniform_span_loss(
            output.initial_entity_pointer_logits, initial_mask, initial_active,
        ),
        "event_entity_address": uniform_span_loss(
            output.event_entity_pointer_logits, event_mask, event_active,
        ),
    }
    total = sum(pieces.values())
    pieces["total"] = total
    return total, {name: float(value.detach()) for name, value in pieces.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fit-rows", type=int, default=40_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing hierarchical output: {args.out_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("hierarchical binding pilot requires CUDA")
    if sha256_file(args.parent_checkpoint) != PARENT_SHA256:
        raise SystemExit("parent byte compiler hash mismatch")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows, board = load_rows(args.data_dir)
    fit_rows, heldout_rows = partition(rows, args.fit_rows)

    payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "r12_sd_cst_byte_addressed_training_pilot_v1":
        raise SystemExit("parent byte compiler schema mismatch")
    model = HierarchicalBindingBusCompiler()
    missing = load_parent_state(model, payload["state"])
    trainable_names = freeze_parent(model)
    model.to(device)
    compiler_parameters = model.parameter_count()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    complete_parameters = (
        BASE_PARAMETERS + compiler_parameters + MOTOR_PARAMETERS + READER_PARAMETERS
    )
    if complete_parameters >= MAX_PARAMETERS:
        raise SystemExit(f"hierarchical system exceeds pilot cap: {complete_parameters}")

    prefit = evaluate(model, heldout_rows, args.eval_batch_size, device)
    prefit_rates = prefit["rates"]
    inherited_prefit = {
        "line_pointer_exact": prefit_rates["line_pointer"] == 1.0,
        "kind_exact": prefit_rates["kind"] == 1.0,
        "amount_exact": prefit_rates["amount"] == 1.0,
        "query_exact": prefit_rates["query"] == 1.0,
        "one_raw_stop": prefit["raw_stop_count_histogram"] == {1: len(heldout_rows)},
    }
    if not all(inherited_prefit.values()):
        raise SystemExit(f"frozen parent prefit gate failed: {inherited_prefit}")
    frozen_before = frozen_parameter_digest(model)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    total_updates = args.epochs * math.ceil(len(fit_rows) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_scale(step, total_updates, args.warmup),
    )
    history = []
    update = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        sums: Counter[str] = Counter()
        seen = 0
        for indices in batches(len(fit_rows), args.batch_size, args.seed, epoch):
            batch = [fit_rows[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pieces = binding_loss(model, batch, device)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite hierarchical binding gradient")
            optimizer.step()
            scheduler.step()
            update += 1
            seen += len(batch)
            for name, value in pieces.items():
                sums[name] += value * len(batch)
        heldout = evaluate(model, heldout_rows, args.eval_batch_size, device)
        record = {
            "epoch": epoch + 1,
            "updates": update,
            "fit_losses": {name: value / seen for name, value in sorted(sums.items())},
            "heldout": heldout,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    frozen_after = frozen_parameter_digest(model)
    fit_metrics = evaluate(model, fit_rows, args.eval_batch_size, device)
    heldout_metrics = evaluate(model, heldout_rows, args.eval_batch_size, device)
    rates = heldout_metrics["rates"]
    raw_one_stop = heldout_metrics["raw_stop_count_histogram"] == {
        1: len(heldout_rows),
    }
    gates = {
        "prefit_parent_exact": all(inherited_prefit.values()),
        "binding_pointer_at_least_90pct": rates["binding_pointer"] >= 0.90,
        "initial_entity_pointer_at_least_90pct": (
            rates["initial_entity_pointer"] >= 0.90
        ),
        "event_entity_pointer_at_least_90pct": (
            rates["event_entity_pointer"] >= 0.90
        ),
        "initial_at_least_80pct": rates["initial"] >= 0.80,
        "kind_at_least_90pct": rates["kind"] >= 0.90,
        "identity_at_least_80pct": rates["identity"] >= 0.80,
        "amount_at_least_90pct": rates["amount"] >= 0.90,
        "query_at_least_98pct": rates["query"] >= 0.98,
        "whole_tape_at_least_60pct": rates["whole_tape"] >= 0.60,
        "exactly_one_raw_stop_every_row": raw_one_stop,
        "frozen_parent_unchanged": frozen_before == frozen_after,
        "complete_system_below_150m": complete_parameters < MAX_PARAMETERS,
        "scored_access_zero": True,
    }

    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save({
        "schema": "r12_sd_cst_hierarchical_binding_training_pilot_v1",
        "state": model.state_dict(),
        "seed": args.seed,
        "parent_sha256": PARENT_SHA256,
        "score_eligible": False,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }, checkpoint_path)
    report = {
        "schema": "r12_sd_cst_hierarchical_binding_training_pilot_report_v1",
        "decision": (
            "advance_hierarchical_binding"
            if all(gates.values()) else "reject_or_revise_hierarchical_binding"
        ),
        "seed": args.seed,
        "board": board,
        "parent": {
            "checkpoint_sha256": PARENT_SHA256,
            "schema": payload["schema"],
            "prefit": prefit,
            "prefit_gates": inherited_prefit,
            "frozen_digest_before": frozen_before,
            "frozen_digest_after": frozen_after,
        },
        "partition": {
            "method": "sha256(row_id) ordering",
            "fit_rows": len(fit_rows),
            "heldout_rows": len(heldout_rows),
        },
        "training": {
            "epochs": args.epochs,
            "updates": update,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup": args.warmup,
            "elapsed_seconds": time.time() - started,
            "trainable_names": trainable_names,
            "parent_missing_keys": missing,
        },
        "parameters": {
            "base": BASE_PARAMETERS,
            "compiler": compiler_parameters,
            "trainable": trainable_parameters,
            "motor": MOTOR_PARAMETERS,
            "reader": READER_PARAMETERS,
            "complete_system": complete_parameters,
            "headroom": MAX_PARAMETERS - complete_parameters,
        },
        "history": history,
        "fit": fit_metrics,
        "heldout": heldout_metrics,
        "gates": gates,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "score_eligible": False,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "claim_boundary": (
            "Consumed training split only; frozen-parent binding pilot, not a score."
        ),
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gates": gates,
        "heldout": heldout_metrics,
        "parameters": report["parameters"],
        "report_sha256": sha256_file(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
