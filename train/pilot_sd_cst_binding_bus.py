#!/usr/bin/env python3
"""Training-only pilot for the SD-CST content-addressable binding bus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MAX_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    PilotRow,
    _line_ranges,
    batches,
    byte_batch,
    cosine_scale,
    labels,
    parse_train_row,
    sha256_file,
)
from sd_cst import STOP_KIND
from sd_cst_binding_bus import BindingBusCompiler


@dataclass(frozen=True, slots=True)
class BindingPilotRow:
    row_id: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    pointer_ranges: tuple[tuple[int, int], ...]
    binding_ranges: tuple[tuple[int, int], ...]
    initial_entity_ranges: tuple[tuple[int, int], ...]
    event_entity_ranges: tuple[tuple[int, int], ...]
    initial_state: int
    event_kind: tuple[int, ...]
    event_identity: tuple[int, ...]
    amount: tuple[int, ...]
    query_position: int


def _find_within(
    source: bytes, needle: bytes, start: int, end: int,
) -> tuple[int, int]:
    found = source.find(needle, start, end)
    if found < 0:
        raise ValueError(f"cannot find entity bytes inside declared source region: {needle!r}")
    return found, found + len(needle)


def parse_binding_row(row: dict[str, object]) -> BindingPilotRow:
    base: PilotRow = parse_train_row(row)
    targets = row["compiler_targets"]
    if not isinstance(targets, dict):
        raise TypeError("binding row lacks compiler targets")
    bindings_raw = targets["entity_bindings"]
    bindings = sorted(bindings_raw, key=lambda item: item["entity_role"])
    names = [str(item["entity"]).encode("utf-8") for item in bindings]
    source = bytes(base.program_bytes)
    lines = _line_ranges(str(row["program_text"]))
    binding_line_start, binding_line_end = lines[0]
    marker = source.find(b"initial ", binding_line_start, binding_line_end)
    if marker < 0:
        raise ValueError("binding line lacks initial-order marker")
    declaration_ranges = tuple(
        _find_within(source, name, binding_line_start, marker) for name in names
    )
    initial_cursor = marker + len(b"initial ")
    initial_ranges = []
    for role in targets["initial_order_roles"]:
        found = _find_within(
            source, names[int(role)], initial_cursor, binding_line_end,
        )
        initial_ranges.append(found)
        initial_cursor = found[1]
    event_ranges = []
    for slot, (start, end) in enumerate(base.pointer_ranges[1:]):
        if base.event_kind[slot] == STOP_KIND:
            event_ranges.append((0, 0))
        else:
            event_ranges.append(_find_within(
                source, names[base.event_identity[slot]], start, end,
            ))
    return BindingPilotRow(
        row_id=base.row_id,
        program_bytes=base.program_bytes,
        query_bytes=base.query_bytes,
        pointer_ranges=base.pointer_ranges,
        binding_ranges=declaration_ranges,
        initial_entity_ranges=tuple(initial_ranges),
        event_entity_ranges=tuple(event_ranges),
        initial_state=base.initial_state,
        event_kind=base.event_kind,
        event_identity=base.event_identity,
        amount=base.amount,
        query_position=base.query_position,
    )


def load_rows(data_dir: Path) -> tuple[list[BindingPilotRow], dict[str, object]]:
    report_path = data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if int(report.get("confirmation_accesses", -1)) != 0:
        raise ValueError("board receipt records confirmation access")
    train_path = data_dir / "train.jsonl"
    declared = report["files"]["train.jsonl"]["sha256"]
    if sha256_file(train_path) != declared:
        raise ValueError("training split hash differs from board receipt")
    rows = [
        parse_binding_row(json.loads(line))
        for line in train_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != 48_000:
        raise ValueError("binding pilot requires exactly 48,000 consumed training rows")
    return rows, {
        "report_sha256": sha256_file(report_path),
        "train_sha256": declared,
        "board_source_commit": report["source_commit"],
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }


def partition(
    rows: Sequence[BindingPilotRow], fit_rows: int,
) -> tuple[list[BindingPilotRow], list[BindingPilotRow]]:
    ordered = sorted(
        rows, key=lambda row: hashlib.sha256(row.row_id.encode()).digest(),
    )
    return ordered[:fit_rows], ordered[fit_rows:]


def span_mask(
    rows: Sequence[BindingPilotRow], field: str, slots: int,
    width: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros((len(rows), slots, width), dtype=torch.bool, device=device)
    active = torch.zeros((len(rows), slots), dtype=torch.bool, device=device)
    for row_index, row in enumerate(rows):
        ranges = getattr(row, field)
        if len(ranges) != slots:
            raise ValueError(f"{field} has wrong slot count")
        for slot, (start, end) in enumerate(ranges):
            if end > start:
                mask[row_index, slot, start:end] = True
                active[row_index, slot] = True
    return mask, active


def uniform_span_loss(
    pointer_logits: torch.Tensor,
    target_mask: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if not bool(active.any()):
        raise ValueError("span loss requires an active target")
    target = target_mask.float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    per_slot = -(pointer_logits.log_softmax(-1) * target).sum(-1)
    return per_slot[active].mean()


def loss_batch(
    model: BindingBusCompiler,
    rows: Sequence[BindingPilotRow],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    target = labels(rows, device)
    output = model.compile_program(program_ids, program_valid)
    query = model.compile_query(query_ids, query_valid)
    tape = output.tape
    initial_loss = F.cross_entropy(tape.initial_state, target["initial"])
    kind_loss = F.cross_entropy(
        tape.event_kind.reshape(-1, 3), target["kind"].reshape(-1),
        weight=torch.tensor([1.0, 1.0, 4.0], device=device),
    )
    active_events = target["kind"].ne(STOP_KIND).reshape(-1)
    identity_loss = F.cross_entropy(
        tape.event_identity.reshape(-1, 3)[active_events],
        target["identity"].reshape(-1)[active_events],
    )
    amount_loss = F.cross_entropy(
        tape.amount.reshape(-1, 2)[active_events],
        target["amount"].reshape(-1)[active_events],
    )
    query_loss = F.cross_entropy(query.logits, target["query"])

    line_mask, line_active = span_mask(
        rows, "pointer_ranges", 9, program_ids.shape[1], device,
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
    address_losses = {
        "line_address": uniform_span_loss(
            output.line_pointer_logits, line_mask, line_active,
        ),
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
    address_total = sum(address_losses.values())
    total = (
        initial_loss + kind_loss + identity_loss + amount_loss + query_loss
        + address_total
    )
    pieces = {
        "initial": initial_loss,
        "kind": kind_loss,
        "identity": identity_loss,
        "amount": amount_loss,
        "query": query_loss,
        **address_losses,
        "total": total,
    }
    return total, {name: float(value.detach()) for name, value in pieces.items()}


@torch.no_grad()
def evaluate(
    model: BindingBusCompiler,
    rows: Sequence[BindingPilotRow],
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    counts: Counter[str] = Counter()
    raw_stop_histogram: Counter[int] = Counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        target = labels(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.compile_program(program_ids, program_valid)
            query = model.compile_query(query_ids, query_valid)
        tape = output.tape
        initial = tape.initial_state.argmax(-1)
        kind = tape.event_kind.argmax(-1)
        identity = tape.event_identity.argmax(-1)
        amount = tape.amount.argmax(-1)
        query_prediction = query.logits.argmax(-1)
        raw_stop_histogram.update(
            int(value) for value in kind.eq(STOP_KIND).sum(-1).tolist()
        )
        active = target["kind"].ne(STOP_KIND)
        exact = {
            "initial": initial.eq(target["initial"]),
            "kind": kind.eq(target["kind"]).all(-1),
            "identity": (identity.eq(target["identity"]) | ~active).all(-1),
            "amount": (amount.eq(target["amount"]) | ~active).all(-1),
            "query": query_prediction.eq(target["query"]),
        }
        exact["whole_tape"] = (
            exact["initial"] & exact["kind"] & exact["identity"] & exact["amount"]
        )
        pointer_specs = (
            ("line_pointer", output.line_pointer_logits, "pointer_ranges", 9),
            ("binding_pointer", output.binding_pointer_logits, "binding_ranges", 3),
            (
                "initial_entity_pointer", output.initial_entity_pointer_logits,
                "initial_entity_ranges", 3,
            ),
            (
                "event_entity_pointer", output.event_entity_pointer_logits,
                "event_entity_ranges", 8,
            ),
        )
        for name, logits, field, slots in pointer_specs:
            mask, pointer_active = span_mask(
                batch, field, slots, program_ids.shape[1], device,
            )
            inside = mask.gather(-1, logits.argmax(-1)[..., None]).squeeze(-1)
            exact[name] = (inside | ~pointer_active).all(-1)
        counts["rows"] += len(batch)
        for name, values in exact.items():
            counts[name] += int(values.sum())
    total = counts.pop("rows")
    return {
        "rows": total,
        "exact": dict(sorted(counts.items())),
        "rates": {name: value / total for name, value in sorted(counts.items())},
        "raw_stop_count_histogram": dict(sorted(raw_stop_histogram.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
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
        raise SystemExit(f"refusing existing binding-pilot output: {args.out_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("binding-bus pilot requires CUDA")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rows, board = load_rows(args.data_dir)
    fit_rows, heldout_rows = partition(rows, args.fit_rows)
    model = BindingBusCompiler().to(device)
    compiler_parameters = model.parameter_count()
    complete_parameters = (
        BASE_PARAMETERS + compiler_parameters + MOTOR_PARAMETERS + READER_PARAMETERS
    )
    if complete_parameters >= MAX_PARAMETERS:
        raise SystemExit(f"binding-bus system exceeds cap: {complete_parameters}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
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
                loss, pieces = loss_batch(model, batch, device)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite binding-bus gradient")
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
    fit_metrics = evaluate(model, fit_rows, args.eval_batch_size, device)
    heldout_metrics = evaluate(model, heldout_rows, args.eval_batch_size, device)
    rates = heldout_metrics["rates"]
    raw_one_stop = heldout_metrics["raw_stop_count_histogram"] == {1: len(heldout_rows)}
    gates = {
        "line_pointer_at_least_90pct": rates["line_pointer"] >= 0.90,
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
        "complete_system_below_150m": complete_parameters < MAX_PARAMETERS,
        "scored_access_zero": True,
    }
    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save({
        "schema": "r12_sd_cst_binding_bus_training_pilot_v1",
        "state": model.state_dict(),
        "seed": args.seed,
        "score_eligible": False,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }, checkpoint_path)
    report = {
        "schema": "r12_sd_cst_binding_bus_training_pilot_report_v1",
        "decision": (
            "advance_binding_bus" if all(gates.values()) else "reject_or_revise_binding_bus"
        ),
        "seed": args.seed,
        "board": board,
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
            "kind_class_weights": [1.0, 1.0, 4.0],
            "span_target": "uniform over exact training-only entity/source span",
        },
        "parameters": {
            "base": BASE_PARAMETERS,
            "compiler": compiler_parameters,
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
            "Consumed training split only; binding architecture pilot, not a reasoning score."
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
