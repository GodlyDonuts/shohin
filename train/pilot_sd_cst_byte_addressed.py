#!/usr/bin/env python3
"""Training-only pilot for the SD-CST byte-addressed evidence compiler."""

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
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from sd_cst import EVENT_STEPS, STOP_KIND
from sd_cst_byte_addressed import BYTE_PAD, PROGRAM_SLOTS, ByteAddressedCompiler


BASE_PARAMETERS = 125_081_664
MOTOR_PARAMETERS = 19_206
READER_PARAMETERS = 835
MAX_PARAMETERS = 150_000_000


@dataclass(frozen=True, slots=True)
class PilotRow:
    row_id: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    pointer_ranges: tuple[tuple[int, int], ...]
    initial_state: int
    event_kind: tuple[int, ...]
    event_identity: tuple[int, ...]
    amount: tuple[int, ...]
    query_position: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_ranges(text: str) -> tuple[tuple[int, int], ...]:
    lines = text.splitlines(keepends=True)
    if len(lines) != 9:
        raise ValueError("byte-addressed pilot requires one binding and eight event lines")
    ranges = []
    offset = 0
    for line in lines:
        width = len(line.encode("utf-8"))
        ranges.append((offset, offset + width))
        offset += width
    if offset != len(text.encode("utf-8")):
        raise ValueError("line byte ranges do not cover the source")
    return tuple(ranges)


def parse_train_row(row: dict[str, object]) -> PilotRow:
    if row.get("split") != "sd_cst_train" or "oracle" in row:
        raise ValueError("pilot accepts outcome-free SD-CST training rows only")
    targets = row["compiler_targets"]
    query = row["late_query_target"]
    if not isinstance(targets, dict) or not isinstance(query, dict):
        raise TypeError("pilot row lacks compiler targets")
    slots = sorted(targets["event_slots"], key=lambda item: item["semantic_ordinal"])
    storage_order = [int(value) for value in targets["storage_order"]]
    if sorted(storage_order) != list(range(1, EVENT_STEPS + 1)):
        raise ValueError("storage order is not a permutation of event ordinals")
    text = str(row["program_text"])
    physical_lines = _line_ranges(text)
    pointer_ranges = [physical_lines[0]]
    for semantic_ordinal in range(1, EVENT_STEPS + 1):
        pointer_ranges.append(
            physical_lines[1 + storage_order.index(semantic_ordinal)]
        )
    program_bytes = tuple(text.encode("utf-8"))
    query_bytes = tuple(str(row["late_query_text"]).encode("utf-8"))
    return PilotRow(
        row_id=str(row["id"]),
        program_bytes=program_bytes,
        query_bytes=query_bytes,
        pointer_ranges=tuple(pointer_ranges),
        initial_state=int(targets["initial_state_id"]),
        event_kind=tuple(int(item["kind_id"]) for item in slots),
        event_identity=tuple(int(item.get("entity_role", 0)) for item in slots),
        amount=tuple(int(item.get("amount_id", 0)) for item in slots),
        query_position=int(query["position"]),
    )


def load_training_rows(data_dir: Path) -> tuple[list[PilotRow], dict[str, object]]:
    report_path = data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if int(report.get("confirmation_accesses", -1)) != 0:
        raise ValueError("board receipt records confirmation access")
    train_path = data_dir / "train.jsonl"
    declared = report["files"]["train.jsonl"]["sha256"]
    if sha256_file(train_path) != declared:
        raise ValueError("training split hash differs from board receipt")
    rows = [
        parse_train_row(json.loads(line))
        for line in train_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != 48_000:
        raise ValueError("pilot requires the 48,000-row consumed training split")
    return rows, {
        "report_sha256": sha256_file(report_path),
        "train_sha256": declared,
        "board_source_commit": report["source_commit"],
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }


def deterministic_partition(
    rows: Sequence[PilotRow], fit_rows: int,
) -> tuple[list[PilotRow], list[PilotRow]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(row.row_id.encode("utf-8")).digest(),
    )
    return ordered[:fit_rows], ordered[fit_rows:]


def batches(count: int, size: int, seed: int, epoch: int) -> Iterable[list[int]]:
    indices = list(range(count))
    random.Random(seed ^ (epoch * 0x9E3779B1)).shuffle(indices)
    for start in range(0, count, size):
        yield indices[start:start + size]


def byte_batch(
    rows: Sequence[PilotRow], field: str, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [getattr(row, field) for row in rows]
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full(
        (len(rows), width), BYTE_PAD, dtype=torch.long, device=device,
    )
    valid = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
        valid[index, :len(sequence)] = True
    return ids, valid


def pointer_mask(
    rows: Sequence[PilotRow], width: int, device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        (len(rows), PROGRAM_SLOTS, width), dtype=torch.bool, device=device,
    )
    for row_index, row in enumerate(rows):
        for slot, (start, end) in enumerate(row.pointer_ranges):
            mask[row_index, slot, start:end] = True
    if not bool(mask.any(-1).all()):
        raise ValueError("every program slot requires nonempty address supervision")
    return mask


def labels(rows: Sequence[PilotRow], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "initial": torch.tensor([row.initial_state for row in rows], device=device),
        "kind": torch.tensor([row.event_kind for row in rows], device=device),
        "identity": torch.tensor([row.event_identity for row in rows], device=device),
        "amount": torch.tensor([row.amount for row in rows], device=device),
        "query": torch.tensor([row.query_position for row in rows], device=device),
    }


def loss_batch(
    model: ByteAddressedCompiler,
    rows: Sequence[PilotRow],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    target = labels(rows, device)
    output = model.compile_program(program_ids, program_valid)
    query = model.compile_query(query_ids, query_valid)
    tape = output.tape
    initial_loss = F.cross_entropy(tape.initial_state, target["initial"])
    kind_weights = torch.tensor([1.0, 1.0, 4.0], device=device)
    kind_loss = F.cross_entropy(
        tape.event_kind.reshape(-1, 3), target["kind"].reshape(-1),
        weight=kind_weights,
    )
    active = target["kind"].ne(STOP_KIND).reshape(-1)
    identity_loss = F.cross_entropy(
        tape.event_identity.reshape(-1, 3)[active],
        target["identity"].reshape(-1)[active],
    )
    amount_loss = F.cross_entropy(
        tape.amount.reshape(-1, 2)[active], target["amount"].reshape(-1)[active],
    )
    query_loss = F.cross_entropy(query.logits, target["query"])
    address_targets = pointer_mask(rows, program_ids.shape[1], device)
    address_log_probs = output.pointer_logits.log_softmax(-1)
    inside = address_log_probs.masked_fill(
        ~address_targets, torch.finfo(address_log_probs.dtype).min,
    )
    address_loss = -inside.logsumexp(-1).mean()
    total = (
        initial_loss + kind_loss + identity_loss + amount_loss + query_loss
        + 2.0 * address_loss
    )
    return total, {
        "initial": float(initial_loss.detach()),
        "kind": float(kind_loss.detach()),
        "identity": float(identity_loss.detach()),
        "amount": float(amount_loss.detach()),
        "query": float(query_loss.detach()),
        "address": float(address_loss.detach()),
        "total": float(total.detach()),
    }


@torch.no_grad()
def evaluate(
    model: ByteAddressedCompiler,
    rows: Sequence[PilotRow],
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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model.compile_program(program_ids, program_valid)
            query = model.compile_query(query_ids, query_valid)
        tape = output.tape
        initial = tape.initial_state.argmax(-1)
        raw_kind = tape.event_kind.argmax(-1)
        identity = tape.event_identity.argmax(-1)
        amount = tape.amount.argmax(-1)
        query_ids = query.logits.argmax(-1)
        raw_stop_histogram.update(
            int(value) for value in raw_kind.eq(STOP_KIND).sum(-1).tolist()
        )
        non_stop_values, non_stop_kind = tape.event_kind[..., :STOP_KIND].max(-1)
        stop_slot = (
            tape.event_kind[..., STOP_KIND] - non_stop_values
        ).argmax(-1)
        constrained_kind = non_stop_kind.clone()
        constrained_kind.scatter_(1, stop_slot[:, None], STOP_KIND)
        active = target["kind"].ne(STOP_KIND)
        exact_initial = initial.eq(target["initial"])
        exact_raw_kind = raw_kind.eq(target["kind"]).all(-1)
        exact_constrained_kind = constrained_kind.eq(target["kind"]).all(-1)
        exact_identity = (identity.eq(target["identity"]) | ~active).all(-1)
        exact_amount = (amount.eq(target["amount"]) | ~active).all(-1)
        exact_query = query_ids.eq(target["query"])
        address_targets = pointer_mask(batch, program_ids.shape[1], device)
        pointer_positions = output.pointer_logits.argmax(-1)
        pointer_exact = address_targets.gather(
            -1, pointer_positions[..., None],
        ).squeeze(-1).all(-1)
        metrics = {
            "initial": exact_initial,
            "raw_kind": exact_raw_kind,
            "constrained_kind": exact_constrained_kind,
            "identity": exact_identity,
            "amount": exact_amount,
            "query": exact_query,
            "pointer_all_slots": pointer_exact,
            "raw_whole_tape": (
                exact_initial & exact_raw_kind & exact_identity & exact_amount
            ),
            "constrained_whole_tape": (
                exact_initial & exact_constrained_kind & exact_identity & exact_amount
            ),
        }
        counts["rows"] += len(batch)
        for name, values in metrics.items():
            counts[name] += int(values.sum())
    total = counts.pop("rows")
    return {
        "rows": total,
        "exact": dict(sorted(counts.items())),
        "rates": {name: value / total for name, value in sorted(counts.items())},
        "raw_stop_count_histogram": dict(sorted(raw_stop_histogram.items())),
    }


def cosine_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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
        raise SystemExit(f"refusing existing pilot output: {args.out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("byte-addressed compiler pilot requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rows, board = load_training_rows(args.data_dir)
    fit_rows, heldout_rows = deterministic_partition(rows, args.fit_rows)
    model = ByteAddressedCompiler().to(device)
    compiler_parameters = model.parameter_count()
    complete_parameters = (
        BASE_PARAMETERS + compiler_parameters + MOTOR_PARAMETERS + READER_PARAMETERS
    )
    if complete_parameters >= MAX_PARAMETERS:
        raise SystemExit(f"pilot system exceeds sub-150M cap: {complete_parameters}")
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
                raise RuntimeError("non-finite pilot gradient")
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
    args.out_dir.mkdir(parents=True)
    checkpoint_path = args.out_dir / "compiler.pt"
    torch.save({
        "schema": "r12_sd_cst_byte_addressed_training_pilot_v1",
        "state": model.state_dict(),
        "architecture": {
            "width": 384,
            "heads": 8,
            "encoder_layers": 6,
            "slot_layers": 2,
            "ff": 1536,
            "slot_ff": 1024,
            "max_bytes": 640,
        },
        "seed": args.seed,
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
    }, checkpoint_path)
    heldout_rates = heldout_metrics["rates"]
    gates = {
        "pointer_all_slots_at_least_90pct": (
            heldout_rates["pointer_all_slots"] >= 0.90
        ),
        "initial_at_least_80pct": heldout_rates["initial"] >= 0.80,
        "constrained_kind_at_least_90pct": (
            heldout_rates["constrained_kind"] >= 0.90
        ),
        "identity_at_least_80pct": heldout_rates["identity"] >= 0.80,
        "amount_at_least_90pct": heldout_rates["amount"] >= 0.90,
        "query_at_least_98pct": heldout_rates["query"] >= 0.98,
        "constrained_whole_tape_at_least_60pct": (
            heldout_rates["constrained_whole_tape"] >= 0.60
        ),
        "complete_system_below_150m": complete_parameters < MAX_PARAMETERS,
        "scored_access_zero": True,
    }
    report = {
        "schema": "r12_sd_cst_byte_addressed_training_pilot_report_v1",
        "decision": (
            "advance_byte_addressed_compiler"
            if all(gates.values()) else "reject_or_revise_byte_addressed_compiler"
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
            "address_loss_weight": 2.0,
            "kind_class_weights": [1.0, 1.0, 4.0],
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
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "claim_boundary": (
            "Consumed training split only; compiler localization pilot, not a reasoning score."
        ),
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "heldout": heldout_metrics,
        "parameters": report["parameters"],
        "report_sha256": sha256_file(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
