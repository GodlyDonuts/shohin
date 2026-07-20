#!/usr/bin/env python3
"""Probe a diagnostic SD-CST checkpoint on its training split only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_sd_cst import (
    STOP_KIND,
    TRAIN_SPLIT,
    _autocast,
    _load_adapter_state,
    label_batch,
    load_rows,
    load_system,
    pad_sources,
    sha256_file,
)


@torch.no_grad()
def probe(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("development_accesses") != 0:
        raise SystemExit("diagnostic checkpoint records development access")
    if checkpoint.get("confirmation_accesses") != 0:
        raise SystemExit("diagnostic checkpoint records confirmation access")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system, _ = load_system(args.base, device, checkpoint["architecture"])
    _load_adapter_state(system, checkpoint["state"])
    system.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows, _ = load_rows(
        args.data_dir,
        tokenizer,
        system.base_model.cfg.seq_len,
        TRAIN_SPLIT,
        source_commit=args.source_commit,
    )
    rows = rows[: args.max_rows]
    slot_kind_correct = torch.zeros(8, dtype=torch.long)
    slot_kind_total = torch.zeros(8, dtype=torch.long)
    slot_identity_correct = torch.zeros(8, dtype=torch.long)
    slot_identity_total = torch.zeros(8, dtype=torch.long)
    slot_amount_correct = torch.zeros(8, dtype=torch.long)
    slot_amount_total = torch.zeros(8, dtype=torch.long)
    raw_stop_counts: Counter[int] = Counter()
    raw_kind_predictions: Counter[int] = Counter()
    gold_kind_labels: Counter[int] = Counter()
    row_counts: Counter[str] = Counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        program_ids, program_mask = pad_sources(batch, "program_ids", device)
        labels = label_batch(batch, device)
        with _autocast(device):
            tape = system.compile_program(program_ids, program_mask)
        initial = tape.initial_state.argmax(-1).cpu()
        raw_kind = tape.event_kind.argmax(-1).cpu()
        identity = tape.event_identity.argmax(-1).cpu()
        amount = tape.amount.argmax(-1).cpu()
        target_initial = labels["initial_state_targets"].cpu()
        target_kind = labels["event_kind_targets"].cpu()
        target_identity = labels["event_identity_targets"].cpu()
        target_amount = labels["amount_targets"].cpu()
        active = target_kind.ne(STOP_KIND)

        # Exact MAP under the public grammar: one STOP and seven non-STOP kinds.
        non_stop_values, non_stop_kind = tape.event_kind[..., :STOP_KIND].max(-1)
        stop_delta = tape.event_kind[..., STOP_KIND] - non_stop_values
        stop_slot = stop_delta.argmax(-1).cpu()
        constrained_kind = non_stop_kind.cpu()
        constrained_kind.scatter_(1, stop_slot[:, None], STOP_KIND)
        gold_stop_slot = target_kind.eq(STOP_KIND).long().argmax(-1)

        raw_stop_counts.update(
            int(value) for value in raw_kind.eq(STOP_KIND).sum(-1).tolist()
        )
        raw_kind_predictions.update(int(value) for value in raw_kind.flatten().tolist())
        gold_kind_labels.update(int(value) for value in target_kind.flatten().tolist())
        slot_kind_correct += raw_kind.eq(target_kind).sum(0)
        slot_kind_total += len(batch)
        slot_identity_correct += (identity.eq(target_identity) & active).sum(0)
        slot_identity_total += active.sum(0)
        slot_amount_correct += (amount.eq(target_amount) & active).sum(0)
        slot_amount_total += active.sum(0)
        row_counts["rows"] += len(batch)
        row_counts["initial_exact"] += int(initial.eq(target_initial).sum())
        row_counts["raw_kind_exact"] += int(raw_kind.eq(target_kind).all(-1).sum())
        row_counts["constrained_kind_exact"] += int(
            constrained_kind.eq(target_kind).all(-1).sum()
        )
        row_counts["constrained_stop_position_exact"] += int(
            stop_slot.eq(gold_stop_slot).sum()
        )
        row_counts["identity_exact"] += int(
            (identity.eq(target_identity) | ~active).all(-1).sum()
        )
        row_counts["amount_exact"] += int(
            (amount.eq(target_amount) | ~active).all(-1).sum()
        )
    total = row_counts["rows"]

    def rates(correct: torch.Tensor, denominator: torch.Tensor) -> list[float]:
        return [
            float(c) / int(n) if int(n) else 0.0
            for c, n in zip(correct, denominator, strict=True)
        ]

    return {
        "schema": "r12_sd_cst_v1_1_training_compiler_probe_v1",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "rows": total,
        "device": str(device),
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "score_eligible": False,
        "row_exact": {
            key: {"correct": value, "rate": value / total}
            for key, value in sorted(row_counts.items()) if key != "rows"
        },
        "per_slot_cell_accuracy": {
            "kind": rates(slot_kind_correct, slot_kind_total),
            "identity": rates(slot_identity_correct, slot_identity_total),
            "amount": rates(slot_amount_correct, slot_amount_total),
        },
        "raw_stop_count_histogram": dict(sorted(raw_stop_counts.items())),
        "raw_kind_prediction_histogram": dict(sorted(raw_kind_predictions.items())),
        "gold_kind_label_histogram": dict(sorted(gold_kind_labels.items())),
        "constrained_decoder": (
            "global exact MAP under exactly-one-STOP grammar using only kind logits"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=48_000)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing probe output: {args.out}")
    report = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "rows": report["rows"],
        "row_exact": report["row_exact"],
        "sha256": sha256_file(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
