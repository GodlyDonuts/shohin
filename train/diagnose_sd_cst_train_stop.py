#!/usr/bin/env python3
"""Training-only SD-CST diagnostic for unconstrained STOP cardinality.

This wrapper monkey-patches only the post-fit training summary that crashed in
v1.1. It cannot evaluate development or confirmation, and its checkpoint is
diagnostic-only. The frozen training implementation and board receipt remain
unchanged and source-verified by ``train_main``.
"""

from __future__ import annotations

from collections import Counter
import sys
from typing import Sequence

import torch

import train_sd_cst
from train_sd_cst import EncodedRow, STOP_KIND, _autocast, label_batch, pad_sources
from sd_cst import SDCSTSystem


@torch.no_grad()
def raw_compiler_train_metrics(
    system: SDCSTSystem, rows: Sequence[EncodedRow], batch_size: int,
) -> dict[str, object]:
    """Summarize raw independent argmaxes without constructing a valid tape."""
    system.eval()
    device = next(system.motor.parameters()).device
    names = ("initial", "kind", "identity", "amount", "query", "whole_tape")
    counts = {name: 0 for name in names}
    stop_histogram: Counter[int] = Counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        program_ids, program_mask = pad_sources(batch, "program_ids", device)
        query_ids, query_mask = pad_sources(batch, "query_ids", device)
        labels = label_batch(batch, device)
        with _autocast(device):
            program = system.compile_program(program_ids, program_mask)
            query = system.compile_late_query(query_ids, query_mask)
        initial_ids = program.initial_state.argmax(-1).cpu()
        kind_ids = program.event_kind.argmax(-1).cpu()
        identity_ids = program.event_identity.argmax(-1).cpu()
        amount_ids = program.amount.argmax(-1).cpu()
        query_ids = query.logits.argmax(-1).cpu()
        stop_histogram.update(
            int(value) for value in kind_ids.eq(STOP_KIND).sum(dim=1).tolist()
        )
        active = labels["event_kind_targets"].cpu().ne(STOP_KIND)
        exact = {
            "initial": initial_ids.eq(labels["initial_state_targets"].cpu()),
            "kind": kind_ids.eq(labels["event_kind_targets"].cpu()).all(dim=1),
            "identity": (
                identity_ids.eq(labels["event_identity_targets"].cpu()) | ~active
            ).all(dim=1),
            "amount": (
                amount_ids.eq(labels["amount_targets"].cpu()) | ~active
            ).all(dim=1),
            "query": query_ids.eq(labels["query_targets"].cpu()),
        }
        exact["whole_tape"] = (
            exact["initial"] & exact["kind"] & exact["identity"] & exact["amount"]
        )
        for name, value in exact.items():
            counts[name] += int(value.sum())
    total = len(rows)
    return {
        "diagnostic_only": True,
        "rows": total,
        "exact": counts,
        "rates": {name: value / total for name, value in counts.items()},
        "unconstrained_stop_count_histogram": {
            str(key): value for key, value in sorted(stop_histogram.items())
        },
        "exactly_one_stop": stop_histogram[1],
    }


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        raise SystemExit("diagnostic wrapper accepts only the train subcommand")
    train_sd_cst.compiler_train_metrics = raw_compiler_train_metrics
    train_sd_cst.main()


if __name__ == "__main__":
    main()
