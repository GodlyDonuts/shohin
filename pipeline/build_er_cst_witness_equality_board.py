#!/usr/bin/env python3
"""Build the fresh ER-CST v1.1 witness-equality qualification board."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import build_er_cst_fresh_board as base


BOARD_SCHEMA = "r12_er_cst_witness_equality_board_report_v1_1"
ROW_SCHEMA = "r12_er_cst_witness_equality_row_v1_1"
PROTOCOL = "R12-ER-CST-WEB-v1.1"
TRAIN_SPLIT = base.TRAIN_SPLIT
DEVELOPMENT_SPLIT = base.DEVELOPMENT_SPLIT
CONFIRMATION_SPLIT = base.CONFIRMATION_SPLIT
DEFAULT_FAMILIES = base.DEFAULT_FAMILIES


def _witness_range_gate(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        source = str(row["program_text"]).encode("utf-8")
        target = row["compiler_targets"]
        rules = sorted(target["rule_cards"], key=lambda item: int(item["slot"]))
        before_ranges = target.get("witness_before_ranges")
        after_ranges = target.get("witness_after_ranges")
        if (
            not isinstance(before_ranges, list)
            or not isinstance(after_ranges, list)
            or len(before_ranges) != 3
            or len(after_ranges) != 3
        ):
            return False
        for rule, before, after in zip(
            rules, before_ranges, after_ranges, strict=True
        ):
            if len(before) != 3 or len(after) != 3:
                return False
            decoded_before = tuple(
                source[int(start):int(end)].decode("utf-8") for start, end in before
            )
            decoded_after = tuple(
                source[int(start):int(end)].decode("utf-8") for start, end in after
            )
            if decoded_before != tuple(map(str, rule["before"])):
                return False
            if decoded_after != tuple(map(str, rule["after"])):
                return False
    return True


def build_board(
    *, seed: int, families: Mapping[str, int] = DEFAULT_FAMILIES
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    splits = {
        split: base.build_split(
            seed=seed,
            split=split,
            families=int(families[split]),
        )
        for split in (TRAIN_SPLIT, DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
    }
    for rows in splits.values():
        for row in rows:
            row["schema"] = ROW_SCHEMA
            row["protocol"] = PROTOCOL
    report = base.audit_board(splits, expected_families=families)
    witness_ranges_exact = all(_witness_range_gate(rows) for rows in splits.values())
    report["schema"] = BOARD_SCHEMA
    report["protocol"] = PROTOCOL
    report["board_seed"] = int(seed)
    report["gates"]["all_witness_ranges_exact"] = witness_ranges_exact
    report["all_gates_pass"] = all(report["gates"].values())
    return splits, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_commit = base._verify_source_commit(args.source_commit)
    splits, report = build_board(seed=args.seed)
    if report["all_gates_pass"] is not True:
        raise SystemExit("ER-CST witness-equality board audit failed before write")
    value = base.write_board(
        output=args.output,
        source_commit=source_commit,
        splits=splits,
        report=report,
    )
    print(
        base.canonical_json(
            {"all_gates_pass": value["all_gates_pass"], "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
