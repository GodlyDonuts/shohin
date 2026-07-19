#!/usr/bin/env python3
"""Build atomic S6 training cells and the sole disjoint development board."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

from s6_contextual_affine_law import (
    ADMITTED_MODULI,
    DIAGNOSTIC_MODULUS,
    AffineLaw,
    execute_program,
    infer_affine_law,
    law_split,
    split_laws,
    treatment_input,
)


def _jsonl_bytes(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nonce_name(rng: random.Random, used: set[str]) -> str:
    while True:
        value = "op_" + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(10))
        if value not in used:
            used.add(value)
            return value


def build_atomic_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    control_law_id = 0
    for modulus in ADMITTED_MODULI:
        for law in split_laws(modulus)["train"]:
            for current_location in range(modulus):
                visible = treatment_input(law, current_location)
                rows.append(
                    {
                        "schema": "r12_s6_atomic_law_cell_v1",
                        **visible,
                        "destination": law.destination(current_location),
                        "control_law_id": control_law_id,
                        "supervision": "atomic_destination_only",
                    }
                )
            control_law_id += 1
    return rows


def _build_program_row(
    rng: random.Random,
    modulus: int,
    depth: int,
    row_index: int,
    board_role: str,
) -> dict[str, object]:
    heldout_laws = split_laws(modulus)["development"]
    pool_size = min(len(heldout_laws), 2 + rng.randrange(3))
    law_pool = rng.sample(list(heldout_laws), pool_size)
    used_names: set[str] = set()
    law_names = {law: _nonce_name(rng, used_names) for law in law_pool}
    law_cards = {
        law_names[law]: {
            "modulus": modulus,
            "card_y0": law.card[0],
            "card_y1": law.card[1],
        }
        for law in law_pool
    }

    initial = list(range(modulus))
    rng.shuffle(initial)
    events: list[dict[str, object]] = []
    exact_events: list[tuple[int, AffineLaw]] = []
    required = list(law_pool[:2])
    for step in range(depth):
        law = required[step] if step < len(required) else rng.choice(law_pool)
        identity = rng.randrange(modulus)
        events.append({"operation": law_names[law], "identity": identity})
        exact_events.append((identity, law))
    final_state = execute_program(initial, exact_events)
    query_position = rng.randrange(modulus)
    return {
        "schema": "r12_s6_contextual_affine_program_v1",
        "board_role": board_role,
        "row_id": f"{board_role}_{row_index:06d}",
        "modulus": modulus,
        "depth": depth,
        "initial_state": initial,
        "law_cards": law_cards,
        "events": events,
        "query_position": query_position,
        "final_state": list(final_state),
        "answer": final_state[query_position],
        "distinct_laws": len({event["operation"] for event in events}),
    }


def build_program_rows(
    seed: int, primary_rows: int, diagnostic_rows: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    primary: list[dict[str, object]] = []
    cells = [
        (modulus, depth)
        for modulus in ADMITTED_MODULI
        for depth in range(3, 9)
    ]
    rng.shuffle(cells)
    for index in range(primary_rows):
        modulus, depth = cells[index % len(cells)]
        primary.append(
            _build_program_row(rng, modulus, depth, index, "development")
        )

    diagnostic: list[dict[str, object]] = []
    diagnostic_depths = list(range(3, 9))
    rng.shuffle(diagnostic_depths)
    for index in range(diagnostic_rows):
        diagnostic.append(
            _build_program_row(
                rng,
                DIAGNOSTIC_MODULUS,
                diagnostic_depths[index % len(diagnostic_depths)],
                index,
                "scale_diagnostic",
            )
        )
    return primary, diagnostic


def audit_rows(
    train_rows: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
) -> dict[str, object]:
    allowed_treatment = {"modulus", "card_y0", "card_y1", "current_location"}
    training_laws = {
        (
            int(row["modulus"]),
            int(row["card_y0"]),
            int(row["card_y1"]),
        )
        for row in train_rows
    }
    development_laws: set[tuple[int, int, int]] = set()
    for row in primary_rows:
        for card in row["law_cards"].values():
            law = infer_affine_law(
                int(card["modulus"]),
                int(card["card_y0"]),
                int(card["card_y1"]),
            )
            if law_split(law) != "development":
                raise ValueError("S6 primary board contains a non-development law")
            development_laws.add((law.modulus, *law.card))
    if training_laws & development_laws:
        raise ValueError("S6 training/development law overlap")

    diagnostic_laws: set[tuple[int, int, int]] = set()
    for row in diagnostic_rows:
        for card in row["law_cards"].values():
            law = infer_affine_law(
                int(card["modulus"]),
                int(card["card_y0"]),
                int(card["card_y1"]),
            )
            if law.modulus != DIAGNOSTIC_MODULUS:
                raise ValueError("S6 scale diagnostic has wrong modulus")
            diagnostic_laws.add((law.modulus, *law.card))

    treatment_fields = {
        key for key in train_rows[0] if key in allowed_treatment
    }
    if treatment_fields != allowed_treatment:
        raise ValueError("S6 atomic treatment schema mismatch")
    if any(row["supervision"] != "atomic_destination_only" for row in train_rows):
        raise ValueError("S6 training contains forbidden supervision")

    cell_counts: dict[str, int] = {}
    for row in primary_rows:
        key = f"m{row['modulus']}_d{row['depth']}"
        cell_counts[key] = cell_counts.get(key, 0) + 1
    if max(cell_counts.values()) - min(cell_counts.values()) > 1:
        raise ValueError("S6 development modulus/depth cells are imbalanced")

    return {
        "atomic_training_rows": len(train_rows),
        "primary_development_rows": len(primary_rows),
        "scale_diagnostic_rows": len(diagnostic_rows),
        "training_law_count": len(training_laws),
        "development_law_count": len(development_laws),
        "diagnostic_law_count": len(diagnostic_laws),
        "train_development_law_overlap": len(training_laws & development_laws),
        "treatment_input_fields": sorted(treatment_fields),
        "development_cell_counts": dict(sorted(cell_counts.items())),
        "minimum_distinct_laws_per_program": min(
            int(row["distinct_laws"]) for row in primary_rows
        ),
        "confirmation_program_rows": 0,
        "confirmation_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--primary-rows", type=int, default=2048)
    parser.add_argument("--diagnostic-rows", type=int, default=512)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing S6 board: {args.out_dir}")
    if args.primary_rows < 18 or args.diagnostic_rows < 6:
        raise SystemExit("S6 board is too small for frozen balance cells")

    train_rows = build_atomic_training_rows()
    primary_rows, diagnostic_rows = build_program_rows(
        args.seed, args.primary_rows, args.diagnostic_rows
    )
    audit = audit_rows(train_rows, primary_rows, diagnostic_rows)
    train_payload = _jsonl_bytes(train_rows)
    primary_payload = _jsonl_bytes(primary_rows)
    diagnostic_payload = _jsonl_bytes(diagnostic_rows)
    report = {
        "schema": "r12_s6_contextual_affine_development_board_report_v1",
        "decision": "admit_s6_development_board",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "audit": audit,
        "files": {
            "atomic_train.jsonl": {
                "sha256": _sha256(train_payload),
                "bytes": len(train_payload),
            },
            "development.jsonl": {
                "sha256": _sha256(primary_payload),
                "bytes": len(primary_payload),
            },
            "scale_diagnostic.jsonl": {
                "sha256": _sha256(diagnostic_payload),
                "bytes": len(diagnostic_payload),
            },
        },
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "atomic_train.jsonl").write_bytes(train_payload)
    (args.out_dir / "development.jsonl").write_bytes(primary_payload)
    (args.out_dir / "scale_diagnostic.jsonl").write_bytes(diagnostic_payload)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"decision": report["decision"], "audit": audit}, sort_keys=True))


if __name__ == "__main__":
    main()

