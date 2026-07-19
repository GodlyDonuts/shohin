#!/usr/bin/env python3
"""Build the frozen S7 learned-Cayley train/development/confirmation board."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from s6_contextual_affine_law import AffineLaw, pop_insert, split_laws
from s7_learned_cayley_law import (
    PRIMARY_MODULI,
    SymbolBinding,
    stride_two_successor,
)


S7_SPLIT_PERSON = "s7-cayley-fresh-v1"
ANCHOR_LAWS = ((1, 0), (1, 1))


def _jsonl_bytes(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _law_order(law: AffineLaw) -> bytes:
    return hashlib.sha256(f"{S7_SPLIT_PERSON}|{law.key}".encode("ascii")).digest()


def s7_law_pools(modulus: int) -> dict[str, tuple[AffineLaw, ...]]:
    old = split_laws(modulus)
    anchors = {
        AffineLaw(modulus, slope, intercept) for slope, intercept in ANCHOR_LAWS
    }
    training = set(old["train"]) | anchors
    fresh = sorted((set(old["confirmation"]) - anchors), key=_law_order)
    if len(fresh) < 4:
        raise ValueError("S7 fresh law pool too small")
    cut = len(fresh) // 2
    development = tuple(sorted(fresh[:cut]))
    confirmation = tuple(sorted(fresh[cut:]))
    if not development or not confirmation:
        raise ValueError("S7 fresh split is empty")
    if training & set(development) or training & set(confirmation):
        raise ValueError("S7 train/fresh overlap")
    return {
        "train": tuple(sorted(training)),
        "development": development,
        "confirmation": confirmation,
        "excluded_closed_s6_development": tuple(
            sorted(set(old["development"]) - anchors)
        ),
    }


def _binding(rng: random.Random, modulus: int) -> SymbolBinding:
    values = list(range(modulus))
    rng.shuffle(values)
    if values == list(range(modulus)):
        values = values[1:] + values[:1]
    return SymbolBinding(modulus, tuple(values))


def _nonce_name(rng: random.Random, used: set[str]) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyz"
    while True:
        value = "law_" + "".join(rng.choice(alphabet) for _ in range(11))
        if value not in used:
            used.add(value)
            return value


def build_training_rows(
    bindings: dict[int, SymbolBinding],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    generator_rows: list[dict[str, object]] = []
    atomic_rows: list[dict[str, object]] = []
    control_law_id = 0
    for modulus in PRIMARY_MODULI:
        binding = bindings[modulus]
        false_successor = stride_two_successor(
            binding.successor, binding.zero_symbol
        )
        for observed in range(modulus):
            generator_rows.append(
                {
                    "schema": "r12_s7_generator_cell_v1",
                    "modulus": modulus,
                    "current_symbol": observed,
                    "next_symbol": binding.successor[observed],
                    "false_next_symbol": false_successor[observed],
                    "zero_symbol": binding.zero_symbol,
                    "supervision": "successor_and_zero_only",
                }
            )
        for law in s7_law_pools(modulus)["train"]:
            card_y0, card_y1 = binding.card(law)
            for observed in range(modulus):
                atomic_rows.append(
                    {
                        "schema": "r12_s7_transformer_atomic_cell_v1",
                        "modulus": modulus,
                        "card_y0": card_y0,
                        "card_y1": card_y1,
                        "current_location": observed,
                        "destination": binding.destination(law, observed),
                        "control_law_id": control_law_id,
                        "supervision": "atomic_destination_only",
                    }
                )
            control_law_id += 1
    return generator_rows, atomic_rows


def build_atomic_development_rows(
    bindings: dict[int, SymbolBinding],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for modulus in PRIMARY_MODULI:
        binding = bindings[modulus]
        for law in s7_law_pools(modulus)["development"]:
            card_y0, card_y1 = binding.card(law)
            for observed in range(modulus):
                rows.append(
                    {
                        "schema": "r12_s7_atomic_development_cell_v1",
                        "modulus": modulus,
                        "card_y0": card_y0,
                        "card_y1": card_y1,
                        "current_location": observed,
                        "destination": binding.destination(law, observed),
                    }
                )
    return rows


def _build_program_row(
    rng: random.Random,
    binding: SymbolBinding,
    laws: Sequence[AffineLaw],
    depth: int,
    row_index: int,
    board_role: str,
) -> dict[str, object]:
    modulus = binding.modulus
    pool_size = min(len(laws), 2 + rng.randrange(3))
    law_pool = rng.sample(list(laws), pool_size)
    used_names: set[str] = set()
    names = {law: _nonce_name(rng, used_names) for law in law_pool}
    cards = {
        names[law]: {
            "card_y0": binding.card(law)[0],
            "card_y1": binding.card(law)[1],
        }
        for law in law_pool
    }
    initial = list(range(modulus))
    rng.shuffle(initial)
    state = tuple(initial)
    events: list[dict[str, object]] = []
    required = list(law_pool[:2])
    for step in range(depth):
        law = required[step] if step < len(required) else rng.choice(law_pool)
        identity = rng.randrange(modulus)
        source = state.index(identity)
        state = pop_insert(state, identity, binding.destination(law, source))
        events.append({"operation": names[law], "identity": identity})
    query_position = rng.randrange(modulus)
    return {
        "schema": "r12_s7_cayley_program_v1",
        "board_role": board_role,
        "row_id": f"{board_role}_{row_index:06d}",
        "modulus": modulus,
        "depth": depth,
        "initial_state": initial,
        "law_cards": cards,
        "events": events,
        "query_position": query_position,
        "final_state": list(state),
        "answer": state[query_position],
        "distinct_laws": len({event["operation"] for event in events}),
    }


def build_program_rows(
    rng: random.Random,
    bindings: dict[int, SymbolBinding],
    role: str,
    count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cells = [(modulus, depth) for modulus in PRIMARY_MODULI for depth in range(3, 9)]
    rng.shuffle(cells)
    for index in range(count):
        modulus, depth = cells[index % len(cells)]
        rows.append(
            _build_program_row(
                rng,
                bindings[modulus],
                s7_law_pools(modulus)[role],
                depth,
                index,
                role,
            )
        )
    return rows


def _audit(
    generator_rows: list[dict[str, object]],
    atomic_train: list[dict[str, object]],
    atomic_development: list[dict[str, object]],
    development: list[dict[str, object]],
    confirmation: list[dict[str, object]],
) -> dict[str, object]:
    if len(generator_rows) != 23:
        raise ValueError("S7 must train exactly 23 successor cells")
    if any(row["supervision"] != "successor_and_zero_only" for row in generator_rows):
        raise ValueError("S7 generator training schema mismatch")
    if any(row["supervision"] != "atomic_destination_only" for row in atomic_train):
        raise ValueError("S7 transformer control schema mismatch")
    if any(int(row["distinct_laws"]) < 2 for row in development + confirmation):
        raise ValueError("S7 program does not contain two laws")

    cell_counts: dict[str, int] = {}
    for row in development:
        key = f"m{row['modulus']}_d{row['depth']}"
        cell_counts[key] = cell_counts.get(key, 0) + 1
    if max(cell_counts.values()) - min(cell_counts.values()) > 1:
        raise ValueError("S7 development cells are imbalanced")

    law_counts = {
        str(modulus): {
            key: len(value)
            for key, value in s7_law_pools(modulus).items()
        }
        for modulus in PRIMARY_MODULI
    }
    return {
        "generator_training_rows": len(generator_rows),
        "zero_anchor_count": len(PRIMARY_MODULI),
        "transformer_atomic_training_rows": len(atomic_train),
        "atomic_development_rows": len(atomic_development),
        "development_rows": len(development),
        "confirmation_rows": len(confirmation),
        "development_cell_counts": dict(sorted(cell_counts.items())),
        "law_counts": law_counts,
        "minimum_distinct_laws_per_program": min(
            int(row["distinct_laws"]) for row in development + confirmation
        ),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--development-rows", type=int, default=2048)
    parser.add_argument("--confirmation-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing S7 board: {args.out_dir}")
    if args.development_rows < 18 or args.confirmation_rows < 18:
        raise SystemExit("S7 boards are too small")

    rng = random.Random(args.seed)
    bindings = {modulus: _binding(rng, modulus) for modulus in PRIMARY_MODULI}
    generator_rows, atomic_train = build_training_rows(bindings)
    atomic_development = build_atomic_development_rows(bindings)
    development = build_program_rows(
        rng, bindings, "development", args.development_rows
    )
    confirmation = build_program_rows(
        rng, bindings, "confirmation", args.confirmation_rows
    )
    audit = _audit(
        generator_rows,
        atomic_train,
        atomic_development,
        development,
        confirmation,
    )
    payloads = {
        "generator_train.jsonl": _jsonl_bytes(generator_rows),
        "transformer_atomic_train.jsonl": _jsonl_bytes(atomic_train),
        "atomic_development.jsonl": _jsonl_bytes(atomic_development),
        "development.jsonl": _jsonl_bytes(development),
        "confirmation.sealed.jsonl": _jsonl_bytes(confirmation),
    }
    binding_hashes = {
        str(modulus): _sha256(
            json.dumps(
                bindings[modulus].observed_to_latent,
                separators=(",", ":"),
            ).encode("ascii")
        )
        for modulus in PRIMARY_MODULI
    }
    report = {
        "schema": "r12_s7_learned_cayley_board_report_v1",
        "decision": "admit_s7_learned_cayley_board",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "binding_hashes": binding_hashes,
        "audit": audit,
        "files": {
            name: {"sha256": _sha256(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    args.out_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        (args.out_dir / name).write_bytes(payload)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"decision": report["decision"], "audit": audit}, sort_keys=True))


if __name__ == "__main__":
    main()
