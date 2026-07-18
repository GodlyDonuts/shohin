#!/usr/bin/env python3
"""CPU mechanics for R12 Conflict-Driven Residual Localization cores.

Exact residual-preserving core extraction for two finite families:

1. Heisenberg-mod-M with identity padding events (compressible positive family)
2. Addressed one-register writes without padding (non-compressible negative)

No neural fit, Shohin checkpoint, or GPU path is included. This module only
supports the CPU falsifier gates in
``R12_CONFLICT_DRIVEN_RESIDUAL_LOCALIZATION.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Heisenberg mod M with padding
# ---------------------------------------------------------------------------


HeisenbergState = tuple[int, int, int]


def heisenberg_apply(state: HeisenbergState, event: str, modulus: int) -> HeisenbergState:
    x, y, z = state
    m = modulus
    if event == "A":
        return ((x + 1) % m, y, z)
    if event == "B":
        return (x, (y + 1) % m, (z + x) % m)
    if event == "C":
        return (x, y, (z + 1) % m)
    if event == "P":
        return state
    raise ValueError(f"unknown Heisenberg event {event!r}")


def heisenberg_fold(
    events: Sequence[str], *, modulus: int, start: HeisenbergState = (0, 0, 0)
) -> HeisenbergState:
    state = start
    for event in events:
        state = heisenberg_apply(state, event, modulus)
    return state


def heisenberg_residual_key(events: Sequence[str], *, modulus: int) -> HeisenbergState:
    """Residual class representative: final state from the zero start."""
    return heisenberg_fold(events, modulus=modulus)


def heisenberg_all_query_answers(state: HeisenbergState) -> tuple[int, int, int]:
    return state


# ---------------------------------------------------------------------------
# Free-word residual (negative control)
# ---------------------------------------------------------------------------
# Late queries may ask for any event by index, so the residual class of a word
# is the word itself. Every event is essential; cores must equal histories.
# Register-overwrite families are compressible and are not used here.


def free_word_residual_key(events: Sequence[str]) -> tuple[str, ...]:
    return tuple(events)


def register_residual_key(
    events: Sequence[str], *, modulus: int, n_registers: int
) -> tuple[int, ...]:
    """Legacy helper retained for overwrite unit tests only."""
    state = [0] * n_registers
    for event in events:
        if not event.startswith("W") or ":" not in event:
            raise ValueError(f"unknown register event {event!r}")
        idx_s, val_s = event[1:].split(":", 1)
        idx = int(idx_s)
        val = int(val_s) % modulus
        if not (0 <= idx < n_registers):
            raise ValueError(f"register index out of range: {idx}")
        state[idx] = val
    return tuple(state)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreExtraction:
    history: tuple[str, ...]
    core: tuple[str, ...]
    residual_key: tuple
    oracle_calls: int
    family: str


def _subsequence_by_mask(history: Sequence[str], mask: Sequence[bool]) -> tuple[str, ...]:
    return tuple(event for event, keep in zip(history, mask) if keep)


def extract_lex_min_core(
    history: Sequence[str],
    residual_key_fn: Callable[[Sequence[str]], tuple],
    *,
    family: str,
) -> CoreExtraction:
    """Return the lexicographically-first minimum-length residual-preserving core.

    Lexicographic order is over boolean masks of length ``len(history)``, which
    is equivalent to preferring earlier-index keep sets when lengths tie.
    """
    history_t = tuple(history)
    n = len(history_t)
    target = residual_key_fn(history_t)
    oracle_calls = 1  # target itself

    if n == 0:
        return CoreExtraction(history_t, (), target, oracle_calls, family)

    # Search by increasing core length; within a length, try masks in lex order
    # of their index tuples.
    for length in range(0, n + 1):
        for idx_tuple in combinations(range(n), length):
            mask = [False] * n
            for i in idx_tuple:
                mask[i] = True
            candidate = _subsequence_by_mask(history_t, mask)
            key = residual_key_fn(candidate)
            oracle_calls += 1
            if key == target:
                return CoreExtraction(
                    history=history_t,
                    core=candidate,
                    residual_key=target,
                    oracle_calls=oracle_calls,
                    family=family,
                )

    raise RuntimeError("core extraction failed; residual_key_fn is inconsistent")


def extract_heisenberg_core(
    history: Sequence[str], *, modulus: int
) -> CoreExtraction:
    return extract_lex_min_core(
        history,
        lambda events: heisenberg_residual_key(events, modulus=modulus),
        family="heisenberg_padding",
    )


def extract_free_word_core(history: Sequence[str]) -> CoreExtraction:
    return extract_lex_min_core(
        history,
        free_word_residual_key,
        family="free_word_negative",
    )


def extract_register_core(
    history: Sequence[str], *, modulus: int, n_registers: int
) -> CoreExtraction:
    """Overwrite-aware register core helper for unit tests only."""
    return extract_lex_min_core(
        history,
        lambda events: register_residual_key(
            events, modulus=modulus, n_registers=n_registers
        ),
        family="register_overwrite_unit",
    )


# ---------------------------------------------------------------------------
# Board mechanics / gates
# ---------------------------------------------------------------------------


def _stable_hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_heisenberg_mechanics_board(modulus: int = 5) -> dict:
    """Frozen compressible examples with identity padding."""
    cases = [
        ("pad_only", ("P", "P", "P")),
        ("A_with_pads", ("P", "A", "P", "P")),
        ("AB_with_pads", ("P", "A", "P", "B", "P")),
        ("CAB_padded", ("C", "P", "A", "P", "B", "P", "P")),
        ("long_pad_A", tuple(["P"] * 6 + ["A"] + ["P"] * 4)),
        ("multi_essential", ("A", "P", "A", "P", "B", "P", "C")),
    ]
    rows = []
    for name, history in cases:
        extraction = extract_heisenberg_core(history, modulus=modulus)
        rows.append(
            {
                "name": name,
                "history": list(extraction.history),
                "core": list(extraction.core),
                "residual_key": list(extraction.residual_key),
                "oracle_calls": extraction.oracle_calls,
                "core_has_padding": "P" in extraction.core,
                "core_shorter": len(extraction.core) < len(extraction.history),
            }
        )
    return {
        "family": "heisenberg_padding",
        "modulus": modulus,
        "rows": rows,
        "board_sha256": _stable_hash({"modulus": modulus, "rows": rows}),
    }


def build_free_word_negative_board() -> dict:
    """Free-word residual: cores must equal histories for every sample."""
    cases = [
        ("single", ("A",)),
        ("two", ("A", "B")),
        ("three", ("A", "B", "C")),
        ("repeat", ("A", "A", "A")),
        ("mixed", ("B", "A", "C", "B")),
    ]
    rows = []
    for name, history in cases:
        extraction = extract_free_word_core(history)
        rows.append(
            {
                "name": name,
                "history": list(extraction.history),
                "core": list(extraction.core),
                "residual_key": list(extraction.residual_key),
                "oracle_calls": extraction.oracle_calls,
                "core_equals_history": extraction.core == extraction.history,
            }
        )
    return {
        "family": "free_word_negative",
        "rows": rows,
        "board_sha256": _stable_hash({"rows": rows}),
    }


def evaluate_mechanics_gates(
    heisenberg_board: dict | None = None,
    free_word_board: dict | None = None,
) -> dict:
    heisenberg_board = heisenberg_board or build_heisenberg_mechanics_board()
    free_word_board = free_word_board or build_free_word_negative_board()
    modulus = heisenberg_board["modulus"]

    gates = []

    # Gate 1: deterministic replay
    replay = build_heisenberg_mechanics_board(modulus=modulus)
    gates.append(
        {
            "id": "deterministic_heisenberg_board",
            "pass": replay["board_sha256"] == heisenberg_board["board_sha256"],
        }
    )
    replay_free = build_free_word_negative_board()
    gates.append(
        {
            "id": "deterministic_free_word_board",
            "pass": replay_free["board_sha256"] == free_word_board["board_sha256"],
        }
    )

    # Gate 2/3: residual preservation and no padding in cores
    heisenberg_preserve = True
    no_padding = True
    some_shorter = False
    for row in heisenberg_board["rows"]:
        key = heisenberg_residual_key(row["core"], modulus=modulus)
        if tuple(key) != tuple(row["residual_key"]):
            heisenberg_preserve = False
        if row["core_has_padding"]:
            no_padding = False
        if row["core_shorter"]:
            some_shorter = True
    gates.append({"id": "heisenberg_residual_preservation", "pass": heisenberg_preserve})
    gates.append({"id": "heisenberg_cores_strip_padding", "pass": no_padding})
    gates.append({"id": "heisenberg_some_cores_shorter", "pass": some_shorter})

    # Gate 4: negative control cores equal histories
    negative_ok = all(row["core_equals_history"] for row in free_word_board["rows"])
    gates.append({"id": "free_word_cores_equal_histories", "pass": negative_ok})

    # Gate 5: oracle calls are positive and reported
    oracle_ok = all(row["oracle_calls"] >= 1 for row in heisenberg_board["rows"]) and all(
        row["oracle_calls"] >= 1 for row in free_word_board["rows"]
    )
    gates.append({"id": "oracle_calls_reported", "pass": oracle_ok})

    # Gate 6: length law identity for rand matching is structural (same lengths)
    length_law = [len(row["core"]) for row in heisenberg_board["rows"]]
    gates.append(
        {
            "id": "core_length_law_exported",
            "pass": len(length_law) == len(heisenberg_board["rows"]),
            "lengths": length_law,
        }
    )

    all_pass = all(gate["pass"] for gate in gates)
    report = {
        "protocol": "R12-CDRL-CPU-MECHANICS-v1",
        "all_pass": all_pass,
        "gates": gates,
        "heisenberg_board_sha256": heisenberg_board["board_sha256"],
        "free_word_board_sha256": free_word_board["board_sha256"],
    }
    report["report_sha256"] = _stable_hash(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full mechanics report as JSON",
    )
    args = parser.parse_args(argv)
    report = evaluate_mechanics_gates()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["all_pass"] else "FAIL"
        print(f"CDRL CPU mechanics: {status}")
        for gate in report["gates"]:
            mark = "ok" if gate["pass"] else "FAIL"
            print(f"  [{mark}] {gate['id']}")
        print(f"report_sha256={report['report_sha256']}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
