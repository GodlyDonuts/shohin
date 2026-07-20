#!/usr/bin/env python3
"""Finite mechanics for Episodic Rule-Card Categorical State Transport."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Mapping, Sequence


ENTITY_COUNT = 3
PERMUTATIONS = tuple(itertools.permutations(range(ENTITY_COUNT)))
REPORT_SCHEMA = "r12_er_cst_rule_card_cpu_falsifier_v1"

Permutation = tuple[int, int, int]
State = tuple[str, str, str]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def infer_position_permutation(
    before: Sequence[str], after: Sequence[str]
) -> Permutation:
    """Infer the unique output-position to input-position permutation."""
    if len(before) != ENTITY_COUNT or len(after) != ENTITY_COUNT:
        raise ValueError("rule witnesses require exactly three symbols")
    before_tuple = tuple(map(str, before))
    after_tuple = tuple(map(str, after))
    if len(set(before_tuple)) != ENTITY_COUNT:
        raise ValueError("rule-witness input symbols must be distinct")
    if set(after_tuple) != set(before_tuple):
        raise ValueError("rule-witness output must be a permutation of its input")
    return tuple(before_tuple.index(symbol) for symbol in after_tuple)  # type: ignore[return-value]


def apply_position_permutation(
    state: Sequence[str], permutation: Sequence[int]
) -> State:
    if len(state) != ENTITY_COUNT or len(set(map(str, state))) != ENTITY_COUNT:
        raise ValueError("state must contain three distinct symbols")
    permutation_tuple = tuple(map(int, permutation))
    if permutation_tuple not in PERMUTATIONS:
        raise ValueError("rule card is not a three-position permutation")
    state_tuple = tuple(map(str, state))
    return tuple(state_tuple[index] for index in permutation_tuple)  # type: ignore[return-value]


def execute_rule_program(
    initial: Sequence[str],
    cards: Mapping[str, Sequence[int]],
    program: Sequence[str],
    halt_after: int,
) -> tuple[State, tuple[State, ...]]:
    if not 0 <= halt_after <= len(program):
        raise ValueError("HALT position is outside the program")
    state = tuple(map(str, initial))
    if len(state) != ENTITY_COUNT or len(set(state)) != ENTITY_COUNT:
        raise ValueError("initial state must contain three distinct symbols")
    trajectory = [state]
    for opcode in program[:halt_after]:
        if opcode not in cards:
            raise ValueError(f"program references unknown rule card: {opcode}")
        state = apply_position_permutation(state, cards[opcode])
        trajectory.append(state)
    return state, tuple(trajectory)


def _names(prefix: str, index: int) -> State:
    return tuple(f"{prefix}-{index}-{slot}" for slot in range(ENTITY_COUNT))  # type: ignore[return-value]


def _random_cards(
    rng: random.Random, trial: int
) -> tuple[dict[str, Permutation], dict[str, tuple[State, State]]]:
    opcodes = tuple(f"opcode-{trial}-{index}" for index in range(3))
    chosen = rng.sample(PERMUTATIONS, len(opcodes))
    cards: dict[str, Permutation] = {}
    witnesses: dict[str, tuple[State, State]] = {}
    for index, (opcode, permutation) in enumerate(zip(opcodes, chosen, strict=True)):
        symbols = list(_names("witness", trial * 3 + index))
        rng.shuffle(symbols)
        before = tuple(symbols)
        after = tuple(before[position] for position in permutation)
        inferred = infer_position_permutation(before, after)
        cards[opcode] = inferred
        witnesses[opcode] = (before, after)
    return cards, witnesses


def _renamed_episode(
    cards: Mapping[str, Permutation],
    program: Sequence[str],
    trial: int,
) -> tuple[dict[str, Permutation], tuple[str, ...]]:
    old = tuple(sorted(cards))
    new = tuple(f"renamed-{trial}-{index}" for index in range(len(old)))
    mapping = dict(zip(old, new, strict=True))
    return (
        {mapping[opcode]: tuple(cards[opcode]) for opcode in old},
        tuple(mapping[opcode] for opcode in program),
    )


def run_falsifier(*, seed: int, trials: int = 10_000) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = random.Random(seed)
    counts = {
        "witness_inference_exact": 0,
        "execution_exact": 0,
        "witness_rename_invariant": 0,
        "opcode_rename_invariant": 0,
        "card_storage_reindex_invariant": 0,
        "post_halt_suffix_invariant": 0,
        "deranged_card_state_exact": 0,
    }
    registration_rows: list[str] = []
    for trial in range(trials):
        cards, witnesses = _random_cards(rng, trial)
        if all(
            infer_position_permutation(before, after) == cards[opcode]
            for opcode, (before, after) in witnesses.items()
        ):
            counts["witness_inference_exact"] += 1

        state = list(_names("entity", trial))
        rng.shuffle(state)
        opcodes = tuple(cards)
        depth = rng.randint(1, 12)
        program = tuple(rng.choice(opcodes) for _ in range(depth))
        halt_after = rng.randint(1, depth)
        expected = tuple(state)
        expected_trajectory = [expected]
        for opcode in program[:halt_after]:
            expected = tuple(expected[index] for index in cards[opcode])
            expected_trajectory.append(expected)
        final, trajectory = execute_rule_program(state, cards, program, halt_after)
        if final == expected and trajectory == tuple(expected_trajectory):
            counts["execution_exact"] += 1

        renamed_witnesses = {
            opcode: (
                tuple(f"alpha-{trial}-{before.index(value)}" for value in before),
                tuple(f"alpha-{trial}-{before.index(value)}" for value in after),
            )
            for opcode, (before, after) in witnesses.items()
        }
        if all(
            infer_position_permutation(before, after) == cards[opcode]
            for opcode, (before, after) in renamed_witnesses.items()
        ):
            counts["witness_rename_invariant"] += 1

        renamed_cards, renamed_program = _renamed_episode(cards, program, trial)
        renamed_final, renamed_trajectory = execute_rule_program(
            state, renamed_cards, renamed_program, halt_after
        )
        if renamed_final == final and renamed_trajectory == trajectory:
            counts["opcode_rename_invariant"] += 1

        reversed_cards = dict(reversed(tuple(cards.items())))
        if execute_rule_program(state, reversed_cards, program, halt_after) == (
            final,
            trajectory,
        ):
            counts["card_storage_reindex_invariant"] += 1

        suffix = tuple(rng.choice(opcodes) for _ in range(4))
        if execute_rule_program(
            state, cards, program + suffix, halt_after
        ) == (final, trajectory):
            counts["post_halt_suffix_invariant"] += 1

        deranged_values = tuple(cards.values())[1:] + tuple(cards.values())[:1]
        deranged = dict(zip(opcodes, deranged_values, strict=True))
        deranged_final, _ = execute_rule_program(state, deranged, program, halt_after)
        if deranged_final == final:
            counts["deranged_card_state_exact"] += 1

        registration_rows.append(
            canonical_json(
                {
                    "cards": cards,
                    "halt_after": halt_after,
                    "initial": state,
                    "program": program,
                    "trial": trial,
                    "witnesses": witnesses,
                }
            )
        )

    rates = {name: value / trials for name, value in counts.items()}
    gates = {
        "witness_inference_exact": counts["witness_inference_exact"] == trials,
        "execution_exact": counts["execution_exact"] == trials,
        "witness_rename_invariant": counts["witness_rename_invariant"] == trials,
        "opcode_rename_invariant": counts["opcode_rename_invariant"] == trials,
        "card_storage_reindex_invariant": counts["card_storage_reindex_invariant"]
        == trials,
        "post_halt_suffix_invariant": counts["post_halt_suffix_invariant"]
        == trials,
        "deranged_card_state_below_40pct": rates["deranged_card_state_exact"]
        < 0.40,
    }
    return {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "trials": trials,
        "counts": counts,
        "rates": rates,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "episode_registration_sha256": sha256_bytes(
            ("\n".join(registration_rows) + "\n").encode("utf-8")
        ),
        "claim_boundary": (
            "Finite CPU mechanics only; no neural compiler, fresh-board score, "
            "or general-reasoning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    report = run_falsifier(seed=args.seed, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(canonical_json({"all_gates_pass": report["all_gates_pass"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
