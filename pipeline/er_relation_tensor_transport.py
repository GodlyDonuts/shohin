#!/usr/bin/env python3
"""CPU mechanics for variable-cardinality Episodic Relation Tensor Transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence


MIN_CARDINALITY = 3
MAX_CARDINALITY = 6
MIN_RULES = 2
MAX_RULES = 4
REPORT_SCHEMA = "r12_er_relation_tensor_cpu_falsifier_v1"

Relation = tuple[int, ...]
State = tuple[str, ...]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def infer_copy_relation(before: Sequence[str], after: Sequence[str]) -> Relation:
    """Infer output-position to input-position routing, allowing repeated outputs."""
    before_tuple = tuple(map(str, before))
    after_tuple = tuple(map(str, after))
    if not MIN_CARDINALITY <= len(before_tuple) <= MAX_CARDINALITY:
        raise ValueError("relation witness cardinality is outside the admitted range")
    if len(after_tuple) != len(before_tuple):
        raise ValueError("relation witness sides have different cardinality")
    if len(set(before_tuple)) != len(before_tuple):
        raise ValueError("relation witness inputs must be distinct")
    if any(symbol not in before_tuple for symbol in after_tuple):
        raise ValueError("relation witness output references an unknown input")
    return tuple(before_tuple.index(symbol) for symbol in after_tuple)


def apply_copy_relation(state: Sequence[str], relation: Sequence[int]) -> State:
    state_tuple = tuple(map(str, state))
    relation_tuple = tuple(map(int, relation))
    if not MIN_CARDINALITY <= len(state_tuple) <= MAX_CARDINALITY:
        raise ValueError("relation state cardinality is outside the admitted range")
    if len(relation_tuple) != len(state_tuple) or any(
        index < 0 or index >= len(state_tuple) for index in relation_tuple
    ):
        raise ValueError("relation row references an invalid input position")
    return tuple(state_tuple[index] for index in relation_tuple)


def compose_relations(outer: Sequence[int], inner: Sequence[int]) -> Relation:
    """Return the relation equivalent to applying inner and then outer."""
    outer_tuple = tuple(map(int, outer))
    inner_tuple = tuple(map(int, inner))
    if len(outer_tuple) != len(inner_tuple):
        raise ValueError("relation composition cardinality differs")
    if any(index < 0 or index >= len(inner_tuple) for index in outer_tuple + inner_tuple):
        raise ValueError("relation composition index is invalid")
    return tuple(inner_tuple[index] for index in outer_tuple)


def execute_relation_program(
    initial: Sequence[str],
    cards: Mapping[str, Sequence[int]],
    program: Sequence[str],
    halt_after: int,
) -> tuple[State, tuple[State, ...]]:
    if not 0 <= halt_after <= len(program):
        raise ValueError("HALT position is outside the relation program")
    state = tuple(map(str, initial))
    if len(set(state)) != len(state):
        raise ValueError("initial relation state must contain distinct entities")
    trajectory = [state]
    for opcode in program[:halt_after]:
        if opcode not in cards:
            raise ValueError(f"relation program references unknown opcode: {opcode}")
        state = apply_copy_relation(state, cards[opcode])
        trajectory.append(state)
    return state, tuple(trajectory)


def pad_relation(relation: Sequence[int], width: int = MAX_CARDINALITY) -> Relation:
    relation_tuple = tuple(map(int, relation))
    if len(relation_tuple) > width:
        raise ValueError("relation padding width is too small")
    return relation_tuple + tuple(range(len(relation_tuple), width))


def _random_cards(
    rng: random.Random, trial: int, cardinality: int, count: int
) -> tuple[dict[str, Relation], dict[str, tuple[State, State]]]:
    cards: dict[str, Relation] = {}
    witnesses: dict[str, tuple[State, State]] = {}
    for rule in range(count):
        opcode = f"op-{trial}-{rule}"
        relation = tuple(rng.randrange(cardinality) for _ in range(cardinality))
        symbols = [f"w-{trial}-{rule}-{index}" for index in range(cardinality)]
        rng.shuffle(symbols)
        before = tuple(symbols)
        after = tuple(before[index] for index in relation)
        cards[opcode] = infer_copy_relation(before, after)
        witnesses[opcode] = (before, after)
    return cards, witnesses


def run_falsifier(*, seed: int, trials: int = 10_000) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = random.Random(seed)
    counts = {
        "witness_inference_exact": 0,
        "execution_exact": 0,
        "relation_composition_exact": 0,
        "witness_alpha_invariant": 0,
        "opcode_alpha_invariant": 0,
        "card_storage_reindex_invariant": 0,
        "cardinality_padding_invariant": 0,
        "post_halt_suffix_invariant": 0,
        "source_deleted_packet_invariant": 0,
        "non_bijective_episode": 0,
        "deranged_card_state_exact": 0,
        "equality_ablated_state_exact": 0,
    }
    cardinalities = {value: 0 for value in range(MIN_CARDINALITY, MAX_CARDINALITY + 1)}
    registration_rows: list[str] = []
    for trial in range(trials):
        cardinality = MIN_CARDINALITY + trial % (MAX_CARDINALITY - MIN_CARDINALITY + 1)
        rule_count = MIN_RULES + trial % (MAX_RULES - MIN_RULES + 1)
        cardinalities[cardinality] += 1
        cards, witnesses = _random_cards(rng, trial, cardinality, rule_count)
        if all(
            infer_copy_relation(before, after) == cards[opcode]
            for opcode, (before, after) in witnesses.items()
        ):
            counts["witness_inference_exact"] += 1
        if any(len(set(relation)) < cardinality for relation in cards.values()):
            counts["non_bijective_episode"] += 1

        initial = [f"e-{trial}-{index}" for index in range(cardinality)]
        rng.shuffle(initial)
        opcodes = tuple(cards)
        depth = rng.randint(1, 12)
        program = tuple(rng.choice(opcodes) for _ in range(depth))
        halt_after = rng.randint(1, depth)
        expected = tuple(initial)
        expected_trajectory = [expected]
        accumulated = tuple(range(cardinality))
        for opcode in program[:halt_after]:
            expected = tuple(expected[index] for index in cards[opcode])
            expected_trajectory.append(expected)
            accumulated = compose_relations(cards[opcode], accumulated)
        final, trajectory = execute_relation_program(initial, cards, program, halt_after)
        if final == expected and trajectory == tuple(expected_trajectory):
            counts["execution_exact"] += 1
        if apply_copy_relation(initial, accumulated) == final:
            counts["relation_composition_exact"] += 1

        renamed_witnesses = {}
        for opcode, (before, after) in witnesses.items():
            mapping = {value: f"alpha-{trial}-{index}" for index, value in enumerate(before)}
            renamed_witnesses[opcode] = (
                tuple(mapping[value] for value in before),
                tuple(mapping[value] for value in after),
            )
        if all(
            infer_copy_relation(before, after) == cards[opcode]
            for opcode, (before, after) in renamed_witnesses.items()
        ):
            counts["witness_alpha_invariant"] += 1

        renamed = {opcode: f"renamed-{trial}-{index}" for index, opcode in enumerate(opcodes)}
        renamed_cards = {renamed[name]: value for name, value in cards.items()}
        renamed_program = tuple(renamed[name] for name in program)
        if execute_relation_program(initial, renamed_cards, renamed_program, halt_after) == (
            final,
            trajectory,
        ):
            counts["opcode_alpha_invariant"] += 1
        if execute_relation_program(
            initial, dict(reversed(tuple(cards.items()))), program, halt_after
        ) == (final, trajectory):
            counts["card_storage_reindex_invariant"] += 1

        padded_state = tuple(initial) + tuple(
            f"pad-{trial}-{index}" for index in range(cardinality, MAX_CARDINALITY)
        )
        padded_cards = {name: pad_relation(value) for name, value in cards.items()}
        padded_final, _ = execute_relation_program(
            padded_state, padded_cards, program, halt_after
        )
        if padded_final[:cardinality] == final:
            counts["cardinality_padding_invariant"] += 1

        suffix = tuple(rng.choice(opcodes) for _ in range(4))
        if execute_relation_program(initial, cards, program + suffix, halt_after) == (
            final,
            trajectory,
        ):
            counts["post_halt_suffix_invariant"] += 1
        compiled_cards = {name: tuple(value) for name, value in cards.items()}
        poisoned_witnesses = {name: (("poison",), ("poison",)) for name in witnesses}
        if poisoned_witnesses and execute_relation_program(
            initial, compiled_cards, program, halt_after
        ) == (final, trajectory):
            counts["source_deleted_packet_invariant"] += 1

        rotated = tuple(cards.values())[1:] + tuple(cards.values())[:1]
        deranged = dict(zip(opcodes, rotated, strict=True))
        if execute_relation_program(initial, deranged, program, halt_after)[0] == final:
            counts["deranged_card_state_exact"] += 1
        ablated = {
            name: tuple(rng.randrange(cardinality) for _ in range(cardinality))
            for name in opcodes
        }
        if execute_relation_program(initial, ablated, program, halt_after)[0] == final:
            counts["equality_ablated_state_exact"] += 1

        registration_rows.append(
            canonical_json(
                {
                    "cards": cards,
                    "cardinality": cardinality,
                    "halt_after": halt_after,
                    "initial": initial,
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
        "relation_composition_exact": counts["relation_composition_exact"] == trials,
        "witness_alpha_invariant": counts["witness_alpha_invariant"] == trials,
        "opcode_alpha_invariant": counts["opcode_alpha_invariant"] == trials,
        "card_storage_reindex_invariant": counts["card_storage_reindex_invariant"] == trials,
        "cardinality_padding_invariant": counts["cardinality_padding_invariant"] == trials,
        "post_halt_suffix_invariant": counts["post_halt_suffix_invariant"] == trials,
        "source_deleted_packet_invariant": counts["source_deleted_packet_invariant"] == trials,
        "non_bijective_episodes_at_least_90pct": rates["non_bijective_episode"] >= 0.90,
        "deranged_card_state_below_40pct": rates["deranged_card_state_exact"] < 0.40,
        "equality_ablated_state_below_40pct": rates["equality_ablated_state_exact"] < 0.40,
        "cardinality_exact_balanced": max(cardinalities.values()) - min(cardinalities.values()) <= 1,
    }
    return {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "trials": trials,
        "cardinality_counts": cardinalities,
        "counts": counts,
        "rates": rates,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "episode_registration_sha256": hashlib.sha256(
            ("\n".join(registration_rows) + "\n").encode()
        ).hexdigest(),
        "claim_boundary": (
            "Variable-cardinality finite relation mechanics only; no neural compiler, "
            "fresh-board score, or broad-reasoning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing ER relation output: {args.output}")
    report = run_falsifier(seed=args.seed, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(canonical_json({"all_gates_pass": report["all_gates_pass"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
