#!/usr/bin/env python3
"""CPU theorem and causal falsifier for S8 nil-linked law graphs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

from s6_contextual_affine_law import AffineLaw, pop_insert
from s7_learned_cayley_law import SymbolBinding
from s8_nil_linked_law_graph import (
    derange_cards,
    execute_graph,
    graph_from_ordered_events,
    linked_path,
    one_witness_unit_completion,
    rewire_path,
)


MODULUS_BINDING_COUNTS = {5: 120, 7: 128, 11: 128, 13: 64}
PROGRAMS_PER_BINDING = 8
CONTROLS = (
    "storage_order_shortcut",
    "reversed_links",
    "deranged_cards",
    "one_witness",
    "state_reset",
    "early_halt",
)


def _bindings(rng: random.Random, modulus: int, count: int) -> list[SymbolBinding]:
    if modulus == 5:
        values = list(itertools.permutations(range(modulus)))
        if len(values) != count:
            raise ValueError("S8 exhaustive modulus-five binding count mismatch")
        return [SymbolBinding(modulus, value) for value in values]
    seen: set[tuple[int, ...]] = set()
    while len(seen) < count:
        value = list(range(modulus))
        rng.shuffle(value)
        seen.add(tuple(value))
    return [SymbolBinding(modulus, value) for value in sorted(seen)]


def _reference(
    binding: SymbolBinding,
    initial_state: tuple[int, ...],
    laws: dict[str, AffineLaw],
    events: tuple[tuple[int, str], ...],
    query_position: int,
) -> tuple[tuple[int, ...], int]:
    state = initial_state
    for identity, operation in events:
        source = state.index(identity)
        destination = binding.destination(laws[operation], source)
        state = pop_insert(state, identity, destination)
    return state, int(state[query_position])


def _score_cell(
    scores: dict[str, dict[str, int]],
    arm: str,
    predicted: tuple[tuple[int, ...], int],
    expected: tuple[tuple[int, ...], int],
) -> None:
    cell = scores.setdefault(
        arm,
        {"state_correct": 0, "answer_correct": 0, "total": 0},
    )
    cell["state_correct"] += int(predicted[0] == expected[0])
    cell["answer_correct"] += int(predicted[1] == expected[1])
    cell["total"] += 1


def _finish(scores: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for arm, cell in sorted(scores.items()):
        total = cell["total"]
        result[arm] = {
            **cell,
            "state_accuracy": cell["state_correct"] / total,
            "answer_accuracy": cell["answer_correct"] / total,
        }
    return result


def run_falsifier(seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    scores: dict[str, dict[str, int]] = {}
    by_modulus: dict[str, dict[str, dict[str, int]]] = {}
    storage_permutation_invariant = 0
    noncanonical_paths = 0
    total = 0
    for modulus, binding_count in MODULUS_BINDING_COUNTS.items():
        modulus_scores: dict[str, dict[str, int]] = {}
        laws_all = [
            AffineLaw(modulus, slope, intercept)
            for slope in range(1, modulus)
            for intercept in range(modulus)
        ]
        for binding in _bindings(rng, modulus, binding_count):
            for _ in range(PROGRAMS_PER_BINDING):
                law_count = rng.randint(2, min(4, len(laws_all)))
                selected = rng.sample(laws_all, law_count)
                laws = {f"op_{index}": law for index, law in enumerate(selected)}
                cards = {name: binding.card(law) for name, law in laws.items()}
                depth = rng.randint(3, 8)
                operation_names = list(laws)
                event_operations = operation_names[:2] + [
                    rng.choice(operation_names) for _ in range(depth - 2)
                ]
                rng.shuffle(event_operations)
                events = tuple(
                    (rng.randrange(modulus), operation)
                    for operation in event_operations
                )
                initial = list(range(modulus))
                rng.shuffle(initial)
                query_position = rng.randrange(modulus)
                storage_ids = list(range(depth))
                rng.shuffle(storage_ids)
                if storage_ids == list(range(depth)):
                    storage_ids = storage_ids[1:] + storage_ids[:1]
                graph = graph_from_ordered_events(
                    modulus=modulus,
                    initial_state=tuple(initial),
                    cards=cards,
                    events=events,
                    storage_ids=storage_ids,
                    query_position=query_position,
                )
                expected = _reference(
                    binding, tuple(initial), laws, events, query_position
                )
                treatment_full = execute_graph(
                    graph, binding.successor, binding.zero_symbol
                )
                treatment = (treatment_full[0], treatment_full[1])
                _score_cell(scores, "treatment", treatment, expected)
                _score_cell(modulus_scores, "treatment", treatment, expected)

                alternate_ids = list(range(depth))
                rng.shuffle(alternate_ids)
                if alternate_ids == storage_ids:
                    alternate_ids = alternate_ids[1:] + alternate_ids[:1]
                alternate = graph_from_ordered_events(
                    modulus=modulus,
                    initial_state=tuple(initial),
                    cards=cards,
                    events=events,
                    storage_ids=alternate_ids,
                    query_position=query_position,
                )
                alternate_output = execute_graph(
                    alternate, binding.successor, binding.zero_symbol
                )
                storage_permutation_invariant += int(
                    alternate_output[:2] == treatment_full[:2]
                )
                noncanonical_paths += int(linked_path(graph) != tuple(range(depth)))

                controls = {
                    "storage_order_shortcut": execute_graph(
                        graph,
                        binding.successor,
                        binding.zero_symbol,
                        storage_order=True,
                    ),
                    "reversed_links": execute_graph(
                        rewire_path(graph, tuple(reversed(linked_path(graph)))),
                        binding.successor,
                        binding.zero_symbol,
                    ),
                    "deranged_cards": execute_graph(
                        derange_cards(graph),
                        binding.successor,
                        binding.zero_symbol,
                    ),
                    "one_witness": execute_graph(
                        one_witness_unit_completion(
                            graph, binding.successor, binding.zero_symbol
                        ),
                        binding.successor,
                        binding.zero_symbol,
                    ),
                    "state_reset": execute_graph(
                        graph,
                        binding.successor,
                        binding.zero_symbol,
                        reset_state=True,
                    ),
                    "early_halt": execute_graph(
                        graph,
                        binding.successor,
                        binding.zero_symbol,
                        halt_after=1,
                    ),
                }
                for arm, output in controls.items():
                    predicted = (output[0], output[1])
                    _score_cell(scores, arm, predicted, expected)
                    _score_cell(modulus_scores, arm, predicted, expected)
                total += 1
        by_modulus[str(modulus)] = _finish(modulus_scores)

    finished = _finish(scores)
    gates = {
        "treatment_exact": finished["treatment"]["state_accuracy"] == 1.0,
        "treatment_answer_exact": finished["treatment"]["answer_accuracy"] == 1.0,
        "storage_permutation_invariant": storage_permutation_invariant == total,
        "noncanonical_link_paths_at_least_95pct": noncanonical_paths / total >= 0.95,
        **{
            f"{arm}_state_below_40pct": finished[arm]["state_accuracy"] < 0.40
            for arm in CONTROLS
        },
    }
    return {
        "schema": "r12_s8_nil_linked_law_graph_cpu_falsifier_v1",
        "seed": seed,
        "programs": total,
        "bindings": MODULUS_BINDING_COUNTS,
        "programs_per_binding": PROGRAMS_PER_BINDING,
        "scores": finished,
        "scores_by_modulus": by_modulus,
        "storage_permutation_invariant": storage_permutation_invariant,
        "noncanonical_link_paths": noncanonical_paths,
        "gates": gates,
        "decision": (
            "admit_s8_nil_linked_law_graph_preregistration"
            if all(gates.values())
            else "reject_s8_nil_linked_law_graph_mechanics"
        ),
        "resource_boundary": {
            "model_owned": [
                "initial-state symbols",
                "law-card witnesses",
                "event entity and operation pointers",
                "entry and next-event pointers",
                "nil termination",
                "query position",
            ],
            "architectural": [
                "categorical argmax/equality",
                "linked-graph traversal with node-count safety bound",
                "confirmed S7 cyclic compiler",
                "pop-insert state mutation",
            ],
        },
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S8 CPU report: {args.out}")
    report = run_falsifier(args.seed)
    _write(args.out, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "out": str(args.out),
                "programs": report["programs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
