#!/usr/bin/env python3
"""Exhaustive/sampled CPU falsifier for S7 learned Cayley compilation."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
from pathlib import Path
import random

from s6_contextual_affine_law import enumerate_laws, pop_insert
from s7_learned_cayley_law import (
    DIAGNOSTIC_MODULUS,
    PRIMARY_MODULI,
    SymbolBinding,
    compile_destination,
    execute_compiled_program,
    stride_two_successor,
)


CPU_SEED = 2026071907
EXHAUSTIVE_BINDING_MODULI = (5, 7)
SAMPLED_BINDINGS = {11: 256, 13: 128}


def _bindings(modulus: int) -> tuple[SymbolBinding, ...]:
    if modulus in EXHAUSTIVE_BINDING_MODULI:
        return tuple(
            SymbolBinding(modulus, tuple(permutation))
            for permutation in itertools.permutations(range(modulus))
        )
    rng = random.Random(CPU_SEED + modulus)
    target = SAMPLED_BINDINGS[modulus]
    permutations: set[tuple[int, ...]] = {tuple(range(modulus))}
    while len(permutations) < target:
        values = list(range(modulus))
        rng.shuffle(values)
        permutations.add(tuple(values))
    return tuple(SymbolBinding(modulus, values) for values in sorted(permutations))


def _audit_modulus(modulus: int) -> dict[str, object]:
    laws = enumerate_laws(modulus)
    bindings = _bindings(modulus)
    exact = 0
    total = 0
    recurrent_exact = 0
    recurrent_total = 0
    for binding in bindings:
        for law in laws:
            y0, y1 = binding.card(law)
            for observed in range(modulus):
                predicted = compile_destination(
                    binding.successor,
                    binding.zero_symbol,
                    y0,
                    y1,
                    observed,
                )
                expected = binding.destination(law, observed)
                exact += int(predicted == expected)
                total += 1

        selected = laws[: min(7, len(laws))]
        initial = tuple(reversed(range(modulus)))
        events: list[tuple[int, int, int]] = []
        expected_state = initial
        for index, law in enumerate(selected):
            identity = index % modulus
            y0, y1 = binding.card(law)
            events.append((identity, y0, y1))
            source = expected_state.index(identity)
            expected_state = pop_insert(
                expected_state,
                identity,
                binding.destination(law, source),
            )
        predicted_state = execute_compiled_program(
            initial, events, binding.successor, binding.zero_symbol
        )
        recurrent_exact += int(predicted_state == expected_state)
        recurrent_total += 1

    reference = bindings[0]
    false_successor = stride_two_successor(
        reference.successor, reference.zero_symbol
    )
    false_correct = 0
    one_witness_correct = 0
    control_total = 0
    for law in laws:
        y0, y1 = reference.card(law)
        unit_y1 = reference.successor[y0]
        for observed in range(modulus):
            expected = reference.destination(law, observed)
            false_correct += int(
                compile_destination(
                    false_successor,
                    reference.zero_symbol,
                    y0,
                    y1,
                    observed,
                )
                == expected
            )
            one_witness_correct += int(
                compile_destination(
                    reference.successor,
                    reference.zero_symbol,
                    y0,
                    unit_y1,
                    observed,
                )
                == expected
            )
            control_total += 1

    return {
        "modulus": modulus,
        "binding_count": len(bindings),
        "binding_mode": (
            "exhaustive" if modulus in EXHAUSTIVE_BINDING_MODULI else "sampled"
        ),
        "law_count": len(laws),
        "exact_destination_cells": exact,
        "destination_cells": total,
        "recurrent_programs_exact": recurrent_exact,
        "recurrent_programs": recurrent_total,
        "stride_two_accuracy": false_correct / control_total,
        "one_witness_unit_default_accuracy": one_witness_correct / control_total,
        "learned_successor_cells": modulus,
        "learned_zero_anchors": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 CPU report: {args.out}")

    source = inspect.getsource(compile_destination)
    source_forbidden = {
        token: token in source
        for token in ("%", "infer_affine_law", ".destination(", "slope *")
    }
    audits = [
        _audit_modulus(modulus)
        for modulus in PRIMARY_MODULI + (DIAGNOSTIC_MODULUS,)
    ]
    gates = {
        "all_hidden_binding_destinations_exact": all(
            row["exact_destination_cells"] == row["destination_cells"]
            for row in audits
        ),
        "all_hidden_binding_programs_exact": all(
            row["recurrent_programs_exact"] == row["recurrent_programs"]
            for row in audits
        ),
        "stride_two_control_is_not_equivalent": all(
            row["stride_two_accuracy"] < 0.8 for row in audits
        ),
        "one_witness_unit_default_is_insufficient": all(
            row["one_witness_unit_default_accuracy"] < 0.8 for row in audits
        ),
        "compiler_source_has_no_field_solver": not any(source_forbidden.values()),
        "primary_learned_cells_are_23_plus_3_anchors": (
            sum(row["learned_successor_cells"] for row in audits[:3]) == 23
            and sum(row["learned_zero_anchors"] for row in audits[:3]) == 3
        ),
    }
    report = {
        "schema": "r12_s7_learned_cayley_cpu_falsifier_v1",
        "decision": (
            "admit_s7_learned_cayley_preregistration"
            if all(gates.values())
            else "reject_s7_learned_cayley_mechanics"
        ),
        "cpu_seed": CPU_SEED,
        "gates": gates,
        "source_forbidden_tokens": source_forbidden,
        "moduli": audits,
        "resource_vector": {
            "learned_successor_cells_primary": 23,
            "learned_zero_anchors_primary": 3,
            "law_specific_parameters": 0,
            "external_arithmetic_at_inference": 0,
            "exact_equality": True,
            "maximum_nested_successor_depth": 121,
            "structural_state_mutation": "pop_insert",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "gates": gates}, sort_keys=True))
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
