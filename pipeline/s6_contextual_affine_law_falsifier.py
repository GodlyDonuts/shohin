#!/usr/bin/env python3
"""Exhaustively falsify the exact mechanics claimed by S6 before neural work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from s6_contextual_affine_law import (
    ADMITTED_MODULI,
    ALL_AUDIT_MODULI,
    apply_law,
    enumerate_laws,
    execute_program,
    infer_affine_law,
    one_witness_candidates,
    repaired_split_laws,
    treatment_input,
)


FORBIDDEN_TREATMENT_FIELDS = {
    "slope",
    "intercept",
    "a",
    "b",
    "law_id",
    "identity",
    "answer",
    "final_state",
}


def _find_order_witness(modulus: int) -> dict[str, object] | None:
    initial = tuple(range(modulus))
    laws = enumerate_laws(modulus)
    for first in laws:
        for second in laws:
            if first == second:
                continue
            for first_identity in range(modulus):
                for second_identity in range(modulus):
                    forward = execute_program(
                        initial,
                        ((first_identity, first), (second_identity, second)),
                    )
                    reverse = execute_program(
                        initial,
                        ((second_identity, second), (first_identity, first)),
                    )
                    if forward == reverse:
                        continue
                    query = next(
                        index
                        for index, (left, right) in enumerate(zip(forward, reverse))
                        if left != right
                    )
                    return {
                        "first_law": first.key,
                        "second_law": second.key,
                        "first_identity": first_identity,
                        "second_identity": second_identity,
                        "forward_state": list(forward),
                        "reverse_state": list(reverse),
                        "separating_query": query,
                        "forward_answer": forward[query],
                        "reverse_answer": reverse[query],
                    }
    return None


def audit_modulus(modulus: int) -> dict[str, object]:
    laws = enumerate_laws(modulus)
    cards = {law.card for law in laws}
    reconstructed_cells = 0
    pop_insert_cells = 0
    one_witness_sizes: set[int] = set()

    for law in laws:
        inferred = infer_affine_law(modulus, *law.card)
        if inferred != law:
            raise AssertionError(f"law reconstruction failed for {law.key}")
        one_witness_sizes.add(len(one_witness_candidates(modulus, law.card[0])))
        for position in range(modulus):
            if inferred.destination(position) != law.destination(position):
                raise AssertionError(f"destination closure failed for {law.key}")
            reconstructed_cells += 1
        initial = tuple(range(modulus))
        for identity in range(modulus):
            updated = apply_law(initial, identity, law)
            if set(updated) != set(initial):
                raise AssertionError(f"state closure failed for {law.key}")
            expected = law.destination(identity)
            if updated.index(identity) != expected:
                raise AssertionError(f"target destination failed for {law.key}")
            pop_insert_cells += 1

    split, promotions = repaired_split_laws(modulus)
    split_sets = {
        name: {law.key for law in values} for name, values in split.items()
    }
    if any(not values for values in split_sets.values()):
        raise AssertionError(f"empty S6 law split at modulus {modulus}")
    if split_sets["train"] & split_sets["development"]:
        raise AssertionError("train/development law overlap")
    if split_sets["train"] & split_sets["confirmation"]:
        raise AssertionError("train/confirmation law overlap")
    if split_sets["development"] & split_sets["confirmation"]:
        raise AssertionError("development/confirmation law overlap")

    train_laws = split["train"]
    train_card_y0 = {law.card[0] for law in train_laws}
    train_card_y1 = {law.card[1] for law in train_laws}
    train_destinations = {
        law.destination(position)
        for law in train_laws
        for position in range(modulus)
    }
    complete_coordinate_coverage = all(
        values == set(range(modulus))
        for values in (train_card_y0, train_card_y1, train_destinations)
    )
    if not complete_coordinate_coverage:
        raise AssertionError(f"incomplete training coordinate coverage at {modulus}")

    visible_fields = set(treatment_input(laws[0], 0))
    if visible_fields & FORBIDDEN_TREATMENT_FIELDS:
        raise AssertionError("forbidden field leaked into S6 treatment input")
    expected_fields = {"modulus", "card_y0", "card_y1", "current_location"}
    if visible_fields != expected_fields:
        raise AssertionError("unexpected S6 treatment input schema")

    order_witness = _find_order_witness(modulus)
    if order_witness is None:
        raise AssertionError(f"no noncommutative order witness at modulus {modulus}")

    return {
        "modulus": modulus,
        "law_count": len(laws),
        "expected_law_count": modulus * (modulus - 1),
        "unique_card_count": len(cards),
        "one_witness_candidate_counts": sorted(one_witness_sizes),
        "expected_one_witness_candidates": modulus - 1,
        "reconstructed_destination_cells": reconstructed_cells,
        "pop_insert_closure_cells": pop_insert_cells,
        "split_counts": {name: len(values) for name, values in split.items()},
        "coverage_promotions": list(promotions),
        "complete_training_coordinate_coverage": complete_coordinate_coverage,
        "treatment_input_fields": sorted(visible_fields),
        "order_witness": order_witness,
        "memorization_table_entries": modulus * modulus * (modulus - 1),
        "affine_card_field_elements_per_law": 2,
    }


def build_report() -> dict[str, object]:
    audits = [audit_modulus(modulus) for modulus in ALL_AUDIT_MODULI]
    gates = {
        "all_laws_have_unique_cards": all(
            row["law_count"] == row["unique_card_count"] for row in audits
        ),
        "one_witness_is_exactly_m_minus_one_ambiguous": all(
            row["one_witness_candidate_counts"]
            == [row["expected_one_witness_candidates"]]
            for row in audits
        ),
        "all_destinations_reconstruct": all(
            row["reconstructed_destination_cells"]
            == row["law_count"] * row["modulus"]
            for row in audits
        ),
        "all_pop_insert_updates_close": all(
            row["pop_insert_closure_cells"] == row["law_count"] * row["modulus"]
            for row in audits
        ),
        "all_scales_have_order_witness": all(
            row["order_witness"] is not None for row in audits
        ),
        "all_admitted_splits_nonempty_and_covered": all(
            row["complete_training_coordinate_coverage"]
            and all(row["split_counts"][name] > 0 for name in ("train", "development", "confirmation"))
            for row in audits
            if row["modulus"] in ADMITTED_MODULI
        ),
        "treatment_schema_has_no_forbidden_fields": all(
            not (set(row["treatment_input_fields"]) & FORBIDDEN_TREATMENT_FIELDS)
            for row in audits
        ),
    }
    return {
        "schema": "r12_s6_contextual_affine_law_cpu_falsifier_v1",
        "decision": "pass_s6_cpu_mechanics" if all(gates.values()) else "reject_s6_cpu_mechanics",
        "gates": gates,
        "moduli": audits,
        "resource_vector": {
            "parameters": "no neural parameters in falsifier",
            "retained_bits": "categorical list state plus two field elements per active law card",
            "precision": "exact Python integers",
            "source_bytes": 0,
            "training_examples": 0,
            "oracle_calls": 0,
            "training_flops": 0,
            "inference_flops": "host exact mechanics only; not a neural score",
            "sequential_depth": 2,
            "external_memory": "none beyond explicit categorical state and law card",
            "external_execution": "yes for falsifier and favorable ceiling; forbidden as neural treatment evidence",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S6 falsifier report: {args.out}")
    report = build_report()
    if report["decision"] != "pass_s6_cpu_mechanics":
        raise SystemExit("S6 CPU falsifier failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
