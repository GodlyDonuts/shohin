#!/usr/bin/env python3
"""Locked fresh-board comparator for the R5 text-only argument graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REGIMES = ("fit_iid", "depth_ood", "language_ood", "full_ood")


def combined_accuracy(report, regimes, metric):
    correct_key = metric.replace("accuracy", "correct")
    correct = sum(report["summary"][regime][correct_key] for regime in regimes)
    cases = sum(report["summary"][regime]["cases"] for regime in regimes)
    return correct / cases


def combined_program_accuracy(report, regimes):
    correct = sum(report["summary"][regime]["program_exact"] for regime in regimes)
    cases = sum(report["summary"][regime]["cases"] for regime in regimes)
    return correct / cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--argument-graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    raw = json.load(open(args.raw))
    candidate = json.load(open(args.argument_graph))
    if raw.get("audit") != "referential_slot_microcode_eval_v4":
        raise SystemExit("invalid raw pointer report")
    if candidate.get("audit") != "referential_argument_graph_eval_v5":
        raise SystemExit("invalid argument-graph report")

    matched_fields = (
        "base_sha256", "adapter_sha256", "data_sha256", "admission_sha256",
        "label_admission_sha256", "evaluation_label_admission_sha256",
    )
    mismatches = {
        field: [raw.get(field), candidate.get(field)]
        for field in matched_fields if raw.get(field) != candidate.get(field)
    }
    if raw.get("adapter_metadata") != candidate.get("adapter_metadata"):
        mismatches["adapter_metadata"] = "reports differ"

    fresh = ("language_ood", "full_ood")
    raw_fresh_answer = combined_accuracy(raw, fresh, "answer_accuracy")
    candidate_fresh_answer = combined_accuracy(candidate, fresh, "answer_accuracy")
    raw_fresh_program = combined_program_accuracy(raw, fresh)
    candidate_fresh_program = combined_program_accuracy(candidate, fresh)
    graph = candidate.get("argument_graph_diagnostics", {})
    fresh_arity_correct = sum(graph.get(regime, {}).get("arity_correct", 0) for regime in fresh)
    fresh_arity_total = sum(graph.get(regime, {}).get("operations", 0) for regime in fresh)
    fresh_arity_accuracy = fresh_arity_correct / fresh_arity_total if fresh_arity_total else 0.0
    operation_coverage = sorted(set().union(*(
        set(graph.get(regime, {}).get("target_operation_kinds", [])) for regime in fresh
    )))
    query_coverage = sorted(set().union(*(
        set(graph.get(regime, {}).get("target_query_kinds", [])) for regime in fresh
    )))
    shape = {regime: candidate["summary"].get(regime, {}).get("cases") for regime in REGIMES}

    gates = {
        "matched_read_only_reports": not mismatches,
        "pointer_adapter": candidate["adapter_metadata"].get("role_mode") == "pointer",
        "frozen_threshold_exactly_0_80": candidate.get("argument_graph", {}).get("threshold") == 0.80,
        "fresh_board_shape_256_language_192_full": shape == {
            "fit_iid": 256, "depth_ood": 192, "language_ood": 256, "full_ood": 192,
        },
        "candidate_original_absolute_gates": all(candidate.get("gates", {}).values()),
        "fresh_language_answer_at_least_0_70": candidate["summary"]["language_ood"]["answer_accuracy"] >= 0.70,
        "fresh_full_answer_at_least_0_55": candidate["summary"]["full_ood"]["answer_accuracy"] >= 0.55,
        "fresh_answer_gain_over_raw_at_least_0_15": candidate_fresh_answer - raw_fresh_answer >= 0.15,
        "fresh_exact_program_gain_over_raw_at_least_0_10": candidate_fresh_program - raw_fresh_program >= 0.10,
        "fresh_argument_arity_accuracy_at_least_0_95": fresh_arity_accuracy >= 0.95,
        "fresh_operation_kind_coverage_all_five": operation_coverage == list(range(5)),
        "fresh_query_kind_coverage_all_three": query_coverage == list(range(3)),
        "fit_answer_regression_at_most_0_10": (
            raw["summary"]["fit_iid"]["answer_accuracy"]
            - candidate["summary"]["fit_iid"]["answer_accuracy"] <= 0.10
        ),
        "depth_answer_regression_at_most_0_10": (
            raw["summary"]["depth_ood"]["answer_accuracy"]
            - candidate["summary"]["depth_ood"]["answer_accuracy"] <= 0.10
        ),
    }
    advance = all(gates.values())
    result = {
        "audit": "referential_argument_graph_comparison_v5",
        "raw": str(Path(args.raw).resolve()),
        "argument_graph": str(Path(args.argument_graph).resolve()),
        "frozen_before_fresh_scoring": True,
        "raw_fresh_answer_accuracy": raw_fresh_answer,
        "argument_graph_fresh_answer_accuracy": candidate_fresh_answer,
        "raw_fresh_program_exact_accuracy": raw_fresh_program,
        "argument_graph_fresh_program_exact_accuracy": candidate_fresh_program,
        "fresh_argument_arity_accuracy": fresh_arity_accuracy,
        "fresh_target_operation_kinds": operation_coverage,
        "fresh_target_query_kinds": query_coverage,
        "mismatches": mismatches,
        "gates": gates,
        "advance_to_future_effect_operator_fit": advance,
        "decision": (
            "advance_argument_graph_r5_to_operator_fit"
            if advance else "reject_argument_graph_r5"
        ),
        "claim_boundary": (
            "A pass establishes fresh text-only argument-graph recovery as a causal compiler aid. "
            "It does not establish autonomous or broadly transferable reasoning."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
