#!/usr/bin/env python3
"""Apply frozen S9 development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from s9_occurrence_quotient_compiler import sha256_file


def assess(evaluation):
    rows = int(evaluation["rows"])
    graph = evaluation["graph"]
    arms = evaluation["arms"]
    span = evaluation["span"]
    invariance = evaluation["invariance"]
    treatment = arms["treatment"]
    no_class = arms["no_class_message"]
    gates = {
        "span_f1_at_least_98pct": span["f1"] >= 0.98,
        "class_membership_exact_at_least_95pct": span["class_exact_accuracy"] >= 0.95,
        "complete_relation_tuple_at_least_90pct": graph["exact_accuracy"] >= 0.90,
        "valid_graph_at_least_90pct": graph["valid_accuracy"] >= 0.90,
        "exact_graph_at_least_85pct": graph["exact_accuracy"] >= 0.85,
        "state_at_least_80pct": treatment["state_accuracy"] >= 0.80,
        "answer_at_least_85pct": treatment["answer_accuracy"] >= 0.85,
        "every_depth_state_at_least_70pct": min(
            value["accuracy"] for value in treatment["depth"].values()
        ) >= 0.70,
        "exact_graph_plus_20pp_over_s8_1": graph["exact_accuracy"] >= 0.4509765625,
        "exact_graph_plus_5pp_over_no_class": (
            graph["exact_accuracy"] - graph["no_class_exact_accuracy"] >= 0.05
        ),
        "shuffled_exact_below_10pct": graph["shuffled_exact_accuracy"] < 0.10,
        "class_reindex_bit_identical": (
            invariance["eligible"] > 0
            and invariance["class_reindex"] == invariance["eligible"]
        ),
        "relation_storage_reindex_bit_identical": (
            invariance["eligible"] > 0
            and invariance["relation_storage_reindex"] == invariance["eligible"]
        ),
        "operation_nonce_bit_identical": (
            invariance["nonce_eligible"] == graph["valid"]
            and invariance["nonce_identical"] == invariance["nonce_eligible"]
        ),
        "reversed_links_drop_at_least_40pp": (
            treatment["state_accuracy"] - arms["reversed_links"]["state_accuracy"] >= 0.40
        ),
        "swapped_cards_drop_at_least_50pp": (
            treatment["state_accuracy"] - arms["deranged_cards"]["state_accuracy"] >= 0.50
        ),
        "one_witness_drop_at_least_30pp": (
            treatment["state_accuracy"] - arms["one_witness"]["state_accuracy"] >= 0.30
        ),
        "state_reset_drop_at_least_20pp": (
            treatment["state_accuracy"] - arms["state_reset"]["state_accuracy"] >= 0.20
        ),
        "early_nil_drop_at_least_30pp": (
            treatment["state_accuracy"] - arms["early_nil"]["state_accuracy"] >= 0.30
        ),
        "complete_system_below_150m": evaluation["parameters"]["complete_system"] < 150_000_000,
        "one_development_zero_confirmation_access": (
            evaluation["development_accesses"] == 1
            and evaluation["confirmation_accesses"] == 0
        ),
        "all_rows_scored": treatment["total"] == rows == 2048,
    }
    return {
        "schema": "r12_s9_occurrence_quotient_development_assessment_v1",
        "decision": (
            "qualify_s9_occurrence_quotient_for_fresh_confirmation"
            if all(gates.values())
            else "reject_s9_occurrence_quotient_v1"
        ),
        "gates": gates,
        "scores": {
            "span": span,
            "graph": graph,
            "treatment": treatment,
            "no_class": no_class,
            "shuffled_exact": graph["shuffled_exact_accuracy"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9 assessment: {args.out}")
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("schema") != "r12_s9_occurrence_quotient_development_evaluation_v1":
        raise SystemExit("unexpected S9 evaluation schema")
    result = assess(evaluation)
    result["evaluation_sha256"] = sha256_file(args.evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
