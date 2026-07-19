#!/usr/bin/env python3
"""Apply frozen public-development gates to the S3 categorical executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from referential_literal_pointer_compiler import sha256_file


def load(path):
    return json.load(open(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--comp-mean", required=True)
    parser.add_argument("--comp-ordered", required=True)
    parser.add_argument("--comp-gold", required=True)
    parser.add_argument("--lexical-mean", required=True)
    parser.add_argument("--depth-mean", required=True)
    parser.add_argument("--depth-ordered", required=True)
    parser.add_argument("--depth-gold", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 assessment")
    paths = {
        "executor": args.executor,
        "comp_mean": args.comp_mean,
        "comp_ordered": args.comp_ordered,
        "comp_gold": args.comp_gold,
        "lexical_mean": args.lexical_mean,
        "depth_mean": args.depth_mean,
        "depth_ordered": args.depth_ordered,
        "depth_gold": args.depth_gold,
    }
    values = {name: load(path) for name, path in paths.items() if name != "executor"}
    comp = values["comp_mean"]
    depth = values["depth_mean"]
    ordered = values["depth_ordered"]
    gold = values["depth_gold"]
    gates = {
        "comp_mean_answer_at_least_95pct": comp["overall"]["answer_accuracy"] >= 0.95,
        "comp_mean_state_at_least_95pct": comp["overall"]["final_assignment_exact"] >= 0.95,
        "comp_mean_transitions_at_least_90pct": (
            comp["overall"]["all_transitions_exact"] >= 0.90
        ),
        "comp_mean_each_surface_answer_at_least_94pct": all(
            row["answer_accuracy"] >= 0.94 for row in comp["by_surface"].values()
        ),
        "lexical_mean_answer_at_least_60pct": (
            values["lexical_mean"]["overall"]["answer_accuracy"] >= 0.60
        ),
        "depth_mean_beats_continuous_by_10_points": (
            depth["overall"]["answer_accuracy"] >= 0.87392578125
        ),
        "depth_mean_state_at_least_85pct": (
            depth["overall"]["final_assignment_exact"] >= 0.85
        ),
        "depth_mean_transitions_at_least_65pct": (
            depth["overall"]["all_transitions_exact"] >= 0.65
        ),
        "depth8_mean_answer_at_least_80pct": (
            depth["by_depth"]["8"]["answer_accuracy"] >= 0.80
        ),
        "depth_ordered_answer_state_at_least_98pct": (
            ordered["overall"]["answer_accuracy"] >= 0.98
            and ordered["overall"]["final_assignment_exact"] >= 0.98
        ),
        "depth_gold_answer_state_at_least_99pct": (
            gold["overall"]["answer_accuracy"] >= 0.99
            and gold["overall"]["final_assignment_exact"] >= 0.99
        ),
        "all_outputs_zero_fit_and_confirmation": all(
            row.get("fit_updates") == 0 and row.get("confirmation_access") == 0
            for row in values.values()
        ),
    }
    decision = (
        "qualify_s3_categorical_register_for_fresh_confirmation"
        if all(gates.values()) else
        "reject_s3_categorical_register"
    )
    result = {
        "schema": "r12_s3_categorical_executor_assessment_v1",
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "scores": {
            name: value["overall"] for name, value in values.items()
        },
        "depth_tables": {
            name: value["by_depth"] for name, value in values.items()
            if name.startswith("depth_")
        },
        "confirmation_access": 0,
        "claim_boundary": (
            "Public S3 component assessment only. A pass authorizes an independent fresh "
            "confirmation, not autonomous reasoning, learned halt, or novelty."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
