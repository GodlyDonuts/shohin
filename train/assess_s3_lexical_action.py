#!/usr/bin/env python3
"""Apply frozen gates to lexical-direction plus closure-complete S3 execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from referential_literal_pointer_compiler import sha256_file


def load(path):
    value = json.load(open(path))
    if value.get("action_protocol") != "closed_s3_v1_2":
        raise SystemExit("lexical S3 assessment requires closed action")
    if value.get("kind_protocol") != "training_lexicon_v1":
        raise SystemExit("lexical S3 assessment requires training lexicon")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--lexicon", required=True)
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
        raise SystemExit("refusing existing lexical-S3 assessment")
    paths = {
        "executor": args.executor,
        "lexicon": args.lexicon,
        "comp_mean": args.comp_mean,
        "comp_ordered": args.comp_ordered,
        "comp_gold": args.comp_gold,
        "lexical_mean": args.lexical_mean,
        "depth_mean": args.depth_mean,
        "depth_ordered": args.depth_ordered,
        "depth_gold": args.depth_gold,
    }
    values = {
        name: load(path) for name, path in paths.items()
        if name not in {"executor", "lexicon"}
    }
    lexicon = json.load(open(args.lexicon))
    if not lexicon.get("all_gates_pass"):
        raise SystemExit("lexical-S3 lexicon failed its own gates")
    comp = values["comp_mean"]
    lexical = values["lexical_mean"]["overall"]
    depth = values["depth_mean"]
    ordered = values["depth_ordered"]["overall"]
    gold = values["depth_gold"]["overall"]
    gates = {
        "lexicon_training_only_and_zero_fit": (
            lexicon.get("fit_updates") == 0
            and lexicon.get("development_access") == 0
            and lexicon.get("confirmation_access") == 0
        ),
        "comp_mean_answer_state_transitions_at_least_95pct": (
            comp["overall"]["answer_accuracy"] >= 0.95
            and comp["overall"]["final_assignment_exact"] >= 0.95
            and comp["overall"]["all_transitions_exact"] >= 0.95
        ),
        "comp_mean_each_surface_answer_at_least_94pct": all(
            row["answer_accuracy"] >= 0.94 for row in comp["by_surface"].values()
        ),
        "lexical_ood_fallback_coverage_at_most_5pct": (
            lexical["kind_lexical_coverage"] <= 0.05
        ),
        "lexical_ood_answer_at_least_closed_baseline": lexical["answer_accuracy"] >= 0.75,
        "depth_lexical_coverage_at_least_995permille": (
            depth["overall"]["kind_lexical_coverage"] >= 0.995
        ),
        "depth_kind_and_amount_at_least_995permille": (
            depth["overall"]["kind_accuracy"] >= 0.995
            and depth["overall"]["amount_accuracy"] >= 0.995
        ),
        "depth_mean_answer_at_least_90pct": depth["overall"]["answer_accuracy"] >= 0.90,
        "depth_mean_state_at_least_88pct": (
            depth["overall"]["final_assignment_exact"] >= 0.88
        ),
        "depth_mean_transitions_at_least_80pct": (
            depth["overall"]["all_transitions_exact"] >= 0.80
        ),
        "depth8_mean_answer_at_least_85pct": (
            depth["by_depth"]["8"]["answer_accuracy"] >= 0.85
        ),
        "depth_ordered_answer_state_transitions_at_least_98pct": (
            ordered["answer_accuracy"] >= 0.98
            and ordered["final_assignment_exact"] >= 0.98
            and ordered["all_transitions_exact"] >= 0.98
        ),
        "depth_gold_answer_at_least_985permille": gold["answer_accuracy"] >= 0.985,
        "depth_gold_state_and_transitions_exact": (
            gold["final_assignment_exact"] == 1.0
            and gold["all_transitions_exact"] == 1.0
        ),
        "all_outputs_zero_fit_and_confirmation": all(
            row.get("fit_updates") == 0 and row.get("confirmation_access") == 0
            for row in values.values()
        ),
    }
    decision = (
        "qualify_lexical_closed_s3_v1_3_for_fresh_confirmation"
        if all(gates.values()) else
        "reject_lexical_closed_s3_v1_3_for_confirmation"
    )
    result = {
        "schema": "r12_s3_lexical_closed_action_assessment_v1",
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "scores": {name: value["overall"] for name, value in values.items()},
        "depth_tables": {
            name: value["by_depth"] for name, value in values.items()
            if name.startswith("depth_")
        },
        "fit_updates": 0,
        "confirmation_access": 0,
        "claim_boundary": (
            "Public known-atom lexical compiler plus exact S3 component. A pass authorizes one "
            "fresh confirmation, not unseen-lexeme generalization, planning, learned halt, or novelty."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
