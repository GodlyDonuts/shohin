#!/usr/bin/env python3
"""Apply the frozen S4 v1.1 public-development qualification gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HELD_OUT_DEPTHS = (5, 6, 7, 8)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy(summary, field):
    return float(summary[field]["accuracy"])


def assess(treatment, shuffled):
    primary = treatment["pointer_anchored_v1_1"]
    control = shuffled["pointer_anchored_v1_1"]
    original = treatment["strict_autonomous"]["overall"]
    gold_count = treatment["gold_count_control"]["overall"]
    gates = {
        "treatment_count_overall_at_least_98pct": accuracy(primary["overall"], "count_exact") >= 0.98,
        "treatment_count_every_depth_at_least_95pct": all(
            accuracy(primary["by_depth"][str(depth)], "count_exact") >= 0.95
            for depth in range(3, 9)
        ),
        "treatment_program_overall_at_least_95pct": accuracy(primary["overall"], "program_exact") >= 0.95,
        "treatment_program_heldout_depths_at_least_90pct": all(
            accuracy(primary["by_depth"][str(depth)], "program_exact") >= 0.90
            for depth in HELD_OUT_DEPTHS
        ),
        "treatment_answer_overall_at_least_95pct": accuracy(primary["overall"], "answer_correct") >= 0.95,
        "treatment_answer_depth8_at_least_90pct": (
            accuracy(primary["by_depth"]["8"], "answer_correct") >= 0.90
        ),
        "gold_count_rescue_below_2_points": (
            accuracy(gold_count, "program_exact") - accuracy(original, "program_exact") < 0.02
        ),
        "shuffled_program_at_most_40pct": accuracy(control["overall"], "program_exact") <= 0.40,
        "locked_s3_gold_sanity": bool(treatment["gold_event_s3_sanity"]),
        "total_parameters_below_150m": int(treatment["parameter_count"]) < 150_000_000,
        "development_access_exactly_one": int(treatment["development_access"]) == 1,
        "confirmation_access_zero": int(treatment["confirmation_access"]) == 0,
    }
    return {
        "schema": "r12_s4_pointer_anchored_assessment_v1",
        "decision": (
            "qualify_s4_v1_1_for_fresh_confirmation"
            if all(gates.values())
            else "reject_s4_v1_1_public_development"
        ),
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "treatment": {
            "count_accuracy": accuracy(primary["overall"], "count_exact"),
            "program_accuracy": accuracy(primary["overall"], "program_exact"),
            "answer_accuracy": accuracy(primary["overall"], "answer_correct"),
            "program_by_depth": {
                depth: accuracy(primary["by_depth"][depth], "program_exact")
                for depth in sorted(primary["by_depth"])
            },
        },
        "shuffled_program_accuracy": accuracy(control["overall"], "program_exact"),
        "gold_count_rescue_points": 100.0 * (
            accuracy(gold_count, "program_exact") - accuracy(original, "program_exact")
        ),
        "claim_boundary": (
            "Public known-atom variable-length event parsing into a locked categorical executor. "
            "No fresh confirmation, unseen semantics, planning, open-language halt, broad reasoning, "
            "novelty, or benchmark claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S4 assessment")
    treatment = json.load(open(args.treatment))
    shuffled = json.load(open(args.shuffled))
    for label, report in (("treatment", treatment), ("shuffled", shuffled)):
        if "pointer_anchored_v1_1" not in report:
            raise SystemExit("{} report lacks pointer-anchored result".format(label))
    result = assess(treatment, shuffled)
    result["treatment_report_sha256"] = sha256_file(args.treatment)
    result["shuffled_report_sha256"] = sha256_file(args.shuffled)
    result["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
