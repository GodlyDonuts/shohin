#!/usr/bin/env python3
"""Apply the frozen S4 v2 fresh-development gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy(summary, field):
    return float(summary[field]["accuracy"])


def assess(treatment, shuffled):
    primary = treatment["overall"]
    control = shuffled["overall"]
    gates = {
        "count_overall_at_least_98pct": accuracy(primary, "count_exact") >= 0.98,
        "count_every_depth_at_least_95pct": all(
            accuracy(treatment["by_depth"][str(depth)], "count_exact") >= 0.95
            for depth in range(3, 9)
        ),
        "program_overall_at_least_95pct": accuracy(primary, "program_exact") >= 0.95,
        "program_depths_5_to_8_at_least_90pct": all(
            accuracy(treatment["by_depth"][str(depth)], "program_exact") >= 0.90
            for depth in range(5, 9)
        ),
        "state_overall_at_least_95pct": accuracy(primary, "state_exact") >= 0.95,
        "answer_overall_at_least_95pct": accuracy(primary, "answer_correct") >= 0.95,
        "state_depth8_at_least_90pct": accuracy(treatment["by_depth"]["8"], "state_exact") >= 0.90,
        "answer_depth8_at_least_90pct": accuracy(treatment["by_depth"]["8"], "answer_correct") >= 0.90,
        "initial_roster_at_least_95pct": accuracy(primary, "initial_exact") >= 0.95,
        "shuffled_program_at_most_40pct": accuracy(control, "program_exact") <= 0.40,
        "gold_s3_sanity": bool(treatment["gold_event_s3_sanity"]),
        "total_parameters_below_150m": int(treatment["parameter_count"]) < 150_000_000,
        "development_access_exactly_one": int(treatment["development_access"]) == 1,
        "confirmation_access_zero": int(treatment["confirmation_access"]) == 0,
    }
    return {
        "schema": "r12_s4_event_relative_pointer_assessment_v2",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "decision": (
            "qualify_s4_v2_for_fresh_confirmation"
            if all(gates.values()) else
            "reject_s4_v2_fresh_development"
        ),
        "treatment": {
            field: accuracy(primary, field)
            for field in ("count_exact", "program_exact", "state_exact", "answer_correct", "initial_exact")
        },
        "shuffled_program_accuracy": accuracy(control, "program_exact"),
        "claim_boundary": (
            "Fresh-development known-atom event-relative parsing only. No confirmation, unseen "
            "semantics, planning, free-form reasoning, benchmark, or novelty claim."
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
        raise SystemExit("refusing existing S4 v2 assessment")
    treatment = json.load(open(args.treatment))
    shuffled = json.load(open(args.shuffled))
    result = assess(treatment, shuffled)
    result["treatment_report_sha256"] = sha256_file(args.treatment)
    result["shuffled_report_sha256"] = sha256_file(args.shuffled)
    result["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
