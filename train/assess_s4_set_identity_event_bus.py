#!/usr/bin/env python3
"""Apply the frozen S4 v3 fresh-development gates."""

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


def baseline_overall(report):
    if report.get("schema") != "r12_s4_self_delimiting_event_tape_eval_v1":
        raise ValueError("invalid S4 v1 baseline schema")
    return report["strict_autonomous"]["overall"]


def assess(treatment, shuffled, baseline):
    primary = treatment["overall"]
    control = shuffled["overall"]
    base = baseline_overall(baseline)
    deranged = treatment["roster_deranged"]["overall"]
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
        "initial_roster_at_least_95pct": accuracy(primary, "initial_exact") >= 0.95,
        "program_at_least_v1_plus_1pp": (
            accuracy(primary, "program_exact") >= accuracy(base, "program_exact") + 0.01
        ),
        "shuffled_program_at_most_40pct": accuracy(control, "program_exact") <= 0.40,
        "roster_deranged_program_at_most_40pct": (
            accuracy(deranged, "program_exact") <= 0.40
        ),
        "gold_s3_sanity": bool(treatment["gold_event_s3_sanity"]),
        "total_parameters_below_150m": int(treatment["parameter_count"]) < 150_000_000,
        "development_access_exactly_one": int(treatment["development_access"]) == 1,
        "confirmation_access_zero": int(treatment["confirmation_access"]) == 0,
    }
    return {
        "schema": "r12_s4_set_identity_event_bus_assessment_v3",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "decision": (
            "qualify_s4_v3_for_fresh_confirmation"
            if all(gates.values()) else
            "reject_s4_v3_fresh_development"
        ),
        "treatment": {
            field: accuracy(primary, field)
            for field in (
                "count_exact", "program_exact", "state_exact", "answer_correct", "initial_exact",
            )
        },
        "baseline_program_accuracy": accuracy(base, "program_exact"),
        "shuffled_program_accuracy": accuracy(control, "program_exact"),
        "roster_deranged_program_accuracy": accuracy(deranged, "program_exact"),
        "claim_boundary": (
            "Fresh-development known-atom set-identity parsing only. No confirmation, unseen "
            "semantics, planning, learned halt, free-form reasoning, benchmark, or novelty claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S4 v3 assessment")
    treatment = json.load(open(args.treatment))
    shuffled = json.load(open(args.shuffled))
    baseline = json.load(open(args.baseline))
    if treatment.get("data_sha256") != shuffled.get("data_sha256"):
        raise SystemExit("S4 v3 arm data mismatch")
    if treatment.get("data_sha256") != baseline.get("data_sha256"):
        raise SystemExit("S4 v3 baseline data mismatch")
    result = assess(treatment, shuffled, baseline)
    result["treatment_report_sha256"] = sha256_file(args.treatment)
    result["shuffled_report_sha256"] = sha256_file(args.shuffled)
    result["baseline_report_sha256"] = sha256_file(args.baseline)
    result["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
