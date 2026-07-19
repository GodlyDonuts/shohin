#!/usr/bin/env python3
"""Apply frozen S4 v5 hard-island/soft-interface development gates."""

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


def assess(treatment, baseline):
    primary = treatment["overall"]
    base = baseline["strict_autonomous"]["overall"]
    roster = treatment["roster_deranged"]["overall"]
    event = treatment["event_region_deranged"]["overall"]
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
        "query_overall_at_least_98pct": accuracy(primary, "query_exact") >= 0.98,
        "initial_roster_at_least_95pct": accuracy(primary, "initial_exact") >= 0.95,
        "program_at_least_v1_plus_1pp": (
            accuracy(primary, "program_exact") >= accuracy(base, "program_exact") + 0.01
        ),
        "roster_deranged_program_at_most_40pct": accuracy(roster, "program_exact") <= 0.40,
        "event_deranged_program_at_most_40pct": accuracy(event, "program_exact") <= 0.40,
        "gold_s3_sanity": bool(treatment["gold_event_s3_sanity"]),
        "zero_new_trainable_parameters": int(treatment["trainable_parameters"]) == 0,
        "total_parameters_below_150m": int(treatment["parameter_count"]) < 150_000_000,
        "development_access_exactly_one": int(treatment["development_access"]) == 1,
        "confirmation_access_zero": int(treatment["confirmation_access"]) == 0,
    }
    return {
        "schema": "r12_s4_hard_island_soft_interface_assessment_v5",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "decision": (
            "qualify_s4_v5_for_fresh_confirmation"
            if all(gates.values()) else
            "reject_s4_v5_fresh_development"
        ),
        "treatment": {field: accuracy(primary, field) for field in (
            "count_exact", "program_exact", "state_exact", "answer_correct", "initial_exact",
        )},
        "baseline_program_accuracy": accuracy(base, "program_exact"),
        "roster_deranged_program_accuracy": accuracy(roster, "program_exact"),
        "event_deranged_program_accuracy": accuracy(event, "program_exact"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S4 v5 assessment")
    treatment = json.load(open(args.treatment))
    baseline = json.load(open(args.baseline))
    if treatment.get("schema") != "r12_s4_hard_island_soft_interface_eval_v5":
        raise SystemExit("invalid S4 v5 treatment schema")
    if baseline.get("schema") != "r12_s4_self_delimiting_event_tape_eval_v1":
        raise SystemExit("invalid S4 v1 baseline schema")
    if treatment.get("data_sha256") != baseline.get("data_sha256"):
        raise SystemExit("S4 v5 baseline data mismatch")
    result = assess(treatment, baseline)
    result["treatment_report_sha256"] = sha256_file(args.treatment)
    result["baseline_report_sha256"] = sha256_file(args.baseline)
    result["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
