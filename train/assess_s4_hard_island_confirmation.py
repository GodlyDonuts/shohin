#!/usr/bin/env python3
"""Apply unchanged S4 v5 gates to its sole confirmation read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assess_s4_hard_island_soft_interface import assess, sha256_file


def assess_confirmation(treatment, baseline):
    result = assess(treatment, baseline)
    result["schema"] = "r12_s4_hard_island_confirmation_assessment_v5"
    result["gates"].pop("development_access_exactly_one")
    result["gates"].pop("confirmation_access_zero")
    result["gates"]["development_access_zero"] = int(treatment["development_access"]) == 0
    result["gates"]["confirmation_access_exactly_one"] = int(treatment["confirmation_access"]) == 1
    result["all_gates_pass"] = all(result["gates"].values())
    result["decision"] = (
        "confirm_s4_v5_hard_island_soft_interface"
        if result["all_gates_pass"] else
        "reject_s4_v5_confirmation"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S4 v5 confirmation assessment")
    treatment = json.load(open(args.treatment))
    baseline = json.load(open(args.baseline))
    if treatment.get("schema") != "r12_s4_hard_island_soft_interface_confirmation_eval_v5":
        raise SystemExit("invalid S4 v5 confirmation treatment")
    if baseline.get("schema") != "r12_s4_self_delimiting_event_tape_eval_v1":
        raise SystemExit("invalid confirmation baseline")
    if treatment.get("data_sha256") != baseline.get("data_sha256"):
        raise SystemExit("confirmation arm data mismatch")
    result = assess_confirmation(treatment, baseline)
    result["treatment_report_sha256"] = sha256_file(args.treatment)
    result["baseline_report_sha256"] = sha256_file(args.baseline)
    result["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
