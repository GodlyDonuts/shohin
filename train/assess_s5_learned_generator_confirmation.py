#!/usr/bin/env python3
"""Apply unchanged S5 gates to the sole confirmation read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assess_s5_learned_generator_executor import assess
from self_delimiting_event_tape import sha256_file


def assess_confirmation(result):
    assessment = assess(result)
    assessment["schema"] = "r12_s5_learned_generator_confirmation_assessment_v1"
    assessment["gates"].pop("development_access_exactly_one")
    assessment["gates"].pop("confirmation_access_zero")
    assessment["gates"]["development_access_zero"] = (
        int(result["development_access"]) == 0
    )
    assessment["gates"]["confirmation_access_exactly_one"] = (
        int(result["confirmation_access"]) == 1
    )
    assessment["all_gates_pass"] = all(assessment["gates"].values())
    assessment["decision"] = (
        "confirm_s5_learned_generator_factored_execution"
        if assessment["all_gates_pass"] else "reject_s5_learned_generator_confirmation"
    )
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S5 confirmation assessment")
    result = json.load(open(args.evaluation))
    if result.get("schema") != "r12_s5_learned_generator_executor_eval_v1":
        raise SystemExit("invalid S5 confirmation evaluation schema")
    if result.get("board_role") != "confirmation":
        raise SystemExit("S5 evaluation is not a confirmation read")
    assessment = assess_confirmation(result)
    assessment["evaluation_sha256"] = sha256_file(args.evaluation)
    assessment["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(assessment, sort_keys=True))


if __name__ == "__main__":
    main()
