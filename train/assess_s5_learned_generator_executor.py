#!/usr/bin/env python3
"""Apply frozen S5 learned-generator development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_delimiting_event_tape import sha256_file


def accuracy(summary, field):
    return float(summary[field]["accuracy"])


def assess(result):
    arms = result["arms"]
    learned = arms["learned"]["overall"]
    host = arms["host_exact"]["overall"]
    shuffled = arms["shuffled_law"]["overall"]
    direction = arms["direction_rotated"]["overall"]
    reset = arms["state_reset"]["overall"]
    learned_state = accuracy(learned, "state_exact")
    learned_answer = accuracy(learned, "answer_correct")
    gates = {
        "parser_parity_exact": result["parser_parity"]["accuracy"] == 1.0,
        "unit_generator_closure_exact": (
            result["closure"]["treatment"]["amount_one"]["accuracy"] == 1.0
        ),
        "unseen_amount_two_closure_exact": (
            result["closure"]["treatment"]["amount_two_unseen"]["accuracy"] == 1.0
        ),
        "program_overall_at_least_95pct": accuracy(learned, "program_exact") >= 0.95,
        "learned_state_at_least_95pct": learned_state >= 0.95,
        "learned_answer_at_least_95pct": learned_answer >= 0.95,
        "every_depth_state_at_least_93pct": all(
            accuracy(arms["learned"]["by_depth"][str(depth)], "state_exact") >= 0.93
            for depth in range(3, 9)
        ),
        "amount_two_rows_state_at_least_95pct": (
            accuracy(arms["learned"]["amount_two_rows"], "state_exact") >= 0.95
        ),
        "learned_matches_host_state_within_point_one_pp": (
            learned_state >= accuracy(host, "state_exact") - 0.001
        ),
        "learned_matches_host_answer_within_point_one_pp": (
            learned_answer >= accuracy(host, "answer_correct") - 0.001
        ),
        "shuffled_law_state_drop_at_least_40pp": (
            learned_state - accuracy(shuffled, "state_exact") >= 0.40
        ),
        "direction_rotation_state_drop_at_least_40pp": (
            learned_state - accuracy(direction, "state_exact") >= 0.40
        ),
        "state_reset_state_drop_at_least_20pp": (
            learned_state - accuracy(reset, "state_exact") >= 0.20
        ),
        "no_amount_two_training": int(result["training_amount_two_examples"]) == 0,
        "no_recurrent_training": int(result["training_recurrent_examples"]) == 0,
        "generator_below_100k_parameters": int(result["generator_parameters"]) < 100_000,
        "total_parameters_below_150m": int(result["parameter_count"]) < 150_000_000,
        "development_access_exactly_one": int(result["development_access"]) == 1,
        "confirmation_access_zero": int(result["confirmation_access"]) == 0,
    }
    return {
        "schema": "r12_s5_learned_generator_executor_assessment_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "decision": (
            "qualify_s5_learned_generator_for_fresh_confirmation"
            if all(gates.values()) else "reject_s5_learned_generator_development"
        ),
        "scores": {
            name: {
                "program": accuracy(values["overall"], "program_exact"),
                "state": accuracy(values["overall"], "state_exact"),
                "answer": accuracy(values["overall"], "answer_correct"),
            }
            for name, values in arms.items()
        },
        "closure": result["closure"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S5 assessment")
    result = json.load(open(args.evaluation))
    if result.get("schema") != "r12_s5_learned_generator_executor_eval_v1":
        raise SystemExit("invalid S5 evaluation schema")
    assessment = assess(result)
    assessment["evaluation_sha256"] = sha256_file(args.evaluation)
    assessment["assessor_sha256"] = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(assessment, sort_keys=True))


if __name__ == "__main__":
    main()
