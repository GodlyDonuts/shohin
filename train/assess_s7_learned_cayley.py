#!/usr/bin/env python3
"""Apply the immutable S7 learned-Cayley development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(evaluation: dict[str, object]) -> dict[str, object]:
    arms = evaluation["arms"]
    treatment = arms["treatment"]
    host = arms["host"]
    ordinary = arms["ordinary_transformer"]
    false_generator = arms["stride_two_generator"]
    deranged = arms["deranged_card"]
    one_witness = arms["one_witness"]
    reset = arms["state_reset"]
    fit = evaluation["fit"]
    atomic = evaluation["atomic_development"]

    gates = {
        "successor_and_zero_only_training_contract": (
            "23 successor cells plus three zero anchors"
            in str(evaluation["training_contract"])
        ),
        "treatment_successor_fit_exact": (
            fit["treatment"]["successor_accuracy"] == 1.0
            and fit["treatment"]["zero_accuracy"] == 1.0
        ),
        "false_generator_fit_exact": (
            fit["false_generator"]["successor_accuracy"] == 1.0
            and fit["false_generator"]["zero_accuracy"] == 1.0
        ),
        "ordinary_transformer_atomic_fit_at_least_99pct": (
            fit["ordinary_transformer"]["atomic_train_accuracy"] >= 0.99
        ),
        "treatment_atomic_development_at_least_99pct": (
            atomic["treatment"]["accuracy"] >= 0.99
        ),
        "treatment_state_at_least_98pct": treatment["state_accuracy"] >= 0.98,
        "treatment_answer_at_least_98pct": treatment["answer_accuracy"] >= 0.98,
        "every_depth_state_at_least_96pct": all(
            cell["accuracy"] >= 0.96 for cell in treatment["depth_state"].values()
        ),
        "state_within_one_point_of_host": (
            host["state_accuracy"] - treatment["state_accuracy"] <= 0.01
        ),
        "answer_within_one_point_of_host": (
            host["answer_accuracy"] - treatment["answer_accuracy"] <= 0.01
        ),
        "treatment_beats_ordinary_by_40pp": (
            treatment["state_accuracy"] - ordinary["state_accuracy"] >= 0.40
        ),
        "treatment_beats_false_generator_by_60pp": (
            treatment["state_accuracy"] - false_generator["state_accuracy"] >= 0.60
        ),
        "deranged_card_state_drop_at_least_60pp": (
            treatment["state_accuracy"] - deranged["state_accuracy"] >= 0.60
        ),
        "one_witness_state_drop_at_least_40pp": (
            treatment["state_accuracy"] - one_witness["state_accuracy"] >= 0.40
        ),
        "state_reset_drop_at_least_20pp": (
            treatment["state_accuracy"] - reset["state_accuracy"] >= 0.20
        ),
        "nonce_operation_invariant": evaluation["nonce_operation_invariance"][
            "all_rows_bit_identical"
        ],
        "one_development_zero_confirmation_access": (
            evaluation["development_accesses"] == 1
            and evaluation["confirmation_accesses"] == 0
        ),
        "treatment_has_218_parameters": evaluation["parameters"]["treatment"] == 218,
        "whole_system_below_150m": evaluation["parameters"]["whole_system"] < 150_000_000,
    }
    return {
        "schema": "r12_s7_learned_cayley_development_assessment_v1",
        "decision": (
            "qualify_s7_learned_cayley_for_fresh_confirmation"
            if all(gates.values())
            else "reject_s7_learned_cayley_development"
        ),
        "gates": gates,
        "scores": {
            "atomic_development": atomic,
            "host": host,
            "treatment": treatment,
            "ordinary_transformer": ordinary,
            "stride_two_generator": false_generator,
            "deranged_card": deranged,
            "one_witness": one_witness,
            "state_reset": reset,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 assessment: {args.out}")
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("schema") != "r12_s7_learned_cayley_development_evaluation_v1":
        raise SystemExit("S7 evaluation schema mismatch")
    result = assess(evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
