#!/usr/bin/env python3
"""Apply the frozen S7 confirmation gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess_confirmation(evaluation: dict[str, object]) -> dict[str, object]:
    arms = evaluation["arms"]
    treatment = arms["treatment"]
    host = arms["host"]
    ordinary = arms["ordinary_transformer"]
    false_generator = arms["stride_two_generator"]
    deranged = arms["deranged_card"]
    one_witness = arms["one_witness"]
    reset = arms["state_reset"]
    gates = {
        "bound_checkpoint_hash": evaluation["checkpoint_sha256"]
        == "c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607",
        "bound_development_assessment_hash": evaluation[
            "development_assessment_sha256"
        ]
        == "2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4",
        "bound_confirmation_hash": evaluation["confirmation_sha256"]
        == "c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5",
        "successor_and_zero_only_training_contract": (
            "23 successor cells plus three zero anchors"
            in str(evaluation["training_contract"])
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
        "one_development_one_confirmation_access": (
            evaluation["development_accesses"] == 1
            and evaluation["confirmation_accesses"] == 1
        ),
        "treatment_has_218_parameters": evaluation["parameters"]["treatment"] == 218,
        "whole_system_below_150m": evaluation["parameters"]["whole_system"] < 150_000_000,
    }
    return {
        "schema": "r12_s7_learned_cayley_confirmation_assessment_v1",
        "decision": (
            "confirm_s7_learned_cayley_contextual_law_compilation"
            if all(gates.values())
            else "reject_s7_learned_cayley_confirmation"
        ),
        "gates": gates,
        "scores": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 confirmation assessment: {args.out}")
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("schema") != "r12_s7_learned_cayley_confirmation_evaluation_v1":
        raise SystemExit("S7 confirmation evaluation schema mismatch")
    result = assess_confirmation(evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
