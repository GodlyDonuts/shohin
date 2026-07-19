#!/usr/bin/env python3
"""Apply the frozen S6 development gates without fitting or model access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_assessment(evaluation: dict[str, object]) -> dict[str, object]:
    if evaluation.get("schema") != "r12_s6_contextual_affine_law_development_eval_v1":
        raise ValueError("invalid S6 evaluation schema")
    treatment = evaluation["arms"]["treatment"]
    host = evaluation["arms"]["host"]
    deranged = evaluation["arms"]["deranged_card"]
    one_witness = evaluation["arms"]["one_witness"]
    reset = evaluation["arms"]["state_reset"]
    law_id = evaluation["arms"]["law_id"]
    depth_gate = all(
        bucket["accuracy"] >= 0.92 for bucket in treatment["depth_state"].values()
    )
    gates = {
        "treatment_atomic_fit_at_least_99pct": evaluation["fit"]["treatment"]["atomic_train_accuracy"] >= 0.99,
        "law_id_control_atomic_fit_at_least_99pct": evaluation["fit"]["law_id_control"]["atomic_train_accuracy"] >= 0.99,
        "heldout_atomic_destination_at_least_95pct": evaluation["atomic_development"]["accuracy"] >= 0.95,
        "treatment_state_at_least_95pct": treatment["state_accuracy"] >= 0.95,
        "treatment_answer_at_least_95pct": treatment["answer_accuracy"] >= 0.95,
        "every_depth_state_at_least_92pct": depth_gate,
        "state_within_one_point_of_host": host["state_accuracy"] - treatment["state_accuracy"] <= 0.01,
        "answer_within_one_point_of_host": host["answer_accuracy"] - treatment["answer_accuracy"] <= 0.01,
        "deranged_card_state_drop_at_least_40pp": treatment["state_accuracy"] - deranged["state_accuracy"] >= 0.40,
        "one_witness_state_drop_at_least_30pp": treatment["state_accuracy"] - one_witness["state_accuracy"] >= 0.30,
        "state_reset_drop_at_least_20pp": treatment["state_accuracy"] - reset["state_accuracy"] >= 0.20,
        "law_id_control_trails_at_least_40pp": treatment["state_accuracy"] - law_id["state_accuracy"] >= 0.40,
        "multi_law_state_at_least_95pct": treatment["multi_law_state_accuracy"] >= 0.95,
        "nonce_name_invariant": evaluation["nonce_name_invariance"]["all_rows_bit_identical"],
        "treatment_module_below_8m": evaluation["parameters"]["treatment"] < 8_000_000,
        "whole_system_below_150m": evaluation["parameters"]["whole_system"] < 150_000_000,
        "one_development_zero_confirmation_access": evaluation["development_accesses"] == 1
        and evaluation["confirmation_accesses"] == 0,
        "atomic_only_training_contract": evaluation["training_contract"].startswith(
            "atomic destination cells from train laws only"
        ),
    }
    qualified = all(gates.values())
    return {
        "schema": "r12_s6_contextual_affine_law_development_assessment_v1",
        "decision": (
            "qualify_s6_for_one_confirmation"
            if qualified
            else "reject_s6_contextual_affine_law_development"
        ),
        "gates": gates,
        "scores": {
            "atomic_development": evaluation["atomic_development"],
            "host": host,
            "treatment": treatment,
            "deranged_card": deranged,
            "one_witness": one_witness,
            "state_reset": reset,
            "law_id": law_id,
            "scale_diagnostic": evaluation["scale_diagnostic"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S6 assessment: {args.out}")
    assessment = build_assessment(json.loads(args.evaluation.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": assessment["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
