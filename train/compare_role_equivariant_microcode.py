#!/usr/bin/env python3
"""Locked attribution gate for role-factorized equivariant compiler arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCHED_METADATA = (
    "base_sha256", "data_sha256", "admission_sha256", "seed", "layer", "hidden",
    "batch_groups", "selected_groups", "selected_examples", "updates", "learning_rate",
    "warmup_updates", "gradient_clip", "basis_weight", "role_factor_contract",
    "initial_adapter_sha256",
)


def combined_accuracy(report, regimes):
    selected = [report["summary"][regime] for regime in regimes]
    return sum(item["answer_correct"] for item in selected) / sum(item["cases"] for item in selected)


def compare(control, candidate):
    control_meta, candidate_meta = control["adapter_metadata"], candidate["adapter_metadata"]
    mismatches = {
        key: [control_meta.get(key), candidate_meta.get(key)]
        for key in MATCHED_METADATA if control_meta.get(key) != candidate_meta.get(key)
    }
    weights_valid = (
        float(control_meta.get("semantic_weight", -1)) == 0.0
        and float(control_meta.get("permutation_weight", -1)) == 0.0
        and float(candidate_meta.get("semantic_weight", -1)) == 0.5
        and float(candidate_meta.get("permutation_weight", -1)) == 1.0
    )
    control_language_full = combined_accuracy(control, ("language_ood", "full_ood"))
    candidate_language_full = combined_accuracy(candidate, ("language_ood", "full_ood"))
    gates = {
        "matched_metadata": not mismatches,
        "matched_weight_contract": weights_valid,
        "language_full_answer_gain_at_least_0_05": candidate_language_full - control_language_full >= 0.05,
        "all_program_exact_gain_at_least_0_05": (
            candidate["summary"]["all"]["program_exact_accuracy"]
            - control["summary"]["all"]["program_exact_accuracy"] >= 0.05
        ),
        "fit_answer_regression_at_most_0_03": (
            candidate["summary"]["fit_iid"]["answer_accuracy"]
            >= control["summary"]["fit_iid"]["answer_accuracy"] - 0.03
        ),
        "depth_answer_regression_at_most_0_03": (
            candidate["summary"]["depth_ood"]["answer_accuracy"]
            >= control["summary"]["depth_ood"]["answer_accuracy"] - 0.03
        ),
    }
    attributed = all(gates.values())
    control_pass = bool(control.get("advance_to_decoder_bridge"))
    candidate_pass = bool(candidate.get("advance_to_decoder_bridge"))
    if candidate_pass and attributed:
        decision = "advance_role_equivariant_compiler_to_manual_bridge_gate"
    elif control_pass:
        decision = "advance_role_factorized_control_only_constraints_not_attributed"
    else:
        decision = "reject_role_equivariant_compiler_r3"
    return {
        "audit": "role_equivariant_microcode_comparison_v3",
        "mismatches": mismatches,
        "control_language_full_answer_accuracy": control_language_full,
        "candidate_language_full_answer_accuracy": candidate_language_full,
        "attribution_gates": gates,
        "constraints_attributed": attributed,
        "control_absolute_pass": control_pass,
        "candidate_absolute_pass": candidate_pass,
        "decision": decision,
        "claim_boundary": (
            "Only a matched gain can be attributed to representation alignment and role equivariance. "
            "A control-only pass supports anchor replay and factorization but not the constraints."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing comparison")
    control = json.loads(Path(args.control).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    result = compare(control, candidate)
    result["control"] = str(Path(args.control).resolve())
    result["candidate"] = str(Path(args.candidate).resolve())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
