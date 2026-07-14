#!/usr/bin/env python3
"""Locked attribution gate for paired-language microcode compiler arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCHED_METADATA = (
    "base_sha256", "data_sha256", "admission_sha256", "seed", "layer", "hidden",
    "batch_pairs", "selected_pairs", "selected_examples", "initial_adapter_sha256",
)


def combined_accuracy(report, regimes, field):
    selected = [report["summary"][regime] for regime in regimes]
    numerator = sum(item[field.replace("accuracy", "correct")] for item in selected)
    denominator = sum(item["cases"] for item in selected)
    return numerator / denominator


def compare(control, equivalence):
    control_meta = control["adapter_metadata"]
    equivalence_meta = equivalence["adapter_metadata"]
    mismatches = {
        key: [control_meta.get(key), equivalence_meta.get(key)]
        for key in MATCHED_METADATA if control_meta.get(key) != equivalence_meta.get(key)
    }
    weights_valid = (
        float(control_meta.get("equivalence_weight", -1)) == 0.0
        and float(equivalence_meta.get("equivalence_weight", -1)) > 0.0
    )
    control_language_full = combined_accuracy(control, ("language_ood", "full_ood"), "answer_accuracy")
    equivalence_language_full = combined_accuracy(
        equivalence, ("language_ood", "full_ood"), "answer_accuracy",
    )
    attribution_gates = {
        "matched_metadata": not mismatches,
        "matched_weight_contract": weights_valid,
        "language_full_answer_gain_at_least_0_05": (
            equivalence_language_full - control_language_full >= 0.05
        ),
        "all_program_exact_gain_at_least_0_05": (
            equivalence["summary"]["all"]["program_exact_accuracy"]
            - control["summary"]["all"]["program_exact_accuracy"] >= 0.05
        ),
        "fit_answer_regression_at_most_0_03": (
            equivalence["summary"]["fit_iid"]["answer_accuracy"]
            >= control["summary"]["fit_iid"]["answer_accuracy"] - 0.03
        ),
        "depth_answer_regression_at_most_0_03": (
            equivalence["summary"]["depth_ood"]["answer_accuracy"]
            >= control["summary"]["depth_ood"]["answer_accuracy"] - 0.03
        ),
    }
    equivalence_attributed = all(attribution_gates.values())
    control_pass = bool(control.get("advance_to_decoder_bridge"))
    equivalence_pass = bool(equivalence.get("advance_to_decoder_bridge"))
    if equivalence_pass and equivalence_attributed:
        decision = "advance_equivalence_compiler_to_manual_bridge_gate"
    elif control_pass:
        decision = "advance_diverse_control_only_equivalence_not_attributed"
    else:
        decision = "reject_paired_compiler_r2"
    return {
        "audit": "categorical_microcode_equivalence_comparison_v2",
        "mismatches": mismatches,
        "control_language_full_answer_accuracy": control_language_full,
        "equivalence_language_full_answer_accuracy": equivalence_language_full,
        "attribution_gates": attribution_gates,
        "equivalence_attributed": equivalence_attributed,
        "control_absolute_pass": control_pass,
        "equivalence_absolute_pass": equivalence_pass,
        "decision": decision,
        "claim_boundary": (
            "Only a matched improvement can be attributed to equivalence loss; a control-only pass "
            "supports diverse compiler supervision but not the new loss."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--equivalence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing comparison: {}".format(args.out))
    control = json.loads(Path(args.control).read_text())
    equivalence = json.loads(Path(args.equivalence).read_text())
    result = compare(control, equivalence)
    result["control"] = str(Path(args.control).resolve())
    result["equivalence"] = str(Path(args.equivalence).resolve())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
