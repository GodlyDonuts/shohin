#!/usr/bin/env python3
"""Locked matched comparator for binding-first referential slot compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCHED_FIELDS = (
    "protocol", "base_sha256", "data_sha256", "admission_sha256",
    "label_admission_sha256", "seed", "layer", "hidden", "batch_groups",
    "selected_groups", "selected_examples", "updates", "learning_rate",
    "warmup_updates", "gradient_clip", "basis_weight", "mention_weight",
    "adapter_parameters", "base_parameters_trainable", "initial_adapter_sha256",
    "view_contract", "inference_inputs",
)


def language_full_answer(report):
    summary = report["summary"]
    correct = summary["language_ood"]["answer_correct"] + summary["full_ood"]["answer_correct"]
    cases = summary["language_ood"]["cases"] + summary["full_ood"]["cases"]
    return correct / cases


def language_full_role_given_kind(report):
    records = [
        record for record in report["records"] if record["regime"] in {"language_ood", "full_ood"}
    ]
    correct = total = 0
    for record in records:
        for target_kind, predicted_kind, target_role, predicted_role in zip(
            record["operation_kind_targets"], record["operation_kind_predictions"],
            record["operation_role_targets"], record["operation_role_predictions"],
        ):
            if target_role >= 0 and predicted_kind == target_kind:
                total += 1
                correct += predicted_role == target_role
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    control = json.load(open(args.control))
    candidate = json.load(open(args.candidate))
    if control.get("audit") != "referential_slot_microcode_eval_v4" or candidate.get("audit") != control.get("audit"):
        raise SystemExit("invalid referential evaluation reports")
    control_metadata = control["adapter_metadata"]
    candidate_metadata = candidate["adapter_metadata"]
    mismatches = {
        field: [control_metadata.get(field), candidate_metadata.get(field)]
        for field in MATCHED_FIELDS if control_metadata.get(field) != candidate_metadata.get(field)
    }
    role_contract = (
        control_metadata.get("role_mode") == "absolute"
        and candidate_metadata.get("role_mode") == "pointer"
    )
    control_lf = language_full_answer(control)
    candidate_lf = language_full_answer(candidate)
    control_role = language_full_role_given_kind(control)
    candidate_role = language_full_role_given_kind(candidate)
    gates = {
        "matched_metadata": not mismatches,
        "matched_role_contract": role_contract,
        "candidate_absolute_gates": all(candidate.get("gates", {}).values()),
        "language_full_answer_gain_at_least_0_05": candidate_lf - control_lf >= 0.05,
        "all_program_exact_gain_at_least_0_05": (
            candidate["summary"]["all"]["program_exact_accuracy"]
            - control["summary"]["all"]["program_exact_accuracy"] >= 0.05
        ),
        "operation_role_given_kind_gain_at_least_0_10": candidate_role - control_role >= 0.10,
        "fit_answer_regression_at_most_0_03": (
            control["summary"]["fit_iid"]["answer_accuracy"]
            - candidate["summary"]["fit_iid"]["answer_accuracy"] <= 0.03
        ),
        "depth_answer_regression_at_most_0_03": (
            control["summary"]["depth_ood"]["answer_accuracy"]
            - candidate["summary"]["depth_ood"]["answer_accuracy"] <= 0.03
        ),
    }
    advance = all(gates.values())
    result = {
        "audit": "referential_slot_microcode_comparison_v4",
        "control": str(Path(args.control).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "control_language_full_answer_accuracy": control_lf,
        "candidate_language_full_answer_accuracy": candidate_lf,
        "control_operation_role_given_kind_accuracy": control_role,
        "candidate_operation_role_given_kind_accuracy": candidate_role,
        "mismatches": mismatches,
        "attribution_gates": gates,
        "binding_attributed": advance,
        "decision": "advance_referential_slot_compiler_r4" if advance else "reject_referential_slot_compiler_r4",
        "claim_boundary": (
            "Only a matched pointer-over-absolute gain with all absolute gates can be attributed to "
            "dynamic text-only entity binding."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
