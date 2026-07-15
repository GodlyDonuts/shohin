#!/usr/bin/env python3
"""Frozen multi-arm advancement decision for the R9c used-board canary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from categorical_microcode import sha256_file


ARMS = ("treatment", "static", "no_syndrome", "shuffled_goal")
OOD_REGIMES = ("language_ood", "full_ood")
PRESERVATION_REGIMES = ("fit_iid", "depth_ood")
THRESHOLDS = {
    "treatment_over_static_ood_operation_margin": 0.03,
    "treatment_over_no_syndrome_ood_operation_margin": 0.02,
    "treatment_over_shuffled_goal_ood_operation_margin": 0.05,
    "treatment_over_static_ood_answer_margin": 0.03,
    "treatment_fresh_language_answer_floor": 0.60,
    "treatment_fresh_full_answer_floor": 0.35,
    "treatment_development_preservation_operation_floor": 0.95,
    "treatment_fresh_common_mode_fraction_of_wrong_ceiling": 0.50,
    "adaptive_operation_regression_ceiling": 0.01,
    "adaptive_update_fraction_ceiling": 0.80,
    "query_accuracy_floor": 0.98,
}


def load_report(path, arm):
    report = json.load(open(path))
    if report.get("audit") != "referential_bidirectional_syndrome_microcode_eval_r9c":
        raise ValueError("{} is not an R9c evaluation".format(path))
    metadata = report.get("adapter_metadata", {})
    if metadata.get("arm") != arm:
        raise ValueError("{} does not contain the {} arm".format(path, arm))
    return report


def aggregate(report, mode, regimes):
    cells = [report[mode][regime] for regime in regimes]
    operations = sum(cell["operations"] for cell in cells)
    cases = sum(cell["cases"] for cell in cells)
    wrong = operations - sum(cell["joint_operation_correct"] for cell in cells)
    agreed_wrong = sum(cell["agreed_wrong_common_mode"] for cell in cells)
    return {
        "cases": cases,
        "operations": operations,
        "operation_accuracy": sum(cell["joint_operation_correct"] for cell in cells) / operations,
        "answer_accuracy": sum(cell["answer_correct"] for cell in cells) / cases,
        "query_accuracy": sum(cell["query_correct"] for cell in cells) / cases,
        "common_mode_fraction_of_wrong": agreed_wrong / wrong if wrong else 0.0,
        "mean_event_updates": sum(cell["event_updates"] for cell in cells) / operations,
    }


def audit_contract(development, fresh):
    reference = development["treatment"]
    reference_metadata = reference["adapter_metadata"]
    invariant_metadata = (
        "base_sha256", "pointer_adapter_sha256", "data_sha256", "tokenizer_sha256",
        "seed", "memory_dim", "rounds", "batch_groups", "selected_groups",
        "selected_examples", "updates", "adapter_parameters", "initial_adapter_sha256",
        "supervision", "inference_inputs",
    )
    errors = []
    for board_name, reports in (("development", development), ("fresh", fresh)):
        data_hashes = {report["data_sha256"] for report in reports.values()}
        if len(data_hashes) != 1:
            errors.append("{} reports use different evaluation data".format(board_name))
        for arm, report in reports.items():
            metadata = report["adapter_metadata"]
            for key in invariant_metadata:
                if metadata.get(key) != reference_metadata.get(key):
                    errors.append("{} {} differs on {}".format(board_name, arm, key))
            if report.get("base_sha256") != reference.get("base_sha256"):
                errors.append("{} {} differs on base".format(board_name, arm))
            if report.get("pointer_adapter_sha256") != reference.get("pointer_adapter_sha256"):
                errors.append("{} {} differs on pointer".format(board_name, arm))
            if report["alu_basis"]["correct"] != report["alu_basis"]["total"]:
                errors.append("{} {} has an inexact ALU".format(board_name, arm))
    if development["treatment"]["data_sha256"] == fresh["treatment"]["data_sha256"]:
        errors.append("development and fresh-used boards are identical")
    return errors


def assess(development, fresh):
    contract_errors = audit_contract(development, fresh)
    development_metrics = {
        arm: {
            "ood_fixed": aggregate(report, "fixed", OOD_REGIMES),
            "preservation_fixed": aggregate(report, "fixed", PRESERVATION_REGIMES),
        } for arm, report in development.items()
    }
    fresh_metrics = {
        arm: {
            "ood_fixed": aggregate(report, "fixed", OOD_REGIMES),
            "ood_adaptive": aggregate(report, "adaptive", OOD_REGIMES),
            "language_fixed": report["fixed"]["language_ood"],
            "full_fixed": report["fixed"]["full_ood"],
        } for arm, report in fresh.items()
    }
    treatment = fresh_metrics["treatment"]["ood_fixed"]
    adaptive = fresh_metrics["treatment"]["ood_adaptive"]
    preservation = development_metrics["treatment"]["preservation_fixed"]
    gates = {
        "all_arm_and_board_contracts_match": not contract_errors,
        "treatment_beats_static_ood_operations": (
            treatment["operation_accuracy"]
            >= fresh_metrics["static"]["ood_fixed"]["operation_accuracy"]
            + THRESHOLDS["treatment_over_static_ood_operation_margin"]
        ),
        "syndrome_adds_ood_operation_value": (
            treatment["operation_accuracy"]
            >= fresh_metrics["no_syndrome"]["ood_fixed"]["operation_accuracy"]
            + THRESHOLDS["treatment_over_no_syndrome_ood_operation_margin"]
        ),
        "semantic_future_goals_add_ood_operation_value": (
            treatment["operation_accuracy"]
            >= fresh_metrics["shuffled_goal"]["ood_fixed"]["operation_accuracy"]
            + THRESHOLDS["treatment_over_shuffled_goal_ood_operation_margin"]
        ),
        "treatment_beats_static_ood_answers": (
            treatment["answer_accuracy"]
            >= fresh_metrics["static"]["ood_fixed"]["answer_accuracy"]
            + THRESHOLDS["treatment_over_static_ood_answer_margin"]
        ),
        "language_ood_answer_floor": (
            fresh_metrics["treatment"]["language_fixed"]["answer_accuracy"]
            >= THRESHOLDS["treatment_fresh_language_answer_floor"]
        ),
        "full_ood_answer_floor": (
            fresh_metrics["treatment"]["full_fixed"]["answer_accuracy"]
            >= THRESHOLDS["treatment_fresh_full_answer_floor"]
        ),
        "fit_and_depth_operation_preservation": (
            preservation["operation_accuracy"]
            >= THRESHOLDS["treatment_development_preservation_operation_floor"]
        ),
        "common_mode_wrong_agreement_is_bounded": (
            treatment["common_mode_fraction_of_wrong"]
            <= THRESHOLDS["treatment_fresh_common_mode_fraction_of_wrong_ceiling"]
        ),
        "adaptive_replay_preserves_ood_operations": (
            adaptive["operation_accuracy"]
            >= treatment["operation_accuracy"] - THRESHOLDS["adaptive_operation_regression_ceiling"]
        ),
        "adaptive_replay_reduces_event_updates": (
            adaptive["mean_event_updates"]
            <= treatment["mean_event_updates"] * THRESHOLDS["adaptive_update_fraction_ceiling"]
        ),
        "text_query_bridge_remains_accurate": (
            treatment["query_accuracy"] >= THRESHOLDS["query_accuracy_floor"]
        ),
    }
    return {
        "audit": "referential_bidirectional_syndrome_microcode_decision_r9c",
        "thresholds": THRESHOLDS,
        "contract_errors": contract_errors,
        "development_metrics": development_metrics,
        "fresh_used_metrics": fresh_metrics,
        "gates": gates,
        "authorize_full_matched_fit": all(gates.values()),
        "decision": (
            "authorize_r9c_full_matched_fit_before_untouched_confirmation"
            if all(gates.values()) else "reject_r9c_used_board_canary"
        ),
        "claim_boundary": (
            "A pass authorizes a full matched-arm fit and one untouched confirmatory board only. "
            "It does not establish broad reasoning, decoder integration, or context scaling."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for board in ("development", "fresh"):
        for arm in ARMS:
            parser.add_argument("--{}-{}".format(board, arm.replace("_", "-")), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output")
    development = {
        arm: load_report(getattr(args, "development_{}".format(arm)), arm) for arm in ARMS
    }
    fresh = {arm: load_report(getattr(args, "fresh_{}".format(arm)), arm) for arm in ARMS}
    result = assess(development, fresh)
    result["inputs"] = {
        "development": {
            arm: {
                "path": os.path.realpath(getattr(args, "development_{}".format(arm))),
                "sha256": sha256_file(getattr(args, "development_{}".format(arm))),
            } for arm in ARMS
        },
        "fresh": {
            arm: {
                "path": os.path.realpath(getattr(args, "fresh_{}".format(arm))),
                "sha256": sha256_file(getattr(args, "fresh_{}".format(arm))),
            } for arm in ARMS
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "decision": result["decision"], "gates": result["gates"],
        "authorize_full_matched_fit": result["authorize_full_matched_fit"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
