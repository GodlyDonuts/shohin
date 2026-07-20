#!/usr/bin/env python3
"""Independently assess complete physical fresh v1.3 confirmation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from assess_sd_cst_complete_physical_fresh import (
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EVIDENCE_SCHEMA,
    EXECUTOR_SCHEMA,
    EXPECTED_PARAMETERS,
    PACKET_SCHEMA,
    REQUIRED_ARMS,
    THRESHOLDS,
    _grouped,
    _hard,
    _metrics_equal,
    _output_exact,
    _packet_fields,
    _pointer_exact,
    _summary,
    load_json,
    load_torch,
    sha256_file,
)
from build_sd_cst_complete_physical_fresh_board import PROTOCOL
from confirm_sd_cst_complete_physical_fresh import (
    ACCESS_SCHEMA,
    CHECKPOINT_SHA256,
    CONFIRMATION_SHA256,
    DEVELOPMENT_ASSESSMENT_SCHEMA,
    DEVELOPMENT_ASSESSMENT_SHA256,
    GATE_CONFIG_SHA256,
    REPORT_SCHEMA,
)


ASSESSMENT_SCHEMA = "r12_sd_cst_complete_physical_fresh_confirmation_assessment_v1_3"


class ConfirmationAssessmentError(ValueError):
    pass


def assess_confirmation(
    *,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    development_assessment: Mapping[str, Any],
    packets: Mapping[str, Any],
    evidence: Mapping[str, Any],
    executor: Mapping[str, Any],
    ledger: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("protocol") != PROTOCOL:
        raise ConfirmationAssessmentError("confirmation report identity differs")
    if report.get("thresholds") != THRESHOLDS:
        raise ConfirmationAssessmentError("confirmation thresholds differ")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ConfirmationAssessmentError("confirmation checkpoint identity differs")
    if config.get("schema") != CONFIG_SCHEMA or config.get("protocol") != PROTOCOL or config.get("thresholds") != THRESHOLDS:
        raise ConfirmationAssessmentError("confirmation gate config differs")
    if packets.get("schema") != PACKET_SCHEMA or evidence.get("schema") != EVIDENCE_SCHEMA or executor.get("schema") != EXECUTOR_SCHEMA:
        raise ConfirmationAssessmentError("confirmation evidence identity differs")
    if development_assessment.get("schema") != DEVELOPMENT_ASSESSMENT_SCHEMA or development_assessment.get("protocol") != PROTOCOL or development_assessment.get("decision") != "authorize_one_sealed_confirmation" or development_assessment.get("all_gates_pass") is not True:
        raise ConfirmationAssessmentError("development authorization differs")
    if ledger.get("schema") != ACCESS_SCHEMA or ledger.get("protocol") != PROTOCOL or ledger.get("split") != "sd_cst_confirmation" or ledger.get("split_sha256") != CONFIRMATION_SHA256 or ledger.get("access_number") != 1:
        raise ConfirmationAssessmentError("confirmation ledger differs")
    arms = packets.get("arms")
    outputs = executor.get("outputs")
    if not isinstance(arms, Mapping) or set(arms) != REQUIRED_ARMS:
        raise ConfirmationAssessmentError("confirmation packet arms differ")
    if not isinstance(outputs, Mapping) or set(outputs) != REQUIRED_ARMS:
        raise ConfirmationAssessmentError("confirmation executor arms differ")
    gold = _hard(arms["gold"])
    pointer_groups = evidence.get("pointers")
    pointer_ranges = evidence.get("pointer_ranges")
    if not isinstance(pointer_groups, Mapping) or not isinstance(pointer_ranges, Mapping):
        raise ConfirmationAssessmentError("confirmation pointer evidence differs")
    renderer_index = evidence.get("renderer_index")
    renderer_names = evidence.get("renderer_names")
    if not isinstance(renderer_index, torch.Tensor) or renderer_index.dtype != torch.uint8 or tuple(renderer_index.shape) != (2_048,) or not isinstance(renderer_names, list) or len(renderer_names) != 4:
        raise ConfirmationAssessmentError("confirmation renderer evidence differs")
    gold_state = outputs["gold"]["final_state"]
    gold_answer = outputs["gold"]["answer"]
    arm_values = {}
    for name in ("treatment", "row_shuffled_labels"):
        fields = _packet_fields(_hard(arms[name]), gold)
        pointers_for_arm = _pointer_exact(pointer_groups[name], pointer_ranges, gold[0])
        state = outputs[name]["final_state"].eq(gold_state)
        answer = outputs[name]["answer"].eq(gold_answer)
        arm_values[name] = fields | pointers_for_arm | {
            "state": state,
            "answer": answer,
            "joint": state & answer,
        }
    independent_metrics = {
        name: {
            "overall": _summary(values),
            "packet_by_renderer": _grouped(values["packet"], renderer_index, renderer_names),
            "joint_by_renderer": _grouped(values["joint"], renderer_index, renderer_names),
        }
        for name, values in arm_values.items()
    }
    controls = {
        name: {
            "state_rate": float(outputs[name]["final_state"].eq(gold_state).float().mean()),
            "answer_rate": float(outputs[name]["answer"].eq(gold_answer).float().mean()),
        }
        for name in REQUIRED_ARMS - {"treatment", "row_shuffled_labels", "gold"}
    }
    treatment = independent_metrics["treatment"]
    shuffled = independent_metrics["row_shuffled_labels"]
    fit = checkpoint["arms"]["treatment"]["fit"]["train_metrics"]
    fit_min = min(float(item["rates"]["packet"]) for item in fit.values())
    source_poison = evidence.get("source_poison_bit_identical")
    parameter_report = checkpoint.get("parameters")
    parameter_exact = isinstance(parameter_report, Mapping) and all(
        int(parameter_report.get(name, -1)) == value
        for name, value in EXPECTED_PARAMETERS.items()
    )
    core_gates = {
        "development_authorization_exact": hashes["development_assessment"] == DEVELOPMENT_ASSESSMENT_SHA256,
        "fit_packet_min_renderer_at_least_99pct": fit_min >= THRESHOLDS["fit_packet_min_renderer"],
        "packet_overall_at_least_90pct": treatment["overall"]["packet"]["rate"] >= THRESHOLDS["packet_overall"],
        "packet_min_renderer_at_least_85pct": min(v["rate"] for v in treatment["packet_by_renderer"].values()) >= THRESHOLDS["packet_min_renderer"],
        "state_answer_joint_at_least_90pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS[f"{name}_overall"] for name in ("state", "answer", "joint")),
        "joint_min_renderer_at_least_85pct": min(v["rate"] for v in treatment["joint_by_renderer"].values()) >= THRESHOLDS["joint_min_renderer"],
        "all_packet_fields_at_least_95pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS["field_overall"] for name in ("initial", "kind", "identity", "amount", "query")),
        "all_pointers_at_least_90pct": all(treatment["overall"][name]["rate"] >= THRESHOLDS["pointer_overall"] for name in ("line", "binding", "initial_entity", "event_entity")),
        "treatment_packet_advantage_at_least_50pp": treatment["overall"]["packet"]["rate"] - shuffled["overall"]["packet"]["rate"] >= THRESHOLDS["treatment_packet_advantage"],
        "row_shuffled_packet_at_most_25pct": shuffled["overall"]["packet"]["rate"] <= THRESHOLDS["row_shuffled_packet_max"],
        "gold_executor_exact": _output_exact(outputs["gold"], arms["gold"]),
        "conditional_execution_exact": _output_exact(outputs["treatment"], arms["treatment"]),
        "post_stop_perturbation_invariant": all(torch.equal(outputs["treatment"][name], outputs["post_stop_perturbation"][name]) for name in ("final_state", "answer", "state_trajectory", "alive_trajectory")),
        "shuffled_packet_state_at_most_35pct": controls["shuffled_packet"]["state_rate"] <= THRESHOLDS["negative_state_max"],
        "reset_and_freeze_state_at_most_75pct": all(controls[name]["state_rate"] <= THRESHOLDS["reset_freeze_state_max"] for name in ("reset", "freeze")),
        "source_deleted_before_separate_execution": source_poison == {"treatment": True, "row_shuffled_labels": True},
        "complete_system_below_200m": parameter_exact and int(parameter_report["complete_system"]) < 200_000_000,
        "frozen_state_unchanged": all(checkpoint["arms"][name]["fit"]["frozen_parent_unchanged"] is True for name in ("treatment", "row_shuffled_labels")),
        "confirmation_one_after_development_one": report.get("custody", {}).get("development_accesses") == 1 and report.get("custody", {}).get("confirmation_accesses") == 1,
    }
    artifact_exact = (
        hashes["checkpoint"] == CHECKPOINT_SHA256
        and hashes["gate_config"] == GATE_CONFIG_SHA256
        and report["artifacts"]["checkpoint_sha256"] == hashes["checkpoint"]
        and report["artifacts"]["authorization_sha256"] == hashes["authorization"]
        and report["artifacts"]["packet_sha256"] == hashes["packets"]
        and report["artifacts"]["evidence_sha256"] == hashes["evidence"]
        and report["artifacts"]["executor_sha256"] == hashes["executor"]
        and report["custody"]["confirmation_ledger"]["sha256"] == hashes["ledger"]
    )
    pilot_metrics_exact = all(
        _metrics_equal(report["metrics"][arm]["overall"], independent_metrics[arm]["overall"])
        and _metrics_equal(report["metrics"][arm]["packet_by_renderer"], independent_metrics[arm]["packet_by_renderer"])
        and _metrics_equal(report["metrics"][arm]["joint_by_renderer"], independent_metrics[arm]["joint_by_renderer"])
        for arm in independent_metrics
    ) and all(
        math.isclose(float(report["controls"][name][metric]), float(controls[name][metric]), abs_tol=1e-12)
        for name in controls
        for metric in ("state_rate", "answer_rate")
    )
    assessor_gates = {
        "artifact_hashes_match": artifact_exact,
        "parameter_certificate_exact": parameter_exact,
        "pilot_metrics_match_independent_recomputation": pilot_metrics_exact,
        "pilot_gate_vector_matches_independent_recomputation": report.get("gates") == core_gates,
    }
    all_pass = all(core_gates.values()) and all(assessor_gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": "confirm_complete_physical_fresh_v1_3" if all_pass else "reject_complete_physical_fresh_confirmation_v1_3",
        "all_gates_pass": all_pass,
        "core_gates": core_gates,
        "assessor_gates": assessor_gates,
        "thresholds": THRESHOLDS,
        "parameters": dict(parameter_report) if isinstance(parameter_report, Mapping) else None,
        "metrics": independent_metrics,
        "controls": controls,
        "artifact_sha256": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 1},
        "claim_boundary": "Confirmed fresh finite renderer/name compilation into a source-deleted categorical executor; not broad natural-language or general reasoning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "checkpoint", "gate_config", "development_assessment", "authorization", "packets", "evidence", "executor", "ledger", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing confirmation assessment: {args.output}")
    paths = {
        "report": args.report,
        "checkpoint": args.checkpoint,
        "gate_config": args.gate_config,
        "development_assessment": args.development_assessment,
        "authorization": args.authorization,
        "packets": args.packets,
        "evidence": args.evidence,
        "executor": args.executor,
        "ledger": args.ledger,
    }
    result = assess_confirmation(
        report=load_json(args.report),
        checkpoint=load_torch(args.checkpoint),
        config=load_json(args.gate_config),
        development_assessment=load_json(args.development_assessment),
        packets=load_torch(args.packets),
        evidence=load_torch(args.evidence),
        executor=load_torch(args.executor),
        ledger=load_json(args.ledger),
        hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
