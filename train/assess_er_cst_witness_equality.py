#!/usr/bin/env python3
"""Independently assess one-read ER-CST witness-equality evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from build_er_cst_witness_equality_board import DEVELOPMENT_SPLIT, PROTOCOL
from er_cst_rule_card_adapter import (
    RULE_CARD_COUNT,
    TiedRuleCardMotor,
    rule_motor_certificate,
)
from er_cst_witness_equality import TRAINING_CONTRACT
from pilot_er_cst_witness_equality import (
    ACCESS_SCHEMA,
    BOARD_REPORT_SHA256,
    BOARD_SOURCE_COMMIT,
    CHECKPOINT_SCHEMA,
    EVIDENCE_SCHEMA,
    REPORT_SCHEMA,
    THRESHOLDS,
)
from pilot_sd_cst_byte_addressed import sha256_file
from sd_cst import CategoricalStateReader
from sd_cst_binding_bus import PERMUTATIONS


ASSESSMENT_SCHEMA = "r12_er_cst_witness_equality_development_assessment_v1_1"
ROWS = 2_048
ARMS = {"treatment", "family_deranged", "equality_ablated"}
EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 67_641_890,
    "motor": 2_438,
    "reader": 835,
    "complete_system": 192_726_827,
    "headroom_below_200m": 7_273_173,
    "trainable": 12_021_276,
}


class AssessmentError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssessmentError(f"JSON object required: {path}")
    return value


def load_torch(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise AssessmentError(f"torch mapping required: {path}")
    return value


def require_tensor(
    raw: Mapping[str, object], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = raw.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.int16
        or tuple(value.shape) != shape
    ):
        raise AssessmentError(f"ER-CST witness raw tensor differs: {name}")
    return value.long()


def summary(value: torch.Tensor) -> dict[str, object]:
    return {
        "correct": int(value.sum()),
        "rows": int(value.numel()),
        "rate": float(value.float().mean()),
    }


def recompute_arm(raw: Mapping[str, object]) -> dict[str, object]:
    predicted = {
        "initial": require_tensor(raw, "pred_initial", (ROWS,)),
        "cards": require_tensor(raw, "pred_cards", (ROWS, 3)),
        "events": require_tensor(raw, "pred_events", (ROWS, 9)),
        "halt": require_tensor(raw, "pred_halt", (ROWS, 9)),
        "query": require_tensor(raw, "pred_query", (ROWS,)),
        "line_pointer": require_tensor(raw, "pred_line_pointer", (ROWS, 13)),
        "binding_pointer": require_tensor(raw, "pred_binding_pointer", (ROWS, 3)),
        "initial_pointer": require_tensor(raw, "pred_initial_pointer", (ROWS, 3)),
        "witness_pointer": require_tensor(raw, "pred_witness_pointer", (ROWS, 3, 6)),
        "query_pointer": require_tensor(raw, "pred_query_pointer", (ROWS,)),
        "state": require_tensor(raw, "pred_state", (ROWS,)),
        "answer": require_tensor(raw, "pred_answer", (ROWS,)),
    }
    target = {
        "initial": require_tensor(raw, "target_initial", (ROWS,)),
        "cards": require_tensor(raw, "target_cards", (ROWS, 3)),
        "events": require_tensor(raw, "target_events", (ROWS, 9)),
        "halt": require_tensor(raw, "target_halt", (ROWS, 9)),
        "query": require_tensor(raw, "target_query", (ROWS,)),
        "line_ranges": require_tensor(raw, "target_line_ranges", (ROWS, 13, 2)),
        "binding_ranges": require_tensor(raw, "target_binding_ranges", (ROWS, 3, 2)),
        "initial_ranges": require_tensor(raw, "target_initial_ranges", (ROWS, 3, 2)),
        "witness_ranges": require_tensor(raw, "target_witness_ranges", (ROWS, 3, 6, 2)),
        "query_range": require_tensor(raw, "target_query_range", (ROWS, 2)),
        "state": require_tensor(raw, "target_state", (ROWS,)),
        "answer": require_tensor(raw, "target_answer", (ROWS,)),
    }

    def pointer_exact(prediction: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        return (
            (prediction >= ranges[..., 0]) & (prediction < ranges[..., 1])
        ).flatten(1).all(-1)

    exact = {
        "initial": predicted["initial"].eq(target["initial"]),
        "cards": predicted["cards"].eq(target["cards"]).all(-1),
        "events": (
            predicted["events"].eq(target["events"]) | target["halt"].bool()
        ).all(-1),
        "halt": predicted["halt"].eq(target["halt"]).all(-1),
        "query": predicted["query"].eq(target["query"]),
        "line_pointer": pointer_exact(
            predicted["line_pointer"], target["line_ranges"]
        ),
        "binding_pointer": pointer_exact(
            predicted["binding_pointer"], target["binding_ranges"]
        ),
        "initial_pointer": pointer_exact(
            predicted["initial_pointer"], target["initial_ranges"]
        ),
        "witness_pointer": pointer_exact(
            predicted["witness_pointer"], target["witness_ranges"]
        ),
        "query_pointer": (
            (predicted["query_pointer"] >= target["query_range"][:, 0])
            & (predicted["query_pointer"] < target["query_range"][:, 1])
        ),
        "state": predicted["state"].eq(target["state"]),
        "answer": predicted["answer"].eq(target["answer"]),
    }
    exact["packet"] = torch.stack(
        [exact[name] for name in ("initial", "cards", "events", "halt", "query")]
    ).all(0)
    exact["joint"] = exact["packet"] & exact["state"] & exact["answer"]
    depth = raw.get("depth")
    renderer_index = raw.get("renderer_index")
    renderer_names = raw.get("renderer_names")
    if (
        not isinstance(depth, torch.Tensor)
        or depth.dtype != torch.uint8
        or tuple(depth.shape) != (ROWS,)
        or not isinstance(renderer_index, torch.Tensor)
        or renderer_index.dtype != torch.uint8
        or tuple(renderer_index.shape) != (ROWS,)
        or not isinstance(renderer_names, list)
        or len(renderer_names) != 4
        or sorted(set(renderer_names)) != sorted(renderer_names)
        or not all(type(name) is str for name in renderer_names)
    ):
        raise AssessmentError("ER-CST witness grouping evidence differs")
    if set(map(int, depth.unique())) != set(range(1, 9)):
        raise AssessmentError("ER-CST witness depth evidence differs")
    if set(map(int, renderer_index.unique())) != set(range(4)):
        raise AssessmentError("ER-CST witness renderer evidence differs")
    by_depth = {}
    by_renderer = {}
    for value in range(1, 9):
        mask = depth.eq(value)
        by_depth[str(value)] = {
            name: summary(exact[name][mask])
            for name in ("packet", "state", "answer", "joint")
        }
    for index, name in enumerate(renderer_names):
        mask = renderer_index.eq(index)
        by_renderer[name] = {
            field: summary(exact[field][mask])
            for field in ("packet", "state", "answer", "joint")
        }
    return {
        "overall": {name: summary(value) for name, value in exact.items()},
        "by_depth": by_depth,
        "by_renderer": by_renderer,
    }


def metric_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            metric_equal(left[name], right[name]) for name in left
        )
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), abs_tol=1e-12)
    return left == right


def minimum_group(metrics: Mapping[str, object], group: str, field: str) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise AssessmentError(f"ER-CST witness {group} metrics are absent")
    return min(float(value[field]["rate"]) for value in values.values())


def certificate_exact(arm: Mapping[str, object]) -> bool:
    motor = TiedRuleCardMotor()
    reader = CategoricalStateReader()
    motor.load_state_dict(arm["motor_state"], strict=True)
    reader.load_state_dict(arm["reader_state"], strict=True)
    state, card, motor_target = rule_motor_certificate()
    with torch.no_grad():
        motor_prediction = motor(
            F.one_hot(state, RULE_CARD_COUNT).float(),
            F.one_hot(card, RULE_CARD_COUNT).float(),
        ).argmax(-1)
        state_ids = torch.arange(RULE_CARD_COUNT).repeat_interleave(3)
        query_ids = torch.arange(3).repeat(RULE_CARD_COUNT)
        reader_target = torch.tensor(
            [
                PERMUTATIONS[int(state_id)][int(query_id)]
                for state_id, query_id in zip(state_ids, query_ids, strict=True)
            ]
        )
        reader_prediction = reader(
            F.one_hot(state_ids, RULE_CARD_COUNT).float(),
            F.one_hot(query_ids, 3).float(),
        ).argmax(-1)
    return torch.equal(motor_prediction, motor_target) and torch.equal(
        reader_prediction, reader_target
    )


def compute_gates(
    metrics: Mapping[str, Mapping[str, object]], checkpoint: Mapping[str, object]
) -> dict[str, bool]:
    treatment = metrics["treatment"]
    controls = [metrics["family_deranged"], metrics["equality_ablated"]]
    overall = treatment["overall"]
    return {
        "treatment_packet_at_least_90pct": float(overall["packet"]["rate"])
        >= THRESHOLDS["packet_overall"],
        "treatment_state_answer_joint_at_least_90pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS[f"{name}_overall"]
            for name in ("state", "answer", "joint")
        ),
        "treatment_min_renderer_joint_at_least_85pct": minimum_group(
            treatment, "by_renderer", "joint"
        )
        >= THRESHOLDS["joint_min_renderer"],
        "treatment_min_depth_joint_at_least_80pct": minimum_group(
            treatment, "by_depth", "joint"
        )
        >= THRESHOLDS["joint_min_depth"],
        "all_packet_fields_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["field_overall"]
            for name in ("initial", "cards", "events", "halt", "query")
        ),
        "all_pointers_at_least_90pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["pointer_overall"]
            for name in (
                "line_pointer",
                "binding_pointer",
                "initial_pointer",
                "witness_pointer",
                "query_pointer",
            )
        ),
        "treatment_packet_advantage_at_least_50pp": all(
            float(overall["packet"]["rate"])
            - float(control["overall"]["packet"]["rate"])
            >= THRESHOLDS["treatment_packet_advantage"]
            for control in controls
        ),
        "treatment_joint_advantage_at_least_50pp": all(
            float(overall["joint"]["rate"])
            - float(control["overall"]["joint"]["rate"])
            >= THRESHOLDS["treatment_joint_advantage"]
            for control in controls
        ),
        "negative_packets_at_most_35pct": all(
            float(control["overall"]["packet"]["rate"])
            <= THRESHOLDS["negative_packet_max"]
            for control in controls
        ),
        "negative_states_at_most_40pct": all(
            float(control["overall"]["state"]["rate"])
            <= THRESHOLDS["negative_state_max"]
            for control in controls
        ),
        "finite_motor_and_reader_certificates_exact": all(
            certificate_exact(arm) for arm in checkpoint["arms"].values()
        ),
        "confirmed_parent_unchanged": all(
            arm["fit"]["frozen_parent_unchanged"] is True
            for arm in checkpoint["arms"].values()
        ),
        "complete_system_below_200m": int(checkpoint["parameters"]["complete_system"])
        < 200_000_000,
        "development_one_confirmation_zero": True,
    }


def assess(
    report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    evidence: Mapping[str, object],
    ledger: Mapping[str, object],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or ledger.get("schema") != ACCESS_SCHEMA
        or any(value.get("protocol") != PROTOCOL for value in (report, checkpoint, evidence, ledger))
    ):
        raise AssessmentError("ER-CST witness artifact identity differs")
    if report.get("thresholds") != THRESHOLDS:
        raise AssessmentError("ER-CST witness thresholds differ")
    if checkpoint.get("training_contract") != TRAINING_CONTRACT:
        raise AssessmentError("ER-CST witness training contract differs")
    if (
        checkpoint.get("board_source_commit") != BOARD_SOURCE_COMMIT
        or checkpoint.get("board_report_sha256") != BOARD_REPORT_SHA256
        or evidence.get("board_report_sha256") != BOARD_REPORT_SHA256
        or ledger.get("board_report_sha256") != BOARD_REPORT_SHA256
    ):
        raise AssessmentError("ER-CST witness board identity differs")
    checkpoint_arms = checkpoint.get("arms")
    evidence_arms = evidence.get("arms")
    if (
        not isinstance(checkpoint_arms, Mapping)
        or set(checkpoint_arms) != ARMS
        or not isinstance(evidence_arms, Mapping)
        or set(evidence_arms) != ARMS
    ):
        raise AssessmentError("ER-CST witness arm set differs")
    parameters = checkpoint.get("parameters")
    parameter_exact = isinstance(parameters, Mapping) and all(
        int(parameters.get(name, -1)) == expected
        for name, expected in EXPECTED_PARAMETERS.items()
    )
    metrics = {name: recompute_arm(evidence_arms[name]) for name in sorted(ARMS)}
    gates = compute_gates(metrics, checkpoint)
    parent_receipt = checkpoint.get("parent_receipt")
    declared_names = (
        set(parent_receipt.get("trainable_names", []))
        if isinstance(parent_receipt, Mapping)
        else set()
    )
    checkpoint_state_exact = (
        len(declared_names) == 103
        and isinstance(parent_receipt, Mapping)
        and parent_receipt.get("direct_rule_classifier_removed") is True
        and parent_receipt.get("card_path_shared_record_gradient") is False
        and all(
            isinstance(arm.get("compiler_trainable_state"), Mapping)
            and set(arm["compiler_trainable_state"]) == declared_names
            and all(
                isinstance(value, torch.Tensor)
                for value in arm["compiler_trainable_state"].values()
            )
            and arm.get("fit", {}).get("arm") == name
            and arm.get("fit", {}).get("updates") == TRAINING_CONTRACT["updates"]
            for name, arm in checkpoint_arms.items()
        )
    )
    source_commit = checkpoint.get("scientific_source_commit")
    source_exact = (
        type(source_commit) is str
        and len(source_commit) == 40
        and report.get("scientific_source_commit") == source_commit
        and evidence.get("scientific_source_commit") == source_commit
        and ledger.get("scientific_source_commit") == source_commit
        and isinstance(checkpoint.get("source_manifest"), Mapping)
        and checkpoint["source_manifest"].get("commit") == source_commit
    )
    artifact_exact = (
        report["artifacts"]["checkpoint_sha256"] == hashes["checkpoint"]
        and report["artifacts"]["evidence_sha256"] == hashes["evidence"]
        and report["artifacts"]["development_ledger_sha256"] == hashes["ledger"]
        and evidence.get("checkpoint_sha256") == hashes["checkpoint"]
    )
    custody_exact = (
        checkpoint.get("development_accesses") == 0
        and checkpoint.get("confirmation_accesses") == 0
        and evidence.get("development_accesses") == 1
        and evidence.get("confirmation_accesses") == 0
        and ledger.get("split") == DEVELOPMENT_SPLIT
        and ledger.get("access_number") == 1
        and report.get("custody", {}).get("development_accesses") == 1
        and report.get("custody", {}).get("confirmation_accesses") == 0
    )
    assessor_gates = {
        "artifact_hashes_match": artifact_exact,
        "parameter_certificate_exact": parameter_exact,
        "checkpoint_trainable_states_exact": checkpoint_state_exact,
        "scientific_source_binding_exact": source_exact,
        "pilot_metrics_match_independent_recomputation": metric_equal(
            report.get("metrics"), metrics
        ),
        "pilot_gate_vector_matches_independent_recomputation": report.get("gates")
        == gates,
        "development_custody_exact": custody_exact,
        "immutable_checkpoint_precedes_development": checkpoint.get(
            "development_accesses"
        )
        == 0,
    }
    all_pass = all(gates.values()) and all(assessor_gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": (
            "authorize_one_sealed_confirmation"
            if all_pass
            else "reject_er_cst_witness_equality_v1_1"
        ),
        "all_gates_pass": all_pass,
        "core_gates": gates,
        "assessor_gates": assessor_gates,
        "thresholds": THRESHOLDS,
        "parameters": dict(parameters) if isinstance(parameters, Mapping) else None,
        "metrics": metrics,
        "artifact_sha256": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 0},
        "claim_boundary": report.get("claim_boundary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--access-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing ER-CST witness assessment: {args.output}")
    paths = {
        "report": args.report,
        "checkpoint": args.checkpoint,
        "evidence": args.evidence,
        "ledger": args.access_ledger,
    }
    value = assess(
        load_json(args.report),
        load_torch(args.checkpoint),
        load_torch(args.evidence),
        load_json(args.access_ledger),
        {name: sha256_file(path) for name, path in paths.items()},
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o444)
    print(json.dumps({"decision": value["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
