#!/usr/bin/env python3
"""Independently assess the complete physical fresh-board development result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from assess_sd_cst_projected_mechanics import semantic_rollout
from build_sd_cst_complete_physical_fresh_board import PROTOCOL
from sd_cst import STOP_KIND, HardLateQuery, HardProgramTape


REPORT_SCHEMA = "r12_sd_cst_complete_physical_fresh_development_report_v1"
ASSESSMENT_SCHEMA = "r12_sd_cst_complete_physical_fresh_assessment_v1"
CHECKPOINT_SCHEMA = "r12_sd_cst_complete_physical_fresh_checkpoint_v1"
CONFIG_SCHEMA = "r12_sd_cst_complete_physical_fresh_gate_config_v1"
EVIDENCE_SCHEMA = "r12_sd_cst_complete_physical_fresh_evidence_v1"
PACKET_SCHEMA = "r12_sd_cst_hard_packet_bundle_v1"
EXECUTOR_SCHEMA = "r12_sd_cst_hard_packet_outputs_v1"
ACCESS_SCHEMA = "r12_sd_cst_complete_physical_fresh_access_v1"
ROWS = 2_048
EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 67_027_474,
    "motor": 19_206,
    "reader": 835,
    "complete_system": 192_129_179,
    "headroom": 7_870_821,
    "trainable": 12_152_855,
}
THRESHOLDS = {
    "fit_packet_min_renderer": 0.99,
    "packet_overall": 0.90,
    "packet_min_renderer": 0.85,
    "state_overall": 0.90,
    "answer_overall": 0.90,
    "joint_overall": 0.90,
    "joint_min_renderer": 0.85,
    "field_overall": 0.95,
    "pointer_overall": 0.90,
    "treatment_packet_advantage": 0.50,
    "row_shuffled_packet_max": 0.25,
    "negative_state_max": 0.35,
    "reset_freeze_state_max": 0.75,
}
REQUIRED_ARMS = {
    "treatment",
    "row_shuffled_labels",
    "gold",
    "uniform",
    "shuffled_packet",
    "reset",
    "freeze",
    "post_stop_perturbation",
    "force_alive_post_stop",
    "query_rotation",
    "initial_state_rotation",
    "event_kind_flip",
    "event_identity_rotation",
    "event_amount_flip",
}


class AssessmentError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise AssessmentError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            AssessmentError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise AssessmentError("top-level JSON value must be an object")
    return value


def load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise AssessmentError(f"torch artifact is not a mapping: {path}")
    return value


def _hard(arm: Mapping[str, Any]) -> tuple[HardProgramTape, HardLateQuery]:
    shapes = {
        "initial_state": (ROWS,),
        "event_kind": (ROWS, 8),
        "event_identity": (ROWS, 8),
        "amount": (ROWS, 8),
        "query": (ROWS,),
    }
    for name, shape in shapes.items():
        value = arm.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise AssessmentError(f"packet arm tensor differs: {name}")
        if value.dtype != torch.uint8:
            raise AssessmentError(f"packet arm dtype differs: {name}")
    return (
        HardProgramTape(
            arm["initial_state"],
            arm["event_kind"],
            arm["event_identity"],
            arm["amount"],
        ),
        HardLateQuery(arm["query"]),
    )


def _packet_fields(
    prediction: tuple[HardProgramTape, HardLateQuery],
    target: tuple[HardProgramTape, HardLateQuery],
) -> dict[str, torch.Tensor]:
    tape, query = prediction
    gold, gold_query = target
    active = gold.event_kind.ne(STOP_KIND)
    values = {
        "initial": tape.initial_state.eq(gold.initial_state),
        "kind": tape.event_kind.eq(gold.event_kind).all(-1),
        "identity": (tape.event_identity.eq(gold.event_identity) | ~active).all(-1),
        "amount": (tape.amount.eq(gold.amount) | ~active).all(-1),
        "query": query.position.eq(gold_query.position),
    }
    values["packet"] = torch.stack(list(values.values())).all(0)
    return values


def _pointer_exact(
    predictions: Mapping[str, torch.Tensor],
    ranges: Mapping[str, torch.Tensor],
    gold: HardProgramTape,
) -> dict[str, torch.Tensor]:
    slots = {"line": 9, "binding": 3, "initial_entity": 3, "event_entity": 8}
    output = {}
    for name, count in slots.items():
        prediction = predictions.get(name)
        target = ranges.get(name)
        if (
            not isinstance(prediction, torch.Tensor)
            or prediction.dtype != torch.int64
            or tuple(prediction.shape) != (ROWS, count)
            or not isinstance(target, torch.Tensor)
            or target.dtype != torch.int64
            or tuple(target.shape) != (ROWS, count, 2)
        ):
            raise AssessmentError(f"pointer evidence differs: {name}")
        exact = (prediction >= target[..., 0]) & (prediction < target[..., 1])
        if name == "event_entity":
            exact |= gold.event_kind.eq(STOP_KIND)
        output[name] = exact.all(-1)
    return output


def _output_exact(
    output: Mapping[str, torch.Tensor],
    arm: Mapping[str, Any],
) -> bool:
    tape, query = _hard(arm)
    expected = semantic_rollout(
        tape,
        query,
        control=str(arm["control"]),
        state_swap=arm["state_swap"],
        swap_after_step=int(arm["swap_after_step"]),
        force_alive=bool(arm["force_alive"]),
    )
    return all(
        isinstance(output.get(name), torch.Tensor) and torch.equal(output[name], target)
        for name, target in zip(
            ("final_state", "answer", "state_trajectory", "alive_trajectory"),
            expected,
            strict=True,
        )
    )


def _summary(values: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "correct": int(value.sum()),
            "rows": int(value.numel()),
            "rate": float(value.float().mean()),
        }
        for name, value in values.items()
    }


def _grouped(
    values: torch.Tensor,
    renderer_index: torch.Tensor,
    renderer_names: list[str],
) -> dict[str, dict[str, Any]]:
    output = {}
    for index, name in enumerate(renderer_names):
        selected = renderer_index.eq(index)
        output[name] = {
            "correct": int(values[selected].sum()),
            "rows": int(selected.sum()),
            "rate": float(values[selected].float().mean()),
        }
    return output


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        if int(left[name]["correct"]) != int(right[name]["correct"]):
            return False
        if int(left[name]["rows"]) != int(right[name]["rows"]):
            return False
        if not math.isclose(
            float(left[name]["rate"]), float(right[name]["rate"]), abs_tol=1e-12
        ):
            return False
    return True


def assess(
    *,
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    packets: Mapping[str, Any],
    evidence: Mapping[str, Any],
    executor: Mapping[str, Any],
    ledger: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("protocol") != PROTOCOL:
        raise AssessmentError("development report identity differs")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise AssessmentError("checkpoint identity differs")
    if config.get("schema") != CONFIG_SCHEMA or config.get("protocol") != PROTOCOL:
        raise AssessmentError("gate config identity differs")
    if packets.get("schema") != PACKET_SCHEMA:
        raise AssessmentError("packet identity differs")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise AssessmentError("evidence identity differs")
    if executor.get("schema") != EXECUTOR_SCHEMA:
        raise AssessmentError("executor identity differs")
    if ledger.get("schema") != ACCESS_SCHEMA or ledger.get("protocol") != PROTOCOL:
        raise AssessmentError("access ledger identity differs")
    if config.get("thresholds") != THRESHOLDS or report.get("thresholds") != THRESHOLDS:
        raise AssessmentError("frozen thresholds differ")

    arms = packets.get("arms")
    outputs = executor.get("outputs")
    if not isinstance(arms, Mapping) or set(arms) != REQUIRED_ARMS:
        raise AssessmentError("packet arm set differs")
    if not isinstance(outputs, Mapping) or set(outputs) != REQUIRED_ARMS:
        raise AssessmentError("executor arm set differs")
    gold = _hard(arms["gold"])
    treatment_fields = _packet_fields(_hard(arms["treatment"]), gold)
    shuffled_fields = _packet_fields(_hard(arms["row_shuffled_labels"]), gold)

    pointer_groups = evidence.get("pointers")
    pointer_ranges = evidence.get("pointer_ranges")
    if not isinstance(pointer_groups, Mapping) or set(pointer_groups) != {
        "treatment",
        "row_shuffled_labels",
    }:
        raise AssessmentError("pointer arm set differs")
    if not isinstance(pointer_ranges, Mapping):
        raise AssessmentError("pointer targets are absent")
    treatment_pointers = _pointer_exact(
        pointer_groups["treatment"], pointer_ranges, gold[0]
    )
    shuffled_pointers = _pointer_exact(
        pointer_groups["row_shuffled_labels"], pointer_ranges, gold[0]
    )
    renderer_index = evidence.get("renderer_index")
    renderer_names = evidence.get("renderer_names")
    if (
        not isinstance(renderer_index, torch.Tensor)
        or renderer_index.dtype != torch.uint8
        or tuple(renderer_index.shape) != (ROWS,)
        or not isinstance(renderer_names, list)
        or len(renderer_names) != 4
        or sorted(set(renderer_names)) != sorted(renderer_names)
        or not all(type(name) is str for name in renderer_names)
    ):
        raise AssessmentError("renderer evidence differs")
    if int(renderer_index.min()) != 0 or int(renderer_index.max()) != 3:
        raise AssessmentError("renderer evidence range differs")

    gold_state = outputs["gold"]["final_state"]
    gold_answer = outputs["gold"]["answer"]
    arm_values = {}
    for name, fields, pointers_for_arm in (
        ("treatment", treatment_fields, treatment_pointers),
        ("row_shuffled_labels", shuffled_fields, shuffled_pointers),
    ):
        state = outputs[name]["final_state"].eq(gold_state)
        answer = outputs[name]["answer"].eq(gold_answer)
        arm_values[name] = (
            fields
            | pointers_for_arm
            | {
                "state": state,
                "answer": answer,
                "joint": state & answer,
            }
        )

    independent_metrics = {
        name: {
            "overall": _summary(values),
            "packet_by_renderer": _grouped(
                values["packet"], renderer_index, renderer_names
            ),
            "joint_by_renderer": _grouped(
                values["joint"], renderer_index, renderer_names
            ),
        }
        for name, values in arm_values.items()
    }
    fit = checkpoint["arms"]["treatment"]["fit"]["train_metrics"]
    fit_min = min(float(item["rates"]["packet"]) for item in fit.values())
    treatment = independent_metrics["treatment"]
    shuffled = independent_metrics["row_shuffled_labels"]
    controls = {
        name: {
            "state_rate": float(
                outputs[name]["final_state"].eq(gold_state).float().mean()
            ),
            "answer_rate": float(
                outputs[name]["answer"].eq(gold_answer).float().mean()
            ),
        }
        for name in REQUIRED_ARMS - {"treatment", "row_shuffled_labels", "gold"}
    }
    source_poison = evidence.get("source_poison_bit_identical")
    if not isinstance(source_poison, Mapping):
        raise AssessmentError("source poison evidence is absent")

    parameter_report = checkpoint.get("parameters")
    parameter_exact = isinstance(parameter_report, Mapping) and all(
        int(parameter_report.get(name, -1)) == value
        for name, value in EXPECTED_PARAMETERS.items()
    )
    artifact_exact = (
        report["artifacts"]["checkpoint_sha256"] == hashes["checkpoint"]
        and report["artifacts"]["gate_config_sha256"] == hashes["gate_config"]
        and report["artifacts"]["packet_sha256"] == hashes["packets"]
        and report["artifacts"]["evidence_sha256"] == hashes["evidence"]
        and report["artifacts"]["executor_sha256"] == hashes["executor"]
        and config["checkpoint_sha256"] == hashes["checkpoint"]
        and report["custody"]["development_ledger"]["sha256"] == hashes["ledger"]
    )
    pilot_metrics_exact = all(
        _metrics_equal(
            report["metrics"][arm]["overall"],
            independent_metrics[arm]["overall"],
        )
        and _metrics_equal(
            report["metrics"][arm]["packet_by_renderer"],
            independent_metrics[arm]["packet_by_renderer"],
        )
        and _metrics_equal(
            report["metrics"][arm]["joint_by_renderer"],
            independent_metrics[arm]["joint_by_renderer"],
        )
        for arm in independent_metrics
    ) and all(
        math.isclose(
            float(report["controls"][name][metric]),
            float(controls[name][metric]),
            abs_tol=1e-12,
        )
        for name in controls
        for metric in ("state_rate", "answer_rate")
    )

    core_gates = {
        "fit_packet_min_renderer_at_least_99pct": fit_min
        >= THRESHOLDS["fit_packet_min_renderer"],
        "packet_overall_at_least_90pct": treatment["overall"]["packet"]["rate"]
        >= THRESHOLDS["packet_overall"],
        "packet_min_renderer_at_least_85pct": min(
            value["rate"] for value in treatment["packet_by_renderer"].values()
        )
        >= THRESHOLDS["packet_min_renderer"],
        "state_answer_joint_at_least_90pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS[f"{name}_overall"]
            for name in ("state", "answer", "joint")
        ),
        "joint_min_renderer_at_least_85pct": min(
            value["rate"] for value in treatment["joint_by_renderer"].values()
        )
        >= THRESHOLDS["joint_min_renderer"],
        "all_packet_fields_at_least_95pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS["field_overall"]
            for name in ("initial", "kind", "identity", "amount", "query")
        ),
        "all_pointers_at_least_90pct": all(
            treatment["overall"][name]["rate"] >= THRESHOLDS["pointer_overall"]
            for name in ("line", "binding", "initial_entity", "event_entity")
        ),
        "treatment_packet_advantage_at_least_50pp": treatment["overall"]["packet"][
            "rate"
        ]
        - shuffled["overall"]["packet"]["rate"]
        >= THRESHOLDS["treatment_packet_advantage"],
        "row_shuffled_packet_at_most_25pct": shuffled["overall"]["packet"]["rate"]
        <= THRESHOLDS["row_shuffled_packet_max"],
        "gold_executor_exact": _output_exact(outputs["gold"], arms["gold"]),
        "conditional_execution_exact": _output_exact(
            outputs["treatment"], arms["treatment"]
        ),
        "post_stop_perturbation_invariant": all(
            torch.equal(
                outputs["treatment"][name], outputs["post_stop_perturbation"][name]
            )
            for name in (
                "final_state",
                "answer",
                "state_trajectory",
                "alive_trajectory",
            )
        ),
        "shuffled_packet_state_at_most_35pct": controls["shuffled_packet"]["state_rate"]
        <= THRESHOLDS["negative_state_max"],
        "reset_and_freeze_state_at_most_75pct": all(
            controls[name]["state_rate"] <= THRESHOLDS["reset_freeze_state_max"]
            for name in ("reset", "freeze")
        ),
        "source_deleted_before_separate_execution": source_poison
        == {"treatment": True, "row_shuffled_labels": True},
        "complete_system_below_200m": parameter_exact
        and int(parameter_report["complete_system"]) < 200_000_000,
        "frozen_state_unchanged": all(
            checkpoint["arms"][name]["fit"]["frozen_parent_unchanged"] is True
            for name in ("treatment", "row_shuffled_labels")
        ),
        "development_one_confirmation_zero": (
            ledger.get("access_number") == 1
            and ledger.get("split") == "sd_cst_development"
            and checkpoint.get("development_accesses") == 0
            and checkpoint.get("confirmation_accesses") == 0
            and config.get("development_accesses") == 0
            and config.get("confirmation_accesses") == 0
            and report["custody"].get("development_accesses") == 1
            and report["custody"].get("confirmation_accesses") == 0
        ),
    }
    assessor_gates = {
        "artifact_hashes_match": artifact_exact,
        "parameter_certificate_exact": parameter_exact,
        "pilot_metrics_match_independent_recomputation": pilot_metrics_exact,
        "pilot_gate_vector_matches_independent_recomputation": report.get("gates")
        == core_gates,
    }
    all_pass = all(core_gates.values()) and all(assessor_gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": (
            "authorize_one_sealed_confirmation"
            if all_pass
            else "reject_complete_physical_fresh_v1"
        ),
        "all_gates_pass": all_pass,
        "core_gates": core_gates,
        "assessor_gates": assessor_gates,
        "thresholds": THRESHOLDS,
        "parameters": dict(parameter_report)
        if isinstance(parameter_report, Mapping)
        else None,
        "metrics": independent_metrics,
        "controls": controls,
        "artifact_sha256": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 0},
        "claim_boundary": (
            "Fresh finite renderer/name transfer into a source-deleted categorical "
            "executor. Passing is not broad natural-language or general reasoning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--access-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing assessment: {args.output}")
    paths = {
        "report": args.report,
        "checkpoint": args.checkpoint,
        "gate_config": args.gate_config,
        "packets": args.packets,
        "evidence": args.evidence,
        "executor": args.executor,
        "ledger": args.access_ledger,
    }
    result = assess(
        report=load_json(args.report),
        checkpoint=load_torch(args.checkpoint),
        config=load_json(args.gate_config),
        packets=load_torch(args.packets),
        evidence=load_torch(args.evidence),
        executor=load_torch(args.executor),
        ledger=load_json(args.access_ledger),
        hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
