from __future__ import annotations

import torch

import assess_sd_cst_complete_physical_confirmation as confirmation
from assess_sd_cst_complete_physical_fresh import (
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EVIDENCE_SCHEMA,
    EXECUTOR_SCHEMA,
    EXPECTED_PARAMETERS,
    PACKET_SCHEMA,
    PROTOCOL,
    REQUIRED_ARMS,
    ROWS,
    THRESHOLDS,
    _grouped,
    _summary,
)
from confirm_sd_cst_complete_physical_fresh import (
    ACCESS_SCHEMA,
    CHECKPOINT_SHA256,
    CONFIRMATION_SHA256,
    DEVELOPMENT_ASSESSMENT_SCHEMA,
    DEVELOPMENT_ASSESSMENT_SHA256,
    GATE_CONFIG_SHA256,
    REPORT_SCHEMA,
)


def test_confirmation_assessor_accepts_complete_source_free_capsule(monkeypatch) -> None:
    treatment = torch.ones(ROWS, dtype=torch.bool)
    shuffled = torch.zeros(ROWS, dtype=torch.bool)
    renderer_names = ["a", "b", "c", "d"]
    renderer_index = torch.arange(ROWS).remainder(4).to(torch.uint8)

    monkeypatch.setattr(confirmation, "_hard", lambda arm: (arm, None))

    def packet_fields(prediction, _gold):
        exact = treatment if prediction[0]["arm"] == "treatment" else shuffled
        return {
            name: exact.clone()
            for name in ("initial", "kind", "identity", "amount", "query", "packet")
        }

    monkeypatch.setattr(confirmation, "_packet_fields", packet_fields)
    monkeypatch.setattr(
        confirmation,
        "_pointer_exact",
        lambda predictions, _ranges, _gold: {
            name: (treatment if predictions["arm"] == "treatment" else shuffled).clone()
            for name in ("line", "binding", "initial_entity", "event_entity")
        },
    )
    monkeypatch.setattr(confirmation, "_output_exact", lambda _output, _arm: True)

    arms = {name: {"arm": "other"} for name in REQUIRED_ARMS}
    arms["treatment"] = {"arm": "treatment"}
    arms["row_shuffled_labels"] = {"arm": "shuffled"}
    outputs = {
        name: {
            "final_state": torch.ones(ROWS, dtype=torch.uint8),
            "answer": torch.ones(ROWS, dtype=torch.uint8),
            "state_trajectory": torch.ones((ROWS, 8), dtype=torch.uint8),
            "alive_trajectory": torch.ones((ROWS, 8), dtype=torch.bool),
        }
        for name in REQUIRED_ARMS
    }
    for name in ("treatment", "gold", "post_stop_perturbation"):
        outputs[name] = {
            "final_state": torch.zeros(ROWS, dtype=torch.uint8),
            "answer": torch.zeros(ROWS, dtype=torch.uint8),
            "state_trajectory": torch.zeros((ROWS, 8), dtype=torch.uint8),
            "alive_trajectory": torch.zeros((ROWS, 8), dtype=torch.bool),
        }
    arm_values = {
        "treatment": {
            name: treatment.clone()
            for name in (
                "initial", "kind", "identity", "amount", "query", "packet",
                "line", "binding", "initial_entity", "event_entity", "state",
                "answer", "joint",
            )
        },
        "row_shuffled_labels": {
            name: shuffled.clone()
            for name in (
                "initial", "kind", "identity", "amount", "query", "packet",
                "line", "binding", "initial_entity", "event_entity", "state",
                "answer", "joint",
            )
        },
    }
    metrics = {
        name: {
            "overall": _summary(values),
            "packet_by_renderer": _grouped(values["packet"], renderer_index, renderer_names),
            "joint_by_renderer": _grouped(values["joint"], renderer_index, renderer_names),
        }
        for name, values in arm_values.items()
    }
    controls = {
        name: {
            "state_rate": float(outputs[name]["final_state"].eq(outputs["gold"]["final_state"]).float().mean()),
            "answer_rate": float(outputs[name]["answer"].eq(outputs["gold"]["answer"]).float().mean()),
        }
        for name in REQUIRED_ARMS - {"treatment", "row_shuffled_labels", "gold"}
    }
    core_gates = {
        "development_authorization_exact": True,
        "fit_packet_min_renderer_at_least_99pct": True,
        "packet_overall_at_least_90pct": True,
        "packet_min_renderer_at_least_85pct": True,
        "state_answer_joint_at_least_90pct": True,
        "joint_min_renderer_at_least_85pct": True,
        "all_packet_fields_at_least_95pct": True,
        "all_pointers_at_least_90pct": True,
        "treatment_packet_advantage_at_least_50pp": True,
        "row_shuffled_packet_at_most_25pct": True,
        "gold_executor_exact": True,
        "conditional_execution_exact": True,
        "post_stop_perturbation_invariant": True,
        "shuffled_packet_state_at_most_35pct": True,
        "reset_and_freeze_state_at_most_75pct": True,
        "source_deleted_before_separate_execution": True,
        "complete_system_below_200m": True,
        "frozen_state_unchanged": True,
        "confirmation_one_after_development_one": True,
    }
    hashes = {
        "report": "report-sha",
        "checkpoint": CHECKPOINT_SHA256,
        "gate_config": GATE_CONFIG_SHA256,
        "development_assessment": DEVELOPMENT_ASSESSMENT_SHA256,
        "authorization": "authorization-sha",
        "packets": "packets-sha",
        "evidence": "evidence-sha",
        "executor": "executor-sha",
        "ledger": "ledger-sha",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "thresholds": dict(THRESHOLDS),
        "gates": core_gates,
        "metrics": metrics,
        "controls": controls,
        "artifacts": {
            "checkpoint_sha256": hashes["checkpoint"],
            "authorization_sha256": hashes["authorization"],
            "packet_sha256": hashes["packets"],
            "evidence_sha256": hashes["evidence"],
            "executor_sha256": hashes["executor"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 1,
            "confirmation_ledger": {"sha256": hashes["ledger"]},
        },
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "parameters": dict(EXPECTED_PARAMETERS),
        "arms": {
            name: {"fit": {"train_metrics": {"r": {"rates": {"packet": 1.0}}}, "frozen_parent_unchanged": True}}
            for name in ("treatment", "row_shuffled_labels")
        },
    }
    result = confirmation.assess_confirmation(
        report=report,
        checkpoint=checkpoint,
        config={"schema": CONFIG_SCHEMA, "protocol": PROTOCOL, "thresholds": dict(THRESHOLDS)},
        development_assessment={
            "schema": DEVELOPMENT_ASSESSMENT_SCHEMA,
            "protocol": PROTOCOL,
            "decision": "authorize_one_sealed_confirmation",
            "all_gates_pass": True,
        },
        packets={"schema": PACKET_SCHEMA, "arms": arms},
        evidence={
            "schema": EVIDENCE_SCHEMA,
            "pointers": {"treatment": {"arm": "treatment"}, "row_shuffled_labels": {"arm": "shuffled"}},
            "pointer_ranges": {},
            "renderer_index": renderer_index,
            "renderer_names": renderer_names,
            "source_poison_bit_identical": {"treatment": True, "row_shuffled_labels": True},
        },
        executor={"schema": EXECUTOR_SCHEMA, "outputs": outputs},
        ledger={
            "schema": ACCESS_SCHEMA,
            "protocol": PROTOCOL,
            "split": "sd_cst_confirmation",
            "split_sha256": CONFIRMATION_SHA256,
            "access_number": 1,
        },
        hashes=hashes,
    )
    assert result["all_gates_pass"] is True
    assert result["decision"] == "confirm_complete_physical_fresh_v1_3"
