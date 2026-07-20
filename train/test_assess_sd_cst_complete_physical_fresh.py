from __future__ import annotations

import torch

from assess_sd_cst_complete_physical_fresh import (
    ACCESS_SCHEMA,
    ASSESSMENT_SCHEMA,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EVIDENCE_SCHEMA,
    EXECUTOR_SCHEMA,
    EXPECTED_PARAMETERS,
    PACKET_SCHEMA,
    PROTOCOL,
    REPORT_SCHEMA,
    REQUIRED_ARMS,
    ROWS,
    THRESHOLDS,
    _grouped,
    _hard,
    _output_exact,
    _packet_fields,
    _pointer_exact,
    _summary,
    assess,
)
from assess_sd_cst_projected_mechanics import packet_arm, semantic_rollout
from pilot_sd_cst_complete_physical_fresh import _minimum_fit_packet
from sd_cst import STOP_KIND, HardLateQuery, HardProgramTape


def _gold() -> tuple[HardProgramTape, HardLateQuery]:
    kind = torch.zeros((ROWS, 8), dtype=torch.uint8)
    kind[:, 3] = STOP_KIND
    return (
        HardProgramTape(
            torch.zeros(ROWS, dtype=torch.uint8),
            kind,
            torch.zeros((ROWS, 8), dtype=torch.uint8),
            torch.zeros((ROWS, 8), dtype=torch.uint8),
        ),
        HardLateQuery(torch.zeros(ROWS, dtype=torch.uint8)),
    )


def test_independent_packet_and_pointer_recomputation() -> None:
    gold = _gold()
    fields = _packet_fields(gold, gold)
    assert all(bool(value.all()) for value in fields.values())
    predictions = {
        "line": torch.ones((ROWS, 9), dtype=torch.long),
        "binding": torch.ones((ROWS, 3), dtype=torch.long),
        "initial_entity": torch.ones((ROWS, 3), dtype=torch.long),
        "event_entity": torch.ones((ROWS, 8), dtype=torch.long),
    }
    ranges = {
        name: torch.tensor([[[1, 2]] * value.shape[1]] * ROWS, dtype=torch.long)
        for name, value in predictions.items()
    }
    ranges["event_entity"][:, 3] = 0
    exact = _pointer_exact(predictions, ranges, gold[0])
    assert all(bool(value.all()) for value in exact.values())
    predictions["binding"][0, 0] = 2
    assert not bool(_pointer_exact(predictions, ranges, gold[0])["binding"][0])


def test_independent_executor_semantics() -> None:
    tape, query = _gold()
    arm = packet_arm(tape, query)
    parsed_tape, parsed_query = _hard(arm)
    expected = semantic_rollout(parsed_tape, parsed_query)
    output = dict(
        zip(
            ("final_state", "answer", "state_trajectory", "alive_trajectory"),
            expected,
            strict=True,
        )
    )
    assert _output_exact(output, arm)
    output["final_state"] = output["final_state"].clone()
    output["final_state"][0] = 1
    assert not _output_exact(output, arm)


def test_fit_minimum_uses_nested_renderer_metrics() -> None:
    fit = {
        "seed": 17,
        "updates": 3_000,
        "history": [],
        "train_metrics": {
            "renderer-a": {"rates": {"packet": 1.0}},
            "renderer-b": {"rates": {"packet": 0.992}},
        },
    }
    assert _minimum_fit_packet(fit) == 0.992


def test_complete_source_free_assessor_accepts_perfect_capsule() -> None:
    gold_tape, gold_query = _gold()
    gold_arm = packet_arm(gold_tape, gold_query)
    wrong_tape = HardProgramTape(
        torch.ones_like(gold_tape.initial_state),
        gold_tape.event_kind.clone(),
        gold_tape.event_identity.clone(),
        gold_tape.amount.clone(),
    )
    wrong_arm = packet_arm(wrong_tape, gold_query)
    arms = {name: gold_arm for name in REQUIRED_ARMS}
    arms["row_shuffled_labels"] = wrong_arm

    expected = semantic_rollout(gold_tape, gold_query)
    exact_output = dict(
        zip(
            ("final_state", "answer", "state_trajectory", "alive_trajectory"),
            expected,
            strict=True,
        )
    )
    wrong_output = {name: value.clone() for name, value in exact_output.items()}
    wrong_output["final_state"] = (wrong_output["final_state"] + 1) % 5
    wrong_output["answer"] = (wrong_output["answer"] + 1) % 3
    outputs = {
        name: {key: value.clone() for key, value in wrong_output.items()}
        for name in REQUIRED_ARMS
    }
    for name in ("treatment", "gold", "post_stop_perturbation"):
        outputs[name] = {key: value.clone() for key, value in exact_output.items()}

    pointer_ranges = {
        name: torch.tensor([[[1, 2]] * slots] * ROWS, dtype=torch.long)
        for name, slots in {
            "line": 9,
            "binding": 3,
            "initial_entity": 3,
            "event_entity": 8,
        }.items()
    }
    treatment_pointers = {
        name: torch.ones((ROWS, ranges.shape[1]), dtype=torch.long)
        for name, ranges in pointer_ranges.items()
    }
    shuffled_pointers = {
        name: torch.full((ROWS, ranges.shape[1]), 2, dtype=torch.long)
        for name, ranges in pointer_ranges.items()
    }
    renderer_names = ["a", "b", "c", "d"]
    renderer_index = torch.arange(ROWS, dtype=torch.long).remainder(4).to(torch.uint8)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "pointers": {
            "treatment": treatment_pointers,
            "row_shuffled_labels": shuffled_pointers,
        },
        "pointer_ranges": pointer_ranges,
        "renderer_names": renderer_names,
        "renderer_index": renderer_index,
        "source_poison_bit_identical": {
            "treatment": True,
            "row_shuffled_labels": True,
        },
    }

    treatment_fields = _packet_fields((gold_tape, gold_query), (gold_tape, gold_query))
    shuffled_fields = _packet_fields((wrong_tape, gold_query), (gold_tape, gold_query))
    treatment_pointer_exact = _pointer_exact(
        treatment_pointers, pointer_ranges, gold_tape
    )
    shuffled_pointer_exact = _pointer_exact(
        shuffled_pointers, pointer_ranges, gold_tape
    )
    arm_values = {}
    for name, fields, pointers in (
        ("treatment", treatment_fields, treatment_pointer_exact),
        ("row_shuffled_labels", shuffled_fields, shuffled_pointer_exact),
    ):
        state = outputs[name]["final_state"].eq(exact_output["final_state"])
        answer = outputs[name]["answer"].eq(exact_output["answer"])
        arm_values[name] = fields | pointers | {
            "state": state,
            "answer": answer,
            "joint": state & answer,
        }
    metrics = {
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
    controls = {
        name: {
            "state_rate": float(
                outputs[name]["final_state"].eq(exact_output["final_state"]).float().mean()
            ),
            "answer_rate": float(
                outputs[name]["answer"].eq(exact_output["answer"]).float().mean()
            ),
        }
        for name in REQUIRED_ARMS - {"treatment", "row_shuffled_labels", "gold"}
    }
    renderer_fit = {
        name: {"rates": {"packet": 1.0}} for name in renderer_names
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "parameters": dict(EXPECTED_PARAMETERS),
        "arms": {
            name: {
                "fit": {
                    "train_metrics": renderer_fit,
                    "frozen_parent_unchanged": True,
                }
            }
            for name in ("treatment", "row_shuffled_labels")
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    hashes = {
        "report": "report-sha",
        "checkpoint": "checkpoint-sha",
        "gate_config": "config-sha",
        "packets": "packet-sha",
        "evidence": "evidence-sha",
        "executor": "executor-sha",
        "ledger": "ledger-sha",
    }
    core_gates = {
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
        "development_one_confirmation_zero": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "thresholds": dict(THRESHOLDS),
        "parameters": dict(EXPECTED_PARAMETERS),
        "metrics": metrics,
        "controls": controls,
        "gates": core_gates,
        "artifacts": {
            "checkpoint_sha256": hashes["checkpoint"],
            "gate_config_sha256": hashes["gate_config"],
            "packet_sha256": hashes["packets"],
            "evidence_sha256": hashes["evidence"],
            "executor_sha256": hashes["executor"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "development_ledger": {"sha256": hashes["ledger"]},
        },
    }
    config = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "thresholds": dict(THRESHOLDS),
        "checkpoint_sha256": hashes["checkpoint"],
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    ledger = {
        "schema": ACCESS_SCHEMA,
        "protocol": PROTOCOL,
        "split": "sd_cst_development",
        "access_number": 1,
    }
    result = assess(
        report=report,
        checkpoint=checkpoint,
        config=config,
        packets={"schema": PACKET_SCHEMA, "arms": arms},
        evidence=evidence,
        executor={"schema": EXECUTOR_SCHEMA, "outputs": outputs},
        ledger=ledger,
        hashes=hashes,
    )
    assert result["schema"] == ASSESSMENT_SCHEMA
    assert result["all_gates_pass"] is True
    assert result["decision"] == "authorize_one_sealed_confirmation"
