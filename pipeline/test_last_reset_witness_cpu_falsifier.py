#!/usr/bin/env python3
"""Hostile tests for the calibrated Last-Reset Witness Attention CPU audit."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from pipeline import last_reset_witness_cpu_falsifier as lrwa  # noqa: E402


@pytest.fixture(scope="module")
def mechanics_evidence() -> dict[str, object]:
    return lrwa.build_mechanics_evidence()


@pytest.fixture(scope="module")
def mechanics_summary(mechanics_evidence: dict[str, object]) -> dict[str, object]:
    return lrwa.recompute_mechanics_from_evidence(mechanics_evidence)


@pytest.fixture(scope="module")
def full_report() -> dict[str, object]:
    return lrwa.build_report()


def test_exact_deployment_budget_and_all_controls_match() -> None:
    budget = lrwa.deployment_budget()
    assert budget["base_parameters"] == 125_081_664
    assert budget["compiler_parameters"] == 21_524_484
    assert budget["motor_parameters"] == 1_195_020
    assert budget["added_parameters"] == 22_719_504
    assert budget["total_parameters"] == 147_801_168
    assert budget["strict_cap"] == 150_000_000
    assert budget["remaining_parameters"] == 2_198_832
    assert budget["strictly_below_cap"]
    assert budget["controls_exactly_parameter_matched"]
    assert budget["control_parameter_counts"] == {
        "last_reset_witness": 22_719_504,
        "serial_recurrent": 22_719_504,
        "dense_attention": 22_719_504,
    }


def test_exhaustive_reset_words_cover_every_query_and_match_serial(
    mechanics_summary: dict[str, object],
) -> None:
    reset = mechanics_summary["reset_words"]
    assert reset["word_case_count"] == 177_144
    assert reset["query_observation_count"] == 1_860_042
    assert reset["witness_serial_exact"] == 1_860_042
    assert reset["all_query_positions_exact"]
    assert reset["board_sha256"] == lrwa.FROZEN_RESET_WORD_BOARD_SHA256


def test_all_400_raw_add_sub_cells_recompute(
    mechanics_summary: dict[str, object],
) -> None:
    local = mechanics_summary["local_cells"]
    assert local == {
        "cell_count": 400,
        "status_exact": 400,
        "transition_exact": 400,
        "all_400_exact": True,
        "board_sha256": lrwa.FROZEN_LOCAL_CELL_BOARD_SHA256,
        "status_scope": "finite_mechanics_label_only",
    }
    assert lrwa.status_for_raw_cell("ADD", 4, 5) == "P"
    assert lrwa.status_for_raw_cell("SUB", 7, 7) == "P"
    assert lrwa.raw_local_transition("ADD", 9, 9, 1) == (9, 1)
    assert lrwa.raw_local_transition("SUB", 0, 9, 1) == (0, 1)


def test_toggle_truth_table_proves_reset_algebra_boundary(
    mechanics_summary: dict[str, object],
) -> None:
    toggle = mechanics_summary["toggle_negative_control"]
    assert toggle["truth_table_cases"] == 6
    assert toggle["candidate_alias_accuracy"] == {"K": 0.5, "P": 0.0, "G": 0.5}
    assert toggle["best_candidate_accuracy"] == 0.5
    assert toggle["candidate_at_most_75_percent"]
    assert toggle["four_function_recurrent_accuracy"] == 1.0
    assert toggle["recurrent_exact"]
    assert toggle["board_sha256"] == lrwa.FROZEN_TOGGLE_BOARD_SHA256


def test_all_frozen_interventions_are_raw_evidence_backed(
    mechanics_summary: dict[str, object],
) -> None:
    interventions = mechanics_summary["interventions"]
    assert interventions["position_cases"] == 19_674
    assert interventions["position_changed"] == 9_760
    assert interventions["gate_shuffle_changed"] == 5_458
    assert interventions["value_shuffle_changed"] == 9_900
    assert interventions["kg_donor_cases"] == 400
    assert interventions["kg_donor_exact"] == 400
    assert interventions["kg_donor_selective"] == 400
    assert interventions["same_status_sham_cases"] == 1_200
    assert interventions["same_status_sham_exact"] == 1_200
    assert interventions["shadowed_sham_cases"] == 56
    assert interventions["shadowed_sham_exact"] == 56
    assert interventions["generated_prefix_cases"] == 50
    assert interventions["generated_prefix_exact"] == 50
    assert interventions["board_sha256"] == lrwa.FROZEN_INTERVENTION_BOARD_SHA256
    assert mechanics_summary["all_gates_pass"]


def test_missing_or_mutated_raw_mechanics_evidence_fails_closed(
    mechanics_evidence: dict[str, object],
) -> None:
    missing = dict(mechanics_evidence)
    missing_reset = list(missing["reset_words"])
    missing_reset.pop()
    missing["reset_words"] = missing_reset
    with pytest.raises((lrwa.FalsifierError, ValueError)):
        lrwa.recompute_mechanics_from_evidence(missing)

    mutated = dict(mechanics_evidence)
    mutated_reset = list(mutated["reset_words"])
    first = list(mutated_reset[0])
    first[3] = "11" if first[3] != "11" else "00"
    mutated_reset[0] = first
    mutated["reset_words"] = mutated_reset
    with pytest.raises(lrwa.FalsifierError, match="does not recompute"):
        lrwa.recompute_mechanics_from_evidence(mutated)


def test_learning_splits_and_raw_input_contract_are_frozen() -> None:
    train, evaluations = lrwa.build_frozen_learning_splits()
    assert lrwa.learning_split_sha256(train) == lrwa.FROZEN_TRAIN_SPLIT_SHA256
    assert {
        width: lrwa.learning_split_sha256(examples)
        for width, examples in evaluations.items()
    } == lrwa.FROZEN_EVAL_SPLIT_SHA256
    tensors = lrwa.tensorize_examples(train[:8])
    assert tensors["features"].shape[-1] == 24
    assert tensors["features"].device.type == "cpu"
    assert set(tensors) == {"features", "queries", "initial", "targets"}
    assert tensors["targets"].min().item() >= 0
    assert tensors["targets"].max().item() < 12

    forward_source = inspect.getsource(lrwa.ScaledRoutingArm.forward)
    assert "status_for_raw_cell" not in forward_source
    assert "raw_local_transition" not in forward_source
    assert tuple(inspect.signature(lrwa.ScaledRoutingArm.forward).parameters) == (
        "self",
        "features",
        "queries",
        "initial",
    )


def test_scaled_arms_are_exactly_parameter_matched_and_flop_matched() -> None:
    parameters = lrwa.scaled_parameter_audit()
    assert parameters["expected_parameters"] == 2_396
    assert parameters["arm_parameter_counts"] == {
        "witness": 2_396,
        "serial": 2_396,
        "dense": 2_396,
    }
    assert parameters["exactly_matched"]
    flops = lrwa.scaled_flop_audit(32)
    assert flops["within_one_percent"]
    assert flops["max_min_ratio"] == pytest.approx(1.0010013142249201)


def test_every_scaled_parameter_participates_in_cpu_forward() -> None:
    examples = lrwa.build_learning_split(
        split="gradient-smoke", widths=(2, 3), size=16, seed=91
    )
    batch = lrwa.tensorize_examples(examples)
    for arm in ("witness", "serial", "dense"):
        torch.manual_seed(7)
        model = lrwa.ScaledRoutingArm(arm, 32)
        logits = model(batch["features"], batch["queries"], batch["initial"])
        loss = torch.nn.functional.cross_entropy(logits, batch["targets"])
        loss.backward()
        assert logits.device.type == "cpu"
        assert logits.shape == (16, 12)
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_bounded_learning_is_deterministic_and_non_promotional() -> None:
    config = lrwa.LearningConfig(
        data_seed=303,
        train_size=96,
        eval_size_per_width=32,
        train_widths=(2, 3),
        eval_widths=(2, 4),
        model_seeds=(11,),
        updates=4,
        batch_size=32,
        learning_rate=0.003,
        weight_decay=0.0001,
        hidden_dimension=32,
    )
    first = lrwa.run_scaled_learning(config)
    second = lrwa.run_scaled_learning(config)
    assert lrwa.canonical_json_bytes(first) == lrwa.canonical_json_bytes(second)
    assert len(first["raw_eval_rows"]) == 192
    assert not first["gpu_launch_authorized"]
    assert not first["architecture_promotion_authorized"]
    assert first["scope"] == "calibrated_exploratory_audit_not_preregistered_evidence"
    assert first["calibration_preceded_contract_freeze"]
    assert not first["outcome_naive_preregistration_claimed"]
    assert first["input_contract"]["host_kpg_input_count"] == 0
    assert first["input_contract"]["generated_prefix_input_count"] == 0
    assert first["input_contract"]["result_tape_slots"] == 0
    assert first["input_contract"]["generated_kv_state_bytes"] == 0


def test_default_full_report_recomputes_from_raw_cases_and_rejects_controls(
    full_report: dict[str, object],
) -> None:
    lrwa.validate_report(full_report)
    assert full_report["protocol_id"] == "R12-LRWA-CPU-v2-calibrated"
    assert full_report["status"] == "CALIBRATED_EXPLORATORY_REJECTION"
    assert full_report["all_audit_integrity_gates_pass"]
    assert not full_report["gpu_launch_authorized"]
    assert not full_report["architecture_promotion_authorized"]
    learning = full_report["scaled_learning"]
    assert learning["decision"] == "CALIBRATED_EXPLORATORY_REJECTION"
    assert learning["dense_within_two_percentage_points"]
    assert not learning["scaled_candidate_gate_passed"]
    assert learning["decision_inputs"]["exact_parameter_matching"]
    assert learning["decision_inputs"]["heuristic_flop_ratio_at_most_1_01"]
    assert not learning["flop_audit"]["executed_graph_measurement"]
    assert not learning["flop_audit"]["hardware_flops_claimed"]
    assert len(learning["raw_eval_rows"]) == 9_216
    assert learning["raw_eval_rows_sha256"] == (
        "15ed95d8f12c77d6e16f41481768f006f06aaaa3a9a840fd1c61b0b1c33849bd"
    )
    bindings = full_report["source_bindings"]
    assert not bindings["trusted_timestamp_claimed"]
    assert not bindings["immutable_source_claimed"]
    assert set(bindings["files"]) == set(lrwa.SOURCE_BINDING_PATHS)


def test_summary_rewrite_with_fresh_content_hash_still_fails(
    full_report: dict[str, object],
) -> None:
    forged = dict(full_report)
    recomputed = dict(forged["recomputed_mechanics"])
    reset = dict(recomputed["reset_words"])
    reset["witness_serial_exact"] = 0
    recomputed["reset_words"] = reset
    forged["recomputed_mechanics"] = recomputed
    forged.pop("report_content_sha256")
    forged["report_content_sha256"] = lrwa.sha256_bytes(
        lrwa.canonical_json_bytes(forged)
    )
    with pytest.raises(lrwa.FalsifierError, match="does not independently recompute"):
        lrwa.validate_report(forged)


def _refresh_report_hash(report: dict[str, object]) -> None:
    report.pop("report_content_sha256", None)
    report["report_content_sha256"] = lrwa.sha256_bytes(
        lrwa.canonical_json_bytes(report)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "decision",
        "budget_999m",
        "split_hash",
        "initial_model_hash",
        "model_hash",
        "loss_deletion",
        "nested_authorization",
        "raw_reorder",
        "target_and_row_aggregate_rewrite",
        "updates",
        "learning_rate",
    ),
)
def test_reviewer_hostile_mutations_fail_after_self_hash_refresh(
    full_report: dict[str, object], mutation: str
) -> None:
    forged = copy.deepcopy(full_report)
    learning = forged["scaled_learning"]
    if mutation == "decision":
        learning["decision"] = "CALIBRATED_EXPLORATORY_NOMINATION_SIGNAL_ONLY"
    elif mutation == "budget_999m":
        forged["deployment_budget"]["total_parameters"] = 999_000_000
    elif mutation == "split_hash":
        learning["split_commitments"]["train_sha256"] = "0" * 64
    elif mutation == "initial_model_hash":
        learning["runs"][0]["initial_state_sha256"] = "2" * 64
    elif mutation == "model_hash":
        learning["runs"][0]["final_state_sha256"] = "1" * 64
    elif mutation == "loss_deletion":
        run = learning["runs"][0]
        losses = run["loss_trace"]
        losses.pop()
        run["final_loss"] = losses[-1]
        run["loss_trace_sha256"] = lrwa._row_stream_hash(losses)
    elif mutation == "nested_authorization":
        learning["gpu_launch_authorized"] = True
        learning["architecture_promotion_authorized"] = True
    elif mutation == "raw_reorder":
        rows = learning["raw_eval_rows"]
        rows[0], rows[1] = rows[1], rows[0]
        learning["raw_eval_rows_sha256"] = lrwa._row_stream_hash(rows)
        rows_per_run = (
            len(lrwa.DEFAULT_LEARNING_CONFIG.eval_widths)
            * lrwa.DEFAULT_LEARNING_CONFIG.eval_size_per_width
        )
        learning["runs"][0]["prediction_rows_sha256"] = lrwa._row_stream_hash(
            rows[:rows_per_run]
        )
    elif mutation == "target_and_row_aggregate_rewrite":
        rows = learning["raw_eval_rows"]
        row = list(rows[0])
        row[4] = (int(row[4]) + 1) % 12
        row[5] = (int(row[5]) + 1) % 12
        rows[0] = row
        learning["raw_eval_rows_sha256"] = lrwa._row_stream_hash(rows)
        learning["recomputed_eval"] = lrwa.recompute_learning_rows(
            rows, lrwa.DEFAULT_LEARNING_CONFIG
        )
        rows_per_run = (
            len(lrwa.DEFAULT_LEARNING_CONFIG.eval_widths)
            * lrwa.DEFAULT_LEARNING_CONFIG.eval_size_per_width
        )
        learning["runs"][0]["prediction_rows_sha256"] = lrwa._row_stream_hash(
            rows[:rows_per_run]
        )
    elif mutation == "updates":
        learning["config"]["updates"] = 95
    elif mutation == "learning_rate":
        learning["config"]["learning_rate"] = 0.03
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(mutation)
    _refresh_report_hash(forged)
    with pytest.raises(lrwa.FalsifierError):
        lrwa.validate_report(forged)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("protocol_id", "R12-LRWA-CPU-v1"),
        ("status", "PREREGISTERED_GO"),
        ("claim_boundary", "canonical evidence"),
        ("gpu_launch_authorized", True),
        ("architecture_promotion_authorized", True),
    ),
)
def test_top_level_contract_mutations_fail_after_self_hash_refresh(
    full_report: dict[str, object], field: str, value: object
) -> None:
    forged = copy.deepcopy(full_report)
    forged[field] = value
    _refresh_report_hash(forged)
    with pytest.raises(lrwa.FalsifierError):
        lrwa.validate_report(forged)


def test_source_resource_gate_and_scientific_summary_mutations_fail(
    full_report: dict[str, object],
) -> None:
    mutations = []
    source = copy.deepcopy(full_report)
    source["source_bindings"]["files"][lrwa.SOURCE_BINDING_PATHS[0]]["sha256"] = (
        "f" * 64
    )
    mutations.append(source)

    resource = copy.deepcopy(full_report)
    resource["resource_contract"]["h100_jobs_launched"] = 1
    mutations.append(resource)

    gate = copy.deepcopy(full_report)
    gate["gates"]["no_h100_launch"] = False
    gate["all_audit_integrity_gates_pass"] = False
    mutations.append(gate)

    median = copy.deepcopy(full_report)
    median["scaled_learning"]["median_accuracy"]["witness"][32] = 1.0
    mutations.append(median)

    delta = copy.deepcopy(full_report)
    delta["scaled_learning"]["per_seed_witness_minus_serial"][0] = 1.0
    mutations.append(delta)

    dense = copy.deepcopy(full_report)
    dense["scaled_learning"]["dense_within_two_percentage_points"] = False
    mutations.append(dense)

    parameter = copy.deepcopy(full_report)
    parameter["scaled_learning"]["parameter_audit"]["arm_parameter_counts"][
        "witness"
    ] = 2_397
    mutations.append(parameter)

    heuristic = copy.deepcopy(full_report)
    heuristic["scaled_learning"]["flop_audit"]["max_min_ratio"] = 1.5
    heuristic["scaled_learning"]["flop_audit"]["within_one_percent"] = False
    mutations.append(heuristic)

    for forged in mutations:
        _refresh_report_hash(forged)
        with pytest.raises(lrwa.FalsifierError):
            lrwa.validate_report(forged)


def test_report_publication_is_canonical_read_only_and_no_overwrite(
    tmp_path: Path, full_report: dict[str, object]
) -> None:
    destination = tmp_path / "last_reset_witness_report.json"
    written_hash = lrwa.write_report_once(destination, full_report)
    payload = destination.read_bytes()
    assert written_hash == lrwa.sha256_bytes(payload)
    assert payload == lrwa.report_bytes(full_report)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    parsed = json.loads(payload)
    lrwa.validate_report(parsed)
    with pytest.raises(FileExistsError):
        lrwa.write_report_once(destination, full_report)


def test_module_has_no_network_subprocess_gpu_or_generated_state_path() -> None:
    source = Path(lrwa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert {"requests", "socket", "subprocess", "urllib"}.isdisjoint(imported_roots)
    assert ".cuda(" not in source
    assert 'device="cuda"' not in source
    assert "generated_kv" not in inspect.getsource(lrwa.ScaledRoutingArm)
    assert "result_tape" not in inspect.getsource(lrwa.ScaledRoutingArm)


def test_calibrated_contract_binds_budget_hashes_controls_and_claim_boundary() -> None:
    prereg = (
        Path(lrwa.__file__).resolve().parents[1]
        / "R12_LAST_RESET_WITNESS_ATTENTION_PREREG.md"
    ).read_text(encoding="ascii")
    for token in (
        "125,081,664",
        "21,524,484",
        "1,195,020",
        "22,719,504",
        "147,801,168",
        "2,198,832",
        lrwa.FROZEN_RESET_WORD_BOARD_SHA256,
        lrwa.FROZEN_LOCAL_CELL_BOARD_SHA256,
        lrwa.FROZEN_TOGGLE_BOARD_SHA256,
        lrwa.FROZEN_INTERVENTION_BOARD_SHA256,
        lrwa.FROZEN_TRAIN_SPLIT_SHA256,
        "not a new computational primitive",
        "CALIBRATED_EXPLORATORY_REJECTION",
        "not an outcome-naive",
        "must never be represented as preregistered",
        "canonical evidence",
        "not an executed-graph measurement",
        "current source/prereg/test bytes",
        "do not establish a trusted timestamp",
        "No H100 launch",
        "dense attention is within 2pp",
        "serial wins or ties",
    ):
        assert token in prereg
