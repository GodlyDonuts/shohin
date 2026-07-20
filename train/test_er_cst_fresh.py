from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import assess_er_cst_fresh as assessor
from build_er_cst_fresh_board import (
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_cst_fresh import (
    _equality_ablated_bytes,
    arm_rows,
    evaluate_arm,
    fit_certificates,
    loss_batch,
    parse_row,
)
from er_cst_rule_card_adapter import (
    EpisodicRuleCardCompiler,
    TiedRuleCardMotor,
    freeze_to_er_adaptive,
)
from sd_cst import CategoricalStateReader


SMALL_FAMILIES = {
    TRAIN_SPLIT: 8,
    DEVELOPMENT_SPLIT: 8,
    "er_cst_confirmation": 8,
}


def _small_model() -> EpisodicRuleCardCompiler:
    return EpisodicRuleCardCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=512,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=96,
        sinkhorn_steps=4,
        occurrence_ff=64,
    )


@pytest.fixture(scope="module")
def board_rows() -> dict[str, list[dict[str, object]]]:
    splits, report = build_board(seed=765_432_101, families=SMALL_FAMILIES)
    assert report["all_gates_pass"] is True
    return splits


def test_parser_rejects_training_outcomes(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    value = dict(board_rows[TRAIN_SPLIT][0])
    value["oracle"] = {"final_state_id": 0, "answer_role": 0}
    with pytest.raises(ValueError, match="outcome supervision"):
        parse_row(value, TRAIN_SPLIT)


def test_deranged_labels_are_family_consistent_and_not_identity(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    family = board_rows[TRAIN_SPLIT][:4]
    rows = [parse_row(value, TRAIN_SPLIT) for value in family]
    assert len({row.family_id for row in rows}) == 1
    synthetic = [replace(row, rule_cards=(0, 1, 2)) for row in rows]
    changed = arm_rows(synthetic, "family_deranged", seed=99)
    assert {row.rule_cards for row in changed} in ({(1, 2, 0)}, {(2, 0, 1)})


def test_equality_ablation_preserves_offsets_and_removes_witness_equality(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    row = parse_row(board_rows[TRAIN_SPLIT][0], TRAIN_SPLIT)
    changed = _equality_ablated_bytes(row, seed=123)
    assert len(changed) == len(row.program_bytes)
    assert changed != row.program_bytes
    witness_lines = [
        line.split()
        for line in bytes(changed).decode().splitlines()
        if line.startswith(("W", "L"))
    ]
    assert len(witness_lines) == 3
    for tokens in witness_lines:
        values = [tokens[index] for index in (2, 3, 4, 6, 7, 8)]
        assert len(set(values)) == 6


def test_equality_ablation_preserves_family_identity_across_renderers(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    rows = [parse_row(value, TRAIN_SPLIT) for value in board_rows[TRAIN_SPLIT][:4]]
    changed = arm_rows(rows, "equality_ablated", seed=456)

    def signatures(row: object) -> dict[str, tuple[str, ...]]:
        output = {}
        for line in bytes(row.program_bytes).decode().splitlines():
            tokens = line.split()
            if tokens and tokens[0].startswith(("W", "L")):
                output[tokens[0][1:]] = tuple(
                    tokens[index] for index in (2, 3, 4, 6, 7, 8)
                )
        return output

    expected = signatures(changed[0])
    assert all(signatures(row) == expected for row in changed[1:])


def test_real_family_loss_is_finite_and_gradients_do_not_leak(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    torch.manual_seed(41)
    rows = [parse_row(value, TRAIN_SPLIT) for value in board_rows[TRAIN_SPLIT][:4]]
    assert len({row.family_id for row in rows}) == 1
    model = _small_model().train()
    declared = set(freeze_to_er_adaptive(model))
    loss, pieces = loss_batch(model, [rows], torch.device("cpu"))
    assert torch.isfinite(loss)
    assert set(pieces) == {
        "line",
        "binding",
        "initial_pointer",
        "query_pointer",
        "initial",
        "cards",
        "events",
        "halt",
        "query",
        "consistency",
    }
    loss.backward()
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert leaked == []
    assert model.er_rule_permutation_head.weight.grad is not None
    assert model.er_event_card_query_projection.weight.grad is not None


def test_finite_motor_and_reader_certificates_are_exact() -> None:
    torch.manual_seed(51)
    motor = TiedRuleCardMotor()
    reader = CategoricalStateReader()
    receipt = fit_certificates(motor, reader)
    assert receipt["motor_exact"] == receipt["motor_cells"] == 36
    assert receipt["reader_exact"] == receipt["reader_cells"] == 18
    assert not any(parameter.requires_grad for parameter in motor.parameters())
    assert not any(parameter.requires_grad for parameter in reader.parameters())


def test_evaluation_requires_oracle_rows(
    board_rows: dict[str, list[dict[str, object]]],
) -> None:
    row = parse_row(board_rows[TRAIN_SPLIT][0], TRAIN_SPLIT)
    model = _small_model()
    motor = TiedRuleCardMotor()
    reader = CategoricalStateReader()
    with pytest.raises(ValueError, match="scored rows"):
        evaluate_arm(model, motor, reader, [row])


def test_raw_evidence_recomputes_to_the_same_metrics(
    board_rows: dict[str, list[dict[str, object]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        parse_row(value, DEVELOPMENT_SPLIT)
        for value in board_rows[DEVELOPMENT_SPLIT]
    ]
    model = _small_model()
    motor = TiedRuleCardMotor()
    reader = CategoricalStateReader()
    result = evaluate_arm(
        model,
        motor,
        reader,
        rows,
        batch_size=32,
        include_raw=True,
    )
    raw = result.pop("raw")
    monkeypatch.setattr(assessor, "ROWS", 32)
    assert assessor.metric_equal(result, assessor.recompute_arm(raw))
