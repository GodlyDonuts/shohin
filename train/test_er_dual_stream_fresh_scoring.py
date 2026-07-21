from __future__ import annotations

from functools import lru_cache
import re

import torch

from build_er_dual_stream_fresh_board import build_board
from build_er_relation_tensor_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
)
from er_dual_stream_fresh_scoring import (
    NEUTRAL_PATTERN,
    SEMANTIC_KEYS,
    alpha_recode_row,
    arm_rows,
    distractor_rotate_row,
    evaluate_fresh_treatment,
    evaluate_source_free,
    invariance_metrics,
    record_reindex_row,
    source_free_row,
)
from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from er_relation_tensor_training import RelationTensorRow, parse_row


@lru_cache(maxsize=1)
def _fixture() -> RelationTensorRow:
    splits, _ = build_board(
        seed=7781,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    return parse_row(splits[TRAIN_SPLIT][0], TRAIN_SPLIT)


@lru_cache(maxsize=1)
def _scored_fixture() -> RelationTensorRow:
    splits, _ = build_board(
        seed=7781,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    return parse_row(splits[DEVELOPMENT_SPLIT][0], DEVELOPMENT_SPLIT)


def _tiny_model() -> DualStreamRelationCompiler:
    return DualStreamRelationCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=1024,
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
        max_line_bytes=128,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    )


def _tokens(row: RelationTensorRow) -> set[bytes]:
    return set(NEUTRAL_PATTERN.findall(bytes(row.program_bytes))) | set(
        NEUTRAL_PATTERN.findall(bytes(row.query_bytes))
    )


def test_alpha_recode_covers_program_and_query_without_changing_width() -> None:
    row = _fixture()
    recoded = alpha_recode_row(row, "fixture-alpha")
    assert len(recoded.program_bytes) == len(row.program_bytes)
    assert len(recoded.query_bytes) == len(row.query_bytes)
    assert bytes(recoded.program_bytes) != bytes(row.program_bytes)
    assert bytes(recoded.query_bytes) != bytes(row.query_bytes)
    assert len(_tokens(recoded)) == len(_tokens(row))
    assert all(re.fullmatch(rb"z[0-9a-z]{5}", token) for token in _tokens(recoded))


def test_distractor_rotation_changes_only_irrelevant_names() -> None:
    row = _fixture()
    rotated = distractor_rotate_row(row)
    assert bytes(rotated.program_bytes) != bytes(row.program_bytes)
    assert bytes(rotated.query_bytes) != bytes(row.query_bytes)
    assert row.initial_order == rotated.initial_order
    assert row.relation_rows == rotated.relation_rows
    assert row.event_cards == rotated.event_cards
    assert row.event_halt == rotated.event_halt
    assert row.query_position == rotated.query_position


def test_source_free_collapses_identity_but_preserves_shape() -> None:
    row = _fixture()
    source_free = source_free_row(row)
    assert _tokens(source_free) == {b"z00000"}
    assert len(source_free.program_bytes) == len(row.program_bytes)
    assert len(source_free.query_bytes) == len(row.query_bytes)


def test_fresh_matched_controls_preserve_neutral_width_contract() -> None:
    row = _fixture()
    equality = arm_rows([row], "equality_ablated", 91)[0]
    assert len(equality.program_bytes) == len(row.program_bytes)
    assert equality.relation_rows == row.relation_rows
    assert all(token.startswith(b"z") for token in _tokens(equality))
    assert any(
        bytes(equality.program_bytes)[start:end]
        != bytes(row.program_bytes)[start:end]
        for spans in equality.witness_after_ranges[: equality.rule_count]
        for start, end in spans
    )


def test_record_reindex_preserves_line_multiset_and_invariance_scoring() -> None:
    row = _fixture()
    original = bytes(row.program_bytes).decode().splitlines()
    rule = bytes(record_reindex_row(row, rule_only=True).program_bytes).decode().splitlines()
    physical = bytes(
        record_reindex_row(row, rule_only=False).program_bytes
    ).decode().splitlines()
    assert sorted(rule) == sorted(original)
    assert physical == list(reversed(original))

    hard = {"field": torch.tensor([[1], [2]], dtype=torch.int16)}
    semantic = {
        key: torch.tensor([[1], [2]], dtype=torch.int16) for key in SEMANTIC_KEYS
    }
    metrics = invariance_metrics(
        hard, hard, hard, semantic, semantic, semantic
    )
    assert all(value["exact"] == value["rows"] == 2 for value in metrics.values())


def test_fresh_evaluation_path_emits_raw_counterfactual_evidence() -> None:
    model = _tiny_model()
    treatment, raw, invariant_raw = evaluate_fresh_treatment(
        model, [_scored_fixture()], batch_size=1
    )
    source_free, source_free_raw = evaluate_source_free(
        model, [_scored_fixture()], batch_size=1
    )
    assert treatment["overall"]["joint"]["rows"] == 1
    assert source_free["overall"]["joint"]["rows"] == 1
    assert raw["pred_relations"].shape[0] == 1
    assert source_free_raw["pred_relations"].shape[0] == 1
    assert set(invariant_raw) == {
        "canonical_hard",
        "alpha_hard",
        "distractor_hard",
        "canonical_semantic",
        "rule_reindex",
        "physical_reindex",
    }
