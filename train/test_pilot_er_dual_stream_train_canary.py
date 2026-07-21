from __future__ import annotations

from dataclasses import replace

from er_relation_tensor_training import RelationTensorRow
from pilot_er_dual_stream_train_canary import (
    OPAQUE_PATTERN,
    alpha_recode_row,
    score_train_row,
    split_train_families,
)


def _row(family: str, renderer: str = "r0") -> RelationTensorRow:
    source = (
        b"D3 e00000 e00001 e00002 ; I e00001 e00000 e00002\n"
        b"W1 o00000 w00000 w00001 w00002 > w00002 w00001 w00002\n"
        b"W2 o00001 w00003 w00004 w00005 > w00003 w00005 w00003\n"
        b"W3 OFF\nW4 OFF\nE1 o00000\nE2 o00001\nE3 HALT\n"
    )
    return RelationTensorRow(
        row_id=f"{family}-{renderer}",
        family_id=family,
        renderer=renderer,
        split="train",
        program_bytes=tuple(source),
        query_bytes=tuple(b"Q2"),
        line_ranges=((0, 1),) * 18,
        binding_ranges=((3, 9), (10, 16), (17, 23)),
        initial_ranges=((28, 34), (35, 41), (42, 48)),
        witness_before_ranges=(((59, 65), (66, 72), (73, 79)), ((110, 116), (117, 123), (124, 130)), (), ()),
        witness_after_ranges=(((82, 88), (89, 95), (96, 102)), ((133, 139), (140, 146), (147, 153)), (), ()),
        query_range=(0, 2),
        cardinality=3,
        rule_count=2,
        initial_order=(1, 0, 2),
        relation_rows=((2, 1, 2), (0, 2, 0), (), ()),
        event_cards=(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        event_halt=(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        query_position=1,
        depth=2,
        non_bijective=True,
        final_state=None,
        answer_role=None,
    )


def test_alpha_recode_uses_one_namespace_and_preserves_equality_and_width() -> None:
    row = _row("f0")
    recoded = alpha_recode_row(row, "test")
    before = bytes(row.program_bytes)
    after = bytes(recoded.program_bytes)
    assert len(before) == len(after)
    assert before != after
    tokens = OPAQUE_PATTERN.findall(after)
    assert tokens and all(token.startswith(b"z") for token in tokens)
    assert len(set(tokens)) == len(set(OPAQUE_PATTERN.findall(before)))
    assert tokens.count(tokens[3]) == OPAQUE_PATTERN.findall(before).count(
        OPAQUE_PATTERN.findall(before)[3]
    )


def test_score_train_row_executes_pre_halt_relations_only() -> None:
    scored = score_train_row(_row("f0"))
    assert scored.final_state == (2, 2, 2)
    assert scored.answer_role == 2


def test_family_split_is_deterministic_disjoint_and_keeps_four_views() -> None:
    rows = [
        replace(_row(f"f{family}"), renderer=f"r{view}")
        for family in range(6)
        for view in range(4)
    ]
    first = split_train_families(rows, 91, fit_families=4, probe_families=2)
    second = split_train_families(rows, 91, fit_families=4, probe_families=2)
    assert [row.row_id for row in first[0]] == [row.row_id for row in second[0]]
    assert {row.family_id for row in first[0]}.isdisjoint(
        {row.family_id for row in first[1]}
    )
    assert first[2]["family_overlap"] == 0
    assert len(first[0]) == 16 and len(first[1]) == 8
