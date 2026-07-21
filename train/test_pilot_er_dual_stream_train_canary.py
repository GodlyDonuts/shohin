from __future__ import annotations

from dataclasses import replace

import torch

from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from er_relation_tensor_training import RelationTensorRow
from pilot_er_dual_stream_train_canary import (
    OPAQUE_PATTERN,
    alpha_recode_row,
    _oracle_pointer_logits,
    oracle_route_transport_metrics,
    score_train_row,
    split_train_families,
)


def _row(family: str, renderer: str = "r0") -> RelationTensorRow:
    source = bytearray(
        b"D3 e00000 e00001 e00002 ; I e00001 e00000 e00002\n"
        b"W1 o00000 w00000 w00001 w00002 > w00002 w00001 w00002\n"
        b"W2 o00001 w00003 w00004 w00005 > w00003 w00005 w00003\n"
        b"W3 OFF\nW4 OFF\nE1 o00000\nE2 o00001\nE3 HALT\n"
    )
    source.extend(b"".join(f"E{slot} o00000\n".encode() for slot in range(4, 14)))
    starts = [0]
    starts.extend(index + 1 for index, value in enumerate(source) if value == 10)
    lines = tuple(
        (start, starts[index + 1] - 1 if index + 1 < len(starts) else len(source))
        for index, start in enumerate(starts[:-1])
    )
    assert len(lines) == 18
    return RelationTensorRow(
        row_id=f"{family}-{renderer}",
        family_id=family,
        renderer=renderer,
        split="train",
        program_bytes=tuple(source),
        query_bytes=tuple(b"Q2"),
        line_ranges=lines,
        binding_ranges=((3, 9), (10, 16), (17, 23)),
        initial_ranges=((28, 34), (35, 41), (42, 48)),
        witness_before_ranges=(((59, 65), (66, 72), (73, 79)), ((113, 119), (120, 126), (127, 133)), (), ()),
        witness_after_ranges=(((82, 88), (89, 95), (96, 102)), ((136, 142), (143, 149), (150, 156)), (), ()),
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


def test_oracle_pointer_logits_select_exact_symbol_starts() -> None:
    starts = torch.zeros((2, 20), dtype=torch.bool)
    starts[0, (2, 10)] = True
    starts[1, (4, 14)] = True
    logits = _oracle_pointer_logits(
        starts,
        [[(10, 16), None], [(4, 10), (14, 20)]],
        device=torch.device("cpu"),
    )
    assert logits.argmax(-1).tolist() == [[10, 2], [4, 14]]


def test_oracle_route_control_proves_identity_transport_end_to_end() -> None:
    model = DualStreamRelationCompiler(
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
        max_line_bytes=96,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    )
    metrics, predictions = oracle_route_transport_metrics(
        model, [_row("f0")], batch_size=1
    )
    assert all(metrics[name]["rate"] == 1.0 for name in ("initial", "relations", "events", "joint"))
    assert predictions["initial"][0, :3].tolist() == [1, 0, 2]
    assert predictions["relations"][0, 0, :3].tolist() == [2, 1, 2]
    assert predictions["relations"][0, 1, :3].tolist() == [0, 2, 0]
