from __future__ import annotations

import torch

from build_er_dual_stream_fresh_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_relation_tensor_training import parse_row
from pilot_er_dual_stream_opcode_canary import (
    compute_gates,
    relocation_consistency,
    renderer_relocation_rows,
)


def test_relocation_rerenders_only_training_semantics() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 3, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    raw = splits[TRAIN_SPLIT]
    families = {str(row["family_id"]) for row in raw[:8]}
    relocated = renderer_relocation_rows(raw, families, seed=701)
    assert len(relocated) == 4 * len(families)
    assert {row.split for row in relocated} == {TRAIN_SPLIT}
    assert {row.family_id for row in relocated} == families
    assert all(row.final_state is None and row.answer_role is None for row in relocated)
    assert all("w" in row.renderer for row in relocated)


def test_renderer_consistency_requires_all_eight_views() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 2, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    raw = splits[TRAIN_SPLIT]
    canonical = [parse_row(row, TRAIN_SPLIT) for row in raw]
    families = {row.family_id for row in canonical}
    relocated = renderer_relocation_rows(raw, families, seed=702)
    rows = sorted(canonical + relocated, key=lambda row: (row.family_id, row.row_id))
    predictions = {}
    for key, shape in (
        ("cardinality", (len(rows),)),
        ("initial", (len(rows), 6)),
        ("relations", (len(rows), 4, 6)),
        ("rule_active", (len(rows), 4)),
        ("events", (len(rows), 13)),
        ("halt", (len(rows), 13)),
        ("query", (len(rows),)),
    ):
        predictions[key] = torch.zeros(shape, dtype=torch.int16)
    result = relocation_consistency(rows, predictions)
    assert result == {"exact": 2, "families": 2, "rate": 1.0}
    predictions["query"][0] = 1
    assert relocation_consistency(rows, predictions)["exact"] == 1


def _metric(rate: float) -> dict[str, object]:
    fields = {
        name: {"correct": int(8_000 * rate), "rows": 8_000, "rate": rate}
        for name in (
            "packet",
            "state",
            "answer",
            "joint",
            "relation_rows",
            "witness_pointer",
        )
    }
    grouped = {"x": {"joint": fields["joint"]}}
    return {"overall": fields, "by_cardinality": grouped, "by_renderer": grouped}


def test_gate_requires_causal_advantage_over_legacy() -> None:
    exact = {"complete": {"exact": 8_000, "rows": 8_000, "rate": 1.0}}
    fit = {"frozen_parent_unchanged": True}
    arms = {
        "opcode_coupled": {
            "canonical": _metric(1.0),
            "relocated": _metric(1.0),
            "relocation_consistency": {"exact": 2_000, "families": 2_000},
            "alpha": exact,
            "distractor": exact,
            "source_free": _metric(0.0),
            "fit": fit,
        },
        "legacy_uncoupled": {
            "canonical": _metric(0.7),
            "relocated": _metric(0.7),
            "fit": fit,
        },
    }
    assert all(
        compute_gates(
            arms, parameters={**__import__("pilot_er_dual_stream_relation_adapter").EXPECTED_PARAMETERS}, shared_initialization=True
        ).values()
    )
    arms["legacy_uncoupled"]["relocated"] = _metric(0.9)
    gates = compute_gates(
        arms, parameters={**__import__("pilot_er_dual_stream_relation_adapter").EXPECTED_PARAMETERS}, shared_initialization=True
    )
    assert not gates["coupling_beats_favorable_legacy_by_20pp"]
    assert not gates["favorable_legacy_relocated_joint_at_most_80pct"]
