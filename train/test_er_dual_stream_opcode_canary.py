from __future__ import annotations

import torch

from assess_er_dual_stream_opcode_canary import verify_path_evidence
from build_er_dual_stream_fresh_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_relation_tensor_training import loss_batch, parse_row
from pilot_er_dual_stream_opcode_canary import (
    compute_gates,
    evaluate_coherent_routes,
    query_only_alpha_row,
    relocation_consistency,
    renderer_relocation_rows,
)
from pilot_er_dual_stream_train_canary import score_train_row
from er_dual_stream_relation_adapter import DualStreamRelationCompiler


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
    assert {row.renderer for row in relocated} == {
        "er-ds-d0w0e0q0-v2",
        "er-ds-d0w0e0q1-v2",
        "er-ds-d0w1e0q0-v2",
        "er-ds-d0w1e0q1-v2",
    }
    by_family = {}
    for row in relocated:
        by_family.setdefault(row.family_id, []).append(row)
    for rows in by_family.values():
        assert len({row.program_bytes for row in rows if "w0" in row.renderer}) == 1
        assert len({row.program_bytes for row in rows if "w1" in row.renderer}) == 1


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


def test_query_only_recode_preserves_program_and_target_span() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    row = parse_row(splits[TRAIN_SPLIT][0], TRAIN_SPLIT)
    recoded = query_only_alpha_row(row, "unit-query-only")
    assert recoded.program_bytes == row.program_bytes
    assert recoded.query_range == row.query_range
    assert recoded.query_bytes != row.query_bytes


def test_coherent_decoder_emits_complete_train_only_evidence() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    rows = [
        score_train_row(parse_row(row, TRAIN_SPLIT))
        for row in splits[TRAIN_SPLIT]
    ]
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
    ).eval()
    metrics, evidence = evaluate_coherent_routes(
        model, rows, opcode_weight=1.0, batch_size=4
    )
    assert metrics["overall"]["joint"]["rows"] == 4
    assert evidence["path_scores"].shape == (4, 4, 4, 13)
    assert evidence["candidate_positions"].shape == (4, 4, 13)
    assert evidence["target_exclusion"].shape == (4, 4)
    assert evidence["pred_relations"].shape == (4, 4, 6)


def test_structured_route_objective_is_finite_and_reaches_both_queries() -> None:
    splits, _ = build_board(
        seed=104729,
        families={TRAIN_SPLIT: 1, DEVELOPMENT_SPLIT: 1, CONFIRMATION_SPLIT: 1},
    )
    rows = [parse_row(row, TRAIN_SPLIT) for row in splits[TRAIN_SPLIT]]
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
    ).train()
    model.structured_route_objective = True
    loss, pieces = loss_batch(model, [rows], torch.device("cpu"))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in pieces.values())
    loss.backward()
    assert model.er_ds_witness_queries.grad is not None
    assert model.er_ds_rule_opcode_query.grad is not None


def test_independent_assessor_verifies_coherent_candidate_partition() -> None:
    rows = 8_000
    scores = torch.zeros((rows, 4, 4, 13), dtype=torch.float16)
    probability = torch.zeros_like(scores)
    predicted_cardinality = torch.zeros(rows, dtype=torch.int16)
    target_cardinality = torch.zeros(rows, dtype=torch.int16)
    target_rule_count = torch.zeros(rows, dtype=torch.int16)
    map_exclusion = torch.full((rows, 4), -1, dtype=torch.int16)
    target_exclusion = torch.full((rows, 4), -1, dtype=torch.int16)
    candidates = torch.full((rows, 4, 13), -1, dtype=torch.int32)
    witness = torch.full((rows, 4, 12), -1, dtype=torch.int32)
    opcode = torch.full((rows, 4), -1, dtype=torch.int32)

    predicted_cardinality[0] = target_cardinality[0] = 3
    target_rule_count[0] = 1
    scores[0, 0, 0, 3] = 5
    probability[0, 0, 0, :7] = scores[0, 0, 0, :7].softmax(-1)
    map_exclusion[0, 0] = target_exclusion[0, 0] = 3
    candidates[0, 0, :7] = torch.arange(7)
    witness[0, 0, [0, 1, 2, 6, 7, 8]] = torch.tensor(
        [0, 1, 2, 4, 5, 6], dtype=torch.int32
    )
    opcode[0, 0] = 3
    relocated = {
        "path_scores": scores,
        "path_probability": probability,
        "pred_cardinality": predicted_cardinality,
        "target_cardinality": target_cardinality,
        "target_rule_count": target_rule_count,
        "map_exclusion": map_exclusion,
        "target_exclusion": target_exclusion,
        "candidate_positions": candidates,
        "pred_witness_pointer": witness,
        "rule_opcode_pointer": opcode,
    }
    evidence = {
        "arms": {
            "unit": {
                "modes": {
                    "s0_qstruct": {"relocated": relocated},
                    "s1_qstruct": {"relocated": relocated},
                }
            }
        }
    }
    result = verify_path_evidence(evidence)
    assert result["active_routes_checked"] == 2
    assert result["all_map_argmax_exact"] is True
    assert result["all_coherent_complements_exact"] is True
    assert result["all_conditional_probabilities_normalized"] is True
    assert result["all_conditional_probabilities_match_softmax"] is True
    assert result["row_exact_by_arm_mode"] == {
        "unit:s0_qstruct": 8_000,
        "unit:s1_qstruct": 8_000,
    }


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
    fit = {"frozen_parent_unchanged": True, "updates": 2_500}

    def mode(coherent: float, marginal: float, shuffled: float | None = None):
        return {
            "canonical": {
                "coherent": _metric(coherent),
                "marginal": _metric(marginal),
            },
            "relocated": {
                "coherent": _metric(coherent),
                "marginal": _metric(marginal),
            },
            "relocation_consistency": {"exact": 2_000, "families": 2_000},
            "alpha": exact,
            "distractor": exact,
            "source_free": _metric(0.0),
            "opcode_shuffled": None if shuffled is None else _metric(shuffled),
        }

    query = {"qstruct": {"recode_a": exact, "recode_b": exact}}
    arms = {
        "zero_update": {"fit": {"frozen_parent_unchanged": True, "updates": 0}},
        "opcode_coupled": {
            "modes": {
                "s0_qstruct": mode(0.7, 0.7),
                "s1_qstruct": mode(1.0, 1.0, 0.7),
            },
            "query_modes": query,
            "fit": fit,
        },
        "legacy_uncoupled": {
            "modes": {
                "s0_qstruct": mode(0.7, 0.7),
                "s1_qstruct": mode(0.7, 0.7, 0.7),
            },
            "query_modes": query,
            "fit": fit,
        },
        "structured_route": {
            "modes": {
                "s0_qstruct": mode(0.7, 0.7),
                "s1_qstruct": mode(1.0, 1.0, 0.7),
            },
            "query_modes": query,
            "fit": fit,
        },
    }
    gates, diagnosis = compute_gates(
        arms,
        parameters={
            **__import__(
                "pilot_er_dual_stream_relation_adapter"
            ).EXPECTED_PARAMETERS
        },
        shared_initialization=True,
    )
    assert all(gates.values())
    assert diagnosis["selected"] == {
        "arm": "opcode_coupled",
        "mode": "s1_qstruct",
        "mechanism": "learned_opcode_coupling",
    }
    arms["opcode_coupled"]["modes"]["s1_qstruct"]["opcode_shuffled"] = _metric(0.9)
    gates, _ = compute_gates(
        arms,
        parameters={
            **__import__(
                "pilot_er_dual_stream_relation_adapter"
            ).EXPECTED_PARAMETERS
        },
        shared_initialization=True,
    )
    assert not gates["opcode_score_is_causal_when_selected"]
