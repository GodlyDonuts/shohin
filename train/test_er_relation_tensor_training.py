from __future__ import annotations

import json

import torch

from build_er_relation_tensor_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_relation_tensor_adapter import EpisodicRelationTensorCompiler
from er_relation_tensor_renderers import independently_execute
from er_relation_tensor_training import (
    _equality_ablated_bytes,
    _post_halt_suffix,
    _reindex_records,
    _rename_tokens,
    _targets,
    arm_rows,
    evaluate_arm,
    group_families,
    intervention_metrics,
    loss_batch,
    parse_row,
)


FAMILIES = {
    TRAIN_SPLIT: 24,
    DEVELOPMENT_SPLIT: 12,
    CONFIRMATION_SPLIT: 12,
}


def _small_model() -> EpisodicRelationTensorCompiler:
    return EpisodicRelationTensorCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=640,
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
        max_line_bytes=112,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    )


def _rows():
    splits, _ = build_board(seed=440_881, families=FAMILIES)
    return {
        split: [parse_row(row, split) for row in values]
        for split, values in splits.items()
    }, splits


def test_parser_masks_variable_cardinality_without_outcome_leakage() -> None:
    rows, _ = _rows()
    train = rows[TRAIN_SPLIT]
    assert {row.cardinality for row in train} == {3, 4, 5, 6}
    assert all(row.final_state is None and row.answer_role is None for row in train)
    target = _targets(train[:16], torch.device("cpu"))
    for index, row in enumerate(train[:16]):
        n = row.cardinality
        assert target["row_active"][index].sum() == n
        assert target["rule_active"][index].sum() == row.rule_count
        assert target["initial"][index, n:].eq(-1).all()
        assert target["relation"][index, row.rule_count :].eq(-1).all()


def test_family_derangement_is_view_consistent_source_preserving_and_nonself() -> None:
    rows, _ = _rows()
    train = rows[TRAIN_SPLIT]
    transformed = arm_rows(train, "family_deranged", 9_919)
    assert [row.program_bytes for row in transformed] == [
        row.program_bytes for row in train
    ]
    original = {group[0].family_id: group[0].relation_rows for group in group_families(train)}
    for group in group_families(transformed):
        assert len({row.relation_rows for row in group}) == 1
        assert group[0].relation_rows != original[group[0].family_id]


def test_equality_ablation_changes_only_after_spans_and_preserves_offsets() -> None:
    rows, _ = _rows()
    row = next(value for value in rows[TRAIN_SPLIT] if value.cardinality == 6)
    original = bytes(row.program_bytes)
    mutated = bytes(_equality_ablated_bytes(row, 7_771))
    changed = {index for index, pair in enumerate(zip(original, mutated, strict=True)) if pair[0] != pair[1]}
    allowed = {
        index
        for rule in range(row.rule_count)
        for start, end in row.witness_after_ranges[rule]
        for index in range(start, end)
    }
    assert changed
    assert changed <= allowed
    before = {
        mutated[start:end]
        for rule in range(row.rule_count)
        for start, end in row.witness_before_ranges[rule]
    }
    after = [
        mutated[start:end]
        for rule in range(row.rule_count)
        for start, end in row.witness_after_ranges[rule]
    ]
    assert not before.intersection(after)
    assert len(after) == len(set(after))


def test_source_invariance_transforms_preserve_public_semantics() -> None:
    rows, raw = _rows()
    original_rows = raw[DEVELOPMENT_SPLIT]
    parsed_rows = rows[DEVELOPMENT_SPLIT]
    for original, row in zip(original_rows[:16], parsed_rows[:16], strict=True):
        expected = independently_execute(original)
        for variant in (
            _reindex_records(row, True),
            _reindex_records(row, False),
            _rename_tokens(row, "w", "witness-alpha"),
            _rename_tokens(row, "o", "opcode-alpha"),
        ):
            changed = json.loads(json.dumps(original))
            changed["program_text"] = bytes(variant.program_bytes).decode()
            actual = independently_execute(changed)
            assert actual["final_state"] == expected["final_state"]
            assert actual["answer_role"] == expected["answer_role"]
            assert actual["rule_relations"] == expected["rule_relations"]
        suffix = json.loads(json.dumps(original))
        suffix["program_text"] = bytes(_post_halt_suffix(row).program_bytes).decode()
        assert independently_execute(suffix)["final_state"] == expected["final_state"]


def test_small_model_loss_is_finite_and_reaches_all_new_heads() -> None:
    torch.manual_seed(18)
    rows, _ = _rows()
    families = group_families(rows[TRAIN_SPLIT])[:2]
    model = _small_model().train()
    loss, pieces = loss_batch(model, families, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert set(pieces) == {
        "binding_pointer",
        "cardinality",
        "consistency",
        "events",
        "halt",
        "initial_pointer",
        "initial_rows",
        "line_pointer",
        "query",
        "query_pointer",
        "relation_rows",
        "rule_active",
        "witness_pointer",
    }
    loss.backward()
    for prefix in (
        "er_tt_record_role_head",
        "er_tt_cardinality_head",
        "er_tt_rule_active_head",
        "er_tt_query_head",
        "er_tt_witness_side_embedding",
        "er_tt_witness_position_embedding",
    ):
        assert any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )


def test_causal_interventions_are_exact_for_an_exact_packet() -> None:
    rows, _ = _rows()
    selected = rows[DEVELOPMENT_SPLIT][:32]
    target = _targets(selected, torch.device("cpu"))
    packet = {
        "cardinality": target["cardinality"],
        "initial": target["initial"],
        "relations": target["relation"],
        "rule_active": target["rule_active"].long(),
        "events": target["events"],
        "halt": target["halt"],
        "query": target["query"],
    }
    result = intervention_metrics(packet, packet)
    for value in result.values():
        assert value["eligible"] == len(selected)
        assert value["exact_on_eligible"] == len(selected)
        assert value["sensitive"] > 0
        assert value["changed_on_sensitive"] == value["sensitive"]


def test_small_model_evaluation_emits_raw_and_invariance_evidence() -> None:
    torch.manual_seed(19)
    rows, _ = _rows()
    model = _small_model().eval()
    result = evaluate_arm(
        model,
        rows[DEVELOPMENT_SPLIT][:8],
        batch_size=4,
        include_raw=True,
        include_invariances=True,
    )
    assert result["overall"]["joint"]["rows"] == 8
    assert set(result["interventions"]) == {
        "relation_derangement",
        "cardinality_mask",
        "state_reset",
        "query_swap",
    }
    assert set(result["invariance"]) == {
        "rule_storage_reindex",
        "physical_record_reindex",
        "witness_alpha_rename",
        "opcode_alpha_rename",
        "post_halt_suffix",
        "source_poison_after_seal",
    }
    raw = result["raw"]
    assert raw["pred_relations"].shape == (8, 4, 6)
    assert raw["invariance_witness_alpha_rename_state"].shape == (8, 6)
