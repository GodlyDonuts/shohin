from __future__ import annotations

import torch
import torch.nn.functional as F

from er_relation_tensor_adapter import (
    EVENT_SLOTS,
    MAX_CARDINALITY,
    MAX_RULES,
    TT_RECORDS,
    EpisodicRelationTensorCompiler,
    RelationTensorProgram,
    freeze_to_relation_tensor_adaptive,
    hard_relation_answer,
    relation_tensor_parameter_report,
)


def _small_model() -> EpisodicRelationTensorCompiler:
    return EpisodicRelationTensorCompiler(
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
        max_line_bytes=48,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    )


def _source_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for row in range(batch):
        lines = [f"R{line:02d} x{row:02d}{line:02d}" for line in range(TT_RECORDS)]
        rows.append("\n".join(lines).encode())
    width = max(map(len, rows))
    ids = torch.full((batch, width), 256, dtype=torch.long)
    valid = torch.zeros((batch, width), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(tuple(row))
        valid[index, : len(row)] = True
    return ids, valid


def _query_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [f"Q{row % MAX_CARDINALITY + 1}".encode() for row in range(batch)]
    ids = torch.tensor([tuple(row) for row in rows], dtype=torch.long)
    return ids, torch.ones_like(ids, dtype=torch.bool)


def _pointer_logits(
    source: bytes,
    ranges: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor([tuple(source)], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    logits = torch.full((1, len(ranges), len(source)), -20.0)
    for slot, (start, end) in enumerate(ranges):
        logits[0, slot, start:end] = 20.0
    return ids, valid, logits


def test_public_compiler_emits_variable_relation_packet() -> None:
    torch.manual_seed(31)
    model = _small_model().eval()
    ids, valid = _source_batch()
    query_ids, query_valid = _query_batch()
    output = model.compile_relation_program(ids, valid, query_ids, query_valid)
    assert output.program.cardinality.shape == (2, 4)
    assert output.program.initial_state.shape == (2, 6, 6)
    assert output.program.rule_cards.shape == (2, 4, 6, 6)
    assert output.program.rule_active.shape == (2, 4, 2)
    assert output.program.event_card.shape == (2, 13, 4)
    assert output.program.event_halt.shape == (2, 13, 2)
    assert output.witness_pointer_logits.shape[:3] == (2, 4, 12)
    assert output.line_pointer_logits.shape[:2] == (2, 18)
    assert output.query.logits.shape == (2, 6)
    assert not hasattr(model, "er_rule_permutation_head")
    assert not hasattr(model, "er_witness_queries")
    assert not hasattr(model, "permutations")


def test_direct_equality_recovers_non_bijective_six_position_relation() -> None:
    torch.manual_seed(32)
    model = _small_model().eval()
    before = [f"symbol-{index}-abcdefgh" for index in range(6)]
    relation = (5, 0, 0, 3, 5, 1)
    words = before + [before[index] for index in relation]
    source = " ".join(words).encode()
    ranges = []
    cursor = 0
    for word in words:
        start = source.index(word.encode(), cursor)
        ranges.append((start, start + len(word)))
        cursor = start + len(word)
    ids, valid, logits = _pointer_logits(source, ranges)
    pointers = logits[:, None].expand(-1, MAX_RULES, -1, -1).contiguous()
    equality = model._equality_logits(ids, valid, pointers)
    assert equality.shape == (1, MAX_RULES, 6, 6)
    expected = torch.tensor(relation)[None, None].expand(1, MAX_RULES, -1)
    assert torch.equal(equality.argmax(-1), expected)


def test_hard_packet_masks_cardinality_rolls_out_and_reads_without_parameters() -> None:
    batch = 2
    cardinality = torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]])
    initial = torch.full((batch, 6, 6), -8.0)
    initial[0, torch.arange(3), torch.tensor([2, 0, 1])] = 8.0
    initial[1, torch.arange(6), torch.arange(6)] = 8.0
    cards = torch.full((batch, MAX_RULES, 6, 6), -8.0)
    for row in range(batch):
        for rule in range(MAX_RULES):
            cards[row, rule, torch.arange(6), torch.arange(6)] = 8.0
    cards[0, 0, torch.arange(3), torch.tensor([1, 1, 0])] = 9.0
    rule_active = torch.zeros(batch, MAX_RULES, 2)
    rule_active[..., 1] = 1.0
    event_card = F.one_hot(torch.zeros(batch, EVENT_SLOTS, dtype=torch.long), MAX_RULES).float()
    event_halt = F.one_hot(torch.ones(batch, EVENT_SLOTS, dtype=torch.long), 2).float()
    event_halt[:, 0] = torch.tensor([1.0, 0.0])
    hard = RelationTensorProgram(
        cardinality,
        initial,
        cards,
        rule_active,
        event_card,
        event_halt,
    ).hard()
    assert hard.cardinality.tolist() == [3, 6]
    assert hard.active.sum(-1).tolist() == [3, 6]
    result = hard.rollout()
    answer = hard_relation_answer(
        result.final_state,
        torch.tensor([[9.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 9.0]]),
        hard.active,
    )
    assert answer.shape == (batch, MAX_CARDINALITY)
    assert int(answer[0, 0]) == 1


def test_witness_path_is_detached_from_shared_record_assignment() -> None:
    model = _small_model().train()
    batch = 2
    records = torch.randn(batch, TT_RECORDS, 32, requires_grad=True)
    memory = torch.randn(batch, TT_RECORDS, 48, 32, requires_grad=True)
    valid = torch.ones(batch, TT_RECORDS, 48, dtype=torch.bool)
    source_indices = torch.arange(48)[None, None].expand(batch, TT_RECORDS, -1)
    assignment = torch.eye(TT_RECORDS)[None].expand(batch, -1, -1).requires_grad_()
    logits = model._global_witness_logits(
        records,
        memory,
        valid,
        source_indices,
        assignment,
        48,
    )
    (-logits[..., 0].mean()).backward()
    assert records.grad is None
    assert memory.grad is None
    assert assignment.grad is None
    assert model.er_tt_witness_side_embedding.grad is not None
    assert model.er_tt_witness_position_embedding.grad is not None


def test_trainable_contract_excludes_fixed_parent_and_parameter_budget_is_exact() -> None:
    model = EpisodicRelationTensorCompiler()
    declared = set(freeze_to_relation_tensor_adaptive(model))
    assert declared
    assert all(
        parameter.requires_grad == (name in declared)
        for name, parameter in model.named_parameters()
    )
    report = relation_tensor_parameter_report(model)
    assert report["motor"] == 0
    assert report["reader"] == 0
    assert report["complete_system"] == 192_740_854
    assert report["headroom_below_200m"] == 7_259_146
    assert report["trainable"] == 12_037_293
    assert report["complete_system"] == 125_081_664 + model.parameter_count()


def test_compiler_field_gradients_do_not_leak_into_excluded_parent() -> None:
    torch.manual_seed(33)
    model = _small_model().train()
    declared = set(freeze_to_relation_tensor_adaptive(model))
    ids, valid = _source_batch(batch=1)
    query_ids, query_valid = _query_batch(batch=1)
    output = model.compile_relation_program(ids, valid, query_ids, query_valid)
    loss = sum(
        value.float().square().mean()
        for value in (
            output.program.cardinality,
            output.program.initial_state,
            output.program.rule_cards,
            output.program.rule_active,
            output.program.event_card,
            output.program.event_halt,
            output.query.logits,
        )
    )
    loss.backward()
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert leaked == []
    assert model.er_tt_record_role_head.weight.grad is not None
    assert model.er_tt_occurrence_head.weight.grad is not None
    assert model.er_tt_query_head.weight.grad is not None


def test_compiler_packet_contains_no_source_tensor_or_outcome_field() -> None:
    model = _small_model().eval()
    ids, valid = _source_batch(batch=1)
    query_ids, query_valid = _query_batch(batch=1)
    output = model.compile_relation_program(ids, valid, query_ids, query_valid)
    fields = set(output.program.__dataclass_fields__)
    assert fields == {
        "cardinality",
        "initial_state",
        "rule_cards",
        "rule_active",
        "event_card",
        "event_halt",
    }
    assert not {"source", "final_state", "answer", "trajectory"} & fields
