from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from general_relational_object_machine import (
    APPLY_KIND,
    EVENT_KIND_COUNT,
    MAX_EVENTS,
    MAX_OBJECTS,
    MAX_RULES,
    MIN_OBJECTS,
    NOOP_KIND,
    STOP_KIND,
    DeletedRelationalProgram,
    DeletedRelationalQuery,
    HardDeletedRelationalProgram,
    HardDeletedRelationalQuery,
    RelationalObjectError,
    RelationalStateTransplant,
    TrunkRelationalObjectCompiler,
    probabilistic_relation_compose,
    rollout_hard_relational_program,
    rollout_relational_program,
)


def _hard_program(
    *,
    cardinality: int = 3,
    suffix_rule: int = 0,
) -> HardDeletedRelationalProgram:
    initial = torch.zeros(
        1,
        MAX_OBJECTS,
        MAX_OBJECTS,
        dtype=torch.uint8,
    )
    initial[0, :cardinality, :cardinality] = torch.eye(
        cardinality,
        dtype=torch.uint8,
    )
    rules = torch.zeros(
        1,
        MAX_RULES,
        MAX_OBJECTS,
        MAX_OBJECTS,
        dtype=torch.uint8,
    )
    for rule in range(MAX_RULES):
        rules[0, rule, :cardinality, :cardinality] = torch.eye(
            cardinality,
            dtype=torch.uint8,
        )
    rules[0, 0, :3, :3] = F.one_hot(
        torch.tensor([1, 0, 2]),
        3,
    ).to(torch.uint8)
    rules[0, 1, :3, :3] = F.one_hot(
        torch.tensor([2, 0, 1]),
        3,
    ).to(torch.uint8)
    active = torch.zeros(1, MAX_RULES, dtype=torch.bool)
    active[0, :2] = True
    event_rule = torch.zeros(1, MAX_EVENTS, dtype=torch.uint8)
    event_rule[0, :2] = torch.tensor([0, 1], dtype=torch.uint8)
    event_rule[0, 3:] = suffix_rule
    event_kind = torch.full(
        (1, MAX_EVENTS),
        NOOP_KIND,
        dtype=torch.uint8,
    )
    event_kind[0, :3] = torch.tensor(
        [APPLY_KIND, APPLY_KIND, STOP_KIND],
        dtype=torch.uint8,
    )
    event_kind[0, 3:] = APPLY_KIND
    return HardDeletedRelationalProgram(
        cardinality=torch.tensor([cardinality], dtype=torch.uint8),
        initial_edges=initial,
        rule_edges=rules,
        rule_active=active,
        event_rule=event_rule,
        event_kind=event_kind,
    )


def _logits_from_hard(
    program: HardDeletedRelationalProgram,
) -> DeletedRelationalProgram:
    cardinality = F.one_hot(
        program.cardinality.long() - MIN_OBJECTS,
        MAX_OBJECTS - MIN_OBJECTS + 1,
    ).float() * 12.0
    initial = program.initial_edges.float() * 24.0 - 12.0
    cards = program.rule_edges.float() * 24.0 - 12.0
    active = F.one_hot(program.rule_active.long(), 2).float() * 12.0
    event_rule = F.one_hot(program.event_rule.long(), MAX_RULES).float() * 12.0
    event_kind = F.one_hot(
        program.event_kind.long(),
        EVENT_KIND_COUNT,
    ).float() * 12.0
    return DeletedRelationalProgram(
        cardinality,
        initial,
        cards,
        active,
        event_rule,
        event_kind,
    )


def test_hard_execution_composes_relations_and_stops() -> None:
    program = _hard_program()
    query = HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8))
    result = rollout_hard_relational_program(program, query)
    assert result.final_state[0, :3].argmax(-1).tolist() == [2, 1, 0]
    assert result.answer_distribution.argmax(-1).item() == 2
    assert result.alive_trajectory[1].item() == pytest.approx(1.0)
    assert result.alive_trajectory[2].item() == pytest.approx(0.0)
    assert result.halted_trajectory[2].item() == pytest.approx(1.0)
    assert program.bytes_per_row == 649


def test_boolean_relation_composition_supports_many_to_many_cards() -> None:
    generator = torch.Generator().manual_seed(29)
    left = torch.randint(
        0,
        2,
        (32, MAX_OBJECTS, MAX_OBJECTS),
        generator=generator,
    ).float()
    right = torch.randint(
        0,
        2,
        (32, MAX_OBJECTS, MAX_OBJECTS),
        generator=generator,
    ).float()
    expected = torch.bmm(left, right).gt(0).float()
    observed = probabilistic_relation_compose(left, right)
    assert torch.equal(observed, expected)

    program = _hard_program()
    cards = program.rule_edges.clone()
    cards[0, 0, 0].zero_()
    cards[0, 0, 0, :2] = 1
    event_kind = torch.full_like(program.event_kind, NOOP_KIND)
    event_kind[0, :2] = torch.tensor(
        [APPLY_KIND, STOP_KIND],
        dtype=torch.uint8,
    )
    many = HardDeletedRelationalProgram(
        program.cardinality,
        program.initial_edges,
        cards,
        program.rule_active,
        program.event_rule,
        event_kind,
    )
    result = rollout_hard_relational_program(
        many,
        HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8)),
    )
    assert result.answer_distribution[0, :3].tolist() == [1.0, 1.0, 0.0]


def test_post_stop_suffix_is_bit_invariant() -> None:
    left = _hard_program(suffix_rule=0)
    right = _hard_program(suffix_rule=1)
    query = HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8))
    left_result = rollout_hard_relational_program(left, query)
    right_result = rollout_hard_relational_program(right, query)
    assert torch.equal(left_result.final_state, right_result.final_state)
    assert torch.equal(
        left_result.answer_distribution,
        right_result.answer_distribution,
    )


def test_late_query_changes_answer_without_changing_state() -> None:
    program = _hard_program()
    first = rollout_hard_relational_program(
        program,
        HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8)),
    )
    second = rollout_hard_relational_program(
        program,
        HardDeletedRelationalQuery(torch.tensor([1], dtype=torch.uint8)),
    )
    assert torch.equal(first.final_state, second.final_state)
    assert first.answer_distribution.argmax(-1).item() == 2
    assert second.answer_distribution.argmax(-1).item() == 1


def test_state_transplant_is_causal() -> None:
    first = _hard_program()
    second = _hard_program()
    first_initial = first.initial_edges.clone()
    first_initial[0, :3] = F.one_hot(
        torch.tensor([2, 1, 0]),
        MAX_OBJECTS,
    ).to(torch.uint8)
    batch = HardDeletedRelationalProgram(
        cardinality=torch.cat((first.cardinality, second.cardinality)),
        initial_edges=torch.cat((first_initial, second.initial_edges)),
        rule_edges=torch.cat((first.rule_edges, second.rule_edges)),
        rule_active=torch.cat((first.rule_active, second.rule_active)),
        event_rule=torch.cat((first.event_rule, second.event_rule)),
        event_kind=torch.cat((first.event_kind, second.event_kind)),
    )
    query = HardDeletedRelationalQuery(
        torch.tensor([0, 0], dtype=torch.uint8)
    )
    native = rollout_hard_relational_program(batch, query)
    transplanted = rollout_hard_relational_program(
        batch,
        query,
        transplant=RelationalStateTransplant(
            after_step=0,
            batch_permutation=torch.tensor([1, 0]),
        ),
    )
    assert not torch.equal(native.final_state, transplanted.final_state)
    assert torch.equal(
        native.state_trajectory[0].flip(0),
        transplanted.state_trajectory[0],
    )


def test_soft_path_has_end_to_end_gradients() -> None:
    generator = torch.Generator().manual_seed(13)

    def leaf(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, requires_grad=True)

    fields = [
        leaf(2, MAX_OBJECTS - MIN_OBJECTS + 1),
        leaf(2, MAX_OBJECTS, MAX_OBJECTS),
        leaf(2, MAX_RULES, MAX_OBJECTS, MAX_OBJECTS),
        leaf(2, MAX_RULES, 2),
        leaf(2, MAX_EVENTS, MAX_RULES),
        leaf(2, MAX_EVENTS, EVENT_KIND_COUNT),
    ]
    query_logits = leaf(2, MAX_OBJECTS)
    program = DeletedRelationalProgram(*fields)
    query = DeletedRelationalQuery(query_logits)
    result = rollout_relational_program(program, query)
    loss = -result.answer_distribution[:, 0].clamp_min(1e-8).log().mean()
    loss.backward()
    for value in (*fields, query_logits):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert float(value.grad.abs().sum()) > 0.0


def test_sealed_object_file_is_source_independent() -> None:
    soft = _logits_from_hard(_hard_program())
    sealed = soft.seal()
    before = rollout_hard_relational_program(
        sealed,
        HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8)),
    )
    with torch.no_grad():
        soft.cardinality.fill_(-999.0)
        soft.initial_state.normal_()
        soft.rule_cards.normal_()
        soft.event_rule.normal_()
        soft.event_kind.normal_()
    after = rollout_hard_relational_program(
        sealed,
        HardDeletedRelationalQuery(torch.tensor([0], dtype=torch.uint8)),
    )
    assert torch.equal(before.final_state, after.final_state)
    assert torch.equal(before.answer_distribution, after.answer_distribution)
    assert {field.name for field in fields(HardDeletedRelationalProgram)} == {
        "cardinality",
        "initial_edges",
        "rule_edges",
        "rule_active",
        "event_rule",
        "event_kind",
    }


def test_hard_file_rejects_live_inactive_rule() -> None:
    program = _hard_program()
    active = program.rule_active.clone()
    active[0, 0] = False
    with pytest.raises(RelationalObjectError, match="inactive rule"):
        HardDeletedRelationalProgram(
            program.cardinality,
            program.initial_edges,
            program.rule_edges,
            active,
            program.event_rule,
            program.event_kind,
        )


def test_hard_file_rejects_state_outside_cardinality() -> None:
    program = _hard_program(cardinality=3)
    initial = program.initial_edges.clone()
    initial[0, 7, 7] = 1
    with pytest.raises(RelationalObjectError, match="outside cardinality"):
        HardDeletedRelationalProgram(
            program.cardinality,
            initial,
            program.rule_edges,
            program.rule_active,
            program.event_rule,
            program.event_kind,
        )

    rules = program.rule_edges.clone()
    rules[0, 0, 0, 7] = 1
    with pytest.raises(RelationalObjectError, match="outside cardinality"):
        HardDeletedRelationalProgram(
            program.cardinality,
            program.initial_edges,
            rules,
            program.rule_active,
            program.event_rule,
            program.event_kind,
        )


def test_hard_rollout_rejects_query_outside_cardinality() -> None:
    program = _hard_program(cardinality=3)
    query = HardDeletedRelationalQuery(torch.tensor([3], dtype=torch.uint8))
    with pytest.raises(RelationalObjectError, match="declared cardinality"):
        rollout_hard_relational_program(program, query)


class _TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(
        self,
        hidden: torch.Tensor,
        _cosine: torch.Tensor,
        _sine: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return self.projection(hidden), None


class _TinyShohin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 16
        self.cfg = SimpleNamespace(
            n_loop=1,
            d_model=width,
            vocab_size=64,
            seq_len=24,
        )
        self.tok = nn.Embedding(self.cfg.vocab_size, width)
        self.blocks = nn.ModuleList([_TinyBlock(width) for _ in range(4)])
        self.register_buffer("cos", torch.zeros(self.cfg.seq_len, 1))
        self.register_buffer("sin", torch.zeros(self.cfg.seq_len, 1))


def test_trunk_compiler_contract_and_unique_parameter_ledger() -> None:
    base = _TinyShohin()
    compiler = TrunkRelationalObjectCompiler(
        base,
        compiler_width=32,
        compiler_heads=4,
        encoder_layers=1,
        encoder_feedforward=64,
        decoder_layers=1,
        decoder_feedforward=64,
        identity_width=12,
        early_layer=1,
        late_layer=2,
    )
    ids = torch.tensor(
        [
            [2, 3, 4, 5, 1, 1],
            [6, 7, 8, 9, 10, 1],
        ],
        dtype=torch.long,
    )
    output, evidence = compiler.compile_program(ids, return_evidence=True)
    query = compiler.compile_late_query(ids)
    assert isinstance(output, DeletedRelationalProgram)
    assert output.cardinality.shape == (2, 7)
    assert output.rule_cards.shape == (2, 8, 8, 8)
    assert output.event_kind.shape == (2, 32, 3)
    assert evidence.pointer_logits.shape == (
        2,
        compiler.object_slot_count,
        ids.shape[1],
    )
    assert query.position.shape == (2, 8)
    report = compiler.parameter_report()
    expected_base = sum(parameter.numel() for parameter in base.parameters())
    expected_complete = sum(
        parameter.numel()
        for parameter in {id(value): value for value in compiler.parameters()}.values()
    )
    assert report["base"] == expected_base
    assert report["complete_system"] == expected_complete
    assert report["complete_system"] < report["strict_cap"]
    assert not any(
        name in {field.name for field in fields(output)}
        for name in ("ids", "memory", "pointer_logits", "source", "query")
    )
