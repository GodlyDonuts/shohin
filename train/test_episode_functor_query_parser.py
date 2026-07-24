from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_machine import (  # noqa: E402
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from episode_functor_pointer_compiler import (  # noqa: E402
    MAX_UNIQUE_KEYS,
    collate_sources,
    scan_source,
)
from episode_functor_query_parser import (  # noqa: E402
    MAX_QUERY_BYTES,
    MAX_QUERY_KEY_OCCURRENCES,
    NeuralOpaqueQueryParser,
    QueryParserError,
    _bind_query_roles_attached_training,
    bind_query_roles_to_hard_keys,
    collate_queries,
    scan_query,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    GrammarFactors,
    LateQuery,
    SOURCE_FACTOR_COMBINATIONS,
    encode_query,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _fixture() -> tuple[object, bytes, LateQuery]:
    machine = generate_machine(
        seed="efc-query-parser-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-query-parser-test-v1",
        split="mechanics",
        index=0,
    )
    source = encode_source(evidence, GrammarFactors(0, 0, 0))
    query = LateQuery(
        start_key=machine.state_keys[3],
        action_keys=(
            machine.action_keys[2],
            machine.action_keys[0],
            machine.action_keys[2],
            machine.action_keys[1],
        ),
        observer_key=machine.observer_keys[1],
    )
    return machine, source, query


def _key_bytes(value: int) -> bytes:
    return value.to_bytes(8, "little")


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_query_scanner_is_role_blind_and_copies_keys_exactly(
    values: tuple[int, int, int],
) -> None:
    _, _, query = _fixture()
    scanned = scan_query(encode_query(query, GrammarFactors(*values)))
    assert tuple(field.name for field in fields(scanned)) == (
        "payload",
        "spans",
        "occurrence_keys",
    )
    assert scanned.occurrence_keys == tuple(
        _key_bytes(value)
        for value in (
            query.start_key,
            *query.action_keys,
            query.observer_key,
        )
    )
    for (start, end), key in zip(
        scanned.spans,
        scanned.occurrence_keys,
        strict=True,
    ):
        token = scanned.payload[start:end]
        value = int(token[1:], 16 if token.startswith(b"h") else 10)
        assert key == _key_bytes(value)


def test_query_scanner_and_collator_fail_closed() -> None:
    with pytest.raises(QueryParserError, match="key geometry"):
        scan_query(b"BEGIN-Q\nEND-Q\n")
    with pytest.raises(QueryParserError, match="too many"):
        scan_query(
            b" ".join(
                f"h{value + 1:016x}".encode("ascii")
                for value in range(MAX_QUERY_KEY_OCCURRENCES + 1)
            )
        )
    with pytest.raises(QueryParserError, match="byte length"):
        scan_query(b"x" * (MAX_QUERY_BYTES + 1))
    with pytest.raises(QueryParserError, match="uint64"):
        scan_query(b"d18446744073709551616 d1")
    with pytest.raises(QueryParserError, match="empty"):
        collate_queries([])


def _oracle_slot_logits(machine, source_batch) -> torch.Tensor:
    logits = torch.full(
        (1, MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS, MAX_UNIQUE_KEYS),
        -20.0,
    )
    inventory = [
        bytes(value.tolist())
        for value in source_batch.unique_key_bytes[0]
    ]
    targets = (
        machine.state_keys
        + machine.action_keys
        + machine.observer_keys
    )
    slots = (
        tuple(range(8))
        + tuple(MAX_STATES + index for index in range(3))
        + tuple(MAX_STATES + MAX_ACTIONS + index for index in range(2))
    )
    for slot, key in zip(slots, targets, strict=True):
        logits[0, slot, inventory.index(_key_bytes(key))] = 20.0
    logits[:, 8:MAX_STATES, 0] = 20.0
    logits[:, MAX_STATES + 3 : MAX_STATES + MAX_ACTIONS, 0] = 20.0
    logits[:, MAX_STATES + MAX_ACTIONS + 2 :, 0] = 20.0
    return logits


def _active_machine() -> HardFunctorMachine:
    state_active = torch.zeros((1, MAX_STATES), dtype=torch.uint8)
    action_active = torch.zeros((1, MAX_ACTIONS), dtype=torch.uint8)
    observer_active = torch.zeros((1, MAX_OBSERVERS), dtype=torch.uint8)
    state_active[:, :8] = 1
    action_active[:, :3] = 1
    observer_active[:, :2] = 1
    action_next = torch.zeros(
        (1, MAX_ACTIONS, MAX_STATES),
        dtype=torch.uint8,
    )
    action_next[:, :3, :8] = torch.arange(8, dtype=torch.uint8)
    observer_answer = torch.zeros(
        (1, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
    )
    return HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_exact_binding_maps_oracle_roles_to_machine_slots(
    values: tuple[int, int, int],
) -> None:
    machine, source, query = _fixture()
    source_batch = collate_sources((scan_source(source),))
    query_batch = collate_queries(
        [scan_query(encode_query(query, GrammarFactors(*values)))]
    )
    max_steps = 8
    role_logits = torch.full(
        (1, 2 + max_steps, MAX_QUERY_KEY_OCCURRENCES),
        -20.0,
    )
    role_logits[0, 0, 0] = 20.0
    role_logits[0, 1, 5] = 20.0
    for step in range(4):
        role_logits[0, 2 + step, 1 + step] = 20.0
    role_logits[0, 6:, 0] = 20.0
    stop_logits = torch.full((1, max_steps + 1), -20.0)
    stop_logits[0, 4] = 20.0
    output = _bind_query_roles_attached_training(
        role_occurrence_logits=role_logits,
        stop_position_logits=stop_logits,
        query_occurrence_key_bytes=query_batch.occurrence_key_bytes,
        query_occurrence_valid=query_batch.occurrence_valid,
        source_unique_key_bytes=source_batch.unique_key_bytes,
        source_unique_key_valid=source_batch.unique_key_valid,
        slot_assignment_logits=_oracle_slot_logits(machine, source_batch),
    )
    hard = output.query.harden(_active_machine())
    assert hard.start_state.tolist() == [3]
    assert hard.action_path[0, :4].tolist() == [2, 0, 2, 1]
    assert hard.stop_position.tolist() == [4]
    assert hard.observer.tolist() == [1]
    assert torch.equal(
        output.query.action_path[:, :, 3:],
        torch.full_like(output.query.action_path[:, :, 3:], -60.0),
    )
    assert torch.equal(
        output.query.observer[:, 2:],
        torch.full_like(output.query.observer[:, 2:], -60.0),
    )
    assert output.exact_query_key_matches.sum(-1)[
        query_batch.occurrence_valid
    ].eq(1).all()


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_post_seal_binding_uses_only_hard_keys(
    values: tuple[int, int, int],
) -> None:
    machine, source, query = _fixture()
    source_batch = collate_sources((scan_source(source),))
    query_batch = collate_queries(
        [scan_query(encode_query(query, GrammarFactors(*values)))]
    )
    hard_machine = _active_machine()
    oracle_logits = _oracle_slot_logits(machine, source_batch)
    from episode_functor_constrained_transport import hard_assign_keys

    sealed_keys = hard_assign_keys(
        slot_assignment_logits=oracle_logits,
        source_unique_key_bytes=source_batch.unique_key_bytes,
        source_unique_key_valid=source_batch.unique_key_valid,
    ).keys
    role_logits = torch.full(
        (1, 10, MAX_QUERY_KEY_OCCURRENCES),
        -20.0,
    )
    role_logits[0, 0, 0] = 20.0
    role_logits[0, 1, 5] = 20.0
    for step in range(4):
        role_logits[0, 2 + step, 1 + step] = 20.0
    role_logits[0, 6:, 0] = 20.0
    stop_logits = torch.full((1, 9), -20.0)
    stop_logits[0, 4] = 20.0
    output = bind_query_roles_to_hard_keys(
        role_occurrence_logits=role_logits,
        stop_position_logits=stop_logits,
        query_occurrence_key_bytes=query_batch.occurrence_key_bytes,
        query_occurrence_valid=query_batch.occurrence_valid,
        sealed_keys=sealed_keys,
    )
    hard = output.query.harden(hard_machine)
    assert hard.start_state.tolist() == [3]
    assert hard.action_path[0, :4].tolist() == [2, 0, 2, 1]
    assert hard.stop_position.tolist() == [4]
    assert hard.observer.tolist() == [1]
    assert output.exact_query_key_matches.sum(-1)[
        query_batch.occurrence_valid
    ].eq(1).all()


def test_exact_binding_rejects_unknown_query_key() -> None:
    machine, source, query = _fixture()
    source_batch = collate_sources((scan_source(source),))
    query_batch = collate_queries(
        [scan_query(encode_query(query, GrammarFactors(0, 0, 0)))]
    )
    query_batch.occurrence_key_bytes[0, 0] = torch.tensor(
        tuple(((1 << 64) - 1).to_bytes(8, "little")),
        dtype=torch.uint8,
    )
    with pytest.raises(QueryParserError, match="exactly one"):
        _bind_query_roles_attached_training(
            role_occurrence_logits=torch.zeros(
                1,
                10,
                MAX_QUERY_KEY_OCCURRENCES,
            ),
            stop_position_logits=torch.zeros(1, 9),
            query_occurrence_key_bytes=query_batch.occurrence_key_bytes,
            query_occurrence_valid=query_batch.occurrence_valid,
            source_unique_key_bytes=source_batch.unique_key_bytes,
            source_unique_key_valid=source_batch.unique_key_valid,
            slot_assignment_logits=_oracle_slot_logits(machine, source_batch),
        )


def test_neural_parser_is_attached_bounded_and_transition_blind() -> None:
    torch.manual_seed(29)
    _, source, query = _fixture()
    source_batch = collate_sources((scan_source(source),))
    query_batch = collate_queries(
        [scan_query(encode_query(query, GrammarFactors(1, 1, 1)))]
    )
    machine, _, _ = _fixture()
    oracle_logits = _oracle_slot_logits(machine, source_batch)
    from episode_functor_constrained_transport import hard_assign_keys

    sealed_keys = hard_assign_keys(
        slot_assignment_logits=oracle_logits,
        source_unique_key_bytes=source_batch.unique_key_bytes,
        source_unique_key_valid=source_batch.unique_key_valid,
    ).keys
    parser = NeuralOpaqueQueryParser(
        width=64,
        layers=1,
        heads=4,
        feedforward=128,
        max_steps=8,
    )
    output = parser(
        query_batch,
        sealed_keys=sealed_keys,
    )
    assert output.query.start_state.shape == (1, MAX_STATES)
    assert output.query.action_path.shape == (1, 8, MAX_ACTIONS)
    assert output.query.stop_position.shape == (1, 9)
    assert output.query.observer.shape == (1, MAX_OBSERVERS)
    assert parser.parameter_count() < 1_000_000
    loss = sum(
        value.square().mean()
        for value in (
            output.query.start_state,
            output.query.action_path,
            output.query.stop_position,
            output.query.observer,
            output.role_occurrence_logits,
        )
    )
    loss.backward()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in parser.parameters()
    )
