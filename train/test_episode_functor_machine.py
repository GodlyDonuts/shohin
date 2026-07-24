from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode_functor_machine import (  # noqa: E402
    DEPLOYED_MACHINE_BYTES,
    FunctorMachineError,
    HardFunctorKeys,
    HardFunctorMachine,
    HardFunctorQuery,
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    LearnedFunctorWireSpec,
    SEMANTIC_BYTES_PER_ROW,
    SoftFunctorMachine,
    SoftFunctorQuery,
    execute_hard,
    execute_soft,
)


def _logits(indices: torch.Tensor, classes: int) -> torch.Tensor:
    output = torch.full((*indices.shape, classes), -20.0)
    return output.scatter(-1, indices.long().unsqueeze(-1), 20.0)


def _hard_machine() -> HardFunctorMachine:
    state_active = torch.zeros((1, MAX_STATES), dtype=torch.uint8)
    state_active[:, :3] = 1
    action_active = torch.zeros((1, MAX_ACTIONS), dtype=torch.uint8)
    action_active[:, :2] = 1
    observer_active = torch.zeros((1, MAX_OBSERVERS), dtype=torch.uint8)
    observer_active[:, 0] = 1
    action_next = torch.zeros(
        (1, MAX_ACTIONS, MAX_STATES),
        dtype=torch.uint8,
    )
    action_next[0, 0, :3] = torch.tensor((1, 2, 0), dtype=torch.uint8)
    action_next[0, 1, :3] = torch.tensor((1, 0, 2), dtype=torch.uint8)
    observer_answer = torch.zeros(
        (1, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
    )
    observer_answer[0, 0, :3] = torch.tensor((0, 1, 2), dtype=torch.uint8)
    return HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )


def _hard_query(actions=(0, 1), stop=2, observer=0, start=0) -> HardFunctorQuery:
    return HardFunctorQuery(
        start_state=torch.tensor((start,), dtype=torch.uint8),
        action_path=torch.tensor((actions,), dtype=torch.uint8),
        stop_position=torch.tensor((stop,), dtype=torch.uint8),
        observer=torch.tensor((observer,), dtype=torch.uint8),
    )


def _soft_from_hard(
    machine: HardFunctorMachine,
    query: HardFunctorQuery,
) -> tuple[SoftFunctorMachine, SoftFunctorQuery]:
    state_active = _logits(machine.state_active.long(), 2)
    action_active = _logits(machine.action_active.long(), 2)
    observer_active = _logits(machine.observer_active.long(), 2)
    action_next = _logits(machine.action_next.long(), MAX_STATES)
    observer_answer = _logits(machine.observer_answer.long(), MAX_ANSWERS)
    soft_machine = SoftFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )
    soft_query = SoftFunctorQuery(
        start_state=_logits(query.start_state.long(), MAX_STATES),
        action_path=_logits(query.action_path.long(), MAX_ACTIONS),
        stop_position=_logits(
            query.stop_position.long(),
            query.max_steps + 1,
        ),
        observer=_logits(query.observer.long(), MAX_OBSERVERS),
    )
    return soft_machine, soft_query


def test_noncommuting_order_and_stop_are_machine_owned() -> None:
    machine = _hard_machine()
    left = execute_hard(machine, _hard_query((0, 1)))
    right = execute_hard(machine, _hard_query((1, 0)))
    stopped = execute_hard(machine, _hard_query((0, 1), stop=1))
    assert int(left.answer.item()) == 0
    assert int(right.answer.item()) == 2
    assert int(stopped.answer.item()) == 1
    assert not torch.equal(left.states[:, -1], right.states[:, -1])


def test_soft_one_hot_execution_matches_hard_execution() -> None:
    hard_machine = _hard_machine()
    hard_query = _hard_query((1, 0))
    soft_machine, soft_query = _soft_from_hard(hard_machine, hard_query)
    soft = execute_soft(soft_machine, soft_query)
    hard = execute_hard(hard_machine, hard_query)
    assert int(soft.answer.argmax(-1).item()) == int(hard.answer.item())
    assert torch.equal(
        soft.states[:, -1].argmax(-1),
        hard.states[:, -1].argmax(-1),
    )
    assert torch.equal(soft_machine.harden().action_next, hard_machine.action_next)
    assert torch.equal(
        soft_query.harden(hard_machine).action_path,
        hard_query.action_path,
    )


def test_attached_straight_through_forward_matches_detached_hard() -> None:
    hard_machine = _hard_machine()
    hard_query = _hard_query((1, 0), stop=2, observer=0, start=1)
    soft_machine, soft_query = _soft_from_hard(hard_machine, hard_query)
    attached = execute_soft(
        soft_machine,
        soft_query,
        straight_through=True,
    )
    detached_machine = soft_machine.harden()
    detached_query = soft_query.harden(detached_machine)
    detached = execute_hard(detached_machine, detached_query)
    assert int(attached.answer.argmax(-1).item()) == int(
        detached.answer.item()
    )
    assert torch.equal(
        attached.states.argmax(-1),
        detached.states.argmax(-1),
    )


def test_inactive_soft_slots_cannot_carry_hidden_execution_state() -> None:
    hard_machine = _hard_machine()
    hard_query = _hard_query((0,), stop=1)
    soft_machine, soft_query = _soft_from_hard(hard_machine, hard_query)
    with torch.no_grad():
        soft_machine.action_next.fill_(-20.0)
        soft_machine.action_next[0, 0, 0, 7] = 100.0
        soft_machine.action_next[0, 0, 0, 0] = 20.0
        soft_machine.observer_answer.fill_(-20.0)
        soft_machine.observer_answer[0, 0, 0, 0] = 20.0
        soft_machine.observer_answer[0, 0, 7, 1] = 100.0
    attached = execute_soft(soft_machine, soft_query)
    detached = execute_hard(soft_machine.harden(), hard_query)
    assert int(attached.states[:, -1].argmax(-1).item()) == 0
    assert int(attached.answer.argmax(-1).item()) == 0
    assert int(detached.states[:, -1].argmax(-1).item()) == 0
    assert int(detached.answer.item()) == 0


def test_attached_execution_reaches_every_machine_and_query_field() -> None:
    torch.manual_seed(7)
    tensors = {
        "state_active": torch.randn(1, MAX_STATES, 2, requires_grad=True),
        "action_active": torch.randn(1, MAX_ACTIONS, 2, requires_grad=True),
        "observer_active": torch.randn(1, MAX_OBSERVERS, 2, requires_grad=True),
        "action_next": torch.randn(
            1,
            MAX_ACTIONS,
            MAX_STATES,
            MAX_STATES,
            requires_grad=True,
        ),
        "observer_answer": torch.randn(
            1,
            MAX_OBSERVERS,
            MAX_STATES,
            MAX_ANSWERS,
            requires_grad=True,
        ),
        "query_action_path": torch.randn(
            1,
            3,
            MAX_ACTIONS,
            requires_grad=True,
        ),
        "query_start": torch.randn(1, MAX_STATES, requires_grad=True),
        "query_stop": torch.randn(1, 4, requires_grad=True),
        "query_observer": torch.randn(
            1,
            MAX_OBSERVERS,
            requires_grad=True,
        ),
    }
    machine = SoftFunctorMachine(
        state_active=tensors["state_active"],
        action_active=tensors["action_active"],
        observer_active=tensors["observer_active"],
        action_next=tensors["action_next"],
        observer_answer=tensors["observer_answer"],
    )
    query = SoftFunctorQuery(
        start_state=tensors["query_start"],
        action_path=tensors["query_action_path"],
        stop_position=tensors["query_stop"],
        observer=tensors["query_observer"],
    )
    rollout = execute_soft(machine, query)
    weights = torch.arange(MAX_ANSWERS, dtype=rollout.answer.dtype)[None]
    state_weights = torch.arange(MAX_STATES, dtype=rollout.states.dtype)[None]
    loss = (rollout.answer * weights).sum() + (
        rollout.states[:, -1] * state_weights
    ).sum()
    loss.backward()
    for name, tensor in tensors.items():
        assert tensor.grad is not None, name
        assert bool(torch.isfinite(tensor.grad).all()), name
        assert float(tensor.grad.abs().sum()) > 0.0, name


def test_detached_machine_is_invariant_to_soft_source_mutation() -> None:
    hard_machine = _hard_machine()
    hard_query = _hard_query((1, 0))
    soft_machine, _ = _soft_from_hard(hard_machine, hard_query)
    sealed = soft_machine.harden()
    before = execute_hard(sealed, hard_query)
    with torch.no_grad():
        for field in (
            "state_active",
            "action_active",
            "observer_active",
            "action_next",
            "observer_answer",
        ):
            getattr(soft_machine, field).normal_()
    after = execute_hard(sealed, hard_query)
    assert torch.equal(before.states, after.states)
    assert torch.equal(before.answer, after.answer)


def test_semantic_payload_roundtrip_is_exact_and_closed() -> None:
    machine = _hard_machine()
    payload = machine.row_bytes(0)
    assert len(payload) == SEMANTIC_BYTES_PER_ROW
    restored = HardFunctorMachine.from_row_bytes(payload)
    for field in (
        "state_active",
        "action_active",
        "observer_active",
        "action_next",
        "observer_answer",
    ):
        assert torch.equal(getattr(machine, field), getattr(restored, field))
    with pytest.raises(FunctorMachineError, match="wrong byte count"):
        HardFunctorMachine.from_row_bytes(payload[:-1])


def test_transition_and_binding_interventions_have_distinct_effects() -> None:
    machine = _hard_machine()
    query = _hard_query((0, 1))
    baseline = execute_hard(machine, query)
    permutation = (1, 0, 2, 3, 4, 5, 6, 7)
    transition_only = machine.permute_action_transitions(permutation)
    changed = execute_hard(transition_only, query)
    compensated = execute_hard(
        transition_only,
        query.remap_actions(permutation),
    )
    assert int(changed.answer.item()) != int(baseline.answer.item())
    assert torch.equal(compensated.states, baseline.states)
    assert torch.equal(compensated.answer, baseline.answer)


def test_noninvolutive_transition_permutation_has_exact_compensation() -> None:
    machine = _hard_machine()
    query = _hard_query((0, 1, 0))
    permutation = (1, 2, 0, 3, 4, 5, 6, 7)
    moved = machine.permute_action_transitions(permutation)
    compensated = execute_hard(moved, query.remap_actions(permutation))
    baseline = execute_hard(machine, query)
    assert torch.equal(compensated.states, baseline.states)
    assert torch.equal(compensated.answer, baseline.answer)


def test_local_transition_transplant_changes_only_reaching_path() -> None:
    machine = _hard_machine()
    changed = machine.transplant_transition_cell(
        row=0,
        action=0,
        state=1,
        destination=0,
    )
    reaches = _hard_query((1, 0))
    misses = _hard_query((0,), stop=1)
    assert int(execute_hard(machine, reaches).answer.item()) == 2
    assert int(execute_hard(changed, reaches).answer.item()) == 0
    assert torch.equal(
        execute_hard(machine, misses).answer,
        execute_hard(changed, misses).answer,
    )


def test_reset_each_step_collapses_composition() -> None:
    machine = _hard_machine()
    query = _hard_query((1, 0))
    normal = execute_hard(machine, query)
    reset = execute_hard(machine, query, reset_each_step=True)
    assert int(normal.answer.item()) == 2
    assert int(reset.answer.item()) == 0


def test_inactive_action_and_observer_fail_closed() -> None:
    machine = _hard_machine()
    with pytest.raises(FunctorMachineError, match="inactive action"):
        execute_hard(machine, _hard_query((4,), stop=1))
    bad_observer = _hard_query((0,), stop=1, observer=3)
    with pytest.raises(FunctorMachineError, match="inactive observer"):
        execute_hard(machine, bad_observer)
    with pytest.raises(FunctorMachineError, match="inactive start state"):
        execute_hard(machine, _hard_query((0,), stop=1, start=7))


def test_invalid_transition_destination_fails_closed() -> None:
    machine = _hard_machine()
    action_next = machine.action_next.clone()
    action_next[0, 0, 0] = 7
    with pytest.raises(FunctorMachineError, match="inactive state"):
        replace(machine, action_next=action_next)


def test_out_of_domain_machine_bytes_fail_closed_before_indexing() -> None:
    machine = _hard_machine()
    action_next = machine.action_next.clone()
    action_next[0, 0, 0] = MAX_STATES
    with pytest.raises(FunctorMachineError, match="transition destination leaves"):
        replace(machine, action_next=action_next)


def test_second_invalid_batch_row_cannot_hide_behind_first_row() -> None:
    machine = _hard_machine()
    values = {
        field: getattr(machine, field).repeat(
            (2,) + (1,) * (getattr(machine, field).ndim - 1)
        )
        for field in (
            "state_active",
            "action_active",
            "observer_active",
            "action_next",
            "observer_answer",
        )
    }
    values["state_active"][1, 2] = 0
    with pytest.raises(FunctorMachineError, match="inactive state"):
        HardFunctorMachine(**values)


def _little_endian_key_rows(
    values: tuple[int, ...],
    capacity: int,
) -> torch.Tensor:
    payload = b"".join(value.to_bytes(8, "little") for value in values)
    payload += b"\0" * (8 * (capacity - len(values)))
    return torch.tensor(tuple(payload), dtype=torch.uint8).reshape(1, capacity, 8)


def _wire_machine_and_keys() -> tuple[HardFunctorMachine, HardFunctorKeys]:
    state_active = torch.zeros((1, MAX_STATES), dtype=torch.uint8)
    action_active = torch.zeros((1, MAX_ACTIONS), dtype=torch.uint8)
    observer_active = torch.zeros((1, MAX_OBSERVERS), dtype=torch.uint8)
    state_active[:, :5] = 1
    action_active[:, :3] = 1
    observer_active[:, :2] = 1
    action_next = torch.zeros(
        (1, MAX_ACTIONS, MAX_STATES),
        dtype=torch.uint8,
    )
    for action, shift in enumerate((1, 2, 4)):
        action_next[0, action, :5] = torch.tensor(
            tuple((state + shift) % 5 for state in range(5)),
            dtype=torch.uint8,
        )
    observer_answer = torch.zeros(
        (1, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
    )
    observer_answer[0, 0, :5] = torch.arange(5, dtype=torch.uint8)
    observer_answer[0, 1, :5] = torch.tensor(
        (0, 1, 0, 1, 0),
        dtype=torch.uint8,
    )
    machine = HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )
    keys = HardFunctorKeys(
        state_keys=_little_endian_key_rows((101, 102, 103, 104, 105), MAX_STATES),
        action_keys=_little_endian_key_rows((201, 202, 203), MAX_ACTIONS),
        observer_keys=_little_endian_key_rows((301, 302), MAX_OBSERVERS),
    )
    return machine, keys


def test_exact_deployed_wire_adapter_matches_strict_decoder() -> None:
    from pipeline.episode_functor_wire_protocol import (
        WireProtocolSpec,
        decode_deployed_machine,
    )

    machine, keys = _wire_machine_and_keys()
    payload = machine.deployed_wire(keys, 0)
    assert len(payload) == DEPLOYED_MACHINE_BYTES
    spec = WireProtocolSpec(
        state_count=5,
        action_count=3,
        observer_count=2,
        answer_count=5,
    )
    tables = decode_deployed_machine(payload, spec)
    assert tables.state_keys == (101, 102, 103, 104, 105)
    assert tables.action_keys == (201, 202, 203)
    assert tables.observer_keys == (301, 302)
    assert tables.transitions == tuple(
        tuple(int(cell) for cell in machine.action_next[0, action, :5])
        for action in range(3)
    )
    assert tables.observations == tuple(
        tuple(int(cell) for cell in machine.observer_answer[0, observer, :5])
        for observer in range(2)
    )


def test_primary_k8_y4_wire_uses_separate_learned_dimensions() -> None:
    from pipeline.episode_functor_wire_protocol import decode_deployed_machine

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
    for action, shift in enumerate((1, 3, 5)):
        action_next[0, action, :8] = torch.tensor(
            tuple((state + shift) % 8 for state in range(8)),
            dtype=torch.uint8,
        )
    observer_answer = torch.zeros(
        (1, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
    )
    observer_answer[0, 0, :8] = torch.tensor(
        (0, 0, 1, 1, 2, 2, 3, 3),
        dtype=torch.uint8,
    )
    observer_answer[0, 1, :8] = torch.tensor(
        (3, 2, 1, 0, 3, 2, 1, 0),
        dtype=torch.uint8,
    )
    machine = HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )
    keys = HardFunctorKeys(
        state_keys=_little_endian_key_rows(tuple(range(101, 109)), MAX_STATES),
        action_keys=_little_endian_key_rows((201, 202, 203), MAX_ACTIONS),
        observer_keys=_little_endian_key_rows((301, 302), MAX_OBSERVERS),
    )
    spec = LearnedFunctorWireSpec()
    tables = decode_deployed_machine(machine.deployed_wire(keys, 0), spec)
    assert len(tables.state_keys) == 8
    assert max(answer for row in tables.observations for answer in row) == 3


def test_deployed_wire_rejects_noncanonical_masks_and_key_padding() -> None:
    machine, keys = _wire_machine_and_keys()
    action_active = machine.action_active.clone()
    action_active[0, 2] = 0
    action_active[0, 3] = 1
    action_next = machine.action_next.clone()
    action_next[0, 2] = 0
    action_next[0, 3, :5] = torch.arange(5, dtype=torch.uint8)
    noncanonical = replace(
        machine,
        action_active=action_active,
        action_next=action_next,
    )
    with pytest.raises(FunctorMachineError, match="prefix-canonical"):
        noncanonical.deployed_wire(keys, 0)

    state_keys = keys.state_keys.clone()
    state_keys[0, 7, 0] = 1
    bad_keys = replace(keys, state_keys=state_keys)
    with pytest.raises(FunctorMachineError, match="padding is nonzero"):
        machine.deployed_wire(bad_keys, 0)


def test_soft_hardening_rejects_nonprefix_active_masks() -> None:
    machine, query = _soft_from_hard(_hard_machine(), _hard_query())
    with torch.no_grad():
        machine.action_active[0, 0] = torch.tensor((20.0, -20.0))
        machine.action_active[0, 2] = torch.tensor((-20.0, 20.0))
    with pytest.raises(FunctorMachineError, match="prefix-canonical"):
        machine.harden()
    del query
