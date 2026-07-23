from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from equivariant_relation_register_machine import (
    MAX_OBJECTS,
    OPERATION_COUNT,
    REGISTER_COUNT,
    DeletedRelationRegisterPacket,
    EquivariantRelationRegisterMachine,
    LateRelationRegisterQuery,
    RelationOperation,
    RelationRegisterError,
    boolean_relation_compose,
    controller_parameter_receipt,
    relation_algebra_candidates,
)


def _active(cardinality: int, batch: int = 1) -> torch.Tensor:
    positions = torch.arange(MAX_OBJECTS)
    objects = positions[None] < torch.full((batch, 1), cardinality)
    return objects[:, :, None] & objects[:, None, :]


def _packet(cardinality: int = 4, batch: int = 1) -> DeletedRelationRegisterPacket:
    registers = torch.zeros(
        batch,
        REGISTER_COUNT,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    identity = torch.eye(MAX_OBJECTS)
    registers[:, 0] = identity
    registers[:, 1, 0, 1] = 1
    registers[:, 1, 1, 2] = 1
    registers[:, 1, 2, 3] = 1
    registers *= _active(cardinality, batch)[:, None]
    return DeletedRelationRegisterPacket(
        cardinality=torch.full((batch,), cardinality, dtype=torch.uint8),
        registers=registers,
    )


def test_boolean_composition_and_all_algebra_candidates() -> None:
    left = torch.zeros(1, MAX_OBJECTS, MAX_OBJECTS)
    right = torch.zeros_like(left)
    left[0, 0, 1] = 1
    left[0, 1, 2] = 1
    right[0, 1, 3] = 1
    right[0, 2, 0] = 1
    active = _active(4)
    composed = boolean_relation_compose(left, right)
    assert composed[0, 0, 3] == 1
    assert composed[0, 1, 0] == 1
    assert int(composed.sum()) == 2

    candidates = relation_algebra_candidates(left, right, active)
    assert candidates.shape == (
        1,
        OPERATION_COUNT,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    assert torch.equal(
        candidates[:, RelationOperation.COMPOSE],
        composed,
    )
    assert candidates[0, RelationOperation.UNION, 0, 1] == 1
    assert candidates[0, RelationOperation.INTERSECTION].sum() == 0
    assert candidates[0, RelationOperation.DIFFERENCE, 0, 1] == 1
    assert candidates[0, RelationOperation.CONVERSE, 1, 0] == 1
    assert torch.equal(
        candidates[:, RelationOperation.COPY],
        left,
    )
    assert candidates[0, RelationOperation.CLEAR].sum() == 0
    assert torch.equal(
        candidates[:, RelationOperation.IDENTITY],
        torch.eye(MAX_OBJECTS)[None] * active,
    )
    assert torch.equal(
        candidates[:, RelationOperation.EXPAND],
        1.0 - (1.0 - right) * (1.0 - composed),
    )


def test_difference_is_genuinely_antitone_in_right_operand() -> None:
    left = torch.ones(1, MAX_OBJECTS, MAX_OBJECTS) * _active(3)
    small = torch.zeros_like(left)
    large = small.clone()
    large[0, 0, 1] = 1
    first = relation_algebra_candidates(left, small, _active(3))[
        :, RelationOperation.DIFFERENCE
    ]
    second = relation_algebra_candidates(left, large, _active(3))[
        :, RelationOperation.DIFFERENCE
    ]
    assert second[0, 0, 1] < first[0, 0, 1]
    assert bool((second <= first).all())


def test_packet_rejects_outside_cardinality_scratch_state() -> None:
    packet = _packet(3)
    bad = packet.registers.clone()
    bad[0, 0, 0, 7] = 1
    with pytest.raises(RelationRegisterError, match="covert outside state"):
        DeletedRelationRegisterPacket(packet.cardinality, bad)


def test_missing_halt_remains_invalid_instead_of_being_repaired() -> None:
    machine = EquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=4,
    )
    with torch.no_grad():
        machine.halt_head.weight.zero_()
        machine.halt_head.bias.copy_(torch.tensor([20.0, -20.0]))
    packet = _packet()
    result = machine(
        packet,
        LateRelationRegisterQuery(
            register=torch.tensor([0]),
            position=torch.tensor([0]),
        ),
        hard=True,
    )
    assert not bool(result.halted_by_deadline.item())
    assert result.alive_trajectory[-1].item() == pytest.approx(1.0)
    assert all(
        bool(action.phase.eq(0).logical_or(action.phase.eq(1)).all())
        and torch.equal(
            action.phase.sum(-1),
            torch.ones(packet.batch_size),
        )
        for action in result.actions
    )


def test_halt_first_commits_input_and_blocks_all_selected_operations() -> None:
    machine = EquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=4,
    )
    with torch.no_grad():
        machine.halt_head.weight.zero_()
        machine.halt_head.bias.copy_(torch.tensor([-20.0, 20.0]))
    packet = _packet()
    result = machine(
        packet,
        LateRelationRegisterQuery(
            register=torch.tensor([1]),
            position=torch.tensor([0]),
        ),
        hard=True,
    )
    assert bool(result.halted_by_deadline.item())
    assert torch.equal(result.final_registers, packet.registers)
    assert result.alive_trajectory[0].item() == pytest.approx(0.0)
    assert all(
        torch.equal(state, packet.registers)
        for state in result.register_trajectory
    )


def test_controller_is_equivariant_to_object_reindexing() -> None:
    torch.manual_seed(73)
    machine = EquivariantRelationRegisterMachine(
        controller_width=48,
        controller_layers=2,
        maximum_steps=6,
    )
    packet = _packet()
    query = LateRelationRegisterQuery(
        register=torch.tensor([1]),
        position=torch.tensor([0]),
    )
    first = machine(packet, query, hard=True)

    permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7])
    inverse = permutation.argsort()
    permuted_registers = packet.registers.index_select(
        -2,
        permutation,
    ).index_select(-1, permutation)
    permuted = DeletedRelationRegisterPacket(
        packet.cardinality,
        permuted_registers,
    )
    permuted_query = LateRelationRegisterQuery(
        register=query.register,
        position=inverse[query.position],
    )
    second = machine(permuted, permuted_query, hard=True)
    restored = second.final_registers.index_select(
        -2,
        inverse,
    ).index_select(-1, inverse)
    assert torch.equal(first.final_registers, restored)
    assert torch.equal(
        first.answer,
        second.answer.index_select(-1, inverse),
    )
    for left_action, right_action in zip(
        first.actions,
        second.actions,
        strict=True,
    ):
        assert torch.equal(left_action.operation, right_action.operation)
        assert torch.equal(left_action.left, right_action.left)
        assert torch.equal(left_action.right, right_action.right)
        assert torch.equal(left_action.destination, right_action.destination)
        assert torch.equal(left_action.halt, right_action.halt)
        assert torch.equal(left_action.phase, right_action.phase)


def test_soft_controller_has_finite_end_to_end_gradients() -> None:
    torch.manual_seed(101)
    machine = EquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=5,
    )
    packet = _packet(batch=2)
    query = LateRelationRegisterQuery(
        register=torch.tensor([4, 5]),
        position=torch.tensor([0, 2]),
    )
    result = machine(packet, query)
    assert torch.allclose(
        result.final_registers[:, :3],
        packet.registers[:, :3],
        atol=1e-6,
        rtol=0.0,
    )
    target = torch.zeros_like(result.answer)
    loss = F.binary_cross_entropy(
        result.answer.clamp(1e-6, 1.0 - 1e-6),
        target,
    ) + result.alive_trajectory[-1].mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in machine.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    assert any(
        gradient is not None and float(gradient.abs().sum()) > 0
        for gradient in gradients
    )


def test_soft_rollout_preserves_probability_and_read_only_domains() -> None:
    torch.manual_seed(211)
    machine = EquivariantRelationRegisterMachine(
        controller_width=48,
        controller_layers=2,
        maximum_steps=8,
    )
    packet = _packet(cardinality=4, batch=16)
    result = machine(
        packet,
        LateRelationRegisterQuery(
            register=torch.full((16,), 5, dtype=torch.long),
            position=torch.arange(16, dtype=torch.long).remainder(4),
        ),
        hard=False,
    )
    for state in (*result.register_trajectory, result.final_registers):
        assert bool(torch.isfinite(state).all())
        assert float(state.detach().min()) >= 0.0
        assert float(state.detach().max()) <= 1.0 + 1e-6
        assert torch.allclose(
            state[:, :3],
            packet.registers[:, :3],
            atol=1e-6,
            rtol=0.0,
        )
    assert float(result.answer.detach().min()) >= 0.0
    assert float(result.answer.detach().max()) <= 1.0 + 1e-6
    assert all(
        torch.allclose(
            action.phase.sum(-1),
            torch.ones(packet.batch_size),
            atol=1e-6,
            rtol=0.0,
        )
        for action in result.actions
    )


def test_parameter_receipt_leaves_large_headroom() -> None:
    machine = EquivariantRelationRegisterMachine()
    receipt = controller_parameter_receipt(machine)
    assert receipt["added"] == machine.added_parameters
    assert receipt["complete_system"] < receipt["strict_cap"]
    assert receipt["headroom"] > 60_000_000
