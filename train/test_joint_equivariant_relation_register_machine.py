from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from equivariant_relation_register_machine import (
    MAX_OBJECTS,
    PHASE_COUNT,
    READ_ONLY_REGISTERS,
    REGISTER_COUNT,
    DeletedRelationRegisterPacket,
    LateRelationRegisterQuery,
    RelationOperation,
)
from joint_equivariant_relation_register_machine import (
    HALT_ACTION_INDEX,
    JOINT_ACTION_COUNT,
    LEGAL_TRANSITION_COUNT,
    JointEquivariantRelationRegisterMachine,
    decode_legal_transition,
    encode_legal_transition,
    expected_joint_controller_parameters,
    joint_controller_parameter_receipt,
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
    registers[:, 0] = torch.eye(MAX_OBJECTS)
    registers[:, 1, 0, 1] = 1
    registers[:, 1, 1, 2] = 1
    registers[:, 1, 2, 3] = 1
    registers[:, 2] = torch.eye(MAX_OBJECTS)
    registers *= _active(cardinality, batch)[:, None]
    return DeletedRelationRegisterPacket(
        cardinality=torch.full((batch,), cardinality, dtype=torch.uint8),
        registers=registers,
    )


def _fixed_logits(
    machine: JointEquivariantRelationRegisterMachine,
    indices: tuple[int, ...],
) -> None:
    with torch.no_grad():
        machine.joint_head.weight.zero_()
        machine.joint_head.bias.fill_(-100.0)
        for index in indices:
            machine.joint_head.bias[index] = 100.0


def test_legal_tuple_mapping_is_a_bijection() -> None:
    seen: set[int] = set()
    for index in range(LEGAL_TRANSITION_COUNT):
        transition = decode_legal_transition(index)
        encoded = encode_legal_transition(
            transition.operation,
            transition.left,
            transition.right,
            transition.destination,
            transition.next_phase,
        )
        assert encoded == index
        assert READ_ONLY_REGISTERS <= transition.destination < REGISTER_COUNT
        seen.add(encoded)
    assert seen == set(range(LEGAL_TRANSITION_COUNT))
    assert HALT_ACTION_INDEX == LEGAL_TRANSITION_COUNT
    assert JOINT_ACTION_COUNT == LEGAL_TRANSITION_COUNT + 1


def test_hard_joint_one_hot_decodes_and_executes_exact_tuple() -> None:
    machine = JointEquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=1,
    )
    selected = encode_legal_transition(
        int(RelationOperation.COPY),
        1,
        0,
        3,
        2,
    )
    _fixed_logits(machine, (selected,))
    packet = _packet()
    result = machine(
        packet,
        LateRelationRegisterQuery(
            register=torch.tensor([3]),
            position=torch.tensor([0]),
        ),
        hard=True,
    )
    action = result.actions[0]
    expected_joint = F.one_hot(
        torch.tensor([selected]),
        JOINT_ACTION_COUNT,
    ).float()
    assert torch.equal(action.joint, expected_joint)
    assert torch.equal(
        action.operation,
        F.one_hot(
            torch.tensor([int(RelationOperation.COPY)]),
            len(RelationOperation),
        ).float(),
    )
    assert torch.equal(
        action.left,
        F.one_hot(torch.tensor([1]), REGISTER_COUNT).float(),
    )
    assert torch.equal(
        action.right,
        F.one_hot(torch.tensor([0]), REGISTER_COUNT).float(),
    )
    assert torch.equal(
        action.destination,
        F.one_hot(torch.tensor([3]), REGISTER_COUNT).float(),
    )
    assert torch.equal(
        action.phase,
        F.one_hot(torch.tensor([2]), PHASE_COUNT).float(),
    )
    expected_registers = packet.registers.clone()
    expected_registers[:, 3] = packet.registers[:, 1]
    assert torch.equal(result.final_registers, expected_registers)
    assert not bool(result.halted_by_deadline.item())


def test_soft_execution_mixes_whole_tuples_without_cross_combinations() -> None:
    machine = JointEquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=1,
    )
    copy_to_three = encode_legal_transition(
        int(RelationOperation.COPY),
        1,
        0,
        3,
        0,
    )
    identity_to_four = encode_legal_transition(
        int(RelationOperation.IDENTITY),
        0,
        0,
        4,
        1,
    )
    _fixed_logits(machine, (copy_to_three, identity_to_four))
    packet = _packet()
    result = machine(
        packet,
        LateRelationRegisterQuery(
            register=torch.tensor([3]),
            position=torch.tensor([0]),
        ),
        hard=False,
    )
    copy_state = packet.registers.clone()
    copy_state[:, 3] = packet.registers[:, 1]
    identity_state = packet.registers.clone()
    identity_state[:, 4] = packet.registers[:, 2]
    expected = 0.5 * copy_state + 0.5 * identity_state
    assert torch.allclose(
        result.final_registers,
        expected,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        result.actions[0].transition.sum(),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_wrong_hard_joint_action_receives_finite_corrective_gradient() -> None:
    machine = JointEquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=1,
    )
    wrong = encode_legal_transition(
        int(RelationOperation.CLEAR),
        0,
        0,
        3,
        0,
    )
    target = encode_legal_transition(
        int(RelationOperation.EXPAND),
        1,
        2,
        5,
        2,
    )
    _fixed_logits(machine, (wrong,))
    result = machine(
        _packet(),
        LateRelationRegisterQuery(
            register=torch.tensor([5]),
            position=torch.tensor([0]),
        ),
        hard=True,
    )
    loss = F.cross_entropy(
        result.actions[0].joint_logits,
        torch.tensor([target]),
    )
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert machine.joint_head.bias.grad is not None
    assert bool(torch.isfinite(machine.joint_head.bias.grad).all())
    assert float(machine.joint_head.bias.grad.abs().sum()) > 0.0
    assert machine.joint_head.weight.grad is not None
    assert bool(torch.isfinite(machine.joint_head.weight.grad).all())
    assert float(machine.joint_head.weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("hard", [False, True])
def test_joint_controller_is_equivariant_to_object_reindexing(hard: bool) -> None:
    torch.manual_seed(7301)
    machine = JointEquivariantRelationRegisterMachine(
        controller_width=48,
        controller_layers=2,
        maximum_steps=4,
    )
    packet = _packet()
    query = LateRelationRegisterQuery(
        register=torch.tensor([1]),
        position=torch.tensor([0]),
    )
    first = machine(packet, query, hard=hard)

    permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7])
    inverse = permutation.argsort()
    permuted = DeletedRelationRegisterPacket(
        packet.cardinality,
        packet.registers.index_select(
            -2,
            permutation,
        ).index_select(-1, permutation),
    )
    second = machine(
        permuted,
        LateRelationRegisterQuery(
            register=query.register,
            position=inverse[query.position],
        ),
        hard=hard,
    )
    restored = second.final_registers.index_select(
        -2,
        inverse,
    ).index_select(-1, inverse)
    assert torch.allclose(
        first.final_registers,
        restored,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        first.answer,
        second.answer.index_select(-1, inverse),
        atol=1e-6,
        rtol=0.0,
    )
    for first_action, second_action in zip(
        first.actions,
        second.actions,
        strict=True,
    ):
        assert torch.allclose(
            first_action.joint,
            second_action.joint,
            atol=1e-6,
            rtol=0.0,
        )


def test_exact_parameter_accounting_and_strict_cap() -> None:
    width = 64
    layers = 2
    machine = JointEquivariantRelationRegisterMachine(
        controller_width=width,
        controller_layers=layers,
    )
    manual = sum(parameter.numel() for parameter in machine.parameters())
    expected = expected_joint_controller_parameters(
        controller_width=width,
        controller_layers=layers,
    )
    receipt = joint_controller_parameter_receipt(machine)
    assert expected == manual == machine.added_parameters
    assert receipt["added"] == expected
    assert receipt["complete_system"] == receipt["base"] + expected
    assert receipt["complete_system"] < receipt["strict_cap"]
    assert receipt["headroom"] == (
        receipt["strict_cap"] - receipt["complete_system"]
    )
