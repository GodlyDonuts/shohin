from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from contextual_bekic_graph_machine import (
    MAX_OPERATION_SLOTS,
    MAX_PROGRAM_CONSTANTS,
    MAX_PROGRAM_NODES,
    PROGRAM_PRIMITIVE_COUNT,
    ContextualBekicGraphMachine,
    ContextualProgramError,
    DeletedContextualProgramPacket,
    LateContextualProgramQuery,
    ProgramNodeKind,
    ProgramPrimitive,
    contextual_graph_parameter_receipt,
)
from contextual_relation_primitive_compiler import (
    ContextualRelationPrimitiveCompiler,
    relation_primitive_candidates,
)
from equivariant_relation_register_machine import MAX_OBJECTS


def _active(cardinality: int) -> torch.Tensor:
    positions = torch.arange(MAX_OBJECTS)
    active = positions < cardinality
    return active[:, None] & active[None, :]


def _program_packet(
    *,
    cardinality: int = 4,
) -> tuple[DeletedContextualProgramPacket, torch.Tensor]:
    """Build X=I union A∘X; Y=B union (X intersect converse(B))."""

    constants = torch.zeros(
        1,
        MAX_PROGRAM_CONSTANTS,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    constants[0, 0, 1, 0] = 1
    constants[0, 0, 2, 1] = 1
    constants[0, 0, 3, 2] = 1
    constants[0, 1, 0, 1] = 1
    constants[0, 1, 1, 2] = 1
    constants *= _active(cardinality)[None, None]
    constant_valid = torch.zeros(1, MAX_PROGRAM_CONSTANTS, dtype=torch.bool)
    constant_valid[:, :2] = True

    # 0:A 1:B 2:X 3:Y 4:I 5:A∘X 6:X-root 7:converse(B)
    # 8:X∩converse(B) 9:Y-root
    count = 10
    node_valid = torch.zeros(1, MAX_PROGRAM_NODES, dtype=torch.bool)
    node_valid[:, :count] = True
    node_valid[:, 3] = False
    node_kind = torch.full((1, MAX_PROGRAM_NODES), -1, dtype=torch.long)
    node_kind[:, 0:2] = int(ProgramNodeKind.CONSTANT)
    node_kind[:, 2] = int(ProgramNodeKind.VARIABLE)
    node_kind[:, 4:count] = int(ProgramNodeKind.OPERATION)
    constant_index = torch.full_like(node_kind, -1)
    constant_index[0, 0] = 0
    constant_index[0, 1] = 1
    variable_index = torch.full_like(node_kind, -1)
    variable_index[0, 2] = 0
    operation_slot = torch.full_like(node_kind, -1)
    # slots 0..4 map identity, compose, union, converse, intersection
    operation_slot[0, 4] = 0
    operation_slot[0, 5] = 1
    operation_slot[0, 6] = 2
    operation_slot[0, 7] = 3
    operation_slot[0, 8] = 4
    operation_slot[0, 9] = 2
    left = torch.full_like(node_kind, -1)
    right = torch.full_like(node_kind, -1)
    left[0, 5], right[0, 5] = 0, 2
    left[0, 6], right[0, 6] = 4, 5
    left[0, 7] = 1
    left[0, 8], right[0, 8] = 2, 7
    left[0, 9], right[0, 9] = 1, 8
    roots = torch.tensor([[6, 9]], dtype=torch.long)
    slot_arity = torch.full(
        (1, MAX_OPERATION_SLOTS),
        -1,
        dtype=torch.long,
    )
    slot_arity[0, :5] = torch.tensor([0, 2, 2, 1, 2])
    packet = DeletedContextualProgramPacket(
        cardinality=torch.tensor([cardinality], dtype=torch.uint8),
        constants=constants,
        constant_valid=constant_valid,
        node_valid=node_valid,
        node_kind=node_kind,
        constant_index=constant_index,
        variable_index=variable_index,
        operation_slot=operation_slot,
        left_index=left,
        right_index=right,
        equation_root=roots,
        slot_arity=slot_arity,
    )
    primitive = torch.zeros(
        1,
        MAX_OPERATION_SLOTS,
        PROGRAM_PRIMITIVE_COUNT,
    )
    mapping = (
        ProgramPrimitive.IDENTITY,
        ProgramPrimitive.COMPOSE,
        ProgramPrimitive.UNION,
        ProgramPrimitive.CONVERSE,
        ProgramPrimitive.INTERSECTION,
    )
    for slot, operation in enumerate(mapping):
        primitive[0, slot, int(operation)] = 1
    return packet, primitive


def _machine() -> ContextualBekicGraphMachine:
    return ContextualBekicGraphMachine(
        expression_ticks=12,
        fixed_point_steps=12,
    )


def test_private_graph_executes_recursive_program_exactly() -> None:
    packet, primitive = _program_packet()
    result = _machine()(
        packet,
        primitive,
        LateContextualProgramQuery(
            variable=torch.tensor([0]),
            position=torch.tensor([3]),
        ),
    )
    identity = torch.eye(MAX_OBJECTS) * _active(4)
    expected_x = identity.clone()
    expected_x[1, 0] = 1
    expected_x[2, 0] = 1
    expected_x[2, 1] = 1
    expected_x[3, 0] = 1
    expected_x[3, 1] = 1
    expected_x[3, 2] = 1
    expected_y = packet.constants[0, 1].clone()
    expected_y[1, 0] = 1
    expected_y[2, 1] = 1
    assert result.converged.item()
    assert torch.equal(result.terminal_variables[0, 0], expected_x)
    assert torch.equal(result.terminal_variables[0, 1], expected_y)
    assert torch.equal(result.answer[0], expected_x[3])


def test_fresh_opaque_cards_compile_and_drive_recursive_graph() -> None:
    packet, expected_assignment = _program_packet()
    witnesses = 8
    generator = torch.Generator().manual_seed(2026072311)
    left = torch.zeros(
        1,
        MAX_OPERATION_SLOTS,
        witnesses,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    right = torch.zeros_like(left)
    left[..., :4, :4] = torch.randint(
        0,
        2,
        (1, MAX_OPERATION_SLOTS, witnesses, 4, 4),
        generator=generator,
    ).float()
    right[..., :4, :4] = torch.randint(
        0,
        2,
        (1, MAX_OPERATION_SLOTS, witnesses, 4, 4),
        generator=generator,
    ).float()
    object_mask = torch.zeros(1, MAX_OBJECTS, dtype=torch.bool)
    object_mask[:, :4] = True
    candidates = relation_primitive_candidates(left, right, object_mask)
    primitive_index = expected_assignment.argmax(-1)
    output = candidates.gather(
        3,
        primitive_index[:, :, None, None, None, None].expand(
            -1,
            -1,
            witnesses,
            1,
            MAX_OBJECTS,
            MAX_OBJECTS,
        ),
    ).squeeze(3)
    witness_mask = torch.zeros(
        1,
        MAX_OPERATION_SLOTS,
        witnesses,
        dtype=torch.bool,
    )
    witness_mask[:, :5] = True
    argument_mask = (
        torch.arange(2)[None, None, None]
        < packet.slot_arity.clamp_min(0)[..., None, None]
    ).expand(-1, -1, witnesses, -1)
    compiler = ContextualRelationPrimitiveCompiler()
    compiled = compiler(
        left,
        right,
        output,
        witness_mask,
        argument_mask,
        object_mask,
        hard=True,
    )
    assert bool(compiled.identifiable[:, :5].all())
    assert not bool(compiled.identifiable[:, 5:].any())
    assert torch.equal(compiled.discrete_assignment, expected_assignment)

    query = LateContextualProgramQuery(
        variable=torch.tensor([0]),
        position=torch.tensor([3]),
    )
    expected = _machine()(packet, expected_assignment, query)
    observed = _machine()(packet, compiled.discrete_assignment, query)
    assert torch.equal(observed.terminal_variables, expected.terminal_variables)
    assert torch.equal(observed.answer, expected.answer)


def test_node_reindexing_preserves_execution() -> None:
    packet, primitive = _program_packet()
    query = LateContextualProgramQuery(
        variable=torch.tensor([1]),
        position=torch.tensor([2]),
    )
    original = _machine()(packet, primitive, query)
    permutation = torch.arange(MAX_PROGRAM_NODES)
    permutation[:10] = torch.tensor([7, 2, 9, 0, 6, 4, 1, 8, 3, 5])
    inverse = permutation.argsort()

    def permute_nodes(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(1, permutation)

    left = permute_nodes(packet.left_index)
    right = permute_nodes(packet.right_index)
    left = torch.where(left.ge(0), inverse[left.clamp_min(0)], left)
    right = torch.where(right.ge(0), inverse[right.clamp_min(0)], right)
    roots = inverse[packet.equation_root]
    permuted = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=packet.constants,
        constant_valid=packet.constant_valid,
        node_valid=permute_nodes(packet.node_valid),
        node_kind=permute_nodes(packet.node_kind),
        constant_index=permute_nodes(packet.constant_index),
        variable_index=permute_nodes(packet.variable_index),
        operation_slot=permute_nodes(packet.operation_slot),
        left_index=left,
        right_index=right,
        equation_root=roots,
        slot_arity=packet.slot_arity,
    )
    changed = _machine()(permuted, primitive, query)
    assert torch.equal(original.terminal_variables, changed.terminal_variables)
    assert torch.equal(original.answer, changed.answer)


def test_active_node_and_padding_exchange_preserves_execution() -> None:
    packet, primitive = _program_packet()
    query = LateContextualProgramQuery(
        variable=torch.tensor([0]),
        position=torch.tensor([2]),
    )
    original = _machine()(packet, primitive, query)
    permutation = torch.arange(MAX_PROGRAM_NODES)
    permutation[5], permutation[73] = 73, 5
    inverse = permutation.argsort()

    def permute_nodes(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(1, permutation)

    left = permute_nodes(packet.left_index)
    right = permute_nodes(packet.right_index)
    left = torch.where(left.ge(0), inverse[left.clamp_min(0)], left)
    right = torch.where(right.ge(0), inverse[right.clamp_min(0)], right)
    changed = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=packet.constants,
        constant_valid=packet.constant_valid,
        node_valid=permute_nodes(packet.node_valid),
        node_kind=permute_nodes(packet.node_kind),
        constant_index=permute_nodes(packet.constant_index),
        variable_index=permute_nodes(packet.variable_index),
        operation_slot=permute_nodes(packet.operation_slot),
        left_index=left,
        right_index=right,
        equation_root=inverse[packet.equation_root],
        slot_arity=packet.slot_arity,
    )
    observed = _machine()(changed, primitive, query)
    assert torch.equal(original.terminal_variables, observed.terminal_variables)
    assert torch.equal(original.answer, observed.answer)


def test_object_reindexing_is_equivariant() -> None:
    packet, primitive = _program_packet()
    query = LateContextualProgramQuery(
        variable=torch.tensor([0]),
        position=torch.tensor([1]),
    )
    original = _machine()(packet, primitive, query)
    permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7])
    inverse = permutation.argsort()
    constants = packet.constants.index_select(
        -2,
        permutation,
    ).index_select(-1, permutation)
    changed_packet = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=constants,
        constant_valid=packet.constant_valid,
        node_valid=packet.node_valid,
        node_kind=packet.node_kind,
        constant_index=packet.constant_index,
        variable_index=packet.variable_index,
        operation_slot=packet.operation_slot,
        left_index=packet.left_index,
        right_index=packet.right_index,
        equation_root=packet.equation_root,
        slot_arity=packet.slot_arity,
    )
    changed = _machine()(
        changed_packet,
        primitive,
        LateContextualProgramQuery(
            variable=query.variable,
            position=inverse[query.position],
        ),
    )
    restored = changed.terminal_variables.index_select(
        -2,
        inverse,
    ).index_select(-1, inverse)
    assert torch.equal(original.terminal_variables, restored)
    assert torch.equal(
        original.answer,
        changed.answer.index_select(-1, inverse),
    )


def test_operation_slot_reindexing_preserves_execution() -> None:
    packet, primitive = _program_packet()
    query = LateContextualProgramQuery(
        variable=torch.tensor([0]),
        position=torch.tensor([0]),
    )
    original = _machine()(packet, primitive, query)
    permutation = torch.tensor([3, 4, 1, 0, 2, 5, 6, 7])
    inverse = permutation.argsort()
    slots = torch.where(
        packet.operation_slot.ge(0),
        inverse[packet.operation_slot.clamp_min(0)],
        packet.operation_slot,
    )
    changed_packet = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=packet.constants,
        constant_valid=packet.constant_valid,
        node_valid=packet.node_valid,
        node_kind=packet.node_kind,
        constant_index=packet.constant_index,
        variable_index=packet.variable_index,
        operation_slot=slots,
        left_index=packet.left_index,
        right_index=packet.right_index,
        equation_root=packet.equation_root,
        slot_arity=packet.slot_arity.index_select(1, permutation),
    )
    changed = _machine()(
        changed_packet,
        primitive.index_select(1, permutation),
        query,
    )
    assert torch.equal(original.terminal_variables, changed.terminal_variables)
    assert torch.equal(original.answer, changed.answer)


def test_active_slot_and_padding_exchange_preserves_execution() -> None:
    packet, primitive = _program_packet()
    query = LateContextualProgramQuery(
        variable=torch.tensor([1]),
        position=torch.tensor([1]),
    )
    original = _machine()(packet, primitive, query)
    permutation = torch.tensor([7, 1, 2, 3, 4, 5, 6, 0])
    inverse = permutation.argsort()
    slots = torch.where(
        packet.operation_slot.ge(0),
        inverse[packet.operation_slot.clamp_min(0)],
        packet.operation_slot,
    )
    changed = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=packet.constants,
        constant_valid=packet.constant_valid,
        node_valid=packet.node_valid,
        node_kind=packet.node_kind,
        constant_index=packet.constant_index,
        variable_index=packet.variable_index,
        operation_slot=slots,
        left_index=packet.left_index,
        right_index=packet.right_index,
        equation_root=packet.equation_root,
        slot_arity=packet.slot_arity.index_select(1, permutation),
    )
    observed = _machine()(
        changed,
        primitive.index_select(1, permutation),
        query,
    )
    assert torch.equal(original.terminal_variables, observed.terminal_variables)
    assert torch.equal(original.answer, observed.answer)


def test_variable_alpha_reindexing_preserves_and_swaps_terminal_state() -> None:
    packet, primitive = _program_packet()
    original = _machine()(
        packet,
        primitive,
        LateContextualProgramQuery(
            variable=torch.tensor([0]),
            position=torch.tensor([3]),
        ),
    )
    variable_index = packet.variable_index.clone()
    variable_nodes = variable_index.ge(0)
    variable_index[variable_nodes] = 1 - variable_index[variable_nodes]
    swapped = DeletedContextualProgramPacket(
        cardinality=packet.cardinality,
        constants=packet.constants,
        constant_valid=packet.constant_valid,
        node_valid=packet.node_valid,
        node_kind=packet.node_kind,
        constant_index=packet.constant_index,
        variable_index=variable_index,
        operation_slot=packet.operation_slot,
        left_index=packet.left_index,
        right_index=packet.right_index,
        equation_root=packet.equation_root.flip(1),
        slot_arity=packet.slot_arity,
    )
    changed = _machine()(
        swapped,
        primitive,
        LateContextualProgramQuery(
            variable=torch.tensor([1]),
            position=torch.tensor([3]),
        ),
    )
    assert torch.equal(
        original.terminal_variables,
        changed.terminal_variables.flip(1),
    )
    assert torch.equal(original.answer, changed.answer)


def test_late_query_changes_answer_without_changing_execution() -> None:
    packet, primitive = _program_packet()
    first = _machine()(
        packet,
        primitive,
        LateContextualProgramQuery(
            variable=torch.tensor([0]),
            position=torch.tensor([3]),
        ),
    )
    second = _machine()(
        packet,
        primitive,
        LateContextualProgramQuery(
            variable=torch.tensor([1]),
            position=torch.tensor([0]),
        ),
    )
    assert torch.equal(first.terminal_variables, second.terminal_variables)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.variable_trajectory,
            second.variable_trajectory,
            strict=True,
        )
    )
    assert not torch.equal(first.answer, second.answer)


def test_soft_contextual_binding_receives_gradient() -> None:
    packet, primitive = _program_packet()
    # UNION and INTERSECTION share arity, so keep the assignment legal.
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    probabilities = logits.softmax(-1)
    soft = primitive.clone()
    soft[0, 2, int(ProgramPrimitive.UNION)] = probabilities[0]
    soft[0, 2, int(ProgramPrimitive.INTERSECTION)] = probabilities[1]
    result = _machine()(
        packet,
        soft,
        LateContextualProgramQuery(
            variable=torch.tensor([0]),
            position=torch.tensor([3]),
        ),
        hard=False,
    )
    loss = F.mse_loss(
        result.terminal_variables[0, 0],
        torch.zeros_like(result.terminal_variables[0, 0]),
    )
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0


def test_assignment_arity_mismatch_fails_closed() -> None:
    packet, primitive = _program_packet()
    broken = primitive.clone()
    broken[0, 0].zero_()
    broken[0, 0, int(ProgramPrimitive.CONVERSE)] = 1
    with pytest.raises(ContextualProgramError, match="arity"):
        _machine()(
            packet,
            broken,
            LateContextualProgramQuery(
                variable=torch.tensor([0]),
                position=torch.tensor([0]),
            ),
        )


def test_hard_execution_rejects_soft_assignment_by_default() -> None:
    packet, primitive = _program_packet()
    soft = primitive.clone()
    soft[0, 2, int(ProgramPrimitive.UNION)] = 0.75
    soft[0, 2, int(ProgramPrimitive.INTERSECTION)] = 0.25
    with pytest.raises(ContextualProgramError, match="not discrete"):
        _machine()(
            packet,
            soft,
            LateContextualProgramQuery(
                variable=torch.tensor([0]),
                position=torch.tensor([0]),
            ),
        )


def test_private_packet_rejects_outside_state() -> None:
    packet, _ = _program_packet()
    constants = packet.constants.clone()
    constants[0, 0, 7, 7] = 1
    with pytest.raises(ContextualProgramError, match="outside"):
        DeletedContextualProgramPacket(
            cardinality=packet.cardinality,
            constants=constants,
            constant_valid=packet.constant_valid,
            node_valid=packet.node_valid,
            node_kind=packet.node_kind,
            constant_index=packet.constant_index,
            variable_index=packet.variable_index,
            operation_slot=packet.operation_slot,
            left_index=packet.left_index,
            right_index=packet.right_index,
            equation_root=packet.equation_root,
            slot_arity=packet.slot_arity,
        )


def test_private_packet_rejects_covert_type_irrelevant_state() -> None:
    packet, _ = _program_packet()
    variable_index = packet.variable_index.clone()
    variable_index[0, 0] = 0
    with pytest.raises(ContextualProgramError, match="covert"):
        DeletedContextualProgramPacket(
            cardinality=packet.cardinality,
            constants=packet.constants,
            constant_valid=packet.constant_valid,
            node_valid=packet.node_valid,
            node_kind=packet.node_kind,
            constant_index=packet.constant_index,
            variable_index=variable_index,
            operation_slot=packet.operation_slot,
            left_index=packet.left_index,
            right_index=packet.right_index,
            equation_root=packet.equation_root,
            slot_arity=packet.slot_arity,
        )


def test_cyclic_program_and_short_tick_budget_fail_closed() -> None:
    packet, primitive = _program_packet()
    cyclic_left = packet.left_index.clone()
    cyclic_left[0, 5] = 6
    with pytest.raises(ContextualProgramError, match="cyclic"):
        DeletedContextualProgramPacket(
            cardinality=packet.cardinality,
            constants=packet.constants,
            constant_valid=packet.constant_valid,
            node_valid=packet.node_valid,
            node_kind=packet.node_kind,
            constant_index=packet.constant_index,
            variable_index=packet.variable_index,
            operation_slot=packet.operation_slot,
            left_index=cyclic_left,
            right_index=packet.right_index,
            equation_root=packet.equation_root,
            slot_arity=packet.slot_arity,
        )
    with pytest.raises(ContextualProgramError, match="shorter"):
        ContextualBekicGraphMachine(
            expression_ticks=2,
            fixed_point_steps=12,
        )(
            packet,
            primitive,
            LateContextualProgramQuery(
                variable=torch.tensor([0]),
                position=torch.tensor([0]),
            ),
        )


def test_parameter_receipt_is_exact_and_below_cap() -> None:
    machine = _machine()
    receipt = contextual_graph_parameter_receipt(machine)
    assert machine.added_parameters == 0
    assert receipt == {
        "base": 125_081_664,
        "added": 0,
        "complete_system": 125_081_664,
        "strict_cap": 200_000_000,
        "headroom": 74_918_336,
    }
