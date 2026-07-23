from __future__ import annotations

import dataclasses
import inspect
from functools import lru_cache

import pytest
import torch

import neural_tcrr_board as board
import neural_tcrr_motor as motor
import tensorize_neural_tcrr_packets as packet_tensors


@lru_cache(maxsize=1)
def _slice() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _packet_map() -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(packet): packet for packet in _slice().packets}


def _small_motor() -> motor.NeuralTcrrMotor:
    torch.manual_seed(20260723)
    value = motor.NeuralTcrrMotor(
        motor.NeuralTcrrMotorConfig(
            hidden_dim=32,
            entity_rounds=1,
            term_rounds=1,
            graph_rounds=1,
            max_arity=packet_tensors.A,
            path_depth=packet_tensors.D,
        )
    )
    value.eval()
    return value


_PACKET_AXES: dict[str, tuple[str, ...]] = {
    "constructor_active": ("C",),
    "constructor_equal": ("C", "C"),
    "constructor_result_type": ("C", "Y"),
    "constructor_argument_type": ("C", "-", "Y"),
    "constructor_argument_mask": ("C", "-"),
    "type_active": ("Y",),
    "type_equal": ("Y", "Y"),
    "rule_active": ("R",),
    "rule_equal": ("R", "R"),
    "rule_delete": ("R",),
    "variable_active": ("V",),
    "variable_equal": ("V", "V"),
    "storage_active": ("N",),
    "storage_equal": ("N", "N"),
    "graph_active": ("N",),
    "graph_root": ("N1",),
    "graph_kind": ("N", "-"),
    "graph_type": ("N", "Y"),
    "graph_constructor": ("N", "C"),
    "graph_variable": ("N", "V"),
    "graph_children": ("N", "-", "N1"),
    "graph_child_type": ("N", "-", "Y"),
    "graph_child_mask": ("N", "-"),
    "graph_capacity": ("N",),
    "lhs_active": ("R", "-"),
    "lhs_kind": ("R", "-", "-"),
    "lhs_type": ("R", "-", "Y"),
    "lhs_constructor": ("R", "-", "C"),
    "lhs_variable": ("R", "-", "V"),
    "lhs_parent_child": ("R", "-", "-", "-"),
    "lhs_parent_child_type": ("R", "-", "-", "Y"),
    "lhs_child_mask": ("R", "-", "-"),
    "lhs_binder_equal": ("R", "-", "-"),
    "rhs_active": ("R", "-"),
    "rhs_kind": ("R", "-", "-"),
    "rhs_type": ("R", "-", "Y"),
    "rhs_constructor": ("R", "-", "C"),
    "rhs_variable": ("R", "-", "V"),
    "rhs_parent_child": ("R", "-", "-", "-"),
    "rhs_parent_child_type": ("R", "-", "-", "Y"),
    "rhs_child_mask": ("R", "-", "-"),
    "rhs_binder_equal": ("R", "-", "-"),
    "rhs_to_lhs_binder_equal": ("R", "-", "-"),
}


_OUTPUT_AXES: dict[str, tuple[str, ...]] = {
    "no_redex_logits": ("-",),
    "halt_logits": ("-",),
    "rule_logits": ("R",),
    "rule_mask": ("R",),
    "path_logits": ("-", "-"),
    "path_mask": ("-", "-"),
    "binding_logits": ("R", "V", "N1"),
    "binding_mask": ("R", "V", "N1"),
    "node_operation_logits": ("N", "-"),
    "node_operation_mask": ("N", "-"),
    "root_pointer_logits": ("N1",),
    "root_pointer_mask": ("N1",),
    "node_kind_logits": ("N", "-"),
    "node_kind_mask": ("N", "-"),
    "node_type_pointer_logits": ("N", "Y"),
    "node_type_pointer_mask": ("N", "Y"),
    "node_constructor_pointer_logits": ("N", "C"),
    "node_constructor_pointer_mask": ("N", "C"),
    "node_variable_pointer_logits": ("N", "V"),
    "node_variable_pointer_mask": ("N", "V"),
    "child_pointer_logits": ("N", "-", "N1"),
    "child_pointer_mask": ("N", "-", "N1"),
    "child_presence_logits": ("N", "-", "-"),
    "child_presence_mask": ("N", "-", "-"),
}


def _axis_widths(packets: packet_tensors.NeuralTcrrPacketTensors) -> dict[str, int]:
    storage = packets.storage_active.shape[1]
    return {
        "C": packets.constructor_active.shape[1],
        "Y": packets.type_active.shape[1],
        "R": packets.rule_active.shape[1],
        "V": packets.variable_active.shape[1],
        "N": storage,
        "N1": storage + 1,
    }


def _axis_permutations(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    axis: str,
) -> dict[str, torch.Tensor]:
    widths = _axis_widths(packets)
    permutations = {
        name: torch.arange(width, dtype=torch.long) for name, width in widths.items()
    }
    base_width = widths[axis]
    permutations[axis] = torch.arange(
        base_width - 1,
        -1,
        -1,
        dtype=torch.long,
    )
    if axis == "N":
        permutations["N1"] = torch.cat(
            (
                permutations["N"],
                torch.tensor([base_width], dtype=torch.long),
            )
        )
    return permutations


def _permute_tensor(
    value: torch.Tensor,
    axes: tuple[str, ...],
    permutations: dict[str, torch.Tensor],
) -> torch.Tensor:
    output = value
    for offset, axis in enumerate(axes, start=1):
        if axis != "-":
            output = output.index_select(
                offset,
                permutations[axis].to(value.device),
            )
    return output


def _permute_packets(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    axis: str,
) -> tuple[
    packet_tensors.NeuralTcrrPacketTensors,
    dict[str, torch.Tensor],
]:
    permutations = _axis_permutations(packets, axis)
    values = {
        field.name: _permute_tensor(
            getattr(packets, field.name),
            _PACKET_AXES[field.name],
            permutations,
        )
        for field in dataclasses.fields(packet_tensors.NeuralTcrrPacketTensors)
    }
    return packet_tensors.NeuralTcrrPacketTensors(**values), permutations


def _output_fields(value: motor.NeuralTcrrMotorOutput) -> dict[str, torch.Tensor]:
    return {
        "no_redex_logits": value.no_redex_logits,
        "halt_logits": value.halt_logits,
        "rule_logits": value.rule_logits,
        "rule_mask": value.rule_mask,
        "path_logits": value.path_logits,
        "path_mask": value.path_mask,
        "binding_logits": value.binding_logits,
        "binding_mask": value.binding_mask,
        **{
            field.name: getattr(value.graph_delta, field.name)
            for field in dataclasses.fields(motor.NeuralTcrrGraphDelta)
        },
    }


def _assert_output_equivariant(
    original: motor.NeuralTcrrMotorOutput,
    transformed: motor.NeuralTcrrMotorOutput,
    permutations: dict[str, torch.Tensor],
) -> None:
    original_fields = _output_fields(original)
    transformed_fields = _output_fields(transformed)
    assert original_fields.keys() == transformed_fields.keys() == _OUTPUT_AXES.keys()
    for name, original_value in original_fields.items():
        expected = _permute_tensor(
            original_value,
            _OUTPUT_AXES[name],
            permutations,
        )
        actual = transformed_fields[name]
        if expected.dtype is torch.bool:
            assert torch.equal(expected, actual), name
        else:
            torch.testing.assert_close(
                expected,
                actual,
                rtol=2e-5,
                atol=2e-6,
                msg=name,
            )


def test_forward_boundary_and_parameter_budget_are_explicit() -> None:
    source = inspect.getsource(motor).lower()
    for forbidden in (
        "expectedtransition",
        "tensorize_neural_tcrr_training",
        "successor",
        "oracle",
        "receipt",
        "identifier",
        "packet_sha256",
        "hash(",
    ):
        assert forbidden not in source
    signature = inspect.signature(motor.NeuralTcrrMotor.forward)
    assert tuple(signature.parameters) == ("self", "packets")
    assert "NeuralTcrrPacketTensors" in str(signature.parameters["packets"].annotation)

    value = motor.NeuralTcrrMotor()
    count = value.parameter_count()
    assert count.total == 1_830_671
    assert count.trainable == count.total
    assert count.under_cap
    assert count.total < 16_000_000


def test_forward_rejects_the_wrapper_and_accepts_batched_packet_tensors() -> None:
    wrapped = packet_tensors.tensorize_neural_tcrr_packets(_slice().packets[:3])
    value = _small_motor()
    with pytest.raises(motor.NeuralTcrrMotorError):
        value(wrapped)  # type: ignore[arg-type]

    with torch.inference_mode():
        output = value(wrapped.tensors)
    batch_size = 3
    assert output.no_redex_logits.shape == (batch_size, 2)
    assert output.halt_logits.shape == (batch_size, 2)
    assert output.rule_logits.shape == (batch_size, packet_tensors.R)
    assert output.path_logits.shape == (
        batch_size,
        packet_tensors.D + 1,
        packet_tensors.A + 1,
    )
    assert output.binding_logits.shape == (
        batch_size,
        packet_tensors.R,
        packet_tensors.V,
        packet_tensors.N + 1,
    )
    assert output.graph_delta.child_pointer_logits.shape == (
        batch_size,
        packet_tensors.N,
        packet_tensors.A,
        packet_tensors.N + 1,
    )


@pytest.mark.parametrize("axis", ("C", "Y", "R", "V", "N"))
def test_all_anonymous_entity_reindexes_are_equivariant(axis: str) -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(_slice().packets[:2]).tensors
    transformed_packets, permutations = _permute_packets(packets, axis)
    value = _small_motor()
    with torch.inference_mode():
        original = value(packets)
        transformed = value(transformed_packets)
    _assert_output_equivariant(original, transformed, permutations)


def test_shared_occurrence_requires_distinct_root_relative_path_channels() -> None:
    active_slice = _slice()
    twin = next(item for item in active_slice.twins if item.kind == "shared_occurrence")
    record = next(
        item
        for item in active_slice.expected_records
        if item.packet_sha256 == twin.left_packet_sha256
    )
    left = record.transitions[twin.left_transition_index]
    right = record.transitions[twin.right_transition_index]
    assert left.target_storage_id == right.target_storage_id
    assert left.occurrence_path == (0,)
    assert right.occurrence_path == (1,)

    packet = _packet_map()[twin.left_packet_sha256]
    tensors = packet_tensors.tensorize_neural_tcrr_packets((packet,)).tensors
    value = _small_motor()
    with torch.inference_mode():
        output = value(tensors)
    assert output.path_mask[0, 0, left.occurrence_path[0]]
    assert output.path_mask[0, 0, right.occurrence_path[0]]
    assert not torch.isclose(
        output.path_logits[0, 0, left.occurrence_path[0]],
        output.path_logits[0, 0, right.occurrence_path[0]],
    )


def test_no_redex_packets_have_terminal_heads_and_legal_delta_masks() -> None:
    active_slice = _slice()
    negative_digests = {
        record.packet_sha256
        for record in active_slice.expected_records
        if not record.transitions
    }
    assert len(negative_digests) == 4
    packets = tuple(
        packet for digest, packet in _packet_map().items() if digest in negative_digests
    )
    tensors = packet_tensors.tensorize_neural_tcrr_packets(packets).tensors
    value = _small_motor()
    with torch.inference_mode():
        output = value(tensors)

    assert torch.isfinite(output.no_redex_logits).all()
    assert torch.isfinite(output.halt_logits).all()
    assert output.path_mask[:, :, packet_tensors.A].all()
    assert not output.path_mask[:, -1, : packet_tensors.A].any()

    capacity = tensors.graph_capacity
    delta = output.graph_delta
    assert delta.node_operation_mask[:, :, motor.NODE_KEEP].all()
    assert torch.equal(
        delta.node_operation_mask[:, :, motor.NODE_WRITE],
        capacity,
    )
    assert torch.equal(
        delta.node_operation_mask[:, :, motor.NODE_CLEAR],
        capacity,
    )
    assert torch.equal(delta.root_pointer_mask[:, :-1], capacity)
    assert delta.root_pointer_mask[:, -1].all()
    assert torch.equal(
        delta.child_pointer_mask[:, :, :, :-1],
        (capacity[:, :, None, None] & capacity[:, None, None, :]).expand(
            -1, -1, packet_tensors.A, -1
        ),
    )
    assert delta.child_pointer_mask[:, :, :, -1].all()


def test_masked_transaction_coordinates_never_receive_finite_choice_scores() -> None:
    tensors = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    value = _small_motor()
    with torch.inference_mode():
        output = value(tensors)
    fields_and_masks = (
        (
            output.graph_delta.node_operation_logits,
            output.graph_delta.node_operation_mask,
        ),
        (
            output.graph_delta.root_pointer_logits,
            output.graph_delta.root_pointer_mask,
        ),
        (
            output.graph_delta.node_type_pointer_logits,
            output.graph_delta.node_type_pointer_mask,
        ),
        (
            output.graph_delta.node_constructor_pointer_logits,
            output.graph_delta.node_constructor_pointer_mask,
        ),
        (
            output.graph_delta.node_variable_pointer_logits,
            output.graph_delta.node_variable_pointer_mask,
        ),
        (
            output.graph_delta.child_pointer_logits,
            output.graph_delta.child_pointer_mask,
        ),
    )
    for logits, mask in fields_and_masks:
        assert torch.all(logits[~mask] == motor.MASKED_LOGIT)
        assert torch.isfinite(logits[mask]).all()


def test_every_explicit_head_has_a_finite_gradient_path() -> None:
    tensors = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    value = _small_motor()
    output = value(tensors)
    delta = output.graph_delta
    loss = (
        output.no_redex_logits.sum()
        + output.halt_logits.sum()
        + output.rule_logits[output.rule_mask].sum()
        + output.path_logits[output.path_mask].sum()
        + output.binding_logits[output.binding_mask].sum()
        + delta.node_operation_logits[delta.node_operation_mask].sum()
        + delta.root_pointer_logits[delta.root_pointer_mask].sum()
        + delta.node_kind_logits[delta.node_kind_mask].sum()
        + delta.node_type_pointer_logits[delta.node_type_pointer_mask].sum()
        + delta.node_constructor_pointer_logits[
            delta.node_constructor_pointer_mask
        ].sum()
        + delta.node_variable_pointer_logits[delta.node_variable_pointer_mask].sum()
        + delta.child_pointer_logits[delta.child_pointer_mask].sum()
        + delta.child_presence_logits[delta.child_presence_mask].sum()
    )
    loss.backward()
    for name in (
        "no_redex_head.layers.2.weight",
        "halt_head.layers.2.weight",
        "rule_score.weight",
        "path_argument.layers.2.weight",
        "binding_query.layers.2.weight",
        "node_operation.weight",
        "root_query.weight",
        "node_kind.weight",
        "node_type_query.weight",
        "node_constructor_query.weight",
        "node_variable_query.weight",
        "child_query.layers.2.weight",
        "child_presence.weight",
    ):
        gradient = dict(value.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name
