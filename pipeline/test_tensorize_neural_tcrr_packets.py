from __future__ import annotations

import dataclasses
import inspect
from functools import lru_cache
from pathlib import Path

import torch

import neural_tcrr_board as board
import tensorize_neural_tcrr_packets as packet_tensors


@lru_cache(maxsize=1)
def _slice() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _packet_map() -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(packet): packet for packet in _slice().packets}


def _permutation(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    width: int,
    *,
    inverse_names: dict[str, str] | None = None,
) -> torch.Tensor:
    inverse_names = inverse_names or {}
    normalized_source = tuple(inverse_names.get(item, item) for item in source_ids)
    assert set(normalized_source) == set(target_ids)
    active = [normalized_source.index(item) for item in target_ids]
    return torch.tensor(
        [*active, *range(len(active), width)],
        dtype=torch.long,
    )


_AXES: dict[str, tuple[str, ...]] = {
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
    "lhs_active": (
        "R",
        "-",
    ),
    "lhs_kind": ("R", "-", "-"),
    "lhs_type": ("R", "-", "Y"),
    "lhs_constructor": ("R", "-", "C"),
    "lhs_variable": ("R", "-", "V"),
    "lhs_parent_child": ("R", "-", "-", "-"),
    "lhs_parent_child_type": ("R", "-", "-", "Y"),
    "lhs_child_mask": ("R", "-", "-"),
    "lhs_binder_equal": ("R", "-", "-"),
    "rhs_active": (
        "R",
        "-",
    ),
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


def _aligned_fields(
    value: packet_tensors.TensorizedNeuralTcrrPackets,
    receipt: packet_tensors.PacketAxisReceipt,
    target: packet_tensors.PacketAxisReceipt,
    *,
    inverse_names: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    permutations = {
        "C": _permutation(
            receipt.constructor_ids,
            target.constructor_ids,
            packet_tensors.C,
            inverse_names=inverse_names,
        ),
        "Y": _permutation(
            receipt.type_ids,
            target.type_ids,
            packet_tensors.Y,
            inverse_names=inverse_names,
        ),
        "R": _permutation(
            receipt.rule_ids,
            target.rule_ids,
            packet_tensors.R,
            inverse_names=inverse_names,
        ),
        "V": _permutation(
            receipt.variable_ids,
            target.variable_ids,
            packet_tensors.V,
            inverse_names=inverse_names,
        ),
        "N": _permutation(
            receipt.storage_ids,
            target.storage_ids,
            packet_tensors.N,
            inverse_names=inverse_names,
        ),
    }
    permutations["N1"] = torch.cat(
        (permutations["N"], torch.tensor([packet_tensors.N]))
    )
    output = {}
    for field, axes in _AXES.items():
        active = getattr(value.tensors, field)[0]
        for dimension, axis in enumerate(axes):
            if axis != "-":
                active = active.index_select(dimension, permutations[axis])
        output[field] = active
    return output


def _assert_exactly_aligned(
    left: packet_tensors.TensorizedNeuralTcrrPackets,
    right: packet_tensors.TensorizedNeuralTcrrPackets,
    *,
    inverse_names: dict[str, str] | None = None,
) -> None:
    left_fields = _aligned_fields(left, left.receipts[0], left.receipts[0])
    right_fields = _aligned_fields(
        right,
        right.receipts[0],
        left.receipts[0],
        inverse_names=inverse_names,
    )
    assert left_fields.keys() == right_fields.keys()
    for field in left_fields:
        assert torch.equal(left_fields[field], right_fields[field]), field


def test_packet_module_has_no_offline_answer_dependency() -> None:
    source = inspect.getsource(packet_tensors).lower()
    for forbidden in (
        "expectedtransition",
        "oracle",
        "successor",
        "label",
        "semantic_alias",
        "hash(",
    ):
        assert forbidden not in source
    signature = inspect.signature(packet_tensors.tensorize_neural_tcrr_packets)
    assert tuple(signature.parameters) == ("packets", "device")


def test_frozen_geometry_and_reference_axes() -> None:
    value = packet_tensors.tensorize_neural_tcrr_packets(_slice().packets)
    tensors = value.tensors
    batch = len(_slice().packets)
    assert tensors.graph_active.shape == (batch, 16)
    assert tensors.constructor_active.shape == (batch, 16)
    assert tensors.type_active.shape == (batch, 8)
    assert tensors.rule_active.shape == (batch, 8)
    assert tensors.variable_active.shape == (batch, 112)
    assert tensors.lhs_active.shape == (batch, 8, 12)
    assert tensors.lhs_parent_child.shape == (batch, 8, 12, 3, 12)
    assert tensors.graph_children.shape == (batch, 16, 3, 17)
    assert tensors.graph_root.shape == (batch, 17)
    for index, receipt in enumerate(value.receipts):
        assert int(tensors.constructor_active[index].sum()) == len(
            receipt.constructor_ids
        )
        assert int(tensors.type_active[index].sum()) == len(receipt.type_ids)
        assert int(tensors.rule_active[index].sum()) == len(receipt.rule_ids)
        assert int(tensors.variable_active[index].sum()) == len(receipt.variable_ids)
        assert int(tensors.storage_active[index].sum()) == len(receipt.storage_ids)
        assert int(tensors.graph_root[index].sum()) == 1
        assert torch.equal(
            tensors.constructor_equal[index].diagonal(),
            tensors.constructor_active[index],
        )
        assert torch.equal(
            tensors.type_equal[index].diagonal(),
            tensors.type_active[index],
        )
        assert torch.equal(
            tensors.rule_equal[index].diagonal(),
            tensors.rule_active[index],
        )
        assert torch.equal(
            tensors.variable_equal[index].diagonal(),
            tensors.variable_active[index],
        )
        assert torch.equal(
            tensors.storage_equal[index].diagonal(),
            tensors.storage_active[index],
        )


def test_packet_round_trip_is_digest_exact_for_every_board_packet() -> None:
    packets = _slice().packets
    value = packet_tensors.tensorize_neural_tcrr_packets(packets)
    for index, packet in enumerate(packets):
        reconstructed = packet_tensors.detensorize_neural_tcrr_packet(
            value,
            index,
        )
        assert reconstructed == packet
        assert board.packet_sha256(reconstructed) == board.packet_sha256(packet)


def test_arbitrary_record_order_is_exactly_reindex_equivariant() -> None:
    packet = _slice().packets[0]
    reordered = dataclasses.replace(
        packet,
        constructors=tuple(reversed(packet.constructors)),
        rules=tuple(reversed(packet.rules)),
        graph=dataclasses.replace(
            packet.graph,
            reservoir=tuple(reversed(packet.graph.reservoir)),
            nodes=tuple(reversed(packet.graph.nodes)),
        ),
    )
    board.validate_source_deleted_packet(reordered)
    left = packet_tensors.tensorize_neural_tcrr_packets((packet,))
    right = packet_tensors.tensorize_neural_tcrr_packets((reordered,))
    _assert_exactly_aligned(left, right)


def test_graph_record_order_alone_is_exactly_invariant() -> None:
    packet = _slice().packets[0]
    reordered = dataclasses.replace(
        packet,
        graph=dataclasses.replace(
            packet.graph,
            nodes=tuple(reversed(packet.graph.nodes)),
        ),
    )
    left = packet_tensors.tensorize_neural_tcrr_packets((packet,))
    right = packet_tensors.tensorize_neural_tcrr_packets((reordered,))
    for field in _AXES:
        assert torch.equal(
            getattr(left.tensors, field),
            getattr(right.tensors, field),
        ), field


def test_all_four_identifier_reindex_twins_are_exactly_equivariant() -> None:
    packets = _packet_map()
    twins = {twin.kind: twin for twin in _slice().twins if twin.namespace is not None}
    assert set(twins) == {
        "constructor_reindex",
        "type_reindex",
        "rule_reindex",
        "storage_reindex",
    }
    for twin in twins.values():
        left = packet_tensors.tensorize_neural_tcrr_packets(
            (packets[twin.left_packet_sha256],)
        )
        right = packet_tensors.tensorize_neural_tcrr_packets(
            (packets[twin.right_packet_sha256],)
        )
        inverse = {item.new: item.old for item in twin.remap}
        _assert_exactly_aligned(left, right, inverse_names=inverse)


def test_every_non_reindex_twin_changes_only_model_visible_structure() -> None:
    packets = _packet_map()
    twins = {twin.kind: twin for twin in _slice().twins}
    assert set(twins) == {
        "rhs_pointer",
        "shared_occurrence",
        "capacity",
        "constructor_reindex",
        "type_reindex",
        "rule_reindex",
        "storage_reindex",
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
    }
    for kind in (
        "rhs_pointer",
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
    ):
        twin = twins[kind]
        left = packet_tensors.tensorize_neural_tcrr_packets(
            (packets[twin.left_packet_sha256],)
        )
        right = packet_tensors.tensorize_neural_tcrr_packets(
            (packets[twin.right_packet_sha256],)
        )
        differences = [
            field
            for field in _AXES
            if not torch.equal(
                getattr(left.tensors, field),
                getattr(right.tensors, field),
            )
        ]
        assert differences, kind

    shared = twins["shared_occurrence"]
    assert shared.left_packet_sha256 == shared.right_packet_sha256
    capacity = twins["capacity"]
    left_capacity = packet_tensors.tensorize_neural_tcrr_packets(
        (packets[capacity.left_packet_sha256],)
    )
    right_capacity = packet_tensors.tensorize_neural_tcrr_packets(
        (packets[capacity.right_packet_sha256],)
    )
    assert int(left_capacity.tensors.graph_capacity.sum()) == 16
    assert int(right_capacity.tensors.graph_capacity.sum()) == 15


def test_packet_only_loader_output_crosses_the_same_boundary(tmp_path) -> None:
    value = _slice()
    receipt = board.export_packet_only_corpus(
        value,
        packet_root=tmp_path / "packets",
        train_label_root=tmp_path / "train-private",
        development_assessment_root=tmp_path / "assessment-private",
    )
    loaded = board.load_packet_only_partition(
        Path(receipt.packet_root),
        "train",
    )
    tensorized = packet_tensors.tensorize_neural_tcrr_packets(loaded)
    assert len(tensorized.receipts) == 16
