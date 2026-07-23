from __future__ import annotations

import dataclasses
import inspect
from functools import lru_cache

import pytest
import torch

import neural_tcrr_board as board
import tensorize_neural_tcrr_packets as packet_tensors
import tensorize_neural_tcrr_training as training_tensors


@lru_cache(maxsize=1)
def _slice() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _packet_map() -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(packet): packet for packet in _slice().packets}


def _record_map() -> dict[str, board.ExpectedTransitionRecord]:
    return {record.packet_sha256: record for record in _slice().expected_records}


def test_training_module_is_offline_only() -> None:
    source = inspect.getsource(training_tensors).lower()
    for forbidden in (
        "load_sealed_development_assessment",
        "sealed_development",
        "development_assessment",
        "pathlib",
        "read_text",
        "read_bytes",
        "open(",
        "json.load",
    ):
        assert forbidden not in source
    packet_source = inspect.getsource(packet_tensors)
    assert "ExpectedTransitionRecord" not in packet_source
    assert "NeuralTcrrTrainingTensors" not in packet_source


def test_training_shapes_and_explicit_stop_contract() -> None:
    value = training_tensors.tensorize_neural_tcrr_training(
        _slice().packets,
        _slice().expected_records,
    )
    tensors = value.tensors
    batch = len(_slice().packets)
    assert tensors.action_mask.shape == (batch, 128)
    assert tensors.rule_pointer.shape == (batch, 128, 8)
    assert tensors.path_tokens.shape == (batch, 128, 9)
    assert tensors.target_storage_pointer.shape == (batch, 128, 16)
    assert tensors.variable_binding.shape == (batch, 128, 112, 16)
    assert tensors.successor_children.shape == (batch, 128, 16, 3, 17)
    assert int(tensors.action_mask.sum()) == 24
    for batch_index, action_index in torch.nonzero(
        tensors.action_mask,
        as_tuple=False,
    ).tolist():
        visible = tensors.path_tokens[
            batch_index,
            action_index,
            tensors.path_token_mask[batch_index, action_index],
        ].tolist()
        assert visible[-1] == training_tensors.PATH_STOP
        assert visible.count(training_tensors.PATH_STOP) == 1
        assert all(0 <= token < 3 for token in visible[:-1])
        assert int(tensors.rule_pointer[batch_index, action_index].sum()) == 1
        assert int(tensors.target_storage_pointer[batch_index, action_index].sum()) == 1
        assert int(tensors.successor_root[batch_index, action_index].sum()) == 1


def test_exact_training_record_round_trip_for_all_packets() -> None:
    source = _slice()
    value = training_tensors.tensorize_neural_tcrr_training(
        source.packets,
        tuple(reversed(source.expected_records)),
    )
    record_by_digest = _record_map()
    for index, receipt in enumerate(value.training_receipts):
        reconstructed = training_tensors.detensorize_neural_tcrr_training_record(
            value, index
        )
        assert reconstructed == record_by_digest[receipt.packet_digest]


def test_packet_and_training_digest_mismatch_fails_closed() -> None:
    packets = _slice().packets[:2]
    records = list(_slice().expected_records[:2])
    records[0] = dataclasses.replace(records[0], packet_sha256="0" * 64)
    with pytest.raises(
        training_tensors.NeuralTcrrTrainingTensorError,
        match="digests do not match",
    ):
        training_tensors.tensorize_neural_tcrr_training(packets, records)
    with pytest.raises(
        training_tensors.NeuralTcrrTrainingTensorError,
        match="duplicated",
    ):
        training_tensors.tensorize_neural_tcrr_training(
            packets,
            (records[1], records[1]),
        )


def test_all_four_no_redex_packets_emit_empty_action_sets() -> None:
    records = _record_map()
    packets = [
        packet
        for packet in _slice().packets
        if not records[board.packet_sha256(packet)].transitions
    ]
    assert len(packets) == 4
    value = training_tensors.tensorize_neural_tcrr_training(
        packets,
        tuple(records[board.packet_sha256(packet)] for packet in packets),
    )
    assert not bool(value.tensors.action_mask.any())
    assert not bool(value.tensors.rule_pointer.any())
    assert not bool(value.tensors.target_storage_pointer.any())
    assert not bool(value.tensors.variable_binding.any())
    assert not bool(value.tensors.successor_active.any())


def test_shared_occurrence_actions_remain_distinguishable() -> None:
    twin = next(item for item in _slice().twins if item.kind == "shared_occurrence")
    packet = _packet_map()[twin.left_packet_sha256]
    record = _record_map()[twin.left_packet_sha256]
    value = training_tensors.tensorize_neural_tcrr_training(
        (packet,),
        (record,),
    )
    left_index = twin.left_transition_index
    right_index = twin.right_transition_index
    assert left_index is not None and right_index is not None
    tensors = value.tensors
    assert torch.equal(
        tensors.rule_pointer[0, left_index],
        tensors.rule_pointer[0, right_index],
    )
    assert not torch.equal(
        tensors.path_tokens[0, left_index],
        tensors.path_tokens[0, right_index],
    )
    assert torch.equal(
        tensors.target_storage_pointer[0, left_index],
        tensors.target_storage_pointer[0, right_index],
    )
    assert not torch.equal(
        tensors.successor_constructor[0, left_index],
        tensors.successor_constructor[0, right_index],
    )


def test_rhs_pointer_twin_keeps_inputs_and_graph_targets_distinct() -> None:
    twin = next(item for item in _slice().twins if item.kind == "rhs_pointer")
    packets = _packet_map()
    records = _record_map()
    pair = (
        packets[twin.left_packet_sha256],
        packets[twin.right_packet_sha256],
    )
    packet_value = packet_tensors.tensorize_neural_tcrr_packets(pair)
    assert not torch.equal(
        packet_value.tensors.rhs_variable[0],
        packet_value.tensors.rhs_variable[1],
    )
    training_value = training_tensors.tensorize_neural_tcrr_training(
        pair,
        (
            records[twin.left_packet_sha256],
            records[twin.right_packet_sha256],
        ),
    )
    assert not torch.equal(
        training_value.tensors.successor_constructor[0, 0],
        training_value.tensors.successor_constructor[1, 0],
    )


def test_every_twin_kind_crosses_packet_or_action_boundary() -> None:
    packets = _packet_map()
    records = _record_map()
    seen = set()
    for twin in _slice().twins:
        seen.add(twin.kind)
        left_record = records[twin.left_packet_sha256]
        right_record = records[twin.right_packet_sha256]
        if twin.kind == "shared_occurrence":
            assert len(left_record.transitions) == 2
            continue
        if twin.namespace is not None:
            assert len(left_record.transitions) == len(right_record.transitions)
            continue
        left_packet = packets[twin.left_packet_sha256]
        right_packet = packets[twin.right_packet_sha256]
        packet_value = packet_tensors.tensorize_neural_tcrr_packets(
            (left_packet, right_packet)
        )
        action_value = training_tensors.tensorize_neural_tcrr_training(
            (left_packet, right_packet),
            (left_record, right_record),
        )
        packet_changed = any(
            not torch.equal(field[0], field[1])
            for field in dataclasses.astuple(packet_value.tensors)
        )
        action_changed = not torch.equal(
            action_value.tensors.action_mask[0],
            action_value.tensors.action_mask[1],
        ) or not torch.equal(
            action_value.tensors.successor_children[0],
            action_value.tensors.successor_children[1],
        )
        assert packet_changed or action_changed, twin.kind
    assert seen == {
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


def test_offline_graph_targets_are_not_part_of_packet_tensors() -> None:
    packet_fields = {
        field.name
        for field in dataclasses.fields(packet_tensors.NeuralTcrrPacketTensors)
    }
    training_fields = {
        field.name
        for field in dataclasses.fields(training_tensors.NeuralTcrrTrainingTensors)
    }
    assert not any(name.startswith("successor_") for name in packet_fields)
    assert all(
        name in training_fields
        for name in (
            "action_mask",
            "rule_pointer",
            "successor_active",
            "successor_children",
        )
    )
