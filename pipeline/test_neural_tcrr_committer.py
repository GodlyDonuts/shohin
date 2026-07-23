from __future__ import annotations

import dataclasses
import inspect
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

import neural_tcrr_board as board
import neural_tcrr_committer as committer
import neural_tcrr_motor as motor
import tensorize_neural_tcrr_packets as packet_tensors


@lru_cache(maxsize=1)
def _slice() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _packet_map() -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(packet): packet for packet in _slice().packets}


def _record_packet(
    record_index: int,
) -> tuple[board.SourceDeletedPacket, board.ExpectedTransitionRecord]:
    record = _slice().expected_records[record_index]
    return _packet_map()[record.packet_sha256], record


_GRAPH_TARGET_FIELDS = {
    "graph_active": "active",
    "graph_root": "root",
    "graph_kind": "kind",
    "graph_type": "type",
    "graph_constructor": "constructor",
    "graph_variable": "variable",
    "graph_children": "children",
    "graph_child_type": "child_type",
    "graph_child_mask": "child_mask",
    "graph_capacity": "capacity",
}


def _graph_values(
    value: committer.NeuralTcrrGraphTensors,
) -> dict[str, torch.Tensor]:
    return {
        field.name: getattr(value, field.name)
        for field in dataclasses.fields(committer.NeuralTcrrGraphTensors)
    }


def _assert_graph_equals_target(
    value: committer.NeuralTcrrGraphTensors,
    target: dict[str, torch.Tensor],
) -> None:
    for output_name, target_name in _GRAPH_TARGET_FIELDS.items():
        torch.testing.assert_close(
            getattr(value, output_name)[0],
            target[target_name],
            msg=output_name,
        )


def _blank_transaction(
    packets: packet_tensors.NeuralTcrrPacketTensors,
) -> committer.NeuralTcrrGraphTransaction:
    batch_size = packets.graph_active.shape[0]
    device = packets.graph_active.device
    operation = torch.zeros(
        (batch_size, packet_tensors.N, 3),
        dtype=torch.bool,
        device=device,
    )
    operation[:, :, committer.NODE_KEEP] = True
    return committer.NeuralTcrrGraphTransaction(
        node_operation=operation,
        root_pointer=packets.graph_root.clone(),
        node_kind=torch.zeros(
            (batch_size, packet_tensors.N, packet_tensors.GRAPH_KIND_COUNT),
            dtype=torch.bool,
            device=device,
        ),
        node_type_pointer=torch.zeros(
            (batch_size, packet_tensors.N, packet_tensors.Y),
            dtype=torch.bool,
            device=device,
        ),
        node_constructor_pointer=torch.zeros(
            (batch_size, packet_tensors.N, packet_tensors.C),
            dtype=torch.bool,
            device=device,
        ),
        node_variable_pointer=torch.zeros(
            (batch_size, packet_tensors.N, packet_tensors.V),
            dtype=torch.bool,
            device=device,
        ),
        child_pointer=torch.zeros(
            (
                batch_size,
                packet_tensors.N,
                packet_tensors.A,
                packet_tensors.N + 1,
            ),
            dtype=torch.bool,
            device=device,
        ),
        child_presence=torch.zeros(
            (batch_size, packet_tensors.N, packet_tensors.A, 2),
            dtype=torch.bool,
            device=device,
        ),
    )


def _write_payload(
    transaction: committer.NeuralTcrrGraphTransaction,
    *,
    batch_index: int,
    storage: int,
    active: torch.Tensor,
    kind: torch.Tensor,
    type_pointer: torch.Tensor,
    constructor_pointer: torch.Tensor,
    variable_pointer: torch.Tensor,
    children: torch.Tensor,
    child_mask: torch.Tensor,
) -> None:
    transaction.node_operation[batch_index, storage].zero_()
    transaction.node_operation[
        batch_index,
        storage,
        committer.NODE_WRITE,
    ] = True
    assert bool(active[storage])
    transaction.node_kind[batch_index, storage] = kind[storage]
    transaction.node_type_pointer[batch_index, storage] = type_pointer[storage]
    transaction.node_constructor_pointer[batch_index, storage] = constructor_pointer[
        storage
    ]
    transaction.node_variable_pointer[batch_index, storage] = variable_pointer[storage]
    transaction.child_pointer[batch_index, storage] = children[storage]
    transaction.child_presence[
        batch_index, storage, :, committer.CHILD_ABSENT
    ] = ~child_mask[storage]
    transaction.child_presence[batch_index, storage, :, committer.CHILD_PRESENT] = (
        child_mask[storage]
    )


def _transaction_for_target(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    target: dict[str, torch.Tensor],
) -> committer.NeuralTcrrGraphTransaction:
    assert packets.graph_active.shape[0] == 1
    value = _blank_transaction(packets)
    value.root_pointer[0] = target["root"]
    source_by_target = {
        "active": packets.graph_active[0],
        "kind": packets.graph_kind[0],
        "type": packets.graph_type[0],
        "constructor": packets.graph_constructor[0],
        "variable": packets.graph_variable[0],
        "children": packets.graph_children[0],
        "child_type": packets.graph_child_type[0],
        "child_mask": packets.graph_child_mask[0],
    }
    for storage in range(packet_tensors.N):
        exact = all(
            torch.equal(source_by_target[name][storage], target[name][storage])
            for name in source_by_target
        )
        if exact:
            continue
        if not bool(target["active"][storage]):
            value.node_operation[0, storage].zero_()
            value.node_operation[0, storage, committer.NODE_CLEAR] = True
            continue
        _write_payload(
            value,
            batch_index=0,
            storage=storage,
            active=target["active"],
            kind=target["kind"],
            type_pointer=target["type"],
            constructor_pointer=target["constructor"],
            variable_pointer=target["variable"],
            children=target["children"],
            child_mask=target["child_mask"],
        )
    return value


def _all_write_identity(
    packets: packet_tensors.NeuralTcrrPacketTensors,
) -> committer.NeuralTcrrGraphTransaction:
    value = _blank_transaction(packets)
    for batch_index in range(packets.graph_active.shape[0]):
        for storage in range(packet_tensors.N):
            if not bool(packets.graph_active[batch_index, storage]):
                continue
            _write_payload(
                value,
                batch_index=batch_index,
                storage=storage,
                active=packets.graph_active[batch_index],
                kind=packets.graph_kind[batch_index],
                type_pointer=packets.graph_type[batch_index],
                constructor_pointer=packets.graph_constructor[batch_index],
                variable_pointer=packets.graph_variable[batch_index],
                children=packets.graph_children[batch_index],
                child_mask=packets.graph_child_mask[batch_index],
            )
    return value


def _target_for_transition(
    packet: board.SourceDeletedPacket,
    transition: board.ExpectedTransition,
) -> tuple[
    packet_tensors.TensorizedNeuralTcrrPackets,
    dict[str, torch.Tensor],
]:
    wrapped = packet_tensors.tensorize_neural_tcrr_packets((packet,))
    target = packet_tensors.tensorize_graph_record(
        transition.successor,
        wrapped.receipts[0],
    )
    return wrapped, target


def _clone_transaction(
    value: committer.NeuralTcrrGraphTransaction,
) -> committer.NeuralTcrrGraphTransaction:
    return committer.NeuralTcrrGraphTransaction(
        **{
            field.name: getattr(value, field.name).clone()
            for field in dataclasses.fields(committer.NeuralTcrrGraphTransaction)
        }
    )


def _packet_snapshot(
    packets: packet_tensors.NeuralTcrrPacketTensors,
) -> dict[str, torch.Tensor]:
    return {
        field.name: getattr(packets, field.name).clone()
        for field in dataclasses.fields(packet_tensors.NeuralTcrrPacketTensors)
    }


def _assert_packet_unchanged(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    snapshot: dict[str, torch.Tensor],
) -> None:
    for name, expected in snapshot.items():
        assert torch.equal(getattr(packets, name), expected), name


def _assert_transaction_unchanged(
    value: committer.NeuralTcrrGraphTransaction,
    snapshot: committer.NeuralTcrrGraphTransaction,
) -> None:
    for field in dataclasses.fields(committer.NeuralTcrrGraphTransaction):
        assert torch.equal(
            getattr(value, field.name),
            getattr(snapshot, field.name),
        ), field.name


def test_source_has_no_symbolic_or_offline_dependency() -> None:
    source = inspect.getsource(committer).lower()
    for forbidden in (
        "neural_tcrr_board",
        "neural_tcrr_motor",
        "tensorize_neural_tcrr_training",
        "expectedtransition",
        "rulerecord",
        "ruletermrecord",
        "receipt",
        "identifier",
        "packet_sha256",
        "oracle",
        "successor",
        "normal_form",
        "normalform",
        "schedule",
        "answer",
        "retry",
        "ranking",
        "repair",
        ".rule_",
        ".lhs_",
        ".rhs_",
    ):
        assert forbidden not in source
    signature = inspect.signature(committer.commit_neural_tcrr_graph)
    assert tuple(signature.parameters) == ("packets", "transaction")
    assert "NeuralTcrrPacketTensors" in str(signature.parameters["packets"].annotation)


def test_exact_mixed_install_matches_offline_target() -> None:
    packet, record = _record_packet(0)
    wrapped, target = _target_for_transition(packet, record.transitions[1])
    transaction = _transaction_for_target(wrapped.tensors, target)
    value = committer.commit_neural_tcrr_graph(wrapped.tensors, transaction)
    _assert_graph_equals_target(value, target)
    operations = transaction.node_operation[0].to(torch.int64).argmax(dim=-1)
    assert bool((operations == committer.NODE_CLEAR).any())
    assert bool((operations == committer.NODE_WRITE).any())
    assert bool((operations == committer.NODE_KEEP).any())


def test_keep_preserves_every_graph_byte() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[14],)
    ).tensors
    transaction = _blank_transaction(packets)
    value = committer.commit_neural_tcrr_graph(packets, transaction)
    for name, output in _graph_values(value).items():
        assert torch.equal(output, getattr(packets, name)), name


def test_deletion_releases_slots_without_changing_capacity() -> None:
    packet, record = _record_packet(0)
    wrapped, target = _target_for_transition(packet, record.transitions[0])
    transaction = _transaction_for_target(wrapped.tensors, target)
    value = committer.commit_neural_tcrr_graph(wrapped.tensors, transaction)
    assert not bool(value.graph_active.any())
    assert bool(value.graph_root[0, packet_tensors.N])
    assert torch.equal(value.graph_capacity, wrapped.tensors.graph_capacity)
    cleared = torch.nonzero(
        transaction.node_operation[0, :, committer.NODE_CLEAR],
        as_tuple=False,
    ).flatten()
    assert cleared.numel() == int(wrapped.tensors.graph_active[0].sum())
    for storage in cleared.tolist():
        assert bool(value.graph_capacity[0, storage])
        assert bool(value.graph_kind[0, storage, committer.KIND_EMPTY])
        assert value.graph_children[0, storage, :, packet_tensors.N].all()


def test_valid_writes_can_fill_every_declared_capacity_slot() -> None:
    packet, record = _record_packet(14)
    wrapped, target = _target_for_transition(packet, record.transitions[0])
    transaction = _transaction_for_target(wrapped.tensors, target)
    value = committer.commit_neural_tcrr_graph(wrapped.tensors, transaction)
    _assert_graph_equals_target(value, target)
    assert int(value.graph_active.sum()) == packet_tensors.N
    written_free_slots = (
        transaction.node_operation[0, :, committer.NODE_WRITE]
        & ~wrapped.tensors.graph_active[0]
    )
    assert int(written_free_slots.sum()) == 2


def test_out_of_capacity_write_fails_atomically() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    transaction = _blank_transaction(packets)
    storage = int(torch.nonzero(~packets.graph_capacity[0])[0].item())
    transaction.node_operation[0, storage].zero_()
    transaction.node_operation[0, storage, committer.NODE_WRITE] = True
    packet_before = _packet_snapshot(packets)
    transaction_before = _clone_transaction(transaction)
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "write_outside_capacity"
    _assert_packet_unchanged(packets, packet_before)
    _assert_transaction_unchanged(transaction, transaction_before)


def test_dangling_child_rejects_without_partial_mutation() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    root = int(torch.nonzero(packets.graph_root[0, :-1])[0].item())
    child = int(
        torch.nonzero(
            packets.graph_children[0, root, 0, :-1],
            as_tuple=False,
        )[0].item()
    )
    transaction = _blank_transaction(packets)
    transaction.node_operation[0, child].zero_()
    transaction.node_operation[0, child, committer.NODE_CLEAR] = True
    packet_before = _packet_snapshot(packets)
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "child_not_active"
    _assert_packet_unchanged(packets, packet_before)


def test_wrong_type_write_is_rejected() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    transaction = _all_write_identity(packets)
    root = int(torch.nonzero(packets.graph_root[0, :-1])[0].item())
    current_type = int(torch.nonzero(packets.graph_type[0, root])[0].item())
    wrong_type = next(
        index
        for index in torch.nonzero(packets.type_active[0]).flatten().tolist()
        if index != current_type
    )
    transaction.node_type_pointer[0, root].zero_()
    transaction.node_type_pointer[0, root, wrong_type] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "constructor_result_type_mismatch"


def test_self_cycle_is_rejected_after_typed_install() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[7],)
    ).tensors
    transaction = _all_write_identity(packets)
    root = int(torch.nonzero(packets.graph_root[0, :-1])[0].item())
    transaction.child_pointer[0, root, 0].zero_()
    transaction.child_pointer[0, root, 0, root] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "graph_cycle"


def test_active_but_unreachable_records_are_rejected() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[7],)
    ).tensors
    transaction = _blank_transaction(packets)
    root = int(torch.nonzero(packets.graph_root[0, :-1])[0].item())
    child = int(
        torch.nonzero(
            packets.graph_children[0, root, 0, :-1],
            as_tuple=False,
        )[0].item()
    )
    leaf = int(
        torch.nonzero(
            packets.graph_children[0, child, 0, :-1],
            as_tuple=False,
        )[0].item()
    )
    transaction.root_pointer[0].zero_()
    transaction.root_pointer[0, leaf] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "unreachable_active_node"


def test_malformed_roots_fail_closed() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    null_root = _blank_transaction(packets)
    null_root.root_pointer[0].zero_()
    null_root.root_pointer[0, packet_tensors.N] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, null_root)
    assert caught.value.reason_code == "null_root_with_active_graph"

    non_one_hot = _blank_transaction(packets)
    non_one_hot.root_pointer[0].zero_()
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, non_one_hot)
    assert caught.value.reason_code == "transaction_root_not_one_hot"


def test_duplicate_graph_variables_are_rejected() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    transaction = _all_write_identity(packets)
    root = int(torch.nonzero(packets.graph_root[0, :-1])[0].item())
    children = [
        int(
            torch.nonzero(
                packets.graph_children[0, root, argument, :-1],
                as_tuple=False,
            )[0].item()
        )
        for argument in range(2)
    ]
    variable = int(torch.nonzero(packets.variable_active[0])[0].item())
    for storage in children:
        transaction.node_kind[0, storage].zero_()
        transaction.node_kind[0, storage, committer.KIND_VARIABLE] = True
        transaction.node_constructor_pointer[0, storage].zero_()
        transaction.node_variable_pointer[0, storage].zero_()
        transaction.node_variable_pointer[0, storage, variable] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "duplicate_graph_variable"


def test_malformed_operation_one_hotness_is_rejected() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
    transaction = _blank_transaction(packets)
    transaction.node_operation[0, 0, committer.NODE_CLEAR] = True
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.commit_neural_tcrr_graph(packets, transaction)
    assert caught.value.reason_code == "operation_not_one_hot"


def _delta_from_transaction(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    value: committer.NeuralTcrrGraphTransaction,
) -> SimpleNamespace:
    masks = committer._structural_delta_masks(packets)  # noqa: SLF001

    def logits_and_mask(
        one_hot: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = one_hot.to(torch.float32) * 7.0
        return logits, mask

    operation_logits, operation_mask = logits_and_mask(
        value.node_operation,
        masks["operation"],
    )
    root_logits, root_mask = logits_and_mask(
        value.root_pointer,
        masks["root"],
    )
    kind_logits, kind_mask = logits_and_mask(
        value.node_kind,
        masks["kind"],
    )
    type_logits, type_mask = logits_and_mask(
        value.node_type_pointer,
        masks["type"],
    )
    constructor_logits, constructor_mask = logits_and_mask(
        value.node_constructor_pointer,
        masks["constructor"],
    )
    variable_logits, variable_mask = logits_and_mask(
        value.node_variable_pointer,
        masks["variable"],
    )
    child_logits, child_mask = logits_and_mask(
        value.child_pointer,
        masks["child"],
    )
    presence_logits, presence_mask = logits_and_mask(
        value.child_presence,
        masks["presence"],
    )
    return SimpleNamespace(
        node_operation_logits=operation_logits,
        node_operation_mask=operation_mask,
        root_pointer_logits=root_logits,
        root_pointer_mask=root_mask,
        node_kind_logits=kind_logits,
        node_kind_mask=kind_mask,
        node_type_pointer_logits=type_logits,
        node_type_pointer_mask=type_mask,
        node_constructor_pointer_logits=constructor_logits,
        node_constructor_pointer_mask=constructor_mask,
        node_variable_pointer_logits=variable_logits,
        node_variable_pointer_mask=variable_mask,
        child_pointer_logits=child_logits,
        child_pointer_mask=child_mask,
        child_presence_logits=presence_logits,
        child_presence_mask=presence_mask,
    )


def test_delta_decode_and_commit_are_deterministic() -> None:
    packet, record = _record_packet(0)
    wrapped, target = _target_for_transition(packet, record.transitions[1])
    transaction = _transaction_for_target(wrapped.tensors, target)
    delta = _delta_from_transaction(wrapped.tensors, transaction)
    first_transaction = committer.decode_neural_tcrr_graph_delta(
        wrapped.tensors,
        delta,
    )
    second_transaction = committer.decode_neural_tcrr_graph_delta(
        wrapped.tensors,
        delta,
    )
    _assert_transaction_unchanged(first_transaction, second_transaction)
    first = committer.commit_neural_tcrr_graph_delta(wrapped.tensors, delta)
    second = committer.commit_neural_tcrr_graph_delta(wrapped.tensors, delta)
    for name in _graph_values(first):
        assert torch.equal(
            getattr(first, name),
            getattr(second, name),
        ), name
    _assert_graph_equals_target(first, target)


def test_delta_cannot_redefine_its_structural_choice_masks() -> None:
    packet, record = _record_packet(0)
    wrapped, target = _target_for_transition(packet, record.transitions[0])
    transaction = _transaction_for_target(wrapped.tensors, target)
    delta = _delta_from_transaction(wrapped.tensors, transaction)
    delta.node_operation_mask = delta.node_operation_mask.clone()
    delta.node_operation_mask[0, 0, committer.NODE_KEEP] = False
    with pytest.raises(committer.NeuralTcrrCommitError) as caught:
        committer.decode_neural_tcrr_graph_delta(
            wrapped.tensors,
            delta,
        )
    assert caught.value.reason_code == "delta_mask_drift"


def test_real_motor_delta_crosses_the_duck_typed_boundary() -> None:
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[0],)
    ).tensors
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
    with torch.inference_mode():
        delta = value(packets).graph_delta
    first = committer.decode_neural_tcrr_graph_delta(packets, delta)
    second = committer.decode_neural_tcrr_graph_delta(packets, delta)
    _assert_transaction_unchanged(first, second)
    assert first.node_operation.sum(dim=-1).all()
    assert first.root_pointer.sum(dim=-1).eq(1).all()
    try:
        committed = committer.commit_neural_tcrr_graph(packets, first)
    except committer.NeuralTcrrCommitError as error:
        assert error.reason_code
    else:
        assert committed.graph_active.shape == packets.graph_active.shape


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

_TRANSACTION_AXES: dict[str, tuple[str, ...]] = {
    "node_operation": ("N", "-"),
    "root_pointer": ("N1",),
    "node_kind": ("N", "-"),
    "node_type_pointer": ("N", "Y"),
    "node_constructor_pointer": ("N", "C"),
    "node_variable_pointer": ("N", "V"),
    "child_pointer": ("N", "-", "N1"),
    "child_presence": ("N", "-", "-"),
}

_GRAPH_AXES: dict[str, tuple[str, ...]] = {
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
}


def _permutations(axis: str) -> dict[str, torch.Tensor]:
    widths = {
        "C": packet_tensors.C,
        "Y": packet_tensors.Y,
        "R": packet_tensors.R,
        "V": packet_tensors.V,
        "N": packet_tensors.N,
        "N1": packet_tensors.N + 1,
    }
    values = {
        name: torch.arange(width, dtype=torch.long) for name, width in widths.items()
    }
    values[axis] = torch.arange(widths[axis] - 1, -1, -1)
    if axis == "N":
        values["N1"] = torch.cat(
            (
                values["N"],
                torch.tensor([packet_tensors.N]),
            )
        )
    return values


def _permute_tensor(
    value: torch.Tensor,
    axes: tuple[str, ...],
    permutations: dict[str, torch.Tensor],
) -> torch.Tensor:
    output = value
    for dimension, axis in enumerate(axes, start=1):
        if axis != "-":
            output = output.index_select(
                dimension,
                permutations[axis].to(output.device),
            )
    return output


def _permute_packets(
    packets: packet_tensors.NeuralTcrrPacketTensors,
    permutations: dict[str, torch.Tensor],
) -> packet_tensors.NeuralTcrrPacketTensors:
    return packet_tensors.NeuralTcrrPacketTensors(
        **{
            field.name: _permute_tensor(
                getattr(packets, field.name),
                _PACKET_AXES[field.name],
                permutations,
            )
            for field in dataclasses.fields(packet_tensors.NeuralTcrrPacketTensors)
        }
    )


def _permute_transaction(
    value: committer.NeuralTcrrGraphTransaction,
    permutations: dict[str, torch.Tensor],
) -> committer.NeuralTcrrGraphTransaction:
    return committer.NeuralTcrrGraphTransaction(
        **{
            field.name: _permute_tensor(
                getattr(value, field.name),
                _TRANSACTION_AXES[field.name],
                permutations,
            )
            for field in dataclasses.fields(committer.NeuralTcrrGraphTransaction)
        }
    )


@pytest.mark.parametrize("axis", ("C", "Y", "V", "N"))
def test_transaction_commit_is_exactly_reindex_equivariant(axis: str) -> None:
    packet_index = 9 if axis == "V" else 7
    packets = packet_tensors.tensorize_neural_tcrr_packets(
        (_slice().packets[packet_index],)
    ).tensors
    transaction = _all_write_identity(packets)
    original = committer.commit_neural_tcrr_graph(packets, transaction)
    permutations = _permutations(axis)
    transformed_packets = _permute_packets(packets, permutations)
    transformed_transaction = _permute_transaction(transaction, permutations)
    transformed = committer.commit_neural_tcrr_graph(
        transformed_packets,
        transformed_transaction,
    )
    for name, axes in _GRAPH_AXES.items():
        expected = _permute_tensor(
            getattr(original, name),
            axes,
            permutations,
        )
        assert torch.equal(expected, getattr(transformed, name)), name
