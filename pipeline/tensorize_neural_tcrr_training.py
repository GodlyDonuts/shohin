"""Offline set-valued action tensors for neural TCRR training.

This module joins packet digests to training records in memory. It performs no
filesystem reads and is intentionally separate from the model-visible packet
boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from neural_tcrr_board import (
    BindingRecord,
    ExpectedTransition,
    ExpectedTransitionRecord,
    SourceDeletedPacket,
    packet_sha256,
    validate_source_deleted_packet,
)
from tensorize_neural_tcrr_packets import (
    A,
    C,
    D,
    GRAPH_KIND_COUNT,
    N,
    R,
    V,
    Y,
    PacketAxisReceipt,
    decode_graph_record,
    tensorize_graph_record,
    tensorize_neural_tcrr_packets,
)


MAX_ACTIONS = R * N
PATH_STOP = A
PATH_PAD = A + 1
PATH_VOCAB_SIZE = A + 2
PATH_WIDTH = D + 1


class NeuralTcrrTrainingTensorError(ValueError):
    """Raised when offline training records fail the frozen tensor contract."""


@dataclass(frozen=True)
class NeuralTcrrTrainingTensors:
    """Padded set-valued action and exact graph targets."""

    action_mask: Tensor
    rule_pointer: Tensor
    path_tokens: Tensor
    path_token_mask: Tensor
    target_storage_pointer: Tensor
    variable_binding: Tensor
    variable_binding_mask: Tensor

    successor_active: Tensor
    successor_root: Tensor
    successor_kind: Tensor
    successor_type: Tensor
    successor_constructor: Tensor
    successor_variable: Tensor
    successor_children: Tensor
    successor_child_type: Tensor
    successor_child_mask: Tensor
    successor_capacity: Tensor


@dataclass(frozen=True)
class TrainingTensorReceipt:
    """Offline reconstruction receipt for one padded action set."""

    packet_digest: str
    graph_node_record_orders: tuple[tuple[str, ...], ...]
    binding_record_orders: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TensorizedNeuralTcrrTraining:
    """Offline tensors plus non-model reconstruction receipts."""

    tensors: NeuralTcrrTrainingTensors
    packet_receipts: tuple[PacketAxisReceipt, ...]
    training_receipts: tuple[TrainingTensorReceipt, ...]


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _new_training_tensors(
    batch_size: int,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    boolean = torch.bool
    return {
        "action_mask": _zeros(
            (batch_size, MAX_ACTIONS),
            dtype=boolean,
            device=device,
        ),
        "rule_pointer": _zeros(
            (batch_size, MAX_ACTIONS, R),
            dtype=boolean,
            device=device,
        ),
        "path_tokens": torch.full(
            (batch_size, MAX_ACTIONS, PATH_WIDTH),
            PATH_PAD,
            dtype=torch.long,
            device=device,
        ),
        "path_token_mask": _zeros(
            (batch_size, MAX_ACTIONS, PATH_WIDTH),
            dtype=boolean,
            device=device,
        ),
        "target_storage_pointer": _zeros(
            (batch_size, MAX_ACTIONS, N),
            dtype=boolean,
            device=device,
        ),
        "variable_binding": _zeros(
            (batch_size, MAX_ACTIONS, V, N),
            dtype=boolean,
            device=device,
        ),
        "variable_binding_mask": _zeros(
            (batch_size, MAX_ACTIONS, V),
            dtype=boolean,
            device=device,
        ),
        "successor_active": _zeros(
            (batch_size, MAX_ACTIONS, N),
            dtype=boolean,
            device=device,
        ),
        "successor_root": _zeros(
            (batch_size, MAX_ACTIONS, N + 1),
            dtype=boolean,
            device=device,
        ),
        "successor_kind": _zeros(
            (batch_size, MAX_ACTIONS, N, GRAPH_KIND_COUNT),
            dtype=boolean,
            device=device,
        ),
        "successor_type": _zeros(
            (batch_size, MAX_ACTIONS, N, Y),
            dtype=boolean,
            device=device,
        ),
        "successor_constructor": _zeros(
            (batch_size, MAX_ACTIONS, N, C),
            dtype=boolean,
            device=device,
        ),
        "successor_variable": _zeros(
            (batch_size, MAX_ACTIONS, N, V),
            dtype=boolean,
            device=device,
        ),
        "successor_children": _zeros(
            (batch_size, MAX_ACTIONS, N, A, N + 1),
            dtype=boolean,
            device=device,
        ),
        "successor_child_type": _zeros(
            (batch_size, MAX_ACTIONS, N, A, Y),
            dtype=boolean,
            device=device,
        ),
        "successor_child_mask": _zeros(
            (batch_size, MAX_ACTIONS, N, A),
            dtype=boolean,
            device=device,
        ),
        "successor_capacity": _zeros(
            (batch_size, MAX_ACTIONS, N),
            dtype=boolean,
            device=device,
        ),
    }


def _record_map(
    records: Sequence[ExpectedTransitionRecord],
) -> dict[str, ExpectedTransitionRecord]:
    output: dict[str, ExpectedTransitionRecord] = {}
    for record in records:
        if not isinstance(record, ExpectedTransitionRecord):
            raise NeuralTcrrTrainingTensorError(
                "training tensorizer accepts ExpectedTransitionRecord objects only"
            )
        if record.packet_sha256 in output:
            raise NeuralTcrrTrainingTensorError("training packet digest is duplicated")
        output[record.packet_sha256] = record
    return output


def _validate_transition(
    transition: ExpectedTransition,
    packet: SourceDeletedPacket,
    receipt: PacketAxisReceipt,
) -> None:
    if transition.rule_id not in receipt.rule_ids:
        raise NeuralTcrrTrainingTensorError("action names an unknown rule")
    if transition.target_storage_id not in receipt.storage_ids:
        raise NeuralTcrrTrainingTensorError("action names an unknown target storage")
    if len(transition.occurrence_path) > D:
        raise NeuralTcrrTrainingTensorError("action path exceeds frozen geometry")
    if any(not 0 <= token < A for token in transition.occurrence_path):
        raise NeuralTcrrTrainingTensorError("action path contains an invalid argument")
    seen_variables = set()
    for binding in transition.bindings:
        if binding.variable_id in seen_variables:
            raise NeuralTcrrTrainingTensorError("action binding variable is duplicated")
        seen_variables.add(binding.variable_id)
        if binding.variable_id not in receipt.variable_ids:
            raise NeuralTcrrTrainingTensorError("action binds an unknown variable")
        if binding.storage_id not in receipt.storage_ids:
            raise NeuralTcrrTrainingTensorError("action binds unknown storage")
    if transition.successor.reservoir != packet.graph.reservoir:
        raise NeuralTcrrTrainingTensorError(
            "action graph changes the fixed reservoir coordinates"
        )
    candidate = SourceDeletedPacket(
        constructors=packet.constructors,
        rules=packet.rules,
        graph=transition.successor,
    )
    validate_source_deleted_packet(candidate)


def tensorize_neural_tcrr_training(
    packets: Sequence[SourceDeletedPacket],
    records: Sequence[ExpectedTransitionRecord],
    *,
    device: torch.device | str = "cpu",
) -> TensorizedNeuralTcrrTraining:
    """Join packets to offline records and emit padded action sets."""

    packet_tensorization = tensorize_neural_tcrr_packets(packets, device=device)
    record_by_digest = _record_map(records)
    packet_digests = [
        receipt.packet_digest for receipt in packet_tensorization.receipts
    ]
    if len(packet_digests) != len(set(packet_digests)):
        raise NeuralTcrrTrainingTensorError("packet digest is duplicated")
    if set(packet_digests) != set(record_by_digest):
        raise NeuralTcrrTrainingTensorError("packet and training digests do not match")

    resolved_device = torch.device(device)
    output = _new_training_tensors(len(packets), device=resolved_device)
    training_receipts = []
    for batch_index, (packet, receipt) in enumerate(
        zip(packets, packet_tensorization.receipts, strict=True)
    ):
        if packet_sha256(packet) != receipt.packet_digest:
            raise NeuralTcrrTrainingTensorError("packet digest receipt changed")
        record = record_by_digest[receipt.packet_digest]
        if len(record.transitions) > MAX_ACTIONS:
            raise NeuralTcrrTrainingTensorError(
                f"action count exceeds frozen geometry {MAX_ACTIONS}"
            )
        rule_index = {
            identifier: index for index, identifier in enumerate(receipt.rule_ids)
        }
        variable_index = {
            identifier: index for index, identifier in enumerate(receipt.variable_ids)
        }
        storage_index = {
            identifier: index for index, identifier in enumerate(receipt.storage_ids)
        }
        graph_orders = []
        binding_orders = []
        for action_index, transition in enumerate(record.transitions):
            _validate_transition(transition, packet, receipt)
            output["action_mask"][batch_index, action_index] = True
            output["rule_pointer"][
                batch_index,
                action_index,
                rule_index[transition.rule_id],
            ] = True
            path_length = len(transition.occurrence_path)
            if path_length:
                output["path_tokens"][
                    batch_index,
                    action_index,
                    :path_length,
                ] = torch.tensor(
                    transition.occurrence_path,
                    dtype=torch.long,
                    device=resolved_device,
                )
            output["path_tokens"][
                batch_index,
                action_index,
                path_length,
            ] = PATH_STOP
            output["path_token_mask"][
                batch_index,
                action_index,
                : path_length + 1,
            ] = True
            output["target_storage_pointer"][
                batch_index,
                action_index,
                storage_index[transition.target_storage_id],
            ] = True
            for binding in transition.bindings:
                variable_position = variable_index[binding.variable_id]
                storage_position = storage_index[binding.storage_id]
                output["variable_binding"][
                    batch_index,
                    action_index,
                    variable_position,
                    storage_position,
                ] = True
                output["variable_binding_mask"][
                    batch_index,
                    action_index,
                    variable_position,
                ] = True
            graph_tensors = tensorize_graph_record(
                transition.successor,
                receipt,
                device=resolved_device,
            )
            for field in (
                "active",
                "root",
                "kind",
                "type",
                "constructor",
                "variable",
                "children",
                "child_type",
                "child_mask",
                "capacity",
            ):
                output[f"successor_{field}"][batch_index, action_index] = graph_tensors[
                    field
                ]
            graph_orders.append(
                tuple(node.storage_id for node in transition.successor.nodes)
            )
            binding_orders.append(
                tuple(binding.variable_id for binding in transition.bindings)
            )
        training_receipts.append(
            TrainingTensorReceipt(
                packet_digest=receipt.packet_digest,
                graph_node_record_orders=tuple(graph_orders),
                binding_record_orders=tuple(binding_orders),
            )
        )
    return TensorizedNeuralTcrrTraining(
        tensors=NeuralTcrrTrainingTensors(**output),
        packet_receipts=packet_tensorization.receipts,
        training_receipts=tuple(training_receipts),
    )


def _one_hot_identifier(
    values: Tensor,
    identifiers: tuple[str, ...],
    *,
    location: str,
) -> str:
    positions = torch.nonzero(values, as_tuple=False).flatten().tolist()
    if len(positions) != 1 or positions[0] >= len(identifiers):
        raise NeuralTcrrTrainingTensorError(f"{location} is not one-hot")
    return identifiers[positions[0]]


def detensorize_neural_tcrr_training_record(
    value: TensorizedNeuralTcrrTraining,
    batch_index: int,
) -> ExpectedTransitionRecord:
    """Reconstruct one exact offline record from its padded action set."""

    if not 0 <= batch_index < len(value.packet_receipts):
        raise IndexError(batch_index)
    tensors = value.tensors
    receipt = value.packet_receipts[batch_index]
    training_receipt = value.training_receipts[batch_index]
    actions = []
    action_count = int(tensors.action_mask[batch_index].sum().item())
    if action_count != len(training_receipt.graph_node_record_orders):
        raise NeuralTcrrTrainingTensorError("action receipt cardinality mismatch")
    if action_count != len(training_receipt.binding_record_orders):
        raise NeuralTcrrTrainingTensorError("binding receipt cardinality mismatch")
    for action_index in range(action_count):
        rule_id = _one_hot_identifier(
            tensors.rule_pointer[batch_index, action_index],
            receipt.rule_ids,
            location="rule pointer",
        )
        path_values = tensors.path_tokens[batch_index, action_index]
        path_mask = tensors.path_token_mask[batch_index, action_index]
        visible = path_values[path_mask].tolist()
        if not visible or visible[-1] != PATH_STOP:
            raise NeuralTcrrTrainingTensorError("path lacks one terminal STOP")
        if PATH_STOP in visible[:-1] or any(token >= A for token in visible[:-1]):
            raise NeuralTcrrTrainingTensorError("path has an invalid token")
        target_storage = _one_hot_identifier(
            tensors.target_storage_pointer[batch_index, action_index],
            receipt.storage_ids,
            location="target storage pointer",
        )
        bindings_by_variable = {}
        for variable_position, variable_id in enumerate(receipt.variable_ids):
            if not bool(
                tensors.variable_binding_mask[
                    batch_index,
                    action_index,
                    variable_position,
                ]
            ):
                continue
            storage_id = _one_hot_identifier(
                tensors.variable_binding[
                    batch_index,
                    action_index,
                    variable_position,
                ],
                receipt.storage_ids,
                location="binding storage pointer",
            )
            bindings_by_variable[variable_id] = BindingRecord(variable_id, storage_id)
        binding_order = training_receipt.binding_record_orders[action_index]
        if set(binding_order) != set(bindings_by_variable):
            raise NeuralTcrrTrainingTensorError("binding order receipt is inconsistent")
        bindings = tuple(
            bindings_by_variable[variable_id] for variable_id in binding_order
        )
        graph = decode_graph_record(
            active=tensors.successor_active[batch_index, action_index],
            root=tensors.successor_root[batch_index, action_index],
            kind=tensors.successor_kind[batch_index, action_index],
            type_ref=tensors.successor_type[batch_index, action_index],
            constructor_ref=tensors.successor_constructor[
                batch_index,
                action_index,
            ],
            variable_ref=tensors.successor_variable[
                batch_index,
                action_index,
            ],
            children=tensors.successor_children[batch_index, action_index],
            child_mask=tensors.successor_child_mask[batch_index, action_index],
            capacity=tensors.successor_capacity[batch_index, action_index],
            receipt=receipt,
            node_record_order=training_receipt.graph_node_record_orders[action_index],
        )
        actions.append(
            ExpectedTransition(
                rule_id=rule_id,
                occurrence_path=tuple(int(token) for token in visible[:-1]),
                target_storage_id=target_storage,
                bindings=bindings,
                successor=graph,
            )
        )
    return ExpectedTransitionRecord(
        packet_sha256=training_receipt.packet_digest,
        transitions=tuple(actions),
    )


__all__ = [
    "MAX_ACTIONS",
    "NeuralTcrrTrainingTensorError",
    "NeuralTcrrTrainingTensors",
    "PATH_PAD",
    "PATH_STOP",
    "PATH_VOCAB_SIZE",
    "PATH_WIDTH",
    "TensorizedNeuralTcrrTraining",
    "TrainingTensorReceipt",
    "detensorize_neural_tcrr_training_record",
    "tensorize_neural_tcrr_training",
]
