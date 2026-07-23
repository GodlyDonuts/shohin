"""Source-deleted tensor boundary for neural TCRR packets.

Opaque identifiers are used only as episode-local equality keys. They are
never converted to numeric features. The returned ``tensors`` object is the
model-visible boundary; ``receipts`` retain opaque coordinate names solely for
offline custody, reindex checks, and exact packet reconstruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from neural_tcrr_board import (
    GraphNodeRecord,
    GraphRecord,
    MAX_ARITY,
    MAX_CAPACITY,
    MAX_CONSTRUCTORS,
    MAX_PATH_DEPTH,
    MAX_RULE_SIDE_NODES,
    MAX_RULES,
    MAX_TYPES,
    ConstructorRecord,
    RuleRecord,
    RuleTermRecord,
    SourceDeletedPacket,
    packet_sha256,
    validate_source_deleted_packet,
)


N = MAX_CAPACITY
C = MAX_CONSTRUCTORS
Y = MAX_TYPES
R = MAX_RULES
P = MAX_RULE_SIDE_NODES
A = MAX_ARITY
D = MAX_PATH_DEPTH
V = N + R * P

KIND_CONSTRUCTOR = 0
KIND_VARIABLE = 1
KIND_EMPTY = 2
GRAPH_KIND_COUNT = 3
TERM_KIND_COUNT = 2


class NeuralTcrrPacketTensorError(ValueError):
    """Raised when a packet cannot cross the frozen tensor boundary."""


@dataclass(frozen=True)
class PacketAxisReceipt:
    """Offline names for one packet's local tensor coordinates."""

    packet_digest: str
    constructor_ids: tuple[str, ...]
    type_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    variable_ids: tuple[str, ...]
    storage_ids: tuple[str, ...]
    graph_node_record_order: tuple[str, ...]


@dataclass(frozen=True)
class NeuralTcrrPacketTensors:
    """Batched model-visible packet tensors."""

    constructor_active: Tensor
    constructor_equal: Tensor
    constructor_result_type: Tensor
    constructor_argument_type: Tensor
    constructor_argument_mask: Tensor

    type_active: Tensor
    type_equal: Tensor

    rule_active: Tensor
    rule_equal: Tensor
    rule_delete: Tensor

    variable_active: Tensor
    variable_equal: Tensor

    storage_active: Tensor
    storage_equal: Tensor

    graph_active: Tensor
    graph_root: Tensor
    graph_kind: Tensor
    graph_type: Tensor
    graph_constructor: Tensor
    graph_variable: Tensor
    graph_children: Tensor
    graph_child_type: Tensor
    graph_child_mask: Tensor
    graph_capacity: Tensor

    lhs_active: Tensor
    lhs_kind: Tensor
    lhs_type: Tensor
    lhs_constructor: Tensor
    lhs_variable: Tensor
    lhs_parent_child: Tensor
    lhs_parent_child_type: Tensor
    lhs_child_mask: Tensor
    lhs_binder_equal: Tensor

    rhs_active: Tensor
    rhs_kind: Tensor
    rhs_type: Tensor
    rhs_constructor: Tensor
    rhs_variable: Tensor
    rhs_parent_child: Tensor
    rhs_parent_child_type: Tensor
    rhs_child_mask: Tensor
    rhs_binder_equal: Tensor
    rhs_to_lhs_binder_equal: Tensor


@dataclass(frozen=True)
class TensorizedNeuralTcrrPackets:
    """Packet tensors plus non-model custody receipts."""

    tensors: NeuralTcrrPacketTensors
    receipts: tuple[PacketAxisReceipt, ...]


@dataclass(frozen=True)
class _FlatTerm:
    record: RuleTermRecord
    parent: int | None
    argument: int | None


@dataclass(frozen=True)
class _EpisodeAxes:
    constructor_ids: tuple[str, ...]
    type_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    variable_ids: tuple[str, ...]
    storage_ids: tuple[str, ...]

    @property
    def constructor_index(self) -> dict[str, int]:
        return {
            identifier: index for index, identifier in enumerate(self.constructor_ids)
        }

    @property
    def type_index(self) -> dict[str, int]:
        return {identifier: index for index, identifier in enumerate(self.type_ids)}

    @property
    def rule_index(self) -> dict[str, int]:
        return {identifier: index for index, identifier in enumerate(self.rule_ids)}

    @property
    def variable_index(self) -> dict[str, int]:
        return {identifier: index for index, identifier in enumerate(self.variable_ids)}

    @property
    def storage_index(self) -> dict[str, int]:
        return {identifier: index for index, identifier in enumerate(self.storage_ids)}


def _append_once(output: list[str], seen: set[str], value: str) -> None:
    if value not in seen:
        seen.add(value)
        output.append(value)


def _walk_terms(term: RuleTermRecord) -> tuple[RuleTermRecord, ...]:
    output = [term]
    for child in term.children:
        output.extend(_walk_terms(child))
    return tuple(output)


def _episode_axes(packet: SourceDeletedPacket) -> _EpisodeAxes:
    constructor_ids = tuple(item.identifier for item in packet.constructors)
    type_output: list[str] = []
    seen_types: set[str] = set()
    for item in packet.constructors:
        _append_once(type_output, seen_types, item.result_type)
        for type_id in item.argument_types:
            _append_once(type_output, seen_types, type_id)

    variable_output: list[str] = []
    seen_variables: set[str] = set()
    for rule in packet.rules:
        for term in _walk_terms(rule.lhs):
            if term.variable_id is not None:
                _append_once(variable_output, seen_variables, term.variable_id)
        if rule.rhs is not None:
            for term in _walk_terms(rule.rhs):
                if term.variable_id is not None:
                    _append_once(variable_output, seen_variables, term.variable_id)
    node_by_storage = {node.storage_id: node for node in packet.graph.nodes}
    for storage_id in packet.graph.reservoir:
        node = node_by_storage.get(storage_id)
        if node is not None and node.variable_id is not None:
            _append_once(variable_output, seen_variables, node.variable_id)

    axes = _EpisodeAxes(
        constructor_ids=constructor_ids,
        type_ids=tuple(type_output),
        rule_ids=tuple(item.identifier for item in packet.rules),
        variable_ids=tuple(variable_output),
        storage_ids=packet.graph.reservoir,
    )
    if len(axes.variable_ids) > V:
        raise NeuralTcrrPacketTensorError(
            f"variable count {len(axes.variable_ids)} exceeds frozen geometry {V}"
        )
    return axes


def _flat_terms(term: RuleTermRecord) -> tuple[_FlatTerm, ...]:
    output: list[_FlatTerm] = []

    def visit(
        active: RuleTermRecord,
        parent: int | None,
        argument: int | None,
    ) -> None:
        index = len(output)
        output.append(_FlatTerm(active, parent, argument))
        for child_argument, child in enumerate(active.children):
            visit(child, index, child_argument)

    visit(term, None, None)
    if len(output) > P:
        raise NeuralTcrrPacketTensorError(
            f"rule side has {len(output)} terms, exceeding frozen geometry {P}"
        )
    return tuple(output)


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _identity(
    active_count: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    output = _zeros((width, width), dtype=torch.bool, device=device)
    if active_count:
        output[:active_count, :active_count] = torch.eye(
            active_count,
            dtype=torch.bool,
            device=device,
        )
    return output


def _fill_term_side(
    *,
    flat: tuple[_FlatTerm, ...],
    rule_index: int,
    axes: _EpisodeAxes,
    active: Tensor,
    kind: Tensor,
    type_ref: Tensor,
    constructor_ref: Tensor,
    variable_ref: Tensor,
    parent_child: Tensor,
    parent_child_type: Tensor,
    child_mask: Tensor,
    binder_equal: Tensor,
) -> None:
    type_index = axes.type_index
    constructor_index = axes.constructor_index
    variable_index = axes.variable_index
    variable_by_position: dict[int, str] = {}
    for position, item in enumerate(flat):
        record = item.record
        active[rule_index, position] = True
        kind_index = KIND_CONSTRUCTOR if record.kind == "constructor" else KIND_VARIABLE
        kind[rule_index, position, kind_index] = True
        type_ref[rule_index, position, type_index[record.type_id]] = True
        if record.constructor_id is not None:
            constructor_ref[
                rule_index,
                position,
                constructor_index[record.constructor_id],
            ] = True
        if record.variable_id is not None:
            variable_ref[
                rule_index,
                position,
                variable_index[record.variable_id],
            ] = True
            variable_by_position[position] = record.variable_id
        if item.parent is not None and item.argument is not None:
            parent_child[
                rule_index,
                item.parent,
                item.argument,
                position,
            ] = True
            parent_child_type[
                rule_index,
                item.parent,
                item.argument,
                type_index[record.type_id],
            ] = True
            child_mask[rule_index, item.parent, item.argument] = True
    for left, left_id in variable_by_position.items():
        for right, right_id in variable_by_position.items():
            binder_equal[rule_index, left, right] = left_id == right_id


def _fill_graph(
    *,
    graph: GraphRecord,
    axes: _EpisodeAxes,
    active: Tensor,
    root: Tensor,
    kind: Tensor,
    type_ref: Tensor,
    constructor_ref: Tensor,
    variable_ref: Tensor,
    children: Tensor,
    child_type: Tensor,
    child_mask: Tensor,
    capacity: Tensor,
) -> None:
    storage_index = axes.storage_index
    type_index = axes.type_index
    constructor_index = axes.constructor_index
    variable_index = axes.variable_index
    node_by_storage = {node.storage_id: node for node in graph.nodes}
    capacity[: len(graph.reservoir)] = True
    root[N] = graph.root is None
    if graph.root is not None:
        root[storage_index[graph.root]] = True
    for storage_position, storage_id in enumerate(graph.reservoir):
        node = node_by_storage.get(storage_id)
        if node is None:
            kind[storage_position, KIND_EMPTY] = True
            continue
        active[storage_position] = True
        kind_index = KIND_CONSTRUCTOR if node.kind == "constructor" else KIND_VARIABLE
        kind[storage_position, kind_index] = True
        type_ref[storage_position, type_index[node.type_id]] = True
        if node.constructor_id is not None:
            constructor_ref[
                storage_position,
                constructor_index[node.constructor_id],
            ] = True
        if node.variable_id is not None:
            variable_ref[
                storage_position,
                variable_index[node.variable_id],
            ] = True
        for argument, child_storage in enumerate(node.children):
            child_position = storage_index[child_storage]
            children[storage_position, argument, child_position] = True
            child_node = node_by_storage[child_storage]
            child_type[
                storage_position,
                argument,
                type_index[child_node.type_id],
            ] = True
            child_mask[storage_position, argument] = True
        for argument in range(len(node.children), A):
            children[storage_position, argument, N] = True
    for storage_position in range(len(graph.reservoir)):
        if not active[storage_position]:
            children[storage_position, :, N] = True


def _new_packet_tensors(
    batch_size: int,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    boolean = torch.bool
    return {
        "constructor_active": _zeros((batch_size, C), dtype=boolean, device=device),
        "constructor_equal": _zeros((batch_size, C, C), dtype=boolean, device=device),
        "constructor_result_type": _zeros(
            (batch_size, C, Y), dtype=boolean, device=device
        ),
        "constructor_argument_type": _zeros(
            (batch_size, C, A, Y), dtype=boolean, device=device
        ),
        "constructor_argument_mask": _zeros(
            (batch_size, C, A), dtype=boolean, device=device
        ),
        "type_active": _zeros((batch_size, Y), dtype=boolean, device=device),
        "type_equal": _zeros((batch_size, Y, Y), dtype=boolean, device=device),
        "rule_active": _zeros((batch_size, R), dtype=boolean, device=device),
        "rule_equal": _zeros((batch_size, R, R), dtype=boolean, device=device),
        "rule_delete": _zeros((batch_size, R), dtype=boolean, device=device),
        "variable_active": _zeros((batch_size, V), dtype=boolean, device=device),
        "variable_equal": _zeros((batch_size, V, V), dtype=boolean, device=device),
        "storage_active": _zeros((batch_size, N), dtype=boolean, device=device),
        "storage_equal": _zeros((batch_size, N, N), dtype=boolean, device=device),
        "graph_active": _zeros((batch_size, N), dtype=boolean, device=device),
        "graph_root": _zeros((batch_size, N + 1), dtype=boolean, device=device),
        "graph_kind": _zeros(
            (batch_size, N, GRAPH_KIND_COUNT), dtype=boolean, device=device
        ),
        "graph_type": _zeros((batch_size, N, Y), dtype=boolean, device=device),
        "graph_constructor": _zeros((batch_size, N, C), dtype=boolean, device=device),
        "graph_variable": _zeros((batch_size, N, V), dtype=boolean, device=device),
        "graph_children": _zeros(
            (batch_size, N, A, N + 1), dtype=boolean, device=device
        ),
        "graph_child_type": _zeros((batch_size, N, A, Y), dtype=boolean, device=device),
        "graph_child_mask": _zeros((batch_size, N, A), dtype=boolean, device=device),
        "graph_capacity": _zeros((batch_size, N), dtype=boolean, device=device),
        "lhs_active": _zeros((batch_size, R, P), dtype=boolean, device=device),
        "lhs_kind": _zeros(
            (batch_size, R, P, TERM_KIND_COUNT),
            dtype=boolean,
            device=device,
        ),
        "lhs_type": _zeros((batch_size, R, P, Y), dtype=boolean, device=device),
        "lhs_constructor": _zeros((batch_size, R, P, C), dtype=boolean, device=device),
        "lhs_variable": _zeros((batch_size, R, P, V), dtype=boolean, device=device),
        "lhs_parent_child": _zeros(
            (batch_size, R, P, A, P), dtype=boolean, device=device
        ),
        "lhs_parent_child_type": _zeros(
            (batch_size, R, P, A, Y), dtype=boolean, device=device
        ),
        "lhs_child_mask": _zeros((batch_size, R, P, A), dtype=boolean, device=device),
        "lhs_binder_equal": _zeros((batch_size, R, P, P), dtype=boolean, device=device),
        "rhs_active": _zeros((batch_size, R, P), dtype=boolean, device=device),
        "rhs_kind": _zeros(
            (batch_size, R, P, TERM_KIND_COUNT),
            dtype=boolean,
            device=device,
        ),
        "rhs_type": _zeros((batch_size, R, P, Y), dtype=boolean, device=device),
        "rhs_constructor": _zeros((batch_size, R, P, C), dtype=boolean, device=device),
        "rhs_variable": _zeros((batch_size, R, P, V), dtype=boolean, device=device),
        "rhs_parent_child": _zeros(
            (batch_size, R, P, A, P), dtype=boolean, device=device
        ),
        "rhs_parent_child_type": _zeros(
            (batch_size, R, P, A, Y), dtype=boolean, device=device
        ),
        "rhs_child_mask": _zeros((batch_size, R, P, A), dtype=boolean, device=device),
        "rhs_binder_equal": _zeros((batch_size, R, P, P), dtype=boolean, device=device),
        "rhs_to_lhs_binder_equal": _zeros(
            (batch_size, R, P, P), dtype=boolean, device=device
        ),
    }


def tensorize_neural_tcrr_packets(
    packets: Sequence[SourceDeletedPacket],
    *,
    device: torch.device | str = "cpu",
) -> TensorizedNeuralTcrrPackets:
    """Tensorize packet objects or packet-only loader output."""

    if not packets:
        raise NeuralTcrrPacketTensorError("at least one packet is required")
    resolved_device = torch.device(device)
    output = _new_packet_tensors(len(packets), device=resolved_device)
    receipts: list[PacketAxisReceipt] = []

    for batch_index, packet in enumerate(packets):
        if not isinstance(packet, SourceDeletedPacket):
            raise NeuralTcrrPacketTensorError(
                "packet tensorizer accepts SourceDeletedPacket objects only"
            )
        validate_source_deleted_packet(packet)
        axes = _episode_axes(packet)
        constructor_index = axes.constructor_index
        type_index = axes.type_index

        output["constructor_active"][
            batch_index,
            : len(axes.constructor_ids),
        ] = True
        output["constructor_equal"][batch_index] = _identity(
            len(axes.constructor_ids),
            C,
            device=resolved_device,
        )
        output["type_active"][batch_index, : len(axes.type_ids)] = True
        output["type_equal"][batch_index] = _identity(
            len(axes.type_ids),
            Y,
            device=resolved_device,
        )
        output["rule_active"][batch_index, : len(axes.rule_ids)] = True
        output["rule_equal"][batch_index] = _identity(
            len(axes.rule_ids),
            R,
            device=resolved_device,
        )
        output["variable_active"][batch_index, : len(axes.variable_ids)] = True
        output["variable_equal"][batch_index] = _identity(
            len(axes.variable_ids),
            V,
            device=resolved_device,
        )
        output["storage_active"][batch_index, : len(axes.storage_ids)] = True
        output["storage_equal"][batch_index] = _identity(
            len(axes.storage_ids),
            N,
            device=resolved_device,
        )

        for item in packet.constructors:
            constructor_position = constructor_index[item.identifier]
            output["constructor_result_type"][
                batch_index,
                constructor_position,
                type_index[item.result_type],
            ] = True
            for argument, type_id in enumerate(item.argument_types):
                output["constructor_argument_type"][
                    batch_index,
                    constructor_position,
                    argument,
                    type_index[type_id],
                ] = True
                output["constructor_argument_mask"][
                    batch_index,
                    constructor_position,
                    argument,
                ] = True

        for rule_position, rule in enumerate(packet.rules):
            output["rule_delete"][batch_index, rule_position] = rule.rhs is None
            lhs_flat = _flat_terms(rule.lhs)
            _fill_term_side(
                flat=lhs_flat,
                rule_index=rule_position,
                axes=axes,
                active=output["lhs_active"][batch_index],
                kind=output["lhs_kind"][batch_index],
                type_ref=output["lhs_type"][batch_index],
                constructor_ref=output["lhs_constructor"][batch_index],
                variable_ref=output["lhs_variable"][batch_index],
                parent_child=output["lhs_parent_child"][batch_index],
                parent_child_type=output["lhs_parent_child_type"][batch_index],
                child_mask=output["lhs_child_mask"][batch_index],
                binder_equal=output["lhs_binder_equal"][batch_index],
            )
            if rule.rhs is not None:
                rhs_flat = _flat_terms(rule.rhs)
                _fill_term_side(
                    flat=rhs_flat,
                    rule_index=rule_position,
                    axes=axes,
                    active=output["rhs_active"][batch_index],
                    kind=output["rhs_kind"][batch_index],
                    type_ref=output["rhs_type"][batch_index],
                    constructor_ref=output["rhs_constructor"][batch_index],
                    variable_ref=output["rhs_variable"][batch_index],
                    parent_child=output["rhs_parent_child"][batch_index],
                    parent_child_type=output["rhs_parent_child_type"][batch_index],
                    child_mask=output["rhs_child_mask"][batch_index],
                    binder_equal=output["rhs_binder_equal"][batch_index],
                )
                lhs_variables = {
                    item.record.variable_id: position
                    for position, item in enumerate(lhs_flat)
                    if item.record.variable_id is not None
                }
                for rhs_position, item in enumerate(rhs_flat):
                    variable_id = item.record.variable_id
                    if variable_id is None:
                        continue
                    for lhs_position, lhs_item in enumerate(lhs_flat):
                        output["rhs_to_lhs_binder_equal"][
                            batch_index,
                            rule_position,
                            rhs_position,
                            lhs_position,
                        ] = lhs_item.record.variable_id == variable_id
                    if variable_id not in lhs_variables:
                        raise NeuralTcrrPacketTensorError(
                            "RHS binder is absent from the LHS"
                        )

        _fill_graph(
            graph=packet.graph,
            axes=axes,
            active=output["graph_active"][batch_index],
            root=output["graph_root"][batch_index],
            kind=output["graph_kind"][batch_index],
            type_ref=output["graph_type"][batch_index],
            constructor_ref=output["graph_constructor"][batch_index],
            variable_ref=output["graph_variable"][batch_index],
            children=output["graph_children"][batch_index],
            child_type=output["graph_child_type"][batch_index],
            child_mask=output["graph_child_mask"][batch_index],
            capacity=output["graph_capacity"][batch_index],
        )
        receipts.append(
            PacketAxisReceipt(
                packet_digest=packet_sha256(packet),
                constructor_ids=axes.constructor_ids,
                type_ids=axes.type_ids,
                rule_ids=axes.rule_ids,
                variable_ids=axes.variable_ids,
                storage_ids=axes.storage_ids,
                graph_node_record_order=tuple(
                    node.storage_id for node in packet.graph.nodes
                ),
            )
        )

    return TensorizedNeuralTcrrPackets(
        tensors=NeuralTcrrPacketTensors(**output),
        receipts=tuple(receipts),
    )


def tensorize_graph_record(
    graph: GraphRecord,
    receipt: PacketAxisReceipt,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Encode one graph against an existing packet's local coordinates."""

    if graph.reservoir != receipt.storage_ids:
        raise NeuralTcrrPacketTensorError(
            "graph reservoir does not match the packet coordinate receipt"
        )
    resolved_device = torch.device(device)
    axes = _EpisodeAxes(
        constructor_ids=receipt.constructor_ids,
        type_ids=receipt.type_ids,
        rule_ids=receipt.rule_ids,
        variable_ids=receipt.variable_ids,
        storage_ids=receipt.storage_ids,
    )
    output = {
        "active": _zeros((N,), dtype=torch.bool, device=resolved_device),
        "root": _zeros((N + 1,), dtype=torch.bool, device=resolved_device),
        "kind": _zeros(
            (N, GRAPH_KIND_COUNT),
            dtype=torch.bool,
            device=resolved_device,
        ),
        "type": _zeros((N, Y), dtype=torch.bool, device=resolved_device),
        "constructor": _zeros((N, C), dtype=torch.bool, device=resolved_device),
        "variable": _zeros((N, V), dtype=torch.bool, device=resolved_device),
        "children": _zeros(
            (N, A, N + 1),
            dtype=torch.bool,
            device=resolved_device,
        ),
        "child_type": _zeros(
            (N, A, Y),
            dtype=torch.bool,
            device=resolved_device,
        ),
        "child_mask": _zeros((N, A), dtype=torch.bool, device=resolved_device),
        "capacity": _zeros((N,), dtype=torch.bool, device=resolved_device),
    }
    _fill_graph(
        graph=graph,
        axes=axes,
        active=output["active"],
        root=output["root"],
        kind=output["kind"],
        type_ref=output["type"],
        constructor_ref=output["constructor"],
        variable_ref=output["variable"],
        children=output["children"],
        child_type=output["child_type"],
        child_mask=output["child_mask"],
        capacity=output["capacity"],
    )
    return output


def _active_reference(
    values: Tensor,
    identifiers: tuple[str, ...],
    *,
    location: str,
) -> str:
    active = torch.nonzero(values, as_tuple=False).flatten().tolist()
    if len(active) != 1 or active[0] >= len(identifiers):
        raise NeuralTcrrPacketTensorError(f"{location} is not one-hot")
    return identifiers[active[0]]


def _decode_term(
    *,
    rule_index: int,
    position: int,
    side: str,
    tensors: NeuralTcrrPacketTensors,
    receipt: PacketAxisReceipt,
    batch_index: int,
) -> RuleTermRecord:
    prefix = "lhs" if side == "lhs" else "rhs"
    active = getattr(tensors, f"{prefix}_active")[batch_index, rule_index]
    kind = getattr(tensors, f"{prefix}_kind")[batch_index, rule_index]
    type_ref = getattr(tensors, f"{prefix}_type")[batch_index, rule_index]
    constructor_ref = getattr(tensors, f"{prefix}_constructor")[
        batch_index,
        rule_index,
    ]
    variable_ref = getattr(tensors, f"{prefix}_variable")[
        batch_index,
        rule_index,
    ]
    parent_child = getattr(tensors, f"{prefix}_parent_child")[
        batch_index,
        rule_index,
    ]
    child_mask = getattr(tensors, f"{prefix}_child_mask")[
        batch_index,
        rule_index,
    ]
    if not bool(active[position]):
        raise NeuralTcrrPacketTensorError("term reconstruction reached padding")
    type_id = _active_reference(
        type_ref[position],
        receipt.type_ids,
        location=f"{side} type",
    )
    kind_index = torch.nonzero(kind[position], as_tuple=False).flatten().tolist()
    if kind_index == [KIND_VARIABLE]:
        variable_id = _active_reference(
            variable_ref[position],
            receipt.variable_ids,
            location=f"{side} variable",
        )
        return RuleTermRecord(
            kind="variable",
            type_id=type_id,
            variable_id=variable_id,
        )
    if kind_index != [KIND_CONSTRUCTOR]:
        raise NeuralTcrrPacketTensorError(f"{side} kind is not one-hot")
    constructor_id = _active_reference(
        constructor_ref[position],
        receipt.constructor_ids,
        location=f"{side} constructor",
    )
    children = []
    for argument in range(A):
        if not bool(child_mask[position, argument]):
            continue
        child_positions = (
            torch.nonzero(
                parent_child[position, argument],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        if len(child_positions) != 1:
            raise NeuralTcrrPacketTensorError(f"{side} child is not one-hot")
        children.append(
            _decode_term(
                rule_index=rule_index,
                position=child_positions[0],
                side=side,
                tensors=tensors,
                receipt=receipt,
                batch_index=batch_index,
            )
        )
    return RuleTermRecord(
        kind="constructor",
        type_id=type_id,
        constructor_id=constructor_id,
        children=tuple(children),
    )


def decode_graph_record(
    *,
    active: Tensor,
    root: Tensor,
    kind: Tensor,
    type_ref: Tensor,
    constructor_ref: Tensor,
    variable_ref: Tensor,
    children: Tensor,
    child_mask: Tensor,
    capacity: Tensor,
    receipt: PacketAxisReceipt,
    node_record_order: tuple[str, ...] | None = None,
) -> GraphRecord:
    """Reconstruct one graph from one unbatched tensor group."""

    capacity_count = int(capacity.sum().item())
    if not torch.equal(
        capacity,
        torch.arange(N, device=capacity.device) < capacity_count,
    ):
        raise NeuralTcrrPacketTensorError("capacity mask is not a prefix")
    storage_ids = receipt.storage_ids[:capacity_count]
    root_positions = torch.nonzero(root, as_tuple=False).flatten().tolist()
    if len(root_positions) != 1:
        raise NeuralTcrrPacketTensorError("graph root is not one-hot")
    root_position = root_positions[0]
    root_id = None if root_position == N else storage_ids[root_position]
    nodes_by_storage: dict[str, GraphNodeRecord] = {}
    for storage_position, storage_id in enumerate(storage_ids):
        if not bool(active[storage_position]):
            continue
        type_id = _active_reference(
            type_ref[storage_position],
            receipt.type_ids,
            location="graph type",
        )
        kind_positions = (
            torch.nonzero(
                kind[storage_position],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        if kind_positions == [KIND_VARIABLE]:
            variable_id = _active_reference(
                variable_ref[storage_position],
                receipt.variable_ids,
                location="graph variable",
            )
            nodes_by_storage[storage_id] = GraphNodeRecord(
                storage_id=storage_id,
                kind="variable",
                type_id=type_id,
                variable_id=variable_id,
            )
            continue
        if kind_positions != [KIND_CONSTRUCTOR]:
            raise NeuralTcrrPacketTensorError("graph kind is not one-hot")
        constructor_id = _active_reference(
            constructor_ref[storage_position],
            receipt.constructor_ids,
            location="graph constructor",
        )
        child_ids = []
        for argument in range(A):
            if not bool(child_mask[storage_position, argument]):
                continue
            child_positions = (
                torch.nonzero(
                    children[storage_position, argument],
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            if len(child_positions) != 1 or child_positions[0] >= capacity_count:
                raise NeuralTcrrPacketTensorError("graph child is not one-hot")
            child_ids.append(storage_ids[child_positions[0]])
        nodes_by_storage[storage_id] = GraphNodeRecord(
            storage_id=storage_id,
            kind="constructor",
            type_id=type_id,
            constructor_id=constructor_id,
            children=tuple(child_ids),
        )
    order = node_record_order or tuple(
        storage_id for storage_id in storage_ids if storage_id in nodes_by_storage
    )
    if set(order) != set(nodes_by_storage):
        raise NeuralTcrrPacketTensorError("graph node-order receipt is inconsistent")
    return GraphRecord(
        reservoir=storage_ids,
        root=root_id,
        nodes=tuple(nodes_by_storage[storage_id] for storage_id in order),
    )


def detensorize_neural_tcrr_packet(
    value: TensorizedNeuralTcrrPackets,
    batch_index: int,
) -> SourceDeletedPacket:
    """Exactly reconstruct one packet using its offline coordinate receipt."""

    if not 0 <= batch_index < len(value.receipts):
        raise IndexError(batch_index)
    tensors = value.tensors
    receipt = value.receipts[batch_index]
    constructors = []
    for constructor_index, constructor_id in enumerate(receipt.constructor_ids):
        result_type = _active_reference(
            tensors.constructor_result_type[batch_index, constructor_index],
            receipt.type_ids,
            location="constructor result type",
        )
        argument_types = []
        for argument in range(A):
            if not bool(
                tensors.constructor_argument_mask[
                    batch_index,
                    constructor_index,
                    argument,
                ]
            ):
                continue
            argument_types.append(
                _active_reference(
                    tensors.constructor_argument_type[
                        batch_index,
                        constructor_index,
                        argument,
                    ],
                    receipt.type_ids,
                    location="constructor argument type",
                )
            )
        constructors.append(
            ConstructorRecord(
                identifier=constructor_id,
                result_type=result_type,
                argument_types=tuple(argument_types),
            )
        )
    rules = []
    for rule_index, rule_id in enumerate(receipt.rule_ids):
        lhs = _decode_term(
            rule_index=rule_index,
            position=0,
            side="lhs",
            tensors=tensors,
            receipt=receipt,
            batch_index=batch_index,
        )
        rhs = (
            None
            if bool(tensors.rule_delete[batch_index, rule_index])
            else _decode_term(
                rule_index=rule_index,
                position=0,
                side="rhs",
                tensors=tensors,
                receipt=receipt,
                batch_index=batch_index,
            )
        )
        rules.append(RuleRecord(identifier=rule_id, lhs=lhs, rhs=rhs))
    graph = decode_graph_record(
        active=tensors.graph_active[batch_index],
        root=tensors.graph_root[batch_index],
        kind=tensors.graph_kind[batch_index],
        type_ref=tensors.graph_type[batch_index],
        constructor_ref=tensors.graph_constructor[batch_index],
        variable_ref=tensors.graph_variable[batch_index],
        children=tensors.graph_children[batch_index],
        child_mask=tensors.graph_child_mask[batch_index],
        capacity=tensors.graph_capacity[batch_index],
        receipt=receipt,
        node_record_order=receipt.graph_node_record_order,
    )
    packet = SourceDeletedPacket(
        constructors=tuple(constructors),
        rules=tuple(rules),
        graph=graph,
    )
    validate_source_deleted_packet(packet)
    if packet_sha256(packet) != receipt.packet_digest:
        raise NeuralTcrrPacketTensorError("packet reconstruction digest mismatch")
    return packet


__all__ = [
    "A",
    "C",
    "D",
    "GRAPH_KIND_COUNT",
    "KIND_CONSTRUCTOR",
    "KIND_EMPTY",
    "KIND_VARIABLE",
    "N",
    "NeuralTcrrPacketTensorError",
    "NeuralTcrrPacketTensors",
    "P",
    "PacketAxisReceipt",
    "R",
    "TERM_KIND_COUNT",
    "TensorizedNeuralTcrrPackets",
    "V",
    "Y",
    "decode_graph_record",
    "detensorize_neural_tcrr_packet",
    "tensorize_graph_record",
    "tensorize_neural_tcrr_packets",
]
