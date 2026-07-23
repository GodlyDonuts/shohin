"""Atomic structural commit boundary for neural TCRR graph proposals.

The committer sees only anonymous packet tensors and a discrete graph
transaction. It validates frozen tensor and graph invariants, then installs
KEEP, WRITE, and CLEAR operations without search or correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from tensorize_neural_tcrr_packets import (
    A,
    C,
    N,
    V,
    Y,
    GRAPH_KIND_COUNT,
    NeuralTcrrPacketTensors,
)
from torch import Tensor


NODE_KEEP = 0
NODE_WRITE = 1
NODE_CLEAR = 2
NODE_OPERATION_COUNT = 3

KIND_CONSTRUCTOR = 0
KIND_VARIABLE = 1
KIND_EMPTY = 2

CHILD_ABSENT = 0
CHILD_PRESENT = 1
CHILD_PRESENCE_COUNT = 2


class NeuralTcrrCommitError(ValueError):
    """A fail-closed commit rejection with a stable audit code."""

    def __init__(
        self,
        reason_code: str,
        *,
        batch_index: int | None = None,
        location: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.batch_index = batch_index
        self.location = location
        details = [reason_code]
        if batch_index is not None:
            details.append(f"batch={batch_index}")
        if location is not None:
            details.append(f"location={location}")
        super().__init__(": ".join(details))


@dataclass(frozen=True)
class NeuralTcrrGraphTransaction:
    """Conditional one-hot choices for one atomic graph mutation."""

    node_operation: Tensor
    root_pointer: Tensor
    node_kind: Tensor
    node_type_pointer: Tensor
    node_constructor_pointer: Tensor
    node_variable_pointer: Tensor
    child_pointer: Tensor
    child_presence: Tensor


@dataclass(frozen=True)
class NeuralTcrrGraphTensors:
    """Validated graph tensors emitted by a successful commit."""

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


class NeuralTcrrGraphDeltaLike(Protocol):
    """Structural interface accepted by deterministic proposal decoding."""

    node_operation_logits: Tensor
    node_operation_mask: Tensor
    root_pointer_logits: Tensor
    root_pointer_mask: Tensor
    node_kind_logits: Tensor
    node_kind_mask: Tensor
    node_type_pointer_logits: Tensor
    node_type_pointer_mask: Tensor
    node_constructor_pointer_logits: Tensor
    node_constructor_pointer_mask: Tensor
    node_variable_pointer_logits: Tensor
    node_variable_pointer_mask: Tensor
    child_pointer_logits: Tensor
    child_pointer_mask: Tensor
    child_presence_logits: Tensor
    child_presence_mask: Tensor


def _reject(
    reason_code: str,
    *,
    batch_index: int | None = None,
    location: str | None = None,
) -> None:
    raise NeuralTcrrCommitError(
        reason_code,
        batch_index=batch_index,
        location=location,
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype | None,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        _reject("tensor_required", location=name)
    if value.shape != shape:
        _reject("tensor_shape", location=name)
    if dtype is not None and value.dtype is not dtype:
        _reject("tensor_dtype", location=name)
    if value.device != device:
        _reject("tensor_device", location=name)
    return value


def _is_zero(value: Tensor) -> bool:
    return not bool(torch.count_nonzero(value).item())


def _one_hot_index(
    value: Tensor,
    *,
    reason_code: str,
    batch_index: int,
    location: str,
    active_mask: Tensor | None = None,
) -> int:
    positions = torch.nonzero(value, as_tuple=False).flatten()
    if positions.numel() != 1:
        _reject(reason_code, batch_index=batch_index, location=location)
    position = int(positions[0].item())
    if active_mask is not None and not bool(active_mask[position].item()):
        _reject(reason_code, batch_index=batch_index, location=location)
    return position


def _validate_static_boundary(packets: NeuralTcrrPacketTensors) -> int:
    if not isinstance(packets, NeuralTcrrPacketTensors):
        _reject("packet_tensor_boundary_required")
    batch_size = packets.graph_active.shape[0]
    if batch_size <= 0:
        _reject("empty_batch")
    device = packets.graph_active.device
    shapes = {
        "constructor_active": (batch_size, C),
        "constructor_result_type": (batch_size, C, Y),
        "constructor_argument_type": (batch_size, C, A, Y),
        "constructor_argument_mask": (batch_size, C, A),
        "type_active": (batch_size, Y),
        "variable_active": (batch_size, V),
        "storage_active": (batch_size, N),
        "graph_active": (batch_size, N),
        "graph_root": (batch_size, N + 1),
        "graph_kind": (batch_size, N, GRAPH_KIND_COUNT),
        "graph_type": (batch_size, N, Y),
        "graph_constructor": (batch_size, N, C),
        "graph_variable": (batch_size, N, V),
        "graph_children": (batch_size, N, A, N + 1),
        "graph_child_type": (batch_size, N, A, Y),
        "graph_child_mask": (batch_size, N, A),
        "graph_capacity": (batch_size, N),
    }
    for name, shape in shapes.items():
        _require_tensor(
            getattr(packets, name),
            name=name,
            shape=shape,
            dtype=torch.bool,
            device=device,
        )

    if not torch.equal(packets.storage_active, packets.graph_capacity):
        _reject("capacity_storage_mismatch")

    for batch_index in range(batch_size):
        active_types = packets.type_active[batch_index]
        active_constructors = packets.constructor_active[batch_index]
        if _is_zero(active_types):
            _reject("no_active_types", batch_index=batch_index)
        if _is_zero(active_constructors):
            _reject("no_active_constructors", batch_index=batch_index)
        for constructor in range(C):
            active = bool(active_constructors[constructor].item())
            result = packets.constructor_result_type[batch_index, constructor]
            argument_mask = packets.constructor_argument_mask[
                batch_index,
                constructor,
            ]
            argument_types = packets.constructor_argument_type[
                batch_index,
                constructor,
            ]
            if not active:
                if not (
                    _is_zero(result)
                    and _is_zero(argument_mask)
                    and _is_zero(argument_types)
                ):
                    _reject(
                        "inactive_constructor_payload",
                        batch_index=batch_index,
                        location=f"constructor={constructor}",
                    )
                continue
            _one_hot_index(
                result,
                reason_code="constructor_result_not_one_hot",
                batch_index=batch_index,
                location=f"constructor={constructor}",
                active_mask=active_types,
            )
            seen_gap = False
            for argument in range(A):
                declared = bool(argument_mask[argument].item())
                if seen_gap and declared:
                    _reject(
                        "constructor_arity_not_prefix",
                        batch_index=batch_index,
                        location=f"constructor={constructor},argument={argument}",
                    )
                if not declared:
                    seen_gap = True
                    if not _is_zero(argument_types[argument]):
                        _reject(
                            "undeclared_argument_payload",
                            batch_index=batch_index,
                            location=f"constructor={constructor},argument={argument}",
                        )
                    continue
                _one_hot_index(
                    argument_types[argument],
                    reason_code="constructor_argument_not_one_hot",
                    batch_index=batch_index,
                    location=f"constructor={constructor},argument={argument}",
                    active_mask=active_types,
                )
    return batch_size


def _graph_from_packets(packets: NeuralTcrrPacketTensors) -> NeuralTcrrGraphTensors:
    return NeuralTcrrGraphTensors(
        graph_active=packets.graph_active,
        graph_root=packets.graph_root,
        graph_kind=packets.graph_kind,
        graph_type=packets.graph_type,
        graph_constructor=packets.graph_constructor,
        graph_variable=packets.graph_variable,
        graph_children=packets.graph_children,
        graph_child_type=packets.graph_child_type,
        graph_child_mask=packets.graph_child_mask,
        graph_capacity=packets.graph_capacity,
    )


def _validate_graph(
    packets: NeuralTcrrPacketTensors,
    graph: NeuralTcrrGraphTensors,
    *,
    input_graph: bool,
) -> None:
    batch_size = packets.graph_active.shape[0]
    for batch_index in range(batch_size):
        capacity = graph.graph_capacity[batch_index]
        active = graph.graph_active[batch_index]
        if bool((active & ~capacity).any().item()):
            _reject(
                "active_outside_capacity",
                batch_index=batch_index,
            )
        root = _one_hot_index(
            graph.graph_root[batch_index],
            reason_code="root_not_one_hot",
            batch_index=batch_index,
            location="root",
        )
        active_count = int(active.sum().item())
        if root == N:
            if active_count:
                _reject(
                    "null_root_with_active_graph",
                    batch_index=batch_index,
                )
        elif not bool(active[root].item()):
            _reject(
                "root_not_active",
                batch_index=batch_index,
                location=f"storage={root}",
            )

        variable_positions: set[int] = set()
        adjacency: list[list[int]] = [[] for _ in range(N)]
        for storage in range(N):
            location = f"storage={storage}"
            in_capacity = bool(capacity[storage].item())
            is_active = bool(active[storage].item())
            kind = graph.graph_kind[batch_index, storage]
            type_pointer = graph.graph_type[batch_index, storage]
            constructor_pointer = graph.graph_constructor[batch_index, storage]
            variable_pointer = graph.graph_variable[batch_index, storage]
            children = graph.graph_children[batch_index, storage]
            child_types = graph.graph_child_type[batch_index, storage]
            child_mask = graph.graph_child_mask[batch_index, storage]

            if not in_capacity:
                if not (
                    not is_active
                    and _is_zero(kind)
                    and _is_zero(type_pointer)
                    and _is_zero(constructor_pointer)
                    and _is_zero(variable_pointer)
                    and _is_zero(children)
                    and _is_zero(child_types)
                    and _is_zero(child_mask)
                ):
                    _reject(
                        "outside_capacity_payload",
                        batch_index=batch_index,
                        location=location,
                    )
                continue

            if not is_active:
                empty_kind = torch.zeros_like(kind)
                empty_kind[KIND_EMPTY] = True
                expected_children = torch.zeros_like(children)
                expected_children[:, N] = True
                if not (
                    torch.equal(kind, empty_kind)
                    and _is_zero(type_pointer)
                    and _is_zero(constructor_pointer)
                    and _is_zero(variable_pointer)
                    and torch.equal(children, expected_children)
                    and _is_zero(child_types)
                    and _is_zero(child_mask)
                ):
                    _reject(
                        "free_slot_not_canonical",
                        batch_index=batch_index,
                        location=location,
                    )
                continue

            kind_index = _one_hot_index(
                kind,
                reason_code="node_kind_not_one_hot",
                batch_index=batch_index,
                location=location,
            )
            if kind_index not in (KIND_CONSTRUCTOR, KIND_VARIABLE):
                _reject(
                    "active_empty_node",
                    batch_index=batch_index,
                    location=location,
                )
            type_index = _one_hot_index(
                type_pointer,
                reason_code="node_type_not_one_hot",
                batch_index=batch_index,
                location=location,
                active_mask=packets.type_active[batch_index],
            )

            if kind_index == KIND_VARIABLE:
                variable_index = _one_hot_index(
                    variable_pointer,
                    reason_code="variable_pointer_not_one_hot",
                    batch_index=batch_index,
                    location=location,
                    active_mask=packets.variable_active[batch_index],
                )
                if variable_index in variable_positions:
                    _reject(
                        "duplicate_graph_variable",
                        batch_index=batch_index,
                        location=location,
                    )
                variable_positions.add(variable_index)
                if not _is_zero(constructor_pointer):
                    _reject(
                        "variable_has_constructor",
                        batch_index=batch_index,
                        location=location,
                    )
                for argument in range(A):
                    pointer = _one_hot_index(
                        children[argument],
                        reason_code="child_pointer_not_one_hot",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )
                    if (
                        pointer != N
                        or bool(child_mask[argument].item())
                        or not _is_zero(child_types[argument])
                    ):
                        _reject(
                            "variable_has_child",
                            batch_index=batch_index,
                            location=f"{location},argument={argument}",
                        )
                continue

            constructor_index = _one_hot_index(
                constructor_pointer,
                reason_code="constructor_pointer_not_one_hot",
                batch_index=batch_index,
                location=location,
                active_mask=packets.constructor_active[batch_index],
            )
            if not _is_zero(variable_pointer):
                _reject(
                    "constructor_has_variable",
                    batch_index=batch_index,
                    location=location,
                )
            if not bool(
                packets.constructor_result_type[
                    batch_index,
                    constructor_index,
                    type_index,
                ].item()
            ):
                _reject(
                    "constructor_result_type_mismatch",
                    batch_index=batch_index,
                    location=location,
                )
            declared_arguments = packets.constructor_argument_mask[
                batch_index,
                constructor_index,
            ]
            if not torch.equal(child_mask, declared_arguments):
                _reject(
                    "constructor_arity_mismatch",
                    batch_index=batch_index,
                    location=location,
                )
            for argument in range(A):
                pointer = _one_hot_index(
                    children[argument],
                    reason_code="child_pointer_not_one_hot",
                    batch_index=batch_index,
                    location=f"{location},argument={argument}",
                )
                declared = bool(declared_arguments[argument].item())
                if not declared:
                    if pointer != N or not _is_zero(child_types[argument]):
                        _reject(
                            "undeclared_child_payload",
                            batch_index=batch_index,
                            location=f"{location},argument={argument}",
                        )
                    continue
                if pointer == N or not bool(active[pointer].item()):
                    _reject(
                        "child_not_active",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )
                declared_type = packets.constructor_argument_type[
                    batch_index,
                    constructor_index,
                    argument,
                ]
                if not torch.equal(
                    graph.graph_type[batch_index, pointer],
                    declared_type,
                ):
                    _reject(
                        "child_type_mismatch",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )
                if not torch.equal(child_types[argument], declared_type):
                    _reject(
                        "child_type_cache_mismatch",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )
                adjacency[storage].append(pointer)

        if root != N:
            visiting: set[int] = set()
            reached: set[int] = set()

            def visit(storage: int) -> None:
                if storage in visiting:
                    _reject(
                        "graph_cycle",
                        batch_index=batch_index,
                        location=f"storage={storage}",
                    )
                if storage in reached:
                    return
                visiting.add(storage)
                for child in adjacency[storage]:
                    visit(child)
                visiting.remove(storage)
                reached.add(storage)

            visit(root)
            active_positions = set(
                torch.nonzero(active, as_tuple=False).flatten().tolist()
            )
            if reached != active_positions:
                _reject(
                    "unreachable_active_node",
                    batch_index=batch_index,
                )
        elif input_graph and active_count:
            _reject("invalid_input_graph", batch_index=batch_index)


def _validate_transaction(
    packets: NeuralTcrrPacketTensors,
    transaction: NeuralTcrrGraphTransaction,
) -> None:
    if not isinstance(transaction, NeuralTcrrGraphTransaction):
        _reject("graph_transaction_required")
    batch_size = packets.graph_active.shape[0]
    device = packets.graph_active.device
    shapes = {
        "node_operation": (batch_size, N, NODE_OPERATION_COUNT),
        "root_pointer": (batch_size, N + 1),
        "node_kind": (batch_size, N, GRAPH_KIND_COUNT),
        "node_type_pointer": (batch_size, N, Y),
        "node_constructor_pointer": (batch_size, N, C),
        "node_variable_pointer": (batch_size, N, V),
        "child_pointer": (batch_size, N, A, N + 1),
        "child_presence": (batch_size, N, A, CHILD_PRESENCE_COUNT),
    }
    for name, shape in shapes.items():
        _require_tensor(
            getattr(transaction, name),
            name=name,
            shape=shape,
            dtype=torch.bool,
            device=device,
        )

    for batch_index in range(batch_size):
        _one_hot_index(
            transaction.root_pointer[batch_index],
            reason_code="transaction_root_not_one_hot",
            batch_index=batch_index,
            location="root",
        )
        for storage in range(N):
            location = f"storage={storage}"
            operation = _one_hot_index(
                transaction.node_operation[batch_index, storage],
                reason_code="operation_not_one_hot",
                batch_index=batch_index,
                location=location,
            )
            payload = (
                transaction.node_kind[batch_index, storage],
                transaction.node_type_pointer[batch_index, storage],
                transaction.node_constructor_pointer[batch_index, storage],
                transaction.node_variable_pointer[batch_index, storage],
                transaction.child_pointer[batch_index, storage],
                transaction.child_presence[batch_index, storage],
            )
            if operation != NODE_WRITE:
                if any(not _is_zero(value) for value in payload):
                    _reject(
                        "non_write_payload",
                        batch_index=batch_index,
                        location=location,
                    )
                if operation == NODE_CLEAR and not bool(
                    packets.graph_capacity[batch_index, storage].item()
                ):
                    _reject(
                        "clear_outside_capacity",
                        batch_index=batch_index,
                        location=location,
                    )
                continue

            if not bool(packets.graph_capacity[batch_index, storage].item()):
                _reject(
                    "write_outside_capacity",
                    batch_index=batch_index,
                    location=location,
                )
            kind = _one_hot_index(
                transaction.node_kind[batch_index, storage],
                reason_code="transaction_kind_not_one_hot",
                batch_index=batch_index,
                location=location,
            )
            if kind not in (KIND_CONSTRUCTOR, KIND_VARIABLE):
                _reject(
                    "write_empty_kind",
                    batch_index=batch_index,
                    location=location,
                )
            _one_hot_index(
                transaction.node_type_pointer[batch_index, storage],
                reason_code="transaction_type_not_one_hot",
                batch_index=batch_index,
                location=location,
                active_mask=packets.type_active[batch_index],
            )
            if kind == KIND_CONSTRUCTOR:
                _one_hot_index(
                    transaction.node_constructor_pointer[batch_index, storage],
                    reason_code="transaction_constructor_not_one_hot",
                    batch_index=batch_index,
                    location=location,
                    active_mask=packets.constructor_active[batch_index],
                )
                if not _is_zero(
                    transaction.node_variable_pointer[batch_index, storage]
                ):
                    _reject(
                        "constructor_write_has_variable",
                        batch_index=batch_index,
                        location=location,
                    )
            else:
                _one_hot_index(
                    transaction.node_variable_pointer[batch_index, storage],
                    reason_code="transaction_variable_not_one_hot",
                    batch_index=batch_index,
                    location=location,
                    active_mask=packets.variable_active[batch_index],
                )
                if not _is_zero(
                    transaction.node_constructor_pointer[batch_index, storage]
                ):
                    _reject(
                        "variable_write_has_constructor",
                        batch_index=batch_index,
                        location=location,
                    )
            for argument in range(A):
                presence = _one_hot_index(
                    transaction.child_presence[
                        batch_index,
                        storage,
                        argument,
                    ],
                    reason_code="child_presence_not_one_hot",
                    batch_index=batch_index,
                    location=f"{location},argument={argument}",
                )
                pointer = _one_hot_index(
                    transaction.child_pointer[
                        batch_index,
                        storage,
                        argument,
                    ],
                    reason_code="transaction_child_not_one_hot",
                    batch_index=batch_index,
                    location=f"{location},argument={argument}",
                )
                if presence == CHILD_ABSENT and pointer != N:
                    _reject(
                        "absent_child_not_null",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )
                if presence == CHILD_PRESENT and pointer == N:
                    _reject(
                        "present_child_is_null",
                        batch_index=batch_index,
                        location=f"{location},argument={argument}",
                    )


def _candidate_graph(
    packets: NeuralTcrrPacketTensors,
    transaction: NeuralTcrrGraphTransaction,
) -> NeuralTcrrGraphTensors:
    active = packets.graph_active.clone()
    root = transaction.root_pointer.clone()
    kind = packets.graph_kind.clone()
    type_pointer = packets.graph_type.clone()
    constructor_pointer = packets.graph_constructor.clone()
    variable_pointer = packets.graph_variable.clone()
    children = packets.graph_children.clone()
    child_type = packets.graph_child_type.clone()
    child_mask = packets.graph_child_mask.clone()
    capacity = packets.graph_capacity.clone()

    batch_size = packets.graph_active.shape[0]
    for batch_index in range(batch_size):
        for storage in range(N):
            operation = int(
                torch.nonzero(
                    transaction.node_operation[batch_index, storage],
                    as_tuple=False,
                )[0].item()
            )
            if operation == NODE_KEEP:
                continue
            if operation == NODE_CLEAR:
                active[batch_index, storage] = False
                kind[batch_index, storage].zero_()
                kind[batch_index, storage, KIND_EMPTY] = True
                type_pointer[batch_index, storage].zero_()
                constructor_pointer[batch_index, storage].zero_()
                variable_pointer[batch_index, storage].zero_()
                children[batch_index, storage].zero_()
                children[batch_index, storage, :, N] = True
                child_type[batch_index, storage].zero_()
                child_mask[batch_index, storage].zero_()
                continue

            active[batch_index, storage] = True
            kind[batch_index, storage] = transaction.node_kind[
                batch_index,
                storage,
            ]
            type_pointer[batch_index, storage] = transaction.node_type_pointer[
                batch_index,
                storage,
            ]
            constructor_pointer[batch_index, storage] = (
                transaction.node_constructor_pointer[batch_index, storage]
            )
            variable_pointer[batch_index, storage] = transaction.node_variable_pointer[
                batch_index, storage
            ]
            children[batch_index, storage] = transaction.child_pointer[
                batch_index,
                storage,
            ]
            child_mask[batch_index, storage] = transaction.child_presence[
                batch_index,
                storage,
                :,
                CHILD_PRESENT,
            ]
            child_type[batch_index, storage].zero_()

    for batch_index in range(batch_size):
        for storage in range(N):
            if not bool(active[batch_index, storage].item()):
                continue
            for argument in range(A):
                if not bool(child_mask[batch_index, storage, argument].item()):
                    continue
                pointer = int(
                    torch.nonzero(
                        children[batch_index, storage, argument],
                        as_tuple=False,
                    )[0].item()
                )
                if pointer < N and bool(active[batch_index, pointer].item()):
                    child_type[batch_index, storage, argument] = type_pointer[
                        batch_index,
                        pointer,
                    ]

    return NeuralTcrrGraphTensors(
        graph_active=active,
        graph_root=root,
        graph_kind=kind,
        graph_type=type_pointer,
        graph_constructor=constructor_pointer,
        graph_variable=variable_pointer,
        graph_children=children,
        graph_child_type=child_type,
        graph_child_mask=child_mask,
        graph_capacity=capacity,
    )


def commit_neural_tcrr_graph(
    packets: NeuralTcrrPacketTensors,
    transaction: NeuralTcrrGraphTransaction,
) -> NeuralTcrrGraphTensors:
    """Validate and atomically install one transaction batch."""

    _validate_static_boundary(packets)
    _validate_graph(packets, _graph_from_packets(packets), input_graph=True)
    _validate_transaction(packets, transaction)
    candidate = _candidate_graph(packets, transaction)
    _validate_graph(packets, candidate, input_graph=False)
    return candidate


def _delta_field(
    delta: NeuralTcrrGraphDeltaLike,
    *,
    logits_name: str,
    mask_name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    logits = _require_tensor(
        getattr(delta, logits_name, None),
        name=logits_name,
        shape=shape,
        dtype=None,
        device=device,
    )
    mask = _require_tensor(
        getattr(delta, mask_name, None),
        name=mask_name,
        shape=shape,
        dtype=torch.bool,
        device=device,
    )
    if not logits.dtype.is_floating_point:
        _reject("delta_logits_not_floating", location=logits_name)
    return logits, mask


def _masked_choice(
    logits: Tensor,
    mask: Tensor,
    *,
    batch_index: int,
    location: str,
) -> int:
    if not bool(mask.any().item()):
        _reject(
            "delta_has_no_choice",
            batch_index=batch_index,
            location=location,
        )
    allowed = logits[mask]
    if not bool(torch.isfinite(allowed).all().item()):
        _reject(
            "delta_choice_not_finite",
            batch_index=batch_index,
            location=location,
        )
    floor = torch.finfo(logits.dtype).min
    return int(torch.argmax(logits.masked_fill(~mask, floor)).item())


def decode_neural_tcrr_graph_delta(
    packets: NeuralTcrrPacketTensors,
    delta: NeuralTcrrGraphDeltaLike,
) -> NeuralTcrrGraphTransaction:
    """Select one masked argmax proposal per active transaction field."""

    batch_size = _validate_static_boundary(packets)
    device = packets.graph_active.device
    specs = {
        "operation": (
            "node_operation_logits",
            "node_operation_mask",
            (batch_size, N, NODE_OPERATION_COUNT),
        ),
        "root": (
            "root_pointer_logits",
            "root_pointer_mask",
            (batch_size, N + 1),
        ),
        "kind": (
            "node_kind_logits",
            "node_kind_mask",
            (batch_size, N, GRAPH_KIND_COUNT),
        ),
        "type": (
            "node_type_pointer_logits",
            "node_type_pointer_mask",
            (batch_size, N, Y),
        ),
        "constructor": (
            "node_constructor_pointer_logits",
            "node_constructor_pointer_mask",
            (batch_size, N, C),
        ),
        "variable": (
            "node_variable_pointer_logits",
            "node_variable_pointer_mask",
            (batch_size, N, V),
        ),
        "child": (
            "child_pointer_logits",
            "child_pointer_mask",
            (batch_size, N, A, N + 1),
        ),
        "presence": (
            "child_presence_logits",
            "child_presence_mask",
            (batch_size, N, A, CHILD_PRESENCE_COUNT),
        ),
    }
    fields = {
        name: _delta_field(
            delta,
            logits_name=logits_name,
            mask_name=mask_name,
            shape=shape,
            device=device,
        )
        for name, (logits_name, mask_name, shape) in specs.items()
    }

    node_operation = torch.zeros(
        (batch_size, N, NODE_OPERATION_COUNT),
        dtype=torch.bool,
        device=device,
    )
    root_pointer = torch.zeros(
        (batch_size, N + 1),
        dtype=torch.bool,
        device=device,
    )
    node_kind = torch.zeros(
        (batch_size, N, GRAPH_KIND_COUNT),
        dtype=torch.bool,
        device=device,
    )
    node_type_pointer = torch.zeros(
        (batch_size, N, Y),
        dtype=torch.bool,
        device=device,
    )
    node_constructor_pointer = torch.zeros(
        (batch_size, N, C),
        dtype=torch.bool,
        device=device,
    )
    node_variable_pointer = torch.zeros(
        (batch_size, N, V),
        dtype=torch.bool,
        device=device,
    )
    child_pointer = torch.zeros(
        (batch_size, N, A, N + 1),
        dtype=torch.bool,
        device=device,
    )
    child_presence = torch.zeros(
        (batch_size, N, A, CHILD_PRESENCE_COUNT),
        dtype=torch.bool,
        device=device,
    )

    for batch_index in range(batch_size):
        root_logits, root_mask = fields["root"]
        selected_root = _masked_choice(
            root_logits[batch_index],
            root_mask[batch_index],
            batch_index=batch_index,
            location="root",
        )
        root_pointer[batch_index, selected_root] = True
        for storage in range(N):
            operation_logits, operation_mask = fields["operation"]
            operation = _masked_choice(
                operation_logits[batch_index, storage],
                operation_mask[batch_index, storage],
                batch_index=batch_index,
                location=f"storage={storage},operation",
            )
            node_operation[batch_index, storage, operation] = True
            if operation != NODE_WRITE:
                continue

            kind_logits, kind_mask = fields["kind"]
            kind = _masked_choice(
                kind_logits[batch_index, storage],
                kind_mask[batch_index, storage],
                batch_index=batch_index,
                location=f"storage={storage},kind",
            )
            node_kind[batch_index, storage, kind] = True

            type_logits, type_mask = fields["type"]
            type_index = _masked_choice(
                type_logits[batch_index, storage],
                type_mask[batch_index, storage],
                batch_index=batch_index,
                location=f"storage={storage},type",
            )
            node_type_pointer[batch_index, storage, type_index] = True

            if kind == KIND_CONSTRUCTOR:
                constructor_logits, constructor_mask = fields["constructor"]
                constructor = _masked_choice(
                    constructor_logits[batch_index, storage],
                    constructor_mask[batch_index, storage],
                    batch_index=batch_index,
                    location=f"storage={storage},constructor",
                )
                node_constructor_pointer[
                    batch_index,
                    storage,
                    constructor,
                ] = True
            elif kind == KIND_VARIABLE:
                variable_logits, variable_mask = fields["variable"]
                variable = _masked_choice(
                    variable_logits[batch_index, storage],
                    variable_mask[batch_index, storage],
                    batch_index=batch_index,
                    location=f"storage={storage},variable",
                )
                node_variable_pointer[batch_index, storage, variable] = True

            for argument in range(A):
                presence_logits, presence_mask = fields["presence"]
                presence = _masked_choice(
                    presence_logits[batch_index, storage, argument],
                    presence_mask[batch_index, storage, argument],
                    batch_index=batch_index,
                    location=f"storage={storage},argument={argument},presence",
                )
                child_presence[
                    batch_index,
                    storage,
                    argument,
                    presence,
                ] = True
                if presence == CHILD_ABSENT:
                    child_pointer[
                        batch_index,
                        storage,
                        argument,
                        N,
                    ] = True
                    continue
                pointer_logits, pointer_mask = fields["child"]
                pointer = _masked_choice(
                    pointer_logits[batch_index, storage, argument],
                    pointer_mask[batch_index, storage, argument],
                    batch_index=batch_index,
                    location=f"storage={storage},argument={argument},pointer",
                )
                child_pointer[
                    batch_index,
                    storage,
                    argument,
                    pointer,
                ] = True

    return NeuralTcrrGraphTransaction(
        node_operation=node_operation,
        root_pointer=root_pointer,
        node_kind=node_kind,
        node_type_pointer=node_type_pointer,
        node_constructor_pointer=node_constructor_pointer,
        node_variable_pointer=node_variable_pointer,
        child_pointer=child_pointer,
        child_presence=child_presence,
    )


def commit_neural_tcrr_graph_delta(
    packets: NeuralTcrrPacketTensors,
    delta: NeuralTcrrGraphDeltaLike,
) -> NeuralTcrrGraphTensors:
    """Decode one proposal and commit it without fallback selection."""

    return commit_neural_tcrr_graph(
        packets,
        decode_neural_tcrr_graph_delta(packets, delta),
    )


__all__ = [
    "CHILD_ABSENT",
    "CHILD_PRESENT",
    "KIND_CONSTRUCTOR",
    "KIND_EMPTY",
    "KIND_VARIABLE",
    "NODE_CLEAR",
    "NODE_KEEP",
    "NODE_WRITE",
    "NeuralTcrrCommitError",
    "NeuralTcrrGraphDeltaLike",
    "NeuralTcrrGraphTensors",
    "NeuralTcrrGraphTransaction",
    "commit_neural_tcrr_graph",
    "commit_neural_tcrr_graph_delta",
    "decode_neural_tcrr_graph_delta",
]
