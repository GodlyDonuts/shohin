"""Bounded neural tensor boundary for endogenous congruence packets.

Opaque identifiers define episode-local coordinates only. They remain in
offline receipts and are never converted into numeric model features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from pipeline.endogenous_congruence_board import (
    MAX_GENERATORS,
    MAX_QUERY_PORTS,
    MAX_RECORDS,
    CongruenceBoardError,
    EndogenousCongruencePacket,
    ObservationWitness,
    TransitionWitness,
    validate_packet,
)


N = MAX_RECORDS
G = MAX_GENERATORS
Q = MAX_QUERY_PORTS
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class EndogenousCongruenceTensorError(ValueError):
    """A packet or tensor batch cannot cross the frozen boundary."""


@dataclass(frozen=True)
class EndogenousCongruenceAxisReceipt:
    """Offline coordinate names for exact physical-fact reconstruction."""

    record_ids: tuple[str, ...]
    generator_ids: tuple[str, ...]
    query_ids: tuple[str, ...]


@dataclass(frozen=True)
class EndogenousCongruenceTensors:
    """The complete batched model-visible tensor payload."""

    record_mask: Tensor
    generator_mask: Tensor
    query_mask: Tensor

    record_equal: Tensor
    generator_equal: Tensor
    query_equal: Tensor

    transition_mask: Tensor
    transition_target: Tensor

    observation_mask: Tensor
    observation_value: Tensor
    observation_equal: Tensor


@dataclass(frozen=True)
class TensorizedEndogenousCongruencePackets:
    """Model tensors plus offline episode-local coordinate receipts."""

    tensors: EndogenousCongruenceTensors
    receipts: tuple[EndogenousCongruenceAxisReceipt, ...]


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _prefix_mask(
    count: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    return torch.arange(width, device=device) < count


def _active_identity(
    count: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    output = _zeros((width, width), dtype=torch.bool, device=device)
    if count:
        output[:count, :count] = torch.eye(
            count,
            dtype=torch.bool,
            device=device,
        )
    return output


def _new_tensors(
    batch_size: int,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    boolean = torch.bool
    return {
        "record_mask": _zeros((batch_size, N), dtype=boolean, device=device),
        "generator_mask": _zeros((batch_size, G), dtype=boolean, device=device),
        "query_mask": _zeros((batch_size, Q), dtype=boolean, device=device),
        "record_equal": _zeros((batch_size, N, N), dtype=boolean, device=device),
        "generator_equal": _zeros((batch_size, G, G), dtype=boolean, device=device),
        "query_equal": _zeros((batch_size, Q, Q), dtype=boolean, device=device),
        "transition_mask": _zeros((batch_size, N, G), dtype=boolean, device=device),
        "transition_target": _zeros(
            (batch_size, N, G, N),
            dtype=boolean,
            device=device,
        ),
        "observation_mask": _zeros(
            (batch_size, N, Q),
            dtype=boolean,
            device=device,
        ),
        "observation_value": _zeros(
            (batch_size, N, Q),
            dtype=torch.int64,
            device=device,
        ),
        "observation_equal": _zeros(
            (batch_size, N, Q, N, Q),
            dtype=boolean,
            device=device,
        ),
    }


def _require_int64(value: int) -> None:
    if value < INT64_MIN or value > INT64_MAX:
        raise EndogenousCongruenceTensorError(
            "observation value exceeds exact signed-int64 representation"
        )


def tensorize_endogenous_congruence_packets(
    packets: Sequence[EndogenousCongruencePacket],
    *,
    device: torch.device | str = "cpu",
) -> TensorizedEndogenousCongruencePackets:
    """Encode complete physical witness packets into fixed-capacity tensors."""

    if not packets:
        raise EndogenousCongruenceTensorError("at least one packet is required")
    resolved_device = torch.device(device)
    output = _new_tensors(len(packets), device=resolved_device)
    receipts: list[EndogenousCongruenceAxisReceipt] = []

    for batch_index, packet in enumerate(packets):
        if type(packet) is not EndogenousCongruencePacket:
            raise EndogenousCongruenceTensorError(
                "only exact EndogenousCongruencePacket objects are accepted"
            )
        try:
            tables = validate_packet(packet)
        except CongruenceBoardError as error:
            raise EndogenousCongruenceTensorError(str(error)) from error

        record_count = len(packet.records)
        generator_count = len(packet.generators)
        query_count = len(packet.query_ports)
        if record_count > N or generator_count > G or query_count > Q:
            raise EndogenousCongruenceTensorError("packet exceeds frozen geometry")

        output["record_mask"][batch_index, :record_count] = True
        output["generator_mask"][batch_index, :generator_count] = True
        output["query_mask"][batch_index, :query_count] = True
        output["record_equal"][batch_index] = _active_identity(
            record_count,
            N,
            device=resolved_device,
        )
        output["generator_equal"][batch_index] = _active_identity(
            generator_count,
            G,
            device=resolved_device,
        )
        output["query_equal"][batch_index] = _active_identity(
            query_count,
            Q,
            device=resolved_device,
        )
        output["transition_mask"][
            batch_index,
            :record_count,
            :generator_count,
        ] = True
        output["observation_mask"][
            batch_index,
            :record_count,
            :query_count,
        ] = True

        for source_index, source in enumerate(packet.records):
            for generator_index, generator in enumerate(packet.generators):
                target = tables.transition[(source, generator)]
                target_index = tables.record_index[target]
                output["transition_target"][
                    batch_index,
                    source_index,
                    generator_index,
                    target_index,
                ] = True

        for record_index, record in enumerate(packet.records):
            for query_index, query in enumerate(packet.query_ports):
                value = tables.observation[(record, query)]
                _require_int64(value)
                output["observation_value"][
                    batch_index,
                    record_index,
                    query_index,
                ] = value

        active_values = output["observation_value"][
            batch_index,
            :record_count,
            :query_count,
        ]
        output["observation_equal"][
            batch_index,
            :record_count,
            :query_count,
            :record_count,
            :query_count,
        ] = active_values[:, :, None, None] == active_values[None, None, :, :]
        receipts.append(
            EndogenousCongruenceAxisReceipt(
                record_ids=packet.records,
                generator_ids=packet.generators,
                query_ids=packet.query_ports,
            )
        )

    return TensorizedEndogenousCongruencePackets(
        tensors=EndogenousCongruenceTensors(**output),
        receipts=tuple(receipts),
    )


_EXPECTED_TAIL_SHAPES = {
    "record_mask": (N,),
    "generator_mask": (G,),
    "query_mask": (Q,),
    "record_equal": (N, N),
    "generator_equal": (G, G),
    "query_equal": (Q, Q),
    "transition_mask": (N, G),
    "transition_target": (N, G, N),
    "observation_mask": (N, Q),
    "observation_value": (N, Q),
    "observation_equal": (N, Q, N, Q),
}


def _validate_geometry(value: TensorizedEndogenousCongruencePackets) -> int:
    if type(value) is not TensorizedEndogenousCongruencePackets:
        raise EndogenousCongruenceTensorError("invalid tensorized packet container")
    batch_size = len(value.receipts)
    if batch_size == 0:
        raise EndogenousCongruenceTensorError("tensor batch is empty")
    for field, tail_shape in _EXPECTED_TAIL_SHAPES.items():
        tensor = getattr(value.tensors, field)
        if tuple(tensor.shape) != (batch_size, *tail_shape):
            raise EndogenousCongruenceTensorError(
                f"{field} has shape {tuple(tensor.shape)}, expected "
                f"{(batch_size, *tail_shape)}"
            )
        expected_dtype = torch.int64 if field == "observation_value" else torch.bool
        if tensor.dtype != expected_dtype:
            raise EndogenousCongruenceTensorError(
                f"{field} has dtype {tensor.dtype}, expected {expected_dtype}"
            )
    return batch_size


def _require_equal(actual: Tensor, expected: Tensor, location: str) -> None:
    if not torch.equal(actual, expected):
        raise EndogenousCongruenceTensorError(f"{location} is inconsistent")


def detensorize_endogenous_congruence_packet(
    value: TensorizedEndogenousCongruencePackets,
    batch_index: int,
) -> EndogenousCongruencePacket:
    """Reconstruct only the complete physical witness packet."""

    batch_size = _validate_geometry(value)
    if not 0 <= batch_index < batch_size:
        raise IndexError(batch_index)
    tensors = value.tensors
    receipt = value.receipts[batch_index]
    record_count = len(receipt.record_ids)
    generator_count = len(receipt.generator_ids)
    query_count = len(receipt.query_ids)
    if (
        not 2 <= record_count <= N
        or not 1 <= generator_count <= G
        or not 1 <= query_count <= Q
        or len(set(receipt.record_ids)) != record_count
        or len(set(receipt.generator_ids)) != generator_count
        or len(set(receipt.query_ids)) != query_count
    ):
        raise EndogenousCongruenceTensorError("axis receipt violates frozen bounds")

    device = tensors.record_mask.device
    record_mask = _prefix_mask(record_count, N, device=device)
    generator_mask = _prefix_mask(generator_count, G, device=device)
    query_mask = _prefix_mask(query_count, Q, device=device)
    _require_equal(tensors.record_mask[batch_index], record_mask, "record mask")
    _require_equal(
        tensors.generator_mask[batch_index],
        generator_mask,
        "generator mask",
    )
    _require_equal(tensors.query_mask[batch_index], query_mask, "query mask")
    _require_equal(
        tensors.record_equal[batch_index],
        _active_identity(record_count, N, device=device),
        "record equality",
    )
    _require_equal(
        tensors.generator_equal[batch_index],
        _active_identity(generator_count, G, device=device),
        "generator equality",
    )
    _require_equal(
        tensors.query_equal[batch_index],
        _active_identity(query_count, Q, device=device),
        "query equality",
    )

    expected_transition_mask = record_mask[:, None] & generator_mask[None, :]
    expected_observation_mask = record_mask[:, None] & query_mask[None, :]
    _require_equal(
        tensors.transition_mask[batch_index],
        expected_transition_mask,
        "transition mask",
    )
    _require_equal(
        tensors.observation_mask[batch_index],
        expected_observation_mask,
        "observation mask",
    )

    transitions: list[TransitionWitness] = []
    for source_index, source in enumerate(receipt.record_ids):
        for generator_index, generator in enumerate(receipt.generator_ids):
            target_row = tensors.transition_target[
                batch_index,
                source_index,
                generator_index,
            ]
            target_positions = torch.nonzero(target_row, as_tuple=False).flatten()
            if (
                target_positions.numel() != 1
                or int(target_positions[0]) >= record_count
            ):
                raise EndogenousCongruenceTensorError(
                    "active transition target is not one-hot in active records"
                )
            transitions.append(
                TransitionWitness(
                    source=source,
                    generator=generator,
                    target=receipt.record_ids[int(target_positions[0])],
                )
            )
    active_transition_block = tensors.transition_target[
        batch_index,
        :record_count,
        :generator_count,
        :record_count,
    ]
    expected_transition = _zeros(
        (N, G, N),
        dtype=torch.bool,
        device=device,
    )
    expected_transition[
        :record_count,
        :generator_count,
        :record_count,
    ] = active_transition_block
    _require_equal(
        tensors.transition_target[batch_index],
        expected_transition,
        "transition padding",
    )

    observations: list[ObservationWitness] = []
    for record_index, record in enumerate(receipt.record_ids):
        for query_index, query in enumerate(receipt.query_ids):
            observations.append(
                ObservationWitness(
                    record=record,
                    query_port=query,
                    value=int(
                        tensors.observation_value[
                            batch_index,
                            record_index,
                            query_index,
                        ].item()
                    ),
                )
            )
    expected_values = _zeros((N, Q), dtype=torch.int64, device=device)
    expected_values[:record_count, :query_count] = tensors.observation_value[
        batch_index,
        :record_count,
        :query_count,
    ]
    _require_equal(
        tensors.observation_value[batch_index],
        expected_values,
        "observation padding",
    )
    expected_observation_equal = _zeros(
        (N, Q, N, Q),
        dtype=torch.bool,
        device=device,
    )
    active_values = expected_values[:record_count, :query_count]
    expected_observation_equal[
        :record_count,
        :query_count,
        :record_count,
        :query_count,
    ] = active_values[:, :, None, None] == active_values[None, None, :, :]
    _require_equal(
        tensors.observation_equal[batch_index],
        expected_observation_equal,
        "observation equality",
    )

    packet = EndogenousCongruencePacket(
        records=receipt.record_ids,
        generators=receipt.generator_ids,
        query_ports=receipt.query_ids,
        transition_witnesses=tuple(transitions),
        observation_witnesses=tuple(observations),
    )
    try:
        validate_packet(packet)
    except CongruenceBoardError as error:
        raise EndogenousCongruenceTensorError(str(error)) from error
    return packet


__all__ = [
    "G",
    "N",
    "Q",
    "EndogenousCongruenceAxisReceipt",
    "EndogenousCongruenceTensorError",
    "EndogenousCongruenceTensors",
    "TensorizedEndogenousCongruencePackets",
    "detensorize_endogenous_congruence_packet",
    "tensorize_endogenous_congruence_packets",
]
