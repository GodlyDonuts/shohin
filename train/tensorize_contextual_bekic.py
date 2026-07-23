"""Tensorize source-deleted contextual Bekić packets for private execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from contextual_bekic_graph_machine import (
    MAX_OPERATION_SLOTS,
    MAX_PROGRAM_CONSTANTS,
    MAX_PROGRAM_NODES,
    PROGRAM_VARIABLES,
    DeletedContextualProgramPacket,
    ProgramNodeKind,
)
from contextualize_bekic_program import (
    CARD_WITNESSES,
    ContextualizationError,
    validate_contextual_packet_structure,
)
from equivariant_relation_register_machine import MAX_OBJECTS


class ContextualTensorizationError(ValueError):
    """Raised when a validated private packet cannot be tensorized."""


@dataclass(frozen=True, slots=True)
class TensorizedContextualProgram:
    packet: DeletedContextualProgramPacket
    witness_left: torch.Tensor
    witness_right: torch.Tensor
    witness_output: torch.Tensor
    witness_mask: torch.Tensor
    argument_mask: torch.Tensor
    object_mask: torch.Tensor


def _copy_relation(
    destination: torch.Tensor,
    value: object,
    cardinality: int,
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != cardinality
        or any(
            not isinstance(row, list)
            or len(row) != cardinality
            or any(bit not in (0, 1) for bit in row)
            for row in value
        )
    ):
        raise ContextualTensorizationError("serialized relation differs")
    destination[:cardinality, :cardinality] = torch.tensor(
        value,
        dtype=destination.dtype,
        device=destination.device,
    )


def tensorize_contextual_packets(
    packets: list[dict[str, object]],
    *,
    device: torch.device | str = "cpu",
) -> TensorizedContextualProgram:
    if not packets:
        raise ContextualTensorizationError("contextual packet batch is empty")
    for packet in packets:
        try:
            validate_contextual_packet_structure(packet)
        except ContextualizationError as error:
            raise ContextualTensorizationError(
                "contextual packet failed validation"
            ) from error
    batch = len(packets)
    device = torch.device(device)
    cardinality = torch.empty(batch, dtype=torch.uint8, device=device)
    constants = torch.zeros(
        batch,
        MAX_PROGRAM_CONSTANTS,
        MAX_OBJECTS,
        MAX_OBJECTS,
        device=device,
    )
    constant_valid = torch.zeros(
        batch,
        MAX_PROGRAM_CONSTANTS,
        dtype=torch.bool,
        device=device,
    )
    node_valid = torch.zeros(
        batch,
        MAX_PROGRAM_NODES,
        dtype=torch.bool,
        device=device,
    )
    node_kind = torch.full(
        (batch, MAX_PROGRAM_NODES),
        -1,
        dtype=torch.long,
        device=device,
    )
    constant_index = torch.full_like(node_kind, -1)
    variable_index = torch.full_like(node_kind, -1)
    operation_slot = torch.full_like(node_kind, -1)
    left_index = torch.full_like(node_kind, -1)
    right_index = torch.full_like(node_kind, -1)
    equation_root = torch.full(
        (batch, PROGRAM_VARIABLES),
        -1,
        dtype=torch.long,
        device=device,
    )
    slot_arity = torch.full(
        (batch, MAX_OPERATION_SLOTS),
        -1,
        dtype=torch.long,
        device=device,
    )
    witness_shape = (
        batch,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    witness_left = torch.zeros(witness_shape, device=device)
    witness_right = torch.zeros_like(witness_left)
    witness_output = torch.zeros_like(witness_left)
    witness_mask = torch.zeros(
        batch,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        dtype=torch.bool,
        device=device,
    )
    argument_mask = torch.zeros(
        batch,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        2,
        dtype=torch.bool,
        device=device,
    )
    object_mask = torch.zeros(
        batch,
        MAX_OBJECTS,
        dtype=torch.bool,
        device=device,
    )

    for batch_index, packet in enumerate(packets):
        size = int(packet["cardinality"])
        cardinality[batch_index] = size
        object_mask[batch_index, :size] = True
        constant_items = packet["constants"]
        node_items = packet["nodes"]
        cards = packet["operation_cards"]
        if (
            len(constant_items) > MAX_PROGRAM_CONSTANTS
            or len(node_items) > MAX_PROGRAM_NODES
            or len(cards) > MAX_OPERATION_SLOTS
        ):
            raise ContextualTensorizationError(
                "contextual packet exceeds tensor-machine capacity"
            )
        constant_lookup: dict[str, int] = {}
        for index, item in enumerate(constant_items):
            constant_lookup[str(item["id"])] = index
            constant_valid[batch_index, index] = True
            _copy_relation(
                constants[batch_index, index],
                item["relation"],
                size,
            )
        variable_lookup = {
            str(variable): index
            for index, variable in enumerate(packet["variables"])
        }
        node_lookup = {
            str(node["id"]): index
            for index, node in enumerate(node_items)
        }
        slot_lookup = {
            str(card["slot"]): index
            for index, card in enumerate(cards)
        }
        for slot, card in enumerate(cards):
            arity = int(card["arity"])
            slot_arity[batch_index, slot] = arity
            witness_mask[batch_index, slot] = True
            argument_mask[batch_index, slot, :, :arity] = True
            for witness_index, witness in enumerate(card["witnesses"]):
                _copy_relation(
                    witness_left[batch_index, slot, witness_index],
                    witness["left"],
                    size,
                )
                _copy_relation(
                    witness_right[batch_index, slot, witness_index],
                    witness["right"],
                    size,
                )
                _copy_relation(
                    witness_output[batch_index, slot, witness_index],
                    witness["output"],
                    size,
                )
        for index, node in enumerate(node_items):
            node_valid[batch_index, index] = True
            kind = str(node["kind"])
            if kind == "CONSTANT":
                node_kind[batch_index, index] = int(
                    ProgramNodeKind.CONSTANT
                )
                constant_index[batch_index, index] = constant_lookup[
                    str(node["constant"])
                ]
            elif kind == "VARIABLE":
                node_kind[batch_index, index] = int(
                    ProgramNodeKind.VARIABLE
                )
                variable_index[batch_index, index] = variable_lookup[
                    str(node["variable"])
                ]
            elif kind == "OPERATION":
                node_kind[batch_index, index] = int(
                    ProgramNodeKind.OPERATION
                )
                operation_slot[batch_index, index] = slot_lookup[
                    str(node["slot"])
                ]
                references = [
                    node_lookup[str(reference)]
                    for reference in node["inputs"]
                ]
                if references:
                    left_index[batch_index, index] = references[0]
                if len(references) == 2:
                    right_index[batch_index, index] = references[1]
            else:
                raise ContextualTensorizationError(
                    "contextual node kind differs"
                )
        for equation in packet["equations"]:
            variable = variable_lookup[str(equation["variable"])]
            equation_root[batch_index, variable] = node_lookup[
                str(equation["root"])
            ]

    private_packet = DeletedContextualProgramPacket(
        cardinality=cardinality,
        constants=constants,
        constant_valid=constant_valid,
        node_valid=node_valid,
        node_kind=node_kind,
        constant_index=constant_index,
        variable_index=variable_index,
        operation_slot=operation_slot,
        left_index=left_index,
        right_index=right_index,
        equation_root=equation_root,
        slot_arity=slot_arity,
    )
    return TensorizedContextualProgram(
        packet=private_packet,
        witness_left=witness_left,
        witness_right=witness_right,
        witness_output=witness_output,
        witness_mask=witness_mask,
        argument_mask=argument_mask,
        object_mask=object_mask,
    )


def tensorize_target_environment(
    environment: dict[str, Any],
    variables: list[str],
    *,
    cardinality: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if len(variables) != PROGRAM_VARIABLES or set(environment) != set(
        variables
    ):
        raise ContextualTensorizationError(
            "target environment/variables differ"
        )
    output = torch.zeros(
        PROGRAM_VARIABLES,
        MAX_OBJECTS,
        MAX_OBJECTS,
        device=torch.device(device),
    )
    for index, variable in enumerate(variables):
        relation = environment[variable]
        _copy_relation(
            output[index],
            [list(row) for row in relation],
            cardinality,
        )
    return output
