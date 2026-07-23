"""Adapt private contextual Bekić tensors to the AHRF graph contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from autocatalytic_hysteretic_relation_field import (
    FEEDBACK_ROLE,
    GRAPH_EDGE_ROLES,
    SourceDeletedRelationGraph,
)
from contextual_bekic_graph_machine import ProgramNodeKind
from tensorize_contextual_bekic import TensorizedContextualProgram


AHRF_NODE_FEATURE_DIM = 3


class ContextualAHRFTensorizationError(ValueError):
    """Raised when a private contextual graph cannot enter AHRF."""


def _gather_constants(
    tensors: TensorizedContextualProgram,
) -> torch.Tensor:
    packet = tensors.packet
    objects = packet.constants.shape[-1]
    indices = packet.constant_index.clamp_min(0)
    gathered = packet.constants.gather(
        1,
        indices[..., None, None].expand(
            -1,
            -1,
            objects,
            objects,
        ),
    )
    is_constant = packet.node_kind.eq(
        int(ProgramNodeKind.CONSTANT)
    )
    return gathered * is_constant[..., None, None].to(gathered.dtype)


def tensorized_contextual_to_ahrf(
    tensors: TensorizedContextualProgram,
) -> SourceDeletedRelationGraph:
    """Create a source-deleted AHRF graph without semantic identifiers."""

    packet = tensors.packet
    batch, nodes = packet.node_kind.shape
    if packet.node_valid.shape != (batch, nodes):
        raise ContextualAHRFTensorizationError("private node geometry differs")
    node_features = F.one_hot(
        packet.node_kind.clamp_min(0),
        AHRF_NODE_FEATURE_DIM,
    ).to(packet.constants.dtype)
    node_features = (
        node_features
        * packet.node_valid[..., None].to(node_features.dtype)
    )
    argument_edges = torch.zeros(
        batch,
        nodes,
        nodes,
        GRAPH_EDGE_ROLES,
        dtype=torch.bool,
        device=packet.constants.device,
    )
    node_card_mask = torch.zeros(
        batch,
        nodes,
        tensors.witness_mask.shape[1],
        dtype=torch.bool,
        device=packet.constants.device,
    )
    root_mask = torch.zeros(
        batch,
        nodes,
        dtype=torch.bool,
        device=packet.constants.device,
    )
    for batch_index in range(batch):
        for node_index in range(nodes):
            if not bool(packet.node_valid[batch_index, node_index]):
                continue
            if packet.node_kind[batch_index, node_index].item() != int(
                ProgramNodeKind.OPERATION
            ):
                continue
            slot = int(packet.operation_slot[batch_index, node_index])
            if slot < 0:
                raise ContextualAHRFTensorizationError(
                    "operation node has no opaque card"
                )
            node_card_mask[batch_index, node_index, slot] = True
            for role, indices in enumerate(
                (packet.left_index, packet.right_index)
            ):
                child = int(indices[batch_index, node_index])
                if child >= 0:
                    argument_edges[
                        batch_index,
                        node_index,
                        child,
                        role,
                    ] = True
        roots = packet.equation_root[batch_index]
        if bool(roots.lt(0).any()):
            raise ContextualAHRFTensorizationError(
                "equation root is absent"
            )
        root_mask[batch_index, roots] = True
        for node_index in range(nodes):
            if (
                not bool(packet.node_valid[batch_index, node_index])
                or packet.node_kind[batch_index, node_index].item()
                != int(ProgramNodeKind.VARIABLE)
            ):
                continue
            variable = int(
                packet.variable_index[batch_index, node_index]
            )
            if variable < 0:
                raise ContextualAHRFTensorizationError(
                    "variable node has no feedback binding"
                )
            argument_edges[
                batch_index,
                node_index,
                int(roots[variable]),
                FEEDBACK_ROLE,
            ] = True

    return SourceDeletedRelationGraph(
        node_features=node_features,
        node_mask=packet.node_valid,
        argument_edges=argument_edges,
        node_card_mask=node_card_mask,
        root_mask=root_mask,
        seed_facts=_gather_constants(tensors),
        witness_left=tensors.witness_left,
        witness_right=tensors.witness_right,
        witness_output=tensors.witness_output,
        witness_mask=tensors.witness_mask,
        argument_mask=tensors.argument_mask,
        object_mask=tensors.object_mask,
    )
