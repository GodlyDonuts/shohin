from __future__ import annotations

import torch

from autocatalytic_hysteretic_relation_field import (
    AutocatalyticHystereticRelationField,
)
from contextualize_bekic_program import contextualize_simultaneous_packet
from contrastive_bekic_program_orbits import (
    generate_orbit,
    select_machine_input,
)
from tensorize_contextual_ahrf import (
    AHRF_NODE_FEATURE_DIM,
    tensorized_contextual_to_ahrf,
)
from tensorize_contextual_bekic import tensorize_contextual_packets


def test_contextual_packet_enters_source_deleted_ahrf_contract() -> None:
    row = generate_orbit(split="train", seed=2026072332)
    source = select_machine_input(
        row,
        arm="p",
        form="simultaneous",
    )
    contextual = contextualize_simultaneous_packet(
        source,
        seed=2026072333,
    )
    tensors = tensorize_contextual_packets([contextual])
    graph = tensorized_contextual_to_ahrf(tensors)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=AHRF_NODE_FEATURE_DIM,
        hidden_dim=16,
        card_rounds=1,
        max_steps=2,
    )
    rollout = model(graph)
    assert rollout.terminal_facts.shape == graph.seed_facts.shape
    assert torch.equal(
        graph.node_card_mask.any(1),
        graph.witness_mask.any(-1),
    )
    assert not graph.node_features[
        ~graph.node_mask
    ].any()


def test_node_features_contain_only_structural_kind() -> None:
    row = generate_orbit(split="train", seed=2026072334)
    source = select_machine_input(
        row,
        arm="p_prime",
        form="simultaneous",
    )
    contextual = contextualize_simultaneous_packet(
        source,
        seed=2026072335,
    )
    graph = tensorized_contextual_to_ahrf(
        tensorize_contextual_packets([contextual])
    )
    active = graph.node_features[graph.node_mask]
    assert active.shape[-1] == AHRF_NODE_FEATURE_DIM
    assert torch.equal(active.sum(-1), torch.ones(active.shape[0]))
