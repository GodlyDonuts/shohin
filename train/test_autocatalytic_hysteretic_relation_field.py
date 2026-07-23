from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from autocatalytic_hysteretic_relation_field import (
    GRAPH_EDGE_ROLES,
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
    AHRFError,
    AutocatalyticHystereticRelationField,
    SourceDeletedRelationGraph,
)


NODE_FEATURES = 4


def _graph() -> SourceDeletedRelationGraph:
    generator = torch.Generator().manual_seed(2026072301)
    batch, nodes, slots, witnesses, objects = 1, 4, 2, 3, 4
    node_mask = torch.tensor([[True, True, True, False]])
    object_mask = torch.tensor([[True, True, True, False]])
    node_features = torch.zeros(batch, nodes, NODE_FEATURES)
    node_features[0, 0, 0] = 1.0
    node_features[0, 1, 2] = 1.0
    node_features[0, 2, 2:] = 1.0
    root_mask = torch.tensor([[False, False, True, False]])
    argument_edges = torch.zeros(
        batch,
        nodes,
        nodes,
        GRAPH_EDGE_ROLES,
        dtype=torch.bool,
    )
    argument_edges[0, 1, 0, 0] = True
    argument_edges[0, 2, 1, 0] = True
    argument_edges[0, 2, 0, 1] = True
    node_card_mask = torch.zeros(
        batch,
        nodes,
        slots,
        dtype=torch.bool,
    )
    node_card_mask[0, 1, 0] = True
    node_card_mask[0, 2, 1] = True
    seed_facts = torch.zeros(batch, nodes, objects, objects)
    seed_facts[0, 0, :3, :3] = torch.eye(3)

    shape = (batch, slots, witnesses, objects, objects)
    witness_left = torch.zeros(shape)
    witness_right = torch.zeros(shape)
    witness_output = torch.zeros(shape)
    witness_mask = torch.zeros(
        batch,
        slots,
        witnesses,
        dtype=torch.bool,
    )
    witness_mask[:, :, :2] = True
    argument_mask = torch.zeros(
        batch,
        slots,
        witnesses,
        2,
        dtype=torch.bool,
    )
    argument_mask[:, 0, :2, 0] = True
    argument_mask[:, 1, :2] = True
    for slot in range(slots):
        for witness in range(2):
            witness_left[0, slot, witness, :3, :3] = torch.randint(
                0,
                2,
                (3, 3),
                generator=generator,
            )
            witness_output[0, slot, witness, :3, :3] = torch.randint(
                0,
                2,
                (3, 3),
                generator=generator,
            )
            if slot == 1:
                witness_right[0, slot, witness, :3, :3] = torch.randint(
                    0,
                    2,
                    (3, 3),
                    generator=generator,
                )
    return SourceDeletedRelationGraph(
        node_features=node_features,
        node_mask=node_mask,
        argument_edges=argument_edges,
        node_card_mask=node_card_mask,
        root_mask=root_mask,
        seed_facts=seed_facts,
        witness_left=witness_left,
        witness_right=witness_right,
        witness_output=witness_output,
        witness_mask=witness_mask,
        argument_mask=argument_mask,
        object_mask=object_mask,
    )


def _permute_nodes(
    graph: SourceDeletedRelationGraph,
    permutation: torch.Tensor,
) -> SourceDeletedRelationGraph:
    return replace(
        graph,
        node_features=graph.node_features[:, permutation],
        node_mask=graph.node_mask[:, permutation],
        argument_edges=graph.argument_edges[
            :,
            permutation,
        ][:, :, permutation],
        node_card_mask=graph.node_card_mask[:, permutation],
        root_mask=graph.root_mask[:, permutation],
        seed_facts=graph.seed_facts[:, permutation],
    )


def _permute_objects(
    value: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    return value.index_select(-2, permutation).index_select(-1, permutation)


def _permute_membrane_objects(
    value: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    return value.index_select(-3, permutation).index_select(-2, permutation)


def _object_permuted_graph(
    graph: SourceDeletedRelationGraph,
    permutation: torch.Tensor,
) -> SourceDeletedRelationGraph:
    return replace(
        graph,
        seed_facts=_permute_objects(graph.seed_facts, permutation),
        witness_left=_permute_objects(graph.witness_left, permutation),
        witness_right=_permute_objects(graph.witness_right, permutation),
        witness_output=_permute_objects(graph.witness_output, permutation),
        object_mask=graph.object_mask[:, permutation],
    )


def _disable_discrete_events(
    model: AutocatalyticHystereticRelationField,
) -> None:
    with torch.no_grad():
        model.write_head.weight.zero_()
        model.write_head.bias.fill_(-20.0)
        model.evidence_head.weight.zero_()
        model.evidence_head.bias.fill_(-20.0)
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(-20.0)


def test_parameter_receipt_is_exact_and_below_system_cap() -> None:
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=24,
        card_rounds=1,
        max_steps=3,
    )
    receipt = model.parameter_receipt()
    exact = sum(parameter.numel() for parameter in model.parameters())
    assert receipt.protected_base == PROTECTED_BASE_PARAMETERS
    assert receipt.ahrf_added == exact
    assert sum(count for _, count in receipt.components) == exact
    assert receipt.complete_system == PROTECTED_BASE_PARAMETERS + exact
    assert receipt.complete_system < SYSTEM_PARAMETER_CAP
    assert receipt.headroom == SYSTEM_PARAMETER_CAP - receipt.complete_system
    with pytest.raises(AHRFError, match="parameter cap"):
        model.parameter_receipt(
            protected_base=SYSTEM_PARAMETER_CAP - exact,
        )


def test_object_and_node_permutations_are_equivariant() -> None:
    torch.manual_seed(2026072302)
    graph = _graph()
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=3,
    )
    _disable_discrete_events(model)
    model.eval()
    expected = model(graph)

    node_permutation = torch.tensor((2, 0, 3, 1))
    node_observed = model(_permute_nodes(graph, node_permutation))
    assert torch.equal(
        expected.terminal_facts[:, node_permutation],
        node_observed.terminal_facts,
    )
    assert torch.allclose(
        expected.terminal_membrane[:, node_permutation],
        node_observed.terminal_membrane,
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        expected.terminal_readout,
        node_observed.terminal_readout,
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.equal(expected.halt_step, node_observed.halt_step)

    object_permutation = torch.tensor((2, 0, 3, 1))
    object_observed = model(
        _object_permuted_graph(graph, object_permutation)
    )
    assert torch.equal(
        _permute_objects(expected.terminal_facts, object_permutation),
        object_observed.terminal_facts,
    )
    assert torch.allclose(
        _permute_membrane_objects(
            expected.terminal_membrane,
            object_permutation,
        ),
        object_observed.terminal_membrane,
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        _permute_objects(
            expected.terminal_readout,
            object_permutation,
        ),
        object_observed.terminal_readout,
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.equal(expected.halt_step, object_observed.halt_step)


def test_fact_latches_are_exact_and_monotone() -> None:
    torch.manual_seed(2026072303)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=5,
    )
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(-20.0)
    rollout = model(_graph(), return_history=True)
    assert rollout.fact_history is not None
    history = rollout.fact_history
    assert torch.equal(history, history.round())
    assert bool(history[:, 1:].ge(history[:, :-1]).all())
    initially_set = history[:, :1].bool().expand_as(history)
    assert bool(history.masked_select(initially_set).eq(1).all())
    assert rollout.safety_exhausted.all()
    assert rollout.halt_step.eq(-1).all()


def test_soft_curriculum_events_remain_monotone_without_early_absorption() -> None:
    torch.manual_seed(2026072306)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=4,
    )
    with torch.no_grad():
        model.halt_head.bias.fill_(20.0)
    rollout = model(
        _graph(),
        hard_events=False,
        return_history=True,
    )
    assert rollout.fact_history is not None
    assert bool(
        rollout.fact_history[:, 1:].ge(
            rollout.fact_history[:, :-1]
        ).all()
    )
    assert rollout.safety_exhausted.all()
    assert rollout.halt_step.eq(-1).all()


def test_learned_halt_is_absorbing_for_all_owned_state() -> None:
    torch.manual_seed(2026072304)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=5,
    )
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(20.0)
    rollout = model(_graph(), return_history=True)
    assert rollout.fact_history is not None
    assert rollout.membrane_history is not None
    assert rollout.evidence_history is not None
    assert rollout.halted_history is not None
    assert rollout.halt_step.eq(1).all()
    assert rollout.learned_halted.all()
    assert not rollout.safety_exhausted.any()
    for history in (
        rollout.fact_history,
        rollout.membrane_history,
        rollout.evidence_history,
    ):
        expected = history[:, 1:2].expand_as(history[:, 1:])
        assert torch.equal(history[:, 1:], expected)
    assert rollout.halted_history[:, 1:].all()


def test_masks_fail_closed_and_outputs_keep_padding_zero() -> None:
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=2,
    )
    graph = _graph()
    rollout = model(graph)
    node_pair = (
        graph.node_mask[..., None, None]
        & graph.object_mask[:, None, :, None]
        & graph.object_mask[:, None, None, :]
    )
    assert not rollout.terminal_facts.masked_select(~node_pair).any()
    assert not rollout.terminal_evidence.masked_select(~node_pair).any()
    assert not rollout.terminal_membrane.masked_select(
        ~node_pair[..., None]
    ).any()

    covert_seed = graph.seed_facts.clone()
    covert_seed[0, 3, 0, 0] = 1.0
    with pytest.raises(AHRFError, match="covert state"):
        model(replace(graph, seed_facts=covert_seed))

    covert_witness = graph.witness_output.clone()
    covert_witness[0, 0, 2, 0, 0] = 1.0
    with pytest.raises(AHRFError, match="covert state"):
        model(replace(graph, witness_output=covert_witness))

    invalid_edge = graph.argument_edges.clone()
    invalid_edge[0, 2, 3, 0] = True
    with pytest.raises(AHRFError, match="masked node"):
        model(replace(graph, argument_edges=invalid_edge))

    disconnected = graph.argument_edges.clone()
    disconnected[0, 2, 0, 1] = False
    disconnected[0, 1, 0, 0] = False
    with pytest.raises(AHRFError, match="disconnected"):
        model(replace(graph, argument_edges=disconnected))

    unused_card = graph.node_card_mask.clone()
    unused_card[0, 1, 0] = False
    with pytest.raises(AHRFError, match="unused"):
        model(replace(graph, node_card_mask=unused_card))


def test_gradients_are_finite_and_reach_learned_mechanics() -> None:
    torch.manual_seed(2026072305)
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=3,
    )
    with torch.no_grad():
        model.halt_head.bias.fill_(-2.0)
    rollout = model(_graph())
    loss = (
        rollout.terminal_membrane.square().mean()
        + rollout.terminal_evidence.mean()
        + rollout.terminal_facts.mean()
        + rollout.halt_logits.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = sum(float(gradient.square().sum()) for gradient in gradients)
    assert math.isfinite(norm)
    assert norm > 0.0
    assert model.card_encoder.pair_input.weight.grad is not None
    assert model.write_head.weight.grad is not None
    assert model.halt_head.weight.grad is not None


def _identity_delay_graph(
    *,
    delayed: bool,
) -> SourceDeletedRelationGraph:
    graph = _graph()
    edges = torch.zeros_like(graph.argument_edges)
    edges[0, 1, 0, 0] = True
    if delayed:
        edges[0, 2, 1, 0] = True
    else:
        edges[0, 2, 0, 0] = True
        edges[0, 2, 1, 1] = True
    seed = torch.zeros_like(graph.seed_facts)
    seed[0, 0, :3, :3] = 1.0
    return replace(
        graph,
        argument_edges=edges,
        seed_facts=seed,
    )


def test_identity_delay_falsifier_moves_halt_not_terminal_state() -> None:
    """A relay preserves semantic facts but adds one learned-halt tick."""

    model = AutocatalyticHystereticRelationField(
        node_feature_dim=NODE_FEATURES,
        hidden_dim=16,
        card_rounds=1,
        max_steps=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.evidence_head.bias.fill_(-6.0)
        model.write_head.bias.fill_(-6.0)
        model.write_head.weight[
            0,
            model.write_incoming_offset,
        ] = 12.0
        model.halt_head.bias.fill_(-6.0)
        model.halt_head.weight[0, 0] = 12.0

    direct = model(_identity_delay_graph(delayed=False))
    delayed = model(_identity_delay_graph(delayed=True))
    assert direct.learned_halted.all()
    assert delayed.learned_halted.all()
    assert direct.halt_step.item() == 1
    assert delayed.halt_step.item() == 2
    assert torch.equal(direct.terminal_facts, delayed.terminal_facts)
    assert torch.equal(
        direct.terminal_readout,
        delayed.terminal_readout,
    )
