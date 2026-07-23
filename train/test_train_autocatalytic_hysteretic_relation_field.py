from __future__ import annotations

from types import SimpleNamespace

import torch

from autocatalytic_hysteretic_relation_field import (
    FEEDBACK_ROLE,
    AutocatalyticHystereticRelationField,
    SourceDeletedRelationGraph,
)
from contextual_witness_equivariant_binder import (
    ContextualWitnessEquivariantBinder,
)
from train_autocatalytic_hysteretic_relation_field import (
    AHRFTrainConfig,
    _apply_control,
    _root_facts,
    _terminal_loss,
    _transfer_binder,
    build_board,
)


def test_root_fact_gather_handles_terminal_and_history() -> None:
    terminal = torch.arange(2 * 5 * 3 * 3).reshape(2, 5, 3, 3)
    roots = torch.tensor(((4, 1), (2, 3)))
    observed = _root_facts(terminal, roots)
    expected = torch.stack(
        (
            torch.stack((terminal[0, 4], terminal[0, 1])),
            torch.stack((terminal[1, 2], terminal[1, 3])),
        )
    )
    assert torch.equal(observed, expected)
    history = torch.stack((terminal, terminal + 1), dim=1)
    history_observed = _root_facts(history, roots)
    assert torch.equal(history_observed[:, 0], expected)
    assert torch.equal(history_observed[:, 1], expected + 1)


def test_hard_terminal_loss_preserves_straight_through_gradient() -> None:
    logits = torch.tensor(
        [[[[2.0, -2.0], [-2.0, 2.0]]] * 2],
        requires_grad=True,
    )
    probability = logits.sigmoid()
    hard = logits.ge(0).to(logits.dtype)
    facts = hard + probability - probability.detach()
    targets = 1.0 - hard.detach()
    loss = _terminal_loss(
        SimpleNamespace(terminal_facts=facts),
        torch.tensor(((0, 1),)),
        targets,
        torch.ones(1, 2, dtype=torch.bool),
        hard_events=True,
    )
    loss.backward()
    assert logits.grad is not None
    assert bool(logits.grad.ne(0).all())
    assert bool(torch.isfinite(logits.grad).all())


def test_frozen_board_receipts_require_a_longer_safety_horizon() -> None:
    board = build_board(AHRFTrainConfig(seed=2026072339))
    assert board.max_expression_depth == 9
    assert board.max_convergence_updates == 8
    assert board.minimum_safety_steps == 56
    assert board.minimum_safety_steps <= 64


def test_contextual_binder_warm_start_copies_equivariant_core(
    tmp_path,
) -> None:
    binder = ContextualWitnessEquivariantBinder(
        width=16,
        rounds=1,
    )
    checkpoint_path = tmp_path / "binder.pt"
    torch.save(
        {
            "protocol": "contextual_witness_equivariant_binder_v1",
            "config": {
                "width": 16,
                "rounds": 1,
                "architecture": "equivariant",
                "triad_mode": "learned",
            },
            "model_state": binder.state_dict(),
        },
        checkpoint_path,
    )
    model = AutocatalyticHystereticRelationField(
        node_feature_dim=3,
        hidden_dim=16,
        card_rounds=1,
        max_steps=2,
    )
    receipt = _transfer_binder(model, checkpoint_path)
    assert receipt["copied_parameters"] > 0
    assert torch.equal(
        model.card_encoder.pair_input.weight,
        binder.pair_input.weight,
    )
    assert torch.equal(
        model.card_encoder.slot_encoder[0].weight,
        binder.card_classifier[0].weight,
    )


def test_graph_controls_remove_only_preregistered_information() -> None:
    graph = SourceDeletedRelationGraph(
        node_features=torch.ones(1, 3, 1),
        node_mask=torch.ones(1, 3, dtype=torch.bool),
        argument_edges=torch.ones(1, 3, 3, 3, dtype=torch.bool),
        node_card_mask=torch.eye(3, dtype=torch.bool)[None],
        root_mask=torch.tensor([[False, False, True]]),
        seed_facts=torch.zeros(1, 3, 2, 2),
        witness_left=torch.zeros(1, 3, 1, 2, 2),
        witness_right=torch.zeros(1, 3, 1, 2, 2),
        witness_output=torch.zeros(1, 3, 1, 2, 2),
        witness_mask=torch.ones(1, 3, 1, dtype=torch.bool),
        argument_mask=torch.ones(1, 3, 1, 2, dtype=torch.bool),
        object_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    no_feedback = _apply_control(graph, "no_feedback")
    assert not no_feedback.argument_edges[..., FEEDBACK_ROLE].any()
    assert torch.equal(
        no_feedback.argument_edges[..., :2],
        graph.argument_edges[..., :2],
    )
    shuffled = _apply_control(graph, "shuffled_cards")
    assert torch.equal(
        shuffled.node_card_mask.any(1),
        graph.node_card_mask.any(1),
    )
    assert not torch.equal(
        shuffled.node_card_mask,
        graph.node_card_mask,
    )
    for control in ("false_triad", "zero_triad"):
        assert _apply_control(graph, control) is graph
