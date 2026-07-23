from __future__ import annotations

import torch

from autocatalytic_hysteretic_relation_field import (
    AutocatalyticHystereticRelationField,
)
from contextual_witness_equivariant_binder import (
    ContextualWitnessEquivariantBinder,
)
from train_autocatalytic_hysteretic_relation_field import (
    _root_facts,
    _transfer_binder,
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
