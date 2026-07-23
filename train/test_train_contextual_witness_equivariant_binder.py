from __future__ import annotations

import torch

from contextual_witness_card_data import generate_contextual_card_batch
from contextual_witness_equivariant_binder import (
    ContextualWitnessEquivariantBinder,
)
from train_contextual_witness_equivariant_binder import (
    _equivariance_receipt,
    compute_losses,
)


def test_losses_are_finite_and_reach_every_parameter() -> None:
    model = ContextualWitnessEquivariantBinder(width=16, rounds=1)
    batch = generate_contextual_card_batch(
        batch_size=4,
        generator=torch.Generator().manual_seed(2026072328),
        invalid_fraction=0.5,
    )
    loss, parts = compute_losses(
        model,
        batch,
        hard=False,
        semantic_weight=0.25,
        margin_weight=0.05,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_equivariance_receipt_is_exact_before_training() -> None:
    model = ContextualWitnessEquivariantBinder(width=16, rounds=1)
    batch = generate_contextual_card_batch(
        batch_size=4,
        generator=torch.Generator().manual_seed(2026072329),
    )
    receipt = _equivariance_receipt(
        model,
        batch,
        generator=torch.Generator().manual_seed(2026072330),
    )
    assert receipt["all_exact"] is True
