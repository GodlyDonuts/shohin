from __future__ import annotations

import math

import torch

from contextual_witness_equivariant_binder import (
    BINDER_CLASS_COUNT,
    CARD_WITNESSES,
    MAX_OBJECTS,
    MAX_OPERATION_SLOTS,
    REJECT_INDEX,
    ContextualWitnessEquivariantBinder,
    ContextualWitnessStatisticsBinder,
)


def _cards(batch: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2026072324)
    shape = (
        batch,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    left = torch.randint(0, 2, shape, generator=generator).float()
    right = torch.randint(0, 2, shape, generator=generator).float()
    output = torch.randint(0, 2, shape, generator=generator).float()
    object_mask = torch.ones(batch, MAX_OBJECTS, dtype=torch.bool)
    witness_mask = torch.ones(
        batch,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        dtype=torch.bool,
    )
    arity = torch.tensor((2, 2, 2, 1, 0, 2, 1, 0))
    argument_mask = (
        torch.arange(2)[None, None, None] < arity[None, :, None, None]
    ).expand(batch, -1, CARD_WITNESSES, -1)
    return (
        left,
        right,
        output,
        witness_mask,
        argument_mask,
        object_mask,
    )


def _permute_objects(
    value: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    return value.index_select(-2, permutation).index_select(-1, permutation)


def test_logits_are_object_permutation_invariant() -> None:
    cards = _cards()
    binder = ContextualWitnessEquivariantBinder()
    expected = binder(*cards, hard=False)
    permutation = torch.tensor((7, 2, 5, 0, 3, 6, 1, 4))
    left, right, output, witness_mask, argument_mask, object_mask = cards
    observed = binder(
        _permute_objects(left, permutation),
        _permute_objects(right, permutation),
        _permute_objects(output, permutation),
        witness_mask,
        argument_mask,
        object_mask[:, permutation],
        hard=False,
    )
    assert torch.allclose(expected.logits, observed.logits, atol=1e-5, rtol=0.0)


def test_witness_and_card_permutations_are_equivariant() -> None:
    cards = _cards()
    binder = ContextualWitnessEquivariantBinder()
    expected = binder(*cards, hard=False)
    witness = torch.tensor((5, 0, 7, 2, 4, 1, 6, 3))
    card = torch.tensor((6, 2, 7, 0, 4, 1, 5, 3))
    left, right, output, witness_mask, argument_mask, object_mask = cards
    observed = binder(
        left[:, card][:, :, witness],
        right[:, card][:, :, witness],
        output[:, card][:, :, witness],
        witness_mask[:, card][:, :, witness],
        argument_mask[:, card][:, :, witness],
        object_mask,
        hard=False,
    )
    assert torch.allclose(
        expected.logits[:, card],
        observed.logits,
        atol=1e-5,
        rtol=0.0,
    )


def test_hard_assignments_are_exact_and_reject_is_fail_closed() -> None:
    cards = _cards(batch=1)
    binder = ContextualWitnessEquivariantBinder()
    with torch.no_grad():
        final = binder.card_classifier[-1]
        final.weight.zero_()
        final.bias.fill_(-10.0)
        final.bias[REJECT_INDEX] = 10.0
    result = binder(*cards, hard=True)
    assert result.logits.shape == (
        1,
        MAX_OPERATION_SLOTS,
        BINDER_CLASS_COUNT,
    )
    assert result.rejected.all()
    assert result.discrete_assignment.count_nonzero() == 0
    assert torch.equal(
        result.discrete_assignment,
        result.discrete_assignment.round(),
    )


def test_gradient_and_parameter_receipt_are_finite() -> None:
    cards = _cards(batch=1)
    binder = ContextualWitnessEquivariantBinder(width=32)
    result = binder(*cards, hard=False)
    loss = result.logits.square().mean()
    loss.backward()
    norm = sum(
        float(parameter.grad.square().sum())
        for parameter in binder.parameters()
        if parameter.grad is not None
    )
    assert math.isfinite(norm)
    assert norm > 0.0
    receipt = binder.parameter_receipt()
    assert receipt["added"] == binder.added_parameters
    assert receipt["complete_system"] < receipt["strict_cap"]
    assert receipt["headroom"] > 0


def test_matched_triad_controls_preserve_size_and_object_equivariance() -> None:
    cards = _cards(batch=1)
    permutation = torch.tensor((2, 0, 3, 1, 7, 4, 6, 5))
    left, right, output, witness_mask, argument_mask, object_mask = cards
    receipts = []
    for mode in ("learned", "false", "zero"):
        binder = ContextualWitnessEquivariantBinder(
            width=16,
            rounds=1,
            triad_mode=mode,
        )
        receipts.append(binder.added_parameters)
        expected = binder(*cards, hard=False).logits
        observed = binder(
            _permute_objects(left, permutation),
            _permute_objects(right, permutation),
            _permute_objects(output, permutation),
            witness_mask,
            argument_mask,
            object_mask[:, permutation],
            hard=False,
        ).logits
        assert torch.allclose(expected, observed, atol=1e-5, rtol=0.0)
    assert len(set(receipts)) == 1


def test_statistics_control_is_object_and_witness_invariant() -> None:
    cards = _cards(batch=1)
    binder = ContextualWitnessStatisticsBinder(width=16)
    expected = binder(*cards, hard=False).logits
    objects = torch.tensor((7, 2, 5, 0, 3, 6, 1, 4))
    witnesses = torch.tensor((5, 0, 7, 2, 4, 1, 6, 3))
    left, right, output, witness_mask, argument_mask, object_mask = cards
    observed = binder(
        _permute_objects(left[:, :, witnesses], objects),
        _permute_objects(right[:, :, witnesses], objects),
        _permute_objects(output[:, :, witnesses], objects),
        witness_mask[:, :, witnesses],
        argument_mask[:, :, witnesses],
        object_mask[:, objects],
        hard=False,
    ).logits
    assert torch.allclose(expected, observed, atol=1e-5, rtol=0.0)
    assert binder.parameter_receipt()["complete_system"] < 200_000_000
