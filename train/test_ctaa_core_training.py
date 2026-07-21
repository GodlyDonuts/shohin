from __future__ import annotations

from itertools import product

import torch

from ctaa_core_training import (
    ARMS,
    AtomicBatch,
    ClosureBatch,
    apply_derangement,
    closure_label_derangement,
    make_core,
    matched_core_loss,
)


def batches() -> tuple[AtomicBatch, ClosureBatch]:
    actions = torch.tensor(list(product(range(3), repeat=3)), dtype=torch.long)
    states = actions.roll(5, 0)
    output = states.gather(1, actions)
    atomic = AtomicBatch(actions, states, output)
    first = actions
    second = actions.roll(7, 0)
    composed = first.gather(1, second)
    closure = ClosureBatch(
        first=first,
        second=second,
        state=states,
        composed=composed,
        output=states.gather(1, composed),
    )
    return atomic, closure


def test_all_arms_have_exactly_four_matched_calls_and_reach_every_parameter() -> None:
    atomic, closure = batches()
    mapping = closure_label_derangement(19, closure.composed)
    for arm in ARMS:
        core = make_core(arm)
        receipt = matched_core_loss(
            core,
            arm,
            atomic,
            closure,
            shuffled_mapping=mapping if arm == "ctaa_shuffled_closure" else None,
        )
        receipt.total.backward()
        assert receipt.transition_calls_per_closure_row == 4
        assert len(receipt.call_losses) == 4
        assert torch.isfinite(receipt.total)
        assert all(parameter.grad is not None for parameter in core.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters())


def test_shuffled_closure_mapping_is_deterministic_and_has_no_fixed_point() -> None:
    _, closure = batches()
    first = closure_label_derangement(23, closure.composed)
    second = closure_label_derangement(23, closure.composed)
    assert first == second
    assert all(left != right for left, right in first.items())
    shuffled = apply_derangement(closure.composed, first)
    assert shuffled.shape == closure.composed.shape
    assert not torch.equal(shuffled, closure.composed)
