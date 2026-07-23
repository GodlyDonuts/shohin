from __future__ import annotations

from itertools import product

import pytest
import torch

from closure_tied_action_algebra import apply_action, compose_actions
from ctaa_neural_core import (
    CTAA_MAX_STEPS,
    CTAA_STOP_ID,
    ClosureFeatureTransitionCore,
    ClosureTiedPointerCore,
    OuterProductTransitionControl,
    execute_streamed_dual,
    execute_streamed_state_route,
)


def packet_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cards = torch.tensor(
        [[[1, 0, 2], [2, 2, 0], [0, 1, 1], [2, 0, 1]]],
        dtype=torch.long,
    )
    active = [0, 1, 0]
    schedule = torch.tensor(
        [[*active, CTAA_STOP_ID, *([1] * (CTAA_MAX_STEPS - len(active) - 1))]],
        dtype=torch.long,
    )
    initial = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return cards, schedule, initial


def exact_core() -> ClosureTiedPointerCore:
    core = ClosureTiedPointerCore(width=3)
    with torch.no_grad():
        core.address_logits.fill_(-20.0)
        core.address_logits.diagonal().fill_(20.0)
    return core


class CountingCopyCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del right
        self.batch_sizes.append(left.shape[0])
        return torch.nn.functional.one_hot(left, 3).float().mul(40).sub(20)


def test_exact_address_core_matches_every_application_and_composition() -> None:
    core = exact_core()
    tuples = tuple(product(range(3), repeat=3))
    left = torch.tensor(
        [first for first in tuples for _second in tuples], dtype=torch.long
    )
    right = torch.tensor(
        [second for _first in tuples for second in tuples], dtype=torch.long
    )
    predicted = core.hard_step(left, right)
    expected = torch.tensor(
        [apply_action(first, second) for first in tuples for second in tuples],
        dtype=torch.long,
    )
    assert torch.equal(predicted, expected)
    composed = torch.tensor(
        [compose_actions(first, second) for first in tuples for second in tuples],
        dtype=torch.long,
    )
    assert torch.equal(predicted, composed)
    assert core.unique_parameters == 9


def test_exact_address_core_state_and_composed_routes_agree_at_every_step() -> None:
    core = exact_core()
    cards, schedule, initial = packet_tensors()
    trace = execute_streamed_dual(core, 3, cards, schedule, initial)
    assert torch.equal(trace.state_route.states, trace.composed_states)
    assert trace.composed_cards[0, :5].tolist() == [
        [0, 1, 2],
        [1, 0, 2],
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
    ]
    assert trace.composed_cards.shape == (1, CTAA_MAX_STEPS + 1, 3)


def test_distribution_path_and_straight_through_commit_are_finite() -> None:
    core = ClosureTiedPointerCore(width=3)
    left = torch.tensor([[2, 0, 1], [1, 1, 0]], dtype=torch.long)
    right = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    logits = core(left, right)
    committed = core.straight_through_commit(logits)
    loss = -logits[..., 0].mean() + committed[..., 1].mean()
    loss.backward()
    assert logits.shape == (2, 3, 3)
    assert committed.shape == logits.shape
    assert torch.isfinite(core.address_logits.grad).all()


def test_hard_packet_uses_only_compiled_cards_and_absorbing_stop() -> None:
    core = exact_core()
    cards, schedule, initial = packet_tensors()
    trace = execute_streamed_state_route(core, 3, cards, schedule, initial)
    assert trace.states.shape == (1, CTAA_MAX_STEPS + 1, 3)
    assert trace.halted[0, :5].tolist() == [False, False, False, False, True]
    assert trace.halted[0, 4:].all()
    assert trace.states[0, :5].tolist() == [
        [0, 1, 2],
        [1, 0, 2],
        [2, 2, 1],
        [2, 2, 1],
        [2, 2, 1],
    ]


def test_stop_physically_prevents_all_post_halt_core_calls() -> None:
    cards, schedule, initial = packet_tensors()
    cards = cards.expand(2, -1, -1).clone()
    initial = initial.expand(2, -1).clone()
    schedule = schedule.expand(2, -1).clone()
    schedule[1, 3] = 1
    schedule[1, 5] = CTAA_STOP_ID

    state_core = CountingCopyCore()
    execute_streamed_state_route(state_core, 3, cards, schedule, initial)
    assert state_core.batch_sizes == [2, 2, 2, 1, 1]

    dual_core = CountingCopyCore()
    execute_streamed_dual(dual_core, 3, cards, schedule, initial)
    assert dual_core.batch_sizes == [
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
    ]


def test_hard_packet_validates_entire_geometry_before_execution() -> None:
    core = exact_core()
    cards, schedule, initial = packet_tensors()
    bad_cards = cards[:, :1]
    with pytest.raises(ValueError, match="action-card geometry"):
        execute_streamed_state_route(core, 3, bad_cards, schedule, initial)
    no_stop = schedule.clone().fill_(0)
    with pytest.raises(ValueError, match="exactly one STOP"):
        execute_streamed_state_route(core, 3, cards, no_stop, initial)
    two_stops = schedule.clone()
    two_stops[:, 8] = CTAA_STOP_ID
    with pytest.raises(ValueError, match="exactly one STOP"):
        execute_streamed_state_route(core, 3, cards, two_stops, initial)
    with pytest.raises(ValueError, match="initial-state batch"):
        execute_streamed_state_route(
            core, 3, cards, schedule, torch.tensor([[0, 1]], dtype=torch.long)
        )


def test_matched_cores_have_exact_parameter_and_flop_contract() -> None:
    treatment = ClosureFeatureTransitionCore()
    control = OuterProductTransitionControl()
    assert treatment.unique_parameters == control.unique_parameters == 107_753
    assert treatment.analytic_inference_flops == 215_530
    assert control.analytic_inference_flops == 215_584
    difference = control.analytic_inference_flops - treatment.analytic_inference_flops
    assert difference == 54
    assert difference / treatment.analytic_inference_flops < 0.0003


def test_outer_product_control_features_separate_all_finite_input_pairs() -> None:
    tuples = tuple(product(range(3), repeat=3))
    left = torch.tensor(
        [first for first in tuples for _second in tuples],
        dtype=torch.long,
    )
    right = torch.tensor(
        [second for _first in tuples for second in tuples],
        dtype=torch.long,
    )
    treatment = ClosureFeatureTransitionCore()
    control = OuterProductTransitionControl()
    treatment_features = treatment.features(left, right)
    control_features = control.features(left, right)
    assert treatment_features.shape == (729, 27)
    assert control_features.shape == (729, 81)
    assert torch.unique(control_features, dim=0).shape[0] == 729


def test_matched_cores_share_output_and_hard_state_geometry() -> None:
    treatment = ClosureFeatureTransitionCore()
    control = OuterProductTransitionControl()
    left = torch.tensor([[2, 0, 1], [1, 1, 0]], dtype=torch.long)
    right = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    assert treatment(left, right).shape == control(left, right).shape == (2, 3, 3)
    cards, schedule, initial = packet_tensors()
    treatment_trace = execute_streamed_state_route(
        treatment, 3, cards, schedule, initial
    )
    control_trace = execute_streamed_state_route(control, 3, cards, schedule, initial)
    assert treatment_trace.states.shape == control_trace.states.shape == (
        1,
        CTAA_MAX_STEPS + 1,
        3,
    )
    assert treatment_trace.states.element_size() == control_trace.states.element_size()
