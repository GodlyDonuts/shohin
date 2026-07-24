from __future__ import annotations

from dataclasses import fields
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_hankel_completion import (  # noqa: E402
    DirectDualCompletionControlProjector,
    HankelCompletionError,
    HankelShiftCompletionProjector,
    project_behavioral_shifts,
)
from episode_functor_machine import HardFunctorMachine  # noqa: E402
from pipeline.episode_functor_hankel_shift import (  # noqa: E402
    commutative_bag_incidence,
    enumerate_action_words,
    prefix_shift_incidence,
    random_shift_incidence,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    generate_machine,
)


def _random_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(311)
    return (
        torch.randn((2, 3, 8, 8), generator=generator),
        torch.randn((2, 2, 8, 4), generator=generator),
    )


def _oracle_probabilities() -> tuple[torch.Tensor, torch.Tensor]:
    machine = generate_machine(
        seed="efc-neural-hankel-oracle-v1",
        split="mechanics",
        index=0,
        family="quaternion-regular",
    )
    transition = F.one_hot(
        torch.tensor(machine.transitions),
        8,
    ).float()[None]
    observer = F.one_hot(
        torch.tensor(machine.observations),
        4,
    ).float()[None]
    return transition, observer


def test_oracle_behavioral_shift_recovers_machine_with_zero_syndrome() -> None:
    transition, observer = _oracle_probabilities()
    result = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=3,
        incidence=torch.tensor(prefix_shift_incidence(3)),
        temperature=0.05,
    )
    assert result.base_signatures.shape == (1, 8, 40, 2, 4)
    assert result.derivative_signatures.shape == (1, 3, 8, 40, 2, 4)
    assert result.shift_incidence.shape == (3, 40)
    assert torch.equal(
        result.projection.machine.action_next[
            0,
            :3,
            :8,
            :8,
        ].argmax(-1),
        transition[0].argmax(-1),
    )
    assert torch.equal(
        result.projection.machine.observer_answer[
            0,
            :2,
            :8,
            :4,
        ].argmax(-1),
        observer[0].argmax(-1),
    )
    assert torch.equal(
        result.shift_syndrome,
        torch.zeros_like(result.shift_syndrome),
    )


def test_hard_machine_is_detached_and_contains_no_behavioral_workspace() -> None:
    transition, observer = _oracle_probabilities()
    result = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=3,
        incidence=torch.tensor(prefix_shift_incidence(3)),
        temperature=0.05,
    )
    hard = result.projection.machine.harden()
    frozen = tuple(
        getattr(hard, field.name).clone()
        for field in fields(HardFunctorMachine)
    )
    transition.zero_()
    observer.zero_()
    assert tuple(field.name for field in fields(hard)) == (
        "state_active",
        "action_active",
        "observer_active",
        "action_next",
        "observer_answer",
    )
    assert all(
        torch.equal(getattr(hard, field.name), expected)
        for field, expected in zip(
            fields(HardFunctorMachine),
            frozen,
            strict=True,
        )
    )


def test_hankel_projector_is_trainable_and_backpropagates_through_shifts() -> None:
    torch.manual_seed(313)
    projector = HankelShiftCompletionProjector(
        width=32,
        iterations=2,
        max_depth=2,
    )
    transitions, observers = _random_inputs()
    transitions.requires_grad_()
    observers.requires_grad_()
    result = projector.detailed_forward(transitions, observers)
    assert result.base_signatures.shape == (
        2,
        8,
        len(enumerate_action_words(2)),
        2,
        4,
    )
    loss = (
        result.shift_distances.square().mean()
        + result.projection.machine.action_next[:, :3, :8, :8].square().mean()
        + result.projection.machine.observer_answer[:, :2, :8, :4]
        .square()
        .mean()
    )
    loss.backward()
    assert transitions.grad is not None
    assert observers.grad is not None
    assert float(transitions.grad.abs().sum()) > 0.0
    assert float(observers.grad.abs().sum()) > 0.0
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in projector.parameters()
    )


def test_soft_projection_is_exactly_recode_equivariant() -> None:
    torch.manual_seed(317)
    projector = HankelShiftCompletionProjector(
        width=32,
        iterations=2,
        max_depth=2,
    )
    transitions, observers = _random_inputs()
    state = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action = torch.tensor((2, 0, 1))
    observer = torch.tensor((1, 0))
    answer = torch.tensor((2, 0, 3, 1))
    original = projector(transitions, observers)
    changed = projector(
        transitions[:, action][:, :, state][:, :, :, state],
        observers[:, observer][:, :, state][:, :, :, answer],
    )
    assert torch.allclose(
        changed.transition_transport,
        original.transition_transport[:, action][
            :,
            :,
            state,
        ][
            :,
            :,
            :,
            state,
        ],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        changed.observer_transport,
        original.observer_transport[:, observer][
            :,
            :,
            state,
        ][
            :,
            :,
            :,
            answer,
        ],
        atol=2e-5,
        rtol=2e-5,
    )


def test_random_and_commutative_controls_have_distinct_incidence() -> None:
    prefix = HankelShiftCompletionProjector(
        width=32,
        iterations=1,
        max_depth=2,
        incidence_mode="prefix",
    )
    random = HankelShiftCompletionProjector(
        width=32,
        iterations=1,
        max_depth=2,
        incidence_mode="random",
        random_seed="matched-control",
    )
    commutative = HankelShiftCompletionProjector(
        width=32,
        iterations=1,
        max_depth=2,
        incidence_mode="commutative",
    )
    assert not torch.equal(prefix.shift_incidence, random.shift_incidence)
    assert not torch.equal(prefix.shift_incidence, commutative.shift_incidence)
    assert "shift_incidence" in prefix.state_dict()


def test_direct_dual_control_has_same_capacity_but_bypasses_shift_decode() -> None:
    torch.manual_seed(319)
    shift = HankelShiftCompletionProjector(
        width=32,
        iterations=2,
        max_depth=2,
    )
    direct = DirectDualCompletionControlProjector(
        width=32,
        iterations=2,
        max_depth=2,
    )
    direct.load_state_dict(shift.state_dict())
    transitions, observers = _random_inputs()
    shifted = shift.detailed_forward(transitions, observers)
    controlled = direct.detailed_forward(transitions, observers)
    assert shift.parameter_count() == direct.parameter_count()
    assert torch.equal(
        shifted.base_signatures,
        controlled.base_signatures,
    )
    assert torch.equal(
        shifted.derivative_signatures,
        controlled.derivative_signatures,
    )
    assert not torch.equal(
        shifted.projection.machine.action_next,
        controlled.projection.machine.action_next,
    )


@pytest.mark.parametrize("mode", ("prefix", "random", "commutative"))
def test_every_incidence_arm_is_action_recode_equivariant_on_oracle(
    mode: str,
) -> None:
    transition, observer = _oracle_probabilities()
    if mode == "prefix":
        incidence = prefix_shift_incidence(3)
    elif mode == "random":
        incidence = random_shift_incidence(
            3,
            seed="hsc-equivariant-position-control-v1",
        )
    else:
        incidence = commutative_bag_incidence(3)
    action = torch.tensor((2, 0, 1))
    original = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=3,
        incidence=torch.tensor(incidence),
        temperature=0.05,
    )
    changed = project_behavioral_shifts(
        base_transition=transition[:, action],
        base_observer=observer,
        derivative_transition=transition[:, action],
        derivative_observer=observer,
        max_depth=3,
        incidence=torch.tensor(incidence),
        temperature=0.05,
    )
    assert torch.allclose(
        changed.projection.transition_transport,
        original.projection.transition_transport[:, action],
        atol=2e-5,
        rtol=2e-5,
    )


def test_invalid_geometry_and_hard_ties_fail_closed() -> None:
    transition, observer = _oracle_probabilities()
    with pytest.raises(HankelCompletionError, match="incidence"):
        project_behavioral_shifts(
            base_transition=transition,
            base_observer=observer,
            derivative_transition=transition,
            derivative_observer=observer,
            max_depth=3,
            incidence=torch.zeros((3, 40), dtype=torch.float32),
            temperature=0.05,
        )
    tied = torch.full((1, 3, 8, 8), 1.0 / 8.0)
    tied_observer = torch.full((1, 2, 8, 4), 0.25)
    with pytest.raises(HankelCompletionError, match="categorical tie"):
        project_behavioral_shifts(
            base_transition=tied,
            base_observer=tied_observer,
            derivative_transition=tied,
            derivative_observer=tied_observer,
            max_depth=1,
            incidence=torch.tensor(prefix_shift_incidence(1)),
            temperature=0.05,
            straight_through=True,
        )
