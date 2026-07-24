from __future__ import annotations

from dataclasses import fields
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from episode_functor_causal_syndrome_observer import (
    ACSO_ADDED_PARAMETERS,
    ACSO_COMPLETE_PARAMETERS,
    ACSO_HEADROOM,
    AdjointCausalSyndromePreconditioner,
    CausalSyndromeObserverError,
    behavioral_closure,
    causal_syndrome_innovation,
    cyclic_control_word,
    explicit_causal_adjoint,
    seal_primary_machine,
)
from episode_functor_capacity_lanes import (
    HANKEL_SHIFT_MAXIMUM_EXPECTED,
)
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_learned_system import GLOBAL_PARAMETER_LIMIT
from episode_functor_machine import HardFunctorMachine
from pipeline.episode_functor_hankel_geometry import (
    enumerate_action_words,
)


def _logits(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    transition = torch.randn(
        (2, PRIMARY_ACTIONS, PRIMARY_STATES, PRIMARY_STATES),
        generator=generator,
        dtype=torch.float32,
    )
    observer = torch.randn(
        (
            2,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        ),
        generator=generator,
        dtype=torch.float32,
    )
    return transition, observer


def _targets(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    transition, observer = _logits(seed)
    closure = behavioral_closure(transition, observer)
    return closure.base.detach(), closure.derivative.detach()


@pytest.mark.parametrize("routing_mode", ("causal", "cyclic-control"))
def test_explicit_adjoint_matches_autograd(
    routing_mode: str,
) -> None:
    transition, observer = _logits()
    target_base, target_derivative = _targets()
    transition.requires_grad_(True)
    observer.requires_grad_(True)
    base, derivative = causal_syndrome_innovation(
        transition,
        observer,
        target_base,
        target_derivative,
        routing_mode=routing_mode,
    )
    expected_transition, expected_observer = torch.autograd.grad(
        base + derivative,
        (transition, observer),
    )
    actual = explicit_causal_adjoint(
        transition.detach(),
        observer.detach(),
        target_base,
        target_derivative,
        routing_mode=routing_mode,
    )
    torch.testing.assert_close(
        actual.transition_logit_adjoint,
        expected_transition,
        atol=2e-7,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        actual.observer_logit_adjoint,
        expected_observer,
        atol=2e-7,
        rtol=2e-5,
    )


def test_oracle_machine_is_an_exact_fixed_point() -> None:
    transition, observer = _logits()
    target = behavioral_closure(transition, observer)
    adjoint = explicit_causal_adjoint(
        transition,
        observer,
        target.base,
        target.derivative,
    )
    assert abs(float(adjoint.base_innovation)) < 1e-7
    assert abs(float(adjoint.derivative_innovation)) < 1e-7
    assert float(adjoint.transition_logit_adjoint.abs().max()) < 1e-7
    assert float(adjoint.observer_logit_adjoint.abs().max()) < 1e-7


def test_small_negative_adjoint_step_reduces_innovation() -> None:
    target_base, target_derivative = _targets(seed=101)
    for seed in range(8):
        transition, observer = _logits(seed)
        before = sum(
            causal_syndrome_innovation(
                transition,
                observer,
                target_base,
                target_derivative,
            )
        )
        adjoint = explicit_causal_adjoint(
            transition,
            observer,
            target_base,
            target_derivative,
        )
        step = 0.05
        after = sum(
            causal_syndrome_innovation(
                transition - step * adjoint.transition_logit_adjoint,
                observer - step * adjoint.observer_logit_adjoint,
                target_base,
                target_derivative,
            )
        )
        assert float(after) < float(before)


def test_small_negative_control_step_reduces_its_own_objective() -> None:
    target_base, target_derivative = _targets(seed=101)
    for seed in range(8):
        transition, observer = _logits(seed)
        before = sum(
            causal_syndrome_innovation(
                transition,
                observer,
                target_base,
                target_derivative,
                routing_mode="cyclic-control",
            )
        )
        adjoint = explicit_causal_adjoint(
            transition,
            observer,
            target_base,
            target_derivative,
            routing_mode="cyclic-control",
        )
        step = 0.05
        after = sum(
            causal_syndrome_innovation(
                transition - step * adjoint.transition_logit_adjoint,
                observer - step * adjoint.observer_logit_adjoint,
                target_base,
                target_derivative,
                routing_mode="cyclic-control",
            )
        )
        assert float(after) < float(before)


def test_cyclic_control_is_depth_preserving_bijective_and_equivariant() -> None:
    for depth in range(1, 5):
        words = tuple(
            itertools.product(range(PRIMARY_ACTIONS), repeat=depth)
        )
        routed = tuple(cyclic_control_word(word) for word in words)
        assert len(set(routed)) == len(words)
        assert all(len(left) == len(right) for left, right in zip(words, routed))
        for permutation in itertools.permutations(range(PRIMARY_ACTIONS)):
            for word in words:
                recoded = tuple(permutation[action] for action in word)
                assert cyclic_control_word(recoded) == tuple(
                    permutation[action]
                    for action in cyclic_control_word(word)
                )


def _permute_machine_logits(
    transition: torch.Tensor,
    observer: torch.Tensor,
    state_permutation: torch.Tensor,
    action_permutation: torch.Tensor,
    observer_permutation: torch.Tensor,
    answer_permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_state = torch.argsort(state_permutation)
    inverse_action = torch.argsort(action_permutation)
    inverse_observer = torch.argsort(observer_permutation)
    inverse_answer = torch.argsort(answer_permutation)
    recoded_transition = transition.index_select(1, inverse_action)
    recoded_transition = recoded_transition.index_select(2, inverse_state)
    recoded_transition = recoded_transition.index_select(3, inverse_state)
    recoded_observer = observer.index_select(1, inverse_observer)
    recoded_observer = recoded_observer.index_select(2, inverse_state)
    recoded_observer = recoded_observer.index_select(3, inverse_answer)
    return recoded_transition, recoded_observer


def _permute_targets(
    base: torch.Tensor,
    derivative: torch.Tensor,
    state_permutation: torch.Tensor,
    action_permutation: torch.Tensor,
    observer_permutation: torch.Tensor,
    answer_permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    words = enumerate_action_words(3)
    inverse_state = torch.argsort(state_permutation)
    inverse_action = torch.argsort(action_permutation)
    inverse_observer = torch.argsort(observer_permutation)
    inverse_answer = torch.argsort(answer_permutation)
    word_index = {word: index for index, word in enumerate(words)}
    inverse_word = torch.tensor(
        tuple(
            word_index[
                tuple(
                    int(inverse_action[action])
                    for action in word
                )
            ]
            for word in words
        ),
        dtype=torch.long,
    )
    recoded_base = base.index_select(1, inverse_state)
    recoded_base = recoded_base.index_select(2, inverse_word)
    recoded_base = recoded_base.index_select(3, inverse_observer)
    recoded_base = recoded_base.index_select(4, inverse_answer)
    recoded_derivative = derivative.index_select(1, inverse_action)
    recoded_derivative = recoded_derivative.index_select(2, inverse_state)
    recoded_derivative = recoded_derivative.index_select(3, inverse_word)
    recoded_derivative = recoded_derivative.index_select(
        4,
        inverse_observer,
    )
    recoded_derivative = recoded_derivative.index_select(5, inverse_answer)
    return recoded_base, recoded_derivative


@pytest.mark.parametrize("routing_mode", ("causal", "cyclic-control"))
def test_causal_adjoint_is_exactly_recoding_equivariant(
    routing_mode: str,
) -> None:
    transition, observer = _logits()
    target_base, target_derivative = _targets()
    state = torch.tensor((3, 0, 7, 1, 6, 2, 5, 4))
    action = torch.tensor((2, 0, 1))
    observers = torch.tensor((1, 0))
    answers = torch.tensor((2, 0, 3, 1))
    recoded_logits = _permute_machine_logits(
        transition,
        observer,
        state,
        action,
        observers,
        answers,
    )
    recoded_targets = _permute_targets(
        target_base,
        target_derivative,
        state,
        action,
        observers,
        answers,
    )
    original = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
        routing_mode=routing_mode,
    )
    recoded = explicit_causal_adjoint(
        *recoded_logits,
        *recoded_targets,
        routing_mode=routing_mode,
    )
    expected = _permute_machine_logits(
        original.transition_logit_adjoint,
        original.observer_logit_adjoint,
        state,
        action,
        observers,
        answers,
    )
    torch.testing.assert_close(
        recoded.transition_logit_adjoint,
        expected[0],
    )
    torch.testing.assert_close(
        recoded.observer_logit_adjoint,
        expected[1],
    )


def test_scrambled_objective_changes_only_multi_step_correspondence() -> None:
    transition, observer = _logits()
    target_base, target_derivative = _targets()
    causal = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
        routing_mode="causal",
    )
    control = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
        routing_mode="cyclic-control",
    )
    assert float(causal.base_innovation) == float(control.base_innovation)
    assert float(causal.derivative_innovation) != float(
        control.derivative_innovation
    )
    assert not torch.equal(
        causal.transition_logit_adjoint,
        control.transition_logit_adjoint,
    )


def test_control_preserves_the_complete_one_step_adjoint() -> None:
    transition, observer = _logits()
    target = behavioral_closure(
        *_logits(seed=101),
        max_depth=0,
    )
    causal = explicit_causal_adjoint(
        transition,
        observer,
        target.base,
        target.derivative,
        max_depth=0,
        routing_mode="causal",
    )
    control = explicit_causal_adjoint(
        transition,
        observer,
        target.base,
        target.derivative,
        max_depth=0,
        routing_mode="cyclic-control",
    )
    torch.testing.assert_close(
        control.transition_logit_adjoint,
        causal.transition_logit_adjoint,
    )
    torch.testing.assert_close(
        control.observer_logit_adjoint,
        causal.observer_logit_adjoint,
    )


def test_softmax_logit_adjoints_are_row_centered() -> None:
    transition, observer = _logits()
    target_base, target_derivative = _targets()
    result = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
    )
    torch.testing.assert_close(
        result.transition_logit_adjoint.sum(-1),
        torch.zeros_like(result.transition_logit_adjoint.sum(-1)),
        atol=2e-8,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.observer_logit_adjoint.sum(-1),
        torch.zeros_like(result.observer_logit_adjoint.sum(-1)),
        atol=2e-8,
        rtol=0.0,
    )


def test_preconditioner_parameter_receipt_and_positive_bounded_step() -> None:
    torch.manual_seed(19)
    module = AdjointCausalSyndromePreconditioner()
    assert module.parameter_count() == ACSO_ADDED_PARAMETERS
    assert ACSO_COMPLETE_PARAMETERS == (
        HANKEL_SHIFT_MAXIMUM_EXPECTED.complete_parameters
        + ACSO_ADDED_PARAMETERS
    )
    assert ACSO_HEADROOM == (
        HANKEL_SHIFT_MAXIMUM_EXPECTED.headroom
        - ACSO_ADDED_PARAMETERS
    )
    assert ACSO_COMPLETE_PARAMETERS + ACSO_HEADROOM == GLOBAL_PARAMETER_LIMIT
    features = torch.randn(2, 256, 10)
    step, hidden = module(features)
    assert step.shape == (2, 256)
    assert hidden.shape == (2, 256, 768)
    assert bool(step.ge(0.001).all())
    assert bool(step.le(0.1).all())
    next_step, next_hidden = module(features, hidden)
    assert next_step.shape == step.shape
    assert next_hidden.shape == hidden.shape
    assert not torch.equal(next_hidden, hidden)


def test_explicit_adjoint_cannot_retain_any_autograd_graph() -> None:
    transition, observer = _logits()
    target_base, target_derivative = _targets()
    for tensor in (
        transition,
        observer,
        target_base,
        target_derivative,
    ):
        tensor.requires_grad_(True)
    result = explicit_causal_adjoint(
        transition,
        observer,
        target_base,
        target_derivative,
    )
    for tensor in (
        result.base_innovation,
        result.derivative_innovation,
        result.transition_logit_adjoint,
        result.observer_logit_adjoint,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_hand_calculated_noncommutative_closure() -> None:
    transition = torch.full(
        (1, PRIMARY_ACTIONS, PRIMARY_STATES, PRIMARY_STATES),
        -30.0,
    )
    destinations = (
        tuple((state + 1) % PRIMARY_STATES for state in range(PRIMARY_STATES)),
        (1, 0, 2, 3, 4, 5, 6, 7),
        tuple((state + 2) % PRIMARY_STATES for state in range(PRIMARY_STATES)),
    )
    for action, rows in enumerate(destinations):
        for state, destination in enumerate(rows):
            transition[0, action, state, destination] = 30.0
    observer = torch.full(
        (
            1,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        ),
        -30.0,
    )
    for state in range(PRIMARY_STATES):
        observer[0, 0, state, state % PRIMARY_ANSWERS] = 30.0
        observer[
            0,
            1,
            state,
            (3 - state) % PRIMARY_ANSWERS,
        ] = 30.0
    closure = behavioral_closure(
        transition,
        observer,
        max_depth=2,
    )
    word_index = {
        word: index
        for index, word in enumerate(enumerate_action_words(2))
    }
    answers = closure.base.argmax(-1)
    assert int(answers[0, 0, word_index[(0, 1)], 0]) == 0
    assert int(answers[0, 0, word_index[(0, 1)], 1]) == 3
    assert int(answers[0, 0, word_index[(1, 0)], 0]) == 2
    assert int(answers[0, 0, word_index[(1, 0)], 1]) == 1
    assert int(
        closure.derivative[
            0,
            0,
            0,
            word_index[(1,)],
            0,
        ].argmax()
    ) == 0
    assert int(
        closure.derivative[
            0,
            1,
            0,
            word_index[(0,)],
            0,
        ].argmax()
    ) == 2
    control = behavioral_closure(
        transition,
        observer,
        max_depth=2,
        routing_mode="cyclic-control",
    )
    assert int(
        control.derivative[
            0,
            0,
            0,
            word_index[(1,)],
            0,
        ].argmax()
    ) == 2
    assert int(
        control.derivative[
            0,
            1,
            0,
            word_index[(0,)],
            0,
        ].argmax()
    ) == 0


def test_seal_contains_only_source_free_hard_machine_fields() -> None:
    transition, observer = _logits()
    sealed = seal_primary_machine(transition, observer)
    assert isinstance(sealed, HardFunctorMachine)
    assert tuple(field.name for field in fields(sealed)) == (
        "state_active",
        "action_active",
        "observer_active",
        "action_next",
        "observer_answer",
    )
    before = tuple(
        getattr(sealed, field.name).clone()
        for field in fields(sealed)
    )
    transition.fill_(float("nan"))
    observer.fill_(float("nan"))
    for field, expected in zip(fields(sealed), before):
        value = getattr(sealed, field.name)
        assert value.dtype == torch.uint8
        assert value.grad_fn is None
        torch.testing.assert_close(value, expected)


def test_seal_fails_closed_on_coordinate_dependent_ties() -> None:
    transition = torch.zeros(
        (1, PRIMARY_ACTIONS, PRIMARY_STATES, PRIMARY_STATES),
    )
    observer = torch.zeros(
        (
            1,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        ),
    )
    with pytest.raises(
        CausalSyndromeObserverError,
        match="transition sealing has a tied maximum",
    ):
        seal_primary_machine(transition, observer)
    transition[..., 0] = 1.0
    with pytest.raises(
        CausalSyndromeObserverError,
        match="observer sealing has a tied maximum",
    ):
        seal_primary_machine(transition, observer)


def test_tie_free_seal_is_exactly_recoding_equivariant() -> None:
    transition, observer = _logits(seed=29)
    state = torch.tensor((3, 0, 7, 1, 6, 2, 5, 4))
    action = torch.tensor((2, 0, 1))
    observers = torch.tensor((1, 0))
    answers = torch.tensor((2, 0, 3, 1))
    recoded_logits = _permute_machine_logits(
        transition,
        observer,
        state,
        action,
        observers,
        answers,
    )
    original = seal_primary_machine(transition, observer)
    recoded = seal_primary_machine(*recoded_logits)
    inverse_state = torch.argsort(state)
    inverse_action = torch.argsort(action)
    inverse_observer = torch.argsort(observers)
    expected_transition = original.action_next[
        :, :PRIMARY_ACTIONS, :PRIMARY_STATES
    ].index_select(1, inverse_action).index_select(2, inverse_state)
    expected_transition = state[expected_transition.long()].to(torch.uint8)
    expected_observer = original.observer_answer[
        :, :PRIMARY_OBSERVERS, :PRIMARY_STATES
    ].index_select(1, inverse_observer).index_select(2, inverse_state)
    expected_observer = answers[expected_observer.long()].to(torch.uint8)
    torch.testing.assert_close(
        recoded.action_next[:, :PRIMARY_ACTIONS, :PRIMARY_STATES],
        expected_transition,
    )
    torch.testing.assert_close(
        recoded.observer_answer[
            :, :PRIMARY_OBSERVERS, :PRIMARY_STATES
        ],
        expected_observer,
    )


def test_target_one_hot_geometry_is_supported() -> None:
    transition, observer = _logits()
    closure = behavioral_closure(transition, observer)
    base = F.one_hot(
        closure.base.argmax(-1),
        PRIMARY_ANSWERS,
    ).to(torch.float32)
    derivative = F.one_hot(
        closure.derivative.argmax(-1),
        PRIMARY_ANSWERS,
    ).to(torch.float32)
    result = explicit_causal_adjoint(
        transition,
        observer,
        base,
        derivative,
    )
    assert math.isfinite(float(result.base_innovation))
    assert math.isfinite(float(result.derivative_innovation))
