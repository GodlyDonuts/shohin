from __future__ import annotations

import torch

from episode_functor_quotient_fisher_retraction import (
    QuotientFisherRetractionError,
    categorical_fisher_apply,
    categorical_fisher_pseudoinverse,
    quotient_fisher_direction,
    run_quotient_fisher_retraction,
)
import pytest
from pipeline.audit_episode_functor_acso_oracle_recovery import (
    _board,
    _fault_logits,
    _targets,
    deep_fault_inventory,
)


def _deep_fault_fixture(
    margin: float = 0.2,
):
    machine = next(
        machine
        for _, machine in sorted(_board().items())
        if deep_fault_inventory(machine)
    )
    fault = deep_fault_inventory(machine)[0]
    transition, observer = _fault_logits(machine, fault, margin)
    base, derivative = _targets(machine)
    return machine, fault, transition, observer, base, derivative


def test_fisher_pseudoinverse_round_trip_on_simplex_tangent() -> None:
    generator = torch.Generator().manual_seed(20260724)
    logits = torch.randn((2, 3, 8, 8), generator=generator)
    cotangent = torch.randn(logits.shape, generator=generator)
    cotangent -= cotangent.mean(-1, keepdim=True)
    raised = categorical_fisher_pseudoinverse(logits, cotangent)
    assert torch.allclose(
        categorical_fisher_apply(logits, raised),
        cotangent,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        raised.mean(-1),
        torch.zeros_like(raised.mean(-1)),
        atol=2e-5,
        rtol=0.0,
    )


def test_fisher_direction_is_invariant_to_row_constant_logit_gauge() -> None:
    (
        _,
        _,
        transition,
        observer,
        base,
        derivative,
    ) = _deep_fault_fixture()
    from episode_functor_causal_syndrome_observer import (
        explicit_causal_adjoint,
    )

    adjoint = explicit_causal_adjoint(
        transition,
        observer,
        base,
        derivative,
    )
    original = quotient_fisher_direction(
        transition,
        observer,
        adjoint.transition_logit_adjoint,
        adjoint.observer_logit_adjoint,
    )
    shifted = quotient_fisher_direction(
        transition + 7.25,
        observer - 3.5,
        adjoint.transition_logit_adjoint,
        adjoint.observer_logit_adjoint,
    )
    assert torch.allclose(
        original.transition,
        shifted.transition,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        original.observer,
        shifted.observer,
        atol=2e-6,
        rtol=2e-6,
    )


def test_fisher_direction_repairs_the_euclidean_sign_failure() -> None:
    (
        machine,
        fault,
        transition,
        observer,
        base,
        derivative,
    ) = _deep_fault_fixture()
    from episode_functor_causal_syndrome_observer import (
        explicit_causal_adjoint,
    )

    adjoint = explicit_causal_adjoint(
        transition,
        observer,
        base,
        derivative,
    )
    correct = machine.transitions[fault.action][fault.state]
    wrong = fault.wrong
    euclidean_descent_gap = (
        -adjoint.transition_logit_adjoint[
            0, fault.action, fault.state, correct
        ]
        + adjoint.transition_logit_adjoint[
            0, fault.action, fault.state, wrong
        ]
    )
    direction = quotient_fisher_direction(
        transition,
        observer,
        adjoint.transition_logit_adjoint,
        adjoint.observer_logit_adjoint,
    )
    fisher_descent_gap = (
        -direction.transition[
            0, fault.action, fault.state, correct
        ]
        + direction.transition[
            0, fault.action, fault.state, wrong
        ]
    )
    assert float(euclidean_descent_gap) < 0.0
    assert float(fisher_descent_gap) > 0.0


def test_four_fisher_cycles_recover_a_hard_deep_fault_monotonically() -> None:
    (
        machine,
        fault,
        transition,
        observer,
        base,
        derivative,
    ) = _deep_fault_fixture()
    result = run_quotient_fisher_retraction(
        transition,
        observer,
        base,
        derivative,
        cycles=4,
        step=1.0,
    )
    correct = machine.transitions[fault.action][fault.state]
    assert (
        int(
            result.transition_logits[
                0, fault.action, fault.state
            ].argmax()
        )
        == correct
    )
    totals = [
        float(cycle.base_innovation + cycle.derivative_innovation)
        for cycle in result.cycles
    ]
    assert all(
        right <= left + 1e-7
        for left, right in zip(totals, totals[1:])
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fisher_rejects_nonfinite_logits_and_cotangents(
    bad: float,
) -> None:
    logits = torch.zeros((1, 2), dtype=torch.float32)
    cotangent = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    bad_logits = logits.clone()
    bad_logits[0, 0] = bad
    with pytest.raises(QuotientFisherRetractionError):
        categorical_fisher_pseudoinverse(
            bad_logits,
            cotangent,
        )
    bad_cotangent = cotangent.clone()
    bad_cotangent[0, 0] = bad
    with pytest.raises(QuotientFisherRetractionError):
        categorical_fisher_pseudoinverse(
            logits,
            bad_cotangent,
        )


def test_fisher_rejects_off_simplex_cotangent() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    cotangent = torch.tensor(
        [[0.5, -0.25, -0.20]],
        dtype=torch.float32,
    )
    with pytest.raises(QuotientFisherRetractionError):
        categorical_fisher_pseudoinverse(logits, cotangent)


def test_fisher_rejects_extreme_softmax_underflow() -> None:
    logits = torch.tensor(
        [[0.0, -1.0e38]],
        dtype=torch.float32,
    )
    cotangent = torch.tensor(
        [[0.5, -0.5]],
        dtype=torch.float32,
    )
    with pytest.raises(QuotientFisherRetractionError):
        categorical_fisher_pseudoinverse(logits, cotangent)


def test_retraction_rejects_nonfinite_post_update() -> None:
    (
        _,
        _,
        transition,
        observer,
        base,
        derivative,
    ) = _deep_fault_fixture()
    with pytest.raises(QuotientFisherRetractionError):
        run_quotient_fisher_retraction(
            transition,
            observer,
            base,
            derivative,
            cycles=1,
            step=3.4e38,
        )
