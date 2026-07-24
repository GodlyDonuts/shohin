"""Categorical quotient-Fisher retraction for EFC machine revision.

ACSO's explicit adjoint is a logit cotangent.  Treating it as a Euclidean
vector can point away from the correct deterministic simplex vertex.  This
module raises that cotangent with the categorical Fisher pseudoinverse, fixes
the row-constant logit gauge, and applies a bounded intrinsic update.

The mechanics are parameter-free and do not use runtime autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from episode_functor_causal_syndrome_observer import (
    causal_syndrome_innovation,
    explicit_causal_adjoint,
)
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from pipeline.episode_functor_hankel_geometry import enumerate_action_words


class QuotientFisherRetractionError(ValueError):
    """The categorical geometry or retraction contract failed."""


@dataclass(frozen=True, slots=True)
class QuotientFisherDirection:
    """Raised and row-normalized intrinsic update directions."""

    transition: torch.Tensor
    observer: torch.Tensor


@dataclass(frozen=True, slots=True)
class QuotientFisherCycle:
    """One pre-update causal-innovation receipt."""

    base_innovation: torch.Tensor
    derivative_innovation: torch.Tensor


@dataclass(frozen=True, slots=True)
class QuotientFisherResult:
    """Revised logits and the complete fixed-cycle receipt."""

    transition_logits: torch.Tensor
    observer_logits: torch.Tensor
    cycles: tuple[QuotientFisherCycle, ...]
    step: float
    routing_mode: str


def _validate_logits(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> int:
    if transition_logits.ndim != 4:
        raise QuotientFisherRetractionError(
            "transition logits must be rank four"
        )
    batch = int(transition_logits.shape[0])
    if (
        transition_logits.shape
        != (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            PRIMARY_STATES,
        )
        or observer_logits.shape
        != (
            batch,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        or not transition_logits.is_floating_point()
        or not observer_logits.is_floating_point()
        or transition_logits.dtype != torch.float32
        or observer_logits.dtype != torch.float32
        or transition_logits.dtype != observer_logits.dtype
        or transition_logits.device != observer_logits.device
        or not bool(torch.isfinite(transition_logits).all())
        or not bool(torch.isfinite(observer_logits).all())
    ):
        raise QuotientFisherRetractionError(
            "quotient-Fisher logit geometry differs"
        )
    return batch


def _validate_cotangent(
    logits: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    label: str,
) -> None:
    if (
        cotangent.shape != logits.shape
        or not logits.is_floating_point()
        or cotangent.dtype != logits.dtype
        or cotangent.device != logits.device
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(cotangent).all())
    ):
        raise QuotientFisherRetractionError(
            f"{label} cotangent geometry differs"
        )


def categorical_fisher_apply(
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    """Apply ``diag(p) - p p^T`` independently to each categorical row."""

    _validate_cotangent(logits, tangent, label="Fisher tangent")
    probabilities = logits.float().softmax(-1).to(logits.dtype)
    return probabilities * (
        tangent - (probabilities * tangent).sum(-1, keepdim=True)
    )


def categorical_fisher_pseudoinverse(
    logits: torch.Tensor,
    cotangent: torch.Tensor,
) -> torch.Tensor:
    """Raise a zero-sum cotangent and select the arithmetic-mean-zero gauge."""

    _validate_cotangent(logits, cotangent, label="Fisher")
    tolerance = 64.0 * torch.finfo(cotangent.dtype).eps
    if bool(cotangent.sum(-1).abs().gt(tolerance).any()):
        raise QuotientFisherRetractionError(
            "Fisher cotangent leaves the simplex tangent"
        )
    probabilities = logits.float().softmax(-1).to(logits.dtype)
    if bool(
        probabilities.eq(0)
        .logical_and(cotangent.abs().gt(tolerance))
        .any()
    ):
        raise QuotientFisherRetractionError(
            "Fisher cotangent leaves numerical softmax support"
        )
    raised = cotangent / probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    result = raised - raised.mean(-1, keepdim=True)
    if not bool(torch.isfinite(result).all()):
        raise QuotientFisherRetractionError(
            "Fisher pseudoinverse is nonfinite"
        )
    return result


def _row_normalize(direction: torch.Tensor) -> torch.Tensor:
    scale = direction.abs().amax(-1, keepdim=True)
    return torch.where(
        scale.gt(0),
        direction
        / scale.clamp_min(torch.finfo(direction.dtype).tiny),
        torch.zeros_like(direction),
    )


def quotient_fisher_direction(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    transition_cotangent: torch.Tensor,
    observer_cotangent: torch.Tensor,
) -> QuotientFisherDirection:
    """Return the row-normalized intrinsic descent direction."""

    _validate_logits(transition_logits, observer_logits)
    transition = categorical_fisher_pseudoinverse(
        transition_logits,
        transition_cotangent,
    )
    observer = categorical_fisher_pseudoinverse(
        observer_logits,
        observer_cotangent,
    )
    return QuotientFisherDirection(
        transition=_row_normalize(transition),
        observer=_row_normalize(observer),
    )


@torch.no_grad()
def run_quotient_fisher_retraction(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    target_base: torch.Tensor,
    target_derivative: torch.Tensor,
    *,
    max_depth: int = 3,
    routing_mode: str = "causal",
    cycles: int = 4,
    step: float = 1.0,
) -> QuotientFisherResult:
    """Run a fixed number of explicit intrinsic causal-revision cycles."""

    batch = _validate_logits(transition_logits, observer_logits)
    word_count = len(enumerate_action_words(max_depth))
    if (
        target_base.shape
        != (
            batch,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        )
        or target_derivative.shape
        != (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        )
        or target_base.dtype != transition_logits.dtype
        or target_derivative.dtype != transition_logits.dtype
        or target_base.device != transition_logits.device
        or target_derivative.device != transition_logits.device
        or not bool(torch.isfinite(target_base).all())
        or not bool(torch.isfinite(target_derivative).all())
        or routing_mode not in {"causal", "one-step-control"}
        or not 1 <= cycles <= 16
        or not math.isfinite(step)
        or step <= 0.0
        or step > 16.0
    ):
        raise QuotientFisherRetractionError(
            "quotient-Fisher run contract differs"
        )
    transition = transition_logits.detach().clone()
    observer = observer_logits.detach().clone()
    receipts = []
    for _ in range(cycles):
        base_value, derivative_value = causal_syndrome_innovation(
            transition,
            observer,
            target_base,
            target_derivative,
            max_depth=max_depth,
            routing_mode=routing_mode,
        )
        if (
            not bool(torch.isfinite(base_value).all())
            or not bool(torch.isfinite(derivative_value).all())
        ):
            raise QuotientFisherRetractionError(
                "quotient-Fisher innovation is nonfinite"
            )
        receipts.append(
            QuotientFisherCycle(
                base_innovation=base_value.detach().clone(),
                derivative_innovation=derivative_value.detach().clone(),
            )
        )
        adjoint = explicit_causal_adjoint(
            transition,
            observer,
            target_base,
            target_derivative,
            max_depth=max_depth,
            routing_mode=routing_mode,
        )
        direction = quotient_fisher_direction(
            transition,
            observer,
            adjoint.transition_logit_adjoint,
            adjoint.observer_logit_adjoint,
        )
        transition = transition - step * direction.transition
        observer = observer - step * direction.observer
        if (
            not bool(torch.isfinite(transition).all())
            or not bool(torch.isfinite(observer).all())
        ):
            raise QuotientFisherRetractionError(
                "quotient-Fisher update is nonfinite"
            )
    base_value, derivative_value = causal_syndrome_innovation(
        transition,
        observer,
        target_base,
        target_derivative,
        max_depth=max_depth,
        routing_mode=routing_mode,
    )
    if (
        not bool(torch.isfinite(base_value).all())
        or not bool(torch.isfinite(derivative_value).all())
    ):
        raise QuotientFisherRetractionError(
            "quotient-Fisher innovation is nonfinite"
        )
    receipts.append(
        QuotientFisherCycle(
            base_innovation=base_value.detach().clone(),
            derivative_innovation=derivative_value.detach().clone(),
        )
    )
    return QuotientFisherResult(
        transition_logits=transition,
        observer_logits=observer,
        cycles=tuple(receipts),
        step=float(step),
        routing_mode=routing_mode,
    )


__all__ = [
    "QuotientFisherCycle",
    "QuotientFisherDirection",
    "QuotientFisherResult",
    "QuotientFisherRetractionError",
    "categorical_fisher_apply",
    "categorical_fisher_pseudoinverse",
    "quotient_fisher_direction",
    "run_quotient_fisher_retraction",
]
