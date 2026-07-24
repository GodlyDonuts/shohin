"""Explicit causal-adjoint mechanics for HSC machine revision.

The module computes a manual reverse dynamic program through the finite
behavioral closure. It does not use runtime autograd for the adjoint and does
not fit, seal, or score a neural candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from episode_functor_capacity_lanes import (
    HANKEL_SHIFT_MAXIMUM_EXPECTED,
)
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import (
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from pipeline.episode_functor_hankel_geometry import (
    enumerate_action_words,
)


ACSO_ADDED_PARAMETERS = 3_995_137
ACSO_COMPLETE_PARAMETERS = (
    HANKEL_SHIFT_MAXIMUM_EXPECTED.complete_parameters
    + ACSO_ADDED_PARAMETERS
)
ACSO_HEADROOM = (
    HANKEL_SHIFT_MAXIMUM_EXPECTED.complete_parameters
    + HANKEL_SHIFT_MAXIMUM_EXPECTED.headroom
    - ACSO_COMPLETE_PARAMETERS
)


class CausalSyndromeObserverError(ValueError):
    """The causal-syndrome geometry or numeric contract failed."""


@dataclass(frozen=True, slots=True)
class BehavioralClosure:
    """Model-implied base and action-derivative future signatures."""

    base: torch.Tensor
    derivative: torch.Tensor


@dataclass(frozen=True, slots=True)
class CausalAdjoint:
    """Innovation values and explicit adjoints with respect to logits."""

    base_innovation: torch.Tensor
    derivative_innovation: torch.Tensor
    transition_logit_adjoint: torch.Tensor
    observer_logit_adjoint: torch.Tensor
    routing_mode: str

    def __post_init__(self) -> None:
        batch = int(self.transition_logit_adjoint.shape[0])
        if (
            self.base_innovation.shape != ()
            or self.derivative_innovation.shape != ()
            or not bool(torch.isfinite(self.base_innovation))
            or not bool(torch.isfinite(self.derivative_innovation))
            or self.transition_logit_adjoint.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.observer_logit_adjoint.shape
            != (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            or not bool(
                torch.isfinite(self.transition_logit_adjoint).all()
            )
            or not bool(
                torch.isfinite(self.observer_logit_adjoint).all()
            )
            or self.routing_mode not in {"causal", "cyclic-control"}
        ):
            raise CausalSyndromeObserverError(
                "causal adjoint receipt geometry differs"
            )


def _validate_logits(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> int:
    if transition_logits.ndim != 4:
        raise CausalSyndromeObserverError(
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
        or transition_logits.device != observer_logits.device
        or not bool(torch.isfinite(transition_logits).all())
        or not bool(torch.isfinite(observer_logits).all())
    ):
        raise CausalSyndromeObserverError(
            "causal-syndrome logit geometry differs"
        )
    return batch


def _probabilities(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logits(transition_logits, observer_logits)
    return (
        transition_logits.float().softmax(-1),
        observer_logits.float().softmax(-1),
    )


def _closure_tables(
    transition: torch.Tensor,
    observer: torch.Tensor,
    *,
    max_depth: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    dict[tuple[int, ...], torch.Tensor],
    dict[tuple[int, ...], torch.Tensor],
]:
    if not 0 <= max_depth <= 6:
        raise CausalSyndromeObserverError(
            "causal-syndrome depth leaves support"
        )
    batch = int(transition.shape[0])
    words = enumerate_action_words(max_depth)
    extended = enumerate_action_words(max_depth + 1)
    distributions: dict[tuple[int, ...], torch.Tensor] = {
        (): torch.eye(
            PRIMARY_STATES,
            dtype=transition.dtype,
            device=transition.device,
        )[None].expand(batch, -1, -1)
    }
    signatures: dict[tuple[int, ...], torch.Tensor] = {}
    for word in extended:
        if word:
            distributions[word] = torch.matmul(
                distributions[word[:-1]],
                transition[:, word[-1]],
            )
        signatures[word] = torch.einsum(
            "bsj,bqjy->bsqy",
            distributions[word],
            observer,
        )
    return words, distributions, signatures


def behavioral_closure(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    *,
    max_depth: int = 3,
    routing_mode: str = "causal",
) -> BehavioralClosure:
    """Return the differentiable model-implied behavioral signatures."""

    if routing_mode not in {"causal", "cyclic-control"}:
        raise CausalSyndromeObserverError(
            "causal-syndrome routing mode differs"
        )
    transition, observer = _probabilities(
        transition_logits,
        observer_logits,
    )
    words, _, signatures = _closure_tables(
        transition,
        observer,
        max_depth=max_depth,
    )
    base = torch.stack(
        tuple(signatures[word] for word in words),
        dim=2,
    )
    derivative = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    signatures[
                        (
                            (action, *word)
                            if routing_mode == "causal"
                            else cyclic_control_word((action, *word))
                        )
                    ]
                    for word in words
                ),
                dim=2,
            )
            for action in range(PRIMARY_ACTIONS)
        ),
        dim=1,
    )
    return BehavioralClosure(base=base, derivative=derivative)


def _normalize_targets(
    targets: torch.Tensor,
    expected: tuple[int, ...],
    *,
    label: str,
) -> torch.Tensor:
    if (
        targets.shape != expected
        or not targets.is_floating_point()
        or not bool(torch.isfinite(targets).all())
        or bool(targets.lt(0).any())
    ):
        raise CausalSyndromeObserverError(
            f"{label} target geometry differs"
        )
    denominator = targets.float().sum(-1, keepdim=True)
    if not bool(denominator.gt(0).all()):
        raise CausalSyndromeObserverError(
            f"{label} target support is empty"
        )
    return targets.float() / denominator


def _js_value_and_model_gradient(
    model: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tiny = torch.finfo(model.dtype).tiny
    midpoint = 0.5 * (model + target)
    model_term = model * (
        model.clamp_min(tiny).log()
        - midpoint.clamp_min(tiny).log()
    )
    target_term = torch.where(
        target.gt(0),
        target
        * (
            target.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        ),
        torch.zeros_like(target),
    )
    distributions = model.numel() // model.shape[-1]
    value = 0.5 * (model_term + target_term).sum(-1).mean()
    gradient = (
        0.5
        * (
            model.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
        / distributions
    )
    return value, gradient


def cyclic_control_word(word: tuple[int, ...]) -> tuple[int, ...]:
    """Rotate only multi-step action words, preserving depth and equivariance."""

    if len(word) < 2:
        return word
    return (*word[1:], word[0])


def _softmax_adjoint(
    probabilities: torch.Tensor,
    probability_adjoint: torch.Tensor,
) -> torch.Tensor:
    centered = probability_adjoint - (
        probability_adjoint * probabilities
    ).sum(-1, keepdim=True)
    return probabilities * centered


def causal_syndrome_innovation(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    target_base: torch.Tensor,
    target_derivative: torch.Tensor,
    *,
    max_depth: int = 3,
    routing_mode: str = "causal",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the scalar innovations used to validate the manual adjoint."""

    if routing_mode not in {"causal", "cyclic-control"}:
        raise CausalSyndromeObserverError(
            "causal-syndrome routing mode differs"
        )
    transition, observer = _probabilities(
        transition_logits,
        observer_logits,
    )
    words, _, signatures = _closure_tables(
        transition,
        observer,
        max_depth=max_depth,
    )
    batch = int(transition_logits.shape[0])
    word_count = len(words)
    base_target = _normalize_targets(
        target_base,
        (
            batch,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        ),
        label="base",
    )
    derivative_target = _normalize_targets(
        target_derivative,
        (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        ),
        label="derivative",
    )
    base_model = torch.stack(
        tuple(signatures[word] for word in words),
        dim=2,
    )
    derivative_model = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    signatures[
                        (
                            (action, *word)
                            if routing_mode == "causal"
                            else cyclic_control_word((action, *word))
                        )
                    ]
                    for word in words
                ),
                dim=2,
            )
            for action in range(PRIMARY_ACTIONS)
        ),
        dim=1,
    )
    base_value, _ = _js_value_and_model_gradient(
        base_model,
        base_target,
    )
    derivative_value, _ = _js_value_and_model_gradient(
        derivative_model,
        derivative_target,
    )
    return base_value, derivative_value


@torch.no_grad()
def explicit_causal_adjoint(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
    target_base: torch.Tensor,
    target_derivative: torch.Tensor,
    *,
    max_depth: int = 3,
    routing_mode: str = "causal",
) -> CausalAdjoint:
    """Manually backpropagate behavioral innovations to machine logits."""

    if routing_mode not in {"causal", "cyclic-control"}:
        raise CausalSyndromeObserverError(
            "causal-syndrome routing mode differs"
        )
    batch = _validate_logits(transition_logits, observer_logits)
    transition, observer = _probabilities(
        transition_logits,
        observer_logits,
    )
    words, distributions, signatures = _closure_tables(
        transition,
        observer,
        max_depth=max_depth,
    )
    word_count = len(words)
    base_target = _normalize_targets(
        target_base,
        (
            batch,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        ),
        label="base",
    )
    derivative_target = _normalize_targets(
        target_derivative,
        (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            word_count,
            PRIMARY_OBSERVERS,
            PRIMARY_ANSWERS,
        ),
        label="derivative",
    )
    base_model = torch.stack(
        tuple(signatures[word] for word in words),
        dim=2,
    )
    derivative_words = tuple(
        tuple(
            (
                (action, *word)
                if routing_mode == "causal"
                else cyclic_control_word((action, *word))
            )
            for word in words
        )
        for action in range(PRIMARY_ACTIONS)
    )
    derivative_model = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    signatures[word]
                    for word in derivative_words[action]
                ),
                dim=2,
            )
            for action in range(PRIMARY_ACTIONS)
        ),
        dim=1,
    )
    base_value, base_gradient = _js_value_and_model_gradient(
        base_model,
        base_target,
    )
    derivative_value, derivative_gradient = (
        _js_value_and_model_gradient(
            derivative_model,
            derivative_target,
        )
    )
    extended = enumerate_action_words(max_depth + 1)
    signature_adjoint = {
        word: torch.zeros_like(signatures[word])
        for word in extended
    }
    for index, word in enumerate(words):
        signature_adjoint[word] = (
            signature_adjoint[word] + base_gradient[:, :, index]
    )
    for action in range(PRIMARY_ACTIONS):
        for index, routed_word in enumerate(
            derivative_words[action]
        ):
            signature_adjoint[routed_word] = (
                signature_adjoint[routed_word]
                + derivative_gradient[:, action, :, index]
            )

    distribution_adjoint = {
        word: torch.einsum(
            "bsqy,bqjy->bsj",
            signature_adjoint[word],
            observer,
        )
        for word in extended
    }
    observer_adjoint = sum(
        (
            torch.einsum(
                "bsj,bsqy->bqjy",
                distributions[word],
                signature_adjoint[word],
            )
            for word in extended
        ),
        torch.zeros_like(observer),
    )
    transition_adjoint = torch.zeros_like(transition)
    for word in reversed(extended[1:]):
        action = word[-1]
        prefix = word[:-1]
        transition_adjoint[:, action] = (
            transition_adjoint[:, action]
            + torch.einsum(
                "bsi,bsj->bij",
                distributions[prefix],
                distribution_adjoint[word],
            )
        )
        distribution_adjoint[prefix] = (
            distribution_adjoint[prefix]
            + torch.einsum(
                "bsj,bij->bsi",
                distribution_adjoint[word],
                transition[:, action],
            )
        )
    transition_logit_adjoint = _softmax_adjoint(
        transition,
        transition_adjoint,
    )
    observer_logit_adjoint = _softmax_adjoint(
        observer,
        observer_adjoint,
    )
    if (
        not bool(torch.isfinite(transition_logit_adjoint).all())
        or not bool(torch.isfinite(observer_logit_adjoint).all())
    ):
        raise CausalSyndromeObserverError(
            "causal-syndrome adjoint is nonfinite"
        )
    return CausalAdjoint(
        base_innovation=base_value,
        derivative_innovation=derivative_value,
        transition_logit_adjoint=transition_logit_adjoint,
        observer_logit_adjoint=observer_logit_adjoint,
        routing_mode=routing_mode,
    )


def _unique_argmax(
    logits: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    top = logits.topk(2, dim=-1).values
    if bool(top[..., 0].eq(top[..., 1]).any()):
        raise CausalSyndromeObserverError(
            f"{label} sealing has a tied maximum"
        )
    return logits.argmax(-1)


@torch.no_grad()
def seal_primary_machine(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> HardFunctorMachine:
    """Harden only ordinary machine fields for the detached evaluator."""

    batch = _validate_logits(transition_logits, observer_logits)
    device = transition_logits.device
    state_active = torch.zeros(
        (batch, MAX_STATES),
        dtype=torch.uint8,
        device=device,
    )
    action_active = torch.zeros(
        (batch, MAX_ACTIONS),
        dtype=torch.uint8,
        device=device,
    )
    observer_active = torch.zeros(
        (batch, MAX_OBSERVERS),
        dtype=torch.uint8,
        device=device,
    )
    state_active[:, :PRIMARY_STATES] = 1
    action_active[:, :PRIMARY_ACTIONS] = 1
    observer_active[:, :PRIMARY_OBSERVERS] = 1
    action_next = torch.zeros(
        (batch, MAX_ACTIONS, MAX_STATES),
        dtype=torch.uint8,
        device=device,
    )
    observer_answer = torch.zeros(
        (batch, MAX_OBSERVERS, MAX_STATES),
        dtype=torch.uint8,
        device=device,
    )
    action_next[:, :PRIMARY_ACTIONS, :PRIMARY_STATES] = (
        _unique_argmax(
            transition_logits,
            label="transition",
        ).to(torch.uint8)
    )
    observer_answer[
        :, :PRIMARY_OBSERVERS, :PRIMARY_STATES
    ] = _unique_argmax(
        observer_logits,
        label="observer",
    ).to(torch.uint8)
    return HardFunctorMachine(
        state_active=state_active,
        action_active=action_active,
        observer_active=observer_active,
        action_next=action_next,
        observer_answer=observer_answer,
    )


class AdjointCausalSyndromePreconditioner(nn.Module):
    """Shared learned positive step controller for explicit adjoint updates."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(10, 384),
            nn.SiLU(),
            nn.Linear(384, 384),
        )
        self.recurrent_cell = nn.GRUCell(384, 768)
        self.step_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1536),
            nn.SiLU(),
            nn.Linear(1536, 1),
        )
        if self.parameter_count() != ACSO_ADDED_PARAMETERS:
            raise CausalSyndromeObserverError(
                "ACSO parameter receipt differs"
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 3
            or features.shape[-1] != 10
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all())
        ):
            raise CausalSyndromeObserverError(
                "ACSO feature geometry differs"
            )
        batch, cells, _ = features.shape
        encoded = self.feature_encoder(features).reshape(
            batch * cells,
            384,
        )
        if hidden is None:
            flat_hidden = torch.zeros(
                (batch * cells, 768),
                dtype=encoded.dtype,
                device=encoded.device,
            )
        else:
            if (
                hidden.shape != (batch, cells, 768)
                or hidden.dtype != encoded.dtype
                or hidden.device != encoded.device
                or not bool(torch.isfinite(hidden).all())
            ):
                raise CausalSyndromeObserverError(
                    "ACSO hidden-state geometry differs"
                )
            flat_hidden = hidden.reshape(batch * cells, 768)
        next_hidden = self.recurrent_cell(encoded, flat_hidden)
        step = (
            0.001
            + 0.099 * torch.sigmoid(self.step_head(next_hidden))
        ).reshape(batch, cells)
        return step, next_hidden.reshape(batch, cells, 768)


__all__ = [
    "ACSO_ADDED_PARAMETERS",
    "ACSO_COMPLETE_PARAMETERS",
    "ACSO_HEADROOM",
    "AdjointCausalSyndromePreconditioner",
    "BehavioralClosure",
    "CausalAdjoint",
    "CausalSyndromeObserverError",
    "behavioral_closure",
    "causal_syndrome_innovation",
    "cyclic_control_word",
    "explicit_causal_adjoint",
    "seal_primary_machine",
]
