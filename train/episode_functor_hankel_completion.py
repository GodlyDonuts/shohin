"""Hankel-shift learned completion for the EFC no-host architecture.

Two independent permutation-equivariant relation predictors emit provisional
machines.  One defines anonymous state codewords by their finite future
behavior; the other defines action derivatives.  Final transitions are not
read directly from either provisional table: they are decoded by matching each
derivative signature to a base state signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_constrained_transport import (
    LawfulProjection,
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_learned_completion import (
    LearnedRelationalCompletionProjector,
)
from episode_functor_machine import (
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorMachine,
)
from pipeline.episode_functor_hankel_geometry import (
    commutative_bag_incidence,
    enumerate_action_words,
    prefix_shift_incidence,
    random_shift_incidence,
)


class HankelCompletionError(ValueError):
    """A neural behavioral code or projection contract failed."""


@dataclass(frozen=True, slots=True)
class NeuralHankelShiftResult:
    """Detailed attached result used for losses and causal diagnostics."""

    projection: LawfulProjection
    base_signatures: torch.Tensor
    derivative_signatures: torch.Tensor
    shift_distances: torch.Tensor
    shift_syndrome: torch.Tensor
    shift_incidence: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.projection.machine.batch_size
        if (
            self.base_signatures.ndim != 5
            or self.base_signatures.shape[0] != batch
            or self.base_signatures.shape[1] != PRIMARY_STATES
            or self.base_signatures.shape[3:] != (
                PRIMARY_OBSERVERS,
                PRIMARY_ANSWERS,
            )
            or self.derivative_signatures.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                self.base_signatures.shape[2],
                PRIMARY_OBSERVERS,
                PRIMARY_ANSWERS,
            )
            or self.shift_distances.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.shift_syndrome.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
            )
            or self.shift_incidence.shape
            != (
                PRIMARY_ACTIONS,
                self.base_signatures.shape[2],
            )
            or self.shift_incidence.dtype != torch.long
        ):
            raise HankelCompletionError("neural Hankel result geometry differs")


def _active_logits(
    *,
    batch: int,
    maximum: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    logits = torch.full(
        (batch, maximum, 2),
        -20.0,
        dtype=torch.float32,
        device=device,
    )
    logits[:, :, 0] = 20.0
    logits[:, :count, 0] = -20.0
    logits[:, :count, 1] = 20.0
    return logits


def _validate_provisional(
    transition: torch.Tensor,
    observer: torch.Tensor,
) -> int:
    if transition.ndim != 4:
        raise HankelCompletionError("provisional transition must be rank four")
    batch = int(transition.shape[0])
    if (
        transition.shape
        != (
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            PRIMARY_STATES,
        )
        or observer.shape
        != (
            batch,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        or not transition.is_floating_point()
        or not observer.is_floating_point()
        or transition.device != observer.device
        or not bool(torch.isfinite(transition).all())
        or not bool(torch.isfinite(observer).all())
    ):
        raise HankelCompletionError("provisional machine geometry differs")
    return batch


def _normalize_probabilities(
    value: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if bool(value.lt(0).any()):
        raise HankelCompletionError(f"{label} contains negative probability")
    denominator = value.sum(-1, keepdim=True)
    if not bool(torch.isfinite(denominator).all()) or not bool(
        denominator.gt(0).all()
    ):
        raise HankelCompletionError(f"{label} has empty categorical support")
    return value.float() / denominator.float()


def _future_signatures(
    transition: torch.Tensor,
    observer: torch.Tensor,
    *,
    max_depth: int,
) -> torch.Tensor:
    """Return ``[B,S,W,O,Y]`` soft future-observation signatures."""

    batch = _validate_provisional(transition, observer)
    transition = _normalize_probabilities(
        transition,
        label="provisional transition",
    )
    observer = _normalize_probabilities(
        observer,
        label="provisional observer",
    )
    words = enumerate_action_words(max_depth)
    identity = torch.eye(
        PRIMARY_STATES,
        dtype=transition.dtype,
        device=transition.device,
    )[None].expand(batch, -1, -1)
    distributions: dict[tuple[int, ...], torch.Tensor] = {(): identity}
    signatures: list[torch.Tensor] = []
    for word in words:
        if word:
            distributions[word] = torch.matmul(
                distributions[word[:-1]],
                transition[:, word[-1]],
            )
        signatures.append(
            torch.einsum(
                "bsj,bojy->bsoy",
                distributions[word],
                observer,
            )
        )
    return torch.stack(signatures, dim=2)


def _incidence_tensor(
    *,
    max_depth: int,
    mode: str,
    random_seed: str,
) -> torch.Tensor:
    if mode == "prefix":
        incidence = prefix_shift_incidence(max_depth)
    elif mode == "random":
        incidence = random_shift_incidence(
            max_depth,
            seed=random_seed,
        )
    elif mode == "commutative":
        incidence = commutative_bag_incidence(max_depth)
    else:
        raise HankelCompletionError(
            f"unknown Hankel incidence mode: {mode}"
        )
    return torch.tensor(incidence, dtype=torch.long)


def _gather_derivatives(
    extended: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,A,S,W,O,Y]`` signatures from an extended codebook."""

    if (
        extended.ndim != 5
        or incidence.ndim != 2
        or incidence.shape[0] != PRIMARY_ACTIONS
        or incidence.dtype != torch.long
        or bool(incidence.lt(0).any())
        or bool(incidence.ge(extended.shape[2]).any())
    ):
        raise HankelCompletionError("neural shift incidence differs")
    return torch.stack(
        tuple(
            extended.index_select(2, incidence[action])
            for action in range(PRIMARY_ACTIONS)
        ),
        dim=1,
    )


def _jensen_shannon_distances(
    derivative: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """Pair every ``(action, source)`` derivative with every base state."""

    left = derivative[:, :, :, None]
    right = base[:, None, None]
    midpoint = 0.5 * (left + right)
    tiny = torch.finfo(midpoint.dtype).tiny
    left_kl = (
        left
        * (
            left.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
    ).sum(-1)
    right_kl = (
        right
        * (
            right.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
    ).sum(-1)
    return 0.5 * (left_kl + right_kl).mean((-2, -1))


def _assert_unique_hardening(
    logits: torch.Tensor,
    *,
    label: str,
) -> None:
    top_two = logits.float().topk(2, dim=-1).values
    if bool(top_two[..., 0].eq(top_two[..., 1]).any()):
        raise HankelCompletionError(
            f"{label} Hankel hardening has a categorical tie"
        )


def _straight_through_logits(
    logits: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    _assert_unique_hardening(logits, label=label)
    hard = F.one_hot(
        logits.argmax(-1),
        logits.shape[-1],
    ).to(logits.dtype)
    hard_logits = hard.mul(40.0).sub(20.0)
    return hard_logits + logits - logits.detach()


def project_behavioral_shifts(
    *,
    base_transition: torch.Tensor,
    base_observer: torch.Tensor,
    derivative_transition: torch.Tensor,
    derivative_observer: torch.Tensor,
    max_depth: int,
    incidence: torch.Tensor,
    temperature: float,
    straight_through: bool = False,
) -> NeuralHankelShiftResult:
    """Decode a machine solely from finite behavioral-code agreement."""

    batch = _validate_provisional(base_transition, base_observer)
    if _validate_provisional(
        derivative_transition,
        derivative_observer,
    ) != batch:
        raise HankelCompletionError("base and derivative batches differ")
    if (
        not math.isfinite(temperature)
        or temperature <= 0.0
        or not 0 <= max_depth <= 6
    ):
        raise HankelCompletionError(
            "Hankel depth or temperature leaves support"
        )
    base_extended = _future_signatures(
        base_transition,
        base_observer,
        max_depth=max_depth + 1,
    )
    derivative_extended = _future_signatures(
        derivative_transition,
        derivative_observer,
        max_depth=max_depth + 1,
    )
    word_count = len(enumerate_action_words(max_depth))
    base_signatures = base_extended[:, :, :word_count]
    derivative_signatures = _gather_derivatives(
        derivative_extended,
        incidence.to(derivative_extended.device),
    )
    distances = _jensen_shannon_distances(
        derivative_signatures,
        base_signatures,
    )
    transition_logits = -distances / temperature
    tiny = torch.finfo(base_signatures.dtype).tiny
    observer_logits = (
        base_signatures[:, :, 0]
        .permute(0, 2, 1, 3)
        .clamp_min(tiny)
        .log()
    )
    if straight_through:
        transition_logits = _straight_through_logits(
            transition_logits,
            label="transition",
        )
        observer_logits = _straight_through_logits(
            observer_logits,
            label="observer",
        )
    transition_transport = transition_logits.float().softmax(-1)
    observer_transport = observer_logits.float().softmax(-1)
    action_next = torch.full(
        (batch, MAX_ACTIONS, MAX_STATES, MAX_STATES),
        -20.0,
        dtype=torch.float32,
        device=transition_logits.device,
    )
    action_next[
        :,
        :PRIMARY_ACTIONS,
        :PRIMARY_STATES,
        :PRIMARY_STATES,
    ] = transition_logits
    observer_answer = torch.full(
        (batch, MAX_OBSERVERS, MAX_STATES, MAX_ANSWERS),
        -20.0,
        dtype=torch.float32,
        device=observer_logits.device,
    )
    observer_answer[
        :,
        :PRIMARY_OBSERVERS,
        :PRIMARY_STATES,
        :PRIMARY_ANSWERS,
    ] = observer_logits
    machine = SoftFunctorMachine(
        state_active=_active_logits(
            batch=batch,
            maximum=MAX_STATES,
            count=PRIMARY_STATES,
            device=transition_logits.device,
        ),
        action_active=_active_logits(
            batch=batch,
            maximum=MAX_ACTIONS,
            count=PRIMARY_ACTIONS,
            device=transition_logits.device,
        ),
        observer_active=_active_logits(
            batch=batch,
            maximum=MAX_OBSERVERS,
            count=PRIMARY_OBSERVERS,
            device=transition_logits.device,
        ),
        action_next=action_next,
        observer_answer=observer_answer,
    )
    selected = distances.gather(
        -1,
        transition_logits.argmax(-1, keepdim=True),
    ).squeeze(-1)
    return NeuralHankelShiftResult(
        projection=LawfulProjection(
            machine=machine,
            transition_transport=transition_transport,
            observer_transport=observer_transport,
        ),
        base_signatures=base_signatures,
        derivative_signatures=derivative_signatures,
        shift_distances=distances,
        shift_syndrome=selected,
        shift_incidence=incidence.detach().clone(),
    )


class HankelShiftCompletionProjector(nn.Module):
    """Two-branch learned behavioral completer with a projector-compatible API."""

    def __init__(
        self,
        *,
        width: int = 96,
        iterations: int = 4,
        max_depth: int = 3,
        incidence_mode: str = "prefix",
        random_seed: str = "efc-hankel-random-control-v1",
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 <= max_depth <= 6:
            raise HankelCompletionError("Hankel depth leaves support")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise HankelCompletionError("Hankel temperature leaves support")
        self.base = LearnedRelationalCompletionProjector(
            width=width,
            iterations=iterations,
        )
        self.derivative = LearnedRelationalCompletionProjector(
            width=width,
            iterations=iterations,
        )
        self.max_depth = int(max_depth)
        self.incidence_mode = str(incidence_mode)
        self.random_seed = str(random_seed)
        self.temperature = float(temperature)
        self.decode_mode = "hankel-shift"
        self.sinkhorn_iterations = 0
        self.register_buffer(
            "shift_incidence",
            _incidence_tensor(
                max_depth=max_depth,
                mode=incidence_mode,
                random_seed=random_seed,
            ),
            persistent=True,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def detailed_forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> NeuralHankelShiftResult:
        base = self.base(transition_logits, observer_logits)
        derivative = self.derivative(
            transition_logits,
            observer_logits,
        )
        return project_behavioral_shifts(
            base_transition=base.transition_transport,
            base_observer=base.observer_transport,
            derivative_transition=derivative.transition_transport,
            derivative_observer=derivative.observer_transport,
            max_depth=self.max_depth,
            incidence=self.shift_incidence,
            temperature=self.temperature,
            straight_through=straight_through,
        )

    def forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> LawfulProjection:
        return self.detailed_forward(
            transition_logits,
            observer_logits,
            straight_through=straight_through,
        ).projection

    @torch.no_grad()
    def hard_project(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
    ) -> HardFunctorMachine:
        projection = self(
            transition_logits,
            observer_logits,
            straight_through=False,
        )
        _assert_unique_hardening(
            projection.machine.action_next[
                :,
                :PRIMARY_ACTIONS,
                :PRIMARY_STATES,
                :PRIMARY_STATES,
            ],
            label="transition",
        )
        _assert_unique_hardening(
            projection.machine.observer_answer[
                :,
                :PRIMARY_OBSERVERS,
                :PRIMARY_STATES,
                :PRIMARY_ANSWERS,
            ],
            label="observer",
        )
        return projection.machine.harden()


class DirectDualCompletionControlProjector(HankelShiftCompletionProjector):
    """Isoparametric control with identical signatures but direct decoding."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decode_mode = "direct-base"

    def detailed_forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> NeuralHankelShiftResult:
        base = self.base(
            transition_logits,
            observer_logits,
            straight_through=straight_through,
        )
        derivative = self.derivative(
            transition_logits,
            observer_logits,
        )
        diagnostics = project_behavioral_shifts(
            base_transition=base.transition_transport,
            base_observer=base.observer_transport,
            derivative_transition=derivative.transition_transport,
            derivative_observer=derivative.observer_transport,
            max_depth=self.max_depth,
            incidence=self.shift_incidence,
            temperature=self.temperature,
            straight_through=False,
        )
        return NeuralHankelShiftResult(
            projection=base,
            base_signatures=diagnostics.base_signatures,
            derivative_signatures=diagnostics.derivative_signatures,
            shift_distances=diagnostics.shift_distances,
            shift_syndrome=diagnostics.shift_syndrome,
            shift_incidence=diagnostics.shift_incidence,
        )


__all__ = [
    "DirectDualCompletionControlProjector",
    "HankelCompletionError",
    "HankelShiftCompletionProjector",
    "NeuralHankelShiftResult",
    "project_behavioral_shifts",
]
