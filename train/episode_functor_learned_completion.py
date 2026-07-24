"""Permutation-equivariant learned completion for the EFC no-host arm.

Unlike ``LawfulMachineProjector``, this module does not enforce permutation or
observer-balance constraints.  It shares a relational update across cells and
must learn any completion law from train-only supervision.  The fixed
categorical executor remains unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from episode_functor_constrained_transport import (
    LawfulProjection,
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import (
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorMachine,
)


class LearnedCompletionError(ValueError):
    """Learned relational completion geometry or values failed."""


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


class _EquivariantRelationCompleter(nn.Module):
    """Shared row/column message passing without coordinate embeddings."""

    def __init__(
        self,
        *,
        width: int,
        iterations: int,
    ) -> None:
        super().__init__()
        if width < 16 or iterations < 1:
            raise LearnedCompletionError(
                "learned relation completer geometry differs"
            )
        self.width = int(width)
        self.iterations = int(iterations)
        self.input = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, evidence_logits: torch.Tensor) -> torch.Tensor:
        if (
            evidence_logits.ndim != 4
            or not evidence_logits.is_floating_point()
            or not bool(torch.isfinite(evidence_logits).all())
        ):
            raise LearnedCompletionError(
                "completion evidence must be finite rank-four logits"
            )
        probabilities = evidence_logits.float().softmax(-1)
        tiny = torch.finfo(probabilities.dtype).tiny
        log_probabilities = probabilities.clamp_min(tiny).log()
        row_max = probabilities.amax(-1, keepdim=True).expand_as(
            probabilities
        )
        row_entropy = -(
            probabilities * log_probabilities
        ).sum(-1, keepdim=True).expand_as(probabilities)
        column_mass = probabilities.sum(-2, keepdim=True).expand_as(
            probabilities
        )
        relation_mean = probabilities.mean(
            (-2, -1),
            keepdim=True,
        ).expand_as(probabilities)
        hidden = self.input(
            torch.stack(
                (
                    probabilities,
                    log_probabilities,
                    row_max,
                    row_entropy,
                    column_mass - relation_mean,
                ),
                dim=-1,
            )
        )
        for _ in range(self.iterations):
            row_context = hidden.mean(-2, keepdim=True).expand_as(hidden)
            column_context = hidden.mean(-3, keepdim=True).expand_as(hidden)
            global_context = hidden.mean(
                (-3, -2),
                keepdim=True,
            ).expand_as(hidden)
            update = self.update(
                torch.cat(
                    (
                        hidden,
                        row_context,
                        column_context,
                        global_context,
                    ),
                    dim=-1,
                )
            )
            hidden = hidden + update / math.sqrt(self.iterations)
        return self.output(hidden).squeeze(-1)


class LearnedRelationalCompletionProjector(nn.Module):
    """No-host learned completion with the projector-compatible API."""

    def __init__(
        self,
        *,
        width: int = 96,
        iterations: int = 4,
    ) -> None:
        super().__init__()
        self.transition = _EquivariantRelationCompleter(
            width=width,
            iterations=iterations,
        )
        self.observer = _EquivariantRelationCompleter(
            width=width,
            iterations=iterations,
        )
        self.sinkhorn_iterations = 0

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _check_inputs(
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
    ) -> int:
        if transition_logits.ndim != 4:
            raise LearnedCompletionError(
                "transition evidence must be rank four"
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
            raise LearnedCompletionError(
                "learned completion input geometry differs"
            )
        return batch

    @staticmethod
    def _assert_unique_hardening(
        logits: torch.Tensor,
        *,
        label: str,
    ) -> None:
        top_two = logits.float().topk(2, dim=-1).values
        if bool(top_two[..., 0].eq(top_two[..., 1]).any()):
            raise LearnedCompletionError(
                f"{label} completion has a categorical hardening tie"
            )

    @classmethod
    def _straight_through_logits(
        cls,
        logits: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        cls._assert_unique_hardening(logits, label=label)
        hard_index = logits.argmax(-1)
        hard = torch.full_like(logits, -20.0)
        hard.scatter_(-1, hard_index[..., None], 20.0)
        return hard + logits - logits.detach()

    def assert_machine_hardening_well_defined(
        self,
        machine: SoftFunctorMachine,
    ) -> None:
        """Reject coordinate-dependent categorical tie breaking."""
        self._assert_unique_hardening(
            machine.action_next[
                :,
                :PRIMARY_ACTIONS,
                :PRIMARY_STATES,
                :PRIMARY_STATES,
            ],
            label="transition",
        )
        self._assert_unique_hardening(
            machine.observer_answer[
                :,
                :PRIMARY_OBSERVERS,
                :PRIMARY_STATES,
                :PRIMARY_ANSWERS,
            ],
            label="observer",
        )

    def forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> LawfulProjection:
        batch = self._check_inputs(
            transition_logits,
            observer_logits,
        )
        completed_transition = self.transition(transition_logits)
        completed_observer = self.observer(observer_logits)
        if straight_through:
            completed_transition = self._straight_through_logits(
                completed_transition,
                label="transition",
            )
            completed_observer = self._straight_through_logits(
                completed_observer,
                label="observer",
            )

        action_next = torch.full(
            (
                batch,
                MAX_ACTIONS,
                MAX_STATES,
                MAX_STATES,
            ),
            -20.0,
            dtype=torch.float32,
            device=transition_logits.device,
        )
        action_next[
            :,
            :PRIMARY_ACTIONS,
            :PRIMARY_STATES,
            :PRIMARY_STATES,
        ] = completed_transition
        observer_answer = torch.full(
            (
                batch,
                MAX_OBSERVERS,
                MAX_STATES,
                MAX_ANSWERS,
            ),
            -20.0,
            dtype=torch.float32,
            device=observer_logits.device,
        )
        observer_answer[
            :,
            :PRIMARY_OBSERVERS,
            :PRIMARY_STATES,
            :PRIMARY_ANSWERS,
        ] = completed_observer
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
        return LawfulProjection(
            machine=machine,
            transition_transport=completed_transition.float().softmax(-1),
            observer_transport=completed_observer.float().softmax(-1),
        )

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
        self.assert_machine_hardening_well_defined(projection.machine)
        return projection.machine.harden()


__all__ = [
    "LearnedCompletionError",
    "LearnedRelationalCompletionProjector",
]
