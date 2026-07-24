"""Fail-closed train step for EFC-C source-compiler qualification.

This module performs no dataset generation, split access, artifact writing, or
job launch. A custody launcher must supply already isolated candidate and
supervisor batches plus a cryptographically connected Shohin system.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.episode_functor_qualification_boundary import (  # noqa: E402
    CandidateCompilerBatch,
)
from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    QualificationSupervisorBatch,
)
from episode_functor_learned_system import (  # noqa: E402
    LearnedEFCSystem,
)
from episode_functor_qualification_loss import (  # noqa: E402
    EFCQualificationLoss,
    QualificationExactMetrics,
)


class QualificationTrainerError(ValueError):
    """EFC-C optimizer, gradient, or custody precondition failed."""


@dataclass(frozen=True, slots=True)
class QualificationTrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    maximum_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.maximum_gradient_norm)
            or self.maximum_gradient_norm <= 0.0
        ):
            raise QualificationTrainerError(
                "qualification optimizer config differs"
            )


@dataclass(frozen=True, slots=True)
class QualificationStepReceipt:
    loss: float
    gradient_norm: float
    trainable_parameters: int
    optimizer_state_bytes: int
    exact_metrics: QualificationExactMetrics

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.loss)
            or self.loss < 0.0
            or not math.isfinite(self.gradient_norm)
            or self.gradient_norm < 0.0
            or self.trainable_parameters < 1
            or self.optimizer_state_bytes < 1
        ):
            raise QualificationTrainerError(
                "qualification step receipt differs"
            )


class EFCQualificationTrainer:
    """Optimizer custody for the source compiler only."""

    def __init__(
        self,
        system: LearnedEFCSystem,
        *,
        objective: EFCQualificationLoss | None = None,
        config: QualificationTrainerConfig | None = None,
        require_verified_trunk: bool = True,
    ) -> None:
        if not isinstance(system, LearnedEFCSystem):
            raise QualificationTrainerError(
                "qualification trainer system type differs"
            )
        self.system = system
        self.objective = (
            EFCQualificationLoss()
            if objective is None
            else objective
        )
        self.config = (
            QualificationTrainerConfig()
            if config is None
            else config
        )
        if require_verified_trunk and (
            self.system.parameter_receipt().integration_status
            != "connected"
        ):
            raise QualificationTrainerError(
                "qualification trainer requires the verified protected trunk"
            )
        if self.system.frozen_trunk is not None and any(
            parameter.requires_grad
            for parameter in self.system.frozen_trunk.parameters()
        ):
            raise QualificationTrainerError(
                "qualification frozen trunk has trainable parameters"
            )
        parameters = tuple(self.system.source_compiler.parameters())
        if not parameters or any(
            not parameter.requires_grad for parameter in parameters
        ):
            raise QualificationTrainerError(
                "qualification source compiler parameters differ"
            )
        self._parameters = parameters
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self._parameters)

    def _trunk_batch(self, candidate: CandidateCompilerBatch):
        return (
            None
            if self.system.frozen_trunk is None
            else candidate.trunk
        )

    def _optimizer_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for state in self.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )

    def train_step(
        self,
        candidate: CandidateCompilerBatch,
        supervisor: QualificationSupervisorBatch,
    ) -> QualificationStepReceipt:
        if not isinstance(candidate, CandidateCompilerBatch):
            raise QualificationTrainerError(
                "qualification candidate batch type differs"
            )
        self.system.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.system.compile_source(
            candidate.witness,
            straight_through=True,
            trunk_batch=self._trunk_batch(candidate),
        )
        losses = self.objective(
            output,
            supervisor,
            candidate_source_sha256=candidate.source_sha256,
        )
        losses.total.backward()
        if any(
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            for parameter in self._parameters
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise QualificationTrainerError(
                "qualification gradient is missing or nonfinite"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.maximum_gradient_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all())
            for parameter in self._parameters
        ):
            raise QualificationTrainerError(
                "qualification parameter became nonfinite"
            )
        metrics = self.objective.exact_metrics(
            output,
            supervisor,
            candidate_source_sha256=candidate.source_sha256,
        )
        state_bytes = self._optimizer_state_bytes()
        if state_bytes < 1:
            raise QualificationTrainerError(
                "qualification optimizer state was not materialized"
            )
        return QualificationStepReceipt(
            loss=float(losses.total.detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
            trainable_parameters=self.trainable_parameters,
            optimizer_state_bytes=state_bytes,
            exact_metrics=metrics,
        )

    @torch.no_grad()
    def evaluate(
        self,
        candidate: CandidateCompilerBatch,
        supervisor: QualificationSupervisorBatch,
    ) -> QualificationExactMetrics:
        if not isinstance(candidate, CandidateCompilerBatch):
            raise QualificationTrainerError(
                "qualification candidate batch type differs"
            )
        self.system.eval()
        output = self.system.compile_source(
            candidate.witness,
            straight_through=False,
            trunk_batch=self._trunk_batch(candidate),
        )
        return self.objective.exact_metrics(
            output,
            supervisor,
            candidate_source_sha256=candidate.source_sha256,
        )


__all__ = [
    "EFCQualificationTrainer",
    "QualificationStepReceipt",
    "QualificationTrainerConfig",
    "QualificationTrainerError",
]
