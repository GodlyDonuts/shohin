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
from pipeline.episode_functor_qualification_custody import (  # noqa: E402
    QualificationSplitCustody,
)
from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    QualificationSupervisorBatch,
)
from episode_functor_learned_system import (  # noqa: E402
    LearnedEFCSystem,
)
from episode_functor_hankel_completion import (  # noqa: E402
    HankelShiftCompletionProjector,
    NeuralHankelShiftResult,
)
from episode_functor_qualification_loss import (  # noqa: E402
    EFCHankelQualificationLoss,
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
    training_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.loss)
            or self.loss < 0.0
            or not math.isfinite(self.gradient_norm)
            or self.gradient_norm < 0.0
            or self.trainable_parameters < 1
            or self.optimizer_state_bytes < 1
            or len(self.training_manifest_sha256) != 64
        ):
            raise QualificationTrainerError(
                "qualification step receipt differs"
            )


@dataclass(frozen=True, slots=True)
class QualificationEvaluationReceipt:
    split: str
    manifest_sha256: str
    exact_metrics: QualificationExactMetrics


class EFCQualificationTrainer:
    """Optimizer custody for the source compiler only."""

    def __init__(
        self,
        system: LearnedEFCSystem,
        *,
        objective: EFCQualificationLoss | None = None,
        config: QualificationTrainerConfig | None = None,
        training_custody: QualificationSplitCustody,
        require_verified_trunk: bool = True,
    ) -> None:
        if not isinstance(system, LearnedEFCSystem):
            raise QualificationTrainerError(
                "qualification trainer system type differs"
            )
        self.system = system
        if not isinstance(training_custody, QualificationSplitCustody):
            raise QualificationTrainerError(
                "qualification training custody differs"
            )
        training_custody.assert_training_split()
        self.training_custody = training_custody
        self.objective = (
            EFCQualificationLoss()
            if objective is None
            else objective
        )
        uses_hankel_projector = isinstance(
            self.system.source_compiler.projector,
            HankelShiftCompletionProjector,
        )
        uses_hankel_objective = isinstance(
            self.objective,
            EFCHankelQualificationLoss,
        )
        if uses_hankel_projector != uses_hankel_objective:
            raise QualificationTrainerError(
                "Hankel projector and qualification objective must be paired"
            )
        self._hankel_incidence = (
            self.system.source_compiler.projector.shift_incidence.detach().clone()
            if uses_hankel_projector
            else None
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

    def _assert_hankel_custody(self, output) -> None:
        if self._hankel_incidence is None:
            return
        projector = self.system.source_compiler.projector
        details = output.projector_auxiliary
        if (
            not isinstance(projector, HankelShiftCompletionProjector)
            or not isinstance(details, NeuralHankelShiftResult)
            or not torch.equal(
                projector.shift_incidence,
                self._hankel_incidence.to(projector.shift_incidence.device),
            )
            or not torch.equal(
                details.shift_incidence,
                self._hankel_incidence.to(details.shift_incidence.device),
            )
        ):
            raise QualificationTrainerError(
                "Hankel incidence changed after trainer custody"
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
        self.training_custody.assert_batches(candidate, supervisor)
        self.system.source_compiler.train()
        self.system.query_parser.eval()
        if self.system.frozen_trunk is not None:
            self.system.frozen_trunk.eval()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.system.compile_source(
            candidate.witness,
            straight_through=True,
            trunk_batch=self._trunk_batch(candidate),
        )
        self._assert_hankel_custody(output)
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
        self.system.source_compiler.eval()
        with torch.no_grad():
            post_update_output = self.system.compile_source(
                candidate.witness,
                straight_through=False,
                trunk_batch=self._trunk_batch(candidate),
            )
            self._assert_hankel_custody(post_update_output)
            metrics = self.objective.exact_metrics(
                post_update_output,
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
            training_manifest_sha256=(
                self.training_custody.receipt_sha256
            ),
        )

    @torch.no_grad()
    def evaluate(
        self,
        candidate: CandidateCompilerBatch,
        supervisor: QualificationSupervisorBatch,
        *,
        custody: QualificationSplitCustody,
    ) -> QualificationEvaluationReceipt:
        if not isinstance(candidate, CandidateCompilerBatch):
            raise QualificationTrainerError(
                "qualification candidate batch type differs"
            )
        if not isinstance(custody, QualificationSplitCustody):
            raise QualificationTrainerError(
                "qualification evaluation custody differs"
            )
        custody.assert_batches(candidate, supervisor)
        self.system.eval()
        output = self.system.compile_source(
            candidate.witness,
            straight_through=False,
            trunk_batch=self._trunk_batch(candidate),
        )
        self._assert_hankel_custody(output)
        return QualificationEvaluationReceipt(
            split=custody.split,
            manifest_sha256=custody.receipt_sha256,
            exact_metrics=self.objective.exact_metrics(
                output,
                supervisor,
                candidate_source_sha256=candidate.source_sha256,
            ),
        )


__all__ = [
    "EFCQualificationTrainer",
    "QualificationEvaluationReceipt",
    "QualificationStepReceipt",
    "QualificationTrainerConfig",
    "QualificationTrainerError",
]
