"""Fail-closed train step for EFC-C source-compiler qualification.

This module performs no dataset generation, split access, artifact writing, or
job launch. A custody launcher must supply already isolated candidate and
supervisor batches plus a cryptographically connected Shohin system.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
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
from pipeline.episode_functor_qualification_batch import (  # noqa: E402
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
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool = False
    maximum_updates: int = 1
    autocast_dtype: str = "none"
    tf32: bool = False
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.maximum_gradient_norm)
            or self.maximum_gradient_norm <= 0.0
            or not math.isfinite(self.beta1)
            or not 0.0 <= self.beta1 < 1.0
            or not math.isfinite(self.beta2)
            or not 0.0 <= self.beta2 < 1.0
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0.0
            or any(
                type(value) is not bool
                for value in (
                    self.amsgrad,
                    self.maximize,
                    self.foreach,
                    self.capturable,
                    self.differentiable,
                    self.fused,
                    self.tf32,
                    self.deterministic_algorithms,
                )
            )
            or not isinstance(self.maximum_updates, int)
            or isinstance(self.maximum_updates, bool)
            or self.maximum_updates < 1
            or self.autocast_dtype not in {"none", "bfloat16"}
        ):
            raise QualificationTrainerError(
                "qualification optimizer config differs"
            )


@dataclass(frozen=True, slots=True)
class QualificationStepReceipt:
    update_index: int
    loss: float
    gradient_norm: float
    trainable_parameters: int
    optimizer_state_bytes: int
    exact_metrics: QualificationExactMetrics
    training_manifest_sha256: str
    candidate_input_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.update_index < 1
            or not math.isfinite(self.loss)
            or self.loss < 0.0
            or not math.isfinite(self.gradient_norm)
            or self.gradient_norm < 0.0
            or self.trainable_parameters < 1
            or self.optimizer_state_bytes < 1
            or len(self.training_manifest_sha256) != 64
            or len(self.candidate_input_manifest_sha256) != 64
        ):
            raise QualificationTrainerError(
                "qualification step receipt differs"
            )


@dataclass(frozen=True, slots=True)
class QualificationEvaluationReceipt:
    split: str
    manifest_sha256: str
    exact_metrics: QualificationExactMetrics
    trainer_phase: str


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
        parameter_devices = {parameter.device for parameter in parameters}
        if len(parameter_devices) != 1:
            raise QualificationTrainerError(
                "qualification parameters span multiple devices"
            )
        self._device = next(iter(parameter_devices))
        if (
            self.config.fused
            and self._device.type != "cuda"
        ):
            raise QualificationTrainerError(
                "qualification fused AdamW requires CUDA"
            )
        if (
            self.config.autocast_dtype == "bfloat16"
            and self._device.type != "cuda"
        ):
            raise QualificationTrainerError(
                "qualification bfloat16 autocast requires CUDA"
            )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad,
            maximize=self.config.maximize,
            foreach=self.config.foreach,
            capturable=self.config.capturable,
            differentiable=self.config.differentiable,
            fused=self.config.fused,
        )
        self._update_index = 0
        self._phase = "training"

    def _autocast(self):
        if self.config.autocast_dtype == "none":
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
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
        if self._phase != "training":
            raise QualificationTrainerError(
                "qualification updates are disabled after training seal"
            )
        if self._update_index >= self.config.maximum_updates:
            raise QualificationTrainerError(
                "qualification update cap was exceeded"
            )
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
        with self._autocast():
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
            with self._autocast():
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
        self._update_index += 1
        return QualificationStepReceipt(
            update_index=self._update_index,
            loss=float(losses.total.detach().cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
            trainable_parameters=self.trainable_parameters,
            optimizer_state_bytes=state_bytes,
            exact_metrics=metrics,
            training_manifest_sha256=(
                self.training_custody.receipt_sha256
            ),
            candidate_input_manifest_sha256=(
                candidate.candidate_input_manifest_sha256
            ),
        )

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def update_index(self) -> int:
        return self._update_index

    def seal_training(self) -> None:
        """Irreversibly disable optimization before mechanics is opened."""

        if (
            self._phase != "training"
            or self._update_index != self.config.maximum_updates
        ):
            raise QualificationTrainerError(
                "qualification training cannot seal before its fixed budget"
            )
        self._phase = "train-sealed"

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
        if self._phase == "training":
            if custody.split != "train":
                raise QualificationTrainerError(
                    "nontrain evaluation requires a sealed trainer"
                )
            evaluation_phase = "training"
        elif self._phase == "train-sealed":
            if custody.split != "mechanics":
                raise QualificationTrainerError(
                    "sealed qualification opens mechanics only"
                )
            evaluation_phase = "mechanics-opened"
        else:
            raise QualificationTrainerError(
                "qualification trainer is closed"
            )
        custody.assert_batches(candidate, supervisor)
        self.system.eval()
        with self._autocast():
            output = self.system.compile_source(
                candidate.witness,
                straight_through=False,
                trunk_batch=self._trunk_batch(candidate),
            )
            self._assert_hankel_custody(output)
            exact_metrics = self.objective.exact_metrics(
                output,
                supervisor,
                candidate_source_sha256=candidate.source_sha256,
            )
        receipt = QualificationEvaluationReceipt(
            split=custody.split,
            manifest_sha256=custody.receipt_sha256,
            exact_metrics=exact_metrics,
            trainer_phase=evaluation_phase,
        )
        if evaluation_phase == "mechanics-opened":
            self._phase = "closed"
        return receipt


__all__ = [
    "EFCQualificationTrainer",
    "QualificationEvaluationReceipt",
    "QualificationStepReceipt",
    "QualificationTrainerConfig",
    "QualificationTrainerError",
]
