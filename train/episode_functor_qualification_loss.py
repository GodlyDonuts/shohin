"""Frozen EFC-C loss and exact qualification metrics.

The compiler forward path consumes only candidate tensors.  This module joins
its output to offline supervisor labels by source SHA-256 after the forward
pass.  It does not parse source bytes or execute task semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import sys
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    QualificationSupervisorBatch,
)
from episode_functor_constrained_transport import (  # noqa: E402
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
    hard_assign_keys,
)
from episode_functor_witness_compiler import (  # noqa: E402
    WitnessCompilerOutput,
)


class QualificationLossError(ValueError):
    """EFC-C supervision alignment, loss, or metric geometry failed."""


@dataclass(frozen=True, slots=True)
class QualificationLossWeights:
    key_assignment: float = 1.0
    record_type: float = 1.0
    occurrence_role: float = 1.0
    record_answer: float = 1.0
    transition: float = 1.0
    observer: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, field.name), float)
            or getattr(self, field.name) <= 0.0
            for field in fields(self)
        ):
            raise QualificationLossError(
                "qualification loss weights must be positive floats"
            )


@dataclass(frozen=True, slots=True)
class QualificationLossOutput:
    total: torch.Tensor
    key_assignment: torch.Tensor
    record_type: torch.Tensor
    occurrence_role: torch.Tensor
    record_answer: torch.Tensor
    transition: torch.Tensor
    observer: torch.Tensor

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != ()
                or not value.is_floating_point()
                or not bool(torch.isfinite(value))
            ):
                raise QualificationLossError(
                    f"{field.name} qualification loss differs"
                )


@dataclass(frozen=True, slots=True)
class QualificationExactMetrics:
    rows: int
    exact_keys: int
    exact_record_types: int
    exact_occurrence_roles: int
    exact_record_answers: int
    exact_transition_cells: int
    exact_observer_cells: int
    exact_hidden_transition_cells: int
    exact_hidden_observer_cells: int
    hidden_transition_cells: int
    hidden_observer_cells: int
    exact_machines: int

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise QualificationLossError(
                "qualification metric row count differs"
            )
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or value < 0:
                raise QualificationLossError(
                    f"{field.name} qualification metric differs"
                )


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        logits.shape[:-1] != target.shape
        or target.shape != valid.shape
        or target.dtype != torch.long
        or valid.dtype != torch.bool
        or not bool(valid.any())
    ):
        raise QualificationLossError(
            f"{label} qualification target geometry differs"
        )
    return F.cross_entropy(logits[valid], target[valid])


class EFCQualificationLoss(nn.Module):
    """Source-hash-joined EFC-C objective with a frozen component contract."""

    def __init__(
        self,
        *,
        weights: QualificationLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.weights = (
            QualificationLossWeights()
            if weights is None
            else weights
        )

    @staticmethod
    def _align(
        output: WitnessCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        candidate_source_sha256: Sequence[str],
    ) -> None:
        if not isinstance(output, WitnessCompilerOutput):
            raise QualificationLossError(
                "qualification output type differs"
            )
        supervisor.assert_candidate_alignment(
            candidate_source_sha256
        )
        if output.projection.machine.batch_size != supervisor.batch_size:
            raise QualificationLossError(
                "qualification output batch differs"
            )

    def forward(
        self,
        output: WitnessCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> QualificationLossOutput:
        self._align(
            output,
            supervisor,
            candidate_source_sha256,
        )
        batch = supervisor.batch_size
        active_slots = (
            tuple(range(PRIMARY_STATES))
            + tuple(16 + index for index in range(PRIMARY_ACTIONS))
            + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
        )
        key_logits = output.raw_key_assignment_logits[
            :,
            active_slots,
        ]
        key_valid = torch.ones(
            supervisor.key_slot_to_unique.shape,
            dtype=torch.bool,
            device=supervisor.key_slot_to_unique.device,
        )
        key_assignment = _masked_cross_entropy(
            key_logits,
            supervisor.key_slot_to_unique,
            key_valid,
            label="key assignment",
        )
        record_type = _masked_cross_entropy(
            output.record_type_logits,
            supervisor.record_type,
            supervisor.record_label_valid,
            label="record type",
        )
        occurrence_role = _masked_cross_entropy(
            output.occurrence_role_logits,
            supervisor.occurrence_role,
            supervisor.occurrence_label_valid,
            label="occurrence role",
        )
        record_answer = _masked_cross_entropy(
            output.answer_logits,
            supervisor.record_answer,
            supervisor.answer_label_valid,
            label="record answer",
        )
        machine = output.projection.machine
        transition = F.cross_entropy(
            machine.action_next[
                :,
                :PRIMARY_ACTIONS,
                :PRIMARY_STATES,
                :PRIMARY_STATES,
            ].reshape(batch * PRIMARY_ACTIONS * PRIMARY_STATES, -1),
            supervisor.transition_next.reshape(-1),
        )
        observer = F.cross_entropy(
            machine.observer_answer[
                :,
                :PRIMARY_OBSERVERS,
                :PRIMARY_STATES,
                :PRIMARY_ANSWERS,
            ].reshape(
                batch * PRIMARY_OBSERVERS * PRIMARY_STATES,
                -1,
            ),
            supervisor.observer_answer.reshape(-1),
        )
        components = {
            "key_assignment": key_assignment,
            "record_type": record_type,
            "occurrence_role": occurrence_role,
            "record_answer": record_answer,
            "transition": transition,
            "observer": observer,
        }
        total = sum(
            getattr(self.weights, name) * value
            for name, value in components.items()
        )
        return QualificationLossOutput(total=total, **components)

    @torch.no_grad()
    def exact_metrics(
        self,
        output: WitnessCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> QualificationExactMetrics:
        self._align(
            output,
            supervisor,
            candidate_source_sha256,
        )
        assigned = hard_assign_keys(
            slot_assignment_logits=output.key_assignment_logits,
            source_unique_key_bytes=output.unique_key_bytes,
            source_unique_key_valid=output.unique_key_valid,
        ).active_unique_indices
        exact_keys = int(
            assigned.eq(supervisor.key_slot_to_unique).all(-1).sum()
        )
        record_prediction = output.record_type_logits.argmax(-1)
        role_prediction = output.occurrence_role_logits.argmax(-1)
        answer_prediction = output.answer_logits.argmax(-1)
        hard = output.projection.machine.harden()
        transition_prediction = hard.action_next[
            :,
            :PRIMARY_ACTIONS,
            :PRIMARY_STATES,
        ].long()
        observer_prediction = hard.observer_answer[
            :,
            :PRIMARY_OBSERVERS,
            :PRIMARY_STATES,
        ].long()
        transition_exact = transition_prediction.eq(
            supervisor.transition_next
        )
        observer_exact = observer_prediction.eq(
            supervisor.observer_answer
        )
        hidden_transition = ~supervisor.transition_exposed
        hidden_observer = ~supervisor.observer_exposed

        def exact_rows(
            prediction: torch.Tensor,
            target: torch.Tensor,
            valid: torch.Tensor,
        ) -> int:
            correct = prediction.eq(target) | ~valid
            return int(correct.flatten(1).all(-1).sum())

        machine_exact = transition_exact.flatten(1).all(-1) & (
            observer_exact.flatten(1).all(-1)
        )
        return QualificationExactMetrics(
            rows=supervisor.batch_size,
            exact_keys=exact_keys,
            exact_record_types=exact_rows(
                record_prediction,
                supervisor.record_type,
                supervisor.record_label_valid,
            ),
            exact_occurrence_roles=exact_rows(
                role_prediction,
                supervisor.occurrence_role,
                supervisor.occurrence_label_valid,
            ),
            exact_record_answers=exact_rows(
                answer_prediction,
                supervisor.record_answer,
                supervisor.answer_label_valid,
            ),
            exact_transition_cells=int(transition_exact.sum()),
            exact_observer_cells=int(observer_exact.sum()),
            exact_hidden_transition_cells=int(
                transition_exact[hidden_transition].sum()
            ),
            exact_hidden_observer_cells=int(
                observer_exact[hidden_observer].sum()
            ),
            hidden_transition_cells=int(hidden_transition.sum()),
            hidden_observer_cells=int(hidden_observer.sum()),
            exact_machines=int(machine_exact.sum()),
        )


__all__ = [
    "EFCQualificationLoss",
    "QualificationExactMetrics",
    "QualificationLossError",
    "QualificationLossOutput",
    "QualificationLossWeights",
]
