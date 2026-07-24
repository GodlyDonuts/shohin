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

from pipeline.episode_functor_qualification_batch import (  # noqa: E402
    QualificationSupervisorBatch,
)
from pipeline.episode_functor_hankel_geometry import (  # noqa: E402
    enumerate_action_words,
    prefix_shift_incidence,
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
from episode_functor_hankel_completion import (  # noqa: E402
    NeuralHankelShiftResult,
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
    unhardenable_rows: int

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


@dataclass(frozen=True, slots=True)
class HankelQualificationLossWeights:
    base_signature: float = 1.0
    derivative_signature: float = 1.0
    syndrome_margin: float = 0.5
    state_separation: float = 0.25

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, field.name), float)
            or getattr(self, field.name) <= 0.0
            for field in fields(self)
        ):
            raise QualificationLossError(
                "Hankel qualification weights must be positive floats"
            )


@dataclass(frozen=True, slots=True)
class HankelQualificationLossOutput:
    total: torch.Tensor
    base: QualificationLossOutput
    base_signature: torch.Tensor
    derivative_signature: torch.Tensor
    syndrome_margin: torch.Tensor
    state_separation: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.base, QualificationLossOutput):
            raise QualificationLossError(
                "Hankel base qualification loss differs"
            )
        for name in (
            "total",
            "base_signature",
            "derivative_signature",
            "syndrome_margin",
            "state_separation",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != ()
                or not value.is_floating_point()
                or not bool(torch.isfinite(value))
            ):
                raise QualificationLossError(
                    f"{name} Hankel qualification loss differs"
                )


@dataclass(frozen=True, slots=True)
class HankelQualificationExactMetrics(QualificationExactMetrics):
    exact_base_signature_cells: int
    base_signature_cells: int
    unhardenable_base_signature_cells: int
    exact_derivative_signature_cells: int
    derivative_signature_cells: int
    unhardenable_derivative_signature_cells: int
    exact_base_codebooks: int
    exact_derivative_codebooks: int

    def __post_init__(self) -> None:
        QualificationExactMetrics.__post_init__(self)
        if (
            self.exact_base_signature_cells > self.base_signature_cells
            or self.unhardenable_base_signature_cells
            > self.base_signature_cells
            or self.exact_base_signature_cells
            + self.unhardenable_base_signature_cells
            > self.base_signature_cells
            or self.exact_derivative_signature_cells
            > self.derivative_signature_cells
            or self.unhardenable_derivative_signature_cells
            > self.derivative_signature_cells
            or self.exact_derivative_signature_cells
            + self.unhardenable_derivative_signature_cells
            > self.derivative_signature_cells
            or self.exact_base_codebooks > self.rows
            or self.exact_derivative_codebooks > self.rows
        ):
            raise QualificationLossError(
                "Hankel exact metric count exceeds its support"
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
        if (
            output.source_sha256 != tuple(candidate_source_sha256)
            or output.source_sha256 != supervisor.source_sha256
            or output.projection.machine.batch_size != supervisor.batch_size
        ):
            raise QualificationLossError(
                "qualification output source receipt or batch differs"
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
        active_slots = (
            tuple(range(PRIMARY_STATES))
            + tuple(16 + index for index in range(PRIMARY_ACTIONS))
            + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
        )
        key_hardenable = _unique_metric_rows(
            output.key_assignment_logits[:, active_slots],
            label="key assignment",
        )
        transition_hardenable = _unique_metric_rows(
            output.projection.machine.action_next[
                :,
                :PRIMARY_ACTIONS,
                :PRIMARY_STATES,
                :PRIMARY_STATES,
            ],
            label="transition",
        )
        observer_hardenable = _unique_metric_rows(
            output.projection.machine.observer_answer[
                :,
                :PRIMARY_OBSERVERS,
                :PRIMARY_STATES,
                :PRIMARY_ANSWERS,
            ],
            label="observer",
        )
        hardenable = (
            key_hardenable
            & transition_hardenable
            & observer_hardenable
        )
        assigned = hard_assign_keys(
            slot_assignment_logits=output.key_assignment_logits,
            source_unique_key_bytes=output.unique_key_bytes,
            source_unique_key_valid=output.unique_key_valid,
        ).active_unique_indices
        exact_keys = int(
            (
                assigned.eq(supervisor.key_slot_to_unique).all(-1)
                & hardenable
            ).sum()
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
        ) & hardenable[:, None, None]
        observer_exact = observer_prediction.eq(
            supervisor.observer_answer
        ) & hardenable[:, None, None]
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
            unhardenable_rows=int((~hardenable).sum()),
        )


def _depth_for_word_count(word_count: int) -> int:
    for depth in range(7):
        if len(enumerate_action_words(depth)) == word_count:
            return depth
    raise QualificationLossError(
        "Hankel signature word count leaves supported depths"
    )


def _unique_metric_rows(
    logits: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        not logits.is_floating_point()
        or logits.shape[-1] < 2
        or not bool(torch.isfinite(logits).all())
    ):
        raise QualificationLossError(
            f"{label} metric logits differ"
        )
    top = logits.float().topk(2, dim=-1).values
    return ~top[..., 0].eq(top[..., 1]).flatten(1).any(-1)


def _unique_metric_cells(
    logits: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        not logits.is_floating_point()
        or logits.shape[-1] < 2
        or not bool(torch.isfinite(logits).all())
    ):
        raise QualificationLossError(
            f"{label} metric logits differ"
        )
    top = logits.float().topk(2, dim=-1).values
    return ~top[..., 0].eq(top[..., 1])


def _gold_future_signatures(
    supervisor: QualificationSupervisorBatch,
    *,
    max_depth: int,
) -> torch.Tensor:
    """Independently execute gold categorical tables into ``[B,S,W,O]``."""

    batch = supervisor.batch_size
    words = enumerate_action_words(max_depth)
    signatures: list[torch.Tensor] = []
    initial = torch.arange(
        PRIMARY_STATES,
        dtype=torch.long,
        device=supervisor.transition_next.device,
    )[None].expand(batch, -1)
    for word in words:
        state = initial
        for action in word:
            state = supervisor.transition_next[:, action].gather(
                1,
                state,
            )
        answers = supervisor.observer_answer.gather(
            2,
            state[:, None].expand(-1, PRIMARY_OBSERVERS, -1),
        ).permute(0, 2, 1)
        signatures.append(answers)
    return torch.stack(signatures, dim=2)


def _gold_hankel_targets(
    supervisor: QualificationSupervisorBatch,
    details: NeuralHankelShiftResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = _depth_for_word_count(details.base_signatures.shape[2])
    extended = _gold_future_signatures(
        supervisor,
        max_depth=depth + 1,
    )
    word_count = details.base_signatures.shape[2]
    incidence = torch.tensor(
        prefix_shift_incidence(depth),
        dtype=torch.long,
        device=extended.device,
    )
    if (
        incidence.shape != (PRIMARY_ACTIONS, word_count)
        or bool(incidence.lt(0).any())
        or bool(incidence.ge(extended.shape[2]).any())
    ):
        raise QualificationLossError(
            "Hankel target incidence leaves extended word inventory"
        )
    derivative = torch.stack(
        tuple(
            extended.index_select(2, incidence[action])
            for action in range(PRIMARY_ACTIONS)
        ),
        dim=1,
    )
    return extended[:, :, :word_count], derivative


def _signature_nll(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        probabilities.shape[:-1] != target.shape
        or target.dtype != torch.long
        or not probabilities.is_floating_point()
        or probabilities.device != target.device
        or not bool(torch.isfinite(probabilities).all())
    ):
        raise QualificationLossError(
            f"{label} signature target geometry differs"
        )
    tiny = torch.finfo(probabilities.dtype).tiny
    return F.nll_loss(
        probabilities.clamp_min(tiny).log().reshape(
            -1,
            probabilities.shape[-1],
        ),
        target.reshape(-1),
    )


def _signature_label_distances(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        probabilities.ndim < 4
        or targets.shape
        != (
            probabilities.shape[0],
            PRIMARY_STATES,
            *probabilities.shape[-3:-1],
        )
        or probabilities.shape[-1] != PRIMARY_ANSWERS
        or targets.dtype != torch.long
    ):
        raise QualificationLossError(
            f"{label} contrastive signature geometry differs"
        )
    prefix = probabilities.shape[:-3]
    expanded_probabilities = probabilities.unsqueeze(-4).expand(
        *prefix,
        PRIMARY_STATES,
        *probabilities.shape[-3:],
    )
    target_shape = (
        probabilities.shape[0],
        *(1 for _ in prefix[1:]),
        PRIMARY_STATES,
        *targets.shape[-2:],
        1,
    )
    expanded_targets = targets.reshape(target_shape).expand(
        *prefix,
        PRIMARY_STATES,
        *targets.shape[-2:],
        1,
    )
    tiny = torch.finfo(probabilities.dtype).tiny
    return -expanded_probabilities.clamp_min(tiny).log().gather(
        -1,
        expanded_targets,
    ).squeeze(-1).mean((-2, -1))


def _state_separation_penalty(
    signatures: torch.Tensor,
    gold_base: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    distance = _signature_label_distances(
        signatures,
        gold_base,
        label="base",
    )
    diagonal = torch.arange(
        PRIMARY_STATES,
        device=signatures.device,
    )
    positive = distance[:, diagonal, diagonal]
    negative = distance.masked_fill(
        torch.eye(
            PRIMARY_STATES,
            dtype=torch.bool,
            device=signatures.device,
        )[None],
        torch.inf,
    ).amin(-1)
    return F.relu(margin + positive - negative).mean()


class EFCHankelQualificationLoss(EFCQualificationLoss):
    """Machine objective plus explicit future-code and shift supervision."""

    def __init__(
        self,
        *,
        weights: QualificationLossWeights | None = None,
        hankel_weights: HankelQualificationLossWeights | None = None,
        syndrome_margin: float = 0.05,
        state_separation_margin: float = 0.10,
    ) -> None:
        super().__init__(weights=weights)
        self.hankel_weights = (
            HankelQualificationLossWeights()
            if hankel_weights is None
            else hankel_weights
        )
        if (
            not isinstance(syndrome_margin, float)
            or syndrome_margin <= 0.0
            or not isinstance(state_separation_margin, float)
            or state_separation_margin <= 0.0
        ):
            raise QualificationLossError(
                "Hankel qualification margins differ"
            )
        self.syndrome_margin = syndrome_margin
        self.state_separation_margin = state_separation_margin

    @staticmethod
    def _details(output: WitnessCompilerOutput) -> NeuralHankelShiftResult:
        details = output.projector_auxiliary
        if not isinstance(details, NeuralHankelShiftResult):
            raise QualificationLossError(
                "Hankel qualification requires projector diagnostics"
            )
        if details.projection is not output.projection:
            raise QualificationLossError(
                "Hankel diagnostics do not own the scored projection"
            )
        return details

    def forward(
        self,
        output: WitnessCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> HankelQualificationLossOutput:
        base = super().forward(
            output,
            supervisor,
            candidate_source_sha256=candidate_source_sha256,
        )
        details = self._details(output)
        gold_base, gold_derivative = _gold_hankel_targets(
            supervisor,
            details,
        )
        base_signature = _signature_nll(
            details.base_signatures,
            gold_base,
            label="base",
        )
        derivative_signature = _signature_nll(
            details.derivative_signatures,
            gold_derivative,
            label="derivative",
        )
        signature_distances = _signature_label_distances(
            details.derivative_signatures,
            gold_base,
            label="derivative",
        )
        positive = signature_distances.gather(
            -1,
            supervisor.transition_next[..., None],
        ).squeeze(-1)
        negative = signature_distances.masked_fill(
            F.one_hot(
                supervisor.transition_next,
                PRIMARY_STATES,
            ).bool(),
            torch.inf,
        ).amin(-1)
        syndrome_margin = F.relu(
            self.syndrome_margin + positive - negative
        ).mean()
        state_separation = _state_separation_penalty(
            details.base_signatures,
            gold_base,
            margin=self.state_separation_margin,
        )
        total = (
            base.total
            + self.hankel_weights.base_signature * base_signature
            + self.hankel_weights.derivative_signature
            * derivative_signature
            + self.hankel_weights.syndrome_margin * syndrome_margin
            + self.hankel_weights.state_separation * state_separation
        )
        return HankelQualificationLossOutput(
            total=total,
            base=base,
            base_signature=base_signature,
            derivative_signature=derivative_signature,
            syndrome_margin=syndrome_margin,
            state_separation=state_separation,
        )

    @torch.no_grad()
    def exact_metrics(
        self,
        output: WitnessCompilerOutput,
        supervisor: QualificationSupervisorBatch,
        *,
        candidate_source_sha256: Sequence[str],
    ) -> HankelQualificationExactMetrics:
        base = super().exact_metrics(
            output,
            supervisor,
            candidate_source_sha256=candidate_source_sha256,
        )
        details = self._details(output)
        gold_base, gold_derivative = _gold_hankel_targets(
            supervisor,
            details,
        )
        base_hardenable = _unique_metric_cells(
            details.base_signatures,
            label="base signature",
        )
        derivative_hardenable = _unique_metric_cells(
            details.derivative_signatures,
            label="derivative signature",
        )
        base_exact = (
            details.base_signatures.argmax(-1).eq(gold_base)
            & base_hardenable
        )
        derivative_exact = (
            details.derivative_signatures.argmax(-1).eq(gold_derivative)
            & derivative_hardenable
        )
        return HankelQualificationExactMetrics(
            **{
                field.name: getattr(base, field.name)
                for field in fields(QualificationExactMetrics)
            },
            exact_base_signature_cells=int(base_exact.sum()),
            base_signature_cells=base_exact.numel(),
            unhardenable_base_signature_cells=int(
                (~base_hardenable).sum()
            ),
            exact_derivative_signature_cells=int(derivative_exact.sum()),
            derivative_signature_cells=derivative_exact.numel(),
            unhardenable_derivative_signature_cells=int(
                (~derivative_hardenable).sum()
            ),
            exact_base_codebooks=int(base_exact.flatten(1).all(-1).sum()),
            exact_derivative_codebooks=int(
                derivative_exact.flatten(1).all(-1).sum()
            ),
        )


__all__ = [
    "EFCQualificationLoss",
    "EFCHankelQualificationLoss",
    "HankelQualificationExactMetrics",
    "HankelQualificationLossOutput",
    "HankelQualificationLossWeights",
    "QualificationExactMetrics",
    "QualificationLossError",
    "QualificationLossOutput",
    "QualificationLossWeights",
]
