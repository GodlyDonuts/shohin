"""Runtime-only EFC qualification supervisor tensor schema.

This module contains no board generator, source decoder, split seed, or label
builder. A train-only worker can validate a prepublished tensor payload
without importing the offline oracle that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from episode_functor_pointer_compiler import (
    MAX_KEY_OCCURRENCES,
    MAX_UNIQUE_KEYS,
)
from episode_functor_witness_compiler import MAX_RECORDS


STATE_COUNT = 8
ACTION_COUNT = 3
OBSERVER_COUNT = 2


class QualificationSupervisorError(ValueError):
    """Offline qualification labels or their source alignment failed."""


@dataclass(frozen=True, slots=True)
class QualificationSupervisorBatch:
    """Gold EFC-C tensors kept outside the candidate forward object."""

    source_sha256: tuple[str, ...]
    key_slot_to_unique: torch.Tensor
    record_type: torch.Tensor
    record_label_valid: torch.Tensor
    occurrence_role: torch.Tensor
    occurrence_label_valid: torch.Tensor
    record_answer: torch.Tensor
    answer_label_valid: torch.Tensor
    transition_next: torch.Tensor
    transition_exposed: torch.Tensor
    observer_answer: torch.Tensor
    observer_exposed: torch.Tensor

    def __post_init__(self) -> None:
        batch = len(self.source_sha256)
        expected = {
            "key_slot_to_unique": (
                (batch, STATE_COUNT + ACTION_COUNT + OBSERVER_COUNT),
                torch.long,
            ),
            "record_type": ((batch, MAX_RECORDS), torch.long),
            "record_label_valid": ((batch, MAX_RECORDS), torch.bool),
            "occurrence_role": (
                (batch, MAX_KEY_OCCURRENCES),
                torch.long,
            ),
            "occurrence_label_valid": (
                (batch, MAX_KEY_OCCURRENCES),
                torch.bool,
            ),
            "record_answer": ((batch, MAX_RECORDS), torch.long),
            "answer_label_valid": ((batch, MAX_RECORDS), torch.bool),
            "transition_next": (
                (batch, ACTION_COUNT, STATE_COUNT),
                torch.long,
            ),
            "transition_exposed": (
                (batch, ACTION_COUNT, STATE_COUNT),
                torch.bool,
            ),
            "observer_answer": (
                (batch, OBSERVER_COUNT, STATE_COUNT),
                torch.long,
            ),
            "observer_exposed": (
                (batch, OBSERVER_COUNT, STATE_COUNT),
                torch.bool,
            ),
        }
        devices: set[torch.device] = set()
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise QualificationSupervisorError(
                    f"{name} supervisor geometry differs"
                )
            devices.add(value.device)
        if len(devices) != 1:
            raise QualificationSupervisorError(
                "supervisor tensors must share one device"
            )
        if (
            len(set(self.source_sha256)) != batch
            or any(len(value) != 64 for value in self.source_sha256)
        ):
            raise QualificationSupervisorError(
                "supervisor source receipts differ"
            )
        if bool(
            self.key_slot_to_unique.lt(0).any()
            or self.key_slot_to_unique.ge(MAX_UNIQUE_KEYS).any()
        ):
            raise QualificationSupervisorError(
                "supervisor key assignment leaves support"
            )
        for row in self.key_slot_to_unique:
            if len(set(int(value) for value in row.tolist())) != row.numel():
                raise QualificationSupervisorError(
                    "supervisor key assignment is not one-to-one"
                )

    @property
    def batch_size(self) -> int:
        return len(self.source_sha256)

    def assert_candidate_alignment(
        self,
        candidate_source_sha256: Sequence[str],
    ) -> None:
        if tuple(candidate_source_sha256) != self.source_sha256:
            raise QualificationSupervisorError(
                "candidate and supervisor source receipts differ"
            )


__all__ = [
    "QualificationSupervisorBatch",
    "QualificationSupervisorError",
]
