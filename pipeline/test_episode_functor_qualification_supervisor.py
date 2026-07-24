from __future__ import annotations

from dataclasses import fields
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    PilotRow,
    generate_pilot_rows,
    project_candidate_sources,
)
from pipeline.episode_functor_qualification_boundary import (  # noqa: E402
    collate_candidate_sources,
)
from pipeline.episode_functor_qualification_supervisor import (  # noqa: E402
    QualificationSupervisorError,
    collate_qualification_supervision,
)
from episode_functor_witness_compiler import (  # noqa: E402
    RECORD_OBSERVATION,
    ROLE_IGNORE,
)


class _Encoded:
    def __init__(self, payload: str) -> None:
        self.ids = list(payload.encode("ascii"))
        self.offsets = [
            (index, index + 1)
            for index in range(len(self.ids))
        ]


class _ByteTokenizer:
    def encode(self, payload: str) -> _Encoded:
        return _Encoded(payload)


def _rows() -> tuple[PilotRow, ...]:
    return generate_pilot_rows(
        seed="efc-qualification-supervisor-test-v1",
        counts={
            "train": 2,
            "mechanics": 2,
            "development": 2,
            "confirmation": 2,
        },
    )


def test_supervisor_is_separate_and_hash_aligned_to_candidate() -> None:
    rows = tuple(row for row in _rows() if row.split == "train")
    candidates = project_candidate_sources(rows, split="train")
    candidate_batch = collate_candidate_sources(
        candidates,
        tokenizer=_ByteTokenizer(),
    )
    supervisor = collate_qualification_supervision(rows)
    supervisor.assert_candidate_alignment(candidate_batch.source_sha256)

    assert tuple(field.name for field in fields(candidates[0])) == (
        "source",
    )
    assert not hasattr(candidate_batch, "transition_next")
    assert not hasattr(candidate_batch, "record_type")
    assert supervisor.batch_size == candidate_batch.witness.batch_size
    assert torch.all(
        supervisor.key_slot_to_unique.sort(-1).values
        == torch.arange(13)[None]
    )
    assert int(supervisor.answer_label_valid.sum()) == 14 * len(rows)
    assert int(
        (
            supervisor.record_type == RECORD_OBSERVATION
        ).logical_and(supervisor.record_label_valid).sum()
    ) == 14 * len(rows)
    assert int(
        (
            supervisor.occurrence_role == ROLE_IGNORE
        ).logical_and(supervisor.occurrence_label_valid).sum()
    ) == 0


def test_supervisor_covers_all_renderer_combinations_and_fail_closes() -> None:
    rows = _rows()
    supervisor = collate_qualification_supervision(rows)
    assert supervisor.batch_size == len(rows)
    assert set(row.factors.values for row in rows) == {
        (framing, organization, codec)
        for framing in range(2)
        for organization in range(2)
        for codec in range(2)
    }
    changed = list(supervisor.source_sha256)
    changed[0] = "0" * 64
    with pytest.raises(
        QualificationSupervisorError,
        match="source receipts differ",
    ):
        supervisor.assert_candidate_alignment(changed)


def test_supervisor_rejects_nonexact_or_duplicate_rows() -> None:
    row = _rows()[0]

    class PoisonedPilotRow(PilotRow):
        pass

    poisoned = PoisonedPilotRow(
        row.world_id,
        row.split,
        row.family,
        row.factors,
        row.source,
        row.machine,
        row.canonical_sha256,
    )
    with pytest.raises(
        QualificationSupervisorError,
        match="exact PilotRow",
    ):
        collate_qualification_supervision((poisoned,))
    with pytest.raises(
        QualificationSupervisorError,
        match="duplicated",
    ):
        collate_qualification_supervision((row, row))
