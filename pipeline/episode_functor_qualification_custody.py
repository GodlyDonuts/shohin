"""Self-hashed split custody for EFC neural qualification batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from typing import Sequence

import torch

from pipeline.episode_functor_identifiable_board import PilotRow
from pipeline.episode_functor_qualification_boundary import (
    CandidateCompilerBatch,
)
from pipeline.episode_functor_qualification_supervisor import (
    QualificationSupervisorBatch,
    collate_qualification_supervision,
)


QUALIFICATION_CUSTODY_SCHEMA = "efc-qualification-split-custody/v1"
_SPLITS = frozenset({"train", "mechanics", "development", "confirmation"})


class QualificationCustodyError(ValueError):
    """A qualification split or ordered batch left frozen custody."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _supervisor_manifest(
    supervisor: QualificationSupervisorBatch,
) -> tuple[dict[str, object], ...]:
    """Serialize every ordered target tensor without device dependence."""

    tensors: list[dict[str, object]] = []
    for field in fields(supervisor):
        value = getattr(supervisor, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        canonical = value.detach().cpu().contiguous()
        tensors.append(
            {
                "dtype": str(canonical.dtype),
                "name": field.name,
                "shape": list(canonical.shape),
                "values": canonical.tolist(),
            }
        )
    if not tensors:
        raise QualificationCustodyError(
            "qualification supervisor manifest is empty"
        )
    return tuple(tensors)


def _supervisor_digest(
    supervisor: QualificationSupervisorBatch,
) -> str:
    return _digest(_supervisor_manifest(supervisor))


@dataclass(frozen=True, slots=True)
class QualificationSplitCustody:
    """Ordered source/world/renderer receipt kept outside candidate input."""

    split: str
    row_count: int
    source_sha256: tuple[str, ...]
    world_manifest_sha256: str
    canonical_manifest_sha256: str
    factor_manifest_sha256: str
    supervisor_manifest_sha256: str
    board_manifest_sha256: str
    receipt_sha256: str
    schema: str = QUALIFICATION_CUSTODY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != QUALIFICATION_CUSTODY_SCHEMA
            or self.split not in _SPLITS
            or self.row_count < 1
            or len(self.source_sha256) != self.row_count
            or len(set(self.source_sha256)) != self.row_count
        ):
            raise QualificationCustodyError(
                "qualification split custody identity differs"
            )
        for value in (
            *self.source_sha256,
            self.world_manifest_sha256,
            self.canonical_manifest_sha256,
            self.factor_manifest_sha256,
            self.supervisor_manifest_sha256,
            self.board_manifest_sha256,
            self.receipt_sha256,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise QualificationCustodyError(
                    "qualification custody digest differs"
                )
        if self.receipt_sha256 != _digest(self._unsigned_mapping()):
            raise QualificationCustodyError(
                "qualification custody receipt hash differs"
            )

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            "board_manifest_sha256": self.board_manifest_sha256,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "factor_manifest_sha256": self.factor_manifest_sha256,
            "row_count": self.row_count,
            "schema": self.schema,
            "source_sha256": list(self.source_sha256),
            "split": self.split,
            "supervisor_manifest_sha256": (
                self.supervisor_manifest_sha256
            ),
            "world_manifest_sha256": self.world_manifest_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"

    def assert_training_split(self) -> None:
        if self.split != "train":
            raise QualificationCustodyError(
                "optimizer access is restricted to the train split"
            )

    def assert_batches(
        self,
        candidate: CandidateCompilerBatch,
        supervisor: QualificationSupervisorBatch,
    ) -> None:
        if (
            not isinstance(candidate, CandidateCompilerBatch)
            or not isinstance(supervisor, QualificationSupervisorBatch)
            or candidate.source_sha256 != self.source_sha256
            or candidate.witness.source_sha256 != self.source_sha256
            or supervisor.source_sha256 != self.source_sha256
            or _supervisor_digest(supervisor)
            != self.supervisor_manifest_sha256
        ):
            raise QualificationCustodyError(
                "qualification batch leaves split custody"
            )


def create_qualification_split_custody(
    rows: Sequence[PilotRow],
    *,
    split: str,
) -> QualificationSplitCustody:
    """Freeze one exact ordered split without exposing metadata to a model."""

    selected = tuple(row for row in rows if row.split == split)
    if split not in _SPLITS or not selected:
        raise QualificationCustodyError(
            "qualification custody split is empty or unknown"
        )
    if any(type(row) is not PilotRow for row in selected):
        raise QualificationCustodyError(
            "qualification custody rows differ"
        )
    source_hashes = tuple(
        sha256(row.source).hexdigest() for row in selected
    )
    supervisor_manifest_sha256 = _supervisor_digest(
        collate_qualification_supervision(selected)
    )
    board_rows = tuple(
        {
            "canonical_sha256": row.canonical_sha256,
            "factors": asdict(row.factors),
            "family": row.family,
            "source_sha256": source_hash,
            "split": row.split,
            "world_id": row.world_id,
        }
        for row, source_hash in zip(selected, source_hashes, strict=True)
    )
    unsigned = {
        "board_manifest_sha256": _digest(board_rows),
        "canonical_manifest_sha256": _digest(
            tuple(row.canonical_sha256 for row in selected)
        ),
        "factor_manifest_sha256": _digest(
            tuple(
                {
                    "factors": asdict(row.factors),
                    "family": row.family,
                }
                for row in selected
            )
        ),
        "row_count": len(selected),
        "schema": QUALIFICATION_CUSTODY_SCHEMA,
        "source_sha256": list(source_hashes),
        "split": split,
        "supervisor_manifest_sha256": supervisor_manifest_sha256,
        "world_manifest_sha256": _digest(
            tuple(row.world_id for row in selected)
        ),
    }
    return QualificationSplitCustody(
        split=split,
        row_count=len(selected),
        source_sha256=source_hashes,
        world_manifest_sha256=unsigned["world_manifest_sha256"],
        canonical_manifest_sha256=unsigned["canonical_manifest_sha256"],
        factor_manifest_sha256=unsigned["factor_manifest_sha256"],
        supervisor_manifest_sha256=supervisor_manifest_sha256,
        board_manifest_sha256=unsigned["board_manifest_sha256"],
        receipt_sha256=_digest(unsigned),
    )


__all__ = [
    "QUALIFICATION_CUSTODY_SCHEMA",
    "QualificationCustodyError",
    "QualificationSplitCustody",
    "create_qualification_split_custody",
]
