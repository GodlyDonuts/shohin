"""Fail-closed train-only candidate boundary for learned EFC qualification.

Only ``CandidateSource.source`` bytes cross into compiler preprocessing.
Targets and custody metadata must remain in a separate supervisor process.
This module performs role-free key/record scanning and tokenizer offset
alignment; it does not construct labels, parse source semantics, or execute a
machine.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_identifiable_board import CandidateSource  # noqa: E402
from episode_functor_shohin_trunk import ShohinTrunkBatch  # noqa: E402
from episode_functor_witness_compiler import (  # noqa: E402
    WitnessCompilerBatch,
    collate_witness_sources,
    scan_witness_source,
)


class QualificationBoundaryError(ValueError):
    """Candidate projection or role-free preprocessing failed."""


@dataclass(frozen=True, slots=True)
class CandidateCompilerBatch:
    witness: WitnessCompilerBatch
    trunk: ShohinTrunkBatch
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.witness.batch_size != len(self.trunk.payloads)
            or len(self.source_sha256) != self.witness.batch_size
        ):
            raise QualificationBoundaryError(
                "candidate compiler batch sizes differ"
            )


def _candidate_payloads(
    candidates: Sequence[CandidateSource],
) -> tuple[bytes, ...]:
    if tuple(field.name for field in fields(CandidateSource)) != (
        "source",
    ):
        raise QualificationBoundaryError(
            "CandidateSource schema exposes non-source fields"
        )
    frozen = tuple(candidates)
    if not frozen or any(
        type(candidate) is not CandidateSource for candidate in frozen
    ):
        raise QualificationBoundaryError(
            "candidate batch must contain exact CandidateSource objects"
        )
    return tuple(candidate.source for candidate in frozen)


def collate_candidate_sources(
    candidates: Sequence[CandidateSource],
    *,
    tokenizer,
    device: torch.device | str = "cpu",
) -> CandidateCompilerBatch:
    """Build target-free witness and frozen-trunk inputs from source bytes."""

    payloads = _candidate_payloads(candidates)
    witness = collate_witness_sources(
        tuple(scan_witness_source(payload) for payload in payloads),
        device=device,
    )
    encoded: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    for payload in payloads:
        try:
            text = payload.decode("ascii")
            result = tokenizer.encode(text)
            ids = tuple(int(value) for value in result.ids)
            offsets = tuple(
                (int(start), int(end))
                for start, end in result.offsets
            )
        except (AttributeError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise QualificationBoundaryError(
                "candidate tokenizer failed"
            ) from exc
        if (
            not ids
            or len(ids) != len(offsets)
            or any(value < 0 for value in ids)
        ):
            raise QualificationBoundaryError(
                "candidate tokenizer geometry differs"
            )
        coverage = [0] * len(payload)
        for start, end in offsets:
            if not 0 <= start < end <= len(payload):
                raise QualificationBoundaryError(
                    "candidate tokenizer offset leaves payload"
                )
            for index in range(start, end):
                coverage[index] += 1
        if any(value != 1 for value in coverage):
            raise QualificationBoundaryError(
                "candidate tokenizer offsets do not partition source bytes"
            )
        encoded.append((ids, offsets))

    maximum_tokens = max(len(ids) for ids, _ in encoded)
    batch = len(payloads)
    token_ids = torch.zeros(
        (batch, maximum_tokens),
        dtype=torch.long,
        device=device,
    )
    token_valid = torch.zeros(
        (batch, maximum_tokens),
        dtype=torch.bool,
        device=device,
    )
    token_bounds = torch.zeros(
        (batch, maximum_tokens, 2),
        dtype=torch.int32,
        device=device,
    )
    for row, (ids, offsets) in enumerate(encoded):
        count = len(ids)
        token_ids[row, :count] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :count] = True
        token_bounds[row, :count] = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=device,
        )
    return CandidateCompilerBatch(
        witness=witness,
        trunk=ShohinTrunkBatch(
            payloads=payloads,
            token_ids=token_ids,
            token_valid=token_valid,
            token_byte_bounds=token_bounds,
        ),
        source_sha256=tuple(
            sha256(payload).hexdigest() for payload in payloads
        ),
    )


__all__ = [
    "CandidateCompilerBatch",
    "QualificationBoundaryError",
    "collate_candidate_sources",
]
