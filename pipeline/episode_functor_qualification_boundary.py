"""Fail-closed train-only candidate boundary for learned EFC qualification.

Only ``CandidateSource.source`` bytes cross into compiler preprocessing.
Targets and custody metadata must remain in a separate supervisor process.
This module performs role-free key/record scanning and tokenizer offset
alignment; it does not construct labels, parse source semantics, or execute a
machine.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from pipeline.episode_functor_candidate_source import CandidateSource  # noqa: E402
from episode_functor_shohin_trunk import ShohinTrunkBatch  # noqa: E402
from episode_functor_witness_compiler import (  # noqa: E402
    WitnessCompilerBatch,
    collate_witness_sources,
    scan_witness_source,
)


class QualificationBoundaryError(ValueError):
    """Candidate projection or role-free preprocessing failed."""


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash_candidate_value(
    digest,
    *,
    path: str,
    value: object,
) -> None:
    """Hash a closed tensor/dataclass tree without pickle or repr."""

    if isinstance(value, torch.Tensor):
        canonical = value.detach().to(device="cpu", copy=True).contiguous()
        header = _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "path": path,
                "shape": list(canonical.shape),
                "type": "tensor",
            }
        )
        raw = canonical.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        return
    if isinstance(value, bytes):
        header = _canonical_json_bytes(
            {"path": path, "type": "bytes"}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        return
    if isinstance(value, tuple):
        header = _canonical_json_bytes(
            {"length": len(value), "path": path, "type": "tuple"}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        for index, item in enumerate(value):
            _hash_candidate_value(
                digest,
                path=f"{path}[{index}]",
                value=item,
            )
        return
    if is_dataclass(value) and not isinstance(value, type):
        header = _canonical_json_bytes(
            {
                "fields": [field.name for field in fields(value)],
                "path": path,
                "type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        for field in fields(value):
            _hash_candidate_value(
                digest,
                path=f"{path}.{field.name}",
                value=getattr(value, field.name),
            )
        return
    if isinstance(value, (bool, int, str, type(None))):
        payload = _canonical_json_bytes(
            {
                "path": path,
                "type": type(value).__name__,
                "value": value,
            }
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        return
    raise QualificationBoundaryError(
        f"candidate manifest contains unsupported value at {path}"
    )


def tokenizer_runtime_sha256(tokenizer) -> str:
    """Hash the exact serialized tokenizer runtime configuration."""

    try:
        serialized = tokenizer.to_str()
    except (AttributeError, TypeError, ValueError) as exc:
        raise QualificationBoundaryError(
            "candidate tokenizer lacks canonical runtime serialization"
        ) from exc
    if not isinstance(serialized, str) or not serialized:
        raise QualificationBoundaryError(
            "candidate tokenizer runtime serialization differs"
        )
    try:
        canonical = json.dumps(
            json.loads(serialized),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (UnicodeEncodeError, json.JSONDecodeError, TypeError) as exc:
        raise QualificationBoundaryError(
            "candidate tokenizer runtime is not canonical JSON"
        ) from exc
    return sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateCompilerBatch:
    witness: WitnessCompilerBatch
    trunk: ShohinTrunkBatch
    source_sha256: tuple[str, ...]
    tokenizer_artifact_sha256: str
    tokenizer_runtime_sha256: str
    candidate_input_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.witness.batch_size != len(self.trunk.payloads)
            or len(self.source_sha256) != self.witness.batch_size
        ):
            raise QualificationBoundaryError(
                "candidate compiler batch sizes differ"
            )
        for name in (
            "tokenizer_artifact_sha256",
            "tokenizer_runtime_sha256",
            "candidate_input_manifest_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise QualificationBoundaryError(
                    f"candidate {name} differs"
                )
        if (
            self.candidate_input_manifest_sha256
            != candidate_input_manifest_sha256(self)
        ):
            raise QualificationBoundaryError(
                "candidate input manifest hash differs"
            )


def candidate_input_manifest_sha256(
    candidate: CandidateCompilerBatch,
) -> str:
    """Recompute the complete source-derived candidate tensor manifest."""

    if not isinstance(candidate, CandidateCompilerBatch):
        raise QualificationBoundaryError(
            "candidate input manifest batch differs"
        )
    digest = sha256(b"EFC-CANDIDATE-INPUT-MANIFEST-V1\0")
    _hash_candidate_value(
        digest,
        path="witness",
        value=candidate.witness,
    )
    _hash_candidate_value(
        digest,
        path="trunk",
        value=candidate.trunk,
    )
    _hash_candidate_value(
        digest,
        path="source_sha256",
        value=candidate.source_sha256,
    )
    _hash_candidate_value(
        digest,
        path="tokenizer_artifact_sha256",
        value=candidate.tokenizer_artifact_sha256,
    )
    _hash_candidate_value(
        digest,
        path="tokenizer_runtime_sha256",
        value=candidate.tokenizer_runtime_sha256,
    )
    return digest.hexdigest()


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
    tokenizer_artifact_sha256: str,
    expected_tokenizer_runtime_sha256: str,
    device: torch.device | str = "cpu",
) -> CandidateCompilerBatch:
    """Build target-free witness and frozen-trunk inputs from source bytes."""

    if (
        _SHA256_PATTERN.fullmatch(tokenizer_artifact_sha256) is None
        or _SHA256_PATTERN.fullmatch(expected_tokenizer_runtime_sha256)
        is None
    ):
        raise QualificationBoundaryError(
            "candidate tokenizer receipt differs"
        )
    observed_runtime_sha256 = tokenizer_runtime_sha256(tokenizer)
    if observed_runtime_sha256 != expected_tokenizer_runtime_sha256:
        raise QualificationBoundaryError(
            "candidate tokenizer runtime SHA-256 mismatch"
        )
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
    provisional = CandidateCompilerBatch.__new__(CandidateCompilerBatch)
    object.__setattr__(provisional, "witness", witness)
    object.__setattr__(
        provisional,
        "trunk",
        ShohinTrunkBatch(
            payloads=payloads,
            token_ids=token_ids,
            token_valid=token_valid,
            token_byte_bounds=token_bounds,
        ),
    )
    object.__setattr__(
        provisional,
        "source_sha256",
        tuple(sha256(payload).hexdigest() for payload in payloads),
    )
    object.__setattr__(
        provisional,
        "tokenizer_artifact_sha256",
        tokenizer_artifact_sha256,
    )
    object.__setattr__(
        provisional,
        "tokenizer_runtime_sha256",
        observed_runtime_sha256,
    )
    manifest_sha256 = candidate_input_manifest_sha256(provisional)
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
        tokenizer_artifact_sha256=tokenizer_artifact_sha256,
        tokenizer_runtime_sha256=observed_runtime_sha256,
        candidate_input_manifest_sha256=manifest_sha256,
    )


__all__ = [
    "CandidateCompilerBatch",
    "QualificationBoundaryError",
    "candidate_input_manifest_sha256",
    "collate_candidate_sources",
    "tokenizer_runtime_sha256",
]
