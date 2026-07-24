"""Atomic train-only package for isolated HSC neural workers.

The package contains raw train sources and offline supervisor tensors.  It does
not contain PilotRow metadata, mechanics, development, confirmation, queries,
answers beyond train supervision, model state, or optimizer state.  The worker
must recollate candidate tensors from these raw bytes with the bound tokenizer.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Sequence

import torch

from pipeline.episode_functor_candidate_source import CandidateSource
from pipeline.episode_functor_qualification_boundary import (
    CandidateCompilerBatch,
    collate_candidate_sources,
)
from pipeline.episode_functor_qualification_custody import (
    QualificationSplitCustody,
)
from pipeline.episode_functor_qualification_batch import (
    QualificationSupervisorBatch,
)
from pipeline.episode_functor_runtime_custody import (
    abort_atomic_bundle,
    atomic_bundle_directory,
    finish_atomic_bundle,
    fsync_directory,
    write_json_fsync,
)


HANKEL_TRAIN_PACKAGE_SCHEMA = "efc-hankel-train-package/v1"
HANKEL_SUPERVISOR_PAYLOAD_SCHEMA = "efc-hankel-supervisor-tensors/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE_FILES = (
    "candidate_receipt.json",
    "split_custody.json",
    "supervisor.pt",
    "train_sources.jsonl",
)


class HankelTrainPackageError(ValueError):
    """A train-only package is incomplete, mutable, or hash-inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_supervisor(
    path: Path,
    supervisor: QualificationSupervisorBatch,
) -> None:
    tensors = {
        field.name: getattr(supervisor, field.name)
        .detach()
        .to(device="cpu", copy=True)
        .contiguous()
        for field in fields(supervisor)
        if isinstance(getattr(supervisor, field.name), torch.Tensor)
    }
    payload = {
        "schema": HANKEL_SUPERVISOR_PAYLOAD_SCHEMA,
        "source_sha256": list(supervisor.source_sha256),
        "tensors": tensors,
    }
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class HankelTrainPackageReceipt:
    row_count: int
    source_bytes: int
    source_sha256: tuple[str, ...]
    candidate_input_manifest_sha256: str
    tokenizer_artifact_sha256: str
    tokenizer_runtime_sha256: str
    split_custody_receipt_sha256: str
    files_sha256: tuple[tuple[str, str], ...]
    package_sha256: str
    schema: str = HANKEL_TRAIN_PACKAGE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_TRAIN_PACKAGE_SCHEMA
            or self.row_count < 1
            or self.source_bytes < self.row_count
            or len(self.source_sha256) != self.row_count
            or len(set(self.source_sha256)) != self.row_count
            or tuple(name for name, _ in self.files_sha256)
            != _PACKAGE_FILES
        ):
            raise HankelTrainPackageError(
                "HSC train package identity differs"
            )
        for value in (
            *self.source_sha256,
            self.candidate_input_manifest_sha256,
            self.tokenizer_artifact_sha256,
            self.tokenizer_runtime_sha256,
            self.split_custody_receipt_sha256,
            *(digest for _, digest in self.files_sha256),
            self.package_sha256,
        ):
            if (
                not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise HankelTrainPackageError(
                    "HSC train package digest differs"
                )
        if self.package_sha256 != sha256(
            _canonical_json_bytes(self._unsigned_mapping())
        ).hexdigest():
            raise HankelTrainPackageError(
                "HSC train package receipt hash differs"
            )

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            "candidate_input_manifest_sha256": (
                self.candidate_input_manifest_sha256
            ),
            "files_sha256": [
                {"name": name, "sha256": digest}
                for name, digest in self.files_sha256
            ],
            "row_count": self.row_count,
            "schema": self.schema,
            "source_bytes": self.source_bytes,
            "source_sha256": list(self.source_sha256),
            "split_custody_receipt_sha256": (
                self.split_custody_receipt_sha256
            ),
            "tokenizer_artifact_sha256": (
                self.tokenizer_artifact_sha256
            ),
            "tokenizer_runtime_sha256": self.tokenizer_runtime_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["package_sha256"] = self.package_sha256
        return result

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "HankelTrainPackageReceipt":
        if not isinstance(value, dict):
            raise HankelTrainPackageError(
                "HSC train package manifest differs"
            )
        expected = {
            "candidate_input_manifest_sha256",
            "files_sha256",
            "package_sha256",
            "row_count",
            "schema",
            "source_bytes",
            "source_sha256",
            "split_custody_receipt_sha256",
            "tokenizer_artifact_sha256",
            "tokenizer_runtime_sha256",
        }
        if set(value) != expected:
            raise HankelTrainPackageError(
                "HSC train package manifest schema differs"
            )
        raw_files = value["files_sha256"]
        if not isinstance(raw_files, list) or any(
            not isinstance(item, dict)
            or set(item) != {"name", "sha256"}
            for item in raw_files
        ):
            raise HankelTrainPackageError(
                "HSC train package file manifest differs"
            )
        source_sha256 = value["source_sha256"]
        if not isinstance(source_sha256, list):
            raise HankelTrainPackageError(
                "HSC train package source manifest differs"
            )
        try:
            return cls(
                row_count=value["row_count"],
                source_bytes=value["source_bytes"],
                source_sha256=tuple(source_sha256),
                candidate_input_manifest_sha256=(
                    value["candidate_input_manifest_sha256"]
                ),
                tokenizer_artifact_sha256=(
                    value["tokenizer_artifact_sha256"]
                ),
                tokenizer_runtime_sha256=(
                    value["tokenizer_runtime_sha256"]
                ),
                split_custody_receipt_sha256=(
                    value["split_custody_receipt_sha256"]
                ),
                files_sha256=tuple(
                    (item["name"], item["sha256"])
                    for item in raw_files
                ),
                package_sha256=value["package_sha256"],
                schema=value["schema"],
            )
        except (KeyError, TypeError) as exc:
            raise HankelTrainPackageError(
                "HSC train package values differ"
            ) from exc


def build_hankel_train_package(
    output: Path,
    *,
    sources: Sequence[bytes],
    candidate: CandidateCompilerBatch,
    supervisor: QualificationSupervisorBatch,
    custody: QualificationSplitCustody,
) -> HankelTrainPackageReceipt:
    """Atomically publish one immutable train-only package."""

    frozen = tuple(sources)
    hashes = tuple(sha256(source).hexdigest() for source in frozen)
    if (
        not frozen
        or any(not isinstance(source, bytes) or not source for source in frozen)
        or not isinstance(candidate, CandidateCompilerBatch)
        or not isinstance(supervisor, QualificationSupervisorBatch)
        or not isinstance(custody, QualificationSplitCustody)
        or hashes != candidate.source_sha256
        or hashes != supervisor.source_sha256
        or hashes != custody.source_sha256
    ):
        raise HankelTrainPackageError(
            "HSC train package inputs differ"
        )
    custody.assert_training_split()
    custody.assert_batches(candidate, supervisor)
    staging, lock = atomic_bundle_directory(output)
    try:
        source_lines = b"".join(
            _canonical_json_bytes(
                {
                    "source_b64": base64.b64encode(source).decode("ascii"),
                    "source_sha256": digest,
                }
            )
            + b"\n"
            for source, digest in zip(frozen, hashes, strict=True)
        )
        _write_bytes_exclusive(
            staging / "train_sources.jsonl",
            source_lines,
        )
        _write_supervisor(staging / "supervisor.pt", supervisor)
        write_json_fsync(
            staging / "candidate_receipt.json",
            {
                "candidate_input_manifest_sha256": (
                    candidate.candidate_input_manifest_sha256
                ),
                "source_sha256": list(candidate.source_sha256),
                "tokenizer_artifact_sha256": (
                    candidate.tokenizer_artifact_sha256
                ),
                "tokenizer_runtime_sha256": (
                    candidate.tokenizer_runtime_sha256
                ),
            },
        )
        write_json_fsync(
            staging / "split_custody.json",
            custody.to_mapping(),
        )
        file_hashes = tuple(
            (name, _file_sha256(staging / name))
            for name in _PACKAGE_FILES
        )
        unsigned = {
            "candidate_input_manifest_sha256": (
                candidate.candidate_input_manifest_sha256
            ),
            "files_sha256": [
                {"name": name, "sha256": digest}
                for name, digest in file_hashes
            ],
            "row_count": len(frozen),
            "schema": HANKEL_TRAIN_PACKAGE_SCHEMA,
            "source_bytes": sum(len(source) for source in frozen),
            "source_sha256": list(hashes),
            "split_custody_receipt_sha256": custody.receipt_sha256,
            "tokenizer_artifact_sha256": (
                candidate.tokenizer_artifact_sha256
            ),
            "tokenizer_runtime_sha256": (
                candidate.tokenizer_runtime_sha256
            ),
        }
        receipt = HankelTrainPackageReceipt(
            row_count=len(frozen),
            source_bytes=unsigned["source_bytes"],
            source_sha256=hashes,
            candidate_input_manifest_sha256=(
                candidate.candidate_input_manifest_sha256
            ),
            tokenizer_artifact_sha256=(
                candidate.tokenizer_artifact_sha256
            ),
            tokenizer_runtime_sha256=(
                candidate.tokenizer_runtime_sha256
            ),
            split_custody_receipt_sha256=custody.receipt_sha256,
            files_sha256=file_hashes,
            package_sha256=sha256(
                _canonical_json_bytes(unsigned)
            ).hexdigest(),
        )
        write_json_fsync(
            staging / "package_manifest.json",
            receipt.to_mapping(),
        )
        fsync_directory(staging)
        finish_atomic_bundle(staging, output, lock)
        return receipt
    except BaseException:
        abort_atomic_bundle(staging, lock)
        raise


def _load_json(path: Path) -> object:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def load_hankel_train_package(
    root: Path,
    *,
    tokenizer,
    expected_package_sha256: str,
    device: torch.device | str = "cpu",
) -> tuple[
    CandidateCompilerBatch,
    QualificationSupervisorBatch,
    QualificationSplitCustody,
    HankelTrainPackageReceipt,
]:
    """Reconstruct candidate tensors inside the worker and verify custody."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise HankelTrainPackageError(
            "HSC train package root is not a directory"
        )
    receipt = HankelTrainPackageReceipt.from_mapping(
        _load_json(root / "package_manifest.json")
    )
    if receipt.package_sha256 != expected_package_sha256:
        raise HankelTrainPackageError(
            "HSC train package expected hash differs"
        )
    expected_names = {
        "package_manifest.json",
        *(name for name, _ in receipt.files_sha256),
    }
    observed_entries = tuple(root.iterdir())
    if (
        {path.name for path in observed_entries} != expected_names
        or any(
            path.is_symlink() or not path.is_file()
            for path in observed_entries
        )
    ):
        raise HankelTrainPackageError(
            "HSC train package directory closure differs"
        )
    for name, expected in receipt.files_sha256:
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or _file_sha256(path) != expected
        ):
            raise HankelTrainPackageError(
                f"HSC train package file differs: {name}"
            )
    sources: list[bytes] = []
    hashes: list[str] = []
    with (root / "train_sources.jsonl").open(
        "r",
        encoding="ascii",
    ) as handle:
        for line in handle:
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row) != {"source_b64", "source_sha256"}
            ):
                raise HankelTrainPackageError(
                    "HSC train source row differs"
                )
            try:
                source = base64.b64decode(
                    row["source_b64"],
                    validate=True,
                )
            except (TypeError, ValueError) as exc:
                raise HankelTrainPackageError(
                    "HSC train source base64 differs"
                ) from exc
            digest = sha256(source).hexdigest()
            if digest != row["source_sha256"]:
                raise HankelTrainPackageError(
                    "HSC train source hash differs"
                )
            sources.append(source)
            hashes.append(digest)
    if (
        tuple(hashes) != receipt.source_sha256
        or sum(len(source) for source in sources)
        != receipt.source_bytes
    ):
        raise HankelTrainPackageError(
            "HSC train source aggregate differs"
        )
    candidate = collate_candidate_sources(
        tuple(CandidateSource(source) for source in sources),
        tokenizer=tokenizer,
        tokenizer_artifact_sha256=receipt.tokenizer_artifact_sha256,
        expected_tokenizer_runtime_sha256=(
            receipt.tokenizer_runtime_sha256
        ),
        device=device,
    )
    if (
        candidate.candidate_input_manifest_sha256
        != receipt.candidate_input_manifest_sha256
    ):
        raise HankelTrainPackageError(
            "HSC recollated candidate manifest differs"
        )
    with (root / "supervisor.pt").open("rb") as handle:
        payload = torch.load(
            handle,
            map_location=device,
            weights_only=True,
        )
    expected_tensor_names = {
        field.name
        for field in fields(QualificationSupervisorBatch)
        if field.name != "source_sha256"
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "source_sha256", "tensors"}
        or payload["schema"] != HANKEL_SUPERVISOR_PAYLOAD_SCHEMA
        or payload["source_sha256"] != list(receipt.source_sha256)
        or not isinstance(payload["tensors"], dict)
        or set(payload["tensors"]) != expected_tensor_names
    ):
        raise HankelTrainPackageError(
            "HSC supervisor payload differs"
        )
    supervisor = QualificationSupervisorBatch(
        source_sha256=receipt.source_sha256,
        **payload["tensors"],
    )
    custody = QualificationSplitCustody.from_mapping(
        _load_json(root / "split_custody.json")
    )
    if custody.receipt_sha256 != receipt.split_custody_receipt_sha256:
        raise HankelTrainPackageError(
            "HSC split custody package binding differs"
        )
    custody.assert_batches(candidate, supervisor)
    return candidate, supervisor, custody, receipt


__all__ = [
    "HANKEL_SUPERVISOR_PAYLOAD_SCHEMA",
    "HANKEL_TRAIN_PACKAGE_SCHEMA",
    "HankelTrainPackageError",
    "HankelTrainPackageReceipt",
    "build_hankel_train_package",
    "load_hankel_train_package",
]
