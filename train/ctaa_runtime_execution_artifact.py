"""Canonical content-addressed artifacts for query-blind CTAA execution.

Each parent and projected attempt is encoded as a canonical binary frame.  A
frame contains canonical JSON metadata followed by exact packet and tensor
bytes.  The immutable aggregate contains only ordered metadata and SHA-256
addresses; callers can pass the attempt addresses directly to the signed
execution-receipt builder.

This codec is intentionally stricter than a general PyTorch serializer.  It
accepts only the frozen CPU tensors emitted by ``ctaa_runtime_execution_engine``
and independently replays every snapshot, record, and execution commitment.
No pickle or producer-controlled class loading is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import sys
from typing import Mapping, Sequence

import torch

from ctaa_intervention_protocol import (
    InterventionFamily,
    LOCKED_SCORED_ROW_COUNT,
    RUNTIME_PANEL_SIZE,
)
from ctaa_packet_io import packet_body
from ctaa_runtime_execution_engine import (
    ATTEMPT_RECORD_SCHEMA,
    PARENT_RECORD_SCHEMA,
    RUNTIME_EXECUTION_SCHEMA,
    SNAPSHOT_SCHEMA,
    AttemptExecutionRecord,
    ExecutionFailure,
    ExecutionSnapshot,
    ParentExecutionRecord,
    RuntimeExecutionResult,
)
from ctaa_runtime_execution_projection import EXECUTION_PROJECTION_SCHEMA
from ctaa_trunk_compiler import HardCTAAPacket


EXECUTION_ARTIFACT_SCHEMA = "r12_ctaa_runtime_execution_artifact_v1"
EXECUTION_AGGREGATE_SCHEMA = "r12_ctaa_runtime_execution_aggregate_v1"
EXECUTION_ARTIFACT_SUFFIX = ".ctaaexec"
EXECUTION_ARTIFACT_MAGIC = b"CTAAEXE1"
EXECUTION_ARTIFACT_HEADER = struct.Struct(">Q")
EXPECTED_PREQUERY_ATTEMPT_COUNT = 21_600

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_AGGREGATE_BYTES = 256 * 1024 * 1024
_MAX_RAW_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_BLOB_ORDER = (
    "packet",
    "h19_residual",
    "h29_residual",
    "state_route",
    "composed_route",
    "halted",
    "terminal",
)
_TENSOR_FIELDS = _BLOB_ORDER[1:]
_FORBIDDEN_FRAGMENTS = (
    "answer",
    "oracle",
    "query_position",
    "query_source",
    "query_bytes",
    "query_token",
    "late_query",
)
_DTYPES: dict[str, torch.dtype] = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}

_FAILURE_KEYS = frozenset({"stage", "code"})
_PARENT_RECORD_KEYS = frozenset(
    {
        "schema",
        "anchor_id",
        "status",
        "program_source_sha256",
        "expected_packet_sha256",
        "snapshot_sha256",
        "failure",
        "record_sha256",
    }
)
_ATTEMPT_RECORD_KEYS = frozenset(
    {
        "schema",
        "attempt_index",
        "attempt_id",
        "operation",
        "anchor_id",
        "donor_anchor_id",
        "status",
        "parent_record_sha256",
        "committed_program_source_sha256",
        "committed_packet_sha256",
        "observed_program_source_sha256",
        "observed_packet_sha256",
        "snapshot_sha256",
        "extra_artifact_hashes",
        "failure",
        "record_sha256",
    }
)
_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "artifact_hashes",
        "snapshot_sha256",
        "packet_blob",
        "h19_residual_blob",
        "h29_residual_blob",
        "state_route_blob",
        "composed_route_blob",
        "halted_blob",
        "terminal_blob",
    }
)
_FRAME_KEYS = frozenset({"schema", "kind", "record", "snapshot", "blobs"})
_TENSOR_BLOB_KEYS = frozenset(
    {"name", "kind", "dtype", "shape", "offset", "length", "raw_sha256"}
)
_PACKET_BLOB_KEYS = frozenset(
    {
        "name",
        "kind",
        "dtype",
        "action_cards_shape",
        "initial_state_shape",
        "schedule_shape",
        "offset",
        "length",
        "raw_sha256",
    }
)
_PARENT_REF_KEYS = frozenset(
    {"anchor_id", "status", "record_sha256", "raw_output_artifact_sha256"}
)
_ATTEMPT_REF_KEYS = frozenset(
    {
        "attempt_index",
        "attempt_id",
        "operation",
        "anchor_id",
        "status",
        "record_sha256",
        "raw_output_artifact_sha256",
    }
)
_AGGREGATE_KEYS = frozenset(
    {
        "schema",
        "runtime_execution_schema",
        "execution_projection_schema",
        "projection_sha256",
        "scored_row_count",
        "runtime_attempts_affect_scored_denominator",
        "runtime_panel_size",
        "parent_count",
        "attempt_count",
        "execution_sha256",
        "parent_artifacts",
        "parent_artifacts_sha256",
        "attempt_artifacts",
        "attempt_artifacts_sha256",
    }
)


class RuntimeExecutionArtifactError(ValueError):
    """A CTAA execution artifact failed closed."""


@dataclass(frozen=True)
class RuntimeExecutionArtifactIndex:
    """Addresses emitted after an immutable aggregate is published."""

    aggregate_sha256: str
    parent_artifact_sha256s: tuple[str, ...]
    attempt_artifact_sha256s: tuple[str, ...]
    attempt_outputs: tuple[dict[str, str], ...]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RuntimeExecutionArtifactError(
            "execution artifact is not canonical JSON"
        ) from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RuntimeExecutionArtifactError(f"invalid {label}")
    return value


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value.strip() != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise RuntimeExecutionArtifactError(f"invalid {label}")
    return value


def _reject_leakage(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold().replace("-", "_").replace(" ", "_")
            if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
                raise RuntimeExecutionArtifactError(
                    f"execution artifact leaks forbidden field: {key}"
                )
            _reject_leakage(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_leakage(item)
    elif isinstance(value, str):
        lowered = value.casefold().replace("-", "_").replace(" ", "_")
        if lowered == "query_isolation":
            return
        if lowered.startswith("query_isolation:"):
            lowered = lowered.removeprefix("query_isolation:")
        if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
            raise RuntimeExecutionArtifactError(
                "execution artifact leaks deferred or oracle data"
            )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeExecutionArtifactError(
                f"duplicate execution artifact key: {key}"
            )
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise RuntimeExecutionArtifactError(f"non-finite {label} constant: {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeExecutionArtifactError(f"malformed {label} JSON") from error
    if not isinstance(value, dict):
        raise RuntimeExecutionArtifactError(f"{label} root must be an object")
    _reject_leakage(value)
    return value


def _path_text(path: Path) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise RuntimeExecutionArtifactError("path must be filesystem text") from error
    if not isinstance(raw, str) or "\x00" in raw:
        raise RuntimeExecutionArtifactError("path must be filesystem text")
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw or raw == "/":
        raise RuntimeExecutionArtifactError("path must be absolute and normalized")
    if any(part in ("", ".", "..") for part in raw.split("/")[1:]):
        raise RuntimeExecutionArtifactError("path contains an unsafe component")
    return raw


def _open_directory(path: Path) -> int:
    raw = _path_text(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeExecutionArtifactError("O_NOFOLLOW is required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in raw.split("/")[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise RuntimeExecutionArtifactError(
                    "artifact directory is missing, symlinked, or invalid"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise RuntimeExecutionArtifactError("artifact path is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_parent_directory(path: Path) -> tuple[int, str]:
    raw = _path_text(path)
    parent, name = os.path.split(raw)
    if not name:
        raise RuntimeExecutionArtifactError("artifact filename is empty")
    return _open_directory(Path(parent)), name


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_immutable_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> bytes:
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        except OSError as error:
            raise RuntimeExecutionArtifactError(
                "immutable execution artifact is missing or symlinked"
            ) from error
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise RuntimeExecutionArtifactError(
                "execution artifact must be immutable, regular, and single-link"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeExecutionArtifactError(
                    "execution artifact changed during read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeExecutionArtifactError("execution artifact grew during read")
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise RuntimeExecutionArtifactError(
                "execution artifact changed during read"
            )
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_immutable_path(path: Path, *, maximum_bytes: int) -> bytes:
    directory_descriptor, name = _open_parent_directory(path)
    try:
        return _read_immutable_at(
            directory_descriptor, name, maximum_bytes=maximum_bytes
        )
    finally:
        os.close(directory_descriptor)


def _publish_immutable_at(
    directory_descriptor: int,
    name: str,
    raw: bytes,
    *,
    allow_identical: bool,
) -> bool:
    if not raw:
        raise RuntimeExecutionArtifactError("execution artifact output is empty")
    temporary = f".{name}.ctaa-{os.getpid()}-{secrets.token_hex(12)}"
    descriptor = -1
    linked = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_descriptor)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise RuntimeExecutionArtifactError(
                    "execution artifact write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o222
            or metadata.st_size != len(raw)
        ):
            raise RuntimeExecutionArtifactError(
                "immutable execution artifact write verification failed"
            )
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if not allow_identical:
                raise FileExistsError(
                    f"refusing existing execution artifact: {name}"
                ) from None
            existing = _read_immutable_at(
                directory_descriptor,
                name,
                maximum_bytes=max(_MAX_RAW_ARTIFACT_BYTES, _MAX_AGGREGATE_BYTES),
            )
            if existing != raw:
                raise RuntimeExecutionArtifactError(
                    "content-addressed execution artifact substitution"
                )
            return False
        linked = True
        os.unlink(temporary, dir_fd=directory_descriptor)
        temporary = ""
        final = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_mode & 0o222
            or final.st_size != len(raw)
        ):
            raise RuntimeExecutionArtifactError(
                "published execution artifact is not immutable"
            )
        os.fsync(directory_descriptor)
        return True
    except OSError as error:
        if linked:
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        if isinstance(error, FileExistsError):
            raise
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise RuntimeExecutionArtifactError(
                "execution artifact path contains a symlink"
            ) from error
        raise
    except Exception:
        if linked:
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _publish_immutable_path(path: Path, raw: bytes) -> None:
    directory_descriptor, name = _open_parent_directory(path)
    try:
        _publish_immutable_at(directory_descriptor, name, raw, allow_identical=False)
    finally:
        os.close(directory_descriptor)


def _tensor_raw(value: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeExecutionArtifactError(
            "canonical CTAA tensor artifacts require little-endian execution"
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.requires_grad
        or not value.is_contiguous()
        or str(value.dtype) not in _DTYPES
    ):
        raise RuntimeExecutionArtifactError(
            "execution tensor is not a frozen contiguous CPU tensor"
        )
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_hash(name: str, dtype: str, shape: Sequence[int], raw: bytes) -> str:
    header = _canonical_json_bytes(
        {
            "schema": "r12_ctaa_tensor_artifact_v1",
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
        }
    )
    return _sha256_bytes(header + b"\0" + raw)


def _tensor_descriptor(
    name: str, value: torch.Tensor, offset: int
) -> tuple[dict[str, object], bytes]:
    raw = _tensor_raw(value)
    descriptor: dict[str, object] = {
        "name": name,
        "kind": "tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "offset": offset,
        "length": len(raw),
        "raw_sha256": _sha256_bytes(raw),
    }
    return descriptor, raw


def _packet_descriptor(
    packet: HardCTAAPacket, offset: int
) -> tuple[dict[str, object], bytes]:
    if not isinstance(packet, HardCTAAPacket):
        raise RuntimeExecutionArtifactError("execution packet type differs")
    for value in (packet.action_cards, packet.initial_state, packet.schedule):
        _tensor_raw(value)
    raw = packet_body(packet)
    descriptor: dict[str, object] = {
        "name": "packet",
        "kind": "packet",
        "dtype": str(torch.uint8),
        "action_cards_shape": list(packet.action_cards.shape),
        "initial_state_shape": list(packet.initial_state.shape),
        "schedule_shape": list(packet.schedule.shape),
        "offset": offset,
        "length": len(raw),
        "raw_sha256": _sha256_bytes(raw),
    }
    return descriptor, raw


def _failure_json(failure: ExecutionFailure | None) -> dict[str, str] | None:
    if failure is None:
        return None
    stage = _require_identifier(failure.stage, "failure stage")
    code = _require_identifier(failure.code, "failure code")
    value = {"stage": stage, "code": code}
    _reject_leakage(value)
    return value


def _hash_pairs(value: Sequence[tuple[str, str]], label: str) -> list[list[str]]:
    rows: list[list[str]] = []
    previous = ""
    for name, digest in value:
        name = _require_identifier(name, f"{label} name")
        digest = _require_hash(digest, f"{label} digest")
        if name <= previous:
            raise RuntimeExecutionArtifactError(f"{label} ordering differs")
        previous = name
        rows.append([name, digest])
    return rows


def _snapshot_parts(
    snapshot: ExecutionSnapshot,
) -> tuple[dict[str, object], list[dict[str, object]], bytes]:
    if (
        not isinstance(snapshot, ExecutionSnapshot)
        or snapshot.schema != SNAPSHOT_SCHEMA
    ):
        raise RuntimeExecutionArtifactError("execution snapshot schema differs")
    values: dict[str, object] = {
        "packet": snapshot.packet,
        "h19_residual": snapshot.h19_residual,
        "h29_residual": snapshot.h29_residual,
        "state_route": snapshot.state_route,
        "composed_route": snapshot.composed_route,
        "halted": snapshot.halted,
        "terminal": snapshot.terminal,
    }
    descriptors: list[dict[str, object]] = []
    body_parts: list[bytes] = []
    offset = 0
    for name in _BLOB_ORDER:
        value = values[name]
        if value is None:
            continue
        if name == "packet":
            descriptor, raw = _packet_descriptor(value, offset)  # type: ignore[arg-type]
        else:
            if not isinstance(value, torch.Tensor):
                raise RuntimeExecutionArtifactError("execution tensor field differs")
            descriptor, raw = _tensor_descriptor(name, value, offset)
        descriptors.append(descriptor)
        body_parts.append(raw)
        offset += len(raw)
    artifact_hashes = _hash_pairs(snapshot.artifact_hashes, "snapshot artifact")
    expected_hashes: list[list[str]] = []
    for descriptor in descriptors:
        name = str(descriptor["name"])
        if descriptor["kind"] == "packet":
            digest = str(descriptor["raw_sha256"])
        else:
            digest = _tensor_hash(
                name,
                str(descriptor["dtype"]),
                list(descriptor["shape"]),  # type: ignore[arg-type]
                body_parts[len(expected_hashes)],
            )
        expected_hashes.append([name, digest])
    expected_hashes.sort()
    if artifact_hashes != expected_hashes:
        raise RuntimeExecutionArtifactError("snapshot artifact hashes differ")
    snapshot_payload = {
        "schema": snapshot.schema,
        "artifact_hashes": artifact_hashes,
    }
    if snapshot.snapshot_sha256 != _sha256_json(snapshot_payload):
        raise RuntimeExecutionArtifactError("snapshot commitment differs")
    descriptor_names = {str(row["name"]) for row in descriptors}
    metadata: dict[str, object] = {
        "schema": snapshot.schema,
        "artifact_hashes": artifact_hashes,
        "snapshot_sha256": snapshot.snapshot_sha256,
        **{
            f"{name}_blob": name if name in descriptor_names else None
            for name in _BLOB_ORDER
        },
    }
    return metadata, descriptors, b"".join(body_parts)


def _parent_record_json(record: ParentExecutionRecord) -> dict[str, object]:
    return {
        "schema": record.schema,
        "anchor_id": record.anchor_id,
        "status": record.status,
        "program_source_sha256": record.program_source_sha256,
        "expected_packet_sha256": record.expected_packet_sha256,
        "snapshot_sha256": (
            None if record.snapshot is None else record.snapshot.snapshot_sha256
        ),
        "failure": _failure_json(record.failure),
        "record_sha256": record.record_sha256,
    }


def _attempt_record_json(record: AttemptExecutionRecord) -> dict[str, object]:
    return {
        "schema": record.schema,
        "attempt_index": record.attempt_index,
        "attempt_id": record.attempt_id,
        "operation": record.operation,
        "anchor_id": record.anchor_id,
        "donor_anchor_id": record.donor_anchor_id,
        "status": record.status,
        "parent_record_sha256": record.parent_record_sha256,
        "committed_program_source_sha256": record.committed_program_source_sha256,
        "committed_packet_sha256": record.committed_packet_sha256,
        "observed_program_source_sha256": record.observed_program_source_sha256,
        "observed_packet_sha256": record.observed_packet_sha256,
        "snapshot_sha256": (
            None if record.snapshot is None else record.snapshot.snapshot_sha256
        ),
        "extra_artifact_hashes": _hash_pairs(
            record.extra_artifact_hashes, "extra artifact"
        ),
        "failure": _failure_json(record.failure),
        "record_sha256": record.record_sha256,
    }


def _frame_bytes(record: ParentExecutionRecord | AttemptExecutionRecord) -> bytes:
    if isinstance(record, ParentExecutionRecord):
        kind = "parent"
        record_json = _parent_record_json(record)
    elif isinstance(record, AttemptExecutionRecord):
        kind = "attempt"
        record_json = _attempt_record_json(record)
    else:
        raise RuntimeExecutionArtifactError("execution record type differs")
    if record.snapshot is None:
        snapshot_json = None
        descriptors: list[dict[str, object]] = []
        body = b""
    else:
        snapshot_json, descriptors, body = _snapshot_parts(record.snapshot)
    header: dict[str, object] = {
        "schema": EXECUTION_ARTIFACT_SCHEMA,
        "kind": kind,
        "record": record_json,
        "snapshot": snapshot_json,
        "blobs": descriptors,
    }
    _reject_leakage(header)
    encoded = _canonical_json_bytes(header)
    return (
        EXECUTION_ARTIFACT_MAGIC
        + EXECUTION_ARTIFACT_HEADER.pack(len(encoded))
        + encoded
        + body
    )


def _shape(value: object, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) > 8
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise RuntimeExecutionArtifactError(f"invalid {label} shape")
    return tuple(value)


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_tensor(
    descriptor: Mapping[str, object], raw: bytes, *, expected_name: str
) -> torch.Tensor:
    if set(descriptor) != _TENSOR_BLOB_KEYS:
        raise RuntimeExecutionArtifactError("tensor descriptor schema differs")
    if descriptor["name"] != expected_name or descriptor["kind"] != "tensor":
        raise RuntimeExecutionArtifactError("tensor descriptor identity differs")
    dtype_name = descriptor["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise RuntimeExecutionArtifactError("tensor dtype differs")
    shape = _shape(descriptor["shape"], expected_name)
    dtype = _DTYPES[dtype_name]
    expected_length = _numel(shape) * torch.empty((), dtype=dtype).element_size()
    if len(raw) != expected_length or descriptor["length"] != expected_length:
        raise RuntimeExecutionArtifactError("tensor shape/byte length mismatch")
    if _sha256_bytes(raw) != descriptor["raw_sha256"]:
        raise RuntimeExecutionArtifactError("tensor raw-byte hash differs")
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    try:
        return tensor.view(dtype).reshape(shape).clone()
    except RuntimeError as error:
        raise RuntimeExecutionArtifactError(
            "tensor dtype/shape reconstruction failed"
        ) from error


def _read_packet(descriptor: Mapping[str, object], raw: bytes) -> HardCTAAPacket:
    if set(descriptor) != _PACKET_BLOB_KEYS:
        raise RuntimeExecutionArtifactError("packet descriptor schema differs")
    if (
        descriptor["name"] != "packet"
        or descriptor["kind"] != "packet"
        or descriptor["dtype"] != str(torch.uint8)
    ):
        raise RuntimeExecutionArtifactError("packet descriptor identity differs")
    cards_shape = _shape(descriptor["action_cards_shape"], "action cards")
    initial_shape = _shape(descriptor["initial_state_shape"], "initial state")
    schedule_shape = _shape(descriptor["schedule_shape"], "schedule")
    if (
        len(cards_shape) != 3
        or len(initial_shape) != 2
        or len(schedule_shape) != 2
        or initial_shape[0] != cards_shape[0]
        or schedule_shape[0] != cards_shape[0]
        or initial_shape[1] != cards_shape[2]
    ):
        raise RuntimeExecutionArtifactError("packet tensor geometry differs")
    bytes_per_row = (
        cards_shape[1] * cards_shape[2] + initial_shape[1] + schedule_shape[1]
    )
    expected_length = cards_shape[0] * bytes_per_row
    if len(raw) != expected_length or descriptor["length"] != expected_length:
        raise RuntimeExecutionArtifactError("packet shape/byte length mismatch")
    if _sha256_bytes(raw) != descriptor["raw_sha256"]:
        raise RuntimeExecutionArtifactError("packet raw-byte hash differs")
    rows = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        cards_shape[0], bytes_per_row
    )
    card_end = cards_shape[1] * cards_shape[2]
    try:
        packet = HardCTAAPacket(
            rows[:, :card_end].reshape(cards_shape).clone(),
            rows[:, card_end : card_end + initial_shape[1]]
            .reshape(initial_shape)
            .clone(),
            rows[:, card_end + initial_shape[1] :].reshape(schedule_shape).clone(),
        )
    except ValueError as error:
        raise RuntimeExecutionArtifactError("packet bytes are invalid") from error
    if packet_body(packet) != raw:
        raise RuntimeExecutionArtifactError("packet byte replay differs")
    return packet


def _failure_from_json(value: object) -> ExecutionFailure | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _FAILURE_KEYS:
        raise RuntimeExecutionArtifactError("execution failure schema differs")
    stage = _require_identifier(value["stage"], "failure stage")
    code = _require_identifier(value["code"], "failure code")
    _reject_leakage(value)
    return ExecutionFailure(stage, code)


def _hash_pairs_from_json(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise RuntimeExecutionArtifactError(f"{label} list differs")
    result: list[tuple[str, str]] = []
    previous = ""
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise RuntimeExecutionArtifactError(f"{label} entry differs")
        name = _require_identifier(row[0], f"{label} name")
        digest = _require_hash(row[1], f"{label} digest")
        if name <= previous:
            raise RuntimeExecutionArtifactError(f"{label} ordering differs")
        previous = name
        result.append((name, digest))
    return tuple(result)


def _decode_frame(
    raw: bytes, *, expected_kind: str
) -> ParentExecutionRecord | AttemptExecutionRecord:
    prefix = len(EXECUTION_ARTIFACT_MAGIC) + EXECUTION_ARTIFACT_HEADER.size
    if (
        len(raw) < prefix
        or raw[: len(EXECUTION_ARTIFACT_MAGIC)] != EXECUTION_ARTIFACT_MAGIC
    ):
        raise RuntimeExecutionArtifactError("execution artifact magic differs")
    header_length = EXECUTION_ARTIFACT_HEADER.unpack(
        raw[len(EXECUTION_ARTIFACT_MAGIC) : prefix]
    )[0]
    header_end = prefix + header_length
    if header_length == 0 or header_end > len(raw):
        raise RuntimeExecutionArtifactError("execution artifact header length differs")
    encoded_header = raw[prefix:header_end]
    header = _decode_json(encoded_header, label="execution artifact header")
    if encoded_header != _canonical_json_bytes(header):
        raise RuntimeExecutionArtifactError(
            "execution artifact header is not canonical"
        )
    if (
        set(header) != _FRAME_KEYS
        or header["schema"] != EXECUTION_ARTIFACT_SCHEMA
        or header["kind"] != expected_kind
    ):
        raise RuntimeExecutionArtifactError("execution artifact frame schema differs")
    descriptors = header["blobs"]
    if not isinstance(descriptors, list):
        raise RuntimeExecutionArtifactError("execution artifact blob table differs")
    body = raw[header_end:]
    decoded_blobs: dict[str, HardCTAAPacket | torch.Tensor] = {}
    expected_offset = 0
    prior_order = -1
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise RuntimeExecutionArtifactError("execution blob descriptor differs")
        name = descriptor.get("name")
        if name not in _BLOB_ORDER:
            raise RuntimeExecutionArtifactError("execution blob name differs")
        order = _BLOB_ORDER.index(str(name))
        if order <= prior_order:
            raise RuntimeExecutionArtifactError("execution blob order differs")
        prior_order = order
        offset = descriptor.get("offset")
        length = descriptor.get("length")
        if (
            type(offset) is not int
            or type(length) is not int
            or offset != expected_offset
            or length < 0
            or offset + length > len(body)
        ):
            raise RuntimeExecutionArtifactError("execution blob byte range differs")
        blob = body[offset : offset + length]
        if name == "packet":
            decoded_blobs[str(name)] = _read_packet(descriptor, blob)
        else:
            decoded_blobs[str(name)] = _read_tensor(
                descriptor, blob, expected_name=str(name)
            )
        expected_offset += length
    if expected_offset != len(body):
        raise RuntimeExecutionArtifactError("execution artifact has trailing bytes")
    snapshot_json = header["snapshot"]
    if snapshot_json is None:
        if descriptors:
            raise RuntimeExecutionArtifactError("snapshot-free record contains blobs")
        snapshot = None
    else:
        if not isinstance(snapshot_json, dict) or set(snapshot_json) != _SNAPSHOT_KEYS:
            raise RuntimeExecutionArtifactError("execution snapshot metadata differs")
        if snapshot_json["schema"] != SNAPSHOT_SCHEMA:
            raise RuntimeExecutionArtifactError("execution snapshot schema differs")
        expected_names = {
            name for name in _BLOB_ORDER if snapshot_json[f"{name}_blob"] is not None
        }
        for name in _BLOB_ORDER:
            reference = snapshot_json[f"{name}_blob"]
            if reference not in (None, name):
                raise RuntimeExecutionArtifactError("snapshot blob reference differs")
        if expected_names != set(decoded_blobs) or "packet" not in expected_names:
            raise RuntimeExecutionArtifactError("snapshot blob coverage differs")
        for required in ("state_route", "halted", "terminal"):
            if required not in expected_names:
                raise RuntimeExecutionArtifactError(
                    "required snapshot tensor is absent"
                )
        artifact_hashes = _hash_pairs_from_json(
            snapshot_json["artifact_hashes"], "snapshot artifact"
        )
        actual_hashes: list[tuple[str, str]] = []
        for name in _BLOB_ORDER:
            if name not in decoded_blobs:
                continue
            descriptor = next(row for row in descriptors if row["name"] == name)
            start = int(descriptor["offset"])
            end = start + int(descriptor["length"])
            blob = body[start:end]
            if name == "packet":
                digest = _sha256_bytes(blob)
            else:
                digest = _tensor_hash(
                    name,
                    str(descriptor["dtype"]),
                    list(descriptor["shape"]),
                    blob,
                )
            actual_hashes.append((name, digest))
        actual_hashes.sort()
        if artifact_hashes != tuple(actual_hashes):
            raise RuntimeExecutionArtifactError("snapshot tensor commitment differs")
        snapshot_payload = {
            "schema": SNAPSHOT_SCHEMA,
            "artifact_hashes": [list(item) for item in artifact_hashes],
        }
        snapshot_sha256 = _require_hash(
            snapshot_json["snapshot_sha256"], "snapshot_sha256"
        )
        if snapshot_sha256 != _sha256_json(snapshot_payload):
            raise RuntimeExecutionArtifactError("snapshot commitment differs")
        packet = decoded_blobs["packet"]
        if not isinstance(packet, HardCTAAPacket):
            raise RuntimeExecutionArtifactError("snapshot packet type differs")
        tensors = {name: decoded_blobs.get(name) for name in _TENSOR_FIELDS}
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in tensors.values()
        ):
            raise RuntimeExecutionArtifactError("snapshot tensor type differs")
        snapshot = ExecutionSnapshot(
            SNAPSHOT_SCHEMA,
            packet,
            tensors["h19_residual"],  # type: ignore[arg-type]
            tensors["h29_residual"],  # type: ignore[arg-type]
            tensors["state_route"],  # type: ignore[arg-type]
            tensors["composed_route"],  # type: ignore[arg-type]
            tensors["halted"],  # type: ignore[arg-type]
            tensors["terminal"],  # type: ignore[arg-type]
            artifact_hashes,
            snapshot_sha256,
        )
        _validate_snapshot_geometry(snapshot)
    record_json = header["record"]
    if not isinstance(record_json, dict):
        raise RuntimeExecutionArtifactError("execution record metadata differs")
    if expected_kind == "parent":
        return _decode_parent_record(record_json, snapshot)
    return _decode_attempt_record(record_json, snapshot)


def _validate_snapshot_geometry(snapshot: ExecutionSnapshot) -> None:
    packet = snapshot.packet
    batch = packet.schedule.shape[0]
    steps = packet.schedule.shape[1] + 1
    width = packet.initial_state.shape[1]
    if batch != 1:
        raise RuntimeExecutionArtifactError("execution snapshot packet batch differs")
    if (
        snapshot.state_route.dtype != torch.uint8
        or snapshot.state_route.shape != (steps, width)
        or snapshot.composed_route is None
        or snapshot.composed_route.dtype != torch.uint8
        or snapshot.composed_route.shape != (steps, width)
        or snapshot.halted.dtype != torch.bool
        or snapshot.halted.shape != (steps,)
        or snapshot.terminal.dtype != torch.uint8
        or snapshot.terminal.shape != (width,)
        or not torch.equal(snapshot.terminal, snapshot.state_route[-1])
    ):
        raise RuntimeExecutionArtifactError("execution trace tensor geometry differs")
    h19 = snapshot.h19_residual
    h29 = snapshot.h29_residual
    if (
        h19 is None
        or h29 is None
        or h19.ndim != 2
        or h19.shape != h29.shape
        or h19.dtype != h29.dtype
        or not (h19.dtype.is_floating_point or h19.dtype.is_complex)
    ):
        raise RuntimeExecutionArtifactError(
            "execution residual tensor geometry differs"
        )


def _validate_status(
    status: object,
    failure: ExecutionFailure | None,
    snapshot: ExecutionSnapshot | None,
    *,
    parent: bool,
) -> str:
    if status not in ("success", "failure"):
        raise RuntimeExecutionArtifactError("execution record status differs")
    if (status == "success") != (failure is None):
        raise RuntimeExecutionArtifactError("execution status/failure relation differs")
    if status == "success" and snapshot is None:
        raise RuntimeExecutionArtifactError("successful execution lacks a snapshot")
    if parent and status == "failure" and snapshot is not None:
        raise RuntimeExecutionArtifactError("failed parent contains a snapshot")
    return str(status)


def _decode_parent_record(
    value: dict[str, object], snapshot: ExecutionSnapshot | None
) -> ParentExecutionRecord:
    if set(value) != _PARENT_RECORD_KEYS or value["schema"] != PARENT_RECORD_SCHEMA:
        raise RuntimeExecutionArtifactError("parent execution record schema differs")
    failure = _failure_from_json(value["failure"])
    status = _validate_status(value["status"], failure, snapshot, parent=True)
    anchor_id = _require_identifier(value["anchor_id"], "parent anchor_id")
    program_hash = _require_hash(
        value["program_source_sha256"], "program_source_sha256"
    )
    expected_packet = _require_hash(
        value["expected_packet_sha256"], "expected_packet_sha256"
    )
    snapshot_hash = None if snapshot is None else snapshot.snapshot_sha256
    if value["snapshot_sha256"] != snapshot_hash:
        raise RuntimeExecutionArtifactError("parent snapshot reference differs")
    payload = {
        "schema": PARENT_RECORD_SCHEMA,
        "anchor_id": anchor_id,
        "status": status,
        "program_source_sha256": program_hash,
        "expected_packet_sha256": expected_packet,
        "snapshot_sha256": snapshot_hash,
        "failure": _failure_json(failure),
    }
    record_hash = _require_hash(value["record_sha256"], "parent record_sha256")
    if record_hash != _sha256_json(payload):
        raise RuntimeExecutionArtifactError("parent record commitment differs")
    return ParentExecutionRecord(
        PARENT_RECORD_SCHEMA,
        anchor_id,
        status,
        program_hash,
        expected_packet,
        snapshot,
        failure,
        record_hash,
    )


def _optional_hash(value: object, label: str) -> str | None:
    return None if value is None else _require_hash(value, label)


def _decode_attempt_record(
    value: dict[str, object], snapshot: ExecutionSnapshot | None
) -> AttemptExecutionRecord:
    if set(value) != _ATTEMPT_RECORD_KEYS or value["schema"] != ATTEMPT_RECORD_SCHEMA:
        raise RuntimeExecutionArtifactError("attempt execution record schema differs")
    failure = _failure_from_json(value["failure"])
    status = _validate_status(value["status"], failure, snapshot, parent=False)
    attempt_index = value["attempt_index"]
    if type(attempt_index) is not int or attempt_index < 0:
        raise RuntimeExecutionArtifactError("attempt index differs")
    attempt_id = _require_identifier(value["attempt_id"], "attempt_id")
    operation = _require_identifier(value["operation"], "operation")
    if operation == InterventionFamily.LATE_QUERY_SWAP.value:
        raise RuntimeExecutionArtifactError("late-query operation entered artifact")
    anchor_id = _require_identifier(value["anchor_id"], "attempt anchor_id")
    donor = value["donor_anchor_id"]
    if donor is not None:
        donor = _require_identifier(donor, "donor_anchor_id")
    parent_hash = _require_hash(value["parent_record_sha256"], "parent_record_sha256")
    committed_program = _optional_hash(
        value["committed_program_source_sha256"],
        "committed_program_source_sha256",
    )
    committed_packet = _optional_hash(
        value["committed_packet_sha256"], "committed_packet_sha256"
    )
    observed_program = _optional_hash(
        value["observed_program_source_sha256"],
        "observed_program_source_sha256",
    )
    observed_packet = _optional_hash(
        value["observed_packet_sha256"], "observed_packet_sha256"
    )
    snapshot_hash = None if snapshot is None else snapshot.snapshot_sha256
    if value["snapshot_sha256"] != snapshot_hash:
        raise RuntimeExecutionArtifactError("attempt snapshot reference differs")
    if snapshot is None and observed_packet is not None:
        raise RuntimeExecutionArtifactError("snapshot-free attempt observes a packet")
    if snapshot is not None:
        actual_packet = dict(snapshot.artifact_hashes)["packet"]
        if observed_packet != actual_packet:
            raise RuntimeExecutionArtifactError("attempt observed packet differs")
    extras = _hash_pairs_from_json(value["extra_artifact_hashes"], "extra artifact")
    payload = {
        "schema": ATTEMPT_RECORD_SCHEMA,
        "attempt_index": attempt_index,
        "attempt_id": attempt_id,
        "operation": operation,
        "anchor_id": anchor_id,
        "donor_anchor_id": donor,
        "status": status,
        "parent_record_sha256": parent_hash,
        "committed_program_source_sha256": committed_program,
        "committed_packet_sha256": committed_packet,
        "observed_program_source_sha256": observed_program,
        "observed_packet_sha256": observed_packet,
        "snapshot_sha256": snapshot_hash,
        "extra_artifact_hashes": [list(item) for item in extras],
        "failure": _failure_json(failure),
    }
    record_hash = _require_hash(value["record_sha256"], "attempt record_sha256")
    if record_hash != _sha256_json(payload):
        raise RuntimeExecutionArtifactError("attempt record commitment differs")
    return AttemptExecutionRecord(
        ATTEMPT_RECORD_SCHEMA,
        attempt_index,
        attempt_id,
        operation,
        anchor_id,
        donor,
        status,
        parent_hash,
        committed_program,
        committed_packet,
        observed_program,
        observed_packet,
        snapshot,
        extras,
        failure,
        record_hash,
    )


def _execution_sha256(result: RuntimeExecutionResult) -> str:
    payload = {
        "schema": RUNTIME_EXECUTION_SCHEMA,
        "execution_projection_schema": EXECUTION_PROJECTION_SCHEMA,
        "projection_sha256": result.projection_sha256,
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "parents": [record.record_sha256 for record in result.parents],
        "attempts": [record.record_sha256 for record in result.attempts],
    }
    return _sha256_json(payload)


def _validate_complete_result(result: RuntimeExecutionResult) -> None:
    if (
        not isinstance(result, RuntimeExecutionResult)
        or result.schema != RUNTIME_EXECUTION_SCHEMA
        or result.execution_projection_schema != EXECUTION_PROJECTION_SCHEMA
        or result.scored_row_count != LOCKED_SCORED_ROW_COUNT
        or result.runtime_attempts_affect_scored_denominator is not False
    ):
        raise RuntimeExecutionArtifactError("runtime execution envelope differs")
    _require_hash(result.projection_sha256, "projection_sha256")
    if len(result.parents) != RUNTIME_PANEL_SIZE:
        raise RuntimeExecutionArtifactError("runtime parent coverage differs")
    parent_ids = [record.anchor_id for record in result.parents]
    if len(set(parent_ids)) != RUNTIME_PANEL_SIZE:
        raise RuntimeExecutionArtifactError("runtime parent identities differ")
    parent_by_id: dict[str, ParentExecutionRecord] = {}
    for record in result.parents:
        decoded = _decode_frame(_frame_bytes(record), expected_kind="parent")
        if not isinstance(decoded, ParentExecutionRecord):
            raise AssertionError("parent frame replay type differs")
        parent_by_id[record.anchor_id] = record
    if len(result.attempts) != EXPECTED_PREQUERY_ATTEMPT_COUNT:
        raise RuntimeExecutionArtifactError("runtime attempt coverage differs")
    seen_attempt_ids: set[str] = set()
    previous_attempt_index = -1
    for record in result.attempts:
        if (
            record.attempt_index <= previous_attempt_index
            or record.attempt_id in seen_attempt_ids
        ):
            raise RuntimeExecutionArtifactError(
                "runtime attempts are missing, extra, or reordered"
            )
        previous_attempt_index = record.attempt_index
        seen_attempt_ids.add(record.attempt_id)
        if record.operation == InterventionFamily.LATE_QUERY_SWAP.value:
            raise RuntimeExecutionArtifactError(
                "late-query operation entered runtime execution"
            )
        if record.anchor_id not in parent_by_id:
            raise RuntimeExecutionArtifactError("attempt anchor binding differs")
        if record.parent_record_sha256 != parent_by_id[record.anchor_id].record_sha256:
            raise RuntimeExecutionArtifactError("attempt parent binding differs")
        if (
            record.donor_anchor_id is not None
            and record.donor_anchor_id not in parent_by_id
        ):
            raise RuntimeExecutionArtifactError("attempt donor binding differs")
        decoded = _decode_frame(_frame_bytes(record), expected_kind="attempt")
        if not isinstance(decoded, AttemptExecutionRecord):
            raise AssertionError("attempt frame replay type differs")
    if result.execution_sha256 != _execution_sha256(result):
        raise RuntimeExecutionArtifactError("runtime execution commitment differs")


def _aggregate_value(
    result: RuntimeExecutionResult,
    parent_hashes: Sequence[str],
    attempt_hashes: Sequence[str],
) -> dict[str, object]:
    parent_refs = [
        {
            "anchor_id": record.anchor_id,
            "status": record.status,
            "record_sha256": record.record_sha256,
            "raw_output_artifact_sha256": digest,
        }
        for record, digest in zip(result.parents, parent_hashes)
    ]
    attempt_refs = [
        {
            "attempt_index": record.attempt_index,
            "attempt_id": record.attempt_id,
            "operation": record.operation,
            "anchor_id": record.anchor_id,
            "status": record.status,
            "record_sha256": record.record_sha256,
            "raw_output_artifact_sha256": digest,
        }
        for record, digest in zip(result.attempts, attempt_hashes)
    ]
    value: dict[str, object] = {
        "schema": EXECUTION_AGGREGATE_SCHEMA,
        "runtime_execution_schema": result.schema,
        "execution_projection_schema": result.execution_projection_schema,
        "projection_sha256": result.projection_sha256,
        "scored_row_count": result.scored_row_count,
        "runtime_attempts_affect_scored_denominator": (
            result.runtime_attempts_affect_scored_denominator
        ),
        "runtime_panel_size": len(result.parents),
        "parent_count": len(result.parents),
        "attempt_count": len(result.attempts),
        "execution_sha256": result.execution_sha256,
        "parent_artifacts": parent_refs,
        "parent_artifacts_sha256": _sha256_json(parent_refs),
        "attempt_artifacts": attempt_refs,
        "attempt_artifacts_sha256": _sha256_json(attempt_refs),
    }
    _reject_leakage(value)
    return value


def _artifact_name(digest: str) -> str:
    return _require_hash(digest, "raw artifact digest") + EXECUTION_ARTIFACT_SUFFIX


def write_runtime_execution_artifact(
    aggregate_path: Path,
    artifact_directory: Path,
    result: RuntimeExecutionResult,
) -> RuntimeExecutionArtifactIndex:
    """Publish all raw records, then one canonical immutable aggregate."""

    _validate_complete_result(result)
    artifact_descriptor = _open_directory(artifact_directory)
    try:
        parent_hashes: list[str] = []
        for record in result.parents:
            raw = _frame_bytes(record)
            if len(raw) > _MAX_RAW_ARTIFACT_BYTES:
                raise RuntimeExecutionArtifactError("parent raw artifact is too large")
            digest = _sha256_bytes(raw)
            _publish_immutable_at(
                artifact_descriptor,
                _artifact_name(digest),
                raw,
                allow_identical=True,
            )
            parent_hashes.append(digest)
        attempt_hashes: list[str] = []
        for record in result.attempts:
            raw = _frame_bytes(record)
            if len(raw) > _MAX_RAW_ARTIFACT_BYTES:
                raise RuntimeExecutionArtifactError("attempt raw artifact is too large")
            digest = _sha256_bytes(raw)
            _publish_immutable_at(
                artifact_descriptor,
                _artifact_name(digest),
                raw,
                allow_identical=True,
            )
            attempt_hashes.append(digest)
    finally:
        os.close(artifact_descriptor)
    aggregate = _aggregate_value(result, parent_hashes, attempt_hashes)
    aggregate_raw = _canonical_json_bytes(aggregate) + b"\n"
    if len(aggregate_raw) > _MAX_AGGREGATE_BYTES:
        raise RuntimeExecutionArtifactError("execution aggregate is too large")
    _publish_immutable_path(aggregate_path, aggregate_raw)
    outputs = tuple(
        {
            "attempt_id": record.attempt_id,
            "status": record.status,
            "raw_output_artifact_sha256": digest,
        }
        for record, digest in zip(result.attempts, attempt_hashes)
    )
    return RuntimeExecutionArtifactIndex(
        _sha256_bytes(aggregate_raw),
        tuple(parent_hashes),
        tuple(attempt_hashes),
        outputs,
    )


def _validate_aggregate(value: dict[str, object]) -> None:
    if set(value) != _AGGREGATE_KEYS:
        raise RuntimeExecutionArtifactError("execution aggregate schema differs")
    if (
        value["schema"] != EXECUTION_AGGREGATE_SCHEMA
        or value["runtime_execution_schema"] != RUNTIME_EXECUTION_SCHEMA
        or value["execution_projection_schema"] != EXECUTION_PROJECTION_SCHEMA
        or value["scored_row_count"] != LOCKED_SCORED_ROW_COUNT
        or value["runtime_attempts_affect_scored_denominator"] is not False
        or value["runtime_panel_size"] != RUNTIME_PANEL_SIZE
        or value["parent_count"] != RUNTIME_PANEL_SIZE
        or value["attempt_count"] != EXPECTED_PREQUERY_ATTEMPT_COUNT
    ):
        raise RuntimeExecutionArtifactError("execution aggregate contract differs")
    for key in (
        "projection_sha256",
        "execution_sha256",
        "parent_artifacts_sha256",
        "attempt_artifacts_sha256",
    ):
        _require_hash(value[key], key)
    parents = value["parent_artifacts"]
    attempts = value["attempt_artifacts"]
    if (
        not isinstance(parents, list)
        or len(parents) != RUNTIME_PANEL_SIZE
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_PREQUERY_ATTEMPT_COUNT
    ):
        raise RuntimeExecutionArtifactError("execution aggregate coverage differs")
    if value["parent_artifacts_sha256"] != _sha256_json(parents):
        raise RuntimeExecutionArtifactError("parent aggregate commitment differs")
    if value["attempt_artifacts_sha256"] != _sha256_json(attempts):
        raise RuntimeExecutionArtifactError("attempt aggregate commitment differs")


def read_runtime_execution_artifact_bundle(
    aggregate_path: Path,
    artifact_directory: Path,
    *,
    expected_aggregate_sha256: str,
    expected_projection_sha256: str | None = None,
) -> tuple[RuntimeExecutionResult, RuntimeExecutionArtifactIndex]:
    """Read and independently replay a bundle, returning its verified index."""

    expected_aggregate_sha256 = _require_hash(
        expected_aggregate_sha256, "expected aggregate SHA-256"
    )
    aggregate_raw = _read_immutable_path(
        aggregate_path, maximum_bytes=_MAX_AGGREGATE_BYTES
    )
    if _sha256_bytes(aggregate_raw) != expected_aggregate_sha256:
        raise RuntimeExecutionArtifactError("execution aggregate hash substitution")
    aggregate = _decode_json(aggregate_raw, label="execution aggregate")
    if aggregate_raw != _canonical_json_bytes(aggregate) + b"\n":
        raise RuntimeExecutionArtifactError("execution aggregate is not canonical")
    _validate_aggregate(aggregate)
    if expected_projection_sha256 is not None and aggregate[
        "projection_sha256"
    ] != _require_hash(expected_projection_sha256, "expected projection SHA-256"):
        raise RuntimeExecutionArtifactError("execution projection binding differs")
    artifact_descriptor = _open_directory(artifact_directory)
    try:
        parents: list[ParentExecutionRecord] = []
        parent_refs = aggregate["parent_artifacts"]
        assert isinstance(parent_refs, list)
        for reference in parent_refs:
            if not isinstance(reference, dict) or set(reference) != _PARENT_REF_KEYS:
                raise RuntimeExecutionArtifactError("parent artifact reference differs")
            digest = _require_hash(
                reference["raw_output_artifact_sha256"], "parent raw artifact hash"
            )
            raw = _read_immutable_at(
                artifact_descriptor,
                _artifact_name(digest),
                maximum_bytes=_MAX_RAW_ARTIFACT_BYTES,
            )
            if _sha256_bytes(raw) != digest:
                raise RuntimeExecutionArtifactError("parent artifact hash substitution")
            record = _decode_frame(raw, expected_kind="parent")
            if not isinstance(record, ParentExecutionRecord):
                raise AssertionError("parent artifact replay type differs")
            if reference != {
                "anchor_id": record.anchor_id,
                "status": record.status,
                "record_sha256": record.record_sha256,
                "raw_output_artifact_sha256": digest,
            }:
                raise RuntimeExecutionArtifactError(
                    "parent artifact reference mismatch"
                )
            parents.append(record)
        attempts: list[AttemptExecutionRecord] = []
        attempt_refs = aggregate["attempt_artifacts"]
        assert isinstance(attempt_refs, list)
        for reference in attempt_refs:
            if not isinstance(reference, dict) or set(reference) != _ATTEMPT_REF_KEYS:
                raise RuntimeExecutionArtifactError(
                    "attempt artifact reference differs"
                )
            digest = _require_hash(
                reference["raw_output_artifact_sha256"], "attempt raw artifact hash"
            )
            raw = _read_immutable_at(
                artifact_descriptor,
                _artifact_name(digest),
                maximum_bytes=_MAX_RAW_ARTIFACT_BYTES,
            )
            if _sha256_bytes(raw) != digest:
                raise RuntimeExecutionArtifactError(
                    "attempt artifact hash substitution"
                )
            record = _decode_frame(raw, expected_kind="attempt")
            if not isinstance(record, AttemptExecutionRecord):
                raise AssertionError("attempt artifact replay type differs")
            if reference != {
                "attempt_index": record.attempt_index,
                "attempt_id": record.attempt_id,
                "operation": record.operation,
                "anchor_id": record.anchor_id,
                "status": record.status,
                "record_sha256": record.record_sha256,
                "raw_output_artifact_sha256": digest,
            }:
                raise RuntimeExecutionArtifactError(
                    "attempt artifact reference mismatch"
                )
            attempts.append(record)
    finally:
        os.close(artifact_descriptor)
    result = RuntimeExecutionResult(
        RUNTIME_EXECUTION_SCHEMA,
        EXECUTION_PROJECTION_SCHEMA,
        str(aggregate["projection_sha256"]),
        LOCKED_SCORED_ROW_COUNT,
        False,
        tuple(parents),
        tuple(attempts),
        str(aggregate["execution_sha256"]),
    )
    _validate_complete_result(result)
    expected = _aggregate_value(
        result,
        [str(row["raw_output_artifact_sha256"]) for row in parent_refs],
        [str(row["raw_output_artifact_sha256"]) for row in attempt_refs],
    )
    if aggregate != expected:
        raise RuntimeExecutionArtifactError("execution aggregate replay differs")
    index = RuntimeExecutionArtifactIndex(
        expected_aggregate_sha256,
        tuple(str(row["raw_output_artifact_sha256"]) for row in parent_refs),
        tuple(str(row["raw_output_artifact_sha256"]) for row in attempt_refs),
        tuple(
            {
                "attempt_id": record.attempt_id,
                "status": record.status,
                "raw_output_artifact_sha256": str(
                    reference["raw_output_artifact_sha256"]
                ),
            }
            for record, reference in zip(result.attempts, attempt_refs)
        ),
    )
    return result, index


def read_runtime_execution_artifact(
    aggregate_path: Path,
    artifact_directory: Path,
    *,
    expected_aggregate_sha256: str,
    expected_projection_sha256: str | None = None,
) -> RuntimeExecutionResult:
    """Read and independently replay an immutable content-addressed bundle."""

    result, _ = read_runtime_execution_artifact_bundle(
        aggregate_path,
        artifact_directory,
        expected_aggregate_sha256=expected_aggregate_sha256,
        expected_projection_sha256=expected_projection_sha256,
    )
    return result
