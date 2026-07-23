"""Signed, query-blind receipt for CTAA runtime execution.

The runtime execution projection is the sole source contract accepted here.
Every projected attempt must have exactly one ordered raw-output commitment,
including failed attempts.  The receipt is deliberately pre-query: it cannot
contain query source/position data, answers, the deferred late-query operation,
oracle access, or producer-authored scientific gate decisions.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ctaa_intervention_protocol import RuntimeInterventionPlan
from ctaa_run_contract import canonical_json
from ctaa_runtime_execution_artifact import (
    EXECUTION_AGGREGATE_SCHEMA,
    RuntimeExecutionArtifactError,
    RuntimeExecutionArtifactIndex,
    read_runtime_execution_artifact_bundle,
)
from ctaa_runtime_execution_engine import RuntimeExecutionResult
from ctaa_runtime_execution_projection import (
    EXECUTION_PROJECTION_SCHEMA,
    validate_execution_projection,
)


EXECUTION_RECEIPT_SCHEMA = "r12_ctaa_runtime_execution_receipt_v2"
EXECUTION_ATTEMPT_RECEIPT_SCHEMA = "r12_ctaa_runtime_attempt_receipt_v1"
EXECUTION_STATUSES = ("success", "failure")

_RECORD_KEYS = frozenset({"payload", "signature", "receipt_sha256"})
_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "execution_projection_schema",
        "execution_projection_file_sha256",
        "execution_projection_sha256",
        "execution_aggregate_schema",
        "execution_aggregate_sha256",
        "execution_sha256",
        "plan_sha256",
        "board_manifest_sha256",
        "board_tree_sha256",
        "run_contract_sha256",
        "compiler_sha256",
        "core_sha256",
        "core_kind",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "base_raw_evidence_receipt_sha256",
        "runtime_implementation_sha256",
        "selection_seed",
        "selection_seed_receipt_sha256",
        "training_seed",
        "arm_id",
        "partition",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order_sha256",
        "execution_attempt_count",
        "execution_attempt_ids_sha256",
        "attempts",
        "attempts_sha256",
        "source_deletion_probe_artifact_sha256s",
        "query_isolation_probe_artifact_sha256s",
        "oracle_access_count",
        "signing_public_key",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "attempt_index",
        "attempt_id",
        "operation",
        "status",
        "raw_output_artifact_sha256",
    }
)
_ATTEMPT_INPUT_KEYS = frozenset({"attempt_id", "status", "raw_output_artifact_sha256"})
_PROJECTION_BINDINGS = (
    "plan_sha256",
    "board_manifest_sha256",
    "board_tree_sha256",
    "run_contract_sha256",
    "compiler_sha256",
    "core_sha256",
    "core_kind",
    "tokenizer_sha256",
    "base_checkpoint_sha256",
    "base_raw_evidence_receipt_sha256",
    "runtime_implementation_sha256",
    "selection_seed",
    "selection_seed_receipt_sha256",
    "training_seed",
    "arm_id",
    "partition",
    "anchor_panel_sha256",
    "donor_registry_sha256",
    "batch_order_sha256",
)
_HASH_BINDINGS = frozenset(
    key for key in _PROJECTION_BINDINGS if key.endswith("_sha256")
)
_FORBIDDEN_FRAGMENTS = (
    "answer",
    "query_position",
    "query_source",
    "late_query_swap",
)
_MAX_FILE_BYTES = 128 * 1024 * 1024

# Edwards25519 constants used to reject non-canonical and non-prime-order keys
# before asking the cryptography backend to verify the signature.
_FIELD_P = 2**255 - 19
_GROUP_L = 2**252 + 27742317777372353535851937790883648493
_CURVE_D = (-121665 * pow(121666, _FIELD_P - 2, _FIELD_P)) % _FIELD_P
_SQRT_M1 = pow(2, (_FIELD_P - 1) // 4, _FIELD_P)
_IDENTITY = (0, 1)


class ExecutionReceiptError(ValueError):
    """The signed pre-query execution receipt failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole byte representation accepted for signing."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ExecutionReceiptError("receipt is not canonical JSON") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_hash(value: object, label: str) -> str:
    if not _is_lower_hex(value, 64):
        raise ExecutionReceiptError(f"invalid {label}")
    return str(value)


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value.strip() != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ExecutionReceiptError(f"invalid {label}")
    return value


def _reject_query_leakage(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
                raise ExecutionReceiptError(f"pre-query receipt leaks field: {key}")
            _reject_query_leakage(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_query_leakage(item)
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
            raise ExecutionReceiptError("pre-query receipt leaks deferred query data")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionReceiptError(f"duplicate receipt key: {key}")
        result[key] = value
    return result


def _decode_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ExecutionReceiptError(f"non-finite {label} constant: {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionReceiptError(f"malformed {label} JSON") from error
    if not isinstance(value, dict):
        raise ExecutionReceiptError(f"{label} root must be an object")
    return value


def _path_text(path: Path) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise ExecutionReceiptError("path must be filesystem text") from error
    if not isinstance(raw, str) or "\x00" in raw:
        raise ExecutionReceiptError("path must be filesystem text")
    if not os.path.isabs(raw) or os.path.normpath(raw) != raw or raw == "/":
        raise ExecutionReceiptError("path must be absolute and normalized")
    if any(part in ("", ".", "..") for part in raw.split("/")[1:]):
        raise ExecutionReceiptError("path contains an unsafe component")
    return raw


def _open_parent_directory(path: Path) -> tuple[int, str]:
    raw = _path_text(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ExecutionReceiptError("O_NOFOLLOW is required")
    components = raw.split("/")[1:]
    name = components[-1]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ExecutionReceiptError(
                    "receipt path parent is missing, symlinked, or not a directory"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ExecutionReceiptError("receipt path parent is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, name
    except Exception:
        os.close(descriptor)
        raise


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


def _read_immutable_bytes(path: Path) -> bytes:
    parent_descriptor, name = _open_parent_directory(path)
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ExecutionReceiptError(
                "immutable receipt input is missing or symlinked"
            ) from error
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_size <= 0
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise ExecutionReceiptError(
                "receipt input must be immutable, regular, and single-link"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ExecutionReceiptError("receipt input changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ExecutionReceiptError("receipt input grew during read")
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise ExecutionReceiptError("receipt input changed during read")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _write_immutable_bytes(path: Path, raw: bytes) -> None:
    if not raw or len(raw) > _MAX_FILE_BYTES:
        raise ExecutionReceiptError("receipt output size is invalid")
    parent_descriptor, name = _open_parent_directory(path)
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
        descriptor = os.open(
            temporary,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise ExecutionReceiptError("receipt write made no progress")
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
            raise ExecutionReceiptError("immutable receipt write verification failed")
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"refusing existing execution receipt: {path}"
            ) from None
        linked = True
        os.unlink(temporary, dir_fd=parent_descriptor)
        temporary = ""
        final = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_mode & 0o222
            or final.st_size != len(raw)
        ):
            raise ExecutionReceiptError("published receipt is not immutable")
        os.fsync(parent_descriptor)
    except OSError as error:
        if linked:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if isinstance(error, FileExistsError):
            raise
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ExecutionReceiptError("receipt path contains a symlink") from error
        raise
    except Exception:
        if linked:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _point_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = left
    x2, y2 = right
    product = (_CURVE_D * x1 * x2 * y1 * y2) % _FIELD_P
    x_denominator = (1 + product) % _FIELD_P
    y_denominator = (1 - product) % _FIELD_P
    if x_denominator == 0 or y_denominator == 0:
        raise ExecutionReceiptError("invalid Ed25519 public key point")
    x3 = (x1 * y2 + y1 * x2) * pow(x_denominator, _FIELD_P - 2, _FIELD_P)
    y3 = (y1 * y2 + x1 * x2) * pow(y_denominator, _FIELD_P - 2, _FIELD_P)
    return x3 % _FIELD_P, y3 % _FIELD_P


def _scalar_multiply(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


def _validate_public_key_bytes(raw_key: bytes) -> bytes:
    if not isinstance(raw_key, bytes) or len(raw_key) != 32:
        raise ExecutionReceiptError("Ed25519 public key must be 32 bytes")
    encoded = int.from_bytes(raw_key, "little")
    sign_bit = encoded >> 255
    y = encoded & ((1 << 255) - 1)
    if y >= _FIELD_P:
        raise ExecutionReceiptError("non-canonical Ed25519 public key")
    y_squared = y * y % _FIELD_P
    denominator = (_CURVE_D * y_squared + 1) % _FIELD_P
    if denominator == 0:
        raise ExecutionReceiptError("invalid Ed25519 public key point")
    x_squared = (y_squared - 1) * pow(denominator, _FIELD_P - 2, _FIELD_P) % _FIELD_P
    x = pow(x_squared, (_FIELD_P + 3) // 8, _FIELD_P)
    if x * x % _FIELD_P != x_squared:
        x = x * _SQRT_M1 % _FIELD_P
    if x * x % _FIELD_P != x_squared:
        raise ExecutionReceiptError("Ed25519 public key is not on curve")
    if x == 0 and sign_bit:
        raise ExecutionReceiptError("non-canonical Ed25519 public key sign")
    if (x & 1) != sign_bit:
        x = (-x) % _FIELD_P
    point = (x, y)
    if point == _IDENTITY or _scalar_multiply(point, _GROUP_L) != _IDENTITY:
        raise ExecutionReceiptError(
            "Ed25519 public key is not in the prime-order subgroup"
        )
    return raw_key


def _public_key_bytes(
    verification_key: bytes | Ed25519PublicKey,
) -> bytes:
    if isinstance(verification_key, Ed25519PublicKey):
        raw = verification_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    elif isinstance(verification_key, bytes):
        raw = verification_key
    else:
        raise ExecutionReceiptError("unsupported Ed25519 public key type")
    return _validate_public_key_bytes(raw)


def _signing_public_key(signing_key: Ed25519PrivateKey) -> bytes:
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise TypeError("signing_key must be an Ed25519PrivateKey")
    return _public_key_bytes(signing_key.public_key())


def _load_projection(
    path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> tuple[dict[str, object], str]:
    raw = _read_immutable_bytes(path)
    value = _decode_object(raw, label="execution projection")
    expected_raw = (canonical_json(value) + "\n").encode("ascii")
    if raw != expected_raw:
        raise ExecutionReceiptError("execution projection is not canonical")
    try:
        projection = validate_execution_projection(value, plan)
    except ValueError as error:
        raise ExecutionReceiptError("execution projection validation failed") from error
    if projection.get("schema") != EXECUTION_PROJECTION_SCHEMA:
        raise ExecutionReceiptError("execution projection schema differs")
    return projection, _sha256_bytes(raw)


def _validate_attempt_outputs(
    outputs: Sequence[Mapping[str, object]],
    projection_attempts: object,
) -> list[dict[str, object]]:
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise ExecutionReceiptError("attempt outputs must be an ordered sequence")
    if not isinstance(projection_attempts, list):
        raise ExecutionReceiptError("execution projection attempts differ")
    if len(outputs) != len(projection_attempts):
        raise ExecutionReceiptError("attempt output count differs from projection")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, (output, projected) in enumerate(zip(outputs, projection_attempts)):
        if not isinstance(output, Mapping) or set(output) != _ATTEMPT_INPUT_KEYS:
            raise ExecutionReceiptError("attempt output schema differs")
        if not isinstance(projected, Mapping):
            raise ExecutionReceiptError("execution projection attempt differs")
        attempt_id = _require_identifier(output["attempt_id"], "attempt_id")
        if attempt_id in seen:
            raise ExecutionReceiptError("duplicate attempt output")
        seen.add(attempt_id)
        projected_index = projected.get("attempt_index")
        if type(projected_index) is not int or int(projected_index) < 0:
            raise ExecutionReceiptError("projection attempt index differs")
        if attempt_id != projected.get("attempt_id"):
            raise ExecutionReceiptError("attempt outputs are missing or reordered")
        status_value = output["status"]
        if status_value not in EXECUTION_STATUSES:
            raise ExecutionReceiptError("invalid execution attempt status")
        artifact_hash = _require_hash(
            output["raw_output_artifact_sha256"],
            "raw_output_artifact_sha256",
        )
        operation = _require_identifier(projected.get("operation"), "operation")
        if operation == "late_query_swap":
            raise ExecutionReceiptError("deferred operation entered pre-query receipt")
        row = {
            "schema": EXECUTION_ATTEMPT_RECEIPT_SCHEMA,
            "attempt_index": projected_index,
            "attempt_id": attempt_id,
            "operation": operation,
            "status": status_value,
            "raw_output_artifact_sha256": artifact_hash,
        }
        if set(row) != _ATTEMPT_KEYS:  # pragma: no cover - construction invariant
            raise AssertionError("attempt receipt schema differs")
        rows.append(row)
    return rows


def _validate_payload_shape(
    payload: object, *, expected_key_hex: str
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ExecutionReceiptError("execution receipt payload schema differs")
    _reject_query_leakage(payload)
    if payload["schema"] != EXECUTION_RECEIPT_SCHEMA:
        raise ExecutionReceiptError("execution receipt schema differs")
    if payload["execution_projection_schema"] != EXECUTION_PROJECTION_SCHEMA:
        raise ExecutionReceiptError("execution projection schema binding differs")
    for key in _HASH_BINDINGS | {
        "execution_projection_file_sha256",
        "execution_projection_sha256",
        "execution_aggregate_sha256",
        "execution_sha256",
        "execution_attempt_ids_sha256",
        "attempts_sha256",
    }:
        _require_hash(payload[key], key)
    if payload["execution_aggregate_schema"] != EXECUTION_AGGREGATE_SCHEMA:
        raise ExecutionReceiptError("execution aggregate schema binding differs")
    for key in ("core_kind", "arm_id", "partition"):
        _require_identifier(payload[key], key)
    for key in ("selection_seed", "training_seed"):
        if type(payload[key]) is not int or int(payload[key]) < 0:
            raise ExecutionReceiptError(f"invalid {key}")
    if (
        type(payload["execution_attempt_count"]) is not int
        or int(payload["execution_attempt_count"]) <= 0
    ):
        raise ExecutionReceiptError("invalid execution_attempt_count")
    if (
        payload["oracle_access_count"] != 0
        or type(payload["oracle_access_count"]) is not int
    ):
        raise ExecutionReceiptError("pre-query receipt must have zero oracle access")
    if payload["signing_public_key"] != expected_key_hex:
        raise ExecutionReceiptError("execution receipt uses the wrong signing key")
    attempts = payload["attempts"]
    if not isinstance(attempts, list):
        raise ExecutionReceiptError("execution receipt attempts differ")
    if len(attempts) != payload["execution_attempt_count"]:
        raise ExecutionReceiptError("execution receipt attempt count differs")
    previous_attempt_index = -1
    for row in attempts:
        if not isinstance(row, dict) or set(row) != _ATTEMPT_KEYS:
            raise ExecutionReceiptError("attempt receipt schema differs")
        if row["schema"] != EXECUTION_ATTEMPT_RECEIPT_SCHEMA:
            raise ExecutionReceiptError("attempt receipt schema version differs")
        if (
            type(row["attempt_index"]) is not int
            or int(row["attempt_index"]) <= previous_attempt_index
        ):
            raise ExecutionReceiptError("attempt receipt order differs")
        previous_attempt_index = int(row["attempt_index"])
        _require_identifier(row["attempt_id"], "attempt_id")
        operation = _require_identifier(row["operation"], "operation")
        if operation == "late_query_swap":
            raise ExecutionReceiptError("deferred operation entered pre-query receipt")
        if row["status"] not in EXECUTION_STATUSES:
            raise ExecutionReceiptError("invalid execution attempt status")
        _require_hash(row["raw_output_artifact_sha256"], "raw output artifact hash")
    for key in (
        "source_deletion_probe_artifact_sha256s",
        "query_isolation_probe_artifact_sha256s",
    ):
        hashes = payload[key]
        if not isinstance(hashes, list) or not hashes:
            raise ExecutionReceiptError(f"invalid {key}")
        for artifact_hash in hashes:
            _require_hash(artifact_hash, key)
    return payload


def _record_hash(payload: Mapping[str, object], signature: str) -> str:
    return _sha256_json({"payload": dict(payload), "signature": signature})


def _make_record(
    payload: Mapping[str, object], signing_key: Ed25519PrivateKey
) -> dict[str, object]:
    signature = signing_key.sign(canonical_json_bytes(dict(payload))).hex()
    return {
        "payload": dict(payload),
        "signature": signature,
        "receipt_sha256": _record_hash(payload, signature),
    }


def _expected_payload(
    *,
    projection: Mapping[str, object],
    projection_file_sha256: str,
    plan_partition: str,
    execution_result: RuntimeExecutionResult,
    execution_index: RuntimeExecutionArtifactIndex,
    signing_public_key: str,
) -> dict[str, object]:
    projection_attempts = projection.get("attempts")
    if not isinstance(projection_attempts, list):
        raise ExecutionReceiptError("execution projection attempts differ")
    if len(execution_result.attempts) != len(execution_index.attempt_outputs):
        raise ExecutionReceiptError("execution artifact attempt index differs")
    for record, output, projected in zip(
        execution_result.attempts,
        execution_index.attempt_outputs,
        projection_attempts,
    ):
        if not isinstance(projected, Mapping):
            raise ExecutionReceiptError("execution projection attempt differs")
        if (
            record.attempt_index != projected.get("attempt_index")
            or record.attempt_id != projected.get("attempt_id")
            or record.operation != projected.get("operation")
            or output.get("attempt_id") != record.attempt_id
            or output.get("status") != record.status
        ):
            raise ExecutionReceiptError("execution artifact differs from projection")
    attempts = _validate_attempt_outputs(
        execution_index.attempt_outputs, projection_attempts
    )
    attempt_ids = [str(row["attempt_id"]) for row in attempts]
    source_probe_hashes = [
        str(row["raw_output_artifact_sha256"])
        for row in attempts
        if row["operation"] == "source_deletion"
    ]
    isolation_probe_hashes = [
        str(row["raw_output_artifact_sha256"])
        for row in attempts
        if row["operation"] == "query_isolation"
    ]
    if not source_probe_hashes or not isolation_probe_hashes:
        raise ExecutionReceiptError("required custody probe attempts are missing")
    payload: dict[str, object] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "execution_projection_schema": EXECUTION_PROJECTION_SCHEMA,
        "execution_projection_file_sha256": projection_file_sha256,
        "execution_projection_sha256": projection["projection_sha256"],
        "execution_aggregate_schema": EXECUTION_AGGREGATE_SCHEMA,
        "execution_aggregate_sha256": execution_index.aggregate_sha256,
        "execution_sha256": execution_result.execution_sha256,
        **{
            key: (plan_partition if key == "partition" else projection[key])
            for key in _PROJECTION_BINDINGS
        },
        "execution_attempt_count": len(attempts),
        "execution_attempt_ids_sha256": _sha256_json(attempt_ids),
        "attempts": attempts,
        "attempts_sha256": _sha256_json(attempts),
        "source_deletion_probe_artifact_sha256s": source_probe_hashes,
        "query_isolation_probe_artifact_sha256s": isolation_probe_hashes,
        "oracle_access_count": 0,
        "signing_public_key": signing_public_key,
    }
    if set(payload) != _PAYLOAD_KEYS:  # pragma: no cover - construction invariant
        raise AssertionError("execution receipt payload schema differs")
    _reject_query_leakage(payload)
    return payload


def _plan_partition(
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> str:
    if isinstance(plan, RuntimeInterventionPlan):
        value: object = plan.bindings.partition
    else:
        bindings = plan.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ExecutionReceiptError("runtime plan bindings differ")
        value = bindings.get("partition")
    value = getattr(value, "value", value)
    return _require_identifier(value, "partition")


def _load_execution_artifacts(
    *,
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    projection_sha256: str,
) -> tuple[RuntimeExecutionResult, RuntimeExecutionArtifactIndex]:
    try:
        return read_runtime_execution_artifact_bundle(
            execution_aggregate_path,
            execution_artifact_directory,
            expected_aggregate_sha256=_require_hash(
                execution_aggregate_sha256, "execution aggregate SHA-256"
            ),
            expected_projection_sha256=_require_hash(
                projection_sha256, "execution projection SHA-256"
            ),
        )
    except (RuntimeExecutionArtifactError, OSError, ValueError) as error:
        raise ExecutionReceiptError("execution artifact replay failed") from error


def make_runtime_execution_receipt(
    *,
    execution_projection_path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Replay actual artifacts, then sign one complete query-blind receipt."""

    raw_key = _signing_public_key(signing_key)
    projection, projection_file_sha256 = _load_projection(
        execution_projection_path, plan
    )
    execution_result, execution_index = _load_execution_artifacts(
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        projection_sha256=str(projection["projection_sha256"]),
    )
    payload = _expected_payload(
        projection=projection,
        projection_file_sha256=projection_file_sha256,
        plan_partition=_plan_partition(plan),
        execution_result=execution_result,
        execution_index=execution_index,
        signing_public_key=raw_key.hex(),
    )
    record = _make_record(payload, signing_key)
    return validate_runtime_execution_receipt(
        record,
        execution_projection_path=execution_projection_path,
        plan=plan,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        verification_key=raw_key,
    )


def validate_runtime_execution_receipt(
    value: Mapping[str, object],
    *,
    execution_projection_path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str | None = None,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Verify signature, projection binding, ordering, and zero-access custody."""

    record = validate_runtime_execution_receipt_envelope(
        value, verification_key=verification_key
    )
    payload = record["payload"]
    signature = record["signature"]
    expected_receipt_hash = record["receipt_sha256"]

    projection, projection_file_sha256 = _load_projection(
        execution_projection_path, plan
    )
    signed_aggregate_sha256 = str(payload["execution_aggregate_sha256"])
    if (
        execution_aggregate_sha256 is not None
        and _require_hash(execution_aggregate_sha256, "execution aggregate SHA-256")
        != signed_aggregate_sha256
    ):
        raise ExecutionReceiptError("execution aggregate binding differs")
    execution_result, execution_index = _load_execution_artifacts(
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=signed_aggregate_sha256,
        projection_sha256=str(projection["projection_sha256"]),
    )
    expected = _expected_payload(
        projection=projection,
        projection_file_sha256=projection_file_sha256,
        plan_partition=_plan_partition(plan),
        execution_result=execution_result,
        execution_index=execution_index,
        signing_public_key=str(payload["signing_public_key"]),
    )
    if payload != expected:
        raise ExecutionReceiptError(
            "execution receipt differs from projection contract"
        )
    return {
        "payload": expected,
        "signature": signature,
        "receipt_sha256": expected_receipt_hash,
    }


def validate_runtime_execution_receipt_envelope(
    value: Mapping[str, object],
    *,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Authenticate one complete query-blind envelope without opening artifacts."""

    if not isinstance(value, Mapping) or set(value) != _RECORD_KEYS:
        raise ExecutionReceiptError("execution receipt record schema differs")
    record = dict(value)
    _reject_query_leakage(record)
    raw_key = _public_key_bytes(verification_key)
    payload = _validate_payload_shape(record["payload"], expected_key_hex=raw_key.hex())
    signature = record["signature"]
    if not _is_lower_hex(signature, 128):
        raise ExecutionReceiptError("malformed Ed25519 signature")
    try:
        Ed25519PublicKey.from_public_bytes(raw_key).verify(
            bytes.fromhex(str(signature)), canonical_json_bytes(payload)
        )
    except InvalidSignature as error:
        raise ExecutionReceiptError("Ed25519 signature verification failed") from error
    expected_receipt_hash = _record_hash(payload, str(signature))
    if record["receipt_sha256"] != expected_receipt_hash:
        raise ExecutionReceiptError("execution receipt hash differs")
    return {
        "payload": payload,
        "signature": signature,
        "receipt_sha256": expected_receipt_hash,
    }


def read_runtime_execution_receipt_envelope_with_sha(
    path: Path,
    *,
    verification_key: bytes | Ed25519PublicKey,
) -> tuple[dict[str, object], str]:
    """Read once and authenticate a receipt before any deferred input is opened."""

    raw = _read_immutable_bytes(path)
    value = _decode_object(raw, label="execution receipt")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ExecutionReceiptError("execution receipt is not canonical")
    verified = validate_runtime_execution_receipt_envelope(
        value, verification_key=verification_key
    )
    return verified, _sha256_bytes(raw)


def write_runtime_execution_receipt(
    path: Path,
    *,
    execution_projection_path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    signing_key: Ed25519PrivateKey,
) -> str:
    """Create one immutable signed receipt and return its file SHA-256."""

    record = make_runtime_execution_receipt(
        execution_projection_path=execution_projection_path,
        plan=plan,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        signing_key=signing_key,
    )
    raw = canonical_json_bytes(record) + b"\n"
    _write_immutable_bytes(path, raw)
    if _read_immutable_bytes(path) != raw:
        raise ExecutionReceiptError("published execution receipt differs")
    return _sha256_bytes(raw)


def read_runtime_execution_receipt_with_sha(
    path: Path,
    *,
    execution_projection_path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str | None = None,
    verification_key: bytes | Ed25519PublicKey,
) -> tuple[dict[str, object], str]:
    """Read once, verify every binding, and return the exact file digest."""

    raw = _read_immutable_bytes(path)
    value = _decode_object(raw, label="execution receipt")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ExecutionReceiptError("execution receipt is not canonical")
    verified = validate_runtime_execution_receipt(
        value,
        execution_projection_path=execution_projection_path,
        plan=plan,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        verification_key=verification_key,
    )
    return verified, _sha256_bytes(raw)


def read_runtime_execution_receipt(
    path: Path,
    *,
    execution_projection_path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str | None = None,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Read a canonical immutable receipt and verify every binding."""

    verified, _ = read_runtime_execution_receipt_with_sha(
        path,
        execution_projection_path=execution_projection_path,
        plan=plan,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        verification_key=verification_key,
    )
    return verified
