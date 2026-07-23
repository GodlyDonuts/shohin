"""Fail-closed five-seed custody for CTAA runtime execution artifacts.

This module deliberately has no query-source inputs.  It joins each of the
five treatment plans in an immutable runtime bundle to one signed pre-query
execution receipt, replays that receipt's artifacts, and requires the
artifact-derived finalized evidence to equal the bundle's committed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ctaa_run_contract import SEED_COUNT
from ctaa_runtime_bundle import (
    read_runtime_plan,
    validate_runtime_bundle,
)
from ctaa_runtime_evidence import read_runtime_evidence
from ctaa_runtime_evidence_finalizer import make_finalized_runtime_evidence
from ctaa_runtime_execution_receipt import (
    read_runtime_execution_receipt_envelope_with_sha,
    validate_runtime_execution_receipt,
)


EXECUTION_SET_SCHEMA = "r12_ctaa_runtime_execution_set_v1"
EXECUTION_SET_MEMBER_SCHEMA = "r12_ctaa_runtime_execution_set_member_v1"

_HEX = frozenset("0123456789abcdef")
_MAX_EXECUTION_SET_BYTES = 4 * 1024 * 1024
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_MEMBER_FILE_BYTES = 256 * 1024 * 1024
_SET_KEYS = frozenset(
    {
        "schema",
        "partition",
        "run_contract_sha256",
        "runtime_bundle_file_sha256",
        "seed_count",
        "entries",
        "execution_set_sha256",
    }
)
_MEMBER_KEYS = frozenset(
    {
        "schema",
        "training_seed",
        "runtime_plan_sha256",
        "runtime_evidence_sha256",
        "runtime_evidence_file_sha256",
        "execution_projection_filename",
        "execution_projection_file_sha256",
        "execution_projection_sha256",
        "execution_aggregate_filename",
        "execution_aggregate_file_sha256",
        "execution_artifact_directory",
        "execution_receipt_filename",
        "execution_receipt_file_sha256",
        "execution_receipt_sha256",
        "execution_sha256",
        "member_sha256",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "training_seed",
        "execution_projection_filename",
        "execution_aggregate_filename",
        "execution_artifact_directory",
        "execution_receipt_filename",
    }
)
_MEMBER_PATH_KEYS = (
    "execution_projection_filename",
    "execution_aggregate_filename",
    "execution_artifact_directory",
    "execution_receipt_filename",
)


class RuntimeExecutionSetError(ValueError):
    """The five-seed execution set or one of its custody links differs."""


@dataclass(frozen=True)
class RuntimeExecutionSetSource:
    """Only the safe relative names needed to derive one member."""

    training_seed: int
    execution_projection_filename: str
    execution_aggregate_filename: str
    execution_artifact_directory: str
    execution_receipt_filename: str


@dataclass(frozen=True)
class _AuthenticatedMember:
    source: Mapping[str, object]
    bundle_entry: Mapping[str, object]
    receipt: Mapping[str, object]
    receipt_file_sha256: str
    receipt_path: Path


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set is not canonical JSON"
        ) from error


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(dict(value)) + "\n").encode("ascii")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_hash(value: object, label: str) -> str:
    if not _is_hash(value):
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} hash differs"
        )
    return str(value)


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} schema differs"
        )
    return dict(value)


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} component differs"
        )
    pure = PurePosixPath(value)
    if (
        pure.name != value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
    ):
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} component is unsafe"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeExecutionSetError(
                f"CTAA runtime execution set duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _decode_object(raw: bytes, label: str) -> dict[str, object]:
    def reject_nonfinite(value: str) -> None:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} has non-finite JSON: {value}"
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} JSON differs"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} root differs"
        )
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise RuntimeExecutionSetError(
                f"CTAA runtime execution set {label} cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeExecutionSetError(
                f"CTAA runtime execution set {label} contains a symlink"
            )


def _identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_immutable_bytes(path: Path, label: str, maximum_bytes: int) -> bytes:
    path = Path(os.path.abspath(path))
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} is unavailable"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
        or metadata.st_size > maximum_bytes
    ):
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} is not a bounded single-link immutable file"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeExecutionSetError("CTAA runtime execution set requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise RuntimeExecutionSetError(
                    f"CTAA runtime execution set {label} exceeds its byte limit"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _identity(metadata) != _identity(before)
        or _identity(before) != _identity(after)
        or after.st_mode & 0o222
        or after.st_nlink != 1
        or not stat.S_ISREG(after.st_mode)
    ):
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} changed while being read"
        )
    return b"".join(chunks)


def _file_sha256(path: Path, label: str, maximum_bytes: int) -> str:
    return hashlib.sha256(_read_immutable_bytes(path, label, maximum_bytes)).hexdigest()


def _direct_child(root: Path, component: object, label: str) -> Path:
    name = _safe_component(component, label)
    absolute_root = Path(os.path.abspath(root))
    target = absolute_root / name
    if target.parent != absolute_root:
        raise RuntimeExecutionSetError(
            f"CTAA runtime execution set {label} escapes its package root"
        )
    return target


def _direct_artifact_directory(root: Path, component: object) -> Path:
    path = _direct_child(root, component, "artifact directory")
    _reject_symlink_components(path, "artifact directory")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set artifact directory is unavailable"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set artifact directory is not a direct non-symlink child"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeExecutionSetError("CTAA runtime execution set requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set artifact directory cannot be opened safely"
        ) from error
    try:
        observed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(metadata) != _identity(observed) or not stat.S_ISDIR(observed.st_mode):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set artifact directory changed while inspected"
        )
    return path


def _load_bundle(
    runtime_bundle_path: Path, run_contract: Mapping[str, object]
) -> tuple[dict[str, object], str, Path]:
    raw = _read_immutable_bytes(
        runtime_bundle_path, "runtime bundle", _MAX_BUNDLE_BYTES
    )
    value = _decode_object(raw, "runtime bundle")
    if raw != _canonical_bytes(value):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime bundle is not canonical JSON"
        )
    try:
        bundle = validate_runtime_bundle(value, run_contract=run_contract)
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime bundle validation failed"
        ) from error
    return (
        bundle,
        hashlib.sha256(raw).hexdigest(),
        Path(os.path.abspath(runtime_bundle_path)).parent,
    )


def _bundle_entries(bundle: Mapping[str, object]) -> list[dict[str, object]]:
    rows = bundle.get("entries")
    if not isinstance(rows, list) or len(rows) != SEED_COUNT:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime bundle seed coverage differs"
        )
    entries: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, Mapping):
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set runtime bundle entry differs"
            )
        entries.append(dict(value))
    seeds = [entry.get("training_seed") for entry in entries]
    if (
        any(type(seed) is not int or int(seed) < 0 for seed in seeds)
        or len(set(seeds)) != SEED_COUNT
        or seeds != sorted(seeds)
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime bundle order/coverage differs"
        )
    return entries


def _read_bundle_plan_and_evidence(
    bundle_root: Path, bundle_entry: Mapping[str, object]
) -> tuple[object, dict[str, object]]:
    plan_path = _direct_child(
        bundle_root, bundle_entry.get("runtime_plan_filename"), "runtime plan"
    )
    evidence_path = _direct_child(
        bundle_root,
        bundle_entry.get("runtime_evidence_filename"),
        "runtime evidence",
    )
    try:
        plan, plan_file_sha = read_runtime_plan(plan_path)
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime plan validation failed"
        ) from error
    if (
        plan_file_sha != bundle_entry.get("runtime_plan_file_sha256")
        or getattr(plan, "plan_sha256", None) != bundle_entry.get("runtime_plan_sha256")
        or getattr(getattr(plan, "bindings", None), "training_seed", None)
        != bundle_entry.get("training_seed")
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime plan binding differs"
        )
    try:
        evidence = read_runtime_evidence(
            evidence_path,
            plan,
            expected_file_sha256=str(bundle_entry.get("runtime_evidence_file_sha256")),
        )
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime evidence validation failed"
        ) from error
    if evidence.get("evidence_sha256") != bundle_entry.get("runtime_evidence_sha256"):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set runtime evidence binding differs"
        )
    return plan, evidence


def _source_mapping(
    value: RuntimeExecutionSetSource | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, RuntimeExecutionSetSource):
        source: dict[str, object] = {
            "training_seed": value.training_seed,
            "execution_projection_filename": value.execution_projection_filename,
            "execution_aggregate_filename": value.execution_aggregate_filename,
            "execution_artifact_directory": value.execution_artifact_directory,
            "execution_receipt_filename": value.execution_receipt_filename,
        }
    else:
        source = _exact_mapping(value, _SOURCE_KEYS, "member source")
    if type(source["training_seed"]) is not int or int(source["training_seed"]) < 0:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set member source seed differs"
        )
    for key in _MEMBER_PATH_KEYS:
        source[key] = _safe_component(source[key], key)
    return source


def _receipt_payload(receipt: Mapping[str, object]) -> Mapping[str, object]:
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set receipt payload differs"
        )
    return payload


def _authenticate_member_receipt(
    *,
    source: Mapping[str, object],
    bundle_entry: Mapping[str, object],
    execution_root: Path,
    run_contract_sha256: str,
    partition: object,
    verification_key: bytes | Ed25519PublicKey,
) -> _AuthenticatedMember:
    seed = source["training_seed"]
    if seed != bundle_entry.get("training_seed"):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set member/bundle seed differs"
        )
    receipt_path = _direct_child(
        execution_root,
        source["execution_receipt_filename"],
        "execution receipt",
    )
    try:
        receipt, receipt_file_sha = read_runtime_execution_receipt_envelope_with_sha(
            receipt_path,
            verification_key=verification_key,
        )
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set signed receipt validation failed"
        ) from error
    payload = _receipt_payload(receipt)
    expected_receipt_bindings = {
        "training_seed": seed,
        "plan_sha256": bundle_entry.get("runtime_plan_sha256"),
        "run_contract_sha256": run_contract_sha256,
        "partition": partition,
    }
    if any(
        payload.get(key) != expected
        for key, expected in expected_receipt_bindings.items()
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set signed receipt binding differs"
        )
    return _AuthenticatedMember(
        source=dict(source),
        bundle_entry=dict(bundle_entry),
        receipt=dict(receipt),
        receipt_file_sha256=receipt_file_sha,
        receipt_path=receipt_path,
    )


def _derive_authenticated_member(
    *,
    authenticated: _AuthenticatedMember,
    bundle_root: Path,
    execution_root: Path,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    source = authenticated.source
    bundle_entry = authenticated.bundle_entry
    seed = source["training_seed"]
    plan, evidence = _read_bundle_plan_and_evidence(bundle_root, bundle_entry)
    projection_path = _direct_child(
        execution_root,
        source["execution_projection_filename"],
        "execution projection",
    )
    aggregate_path = _direct_child(
        execution_root,
        source["execution_aggregate_filename"],
        "execution aggregate",
    )
    artifact_directory = _direct_artifact_directory(
        execution_root, source["execution_artifact_directory"]
    )
    projection_file_sha = _file_sha256(
        projection_path, "execution projection", _MAX_MEMBER_FILE_BYTES
    )
    aggregate_file_sha = _file_sha256(
        aggregate_path, "execution aggregate", _MAX_MEMBER_FILE_BYTES
    )
    if (
        _file_sha256(
            authenticated.receipt_path,
            "execution receipt",
            _MAX_MEMBER_FILE_BYTES,
        )
        != authenticated.receipt_file_sha256
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set receipt file identity differs"
        )
    try:
        receipt = validate_runtime_execution_receipt(
            authenticated.receipt,
            execution_projection_path=projection_path,
            plan=plan,
            execution_aggregate_path=aggregate_path,
            execution_artifact_directory=artifact_directory,
            execution_aggregate_sha256=aggregate_file_sha,
            verification_key=verification_key,
        )
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set signed receipt validation failed"
        ) from error
    payload = _receipt_payload(receipt)
    expected_artifact_bindings = {
        "execution_projection_file_sha256": projection_file_sha,
        "execution_aggregate_sha256": aggregate_file_sha,
    }
    if any(
        payload.get(key) != expected
        for key, expected in expected_artifact_bindings.items()
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set signed receipt binding differs"
        )
    projection_sha = _require_hash(
        payload.get("execution_projection_sha256"), "projection logical"
    )
    execution_sha = _require_hash(payload.get("execution_sha256"), "execution")
    receipt_sha = _require_hash(receipt.get("receipt_sha256"), "receipt logical")
    try:
        finalized = make_finalized_runtime_evidence(
            plan=plan,
            execution_projection_path=projection_path,
            execution_aggregate_path=aggregate_path,
            execution_artifact_directory=artifact_directory,
            execution_aggregate_sha256=aggregate_file_sha,
            execution_receipt_path=authenticated.receipt_path,
            receipt_verification_key=verification_key,
        )
    except Exception as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set evidence finalization failed"
        ) from error
    if finalized != evidence:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set finalized evidence differs from bundle evidence"
        )
    member: dict[str, object] = {
        "schema": EXECUTION_SET_MEMBER_SCHEMA,
        "training_seed": seed,
        "runtime_plan_sha256": bundle_entry["runtime_plan_sha256"],
        "runtime_evidence_sha256": bundle_entry["runtime_evidence_sha256"],
        "runtime_evidence_file_sha256": bundle_entry["runtime_evidence_file_sha256"],
        "execution_projection_filename": source["execution_projection_filename"],
        "execution_projection_file_sha256": projection_file_sha,
        "execution_projection_sha256": projection_sha,
        "execution_aggregate_filename": source["execution_aggregate_filename"],
        "execution_aggregate_file_sha256": aggregate_file_sha,
        "execution_artifact_directory": source["execution_artifact_directory"],
        "execution_receipt_filename": source["execution_receipt_filename"],
        "execution_receipt_file_sha256": authenticated.receipt_file_sha256,
        "execution_receipt_sha256": receipt_sha,
        "execution_sha256": execution_sha,
    }
    member["member_sha256"] = _canonical_hash(member)
    return member


def _validate_declared_shape(value: Mapping[str, object]) -> dict[str, object]:
    execution_set = _exact_mapping(value, _SET_KEYS, "root")
    if execution_set["schema"] != EXECUTION_SET_SCHEMA:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set schema version differs"
        )
    if execution_set["seed_count"] != SEED_COUNT:
        raise RuntimeExecutionSetError("CTAA runtime execution set seed count differs")
    _require_hash(execution_set["run_contract_sha256"], "run contract")
    _require_hash(execution_set["runtime_bundle_file_sha256"], "runtime bundle file")
    rows = execution_set["entries"]
    if not isinstance(rows, list) or len(rows) != SEED_COUNT:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set member count differs"
        )
    entries: list[dict[str, object]] = []
    used_paths: set[str] = set()
    for value in rows:
        row = _exact_mapping(value, _MEMBER_KEYS, "member")
        if row["schema"] != EXECUTION_SET_MEMBER_SCHEMA:
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set member schema version differs"
            )
        if type(row["training_seed"]) is not int or int(row["training_seed"]) < 0:
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set member seed differs"
            )
        for key in (
            "runtime_plan_sha256",
            "runtime_evidence_sha256",
            "runtime_evidence_file_sha256",
            "execution_projection_file_sha256",
            "execution_projection_sha256",
            "execution_aggregate_file_sha256",
            "execution_receipt_file_sha256",
            "execution_receipt_sha256",
            "execution_sha256",
        ):
            _require_hash(row[key], key)
        for key in _MEMBER_PATH_KEYS:
            name = _safe_component(row[key], key)
            if name in used_paths:
                raise RuntimeExecutionSetError(
                    "CTAA runtime execution set member path repeats"
                )
            used_paths.add(name)
        expected_member_sha = _canonical_hash(
            {key: item for key, item in row.items() if key != "member_sha256"}
        )
        if row["member_sha256"] != expected_member_sha:
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set member commitment differs"
            )
        entries.append(row)
    seeds = [row["training_seed"] for row in entries]
    if len(set(seeds)) != SEED_COUNT or seeds != sorted(seeds):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set member order/coverage differs"
        )
    expected_set_sha = _canonical_hash(
        {
            key: item
            for key, item in execution_set.items()
            if key != "execution_set_sha256"
        }
    )
    if execution_set["execution_set_sha256"] != expected_set_sha:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set canonical commitment differs"
        )
    execution_set["entries"] = entries
    return execution_set


def _replay_declared_set(
    execution_set: Mapping[str, object],
    *,
    runtime_bundle_path: Path,
    run_contract: Mapping[str, object],
    verification_key: bytes | Ed25519PublicKey,
    execution_root: Path,
) -> dict[str, object]:
    declared = _validate_declared_shape(execution_set)
    bundle, bundle_file_sha, bundle_root = _load_bundle(
        runtime_bundle_path, run_contract
    )
    run_contract_sha = run_contract.get("run_contract_sha256")
    if (
        declared["partition"] != bundle.get("partition")
        or declared["partition"] != run_contract.get("partition")
        or declared["run_contract_sha256"] != bundle.get("run_contract_sha256")
        or declared["run_contract_sha256"] != run_contract_sha
        or declared["runtime_bundle_file_sha256"] != bundle_file_sha
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set top-level custody binding differs"
        )
    bundle_entries = _bundle_entries(bundle)
    declared_entries = declared["entries"]
    assert isinstance(declared_entries, list)
    if [row["training_seed"] for row in declared_entries] != [
        row["training_seed"] for row in bundle_entries
    ]:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set five-seed bijection differs"
        )
    authenticated_members: list[_AuthenticatedMember] = []
    for row, bundle_entry in zip(declared_entries, bundle_entries, strict=True):
        source = {key: row[key] for key in _SOURCE_KEYS}
        authenticated = _authenticate_member_receipt(
            source=source,
            bundle_entry=bundle_entry,
            execution_root=execution_root,
            run_contract_sha256=str(run_contract_sha),
            partition=declared["partition"],
            verification_key=verification_key,
        )
        payload = _receipt_payload(authenticated.receipt)
        authenticated_bindings = {
            "execution_projection_file_sha256": payload.get(
                "execution_projection_file_sha256"
            ),
            "execution_projection_sha256": payload.get("execution_projection_sha256"),
            "execution_aggregate_file_sha256": payload.get(
                "execution_aggregate_sha256"
            ),
            "execution_receipt_file_sha256": authenticated.receipt_file_sha256,
            "execution_receipt_sha256": authenticated.receipt.get("receipt_sha256"),
            "execution_sha256": payload.get("execution_sha256"),
        }
        if any(row.get(key) != value for key, value in authenticated_bindings.items()):
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set authenticated member binding differs"
            )
        authenticated_members.append(authenticated)

    rebuilt: list[dict[str, object]] = []
    for row, authenticated in zip(declared_entries, authenticated_members, strict=True):
        observed = _derive_authenticated_member(
            authenticated=authenticated,
            bundle_root=bundle_root,
            execution_root=execution_root,
            verification_key=verification_key,
        )
        if observed != row:
            raise RuntimeExecutionSetError(
                "CTAA runtime execution set member replay differs"
            )
        rebuilt.append(observed)
    result = dict(declared)
    result["entries"] = rebuilt
    return result


def make_runtime_execution_set(
    *,
    runtime_bundle_path: Path,
    run_contract: Mapping[str, object],
    artifact_root: Path,
    members: Sequence[RuntimeExecutionSetSource | Mapping[str, object]],
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Derive a five-member set from observed immutable artifacts."""

    bundle, bundle_file_sha, bundle_root = _load_bundle(
        runtime_bundle_path, run_contract
    )
    bundle_entries = _bundle_entries(bundle)
    sources = [_source_mapping(value) for value in members]
    sources.sort(key=lambda value: int(value["training_seed"]))
    if (
        len(sources) != SEED_COUNT
        or len({value["training_seed"] for value in sources}) != SEED_COUNT
        or [value["training_seed"] for value in sources]
        != [value["training_seed"] for value in bundle_entries]
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set source seed coverage differs"
        )
    names = [str(value[key]) for value in sources for key in _MEMBER_PATH_KEYS]
    if len(set(names)) != len(names):
        raise RuntimeExecutionSetError("CTAA runtime execution set source path repeats")
    run_contract_sha = run_contract.get("run_contract_sha256")
    if not _is_hash(run_contract_sha):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set run contract hash differs"
        )
    execution_root = Path(os.path.abspath(artifact_root))
    authenticated_members = [
        _authenticate_member_receipt(
            source=source,
            bundle_entry=bundle_entry,
            execution_root=execution_root,
            run_contract_sha256=str(run_contract_sha),
            partition=bundle.get("partition"),
            verification_key=verification_key,
        )
        for source, bundle_entry in zip(sources, bundle_entries, strict=True)
    ]
    entries = [
        _derive_authenticated_member(
            authenticated=authenticated,
            bundle_root=bundle_root,
            execution_root=execution_root,
            verification_key=verification_key,
        )
        for authenticated in authenticated_members
    ]
    result: dict[str, object] = {
        "schema": EXECUTION_SET_SCHEMA,
        "partition": bundle["partition"],
        "run_contract_sha256": run_contract_sha,
        "runtime_bundle_file_sha256": bundle_file_sha,
        "seed_count": SEED_COUNT,
        "entries": entries,
    }
    result["execution_set_sha256"] = _canonical_hash(result)
    return _validate_declared_shape(result)


def read_runtime_execution_set_with_replay(
    path: Path,
    *,
    runtime_bundle_path: Path,
    run_contract: Mapping[str, object],
    verification_key: bytes | Ed25519PublicKey,
) -> tuple[dict[str, object], str]:
    """Read once, replay all five custody chains, and return the exact file SHA."""

    raw = _read_immutable_bytes(path, "execution set", _MAX_EXECUTION_SET_BYTES)
    value = _decode_object(raw, "execution set")
    if raw != _canonical_bytes(value):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set file is not canonical JSON"
        )
    validated = _replay_declared_set(
        value,
        runtime_bundle_path=runtime_bundle_path,
        run_contract=run_contract,
        verification_key=verification_key,
        execution_root=Path(os.path.abspath(path)).parent,
    )
    return validated, hashlib.sha256(raw).hexdigest()


def read_runtime_execution_set(
    path: Path,
    *,
    runtime_bundle_path: Path,
    run_contract: Mapping[str, object],
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Replay and return a five-seed execution set."""

    return read_runtime_execution_set_with_replay(
        path,
        runtime_bundle_path=runtime_bundle_path,
        run_contract=run_contract,
        verification_key=verification_key,
    )[0]


def _publish_once(path: Path, payload: bytes) -> None:
    target = Path(os.path.abspath(path))
    _reject_symlink_components(target.parent, "output parent")
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing CTAA runtime execution set: {target}")
    try:
        parent_metadata = target.parent.lstat()
    except OSError as error:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set output parent is unavailable"
        ) from error
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(
        parent_metadata.st_mode
    ):
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set output parent differs"
        )
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeExecutionSetError("CTAA runtime execution set requires O_NOFOLLOW")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeExecutionSetError(
                    "CTAA runtime execution set write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            os.chmod(temporary, 0o600, follow_symlinks=False)
            temporary.unlink()


def write_runtime_execution_set(
    path: Path,
    *,
    runtime_bundle_path: Path,
    run_contract: Mapping[str, object],
    members: Sequence[RuntimeExecutionSetSource | Mapping[str, object]],
    verification_key: bytes | Ed25519PublicKey,
) -> str:
    """Derive and publish one immutable execution set; return its file SHA."""

    target = Path(os.path.abspath(path))
    value = make_runtime_execution_set(
        runtime_bundle_path=runtime_bundle_path,
        run_contract=run_contract,
        artifact_root=target.parent,
        members=members,
        verification_key=verification_key,
    )
    raw = _canonical_bytes(value)
    _publish_once(target, raw)
    observed = _read_immutable_bytes(
        target, "published execution set", _MAX_EXECUTION_SET_BYTES
    )
    if observed != raw:
        raise RuntimeExecutionSetError(
            "CTAA runtime execution set published bytes differ"
        )
    return hashlib.sha256(raw).hexdigest()
