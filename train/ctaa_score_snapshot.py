#!/usr/bin/env python3
"""Canonical, immutable score-input snapshots for CTAA independent scorers.

This module is intentionally not integrated with the CTAA authority.  It
defines only the outcome-input inventory, deterministic byte codec, immutable
source acquisition, and Linux sealed-memfd fanout needed by two future scorers.
"""

from __future__ import annotations

from collections.abc import Mapping
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys


INVENTORY_SCHEMA = "r12_ctaa_score_input_inventory_v1"
RUN_INPUT_SCHEMA = "r12_ctaa_score_run_input_v1"
MEMBER_SCHEMA = "r12_ctaa_score_input_member_v1"
SNAPSHOT_SCHEMA = "r12_ctaa_score_snapshot_v1"
SNAPSHOT_MEMBER_SCHEMA = "r12_ctaa_score_snapshot_member_v1"

ARMS = (
    "ctaa_closure",
    "oprc_closure",
    "ctaa_no_closure",
    "ctaa_shuffled_closure",
)
DATASETS = ("base", "intervention")
MEMBER_ROLES = ("evidence", "oracle", "raw_evidence_receipt")
SEED_COUNT = 5
RUN_COUNT = SEED_COUNT * len(ARMS) * len(DATASETS)

_INVENTORY_UNSIGNED_KEYS = frozenset(
    {
        "schema",
        "partition",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_execution_set_sha256",
        "runs",
        "members",
    }
)
_INVENTORY_KEYS = _INVENTORY_UNSIGNED_KEYS | {"inventory_sha256"}
_RUN_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "seed",
        "arm",
        "dataset",
        "receipt_member_id",
        "evidence_member_id",
        "oracle_member_id",
        "parent_evidence_member_id",
    }
)
_MEMBER_KEYS = frozenset(
    {
        "schema",
        "member_id",
        "role",
        "path",
        "sha256",
        "size_bytes",
    }
)
_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "inventory_sha256",
        "partition",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_execution_set_sha256",
        "run_count",
        "runs",
        "members",
        "content_size_bytes",
        "content_sha256",
    }
)
_SNAPSHOT_MEMBER_KEYS = frozenset(
    {
        "schema",
        "member_id",
        "role",
        "sha256",
        "size_bytes",
        "offset",
    }
)
_PARTITIONS = frozenset({"development", "confirmation"})
_HEX = frozenset("0123456789abcdef")
_HEADER_LENGTH = struct.Struct(">Q")
_READ_CHUNK = 1024 * 1024
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_CONTENT_BYTES = 16 * 1024 * 1024 * 1024


class ScoreSnapshotError(ValueError):
    """A score inventory, source member, or snapshot failed closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole canonical JSON encoding used by this component."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ScoreSnapshotError("CTAA score JSON cannot be canonicalized") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ScoreSnapshotError(f"CTAA score {label} SHA-256 differs")
    return str(value)


def _exact_mapping(
    value: object,
    keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ScoreSnapshotError(f"CTAA score {label} schema differs")
    return value


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScoreSnapshotError("CTAA score JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ScoreSnapshotError(f"CTAA score JSON constant {value} is forbidden")


def _decode_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw:
        raise ScoreSnapshotError(f"CTAA score {label} bytes differ")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreSnapshotError(f"CTAA score {label} JSON differs") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ScoreSnapshotError(f"CTAA score {label} is not canonical JSON")
    return value


def _run_id(seed: int, arm: str, dataset: str) -> str:
    return f"seed-{seed}:{arm}:{dataset}"


def _receipt_member_id(run_id: str) -> str:
    return f"receipt:{run_id}"


def _evidence_member_id(run_id: str) -> str:
    return f"evidence:{run_id}"


def _oracle_member_id(dataset: str) -> str:
    return f"oracle:{dataset}"


def _validate_run_lattice(
    runs_value: object,
) -> tuple[list[dict[str, object]], frozenset[str]]:
    if not isinstance(runs_value, list) or len(runs_value) != RUN_COUNT:
        raise ScoreSnapshotError("CTAA score inventory must contain exactly 40 runs")

    runs: list[dict[str, object]] = []
    seen_identities: set[tuple[int, str, str]] = set()
    for value in runs_value:
        row = _exact_mapping(value, _RUN_KEYS, "run input")
        if row["schema"] != RUN_INPUT_SCHEMA:
            raise ScoreSnapshotError("CTAA score run-input identity differs")
        seed = row["seed"]
        arm = row["arm"]
        dataset = row["dataset"]
        if (
            type(seed) is not int
            or int(seed) < 0
            or arm not in ARMS
            or dataset not in DATASETS
        ):
            raise ScoreSnapshotError("CTAA score run-input coordinates differ")
        identity = (int(seed), str(arm), str(dataset))
        if identity in seen_identities:
            raise ScoreSnapshotError("CTAA score run-input coordinates repeat")
        seen_identities.add(identity)

        expected_run_id = _run_id(*identity)
        if (
            row["run_id"] != expected_run_id
            or row["receipt_member_id"] != _receipt_member_id(expected_run_id)
            or row["evidence_member_id"] != _evidence_member_id(expected_run_id)
            or row["oracle_member_id"] != _oracle_member_id(str(dataset))
        ):
            raise ScoreSnapshotError("CTAA score run/member binding differs")
        expected_parent = (
            None
            if dataset == "base"
            else _evidence_member_id(_run_id(int(seed), str(arm), "base"))
        )
        if row["parent_evidence_member_id"] != expected_parent:
            raise ScoreSnapshotError("CTAA score parent-evidence binding differs")
        runs.append(row)

    seeds = sorted({identity[0] for identity in seen_identities})
    if len(seeds) != SEED_COUNT:
        raise ScoreSnapshotError("CTAA score inventory seed count differs")
    expected_identities = {
        (seed, arm, dataset) for seed in seeds for arm in ARMS for dataset in DATASETS
    }
    if seen_identities != expected_identities:
        raise ScoreSnapshotError("CTAA score 40-run lattice is incomplete")
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    expected_order = sorted(
        runs,
        key=lambda row: (
            int(row["seed"]),
            arm_order[str(row["arm"])],
            dataset_order[str(row["dataset"])],
        ),
    )
    if runs != expected_order:
        raise ScoreSnapshotError("CTAA score run order is not canonical")

    expected_member_ids = {_oracle_member_id(dataset) for dataset in DATASETS}
    for row in runs:
        expected_member_ids.add(str(row["receipt_member_id"]))
        expected_member_ids.add(str(row["evidence_member_id"]))
    return runs, frozenset(expected_member_ids)


def _expected_member_role(member_id: str) -> str:
    if member_id.startswith("evidence:"):
        return "evidence"
    if member_id.startswith("oracle:"):
        return "oracle"
    if member_id.startswith("receipt:"):
        return "raw_evidence_receipt"
    raise ScoreSnapshotError("CTAA score member identity differs")


def _validate_source_members(
    members_value: object,
    expected_member_ids: frozenset[str],
) -> list[dict[str, object]]:
    if not isinstance(members_value, list):
        raise ScoreSnapshotError("CTAA score inventory members differ")
    members: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    total_size = 0
    for value in members_value:
        row = _exact_mapping(value, _MEMBER_KEYS, "input member")
        member_id = row["member_id"]
        role = row["role"]
        path = row["path"]
        size_bytes = row["size_bytes"]
        if (
            row["schema"] != MEMBER_SCHEMA
            or not isinstance(member_id, str)
            or not member_id
            or member_id in seen_ids
            or role not in MEMBER_ROLES
            or role != _expected_member_role(member_id)
            or not isinstance(path, str)
            or "\x00" in path
            or not os.path.isabs(path)
            or path != os.path.abspath(path)
            or path in seen_paths
            or type(size_bytes) is not int
            or not 0 <= int(size_bytes) <= _MAX_MEMBER_BYTES
        ):
            raise ScoreSnapshotError("CTAA score input-member identity differs")
        _require_sha256(row["sha256"], "input member")
        seen_ids.add(member_id)
        seen_paths.add(path)
        total_size += int(size_bytes)
        if total_size > _MAX_CONTENT_BYTES:
            raise ScoreSnapshotError("CTAA score snapshot content is too large")
        members.append(row)

    if seen_ids != set(expected_member_ids):
        raise ScoreSnapshotError("CTAA score input-member set differs")
    if [row["member_id"] for row in members] != sorted(seen_ids):
        raise ScoreSnapshotError("CTAA score input-member order is not canonical")
    return members


def _validate_snapshot_members(
    members_value: object,
    expected_member_ids: frozenset[str],
    content_size_bytes: int,
) -> list[dict[str, object]]:
    if not isinstance(members_value, list):
        raise ScoreSnapshotError("CTAA score snapshot members differ")
    members: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    next_offset = 0
    for value in members_value:
        row = _exact_mapping(value, _SNAPSHOT_MEMBER_KEYS, "snapshot member")
        member_id = row["member_id"]
        role = row["role"]
        size_bytes = row["size_bytes"]
        offset = row["offset"]
        if (
            row["schema"] != SNAPSHOT_MEMBER_SCHEMA
            or not isinstance(member_id, str)
            or not member_id
            or member_id in seen_ids
            or role not in MEMBER_ROLES
            or role != _expected_member_role(member_id)
            or type(size_bytes) is not int
            or not 0 <= int(size_bytes) <= _MAX_MEMBER_BYTES
            or type(offset) is not int
            or int(offset) != next_offset
        ):
            raise ScoreSnapshotError("CTAA score snapshot-member identity differs")
        _require_sha256(row["sha256"], "snapshot member")
        seen_ids.add(member_id)
        next_offset += int(size_bytes)
        if next_offset > _MAX_CONTENT_BYTES:
            raise ScoreSnapshotError("CTAA score snapshot content is too large")
        members.append(row)
    if (
        seen_ids != set(expected_member_ids)
        or [row["member_id"] for row in members] != sorted(seen_ids)
        or next_offset != content_size_bytes
    ):
        raise ScoreSnapshotError("CTAA score snapshot-member set differs")
    return members


def _validate_inventory(
    value: object,
    *,
    require_commitment: bool,
) -> dict[str, object]:
    keys = _INVENTORY_KEYS if require_commitment else _INVENTORY_UNSIGNED_KEYS
    inventory = _exact_mapping(value, keys, "input inventory")
    if (
        inventory["schema"] != INVENTORY_SCHEMA
        or inventory["partition"] not in _PARTITIONS
    ):
        raise ScoreSnapshotError("CTAA score input-inventory identity differs")
    for key in (
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_execution_set_sha256",
    ):
        _require_sha256(inventory[key], key)
    _runs, expected_member_ids = _validate_run_lattice(inventory["runs"])
    _validate_source_members(inventory["members"], expected_member_ids)
    if require_commitment:
        expected = {key: inventory[key] for key in sorted(_INVENTORY_UNSIGNED_KEYS)}
        if inventory["inventory_sha256"] != _sha256_bytes(
            canonical_json_bytes(expected)
        ):
            raise ScoreSnapshotError(
                "CTAA score input-inventory canonical commitment differs"
            )
    return inventory


def finalize_score_input_inventory(
    unsigned_inventory: Mapping[str, object],
) -> dict[str, object]:
    """Validate and commit a complete outcome-free inventory without I/O."""

    if not isinstance(unsigned_inventory, Mapping):
        raise ScoreSnapshotError("CTAA score unsigned inventory differs")
    unsigned_raw = canonical_json_bytes(dict(unsigned_inventory))
    unsigned = _decode_canonical_json(unsigned_raw, "unsigned input inventory")
    validated = _validate_inventory(unsigned, require_commitment=False)
    digest = _sha256_bytes(canonical_json_bytes(validated))
    result = {**validated, "inventory_sha256": digest}
    return _validate_inventory(result, require_commitment=True)


def encode_score_input_inventory(inventory: Mapping[str, object]) -> bytes:
    """Validate and encode a committed score-input inventory."""

    if not isinstance(inventory, Mapping):
        raise ScoreSnapshotError("CTAA score input inventory differs")
    raw = canonical_json_bytes(dict(inventory))
    value = _decode_canonical_json(raw, "input inventory")
    _validate_inventory(value, require_commitment=True)
    return raw


def decode_score_input_inventory(raw: bytes) -> dict[str, object]:
    """Decode canonical committed inventory bytes and reject all alternatives."""

    value = _decode_canonical_json(raw, "input inventory")
    return _validate_inventory(value, require_commitment=True)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_parent_directory(path: Path, label: str) -> tuple[Path, int]:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ScoreSnapshotError("CTAA score held-dirfd custody is unavailable")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(os.path.sep, flags)
        for component in absolute.parent.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ScoreSnapshotError(
            f"CTAA score {label} parent cannot be opened safely"
        ) from error
    return absolute, descriptor


def _read_immutable_member_once(member: Mapping[str, object]) -> bytes:
    """Open and read one expected source exactly once under held-dirfd custody."""

    member_id = str(member["member_id"])
    expected_size = int(member["size_bytes"])
    expected_sha256 = str(member["sha256"])
    absolute, parent_descriptor = _open_parent_directory(
        Path(str(member["path"])),
        f"member {member_id}",
    )
    descriptor = -1
    try:
        parent_before = os.fstat(parent_descriptor)
        try:
            path_before = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} is unavailable"
            ) from error
        if (
            not stat.S_ISREG(path_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or path_before.st_nlink != 1
            or path_before.st_mode & 0o222
            or path_before.st_size != expected_size
        ):
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} is not the expected immutable file"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(
                absolute.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} cannot be opened safely"
            ) from error
        before = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(path_before):
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} changed before open"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed = 0
        while observed < expected_size:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK, expected_size - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed += len(chunk)
        if os.read(descriptor, 1):
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} exceeds expected size"
            )
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ScoreSnapshotError(
                f"CTAA score member {member_id} disappeared while being read"
            ) from error
        parent_after = os.fstat(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)

    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or _metadata_identity(after) != _metadata_identity(before)
        or _metadata_identity(path_after) != _metadata_identity(before)
        or observed != expected_size
        or digest.hexdigest() != expected_sha256
    ):
        raise ScoreSnapshotError(f"CTAA score member {member_id} changed or differs")
    return b"".join(chunks)


def build_score_snapshot(inventory_raw: bytes) -> bytes:
    """Acquire every member once after fully validating the whole inventory."""

    inventory = decode_score_input_inventory(inventory_raw)
    members = inventory["members"]
    assert isinstance(members, list)

    content_parts: list[bytes] = []
    snapshot_members: list[dict[str, object]] = []
    offset = 0
    content_digest = hashlib.sha256()
    for member in members:
        assert isinstance(member, dict)
        payload = _read_immutable_member_once(member)
        content_parts.append(payload)
        content_digest.update(payload)
        snapshot_members.append(
            {
                "schema": SNAPSHOT_MEMBER_SCHEMA,
                "member_id": member["member_id"],
                "role": member["role"],
                "sha256": member["sha256"],
                "size_bytes": member["size_bytes"],
                "offset": offset,
            }
        )
        offset += len(payload)

    header = {
        "schema": SNAPSHOT_SCHEMA,
        "inventory_sha256": inventory["inventory_sha256"],
        "partition": inventory["partition"],
        "manifest_sha256": inventory["manifest_sha256"],
        "board_sha256": inventory["board_sha256"],
        "run_contract_sha256": inventory["run_contract_sha256"],
        "runtime_execution_set_sha256": inventory["runtime_execution_set_sha256"],
        "run_count": RUN_COUNT,
        "runs": inventory["runs"],
        "members": snapshot_members,
        "content_size_bytes": offset,
        "content_sha256": content_digest.hexdigest(),
    }
    header_raw = canonical_json_bytes(header)
    if len(header_raw) > _MAX_HEADER_BYTES:
        raise ScoreSnapshotError("CTAA score snapshot header is too large")
    snapshot = (
        _HEADER_LENGTH.pack(len(header_raw)) + header_raw + b"".join(content_parts)
    )
    verify_score_snapshot(
        snapshot,
        expected_inventory_sha256=str(inventory["inventory_sha256"]),
    )
    return snapshot


def verify_score_snapshot(
    snapshot: bytes,
    *,
    expected_inventory_sha256: str,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Verify the complete byte snapshot and return its logical member views."""

    _require_sha256(expected_inventory_sha256, "expected inventory")
    if not isinstance(snapshot, bytes) or len(snapshot) < _HEADER_LENGTH.size:
        raise ScoreSnapshotError("CTAA score snapshot bytes differ")
    (header_length,) = _HEADER_LENGTH.unpack_from(snapshot)
    if not 0 < header_length <= _MAX_HEADER_BYTES:
        raise ScoreSnapshotError("CTAA score snapshot header length differs")
    content_start = _HEADER_LENGTH.size + header_length
    if content_start > len(snapshot):
        raise ScoreSnapshotError("CTAA score snapshot is truncated")
    header_raw = snapshot[_HEADER_LENGTH.size : content_start]
    header = _decode_canonical_json(header_raw, "snapshot header")
    _exact_mapping(header, _SNAPSHOT_KEYS, "snapshot header")
    if (
        header["schema"] != SNAPSHOT_SCHEMA
        or header["inventory_sha256"] != expected_inventory_sha256
        or header["partition"] not in _PARTITIONS
        or header["run_count"] != RUN_COUNT
        or type(header["content_size_bytes"]) is not int
        or not 0 <= int(header["content_size_bytes"]) <= _MAX_CONTENT_BYTES
    ):
        raise ScoreSnapshotError("CTAA score snapshot identity differs")
    for key in (
        "inventory_sha256",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_execution_set_sha256",
        "content_sha256",
    ):
        _require_sha256(header[key], f"snapshot {key}")

    _runs, expected_member_ids = _validate_run_lattice(header["runs"])
    members = _validate_snapshot_members(
        header["members"],
        expected_member_ids,
        int(header["content_size_bytes"]),
    )
    content = snapshot[content_start:]
    if (
        len(content) != int(header["content_size_bytes"])
        or _sha256_bytes(content) != header["content_sha256"]
    ):
        raise ScoreSnapshotError("CTAA score snapshot content differs")

    member_payloads: dict[str, bytes] = {}
    for member in members:
        offset = int(member["offset"])
        end = offset + int(member["size_bytes"])
        payload = content[offset:end]
        if (
            len(payload) != int(member["size_bytes"])
            or _sha256_bytes(payload) != member["sha256"]
        ):
            raise ScoreSnapshotError(
                f"CTAA score snapshot member {member['member_id']} differs"
            )
        member_payloads[str(member["member_id"])] = payload
    return header, member_payloads


def score_snapshot_sha256(snapshot: bytes) -> str:
    if not isinstance(snapshot, bytes):
        raise ScoreSnapshotError("CTAA score snapshot bytes differ")
    return _sha256_bytes(snapshot)


def _linux_seal_constants() -> tuple[int, int, int]:
    names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_GROW",
        "F_SEAL_WRITE",
    )
    if (
        sys.platform != "linux"
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
        or any(not hasattr(fcntl, name) for name in names)
    ):
        raise ScoreSnapshotError("CTAA score Linux memfd sealing is unavailable")
    required = (
        fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    )
    return fcntl.F_ADD_SEALS, fcntl.F_GET_SEALS, required


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise ScoreSnapshotError("CTAA score memfd write made no progress")
        written += count


def _pread_all(descriptor: int, size_bytes: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(descriptor, min(_READ_CHUNK, size_bytes - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def validate_sealed_score_snapshot_fd(
    descriptor: int,
    *,
    expected_snapshot_sha256: str,
    expected_inventory_sha256: str,
) -> bytes:
    """Validate one read-only description of a fully sealed snapshot memfd."""

    _require_sha256(expected_snapshot_sha256, "expected snapshot")
    _require_sha256(expected_inventory_sha256, "expected inventory")
    _add_seals, get_seals, required_seals = _linux_seal_constants()
    if type(descriptor) is not int or descriptor < 0:
        raise ScoreSnapshotError("CTAA score sealed descriptor differs")
    try:
        metadata = os.fstat(descriptor)
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        observed_seals = fcntl.fcntl(descriptor, get_seals)
        inheritable = os.get_inheritable(descriptor)
    except OSError as error:
        raise ScoreSnapshotError(
            "CTAA score sealed descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_size < _HEADER_LENGTH.size
        or access_mode != os.O_RDONLY
        or observed_seals != required_seals
        or not inheritable
    ):
        raise ScoreSnapshotError("CTAA score sealed descriptor custody differs")
    snapshot = _pread_all(descriptor, metadata.st_size)
    after = os.fstat(descriptor)
    if (
        len(snapshot) != metadata.st_size
        or _metadata_identity(after) != _metadata_identity(metadata)
        or _sha256_bytes(snapshot) != expected_snapshot_sha256
    ):
        raise ScoreSnapshotError("CTAA score sealed descriptor bytes differ")
    verify_score_snapshot(
        snapshot,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    return snapshot


def create_sealed_score_snapshot_fds(
    snapshot: bytes,
    *,
    expected_inventory_sha256: str,
) -> tuple[int, int]:
    """Create two independent O_RDONLY descriptions of one sealed Linux memfd."""

    verify_score_snapshot(
        snapshot,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    add_seals, _get_seals, required_seals = _linux_seal_constants()
    writer = -1
    first = -1
    second = -1
    try:
        writer = os.memfd_create(
            "r12-ctaa-score-snapshot",
            os.MFD_ALLOW_SEALING,
        )
        _write_all(writer, snapshot)
        os.fchmod(writer, 0o400)
        fcntl.fcntl(writer, add_seals, required_seals)
        reader_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        first = os.open(f"/proc/self/fd/{writer}", reader_flags)
        second = os.open(f"/proc/self/fd/{writer}", reader_flags)
        os.set_inheritable(first, True)
        os.set_inheritable(second, True)
        os.close(writer)
        writer = -1
        expected_snapshot_sha256 = _sha256_bytes(snapshot)
        validate_sealed_score_snapshot_fd(
            first,
            expected_snapshot_sha256=expected_snapshot_sha256,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        validate_sealed_score_snapshot_fd(
            second,
            expected_snapshot_sha256=expected_snapshot_sha256,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        return first, second
    except (OSError, ScoreSnapshotError) as error:
        for descriptor in (first, second, writer):
            if descriptor >= 0:
                os.close(descriptor)
        if isinstance(error, ScoreSnapshotError):
            raise
        raise ScoreSnapshotError("CTAA score sealed memfd creation failed") from error


__all__ = [
    "ARMS",
    "DATASETS",
    "INVENTORY_SCHEMA",
    "MEMBER_SCHEMA",
    "RUN_COUNT",
    "RUN_INPUT_SCHEMA",
    "SNAPSHOT_MEMBER_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "ScoreSnapshotError",
    "build_score_snapshot",
    "canonical_json_bytes",
    "create_sealed_score_snapshot_fds",
    "decode_score_input_inventory",
    "encode_score_input_inventory",
    "finalize_score_input_inventory",
    "score_snapshot_sha256",
    "validate_sealed_score_snapshot_fd",
    "verify_score_snapshot",
]
