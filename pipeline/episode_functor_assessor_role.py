#!/usr/bin/env python3
"""Standalone assessor for one source-blind EFC candidate machine."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys


ROLE_SCHEMA = "efc-machine-assessor-role-v1"
MACHINE_MAGIC = b"EFCMACH\0"
MACHINE_SIZE = 1_536
MACHINE_HASH_OFFSET = 1_504
FORMAT_VERSION = 1
STATE_COUNT = 5
ACTION_COUNT = 3
OBSERVER_COUNT = 2


class AssessorRoleError(RuntimeError):
    """The assessor input, custody boundary, or candidate machine is invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _validate_filename(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise AssessorRoleError("role filenames must be one relative component")
    return value


def _regular_files() -> tuple[str, ...]:
    return tuple(sorted(path.name for path in Path.cwd().iterdir() if path.is_file()))


def _read_u16(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<H", payload, offset)[0]


def _read_u32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<I", payload, offset)[0]


def _read_u64(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", payload, offset)[0]


def _validate_machine(machine: bytes) -> None:
    if len(machine) != MACHINE_SIZE or machine[:8] != MACHINE_MAGIC:
        raise AssessorRoleError("candidate machine size or magic differs")
    if (
        _read_u32(machine, 8) != FORMAT_VERSION
        or _read_u32(machine, 12) != 64
        or _read_u32(machine, 16) != MACHINE_SIZE
        or _read_u32(machine, 20) != 0
    ):
        raise AssessorRoleError("candidate machine header differs")
    if (
        _read_u16(machine, 24) != STATE_COUNT
        or _read_u16(machine, 26) != ACTION_COUNT
        or _read_u16(machine, 28) != OBSERVER_COUNT
        or _read_u16(machine, 30) != 0
    ):
        raise AssessorRoleError("candidate machine dimensions differ")
    if (
        _read_u64(machine, 32) != (1 << STATE_COUNT) - 1
        or _read_u64(machine, 40) != (1 << ACTION_COUNT) - 1
        or _read_u64(machine, 48) != (1 << OBSERVER_COUNT) - 1
        or any(machine[56:64])
        or any(machine[1472:MACHINE_HASH_OFFSET])
    ):
        raise AssessorRoleError("candidate machine masks or padding differ")
    if (
        machine[MACHINE_HASH_OFFSET:]
        != hashlib.sha256(machine[:MACHINE_HASH_OFFSET]).digest()
    ):
        raise AssessorRoleError("candidate machine self-hash differs")


def _write_immutable(filename: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AssessorRoleError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sandbox_receipt() -> tuple[str, str]:
    for prefix in ("SHOHIN_EFC_SANDBOX", "SHOHIN_LANDLOCK"):
        if (
            os.environ.get(f"{prefix}_ENFORCED") == "1"
            and os.environ.get(f"{prefix}_STAGE")
            and os.environ.get(f"{prefix}_POLICY_SHA256")
        ):
            return (
                os.environ[f"{prefix}_STAGE"],
                os.environ[f"{prefix}_POLICY_SHA256"],
            )
    raise AssessorRoleError("assessor role is not inside the frozen sandbox")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        raise AssessorRoleError("usage: assessor-role CANDIDATE EXPECTED ASSESSMENT")
    candidate_name, expected_name, assessment_name = (
        _validate_filename(value) for value in arguments
    )
    if len({candidate_name, expected_name, assessment_name}) != 3:
        raise AssessorRoleError("role filenames must be distinct")
    sandbox_stage, sandbox_policy_sha256 = _sandbox_receipt()
    if sandbox_stage != "machine-assessor":
        raise AssessorRoleError("assessor sandbox stage differs")
    before = _regular_files()
    if before != tuple(sorted((candidate_name, expected_name))):
        raise AssessorRoleError("assessor invocation contains undeclared files")
    candidate = Path(candidate_name).read_bytes()
    expected = Path(expected_name).read_bytes()
    _validate_machine(candidate)
    _validate_machine(expected)
    exact = candidate == expected
    assessment = {
        "assessor_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "candidate_machine_sha256": hashlib.sha256(candidate).hexdigest(),
        "declared_input_files": sorted((candidate_name, expected_name)),
        "declared_output_files": [assessment_name],
        "exact_machine_match": exact,
        "expected_machine_sha256": hashlib.sha256(expected).hexdigest(),
        "regular_files_before": list(before),
        "sandbox_enforced": True,
        "sandbox_policy_sha256": sandbox_policy_sha256,
        "sandbox_stage": sandbox_stage,
        "schema": ROLE_SCHEMA,
    }
    _write_immutable(assessment_name, _canonical_json_bytes(assessment))
    return 0 if exact else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssessorRoleError, OSError) as exc:
        print(f"efc-assessor-role: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(125) from exc
