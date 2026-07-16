#!/usr/bin/env python3
"""Process-separated exact packet-transport falsifier for R12 PCIF v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import post_commit_interface_falsifier as v1


PROTOCOL_ID = "R12-PCPT-F17x4-v2"
SCHEMA_VERSION = 2
PRIMARY_CHALLENGE_DOMAIN = "R12-PCPT-v2-primary-challenge"
ALTERNATE_CHALLENGE_DOMAIN = "R12-PCPT-v2-alternate-challenge"
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
ROLE_SCRIPT = REPO_ROOT / "pipeline/post_commit_packet_transport_roles.py"
SCIENTIFIC_PATHS = (
    "R12_POST_COMMIT_PACKET_TRANSPORT_V2_PREREG.md",
    "pipeline/post_commit_interface_falsifier.py",
    "pipeline/post_commit_packet_transport_falsifier.py",
    "pipeline/post_commit_packet_transport_roles.py",
    "pipeline/test_post_commit_packet_transport_falsifier.py",
)
EXPECTED_PUBLIC_LAYOUT = tuple(
    (f"public-d{depth}", "public", depth) for depth in v1.DEPTHS
)
EXPECTED_DECISIVE_LAYOUT = tuple(
    (f"{kind}-d{depth}-a0", kind, depth)
    for depth in v1.DEPTHS
    for kind in v1.DECISIVE_KINDS
)
DECOY_CHALLENGE_ID = "hidden_consumer-d2-a0"
EXPECTED_ROLE_COUNTS = {
    "writer": 5,
    "updater": 227,
    "reader": 64,
    "oracle": 40,
    "raw_reader": 1,
}
FROZEN_GATE_NAMES = frozenset(
    {
        "scientific_paths_match_head",
        "phase_one_manifest_precedes_challenges",
        "phase_two_manifest_binds_phase_one",
        "phase_one_files_are_read_only",
        "phase_one_writers_exit_cleanly",
        "phase_one_seed_independent",
        "same_seed_challenges_byte_identical",
        "per_cell_output_permutations_are_nonidentity",
        "frozen_cell_layout_exact",
        "public_state_all_exact",
        "public_motor_all_exact",
        "decisive_state_all_exact",
        "decisive_motor_exactly_one_over_17",
        "state_direct_incremental_all_exact",
        "all_decisive_collisions_replay",
        "horizon_exact_through_8",
        "horizon_rejected_at_9",
        "query_visible_writer_rejected",
        "event_history_updater_rejected",
        "source_visible_updater_rejected",
        "source_visible_reader_rejected",
        "source_pointer_packet_rejected",
        "stale_packet_rejected",
        "stale_packet_skips_exactly_one_nonidentity_event",
        "shuffled_packet_rejected",
        "unrecoded_reader_schema_rejected",
        "role_invocation_counts_exact",
        "successful_updaters_are_file_to_file",
        "successful_readers_receive_one_terminal_file",
        "all_successful_role_calls_have_empty_stderr",
        "every_role_call_binds_scientific_source_tree",
        "role_executable_is_seed_free",
        "full_deterministic_replay_byte_identical",
    }
)
REPORT_FIELDS = frozenset(
    {
        "audit",
        "protocol_id",
        "schema_version",
        "code_sha256",
        "scientific_identity",
        "config",
        "phase_one",
        "phase_two",
        "custody_events",
        "public_results",
        "decisive_results",
        "horizon_decoy_results",
        "executed_decoys",
        "role_invocations",
        "role_invocation_counts",
        "deterministic_replay",
        "gates",
        "pass",
        "claim_boundary",
        "payload_sha256",
    }
)
ROLE_RUN_FIELDS = frozenset(
    {
        "role",
        "command",
        "arguments",
        "input_sha256",
        "output_sha256",
        "stderr_sha256",
        "exit_code",
        "stdin_sha256",
        "stdout_sha256",
        "file_inputs",
        "file_outputs",
        "scientific_source_tree_sha256",
        "cwd_regular_files_before",
        "cwd_regular_files_after",
    }
)


class TransportError(ValueError):
    """Raised when a role or custody gate violates the frozen contract."""


@dataclass(frozen=True)
class RoleRun:
    role: str
    command: tuple[str, ...]
    arguments: tuple[str, ...]
    input_sha256: str
    output_sha256: str
    stderr_sha256: str
    exit_code: int
    output: bytes
    stdin_sha256: str
    stdout_sha256: str
    file_inputs: tuple[dict[str, Any], ...]
    file_outputs: tuple[dict[str, Any], ...]
    scientific_source_tree_sha256: str
    cwd_regular_files_before: tuple[str, ...]
    cwd_regular_files_after: tuple[str, ...]

    def serialized(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "command": list(self.command),
            "arguments": list(self.arguments),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "stderr_sha256": self.stderr_sha256,
            "exit_code": self.exit_code,
            "stdin_sha256": self.stdin_sha256,
            "stdout_sha256": self.stdout_sha256,
            "file_inputs": list(self.file_inputs),
            "file_outputs": list(self.file_outputs),
            "scientific_source_tree_sha256": self.scientific_source_tree_sha256,
            "cwd_regular_files_before": list(self.cwd_regular_files_before),
            "cwd_regular_files_after": list(self.cwd_regular_files_after),
        }


@dataclass(frozen=True)
class PhaseOneCommit:
    manifest: dict[str, Any]
    manifest_path: Path
    state_path: Path
    motor_path: Path
    frozen_monotonic_ns: int
    role_runs: tuple[dict[str, Any], ...]


def canonical_json_bytes(value: Any) -> bytes:
    return v1.canonical_json_bytes(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def current_source_tree_sha256() -> str:
    path_hashes = {
        relative: sha256_bytes((REPO_ROOT / relative).read_bytes())
        for relative in SCIENTIFIC_PATHS
    }
    return sha256_bytes(canonical_json_bytes(path_hashes))


def scientific_identity() -> dict[str, Any]:
    commit_run = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if commit_run.returncode != 0:
        raise TransportError("cannot resolve scientific commit")
    commit = commit_run.stdout.decode("ascii").strip()
    path_hashes: dict[str, str] = {}
    for relative in SCIENTIFIC_PATHS:
        current = (REPO_ROOT / relative).read_bytes()
        committed_run = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if committed_run.returncode != 0:
            raise TransportError(f"scientific path is not committed: {relative}")
        if committed_run.stdout != current:
            raise TransportError(f"scientific path differs from HEAD: {relative}")
        path_hashes[relative] = sha256_bytes(current)
    policy = {
        "PATH": "inherited_for_python_resolution_only",
        "PYTHONPATH": "repository_root",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C",
        "LANG": "C",
    }
    return {
        "scientific_commit": commit,
        "scientific_paths": path_hashes,
        "scientific_source_tree_sha256": sha256_bytes(
            canonical_json_bytes(path_hashes)
        ),
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "role_environment_policy": policy,
        "role_environment_policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "verified_against_head": True,
    }


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise TransportError(
            f"{label} fields must be {sorted(expected)}, got {sorted(value)}"
        )


def _json_lines(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            raise TransportError(f"blank JSONL row at line {line_number}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TransportError(f"JSONL row {line_number} must be an object")
        rows.append(value)
    return rows


def _stream(header: Mapping[str, Any], rows: bytes) -> bytes:
    return canonical_json_bytes(dict(header)) + rows


def _split_stream(payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _json_lines(payload)
    if not rows:
        raise TransportError("role input requires a header row")
    return rows[0], rows[1:]


def _source_from_row(row: Mapping[str, Any]) -> v1.Vector:
    _strict_keys(row, {"source"}, "source row")
    raw = row["source"]
    if not isinstance(raw, list) or len(raw) != v1.DIMENSION:
        raise TransportError("source must contain four field elements")
    if any(not isinstance(value, int) or not 0 <= value < v1.MODULUS for value in raw):
        raise TransportError("source value is outside F_17")
    return tuple(raw)  # type: ignore[return-value]


def _update_from_mapping(value: Mapping[str, Any]) -> v1.AffineUpdate:
    _strict_keys(value, {"matrix", "offset"}, "update")
    matrix = value["matrix"]
    offset = value["offset"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != v1.DIMENSION
        or any(not isinstance(row, list) or len(row) != v1.DIMENSION for row in matrix)
    ):
        raise TransportError("update matrix must be four by four")
    if not isinstance(offset, list) or len(offset) != v1.DIMENSION:
        raise TransportError("update offset must contain four values")
    flat = [item for row in matrix for item in row] + list(offset)
    if any(not isinstance(item, int) or not 0 <= item < v1.MODULUS for item in flat):
        raise TransportError("update value is outside F_17")
    parsed_matrix: v1.Matrix = tuple(tuple(row) for row in matrix)  # type: ignore[assignment]
    parsed_offset: v1.Vector = tuple(offset)  # type: ignore[assignment]
    if not v1.is_invertible(parsed_matrix):
        raise TransportError("update matrix must be invertible")
    return v1.AffineUpdate(parsed_matrix, parsed_offset)


def _permutation(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != v1.MODULUS
        or any(not isinstance(item, int) for item in value)
        or sorted(value) != list(range(v1.MODULUS))
    ):
        raise TransportError("output permutation must be a permutation of F_17")
    if all(index == item for index, item in enumerate(value)):
        raise TransportError("output permutation must be nonidentity")
    return tuple(value)


def _consumer(value: Any) -> v1.Vector:
    if (
        not isinstance(value, list)
        or len(value) != v1.DIMENSION
        or any(not isinstance(item, int) or not 0 <= item < v1.MODULUS for item in value)
    ):
        raise TransportError("consumer must contain four field elements")
    return tuple(value)  # type: ignore[return-value]


def challenge_from_mapping(value: Mapping[str, Any]) -> v1.Challenge:
    _strict_keys(
        value,
        {
            "challenge_id",
            "kind",
            "depth",
            "updates",
            "consumer",
            "output_permutation",
        },
        "challenge",
    )
    if not isinstance(value["challenge_id"], str) or not isinstance(value["kind"], str):
        raise TransportError("challenge identifiers must be strings")
    if not isinstance(value["depth"], int) or value["depth"] < 1:
        raise TransportError("challenge depth must be positive")
    if not isinstance(value["updates"], list):
        raise TransportError("challenge updates must be a list")
    updates = tuple(_update_from_mapping(item) for item in value["updates"])
    if len(updates) != value["depth"]:
        raise TransportError("challenge depth does not match update count")
    return v1.Challenge(
        challenge_id=value["challenge_id"],
        kind=value["kind"],
        depth=value["depth"],
        updates=updates,
        consumer=_consumer(value["consumer"]),
        output_permutation=_permutation(value["output_permutation"]),
    )


def _packet_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[v1.SealedPacket]:
    for row in rows:
        yield v1.validate_serialized_packet(row)


def _file_evidence(directory: Path, relative: str) -> dict[str, Any]:
    path = directory / relative
    evidence: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if path.is_file():
        payload = path.read_bytes()
        evidence.update(
            {
                "bytes": len(payload),
                "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
                "sha256": sha256_bytes(payload),
            }
        )
    return evidence


def _regular_files(directory: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))


def run_role(
    role: str,
    payload: bytes,
    *arguments: str,
    cwd: Path | None = None,
    file_inputs: Sequence[str] = (),
    file_outputs: Sequence[str] = (),
) -> RoleRun:
    role_cwd = (cwd or REPO_ROOT).resolve()
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C",
        "LANG": "C",
    }
    cwd_regular_files_before = (
        _regular_files(role_cwd) if file_inputs or file_outputs else ()
    )
    input_files = tuple(_file_evidence(role_cwd, item) for item in file_inputs)
    command = (sys.executable, str(ROLE_SCRIPT), role, *arguments)
    completed = subprocess.run(
        command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=role_cwd,
        env=environment,
        check=False,
    )
    cwd_regular_files_after = (
        _regular_files(role_cwd) if file_inputs or file_outputs else ()
    )
    output_files = tuple(_file_evidence(role_cwd, item) for item in file_outputs)
    stdin_sha256 = sha256_bytes(payload)
    stdout_sha256 = sha256_bytes(completed.stdout)
    return RoleRun(
        role=role,
        command=command,
        arguments=tuple(arguments),
        input_sha256=sha256_bytes(
            canonical_json_bytes(
                {"stdin_sha256": stdin_sha256, "file_inputs": input_files}
            )
        ),
        output_sha256=sha256_bytes(
            canonical_json_bytes(
                {"stdout_sha256": stdout_sha256, "file_outputs": output_files}
            )
        ),
        stderr_sha256=sha256_bytes(completed.stderr),
        exit_code=completed.returncode,
        output=completed.stdout,
        stdin_sha256=stdin_sha256,
        stdout_sha256=stdout_sha256,
        file_inputs=input_files,
        file_outputs=output_files,
        scientific_source_tree_sha256=current_source_tree_sha256(),
        cwd_regular_files_before=cwd_regular_files_before,
        cwd_regular_files_after=cwd_regular_files_after,
    )


def require_success(run: RoleRun) -> bytes:
    if run.exit_code != 0:
        raise TransportError(
            f"role {run.role} failed with exit {run.exit_code}, stderr {run.stderr_sha256}"
        )
    return run.output


def source_rows_bytes() -> bytes:
    return b"".join(
        canonical_json_bytes({"source": list(source)}) for source in v1.enumerate_states()
    )


def _cell_permutation(challenge_seed: int, challenge_id: str) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}:{challenge_seed}:{challenge_id}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    while True:
        values = list(range(v1.MODULUS))
        rng.shuffle(values)
        if any(index != value for index, value in enumerate(values)):
            return tuple(values)


def generate_challenges(challenge_seed: int) -> dict[str, Any]:
    base = v1.generate_challenges(challenge_seed)
    public = tuple(
        replace(item, output_permutation=_cell_permutation(challenge_seed, item.challenge_id))
        for item in base["public"]
    )
    decisive = tuple(
        replace(item, output_permutation=_cell_permutation(challenge_seed, item.challenge_id))
        for item in base["decisive"]
    )
    serialized = {
        "challenge_seed": challenge_seed,
        "public": [item.serialized() for item in public],
        "decisive": [item.serialized() for item in decisive],
    }
    return {
        "challenge_seed": challenge_seed,
        "public": public,
        "decisive": decisive,
        "payload_sha256": sha256_bytes(canonical_json_bytes(serialized)),
        "serialized": serialized,
    }


def _derive_post_commit_seed(manifest_sha256: str, domain: str) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": domain,
                "phase_one_manifest_sha256": manifest_sha256,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big")


def phase_one_seed_independence(
    repeat_state_payload: bytes,
    repeat_motor_payload: bytes,
    phase_one_manifest: Mapping[str, Any],
    primary_challenge_payload_sha256: str,
    alternate_challenge_payload_sha256: str,
) -> bool:
    """Verify seed independence against committed hashes, not temporary paths."""
    return (
        sha256_bytes(repeat_state_payload)
        == phase_one_manifest["state_packets_sha256"]
        and sha256_bytes(repeat_motor_payload)
        == phase_one_manifest["motor_packets_sha256"]
        and primary_challenge_payload_sha256
        != alternate_challenge_payload_sha256
    )


def generate_challenges_after_commit(
    phase_one: PhaseOneCommit,
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_at = time.monotonic_ns()
    if observed_at <= phase_one.frozen_monotonic_ns:
        raise TransportError("phase-two clock did not follow phase-one freeze")
    if stat.S_IMODE(phase_one.manifest_path.stat().st_mode) != 0o444:
        raise TransportError("phase-one manifest changed mode before phase two")
    manifest_bytes = phase_one.manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if manifest_sha256 != phase_one.manifest["manifest_sha256"]:
        raise TransportError("phase-one manifest changed before phase two")

    challenge_seed = _derive_post_commit_seed(
        manifest_sha256, PRIMARY_CHALLENGE_DOMAIN
    )
    bundle = generate_challenges(challenge_seed)
    challenge_bytes = canonical_json_bytes(bundle["serialized"])
    challenge_path = workdir / "phase2_challenges.json"
    _immutable_write(challenge_path, challenge_bytes)
    challenge_manifest = {
        "protocol_id": PROTOCOL_ID,
        "challenge_payload_sha256": sha256_bytes(challenge_bytes),
        "challenge_seed": challenge_seed,
        "challenge_seed_derivation_domain": PRIMARY_CHALLENGE_DOMAIN,
        "challenge_seed_derived_after_manifest_observation": True,
        "phase_one_manifest_sha256": manifest_sha256,
        "phase_one_manifest_mode_observed": "0444",
        "challenge_file_mode": "0444",
        "generated_only_after_phase_one_manifest_observation": True,
    }
    challenge_manifest_bytes = canonical_json_bytes(challenge_manifest)
    challenge_manifest_path = workdir / "phase2_manifest.json"
    _immutable_write(challenge_manifest_path, challenge_manifest_bytes)
    if stat.S_IMODE(challenge_path.stat().st_mode) != 0o444 or stat.S_IMODE(
        challenge_manifest_path.stat().st_mode
    ) != 0o444:
        raise TransportError("phase-two challenge custody files are not read-only")
    proof = {
        **challenge_manifest,
        "challenge_manifest_sha256": sha256_bytes(challenge_manifest_bytes),
        "ordering_clock": "time.monotonic_ns",
        "ordering_measurement": "strictly_after",
        "raw_clock_values_omitted_for_deterministic_replay": True,
    }
    return bundle, proof


def _symbol_records(payload: bytes) -> list[tuple[bytes, int]]:
    result: list[tuple[bytes, int]] = []
    raw_rows = payload.splitlines(keepends=True)
    if b"".join(raw_rows) != payload or any(not row.endswith(b"\n") for row in raw_rows):
        raise TransportError("symbol stream must be newline-terminated JSONL")
    for raw, row in zip(raw_rows, _json_lines(payload), strict=True):
        _strict_keys(row, {"symbol"}, "reader symbol")
        value = row["symbol"]
        if not isinstance(value, int) or not 0 <= value < v1.MODULUS:
            raise TransportError("reader symbol is outside F_17")
        if raw != canonical_json_bytes(row):
            raise TransportError("symbol stream is not canonical JSONL")
        result.append((raw, value))
    return result


def _symbol_rows(payload: bytes) -> list[int]:
    return [value for _, value in _symbol_records(payload)]


def _packet_values(payload: bytes) -> list[v1.Vector]:
    return [packet.values for packet in _packet_rows(_json_lines(payload))]


def _score_symbols(candidate: bytes, truth: bytes) -> tuple[int, int]:
    candidate_rows = _symbol_records(candidate)
    truth_rows = _symbol_records(truth)
    if len(candidate_rows) != len(truth_rows):
        raise TransportError("candidate/oracle row counts differ")
    return sum(a[0] == b[0] for a, b in zip(candidate_rows, truth_rows, strict=True)), len(truth_rows)


def _immutable_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)


def _immutable_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb", closefd=False
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    finally:
        os.close(descriptor)
    destination.chmod(0o444)


def _phase_one(
    source_payload: bytes,
    workdir: Path,
    identity: Mapping[str, Any],
) -> PhaseOneCommit:
    header = {"protocol_id": PROTOCOL_ID, "role": "writer"}
    writer_input = _stream(header, source_payload)
    state_run = run_role("writer", writer_input, "--arm", "state")
    motor_run = run_role("writer", writer_input, "--arm", "motor")
    state_bytes = require_success(state_run)
    motor_bytes = require_success(motor_run)
    if len(_json_lines(state_bytes)) != v1.MODULUS**v1.DIMENSION:
        raise TransportError("state writer row count mismatch")
    if len(_json_lines(motor_bytes)) != v1.MODULUS**v1.DIMENSION:
        raise TransportError("motor writer row count mismatch")
    state_path = workdir / "phase1_state_packets.jsonl"
    motor_path = workdir / "phase1_motor_packets.jsonl"
    _immutable_write(state_path, state_bytes)
    _immutable_write(motor_path, motor_bytes)
    state_mode = stat.S_IMODE(state_path.stat().st_mode)
    motor_mode = stat.S_IMODE(motor_path.stat().st_mode)
    if state_mode != 0o444 or motor_mode != 0o444:
        raise TransportError("phase-one packet streams are not read-only")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_count": v1.MODULUS**v1.DIMENSION,
        "packet_width_field_elements": v1.PACKET_WIDTH,
        "source_payload_sha256": sha256_bytes(source_payload),
        "state_packets_sha256": sha256_bytes(state_bytes),
        "motor_packets_sha256": sha256_bytes(motor_bytes),
        "state_packets_mode": "0444",
        "motor_packets_mode": "0444",
        "code_sha256": sha256_bytes(SCRIPT.read_bytes()),
        "scientific_commit": identity["scientific_commit"],
        "scientific_source_tree_sha256": identity[
            "scientific_source_tree_sha256"
        ],
        "manifest_mode": "0444",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = workdir / "phase1_manifest.json"
    _immutable_write(manifest_path, manifest_bytes)
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o444:
        raise TransportError("phase-one manifest is not read-only")
    manifest["manifest_sha256"] = sha256_bytes(manifest_bytes)
    return PhaseOneCommit(
        manifest=manifest,
        manifest_path=manifest_path,
        state_path=state_path,
        motor_path=motor_path,
        frozen_monotonic_ns=time.monotonic_ns(),
        role_runs=(state_run.serialized(), motor_run.serialized()),
    )


def _reader_input(challenge: v1.Challenge, role: str = "reader") -> bytes:
    return _stream(
        {
            "protocol_id": PROTOCOL_ID,
            "role": role,
            "consumer": list(challenge.consumer),
            "output_permutation": list(challenge.output_permutation),
        },
        b"",
    )


def _oracle_input(challenge: v1.Challenge, source_payload: bytes, emit: str) -> bytes:
    return _stream(
        {
            "protocol_id": PROTOCOL_ID,
            "role": "oracle",
            "challenge": challenge.serialized(),
            "emit": emit,
        },
        source_payload,
    )


def _transport(
    initial_packets: Path,
    challenge: v1.Challenge,
    invocations: list[dict[str, Any]],
    directory: Path,
    prefix: str,
    updates: Sequence[v1.AffineUpdate] | None = None,
) -> tuple[Path, list[str]]:
    directory.mkdir(parents=True, exist_ok=False)
    current_source = initial_packets
    hashes = [sha256_bytes(current_source.read_bytes())]
    applied_updates = tuple(challenge.updates if updates is None else updates)
    for index, update in enumerate(applied_updates, start=1):
        invocation_dir = directory / f"{prefix}_event_{index:03d}"
        invocation_dir.mkdir()
        current = invocation_dir / "packet_in.jsonl"
        next_path = invocation_dir / "packet_out.jsonl"
        _immutable_copy(current_source, current)
        payload = _stream(
            {
                "protocol_id": PROTOCOL_ID,
                "role": "updater",
                "update": update.serialized(),
            },
            b"",
        )
        run = run_role(
            "updater",
            payload,
            "--packet-in",
            "packet_in.jsonl",
            "--packet-out",
            "packet_out.jsonl",
            cwd=invocation_dir,
            file_inputs=("packet_in.jsonl",),
            file_outputs=("packet_out.jsonl",),
        )
        invocations.append(run.serialized())
        require_success(run)
        if not next_path.is_file() or stat.S_IMODE(next_path.stat().st_mode) != 0o444:
            raise TransportError("updater did not emit one immutable packet file")
        current_source = next_path
        hashes.append(sha256_bytes(current_source.read_bytes()))
    return current_source, hashes


def _run_reader(
    challenge: v1.Challenge,
    packet_path: Path,
    invocations: list[dict[str, Any]],
    *,
    directory: Path,
    role: str = "reader",
) -> RoleRun:
    directory.mkdir(parents=True, exist_ok=False)
    terminal = directory / "terminal_packet.jsonl"
    _immutable_copy(packet_path, terminal)
    run = run_role(
        role,
        _reader_input(challenge, role),
        "--packet-in",
        "terminal_packet.jsonl",
        cwd=directory,
        file_inputs=("terminal_packet.jsonl",),
    )
    invocations.append(run.serialized())
    return run


def _source_index(source: Sequence[int]) -> int:
    result = 0
    for value in source:
        result = result * v1.MODULUS + int(value)
    return result


def score_challenge(
    challenge: v1.Challenge,
    source_payload: bytes,
    state_initial: Path,
    motor_initial: Path,
    invocations: list[dict[str, Any]],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle_symbols_run = run_role("oracle", _oracle_input(challenge, source_payload, "symbols"))
    oracle_packets_run = run_role("oracle", _oracle_input(challenge, source_payload, "packets"))
    invocations.extend([oracle_symbols_run.serialized(), oracle_packets_run.serialized()])
    oracle_symbols = require_success(oracle_symbols_run)
    direct_packets = require_success(oracle_packets_run)

    cell_root = workdir / "cells" / challenge.challenge_id
    cell_root.mkdir(parents=True, exist_ok=False)
    terminals: dict[str, Path] = {}
    transport_hashes: dict[str, list[str]] = {}
    reader_outputs: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    total = 0
    for arm, initial in (("state", state_initial), ("motor", motor_initial)):
        terminal, hashes = _transport(
            initial,
            challenge,
            invocations,
            cell_root / f"{arm}_transport",
            arm,
        )
        terminals[arm] = terminal
        transport_hashes[arm] = hashes
        reader_run = _run_reader(
            challenge,
            terminal,
            invocations,
            directory=cell_root / f"{arm}_reader",
        )
        reader_output = require_success(reader_run)
        reader_outputs[arm] = reader_output
        correct, arm_total = _score_symbols(reader_output, oracle_symbols)
        counts[arm] = correct
        total = arm_total

    state_terminal_bytes = terminals["state"].read_bytes()
    motor_terminal_bytes = terminals["motor"].read_bytes()
    state_direct_match = state_terminal_bytes == direct_packets
    result: dict[str, Any] = {
        "challenge_id": challenge.challenge_id,
        "kind": challenge.kind,
        "depth": challenge.depth,
        "output_permutation": list(challenge.output_permutation),
        "total_sources": total,
        "state_correct": counts["state"],
        "motor_correct": counts["motor"],
        "state_accuracy": f"{counts['state']}/{total}",
        "motor_accuracy": f"{counts['motor']}/{total}",
        "state_direct_incremental_packet_match": state_direct_match,
        "state_terminal_sha256": sha256_bytes(state_terminal_bytes),
        "motor_terminal_sha256": sha256_bytes(motor_terminal_bytes),
        "oracle_symbols_sha256": sha256_bytes(oracle_symbols),
        "transport_packet_sha256": transport_hashes,
    }
    coefficient, _ = v1.effective_functional(challenge)
    decisive = any(coefficient[v1.PUBLIC_DIMENSION :])
    result["decisive_outside_public_span"] = decisive
    if decisive:
        witness = v1._collision_witness(challenge)
        left_index = _source_index(witness["left_source"])
        right_index = _source_index(witness["right_source"])
        motor_packets = _packet_values(motor_terminal_bytes)
        motor_symbols = _symbol_rows(reader_outputs["motor"])
        oracle_values = _symbol_rows(oracle_symbols)
        witness.update(
            {
                "terminal_motor_packets_equal": motor_packets[left_index]
                == motor_packets[right_index],
                "terminal_motor_symbols_equal": motor_symbols[left_index]
                == motor_symbols[right_index],
                "oracle_symbols_distinct": oracle_values[left_index]
                != oracle_values[right_index],
            }
        )
        result["collision_witness"] = witness

    if challenge.depth <= 8:
        horizon_packet = terminals["state"]
        horizon_applied_events = challenge.depth
    else:
        horizon_packet, _ = _transport(
            state_initial,
            challenge,
            invocations,
            cell_root / "horizon_transport",
            "horizon",
            updates=challenge.updates[:8],
        )
        horizon_applied_events = 8
    horizon_run = _run_reader(
        challenge,
        horizon_packet,
        invocations,
        directory=cell_root / "horizon_reader",
    )
    horizon_correct, horizon_total = _score_symbols(
        require_success(horizon_run), oracle_symbols
    )
    horizon = {
        "challenge_id": challenge.challenge_id,
        "depth": challenge.depth,
        "reader_role": "reader",
        "reader_interface_matches_canonical": True,
        "applied_events": horizon_applied_events,
        "correct": horizon_correct,
        "total_sources": horizon_total,
    }
    return result, {
        "horizon": horizon,
        "oracle_symbols": oracle_symbols,
        "state_terminal": terminals["state"],
    }


def executed_decoys(
    source_payload: bytes,
    state_initial: Path,
    challenge: v1.Challenge,
    oracle_symbols: bytes,
    state_terminal: Path,
    invocations: list[dict[str, Any]],
    workdir: Path,
) -> dict[str, Any]:
    decoy_root = workdir / "decoys"
    decoy_root.mkdir(parents=True, exist_ok=False)
    one_source = canonical_json_bytes({"source": [0, 0, 0, 0]})
    bad_writer = run_role(
        "writer",
        _stream(
            {
                "protocol_id": PROTOCOL_ID,
                "role": "writer",
                "consumer": [1, 0, 0, 0],
            },
            one_source,
        ),
        "--arm",
        "state",
    )
    bad_update_dir = decoy_root / "bad_history_updater"
    bad_update_dir.mkdir()
    _immutable_copy(state_initial, bad_update_dir / "packet_in.jsonl")
    bad_update = run_role(
        "updater",
        _stream(
            {
                "protocol_id": PROTOCOL_ID,
                "role": "updater",
                "update": challenge.updates[0].serialized(),
                "history": [item.serialized() for item in challenge.updates],
            },
            b"",
        ),
        "--packet-in",
        "packet_in.jsonl",
        "--packet-out",
        "packet_out.jsonl",
        cwd=bad_update_dir,
        file_inputs=("packet_in.jsonl",),
        file_outputs=("packet_out.jsonl",),
    )
    bad_reader_dir = decoy_root / "bad_source_reader"
    bad_reader_dir.mkdir()
    _immutable_copy(state_terminal, bad_reader_dir / "terminal_packet.jsonl")
    bad_reader_source = run_role(
        "reader",
        _stream(
            {
                "protocol_id": PROTOCOL_ID,
                "role": "reader",
                "consumer": list(challenge.consumer),
                "output_permutation": list(challenge.output_permutation),
                "source": [0, 0, 0, 0],
            },
            b"",
        ),
        "--packet-in",
        "terminal_packet.jsonl",
        cwd=bad_reader_dir,
        file_inputs=("terminal_packet.jsonl",),
    )
    bad_updater_dir = decoy_root / "bad_source_updater"
    bad_updater_dir.mkdir()
    _immutable_copy(state_initial, bad_updater_dir / "packet_in.jsonl")
    bad_updater_source = run_role(
        "updater",
        _stream(
            {
                "protocol_id": PROTOCOL_ID,
                "role": "updater",
                "update": challenge.updates[0].serialized(),
                "source": [0, 0, 0, 0],
            },
            b"",
        ),
        "--packet-in",
        "packet_in.jsonl",
        "--packet-out",
        "packet_out.jsonl",
        cwd=bad_updater_dir,
        file_inputs=("packet_in.jsonl",),
        file_outputs=("packet_out.jsonl",),
    )
    pointer_packet_bytes = canonical_json_bytes(
        {"values": [0, 0, 0, 0], "source_id": "forbidden"}
    )
    pointer_packet = decoy_root / "pointer_packet.jsonl"
    _immutable_write(pointer_packet, pointer_packet_bytes)
    bad_pointer = _run_reader(
        challenge,
        pointer_packet,
        invocations,
        directory=decoy_root / "pointer_reader",
    )
    invocations.extend(
        run.serialized()
        for run in (
            bad_writer,
            bad_update,
            bad_updater_source,
            bad_reader_source,
        )
    )

    identity = v1.identity_matrix()
    zero = (0,) * v1.DIMENSION
    skipped_index = next(
        (
            index
            for index, update in enumerate(challenge.updates)
            if update.matrix != identity or update.offset != zero
        ),
        None,
    )
    if skipped_index is None:
        raise TransportError("stale decoy requires one nonidentity event")
    stale_updates = tuple(
        update
        for index, update in enumerate(challenge.updates)
        if index != skipped_index
    )
    stale_packet, stale_hashes = _transport(
        state_initial,
        challenge,
        invocations,
        decoy_root / "stale_transport",
        "stale",
        updates=stale_updates,
    )
    stale_run = _run_reader(
        challenge,
        stale_packet,
        invocations,
        directory=decoy_root / "stale_reader",
    )
    stale_correct, stale_total = _score_symbols(require_success(stale_run), oracle_symbols)

    state_rows = _json_lines(state_terminal.read_bytes())
    shuffled = b"".join(canonical_json_bytes(row) for row in reversed(state_rows))
    shuffled_packet = decoy_root / "shuffled_packet.jsonl"
    _immutable_write(shuffled_packet, shuffled)
    shuffled_run = _run_reader(
        challenge,
        shuffled_packet,
        invocations,
        directory=decoy_root / "shuffled_reader",
    )
    shuffled_correct, shuffled_total = _score_symbols(
        require_success(shuffled_run), oracle_symbols
    )

    raw_run = _run_reader(
        challenge,
        state_terminal,
        invocations,
        directory=decoy_root / "raw_reader",
        role="raw_reader",
    )
    raw_schema_rejected = False
    if raw_run.exit_code == 0:
        try:
            _score_symbols(raw_run.output, oracle_symbols)
        except TransportError:
            raw_schema_rejected = True

    return {
        "query_visible_writer_rejected": bad_writer.exit_code != 0,
        "event_history_updater_rejected": bad_update.exit_code != 0,
        "source_visible_updater_rejected": bad_updater_source.exit_code != 0,
        "source_visible_reader_rejected": bad_reader_source.exit_code != 0,
        "source_pointer_packet_rejected": bad_pointer.exit_code != 0,
        "stale_packet": {
            "skipped_event_index": skipped_index,
            "skipped_event_nonidentity": True,
            "applied_event_count": len(stale_updates),
            "transport_packet_sha256": stale_hashes,
            "correct": stale_correct,
            "total_sources": stale_total,
            "rejected": stale_correct < stale_total,
        },
        "shuffled_packet": {
            "correct": shuffled_correct,
            "total_sources": shuffled_total,
            "rejected": shuffled_correct < shuffled_total,
        },
        "unrecoded_reader_schema_rejected": raw_schema_rejected,
        "source_payload_sha256": sha256_bytes(source_payload),
    }


def code_sha256() -> str:
    return sha256_bytes(SCRIPT.read_bytes())


def pending_replay_record() -> dict[str, Any]:
    return {
        "status": "pending_second_full_run",
        "first_core_payload_sha256": None,
        "second_core_payload_sha256": None,
        "second_core_report": None,
    }


def build_report() -> dict[str, Any]:
    identity = scientific_identity()
    source_payload = source_rows_bytes()
    invocations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pcpt-v2-") as directory:
        workdir = Path(directory)
        phase_one_commit = _phase_one(
            source_payload, workdir, identity
        )
        phase_one = phase_one_commit.manifest
        state_initial = phase_one_commit.state_path
        motor_initial = phase_one_commit.motor_path
        phase_runs = list(phase_one_commit.role_runs)
        invocations.extend(phase_runs)
        custody_events = [
            {
                "ordinal": 1,
                "event": "writers_exited",
                "evidence_sha256": sha256_bytes(canonical_json_bytes(phase_runs)),
            },
            {
                "ordinal": 2,
                "event": "phase_one_manifest_fsynced_and_frozen",
                "evidence_sha256": phase_one["manifest_sha256"],
            },
        ]

        challenge_bundle, phase_order_proof = generate_challenges_after_commit(
            phase_one_commit, workdir
        )
        challenge_seed = challenge_bundle["challenge_seed"]
        custody_events.append(
            {
                "ordinal": 3,
                "event": "phase_two_challenges_generated_and_bound",
                "evidence_sha256": phase_order_proof[
                    "challenge_manifest_sha256"
                ],
            }
        )
        repeated_bundle = generate_challenges(challenge_seed)
        alternate_challenge_seed = _derive_post_commit_seed(
            phase_one["manifest_sha256"], ALTERNATE_CHALLENGE_DOMAIN
        )
        alternate_bundle = generate_challenges(alternate_challenge_seed)

        repeat_state = run_role(
            "writer",
            _stream({"protocol_id": PROTOCOL_ID, "role": "writer"}, source_payload),
            "--arm",
            "state",
        )
        repeat_motor = run_role(
            "writer",
            _stream({"protocol_id": PROTOCOL_ID, "role": "writer"}, source_payload),
            "--arm",
            "motor",
        )
        invocations.extend([repeat_state.serialized(), repeat_motor.serialized()])
        repeat_state_payload = require_success(repeat_state)
        repeat_motor_payload = require_success(repeat_motor)

        public_results: list[dict[str, Any]] = []
        decisive_results: list[dict[str, Any]] = []
        horizons: list[dict[str, Any]] = []
        selected_decoy_inputs: dict[str, Any] | None = None
        all_challenges = list(challenge_bundle["public"]) + list(
            challenge_bundle["decisive"]
        )
        for challenge in all_challenges:
            result, sidecar = score_challenge(
                challenge,
                source_payload,
                state_initial,
                motor_initial,
                invocations,
                workdir,
            )
            horizons.append(sidecar["horizon"])
            if challenge.kind == "public":
                public_results.append(result)
            else:
                decisive_results.append(result)
                if challenge.challenge_id == DECOY_CHALLENGE_ID:
                    selected_decoy_inputs = {
                        "challenge": challenge,
                        "oracle_symbols": sidecar["oracle_symbols"],
                        "state_terminal": sidecar["state_terminal"],
                    }

        if selected_decoy_inputs is None:
            raise AssertionError("a decisive decoy challenge is required")
        decoys = executed_decoys(
            source_payload,
            state_initial,
            selected_decoy_inputs["challenge"],
            selected_decoy_inputs["oracle_symbols"],
            selected_decoy_inputs["state_terminal"],
            invocations,
            workdir,
        )

    expected_total = v1.MODULUS**v1.DIMENSION
    expected_motor = v1.MODULUS ** (v1.DIMENSION - 1)
    challenges = public_results + decisive_results
    permutations = [tuple(row["output_permutation"]) for row in challenges]
    role_counts: dict[str, int] = {}
    for invocation in invocations:
        role = invocation["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    observed_public_layout = tuple(
        (row["challenge_id"], row["kind"], row["depth"])
        for row in public_results
    )
    observed_decisive_layout = tuple(
        (row["challenge_id"], row["kind"], row["depth"])
        for row in decisive_results
    )
    successful_updaters = [
        run
        for run in invocations
        if run["role"] == "updater" and run["exit_code"] == 0
    ]
    successful_readers = [
        run
        for run in invocations
        if run["role"] in {"reader", "raw_reader"} and run["exit_code"] == 0
    ]
    gates = {
        "scientific_paths_match_head": identity["verified_against_head"],
        "phase_one_manifest_precedes_challenges": (
            [row["ordinal"] for row in custody_events] == [1, 2, 3]
            and phase_order_proof["ordering_measurement"] == "strictly_after"
            and phase_order_proof[
                "generated_only_after_phase_one_manifest_observation"
            ]
            and phase_order_proof["phase_one_manifest_sha256"]
            == phase_one["manifest_sha256"]
            and phase_order_proof["challenge_seed_derived_after_manifest_observation"]
            and challenge_seed
            == _derive_post_commit_seed(
                phase_one["manifest_sha256"], PRIMARY_CHALLENGE_DOMAIN
            )
        ),
        "phase_two_manifest_binds_phase_one": (
            phase_order_proof["challenge_payload_sha256"]
            == challenge_bundle["payload_sha256"]
            and phase_order_proof["phase_one_manifest_mode_observed"] == "0444"
            and phase_order_proof["challenge_file_mode"] == "0444"
        ),
        "phase_one_files_are_read_only": all(
            phase_one[key] == "0444"
            for key in (
                "state_packets_mode",
                "motor_packets_mode",
                "manifest_mode",
            )
        ),
        "phase_one_writers_exit_cleanly": all(run["exit_code"] == 0 for run in phase_runs),
        "phase_one_seed_independent": phase_one_seed_independence(
            repeat_state_payload,
            repeat_motor_payload,
            phase_one,
            challenge_bundle["payload_sha256"],
            alternate_bundle["payload_sha256"],
        ),
        "same_seed_challenges_byte_identical": (
            canonical_json_bytes(challenge_bundle["serialized"])
            == canonical_json_bytes(repeated_bundle["serialized"])
        ),
        "per_cell_output_permutations_are_nonidentity": all(
            sorted(permutation) == list(range(v1.MODULUS))
            and any(index != value for index, value in enumerate(permutation))
            for permutation in permutations
        ),
        "frozen_cell_layout_exact": (
            observed_public_layout == EXPECTED_PUBLIC_LAYOUT
            and observed_decisive_layout == EXPECTED_DECISIVE_LAYOUT
            and selected_decoy_inputs["challenge"].challenge_id
            == DECOY_CHALLENGE_ID
        ),
        "public_state_all_exact": all(row["state_correct"] == expected_total for row in public_results),
        "public_motor_all_exact": all(row["motor_correct"] == expected_total for row in public_results),
        "decisive_state_all_exact": all(row["state_correct"] == expected_total for row in decisive_results),
        "decisive_motor_exactly_one_over_17": all(row["motor_correct"] == expected_motor for row in decisive_results),
        "state_direct_incremental_all_exact": all(row["state_direct_incremental_packet_match"] for row in public_results + decisive_results),
        "all_decisive_collisions_replay": all(
            row["collision_witness"]["terminal_motor_packets_equal"]
            and row["collision_witness"]["terminal_motor_symbols_equal"]
            and row["collision_witness"]["oracle_symbols_distinct"]
            for row in decisive_results
        ),
        "horizon_exact_through_8": all(
            row["correct"] == expected_total for row in horizons if row["depth"] <= 8
        ),
        "horizon_rejected_at_9": all(
            row["correct"] < expected_total
            and row["reader_role"] == "reader"
            and row["reader_interface_matches_canonical"]
            and row["applied_events"] == 8
            for row in horizons if row["depth"] == 9
        ),
        "query_visible_writer_rejected": decoys["query_visible_writer_rejected"],
        "event_history_updater_rejected": decoys["event_history_updater_rejected"],
        "source_visible_updater_rejected": decoys["source_visible_updater_rejected"],
        "source_visible_reader_rejected": decoys["source_visible_reader_rejected"],
        "source_pointer_packet_rejected": decoys["source_pointer_packet_rejected"],
        "stale_packet_rejected": decoys["stale_packet"]["rejected"],
        "stale_packet_skips_exactly_one_nonidentity_event": (
            decoys["stale_packet"]["skipped_event_nonidentity"]
            and decoys["stale_packet"]["applied_event_count"]
            == selected_decoy_inputs["challenge"].depth - 1
        ),
        "shuffled_packet_rejected": decoys["shuffled_packet"]["rejected"],
        "unrecoded_reader_schema_rejected": decoys["unrecoded_reader_schema_rejected"],
        "role_invocation_counts_exact": role_counts == EXPECTED_ROLE_COUNTS,
        "successful_updaters_are_file_to_file": all(
            run["arguments"][:1] == ["--packet-in"]
            and len(run["file_inputs"]) == 1
            and len(run["file_outputs"]) == 1
            and run["file_inputs"][0].get("mode") == "0444"
            and run["file_outputs"][0].get("mode") == "0444"
            and run["cwd_regular_files_before"] == ["packet_in.jsonl"]
            and run["cwd_regular_files_after"]
            == ["packet_in.jsonl", "packet_out.jsonl"]
            for run in successful_updaters
        ),
        "successful_readers_receive_one_terminal_file": all(
            run["arguments"][:1] == ["--packet-in"]
            and len(run["file_inputs"]) == 1
            and not run["file_outputs"]
            and run["file_inputs"][0].get("mode") == "0444"
            and run["cwd_regular_files_before"] == ["terminal_packet.jsonl"]
            and run["cwd_regular_files_after"] == ["terminal_packet.jsonl"]
            for run in successful_readers
        ),
        "all_successful_role_calls_have_empty_stderr": all(
            run["exit_code"] != 0 or run["stderr_sha256"] == sha256_bytes(b"")
            for run in invocations
        ),
        "every_role_call_binds_scientific_source_tree": all(
            run["scientific_source_tree_sha256"]
            == identity["scientific_source_tree_sha256"]
            for run in invocations
        ),
        "role_executable_is_seed_free": (
            b"CHALLENGE_SEED" not in ROLE_SCRIPT.read_bytes()
            and b"post_commit_interface_falsifier" not in ROLE_SCRIPT.read_bytes()
            and all(run["command"][1] == str(ROLE_SCRIPT) for run in invocations)
        ),
        "full_deterministic_replay_byte_identical": False,
    }
    report: dict[str, Any] = {
        "audit": "post_commit_packet_transport_v2",
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_sha256(),
        "scientific_identity": identity,
        "config": {
            "field_modulus": v1.MODULUS,
            "state_dimension": v1.DIMENSION,
            "packet_width_field_elements": v1.PACKET_WIDTH,
            "source_count": expected_total,
            "depths": list(v1.DEPTHS),
            "challenge_seed": challenge_seed,
            "alternate_challenge_seed": alternate_challenge_seed,
        },
        "phase_one": phase_one,
        "phase_two": {
            "challenge_payload_sha256": challenge_bundle["payload_sha256"],
            "alternate_challenge_payload_sha256": alternate_bundle["payload_sha256"],
            "public_challenges": len(public_results),
            "decisive_challenges": len(decisive_results),
            "one_per_cell_output_permutations": len(permutations),
            "unique_output_permutations_observed": len(set(permutations)),
            "sampling_contract": "independent uniform conditional on nonidentity",
            "phase_order_proof": phase_order_proof,
        },
        "custody_events": custody_events,
        "public_results": public_results,
        "decisive_results": decisive_results,
        "horizon_decoy_results": horizons,
        "executed_decoys": decoys,
        "role_invocations": invocations,
        "role_invocation_counts": role_counts,
        "deterministic_replay": pending_replay_record(),
        "gates": gates,
        "pass": all(gates.values()),
        "claim_boundary": (
            "A pass validates only the exact process-separated packet transport and "
            "custody harness. It is symbolic, compares a full state with a rank-two "
            "control, and does not establish learned reasoning."
        ),
    }
    report["payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def finalize_deterministic_replay(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    first_bytes = canonical_json_bytes(first)
    second_bytes = canonical_json_bytes(second)
    if first_bytes != second_bytes:
        raise TransportError("two full core reports are not byte-identical")
    first_gates = first.get("gates")
    if not isinstance(first_gates, Mapping) or set(first_gates) != FROZEN_GATE_NAMES:
        raise TransportError("core report gate schema is not frozen")
    if any(
        not value
        for name, value in first_gates.items()
        if name != "full_deterministic_replay_byte_identical"
    ):
        raise TransportError("a non-replay core gate failed")
    if first_gates["full_deterministic_replay_byte_identical"]:
        raise TransportError("core report claimed replay before the second run")

    finalized = json.loads(json.dumps(first))
    first_payload_sha256 = finalized["payload_sha256"]
    finalized["deterministic_replay"] = {
        "status": "confirmed_byte_identical",
        "first_core_payload_sha256": first_payload_sha256,
        "second_core_payload_sha256": second["payload_sha256"],
        "second_core_report": json.loads(json.dumps(second)),
    }
    finalized["gates"]["full_deterministic_replay_byte_identical"] = True
    finalized["pass"] = all(finalized["gates"].values())
    finalized.pop("payload_sha256", None)
    finalized["payload_sha256"] = sha256_bytes(canonical_json_bytes(finalized))
    return finalized


def verify_evidence_shape(report: Mapping[str, Any]) -> None:
    public = report.get("public_results")
    decisive = report.get("decisive_results")
    horizons = report.get("horizon_decoy_results")
    invocations = report.get("role_invocations")
    if not all(isinstance(value, list) for value in (public, decisive, horizons, invocations)):
        raise TransportError("report evidence lists are missing")
    public_layout = tuple(
        (row.get("challenge_id"), row.get("kind"), row.get("depth"))
        for row in public
        if isinstance(row, Mapping)
    )
    decisive_layout = tuple(
        (row.get("challenge_id"), row.get("kind"), row.get("depth"))
        for row in decisive
        if isinstance(row, Mapping)
    )
    if public_layout != EXPECTED_PUBLIC_LAYOUT or decisive_layout != EXPECTED_DECISIVE_LAYOUT:
        raise TransportError("report cell layout is not the frozen layout")
    expected_total = v1.MODULUS**v1.DIMENSION
    expected_motor = v1.MODULUS ** (v1.DIMENSION - 1)
    if not all(
        row.get("state_correct") == expected_total
        and row.get("motor_correct") == expected_total
        and row.get("state_direct_incremental_packet_match")
        for row in public
    ):
        raise TransportError("public score evidence is incomplete")
    if not all(
        row.get("state_correct") == expected_total
        and row.get("motor_correct") == expected_motor
        and row.get("state_direct_incremental_packet_match")
        and isinstance(row.get("collision_witness"), Mapping)
        and row["collision_witness"].get("terminal_motor_packets_equal")
        and row["collision_witness"].get("terminal_motor_symbols_equal")
        and row["collision_witness"].get("oracle_symbols_distinct")
        for row in decisive
    ):
        raise TransportError("decisive score evidence is incomplete")
    if len(horizons) != len(EXPECTED_PUBLIC_LAYOUT) + len(EXPECTED_DECISIVE_LAYOUT):
        raise TransportError("horizon evidence count is wrong")
    if not all(
        isinstance(row, Mapping)
        and isinstance(row.get("depth"), int)
        and row.get("reader_role") == "reader"
        and row.get("reader_interface_matches_canonical")
        and (
            (row.get("depth") <= 8 and row.get("correct") == expected_total)
            or (
                row.get("depth") == 9
                and row.get("applied_events") == 8
                and isinstance(row.get("correct"), int)
                and row.get("correct") < expected_total
            )
        )
        for row in horizons
    ):
        raise TransportError("horizon evidence is incomplete")

    counts: dict[str, int] = {}
    if len(invocations) != sum(EXPECTED_ROLE_COUNTS.values()):
        raise TransportError("role invocation evidence count is wrong")
    source_tree = report["scientific_identity"]["scientific_source_tree_sha256"]
    for invocation in invocations:
        if not isinstance(invocation, Mapping) or set(invocation) != ROLE_RUN_FIELDS:
            raise TransportError("role invocation schema is invalid")
        role = invocation.get("role")
        command = invocation.get("command")
        arguments = invocation.get("arguments")
        if (
            role not in EXPECTED_ROLE_COUNTS
            or not isinstance(command, list)
            or len(command) < 3
            or command[1] != str(ROLE_SCRIPT)
            or command[2] != role
            or not isinstance(arguments, list)
            or command[3:] != arguments
            or invocation.get("scientific_source_tree_sha256") != source_tree
        ):
            raise TransportError("role invocation is not bound to the seed-free executable")
        counts[role] = counts.get(role, 0) + 1
    if counts != EXPECTED_ROLE_COUNTS or report.get("role_invocation_counts") != counts:
        raise TransportError("role invocation counts do not match frozen constants")

    phase_one = report.get("phase_one")
    if not isinstance(phase_one, Mapping):
        raise TransportError("phase-one evidence is missing")
    phase_one_payload = dict(phase_one)
    claimed_manifest_sha256 = phase_one_payload.pop("manifest_sha256", None)
    if claimed_manifest_sha256 != sha256_bytes(canonical_json_bytes(phase_one_payload)):
        raise TransportError("phase-one manifest hash does not replay")
    config = report.get("config")
    phase_two = report.get("phase_two")
    if not isinstance(config, Mapping) or not isinstance(phase_two, Mapping):
        raise TransportError("configuration evidence is missing")
    primary_seed = _derive_post_commit_seed(
        claimed_manifest_sha256, PRIMARY_CHALLENGE_DOMAIN
    )
    alternate_seed = _derive_post_commit_seed(
        claimed_manifest_sha256, ALTERNATE_CHALLENGE_DOMAIN
    )
    if (
        config.get("field_modulus") != v1.MODULUS
        or config.get("state_dimension") != v1.DIMENSION
        or config.get("packet_width_field_elements") != v1.PACKET_WIDTH
        or config.get("source_count") != expected_total
        or config.get("depths") != list(v1.DEPTHS)
        or config.get("challenge_seed") != primary_seed
        or config.get("alternate_challenge_seed") != alternate_seed
        or phase_two.get("public_challenges") != len(EXPECTED_PUBLIC_LAYOUT)
        or phase_two.get("decisive_challenges") != len(EXPECTED_DECISIVE_LAYOUT)
    ):
        raise TransportError("configuration or frozen cell counts do not replay")


def verify_report(report: Mapping[str, Any]) -> None:
    if set(report) != REPORT_FIELDS:
        raise TransportError("report top-level schema does not match frozen fields")
    if (
        report.get("audit") != "post_commit_packet_transport_v2"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("schema_version") != SCHEMA_VERSION
    ):
        raise TransportError("report protocol identity is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != FROZEN_GATE_NAMES:
        raise TransportError("report gate schema does not match frozen gates")
    replay = report.get("deterministic_replay")
    if (
        not isinstance(replay, Mapping)
        or set(replay)
        != {
            "status",
            "first_core_payload_sha256",
            "second_core_payload_sha256",
            "second_core_report",
        }
        or replay.get("status") != "confirmed_byte_identical"
        or replay.get("first_core_payload_sha256")
        != replay.get("second_core_payload_sha256")
    ):
        raise TransportError("report lacks a confirmed full deterministic replay")
    second_core = replay.get("second_core_report")
    if not isinstance(second_core, Mapping) or set(second_core) != REPORT_FIELDS:
        raise TransportError("second full core report is missing or malformed")
    reconstructed_first = json.loads(json.dumps(report))
    reconstructed_first["deterministic_replay"] = pending_replay_record()
    reconstructed_first["gates"]["full_deterministic_replay_byte_identical"] = False
    reconstructed_first["pass"] = False
    reconstructed_first.pop("payload_sha256", None)
    reconstructed_first["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(reconstructed_first)
    )
    if canonical_json_bytes(reconstructed_first) != canonical_json_bytes(second_core):
        raise TransportError("embedded second core report is not byte-identical")
    if (
        reconstructed_first["payload_sha256"]
        != replay.get("first_core_payload_sha256")
        or second_core.get("payload_sha256")
        != replay.get("second_core_payload_sha256")
    ):
        raise TransportError("deterministic replay hashes do not bind both cores")
    copy = dict(report)
    claimed = copy.pop("payload_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(copy)):
        raise TransportError("report payload hash mismatch")
    current_identity = scientific_identity()
    if report.get("scientific_identity") != current_identity:
        raise TransportError("report scientific identity does not match committed harness")
    if report.get("code_sha256") != sha256_bytes(SCRIPT.read_bytes()):
        raise TransportError("report code hash does not match current implementation")
    phase_one = report.get("phase_one")
    if not isinstance(phase_one, Mapping):
        raise TransportError("report phase-one manifest is missing")
    if (
        phase_one.get("scientific_commit") != current_identity["scientific_commit"]
        or phase_one.get("scientific_source_tree_sha256")
        != current_identity["scientific_source_tree_sha256"]
    ):
        raise TransportError("phase-one manifest is not bound to scientific identity")
    verify_evidence_shape(report)
    verify_evidence_shape(second_core)
    if report.get("pass") is not True or not all(gates.values()):
        raise TransportError("one or more frozen v2 gates failed")


def write_immutable_report(path: Path, report: Mapping[str, Any]) -> None:
    verify_report(report)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _immutable_write(path, canonical_json_bytes(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first = build_report()
    second = build_report()
    report = finalize_deterministic_replay(first, second)
    verify_report(report)
    write_immutable_report(args.out, report)
    print(
        "[pcpt-v2] pass={} public={} decisive={} payload_sha256={}".format(
            report["pass"],
            report["phase_two"]["public_challenges"],
            report["phase_two"]["decisive_challenges"],
            report["payload_sha256"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
