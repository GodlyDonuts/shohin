#!/usr/bin/env python3
"""Independent, fail-closed verifier for the PCPT v3 artifact.

This verifier intentionally does not import either PCPT implementation module.
It reimplements the finite-field protocol, nonce-bound challenge generator,
role byte streams, evidence records, and replay binding from the frozen public
contract. It independently verifies every role record and refuses incomplete
nonce-commitment or filesystem-sandbox evidence. Its publisher mode, not the
parent harness, creates the canonical artifact and its bound receipt.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import itertools
import json
import os
import random
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "R12-PCPT-F17x4-v3"
AUDIT_ID = "post_commit_packet_transport_v3"
SCHEMA_VERSION = 3
MODULUS = 17
DIMENSION = 4
PUBLIC_DIMENSION = 2
PACKET_WIDTH = 4
DEPTHS = (1, 2, 4, 8, 9)
DECISIVE_KINDS = ("hidden_consumer", "hidden_update", "joint")
PRIMARY_CHALLENGE_DOMAIN = "R12-PCPT-v3-primary-challenge"
ALTERNATE_CHALLENGE_DOMAIN = "R12-PCPT-v3-alternate-challenge"
DECOY_CHALLENGE_ID = "hidden_consumer-d2-a0"
SOURCE_COUNT = MODULUS**DIMENSION
EXPECTED_MOTOR_CORRECT = MODULUS ** (DIMENSION - 1)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ROLE_SCRIPT_SUFFIX = "/pipeline/post_commit_packet_transport_roles.py"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
SANDBOX_POLICY_NAME = "default-deny-protected-runtime-role-cwd-v3-r2"
PROTECTED_RUNTIME_ROOT = "/Library/Developer/CommandLineTools"
ROLE_PYTHON = (
    PROTECTED_RUNTIME_ROOT
    + "/Library/Frameworks/Python3.framework/Versions/3.9/Resources/"
    "Python.app/Contents/MacOS/Python"
)
SANDBOX_FORBIDDEN_ROOTS = ["all_unlisted_paths_default_denied"]
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
CANONICAL_ARTIFACT = (
    REPO_ROOT / "artifacts/r12/post_commit_packet_transport_v3.json"
)
CANONICAL_RECEIPT = CANONICAL_ARTIFACT.with_suffix(".receipt.json")
RECEIPT_FIELDS = frozenset(
    {
        "status",
        "protocol_id",
        "scientific_commit",
        "core_payload_sha256",
        "role_invocations_per_core",
        "public_cells",
        "decisive_cells",
        "evidence_reconstruction_pass",
        "git_verification_mode",
        "artifact_path",
        "artifact_bytes",
        "artifact_mode",
        "artifact_md5",
        "artifact_sha256",
        "artifact_payload_sha256",
        "receipt_path",
        "verifier_source_path",
        "verifier_source_sha256",
        "verifier_scientific_path_sha256",
        "receipt_payload_sha256",
    }
)

EXPECTED_PUBLIC_LAYOUT = tuple(
    (f"public-d{depth}", "public", depth) for depth in DEPTHS
)
EXPECTED_DECISIVE_LAYOUT = tuple(
    (f"{kind}-d{depth}-a0", kind, depth) for depth in DEPTHS for kind in DECISIVE_KINDS
)
EXPECTED_ROLE_COUNTS = {
    "writer": 5,
    "updater": 227,
    "reader": 64,
    "oracle": 40,
    "raw_reader": 1,
}

SCIENTIFIC_PATHS = (
    "R12_POST_COMMIT_PACKET_TRANSPORT_V3_PREREG.md",
    "pipeline/post_commit_interface_falsifier.py",
    "pipeline/post_commit_packet_transport_falsifier.py",
    "pipeline/post_commit_packet_transport_roles.py",
    "pipeline/test_post_commit_packet_transport_falsifier.py",
    "pipeline/audit_post_commit_packet_transport_v3.py",
    "pipeline/test_audit_post_commit_packet_transport_v3.py",
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
        "sandbox_probe",
        "deterministic_replay",
        "gates",
        "pass",
        "claim_boundary",
        "payload_sha256",
    }
)
CLAIM_BOUNDARY = (
    "A pass validates the exact symbolic packet algebra and reproducible "
    "process-isolation behavior under the committed harness. It is not a "
    "tamper-proof historical attestation against a malicious machine owner, "
    "does not cryptographically attest nonce entropy provenance, "
    "compares a full state with a rank-two control, and does not establish "
    "learned reasoning."
)

GATE_NAMES = frozenset(
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
        "os_filesystem_sandbox_enforced",
        "nonce_commitment_bound_after_phase_one",
        "full_deterministic_replay_byte_identical",
    }
)

ROLE_FIELDS = frozenset(
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
        "sandbox_launcher",
        "sandbox_policy_name",
        "sandbox_profile_sha256",
        "sandbox_enforced",
        "sandbox_forbidden_roots",
    }
)

FILE_PRESENT_FIELDS = frozenset({"path", "exists", "bytes", "mode", "sha256"})
FILE_MISSING_FIELDS = frozenset({"path", "exists"})
SCIENTIFIC_IDENTITY_FIELDS = frozenset(
    {
        "scientific_commit",
        "scientific_paths",
        "scientific_source_tree_sha256",
        "runtime",
        "role_environment_policy",
        "role_environment_policy_sha256",
        "verified_against_head",
    }
)
RUNTIME_FIELDS = frozenset({"python_implementation", "python_version", "platform"})
ENVIRONMENT_POLICY = {
    "PATH": "protected_role_python_bin_and_usr_bin_only",
    "PYTHONPATH": "absent",
    "PYTHONNOUSERSITE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "role_cwd",
    "TMPDIR": "role_cwd",
    "LC_ALL": "C",
    "LANG": "C",
    "python_flags": ["-I", "-S"],
    "role_python": ROLE_PYTHON,
    "protected_runtime_root": PROTECTED_RUNTIME_ROOT,
    "sandbox_launcher": SANDBOX_EXEC,
    "sandbox_policy_name": SANDBOX_POLICY_NAME,
}
SANDBOX_PROBE_FIELDS = frozenset(
    {
        "sandbox_launcher",
        "sandbox_policy_name",
        "sandbox_profile_sha256",
        "sandbox_forbidden_roots",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "checks",
        "cwd_regular_files_before",
        "cwd_regular_files_after",
    }
)
SANDBOX_PROBE_CHECKS = frozenset(
    {
        "allowed_input_read",
        "role_executable_read",
        "parent_read_blocked",
        "repository_read_blocked",
        "parent_listing_blocked",
        "repository_listing_blocked",
        "etc_passwd_read_blocked",
        "etc_passwd_metadata_blocked",
        "system_listing_blocked",
        "parent_write_blocked",
        "tmp_read_blocked",
        "tmp_write_blocked",
        "applications_write_blocked",
        "library_cache_read_blocked",
        "library_cache_write_blocked",
        "private_var_db_read_blocked",
        "private_var_db_write_blocked",
        "osanalytics_read_blocked",
        "osanalytics_write_blocked",
        "data_osanalytics_read_blocked",
        "data_osanalytics_write_blocked",
        "diagnostic_reports_read_blocked",
        "diagnostic_reports_write_blocked",
        "blackmagic_support_read_blocked",
        "blackmagic_support_write_blocked",
        "network_socket_blocked",
        "local_write_allowed",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "field_modulus",
        "state_dimension",
        "packet_width_field_elements",
        "source_count",
        "depths",
        "challenge_nonce_hex",
        "challenge_seed",
        "alternate_challenge_seed",
    }
)
PHASE_ONE_FIELDS = frozenset(
    {
        "protocol_id",
        "source_count",
        "packet_width_field_elements",
        "source_payload_sha256",
        "state_packets_sha256",
        "motor_packets_sha256",
        "state_packets_mode",
        "motor_packets_mode",
        "code_sha256",
        "scientific_commit",
        "scientific_source_tree_sha256",
        "manifest_mode",
        "manifest_sha256",
    }
)
PHASE_TWO_FIELDS = frozenset(
    {
        "challenge_payload_sha256",
        "alternate_challenge_payload_sha256",
        "public_challenges",
        "decisive_challenges",
        "one_per_cell_output_permutations",
        "unique_output_permutations_observed",
        "sampling_contract",
        "phase_order_proof",
    }
)
PHASE_ORDER_FIELDS = frozenset(
    {
        "protocol_id",
        "challenge_payload_sha256",
        "challenge_seed",
        "challenge_seed_derivation_domain",
        "challenge_nonce_hex",
        "challenge_nonce_commitment_sha256",
        "challenge_nonce_file_mode",
        "challenge_nonce_committed_after_phase_one",
        "challenge_seed_derived_after_manifest_observation",
        "phase_one_manifest_sha256",
        "phase_one_manifest_mode_observed",
        "challenge_file_mode",
        "generated_only_after_phase_one_manifest_observation",
        "challenge_manifest_sha256",
        "ordering_clock",
        "ordering_measurement",
        "raw_clock_values_omitted_for_deterministic_replay",
    }
)

Vector = tuple[int, int, int, int]
Matrix = tuple[Vector, Vector, Vector, Vector]


class AuditFailure(ValueError):
    """Raised when evidence fails independent reconstruction."""


@dataclass(frozen=True)
class AffineUpdate:
    matrix: Matrix
    offset: Vector

    def serialized(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "offset": list(self.offset),
        }


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    kind: str
    depth: int
    updates: tuple[AffineUpdate, ...]
    consumer: Vector
    output_permutation: tuple[int, ...]

    def serialized(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "kind": self.kind,
            "depth": self.depth,
            "updates": [update.serialized() for update in self.updates],
            "consumer": list(self.consumer),
            "output_permutation": list(self.output_permutation),
        }


@dataclass(frozen=True)
class PacketStream:
    indices: tuple[int, ...]
    payload: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.payload)


@dataclass(frozen=True)
class SymbolStream:
    values: tuple[int, ...]
    payload: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.payload)


@dataclass(frozen=True)
class CommandContract:
    sandbox_executable: str
    sandbox_profile: str
    python_executable: str
    role_script: str

    def command(self, role: str, arguments: Sequence[str]) -> list[str]:
        return [
            self.sandbox_executable,
            "-p",
            self.sandbox_profile,
            self.python_executable,
            "-I",
            "-S",
            self.role_script,
            role,
            *arguments,
        ]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_sandbox_profile() -> str:
    """Rebuild the path-normalized SBPL profile without implementation imports."""

    metadata_paths = (
        "/",
        "/System",
        "/System/Cryptexes",
        "<SYSTEM_CRYPTEX_OS>",
        "/Library",
        "/Library/Developer",
        "<PROTECTED_RUNTIME_ROOT>",
        "<USERS_ROOT>",
        "<USER_HOME>",
        "<PROJECTS_ROOT>",
        "<REPOSITORY_ROOT>",
        "<ROLE_SCRIPT_PARENT>",
        "/private",
        "/private/tmp",
        "<ISOLATION_ROOT>",
        "<ROLE_CWD>",
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow syscall-unix (syscall-number SYS___mac_syscall "
        "SYS_getfsstat SYS_getfsstat64 SYS_map_with_linking_np SYS_open "
        "SYS_openat SYS_fstatat SYS_fstatat64 SYS_dup))",
        "(allow system-fcntl (fcntl-command F_ADDFILESIGS_RETURN "
        "F_CHECK_LV F_GETPATH))",
        '(with-filter (mac-policy-name "Sandbox") '
        '(allow system-mac-syscall (mac-syscall-number 2)))',
        "(deny network*)",
        "(allow sysctl-read)",
        '(allow process-exec (literal "<ROLE_PYTHON>"))',
        '(allow file-read* (literal "/"))',
        '(allow file-read* (literal "/dev/urandom"))',
        '(allow file-read* file-write-data (literal "/dev/null"))',
    ]
    lines.extend(
        f"(allow file-read-metadata (literal {json.dumps(path)}))"
        for path in metadata_paths
    )
    lines.extend(
        [
            '(allow file-read* (subpath "<PROTECTED_RUNTIME_ROOT>"))',
            '(allow file-read* (literal "<ROLE_SCRIPT>"))',
            '(allow file-read* (subpath "<ROLE_CWD>"))',
            '(allow file-write* (subpath "<ROLE_CWD>"))',
        ]
    )
    return "".join(lines)


def normalized_sandbox_profile_sha256() -> str:
    return sha256_bytes(normalized_sandbox_profile().encode("utf-8"))


def expected_sandbox_probe() -> dict[str, Any]:
    checks = {name: True for name in SANDBOX_PROBE_CHECKS}
    return {
        "sandbox_launcher": SANDBOX_EXEC,
        "sandbox_policy_name": SANDBOX_POLICY_NAME,
        "sandbox_profile_sha256": normalized_sandbox_profile_sha256(),
        "sandbox_forbidden_roots": list(SANDBOX_FORBIDDEN_ROOTS),
        "exit_code": 0,
        "stdout_sha256": sha256_bytes(canonical_json_bytes(checks)),
        "stderr_sha256": EMPTY_SHA256,
        "checks": checks,
        "cwd_regular_files_before": ["allowed.txt"],
        "cwd_regular_files_after": ["allowed.txt", "local_write.txt"],
    }


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditFailure(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_document(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditFailure(f"artifact is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure("artifact root must be an object")
    if payload != canonical_json_bytes(value):
        raise AuditFailure("artifact bytes are not canonical newline-terminated JSON")
    return value


def _strict_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditFailure(f"{label} must be an object")
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise AuditFailure(f"{label} schema mismatch; missing={missing}, extra={extra}")
    return value


def _require_hex_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AuditFailure(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise AuditFailure(f"{label} must be boolean")
    return value


def mod(value: int) -> int:
    return int(value) % MODULUS


def dot(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditFailure("dot operands must have dimension four")
    return mod(sum(int(a) * int(b) for a, b in zip(left, right, strict=True)))


def vec_add(left: Sequence[int], right: Sequence[int]) -> Vector:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditFailure("vector operands must have dimension four")
    return tuple(mod(int(a) + int(b)) for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def mat_vec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> Vector:
    if len(matrix) != DIMENSION or any(len(row) != DIMENSION for row in matrix):
        raise AuditFailure("matrix must be four by four")
    return tuple(dot(row, vector) for row in matrix)  # type: ignore[return-value]


def mat_mul(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> Matrix:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditFailure("matrices must be four by four")
    columns = tuple(
        tuple(int(right[row][column]) for row in range(DIMENSION))
        for column in range(DIMENSION)
    )
    return tuple(tuple(dot(row, column) for column in columns) for row in left)  # type: ignore[return-value]


def transpose(matrix: Sequence[Sequence[int]]) -> Matrix:
    return tuple(
        tuple(int(matrix[row][column]) for row in range(DIMENSION))
        for column in range(DIMENSION)
    )  # type: ignore[return-value]


def identity_matrix() -> Matrix:
    return tuple(
        tuple(1 if row == column else 0 for column in range(DIMENSION))
        for row in range(DIMENSION)
    )  # type: ignore[return-value]


def is_invertible(matrix: Sequence[Sequence[int]]) -> bool:
    work = [[mod(value) for value in row] for row in matrix]
    rank = 0
    for column in range(DIMENSION):
        pivot = next((row for row in range(rank, DIMENSION) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, MODULUS)
        work[rank] = [mod(value * inverse) for value in work[rank]]
        for row in range(DIMENSION):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    mod(value - factor * pivot_value)
                    for value, pivot_value in zip(work[row], work[rank], strict=True)
                ]
        rank += 1
    return rank == DIMENSION


def _random_vector(rng: random.Random) -> Vector:
    return tuple(rng.randrange(MODULUS) for _ in range(DIMENSION))  # type: ignore[return-value]


def _random_matrix(rng: random.Random) -> Matrix:
    while True:
        matrix = tuple(
            tuple(rng.randrange(MODULUS) for _ in range(DIMENSION))
            for _ in range(DIMENSION)
        )
        if is_invertible(matrix):
            return matrix  # type: ignore[return-value]


def _random_invertible_two(
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        matrix = (
            (rng.randrange(MODULUS), rng.randrange(MODULUS)),
            (rng.randrange(MODULUS), rng.randrange(MODULUS)),
        )
        determinant = mod(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
        if determinant:
            return matrix


def _public_matrix(rng: random.Random) -> Matrix:
    upper = _random_invertible_two(rng)
    lower = _random_invertible_two(rng)
    coupling = (
        (rng.randrange(MODULUS), rng.randrange(MODULUS)),
        (rng.randrange(MODULUS), rng.randrange(MODULUS)),
    )
    matrix: Matrix = (
        (upper[0][0], upper[0][1], 0, 0),
        (upper[1][0], upper[1][1], 0, 0),
        (coupling[0][0], coupling[0][1], lower[0][0], lower[0][1]),
        (coupling[1][0], coupling[1][1], lower[1][0], lower[1][1]),
    )
    if not is_invertible(matrix):
        raise AssertionError("block-preserving matrix must be invertible")
    return matrix


def _updates(
    rng: random.Random, depth: int, *, public: bool
) -> tuple[AffineUpdate, ...]:
    factory = _public_matrix if public else _random_matrix
    return tuple(AffineUpdate(factory(rng), _random_vector(rng)) for _ in range(depth))


def compose_affine(updates: Sequence[AffineUpdate]) -> AffineUpdate:
    matrix = identity_matrix()
    offset: Vector = (0, 0, 0, 0)
    for update in updates:
        offset = vec_add(mat_vec(update.matrix, offset), update.offset)
        matrix = mat_mul(update.matrix, matrix)
    return AffineUpdate(matrix, offset)


def effective_functional(challenge: Challenge) -> tuple[Vector, int]:
    total = compose_affine(challenge.updates)
    coefficient = mat_vec(transpose(total.matrix), challenge.consumer)
    constant = dot(challenge.consumer, total.offset)
    return coefficient, constant


def _public_consumer(rng: random.Random) -> Vector:
    while True:
        result: Vector = (
            rng.randrange(MODULUS),
            rng.randrange(MODULUS),
            0,
            0,
        )
        if any(result):
            return result


def _hidden_consumer(rng: random.Random) -> Vector:
    while True:
        result = _random_vector(rng)
        if any(result[PUBLIC_DIMENSION:]):
            return result


def _make_decisive_challenge(
    rng: random.Random,
    kind: str,
    depth: int,
    output_permutation: tuple[int, ...],
) -> Challenge:
    for attempt in range(10_000):
        if kind == "hidden_consumer":
            updates = _updates(rng, depth, public=True)
            consumer = _hidden_consumer(rng)
        elif kind == "hidden_update":
            updates = _updates(rng, depth, public=False)
            consumer = _public_consumer(rng)
        elif kind == "joint":
            updates = _updates(rng, depth, public=False)
            consumer = _hidden_consumer(rng)
        else:
            raise AuditFailure(f"unknown decisive kind: {kind}")
        challenge = Challenge(
            challenge_id=f"{kind}-d{depth}-a{attempt}",
            kind=kind,
            depth=depth,
            updates=updates,
            consumer=consumer,
            output_permutation=output_permutation,
        )
        coefficient, _ = effective_functional(challenge)
        if any(coefficient[PUBLIC_DIMENSION:]):
            return challenge
    raise AuditFailure("failed to generate a decisive challenge")


def _cell_permutation(challenge_seed: int, challenge_id: str) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}:{challenge_seed}:{challenge_id}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    while True:
        values = list(range(MODULUS))
        rng.shuffle(values)
        if any(index != value for index, value in enumerate(values)):
            return tuple(values)


def generate_challenges(challenge_seed: int) -> dict[str, Any]:
    rng = random.Random(int(challenge_seed))
    shift = rng.randrange(1, MODULUS)
    base_permutation = tuple((value + shift) % MODULUS for value in range(MODULUS))
    public: list[Challenge] = []
    decisive: list[Challenge] = []
    for depth in DEPTHS:
        public.append(
            Challenge(
                challenge_id=f"public-d{depth}",
                kind="public",
                depth=depth,
                updates=_updates(rng, depth, public=True),
                consumer=_public_consumer(rng),
                output_permutation=base_permutation,
            )
        )
        for kind in DECISIVE_KINDS:
            decisive.append(
                _make_decisive_challenge(rng, kind, depth, base_permutation)
            )
    public = [
        Challenge(
            item.challenge_id,
            item.kind,
            item.depth,
            item.updates,
            item.consumer,
            _cell_permutation(challenge_seed, item.challenge_id),
        )
        for item in public
    ]
    decisive = [
        Challenge(
            item.challenge_id,
            item.kind,
            item.depth,
            item.updates,
            item.consumer,
            _cell_permutation(challenge_seed, item.challenge_id),
        )
        for item in decisive
    ]
    serialized = {
        "challenge_seed": int(challenge_seed),
        "public": [item.serialized() for item in public],
        "decisive": [item.serialized() for item in decisive],
    }
    return {
        "challenge_seed": int(challenge_seed),
        "public": tuple(public),
        "decisive": tuple(decisive),
        "serialized": serialized,
        "payload_sha256": sha256_bytes(canonical_json_bytes(serialized)),
    }


def _validate_nonce_hex(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AuditFailure("challenge nonce must be 32 lowercase hexadecimal bytes")
    if len(bytes.fromhex(value)) != 32:
        raise AuditFailure("challenge nonce must decode to 32 bytes")
    return value


def _derive_nonce_seed(
    manifest_sha256: str, challenge_nonce_hex: str, domain: str
) -> int:
    challenge_nonce_hex = _validate_nonce_hex(challenge_nonce_hex)
    digest = hashlib.sha256(
        domain.encode("ascii")
        + manifest_sha256.encode("ascii")
        + bytes.fromhex(challenge_nonce_hex)
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_nonce_and_phase_bindings(report: Mapping[str, Any]) -> str:
    """Validate the v3 nonce ceremony before full packet reconstruction."""

    raw_config = report.get("config")
    if not isinstance(raw_config, Mapping):
        raise AuditFailure("config must be an object containing the challenge nonce")
    if "challenge_nonce_hex" not in raw_config:
        raise AuditFailure("challenge nonce is absent from config")
    config = _strict_keys(raw_config, CONFIG_FIELDS, "config")
    phase_one = _strict_keys(report.get("phase_one"), PHASE_ONE_FIELDS, "phase_one")
    phase_two = _strict_keys(report.get("phase_two"), PHASE_TWO_FIELDS, "phase_two")
    _strict_keys(
        phase_two["phase_order_proof"],
        PHASE_ORDER_FIELDS,
        "phase_two.phase_order_proof",
    )

    phase_one_payload = dict(phase_one)
    manifest_sha256 = _require_hex_digest(
        phase_one_payload.pop("manifest_sha256"), "phase_one.manifest_sha256"
    )
    if sha256_bytes(canonical_json_bytes(phase_one_payload)) != manifest_sha256:
        raise AuditFailure("phase-one manifest hash mismatch")

    challenge_nonce_hex = _validate_nonce_hex(config["challenge_nonce_hex"])
    challenge_seed = _derive_nonce_seed(
        manifest_sha256, challenge_nonce_hex, PRIMARY_CHALLENGE_DOMAIN
    )
    alternate_seed = _derive_nonce_seed(
        manifest_sha256, challenge_nonce_hex, ALTERNATE_CHALLENGE_DOMAIN
    )
    if config["challenge_seed"] != challenge_seed:
        raise AuditFailure("config.challenge_seed is not bound to nonce and phase one")
    if config["alternate_challenge_seed"] != alternate_seed:
        raise AuditFailure(
            "config.alternate_challenge_seed is not bound to nonce and phase one"
        )
    expected_config = {
        "field_modulus": MODULUS,
        "state_dimension": DIMENSION,
        "packet_width_field_elements": PACKET_WIDTH,
        "source_count": SOURCE_COUNT,
        "depths": list(DEPTHS),
        "challenge_nonce_hex": challenge_nonce_hex,
        "challenge_seed": challenge_seed,
        "alternate_challenge_seed": alternate_seed,
    }
    difference = _first_difference(expected_config, dict(config), "$.config")
    if difference:
        raise AuditFailure(f"invalid v3 configuration: {difference}")

    bundle = generate_challenges(challenge_seed)
    alternate_bundle = generate_challenges(alternate_seed)
    nonce_commitment_sha256 = sha256_bytes((challenge_nonce_hex + "\n").encode("ascii"))
    challenge_manifest = {
        "protocol_id": PROTOCOL_ID,
        "challenge_payload_sha256": bundle["payload_sha256"],
        "challenge_seed": challenge_seed,
        "challenge_seed_derivation_domain": PRIMARY_CHALLENGE_DOMAIN,
        "challenge_nonce_hex": challenge_nonce_hex,
        "challenge_nonce_commitment_sha256": nonce_commitment_sha256,
        "challenge_nonce_file_mode": "0444",
        "challenge_nonce_committed_after_phase_one": True,
        "challenge_seed_derived_after_manifest_observation": True,
        "phase_one_manifest_sha256": manifest_sha256,
        "phase_one_manifest_mode_observed": "0444",
        "challenge_file_mode": "0444",
        "generated_only_after_phase_one_manifest_observation": True,
    }
    phase_order_proof = {
        **challenge_manifest,
        "challenge_manifest_sha256": sha256_bytes(
            canonical_json_bytes(challenge_manifest)
        ),
        "ordering_clock": "time.monotonic_ns",
        "ordering_measurement": "strictly_after",
        "raw_clock_values_omitted_for_deterministic_replay": True,
    }
    permutations = [
        tuple(challenge.output_permutation)
        for challenge in (*bundle["public"], *bundle["decisive"])
    ]
    expected_phase_two = {
        "challenge_payload_sha256": bundle["payload_sha256"],
        "alternate_challenge_payload_sha256": alternate_bundle["payload_sha256"],
        "public_challenges": len(bundle["public"]),
        "decisive_challenges": len(bundle["decisive"]),
        "one_per_cell_output_permutations": len(permutations),
        "unique_output_permutations_observed": len(set(permutations)),
        "sampling_contract": "independent uniform conditional on nonidentity",
        "phase_order_proof": phase_order_proof,
    }
    difference = _first_difference(expected_phase_two, dict(phase_two), "$.phase_two")
    if difference:
        raise AuditFailure(f"invalid nonce or phase-two binding: {difference}")
    return challenge_nonce_hex


@functools.lru_cache(maxsize=1)
def all_states() -> tuple[Vector, ...]:
    return tuple(itertools.product(range(MODULUS), repeat=DIMENSION))  # type: ignore[return-value]


def encode_vector(vector: Sequence[int]) -> int:
    if len(vector) != DIMENSION:
        raise AuditFailure("vector must have dimension four")
    result = 0
    for value in vector:
        if not isinstance(value, int) or not 0 <= value < MODULUS:
            raise AuditFailure("vector value is outside F_17")
        result = result * MODULUS + value
    return result


@functools.lru_cache(maxsize=1)
def packet_rows() -> tuple[bytes, ...]:
    return tuple(
        canonical_json_bytes({"values": list(state)}) for state in all_states()
    )


@functools.lru_cache(maxsize=1)
def source_payload() -> bytes:
    return b"".join(
        canonical_json_bytes({"source": list(state)}) for state in all_states()
    )


def _packet_stream(indices: Iterable[int]) -> PacketStream:
    frozen = tuple(indices)
    rows = packet_rows()
    return PacketStream(frozen, b"".join(rows[index] for index in frozen))


@functools.lru_cache(maxsize=1)
def state_packets() -> PacketStream:
    return _packet_stream(range(SOURCE_COUNT))


@functools.lru_cache(maxsize=1)
def motor_packets() -> PacketStream:
    return _packet_stream(
        encode_vector((state[0], state[1], 0, 0)) for state in all_states()
    )


def _apply_update_to_index(update: AffineUpdate, index: int) -> int:
    vector = all_states()[index]
    return encode_vector(vec_add(mat_vec(update.matrix, vector), update.offset))


def transform_packets(stream: PacketStream, update: AffineUpdate) -> PacketStream:
    mapping = {
        index: _apply_update_to_index(update, index) for index in set(stream.indices)
    }
    return _packet_stream(mapping[index] for index in stream.indices)


def packet_chain(
    initial: PacketStream, updates: Sequence[AffineUpdate]
) -> tuple[PacketStream, ...]:
    chain = [initial]
    for update in updates:
        chain.append(transform_packets(chain[-1], update))
    return tuple(chain)


@functools.lru_cache(maxsize=1)
def symbol_rows() -> tuple[bytes, ...]:
    return tuple(canonical_json_bytes({"symbol": value}) for value in range(MODULUS))


@functools.lru_cache(maxsize=1)
def raw_rows() -> tuple[bytes, ...]:
    return tuple(canonical_json_bytes({"raw": value}) for value in range(MODULUS))


def read_symbols(
    packets: PacketStream,
    consumer: Vector,
    permutation: Sequence[int],
    *,
    raw: bool = False,
) -> SymbolStream:
    values = tuple(dot(consumer, all_states()[index]) for index in packets.indices)
    if raw:
        rows = raw_rows()
        return SymbolStream(values, b"".join(rows[value] for value in values))
    recoded = tuple(int(permutation[value]) for value in values)
    rows = symbol_rows()
    return SymbolStream(recoded, b"".join(rows[value] for value in recoded))


def _stream(header: Mapping[str, Any], rows: bytes = b"") -> bytes:
    return canonical_json_bytes(dict(header)) + rows


def _file_present(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "exists": True,
        "bytes": len(payload),
        "mode": "0444",
        "sha256": sha256_bytes(payload),
    }


def _file_missing(path: str) -> dict[str, Any]:
    return {"path": path, "exists": False}


def _role_record(
    command: CommandContract,
    source_tree_sha256: str,
    role: str,
    arguments: Sequence[str],
    stdin: bytes,
    stdout: bytes,
    *,
    stderr: bytes = b"",
    exit_code: int = 0,
    file_inputs: Sequence[Mapping[str, Any]] = (),
    file_outputs: Sequence[Mapping[str, Any]] = (),
    cwd_before: Sequence[str] = (),
    cwd_after: Sequence[str] = (),
) -> dict[str, Any]:
    input_files = [dict(item) for item in file_inputs]
    output_files = [dict(item) for item in file_outputs]
    stdin_sha256 = sha256_bytes(stdin)
    stdout_sha256 = sha256_bytes(stdout)
    args = list(arguments)
    return {
        "role": role,
        "command": command.command(role, args),
        "arguments": args,
        "input_sha256": sha256_bytes(
            canonical_json_bytes(
                {"stdin_sha256": stdin_sha256, "file_inputs": input_files}
            )
        ),
        "output_sha256": sha256_bytes(
            canonical_json_bytes(
                {"stdout_sha256": stdout_sha256, "file_outputs": output_files}
            )
        ),
        "stderr_sha256": sha256_bytes(stderr),
        "exit_code": exit_code,
        "stdin_sha256": stdin_sha256,
        "stdout_sha256": stdout_sha256,
        "file_inputs": input_files,
        "file_outputs": output_files,
        "scientific_source_tree_sha256": source_tree_sha256,
        "cwd_regular_files_before": list(cwd_before),
        "cwd_regular_files_after": list(cwd_after),
        "sandbox_launcher": SANDBOX_EXEC,
        "sandbox_policy_name": SANDBOX_POLICY_NAME,
        "sandbox_profile_sha256": normalized_sandbox_profile_sha256(),
        "sandbox_enforced": True,
        "sandbox_forbidden_roots": list(SANDBOX_FORBIDDEN_ROOTS),
    }


def _writer_record(
    command: CommandContract,
    source_tree_sha256: str,
    arm: str,
    sources: bytes,
    packets: PacketStream,
) -> dict[str, Any]:
    stdin = _stream({"protocol_id": PROTOCOL_ID, "role": "writer"}, sources)
    return _role_record(
        command,
        source_tree_sha256,
        "writer",
        ["--arm", arm],
        stdin,
        packets.payload,
    )


def _updater_stdin(update: AffineUpdate, **extra: Any) -> bytes:
    header = {
        "protocol_id": PROTOCOL_ID,
        "role": "updater",
        "update": update.serialized(),
        **extra,
    }
    return _stream(header)


def _append_transport_records(
    invocations: list[dict[str, Any]],
    command: CommandContract,
    source_tree_sha256: str,
    chain: Sequence[PacketStream],
    updates: Sequence[AffineUpdate],
) -> list[str]:
    hashes = [chain[0].sha256]
    for before, after, update in zip(chain[:-1], chain[1:], updates, strict=True):
        invocations.append(
            _role_record(
                command,
                source_tree_sha256,
                "updater",
                [
                    "--packet-in",
                    "packet_in.jsonl",
                    "--packet-out",
                    "packet_out.jsonl",
                ],
                _updater_stdin(update),
                b"",
                file_inputs=[_file_present("packet_in.jsonl", before.payload)],
                file_outputs=[_file_present("packet_out.jsonl", after.payload)],
                cwd_before=["packet_in.jsonl"],
                cwd_after=["packet_in.jsonl", "packet_out.jsonl"],
            )
        )
        hashes.append(after.sha256)
    return hashes


def _reader_stdin(challenge: Challenge, role: str = "reader", **extra: Any) -> bytes:
    return _stream(
        {
            "protocol_id": PROTOCOL_ID,
            "role": role,
            "consumer": list(challenge.consumer),
            "output_permutation": list(challenge.output_permutation),
            **extra,
        }
    )


def _reader_record(
    command: CommandContract,
    source_tree_sha256: str,
    challenge: Challenge,
    packets: PacketStream,
    output: bytes,
    *,
    role: str = "reader",
    stdin: bytes | None = None,
    stderr: bytes = b"",
    exit_code: int = 0,
) -> dict[str, Any]:
    return _role_record(
        command,
        source_tree_sha256,
        role,
        ["--packet-in", "terminal_packet.jsonl"],
        _reader_stdin(challenge, role) if stdin is None else stdin,
        output,
        stderr=stderr,
        exit_code=exit_code,
        file_inputs=[_file_present("terminal_packet.jsonl", packets.payload)],
        cwd_before=["terminal_packet.jsonl"],
        cwd_after=["terminal_packet.jsonl"],
    )


def _oracle_stdin(challenge: Challenge, sources: bytes, emit: str) -> bytes:
    return _stream(
        {
            "protocol_id": PROTOCOL_ID,
            "role": "oracle",
            "challenge": challenge.serialized(),
            "emit": emit,
        },
        sources,
    )


def _oracle_record(
    command: CommandContract,
    source_tree_sha256: str,
    challenge: Challenge,
    sources: bytes,
    emit: str,
    output: bytes,
) -> dict[str, Any]:
    return _role_record(
        command,
        source_tree_sha256,
        "oracle",
        [],
        _oracle_stdin(challenge, sources, emit),
        output,
    )


def _score(candidate: SymbolStream, truth: SymbolStream) -> tuple[int, int]:
    if len(candidate.values) != len(truth.values):
        raise AuditFailure("candidate and oracle row counts differ")
    return (
        sum(
            left == right
            for left, right in zip(candidate.values, truth.values, strict=True)
        ),
        len(truth.values),
    )


def _collision_witness(
    challenge: Challenge,
    motor_terminal: PacketStream,
    motor_symbols: SymbolStream,
    oracle_symbols: SymbolStream,
) -> dict[str, Any]:
    coefficient, constant = effective_functional(challenge)
    hidden_coordinate = next(
        index for index in range(PUBLIC_DIMENSION, DIMENSION) if coefficient[index]
    )
    left: Vector = (0, 0, 0, 0)
    right_values = [0, 0, 0, 0]
    right_values[hidden_coordinate] = 1
    right: Vector = tuple(right_values)  # type: ignore[assignment]
    left_index = encode_vector(left)
    right_index = encode_vector(right)
    left_answer = mod(dot(coefficient, left) + constant)
    right_answer = mod(dot(coefficient, right) + constant)
    return {
        "left_source": list(left),
        "right_source": list(right),
        "shared_motor_packet": {"values": [0, 0, 0, 0]},
        "distinguishing_hidden_coordinate": hidden_coordinate,
        "left_answer": left_answer,
        "right_answer": right_answer,
        "left_recoded": challenge.output_permutation[left_answer],
        "right_recoded": challenge.output_permutation[right_answer],
        "terminal_motor_packets_equal": (
            motor_terminal.indices[left_index] == motor_terminal.indices[right_index]
        ),
        "terminal_motor_symbols_equal": (
            motor_symbols.values[left_index] == motor_symbols.values[right_index]
        ),
        "oracle_symbols_distinct": (
            oracle_symbols.values[left_index] != oracle_symbols.values[right_index]
        ),
    }


def _challenge_result(
    challenge: Challenge,
    sources: bytes,
    initial_state: PacketStream,
    initial_motor: PacketStream,
    invocations: list[dict[str, Any]],
    command: CommandContract,
    source_tree_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_chain = packet_chain(initial_state, challenge.updates)
    motor_chain = packet_chain(initial_motor, challenge.updates)
    state_terminal = state_chain[-1]
    motor_terminal = motor_chain[-1]
    direct_update = compose_affine(challenge.updates)
    oracle_terminal = transform_packets(initial_state, direct_update)
    oracle = read_symbols(
        oracle_terminal, challenge.consumer, challenge.output_permutation
    )

    invocations.append(
        _oracle_record(
            command,
            source_tree_sha256,
            challenge,
            sources,
            "symbols",
            oracle.payload,
        )
    )
    invocations.append(
        _oracle_record(
            command,
            source_tree_sha256,
            challenge,
            sources,
            "packets",
            oracle_terminal.payload,
        )
    )

    state_hashes = _append_transport_records(
        invocations,
        command,
        source_tree_sha256,
        state_chain,
        challenge.updates,
    )
    state_output = read_symbols(
        state_terminal, challenge.consumer, challenge.output_permutation
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            state_terminal,
            state_output.payload,
        )
    )

    motor_hashes = _append_transport_records(
        invocations,
        command,
        source_tree_sha256,
        motor_chain,
        challenge.updates,
    )
    motor_output = read_symbols(
        motor_terminal, challenge.consumer, challenge.output_permutation
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            motor_terminal,
            motor_output.payload,
        )
    )

    state_correct, total = _score(state_output, oracle)
    motor_correct, _ = _score(motor_output, oracle)
    coefficient, _ = effective_functional(challenge)
    decisive = any(coefficient[PUBLIC_DIMENSION:])
    result: dict[str, Any] = {
        "challenge_id": challenge.challenge_id,
        "kind": challenge.kind,
        "depth": challenge.depth,
        "output_permutation": list(challenge.output_permutation),
        "total_sources": total,
        "state_correct": state_correct,
        "motor_correct": motor_correct,
        "state_accuracy": f"{state_correct}/{total}",
        "motor_accuracy": f"{motor_correct}/{total}",
        "state_direct_incremental_packet_match": (
            state_terminal.payload == oracle_terminal.payload
        ),
        "state_terminal_sha256": state_terminal.sha256,
        "motor_terminal_sha256": motor_terminal.sha256,
        "oracle_symbols_sha256": oracle.sha256,
        "transport_packet_sha256": {
            "state": state_hashes,
            "motor": motor_hashes,
        },
        "decisive_outside_public_span": decisive,
    }
    if decisive:
        result["collision_witness"] = _collision_witness(
            challenge, motor_terminal, motor_output, oracle
        )

    if challenge.depth <= 8:
        horizon_terminal = state_terminal
        applied_events = challenge.depth
    else:
        horizon_chain = packet_chain(initial_state, challenge.updates[:8])
        _append_transport_records(
            invocations,
            command,
            source_tree_sha256,
            horizon_chain,
            challenge.updates[:8],
        )
        horizon_terminal = horizon_chain[-1]
        applied_events = 8
    horizon_output = read_symbols(
        horizon_terminal, challenge.consumer, challenge.output_permutation
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            horizon_terminal,
            horizon_output.payload,
        )
    )
    horizon_correct, _ = _score(horizon_output, oracle)
    horizon = {
        "challenge_id": challenge.challenge_id,
        "depth": challenge.depth,
        "reader_role": "reader",
        "reader_interface_matches_canonical": True,
        "applied_events": applied_events,
        "correct": horizon_correct,
        "total_sources": SOURCE_COUNT,
    }
    return result, {
        "horizon": horizon,
        "oracle": oracle,
        "state_terminal": state_terminal,
    }


def _role_error(message: str) -> bytes:
    return f"RoleError:{message}\n".encode("utf-8")


def _decoy_evidence(
    sources: bytes,
    initial_state: PacketStream,
    challenge: Challenge,
    oracle: SymbolStream,
    state_terminal: PacketStream,
    invocations: list[dict[str, Any]],
    command: CommandContract,
    source_tree_sha256: str,
) -> dict[str, Any]:
    pointer_payload = canonical_json_bytes(
        {"values": [0, 0, 0, 0], "source_id": "forbidden"}
    )
    pointer_stream = PacketStream((0,), pointer_payload)
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            pointer_stream,
            b"",
            stderr=_role_error(
                "packet fields must be ['values'], got ['source_id', 'values']"
            ),
            exit_code=2,
        )
    )

    one_source = canonical_json_bytes({"source": [0, 0, 0, 0]})
    bad_writer_stdin = _stream(
        {
            "protocol_id": PROTOCOL_ID,
            "role": "writer",
            "consumer": [1, 0, 0, 0],
        },
        one_source,
    )
    bad_writer = _role_record(
        command,
        source_tree_sha256,
        "writer",
        ["--arm", "state"],
        bad_writer_stdin,
        b"",
        stderr=_role_error(
            "writer header fields must be ['protocol_id', 'role'], got "
            "['consumer', 'protocol_id', 'role']"
        ),
        exit_code=2,
    )

    updater_args = [
        "--packet-in",
        "packet_in.jsonl",
        "--packet-out",
        "packet_out.jsonl",
    ]
    bad_history = _role_record(
        command,
        source_tree_sha256,
        "updater",
        updater_args,
        _updater_stdin(
            challenge.updates[0],
            history=[item.serialized() for item in challenge.updates],
        ),
        b"",
        stderr=_role_error(
            "updater header fields must be ['protocol_id', 'role', 'update'], got "
            "['history', 'protocol_id', 'role', 'update']"
        ),
        exit_code=2,
        file_inputs=[_file_present("packet_in.jsonl", initial_state.payload)],
        file_outputs=[_file_missing("packet_out.jsonl")],
        cwd_before=["packet_in.jsonl"],
        cwd_after=["packet_in.jsonl"],
    )
    bad_updater_source = _role_record(
        command,
        source_tree_sha256,
        "updater",
        updater_args,
        _updater_stdin(challenge.updates[0], source=[0, 0, 0, 0]),
        b"",
        stderr=_role_error(
            "updater header fields must be ['protocol_id', 'role', 'update'], got "
            "['protocol_id', 'role', 'source', 'update']"
        ),
        exit_code=2,
        file_inputs=[_file_present("packet_in.jsonl", initial_state.payload)],
        file_outputs=[_file_missing("packet_out.jsonl")],
        cwd_before=["packet_in.jsonl"],
        cwd_after=["packet_in.jsonl"],
    )
    bad_reader_source = _reader_record(
        command,
        source_tree_sha256,
        challenge,
        state_terminal,
        b"",
        stdin=_reader_stdin(challenge, source=[0, 0, 0, 0]),
        stderr=_role_error(
            "reader header fields must be ['consumer', 'output_permutation', "
            "'protocol_id', 'role'], got ['consumer', 'output_permutation', "
            "'protocol_id', 'role', 'source']"
        ),
        exit_code=2,
    )
    invocations.extend([bad_writer, bad_history, bad_updater_source, bad_reader_source])

    identity = identity_matrix()
    skipped_index = next(
        index
        for index, update in enumerate(challenge.updates)
        if update.matrix != identity or update.offset != (0, 0, 0, 0)
    )
    stale_updates = tuple(
        update
        for index, update in enumerate(challenge.updates)
        if index != skipped_index
    )
    stale_chain = packet_chain(initial_state, stale_updates)
    stale_hashes = _append_transport_records(
        invocations,
        command,
        source_tree_sha256,
        stale_chain,
        stale_updates,
    )
    stale_output = read_symbols(
        stale_chain[-1], challenge.consumer, challenge.output_permutation
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            stale_chain[-1],
            stale_output.payload,
        )
    )
    stale_correct, _ = _score(stale_output, oracle)

    shuffled = _packet_stream(reversed(state_terminal.indices))
    shuffled_output = read_symbols(
        shuffled, challenge.consumer, challenge.output_permutation
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            shuffled,
            shuffled_output.payload,
        )
    )
    shuffled_correct, _ = _score(shuffled_output, oracle)

    raw_output = read_symbols(
        state_terminal,
        challenge.consumer,
        challenge.output_permutation,
        raw=True,
    )
    invocations.append(
        _reader_record(
            command,
            source_tree_sha256,
            challenge,
            state_terminal,
            raw_output.payload,
            role="raw_reader",
        )
    )

    return {
        "query_visible_writer_rejected": True,
        "event_history_updater_rejected": True,
        "source_visible_updater_rejected": True,
        "source_visible_reader_rejected": True,
        "source_pointer_packet_rejected": True,
        "stale_packet": {
            "skipped_event_index": skipped_index,
            "skipped_event_nonidentity": True,
            "applied_event_count": len(stale_updates),
            "transport_packet_sha256": stale_hashes,
            "correct": stale_correct,
            "total_sources": SOURCE_COUNT,
            "rejected": stale_correct < SOURCE_COUNT,
        },
        "shuffled_packet": {
            "correct": shuffled_correct,
            "total_sources": SOURCE_COUNT,
            "rejected": shuffled_correct < SOURCE_COUNT,
        },
        "unrecoded_reader_schema_rejected": True,
        "source_payload_sha256": sha256_bytes(sources),
    }


def _pending_replay() -> dict[str, Any]:
    return {
        "status": "pending_second_full_run",
        "first_core_payload_sha256": None,
        "second_core_payload_sha256": None,
        "second_core_report": None,
    }


def _expected_gates() -> dict[str, bool]:
    return {
        name: name != "full_deterministic_replay_byte_identical" for name in GATE_NAMES
    }


def _validate_command_contract(report: Mapping[str, Any]) -> CommandContract:
    invocations = report.get("role_invocations")
    if not isinstance(invocations, list) or not invocations:
        raise AuditFailure("role_invocations must be a nonempty list")
    first = invocations[0]
    if not isinstance(first, Mapping):
        raise AuditFailure("role_invocations[0] must be an object")
    command = first.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 8
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise AuditFailure("role_invocations[0].command is malformed")
    sandbox_executable = command[0]
    sandbox_profile = command[2]
    python_executable = command[3]
    role_script = command[6]
    if sandbox_executable != SANDBOX_EXEC:
        raise AuditFailure("role command does not use /usr/bin/sandbox-exec")
    if command[1] != "-p" or sandbox_profile != normalized_sandbox_profile():
        raise AuditFailure("role command does not bind the normalized SBPL profile")
    if python_executable != ROLE_PYTHON:
        raise AuditFailure("role command does not use the protected runtime")
    if command[4:6] != ["-I", "-S"]:
        raise AuditFailure("role command does not isolate Python startup")
    expected_role_script = str(
        Path(__file__).resolve().parent / "post_commit_packet_transport_roles.py"
    )
    if role_script != expected_role_script:
        raise AuditFailure("role command script path is not the frozen role executable")
    return CommandContract(
        sandbox_executable, sandbox_profile, python_executable, role_script
    )


def _validate_scientific_identity(
    identity: Any, *, verify_git: bool
) -> Mapping[str, Any]:
    identity = _strict_keys(identity, SCIENTIFIC_IDENTITY_FIELDS, "scientific_identity")
    commit = identity["scientific_commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise AuditFailure("scientific_identity.scientific_commit is invalid")
    paths = identity["scientific_paths"]
    if not isinstance(paths, Mapping) or tuple(sorted(paths)) != tuple(
        sorted(SCIENTIFIC_PATHS)
    ):
        raise AuditFailure("scientific_identity.scientific_paths is not frozen")
    for path, digest in paths.items():
        if not isinstance(path, str):
            raise AuditFailure("scientific path must be a string")
        _require_hex_digest(digest, f"scientific path {path}")
    expected_tree = sha256_bytes(canonical_json_bytes(dict(paths)))
    if identity["scientific_source_tree_sha256"] != expected_tree:
        raise AuditFailure("scientific source-tree hash does not replay")
    runtime = _strict_keys(identity["runtime"], RUNTIME_FIELDS, "runtime")
    if any(not isinstance(runtime[name], str) or not runtime[name] for name in runtime):
        raise AuditFailure("runtime fields must be nonempty strings")
    if identity["role_environment_policy"] != ENVIRONMENT_POLICY:
        raise AuditFailure("role environment policy differs from the frozen policy")
    expected_policy_hash = sha256_bytes(canonical_json_bytes(ENVIRONMENT_POLICY))
    if identity["role_environment_policy_sha256"] != expected_policy_hash:
        raise AuditFailure("role environment policy hash does not replay")
    if identity["verified_against_head"] is not True:
        raise AuditFailure("scientific identity was not verified against its commit")
    if verify_git:
        _verify_git_identity(commit, tuple((str(k), str(v)) for k, v in paths.items()))
    return identity


@functools.lru_cache(maxsize=16)
def _verify_git_identity(commit: str, path_hashes: tuple[tuple[str, str], ...]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for relative, expected_hash in path_hashes:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AuditFailure(
                f"cannot read {relative} from scientific commit {commit}"
            )
        if sha256_bytes(completed.stdout) != expected_hash:
            raise AuditFailure(f"committed scientific path hash mismatch: {relative}")
    role_bytes = subprocess.run(
        ["git", "show", f"{commit}:pipeline/post_commit_packet_transport_roles.py"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if role_bytes.returncode != 0:
        raise AuditFailure("cannot inspect committed role executable")
    if (
        b"CHALLENGE_SEED" in role_bytes.stdout
        or b"post_commit_interface_falsifier" in role_bytes.stdout
    ):
        raise AuditFailure(
            "committed role executable contains forbidden seed/source import"
        )


def _validate_file_evidence(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise AuditFailure(f"{label} must be an object")
    fields = frozenset(value)
    if fields == FILE_PRESENT_FIELDS:
        if value["exists"] is not True:
            raise AuditFailure(f"{label}.exists contradicts present-file schema")
        if not isinstance(value["path"], str) or not value["path"]:
            raise AuditFailure(f"{label}.path is invalid")
        if not isinstance(value["bytes"], int) or value["bytes"] < 0:
            raise AuditFailure(f"{label}.bytes is invalid")
        if value["mode"] != "0444":
            raise AuditFailure(f"{label}.mode is not 0444")
        _require_hex_digest(value["sha256"], f"{label}.sha256")
    elif fields == FILE_MISSING_FIELDS:
        if value["exists"] is not False:
            raise AuditFailure(f"{label}.exists contradicts missing-file schema")
        if not isinstance(value["path"], str) or not value["path"]:
            raise AuditFailure(f"{label}.path is invalid")
    else:
        raise AuditFailure(f"{label} has an invalid file-evidence schema")


def _validate_role_record_schema(
    record: Any, index: int, command: CommandContract, source_tree: str
) -> None:
    record = _strict_keys(record, ROLE_FIELDS, f"role_invocations[{index}]")
    role = record["role"]
    if role not in EXPECTED_ROLE_COUNTS:
        raise AuditFailure(f"role_invocations[{index}].role is invalid")
    arguments = record["arguments"]
    if not isinstance(arguments, list) or any(
        not isinstance(v, str) for v in arguments
    ):
        raise AuditFailure(f"role_invocations[{index}].arguments is invalid")
    if record["command"] != command.command(str(role), arguments):
        raise AuditFailure(f"role_invocations[{index}].command/arguments mismatch")
    if type(record["exit_code"]) is not int:
        raise AuditFailure(f"role_invocations[{index}].exit_code must be integer")
    for name in (
        "input_sha256",
        "output_sha256",
        "stderr_sha256",
        "stdin_sha256",
        "stdout_sha256",
    ):
        _require_hex_digest(record[name], f"role_invocations[{index}].{name}")
    if record["scientific_source_tree_sha256"] != source_tree:
        raise AuditFailure(f"role_invocations[{index}] source-tree binding mismatch")
    if record["sandbox_launcher"] != SANDBOX_EXEC:
        raise AuditFailure(f"role_invocations[{index}].sandbox_launcher is invalid")
    if record["sandbox_policy_name"] != SANDBOX_POLICY_NAME:
        raise AuditFailure(f"role_invocations[{index}].sandbox_policy_name is invalid")
    if record["sandbox_profile_sha256"] != normalized_sandbox_profile_sha256():
        raise AuditFailure(
            f"role_invocations[{index}].sandbox_profile_sha256 does not replay"
        )
    if record["sandbox_enforced"] is not True:
        raise AuditFailure(f"role_invocations[{index}].sandbox_enforced is not true")
    if record["sandbox_forbidden_roots"] != SANDBOX_FORBIDDEN_ROOTS:
        raise AuditFailure(
            f"role_invocations[{index}].sandbox_forbidden_roots is invalid"
        )
    for name in ("file_inputs", "file_outputs"):
        values = record[name]
        if not isinstance(values, list):
            raise AuditFailure(f"role_invocations[{index}].{name} must be a list")
        for file_index, value in enumerate(values):
            _validate_file_evidence(
                value, f"role_invocations[{index}].{name}[{file_index}]"
            )
    for name in ("cwd_regular_files_before", "cwd_regular_files_after"):
        values = record[name]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            raise AuditFailure(f"role_invocations[{index}].{name} is invalid")


def _validate_sandbox_probe(value: Any) -> None:
    probe = _strict_keys(value, SANDBOX_PROBE_FIELDS, "sandbox_probe")
    checks = _strict_keys(probe["checks"], SANDBOX_PROBE_CHECKS, "sandbox_probe.checks")
    if any(checks[name] is not True for name in checks):
        raise AuditFailure("sandbox_probe contains a failed confinement check")
    expected = expected_sandbox_probe()
    difference = _first_difference(expected, dict(probe), "$.sandbox_probe")
    if difference:
        raise AuditFailure(f"sandbox probe does not replay: {difference}")


def _phase_one_manifest(
    identity: Mapping[str, Any],
    sources: bytes,
    state: PacketStream,
    motor: PacketStream,
) -> dict[str, Any]:
    code_sha256 = identity["scientific_paths"][
        "pipeline/post_commit_packet_transport_falsifier.py"
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_count": SOURCE_COUNT,
        "packet_width_field_elements": PACKET_WIDTH,
        "source_payload_sha256": sha256_bytes(sources),
        "state_packets_sha256": state.sha256,
        "motor_packets_sha256": motor.sha256,
        "state_packets_mode": "0444",
        "motor_packets_mode": "0444",
        "code_sha256": code_sha256,
        "scientific_commit": identity["scientific_commit"],
        "scientific_source_tree_sha256": identity["scientific_source_tree_sha256"],
        "manifest_mode": "0444",
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _build_expected_core_uncached(
    identity: Mapping[str, Any], command: CommandContract, challenge_nonce_hex: str
) -> dict[str, Any]:
    challenge_nonce_hex = _validate_nonce_hex(challenge_nonce_hex)
    sources = source_payload()
    state = state_packets()
    motor = motor_packets()
    source_tree = str(identity["scientific_source_tree_sha256"])
    invocations = [
        _writer_record(command, source_tree, "state", sources, state),
        _writer_record(command, source_tree, "motor", sources, motor),
    ]
    phase_runs = copy.deepcopy(invocations)
    phase_one = _phase_one_manifest(identity, sources, state, motor)
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

    nonce_commitment_sha256 = sha256_bytes((challenge_nonce_hex + "\n").encode("ascii"))
    challenge_seed = _derive_nonce_seed(
        phase_one["manifest_sha256"], challenge_nonce_hex, PRIMARY_CHALLENGE_DOMAIN
    )
    alternate_seed = _derive_nonce_seed(
        phase_one["manifest_sha256"], challenge_nonce_hex, ALTERNATE_CHALLENGE_DOMAIN
    )
    bundle = generate_challenges(challenge_seed)
    alternate_bundle = generate_challenges(alternate_seed)
    challenge_manifest = {
        "protocol_id": PROTOCOL_ID,
        "challenge_payload_sha256": bundle["payload_sha256"],
        "challenge_seed": challenge_seed,
        "challenge_seed_derivation_domain": PRIMARY_CHALLENGE_DOMAIN,
        "challenge_nonce_hex": challenge_nonce_hex,
        "challenge_nonce_commitment_sha256": nonce_commitment_sha256,
        "challenge_nonce_file_mode": "0444",
        "challenge_nonce_committed_after_phase_one": True,
        "challenge_seed_derived_after_manifest_observation": True,
        "phase_one_manifest_sha256": phase_one["manifest_sha256"],
        "phase_one_manifest_mode_observed": "0444",
        "challenge_file_mode": "0444",
        "generated_only_after_phase_one_manifest_observation": True,
    }
    challenge_manifest_sha256 = sha256_bytes(canonical_json_bytes(challenge_manifest))
    phase_order_proof = {
        **challenge_manifest,
        "challenge_manifest_sha256": challenge_manifest_sha256,
        "ordering_clock": "time.monotonic_ns",
        "ordering_measurement": "strictly_after",
        "raw_clock_values_omitted_for_deterministic_replay": True,
    }
    custody_events.extend(
        [
            {
                "ordinal": 3,
                "event": "parent_nonce_committed_after_phase_one",
                "evidence_sha256": nonce_commitment_sha256,
            },
            {
                "ordinal": 4,
                "event": "phase_two_challenges_generated_and_bound",
                "evidence_sha256": challenge_manifest_sha256,
            },
        ]
    )

    invocations.extend(
        [
            _writer_record(command, source_tree, "state", sources, state),
            _writer_record(command, source_tree, "motor", sources, motor),
        ]
    )

    public_results: list[dict[str, Any]] = []
    decisive_results: list[dict[str, Any]] = []
    horizons: list[dict[str, Any]] = []
    decoy_sidecar: dict[str, Any] | None = None
    for challenge in (*bundle["public"], *bundle["decisive"]):
        result, sidecar = _challenge_result(
            challenge,
            sources,
            state,
            motor,
            invocations,
            command,
            source_tree,
        )
        horizons.append(sidecar["horizon"])
        if challenge.kind == "public":
            public_results.append(result)
        else:
            decisive_results.append(result)
            if challenge.challenge_id == DECOY_CHALLENGE_ID:
                decoy_sidecar = {
                    "challenge": challenge,
                    "oracle": sidecar["oracle"],
                    "state_terminal": sidecar["state_terminal"],
                }
    if decoy_sidecar is None:
        raise AssertionError("frozen decoy challenge was not generated")
    decoys = _decoy_evidence(
        sources,
        state,
        decoy_sidecar["challenge"],
        decoy_sidecar["oracle"],
        decoy_sidecar["state_terminal"],
        invocations,
        command,
        source_tree,
    )

    role_counts: dict[str, int] = {}
    for invocation in invocations:
        role = invocation["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise AssertionError(f"independent role count drift: {role_counts}")

    permutations = [
        tuple(row["output_permutation"]) for row in (*public_results, *decisive_results)
    ]
    report: dict[str, Any] = {
        "audit": AUDIT_ID,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "code_sha256": identity["scientific_paths"][
            "pipeline/post_commit_packet_transport_falsifier.py"
        ],
        "scientific_identity": copy.deepcopy(dict(identity)),
        "config": {
            "field_modulus": MODULUS,
            "state_dimension": DIMENSION,
            "packet_width_field_elements": PACKET_WIDTH,
            "source_count": SOURCE_COUNT,
            "depths": list(DEPTHS),
            "challenge_nonce_hex": challenge_nonce_hex,
            "challenge_seed": challenge_seed,
            "alternate_challenge_seed": alternate_seed,
        },
        "phase_one": phase_one,
        "phase_two": {
            "challenge_payload_sha256": bundle["payload_sha256"],
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
        "sandbox_probe": expected_sandbox_probe(),
        "deterministic_replay": _pending_replay(),
        "gates": _expected_gates(),
        "pass": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report["payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


@functools.lru_cache(maxsize=8)
def _build_expected_core_cached(
    identity_bytes: bytes, command_bytes: bytes, challenge_nonce_hex: str
) -> bytes:
    identity = json.loads(identity_bytes)
    command_value = json.loads(command_bytes)
    command = CommandContract(
        command_value["sandbox_executable"],
        command_value["sandbox_profile"],
        command_value["python_executable"],
        command_value["role_script"],
    )
    return canonical_json_bytes(
        _build_expected_core_uncached(identity, command, challenge_nonce_hex)
    )


def _expected_core(
    identity: Mapping[str, Any], command: CommandContract, challenge_nonce_hex: str
) -> dict[str, Any]:
    payload = _build_expected_core_cached(
        canonical_json_bytes(dict(identity)),
        canonical_json_bytes(
            {
                "sandbox_executable": command.sandbox_executable,
                "sandbox_profile": command.sandbox_profile,
                "python_executable": command.python_executable,
                "role_script": command.role_script,
            }
        ),
        challenge_nonce_hex,
    )
    return json.loads(payload)


def _first_difference(expected: Any, observed: Any, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return f"{path}: type {type(observed).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return (
                f"{path}: missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
        for key in sorted(expected, key=lambda item: (item == "payload_sha256", item)):
            difference = _first_difference(
                expected[key], observed[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: length {len(observed)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != observed:
        return f"{path}: observed {observed!r} != expected {expected!r}"
    return None


def _pending_core_from_final(report: Mapping[str, Any]) -> dict[str, Any]:
    pending = copy.deepcopy(dict(report))
    pending["deterministic_replay"] = _pending_replay()
    gates = pending.get("gates")
    if not isinstance(gates, dict):
        raise AuditFailure("final report has no gate object")
    gates["full_deterministic_replay_byte_identical"] = False
    pending["pass"] = False
    pending.pop("payload_sha256", None)
    pending["payload_sha256"] = sha256_bytes(canonical_json_bytes(pending))
    return pending


def _verify_payload_hash(report: Mapping[str, Any], label: str) -> None:
    payload = dict(report)
    claimed = payload.pop("payload_sha256", None)
    _require_hex_digest(claimed, f"{label}.payload_sha256")
    actual = sha256_bytes(canonical_json_bytes(payload))
    if claimed != actual:
        raise AuditFailure(f"{label} payload hash mismatch")


def _verify_replay(report: Mapping[str, Any]) -> dict[str, Any]:
    replay = _strict_keys(
        report.get("deterministic_replay"),
        frozenset(
            {
                "status",
                "first_core_payload_sha256",
                "second_core_payload_sha256",
                "second_core_report",
            }
        ),
        "deterministic_replay",
    )
    if replay["status"] != "confirmed_byte_identical":
        raise AuditFailure("deterministic replay status is not confirmed")
    first_hash = _require_hex_digest(
        replay["first_core_payload_sha256"],
        "deterministic_replay.first_core_payload_sha256",
    )
    second_hash = _require_hex_digest(
        replay["second_core_payload_sha256"],
        "deterministic_replay.second_core_payload_sha256",
    )
    if first_hash != second_hash:
        raise AuditFailure("deterministic replay core hashes differ")
    second_core = _strict_keys(
        replay["second_core_report"], REPORT_FIELDS, "second_core_report"
    )
    _verify_payload_hash(second_core, "second_core_report")
    pending = _pending_core_from_final(report)
    if pending["payload_sha256"] != first_hash:
        raise AuditFailure("reconstructed first-core hash is not replay-bound")
    if second_core["payload_sha256"] != second_hash:
        raise AuditFailure("embedded second-core hash is not replay-bound")
    if canonical_json_bytes(pending) != canonical_json_bytes(second_core):
        difference = _first_difference(pending, second_core)
        raise AuditFailure(
            "embedded second core is not byte-identical to reconstructed first: "
            f"{difference}"
        )
    return pending


def verify_evidence(
    report: Mapping[str, Any], *, verify_git: bool = True
) -> dict[str, Any]:
    """Reconstruct every PCPT v3 byte and evidence record.

    Missing nonce, sandbox, sentinel, or replay evidence is fatal.
    """

    report = _strict_keys(report, REPORT_FIELDS, "report")
    if (
        report["audit"] != AUDIT_ID
        or report["protocol_id"] != PROTOCOL_ID
        or report["schema_version"] != SCHEMA_VERSION
    ):
        raise AuditFailure("report protocol identity is invalid")
    _verify_payload_hash(report, "report")
    gates = _strict_keys(report["gates"], GATE_NAMES, "gates")
    if report["pass"] is not True or any(value is not True for value in gates.values()):
        raise AuditFailure("final report does not assert every frozen gate")
    identity = _validate_scientific_identity(
        report["scientific_identity"], verify_git=verify_git
    )
    _validate_sandbox_probe(report["sandbox_probe"])
    command = _validate_command_contract(report)
    challenge_nonce_hex = _validate_nonce_and_phase_bindings(report)
    invocations = report["role_invocations"]
    if len(invocations) != sum(EXPECTED_ROLE_COUNTS.values()):
        raise AuditFailure("role invocation count is not exactly 337")
    for index, invocation in enumerate(invocations):
        _validate_role_record_schema(
            invocation,
            index,
            command,
            str(identity["scientific_source_tree_sha256"]),
        )
    pending = _verify_replay(report)
    expected = _expected_core(identity, command, challenge_nonce_hex)
    difference = _first_difference(expected, pending)
    if difference:
        raise AuditFailure(f"independent evidence reconstruction failed: {difference}")
    return {
        "status": "v3_evidence_reconstructed",
        "protocol_id": PROTOCOL_ID,
        "scientific_commit": identity["scientific_commit"],
        "core_payload_sha256": pending["payload_sha256"],
        "role_invocations_per_core": len(invocations),
        "public_cells": len(report["public_results"]),
        "decisive_cells": len(report["decisive_results"]),
        "evidence_reconstruction_pass": True,
    }


def verify_artifact(
    report: Mapping[str, Any], *, verify_git: bool = True
) -> dict[str, Any]:
    """Verify that an artifact exactly reconstructs under the frozen contract."""

    summary = verify_evidence(report, verify_git=verify_git)
    return {**summary, "status": "evidence_reconstructed"}


def _immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publication_receipt(
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_path: Path,
    receipt_path: Path,
    artifact_bytes: bytes,
    artifact_mode: int,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    verifier_relative = "pipeline/audit_post_commit_packet_transport_v3.py"
    verifier_source_sha256 = sha256_bytes(SCRIPT.read_bytes())
    recorded_verifier_sha256 = report["scientific_identity"]["scientific_paths"][
        verifier_relative
    ]
    if verifier_source_sha256 != recorded_verifier_sha256:
        raise AuditFailure("running verifier source differs from scientific identity")
    receipt = {
        "status": "evidence_reconstructed_and_published",
        "protocol_id": PROTOCOL_ID,
        "scientific_commit": summary["scientific_commit"],
        "core_payload_sha256": summary["core_payload_sha256"],
        "role_invocations_per_core": summary["role_invocations_per_core"],
        "public_cells": summary["public_cells"],
        "decisive_cells": summary["decisive_cells"],
        "evidence_reconstruction_pass": True,
        "git_verification_mode": (
            "required_and_passed" if verify_git else "disabled_for_test_only"
        ),
        "artifact_path": str(artifact_path),
        "artifact_bytes": len(artifact_bytes),
        "artifact_mode": f"{artifact_mode:04o}",
        "artifact_md5": hashlib.md5(artifact_bytes).hexdigest(),
        "artifact_sha256": sha256_bytes(artifact_bytes),
        "artifact_payload_sha256": report["payload_sha256"],
        "receipt_path": str(receipt_path),
        "verifier_source_path": verifier_relative,
        "verifier_source_sha256": verifier_source_sha256,
        "verifier_scientific_path_sha256": recorded_verifier_sha256,
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def verify_published_artifact_and_receipt(
    artifact_path: Path,
    receipt_path: Path,
    *,
    verify_git: bool = True,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Reconstruct an artifact and reject any publication-receipt mutation."""

    artifact_path = artifact_path.resolve()
    receipt_path = receipt_path.resolve()
    if require_canonical and (
        artifact_path != CANONICAL_ARTIFACT.resolve()
        or receipt_path != CANONICAL_RECEIPT.resolve()
    ):
        raise AuditFailure("verification paths are not the frozen canonical paths")
    report = load_json_document(artifact_path)
    receipt = load_json_document(receipt_path)
    _strict_keys(receipt, RECEIPT_FIELDS, "publication receipt")
    artifact_bytes = artifact_path.read_bytes()
    artifact_mode = stat.S_IMODE(artifact_path.stat().st_mode)
    receipt_mode = stat.S_IMODE(receipt_path.stat().st_mode)
    if artifact_mode != 0o444 or receipt_mode != 0o444:
        raise AuditFailure("published artifact or receipt is not mode 0444")
    summary = verify_evidence(report, verify_git=verify_git)
    expected = _publication_receipt(
        report,
        summary,
        artifact_path,
        receipt_path,
        artifact_bytes,
        artifact_mode,
        verify_git=verify_git,
    )
    difference = _first_difference(expected, receipt, "$.publication_receipt")
    if difference:
        raise AuditFailure(f"publication receipt does not replay: {difference}")
    return dict(receipt)


def publish_verified_artifact(
    candidate: Path,
    artifact_path: Path,
    receipt_path: Path,
    *,
    verify_git: bool = True,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Independently reconstruct, then atomically publish artifact and receipt."""

    artifact_path = artifact_path.resolve()
    receipt_path = receipt_path.resolve()
    if require_canonical and (
        artifact_path != CANONICAL_ARTIFACT.resolve()
        or receipt_path != CANONICAL_RECEIPT.resolve()
    ):
        raise AuditFailure("publisher paths are not the frozen canonical paths")
    if artifact_path.exists() or receipt_path.exists():
        raise AuditFailure("canonical artifact or receipt already exists")
    report = load_json_document(candidate)
    summary = verify_evidence(report, verify_git=verify_git)
    artifact_payload = canonical_json_bytes(report)
    _immutable_write(artifact_path, artifact_payload)
    artifact_bytes = artifact_path.read_bytes()
    artifact_stat = artifact_path.stat()
    if artifact_bytes != artifact_payload or stat.S_IMODE(artifact_stat.st_mode) != 0o444:
        raise AuditFailure("published artifact bytes or mode changed after write")
    receipt = _publication_receipt(
        report,
        summary,
        artifact_path,
        receipt_path,
        artifact_bytes,
        stat.S_IMODE(artifact_stat.st_mode),
        verify_git=verify_git,
    )
    _immutable_write(receipt_path, canonical_json_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("artifact", type=Path)
    verify_parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="return the reconstruction summary without the wrapper status",
    )
    verify_parser.add_argument(
        "--no-git",
        action="store_true",
        help="test-only: skip reading scientific files from the recorded Git commit",
    )
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("candidate", type=Path)
    publish_parser.add_argument("--out", type=Path, required=True)
    publish_parser.add_argument("--receipt", type=Path, required=True)
    publication_parser = subparsers.add_parser("verify-publication")
    publication_parser.add_argument("--artifact", type=Path, required=True)
    publication_parser.add_argument("--receipt", type=Path, required=True)
    publication_parser.add_argument(
        "--no-git",
        action="store_true",
        help="test-only: skip reading scientific files from the recorded Git commit",
    )
    args = parser.parse_args()
    try:
        if args.command == "publish":
            result = publish_verified_artifact(
                args.candidate,
                args.out,
                args.receipt,
                verify_git=True,
                require_canonical=True,
            )
        elif args.command == "verify-publication":
            result = verify_published_artifact_and_receipt(
                args.artifact,
                args.receipt,
                verify_git=not args.no_git,
                require_canonical=not args.no_git,
            )
        else:
            report = load_json_document(args.artifact)
            if args.evidence_only:
                result = verify_evidence(report, verify_git=not args.no_git)
            else:
                result = verify_artifact(report, verify_git=not args.no_git)
    except (AuditFailure, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(canonical_json_bytes(result).decode("ascii"), end="")


if __name__ == "__main__":
    main()
