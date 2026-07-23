#!/usr/bin/env python3
"""One-shot authority for a sealed CTAA assessment access.

This process is the only component allowed to turn a precommitted run contract
into an oracle access.  It spends access before launching the assessor and
closes that access only after independently validating the immutable result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ctaa_access_registry import (
    ACCESS_SPEND,
    ASSESSMENT_COMMIT,
    GENESIS_PREVIOUS_HASH,
    append_access_spend,
    append_assessment_commit,
    canonical_json_bytes,
    serialize_head_receipt,
    verify_head_receipt,
    verify_registry,
    verify_registry_events,
)
from ctaa_assessment_source_bundle import (
    AssessmentSourceBundleError,
    load_sealed_assessment_bundle_memfd,
    read_immutable_file,
    validate_assessment_source_bundle,
    validate_sealed_assessment_bundle_fd,
)
from ctaa_process_sandbox import hidden_board_command
from ctaa_run_contract import validate_run_contract
from ctaa_runtime_bundle import read_runtime_bundle_with_replay
from ctaa_runtime_execution_set import (
    RuntimeExecutionSetError,
    read_runtime_execution_set_with_replay,
)
from ctaa_statistical_gate_spec import (
    StatisticalGateBindings,
    StatisticalGateSpecError,
    read_signed_statistical_gate_spec_with_sha,
    write_signed_statistical_gate_spec,
)


CLAIM_SCHEMA = "r12_ctaa_signed_assessment_claim_v5"
ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v2"
ASSESSMENT_ACCESS_SCHEMA = "r12_ctaa_v2_assessment_access_v7"
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class SignedAssessmentError(RuntimeError):
    """The authority could not safely complete the one-shot assessment."""


class ChildAssessmentError(SignedAssessmentError):
    """The sandboxed assessor failed after access was irreversibly spent."""


@dataclass(frozen=True)
class AssessmentAuthorityConfig:
    manifest_path: Path
    run_plan_path: Path
    run_contract_path: Path
    bootstrap_seed_receipt_path: Path
    runtime_bundle_path: Path
    runtime_program_source_path: Path
    runtime_query_source_path: Path
    runtime_tokenizer_path: Path
    runtime_execution_set_path: Path
    assessment_source_bundle_path: Path
    assessment_source_manifest_path: Path
    statistical_gate_spec_path: Path
    python_executable_path: Path
    bwrap_executable_path: Path
    registry_path: Path
    registry_public_key_path: Path
    registry_private_key_path: Path
    previous_head_receipt_path: Path | None
    claim_path: Path
    spend_head_receipt_path: Path
    commit_head_receipt_path: Path
    assessment_output_path: Path
    writable_root: Path
    board_root: Path
    registry_id: str
    access_id: str
    spend_event_id: str
    commit_event_id: str
    partition: str
    timeout_seconds: int = 7200


@dataclass(frozen=True)
class AssessmentAuthorityResult:
    access_id: str
    assessment_sha256: str
    spend_head_entry_hash: str
    commit_head_entry_hash: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SignedAssessmentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
            raise SignedAssessmentError(f"{label} path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SignedAssessmentError(f"{label} path contains a symlink")


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    _reject_symlink_components(path, label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SignedAssessmentError(f"{label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        raise SignedAssessmentError(f"{label} is not a single-link read-only file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SignedAssessmentError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != metadata.st_dev
        or before.st_ino != metadata.st_ino
        or before.st_size != metadata.st_size
        or before.st_mtime_ns != metadata.st_mtime_ns
        or before.st_ctime_ns != metadata.st_ctime_ns
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise SignedAssessmentError(f"{label} changed while being read")
    return b"".join(chunks)


def _decode_json(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SignedAssessmentError(f"non-finite JSON constant in {label}: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SignedAssessmentError(f"{label} JSON differs") from error
    if not isinstance(value, dict):
        raise SignedAssessmentError(f"{label} root is not an object")
    return value


def _read_immutable_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    data = _read_immutable_bytes(path, label)
    return _decode_json(data, label), data


def _write_immutable_once(path: Path, data: bytes, label: str) -> None:
    path = Path(path)
    _reject_symlink_components(path.parent, f"{label} parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing {label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise SignedAssessmentError(f"cannot create {label}") from error
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if _read_immutable_bytes(path, label) != data:
        raise SignedAssessmentError(f"published {label} differs")


def _load_public_key(path: Path) -> bytes:
    raw = _read_immutable_bytes(path, "registry public key")
    if len(raw) == 32:
        key = raw
    else:
        try:
            text = raw.decode("ascii").strip()
            key = bytes.fromhex(text)
        except (UnicodeDecodeError, ValueError) as error:
            raise SignedAssessmentError(
                "registry public key encoding differs"
            ) from error
        if text != key.hex():
            raise SignedAssessmentError("registry public key encoding differs")
    if len(key) != 32:
        raise SignedAssessmentError("registry public key length differs")
    try:
        Ed25519PublicKey.from_public_bytes(key)
    except ValueError as error:
        raise SignedAssessmentError("registry public key differs") from error
    return key


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    path = Path(path)
    if not path.is_absolute():
        raise SignedAssessmentError(
            "registry private key path must be explicit and absolute"
        )
    repository_root = Path(__file__).resolve().parent.parent
    try:
        path.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise SignedAssessmentError(
            "registry private key must be external to the repository"
        )
    raw = _read_immutable_bytes(path, "registry private key")
    try:
        if len(raw) == 32:
            key: object = Ed25519PrivateKey.from_private_bytes(raw)
        else:
            key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as error:
        raise SignedAssessmentError("registry private key encoding differs") from error
    if not isinstance(key, Ed25519PrivateKey):
        raise SignedAssessmentError("registry private key type differs")
    return key


def _raw_public_key(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _require_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise SignedAssessmentError(f"unsafe {label}")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise SignedAssessmentError(f"{label} differs")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return False
    return True


def _preflight_paths(config: AssessmentAuthorityConfig) -> None:
    for value, label in (
        (config.registry_id, "registry_id"),
        (config.access_id, "access_id"),
        (config.spend_event_id, "spend_event_id"),
        (config.commit_event_id, "commit_event_id"),
    ):
        _require_identifier(value, label)
    if config.spend_event_id == config.commit_event_id:
        raise SignedAssessmentError("spend and commit event IDs must differ")
    if config.partition not in {"development", "confirmation"}:
        raise SignedAssessmentError("partition differs")
    if type(config.timeout_seconds) is not int or config.timeout_seconds < 1:
        raise SignedAssessmentError("timeout differs")

    writable = Path(os.path.abspath(config.writable_root))
    board = Path(os.path.abspath(config.board_root))
    _reject_symlink_components(writable, "writable root")
    _reject_symlink_components(board, "board root")
    try:
        writable_meta = writable.lstat()
        board_meta = board.lstat()
    except OSError as error:
        raise SignedAssessmentError("sandbox root is unavailable") from error
    if not stat.S_ISDIR(writable_meta.st_mode) or not stat.S_ISDIR(board_meta.st_mode):
        raise SignedAssessmentError("sandbox root is not a directory")
    if writable == board or _is_within(writable, board) or _is_within(board, writable):
        raise SignedAssessmentError("sandbox roots overlap")
    if _is_within(config.runtime_bundle_path, board):
        raise SignedAssessmentError("runtime bundle cannot be inside sealed board root")
    for path, label in (
        (config.runtime_program_source_path, "runtime program source"),
        (config.runtime_query_source_path, "runtime query source"),
    ):
        if not _is_within(path, board):
            raise SignedAssessmentError(f"{label} is outside sealed board root")
    output = Path(os.path.abspath(config.assessment_output_path))
    if Path(os.path.abspath(output.parent)) != writable:
        raise SignedAssessmentError(
            "assessment output must be directly inside writable root"
        )

    outputs = {
        Path(os.path.abspath(config.claim_path)),
        Path(os.path.abspath(config.spend_head_receipt_path)),
        Path(os.path.abspath(config.commit_head_receipt_path)),
        Path(os.path.abspath(config.statistical_gate_spec_path)),
        output,
    }
    if len(outputs) != 5:
        raise SignedAssessmentError("authority output paths overlap")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing authority output: {path.name}")
    protected = (
        config.manifest_path,
        config.run_plan_path,
        config.run_contract_path,
        config.bootstrap_seed_receipt_path,
        config.runtime_bundle_path,
        config.runtime_program_source_path,
        config.runtime_query_source_path,
        config.runtime_tokenizer_path,
        config.runtime_execution_set_path,
        config.assessment_source_bundle_path,
        config.assessment_source_manifest_path,
        config.statistical_gate_spec_path,
        config.python_executable_path,
        config.bwrap_executable_path,
        config.registry_path,
        config.registry_public_key_path,
        config.registry_private_key_path,
        config.previous_head_receipt_path,
        config.claim_path,
        config.spend_head_receipt_path,
        config.commit_head_receipt_path,
    )
    if any(path is not None and _is_within(path, writable) for path in protected):
        raise SignedAssessmentError(
            "authority custody path is inside assessor writable root"
        )


def _plan_rows(
    plan: Mapping[str, object],
) -> dict[tuple[int, str, str], dict[str, object]]:
    rows = plan.get("runs")
    if not isinstance(rows, list):
        raise SignedAssessmentError("run plan rows differ")
    indexed: dict[tuple[int, str, str], dict[str, object]] = {}
    for value in rows:
        if not isinstance(value, dict):
            raise SignedAssessmentError("run plan entry differs")
        seed, arm, dataset = value.get("seed"), value.get("arm"), value.get("dataset")
        if (
            type(seed) is not int
            or not isinstance(arm, str)
            or not isinstance(dataset, str)
        ):
            raise SignedAssessmentError("run plan identity differs")
        identity = (seed, arm, dataset)
        if identity in indexed:
            raise SignedAssessmentError("run plan identity repeats")
        indexed[identity] = value
    return indexed


def _receipt_directory(value: object, plan_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SignedAssessmentError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute():
        path = plan_root / path
    if path.name != "receipt.json":
        raise SignedAssessmentError(f"{label} is not a receipt path")
    return Path(os.path.abspath(path.parent))


def _derive_assessor_arguments(
    *,
    config: AssessmentAuthorityConfig,
    contract: Mapping[str, object],
    plan: Mapping[str, object],
    assessor_bundle_descriptor: int,
) -> tuple[list[str], tuple[Path, ...]]:
    if (
        plan.get("partition") != config.partition
        or contract.get("partition") != config.partition
    ):
        raise SignedAssessmentError("partition differs from run commitment")
    indexed = _plan_rows(plan)
    contract_runs = contract.get("runs")
    oracle_files = contract.get("oracle_files")
    if not isinstance(contract_runs, list) or not isinstance(oracle_files, dict):
        raise SignedAssessmentError("run contract structure differs")
    board = Path(os.path.abspath(config.board_root))
    manifest = Path(os.path.abspath(config.manifest_path))
    if Path(os.path.abspath(manifest.parent)) != board:
        raise SignedAssessmentError("manifest is not directly inside sealed board root")

    command = [
        str(Path(os.path.abspath(config.python_executable_path))),
        f"/proc/self/fd/{assessor_bundle_descriptor}",
        "--manifest",
        str(manifest),
        "--run-plan",
        str(Path(os.path.abspath(config.run_plan_path))),
        "--run-contract",
        str(Path(os.path.abspath(config.run_contract_path))),
        "--bootstrap-seed-receipt",
        str(Path(os.path.abspath(config.bootstrap_seed_receipt_path))),
        "--runtime-bundle",
        str(Path(os.path.abspath(config.runtime_bundle_path))),
        "--runtime-program-source",
        str(Path(os.path.abspath(config.runtime_program_source_path))),
        "--runtime-query-source",
        str(Path(os.path.abspath(config.runtime_query_source_path))),
        "--runtime-tokenizer",
        str(Path(os.path.abspath(config.runtime_tokenizer_path))),
        "--runtime-execution-set",
        str(Path(os.path.abspath(config.runtime_execution_set_path))),
        "--statistical-gate-spec",
        str(Path(os.path.abspath(config.statistical_gate_spec_path))),
        "--access-registry",
        str(Path(os.path.abspath(config.registry_path))),
        "--access-head-receipt",
        str(Path(os.path.abspath(config.spend_head_receipt_path))),
        "--registry-public-key",
        str(Path(os.path.abspath(config.registry_public_key_path))),
        "--partition",
        config.partition,
    ]
    disclosed = {
        manifest,
        Path(os.path.abspath(config.runtime_program_source_path)),
        Path(os.path.abspath(config.runtime_query_source_path)),
    }
    if _is_within(config.runtime_tokenizer_path, board):
        disclosed.add(Path(os.path.abspath(config.runtime_tokenizer_path)))
    seen_names: set[str] = set()
    for value in contract_runs:
        if not isinstance(value, dict):
            raise SignedAssessmentError("run contract entry differs")
        seed, arm, dataset = value.get("seed"), value.get("arm"), value.get("dataset")
        identity = (seed, arm, dataset)
        if type(seed) is not int or identity not in indexed:
            raise SignedAssessmentError("run contract identity differs from run plan")
        expected_name = f"seed-{seed}:{arm}:{dataset}"
        if value.get("run_id") != expected_name or expected_name in seen_names:
            raise SignedAssessmentError("run contract run_id differs")
        seen_names.add(expected_name)
        source = value.get("sealed_sources")
        oracle = oracle_files.get(dataset) if isinstance(dataset, str) else None
        if not isinstance(source, dict) or not isinstance(oracle, dict):
            raise SignedAssessmentError("run oracle binding differs")
        filename = source.get("oracle_filename")
        if (
            not isinstance(filename, str)
            or filename != oracle.get("filename")
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise SignedAssessmentError("run oracle filename differs")
        oracle_path = board / filename
        disclosed.add(oracle_path)
        plan_row = indexed[identity]
        evidence = _receipt_directory(
            plan_row.get("evidence_receipt_path"),
            Path(config.run_plan_path).parent,
            "evidence receipt",
        )
        command.extend(
            [
                "--run",
                expected_name,
                str(seed),
                str(arm),
                str(dataset),
                str(evidence),
                str(oracle_path),
            ]
        )
        parent = plan_row.get("parent_evidence_receipt_path")
        if dataset == "intervention":
            parent_dir = _receipt_directory(
                parent,
                Path(config.run_plan_path).parent,
                "parent evidence receipt",
            )
            command.extend(["--parent-evidence", expected_name, str(parent_dir)])
        elif parent is not None:
            raise SignedAssessmentError("base run unexpectedly names parent evidence")
    if set(indexed) != {
        (row.get("seed"), row.get("arm"), row.get("dataset"))
        for row in contract_runs
        if isinstance(row, dict)
    }:
        raise SignedAssessmentError("run plan contains uncontracted entries")
    command.extend(
        ["--output", str(Path(os.path.abspath(config.assessment_output_path)))]
    )
    return command, tuple(sorted(disclosed, key=str))


def _sandboxed_command(
    command: list[str],
    *,
    writable_root: Path,
    board_root: Path,
    disclosed_board_files: tuple[Path, ...],
    hidden_authority_files: tuple[Path, ...],
    expected_bwrap_path: Path,
) -> list[str]:
    wrapped = hidden_board_command(
        command,
        writable_root=Path(os.path.abspath(writable_root)),
        board_root=Path(os.path.abspath(board_root)),
    )
    expected_bwrap = Path(os.path.abspath(expected_bwrap_path))
    if (
        not wrapped
        or Path(wrapped[0]).name != "bwrap"
        or Path(os.path.abspath(wrapped[0])) != expected_bwrap
    ):
        raise SignedAssessmentError(
            "selective CTAA oracle disclosure requires bubblewrap"
        )
    try:
        separator = wrapped.index("--")
    except ValueError as error:
        raise SignedAssessmentError("physical sandbox command differs") from error
    board = Path(os.path.abspath(board_root))
    bindings: list[str] = []
    for path in disclosed_board_files:
        absolute = Path(os.path.abspath(path))
        if not _is_within(absolute, board) or absolute == board:
            raise SignedAssessmentError("disclosed board path escapes sealed root")
        bindings.extend(["--ro-bind", str(absolute), str(absolute)])
    for path in hidden_authority_files:
        absolute = Path(os.path.abspath(path))
        if _is_within(absolute, board) or _is_within(absolute, writable_root):
            raise SignedAssessmentError("hidden authority path overlaps sandbox roots")
        bindings.extend(["--ro-bind", "/dev/null", str(absolute)])
    return [*wrapped[:separator], *bindings, *wrapped[separator:]]


def _signed_claim(
    *,
    config: AssessmentAuthorityConfig,
    signing_key: Ed25519PrivateKey,
    public_key: bytes,
    contract: Mapping[str, object],
    command: list[str],
    expected_previous_hash: str,
    runtime_bundle_sha256: str,
    execution_set_file_sha256: str,
    execution_set_sha256: str,
    statistical_gate_spec_file_sha256: str,
    gate_spec_sha256: str,
    assessment_source_manifest: Mapping[str, object],
    assessment_source_manifest_sha256: str,
) -> bytes:
    payload = {
        "schema": CLAIM_SCHEMA,
        "registry_id": config.registry_id,
        "access_id": config.access_id,
        "spend_event_id": config.spend_event_id,
        "commit_event_id": config.commit_event_id,
        "partition": config.partition,
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_plan_sha256": contract["run_plan_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "execution_set_file_sha256": execution_set_file_sha256,
        "execution_set_sha256": execution_set_sha256,
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
        "assessment_source_bundle_sha256": assessment_source_manifest["bundle_sha256"],
        "assessment_source_manifest_sha256": assessment_source_manifest_sha256,
        "python_interpreter_sha256": assessment_source_manifest["python_interpreter"][
            "sha256"
        ],
        "bwrap_executable_sha256": assessment_source_manifest["bwrap_executable"][
            "sha256"
        ],
        "expected_previous_hash": expected_previous_hash,
        "assessment_output": str(Path(os.path.abspath(config.assessment_output_path))),
        "assessor_argv_sha256": hashlib.sha256(
            canonical_json_bytes(command)
        ).hexdigest(),
        "signing_public_key": public_key.hex(),
    }
    signature = signing_key.sign(canonical_json_bytes(payload)).hex()
    return canonical_json_bytes({"payload": payload, "signature": signature}) + b"\n"


def _validate_assessment(
    *,
    path: Path,
    config: AssessmentAuthorityConfig,
    contract: Mapping[str, object],
    spend_receipt: Mapping[str, object],
    public_key: bytes,
    runtime_bundle_sha256: str,
    execution_set_file_sha256: str,
    execution_set_sha256: str,
    statistical_gate_spec_file_sha256: str,
    gate_spec_sha256: str,
) -> str:
    report, raw = _read_immutable_json(path, "assessment output")
    expected_top = {
        "schema",
        "partition",
        "manifest_sha256",
        "access",
        "oracle_sha256",
        "runs",
        "capability_gate_computed",
    }
    if set(report) != expected_top or report.get("schema") != ASSESSMENT_SCHEMA:
        raise SignedAssessmentError("assessment output schema differs")
    if (
        report.get("partition") != config.partition
        or report.get("manifest_sha256") != contract.get("manifest_sha256")
        or report.get("capability_gate_computed") is not False
    ):
        raise SignedAssessmentError("assessment output commitment differs")

    spend_state = verify_registry(
        config.registry_path,
        public_key,
        expected_head_receipt=dict(spend_receipt),
    )
    spend_events = verify_registry_events(
        config.registry_path,
        public_key,
        expected_head_receipt=dict(spend_receipt),
    )
    spend_event = spend_events[-1]
    access = report.get("access")
    expected_access = {
        "schema": ASSESSMENT_ACCESS_SCHEMA,
        "registry_id": config.registry_id,
        "registry_head_receipt_sha256": hashlib.sha256(
            _read_immutable_bytes(config.spend_head_receipt_path, "spend head receipt")
        ).hexdigest(),
        "registry_head_entry_hash": spend_state.head_hash,
        "access_event_payload_sha256": hashlib.sha256(
            spend_event.canonical_payload
        ).hexdigest(),
        "access_id": config.access_id,
        "partition": config.partition,
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "assessment_claim_sha256": hashlib.sha256(
            _read_immutable_bytes(config.claim_path, "access claim")
        ).hexdigest(),
        "execution_set_file_sha256": execution_set_file_sha256,
        "execution_set_sha256": execution_set_sha256,
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
        "access": 1,
    }
    if access != expected_access:
        raise SignedAssessmentError("assessment signed-access binding differs")
    runs = report.get("runs")
    oracle_hashes = report.get("oracle_sha256")
    contract_runs = contract.get("runs")
    oracle_files = contract.get("oracle_files")
    if (
        not isinstance(runs, dict)
        or not isinstance(oracle_hashes, dict)
        or not isinstance(contract_runs, list)
        or not isinstance(oracle_files, dict)
    ):
        raise SignedAssessmentError("assessment run result structure differs")
    expected_names = {
        str(row["run_id"]) for row in contract_runs if isinstance(row, dict)
    }
    if set(runs) != expected_names or set(oracle_hashes) != expected_names:
        raise SignedAssessmentError("assessment run set differs")
    for row in contract_runs:
        if not isinstance(row, dict):
            raise SignedAssessmentError("assessment contract run differs")
        oracle = oracle_files.get(row.get("dataset"))
        if not isinstance(oracle, dict) or oracle_hashes.get(
            row.get("run_id")
        ) != oracle.get("sha256"):
            raise SignedAssessmentError("assessment oracle hash differs")
    return hashlib.sha256(raw).hexdigest()


def _load_sealed_assessor(
    config: AssessmentAuthorityConfig,
) -> tuple[int, dict[str, object], str]:
    descriptor = -1
    try:
        descriptor = load_sealed_assessment_bundle_memfd(
            source_root=Path(__file__).resolve().parent,
            bundle_path=config.assessment_source_bundle_path,
            manifest_path=config.assessment_source_manifest_path,
            python_executable=config.python_executable_path,
            bwrap_executable=config.bwrap_executable_path,
        )
        manifest_raw = read_immutable_file(
            config.assessment_source_manifest_path,
            "assessment source manifest",
        )
        manifest = _decode_json(manifest_raw, "assessment source manifest")
        validated = validate_assessment_source_bundle(
            source_root=Path(__file__).resolve().parent,
            bundle_path=config.assessment_source_bundle_path,
            manifest_path=config.assessment_source_manifest_path,
            python_executable=config.python_executable_path,
            bwrap_executable=config.bwrap_executable_path,
        )
        if manifest != validated:
            raise SignedAssessmentError("assessment source manifest changed")
        bundle_sha256 = _require_hash(
            validated.get("bundle_sha256"), "assessment source bundle SHA-256"
        )
        validate_sealed_assessment_bundle_fd(descriptor, bundle_sha256)
        return descriptor, validated, hashlib.sha256(manifest_raw).hexdigest()
    except (AssessmentSourceBundleError, OSError, ValueError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, SignedAssessmentError):
            raise
        raise SignedAssessmentError("sealed assessor source bundle differs") from error


def run_signed_assessment(
    config: AssessmentAuthorityConfig,
) -> AssessmentAuthorityResult:
    """Spend one access, run one sandboxed assessor, and close on success."""

    _preflight_paths(config)
    manifest_raw = _read_immutable_bytes(config.manifest_path, "board manifest")
    plan, plan_raw = _read_immutable_json(config.run_plan_path, "run plan")
    published_contract, _ = _read_immutable_json(
        config.run_contract_path, "run contract"
    )
    bootstrap_raw = _read_immutable_bytes(
        config.bootstrap_seed_receipt_path, "bootstrap seed receipt"
    )
    public_key = _load_public_key(config.registry_public_key_path)
    signing_key = _load_private_key(config.registry_private_key_path)
    if _raw_public_key(signing_key) != public_key:
        raise SignedAssessmentError("registry signing and verification keys differ")

    contract = validate_run_contract(
        contract_path=config.run_contract_path,
        manifest_path=config.manifest_path,
        run_plan_path=config.run_plan_path,
        bootstrap_seed_receipt_path=config.bootstrap_seed_receipt_path,
    )
    if contract != published_contract:
        raise SignedAssessmentError(
            "validated run contract differs from published bytes"
        )
    if hashlib.sha256(manifest_raw).hexdigest() != contract.get(
        "manifest_sha256"
    ) or hashlib.sha256(plan_raw).hexdigest() != contract.get("run_plan_sha256"):
        raise SignedAssessmentError("manifest or run-plan hash differs from contract")
    for key in (
        "manifest_sha256",
        "board_sha256",
        "run_plan_sha256",
        "run_contract_sha256",
        "bootstrap_seed_receipt_sha256",
    ):
        _require_hash(contract.get(key), key)
    if (
        type(contract.get("bootstrap_seed")) is not int
        or int(contract["bootstrap_seed"]) < 0
    ):
        raise SignedAssessmentError("bootstrap seed differs")
    if hashlib.sha256(bootstrap_raw).hexdigest() != contract.get(
        "bootstrap_seed_receipt_sha256"
    ):
        raise SignedAssessmentError("bootstrap receipt hash differs from contract")

    # This is the irreversible query-release gate. No query source is opened,
    # replayed, or disclosed until every seed's signed execution artifacts have
    # been replayed and joined bijectively to the five-member runtime bundle.
    try:
        execution_set, execution_set_file_sha256 = (
            read_runtime_execution_set_with_replay(
                config.runtime_execution_set_path,
                runtime_bundle_path=config.runtime_bundle_path,
                run_contract=contract,
                verification_key=public_key,
            )
        )
    except (RuntimeExecutionSetError, OSError, ValueError) as error:
        raise SignedAssessmentError(
            "pre-query execution set failed before query release"
        ) from error
    execution_set_sha256 = execution_set.get("execution_set_sha256")
    if execution_set.get("partition") != config.partition or execution_set.get(
        "run_contract_sha256"
    ) != contract.get("run_contract_sha256"):
        raise SignedAssessmentError("pre-query execution set binding differs")
    execution_set_file_sha256 = _require_hash(
        execution_set_file_sha256, "execution set file SHA-256"
    )
    execution_set_sha256 = _require_hash(execution_set_sha256, "execution set SHA-256")

    runtime_bundle_raw = _read_immutable_bytes(
        config.runtime_bundle_path, "runtime bundle"
    )
    runtime_bundle_sha256 = hashlib.sha256(runtime_bundle_raw).hexdigest()
    if execution_set.get("runtime_bundle_file_sha256") != runtime_bundle_sha256:
        raise SignedAssessmentError("execution set runtime-bundle file binding differs")
    runtime_bundle = read_runtime_bundle_with_replay(
        config.runtime_bundle_path,
        run_contract=contract,
        program_path=config.runtime_program_source_path,
        query_path=config.runtime_query_source_path,
        tokenizer_path=config.runtime_tokenizer_path,
    )
    runtime_bundle_logical_sha256 = _require_hash(
        runtime_bundle.get("bundle_sha256"), "runtime bundle logical SHA-256"
    )
    if (
        hashlib.sha256(
            _read_immutable_bytes(config.runtime_bundle_path, "runtime bundle")
        ).hexdigest()
        != runtime_bundle_sha256
    ):
        raise SignedAssessmentError("runtime bundle changed during preflight")

    previous_receipt: dict[str, object] | None = None
    if config.registry_path.exists() or config.registry_path.is_symlink():
        if (
            config.registry_path.is_symlink()
            or config.previous_head_receipt_path is None
        ):
            raise SignedAssessmentError(
                "existing registry requires an exact retained head receipt"
            )
        previous_receipt, _ = _read_immutable_json(
            config.previous_head_receipt_path, "previous head receipt"
        )
        verify_head_receipt(previous_receipt, public_key)
        previous_state = verify_registry(
            config.registry_path,
            public_key,
            expected_head_receipt=previous_receipt,
        )
        if previous_state.registry_id != config.registry_id:
            raise SignedAssessmentError("registry identifier differs")
        if previous_state.open_access_id is not None:
            raise SignedAssessmentError("registry already has an open access")
        expected_previous_hash = previous_state.head_hash
    else:
        if config.previous_head_receipt_path is not None:
            raise SignedAssessmentError("head receipt supplied for missing registry")
        expected_previous_hash = GENESIS_PREVIOUS_HASH

    assessor_descriptor, source_manifest, source_manifest_sha256 = (
        _load_sealed_assessor(config)
    )
    try:
        training_seeds = contract.get("training_seeds")
        if (
            not isinstance(training_seeds, list)
            or len(training_seeds) != 5
            or any(type(seed) is not int or seed < 0 for seed in training_seeds)
        ):
            raise SignedAssessmentError(
                "statistical gate requires exactly five ordered training seeds"
            )
        gate_bindings = StatisticalGateBindings(
            manifest_sha256=str(contract["manifest_sha256"]),
            board_sha256=str(contract["board_sha256"]),
            run_plan_sha256=str(contract["run_plan_sha256"]),
            run_contract_sha256=str(contract["run_contract_sha256"]),
            runtime_bundle_file_sha256=runtime_bundle_sha256,
            runtime_bundle_sha256=runtime_bundle_logical_sha256,
            runtime_execution_set_file_sha256=execution_set_file_sha256,
            runtime_execution_set_sha256=execution_set_sha256,
            assessment_source_bundle_sha256=str(source_manifest["bundle_sha256"]),
            assessment_source_manifest_sha256=source_manifest_sha256,
            bootstrap_seed_receipt_sha256=str(
                contract["bootstrap_seed_receipt_sha256"]
            ),
            bootstrap_seed=int(contract["bootstrap_seed"]),
            training_seeds=tuple(training_seeds),
        )
        try:
            written_spec_file_sha256 = write_signed_statistical_gate_spec(
                config.statistical_gate_spec_path,
                bindings=gate_bindings,
                signing_key=signing_key,
            )
            statistical_gate_spec, statistical_gate_spec_file_sha256 = (
                read_signed_statistical_gate_spec_with_sha(
                    config.statistical_gate_spec_path,
                    verification_key=public_key,
                    expected_bindings=gate_bindings,
                )
            )
        except (StatisticalGateSpecError, OSError, ValueError) as error:
            raise SignedAssessmentError(
                "signed statistical gate specification failed before access spend"
            ) from error
        if written_spec_file_sha256 != statistical_gate_spec_file_sha256:
            raise SignedAssessmentError(
                "published statistical gate specification changed"
            )
        gate_spec_sha256 = _require_hash(
            statistical_gate_spec.get("gate_spec_sha256"),
            "statistical gate specification logical SHA-256",
        )
        assessor_command, disclosed = _derive_assessor_arguments(
            config=config,
            contract=contract,
            plan=plan,
            assessor_bundle_descriptor=assessor_descriptor,
        )
        sandboxed = _sandboxed_command(
            assessor_command,
            writable_root=config.writable_root,
            board_root=config.board_root,
            disclosed_board_files=disclosed,
            hidden_authority_files=(config.registry_private_key_path,),
            expected_bwrap_path=config.bwrap_executable_path,
        )
        claim = _signed_claim(
            config=config,
            signing_key=signing_key,
            public_key=public_key,
            contract=contract,
            command=sandboxed,
            expected_previous_hash=expected_previous_hash,
            runtime_bundle_sha256=runtime_bundle_sha256,
            execution_set_file_sha256=execution_set_file_sha256,
            execution_set_sha256=execution_set_sha256,
            statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
            gate_spec_sha256=gate_spec_sha256,
            assessment_source_manifest=source_manifest,
            assessment_source_manifest_sha256=source_manifest_sha256,
        )
        _write_immutable_once(config.claim_path, claim, "access claim")
        assessment_claim_sha256 = hashlib.sha256(claim).hexdigest()

        spend_receipt = append_access_spend(
            config.registry_path,
            signing_key=signing_key,
            registry_id=config.registry_id,
            event_id=config.spend_event_id,
            access_id=config.access_id,
            partition=config.partition,
            manifest_sha256=str(contract["manifest_sha256"]),
            board_sha256=str(contract["board_sha256"]),
            run_contract_sha256=str(contract["run_contract_sha256"]),
            runtime_bundle_sha256=runtime_bundle_sha256,
            assessment_claim_sha256=assessment_claim_sha256,
            bootstrap_seed_receipt_sha256=str(
                contract["bootstrap_seed_receipt_sha256"]
            ),
            bootstrap_seed=int(contract["bootstrap_seed"]),
            statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
            gate_spec_sha256=gate_spec_sha256,
            expected_previous_hash=expected_previous_hash,
            expected_head_receipt=previous_receipt,
        )
        _write_immutable_once(
            config.spend_head_receipt_path,
            serialize_head_receipt(spend_receipt),
            "spend head receipt",
        )
        spend_state = verify_registry(
            config.registry_path,
            public_key,
            expected_head_receipt=spend_receipt,
        )
        if (
            spend_state.head_event_type != ACCESS_SPEND
            or spend_state.open_access_id != config.access_id
        ):
            raise SignedAssessmentError(
                "access spend did not open the expected registry state"
            )

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
        }
        try:
            child = subprocess.run(
                sandboxed,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                close_fds=True,
                pass_fds=(assessor_descriptor,),
                cwd=Path(__file__).resolve().parent,
                env=environment,
                timeout=config.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ChildAssessmentError(
                "sandboxed assessor failed after access spend; registry remains open"
            ) from error
    finally:
        os.close(assessor_descriptor)
    if child.returncode != 0:
        raise ChildAssessmentError(
            "sandboxed assessor exited unsuccessfully after access spend; registry remains open"
        )

    assessment_sha = _validate_assessment(
        path=config.assessment_output_path,
        config=config,
        contract=contract,
        spend_receipt=spend_receipt,
        public_key=public_key,
        runtime_bundle_sha256=runtime_bundle_sha256,
        execution_set_file_sha256=execution_set_file_sha256,
        execution_set_sha256=execution_set_sha256,
        statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
        gate_spec_sha256=gate_spec_sha256,
    )
    # Recheck the exact spend head immediately before the irreversible close.
    verify_registry(
        config.registry_path,
        public_key,
        expected_head_receipt=spend_receipt,
    )
    commit_receipt = append_assessment_commit(
        config.registry_path,
        signing_key=signing_key,
        registry_id=config.registry_id,
        event_id=config.commit_event_id,
        access_id=config.access_id,
        assessment_sha256=assessment_sha,
        statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
        gate_spec_sha256=gate_spec_sha256,
        expected_previous_hash=spend_state.head_hash,
        expected_head_receipt=spend_receipt,
    )
    _write_immutable_once(
        config.commit_head_receipt_path,
        serialize_head_receipt(commit_receipt),
        "assessment commit head receipt",
    )
    final_state = verify_registry(
        config.registry_path,
        public_key,
        expected_head_receipt=commit_receipt,
    )
    final_event = verify_registry_events(
        config.registry_path,
        public_key,
        expected_head_receipt=commit_receipt,
    )[-1]
    if (
        final_state.head_event_type != ASSESSMENT_COMMIT
        or final_state.open_access_id is not None
        or final_event.payload.get("access_id") != config.access_id
        or final_event.payload.get("assessment_sha256") != assessment_sha
        or final_event.payload.get("statistical_gate_spec_file_sha256")
        != statistical_gate_spec_file_sha256
        or final_event.payload.get("gate_spec_sha256") != gate_spec_sha256
    ):
        raise SignedAssessmentError("assessment commit verification differs")
    return AssessmentAuthorityResult(
        access_id=config.access_id,
        assessment_sha256=assessment_sha,
        spend_head_entry_hash=spend_state.head_hash,
        commit_head_entry_hash=final_state.head_hash,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--bootstrap-seed-receipt", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--runtime-program-source", type=Path, required=True)
    parser.add_argument("--runtime-query-source", type=Path, required=True)
    parser.add_argument("--runtime-tokenizer", type=Path, required=True)
    parser.add_argument("--runtime-execution-set", type=Path, required=True)
    parser.add_argument("--assessment-source-bundle", type=Path, required=True)
    parser.add_argument("--assessment-source-manifest", type=Path, required=True)
    parser.add_argument("--statistical-gate-spec", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--bwrap-executable", type=Path, required=True)
    parser.add_argument("--access-registry", type=Path, required=True)
    parser.add_argument("--registry-public-key", type=Path, required=True)
    parser.add_argument("--registry-private-key", type=Path, required=True)
    parser.add_argument("--previous-head-receipt", type=Path)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--spend-head-receipt", type=Path, required=True)
    parser.add_argument("--commit-head-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--writable-root", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--access-id", required=True)
    parser.add_argument("--spend-event-id", required=True)
    parser.add_argument("--commit-event-id", required=True)
    parser.add_argument(
        "--partition", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = AssessmentAuthorityConfig(
        manifest_path=args.manifest,
        run_plan_path=args.run_plan,
        run_contract_path=args.run_contract,
        bootstrap_seed_receipt_path=args.bootstrap_seed_receipt,
        runtime_bundle_path=args.runtime_bundle,
        runtime_program_source_path=args.runtime_program_source,
        runtime_query_source_path=args.runtime_query_source,
        runtime_tokenizer_path=args.runtime_tokenizer,
        runtime_execution_set_path=args.runtime_execution_set,
        assessment_source_bundle_path=args.assessment_source_bundle,
        assessment_source_manifest_path=args.assessment_source_manifest,
        statistical_gate_spec_path=args.statistical_gate_spec,
        python_executable_path=args.python_executable,
        bwrap_executable_path=args.bwrap_executable,
        registry_path=args.access_registry,
        registry_public_key_path=args.registry_public_key,
        registry_private_key_path=args.registry_private_key,
        previous_head_receipt_path=args.previous_head_receipt,
        claim_path=args.claim,
        spend_head_receipt_path=args.spend_head_receipt,
        commit_head_receipt_path=args.commit_head_receipt,
        assessment_output_path=args.output,
        writable_root=args.writable_root,
        board_root=args.board_root,
        registry_id=args.registry_id,
        access_id=args.access_id,
        spend_event_id=args.spend_event_id,
        commit_event_id=args.commit_event_id,
        partition=args.partition,
        timeout_seconds=args.timeout_seconds,
    )
    result = run_signed_assessment(config)
    print(
        json.dumps(
            {
                "access_id": result.access_id,
                "assessment_sha256": result.assessment_sha256,
                "spend_head_entry_hash": result.spend_head_entry_hash,
                "commit_head_entry_hash": result.commit_head_entry_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
