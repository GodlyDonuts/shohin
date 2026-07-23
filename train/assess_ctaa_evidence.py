#!/usr/bin/env python3
"""Spend one sealed-board access and score committed CTAA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ctaa_access_registry import (
    ACCESS_SPEND,
    verify_registry_events,
)
from ctaa_assessment import (
    EVIDENCE_KEYS,
    load_oracle,
    score_evidence,
)
from ctaa_core_training import ARMS
from ctaa_evaluation_io import write_json_once
from ctaa_intervention_protocol import RuntimeInterventionPlan
from ctaa_run_contract import validate_run_contract
from ctaa_runtime_bundle import (
    ATTEMPT_COUNT_PER_SEED,
    RuntimeBundleError,
    make_runtime_bundle,
    read_runtime_plan,
    validate_runtime_bundle,
)
from ctaa_runtime_evidence import read_runtime_evidence
from ctaa_runtime_execution_set import (
    RuntimeExecutionSetError,
    read_runtime_execution_set_with_replay,
)
from ctaa_runtime_plan_replay import load_runtime_replay_rows, replay_runtime_plan
from ctaa_statistical_gate_spec import (
    StatisticalGateSpecError,
    read_signed_statistical_gate_spec_with_sha,
)
from commit_ctaa_raw_evidence import (
    RAW_EVIDENCE_RECEIPT_SCHEMA,
    RAW_EVIDENCE_SCHEMA,
)


ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v3"
ASSESSMENT_ACCESS_SCHEMA = "r12_ctaa_v2_assessment_access_v7"
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _load_manifest(path: Path) -> tuple[dict[str, object], str]:
    value, digest = _load_read_only_json_with_sha(path, "board manifest")
    if (
        not isinstance(value, dict)
        or value.get("schema") != "r12_ctaa_v2_manifest_v2"
        or not isinstance(value.get("files"), dict)
    ):
        raise ValueError("CTAA assessment board manifest differs")
    return value, digest


def _verify_board_path_declared(
    path: Path,
    manifest_path: Path,
    manifest: dict[str, object],
) -> None:
    if Path(os.path.abspath(path.parent)) != Path(
        os.path.abspath(manifest_path.parent)
    ) or not isinstance(manifest["files"].get(path.name), str):
        raise ValueError("CTAA assessed file is outside the sealed board")


def _verify_evidence_source_binding(
    commitment: dict[str, object],
    oracle_path: Path,
    manifest: dict[str, object],
) -> None:
    suffix = "_oracle.jsonl"
    if not oracle_path.name.endswith(suffix):
        raise ValueError("CTAA oracle filename cannot derive source commitments")
    prefix = oracle_path.name[: -len(suffix)]
    program_name = prefix + "_program.jsonl"
    query_name = prefix + "_query.jsonl"
    files = manifest["files"]
    query_bound = commitment.get("query_source_sha256") == files.get(query_name)
    query_never_disclosed = (
        commitment.get("query_source_sha256") is None
        and commitment.get("executed_rows") == 0
        and commitment.get("queried_rows") == 0
        and commitment.get("answered_rows") == 0
    )
    if (
        not isinstance(files.get(program_name), str)
        or not isinstance(files.get(query_name), str)
        or commitment.get("program_source_sha256") != files[program_name]
        or not (query_bound or query_never_disclosed)
    ):
        raise ValueError("CTAA evidence is not bound to sealed program/query sources")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"CTAA assessment duplicate JSON key: {key}")
        result[key] = value
    return result


def _open_parent_directory(path: Path, label: str) -> tuple[int, str]:
    raw = os.path.abspath(os.fspath(path))
    if "\x00" in raw or raw == "/":
        raise ValueError(f"CTAA assessment {label} path differs")
    components = raw.split("/")[1:]
    if any(component in ("", ".", "..") for component in components):
        raise ValueError(f"CTAA assessment {label} path differs")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open("/", flags)
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"CTAA assessment {label} parent is missing or symlinked"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ValueError(f"CTAA assessment {label} parent is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, components[-1]
    except Exception:
        os.close(descriptor)
        raise


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    parent_descriptor, name = _open_parent_directory(path, label)
    descriptor = -1
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} is not a single-link immutable file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} cannot be opened safely") from error
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
        os.close(parent_descriptor)
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
        raise ValueError(f"CTAA assessment {label} changed while being read")
    return b"".join(chunks)


def _load_read_only_json_with_sha(
    path: Path, label: str
) -> tuple[dict[str, object], str]:
    raw = _read_immutable_bytes(path, label)
    return _decode_read_only_json(raw, label), hashlib.sha256(raw).hexdigest()


def _decode_read_only_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CTAA assessment {label} JSON differs") from error
    if not isinstance(value, dict):
        raise ValueError(f"CTAA assessment {label} root differs")
    return value


def _load_read_only_json(path: Path, label: str) -> dict[str, object]:
    value, _ = _load_read_only_json_with_sha(path, label)
    return value


def _load_registry_public_key(path: Path) -> bytes:
    raw = _read_immutable_bytes(path, "registry public key")
    if len(raw) == 32:
        return raw
    try:
        text = raw.decode("ascii").strip()
        key = bytes.fromhex(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("CTAA assessment registry public key differs") from error
    if len(key) != 32 or text != key.hex():
        raise ValueError("CTAA assessment registry public key differs")
    return key


def _load_jsonl_bytes(data: bytes, label: str) -> list[dict[str, object]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"CTAA assessment {label} encoding differs") from error
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {item}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"CTAA assessment {label} row {line_number} JSON differs"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"CTAA assessment {label} row {line_number} differs")
        rows.append(value)
    if not rows:
        raise ValueError(f"CTAA assessment {label} is empty")
    return rows


def _load_committed_evidence_bundle_with_receipt_sha(
    directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    receipt_path = directory / "receipt.json"
    evidence_path = directory / "evidence.jsonl"
    receipt, receipt_sha256 = _load_read_only_json_with_sha(
        receipt_path, "evidence receipt"
    )
    evidence_bytes = _read_immutable_bytes(evidence_path, "committed evidence")
    if (
        receipt.get("schema") != RAW_EVIDENCE_RECEIPT_SCHEMA
        or receipt.get("evidence_sha256") != hashlib.sha256(evidence_bytes).hexdigest()
    ):
        raise ValueError("CTAA committed-evidence receipt differs")
    rows = _load_jsonl_bytes(evidence_bytes, "committed evidence")
    if receipt.get("rows") != len(rows):
        raise ValueError("CTAA committed-evidence row count differs")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        family_id = row.get("family_id")
        if (
            set(row) != EVIDENCE_KEYS
            or row.get("schema") != RAW_EVIDENCE_SCHEMA
            or row.get("source_index") != index
            or not isinstance(family_id, str)
            or family_id in seen
        ):
            raise ValueError("CTAA committed-evidence row schema differs")
        seen.add(family_id)
    return receipt, rows, receipt_sha256


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RuntimeBundleError("CTAA runtime bundle is not canonical JSON") from error


def _load_runtime_bundle_with_replay_and_sha(
    bundle_path: Path,
    *,
    run_contract: Mapping[str, object],
    program_path: Path,
    query_path: Path,
    tokenizer_path: Path,
) -> tuple[dict[str, object], str]:
    """Validate and hash the runtime bundle from one immutable byte snapshot."""

    raw = _read_immutable_bytes(bundle_path, "runtime bundle")
    value = _decode_read_only_json(raw, "runtime bundle")
    if raw != _canonical_json_bytes(value):
        raise RuntimeBundleError("CTAA runtime bundle is not canonical JSON")
    bundle_sha256 = hashlib.sha256(raw).hexdigest()
    bundle = validate_runtime_bundle(value, run_contract=run_contract)
    root = Path(os.path.abspath(bundle_path)).parent
    artifacts = []
    plans: list[RuntimeInterventionPlan] = []
    for entry in bundle["entries"]:
        if not isinstance(entry, dict):
            raise RuntimeBundleError("CTAA runtime bundle entry differs")
        plan_path = root / str(entry["runtime_plan_filename"])
        evidence_path = root / str(entry["runtime_evidence_filename"])
        if plan_path.parent != root or evidence_path.parent != root:
            raise RuntimeBundleError("CTAA runtime bundle member escapes package root")
        plan, plan_file_sha = read_runtime_plan(plan_path)
        if (
            plan_file_sha != entry["runtime_plan_file_sha256"]
            or plan.plan_sha256 != entry["runtime_plan_sha256"]
            or plan.bindings.training_seed != entry["training_seed"]
        ):
            raise RuntimeBundleError("CTAA runtime plan member differs")
        evidence = read_runtime_evidence(
            evidence_path,
            plan,
            expected_file_sha256=str(entry["runtime_evidence_file_sha256"]),
        )
        if evidence.get("evidence_sha256") != entry["runtime_evidence_sha256"]:
            raise RuntimeBundleError("CTAA runtime evidence member differs")
        artifacts.append(
            (
                plan,
                evidence,
                str(entry["runtime_plan_filename"]),
                plan_file_sha,
                str(entry["runtime_evidence_filename"]),
                str(entry["runtime_evidence_file_sha256"]),
            )
        )
        plans.append(plan)
    if make_runtime_bundle(run_contract=run_contract, artifacts=artifacts) != bundle:
        raise RuntimeBundleError("CTAA runtime bundle member recomputation differs")
    for plan in plans:
        try:
            rows = load_runtime_replay_rows(
                plan=plan,
                program_path=program_path,
                query_path=query_path,
                tokenizer_path=tokenizer_path,
            )
            replay = replay_runtime_plan(plan, rows)
        except ValueError as error:
            raise RuntimeBundleError(
                "CTAA runtime plan semantic replay failed"
            ) from error
        if (
            replay.plan_sha256 != plan.plan_sha256
            or replay.attempt_count != ATTEMPT_COUNT_PER_SEED
        ):
            raise RuntimeBundleError("CTAA runtime plan semantic replay differs")
    return bundle, bundle_sha256


def _validate_preaccess_custody(
    *,
    manifest_path: Path,
    run_plan_path: Path,
    run_contract_path: Path,
    bootstrap_seed_receipt_path: Path,
    runtime_bundle_path: Path,
    runtime_program_source_path: Path,
    runtime_query_source_path: Path,
    runtime_tokenizer_path: Path,
    runtime_execution_set_path: Path,
    statistical_gate_spec_path: Path,
    access_registry_path: Path,
    access_head_receipt_path: Path,
    registry_verification_key: bytes | Ed25519PublicKey,
    partition: str,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = validate_run_contract(
        contract_path=run_contract_path,
        manifest_path=manifest_path,
        run_plan_path=run_plan_path,
        bootstrap_seed_receipt_path=bootstrap_seed_receipt_path,
    )
    try:
        execution_set, execution_set_file_sha256 = (
            read_runtime_execution_set_with_replay(
                runtime_execution_set_path,
                runtime_bundle_path=runtime_bundle_path,
                run_contract=contract,
                verification_key=registry_verification_key,
            )
        )
    except (RuntimeExecutionSetError, OSError, ValueError) as error:
        raise ValueError("CTAA assessment execution set differs") from error
    execution_set_sha256 = execution_set.get("execution_set_sha256")
    if (
        execution_set.get("partition") != partition
        or execution_set.get("run_contract_sha256")
        != contract.get("run_contract_sha256")
        or not _is_sha256(execution_set_file_sha256)
        or not _is_sha256(execution_set_sha256)
    ):
        raise ValueError("CTAA assessment execution set binding differs")

    runtime_bundle, runtime_bundle_sha256 = _load_runtime_bundle_with_replay_and_sha(
        runtime_bundle_path,
        run_contract=contract,
        program_path=runtime_program_source_path,
        query_path=runtime_query_source_path,
        tokenizer_path=runtime_tokenizer_path,
    )
    if (
        runtime_bundle.get("partition") != partition
        or execution_set.get("runtime_bundle_file_sha256") != runtime_bundle_sha256
    ):
        raise ValueError("CTAA runtime bundle binding differs")
    try:
        statistical_gate_spec, statistical_gate_spec_file_sha256 = (
            read_signed_statistical_gate_spec_with_sha(
                statistical_gate_spec_path,
                verification_key=registry_verification_key,
            )
        )
    except (StatisticalGateSpecError, OSError, ValueError) as error:
        raise ValueError("CTAA statistical gate specification differs") from error
    gate_spec_sha256 = statistical_gate_spec.get("gate_spec_sha256")
    gate_payload = statistical_gate_spec.get("payload")
    gate_bindings = (
        gate_payload.get("bindings") if isinstance(gate_payload, dict) else None
    )
    expected_gate_bindings = {
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_plan_sha256": contract["run_plan_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_bundle_file_sha256": runtime_bundle_sha256,
        "runtime_bundle_sha256": runtime_bundle.get("bundle_sha256"),
        "runtime_execution_set_file_sha256": execution_set_file_sha256,
        "runtime_execution_set_sha256": execution_set_sha256,
        "bootstrap_seed_receipt_sha256": contract[
            "bootstrap_seed_receipt_sha256"
        ],
        "bootstrap_seed": contract["bootstrap_seed"],
        "training_seeds": contract.get("training_seeds"),
    }
    if (
        not _is_sha256(statistical_gate_spec_file_sha256)
        or not _is_sha256(gate_spec_sha256)
        or not isinstance(gate_bindings, dict)
        or any(
            gate_bindings.get(key) != expected
            for key, expected in expected_gate_bindings.items()
        )
    ):
        raise ValueError("CTAA statistical gate specification binding differs")
    head_receipt, head_receipt_sha256 = _load_read_only_json_with_sha(
        access_head_receipt_path, "access head receipt"
    )
    events = verify_registry_events(
        access_registry_path,
        registry_verification_key,
        expected_head_receipt=head_receipt,
    )
    if not events:
        raise ValueError("CTAA assessment access registry is empty")
    event = events[-1]
    payload = event.payload
    if payload.get("event_type") != ACCESS_SPEND:
        raise ValueError("CTAA assessment signed access binding differs")
    expected = {
        "event_type": ACCESS_SPEND,
        "partition": partition,
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "assessment_claim_sha256": payload["assessment_claim_sha256"],
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
    }
    if any(payload.get(key) != item for key, item in expected.items()):
        raise ValueError("CTAA assessment signed access binding differs")
    access = {
        "schema": ASSESSMENT_ACCESS_SCHEMA,
        "registry_id": payload["registry_id"],
        "registry_head_receipt_sha256": head_receipt_sha256,
        "registry_head_entry_hash": event.entry_hash,
        "access_event_payload_sha256": hashlib.sha256(
            event.canonical_payload
        ).hexdigest(),
        "access_id": payload["access_id"],
        "partition": partition,
        "manifest_sha256": contract["manifest_sha256"],
        "board_sha256": contract["board_sha256"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "assessment_claim_sha256": payload["assessment_claim_sha256"],
        "execution_set_file_sha256": execution_set_file_sha256,
        "execution_set_sha256": execution_set_sha256,
        "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
        "access": 1,
    }
    return contract, access


def _validate_runs_against_contract(
    *,
    contract: dict[str, object],
    names: list[str],
    evidence_commitments: dict[str, dict[str, object]],
    evidence_receipt_sha256: dict[str, str],
    evidence_directories: dict[str, Path],
    parent_commitments: dict[str, dict[str, object]],
    parent_receipt_sha256: dict[str, str],
    parent_directories: dict[str, Path],
    metadata: dict[str, dict[str, object]],
    oracle_paths: dict[str, Path],
) -> None:
    contract_runs = contract.get("runs")
    if not isinstance(contract_runs, list):
        raise ValueError("CTAA assessment run contract differs")
    indexed = {row.get("run_id"): row for row in contract_runs if isinstance(row, dict)}
    if set(indexed) != set(names) or len(indexed) != len(contract_runs):
        raise ValueError("CTAA assessment run set differs from contract")
    oracle_files = contract.get("oracle_files")
    if not isinstance(oracle_files, dict):
        raise ValueError("CTAA assessment oracle contract differs")
    for name in names:
        row = indexed[name]
        run_meta = metadata[name]
        if any(run_meta.get(key) != row.get(key) for key in ("seed", "arm", "dataset")):
            raise ValueError("CTAA assessment run metadata differs from contract")
        if evidence_receipt_sha256[name] != row.get("raw_evidence_receipt_sha256"):
            raise ValueError("CTAA assessment evidence receipt differs from contract")
        commitment = evidence_commitments[name]
        if (
            commitment.get("core_training") != row.get("core_training")
            or commitment.get("compiler_sha256") != row.get("compiler_sha256")
            or any(
                commitment.get(key) != row.get("evidence_artifacts", {}).get(key)
                for key in row.get("evidence_artifacts", {})
            )
        ):
            raise ValueError("CTAA assessment evidence artifacts differ from contract")
        dataset = str(row["dataset"])
        oracle = oracle_files.get(dataset)
        if (
            not isinstance(oracle, dict)
            or oracle_paths[name].name != oracle.get("filename")
            or row.get("sealed_sources", {}).get("oracle_sha256")
            != oracle.get("sha256")
        ):
            raise ValueError("CTAA assessment oracle identity differs from contract")
        if dataset == "intervention":
            if parent_receipt_sha256[name] != row.get(
                "parent_evidence_receipt_sha256"
            ) or parent_commitments[name].get("evidence_sha256") != row.get(
                "parent_evidence_sha256"
            ):
                raise ValueError(
                    "CTAA assessment parent evidence differs from contract"
                )


def assess(
    *,
    manifest_path: Path,
    run_plan_path: Path,
    run_contract_path: Path,
    bootstrap_seed_receipt_path: Path,
    runtime_bundle_path: Path,
    runtime_program_source_path: Path,
    runtime_query_source_path: Path,
    runtime_tokenizer_path: Path,
    runtime_execution_set_path: Path,
    statistical_gate_spec_path: Path,
    access_registry_path: Path,
    access_head_receipt_path: Path,
    registry_verification_key: bytes | Ed25519PublicKey,
    partition: str,
    runs: list[tuple[str, Path, Path]],
    output_path: Path,
    parent_evidence: dict[str, Path] | None = None,
    run_metadata: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    if output_path.exists() or not runs:
        raise FileExistsError("CTAA assessment output exists or run set is empty")
    names = [name for name, _, _ in runs]
    if len(set(names)) != len(names):
        raise ValueError("CTAA assessment run names differ")
    # This barrier belongs to the full assessor. Authenticate every seed's
    # query-blind execution receipt before opening query-derived evidence.
    contract, access = _validate_preaccess_custody(
        manifest_path=manifest_path,
        run_plan_path=run_plan_path,
        run_contract_path=run_contract_path,
        bootstrap_seed_receipt_path=bootstrap_seed_receipt_path,
        runtime_bundle_path=runtime_bundle_path,
        runtime_program_source_path=runtime_program_source_path,
        runtime_query_source_path=runtime_query_source_path,
        runtime_tokenizer_path=runtime_tokenizer_path,
        runtime_execution_set_path=runtime_execution_set_path,
        statistical_gate_spec_path=statistical_gate_spec_path,
        access_registry_path=access_registry_path,
        access_head_receipt_path=access_head_receipt_path,
        registry_verification_key=registry_verification_key,
        partition=partition,
    )
    manifest, manifest_sha = _load_manifest(manifest_path)
    evidence_directories = {name: evidence_dir for name, evidence_dir, _ in runs}
    oracle_paths = {name: oracle_path for name, _, oracle_path in runs}
    evidence_bundles = {
        name: _load_committed_evidence_bundle_with_receipt_sha(evidence_dir)
        for name, evidence_dir in evidence_directories.items()
    }
    evidence_commitments = {
        name: bundle[0] for name, bundle in evidence_bundles.items()
    }
    evidence_by_name = {name: bundle[1] for name, bundle in evidence_bundles.items()}
    evidence_receipt_sha256 = {
        name: bundle[2] for name, bundle in evidence_bundles.items()
    }
    parent_directories = dict(parent_evidence or {})
    parent_bundles = {
        name: _load_committed_evidence_bundle_with_receipt_sha(path)
        for name, path in parent_directories.items()
    }
    parent_commitments = {name: bundle[0] for name, bundle in parent_bundles.items()}
    parents = {name: bundle[1] for name, bundle in parent_bundles.items()}
    parent_receipt_sha256 = {name: bundle[2] for name, bundle in parent_bundles.items()}
    for name, _, oracle_path in runs:
        _verify_board_path_declared(oracle_path, manifest_path, manifest)
        _verify_evidence_source_binding(
            evidence_commitments[name],
            oracle_path,
            manifest,
        )
    parent_oracle_paths: dict[str, Path] = {}
    for name, _, oracle_path in runs:
        if name not in parents:
            continue
        marker = "_intervention_oracle.jsonl"
        if not oracle_path.name.endswith(marker):
            raise ValueError("CTAA parent evidence supplied for a non-intervention run")
        parent_oracle_path = oracle_path.with_name(
            oracle_path.name[: -len(marker)] + "_oracle.jsonl"
        )
        _verify_board_path_declared(parent_oracle_path, manifest_path, manifest)
        _verify_evidence_source_binding(
            parent_commitments[name],
            parent_oracle_path,
            manifest,
        )
        parent_oracle_paths[name] = parent_oracle_path
    if run_metadata is None:
        raise ValueError("CTAA assessment requires checkpoint-bound run metadata")
    metadata = run_metadata
    if set(metadata) != set(names):
        raise ValueError("CTAA assessment run metadata set differs")
    intervention_names = {
        name for name in names if metadata[name].get("dataset") == "intervention"
    }
    if set(parents) != intervention_names:
        raise ValueError("CTAA intervention assessment requires exact parent evidence")
    for name in names:
        run_meta = metadata[name]
        commitment_training = evidence_commitments[name].get("core_training")
        if not isinstance(commitment_training, dict):
            raise ValueError("CTAA assessment lacks core-training commitment")
        if run_meta.get("arm") not in ARMS:
            raise ValueError("CTAA assessment arm metadata differs")
        if not isinstance(run_meta.get("seed"), int) or int(run_meta["seed"]) < 0:
            raise ValueError("CTAA assessment seed metadata differs")
        if run_meta.get("dataset") not in {"base", "intervention"}:
            raise ValueError("CTAA assessment dataset metadata differs")
        if (
            commitment_training.get("training_seed") != run_meta["seed"]
            or commitment_training.get("training_arm") != run_meta["arm"]
        ):
            raise ValueError("CTAA assessment metadata differs from core checkpoint")
    if contract.get("manifest_sha256") != manifest_sha:
        raise ValueError("CTAA assessment manifest differs from run contract")
    _validate_runs_against_contract(
        contract=contract,
        names=names,
        evidence_commitments=evidence_commitments,
        evidence_receipt_sha256=evidence_receipt_sha256,
        evidence_directories=evidence_directories,
        parent_commitments=parent_commitments,
        parent_receipt_sha256=parent_receipt_sha256,
        parent_directories=parent_directories,
        metadata=metadata,
        oracle_paths=oracle_paths,
    )
    oracle_hashes: dict[str, str] = {}
    oracles: dict[str, list[dict[str, object]]] = {}
    for name, _, oracle_path in runs:
        expected = manifest["files"][oracle_path.name]
        assert isinstance(expected, str)
        oracles[name] = load_oracle(oracle_path, partition, expected_sha256=expected)
        oracle_hashes[name] = expected
    parent_oracles: dict[str, list[dict[str, object]]] = {}
    for name, parent_oracle_path in parent_oracle_paths.items():
        expected = manifest["files"][parent_oracle_path.name]
        assert isinstance(expected, str)
        parent_oracles[name] = load_oracle(
            parent_oracle_path, partition, expected_sha256=expected
        )
    scores = {}
    for name, _, _oracle_path in runs:
        run_meta = metadata[name]
        scores[name] = {
            "seed": run_meta.get("seed"),
            "arm": run_meta.get("arm"),
            "dataset": run_meta.get("dataset"),
            "evidence_commitment": evidence_commitments[name],
            "scores": score_evidence(
                evidence_by_name[name],
                oracles[name],
                parent_evidence_rows=parents.get(name),
                parent_oracle_rows=parent_oracles.get(name),
            ),
        }
    report = {
        "schema": ASSESSMENT_SCHEMA,
        "partition": partition,
        "manifest_sha256": manifest_sha,
        "access": access,
        "oracle_sha256": oracle_hashes,
        "runs": scores,
        "capability_gate_computed": False,
    }
    report_sha = write_json_once(output_path, report)
    return {**report, "report_sha256": report_sha}


def main() -> None:
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
    parser.add_argument("--statistical-gate-spec", type=Path, required=True)
    parser.add_argument("--access-registry", type=Path, required=True)
    parser.add_argument("--access-head-receipt", type=Path, required=True)
    parser.add_argument("--registry-public-key", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "confirmation"), required=True
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=6,
        metavar=("NAME", "SEED", "ARM", "DATASET", "EVIDENCE_DIR", "ORACLE_JSONL"),
        required=True,
    )
    parser.add_argument(
        "--parent-evidence",
        action="append",
        nargs=2,
        metavar=("NAME", "EVIDENCE_DIR"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        (name, Path(evidence), Path(oracle))
        for name, _seed, _arm, _dataset, evidence, oracle in args.run
    ]
    metadata = {
        name: {"seed": int(seed), "arm": arm, "dataset": dataset}
        for name, seed, arm, dataset, _evidence, _oracle in args.run
    }
    parents = {name: Path(path) for name, path in args.parent_evidence}
    report = assess(
        manifest_path=args.manifest,
        run_plan_path=args.run_plan,
        run_contract_path=args.run_contract,
        bootstrap_seed_receipt_path=args.bootstrap_seed_receipt,
        runtime_bundle_path=args.runtime_bundle,
        runtime_program_source_path=args.runtime_program_source,
        runtime_query_source_path=args.runtime_query_source,
        runtime_tokenizer_path=args.runtime_tokenizer,
        runtime_execution_set_path=args.runtime_execution_set,
        statistical_gate_spec_path=args.statistical_gate_spec,
        access_registry_path=args.access_registry,
        access_head_receipt_path=args.access_head_receipt,
        registry_verification_key=_load_registry_public_key(args.registry_public_key),
        partition=args.partition,
        runs=runs,
        output_path=args.output,
        parent_evidence=parents,
        run_metadata=metadata,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
