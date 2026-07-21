#!/usr/bin/env python3
"""Spend one sealed-board access and score committed CTAA evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from ctaa_assessment import (
    load_committed_evidence,
    load_committed_evidence_receipt,
    load_oracle,
    score_evidence,
)
from ctaa_core_training import ARMS
from ctaa_evaluation_io import sha256_file, write_json_once


ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v1"


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != "r12_ctaa_v2_manifest_v1"
        or not isinstance(value.get("files"), dict)
    ):
        raise ValueError("CTAA assessment board manifest differs")
    return value


def _verify_board_file(path: Path, manifest_path: Path, manifest: dict[str, object]) -> str:
    if path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("CTAA assessed oracle is outside the sealed board")
    expected = manifest["files"].get(path.name)
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError("CTAA assessed oracle hash differs from sealed board")
    return actual


def _verify_board_path_declared(
    path: Path,
    manifest_path: Path,
    manifest: dict[str, object],
) -> None:
    if (
        path.resolve().parent != manifest_path.resolve().parent
        or not isinstance(manifest["files"].get(path.name), str)
    ):
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


def spend_partition_access(
    ledger_path: Path,
    *,
    partition: str,
    manifest_sha256: str,
    run_names: list[str],
    expected_ledger_sha256: str,
) -> dict[str, object]:
    key = f"{partition}_access"
    if key not in {"development_access", "confirmation_access"}:
        raise ValueError("CTAA assessment partition differs")
    if ledger_path.stat().st_mode & 0o077:
        raise PermissionError("CTAA access ledger permissions differ")
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sha256_file(ledger_path) != expected_ledger_sha256:
            raise ValueError("CTAA access ledger changed before atomic spend")
        value = json.loads(ledger_path.read_text())
        if (
            not isinstance(value, dict)
            or value.get("schema") != "r12_ctaa_v2_access_ledger_v1"
            or value.get("development_access") not in {0, 1}
            or value.get("confirmation_access") not in {0, 1}
            or value.get(key) != 0
        ):
            raise ValueError("CTAA sealed partition access is already spent or malformed")
        previous_sha = sha256_file(ledger_path)
        history = value.get("history", [])
        if not isinstance(history, list):
            raise ValueError("CTAA access-ledger history differs")
        value[key] = 1
        value["history"] = [
            *history,
            {
                "partition": partition,
                "manifest_sha256": manifest_sha256,
                "previous_ledger_sha256": previous_sha,
                "run_names": run_names,
            },
        ]
        temporary = ledger_path.with_name(ledger_path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError("refusing existing CTAA access-ledger temporary")
        try:
            temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
            temporary.chmod(0o600)
            temporary.replace(ledger_path)
        finally:
            if temporary.exists():
                temporary.chmod(0o600)
                temporary.unlink()
        ledger_path.chmod(0o600)
        return {
            "partition": partition,
            "previous_ledger_sha256": previous_sha,
            "ledger_sha256": sha256_file(ledger_path),
            "access": 1,
        }


def assess(
    *,
    manifest_path: Path,
    ledger_path: Path,
    partition: str,
    runs: list[tuple[str, Path, Path]],
    output_path: Path,
    parent_evidence: dict[str, Path] | None = None,
    run_metadata: dict[str, dict[str, object]] | None = None,
    development_gate_receipt: Path | None = None,
) -> dict[str, object]:
    if output_path.exists() or not runs:
        raise FileExistsError("CTAA assessment output exists or run set is empty")
    names = [name for name, _, _ in runs]
    if len(set(names)) != len(names):
        raise ValueError("CTAA assessment run names differ")
    manifest = _load_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if ledger_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("CTAA access ledger is outside the sealed board")
    if ledger_path.name != "access_ledger.json":
        raise ValueError("CTAA access ledger is not the canonical board ledger")
    evidence_by_name = {
        name: load_committed_evidence(evidence_dir)
        for name, evidence_dir, _ in runs
    }
    evidence_commitments = {
        name: load_committed_evidence_receipt(evidence_dir)
        for name, evidence_dir, _ in runs
    }
    parents = {
        name: load_committed_evidence(path)
        for name, path in (parent_evidence or {}).items()
    }
    parent_commitments = {
        name: load_committed_evidence_receipt(path)
        for name, path in (parent_evidence or {}).items()
    }
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
        if (
            not isinstance(run_meta.get("seed"), int)
            or int(run_meta["seed"]) < 0
        ):
            raise ValueError("CTAA assessment seed metadata differs")
        if run_meta.get("dataset") not in {"base", "intervention"}:
            raise ValueError("CTAA assessment dataset metadata differs")
        if (
            commitment_training.get("training_seed") != run_meta["seed"]
            or commitment_training.get("training_arm") != run_meta["arm"]
        ):
            raise ValueError("CTAA assessment metadata differs from core checkpoint")
    initial_ledger_sha = manifest["files"].get("access_ledger.json")
    if not isinstance(initial_ledger_sha, str):
        raise ValueError("CTAA manifest lacks the initial access ledger")
    current_ledger_sha = sha256_file(ledger_path)
    ledger_value = json.loads(ledger_path.read_text())
    if not isinstance(ledger_value, dict):
        raise ValueError("CTAA access ledger differs")
    if partition == "development":
        if (
            current_ledger_sha != initial_ledger_sha
            or ledger_value.get("development_access") != 0
            or ledger_value.get("confirmation_access") != 0
            or ledger_value.get("history", []) != []
        ):
            raise ValueError("CTAA development access ledger lineage differs")
    if partition == "confirmation":
        if development_gate_receipt is None:
            raise ValueError("CTAA confirmation requires a frozen development gate receipt")
        gate = json.loads(development_gate_receipt.read_text())
        if (
            not isinstance(gate, dict)
            or gate.get("schema") != "r12_ctaa_v2_development_gate_v1"
            or gate.get("manifest_sha256") != manifest_sha
            or gate.get("all_development_gates_pass") is not True
            or gate.get("development_access_ledger_sha256") != current_ledger_sha
            or development_gate_receipt.stat().st_mode & 0o222
        ):
            raise ValueError("CTAA development gate receipt differs")
        history = ledger_value.get("history")
        if (
            ledger_value.get("development_access") != 1
            or ledger_value.get("confirmation_access") != 0
            or not isinstance(history, list)
            or len(history) != 1
            or history[0].get("partition") != "development"
            or history[0].get("previous_ledger_sha256") != initial_ledger_sha
        ):
            raise ValueError("CTAA confirmation access ledger lineage differs")
    access = spend_partition_access(
        ledger_path,
        partition=partition,
        manifest_sha256=manifest_sha,
        run_names=names,
        expected_ledger_sha256=current_ledger_sha,
    )
    oracle_hashes = {
        name: _verify_board_file(oracle_path, manifest_path, manifest)
        for name, _, oracle_path in runs
    }
    for parent_oracle_path in parent_oracle_paths.values():
        _verify_board_file(parent_oracle_path, manifest_path, manifest)
    scores = {}
    for name, _, oracle_path in runs:
        oracle = load_oracle(oracle_path, partition)
        run_meta = metadata[name]
        scores[name] = {
            "seed": run_meta.get("seed"),
            "arm": run_meta.get("arm"),
            "dataset": run_meta.get("dataset"),
            "evidence_commitment": evidence_commitments[name],
            "scores": score_evidence(
                evidence_by_name[name],
                oracle,
                parent_evidence_rows=parents.get(name),
                parent_oracle_rows=(
                    load_oracle(parent_oracle_paths[name], partition)
                    if name in parent_oracle_paths
                    else None
                ),
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
    parser.add_argument("--access-ledger", type=Path, required=True)
    parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
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
    parser.add_argument("--development-gate-receipt", type=Path)
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
        ledger_path=args.access_ledger,
        partition=args.partition,
        runs=runs,
        output_path=args.output,
        parent_evidence=parents,
        run_metadata=metadata,
        development_gate_receipt=args.development_gate_receipt,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
