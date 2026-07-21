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


def spend_partition_access(
    ledger_path: Path,
    *,
    partition: str,
    manifest_sha256: str,
    run_names: list[str],
) -> dict[str, object]:
    key = f"{partition}_access"
    if key not in {"development_access", "confirmation_access"}:
        raise ValueError("CTAA assessment partition differs")
    if ledger_path.stat().st_mode & 0o077:
        raise PermissionError("CTAA access ledger permissions differ")
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
    oracle_hashes = {
        name: _verify_board_file(oracle_path, manifest_path, manifest)
        for name, _, oracle_path in runs
    }
    if partition == "confirmation":
        if development_gate_receipt is None:
            raise ValueError("CTAA confirmation requires a frozen development gate receipt")
        gate = json.loads(development_gate_receipt.read_text())
        if (
            not isinstance(gate, dict)
            or gate.get("schema") != "r12_ctaa_v2_development_gate_v1"
            or gate.get("manifest_sha256") != manifest_sha
            or gate.get("all_development_gates_pass") is not True
            or development_gate_receipt.stat().st_mode & 0o222
        ):
            raise ValueError("CTAA development gate receipt differs")
    access = spend_partition_access(
        ledger_path,
        partition=partition,
        manifest_sha256=manifest_sha,
        run_names=names,
    )
    metadata = run_metadata or {
        name: {"seed": None, "arm": None, "dataset": None} for name in names
    }
    if set(metadata) != set(names):
        raise ValueError("CTAA assessment run metadata set differs")
    scores = {}
    for name, _, oracle_path in runs:
        oracle = load_oracle(oracle_path, partition)
        run_meta = metadata[name]
        if run_meta.get("arm") is not None and run_meta.get("arm") not in ARMS:
            raise ValueError("CTAA assessment arm metadata differs")
        if run_meta.get("seed") is not None and (
            not isinstance(run_meta.get("seed"), int) or int(run_meta["seed"]) < 0
        ):
            raise ValueError("CTAA assessment seed metadata differs")
        if run_meta.get("dataset") is not None and run_meta.get("dataset") not in {
            "base",
            "intervention",
        }:
            raise ValueError("CTAA assessment dataset metadata differs")
        scores[name] = {
            "seed": run_meta.get("seed"),
            "arm": run_meta.get("arm"),
            "dataset": run_meta.get("dataset"),
            "evidence_commitment": evidence_commitments[name],
            "scores": score_evidence(
                evidence_by_name[name],
                oracle,
                parent_evidence_rows=parents.get(name),
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
