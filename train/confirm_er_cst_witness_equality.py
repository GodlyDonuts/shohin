#!/usr/bin/env python3
"""Run the sole sealed confirmation for ER-CST Witness Equality Bus v1.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Mapping

import torch

from assess_er_cst_witness_equality import ASSESSMENT_SCHEMA as DEVELOPMENT_ASSESSMENT_SCHEMA
from build_er_cst_fresh_board import CONFIRMATION_SPLIT
from build_er_cst_witness_equality_board import BOARD_SCHEMA, PROTOCOL
from er_cst_fresh import load_trainable_state
from er_cst_witness_equality import TRAINING_CONTRACT, evaluate_arm, load_split
from pilot_er_cst_witness_equality import (
    ACCESS_SCHEMA as DEVELOPMENT_ACCESS_SCHEMA,
    BOARD_REPORT_SHA256,
    BOARD_SOURCE_COMMIT,
    CHECKPOINT_SCHEMA,
    REPORT_SCHEMA as DEVELOPMENT_REPORT_SCHEMA,
    THRESHOLDS,
    atomic_json_save,
    atomic_torch_save,
    compute_gates,
    initialize_system,
    release_cuda,
    runtime_manifest,
    source_manifest,
)
from pilot_sd_cst_byte_addressed import sha256_file


SCIENTIFIC_SOURCE_COMMIT = "87d53b53462d8d15660663238fd33886c010efb7"
TRAINING_SEED = 2_262_748_995_832_026_278
CHECKPOINT_SHA256 = "917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7"
DEVELOPMENT_EVIDENCE_SHA256 = "1a7504eb9b08d7d123e89705360f2eb37a861f5cd75b3ebc73570c8e904327fb"
DEVELOPMENT_REPORT_SHA256 = "d295f8f67f32916386e04674fc782a0982b9b1b55f7b82aa1eaab6f59bb1ae35"
DEVELOPMENT_ASSESSMENT_SHA256 = "29e4349225ed9523ec3b8096cd2cd16ef1b55c727797421a1ac0b39c042f11b2"
DEVELOPMENT_LEDGER_SHA256 = "5b6e233b3cc9d3cf49a32525ca11f6c6f846005486df67a252e9ca4ec36b4db3"
CONFIRMATION_SHA256 = "6593bb17690fc72e5392b953af75f8686a92e799bcd600307affcb7fc0080c4d"

AUTHORIZATION_SCHEMA = "r12_er_cst_witness_equality_confirmation_authorization_v1_1"
ACCESS_SCHEMA = "r12_er_cst_witness_equality_confirmation_access_v1_1"
EVIDENCE_SCHEMA = "r12_er_cst_witness_equality_confirmation_evidence_v1_1"
REPORT_SCHEMA = "r12_er_cst_witness_equality_confirmation_report_v1_1"
EVALUATOR_SOURCE_PATHS = (
    "R12_ER_CST_WITNESS_EQUALITY_CONFIRMATION_PREREG.md",
    "train/confirm_er_cst_witness_equality.py",
    "train/assess_er_cst_witness_equality_confirmation.py",
    "train/jobs/er_cst_witness_equality_confirmation.sbatch",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def evaluator_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args), cwd=repo_root, check=False, capture_output=True, text=True
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("ER-CST witness confirmation evaluator commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("ER-CST witness confirmation evaluator is not an ancestor")
    hashes = {}
    for relative in EVALUATOR_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"ER-CST witness evaluator omits path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"ER-CST witness evaluator runtime differs: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"ER-CST witness confirmation JSON is not an object: {path}")
    return value


def verify_authorization(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    board_path = args.data_dir / "report.json"
    if sha256_file(board_path) != BOARD_REPORT_SHA256:
        raise RuntimeError("ER-CST witness confirmation board report differs")
    board = load_json(board_path)
    if (
        board.get("schema") != BOARD_SCHEMA
        or board.get("protocol") != PROTOCOL
        or board.get("source_commit") != BOARD_SOURCE_COMMIT
        or board.get("all_gates_pass") is not True
        or board.get("development_accesses") != 0
        or board.get("confirmation_accesses") != 0
        or board["files"]["confirmation.jsonl"]["sha256"] != CONFIRMATION_SHA256
    ):
        raise RuntimeError("ER-CST witness confirmation board identity differs")
    expected = {
        args.checkpoint: CHECKPOINT_SHA256,
        args.development_evidence: DEVELOPMENT_EVIDENCE_SHA256,
        args.development_report: DEVELOPMENT_REPORT_SHA256,
        args.development_assessment: DEVELOPMENT_ASSESSMENT_SHA256,
        args.development_ledger: DEVELOPMENT_LEDGER_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"ER-CST witness authorization artifact differs: {path}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    development_report = load_json(args.development_report)
    development_assessment = load_json(args.development_assessment)
    development_ledger = load_json(args.development_ledger)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("scientific_source_commit") != SCIENTIFIC_SOURCE_COMMIT
        or checkpoint.get("training_seed") != TRAINING_SEED
        or checkpoint.get("training_contract") != TRAINING_CONTRACT
        or checkpoint.get("board_report_sha256") != BOARD_REPORT_SHA256
        or development_report.get("schema") != DEVELOPMENT_REPORT_SCHEMA
        or development_report.get("decision") != "authorize_one_sealed_confirmation"
        or development_report.get("all_gates_pass") is not True
        or development_assessment.get("schema") != DEVELOPMENT_ASSESSMENT_SCHEMA
        or development_assessment.get("decision") != "authorize_one_sealed_confirmation"
        or development_assessment.get("all_gates_pass") is not True
        or development_assessment.get("custody")
        != {"development_accesses": 1, "confirmation_accesses": 0}
        or development_ledger.get("schema") != DEVELOPMENT_ACCESS_SCHEMA
        or development_ledger.get("split") != "er_cst_development"
        or development_ledger.get("access_number") != 1
        or development_ledger.get("scientific_source_commit")
        != SCIENTIFIC_SOURCE_COMMIT
    ):
        raise RuntimeError("ER-CST witness development authorization is not exact")
    access_files = sorted((args.data_dir / "access").glob("*.json"))
    if access_files != [args.development_ledger]:
        raise RuntimeError("ER-CST witness confirmation pre-access ledger set differs")
    confirmation_path = args.data_dir / "confirmation.jsonl"
    if stat.S_IMODE(confirmation_path.stat().st_mode) != 0o600:
        raise RuntimeError("ER-CST witness confirmation file mode differs")
    return board, checkpoint, development_assessment


def consume_confirmation_access(
    data_dir: Path, evaluator_source_commit: str
) -> dict[str, str]:
    payload = (
        json.dumps(
            {
                "schema": ACCESS_SCHEMA,
                "protocol": PROTOCOL,
                "split": CONFIRMATION_SPLIT,
                "board_report_sha256": BOARD_REPORT_SHA256,
                "split_sha256": CONFIRMATION_SHA256,
                "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
                "evaluator_source_commit": evaluator_source_commit,
                "access_number": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path = data_dir / "access" / f"er_cst_witness_confirmation_{CONFIRMATION_SHA256}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def confirmation_gates(
    metrics: Mapping[str, Mapping[str, object]], checkpoint: Mapping[str, object]
) -> dict[str, bool]:
    gates = compute_gates(metrics, checkpoint)
    if gates.pop("development_one_confirmation_zero", None) is not True:
        raise RuntimeError("ER-CST witness development custody gate differs")
    gates["development_one_confirmation_one"] = True
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--development-assessment", type=Path, required=True)
    parser.add_argument("--development-ledger", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmed-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-assessment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluator-source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing ER-CST witness confirmation output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("ER-CST witness confirmation requires bf16 CUDA")
    evaluator = evaluator_manifest(args.repo_root.resolve(), args.evaluator_source_commit)
    scientific = source_manifest(args.repo_root.resolve(), SCIENTIFIC_SOURCE_COMMIT)
    board, checkpoint, development_assessment = verify_authorization(args)
    args.out_dir.mkdir(parents=True)
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source": scientific,
        "evaluator_source": evaluator,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "development_evidence_sha256": DEVELOPMENT_EVIDENCE_SHA256,
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "development_assessment_sha256": DEVELOPMENT_ASSESSMENT_SHA256,
        "development_ledger_sha256": DEVELOPMENT_LEDGER_SHA256,
        "development_decision": development_assessment["decision"],
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    authorization_path = args.out_dir / "authorization.json"
    atomic_json_save(authorization, authorization_path)
    ledger = consume_confirmation_access(args.data_dir, args.evaluator_source_commit)
    confirmation_path = args.data_dir / "confirmation.jsonl"
    if sha256_file(confirmation_path) != CONFIRMATION_SHA256:
        raise RuntimeError("ER-CST witness sealed confirmation hash differs after access")
    rows = load_split(
        args.data_dir,
        board,
        filename="confirmation.jsonl",
        split=CONFIRMATION_SPLIT,
        expected=2_048,
    )
    args.seed = TRAINING_SEED
    device = torch.device("cuda")
    metrics: dict[str, Mapping[str, object]] = {}
    raw_evidence: dict[str, object] = {}
    for arm_name in TRAINING_CONTRACT["arms"]:
        model, motor, reader, parameters, _, receipt = initialize_system(args, device)
        if parameters != checkpoint["parameters"] or receipt != checkpoint["parent_receipt"]:
            raise RuntimeError("ER-CST witness confirmation reconstruction differs")
        arm = checkpoint["arms"][arm_name]
        load_trainable_state(model, arm["compiler_trainable_state"])
        motor.load_state_dict(arm["motor_state"], strict=True)
        reader.load_state_dict(arm["reader_state"], strict=True)
        result = evaluate_arm(
            model, motor, reader, rows, batch_size=args.batch_size, include_raw=True
        )
        raw_evidence[arm_name] = result.pop("raw")
        metrics[arm_name] = result
        release_cuda(model, motor, reader)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "evaluator_source_commit": args.evaluator_source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "confirmation_sha256": CONFIRMATION_SHA256,
        "arms": raw_evidence,
        "development_accesses": 1,
        "confirmation_accesses": 1,
    }
    evidence_path = args.out_dir / "confirmation_evidence.pt"
    atomic_torch_save(evidence, evidence_path)
    gates = confirmation_gates(metrics, checkpoint)
    decision = (
        "confirm_er_cst_witness_equality_v1_1"
        if all(gates.values())
        else "reject_er_cst_witness_equality_confirmation_v1_1"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "evaluator_source_commit": args.evaluator_source_commit,
        "training_seed": TRAINING_SEED,
        "training_contract": TRAINING_CONTRACT,
        "thresholds": THRESHOLDS,
        "parameters": checkpoint["parameters"],
        "metrics": metrics,
        "gates": gates,
        "authorization": authorization,
        "artifacts": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "authorization_sha256": sha256_file(authorization_path),
            "evidence_sha256": sha256_file(evidence_path),
            "development_assessment_sha256": DEVELOPMENT_ASSESSMENT_SHA256,
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 1,
            "confirmation_ledger": ledger,
        },
        "runtime": runtime_manifest(),
        "claim_boundary": (
            "Confirmed bounded fresh episodic S3 rule inference by learned witness "
            "equality, source-deleted categorical composition, internal halt, and "
            "late-query readout. This is not unrestricted language grounding, "
            "arbitrary algorithms, arithmetic, planning, or broad general reasoning."
        ),
    }
    report_path = args.out_dir / "confirmation_report.json"
    atomic_json_save(report, report_path)
    print(
        json.dumps(
            {"decision": decision, "report": str(report_path), "sha256": sha256_file(report_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
