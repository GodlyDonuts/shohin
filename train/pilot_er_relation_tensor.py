#!/usr/bin/env python3
"""Fit three ER-TT arms and consume the sole fresh development read."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping

import torch

from build_er_relation_tensor_board import DEVELOPMENT_SPLIT, PROTOCOL, TRAIN_SPLIT
from er_cst_fresh import canonical_json, derived_seed, load_trainable_state, trainable_state
from er_relation_tensor_training import (
    TRAINING_CONTRACT,
    evaluate_arm,
    fit_arm,
    load_board_receipt,
    load_split,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_relation_tensor_adapter import initialize_er_relation_tensor
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_renderer_native_program import frozen_state_digest


CHECKPOINT_SCHEMA = "r12_er_relation_tensor_checkpoint_v1"
EVIDENCE_SCHEMA = "r12_er_relation_tensor_development_evidence_v1"
REPORT_SCHEMA = "r12_er_relation_tensor_development_report_v1"
ACCESS_SCHEMA = "r12_er_relation_tensor_development_access_v1"
BOARD_SOURCE_COMMIT = "bd77c0fafdbc527688ba57aedd74ccdfbe2ed1cf"
BOARD_REPORT_SHA256 = (
    "64ea4c0e19ea029102af240d44242c830d7b014e49a59af09a836b2d3efb6010"
)
EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 67_659_190,
    "motor": 0,
    "reader": 0,
    "complete_system": 192_740_854,
    "headroom_below_200m": 7_259_146,
    "trainable": 12_037_293,
}
THRESHOLDS = {
    "packet_overall": 0.90,
    "state_overall": 0.90,
    "answer_overall": 0.90,
    "joint_overall": 0.90,
    "field_overall": 0.95,
    "pointer_overall": 0.95,
    "joint_min_cardinality": 0.80,
    "joint_min_depth": 0.80,
    "joint_min_renderer": 0.80,
    "joint_non_bijective": 0.85,
    "treatment_advantage": 0.50,
    "negative_max": 0.35,
}
FROZEN_SOURCE_PATHS = (
    "R12_ER_RELATION_TENSOR_TRANSPORT_THEORY.md",
    "R12_ER_RELATION_TENSOR_ADAPTER_PREREG.md",
    "R12_ER_RELATION_TENSOR_BOARD_PREREG.md",
    "R12_ER_RELATION_TENSOR_BOARD_RECEIPT.md",
    "pipeline/build_er_relation_tensor_board.py",
    "pipeline/er_relation_tensor_renderers.py",
    "train/er_relation_tensor_motor.py",
    "train/er_relation_tensor_adapter.py",
    "train/pilot_er_relation_tensor_adapter.py",
    "train/er_relation_tensor_training.py",
    "train/pilot_er_relation_tensor.py",
    "train/assess_er_relation_tensor.py",
    "train/jobs/er_relation_tensor.sbatch",
    "train/er_cst_fresh.py",
    "train/er_cst_witness_equality_bus.py",
    "train/pilot_er_cst_witness_equality_bus.py",
)


def runtime_manifest() -> dict[str, object]:
    value: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        value.update(
            {
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("ER-TT scientific source commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("ER-TT scientific source is not an ancestor")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"ER-TT source omits frozen path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"ER-TT runtime differs from source: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def atomic_json_save(value: object, path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def consume_development_access(
    data_dir: Path, board: Mapping[str, object], source_commit: str
) -> dict[str, str]:
    split_sha = str(board["files"]["development.jsonl"]["sha256"])
    payload = (
        json.dumps(
            {
                "schema": ACCESS_SCHEMA,
                "protocol": PROTOCOL,
                "split": DEVELOPMENT_SPLIT,
                "board_report_sha256": BOARD_REPORT_SHA256,
                "split_sha256": split_sha,
                "scientific_source_commit": source_commit,
                "board_source_commit": BOARD_SOURCE_COMMIT,
                "access_number": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    directory = data_dir / "access"
    directory.mkdir(exist_ok=True)
    path = directory / f"er_relation_tensor_development_{split_sha}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def initialize_system(
    args: argparse.Namespace, device: torch.device
) -> tuple[object, dict[str, int], str, dict[str, object]]:
    return initialize_er_relation_tensor(
        joint_checkpoint=args.joint_checkpoint,
        physical_checkpoint=args.physical_checkpoint,
        v1_checkpoint=args.v1_checkpoint,
        v1_2_checkpoint=args.v1_2_checkpoint,
        confirmed_checkpoint=args.confirmed_checkpoint,
        confirmation_assessment=args.confirmation_assessment,
        witness_checkpoint=args.witness_checkpoint,
        witness_confirmation_assessment=args.witness_confirmation_assessment,
        seed=args.seed,
        device=device,
    )


def release_cuda(module: object) -> None:
    if isinstance(module, torch.nn.Module):
        module.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def _minimum_group(metrics: Mapping[str, object], group: str, field: str) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise RuntimeError(f"ER-TT {group} metrics are absent")
    return min(float(value[field]["rate"]) for value in values.values())


def compute_gates(
    metrics: Mapping[str, Mapping[str, object]], checkpoint: Mapping[str, object]
) -> dict[str, bool]:
    treatment = metrics["treatment"]
    controls = [metrics["family_deranged"], metrics["equality_ablated"]]
    overall = treatment["overall"]
    invariance = treatment["invariance"]
    interventions = treatment["interventions"]
    return {
        "treatment_packet_state_answer_joint_at_least_90pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS[f"{name}_overall"]
            for name in ("packet", "state", "answer", "joint")
        ),
        "all_semantic_fields_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["field_overall"]
            for name in (
                "cardinality",
                "initial_rows",
                "relation_rows",
                "rule_active",
                "events",
                "halt",
                "query",
            )
        ),
        "all_active_pointers_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["pointer_overall"]
            for name in (
                "line_pointer",
                "binding_pointer",
                "initial_pointer",
                "witness_pointer",
                "query_pointer",
            )
        ),
        "minimum_cardinality_joint_at_least_80pct": _minimum_group(
            treatment, "by_cardinality", "joint"
        )
        >= THRESHOLDS["joint_min_cardinality"],
        "minimum_depth_joint_at_least_80pct": _minimum_group(
            treatment, "by_depth", "joint"
        )
        >= THRESHOLDS["joint_min_depth"],
        "minimum_renderer_joint_at_least_80pct": _minimum_group(
            treatment, "by_renderer", "joint"
        )
        >= THRESHOLDS["joint_min_renderer"],
        "non_bijective_joint_at_least_85pct": float(
            treatment["non_bijective"]["joint"]["rate"]
        )
        >= THRESHOLDS["joint_non_bijective"],
        "treatment_packet_and_joint_advantage_at_least_50pp": all(
            float(overall[name]["rate"])
            - float(control["overall"][name]["rate"])
            >= THRESHOLDS["treatment_advantage"]
            for control in controls
            for name in ("packet", "joint")
        ),
        "negative_packet_and_joint_at_most_35pct": all(
            float(control["overall"][name]["rate"]) <= THRESHOLDS["negative_max"]
            for control in controls
            for name in ("packet", "joint")
        ),
        "all_source_invariances_exact": all(
            int(value["exact"]) == int(value["rows"]) == 2_048
            for value in invariance.values()
        ),
        "all_causal_interventions_exact_and_effective": all(
            int(value["eligible"]) > 0
            and int(value["exact_on_eligible"]) == int(value["eligible"])
            and int(value["sensitive"]) > 0
            and int(value["changed_on_sensitive"]) == int(value["sensitive"])
            for value in interventions.values()
        ),
        "confirmed_parent_unchanged": all(
            arm["fit"]["frozen_parent_unchanged"] is True
            for arm in checkpoint["arms"].values()
        ),
        "parameter_certificate_exact_and_below_200m": checkpoint["parameters"]
        == EXPECTED_PARAMETERS,
        "motor_and_reader_are_parameter_free": all(
            arm["fit"]["motor_parameters"] == 0
            and arm["fit"]["reader_parameters"] == 0
            for arm in checkpoint["arms"].values()
        ),
        "development_one_confirmation_zero": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmed-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-assessment", type=Path, required=True)
    parser.add_argument("--witness-checkpoint", type=Path, required=True)
    parser.add_argument("--witness-confirmation-assessment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing existing ER-TT output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("ER-TT qualification requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise SystemExit("ER-TT board identity differs")
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda")
    fit_seed = derived_seed(args.seed, "shared-fit-order-and-control")
    arms: dict[str, dict[str, object]] = {}
    parameters: dict[str, int] | None = None
    parent_receipt: dict[str, object] | None = None
    initial_digest: str | None = None

    for arm_name in TRAINING_CONTRACT["arms"]:
        model, arm_parameters, frozen_digest, receipt = initialize_system(args, device)
        current_initial = state_dict_digest(trainable_state(model))
        if initial_digest is None:
            initial_digest = current_initial
            parameters = arm_parameters
            parent_receipt = receipt
        elif (
            current_initial != initial_digest
            or arm_parameters != parameters
            or receipt != parent_receipt
        ):
            raise RuntimeError("ER-TT arms differ at initialization")
        declared = frozenset(receipt["trainable_names"])
        fit = fit_arm(
            model,
            train_rows,
            seed=fit_seed,
            arm=str(arm_name),
            frozen_digest=frozen_digest,
            digest_fn=lambda value, names=declared: frozen_state_digest(value, names),
        )
        arms[str(arm_name)] = {
            "fit": fit,
            "compiler_trainable_state": trainable_state(model),
        }
        release_cuda(model)

    if parameters is None or parent_receipt is None or initial_digest is None:
        raise RuntimeError("ER-TT produced no fitted arms")
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": args.source_commit,
        "source_manifest": source,
        "board_source_commit": BOARD_SOURCE_COMMIT,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "board_files": board["files"],
        "training_seed": args.seed,
        "shared_initial_state_sha256": initial_digest,
        "training_contract": TRAINING_CONTRACT,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "arms": arms,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    checkpoint_path = args.out_dir / "compiler.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    ledger = consume_development_access(args.data_dir, board, args.source_commit)
    development_rows = load_split(
        args.data_dir,
        board,
        filename="development.jsonl",
        split=DEVELOPMENT_SPLIT,
        expected=2_048,
    )
    metrics: dict[str, Mapping[str, object]] = {}
    raw_evidence: dict[str, object] = {}
    for arm_name in TRAINING_CONTRACT["arms"]:
        model, arm_parameters, _, receipt = initialize_system(args, device)
        if arm_parameters != parameters or receipt != parent_receipt:
            raise RuntimeError("ER-TT evaluation reconstruction differs")
        load_trainable_state(model, arms[str(arm_name)]["compiler_trainable_state"])
        result = evaluate_arm(
            model,
            development_rows,
            batch_size=args.batch_size,
            include_raw=True,
            include_invariances=arm_name == "treatment",
        )
        raw_evidence[str(arm_name)] = result.pop("raw")
        metrics[str(arm_name)] = result
        release_cuda(model)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": args.source_commit,
        "checkpoint_sha256": checkpoint_sha,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "development_sha256": board["files"]["development.jsonl"]["sha256"],
        "arms": raw_evidence,
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    evidence_path = args.out_dir / "development_evidence.pt"
    atomic_torch_save(evidence, evidence_path)
    evidence_sha = sha256_file(evidence_path)
    gates = compute_gates(metrics, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": args.source_commit,
        "runtime": runtime_manifest(),
        "training_seed": args.seed,
        "training_contract": TRAINING_CONTRACT,
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_one_sealed_confirmation"
            if all(gates.values())
            else "reject_er_relation_tensor_v1"
        ),
        "artifacts": {
            "checkpoint_sha256": checkpoint_sha,
            "evidence_sha256": evidence_sha,
            "development_ledger_sha256": ledger["sha256"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "development_ledger": ledger,
        },
        "claim_boundary": (
            "Passing establishes bounded variable-cardinality episodic relation "
            "compilation and source-deleted parameter-free tensor composition. "
            "It does not establish free-form grounding, unbounded algorithms, "
            "arithmetic, planning, or broad general reasoning."
        ),
    }
    report_path = args.out_dir / "development_report.json"
    atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checkpoint_sha256": checkpoint_sha,
                "evidence_sha256": evidence_sha,
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
