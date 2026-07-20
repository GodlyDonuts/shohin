#!/usr/bin/env python3
"""Fit three ER-CST-WEB arms and consume one fresh development read."""

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

from build_er_cst_witness_equality_board import (
    DEVELOPMENT_SPLIT,
    PROTOCOL,
    TRAIN_SPLIT,
)
from er_cst_fresh import (
    canonical_json,
    derived_seed,
    load_trainable_state,
    module_state,
    trainable_state,
)
from er_cst_witness_equality import (
    TRAINING_CONTRACT,
    evaluate_arm,
    fit_arm,
    load_board_receipt,
    load_split,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_cst_witness_equality_bus import initialize_er_cst_witness_equality
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from sd_cst import CategoricalStateReader


CHECKPOINT_SCHEMA = "r12_er_cst_witness_equality_checkpoint_v1_1"
EVIDENCE_SCHEMA = "r12_er_cst_witness_equality_development_evidence_v1_1"
REPORT_SCHEMA = "r12_er_cst_witness_equality_development_report_v1_1"
ACCESS_SCHEMA = "r12_er_cst_witness_equality_development_access_v1_1"
BOARD_SOURCE_COMMIT = "5670ad83ae4e5806ec351337997c16990f2b5452"
BOARD_REPORT_SHA256 = (
    "22cb355e58e9f3b8125a57c60c7aafb7aadead4406abc426da368e3b3b2cff75"
)
THRESHOLDS = {
    "packet_overall": 0.90,
    "state_overall": 0.90,
    "answer_overall": 0.90,
    "joint_overall": 0.90,
    "joint_min_renderer": 0.85,
    "joint_min_depth": 0.80,
    "field_overall": 0.95,
    "pointer_overall": 0.90,
    "treatment_packet_advantage": 0.50,
    "treatment_joint_advantage": 0.50,
    "negative_packet_max": 0.35,
    "negative_state_max": 0.40,
}
FROZEN_SOURCE_PATHS = (
    "R12_ER_CST_WITNESS_EQUALITY_BUS_PREREG.md",
    "pipeline/build_er_cst_fresh_board.py",
    "pipeline/er_cst_fresh_renderers.py",
    "pipeline/build_er_cst_witness_equality_board.py",
    "train/er_cst_rule_card_adapter.py",
    "train/pilot_er_cst_rule_card_adapter.py",
    "train/er_cst_fresh.py",
    "train/er_cst_witness_equality_bus.py",
    "train/er_cst_witness_equality.py",
    "train/pilot_er_cst_witness_equality_bus.py",
    "train/pilot_er_cst_witness_equality.py",
    "train/assess_er_cst_witness_equality.py",
    "train/jobs/er_cst_witness_equality.sbatch",
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
        raise RuntimeError("ER-CST witness scientific source commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("ER-CST witness scientific source is not an ancestor")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"ER-CST witness source omits frozen path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"ER-CST witness runtime differs from source: {relative}")
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
    path = directory / f"er_cst_witness_development_{split_sha}.json"
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
) -> tuple[object, object, CategoricalStateReader, dict[str, int], str, dict[str, object]]:
    model, motor, parameters, excluded_digest, receipt = (
        initialize_er_cst_witness_equality(
            joint_checkpoint=args.joint_checkpoint,
            physical_checkpoint=args.physical_checkpoint,
            v1_checkpoint=args.v1_checkpoint,
            v1_2_checkpoint=args.v1_2_checkpoint,
            confirmed_checkpoint=args.confirmed_checkpoint,
            confirmation_assessment=args.confirmation_assessment,
            seed=args.seed,
            device=device,
        )
    )
    torch.manual_seed(derived_seed(args.seed, "er-cst-reader"))
    reader = CategoricalStateReader().to(device)
    return model, motor, reader, parameters, excluded_digest, receipt


def release_cuda(*modules: object) -> None:
    for module in modules:
        if isinstance(module, torch.nn.Module):
            module.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def _minimum_group(metrics: Mapping[str, object], group: str, field: str) -> float:
    values = metrics[group]
    if not isinstance(values, Mapping) or not values:
        raise RuntimeError(f"ER-CST witness {group} metrics are absent")
    return min(float(value[field]["rate"]) for value in values.values())


def compute_gates(
    metrics: Mapping[str, Mapping[str, object]],
    checkpoint: Mapping[str, object],
) -> dict[str, bool]:
    treatment = metrics["treatment"]
    controls = [metrics["family_deranged"], metrics["equality_ablated"]]
    overall = treatment["overall"]
    return {
        "treatment_packet_at_least_90pct": float(overall["packet"]["rate"])
        >= THRESHOLDS["packet_overall"],
        "treatment_state_answer_joint_at_least_90pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS[f"{name}_overall"]
            for name in ("state", "answer", "joint")
        ),
        "treatment_min_renderer_joint_at_least_85pct": _minimum_group(
            treatment, "by_renderer", "joint"
        )
        >= THRESHOLDS["joint_min_renderer"],
        "treatment_min_depth_joint_at_least_80pct": _minimum_group(
            treatment, "by_depth", "joint"
        )
        >= THRESHOLDS["joint_min_depth"],
        "all_packet_fields_at_least_95pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["field_overall"]
            for name in ("initial", "cards", "events", "halt", "query")
        ),
        "all_pointers_at_least_90pct": all(
            float(overall[name]["rate"]) >= THRESHOLDS["pointer_overall"]
            for name in (
                "line_pointer",
                "binding_pointer",
                "initial_pointer",
                "witness_pointer",
                "query_pointer",
            )
        ),
        "treatment_packet_advantage_at_least_50pp": all(
            float(overall["packet"]["rate"])
            - float(control["overall"]["packet"]["rate"])
            >= THRESHOLDS["treatment_packet_advantage"]
            for control in controls
        ),
        "treatment_joint_advantage_at_least_50pp": all(
            float(overall["joint"]["rate"])
            - float(control["overall"]["joint"]["rate"])
            >= THRESHOLDS["treatment_joint_advantage"]
            for control in controls
        ),
        "negative_packets_at_most_35pct": all(
            float(control["overall"]["packet"]["rate"])
            <= THRESHOLDS["negative_packet_max"]
            for control in controls
        ),
        "negative_states_at_most_40pct": all(
            float(control["overall"]["state"]["rate"])
            <= THRESHOLDS["negative_state_max"]
            for control in controls
        ),
        "finite_motor_and_reader_certificates_exact": all(
            arm["fit"]["certificate"]["motor_exact"] == 36
            and arm["fit"]["certificate"]["reader_exact"] == 18
            for arm in checkpoint["arms"].values()
        ),
        "confirmed_parent_unchanged": all(
            arm["fit"]["frozen_parent_unchanged"] is True
            for arm in checkpoint["arms"].values()
        ),
        "complete_system_below_200m": int(checkpoint["parameters"]["complete_system"])
        < 200_000_000,
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing existing ER-CST witness output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("ER-CST witness qualification requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if (
        board.get("source_commit") != BOARD_SOURCE_COMMIT
        or board.get("report_sha256") != BOARD_REPORT_SHA256
    ):
        raise SystemExit("ER-CST witness board identity differs")
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
        model, motor, reader, arm_parameters, frozen_digest, receipt = initialize_system(
            args, device
        )
        current_initial = state_dict_digest(
            {
                **{
                    f"compiler.{name}": value
                    for name, value in trainable_state(model).items()
                },
                **{
                    f"motor.{name}": value
                    for name, value in module_state(motor).items()
                },
                **{
                    f"reader.{name}": value
                    for name, value in module_state(reader).items()
                },
            }
        )
        if initial_digest is None:
            initial_digest = current_initial
            parameters = arm_parameters
            parent_receipt = receipt
        elif (
            current_initial != initial_digest
            or arm_parameters != parameters
            or receipt != parent_receipt
        ):
            raise RuntimeError("ER-CST witness arms differ at initialization")
        declared = frozenset(receipt["trainable_names"])
        fit = fit_arm(
            model,
            motor,
            reader,
            train_rows,
            seed=fit_seed,
            arm=str(arm_name),
            frozen_digest=frozen_digest,
            digest_fn=lambda value, names=declared: frozen_state_digest(value, names),
        )
        arms[str(arm_name)] = {
            "fit": fit,
            "compiler_trainable_state": trainable_state(model),
            "motor_state": module_state(motor),
            "reader_state": module_state(reader),
        }
        release_cuda(model, motor, reader)

    if parameters is None or parent_receipt is None or initial_digest is None:
        raise RuntimeError("ER-CST witness produced no fitted arms")
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
        model, motor, reader, arm_parameters, _, receipt = initialize_system(args, device)
        if arm_parameters != parameters or receipt != parent_receipt:
            raise RuntimeError("ER-CST witness evaluation reconstruction differs")
        load_trainable_state(model, arms[str(arm_name)]["compiler_trainable_state"])
        motor.load_state_dict(arms[str(arm_name)]["motor_state"], strict=True)
        reader.load_state_dict(arms[str(arm_name)]["reader_state"], strict=True)
        result = evaluate_arm(
            model,
            motor,
            reader,
            development_rows,
            batch_size=args.batch_size,
            include_raw=True,
        )
        raw_evidence[str(arm_name)] = result.pop("raw")
        metrics[str(arm_name)] = result
        release_cuda(model, motor, reader)

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
            else "reject_er_cst_witness_equality_v1_1"
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
            "Passing establishes bounded fresh episodic S3 rule inference by "
            "learned witness equality, source-deleted categorical composition, "
            "internal halt, and late-query readout. It does not establish "
            "unrestricted language grounding, arbitrary algorithms, arithmetic, "
            "planning, or broad general reasoning."
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
