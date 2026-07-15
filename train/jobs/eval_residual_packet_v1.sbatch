#!/bin/bash
# Frozen RSP-C1 raw-transcript acquisition. All code, checkpoints, manifests,
# and data contracts are copied into a verified job-local snapshot. The
# evaluator writes an exclusive read-only transcript; this wrapper independently
# validates its identity/cardinality and emits an immutable acquisition receipt.
#SBATCH -J shohin-eval-rsp-c1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_rsp_c1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

echo "RSP-C1 is permanently closed; this wrapper is retained as audit evidence only" >&2
exit 64

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
RUN_MANIFEST=${RUN_MANIFEST:?set the frozen rsp_c1_eval_run_manifest_v1 JSON}
RUN_MANIFEST_SHA256=${RUN_MANIFEST_SHA256:?set its precommitted SHA-256}
SEED=${SEED:?set paired fit seed 2026071511 or 2026071512}
TRANSCRIPT_OUT=${TRANSCRIPT_OUT:?set a fresh transcript path below EVAL_ROOT}
EVAL_ROOT=${EVAL_ROOT:-$BASE/artifacts/evals/rsp_c1}

case "$SEED" in
  2026071511|2026071512) ;;
  *) echo "[rsp-eval] seed is outside the frozen paired-seed set: $SEED" >&2; exit 2 ;;
esac
for path in "$PY" "$RUN_MANIFEST"; do
  [ -s "$path" ] || { echo "[rsp-eval] missing required input: $path" >&2; exit 2; }
done
PY=$(realpath -e -- "$PY")
BASE=$(realpath -e -- "$BASE")
RUN_MANIFEST=$(realpath -e -- "$RUN_MANIFEST")
SCRIPT_PATH=$(realpath -e -- "${BASH_SOURCE[0]}")

mkdir -p -- "$EVAL_ROOT"
EVAL_ROOT=$(realpath -e -- "$EVAL_ROOT")
TRANSCRIPT_OUT=$(realpath -m -- "$TRANSCRIPT_OUT")
if [[ "$TRANSCRIPT_OUT" != "$EVAL_ROOT/"* ]]; then
  echo "[rsp-eval] TRANSCRIPT_OUT must be a child of EVAL_ROOT=$EVAL_ROOT" >&2
  exit 3
fi
if [ -e "$TRANSCRIPT_OUT" ] || [ -L "$TRANSCRIPT_OUT" ]; then
  echo "[rsp-eval] refusing existing transcript: $TRANSCRIPT_OUT" >&2
  exit 3
fi
TRANSCRIPT_RECEIPT=$TRANSCRIPT_OUT.receipt.json
if [ -e "$TRANSCRIPT_RECEIPT" ] || [ -L "$TRANSCRIPT_RECEIPT" ]; then
  echo "[rsp-eval] refusing existing transcript receipt: $TRANSCRIPT_RECEIPT" >&2
  exit 3
fi
mkdir -p -- "$(dirname "$TRANSCRIPT_OUT")"
OUT_PARENT=$(realpath -e -- "$(dirname "$TRANSCRIPT_OUT")")
if [[ "$OUT_PARENT" != "$EVAL_ROOT" && "$OUT_PARENT" != "$EVAL_ROOT/"* ]]; then
  echo "[rsp-eval] resolved output parent escaped EVAL_ROOT" >&2
  exit 3
fi
TRANSCRIPT_OUT=$OUT_PARENT/$(basename "$TRANSCRIPT_OUT")
TRANSCRIPT_RECEIPT=$TRANSCRIPT_OUT.receipt.json

SCRATCH_PARENT=${SLURM_TMPDIR:-/tmp}
SNAPSHOT_ROOT=$(mktemp -d "$SCRATCH_PARENT/rsp_eval_${SLURM_JOB_ID:-manual}.XXXXXX")
trap 'chmod -R u+w -- "$SNAPSHOT_ROOT" 2>/dev/null || true; rm -rf -- "$SNAPSHOT_ROOT"' EXIT
cd "$SNAPSHOT_ROOT"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=$SEED
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" - "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256" "$SEED" "$SNAPSHOT_ROOT" "$SCRIPT_PATH" <<'PY'
# EVAL_CUSTODY_PY_BEGIN
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import tokenizers

EVAL_RUN_SCHEMA = "rsp_c1_eval_run_manifest_v1"
FIT_RECEIPT_SCHEMA = "rsp_c1_fit_receipt_v1"
CONTROLLER_SCHEMA = "rsp_c1_controller_pair_manifest_v1"
EXECUTOR_SCHEMA = "rsp_c1_executor_manifest_v1"
RESOURCE_SCHEMA = "rsp_c1_training_resources_v1"
GENERATION_SCHEMA = "residual_packet_generation_manifest_v1"
AUDIT_SCHEMA = "residual_packet_admission_audit_v1"
BOARD_SCHEMA = "residual_packet_board_v1"
TRANSCRIPT_SCHEMA = "residual_packet_v1_raw_transcript"
PREFLIGHT_SCHEMA = "rsp_c1_eval_preflight_v1"
ACQUISITION_SCHEMA = "rsp_c1_transcript_receipt_v1"
FIT_SEEDS = (2026071511, 2026071512)
GENERATION_SEEDS = {"board": 2026071503, "training": 2026071504, "observation": 2026071505, "sham": 2026071506}
TRAIN_LENGTH_COUNTS = {"2": 1024, "3": 2048, "4": 1024}
BOARD_SEED = 2026071503
BOARD_SHA256 = "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
BOARD_ROWS_SHA256 = "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
EXECUTOR_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
FIT = {"epochs": 10, "pack_len": 128, "batch_size": 64, "lr_muon": 8e-4, "lr_adam": 2e-4, "warmup": 50, "clip": 1.0}
SOURCE_KEYS = {"job", "evaluator", "model", "protocol"}
ARTIFACT_KEYS = {
    "preregistration", "controller_manifest", "executor_manifest", "board",
    "admission_audit", "training_manifest", "tokenizer", "executor", "treatment",
    "sham", "training_resources", "treatment_receipt", "sham_receipt",
}
REF_KEYS = {"path", "sha256"}
FIT_RECEIPT_KEYS = {
    "schema", "arm", "paired_seed", "run_manifest_sha256", "checkpoint_step",
    "artifacts", "sources", "fit", "accounting", "pair_geometry_sha256",
    "runtime", "slurm_job_id",
}
FIT_ARTIFACT_KEYS = {
    "checkpoint_sha256", "metadata_sha256", "pair_equality_sha256", "init_sha256",
    "tokenizer_sha256", "data_sha256", "treatment_data_sha256", "sham_data_sha256",
    "training_manifest_sha256", "admission_audit_sha256", "preregistration_sha256",
}
FIT_SOURCE_KEYS = {"job_sha256", "trainer_sha256", "encoding_sha256", "model_sha256", "muon_sha256", "protocol_sha256"}
ACCOUNTING_KEYS = {
    "full_tokens_before_packing", "packed_sequences", "discarded_tail_tokens",
    "steps_per_epoch", "optimizer_updates", "packed_forward_token_positions",
    "supervised_completion_tokens",
}
AUDIT_ROW_KEYS = {
    "compiler_controls_matched", "correct_arithmetic_updater_transitions",
    "evaluation_answer_response_occurrences", "malformed_compiler_rows",
    "malformed_updater_rows", "reconstructed_sham_mapping_mismatches",
    "source_final_answer_response_occurrences", "updater_pairs_byte_identical",
}
TOKEN_ACCOUNTING_KEYS = {
    "compiler_rows", "examples", "forward_token_count", "full_token_count",
    "packed_sequence_count", "prompt_token_count", "response_token_count",
    "skipped_too_long", "supervised_token_count", "updater_rows",
}
CONTROLLER_COMMON_KEYS = {
    "init_sha256", "tokenizer_sha256", "preregistration_sha256",
    "training_manifest_sha256", "admission_audit_sha256", "sft_run_manifest_sha256",
    "fit", "sources", "pair_geometry_sha256", "runtime_contract_sha256",
}
CONTROLLER_ARM_KEYS = {"checkpoint_sha256", "fit_receipt_sha256", "data_sha256"}
RESOURCE_ARM_KEYS = {
    "checkpoint_sha256", "fit_receipt_sha256", "supervised_completion_tokens",
    "packed_forward_token_positions", "optimizer_updates",
}
DESTINATIONS = {
    "job": "train/jobs/eval_residual_packet_v1.sbatch",
    "evaluator": "train/eval_residual_packet_v1.py",
    "model": "train/model.py",
    "protocol": "train/residual_packet_protocol.py",
    "preregistration": "inputs/preregistration.md",
    "controller_manifest": "inputs/controller_manifest.json",
    "executor_manifest": "inputs/executor_manifest.json",
    "board": "inputs/board.json",
    "admission_audit": "inputs/admission_audit.json",
    "training_manifest": "inputs/training_manifest.json",
    "tokenizer": "inputs/tokenizer.json",
    "executor": "inputs/executor.pt",
    "treatment": "inputs/treatment.pt",
    "sham": "inputs/sham.pt",
    "training_resources": "inputs/training_resources.json",
    "treatment_receipt": "inputs/treatment_receipt.json",
    "sham_receipt": "inputs/sham_receipt.json",
}


def require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} keys differ: {actual} != {sorted(expected)}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable JSON write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def validate_ref(ref: object, label: str) -> Mapping[str, str]:
    value = require_exact_keys(ref, REF_KEYS, label)
    path, digest = value["path"], value["sha256"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path must be nonempty")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    return {"path": path, "sha256": digest}


def resolve_ref(ref: Mapping[str, str], directory: Path) -> Path:
    path = Path(ref["path"])
    if not path.is_absolute():
        path = directory / path
    return path.resolve(strict=True)


def copy_verified(source: Path, destination: Path, expected: str) -> dict[str, Any]:
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        with source.open("rb") as reader:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short snapshot write")
                    view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    after = source.stat()
    if digest.hexdigest() != expected or sha256_file(destination) != expected or sha256_file(source) != expected:
        raise ValueError(f"snapshot/source hash mismatch for {source}")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while it was snapshotted: {source}")
    return {"source": str(source), "snapshot": str(destination), "sha256": expected, "size": destination.stat().st_size, "source_device": before.st_dev, "source_inode": before.st_ino}


def validate_run_manifest(manifest: object, seed: int) -> Mapping[str, Any]:
    value = require_exact_keys(manifest, {"schema", "frozen", "paired_seed", "sources", "artifacts"}, "eval run manifest")
    if value["schema"] != EVAL_RUN_SCHEMA or value["frozen"] is not True or value["paired_seed"] != seed or seed not in FIT_SEEDS:
        raise ValueError("eval run manifest identity differs")
    sources = require_exact_keys(value["sources"], SOURCE_KEYS, "eval run sources")
    artifacts = require_exact_keys(value["artifacts"], ARTIFACT_KEYS, "eval run artifacts")
    for label, ref in {**sources, **artifacts}.items():
        validate_ref(ref, label)
    return value


def validate_fit_receipt(receipt: object, arm: str, seed: int) -> Mapping[str, Any]:
    value = require_exact_keys(receipt, FIT_RECEIPT_KEYS, f"{arm} fit receipt")
    if value["schema"] != FIT_RECEIPT_SCHEMA or value["arm"] != arm or value["paired_seed"] != seed or value["checkpoint_step"] != "sft_ep10":
        raise ValueError(f"{arm} fit receipt identity differs")
    if value["fit"] != FIT:
        raise ValueError(f"{arm} fit settings differ")
    artifacts = require_exact_keys(value["artifacts"], FIT_ARTIFACT_KEYS, f"{arm} receipt artifacts")
    sources = require_exact_keys(value["sources"], FIT_SOURCE_KEYS, f"{arm} receipt sources")
    digest_fields = {
        "run_manifest_sha256": value["run_manifest_sha256"],
        "pair_geometry_sha256": value["pair_geometry_sha256"],
        **artifacts,
        **sources,
    }
    for name, digest in digest_fields.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{arm} fit receipt has invalid named digest: {name}")
    accounting = require_exact_keys(value["accounting"], ACCOUNTING_KEYS, f"{arm} receipt accounting")
    if any(type(accounting[name]) is not int or accounting[name] < 0 for name in ACCOUNTING_KEYS):
        raise ValueError(f"{arm} accounting is not exact nonnegative integers")
    if accounting["optimizer_updates"] != accounting["steps_per_epoch"] * FIT["epochs"]:
        raise ValueError(f"{arm} optimizer-update accounting differs")
    if not isinstance(value["runtime"], Mapping) or not value["runtime"] or not isinstance(value["slurm_job_id"], str) or not value["slurm_job_id"].isdigit():
        raise ValueError(f"{arm} runtime/job receipt is incomplete")
    return value


def stable_runtime_contract(runtime: object, seed: int) -> dict[str, Any]:
    value = require_exact_keys(
        runtime,
        {
            "python_executable", "python_executable_sha256", "python_version", "platform",
            "numpy", "torch", "tokenizers", "cuda_runtime", "cudnn", "gpu",
            "nvidia_smi", "deterministic_algorithms_requested", "cublas_workspace_config",
            "pythonhashseed",
        },
        "fit runtime fingerprint",
    )
    if value["deterministic_algorithms_requested"] is not True or value["cublas_workspace_config"] != ":4096:8" or value["pythonhashseed"] != str(seed):
        raise ValueError("fit deterministic-runtime settings differ")
    gpu = value["gpu"]
    if not isinstance(gpu, list) or len(gpu) != 1:
        raise ValueError("fit receipt must describe exactly one GPU")
    gpu_item = require_exact_keys(gpu[0], {"index", "name", "capability", "total_memory"}, "fit GPU fingerprint")
    driver_models = []
    if not isinstance(value["nvidia_smi"], list):
        raise ValueError("fit nvidia-smi fingerprint is malformed")
    for line in value["nvidia_smi"]:
        if not isinstance(line, str) or len(line.split(",", 1)) != 2:
            raise ValueError("fit nvidia-smi line is malformed")
        driver_models.append(line.split(",", 1)[1].strip())
    return {
        key: value[key]
        for key in (
            "python_executable_sha256", "python_version", "platform", "numpy", "torch",
            "tokenizers", "cuda_runtime", "cudnn", "cublas_workspace_config",
            "pythonhashseed",
        )
    } | {
        "gpu": {"name": gpu_item["name"], "capability": gpu_item["capability"], "total_memory": gpu_item["total_memory"]},
        "driver_models": sorted(set(driver_models)),
    }


def validate_pair_contract(
    controller: object,
    executor_manifest: object,
    resources: object,
    treatment_receipt: object,
    sham_receipt: object,
    hashes: Mapping[str, str],
    seed: int,
) -> None:
    treatment = validate_fit_receipt(treatment_receipt, "treatment", seed)
    sham = validate_fit_receipt(sham_receipt, "sham", seed)
    common_artifacts = {
        "init_sha256", "tokenizer_sha256", "treatment_data_sha256", "sham_data_sha256",
        "training_manifest_sha256", "admission_audit_sha256", "preregistration_sha256",
    }
    for name in common_artifacts:
        if treatment["artifacts"][name] != sham["artifacts"][name]:
            raise ValueError(f"paired receipts differ on {name}")
    expected_receipt_bindings = {
        "tokenizer_sha256": hashes["tokenizer"],
        "preregistration_sha256": hashes["preregistration"],
        "training_manifest_sha256": hashes["training_manifest"],
        "admission_audit_sha256": hashes["admission_audit"],
    }
    if any(treatment["artifacts"][name] != expected for name, expected in expected_receipt_bindings.items()):
        raise ValueError("fit receipt artifact named bindings differ from eval inputs")
    if treatment["sources"] != sham["sources"] or treatment["fit"] != sham["fit"] or treatment["accounting"] != sham["accounting"] or treatment["pair_geometry_sha256"] != sham["pair_geometry_sha256"] or treatment["run_manifest_sha256"] != sham["run_manifest_sha256"]:
        raise ValueError("paired fit receipts differ on common execution contract")
    treatment_runtime = stable_runtime_contract(treatment["runtime"], seed)
    sham_runtime = stable_runtime_contract(sham["runtime"], seed)
    if treatment_runtime != sham_runtime:
        raise ValueError("paired fit runtime contracts differ")
    if treatment["artifacts"]["data_sha256"] != treatment["artifacts"]["treatment_data_sha256"]:
        raise ValueError("treatment receipt active-data binding is wrong")
    if sham["artifacts"]["data_sha256"] != sham["artifacts"]["sham_data_sha256"]:
        raise ValueError("sham receipt active-data binding is wrong")
    if treatment["artifacts"]["checkpoint_sha256"] != hashes["treatment"] or sham["artifacts"]["checkpoint_sha256"] != hashes["sham"]:
        raise ValueError("checkpoint paths are swapped or do not match their arm receipts")
    if hashes["treatment_receipt"] == hashes["sham_receipt"] or hashes["treatment"] == hashes["sham"]:
        raise ValueError("paired arms must use distinct receipts and checkpoints")

    controller = require_exact_keys(controller, {"schema", "paired_seed", "common", "arms", "training_resources_sha256"}, "controller manifest")
    if controller["schema"] != CONTROLLER_SCHEMA or controller["paired_seed"] != seed:
        raise ValueError("controller manifest identity differs")
    common = require_exact_keys(controller["common"], CONTROLLER_COMMON_KEYS, "controller common")
    expected_common = {
        "init_sha256": treatment["artifacts"]["init_sha256"],
        "tokenizer_sha256": hashes["tokenizer"],
        "preregistration_sha256": hashes["preregistration"],
        "training_manifest_sha256": hashes["training_manifest"],
        "admission_audit_sha256": hashes["admission_audit"],
        "sft_run_manifest_sha256": treatment["run_manifest_sha256"],
        "fit": FIT,
        "sources": treatment["sources"],
        "pair_geometry_sha256": treatment["pair_geometry_sha256"],
        "runtime_contract_sha256": canonical_digest(treatment_runtime),
    }
    if dict(common) != expected_common:
        raise ValueError("controller common named bindings differ")
    arms = require_exact_keys(controller["arms"], {"treatment", "sham"}, "controller arms")
    for arm, receipt in (("treatment", treatment), ("sham", sham)):
        item = require_exact_keys(arms[arm], CONTROLLER_ARM_KEYS, f"controller {arm} arm")
        expected = {
            "checkpoint_sha256": hashes[arm],
            "fit_receipt_sha256": hashes[f"{arm}_receipt"],
            "data_sha256": receipt["artifacts"]["data_sha256"],
        }
        if dict(item) != expected:
            raise ValueError(f"controller {arm} named bindings differ")
    if controller["training_resources_sha256"] != hashes["training_resources"]:
        raise ValueError("controller training-resource binding differs")

    resources = require_exact_keys(resources, {"schema", "paired_seed", "treatment", "sham"}, "training resources")
    if resources["schema"] != RESOURCE_SCHEMA or resources["paired_seed"] != seed:
        raise ValueError("training-resource identity differs")
    for arm, receipt in (("treatment", treatment), ("sham", sham)):
        item = require_exact_keys(resources[arm], RESOURCE_ARM_KEYS, f"training resources {arm}")
        accounting = receipt["accounting"]
        expected = {
            "checkpoint_sha256": hashes[arm],
            "fit_receipt_sha256": hashes[f"{arm}_receipt"],
            "supervised_completion_tokens": accounting["supervised_completion_tokens"],
            "packed_forward_token_positions": accounting["packed_forward_token_positions"],
            "optimizer_updates": accounting["optimizer_updates"],
        }
        if dict(item) != expected:
            raise ValueError(f"training resources {arm} differ from fit receipt")
    if {key: resources["treatment"][key] for key in ("supervised_completion_tokens", "packed_forward_token_positions", "optimizer_updates")} != {key: resources["sham"][key] for key in ("supervised_completion_tokens", "packed_forward_token_positions", "optimizer_updates")}:
        raise ValueError("paired training-resource exposure differs")

    executor_manifest = require_exact_keys(executor_manifest, {"schema", "checkpoint_sha256", "checkpoint_step", "tokenizer_sha256", "model_sha256", "protocol_sha256"}, "executor manifest")
    expected_executor = {
        "schema": EXECUTOR_SCHEMA,
        "checkpoint_sha256": hashes["executor"],
        "checkpoint_step": 260000,
        "tokenizer_sha256": hashes["tokenizer"],
        "model_sha256": hashes["model"],
        "protocol_sha256": hashes["protocol"],
    }
    if dict(executor_manifest) != expected_executor:
        raise ValueError("executor manifest named bindings differ")
    if treatment["sources"]["model_sha256"] != hashes["model"] or treatment["sources"]["protocol_sha256"] != hashes["protocol"]:
        raise ValueError("fit receipts do not bind the evaluated model/protocol sources")


def validate_generation_audit_board(
    generation: object,
    audit: object,
    board: object,
    hashes: Mapping[str, str],
    treatment_receipt: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    generation = require_exact_keys(generation, {"artifacts", "length_counts", "pack_length", "program_count", "protocol_sha256", "schema", "seeds", "token_accounting", "tokenizer_sha256", "sft_encoding_sha256"}, "generation manifest")
    if generation["schema"] != GENERATION_SCHEMA or generation["pack_length"] != FIT["pack_len"]:
        raise ValueError("generation manifest identity differs")
    if generation["program_count"] != 4096 or generation["length_counts"] != TRAIN_LENGTH_COUNTS or generation["seeds"] != GENERATION_SEEDS:
        raise ValueError("generation manifest program/length/seed contract differs")
    artifacts = require_exact_keys(generation["artifacts"], {"board_sha256", "board_rows_sha256", "sham_rows", "sham_sha256", "treatment_rows", "treatment_sha256"}, "generation artifacts")
    if artifacts["board_sha256"] != hashes["board"] or artifacts["board_rows_sha256"] != BOARD_ROWS_SHA256 or artifacts["treatment_sha256"] != treatment_receipt["artifacts"]["treatment_data_sha256"] or artifacts["sham_sha256"] != treatment_receipt["artifacts"]["sham_data_sha256"]:
        raise ValueError("generation manifest named artifact bindings differ")
    if generation["tokenizer_sha256"] != hashes["tokenizer"] or generation["protocol_sha256"] != hashes["protocol"]:
        raise ValueError("generation manifest tokenizer/protocol binding differs")
    if generation["sft_encoding_sha256"] != treatment_receipt["sources"]["encoding_sha256"]:
        raise ValueError("generation manifest SFT-encoding binding differs from fit receipt")
    token_accounting = require_exact_keys(generation["token_accounting"], {"treatment", "sham"}, "generation token accounting")
    if token_accounting["treatment"] != token_accounting["sham"]:
        raise ValueError("generation token accounting differs between arms")
    accounting = require_exact_keys(token_accounting["treatment"], TOKEN_ACCOUNTING_KEYS, "generation arm accounting")
    if accounting["compiler_rows"] != 4096 or accounting["updater_rows"] != 12288 or accounting["skipped_too_long"] != 0 or accounting["forward_token_count"] != accounting["packed_sequence_count"] * FIT["pack_len"]:
        raise ValueError("generation arm accounting differs")

    audit = require_exact_keys(audit, {"admitted", "artifact_sha256", "board", "failures", "overlap", "row_metrics", "schema", "sham", "token_accounting", "training"}, "admission audit")
    if audit["schema"] != AUDIT_SCHEMA or audit["admitted"] is not True or audit["failures"] != []:
        raise ValueError("admission audit is not an exact passing v1 audit")
    audit_hashes = require_exact_keys(audit["artifact_sha256"], {"board", "manifest", "protocol", "sham", "sft_encoding", "treatment", "tokenizer"}, "audit artifact hashes")
    expected_audit = {
        "board": hashes["board"], "manifest": hashes["training_manifest"],
        "protocol": hashes["protocol"], "sham": artifacts["sham_sha256"],
        "sft_encoding": generation["sft_encoding_sha256"], "treatment": artifacts["treatment_sha256"],
        "tokenizer": hashes["tokenizer"],
    }
    if dict(audit_hashes) != expected_audit:
        raise ValueError("admission audit named artifact bindings differ")
    if any(type(value) is not int or value != 0 for value in require_exact_keys(audit["overlap"], {"complete_trajectory_overlap", "exact_packet_overlap", "exact_source_overlap", "normalized_13_token_source_ngram_overlap", "normalized_semantic_program_overlap", "reserved_renderer_training_occurrences"}, "audit overlap").values()):
        raise ValueError("admission overlap gate failed")
    metrics = require_exact_keys(audit["row_metrics"], AUDIT_ROW_KEYS, "audit row metrics")
    for name in AUDIT_ROW_KEYS - {"compiler_controls_matched", "updater_pairs_byte_identical"}:
        if type(metrics[name]) is not int or metrics[name] != 0:
            raise ValueError(f"admission row gate failed: {name}")
    if metrics["compiler_controls_matched"] != 4096 or metrics["updater_pairs_byte_identical"] != 12288:
        raise ValueError("admission paired-control counts differ")
    if audit["token_accounting"] != token_accounting:
        raise ValueError("admission and generation token accounting differ")
    audit_board = require_exact_keys(audit["board"], {"case_count", "rows_sha256", "strata"}, "audit board summary")
    if audit_board["case_count"] != 256 or audit_board["rows_sha256"] != BOARD_ROWS_SHA256 or audit_board["strata"] != {"length_ood": 64, "order_ood": 64, "renderer_ood": 64, "value_ood": 64}:
        raise ValueError("audit board summary differs")
    audit_sham = require_exact_keys(audit["sham"], {"fixed_points", "same_answer", "same_program", "same_trajectory", "strata", "minimum_stratum_size"}, "audit sham summary")
    if any(audit_sham[name] != 0 for name in ("fixed_points", "same_answer", "same_program", "same_trajectory")) or type(audit_sham["minimum_stratum_size"]) is not int or audit_sham["minimum_stratum_size"] < 2:
        raise ValueError("audit sham derangement summary differs")
    audit_training = require_exact_keys(audit["training"], {"duplicate_normalized_sham_prompts", "duplicate_normalized_treatment_prompts", "length_counts", "programs", "sham_rows", "treatment_rows"}, "audit training summary")
    if audit_training["programs"] != 4096 or audit_training["length_counts"] != TRAIN_LENGTH_COUNTS or audit_training["duplicate_normalized_sham_prompts"] != 0 or audit_training["duplicate_normalized_treatment_prompts"] != 0 or audit_training["treatment_rows"] != artifacts["treatment_rows"] or audit_training["sham_rows"] != artifacts["sham_rows"]:
        raise ValueError("audit training summary differs")

    board = require_exact_keys(board, {"case_count", "per_stratum", "rows", "rows_sha256", "schema", "seed", "stratum_order"}, "board")
    if board["schema"] != BOARD_SCHEMA or board["seed"] != BOARD_SEED or board["case_count"] != 256 or board["per_stratum"] != 64 or board["stratum_order"] != ["renderer_ood", "value_ood", "order_ood", "length_ood"] or board["rows_sha256"] != BOARD_ROWS_SHA256:
        raise ValueError("board metadata differs from frozen contract")
    rows = board["rows"]
    if not isinstance(rows, list) or len(rows) != 256 or len({row.get("id") for row in rows if isinstance(row, Mapping)}) != 256:
        raise ValueError("board rows are missing or duplicated")
    if canonical_digest(rows) != BOARD_ROWS_SHA256:
        raise ValueError("board canonical rows hash differs")
    return rows


def runtime_fingerprint() -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": properties.name, "capability": list(torch.cuda.get_device_capability(index)), "total_memory": properties.total_memory})
    try:
        nvidia_smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia_smi = []
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "tokenizers": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu, "nvidia_smi": nvidia_smi,
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def main() -> None:
    manifest_source = Path(sys.argv[1]).resolve(strict=True)
    manifest_expected = sys.argv[2].lower()
    seed = int(sys.argv[3])
    snapshot_root = Path(sys.argv[4]).resolve(strict=True)
    script_path = Path(sys.argv[5]).resolve(strict=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if sha256_file(manifest_source) != manifest_expected:
        raise SystemExit("[rsp-eval] run-manifest trust-root hash mismatch")
    directory = manifest_source.parent
    records: dict[str, Any] = {}
    manifest_snapshot = snapshot_root / "inputs/run_manifest.json"
    records["run_manifest"] = copy_verified(manifest_source, manifest_snapshot, manifest_expected)
    manifest = validate_run_manifest(read_json(manifest_snapshot, "eval run manifest"), seed)
    for label, raw_ref in {**manifest["sources"], **manifest["artifacts"]}.items():
        ref = validate_ref(raw_ref, label)
        source = resolve_ref(ref, directory)
        if label == "job" and source != script_path:
            raise ValueError("run manifest job reference does not name the executing wrapper")
        records[label] = copy_verified(source, snapshot_root / DESTINATIONS[label], ref["sha256"])
    hashes = {label: record["sha256"] for label, record in records.items()}
    if hashes["board"] != BOARD_SHA256 or hashes["tokenizer"] != TOKENIZER_SHA256 or hashes["executor"] != EXECUTOR_SHA256:
        raise ValueError("frozen board/tokenizer/executor digest differs")
    treatment_receipt = read_json(snapshot_root / DESTINATIONS["treatment_receipt"], "treatment receipt")
    sham_receipt = read_json(snapshot_root / DESTINATIONS["sham_receipt"], "sham receipt")
    controller = read_json(snapshot_root / DESTINATIONS["controller_manifest"], "controller manifest")
    executor_manifest = read_json(snapshot_root / DESTINATIONS["executor_manifest"], "executor manifest")
    resources = read_json(snapshot_root / DESTINATIONS["training_resources"], "training resources")
    validate_pair_contract(controller, executor_manifest, resources, treatment_receipt, sham_receipt, hashes, seed)
    rows = validate_generation_audit_board(
        read_json(snapshot_root / DESTINATIONS["training_manifest"], "generation manifest"),
        read_json(snapshot_root / DESTINATIONS["admission_audit"], "admission audit"),
        read_json(snapshot_root / DESTINATIONS["board"], "board"),
        hashes,
        treatment_receipt,
    )
    current_runtime = runtime_fingerprint()
    if stable_runtime_contract(current_runtime, seed) != stable_runtime_contract(treatment_receipt["runtime"], seed):
        raise ValueError("evaluation runtime differs from paired fit runtime contract")
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "paired_seed": seed,
        "run_manifest_sha256": manifest_expected,
        "records": records,
        "board_ids": [row["id"] for row in rows],
        "board_transition_count": sum(len(row["operations"]) for row in rows),
        "runtime": current_runtime,
    }
    write_immutable_json(snapshot_root / "custody_preflight.json", preflight)
    for directory_path in (snapshot_root / "train", snapshot_root / "train/jobs", snapshot_root / "inputs"):
        os.chmod(directory_path, 0o555)
    os.chmod(snapshot_root, 0o555)
    print(json.dumps({"snapshot": str(snapshot_root), "seed": seed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
# EVAL_CUSTODY_PY_END
PY

PREFLIGHT=$SNAPSHOT_ROOT/custody_preflight.json
SNAP_EVALUATOR=$SNAPSHOT_ROOT/train/eval_residual_packet_v1.py
SNAP_BOARD=$SNAPSHOT_ROOT/inputs/board.json
SNAP_TOKENIZER=$SNAPSHOT_ROOT/inputs/tokenizer.json
SNAP_EXECUTOR=$SNAPSHOT_ROOT/inputs/executor.pt
SNAP_TREATMENT=$SNAPSHOT_ROOT/inputs/treatment.pt
SNAP_SHAM=$SNAPSHOT_ROOT/inputs/sham.pt
SNAP_RESOURCES=$SNAPSHOT_ROOT/inputs/training_resources.json

if [ "${CUSTODY_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "[rsp-eval] custody preflight only"
  exit 0
fi

timeout --kill-after=10s 180s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
left = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
right = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
product = left @ right
torch.cuda.synchronize()
if not product.is_cuda or float(product[0, 0]) != 1024.0:
    raise SystemExit(75)
print("[rsp-eval] CUDA preflight", torch.cuda.get_device_name(0), flush=True)
PY

cd "$SNAPSHOT_ROOT/train"
"$PY" - "$SNAP_EVALUATOR" \
  --board "$SNAP_BOARD" \
  --tokenizer "$SNAP_TOKENIZER" \
  --executor "$SNAP_EXECUTOR" \
  --treatment "$SNAP_TREATMENT" \
  --sham "$SNAP_SHAM" \
  --seed "$SEED" \
  --training-resources "$SNAP_RESOURCES" \
  --out "$TRANSCRIPT_OUT" <<'PY'
import os
import runpy
import sys
import torch

script = os.path.realpath(sys.argv[1])
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script, *sys.argv[2:]]
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
runpy.run_path(script, run_name="__main__")
PY

"$PY" - "$PREFLIGHT" "$TRANSCRIPT_OUT" "$TRANSCRIPT_RECEIPT" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import tokenizers


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_fingerprint():
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({"index": index, "name": properties.name, "capability": list(torch.cuda.get_device_capability(index)), "total_memory": properties.total_memory})
    try:
        nvidia_smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia_smi = []
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "tokenizers": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu, "nvidia_smi": nvidia_smi,
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise SystemExit(f"[rsp-eval] {label} schema differs")
    return value


def write_immutable(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short transcript-receipt write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


preflight_path, transcript_path, receipt_path = map(Path, sys.argv[1:])
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
preflight = json.loads(preflight_path.read_text())
if preflight.get("schema") != "rsp_c1_eval_preflight_v1":
    raise SystemExit("[rsp-eval] preflight schema mismatch")
for label, record in preflight["records"].items():
    if sha256_file(record["source"]) != record["sha256"] or sha256_file(record["snapshot"]) != record["sha256"]:
        raise SystemExit(f"[rsp-eval] final input rehash failed: {label}")
if runtime_fingerprint() != preflight["runtime"]:
    raise SystemExit("[rsp-eval] runtime fingerprint changed during acquisition")
info = transcript_path.lstat()
if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
    raise SystemExit("[rsp-eval] transcript is absent, nonregular, or writable")
transcript = json.loads(transcript_path.read_text())
exact_keys(transcript, {"schema", "seed", "protocol_module", "input_hashes", "decode_caps", "models", "external_scheduler", "controllers", "resource_ledger", "call_count"}, "transcript")
if transcript["schema"] != "residual_packet_v1_raw_transcript" or transcript["seed"] != preflight["paired_seed"]:
    raise SystemExit("[rsp-eval] transcript schema/seed mismatch")
if transcript["protocol_module"] != "residual_packet_protocol":
    raise SystemExit("[rsp-eval] transcript protocol-module identity differs")
expected_input_hashes = {
    "board": preflight["records"]["board"]["sha256"],
    "tokenizer": preflight["records"]["tokenizer"]["sha256"],
    "raw_260k_executor": preflight["records"]["executor"]["sha256"],
    "treatment_checkpoint": preflight["records"]["treatment"]["sha256"],
    "sham_checkpoint": preflight["records"]["sham"]["sha256"],
    "protocol": preflight["records"]["protocol"]["sha256"],
    "evaluator": preflight["records"]["evaluator"]["sha256"],
    "training_resources": preflight["records"]["training_resources"]["sha256"],
}
if transcript["input_hashes"] != expected_input_hashes:
    raise SystemExit("[rsp-eval] transcript input hashes differ")
if transcript["decode_caps"] != {"controller": 80, "executor": 48, "maximum_transitions": 5}:
    raise SystemExit("[rsp-eval] transcript decode caps differ")
models = exact_keys(transcript["models"], {"treatment", "sham", "raw_260k_executor"}, "transcript models")
for model, record_label in (("treatment", "treatment"), ("sham", "sham"), ("raw_260k_executor", "executor")):
    entry = exact_keys(models[model], {"checkpoint_path", "checkpoint_sha256", "checkpoint_step"}, f"model {model}")
    if entry["checkpoint_sha256"] != preflight["records"][record_label]["sha256"]:
        raise SystemExit(f"[rsp-eval] transcript model hash differs: {model}")
    expected_step = "sft_ep10" if model in {"treatment", "sham"} else 260000
    if entry["checkpoint_step"] != expected_step:
        raise SystemExit(f"[rsp-eval] transcript model step differs: {model}")
ids = preflight["board_ids"]
if [item.get("id") for item in transcript["external_scheduler"]] != ids:
    raise SystemExit("[rsp-eval] external-scheduler board coverage differs")
controllers = exact_keys(transcript["controllers"], {"treatment", "sham"}, "controllers")
for arm in ("treatment", "sham"):
    sections = exact_keys(controllers[arm], {"strict_closed_loop", "oracle_packet_loop", "teacher_forced_updater", "packet_swaps"}, f"controller {arm}")
    if [item.get("id") for item in sections["strict_closed_loop"]] != ids or [item.get("id") for item in sections["oracle_packet_loop"]] != ids:
        raise SystemExit(f"[rsp-eval] {arm} board coverage differs")
    if len(sections["teacher_forced_updater"]) != preflight["board_transition_count"] or len(sections["packet_swaps"]) != 64:
        raise SystemExit(f"[rsp-eval] {arm} auxiliary transcript cardinality differs")
if type(transcript["call_count"]) is not int or transcript["call_count"] <= 0:
    raise SystemExit("[rsp-eval] transcript call count is invalid")

transcript_sha = sha256_file(transcript_path)
slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
if not slurm_job_id.isdigit():
    raise SystemExit("[rsp-eval] transcript receipt requires a numeric Slurm job ID")
receipt = {
    "schema": "rsp_c1_transcript_receipt_v1",
    "paired_seed": preflight["paired_seed"],
    "run_manifest_sha256": preflight["run_manifest_sha256"],
    "transcript_sha256": transcript_sha,
    "input_hashes": expected_input_hashes,
    "case_count": len(ids),
    "transition_count": preflight["board_transition_count"],
    "call_count": transcript["call_count"],
    "runtime": preflight["runtime"],
    "slurm_job_id": slurm_job_id,
}
receipt_sha = write_immutable(receipt_path, receipt)
final_info = transcript_path.lstat()
if (
    (final_info.st_dev, final_info.st_ino, final_info.st_size, final_info.st_mtime_ns)
    != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    or not stat.S_ISREG(final_info.st_mode)
    or final_info.st_mode & 0o222
    or sha256_file(transcript_path) != transcript_sha
):
    raise SystemExit("[rsp-eval] transcript changed while its receipt was sealed")
receipt_info = receipt_path.lstat()
if not stat.S_ISREG(receipt_info.st_mode) or receipt_info.st_mode & 0o222 or sha256_file(receipt_path) != receipt_sha:
    raise SystemExit("[rsp-eval] transcript receipt postcondition failed")
print(json.dumps({"transcript": str(transcript_path), "transcript_sha256": transcript_sha, "receipt": str(receipt_path), "receipt_sha256": receipt_sha}, sort_keys=True), flush=True)
PY

echo "[rsp-eval] completed immutable transcript seed=$SEED out=$TRANSCRIPT_OUT"
