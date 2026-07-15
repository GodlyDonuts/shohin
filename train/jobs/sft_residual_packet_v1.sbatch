#!/bin/bash
# Frozen RSP-C1 paired fit. The only trust root supplied at launch is one
# precommitted run-manifest digest; every consumed byte is copied into job-local
# scratch, verified, and loaded from that immutable snapshot.
#SBATCH -J shohin-sft-rsp-c1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_rsp_c1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

echo "RSP-C1 is permanently closed; this wrapper is retained as audit evidence only" >&2
exit 64

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
RUN_MANIFEST=${RUN_MANIFEST:?set the frozen rsp_c1_sft_run_manifest_v1 JSON}
RUN_MANIFEST_SHA256=${RUN_MANIFEST_SHA256:?set its precommitted SHA-256}
ARM=${ARM:?set ARM to treatment or sham}
SEED=${SEED:?set SEED to paired seed 2026071511 or 2026071512}
OUT=${OUT:?set OUT to a fresh directory below FIT_ROOT}
FIT_ROOT=${FIT_ROOT:-$BASE/train/rsp_c1_fits}

# Frozen fit settings. The run manifest must state these exact values.
EPOCHS=10
PACK_LEN=128
BS=64
LR_MUON=8e-4
LR_ADAM=2e-4
WARMUP=50
CLIP=1.0

case "$ARM" in
  treatment|sham) ;;
  *) echo "[rsp-sft] ARM must be treatment or sham" >&2; exit 2 ;;
esac
case "$SEED" in
  2026071511|2026071512) ;;
  *) echo "[rsp-sft] seed is outside the frozen paired-seed set: $SEED" >&2; exit 2 ;;
esac

for path in "$PY" "$RUN_MANIFEST"; do
  [ -s "$path" ] || { echo "[rsp-sft] missing required input: $path" >&2; exit 2; }
done
PY=$(realpath -e -- "$PY")
BASE=$(realpath -e -- "$BASE")
RUN_MANIFEST=$(realpath -e -- "$RUN_MANIFEST")
SCRIPT_PATH=$(realpath -e -- "${BASH_SOURCE[0]}")

mkdir -p -- "$FIT_ROOT"
FIT_ROOT=$(realpath -e -- "$FIT_ROOT")
OUT=$(realpath -m -- "$OUT")
if [[ "$OUT" != "$FIT_ROOT/"* ]]; then
  echo "[rsp-sft] OUT must be a child of FIT_ROOT=$FIT_ROOT" >&2
  exit 3
fi
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "[rsp-sft] refusing existing output: $OUT" >&2
  exit 3
fi
mkdir -p -- "$(dirname "$OUT")"
OUT_PARENT=$(realpath -e -- "$(dirname "$OUT")")
if [[ "$OUT_PARENT" != "$FIT_ROOT" && "$OUT_PARENT" != "$FIT_ROOT/"* ]]; then
  echo "[rsp-sft] resolved output parent escaped FIT_ROOT" >&2
  exit 3
fi
OUT=$OUT_PARENT/$(basename "$OUT")

SCRATCH_PARENT=${SLURM_TMPDIR:-/tmp}
SNAPSHOT_ROOT=$(mktemp -d "$SCRATCH_PARENT/rsp_sft_${SLURM_JOB_ID:-manual}.XXXXXX")
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

"$PY" - "$RUN_MANIFEST" "$RUN_MANIFEST_SHA256" "$ARM" "$SEED" "$SNAPSHOT_ROOT" "$SCRIPT_PATH" <<'PY'
# SFT_CUSTODY_PY_BEGIN
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import tokenizers
from tokenizers import Tokenizer

SFT_RUN_SCHEMA = "rsp_c1_sft_run_manifest_v1"
GENERATION_SCHEMA = "residual_packet_generation_manifest_v1"
AUDIT_SCHEMA = "residual_packet_admission_audit_v1"
PREFLIGHT_SCHEMA = "rsp_c1_sft_preflight_v1"
FIT_RECEIPT_SCHEMA = "rsp_c1_fit_receipt_v1"
FIT_SEEDS = (2026071511, 2026071512)
GENERATION_SEEDS = {"board": 2026071503, "training": 2026071504, "observation": 2026071505, "sham": 2026071506}
TRAIN_LENGTH_COUNTS = {"2": 1024, "3": 2048, "4": 1024}
BOARD_ROWS_SHA256 = "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
FIT = {
    "epochs": 10,
    "pack_len": 128,
    "batch_size": 64,
    "lr_muon": 8e-4,
    "lr_adam": 2e-4,
    "warmup": 50,
    "clip": 1.0,
}
SOURCE_KEYS = {"job", "trainer", "encoding", "model", "muon", "protocol"}
ARTIFACT_KEYS = {
    "preregistration", "init", "tokenizer", "treatment_data", "sham_data",
    "training_manifest", "admission_audit",
}
REF_KEYS = {"path", "sha256"}
COMPILER_FIELDS = {
    "completion_prompt", "id", "kind", "program_id", "question", "response",
    "training_group",
}
UPDATER_FIELDS = COMPILER_FIELDS | {"step"}
AUDIT_TOP_KEYS = {
    "admitted", "artifact_sha256", "board", "failures", "overlap", "row_metrics",
    "schema", "sham", "token_accounting", "training",
}
AUDIT_HASH_KEYS = {"board", "manifest", "protocol", "sham", "sft_encoding", "treatment", "tokenizer"}
ROW_METRIC_KEYS = {
    "compiler_controls_matched", "correct_arithmetic_updater_transitions",
    "evaluation_answer_response_occurrences", "malformed_compiler_rows",
    "malformed_updater_rows", "reconstructed_sham_mapping_mismatches",
    "source_final_answer_response_occurrences", "updater_pairs_byte_identical",
}
OVERLAP_KEYS = {
    "complete_trajectory_overlap", "exact_packet_overlap", "exact_source_overlap",
    "normalized_13_token_source_ngram_overlap", "normalized_semantic_program_overlap",
    "reserved_renderer_training_occurrences",
}
TOKEN_ACCOUNTING_KEYS = {
    "compiler_rows", "examples", "forward_token_count", "full_token_count",
    "packed_sequence_count", "prompt_token_count", "response_token_count",
    "skipped_too_long", "supervised_token_count", "updater_rows",
}
DESTINATIONS = {
    "job": "train/jobs/sft_residual_packet_v1.sbatch",
    "trainer": "train/sft.py",
    "encoding": "train/sft_encoding.py",
    "model": "train/model.py",
    "muon": "train/muon.py",
    "protocol": "train/residual_packet_protocol.py",
    "preregistration": "inputs/preregistration.md",
    "init": "inputs/init.pt",
    "tokenizer": "inputs/tokenizer.json",
    "treatment_data": "inputs/treatment.jsonl",
    "sham_data": "inputs/sham.jsonl",
    "training_manifest": "inputs/training_manifest.json",
    "admission_audit": "inputs/admission_audit.json",
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


def validate_run_manifest(manifest: object, arm: str, seed: int) -> Mapping[str, Any]:
    value = require_exact_keys(manifest, {"schema", "frozen", "sources", "artifacts", "fit", "paired_seeds"}, "SFT run manifest")
    if value["schema"] != SFT_RUN_SCHEMA or value["frozen"] is not True:
        raise ValueError("SFT run manifest schema/frozen contract failed")
    if arm not in {"treatment", "sham"} or seed not in FIT_SEEDS:
        raise ValueError("arm or paired seed is outside the frozen set")
    if value["paired_seeds"] != list(FIT_SEEDS):
        raise ValueError("SFT run manifest paired seeds differ")
    if value["fit"] != FIT:
        raise ValueError("SFT run manifest fit settings differ")
    sources = require_exact_keys(value["sources"], SOURCE_KEYS, "SFT run sources")
    artifacts = require_exact_keys(value["artifacts"], ARTIFACT_KEYS, "SFT run artifacts")
    for label, ref in {**sources, **artifacts}.items():
        validate_ref(ref, label)
    return value


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
    observed = digest.hexdigest()
    after = source.stat()
    if observed != expected or sha256_file(destination) != expected:
        raise ValueError(f"snapshot hash mismatch for {source}")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while it was snapshotted: {source}")
    if sha256_file(source) != expected:
        raise RuntimeError(f"input changed immediately after snapshot: {source}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "sha256": expected,
        "size": destination.stat().st_size,
        "source_device": before.st_dev,
        "source_inode": before.st_ino,
    }


def load_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} has no rows")
    return rows


def import_encoding(path: Path):
    spec = importlib.util.spec_from_file_location("rsp_snapshot_sft_encoding", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load snapshotted SFT encoding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_supervised_example


def validate_generation_and_audit(
    generation: object,
    audit: object,
    hashes: Mapping[str, str],
) -> None:
    generation = require_exact_keys(
        generation,
        {"artifacts", "length_counts", "pack_length", "program_count", "protocol_sha256", "schema", "seeds", "token_accounting", "tokenizer_sha256", "sft_encoding_sha256"},
        "generation manifest",
    )
    if generation["schema"] != GENERATION_SCHEMA or generation["pack_length"] != FIT["pack_len"]:
        raise ValueError("generation manifest schema or pack length differs")
    if generation["program_count"] != 4096 or generation["length_counts"] != TRAIN_LENGTH_COUNTS or generation["seeds"] != GENERATION_SEEDS:
        raise ValueError("generation manifest program/length/seed contract differs")
    artifacts = require_exact_keys(
        generation["artifacts"],
        {"board_sha256", "board_rows_sha256", "sham_rows", "sham_sha256", "treatment_rows", "treatment_sha256"},
        "generation artifacts",
    )
    if artifacts["treatment_sha256"] != hashes["treatment_data"] or artifacts["sham_sha256"] != hashes["sham_data"]:
        raise ValueError("generation manifest data-arm binding failed")
    if generation["tokenizer_sha256"] != hashes["tokenizer"] or generation["sft_encoding_sha256"] != hashes["encoding"] or generation["protocol_sha256"] != hashes["protocol"]:
        raise ValueError("generation manifest source/tokenizer binding failed")
    token_accounting = require_exact_keys(generation["token_accounting"], {"treatment", "sham"}, "generation token accounting")
    if token_accounting["treatment"] != token_accounting["sham"]:
        raise ValueError("generation manifest token accounting differs between arms")
    accounting = require_exact_keys(token_accounting["treatment"], TOKEN_ACCOUNTING_KEYS, "generation arm accounting")
    if (
        accounting["compiler_rows"] != 4096
        or accounting["updater_rows"] != 12288
        or accounting["examples"] != artifacts["treatment_rows"]
        or accounting["skipped_too_long"] != 0
        or accounting["forward_token_count"] != accounting["packed_sequence_count"] * FIT["pack_len"]
    ):
        raise ValueError("generation arm accounting is internally inconsistent")

    audit = require_exact_keys(audit, AUDIT_TOP_KEYS, "admission audit")
    if audit["schema"] != AUDIT_SCHEMA or audit["admitted"] is not True or audit["failures"] != []:
        raise ValueError("admission audit is not an exact passing v1 audit")
    artifact_hashes = require_exact_keys(audit["artifact_sha256"], AUDIT_HASH_KEYS, "admission artifact hashes")
    expected_hashes = {
        "manifest": hashes["training_manifest"],
        "protocol": hashes["protocol"],
        "sham": hashes["sham_data"],
        "sft_encoding": hashes["encoding"],
        "treatment": hashes["treatment_data"],
        "tokenizer": hashes["tokenizer"],
        "board": artifacts["board_sha256"],
    }
    if dict(artifact_hashes) != expected_hashes:
        raise ValueError("admission audit named artifact bindings differ")
    overlap = require_exact_keys(audit["overlap"], OVERLAP_KEYS, "admission overlap gates")
    if any(type(value) is not int or value != 0 for value in overlap.values()):
        raise ValueError("one or more admission overlap gates are nonzero")
    metrics = require_exact_keys(audit["row_metrics"], ROW_METRIC_KEYS, "admission row metrics")
    for name in ROW_METRIC_KEYS - {"compiler_controls_matched", "updater_pairs_byte_identical"}:
        if type(metrics[name]) is not int or metrics[name] != 0:
            raise ValueError(f"admission row gate failed: {name}")
    if metrics["compiler_controls_matched"] != 4096 or metrics["updater_pairs_byte_identical"] != 12288:
        raise ValueError("admission paired-control counts differ from the frozen contract")
    if audit["token_accounting"] != token_accounting:
        raise ValueError("admission and generation token accounting differ")
    board = require_exact_keys(audit["board"], {"case_count", "rows_sha256", "strata"}, "admission board summary")
    if board["case_count"] != 256 or board["rows_sha256"] != BOARD_ROWS_SHA256 or board["strata"] != {"length_ood": 64, "order_ood": 64, "renderer_ood": 64, "value_ood": 64}:
        raise ValueError("admission board summary differs")
    sham = require_exact_keys(audit["sham"], {"fixed_points", "same_answer", "same_program", "same_trajectory", "strata", "minimum_stratum_size"}, "admission sham summary")
    if any(sham[name] != 0 for name in ("fixed_points", "same_answer", "same_program", "same_trajectory")) or type(sham["strata"]) is not int or sham["strata"] < 1 or type(sham["minimum_stratum_size"]) is not int or sham["minimum_stratum_size"] < 2:
        raise ValueError("admission sham derangement summary differs")
    training = require_exact_keys(
        audit["training"],
        {"duplicate_normalized_sham_prompts", "duplicate_normalized_treatment_prompts", "length_counts", "programs", "sham_rows", "treatment_rows"},
        "admission training summary",
    )
    if training["sham_rows"] != artifacts["sham_rows"] or training["treatment_rows"] != artifacts["treatment_rows"]:
        raise ValueError("admission row counts do not bind generation artifacts")
    if training["programs"] != 4096 or training["length_counts"] != TRAIN_LENGTH_COUNTS:
        raise ValueError("admission program/length summary differs")
    if training["duplicate_normalized_sham_prompts"] != 0 or training["duplicate_normalized_treatment_prompts"] != 0:
        raise ValueError("admission audit reports duplicate normalized prompts")


def compute_pair_geometry(
    treatment_rows: list[dict[str, Any]],
    sham_rows: list[dict[str, Any]],
    tokenizer: Any,
    eos_id: int,
    encode_supervised_example: Any,
    seed: int,
) -> dict[str, Any]:
    if len(treatment_rows) != len(sham_rows):
        raise ValueError("treatment/sham row counts differ")
    arm_buffers: dict[str, dict[str, Any]] = {}
    response_multisets: dict[str, Counter[str]] = {}
    for label, rows in (("treatment", treatment_rows), ("sham", sham_rows)):
        tokens: list[int] = []
        masks: list[int] = []
        row_geometry = []
        responses: Counter[str] = Counter()
        for index, row in enumerate(rows):
            expected_fields = COMPILER_FIELDS if row.get("kind") == "compiler" else UPDATER_FIELDS if row.get("kind") == "updater" else set()
            if set(row) != expected_fields:
                raise ValueError(f"{label} row {index} fields differ from its exact kind schema")
            prompt, response = row["completion_prompt"], row["response"]
            if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response.rstrip():
                raise ValueError(f"{label} row {index} has invalid prompt/response")
            completion = response.rstrip()
            prompt_ids, full_ids, mask = encode_supervised_example(tokenizer, prompt, completion, eos_id)
            if len(full_ids) > FIT["pack_len"]:
                raise ValueError(f"{label} row {index} exceeds pack length")
            if sum(mask) != len(full_ids) - len(prompt_ids):
                raise ValueError(f"{label} row {index} completion boundary differs")
            tokens.extend(full_ids)
            masks.extend(mask)
            responses[response] += 1
            row_geometry.append({
                "id": row["id"],
                "kind": row["kind"],
                "prompt_tokens": len(prompt_ids),
                "full_tokens": len(full_ids),
                "supervised_tokens": sum(mask),
                "mask_sha256": canonical_digest(mask),
            })
        arm_buffers[label] = {"tokens": tokens, "masks": masks, "rows": row_geometry}
        response_multisets[label] = responses

    for index, (left, right) in enumerate(zip(treatment_rows, sham_rows)):
        left_control = {key: value for key, value in left.items() if key != "response"}
        right_control = {key: value for key, value in right.items() if key != "response"}
        if left_control != right_control:
            raise ValueError(f"row controls differ at index {index}")
        if left["kind"] == "updater" and left != right:
            raise ValueError(f"updater arms differ at index {index}")
    if response_multisets["treatment"] != response_multisets["sham"]:
        raise ValueError("response byte multisets differ between arms")
    if arm_buffers["treatment"]["rows"] != arm_buffers["sham"]["rows"]:
        raise ValueError("per-row token/mask geometry differs between arms")
    if arm_buffers["treatment"]["masks"] != arm_buffers["sham"]["masks"]:
        raise ValueError("position-wise completion masks differ between arms")
    treatment_token_multiset = Counter(arm_buffers["treatment"]["tokens"])
    sham_token_multiset = Counter(arm_buffers["sham"]["tokens"])
    if treatment_token_multiset != sham_token_multiset:
        raise ValueError("full token-ID multisets differ between arms")

    full_tokens = len(arm_buffers["treatment"]["tokens"])
    if full_tokens != len(arm_buffers["sham"]["tokens"]):
        raise ValueError("full-token totals differ between arms")
    starts = list(range(0, full_tokens - FIT["pack_len"] - 1, FIT["pack_len"]))
    packed_mask_counts = [
        sum(arm_buffers["treatment"]["masks"][start + 1:start + FIT["pack_len"] + 1])
        for start in starts
    ]
    sham_packed_mask_counts = [
        sum(arm_buffers["sham"]["masks"][start + 1:start + FIT["pack_len"] + 1])
        for start in starts
    ]
    if packed_mask_counts != sham_packed_mask_counts:
        raise ValueError("packed supervision geometry differs between arms")
    packed_sequences = len(starts)
    tail_start = packed_sequences * FIT["pack_len"]
    treatment_tail = arm_buffers["treatment"]["tokens"][tail_start:]
    sham_tail = arm_buffers["sham"]["tokens"][tail_start:]
    treatment_tail_mask = arm_buffers["treatment"]["masks"][tail_start:]
    sham_tail_mask = arm_buffers["sham"]["masks"][tail_start:]
    if treatment_tail != sham_tail or treatment_tail_mask != sham_tail_mask:
        raise ValueError("discarded token/mask tails differ between arms")
    steps_per_epoch = packed_sequences // FIT["batch_size"]
    if steps_per_epoch < 1:
        raise ValueError("paired dataset produces fewer than one full batch")
    rng = np.random.default_rng(seed)
    supervised = 0
    consumed_sequences = 0
    for _ in range(FIT["epochs"]):
        order = rng.permutation(packed_sequences)
        consumed = order[:steps_per_epoch * FIT["batch_size"]]
        supervised += sum(packed_mask_counts[int(index)] for index in consumed)
        consumed_sequences += len(consumed)
    geometry = {
        "rows": len(treatment_rows),
        "compiler_rows": sum(row["kind"] == "compiler" for row in treatment_rows),
        "updater_rows": sum(row["kind"] == "updater" for row in treatment_rows),
        "full_tokens_before_packing": full_tokens,
        "prompt_tokens_before_packing": sum(row["prompt_tokens"] for row in arm_buffers["treatment"]["rows"]),
        "response_tokens_before_packing": sum(row["full_tokens"] - row["prompt_tokens"] - 1 for row in arm_buffers["treatment"]["rows"]),
        "supervised_tokens_before_packing": sum(arm_buffers["treatment"]["masks"]),
        "packed_sequences": packed_sequences,
        "discarded_tail_tokens": full_tokens - packed_sequences * FIT["pack_len"],
        "steps_per_epoch": steps_per_epoch,
        "optimizer_updates": steps_per_epoch * FIT["epochs"],
        "packed_forward_token_positions": consumed_sequences * FIT["pack_len"],
        "supervised_completion_tokens": supervised,
        "row_geometry_sha256": canonical_digest(arm_buffers["treatment"]["rows"]),
        "position_mask_sha256": canonical_digest(arm_buffers["treatment"]["masks"]),
        "packed_mask_counts_sha256": canonical_digest(packed_mask_counts),
        "response_multiset_sha256": canonical_digest(sorted(response_multisets["treatment"].items())),
        "full_token_multiset_sha256": canonical_digest(sorted(treatment_token_multiset.items())),
        "discarded_tail_token_ids_sha256": canonical_digest(treatment_tail),
        "discarded_tail_mask_sha256": canonical_digest(treatment_tail_mask),
    }
    geometry["pair_geometry_sha256"] = canonical_digest(geometry)
    return geometry


def runtime_fingerprint() -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": properties.total_memory,
            })
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia_smi = []
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def main() -> None:
    manifest_source = Path(sys.argv[1]).resolve(strict=True)
    manifest_expected = sys.argv[2].lower()
    arm = sys.argv[3]
    seed = int(sys.argv[4])
    snapshot_root = Path(sys.argv[5]).resolve(strict=True)
    script_path = Path(sys.argv[6]).resolve(strict=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if sha256_file(manifest_source) != manifest_expected:
        raise SystemExit("[rsp-sft] run-manifest trust-root hash mismatch")
    manifest_directory = manifest_source.parent
    records: dict[str, Any] = {}
    manifest_snapshot = snapshot_root / "inputs/run_manifest.json"
    records["run_manifest"] = copy_verified(manifest_source, manifest_snapshot, manifest_expected)
    manifest = validate_run_manifest(read_json(manifest_snapshot, "SFT run manifest"), arm, seed)
    combined = {**manifest["sources"], **manifest["artifacts"]}
    for label, raw_ref in combined.items():
        ref = validate_ref(raw_ref, label)
        source = resolve_ref(ref, manifest_directory)
        if label == "job" and source != script_path:
            raise ValueError("run manifest job reference does not name the executing wrapper")
        records[label] = copy_verified(source, snapshot_root / DESTINATIONS[label], ref["sha256"])
    hashes = {label: record["sha256"] for label, record in records.items()}
    generation = read_json(snapshot_root / DESTINATIONS["training_manifest"], "generation manifest")
    audit = read_json(snapshot_root / DESTINATIONS["admission_audit"], "admission audit")
    validate_generation_and_audit(generation, audit, hashes)
    tokenizer = Tokenizer.from_file(str(snapshot_root / DESTINATIONS["tokenizer"]))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer lacks <|endoftext|>")
    encode = import_encoding(snapshot_root / DESTINATIONS["encoding"])
    treatment_rows = load_rows(snapshot_root / DESTINATIONS["treatment_data"], "treatment")
    sham_rows = load_rows(snapshot_root / DESTINATIONS["sham_data"], "sham")
    geometry = compute_pair_geometry(treatment_rows, sham_rows, tokenizer, eos_id, encode, seed)
    artifacts = generation["artifacts"]
    accounting = generation["token_accounting"]["treatment"]
    if geometry["rows"] != artifacts["treatment_rows"] or geometry["rows"] != artifacts["sham_rows"]:
        raise ValueError("pair geometry row count differs from generation manifest")
    if geometry["compiler_rows"] != 4096 or geometry["updater_rows"] != 12288:
        raise ValueError("pair geometry row-kind counts differ from the frozen contract")
    expected_geometry_accounting = {
        "rows": accounting["examples"],
        "compiler_rows": accounting["compiler_rows"],
        "updater_rows": accounting["updater_rows"],
        "full_tokens_before_packing": accounting["full_token_count"],
        "prompt_tokens_before_packing": accounting["prompt_token_count"],
        "response_tokens_before_packing": accounting["response_token_count"],
        "supervised_tokens_before_packing": accounting["supervised_token_count"],
        "packed_sequences": accounting["packed_sequence_count"],
        "discarded_tail_tokens": accounting["full_token_count"] - accounting["forward_token_count"],
    }
    if any(geometry[name] != expected for name, expected in expected_geometry_accounting.items()):
        raise ValueError("recomputed pair geometry differs from generation token accounting")
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "arm": arm,
        "paired_seed": seed,
        "run_manifest_sha256": manifest_expected,
        "fit": FIT,
        "records": records,
        "geometry": geometry,
        "runtime": runtime_fingerprint(),
    }
    write_immutable_json(snapshot_root / "custody_preflight.json", preflight)
    for directory in (snapshot_root / "train", snapshot_root / "train/jobs", snapshot_root / "inputs"):
        os.chmod(directory, 0o555)
    os.chmod(snapshot_root, 0o555)
    print(json.dumps({"snapshot": str(snapshot_root), "geometry": geometry}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
# SFT_CUSTODY_PY_END
PY

PREFLIGHT=$SNAPSHOT_ROOT/custody_preflight.json
SNAP_TRAINER=$SNAPSHOT_ROOT/train/sft.py
SNAP_INIT=$SNAPSHOT_ROOT/inputs/init.pt
SNAP_TOKENIZER=$SNAPSHOT_ROOT/inputs/tokenizer.json
if [ "$ARM" = treatment ]; then
  SNAP_DATA=$SNAPSHOT_ROOT/inputs/treatment.jsonl
else
  SNAP_DATA=$SNAPSHOT_ROOT/inputs/sham.jsonl
fi

if [ "${CUSTODY_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "[rsp-sft] custody preflight only"
  exit 0
fi

timeout --kill-after=10s 180s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.use_deterministic_algorithms(True)
left = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
right = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
product = left @ right
torch.cuda.synchronize()
if not product.is_cuda or float(product[0, 0]) != 1024.0:
    raise SystemExit(75)
print("[rsp-sft] deterministic CUDA preflight", torch.cuda.get_device_name(0), flush=True)
PY

mkdir -- "$OUT"
echo "[rsp-sft] arm=$ARM seed=$SEED snapshot=$SNAPSHOT_ROOT out=$OUT"
"$PY" - "$SNAP_TRAINER" \
  --init "$SNAP_INIT" \
  --data "$SNAP_DATA" \
  --tokenizer "$SNAP_TOKENIZER" \
  --q-fields completion_prompt \
  --r-fields response \
  --prompt-override-field completion_prompt \
  --epochs "$EPOCHS" \
  --batch-size "$BS" \
  --lr-muon "$LR_MUON" \
  --lr-adam "$LR_ADAM" \
  --warmup "$WARMUP" \
  --clip "$CLIP" \
  --pack-len "$PACK_LEN" \
  --seed "$SEED" \
  --out "$OUT" <<'PY'
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

"$PY" - "$PREFLIGHT" "$OUT" "$ARM" "$SEED" <<'PY'
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
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "tokenizers": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu, "nvidia_smi": nvidia_smi,
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def write_immutable(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable receipt write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


preflight_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
arm = sys.argv[3]
seed_text = sys.argv[4]
seed = int(seed_text)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
preflight = json.loads(preflight_path.read_text())
if preflight["schema"] != "rsp_c1_sft_preflight_v1" or preflight["arm"] != arm or preflight["paired_seed"] != seed:
    raise SystemExit("[rsp-sft] preflight identity mismatch")
for label, record in preflight["records"].items():
    if sha256_file(record["source"]) != record["sha256"] or sha256_file(record["snapshot"]) != record["sha256"]:
        raise SystemExit(f"[rsp-sft] final input rehash failed: {label}")
if runtime_fingerprint() != preflight["runtime"]:
    raise SystemExit("[rsp-sft] runtime fingerprint changed during fit")

checkpoint = out_path / "sft_ep10.pt"
metadata_path = out_path / "sft_metadata.json"
if not checkpoint.is_file() or not metadata_path.is_file():
    raise SystemExit("[rsp-sft] missing claim-bearing checkpoint or metadata")
for claim_path in (checkpoint, metadata_path):
    if not stat.S_ISREG(claim_path.lstat().st_mode):
        raise SystemExit(f"[rsp-sft] claim-bearing output is not a regular file: {claim_path}")
metadata = json.loads(metadata_path.read_text())
geometry = preflight["geometry"]
active_data = "treatment_data" if arm == "treatment" else "sham_data"
expected_metadata = {
    "seed": seed,
    "pack_len": preflight["fit"]["pack_len"],
    "packed_sequences": geometry["packed_sequences"],
    "init_sha256": preflight["records"]["init"]["sha256"],
}
for key, expected in expected_metadata.items():
    if metadata.get(key) != expected:
        raise SystemExit(f"[rsp-sft] trainer metadata mismatch for {key}")
data_hashes = metadata.get("data_sha256")
if not isinstance(data_hashes, dict) or list(data_hashes.values()) != [preflight["records"][active_data]["sha256"]]:
    raise SystemExit("[rsp-sft] trainer metadata data binding failed")
loaded = torch.load(checkpoint, map_location="cpu")
if loaded.get("step") != "sft_ep10":
    raise SystemExit("[rsp-sft] checkpoint epoch marker is ambiguous")
slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
if not slurm_job_id.isdigit():
    raise SystemExit("[rsp-sft] final fit receipt requires a numeric Slurm job ID")

evidence_path = out_path / "pair_equality_evidence.json"
evidence_payload = preflight_path.read_bytes()
evidence_sha = write_immutable(evidence_path, evidence_payload)
checkpoint_sha = sha256_file(checkpoint)
metadata_sha = sha256_file(metadata_path)
receipt = {
    "schema": "rsp_c1_fit_receipt_v1",
    "arm": arm,
    "paired_seed": seed,
    "run_manifest_sha256": preflight["run_manifest_sha256"],
    "checkpoint_step": "sft_ep10",
    "artifacts": {
        "checkpoint_sha256": checkpoint_sha,
        "metadata_sha256": metadata_sha,
        "pair_equality_sha256": evidence_sha,
        "init_sha256": preflight["records"]["init"]["sha256"],
        "tokenizer_sha256": preflight["records"]["tokenizer"]["sha256"],
        "data_sha256": preflight["records"][active_data]["sha256"],
        "treatment_data_sha256": preflight["records"]["treatment_data"]["sha256"],
        "sham_data_sha256": preflight["records"]["sham_data"]["sha256"],
        "training_manifest_sha256": preflight["records"]["training_manifest"]["sha256"],
        "admission_audit_sha256": preflight["records"]["admission_audit"]["sha256"],
        "preregistration_sha256": preflight["records"]["preregistration"]["sha256"],
    },
    "sources": {name + "_sha256": preflight["records"][name]["sha256"] for name in ("job", "trainer", "encoding", "model", "muon", "protocol")},
    "fit": preflight["fit"],
    "accounting": {key: geometry[key] for key in (
        "full_tokens_before_packing", "packed_sequences", "discarded_tail_tokens",
        "steps_per_epoch", "optimizer_updates", "packed_forward_token_positions",
        "supervised_completion_tokens",
    )},
    "pair_geometry_sha256": geometry["pair_geometry_sha256"],
    "runtime": preflight["runtime"],
    "slurm_job_id": slurm_job_id,
}
receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
receipt_path = out_path / "fit_receipt.json"
receipt_sha = write_immutable(receipt_path, receipt_payload)
for child in out_path.iterdir():
    if not stat.S_ISREG(child.lstat().st_mode):
        raise SystemExit(f"[rsp-sft] unexpected nonregular fit output: {child}")
    os.chmod(child, 0o444)
os.chmod(out_path, 0o555)
final_hashes = {
    checkpoint: checkpoint_sha,
    metadata_path: metadata_sha,
    evidence_path: evidence_sha,
    receipt_path: receipt_sha,
}
for path, expected in final_hashes.items():
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222 or sha256_file(path) != expected:
        raise SystemExit(f"[rsp-sft] sealed output postcondition failed: {path}")
print(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "fit_receipt": str(receipt_path), "fit_receipt_sha256": receipt_sha}, sort_keys=True), flush=True)
PY

echo "[rsp-sft] completed immutable fit arm=$ARM seed=$SEED out=$OUT"
