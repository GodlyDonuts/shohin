#!/bin/bash
#SBATCH -J shohin-r12-token-tape
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r12_cursor_token_tape_%j.out
#SBATCH --open-mode=append
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PYTHONNOUSERSITE=1

BASE=/lustre/fs1/home/sa305415/shohin
CKPT=${CKPT:-$BASE/train/flagship_out/best_step260000.pt}
VIEW=${VIEW:-$BASE/artifacts/r12/counterfactual_cursor_action_dev_view_v1.json}
VIEW_AUDIT=${VIEW_AUDIT:-$BASE/artifacts/r12/counterfactual_cursor_action_dev_view_v1.audit.json}
OUT=${OUT:-$BASE/artifacts/r12/cursor_token_tape_dev_v1.json}
IMPLEMENTATION_COMMIT=${IMPLEMENTATION_COMMIT:?missing implementation commit}
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/miniforge-torch-runtime-v1.tar}
RUNTIME_TAR_SHA256=af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c
BASE_SHA256=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d
VIEW_SHA256=24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9
VIEW_AUDIT_SHA256=33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25

for input in "$CKPT" "$VIEW" "$VIEW_AUDIT" "$RUNTIME_TAR"; do
  [ -s "$input" ] || { echo "[token-tape-job] missing input: $input" >&2; exit 2; }
  [ ! -L "$input" ] || { echo "[token-tape-job] symlink input rejected: $input" >&2; exit 3; }
  [ ! -w "$input" ] || { echo "[token-tape-job] writable input rejected: $input" >&2; exit 4; }
done
[ ! -e "$OUT" ] || { echo "[token-tape-job] refusing existing output: $OUT" >&2; exit 5; }

STAGE_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
STAGE_DIR=$(mktemp -d "$STAGE_PARENT/shohin_token_tape.${SLURM_JOB_ID:-manual}.XXXXXX")
chmod 0700 "$STAGE_DIR"
cleanup_stage() {
  chmod -R u+w "${STAGE_DIR:-}" 2>/dev/null || true
  rm -rf -- "${STAGE_DIR:-}"
}
trap cleanup_stage EXIT

git -C "$BASE" cat-file -e "${IMPLEMENTATION_COMMIT}^{commit}"
SOURCE_DIR=$STAGE_DIR/source
mkdir -m 0700 "$SOURCE_DIR"
for relative in \
  R12_CURSOR_TOKEN_TAPE_DIAGNOSTIC.md \
  train/model.py \
  train/counterfactual_cursor_action_dev_view.py \
  train/probe_counterfactual_cursor_readout.py \
  train/probe_counterfactual_cursor_token_tape.py \
  train/test_probe_counterfactual_cursor_token_tape.py \
  train/jobs/probe_counterfactual_cursor_token_tape.sbatch; do
  mkdir -p "$SOURCE_DIR/$(dirname "$relative")"
  git -C "$BASE" show "${IMPLEMENTATION_COMMIT}:${relative}" > "$SOURCE_DIR/$relative"
done
printf '%s\n' "$IMPLEMENTATION_COMMIT" > "$SOURCE_DIR/.implementation_commit"
chmod -R a-w "$SOURCE_DIR"

INPUT_DIR=$STAGE_DIR/inputs
mkdir -m 0700 "$INPUT_DIR"
cp "$CKPT" "$INPUT_DIR/base.pt"
cp "$VIEW" "$INPUT_DIR/view.json"
cp "$VIEW_AUDIT" "$INPUT_DIR/view.audit.json"
cp "$RUNTIME_TAR" "$INPUT_DIR/runtime.tar"
chmod 0444 "$INPUT_DIR"/*
check_hash() {
  local path=$1
  local expected=$2
  local label=$3
  local observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [ "$observed" = "$expected" ] || {
    echo "[token-tape-job] $label SHA-256 mismatch" >&2
    exit 6
  }
}
check_hash "$INPUT_DIR/base.pt" "$BASE_SHA256" base
check_hash "$INPUT_DIR/view.json" "$VIEW_SHA256" development-view
check_hash "$INPUT_DIR/view.audit.json" "$VIEW_AUDIT_SHA256" development-view-audit
check_hash "$INPUT_DIR/runtime.tar" "$RUNTIME_TAR_SHA256" runtime

RUNTIME_DIR=$STAGE_DIR/runtime
mkdir -m 0700 "$RUNTIME_DIR"
START=$(date +%s)
tar -C "$RUNTIME_DIR" -xf "$INPUT_DIR/runtime.tar"
PY=$RUNTIME_DIR/miniforge3/bin/python
[ -x "$PY" ] || { echo "[token-tape-job] extracted Python missing" >&2; exit 7; }
echo "[token-tape-job] runtime staged sha256=$RUNTIME_TAR_SHA256 seconds=$(( $(date +%s) - START ))"

cd "$SOURCE_DIR/train"
timeout --kill-after=10s 120s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    value = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    value.add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[token-tape-job] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
print("[token-tape-job] CUDA preflight passed")
PY

"$PY" probe_counterfactual_cursor_token_tape.py \
  --base "$INPUT_DIR/base.pt" \
  --development-view "$INPUT_DIR/view.json" \
  --development-view-audit "$INPUT_DIR/view.audit.json" \
  --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --runtime-sha256 "$RUNTIME_TAR_SHA256" \
  --out "$OUT" \
  --cache-batch-size 64
