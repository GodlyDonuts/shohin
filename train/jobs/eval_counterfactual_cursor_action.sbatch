#!/bin/bash
#SBATCH -J shohin-r12-ccaa-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --array=0-5%1
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r12_ccaa_eval_%A_%a.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/best_step260000.pt}
CANARY=${CANARY:-$BASE/artifacts/r12/counterfactual_cursor_action_canary_v1.json}
AUDIT=${AUDIT:-$BASE/artifacts/r12/counterfactual_cursor_action_canary_v1.audit.json}
TRAIN_ROOT=${TRAIN_ROOT:-$BASE/train/r12_cursor_action_canary_v1}
MANIFEST=${MANIFEST:-$TRAIN_ROOT/training_manifest.json}
EVAL_ROOT=${EVAL_ROOT:-$BASE/artifacts/r12/cursor_action_eval_v1}

ARMS=(
  orbit_interchange
  ordinary_loss
  relation_sham
  source_only
  cursor_table
  text_cursor_lora
)
TASK_ID=${SLURM_ARRAY_TASK_ID:?missing array task ID}
(( TASK_ID >= 0 && TASK_ID < ${#ARMS[@]} )) || {
  echo "[ccaa-eval-job] invalid array task: $TASK_ID" >&2
  exit 2
}
ARM=${ARMS[$TASK_ID]}
ADAPTER=$TRAIN_ROOT/$ARM/adapter.pt
RAW=$EVAL_ROOT/$ARM.raw.json
RECEIPT=$EVAL_ROOT/$ARM.receipt.json

for path in "$CKPT" "$CANARY" "$AUDIT" "$MANIFEST" "$ADAPTER"; do
  [ -s "$path" ] || { echo "[ccaa-eval-job] missing input: $path" >&2; exit 3; }
  [ ! -L "$path" ] || { echo "[ccaa-eval-job] symlink input rejected: $path" >&2; exit 4; }
  [ ! -w "$path" ] || { echo "[ccaa-eval-job] writable input rejected: $path" >&2; exit 5; }
done
[ ! -e "$RAW" ] || { echo "[ccaa-eval-job] refusing existing raw output: $RAW" >&2; exit 6; }
[ ! -e "$RECEIPT" ] || {
  echo "[ccaa-eval-job] refusing existing receipt output: $RECEIPT" >&2
  exit 7
}
mkdir -p "$EVAL_ROOT"

read -r BASE_SHA ADAPTER_SHA CANARY_SHA AUDIT_SHA MANIFEST_SHA < <(
  "$PY" - "$CKPT" "$ADAPTER" "$CANARY" "$AUDIT" "$MANIFEST" <<'PY'
import hashlib
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


print(*(digest(path) for path in sys.argv[1:]))
PY
)

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    value = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    value.add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[ccaa-eval-job] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
print("[ccaa-eval-job] CUDA preflight passed")
PY

"$PY" eval_counterfactual_cursor_action.py \
  --base "$CKPT" \
  --base-sha256 "$BASE_SHA" \
  --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA" \
  --training-manifest "$MANIFEST" \
  --training-manifest-sha256 "$MANIFEST_SHA" \
  --canary "$CANARY" \
  --canary-sha256 "$CANARY_SHA" \
  --audit "$AUDIT" \
  --audit-sha256 "$AUDIT_SHA" \
  --out-raw "$RAW" \
  --out-receipt "$RECEIPT" \
  --device cuda \
  --batch-size 32 \
  --conditions canonical,clamped_zero,deranged_cycle \
  --scheduler slurm \
  --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
  --array-task-id "$TASK_ID" \
  --attempt-id "${SLURM_RESTART_COUNT:-0}"
