#!/bin/bash
#SBATCH --job-name=carry-motor
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/carry_motor_%j.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/carry_motor_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-$BASE/.venv/bin/python}
MODE=${MODE:-train}
CKPT=${CKPT:-$BASE/train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
CYCLE=${CYCLE:-$BASE/artifacts/evals/drs_causal_cycle_post_drs_r3.json}
OUT=${OUT:?set OUT to a new isolated artifact path}
MOTOR=${MOTOR:-}
MOTOR_SHA256=${MOTOR_SHA256:-}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed clean commit}
SOURCE_FILES=(
  R12_CAUSAL_CARRY_MOTOR_PREREG.md
  train/causal_carry_motor.py
  train/test_causal_carry_motor.py
  train/jobs/causal_carry_motor.sbatch
  train/digitwise_controller.py
  train/digitwise_protocol.py
  train/eval_suite.py
  train/model.py
  train/probe_digitwise_workspace.py
)

if [[ "${SLURM_RESTART_COUNT:-0}" != 0 ]]; then
  echo "refusing restarted carry-motor job" >&2
  exit 2
fi
if [[ -e "$OUT" || -e "$OUT.json" ]]; then
  echo "refusing existing output: $OUT" >&2
  exit 2
fi
if [[ -e "$(dirname "$OUT")" ]]; then
  echo "canonical output directory must be new: $(dirname "$OUT")" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" != 1 ]]; then
  echo "carry-motor job requires exactly one visible GPU" >&2
  exit 2
fi

cd "$BASE"
export PYTHONPATH="$BASE/train"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "reviewed source commit is not checked out" >&2
  exit 2
fi
SOURCE_MANIFEST_SHA256=$("$PY" - "$SOURCE_COMMIT" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

commit, *paths = sys.argv[1:]
sources = {}
for name in paths:
    payload = Path(name).read_bytes()
    committed = subprocess.check_output(["git", "show", f"{commit}:{name}"])
    if payload != committed:
        raise SystemExit(f"source differs from reviewed commit: {name}")
    sources[name] = hashlib.sha256(payload).hexdigest()
encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
)

if [[ "$MODE" == train ]]; then
  "$PY" train/causal_carry_motor.py train \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" --out "$OUT"
elif [[ "$MODE" == eval ]]; then
  if [[ -z "$MOTOR" || ! -f "$MOTOR" || -z "$MOTOR_SHA256" ]]; then
    echo "eval mode requires MOTOR and MOTOR_SHA256" >&2
    exit 2
  fi
  "$PY" train/causal_carry_motor.py eval \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --motor "$MOTOR" --motor-sha256 "$MOTOR_SHA256" --out "$OUT"
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi
