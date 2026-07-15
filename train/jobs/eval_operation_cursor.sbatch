#!/bin/bash
# Read-only raw-260k operation-cursor diagnostic. This wrapper writes one
# immutable transcript under artifacts/eval_history and performs no training.
# It does not submit itself or any downstream job.
#SBATCH -J shohin-op-cursor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc34,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_operation_cursor_%j.out
#SBATCH --open-mode=append
set -euo pipefail
umask 077

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CKPT=${CKPT:-$BASE/best_step260000.pt}
SOURCE=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
EVAL_ROOT=$BASE/artifacts/eval_history
OUT=${OUT:?set OUT to a fresh raw260k_operation_cursor_*.json path under artifacts/eval_history}

EXPECTED_CKPT=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d
EXPECTED_TOKENIZER=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
EXPECTED_SOURCE=19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474

for path in "$PY" "$CKPT" "$SOURCE" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[operation-cursor] missing required input: $path" >&2; exit 2; }
done

BASE=$(realpath -e -- "$BASE")
PY=$(realpath -e -- "$PY")
CKPT=$(realpath -e -- "$CKPT")
SOURCE=$(realpath -e -- "$SOURCE")
TOKENIZER=$(realpath -e -- "$TOKENIZER")
mkdir -p -- "$EVAL_ROOT"
EVAL_ROOT=$(realpath -e -- "$EVAL_ROOT")
OUT=$(realpath -m -- "$OUT")

if [ "$(dirname -- "$OUT")" != "$EVAL_ROOT" ]; then
  echo "[operation-cursor] OUT must be a direct child of $EVAL_ROOT" >&2
  exit 3
fi
case "$(basename -- "$OUT")" in
  raw260k_operation_cursor_*.json) ;;
  *) echo "[operation-cursor] OUT has the wrong frozen filename pattern" >&2; exit 3 ;;
esac
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "[operation-cursor] refusing existing output: $OUT" >&2
  exit 3
fi
if [ -w "$SOURCE" ]; then
  echo "[operation-cursor] frozen source must be read-only" >&2
  exit 4
fi

check_sha256() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum -- "$path" | awk '{print $1}')
  if [ "$observed" != "$expected" ]; then
    echo "[operation-cursor] SHA-256 mismatch for $path" >&2
    exit 5
  fi
}

check_sha256 "$CKPT" "$EXPECTED_CKPT"
check_sha256 "$TOKENIZER" "$EXPECTED_TOKENIZER"
check_sha256 "$SOURCE" "$EXPECTED_SOURCE"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0

"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[operation-cursor] CUDA unavailable")
torch.empty(1, device="cuda")
print("[operation-cursor] CUDA preflight passed", flush=True)
PY

cd "$BASE/train"
"$PY" eval_operation_cursor.py \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --source "$SOURCE" \
  --out "$OUT" \
  --device cuda

[ -s "$OUT" ] || { echo "[operation-cursor] evaluator wrote no result" >&2; exit 6; }
[ ! -w "$OUT" ] || { echo "[operation-cursor] output remained writable" >&2; exit 6; }

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
if result.get("schema") != "raw260k_operation_cursor_diagnostic_v1":
    raise SystemExit("[operation-cursor] wrong result schema")
if result.get("resource_ledger", {}).get("model_calls") != 528:
    raise SystemExit("[operation-cursor] wrong preserved model-call count")
if result.get("summary", {}).get("transition_count") != 176:
    raise SystemExit("[operation-cursor] wrong preserved transition count")
if result.get("diagnostic_scope", {}).get("reasoning_claim") != "none":
    raise SystemExit("[operation-cursor] result crossed its claim boundary")
print("[operation-cursor] immutable diagnostic postflight passed", flush=True)
PY

sha256sum -- "$OUT"
