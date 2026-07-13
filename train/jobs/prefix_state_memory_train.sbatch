#!/bin/bash
#SBATCH -J shohin-prefix-state
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/prefix_state_memory_train_%j.out
#SBATCH --open-mode=append
# Isolated prefix-supervised source-free memory research. Never writes flagship paths.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to an immutable pretraining checkpoint}
DATA=${DATA:?set DATA to admitted LSA pair JSONL}
AUDIT=${AUDIT:?set AUDIT to matching LSA audit JSON}
OUT=${OUT:?set OUT to a fresh isolated output directory}

for path in "$INIT" "$DATA" "$AUDIT"; do [ -s "$path" ] || { echo "[prefix-state] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[prefix-state] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[prefix-state] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
args=(
  --init "$INIT" --data "$DATA" --audit "$AUDIT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
  --slots "${SLOTS:-8}" --max-chunks "${MAX_CHUNKS:-8}" --epochs "${EPOCHS:-1}" --batch-size "${BS:-4}"
  --max-pairs "${MAX_PAIRS:-0}" --seed "${SEED:-20260716}" --prefix-mode "${PREFIX_MODE:-verified}"
  --weight-state "${W_STATE:-1.0}" --weight-delta "${W_DELTA:-0.5}"
)
[ "${ZERO_AUXILIARY:-0}" = 1 ] && args+=(--zero-auxiliary)
"$PY" prefix_state_memory_train.py "${args[@]}"
