#!/bin/bash
#SBATCH -J shohin-source-memory
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/source_memory_train_%j.out
#SBATCH --open-mode=append
# Isolated source-removal memory training. It cannot write flagship paths.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to an immutable pretraining checkpoint}
DATA=${DATA:?set DATA to the admitted source-memory train JSONL}
AUDIT=${AUDIT:?set AUDIT to the matching source-memory audit JSON}
OUT=${OUT:?set OUT to a fresh isolated output directory}

for path in "$INIT" "$DATA" "$AUDIT"; do [ -s "$path" ] || { echo "[source-memory] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[source-memory] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[source-memory] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" source_dropping_memory_train.py --init "$INIT" --data "$DATA" --audit "$AUDIT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --slots "${SLOTS:-8}" --max-chunks "${MAX_CHUNKS:-8}" --epochs "${EPOCHS:-1}" \
  --batch-size "${BS:-4}" --max-examples "${MAX_EXAMPLES:-0}" --seed "${SEED:-20260714}"
