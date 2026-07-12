#!/bin/bash
#SBATCH -J shohin-algocode-pack
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/algocode_packing_%j.out
#SBATCH --open-mode=append
# Read-only packing audit for an execution-verified algorithmic-code candidate.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to a frozen algorithmic-code candidate}
OUT=${OUT:?set OUT to a fresh packing report path}
PACK_LEN=${PACK_LEN:-2048}

[ -s "$DATA" ] || { echo "[algocode-pack] input missing: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[algocode-pack] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --pack-len "$PACK_LEN" --out "$OUT" --weights algorithmic_code=1.0
echo "[algocode-pack] wrote $OUT"
