#!/bin/bash
#SBATCH -J shohin-openmath-pt
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openmath_pt_%j.out
#SBATCH --open-mode=append
# Stream a reasoning-dense OpenMath document corpus directly into compressed
# uint16 shards. The output is a new future-relaunch candidate, never an
# in-place edit of SHARDS used by the active flagship.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/shards/openmath_pt}
MAX_TOKENS=${MAX_TOKENS:-5000000000}

[ -e "$OUT" ] && { echo "[openmath-pt] refusing to overwrite existing $OUT"; exit 1; }
cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset nvidia/OpenMathInstruct-2 \
  --text-cols problem generated_solution \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --min-chars 200 \
  --out-dir "$OUT" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"
test -s "$OUT/manifest.json"
echo "[openmath-pt] committed $OUT"
