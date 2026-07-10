#!/bin/bash
#SBATCH -J shohin-rg-scale
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rg_scale_%j.out
#SBATCH --open-mode=append
# Generate a larger verifier-backed procedural curriculum in an isolated
# directory. Train and evaluation seeds remain disjoint in the generator.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/rg_v2}
PER_FAMILY=${PER_FAMILY:-20000}
EVAL_PER_FAMILY=${EVAL_PER_FAMILY:-500}

[ -e "$OUT" ] && { echo "[rg-scale] refusing to overwrite existing $OUT"; exit 1; }
[ -e "${OUT}.partial" ] && { echo "[rg-scale] stale partial exists: ${OUT}.partial"; exit 1; }
mkdir -p "${OUT}.partial"
cd "$BASE/pipeline"
"$PY" gen_reasoning_gym.py \
  --out-dir "${OUT}.partial" \
  --per-family "$PER_FAMILY" \
  --eval-per-family "$EVAL_PER_FAMILY"
"$PY" prepare_rg_sft.py \
  --input "${OUT}.partial/rg_traces_train.jsonl" \
  --out "${OUT}.partial/rg_traces_sft.jsonl"
"$PY" audit_sft_quality.py \
  --data "${OUT}.partial/rg_traces_sft.jsonl" \
  --out "${OUT}.partial/rg_traces_sft.quality.json"
mv "${OUT}.partial" "$OUT"
echo "[rg-scale] committed $OUT"
