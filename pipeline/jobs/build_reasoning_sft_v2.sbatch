#!/bin/bash
#SBATCH -J shohin-sft-mix-v2
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_mix_v2_%j.out
#SBATCH --open-mode=append
# Freeze the next SFT candidate only after its independent source jobs finish.
# It is intentionally a new artifact: never replace sft_mix_core while a run may
# depend on that file.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=$BASE/artifacts/sft/sft_mix_reasoning_v2.jsonl
REPORT=$BASE/artifacts/sft/sft_mix_reasoning_v2.report.json
QUALITY=$BASE/artifacts/sft/sft_mix_reasoning_v2.quality.json

[ ! -e "$OUT" ] || { echo "[sft-v2] refusing to overwrite $OUT"; exit 1; }
cd "$BASE"
"$PY" pipeline/build_sft_mix.py \
  --inputs \
    "$BASE/artifacts/sft/openmath2_600k.jsonl" \
    "$BASE/artifacts/rg_v2/rg_traces_sft.jsonl" \
    "$BASE/artifacts/sft/code.jsonl" \
    "$BASE/artifacts/sft/hy3_reasoning_claude.jsonl" \
    "$BASE/artifacts/sft/hy3_reasoning_glm.jsonl" \
    "$BASE/artifacts/sft/hy3_reasoning_nemotron.jsonl" \
    "$BASE/artifacts/sft/hy3_reasoning.jsonl" \
    "$BASE/artifacts/sft/hy3_gsm8k.jsonl" \
    "$BASE/artifacts/sft/hy3_arc_challenge.jsonl" \
  --out "$OUT" --report "$REPORT" --max-openmath 600000
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$QUALITY"
echo "[sft-v2] committed $OUT"
