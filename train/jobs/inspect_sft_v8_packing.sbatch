#!/bin/bash
#SBATCH -J shohin-sft-v8-audit
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v8_audit_%j.out
#SBATCH --open-mode=append
# Read-only V8 replay-capacity audit. Passing data quality is not enough to
# justify oversampling a scarce group during SFT.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.packing.json}
WEIGHTS=${WEIGHTS:-math=0.40 procedural=0.45 code=0.10 teacher=0.05}

[ -s "$DATA" ] || { echo "[sft-v8-audit] data missing: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[sft-v8-audit] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --weights $WEIGHTS
echo "[sft-v8-audit] wrote $OUT"
