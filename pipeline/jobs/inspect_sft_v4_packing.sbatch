#!/bin/bash
#SBATCH -J shohin-sft-v4-audit
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v4_audit_%j.out
#SBATCH --open-mode=append
# Read-only packed-sequence capacity audit. It must complete before the v4 pilot
# chooses source weights, so a small group cannot be blindly over-replayed.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl
OUT=$BASE/artifacts/sft/sft_mix_reasoning_v4.packing.json
WEIGHTS="math=0.40 procedural=0.35 code=0.15 teacher=0.10"

[ -s "$DATA" ] || { echo "[sft-v4-audit] frozen mix missing: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[sft-v4-audit] refusing to overwrite $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --weights $WEIGHTS
echo "[sft-v4-audit] wrote $OUT"
