#!/bin/bash
#SBATCH -J shohin-dclm
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dclm_baseline_%j.out
#SBATCH --open-mode=append
# Build a decontaminated English curriculum source in a fresh atomic directory.
# It never mutates the live flagship's SHARDS or any frozen SFT artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/shards/dclm_baseline_5b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-5000000000}
MIN_EN_PROB=${MIN_EN_PROB:-0.90}

[ -e "$OUT" ] && { echo "[dclm] refusing to overwrite $OUT" >&2; exit 2; }
[ -e "$TMP" ] && { echo "[dclm] refusing stale partial directory $TMP" >&2; exit 2; }

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset mlfoundations/dclm-baseline-1.0 \
  --config default \
  --text-col text \
  --min-number-field language_id_whole_page_fasttext.en \
  --min-number "$MIN_EN_PROB" \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --min-chars 200 \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
mv "$TMP" "$OUT"
echo "[dclm] committed $OUT"
