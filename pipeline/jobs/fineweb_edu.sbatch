#!/bin/bash
#SBATCH -J shohin-fineweb-edu
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_%j.out
#SBATCH --open-mode=append
# Build a high-quality general-language future pretrain source. It is isolated
# under a partial directory until tokenization, decontamination, and manifest
# creation finish; it never changes the running flagship shard list.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/shards/fineweb_edu_5b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-5000000000}
MIN_SCORE=${MIN_SCORE:-3.0}

[ -e "$OUT" ] && { echo "[fineweb-edu] refusing to overwrite $OUT" >&2; exit 2; }
[ -e "$TMP" ] && { echo "[fineweb-edu] refusing stale partial directory $TMP" >&2; exit 2; }

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --text-col text \
  --lang en \
  --lang-field language \
  --min-number-field score \
  --min-number "$MIN_SCORE" \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --min-chars 200 \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
mv "$TMP" "$OUT"
echo "[fineweb-edu] committed $OUT"
