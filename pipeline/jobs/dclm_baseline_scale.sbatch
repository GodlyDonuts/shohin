#!/bin/bash
#SBATCH -J shohin-dclm-scale
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dclm_baseline_scale_%j.out
#SBATCH --open-mode=append
# Build a larger DCLM replacement only after the 5B pilot's manifest and full
# shard scan have been manually accepted. The larger source replaces, rather
# than supplements, the pilot at a future flagship handoff so the first 5B
# prefix is not double-counted.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
PILOT=${PILOT:-$BASE/artifacts/shards/dclm_baseline_5b}
OUT=${OUT:-$BASE/artifacts/shards/dclm_baseline_25b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-25000000000}
MIN_EN_PROB=${MIN_EN_PROB:-0.90}

[ -s "$PILOT/manifest.json" ] || { echo "[dclm-scale] pilot manifest missing: $PILOT" >&2; exit 2; }
[ "$OUT" != "$PILOT" ] || { echo "[dclm-scale] output must replace, not overwrite, pilot" >&2; exit 3; }
[ -e "$OUT" ] && { echo "[dclm-scale] refusing to overwrite $OUT" >&2; exit 4; }
[ -e "$TMP" ] && { echo "[dclm-scale] refusing stale partial directory $TMP" >&2; exit 5; }

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset mlfoundations/dclm-baseline-1.0 \
  --config default \
  --text-col text \
  --min-number-field language_id_whole_page_fasttext.en \
  --min-number "$MIN_EN_PROB" \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 200 \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
mv "$TMP" "$OUT"
echo "[dclm-scale] committed $OUT; require a fresh full scan before admission"
