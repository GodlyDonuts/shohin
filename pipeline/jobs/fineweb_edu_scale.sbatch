#!/bin/bash
#SBATCH -J shohin-fineweb-edu-scale
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_scale_%j.out
#SBATCH --open-mode=append
# Build a 25B FineWeb-Edu replacement after the accepted 5B pilot. It is a
# future-only source and must receive its own full scan before admission.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
PILOT=${PILOT:-$BASE/artifacts/shards/fineweb_edu_5b}
OUT=${OUT:-$BASE/artifacts/shards/fineweb_edu_25b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-25000000000}
MIN_TOKENS=${MIN_TOKENS:-24500000000}
MIN_SCORE=${MIN_SCORE:-3.0}
DATASET_CONFIG=${DATASET_CONFIG:-sample-100BT}

[ -s "$PILOT/manifest.json" ] || { echo "[fineweb-edu-scale] pilot manifest missing: $PILOT" >&2; exit 2; }
[ "$OUT" != "$PILOT" ] || { echo "[fineweb-edu-scale] output must replace, not overwrite, pilot" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[fineweb-edu-scale] output exists: $OUT" >&2; exit 4; }
[ ! -e "$TMP" ] || { echo "[fineweb-edu-scale] stale partial directory: $TMP" >&2; exit 5; }

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset HuggingFaceFW/fineweb-edu \
  --config "$DATASET_CONFIG" \
  --text-col text \
  --lang en \
  --lang-field language \
  --min-number-field score \
  --min-number "$MIN_SCORE" \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 200 \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
"$PY" - "$TMP/manifest.json" "$MIN_TOKENS" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
tokens = int(manifest.get("total_tokens", manifest.get("tokens", 0)))
required = int(sys.argv[2])
if tokens < required:
    raise SystemExit(
        "[fineweb-edu-scale] corpus is too short for a 25B replacement: "
        f"{tokens} < {required}; partial retained for diagnosis"
    )
print(f"[fineweb-edu-scale] token floor passed: {tokens} >= {required}")
PY
mv "$TMP" "$OUT"
echo "[fineweb-edu-scale] committed $OUT; require a fresh full scan before admission"
