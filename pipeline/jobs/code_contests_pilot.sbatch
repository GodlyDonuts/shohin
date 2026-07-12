#!/bin/bash
#SBATCH -J shohin-code-contests
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/code_contests_%j.out
#SBATCH --open-mode=append
# Build an isolated train-only Python 3 CodeContests SFT pilot. The source,
# output, and frozen SFT mixes are never overwritten.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/code_contests_pilot.jsonl}
MAX_SEEN=${MAX_SEEN:-500}
MAX_KEPT=${MAX_KEPT:-250}
[ -e "$OUT" ] && { echo "[code-contests] refusing to overwrite $OUT" >&2; exit 2; }
echo "[code-contests] out=$OUT max_seen=$MAX_SEEN max_kept=$MAX_KEPT"
cd "$BASE/pipeline"
"$PY" curate_code_contests.py \
  --out "$OUT" \
  --evals "$BASE/artifacts/evals" \
  --max-seen "$MAX_SEEN" \
  --max-kept "$MAX_KEPT" \
  --max-tests 3 \
  --timeout 2.0
"$PY" audit_sft_quality.py --data "$OUT" --out "${OUT%.jsonl}.quality.json"
