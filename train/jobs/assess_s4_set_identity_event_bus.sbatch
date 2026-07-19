#!/bin/bash
#SBATCH -J shohin-s4-v3-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:05:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s4_v3_assess_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
TREATMENT=${TREATMENT:?set TREATMENT report}
SHUFFLED=${SHUFFLED:?set SHUFFLED report}
BASELINE=${BASELINE:?set BASELINE report}
OUT=${OUT:?set OUT assessment}
for path in "$TREATMENT" "$SHUFFLED" "$BASELINE"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing $OUT" >&2; exit 2; }
cd "$BASE"
"$PYTHON" train/assess_s4_set_identity_event_bus.py \
  --treatment "$TREATMENT" --shuffled "$SHUFFLED" --baseline "$BASELINE" --out "$OUT"
