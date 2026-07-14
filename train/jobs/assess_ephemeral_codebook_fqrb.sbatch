#!/bin/bash
# CPU-only, one-shot ECLI assessment. It never allocates a GPU or submits work.
#SBATCH -J shohin-ecli-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ecli_assess_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
REPORT=${REPORT:?set ECLI evaluation report}
AUDIT=${AUDIT:?set ECLI frozen audit}
OUT=${OUT:?set fresh ECLI assessment output}

[ -s "$REPORT" ] || { echo "[ecli-assess] report missing" >&2; exit 2; }
[ -s "$AUDIT" ] || { echo "[ecli-assess] audit missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[ecli-assess] output exists" >&2; exit 4; }
cd "$BASE"
"$PY" pipeline/assess_ephemeral_codebook_fqrb_v1.py --report "$REPORT" --audit "$AUDIT" --out "$OUT"
test -s "$OUT"
echo "[ecli-assess] wrote $OUT"
