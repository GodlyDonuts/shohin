#!/bin/bash
#SBATCH -J shohin-prefix-audit
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/prefix_state_audit_%j.out
#SBATCH --open-mode=append
# CPU-only data audit: no model, checkpoint, or shard writer is touched.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to an admitted LSA pair JSONL}
OUT=${OUT:?set OUT to a fresh audit report path}
[ -s "$DATA" ] || { echo "[prefix-audit] missing data: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[prefix-audit] refusing existing report: $OUT" >&2; exit 3; }
cd "$BASE/pipeline"
"$PY" audit_prefix_state_supervision.py --data "$DATA" --out "$OUT"
