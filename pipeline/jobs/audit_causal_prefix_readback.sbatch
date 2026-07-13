#!/bin/bash
#SBATCH -J shohin-causal-prefix-audit
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/causal_prefix_readback_audit_%j.out
#SBATCH --open-mode=append
# CPU-only label audit. Prefer Stokes; it never writes data, checkpoints, or shards.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to admitted LSA pair JSONL}
OUT=${OUT:?set OUT to a fresh audit report path}
[ -s "$DATA" ] || { echo "[causal-prefix-audit] missing data: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[causal-prefix-audit] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/pipeline"
"$PY" audit_causal_prefix_readback.py --data "$DATA" --out "$OUT"
