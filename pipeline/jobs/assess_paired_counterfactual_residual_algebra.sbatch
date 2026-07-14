#!/bin/bash
# Aggregate paired CRA evidence only when the first CRA arm rejected the primitive.
# This CPU-only gate never trains, evaluates, or changes a model checkpoint.
#SBATCH -J shohin-cra-paired-gate
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_paired_gate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DECISION=${DECISION:-$BASE/artifacts/eval_history/cra_200k_l19_r1_decision.json}
OUT=${OUT:-$BASE/artifacts/eval_history/cra_paired_200k_l19_r1_decision.json}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1

if [ "$("$PY" - "$DECISION" <<'PY'
import json,sys
print('1' if json.load(open(sys.argv[1])).get('decision') == 'reject_cra_l19_r1_no_recurrence_or_public_benchmark' else '0')
PY
)" != 1 ]; then
  echo "[cra-paired-gate] first CRA arm did not authorize fallback assessment"
  exit 0
fi
[ ! -e "$OUT" ] || { echo "[cra-paired-gate] refusing existing output" >&2; exit 2; }
case "$OUT" in "$BASE"/artifacts/eval_history/cra_paired_200k_l19_r1_decision.json) ;; *) echo "[cra-paired-gate] invalid output path" >&2; exit 3;; esac
exec env TAG=cra_paired_200k_l19_r1 OUT="$OUT" bash "$BASE/pipeline/jobs/assess_counterfactual_residual_algebra.sbatch"
