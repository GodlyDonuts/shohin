#!/bin/bash
# CPU-only aggregation of already-written CRA reports. No model or corpus is touched.
#SBATCH -J shohin-cra-gate
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_gate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:?set a fresh gate output}
TAG=${TAG:-cra_200k_l19_r1}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1

[ ! -e "$OUT" ] || { echo "[cra-gate] refusing existing output" >&2; exit 2; }
case "$OUT" in "$BASE"/artifacts/eval_history/cra_*_decision.json) ;; *) echo "[cra-gate] OUT must be CRA decision artifact" >&2; exit 3;; esac
cd "$BASE"
"$PY" pipeline/assess_counterfactual_residual_algebra.py \
  --raw "$BASE/artifacts/eval_history/cra_raw200k_l19_r1_combined.json" \
  --raw-nll "$BASE/artifacts/eval_history/cra_raw200k_l19_r1_combined_nll.json" \
  --combined "$BASE/artifacts/eval_history/${TAG}_combined.json" \
  --combined-nll "$BASE/artifacts/eval_history/${TAG}_combined_nll.json" \
  --train "$BASE/artifacts/eval_history/${TAG}_train_diagnostic.json" \
  --language "$BASE/artifacts/eval_history/${TAG}_language.json" \
  --values "$BASE/artifacts/eval_history/${TAG}_values.json" \
  --delta "$BASE/artifacts/eval_history/${TAG}_delta.json" \
  --query "$BASE/artifacts/eval_history/${TAG}_query.json" \
  --two-edit "$BASE/artifacts/eval_history/${TAG}_two_edit.json" \
  --out "$OUT"
test -s "$OUT"
echo "[cra-gate] wrote $OUT"
