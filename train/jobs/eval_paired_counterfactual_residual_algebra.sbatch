#!/bin/bash
# Evaluate the paired CRA fallback only when the first CRA gate rejected it.
#SBATCH -J shohin-cra-paired-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_paired_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DECISION=${DECISION:-$BASE/artifacts/eval_history/cra_200k_l19_r1_decision.json}
CKPT=${CKPT:-$BASE/train/cra_paired_200k_l19_r1/cra_ep1.pt}
OUT=${OUT:-$BASE/artifacts/eval_history/cra_paired_200k_l19_r1_combined.json}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [ "$("$PY" - "$DECISION" <<'PY'
import json,sys
print('1' if json.load(open(sys.argv[1])).get('decision') == 'reject_cra_l19_r1_no_recurrence_or_public_benchmark' else '0')
PY
)" != 1 ]; then
  echo "[cra-paired-eval] first CRA arm did not authorize fallback evaluation"
  exit 0
fi
[ -s "$CKPT" ] || { echo "[cra-paired-eval] paired checkpoint missing" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[cra-paired-eval] refusing existing output" >&2; exit 3; }
export CKPT OUT DATA="$BASE/artifacts/evals/counterfactual_residual_algebra_v1_heldout.jsonl"
export DATA_SHA=f66536b85cfe2fafa71690a0e3a5a2adc8fb5db0dd5b912ad69f8362c927b248 ROWS=500 SPLIT=heldout
exec bash "$BASE/train/jobs/eval_counterfactual_residual_algebra.sbatch"
