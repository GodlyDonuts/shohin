#!/bin/bash
# Conditional post-CRA fallback. It only starts if the preregistered first CRA
# arm is rejected; otherwise it exits without touching any output or model.
#SBATCH -J shohin-cra-paired
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_paired_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DECISION=${DECISION:-$BASE/artifacts/eval_history/cra_200k_l19_r1_decision.json}
OUT=${OUT:-$BASE/train/cra_paired_200k_l19_r1}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

"$PY" - "$DECISION" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
decision = report.get('decision', '')
if decision == 'factor_review_required_before_any_extension':
    print('[cra-paired] first CRA arm passed combined gate; paired fallback intentionally skipped')
    raise SystemExit(0)
if decision != 'reject_cra_l19_r1_no_recurrence_or_public_benchmark':
    raise SystemExit('[cra-paired] decision does not authorize fallback: {}'.format(decision))
print('[cra-paired] first CRA arm rejected; launching paired causal-discrimination fallback')
PY
if [ "$("$PY" - "$DECISION" <<'PY'
import json,sys
print('1' if json.load(open(sys.argv[1])).get('decision') == 'reject_cra_l19_r1_no_recurrence_or_public_benchmark' else '0')
PY
)" != 1 ]; then
  exit 0
fi

[ ! -e "$OUT" ] || { echo "[cra-paired] refusing existing output" >&2; exit 2; }
export OUT PAIRED_COUNTERFACTUAL=1 CONTRAST_MARGIN=${CONTRAST_MARGIN:-0.2}
exec bash "$BASE/train/jobs/train_counterfactual_residual_algebra.sbatch"
