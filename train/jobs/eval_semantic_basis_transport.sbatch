#!/bin/bash
# Isolated exact-carrier evaluation. It reads a named checkpoint and frozen
# held-out data, writes one fresh report, and never writes a training output.
#SBATCH -J shohin-sbasis-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_basis_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/best_step200000.pt}
TAG=${TAG:-semantic_basis_raw200k_v2}
DATA=${DATA:-$BASE/artifacts/evals/semantic_basis_transport_v2_heldout.jsonl}
OUT=${OUT:-$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json}
PAIRS=${PAIRS:-100}
SEED=${SEED:-20260713}
EXAMPLES=${EXAMPLES:-8}
SPLIT=${SPLIT:-heldout}
PROMPT_MODE=${PROMPT_MODE:-qa}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[semantic-basis] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[semantic-basis] held-out data missing: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[semantic-basis] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[semantic-basis] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
echo "[semantic-basis] ckpt=$CKPT data=$DATA split=$SPLIT out=$OUT pairs=$PAIRS prompt_mode=$PROMPT_MODE"
"$PY" eval_semantic_basis_transport.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --data "$DATA" --out "$OUT" --pairs "$PAIRS" --seed "$SEED" --examples "$EXAMPLES" \
  --split "$SPLIT" --prompt-mode "$PROMPT_MODE"
echo "[semantic-basis] wrote $OUT"
