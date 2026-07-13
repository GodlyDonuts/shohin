#!/bin/bash
# Isolated residual-patching diagnostic. It reads a named checkpoint and held-out
# episodes and writes one fresh JSON; it never trains or writes a model output.
#SBATCH -J shohin-workspace-probe
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/workspace_probe_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
CKPT=${CKPT:?set CKPT to an immutable raw or isolated SFT checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh result JSON}
TRANSITION_INDEX=${TRANSITION_INDEX:-2}
PAIRS_PER_REGIME=${PAIRS_PER_REGIME:-1}
LAYERS=${LAYERS:-5,9,13,17,21,25,29}
ALPHA=${ALPHA:-1.0}

[ -f "$CKPT" ] || { echo "[workspace-probe] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[workspace-probe] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[workspace-probe] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[workspace-probe] CUDA unavailable after allocation")
torch.empty(1, device="cuda")
PY
echo "[workspace-probe] ckpt=$CKPT episodes=$EPISODES out=$OUT transition=$TRANSITION_INDEX pairs=$PAIRS_PER_REGIME layers=$LAYERS alpha=$ALPHA"
"$PY" probe_digitwise_workspace.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --transition-index "$TRANSITION_INDEX" \
  --pairs-per-regime "$PAIRS_PER_REGIME" --layers "$LAYERS" --alpha "$ALPHA"
echo "[workspace-probe] done"
