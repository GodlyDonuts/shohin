#!/bin/bash
# Isolated closed-loop DRS evaluation. It never writes a training directory.
#SBATCH -J shohin-drs-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# DRS-specific CUDA exclusions. `evc23`/`evc32` stalled before framework
# startup and `evc44` failed a real CUDA allocation; do not generalize this
# list to the flagship scheduler policy.
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a raw or isolated SFT checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh result JSON}
PER_REGIME=${PER_REGIME:-100}
MAX_NEW=${MAX_NEW:-96}
STYLE=${STYLE:-auto}
EXAMPLES_PER_REGIME=${EXAMPLES_PER_REGIME:-0}

[ -f "$CKPT" ] || { echo "[drs-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[drs-eval] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[drs-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("[drs-eval] CUDA unavailable after allocation")
torch.empty(1, device="cuda")
PY

echo "[drs-eval] ckpt=$CKPT episodes=$EPISODES out=$OUT per_regime=$PER_REGIME style=$STYLE examples_per_regime=$EXAMPLES_PER_REGIME"
"$PY" eval_digitwise_recurrent.py \
  --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" \
  --out "$OUT" \
  --per-regime "$PER_REGIME" \
  --max-new "$MAX_NEW" \
  --prompt-style "$STYLE" \
  --examples-per-regime "$EXAMPLES_PER_REGIME"
echo "[drs-eval] done"
