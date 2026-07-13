#!/bin/bash
# Isolated causal diagnostic. It reads an immutable checkpoint and one held-out
# JSONL and writes a fresh result. It never trains or shares a flagship path.
#SBATCH -J shohin-rjdl-probe
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/restricted_jacobian_digit_lens_%j.out
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
LAYERS=${LAYERS:-13,17,21,25}
DISCOVERY_PER_DIGIT=${DISCOVERY_PER_DIGIT:-8}
PAIRS_PER_REGIME=${PAIRS_PER_REGIME:-4}
ALPHA=${ALPHA:-1.0}

[ -f "$CKPT" ] || { echo "[rjdl] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[rjdl] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[rjdl] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[rjdl] CUDA unavailable after allocation")
torch.empty(1, device="cuda")
PY
echo "[rjdl] ckpt=$CKPT episodes=$EPISODES out=$OUT"
"$PY" probe_restricted_jacobian_digit_lens.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --transition-index "$TRANSITION_INDEX" \
  --layers "$LAYERS" --discovery-per-digit "$DISCOVERY_PER_DIGIT" \
  --pairs-per-regime "$PAIRS_PER_REGIME" --alpha "$ALPHA"
echo "[rjdl] done"
