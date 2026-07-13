#!/bin/bash
# Isolated source-free factorized DRS evaluation. It writes only a fresh JSON.
#SBATCH -J shohin-factor-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_factor_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a raw or isolated factor SFT checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_factor_v1_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh result JSON}
PER_REGIME=${PER_REGIME:-100}
MAX_NEW=${MAX_NEW:-96}
STYLE=${STYLE:-auto}

[ -f "$CKPT" ] || { echo "[factor-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[factor-eval] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[factor-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[factor-eval] CUDA unavailable after allocation")
torch.empty(1, device="cuda")
PY
echo "[factor-eval] ckpt=$CKPT episodes=$EPISODES out=$OUT per_regime=$PER_REGIME style=$STYLE"
"$PY" eval_digitwise_factor.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "$PER_REGIME" \
  --max-new "$MAX_NEW" --prompt-style "$STYLE"
echo "[factor-eval] done"
