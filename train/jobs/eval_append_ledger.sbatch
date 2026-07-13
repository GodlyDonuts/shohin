#!/bin/bash
# Isolated closed-loop ADL evaluation. It never writes a training directory.
#SBATCH -J shohin-ledger-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc31,evc36,evc43,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/append_ledger_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a raw or isolated SFT checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/append_ledger_v1_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh result JSON}
PER_REGIME=${PER_REGIME:-100}
MAX_NEW=${MAX_NEW:-96}
STYLE=${STYLE:-auto}

[ -f "$CKPT" ] || { echo "[append-ledger-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[append-ledger-eval] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[append-ledger-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[append-ledger-eval] CUDA unavailable after allocation')
torch.empty(1, device='cuda')
PY

echo "[append-ledger-eval] ckpt=$CKPT episodes=$EPISODES out=$OUT per_regime=$PER_REGIME style=$STYLE"
"$PY" eval_append_ledger.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "$PER_REGIME" --max-new "$MAX_NEW" --prompt-style "$STYLE"
echo "[append-ledger-eval] done"
