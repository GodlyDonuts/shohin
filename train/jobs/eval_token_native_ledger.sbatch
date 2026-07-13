#!/bin/bash
# Isolated token-native delta-ledger evaluation. It reads named immutable
# artifacts and writes one fresh JSON; it never writes a training directory.
#SBATCH -J shohin-tnl-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/token_native_ledger_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
CKPT=${CKPT:?set CKPT to a raw or isolated token-native-ledger SFT checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/token_native_ledger_v1_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh result JSON}
PER_REGIME=${PER_REGIME:-100}
MAX_NEW_FINAL=${MAX_NEW_FINAL:-48}
STYLE=${STYLE:-auto}
EXAMPLES_PER_REGIME=${EXAMPLES_PER_REGIME:-0}

[ -f "$CKPT" ] || { echo "[tnl-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[tnl-eval] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[tnl-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[tnl-eval] CUDA unavailable after allocation")
torch.empty(1, device="cuda")
PY
echo "[tnl-eval] ckpt=$CKPT episodes=$EPISODES out=$OUT per_regime=$PER_REGIME style=$STYLE"
"$PY" eval_token_native_ledger.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "$PER_REGIME" \
  --max-new-final "$MAX_NEW_FINAL" --prompt-style "$STYLE" --examples-per-regime "$EXAMPLES_PER_REGIME"
echo "[tnl-eval] done"
