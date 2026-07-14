#!/bin/bash
# Read-only activation-exchange audit for an isolated checkpoint.
# Never writes a pretraining or SFT output directory.
#SBATCH -J shohin-psa-causal
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/psa_causal_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/semantic_basis_transport_v2_factor_language.jsonl}
DATA_SHA=482f303465aef45bb3f046ebdb3a1e5f7bc8d24f4ab5bae8973e8dcf9c7f5a3a
OUT=${OUT:?set OUT to a fresh evaluation-history JSON path}
PAIRS=${PAIRS:-50}
LAYER=${LAYER:-19}
ALPHA=${ALPHA:-1.0}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[psa-causal] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[psa-causal] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[psa-causal] refusing existing output: $OUT" >&2; exit 4; }
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[psa-causal] data hash mismatch" >&2; exit 5; }
case "$OUT" in "$BASE/artifacts/eval_history/"*.json) ;; *) echo "[psa-causal] OUT must be a fresh eval-history JSON" >&2; exit 6;; esac

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[psa-causal] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
echo "[psa-causal] ckpt=$CKPT data=$DATA layer=$LAYER alpha=$ALPHA pairs=$PAIRS out=$OUT"
"$PY" eval_paraphrase_state_causality.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --data "$DATA" --out "$OUT" \
  --split factor_language --pairs "$PAIRS" --layer "$LAYER" --alpha "$ALPHA"
test -s "$OUT"
echo "[psa-causal] wrote $OUT"
