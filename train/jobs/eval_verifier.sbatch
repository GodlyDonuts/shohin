#!/bin/bash
#SBATCH -J shohin-verifier-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
VERIFIER=${VERIFIER:?set VERIFIER}
ROLLOUTS=${ROLLOUTS:?set ROLLOUTS to held-out candidates}
OUT=${OUT:?set OUT}
[ -f "$VERIFIER" ] || { echo "[verifier-eval] missing verifier: $VERIFIER" >&2; exit 2; }
[ -f "$ROLLOUTS" ] || { echo "[verifier-eval] missing rollouts: $ROLLOUTS" >&2; exit 3; }
cd "$BASE"
"$PY" train/eval_verifier.py --verifier "$VERIFIER" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --rollouts "$ROLLOUTS" --out "$OUT"
