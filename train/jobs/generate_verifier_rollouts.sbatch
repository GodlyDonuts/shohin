#!/bin/bash
#SBATCH -J shohin-verifier-rollouts
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_rollouts_%j.out
#SBATCH --open-mode=append
# Generate train-only or held-out student candidates. The caller must set every
# path explicitly; this job never writes to a flagship or SFT checkpoint folder.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT}
DATA=${DATA:?set DATA}
OUT=${OUT:?set OUT}
N=${N:-2000}
K=${K:-8}
TAG=${TAG:-verifier-rollouts}

[ -f "$CKPT" ] || { echo "[verifier-rollouts] missing checkpoint: $CKPT" >&2; exit 2; }
[ -f "$DATA" ] || { echo "[verifier-rollouts] missing data: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[verifier-rollouts] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
"$PY" generate_verifier_rollouts.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --data "$DATA" --out "$OUT" --n "$N" --k "$K"
echo "[verifier-rollouts] tag=$TAG wrote $OUT"
