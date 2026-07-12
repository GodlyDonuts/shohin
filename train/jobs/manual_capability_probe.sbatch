#!/bin/bash
#SBATCH -J shohin-manual-probe
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0:35:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/manual_capability_probe_%j.out
# Run only as an isolated diagnostic.  It never writes a training artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW_CKPT=${RAW_CKPT:?set RAW_CKPT to a preserved raw checkpoint}
SFT_CKPT=${SFT_CKPT:?set SFT_CKPT to an isolated SFT checkpoint}
OUT=${OUT:?set OUT to an isolated JSON output}

for path in "$RAW_CKPT" "$SFT_CKPT"; do
  [ -f "$path" ] || { echo "[manual-probe] checkpoint missing: $path" >&2; exit 2; }
done
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
echo "[manual-probe] raw=$RAW_CKPT sft=$SFT_CKPT out=$OUT"
"$PY" manual_capability_probe.py \
  --ckpt "$RAW_CKPT" "$SFT_CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT"
