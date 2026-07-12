#!/bin/bash
#SBATCH -J shohin-sft-v5
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v5_%j.out
#SBATCH --open-mode=append
# Isolated source-balanced SFT ablation for the verified primitives curriculum.
#
# This is deliberately not a mutation of frozen v4 or the flagship. It starts
# from the same pinned raw checkpoint as v4 so public/RG/direct comparisons can
# attribute changes to data and SFT settings rather than a moving base model.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=(
  "$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl"
  "$BASE/artifacts/sft/primitives_v1_train.jsonl"
)
INIT=${INIT:?set INIT to a preserved raw flagship checkpoint}
OUT=${OUT:?set OUT to a fresh isolated SFT directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0015}
LRADAM=${LRADAM:-0.00035}
# Retain broad contest-math pressure, add the missing compact operations, and
# cap small code/teacher groups below memorization-heavy replay rates.
WEIGHTS="math=0.45 procedural=0.15 code=0.05 teacher=0.05 primitives=0.30"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-v5] data missing: $data" >&2; exit 2; }
done
[ -f "$INIT" ] || { echo "[sft-v5] init missing: $INIT" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-v5] refusing to overwrite $OUT" >&2; exit 4; }

cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[sft-v5] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[sft-v5] init=$INIT data=${DATA[*]} out=$OUT epochs=$EPOCHS weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights $WEIGHTS --compile --out "$OUT"
echo "[sft-v5] done"
