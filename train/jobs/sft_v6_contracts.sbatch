#!/bin/bash
#SBATCH -J shohin-sft-v6
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v6_%j.out
#SBATCH --open-mode=append
# Isolated SFT candidate for direct/review/scaffold/compact-state contracts.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=(
  "$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl"
  "$BASE/artifacts/sft/primitives_v2_contract_train.jsonl"
)
INIT=${INIT:?set INIT to a preserved raw flagship checkpoint}
OUT=${OUT:?set OUT to a fresh isolated directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0015}
LRADAM=${LRADAM:-0.00035}
WEIGHTS="math=0.45 procedural=0.15 code=0.05 teacher=0.05 contracts=0.30"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-v6] data missing: $data" >&2; exit 2; }
done
[ -f "$INIT" ] || { echo "[sft-v6] init missing: $INIT" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-v6] refusing to overwrite $OUT" >&2; exit 4; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
echo "[sft-v6] init=$INIT data=${DATA[*]} out=$OUT epochs=$EPOCHS weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights $WEIGHTS --compile --out "$OUT"
echo "[sft-v6] done"
