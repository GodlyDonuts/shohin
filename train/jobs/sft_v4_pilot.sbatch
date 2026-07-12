#!/bin/bash
#SBATCH -J shohin-sft-v4
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v4_%j.out
#SBATCH --open-mode=append
# Source-balanced v4 SFT pilot. The canonical data mix is immutable and must
# already have passed build_reasoning_sft_v4's quality gates. This job is never
# allowed to write into sft_out or an earlier pilot directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl
INIT=${INIT:?set INIT to a preserved numbered flagship checkpoint}
OUT=${OUT:?set OUT to a new isolated SFT output directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.002}
LRADAM=${LRADAM:-0.0005}
# Sequence-level sampling keeps these high-level groups visible without
# duplicating JSONL rows: math 40%, procedural 35%, code 15%, teacher 10%.
WEIGHTS="math=0.40 procedural=0.35 code=0.15 teacher=0.10"

[ -s "$DATA" ] || { echo "[sft-v4] frozen mix missing: $DATA" >&2; exit 2; }
[ -f "$INIT" ] || { echo "[sft-v4] init missing: $INIT" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-v4] refusing to overwrite $OUT" >&2; exit 4; }

cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[sft-v4] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[sft-v4] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights $WEIGHTS --compile --out "$OUT"
echo "[sft-v4] done"
