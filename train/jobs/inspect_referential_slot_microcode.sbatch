#!/bin/bash
#SBATCH -J shohin-referential-manual
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_slot_manual_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
ADAPTER=${ADAPTER:?set adapter}
DATA=${DATA:-$BASE/artifacts/evals/categorical_microcode_manual_v1.jsonl}
LABEL_ADMISSION=${LABEL_ADMISSION:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}
OUT=${OUT:?set fresh manual output}

for path in "$BASE_CKPT" "$ADAPTER" "$DATA" "$LABEL_ADMISSION"; do
  [ -s "$path" ] || { echo "[referential-manual] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[referential-manual] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" inspect_referential_slot_microcode.py --base "$BASE_CKPT" --adapter "$ADAPTER" \
  --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --label-admission "$LABEL_ADMISSION" --out "$OUT"
