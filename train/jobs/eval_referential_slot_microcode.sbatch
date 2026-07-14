#!/bin/bash
#SBATCH -J shohin-referential-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_slot_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
ADAPTER=${ADAPTER:?set adapter}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/role_equivariant_microcode_v3.admission.json}
LABEL_ADMISSION=${LABEL_ADMISSION:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}
OUT=${OUT:?set fresh evaluation output}

for path in "$BASE_CKPT" "$ADAPTER" "$DATA" "$ADMISSION" "$LABEL_ADMISSION"; do
  [ -s "$path" ] || { echo "[referential-eval] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[referential-eval] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" eval_referential_slot_microcode.py --base "$BASE_CKPT" --adapter "$ADAPTER" \
  --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --admission "$ADMISSION" --label-admission "$LABEL_ADMISSION" --out "$OUT"
