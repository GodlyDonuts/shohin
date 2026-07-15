#!/bin/bash
#SBATCH -J shohin-r9c-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r9c_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
POINTER=${POINTER:-$BASE/train/microcode_referential_pointer_200k_r4/microcode_adapter_ep1.pt}
DATA=${DATA:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/role_equivariant_microcode_v3.admission.json}
LABEL_ADMISSION=${LABEL_ADMISSION:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}
OUT=${OUT:?set fresh R9c output directory}
ARM=${ARM:?set treatment, static, no_syndrome, or shuffled_goal}

for path in "$BASE_CKPT" "$POINTER" "$DATA" "$ADMISSION" "$LABEL_ADMISSION"; do
  [ -s "$path" ] || { echo "[r9c] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[r9c] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/train/r9c_*) ;; *) echo "[r9c] invalid OUT path" >&2; exit 4;; esac
case "$ARM" in treatment|static|no_syndrome|shuffled_goal) ;; *) echo "[r9c] invalid ARM" >&2; exit 5;; esac

cd "$BASE/train"
nvidia-smi -L

"$PY" train_referential_syndrome_microcode.py \
  --base "$BASE_CKPT" --pointer-adapter "$POINTER" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --admission "$ADMISSION" \
  --label-admission "$LABEL_ADMISSION" --out "$OUT" --arm "$ARM" \
  --memory-dim "${MEMORY_DIM:-96}" --rounds "${ROUNDS:-3}" \
  --batch-groups "${BATCH_GROUPS:-4}" --max-groups "${MAX_GROUPS:-4096}" \
  --lr "${LR:-0.001}" --seed "${SEED:-20260714}"
