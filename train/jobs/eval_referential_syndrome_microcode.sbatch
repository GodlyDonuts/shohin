#!/bin/bash
#SBATCH -J shohin-r9c-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r9c_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
POINTER=${POINTER:-$BASE/train/microcode_referential_pointer_200k_r4/microcode_adapter_ep1.pt}
ADAPTER=${ADAPTER:?set R9c adapter}
DATA=${DATA:?set used-board evaluation JSONL}
OUT=${OUT:?set fresh evaluation output}

for path in "$BASE_CKPT" "$POINTER" "$ADAPTER" "$DATA"; do
  [ -s "$path" ] || { echo "[r9c-eval] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[r9c-eval] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" eval_referential_syndrome_microcode.py \
  --base "$BASE_CKPT" --pointer-adapter "$POINTER" --adapter "$ADAPTER" \
  --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --batch-size "${BATCH_SIZE:-16}" --syndrome-threshold "${SYNDROME_THRESHOLD:-0.05}"
