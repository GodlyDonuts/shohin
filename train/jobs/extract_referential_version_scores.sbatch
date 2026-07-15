#!/bin/bash
#SBATCH -J shohin-r10-scores
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc34,evc36,evc37,evc38,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r10_scores_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
POINTER=${POINTER:-$BASE/train/microcode_referential_pointer_200k_r4/microcode_adapter_ep1.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
ADAPTER=${ADAPTER:?set the frozen R9c no_syndrome adapter}
DATA=${DATA:?set the exact evaluation JSONL}
ADMISSION=${ADMISSION:?set the evaluation structural admission manifest}
LABEL_ADMISSION=${LABEL_ADMISSION:?set the evaluation referential-label admission manifest}
OUT=${OUT:?set a fresh R10 score output}
CODE_REVISION=${CODE_REVISION:?set the exact committed extractor git revision}
EXTRACTOR_SHA256=${EXTRACTOR_SHA256:?set the frozen extractor file SHA-256}

for path in "$BASE_CKPT" "$POINTER" "$TOKENIZER" "$ADAPTER" "$DATA" "$ADMISSION" "$LABEL_ADMISSION"; do
  [ -s "$path" ] || { echo "[r10-scores] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[r10-scores] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
nvidia-smi -L
"$PY" extract_referential_version_scores.py \
  --base "$BASE_CKPT" --pointer-adapter "$POINTER" --adapter "$ADAPTER" \
  --data "$DATA" --tokenizer "$TOKENIZER" --admission "$ADMISSION" \
  --label-admission "$LABEL_ADMISSION" --out "$OUT" \
  --batch-size "${BATCH_SIZE:-16}" --seed "${SEED:-20260714}" \
  --code-revision "$CODE_REVISION" --extractor-sha256 "$EXTRACTOR_SHA256"
