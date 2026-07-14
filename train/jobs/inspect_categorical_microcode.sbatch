#!/bin/bash
#SBATCH -J shohin-microcode-manual
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microcode_manual_%j.out
#SBATCH --open-mode=append
# Read-only hand-authored interaction transcript.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set base checkpoint}
ADAPTER=${ADAPTER:?set adapter checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/categorical_microcode_manual_v1.jsonl}
OUT=${OUT:?set fresh transcript output}
for path in "$BASE_CKPT" "$ADAPTER" "$DATA"; do
  [ -s "$path" ] || { echo "[microcode-manual] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[microcode-manual] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" inspect_categorical_microcode.py \
  --base "$BASE_CKPT" --adapter "$ADAPTER" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
