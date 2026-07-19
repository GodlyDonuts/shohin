#!/bin/bash
#SBATCH -J shohin-rlpc-qual
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_literal_pointer_qualification_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
SEED=${SEED:?set qualification seed}
DATA_DIR=${DATA_DIR:?set qualification data directory}
ADAPTER=${ADAPTER:-$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt}
CKPT=${CKPT:-$BASE/train/flagship_out/ckpt_0300000.pt}
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
DATA=$DATA_DIR/development_compositional.jsonl
REPORT=$DATA_DIR/report.json
OUT=${OUT:-$BASE/train/referential_literal_pointer_qualification_${SEED}/development_compositional.json}
cd "$BASE"

for path in "$CKPT" "$ADAPTER" "$TOKENIZER" "$DATA" "$REPORT"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
echo "[rlpc-qualification] seed=$SEED node=${SLURMD_NODENAME:-?} adapter=$ADAPTER"
nvidia-smi -L

"$PYTHON" train/eval_referential_literal_pointer_compiler.py \
  --base "$CKPT" --adapter "$ADAPTER" --data "$DATA" --report "$REPORT" \
  --tokenizer "$TOKENIZER" --split development_compositional \
  --out "$OUT" --batch-size 64

echo "[rlpc-qualification] one-shot evaluation complete; sealed confirmation was never referenced"
