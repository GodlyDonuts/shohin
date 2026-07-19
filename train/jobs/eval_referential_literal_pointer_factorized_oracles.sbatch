#!/bin/bash
#SBATCH -J shohin-rlpc-oracle
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_literal_pointer_factorized_oracle_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
export OMP_NUM_THREADS=4
cd "$BASE"

ARM=${ARM:?set ARM to islands or ordinary}
SEED=${SEED:-2026071810}
case "$ARM" in
  islands|ordinary) ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
DATA_DIR=$BASE/artifacts/r12/referential_literal_pointer_factorized_v1
ARM_DIR=$BASE/train/referential_literal_pointer_factorized_${ARM}_${SEED}
ADAPTER=$ARM_DIR/compiler.pt
DATA=$DATA_DIR/development_lexical_ood.jsonl
REPORT=$DATA_DIR/development_report.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json

for path in "$CKPT" "$ADAPTER" "$DATA" "$REPORT" "$TOKENIZER"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done

echo "[rlpc-oracle] arm=$ARM node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
for oracle in lexical structure full; do
  out=$ARM_DIR/development_lexical_ood_${oracle}_oracle.json
  [ ! -e "$out" ] || { echo "refusing existing output: $out" >&2; exit 2; }
  "$PYTHON" train/eval_referential_literal_pointer_compiler.py \
    --base "$CKPT" --adapter "$ADAPTER" --data "$DATA" --report "$REPORT" \
    --tokenizer "$TOKENIZER" --split development_lexical_ood \
    --oracle "$oracle" --out "$out" --batch-size 64
done

echo "[rlpc-oracle] explicit host ceilings complete; no confirmation path was referenced"
