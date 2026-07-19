#!/bin/bash
#SBATCH -J shohin-s5-generator
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s5_generator_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to the frozen fresh-development board directory}
TAG=${TAG:-2026071911}
FIT=$BASE/train/s5_learned_generator_${TAG}
EVAL=$BASE/artifacts/r12/s5_learned_generator_${TAG}.eval.json
ASSESS=$BASE/artifacts/r12/s5_learned_generator_${TAG}.assessment.json
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" \
  "$BASE/train/s4_event_tape_treatment_2026071904/parser.pt" \
  "$BASE/artifacts/r12/s4_hard_island_soft_interface_v5.promoted.json" \
  "$DATA/development.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
for path in "$FIT" "$EVAL" "$ASSESS"; do
  test ! -e "$path" || { echo "refusing existing $path" >&2; exit 2; }
done
cd "$BASE"
"$PYTHON" train/train_s5_learned_generator.py \
  --out-dir "$FIT" --width 64 --updates 500 --lr 0.003 \
  --seed 2026071911 --base-parameters 133689935 --device cpu
"$PYTHON" train/eval_s5_learned_generator_executor.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --parser train/s4_event_tape_treatment_2026071904/parser.pt \
  --promoted artifacts/r12/s4_hard_island_soft_interface_v5.promoted.json \
  --treatment-generator "$FIT/treatment.pt" \
  --shuffled-generator "$FIT/shuffled.pt" \
  --fit-report "$FIT/report.json" \
  --data "$DATA/development.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$EVAL" --batch-size 64
"$PYTHON" train/assess_s5_learned_generator_executor.py \
  --evaluation "$EVAL" --out "$ASSESS"
