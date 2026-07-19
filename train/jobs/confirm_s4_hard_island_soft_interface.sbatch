#!/bin/bash
#SBATCH -J shohin-s4-v5-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s4_v5_confirm_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to the frozen confirmation-board directory}
TAG=${TAG:-2026071909}
BASELINE=$BASE/artifacts/r12/s4_hard_island_confirmation_v1_${TAG}.eval.json
TREATMENT=$BASE/artifacts/r12/s4_hard_island_confirmation_${TAG}.eval.json
ASSESSMENT=$BASE/artifacts/r12/s4_hard_island_confirmation_${TAG}.assessment.json
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" \
  "$BASE/train/s4_event_tape_treatment_2026071904/parser.pt" \
  "$BASE/train/s3_equivariant_executor_2026071903/executor.pt" \
  "$DATA/confirmation.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
for path in "$BASELINE" "$TREATMENT" "$ASSESSMENT"; do
  test ! -e "$path" || { echo "refusing existing $path" >&2; exit 2; }
done
cd "$BASE"
"$PYTHON" train/eval_s4_event_tape.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --parser train/s4_event_tape_treatment_2026071904/parser.pt \
  --s3-executor train/s3_equivariant_executor_2026071903/executor.pt \
  --data "$DATA/confirmation.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$BASELINE" --batch-size 64
"$PYTHON" train/eval_s4_hard_island_confirmation.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --parser train/s4_event_tape_treatment_2026071904/parser.pt \
  --s3-executor train/s3_equivariant_executor_2026071903/executor.pt \
  --data "$DATA/confirmation.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$TREATMENT" --batch-size 64
"$PYTHON" train/assess_s4_hard_island_confirmation.py \
  --treatment "$TREATMENT" --baseline "$BASELINE" --out "$ASSESSMENT"
