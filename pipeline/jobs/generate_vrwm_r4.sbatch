#!/bin/bash
#SBATCH -J shohin-vrwm-r4
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/vrwm_r4_%j.out
#SBATCH --open-mode=append
# Generate one auditable VRWM r4 branch on CPU. The two branches must use the
# same seed and episode parameters; only RESPONSE_MODE differs.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
MODE=${MODE:?set MODE to state or scratch}
case "$MODE" in state|scratch) ;; *) echo "invalid MODE=$MODE" >&2; exit 2 ;; esac

EPISODES=${EPISODES:-60000}
MAX_STEPS=${MAX_STEPS:-8}
VALUE_LIMIT=${VALUE_LIMIT:-999}
CONST_LIMIT=${CONST_LIMIT:-127}
EVAL_PER_LENGTH=${EVAL_PER_LENGTH:-80}
SEED=${SEED:-20260715}
REPAIR_EXAMPLES=${REPAIR_EXAMPLES:-0}
PREFIX=$BASE/artifacts/sft/vrwm_v2_${MODE}_r4
TRAIN=${TRAIN:-${PREFIX}.jsonl}
EVAL=${EVAL:-$BASE/artifacts/vrwm/vrwm_v2_${MODE}_eval_r4.jsonl}
REPORT=${REPORT:-$BASE/artifacts/vrwm/vrwm_v2_${MODE}_generation_r4.json}
QUALITY=${QUALITY:-${PREFIX}.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-${PREFIX}.training_text_overlap.r1.json}
PACKING=${PACKING:-${PREFIX}.packing.r1.json}

for path in "$TRAIN" "$EVAL" "$REPORT" "$QUALITY" "$FULL_TEXT" "$PACKING"; do
  [ ! -e "$path" ] || { echo "[vrwm-r4] refusing existing path: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/generate_vrwm_curriculum.py \
  --train-out "$TRAIN" --eval-out "$EVAL" --report "$REPORT" \
  --train-episodes "$EPISODES" --train-max-steps "$MAX_STEPS" \
  --train-value-limit "$VALUE_LIMIT" --train-const-limit "$CONST_LIMIT" \
  --train-styles default paraphrase --eval-style semantic --response-mode "$MODE" \
  --repair-examples "$REPAIR_EXAMPLES" --eval-per-length "$EVAL_PER_LENGTH" --seed "$SEED"
"$PY" pipeline/audit_sft_quality.py --data "$TRAIN" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$TRAIN" --evals "$BASE/artifacts/evals" \
  --out "$FULL_TEXT" --require-zero
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$TRAIN" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$PACKING" --weights vrwm=1.0
echo "[vrwm-r4] complete mode=$MODE train=$TRAIN eval=$EVAL"
