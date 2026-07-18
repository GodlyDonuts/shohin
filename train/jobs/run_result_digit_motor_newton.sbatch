#!/bin/bash
#SBATCH -J result-digit-motor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc25,evc26,evc27,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/result_digit_motor_%j.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/result_digit_motor_%j.err

# Exploratory Newton job for the R12 result-digit motor (sibling to Codex's
# carry motor; does not touch or read its output paths). This lane does not
# require the git source-commit seal, so --allow-non-canonical is always
# passed: canonical custody is deferred until the carry-motor lane reports.

set -euo pipefail
echo "[job] start $(date) host=$(hostname)"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-$BASE/miniforge3/bin/python}
REPO_ROOT=${REPO_ROOT:-/lustre/fs1/home/sa305415/shohin}
MODE=${MODE:-train}
CKPT=${CKPT:-$BASE/train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
CYCLE=${CYCLE:-$BASE/artifacts/evals/drs_causal_cycle_post_drs_r3.json}
OUT_DIR=${OUT_DIR:-$BASE/artifacts/r12/result_digit_motor_r2}
OUT=${OUT:-$OUT_DIR/motor.pt}
MOTOR=${MOTOR:-}
EVAL_OUT=${EVAL_OUT:-$OUT_DIR/eval.json}
ALLOW_EVALUATOR_PATCH=${ALLOW_EVALUATOR_PATCH:-0}

for path in "$REPO_ROOT" "$CKPT" "$TOKENIZER" "$EPISODES" "$CYCLE" "$OUT"; do
  if [[ "$path" != /* ]]; then
    echo "canonical paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$REPO_ROOT/.git" ]]; then
  echo "REPO_ROOT must be a Git checkout" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$BASE/logs"

if [[ "$MODE" == train ]]; then
  if [[ -e "$OUT" || -e "$OUT_DIR/report.json" ]]; then
    echo "refusing existing output: $OUT" >&2
    exit 4
  fi
elif [[ "$MODE" == eval ]]; then
  if [[ -z "$MOTOR" || ! -f "$MOTOR" ]]; then
    echo "eval mode requires MOTOR to point at an existing motor.pt" >&2
    exit 2
  fi
  if [[ -e "$EVAL_OUT" ]]; then
    echo "refusing existing output: $EVAL_OUT" >&2
    exit 4
  fi
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi

for attempt in 1 2 3 4 5 6 7 8; do
  echo "[job] cuda attempt $attempt on $(hostname)"
  if timeout 180 "$PY" -u - <<'CUDAPY'
import torch
assert torch.cuda.is_available()
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0), flush=True)
CUDAPY
  then echo cuda_ok; break; fi
  sleep 10
  [[ "$attempt" -eq 8 ]] && exit 12
done

cd "$REPO_ROOT/train"
export PYTHONPATH="$REPO_ROOT/train"

if [[ "$MODE" == train ]]; then
  "$PY" -u causal_result_digit_motor.py train \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" --cycle "$CYCLE" \
    --allow-non-canonical --out "$OUT"
else
  EVAL_EXTRA=()
  if [[ "$ALLOW_EVALUATOR_PATCH" == "1" ]]; then
    EVAL_EXTRA+=(--allow-evaluator-patch)
  elif [[ "$ALLOW_EVALUATOR_PATCH" != "0" ]]; then
    echo "ALLOW_EVALUATOR_PATCH must be 0 or 1" >&2
    exit 2
  fi
  "$PY" -u causal_result_digit_motor.py eval \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" --cycle "$CYCLE" \
    --allow-non-canonical --motor "$MOTOR" --out "$EVAL_OUT" "${EVAL_EXTRA[@]}"
fi
echo "[job] done $(date)"
