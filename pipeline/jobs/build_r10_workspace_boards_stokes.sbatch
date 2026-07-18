#!/bin/bash
#SBATCH -J shohin-r10-boards
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r10_workspace_boards_%j.out
#SBATCH --open-mode=append
# CPU-only, score-blind build and independent admission on shared Lustre.
set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
DATA_BASE=${DATA_BASE:-$BASE}
CODE_BASE=${CODE_BASE:-$DATA_BASE}
PY=${PY:-$DATA_BASE/miniforge3/bin/python}
TRAINING_DATA=${TRAINING_DATA:-$DATA_BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
R5_BOARD=${R5_BOARD:-$DATA_BASE/artifacts/evals/referential_argument_graph_v5_fresh.jsonl}
TOKENIZER=${TOKENIZER:-$DATA_BASE/artifacts/shohin-tok-32k.json}
MANUAL=${MANUAL:-$DATA_BASE/artifacts/evals/categorical_microcode_manual_v1.jsonl}
EVALUATOR=${EVALUATOR:-$CODE_BASE/train/evaluate_version_space_workspace.py}
EXTRACTOR=${EXTRACTOR:-$CODE_BASE/train/extract_referential_version_scores.py}
CODE_REVISION=${CODE_REVISION:-$(git -C "$CODE_BASE" rev-parse HEAD)}
CANONICAL_R5_SHA256=d85f16ff374b0c650cf3603826cc5f3b377842818db62bada3b84e71308b9473
CANONICAL_CALIBRATION_SEED=2026071401
CANONICAL_CONFIRMATION_SEED=2026071402
CALIBRATION=${CALIBRATION:-$DATA_BASE/artifacts/evals/r10_workspace_calibration_v2.jsonl}
CONFIRMATION=${CONFIRMATION:-$DATA_BASE/artifacts/evals/r10_workspace_confirmation_v2.jsonl}
BUILD_MANIFEST=${BUILD_MANIFEST:-$DATA_BASE/artifacts/evals/r10_workspace_boards_v2.build.json}
AUDIT_MANIFEST=${AUDIT_MANIFEST:-$DATA_BASE/artifacts/evals/r10_workspace_boards_v2.admission.json}
GATE_MANIFEST=${GATE_MANIFEST:-$DATA_BASE/artifacts/evals/r10_workspace_boards_v2.frozen_gate.json}
CALIBRATION_STRUCTURAL=${CALIBRATION_STRUCTURAL:-$DATA_BASE/artifacts/evals/r10_workspace_calibration_v2.structural_admission.json}
CALIBRATION_LABELS=${CALIBRATION_LABELS:-$DATA_BASE/artifacts/evals/r10_workspace_calibration_v2.label_admission.json}
CONFIRMATION_STRUCTURAL=${CONFIRMATION_STRUCTURAL:-$DATA_BASE/artifacts/evals/r10_workspace_confirmation_v2.structural_admission.json}
CONFIRMATION_LABELS=${CONFIRMATION_LABELS:-$DATA_BASE/artifacts/evals/r10_workspace_confirmation_v2.label_admission.json}

if [[ "${SLURM_JOB_GPUS:-0}" != "0" || "${SLURM_GPUS_ON_NODE:-0}" != "0" ]]; then
  echo "[r10-boards] refusing GPU allocation for CPU-only board custody" >&2
  exit 2
fi
if [[ ! "$CODE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[r10-boards] code revision must be a full lowercase 40-hex git revision" >&2
  exit 2
fi
ACTUAL_HEAD=$(git -C "$CODE_BASE" rev-parse HEAD 2>/dev/null) || {
  echo "[r10-boards] CODE_BASE must be an exact git checkout: $CODE_BASE" >&2
  exit 2
}
if [[ "$ACTUAL_HEAD" != "$CODE_REVISION" ]]; then
  echo "[r10-boards] CODE_BASE HEAD differs from CODE_REVISION" >&2
  exit 2
fi
CODE_STATUS=$(git -C "$CODE_BASE" status --porcelain=v1 --untracked-files=all)
if [[ -n "$CODE_STATUS" ]]; then
  echo "[r10-boards] refusing a checkout with dirty or untracked files" >&2
  exit 2
fi
for path in \
  "$PY" "$TRAINING_DATA" "$R5_BOARD" "$TOKENIZER" "$MANUAL" \
  "$EVALUATOR" "$EXTRACTOR"; do
  [[ -s "$path" ]] || { echo "[r10-boards] missing input: $path" >&2; exit 2; }
done
ACTUAL_R5_SHA256=$(sha256sum "$R5_BOARD" | awk '{print $1}')
if [[ "$ACTUAL_R5_SHA256" != "$CANONICAL_R5_SHA256" ]]; then
  echo "[r10-boards] R5 novelty board differs from the frozen canonical SHA256" >&2
  exit 2
fi
for path in \
  "$CALIBRATION" "$CONFIRMATION" "$BUILD_MANIFEST" "$AUDIT_MANIFEST" "$GATE_MANIFEST" \
  "$CALIBRATION_STRUCTURAL" "$CALIBRATION_LABELS" \
  "$CONFIRMATION_STRUCTURAL" "$CONFIRMATION_LABELS"; do
  [[ ! -e "$path" ]] || { echo "[r10-boards] refusing existing output: $path" >&2; exit 3; }
done

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$CODE_BASE"
"$PY" pipeline/test_r10_workspace_boards.py
"$PY" pipeline/generate_r10_workspace_boards.py \
  --training-data "$TRAINING_DATA" \
  --r5-board "$R5_BOARD" \
  --tokenizer "$TOKENIZER" \
  --calibration-out "$CALIBRATION" \
  --confirmation-out "$CONFIRMATION" \
  --manifest-out "$BUILD_MANIFEST" \
  --calibration-count 800 \
  --confirmation-count 1840 \
  --calibration-seed "$CANONICAL_CALIBRATION_SEED" \
  --confirmation-seed "$CANONICAL_CONFIRMATION_SEED" \
  --max-tokens 2048
"$PY" pipeline/audit_role_equivariant_microcode_v3.py \
  --train "$TRAINING_DATA" --eval "$CALIBRATION" --tokenizer "$TOKENIZER" \
  --out "$CALIBRATION_STRUCTURAL" --programs 48000
"$PY" pipeline/audit_referential_slot_labels.py \
  --train "$TRAINING_DATA" --eval "$CALIBRATION" --manual "$MANUAL" \
  --tokenizer "$TOKENIZER" --out "$CALIBRATION_LABELS" \
  --train-rows 288000 --eval-rows 800 --manual-rows 8
"$PY" pipeline/audit_role_equivariant_microcode_v3.py \
  --train "$TRAINING_DATA" --eval "$CONFIRMATION" --tokenizer "$TOKENIZER" \
  --out "$CONFIRMATION_STRUCTURAL" --programs 48000
"$PY" pipeline/audit_referential_slot_labels.py \
  --train "$TRAINING_DATA" --eval "$CONFIRMATION" --manual "$MANUAL" \
  --tokenizer "$TOKENIZER" --out "$CONFIRMATION_LABELS" \
  --train-rows 288000 --eval-rows 1840 --manual-rows 8
"$PY" pipeline/audit_r10_workspace_boards.py \
  --training-data "$TRAINING_DATA" \
  --r5-board "$R5_BOARD" \
  --tokenizer "$TOKENIZER" \
  --calibration "$CALIBRATION" \
  --confirmation "$CONFIRMATION" \
  --build-manifest "$BUILD_MANIFEST" \
  --calibration-structural-admission "$CALIBRATION_STRUCTURAL" \
  --calibration-label-admission "$CALIBRATION_LABELS" \
  --confirmation-structural-admission "$CONFIRMATION_STRUCTURAL" \
  --confirmation-label-admission "$CONFIRMATION_LABELS" \
  --out "$AUDIT_MANIFEST" \
  --gate-manifest-out "$GATE_MANIFEST" \
  --evaluator "$EVALUATOR" \
  --extractor "$EXTRACTOR" \
  --code-revision "$CODE_REVISION" \
  --max-tokens 2048 \
  --minimum-calibration 800 \
  --minimum-confirmation 1840

FINAL_HEAD=$(git -C "$CODE_BASE" rev-parse --verify 'HEAD^{commit}')
FINAL_STATUS=$(git -C "$CODE_BASE" status --porcelain=v1 --untracked-files=all)
if [[ "$FINAL_HEAD" != "$ACTUAL_HEAD" || -n "$FINAL_STATUS" ]]; then
  echo "[r10-boards] code checkout changed during build or admission" >&2
  exit 2
fi

echo "[r10-boards] v2 build, independent admission, and frozen gates complete; no score artifact was read"
