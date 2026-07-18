#!/bin/bash
#SBATCH -J shohin-r10-static
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r10_static_%j.out
#SBATCH --open-mode=append
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
EVALUATOR=${EVALUATOR:-$BASE/train/evaluate_version_space_workspace.py}
EVALUATOR_SHA256=${EVALUATOR_SHA256:?set the preregistered evaluator SHA-256}
EXTRACTOR=${EXTRACTOR:-$BASE/train/extract_referential_version_scores.py}
EXTRACTOR_SHA256=${EXTRACTOR_SHA256:?set the preregistered extractor SHA-256}
GATE_MANIFEST=${GATE_MANIFEST:?set the frozen v2 score gate manifest}
GATE_MANIFEST_SHA256=${GATE_MANIFEST_SHA256:?set its preregistered SHA-256}
GATE_ADMISSION=${GATE_ADMISSION:?set the independent v2 admission artifact}
GATE_ADMISSION_SHA256=${GATE_ADMISSION_SHA256:?set its preregistered SHA-256}
CALIBRATION_SCORES=${CALIBRATION_SCORES:?set the frozen calibration score report}
CALIBRATION_SCORES_SHA256=${CALIBRATION_SCORES_SHA256:?set its preregistered SHA-256}
CALIBRATION_DATA_SHA256=${CALIBRATION_DATA_SHA256:?set the calibration board SHA-256}
CALIBRATION_STRUCTURAL_ADMISSION_SHA256=${CALIBRATION_STRUCTURAL_ADMISSION_SHA256:?set the calibration structural admission SHA-256}
CALIBRATION_LABEL_ADMISSION_SHA256=${CALIBRATION_LABEL_ADMISSION_SHA256:?set the calibration label admission SHA-256}
TEST_SCORES=${TEST_SCORES:?set the frozen confirmation score report}
TEST_SCORES_SHA256=${TEST_SCORES_SHA256:?set its preregistered SHA-256}
TEST_DATA_SHA256=${TEST_DATA_SHA256:?set the confirmation board SHA-256}
TEST_STRUCTURAL_ADMISSION_SHA256=${TEST_STRUCTURAL_ADMISSION_SHA256:?set the confirmation structural admission SHA-256}
TEST_LABEL_ADMISSION_SHA256=${TEST_LABEL_ADMISSION_SHA256:?set the confirmation label admission SHA-256}
CODE_REVISION=${CODE_REVISION:?set the exact committed extractor revision}
OUT=${OUT:?set a fresh R10 static confirmation output}

if [[ "${SLURM_JOB_GPUS:-0}" != "0" || "${SLURM_GPUS_ON_NODE:-0}" != "0" ]]; then
  echo "[r10-static] refusing GPU allocation for CPU-only evaluation" >&2
  exit 2
fi

for path in \
  "$PY" "$EVALUATOR" "$EXTRACTOR" "$GATE_MANIFEST" "$GATE_ADMISSION" \
  "$CALIBRATION_SCORES" "$TEST_SCORES"; do
  [ -s "$path" ] || { echo "[r10-static] missing input: $path" >&2; exit 2; }
done
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "[r10-static] refusing existing output: $OUT" >&2
  exit 2
fi

ARGS=(
  --calibration-scores "$CALIBRATION_SCORES"
  --calibration-scores-sha256 "$CALIBRATION_SCORES_SHA256"
  --calibration-data-sha256 "$CALIBRATION_DATA_SHA256"
  --calibration-structural-admission-sha256 "$CALIBRATION_STRUCTURAL_ADMISSION_SHA256"
  --calibration-label-admission-sha256 "$CALIBRATION_LABEL_ADMISSION_SHA256"
  --test-scores "$TEST_SCORES"
  --test-scores-sha256 "$TEST_SCORES_SHA256"
  --test-data-sha256 "$TEST_DATA_SHA256"
  --test-structural-admission-sha256 "$TEST_STRUCTURAL_ADMISSION_SHA256"
  --test-label-admission-sha256 "$TEST_LABEL_ADMISSION_SHA256"
  --gate-manifest "$GATE_MANIFEST"
  --gate-manifest-sha256 "$GATE_MANIFEST_SHA256"
  --gate-admission "$GATE_ADMISSION"
  --gate-admission-sha256 "$GATE_ADMISSION_SHA256"
  --extractor "$EXTRACTOR"
  --extractor-sha256 "$EXTRACTOR_SHA256"
  --evaluator-sha256 "$EVALUATOR_SHA256"
  --code-revision "$CODE_REVISION"
  --out "$OUT"
)

if [ -n "${CALIBRATION_OLD_FIXED_REPORT:-}" ] || [ -n "${CALIBRATION_OLD_FIXED_REPORT_SHA256:-}" ]; then
  [ -s "${CALIBRATION_OLD_FIXED_REPORT:-}" ] || {
    echo "[r10-static] calibration old fixed report is missing" >&2
    exit 2
  }
  [ -n "${CALIBRATION_OLD_FIXED_REPORT_SHA256:-}" ] || {
    echo "[r10-static] calibration old fixed report hash is missing" >&2
    exit 2
  }
  ARGS+=(
    --calibration-old-fixed-report "$CALIBRATION_OLD_FIXED_REPORT"
    --calibration-old-fixed-report-sha256 "$CALIBRATION_OLD_FIXED_REPORT_SHA256"
  )
fi
if [ -n "${TEST_OLD_FIXED_REPORT:-}" ] || [ -n "${TEST_OLD_FIXED_REPORT_SHA256:-}" ]; then
  [ -s "${TEST_OLD_FIXED_REPORT:-}" ] || {
    echo "[r10-static] test old fixed report is missing" >&2
    exit 2
  }
  [ -n "${TEST_OLD_FIXED_REPORT_SHA256:-}" ] || {
    echo "[r10-static] test old fixed report hash is missing" >&2
    exit 2
  }
  ARGS+=(
    --test-old-fixed-report "$TEST_OLD_FIXED_REPORT"
    --test-old-fixed-report-sha256 "$TEST_OLD_FIXED_REPORT_SHA256"
  )
fi

cd "$BASE/train"
"$PY" "$EVALUATOR" "${ARGS[@]}"
