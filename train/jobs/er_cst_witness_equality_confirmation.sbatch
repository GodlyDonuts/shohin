#!/bin/bash
#SBATCH --job-name=er-cst-web-confirm
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/er_cst_witness_equality_confirmation_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set exact confirmation evaluator source}
DATA=${DATA:?set exact ER-CST witness board}
RUN=${RUN:?set exact passing development output}
JOINT=${JOINT:?set exact joint checkpoint}
PHYSICAL=${PHYSICAL:?set exact physical checkpoint}
V1=${V1:?set exact complete-local v1 checkpoint}
V1_2=${V1_2:?set exact occurrence-head checkpoint}
CONFIRMED=${CONFIRMED:?set exact confirmed v1.3 compiler}
PARENT_CONFIRMATION_ASSESSMENT=${PARENT_CONFIRMATION_ASSESSMENT:?set exact parent confirmation assessment}
OUTDIR=${OUTDIR:?set new isolated confirmation output}
EVALUATOR_SOURCE_COMMIT=${EVALUATOR_SOURCE_COMMIT:?set frozen evaluator commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$EVALUATOR_SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
DEVELOPMENT_LEDGER=$(find "$DATA/access" -maxdepth 1 -type f -name 'er_cst_witness_development_*.json' -print -quit)
test -n "$DEVELOPMENT_LEDGER"
for path in \
  "$DATA/report.json" \
  "$DATA/confirmation.jsonl" \
  "$RUN/compiler.pt" \
  "$RUN/development_evidence.pt" \
  "$RUN/development_report.json" \
  "$RUN/development_assessment.json" \
  "$DEVELOPMENT_LEDGER" \
  "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$CONFIRMED" \
  "$PARENT_CONFIRMATION_ASSESSMENT"; do
  test -f "$path" || { echo "missing ER-CST witness confirmation input: $path" >&2; exit 2; }
done
test "$(stat -c '%a' "$DATA/confirmation.jsonl")" = 600
test "$(find "$DATA/access" -maxdepth 1 -type f | wc -l)" -eq 1
test ! -e "$OUTDIR" || { echo "refusing existing ER-CST witness confirmation output" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("ER-CST witness confirmation requires bf16 CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/confirm_er_cst_witness_equality.py \
  --repo-root "$CODE_ROOT" \
  --data-dir "$DATA" \
  --checkpoint "$RUN/compiler.pt" \
  --development-evidence "$RUN/development_evidence.pt" \
  --development-report "$RUN/development_report.json" \
  --development-assessment "$RUN/development_assessment.json" \
  --development-ledger "$DEVELOPMENT_LEDGER" \
  --joint-checkpoint "$JOINT" \
  --physical-checkpoint "$PHYSICAL" \
  --v1-checkpoint "$V1" \
  --v1-2-checkpoint "$V1_2" \
  --confirmed-checkpoint "$CONFIRMED" \
  --confirmation-assessment "$PARENT_CONFIRMATION_ASSESSMENT" \
  --out-dir "$OUTDIR" \
  --evaluator-source-commit "$EVALUATOR_SOURCE_COMMIT"

CONFIRMATION_LEDGER=$(find "$DATA/access" -maxdepth 1 -type f -name 'er_cst_witness_confirmation_*.json' -print -quit)
test -n "$CONFIRMATION_LEDGER"
"$PYTHON" train/assess_er_cst_witness_equality_confirmation.py \
  --report "$OUTDIR/confirmation_report.json" \
  --checkpoint "$RUN/compiler.pt" \
  --development-evidence "$RUN/development_evidence.pt" \
  --development-report "$RUN/development_report.json" \
  --development-assessment "$RUN/development_assessment.json" \
  --development-ledger "$DEVELOPMENT_LEDGER" \
  --authorization "$OUTDIR/authorization.json" \
  --evidence "$OUTDIR/confirmation_evidence.pt" \
  --ledger "$CONFIRMATION_LEDGER" \
  --output "$OUTDIR/confirmation_assessment.json"

sha256sum "$OUTDIR"/* "$CONFIRMATION_LEDGER"
echo "[er-cst-web-confirmation] complete"
