#!/bin/bash
#SBATCH --job-name=sd-cst-phys-confirm
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_complete_physical_confirmation_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set exact confirmation evaluator source}
DATA=${DATA:?set exact v1.3 board}
RUN=${RUN:?set exact passing development output}
JOINT=${JOINT:?set exact joint checkpoint}
PHYSICAL=${PHYSICAL:?set exact physical checkpoint}
V1=${V1:?set exact complete-local v1 checkpoint}
V1_2=${V1_2:?set exact occurrence-head checkpoint}
EXECUTION_CORE=${EXECUTION_CORE:?set exact categorical core}
OUTDIR=${OUTDIR:?set new isolated confirmation output}
EVALUATOR_SOURCE_COMMIT=${EVALUATOR_SOURCE_COMMIT:?set frozen evaluator commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$EVALUATOR_SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
DEVELOPMENT_LEDGER=$(find "$DATA/access" -maxdepth 1 -type f -name 'complete_physical_fresh_development_*.json' -print -quit)
test -n "$DEVELOPMENT_LEDGER"
for path in \
  "$DATA/report.json" \
  "$DATA/confirmation.sealed.jsonl" \
  "$RUN/compiler.pt" \
  "$RUN/gate_config.json" \
  "$RUN/development_report.json" \
  "$RUN/development_assessment.json" \
  "$DEVELOPMENT_LEDGER" \
  "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$EXECUTION_CORE"; do
  test -f "$path" || { echo "missing confirmation input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing confirmation output" >&2; exit 2; }
test "$(find "$DATA/access" -maxdepth 1 -type f | wc -l)" -eq 1

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("complete physical confirmation requires bf16 CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/confirm_sd_cst_complete_physical_fresh.py \
  --repo-root "$CODE_ROOT" \
  --data-dir "$DATA" \
  --checkpoint "$RUN/compiler.pt" \
  --gate-config "$RUN/gate_config.json" \
  --development-report "$RUN/development_report.json" \
  --development-assessment "$RUN/development_assessment.json" \
  --development-ledger "$DEVELOPMENT_LEDGER" \
  --joint-checkpoint "$JOINT" \
  --physical-checkpoint "$PHYSICAL" \
  --v1-checkpoint "$V1" \
  --v1-2-checkpoint "$V1_2" \
  --execution-core "$EXECUTION_CORE" \
  --out-dir "$OUTDIR" \
  --evaluator-source-commit "$EVALUATOR_SOURCE_COMMIT"

CONFIRMATION_LEDGER=$(find "$DATA/access" -maxdepth 1 -type f -name 'complete_physical_fresh_confirmation_*.json' -print -quit)
test -n "$CONFIRMATION_LEDGER"
"$PYTHON" train/assess_sd_cst_complete_physical_confirmation.py \
  --report "$OUTDIR/confirmation_report.json" \
  --checkpoint "$RUN/compiler.pt" \
  --gate-config "$RUN/gate_config.json" \
  --development-assessment "$RUN/development_assessment.json" \
  --authorization "$OUTDIR/authorization.json" \
  --packets "$OUTDIR/confirmation_packets.pt" \
  --evidence "$OUTDIR/confirmation_evidence.pt" \
  --executor "$OUTDIR/confirmation_executor.pt" \
  --ledger "$CONFIRMATION_LEDGER" \
  --output "$OUTDIR/confirmation_assessment.json"

sha256sum "$OUTDIR"/* "$CONFIRMATION_LEDGER"
echo "[sd-cst-complete-physical-confirmation] complete"
