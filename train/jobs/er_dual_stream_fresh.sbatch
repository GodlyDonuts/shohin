#!/bin/bash
#SBATCH --job-name=er-ds-fresh-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/er_dual_stream_fresh_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the immutable fresh dual-stream board}
CANARY=${CANARY:?set CANARY to the exact qualified v1.2 checkpoint}
JOINT=${JOINT:?set JOINT to the exact joint checkpoint}
PHYSICAL=${PHYSICAL:?set PHYSICAL to the exact physical checkpoint}
V1=${V1:?set V1 to the exact complete-local v1 checkpoint}
V1_2=${V1_2:?set V1_2 to the exact v1.2 endpoint}
CONFIRMED=${CONFIRMED:?set CONFIRMED to the confirmed v1.3 compiler}
CONFIRMATION_ASSESSMENT=${CONFIRMATION_ASSESSMENT:?set the confirmed v1.3 assessment}
WITNESS=${WITNESS:?set WITNESS to the confirmed witness-equality compiler}
WITNESS_CONFIRMATION_ASSESSMENT=${WITNESS_CONFIRMATION_ASSESSMENT:?set witness confirmation assessment}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated output directory}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact scientific source commit}
SEED=${SEED:?set the post-commit training seed}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in \
  "$DATA/report.json" \
  "$DATA/train.jsonl" \
  "$DATA/development.jsonl" \
  "$DATA/confirmation.jsonl" \
  "$CANARY" "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$CONFIRMED" \
  "$CONFIRMATION_ASSESSMENT" "$WITNESS" "$WITNESS_CONFIRMATION_ASSESSMENT"; do
  test -f "$path" || { echo "missing fresh dual-stream input: $path" >&2; exit 2; }
done
test "$(stat -c '%a' "$DATA/confirmation.jsonl")" = 600
test ! -e "$DATA/access" || ! find "$DATA/access" -mindepth 1 -print -quit | grep -q .
test ! -e "$OUTDIR" || { echo "refusing existing fresh output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("fresh dual-stream qualification has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("fresh dual-stream qualification requires bf16 H100-class CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/pilot_er_dual_stream_fresh.py \
  --repo-root "$CODE_ROOT" \
  --data-dir "$DATA" \
  --canary-checkpoint "$CANARY" \
  --joint-checkpoint "$JOINT" \
  --physical-checkpoint "$PHYSICAL" \
  --v1-checkpoint "$V1" \
  --v1-2-checkpoint "$V1_2" \
  --confirmed-checkpoint "$CONFIRMED" \
  --confirmation-assessment "$CONFIRMATION_ASSESSMENT" \
  --witness-checkpoint "$WITNESS" \
  --witness-confirmation-assessment "$WITNESS_CONFIRMATION_ASSESSMENT" \
  --out-dir "$OUTDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --seed "$SEED"

LEDGER=$(find "$DATA/access" -maxdepth 1 -type f -name 'er_dual_stream_fresh_development_*.json' -print -quit)
test -n "$LEDGER"
"$PYTHON" train/assess_er_dual_stream_fresh.py \
  --report "$OUTDIR/development_report.json" \
  --checkpoint "$OUTDIR/compiler.pt" \
  --evidence "$OUTDIR/development_evidence.pt" \
  --access-ledger "$LEDGER" \
  --output "$OUTDIR/development_assessment.json"

sha256sum \
  "$OUTDIR/compiler.pt" \
  "$OUTDIR/development_evidence.pt" \
  "$OUTDIR/development_report.json" \
  "$OUTDIR/development_assessment.json"
echo "[er-ds-fresh-v1] complete; confirmation remains unopened"
