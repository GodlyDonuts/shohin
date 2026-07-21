#!/bin/bash
#SBATCH --job-name=er-am-canary
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/er_addressed_marginal_train_canary_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the fixed ER-TT board}
JOINT=${JOINT:?set JOINT to the exact joint checkpoint}
PHYSICAL=${PHYSICAL:?set PHYSICAL to the exact physical checkpoint}
V1=${V1:?set V1 to the exact complete-local v1 checkpoint}
V1_2=${V1_2:?set V1_2 to the exact v1.2 endpoint}
CONFIRMED=${CONFIRMED:?set CONFIRMED to the confirmed v1.3 compiler}
CONFIRMATION_ASSESSMENT=${CONFIRMATION_ASSESSMENT:?set confirmed assessment}
WITNESS=${WITNESS:?set WITNESS to the confirmed witness compiler}
WITNESS_CONFIRMATION_ASSESSMENT=${WITNESS_CONFIRMATION_ASSESSMENT:?set witness assessment}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated output directory}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact scientific source commit}
SEED=${SEED:?set the post-commit canary seed}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in \
  "$DATA/report.json" "$DATA/train.jsonl" \
  "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$CONFIRMED" \
  "$CONFIRMATION_ASSESSMENT" "$WITNESS" "$WITNESS_CONFIRMATION_ASSESSMENT"; do
  test -f "$path" || { echo "missing addressed marginal input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing addressed output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("addressed marginal canary has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("addressed marginal canary requires bf16 H100-class CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/pilot_er_addressed_marginal_train_canary.py \
  --repo-root "$CODE_ROOT" \
  --data-dir "$DATA" \
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

sha256sum \
  "$OUTDIR/compiler.pt" \
  "$OUTDIR/train_probe_evidence.pt" \
  "$OUTDIR/train_probe_report.json"
echo "[er-am-canary] complete; no development or confirmation split was read"
