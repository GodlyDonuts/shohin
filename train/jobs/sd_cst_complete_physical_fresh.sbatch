#!/bin/bash
#SBATCH --job-name=sd-cst-phys-fresh
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_complete_physical_fresh_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the fresh physical board}
JOINT=${JOINT:?set JOINT to the exact joint checkpoint}
PHYSICAL=${PHYSICAL:?set PHYSICAL to the exact physical checkpoint}
V1=${V1:?set V1 to the exact complete-local v1 checkpoint}
V1_2=${V1_2:?set V1_2 to the exact v1.2 endpoint}
EXECUTION_CORE=${EXECUTION_CORE:?set EXECUTION_CORE to the exact categorical core}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated output directory}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact scientific source commit}
SEED=${SEED:?set the post-source-commit training seed}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in \
  "$DATA/report.json" \
  "$DATA/train.jsonl" \
  "$DATA/development.jsonl" \
  "$DATA/confirmation.sealed.jsonl" \
  "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$EXECUTION_CORE"; do
  test -f "$path" || { echo "missing fresh physical input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing fresh output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("fresh physical qualification has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("fresh physical qualification requires bf16 H100-class CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/pilot_sd_cst_complete_physical_fresh.py \
  --repo-root "$CODE_ROOT" \
  --data-dir "$DATA" \
  --joint-checkpoint "$JOINT" \
  --physical-checkpoint "$PHYSICAL" \
  --v1-checkpoint "$V1" \
  --v1-2-checkpoint "$V1_2" \
  --execution-core "$EXECUTION_CORE" \
  --out-dir "$OUTDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --seed "$SEED"

"$PYTHON" train/assess_sd_cst_complete_physical_fresh.py \
  --report "$OUTDIR/development_report.json" \
  --checkpoint "$OUTDIR/compiler.pt" \
  --gate-config "$OUTDIR/gate_config.json" \
  --packets "$OUTDIR/development_packets.pt" \
  --evidence "$OUTDIR/development_evidence.pt" \
  --executor "$OUTDIR/development_executor.pt" \
  --access-ledger "$(find "$DATA/access" -maxdepth 1 -type f -name 'complete_physical_fresh_development_*.json' -print -quit)" \
  --output "$OUTDIR/development_assessment.json"

sha256sum \
  "$OUTDIR/compiler.pt" \
  "$OUTDIR/gate_config.json" \
  "$OUTDIR/development_packets.pt" \
  "$OUTDIR/development_evidence.pt" \
  "$OUTDIR/development_executor.pt" \
  "$OUTDIR/development_report.json" \
  "$OUTDIR/development_assessment.json"
echo "[sd-cst-complete-physical-fresh] complete; confirmation remains unopened"
