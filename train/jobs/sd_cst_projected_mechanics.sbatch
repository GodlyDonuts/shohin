#!/bin/bash
#SBATCH --job-name=sd-cst-proj-mech
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_projected_mechanics_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the consumed frozen SD-CST v1.1 board}
COMPILER=${COMPILER:?set COMPILER to the exact projected compiler checkpoint}
PILOT_REPORT=${PILOT_REPORT:?set PILOT_REPORT to the exact projected report}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated training-only output directory}
MECHANICS_SEED=${MECHANICS_SEED:?set the post-source-commit mechanics seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact mechanics source commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

SOURCE_PATHS=(
  R12_SD_CST_PROJECTED_BINDING_PILOT_PREREG.md
  R12_SD_CST_PROJECTED_MECHANICS_PREREG.md
  train/sd_cst.py
  train/sd_cst_byte_addressed.py
  train/sd_cst_binding_bus.py
  train/pilot_sd_cst_byte_addressed.py
  train/pilot_sd_cst_binding_bus.py
  train/pilot_sd_cst_hierarchical_binding.py
  train/assess_sd_cst_projected_mechanics.py
  train/run_sd_cst_hard_packets.py
  train/jobs/sd_cst_projected_mechanics.sbatch
)

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in "${SOURCE_PATHS[@]}"; do
  git cat-file -e "$SOURCE_COMMIT:$path"
done
for path in "$DATA/report.json" "$DATA/train.jsonl" "$COMPILER" "$PILOT_REPORT"; do
  test -f "$path" || { echo "missing training-only mechanics input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing mechanics output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((MECHANICS_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("projected mechanics has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("projected mechanics requires a bf16-capable GPU")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/assess_sd_cst_projected_mechanics.py \
  --data-dir "$DATA" \
  --compiler-checkpoint "$COMPILER" \
  --pilot-report "$PILOT_REPORT" \
  --out-dir "$OUTDIR" \
  --source-commit "$SOURCE_COMMIT" \
  --seed "$MECHANICS_SEED"

sha256sum \
  "$OUTDIR/execution_core.pt" \
  "$OUTDIR/hard_packets.pt" \
  "$OUTDIR/hard_packet_outputs.pt" \
  "$OUTDIR/report.json"
echo "[sd-cst-projected-mechanics] complete; no scored split was addressed"
