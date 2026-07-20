#!/bin/bash
#SBATCH --job-name=sd-cst-proj-fresh
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_projected_fresh_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the one-shot projected fresh board}
PARENT=${PARENT:?set PARENT to the exact byte-parent checkpoint}
EXECUTION_CORE=${EXECUTION_CORE:?set EXECUTION_CORE to the certified mechanics core}
CONSUMED_PROJECTED=${CONSUMED_PROJECTED:?set CONSUMED_PROJECTED to the diagnostic compiler}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated output directory}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact post-prereg source commit}
PHASE=${PHASE:-development}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

SOURCE_PATHS=(
  R12_SD_CST_PROJECTED_MECHANICS_PREREG.md
  R12_SD_CST_PROJECTED_BINDING_PILOT_PREREG.md
  R12_SD_CST_PROJECTED_FRESH_BOARD_PREREG.md
  pipeline/assess_sd_cst_projected_fresh.py
  pipeline/audit_sd_cst_board.py
  pipeline/build_sd_cst_board.py
  pipeline/build_sd_cst_projected_board.py
  pipeline/test_assess_sd_cst_projected_fresh.py
  pipeline/test_build_sd_cst_projected_board.py
  train/assess_sd_cst_projected_mechanics.py
  train/pilot_sd_cst_binding_bus.py
  train/pilot_sd_cst_byte_addressed.py
  train/pilot_sd_cst_hierarchical_binding.py
  train/projected_sd_cst_fresh.py
  train/run_sd_cst_hard_packets.py
  train/sd_cst.py
  train/sd_cst_binding_bus.py
  train/sd_cst_byte_addressed.py
  train/model.py
  train/train_sd_cst.py
  train/test_projected_sd_cst_fresh.py
  train/test_run_sd_cst_hard_packets.py
  train/train_eval_sd_cst_projected_fresh.py
  train/jobs/sd_cst_projected_fresh.sbatch
)

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in "${SOURCE_PATHS[@]}"; do
  git cat-file -e "$SOURCE_COMMIT:$path"
done
for path in "$DATA/report.json" "$PARENT" "$EXECUTION_CORE" "$CONSUMED_PROJECTED"; do
  test -f "$path" || { echo "missing projected fresh input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing projected fresh output: $OUTDIR" >&2; exit 2; }
mkdir -p "$OUTDIR"

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("projected fresh run requires bf16 CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8:
    raise SystemExit("projected fresh run requires Ampere-or-newer GPU")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

if [[ "$PHASE" == development ]]; then
  TRAINING_SEED=${TRAINING_SEED:?set the post-source-commit training seed}
  export PYTHONHASHSEED=$((TRAINING_SEED % 4294967296))
  "$PYTHON" train/train_eval_sd_cst_projected_fresh.py train \
    --data-dir "$DATA" \
    --parent-checkpoint "$PARENT" \
    --out "$OUTDIR/checkpoint.pt" \
    --repo-root "$CODE_ROOT" \
    --source-commit "$SOURCE_COMMIT" \
    --seed "$TRAINING_SEED"

  "$PYTHON" train/train_eval_sd_cst_projected_fresh.py config \
    --data-dir "$DATA" \
    --checkpoint "$OUTDIR/checkpoint.pt" \
    --parent-checkpoint "$PARENT" \
    --execution-core "$EXECUTION_CORE" \
    --consumed-checkpoint "$CONSUMED_PROJECTED" \
    --out "$OUTDIR/gate_config.json" \
    --repo-root "$CODE_ROOT" \
    --source-commit "$SOURCE_COMMIT"

  "$PYTHON" train/train_eval_sd_cst_projected_fresh.py evaluate \
    --split sd_cst_development \
    --data-dir "$DATA" \
    --checkpoint "$OUTDIR/checkpoint.pt" \
    --parent-checkpoint "$PARENT" \
    --execution-core "$EXECUTION_CORE" \
    --consumed-checkpoint "$CONSUMED_PROJECTED" \
    --gate-config "$OUTDIR/gate_config.json" \
    --out "$OUTDIR/development_evaluation.json" \
    --repo-root "$CODE_ROOT" \
    --source-commit "$SOURCE_COMMIT"

  "$PYTHON" pipeline/assess_sd_cst_projected_fresh.py \
    --evaluation "$OUTDIR/development_evaluation.json" \
    --gate-config "$OUTDIR/gate_config.json" \
    --hard-packets "$OUTDIR/development_evaluation.packets.pt" \
    --executor-outputs "$OUTDIR/development_evaluation.executor.pt" \
    --out "$OUTDIR/development_assessment.json"
elif [[ "$PHASE" == confirmation ]]; then
  CHECKPOINT=${CHECKPOINT:?set the exact authorized development checkpoint}
  GATE_CONFIG=${GATE_CONFIG:?set the exact development gate config}
  AUTHORIZATION=${AUTHORIZATION:?set the passing development assessment}
  DEVELOPMENT_EVALUATION=${DEVELOPMENT_EVALUATION:?set the exact development evaluation}
  DEVELOPMENT_PACKETS=${DEVELOPMENT_PACKETS:?set the exact development packet bundle}
  DEVELOPMENT_EXECUTOR=${DEVELOPMENT_EXECUTOR:?set the exact development executor output}
  "$PYTHON" train/train_eval_sd_cst_projected_fresh.py evaluate \
    --split sd_cst_confirmation \
    --data-dir "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --parent-checkpoint "$PARENT" \
    --execution-core "$EXECUTION_CORE" \
    --consumed-checkpoint "$CONSUMED_PROJECTED" \
    --gate-config "$GATE_CONFIG" \
    --authorization "$AUTHORIZATION" \
    --development-evaluation "$DEVELOPMENT_EVALUATION" \
    --development-packets "$DEVELOPMENT_PACKETS" \
    --development-executor "$DEVELOPMENT_EXECUTOR" \
    --out "$OUTDIR/confirmation_evaluation.json" \
    --repo-root "$CODE_ROOT" \
    --source-commit "$SOURCE_COMMIT"

  "$PYTHON" pipeline/assess_sd_cst_projected_fresh.py \
    --evaluation "$OUTDIR/confirmation_evaluation.json" \
    --gate-config "$GATE_CONFIG" \
    --hard-packets "$OUTDIR/confirmation_evaluation.packets.pt" \
    --executor-outputs "$OUTDIR/confirmation_evaluation.executor.pt" \
    --out "$OUTDIR/confirmation_assessment.json"
else
  echo "invalid PHASE: $PHASE" >&2
  exit 2
fi

find "$OUTDIR" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum
echo "[sd-cst-projected-fresh] $PHASE complete"
