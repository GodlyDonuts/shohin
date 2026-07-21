#!/bin/bash
#SBATCH --job-name=er-ds-opcode-v13
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/er_dual_stream_opcode_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set exact committed capsule}
DATA=${DATA:?set immutable fresh board}
CANARY=${CANARY:?set qualified v1.2 canary}
JOINT=${JOINT:?set joint checkpoint}
PHYSICAL=${PHYSICAL:?set physical checkpoint}
V1=${V1:?set complete-local v1 checkpoint}
V1_2=${V1_2:?set complete-local v1.2 checkpoint}
CONFIRMED=${CONFIRMED:?set confirmed physical compiler}
CONFIRMATION_ASSESSMENT=${CONFIRMATION_ASSESSMENT:?set physical confirmation assessment}
WITNESS=${WITNESS:?set confirmed witness compiler}
WITNESS_CONFIRMATION_ASSESSMENT=${WITNESS_CONFIRMATION_ASSESSMENT:?set witness assessment}
OUTDIR=${OUTDIR:?set isolated output}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
SEED=${SEED:?set post-commit seed}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$OUTDIR"
for path in "$DATA/report.json" "$DATA/train.jsonl" "$CANARY" "$JOINT" "$PHYSICAL" "$V1" "$V1_2" "$CONFIRMED" "$CONFIRMATION_ASSESSMENT" "$WITNESS" "$WITNESS_CONFIRMATION_ASSESSMENT"; do
  test -f "$path"
done

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" train/pilot_er_dual_stream_opcode_canary.py \
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

sha256sum "$OUTDIR/compiler.pt" "$OUTDIR/train_probe_evidence.pt" "$OUTDIR/train_probe_report.json"
