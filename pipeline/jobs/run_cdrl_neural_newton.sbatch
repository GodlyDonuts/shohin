#!/bin/bash
#SBATCH -J shohin-cdrl-neural-v1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cdrl_neural_v1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$BASE/artifacts/r12/cdrl_neural_v1}
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/miniforge-torch-runtime-v1.tar}
RUNTIME_TAR_SHA256=${RUNTIME_TAR_SHA256:-af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c}

mkdir -p "$BASE/logs" "$(dirname "$ARTIFACT_ROOT")"
if [ -e "$ARTIFACT_ROOT/decision.json" ]; then
  echo "[cdrl-neural] refusing existing decision.json under $ARTIFACT_ROOT" >&2
  exit 3
fi

# Prefer the immutable runtime tarball when present; otherwise use login env python.
PY=""
cleanup_runtime() {
  if [ -n "${RUNTIME_DIR:-}" ]; then
    rm -rf -- "${RUNTIME_DIR}"
  fi
}
trap cleanup_runtime EXIT

if [ -s "$RUNTIME_TAR" ]; then
  OBSERVED=$(sha256sum "$RUNTIME_TAR" | awk '{print $1}')
  if [ "$OBSERVED" != "$RUNTIME_TAR_SHA256" ]; then
    echo "[cdrl-neural] runtime SHA mismatch: $OBSERVED" >&2
    exit 4
  fi
  RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
  RUNTIME_DIR=$(mktemp -d "$RUNTIME_PARENT/shohin_cdrl_runtime.${SLURM_JOB_ID:-manual}.XXXXXX")
  chmod 0700 "$RUNTIME_DIR"
  tar -C "$RUNTIME_DIR" -xf "$RUNTIME_TAR"
  PY=$RUNTIME_DIR/miniforge3/bin/python
  echo "[cdrl-neural] using runtime tar sha256=$OBSERVED"
elif [ -x "$BASE/miniforge3/bin/python" ]; then
  PY=$BASE/miniforge3/bin/python
  echo "[cdrl-neural] using BASE miniforge python"
else
  echo "[cdrl-neural] no python runtime found" >&2
  exit 5
fi

# CUDA preflight with retry for contended H100 nodes
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if "$PY" - <<'PY'
import torch, sys
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(11)
x = torch.ones(4, device="cuda", dtype=torch.bfloat16)
y = x @ x
print("bf16_ok", float(y.sum().item()), torch.cuda.get_device_name(0))
PY
  then
    echo "[cdrl-neural] cuda preflight ok attempt=$attempt"
    break
  fi
  echo "[cdrl-neural] cuda preflight failed attempt=$attempt; sleeping"
  sleep 30
  if [ "$attempt" -eq 10 ]; then
    echo "[cdrl-neural] cuda preflight exhausted" >&2
    exit 12
  fi
done

cd "$BASE"
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
echo "[cdrl-neural] host=$(hostname) job=${SLURM_JOB_ID:-none} artifact=$ARTIFACT_ROOT"
"$PY" -m pipeline.cdrl_neural_board --artifact-root "$ARTIFACT_ROOT" --device cuda
echo "[cdrl-neural] done"
