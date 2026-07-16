#!/bin/bash
# Verify the published minimal runtime by extracting and importing it locally.
#SBATCH -J shohin-runtime-smoke
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/runtime_smoke_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME_TAR=$BASE/artifacts/runtime/miniforge-torch-runtime-v1.tar
EXPECTED_SHA256=af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c

[ -s "$RUNTIME_TAR" ] || { echo "[runtime-smoke] missing runtime tar" >&2; exit 2; }
[ ! -L "$RUNTIME_TAR" ] || { echo "[runtime-smoke] symlink rejected" >&2; exit 3; }
[ ! -w "$RUNTIME_TAR" ] || { echo "[runtime-smoke] writable runtime rejected" >&2; exit 4; }
OBSERVED_SHA256=$(sha256sum "$RUNTIME_TAR" | awk '{print $1}')
[ "$OBSERVED_SHA256" = "$EXPECTED_SHA256" ] || {
  echo "[runtime-smoke] runtime SHA-256 mismatch" >&2
  exit 5
}

RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
RUNTIME_DIR=$(mktemp -d "$RUNTIME_PARENT/shohin_runtime_smoke.${SLURM_JOB_ID:-manual}.XXXXXX")
chmod 0700 "$RUNTIME_DIR"
cleanup() {
  rm -rf -- "${RUNTIME_DIR:-}"
}
trap cleanup EXIT

START=$(date +%s)
tar -C "$RUNTIME_DIR" -xf "$RUNTIME_TAR"
EXTRACT_SECONDS=$(( $(date +%s) - START ))
PY=$RUNTIME_DIR/miniforge3/bin/python
[ -x "$PY" ] || { echo "[runtime-smoke] extracted Python missing" >&2; exit 6; }

export PYTHONNOUSERSITE=1
export SHOHIN_RUNTIME_PREFIX=$RUNTIME_DIR/miniforge3
START=$(date +%s)
"$PY" -u - <<'PY'
import os
import pathlib
import sys

import numpy
import sympy
import torch

prefix = pathlib.Path(os.environ["SHOHIN_RUNTIME_PREFIX"]).resolve()
if pathlib.Path(sys.prefix).resolve() != prefix:
    raise SystemExit("[runtime-smoke] Python prefix escaped node-local runtime")
for name, module in (("torch", torch), ("numpy", numpy), ("sympy", sympy)):
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to(prefix):
        raise SystemExit(f"[runtime-smoke] {name} escaped node-local runtime: {path}")
print(
    "[runtime-smoke] imports passed "
    f"python={sys.version.split()[0]} torch={torch.__version__} "
    f"numpy={numpy.__version__} sympy={sympy.__version__}",
    flush=True,
)
PY
IMPORT_SECONDS=$(( $(date +%s) - START ))
echo "[runtime-smoke] pass sha256=$OBSERVED_SHA256 extract_seconds=$EXTRACT_SECONDS import_seconds=$IMPORT_SECONDS"
