#!/bin/bash
# Build a relocatable, read-only Python/Torch runtime for node-local staging.
# This is CPU-only infrastructure and never reads a checkpoint or training data.
#SBATCH -J shohin-runtime-pack
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/runtime_pack_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
ENV_ROOT=$BASE/miniforge3
SITE=$ENV_ROOT/lib/python3.13/site-packages
OUT=${OUT:-$BASE/artifacts/runtime/miniforge-torch-runtime-v1.tar}
PART=$OUT.part.${SLURM_JOB_ID:-manual}

PACKAGES=(
  torch
  torch-2.6.0+cu124.dist-info
  functorch
  torchgen
  numpy
  numpy.libs
  numpy-2.5.0.dist-info
  nvidia
  cusparselt
  sympy
  sympy-1.13.1.dist-info
  mpmath
  networkx
  jinja2
  markupsafe
  filelock
  fsspec
  packaging
  typing_extensions.py
)

[ -x "$ENV_ROOT/bin/python" ] || { echo "[runtime-pack] missing Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[runtime-pack] refusing existing output: $OUT" >&2; exit 3; }
[ ! -e "$OUT.sha256" ] || { echo "[runtime-pack] refusing existing digest" >&2; exit 3; }
for package in "${PACKAGES[@]}"; do
  [ -e "$SITE/$package" ] || { echo "[runtime-pack] missing package: $package" >&2; exit 4; }
done

mkdir -p "$(dirname "$OUT")"
umask 077
cleanup() {
  rm -f -- "${PART:-}"
}
trap cleanup EXIT

echo "[runtime-pack] archiving Python standard library and native runtime"
tar \
  --exclude='miniforge3/lib/python3.13/site-packages/*' \
  --exclude='*/__pycache__/*' \
  --exclude='miniforge3/lib/python3.13/test/*' \
  -C "$BASE" -cf "$PART" miniforge3/bin miniforge3/lib

echo "[runtime-pack] appending required site packages"
PACKAGE_PATHS=()
for package in "${PACKAGES[@]}"; do
  PACKAGE_PATHS+=("miniforge3/lib/python3.13/site-packages/$package")
done
tar \
  --exclude='*/__pycache__/*' \
  --exclude='*/tests/*' \
  --exclude='*/torch/include/*' \
  --exclude='*/torch/share/*' \
  -C "$BASE" -rf "$PART" "${PACKAGE_PATHS[@]}"
tar -tf "$PART" >/dev/null

DIGEST=$(sha256sum "$PART" | awk '{print $1}')
printf '%s  %s\n' "$DIGEST" "$(basename "$OUT")" > "$OUT.sha256.part"
mv "$PART" "$OUT"
mv "$OUT.sha256.part" "$OUT.sha256"
chmod 0444 "$OUT" "$OUT.sha256"
trap - EXIT
echo "[runtime-pack] complete sha256=$DIGEST"
ls -lh "$OUT"
