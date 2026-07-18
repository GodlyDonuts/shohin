#!/usr/bin/env bash
#SBATCH --job-name=scert-cpu-reference
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
umask 077

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "SCERT CPU protocol refuses visible GPUs" >&2
  exit 64
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT="${SCERT_ROOT:-${HOME}/shohin}"
PYTHON="${SCERT_PYTHON:-${ROOT}/.venv/bin/python}"
MODE="${SCERT_CPU_MODE:-cpu-reference}"
OUT="${SCERT_CPU_OUT:-}"

if [[ ! -d "${ROOT}/.git" || ! -x "${PYTHON}" ]]; then
  echo "SCERT root or Python runtime is unavailable" >&2
  exit 65
fi
if [[ -z "${OUT}" || -e "${OUT}" ]]; then
  echo "SCERT_CPU_OUT must name a new output directory" >&2
  exit 66
fi
mkdir -m 700 "${OUT}"

case "${MODE}" in
  cpu-reference)
    exec "${PYTHON}" "${ROOT}/train/scert.py" cpu-reference \
      --report "${OUT}/cpu_reference.json"
    ;;
  write-cpu-manifests)
    exec "${PYTHON}" "${ROOT}/train/scert.py" write-cpu-manifests \
      --source "${OUT}/source_manifest.json" \
      --runtime "${OUT}/runtime_manifest.json"
    ;;
  *)
    echo "Only CPU reference and CPU manifest modes are authorized" >&2
    exit 67
    ;;
esac
