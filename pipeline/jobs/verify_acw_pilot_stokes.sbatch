#!/bin/bash
#SBATCH -J shohin-acw-verify
#SBATCH -p normal
#SBATCH --nodelist=ec52
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_pilot_verify_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE
# The verifier also rejects the same job ID, Slurm node, or hostname before
# performing a fresh 3,400-update replay.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
PILOT=$BASE/artifacts/r12/acw_cgbr_pilot_v6
VERIFY=$BASE/artifacts/r12/acw_cgbr_pilot_v6_independent_verification
REGISTRY=$BASE/R12_ACW_PILOT_ARTIFACT_REGISTRY_V2.json
PYZIP=/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python313.zip

: "${SLURM_JOB_ID:?}"
: "${SLURM_CPUS_PER_TASK:?}"
: "${SLURM_JOB_NODELIST:?}"
: "${SLURM_NODELIST:?}"
: "${SLURM_SUBMIT_DIR:?}"
/usr/bin/rm -rf /tmp/shohin-acw-pycache
/usr/bin/install -d -m 700 /tmp/shohin-acw-pycache
[[ ! -e "$PYZIP" ]]

run_python() {
  /usr/bin/env -i \
    ATEN_CPU_CAPABILITY=avx2 \
    CUDA_VISIBLE_DEVICES= \
    DNNL_MAX_CPU_ISA=AVX2 \
    HOME=/home/sa305415 \
    KMP_DETERMINISTIC_REDUCTION=TRUE \
    LANG=C \
    LC_ALL=C \
    LOGNAME=sa305415 \
    MKL_CBWR=AVX2 \
    MKL_DYNAMIC=FALSE \
    MKL_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    OMP_NUM_THREADS=1 \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    OPENBLAS_CORETYPE=Haswell \
    OPENBLAS_NUM_THREADS=1 \
    PATH=/apps/slurm/current/bin:/usr/bin:/bin \
    PYTHONPATH=/lustre/fs1/home/sa305415/shohin_acw:/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/shohin-acw-pycache \
    PYTHONUTF8=1 \
    SHELL=/bin/bash \
    SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    SLURM_JOB_ID="$SLURM_JOB_ID" \
    SLURM_JOB_NAME="$SLURM_JOB_NAME" \
    SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
    SLURM_NODELIST="$SLURM_NODELIST" \
    SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR" \
    TZ=UTC \
    USER=sa305415 \
    "$PY" -S -P "$@"
}

test -s "$PILOT/report.json"
test ! -e "$VERIFY"
test ! -e "$REGISTRY"
cleanup_on_failure() {
  status=$?
  if ((status != 0)); then
    if [[ -e "$VERIFY" ]]; then
      /usr/bin/chmod -R u+w "$VERIFY" || true
      /usr/bin/rm -rf "$VERIFY"
    fi
    /usr/bin/rm -f "$REGISTRY" "$REGISTRY.partial"
  fi
  exit "$status"
}
trap cleanup_on_failure EXIT
cd "$BASE"
run_python "$BASE/pipeline/freeze_acw_curriculum.py" verify-pilot-independent
test -s "$VERIFY/verification.json"
test -s "$REGISTRY"
trap - EXIT
echo "[acw-pilot-independent-job] fresh different-node replay complete"
