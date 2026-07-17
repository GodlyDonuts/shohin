#!/bin/bash
#SBATCH -J shohin-acw-dev-monitor
#SBATCH -p normal
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH --no-requeue
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_development_monitor_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE
# Canonical post-job monitor. This fifth held CPU job runs only after all four
# scientific roles complete. It reopens their evidence under the pinned Stokes
# runtime, computes the retained baseline without printing metrics, freezes the
# trees, and writes an externally anchorable terminal receipt.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
PYZIP=/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python313.zip
ROOT=$BASE/artifacts/r12/acw_development_g1
BUILDER=$BASE/pipeline/build_acw_development_manifest.py
PYCACHE=/tmp/shohin-acw-monitor-pycache-$SLURM_JOB_ID

[[ $# = 0 ]]
: "${SLURM_CPUS_PER_TASK:?}"
: "${SLURM_JOB_ID:?}"
: "${SLURM_JOB_NAME:?}"
: "${SLURM_JOB_NODELIST:?}"
: "${SLURM_NODELIST:?}"
: "${SLURM_SUBMIT_DIR:?}"
[[ "$SLURM_CPUS_PER_TASK" = 4 ]]
[[ "$SLURM_JOB_NODELIST" = "$SLURM_NODELIST" ]]
[[ "$SLURM_SUBMIT_DIR" = "$BASE" ]]
[[ ! -e "$PYZIP" ]]
test "$({ /usr/bin/awk -F ': ' '/model name/{print $2; exit}' /proc/cpuinfo; })" = \
  "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz"

BATCH_TASK="/slurm/uid_$(/usr/bin/id -u)/job_${SLURM_JOB_ID}/step_batch/task_0"
/usr/bin/grep -Fq "$BATCH_TASK" /proc/self/cgroup

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
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$PYCACHE" \
    PYTHONPATH=$BASE:/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages \
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

cleanup() {
  local status=$?
  /usr/bin/rm -rf "$PYCACHE"
  exit "$status"
}
trap cleanup EXIT

/usr/bin/rm -rf "$PYCACHE"
/usr/bin/install -d -m 700 "$PYCACHE"
cd "$BASE"
test "$({ /usr/bin/git --no-replace-objects rev-parse HEAD; })" = \
  "$({ /usr/bin/git --no-replace-objects ls-remote origin refs/heads/main | /usr/bin/awk '{print $1}'; })"
run_python "$BUILDER" validate-plan --require-ready
if ! run_python "$BUILDER" terminal-accounting --root "$ROOT"; then
  echo "[acw-development-monitor] retrying resumable terminal accounting" >&2
  run_python "$BUILDER" terminal-accounting --root "$ROOT"
fi
if ! run_python "$BUILDER" verify-terminal-accounting --root "$ROOT"; then
  echo "[acw-development-monitor] retrying terminal verification" >&2
  run_python "$BUILDER" terminal-accounting --root "$ROOT"
  run_python "$BUILDER" verify-terminal-accounting --root "$ROOT"
fi

trap - EXIT
/usr/bin/rm -rf "$PYCACHE"
echo "[acw-development-monitor] complete root=$ROOT performance_claim_ready=0"
