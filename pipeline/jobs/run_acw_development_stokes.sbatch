#!/bin/bash
#SBATCH -J shohin-acw-development
#SBATCH -p normal
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH --no-requeue
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_development_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE
# Canonical four-job ACW development entrypoint. Submit these roles as four
# separate held jobs in dependency order; the committed plan binds each role
# to its exact job ID, node, job name, dependency, and script bytes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
PYZIP=/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python313.zip
ROOT=$BASE/artifacts/r12/acw_development_g1
DIRECT_PRIVATE_ROOT=$BASE/artifacts/r12/acw_development_g1_direct_verifier
FINAL_PRIVATE_ROOT=$BASE/artifacts/r12/acw_development_g1_final_verifier
PLAN=$BASE/R12_ACW_DEVELOPMENT_PLAN_V1.json
BUILDER=$BASE/pipeline/build_acw_development_manifest.py
ADJUDICATOR=$BASE/pipeline/adjudicate_acw_hidden_basis.py
PYCACHE=/tmp/shohin-acw-pycache
DIRECT_ARM=direct_state_acw
scored_arms=(
  acw
  dense_categorical
  addressed_continuous
  gru
  packet_token_transformer
  uniform_query_acw
  answer_motor
  source_retained
)

usage() {
  echo "usage: $0 phase1_producer|phase1_verifier|phase2_producer|phase2_verifier" >&2
  exit 64
}

[[ $# = 1 ]] || usage
ROLE=$1
case "$ROLE" in
  phase1_producer|phase1_verifier|phase2_producer|phase2_verifier) ;;
  *) usage ;;
esac

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

# Every scientific process must remain in the allocation's top-level batch
# task. The builder performs the authoritative role/job/plan validation.
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

builder() {
  run_python "$BUILDER" "$@"
}

adjudicate() {
  run_python "$ADJUDICATOR" "$@"
}

run_three_inputs() {
  local role=$1
  local pids=()
  local index
  local failed=0
  for index in 0 1 2; do
    builder run-inputs --root "$ROOT" --role "$role" --index "$index" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ((failed == 0))
}

run_three_attempts() {
  local role=$1
  local arm=$2
  local pids=()
  local index
  local failed=0
  for index in 0 1 2; do
    builder run-attempt --root "$ROOT" --role "$role" --index "$index" --arm "$arm" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  ((failed == 0))
}

record_predecessor_accounting() {
  local predecessor=$1
  builder stage-accounting --root "$ROOT" --role "$predecessor"
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
test -s "$PLAN"
test -s "$BUILDER"
test -s "$ADJUDICATOR"
test "$(/usr/bin/git --no-replace-objects rev-parse HEAD)" = \
  "$(/usr/bin/git --no-replace-objects ls-remote origin refs/heads/main | /usr/bin/awk '{print $1}')"
builder validate-plan --require-ready

case "$ROLE" in
  phase1_producer)
    test ! -e "$ROOT"
    /usr/bin/install -d -m 700 \
      "$ROOT" \
      "$ROOT/custody/stages" \
      "$ROOT/inputs/datasets" \
      "$ROOT/inputs/bundles" \
      "$ROOT/runs"
    /usr/bin/install -m 0444 "$PLAN" "$ROOT/development_plan.json"
    builder attempt-start --root "$ROOT"
    builder attempt-claim --root "$ROOT"
    builder stage-start --root "$ROOT" --role "$ROLE"
    run_three_inputs "$ROLE"
    run_three_attempts "$ROLE" "$DIRECT_ARM"
    builder direct-producer --root "$ROOT"
    builder stage-complete --root "$ROOT" --role "$ROLE" --stage phase1
    ;;

  phase1_verifier)
    test -d "$ROOT"
    record_predecessor_accounting phase1_producer
    builder stage-start --root "$ROOT" --role "$ROLE"
    test ! -e "$DIRECT_PRIVATE_ROOT"
    /usr/bin/install -d -m 700 \
      "$DIRECT_PRIVATE_ROOT/inputs/datasets" \
      "$DIRECT_PRIVATE_ROOT/inputs/bundles" \
      "$DIRECT_PRIVATE_ROOT/runs"
    run_three_inputs "$ROLE"
    run_three_attempts "$ROLE" "$DIRECT_ARM"
    builder verify-refits --root "$ROOT" --scope direct \
      --out "$ROOT/direct_refit_verification.json"
    builder direct --root "$ROOT"
    adjudicate \
      --manifest "$ROOT/direct_state_manifest.json" \
      --out "$ROOT/direct_state_decision.json" \
      --qualify-direct-state \
      --phase2-authorization-out "$ROOT/phase2_authorization.json"
    test -s "$ROOT/phase2_authorization.json"
    builder stage-complete --root "$ROOT" --role "$ROLE" --stage direct_verified
    ;;

  phase2_producer)
    test -s "$ROOT/phase2_authorization.json"
    record_predecessor_accounting phase1_verifier
    builder stage-start --root "$ROOT" --role "$ROLE"
    for arm in "${scored_arms[@]}"; do
      run_three_attempts "$ROLE" "$arm"
    done
    builder development-producer --root "$ROOT"
    builder stage-complete --root "$ROOT" --role "$ROLE" --stage phase2
    ;;

  phase2_verifier)
    record_predecessor_accounting phase2_producer
    builder stage-start --root "$ROOT" --role "$ROLE"
    test ! -e "$FINAL_PRIVATE_ROOT"
    /usr/bin/install -d -m 700 \
      "$FINAL_PRIVATE_ROOT/inputs/datasets" \
      "$FINAL_PRIVATE_ROOT/inputs/bundles" \
      "$FINAL_PRIVATE_ROOT/runs"
    run_three_inputs "$ROLE"
    for arm in "${scored_arms[@]}"; do
      run_three_attempts "$ROLE" "$arm"
    done
    builder verify-refits --root "$ROOT" --scope final \
      --out "$ROOT/final_refit_verification.json"
    builder development --root "$ROOT"
    builder stage-complete --root "$ROOT" --role "$ROLE" --stage final
    ;;
esac

trap - EXIT
/usr/bin/rm -rf "$PYCACHE"
echo "[acw-development] role=$ROLE complete root=$ROOT"
