#!/bin/bash
#SBATCH -J shohin-acw-pilot
#SBATCH -p normal
#SBATCH --nodelist=ec51
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_pilot_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE
# Non-scored deterministic CPU pilot. One command owns data generation, two
# isolated fits, independent recomputation, publication, and final reopening.
# This cannot write a Shohin checkpoint or open a confirmation seed.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
DATA=$BASE/artifacts/r12/acw_pilot_domain_v3_runtime_v2
REPLAY_A=$BASE/artifacts/r12/acw_cgbr_pilot_v6_replay_a
REPLAY_B=$BASE/artifacts/r12/acw_cgbr_pilot_v6_replay_b
OUT=$BASE/artifacts/r12/acw_cgbr_pilot_v6
VERIFY=$BASE/artifacts/r12/acw_cgbr_pilot_v6_independent_verification
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

/usr/bin/mkdir -p "$BASE/logs" "$BASE/artifacts/r12"
test ! -e "$DATA"
test ! -e "$REPLAY_A"
test ! -e "$REPLAY_B"
test ! -e "$OUT"
test ! -e "$VERIFY"
cd "$BASE"
run_python -m unittest \
  pipeline.test_addressed_categorical_workspace \
  pipeline.test_audit_addressed_categorical_workspace_symbolic \
  pipeline.test_generate_acw_hidden_basis \
  pipeline.test_acw_nist_beacon \
  pipeline.test_acw_hidden_basis_training \
  pipeline.test_freeze_acw_curriculum \
  pipeline.test_evaluate_acw_hidden_basis \
  pipeline.test_adjudicate_acw_hidden_basis
run_python "$BASE/pipeline/freeze_acw_curriculum.py" pilot-run
run_python "$BASE/pipeline/freeze_acw_curriculum.py" verify-pilot
test -s "$DATA/manifest.json"
test -s "$OUT/report.json"
test -s "$OUT/cgb_schedule.jsonl"
test -s "$OUT/uniform_schedule.jsonl"
test -s "$OUT/replay_comparison.json"
echo "[acw-pilot-job] independently recomputed non-scored curriculum pilot"
