#!/bin/bash
#SBATCH -J shohin-acw-pilot
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin_acw/logs/acw_pilot_%j.out
#SBATCH --open-mode=append
# Non-scored deterministic CPU pilot. One command owns data generation, two
# isolated fits, independent recomputation, publication, and final reopening.
# This cannot write a Shohin checkpoint or open a confirmation seed.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin_acw
PY=${PY:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}
DATA=$BASE/artifacts/r12/acw_pilot_domain_v3
REPLAY_A=$BASE/artifacts/r12/acw_cgbr_pilot_v4_replay_a
REPLAY_B=$BASE/artifacts/r12/acw_cgbr_pilot_v4_replay_b
OUT=$BASE/artifacts/r12/acw_cgbr_pilot_v4

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0

mkdir -p "$BASE/logs" "$BASE/artifacts/r12"
test ! -e "$DATA"
test ! -e "$REPLAY_A"
test ! -e "$REPLAY_B"
test ! -e "$OUT"
cd "$BASE"
"$PY" -m unittest \
  pipeline.test_addressed_categorical_workspace \
  pipeline.test_audit_addressed_categorical_workspace_symbolic \
  pipeline.test_generate_acw_hidden_basis \
  pipeline.test_acw_nist_beacon \
  pipeline.test_acw_hidden_basis_training \
  pipeline.test_freeze_acw_curriculum \
  pipeline.test_evaluate_acw_hidden_basis \
  pipeline.test_adjudicate_acw_hidden_basis
"$PY" -m pipeline.freeze_acw_curriculum pilot-run
"$PY" -m pipeline.freeze_acw_curriculum verify-pilot
test -s "$DATA/manifest.json"
test -s "$OUT/report.json"
test -s "$OUT/cgb_schedule.jsonl"
test -s "$OUT/uniform_schedule.jsonl"
test -s "$OUT/replay_comparison.json"
echo "[acw-pilot-job] independently recomputed non-scored curriculum pilot"
