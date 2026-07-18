#!/bin/bash
#SBATCH --job-name=carry-motor-plan
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/carry_motor_plan_%j_%r.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/carry_motor_plan_%j_%r.err

# Freeze the canonical carry-motor extraction plan on CPU-only Stokes compute.
# The plan binds every later shard before any H100 extraction is allowed.
set -euo pipefail
umask 077

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to an isolated clean reviewed checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed lowercase 40-hex commit}
CANONICAL_DATA_ROOT=/lustre/fs1/home/sa305415/shohin
DATA_ROOT=$CANONICAL_DATA_ROOT
PY=$DATA_ROOT/miniforge3/bin/python
CKPT=$DATA_ROOT/train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt
TOKENIZER=$DATA_ROOT/artifacts/shohin-tok-32k.json
EPISODES=$DATA_ROOT/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl
CYCLE=$DATA_ROOT/artifacts/evals/drs_causal_cycle_post_drs_r3.json
CONFIRMATION_COMMITMENT=${CONFIRMATION_COMMITMENT:?set the sealed pre-fit confirmation commitment}
CONFIRMATION_COMMITMENT_SHA256=${CONFIRMATION_COMMITMENT_SHA256:?set the reviewed confirmation commitment digest}
EXPECTED_CONFIRMATION_COMMITMENT=$DATA_ROOT/artifacts/carry_motor/confirmation_commitments/commitment_${SOURCE_COMMIT}/commitment.json
PLAN_ROOT=$DATA_ROOT/artifacts/carry_motor/canonical_${SOURCE_COMMIT}
JOB_ID=${SLURM_JOB_ID:?this canonical plan must run under Slurm}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}

SCIENTIFIC_SOURCES=(
  R12_CAUSAL_CARRY_MOTOR_PREREG.md
  train/causal_carry_motor.py
  train/test_causal_carry_motor.py
  train/jobs/causal_carry_motor.sbatch
  train/digitwise_controller.py
  train/digitwise_protocol.py
  train/eval_suite.py
  train/model.py
  train/probe_digitwise_workspace.py
  pipeline/jobs/causal_carry_motor_plan_stokes.sbatch
)

fail() {
  echo "[carry-motor-plan] $*" >&2
  exit 2
}

[[ "$SOURCE_ROOT" = /* ]] || fail "SOURCE_ROOT must be absolute"
[[ "$DATA_ROOT" = "$CANONICAL_DATA_ROOT" ]] || \
  fail "canonical data root is immutable"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  fail "SOURCE_COMMIT must be a lowercase 40-hex commit"
[[ "$CONFIRMATION_COMMITMENT" = "$EXPECTED_CONFIRMATION_COMMITMENT" ]] || \
  fail "confirmation commitment path is not commit-bound"
[[ "$CONFIRMATION_COMMITMENT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  fail "confirmation commitment receipt must be lowercase SHA-256"
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] && [[ "$RESTART_COUNT" -eq 0 ]] || \
  fail "refusing restarted canonical plan attempt"
[[ ${SLURM_CPUS_PER_TASK:-} = 4 ]] || fail "allocation must expose exactly four CPUs"
[[ -z ${SLURM_JOB_GPUS:-} && -z ${SLURM_GPUS:-} ]] || \
  fail "canonical plan must not receive a GPU allocation"

JOB_RECORD=$(scontrol show job -o "$JOB_ID")
[[ " $JOB_RECORD " == *" Requeue=0 "* ]] || \
  fail "live Slurm policy does not enforce no-requeue"
for spec in "StdOut=carry_motor_plan_${JOB_ID}_${RESTART_COUNT}.out" \
            "StdErr=carry_motor_plan_${JOB_ID}_${RESTART_COUNT}.err"; do
  key=${spec%%=*}
  suffix=${spec#*=}
  observed=$(sed -n "s/.* ${key}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")
  [[ "$observed" = *"$suffix" ]] || \
    fail "live Slurm $key path is not restart-distinguishing: $observed"
done

[[ ! -e "$PLAN_ROOT" && ! -L "$PLAN_ROOT" ]] || \
  fail "refusing existing canonical plan root: $PLAN_ROOT"
[[ -x "$PY" ]] || fail "missing DATA_ROOT Python: $PY"
git -C "$SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "SOURCE_ROOT is not a Git checkout"
[[ $(git -C "$SOURCE_ROOT" rev-parse HEAD) = "$SOURCE_COMMIT" ]] || \
  fail "reviewed source commit is not checked out"
[[ -z $(git -C "$SOURCE_ROOT" status --porcelain) ]] || \
  fail "SOURCE_ROOT is not clean"

for path in "$CKPT" "$TOKENIZER" "$EPISODES" "$CYCLE"; do
  [[ -f "$path" && ! -L "$path" ]] || \
    fail "frozen input is not a regular non-symlink file: $path"
done

CONFIRMATION_ROOT=${CONFIRMATION_COMMITMENT%/commitment.json}
[[ -d "$CONFIRMATION_ROOT" && ! -L "$CONFIRMATION_ROOT" && \
   $(stat -c '%a' "$CONFIRMATION_ROOT") = 555 ]] || \
  fail "confirmation commitment root is not a non-symlink mode-0555 directory"
[[ $(find "$CONFIRMATION_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n') = commitment.json ]] || \
  fail "confirmation commitment root is not one-file closed-world"
[[ -f "$CONFIRMATION_COMMITMENT" && ! -L "$CONFIRMATION_COMMITMENT" && \
   $(stat -c '%F' "$CONFIRMATION_COMMITMENT") = "regular file" && \
   $(stat -c '%a' "$CONFIRMATION_COMMITMENT") = 444 && \
   $(stat -c '%h' "$CONFIRMATION_COMMITMENT") = 1 ]] || \
  fail "confirmation commitment is not regular mode-0444 with one link"
[[ $(sha256sum "$CONFIRMATION_COMMITMENT" | awk '{print $1}') = \
   "$CONFIRMATION_COMMITMENT_SHA256" ]] || \
  fail "confirmation commitment SHA-256 mismatch"

for relative in "${SCIENTIFIC_SOURCES[@]}"; do
  path=$SOURCE_ROOT/$relative
  [[ -f "$path" && ! -L "$path" ]] || \
    fail "scientific source is not a regular non-symlink file: $relative"
  git -C "$SOURCE_ROOT" ls-files --error-unmatch "$relative" >/dev/null 2>&1 || \
    fail "scientific source is not tracked: $relative"
  git -C "$SOURCE_ROOT" cat-file -e "$SOURCE_COMMIT:$relative" 2>/dev/null || \
    fail "scientific source is absent from reviewed commit: $relative"
done

# Slurm executes a private spool copy. Bind those executed bytes to the reviewed
# wrapper instead of trusting only the checkout copy used in the manifest.
[[ -f "$0" && ! -L "$0" ]] || fail "spooled wrapper is not a regular file"
cmp --silent -- "$0" \
  "$SOURCE_ROOT/pipeline/jobs/causal_carry_motor_plan_stokes.sbatch" || \
  fail "spooled wrapper differs from reviewed checkout"

cd "$SOURCE_ROOT"
SOURCE_MANIFEST_SHA256=$("$PY" - "$SOURCE_COMMIT" "${SCIENTIFIC_SOURCES[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

commit, *paths = sys.argv[1:]
sources = {}
for relative in paths:
    working = Path(relative).read_bytes()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], stderr=subprocess.DEVNULL
    )
    if working != committed:
        raise SystemExit(f"scientific source differs from reviewed commit: {relative}")
    sources[relative] = hashlib.sha256(working).hexdigest()
payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
)
[[ "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  fail "failed to derive canonical scientific source manifest"

# The Python plan command performs the authoritative SHA-256 checks for all
# four frozen inputs plus the commitment and atomically publishes the plan root.
export CUDA_VISIBLE_DEVICES=
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/train
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$(dirname "$PLAN_ROOT")"
"$PY" train/causal_carry_motor.py plan \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --episodes "$EPISODES" \
  --cycle "$CYCLE" \
  --source-commit "$SOURCE_COMMIT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
  --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256" \
  --plan-root "$PLAN_ROOT"

[[ -d "$PLAN_ROOT" && ! -L "$PLAN_ROOT" ]] || \
  fail "canonical plan command did not publish the expected plan root"
[[ $(stat -c '%a' "$PLAN_ROOT") = 555 ]] || \
  fail "canonical plan root is not sealed mode 0555"
[[ -f "$PLAN_ROOT/plan.json" && ! -L "$PLAN_ROOT/plan.json" ]] || \
  fail "canonical plan file is missing or linked"
[[ $(stat -c '%F' "$PLAN_ROOT/plan.json") = "regular file" && \
   $(stat -c '%a' "$PLAN_ROOT/plan.json") = 444 && \
   $(stat -c '%h' "$PLAN_ROOT/plan.json") = 1 ]] || \
  fail "canonical plan file is not regular mode-0444 with one link"
for directory in "$PLAN_ROOT"/shard_{00..07} "$PLAN_ROOT/fit" \
                 "$PLAN_ROOT/development_eval" "$PLAN_ROOT/confirmation_eval"; do
  [[ -d "$directory" && ! -L "$directory" ]] || \
    fail "planned output directory is missing or linked: $directory"
  [[ $(stat -c '%a' "$directory") = 700 ]] || \
    fail "planned output directory is not mode 0700: $directory"
  [[ -z $(find "$directory" -mindepth 1 -maxdepth 1 -print -quit) ]] || \
    fail "planned output directory is not empty: $directory"
done
