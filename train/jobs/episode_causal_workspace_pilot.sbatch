#!/bin/bash
#SBATCH --job-name=shohin-episode-ws
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/episode_causal_workspace_%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE

# Isolated synthetic reasoning pilot. It never invokes train.py, never opens
# pretraining shards, and never writes inside train/flagship_out.
set -euo pipefail
umask 077

BASE=/lustre/fs1/home/sa305415/shohin
PY="$BASE/miniforge3/bin/python"
CHECKPOINT="$BASE/train/flagship_out/ckpt_0300000.pt"
CUSTODY="$BASE/artifacts/r12/episode_workspace_custody_v1_1532fe2_seed2026072309"
SEED=${SEED:-2026072347}
UPDATES=${UPDATES:-800}
GROUPS_PER_BATCH=${GROUPS_PER_BATCH:-8}
COMMIT_FULL=$(git -C "$BASE" rev-parse HEAD)
COMMIT=${COMMIT_FULL:0:12}
FINAL=${OUTPUT_ROOT:-"$BASE/train/episode_causal_workspace_${COMMIT}_seed${SEED}"}
JOB_TAG=${SLURM_JOB_ID:-manual}-$$
HIDDEN="$BASE/train/.episode_causal_workspace_${COMMIT}_seed${SEED}.${JOB_TAG}.staging"
RUNTIME="$BASE/train/.episode_causal_workspace_${COMMIT}_seed${SEED}.${JOB_TAG}.runtime"

case "$FINAL" in
  "$BASE"/train/episode_causal_workspace_*) ;;
  *)
    echo "refusing non-isolated output root: $FINAL" >&2
    exit 2
    ;;
esac
if [ -e "$FINAL" ] || [ -e "$HIDDEN" ] || [ -e "$RUNTIME" ]; then
  echo "refusing to replace an experiment path" >&2
  exit 2
fi

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

EXPECTED_CHECKPOINT=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
if [ "$(sha_file "$CHECKPOINT")" != "$EXPECTED_CHECKPOINT" ]; then
  echo "protected checkpoint hash mismatch" >&2
  exit 2
fi

TRUE_GROUPS="$CUSTODY/train_true_groups.jsonl"
SHUFFLED_GROUPS="$CUSTODY/train_shuffled_groups.jsonl"
WORLDS="$CUSTODY/development_worlds.jsonl"
QUERIES="$CUSTODY/development_queries.jsonl"
ASSESSOR="$CUSTODY/development_assessor.jsonl"
TRUE_GROUPS_SHA=80d7e6e503d4aebbda506fcb3d321f1a91db556f7fe1bd2ab8e6ee92d2fbec27
SHUFFLED_GROUPS_SHA=5917049c910cdc2beae667165465052c71329033be30532a4cec5a04fe419038
WORLDS_SHA=e448d0585427c7314b3f21c567b231b6e8ebe08dbd4acf4e103151a3af56851e
QUERIES_SHA=456b9bcef3762a9922323a5928078a8998f51c8d6c9d358e5f6adae69aa1bfba
ASSESSOR_SHA=0bf99717d677b5807c47873cf365fc408230e2eef16218e9f768bb9489294235

for pair in \
  "$TRUE_GROUPS:$TRUE_GROUPS_SHA" \
  "$SHUFFLED_GROUPS:$SHUFFLED_GROUPS_SHA" \
  "$WORLDS:$WORLDS_SHA" \
  "$QUERIES:$QUERIES_SHA" \
  "$ASSESSOR:$ASSESSOR_SHA"; do
  path=${pair%%:*}
  expected=${pair##*:}
  if [ "$(sha_file "$path")" != "$expected" ]; then
    echo "custody input hash mismatch: $path" >&2
    exit 2
  fi
done

SOURCE_STAGER="$BASE/pipeline/stage_episode_workspace_sources.py"
PUBLISHER_SOURCE="$BASE/pipeline/publish_episode_workspace_experiment.py"
JOB_SOURCE="$BASE/train/jobs/episode_causal_workspace_pilot.sbatch"
git -C "$BASE" ls-files --error-unmatch \
  "${SOURCE_STAGER#"$BASE"/}" \
  "${PUBLISHER_SOURCE#"$BASE"/}" \
  "${JOB_SOURCE#"$BASE"/}" >/dev/null
git -C "$BASE" diff --quiet HEAD -- \
  "${SOURCE_STAGER#"$BASE"/}" \
  "${PUBLISHER_SOURCE#"$BASE"/}" \
  "${JOB_SOURCE#"$BASE"/}"

mkdir "$HIDDEN" "$RUNTIME"
trap 'rm -rf "$RUNTIME"' EXIT

export PYTHONPATH="$BASE/train:$BASE"
"$PY" "$SOURCE_STAGER" \
  --expected-commit "$COMMIT_FULL" \
  --output "$HIDDEN/source"

SOURCE_ROOT="$HIDDEN/source"
SOURCE_RECEIPT="$SOURCE_ROOT/source_receipt.json"
SOURCE_RECEIPT_SHA=$(sha_file "$SOURCE_RECEIPT")
LANDLOCK="$SOURCE_ROOT/train/landlock_stage_exec.py"
FIT_SOURCE="$SOURCE_ROOT/train/fit_episode_causal_workspace.py"
COMPILE_SOURCE="$SOURCE_ROOT/train/compile_episode_causal_workspace.py"
EXECUTE_SOURCE="$SOURCE_ROOT/train/execute_episode_causal_workspace.py"
ASSESS_SOURCE="$SOURCE_ROOT/pipeline/assess_episode_causal_workspace.py"
DECIDE_SOURCE="$SOURCE_ROOT/pipeline/decide_episode_causal_workspace.py"
PUBLISHER="$SOURCE_ROOT/pipeline/publish_episode_workspace_experiment.py"
COMPILER_SOURCE_SHA=$(sha_file "$COMPILE_SOURCE")

export PYTHONPATH="$SOURCE_ROOT/train:$SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SHOHIN_SOURCE_RECEIPT_PATH="$SOURCE_RECEIPT"
export SHOHIN_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA"
export SHOHIN_REPOSITORY_COMMIT="$COMMIT_FULL"
unset SSH_AUTH_SOCK LD_PRELOAD

TRUSTED_LD_LIBRARY_PATH=""
for path in \
  "$BASE/miniforge3/lib" \
  /usr/local/cuda/lib64 \
  /usr/local/cuda/compat \
  /usr/lib64 \
  /usr/lib/x86_64-linux-gnu; do
  if [ -d "$path" ]; then
    if [ -n "$TRUSTED_LD_LIBRARY_PATH" ]; then
      TRUSTED_LD_LIBRARY_PATH+=":"
    fi
    TRUSTED_LD_LIBRARY_PATH+="$path"
  fi
done

common_landlock_args() {
  launcher=(
    "$PY" "$LANDLOCK"
    --ro "$SOURCE_ROOT"
    --ro "$BASE/miniforge3"
    --ro "$CHECKPOINT"
  )
  local path
  for path in \
    /usr \
    /opt \
    /lib \
    /lib64 \
    /etc/ld.so.cache \
    /etc/localtime \
    /sys \
    /proc/self \
    /proc/cpuinfo \
    /proc/meminfo \
    /proc/driver/nvidia \
    /dev/urandom \
    /dev/zero; do
    if [ -e "$path" ]; then
      launcher+=(--ro "$path")
    fi
  done
  if [ -e /dev/null ]; then
    launcher+=(--rw /dev/null)
  fi
  for path in /dev/nvidia*; do
    if [ -e "$path" ]; then
      launcher+=(--rw "$path")
    fi
  done
}

run_confined() {
  local stage=$1
  local deny_probe=$2
  local slot=$3
  shift 3
  local -a launcher
  common_landlock_args
  launcher+=(--stage "$stage" --rw "$slot")
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    launcher+=(--ro "$1")
    shift
  done
  if [ "$#" -eq 0 ]; then
    echo "confined command separator is missing" >&2
    exit 2
  fi
  shift
  local stage_runtime="$RUNTIME/$stage-$(basename "$slot")"
  mkdir -p "$stage_runtime/home" "$stage_runtime/tmp" "$stage_runtime/cache"
  launcher+=(
    --rw "$stage_runtime"
    --policy-receipt "$stage_runtime/landlock_policy.json"
  )
  env -i \
    PATH="$BASE/miniforge3/bin:/usr/bin:/bin" \
    LANG=C.UTF-8 \
    HOME="$stage_runtime/home" \
    TMPDIR="$stage_runtime/tmp" \
    XDG_CACHE_HOME="$stage_runtime/cache" \
    CUDA_CACHE_PATH="$stage_runtime/cache" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    LD_LIBRARY_PATH="$TRUSTED_LD_LIBRARY_PATH" \
    PYTHONPATH="$SOURCE_ROOT/train:$SOURCE_ROOT" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SHOHIN_SOURCE_RECEIPT_PATH="$SOURCE_RECEIPT" \
    SHOHIN_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA" \
    SHOHIN_REPOSITORY_COMMIT="$COMMIT_FULL" \
    "${launcher[@]}" -- "$@" --deny-probe "$deny_probe"
}

publish_stage() {
  local slot=$1
  local name=$2
  if [ ! -d "$slot/result" ] || [ -e "$HIDDEN/$name" ]; then
    echo "stage publication invariant failed: $name" >&2
    exit 2
  fi
  mv "$slot/result" "$HIDDEN/$name"
}

run_arm() {
  local arm=$1
  local groups=$2
  local groups_sha=$3
  local fit_name="${arm}_fit"
  local compiled_name="${arm}_compiled"
  local execution_name="${arm}_execution"
  local assessment_name="${arm}_assessment"
  local fit_slot="$RUNTIME/slots/$fit_name"
  local compiled_slot="$RUNTIME/slots/$compiled_name"
  local execution_slot="$RUNTIME/slots/$execution_name"
  local assessment_slot="$RUNTIME/slots/$assessment_name"
  mkdir -p "$fit_slot" "$compiled_slot" "$execution_slot" "$assessment_slot"

  run_confined fit "$ASSESSOR" "$fit_slot" "$groups" -- \
    "$PY" "$FIT_SOURCE" \
      --train-groups "$groups" \
      --expected-train-sha256 "$groups_sha" \
      --checkpoint "$CHECKPOINT" \
      --output "$fit_slot/result" \
      --expected-source-sha256 "$(sha_file "$FIT_SOURCE")" \
      --arm "$arm" \
      --device cuda \
      --updates "$UPDATES" \
      --groups-per-batch "$GROUPS_PER_BATCH" \
      --seed "$SEED"
  publish_stage "$fit_slot" "$fit_name"

  local delta="$HIDDEN/$fit_name/workspace_delta.pt"
  local delta_sha
  delta_sha=$(sha_file "$delta")
  run_confined compiler "$QUERIES" "$compiled_slot" "$WORLDS" "$delta" -- \
    "$PY" "$COMPILE_SOURCE" \
      --worlds "$WORLDS" \
      --expected-worlds-sha256 "$WORLDS_SHA" \
      --workspace-delta "$delta" \
      --expected-delta-sha256 "$delta_sha" \
      --checkpoint "$CHECKPOINT" \
      --output "$compiled_slot/result" \
      --expected-source-sha256 "$COMPILER_SOURCE_SHA" \
      --device cuda
  publish_stage "$compiled_slot" "$compiled_name"

  local states="$HIDDEN/$compiled_name/compiled_states.pt"
  local states_sha
  states_sha=$(sha_file "$states")
  run_confined executor "$WORLDS" "$execution_slot" \
    "$QUERIES" "$states" "$delta" -- \
    "$PY" "$EXECUTE_SOURCE" \
      --queries "$QUERIES" \
      --expected-queries-sha256 "$QUERIES_SHA" \
      --compiled-states "$states" \
      --expected-states-sha256 "$states_sha" \
      --expected-worlds-sha256 "$WORLDS_SHA" \
      --workspace-delta "$delta" \
      --expected-delta-sha256 "$delta_sha" \
      --checkpoint "$CHECKPOINT" \
      --output "$execution_slot/result" \
      --expected-source-sha256 "$(sha_file "$EXECUTE_SOURCE")" \
      --expected-compiler-source-sha256 "$COMPILER_SOURCE_SHA" \
      --expected-repository-commit "$COMMIT_FULL" \
      --device cuda
  publish_stage "$execution_slot" "$execution_name"

  local execution="$HIDDEN/$execution_name"
  local execution_manifest_sha
  execution_manifest_sha=$(sha_file "$execution/bundle_manifest.json")
  local -a execution_files=(
    "$execution/bundle_manifest.json"
    "$execution/label_blind_predictions.jsonl"
    "$execution/execution_report.json"
  )
  local control
  for control in \
    treatment \
    zero_workspace \
    uniform_binding \
    uniform_operator \
    binding_permutation \
    operator_permutation \
    selected_slot_scramble \
    discarded_slot_scramble; do
    execution_files+=("$execution/answer_logits_${control}.pt")
  done
  run_confined assessor "$WORLDS" "$assessment_slot" \
    "$ASSESSOR" "${execution_files[@]}" -- \
    "$PY" "$ASSESS_SOURCE" \
      --assessor "$ASSESSOR" \
      --expected-assessor-sha256 "$ASSESSOR_SHA" \
      --execution-bundle "$execution" \
      --expected-execution-manifest-sha256 "$execution_manifest_sha" \
      --output "$assessment_slot/result" \
      --expected-source-sha256 "$(sha_file "$ASSESS_SOURCE")"
  publish_stage "$assessment_slot" "$assessment_name"
}

echo "event=episode_workspace_start job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-none} commit=$COMMIT_FULL"
nvidia-smi -L
run_arm true "$TRUE_GROUPS" "$TRUE_GROUPS_SHA"
run_arm shuffled "$SHUFFLED_GROUPS" "$SHUFFLED_GROUPS_SHA"

TRUE_FIT="$HIDDEN/true_fit/fit_report.json"
SHUFFLED_FIT="$HIDDEN/shuffled_fit/fit_report.json"
TRUE_ASSESSMENT="$HIDDEN/true_assessment/assessment_report.json"
SHUFFLED_ASSESSMENT="$HIDDEN/shuffled_assessment/assessment_report.json"
DECISION_SLOT="$RUNTIME/slots/decision"
mkdir -p "$DECISION_SLOT"
run_confined decision "$ASSESSOR" "$DECISION_SLOT" \
  "$TRUE_FIT" "$SHUFFLED_FIT" "$TRUE_ASSESSMENT" "$SHUFFLED_ASSESSMENT" -- \
  "$PY" "$DECIDE_SOURCE" \
    --true-fit "$TRUE_FIT" \
    --true-fit-sha256 "$(sha_file "$TRUE_FIT")" \
    --shuffled-fit "$SHUFFLED_FIT" \
    --shuffled-fit-sha256 "$(sha_file "$SHUFFLED_FIT")" \
    --true-assessment "$TRUE_ASSESSMENT" \
    --true-assessment-sha256 "$(sha_file "$TRUE_ASSESSMENT")" \
    --shuffled-assessment "$SHUFFLED_ASSESSMENT" \
    --shuffled-assessment-sha256 "$(sha_file "$SHUFFLED_ASSESSMENT")" \
    --expected-source-sha256 "$(sha_file "$DECIDE_SOURCE")" \
    --output "$DECISION_SLOT/result"
publish_stage "$DECISION_SLOT" decision

DECISION_SHA=$(sha_file "$HIDDEN/decision/decision.json")
export PYTHONPATH="$SOURCE_ROOT/train:$SOURCE_ROOT"
"$PY" "$PUBLISHER" \
  --staging "$HIDDEN" \
  --output "$FINAL" \
  --expected-commit "$COMMIT_FULL" \
  --expected-source-receipt-sha256 "$SOURCE_RECEIPT_SHA" \
  --expected-decision-sha256 "$DECISION_SHA" \
  --slurm-job-id "${SLURM_JOB_ID:-none}" \
  --slurm-node "${SLURMD_NODENAME:-unknown}"

echo "event=episode_workspace_complete output=$FINAL"
