#!/bin/bash
#SBATCH --job-name=carry-recovery
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=evc22,evc26,evc31,evc32,evc36,evc37,evc40,evc43,evc44
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/carry_recovery_%j_%r.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/carry_recovery_%j_%r.err

set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export LANG=C
export LC_ALL=C

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to the clean reviewed recovery checkout}
RECOVERY_SOURCE_COMMIT=${RECOVERY_SOURCE_COMMIT:?set reviewed recovery commit}
RECOVERY_SOURCE_MANIFEST_SHA256=${RECOVERY_SOURCE_MANIFEST_SHA256:?set reviewed recovery source manifest}
HOSTILE_REVIEW=${HOSTILE_REVIEW:?set sealed hostile-review receipt}
HOSTILE_REVIEW_SHA256=${HOSTILE_REVIEW_SHA256:?set hostile-review SHA-256}
RECOVERY_PLAN_SHA256=${RECOVERY_PLAN_SHA256:?set sealed recovery-plan SHA-256}

readonly DATA_ROOT=/lustre/fs1/home/sa305415/shohin
readonly PY=$DATA_ROOT/miniforge3/bin/python
readonly GIT=/usr/bin/git
readonly UPSTREAM_COMMIT=a0c258e6709766c643cf127a429a7d6ef4a4211b
readonly UPSTREAM_PLAN_SHA256=1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d
readonly UPSTREAM_ROOT=$DATA_ROOT/artifacts/carry_motor/canonical_$UPSTREAM_COMMIT
readonly RECOVERY_ROOT=$DATA_ROOT/artifacts/carry_motor/recoveries/upstream_${UPSTREAM_PLAN_SHA256}_executor_${RECOVERY_SOURCE_COMMIT}
readonly RECOVERY_PLAN=$RECOVERY_ROOT/recovery_plan.json
readonly EXPECTED_REVIEW=$DATA_ROOT/artifacts/carry_motor/recovery_reviews/review_${RECOVERY_SOURCE_COMMIT}/hostile_review.json
RECOVERY_FILES=(
  R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md
  train/causal_carry_motor_recovery.py
  train/test_causal_carry_motor_recovery.py
  train/jobs/causal_carry_motor_recovery.sbatch
)
UPSTREAM_FILES=(
  R12_CAUSAL_CARRY_MOTOR_PREREG.md
  train/causal_carry_motor.py
  train/test_causal_carry_motor.py
  train/jobs/causal_carry_motor.sbatch
  pipeline/jobs/causal_carry_motor_plan_stokes.sbatch
  train/digitwise_controller.py
  train/digitwise_protocol.py
  train/eval_suite.py
  train/model.py
  train/probe_digitwise_workspace.py
)

for value in "$SOURCE_ROOT" "$PY" "$GIT" "$HOSTILE_REVIEW" "$RECOVERY_ROOT" "$RECOVERY_PLAN"; do
  if [[ "$value" != /* ]]; then
    echo "recovery paths must be absolute: $value" >&2
    exit 2
  fi
done
if [[ ! -x "$PY" || ! -f "$GIT" || -L "$GIT" ]]; then
  echo "pinned recovery interpreter or git executable is unavailable" >&2
  exit 2
fi
if [[ "$(realpath -e -- "$SOURCE_ROOT")" != "$SOURCE_ROOT" ]]; then
  echo "recovery source root must be an exact physical path" >&2
  exit 2
fi
if [[ ! "$RECOVERY_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ || \
      "$RECOVERY_SOURCE_COMMIT" == "$UPSTREAM_COMMIT" ]]; then
  echo "recovery source commit is invalid or aliases upstream" >&2
  exit 2
fi
if [[ ! "$RECOVERY_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ || \
      ! "$HOSTILE_REVIEW_SHA256" =~ ^[0-9a-f]{64}$ || \
      ! "$RECOVERY_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "recovery receipts must be lowercase SHA-256" >&2
  exit 2
fi
if [[ "$HOSTILE_REVIEW" != "$EXPECTED_REVIEW" ]]; then
  echo "hostile-review path differs from commit-derived custody" >&2
  exit 2
fi
if [[ -n "${CONFIRMATION_SECRET_FILE:-}" ]]; then
  echo "confirmation secret is forbidden in recovery fit" >&2
  exit 2
fi
for name in \
  LD_PRELOAD \
  PYTHONHOME \
  PYTHONINSPECT \
  PYTHONSAFEPATH \
  PYTHONSTARTUP \
  PYTHONUSERBASE \
  PYTHONWARNINGS \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD; do
  if [[ -n "${!name:-}" ]]; then
    echo "ambient torch deserialization override is forbidden: $name" >&2
    exit 2
  fi
done

if [[ "$("$GIT" -C "$SOURCE_ROOT" rev-parse HEAD)" != "$RECOVERY_SOURCE_COMMIT" ]]; then
  echo "reviewed recovery commit is not checked out" >&2
  exit 2
fi
read -r -a LINEAGE <<< "$("$GIT" -C "$SOURCE_ROOT" rev-list --parents -n 1 "$RECOVERY_SOURCE_COMMIT")"
if [[ ${#LINEAGE[@]} -ne 2 || \
      "${LINEAGE[0]}" != "$RECOVERY_SOURCE_COMMIT" || \
      "${LINEAGE[1]}" != "$UPSTREAM_COMMIT" ]]; then
  echo "recovery commit must have a0c258e as its sole direct parent" >&2
  exit 2
fi
mapfile -t NAME_STATUS < <(
  "$GIT" -C "$SOURCE_ROOT" diff --name-status --no-renames \
    "$UPSTREAM_COMMIT" "$RECOVERY_SOURCE_COMMIT"
)
if [[ ${#NAME_STATUS[@]} -ne 4 || \
      "${NAME_STATUS[0]}" != $'A\tR12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md' || \
      "${NAME_STATUS[1]}" != $'A\ttrain/causal_carry_motor_recovery.py' || \
      "${NAME_STATUS[2]}" != $'A\ttrain/jobs/causal_carry_motor_recovery.sbatch' || \
      "${NAME_STATUS[3]}" != $'A\ttrain/test_causal_carry_motor_recovery.py' ]]; then
  echo "recovery diff must be exactly four added recovery files" >&2
  exit 2
fi
if [[ -n "$("$GIT" -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "recovery source checkout is not clean" >&2
  exit 2
fi
mapfile -d '' -t TRACKED_FILES < <("$GIT" -C "$SOURCE_ROOT" ls-files -z)
mapfile -d '' -t OBSERVED_FILES < <(
  find "$SOURCE_ROOT" -path "$SOURCE_ROOT/.git" -prune -o \
    \( -type f -o -type l \) -printf '%P\0' | sort -z
)
if [[ ${#TRACKED_FILES[@]} -ne ${#OBSERVED_FILES[@]} ]]; then
  echo "recovery source checkout contains an extra or missing filesystem leaf" >&2
  exit 2
fi
for index in "${!TRACKED_FILES[@]}"; do
  if [[ "${TRACKED_FILES[$index]}" != "${OBSERVED_FILES[$index]}" ]]; then
    echo "recovery source checkout is not closed-world" >&2
    exit 2
  fi
done
for path in "${RECOVERY_FILES[@]}"; do
  TREE_ENTRY=$("$GIT" -C "$SOURCE_ROOT" ls-tree "$RECOVERY_SOURCE_COMMIT" -- "$path")
  TREE_MODE=${TREE_ENTRY%% *}
  TREE_PATH=${TREE_ENTRY#*$'\t'}
  if [[ "$TREE_MODE" != 100644 || "$TREE_PATH" != "$path" || \
        ! -f "$SOURCE_ROOT/$path" || -L "$SOURCE_ROOT/$path" || \
        "$(stat -c '%a' "$SOURCE_ROOT/$path")" != 644 || \
        "$(stat -c '%h' "$SOURCE_ROOT/$path")" != 1 ]]; then
    echo "recovery source file identity mismatch: $path" >&2
    exit 2
  fi
done
for path in "${UPSTREAM_FILES[@]}"; do
  working_sha=$(sha256sum "$SOURCE_ROOT/$path" | awk '{print $1}')
  committed_sha=$("$GIT" -C "$SOURCE_ROOT" show "$UPSTREAM_COMMIT:$path" | sha256sum | awk '{print $1}')
  if [[ "$working_sha" != "$committed_sha" ]]; then
    echo "loaded upstream dependency differs from a0c258e: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$0" || -L "$0" ]] || ! cmp --silent -- "$0" \
  "$SOURCE_ROOT/train/jobs/causal_carry_motor_recovery.sbatch"; then
  echo "spooled recovery wrapper differs from reviewed checkout" >&2
  exit 2
fi

export PYTHONPATH="$SOURCE_ROOT/train"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SOURCE_ROOT"
OBSERVED_MANIFEST=$(
  "$PY" - "$GIT" "$RECOVERY_SOURCE_COMMIT" "${RECOVERY_FILES[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

git, commit, *paths = sys.argv[1:]
sources = {}
for name in paths:
    working = Path(name).read_bytes()
    committed = subprocess.check_output([git, "show", f"{commit}:{name}"])
    if working != committed:
        raise SystemExit(f"recovery source differs from reviewed commit: {name}")
    sources[name] = hashlib.sha256(working).hexdigest()
payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
)
if [[ "$OBSERVED_MANIFEST" != "$RECOVERY_SOURCE_MANIFEST_SHA256" ]]; then
  echo "recovery source manifest mismatch" >&2
  exit 2
fi

for pair in \
  "$HOSTILE_REVIEW:$HOSTILE_REVIEW_SHA256:444" \
  "$RECOVERY_PLAN:$RECOVERY_PLAN_SHA256:444"; do
  path=${pair%%:*}
  remainder=${pair#*:}
  receipt=${remainder%%:*}
  mode=${remainder##*:}
  if [[ ! -f "$path" || -L "$path" || \
        "$(stat -c '%F' "$path")" != "regular file" || \
        "$(stat -c '%a' "$path")" != "$mode" || \
        "$(stat -c '%h' "$path")" != 1 || \
        "$(sha256sum "$path" | awk '{print $1}')" != "$receipt" ]]; then
    echo "sealed recovery input failed: $path" >&2
    exit 2
  fi
  if [[ "$(stat -c '%a' "${path%/*}")" != 555 ]]; then
    echo "sealed recovery input parent mode failed: $path" >&2
    exit 2
  fi
done

if [[ ! -d "$UPSTREAM_ROOT" || -L "$UPSTREAM_ROOT" || \
      "$(stat -c '%a' "$UPSTREAM_ROOT")" != 555 ]]; then
  echo "upstream canonical root is not immutable" >&2
  exit 2
fi
if [[ ! -d "$RECOVERY_ROOT" || -L "$RECOVERY_ROOT" || \
      "$(stat -c '%a' "$RECOVERY_ROOT")" != 555 ]]; then
  echo "recovery root is not immutable" >&2
  exit 2
fi
FIT_MODE=$(stat -c '%a' "$RECOVERY_ROOT/fit" 2>/dev/null || true)
FIT_CHILDREN=$(find "$RECOVERY_ROOT/fit" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null || true)
if [[ ! -d "$RECOVERY_ROOT/fit" || -L "$RECOVERY_ROOT/fit" || \
      ( "$FIT_MODE" != 700 && "$FIT_MODE" != 555 ) || \
      ( -n "$FIT_CHILDREN" && "$FIT_CHILDREN" != motor.pt ) || \
      ( "$FIT_MODE" == 555 && "$FIT_CHILDREN" != motor.pt ) ]]; then
  echo "recovery fit output is not empty, recoverable, or sealed" >&2
  exit 2
fi
if [[ "$FIT_CHILDREN" == motor.pt ]]; then
  FIT_ARTIFACT_MODE=$(stat -c '%a' "$RECOVERY_ROOT/fit/motor.pt" 2>/dev/null || true)
  if [[ ! -f "$RECOVERY_ROOT/fit/motor.pt" || -L "$RECOVERY_ROOT/fit/motor.pt" || \
        "$(stat -c '%h' "$RECOVERY_ROOT/fit/motor.pt")" != 1 || \
        ( "$FIT_MODE" == 700 && "$FIT_ARTIFACT_MODE" != 600 && "$FIT_ARTIFACT_MODE" != 444 ) || \
        ( "$FIT_MODE" == 555 && "$FIT_ARTIFACT_MODE" != 444 ) ]]; then
    echo "existing recovery fit artifact is not interrupted, recoverable, or sealed" >&2
    exit 2
  fi
fi

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if [[ "$RESTART_COUNT" != 0 ]]; then
  echo "refusing restarted recovery fit" >&2
  exit 2
fi
JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
if [[ "$JOB_RECORD" != *" Requeue=0 "* ]]; then
  echo "recovery fit requires live Requeue=0" >&2
  exit 2
fi
for spec in "StdOut=carry_recovery_${SLURM_JOB_ID}_${RESTART_COUNT}.out" \
            "StdErr=carry_recovery_${SLURM_JOB_ID}_${RESTART_COUNT}.err"; do
  key=${spec%%=*}
  suffix=${spec#*=}
  observed=$(sed -n "s/.* ${key}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")
  if [[ "$observed" != *"$suffix" ]]; then
    echo "recovery $key is not restart-distinguishing: $observed" >&2
    exit 2
  fi
done

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" != 1 || \
      "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')" != \
      "NVIDIA H100 PCIe" ]]; then
  echo "recovery fit requires exactly one NVIDIA H100 PCIe" >&2
  exit 75
fi

"$PY" train/causal_carry_motor_recovery.py fit \
  --recovery-source-commit "$RECOVERY_SOURCE_COMMIT" \
  --recovery-source-manifest-sha256 "$RECOVERY_SOURCE_MANIFEST_SHA256" \
  --hostile-review "$HOSTILE_REVIEW" \
  --hostile-review-sha256 "$HOSTILE_REVIEW_SHA256" \
  --recovery-root "$RECOVERY_ROOT" \
  --recovery-plan "$RECOVERY_PLAN" \
  --recovery-plan-sha256 "$RECOVERY_PLAN_SHA256"
