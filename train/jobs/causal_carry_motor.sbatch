#!/bin/bash
#SBATCH --job-name=carry-motor
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
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/carry_motor_%j_%r.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/carry_motor_%j_%r.err

set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to an isolated clean Git checkout}
CANONICAL_DATA_ROOT=/lustre/fs1/home/sa305415/shohin
DATA_ROOT=${DATA_ROOT:-$CANONICAL_DATA_ROOT}
PY=${PY:-$DATA_ROOT/miniforge3/bin/python}
MODE=${MODE:-extract}
CKPT=${CKPT:-$DATA_ROOT/train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt}
TOKENIZER=${TOKENIZER:-$DATA_ROOT/artifacts/shohin-tok-32k.json}
EPISODES=${EPISODES:-$DATA_ROOT/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
CYCLE=${CYCLE:-$DATA_ROOT/artifacts/evals/drs_causal_cycle_post_drs_r3.json}
CONFIRMATION_COMMITMENT=${CONFIRMATION_COMMITMENT:?set the sealed pre-fit confirmation commitment}
CONFIRMATION_COMMITMENT_SHA256=${CONFIRMATION_COMMITMENT_SHA256:?set the reviewed confirmation commitment digest}
CONFIRMATION_SECRET_FILE=${CONFIRMATION_SECRET_FILE:-}
PLAN=${PLAN:?set PLAN to the frozen canonical plan}
PLAN_SHA256=${PLAN_SHA256:?set PLAN_SHA256 to the reviewed plan digest}
SHARD_INDEX=${SHARD_INDEX:-}
SHARD_COUNT=${SHARD_COUNT:-8}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed clean commit}
EXPECTED_PLAN=$CANONICAL_DATA_ROOT/artifacts/carry_motor/canonical_${SOURCE_COMMIT}/plan.json
EXPECTED_CONFIRMATION_COMMITMENT=$CANONICAL_DATA_ROOT/artifacts/carry_motor/confirmation_commitments/commitment_${SOURCE_COMMIT}/commitment.json
EXPECTED_PLAN_ROOT=${EXPECTED_PLAN%/plan.json}
PLAN_ROOT=${PLAN%/plan.json}
SOURCE_FILES=(
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

for path in "$SOURCE_ROOT" "$DATA_ROOT" "$CKPT" "$TOKENIZER" "$EPISODES" "$CYCLE" \
            "$CONFIRMATION_COMMITMENT" "$PLAN"; do
  if [[ "$path" != /* ]]; then
    echo "canonical paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ "$MODE" == confirmation-eval ]]; then
  if [[ "$CONFIRMATION_SECRET_FILE" != /* || ! -f "$CONFIRMATION_SECRET_FILE" || \
        -L "$CONFIRMATION_SECRET_FILE" || \
        "$(stat -c '%F' "$CONFIRMATION_SECRET_FILE")" != "regular file" || \
        "$(stat -c '%a' "$CONFIRMATION_SECRET_FILE")" != 400 || \
        "$(stat -c '%h' "$CONFIRMATION_SECRET_FILE")" != 1 || \
        "$(stat -c '%s' "$CONFIRMATION_SECRET_FILE")" != 32 ]]; then
    echo "revealed confirmation secret must be an absolute sealed 32-byte file" >&2
    exit 2
  fi
elif [[ -n "$CONFIRMATION_SECRET_FILE" ]]; then
  echo "revealed confirmation secret is forbidden outside confirmation-eval" >&2
  exit 2
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be a lowercase 40-hex commit" >&2
  exit 2
fi
if [[ "$DATA_ROOT" != "$CANONICAL_DATA_ROOT" || \
      "$CONFIRMATION_COMMITMENT" != "$EXPECTED_CONFIRMATION_COMMITMENT" || \
      "$PLAN" != "$EXPECTED_PLAN" || \
      "$PLAN_ROOT" != "$EXPECTED_PLAN_ROOT" ]]; then
  echo "canonical data, commitment, or plan path differs from commit-bound custody" >&2
  exit 2
fi
CONFIRMATION_ROOT=${CONFIRMATION_COMMITMENT%/commitment.json}
if [[ ! -d "$CONFIRMATION_ROOT" || -L "$CONFIRMATION_ROOT" || \
      "$(stat -c '%a' "$CONFIRMATION_ROOT")" != 555 || \
      "$(find "$CONFIRMATION_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n')" != commitment.json ]]; then
  echo "confirmation commitment root is not sealed one-file custody" >&2
  exit 2
fi
if [[ ! "$CONFIRMATION_COMMITMENT_SHA256" =~ ^[0-9a-f]{64}$ || \
      ! -f "$CONFIRMATION_COMMITMENT" || -L "$CONFIRMATION_COMMITMENT" || \
      "$(stat -c '%F' "$CONFIRMATION_COMMITMENT")" != "regular file" || \
      "$(stat -c '%a' "$CONFIRMATION_COMMITMENT")" != 444 || \
      "$(stat -c '%h' "$CONFIRMATION_COMMITMENT")" != 1 || \
      "$(sha256sum "$CONFIRMATION_COMMITMENT" | awk '{print $1}')" != "$CONFIRMATION_COMMITMENT_SHA256" ]]; then
  echo "confirmation commitment or SHA-256 receipt is invalid" >&2
  exit 2
fi
if [[ ! -d "$PLAN_ROOT" || -L "$PLAN_ROOT" || \
      "$(stat -c '%a' "$PLAN_ROOT")" != 555 ]]; then
  echo "canonical plan root is not a non-symlink mode-0555 directory" >&2
  exit 2
fi
if [[ ! "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ || ! -f "$PLAN" || -L "$PLAN" || \
      "$(stat -c '%F' "$PLAN")" != "regular file" || \
      "$(stat -c '%a' "$PLAN")" != 444 || \
      "$(stat -c '%h' "$PLAN")" != 1 ]]; then
  echo "canonical plan or SHA-256 receipt is invalid" >&2
  exit 2
fi
if [[ "$(sha256sum "$PLAN" | awk '{print $1}')" != "$PLAN_SHA256" ]]; then
  echo "canonical plan SHA-256 mismatch" >&2
  exit 2
fi

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if [[ "$RESTART_COUNT" != 0 ]]; then
  echo "refusing restarted carry-motor job" >&2
  exit 2
fi
JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
if [[ "$JOB_RECORD" != *" Requeue=0 "* ]]; then
  echo "live Slurm policy does not preserve --no-requeue" >&2
  exit 2
fi
for spec in "StdOut=carry_motor_${SLURM_JOB_ID}_${RESTART_COUNT}.out" \
            "StdErr=carry_motor_${SLURM_JOB_ID}_${RESTART_COUNT}.err"; do
  key=${spec%%=*}
  suffix=${spec#*=}
  observed=$(sed -n "s/.* ${key}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")
  if [[ "$observed" != *"$suffix" ]]; then
    echo "live Slurm $key path is not restart-distinguishing: $observed" >&2
    exit 2
  fi
done
if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  echo "SOURCE_ROOT must be a Git checkout" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/train"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "reviewed source commit is not checked out" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "SOURCE_ROOT must be a clean reviewed checkout" >&2
  exit 2
fi
if [[ ! -f "$0" || -L "$0" ]] || ! cmp --silent -- "$0" \
  "$SOURCE_ROOT/train/jobs/causal_carry_motor.sbatch"; then
  echo "spooled wrapper differs from reviewed checkout" >&2
  exit 2
fi
SOURCE_MANIFEST_SHA256=$("$PY" - "$SOURCE_COMMIT" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

commit, *paths = sys.argv[1:]
sources = {}
for name in paths:
    payload = Path(name).read_bytes()
    committed = subprocess.check_output(["git", "show", f"{commit}:{name}"])
    if payload != committed:
        raise SystemExit(f"source differs from reviewed commit: {name}")
    sources[name] = hashlib.sha256(payload).hexdigest()
encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
)

CUDA_VISIBLE_DEVICES= "$PY" train/causal_carry_motor.py validate-confirmation \
  --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
  --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
  --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256"

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" != 1 ]]; then
  echo "carry-motor job requires exactly one visible GPU" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r')" != "NVIDIA H100 PCIe" ]]; then
  echo "carry-motor job requires exactly one NVIDIA H100 PCIe" >&2
  exit 2
fi

if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch

try:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA H100 PCIe"
    ):
        raise RuntimeError("allocated device is not the canonical H100 PCIe")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    probe.add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[carry-motor] H100 preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "carry-motor H100 preflight failed after allocation" >&2
  exit 75
fi

if [[ "$MODE" == extract ]]; then
  if [[ ! "$SHARD_INDEX" =~ ^[0-9]+$ || "$SHARD_COUNT" != 8 ]]; then
    echo "extract mode requires SHARD_INDEX and canonical SHARD_COUNT=8" >&2
    exit 2
  fi
  "$PY" train/causal_carry_motor.py extract \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
    --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256" \
    --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
elif [[ "$MODE" == fit-shards ]]; then
  for index in {00..07}; do
    shard_directory=$PLAN_ROOT/shard_$index
    shard=$shard_directory/features.pt
    if [[ ! -d "$shard_directory" || -L "$shard_directory" || \
          "$(stat -c '%a' "$shard_directory")" != 555 || \
          "$(find "$shard_directory" -mindepth 1 -maxdepth 1 -printf '%f\n')" != features.pt || \
          ! -f "$shard" || -L "$shard" || \
          "$(stat -c '%F' "$shard")" != "regular file" || \
          "$(stat -c '%a' "$shard")" != 444 || \
          "$(stat -c '%h' "$shard")" != 1 ]]; then
      echo "canonical fit shard preflight failed: $shard" >&2
      exit 2
    fi
  done
  "$PY" train/causal_carry_motor.py fit-shards \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
    --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256" \
    --plan "$PLAN" --plan-sha256 "$PLAN_SHA256"
elif [[ "$MODE" == eval ]]; then
  "$PY" train/causal_carry_motor.py eval \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
    --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256" \
    --plan "$PLAN" --plan-sha256 "$PLAN_SHA256"
elif [[ "$MODE" == confirmation-eval ]]; then
  "$PY" train/causal_carry_motor.py confirmation-eval \
    --ckpt "$CKPT" --tokenizer "$TOKENIZER" --episodes "$EPISODES" \
    --cycle "$CYCLE" --source-commit "$SOURCE_COMMIT" \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --confirmation-commitment "$CONFIRMATION_COMMITMENT" \
    --confirmation-commitment-sha256 "$CONFIRMATION_COMMITMENT_SHA256" \
    --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" \
    --confirmation-secret-file "$CONFIRMATION_SECRET_FILE"
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi
