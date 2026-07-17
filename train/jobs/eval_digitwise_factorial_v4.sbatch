#!/bin/bash
# Frozen full-board evaluation of the four completed factorial-v4 SFT arms.
# Each array task evaluates exactly one checkpoint and publishes one report.
#SBATCH -J shohin-factorial-v4-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --array=0-3%4
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/factorial_v4_eval_%A_%a.site.log
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to a clean exact-commit checkout}
DATA_ROOT=${DATA_ROOT:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$DATA_ROOT/miniforge3/bin/python}
JOB_ID=${SLURM_JOB_ID:?factorial evaluator must run under Slurm}
TASK_ID=${SLURM_ARRAY_TASK_ID:?factorial evaluator requires array task 0-3}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
EVAL_COMMIT=${EVAL_COMMIT:?set EVAL_COMMIT to the reviewed clean evaluator commit}

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "[factorial-eval] invalid job id" >&2; exit 2; }
[[ "$TASK_ID" =~ ^[0-3]$ ]] || { echo "[factorial-eval] invalid array task" >&2; exit 2; }
[[ "$RESTART_COUNT" = 0 ]] || { echo "[factorial-eval] refusing restart" >&2; exit 2; }
[[ "$EVAL_COMMIT" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || {
  echo "[factorial-eval] malformed EVAL_COMMIT" >&2
  exit 2
}

for forbidden in ARM CKPT CHECKPOINT TOKENIZER EPISODES OUT OUTPUT PER_REGIME MAX_NEW \
  TRANSITION_MAX_NEW FINAL_MAX_NEW PROMPT_STYLE TEMPERATURE TOP_K SEED; do
  if [ "${!forbidden+x}" = x ]; then
    echo "[factorial-eval] forbidden scientific override: $forbidden" >&2
    exit 3
  fi
done

case "$TASK_ID" in
  0) arm=iid; training_job=692071 ;;
  1) arm=term; training_job=692073 ;;
  2) arm=width; training_job=692075 ;;
  3) arm=term_width; training_job=692077 ;;
esac

FACTORIAL_SOURCE=/lustre/fs1/home/sa305415/shohin_factorial_source_c24e16e_v3
CHECKPOINT_ORIGIN=$FACTORIAL_SOURCE/train/sft_factorial_runs/job_${training_job}_attempt_0
TOKENIZER_ORIGIN=$DATA_ROOT/artifacts/shohin-tok-32k.json
EPISODES_ORIGIN=$DATA_ROOT/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl
OUTPUT_PARENT=$DATA_ROOT/artifacts/evals/digitwise_factorial_v4_full_${EVAL_COMMIT}
OUTPUT_DIR=$OUTPUT_PARENT/$arm
SNAP=$DATA_ROOT/scratchpad/factorial_v4_eval_${JOB_ID}_${TASK_ID}_attempt_${RESTART_COUNT}

sources=(
  train/eval_digitwise_factorial_v4.py
  train/test_eval_digitwise_factorial_v4.py
  train/jobs/eval_digitwise_factorial_v4.sbatch
  train/digitwise_protocol.py
  train/model.py
)

check_regular() {
  local path=$1 label=$2
  [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "[factorial-eval] $label is not one regular non-symlink file: $path" >&2
    exit 4
  }
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

for root in "$SOURCE_ROOT" "$DATA_ROOT"; do
  [[ "$root" = /* ]] || {
    echo "[factorial-eval] source and data roots must be absolute" >&2
    exit 5
  }
done
[ -x "$PY" ] || {
  echo "[factorial-eval] Python runtime is unavailable: $PY" >&2
  exit 5
}
git -C "$SOURCE_ROOT" cat-file -e "${EVAL_COMMIT}^{commit}" || {
  echo "[factorial-eval] reviewed evaluator commit is unavailable" >&2
  exit 5
}
[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EVAL_COMMIT" ] || {
  echo "[factorial-eval] checkout HEAD differs from reviewed evaluator commit" >&2
  exit 5
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo "[factorial-eval] reviewed source checkout is not clean" >&2
  exit 5
}

declare -A expected_sources=()
for relative in "${sources[@]}"; do
  git -C "$SOURCE_ROOT" ls-files --error-unmatch "$relative" >/dev/null || {
    echo "[factorial-eval] untracked evaluator source: $relative" >&2
    exit 5
  }
  git -C "$SOURCE_ROOT" diff --quiet "$EVAL_COMMIT" -- "$relative" || {
    echo "[factorial-eval] evaluator source differs from reviewed commit: $relative" >&2
    exit 5
  }
  expected_sources[$relative]=$(git -C "$SOURCE_ROOT" show "$EVAL_COMMIT:$relative" | sha256sum | awk '{print $1}')
  check_regular "$SOURCE_ROOT/$relative" "evaluator source $relative"
  [ "$(sha256_file "$SOURCE_ROOT/$relative")" = "${expected_sources[$relative]}" ] || {
    echo "[factorial-eval] live evaluator source hash mismatch: $relative" >&2
    exit 5
  }
done

SPOOLED_WRAPPER_SHA256=$(sha256_file "$0")
[ "$SPOOLED_WRAPPER_SHA256" = "${expected_sources[train/jobs/eval_digitwise_factorial_v4.sbatch]}" ] || {
  echo "[factorial-eval] spooled wrapper differs from reviewed source" >&2
  exit 6
}

check_regular "$TOKENIZER_ORIGIN" tokenizer
check_regular "$EPISODES_ORIGIN" heldout
[ "$(sha256_file "$TOKENIZER_ORIGIN")" = 87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4 ] || {
  echo "[factorial-eval] tokenizer hash mismatch" >&2
  exit 7
}
[ "$(sha256_file "$EPISODES_ORIGIN")" = 89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646 ] || {
  echo "[factorial-eval] heldout hash mismatch" >&2
  exit 7
}
[ -d "$CHECKPOINT_ORIGIN" ] && [ ! -L "$CHECKPOINT_ORIGIN" ] || {
  echo "[factorial-eval] checkpoint origin is unavailable" >&2
  exit 7
}
[ "$(stat -c %a "$CHECKPOINT_ORIGIN")" = 555 ] || {
  echo "[factorial-eval] checkpoint origin lost its 0555 seal" >&2
  exit 7
}
expected_checkpoint_files='exact_budget_final.json
exact_budget_preflight.json
production_admission.json
reviewed_source_manifest.json
scientific_sources.json
sft_ep1.pt'
actual_checkpoint_files=$(find "$CHECKPOINT_ORIGIN" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
[ "$actual_checkpoint_files" = "$expected_checkpoint_files" ] || {
  echo "[factorial-eval] checkpoint origin is not closed-world" >&2
  exit 7
}
while IFS= read -r name; do
  check_regular "$CHECKPOINT_ORIGIN/$name" "checkpoint artifact $name"
  [ "$(stat -c %a "$CHECKPOINT_ORIGIN/$name")" = 444 ] || {
    echo "[factorial-eval] checkpoint artifact lost its 0444 seal: $name" >&2
    exit 7
  }
done <<< "$expected_checkpoint_files"

if command -v scontrol >/dev/null 2>&1; then
  policy=$(scontrol show job -o "$JOB_ID") || {
    echo "[factorial-eval] cannot inspect live Slurm policy" >&2
    exit 8
  }
  [[ " $policy " == *" Requeue=0 "* ]] || {
    echo "[factorial-eval] live Slurm policy permits requeue" >&2
    exit 8
  }
fi

[ ! -e "$SNAP" ] && [ ! -L "$SNAP" ] || {
  echo "[factorial-eval] refusing existing snapshot: $SNAP" >&2
  exit 9
}
[ ! -e "$OUTPUT_DIR" ] && [ ! -L "$OUTPUT_DIR" ] || {
  echo "[factorial-eval] refusing existing scientific output: $OUTPUT_DIR" >&2
  exit 9
}
umask 077
mkdir -p "$DATA_ROOT/logs" "$OUTPUT_PARENT"
mkdir -p "$SNAP/train/jobs" "$SNAP/input/checkpoint"

cleanup() {
  local status=$?
  chmod -R u+w "$SNAP" 2>/dev/null || true
  rm -rf "$SNAP"
  exit "$status"
}
trap cleanup EXIT

for relative in "${sources[@]}"; do
  mkdir -p "$SNAP/$(dirname "$relative")"
  cp --reflink=auto -- "$SOURCE_ROOT/$relative" "$SNAP/$relative"
  [ "$(sha256_file "$SNAP/$relative")" = "${expected_sources[$relative]}" ] || {
    echo "[factorial-eval] snapshot source hash mismatch: $relative" >&2
    exit 10
  }
done
cp --reflink=auto -- "$TOKENIZER_ORIGIN" "$SNAP/input/tokenizer.json"
cp --reflink=auto -- "$EPISODES_ORIGIN" "$SNAP/input/heldout.jsonl"
while IFS= read -r name; do
  cp --reflink=auto -- "$CHECKPOINT_ORIGIN/$name" "$SNAP/input/checkpoint/$name"
done <<< "$expected_checkpoint_files"

export ARM_FOR_BINDING=$arm
export TASK_FOR_BINDING=$TASK_ID
export JOB_FOR_BINDING=$JOB_ID
export EVAL_COMMIT_FOR_BINDING=$EVAL_COMMIT
export SPOOLED_WRAPPER_FOR_BINDING=$SPOOLED_WRAPPER_SHA256
export CHECKPOINT_ORIGIN_FOR_BINDING=$CHECKPOINT_ORIGIN
export SOURCE_ROOT_FOR_BINDING=$SOURCE_ROOT
export SNAP_FOR_BINDING=$SNAP
for relative in "${sources[@]}"; do
  key=$(printf '%s' "$relative" | tr '/.' '__')
  export "SOURCE_${key}=${expected_sources[$relative]}"
done
"$PY" - "$SNAP/input/source_binding.json" <<'PY'
import json
import base64
import os
import pathlib
import subprocess
import sys

paths = (
    "train/eval_digitwise_factorial_v4.py",
    "train/test_eval_digitwise_factorial_v4.py",
    "train/jobs/eval_digitwise_factorial_v4.sbatch",
    "train/digitwise_protocol.py",
    "train/model.py",
)


def key(path):
    return "SOURCE_" + path.replace("/", "_").replace(".", "_")


source_root = os.environ["SOURCE_ROOT_FOR_BINDING"]
eval_commit = os.environ["EVAL_COMMIT_FOR_BINDING"]
object_id_bytes = len(eval_commit) // 2
commit_payload = subprocess.check_output(
    ["git", "-C", source_root, "cat-file", "commit", eval_commit]
)
first_line = commit_payload.splitlines()[0].split()
if len(first_line) != 2 or first_line[0] != b"tree":
    raise SystemExit("reviewed commit does not declare one root tree")
root_tree_id = first_line[1].decode("ascii")
tree_payloads = {}


def tree_entries(tree_id):
    payload = subprocess.check_output(
        ["git", "-C", source_root, "cat-file", "tree", tree_id]
    )
    tree_payloads[tree_id] = payload
    entries = {}
    cursor = 0
    while cursor < len(payload):
        space = payload.index(b" ", cursor)
        nul = payload.index(b"\0", space + 1)
        mode = payload[cursor:space].decode("ascii")
        name = payload[space + 1 : nul].decode("utf-8")
        object_start = nul + 1
        object_end = object_start + object_id_bytes
        if object_end > len(payload) or name in entries:
            raise SystemExit("malformed reviewed Git tree")
        entries[name] = (mode, payload[object_start:object_end].hex())
        cursor = object_end
    return entries


source_receipts = {}
for source_path in paths:
    current_tree_id = root_tree_id
    components = source_path.split("/")
    for index, component in enumerate(components):
        entries = tree_entries(current_tree_id)
        if component not in entries:
            raise SystemExit("reviewed source is absent from commit tree: " + source_path)
        mode, object_id = entries[component]
        if index < len(components) - 1:
            if mode != "40000":
                raise SystemExit("reviewed source parent is not a tree: " + source_path)
            current_tree_id = object_id
        else:
            if mode not in {"100644", "100755"}:
                raise SystemExit("reviewed source is not a regular blob: " + source_path)
            source_receipts[source_path] = {
                "sha256": os.environ[key(source_path)],
                "git_blob_id": object_id,
                "git_mode": mode,
            }


value = {
    "schema": "shohin-digitwise-factorial-v4-eval-source-binding-v2",
    "eval_commit": eval_commit,
    "git_commit_object_b64": base64.b64encode(commit_payload).decode("ascii"),
    "git_tree_objects_b64": {
        object_id: base64.b64encode(tree_payloads[object_id]).decode("ascii")
        for object_id in sorted(tree_payloads)
    },
    "reviewed_clean_checkout": True,
    "executing_source_root": str(pathlib.Path(os.environ["SNAP_FOR_BINDING"]).resolve()),
    "sources": source_receipts,
    "spooled_wrapper_sha256": os.environ["SPOOLED_WRAPPER_FOR_BINDING"],
    "slurm": {
        "job_id": int(os.environ["JOB_FOR_BINDING"]),
        "array_task_id": int(os.environ["TASK_FOR_BINDING"]),
        "restart_count": 0,
    },
    "scientific_inputs": {
        "arm": os.environ["ARM_FOR_BINDING"],
        "checkpoint_origin": os.environ["CHECKPOINT_ORIGIN_FOR_BINDING"],
        "checkpoint_sha256": {
            "iid": "e2cfa811da5e7150c093f160412d9358d603ec9563a9ee9c432193e4516b8846",
            "term": "1f7307705f632e9290e9f5c8580d8445981d5ffffc974a5f2854c5b93c4ad96d",
            "width": "f8f9a90ce2e15b58a3e93e255e71f8f0abe517cfacea1e5e51d2cdd45bb8c847",
            "term_width": "06b5d9f2dc4a210fbc4fbe9f62fe036dca2a05598d5e463cf6a0e9133866779e",
        }[os.environ["ARM_FOR_BINDING"]],
        "heldout_sha256": "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646",
        "tokenizer_sha256": "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4",
    },
}
payload = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
path = pathlib.Path(sys.argv[1])
with path.open("xb") as sink:
    sink.write(payload)
PY
SOURCE_BINDING_SHA256=$(sha256_file "$SNAP/input/source_binding.json")

# Lustre parents carry SGID. Coreutils chmod preserves that bit on directories,
# so clear all special bits explicitly before applying the frozen modes.
find "$SNAP" -exec chmod a-s {} +
find "$SNAP" -type f -exec chmod 0400 {} +
find "$SNAP" -type d -exec chmod 0500 {} +
while IFS= read -r -d '' path; do
  [ "$(stat -c %a "$path")" = 400 ] || {
    echo "[factorial-eval] snapshot file did not seal mode 0400: $path" >&2
    exit 10
  }
done < <(find "$SNAP" -type f -print0)
while IFS= read -r -d '' path; do
  [ "$(stat -c %a "$path")" = 500 ] || {
    echo "[factorial-eval] snapshot directory did not seal mode 0500: $path" >&2
    exit 10
  }
done < <(find "$SNAP" -type d -print0)

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0
export PYTHONPATH=$SNAP/train

timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA GPU is required")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory = torch.cuda.get_device_properties(0).total_memory
if "H100" not in name or capability != (9, 0) or memory < 79 * 1024**3:
    raise SystemExit(f"wrong GPU: name={name!r} capability={capability} memory={memory}")
torch.empty(1, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print(f"[factorial-eval] H100 preflight name={name} memory={memory}", flush=True)
PY

cd "$SNAP/train"
"$PY" eval_digitwise_factorial_v4.py \
  --arm "$arm" \
  --checkpoint-dir "$SNAP/input/checkpoint" \
  --tokenizer "$SNAP/input/tokenizer.json" \
  --episodes "$SNAP/input/heldout.jsonl" \
  --source-root "$SNAP" \
  --source-binding "$SNAP/input/source_binding.json" \
  --source-binding-sha256 "$SOURCE_BINDING_SHA256" \
  --output-dir "$OUTPUT_DIR"

[ "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n')" = report.json ] || {
  echo "[factorial-eval] output is not one-purpose" >&2
  exit 11
}
[ "$(stat -c %a "$OUTPUT_DIR")" = 500 ] || exit 11
[ "$(stat -c %a "$OUTPUT_DIR/report.json")" = 400 ] || exit 11
echo "[factorial-eval] complete arm=$arm report_sha256=$(sha256_file "$OUTPUT_DIR/report.json")"
