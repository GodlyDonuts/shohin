#!/bin/bash
# Read-only raw-260k next-operation likelihood diagnostic. This wrapper writes
# one immutable result under artifacts/eval_history and performs no training,
# generation, retry, downstream submission, or flagship-output write.
#SBATCH --job-name=shohin-op-like
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc34
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/operation_selection_likelihood_%j.out
#SBATCH --open-mode=append
set -euo pipefail
umask 077

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CKPT=${CKPT:-$BASE/best_step260000.pt}
SOURCE=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
PROBE=$BASE/train/probe_operation_selection_likelihood.py
EVAL_ROOT=$BASE/artifacts/eval_history
MIRROR_ROOT=${MIRROR_ROOT:-$BASE/artifacts/eval_history_mirror}
OUT=${OUT:?set OUT to a fresh raw260k_operation_selection_likelihood_*.json path under artifacts/eval_history}
FROZEN_COMMIT=${FROZEN_COMMIT:?set FROZEN_COMMIT to the public pre-score Git commit}

EXPECTED_CKPT=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d
EXPECTED_TOKENIZER=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
EXPECTED_SOURCE=19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474

for path in "$PY" "$CKPT" "$SOURCE" "$TOKENIZER" "$PROBE"; do
  [ -s "$path" ] || { echo "[operation-selection] missing required input: $path" >&2; exit 2; }
done

BASE=$(realpath -e -- "$BASE")
PY=$(realpath -e -- "$PY")
CKPT=$(realpath -e -- "$CKPT")
SOURCE=$(realpath -e -- "$SOURCE")
TOKENIZER=$(realpath -e -- "$TOKENIZER")
PROBE=$(realpath -e -- "$PROBE")
mkdir -p -- "$EVAL_ROOT"
mkdir -p -- "$MIRROR_ROOT"
EVAL_ROOT=$(realpath -e -- "$EVAL_ROOT")
MIRROR_ROOT=$(realpath -e -- "$MIRROR_ROOT")
OUT=$(realpath -m -- "$OUT")
MIRROR=$(realpath -m -- "$MIRROR_ROOT/$(basename -- "$OUT")")
RECEIPT=$(realpath -m -- "$MIRROR_ROOT/$(basename -- "${OUT%.json}").receipt.json")

if [ "$(dirname -- "$OUT")" != "$EVAL_ROOT" ]; then
  echo "[operation-selection] OUT must be a direct child of $EVAL_ROOT" >&2
  exit 3
fi
if [ "$MIRROR_ROOT" = "$EVAL_ROOT" ]; then
  echo "[operation-selection] mirror root must differ from result root" >&2
  exit 3
fi
for path in "$MIRROR" "$RECEIPT"; do
  if [ "$(dirname -- "$path")" != "$MIRROR_ROOT" ] || [ -e "$path" ] || [ -L "$path" ]; then
    echo "[operation-selection] mirror/receipt path is not fresh and direct: $path" >&2
    exit 3
  fi
done
case "$(basename -- "$OUT")" in
  raw260k_operation_selection_likelihood_*.json) ;;
  *) echo "[operation-selection] OUT has the wrong frozen filename pattern" >&2; exit 3 ;;
esac
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "[operation-selection] refusing existing output: $OUT" >&2
  exit 3
fi
if [ -w "$SOURCE" ]; then
  echo "[operation-selection] frozen source must be read-only" >&2
  exit 4
fi

check_sha256() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum -- "$path" | awk '{print $1}')
  if [ "$observed" != "$expected" ]; then
    echo "[operation-selection] SHA-256 mismatch for $path" >&2
    exit 5
  fi
}

check_sha256 "$CKPT" "$EXPECTED_CKPT"
check_sha256 "$TOKENIZER" "$EXPECTED_TOKENIZER"
check_sha256 "$SOURCE" "$EXPECTED_SOURCE"

if [[ ! "$FROZEN_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[operation-selection] FROZEN_COMMIT must be a full lowercase SHA-1" >&2
  exit 5
fi
git -C "$BASE" fetch --quiet --force origin main:refs/remotes/origin/main
git -C "$BASE" cat-file -e "${FROZEN_COMMIT}^{commit}"
git -C "$BASE" merge-base --is-ancestor "$FROZEN_COMMIT" refs/remotes/origin/main || {
  echo "[operation-selection] frozen commit is not on the fetched public origin/main" >&2
  exit 5
}
IMPLEMENTATION_FILES=(
  "R12_OPERATION_SELECTION_LIKELIHOOD_PREREG.md"
  "R12_OPERATION_CURSOR_DIAGNOSTIC.md"
  "train/probe_operation_selection_likelihood.py"
  "train/test_probe_operation_selection_likelihood.py"
  "train/jobs/probe_operation_selection_likelihood.sbatch"
  "train/model.py"
  "train/eval_operation_cursor.py"
)
for relative in "${IMPLEMENTATION_FILES[@]}"; do
  observed=$(sha256sum -- "$BASE/$relative" | awk '{print $1}')
  committed=$(git -C "$BASE" show "$FROZEN_COMMIT:$relative" | sha256sum | awk '{print $1}')
  if [ "$observed" != "$committed" ]; then
    echo "[operation-selection] working bytes differ from frozen commit: $relative" >&2
    exit 5
  fi
done
echo "[operation-selection] implementation frozen at $FROZEN_COMMIT" | tee /dev/stderr

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0

cd "$BASE/train"
QUARANTINE_DIR=$(mktemp -d "$EVAL_ROOT/.operation_selection_quarantine.XXXXXX")
chmod 0700 "$QUARANTINE_DIR"
QUARANTINE_OUT=$QUARANTINE_DIR/$(basename -- "$OUT")
cleanup_quarantine() {
  rm -f -- "${QUARANTINE_OUT:-}"
  rmdir -- "${QUARANTINE_DIR:-}" 2>/dev/null || true
}
trap cleanup_quarantine EXIT

"$PY" "$PROBE" \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --source "$SOURCE" \
  --out "$QUARANTINE_OUT" \
  --frozen-commit "$FROZEN_COMMIT"

[ -s "$QUARANTINE_OUT" ] || { echo "[operation-selection] evaluator wrote no result" >&2; exit 6; }
[ ! -w "$QUARANTINE_OUT" ] || { echo "[operation-selection] output remained writable" >&2; exit 6; }

"$PY" "$PROBE" \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --source "$SOURCE" \
  --out "$QUARANTINE_OUT" \
  --audit-result "$QUARANTINE_OUT" \
  --frozen-commit "$FROZEN_COMMIT"

"$PY" - "$QUARANTINE_OUT" "$OUT" <<'PY'
import os
import sys

source, destination = sys.argv[1:]
if os.path.lexists(destination):
    raise SystemExit("[operation-selection] final result path appeared during quarantine")
os.link(source, destination, follow_symlinks=False)
os.unlink(source)
PY
rmdir -- "$QUARANTINE_DIR"
trap - EXIT

"$PY" - "$OUT" "$MIRROR" "$RECEIPT" "$FROZEN_COMMIT" "${SLURM_JOB_ID:-none}" <<'PY'
import hashlib
import json
import os
import stat
import sys

source_path, mirror_path, receipt_path, frozen_commit, job_id = sys.argv[1:]
read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
source_fd = os.open(source_path, read_flags)
with os.fdopen(source_fd, "rb") as source:
    metadata = os.fstat(source.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
        raise SystemExit("[operation-selection] final result is not a read-only regular file")
    payload = source.read()
result_sha256 = hashlib.sha256(payload).hexdigest()

write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
mirror_fd = os.open(mirror_path, write_flags, 0o400)
with os.fdopen(mirror_fd, "wb") as mirror:
    mirror.write(payload)
    mirror.flush()
    os.fsync(mirror.fileno())
    os.fchmod(mirror.fileno(), 0o444)
mirror_fd = os.open(mirror_path, read_flags)
with os.fdopen(mirror_fd, "rb") as mirror:
    mirror_sha256 = hashlib.sha256(mirror.read()).hexdigest()
if mirror_sha256 != result_sha256:
    raise SystemExit("[operation-selection] external mirror hash mismatch")

receipt = {
    "schema": "raw260k_operation_selection_likelihood_receipt_v1",
    "frozen_commit": frozen_commit,
    "slurm_job_id": job_id,
    "full_preserved_result_audit": True,
    "result_path": os.path.realpath(source_path),
    "result_sha256": result_sha256,
    "mirror_path": os.path.realpath(mirror_path),
    "mirror_sha256": mirror_sha256,
    "score_printed_by_wrapper": False,
}
receipt_payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
receipt_fd = os.open(receipt_path, write_flags, 0o400)
with os.fdopen(receipt_fd, "wb") as sink:
    sink.write(receipt_payload)
    sink.flush()
    os.fsync(sink.fileno())
    os.fchmod(sink.fileno(), 0o444)
print(json.dumps({
    "receipt_path": receipt_path,
    "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    "result_sha256": result_sha256,
    "score_released": False,
}, sort_keys=True), flush=True)
PY

sha256sum -- "$OUT" "$MIRROR" "$RECEIPT"
