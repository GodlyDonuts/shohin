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
export PATH=/usr/bin:/bin
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
unset LD_PRELOAD PYTHONHOME PYTHONPATH
export PYTHONOPTIMIZE=0

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/best_step260000.pt}
SOURCE=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
PROBE=$BASE/train/probe_operation_selection_likelihood.py
EVAL_ROOT=$BASE/artifacts/eval_history
MIRROR_ROOT=${MIRROR_ROOT:-$BASE/artifacts/eval_history_mirror}
OUT=${OUT:?set OUT to a fresh raw260k_operation_selection_likelihood_*.json path under artifacts/eval_history}
FROZEN_COMMIT=${FROZEN_COMMIT:?set FROZEN_COMMIT to the pre-score code commit}
EVIDENCE_COMMIT=${EVIDENCE_COMMIT:?set EVIDENCE_COMMIT to the pushed pre-score receipt commit}
EVIDENCE_BUNDLE=${EVIDENCE_BUNDLE:-$BASE/artifacts/evidence/shohin_operation_likelihood_prescore.bundle}
PRESCORE_RECEIPT_RELATIVE=R12_OPERATION_SELECTION_PRESCORE_RECEIPT.json

EXPECTED_CKPT=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d
EXPECTED_TOKENIZER=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
EXPECTED_SOURCE=19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474

for path in "$PY" "$CKPT" "$SOURCE" "$TOKENIZER" "$PROBE" "$EVIDENCE_BUNDLE"; do
  [ -s "$path" ] || { echo "[operation-selection] missing required input: $path" >&2; exit 2; }
done

BASE=$(realpath -e -- "$BASE")
PY=$(realpath -e -- "$PY")
CKPT=$(realpath -e -- "$CKPT")
SOURCE=$(realpath -e -- "$SOURCE")
TOKENIZER=$(realpath -e -- "$TOKENIZER")
PROBE=$(realpath -e -- "$PROBE")
EVIDENCE_BUNDLE=$(realpath -e -- "$EVIDENCE_BUNDLE")
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
if [ -w "$EVIDENCE_BUNDLE" ]; then
  echo "[operation-selection] evidence bundle must be read-only" >&2
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
if [[ ! "$EVIDENCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[operation-selection] EVIDENCE_COMMIT must be a full lowercase SHA-1" >&2
  exit 5
fi
EVIDENCE_GIT_DIR=${SLURM_TMPDIR:?SLURM_TMPDIR is required}/shohin_operation_likelihood_evidence.git
[ ! -e "$EVIDENCE_GIT_DIR" ] || { echo "[operation-selection] evidence Git path already exists" >&2; exit 5; }
git init --bare --quiet "$EVIDENCE_GIT_DIR"
git --git-dir="$EVIDENCE_GIT_DIR" bundle verify "$EVIDENCE_BUNDLE" >/dev/null
git --git-dir="$EVIDENCE_GIT_DIR" -c pack.threads=1 -c index.threads=1 fetch --quiet "$EVIDENCE_BUNDLE" refs/heads/main:refs/heads/main
test "$(git --git-dir="$EVIDENCE_GIT_DIR" rev-parse refs/heads/main)" = "$EVIDENCE_COMMIT" || {
  echo "[operation-selection] evidence bundle main does not match EVIDENCE_COMMIT" >&2
  exit 5
}
git --git-dir="$EVIDENCE_GIT_DIR" cat-file -e "${FROZEN_COMMIT}^{commit}"
git --git-dir="$EVIDENCE_GIT_DIR" merge-base --is-ancestor "$FROZEN_COMMIT" "$EVIDENCE_COMMIT" || {
  echo "[operation-selection] frozen commit is not an ancestor of evidence commit" >&2
  exit 5
}
PRESCORE_RECEIPT_TMP=$SLURM_TMPDIR/shohin_operation_likelihood_prescore_receipt.json
git --git-dir="$EVIDENCE_GIT_DIR" show "$EVIDENCE_COMMIT:$PRESCORE_RECEIPT_RELATIVE" > "$PRESCORE_RECEIPT_TMP"
PRESCORE_RECEIPT_SHA256=$(sha256sum "$PRESCORE_RECEIPT_TMP" | awk '{print $1}')
"$PY" - "$PRESCORE_RECEIPT_TMP" "$FROZEN_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
expected = {
    "schema": "raw260k_operation_selection_prescore_receipt_v1",
    "repository": "GodlyDonuts/shohin",
    "branch": "main",
    "frozen_commit": sys.argv[2],
    "verified_remote_commit": sys.argv[2],
    "verification_method": "authenticated_gh_api_commit_lookup_and_origin_main_match",
    "score_observed_before_receipt": False,
}
if receipt != expected:
    raise SystemExit("[operation-selection] pre-score receipt contract mismatch")
print("[operation-selection] pushed pre-score receipt verified", flush=True)
PY
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
  committed=$(git --git-dir="$EVIDENCE_GIT_DIR" show "$FROZEN_COMMIT:$relative" | sha256sum | awk '{print $1}')
  if [ "$observed" != "$committed" ]; then
    echo "[operation-selection] working bytes differ from frozen commit: $relative" >&2
    exit 5
  fi
done
run_frozen_probe() {
  "$PY" - "$EVIDENCE_GIT_DIR" "$FROZEN_COMMIT" "$@" <<'PY'
import fcntl
import ctypes
import hashlib
import json
import os
import subprocess
import sys

git_dir, frozen_commit, *probe_args = sys.argv[1:]
implementation_files = {
    "preregistration": "R12_OPERATION_SELECTION_LIKELIHOOD_PREREG.md",
    "evaluator": "train/probe_operation_selection_likelihood.py",
    "tests": "train/test_probe_operation_selection_likelihood.py",
    "job": "train/jobs/probe_operation_selection_likelihood.sbatch",
    "model_loader": "train/model.py",
    "inherited_operation_cursor_contract": "R12_OPERATION_CURSOR_DIAGNOSTIC.md",
    "inherited_operation_cursor_geometry": "train/eval_operation_cursor.py",
}
payloads = {}
paths = {}
hashes = {}
descriptors = []
f_add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
f_get_seals = getattr(fcntl, "F_GET_SEALS", 1034)
f_seal_seal = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
f_seal_shrink = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
f_seal_grow = getattr(fcntl, "F_SEAL_GROW", 0x0004)
f_seal_write = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
required_seals = f_seal_write | f_seal_grow | f_seal_shrink | f_seal_seal

def create_memfd(name):
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, getattr(os, "MFD_ALLOW_SEALING", 0x0002))
    libc = ctypes.CDLL(None, use_errno=True)
    encoded = name.encode("ascii")
    try:
        function = libc.memfd_create
        function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        descriptor = function(encoded, 0x0002)
    except AttributeError:
        machine = os.uname().machine
        if machine != "x86_64":
            raise SystemExit(f"[operation-selection] no memfd_create binding on {machine}")
        descriptor = libc.syscall(319, encoded, 0x0002)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor

for name, relative in implementation_files.items():
    payload = subprocess.run(
        ["git", f"--git-dir={git_dir}", "show", f"{frozen_commit}:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    descriptor = create_memfd(f"shohin-operation-{name}")
    descriptors.append(descriptor)
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])
    os.fchmod(descriptor, 0o444)
    fcntl.fcntl(descriptor, f_add_seals, required_seals)
    if fcntl.fcntl(descriptor, f_get_seals) != required_seals:
        raise SystemExit(f"[operation-selection] incomplete memfd seals: {name}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payloads[name] = payload
    paths[name] = f"/proc/self/fd/{descriptor}"
    hashes[name] = hashlib.sha256(payload).hexdigest()

os.environ["SHOHIN_SEALED_IMPLEMENTATION_PATHS"] = json.dumps(
    paths, sort_keys=True, separators=(",", ":")
)
os.environ["SHOHIN_SEALED_IMPLEMENTATION_SHA256"] = json.dumps(
    hashes, sort_keys=True, separators=(",", ":")
)
sys.argv = [paths["evaluator"], *probe_args]
namespace = {
    "__builtins__": __builtins__,
    "__file__": paths["evaluator"],
    "__name__": "__main__",
    "__package__": None,
}
exec(
    compile(
        payloads["evaluator"],
        f"<frozen-git:{frozen_commit}:train/probe_operation_selection_likelihood.py>",
        "exec",
        dont_inherit=True,
    ),
    namespace,
    namespace,
)
PY
}
echo "[operation-selection] implementation frozen at $FROZEN_COMMIT" | tee /dev/stderr

cd "$BASE/train"
QUARANTINE_DIR=$(mktemp -d "$EVAL_ROOT/.operation_selection_quarantine.XXXXXX")
chmod 0700 "$QUARANTINE_DIR"
QUARANTINE_OUT=$QUARANTINE_DIR/$(basename -- "$OUT")
PUBLISH_COMPLETE=0
cleanup_publication() {
  rm -f -- "${QUARANTINE_OUT:-}"
  rmdir -- "${QUARANTINE_DIR:-}" 2>/dev/null || true
  if [ "${PUBLISH_COMPLETE:-0}" != 1 ]; then
    rm -f -- "${OUT:-}" "${MIRROR:-}" "${RECEIPT:-}"
  fi
}
trap cleanup_publication EXIT

run_frozen_probe \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --source "$SOURCE" \
  --out "$QUARANTINE_OUT" \
  --frozen-commit "$FROZEN_COMMIT" \
  --evidence-commit "$EVIDENCE_COMMIT" \
  --prescore-receipt-sha256 "$PRESCORE_RECEIPT_SHA256"

[ -s "$QUARANTINE_OUT" ] || { echo "[operation-selection] evaluator wrote no result" >&2; exit 6; }
[ ! -w "$QUARANTINE_OUT" ] || { echo "[operation-selection] output remained writable" >&2; exit 6; }

AUDIT_JSON=$(run_frozen_probe \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --source "$SOURCE" \
  --out "$QUARANTINE_OUT" \
  --audit-result "$QUARANTINE_OUT" \
  --frozen-commit "$FROZEN_COMMIT" \
  --evidence-commit "$EVIDENCE_COMMIT" \
  --prescore-receipt-sha256 "$PRESCORE_RECEIPT_SHA256")
AUDITED_RESULT_SHA256=$(printf '%s\n' "$AUDIT_JSON" | "$PY" -c '
import json
import sys
value = json.load(sys.stdin)
if value.get("full_preserved_result_audit") is not True:
    raise SystemExit("full preserved-result audit flag is absent")
digest = value.get("sha256")
if not isinstance(digest, str):
    raise SystemExit("full preserved-result audit SHA-256 is absent")
print(digest)
')
[[ "$AUDITED_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "[operation-selection] full audit did not return a SHA-256" >&2
  exit 6
}
echo "[operation-selection] full preserved-result audit passed" >&2

"$PY" - "$QUARANTINE_OUT" "$OUT" "$MIRROR" "$RECEIPT" "$AUDITED_RESULT_SHA256" "$FROZEN_COMMIT" "$EVIDENCE_COMMIT" "$PRESCORE_RECEIPT_SHA256" "${SLURM_JOB_ID:-none}" <<'PY'
import hashlib
import json
import os
import stat
import sys

source_path, result_path, mirror_path, receipt_path, audited_sha256, frozen_commit, evidence_commit, prescore_receipt_sha256, job_id = sys.argv[1:]
read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
source_fd = os.open(source_path, read_flags)
with os.fdopen(source_fd, "rb") as source:
    metadata = os.fstat(source.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
        raise SystemExit("[operation-selection] final result is not a read-only regular file")
    payload = source.read()
result_sha256 = hashlib.sha256(payload).hexdigest()
if result_sha256 != audited_sha256:
    raise SystemExit("[operation-selection] quarantined bytes differ from audited SHA-256")

write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

def write_verified_copy(path):
    descriptor = os.open(path, write_flags, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fchmod(sink.fileno(), 0o444)
        os.fsync(sink.fileno())
    descriptor = os.open(path, read_flags)
    with os.fdopen(descriptor, "rb") as source:
        observed = hashlib.sha256(source.read()).hexdigest()
    if observed != audited_sha256:
        raise SystemExit(f"[operation-selection] published copy hash mismatch: {path}")
    return observed

def fsync_parent(path):
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(os.path.dirname(path), directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

receipt = {
    "schema": "raw260k_operation_selection_likelihood_receipt_v1",
    "frozen_commit": frozen_commit,
    "evidence_commit": evidence_commit,
    "prescore_receipt_sha256": prescore_receipt_sha256,
    "slurm_job_id": job_id,
    "full_preserved_result_audit": True,
    "audited_result_sha256": audited_sha256,
    "result_path": os.path.realpath(result_path),
    "result_sha256": audited_sha256,
    "mirror_path": os.path.realpath(mirror_path),
    "mirror_sha256": audited_sha256,
    "publication_protocol": "receipt_precedes_score_files",
    "score_printed_by_wrapper": False,
}
receipt_payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
receipt_fd = os.open(receipt_path, write_flags, 0o400)
with os.fdopen(receipt_fd, "wb") as sink:
    sink.write(receipt_payload)
    sink.flush()
    os.fchmod(sink.fileno(), 0o444)
    os.fsync(sink.fileno())
fsync_parent(receipt_path)
mirror_sha256 = write_verified_copy(mirror_path)
published_sha256 = write_verified_copy(result_path)
print(json.dumps({
    "receipt_path": receipt_path,
    "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    "result_sha256": result_sha256,
    "score_released": False,
}, sort_keys=True), flush=True)
PY

rm -f -- "$QUARANTINE_OUT"
rmdir -- "$QUARANTINE_DIR"
PUBLISH_COMPLETE=1
trap - EXIT
sha256sum -- "$OUT" "$MIRROR" "$RECEIPT"
