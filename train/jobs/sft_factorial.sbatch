#!/bin/bash
# Canonical equal-forward-budget SFT for one independently admitted DRS arm.
# Scientific inputs must match a reviewed clean source manifest and commit.
# The arm is derived by train/sft.py from the admission receipt and corpus.
#SBATCH -J shohin-drs-factorial
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_factorial_%j.site.log
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -euo pipefail

BASE=${SHOHIN_BASE:-/lustre/fs1/home/sa305415/shohin}
PY=$BASE/miniforge3/bin/python
JOB_ID=${SLURM_JOB_ID:?canonical factorial SFT must run under Slurm}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "[factorial-sft] invalid Slurm job id '$JOB_ID'" >&2
  exit 10
}
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || {
  echo "[factorial-sft] invalid Slurm restart count '$RESTART_COUNT'" >&2
  exit 10
}

# Slurm opens the site log before this script. Append mode prevents a forced
# requeue from truncating prior evidence; this directory separates attempts.
ATTEMPT_LOG_ROOT=${FACTORIAL_ATTEMPT_LOG_ROOT:-$BASE/logs/sft_factorial_attempts}
ATTEMPT_PARENT=$ATTEMPT_LOG_ROOT/job_${JOB_ID}
ATTEMPT_DIR=$ATTEMPT_PARENT/attempt_${RESTART_COUNT}
umask 077
mkdir -p "$ATTEMPT_PARENT"
mkdir "$ATTEMPT_DIR" || {
  echo "[factorial-sft] refusing existing attempt evidence $ATTEMPT_DIR" >&2
  exit 11
}
: > "$ATTEMPT_DIR/wrapper.log"
exec > >(tee -a "$ATTEMPT_DIR/wrapper.log") 2>&1
printf 'job_id\t%s\nrestart_count\t%s\nstarted_utc\t%s\n' \
  "$JOB_ID" "$RESTART_COUNT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$ATTEMPT_DIR/attempt_start.tsv"

attempt_status=running
record_attempt_exit() {
  local status=$1
  set +e
  printf 'status\t%s\nexit_code\t%s\nfinished_utc\t%s\n' \
    "$attempt_status" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$ATTEMPT_DIR/attempt_exit.tsv"
  return "$status"
}
trap 'record_attempt_exit $?' EXIT

if [ "$RESTART_COUNT" -ne 0 ]; then
  attempt_status=refused_requeue
  echo "[factorial-sft] refusing Slurm restart count '$RESTART_COUNT'; prior attempt evidence retained" >&2
  exit 10
fi

required=(
  DATA DATA_SHA256 INIT INIT_SHA256 TOKENIZER TOKENIZER_SHA256
  ADMISSION ADMISSION_SHA256 REVIEWED_SOURCE_MANIFEST
  REVIEWED_SOURCE_MANIFEST_SHA256 REVIEWED_COMMIT
  GENERATOR_PY_SHA256 RECURRENT_PY_SHA256 GENERATOR_TEST_PY_SHA256
  AUDITOR_PY_SHA256 AUDITOR_TEST_PY_SHA256 PROTOCOL_PY_SHA256
  SFT_PY_SHA256 MODEL_PY_SHA256 MUON_PY_SHA256 SFT_ENCODING_PY_SHA256
  SFT_TEST_PY_SHA256 SBATCH_SHA256
)
for name in "${required[@]}"; do
  [ -n "${!name:-}" ] || {
    echo "[factorial-sft] required variable $name is absent" >&2
    exit 2
  }
done

# Canonical mode has no caller arm and no environment-level tuning surface.
forbidden=(
  ARM OUT EPOCHS BS BATCH_SIZE PACKLEN PACK_LEN MAX_UPDATES EXACT_UPDATES SEED
  LRMUON LRADAM WARMUP CLIP COMPILE MAX_EXAMPLES GROUPFIELD SAMPLEWEIGHTS
  REFERENCE REPLAY_PROMPTS REPLAY_WEIGHT FREEZE_LEXICON PROMPT_OVERRIDE_FIELD
)
for name in "${forbidden[@]}"; do
  if [ "${!name+x}" = x ]; then
    echo "[factorial-sft] canonical scientific override is forbidden: $name" >&2
    exit 5
  fi
done

hash_vars=(
  DATA_SHA256 INIT_SHA256 TOKENIZER_SHA256 ADMISSION_SHA256
  REVIEWED_SOURCE_MANIFEST_SHA256 GENERATOR_PY_SHA256 RECURRENT_PY_SHA256
  GENERATOR_TEST_PY_SHA256 AUDITOR_PY_SHA256 AUDITOR_TEST_PY_SHA256
  PROTOCOL_PY_SHA256 SFT_PY_SHA256 MODEL_PY_SHA256 MUON_PY_SHA256
  SFT_ENCODING_PY_SHA256 SFT_TEST_PY_SHA256 SBATCH_SHA256
)
for name in "${hash_vars[@]}"; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[factorial-sft] $name is not a lowercase SHA-256" >&2
    exit 4
  }
  export "$name"
done
[[ "$REVIEWED_COMMIT" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || {
  echo "[factorial-sft] REVIEWED_COMMIT is not a lowercase Git object id" >&2
  exit 4
}
export REVIEWED_COMMIT

declare -A sources=(
  [pipeline/generate_digitwise_factorial_v4.py]="$GENERATOR_PY_SHA256"
  [pipeline/generate_digitwise_recurrent_v1.py]="$RECURRENT_PY_SHA256"
  [pipeline/test_generate_digitwise_factorial_v4.py]="$GENERATOR_TEST_PY_SHA256"
  [pipeline/audit_digitwise_factorial_v4.py]="$AUDITOR_PY_SHA256"
  [pipeline/test_audit_digitwise_factorial_v4.py]="$AUDITOR_TEST_PY_SHA256"
  [train/digitwise_protocol.py]="$PROTOCOL_PY_SHA256"
  [train/sft.py]="$SFT_PY_SHA256"
  [train/model.py]="$MODEL_PY_SHA256"
  [train/muon.py]="$MUON_PY_SHA256"
  [train/sft_encoding.py]="$SFT_ENCODING_PY_SHA256"
  [train/test_sft_exact_budget.py]="$SFT_TEST_PY_SHA256"
  [train/jobs/sft_factorial.sbatch]="$SBATCH_SHA256"
)

check_regular() {
  local path=$1 label=$2
  [ -f "$path" ] && [ ! -L "$path" ] || {
    echo "[factorial-sft] $label is not a regular non-symlink file" >&2
    exit 6
  }
}

check_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    echo "[factorial-sft] $label hash mismatch: expected=$expected actual=$actual" >&2
    exit 7
  }
}

check_regular "$DATA" corpus
check_regular "$INIT" init
check_regular "$TOKENIZER" tokenizer
check_regular "$ADMISSION" production_admission
check_regular "$REVIEWED_SOURCE_MANIFEST" reviewed_source_manifest
check_regular "$0" "spooled batch script"
check_hash "$DATA" "$DATA_SHA256" corpus
check_hash "$INIT" "$INIT_SHA256" init
check_hash "$TOKENIZER" "$TOKENIZER_SHA256" tokenizer
check_hash "$ADMISSION" "$ADMISSION_SHA256" production_admission
check_hash "$REVIEWED_SOURCE_MANIFEST" "$REVIEWED_SOURCE_MANIFEST_SHA256" reviewed_source_manifest
check_hash "$0" "$SBATCH_SHA256" "spooled batch script"
for relative in "${!sources[@]}"; do
  check_regular "$BASE/$relative" "source $relative"
  check_hash "$BASE/$relative" "${sources[$relative]}" "source $relative"
done

# The reviewed source set must be tracked, byte-identical to REVIEWED_COMMIT,
# and clean relative to that commit. This is local Git evidence, not attestation.
git -C "$BASE" cat-file -e "${REVIEWED_COMMIT}^{commit}" || {
  echo "[factorial-sft] reviewed commit is unavailable locally" >&2
  exit 12
}
[ "$(git -C "$BASE" rev-parse HEAD)" = "$REVIEWED_COMMIT" ] || {
  echo "[factorial-sft] checkout HEAD differs from reviewed commit" >&2
  exit 12
}
for relative in "${!sources[@]}"; do
  git -C "$BASE" ls-files --error-unmatch "$relative" >/dev/null || {
    echo "[factorial-sft] reviewed source is not tracked: $relative" >&2
    exit 12
  }
  git -C "$BASE" diff --quiet "$REVIEWED_COMMIT" -- "$relative" || {
    echo "[factorial-sft] reviewed source differs from clean commit: $relative" >&2
    exit 12
  }
  commit_hash=$(git -C "$BASE" show "$REVIEWED_COMMIT:$relative" | sha256sum | awk '{print $1}')
  [ "$commit_hash" = "${sources[$relative]}" ] || {
    echo "[factorial-sft] reviewed commit blob hash mismatch: $relative" >&2
    exit 12
  }
done

"$PY" - "$REVIEWED_SOURCE_MANIFEST" "$REVIEWED_SOURCE_MANIFEST_SHA256" "$REVIEWED_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_sha, expected_commit = sys.argv[2:]
raw = path.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit("reviewed source manifest hash changed during validation")
value = json.loads(raw)
canonical = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
if raw != canonical:
    raise SystemExit("reviewed source manifest is not canonical JSON")
expected_sources = {
    "pipeline/generate_digitwise_factorial_v4.py": os.environ["GENERATOR_PY_SHA256"],
    "pipeline/generate_digitwise_recurrent_v1.py": os.environ["RECURRENT_PY_SHA256"],
    "pipeline/test_generate_digitwise_factorial_v4.py": os.environ["GENERATOR_TEST_PY_SHA256"],
    "pipeline/audit_digitwise_factorial_v4.py": os.environ["AUDITOR_PY_SHA256"],
    "pipeline/test_audit_digitwise_factorial_v4.py": os.environ["AUDITOR_TEST_PY_SHA256"],
    "train/digitwise_protocol.py": os.environ["PROTOCOL_PY_SHA256"],
    "train/sft.py": os.environ["SFT_PY_SHA256"],
    "train/model.py": os.environ["MODEL_PY_SHA256"],
    "train/muon.py": os.environ["MUON_PY_SHA256"],
    "train/sft_encoding.py": os.environ["SFT_ENCODING_PY_SHA256"],
    "train/test_sft_exact_budget.py": os.environ["SFT_TEST_PY_SHA256"],
    "train/jobs/sft_factorial.sbatch": os.environ["SBATCH_SHA256"],
}
required = {
    "schema": "shohin-factorial-v4-reviewed-source-manifest-v1",
    "review_status": "approved",
    "reviewed_clean_commit": expected_commit,
    "clean_source_tree": True,
    "remote_attestation": False,
    "sources": expected_sources,
}
if value != required:
    raise SystemExit("reviewed source manifest content mismatch")
PY

# Verify the live scheduler policy when scontrol is available and preserve the
# exact response in attempt evidence. A mismatch is fatal; unavailability is explicit.
if command -v scontrol >/dev/null 2>&1; then
  slurm_policy=$(scontrol show job -o "$JOB_ID") || {
    echo "[factorial-sft] unable to inspect live Slurm job policy" >&2
    exit 13
  }
  printf '%s\n' "$slurm_policy" > "$ATTEMPT_DIR/scontrol_job.txt"
  [[ " $slurm_policy " == *" Requeue=0 "* ]] || {
    echo "[factorial-sft] live Slurm policy does not enforce Requeue=0" >&2
    exit 13
  }
  if [[ " $slurm_policy " =~ [[:space:]]Restarts=([0-9]+)[[:space:]] ]]; then
    [ "${BASH_REMATCH[1]}" = "$RESTART_COUNT" ] || {
      echo "[factorial-sft] live Slurm restart accounting mismatch" >&2
      exit 13
    }
  fi
else
  printf 'scontrol unavailable; #SBATCH --no-requeue remains source-bound\n' \
    > "$ATTEMPT_DIR/scontrol_unavailable.txt"
fi

RUN_ROOT=$BASE/train/sft_factorial_runs
OUT=$RUN_ROOT/job_${JOB_ID}_attempt_${RESTART_COUNT}
SNAP=$BASE/scratchpad/sft_factorial_snapshot_${JOB_ID}_attempt_${RESTART_COUNT}
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || {
  echo "[factorial-sft] refusing existing isolated output $OUT" >&2
  exit 8
}
[ ! -e "$SNAP" ] && [ ! -L "$SNAP" ] || {
  echo "[factorial-sft] refusing existing private snapshot $SNAP" >&2
  exit 9
}

mkdir -p "$BASE/logs" "$RUN_ROOT" "$SNAP/input"
cleanup() {
  chmod -R u+w "$SNAP" 2>/dev/null || true
  rm -rf "$SNAP"
}
finish_attempt() {
  local status=$?
  cleanup
  record_attempt_exit "$status"
}
trap finish_attempt EXIT

for relative in "${!sources[@]}"; do
  mkdir -p "$SNAP/$(dirname "$relative")"
  cp --reflink=auto -- "$BASE/$relative" "$SNAP/$relative"
done
cp --reflink=auto -- "$DATA" "$SNAP/input/corpus.jsonl"
cp --reflink=auto -- "$INIT" "$SNAP/input/init.pt"
cp --reflink=auto -- "$TOKENIZER" "$SNAP/input/tokenizer.json"
cp --reflink=auto -- "$ADMISSION" "$SNAP/input/production_admission.json"
cp --reflink=auto -- "$REVIEWED_SOURCE_MANIFEST" "$SNAP/input/reviewed_source_manifest.json"

for relative in "${!sources[@]}"; do
  check_hash "$SNAP/$relative" "${sources[$relative]}" "snapshot source $relative"
done
check_hash "$SNAP/input/corpus.jsonl" "$DATA_SHA256" "snapshot corpus"
check_hash "$SNAP/input/init.pt" "$INIT_SHA256" "snapshot init"
check_hash "$SNAP/input/tokenizer.json" "$TOKENIZER_SHA256" "snapshot tokenizer"
check_hash "$SNAP/input/production_admission.json" "$ADMISSION_SHA256" "snapshot admission"
check_hash "$SNAP/input/reviewed_source_manifest.json" "$REVIEWED_SOURCE_MANIFEST_SHA256" "snapshot reviewed source manifest"
find "$SNAP" -type f -exec chmod 0400 {} +
find "$SNAP" -type d -exec chmod 0500 {} +

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=1337
export PYTHONPATH=$SNAP/train:$SNAP/pipeline

timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("allocated canonical job has no CUDA device")
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0), flush=True)
PY

echo "[factorial-sft] output=$OUT snapshot=$SNAP admission=$ADMISSION_SHA256"
cd "$SNAP/train"
"$PY" - "$SNAP" \
  --canonical-exact-budget \
  --exact-updates 1560 \
  --init "$SNAP/input/init.pt" \
  --data "$SNAP/input/corpus.jsonl" \
  --tokenizer "$SNAP/input/tokenizer.json" \
  --production-admission "$SNAP/input/production_admission.json" \
  --reviewed-source-manifest "$SNAP/input/reviewed_source_manifest.json" \
  --expected-init-sha256 "$INIT_SHA256" \
  --expected-data-sha256 "$DATA_SHA256" \
  --expected-tokenizer-sha256 "$TOKENIZER_SHA256" \
  --expected-production-admission-sha256 "$ADMISSION_SHA256" \
  --expected-reviewed-source-manifest-sha256 "$REVIEWED_SOURCE_MANIFEST_SHA256" \
  --expected-reviewed-commit "$REVIEWED_COMMIT" \
  --expected-source-sha256 "pipeline/generate_digitwise_factorial_v4.py=$GENERATOR_PY_SHA256" \
  --expected-source-sha256 "pipeline/generate_digitwise_recurrent_v1.py=$RECURRENT_PY_SHA256" \
  --expected-source-sha256 "pipeline/test_generate_digitwise_factorial_v4.py=$GENERATOR_TEST_PY_SHA256" \
  --expected-source-sha256 "pipeline/audit_digitwise_factorial_v4.py=$AUDITOR_PY_SHA256" \
  --expected-source-sha256 "pipeline/test_audit_digitwise_factorial_v4.py=$AUDITOR_TEST_PY_SHA256" \
  --expected-source-sha256 "train/digitwise_protocol.py=$PROTOCOL_PY_SHA256" \
  --expected-source-sha256 "train/sft.py=$SFT_PY_SHA256" \
  --expected-source-sha256 "train/model.py=$MODEL_PY_SHA256" \
  --expected-source-sha256 "train/muon.py=$MUON_PY_SHA256" \
  --expected-source-sha256 "train/sft_encoding.py=$SFT_ENCODING_PY_SHA256" \
  --expected-source-sha256 "train/test_sft_exact_budget.py=$SFT_TEST_PY_SHA256" \
  --expected-source-sha256 "train/jobs/sft_factorial.sbatch=$SBATCH_SHA256" \
  --snapshot-root "$SNAP" \
  --epochs 1 \
  --batch-size 16 \
  --pack-len 2048 \
  --seed 1337 \
  --lr-muon 1e-3 \
  --lr-adam 2e-4 \
  --warmup 50 \
  --clip 1.0 \
  --out "$OUT" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys
import types

snapshot_root = pathlib.Path(sys.argv[1]).resolve(strict=True)
training_argv = [str(snapshot_root / "train" / "sft.py"), *sys.argv[2:]]
expected_hashes = {
    "pipeline/generate_digitwise_factorial_v4.py": os.environ["GENERATOR_PY_SHA256"],
    "pipeline/generate_digitwise_recurrent_v1.py": os.environ["RECURRENT_PY_SHA256"],
    "pipeline/test_generate_digitwise_factorial_v4.py": os.environ["GENERATOR_TEST_PY_SHA256"],
    "pipeline/audit_digitwise_factorial_v4.py": os.environ["AUDITOR_PY_SHA256"],
    "pipeline/test_audit_digitwise_factorial_v4.py": os.environ["AUDITOR_TEST_PY_SHA256"],
    "train/digitwise_protocol.py": os.environ["PROTOCOL_PY_SHA256"],
    "train/sft.py": os.environ["SFT_PY_SHA256"],
    "train/model.py": os.environ["MODEL_PY_SHA256"],
    "train/muon.py": os.environ["MUON_PY_SHA256"],
    "train/sft_encoding.py": os.environ["SFT_ENCODING_PY_SHA256"],
    "train/test_sft_exact_budget.py": os.environ["SFT_TEST_PY_SHA256"],
    "train/jobs/sft_factorial.sbatch": os.environ["SBATCH_SHA256"],
}


def identity(value):
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        stat.S_IMODE(value.st_mode),
    )


def capture_exact_bytes(relative, expected_sha256):
    """Read once; captured bytes are both source identity and execution input."""
    path = snapshot_root / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or identity(before) != identity(path_before):
            raise SystemExit(f"source changed while opening: {relative}")
        if stat.S_IMODE(before.st_mode) & 0o222:
            raise SystemExit(f"source remains writable: {relative}")
        digest = hashlib.sha256()
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if identity(after) != identity(before) or identity(path_after) != identity(before):
            raise SystemExit(f"source changed while capturing: {relative}")
        if len(payload) != before.st_size:
            raise SystemExit(f"source size changed while capturing: {relative}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"source exact-byte hash mismatch: {relative}; "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        return payload
    finally:
        os.close(descriptor)


frozen_sources = types.MappingProxyType(
    {
        relative: capture_exact_bytes(relative, expected_sha256)
        for relative, expected_sha256 in expected_hashes.items()
    }
)


def load_frozen_module(name, relative):
    module = types.ModuleType(name)
    module.__file__ = str(snapshot_root / relative)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(frozen_sources[relative], module.__file__, "exec"), module.__dict__)


load_frozen_module("model", "train/model.py")
load_frozen_module("muon", "train/muon.py")
load_frozen_module("sft_encoding", "train/sft_encoding.py")

entry = types.ModuleType("__main__")
entry.__file__ = training_argv[0]
entry.__package__ = None
entry.__loader__ = None
entry.__spec__ = None
entry._CANONICAL_BOOTSTRAP_SOURCE_BYTES = frozen_sources
sys.modules["__main__"] = entry
sys.argv = training_argv
exec(compile(frozen_sources["train/sft.py"], entry.__file__, "exec"), entry.__dict__)
PY

"$PY" - "$OUT" "$DATA_SHA256" "$INIT_SHA256" "$TOKENIZER_SHA256" \
  "$ADMISSION_SHA256" "$REVIEWED_SOURCE_MANIFEST_SHA256" "$REVIEWED_COMMIT" <<'PY'
import base64
import hashlib
import json
import pathlib
import stat
import sys

out = pathlib.Path(sys.argv[1])
data_sha, init_sha, tokenizer_sha, admission_sha, manifest_sha, commit = sys.argv[2:]
if out.is_symlink() or not out.is_dir():
    raise SystemExit("canonical output path is not a regular directory")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if raw != expected:
        raise SystemExit(f"noncanonical JSON: {path}")
    return value


names = {
    "exact_budget_preflight.json",
    "exact_budget_final.json",
    "production_admission.json",
    "reviewed_source_manifest.json",
    "scientific_sources.json",
    "sft_ep1.pt",
}
if {path.name for path in out.iterdir()} != names:
    raise SystemExit("canonical output is not closed-world")
if stat.S_IMODE(out.stat().st_mode) != 0o555:
    raise SystemExit("canonical output directory does not have the recorded 0555 seal")
for path in out.iterdir():
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise SystemExit(f"missing canonical output: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise SystemExit(f"canonical output does not have the recorded 0444 seal: {path}")

preflight_path = out / "exact_budget_preflight.json"
final_path = out / "exact_budget_final.json"
checkpoint = out / "sft_ep1.pt"
admission_path = out / "production_admission.json"
manifest_path = out / "reviewed_source_manifest.json"
bundle_path = out / "scientific_sources.json"
preflight = load_canonical(preflight_path)
final = load_canonical(final_path)
admission = load_canonical(admission_path)
manifest = load_canonical(manifest_path)
bundle = load_canonical(bundle_path)
if sha256(admission_path) != admission_sha or sha256(manifest_path) != manifest_sha:
    raise SystemExit("preserved admission or reviewed manifest hash mismatch")
arm = admission.get("admitted_arm")
if arm not in {"iid", "term", "width", "term_width"}:
    raise SystemExit("admission does not identify one factorial arm")
if admission.get("production_admission") is not True or admission.get("admission_pass") is not True:
    raise SystemExit("preserved receipt is not a production admission")
if preflight.get("arm") != arm or final.get("arm") != arm or final.get("status") != "complete":
    raise SystemExit("derived arm or completion status mismatch")
if manifest.get("reviewed_clean_commit") != commit or manifest.get("clean_source_tree") is not True:
    raise SystemExit("reviewed source commit mismatch")
if manifest.get("remote_attestation") is not False:
    raise SystemExit("reviewed source manifest overstates attestation")
if preflight.get("admission", {}).get("sha256") != admission_sha:
    raise SystemExit("preflight does not bind independent admission")
if final.get("admission") != preflight.get("admission"):
    raise SystemExit("admission binding changed between receipts")
if preflight.get("reviewed_source_manifest", {}).get("sha256") != manifest_sha:
    raise SystemExit("preflight does not bind reviewed source manifest")
if final.get("reviewed_source_manifest") != preflight.get("reviewed_source_manifest"):
    raise SystemExit("reviewed source binding changed between receipts")
if preflight.get("source_bundle", {}).get("sha256") != sha256(bundle_path):
    raise SystemExit("preflight does not bind exact source bundle")
if final.get("source_bundle") != preflight.get("source_bundle"):
    raise SystemExit("source bundle binding changed between receipts")

bundle_sources = bundle.get("sources", {})
if set(bundle_sources) != set(manifest.get("sources", {})):
    raise SystemExit("source bundle and reviewed manifest sets differ")
for relative, expected in manifest["sources"].items():
    record = bundle_sources[relative]
    payload = base64.b64decode(record["payload_base64"], validate=True)
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"source recovery failed: {relative}")
    if record["sha256"] != expected:
        raise SystemExit(f"source bundle manifest mismatch: {relative}")

completion = final.get("completion_requires", {})
if completion.get("closed_world_files") != sorted(names):
    raise SystemExit("final receipt has the wrong closed-world inventory")
if completion.get("directory_mode") != "0555" or completion.get("file_mode") != "0444":
    raise SystemExit("final receipt has the wrong recorded seal modes")
if completion.get("permission_mode_boundary") != "accidental_write_seal_not_owner_proof_immutability":
    raise SystemExit("final receipt overstates permission-mode immutability")
if completion.get("recipient_verification") != "rehash_every_closed_world_artifact":
    raise SystemExit("final receipt omits recipient rehash requirement")
if preflight.get("trust_boundary") != final.get("trust_boundary"):
    raise SystemExit("trust boundary changed between receipts")
trust = final["trust_boundary"]
if trust.get("remote_attestation") is not False or "not provide owner-proof immutability" not in trust.get("scope", ""):
    raise SystemExit("canonical receipt overstates custody")
if "accidental-write seals only" not in trust.get("output_same_uid_boundary", ""):
    raise SystemExit("canonical receipt omits the same-UID output boundary")
if preflight.get("consumption_custody") != final.get("consumption_custody"):
    raise SystemExit("consumption custody changed between receipts")
if preflight.get("inputs") != final.get("inputs") or preflight.get("sources") != final.get("sources"):
    raise SystemExit("input or source byte bindings changed between receipts")

expected_inputs = {
    "corpus": data_sha,
    "init": init_sha,
    "tokenizer": tokenizer_sha,
    "production_admission": admission_sha,
    "reviewed_source_manifest": manifest_sha,
}
for name, expected in expected_inputs.items():
    binding = final["inputs"][name]
    if binding.get("sha256") != expected or binding.get("expected_sha256") != expected:
        raise SystemExit(f"final input mismatch: {name}")
    if binding.get("custody") != "one_time_exact_byte_immutable_private_memory_v1":
        raise SystemExit(f"final input has wrong byte custody: {name}")
if set(final["sources"]) != set(manifest["sources"]):
    raise SystemExit("final source set differs from reviewed manifest")
for relative, expected in manifest["sources"].items():
    binding = final["sources"][relative]
    if binding.get("sha256") != expected or binding.get("expected_sha256") != expected:
        raise SystemExit(f"final source mismatch: {relative}")

if final["preflight"].get("sha256") != sha256(preflight_path):
    raise SystemExit("final receipt does not bind preflight")
if final["checkpoint"].get("sha256") != sha256(checkpoint):
    raise SystemExit("final receipt does not bind checkpoint")
if final["checkpoint"].get("bytes") != checkpoint.stat().st_size:
    raise SystemExit("final receipt has the wrong checkpoint size")
actual = final["actual"]
if actual["optimizer_updates"] != 1560 or actual["packs"] != 24960:
    raise SystemExit("actual optimizer update or pack budget mismatch")
if actual["forward_token_positions"] != 1560 * 16 * 2048:
    raise SystemExit("actual forward-token-position budget mismatch")
if actual["supervised_tokens"] != final["planned"]["supervised_tokens"]:
    raise SystemExit("actual supervised-token accounting mismatch")
if actual["supervised_tokens_per_update_sha256"] != final["planned"]["supervised_tokens_per_update_sha256"]:
    raise SystemExit("actual per-update supervised-token schedule mismatch")
if preflight["semantics"]["target_thinning"] is not False:
    raise SystemExit("canonical run unexpectedly thinned targets")
if preflight["budget"]["consumed_packs"] != 24960:
    raise SystemExit("planned exact pack budget mismatch")
print(f"[factorial-sft] verified final receipt {sha256(final_path)} arm={arm}", flush=True)
PY

attempt_status=complete
echo "[factorial-sft] complete output=$OUT"
