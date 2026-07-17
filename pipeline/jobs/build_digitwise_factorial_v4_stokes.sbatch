#!/bin/bash
# Build and independently admit all four production arms of the DRS factorial.
# CPU-only: run on Stokes, never on an H100 allocation.
#SBATCH -J shohin-factorial-v4-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/factorial_v4_build_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to a clean commit-pinned checkout}
DATA_ROOT=${DATA_ROOT:-/lustre/fs1/home/sa305415/shohin}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed 40-hex commit}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to a fresh final production directory}
HELDOUT=${HELDOUT:-$DATA_ROOT/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
TOKENIZER=${TOKENIZER:-$DATA_ROOT/artifacts/shohin-tok-32k.json}
HELDOUT_SHA256=${HELDOUT_SHA256:?set the frozen heldout SHA-256}
TOKENIZER_SHA256=${TOKENIZER_SHA256:?set the tokenizer SHA-256}
JOB_ID=${SLURM_JOB_ID:?this production build must run under Slurm}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
PY=${PYTHON:-/usr/bin/python3.12}

[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[factorial-v4] SOURCE_COMMIT must be a lowercase 40-hex commit" >&2
  exit 2
}
for digest in "$HELDOUT_SHA256" "$TOKENIZER_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[factorial-v4] input hash must be lowercase SHA-256" >&2
    exit 2
  }
done
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] && [ "$RESTART_COUNT" -eq 0 ] || {
  echo "[factorial-v4] refusing restarted production attempt" >&2
  exit 3
}

check_regular() {
  [ -f "$1" ] && [ ! -L "$1" ] || {
    echo "[factorial-v4] $2 is not a regular non-symlink file: $1" >&2
    exit 4
  }
}

check_hash() {
  local actual
  actual=$(sha256sum "$1" | awk '{print $1}')
  [ "$actual" = "$2" ] || {
    echo "[factorial-v4] $3 hash mismatch: expected=$2 actual=$actual" >&2
    exit 5
  }
}

check_regular "$HELDOUT" heldout
check_regular "$TOKENIZER" tokenizer
check_hash "$HELDOUT" "$HELDOUT_SHA256" heldout
check_hash "$TOKENIZER" "$TOKENIZER_SHA256" tokenizer
[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  echo "[factorial-v4] source HEAD does not match reviewed commit" >&2
  exit 6
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain)" ] || {
  echo "[factorial-v4] source checkout is not clean" >&2
  exit 6
}

required_sources=(
  pipeline/generate_digitwise_factorial_v4.py
  pipeline/generate_digitwise_recurrent_v1.py
  pipeline/test_generate_digitwise_factorial_v4.py
  pipeline/audit_digitwise_factorial_v4.py
  pipeline/test_audit_digitwise_factorial_v4.py
  train/digitwise_protocol.py
  train/sft.py
  train/sft_encoding.py
  train/test_sft_exact_budget.py
  train/jobs/sft_factorial.sbatch
  pipeline/jobs/build_digitwise_factorial_v4_stokes.sbatch
)
for relative in "${required_sources[@]}"; do
  check_regular "$SOURCE_ROOT/$relative" "source $relative"
  git -C "$SOURCE_ROOT" ls-files --error-unmatch "$relative" >/dev/null
  git -C "$SOURCE_ROOT" diff --quiet "$SOURCE_COMMIT" -- "$relative"
done
wrapper_sha=$(sha256sum "$SOURCE_ROOT/pipeline/jobs/build_digitwise_factorial_v4_stokes.sbatch" | awk '{print $1}')
check_regular "$0" "spooled batch script"
check_hash "$0" "$wrapper_sha" "spooled batch script"
job_record=$(scontrol show job -o "$JOB_ID")
[[ " $job_record " == *" Requeue=0 "* ]] || {
  echo "[factorial-v4] live Slurm policy does not enforce no-requeue" >&2
  exit 6
}
"$PY" -c 'import tokenizers; print(tokenizers.__version__)'

parent=$(dirname "$OUT_ROOT")
partial="${OUT_ROOT}.partial.${JOB_ID}"
mkdir -p "$parent"
[ ! -e "$OUT_ROOT" ] && [ ! -L "$OUT_ROOT" ] || {
  echo "[factorial-v4] refusing existing final output $OUT_ROOT" >&2
  exit 7
}
[ ! -e "$partial" ] && [ ! -L "$partial" ] || {
  echo "[factorial-v4] refusing existing partial output $partial" >&2
  exit 7
}
mkdir "$partial"

snapshot=$partial/snapshot
snapshot_source=$snapshot/source
snapshot_input=$snapshot/input
mkdir -p "$snapshot_source" "$snapshot_input"
for relative in "${required_sources[@]}"; do
  mkdir -p "$snapshot_source/$(dirname "$relative")"
  cp --reflink=auto -- "$SOURCE_ROOT/$relative" "$snapshot_source/$relative"
  working_sha=$(sha256sum "$snapshot_source/$relative" | awk '{print $1}')
  commit_sha=$(git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:$relative" | sha256sum | awk '{print $1}')
  [ "$working_sha" = "$commit_sha" ] || {
    echo "[factorial-v4] snapshot source differs from reviewed commit: $relative" >&2
    exit 9
  }
done
cp --reflink=auto -- "$HELDOUT" "$snapshot_input/heldout.jsonl"
cp --reflink=auto -- "$TOKENIZER" "$snapshot_input/tokenizer.json"
check_hash "$snapshot_input/heldout.jsonl" "$HELDOUT_SHA256" "snapshot heldout"
check_hash "$snapshot_input/tokenizer.json" "$TOKENIZER_SHA256" "snapshot tokenizer"
find "$snapshot" -type f -exec chmod 0444 {} +
find "$snapshot" -type d -exec chmod 0555 {} +

# Revalidate the mutable origins after the immutable experiment snapshot exists.
[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT" ]
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain)" ]
check_hash "$HELDOUT" "$HELDOUT_SHA256" heldout
check_hash "$TOKENIZER" "$TOKENIZER_SHA256" tokenizer

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
generator=$snapshot_source/pipeline/generate_digitwise_factorial_v4.py
auditor=$snapshot_source/pipeline/audit_digitwise_factorial_v4.py
frozen_heldout=$snapshot_input/heldout.jsonl
frozen_tokenizer=$snapshot_input/tokenizer.json
arms=(iid term width term_width)

for arm in "${arms[@]}"; do
  mkdir "$partial/$arm"
  "$PY" "$generator" \
    --mode production \
    --arm "$arm" \
    --heldout "$frozen_heldout" \
    --data-out "$partial/$arm/data.jsonl" \
    --episodes-out "$partial/$arm/episodes.jsonl" \
    --report "$partial/$arm/generated.json" \
    > "$partial/$arm/generator.stdout"
done

paired_arm() {
  case "$1" in
    iid) printf 'term\n' ;;
    term) printf 'iid\n' ;;
    width) printf 'term_width\n' ;;
    term_width) printf 'width\n' ;;
    *) return 1 ;;
  esac
}

for arm in "${arms[@]}"; do
  pair=$(paired_arm "$arm")
  "$PY" "$auditor" \
    --mode production \
    --arm "$arm" \
    --data "$partial/$arm/data.jsonl" \
    --episodes "$partial/$arm/episodes.jsonl" \
    --paired-data "$partial/$pair/data.jsonl" \
    --paired-episodes "$partial/$pair/episodes.jsonl" \
    --heldout "$frozen_heldout" \
    --tokenizer "$frozen_tokenizer" \
    --generator-report "$partial/$arm/generated.json" \
    --paired-generator-report "$partial/$pair/generated.json" \
    --pack-length 2048 \
    --out "$partial/$arm/production_admission.json" \
    > "$partial/$arm/auditor.stdout"
done

# Reopen every frozen input in a fresh independent auditor process. The two
# canonical receipts must be byte-identical before the finalizer trusts either.
for arm in "${arms[@]}"; do
  pair=$(paired_arm "$arm")
  "$PY" "$auditor" \
    --mode production \
    --arm "$arm" \
    --data "$partial/$arm/data.jsonl" \
    --episodes "$partial/$arm/episodes.jsonl" \
    --paired-data "$partial/$pair/data.jsonl" \
    --paired-episodes "$partial/$pair/episodes.jsonl" \
    --heldout "$frozen_heldout" \
    --tokenizer "$frozen_tokenizer" \
    --generator-report "$partial/$arm/generated.json" \
    --paired-generator-report "$partial/$pair/generated.json" \
    --pack-length 2048 \
    --out "$partial/$arm/production_admission_replay.json" \
    > "$partial/$arm/auditor_replay.stdout"
  cmp --silent \
    "$partial/$arm/production_admission.json" \
    "$partial/$arm/production_admission_replay.json" || {
      echo "[factorial-v4] independent admission replay differs for $arm" >&2
      exit 10
    }
done

"$PY" - "$partial" "$snapshot_source" "$SOURCE_COMMIT" \
  "$frozen_heldout" "$HELDOUT_SHA256" "$frozen_tokenizer" "$TOKENIZER_SHA256" \
  "$JOB_ID" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
source_commit, heldout_path, heldout_sha, tokenizer_path, tokenizer_sha, job_id = sys.argv[3:]
arms = ("iid", "term", "width", "term_width")
paired = {"iid": "term", "term": "iid", "width": "term_width", "term_width": "width"}
source_paths = (
    "pipeline/generate_digitwise_factorial_v4.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "pipeline/test_generate_digitwise_factorial_v4.py",
    "pipeline/audit_digitwise_factorial_v4.py",
    "pipeline/test_audit_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
    "train/sft.py",
    "train/sft_encoding.py",
    "train/test_sft_exact_budget.py",
    "train/jobs/sft_factorial.sbatch",
    "pipeline/jobs/build_digitwise_factorial_v4_stokes.sbatch",
)
generator_sources = {
    "pipeline/generate_digitwise_factorial_v4.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "pipeline/test_generate_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
}
auditor_sources = {
    "pipeline/audit_digitwise_factorial_v4.py",
    "pipeline/test_audit_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
    "train/sft.py",
    "train/sft_encoding.py",
    "train/test_sft_exact_budget.py",
    "train/jobs/sft_factorial.sbatch",
}
receipt_keys = {
    "audit",
    "receipt_schema",
    "schema",
    "declared_arm",
    "admitted_arm",
    "paired_arm",
    "declared_factors",
    "seed",
    "board",
    "board_seed",
    "mode",
    "production_contract",
    "test_scale",
    "mechanical_pass",
    "runtime",
    "test_mechanics_pass",
    "production_admission",
    "admission_pass",
    "checks",
    "failures",
    "failure_examples",
    "target",
    "data",
    "data_sha256",
    "episodes",
    "episodes_sha256",
    "inputs",
    "generator_reports",
    "independent_audit",
    "scientific_contract",
    "source_manifests",
    "heldout",
    "contamination",
    "paired_board",
    "episodes_observed",
    "rows_observed",
    "coverage",
    "terminal_transition_classes",
    "pair_distribution",
    "tokenizer_accounting",
    "iid_semantics",
    "residual_bundle_confound",
    "claim_boundary",
    "auditor_sha256",
    "protocol_sha256",
    "sft_packing_sha256",
    "sft_encoding_sha256",
}
expected_check_keys = {
    "board_terminal_class_contract",
    "complete_episode_row_slots",
    "data_arm_identity",
    "designated_arithmetic_classes",
    "episode_arithmetic_class_coverage",
    "episode_ids",
    "episode_no_blank_lines",
    "episode_operation_allocations",
    "episode_width_allocations",
    "episode_width_operation_allocations",
    "episode_width_position_control_coverage",
    "episode_width_position_coverage",
    "generator_reports_exact_and_current",
    "heldout_frozen_counts",
    "heldout_frozen_regimes",
    "heldout_frozen_sha256",
    "heldout_no_blank_lines",
    "heldout_solver_valid",
    "nonconcentrated_terminal_and_penultimate_pairs",
    "paired_allocation_format_distribution",
    "paired_board_episode_ids",
    "paired_board_literal_bytes",
    "paired_data_arm_identity",
    "paired_digit_pair_distribution",
    "paired_full_visible_operand_tape_features",
    "paired_operand_magnitude_distribution",
    "paired_position_distribution",
    "paired_row_count",
    "paired_row_no_blank_lines",
    "paired_unique_prompts_and_slots",
    "predeclared_control_terminal_contract",
    "raw_episode_count",
    "raw_row_count",
    "row_kind_counts",
    "row_no_blank_lines",
    "row_width_counts",
    "tokenizer_all_rows_accepted",
    "tokenizer_blank_line_parity",
    "tokenizer_canonical_exact_budget",
    "tokenizer_no_blank_lines",
    "tokenizer_no_overlength_rows",
    "tokenizer_no_zero_pack",
    "tokenizer_packing_invariants",
    "tokenizer_row_accounting",
    "transition_arithmetic_class_coverage",
    "transition_position_distribution",
    "transition_terminal_class_contract",
    "transition_width_position_control_coverage",
    "transition_width_position_coverage",
    "unique_complete_episode_signatures",
    "unique_episode_ids",
    "unique_episode_row_slots",
    "unique_episode_signatures",
    "unique_normalized_prompts",
    "valid_episode_count",
    "valid_row_count",
    "zero_train_heldout_exact_normalized_prompt_hits",
    "zero_train_heldout_literal_13gram_hits",
    "zero_train_heldout_reserved_signature_hits",
}


def file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        stat.S_IMODE(value.st_mode),
    )


def capture(path: Path, *, retain: bool = False) -> tuple[dict, bytes | None]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"not a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or file_identity(before) != file_identity(path_before):
            raise SystemExit(f"file changed while opening: {path}")
        digest = hashlib.sha256()
        chunks = [] if retain else None
        size = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        path_after = os.stat(resolved, follow_symlinks=False)
        if file_identity(before) != file_identity(after) or file_identity(before) != file_identity(path_after):
            raise SystemExit(f"file changed while capturing: {path}")
        if size != int(before.st_size):
            raise SystemExit(f"file size changed while capturing: {path}")
        binding = {
            "bytes": size,
            "capture": "one_pass_stable_descriptor_to_immutable_bytes_v1",
            "fd_identity": {
                "device": int(before.st_dev),
                "inode": int(before.st_ino),
                "mtime_ns": int(before.st_mtime_ns),
                "mode": "{:04o}".format(stat.S_IMODE(before.st_mode)),
                "size": int(before.st_size),
            },
            "path": str(resolved),
            "sha256": digest.hexdigest(),
        }
        return binding, b"".join(chunks) if chunks is not None else None
    finally:
        os.close(descriptor)


def compact(binding: dict) -> dict:
    return {"bytes": binding["bytes"], "sha256": binding["sha256"]}


def validate_source_manifest(manifest: dict, expected: set[str], source_bindings: dict) -> None:
    if not isinstance(manifest, dict) or set(manifest.get("sources", {})) != expected:
        raise SystemExit("receipt source manifest set mismatch")
    for relative in expected:
        if manifest["sources"][relative] != compact(source_bindings[relative]):
            raise SystemExit(f"receipt source binding mismatch: {relative}")

source_bindings = {relative: capture(source_root / relative)[0] for relative in source_paths}
heldout_binding = capture(Path(heldout_path))[0]
tokenizer_binding = capture(Path(tokenizer_path))[0]
if heldout_binding["sha256"] != heldout_sha or tokenizer_binding["sha256"] != tokenizer_sha:
    raise SystemExit("frozen input digest mismatch")

records = {}
common_runtime = None
for arm in arms:
    directory = root / arm
    admission_path = directory / "production_admission.json"
    admission_binding, admission_payload = capture(admission_path, retain=True)
    replay_binding, replay_payload = capture(
        directory / "production_admission_replay.json", retain=True
    )
    if replay_payload != admission_payload:
        raise SystemExit(f"{arm} independent admission replay changed")
    admission = json.loads(admission_payload.decode("ascii"))
    canonical = (json.dumps(admission, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if admission_payload != canonical or set(admission) != receipt_keys:
        raise SystemExit(f"{arm} admission is noncanonical or has the wrong complete schema")
    checks = admission.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_check_keys
        or any(value is not True for value in checks.values())
    ):
        raise SystemExit(f"{arm} admission does not preserve the complete passing check set")
    expected_identity = {
        "audit": "digitwise_factorial_v4_admission",
        "receipt_schema": "shohin-factorial-v4-production-admission-v1",
        "schema": "shohin-digitwise-factorial-v4",
        "declared_arm": arm,
        "admitted_arm": arm,
        "paired_arm": paired[arm],
        "mode": "production",
        "production_contract": True,
        "test_scale": None,
        "mechanical_pass": True,
        "test_mechanics_pass": None,
        "production_admission": True,
        "admission_pass": True,
        "failures": {},
        "failure_examples": [],
    }
    if any(admission.get(key) != value for key, value in expected_identity.items()):
        raise SystemExit(f"{arm} admission identity or pass state is invalid")
    if common_runtime is None:
        common_runtime = admission["runtime"]
    elif admission["runtime"] != common_runtime:
        raise SystemExit("auditor runtime changed between arms")

    input_paths = {
        "data": directory / "data.jsonl",
        "episodes": directory / "episodes.jsonl",
        "paired_data": root / paired[arm] / "data.jsonl",
        "paired_episodes": root / paired[arm] / "episodes.jsonl",
        "heldout": Path(heldout_path),
        "tokenizer": Path(tokenizer_path),
    }
    if set(admission.get("inputs", {})) != set(input_paths):
        raise SystemExit(f"{arm} receipt input set mismatch")
    input_bindings = {name: capture(path)[0] for name, path in input_paths.items()}
    for name, binding in input_bindings.items():
        if admission["inputs"][name] != binding:
            raise SystemExit(f"{arm} receipt no longer binds {name}")
    if admission["data_sha256"] != input_bindings["data"]["sha256"]:
        raise SystemExit(f"{arm} top-level data digest mismatch")
    if admission["episodes_sha256"] != input_bindings["episodes"]["sha256"]:
        raise SystemExit(f"{arm} top-level episode digest mismatch")

    reports = admission.get("generator_reports")
    if not isinstance(reports, dict) or set(reports) != {"primary", "counterpart"}:
        raise SystemExit(f"{arm} generator evidence set mismatch")
    for role, report_arm in (("primary", arm), ("counterpart", paired[arm])):
        generated_binding = capture(root / report_arm / "generated.json")[0]
        evidence = reports[role]
        if evidence.get("validated") is not True or evidence.get("bytes") != generated_binding["bytes"] or evidence.get("sha256") != generated_binding["sha256"]:
            raise SystemExit(f"{arm} generator report binding mismatch: {role}")
        validate_source_manifest(evidence.get("source_manifest"), generator_sources, source_bindings)

    manifests = admission.get("source_manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"generator", "auditor"}:
        raise SystemExit(f"{arm} source manifest set mismatch")
    validate_source_manifest(manifests["generator"], generator_sources, source_bindings)
    validate_source_manifest(manifests["auditor"], auditor_sources, source_bindings)
    if admission.get("heldout", {}).get("sha256") != heldout_sha:
        raise SystemExit(f"{arm} frozen heldout binding mismatch")

    exact = admission.get("tokenizer_accounting", {}).get("canonical_exact_budget", {})
    expected = {
        "optimizer_updates": 1560,
        "batch_size": 16,
        "consumed_packs": 24960,
        "pack_len": 2048,
        "forward_token_positions": 51118080,
    }
    if any(exact.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"{arm} exact budget differs from the frozen contract")
    records[arm] = {}
    for name in (
        "data.jsonl",
        "episodes.jsonl",
        "generated.json",
        "production_admission.json",
        "production_admission_replay.json",
        "generator.stdout",
        "auditor.stdout",
        "auditor_replay.stdout",
    ):
        if name == "production_admission.json":
            binding = admission_binding
        elif name == "production_admission_replay.json":
            binding = replay_binding
        else:
            binding = capture(directory / name)[0]
        records[arm][name] = compact(binding)

manifest = {
    "schema": "shohin-factorial-v4-production-set-v1",
    "slurm_job_id": int(job_id),
    "source_commit": source_commit,
    "source_files": {relative: compact(source_bindings[relative]) for relative in source_paths},
    "inputs": {
        "heldout": compact(heldout_binding),
        "tokenizer": compact(tokenizer_binding),
    },
    "arms": records,
    "all_arms_admitted": True,
    "exact_budget_per_arm": {
        "optimizer_updates": 1560,
        "batch_size": 16,
        "consumed_packs": 24960,
        "pack_len": 2048,
        "forward_token_positions": 51118080,
    },
}
payload = (json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
path = root / "production_manifest.json"
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
try:
    with os.fdopen(fd, "wb", closefd=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
finally:
    os.close(fd)
PY

find "$partial" -type f -exec chmod 0444 {} +
find "$partial" -type d -exec chmod 0555 {} +
verify_frozen_tree() {
  "$PY" - "$1" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
manifest_path = root / "production_manifest.json"
raw = manifest_path.read_bytes()
manifest = json.loads(raw)
canonical = (json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
if raw != canonical or manifest.get("schema") != "shohin-factorial-v4-production-set-v1":
    raise SystemExit("production manifest is not canonical")
if stat.S_IMODE(root.stat().st_mode) != 0o555:
    raise SystemExit("published root has wrong mode")

def sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()

for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"published tree contains a symlink: {path}")
    expected_mode = 0o555 if path.is_dir() else 0o444
    if stat.S_IMODE(path.stat().st_mode) != expected_mode:
        raise SystemExit(f"published tree has wrong mode: {path}")
for arm, files in manifest["arms"].items():
    for name, expected in files.items():
        size, digest = sha256(root / arm / name)
        if expected != {"bytes": size, "sha256": digest}:
            raise SystemExit(f"published arm bytes changed: {arm}/{name}")
for relative, expected in manifest["source_files"].items():
    size, digest = sha256(root / "snapshot" / "source" / relative)
    if expected != {"bytes": size, "sha256": digest}:
        raise SystemExit(f"published source bytes changed: {relative}")
for name, expected in manifest["inputs"].items():
    suffix = "heldout.jsonl" if name == "heldout" else "tokenizer.json"
    size, digest = sha256(root / "snapshot" / "input" / suffix)
    if expected != {"bytes": size, "sha256": digest}:
        raise SystemExit(f"published input bytes changed: {name}")
PY
}
verify_frozen_tree "$partial"
[ ! -e "$OUT_ROOT" ] && [ ! -L "$OUT_ROOT" ] || {
  echo "[factorial-v4] final output appeared during build" >&2
  exit 8
}
mv -T --no-clobber "$partial" "$OUT_ROOT"
[ ! -e "$partial" ] || {
  echo "[factorial-v4] no-clobber publication refused an existing final output" >&2
  exit 8
}
verify_frozen_tree "$OUT_ROOT"
printf '[factorial-v4] published %s\n' "$OUT_ROOT"
sha256sum "$OUT_ROOT/production_manifest.json"
