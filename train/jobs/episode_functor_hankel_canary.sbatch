#!/bin/bash
#SBATCH -J efc-hsc-canary
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/efc_hsc_canary_%j.out
#SBATCH --open-mode=append
#
# Two-update measurement canary only. It does not save weights, authorize an
# HSC qualification fit, touch flagship output, or start continuation pretrain.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
AUTH=$BASE/artifacts/r12/episode_functor_hankel_qualification_v6_20260724
PACKAGE=$AUTH/canary_package
SOURCE_MANIFEST=$AUTH/source_manifest.json
CHECKPOINT=$BASE/train/flagship_out/ckpt_0300000.pt
TOKENIZER=$BASE/artifacts/tokenizer/tokenizer.json
DENY_PROBE=$BASE/AGENT_RUNBOOK.md
RESULT_ROOT=$BASE/artifacts/r12/episode_functor_hankel_canary_results
OUTPUT=$RESULT_ROOT/prefix_${SLURM_JOB_ID}
POLICY=$RESULT_ROOT/.prefix_${SLURM_JOB_ID}.landlock.json
SNAPSHOT=${SLURM_TMPDIR:?}/efc_hsc_source

EXPECTED_SOURCE_MANIFEST_SHA256=ae4357df26353bbc6d8703f6074f133fc3991d00e751c5a22627aa2c387be64f
EXPECTED_PACKAGE_SHA256=ed024dd9c47e766871a7253c1117a4f5b7c3ac551a1f08153618854091164095
EXPECTED_CHECKPOINT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
EXPECTED_TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
EXPECTED_ARM_RECEIPT_SHA256=67bf59d568e64256939ebb74acdca04c6cc16bb7e71016616db315094fabc8b2
EXPECTED_INITIALIZATION_RECEIPT_SHA256=333a53adbdcda7224b070c27fcc76649049b048997d3e6e0883e0d18f891a889
EXPECTED_CANARY_RECEIPT_SHA256=1b14ee63add971f9022377847c8d9a1fe53911938bb65ee2cf693e0dd81a6cf4
SOURCE_COMMIT=db5786cf37390099db62acdbeba51cd9d796cb52

mkdir -p "$RESULT_ROOT" "$SNAPSHOT"
test ! -e "$OUTPUT"
test ! -e "$POLICY"
test -f "$CHECKPOINT"
test -f "$TOKENIZER"
test -d "$PACKAGE"

# Build a closed Python source snapshot from the hash-bound manifest. The
# confined role later repeats the same closure and hash verification.
"$PYTHON" - "$BASE" "$SOURCE_MANIFEST" "$SNAPSHOT" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
snapshot = Path(sys.argv[3]).resolve()
expected = sys.argv[4]
source_commit = sys.argv[5]
observed_commit = subprocess.run(
    ["git", "-C", str(base), "rev-parse", f"{source_commit}^{{commit}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if observed_commit != source_commit:
    raise SystemExit("source commit differs")
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
canonical = json.dumps(
    tuple(manifest["sources"]),
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
if manifest["manifest_sha256"] != expected:
    raise SystemExit("source manifest commitment differs")
if hashlib.sha256(canonical).hexdigest() != expected:
    raise SystemExit("source manifest payload differs")
for row in manifest["sources"]:
    payload = subprocess.run(
        ["git", "-C", str(base), "show", f"{source_commit}:{row['path']}"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise SystemExit(f"source hash differs: {row['path']}")
    destination = snapshot / row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    destination.chmod(0o444)
for directory in (snapshot / "pipeline", snapshot / "train"):
    directory.chmod(0o555)
snapshot.chmod(0o555)
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SNAPSHOT:$SNAPSHOT/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/tmp/efc_hsc_torch_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/efc_hsc_triton_${SLURM_JOB_ID}

echo "[hsc-canary] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
command -v unshare >/dev/null

# A missing unprivileged network namespace or Landlock ABI is a clean NO-GO.
# The role receives only its source snapshot, four train sources, immutable
# checkpoint/tokenizer, runtime libraries/devices, and a fresh output parent.
unshare --user --map-root-user --net \
  env SHOHIN_NETWORK_NAMESPACE_ISOLATED=1 \
  "$PYTHON" "$SNAPSHOT/train/landlock_stage_exec.py" \
    --stage efc-hsc-canary \
    --ro "$SNAPSHOT" \
    --ro "$PACKAGE" \
    --ro "$SOURCE_MANIFEST" \
    --ro "$CHECKPOINT" \
    --ro "$TOKENIZER" \
    --ro "$BASE/miniforge3" \
    --ro /usr \
    --ro /proc \
    --ro /sys \
    --rw /dev \
    --rw /tmp \
    --rw "$RESULT_ROOT" \
    --policy-receipt "$POLICY" \
    -- \
    "$PYTHON" "$SNAPSHOT/train/run_episode_functor_hankel_canary.py" \
      --source-root "$SNAPSHOT" \
      --source-manifest "$SOURCE_MANIFEST" \
      --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
      --package "$PACKAGE" \
      --expected-package-sha256 "$EXPECTED_PACKAGE_SHA256" \
      --checkpoint "$CHECKPOINT" \
      --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
      --tokenizer "$TOKENIZER" \
      --expected-tokenizer-sha256 "$EXPECTED_TOKENIZER_SHA256" \
      --incidence-mode prefix \
      --decode-mode hankel-shift \
      --expected-arm-receipt-sha256 "$EXPECTED_ARM_RECEIPT_SHA256" \
      --expected-initialization-receipt-sha256 \
        "$EXPECTED_INITIALIZATION_RECEIPT_SHA256" \
      --expected-canary-receipt-sha256 "$EXPECTED_CANARY_RECEIPT_SHA256" \
      --deny-probe "$DENY_PROBE" \
      --output "$OUTPUT"

test -f "$OUTPUT/canary_report.json"
test -f "$OUTPUT/COMPLETE.json"
echo "[hsc-canary] measurement report published at $OUTPUT"
