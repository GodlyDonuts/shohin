#!/bin/bash
#SBATCH -J shohin-reasoning-relaunch
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/reasoning_relaunch_%j.out
#SBATCH --open-mode=append
# Future-only curriculum handoff. Submit only after the active flagship exits
# and every manifest below has passed its own artifact-quality gate. It resumes
# the newest numbered checkpoint in flagship_out with a fresh optimizer because
# the data distribution intentionally changes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=$BASE/train/flagship_out
RESUME=${RESUME:?set RESUME to an explicitly preserved newest numbered checkpoint}
# ``train.py --steps`` is an absolute global target. The active narrow-data
# phase ends at 300k, so the first language-balanced continuation must target
# 600k or it would immediately exit without taking an update.
STEPS=${STEPS:-600000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-1e-3}
DSEED=${DSEED:-20260713}
CKPT=${CKPT:-250}
BS=${BS:-32}
ACC=${ACC:-8}
SHARDS=(
  "$BASE/artifacts/shards/openwebmath"
  "$BASE/artifacts/shards/code_python"
  "$BASE/artifacts/shards/finemath3"
  "$BASE/artifacts/shards/openmath_pt"
  "$BASE/artifacts/shards/fineweb_edu_25b"
  "$BASE/artifacts/shards/dclm_baseline_25b"
)
FINEWEB_APPROVAL=${FINEWEB_APPROVAL:-$BASE/artifacts/shard_scans/fineweb_edu_25b.approved.json}
DCLM_APPROVAL=${DCLM_APPROVAL:-$BASE/artifacts/shard_scans/dclm_baseline_25b.approved.json}
# FineMath-4 is a subset of FineMath-3, so it is deliberately excluded rather
# than replayed as a separate narrow-math domain. With the loader's per-domain
# floor at BS=32, these weights yield about 24.8% math, 25.1% code, and 50.1%
# educational English. Both language sources are scanned 25B replacements,
# never their 5B pilots.
# Order: openwebmath, code, finemath3, openmath, fineweb-edu, dclm.
WEIGHTS=(5 27 6 8 27 27)

for shard in "${SHARDS[@]}"; do
  [ -s "$shard/manifest.json" ] || { echo "[relaunch] missing manifest: $shard" >&2; exit 2; }
done
LATEST=$(ls "$OUT"/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1 || true)
[ -n "$LATEST" ] || { echo "[relaunch] no numbered resume checkpoint in $OUT" >&2; exit 3; }
[ "$(realpath "$LATEST")" = "$(realpath "$RESUME")" ] || {
  echo "[relaunch] RESUME must be the latest numbered checkpoint: $RESUME != $LATEST" >&2
  exit 3
}
for approval in "$FINEWEB_APPROVAL" "$DCLM_APPROVAL"; do
  [ -s "$approval" ] || { echo "[relaunch] missing scan approval: $approval" >&2; exit 4; }
done
"$PY" - "$BASE" "$FINEWEB_APPROVAL" "$DCLM_APPROVAL" <<'PY'
import hashlib
import json
import os
import sys

base = sys.argv[1]
targets = [
    (os.path.join(base, "artifacts/shards/fineweb_edu_25b"), sys.argv[2]),
    (os.path.join(base, "artifacts/shards/dclm_baseline_25b"), sys.argv[3]),
]
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for shard, approval_path in targets:
    approval = json.load(open(approval_path))
    manifest = os.path.join(shard, "manifest.json")
    if approval.get("schema") != "shohin-shard-scan-approval-v1":
        raise SystemExit(f"invalid scan approval schema: {approval_path}")
    if os.path.realpath(approval.get("shard_dir", "")) != os.path.realpath(shard):
        raise SystemExit(f"approval source mismatch: {approval_path}")
    if approval.get("manifest_sha256") != sha256(manifest):
        raise SystemExit(f"manifest changed after approval: {manifest}")
    report = approval.get("scan_report")
    if not report or not os.path.isfile(report) or approval.get("scan_report_sha256") != sha256(report):
        raise SystemExit(f"scan report changed or missing: {approval_path}")
    if int(approval.get("total_tokens", 0)) < 24_500_000_000:
        raise SystemExit(f"language source too small for relaunch: {approval_path}")
    if float(approval.get("max_bytef", 1.0)) > 0.01:
        raise SystemExit(f"byte fallback above admission bound: {approval_path}")
print("[relaunch] scan approvals verified")
PY

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[relaunch] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[relaunch] resume=$RESUME steps=$STEPS bs=$BS acc=$ACC weights=${WEIGHTS[*]}"
"$PY" train.py --size shohin --shard-dirs "${SHARDS[@]}" --domain-weights "${WEIGHTS[@]}" \
  --steps "$STEPS" --batch-size "$BS" --grad-accum "$ACC" \
  --warmup "$WARMUP" --lr-muon "$LRMUON" --lr-adam "$LRADAM" --data-seed "$DSEED" \
  --log-every 10 --ckpt-every "$CKPT" --gnorm-mult 8 --out "$OUT" --resume --fresh-opt --compile
echo "[relaunch] completed"
