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
STEPS=${STEPS:-300000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-1e-3}
DSEED=${DSEED:-20260713}
CKPT=${CKPT:-250}
BS=${BS:-32}
ACC=${ACC:-8}
SHARDS=(
  "$BASE/artifacts/shards/finemath4"
  "$BASE/artifacts/shards/openwebmath"
  "$BASE/artifacts/shards/code_python"
  "$BASE/artifacts/shards/finemath3"
  "$BASE/artifacts/shards/openmath_pt"
  "$BASE/artifacts/shards/fineweb_edu_5b"
)
# fine4=5%, openwebmath=12%, code=25%, fine3=13%, openmath=20%, fineweb-edu=25%.
# The loader retains every enabled domain in each batch before weighted remainder.
WEIGHTS=(5 12 25 13 20 25)

for shard in "${SHARDS[@]}"; do
  [ -s "$shard/manifest.json" ] || { echo "[relaunch] missing manifest: $shard" >&2; exit 2; }
done
LATEST=$(ls -t "$OUT"/ckpt_[0-9]*.pt 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] || { echo "[relaunch] no numbered resume checkpoint in $OUT" >&2; exit 3; }

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

echo "[relaunch] resume=$LATEST steps=$STEPS bs=$BS acc=$ACC weights=${WEIGHTS[*]}"
"$PY" train.py --size shohin --shard-dirs "${SHARDS[@]}" --domain-weights "${WEIGHTS[@]}" \
  --steps "$STEPS" --batch-size "$BS" --grad-accum "$ACC" \
  --warmup "$WARMUP" --lr-muon "$LRMUON" --lr-adam "$LRADAM" --data-seed "$DSEED" \
  --log-every 10 --ckpt-every "$CKPT" --gnorm-mult 8 --out "$OUT" --resume --fresh-opt --compile
echo "[relaunch] completed"
