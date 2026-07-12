#!/bin/bash
#SBATCH -J shohin-language-relaunch
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/language_relaunch_%j.out
#SBATCH --open-mode=append
# Future-only curriculum handoff for the single-H100 flagship.
#
# The active pretrain is 75% math-oriented text / 25% raw code and has no broad
# English source.  This recipe restores a language floor without dropping the
# math/code specialization.  It must be submitted only from an explicitly
# preserved checkpoint after every source's manifest and content audit passes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=$BASE/train/flagship_out
RESUME=${RESUME:?set RESUME to an explicitly preserved numbered checkpoint}
STEPS=${STEPS:-300000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-0.001}
DSEED=${DSEED:-20260714}
CKPT=${CKPT:-250}
BS=${BS:-32}
ACC=${ACC:-8}
DCLM_DIR=${DCLM_DIR:-$BASE/artifacts/shards/dclm_baseline_5b}

SHARDS=(
  "$BASE/artifacts/shards/finemath4"
  "$BASE/artifacts/shards/openwebmath"
  "$BASE/artifacts/shards/code_python"
  "$BASE/artifacts/shards/finemath3"
  "$BASE/artifacts/shards/openmath_pt"
  "$BASE/artifacts/shards/fineweb_edu_5b"
  "$DCLM_DIR"
)
# FineMath4 is retained for diversity but deliberately no longer replayed at the
# old equal-domain rate. DCLM_DIR can point to a scanned larger replacement such
# as dclm_baseline_25b; never include both the pilot and its replacement.
WEIGHTS=(2 4 28 4 6 32 24)
# ShardLoader reserves one of BS=32 sequences for each of these seven domains,
# then fills the remaining 25 by WEIGHTS. These values therefore yield the
# intended effective mix: math 25%, code 25%, English 50%.

[ -f "$RESUME" ] || { echo "[language-relaunch] resume checkpoint missing: $RESUME" >&2; exit 2; }
LATEST=$(ls "$OUT"/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1 || true)
[ -n "$LATEST" ] || { echo "[language-relaunch] no numbered checkpoint in $OUT" >&2; exit 2; }
[ "$(realpath "$LATEST")" = "$(realpath "$RESUME")" ] || {
  echo "[language-relaunch] RESUME must be the latest numbered checkpoint: $RESUME != $LATEST" >&2
  exit 2
}
for shard in "${SHARDS[@]}"; do
  [ -s "$shard/manifest.json" ] || {
    echo "[language-relaunch] manifest missing or unapproved: $shard" >&2
    exit 3
  }
done

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[language-relaunch] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[language-relaunch] resume=$RESUME steps=$STEPS bs=$BS acc=$ACC weights=${WEIGHTS[*]}"
"$PY" train.py --size shohin --shard-dirs "${SHARDS[@]}" --domain-weights "${WEIGHTS[@]}" \
  --steps "$STEPS" --batch-size "$BS" --grad-accum "$ACC" \
  --warmup "$WARMUP" --lr-muon "$LRMUON" --lr-adam "$LRADAM" --data-seed "$DSEED" \
  --log-every 10 --ckpt-every "$CKPT" --gnorm-mult 8 --out "$OUT" \
  --resume --fresh-opt --compile
echo "[language-relaunch] completed"
