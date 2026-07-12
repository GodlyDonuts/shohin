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
# ``train.py --steps`` is an absolute global target. The narrow phase ends at
# 300k, so this language-balanced continuation must target 600k.
STEPS=${STEPS:-600000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-0.001}
DSEED=${DSEED:-20260714}
CKPT=${CKPT:-250}
BS=${BS:-32}
ACC=${ACC:-8}
FINEWEB_DIR=${FINEWEB_DIR:-$BASE/artifacts/shards/fineweb_edu_25b}
DCLM_DIR=${DCLM_DIR:-$BASE/artifacts/shards/dclm_baseline_25b}

SHARDS=(
  "$BASE/artifacts/shards/openwebmath"
  "$BASE/artifacts/shards/code_python"
  "$BASE/artifacts/shards/finemath3"
  "$BASE/artifacts/shards/openmath_pt"
  "$FINEWEB_DIR"
  "$DCLM_DIR"
)
# FineMath-4 is contained in FineMath-3, so it is excluded instead of replayed.
# Both language sources are scanned 25B replacements, never their 5B pilots.
# Order: openwebmath, code, finemath3, openmath, fineweb, dclm.
WEIGHTS=(5 27 6 8 27 27)
# With the six-domain BS=32 floor, this yields about 24.8% math, 25.1% code,
# and 50.1% educational English.

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
