#!/bin/bash
#SBATCH -J shohin-mame-loop
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/mame_loop_%j.out
#SBATCH --open-mode=append
# Train a short, isolated recurrent-depth proxy. It is a stability/throughput
# canary only: no flagship checkpoint, output directory, or shard list is used
# for continuation. Compare NLOOP=1 and NLOOP=2 under the same update/token setup.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
NLOOP=${NLOOP:-2}
STEPS=${STEPS:-800}
BS=${BS:-32}
ACC=${ACC:-8}
SEED=${SEED:-20260712}
OUT=${OUT:-$BASE/train/mame_loop${NLOOP}_s${STEPS}_seed${SEED}}
SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"

[ "$NLOOP" = "1" ] || [ "$NLOOP" = "2" ] || { echo "[mame-loop] NLOOP must be 1 or 2" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[mame-loop] refusing to overwrite $OUT" >&2; exit 3; }

cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[mame-loop] CUDA unavailable after allocation" >&2
  exit 75
fi

"$PY" train.py --size mame --n-loop "$NLOOP" --shard-dirs $SHARDS \
  --steps "$STEPS" --batch-size "$BS" --grad-accum "$ACC" \
  --warmup 100 --lr-muon 0.02 --lr-adam 3e-3 --data-seed "$SEED" \
  --log-every 20 --ckpt-every 400 --out "$OUT" --compile
echo "[mame-loop] done n_loop=$NLOOP out=$OUT"
