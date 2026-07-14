#!/bin/bash
#SBATCH -J shohin-jacobian-readout
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/jacobian_readout_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
LENS_A=${LENS_A:?set first independent Jacobian lens}
LENS_B=${LENS_B:?set second independent Jacobian lens}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
OUT=${OUT:?set a fresh readout report}

for path in "$INIT" "$TOKENIZER" "$LENS_A" "$LENS_B" "$DATA"; do
  [ -s "$path" ] || { echo "[jacobian-readout] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[jacobian-readout] refusing existing output" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/diagnostics/jacobian_readout_*.json) ;;
  *) echo "[jacobian-readout] invalid output path" >&2; exit 4;;
esac

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize()
except Exception as exc:
    print("[jacobian-readout] CUDA preflight failed: {}".format(exc),file=sys.stderr)
    sys.exit(75)
PY

"$PY" eval_jacobian_workspace.py \
  --base "$INIT" --tokenizer "$TOKENIZER" --lens-a "$LENS_A" --lens-b "$LENS_B" \
  --data "$DATA" --out "$OUT" --batch-size "${BATCH_SIZE:-16}"
