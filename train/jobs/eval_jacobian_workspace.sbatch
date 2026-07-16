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
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
LENS_A=${LENS_A:?set first independent Jacobian lens}
LENS_B=${LENS_B:?set second independent Jacobian lens}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
OUT=${OUT:?set a fresh readout report}
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/miniforge-torch-runtime-v1.tar}
RUNTIME_TAR_SHA256=af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c

for path in "$INIT" "$TOKENIZER" "$LENS_A" "$LENS_B" "$DATA" "$RUNTIME_TAR"; do
  [ -s "$path" ] || { echo "[jacobian-readout] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[jacobian-readout] refusing existing output" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/diagnostics/jacobian_readout_*.json) ;;
  *) echo "[jacobian-readout] invalid output path" >&2; exit 4;;
esac

OBSERVED_RUNTIME_SHA256=$(sha256sum "$RUNTIME_TAR" | awk '{print $1}')
[ "$OBSERVED_RUNTIME_SHA256" = "$RUNTIME_TAR_SHA256" ] || {
  echo "[jacobian-readout] runtime SHA-256 mismatch" >&2
  exit 5
}
RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
RUNTIME_DIR=$(mktemp -d "$RUNTIME_PARENT/shohin_jacobian_readout_runtime.${SLURM_JOB_ID:-manual}.XXXXXX")
chmod 0700 "$RUNTIME_DIR"
cleanup_runtime() {
  chmod -R u+w "${RUNTIME_DIR:-}" 2>/dev/null || true
  rm -rf -- "${RUNTIME_DIR:-}"
}
trap cleanup_runtime EXIT
START=$(date +%s)
tar -C "$RUNTIME_DIR" -xf "$RUNTIME_TAR"
PY=$RUNTIME_DIR/miniforge3/bin/python
[ -x "$PY" ] || { echo "[jacobian-readout] extracted Python missing" >&2; exit 6; }
export PYTHONNOUSERSITE=1
echo "[jacobian-readout] runtime staged sha256=$OBSERVED_RUNTIME_SHA256 seconds=$(( $(date +%s) - START ))"

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
