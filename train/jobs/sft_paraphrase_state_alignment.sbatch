#!/bin/bash
# Isolated paraphrase-state-alignment SFT. Never writes a flagship path.
# The same/mismatch/none modes are distinct controlled experiments.
#SBATCH -J shohin-psa
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/psa_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.jsonl}
DATA_SHA=4363101c9773b055e24bce3e79f727f3c103a0fb357a6820206ddeab2567234f
OUT=${OUT:?set OUT to a fresh isolated output directory}
MODE=${MODE:-same}
ALIGN_WEIGHT=${ALIGN_WEIGHT:-0.05}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0}
CONTRASTIVE_TEMPERATURE=${CONTRASTIVE_TEMPERATURE:-0.1}
CAPTURE_LAYER=${CAPTURE_LAYER:-19}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
PAIR_BS=${PAIR_BS:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[ -f "$INIT" ] || { echo "[psa] init checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[psa] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[psa] refusing existing output: $OUT" >&2; exit 4; }
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[psa] data hash mismatch" >&2; exit 5; }
case "$MODE" in same|mismatch|none) ;; *) echo "[psa] invalid mode: $MODE" >&2; exit 6;; esac
if [ "$MODE" = none ] && { [ "$ALIGN_WEIGHT" != 0 ] || [ "$CONTRASTIVE_WEIGHT" != 0 ]; }; then
  echo "[psa] CE-only mode requires zero alignment weights" >&2
  exit 7
fi
if [ "$MODE" != same ] && [ "$CONTRASTIVE_WEIGHT" != 0 ]; then
  echo "[psa] contrastive alignment requires MODE=same" >&2
  exit 8
fi

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[psa] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
echo "[psa] init=$INIT data=$DATA mode=$MODE align=$ALIGN_WEIGHT contrastive=$CONTRASTIVE_WEIGHT layer=$CAPTURE_LAYER out=$OUT"
ARGS=()
if [ "$MAX_EXAMPLES" -gt 0 ]; then ARGS+=(--max-examples "$MAX_EXAMPLES"); fi
"$PY" sft_paraphrase_state_alignment.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BS" --pair-batch-size "$PAIR_BS" --capture-layer "$CAPTURE_LAYER" \
  --align-weight "$ALIGN_WEIGHT" --contrastive-weight "$CONTRASTIVE_WEIGHT" \
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE" --pair-mode "$MODE" "${ARGS[@]}"
test -s "$OUT/psa_ep${EPOCHS}.pt"
echo "[psa] wrote $OUT/psa_ep${EPOCHS}.pt"
