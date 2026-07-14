#!/bin/bash
#SBATCH -J shohin-sft-retain
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_retention_%j.out
#SBATCH --open-mode=append
# Isolated behavior-preserving direct-skill SFT. This never changes the
# flagship or any earlier SFT artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=(
  "$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl"
  "$BASE/artifacts/sft/primitives_v1_train.jsonl"
)
INIT=${INIT:?set INIT to immutable best_step200000.pt}
REFERENCE=${REFERENCE:-$INIT}
REPLAY=${REPLAY:?set REPLAY to a frozen prompt-only retention JSONL}
REPLAY_SHA=${REPLAY_SHA:?set REPLAY_SHA to the expected SHA-256}
OUT=${OUT:?set OUT to a fresh train/sft_retention_* directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.001}
LRADAM=${LRADAM:-0.00025}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.25}
REPLAY_BATCH=${REPLAY_BATCH:-4}
REPLAY_MAX_TOKENS=${REPLAY_MAX_TOKENS:-128}
# This favors compact solver-verified operations while retaining broad
# math/procedural/code pressure. The prompt-only KL is the behavior guard.
WEIGHTS="math=0.35 procedural=0.20 code=0.10 teacher=0.05 primitives=0.30"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-retain] data missing: $data" >&2; exit 2; }
done
for path in "$INIT" "$REFERENCE" "$REPLAY"; do
  [ -s "$path" ] || { echo "[sft-retain] required file missing: $path" >&2; exit 3; }
done
[ "$(sha256sum "$REPLAY" | awk '{print $1}')" = "$REPLAY_SHA" ] || {
  echo "[sft-retain] replay hash mismatch" >&2; exit 4;
}
[ ! -e "$OUT" ] || { echo "[sft-retain] refusing existing output: $OUT" >&2; exit 5; }
case "$OUT" in "$BASE"/train/sft_retention_*) ;; *)
  echo "[sft-retain] OUT must be an isolated train/sft_retention_* directory" >&2; exit 6;;
esac

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    del probe
except Exception as exc:
    print(f"[sft-retain] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "[sft-retain] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[sft-retain] init=$INIT reference=$REFERENCE replay=$REPLAY out=$OUT epochs=$EPOCHS weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt --sample-weights $WEIGHTS \
  --reference "$REFERENCE" --replay-prompts "$REPLAY" --replay-weight "$REPLAY_WEIGHT" \
  --replay-batch-size "$REPLAY_BATCH" --replay-max-tokens "$REPLAY_MAX_TOKENS" \
  --freeze-lexicon --compile --out "$OUT"
echo "[sft-retain] done"
