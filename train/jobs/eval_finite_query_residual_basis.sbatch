#!/bin/bash
# Read-only FQRB evaluation. It requires an isolated FQRB checkpoint or an
# explicit immutable raw baseline, and writes only a fresh FQRB eval report.
#SBATCH -J shohin-fqrb-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fqrb_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated train/fqrb_* checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/finite_query_residual_basis_v1_heldout.jsonl}
DATA_SHA=${DATA_SHA:-832da1a5264dbd0ec9ccf5b8515276794d545531da5f3dbdacdf4c239452ef97}
OUT=${OUT:?set OUT to a fresh artifacts/eval_history/fqrb_*.json path}
GROUPS=${GROUPS:-500}
MAX_NEW=${MAX_NEW:-12}
SPLIT=${SPLIT:-heldout}
ALLOW_RAW=${ALLOW_RAW:-0}
LAYER=${LAYER:-19}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$CKPT" ] || { echo "[fqrb-eval] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[fqrb-eval] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[fqrb-eval] refusing existing output" >&2; exit 4; }
case "$CKPT" in
  "$BASE"/train/fqrb_*/cra_ep1.pt) ;;
  "$BASE"/train/flagship_out/best_step200000.pt)
    [ "$ALLOW_RAW" = 1 ] || { echo "[fqrb-eval] raw checkpoint requires ALLOW_RAW=1" >&2; exit 5; }
    ;;
  *) echo "[fqrb-eval] checkpoint must be isolated FQRB output or explicit immutable raw baseline" >&2; exit 5;;
esac
case "$OUT" in "$BASE"/artifacts/eval_history/fqrb_*.json) ;; *) echo "[fqrb-eval] OUT must be FQRB eval-history JSON" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[fqrb-eval] held-out hash mismatch" >&2; exit 7; }

if [ "$ALLOW_RAW" != 1 ]; then
  "$PY" - "$CKPT" "$DATA_SHA" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu')
metadata = checkpoint.get('counterfactual_residual_algebra', {})
if metadata.get('data_sha256') != '6d42ce87a202f293b64708e8bbca2193fb90f3cb60c232f6c968b0edaa46f113':
    raise SystemExit('[fqrb-eval] checkpoint does not bind FQRB train data')
if metadata.get('source_present_at_suffix') is not False or metadata.get('composition') != 'donor + edited - base':
    raise SystemExit('[fqrb-eval] checkpoint does not certify native source-free composition')
print('[fqrb-eval] checkpoint-bound FQRB admission passed')
PY
fi

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[fqrb-eval] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
RAW_ARGS=()
if [ "$ALLOW_RAW" = 1 ]; then RAW_ARGS=(--allow-raw --layer "$LAYER"); fi
echo "[fqrb-eval] ckpt=$CKPT data=$DATA split=$SPLIT groups=$GROUPS out=$OUT allow_raw=$ALLOW_RAW"
"$PY" eval_finite_query_residual_basis.py \
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --split "$SPLIT" --max-groups "$GROUPS" --max-new "$MAX_NEW" "${RAW_ARGS[@]}"
test -s "$OUT"
echo "[fqrb-eval] wrote $OUT"
