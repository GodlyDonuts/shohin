#!/bin/bash
# Read-only ECLI evaluation. The generic finite-query evaluator enforces the
# additional codebook-swap path when the frozen data contains that field.
#SBATCH -J shohin-ecli-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ecli_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set ECLI checkpoint}
DATA=${DATA:?set frozen ECLI evaluation data}
DATA_SHA=${DATA_SHA:?set expected ECLI evaluation SHA-256}
OUT=${OUT:?set fresh evaluation report path}
SPLIT=${SPLIT:-heldout}
GROUPS=${GROUPS:-500}
MAX_NEW=${MAX_NEW:-12}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -s "$CKPT" ] || { echo "[ecli-eval] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[ecli-eval] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[ecli-eval] refusing existing output" >&2; exit 4; }
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[ecli-eval] data hash mismatch" >&2; exit 5; }

"$PY" - "$CKPT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu')
metadata = checkpoint.get('ephemeral_codebook_fqrb')
if not isinstance(metadata, dict) or metadata.get('mechanism') != 'ephemeral_codebook_fqrb_v1':
    raise SystemExit('[ecli-eval] checkpoint is not an ECLI model')
PY

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
echo "[ecli-eval] ckpt=$CKPT data=$DATA split=$SPLIT groups=$GROUPS max_new=$MAX_NEW out=$OUT"
"$PY" eval_finite_query_residual_basis.py \
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --split "$SPLIT" --max-groups "$GROUPS" --max-new "$MAX_NEW"
test -s "$OUT"
echo "[ecli-eval] wrote $OUT"
