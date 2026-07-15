#!/bin/bash
# Isolated one-epoch static-tape/recurrent-register SFT. This is deliberately
# staged only: data must be generated and independently admitted before submit.
# It never reads or writes the flagship output directory.
#SBATCH -J shohin-factor-sft
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_factor_sft_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4

INIT=${INIT:?set INIT to an immutable raw checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/digitwise_factor_v1_train.jsonl}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_factor_v1_heldout.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/digitwise_factor_v1_train.protocol_audit.json}
OUT=${OUT:?set OUT to a fresh isolated directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.001}
LRADAM=${LRADAM:-0.0002}
COMPILE=${COMPILE:-1}

[ -f "$INIT" ] || { echo "[factor-sft] init missing: $INIT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[factor-sft] data missing: $DATA" >&2; exit 3; }
[ -s "$EPISODES" ] || { echo "[factor-sft] heldout missing: $EPISODES" >&2; exit 4; }
[ -s "$AUDIT" ] || { echo "[factor-sft] audit missing: $AUDIT" >&2; exit 5; }
[ ! -e "$OUT" ] || { echo "[factor-sft] refusing existing output: $OUT" >&2; exit 6; }

"$PY" - "$DATA" "$EPISODES" "$AUDIT" <<'PY'
import hashlib
import json
import sys

data, episodes, audit_path = sys.argv[1:]

def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

audit = json.load(open(audit_path))
if audit.get('audit') != 'digitwise_factor_v1_admission':
    raise SystemExit('[factor-sft] wrong audit schema')
for field in (
    'invalid_train_rows',
    'duplicate_normalized_train_prompts',
    'invalid_heldout_episodes',
    'counterfactual_mismatches',
    'missing_local_contexts',
    'train_heldout_exact_prompt_hits',
    'train_heldout_13gram_hits',
):
    if audit.get(field) != 0:
        raise SystemExit('[factor-sft] admission audit failed: {}'.format(field))
if audit.get('required_local_contexts') != 3400 or audit.get('covered_local_contexts') != 3400:
    raise SystemExit('[factor-sft] incomplete local-transition basis')
if set(audit.get('heldout_regimes', ())) != {'recombine_w4', 'recombine_w6', 'width_ood_w8'}:
    raise SystemExit('[factor-sft] incomplete heldout regime coverage')
if audit.get('data_sha256') != digest(data) or audit.get('episodes_sha256') != digest(episodes):
    raise SystemExit('[factor-sft] artifact hash does not match audit')
print('[factor-sft] hash-bound static-tape admission passed')
PY

cd "$BASE/train"
timeout 600 "$PY" - "$DATA" "$BASE/artifacts/shohin-tok-32k.json" <<'PY'
import json
import sys

from sft_encoding import encode_supervised_example
from tokenizers import Tokenizer

row = json.loads(open(sys.argv[1]).readline())
prompt = row.get('completion_prompt', '')
if not prompt.endswith('Answer:') or row.get('question') != prompt:
    raise SystemExit('[factor-sft] data does not expose an exact completion prompt')
tokenizer = Tokenizer.from_file(sys.argv[2])
prompt_ids, _, _ = encode_supervised_example(
    tokenizer, prompt, row['response'], tokenizer.token_to_id('<|endoftext|>')
)
if prompt_ids != tokenizer.encode(prompt).ids:
    raise SystemExit('[factor-sft] inference/SFT prompt boundary mismatch')
print('[factor-sft] exact completion-prompt boundary passed')
PY

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[factor-sft] CUDA unavailable after allocation')
torch.empty(1, device='cuda')
PY

if [ "$COMPILE" != 0 ] && [ "$COMPILE" != 1 ]; then
  echo "[factor-sft] COMPILE must be 0 or 1" >&2
  exit 7
fi
ARGS=(--init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json"
  --q-fields question --r-fields response --prompt-override-field completion_prompt
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" --out "$OUT")
if [ "$COMPILE" = 1 ]; then
  ARGS+=(--compile)
fi
echo "[factor-sft] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS compile=$COMPILE"
"$PY" sft.py "${ARGS[@]}"
echo "[factor-sft] done"
