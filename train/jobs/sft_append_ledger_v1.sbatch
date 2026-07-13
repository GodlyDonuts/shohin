#!/bin/bash
# Isolated one-epoch ADL SFT. Do not submit before the DRS causal gate decides
# whether the shorter write interface addresses the observed failure.
#SBATCH -J shohin-ledger-sft
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc31,evc36,evc43,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/append_ledger_sft_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to a pinned raw checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/append_ledger_v1_train.jsonl}
EPISODES=${EPISODES:-$BASE/artifacts/evals/append_ledger_v1_heldout.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/append_ledger_v1_train.protocol_audit.json}
OUT=${OUT:?set OUT to a fresh isolated directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.001}
LRADAM=${LRADAM:-0.0002}

[ -f "$INIT" ] || { echo "[append-ledger-sft] init missing: $INIT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[append-ledger-sft] data missing: $DATA" >&2; exit 3; }
[ -s "$EPISODES" ] || { echo "[append-ledger-sft] heldout missing: $EPISODES" >&2; exit 4; }
[ -s "$AUDIT" ] || { echo "[append-ledger-sft] audit missing: $AUDIT" >&2; exit 5; }
[ ! -e "$OUT" ] || { echo "[append-ledger-sft] refusing existing output: $OUT" >&2; exit 6; }
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
if audit['invalid_rows_or_episodes'] or audit['duplicate_normalized_train_questions']:
    raise SystemExit('[append-ledger-sft] structural audit failed')
if audit['overlap']['exact_prompt_hits'] or audit['overlap']['ngram13_hits']:
    raise SystemExit('[append-ledger-sft] split-overlap audit failed')
if audit['data_sha256'] != digest(data) or audit['episodes_sha256'] != digest(episodes):
    raise SystemExit('[append-ledger-sft] artifact hash does not match audit')
expected = {'fit_w8', 'fit_w16', 'value_ood_w8', 'value_ood_w16', 'width_ood_w32'}
if set(audit['regimes']) != expected or audit['counterfactual_pairs'] != audit['heldout_episodes']:
    raise SystemExit('[append-ledger-sft] incomplete heldout audit')
print('[append-ledger-sft] hash-bound protocol audit passed')
PY
cd "$BASE/train"
"$PY" - "$DATA" "$BASE/artifacts/shohin-tok-32k.json" <<'PY'
import json
import sys

from sft import encode_supervised_example
from tokenizers import Tokenizer

row = json.loads(open(sys.argv[1]).readline())
prompt = row.get('completion_prompt', '')
if not prompt.endswith('Answer:') or row.get('question') != prompt:
    raise SystemExit('[append-ledger-sft] data does not expose an exact completion prompt')
tokenizer = Tokenizer.from_file(sys.argv[2])
prompt_ids, _, _ = encode_supervised_example(
    tokenizer, prompt, row['response'], tokenizer.token_to_id('<|endoftext|>')
)
if prompt_ids != tokenizer.encode(prompt).ids:
    raise SystemExit('[append-ledger-sft] inference/SFT prompt boundary mismatch')
print('[append-ledger-sft] exact completion-prompt boundary passed')
PY
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[append-ledger-sft] CUDA unavailable after allocation')
torch.empty(1, device='cuda')
PY

echo "[append-ledger-sft] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --q-fields question --r-fields response --prompt-override-field completion_prompt \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" --compile --out "$OUT"
echo "[append-ledger-sft] done"
