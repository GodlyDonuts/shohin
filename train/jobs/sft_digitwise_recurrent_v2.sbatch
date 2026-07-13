#!/bin/bash
# Isolated one-epoch DRS SFT. It starts from a fixed raw checkpoint and cannot
# read or write the flagship output directory.
#SBATCH -J shohin-drs-sft
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
# `evc44` failed the same real CUDA allocation after the data and prompt
# preflights had passed. Keep this exclusion local to the DRS experiment.
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_sft_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
# The account-wide process limit is low relative to NumPy/OpenBLAS's default
# thread fan-out. Keep preflight and SFT imports below that limit; the model
# still uses the job's four CPU cores through PyTorch's OpenMP pool.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
INIT=${INIT:?set INIT to best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.jsonl}
EPISODES=${EPISODES:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.protocol_audit.json}
OUT=${OUT:?set OUT to a fresh isolated directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.001}
LRADAM=${LRADAM:-0.0002}

[ -f "$INIT" ] || { echo "[drs-sft] init missing: $INIT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[drs-sft] data missing: $DATA" >&2; exit 3; }
[ -s "$EPISODES" ] || { echo "[drs-sft] heldout missing: $EPISODES" >&2; exit 4; }
[ -s "$AUDIT" ] || { echo "[drs-sft] audit missing: $AUDIT" >&2; exit 5; }
[ ! -e "$OUT" ] || { echo "[drs-sft] refusing existing output: $OUT" >&2; exit 6; }
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
    raise SystemExit('[drs-sft] structural audit failed')
if audit['overlap']['exact_prompt_hits'] or audit['overlap']['ngram13_hits']:
    raise SystemExit('[drs-sft] split-overlap audit failed')
if audit['data_sha256'] != digest(data) or audit['episodes_sha256'] != digest(episodes):
    raise SystemExit('[drs-sft] artifact hash does not match audit')
expected = {'fit_w4', 'fit_w6', 'value_ood_w4', 'value_ood_w6', 'width_ood_w8'}
if set(audit['regimes']) != expected or audit['counterfactual_pairs'] != audit['heldout_episodes']:
    raise SystemExit('[drs-sft] incomplete heldout audit')
print('[drs-sft] hash-bound protocol audit passed')
PY
cd "$BASE/train"
# A cold framework import can take several minutes from Lustre. Bound it well
# below wall time while allowing a known-good node to finish initialization.
timeout 600 "$PY" - "$DATA" "$BASE/artifacts/shohin-tok-32k.json" <<'PY'
import json
import sys

from sft import encode_supervised_example
from tokenizers import Tokenizer

row = json.loads(open(sys.argv[1]).readline())
prompt = row.get('completion_prompt', '')
if not prompt.endswith('Answer:') or row.get('question') != prompt:
    raise SystemExit('[drs-sft] data does not expose an exact completion prompt')
tokenizer = Tokenizer.from_file(sys.argv[2])
prompt_ids, _, _ = encode_supervised_example(
    tokenizer, prompt, row['response'], tokenizer.token_to_id('<|endoftext|>')
)
if prompt_ids != tokenizer.encode(prompt).ids:
    raise SystemExit('[drs-sft] inference/SFT prompt boundary mismatch')
print('[drs-sft] exact completion-prompt boundary passed')
PY
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[drs-sft] CUDA unavailable after allocation')
torch.empty(1, device='cuda')
PY

echo "[drs-sft] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --q-fields question --r-fields response --prompt-override-field completion_prompt \
  --epochs "$EPOCHS" --batch-size "$BS" \
  --lr-muon "$LRMUON" --lr-adam "$LRADAM" --compile --out "$OUT"
echo "[drs-sft] done"
