#!/bin/bash
#SBATCH -J shohin-causal-prefix-preflight
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/causal_prefix_readback_preflight_%j.out
#SBATCH --open-mode=append
# CPU-only real-data tokenizer/batching preflight. Prefer Stokes; no model is loaded.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to admitted LSA pair JSONL}
AUDIT=${AUDIT:?set AUDIT to matching LSA audit JSON}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
for path in "$DATA" "$AUDIT" "$TOKENIZER"; do [ -s "$path" ] || { echo "[causal-prefix-preflight] missing input: $path" >&2; exit 2; }; done
cd "$BASE/train"
"$PY" - "$DATA" "$AUDIT" "$TOKENIZER" <<'PY'
import hashlib
import json
import sys

from tokenizers import Tokenizer

from causal_prefix_readback_memory_train import load_readback_pairs
from latent_state_algebra_train import audit_admits_training, bucketed_batches

data, audit_path, tokenizer_path = sys.argv[1:]
digest = hashlib.sha256(open(data, "rb").read()).hexdigest()
audit = json.load(open(audit_path))
if not audit_admits_training(audit, digest):
    raise SystemExit("LSA audit does not admit requested data")
pairs, skipped = load_readback_pairs(data, Tokenizer.from_file(tokenizer_path), slots=8, max_chunks=8, seq_len=2048)
batches, batch_report = bucketed_batches(pairs, batch_size=4, seed=20260717)
print(json.dumps({
    "audit": "causal_prefix_readback_preflight_v1",
    "data_sha256": digest,
    "pairs": len(pairs),
    "skipped": skipped,
    "batch_report": batch_report,
    "status": "passed",
}, sort_keys=True))
PY
