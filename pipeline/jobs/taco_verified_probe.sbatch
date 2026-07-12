#!/bin/bash
#SBATCH -J shohin-taco-probe
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/taco_verified_probe_%j.out
#SBATCH --open-mode=append
# Read-only schema/provenance probe for a possible execution-verified code source.
# It emits no training data and never changes a frozen mix or model output.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATASET=${DATASET:-likaixin/TACO-verified}

cd "$BASE"
"$PY" - "$DATASET" <<'PY'
import json
import sys

from datasets import get_dataset_config_names, load_dataset

dataset = sys.argv[1]
print(f"[taco-probe] dataset={dataset}")
configs = get_dataset_config_names(dataset)
print(f"[taco-probe] configs={configs}")
for config in configs[:3]:
    stream = load_dataset(dataset, name=config, split="train", streaming=True)
    row = next(iter(stream))
    summary = {}
    for key, value in row.items():
        if isinstance(value, str):
            summary[key] = {"type": "str", "chars": len(value)}
        elif isinstance(value, list):
            summary[key] = {
                "type": "list", "length": len(value),
                "first_type": type(value[0]).__name__ if value else None,
                "first_chars": len(str(value[0])) if value else 0,
            }
        elif isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(value)[:20]}
        else:
            summary[key] = {"type": type(value).__name__}
    print(f"[taco-probe] config={config} fields={json.dumps(summary, sort_keys=True)}")
PY
