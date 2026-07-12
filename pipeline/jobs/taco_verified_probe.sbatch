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
    try:
        io = json.loads(row.get("input_output") or "")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[taco-probe] config={config} input_output_parse_error={type(exc).__name__}")
        continue
    inputs = io.get("inputs")
    outputs = io.get("outputs")
    pair_count = min(len(inputs or []), len(outputs or []))
    print(json.dumps({
        "taco_input_output": {
            "config": config,
            "keys": sorted(io),
            "fn_name": io.get("fn_name"),
            "inputs_type": type(inputs).__name__,
            "outputs_type": type(outputs).__name__,
            "paired_cases": pair_count,
            "first_input_chars": len(str(inputs[0])) if pair_count else 0,
            "first_output_chars": len(str(outputs[0])) if pair_count else 0,
        },
    }, sort_keys=True))
PY
