#!/bin/bash
#SBATCH -J shohin-openr1-probe
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openr1_math_probe_%j.out
#SBATCH --open-mode=append
# CPU-only schema/licensing intake probe. It writes no training data and must
# pass before a decontaminated, answer-checked OpenR1 curator is implemented.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATASET=${DATASET:-open-r1/OpenR1-Math-220k}

cd "$BASE"
"$PY" - <<PY
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

dataset = "${DATASET}"
configs = get_dataset_config_names(dataset)
print(f"[openr1] dataset={dataset} configs={configs}")
for config in configs:
    splits = get_dataset_split_names(dataset, config)
    print(f"[openr1] config={config} splits={splits}")
    for split in splits:
        stream = load_dataset(dataset, name=config, split=split, streaming=True)
        row = next(iter(stream))
        print(f"[openr1] config={config} split={split} keys={sorted(row)}")
        for key in sorted(row):
            value = row[key]
            text = str(value).replace("\n", " ")
            print(f"[openr1] {key}: type={type(value).__name__} chars={len(str(value))} sample={text[:240]}")
        break
PY
