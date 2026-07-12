#!/bin/bash
#SBATCH -J shohin-fineweb-probe
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_probe_%j.out
#SBATCH --open-mode=append
# Read-only schema/access probe for a future general-language educational source.
# It produces no shard output and never changes the live flagship data mix.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
"$PY" - <<'PY'
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

dataset = "HuggingFaceFW/fineweb-edu"
configs = get_dataset_config_names(dataset)
print(f"[fineweb-edu] configs={configs}")
# Prefer the compact, already quality-selected public subset we would tokenize;
# fall back to default only if the Hub changes its configuration names.
preferred = [name for name in ("sample-10BT", "default") if name in configs]
for config in preferred:
    splits = get_dataset_split_names(dataset, config)
    print(f"[fineweb-edu] config={config} splits={splits}")
    if "train" not in splits:
        continue
    stream = load_dataset(dataset, name=config, split="train", streaming=True)
    row = next(iter(stream))
    print(f"[fineweb-edu] config={config} keys={sorted(row)}")
    for key in ("text", "language", "score", "url"):
        if key in row:
            value = str(row[key]).replace("\n", " ")
            print(f"[fineweb-edu] {key}={value[:240]}")
    break
PY
