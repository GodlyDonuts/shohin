#!/bin/bash
#SBATCH -J shohin-dclm-probe
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dclm_baseline_probe_%j.out
#SBATCH --open-mode=append
# Read-only schema/access probe for DCLM-Baseline. It produces no data artifact
# and cannot change the live flagship's shard mix.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" - <<'PY'
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

dataset = "mlfoundations/dclm-baseline-1.0"
configs = get_dataset_config_names(dataset)
print(f"[dclm] configs={configs}")

for config in configs:
    try:
        splits = get_dataset_split_names(dataset, config)
        print(f"[dclm] config={config} splits={splits}")
        if "train" not in splits:
            continue
        stream = load_dataset(dataset, name=config, split="train", streaming=True)
        row = next(iter(stream))
        print(f"[dclm] config={config} keys={sorted(row)}")
        for key, value in row.items():
            text = str(value).replace("\n", " ")
            print(f"[dclm] {key}={text[:240]}")
        break
    except Exception as exc:
        print(f"[dclm] config={config} error={type(exc).__name__}: {exc}")
else:
    raise SystemExit("[dclm] no readable train configuration")
PY
