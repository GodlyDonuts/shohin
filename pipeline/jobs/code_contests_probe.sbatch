#!/bin/bash
#SBATCH -J shohin-code-probe
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/code_contests_probe_%j.out
#SBATCH --open-mode=append
# Read-only schema probe for a second test-backed code corpus. It writes no
# model data and streams a single record rather than materializing a full split.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PY" - <<'PY'
from datasets import get_dataset_split_names, load_dataset

dataset = "Imandra/code_contests"
print(f"[probe] dataset={dataset}")
print(f"[probe] splits={get_dataset_split_names(dataset)}")
for split in get_dataset_split_names(dataset):
    row = next(iter(load_dataset(dataset, split=split, streaming=True)))
    print(f"[probe] split={split} keys={sorted(row)}")
    for key in ("name", "description", "public_tests", "private_tests", "generated_tests", "solutions"):
        value = row.get(key)
        print(f"[probe] {key}: type={type(value).__name__} preview={str(value)[:400]}")
PY
