#!/bin/bash
#SBATCH -J shohin-rg-trace-smoke
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rg_tracer_smoke_%j.out
#SBATCH --open-mode=append
# Validate every registered deterministic tracer against fresh Reasoning-Gym
# samples on a compute node. This produces no training data and cannot mutate
# any frozen corpus.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$BASE/pipeline"
"$PY" - <<'PY'
import reasoning_gym as rg
from tracers import TRACERS

for family, tracer in sorted(TRACERS.items()):
    dataset = rg.create_dataset(family, size=80, seed=20260712)
    accepted = 0
    for row in dataset:
        trace = tracer(
            row.get("question", ""),
            row.get("answer", ""),
            row.get("metadata") or {},
        )
        accepted += bool(trace)
    print(f"[tracer] {family} {accepted}/80")
    if accepted == 0:
        raise SystemExit(f"no valid traces produced for {family}")
print(f"[tracer] validated {len(TRACERS)} families")
PY
