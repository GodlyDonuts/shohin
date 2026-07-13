#!/usr/bin/env python3
"""Static isolation and startup-environment checks for the DRS evaluator job."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "eval_digitwise_recurrent.sbatch"


text = JOB.read_text()
assert "eval_digitwise_recurrent.py" in text
assert "[ ! -e \"$OUT\" ]" in text
assert "torch.empty(1, device=\"cuda\")" in text
assert "OPENBLAS_NUM_THREADS=1" in text
assert "--gres=gpu:nvidia_h100_pcie:1" in text
assert "flagship_out" not in text
print("digitwise recurrent evaluator job contracts passed")
