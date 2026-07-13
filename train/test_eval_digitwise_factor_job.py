#!/usr/bin/env python3
"""Static isolation contract for the factorized closed-loop evaluator job."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "eval_digitwise_factor.sbatch"


text = JOB.read_text()
assert "eval_digitwise_factor.py" in text
assert "[ ! -e \"$OUT\" ]" in text
assert "torch.empty(1, device=\"cuda\")" in text
assert "flagship_out" not in text
assert "--gres=gpu:nvidia_h100_pcie:1" in text
print("digitwise factor evaluator job contracts passed")
