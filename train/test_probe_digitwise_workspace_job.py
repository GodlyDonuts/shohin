#!/usr/bin/env python3
"""Static isolation contract for the residual-patching workspace job."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "probe_digitwise_workspace.sbatch"


text = JOB.read_text()
assert "probe_digitwise_workspace.py" in text
assert "--gres=gpu:nvidia_h100_pcie:1" in text
assert "torch.empty(1, device=\"cuda\")" in text
assert "OUT=${OUT:?set OUT to a fresh result JSON}" in text
assert "flagship_out" not in text
assert "sft.py" not in text
print("workspace probe job checks: passed")
