#!/usr/bin/env python3
"""Static safety checks for the isolated H100 allocation preflight wrapper."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "cuda_h100_preflight.sbatch"
text = JOB.read_text()
for required in (
    "--gres=gpu:nvidia_h100_pcie:1",
    "evc45",
    "nvidia-smi",
    "timeout --kill-after=10s 90s",
    "torch.cuda.is_available()",
    "torch.bfloat16",
    "torch.cuda.synchronize()",
):
    assert required in text, required
assert "sft.py" not in text
assert "train.py" not in text
assert "--resume" not in text
print("H100 CUDA preflight job checks: passed")
