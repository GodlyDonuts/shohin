#!/usr/bin/env python3
"""Static safety checks for the reusable semantic-bridge evaluation wrapper."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "eval_semantic_bridge.sbatch"
text = JOB.read_text()
for required in (
    "--gres=gpu:nvidia_h100_pcie:1",
    "timeout --kill-after=10s 90s",
    "torch.empty(1, device=\"cuda\", dtype=torch.bfloat16)",
    "PER_FAMILY=${PER_FAMILY:-100}",
    "--per-family \"$PER_FAMILY\"",
    "--max-new \"$MAX_NEW\"",
):
    assert required in text, required
assert "sft.py" not in text
assert "flagship_out" not in text
print("semantic-bridge evaluation job checks: passed")
