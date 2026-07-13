#!/usr/bin/env python3
"""Static safety checks for the isolated semantic-capsule evaluator."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "eval_semantic_capsule.sbatch"
text = JOB.read_text()

for required in (
    "--gres=gpu:nvidia_h100_pcie:1",
    "evc49",
    "CKPT=${CKPT:?",
    "EPISODES=${EPISODES:?",
    "[ ! -e \"$OUT\" ]",
    "timeout --kill-after=10s 90s",
    "torch.empty(1, device=\"cuda\", dtype=torch.bfloat16)",
    "eval_semantic_capsule.py",
):
    assert required in text, required

assert "sft.py" not in text
assert "flagship_out" not in text
assert "SHARDS=" not in text
print("semantic-capsule evaluation job checks: passed")
