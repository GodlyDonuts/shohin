#!/usr/bin/env python3
"""Static contracts for isolated CBC evaluation scheduling."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "eval_counterfactual_bisimulation.sbatch"
text = JOB.read_text()
for required in (
    "--gres=gpu:nvidia_h100_pcie:1",
    "evc49",
    "timeout --kill-after=10s 90s",
    "eval_counterfactual_bisimulation.py",
    "counterfactual_bisimulation_v1_heldout.jsonl",
    "[ ! -e \"$OUT\" ]",
):
    assert required in text, required
print("counterfactual bisimulation evaluation job checks: passed")
