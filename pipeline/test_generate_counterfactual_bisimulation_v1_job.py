#!/usr/bin/env python3
"""Static safety contracts for the CPU-only CBC build wrapper."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline" / "jobs" / "generate_counterfactual_bisimulation_v1_stokes.sbatch"
text = JOB.read_text()

for required in (
    "#SBATCH -c 4",
    "/usr/bin/python3.12",
    "generate_counterfactual_bisimulation_v1.py",
    "audit_counterfactual_bisimulation_v1.py",
    "[ ! -e \"$path\" ]",
    "counterfactual_bisimulation_v1_train.jsonl",
    "counterfactual_bisimulation_v1_heldout.jsonl",
):
    assert required in text, required
assert "--gres=" not in text
assert "valid_train_rows" in text and "heldout_regimes" in text
print("counterfactual bisimulation Stokes job checks: passed")
