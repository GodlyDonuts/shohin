#!/usr/bin/env python3
"""Static safety contracts for the CPU-only CWI build wrapper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline" / "jobs" / "generate_counterfactual_workspace_v1_stokes.sbatch"
text = JOB.read_text()

for required in (
    "#SBATCH -c 4",
    "/usr/bin/python3.12",
    "build_counterfactual_workspace_v1.py",
    "audit_counterfactual_workspace_v1.py",
    "[ ! -e \"$path\" ]",
    "counterfactual_workspace_v1_train.jsonl",
    "counterfactual_workspace_v1_heldout.jsonl",
    "missing_legal_local_contexts",
    "invalid_heldout_world_pairs",
):
    assert required in text, required
assert "--gres=" not in text
assert "valid_train_rows" in text and "train_foil_counts" in text
print("counterfactual workspace Stokes job checks: passed")
