#!/usr/bin/env python3
"""Static safety checks for the conditional Stokes semantic-basis build job."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline" / "jobs" / "generate_semantic_basis_transport_v1_stokes.sbatch"
text = JOB.read_text()
for required in (
    "#SBATCH -c 4",
    "#SBATCH --mem=24G",
    "semantic_basis_transport_v1_train.jsonl",
    "semantic_basis_transport_v1_heldout.jsonl",
    "semantic_basis_transport_v1.audit.json",
    "[ ! -e \"$path\" ]",
    "generate_semantic_basis_transport_v1.py",
    "audit_semantic_basis_transport_v1.py",
    "TRAIN_EPISODES=${TRAIN_EPISODES:-30000}",
    "HELDOUT_EPISODES=${HELDOUT_EPISODES:-1000}",
):
    assert required in text, required
assert "--gres" not in text
assert "sft.py" not in text
assert "train.py" not in text
print("semantic-basis Stokes build job checks: passed")
