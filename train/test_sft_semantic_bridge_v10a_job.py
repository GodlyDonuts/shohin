#!/usr/bin/env python3
"""Static safety checks for the isolated V10A semantic-bridge SFT wrapper."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "sft_semantic_bridge_v10a.sbatch"
text = JOB.read_text()
for required in (
    "semantic_bridge_v1_train.jsonl",
    "semantic_bridge_v1_train.quality.r1.json",
    "semantic_bridge_v1_train.packing.r1.json",
    "semantic_bridge_v1_train.response_contracts.r1.json",
    "best_step200000.pt",
    "[ ! -e \"$OUT\" ]",
    "timeout --kill-after=10s 90s",
    "--epochs \"$EPOCHS\"",
    "--group-field training_group",
    "sft_ep${EPOCHS}.pt",
):
    assert required in text, required
assert "flagship_out/ckpt_[" not in text
assert "train.py" not in text
assert "SHARDS=" not in text
print("V10A semantic-bridge SFT job checks: passed")
