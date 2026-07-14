#!/usr/bin/env python3
"""Static safety tests for the exact-carrier semantic-basis SFT wrapper."""
from pathlib import Path


text = Path(__file__).with_name("jobs").joinpath("sft_semantic_basis_transport_v2.sbatch").read_text()
for required in (
    "semantic_basis_transport_v2_train.jsonl",
    "semantic_basis_transport_v2_train.quality.json",
    "semantic_basis_transport_v2_train.training_text_overlap.json",
    "semantic_basis_transport_v2_train.response_contracts.json",
    "semantic_basis_transport_v2_train.packing_qa.json",
    "best_step200000.pt",
    "[ ! -e \"$OUT\" ]",
    "timeout --kill-after=10s 90s",
    "--epochs \"$EPOCHS\"",
    "--group-field training_group",
    "sft_ep${EPOCHS}.pt",
    "think_marker_rate",
):
    assert required in text, required
assert "train.py" not in text
assert "SHARDS=" not in text
assert "flagship.sbatch" not in text
print("semantic-basis v2 SFT job checks: passed")
