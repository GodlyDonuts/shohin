#!/usr/bin/env python3
"""Static safety checks for the post-bridge capsule SFT wrapper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "sft_semantic_capsule_v11a.sbatch"
text = JOB.read_text()

for required in (
    "INIT=${INIT:?",
    "BRIDGE_EVAL=${BRIDGE_EVAL:?",
    "COMPOSITION_EVAL=${COMPOSITION_EVAL:?",
    "semantic_capsule_v1_train.jsonl",
    "semantic_capsule_v1_train.protocol_audit.r1.json",
    "semantic_bridge_v1_heldout.jsonl",
    "semantic_composition_transfer_v1.jsonl",
    "MIN_BRIDGE_ANSWERS=${MIN_BRIDGE_ANSWERS:-250}",
    "MIN_BRIDGE_TRACE_CONTRACT=${MIN_BRIDGE_TRACE_CONTRACT:-200}",
    "MIN_BRIDGE_TRACE_CONTRACT_PER_FAMILY=${MIN_BRIDGE_TRACE_CONTRACT_PER_FAMILY:-25}",
    "MIN_COMPOSITION_ANSWERS=${MIN_COMPOSITION_ANSWERS:-50}",
    "completion_prompt\") != row.get(\"question\")",
    "--prompt-override-field completion_prompt",
    "timeout --kill-after=10s 90s",
    "[ ! -e \"$OUT\" ]",
    "sft.py",
    "--compile",
):
    assert required in text, required

assert "best_step200000.pt" not in text
assert "flagship_out" not in text
assert "SHARDS=" not in text
assert "--gres=gpu:nvidia_h100_pcie:1" in text
assert "evc49" in text
assert "evc45" in text
print("V11A semantic-capsule SFT job checks: passed")
