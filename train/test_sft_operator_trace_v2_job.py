#!/usr/bin/env python3
"""Static safety checks for the isolated operator-trace SFT wrapper."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
job = (ROOT / "jobs" / "sft_operator_trace_v2.sbatch").read_text()
for required in (
    "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
    "operator_trace_contrast_v2_train.jsonl",
    "operator_trace_contrast_v2.factor_wf_overlap.json",
    "--freeze-lexicon",
    "--replay-prompts",
    "--sample-weights operator_trace_contrast=1",
    "sft_operator_trace_",
    "[ ! -e \"$OUT\" ]",
):
    assert required in job, required
assert "flagship_out" not in job
assert "--gres=gpu:nvidia_h100_pcie:2" not in job
print("operator-trace SFT job checks: passed")
