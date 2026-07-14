#!/usr/bin/env python3
"""Static safety contracts for the isolated paraphrase-state-alignment job."""
from pathlib import Path


text = Path(__file__).with_name("jobs").joinpath("sft_paraphrase_state_alignment.sbatch").read_text()
for required in (
    "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
    "best_step200000.pt",
    "DATA_SHA=4363101c9773b055e24bce3e79f727f3c103a0fb357a6820206ddeab2567234f",
    "[ ! -e \"$OUT\" ]",
    "sha256sum \"$DATA\"",
    "torch.empty(1, device=\"cuda\", dtype=torch.bfloat16)",
    "case \"$MODE\" in same|mismatch|none)",
    "sft_paraphrase_state_alignment.py",
    "test -s \"$OUT/psa_ep${EPOCHS}.pt\"",
):
    assert required in text, required
assert "flagship.sbatch" not in text
assert "OUT=${OUT:?set OUT to a fresh isolated output directory}" in text
print("paraphrase state alignment job checks: passed")
