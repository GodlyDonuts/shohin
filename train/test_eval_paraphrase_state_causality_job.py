#!/usr/bin/env python3
"""Static isolation contract for the paraphrase activation-exchange job."""
from pathlib import Path

text = Path(__file__).with_name("jobs").joinpath("eval_paraphrase_state_causality.sbatch").read_text()
for required in (
    "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
    "semantic_basis_transport_v2_factor_language.jsonl",
    "482f303465aef45bb3f046ebdb3a1e5f7bc8d24f4ab5bae8973e8dcf9c7f5a3a",
    "eval_paraphrase_state_causality.py",
    "OUT must be a fresh eval-history JSON",
    "torch.empty(1, device=\"cuda\", dtype=torch.bfloat16)",
):
    assert required in text, required
assert "train/psa" not in text
assert "--out \"$OUT\"" in text
print("paraphrase state causality job checks: passed")
