#!/usr/bin/env python3
"""Static contracts for the isolated token-native-ledger evaluator job."""
from pathlib import Path


text = Path(__file__).with_name("jobs").joinpath("eval_token_native_ledger.sbatch").read_text()
for required in (
    "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
    "token_native_ledger_v1_heldout.jsonl",
    "[tnl-eval] CUDA unavailable after allocation",
    "eval_token_native_ledger.py",
    "--max-new-final",
    "--prompt-style",
    "[ ! -e \"$OUT\" ]",
):
    assert required in text, required
print("token-native ledger evaluator job checks: passed")
