#!/usr/bin/env python3
"""Static safety contracts for the exact-carrier semantic-basis job."""
from pathlib import Path


text = Path(__file__).with_name("jobs").joinpath("eval_semantic_basis_transport.sbatch").read_text()
for required in (
    "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
    "semantic_basis_transport_v2_heldout.jsonl",
    "best_step200000.pt",
    "timeout --kill-after=10s 90s",
    "torch.empty(1, device=\"cuda\", dtype=torch.bfloat16)",
    "[ ! -e \"$OUT\" ]",
    "eval_semantic_basis_transport.py",
    "--pairs \"$PAIRS\"",
    "PROMPT_MODE=${PROMPT_MODE:-qa}",
    "--prompt-mode \"$PROMPT_MODE\"",
):
    assert required in text, required
assert "sft.py" not in text
assert "flagship.sbatch" not in text
evaluator = Path(__file__).with_name("eval_semantic_basis_transport.py").read_text()
for required in (
    '"results": results',
    '"prompt_mode": args.prompt_mode',
    '"inference_prompt_template"',
    'model_prompt(prompt, args.prompt_mode)',
    '"claim_boundary"',
    '"control_boundary"',
):
    assert required in evaluator, required
print("semantic-basis evaluator job checks: passed")
