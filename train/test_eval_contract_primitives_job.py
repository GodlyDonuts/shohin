#!/usr/bin/env python3
"""Static CUDA-allocation contract for the primitive evaluator job."""

from pathlib import Path


text = (Path(__file__).resolve().parent / "jobs" / "eval_contract_primitives.sbatch").read_text()
for required in (
    "timeout --kill-after=10s 90s",
    "torch.empty(1024, device=\"cuda\", dtype=torch.bfloat16)",
    "torch.cuda.synchronize()",
    "[contract-eval] CUDA unavailable after allocation",
):
    assert required in text, required
print("contract primitive evaluator CUDA checks: passed")
