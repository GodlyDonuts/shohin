#!/usr/bin/env python3
"""Static safety checks for the CPU-only operator-trace assessor."""

from pathlib import Path


text = (Path(__file__).resolve().parent / "jobs" / "assess_operator_trace_v2.sbatch").read_text()
assert "#SBATCH --gres=" not in text
assert "sbatch " not in text
assert "assess_operator_trace_v2.py" in text
assert "[ ! -e \"$OUT\" ]" in text
print("operator-trace assessor job checks: passed")
