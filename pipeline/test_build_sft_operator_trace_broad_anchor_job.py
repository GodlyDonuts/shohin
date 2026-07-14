#!/usr/bin/env python3
"""Static guardrails for the CPU-only broad-anchor data build."""
from pathlib import Path


def main():
    text = (Path(__file__).resolve().parent / "jobs" / "build_sft_operator_trace_broad_anchor.sbatch").read_text()
    required = (
        "#SBATCH -c 4",
        "build_sft_multisource_mix.py",
        "--eval-glob",
        "audit_sft_quality.py",
        "audit_training_text_overlap.py",
        "audit_generalization_overlap.py",
        "--case-regimes wording full",
        "EXCLUDE_CONTRACTS",
        "--exclude-contract",
        "hard public-eval filter was not enabled",
        "cannot write a checkpoint",
    )
    for value in required:
        assert value in text, value
    assert "sbatch " not in text
    assert "sft.py" not in text
    assert "flagship_out" not in text
    print("operator trace broad-anchor CPU job: passed")


if __name__ == "__main__":
    main()
