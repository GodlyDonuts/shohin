#!/usr/bin/env python3
"""Static guardrails for CPU-only matched-reflection data admission."""
from pathlib import Path


def main() -> None:
    text = (Path(__file__).resolve().parent / "jobs" / "build_operator_counterfactual_reflection_v1.sbatch").read_text()
    required = (
        "#SBATCH -c 4",
        "generate_operator_counterfactual_reflection_v1.py",
        "audit_operator_counterfactual_reflection_v1.py",
        "audit_sft_quality.py",
        "audit_training_text_overlap.py",
        "audit_generalization_overlap.py",
        "--case-source",
        "--cases-json",
        "operator_counterfactual_reflection_v1_r2",
        "construction-time held-out filter is incomplete",
        "--require-zero",
        "prompt-token delta was not measured",
        "refusing existing output",
        "cannot write a checkpoint",
    )
    for value in required:
        assert value in text, value
    assert "#SBATCH --gres=" not in text
    assert "sft.py" not in text
    assert "sbatch " not in text
    assert "flagship_out" not in text
    print("counterfactual reflection CPU job: passed")


if __name__ == "__main__":
    main()
