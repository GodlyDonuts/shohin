#!/usr/bin/env python3
"""Static checks for the isolated broad-anchor operator SFT wrapper."""
from pathlib import Path


def main():
    text = (Path(__file__).resolve().parent / "jobs" / "sft_operator_trace_broad_anchor_v1.sbatch").read_text()
    required = (
        "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
        "operator_trace_anchor_*",
        "public-eval filter was not enabled",
        "public-eval prompt overlap remains",
        "public-eval text overlap remains",
        "REQUIRED_EXCLUDED_CONTRACT",
        "DIRECTNESS",
        "REQUIRED_OPERATOR_MIN_ROWS",
        "paired-answer response grammar remains",
        "--freeze-lexicon",
        "--replay-prompts",
        "--reference",
        "--sample-weights $WEIGHTS",
    )
    for value in required:
        assert value in text, value
    assert "flagship_out" not in text
    assert "sbatch " not in text
    print("operator trace broad-anchor SFT job: passed")


if __name__ == "__main__":
    main()
