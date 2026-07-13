#!/usr/bin/env python3
"""Static contract tests for the staged factorized-register SFT job."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train" / "jobs" / "sft_digitwise_factor_v1.sbatch"


def test_hash_bound_static_tape_gate():
    text = JOB.read_text()
    for required in (
        "digitwise_factor_v1_admission",
        "counterfactual_mismatches",
        "missing_local_contexts",
        "train_heldout_13gram_hits",
        "required_local_contexts') != 3400",
        "recombine_w4",
        "width_ood_w8",
    ):
        assert required in text


def test_sft_boundary_and_isolation_contract():
    text = JOB.read_text()
    assert "--prompt-override-field completion_prompt" in text
    assert "[ ! -e \"$OUT\" ]" in text
    assert "flagship_out" not in text
    assert "--gres=gpu:nvidia_h100_pcie:1" in text


if __name__ == "__main__":
    test_hash_bound_static_tape_gate()
    test_sft_boundary_and_isolation_contract()
    print("sft digitwise factor v1 job contracts passed")
