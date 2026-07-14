#!/usr/bin/env python3
"""Static safety contracts for isolated NRR Slurm wrappers."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train" / "jobs" / "train_native_residual_relay.sbatch"
EVAL = ROOT / "train" / "jobs" / "eval_native_residual_relay.sbatch"


def main():
    train, evaluate = TRAIN.read_text(), EVAL.read_text()
    for required in (
        "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
        "native_residual_relay_v1_train.jsonl",
        "bac1e8d041abbfefa892056302a8d78c14abd0d31dd1694e9bc92aefac2fe03c",
        "train_native_residual_relay.py",
        "OUT must remain an isolated train/nrr_* directory",
        "torch.empty(1, device='cuda', dtype=torch.bfloat16)",
        "--max-batches",
    ):
        assert required in train, required
    for required in (
        "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
        "native_residual_relay_v1_heldout.jsonl",
        "1d8b633713fff41b331e7c2728e9c0aa3ae307a7b99622c526d99d6dc84120f2",
        "DATA_SHA=${DATA_SHA:-1d8b633713fff41b331e7c2728e9c0aa3ae307a7b99622c526d99d6dc84120f2}",
        "eval_native_residual_relay.py",
        "OUT must be an NRR eval-history JSON",
        "torch.empty(1, device='cuda', dtype=torch.bfloat16)",
        "SPLIT=${SPLIT:-heldout}",
        "--split \"$SPLIT\"",
    ):
        assert required in evaluate, required
    assert "flagship.sbatch" not in train + evaluate
    assert "flagship_out" not in evaluate
    print("native residual relay job contracts: passed")


if __name__ == "__main__":
    main()
