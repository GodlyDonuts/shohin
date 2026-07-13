#!/usr/bin/env python3
"""Focused CPU contracts for the prefix-state trainer's causal labels."""

import torch

from prefix_state_memory_train import prefix_targets_for_row, shuffled_targets


def main():
    row = {
        "initial": {"left": 8, "right": 5},
        "keys": ["left", "right"],
        "operations": [
            {"kind": "add", "target": "left", "value": 2},
            {"kind": "move", "source": "left", "target": "right", "value": 4},
        ],
        "state_scale": 10,
        "state": [6, 9],
    }
    assert prefix_targets_for_row(row) == [[1.0, 0.5], [0.6, 0.9]]
    torch.manual_seed(1)
    targets = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    assert torch.equal(shuffled_targets(targets, "verified"), targets)
    shuffled = shuffled_targets(targets, "shuffled")
    assert shuffled.shape == targets.shape and not torch.equal(shuffled, targets)
    print("prefix-state trainer contracts passed")


if __name__ == "__main__":
    main()
