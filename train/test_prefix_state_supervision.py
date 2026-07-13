#!/usr/bin/env python3
"""CPU-only invariants for training-only causal prefix supervision."""

import torch

from prefix_state_supervision import prefix_state_targets, prefix_trajectory_losses


def main():
    targets = prefix_state_targets(
        {"left": 4, "right": 9},
        (
            {"kind": "add", "target": "left", "value": 3},
            {"kind": "move", "source": "right", "target": "left", "value": 2},
            {"kind": "swap", "left": "left", "right": "right"},
        ),
        ("left", "right"),
        10,
    )
    assert targets == [[0.7, 0.9], [0.9, 0.7], [0.7, 0.9]]

    torch.manual_seed(3)
    packets = tuple(torch.randn(2, 3, 4, requires_grad=True) for _ in range(3))
    expected = torch.tensor([targets, targets], dtype=torch.float32)
    probe = torch.nn.Linear(4, 2, bias=False)
    losses = prefix_trajectory_losses(packets, expected, lambda packet: probe(packet.mean(dim=1)))
    assert losses["predicted"].shape == expected.shape
    assert torch.isfinite(losses["state"]) and torch.isfinite(losses["delta"])
    (losses["state"] + losses["delta"]).backward()
    assert probe.weight.grad is not None and probe.weight.grad.abs().sum() > 0
    assert all(packet.grad is not None and packet.grad.abs().sum() > 0 for packet in packets)
    print("prefix state supervision tests passed")


if __name__ == "__main__":
    main()
