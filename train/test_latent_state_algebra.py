#!/usr/bin/env python3
"""Focused invariants for the training-only latent-state algebra auxiliary."""

import torch

from latent_state_algebra import LatentStateAlgebra


def main():
    auxiliary = LatentStateAlgebra(d_model=2, state_dim=2, temperature=0.2)
    with torch.no_grad():
        auxiliary.project.weight.copy_(torch.eye(2))
        auxiliary.state_probe.weight.copy_(torch.eye(2))
        auxiliary.state_probe.bias.zero_()

    packet_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    packet_b = packet_a.clone()
    states = auxiliary.packet_state(packet_a).detach()
    losses = auxiliary.losses(packet_a, packet_b, states, states)
    assert losses["alignment"].item() < 1e-6
    assert losses["state"].item() < 1e-6
    assert losses["delta"].item() < 1e-6
    assert losses["contrastive"].item() > 0.0
    total = auxiliary.total(losses, {
        "alignment": 0.1,
        "contrastive": 0.1,
        "separation": 0.1,
        "state": 1.0,
        "delta": 0.5,
    })
    assert torch.isfinite(total)

    singleton = auxiliary.losses(
        packet_a[:1], packet_b[:1], states[:1], states[:1],
    )
    assert singleton["contrastive"].item() == 0.0

    intervention_packet = torch.tensor([[[0.0, 1.0]]])
    intervention_state = auxiliary.packet_state(intervention_packet).detach()
    intervention = auxiliary.losses(
        packet_a[:1],
        intervention_packet,
        states[:1],
        intervention_state,
        equivalent=torch.tensor([False]),
    )
    assert intervention["alignment"].item() == 0.0
    assert intervention["contrastive"].item() == 0.0
    assert intervention["separation"].item() == 0.0
    assert intervention["state"].item() < 1e-6
    assert intervention["delta"].item() < 1e-6

    collapsed = auxiliary.losses(
        packet_a[:1],
        packet_a[:1],
        states[:1],
        states[:1] + torch.tensor([[0.5, 0.0]]),
        equivalent=torch.tensor([False]),
    )
    assert collapsed["separation"].item() > 0.0
    try:
        auxiliary.losses(packet_a, packet_b, states[:1], states[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched paired states must fail")
    try:
        auxiliary.losses(packet_a, packet_b, states, states, equivalent=torch.tensor([True]))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched equivalence mask must fail")
    print("latent state algebra auxiliary tests passed")


if __name__ == "__main__":
    main()
