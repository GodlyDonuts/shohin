"""Training-only geometry losses for a source-free latent state packet.

LSA never changes the inference boundary: the decoder still receives only a
continuous packet and a fresh query. These modules supervise packet geometry
while training and are discarded before checkpoint promotion or inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class LatentStateAlgebra(nn.Module):
    """Measure state equivalence and verified deltas in continuous packets."""

    def __init__(self, d_model: int, state_dim: int, temperature: float = 0.1):
        super().__init__()
        if d_model <= 0 or state_dim <= 0:
            raise ValueError("d_model and state_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.temperature = float(temperature)
        self.project = nn.Linear(d_model, d_model, bias=False)
        self.state_probe = nn.Linear(d_model, state_dim)

    def packet_state(self, packet: torch.Tensor) -> torch.Tensor:
        """Pool normalized slots without adding an inference-time controller."""
        if packet.ndim != 3 or packet.shape[1] <= 0 or packet.shape[2] != self.d_model:
            raise ValueError("packet must have shape [batch, positive_slots, d_model]")
        return functional.layer_norm(packet, (self.d_model,)).mean(dim=1)

    def predict_state(self, packet: torch.Tensor) -> torch.Tensor:
        return self.state_probe(self.packet_state(packet))

    def losses(
        self,
        packet_a: torch.Tensor,
        packet_b: torch.Tensor,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return aligned, contrastive, state, and delta losses for paired records.

        A and B may encode equivalent records or verified interventions. The
        target state vectors are normalized numeric state labels supplied only
        while training. Batch rows must be independently mismatched semantic
        states for the contrastive term to be anti-collapse.
        """
        hidden_a, hidden_b = self.packet_state(packet_a), self.packet_state(packet_b)
        if (
            hidden_a.shape != hidden_b.shape
            or state_a.shape != state_b.shape
            or state_a.ndim != 2
            or state_a.shape[0] != hidden_a.shape[0]
            or state_a.shape[1] != self.state_dim
        ):
            raise ValueError("paired packets and states have incompatible shapes")

        projected_a = functional.normalize(self.project(hidden_a), dim=-1)
        projected_b = functional.normalize(self.project(hidden_b), dim=-1)
        alignment = (1.0 - (projected_a * projected_b).sum(dim=-1)).mean()

        if hidden_a.shape[0] == 1:
            contrastive = alignment.new_zeros(())
        else:
            labels = torch.arange(hidden_a.shape[0], device=hidden_a.device)
            logits = projected_a @ projected_b.T / self.temperature
            contrastive = 0.5 * (
                functional.cross_entropy(logits, labels)
                + functional.cross_entropy(logits.T, labels)
            )

        predicted_a = self.state_probe(hidden_a)
        predicted_b = self.state_probe(hidden_b)
        state = 0.5 * (
            functional.mse_loss(predicted_a, state_a)
            + functional.mse_loss(predicted_b, state_b)
        )
        delta = functional.mse_loss(predicted_b - predicted_a, state_b - state_a)
        return {
            "alignment": alignment,
            "contrastive": contrastive,
            "state": state,
            "delta": delta,
        }

    @staticmethod
    def total(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
        """Combine explicitly named losses so the zero-auxiliary control is exact."""
        required = {"alignment", "contrastive", "state", "delta"}
        if set(losses) != required or set(weights) != required:
            raise ValueError("losses and weights must contain exactly {}".format(sorted(required)))
        total = None
        for name in sorted(required):
            term = losses[name] * float(weights[name])
            total = term if total is None else total + term
        return total
