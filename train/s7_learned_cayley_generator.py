"""Tiny trainable generator tables for S7 learned Cayley compilation."""

from __future__ import annotations

import torch
from torch import nn


PRIMARY_MODULI = (5, 7, 11)
MAX_MODULUS = max(PRIMARY_MODULI)
PROMOTED_STACK_PARAMETERS = 133_694_869


class LearnedCayleyGenerator(nn.Module):
    """Learn one successor permutation and zero symbol per admitted modulus."""

    def __init__(self, moduli: tuple[int, ...] = PRIMARY_MODULI) -> None:
        super().__init__()
        self.moduli = tuple(moduli)
        self.successor_logits = nn.ParameterDict(
            {
                str(modulus): nn.Parameter(torch.empty(modulus, modulus))
                for modulus in self.moduli
            }
        )
        self.zero_logits = nn.ParameterDict(
            {
                str(modulus): nn.Parameter(torch.empty(modulus))
                for modulus in self.moduli
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def successor(self, modulus: int) -> torch.Tensor:
        return self.successor_logits[str(int(modulus))]

    def zero(self, modulus: int) -> torch.Tensor:
        return self.zero_logits[str(int(modulus))]

    def discrete_successor(self, modulus: int) -> tuple[int, ...]:
        logits = self.successor(modulus).detach().cpu()
        return tuple(int(value) for value in logits.argmax(-1).tolist())

    def discrete_zero(self, modulus: int) -> int:
        return int(self.zero(modulus).detach().cpu().argmax().item())

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def total_system_params(self) -> int:
        return PROMOTED_STACK_PARAMETERS + self.num_params()
