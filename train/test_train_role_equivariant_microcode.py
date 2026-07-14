#!/usr/bin/env python3
"""Tensor contracts for role-equivariant training losses."""

import torch

from role_equivariant_microcode import IGNORE_ROLE
from train_role_equivariant_microcode import masked_cross_entropy, normalized_distance, symmetric_kl


def main():
    logits = torch.tensor([[3.0, -1.0], [-2.0, 4.0]])
    targets = torch.tensor([0, IGNORE_ROLE])
    assert masked_cross_entropy(logits, targets).item() < 0.1
    assert normalized_distance(logits, logits).item() == 0.0
    assert abs(symmetric_kl(logits, logits).item()) < 1e-7
    flipped = logits.flip(-1)
    assert abs(symmetric_kl(logits, flipped.flip(-1)).item()) < 1e-7
    ignored = masked_cross_entropy(logits, torch.full((2,), IGNORE_ROLE))
    assert ignored.item() == 0.0
    print("role-equivariant trainer tests passed")


if __name__ == "__main__":
    main()
