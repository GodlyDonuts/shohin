#!/usr/bin/env python3
"""Tensor contracts for role-equivariant training losses."""

import torch

from role_equivariant_microcode import IGNORE_ROLE
from train_role_equivariant_microcode import (
    group_constraints,
    masked_cross_entropy,
    normalized_distance,
    symmetric_kl,
)


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

    # VIEW_ORDER is anchor/a/b at permutation 0 followed by anchor/a/b at 1.
    op_features = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    op_kind = torch.tensor([[4.0, -2.0]] * 6)
    op_role = torch.tensor([[5.0, -3.0]] * 3 + [[-3.0, 5.0]] * 3)
    op_targets = torch.tensor([0, 0, 0, 1, 1, 1])
    op_slices = [slice(index, index + 1) for index in range(6)]
    query_features = op_features.clone()
    query_kind = op_kind.clone()
    query_role = op_role.clone()
    query_targets = op_targets.clone()
    semantic, permutation = group_constraints(
        op_features, op_kind, op_role, op_targets, op_slices,
        query_features, query_kind, query_role, query_targets, 1,
    )
    assert abs(semantic.item()) < 1e-7
    assert abs(permutation.item()) < 1e-7
    broken_role = op_role.clone()
    broken_role[3:] = broken_role[:3]
    _, broken = group_constraints(
        op_features, op_kind, broken_role, op_targets, op_slices,
        query_features, query_kind, query_role, query_targets, 1,
    )
    assert broken.item() > 1.0
    print("role-equivariant trainer tests passed")


if __name__ == "__main__":
    main()
