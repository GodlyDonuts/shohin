#!/usr/bin/env python3
"""Tensor contracts for role-equivariant training losses."""

import torch

from role_equivariant_microcode import IGNORE_ROLE
from train_role_equivariant_microcode import (
    group_constraints,
    masked_cross_entropy,
    normalized_distance,
)


def main():
    logits = torch.tensor([[3.0, -1.0], [-2.0, 4.0]])
    targets = torch.tensor([0, IGNORE_ROLE])
    assert masked_cross_entropy(logits, targets).item() < 0.1
    assert normalized_distance(logits, logits).item() == 0.0
    ignored = masked_cross_entropy(logits, torch.full((2,), IGNORE_ROLE))
    assert ignored.item() == 0.0

    # VIEW_ORDER is anchor/a/b at permutation 0 followed by anchor/a/b at 1.
    op_kind_features = torch.tensor([[1.0, 0.0]] * 6)
    op_role_features = torch.tensor([[1.0, 0.0]] * 3 + [[-1.0, 0.0]] * 3)
    op_targets = torch.tensor([0, 0, 0, 1, 1, 1])
    op_slices = [slice(index, index + 1) for index in range(6)]
    query_kind_features = op_kind_features.clone()
    query_role_features = op_role_features.clone()
    query_targets = op_targets.clone()
    semantic, permutation = group_constraints(
        op_kind_features, op_role_features, op_targets, op_slices,
        query_kind_features, query_role_features, query_targets, 1,
    )
    assert abs(semantic.item()) < 1e-7
    assert abs(permutation.item()) < 1e-7
    broken_role = op_role_features.clone()
    broken_role[3:] = broken_role[:3]
    _, broken = group_constraints(
        op_kind_features, broken_role, op_targets, op_slices,
        query_kind_features, query_role_features, query_targets, 1,
    )
    assert broken.item() > 0.5
    print("role-equivariant trainer tests passed")


if __name__ == "__main__":
    main()
