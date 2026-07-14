#!/usr/bin/env python3
"""CPU contracts for Jacobian workspace stability and rank summaries."""

import torch

from eval_jacobian_workspace import rank_summary, stability


def main():
    generator = torch.Generator().manual_seed(9)
    matrix = torch.randn(24, 24, generator=generator)
    left = {"metadata": {"source_layers": [3]}, "jacobians": {3: matrix}}
    right = {"metadata": {"source_layers": [3]}, "jacobians": {3: matrix.clone()}}
    cell = stability(left, right)["3"]
    assert abs(cell["frobenius_cosine"] - 1.0) < 1e-6
    assert cell["relative_delta"] == 0.0
    assert abs(cell["right_subspace_overlap_k16"] - 1.0) < 1e-5
    records = [
        {"layer": 3, "regime": "language_ood", "future_rank": 1, "immediate_rank": 5},
        {"layer": 3, "regime": "full_ood", "future_rank": 2, "immediate_rank": 20},
    ]
    summary = rank_summary(records, "future", 3, {"language_ood", "full_ood"})
    assert summary["cases"] == 2
    assert summary["mrr"] == 0.75
    assert summary["top10"] == 1.0
    print("jacobian workspace evaluation tests passed")


if __name__ == "__main__":
    main()
