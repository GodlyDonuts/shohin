#!/usr/bin/env python3
"""Exact mechanics for active counterfactual discrimination."""

import torch

from future_distinction_cell import (
    hypothesis_effect_codes,
    identify_with_oracle,
    legal_operator_hypotheses,
)


def main():
    hypotheses = legal_operator_hypotheses(range(1, 100))
    codes = hypothesis_effect_codes(hypotheses)
    assert len(hypotheses) == 597
    assert codes.shape == (597, 64)
    assert torch.unique(codes, dim=0).shape[0] == len(hypotheses)

    active_lengths = []
    random_lengths = []
    for target in range(len(hypotheses)):
        active = identify_with_oracle(codes, target, policy="active")
        random = identify_with_oracle(codes, target, policy="random", seed=20260714 + target)
        assert active["resolved"]
        assert random["resolved"]
        active_lengths.append(len(active["observed"]))
        random_lengths.append(len(random["observed"]))

    assert max(active_lengths) < 64
    print(
        "future-distinction mechanics passed: hypotheses={} active_mean={:.3f} "
        "active_max={} random_mean={:.3f} random_max={}".format(
            len(hypotheses),
            sum(active_lengths) / len(active_lengths),
            max(active_lengths),
            sum(random_lengths) / len(random_lengths),
            max(random_lengths),
        )
    )


if __name__ == "__main__":
    main()
