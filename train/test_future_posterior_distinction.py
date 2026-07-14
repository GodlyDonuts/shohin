#!/usr/bin/env python3
"""Deterministic robustness gates for the separate R6b scheduler mechanics."""

import torch

from future_distinction_cell import (
    hypothesis_effect_codes,
    legal_operator_hypotheses,
    select_discriminating_probe,
)
from future_posterior_distinction import PosteriorDistinctionScheduler, identify_with_noisy_oracle


def topk_accuracy(codes, noise):
    correct = 0
    for target in range(codes.shape[0]):
        generator = torch.Generator().manual_seed(20260714 + target)
        scores = torch.zeros(codes.shape[0])
        observed = []
        for step in range(3):
            plausible = (
                tuple(range(codes.shape[0]))
                if step == 0
                else torch.topk(scores, 64, largest=False).indices.tolist()
            )
            probe = select_discriminating_probe(codes, plausible, observed)
            effect = codes[target, probe] + noise * torch.randn((), generator=generator)
            scores += (codes[:, probe] - effect).square()
            observed.append(probe)
        correct += int(scores.argmin().item()) == target
    return correct / codes.shape[0]


def posterior_accuracy(codes, noise):
    scheduler = PosteriorDistinctionScheduler(codes, assumed_noise=1.0)
    correct = sum(
        identify_with_noisy_oracle(
            codes, target, actual_noise=noise, assumed_noise=1.0,
            seed=20260714 + target, scheduler=scheduler,
        )["correct"]
        for target in range(codes.shape[0])
    )
    return correct / codes.shape[0]


def main():
    codes = hypothesis_effect_codes(legal_operator_hypotheses(range(1, 100))).float()
    exact = posterior_accuracy(codes, 0.0)
    posterior = posterior_accuracy(codes, 0.5)
    topk = topk_accuracy(codes, 0.5)
    assert exact == 1.0
    assert posterior >= 0.92
    assert posterior >= topk + 0.02
    print(
        "posterior-distinction mechanics: exact={:.4f} noise0.5={:.4f} "
        "topk_noise0.5={:.4f}".format(exact, posterior, topk)
    )


if __name__ == "__main__":
    main()
