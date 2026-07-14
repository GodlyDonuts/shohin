"""Noise-calibrated posterior probe selection for a possible R6b control.

R6a uses a hard top-k candidate approximation. This CPU-only successor keeps a
Gaussian score posterior over every lawful operator and selects the probe with
maximum quantized partition entropy. It changes no learned parameters and must
not replace or reinterpret the already frozen R6a result.
"""

from __future__ import annotations

import torch


class PosteriorDistinctionScheduler:
    def __init__(self, codes, *, assumed_noise=1.0):
        if codes.ndim != 2:
            raise ValueError("codes must have shape [hypotheses, probes]")
        if codes.device.type != "cpu":
            raise ValueError("posterior scheduler is a CPU decision table")
        self.codes = codes.float()
        self.assumed_noise = float(assumed_noise)
        if self.assumed_noise <= 0:
            raise ValueError("assumed_noise must be positive")
        width = max(2.0 * self.assumed_noise, 0.25)
        group_ids = []
        group_probes = []
        total_groups = 0
        for probe in range(self.codes.shape[1]):
            bins = torch.round(self.codes[:, probe] / width).long()
            _, inverse = torch.unique(bins, sorted=True, return_inverse=True)
            group_ids.append(inverse + total_groups)
            group_probes.append(
                torch.full((int(inverse.max()) + 1,), probe, dtype=torch.long)
            )
            total_groups += int(inverse.max()) + 1
        self.group_ids = torch.stack(group_ids).reshape(-1)
        self.group_probes = torch.cat(group_probes)
        self.total_groups = total_groups

    def posterior(self, scores):
        scores = scores.float().cpu()
        if scores.shape != (self.codes.shape[0],):
            raise ValueError("scores must have one entry per hypothesis")
        centered = scores - scores.min()
        return torch.softmax(centered / (-2.0 * self.assumed_noise ** 2), dim=0)

    def select(self, scores, observed=()):
        weights = self.posterior(scores)
        masses = torch.zeros(self.total_groups).scatter_add_(
            0, self.group_ids, weights.repeat(self.codes.shape[1]),
        )
        terms = torch.where(masses > 0, -masses * torch.log2(masses), masses)
        entropy = torch.zeros(self.codes.shape[1]).scatter_add_(
            0, self.group_probes, terms,
        )
        observed = tuple(int(probe) for probe in observed)
        if observed:
            entropy[list(observed)] = -1
        probe = int(entropy.argmax().item())
        return probe, float(entropy[probe].item())


def identify_with_noisy_oracle(
    codes, target, *, actual_noise=0.0, assumed_noise=1.0, steps=3, seed=20260714,
    scheduler=None,
):
    """Evaluate scheduler mechanics; the target supplies effects but not probes."""
    codes = codes.float().cpu()
    target = int(target)
    if not 0 <= target < codes.shape[0]:
        raise ValueError("target hypothesis is out of range")
    scheduler = scheduler or PosteriorDistinctionScheduler(codes, assumed_noise=assumed_noise)
    if scheduler.codes.shape != codes.shape:
        raise ValueError("scheduler code shape differs")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scores = torch.zeros(codes.shape[0])
    observed = []
    trace = []
    for latent_step in range(int(steps)):
        probe, entropy = scheduler.select(scores, observed)
        effect = codes[target, probe] + float(actual_noise) * torch.randn((), generator=generator)
        scores += (codes[:, probe] - effect).square()
        observed.append(probe)
        trace.append({
            "latent_step": latent_step,
            "probe": probe,
            "effect": float(effect.item()),
            "partition_entropy": entropy,
            "top_hypothesis": int(scores.argmin().item()),
        })
    selected = int(scores.argmin().item())
    return {"target": target, "selected": selected, "correct": selected == target, "trace": trace}
