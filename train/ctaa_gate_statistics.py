"""Preregistered finite-family and paired hierarchical CTAA statistics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta, binomtest


def one_sided_exact_lower(successes: int, total: int, *, alpha: float = 0.05) -> float:
    """Clopper-Pearson lower bound for one Bernoulli result per semantic family."""
    if total < 1 or not 0 <= successes <= total or not 0 < alpha < 1:
        raise ValueError("CTAA exact-bound inputs differ")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def one_sided_family_pvalue(successes: int, total: int, threshold: float) -> float:
    if total < 1 or not 0 <= successes <= total or not 0 <= threshold <= 1:
        raise ValueError("CTAA family-test inputs differ")
    return float(binomtest(successes, total, threshold, alternative="greater").pvalue)


def holm_rejections(pvalues: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    if not pvalues or not 0 < alpha < 1:
        raise ValueError("CTAA Holm inputs differ")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    if any(not 0 <= value <= 1 for _, value in ordered):
        raise ValueError("CTAA Holm p-value differs")
    rejected = {name: False for name in pvalues}
    for index, (name, value) in enumerate(ordered):
        cutoff = alpha / (len(ordered) - index)
        if value > cutoff:
            break
        rejected[name] = True
    return rejected


def paired_hierarchical_lower(
    differences_by_seed: Mapping[int, Sequence[int]],
    *,
    draws: int = 100_000,
    alpha: float = 0.05,
    seed: int,
) -> dict[str, float | int]:
    """Resample seeds and paired family differences without pooling renderings.

    Every family difference must be -1, 0, or +1. Conditional on one selected
    seed, a nonparametric family bootstrap depends only on those three counts,
    so multinomial sampling is exactly equivalent to materializing sampled
    family indices and is substantially cheaper.
    """
    if draws != 100_000 or not 0 < alpha < 1:
        raise ValueError("CTAA hierarchical-bootstrap contract differs")
    seeds = sorted(differences_by_seed)
    if len(seeds) != 5:
        raise ValueError("CTAA hierarchical bootstrap requires five paired seeds")
    arrays = []
    for master_seed in seeds:
        values = np.asarray(differences_by_seed[master_seed], dtype=np.int8)
        if values.ndim != 1 or values.size < 1 or not np.isin(values, (-1, 0, 1)).all():
            raise ValueError("CTAA paired family differences differ")
        arrays.append(values)
    generator = np.random.default_rng(seed)
    # Independent family resamples for each bootstrap seed slot and source seed.
    boot = np.empty((5, 5, draws), dtype=np.float32)
    for slot in range(5):
        for seed_index, values in enumerate(arrays):
            counts = np.bincount(values + 1, minlength=3)
            sampled = generator.multinomial(
                values.size,
                counts / values.size,
                size=draws,
            )
            boot[slot, seed_index] = (
                sampled[:, 2] - sampled[:, 0]
            ) / values.size
    selected = generator.integers(0, 5, size=(draws, 5))
    draw_index = np.arange(draws)
    distribution = np.zeros(draws, dtype=np.float32)
    for slot in range(5):
        distribution += boot[slot, selected[:, slot], draw_index]
    distribution /= 5.0
    observed_seed_means = np.asarray([values.mean() for values in arrays])
    return {
        "draws": draws,
        "seeds": len(seeds),
        "families_minimum": min(values.size for values in arrays),
        "observed_mean": float(observed_seed_means.mean()),
        "lower_bound": float(np.quantile(distribution, alpha)),
        "bootstrap_seed": seed,
    }
