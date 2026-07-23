"""Preregistered finite-family and paired hierarchical CTAA statistics."""

from __future__ import annotations

from typing import Mapping

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


def holm_rejections(
    pvalues: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, bool]:
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
    differences_by_seed: Mapping[int, Mapping[str, int]],
    *,
    stratum_by_family: Mapping[str, str],
    draws: int = 100_000,
    alpha: float = 0.05,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap crossed seeds and shared semantic-family roots within strata.

    The five training seeds evaluate the same semantic family roots.  A valid
    resample therefore selects seed IDs and shared root IDs, then applies each
    sampled root to every selected seed.  Sampling anonymous outcomes within
    each seed independently would destroy that crossing and understate shared
    family variance.

    Root sampling is implemented exactly with multinomial counts.  For a fixed
    five-slot seed multiplicity, every root has an integer sum in ``[-5, 5]``;
    drawing roots with replacement is therefore equivalent to drawing from
    those eleven categories and avoids materializing a draws-by-families array.
    """
    if (
        draws != 100_000
        or alpha != 0.05
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("CTAA hierarchical-bootstrap contract differs")
    seeds = sorted(differences_by_seed)
    if len(seeds) != 5:
        raise ValueError("CTAA hierarchical bootstrap requires five paired seeds")
    if any(type(master_seed) is not int for master_seed in seeds):
        raise ValueError("CTAA hierarchical bootstrap seed identity differs")
    family_ids: list[str] | None = None
    arrays: list[np.ndarray] = []
    for master_seed in seeds:
        family_map = differences_by_seed[master_seed]
        if not isinstance(family_map, Mapping) or not family_map:
            raise ValueError("CTAA paired family differences differ")
        current_ids = sorted(family_map)
        if (
            any(not isinstance(item, str) or not item for item in current_ids)
            or family_ids is not None
            and current_ids != family_ids
        ):
            raise ValueError("CTAA shared family-root coverage differs")
        if family_ids is None:
            family_ids = current_ids
        raw_values = [family_map[item] for item in current_ids]
        if any(
            type(value) is not int or value not in (-1, 0, 1) for value in raw_values
        ):
            raise ValueError("CTAA paired family differences differ")
        values = np.asarray(raw_values, dtype=np.int8)
        arrays.append(values)
    assert family_ids is not None
    if set(stratum_by_family) != set(family_ids) or any(
        not isinstance(stratum_by_family[item], str) or not stratum_by_family[item]
        for item in family_ids
    ):
        raise ValueError("CTAA frozen family strata differ")
    strata: dict[str, np.ndarray] = {}
    for stratum in sorted(set(stratum_by_family.values())):
        indices = np.asarray(
            [
                index
                for index, family_id in enumerate(family_ids)
                if stratum_by_family[family_id] == stratum
            ],
            dtype=np.int64,
        )
        if indices.size < 1:
            raise ValueError("CTAA frozen family strata differ")
        strata[stratum] = indices

    matrix = np.stack(arrays, axis=0)
    generator = np.random.Generator(np.random.PCG64(seed))
    selected = generator.integers(0, 5, size=(draws, 5), dtype=np.int8)
    seed_counts = np.stack(
        [np.bincount(row, minlength=5) for row in selected], axis=0
    ).astype(np.int8, copy=False)
    unique_counts, inverse = np.unique(seed_counts, axis=0, return_inverse=True)
    distribution = np.empty(draws, dtype=np.float64)
    category_values = np.arange(-5, 6, dtype=np.int64)
    total_families = len(family_ids)
    for group, multiplicities in enumerate(unique_counts):
        draw_indices = np.flatnonzero(inverse == group)
        sampled_sum = np.zeros(draw_indices.size, dtype=np.int64)
        weighted_roots = multiplicities.astype(np.int16) @ matrix.astype(np.int16)
        for indices in strata.values():
            values = weighted_roots[indices]
            category_counts = np.bincount(values + 5, minlength=11)
            sampled = generator.multinomial(
                indices.size,
                category_counts / indices.size,
                size=draw_indices.size,
            )
            sampled_sum += sampled @ category_values
        distribution[draw_indices] = sampled_sum / (5.0 * total_families)
    return {
        "draws": draws,
        "seeds": len(seeds),
        "families": total_families,
        "strata": len(strata),
        "observed_mean": float(matrix.astype(np.float64).mean()),
        "lower_bound": float(np.quantile(distribution, alpha, method="linear")),
        "bootstrap_seed": seed,
        "bit_generator": "PCG64",
        "dtype": "float64",
        "resampling_unit": "crossed_seed_x_shared_family_root_within_stratum",
    }
