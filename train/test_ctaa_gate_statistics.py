from __future__ import annotations

import pytest

from ctaa_gate_statistics import (
    holm_rejections,
    one_sided_exact_lower,
    one_sided_family_pvalue,
    paired_hierarchical_lower,
)


def test_exact_family_bound_and_holm_are_not_naive_point_estimates() -> None:
    lower = one_sided_exact_lower(100, 100)
    assert 0.96 < lower < 1.0
    assert one_sided_family_pvalue(100, 100, 0.9) < 0.001
    rejected = holm_rejections({"a": 0.001, "b": 0.02, "c": 0.9})
    assert rejected == {"a": True, "b": True, "c": False}


def test_hierarchical_bootstrap_preserves_five_seed_and_family_levels() -> None:
    all_positive = {seed: [1] * 128 for seed in range(5)}
    report = paired_hierarchical_lower(all_positive, seed=17)
    assert report["draws"] == 100_000
    assert report["observed_mean"] == 1.0
    assert report["lower_bound"] == 1.0

    balanced = {seed: [-1, 1] * 64 for seed in range(5)}
    neutral = paired_hierarchical_lower(balanced, seed=18)
    assert neutral["observed_mean"] == 0.0
    assert neutral["lower_bound"] < 0.0


def test_hierarchical_bootstrap_rejects_wrong_draw_count_or_unpaired_values() -> None:
    with pytest.raises(ValueError, match="contract"):
        paired_hierarchical_lower({seed: [1] for seed in range(5)}, draws=999, seed=1)
    with pytest.raises(ValueError, match="differences"):
        paired_hierarchical_lower(
            {seed: ([2] if seed == 0 else [1]) for seed in range(5)},
            seed=1,
        )
