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
    family_ids = [f"family-{index:03d}" for index in range(128)]
    strata = {
        family_id: f"stratum-{(index // 2) % 8}"
        for index, family_id in enumerate(family_ids)
    }
    all_positive = {
        seed: {family_id: 1 for family_id in family_ids} for seed in range(5)
    }
    report = paired_hierarchical_lower(all_positive, stratum_by_family=strata, seed=17)
    assert report["draws"] == 100_000
    assert report["observed_mean"] == 1.0
    assert report["lower_bound"] == 1.0
    assert report["families"] == 128
    assert report["strata"] == 8
    assert report["bit_generator"] == "PCG64"
    assert report["dtype"] == "float64"
    assert (
        report["resampling_unit"] == "crossed_seed_x_shared_family_root_within_stratum"
    )

    balanced = {
        seed: {
            family_id: (-1 if index % 2 == 0 else 1)
            for index, family_id in enumerate(family_ids)
        }
        for seed in range(5)
    }
    neutral = paired_hierarchical_lower(balanced, stratum_by_family=strata, seed=18)
    assert neutral["observed_mean"] == 0.0
    assert neutral["lower_bound"] < 0.0


def test_hierarchical_bootstrap_rejects_wrong_draw_count_or_unpaired_values() -> None:
    families = {"a": "one"}
    with pytest.raises(ValueError, match="contract"):
        paired_hierarchical_lower(
            {seed: {"a": 1} for seed in range(5)},
            stratum_by_family=families,
            draws=999,
            seed=1,
        )
    with pytest.raises(ValueError, match="differences"):
        paired_hierarchical_lower(
            {seed: {"a": (2 if seed == 0 else 1)} for seed in range(5)},
            stratum_by_family=families,
            seed=1,
        )


def test_hierarchical_bootstrap_rejects_anonymous_or_misaligned_family_roots() -> None:
    with pytest.raises(ValueError, match="differences"):
        paired_hierarchical_lower(
            {seed: [1] for seed in range(5)},  # type: ignore[arg-type]
            stratum_by_family={"a": "one"},
            seed=7,
        )
    mismatched = {seed: {"a": 1, "b": 0} for seed in range(5)}
    mismatched[4] = {"a": 1, "c": 0}
    with pytest.raises(ValueError, match="family-root coverage"):
        paired_hierarchical_lower(
            mismatched,
            stratum_by_family={"a": "one", "b": "one"},
            seed=8,
        )
    with pytest.raises(ValueError, match="strata"):
        paired_hierarchical_lower(
            {seed: {"a": 1, "b": 0} for seed in range(5)},
            stratum_by_family={"a": "one"},
            seed=9,
        )
