from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_capacity_lanes import (  # noqa: E402
    CAPACITY_LANES,
    EFCCapacityError,
    build_no_host_capacity_lane,
)


@pytest.mark.parametrize(
    ("name", "added", "complete", "headroom"),
    (
        ("minimal", 4_550_195, 129_631_859, 70_368_141),
        ("wide", 35_625_267, 160_706_931, 39_293_069),
        ("maximum", 60_552_883, 185_634_547, 14_365_453),
    ),
)
def test_capacity_lane_exact_constructor_receipt(
    name: str,
    added: int,
    complete: int,
    headroom: int,
) -> None:
    compiler, query, receipt = build_no_host_capacity_lane(
        name,
        external_feature_width=1_728,
    )
    assert receipt.added_parameters == added
    assert receipt.complete_parameters == complete
    assert receipt.headroom == headroom
    assert compiler.external_feature_width == 1_728
    assert query.external_feature_width == 1_728
    del compiler, query
    gc.collect()


def test_capacity_lanes_are_monotone_and_immutable() -> None:
    sizes = [
        CAPACITY_LANES[name].expected_added_parameters
        for name in ("minimal", "wide", "maximum")
    ]
    assert sizes == sorted(sizes)
    with pytest.raises(TypeError):
        CAPACITY_LANES["other"] = CAPACITY_LANES["minimal"]


def test_unknown_capacity_lane_fails_closed() -> None:
    with pytest.raises(EFCCapacityError, match="unknown EFC capacity lane"):
        build_no_host_capacity_lane(
            "unregistered",
            external_feature_width=1_728,
        )
