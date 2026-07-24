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
    HANKEL_SHIFT_MAXIMUM_EXPECTED,
    build_hankel_shift_capacity_lane,
    build_no_host_capacity_lane,
)
from episode_functor_hankel_completion import (  # noqa: E402
    HankelShiftCompletionProjector,
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


@pytest.mark.parametrize(
    "incidence_mode",
    ("prefix", "random", "commutative"),
)
def test_hankel_shift_maximum_lane_has_exact_isoparametric_receipt(
    incidence_mode: str,
) -> None:
    compiler, query, receipt = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        incidence_mode=incidence_mode,
        random_seed="capacity-control-v1",
    )
    assert isinstance(
        compiler.projector,
        HankelShiftCompletionProjector,
    )
    assert compiler.projector.incidence_mode == incidence_mode
    assert receipt.decode_mode == "hankel-shift"
    assert receipt.compiler_parameters == 64_407_956
    assert receipt.projector_parameters == 19_717_124
    assert receipt.query_parameters == 6_003_489
    assert receipt.added_parameters == 70_411_445
    assert receipt.complete_parameters == 195_493_109
    assert receipt.headroom == 4_506_891
    assert HANKEL_SHIFT_MAXIMUM_EXPECTED.incidence_mode == "prefix"
    del compiler, query
    gc.collect()


def test_unknown_hankel_incidence_fails_closed() -> None:
    with pytest.raises(EFCCapacityError, match="unknown Hankel incidence"):
        build_hankel_shift_capacity_lane(
            external_feature_width=1_728,
            incidence_mode="unregistered",
        )


def test_dual_direct_control_is_exactly_isoparametric() -> None:
    compiler, query, receipt = build_hankel_shift_capacity_lane(
        external_feature_width=1_728,
        decode_mode="direct-base",
    )
    assert receipt.lane == "maximum-hankel-direct"
    assert receipt.decode_mode == "direct-base"
    assert receipt.added_parameters == 70_411_445
    assert receipt.complete_parameters == 195_493_109
    del compiler, query
    gc.collect()
