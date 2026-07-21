from __future__ import annotations

import pytest
import torch

from ctaa_neural_core import ClosureFeatureTransitionCore
from profile_ctaa_resources import PROFILE_DEPTHS, _runtime_profile


def test_runtime_profile_covers_real_packet_depth_geometry() -> None:
    assert PROFILE_DEPTHS == (1, 16, 32, 39)
    report = _runtime_profile(
        ClosureFeatureTransitionCore(),
        torch.device("cpu"),
        batch_size=1,
        repeats=1,
        depth=39,
    )
    assert report["active_depth"] == 39
    assert report["rows_per_second"] > 0


def test_runtime_profile_rejects_impossible_depth_64() -> None:
    with pytest.raises(ValueError, match="depth"):
        _runtime_profile(
            ClosureFeatureTransitionCore(),
            torch.device("cpu"),
            batch_size=1,
            repeats=1,
            depth=64,
        )
