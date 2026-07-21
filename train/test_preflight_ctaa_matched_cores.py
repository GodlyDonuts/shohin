from __future__ import annotations

import torch

from ctaa_neural_core import OuterProductTransitionControl
from preflight_ctaa_matched_cores import (
    arbitrary_targets,
    closure_targets,
    finite_pairs,
    optimize,
)


def test_preflight_finite_tables_have_exact_geometry_and_identity() -> None:
    left, right = finite_pairs(torch.device("cpu"))
    targets = closure_targets(left, right)
    arbitrary = arbitrary_targets(91, torch.device("cpu"))
    assert left.shape == right.shape == targets.shape == arbitrary.shape == (729, 3)
    assert torch.unique(
        OuterProductTransitionControl().features(left, right),
        dim=0,
    ).shape[0] == 729
    assert torch.equal(targets, right.gather(1, left))


def test_short_optimization_smoke_reports_bounded_metrics() -> None:
    torch.manual_seed(7)
    left, right = finite_pairs(torch.device("cpu"))
    model = OuterProductTransitionControl(hidden=1184)
    result = optimize(
        model,
        left,
        right,
        closure_targets(left, right),
        learning_rate=1e-3,
        max_steps=2,
    )
    assert result["optimizer_steps"] == 2
    assert 0.0 <= result["exact_accuracy"] <= 1.0
    assert 0.0 <= result["coordinate_accuracy"] <= 1.0
