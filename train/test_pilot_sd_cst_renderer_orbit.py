from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from pilot_sd_cst_renderer_orbit import (  # noqa: E402
    _finite_uniform_span_loss,
    _orbit_consistency,
    _query_span_mask,
    partition_rows,
)


class _Row:
    def __init__(self, start: int, end: int) -> None:
        self.query_span = (start, end)


def test_partition_is_deterministic_and_disjoint() -> None:
    rows = [{"id": f"row-{index}"} for index in range(20)]
    first = partition_rows(rows, 12, 4)
    second = partition_rows(list(reversed(rows)), 12, 4)
    assert first == second
    assert {row["id"] for row in first[0]}.isdisjoint(row["id"] for row in first[1])


def test_query_span_mask_is_exact() -> None:
    mask = _query_span_mask([_Row(2, 4), _Row(0, 1)], 5, torch.device("cpu"))
    assert mask.tolist() == [
        [False, False, True, True, False],
        [True, False, False, False, False],
    ]


def test_orbit_consistency_zero_for_identical_views() -> None:
    base = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    logits = torch.stack((base, base, base, base), dim=1).reshape(8, 2)
    assert float(_orbit_consistency(logits, families=2, views=4)) < 1e-7


def test_orbit_consistency_detects_view_disagreement() -> None:
    logits = torch.tensor(
        [
            [10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
        ]
    )
    assert float(_orbit_consistency(logits, families=1, views=4)) > 0.5


def test_uniform_span_loss_is_finite_with_masked_negative_infinity() -> None:
    logits = torch.tensor(
        [[[-torch.inf, 2.0, 0.0, -torch.inf], [-torch.inf] * 4]],
        requires_grad=True,
    )
    target = torch.tensor([[[False, True, True, False], [False, False, False, False]]])
    active = torch.tensor([[True, False]])
    loss = _finite_uniform_span_loss(logits, target, active)
    expected = -(torch.tensor([2.0, 0.0]).log_softmax(0)).mean()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad[0, 0, 1:3]).all()
