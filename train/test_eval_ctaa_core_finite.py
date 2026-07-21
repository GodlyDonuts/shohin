from __future__ import annotations

import torch
import torch.nn.functional as F

from eval_ctaa_core_finite import score_core


class ExactPointerCore(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        target = right.gather(1, left)
        return F.one_hot(target, 3).float() * 20.0


class IdentityMutation(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del left
        return F.one_hot(right, 3).float() * 20.0


def test_exact_pointer_core_passes_every_finite_axis() -> None:
    report = score_core(ExactPointerCore(), torch.device("cpu"))
    assert set(report) == {"train", "development", "confirmation"}
    for axis in report.values():
        assert axis["atomic_cases"] == 243
        assert axis["two_action_cases"] == 2_187
        assert axis["atomic_exact"] == 1.0
        assert axis["two_action_exact"] == 1.0
        assert axis["composition_exact"] == 1.0
        assert axis["route_agreement"] == 1.0


def test_identity_executor_mutation_fails_finite_axes() -> None:
    report = score_core(IdentityMutation(), torch.device("cpu"))
    assert all(axis["atomic_exact"] < 1.0 for axis in report.values())
    assert all(axis["two_action_exact"] < 1.0 for axis in report.values())
