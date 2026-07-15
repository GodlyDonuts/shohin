"""Exact audit primitives for static orbit-consensus recurrence.

This module intentionally describes the simplest proposed R9 cell and exposes
when it is only a feed-forward multi-view classifier written as a loop.  It is
not a trainable Shohin mechanism.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_permutations(permutations: torch.Tensor, hypotheses: int) -> None:
    if permutations.ndim != 2 or permutations.shape[1] != int(hypotheses):
        raise ValueError("permutations must have shape [views,hypotheses]")
    target = torch.arange(hypotheses, device=permutations.device)
    if any(not torch.equal(torch.sort(row).values, target) for row in permutations):
        raise ValueError("each orbit action must be a class permutation")


def pull_back_logits(view_logits: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    """Express every transformed view in the base hypothesis coordinates."""
    if view_logits.ndim != 3:
        raise ValueError("view logits must have shape [batch,views,hypotheses]")
    batch, views, hypotheses = view_logits.shape
    validate_permutations(permutations, hypotheses)
    if permutations.shape[0] != views:
        raise ValueError("orbit view count differs")
    indices = permutations.unsqueeze(0).expand(batch, -1, -1)
    return view_logits.gather(-1, indices)


def static_consensus(pulled_logits: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if pulled_logits.ndim != 3:
        raise ValueError("pulled logits must have shape [batch,views,hypotheses]")
    views = pulled_logits.shape[1]
    if weights is None:
        weights = torch.ones(views, dtype=pulled_logits.dtype, device=pulled_logits.device)
    if weights.shape != (views,) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ValueError("weights must be nonnegative with one value per view")
    normalized = weights / weights.sum()
    return torch.einsum("v,bvh->bh", normalized, pulled_logits)


def recurrent_static_consensus(
    initial_logits: torch.Tensor,
    pulled_logits: torch.Tensor,
    steps: int,
    rate: float,
) -> torch.Tensor:
    """Replay an affine consensus cell over evidence that never changes."""
    if initial_logits.ndim != 2 or initial_logits.shape != pulled_logits.shape[::2]:
        raise ValueError("initial logits have the wrong shape")
    if int(steps) <= 0 or not 0.0 < float(rate) <= 1.0:
        raise ValueError("invalid recurrence schedule")
    target = static_consensus(pulled_logits)
    state = initial_logits
    for _ in range(int(steps)):
        state = (1.0 - float(rate)) * state + float(rate) * target
    return state


def closed_form_static_consensus(
    initial_logits: torch.Tensor,
    pulled_logits: torch.Tensor,
    steps: int,
    rate: float,
) -> torch.Tensor:
    """The exact one-shot function represented by recurrent_static_consensus."""
    decay = (1.0 - float(rate)) ** int(steps)
    target = static_consensus(pulled_logits)
    return decay * initial_logits + (1.0 - decay) * target


def orbit_cross_entropy(
    view_logits: torch.Tensor,
    base_targets: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Orbit-output CE; exactly transformed-label data augmentation."""
    pulled = pull_back_logits(view_logits, permutations)
    expanded = base_targets[:, None].expand(-1, pulled.shape[1])
    return F.cross_entropy(
        pulled.reshape(-1, pulled.shape[-1]), expanded.reshape(-1), reduction="mean",
    )


def transformed_label_cross_entropy(
    view_logits: torch.Tensor,
    base_targets: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Ordinary CE after explicitly transforming each view's target label."""
    batch, views, _ = view_logits.shape
    labels = permutations[:, base_targets].transpose(0, 1)
    if labels.shape != (batch, views):
        raise ValueError("transformed labels have the wrong shape")
    return F.cross_entropy(view_logits.reshape(batch * views, -1), labels.reshape(-1))


def static_orbit_syndrome(pulled_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon disagreement, computable without recurrent state."""
    probabilities = pulled_logits.float().softmax(dim=-1)
    mean = probabilities.mean(dim=1, keepdim=True)
    return (
        probabilities * (probabilities.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log())
    ).sum(dim=-1).mean(dim=1)
