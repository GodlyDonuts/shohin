"""Matched finite supervision for CTAA recurrent-core falsifier arms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl


Arm = Literal["ctaa_closure", "oprc_closure", "ctaa_no_closure", "ctaa_shuffled_closure"]
ARMS: tuple[Arm, ...] = (
    "ctaa_closure",
    "oprc_closure",
    "ctaa_no_closure",
    "ctaa_shuffled_closure",
)


@dataclass(frozen=True)
class AtomicBatch:
    action: torch.Tensor
    state: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class ClosureBatch:
    first: torch.Tensor
    second: torch.Tensor
    state: torch.Tensor
    composed: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class CoreLossReceipt:
    total: torch.Tensor
    atomic: torch.Tensor
    transition_calls_per_closure_row: int
    call_losses: tuple[torch.Tensor, ...]


def make_core(arm: Arm) -> nn.Module:
    if arm not in ARMS:
        raise ValueError("CTAA training arm differs")
    return OuterProductTransitionControl() if arm == "oprc_closure" else ClosureFeatureTransitionCore()


def tuple_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != (*target.shape, 3):
        raise ValueError("CTAA core loss geometry differs")
    return F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))


def closure_label_derangement(seed: int, actions: torch.Tensor) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    unique = sorted({tuple(int(value) for value in row) for row in actions.tolist()})
    if len(unique) < 2:
        raise ValueError("CTAA closure-label domain is too small")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}|ctaa-shuffle".encode()).digest()[:8], "big"))
    shuffled = list(unique)
    for _ in range(10_000):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(unique, shuffled, strict=True)):
            return dict(zip(unique, shuffled, strict=True))
    raise RuntimeError("CTAA closure-label derangement search exhausted")


def apply_derangement(
    composed: torch.Tensor,
    mapping: Mapping[tuple[int, int, int], tuple[int, int, int]],
) -> torch.Tensor:
    rows = [mapping[tuple(int(value) for value in row)] for row in composed.tolist()]
    return torch.tensor(rows, dtype=torch.long, device=composed.device)


def _apply_copy_batch(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return state.gather(1, action)


def matched_core_loss(
    core: nn.Module,
    arm: Arm,
    atomic: AtomicBatch,
    closure: ClosureBatch,
    *,
    shuffled_mapping: Mapping[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> CoreLossReceipt:
    if arm not in ARMS:
        raise ValueError("CTAA training arm differs")
    atomic_loss = tuple_cross_entropy(core(atomic.action, atomic.state), atomic.output)
    state_one = _apply_copy_batch(closure.first, closure.state)
    first_apply = tuple_cross_entropy(core(closure.first, closure.state), state_one)
    second_apply = tuple_cross_entropy(core(closure.second, state_one), closure.output)
    if arm == "ctaa_no_closure":
        third = tuple_cross_entropy(core(closure.first, closure.state), state_one)
        fourth = tuple_cross_entropy(core(closure.second, state_one), closure.output)
    else:
        composed_target = closure.composed
        composed_output = closure.output
        if arm == "ctaa_shuffled_closure":
            if shuffled_mapping is None:
                raise ValueError("CTAA shuffled arm lacks its fixed derangement")
            composed_target = apply_derangement(closure.composed, shuffled_mapping)
            composed_output = _apply_copy_batch(composed_target, closure.state)
        third = tuple_cross_entropy(
            core(closure.second, closure.first),
            composed_target,
        )
        fourth = tuple_cross_entropy(
            core(composed_target, closure.state),
            composed_output,
        )
    calls = (first_apply, second_apply, third, fourth)
    total = atomic_loss + torch.stack(calls).mean()
    return CoreLossReceipt(
        total=total,
        atomic=atomic_loss,
        transition_calls_per_closure_row=4,
        call_losses=calls,
    )
