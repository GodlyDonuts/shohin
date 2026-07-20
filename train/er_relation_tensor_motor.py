"""Parameter-free recurrent motor for source-deleted finite relation tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class RelationTensorRollout:
    final_state: torch.Tensor
    state_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]


def hard_relation(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Convert model logits to one selected input per active output row."""
    if logits.ndim != 4 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("relation logits must be [batch, rules, output, input]")
    if active.shape != (logits.shape[0], logits.shape[-1]):
        raise ValueError("relation cardinality mask differs")
    valid = active[:, None, None, :].expand_as(logits)
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    selected = masked.argmax(-1)
    relation = F.one_hot(selected, logits.shape[-1]).to(logits.dtype)
    return relation * active[:, None, :, None]


def rollout_relation_tensor(
    initial: torch.Tensor,
    cards: torch.Tensor,
    event_card: torch.Tensor,
    event_halt: torch.Tensor,
    active: torch.Tensor,
) -> RelationTensorRollout:
    """Compose selected relations recurrently; HALT is persistent and pre-apply."""
    if initial.ndim != 3 or initial.shape[-1] != initial.shape[-2]:
        raise ValueError("initial relation state must be [batch, position, entity]")
    batch, width, _ = initial.shape
    if cards.ndim != 4 or cards.shape[0] != batch or cards.shape[-2:] != (width, width):
        raise ValueError("relation cards differ from initial state")
    if event_card.shape != event_halt.shape or event_card.shape[0] != batch:
        raise ValueError("relation event fields differ")
    if active.shape != (batch, width):
        raise ValueError("relation active mask differs")
    if event_card.lt(0).any() or event_card.ge(cards.shape[1]).any():
        raise ValueError("relation event references an invalid card")
    state = initial
    alive = torch.ones(batch, dtype=torch.bool, device=initial.device)
    trajectory = [state.clone()]
    alive_trajectory = [alive.clone()]
    rows = torch.arange(batch, device=initial.device)
    row_mask = active[:, :, None].to(initial.dtype)
    for slot in range(event_card.shape[1]):
        step_active = alive & ~event_halt[:, slot].bool()
        selected = cards[rows, event_card[:, slot]]
        proposal = torch.bmm(selected, state) * row_mask
        state = torch.where(step_active[:, None, None], proposal, state)
        alive = step_active
        trajectory.append(state.clone())
        alive_trajectory.append(alive.clone())
    return RelationTensorRollout(state, tuple(trajectory), tuple(alive_trajectory))
