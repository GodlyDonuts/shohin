"""Frozen relation objectives for the R12 cursor-action neural canary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


LABEL_COUNT = 5
FROZEN_LABEL_TOKEN_IDS = (820, 5498, 4307, 7486, 2165)


@dataclass(frozen=True)
class RelationLosses:
    cursor_interchange: torch.Tensor
    adjacent_equivariance: torch.Tensor
    renderer_invariance: torch.Tensor
    cursor_pairs: int
    adjacent_pairs: int
    renderer_pairs: int

    def total(self) -> torch.Tensor:
        return (
            self.cursor_interchange
            + self.adjacent_equivariance
            + self.renderer_invariance
        )


def restricted_action_logits(
    full_logits: torch.Tensor, label_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the five preregistered action-token logits in frozen label order."""
    if full_logits.ndim < 2:
        raise ValueError("full logits must include a vocabulary dimension")
    if label_token_ids.dtype != torch.long or label_token_ids.shape != (LABEL_COUNT,):
        raise ValueError("label_token_ids must be an int64 five-vector")
    if tuple(label_token_ids.detach().cpu().tolist()) != FROZEN_LABEL_TOKEN_IDS:
        raise ValueError("label_token_ids do not match the frozen action-token contract")
    if bool(((label_token_ids < 0) | (label_token_ids >= full_logits.shape[-1])).any()):
        raise ValueError("an action token ID is outside the vocabulary")
    if torch.unique(label_token_ids).numel() != LABEL_COUNT:
        raise ValueError("action token IDs must be distinct")
    return full_logits.index_select(-1, label_token_ids)


def _validate_relation_tensors(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 4 or logits.shape[0] != 2 or logits.shape[2:] != (
        LABEL_COUNT, LABEL_COUNT,
    ):
        raise ValueError("relation logits must have shape [2,renderers,5,5]")
    if labels.dtype != torch.long or labels.shape != logits.shape[:-1]:
        raise ValueError("relation labels must be an int64 [2,renderers,5] tensor")
    expected = torch.arange(LABEL_COUNT, device=labels.device)
    sorted_labels = labels.sort(dim=-1).values
    if not bool(sorted_labels.eq(expected).all()):
        raise ValueError("every source must contain one target from each action class")
    if logits.shape[1] < 2:
        raise ValueError("renderer relations require at least two renderers")


def _centered_logit_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float() - left.float().mean(dim=-1, keepdim=True)
    right = right.float() - right.float().mean(dim=-1, keepdim=True)
    return (left - right).pow(2).mean(dim=-1)


def cursor_interchange_loss(
    logits: torch.Tensor, labels: torch.Tensor, *, sham: bool, margin: float,
) -> tuple[torch.Tensor, int]:
    """Require a cursor swap to raise the donor cursor's action preference.

    For every ordered pair of distinct cursor states, the treatment compares the
    donor action's logit under the donor and receiver cursors.  The sham rotates
    donor identities by one while retaining exactly the same tensor operations.
    """
    _validate_relation_tensors(logits, labels)
    if margin <= 0:
        raise ValueError("cursor interchange margin must be positive")
    flat_logits = logits.reshape(-1, LABEL_COUNT, LABEL_COUNT)
    flat_labels = labels.reshape(-1, LABEL_COUNT)
    terms = []
    for receiver in range(LABEL_COUNT):
        for donor in range(LABEL_COUNT):
            if receiver == donor:
                continue
            relation_donor = (donor + 1) % LABEL_COUNT if sham else donor
            target = flat_labels[:, relation_donor]
            donor_score = flat_logits[:, donor].gather(1, target[:, None]).squeeze(1)
            receiver_score = flat_logits[:, receiver].gather(1, target[:, None]).squeeze(1)
            terms.append(F.softplus(float(margin) - (donor_score - receiver_score)))
    values = torch.stack(terms, dim=1)
    return values.mean(), values.numel()


def adjacent_equivariance_loss(
    logits: torch.Tensor, *, swap_index: int, sham: bool,
) -> tuple[torch.Tensor, int]:
    """Match a clause transposition to the corresponding cursor transposition."""
    if type(swap_index) is not int or not 0 <= swap_index < 3:
        raise ValueError("swap_index must be one of 0, 1, or 2")
    if logits.ndim != 4 or logits.shape[0] != 2 or logits.shape[2:] != (
        LABEL_COUNT, LABEL_COUNT,
    ):
        raise ValueError("relation logits must have shape [2,renderers,5,5]")
    right_cursors = torch.arange(LABEL_COUNT, device=logits.device)
    if sham:
        left_cursors = (right_cursors + 1) % LABEL_COUNT
    else:
        left_cursors = right_cursors.clone()
        left_cursors[swap_index] = swap_index + 1
        left_cursors[swap_index + 1] = swap_index
    left = logits[0].index_select(1, left_cursors)
    right = logits[1]
    values = _centered_logit_mse(left, right)
    return values.mean(), values.numel()


def renderer_invariance_loss(
    logits: torch.Tensor, *, sham: bool,
) -> tuple[torch.Tensor, int]:
    """Match content-identical renderers at each cursor, or a wrong sham cursor."""
    if logits.ndim != 4 or logits.shape[0] != 2 or logits.shape[2:] != (
        LABEL_COUNT, LABEL_COUNT,
    ):
        raise ValueError("relation logits must have shape [2,renderers,5,5]")
    renderer_count = logits.shape[1]
    values = []
    cursor = torch.arange(LABEL_COUNT, device=logits.device)
    paired_cursor = (cursor + 1) % LABEL_COUNT if sham else cursor
    for side in range(2):
        for left_renderer in range(renderer_count):
            for right_renderer in range(left_renderer + 1, renderer_count):
                left = logits[side, left_renderer]
                right = logits[side, right_renderer].index_select(0, paired_cursor)
                values.append(_centered_logit_mse(left, right))
    stacked = torch.stack(values)
    return stacked.mean(), stacked.numel()


def relation_losses(
    logits: torch.Tensor, labels: torch.Tensor, *, swap_index: int,
    sham: bool = False, cursor_margin: float = 1.0,
) -> RelationLosses:
    """Compute all three frozen relation losses for one complete training unit."""
    _validate_relation_tensors(logits, labels)
    cursor, cursor_pairs = cursor_interchange_loss(
        logits, labels, sham=sham, margin=cursor_margin,
    )
    adjacent, adjacent_pairs = adjacent_equivariance_loss(
        logits, swap_index=swap_index, sham=sham,
    )
    renderer, renderer_pairs = renderer_invariance_loss(logits, sham=sham)
    return RelationLosses(
        cursor_interchange=cursor,
        adjacent_equivariance=adjacent,
        renderer_invariance=renderer,
        cursor_pairs=cursor_pairs,
        adjacent_pairs=adjacent_pairs,
        renderer_pairs=renderer_pairs,
    )
