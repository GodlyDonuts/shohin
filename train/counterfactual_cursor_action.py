"""Isolated finite-state cursor and 192-scalar final-head Q sidecar."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


SELECT = 0
EXECUTE = 1
HALT_PENDING = 2
HALT = 3
PHASE_NAMES = ("SELECT", "EXECUTE", "HALT_PENDING", "HALT")


@dataclass(frozen=True)
class EventTokenManifest:
    operation_ids: tuple[int, ...]
    commit_id: int
    done_id: int
    eos_id: int
    max_cursor: int = 4

    def __post_init__(self):
        values = self.operation_ids + (self.commit_id, self.done_id, self.eos_id)
        if len(self.operation_ids) != 4 or len(set(values)) != len(values):
            raise ValueError("event token IDs must be seven distinct integers")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("event token IDs must be nonnegative integers")
        if self.max_cursor != 4:
            raise ValueError("cursor-action v1 requires max_cursor=4")


@dataclass(frozen=True)
class DecodeState:
    cursor: torch.Tensor
    phase: torch.Tensor

    def validate(self) -> None:
        if self.cursor.dtype != torch.long or self.phase.dtype != torch.long:
            raise ValueError("cursor and phase must be int64 tensors")
        if self.cursor.shape != self.phase.shape or self.cursor.ndim != 1:
            raise ValueError("cursor and phase must be same-shape batch vectors")
        if bool(((self.cursor < 0) | (self.cursor > 4)).any()):
            raise ValueError("cursor is outside [0,4]")
        if bool(((self.phase < SELECT) | (self.phase > HALT)).any()):
            raise ValueError("phase is outside the registered state set")


def initial_state(batch: int, device=None) -> DecodeState:
    if batch <= 0:
        raise ValueError("batch must be positive")
    return DecodeState(
        cursor=torch.zeros(batch, dtype=torch.long, device=device),
        phase=torch.full((batch,), SELECT, dtype=torch.long, device=device),
    )


def _is_operation(tokens: torch.Tensor, manifest: EventTokenManifest) -> torch.Tensor:
    result = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in manifest.operation_ids:
        result |= tokens.eq(token_id)
    return result


def advance_state(
    state: DecodeState, emitted_tokens: torch.Tensor, manifest: EventTokenManifest,
) -> tuple[DecodeState, torch.Tensor]:
    """Advance from emitted token IDs and return the next state plus violations."""
    state.validate()
    if emitted_tokens.dtype != torch.long or emitted_tokens.shape != state.cursor.shape:
        raise ValueError("emitted_tokens must be an int64 batch vector")

    cursor = state.cursor.clone()
    phase = state.phase.clone()
    operations = _is_operation(emitted_tokens, manifest)
    select = phase.eq(SELECT)
    execute = phase.eq(EXECUTE)
    pending = phase.eq(HALT_PENDING)

    valid_operation = select & cursor.lt(manifest.max_cursor) & operations
    valid_commit = execute & emitted_tokens.eq(manifest.commit_id)
    valid_done = select & cursor.eq(manifest.max_cursor) & emitted_tokens.eq(manifest.done_id)
    valid_eos = pending & emitted_tokens.eq(manifest.eos_id)
    premature_done = select & cursor.lt(manifest.max_cursor) & emitted_tokens.eq(manifest.done_id)
    terminal_operation = select & cursor.eq(manifest.max_cursor) & operations
    violations = premature_done | terminal_operation

    cursor = torch.where(valid_operation, cursor + 1, cursor)
    phase = torch.where(valid_operation, torch.full_like(phase, EXECUTE), phase)
    phase = torch.where(valid_commit, torch.full_like(phase, SELECT), phase)
    phase = torch.where(valid_done, torch.full_like(phase, HALT_PENDING), phase)
    phase = torch.where(valid_eos, torch.full_like(phase, HALT), phase)
    next_state = DecodeState(cursor=cursor, phase=phase)
    next_state.validate()
    return next_state, violations


def centered_cursor_bits(cursor: torch.Tensor) -> torch.Tensor:
    if cursor.dtype != torch.long:
        raise ValueError("cursor must be int64")
    if bool(((cursor < 0) | (cursor > 4)).any()):
        raise ValueError("cursor is outside [0,4]")
    shifts = torch.arange(3, device=cursor.device, dtype=torch.long)
    bits = ((cursor.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.mul(2.0).sub(1.0)


class CursorQSidecar(nn.Module):
    """Project centered cursor bits into one 64-wide attention-query head."""

    def __init__(self, head_dim: int = 64):
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.projection = nn.Linear(3, head_dim, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, cursor: torch.Tensor, select_mask: torch.Tensor) -> torch.Tensor:
        if cursor.shape != select_mask.shape:
            raise ValueError("cursor and select_mask shapes differ")
        if select_mask.dtype != torch.bool:
            raise ValueError("select_mask must be boolean")
        delta = self.projection(centered_cursor_bits(cursor).to(self.projection.weight.dtype))
        return delta * select_mask.unsqueeze(-1).to(delta.dtype)

    def metadata(self) -> dict[str, int | str]:
        return {
            "schema": "counterfactual_cursor_q_sidecar_v1",
            "parameters": self.projection.weight.numel(),
            "cursor_bits": 3,
            "head_dim": self.projection.out_features,
            "encoding": "centered_binary_minus_one_plus_one",
        }


def selector_grid(cursor: torch.Tensor, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Place a supplied selector cursor only at each sequence's final position."""
    if cursor.dtype != torch.long or cursor.ndim != 1:
        raise ValueError("cursor must be an int64 batch vector")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    grid = cursor[:, None].expand(-1, tokens).clone()
    mask = torch.zeros_like(grid, dtype=torch.bool)
    mask[:, -1] = True
    return grid, mask


def teacher_forced_state_grid(
    input_ids: torch.Tensor, prompt_last: torch.Tensor, manifest: EventTokenManifest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct post-prefix state for every teacher-forced input position.

    Source tokens, including operation words, are inert. The final prompt token
    predicts the first operation under cursor zero/SELECT. Completion token
    events update state before that token's logits predict the following token.
    """
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("input_ids must be an int64 [batch,tokens] tensor")
    batch, tokens = input_ids.shape
    if prompt_last.dtype != torch.long or prompt_last.shape != (batch,):
        raise ValueError("prompt_last must be an int64 batch vector")
    if bool(((prompt_last < 0) | (prompt_last >= tokens)).any()):
        raise ValueError("prompt_last is outside the sequence")

    state = initial_state(batch, input_ids.device)
    cursors = torch.zeros_like(input_ids)
    phases = torch.full_like(input_ids, SELECT)
    violations = torch.zeros_like(input_ids, dtype=torch.bool)
    for position in range(tokens):
        after_prompt = position > prompt_last
        if bool(after_prompt.any()):
            candidate, invalid = advance_state(state, input_ids[:, position], manifest)
            state = DecodeState(
                cursor=torch.where(after_prompt, candidate.cursor, state.cursor),
                phase=torch.where(after_prompt, candidate.phase, state.phase),
            )
            violations[:, position] = after_prompt & invalid
        cursors[:, position] = state.cursor
        phases[:, position] = state.phase
    active = torch.arange(tokens, device=input_ids.device)[None, :] >= prompt_last[:, None]
    select_mask = active & phases.eq(SELECT)
    return cursors, select_mask, violations


class FrozenBaseCursorSelector(nn.Module):
    """Frozen GPT plus the separately serializable cursor sidecar."""

    def __init__(self, base: nn.Module, layer: int = -1, head: int = 0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        head_dim = base.cfg.d_model // base.cfg.n_head
        self.sidecar = CursorQSidecar(head_dim)
        self.layer = layer if layer >= 0 else base.cfg.n_layer + layer
        self.head = head
        if self.layer != base.cfg.n_layer - 1 or self.head != 0:
            raise ValueError("cursor-action v1 is frozen to final block, head zero")

    def forward(self, idx: torch.Tensor, cursor: torch.Tensor, targets=None):
        grid, select_mask = selector_grid(cursor, idx.shape[1])
        q_delta = self.sidecar(grid, select_mask)
        return self.base(
            idx, targets=targets, q_delta=q_delta,
            q_delta_layer=self.layer, q_delta_head=self.head,
        )

    def sidecar_state(self) -> dict[str, object]:
        return {
            "metadata": {**self.sidecar.metadata(), "layer": self.layer, "head": self.head},
            "adapter_state": self.sidecar.state_dict(),
        }
