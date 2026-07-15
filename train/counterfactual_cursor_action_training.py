"""Shared frozen-base training mechanics for the R12 cursor-action canary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from counterfactual_cursor_action import (
    CursorQSidecar,
    CursorTableSidecar,
    RankOneQueryLoRA,
    selector_position_grid,
)
from counterfactual_cursor_action_objectives import restricted_action_logits


ARMS = (
    "orbit_interchange",
    "ordinary_loss",
    "relation_sham",
    "source_only",
    "cursor_table",
    "text_cursor_lora",
)
SIDECAR_ARMS = frozenset((
    "orbit_interchange", "ordinary_loss", "relation_sham", "source_only",
))


@dataclass(frozen=True)
class AdapterSpec:
    arm: str
    adapter_type: str
    parameters: int
    retained_cursor_bits: int
    applies_at_all_tokens: bool


def freeze_base(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def build_adapter(arm: str, cfg, seed: int) -> tuple[nn.Module, AdapterSpec]:
    if arm not in ARMS:
        raise ValueError(f"unknown cursor-action arm: {arm}")
    head_dim = cfg.d_model // cfg.n_head
    if arm in SIDECAR_ARMS:
        adapter = CursorQSidecar(head_dim)
        adapter_type = "centered_three_bit_q_sidecar"
        retained_bits = 0 if arm == "source_only" else 3
        all_tokens = False
    elif arm == "cursor_table":
        adapter = CursorTableSidecar(head_dim)
        adapter_type = "eight_entry_cursor_q_table"
        retained_bits = 3
        all_tokens = False
    else:
        adapter = RankOneQueryLoRA(cfg.d_model, head_dim, seed=seed)
        adapter_type = "rank_one_text_cursor_q_lora"
        retained_bits = 0
        all_tokens = True
    parameters = sum(parameter.numel() for parameter in adapter.parameters())
    expected = {
        "orbit_interchange": 192,
        "ordinary_loss": 192,
        "relation_sham": 192,
        "source_only": 192,
        "cursor_table": 512,
        "text_cursor_lora": 640,
    }[arm]
    if parameters != expected:
        raise ValueError(f"{arm} has {parameters} parameters instead of {expected}")
    return adapter, AdapterSpec(
        arm=arm,
        adapter_type=adapter_type,
        parameters=parameters,
        retained_cursor_bits=retained_bits,
        applies_at_all_tokens=all_tokens,
    )


@torch.no_grad()
def encode_before_final_block(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Cache the exact frozen prefix before the only intervened block."""
    if input_ids.dtype != torch.long or input_ids.ndim != 2:
        raise ValueError("input_ids must be an int64 [batch,tokens] tensor")
    if model.cfg.n_loop != 1:
        raise ValueError("cursor-action v1 requires n_loop=1")
    if input_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("input exceeds the base context length")
    hidden = model.tok(input_ids)
    cos = model.cos[:input_ids.shape[1]].to(hidden.device)
    sin = model.sin[:input_ids.shape[1]].to(hidden.device)
    for block in model.blocks[:-1]:
        hidden, _ = block(hidden, cos, sin)
    return hidden.detach()


def logits_from_final_block_cache(
    model: nn.Module,
    prefix_hidden: torch.Tensor,
    prompt_last: torch.Tensor,
    adapter: nn.Module,
    arm: str,
    cursor: torch.Tensor | None,
    label_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the intervened final block and project only selector positions."""
    if arm not in ARMS:
        raise ValueError(f"unknown cursor-action arm: {arm}")
    if prefix_hidden.ndim != 3 or prefix_hidden.shape[-1] != model.cfg.d_model:
        raise ValueError("prefix_hidden has the wrong shape")
    batch, tokens, _ = prefix_hidden.shape
    if prompt_last.dtype != torch.long or prompt_last.shape != (batch,):
        raise ValueError("prompt_last must be an int64 batch vector")
    if bool(((prompt_last < 0) | (prompt_last >= tokens)).any()):
        raise ValueError("prompt_last is outside the cached token grid")
    if arm == "text_cursor_lora":
        if cursor is not None:
            raise ValueError("text cursor control may not receive side-state cursor values")
        q_delta = None
        q_adapter = adapter
    else:
        if cursor is None or cursor.dtype != torch.long or cursor.shape != (batch,):
            raise ValueError("sidecar arm requires an int64 cursor batch vector")
        applied_cursor = torch.zeros_like(cursor) if arm == "source_only" else cursor
        grid, select_mask = selector_position_grid(applied_cursor, prompt_last, tokens)
        q_delta = adapter(grid, select_mask)
        q_adapter = None
    cos = model.cos[:tokens].to(prefix_hidden.device)
    sin = model.sin[:tokens].to(prefix_hidden.device)
    hidden, _ = model.blocks[-1](
        prefix_hidden, cos, sin, q_delta=q_delta, q_adapter=q_adapter,
        q_delta_head=0,
    )
    selected = hidden[torch.arange(batch, device=hidden.device), prompt_last]
    full_logits = model.head(model.norm(selected))
    action_logits = restricted_action_logits(full_logits, label_token_ids)
    return full_logits, action_logits


def adapter_state_payload(adapter: nn.Module, spec: AdapterSpec) -> dict[str, object]:
    return {
        "adapter_spec": {
            "arm": spec.arm,
            "adapter_type": spec.adapter_type,
            "parameters": spec.parameters,
            "retained_cursor_bits": spec.retained_cursor_bits,
            "applies_at_all_tokens": spec.applies_at_all_tokens,
            "layer": "final",
            "head": 0,
            "query_only": True,
        },
        "adapter_state": {
            name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()
        },
    }
