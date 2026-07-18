#!/usr/bin/env python3
"""Discrete Controller Heads — architectural split of op/cursor/done from LM.

Frozen GPT trunk. Three small heads read the last-token residual at a chosen
layer and predict:
  - next opcode (add/subtract/multiply/remainder/horner/HALT)
  - operand (bucketed / free int via digit registers)
  - done bit

This is the trainable half of SCEB without requiring the LM to emit arithmetic.
Inference pairs these heads with host apply_op (SCEB-A) or SRR value bus (SCEB-B).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

OPCODES = ("add", "subtract", "multiply", "remainder", "horner", "HALT")
OP2ID = {o: i for i, o in enumerate(OPCODES)}


@dataclass
class ControllerHeadConfig:
    d_model: int = 576
    n_opcodes: int = len(OPCODES)
    max_operand: int = 1000  # classify operand in 0..max_operand-1; horner packed separately
    read_layer: int = 29


class DiscreteControllerHeads(nn.Module):
    def __init__(self, cfg: ControllerHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.op_head = nn.Linear(cfg.d_model, cfg.n_opcodes)
        self.operand_head = nn.Linear(cfg.d_model, cfg.max_operand)
        self.done_head = nn.Linear(cfg.d_model, 2)

    def forward(self, residual_last: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "op_logits": self.op_head(residual_last),
            "operand_logits": self.operand_head(residual_last),
            "done_logits": self.done_head(residual_last),
        }

    def loss(self, residual_last: torch.Tensor, op_id: torch.Tensor, operand: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        out = self.forward(residual_last)
        return (
            F.cross_entropy(out["op_logits"], op_id)
            + F.cross_entropy(out["operand_logits"], operand.clamp(0, self.cfg.max_operand - 1))
            + F.cross_entropy(out["done_logits"], done)
        )

    @torch.no_grad()
    def decode(self, residual_last: torch.Tensor) -> list[dict]:
        out = self.forward(residual_last)
        ops = out["op_logits"].argmax(-1)
        operands = out["operand_logits"].argmax(-1)
        dones = out["done_logits"].argmax(-1)
        rows = []
        for i in range(ops.shape[0]):
            rows.append(
                {
                    "op": OPCODES[int(ops[i].item())],
                    "operand": int(operands[i].item()),
                    "done": int(dones[i].item()),
                }
            )
        return rows


def capture_layer_residual(model, idx: torch.Tensor, layer: int) -> torch.Tensor:
    """Run frozen GPT through ``layer`` and return last-token residual [B,d]."""
    if model.cfg.n_loop != 1:
        raise ValueError("controller heads require n_loop=1")
    x = model.tok(idx)
    cos = model.cos[: idx.shape[1]].to(x.device)
    sin = model.sin[: idx.shape[1]].to(x.device)
    for block in model.blocks[: layer + 1]:
        x, _ = block(x, cos, sin)
    return x[:, -1, :]
