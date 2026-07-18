#!/usr/bin/env python3
"""Persistent register-augmented GPT forward (architectural SCEB core).

Extends the frozen transformer with a carryable digit register that is written
once per forward and re-injected every read layer. Unlike text-only typed
state, the register is an explicit architectural activation surviving across
host-side step boundaries when the caller reuses ``RegisterState``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model import GPT
from stateful_residual_register import (
    SRRConfig,
    StatefulResidualRegister,
    digits_to_int,
    int_to_digits,
)


@dataclass
class RegisterState:
    digit_ids: torch.Tensor  # [B, 1+n_digits]


class RegisterAugmentedGPT(nn.Module):
    """Frozen GPT + trainable SRR bus. Flagship GPT weights stay frozen."""

    def __init__(self, gpt: GPT, srr: StatefulResidualRegister | None = None):
        super().__init__()
        self.gpt = gpt
        self.srr = srr or StatefulResidualRegister(
            SRRConfig(d_model=gpt.cfg.d_model, n_layer=gpt.cfg.n_layer)
        )
        for p in self.gpt.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        idx: torch.Tensor,
        register_state: RegisterState | None = None,
        teacher_value: int | None = None,
        update_register: bool = True,
    ):
        """Return logits, aux_loss, new RegisterState.

        If ``register_state`` is provided it is injected from the first read
        layer; write layer may refresh it from residual (or teacher_value).
        """
        cfg = self.gpt.cfg
        B, T = idx.shape
        x = self.gpt.tok(idx)
        cos = self.gpt.cos[:T].to(x.device)
        sin = self.gpt.sin[:T].to(x.device)
        aux = None
        if register_state is not None:
            reg_ids = register_state.digit_ids.to(idx.device)
        elif teacher_value is not None:
            reg_ids = int_to_digits(teacher_value, self.srr.cfg.n_digits).unsqueeze(0).repeat(B, 1).to(idx.device)
        else:
            reg_ids = torch.zeros(B, 1 + self.srr.cfg.n_digits, dtype=torch.long, device=idx.device)

        reg_vec = self.srr.encode_registers(reg_ids)
        for layer_index, block in enumerate(self.gpt.blocks):
            x, _ = block(x, cos, sin)
            if layer_index == self.srr.cfg.write_layer and update_register:
                if teacher_value is not None:
                    tgt = int_to_digits(teacher_value, self.srr.cfg.n_digits).unsqueeze(0).repeat(B, 1).to(idx.device)
                    aux = self.srr.aux_loss(x[:, -1, :], tgt)
                    reg_ids = tgt
                    reg_vec = self.srr.encode_registers(reg_ids)
                else:
                    reg_ids = self.srr.predicted_digits(x[:, -1, :])
                    reg_vec = self.srr.encode_registers(reg_ids)
            x = self.srr.inject(x, reg_vec, layer_index)
        logits = self.gpt.head(self.gpt.norm(x))
        return logits, aux, RegisterState(digit_ids=reg_ids.detach())

    def readout_int(self, state: RegisterState) -> list[int]:
        return [digits_to_int(state.digit_ids[i].cpu()) for i in range(state.digit_ids.shape[0])]
