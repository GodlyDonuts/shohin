#!/usr/bin/env python3
"""Stateful Residual Register (SRR) — architectural register bus for SCEB-B.

Frozen GPT backbone. A small bank of decimal digit embeddings is written from
late residual and re-injected into selected layers. Aux CE supervises the
integer state; LM path optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRRConfig:
    d_model: int = 576
    n_digits: int = 8  # absolute value digits, least-significant first
    write_layer: int = 17
    read_layers: tuple[int, ...] = (17, 21, 25, 29)
    n_layer: int = 30


def int_to_digits(value: int, n_digits: int) -> torch.Tensor:
    """Map integer to [sign, d0, d1, ...] with d0 = ones place. sign: 0=+, 1=-."""
    sign = 1 if value < 0 else 0
    v = abs(int(value))
    digits = []
    for _ in range(n_digits):
        digits.append(v % 10)
        v //= 10
    return torch.tensor([sign, *digits], dtype=torch.long)


def digits_to_int(digits: torch.Tensor) -> int:
    """Inverse of int_to_digits for a 1D long tensor [sign, d0, ...]."""
    sign = int(digits[0].item())
    v = 0
    place = 1
    for d in digits[1:]:
        v += int(d.item()) * place
        place *= 10
    return -v if sign else v


class StatefulResidualRegister(nn.Module):
    """Digit register bank with residual write/read interfaces."""

    def __init__(self, cfg: SRRConfig):
        super().__init__()
        self.cfg = cfg
        # 2 sign + 10 digit embeddings sharing a table of size 10 for digits;
        # sign uses indices 0/1 in a separate 2-row table.
        self.sign_emb = nn.Embedding(2, cfg.d_model)
        self.digit_emb = nn.Embedding(10, cfg.d_model)
        self.n_fields = 1 + cfg.n_digits  # sign + digits
        self.write_proj = nn.Linear(cfg.d_model, self.n_fields * 10)
        # For sign field we only use first 2 logits of the 10-wide slice.
        self.read_scale = nn.Parameter(torch.ones(len(cfg.read_layers)) * 0.1)

    def encode_registers(self, digit_ids: torch.Tensor) -> torch.Tensor:
        """digit_ids: [B, 1+n_digits] -> [B, d_model] summed embeddings."""
        sign = self.sign_emb(digit_ids[:, 0].clamp(0, 1))
        digs = self.digit_emb(digit_ids[:, 1:].clamp(0, 9)).sum(dim=1)
        return sign + digs

    def write_logits(self, residual_last: torch.Tensor) -> torch.Tensor:
        """residual_last [B, d] -> logits [B, n_fields, 10]."""
        B = residual_last.shape[0]
        return self.write_proj(residual_last).view(B, self.n_fields, 10)

    def predicted_digits(self, residual_last: torch.Tensor) -> torch.Tensor:
        logits = self.write_logits(residual_last)
        # sign: argmax over first 2; digits: argmax over 10
        sign = logits[:, 0, :2].argmax(dim=-1)
        digs = logits[:, 1:, :].argmax(dim=-1)
        return torch.cat([sign.unsqueeze(-1), digs], dim=-1)

    def aux_loss(self, residual_last: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = self.write_logits(residual_last)
        # sign CE on 2-class
        loss_sign = F.cross_entropy(logits[:, 0, :2], target_ids[:, 0].clamp(0, 1))
        loss_dig = F.cross_entropy(
            logits[:, 1:, :].reshape(-1, 10),
            target_ids[:, 1:].reshape(-1).clamp(0, 9),
        )
        return loss_sign + loss_dig

    def inject(self, residual: torch.Tensor, register_vec: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Add register vector to last-token residual if layer is a read layer."""
        if layer_index not in self.cfg.read_layers:
            return residual
        idx = self.cfg.read_layers.index(layer_index)
        scale = self.read_scale[idx]
        out = residual.clone()
        out[:, -1, :] = out[:, -1, :] + scale * register_vec.to(dtype=out.dtype)
        return out


def run_with_srr(model, srr: StatefulResidualRegister, idx: torch.Tensor, targets=None):
    """Forward GPT with SRR write@L_w and read injections; returns logits, lm_loss, aux_loss, digits.

    model must be a GPT with n_loop=1. Does not mutate model weights.
    """
    cfg = model.cfg
    if cfg.n_loop != 1:
        raise ValueError("SRR requires n_loop=1")
    B, T = idx.shape
    x = model.tok(idx)
    cos = model.cos[:T].to(x.device)
    sin = model.sin[:T].to(x.device)
    register_vec = None
    aux = None
    pred_digits = None
    for layer_index, block in enumerate(model.blocks):
        x, _ = block(x, cos, sin)
        if layer_index == srr.cfg.write_layer:
            pred_digits = srr.predicted_digits(x[:, -1, :])
            if targets is not None:
                # targets here means register target ids, not LM targets
                pass
            register_vec = srr.encode_registers(pred_digits)
        if register_vec is not None:
            x = srr.inject(x, register_vec, layer_index)
    logits = model.head(model.norm(x))
    return logits, pred_digits


def run_with_srr_teacher(
    model,
    srr: StatefulResidualRegister,
    idx: torch.Tensor,
    register_targets: torch.Tensor,
    lm_targets: torch.Tensor | None = None,
):
    """Teacher-forced register write (targets) + optional LM loss."""
    cfg = model.cfg
    B, T = idx.shape
    x = model.tok(idx)
    cos = model.cos[:T].to(x.device)
    sin = model.sin[:T].to(x.device)
    aux = None
    register_vec = srr.encode_registers(register_targets)
    for layer_index, block in enumerate(model.blocks):
        x, _ = block(x, cos, sin)
        if layer_index == srr.cfg.write_layer:
            aux = srr.aux_loss(x[:, -1, :], register_targets)
            # Re-encode from predicted for inject consistency option:
            # use teacher embeddings for inject during training (stronger bus).
            register_vec = srr.encode_registers(register_targets)
        x = srr.inject(x, register_vec, layer_index)
    logits = model.head(model.norm(x))
    lm_loss = None
    if lm_targets is not None:
        from model import _supervised_lm_loss

        lm_loss = _supervised_lm_loss(logits, lm_targets, cfg.zloss)
    return logits, lm_loss, aux
