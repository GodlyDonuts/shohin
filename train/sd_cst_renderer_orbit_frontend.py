"""Renderer-orbit grounding front end for projected state transport.

The recurrent executor remains unchanged. This module adds a trainable byte
encoder whose residual is consumed by the frozen projected compiler, plus a
late-query bus that must first address the ordinal surface and then classifies
only its position-free byte content.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from sd_cst import LateQuery
from sd_cst_binding_bus import (
    BindingBusOutput,
    ProjectedHierarchicalBindingBusCompiler,
)
from sd_cst_byte_addressed import BYTE_PAD, BYTE_VOCAB


@dataclass(frozen=True, slots=True)
class OrbitLateQueryOutput:
    query: LateQuery
    pointer_logits: torch.Tensor


class RendererOrbitGroundedCompiler(ProjectedHierarchicalBindingBusCompiler):
    """Add renderer-orbit grounding without widening the proven executor.

    Program fields receive a residual from the orbit encoder. Late-query
    classification has a deliberately narrower information path: contextual
    memory can select a byte span, but the classifier receives only a weighted
    sum of position-free raw-byte embeddings. This prevents a query-template
    representation from directly selecting one of the three positions.
    """

    def __init__(
        self,
        *,
        orbit_width: int = 512,
        orbit_heads: int = 8,
        orbit_layers: int = 8,
        orbit_ff: int = 2048,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        if orbit_width <= 0 or orbit_heads <= 0 or orbit_width % orbit_heads:
            raise ValueError("orbit width must be positive and divisible by heads")
        if orbit_layers <= 0 or orbit_ff <= 0:
            raise ValueError("orbit depth and feed-forward width must be positive")
        self.orbit_width = int(orbit_width)
        self.orbit_byte_embedding = nn.Embedding(
            BYTE_VOCAB,
            orbit_width,
            padding_idx=BYTE_PAD,
        )
        self.orbit_position_embedding = nn.Embedding(self.max_bytes, orbit_width)
        layer = nn.TransformerEncoderLayer(
            d_model=orbit_width,
            nhead=orbit_heads,
            dim_feedforward=orbit_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.orbit_encoder = nn.TransformerEncoder(
            layer,
            num_layers=orbit_layers,
            enable_nested_tensor=False,
        )
        self.orbit_norm = nn.LayerNorm(orbit_width)
        self.orbit_to_parent = nn.Linear(orbit_width, self.width, bias=False)
        self.orbit_residual_scale = nn.Parameter(torch.tensor(0.0))

        self.ordinal_query = nn.Parameter(torch.empty(orbit_width))
        self.ordinal_query_projection = nn.Linear(
            orbit_width,
            orbit_width,
            bias=False,
        )
        self.ordinal_key_projection = nn.Linear(
            orbit_width,
            orbit_width,
            bias=False,
        )
        self.ordinal_value_projection = nn.Linear(
            orbit_width,
            orbit_width,
            bias=False,
        )
        self.ordinal_norm = nn.LayerNorm(orbit_width)
        self.ordinal_head = nn.Linear(orbit_width, 3)
        nn.init.normal_(self.ordinal_query, std=0.02)

    def _validate_orbit_input(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("byte ids must be a rank-2 torch.long tensor")
        if valid_mask.shape != ids.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid mask must be boolean and match byte ids")
        if ids.shape[1] < 1 or ids.shape[1] > self.max_bytes:
            raise ValueError("byte source length is outside the compiler window")
        if not bool(valid_mask.any(-1).all()):
            raise ValueError("every byte source must be nonempty")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= BYTE_VOCAB):
            raise ValueError("byte ids are outside the byte vocabulary")

    def _orbit_encode(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_orbit_input(ids, valid_mask)
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.orbit_byte_embedding(ids)
        hidden = hidden + self.orbit_position_embedding(positions)[None]
        hidden = self.orbit_encoder(hidden, src_key_padding_mask=~valid_mask)
        return self.orbit_norm(hidden)

    def _encode(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        combined, _ = self._encode_components(ids, valid_mask)
        return combined

    def _encode_components(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent = super()._encode(ids, valid_mask)
        orbit = self._orbit_encode(ids, valid_mask)
        projected = self.orbit_to_parent(orbit)
        scale = self.orbit_residual_scale.tanh()
        return parent + scale * projected, orbit

    def compile_query_with_evidence(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> OrbitLateQueryOutput:
        memory = self._orbit_encode(ids, valid_mask)
        query = self.ordinal_query_projection(self.ordinal_query)
        keys = self.ordinal_key_projection(memory)
        logits = torch.einsum("w,blw->bl", query, keys) / math.sqrt(self.orbit_width)
        logits = logits.masked_fill(
            ~valid_mask,
            torch.finfo(logits.dtype).min,
        ).float()
        weights = logits.softmax(-1).to(memory.dtype)

        # Values are raw, position-free byte embeddings. Context and absolute
        # position may choose the span but cannot directly select the class.
        raw_values = self.ordinal_value_projection(self.orbit_byte_embedding(ids))
        selected = torch.einsum("bl,blw->bw", weights, raw_values)
        selected = self.ordinal_norm(selected)
        return OrbitLateQueryOutput(
            query=LateQuery(self.ordinal_head(selected).float()),
            pointer_logits=logits,
        )

    def compile_query(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> LateQuery:
        return self.compile_query_with_evidence(ids, valid_mask).query

    def compile_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
        return super().compile_program(ids, valid_mask)


def renderer_orbit_trainable_names(
    model: RendererOrbitGroundedCompiler,
) -> frozenset[str]:
    """Return the exact new-front-end parameter whitelist."""
    prefixes = (
        "orbit_",
        "ordinal_",
    )
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    )


def freeze_to_renderer_orbit_front_end(
    model: RendererOrbitGroundedCompiler,
    *,
    additional_trainable: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    declared = renderer_orbit_trainable_names(model) | additional_trainable
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("renderer-orbit trainable parameter contract mismatch")
    return tuple(sorted(actual))
