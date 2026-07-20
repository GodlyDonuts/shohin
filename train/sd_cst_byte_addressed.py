"""Byte-addressed evidence compiler for source-deleted categorical state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from sd_cst import (
    EVENT_STEPS,
    EVENT_KIND_COUNT,
    IDENTITY_COUNT,
    AMOUNT_COUNT,
    QUERY_COUNT,
    STATE_COUNT,
    DeletedProgramTape,
    LateQuery,
)


BYTE_VOCAB = 257
BYTE_PAD = 256
PROGRAM_SLOTS = 1 + EVENT_STEPS


@dataclass(frozen=True, slots=True)
class ByteProgramOutput:
    tape: DeletedProgramTape
    pointer_logits: torch.Tensor


class ByteAddressedCompiler(nn.Module):
    """Select source evidence before emitting the private categorical tape."""

    def __init__(
        self,
        *,
        width: int = 384,
        heads: int = 8,
        encoder_layers: int = 6,
        slot_layers: int = 2,
        ff: int = 1536,
        slot_ff: int = 1024,
        max_bytes: int = 640,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        if encoder_layers <= 0 or slot_layers <= 0 or max_bytes <= 0:
            raise ValueError("encoder depth, slot depth, and max bytes must be positive")
        self.width = int(width)
        self.max_bytes = int(max_bytes)
        self.byte_embedding = nn.Embedding(BYTE_VOCAB, width, padding_idx=BYTE_PAD)
        self.position_embedding = nn.Embedding(max_bytes, width)
        source_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            source_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        self.source_norm = nn.LayerNorm(width)
        self.program_queries = nn.Parameter(torch.empty(PROGRAM_SLOTS, width))
        self.query_query = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.program_queries, std=0.02)
        nn.init.normal_(self.query_query, std=0.02)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        slot_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=slot_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(
            slot_layer,
            num_layers=slot_layers,
            enable_nested_tensor=False,
        )
        self.slot_norm = nn.LayerNorm(width)
        self.initial_head = nn.Linear(width, STATE_COUNT)
        self.kind_head = nn.Linear(width, EVENT_KIND_COUNT)
        self.identity_head = nn.Linear(width, IDENTITY_COUNT)
        self.amount_head = nn.Linear(width, AMOUNT_COUNT)
        self.query_head = nn.Linear(width, QUERY_COUNT)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
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
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.byte_embedding(ids) + self.position_embedding(positions)[None]
        hidden = self.source_encoder(hidden, src_key_padding_mask=~valid_mask)
        return self.source_norm(hidden)

    def _address(
        self,
        memory: torch.Tensor,
        valid_mask: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_queries = self.query_projection(queries)
        keys = self.key_projection(memory)
        logits = torch.einsum(
            "sw,blw->bsl", projected_queries, keys,
        ) / math.sqrt(self.width)
        logits = logits.masked_fill(~valid_mask[:, None], torch.finfo(logits.dtype).min)
        weights = logits.float().softmax(-1).to(memory.dtype)
        values = self.value_projection(memory)
        slots = torch.einsum("bsl,blw->bsw", weights, values)
        slots = slots + queries[None]
        return slots, logits.float()

    def compile_program(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> ByteProgramOutput:
        memory = self._encode(ids, valid_mask)
        slots, pointer_logits = self._address(
            memory, valid_mask, self.program_queries,
        )
        slots = self.slot_norm(self.slot_encoder(slots))
        initial = slots[:, 0]
        events = slots[:, 1:]
        return ByteProgramOutput(
            tape=DeletedProgramTape(
                initial_state=self.initial_head(initial).float(),
                event_kind=self.kind_head(events).float(),
                event_identity=self.identity_head(events).float(),
                amount=self.amount_head(events).float(),
            ),
            pointer_logits=pointer_logits,
        )

    def compile_query(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> LateQuery:
        memory = self._encode(ids, valid_mask)
        slots, _ = self._address(memory, valid_mask, self.query_query)
        slot = self.slot_norm(slots[:, 0])
        return LateQuery(self.query_head(slot).float())
