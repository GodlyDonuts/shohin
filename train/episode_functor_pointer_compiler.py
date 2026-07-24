"""Pointer-copy front end and favorable direct-machine EFC baseline.

The generic scanner copies bounded opaque-key bytes without assigning roles.
The neural baseline receives raw bytes plus those role-free candidates and
emits an attached soft categorical machine.  It is the preregistered direct
hypernetwork control, not the claim-bearing witness-identification treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_machine import (
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorMachine,
)


MAX_SOURCE_BYTES = 16_384
MAX_KEY_OCCURRENCES = 128
MAX_UNIQUE_KEYS = MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS
BYTE_PAD_ID = 256
_OPAQUE_KEY = re.compile(
    rb"(?<![A-Za-z0-9])(?:h[0-9a-f]{16}|d[1-9][0-9]{0,19})(?![A-Za-z0-9])"
)


class PointerCompilerError(ValueError):
    """The raw-byte, pointer-copy, or direct-machine contract failed."""


@dataclass(frozen=True, slots=True)
class ScannedSource:
    payload: bytes
    spans: tuple[tuple[int, int], ...]
    occurrence_keys: tuple[bytes, ...]
    unique_keys: tuple[bytes, ...]
    occurrence_to_unique: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.payload or len(self.payload) > MAX_SOURCE_BYTES:
            raise PointerCompilerError("source byte length is outside support")
        if (
            not self.spans
            or len(self.spans) > MAX_KEY_OCCURRENCES
            or len(self.spans) != len(self.occurrence_keys)
            or len(self.spans) != len(self.occurrence_to_unique)
            or not self.unique_keys
            or len(self.unique_keys) > MAX_UNIQUE_KEYS
        ):
            raise PointerCompilerError("source key geometry differs")
        if any(
            len(key) != 8 or key == b"\0" * 8
            for key in (*self.occurrence_keys, *self.unique_keys)
        ):
            raise PointerCompilerError("source key bytes are invalid")
        if len(set(self.unique_keys)) != len(self.unique_keys):
            raise PointerCompilerError("unique key inventory is duplicated")
        for span, key, unique in zip(
            self.spans,
            self.occurrence_keys,
            self.occurrence_to_unique,
            strict=True,
        ):
            start, end = span
            if not 0 <= start < end <= len(self.payload):
                raise PointerCompilerError("source key span leaves payload")
            if unique not in range(len(self.unique_keys)):
                raise PointerCompilerError("source key index leaves inventory")
            if key != self.unique_keys[unique]:
                raise PointerCompilerError("occurrence key differs from inventory")


@dataclass(frozen=True, slots=True)
class PointerCompilerBatch:
    byte_ids: torch.Tensor
    byte_valid: torch.Tensor
    span_bounds: torch.Tensor
    occurrence_valid: torch.Tensor
    occurrence_to_unique: torch.Tensor
    unique_key_bytes: torch.Tensor
    unique_key_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.byte_ids.ndim != 2 or self.byte_ids.dtype != torch.long:
            raise PointerCompilerError("byte_ids must be rank-two long")
        batch, length = self.byte_ids.shape
        expected = {
            "byte_valid": ((batch, length), torch.bool),
            "span_bounds": (
                (batch, MAX_KEY_OCCURRENCES, 2),
                torch.int32,
            ),
            "occurrence_valid": (
                (batch, MAX_KEY_OCCURRENCES),
                torch.bool,
            ),
            "occurrence_to_unique": (
                (batch, MAX_KEY_OCCURRENCES),
                torch.long,
            ),
            "unique_key_bytes": (
                (batch, MAX_UNIQUE_KEYS, 8),
                torch.uint8,
            ),
            "unique_key_valid": (
                (batch, MAX_UNIQUE_KEYS),
                torch.bool,
            ),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise PointerCompilerError(f"{name} geometry differs")
        devices = {
            self.byte_ids.device,
            self.byte_valid.device,
            self.span_bounds.device,
            self.occurrence_valid.device,
            self.occurrence_to_unique.device,
            self.unique_key_bytes.device,
            self.unique_key_valid.device,
        }
        if len(devices) != 1:
            raise PointerCompilerError("pointer batch tensors must share a device")

    @property
    def batch_size(self) -> int:
        return int(self.byte_ids.shape[0])


@dataclass(frozen=True, slots=True)
class DirectCompilerOutput:
    machine: SoftFunctorMachine
    key_assignment_logits: torch.Tensor
    slot_states: torch.Tensor
    unique_key_states: torch.Tensor
    unique_key_bytes: torch.Tensor
    unique_key_valid: torch.Tensor


def _key_bytes(token: bytes) -> bytes:
    if token.startswith(b"h"):
        value = int(token[1:], 16)
    elif token.startswith(b"d"):
        value = int(token[1:], 10)
    else:
        raise PointerCompilerError("opaque key token has unknown codec")
    if value <= 0 or value >= 1 << 64:
        raise PointerCompilerError("opaque key token leaves uint64")
    return value.to_bytes(8, "little")


def scan_source(payload: bytes) -> ScannedSource:
    if not isinstance(payload, bytes):
        raise PointerCompilerError("source must be bytes")
    if not payload or len(payload) > MAX_SOURCE_BYTES:
        raise PointerCompilerError("source byte length is outside support")
    spans: list[tuple[int, int]] = []
    occurrence_keys: list[bytes] = []
    unique_keys: list[bytes] = []
    unique_index: dict[bytes, int] = {}
    occurrence_to_unique: list[int] = []
    for match in _OPAQUE_KEY.finditer(payload):
        key = _key_bytes(match.group())
        if key not in unique_index:
            if len(unique_keys) >= MAX_UNIQUE_KEYS:
                raise PointerCompilerError("source has too many unique keys")
            unique_index[key] = len(unique_keys)
            unique_keys.append(key)
        spans.append(match.span())
        occurrence_keys.append(key)
        occurrence_to_unique.append(unique_index[key])
        if len(spans) > MAX_KEY_OCCURRENCES:
            raise PointerCompilerError("source has too many key occurrences")
    return ScannedSource(
        payload=payload,
        spans=tuple(spans),
        occurrence_keys=tuple(occurrence_keys),
        unique_keys=tuple(unique_keys),
        occurrence_to_unique=tuple(occurrence_to_unique),
    )


def collate_sources(
    sources: Sequence[ScannedSource],
    *,
    device: torch.device | str = "cpu",
) -> PointerCompilerBatch:
    if not sources:
        raise PointerCompilerError("source batch is empty")
    length = max(len(source.payload) for source in sources)
    if length > MAX_SOURCE_BYTES:
        raise PointerCompilerError("source batch exceeds byte support")
    batch = len(sources)
    byte_ids = torch.full(
        (batch, length),
        BYTE_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    byte_valid = torch.zeros((batch, length), dtype=torch.bool, device=device)
    span_bounds = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES, 2),
        dtype=torch.int32,
        device=device,
    )
    occurrence_valid = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.bool,
        device=device,
    )
    occurrence_to_unique = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.long,
        device=device,
    )
    unique_key_bytes = torch.zeros(
        (batch, MAX_UNIQUE_KEYS, 8),
        dtype=torch.uint8,
        device=device,
    )
    unique_key_valid = torch.zeros(
        (batch, MAX_UNIQUE_KEYS),
        dtype=torch.bool,
        device=device,
    )
    for row, source in enumerate(sources):
        payload = torch.tensor(
            tuple(source.payload),
            dtype=torch.long,
            device=device,
        )
        byte_ids[row, : payload.numel()] = payload
        byte_valid[row, : payload.numel()] = True
        for occurrence, ((start, end), unique) in enumerate(
            zip(source.spans, source.occurrence_to_unique, strict=True)
        ):
            span_bounds[row, occurrence] = torch.tensor(
                (start, end),
                dtype=torch.int32,
                device=device,
            )
            occurrence_valid[row, occurrence] = True
            occurrence_to_unique[row, occurrence] = unique
        for unique, key in enumerate(source.unique_keys):
            unique_key_bytes[row, unique] = torch.tensor(
                tuple(key),
                dtype=torch.uint8,
                device=device,
            )
            unique_key_valid[row, unique] = True
    return PointerCompilerBatch(
        byte_ids=byte_ids,
        byte_valid=byte_valid,
        span_bounds=span_bounds,
        occurrence_valid=occurrence_valid,
        occurrence_to_unique=occurrence_to_unique,
        unique_key_bytes=unique_key_bytes,
        unique_key_valid=unique_key_valid,
    )


class _GatedConvolutionBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.convolution = nn.Conv1d(
            width,
            2 * width,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
        )
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = self.norm(values).transpose(1, 2)
        hidden = F.glu(self.convolution(hidden), dim=1).transpose(1, 2)
        return residual + self.projection(hidden)


class DirectByteEFCCompiler(nn.Module):
    """Favorable direct-machine hypernetwork control over raw source bytes."""

    def __init__(
        self,
        *,
        width: int = 256,
        convolution_layers: int = 8,
        decoder_layers: int = 4,
        heads: int = 8,
        feedforward: int = 1_024,
    ) -> None:
        super().__init__()
        if (
            width < 32
            or width % 2
            or width % heads
            or convolution_layers < 1
            or decoder_layers < 1
            or feedforward < width
        ):
            raise PointerCompilerError("direct compiler geometry differs")
        self.width = int(width)
        self.byte_embedding = nn.Embedding(BYTE_PAD_ID + 1, width)
        self.byte_blocks = nn.ModuleList(
            _GatedConvolutionBlock(width, 2 ** (layer % 4))
            for layer in range(convolution_layers)
        )
        self.byte_norm = nn.LayerNorm(width)
        self.key_bit_projection = nn.Linear(64, width, bias=False)
        self.key_fusion = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.slot_count = MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS
        self.slot_queries = nn.Parameter(torch.empty(self.slot_count, width))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )
        self.slot_norm = nn.LayerNorm(width)
        self.assignment_query = nn.Linear(width, width, bias=False)
        self.assignment_key = nn.Linear(width, width, bias=False)
        self.active_head = nn.Linear(width, 2)
        self.transition_pair = nn.Linear(2 * width, width)
        self.transition_destination = nn.Linear(width, width, bias=False)
        self.observer_pair = nn.Linear(2 * width, width)
        self.observer_answer = nn.Linear(width, MAX_ANSWERS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, batch: PointerCompilerBatch) -> DirectCompilerOutput:
        if batch.byte_ids.numel() and (
            int(batch.byte_ids.min()) < 0 or int(batch.byte_ids.max()) > BYTE_PAD_ID
        ):
            raise PointerCompilerError("byte id leaves embedding domain")
        byte_states = self.byte_embedding(batch.byte_ids)
        positions = torch.arange(
            batch.byte_ids.shape[1],
            device=batch.byte_ids.device,
            dtype=byte_states.dtype,
        )
        frequencies = torch.exp(
            torch.arange(
                0,
                self.width,
                2,
                device=batch.byte_ids.device,
                dtype=byte_states.dtype,
            )
            * (-math.log(10_000.0) / self.width)
        )
        angles = positions[:, None] * frequencies[None]
        position_encoding = torch.zeros_like(byte_states[0])
        position_encoding[:, 0::2] = angles.sin()
        position_encoding[:, 1::2] = angles.cos()
        byte_states = byte_states + position_encoding[None]
        byte_states = byte_states * batch.byte_valid[..., None]
        for block in self.byte_blocks:
            byte_states = block(byte_states)
            byte_states = byte_states * batch.byte_valid[..., None]
        byte_states = self.byte_norm(byte_states)

        occurrence_states = torch.zeros(
            (
                batch.batch_size,
                MAX_KEY_OCCURRENCES,
                self.width,
            ),
            dtype=byte_states.dtype,
            device=byte_states.device,
        )
        for row in range(batch.batch_size):
            count = int(batch.occurrence_valid[row].sum())
            for occurrence in range(count):
                start, end = (
                    int(value)
                    for value in batch.span_bounds[row, occurrence].tolist()
                )
                occurrence_states[row, occurrence] = byte_states[
                    row,
                    start:end,
                ].mean(0)

        unique_states = torch.zeros(
            (batch.batch_size, MAX_UNIQUE_KEYS, self.width),
            dtype=byte_states.dtype,
            device=byte_states.device,
        )
        unique_counts = torch.zeros(
            (batch.batch_size, MAX_UNIQUE_KEYS, 1),
            dtype=byte_states.dtype,
            device=byte_states.device,
        )
        for row in range(batch.batch_size):
            count = int(batch.occurrence_valid[row].sum())
            index = batch.occurrence_to_unique[row, :count]
            unique_states[row].index_add_(
                0,
                index,
                occurrence_states[row, :count],
            )
            unique_counts[row].index_add_(
                0,
                index,
                torch.ones(
                    (count, 1),
                    dtype=byte_states.dtype,
                    device=byte_states.device,
                ),
            )
        unique_states = unique_states / unique_counts.clamp_min(1.0)
        key_bits = (
            (
                batch.unique_key_bytes[..., None]
                >> torch.arange(8, device=byte_states.device)
            )
            & 1
        ).reshape(batch.batch_size, MAX_UNIQUE_KEYS, 64)
        key_bits = key_bits.to(byte_states.dtype)
        unique_states = self.key_fusion(
            torch.cat(
                (unique_states, self.key_bit_projection(key_bits)),
                dim=-1,
            )
        )
        unique_states = unique_states * batch.unique_key_valid[..., None]

        slot_queries = self.slot_queries[None].expand(batch.batch_size, -1, -1)
        slot_states = self.slot_decoder(
            slot_queries,
            byte_states,
            memory_key_padding_mask=~batch.byte_valid,
        )
        slot_states = self.slot_norm(slot_states)
        assignment_logits = torch.einsum(
            "bsd,bud->bsu",
            self.assignment_query(slot_states),
            self.assignment_key(unique_states),
        ) / math.sqrt(self.width)
        assignment_logits = assignment_logits.masked_fill(
            ~batch.unique_key_valid[:, None],
            torch.finfo(assignment_logits.dtype).min,
        )

        state_slots = slot_states[:, :MAX_STATES]
        action_slots = slot_states[
            :,
            MAX_STATES : MAX_STATES + MAX_ACTIONS,
        ]
        observer_slots = slot_states[:, -MAX_OBSERVERS:]
        state_active = self.active_head(state_slots)
        action_active = self.active_head(action_slots)
        observer_active = self.active_head(observer_slots)
        action_pair = torch.cat(
            (
                action_slots[:, :, None].expand(-1, -1, MAX_STATES, -1),
                state_slots[:, None].expand(-1, MAX_ACTIONS, -1, -1),
            ),
            dim=-1,
        )
        transition_query = self.transition_pair(action_pair)
        transition_key = self.transition_destination(state_slots)
        action_next = torch.einsum(
            "bmsd,bkd->bmsk",
            transition_query,
            transition_key,
        ) / math.sqrt(self.width)
        observer_pair = torch.cat(
            (
                observer_slots[:, :, None].expand(-1, -1, MAX_STATES, -1),
                state_slots[:, None].expand(-1, MAX_OBSERVERS, -1, -1),
            ),
            dim=-1,
        )
        observer_answer = self.observer_answer(
            F.gelu(self.observer_pair(observer_pair))
        )
        machine = SoftFunctorMachine(
            state_active=state_active,
            action_active=action_active,
            observer_active=observer_active,
            action_next=action_next,
            observer_answer=observer_answer,
        )
        return DirectCompilerOutput(
            machine=machine,
            key_assignment_logits=assignment_logits,
            slot_states=slot_states,
            unique_key_states=unique_states,
            unique_key_bytes=batch.unique_key_bytes,
            unique_key_valid=batch.unique_key_valid,
        )


__all__ = [
    "BYTE_PAD_ID",
    "DirectByteEFCCompiler",
    "DirectCompilerOutput",
    "MAX_KEY_OCCURRENCES",
    "MAX_SOURCE_BYTES",
    "MAX_UNIQUE_KEYS",
    "PointerCompilerBatch",
    "PointerCompilerError",
    "ScannedSource",
    "collate_sources",
    "scan_source",
]
