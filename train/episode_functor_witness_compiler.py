"""Proof-carrying witness compiler for the primary EFC board.

This treatment receives only raw source bytes and role-free copied key spans.
It predicts record and key-occurrence roles, assigns the thirteen opaque keys
to anonymous semantic slots, and uses that *same* assignment to assemble
transition/observer evidence.  A zero-parameter constrained-transport layer
then emits the only lawful K=8/M=3/P=2/Y=4 machine.

There is no deterministic source grammar parser in the neural path.  Generic
newline/semicolon chunks are candidate records only; their semantic type,
field roles, and observer answer are model-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_constrained_transport import (
    LawfulMachineProjector,
    LawfulProjection,
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
    project_key_assignment_logits,
)
from episode_functor_machine import (
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from episode_functor_pointer_compiler import (
    BYTE_PAD_ID,
    MAX_KEY_OCCURRENCES,
    MAX_UNIQUE_KEYS,
    PointerCompilerBatch,
    ScannedSource,
    collate_sources,
    scan_source,
)


MAX_RECORDS = 64
RECORD_TYPES = 5
RECORD_STATE = 0
RECORD_TRANSITION = 1
RECORD_OBSERVATION = 2
RECORD_LAW = 3
RECORD_OTHER = 4
OCCURRENCE_ROLES = 7
ROLE_STATE_DECLARATION = 0
ROLE_ACTION = 1
ROLE_TRANSITION_SOURCE = 2
ROLE_TRANSITION_DESTINATION = 3
ROLE_OBSERVER = 4
ROLE_OBSERVATION_STATE = 5
ROLE_IGNORE = 6


class WitnessCompilerError(ValueError):
    """A role-free record scan or witness compiler contract failed."""


@dataclass(frozen=True, slots=True)
class WitnessScannedSource:
    pointer: ScannedSource
    record_spans: tuple[tuple[int, int], ...]
    occurrence_to_record: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.record_spans
            or len(self.record_spans) > MAX_RECORDS
            or len(self.occurrence_to_record) != len(self.pointer.spans)
        ):
            raise WitnessCompilerError("witness record geometry differs")
        for start, end in self.record_spans:
            if not 0 <= start < end <= len(self.pointer.payload):
                raise WitnessCompilerError(
                    "witness record span leaves source"
                )
        for occurrence, record in enumerate(self.occurrence_to_record):
            if record not in range(len(self.record_spans)):
                raise WitnessCompilerError(
                    "key occurrence leaves record inventory"
                )
            key_start, key_end = self.pointer.spans[occurrence]
            record_start, record_end = self.record_spans[record]
            if not (
                record_start <= key_start < key_end <= record_end
            ):
                raise WitnessCompilerError(
                    "key occurrence is not contained by its record"
                )


@dataclass(frozen=True, slots=True)
class WitnessCompilerBatch:
    pointer: PointerCompilerBatch
    record_bounds: torch.Tensor
    record_valid: torch.Tensor
    occurrence_to_record: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.pointer.batch_size
        if (
            self.record_bounds.shape != (batch, MAX_RECORDS, 2)
            or self.record_bounds.dtype != torch.int32
            or self.record_valid.shape != (batch, MAX_RECORDS)
            or self.record_valid.dtype != torch.bool
            or self.occurrence_to_record.shape
            != (batch, MAX_KEY_OCCURRENCES)
            or self.occurrence_to_record.dtype != torch.long
        ):
            raise WitnessCompilerError("witness compiler batch differs")
        if len(
            {
                self.pointer.byte_ids.device,
                self.record_bounds.device,
                self.record_valid.device,
                self.occurrence_to_record.device,
            }
        ) != 1:
            raise WitnessCompilerError(
                "witness compiler tensors must share one device"
            )

    @property
    def batch_size(self) -> int:
        return self.pointer.batch_size


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    transition_logits: torch.Tensor
    observer_logits: torch.Tensor
    record_role_unique: torch.Tensor
    record_role_slot: torch.Tensor


@dataclass(frozen=True, slots=True)
class WitnessCompilerOutput:
    projection: LawfulProjection
    relation_evidence: RelationEvidence
    key_assignment_logits: torch.Tensor
    raw_key_assignment_logits: torch.Tensor
    record_type_logits: torch.Tensor
    occurrence_role_logits: torch.Tensor
    answer_logits: torch.Tensor
    unique_key_bytes: torch.Tensor
    unique_key_valid: torch.Tensor


def _record_spans(payload: bytes) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, value in enumerate(payload):
        if value not in (ord("\n"), ord(";")):
            continue
        left = start
        right = index
        while left < right and payload[left] in b" \t":
            left += 1
        while right > left and payload[right - 1] in b" \t":
            right -= 1
        if left < right:
            spans.append((left, right))
        start = index + 1
    left = start
    right = len(payload)
    while left < right and payload[left] in b" \t":
        left += 1
    while right > left and payload[right - 1] in b" \t":
        right -= 1
    if left < right:
        spans.append((left, right))
    if not spans or len(spans) > MAX_RECORDS:
        raise WitnessCompilerError("generic record candidate count differs")
    return tuple(spans)


def scan_witness_source(payload: bytes) -> WitnessScannedSource:
    pointer = scan_source(payload)
    records = _record_spans(payload)
    occurrence_to_record: list[int] = []
    for key_start, key_end in pointer.spans:
        matches = tuple(
            index
            for index, (record_start, record_end) in enumerate(records)
            if record_start <= key_start < key_end <= record_end
        )
        if len(matches) != 1:
            raise WitnessCompilerError(
                "key occurrence does not have exactly one generic record"
            )
        occurrence_to_record.append(matches[0])
    return WitnessScannedSource(
        pointer=pointer,
        record_spans=records,
        occurrence_to_record=tuple(occurrence_to_record),
    )


def collate_witness_sources(
    sources: Sequence[WitnessScannedSource],
    *,
    device: torch.device | str = "cpu",
) -> WitnessCompilerBatch:
    if not sources:
        raise WitnessCompilerError("witness source batch is empty")
    pointer = collate_sources(
        tuple(source.pointer for source in sources),
        device=device,
    )
    batch = len(sources)
    record_bounds = torch.zeros(
        (batch, MAX_RECORDS, 2),
        dtype=torch.int32,
        device=device,
    )
    record_valid = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.bool,
        device=device,
    )
    occurrence_to_record = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.long,
        device=device,
    )
    for row, source in enumerate(sources):
        for record, bounds in enumerate(source.record_spans):
            record_bounds[row, record] = torch.tensor(
                bounds,
                dtype=torch.int32,
                device=device,
            )
            record_valid[row, record] = True
        count = len(source.occurrence_to_record)
        occurrence_to_record[row, :count] = torch.tensor(
            source.occurrence_to_record,
            dtype=torch.long,
            device=device,
        )
    return WitnessCompilerBatch(
        pointer=pointer,
        record_bounds=record_bounds,
        record_valid=record_valid,
        occurrence_to_record=occurrence_to_record,
    )


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise WitnessCompilerError("witness masked softmax geometry differs")
    if not bool(mask.any(-1).all()):
        raise WitnessCompilerError("witness masked softmax has empty support")
    negative = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, negative).float().softmax(-1)


def _normalize(values: torch.Tensor) -> torch.Tensor:
    denominator = values.sum(-1, keepdim=True)
    return values / denominator.clamp_min(
        torch.finfo(values.dtype).tiny
    )


def assemble_relation_evidence(
    *,
    record_type_logits: torch.Tensor,
    occurrence_role_logits: torch.Tensor,
    answer_logits: torch.Tensor,
    occurrence_valid: torch.Tensor,
    occurrence_to_record: torch.Tensor,
    occurrence_to_unique: torch.Tensor,
    source_unique_key_valid: torch.Tensor,
    key_assignment_logits: torch.Tensor,
) -> RelationEvidence:
    """Use one key-role transport to index both copied keys and table axes."""

    if record_type_logits.ndim != 3:
        raise WitnessCompilerError(
            "record type logits must be rank three"
        )
    batch, records, record_types = record_type_logits.shape
    occurrences = int(occurrence_role_logits.shape[1])
    if (
        records != MAX_RECORDS
        or record_types != RECORD_TYPES
        or occurrence_role_logits.shape
        != (batch, MAX_KEY_OCCURRENCES, OCCURRENCE_ROLES)
        or answer_logits.shape
        != (batch, MAX_RECORDS, PRIMARY_ANSWERS)
        or occurrence_valid.shape
        != (batch, MAX_KEY_OCCURRENCES)
        or occurrence_valid.dtype != torch.bool
        or occurrence_to_record.shape
        != (batch, MAX_KEY_OCCURRENCES)
        or occurrence_to_record.dtype != torch.long
        or occurrence_to_unique.shape
        != (batch, MAX_KEY_OCCURRENCES)
        or occurrence_to_unique.dtype != torch.long
        or source_unique_key_valid.shape != (batch, MAX_UNIQUE_KEYS)
        or source_unique_key_valid.dtype != torch.bool
        or key_assignment_logits.shape
        != (
            batch,
            MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS,
            MAX_UNIQUE_KEYS,
        )
        or occurrences != MAX_KEY_OCCURRENCES
    ):
        raise WitnessCompilerError("relation evidence geometry differs")
    if bool(
        occurrence_to_record[occurrence_valid].ge(MAX_RECORDS).any()
    ) or bool(
        occurrence_to_unique[occurrence_valid].ge(MAX_UNIQUE_KEYS).any()
    ):
        raise WitnessCompilerError(
            "witness occurrence index leaves support"
        )

    record_map = F.one_hot(
        occurrence_to_record.clamp(0, MAX_RECORDS - 1),
        MAX_RECORDS,
    ).to(record_type_logits.dtype)
    unique_map = F.one_hot(
        occurrence_to_unique.clamp(0, MAX_UNIQUE_KEYS - 1),
        MAX_UNIQUE_KEYS,
    ).to(record_type_logits.dtype)
    valid = occurrence_valid.to(record_type_logits.dtype)
    record_map = record_map * valid[..., None]
    unique_map = unique_map * valid[..., None]
    occurrence_roles = occurrence_role_logits.float().softmax(-1)
    role_unique = torch.einsum(
        "bor,bou,bol->brlu",
        record_map,
        unique_map,
        occurrence_roles,
    )
    role_unique = _normalize(role_unique)
    assignment = _masked_softmax(
        key_assignment_logits,
        source_unique_key_valid[:, None].expand_as(key_assignment_logits),
    )
    role_slot = torch.einsum(
        "brlu,bsu->brls",
        role_unique,
        assignment,
    )

    action_slot = _normalize(
        role_slot[
            :,
            :,
            ROLE_ACTION,
            MAX_STATES : MAX_STATES + PRIMARY_ACTIONS,
        ]
    )
    transition_source_slot = _normalize(
        role_slot[
            :,
            :,
            ROLE_TRANSITION_SOURCE,
            :PRIMARY_STATES,
        ]
    )
    transition_destination_slot = _normalize(
        role_slot[
            :,
            :,
            ROLE_TRANSITION_DESTINATION,
            :PRIMARY_STATES,
        ]
    )
    observer_slot = _normalize(
        role_slot[
            :,
            :,
            ROLE_OBSERVER,
            MAX_STATES
            + MAX_ACTIONS : MAX_STATES
            + MAX_ACTIONS
            + PRIMARY_OBSERVERS,
        ]
    )
    observation_state_slot = _normalize(
        role_slot[
            :,
            :,
            ROLE_OBSERVATION_STATE,
            :PRIMARY_STATES,
        ]
    )
    record_type = record_type_logits.float().softmax(-1)
    answer = answer_logits.float().softmax(-1)
    transition_evidence = torch.einsum(
        "br,bra,brs,brd->basd",
        record_type[:, :, RECORD_TRANSITION],
        action_slot,
        transition_source_slot,
        transition_destination_slot,
    )
    observer_evidence = torch.einsum(
        "br,bro,brs,bry->bosy",
        record_type[:, :, RECORD_OBSERVATION],
        observer_slot,
        observation_state_slot,
        answer,
    )
    tiny = torch.finfo(transition_evidence.dtype).tiny
    return RelationEvidence(
        transition_logits=transition_evidence.clamp_min(tiny).log(),
        observer_logits=observer_evidence.clamp_min(tiny).log(),
        record_role_unique=role_unique,
        record_role_slot=role_slot,
    )


class ProofCarryingWitnessCompiler(nn.Module):
    """Neural witness parser whose outputs are law-projected into a machine."""

    def __init__(
        self,
        *,
        width: int = 192,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        heads: int = 6,
        feedforward: int = 768,
        sinkhorn_iterations: int = 64,
        external_feature_width: int = 0,
        projector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if (
            width < 48
            or width % 2
            or width % heads
            or encoder_layers < 1
            or decoder_layers < 1
            or feedforward < width
            or sinkhorn_iterations < 8
            or external_feature_width < 0
        ):
            raise WitnessCompilerError("witness compiler geometry differs")
        self.width = int(width)
        self.external_feature_width = int(external_feature_width)
        self.key_sinkhorn_iterations = int(sinkhorn_iterations)
        self.byte_embedding = nn.Embedding(BYTE_PAD_ID + 1, width)
        self.external_projection = (
            nn.Sequential(
                nn.LayerNorm(external_feature_width),
                nn.Linear(external_feature_width, width),
                nn.GELU(),
                nn.Linear(width, width, bias=False),
            )
            if external_feature_width
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
        )
        self.encoder_norm = nn.LayerNorm(width)
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
        self.record_type_head = nn.Linear(width, RECORD_TYPES)
        self.occurrence_role_head = nn.Linear(width, OCCURRENCE_ROLES)
        self.answer_head = nn.Linear(width, PRIMARY_ANSWERS)
        self.projector = (
            LawfulMachineProjector(
                sinkhorn_iterations=sinkhorn_iterations
            )
            if projector is None
            else projector
        )
        if not all(
            callable(getattr(self.projector, name, None))
            for name in ("forward", "hard_project")
        ):
            raise WitnessCompilerError(
                "witness projector API differs"
            )
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        batch: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        frozen_byte_features: torch.Tensor | None = None,
    ) -> WitnessCompilerOutput:
        pointer = batch.pointer
        if pointer.byte_ids.numel() and (
            int(pointer.byte_ids.min()) < 0
            or int(pointer.byte_ids.max()) > BYTE_PAD_ID
        ):
            raise WitnessCompilerError(
                "witness byte id leaves embedding domain"
            )
        states = self.byte_embedding(pointer.byte_ids)
        if self.external_projection is None:
            if frozen_byte_features is not None:
                raise WitnessCompilerError(
                    "standalone witness compiler received frozen features"
                )
        else:
            if (
                frozen_byte_features is None
                or frozen_byte_features.shape
                != (
                    batch.batch_size,
                    pointer.byte_ids.shape[1],
                    self.external_feature_width,
                )
                or not frozen_byte_features.is_floating_point()
                or frozen_byte_features.device != states.device
                or not bool(torch.isfinite(frozen_byte_features).all())
            ):
                raise WitnessCompilerError(
                    "frozen source feature geometry differs"
                )
            states = states + self.external_projection(
                frozen_byte_features
            )
        positions = torch.arange(
            pointer.byte_ids.shape[1],
            device=states.device,
            dtype=states.dtype,
        )
        frequencies = torch.exp(
            torch.arange(
                0,
                self.width,
                2,
                device=states.device,
                dtype=states.dtype,
            )
            * (-math.log(10_000.0) / self.width)
        )
        angles = positions[:, None] * frequencies[None]
        position_encoding = torch.zeros_like(states[0])
        position_encoding[:, 0::2] = angles.sin()
        position_encoding[:, 1::2] = angles.cos()
        states = states + position_encoding[None]
        states = self.encoder(
            states,
            src_key_padding_mask=~pointer.byte_valid,
        )
        states = self.encoder_norm(states)

        record_states = torch.zeros(
            (batch.batch_size, MAX_RECORDS, self.width),
            dtype=states.dtype,
            device=states.device,
        )
        occurrence_states = torch.zeros(
            (
                batch.batch_size,
                MAX_KEY_OCCURRENCES,
                self.width,
            ),
            dtype=states.dtype,
            device=states.device,
        )
        for row in range(batch.batch_size):
            record_count = int(batch.record_valid[row].sum())
            for record in range(record_count):
                start, end = (
                    int(value)
                    for value in batch.record_bounds[row, record].tolist()
                )
                record_states[row, record] = states[row, start:end].mean(0)
            occurrence_count = int(pointer.occurrence_valid[row].sum())
            for occurrence in range(occurrence_count):
                start, end = (
                    int(value)
                    for value in pointer.span_bounds[row, occurrence].tolist()
                )
                occurrence_states[row, occurrence] = states[
                    row,
                    start:end,
                ].mean(0)

        unique_states = torch.zeros(
            (batch.batch_size, MAX_UNIQUE_KEYS, self.width),
            dtype=states.dtype,
            device=states.device,
        )
        unique_counts = torch.zeros(
            (batch.batch_size, MAX_UNIQUE_KEYS, 1),
            dtype=states.dtype,
            device=states.device,
        )
        for row in range(batch.batch_size):
            occurrence_count = int(pointer.occurrence_valid[row].sum())
            unique_index = pointer.occurrence_to_unique[
                row,
                :occurrence_count,
            ]
            unique_states[row].index_add_(
                0,
                unique_index,
                occurrence_states[row, :occurrence_count],
            )
            unique_counts[row].index_add_(
                0,
                unique_index,
                torch.ones(
                    (occurrence_count, 1),
                    dtype=states.dtype,
                    device=states.device,
                ),
            )
        unique_states = unique_states / unique_counts.clamp_min(1.0)
        key_bits = (
            (
                pointer.unique_key_bytes[..., None]
                >> torch.arange(8, device=states.device)
            )
            & 1
        ).reshape(batch.batch_size, MAX_UNIQUE_KEYS, 64)
        unique_states = self.key_fusion(
            torch.cat(
                (
                    unique_states,
                    self.key_bit_projection(key_bits.to(states.dtype)),
                ),
                dim=-1,
            )
        )
        unique_states = unique_states * pointer.unique_key_valid[..., None]

        slots = self.slot_decoder(
            self.slot_queries[None].expand(batch.batch_size, -1, -1),
            states,
            memory_key_padding_mask=~pointer.byte_valid,
        )
        slots = self.slot_norm(slots)
        raw_key_assignment_logits = torch.einsum(
            "bsd,bud->bsu",
            self.assignment_query(slots),
            self.assignment_key(unique_states),
        ) / math.sqrt(self.width)
        raw_key_assignment_logits = raw_key_assignment_logits.masked_fill(
            ~pointer.unique_key_valid[:, None],
            torch.finfo(raw_key_assignment_logits.dtype).min,
        )
        key_assignment_logits = project_key_assignment_logits(
            slot_assignment_logits=raw_key_assignment_logits,
            source_unique_key_valid=pointer.unique_key_valid,
            sinkhorn_iterations=self.key_sinkhorn_iterations,
            straight_through=straight_through,
        )
        record_type_logits = self.record_type_head(record_states)
        occurrence_role_logits = self.occurrence_role_head(
            occurrence_states
        )
        answer_logits = self.answer_head(record_states)
        relation_evidence = assemble_relation_evidence(
            record_type_logits=record_type_logits,
            occurrence_role_logits=occurrence_role_logits,
            answer_logits=answer_logits,
            occurrence_valid=pointer.occurrence_valid,
            occurrence_to_record=batch.occurrence_to_record,
            occurrence_to_unique=pointer.occurrence_to_unique,
            source_unique_key_valid=pointer.unique_key_valid,
            key_assignment_logits=key_assignment_logits,
        )
        projection = self.projector(
            relation_evidence.transition_logits,
            relation_evidence.observer_logits,
            straight_through=straight_through,
        )
        return WitnessCompilerOutput(
            projection=projection,
            relation_evidence=relation_evidence,
            key_assignment_logits=key_assignment_logits,
            raw_key_assignment_logits=raw_key_assignment_logits,
            record_type_logits=record_type_logits,
            occurrence_role_logits=occurrence_role_logits,
            answer_logits=answer_logits,
            unique_key_bytes=pointer.unique_key_bytes,
            unique_key_valid=pointer.unique_key_valid,
        )


__all__ = [
    "MAX_RECORDS",
    "OCCURRENCE_ROLES",
    "ProofCarryingWitnessCompiler",
    "RECORD_OBSERVATION",
    "RECORD_OTHER",
    "RECORD_STATE",
    "RECORD_TRANSITION",
    "RelationEvidence",
    "WitnessCompilerBatch",
    "WitnessCompilerError",
    "WitnessCompilerOutput",
    "WitnessScannedSource",
    "assemble_relation_evidence",
    "collate_witness_sources",
    "scan_witness_source",
]
