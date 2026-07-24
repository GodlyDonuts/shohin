"""Role-blind opaque-key binding for late EFC queries.

The scanner copies every bounded opaque uint64 token without assigning a
semantic role.  A neural parser must infer start, ordered action positions,
observer, and STOP from raw query bytes.  Exact byte equality then binds the
selected query occurrences to compiler-retained key slots.  No transition or
observer table is visible to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
import torch.nn as nn

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import (
    HardFunctorKeys,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
    MAX_STEPS,
    SoftFunctorQuery,
)
from episode_functor_pointer_compiler import (
    BYTE_PAD_ID,
    MAX_UNIQUE_KEYS,
)


MAX_QUERY_BYTES = 2_048
MAX_QUERY_KEY_OCCURRENCES = MAX_STEPS + 2
_OPAQUE_KEY = re.compile(
    rb"(?<![A-Za-z0-9])(?:h[0-9a-f]{16}|d[1-9][0-9]{0,19})(?![A-Za-z0-9])"
)


class QueryParserError(ValueError):
    """The role-blind query scanner or neural binding contract failed."""


@dataclass(frozen=True, slots=True)
class ScannedQuery:
    """Offline role-free scan result; spans carry no grammar labels."""

    payload: bytes
    spans: tuple[tuple[int, int], ...]
    occurrence_keys: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not self.payload or len(self.payload) > MAX_QUERY_BYTES:
            raise QueryParserError("query byte length is outside support")
        if not 2 <= len(self.spans) <= MAX_QUERY_KEY_OCCURRENCES:
            raise QueryParserError("query key geometry differs")
        if len(self.spans) != len(self.occurrence_keys):
            raise QueryParserError("query span and key counts differ")
        for (start, end), key in zip(
            self.spans,
            self.occurrence_keys,
            strict=True,
        ):
            if (
                not 0 <= start < end <= len(self.payload)
                or len(key) != 8
                or key == b"\0" * 8
            ):
                raise QueryParserError("query key occurrence is invalid")


@dataclass(frozen=True, slots=True)
class QueryPointerBatch:
    """Tensor-only, target-free neural query input."""

    byte_ids: torch.Tensor
    byte_valid: torch.Tensor
    span_bounds: torch.Tensor
    occurrence_valid: torch.Tensor
    occurrence_key_bytes: torch.Tensor

    def __post_init__(self) -> None:
        if self.byte_ids.ndim != 2 or self.byte_ids.dtype != torch.long:
            raise QueryParserError("query byte ids must be rank-two long")
        batch, length = self.byte_ids.shape
        expected = {
            "byte_valid": ((batch, length), torch.bool),
            "span_bounds": (
                (batch, MAX_QUERY_KEY_OCCURRENCES, 2),
                torch.int32,
            ),
            "occurrence_valid": (
                (batch, MAX_QUERY_KEY_OCCURRENCES),
                torch.bool,
            ),
            "occurrence_key_bytes": (
                (batch, MAX_QUERY_KEY_OCCURRENCES, 8),
                torch.uint8,
            ),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise QueryParserError(f"{name} geometry differs")
        if len(
            {
                self.byte_ids.device,
                self.byte_valid.device,
                self.span_bounds.device,
                self.occurrence_valid.device,
                self.occurrence_key_bytes.device,
            }
        ) != 1:
            raise QueryParserError("query pointer tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.byte_ids.shape[0])


@dataclass(frozen=True, slots=True)
class QueryParserOutput:
    query: SoftFunctorQuery
    role_occurrence_logits: torch.Tensor
    stop_position_logits: torch.Tensor
    exact_query_key_matches: torch.Tensor


@dataclass(frozen=True, slots=True)
class QueryRoleOutput:
    role_occurrence_logits: torch.Tensor
    stop_position_logits: torch.Tensor


def _key_bytes(token: bytes) -> bytes:
    if token.startswith(b"h"):
        value = int(token[1:], 16)
    elif token.startswith(b"d"):
        value = int(token[1:], 10)
    else:
        raise QueryParserError("opaque query key has unknown codec")
    if value <= 0 or value >= 1 << 64:
        raise QueryParserError("opaque query key leaves uint64")
    return value.to_bytes(8, "little")


def scan_query(payload: bytes) -> ScannedQuery:
    if not isinstance(payload, bytes):
        raise QueryParserError("query must be bytes")
    if not payload or len(payload) > MAX_QUERY_BYTES:
        raise QueryParserError("query byte length is outside support")
    spans: list[tuple[int, int]] = []
    keys: list[bytes] = []
    for match in _OPAQUE_KEY.finditer(payload):
        spans.append(match.span())
        keys.append(_key_bytes(match.group()))
        if len(spans) > MAX_QUERY_KEY_OCCURRENCES:
            raise QueryParserError("query has too many key occurrences")
    return ScannedQuery(
        payload=payload,
        spans=tuple(spans),
        occurrence_keys=tuple(keys),
    )


def collate_queries(
    queries: tuple[ScannedQuery, ...] | list[ScannedQuery],
    *,
    device: torch.device | str = "cpu",
) -> QueryPointerBatch:
    if not queries:
        raise QueryParserError("query batch is empty")
    length = max(len(query.payload) for query in queries)
    if length > MAX_QUERY_BYTES:
        raise QueryParserError("query batch exceeds byte support")
    batch = len(queries)
    byte_ids = torch.full(
        (batch, length),
        BYTE_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    byte_valid = torch.zeros((batch, length), dtype=torch.bool, device=device)
    span_bounds = torch.zeros(
        (batch, MAX_QUERY_KEY_OCCURRENCES, 2),
        dtype=torch.int32,
        device=device,
    )
    occurrence_valid = torch.zeros(
        (batch, MAX_QUERY_KEY_OCCURRENCES),
        dtype=torch.bool,
        device=device,
    )
    occurrence_key_bytes = torch.zeros(
        (batch, MAX_QUERY_KEY_OCCURRENCES, 8),
        dtype=torch.uint8,
        device=device,
    )
    for row, query in enumerate(queries):
        payload = torch.tensor(
            tuple(query.payload),
            dtype=torch.long,
            device=device,
        )
        byte_ids[row, : payload.numel()] = payload
        byte_valid[row, : payload.numel()] = True
        for occurrence, ((start, end), key) in enumerate(
            zip(query.spans, query.occurrence_keys, strict=True)
        ):
            span_bounds[row, occurrence] = torch.tensor(
                (start, end),
                dtype=torch.int32,
                device=device,
            )
            occurrence_valid[row, occurrence] = True
            occurrence_key_bytes[row, occurrence] = torch.tensor(
                tuple(key),
                dtype=torch.uint8,
                device=device,
            )
    return QueryPointerBatch(
        byte_ids=byte_ids,
        byte_valid=byte_valid,
        span_bounds=span_bounds,
        occurrence_valid=occurrence_valid,
        occurrence_key_bytes=occurrence_key_bytes,
    )


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise QueryParserError("masked softmax geometry differs")
    if not bool(mask.any(-1).all()):
        raise QueryParserError("masked softmax has an empty support")
    negative = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, negative).float().softmax(-1)


def _normalized_log(probabilities: torch.Tensor) -> torch.Tensor:
    denominator = probabilities.sum(-1, keepdim=True)
    if not bool(denominator.detach().gt(0).all()):
        raise QueryParserError("opaque key binding has an empty slot support")
    normalized = probabilities / denominator
    return normalized.clamp_min(torch.finfo(normalized.dtype).tiny).log()


def _bind_query_roles_attached_training(
    *,
    role_occurrence_logits: torch.Tensor,
    stop_position_logits: torch.Tensor,
    query_occurrence_key_bytes: torch.Tensor,
    query_occurrence_valid: torch.Tensor,
    source_unique_key_bytes: torch.Tensor,
    source_unique_key_valid: torch.Tensor,
    slot_assignment_logits: torch.Tensor,
) -> QueryParserOutput:
    """Train-only diagnostic binding before source deletion.

    This helper is deliberately private and is not reachable through
    ``NeuralOpaqueQueryParser.forward`` or ``LearnedEFCSystem``.
    """

    if role_occurrence_logits.ndim != 3:
        raise QueryParserError("role occurrence logits must be rank three")
    batch, roles, occurrences = role_occurrence_logits.shape
    steps = roles - 2
    if (
        not 1 <= steps <= MAX_STEPS
        or occurrences != MAX_QUERY_KEY_OCCURRENCES
        or stop_position_logits.shape != (batch, steps + 1)
        or query_occurrence_key_bytes.shape
        != (batch, MAX_QUERY_KEY_OCCURRENCES, 8)
        or query_occurrence_valid.shape
        != (batch, MAX_QUERY_KEY_OCCURRENCES)
        or query_occurrence_valid.dtype != torch.bool
        or source_unique_key_bytes.shape != (batch, MAX_UNIQUE_KEYS, 8)
        or source_unique_key_valid.shape != (batch, MAX_UNIQUE_KEYS)
        or source_unique_key_valid.dtype != torch.bool
        or slot_assignment_logits.shape
        != (
            batch,
            MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS,
            MAX_UNIQUE_KEYS,
        )
    ):
        raise QueryParserError("opaque key binding geometry differs")
    devices = {
        role_occurrence_logits.device,
        stop_position_logits.device,
        query_occurrence_key_bytes.device,
        query_occurrence_valid.device,
        source_unique_key_bytes.device,
        source_unique_key_valid.device,
        slot_assignment_logits.device,
    }
    if len(devices) != 1:
        raise QueryParserError("opaque key binding tensors must share one device")

    exact = query_occurrence_key_bytes[:, :, None].eq(
        source_unique_key_bytes[:, None]
    ).all(-1)
    exact = (
        exact
        & query_occurrence_valid[:, :, None]
        & source_unique_key_valid[:, None]
    )
    match_count = exact.sum(-1)
    if not bool(match_count[query_occurrence_valid].eq(1).all()):
        raise QueryParserError(
            "every query key occurrence must match exactly one retained source key"
        )

    role_occurrence = _masked_softmax(
        role_occurrence_logits,
        query_occurrence_valid[:, None].expand_as(role_occurrence_logits),
    )
    role_unique = torch.einsum(
        "bro,bou->bru",
        role_occurrence,
        exact.to(role_occurrence.dtype),
    )
    assignment = _masked_softmax(
        slot_assignment_logits,
        source_unique_key_valid[:, None].expand_as(slot_assignment_logits),
    )
    role_slot = torch.einsum("bru,bsu->brs", role_unique, assignment)
    active_state = _normalized_log(
        role_slot[:, 0, :PRIMARY_STATES]
    )
    observer_offset = MAX_STATES + MAX_ACTIONS
    active_observer = _normalized_log(
        role_slot[
            :,
            1,
            observer_offset : observer_offset + PRIMARY_OBSERVERS,
        ]
    )
    active_action = _normalized_log(
        role_slot[
            :,
            2:,
            MAX_STATES : MAX_STATES + PRIMARY_ACTIONS,
        ]
    )
    action = torch.full(
        (batch, steps, MAX_ACTIONS),
        -60.0,
        dtype=active_action.dtype,
        device=active_action.device,
    )
    action[:, :, :PRIMARY_ACTIONS] = active_action
    observer = torch.full(
        (batch, MAX_OBSERVERS),
        -60.0,
        dtype=active_observer.dtype,
        device=active_observer.device,
    )
    observer[:, :PRIMARY_OBSERVERS] = active_observer
    state = torch.full(
        (batch, MAX_STATES),
        -60.0,
        dtype=active_state.dtype,
        device=active_state.device,
    )
    state[:, :PRIMARY_STATES] = active_state
    query = SoftFunctorQuery(
        start_state=state,
        action_path=action,
        stop_position=stop_position_logits,
        observer=observer,
    )
    return QueryParserOutput(
        query=query,
        role_occurrence_logits=role_occurrence_logits,
        stop_position_logits=stop_position_logits,
        exact_query_key_matches=exact,
    )


def _masked_normalized_log(
    probabilities: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if (
        probabilities.shape != active.shape
        or active.dtype != torch.bool
        or probabilities.device != active.device
    ):
        raise QueryParserError("sealed query active-mask geometry differs")
    masked = probabilities * active.to(probabilities.dtype)
    denominator = masked.sum(-1, keepdim=True)
    if not bool(denominator.detach().gt(0).all()):
        raise QueryParserError("sealed query role has no active support")
    normalized = masked / denominator
    log_values = normalized.clamp_min(
        torch.finfo(normalized.dtype).tiny
    ).log()
    return torch.where(
        active,
        log_values,
        torch.full_like(log_values, -60.0),
    )


def bind_query_roles_to_hard_keys(
    *,
    role_occurrence_logits: torch.Tensor,
    stop_position_logits: torch.Tensor,
    query_occurrence_key_bytes: torch.Tensor,
    query_occurrence_valid: torch.Tensor,
    sealed_keys: HardFunctorKeys,
) -> QueryParserOutput:
    """Resolve a raw late query using only the already sealed key table."""

    if role_occurrence_logits.ndim != 3:
        raise QueryParserError("role occurrence logits must be rank three")
    batch, roles, occurrences = role_occurrence_logits.shape
    steps = roles - 2
    if (
        not 1 <= steps <= MAX_STEPS
        or occurrences != MAX_QUERY_KEY_OCCURRENCES
        or stop_position_logits.shape != (batch, steps + 1)
        or query_occurrence_key_bytes.shape
        != (batch, MAX_QUERY_KEY_OCCURRENCES, 8)
        or query_occurrence_key_bytes.dtype != torch.uint8
        or query_occurrence_valid.shape
        != (batch, MAX_QUERY_KEY_OCCURRENCES)
        or query_occurrence_valid.dtype != torch.bool
        or sealed_keys.batch_size != batch
    ):
        raise QueryParserError("sealed opaque-key binding geometry differs")
    devices = {
        role_occurrence_logits.device,
        stop_position_logits.device,
        query_occurrence_key_bytes.device,
        query_occurrence_valid.device,
        sealed_keys.state_keys.device,
    }
    if len(devices) != 1:
        raise QueryParserError(
            "sealed opaque-key binding tensors must share one device"
        )
    slot_keys = torch.cat(
        (
            sealed_keys.state_keys,
            sealed_keys.action_keys,
            sealed_keys.observer_keys,
        ),
        dim=1,
    )
    state_active = torch.zeros(
        (batch, MAX_STATES),
        dtype=torch.bool,
        device=slot_keys.device,
    )
    action_active = torch.zeros(
        (batch, MAX_ACTIONS),
        dtype=torch.bool,
        device=slot_keys.device,
    )
    observer_active = torch.zeros(
        (batch, MAX_OBSERVERS),
        dtype=torch.bool,
        device=slot_keys.device,
    )
    state_active[:, :PRIMARY_STATES] = True
    action_active[:, :PRIMARY_ACTIONS] = True
    observer_active[:, :PRIMARY_OBSERVERS] = True
    slot_active = torch.cat(
        (state_active, action_active, observer_active),
        dim=1,
    )
    for row in range(batch):
        active_keys = slot_keys[row, slot_active[row]]
        if (
            bool(active_keys.eq(0).all(-1).any())
            or len({bytes(value.tolist()) for value in active_keys})
            != int(active_keys.shape[0])
            or bool(slot_keys[row, ~slot_active[row]].ne(0).any())
        ):
            raise QueryParserError(
                "sealed primary key inventory is zero, duplicated, or unpadded"
            )
    exact = query_occurrence_key_bytes[:, :, None].eq(
        slot_keys[:, None]
    ).all(-1)
    exact = (
        exact
        & query_occurrence_valid[:, :, None]
        & slot_active[:, None]
    )
    match_count = exact.sum(-1)
    if not bool(match_count[query_occurrence_valid].eq(1).all()):
        raise QueryParserError(
            "every late-query key must match exactly one sealed active key"
        )
    role_occurrence = _masked_softmax(
        role_occurrence_logits,
        query_occurrence_valid[:, None].expand_as(
            role_occurrence_logits
        ),
    )
    role_slot = torch.einsum(
        "bro,bos->brs",
        role_occurrence,
        exact.to(role_occurrence.dtype),
    )
    state = _masked_normalized_log(
        role_slot[:, 0, :MAX_STATES],
        state_active,
    )
    action_offset = MAX_STATES
    observer_offset = MAX_STATES + MAX_ACTIONS
    action = _masked_normalized_log(
        role_slot[:, 2:, action_offset:observer_offset],
        action_active[:, None].expand(
            -1,
            steps,
            -1,
        ),
    )
    observer = _masked_normalized_log(
        role_slot[:, 1, observer_offset:],
        observer_active,
    )
    return QueryParserOutput(
        query=SoftFunctorQuery(
            start_state=state,
            action_path=action,
            stop_position=stop_position_logits,
            observer=observer,
        ),
        role_occurrence_logits=role_occurrence_logits,
        stop_position_logits=stop_position_logits,
        exact_query_key_matches=exact,
    )


class NeuralOpaqueQueryParser(nn.Module):
    """Raw-byte grammar parser with exact role-free opaque-key binding."""

    def __init__(
        self,
        *,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward: int = 512,
        max_steps: int = MAX_STEPS,
        external_feature_width: int = 0,
    ) -> None:
        super().__init__()
        if (
            width < 32
            or width % 2
            or width % heads
            or layers < 1
            or feedforward < width
            or not 1 <= max_steps <= MAX_STEPS
            or external_feature_width < 0
        ):
            raise QueryParserError("neural query parser geometry differs")
        self.width = int(width)
        self.max_steps = int(max_steps)
        self.external_feature_width = int(external_feature_width)
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
            num_layers=layers,
        )
        self.encoder_norm = nn.LayerNorm(width)
        self.role_queries = nn.Parameter(
            torch.empty(2 + self.max_steps, width)
        )
        self.role_projection = nn.Linear(width, width, bias=False)
        self.occurrence_projection = nn.Linear(width, width, bias=False)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.max_steps + 1),
        )
        nn.init.normal_(self.role_queries, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        batch: QueryPointerBatch,
        *,
        sealed_keys: HardFunctorKeys,
        frozen_byte_features: torch.Tensor | None = None,
    ) -> QueryParserOutput:
        roles = self.parse_roles(
            batch,
            frozen_byte_features=frozen_byte_features,
        )
        return bind_query_roles_to_hard_keys(
            role_occurrence_logits=roles.role_occurrence_logits,
            stop_position_logits=roles.stop_position_logits,
            query_occurrence_key_bytes=batch.occurrence_key_bytes,
            query_occurrence_valid=batch.occurrence_valid,
            sealed_keys=sealed_keys,
        )

    def parse_roles(
        self,
        batch: QueryPointerBatch,
        *,
        frozen_byte_features: torch.Tensor | None = None,
    ) -> QueryRoleOutput:
        if batch.byte_ids.numel() and (
            int(batch.byte_ids.min()) < 0
            or int(batch.byte_ids.max()) > BYTE_PAD_ID
        ):
            raise QueryParserError("query byte id leaves embedding domain")
        states = self.byte_embedding(batch.byte_ids)
        if self.external_projection is None:
            if frozen_byte_features is not None:
                raise QueryParserError(
                    "standalone query parser received frozen features"
                )
        else:
            if (
                frozen_byte_features is None
                or frozen_byte_features.shape
                != (
                    batch.batch_size,
                    batch.byte_ids.shape[1],
                    self.external_feature_width,
                )
                or not frozen_byte_features.is_floating_point()
                or frozen_byte_features.device != states.device
                or not bool(torch.isfinite(frozen_byte_features).all())
            ):
                raise QueryParserError(
                    "frozen query feature geometry differs"
                )
            states = states + self.external_projection(
                frozen_byte_features
            )
        positions = torch.arange(
            batch.byte_ids.shape[1],
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
            src_key_padding_mask=~batch.byte_valid,
        )
        states = self.encoder_norm(states)

        occurrence_states = torch.zeros(
            (
                batch.batch_size,
                MAX_QUERY_KEY_OCCURRENCES,
                self.width,
            ),
            dtype=states.dtype,
            device=states.device,
        )
        for row in range(batch.batch_size):
            count = int(batch.occurrence_valid[row].sum())
            for occurrence in range(count):
                start, end = (
                    int(value)
                    for value in batch.span_bounds[row, occurrence].tolist()
                )
                occurrence_states[row, occurrence] = states[
                    row,
                    start:end,
                ].mean(0)

        pooled = (
            states * batch.byte_valid[..., None]
        ).sum(1) / batch.byte_valid.sum(1, keepdim=True).clamp_min(1)
        role_states = self.role_queries[None] + pooled[:, None]
        role_occurrence_logits = torch.einsum(
            "brd,bod->bro",
            self.role_projection(role_states),
            self.occurrence_projection(occurrence_states),
        ) / math.sqrt(self.width)
        stop_position_logits = self.stop_head(pooled)
        return QueryRoleOutput(
            role_occurrence_logits=role_occurrence_logits,
            stop_position_logits=stop_position_logits,
        )


__all__ = [
    "MAX_QUERY_BYTES",
    "MAX_QUERY_KEY_OCCURRENCES",
    "NeuralOpaqueQueryParser",
    "QueryParserError",
    "QueryParserOutput",
    "QueryRoleOutput",
    "QueryPointerBatch",
    "ScannedQuery",
    "bind_query_roles_to_hard_keys",
    "collate_queries",
    "scan_query",
]
