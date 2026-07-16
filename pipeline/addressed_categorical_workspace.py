"""Addressed Categorical Workspace primitives and exact finite controls.

This module implements only the Track-S state substrate preregistered in
R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md.  The destination address is an
input.  There is no autonomous operator or halting controller here, so this
module cannot by itself support a reasoning claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FIELD_SIZE = 17
SYMBOLIC_DIMS = (2, 3)
SYMBOLIC_HORIZON = 16
SYMBOLIC_PROTOCOL = "R12-ACW-SYMBOLIC-v1"
SYMBOLIC_SCIENTIFIC_PATHS = (
    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md",
    "pipeline/addressed_categorical_workspace.py",
    "pipeline/audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/test_addressed_categorical_workspace.py",
    "pipeline/test_audit_addressed_categorical_workspace_symbolic.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def payload_sha256(value: dict) -> str:
    body = dict(value)
    body.pop("payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


@dataclass(frozen=True)
class AffineEvent:
    destination: int
    source: int
    alpha: int
    beta: int
    gamma: int

    def validate(self, dimension: int) -> None:
        if not 0 <= self.destination < dimension:
            raise ValueError("destination is outside the state")
        if not 0 <= self.source < dimension:
            raise ValueError("source is outside the state")
        for value in (self.alpha, self.beta, self.gamma):
            if not 0 <= value < FIELD_SIZE:
                raise ValueError("affine coefficients must be in F_17")


def apply_affine_event(
    state: Sequence[int], event: AffineEvent,
) -> tuple[int, ...]:
    event.validate(len(state))
    if any(not 0 <= value < FIELD_SIZE for value in state):
        raise ValueError("state symbols must be in F_17")
    result = list(state)
    result[event.destination] = (
        event.alpha * state[event.destination]
        + event.beta * state[event.source]
        + event.gamma
    ) % FIELD_SIZE
    return tuple(result)


def literal_packet_update(
    packet: Sequence[int], event: AffineEvent,
) -> tuple[int, ...]:
    """Apply one event while enforcing an exact single-register write."""
    before = np.asarray(tuple(packet), dtype=np.int16).reshape(1, -1)
    after = literal_packet_update_array(before, event)
    return tuple(int(value) for value in after[0])


def literal_packet_update_array(
    packets: np.ndarray, event: AffineEvent,
) -> np.ndarray:
    """Vectorized literal update used by the exhaustive finite falsifier."""

    if packets.ndim != 2:
        raise ValueError("packets must have shape [count,width]")
    event.validate(packets.shape[1])
    if bool(np.any((packets < 0) | (packets >= FIELD_SIZE))):
        raise ValueError("packet symbols must be in F_17")
    before = np.ascontiguousarray(packets)
    replacement = (
        event.alpha * before[:, event.destination].astype(np.int32)
        + event.beta * before[:, event.source].astype(np.int32)
        + event.gamma
    ) % FIELD_SIZE
    after = before.copy()
    after[:, event.destination] = replacement.astype(after.dtype)
    for index in range(before.shape[1]):
        if index != event.destination and np.any(after[:, index] != before[:, index]):
            raise AssertionError("an unaddressed register changed")
    return after


def read_packet(packet: Sequence[int], query: int) -> int:
    if not 0 <= query < len(packet):
        raise ValueError("query is outside the packet")
    value = packet[query]
    if not 0 <= value < FIELD_SIZE:
        raise ValueError("packet symbols must be in F_17")
    return value


def affine_events(
    dimension: int, coefficient_values: Sequence[int],
) -> Iterable[AffineEvent]:
    for destination in range(dimension):
        for source in range(dimension):
            for alpha in coefficient_values:
                for beta in coefficient_values:
                    for gamma in coefficient_values:
                        yield AffineEvent(destination, source, alpha, beta, gamma)


def enumerate_states(dimension: int) -> Iterable[tuple[int, ...]]:
    return itertools.product(range(FIELD_SIZE), repeat=dimension)


def recode_answer(value: int, query: int) -> int:
    multiplier = (2 * query + 1) % FIELD_SIZE
    if multiplier == 0:
        multiplier = 1
    offset = (7 * query + 3) % FIELD_SIZE
    return (multiplier * value + offset) % FIELD_SIZE


def _explicit_capacity_collision(dimension: int) -> dict:
    narrow_width = dimension - 1
    left = (0,) * dimension
    right = (0,) * narrow_width + (1,)
    left_packet = left[:narrow_width]
    right_packet = right[:narrow_width]
    query = dimension - 1
    return {
        "dimension": dimension,
        "packet_width": narrow_width,
        "packet_capacity": FIELD_SIZE ** narrow_width,
        "causal_states": FIELD_SIZE ** dimension,
        "left_state": left,
        "right_state": right,
        "left_packet": left_packet,
        "right_packet": right_packet,
        "separating_query": query,
        "left_answer": left[query],
        "right_answer": right[query],
        "collision": left_packet == right_packet,
        "separated": left[query] != right[query],
    }


def _state_array(dimension: int) -> np.ndarray:
    return np.asarray(list(enumerate_states(dimension)), dtype=np.uint8)


def _stream_event_and_array(
    digest: "hashlib._Hash", event: AffineEvent, values: np.ndarray,
) -> None:
    digest.update(bytes((
        event.destination, event.source, event.alpha, event.beta, event.gamma,
    )))
    digest.update(np.ascontiguousarray(values, dtype=np.uint8).tobytes())


def run_symbolic_dimension(
    dimension: int, *, coefficient_values: Sequence[int] | None = None,
) -> dict:
    if dimension not in SYMBOLIC_DIMS:
        raise ValueError(f"dimension must be one of {SYMBOLIC_DIMS}")
    if coefficient_values is None:
        coefficient_values = tuple(range(FIELD_SIZE))
    coefficient_values = tuple(sorted(set(int(value) for value in coefficient_values)))
    if not coefficient_values or any(
        not 0 <= value < FIELD_SIZE for value in coefficient_values
    ):
        raise ValueError("coefficient values must be a nonempty subset of F_17")
    states = _state_array(dimension)
    state_space = len(states)
    event_count = dimension * dimension * len(coefficient_values) ** 3
    exact_digest = hashlib.sha256()
    overcomplete_digest = hashlib.sha256()
    narrow_digest = hashlib.sha256(states[:, : dimension - 1].tobytes())
    event_sequence: list[AffineEvent] = []
    illegal_writes = 0
    overcomplete_illegal_writes = 0
    sentinel = (states.astype(np.int16).sum(axis=1) % FIELD_SIZE).astype(np.uint8)
    overcomplete_before = np.concatenate((states, sentinel[:, None]), axis=1)
    for event in affine_events(dimension, coefficient_values):
        if len(event_sequence) < SYMBOLIC_HORIZON:
            event_sequence.append(event)
        updated = literal_packet_update_array(states, event)
        for register in range(dimension):
            if register != event.destination:
                illegal_writes += int(np.count_nonzero(updated[:, register] != states[:, register]))
        _stream_event_and_array(exact_digest, event, updated)

        overcomplete = literal_packet_update_array(overcomplete_before, event)
        for register in range(dimension + 1):
            if register != event.destination:
                overcomplete_illegal_writes += int(np.count_nonzero(
                    overcomplete[:, register] != overcomplete_before[:, register]
                ))
        _stream_event_and_array(overcomplete_digest, event, overcomplete)

    query_digest = hashlib.sha256()
    recoding_digest = hashlib.sha256()
    for query in range(dimension):
        query_digest.update(bytes((query,)))
        query_digest.update(states[:, query].tobytes())
        recoded = np.asarray(
            [recode_answer(int(value), query) for value in states[:, query]],
            dtype=np.uint8,
        )
        recoding_digest.update(bytes((query,)))
        recoding_digest.update(recoded.tobytes())

    donor = ((states.astype(np.int16) + 1) % FIELD_SIZE).astype(np.uint8)
    donor_digest = hashlib.sha256()
    donor_digest.update(states.tobytes())
    donor_digest.update(donor.tobytes())
    for query in range(dimension):
        donor_values = donor[:, query]
        if np.any(donor_values == states[:, query]):
            raise AssertionError("literal donor did not separate the held-fixed source")
        donor_digest.update(bytes((query,)))
        donor_digest.update(donor_values.tobytes())

    horizon = states.copy()
    for event in event_sequence:
        before = horizon.astype(np.int16, copy=False)
        replacement = (
            event.alpha * before[:, event.destination]
            + event.beta * before[:, event.source]
            + event.gamma
        ) % FIELD_SIZE
        horizon[:, event.destination] = replacement.astype(np.uint8)
    horizon_digest = hashlib.sha256(horizon.tobytes())

    collision = _explicit_capacity_collision(dimension)
    narrow_unique = int(len(np.unique(states[:, : dimension - 1], axis=0)))
    gates = {
        "exact_capacity_matches_causal_state_count": (
            FIELD_SIZE ** dimension == state_space
        ),
        "narrow_capacity_is_insufficient": (
            collision["packet_capacity"] < collision["causal_states"]
        ),
        "narrow_collision_is_explicit_and_separated": (
            collision["collision"] and collision["separated"]
        ),
        "narrow_width_exhaustively_maps_all_states": (
            narrow_unique == FIELD_SIZE ** (dimension - 1)
        ),
        "exact_width_stream_is_complete": exact_digest.digest_size == 32,
        "overcomplete_width_stream_is_complete": overcomplete_digest.digest_size == 32,
        "all_states_checked": state_space == FIELD_SIZE ** dimension,
        "all_affine_updates_checked": (
            state_space * event_count
            == FIELD_SIZE ** dimension * dimension * dimension
            * len(coefficient_values) ** 3
        ),
        "all_queries_checked": state_space * dimension > 0,
        "all_recodings_checked": state_space * dimension > 0,
        "all_literal_donor_reads_checked": state_space * dimension > 0,
        "all_horizons_checked": len(event_sequence) == SYMBOLIC_HORIZON,
        "zero_illegal_writes": illegal_writes == 0,
        "full_coefficient_field_exhausted": (
            coefficient_values == tuple(range(FIELD_SIZE))
        ),
        "overcomplete_sentinel_is_byte_preserved": overcomplete_illegal_writes == 0,
    }
    return {
        "dimension": dimension,
        "field_size": FIELD_SIZE,
        "state_space": state_space,
        "packet_symbols": dimension,
        "utilized_bits": dimension * math.log2(FIELD_SIZE),
        "physical_bits": dimension * math.ceil(math.log2(FIELD_SIZE)),
        "coefficient_values": coefficient_values,
        "full_coefficient_field": coefficient_values == tuple(range(FIELD_SIZE)),
        "events": event_count,
        "states_checked": state_space,
        "updates_checked": state_space * event_count,
        "queries_checked": state_space * dimension,
        "recodings_checked": state_space * dimension,
        "literal_donor_reads_checked": state_space * dimension,
        "horizon_checks": state_space,
        "horizon_depth": SYMBOLIC_HORIZON,
        "illegal_writes": illegal_writes,
        "overcomplete_illegal_writes": overcomplete_illegal_writes,
        "widths_tested": (dimension - 1, dimension, dimension + 1),
        "narrow_unique_packets": narrow_unique,
        "narrow_packet_stream_sha256": narrow_digest.hexdigest(),
        "exact_update_stream_sha256": exact_digest.hexdigest(),
        "overcomplete_update_stream_sha256": overcomplete_digest.hexdigest(),
        "query_stream_sha256": query_digest.hexdigest(),
        "recoding_stream_sha256": recoding_digest.hexdigest(),
        "literal_donor_stream_sha256": donor_digest.hexdigest(),
        "horizon_stream_sha256": horizon_digest.hexdigest(),
        "narrow_collision": collision,
        "gates": gates,
        "pass": all(gates.values()),
    }


def scientific_identity(*, require_clean: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *SYMBOLIC_SCIENTIFIC_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("symbolic scientific paths are not clean in Git")
    hashes = {}
    for relative in SYMBOLIC_SCIENTIFIC_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing symbolic scientific path: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"scientific_commit": commit, "scientific_path_sha256": hashes}


def run_symbolic_falsifier(
    *,
    coefficient_values: Sequence[int] | None = None,
    bind_identity: bool = False,
) -> dict:
    dimensions = [
        run_symbolic_dimension(
            dimension, coefficient_values=coefficient_values,
        )
        for dimension in SYMBOLIC_DIMS
    ]
    report = {
        "protocol": SYMBOLIC_PROTOCOL,
        "field_size": FIELD_SIZE,
        "scientific_identity": (
            scientific_identity(require_clean=True) if bind_identity else None
        ),
        "dimensions": dimensions,
        "claim_boundary": (
            "Exact affine packet mechanics only; literal donor read is not a learned "
            "causal intervention. No neural learning, "
            "language, autonomous control, novelty, or reasoning claim."
        ),
        "pass": all(item["pass"] for item in dimensions),
    }
    report["payload_sha256"] = payload_sha256(report)
    return report


class _ExactStraightThroughCategorical(torch.autograd.Function):
    """Exact one-hot forward with the softmax Jacobian as its backward rule."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        probabilities = logits.softmax(dim=-1)
        ctx.save_for_backward(probabilities)
        hard = torch.zeros_like(logits)
        hard.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1)
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (probabilities,) = ctx.saved_tensors
        centered = gradient - (gradient * probabilities).sum(dim=-1, keepdim=True)
        return (probabilities * centered,)


def hard_categorical(logits: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("categorical logits require at least two categories")
    if straight_through:
        return _ExactStraightThroughCategorical.apply(logits)
    hard = torch.zeros_like(logits)
    hard.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1)
    return hard


def validate_literal_one_hot(packet: torch.Tensor) -> None:
    if packet.ndim < 2:
        raise ValueError("categorical packet must have at least two dimensions")
    if not packet.is_floating_point():
        raise ValueError("one-hot packet must use a floating dtype")
    reconstructed = torch.zeros_like(packet)
    reconstructed.scatter_(-1, packet.argmax(dim=-1, keepdim=True), 1)
    if not torch.equal(packet, reconstructed):
        raise ValueError("packet is not literal one-hot")


def packet_to_symbols(packet: torch.Tensor) -> torch.Tensor:
    validate_literal_one_hot(packet)
    if packet.shape[-1] > 256:
        raise ValueError("uint8 persistence supports at most 256 categories")
    return packet.argmax(dim=-1).to(torch.uint8)


def symbols_to_packet(
    symbols: torch.Tensor, categories: int, *, dtype: torch.dtype, device=None,
) -> torch.Tensor:
    if symbols.dtype != torch.uint8:
        raise ValueError("persistent packet symbols must use uint8")
    if categories <= 1 or categories > 256:
        raise ValueError("categories must be in [2,256]")
    if bool((symbols.to(torch.int16) >= categories).any()):
        raise ValueError("persistent symbol is outside the category range")
    one_hot = F.one_hot(symbols.to(torch.int64), categories)
    return one_hot.to(device=device, dtype=dtype)


class AddressedCategoricalWorkspace(nn.Module):
    """Small Track-S sidecar with an externally scheduled one-register write."""

    def __init__(
        self,
        source_dim: int = 576,
        event_dim: int = 576,
        packet_symbols: int = 4,
        categories: int = FIELD_SIZE,
        updater_hidden: int = 64,
        bridge_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(source_dim, event_dim, packet_symbols, categories, updater_hidden, bridge_dim) <= 0:
            raise ValueError("all workspace dimensions must be positive")
        self.source_dim = source_dim
        self.event_dim = event_dim
        self.packet_symbols = packet_symbols
        self.categories = categories
        self.updater_hidden = updater_hidden
        self.bridge_dim = bridge_dim
        flat_packet = packet_symbols * categories
        self.source_projector = nn.Linear(source_dim, flat_packet)
        self.event_projector = nn.Linear(event_dim, flat_packet)
        self.updater = nn.Sequential(
            nn.Linear(2 * flat_packet + packet_symbols, updater_hidden),
            nn.SiLU(),
            nn.Linear(updater_hidden, categories),
        )
        self.bridge = nn.Linear(flat_packet, bridge_dim, bias=False)

    def _project_packet(
        self, hidden: torch.Tensor, projector: nn.Linear, *, straight_through: bool,
    ) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != projector.in_features:
            raise ValueError("hidden input has the wrong shape")
        logits = projector(hidden).reshape(
            hidden.shape[0], self.packet_symbols, self.categories,
        )
        return hard_categorical(logits, straight_through=straight_through)

    def encode_source(
        self, source_hidden: torch.Tensor, *, straight_through: bool | None = None,
    ) -> torch.Tensor:
        if straight_through is None:
            straight_through = self.training
        return self._project_packet(
            source_hidden, self.source_projector, straight_through=straight_through,
        )

    def encode_event(
        self, event_hidden: torch.Tensor, *, straight_through: bool | None = None,
    ) -> torch.Tensor:
        if straight_through is None:
            straight_through = self.training
        return self._project_packet(
            event_hidden, self.event_projector, straight_through=straight_through,
        )

    def update(
        self,
        packet: torch.Tensor,
        event_code: torch.Tensor,
        address: torch.Tensor,
        *,
        straight_through: bool | None = None,
    ) -> torch.Tensor:
        expected = (packet.shape[0], self.packet_symbols, self.categories)
        if packet.shape != expected or event_code.shape != expected:
            raise ValueError("packet and event code have the wrong shape")
        if address.shape != (packet.shape[0],):
            raise ValueError("address must have shape [batch]")
        if address.dtype not in (torch.int32, torch.int64):
            raise ValueError("address must use an integer dtype")
        if bool(((address < 0) | (address >= self.packet_symbols)).any()):
            raise ValueError("address is outside the packet")
        if straight_through is None:
            straight_through = self.training
        if not straight_through:
            validate_literal_one_hot(packet)
            validate_literal_one_hot(event_code)
        address_one_hot = F.one_hot(
            address.to(torch.int64), self.packet_symbols,
        ).to(device=packet.device, dtype=packet.dtype)
        features = torch.cat(
            (
                packet.reshape(packet.shape[0], -1),
                event_code.reshape(event_code.shape[0], -1),
                address_one_hot,
            ),
            dim=-1,
        )
        replacement = hard_categorical(
            self.updater(features), straight_through=straight_through,
        )
        write_mask = address_one_hot.unsqueeze(-1)
        return packet * (1 - write_mask) + replacement.unsqueeze(1) * write_mask

    @torch.no_grad()
    def encode_source_symbols(self, source_hidden: torch.Tensor) -> torch.Tensor:
        return packet_to_symbols(
            self.encode_source(source_hidden, straight_through=False),
        )

    @torch.no_grad()
    def encode_event_symbols(self, event_hidden: torch.Tensor) -> torch.Tensor:
        return packet_to_symbols(
            self.encode_event(event_hidden, straight_through=False),
        )

    @torch.no_grad()
    def update_symbols(
        self,
        packet_symbols: torch.Tensor,
        event_symbols: torch.Tensor,
        address: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.updater[0].weight.dtype
        device = self.updater[0].weight.device
        packet = symbols_to_packet(
            packet_symbols, self.categories, dtype=dtype, device=device,
        )
        event = symbols_to_packet(
            event_symbols, self.categories, dtype=dtype, device=device,
        )
        updated = self.update(
            packet, event, address.to(device), straight_through=False,
        )
        return packet_to_symbols(updated)

    @torch.no_grad()
    def packet_delta_symbols(self, packet_symbols: torch.Tensor) -> torch.Tensor:
        dtype = self.bridge.weight.dtype
        packet = symbols_to_packet(
            packet_symbols, self.categories, dtype=dtype, device=self.bridge.weight.device,
        )
        return self.packet_delta(packet)

    def packet_delta(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim != 3 or packet.shape[1:] != (
            self.packet_symbols, self.categories,
        ):
            raise ValueError("packet has the wrong shape")
        return self.bridge(packet.reshape(packet.shape[0], -1))

    def parameter_ledger(self) -> dict:
        components = {
            "source_projector": sum(p.numel() for p in self.source_projector.parameters()),
            "event_projector": sum(p.numel() for p in self.event_projector.parameters()),
            "updater": sum(p.numel() for p in self.updater.parameters()),
            "bridge": sum(p.numel() for p in self.bridge.parameters()),
        }
        return {"components": components, "total": sum(components.values())}


class ScalarQueryReader(nn.Module):
    """Shared scalar-query reader used by every parameter-matched CPU arm."""

    def __init__(
        self,
        state_dim: int = 32,
        queries: int = 24,
        query_dim: int = 16,
        hidden: int = 64,
        answers: int = FIELD_SIZE,
    ) -> None:
        super().__init__()
        self.queries = queries
        self.query_embedding = nn.Embedding(queries, query_dim)
        self.network = nn.Sequential(
            nn.Linear(state_dim + query_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, answers),
        )

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("reader state must have shape [batch, width]")
        if query.shape != (state.shape[0],):
            raise ValueError("query must have shape [batch]")
        if query.dtype not in (torch.int32, torch.int64):
            raise ValueError("query must use an integer dtype")
        if bool(((query < 0) | (query >= self.queries)).any()):
            raise ValueError("query is outside the reader bank")
        features = torch.cat((state, self.query_embedding(query)), dim=-1)
        return self.network(features)


class CategoricalTrackSModel(nn.Module):
    """Preregistered d=3 CPU ACW treatment including the shared reader."""

    def __init__(self) -> None:
        super().__init__()
        self.workspace = AddressedCategoricalWorkspace(
            source_dim=96,
            event_dim=96,
            packet_symbols=3,
            categories=FIELD_SIZE,
            updater_hidden=80,
            bridge_dim=32,
        )
        self.reader = ScalarQueryReader()

    def read(self, packet: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.reader(self.workspace.packet_delta(packet), query)


class DenseCategoricalTrackSModel(nn.Module):
    """Categorical control that may rewrite every register on every event."""

    def __init__(self) -> None:
        super().__init__()
        self.packet_symbols = 3
        self.categories = FIELD_SIZE
        flat = self.packet_symbols * self.categories
        self.source_projector = nn.Linear(96, flat)
        self.event_projector = nn.Linear(96, flat)
        self.updater = nn.Sequential(
            nn.Linear(2 * flat + self.packet_symbols, 64),
            nn.SiLU(),
            nn.Linear(64, flat),
        )
        self.bridge = nn.Linear(flat, 32, bias=False)
        self.reader = ScalarQueryReader()

    def encode_source(self, hidden: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
        logits = self.source_projector(hidden).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def encode_event(self, hidden: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
        logits = self.event_projector(hidden).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def update(
        self,
        packet: torch.Tensor,
        event_code: torch.Tensor,
        address: torch.Tensor,
        *,
        straight_through: bool,
    ) -> torch.Tensor:
        address_one_hot = F.one_hot(address, 3).to(packet.dtype)
        features = torch.cat(
            (packet.flatten(1), event_code.flatten(1), address_one_hot), dim=-1,
        )
        logits = self.updater(features).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def read(self, packet: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.reader(self.bridge(packet.flatten(1)), query)


class AddressedContinuousTrackSModel(nn.Module):
    """Continuous control with the same externally supplied one-write mask."""

    def __init__(self) -> None:
        super().__init__()
        self.source_projector = nn.Linear(96, 3)
        self.event_projector = nn.Linear(96, 3 * FIELD_SIZE)
        self.updater = nn.Sequential(
            nn.Linear(3 + 3 * FIELD_SIZE + 3, 272),
            nn.SiLU(),
            nn.Linear(272, 1),
        )
        self.bridge = nn.Linear(3, 32, bias=False)
        self.reader = ScalarQueryReader()

    def encode_source(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.source_projector(hidden)

    def encode_event(self, hidden: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
        logits = self.event_projector(hidden).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def update(
        self, state: torch.Tensor, event_code: torch.Tensor, address: torch.Tensor,
    ) -> torch.Tensor:
        address_one_hot = F.one_hot(address, 3).to(state.dtype)
        features = torch.cat((state, event_code.flatten(1), address_one_hot), dim=-1)
        replacement = self.updater(features)
        return state * (1 - address_one_hot) + replacement * address_one_hot

    def read(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.reader(self.bridge(state), query)


class GRUTrackSModel(nn.Module):
    """Favorable dense continuous recurrent control."""

    def __init__(self) -> None:
        super().__init__()
        self.source_projector = nn.Linear(96, 39)
        self.updater = nn.GRUCell(96 + 3, 39)
        self.bridge = nn.Linear(39, 32, bias=False)
        self.reader = ScalarQueryReader()

    def encode_source(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.source_projector(hidden)

    def update(
        self, state: torch.Tensor, event_hidden: torch.Tensor, address: torch.Tensor,
    ) -> torch.Tensor:
        address_one_hot = F.one_hot(address, 3).to(state.dtype)
        return self.updater(torch.cat((event_hidden, address_one_hot), dim=-1), state)

    def read(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.reader(self.bridge(state), query)


class PacketTokenTransformerTrackSModel(nn.Module):
    """One-block recurrent packet-token transformer control."""

    def __init__(self) -> None:
        super().__init__()
        flat = 3 * FIELD_SIZE
        self.source_projector = nn.Linear(96, flat)
        self.event_projector = nn.Linear(96, flat)
        self.token_input = nn.Linear(FIELD_SIZE, 24)
        self.address_embedding = nn.Embedding(3, 24)
        self.block = nn.TransformerEncoderLayer(
            d_model=24,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation=F.silu,
            batch_first=True,
        )
        self.token_output = nn.Linear(24, FIELD_SIZE)
        self.bridge = nn.Linear(flat, 32, bias=False)
        self.reader = ScalarQueryReader()

    def encode_source(self, hidden: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
        logits = self.source_projector(hidden).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def encode_event(self, hidden: torch.Tensor, *, straight_through: bool) -> torch.Tensor:
        logits = self.event_projector(hidden).reshape(-1, 3, FIELD_SIZE)
        return hard_categorical(logits, straight_through=straight_through)

    def update(
        self,
        packet: torch.Tensor,
        event_code: torch.Tensor,
        address: torch.Tensor,
        *,
        straight_through: bool,
    ) -> torch.Tensor:
        packet_tokens = self.token_input(packet)
        event_tokens = self.token_input(event_code)
        address_token = self.address_embedding(address).unsqueeze(1)
        tokens = torch.cat((packet_tokens, event_tokens, address_token), dim=1)
        updated = self.block(tokens)[:, :3]
        return hard_categorical(
            self.token_output(updated), straight_through=straight_through,
        )

    def read(self, packet: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.reader(self.bridge(packet.flatten(1)), query)


class AnswerMotorControl(nn.Module):
    """Commutative answer-specific control with no recurrent state."""

    def __init__(self) -> None:
        super().__init__()
        self.query_embedding = nn.Embedding(24, 16)
        self.network = nn.Sequential(
            nn.Linear(96 + 96 + 16, 113),
            nn.SiLU(),
            nn.Linear(113, FIELD_SIZE),
        )

    def forward(
        self,
        source_hidden: torch.Tensor,
        mean_event_hidden: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if source_hidden.shape != mean_event_hidden.shape:
            raise ValueError("source and mean-event features must have equal shape")
        if source_hidden.ndim != 2 or source_hidden.shape[-1] != 96:
            raise ValueError("motor features must have shape [batch, 96]")
        features = torch.cat(
            (source_hidden, mean_event_hidden, self.query_embedding(query)), dim=-1,
        )
        return self.network(features)


class SourceRetainedTrackSModel(nn.Module):
    """Favorable recurrent upper bound that retains source features at readout."""

    def __init__(self) -> None:
        super().__init__()
        self.source_projector = nn.Linear(96, 128)
        self.updater = nn.GRUCell(96 + 3, 128)
        self.query_embedding = nn.Embedding(24, 16)
        self.reader = nn.Sequential(
            nn.Linear(128 + 96 + 16, 256),
            nn.SiLU(),
            nn.Linear(256, FIELD_SIZE),
        )

    def encode_source(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.source_projector(hidden)

    def update(
        self, state: torch.Tensor, event_hidden: torch.Tensor, address: torch.Tensor,
    ) -> torch.Tensor:
        address_one_hot = F.one_hot(address, 3).to(state.dtype)
        return self.updater(torch.cat((event_hidden, address_one_hot), dim=-1), state)

    def read(
        self,
        state: torch.Tensor,
        retained_source: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (state, retained_source, self.query_embedding(query)), dim=-1,
        )
        return self.reader(features)


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def write_report(report: dict, path: Path) -> None:
    expected_payload = payload_sha256(report)
    if report.get("payload_sha256") != expected_payload:
        raise ValueError("report payload hash is missing or stale")
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite symbolic report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(report) + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale symbolic temporary output: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)
    path.chmod(0o444)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--coefficient-values",
        type=int,
        nargs="+",
        help="test-only coefficient subset; omission means exhaustive F_17",
    )
    parser.add_argument(
        "--allow-uncommitted",
        action="store_true",
        help="test-only: omit clean committed scientific identity",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_symbolic_falsifier(
        coefficient_values=args.coefficient_values,
        bind_identity=not args.allow_uncommitted,
    )
    if args.out is not None:
        write_report(report, args.out)
    print(
        f"[acw-symbolic] pass={report['pass']} "
        f"payload_sha256={report['payload_sha256']}"
    )
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
