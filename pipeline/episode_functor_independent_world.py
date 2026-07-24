#!/usr/bin/env python3
"""Independent consumed world generator for the EFC deployed-wire rehearsal.

This file intentionally imports no EPISODE generator, protocol, serializer, or
runtime module. It uses a counter-mode SHA-256 stream and Fisher-Yates
construction distinct from the original hash-ranked generator. Candidate
admission reads mechanics only: transition totality/bijection,
noncommutativity, observer shape, and observer separation. It has no query
type, query argument, challenge seed, prediction, or answer-selection API.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import struct
from typing import Sequence


GENERATOR_SCHEMA = "efc-independent-counter-world-v1"
SEED_DOMAIN = b"EFC-independent-world-seed-v1\0"
STREAM_LABELS = (
    "state-keys",
    "action-keys",
    "observer-keys",
    "demonstration-order",
    "observation-order",
    "renderer-choice",
)


class IndependentWorldError(ValueError):
    """The independent generator failed a frozen mechanics invariant."""


@dataclass(frozen=True)
class IndependentWorld:
    evidence: bytes
    transitions: tuple[tuple[int, ...], ...]
    observers: tuple[tuple[int, ...], ...]
    stream_commitments: tuple[tuple[str, str], ...]
    world_seed_commitment: str
    admissibility_receipt: dict[str, object]
    accepted_candidate: int


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _framed(*parts: bytes) -> bytes:
    output = bytearray()
    for part in parts:
        output.extend(struct.pack(">Q", len(part)))
        output.extend(part)
    return bytes(output)


def _derive_world_seed(
    protocol_root: str,
    beacon_round: int,
    beacon_value: str,
) -> bytes:
    if len(protocol_root) != 64:
        raise IndependentWorldError("protocol root is not a SHA-256 hex digest")
    if beacon_round < 0 or not beacon_value:
        raise IndependentWorldError("world beacon is malformed")
    return sha256(
        SEED_DOMAIN
        + _framed(
            bytes.fromhex(protocol_root),
            struct.pack(">Q", beacon_round),
            beacon_value.encode("utf-8"),
        )
    ).digest()


class _CounterStream:
    def __init__(self, seed: bytes, label: str) -> None:
        self._key = _stream_key(seed, label)
        self._counter = 0
        self._buffer = bytearray()

    def _fill(self, count: int) -> None:
        while len(self._buffer) < count:
            self._buffer.extend(
                sha256(
                    self._key + struct.pack(">Q", self._counter)
                ).digest()
            )
            self._counter += 1

    def take(self, count: int) -> bytes:
        self._fill(count)
        output = bytes(self._buffer[:count])
        del self._buffer[:count]
        return output

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def below(self, upper: int) -> int:
        if upper <= 0:
            raise IndependentWorldError("sampling upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.u64()
            if value < limit:
                return value % upper

    def shuffled(self, values: Sequence[int]) -> tuple[int, ...]:
        result = list(values)
        for right in range(len(result) - 1, 0, -1):
            left = self.below(right + 1)
            result[left], result[right] = result[right], result[left]
        return tuple(result)


def _stream_key(seed: bytes, label: str) -> bytes:
    return sha256(
        b"EFC-independent-stream-v1\0"
        + _framed(seed, label.encode("ascii"))
    ).digest()


def _unique_nonzero_u64(
    stream: _CounterStream,
    count: int,
) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        value = stream.u64()
        if value != 0 and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


def _compose(
    first: Sequence[int],
    second: Sequence[int],
) -> tuple[int, ...]:
    return tuple(second[first[state]] for state in range(len(first)))


def _mechanics_receipt(
    transitions: tuple[tuple[int, ...], ...],
    observers: tuple[tuple[int, ...], ...],
    state_count: int,
    action_count: int,
    observer_count: int,
    answer_count: int,
) -> dict[str, object]:
    complete = (
        len(transitions) == action_count
        and all(len(row) == state_count for row in transitions)
        and all(
            destination in range(state_count)
            for row in transitions
            for destination in row
        )
    )
    bijective = complete and all(
        sorted(row) == list(range(state_count)) for row in transitions
    )
    noncommuting = complete and any(
        _compose(transitions[left], transitions[right])
        != _compose(transitions[right], transitions[left])
        for left in range(action_count)
        for right in range(left + 1, action_count)
    )
    observer_shape = (
        len(observers) == observer_count
        and all(len(row) == state_count for row in observers)
        and all(
            answer in range(answer_count)
            for row in observers
            for answer in row
        )
    )
    empty_signatures = (
        tuple(
            tuple(observer[state] for observer in observers)
            for state in range(state_count)
        )
        if observer_shape
        else ()
    )
    empty_class_count = len(set(empty_signatures))
    nontrivial_empty_partition = (
        observer_shape and 1 < empty_class_count < state_count
    )
    future_class_count = 0
    if observer_shape and complete:
        classes = _class_ids(empty_signatures)
        while True:
            signatures = tuple(
                (
                    empty_signatures[state],
                    tuple(
                        classes[transition[state]]
                        for transition in transitions
                    ),
                )
                for state in range(state_count)
            )
            refined = _class_ids(signatures)
            if all(
                (refined[left] == refined[right])
                == (classes[left] == classes[right])
                for left in range(state_count)
                for right in range(state_count)
            ):
                future_class_count = len(set(refined))
                break
            classes = refined
    future_separation = future_class_count == state_count
    checks = {
        "future_separation": future_separation,
        "noncommutativity": noncommuting,
        "nontrivial_empty_observer_partition": (
            nontrivial_empty_partition
        ),
        "observer_shape": observer_shape,
        "transition_bijection": bijective,
        "transition_completeness": complete,
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "generator": GENERATOR_SCHEMA,
        "inspected_fields": sorted(checks),
        "empty_observer_class_count": empty_class_count,
        "future_behavior_class_count": future_class_count,
        "query_fields_seen": 0,
    }


def _class_ids(signatures: Sequence[object]) -> tuple[int, ...]:
    identifiers: dict[object, int] = {}
    result: list[int] = []
    for signature in signatures:
        if signature not in identifiers:
            identifiers[signature] = len(identifiers)
        result.append(identifiers[signature])
    return tuple(result)


def generate_independent_world(
    *,
    protocol_root: str,
    beacon_round: int,
    beacon_value: str,
    state_count: int,
    action_count: int,
    observer_count: int,
    answer_count: int,
    renderer_count: int,
) -> IndependentWorld:
    """Generate one world without accepting challenge-related input."""

    if not 4 <= state_count <= 16:
        raise IndependentWorldError("state count is outside independent support")
    if not 2 <= action_count <= 8:
        raise IndependentWorldError("action count is outside independent support")
    if not 1 <= observer_count <= 8:
        raise IndependentWorldError("observer count is outside independent support")
    if answer_count < state_count or renderer_count <= 0:
        raise IndependentWorldError("observer or renderer support is invalid")

    world_seed = _derive_world_seed(
        protocol_root, beacon_round, beacon_value
    )
    accepted_candidate = -1
    accepted_candidate_seed: bytes | None = None
    transitions: tuple[tuple[int, ...], ...] | None = None
    observers: tuple[tuple[int, ...], ...] | None = None
    for candidate in range(1_024):
        candidate_seed = sha256(
            world_seed
            + b"\0candidate\0"
            + struct.pack(">I", candidate)
        ).digest()
        transition_rows = tuple(
            _CounterStream(
                candidate_seed,
                f"transition-action-{action}",
            ).shuffled(range(state_count))
            for action in range(action_count)
        )
        observer_rows: list[tuple[int, ...]] = []
        observer_alphabet = min(answer_count, 2)
        for observer in range(observer_count):
            observer_stream = _CounterStream(
                candidate_seed, f"observer-values-{observer}"
            )
            observer_rows.append(
                tuple(
                    observer_stream.below(observer_alphabet)
                    for _ in range(state_count)
                )
            )
        candidate_observers = tuple(observer_rows)
        receipt = _mechanics_receipt(
            transition_rows,
            candidate_observers,
            state_count,
            action_count,
            observer_count,
            answer_count,
        )
        if receipt["admitted"]:
            accepted_candidate = candidate
            accepted_candidate_seed = candidate_seed
            transitions = transition_rows
            observers = candidate_observers
            break
    if (
        transitions is None
        or observers is None
        or accepted_candidate_seed is None
    ):
        raise IndependentWorldError("no mechanics-only candidate was admitted")

    streams = {
        label: _CounterStream(world_seed, label) for label in STREAM_LABELS
    }
    state_keys = _unique_nonzero_u64(
        streams["state-keys"], state_count
    )
    action_keys = _unique_nonzero_u64(
        streams["action-keys"], action_count
    )
    observer_keys = _unique_nonzero_u64(
        streams["observer-keys"], observer_count
    )
    state_old_for_new = tuple(
        sorted(range(state_count), key=state_keys.__getitem__)
    )
    state_old_to_new = {
        old: new for new, old in enumerate(state_old_for_new)
    }
    action_old_for_new = tuple(
        sorted(range(action_count), key=action_keys.__getitem__)
    )
    observer_old_for_new = tuple(
        sorted(range(observer_count), key=observer_keys.__getitem__)
    )
    canonical_state_keys = tuple(
        state_keys[old] for old in state_old_for_new
    )
    canonical_action_keys = tuple(
        action_keys[old] for old in action_old_for_new
    )
    canonical_observer_keys = tuple(
        observer_keys[old] for old in observer_old_for_new
    )
    canonical_transitions = tuple(
        tuple(
            state_old_to_new[transitions[old_action][old_state]]
            for old_state in state_old_for_new
        )
        for old_action in action_old_for_new
    )
    canonical_observers = tuple(
        tuple(
            observers[old_observer][old_state]
            for old_state in state_old_for_new
        )
        for old_observer in observer_old_for_new
    )
    demonstrations = [
        {
            "action_key": canonical_action_keys[action],
            "source_key": canonical_state_keys[state],
            "target_key": canonical_state_keys[
                canonical_transitions[action][state]
            ],
        }
        for state in range(state_count)
        for action in range(action_count)
    ]
    order = streams["demonstration-order"].shuffled(
        range(len(demonstrations))
    )
    observation_events = [
        {
            "answer": canonical_observers[observer][state],
            "observer_key": canonical_observer_keys[observer],
            "state_key": canonical_state_keys[state],
        }
        for state in range(state_count)
        for observer in range(observer_count)
    ]
    observation_order = streams["observation-order"].shuffled(
        range(len(observation_events))
    )
    evidence_row = {
        "demonstrations": [demonstrations[index] for index in order],
        "observations": [
            observation_events[index] for index in observation_order
        ],
        "renderer_choice": streams["renderer-choice"].below(renderer_count),
        "schema": "efc-raw-world-evidence-v2",
    }
    admissibility = _mechanics_receipt(
        canonical_transitions,
        canonical_observers,
        state_count,
        action_count,
        observer_count,
        answer_count,
    )
    if not admissibility["admitted"]:
        raise IndependentWorldError("accepted mechanics failed replay")
    mechanics_labels = (
        *(f"transition-action-{action}" for action in range(action_count)),
        *(
            f"observer-values-{observer}"
            for observer in range(observer_count)
        ),
    )
    stream_commitments = (
        (
            f"accepted-mechanics-candidate-{accepted_candidate}",
            sha256(
                b"EFC-independent-candidate-commitment-v1\0"
                + accepted_candidate_seed
            ).hexdigest(),
        ),
        *(
            (
                f"mechanics/{label}",
                sha256(
                    b"EFC-independent-stream-key-commitment-v1\0"
                    + _stream_key(accepted_candidate_seed, label)
                ).hexdigest(),
            )
            for label in mechanics_labels
        ),
        *tuple(
        (
            f"world/{label}",
            sha256(
                b"EFC-independent-stream-key-commitment-v1\0"
                + _stream_key(world_seed, label)
            ).hexdigest(),
        )
        for label in STREAM_LABELS
        ),
    )
    return IndependentWorld(
        evidence=_canonical_bytes(evidence_row),
        transitions=canonical_transitions,
        observers=canonical_observers,
        stream_commitments=stream_commitments,
        world_seed_commitment=sha256(
            b"EFC-independent-world-seed-commitment-v1\0" + world_seed
        ).hexdigest(),
        admissibility_receipt=admissibility,
        accepted_candidate=accepted_candidate,
    )


__all__ = [
    "GENERATOR_SCHEMA",
    "IndependentWorld",
    "IndependentWorldError",
    "generate_independent_world",
]
