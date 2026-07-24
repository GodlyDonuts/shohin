#!/usr/bin/env python3
"""Consumed CPU rehearsal of the EFC seal protocol on the deployed wire.

This module unifies the two-beacon custody order with the exact fixed-width
format consumed by ``tools/episode_functor_runtime_c.c`` and
``tools/episode_functor_runtime_rust.rs``. It does not fit a neural compiler,
create an official board, use a checkpoint, or authorize pretraining.

The compiler receives canonical public world evidence only. After one
1,536-byte machine is sealed, the source is deliberately poisoned and deleted.
A later challenge beacon selects abstract coordinates, those coordinates are
sealed, and only then are opaque keys rendered into ``queries.bin``. The C and
Rust runtimes receive only ``machine.bin`` and ``queries.bin``. Their
byte-identical transcripts are checked by a third relation-composition
assessor that never parses the machine wire.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Mapping, Sequence

from pipeline.episode_functor_seal_protocol import (
    AbstractCoordinate,
    Beacon,
    ProtocolSpec as CompactProtocolSpec,
    ProtocolViolation,
    WorldFixture,
    WorldMechanics,
    _hex_commitment,
    _mutate_consumed_source_for_invariance_test,
    _publish_immutable,
    canonical_json_bytes,
    derive_challenge_seed,
    generate_abstract_coordinates,
    generate_consumed_world_fixture,
)
from pipeline.episode_functor_independent_world import (
    GENERATOR_SCHEMA as INDEPENDENT_GENERATOR_SCHEMA,
    generate_independent_world,
)
from pipeline.episode_functor_source_renderers import (
    LINE_MAGIC as SOURCE_LINE_MAGIC,
    SourceRendererError,
    decode_line_events,
)


ROOT = Path(__file__).resolve().parents[1]
C_RUNTIME_SOURCE = ROOT / "tools" / "episode_functor_runtime_c.c"
RUST_RUNTIME_SOURCE = ROOT / "tools" / "episode_functor_runtime_rust.rs"
INDEPENDENT_GENERATOR_SOURCE = (
    ROOT / "pipeline" / "episode_functor_independent_world.py"
)
SOURCE_RENDERER_SOURCE = (
    ROOT / "pipeline" / "episode_functor_source_renderers.py"
)

FORMAT_VERSION = 1
MACHINE_SIZE = 1_536
MACHINE_HASH_OFFSET = 1_504
QUERY_HEADER_SIZE = 64
QUERY_RECORD_SIZE = 320
TRANSCRIPT_HEADER_SIZE = 96
TRANSCRIPT_RECORD_SIZE = 32
HASH_SIZE = 32
MAX_STATES = 16
MAX_ACTIONS = 8
MAX_OBSERVERS = 8
MAX_WORD = 32
MACHINE_MAGIC = b"EFCMACH\0"
QUERY_MAGIC = b"EFCQRY\0\0"
TRANSCRIPT_MAGIC = b"EFCOUT\0\0"
PROTOCOL_DOMAIN = "EFC/deployed-wire-protocol-root/v2"
MACHINE_DOMAIN = "EFC/deployed-wire-machine-root/v1"
COORDINATE_DOMAIN = "EFC/deployed-wire-coordinate-root/v1"
QUERY_DOMAIN = "EFC/deployed-wire-query-root/v1"
PREDICTION_DOMAIN = "EFC/deployed-wire-prediction-root/v1"
ANSWER_DOMAIN = "EFC/deployed-wire-answer-root/v1"

WORLD_EVIDENCE_FIELDS = frozenset(
    {
        "action_keys",
        "demonstrations",
        "observer_keys",
        "observer_rows",
        "renderer_choice",
        "schema",
        "state_keys",
    }
)
RAW_WORLD_EVIDENCE_FIELDS = frozenset(
    {
        "demonstrations",
        "observations",
        "renderer_choice",
        "schema",
    }
)


def _sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _read_u16(buffer: bytes, offset: int) -> int:
    return struct.unpack_from("<H", buffer, offset)[0]


def _read_u32(buffer: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buffer, offset)[0]


def _read_u64(buffer: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", buffer, offset)[0]


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolViolation(f"{field} must be a plain integer")
    return value


def _integer_vector(
    row: Mapping[str, object],
    field: str,
    count: int,
) -> tuple[int, ...]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != count:
        raise ProtocolViolation(f"{field} has incorrect cardinality")
    values = tuple(_plain_int(item, field) for item in value)
    if any(item <= 0 or item >= 1 << 64 for item in values):
        raise ProtocolViolation(f"{field} values must be nonzero uint64")
    if len(set(values)) != len(values):
        raise ProtocolViolation(f"{field} contains duplicate opaque keys")
    return values


@dataclass(frozen=True)
class WireProtocolSpec:
    """Fields committed before either consumed rehearsal beacon exists."""

    schema: str = "efc-deployed-wire-seal-rehearsal-v2"
    state_count: int = 5
    action_count: int = 3
    observer_count: int = 2
    answer_count: int = 5
    renderer_count: int = 1
    source_renderer_count: int = 2
    world_count: int = 1
    depth_quotas: tuple[tuple[int, int], ...] = (
        (0, 10),
        (1, 20),
        (3, 30),
        (6, 40),
    )
    duplicate_policy: str = "reject"
    world_generator: str = INDEPENDENT_GENERATOR_SCHEMA
    runtime_binary_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema != "efc-deployed-wire-seal-rehearsal-v2":
            raise ProtocolViolation("unknown deployed-wire protocol schema")
        if not 4 <= self.state_count <= MAX_STATES:
            raise ProtocolViolation("state count is outside deployed capacity")
        if not 2 <= self.action_count <= MAX_ACTIONS:
            raise ProtocolViolation("action count is outside deployed capacity")
        if not 1 <= self.observer_count <= MAX_OBSERVERS:
            raise ProtocolViolation("observer count is outside deployed capacity")
        if self.answer_count < self.state_count:
            raise ProtocolViolation("identity observer does not fit answer alphabet")
        if (
            self.renderer_count != 1
            or self.source_renderer_count != 2
            or self.world_count != 1
        ):
            raise ProtocolViolation(
                "rehearsal requires two source renderers, one query renderer, "
                "and one world"
            )
        if self.duplicate_policy != "reject":
            raise ProtocolViolation("only reject-duplicate custody is rehearsed")
        if self.world_generator not in {
            INDEPENDENT_GENERATOR_SCHEMA,
            "legacy-domain-stream-a-v1",
        }:
            raise ProtocolViolation("unknown world generator implementation")
        if self.runtime_binary_sha256:
            if tuple(name for name, _ in self.runtime_binary_sha256) != (
                "c",
                "rust",
            ):
                raise ProtocolViolation(
                    "runtime binary attestations must be ordered c/rust"
                )
            for _, digest in self.runtime_binary_sha256:
                if (
                    len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise ProtocolViolation(
                        "runtime binary attestation is not SHA-256 hex"
                    )
        depths = [depth for depth, _ in self.depth_quotas]
        if depths != sorted(set(depths)):
            raise ProtocolViolation("depth quotas must be sorted and unique")
        for depth, quota in self.depth_quotas:
            if depth < 0 or depth > MAX_WORD or quota <= 0:
                raise ProtocolViolation("depth quota is outside deployed support")
            support = (
                self.state_count
                * self.action_count**depth
                * self.observer_count
                * self.renderer_count
            )
            if quota > support:
                raise ProtocolViolation("depth quota exceeds coordinate support")

    def compact_spec(self) -> CompactProtocolSpec:
        """Return dimensions only for the query-free consumed world generator."""

        return CompactProtocolSpec(
            state_count=self.state_count,
            action_count=self.action_count,
            observer_count=self.observer_count,
            answer_count=self.answer_count,
            renderer_count=self.renderer_count,
            world_count=self.world_count,
            depth_quotas=self.depth_quotas,
            duplicate_policy=self.duplicate_policy,
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "action_count": self.action_count,
            "answer_count": self.answer_count,
            "c_runtime_source_sha256": _sha256_hex(
                C_RUNTIME_SOURCE.read_bytes()
            ),
            "depth_quotas": [list(row) for row in self.depth_quotas],
            "duplicate_policy": self.duplicate_policy,
            "machine_bytes": MACHINE_SIZE,
            "machine_format": "efc-deployed-little-endian-v1",
            "machine_hash_offset": MACHINE_HASH_OFFSET,
            "max_actions": MAX_ACTIONS,
            "max_observers": MAX_OBSERVERS,
            "max_states": MAX_STATES,
            "max_word": MAX_WORD,
            "observer_count": self.observer_count,
            "independent_generator_source_sha256": _sha256_hex(
                INDEPENDENT_GENERATOR_SOURCE.read_bytes()
            ),
            "query_header_bytes": QUERY_HEADER_SIZE,
            "query_record_bytes": QUERY_RECORD_SIZE,
            "renderer_count": self.renderer_count,
            "runtime_claim": (
                "independent-c-rust-byte-identical-cpu-rehearsal"
                if self.runtime_binary_sha256
                else "serialization-only-unattested"
            ),
            "runtime_binary_sha256": dict(self.runtime_binary_sha256),
            "runtime_binaries_attested": bool(self.runtime_binary_sha256),
            "runtime_build_flags": {
                "c": "-std=c11 -O2 -Wall -Wextra -Werror -pedantic",
                "rust": (
                    "--edition=2021 -C opt-level=2 -D warnings"
                ),
            },
            "rust_runtime_source_sha256": _sha256_hex(
                RUST_RUNTIME_SOURCE.read_bytes()
            ),
            "source_renderer_count": self.source_renderer_count,
            "source_renderer_source_sha256": _sha256_hex(
                SOURCE_RENDERER_SOURCE.read_bytes()
            ),
            "schema": self.schema,
            "state_count": self.state_count,
            "transcript_header_bytes": TRANSCRIPT_HEADER_SIZE,
            "transcript_record_bytes": TRANSCRIPT_RECORD_SIZE,
            "wire_compiler_and_assessor_source_sha256": _sha256_hex(
                Path(__file__).read_bytes()
            ),
            "world_count": self.world_count,
            "world_generator": self.world_generator,
        }


@dataclass(frozen=True)
class MachineTables:
    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    observer_keys: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TranscriptRecord:
    challenge_id: int
    final_state_key: int
    answer: int
    final_state_slot: int
    word_length: int


@dataclass(frozen=True)
class WireChallengeReceipt:
    beacon: dict[str, object]
    challenge_seed_commitment: str
    coordinate_root: str
    query_root: str
    prediction_root: str
    answer_root: str
    machine_root: str
    machine_sha_before: str
    machine_sha_after: str
    query_sha256: str
    transcript_sha256: str
    compile_count_before: int
    compile_count_after: int
    total_coordinates: int
    machine_seal_event: int
    challenge_seed_event: int
    coordinate_commit_event: int
    query_render_event: int
    prediction_seal_event: int
    answer_assessment_event: int

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_world_evidence(
    evidence: bytes,
    spec: WireProtocolSpec,
) -> MachineTables:
    source_renderer = 0
    try:
        row = json.loads(evidence)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if evidence.startswith((SOURCE_LINE_MAGIC + "\t").encode("ascii")):
            try:
                row = decode_line_events(evidence)
                source_renderer = 1
            except SourceRendererError as renderer_exc:
                raise ProtocolViolation(
                    "line-rendered world evidence is malformed"
                ) from renderer_exc
        else:
            raise ProtocolViolation(
                "world evidence is neither JSON nor line events"
            ) from exc
    else:
        if canonical_json_bytes(row) != evidence:
            raise ProtocolViolation("world evidence is not canonical JSON")
    if not isinstance(row, dict):
        raise ProtocolViolation("world evidence root must be an object")
    if row.get("schema") == "efc-raw-world-evidence-v2":
        if row.get("renderer_choice") != source_renderer:
            raise ProtocolViolation(
                "renderer tag does not match source serialization"
            )
        return _parse_raw_world_evidence(row, spec)
    if set(row) != WORLD_EVIDENCE_FIELDS:
        raise ProtocolViolation("world evidence schema contains drift or query taint")
    if row["schema"] != "efc-consumed-world-evidence-v1":
        raise ProtocolViolation("unknown world evidence schema")
    renderer_choice = _plain_int(row["renderer_choice"], "renderer choice")
    if renderer_choice != source_renderer:
        raise ProtocolViolation(
            "renderer tag does not match source serialization"
        )

    state_keys = _integer_vector(row, "state_keys", spec.state_count)
    action_keys = _integer_vector(row, "action_keys", spec.action_count)
    observer_keys = _integer_vector(
        row, "observer_keys", spec.observer_count
    )
    state_index = {key: slot for slot, key in enumerate(state_keys)}
    action_index = {key: slot for slot, key in enumerate(action_keys)}

    raw_demonstrations = row["demonstrations"]
    if not isinstance(raw_demonstrations, list):
        raise ProtocolViolation("demonstrations must be a list")
    transitions: list[list[int | None]] = [
        [None] * spec.state_count for _ in range(spec.action_count)
    ]
    for demonstration in raw_demonstrations:
        if not isinstance(demonstration, dict) or set(demonstration) != {
            "action_key",
            "source_key",
            "target_key",
        }:
            raise ProtocolViolation("malformed transition demonstration")
        action_key = _plain_int(
            demonstration["action_key"], "demonstration action_key"
        )
        source_key = _plain_int(
            demonstration["source_key"], "demonstration source_key"
        )
        target_key = _plain_int(
            demonstration["target_key"], "demonstration target_key"
        )
        try:
            action = action_index[action_key]
            source = state_index[source_key]
            target = state_index[target_key]
        except KeyError as exc:
            raise ProtocolViolation("demonstration contains an unknown key") from exc
        if transitions[action][source] is not None:
            raise ProtocolViolation("duplicate transition demonstration")
        transitions[action][source] = target
    if any(cell is None for relation in transitions for cell in relation):
        raise ProtocolViolation("world evidence omits a transition")

    raw_observer_rows = row["observer_rows"]
    if not isinstance(raw_observer_rows, list) or len(raw_observer_rows) != (
        spec.observer_count
    ):
        raise ProtocolViolation("observer rows have incorrect cardinality")
    observer_index = {key: slot for slot, key in enumerate(observer_keys)}
    observations: list[tuple[int, ...] | None] = [None] * spec.observer_count
    for observer_row in raw_observer_rows:
        if not isinstance(observer_row, dict) or set(observer_row) != {
            "answers",
            "observer_key",
        }:
            raise ProtocolViolation("malformed observer row")
        key = _plain_int(observer_row["observer_key"], "observer key")
        try:
            slot = observer_index[key]
        except KeyError as exc:
            raise ProtocolViolation("observer row contains an unknown key") from exc
        if observations[slot] is not None:
            raise ProtocolViolation("duplicate observer row")
        answers = observer_row["answers"]
        if not isinstance(answers, list) or len(answers) != spec.state_count:
            raise ProtocolViolation("observer answer row has incorrect width")
        checked = tuple(_plain_int(answer, "observer answer") for answer in answers)
        if any(answer < 0 or answer >= 1 << 64 for answer in checked):
            raise ProtocolViolation("observer answer does not fit uint64")
        if any(answer >= spec.answer_count for answer in checked):
            raise ProtocolViolation("observer answer is outside frozen alphabet")
        observations[slot] = checked
    if any(row_value is None for row_value in observations):
        raise ProtocolViolation("world evidence omits an observer")

    return MachineTables(
        state_keys=state_keys,
        action_keys=action_keys,
        observer_keys=observer_keys,
        transitions=tuple(
            tuple(int(cell) for cell in relation) for relation in transitions
        ),
        observations=tuple(
            row_value for row_value in observations if row_value is not None
        ),
    )


def _parse_raw_world_evidence(
    row: Mapping[str, object],
    spec: WireProtocolSpec,
) -> MachineTables:
    """Infer typed key classes and complete tables from shuffled event rows."""

    if set(row) != RAW_WORLD_EVIDENCE_FIELDS:
        raise ProtocolViolation("raw evidence schema contains drift or query taint")
    renderer_choice = _plain_int(row["renderer_choice"], "renderer choice")
    if renderer_choice not in range(spec.source_renderer_count):
        raise ProtocolViolation("source renderer choice is outside protocol")

    raw_demonstrations = row["demonstrations"]
    if not isinstance(raw_demonstrations, list):
        raise ProtocolViolation("demonstrations must be a list")
    parsed_demonstrations: list[tuple[int, int, int]] = []
    state_key_set: set[int] = set()
    action_key_set: set[int] = set()
    for event in raw_demonstrations:
        if not isinstance(event, dict) or set(event) != {
            "action_key",
            "source_key",
            "target_key",
        }:
            raise ProtocolViolation("malformed transition demonstration")
        action = _plain_int(event["action_key"], "demonstration action_key")
        source = _plain_int(event["source_key"], "demonstration source_key")
        target = _plain_int(event["target_key"], "demonstration target_key")
        if min(action, source, target) <= 0 or max(
            action, source, target
        ) >= 1 << 64:
            raise ProtocolViolation("demonstration key is not nonzero uint64")
        parsed_demonstrations.append((action, source, target))
        action_key_set.add(action)
        state_key_set.update((source, target))
    if (
        len(state_key_set) != spec.state_count
        or len(action_key_set) != spec.action_count
    ):
        raise ProtocolViolation("raw evidence inferred key counts are wrong")
    state_keys = tuple(sorted(state_key_set))
    action_keys = tuple(sorted(action_key_set))
    state_index = {key: slot for slot, key in enumerate(state_keys)}
    action_index = {key: slot for slot, key in enumerate(action_keys)}
    transitions: list[list[int | None]] = [
        [None] * spec.state_count for _ in range(spec.action_count)
    ]
    for action_key, source_key, target_key in parsed_demonstrations:
        action = action_index[action_key]
        source = state_index[source_key]
        target = state_index[target_key]
        if transitions[action][source] is not None:
            raise ProtocolViolation("duplicate transition demonstration")
        transitions[action][source] = target
    if any(cell is None for relation in transitions for cell in relation):
        raise ProtocolViolation("world evidence omits a transition")

    raw_observations = row["observations"]
    if not isinstance(raw_observations, list):
        raise ProtocolViolation("observations must be a list")
    parsed_observations: list[tuple[int, int, int]] = []
    observer_key_set: set[int] = set()
    for event in raw_observations:
        if not isinstance(event, dict) or set(event) != {
            "answer",
            "observer_key",
            "state_key",
        }:
            raise ProtocolViolation("malformed observation event")
        observer_key = _plain_int(event["observer_key"], "observer key")
        state_key = _plain_int(event["state_key"], "observation state key")
        answer = _plain_int(event["answer"], "observer answer")
        if observer_key <= 0 or observer_key >= 1 << 64:
            raise ProtocolViolation("observer key is not nonzero uint64")
        if state_key not in state_index:
            raise ProtocolViolation("observation contains an unknown state key")
        if answer < 0 or answer >= spec.answer_count:
            raise ProtocolViolation("observer answer is outside frozen alphabet")
        observer_key_set.add(observer_key)
        parsed_observations.append((observer_key, state_key, answer))
    if len(observer_key_set) != spec.observer_count:
        raise ProtocolViolation("raw evidence inferred observer count is wrong")
    observer_keys = tuple(sorted(observer_key_set))
    observer_index = {
        key: slot for slot, key in enumerate(observer_keys)
    }
    observations: list[list[int | None]] = [
        [None] * spec.state_count for _ in range(spec.observer_count)
    ]
    for observer_key, state_key, answer in parsed_observations:
        observer = observer_index[observer_key]
        state = state_index[state_key]
        if observations[observer][state] is not None:
            raise ProtocolViolation("duplicate observation event")
        observations[observer][state] = answer
    if any(cell is None for observer in observations for cell in observer):
        raise ProtocolViolation("world evidence omits an observation")
    return MachineTables(
        state_keys=state_keys,
        action_keys=action_keys,
        observer_keys=observer_keys,
        transitions=tuple(
            tuple(int(cell) for cell in relation) for relation in transitions
        ),
        observations=tuple(
            tuple(int(cell) for cell in observer)
            for observer in observations
        ),
    )


def encode_deployed_machine(
    evidence: bytes,
    spec: WireProtocolSpec,
) -> bytes:
    """Compile canonical public evidence directly into the deployed wire."""

    tables = _parse_world_evidence(evidence, spec)
    machine = bytearray(MACHINE_SIZE)
    machine[:8] = MACHINE_MAGIC
    _put_u32(machine, 8, FORMAT_VERSION)
    _put_u32(machine, 12, 64)
    _put_u32(machine, 16, MACHINE_SIZE)
    _put_u16(machine, 24, spec.state_count)
    _put_u16(machine, 26, spec.action_count)
    _put_u16(machine, 28, spec.observer_count)
    _put_u64(machine, 32, (1 << spec.state_count) - 1)
    _put_u64(machine, 40, (1 << spec.action_count) - 1)
    _put_u64(machine, 48, (1 << spec.observer_count) - 1)
    machine[56] = 0

    for slot, key in enumerate(tables.state_keys):
        _put_u64(machine, 64 + slot * 8, key)
    for slot, key in enumerate(tables.action_keys):
        _put_u64(machine, 192 + slot * 8, key)
    for slot, key in enumerate(tables.observer_keys):
        _put_u64(machine, 256 + slot * 8, key)
    for action, relation in enumerate(tables.transitions):
        for state, destination in enumerate(relation):
            machine[320 + action * MAX_STATES + state] = destination
    for observer, row in enumerate(tables.observations):
        for state, answer in enumerate(row):
            _put_u64(
                machine,
                448 + (observer * MAX_STATES + state) * 8,
                answer,
            )
    machine[MACHINE_HASH_OFFSET:] = sha256(
        machine[:MACHINE_HASH_OFFSET]
    ).digest()
    return bytes(machine)


def decode_deployed_machine(
    machine: bytes,
    spec: WireProtocolSpec,
) -> MachineTables:
    """Strict Python decoder used only to render post-seal opaque query keys."""

    if len(machine) != MACHINE_SIZE:
        raise ProtocolViolation("deployed machine length is not 1,536 bytes")
    if machine[:8] != MACHINE_MAGIC:
        raise ProtocolViolation("deployed machine magic is invalid")
    if (
        _read_u32(machine, 8) != FORMAT_VERSION
        or _read_u32(machine, 12) != 64
        or _read_u32(machine, 16) != MACHINE_SIZE
        or _read_u32(machine, 20) != 0
    ):
        raise ProtocolViolation("deployed machine header is invalid")
    if (
        _read_u16(machine, 24) != spec.state_count
        or _read_u16(machine, 26) != spec.action_count
        or _read_u16(machine, 28) != spec.observer_count
        or _read_u16(machine, 30) != 0
    ):
        raise ProtocolViolation("deployed machine counts disagree with protocol")
    if (
        _read_u64(machine, 32) != (1 << spec.state_count) - 1
        or _read_u64(machine, 40) != (1 << spec.action_count) - 1
        or _read_u64(machine, 48) != (1 << spec.observer_count) - 1
        or machine[56] != 0
        or any(machine[57:64])
        or any(machine[1472:1504])
    ):
        raise ProtocolViolation("deployed machine masks or padding are invalid")
    if machine[MACHINE_HASH_OFFSET:] != sha256(
        machine[:MACHINE_HASH_OFFSET]
    ).digest():
        raise ProtocolViolation("deployed machine hash is invalid")

    state_keys = tuple(
        _read_u64(machine, 64 + slot * 8)
        for slot in range(spec.state_count)
    )
    action_keys = tuple(
        _read_u64(machine, 192 + slot * 8)
        for slot in range(spec.action_count)
    )
    observer_keys = tuple(
        _read_u64(machine, 256 + slot * 8)
        for slot in range(spec.observer_count)
    )
    for values, name in (
        (state_keys, "state"),
        (action_keys, "action"),
        (observer_keys, "observer"),
    ):
        if any(value == 0 for value in values) or len(set(values)) != len(values):
            raise ProtocolViolation(f"deployed {name} keys are invalid")
    if any(machine[64 + spec.state_count * 8 : 192]):
        raise ProtocolViolation("state-key padding is nonzero")
    if any(machine[192 + spec.action_count * 8 : 256]):
        raise ProtocolViolation("action-key padding is nonzero")
    if any(machine[256 + spec.observer_count * 8 : 320]):
        raise ProtocolViolation("observer-key padding is nonzero")

    transitions = tuple(
        tuple(
            machine[320 + action * MAX_STATES + state]
            for state in range(spec.state_count)
        )
        for action in range(spec.action_count)
    )
    if any(
        destination >= spec.state_count
        for relation in transitions
        for destination in relation
    ):
        raise ProtocolViolation("deployed transition destination is inactive")
    for action in range(MAX_ACTIONS):
        for state in range(MAX_STATES):
            live = action < spec.action_count and state < spec.state_count
            if not live and machine[320 + action * MAX_STATES + state] != 0:
                raise ProtocolViolation("transition padding is nonzero")

    observations = tuple(
        tuple(
            _read_u64(
                machine,
                448 + (observer * MAX_STATES + state) * 8,
            )
            for state in range(spec.state_count)
        )
        for observer in range(spec.observer_count)
    )
    for observer in range(MAX_OBSERVERS):
        for state in range(MAX_STATES):
            live = observer < spec.observer_count and state < spec.state_count
            answer = _read_u64(
                machine,
                448 + (observer * MAX_STATES + state) * 8,
            )
            if not live and answer != 0:
                raise ProtocolViolation("observer padding is nonzero")
    return MachineTables(
        state_keys,
        action_keys,
        observer_keys,
        transitions,
        observations,
    )


def encode_query_panel(
    machine: bytes,
    spec: WireProtocolSpec,
    coordinates: Sequence[AbstractCoordinate],
) -> bytes:
    """Render committed abstract coordinates into the deployed query wire."""

    if not coordinates:
        raise ProtocolViolation("query panel must be nonempty")
    semantic_coordinates = {
        (
            coordinate.world,
            coordinate.start,
            coordinate.actions,
            coordinate.observer,
        )
        for coordinate in coordinates
    }
    if len(semantic_coordinates) != len(coordinates):
        raise ProtocolViolation("deployed query panel contains duplicates")
    tables = decode_deployed_machine(machine, spec)
    queries = bytearray(
        QUERY_HEADER_SIZE + len(coordinates) * QUERY_RECORD_SIZE + HASH_SIZE
    )
    queries[:8] = QUERY_MAGIC
    _put_u32(queries, 8, FORMAT_VERSION)
    _put_u32(queries, 12, QUERY_HEADER_SIZE)
    _put_u32(queries, 16, QUERY_RECORD_SIZE)
    _put_u32(queries, 20, len(coordinates))
    queries[24:56] = machine[MACHINE_HASH_OFFSET:]
    for index, coordinate in enumerate(coordinates):
        if (
            coordinate.world != 0
            or coordinate.start not in range(spec.state_count)
            or coordinate.observer not in range(spec.observer_count)
            or coordinate.renderer not in range(spec.renderer_count)
            or len(coordinate.actions) > MAX_WORD
            or any(
                action not in range(spec.action_count)
                for action in coordinate.actions
            )
        ):
            raise ProtocolViolation("abstract coordinate is outside protocol")
        offset = QUERY_HEADER_SIZE + index * QUERY_RECORD_SIZE
        _put_u64(queries, offset, index + 1)
        _put_u64(
            queries, offset + 8, tables.state_keys[coordinate.start]
        )
        _put_u64(
            queries,
            offset + 16,
            tables.observer_keys[coordinate.observer],
        )
        _put_u16(queries, offset + 24, len(coordinate.actions))
        for word_index, action in enumerate(coordinate.actions):
            _put_u64(
                queries,
                offset + 32 + word_index * 8,
                tables.action_keys[action],
            )
    queries[-HASH_SIZE:] = sha256(queries[:-HASH_SIZE]).digest()
    return bytes(queries)


def decode_transcript(
    transcript: bytes,
    machine: bytes,
    queries: bytes,
) -> tuple[TranscriptRecord, ...]:
    if len(transcript) < TRANSCRIPT_HEADER_SIZE + HASH_SIZE:
        raise ProtocolViolation("transcript is truncated")
    if transcript[:8] != TRANSCRIPT_MAGIC:
        raise ProtocolViolation("transcript magic is invalid")
    count = _read_u32(transcript, 20)
    expected = (
        TRANSCRIPT_HEADER_SIZE
        + count * TRANSCRIPT_RECORD_SIZE
        + HASH_SIZE
    )
    if (
        _read_u32(transcript, 8) != FORMAT_VERSION
        or _read_u32(transcript, 12) != TRANSCRIPT_HEADER_SIZE
        or _read_u32(transcript, 16) != TRANSCRIPT_RECORD_SIZE
        or len(transcript) != expected
    ):
        raise ProtocolViolation("transcript sizes are invalid")
    if transcript[24:56] != machine[MACHINE_HASH_OFFSET:]:
        raise ProtocolViolation("transcript machine binding is invalid")
    if transcript[56:88] != queries[-HASH_SIZE:] or any(
        transcript[88:96]
    ):
        raise ProtocolViolation("transcript query binding or padding is invalid")
    if transcript[-HASH_SIZE:] != sha256(
        transcript[:-HASH_SIZE]
    ).digest():
        raise ProtocolViolation("transcript hash is invalid")
    records: list[TranscriptRecord] = []
    for index in range(count):
        offset = TRANSCRIPT_HEADER_SIZE + index * TRANSCRIPT_RECORD_SIZE
        flags = _read_u16(transcript, offset + 26)
        trailing = _read_u16(transcript, offset + 30)
        if flags != 0 or trailing != 0:
            raise ProtocolViolation("transcript record flags are nonzero")
        records.append(
            TranscriptRecord(
                challenge_id=_read_u64(transcript, offset),
                final_state_key=_read_u64(transcript, offset + 8),
                answer=_read_u64(transcript, offset + 16),
                final_state_slot=_read_u16(transcript, offset + 24),
                word_length=_read_u16(transcript, offset + 28),
            )
        )
    return tuple(records)


def assess_by_relation_composition(
    mechanics: WorldMechanics,
    coordinates: Sequence[AbstractCoordinate],
) -> tuple[tuple[int, int], ...]:
    """Third assessor over latent relations, with no machine-wire parser."""

    state_count = len(mechanics.transition_relations[0])
    action_relations = tuple(
        frozenset(
            (source, destination)
            for source, destination in enumerate(transition)
        )
        for transition in mechanics.transition_relations
    )
    identity = frozenset((state, state) for state in range(state_count))
    results: list[tuple[int, int]] = []
    for coordinate in coordinates:
        relation = identity
        for action in coordinate.actions:
            right = action_relations[action]
            relation = frozenset(
                (source, destination)
                for source, middle in relation
                for right_source, destination in right
                if middle == right_source
            )
        destinations = tuple(
            destination
            for source, destination in relation
            if source == coordinate.start
        )
        if len(destinations) != 1:
            raise ProtocolViolation("third-assessor relation is not functional")
        final_state = destinations[0]
        results.append(
            (
                final_state,
                mechanics.observer_maps[coordinate.observer][final_state],
            )
        )
    return tuple(results)


def machine_byte_receipt(spec: WireProtocolSpec) -> dict[str, object]:
    """Account for every deployed machine byte and its source dependence."""

    labels: list[tuple[str, str]] = [
        ("zero-padding", "protocol-constant") for _ in range(MACHINE_SIZE)
    ]

    def mark(start: int, end: int, field: str, dependence: str) -> None:
        labels[start:end] = [(field, dependence)] * (end - start)

    mark(0, 24, "wire-header", "protocol-constant")
    mark(24, 30, "active-counts", "protocol-constant")
    mark(32, 56, "active-masks", "protocol-constant")
    mark(56, 57, "initial-state-slot", "protocol-constant")
    mark(
        64,
        64 + spec.state_count * 8,
        "active-state-keys",
        "world-source-direct",
    )
    mark(
        192,
        192 + spec.action_count * 8,
        "active-action-keys",
        "world-source-direct",
    )
    mark(
        256,
        256 + spec.observer_count * 8,
        "active-observer-keys",
        "world-source-direct",
    )
    for action in range(spec.action_count):
        start = 320 + action * MAX_STATES
        mark(
            start,
            start + spec.state_count,
            f"transition-row-{action}",
            "world-source-derived",
        )
    for observer in range(spec.observer_count):
        start = 448 + observer * MAX_STATES * 8
        mark(
            start,
            start + spec.state_count * 8,
            f"observer-row-{observer}",
            "world-source-direct",
        )
    mark(
        MACHINE_HASH_OFFSET,
        MACHINE_SIZE,
        "machine-sha256",
        "world-source-derived",
    )

    segments: list[dict[str, object]] = []
    segment_start = 0
    for index in range(1, MACHINE_SIZE + 1):
        if index == MACHINE_SIZE or labels[index] != labels[segment_start]:
            field, dependence = labels[segment_start]
            segments.append(
                {
                    "dependence": dependence,
                    "end_exclusive": index,
                    "field": field,
                    "length": index - segment_start,
                    "start": segment_start,
                }
            )
            segment_start = index
    source_dependent = sum(
        1
        for _, dependence in labels
        if dependence.startswith("world-source")
    )
    return {
        "accounted_bytes": sum(int(row["length"]) for row in segments),
        "machine_bytes": MACHINE_SIZE,
        "schema": "efc-deployed-machine-byte-receipt-v1",
        "segments": segments,
        "source_dependent_bytes_including_hash": source_dependent,
        "unaccounted_bytes": 0,
    }


def query_byte_receipt(
    coordinates: Sequence[AbstractCoordinate],
) -> dict[str, object]:
    source_key_bytes = sum(
        16 + len(coordinate.actions) * 8 for coordinate in coordinates
    )
    coordinate_bytes = len(coordinates) * (8 + 2)
    return {
        "coordinate_dependent_bytes": coordinate_bytes,
        "machine_dependent_bytes": 32 + source_key_bytes,
        "query_bytes": (
            QUERY_HEADER_SIZE
            + len(coordinates) * QUERY_RECORD_SIZE
            + HASH_SIZE
        ),
        "query_count": len(coordinates),
        "schema": "efc-deployed-query-byte-receipt-v1",
        "trailing_hash_bytes": HASH_SIZE,
        "zero_or_protocol_constant_bytes": (
            QUERY_HEADER_SIZE
            + len(coordinates) * QUERY_RECORD_SIZE
            + HASH_SIZE
            - coordinate_bytes
            - 32
            - source_key_bytes
            - HASH_SIZE
        ),
    }


class WireSealFirstRehearsal:
    """Filesystem-backed, consumed, deployed-wire two-beacon rehearsal."""

    def __init__(
        self,
        root: Path,
        spec: WireProtocolSpec = WireProtocolSpec(),
    ) -> None:
        self.root = root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ProtocolViolation("wire protocol root must start empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        protocol_payload = canonical_json_bytes(spec.canonical_dict())
        self._protocol_payload = protocol_payload
        self.protocol_root = _hex_commitment(
            PROTOCOL_DOMAIN, protocol_payload
        )
        _publish_immutable(self.root / "protocol.json", protocol_payload)
        _publish_immutable(
            self.root / "protocol_root.txt",
            (self.protocol_root + "\n").encode("ascii"),
        )
        _publish_immutable(
            self.root / "machine_byte_receipt.json",
            canonical_json_bytes(machine_byte_receipt(spec)),
        )
        self._events: list[dict[str, object]] = []
        self._world_beacon: Beacon | None = None
        self._fixture: WorldFixture | None = None
        self.world_root: str | None = None
        self.world_evidence_root: str | None = None
        self._latent_sha256: str | None = None
        self.machine_root: str | None = None
        self._machine_seal_event: int | None = None
        self._machine_sha256: str | None = None
        self._compile_count = 0
        self._source_deleted = False
        self._event_tip = bytes(HASH_SIZE).hex()
        self._record_event("protocol_committed", root=self.protocol_root)

    @property
    def source_path(self) -> Path:
        return self.root / "source" / "world_evidence.json"

    @property
    def latent_path(self) -> Path:
        return self.root / "assessor" / "latent_world.json"

    @property
    def machine_path(self) -> Path:
        return self.root / "sealed" / "machine.bin"

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def _record_event(self, event: str, **fields: object) -> int:
        event_id = len(self._events) + 1
        row = {
            "event": event,
            "event_id": event_id,
            "previous_event_sha256": self._event_tip,
            **fields,
        }
        payload = canonical_json_bytes(row)
        self._events.append(row)
        _publish_immutable(
            self.root / "events" / f"{event_id:06d}.json",
            payload,
        )
        self._event_tip = _sha256_hex(payload)
        return event_id

    def _verify_protocol_seal(self) -> None:
        if self.root.joinpath("protocol.json").read_bytes() != (
            self._protocol_payload
        ):
            raise ProtocolViolation("protocol bytes changed after commitment")
        expected_root = (self.protocol_root + "\n").encode("ascii")
        if self.root.joinpath("protocol_root.txt").read_bytes() != expected_root:
            raise ProtocolViolation("protocol root receipt changed after commitment")

    def supply_world_beacon(self, beacon: Beacon) -> WorldFixture:
        self._verify_protocol_seal()
        if self._fixture is not None or self._world_beacon is not None:
            raise ProtocolViolation("world beacon has already been consumed")
        if self.spec.world_generator == INDEPENDENT_GENERATOR_SCHEMA:
            independent = generate_independent_world(
                protocol_root=self.protocol_root,
                beacon_round=beacon.round,
                beacon_value=beacon.value,
                state_count=self.spec.state_count,
                action_count=self.spec.action_count,
                observer_count=self.spec.observer_count,
                answer_count=self.spec.answer_count,
                renderer_count=self.spec.renderer_count,
            )
            fixture = WorldFixture(
                mechanics=WorldMechanics(
                    independent.transitions, independent.observers
                ),
                evidence=independent.evidence,
                world_seed_commitment=(
                    independent.world_seed_commitment
                ),
                stream_commitments=independent.stream_commitments,
                admissibility_receipt={
                    **independent.admissibility_receipt,
                    "accepted_candidate": (
                        independent.accepted_candidate
                    ),
                },
            )
        else:
            fixture = generate_consumed_world_fixture(
                self.spec.compact_spec(),
                self.protocol_root,
                beacon,
            )
        self._fixture = fixture
        self._world_beacon = beacon
        latent_payload = canonical_json_bytes(
            fixture.mechanics.canonical_dict()
        )
        self.world_evidence_root = _hex_commitment(
            "EFC/deployed-wire-world-evidence-root/v1", fixture.evidence
        )
        self._latent_sha256 = _sha256_hex(latent_payload)
        self.world_root = _hex_commitment(
            "EFC/deployed-wire-world-root/v1",
            fixture.evidence,
            bytes.fromhex(self._latent_sha256),
        )
        _publish_immutable(self.source_path, fixture.evidence)
        _publish_immutable(self.latent_path, latent_payload)
        _publish_immutable(
            self.root / "world_receipt.json",
            canonical_json_bytes(
                {
                    "admissibility": fixture.admissibility_receipt,
                    "beacon": asdict(beacon),
                    "latent_sha256": self._latent_sha256,
                    "protocol_root": self.protocol_root,
                    "stream_commitments": dict(fixture.stream_commitments),
                    "world_evidence_root": self.world_evidence_root,
                    "world_root": self.world_root,
                    "world_seed_commitment": (
                        fixture.world_seed_commitment
                    ),
                }
            ),
        )
        self._record_event("world_sealed", root=self.world_root)
        return fixture

    def seal_machine(self, source_path: Path | None = None) -> str:
        self._verify_protocol_seal()
        if (
            self._fixture is None
            or self.world_root is None
            or self.world_evidence_root is None
        ):
            raise ProtocolViolation("world must be sealed before compilation")
        if self.machine_root is not None or self._compile_count != 0:
            raise ProtocolViolation("machine compilation is single-shot")
        source = (source_path or self.source_path).resolve()
        evidence = source.read_bytes()
        if _hex_commitment(
            "EFC/deployed-wire-world-evidence-root/v1", evidence
        ) != self.world_evidence_root:
            raise ProtocolViolation("compiler source does not match world root")
        machine = encode_deployed_machine(evidence, self.spec)
        self._compile_count += 1
        self.machine_root = _hex_commitment(
            MACHINE_DOMAIN,
            bytes.fromhex(self.protocol_root),
            bytes.fromhex(self.world_root),
            machine,
        )
        _publish_immutable(self.machine_path, machine)
        self._machine_sha256 = _sha256_hex(machine)
        _publish_immutable(
            self.root / "machine_receipt.json",
            canonical_json_bytes(
                {
                    "compile_count": self._compile_count,
                    "machine_bytes": len(machine),
                    "machine_format": "efc-deployed-little-endian-v1",
                    "machine_payload_sha256": (
                        machine[MACHINE_HASH_OFFSET:].hex()
                    ),
                    "machine_root": self.machine_root,
                    "machine_sha256": _sha256_hex(machine),
                    "protocol_root": self.protocol_root,
                    "source_sha256": _sha256_hex(evidence),
                    "world_evidence_root": self.world_evidence_root,
                    "world_root": self.world_root,
                }
            ),
        )
        self._machine_seal_event = self._record_event(
            "machine_sealed", root=self.machine_root
        )
        return self.machine_root

    def poison_and_delete_source(self) -> dict[str, object]:
        self._verify_protocol_seal()
        if self.machine_root is None:
            raise ProtocolViolation("machine must be sealed before source deletion")
        if self._source_deleted or not self.source_path.exists():
            raise ProtocolViolation("source has already been deleted")
        machine_sha_before = _sha256_hex(self.machine_path.read_bytes())
        if machine_sha_before != self._machine_sha256:
            raise ProtocolViolation("machine bytes changed after seal")
        _mutate_consumed_source_for_invariance_test(
            self.source_path,
            b'{"poison":"SOURCE_MUST_NOT_BE_READ_AFTER_SEAL"}\n',
        )
        poison_sha = _sha256_hex(self.source_path.read_bytes())
        self.source_path.unlink()
        descriptor = os.open(
            self.source_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._source_deleted = True
        receipt = {
            "machine_sha_after": _sha256_hex(self.machine_path.read_bytes()),
            "machine_sha_before": machine_sha_before,
            "poison_sha256": poison_sha,
            "source_deleted": not self.source_path.exists(),
        }
        if (
            receipt["machine_sha_before"] != receipt["machine_sha_after"]
            or not receipt["source_deleted"]
        ):
            raise ProtocolViolation("source deletion changed the sealed machine")
        _publish_immutable(
            self.root / "source_delete_receipt.json",
            canonical_json_bytes(receipt),
        )
        self._record_event("source_poisoned_and_deleted")
        return receipt

    def run_challenge(
        self,
        beacon: Beacon,
        runtimes: Mapping[str, Path],
    ) -> WireChallengeReceipt:
        self._verify_protocol_seal()
        if (
            self.machine_root is None
            or self.world_root is None
            or self._machine_seal_event is None
            or self._world_beacon is None
            or self._fixture is None
            or self._latent_sha256 is None
            or self._machine_sha256 is None
        ):
            raise ProtocolViolation("machine must be sealed before challenge")
        if not self._source_deleted or self.source_path.exists():
            raise ProtocolViolation("source must be deleted before challenge")
        if set(runtimes) != {"c", "rust"}:
            raise ProtocolViolation("both named independent runtimes are required")
        attested_binaries = dict(self.spec.runtime_binary_sha256)
        if set(attested_binaries) != {"c", "rust"}:
            raise ProtocolViolation(
                "runtime binaries were not frozen before protocol commitment"
            )
        for name in ("c", "rust"):
            if _sha256_hex(runtimes[name].resolve().read_bytes()) != (
                attested_binaries[name]
            ):
                raise ProtocolViolation(
                    f"{name} runtime binary differs from protocol attestation"
                )
        if _sha256_hex(self.latent_path.read_bytes()) != self._latent_sha256:
            raise ProtocolViolation("assessor latent differs from world seal")
        if beacon.round <= self._world_beacon.round:
            raise ProtocolViolation("challenge beacon is not strictly later")
        if beacon.value == self._world_beacon.value:
            raise ProtocolViolation("challenge beacon must differ from world")

        machine = self.machine_path.read_bytes()
        machine_sha_before = _sha256_hex(machine)
        if machine_sha_before != self._machine_sha256:
            raise ProtocolViolation("machine bytes changed after seal")
        compile_count_before = self._compile_count
        seed = derive_challenge_seed(
            self.protocol_root,
            self.world_root,
            self.machine_root,
            beacon,
        )
        seed_commitment = _hex_commitment(
            "EFC/deployed-wire-challenge-seed/v1", seed
        )
        challenge_dir = self.root / "challenges" / seed_commitment
        if challenge_dir.exists():
            raise ProtocolViolation("challenge beacon has already been consumed")
        challenge_seed_event = self._record_event(
            "challenge_seed_derived", commitment=seed_commitment
        )

        coordinates = generate_abstract_coordinates(
            seed, self.spec.compact_spec()
        )
        coordinate_payload = canonical_json_bytes(
            [coordinate.canonical_dict() for coordinate in coordinates]
        )
        coordinate_root = _hex_commitment(
            COORDINATE_DOMAIN, coordinate_payload
        )
        _publish_immutable(
            challenge_dir / "abstract_coordinates.json",
            coordinate_payload,
        )
        coordinate_commit_event = self._record_event(
            "abstract_coordinates_committed", root=coordinate_root
        )

        queries = encode_query_panel(machine, self.spec, coordinates)
        query_root = _hex_commitment(QUERY_DOMAIN, queries)
        _publish_immutable(challenge_dir / "queries.bin", queries)
        _publish_immutable(
            challenge_dir / "query_byte_receipt.json",
            canonical_json_bytes(query_byte_receipt(coordinates)),
        )
        query_render_event = self._record_event(
            "opaque_query_wire_rendered", root=query_root
        )

        transcripts: dict[str, bytes] = {}
        runtime_binary_hashes: dict[str, str] = {}
        with tempfile.TemporaryDirectory(
            dir=self.root, prefix=".runtime-staging."
        ) as staging_name:
            staging = Path(staging_name)
            for name in ("c", "rust"):
                runtime = runtimes[name].resolve()
                before_hash = _sha256_hex(runtime.read_bytes())
                runtime_binary_hashes[name] = before_hash
                output = staging / f"transcript.{name}.bin"
                completed = subprocess.run(
                    [
                        runtime,
                        self.machine_path,
                        challenge_dir / "queries.bin",
                        output,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                after_hash = _sha256_hex(runtime.read_bytes())
                if before_hash != after_hash:
                    raise ProtocolViolation(
                        f"{name} runtime changed during execution"
                    )
                if completed.returncode != 0:
                    raise ProtocolViolation(
                        f"{name} runtime rejected sealed input: "
                        f"{completed.stderr}"
                    )
                transcripts[name] = output.read_bytes()
        if transcripts["c"] != transcripts["rust"]:
            raise ProtocolViolation("independent runtime transcripts disagree")
        for name in ("c", "rust"):
            _publish_immutable(
                challenge_dir / f"transcript.{name}.bin",
                transcripts[name],
            )
        transcript = transcripts["c"]
        prediction_root = _hex_commitment(PREDICTION_DOMAIN, transcript)
        prediction_seal_event = self._record_event(
            "runtime_predictions_sealed", root=prediction_root
        )

        records = decode_transcript(transcript, machine, queries)
        expected = assess_by_relation_composition(
            self._fixture.mechanics, coordinates
        )
        if len(records) != len(coordinates):
            raise ProtocolViolation("runtime transcript count is wrong")
        for index, (record, coordinate, target) in enumerate(
            zip(records, coordinates, expected, strict=True)
        ):
            if (
                record.challenge_id != index + 1
                or record.final_state_slot != target[0]
                or record.answer != target[1]
                or record.word_length != len(coordinate.actions)
            ):
                raise ProtocolViolation(
                    "runtime transcript disagrees with third assessor"
                )
        answer_payload = canonical_json_bytes(
            [
                {"answer": answer, "final_state_slot": state}
                for state, answer in expected
            ]
        )
        answer_root = _hex_commitment(ANSWER_DOMAIN, answer_payload)
        _publish_immutable(
            challenge_dir / "assessor_answers.json", answer_payload
        )
        answer_assessment_event = self._record_event(
            "relation_answers_opened", root=answer_root
        )

        machine_sha_after = _sha256_hex(self.machine_path.read_bytes())
        if (
            machine_sha_before != machine_sha_after
            or compile_count_before != self._compile_count
        ):
            raise ProtocolViolation("challenge changed or recompiled machine")
        receipt = WireChallengeReceipt(
            beacon=asdict(beacon),
            challenge_seed_commitment=seed_commitment,
            coordinate_root=coordinate_root,
            query_root=query_root,
            prediction_root=prediction_root,
            answer_root=answer_root,
            machine_root=self.machine_root,
            machine_sha_before=machine_sha_before,
            machine_sha_after=machine_sha_after,
            query_sha256=_sha256_hex(queries),
            transcript_sha256=_sha256_hex(transcript),
            compile_count_before=compile_count_before,
            compile_count_after=self._compile_count,
            total_coordinates=len(coordinates),
            machine_seal_event=self._machine_seal_event,
            challenge_seed_event=challenge_seed_event,
            coordinate_commit_event=coordinate_commit_event,
            query_render_event=query_render_event,
            prediction_seal_event=prediction_seal_event,
            answer_assessment_event=answer_assessment_event,
        )
        _publish_immutable(
            challenge_dir / "receipt.json",
            canonical_json_bytes(
                {
                    **receipt.canonical_dict(),
                    "runtime_binary_sha256": runtime_binary_hashes,
                }
            ),
        )
        return receipt


__all__ = [
    "C_RUNTIME_SOURCE",
    "INDEPENDENT_GENERATOR_SOURCE",
    "MACHINE_HASH_OFFSET",
    "MACHINE_SIZE",
    "RUST_RUNTIME_SOURCE",
    "SOURCE_RENDERER_SOURCE",
    "TranscriptRecord",
    "WireChallengeReceipt",
    "WireProtocolSpec",
    "WireSealFirstRehearsal",
    "assess_by_relation_composition",
    "decode_deployed_machine",
    "decode_transcript",
    "encode_deployed_machine",
    "encode_query_panel",
    "machine_byte_receipt",
    "query_byte_receipt",
]
