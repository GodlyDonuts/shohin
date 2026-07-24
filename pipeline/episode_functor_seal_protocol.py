#!/usr/bin/env python3
"""CPU-only two-beacon seal-first protocol rehearsal for EPISODE EFC.

This module uses generated, consumed synthetic fixtures only. It does not
create an official board, fit a model, start a job, or authorize pretraining.

The temporal contract is:

1. commit a canonical protocol;
2. receive a world beacon and seal world evidence;
3. compile and seal one fixed-width machine artifact;
4. receive a strictly later challenge beacon;
5. commit abstract coordinates;
6. render opaque keys, execute, and independently assess.

World generation uses stateless, domain-separated SHA-256 streams. Challenge
coordinates never receive world mechanics, opaque keys, machine bytes,
predictions, or answers.

The compact big-endian machine below is a Python protocol-rehearsal format. It
is explicitly not the deployed C/Rust wire format and establishes no
end-to-end independent-runtime claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Mapping, Sequence


PROTOCOL_DOMAIN = "EFC/protocol-root/v1"
WORLD_DOMAIN = "EFC/world/v1"
CHALLENGE_DOMAIN = "EFC/challenge/v1"
MACHINE_DOMAIN = "EFC/machine-root/v1"
COORDINATE_DOMAIN = "EFC/coordinate-root/v1"
RENDER_DOMAIN = "EFC/render-root/v1"
PREDICTION_DOMAIN = "EFC/prediction-root/v1"
ANSWER_DOMAIN = "EFC/answer-root/v1"
MACHINE_MAGIC = b"EFC1"
MACHINE_FORMAT = "compact-big-endian-python-rehearsal-v1"
MACHINE_FORMAT_STATUS = "not-deployed-c-rust-wire-format"
RUNTIME_CLAIM = "none-protocol-rehearsal-only"
MACHINE_HEADER = struct.Struct(">4sBBBB")
WORLD_STREAM_LABELS = (
    "world/transitions",
    "world/state-nonces",
    "world/action-nonces",
    "world/observer-nonces",
    "world/observer-maps",
    "world/demonstration-order",
    "world/renderer-choices",
)
WORLD_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "state_keys",
        "action_keys",
        "observer_keys",
        "demonstrations",
        "observer_rows",
        "renderer_choice",
    }
)


class ProtocolViolation(ValueError):
    """A seal order, canonicalization, custody, or receipt invariant failed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole JSON encoding admitted by this rehearsal."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation("value is not canonical-JSON serializable") from exc
    return (text + "\n").encode("ascii")


def _commitment(domain: str, *parts: bytes) -> bytes:
    digest = sha256()
    encoded_domain = domain.encode("ascii")
    digest.update(len(encoded_domain).to_bytes(4, "big"))
    digest.update(encoded_domain)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def _hex_commitment(domain: str, *parts: bytes) -> str:
    return _commitment(domain, *parts).hex()


def _sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Atomically publish once; an existing destination is a custody failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProtocolViolation(
                f"immutable artifact already exists: {path}"
            ) from exc
        linked = True
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        if linked:
            _fsync_directory(path.parent)


def _replace_mutable_event_view(path: Path, payload: bytes) -> None:
    """Replace only the explicitly named, derived event-log view."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _mutate_consumed_source_for_invariance_test(
    path: Path,
    payload: bytes,
) -> None:
    """Intentionally replace only consumed synthetic source during a kill test."""

    if not path.exists():
        raise ProtocolViolation("consumed source is unavailable for mutation")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.poison.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_canonical_json(path: Path) -> object:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolViolation(f"{path} is not JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise ProtocolViolation(f"{path} is not canonical JSON")
    return value


@dataclass(frozen=True)
class Beacon:
    round: int
    value: str

    def __post_init__(self) -> None:
        if self.round < 0:
            raise ProtocolViolation("beacon round must be nonnegative")
        if not self.value:
            raise ProtocolViolation("beacon value must be nonempty")
        try:
            self.value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ProtocolViolation("beacon value must be ASCII") from exc

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(asdict(self))


@dataclass(frozen=True)
class AbstractCoordinate:
    world: int
    start: int
    actions: tuple[int, ...]
    observer: int
    renderer: int

    @property
    def depth(self) -> int:
        return len(self.actions)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "observer": self.observer,
            "renderer": self.renderer,
            "start": self.start,
            "world": self.world,
        }


@dataclass(frozen=True)
class ProtocolSpec:
    """Fields frozen before either beacon is supplied."""

    schema: str = "efc-seal-rehearsal-v1"
    state_count: int = 5
    action_count: int = 3
    observer_count: int = 2
    answer_count: int = 5
    renderer_count: int = 2
    world_count: int = 1
    depth_quotas: tuple[tuple[int, int], ...] = (
        (0, 10),
        (1, 20),
        (3, 30),
        (6, 40),
    )
    duplicate_policy: str = "reject"
    machine_format: str = MACHINE_FORMAT
    machine_format_status: str = MACHINE_FORMAT_STATUS
    runtime_claim: str = RUNTIME_CLAIM
    world_stream_labels: tuple[str, ...] = WORLD_STREAM_LABELS
    world_admissibility_fields: tuple[str, ...] = (
        "transition_completeness",
        "transition_bijection",
        "noncommutativity",
        "observer_shape",
        "observer_separation",
    )

    def __post_init__(self) -> None:
        if self.schema != "efc-seal-rehearsal-v1":
            raise ProtocolViolation("unknown protocol schema")
        dimensions = (
            self.state_count,
            self.action_count,
            self.observer_count,
            self.answer_count,
            self.renderer_count,
            self.world_count,
        )
        if any(value <= 0 or value > 255 for value in dimensions):
            raise ProtocolViolation("protocol dimensions must be in [1, 255]")
        if self.state_count < 4 or self.action_count < 2:
            raise ProtocolViolation("synthetic mechanics require K>=4 and M>=2")
        if self.answer_count < self.state_count:
            raise ProtocolViolation(
                "identity observer requires answer_count >= state_count"
            )
        if self.duplicate_policy != "reject":
            raise ProtocolViolation("only duplicate_policy='reject' is rehearsed")
        if self.machine_format != MACHINE_FORMAT:
            raise ProtocolViolation("unknown rehearsal machine format")
        if self.machine_format_status != MACHINE_FORMAT_STATUS:
            raise ProtocolViolation(
                "rehearsal machine was misrepresented as a deployed wire format"
            )
        if self.runtime_claim != RUNTIME_CLAIM:
            raise ProtocolViolation(
                "CPU rehearsal cannot make an end-to-end runtime claim"
            )
        depths = [depth for depth, _ in self.depth_quotas]
        if depths != sorted(set(depths)):
            raise ProtocolViolation("depth quotas must be unique and sorted")
        for depth, quota in self.depth_quotas:
            if depth < 0 or quota <= 0:
                raise ProtocolViolation("depth and quota must be nonnegative/positive")
            support = (
                self.world_count
                * self.state_count
                * (self.action_count**depth)
                * self.observer_count
                * self.renderer_count
            )
            if quota > support:
                raise ProtocolViolation(
                    f"depth {depth} quota {quota} exceeds support {support}"
                )
        if self.world_stream_labels != WORLD_STREAM_LABELS:
            raise ProtocolViolation("world stream domains are not the frozen set")
        if len(set(self.world_stream_labels)) != len(self.world_stream_labels):
            raise ProtocolViolation("world stream domains must be unique")
        if any(not label.startswith("world/") for label in self.world_stream_labels):
            raise ProtocolViolation("world stream domain escaped world namespace")

    @property
    def machine_bytes(self) -> int:
        key_bytes = 8 * (
            self.state_count + self.action_count + self.observer_count
        )
        table_bytes = (
            self.action_count * self.state_count
            + self.observer_count * self.state_count
        )
        return MACHINE_HEADER.size + key_bytes + table_bytes

    def canonical_dict(self) -> dict[str, object]:
        return {
            "action_count": self.action_count,
            "answer_count": self.answer_count,
            "depth_quotas": [list(row) for row in self.depth_quotas],
            "duplicate_policy": self.duplicate_policy,
            "machine_format": self.machine_format,
            "machine_format_status": self.machine_format_status,
            "machine_bytes": self.machine_bytes,
            "observer_count": self.observer_count,
            "renderer_count": self.renderer_count,
            "runtime_claim": self.runtime_claim,
            "schema": self.schema,
            "state_count": self.state_count,
            "world_admissibility_fields": list(
                self.world_admissibility_fields
            ),
            "world_count": self.world_count,
            "world_stream_labels": list(self.world_stream_labels),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ProtocolSpec":
        expected = {
            "action_count",
            "answer_count",
            "depth_quotas",
            "duplicate_policy",
            "machine_format",
            "machine_format_status",
            "machine_bytes",
            "observer_count",
            "renderer_count",
            "runtime_claim",
            "schema",
            "state_count",
            "world_admissibility_fields",
            "world_count",
            "world_stream_labels",
        }
        if set(row) != expected:
            raise ProtocolViolation("protocol fields differ from canonical schema")
        spec = cls(
            schema=_required_str(row, "schema"),
            state_count=_required_int(row, "state_count"),
            action_count=_required_int(row, "action_count"),
            observer_count=_required_int(row, "observer_count"),
            answer_count=_required_int(row, "answer_count"),
            renderer_count=_required_int(row, "renderer_count"),
            world_count=_required_int(row, "world_count"),
            depth_quotas=_required_pair_tuple(row, "depth_quotas"),
            duplicate_policy=_required_str(row, "duplicate_policy"),
            machine_format=_required_str(row, "machine_format"),
            machine_format_status=_required_str(
                row, "machine_format_status"
            ),
            runtime_claim=_required_str(row, "runtime_claim"),
            world_stream_labels=_required_str_tuple(row, "world_stream_labels"),
            world_admissibility_fields=_required_str_tuple(
                row, "world_admissibility_fields"
            ),
        )
        if _required_int(row, "machine_bytes") != spec.machine_bytes:
            raise ProtocolViolation("declared fixed machine byte count is false")
        return spec


@dataclass(frozen=True)
class WorldMechanics:
    """Assessor-only latent relations, with no query fields."""

    transition_relations: tuple[tuple[int, ...], ...]
    observer_maps: tuple[tuple[int, ...], ...]

    def canonical_dict(self) -> dict[str, object]:
        return {
            "observer_maps": [list(row) for row in self.observer_maps],
            "schema": "efc-latent-relations-v1",
            "transition_relations": [
                list(row) for row in self.transition_relations
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        spec: ProtocolSpec,
    ) -> "WorldMechanics":
        if set(row) != {
            "schema",
            "transition_relations",
            "observer_maps",
        }:
            raise ProtocolViolation(
                "latent world fields contain query taint or schema drift"
            )
        if row["schema"] != "efc-latent-relations-v1":
            raise ProtocolViolation("unknown latent world schema")
        transitions = _required_matrix(
            row,
            "transition_relations",
            spec.action_count,
            spec.state_count,
        )
        observers = _required_matrix(
            row,
            "observer_maps",
            spec.observer_count,
            spec.state_count,
        )
        return cls(transitions, observers)


@dataclass(frozen=True)
class WorldFixture:
    mechanics: WorldMechanics
    evidence: bytes
    world_seed_commitment: str
    stream_commitments: tuple[tuple[str, str], ...]
    admissibility_receipt: Mapping[str, object]


@dataclass(frozen=True)
class ChallengeReceipt:
    beacon: Mapping[str, object]
    challenge_seed_commitment: str
    machine_root: str
    world_root: str
    coordinate_root: str
    render_root: str
    prediction_root: str
    answer_root: str
    requested_depth_quotas: tuple[tuple[int, int], ...]
    realized_depth_counts: tuple[tuple[int, int], ...]
    duplicate_policy: str
    duplicate_count: int
    total_coordinates: int
    machine_sha_before: str
    machine_sha_after: str
    compile_count_before: int
    compile_count_after: int
    machine_seal_event: int
    challenge_seed_event: int
    coordinate_commit_event: int
    key_render_event: int
    prediction_seal_event: int
    answer_assessment_event: int

    def canonical_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["requested_depth_quotas"] = [
            list(item) for item in self.requested_depth_quotas
        ]
        row["realized_depth_counts"] = [
            list(item) for item in self.realized_depth_counts
        ]
        return row


@dataclass(frozen=True)
class SourceInvarianceReceipt:
    sealed_machine_sha_before: str
    sealed_machine_sha_after: str
    baseline_prediction_root: str
    poisoned_source_prediction_root: str
    deleted_source_prediction_root: str
    poison_written: bool
    source_deleted: bool
    invariant: bool


@dataclass(frozen=True)
class TranscriptAssessment:
    passed: bool
    checks: Mapping[str, bool]
    challenge_count: int
    independently_assessed_answers: int


class _HashStream:
    """Deterministic stream local to one frozen domain label."""

    def __init__(self, seed: bytes, label: str) -> None:
        self._seed = _commitment("EFC/stream/v1", seed, label.encode("ascii"))
        self._counter = 0
        self._buffer = bytearray()

    def take(self, count: int) -> bytes:
        if count < 0:
            raise ProtocolViolation("stream read size must be nonnegative")
        while len(self._buffer) < count:
            block = _commitment(
                "EFC/stream-block/v1",
                self._seed,
                self._counter.to_bytes(8, "big"),
            )
            self._counter += 1
            self._buffer.extend(block)
        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ProtocolViolation("random bound must be positive")
        ceiling = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = int.from_bytes(self.take(8), "big")
            if value < ceiling:
                return value % bound

    def shuffled(self, values: Sequence[int]) -> tuple[int, ...]:
        output = list(values)
        for index in range(len(output) - 1, 0, -1):
            other = self.below(index + 1)
            output[index], output[other] = output[other], output[index]
        return tuple(output)


def derive_world_seed(protocol_root: str, beacon: Beacon) -> bytes:
    return _commitment(
        WORLD_DOMAIN,
        bytes.fromhex(protocol_root),
        beacon.canonical_bytes(),
    )


def derive_world_stream_seed(world_seed: bytes, label: str) -> bytes:
    if label not in WORLD_STREAM_LABELS:
        raise ProtocolViolation("unfrozen world stream domain")
    return _commitment("EFC/world-stream/v1", world_seed, label.encode("ascii"))


def derive_challenge_seed(
    protocol_root: str,
    world_root: str,
    machine_root: str,
    beacon: Beacon,
) -> bytes:
    return _commitment(
        CHALLENGE_DOMAIN,
        bytes.fromhex(protocol_root),
        bytes.fromhex(world_root),
        bytes.fromhex(machine_root),
        beacon.canonical_bytes(),
    )


def _unique_u64(stream: _HashStream, count: int) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    while len(values) < count:
        value = int.from_bytes(stream.take(8), "big")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return tuple(values)


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[value] = index
    return tuple(inverse)


def _compose_maps(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Return right after left."""

    return tuple(right[left[state]] for state in range(len(left)))


def _canonical_generators(state_count: int, action_count: int) -> tuple[
    tuple[int, ...], ...
]:
    rotation = tuple((state + 1) % state_count for state in range(state_count))
    swap = tuple(
        1 if state == 0 else 0 if state == 1 else state
        for state in range(state_count)
    )
    generators = [rotation, swap]
    for action in range(2, action_count):
        shift = action % state_count
        if shift == 0:
            shift = 1
        generators.append(
            tuple((state + shift) % state_count for state in range(state_count))
        )
    return tuple(generators)


def _conjugate(
    mapping: Sequence[int],
    gauge: Sequence[int],
) -> tuple[int, ...]:
    inverse = _inverse_permutation(gauge)
    return tuple(gauge[mapping[inverse[state]]] for state in range(len(mapping)))


def generate_consumed_world_fixture(
    spec: ProtocolSpec,
    protocol_root: str,
    beacon: Beacon,
) -> WorldFixture:
    """Generate one synthetic world without constructing any query."""

    world_seed = derive_world_seed(protocol_root, beacon)
    streams = {
        label: _HashStream(derive_world_stream_seed(world_seed, label), label)
        for label in spec.world_stream_labels
    }
    gauge = streams["world/transitions"].shuffled(range(spec.state_count))
    transitions = tuple(
        _conjugate(generator, gauge)
        for generator in _canonical_generators(
            spec.state_count,
            spec.action_count,
        )
    )
    state_keys = _unique_u64(
        streams["world/state-nonces"], spec.state_count
    )
    action_keys = _unique_u64(
        streams["world/action-nonces"], spec.action_count
    )
    observer_keys = _unique_u64(
        streams["world/observer-nonces"], spec.observer_count
    )
    observer_maps: list[tuple[int, ...]] = [
        tuple(range(spec.state_count))
    ]
    for _ in range(1, spec.observer_count):
        observer_maps.append(
            streams["world/observer-maps"].shuffled(range(spec.state_count))
        )
    mechanics = WorldMechanics(transitions, tuple(observer_maps))
    admissibility = assess_world_admissibility(mechanics, spec)

    demonstrations = [
        {
            "action_key": action_keys[action],
            "source_key": state_keys[state],
            "target_key": state_keys[transitions[action][state]],
        }
        for action in range(spec.action_count)
        for state in range(spec.state_count)
    ]
    order = streams["world/demonstration-order"].shuffled(
        range(len(demonstrations))
    )
    evidence_row = {
        "action_keys": list(action_keys),
        "demonstrations": [demonstrations[index] for index in order],
        "observer_keys": list(observer_keys),
        "observer_rows": [
            {
                "answers": list(observer_maps[index]),
                "observer_key": observer_keys[index],
            }
            for index in range(spec.observer_count)
        ],
        "renderer_choice": streams["world/renderer-choices"].below(
            spec.renderer_count
        ),
        "schema": "efc-consumed-world-evidence-v1",
        "state_keys": list(state_keys),
    }
    stream_commitments = tuple(
        (
            label,
            _hex_commitment(
                "EFC/world-stream-commitment/v1",
                derive_world_stream_seed(world_seed, label),
            ),
        )
        for label in spec.world_stream_labels
    )
    return WorldFixture(
        mechanics=mechanics,
        evidence=canonical_json_bytes(evidence_row),
        world_seed_commitment=_hex_commitment(
            "EFC/world-seed-commitment/v1", world_seed
        ),
        stream_commitments=stream_commitments,
        admissibility_receipt=admissibility,
    )


def assess_world_admissibility(
    mechanics: WorldMechanics,
    spec: ProtocolSpec,
) -> dict[str, object]:
    """Inspect world relations only; this API has no query argument."""

    transitions = mechanics.transition_relations
    observers = mechanics.observer_maps
    complete = (
        len(transitions) == spec.action_count
        and all(len(row) == spec.state_count for row in transitions)
        and all(
            destination in range(spec.state_count)
            for row in transitions
            for destination in row
        )
    )
    bijective = complete and all(
        sorted(row) == list(range(spec.state_count)) for row in transitions
    )
    noncommuting = False
    if complete:
        for left in range(spec.action_count):
            for right in range(left + 1, spec.action_count):
                if _compose_maps(transitions[left], transitions[right]) != (
                    _compose_maps(transitions[right], transitions[left])
                ):
                    noncommuting = True
                    break
            if noncommuting:
                break
    observer_shape = (
        len(observers) == spec.observer_count
        and all(len(row) == spec.state_count for row in observers)
        and all(
            answer in range(spec.answer_count)
            for row in observers
            for answer in row
        )
    )
    observer_separation = observer_shape and len(
        {
            tuple(observer[state] for observer in observers)
            for state in range(spec.state_count)
        }
    ) == spec.state_count
    checks = {
        "noncommutativity": noncommuting,
        "observer_separation": observer_separation,
        "observer_shape": observer_shape,
        "transition_bijection": bijective,
        "transition_completeness": complete,
    }
    if tuple(sorted(checks)) != tuple(sorted(spec.world_admissibility_fields)):
        raise ProtocolViolation("admissibility implementation/schema mismatch")
    if not all(checks.values()):
        raise ProtocolViolation("synthetic world failed mechanics-only admission")
    return {
        "admitted": True,
        "checks": checks,
        "inspected_fields": list(spec.world_admissibility_fields),
        "query_fields_seen": 0,
    }


def compile_world_evidence(
    evidence: bytes,
    spec: ProtocolSpec,
) -> bytes:
    """Compile public evidence into the fixed-width machine format."""

    try:
        row = json.loads(evidence)
    except json.JSONDecodeError as exc:
        raise ProtocolViolation("world evidence is not JSON") from exc
    if canonical_json_bytes(row) != evidence:
        raise ProtocolViolation("world evidence is not canonical JSON")
    if not isinstance(row, dict) or set(row) != WORLD_EVIDENCE_FIELDS:
        raise ProtocolViolation(
            "world evidence fields contain query taint or schema drift"
        )
    if row["schema"] != "efc-consumed-world-evidence-v1":
        raise ProtocolViolation("unknown world evidence schema")

    state_keys = _required_int_vector(row, "state_keys", spec.state_count)
    action_keys = _required_int_vector(row, "action_keys", spec.action_count)
    observer_keys = _required_int_vector(
        row, "observer_keys", spec.observer_count
    )
    if any(len(set(keys)) != len(keys) for keys in (
        state_keys,
        action_keys,
        observer_keys,
    )):
        raise ProtocolViolation("opaque key classes must be internally unique")
    state_index = {key: index for index, key in enumerate(state_keys)}
    action_index = {key: index for index, key in enumerate(action_keys)}

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
            raise ProtocolViolation("malformed world demonstration")
        try:
            action = action_index[_plain_int(demonstration["action_key"])]
            source = state_index[_plain_int(demonstration["source_key"])]
            target = state_index[_plain_int(demonstration["target_key"])]
        except KeyError as exc:
            raise ProtocolViolation("demonstration uses an unknown key") from exc
        if transitions[action][source] is not None:
            raise ProtocolViolation("duplicate transition demonstration")
        transitions[action][source] = target
    if any(value is None for row_values in transitions for value in row_values):
        raise ProtocolViolation("world evidence does not define every transition")

    raw_observers = row["observer_rows"]
    if not isinstance(raw_observers, list) or len(raw_observers) != (
        spec.observer_count
    ):
        raise ProtocolViolation("observer rows have incorrect cardinality")
    observers: list[tuple[int, ...] | None] = [None] * spec.observer_count
    observer_index = {key: index for index, key in enumerate(observer_keys)}
    for observer_row in raw_observers:
        if not isinstance(observer_row, dict) or set(observer_row) != {
            "answers",
            "observer_key",
        }:
            raise ProtocolViolation("malformed observer row")
        try:
            index = observer_index[_plain_int(observer_row["observer_key"])]
        except KeyError as exc:
            raise ProtocolViolation("observer row uses an unknown key") from exc
        if observers[index] is not None:
            raise ProtocolViolation("duplicate observer row")
        answers = observer_row["answers"]
        if not isinstance(answers, list) or len(answers) != spec.state_count:
            raise ProtocolViolation("observer answer row has incorrect width")
        checked_answers = tuple(_plain_int(answer) for answer in answers)
        if any(answer not in range(spec.answer_count) for answer in checked_answers):
            raise ProtocolViolation("observer answer is outside alphabet")
        observers[index] = checked_answers
    if any(row_values is None for row_values in observers):
        raise ProtocolViolation("world evidence omits an observer")

    machine = bytearray(
        MACHINE_HEADER.pack(
            MACHINE_MAGIC,
            spec.state_count,
            spec.action_count,
            spec.observer_count,
            spec.answer_count,
        )
    )
    for key in (*state_keys, *action_keys, *observer_keys):
        if key < 0 or key >= 1 << 64:
            raise ProtocolViolation("opaque key does not fit uint64")
        machine.extend(struct.pack(">Q", key))
    machine.extend(
        bytes(
            int(destination)
            for row_values in transitions
            for destination in row_values
        )
    )
    machine.extend(
        bytes(
            int(answer)
            for row_values in observers
            if row_values is not None
            for answer in row_values
        )
    )
    if len(machine) != spec.machine_bytes:
        raise ProtocolViolation("compiler emitted the wrong fixed byte count")
    return bytes(machine)


@dataclass(frozen=True)
class _DecodedMachine:
    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    observer_keys: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    observers: tuple[tuple[int, ...], ...]


def _decode_machine(machine: bytes, spec: ProtocolSpec) -> _DecodedMachine:
    """Decode the Python rehearsal format, not a deployed C/Rust wire format."""

    if len(machine) != spec.machine_bytes:
        raise ProtocolViolation("sealed machine has incorrect fixed length")
    magic, states, actions, observers, answers = MACHINE_HEADER.unpack_from(machine)
    if (
        magic != MACHINE_MAGIC
        or states != spec.state_count
        or actions != spec.action_count
        or observers != spec.observer_count
        or answers != spec.answer_count
    ):
        raise ProtocolViolation("sealed machine header disagrees with protocol")
    cursor = MACHINE_HEADER.size

    def read_keys(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(
            struct.unpack_from(">Q", machine, cursor + 8 * index)[0]
            for index in range(count)
        )
        cursor += 8 * count
        if len(set(values)) != len(values):
            raise ProtocolViolation("sealed machine contains duplicate opaque keys")
        return values

    state_keys = read_keys(spec.state_count)
    action_keys = read_keys(spec.action_count)
    observer_keys = read_keys(spec.observer_count)
    transitions = tuple(
        tuple(machine[cursor + action * spec.state_count + state]
              for state in range(spec.state_count))
        for action in range(spec.action_count)
    )
    cursor += spec.action_count * spec.state_count
    observer_rows = tuple(
        tuple(machine[cursor + observer * spec.state_count + state]
              for state in range(spec.state_count))
        for observer in range(spec.observer_count)
    )
    cursor += spec.observer_count * spec.state_count
    if cursor != len(machine):
        raise ProtocolViolation("sealed machine has undeclared trailing bytes")
    if any(
        destination not in range(spec.state_count)
        for row in transitions
        for destination in row
    ):
        raise ProtocolViolation("sealed transition is outside state space")
    if any(
        answer not in range(spec.answer_count)
        for row in observer_rows
        for answer in row
    ):
        raise ProtocolViolation("sealed observer is outside answer alphabet")
    return _DecodedMachine(
        state_keys,
        action_keys,
        observer_keys,
        transitions,
        observer_rows,
    )


def execute_sealed_machine(
    machine: bytes,
    spec: ProtocolSpec,
    coordinate: AbstractCoordinate,
) -> int:
    """Production-style table executor used only by the synthetic rehearsal."""

    decoded = _decode_machine(machine, spec)
    _validate_coordinate(coordinate, spec)
    state = coordinate.start
    for action in coordinate.actions:
        state = decoded.transitions[action][state]
    return decoded.observers[coordinate.observer][state]


def assess_by_relation_composition(
    mechanics: WorldMechanics,
    spec: ProtocolSpec,
    coordinates: Sequence[AbstractCoordinate],
) -> tuple[int, ...]:
    """Third assessor using boolean relation composition and enumeration.

    This function deliberately does not decode machine bytes and does not call
    ``execute_sealed_machine``. Each action is expanded to a boolean relation;
    words are composed by exhaustive three-loop relational multiplication.
    """

    assess_world_admissibility(mechanics, spec)
    action_relations = tuple(
        tuple(
            tuple(
                destination == mechanics.transition_relations[action][source]
                for destination in range(spec.state_count)
            )
            for source in range(spec.state_count)
        )
        for action in range(spec.action_count)
    )
    identity = tuple(
        tuple(left == right for right in range(spec.state_count))
        for left in range(spec.state_count)
    )
    answers: list[int] = []
    for coordinate in coordinates:
        _validate_coordinate(coordinate, spec)
        relation = identity
        for action in coordinate.actions:
            right = action_relations[action]
            relation = tuple(
                tuple(
                    any(
                        relation[source][middle] and right[middle][destination]
                        for middle in range(spec.state_count)
                    )
                    for destination in range(spec.state_count)
                )
                for source in range(spec.state_count)
            )
        destinations = [
            state
            for state, active in enumerate(relation[coordinate.start])
            if active
        ]
        if len(destinations) != 1:
            raise ProtocolViolation("latent relation is not deterministic and total")
        answers.append(
            mechanics.observer_maps[coordinate.observer][destinations[0]]
        )
    return tuple(answers)


def generate_abstract_coordinates(
    seed: bytes,
    spec: ProtocolSpec,
) -> tuple[AbstractCoordinate, ...]:
    """Hash-rank abstract indices without inspecting a world or machine."""

    selected: list[AbstractCoordinate] = []
    for depth, quota in spec.depth_quotas:
        candidates = (
            AbstractCoordinate(world, start, tuple(actions), observer, renderer)
            for world in range(spec.world_count)
            for start in range(spec.state_count)
            for actions in product(range(spec.action_count), repeat=depth)
            for observer in range(spec.observer_count)
            for renderer in range(spec.renderer_count)
        )
        ranked = sorted(
            candidates,
            key=lambda coordinate: _commitment(
                "EFC/coordinate-rank/v1",
                seed,
                canonical_json_bytes(coordinate.canonical_dict()),
            ),
        )
        selected.extend(ranked[:quota])
    coordinate_keys = [
        canonical_json_bytes(coordinate.canonical_dict())
        for coordinate in selected
    ]
    if len(set(coordinate_keys)) != len(coordinate_keys):
        raise ProtocolViolation("abstract coordinate generator produced duplicates")
    return tuple(selected)


def render_coordinates(
    machine: bytes,
    spec: ProtocolSpec,
    coordinates: Sequence[AbstractCoordinate],
) -> tuple[dict[str, object], ...]:
    decoded = _decode_machine(machine, spec)
    rendered: list[dict[str, object]] = []
    for coordinate in coordinates:
        _validate_coordinate(coordinate, spec)
        if coordinate.renderer == 0:
            row = {
                "action_keys": [
                    decoded.action_keys[action] for action in coordinate.actions
                ],
                "observer_key": decoded.observer_keys[coordinate.observer],
                "renderer": "path-list",
                "start_key": decoded.state_keys[coordinate.start],
                "world": coordinate.world,
            }
        else:
            row = {
                "observer_key": decoded.observer_keys[coordinate.observer],
                "renderer": "edge-sequence",
                "sequence": [
                    decoded.state_keys[coordinate.start],
                    *(
                        decoded.action_keys[action]
                        for action in coordinate.actions
                    ),
                ],
                "world": coordinate.world,
            }
        rendered.append(row)
    return tuple(rendered)


class SealFirstRehearsal:
    """Filesystem-backed consumed-fixture rehearsal with strict phase guards."""

    def __init__(self, root: Path, spec: ProtocolSpec = ProtocolSpec()) -> None:
        self.root = root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ProtocolViolation("canonical protocol root must start empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        protocol_payload = canonical_json_bytes(spec.canonical_dict())
        self.protocol_root = _hex_commitment(PROTOCOL_DOMAIN, protocol_payload)
        _publish_immutable(self.root / "protocol.json", protocol_payload)
        _publish_immutable(
            self.root / "protocol_root.txt",
            (self.protocol_root + "\n").encode("ascii"),
        )
        self._events: list[dict[str, object]] = []
        self._world_beacon: Beacon | None = None
        self._world_fixture: WorldFixture | None = None
        self.world_root: str | None = None
        self.machine_root: str | None = None
        self._machine_seal_event: int | None = None
        self._compile_count = 0
        self._record_event("protocol_committed", root=self.protocol_root)

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def machine_path(self) -> Path:
        return self.root / "sealed" / "machine.bin"

    @property
    def source_path(self) -> Path:
        return self.root / "source" / "world_evidence.json"

    @property
    def latent_path(self) -> Path:
        return self.root / "assessor" / "latent_world.json"

    def supply_world_beacon(self, beacon: Beacon) -> WorldFixture:
        if self._world_beacon is not None or self._world_fixture is not None:
            raise ProtocolViolation("world beacon/world are already sealed")
        fixture = generate_consumed_world_fixture(
            self.spec,
            self.protocol_root,
            beacon,
        )
        self._world_beacon = beacon
        self._world_fixture = fixture
        self.world_root = _hex_commitment(
            "EFC/world-root/v1", fixture.evidence
        )
        _publish_immutable(self.source_path, fixture.evidence)
        _publish_immutable(
            self.latent_path,
            canonical_json_bytes(fixture.mechanics.canonical_dict()),
        )
        receipt = {
            "admissibility": fixture.admissibility_receipt,
            "beacon": asdict(beacon),
            "protocol_root": self.protocol_root,
            "stream_commitments": dict(fixture.stream_commitments),
            "world_root": self.world_root,
            "world_seed_commitment": fixture.world_seed_commitment,
        }
        _publish_immutable(
            self.root / "world_receipt.json",
            canonical_json_bytes(receipt),
        )
        self._record_event("world_sealed", root=self.world_root)
        return fixture

    def seal_machine(self, source_path: Path | None = None) -> str:
        if self._world_fixture is None or self.world_root is None:
            raise ProtocolViolation("world must be sealed before machine compilation")
        if self.machine_root is not None or self._compile_count != 0:
            raise ProtocolViolation("machine compilation is single-shot")
        source = (source_path or self.source_path).resolve()
        evidence = source.read_bytes()
        if _hex_commitment("EFC/world-root/v1", evidence) != self.world_root:
            raise ProtocolViolation("compiler source does not match sealed world root")
        machine = compile_world_evidence(evidence, self.spec)
        self._compile_count += 1
        machine_sha = _sha256_hex(machine)
        self.machine_root = _hex_commitment(
            MACHINE_DOMAIN,
            bytes.fromhex(self.protocol_root),
            bytes.fromhex(self.world_root),
            machine,
        )
        _publish_immutable(self.machine_path, machine)
        _publish_immutable(
            self.root / "sealed" / "world_evidence.copy",
            evidence,
        )
        receipt = {
            "compile_count": self._compile_count,
            "machine_format": self.spec.machine_format,
            "machine_format_status": self.spec.machine_format_status,
            "machine_bytes": len(machine),
            "machine_root": self.machine_root,
            "machine_sha256": machine_sha,
            "protocol_root": self.protocol_root,
            "runtime_claim": self.spec.runtime_claim,
            "source_sha256": _sha256_hex(evidence),
            "world_root": self.world_root,
        }
        _publish_immutable(
            self.root / "machine_receipt.json",
            canonical_json_bytes(receipt),
        )
        self._machine_seal_event = self._record_event(
            "machine_sealed", root=self.machine_root
        )
        return self.machine_root

    def run_challenge(self, beacon: Beacon) -> ChallengeReceipt:
        if (
            self.machine_root is None
            or self.world_root is None
            or self._world_beacon is None
            or self._machine_seal_event is None
        ):
            raise ProtocolViolation(
                "machine root must be sealed before challenge seed derivation"
            )
        if beacon.round <= self._world_beacon.round:
            raise ProtocolViolation(
                "challenge beacon must be strictly later than world beacon"
            )
        if beacon.value == self._world_beacon.value:
            raise ProtocolViolation("world and challenge beacons must be distinct")
        machine_before = self.machine_path.read_bytes()
        machine_sha_before = _sha256_hex(machine_before)
        compile_count_before = self._compile_count
        seed = derive_challenge_seed(
            self.protocol_root,
            self.world_root,
            self.machine_root,
            beacon,
        )
        seed_commitment = _hex_commitment(
            "EFC/challenge-seed-commitment/v1", seed
        )
        challenge_dir = self.root / "challenges" / seed_commitment
        if challenge_dir.exists():
            raise ProtocolViolation("challenge beacon has already been consumed")
        challenge_seed_event = self._record_event(
            "challenge_seed_derived",
            commitment=seed_commitment,
        )

        coordinates = generate_abstract_coordinates(seed, self.spec)
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
            "abstract_coordinates_committed",
            root=coordinate_root,
        )

        rendered = render_coordinates(machine_before, self.spec, coordinates)
        render_payload = canonical_json_bytes(list(rendered))
        render_root = _hex_commitment(RENDER_DOMAIN, render_payload)
        _publish_immutable(
            challenge_dir / "rendered_queries.json",
            render_payload,
        )
        key_render_event = self._record_event(
            "opaque_keys_rendered",
            root=render_root,
        )

        predictions = tuple(
            execute_sealed_machine(machine_before, self.spec, coordinate)
            for coordinate in coordinates
        )
        prediction_payload = canonical_json_bytes(list(predictions))
        prediction_root = _hex_commitment(
            PREDICTION_DOMAIN, prediction_payload
        )
        _publish_immutable(
            challenge_dir / "predictions.json",
            prediction_payload,
        )
        prediction_seal_event = self._record_event(
            "predictions_sealed",
            root=prediction_root,
        )

        if self._world_fixture is None:
            raise ProtocolViolation("assessor fixture unexpectedly unavailable")
        answers = assess_by_relation_composition(
            self._world_fixture.mechanics,
            self.spec,
            coordinates,
        )
        answer_payload = canonical_json_bytes(list(answers))
        answer_root = _hex_commitment(ANSWER_DOMAIN, answer_payload)
        _publish_immutable(
            challenge_dir / "assessor_answers.json",
            answer_payload,
        )
        answer_assessment_event = self._record_event(
            "answers_independently_assessed",
            root=answer_root,
        )
        if predictions != answers:
            raise ProtocolViolation(
                "sealed executor and relation-composition assessor disagree"
            )

        machine_after = self.machine_path.read_bytes()
        machine_sha_after = _sha256_hex(machine_after)
        depth_counts = tuple(
            (
                depth,
                sum(coordinate.depth == depth for coordinate in coordinates),
            )
            for depth, _ in self.spec.depth_quotas
        )
        duplicate_count = len(coordinates) - len(
            {
                canonical_json_bytes(coordinate.canonical_dict())
                for coordinate in coordinates
            }
        )
        receipt = ChallengeReceipt(
            beacon=asdict(beacon),
            challenge_seed_commitment=seed_commitment,
            machine_root=self.machine_root,
            world_root=self.world_root,
            coordinate_root=coordinate_root,
            render_root=render_root,
            prediction_root=prediction_root,
            answer_root=answer_root,
            requested_depth_quotas=self.spec.depth_quotas,
            realized_depth_counts=depth_counts,
            duplicate_policy=self.spec.duplicate_policy,
            duplicate_count=duplicate_count,
            total_coordinates=len(coordinates),
            machine_sha_before=machine_sha_before,
            machine_sha_after=machine_sha_after,
            compile_count_before=compile_count_before,
            compile_count_after=self._compile_count,
            machine_seal_event=self._machine_seal_event,
            challenge_seed_event=challenge_seed_event,
            coordinate_commit_event=coordinate_commit_event,
            key_render_event=key_render_event,
            prediction_seal_event=prediction_seal_event,
            answer_assessment_event=answer_assessment_event,
        )
        validate_challenge_receipt(receipt, self.spec)
        _publish_immutable(
            challenge_dir / "receipt.json",
            canonical_json_bytes(receipt.canonical_dict()),
        )
        return receipt

    def prove_source_poison_delete_invariance(
        self,
        coordinates: Sequence[AbstractCoordinate],
    ) -> SourceInvarianceReceipt:
        """Poison then delete consumed source while executing sealed bytes."""

        if self.machine_root is None:
            raise ProtocolViolation("machine must be sealed before source mutation")
        machine_before = self.machine_path.read_bytes()
        machine_sha_before = _sha256_hex(machine_before)

        def prediction_root() -> str:
            machine = self.machine_path.read_bytes()
            predictions = [
                execute_sealed_machine(machine, self.spec, coordinate)
                for coordinate in coordinates
            ]
            return _hex_commitment(
                PREDICTION_DOMAIN,
                canonical_json_bytes(predictions),
            )

        baseline = prediction_root()
        _mutate_consumed_source_for_invariance_test(
            self.source_path,
            b'{"query":"POISONED_AFTER_MACHINE_SEAL"}\n',
        )
        poisoned = prediction_root()
        self.source_path.unlink()
        deleted = prediction_root()
        machine_sha_after = _sha256_hex(self.machine_path.read_bytes())
        invariant = (
            baseline == poisoned == deleted
            and machine_sha_before == machine_sha_after
        )
        receipt = SourceInvarianceReceipt(
            sealed_machine_sha_before=machine_sha_before,
            sealed_machine_sha_after=machine_sha_after,
            baseline_prediction_root=baseline,
            poisoned_source_prediction_root=poisoned,
            deleted_source_prediction_root=deleted,
            poison_written=True,
            source_deleted=not self.source_path.exists(),
            invariant=invariant,
        )
        if not invariant:
            raise ProtocolViolation("sealed execution depends on consumed source")
        _publish_immutable(
            self.root / "source_invariance_receipt.json",
            canonical_json_bytes(asdict(receipt)),
        )
        return receipt

    def _record_event(self, name: str, **fields: object) -> int:
        sequence = len(self._events) + 1
        event = {"event": name, "sequence": sequence, **fields}
        _publish_immutable(
            self.root / "events" / f"{sequence:06d}.json",
            canonical_json_bytes(event),
        )
        self._events.append(event)
        _replace_mutable_event_view(
            self.root / "event_log.mutable_view.json",
            canonical_json_bytes(self._events),
        )
        return sequence


def validate_challenge_receipt(
    receipt: ChallengeReceipt,
    spec: ProtocolSpec,
) -> None:
    if receipt.machine_sha_before != receipt.machine_sha_after:
        raise ProtocolViolation("challenge changed sealed machine bytes")
    if (
        receipt.compile_count_before != 1
        or receipt.compile_count_after != 1
    ):
        raise ProtocolViolation("challenge invoked or followed a recompilation")
    if receipt.requested_depth_quotas != spec.depth_quotas:
        raise ProtocolViolation("challenge quota differs from frozen protocol")
    if receipt.realized_depth_counts != spec.depth_quotas:
        raise ProtocolViolation("realized depth quota is incorrect")
    if receipt.duplicate_policy != "reject" or receipt.duplicate_count != 0:
        raise ProtocolViolation("challenge duplicate receipt failed")
    if receipt.total_coordinates != sum(
        quota for _, quota in spec.depth_quotas
    ):
        raise ProtocolViolation("challenge total does not match frozen quotas")
    sequence = (
        receipt.machine_seal_event,
        receipt.challenge_seed_event,
        receipt.coordinate_commit_event,
        receipt.key_render_event,
        receipt.prediction_seal_event,
        receipt.answer_assessment_event,
    )
    if tuple(sorted(sequence)) != sequence or len(set(sequence)) != len(sequence):
        raise ProtocolViolation("seal/commit/render/assessment order is invalid")


def assess_rehearsal_transcript(root: Path) -> TranscriptAssessment:
    """Independent transcript assessor; targets come from latent relations."""

    canonical_root = root.resolve()
    protocol_row = _read_canonical_json(canonical_root / "protocol.json")
    if not isinstance(protocol_row, dict):
        raise ProtocolViolation("protocol must be a JSON object")
    spec = ProtocolSpec.from_mapping(protocol_row)
    protocol_payload = canonical_json_bytes(protocol_row)
    protocol_root = _hex_commitment(PROTOCOL_DOMAIN, protocol_payload)
    declared_protocol_root = (
        canonical_root / "protocol_root.txt"
    ).read_text(encoding="ascii").strip()

    world_receipt = _read_canonical_json(canonical_root / "world_receipt.json")
    machine_receipt = _read_canonical_json(
        canonical_root / "machine_receipt.json"
    )
    latent_row = _read_canonical_json(
        canonical_root / "assessor" / "latent_world.json"
    )
    if not isinstance(world_receipt, dict) or not isinstance(
        machine_receipt, dict
    ) or not isinstance(latent_row, dict):
        raise ProtocolViolation("transcript receipts must be objects")
    mechanics = WorldMechanics.from_mapping(latent_row, spec)
    admissibility = assess_world_admissibility(mechanics, spec)
    world_beacon = _beacon_from_mapping(
        _required_mapping(world_receipt, "beacon")
    )
    world_seed = derive_world_seed(protocol_root, world_beacon)
    expected_world_seed_commitment = _hex_commitment(
        "EFC/world-seed-commitment/v1", world_seed
    )
    expected_streams = {
        label: _hex_commitment(
            "EFC/world-stream-commitment/v1",
            derive_world_stream_seed(world_seed, label),
        )
        for label in spec.world_stream_labels
    }
    declared_streams = _required_mapping(
        world_receipt, "stream_commitments"
    )
    sealed_world_copy = (
        canonical_root / "sealed" / "world_evidence.copy"
    ).read_bytes()
    world_root = _hex_commitment("EFC/world-root/v1", sealed_world_copy)
    compile_world_evidence(sealed_world_copy, spec)
    machine = (canonical_root / "sealed" / "machine.bin").read_bytes()
    machine_root = _hex_commitment(
        MACHINE_DOMAIN,
        bytes.fromhex(protocol_root),
        bytes.fromhex(world_root),
        machine,
    )
    event_paths = sorted((canonical_root / "events").glob("*.json"))
    event_rows = [_read_canonical_json(path) for path in event_paths]
    expected_event_names = [
        f"{index:06d}.json" for index in range(1, len(event_paths) + 1)
    ]
    event_by_sequence: dict[int, Mapping[str, object]] = {}
    for index, event_row in enumerate(event_rows, start=1):
        if not isinstance(event_row, dict):
            raise ProtocolViolation("append-only event must be an object")
        if event_row.get("sequence") != index:
            raise ProtocolViolation("append-only event sequence is not contiguous")
        event_by_sequence[index] = event_row
    event_view = _read_canonical_json(
        canonical_root / "event_log.mutable_view.json"
    )

    checks: dict[str, bool] = {
        "canonical_protocol_root": protocol_root == declared_protocol_root,
        "append_only_events_back_mutable_view": (
            [path.name for path in event_paths] == expected_event_names
            and event_view == event_rows
            and bool(event_rows)
            and event_rows[0].get("event") == "protocol_committed"
        ),
        "world_seed_from_post_protocol_beacon": (
            world_receipt.get("protocol_root") == protocol_root
            and world_receipt.get("world_seed_commitment")
            == expected_world_seed_commitment
        ),
        "world_streams_domain_separated": (
            declared_streams == expected_streams
            and len(set(expected_streams.values())) == len(expected_streams)
        ),
        "world_admissibility_query_free": (
            admissibility["query_fields_seen"] == 0
            and world_receipt.get("admissibility") == admissibility
        ),
        "world_root_matches_sealed_copy": (
            world_receipt.get("world_root") == world_root
            and machine_receipt.get("world_root") == world_root
        ),
        "fixed_machine_root": (
            machine_receipt.get("machine_root") == machine_root
            and machine_receipt.get("machine_bytes") == spec.machine_bytes
            and machine_receipt.get("machine_sha256") == _sha256_hex(machine)
            and machine_receipt.get("compile_count") == 1
            and machine_receipt.get("machine_format") == MACHINE_FORMAT
            and machine_receipt.get("machine_format_status")
            == MACHINE_FORMAT_STATUS
            and machine_receipt.get("runtime_claim") == RUNTIME_CLAIM
        ),
        "no_deployed_runtime_claim": (
            spec.machine_format == MACHINE_FORMAT
            and spec.machine_format_status == MACHINE_FORMAT_STATUS
            and spec.runtime_claim == RUNTIME_CLAIM
        ),
    }

    challenge_dirs = sorted(
        path
        for path in (canonical_root / "challenges").glob("*")
        if path.is_dir()
    )
    independently_assessed = 0
    challenge_machine_roots: set[str] = set()
    for challenge_dir in challenge_dirs:
        receipt_row = _read_canonical_json(challenge_dir / "receipt.json")
        coordinate_rows = _read_canonical_json(
            challenge_dir / "abstract_coordinates.json"
        )
        rendered_rows = _read_canonical_json(
            challenge_dir / "rendered_queries.json"
        )
        prediction_rows = _read_canonical_json(
            challenge_dir / "predictions.json"
        )
        answer_rows = _read_canonical_json(
            challenge_dir / "assessor_answers.json"
        )
        if not isinstance(receipt_row, dict):
            raise ProtocolViolation("challenge receipt must be an object")
        receipt = _challenge_receipt_from_mapping(receipt_row)
        validate_challenge_receipt(receipt, spec)
        expected_receipt_events = {
            receipt.machine_seal_event: "machine_sealed",
            receipt.challenge_seed_event: "challenge_seed_derived",
            receipt.coordinate_commit_event: "abstract_coordinates_committed",
            receipt.key_render_event: "opaque_keys_rendered",
            receipt.prediction_seal_event: "predictions_sealed",
            receipt.answer_assessment_event: "answers_independently_assessed",
        }
        if any(
            event_by_sequence.get(sequence, {}).get("event") != expected_name
            for sequence, expected_name in expected_receipt_events.items()
        ):
            raise ProtocolViolation(
                "challenge receipt is not backed by append-only events"
            )
        beacon = _beacon_from_mapping(receipt.beacon)
        if beacon.round <= world_beacon.round or beacon.value == world_beacon.value:
            raise ProtocolViolation("challenge beacon ordering/distinction failed")
        seed = derive_challenge_seed(
            protocol_root, world_root, machine_root, beacon
        )
        expected_seed_commitment = _hex_commitment(
            "EFC/challenge-seed-commitment/v1", seed
        )
        coordinates = generate_abstract_coordinates(seed, spec)
        expected_coordinate_payload = canonical_json_bytes(
            [coordinate.canonical_dict() for coordinate in coordinates]
        )
        if coordinate_rows != [
            coordinate.canonical_dict() for coordinate in coordinates
        ]:
            raise ProtocolViolation("abstract coordinate transcript mismatch")
        expected_rendered = render_coordinates(machine, spec, coordinates)
        if rendered_rows != list(expected_rendered):
            raise ProtocolViolation("opaque-key rendering transcript mismatch")
        expected_answers = assess_by_relation_composition(
            mechanics, spec, coordinates
        )
        if prediction_rows != list(expected_answers):
            raise ProtocolViolation(
                "predictions disagree with independent relation assessor"
            )
        if answer_rows != list(expected_answers):
            raise ProtocolViolation("committed assessor answers are false")
        independently_assessed += len(expected_answers)
        challenge_machine_roots.add(receipt.machine_root)
        checks[f"challenge_{challenge_dir.name}_seed"] = (
            receipt.challenge_seed_commitment == expected_seed_commitment
        )
        checks[f"challenge_{challenge_dir.name}_coordinate_commit"] = (
            receipt.coordinate_root
            == _hex_commitment(COORDINATE_DOMAIN, expected_coordinate_payload)
        )
        checks[f"challenge_{challenge_dir.name}_render_commit"] = (
            receipt.render_root
            == _hex_commitment(
                RENDER_DOMAIN,
                canonical_json_bytes(list(expected_rendered)),
            )
        )
        checks[f"challenge_{challenge_dir.name}_answer_commit"] = (
            receipt.answer_root
            == _hex_commitment(
                ANSWER_DOMAIN,
                canonical_json_bytes(list(expected_answers)),
            )
        )
    checks["same_machine_across_challenge_seeds"] = (
        bool(challenge_dirs)
        and challenge_machine_roots == {machine_root}
        and machine_receipt.get("compile_count") == 1
    )
    passed = all(checks.values())
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise ProtocolViolation(
            "independent transcript assessment failed: " + ", ".join(failed)
        )
    return TranscriptAssessment(
        passed=True,
        checks=checks,
        challenge_count=len(challenge_dirs),
        independently_assessed_answers=independently_assessed,
    )


def _validate_coordinate(
    coordinate: AbstractCoordinate,
    spec: ProtocolSpec,
) -> None:
    if coordinate.world not in range(spec.world_count):
        raise ProtocolViolation("coordinate world is out of range")
    if coordinate.start not in range(spec.state_count):
        raise ProtocolViolation("coordinate start is out of range")
    if coordinate.observer not in range(spec.observer_count):
        raise ProtocolViolation("coordinate observer is out of range")
    if coordinate.renderer not in range(spec.renderer_count):
        raise ProtocolViolation("coordinate renderer is out of range")
    if any(action not in range(spec.action_count) for action in coordinate.actions):
        raise ProtocolViolation("coordinate action is out of range")


def _required_mapping(
    row: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = row.get(key)
    if not isinstance(value, dict) or any(
        not isinstance(item, str) for item in value
    ):
        raise ProtocolViolation(f"{key} must be a string-keyed object")
    return value


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ProtocolViolation(f"{key} must be a string")
    return value


def _plain_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolViolation("expected a plain integer")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    return _plain_int(row.get(key))


def _required_int_vector(
    row: Mapping[str, object],
    key: str,
    width: int,
) -> tuple[int, ...]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != width:
        raise ProtocolViolation(f"{key} must have width {width}")
    return tuple(_plain_int(item) for item in value)


def _required_str_tuple(
    row: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ProtocolViolation(f"{key} must be a string list")
    return tuple(value)


def _required_pair_tuple(
    row: Mapping[str, object],
    key: str,
) -> tuple[tuple[int, int], ...]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ProtocolViolation(f"{key} must be a list")
    pairs: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ProtocolViolation(f"{key} row must be a pair")
        pairs.append((_plain_int(item[0]), _plain_int(item[1])))
    return tuple(pairs)


def _required_matrix(
    row: Mapping[str, object],
    key: str,
    height: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != height:
        raise ProtocolViolation(f"{key} has incorrect height")
    matrix: list[tuple[int, ...]] = []
    for raw_row in value:
        if not isinstance(raw_row, list) or len(raw_row) != width:
            raise ProtocolViolation(f"{key} has incorrect width")
        matrix.append(tuple(_plain_int(item) for item in raw_row))
    return tuple(matrix)


def _beacon_from_mapping(row: Mapping[str, object]) -> Beacon:
    if set(row) != {"round", "value"}:
        raise ProtocolViolation("beacon fields differ from schema")
    return Beacon(_required_int(row, "round"), _required_str(row, "value"))


def _challenge_receipt_from_mapping(
    row: Mapping[str, object],
) -> ChallengeReceipt:
    expected = set(ChallengeReceipt.__dataclass_fields__)
    if set(row) != expected:
        raise ProtocolViolation("challenge receipt fields differ from schema")
    return ChallengeReceipt(
        beacon=_required_mapping(row, "beacon"),
        challenge_seed_commitment=_required_str(
            row, "challenge_seed_commitment"
        ),
        machine_root=_required_str(row, "machine_root"),
        world_root=_required_str(row, "world_root"),
        coordinate_root=_required_str(row, "coordinate_root"),
        render_root=_required_str(row, "render_root"),
        prediction_root=_required_str(row, "prediction_root"),
        answer_root=_required_str(row, "answer_root"),
        requested_depth_quotas=_required_pair_tuple(
            row, "requested_depth_quotas"
        ),
        realized_depth_counts=_required_pair_tuple(
            row, "realized_depth_counts"
        ),
        duplicate_policy=_required_str(row, "duplicate_policy"),
        duplicate_count=_required_int(row, "duplicate_count"),
        total_coordinates=_required_int(row, "total_coordinates"),
        machine_sha_before=_required_str(row, "machine_sha_before"),
        machine_sha_after=_required_str(row, "machine_sha_after"),
        compile_count_before=_required_int(row, "compile_count_before"),
        compile_count_after=_required_int(row, "compile_count_after"),
        machine_seal_event=_required_int(row, "machine_seal_event"),
        challenge_seed_event=_required_int(row, "challenge_seed_event"),
        coordinate_commit_event=_required_int(row, "coordinate_commit_event"),
        key_render_event=_required_int(row, "key_render_event"),
        prediction_seal_event=_required_int(row, "prediction_seal_event"),
        answer_assessment_event=_required_int(
            row, "answer_assessment_event"
        ),
    )


__all__ = [
    "AbstractCoordinate",
    "Beacon",
    "ChallengeReceipt",
    "ProtocolSpec",
    "ProtocolViolation",
    "SealFirstRehearsal",
    "SourceInvarianceReceipt",
    "TranscriptAssessment",
    "WorldMechanics",
    "assess_by_relation_composition",
    "assess_rehearsal_transcript",
    "assess_world_admissibility",
    "canonical_json_bytes",
    "compile_world_evidence",
    "derive_challenge_seed",
    "derive_world_seed",
    "derive_world_stream_seed",
    "execute_sealed_machine",
    "generate_abstract_coordinates",
    "generate_consumed_world_fixture",
    "render_coordinates",
    "validate_challenge_receipt",
]
