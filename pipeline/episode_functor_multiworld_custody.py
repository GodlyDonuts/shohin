#!/usr/bin/env python3
"""Consumed multiworld custody rehearsal for EFC source compilation.

This CPU-only module freezes train/development worlds before a candidate root,
then refuses to materialize confirmation worlds until that candidate is sealed
and a strictly later beacon is supplied. The caller-supplied beacons are not
proof of external unpredictability; the rehearsal proves phase guards,
content-addressing, source-renderer balance, and structural split isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
from typing import Iterable, Sequence

from pipeline.episode_functor_independent_world import (
    IndependentWorld,
    generate_independent_world,
)
from pipeline.episode_functor_seal_protocol import (
    Beacon,
    ProtocolViolation,
    _hex_commitment,
    _publish_immutable,
    canonical_json_bytes,
)
from pipeline.episode_functor_source_renderers import (
    LINE_MAGIC as SOURCE_LINE_MAGIC,
    encode_line_events,
)
from pipeline.episode_functor_wire_protocol import (
    INDEPENDENT_GENERATOR_SOURCE,
    SOURCE_RENDERER_SOURCE,
    WireProtocolSpec,
    decode_deployed_machine,
    encode_deployed_machine,
)


PROTOCOL_DOMAIN = "EFC/multiworld-custody-protocol/v2"
OPEN_SPLIT_DOMAIN = "EFC/multiworld-open-splits/v2"
CONFIRMATION_DOMAIN = "EFC/multiworld-confirmation/v2"
MAX_GENERATION_ATTEMPTS = 256


def _sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class MultiworldCustodySpec:
    schema: str = "efc-multiworld-custody-rehearsal-v2"
    train_worlds: int = 8
    development_worlds: int = 4
    confirmation_worlds: int = 4
    state_count: int = 5
    action_count: int = 3
    observer_count: int = 2
    answer_count: int = 5
    source_renderer_count: int = 2

    def __post_init__(self) -> None:
        if self.schema != "efc-multiworld-custody-rehearsal-v2":
            raise ProtocolViolation("unknown multiworld custody schema")
        if (
            min(
                self.train_worlds,
                self.development_worlds,
                self.confirmation_worlds,
            )
            < 2
        ):
            raise ProtocolViolation("every custody split needs both source renderers")
        if (
            self.state_count != 5
            or self.action_count != 3
            or self.observer_count != 2
            or self.answer_count < 2
            or self.source_renderer_count != 2
        ):
            raise ProtocolViolation("multiworld mechanics differ from frozen cell")

    def canonical_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "generator_source_sha256": _sha256_hex(
                INDEPENDENT_GENERATOR_SOURCE.read_bytes()
            ),
            "machine_bytes": 1_536,
            "multiworld_source_sha256": _sha256_hex(Path(__file__).read_bytes()),
            "source_renderer_source_sha256": _sha256_hex(
                SOURCE_RENDERER_SOURCE.read_bytes()
            ),
            "wire_compiler_source_sha256": _sha256_hex(
                Path(encode_deployed_machine.__code__.co_filename).read_bytes()
            ),
        }


@dataclass(frozen=True)
class WorldRecord:
    split: str
    ordinal: int
    generation_attempt: int
    child_beacon_value: str
    source_renderer: str
    admissibility_sha256: str
    evidence_sha256: str
    latent_sha256: str
    machine_sha256: str
    machine_payload_sha256: str
    structural_canonical_hex: str
    structural_signature: str
    stream_commitments_sha256: str
    world_seed_commitment: str
    empty_observer_class_count: int
    future_behavior_class_count: int

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SplitManifest:
    schema: str
    split: str
    beacon: dict[str, object]
    records: tuple[WorldRecord, ...]
    manifest_root: str

    def canonical_dict(self) -> dict[str, object]:
        return {
            "beacon": self.beacon,
            "manifest_root": self.manifest_root,
            "records": [record.canonical_dict() for record in self.records],
            "schema": self.schema,
            "split": self.split,
        }


def canonical_structural_form(
    transitions: Sequence[Sequence[int]],
    observations: Sequence[Sequence[int]],
) -> bytes:
    """Return the exact canonical form for the frozen 5/3/2 cell."""

    if len(transitions) != 3 or len(observations) != 2:
        raise ProtocolViolation("structural signature input is outside frozen cell")
    state_count = len(transitions[0])
    action_count = len(transitions)
    observer_count = len(observations)
    if (
        state_count != 5
        or action_count != 3
        or observer_count != 2
        or any(len(row) != state_count for row in transitions)
        or any(len(row) != state_count for row in observations)
        or any(
            destination not in range(state_count)
            for row in transitions
            for destination in row
        )
        or any(value not in {0, 1} for row in observations for value in row)
    ):
        raise ProtocolViolation("structural signature input is outside frozen cell")

    best: bytes | None = None
    for state_old_for_new in permutations(range(state_count)):
        old_to_new = {old: new for new, old in enumerate(state_old_for_new)}
        for action_old_for_new in permutations(range(action_count)):
            transition_bytes = bytes(
                old_to_new[transitions[old_action][old_state]]
                for old_action in action_old_for_new
                for old_state in state_old_for_new
            )
            for observer_old_for_new in permutations(range(observer_count)):
                for flips in product((0, 1), repeat=observer_count):
                    observation_bytes = bytes(
                        observations[old_observer][old_state] ^ flips[new_observer]
                        for new_observer, old_observer in enumerate(
                            observer_old_for_new
                        )
                        for old_state in state_old_for_new
                    )
                    candidate = transition_bytes + observation_bytes
                    if best is None or candidate < best:
                        best = candidate
    if best is None:
        raise ProtocolViolation("structural signature enumeration was empty")
    return best


def canonical_structural_signature(
    transitions: Sequence[Sequence[int]],
    observations: Sequence[Sequence[int]],
) -> str:
    """Return a SHA-256 receipt for the exact canonical structural form."""

    return _sha256_hex(canonical_structural_form(transitions, observations))


def _child_beacon_value(
    *,
    protocol_root: str,
    beacon: Beacon,
    split: str,
    ordinal: int,
    attempt: int,
    candidate_root: str | None,
    open_splits_root: str | None,
) -> str:
    payload = canonical_json_bytes(
        {
            "attempt": attempt,
            "beacon": asdict(beacon),
            "candidate_root": candidate_root,
            "open_splits_root": open_splits_root,
            "ordinal": ordinal,
            "protocol_root": protocol_root,
            "split": split,
        }
    )
    return _sha256_hex(b"EFC/multiworld-child-beacon/v1\0" + payload)


def _world_source_payload(
    world: IndependentWorld,
    renderer_index: int,
) -> tuple[str, bytes]:
    if renderer_index == 0:
        return "canonical-json-events-v2", world.evidence
    if renderer_index == 1:
        return "strict-line-events-v1", encode_line_events(world.evidence)
    raise ProtocolViolation("source renderer index is outside frozen support")


def _manifest_root(
    domain: str,
    split: str,
    beacon: Beacon,
    records: Sequence[WorldRecord],
    *,
    protocol_root: str,
    candidate_root: str | None,
    open_splits_root: str | None,
) -> str:
    payload = canonical_json_bytes(
        {
            "beacon": asdict(beacon),
            "candidate_root": candidate_root,
            "open_splits_root": open_splits_root,
            "protocol_root": protocol_root,
            "records": [record.canonical_dict() for record in records],
            "split": split,
        }
    )
    return _hex_commitment(domain, payload)


class MultiworldCustodyRehearsal:
    """Filesystem-backed phase guards for a consumed multiworld rehearsal."""

    def __init__(
        self,
        root: Path,
        spec: MultiworldCustodySpec = MultiworldCustodySpec(),
    ) -> None:
        self.root = root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise ProtocolViolation("multiworld custody root must start empty")
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self._protocol_payload = canonical_json_bytes(spec.canonical_dict())
        self.protocol_root = _hex_commitment(PROTOCOL_DOMAIN, self._protocol_payload)
        _publish_immutable(self.root / "protocol.json", self._protocol_payload)
        _publish_immutable(
            self.root / "protocol_root.txt",
            (self.protocol_root + "\n").encode("ascii"),
        )
        self._open_beacon: Beacon | None = None
        self._candidate_root: str | None = None
        self._open_manifests: dict[str, SplitManifest] = {}
        self._confirmation_manifest: SplitManifest | None = None
        self._structural_forms: set[bytes] = set()
        self.open_splits_root: str | None = None
        self._events: list[dict[str, object]] = []
        self._event_tip = bytes(32).hex()
        self._record_event("protocol_committed", root=self.protocol_root)

    def _verify_protocol(self) -> None:
        if self.root.joinpath("protocol.json").read_bytes() != (self._protocol_payload):
            raise ProtocolViolation("multiworld protocol changed after commitment")
        if self.root.joinpath("protocol_root.txt").read_text(encoding="ascii") != (
            self.protocol_root + "\n"
        ):
            raise ProtocolViolation("multiworld protocol root receipt changed")

    def _verify_event_chain(self) -> None:
        events_root = self.root / "events"
        expected_names = {
            f"{event_id:06d}.json"
            for event_id in range(1, len(self._events) + 1)
        }
        event_paths = sorted(events_root.iterdir())
        if (
            {path.name for path in event_paths} != expected_names
            or any(not path.is_file() or path.is_symlink() for path in event_paths)
        ):
            raise ProtocolViolation("multiworld event file set changed")
        previous = bytes(32).hex()
        for event_id, (path, expected_row) in enumerate(
            zip(event_paths, self._events, strict=True),
            start=1,
        ):
            payload = path.read_bytes()
            if payload != canonical_json_bytes(expected_row):
                raise ProtocolViolation("multiworld event payload changed")
            row = json.loads(payload)
            if (
                row["event_id"] != event_id
                or row["previous_event_sha256"] != previous
            ):
                raise ProtocolViolation("multiworld event chain is invalid")
            previous = _sha256_hex(payload)
        if previous != self._event_tip:
            raise ProtocolViolation("multiworld event tip changed")

    def _expected_record_payloads(
        self,
        record: WorldRecord,
        *,
        beacon: Beacon,
        candidate_root: str | None,
        open_splits_root: str | None,
    ) -> tuple[WorldRecord, dict[str, bytes]]:
        expected_child = _child_beacon_value(
            protocol_root=self.protocol_root,
            beacon=beacon,
            split=record.split,
            ordinal=record.ordinal,
            attempt=record.generation_attempt,
            candidate_root=candidate_root,
            open_splits_root=open_splits_root,
        )
        world = generate_independent_world(
            protocol_root=self.protocol_root,
            beacon_round=beacon.round,
            beacon_value=expected_child,
            state_count=self.spec.state_count,
            action_count=self.spec.action_count,
            observer_count=self.spec.observer_count,
            answer_count=self.spec.answer_count,
            renderer_count=1,
        )
        renderer_index = (
            record.ordinal
            + {"train": 0, "development": 1, "confirmation": 0}[record.split]
        ) % self.spec.source_renderer_count
        renderer, evidence = _world_source_payload(world, renderer_index)
        machine = encode_deployed_machine(evidence, WireProtocolSpec())
        latent = canonical_json_bytes(
            {
                "observer_maps": [list(row) for row in world.observers],
                "schema": "efc-latent-relations-v1",
                "transition_relations": [
                    list(row) for row in world.transitions
                ],
            }
        )
        admissibility = canonical_json_bytes(dict(world.admissibility_receipt))
        stream_commitments = canonical_json_bytes(dict(world.stream_commitments))
        structural_form = canonical_structural_form(
            world.transitions,
            world.observers,
        )
        expected_record = WorldRecord(
            split=record.split,
            ordinal=record.ordinal,
            generation_attempt=record.generation_attempt,
            child_beacon_value=expected_child,
            source_renderer=renderer,
            admissibility_sha256=_sha256_hex(admissibility),
            evidence_sha256=_sha256_hex(evidence),
            latent_sha256=_sha256_hex(latent),
            machine_sha256=_sha256_hex(machine),
            machine_payload_sha256=machine[-32:].hex(),
            structural_canonical_hex=structural_form.hex(),
            structural_signature=_sha256_hex(structural_form),
            stream_commitments_sha256=_sha256_hex(stream_commitments),
            world_seed_commitment=world.world_seed_commitment,
            empty_observer_class_count=int(
                world.admissibility_receipt["empty_observer_class_count"]
            ),
            future_behavior_class_count=int(
                world.admissibility_receipt["future_behavior_class_count"]
            ),
        )
        return expected_record, {
            "admissibility.json": admissibility,
            "evidence.bin": evidence,
            "latent.json": latent,
            "machine.bin": machine,
            "stream_commitments.json": stream_commitments,
        }

    def _verify_record(
        self,
        record: WorldRecord,
        *,
        beacon: Beacon,
        candidate_root: str | None,
        open_splits_root: str | None,
    ) -> None:
        try:
            expected_record, expected_payloads = self._expected_record_payloads(
                record,
                beacon=beacon,
                candidate_root=candidate_root,
                open_splits_root=open_splits_root,
            )
        except (KeyError, ValueError) as exc:
            raise ProtocolViolation(
                "multiworld semantic receipt is malformed"
            ) from exc
        if expected_record != record:
            raise ProtocolViolation("multiworld semantic receipt is inconsistent")
        world_root = (
            self.root / "worlds" / record.split / f"{record.ordinal:04d}"
        )
        expected_files = {
            "admissibility.json",
            "evidence.bin",
            "latent.json",
            "machine.bin",
            "receipt.json",
            "stream_commitments.json",
        }
        world_paths = tuple(world_root.iterdir())
        if (
            {path.name for path in world_paths} != expected_files
            or any(not path.is_file() or path.is_symlink() for path in world_paths)
        ):
            raise ProtocolViolation("multiworld world file set changed")
        for filename, expected_payload in expected_payloads.items():
            if world_root.joinpath(filename).read_bytes() != expected_payload:
                raise ProtocolViolation("multiworld world payload changed")
        if world_root.joinpath("receipt.json").read_bytes() != canonical_json_bytes(
            record.canonical_dict()
        ):
            raise ProtocolViolation("multiworld world receipt changed")
        evidence = world_root.joinpath("evidence.bin").read_bytes()
        machine = world_root.joinpath("machine.bin").read_bytes()
        expected_renderer = (
            "strict-line-events-v1"
            if evidence.startswith((SOURCE_LINE_MAGIC + "\t").encode("ascii"))
            else "canonical-json-events-v2"
        )
        if record.source_renderer != expected_renderer:
            raise ProtocolViolation("multiworld source renderer receipt changed")
        if encode_deployed_machine(evidence, WireProtocolSpec()) != machine:
            raise ProtocolViolation("multiworld source no longer compiles to machine")
        if machine[-32:].hex() != record.machine_payload_sha256:
            raise ProtocolViolation("multiworld machine payload hash changed")
        tables = decode_deployed_machine(machine, WireProtocolSpec())
        structural_form = canonical_structural_form(
            tables.transitions,
            tables.observations,
        )
        if (
            structural_form.hex() != record.structural_canonical_hex
            or _sha256_hex(structural_form) != record.structural_signature
        ):
            raise ProtocolViolation("multiworld structural receipt changed")

    def _verify_manifest(
        self,
        manifest: SplitManifest,
        *,
        candidate_root: str | None,
        open_splits_root: str | None,
    ) -> None:
        expected_count = {
            "train": self.spec.train_worlds,
            "development": self.spec.development_worlds,
            "confirmation": self.spec.confirmation_worlds,
        }.get(manifest.split)
        if (
            manifest.schema != "efc-multiworld-split-manifest-v1"
            or expected_count is None
            or len(manifest.records) != expected_count
            or tuple(record.ordinal for record in manifest.records)
            != tuple(range(expected_count))
            or any(record.split != manifest.split for record in manifest.records)
        ):
            raise ProtocolViolation("multiworld split manifest is inconsistent")
        if self.root.joinpath(
            f"{manifest.split}_manifest.json"
        ).read_bytes() != canonical_json_bytes(manifest.canonical_dict()):
            raise ProtocolViolation("multiworld split manifest changed")
        beacon = Beacon(**manifest.beacon)
        domain = (
            OPEN_SPLIT_DOMAIN
            if manifest.split in {"train", "development"}
            else CONFIRMATION_DOMAIN
        )
        expected_root = _manifest_root(
            domain,
            manifest.split,
            beacon,
            manifest.records,
            protocol_root=self.protocol_root,
            candidate_root=candidate_root,
            open_splits_root=open_splits_root,
        )
        if expected_root != manifest.manifest_root:
            raise ProtocolViolation("multiworld split root changed")
        for record in manifest.records:
            self._verify_record(
                record,
                beacon=beacon,
                candidate_root=candidate_root,
                open_splits_root=open_splits_root,
            )

    def _verify_filesystem_shape(
        self,
        *,
        allowed_top_level_files: frozenset[str] = frozenset(),
    ) -> None:
        expected_top = {
            "events",
            "protocol.json",
            "protocol_root.txt",
            "worlds",
            *allowed_top_level_files,
        }
        expected_splits = set(self._open_manifests)
        if self._open_manifests:
            expected_top.update(
                {
                    "development_manifest.json",
                    "open_splits_root.txt",
                    "train_manifest.json",
                }
            )
        if self._candidate_root is not None:
            expected_top.add("candidate_seal.json")
        if self._confirmation_manifest is not None:
            expected_top.add("confirmation_manifest.json")
            expected_splits.add("confirmation")
        top_paths = tuple(self.root.iterdir())
        if {path.name for path in top_paths} != expected_top:
            raise ProtocolViolation("multiworld top-level file set changed")
        for path in top_paths:
            if path.name in {"events", "worlds"}:
                if not path.is_dir() or path.is_symlink():
                    raise ProtocolViolation("multiworld artifact directory changed")
            elif not path.is_file() or path.is_symlink():
                raise ProtocolViolation("multiworld artifact file type changed")

        world_splits = tuple(self.root.joinpath("worlds").iterdir())
        if (
            {path.name for path in world_splits} != expected_splits
            or any(not path.is_dir() or path.is_symlink() for path in world_splits)
        ):
            raise ProtocolViolation("multiworld split directory set changed")
        manifests = dict(self._open_manifests)
        if self._confirmation_manifest is not None:
            manifests["confirmation"] = self._confirmation_manifest
        for split, manifest in manifests.items():
            split_root = self.root / "worlds" / split
            expected_worlds = {
                f"{record.ordinal:04d}" for record in manifest.records
            }
            world_paths = tuple(split_root.iterdir())
            if (
                {path.name for path in world_paths} != expected_worlds
                or any(
                    not path.is_dir() or path.is_symlink()
                    for path in world_paths
                )
            ):
                raise ProtocolViolation("multiworld world directory set changed")

    def _verify_structural_sequence(
        self,
        *,
        include_confirmation: bool,
    ) -> None:
        manifests = [
            self._open_manifests["train"],
            self._open_manifests["development"],
        ]
        if include_confirmation:
            if self._confirmation_manifest is None:
                raise ProtocolViolation("multiworld confirmation state is incomplete")
            manifests.append(self._confirmation_manifest)
        seen: set[bytes] = set()
        for manifest in manifests:
            beacon = Beacon(**manifest.beacon)
            candidate_root = (
                self._candidate_root
                if manifest.split == "confirmation"
                else None
            )
            open_splits_root = (
                self.open_splits_root
                if manifest.split == "confirmation"
                else None
            )
            for record in manifest.records:
                if not 0 <= record.generation_attempt < MAX_GENERATION_ATTEMPTS:
                    raise ProtocolViolation(
                        "multiworld generation attempt is outside protocol"
                    )
                for attempt in range(record.generation_attempt):
                    child = _child_beacon_value(
                        protocol_root=self.protocol_root,
                        beacon=beacon,
                        split=record.split,
                        ordinal=record.ordinal,
                        attempt=attempt,
                        candidate_root=candidate_root,
                        open_splits_root=open_splits_root,
                    )
                    world = generate_independent_world(
                        protocol_root=self.protocol_root,
                        beacon_round=beacon.round,
                        beacon_value=child,
                        state_count=self.spec.state_count,
                        action_count=self.spec.action_count,
                        observer_count=self.spec.observer_count,
                        answer_count=self.spec.answer_count,
                        renderer_count=1,
                    )
                    if canonical_structural_form(
                        world.transitions,
                        world.observers,
                    ) not in seen:
                        raise ProtocolViolation(
                            "multiworld generator skipped an admissible attempt"
                        )
                selected = bytes.fromhex(record.structural_canonical_hex)
                if len(selected) != 25 or selected in seen:
                    raise ProtocolViolation(
                        "multiworld structural sequence is inconsistent"
                    )
                seen.add(selected)
        expected_forms = {
            bytes.fromhex(record.structural_canonical_hex)
            for manifest in manifests
            for record in manifest.records
        }
        if seen != expected_forms or (
            include_confirmation and seen != self._structural_forms
        ) or (
            not include_confirmation and not seen.issubset(self._structural_forms)
        ):
            raise ProtocolViolation("multiworld structural state changed")

    def _verify_open_state(
        self,
        *,
        allowed_top_level_files: frozenset[str] = frozenset(),
    ) -> None:
        self._verify_filesystem_shape(
            allowed_top_level_files=allowed_top_level_files
        )
        self._verify_protocol()
        self._verify_event_chain()
        if self.open_splits_root is None or set(self._open_manifests) != {
            "train",
            "development",
        }:
            raise ProtocolViolation("multiworld open state is incomplete")
        for manifest in self._open_manifests.values():
            self._verify_manifest(
                manifest,
                candidate_root=None,
                open_splits_root=None,
            )
        expected_root = _hex_commitment(
            OPEN_SPLIT_DOMAIN,
            bytes.fromhex(self._open_manifests["train"].manifest_root),
            bytes.fromhex(self._open_manifests["development"].manifest_root),
        )
        if (
            expected_root != self.open_splits_root
            or self.root.joinpath("open_splits_root.txt").read_text(
                encoding="ascii"
            )
            != self.open_splits_root + "\n"
        ):
            raise ProtocolViolation("multiworld open-splits root changed")
        self._verify_structural_sequence(include_confirmation=False)

    def _verify_candidate_state(
        self,
        *,
        allowed_top_level_files: frozenset[str] = frozenset(),
    ) -> None:
        self._verify_open_state(
            allowed_top_level_files=allowed_top_level_files
        )
        if self._candidate_root is None:
            raise ProtocolViolation("multiworld candidate is not sealed")
        expected = canonical_json_bytes(
            {
                "candidate_root": self._candidate_root,
                "open_splits_root": self.open_splits_root,
                "protocol_root": self.protocol_root,
            }
        )
        if self.root.joinpath("candidate_seal.json").read_bytes() != expected:
            raise ProtocolViolation("multiworld candidate seal changed")

    def verify_published_state(
        self,
        *,
        allowed_top_level_files: frozenset[str] = frozenset(),
    ) -> None:
        """Revalidate every phase artifact currently published by this process."""

        if self._confirmation_manifest is None:
            if self._candidate_root is None:
                self._verify_open_state(
                    allowed_top_level_files=allowed_top_level_files
                )
            else:
                self._verify_candidate_state(
                    allowed_top_level_files=allowed_top_level_files
                )
            return
        self._verify_candidate_state(
            allowed_top_level_files=allowed_top_level_files
        )
        self._verify_manifest(
            self._confirmation_manifest,
            candidate_root=self._candidate_root,
            open_splits_root=self.open_splits_root,
        )
        self._verify_structural_sequence(include_confirmation=True)

    def _record_event(self, event: str, **fields: object) -> int:
        event_id = len(self._events) + 1
        row = {
            "event": event,
            "event_id": event_id,
            "previous_event_sha256": self._event_tip,
            **fields,
        }
        payload = canonical_json_bytes(row)
        _publish_immutable(self.root / "events" / f"{event_id:06d}.json", payload)
        self._events.append(row)
        self._event_tip = _sha256_hex(payload)
        return event_id

    def _generate_split(
        self,
        split: str,
        count: int,
        beacon: Beacon,
        *,
        candidate_root: str | None,
        open_splits_root: str | None,
    ) -> SplitManifest:
        records: list[WorldRecord] = []
        wire_spec = WireProtocolSpec()
        for ordinal in range(count):
            for attempt in range(MAX_GENERATION_ATTEMPTS):
                child_value = _child_beacon_value(
                    protocol_root=self.protocol_root,
                    beacon=beacon,
                    split=split,
                    ordinal=ordinal,
                    attempt=attempt,
                    candidate_root=candidate_root,
                    open_splits_root=open_splits_root,
                )
                world = generate_independent_world(
                    protocol_root=self.protocol_root,
                    beacon_round=beacon.round,
                    beacon_value=child_value,
                    state_count=self.spec.state_count,
                    action_count=self.spec.action_count,
                    observer_count=self.spec.observer_count,
                    answer_count=self.spec.answer_count,
                    renderer_count=1,
                )
                structural_form = canonical_structural_form(
                    world.transitions, world.observers
                )
                if structural_form in self._structural_forms:
                    continue
                signature = _sha256_hex(structural_form)
                renderer_index = (
                    ordinal + {"train": 0, "development": 1, "confirmation": 0}[split]
                ) % self.spec.source_renderer_count
                renderer, evidence = _world_source_payload(world, renderer_index)
                machine = encode_deployed_machine(evidence, wire_spec)
                latent = canonical_json_bytes(
                    {
                        "observer_maps": [list(row) for row in world.observers],
                        "schema": "efc-latent-relations-v1",
                        "transition_relations": [
                            list(row) for row in world.transitions
                        ],
                    }
                )
                admissibility = canonical_json_bytes(dict(world.admissibility_receipt))
                stream_commitments = canonical_json_bytes(
                    dict(world.stream_commitments)
                )
                record = WorldRecord(
                    split=split,
                    ordinal=ordinal,
                    generation_attempt=attempt,
                    child_beacon_value=child_value,
                    source_renderer=renderer,
                    admissibility_sha256=_sha256_hex(admissibility),
                    evidence_sha256=_sha256_hex(evidence),
                    latent_sha256=_sha256_hex(latent),
                    machine_sha256=_sha256_hex(machine),
                    machine_payload_sha256=machine[-32:].hex(),
                    structural_canonical_hex=structural_form.hex(),
                    structural_signature=signature,
                    stream_commitments_sha256=_sha256_hex(stream_commitments),
                    world_seed_commitment=(world.world_seed_commitment),
                    empty_observer_class_count=int(
                        world.admissibility_receipt["empty_observer_class_count"]
                    ),
                    future_behavior_class_count=int(
                        world.admissibility_receipt["future_behavior_class_count"]
                    ),
                )
                world_root = self.root / "worlds" / split / f"{ordinal:04d}"
                _publish_immutable(world_root / "admissibility.json", admissibility)
                _publish_immutable(world_root / "evidence.bin", evidence)
                _publish_immutable(world_root / "latent.json", latent)
                _publish_immutable(world_root / "machine.bin", machine)
                _publish_immutable(
                    world_root / "stream_commitments.json",
                    stream_commitments,
                )
                _publish_immutable(
                    world_root / "receipt.json",
                    canonical_json_bytes(record.canonical_dict()),
                )
                records.append(record)
                self._structural_forms.add(structural_form)
                break
            else:
                raise ProtocolViolation(
                    f"could not find disjoint {split} world {ordinal}"
                )
        domain = (
            OPEN_SPLIT_DOMAIN
            if split in {"train", "development"}
            else CONFIRMATION_DOMAIN
        )
        root = _manifest_root(
            domain,
            split,
            beacon,
            records,
            protocol_root=self.protocol_root,
            candidate_root=candidate_root,
            open_splits_root=open_splits_root,
        )
        manifest = SplitManifest(
            schema="efc-multiworld-split-manifest-v1",
            split=split,
            beacon=asdict(beacon),
            records=tuple(records),
            manifest_root=root,
        )
        _publish_immutable(
            self.root / f"{split}_manifest.json",
            canonical_json_bytes(manifest.canonical_dict()),
        )
        return manifest

    def freeze_open_splits(
        self,
        beacon: Beacon,
    ) -> tuple[SplitManifest, SplitManifest]:
        self._verify_protocol()
        if self._open_beacon is not None or self._open_manifests:
            raise ProtocolViolation("open splits have already been frozen")
        train = self._generate_split(
            "train",
            self.spec.train_worlds,
            beacon,
            candidate_root=None,
            open_splits_root=None,
        )
        development = self._generate_split(
            "development",
            self.spec.development_worlds,
            beacon,
            candidate_root=None,
            open_splits_root=None,
        )
        self._open_beacon = beacon
        self._open_manifests = {
            "train": train,
            "development": development,
        }
        self.open_splits_root = _hex_commitment(
            OPEN_SPLIT_DOMAIN,
            bytes.fromhex(train.manifest_root),
            bytes.fromhex(development.manifest_root),
        )
        _publish_immutable(
            self.root / "open_splits_root.txt",
            (self.open_splits_root + "\n").encode("ascii"),
        )
        self._record_event("open_splits_frozen", root=self.open_splits_root)
        self._verify_open_state()
        return train, development

    def seal_candidate(self, candidate_root: str) -> None:
        if self.open_splits_root is None:
            raise ProtocolViolation("open splits must be frozen first")
        self._verify_open_state()
        if self._candidate_root is not None:
            raise ProtocolViolation("candidate root has already been sealed")
        if not _valid_digest(candidate_root):
            raise ProtocolViolation("candidate root is not SHA-256 hex")
        self._candidate_root = candidate_root
        receipt = {
            "candidate_root": candidate_root,
            "open_splits_root": self.open_splits_root,
            "protocol_root": self.protocol_root,
        }
        _publish_immutable(
            self.root / "candidate_seal.json",
            canonical_json_bytes(receipt),
        )
        self._record_event("candidate_sealed", root=candidate_root)
        self._verify_candidate_state()

    def open_confirmation(self, beacon: Beacon) -> SplitManifest:
        if (
            self._candidate_root is None
            or self.open_splits_root is None
            or self._open_beacon is None
        ):
            raise ProtocolViolation("candidate and open splits must be sealed first")
        self._verify_candidate_state()
        if self._confirmation_manifest is not None:
            raise ProtocolViolation("confirmation has already been opened")
        if beacon.round <= self._open_beacon.round:
            raise ProtocolViolation("confirmation beacon is not strictly later")
        if beacon.value == self._open_beacon.value:
            raise ProtocolViolation("confirmation beacon repeats open beacon")
        confirmation = self._generate_split(
            "confirmation",
            self.spec.confirmation_worlds,
            beacon,
            candidate_root=self._candidate_root,
            open_splits_root=self.open_splits_root,
        )
        self._confirmation_manifest = confirmation
        self._record_event("confirmation_opened", root=confirmation.manifest_root)
        self.verify_published_state()
        return confirmation

    @property
    def structural_signatures(self) -> frozenset[str]:
        return frozenset(_sha256_hex(form) for form in self._structural_forms)

    @property
    def structural_forms(self) -> frozenset[bytes]:
        return frozenset(self._structural_forms)

    @property
    def open_manifests(self) -> dict[str, SplitManifest]:
        return dict(self._open_manifests)


def structural_overlap(
    manifests: Iterable[SplitManifest],
) -> dict[str, int]:
    sets = {
        manifest.split: {
            bytes.fromhex(record.structural_canonical_hex)
            for record in manifest.records
        }
        for manifest in manifests
    }
    names = sorted(sets)
    return {
        f"{left}__{right}": len(sets[left] & sets[right])
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }


__all__ = [
    "MultiworldCustodyRehearsal",
    "MultiworldCustodySpec",
    "SplitManifest",
    "WorldRecord",
    "canonical_structural_form",
    "canonical_structural_signature",
    "structural_overlap",
]
