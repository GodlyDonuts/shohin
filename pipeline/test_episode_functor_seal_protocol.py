from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest

import pipeline.episode_functor_seal_protocol as seal_protocol
from pipeline.episode_functor_seal_protocol import (
    AbstractCoordinate,
    Beacon,
    ProtocolViolation,
    SealFirstRehearsal,
    WorldMechanics,
    assess_rehearsal_transcript,
    canonical_json_bytes,
    compile_world_evidence,
    derive_challenge_seed,
    derive_world_seed,
    derive_world_stream_seed,
    generate_abstract_coordinates,
    generate_consumed_world_fixture,
    validate_challenge_receipt,
)


WORLD_BEACON = Beacon(round=100, value="consumed-world-beacon-A")
CHALLENGE_A = Beacon(round=200, value="later-challenge-beacon-A")
CHALLENGE_B = Beacon(round=201, value="later-challenge-beacon-B")


def _third_assessor_from_latent_relations(
    latent: Mapping[str, object],
    coordinates: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    """Source-disjoint assessor over relation pairs, not the table executor."""

    raw_transitions = latent["transition_relations"]
    raw_observers = latent["observer_maps"]
    assert isinstance(raw_transitions, list)
    assert isinstance(raw_observers, list)
    transitions = tuple(
        tuple(int(destination) for destination in row)
        for row in raw_transitions
        if isinstance(row, list)
    )
    observers = tuple(
        tuple(int(answer) for answer in row)
        for row in raw_observers
        if isinstance(row, list)
    )
    state_count = len(transitions[0])
    action_relations = tuple(
        frozenset(
            (source, destination)
            for source, destination in enumerate(transition)
        )
        for transition in transitions
    )
    identity = frozenset((state, state) for state in range(state_count))
    answers: list[int] = []
    for coordinate in coordinates:
        actions = coordinate["actions"]
        assert isinstance(actions, list)
        relation = identity
        for action in actions:
            right = action_relations[int(action)]
            relation = frozenset(
                (source, destination)
                for source, middle in relation
                for right_source, destination in right
                if middle == right_source
            )
        start = int(coordinate["start"])
        destinations = sorted(
            destination
            for source, destination in relation
            if source == start
        )
        assert len(destinations) == 1
        answers.append(
            observers[int(coordinate["observer"])][destinations[0]]
        )
    return tuple(answers)


def _sealed_rehearsal(tmp_path: Path) -> SealFirstRehearsal:
    rehearsal = SealFirstRehearsal(tmp_path / "protocol")
    rehearsal.supply_world_beacon(WORLD_BEACON)
    rehearsal.seal_machine()
    return rehearsal


def test_protocol_root_and_world_streams_are_canonical_and_domain_separated(
    tmp_path: Path,
) -> None:
    rehearsal = SealFirstRehearsal(tmp_path / "protocol")
    protocol_payload = rehearsal.root.joinpath("protocol.json").read_bytes()
    assert protocol_payload == canonical_json_bytes(
        json.loads(protocol_payload)
    )
    assert (
        rehearsal.root.joinpath("protocol_root.txt")
        .read_text(encoding="ascii")
        .strip()
        == rehearsal.protocol_root
    )

    fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
    world_seed = derive_world_seed(rehearsal.protocol_root, WORLD_BEACON)
    stream_seeds = [
        derive_world_stream_seed(world_seed, label)
        for label in rehearsal.spec.world_stream_labels
    ]
    assert len(set(stream_seeds)) == len(stream_seeds)
    assert len(dict(fixture.stream_commitments)) == len(stream_seeds)
    assert fixture.admissibility_receipt["query_fields_seen"] == 0
    protocol_row = json.loads(protocol_payload)
    assert protocol_row["machine_format"] == (
        "compact-big-endian-python-rehearsal-v1"
    )
    assert protocol_row["machine_format_status"] == (
        "not-deployed-c-rust-wire-format"
    )
    assert protocol_row["runtime_claim"] == (
        "none-protocol-rehearsal-only"
    )


def test_immutable_protocol_world_machine_and_challenge_publications(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    challenge_dir = (
        rehearsal.root
        / "challenges"
        / receipt.challenge_seed_commitment
    )
    immutable_paths = (
        rehearsal.root / "protocol.json",
        rehearsal.root / "protocol_root.txt",
        rehearsal.root / "world_receipt.json",
        rehearsal.source_path,
        rehearsal.root / "assessor" / "latent_world.json",
        rehearsal.root / "sealed" / "world_evidence.copy",
        rehearsal.root / "sealed" / "machine.bin",
        rehearsal.root / "machine_receipt.json",
        challenge_dir / "abstract_coordinates.json",
        challenge_dir / "rendered_queries.json",
        challenge_dir / "predictions.json",
        challenge_dir / "assessor_answers.json",
        challenge_dir / "receipt.json",
        rehearsal.root / "events" / "000001.json",
    )
    originals = {path: path.read_bytes() for path in immutable_paths}
    original_hashes = {
        path: sha256(payload).hexdigest()
        for path, payload in originals.items()
    }
    for path in immutable_paths:
        with pytest.raises(ProtocolViolation, match="already exists"):
            seal_protocol._publish_immutable(path, b"hostile replacement\n")
        assert path.read_bytes() == originals[path]
        assert sha256(path.read_bytes()).hexdigest() == original_hashes[path]

    compile_count = rehearsal.compile_count
    with pytest.raises(ProtocolViolation, match="already sealed"):
        rehearsal.supply_world_beacon(
            Beacon(round=101, value="replacement-world")
        )
    with pytest.raises(ProtocolViolation, match="single-shot"):
        rehearsal.seal_machine()
    with pytest.raises(ProtocolViolation, match="already been consumed"):
        rehearsal.run_challenge(CHALLENGE_A)
    assert rehearsal.compile_count == compile_count == 1
    for path in immutable_paths:
        assert path.read_bytes() == originals[path]
        assert sha256(path.read_bytes()).hexdigest() == original_hashes[path]


def test_event_history_is_append_only_and_view_is_explicitly_mutable(
    tmp_path: Path,
) -> None:
    rehearsal = SealFirstRehearsal(tmp_path / "protocol")
    first_event = rehearsal.root / "events" / "000001.json"
    first_bytes = first_event.read_bytes()
    first_hash = sha256(first_bytes).hexdigest()
    first_view = rehearsal.root.joinpath(
        "event_log.mutable_view.json"
    ).read_bytes()
    rehearsal.supply_world_beacon(WORLD_BEACON)
    rehearsal.seal_machine()
    rehearsal.run_challenge(CHALLENGE_A)

    events = sorted((rehearsal.root / "events").glob("*.json"))
    assert [path.name for path in events] == [
        f"{index:06d}.json" for index in range(1, len(events) + 1)
    ]
    assert first_event.read_bytes() == first_bytes
    assert sha256(first_event.read_bytes()).hexdigest() == first_hash
    assert rehearsal.root.joinpath(
        "event_log.mutable_view.json"
    ).read_bytes() != first_view
    derived_view = json.loads(
        rehearsal.root.joinpath(
            "event_log.mutable_view.json"
        ).read_bytes()
    )
    assert derived_view == [
        json.loads(path.read_bytes()) for path in events
    ]
    with pytest.raises(ProtocolViolation, match="already exists"):
        seal_protocol._publish_immutable(
            first_event,
            b'{"event":"forged"}\n',
        )
    assert first_event.read_bytes() == first_bytes
    assert sha256(first_event.read_bytes()).hexdigest() == first_hash


def test_world_admissibility_is_independent_of_challenge_seed(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    machine_before = rehearsal.machine_path.read_bytes()
    world_copy_before = rehearsal.root.joinpath(
        "sealed/world_evidence.copy"
    ).read_bytes()
    root_before = rehearsal.machine_root

    first = rehearsal.run_challenge(CHALLENGE_A)
    second = rehearsal.run_challenge(CHALLENGE_B)

    assert first.challenge_seed_commitment != second.challenge_seed_commitment
    assert first.coordinate_root != second.coordinate_root
    assert first.machine_root == second.machine_root == root_before
    assert rehearsal.machine_path.read_bytes() == machine_before
    assert rehearsal.root.joinpath(
        "sealed/world_evidence.copy"
    ).read_bytes() == world_copy_before
    assert rehearsal.compile_count == 1


def test_abstract_coordinates_are_committed_before_opaque_key_rendering(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    assert receipt.machine_seal_event < receipt.challenge_seed_event
    assert receipt.challenge_seed_event < receipt.coordinate_commit_event
    assert receipt.coordinate_commit_event < receipt.key_render_event
    assert receipt.key_render_event < receipt.prediction_seal_event
    assert receipt.prediction_seal_event < receipt.answer_assessment_event

    challenge_dir = rehearsal.root / "challenges" / (
        receipt.challenge_seed_commitment
    )
    abstract_payload = json.loads(
        challenge_dir.joinpath("abstract_coordinates.json").read_bytes()
    )
    rendered_payload = json.loads(
        challenge_dir.joinpath("rendered_queries.json").read_bytes()
    )
    assert all(
        set(row) == {"actions", "observer", "renderer", "start", "world"}
        for row in abstract_payload
    )
    assert all(
        "start_key" in row or "sequence" in row for row in rendered_payload
    )
    abstract_text = canonical_json_bytes(abstract_payload)
    machine = rehearsal.machine_path.read_bytes()
    decoded = seal_protocol._decode_machine(machine, rehearsal.spec)
    for key in (
        *decoded.state_keys,
        *decoded.action_keys,
        *decoded.observer_keys,
    ):
        assert str(key).encode("ascii") not in abstract_text


def test_depth_quota_and_duplicate_receipts_are_exact(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    assert receipt.requested_depth_quotas == rehearsal.spec.depth_quotas
    assert receipt.realized_depth_counts == rehearsal.spec.depth_quotas
    assert receipt.total_coordinates == sum(
        quota for _, quota in rehearsal.spec.depth_quotas
    )
    assert receipt.duplicate_policy == "reject"
    assert receipt.duplicate_count == 0
    validate_challenge_receipt(receipt, rehearsal.spec)


def test_source_poison_and_delete_cannot_change_sealed_execution(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    seed = derive_challenge_seed(
        rehearsal.protocol_root,
        str(rehearsal.world_root),
        str(rehearsal.machine_root),
        CHALLENGE_A,
    )
    coordinates = generate_abstract_coordinates(seed, rehearsal.spec)[:25]
    sealed_copy = rehearsal.root / "sealed" / "machine.bin"
    assert sealed_copy.exists()

    receipt = rehearsal.prove_source_poison_delete_invariance(coordinates)

    assert receipt.poison_written
    assert receipt.source_deleted
    assert receipt.invariant
    assert receipt.sealed_machine_sha_before == (
        receipt.sealed_machine_sha_after
    )
    assert receipt.baseline_prediction_root == (
        receipt.poisoned_source_prediction_root
    )
    assert receipt.baseline_prediction_root == (
        receipt.deleted_source_prediction_root
    )
    assert not rehearsal.source_path.exists()
    assert sealed_copy.exists()


def test_source_disjoint_relation_assessor_does_not_call_protocol_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    latent = json.loads(rehearsal.latent_path.read_bytes())
    challenge_dir = (
        rehearsal.root
        / "challenges"
        / receipt.challenge_seed_commitment
    )
    coordinates = json.loads(
        challenge_dir.joinpath("abstract_coordinates.json").read_bytes()
    )
    committed_answers = tuple(
        json.loads(
            challenge_dir.joinpath("assessor_answers.json").read_bytes()
        )
    )

    def forbidden_protocol_path(*args: object, **kwargs: object) -> int:
        raise AssertionError("third assessor called a protocol execution path")

    monkeypatch.setattr(
        seal_protocol,
        "execute_sealed_machine",
        forbidden_protocol_path,
    )
    monkeypatch.setattr(
        seal_protocol,
        "assess_by_relation_composition",
        forbidden_protocol_path,
    )
    assert (
        _third_assessor_from_latent_relations(latent, coordinates)
        == committed_answers
    )


def test_relation_assessor_exhaustively_matches_sealed_machine(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    latent = json.loads(rehearsal.latent_path.read_bytes())
    coordinate_rows = [
        {
            "actions": list(actions),
            "observer": observer,
            "renderer": 0,
            "start": start,
            "world": 0,
        }
        for depth in range(5)
        for start in range(rehearsal.spec.state_count)
        for actions in product(
            range(rehearsal.spec.action_count),
            repeat=depth,
        )
        for observer in range(rehearsal.spec.observer_count)
    ]
    coordinates = tuple(
        AbstractCoordinate(
            int(row["world"]),
            int(row["start"]),
            tuple(int(action) for action in row["actions"]),
            int(row["observer"]),
            int(row["renderer"]),
        )
        for row in coordinate_rows
    )
    independent = _third_assessor_from_latent_relations(
        latent, coordinate_rows
    )
    deployed = tuple(
        seal_protocol.execute_sealed_machine(
            rehearsal.machine_path.read_bytes(),
            rehearsal.spec,
            coordinate,
        )
        for coordinate in coordinates
    )
    assert independent == deployed
    assert len(independent) == (
        rehearsal.spec.state_count
        * rehearsal.spec.observer_count
        * sum(
            rehearsal.spec.action_count**depth for depth in range(5)
        )
    )


def test_independent_transcript_assessor_recomputes_two_challenges(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    rehearsal.run_challenge(CHALLENGE_A)
    rehearsal.run_challenge(CHALLENGE_B)

    assessment = assess_rehearsal_transcript(rehearsal.root)

    assert assessment.passed
    assert assessment.challenge_count == 2
    assert assessment.independently_assessed_answers == 2 * sum(
        quota for _, quota in rehearsal.spec.depth_quotas
    )
    assert all(assessment.checks.values())


def test_kill_ordering_challenge_before_machine_or_before_world(
    tmp_path: Path,
) -> None:
    rehearsal = SealFirstRehearsal(tmp_path / "protocol")
    with pytest.raises(ProtocolViolation, match="machine root"):
        rehearsal.run_challenge(CHALLENGE_A)
    with pytest.raises(ProtocolViolation, match="world must be sealed"):
        rehearsal.seal_machine()
    rehearsal.supply_world_beacon(WORLD_BEACON)
    with pytest.raises(ProtocolViolation, match="machine root"):
        rehearsal.run_challenge(CHALLENGE_A)
    rehearsal.seal_machine()
    with pytest.raises(ProtocolViolation, match="strictly later"):
        rehearsal.run_challenge(
            Beacon(round=WORLD_BEACON.round, value="same-round")
        )
    with pytest.raises(ProtocolViolation, match="distinct"):
        rehearsal.run_challenge(
            Beacon(round=CHALLENGE_A.round, value=WORLD_BEACON.value)
        )


def test_kill_rng_coupling_attempts_and_challenge_world_feedback(
    tmp_path: Path,
) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    with pytest.raises(ProtocolViolation, match="unfrozen world stream"):
        derive_world_stream_seed(b"x" * 32, "challenge/coordinates")

    world_seed = derive_world_seed(rehearsal.protocol_root, WORLD_BEACON)
    challenge_seed = derive_challenge_seed(
        rehearsal.protocol_root,
        str(rehearsal.world_root),
        str(rehearsal.machine_root),
        CHALLENGE_A,
    )
    assert world_seed != challenge_seed
    assert all(
        derive_world_stream_seed(world_seed, label) != challenge_seed
        for label in rehearsal.spec.world_stream_labels
    )

    alternate_fixture = generate_consumed_world_fixture(
        rehearsal.spec,
        rehearsal.protocol_root,
        WORLD_BEACON,
    )
    assert alternate_fixture.evidence == (
        rehearsal.root / "sealed" / "world_evidence.copy"
    ).read_bytes()
    rehearsal.run_challenge(CHALLENGE_A)
    assert alternate_fixture.evidence == (
        rehearsal.root / "sealed" / "world_evidence.copy"
    ).read_bytes()


def test_kill_query_taint_in_world_or_latent_mechanics(
    tmp_path: Path,
) -> None:
    rehearsal = SealFirstRehearsal(tmp_path / "protocol")
    fixture = rehearsal.supply_world_beacon(WORLD_BEACON)
    tainted_evidence = json.loads(fixture.evidence)
    tainted_evidence["query"] = {
        "actions": [0, 1],
        "start": 0,
    }
    with pytest.raises(ProtocolViolation, match="query taint"):
        compile_world_evidence(
            canonical_json_bytes(tainted_evidence),
            rehearsal.spec,
        )

    tainted_latent = fixture.mechanics.canonical_dict()
    tainted_latent["target"] = 3
    with pytest.raises(ProtocolViolation, match="query taint"):
        WorldMechanics.from_mapping(tainted_latent, rehearsal.spec)


def test_kill_recompile_and_receipt_forgery(tmp_path: Path) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    with pytest.raises(ProtocolViolation, match="single-shot"):
        rehearsal.seal_machine()
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    with pytest.raises(ProtocolViolation, match="recompilation"):
        validate_challenge_receipt(
            replace(receipt, compile_count_after=2),
            rehearsal.spec,
        )
    with pytest.raises(ProtocolViolation, match="order"):
        validate_challenge_receipt(
            replace(
                receipt,
                coordinate_commit_event=receipt.key_render_event,
                key_render_event=receipt.coordinate_commit_event,
            ),
            rehearsal.spec,
        )


def test_kill_duplicate_or_quota_receipt_forgery(tmp_path: Path) -> None:
    rehearsal = _sealed_rehearsal(tmp_path)
    receipt = rehearsal.run_challenge(CHALLENGE_A)
    with pytest.raises(ProtocolViolation, match="duplicate"):
        validate_challenge_receipt(
            replace(receipt, duplicate_count=1),
            rehearsal.spec,
        )
    wrong_counts = (
        (receipt.realized_depth_counts[0][0], 1),
        *receipt.realized_depth_counts[1:],
    )
    with pytest.raises(ProtocolViolation, match="depth quota"):
        validate_challenge_receipt(
            replace(receipt, realized_depth_counts=wrong_counts),
            rehearsal.spec,
        )


def test_consumed_fixture_is_repeatable_but_not_an_official_board(
    tmp_path: Path,
) -> None:
    left = SealFirstRehearsal(tmp_path / "left")
    right = SealFirstRehearsal(tmp_path / "right")
    assert left.protocol_root == right.protocol_root
    left_fixture = left.supply_world_beacon(WORLD_BEACON)
    right_fixture = right.supply_world_beacon(WORLD_BEACON)
    assert left_fixture.evidence == right_fixture.evidence
    assert left_fixture.mechanics == right_fixture.mechanics
    assert left.spec.schema.endswith("rehearsal-v1")
    assert b"official" not in left_fixture.evidence.lower()
