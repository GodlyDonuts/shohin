from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from pipeline.episode_functor_independent_world import (
    generate_independent_world,
)
from pipeline.episode_functor_multiworld_custody import (
    MultiworldCustodyRehearsal,
    MultiworldCustodySpec,
    canonical_structural_form,
    canonical_structural_signature,
    structural_overlap,
)
from pipeline.episode_functor_seal_protocol import (
    Beacon,
    ProtocolViolation,
    canonical_json_bytes,
)
from pipeline.episode_functor_wire_protocol import (
    WireProtocolSpec,
    decode_deployed_machine,
)


SPEC = MultiworldCustodySpec(
    train_worlds=4,
    development_worlds=2,
    confirmation_worlds=2,
)
OPEN_BEACON = Beacon(round=40_000, value="consumed-open-worlds")
CONFIRMATION_BEACON = Beacon(round=40_001, value="consumed-confirmation-worlds")
CANDIDATE_ROOT = "ab" * 32


def _open(root: Path) -> MultiworldCustodyRehearsal:
    rehearsal = MultiworldCustodyRehearsal(root, SPEC)
    rehearsal.freeze_open_splits(OPEN_BEACON)
    return rehearsal


def test_open_splits_are_balanced_and_structurally_disjoint(
    tmp_path: Path,
) -> None:
    rehearsal = MultiworldCustodyRehearsal(tmp_path / "open", SPEC)
    train, development = rehearsal.freeze_open_splits(OPEN_BEACON)
    assert len(train.records) == 4
    assert len(development.records) == 2
    assert structural_overlap((train, development)) == {"development__train": 0}
    assert {record.source_renderer for record in train.records} == {
        "canonical-json-events-v2",
        "strict-line-events-v1",
    }
    assert {record.source_renderer for record in development.records} == {
        "canonical-json-events-v2",
        "strict-line-events-v1",
    }
    assert all(
        1 < record.empty_observer_class_count < SPEC.state_count
        for record in (*train.records, *development.records)
    )
    assert all(
        record.future_behavior_class_count == SPEC.state_count
        for record in (*train.records, *development.records)
    )
    assert len(rehearsal.structural_signatures) == 6
    assert len(rehearsal.structural_forms) == 6


def test_confirmation_requires_open_splits_candidate_and_later_beacon(
    tmp_path: Path,
) -> None:
    rehearsal = MultiworldCustodyRehearsal(tmp_path / "guards", SPEC)
    with pytest.raises(ProtocolViolation, match="sealed first"):
        rehearsal.open_confirmation(CONFIRMATION_BEACON)
    rehearsal.freeze_open_splits(OPEN_BEACON)
    with pytest.raises(ProtocolViolation, match="sealed first"):
        rehearsal.open_confirmation(CONFIRMATION_BEACON)
    with pytest.raises(ProtocolViolation, match="SHA-256"):
        rehearsal.seal_candidate("not-a-root")
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    with pytest.raises(ProtocolViolation, match="strictly later"):
        rehearsal.open_confirmation(OPEN_BEACON)
    with pytest.raises(ProtocolViolation, match="repeats"):
        rehearsal.open_confirmation(Beacon(round=40_001, value=OPEN_BEACON.value))


def test_confirmation_is_disjoint_and_bound_to_candidate(
    tmp_path: Path,
) -> None:
    rehearsal = _open(tmp_path / "confirmation")
    open_root = rehearsal.open_splits_root
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    confirmation = rehearsal.open_confirmation(CONFIRMATION_BEACON)
    train = rehearsal.open_manifests["train"]
    development = rehearsal.open_manifests["development"]
    assert structural_overlap((train, development, confirmation)) == {
        "confirmation__development": 0,
        "confirmation__train": 0,
        "development__train": 0,
    }
    assert rehearsal.open_splits_root == open_root
    assert len(rehearsal.structural_signatures) == 8
    with pytest.raises(ProtocolViolation, match="already been opened"):
        rehearsal.open_confirmation(Beacon(round=40_002, value="second-confirmation"))


def test_same_inputs_reproduce_all_roots(tmp_path: Path) -> None:
    first = _open(tmp_path / "first")
    second = _open(tmp_path / "second")
    assert first.protocol_root == second.protocol_root
    assert first.open_splits_root == second.open_splits_root
    first.seal_candidate(CANDIDATE_ROOT)
    second.seal_candidate(CANDIDATE_ROOT)
    first_confirmation = first.open_confirmation(CONFIRMATION_BEACON)
    second_confirmation = second.open_confirmation(CONFIRMATION_BEACON)
    assert first_confirmation == second_confirmation


def test_changed_candidate_or_beacon_changes_confirmation_root(
    tmp_path: Path,
) -> None:
    first = _open(tmp_path / "first-changed")
    second = _open(tmp_path / "second-changed")
    third = _open(tmp_path / "third-changed")
    first.seal_candidate(CANDIDATE_ROOT)
    second.seal_candidate("cd" * 32)
    third.seal_candidate(CANDIDATE_ROOT)
    first_confirmation = first.open_confirmation(CONFIRMATION_BEACON)
    second_confirmation = second.open_confirmation(CONFIRMATION_BEACON)
    third_confirmation = third.open_confirmation(
        Beacon(round=40_002, value="different-confirmation")
    )
    assert first.open_splits_root == second.open_splits_root
    assert first.open_splits_root == third.open_splits_root
    assert first_confirmation.manifest_root != (second_confirmation.manifest_root)
    assert first_confirmation.manifest_root != (third_confirmation.manifest_root)
    first_machines = {record.machine_sha256 for record in first_confirmation.records}
    assert first_machines != {
        record.machine_sha256 for record in second_confirmation.records
    }
    assert first_machines != {
        record.machine_sha256 for record in third_confirmation.records
    }


def test_structural_signature_is_gauge_and_answer_recode_invariant() -> None:
    world = generate_independent_world(
        protocol_root="78" * 32,
        beacon_round=50_000,
        beacon_value="signature-world",
        state_count=5,
        action_count=3,
        observer_count=2,
        answer_count=5,
        renderer_count=1,
    )
    baseline = canonical_structural_signature(world.transitions, world.observers)
    baseline_form = canonical_structural_form(world.transitions, world.observers)
    gauge = (3, 0, 4, 1, 2)
    inverse = [0] * 5
    for old, new in enumerate(gauge):
        inverse[new] = old
    changed_transitions = tuple(
        tuple(
            gauge[world.transitions[old_action][inverse[new_state]]]
            for new_state in range(5)
        )
        for old_action in (2, 0, 1)
    )
    changed_observations = tuple(
        tuple(
            world.observers[old_observer][inverse[new_state]] ^ flip
            for new_state in range(5)
        )
        for old_observer, flip in ((1, 1), (0, 0))
    )
    assert (
        canonical_structural_signature(changed_transitions, changed_observations)
        == baseline
    )
    assert (
        canonical_structural_form(changed_transitions, changed_observations)
        == baseline_form
    )

    changed_cell = [list(row) for row in changed_transitions]
    changed_cell[0][0] = (changed_cell[0][0] + 1) % 5
    assert (
        canonical_structural_signature(
            tuple(tuple(row) for row in changed_cell),
            changed_observations,
        )
        != baseline
    )
    with pytest.raises(ProtocolViolation, match="outside frozen cell"):
        canonical_structural_form((), ())


def test_confirmation_beacon_is_absent_before_candidate_seal(
    tmp_path: Path,
) -> None:
    rehearsal = _open(tmp_path / "unopened")
    preseal_files = tuple(
        path.relative_to(rehearsal.root).as_posix()
        for path in rehearsal.root.rglob("*")
        if path.is_file()
    )
    preseal_payload = b"".join(
        path.read_bytes() for path in rehearsal.root.rglob("*") if path.is_file()
    )
    assert not any("confirmation" in path for path in preseal_files)
    assert CONFIRMATION_BEACON.value.encode("ascii") not in preseal_payload
    assert CANDIDATE_ROOT.encode("ascii") not in preseal_payload


def test_protocol_mutation_and_repeated_phases_fail_closed(
    tmp_path: Path,
) -> None:
    rehearsal = MultiworldCustodyRehearsal(tmp_path / "mutation", SPEC)
    rehearsal.root.joinpath("protocol.json").write_text("{}\n", encoding="ascii")
    with pytest.raises(ProtocolViolation, match="changed"):
        rehearsal.freeze_open_splits(OPEN_BEACON)

    repeated = _open(tmp_path / "repeated")
    with pytest.raises(ProtocolViolation, match="already been frozen"):
        repeated.freeze_open_splits(OPEN_BEACON)
    repeated.seal_candidate(CANDIDATE_ROOT)
    with pytest.raises(ProtocolViolation, match="already been sealed"):
        repeated.seal_candidate("cd" * 32)


def test_persisted_world_receipts_bind_every_payload(tmp_path: Path) -> None:
    rehearsal = _open(tmp_path / "payload-receipts")
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    confirmation = rehearsal.open_confirmation(CONFIRMATION_BEACON)
    manifests = (
        rehearsal.open_manifests["train"],
        rehearsal.open_manifests["development"],
        confirmation,
    )
    for manifest in manifests:
        for record in manifest.records:
            world_root = (
                rehearsal.root / "worlds" / record.split / f"{record.ordinal:04d}"
            )
            payload_hashes = {
                "admissibility_sha256": "admissibility.json",
                "evidence_sha256": "evidence.bin",
                "latent_sha256": "latent.json",
                "machine_sha256": "machine.bin",
                "stream_commitments_sha256": "stream_commitments.json",
            }
            for receipt_field, filename in payload_hashes.items():
                assert sha256(
                    world_root.joinpath(filename).read_bytes()
                ).hexdigest() == (getattr(record, receipt_field))
            machine = world_root.joinpath("machine.bin").read_bytes()
            assert machine[-32:].hex() == record.machine_payload_sha256
            tables = decode_deployed_machine(machine, WireProtocolSpec())
            structural_form = canonical_structural_form(
                tables.transitions,
                tables.observations,
            )
            assert structural_form.hex() == record.structural_canonical_hex
            assert sha256(structural_form).hexdigest() == (
                record.structural_signature
            )
            assert (
                json.loads(world_root.joinpath("receipt.json").read_bytes())
                == record.canonical_dict()
            )
    rehearsal.verify_published_state()


def test_event_files_form_exact_sha256_chain(tmp_path: Path) -> None:
    rehearsal = _open(tmp_path / "event-chain")
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    rehearsal.open_confirmation(CONFIRMATION_BEACON)
    previous = bytes(32).hex()
    for event_id, path in enumerate(
        sorted(rehearsal.root.joinpath("events").glob("*.json")),
        start=1,
    ):
        row = json.loads(path.read_bytes())
        assert row["event_id"] == event_id
        assert row["previous_event_sha256"] == previous
        previous = sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "target",
    [
        "train_manifest.json",
        "open_splits_root.txt",
        "worlds/train/0000/evidence.bin",
        "worlds/train/0000/latent.json",
        "worlds/train/0000/machine.bin",
        "worlds/train/0000/receipt.json",
        "events/000002.json",
    ],
)
def test_open_state_tampering_blocks_candidate_seal(
    tmp_path: Path,
    target: str,
) -> None:
    rehearsal = _open(tmp_path / target.replace("/", "_"))
    path = rehearsal.root / target
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ProtocolViolation, match="changed|invalid"):
        rehearsal.seal_candidate(CANDIDATE_ROOT)


def test_candidate_seal_tampering_blocks_confirmation(tmp_path: Path) -> None:
    rehearsal = _open(tmp_path / "candidate-tamper")
    rehearsal.seal_candidate(CANDIDATE_ROOT)
    seal = rehearsal.root / "candidate_seal.json"
    seal.write_bytes(seal.read_bytes() + b"tamper")
    with pytest.raises(ProtocolViolation, match="candidate seal changed"):
        rehearsal.open_confirmation(CONFIRMATION_BEACON)


def test_source_renderer_receipt_and_machine_are_revalidated(tmp_path: Path) -> None:
    rehearsal = _open(tmp_path / "source-revalidation")
    manifest = rehearsal.open_manifests["train"]
    line_record = next(
        record
        for record in manifest.records
        if record.source_renderer == "strict-line-events-v1"
    )
    object.__setattr__(line_record, "source_renderer", "canonical-json-events-v2")
    receipt = (
        rehearsal.root
        / "worlds"
        / line_record.split
        / f"{line_record.ordinal:04d}"
        / "receipt.json"
    )
    receipt.write_bytes(canonical_json_bytes(line_record.canonical_dict()))
    with pytest.raises(ProtocolViolation, match="semantic receipt is inconsistent"):
        rehearsal._verify_record(
            line_record,
            beacon=OPEN_BEACON,
            candidate_root=None,
            open_splits_root=None,
        )


def test_semantic_payloads_are_regenerated_from_recorded_beacon(
    tmp_path: Path,
) -> None:
    rehearsal = _open(tmp_path / "semantic-revalidation")
    record = rehearsal.open_manifests["train"].records[0]
    world_root = (
        rehearsal.root / "worlds" / record.split / f"{record.ordinal:04d}"
    )
    latent = json.loads(world_root.joinpath("latent.json").read_bytes())
    latent["transition_relations"][0][0] = (
        latent["transition_relations"][0][0] + 1
    ) % SPEC.state_count
    latent_payload = canonical_json_bytes(latent)
    forged_record = replace(
        record,
        latent_sha256=sha256(latent_payload).hexdigest(),
    )
    world_root.joinpath("latent.json").write_bytes(latent_payload)
    world_root.joinpath("receipt.json").write_bytes(
        canonical_json_bytes(forged_record.canonical_dict())
    )
    with pytest.raises(ProtocolViolation, match="semantic receipt is inconsistent"):
        rehearsal._verify_record(
            forged_record,
            beacon=OPEN_BEACON,
            candidate_root=None,
            open_splits_root=None,
        )


@pytest.mark.parametrize(
    "target",
    [
        "hidden-answer.txt",
        "events/hidden-answer.txt",
        "worlds/hidden/answer.txt",
        "worlds/train/0000/hidden/answer.txt",
    ],
)
def test_extra_files_or_directories_block_candidate_seal(
    tmp_path: Path,
    target: str,
) -> None:
    rehearsal = _open(tmp_path / target.replace("/", "_"))
    path = rehearsal.root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("forbidden\n", encoding="ascii")
    with pytest.raises(ProtocolViolation, match="file set|directory set"):
        rehearsal.seal_candidate(CANDIDATE_ROOT)


def test_generation_attempt_must_be_first_structurally_admissible_world(
    tmp_path: Path,
) -> None:
    rehearsal = _open(tmp_path / "attempt-order")
    first = rehearsal.open_manifests["train"].records[0]
    object.__setattr__(first, "generation_attempt", 1)
    with pytest.raises(ProtocolViolation, match="skipped an admissible attempt"):
        rehearsal._verify_structural_sequence(include_confirmation=False)
