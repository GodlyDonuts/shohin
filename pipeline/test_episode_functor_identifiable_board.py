from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import itertools

import pytest

from pipeline.episode_functor_identifiable_board import (
    ACTION_COUNT,
    FAMILIES,
    GrammarFactors,
    HELD_FACTOR_COMBINATIONS,
    IdentifiableBoardError,
    IdentifiableMachine,
    LateQuery,
    SOURCE_FACTOR_COMBINATIONS,
    TRAIN_FACTOR_COMBINATIONS,
    canonical_action_bytes,
    decode_query,
    decode_source,
    encode_query,
    encode_source,
    execute_query,
    generate_machine,
    generate_pilot_rows,
    hide_one_cell_per_relation,
    project_candidate_sources,
    resource_receipt,
    solve_unique_completion,
)
from pipeline.episode_functor_identifiable_reference import (
    enumerate_consistent_machines,
    version_space_receipt,
)


def _world(family: str = "affine-f2-3") -> IdentifiableMachine:
    return generate_machine(
        seed="efc-identifiable-test-v1",
        split="mechanics",
        index=FAMILIES.index(family),
        family=family,
    )


def test_every_action_family_is_nontrivial_and_future_separating() -> None:
    canonical = set()
    for family in FAMILIES:
        machine = _world(family)
        assert len(machine.transitions) == 3
        assert any(
            tuple(
                machine.transitions[right][machine.transitions[left][state]]
                for state in range(8)
            )
            != tuple(
                machine.transitions[left][machine.transitions[right][state]]
                for state in range(8)
            )
            for left in range(ACTION_COUNT)
            for right in range(left + 1, ACTION_COUNT)
        )
        canonical.add(sha256(machine.canonical_structural_bytes()).hexdigest())
    assert len(canonical) >= 4


def test_action_family_holdouts_have_no_shared_transition_orbits() -> None:
    by_family: dict[str, set[bytes]] = {}
    for family in FAMILIES:
        by_family[family] = {
            canonical_action_bytes(
                generate_machine(
                    seed="efc-family-disjoint-test-v1",
                    split="mechanics",
                    index=index,
                    family=family,
                ).transitions
            )
            for index in range(12)
        }
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1 :]:
            assert by_family[left].isdisjoint(by_family[right]), (left, right)


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_all_renderer_combinations_have_one_identical_version(
    values: tuple[int, int, int],
) -> None:
    machine = _world()
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-identifiable-test-v1",
        split="mechanics",
        index=17,
    )
    factors = GrammarFactors(*values)
    payload = encode_source(evidence, factors)
    decoded = decode_source(payload)
    assert decoded == evidence
    direct = solve_unique_completion(decoded)
    reference = enumerate_consistent_machines(payload)
    assert len(reference) == 1
    assert (
        direct.state_keys,
        direct.action_keys,
        direct.observer_keys,
        direct.transitions,
        direct.observations,
    ) == (
        reference[0].state_keys,
        reference[0].action_keys,
        reference[0].observer_keys,
        reference[0].transitions,
        reference[0].observations,
    )
    assert direct.canonical_structural_bytes() == machine.canonical_structural_bytes()
    receipt = version_space_receipt(payload, max_depth=4)
    assert receipt == {
        "behavior_classes": 1,
        "coordinates": 1_936,
        "version_space": 1,
    }
    assert b"renderer" not in payload.lower()
    assert b"family" not in payload.lower()


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_query_metagrammar_roundtrips_without_semantic_drift(
    values: tuple[int, int, int],
) -> None:
    machine = _world("cube-rotations")
    query = LateQuery(
        start_key=machine.state_keys[3],
        action_keys=(
            machine.action_keys[2],
            machine.action_keys[0],
            machine.action_keys[2],
            machine.action_keys[1],
        ),
        observer_key=machine.observer_keys[1],
    )
    payload = encode_query(query, GrammarFactors(*values))
    assert decode_query(payload) == query
    assert execute_query(machine, decode_query(payload)) == execute_query(
        machine,
        query,
    )
    assert b"renderer" not in payload.lower()


def test_empty_query_path_is_supported_in_both_organizations() -> None:
    machine = _world("dihedral-regular")
    query = LateQuery(
        start_key=machine.state_keys[5],
        action_keys=(),
        observer_key=machine.observer_keys[0],
    )
    for organization in range(2):
        factors = GrammarFactors(organization, organization, organization)
        assert decode_query(encode_query(query, factors)) == query


def test_canonical_form_quotients_slot_and_answer_recodings() -> None:
    machine = _world("quaternion-regular")
    old_for_new = (3, 7, 1, 5, 0, 6, 2, 4)
    old_to_new = {old: new for new, old in enumerate(old_for_new)}
    action_order = (2, 0, 1)
    observer_order = (1, 0)
    answer_recode = (2, 0, 3, 1)
    recoded = IdentifiableMachine(
        state_keys=tuple(machine.state_keys[old] for old in old_for_new),
        action_keys=tuple(machine.action_keys[old] for old in action_order),
        observer_keys=tuple(machine.observer_keys[old] for old in observer_order),
        transitions=tuple(
            tuple(
                old_to_new[machine.transitions[action][old]]
                for old in old_for_new
            )
            for action in action_order
        ),
        observations=tuple(
            tuple(
                answer_recode[machine.observations[observer][old]]
                for old in old_for_new
            )
            for observer in observer_order
        ),
    )
    assert recoded.canonical_structural_bytes() == machine.canonical_structural_bytes()


def test_missing_or_corrupted_constraints_fail_closed() -> None:
    machine = _world()
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-identifiable-test-v1",
        split="mechanics",
        index=29,
    )
    payload = encode_source(evidence, GrammarFactors(0, 0, 0))
    without_law = payload.replace(b"LAW-A PERMUTATION\n", b"")
    with pytest.raises(IdentifiableBoardError, match="completion law"):
        decode_source(without_law)
    with pytest.raises(IdentifiableBoardError, match="both laws"):
        enumerate_consistent_machines(without_law)

    transition = list(evidence.transition_events)
    action, source, _ = transition[0]
    transition[0] = (action, source, 1)
    corrupted = replace(evidence, transition_events=tuple(transition))
    corrupted_payload = encode_source(corrupted, GrammarFactors(1, 1, 1))
    with pytest.raises(IdentifiableBoardError):
        solve_unique_completion(decode_source(corrupted_payload))
    with pytest.raises(IdentifiableBoardError):
        enumerate_consistent_machines(corrupted_payload)


def test_labeled_field_reordering_is_rejected_by_both_parsers() -> None:
    machine = _world()
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-identifiable-test-v1",
        split="mechanics",
        index=31,
    )
    payload = encode_source(evidence, GrammarFactors(0, 1, 0))
    line = next(
        record
        for record in payload.splitlines()
        if record.startswith(b"T dst=")
    )
    parts = line.split()
    reordered = b" ".join((parts[0], parts[2], parts[1], parts[3]))
    mutated = payload.replace(line, reordered, 1)
    with pytest.raises(IdentifiableBoardError, match="order or names"):
        decode_source(mutated)
    with pytest.raises(IdentifiableBoardError, match="unknown record"):
        enumerate_consistent_machines(mutated)


def test_pilot_splits_are_deterministic_disjoint_and_factorized() -> None:
    counts = {
        "train": 12,
        "mechanics": 12,
        "development": 8,
        "confirmation": 6,
    }
    left = generate_pilot_rows(seed="efc-identifiable-pilot-v1", counts=counts)
    right = generate_pilot_rows(seed="efc-identifiable-pilot-v1", counts=counts)
    assert left == right
    expected_rows = 12 * 4 + 12 * 8 + 8 * 3 + 6
    assert len(left) == expected_rows
    assert len({row.canonical_sha256 for row in left}) == sum(counts.values())
    assert len({row.world_id for row in left}) == sum(counts.values())
    assert {
        row.factors.values for row in left if row.split == "train"
    }.issubset(set(TRAIN_FACTOR_COMBINATIONS))
    assert {
        row.factors.values for row in left if row.split == "development"
    }.issubset(set(HELD_FACTOR_COMBINATIONS))
    for row in left:
        reference = enumerate_consistent_machines(row.source)
        assert len(reference) == 1
        direct = solve_unique_completion(decode_source(row.source))
        assert (
            direct.state_keys,
            direct.action_keys,
            direct.observer_keys,
            direct.transitions,
            direct.observations,
        ) == (
            reference[0].state_keys,
            reference[0].action_keys,
            reference[0].observer_keys,
            reference[0].transitions,
            reference[0].observations,
        )
        assert direct.canonical_structural_bytes() == row.machine.canonical_structural_bytes()
    for world_id in {row.world_id for row in left}:
        orbit = [row for row in left if row.world_id == world_id]
        assert len({row.canonical_sha256 for row in orbit}) == 1
        split = orbit[0].split
        expected = {
            "train": 4,
            "mechanics": 8,
            "development": 3,
            "confirmation": 1,
        }[split]
        assert len(orbit) == expected

    candidate = project_candidate_sources(left, split="train")
    assert len(candidate) == 12 * 4
    assert tuple(field.name for field in fields(candidate[0])) == ("source",)
    assert all(
        marker not in row.source.lower()
        for row in candidate
        for marker in (b"renderer", b"family", b"split")
    )


def test_resource_receipt_exposes_large_answer_table_gap() -> None:
    receipt = resource_receipt(12)
    assert receipt["coordinates"] == 12_754_576
    assert receipt["deployed_machine_bits"] == 12_288
    assert receipt["answer_bits"] == 25_509_152
    assert receipt["answer_to_machine_ratio"] == pytest.approx(2_075.94, rel=1e-5)


def test_same_action_bag_can_require_different_answers() -> None:
    found = False
    for family in FAMILIES:
        machine = _world(family)
        for left, right in itertools.permutations(range(ACTION_COUNT), 2):
            for start, observer in itertools.product(range(8), range(2)):
                first = LateQuery(
                    machine.state_keys[start],
                    (machine.action_keys[left], machine.action_keys[right]),
                    machine.observer_keys[observer],
                )
                second = LateQuery(
                    machine.state_keys[start],
                    (machine.action_keys[right], machine.action_keys[left]),
                    machine.observer_keys[observer],
                )
                if execute_query(machine, first) != execute_query(machine, second):
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found
