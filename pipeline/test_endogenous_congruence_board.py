from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
from itertools import product

import pytest

from pipeline.endogenous_congruence_board import (
    BoardCapacityError,
    CongruenceInvariantError,
    EndogenousCongruencePacket,
    EpisodeAmbiguityError,
    EpisodeUnderidentifiedError,
    ObservationWitness,
    TransitionWitness,
    audit_collision_orbit,
    build_congruence_collision_orbit,
    compute_exhaustive_partition,
    compute_refinement_partition,
    model_packet_payload,
    path_equivalent,
    solve_with_independent_crosscheck,
    validate_candidate_partition,
    validate_descent,
    validate_merge_certificate,
    validate_packet,
    validate_path_partition,
    validate_presentation_naturality,
)


FORBIDDEN_MODEL_KEYS = {
    "answer",
    "expected",
    "family",
    "label",
    "oracle",
    "schedule",
}


def _packet_from_functions(
    size: int,
    transition_functions: tuple[tuple[int, ...], ...],
    observations: tuple[int, ...],
) -> EndogenousCongruencePacket:
    records = tuple(f"r_{index}" for index in range(size))
    generators = tuple(f"g_{index}" for index in range(len(transition_functions)))
    queries = ("q_0",)
    return EndogenousCongruencePacket(
        records=records,
        generators=generators,
        query_ports=queries,
        transition_witnesses=tuple(
            TransitionWitness(records[source], generator, records[target])
            for generator, function in zip(
                generators,
                transition_functions,
                strict=True,
            )
            for source, target in enumerate(function)
        ),
        observation_witnesses=tuple(
            ObservationWitness(record, "q_0", value)
            for record, value in zip(records, observations, strict=True)
        ),
    )


def _all_functions(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(size), repeat=size))


def _equivalence(blocks: tuple[tuple[str, ...], ...]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(block) for block in blocks)


def _word_index(solution: object) -> dict[tuple[str, ...], int]:
    path = solution.path_congruence
    return {word: index for index, word in enumerate(path.words)}


def test_source_deleted_model_packet_has_only_physical_witnesses() -> None:
    packet = build_congruence_collision_orbit().base
    field_names = {field.name for field in fields(EndogenousCongruencePacket)}
    assert not field_names & FORBIDDEN_MODEL_KEYS
    payload = model_packet_payload(packet)
    assert set(payload) == {
        "records",
        "generators",
        "query_ports",
        "transition_witnesses",
        "observation_witnesses",
    }
    assert not set(payload) & FORBIDDEN_MODEL_KEYS
    assert all(
        isinstance(item, tuple) and len(item) == 3
        for item in payload["transition_witnesses"]
    )
    assert all(
        isinstance(item, tuple) and len(item) == 3
        for item in payload["observation_witnesses"]
    )


def test_two_independent_algorithms_agree_on_all_seven_orbit_presentations() -> None:
    orbit = build_congruence_collision_orbit()
    packets = (
        orbit.base,
        orbit.reindexed,
        orbit.split_bisimilar,
        orbit.merged,
        orbit.minimal_noncongruent,
        orbit.commuting_path_twin,
        orbit.noncommuting_twin,
    )
    for packet in packets:
        refinement = compute_refinement_partition(packet)
        exhaustive = compute_exhaustive_partition(packet)
        assert _equivalence(refinement) == _equivalence(exhaustive)
        solve_with_independent_crosscheck(packet)


def test_exhaustive_small_n_agreement_covers_296_complete_episodes() -> None:
    cases = 0
    for size, generator_count in ((2, 1), (3, 1), (2, 2)):
        functions = _all_functions(size)
        for selected_functions in product(functions, repeat=generator_count):
            for observations in product((0, 1), repeat=size):
                packet = _packet_from_functions(
                    size,
                    tuple(selected_functions),
                    tuple(observations),
                )
                assert _equivalence(
                    compute_refinement_partition(packet)
                ) == _equivalence(compute_exhaustive_partition(packet))
                cases += 1
    assert cases == 296


def test_descent_factorization_and_all_certificates_are_executable() -> None:
    packet = build_congruence_collision_orbit().base
    solution = solve_with_independent_crosscheck(packet)
    assert len(solution.blocks) == 4
    assert len(solution.merge_certificates) == 2
    assert len(solution.distinction_certificates) == 6
    assert any(
        certificate.continuation for certificate in solution.distinction_certificates
    )
    for certificate in solution.merge_certificates:
        validate_merge_certificate(packet, certificate)
    validate_descent(solution)
    assert all(sum(row) == 1 for row in solution.quotient)


def test_shortest_distinction_continuations_reject_shorter_words() -> None:
    packet = build_congruence_collision_orbit().base
    solution = solve_with_independent_crosscheck(packet)
    delayed = [
        certificate
        for certificate in solution.distinction_certificates
        if certificate.continuation
    ]
    assert delayed
    assert min(len(item.continuation) for item in delayed) == 1


def test_reindex_and_nonbijective_split_merge_naturality() -> None:
    orbit = build_congruence_collision_orbit()
    reindex = validate_presentation_naturality(
        orbit.base,
        orbit.reindexed,
        orbit.base_to_reindexed,
    )
    split = validate_presentation_naturality(
        orbit.split_bisimilar,
        orbit.base,
        orbit.split_to_base,
    )
    merge = validate_presentation_naturality(
        orbit.base,
        orbit.merged,
        orbit.base_to_merged,
    )
    assert sorted(reindex.source_to_target_class) == [0, 1, 2, 3]
    assert sorted(split.source_to_target_class) == [0, 1, 2, 3]
    assert sorted(merge.source_to_target_class) == [0, 1, 2, 3]


def test_collision_twins_force_split_and_separate_path_equations() -> None:
    orbit = build_congruence_collision_orbit()
    base = solve_with_independent_crosscheck(orbit.base)
    changed = solve_with_independent_crosscheck(orbit.minimal_noncongruent)
    commuting = solve_with_independent_crosscheck(orbit.commuting_path_twin)
    noncommuting = solve_with_independent_crosscheck(orbit.noncommuting_twin)
    assert len(changed.blocks) == len(base.blocks) + 1
    left, right = orbit.commuting_pair
    assert path_equivalent(commuting.path_congruence, left, right)
    assert not path_equivalent(noncommuting.path_congruence, left, right)


def test_bounded_path_congruence_is_closed_under_all_fitting_contexts() -> None:
    packet = build_congruence_collision_orbit().base
    path = solve_with_independent_crosscheck(packet).path_congruence
    index = {word: position for position, word in enumerate(path.words)}
    for left, right in product(path.words, repeat=2):
        if path.class_assignment[index[left]] != path.class_assignment[index[right]]:
            continue
        remaining = path.max_depth - max(len(left), len(right))
        for prefix_depth in range(remaining + 1):
            for prefix in product(packet.generators, repeat=prefix_depth):
                suffix_room = remaining - prefix_depth
                for suffix_depth in range(suffix_room + 1):
                    for suffix in product(packet.generators, repeat=suffix_depth):
                        contextual_left = (*prefix, *left, *suffix)
                        contextual_right = (*prefix, *right, *suffix)
                        assert (
                            path.class_assignment[index[contextual_left]]
                            == path.class_assignment[index[contextual_right]]
                        )


def test_merge_all_identity_no_descent_and_no_path_controls_die() -> None:
    orbit = build_congruence_collision_orbit()
    packet = orbit.base
    solution = solve_with_independent_crosscheck(packet)
    with pytest.raises(CongruenceInvariantError):
        validate_candidate_partition(packet, (packet.records,))
    with pytest.raises(CongruenceInvariantError):
        validate_candidate_partition(
            packet,
            tuple((record,) for record in packet.records),
        )

    corrupted = list(solution.induced_generators)
    first = [list(row) for row in corrupted[0]]
    first[0], first[1] = first[1], first[0]
    corrupted[0] = tuple(tuple(row) for row in first)
    with pytest.raises(CongruenceInvariantError):
        validate_descent(solution, induced_override=corrupted)

    identity_path = tuple(range(len(solution.path_congruence.words)))
    with pytest.raises(CongruenceInvariantError):
        validate_path_partition(solution.path_congruence, identity_path)


def test_false_merge_certificate_is_rejected() -> None:
    orbit = build_congruence_collision_orbit()
    solution = solve_with_independent_crosscheck(orbit.base)
    certificate = solution.merge_certificates[0]
    false_certificate = replace(
        certificate,
        relation=((certificate.left_record, "x_11"),),
    )
    with pytest.raises(CongruenceInvariantError):
        validate_merge_certificate(orbit.base, false_certificate)


def test_missing_duplicate_conflicting_and_overcapacity_packets_fail_closed() -> None:
    packet = build_congruence_collision_orbit().base
    with pytest.raises(EpisodeUnderidentifiedError):
        validate_packet(
            replace(
                packet,
                transition_witnesses=packet.transition_witnesses[:-1],
            )
        )
    with pytest.raises(EpisodeAmbiguityError):
        validate_packet(
            replace(
                packet,
                transition_witnesses=(
                    *packet.transition_witnesses,
                    packet.transition_witnesses[0],
                ),
            )
        )
    first = packet.observation_witnesses[0]
    with pytest.raises(EpisodeAmbiguityError):
        validate_packet(
            replace(
                packet,
                observation_witnesses=(
                    *packet.observation_witnesses,
                    replace(first, value=1 - first.value),
                ),
            )
        )
    too_many = tuple(f"overflow_{index}" for index in range(9))
    with pytest.raises(BoardCapacityError):
        validate_packet(
            EndogenousCongruencePacket(
                records=too_many,
                generators=("g",),
                query_ports=("q",),
                transition_witnesses=tuple(
                    TransitionWitness(record, "g", record) for record in too_many
                ),
                observation_witnesses=tuple(
                    ObservationWitness(record, "q", 0) for record in too_many
                ),
            )
        )


def test_orbit_marginals_and_full_audit_receipt() -> None:
    orbit = build_congruence_collision_orbit()
    same_size = (
        orbit.base,
        orbit.minimal_noncongruent,
        orbit.commuting_path_twin,
        orbit.noncommuting_twin,
    )
    assert {
        (
            len(packet.records),
            len(packet.generators),
            len(packet.query_ports),
            len(packet.transition_witnesses),
            len(packet.observation_witnesses),
            tuple(sorted(item.value for item in packet.observation_witnesses)),
        )
        for packet in same_size
    } == {(6, 3, 2, 18, 12, (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1))}

    def degree_histograms(
        packet: EndogenousCongruencePacket,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        incoming = Counter(item.target for item in packet.transition_witnesses)
        outgoing = Counter(item.source for item in packet.transition_witnesses)
        return (
            tuple(sorted(incoming[record] for record in packet.records)),
            tuple(sorted(outgoing[record] for record in packet.records)),
        )

    assert len({degree_histograms(packet) for packet in same_size}) == 1
    base_transitions = {
        (item.source, item.generator): item.target
        for item in orbit.base.transition_witnesses
    }
    for twin in (orbit.minimal_noncongruent, orbit.noncommuting_twin):
        twin_transitions = {
            (item.source, item.generator): item.target
            for item in twin.transition_witnesses
        }
        assert (
            sum(
                base_transitions[key] != twin_transitions[key]
                for key in base_transitions
            )
            == 2
        )

    receipt = audit_collision_orbit(orbit)
    assert receipt == {
        "presentations": 7,
        "physical_records": 42,
        "transition_witnesses": 126,
        "observation_witnesses": 84,
        "merge_certificates": 11,
        "distinction_certificates": 55,
        "independent_oracles_agree": True,
        "split_naturality": True,
        "merge_naturality": True,
        "commuting_separated": True,
    }
