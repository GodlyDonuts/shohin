from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.audit_episode_functor_identifiable_board import DEFAULT_COUNTS
from pipeline.episode_functor_hankel_shift import (
    HankelCodebook,
    HankelShiftError,
    build_hankel_codebook,
    commutative_bag_incidence,
    decode_hankel_shifts,
    derivative_only_correction_radius,
    enumerate_action_words,
    exact_codebook_receipt,
    joint_codebook_correction_radius,
    minimum_signature_distance,
    prefix_shift_incidence,
    random_shift_incidence,
)
from pipeline.episode_functor_identifiable_board import (
    ACTION_COUNT,
    IdentifiableMachine,
    STATE_COUNT,
    generate_machine,
    generate_pilot_rows,
)


def _machine() -> IdentifiableMachine:
    return generate_machine(
        seed="efc-hankel-shift-test-v1",
        split="mechanics",
        index=0,
        family="cube-rotations",
    )


def test_prefix_incidence_matches_left_to_right_execution() -> None:
    machine = _machine()
    codebook = build_hankel_codebook(machine, max_depth=3)
    assert codebook.incidence == prefix_shift_incidence(3)
    assert len(codebook.words) == 40
    assert codebook.coordinate_count == 80
    for action in range(ACTION_COUNT):
        for state in range(STATE_COUNT):
            target = machine.transitions[action][state]
            assert codebook.derivative[action][state] == codebook.base[target]


def test_exact_shift_decode_recovers_the_complete_machine() -> None:
    machine = _machine()
    decoded = decode_hankel_shifts(
        build_hankel_codebook(machine, max_depth=3)
    )
    assert decoded.transitions == machine.transitions
    assert decoded.observations == machine.observations
    assert all(
        syndrome == 0
        for row in decoded.syndromes
        for syndrome in row
    )


def test_frozen_200_world_board_has_audited_depth_three_margin() -> None:
    rows = generate_pilot_rows(
        seed="efc-identifiable-pilot-v1",
        counts=DEFAULT_COUNTS,
    )
    machines = tuple(
        {
            row.world_id: row.machine
            for row in rows
        }.values()
    )
    assert exact_codebook_receipt(machines, max_depth=0) == {
        "coordinate_count": 2,
        "derivative_only_radius": -1,
        "joint_codebook_radius": -1,
        "maximum_minimum_distance": 1,
        "minimum_distance": 0,
        "separated_machines": 112,
        "world_count": 200,
    }
    assert exact_codebook_receipt(machines, max_depth=1)[
        "separated_machines"
    ] == 200
    assert exact_codebook_receipt(machines, max_depth=3) == {
        "coordinate_count": 80,
        "derivative_only_radius": 15,
        "joint_codebook_radius": 7,
        "maximum_minimum_distance": 66,
        "minimum_distance": 32,
        "separated_machines": 200,
        "world_count": 200,
    }


def test_random_incidence_is_length_matched_bijective_and_noncausal() -> None:
    true = prefix_shift_incidence(3)
    random_left = random_shift_incidence(3, seed="random-control-v1")
    random_right = random_shift_incidence(3, seed="random-control-v1")
    assert random_left == random_right
    assert random_left != true
    words = enumerate_action_words(3)
    extended = enumerate_action_words(4)
    for length in range(4):
        targets = [
            random_left[action][index]
            for action in range(ACTION_COUNT)
            for index, word in enumerate(words)
            if len(word) == length
        ]
        assert len(targets) == len(set(targets))
        assert all(len(extended[target]) == length + 1 for target in targets)

    machine = _machine()
    random_decode = decode_hankel_shifts(
        build_hankel_codebook(
            machine,
            max_depth=3,
            incidence=random_left,
        ),
        require_unique=False,
    )
    assert random_decode.transitions != machine.transitions


def test_stable_bag_control_erases_repeated_symbol_interleaving() -> None:
    incidence = commutative_bag_incidence(3)
    words = enumerate_action_words(3)
    extended = enumerate_action_words(4)
    left = incidence[0][words.index((1, 0))]
    right = incidence[0][words.index((0, 1))]
    assert extended[left] == extended[right] == (0, 0, 1)


def test_correction_radii_follow_strict_triangle_inequality() -> None:
    codebook = build_hankel_codebook(_machine(), max_depth=3)
    distance = minimum_signature_distance(codebook)
    assert derivative_only_correction_radius(codebook) == (distance - 1) // 2
    assert joint_codebook_correction_radius(codebook) == (distance - 1) // 4
    assert 2 * derivative_only_correction_radius(codebook) < distance
    assert 4 * joint_codebook_correction_radius(codebook) < distance


def test_tied_or_malformed_codes_fail_closed() -> None:
    codebook = build_hankel_codebook(_machine(), max_depth=1)
    tied_base = (codebook.base[0], codebook.base[0], *codebook.base[2:])
    with pytest.raises(HankelShiftError, match="coordinate-dependent tie"):
        decode_hankel_shifts(replace(codebook, base=tied_base))

    with pytest.raises(HankelShiftError, match="geometry"):
        HankelCodebook(
            depth=1,
            words=codebook.words,
            base=codebook.base,
            derivative=codebook.derivative,
            incidence=tuple((0,) for _ in range(ACTION_COUNT)),
        )
    with pytest.raises(HankelShiftError, match="word-length stratum"):
        HankelCodebook(
            depth=1,
            words=codebook.words,
            base=codebook.base,
            derivative=codebook.derivative,
            incidence=tuple(
                tuple(0 for _ in codebook.words)
                for _ in range(ACTION_COUNT)
            ),
        )
