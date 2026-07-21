from __future__ import annotations

import pytest

from closure_tied_action_algebra import (
    ActionPacket,
    all_copy_actions,
    all_states,
    apply_action,
    behavioral_signature,
    compose_actions,
    execute_packet,
    reindex_action,
    reindex_state,
    relabel_values,
)


def test_copy_actions_are_closed_noncommutative_and_associative() -> None:
    actions = all_copy_actions(3)
    states = all_states(3, 3)
    assert len(actions) == 27
    assert len(states) == 27
    assert apply_action((2, 2, 0), (0, 1, 2)) == (2, 2, 0)
    assert compose_actions((1, 2, 0), (2, 0, 0)) == (0, 0, 2)
    assert any(
        compose_actions(left, right) != compose_actions(right, left)
        for left in actions
        for right in actions
    )
    for first in actions:
        for second in actions:
            composed = compose_actions(second, first)
            for state in states:
                assert apply_action(composed, state) == apply_action(
                    second, apply_action(first, state)
                )
        for second in actions:
            for third in actions:
                assert compose_actions(
                    third, compose_actions(second, first)
                ) == compose_actions(compose_actions(third, second), first)


def test_alpha_and_storage_reindex_commute_with_actions() -> None:
    state = (0, 1, 2)
    action = (2, 2, 0)
    value_permutation = (2, 0, 1)
    position_permutation = (1, 2, 0)
    assert apply_action(
        action, relabel_values(state, value_permutation)
    ) == relabel_values(apply_action(action, state), value_permutation)
    transformed = reindex_action(action, position_permutation)
    assert apply_action(
        transformed, reindex_state(state, position_permutation)
    ) == reindex_state(apply_action(action, state), position_permutation)


def test_behavioral_basis_separates_every_state() -> None:
    identity = (0, 1, 2)
    actions = tuple(action for action in all_copy_actions(3) if action != identity)
    signatures = {
        behavioral_signature(state, actions, query_indices=(0,))
        for state in all_states(3, 3)
    }
    assert len(signatures) == 27


def test_halt_is_absorbing_and_source_free_packet_is_sufficient() -> None:
    actions = all_copy_actions(3)
    packet = ActionPacket(actions=actions[:8], halt_at=3)
    clean = execute_packet(packet, (0, 1, 2))
    poisoned_suffix = execute_packet(packet, (0, 1, 2), suffix=actions[8:16])
    assert poisoned_suffix.states[: len(clean.states)] == clean.states
    assert all(state == clean.states[3] for state in poisoned_suffix.states[3:])


def test_suffix_is_validated_even_after_halt() -> None:
    packet = ActionPacket(actions=((0, 1, 2),), halt_at=0)
    with pytest.raises(ValueError, match="copy action"):
        execute_packet(packet, (0, 1, 2), suffix=((3, 0, 1),))


def test_packet_and_initial_geometry_are_validated_before_halt() -> None:
    with pytest.raises(ValueError, match="action width"):
        ActionPacket(actions=((),), halt_at=0)
    packet = ActionPacket(actions=((0, 1, 2),), halt_at=0)
    with pytest.raises(ValueError, match="initial state width"):
        execute_packet(packet, (0, 1))
