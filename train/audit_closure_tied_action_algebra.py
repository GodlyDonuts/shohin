#!/usr/bin/env python3
"""Exhaustive CPU mechanics audit for closure-tied action algebra."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Sequence

from closure_tied_action_algebra import (
    ActionPacket,
    CopyAction,
    ExecutionTrace,
    State,
    all_copy_actions,
    all_states,
    apply_action,
    behavioral_signature,
    compose_actions,
    execute_packet,
    identity_action,
    position_permutations,
    reindex_action,
    reindex_state,
    relabel_values,
    value_permutations,
)


SCHEMA = "r12_closure_tied_action_algebra_cpu_audit_v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _oracle_apply(action: CopyAction, state: State) -> State:
    if len(action) != len(state) or any(
        index < 0 or index >= len(state) for index in action
    ):
        raise ValueError("oracle action differs")
    return tuple(state[index] for index in action)


def _oracle_compose(after: CopyAction, before: CopyAction) -> CopyAction:
    if len(after) != len(before):
        raise ValueError("oracle composition width differs")
    return tuple(before[index] for index in after)


def _oracle_reindex_action(
    action: CopyAction, permutation: Sequence[int]
) -> CopyAction:
    inverse = tuple(permutation.index(old_index) for old_index in range(len(action)))
    return tuple(inverse[action[old_index]] for old_index in permutation)


def _oracle_trace(
    actions: Sequence[CopyAction],
    halt_at: int,
    initial: State,
) -> tuple[tuple[State, ...], tuple[bool, ...]]:
    state = initial
    states = [state]
    halted = [halt_at == 0]
    for step, action in enumerate(actions):
        if step < halt_at:
            state = _oracle_apply(action, state)
        states.append(state)
        halted.append(step + 1 >= halt_at)
    return tuple(states), tuple(halted)


def audit(
    width: int = 3,
    alphabet: int = 3,
    *,
    apply_impl: Callable[[CopyAction, State], State] = apply_action,
    compose_impl: Callable[[CopyAction, CopyAction], CopyAction] = compose_actions,
    signature_impl: Callable[..., tuple[int, ...]] = behavioral_signature,
    execute_impl: Callable[..., ExecutionTrace] = execute_packet,
) -> dict[str, object]:
    actions = all_copy_actions(width)
    states = all_states(width, alphabet)
    identity = identity_action(width)

    atomic = 0
    closure = 0
    associative = 0
    noncommuting = 0
    for action in actions:
        for state in states:
            if apply_impl(action, state) != _oracle_apply(action, state):
                raise AssertionError(
                    "CTAA atomic action differs from independent oracle"
                )
            atomic += 1

    for before in actions:
        for after in actions:
            composed = compose_impl(after, before)
            oracle_composed = _oracle_compose(after, before)
            if composed != oracle_composed:
                raise AssertionError(
                    "CTAA closure card differs from independent oracle"
                )
            if compose_impl(identity, before) != before:
                raise AssertionError("CTAA left identity differs")
            if compose_impl(before, identity) != before:
                raise AssertionError("CTAA right identity differs")
            if oracle_composed != _oracle_compose(before, after):
                noncommuting += 1
            for state in states:
                expected = _oracle_apply(after, _oracle_apply(before, state))
                if apply_impl(composed, state) != expected:
                    raise AssertionError("CTAA closure execution differs")
                closure += 1

    for first in actions:
        for second in actions:
            for third in actions:
                left = compose_impl(third, compose_impl(second, first))
                right = compose_impl(compose_impl(third, second), first)
                expected_card = tuple(
                    first[second[third[index]]] for index in range(width)
                )
                if left != expected_card or right != expected_card:
                    raise AssertionError("CTAA associativity differs")
                for state in states:
                    expected = _oracle_apply(
                        third,
                        _oracle_apply(second, _oracle_apply(first, state)),
                    )
                    if apply_impl(left, state) != expected:
                        raise AssertionError("CTAA associative execution differs")
                    associative += 1

    alpha_checks = 0
    for permutation in value_permutations(alphabet):
        for action in actions:
            for state in states:
                relabeled = tuple(permutation[value] for value in state)
                left = apply_impl(action, relabel_values(state, permutation))
                right = tuple(
                    permutation[value] for value in _oracle_apply(action, state)
                )
                if left != right or relabel_values(state, permutation) != relabeled:
                    raise AssertionError("CTAA value alpha equivariance differs")
                alpha_checks += 1

    reindex_checks = 0
    for permutation in position_permutations(width):
        for action in actions:
            transformed = reindex_action(action, permutation)
            oracle_transformed = _oracle_reindex_action(action, permutation)
            if transformed != oracle_transformed:
                raise AssertionError("CTAA reindexed action differs from oracle")
            for state in states:
                permuted_state = tuple(state[index] for index in permutation)
                left = apply_impl(transformed, reindex_state(state, permutation))
                expected = tuple(
                    _oracle_apply(action, state)[index] for index in permutation
                )
                if (
                    left != expected
                    or reindex_state(state, permutation) != permuted_state
                ):
                    raise AssertionError("CTAA storage reindex differs")
                reindex_checks += 1

    continuations = tuple(action for action in actions if action != identity)
    signatures = {}
    for state in states:
        signature = signature_impl(state, continuations, query_indices=(0,))
        expected_signature = tuple(
            _oracle_apply(action, state)[0] for action in continuations
        )
        if signature != expected_signature:
            raise AssertionError("CTAA behavioral signature differs from oracle")
        signatures[state] = signature
    signature_values = tuple(signatures.values())
    min_signature_hamming = min(
        sum(left != right for left, right in zip(first, second, strict=True))
        for index, first in enumerate(signature_values)
        for second in signature_values[index + 1 :]
    )

    sample_actions: tuple[CopyAction, ...] = (
        (1, 0, 2),
        (2, 1, 0),
        (0, 2, 1),
        (1, 2, 0),
        (0, 0, 2),
        (2, 1, 1),
        (1, 0, 2),
        (2, 0, 1),
    )
    executable_action_checks = 0
    for action in actions:
        packet = ActionPacket((action,), halt_at=1)
        for initial in states:
            expected_states, expected_halted = _oracle_trace((action,), 1, initial)
            trace = execute_impl(packet, initial)
            if trace.states != expected_states or trace.halted != expected_halted:
                raise AssertionError("CTAA executable atomic action differs")
            executable_action_checks += 1

    halt_trace_checks = 0
    for halt_at in range(len(sample_actions) + 1):
        packet = ActionPacket(sample_actions, halt_at=halt_at)
        for initial in states:
            for suffix_action in actions:
                schedule = sample_actions + (suffix_action,)
                expected_states, expected_halted = _oracle_trace(
                    schedule, halt_at, initial
                )
                trace = execute_impl(packet, initial, suffix=(suffix_action,))
                if trace.states != expected_states or trace.halted != expected_halted:
                    raise AssertionError("CTAA absorbing halt trace differs")
                halt_trace_checks += 1

    prefix_actions: tuple[CopyAction, ...] = ((1, 0, 2), (2, 1, 0))
    suffix_actions: tuple[CopyAction, ...] = ((0, 0, 2), (2, 1, 0))
    prefix_packet = ActionPacket(prefix_actions, halt_at=len(prefix_actions))
    suffix_packet = ActionPacket(suffix_actions, halt_at=len(suffix_actions))
    donor_changed_checks = 0
    donor_same_terminal_shams = 0
    for recipient in states:
        recipient_middle = _oracle_trace(
            prefix_actions, len(prefix_actions), recipient
        )[0][-1]
        recipient_terminal = _oracle_trace(
            suffix_actions, len(suffix_actions), recipient_middle
        )[0][-1]
        actual_recipient_middle = execute_impl(prefix_packet, recipient).states[-1]
        if actual_recipient_middle != recipient_middle:
            raise AssertionError("CTAA recipient prefix differs from oracle")
        for donor in states:
            if donor == recipient:
                continue
            donor_middle = _oracle_trace(prefix_actions, len(prefix_actions), donor)[0][
                -1
            ]
            expected_states, expected_halted = _oracle_trace(
                suffix_actions, len(suffix_actions), donor_middle
            )
            intervened = execute_impl(suffix_packet, donor_middle)
            if (
                intervened.states != expected_states
                or intervened.halted != expected_halted
            ):
                raise AssertionError(
                    "CTAA donor suffix differs from independent oracle"
                )
            if intervened.states[-1] == recipient_terminal:
                donor_same_terminal_shams += 1
            else:
                donor_changed_checks += 1

    signature_parameters = tuple(inspect.signature(execute_packet).parameters)
    gates = {
        "all_atomic_cells_match_independent_oracle": atomic
        == len(actions) * len(states),
        "closure_cards_and_execution_match_independent_oracle": closure
        == len(actions) ** 2 * len(states),
        "associativity_matches_explicit_index_oracle": associative
        == len(actions) ** 3 * len(states),
        "ordered_noncommuting_pairs_exact": noncommuting == 588,
        "value_alpha_equivariance_exact": alpha_checks
        == len(value_permutations(alphabet)) * len(actions) * len(states),
        "storage_reindex_equivariance_exact": reindex_checks
        == len(position_permutations(width)) * len(actions) * len(states),
        "limited_query_basis_matches_oracle_and_separates": len(set(signature_values))
        == len(states)
        and all(len(value) == 26 for value in signature_values)
        and min_signature_hamming == 8,
        "every_atomic_action_executes_against_oracle": executable_action_checks == 729,
        "all_halt_boundaries_absorb_every_legal_suffix": halt_trace_checks == 6_561,
        "changed_donor_suffixes_follow_donor_oracle": donor_changed_checks == 648,
        "same_terminal_donor_shams_are_counted": donor_same_terminal_shams == 54,
        "packet_interface_excludes_source": signature_parameters
        == ("packet", "initial", "suffix"),
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "width": width,
        "alphabet": alphabet,
        "counts": {
            "actions": len(actions),
            "states": len(states),
            "atomic_action_state_cells": atomic,
            "action_pair_state_closure_checks": closure,
            "action_triple_state_associativity_checks": associative,
            "ordered_noncommuting_action_pairs": noncommuting,
            "value_alpha_checks": alpha_checks,
            "storage_reindex_checks": reindex_checks,
            "behavioral_signatures": len(set(signature_values)),
            "behavioral_signature_length": len(signature_values[0]),
            "minimum_signature_hamming": min_signature_hamming,
            "executable_atomic_action_checks": executable_action_checks,
            "halt_trace_checks": halt_trace_checks,
            "donor_changed_terminal_checks": donor_changed_checks,
            "donor_same_terminal_shams": donor_same_terminal_shams,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": (
            "Exact finite action-algebra and reference-interpreter mechanics only. "
            "Halt and source exclusion are properties of this CPU interface, not "
            "a learned model. This report contains no language grounding or "
            "reasoning capability."
        ),
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json(report).encode()
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--alphabet", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.width, args.alphabet)
    payload = canonical_json(report) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
