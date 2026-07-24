from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_constrained_transport import (  # noqa: E402
    LawfulMachineProjector,
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
    hard_assign_keys,
    project_key_assignment_logits,
)
from episode_functor_machine import (  # noqa: E402
    HardFunctorQuery,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorQuery,
    execute_hard,
    execute_soft,
)
from episode_functor_pointer_compiler import (  # noqa: E402
    MAX_UNIQUE_KEYS,
    collate_sources,
    scan_source,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    decode_source,
    generate_pilot_rows,
    solve_unique_completion,
)


def _oracle_logits(rows):
    transition_logits = torch.zeros(
        len(rows),
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    observer_logits = torch.zeros(
        len(rows),
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    for row_index, row in enumerate(rows):
        evidence = decode_source(row.source)
        states = tuple(sorted(evidence.state_keys))
        actions = tuple(
            sorted({event[0] for event in evidence.transition_events})
        )
        observers = tuple(
            sorted({event[0] for event in evidence.observation_events})
        )
        state_index = {key: index for index, key in enumerate(states)}
        action_index = {key: index for index, key in enumerate(actions)}
        observer_index = {
            key: index for index, key in enumerate(observers)
        }
        for action, source, destination in evidence.transition_events:
            transition_logits[
                row_index,
                action_index[action],
                state_index[source],
                state_index[destination],
            ] = 40.0
        for observer, state, answer in evidence.observation_events:
            observer_logits[
                row_index,
                observer_index[observer],
                state_index[state],
                answer,
            ] = 40.0
    return transition_logits, observer_logits


def test_soft_projection_enforces_public_laws_and_has_gradients() -> None:
    torch.manual_seed(43)
    transition_logits = torch.randn(
        2,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
        requires_grad=True,
    )
    observer_logits = torch.randn(
        2,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
        requires_grad=True,
    )
    projection = LawfulMachineProjector()(transition_logits, observer_logits)
    assert torch.allclose(
        projection.transition_transport.sum(-1),
        torch.ones(2, PRIMARY_ACTIONS, PRIMARY_STATES),
        atol=2e-5,
    )
    assert torch.allclose(
        projection.transition_transport.sum(-2),
        torch.ones(2, PRIMARY_ACTIONS, PRIMARY_STATES),
        atol=2e-5,
    )
    assert torch.allclose(
        projection.observer_transport.sum(-1),
        torch.ones(2, PRIMARY_OBSERVERS, PRIMARY_STATES),
        atol=2e-5,
    )
    assert torch.allclose(
        projection.observer_transport.sum(-2),
        torch.full((2, PRIMARY_OBSERVERS, PRIMARY_ANSWERS), 2.0),
        atol=2e-5,
    )
    loss = (
        projection.transition_transport.square().mean()
        + projection.observer_transport.square().mean()
        + projection.machine.action_next.square().mean()
        + projection.machine.observer_answer.square().mean()
    )
    loss.backward()
    assert transition_logits.grad is not None
    assert observer_logits.grad is not None
    assert float(transition_logits.grad.abs().sum()) > 0.0
    assert float(observer_logits.grad.abs().sum()) > 0.0


def test_hard_projection_is_permutation_and_balanced_by_construction() -> None:
    torch.manual_seed(47)
    transitions = torch.randn(
        4,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    observers = torch.randn(
        4,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    machine = LawfulMachineProjector().hard_project(
        transitions,
        observers,
    )
    for row in range(4):
        for action in range(PRIMARY_ACTIONS):
            assert sorted(
                machine.action_next[
                    row,
                    action,
                    :PRIMARY_STATES,
                ].tolist()
            ) == list(range(PRIMARY_STATES))
        for observer in range(PRIMARY_OBSERVERS):
            assert sorted(
                machine.observer_answer[
                    row,
                    observer,
                    :PRIMARY_STATES,
                ].tolist()
            ) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_oracle_visible_witnesses_recover_all_888_pilot_sources() -> None:
    rows = generate_pilot_rows(
        seed="efc-identifiable-pilot-20260724",
        counts={
            "confirmation": 24,
            "development": 32,
            "mechanics": 48,
            "train": 96,
        },
    )
    assert len(rows) == 888
    transition_logits, observer_logits = _oracle_logits(rows)
    projected = LawfulMachineProjector().hard_project(
        transition_logits,
        observer_logits,
    )
    for index, row in enumerate(rows):
        solved = solve_unique_completion(decode_source(row.source))
        assert tuple(
            tuple(
                projected.action_next[
                    index,
                    action,
                    :PRIMARY_STATES,
                ].tolist()
            )
            for action in range(PRIMARY_ACTIONS)
        ) == solved.transitions
        assert tuple(
            tuple(
                projected.observer_answer[
                    index,
                    observer,
                    :PRIMARY_STATES,
                ].tolist()
            )
            for observer in range(PRIMARY_OBSERVERS)
        ) == solved.observations


def test_relation_change_is_scoped_to_one_lawful_action() -> None:
    rows = generate_pilot_rows(
        seed="efc-transport-locality-v1",
        counts={
            "confirmation": 1,
            "development": 1,
            "mechanics": 6,
            "train": 3,
        },
    )
    transitions, observers = _oracle_logits(rows[:1])
    before = LawfulMachineProjector().hard_project(transitions, observers)
    changed = transitions.clone()
    changed[0, 0] = 0.0
    replacement = tuple(reversed(range(PRIMARY_STATES)))
    for state, destination in enumerate(replacement):
        changed[0, 0, state, destination] = 40.0
    after = LawfulMachineProjector().hard_project(changed, observers)
    assert after.action_next[0, 0, :PRIMARY_STATES].tolist() == list(
        replacement
    )
    assert torch.equal(
        before.action_next[0, 1:],
        after.action_next[0, 1:],
    )
    assert torch.equal(before.observer_answer, after.observer_answer)


def test_hard_key_assignment_is_unique_and_axis_coupled() -> None:
    rows = generate_pilot_rows(
        seed="efc-transport-key-v1",
        counts={
            "confirmation": 1,
            "development": 1,
            "mechanics": 6,
            "train": 3,
        },
    )
    row = rows[0]
    solved = solve_unique_completion(decode_source(row.source))
    source_batch = collate_sources((scan_source(row.source),))
    inventory = [
        bytes(value.tolist())
        for value in source_batch.unique_key_bytes[0]
    ]
    logits = torch.full(
        (
            1,
            16 + 8 + 8,
            MAX_UNIQUE_KEYS,
        ),
        -20.0,
    )
    slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(16 + index for index in range(PRIMARY_ACTIONS))
        + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
    )
    targets = (
        solved.state_keys
        + solved.action_keys
        + solved.observer_keys
    )
    for slot, key in zip(slots, targets, strict=True):
        logits[
            0,
            slot,
            inventory.index(key.to_bytes(8, "little")),
        ] = 20.0
    result = hard_assign_keys(
        slot_assignment_logits=logits,
        source_unique_key_bytes=source_batch.unique_key_bytes,
        source_unique_key_valid=source_batch.unique_key_valid,
    )
    assert len(set(result.active_unique_indices[0].tolist())) == 13
    decoded_state = tuple(
        int.from_bytes(bytes(key.tolist()), "little")
        for key in result.keys.state_keys[0, :PRIMARY_STATES]
    )
    decoded_action = tuple(
        int.from_bytes(bytes(key.tolist()), "little")
        for key in result.keys.action_keys[0, :PRIMARY_ACTIONS]
    )
    decoded_observer = tuple(
        int.from_bytes(bytes(key.tolist()), "little")
        for key in result.keys.observer_keys[0, :PRIMARY_OBSERVERS]
    )
    assert decoded_state == solved.state_keys
    assert decoded_action == solved.action_keys
    assert decoded_observer == solved.observer_keys


def test_soft_key_transport_is_one_to_one_on_active_slots() -> None:
    torch.manual_seed(59)
    logits = torch.randn(3, 32, MAX_UNIQUE_KEYS, requires_grad=True)
    valid = torch.zeros((3, MAX_UNIQUE_KEYS), dtype=torch.bool)
    valid[:, :13] = True
    projected = project_key_assignment_logits(
        slot_assignment_logits=logits,
        source_unique_key_valid=valid,
    )
    active_slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(16 + index for index in range(PRIMARY_ACTIONS))
        + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
    )
    transport = projected[:, active_slots, :13].softmax(-1)
    assert torch.allclose(
        transport.sum(-1),
        torch.ones(3, 13),
        atol=2e-5,
    )
    assert torch.allclose(
        transport.sum(-2),
        torch.ones(3, 13),
        atol=2e-5,
    )
    transport.square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_global_straight_through_key_transport_matches_sealed_assignment() -> None:
    torch.manual_seed(60)
    logits = torch.randn(2, 32, MAX_UNIQUE_KEYS, requires_grad=True)
    valid = torch.zeros((2, MAX_UNIQUE_KEYS), dtype=torch.bool)
    valid[:, :13] = True
    key_bytes = torch.zeros(
        (2, MAX_UNIQUE_KEYS, 8),
        dtype=torch.uint8,
    )
    for row in range(2):
        for index in range(13):
            key_bytes[row, index] = torch.tensor(
                tuple((index + 1).to_bytes(8, "little")),
                dtype=torch.uint8,
            )
    projected = project_key_assignment_logits(
        slot_assignment_logits=logits,
        source_unique_key_valid=valid,
        straight_through=True,
    )
    active_slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(16 + index for index in range(PRIMARY_ACTIONS))
        + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
    )
    attached_indices = projected[:, active_slots, :13].argmax(-1)
    assert all(
        len(set(row.tolist())) == 13
        for row in attached_indices
    )
    sealed = hard_assign_keys(
        slot_assignment_logits=projected,
        source_unique_key_bytes=key_bytes,
        source_unique_key_valid=valid,
    )
    assert torch.equal(attached_indices, sealed.active_unique_indices)
    projected[:, active_slots, :13].square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_global_straight_through_matches_lawful_hard_projection() -> None:
    torch.manual_seed(61)
    transitions = torch.randn(
        2,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
        requires_grad=True,
    )
    observers = torch.randn(
        2,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
        requires_grad=True,
    )
    projector = LawfulMachineProjector()
    attached_machine = projector(
        transitions,
        observers,
        straight_through=True,
    ).machine
    detached_machine = projector.hard_project(transitions, observers)
    assert torch.equal(
        attached_machine.harden().action_next,
        detached_machine.action_next,
    )
    assert torch.equal(
        attached_machine.harden().observer_answer,
        detached_machine.observer_answer,
    )
    start = torch.tensor((1, 6), dtype=torch.long)
    actions = torch.tensor(((0, 2, 1), (2, 0, 2)), dtype=torch.long)
    stop = torch.tensor((3, 2), dtype=torch.long)
    observer = torch.tensor((0, 1), dtype=torch.long)

    def logits(indices: torch.Tensor, classes: int) -> torch.Tensor:
        output = torch.full((*indices.shape, classes), -20.0)
        return output.scatter(
            -1,
            indices.unsqueeze(-1),
            20.0,
        )

    soft_query = SoftFunctorQuery(
        start_state=logits(start, MAX_STATES),
        action_path=logits(actions, MAX_ACTIONS),
        stop_position=logits(stop, actions.shape[1] + 1),
        observer=logits(observer, MAX_OBSERVERS),
    )
    hard_query = HardFunctorQuery(
        start_state=start.to(torch.uint8),
        action_path=actions.to(torch.uint8),
        stop_position=stop.to(torch.uint8),
        observer=observer.to(torch.uint8),
    )
    attached = execute_soft(
        attached_machine,
        soft_query,
        straight_through=True,
    )
    detached = execute_hard(detached_machine, hard_query)
    assert torch.equal(
        attached.states.argmax(-1),
        detached.states.argmax(-1),
    )
    assert torch.equal(
        attached.answer.argmax(-1),
        detached.answer,
    )
    attached.answer.square().mean().backward()
    assert transitions.grad is not None
    assert observers.grad is not None
    assert float(transitions.grad.abs().sum()) > 0.0
    assert float(observers.grad.abs().sum()) > 0.0


def test_extreme_finite_logits_keep_sinkhorn_gradients_finite() -> None:
    transitions = torch.full(
        (1, PRIMARY_ACTIONS, PRIMARY_STATES, PRIMARY_STATES),
        -1e20,
        requires_grad=True,
    )
    observers = torch.full(
        (1, PRIMARY_OBSERVERS, PRIMARY_STATES, PRIMARY_ANSWERS),
        -1e20,
        requires_grad=True,
    )
    with torch.no_grad():
        transitions[:, :, :, 0] = 1e20
        observers[:, :, :, 0] = 1e20
    projected = LawfulMachineProjector()(
        transitions,
        observers,
        straight_through=True,
    )
    loss = (
        projected.transition_transport.square().sum()
        + projected.observer_transport.square().sum()
        + projected.machine.action_next.square().sum()
        + projected.machine.observer_answer.square().sum()
    )
    loss.backward()
    assert transitions.grad is not None
    assert observers.grad is not None
    assert bool(torch.isfinite(transitions.grad).all())
    assert bool(torch.isfinite(observers.grad).all())

    key_logits = torch.full(
        (1, 32, MAX_UNIQUE_KEYS),
        -1e20,
        requires_grad=True,
    )
    valid = torch.zeros((1, MAX_UNIQUE_KEYS), dtype=torch.bool)
    valid[:, :13] = True
    with torch.no_grad():
        for slot in range(13):
            key_logits[0, slot, slot] = 1e20
    key_transport = project_key_assignment_logits(
        slot_assignment_logits=key_logits,
        source_unique_key_valid=valid,
        straight_through=True,
    )
    key_transport.square().sum().backward()
    assert key_logits.grad is not None
    assert bool(torch.isfinite(key_logits.grad).all())
