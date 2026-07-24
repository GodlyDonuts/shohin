#!/usr/bin/env python3
"""Independent parser and exhaustive version-space solver for EFC-I sources.

This module intentionally does not import the production source decoder or
completion solver.  It parses the frozen surface forms with separate regular
expressions and enumerates every missing-cell assignment satisfying the public
permutation and observer-balance laws.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import re
from typing import Sequence

from pipeline.episode_functor_identifiable_board import (
    ACTION_COUNT,
    ANSWER_COUNT,
    IdentifiableBoardError,
    OBSERVER_COUNT,
    STATE_COUNT,
)


_KEY = r"(?:h[0-9a-f]{16}|d[1-9][0-9]{0,19})"
_STATE_POSITIONAL = re.compile(rf"S ({_KEY})")
_STATE_LABELED = re.compile(rf"S key=({_KEY})")
_TRANSITION_POSITIONAL = re.compile(rf"T ({_KEY}) ({_KEY}) ({_KEY})")
_TRANSITION_LABELED = re.compile(
    rf"T dst=({_KEY}) action=({_KEY}) src=({_KEY})"
)
_OBSERVATION_POSITIONAL = re.compile(rf"O ({_KEY}) ({_KEY}) ([0-3])")
_OBSERVATION_LABELED = re.compile(
    rf"O answer=([0-3]) state=({_KEY}) observer=({_KEY})"
)


@dataclass(frozen=True, slots=True)
class ReferenceMachine:
    """Reference-only completed machine with independent validation."""

    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    observer_keys: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.state_keys) != STATE_COUNT
            or len(set(self.state_keys)) != STATE_COUNT
            or len(self.action_keys) != ACTION_COUNT
            or len(set(self.action_keys)) != ACTION_COUNT
            or len(self.observer_keys) != OBSERVER_COUNT
            or len(set(self.observer_keys)) != OBSERVER_COUNT
        ):
            raise IdentifiableBoardError("reference machine key geometry differs")
        if any(sorted(row) != list(range(STATE_COUNT)) for row in self.transitions):
            raise IdentifiableBoardError("reference transition is not a permutation")
        expected_answers = [
            answer
            for answer in range(ANSWER_COUNT)
            for _ in range(STATE_COUNT // ANSWER_COUNT)
        ]
        if (
            len(self.transitions) != ACTION_COUNT
            or len(self.observations) != OBSERVER_COUNT
            or any(sorted(row) != expected_answers for row in self.observations)
        ):
            raise IdentifiableBoardError("reference relation geometry differs")


def _key(token: str) -> int:
    value = int(token[1:], 16 if token.startswith("h") else 10)
    if value <= 0 or value >= 1 << 64:
        raise IdentifiableBoardError("reference key leaves uint64")
    return value


def _records(payload: bytes) -> tuple[str, ...]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise IdentifiableBoardError("reference source is not ASCII") from exc
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise IdentifiableBoardError("reference source framing is noncanonical")
    if text[:10] == "BEGIN-EFC\n" and text[-8:] == "END-EFC\n":
        records = tuple(text[10:-8].splitlines())
    elif text[:5] == "EFC{ " and text[-3:] == " }\n":
        records = tuple(text[5:-3].split(" ; "))
    else:
        raise IdentifiableBoardError("reference source wrapper is unknown")
    if not records or any(not record for record in records):
        raise IdentifiableBoardError("reference source contains an empty record")
    return records


def _parse(
    payload: bytes,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int, int], ...],
]:
    states: list[int] = []
    transitions: list[tuple[int, int, int]] = []
    observations: list[tuple[int, int, int]] = []
    laws: set[str] = set()
    for record in _records(payload):
        match = _STATE_POSITIONAL.fullmatch(record)
        if match is None:
            match = _STATE_LABELED.fullmatch(record)
        if match is not None:
            states.append(_key(match.group(1)))
            continue
        if record in ("LAW-A PERMUTATION", "LAW-A kind=permutation"):
            laws.add("action")
            continue
        if record in (
            "LAW-O BALANCED 2 EACH 0 1 2 3",
            "LAW-O multiplicity=2 answers=0,1,2,3",
        ):
            laws.add("observer")
            continue
        match = _TRANSITION_POSITIONAL.fullmatch(record)
        if match is not None:
            transitions.append(tuple(_key(match.group(index)) for index in (1, 2, 3)))
            continue
        match = _TRANSITION_LABELED.fullmatch(record)
        if match is not None:
            transitions.append(
                (
                    _key(match.group(2)),
                    _key(match.group(3)),
                    _key(match.group(1)),
                )
            )
            continue
        match = _OBSERVATION_POSITIONAL.fullmatch(record)
        if match is not None:
            observations.append(
                (_key(match.group(1)), _key(match.group(2)), int(match.group(3)))
            )
            continue
        match = _OBSERVATION_LABELED.fullmatch(record)
        if match is not None:
            observations.append(
                (_key(match.group(3)), _key(match.group(2)), int(match.group(1)))
            )
            continue
        raise IdentifiableBoardError("reference parser found an unknown record")
    if laws != {"action", "observer"}:
        raise IdentifiableBoardError("reference parser did not find both laws")
    if (
        len(states) != STATE_COUNT
        or len(set(states)) != STATE_COUNT
        or len(transitions) != ACTION_COUNT * (STATE_COUNT - 1)
        or len(observations) != OBSERVER_COUNT * (STATE_COUNT - 1)
    ):
        raise IdentifiableBoardError("reference source geometry differs")
    return tuple(states), tuple(transitions), tuple(observations)


def _transition_rows(
    states: tuple[int, ...],
    events: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    state_set = set(states)
    state_index = {key: index for index, key in enumerate(states)}
    action_keys = tuple(sorted({action for action, _, _ in events}))
    if len(action_keys) != ACTION_COUNT:
        raise IdentifiableBoardError("reference action cardinality differs")
    result: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    for action_key in action_keys:
        selected = [
            (source, target)
            for action, source, target in events
            if action == action_key
        ]
        if (
            len(selected) != STATE_COUNT - 1
            or len({source for source, _ in selected}) != STATE_COUNT - 1
            or any(source not in state_set or target not in state_set for source, target in selected)
        ):
            raise IdentifiableBoardError("reference action evidence is malformed")
        fixed = dict(selected)
        missing_sources = state_set - set(fixed)
        if len(missing_sources) != 1:
            raise IdentifiableBoardError("reference action has multiple missing sources")
        missing_source = next(iter(missing_sources))
        rows: list[tuple[int, ...]] = []
        for candidate_target in states:
            candidate = dict(fixed)
            candidate[missing_source] = candidate_target
            destinations = tuple(candidate[state] for state in states)
            if set(destinations) == state_set:
                rows.append(tuple(state_index[target] for target in destinations))
        if not rows:
            raise IdentifiableBoardError("reference action has no lawful completion")
        result.append((action_key, tuple(rows)))
    return tuple(result)


def _observer_rows(
    states: tuple[int, ...],
    events: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    state_set = set(states)
    observer_keys = tuple(sorted({observer for observer, _, _ in events}))
    if len(observer_keys) != OBSERVER_COUNT:
        raise IdentifiableBoardError("reference observer cardinality differs")
    result: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    for observer_key in observer_keys:
        selected = [
            (state, answer)
            for observer, state, answer in events
            if observer == observer_key
        ]
        if (
            len(selected) != STATE_COUNT - 1
            or len({state for state, _ in selected}) != STATE_COUNT - 1
            or any(state not in state_set for state, _ in selected)
        ):
            raise IdentifiableBoardError("reference observer evidence is malformed")
        fixed = dict(selected)
        missing_states = state_set - set(fixed)
        if len(missing_states) != 1:
            raise IdentifiableBoardError("reference observer has multiple missing states")
        missing_state = next(iter(missing_states))
        rows: list[tuple[int, ...]] = []
        for candidate_answer in range(ANSWER_COUNT):
            candidate = dict(fixed)
            candidate[missing_state] = candidate_answer
            row = tuple(candidate[state] for state in states)
            if Counter(row) == Counter(
                {
                    answer: STATE_COUNT // ANSWER_COUNT
                    for answer in range(ANSWER_COUNT)
                }
            ):
                rows.append(row)
        if not rows:
            raise IdentifiableBoardError("reference observer has no lawful completion")
        result.append((observer_key, tuple(rows)))
    return tuple(result)


def enumerate_consistent_machines(payload: bytes) -> tuple[ReferenceMachine, ...]:
    """Enumerate the complete finite version space admitted by one source."""

    unsorted_states, transition_events, observation_events = _parse(payload)
    states = tuple(sorted(unsorted_states))
    action_options = _transition_rows(states, transition_events)
    observer_options = _observer_rows(states, observation_events)
    machines: list[ReferenceMachine] = []
    for transitions in itertools.product(*(rows for _, rows in action_options)):
        for observations in itertools.product(*(rows for _, rows in observer_options)):
            machines.append(
                ReferenceMachine(
                    state_keys=states,
                    action_keys=tuple(key for key, _ in action_options),
                    observer_keys=tuple(key for key, _ in observer_options),
                    transitions=tuple(transitions),
                    observations=tuple(observations),
                )
            )
    return tuple(machines)


def _execute(
    machine: ReferenceMachine,
    start: int,
    actions: Sequence[int],
    observer: int,
) -> int:
    state = start
    for action in actions:
        state = machine.transitions[action][state]
    return machine.observations[observer][state]


def version_space_receipt(payload: bytes, *, max_depth: int) -> dict[str, int]:
    if not 0 <= max_depth <= 8:
        raise IdentifiableBoardError("reference audit depth leaves support")
    machines = enumerate_consistent_machines(payload)
    if not machines:
        raise IdentifiableBoardError("reference version space is empty")
    signatures: set[tuple[int, ...]] = set()
    coordinate_count = 0
    for machine in machines:
        signature: list[int] = []
        for depth in range(max_depth + 1):
            for actions in itertools.product(range(ACTION_COUNT), repeat=depth):
                for start in range(STATE_COUNT):
                    for observer in range(OBSERVER_COUNT):
                        signature.append(_execute(machine, start, actions, observer))
                        if len(signatures) == 0:
                            coordinate_count += 1
        signatures.add(tuple(signature))
    return {
        "behavior_classes": len(signatures),
        "coordinates": coordinate_count,
        "version_space": len(machines),
    }


__all__ = [
    "ReferenceMachine",
    "enumerate_consistent_machines",
    "version_space_receipt",
]
