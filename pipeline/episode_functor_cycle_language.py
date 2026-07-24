#!/usr/bin/env python3
"""Canonical cycle/program source language for EFC world evidence.

This source family is intentionally independent of the EFC generator,
protocol, compiler, and runtimes.  Unlike the event-row renderers, it declares
one closed algebraic program:

* states are opaque typed uint64 keys;
* every action is a complete canonical disjoint-cycle decomposition; and
* every observer is a canonical partition into answer-labelled classes.

The decoder returns ``efc-raw-world-evidence-v2`` semantics with source
renderer choice 2.  Narrow future wire integration consists only of dispatch
on ``MAGIC``, calling :func:`decode_cycle_language`, admitting renderer 2 in
the source-renderer count, and passing the returned row to the existing raw
evidence table parser.  No compiler behavior or machine format must change.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


MAGIC = "EFC-CYCLE-PROGRAM-V1"
NORMALIZED_SCHEMA = "efc-raw-world-evidence-v2"
RENDERER_CHOICE = 2

MAX_STATES = 16
MAX_ACTIONS = 8
MAX_OBSERVERS = 8
MAX_DEMONSTRATIONS = MAX_STATES * MAX_ACTIONS
MAX_OBSERVATIONS = MAX_STATES * MAX_OBSERVERS
MAX_LINES = 32
MAX_LINE_BYTES = 2_048
MAX_TOKEN_COUNT = 2_048
MAX_PAYLOAD_BYTES = MAX_LINES * (MAX_LINE_BYTES + 1)
MAX_U64 = (1 << 64) - 1

_ROW_FIELDS = frozenset(
    {
        "demonstrations",
        "observations",
        "renderer_choice",
        "schema",
    }
)
_HEX = r"[0-9a-f]{16}"
_STATE_TOKEN = re.compile(rf"s#({_HEX})")
_ACTION_LINE = re.compile(rf"  a#({_HEX}) := (.+);")
_OBSERVER_LINE = re.compile(rf"  o#({_HEX}) := (.+);")
_CYCLE = re.compile(r"cycle\[(.+)\]")
_CLASS = re.compile(r"class\[(0|[1-9][0-9]{0,19})\]\{(.+)\}")
_STATES_LINE = re.compile(r"states = \[(.+)\];")
_LEXICAL_TOKEN = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9 \n]")


class CycleLanguageError(ValueError):
    """The cycle source is malformed, noncanonical, or semantically invalid."""


@dataclass(frozen=True)
class _Tables:
    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    observer_keys: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleLanguageError(f"{field} must be a plain integer")
    return value


def _nonzero_u64(value: object, field: str) -> int:
    checked = _plain_int(value, field)
    if checked <= 0 or checked > MAX_U64:
        raise CycleLanguageError(f"{field} must be a nonzero uint64")
    return checked


def _answer_u64(value: object) -> int:
    checked = _plain_int(value, "answer")
    if checked < 0 or checked > MAX_U64:
        raise CycleLanguageError("answer must be a uint64")
    return checked


def _bounded_count(values: set[int], maximum: int, kind: str) -> tuple[int, ...]:
    if not values or len(values) > maximum:
        raise CycleLanguageError(f"{kind} count is outside the language bound")
    return tuple(sorted(values))


def _tables_from_row(row: object) -> _Tables:
    if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
        raise CycleLanguageError("normalized evidence has incorrect fields")
    if row["schema"] != NORMALIZED_SCHEMA:
        raise CycleLanguageError("normalized evidence has unknown schema")
    renderer = _plain_int(row["renderer_choice"], "renderer_choice")
    if renderer not in (0, 1, RENDERER_CHOICE):
        raise CycleLanguageError("normalized evidence has unknown renderer")

    demonstrations = row["demonstrations"]
    observations = row["observations"]
    if not isinstance(demonstrations, list) or not isinstance(observations, list):
        raise CycleLanguageError("normalized evidence events must be lists")
    if (
        not demonstrations
        or len(demonstrations) > MAX_DEMONSTRATIONS
        or not observations
        or len(observations) > MAX_OBSERVATIONS
    ):
        raise CycleLanguageError("normalized evidence event count is out of bounds")

    transition_cells: dict[tuple[int, int], int] = {}
    state_set: set[int] = set()
    action_set: set[int] = set()
    for event in demonstrations:
        if not isinstance(event, dict) or set(event) != {
            "action_key",
            "source_key",
            "target_key",
        }:
            raise CycleLanguageError("transition event is malformed")
        action = _nonzero_u64(event["action_key"], "action_key")
        source = _nonzero_u64(event["source_key"], "source_key")
        target = _nonzero_u64(event["target_key"], "target_key")
        cell = (action, source)
        if cell in transition_cells:
            raise CycleLanguageError("transition cell is duplicated")
        transition_cells[cell] = target
        action_set.add(action)
        state_set.update((source, target))

    state_keys = _bounded_count(state_set, MAX_STATES, "state")
    action_keys = _bounded_count(action_set, MAX_ACTIONS, "action")
    expected_transition_cells = len(state_keys) * len(action_keys)
    if len(transition_cells) != expected_transition_cells:
        raise CycleLanguageError("action program is incomplete")
    state_values = set(state_keys)
    transition_rows: list[tuple[int, ...]] = []
    for action in action_keys:
        try:
            relation = tuple(transition_cells[(action, state)] for state in state_keys)
        except KeyError as exc:
            raise CycleLanguageError("action program is incomplete") from exc
        if set(relation) != state_values or len(set(relation)) != len(state_keys):
            raise CycleLanguageError("action relation is not a permutation")
        transition_rows.append(relation)

    observation_cells: dict[tuple[int, int], int] = {}
    observer_set: set[int] = set()
    for event in observations:
        if not isinstance(event, dict) or set(event) != {
            "answer",
            "observer_key",
            "state_key",
        }:
            raise CycleLanguageError("observation event is malformed")
        observer = _nonzero_u64(event["observer_key"], "observer_key")
        state = _nonzero_u64(event["state_key"], "state_key")
        if state not in state_values:
            raise CycleLanguageError("observer references an unknown state")
        answer = _answer_u64(event["answer"])
        cell = (observer, state)
        if cell in observation_cells:
            raise CycleLanguageError("observation cell is duplicated")
        observation_cells[cell] = answer
        observer_set.add(observer)

    observer_keys = _bounded_count(observer_set, MAX_OBSERVERS, "observer")
    expected_observation_cells = len(state_keys) * len(observer_keys)
    if len(observation_cells) != expected_observation_cells:
        raise CycleLanguageError("observer partition is incomplete")
    observation_rows: list[tuple[int, ...]] = []
    for observer in observer_keys:
        try:
            observation_rows.append(
                tuple(observation_cells[(observer, state)] for state in state_keys)
            )
        except KeyError as exc:
            raise CycleLanguageError("observer partition is incomplete") from exc

    return _Tables(
        state_keys=state_keys,
        action_keys=action_keys,
        observer_keys=observer_keys,
        transitions=tuple(transition_rows),
        observations=tuple(observation_rows),
    )


def _state_token(key: int) -> str:
    return f"s#{key:016x}"


def _canonical_cycles(
    state_keys: tuple[int, ...],
    relation: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    destination = dict(zip(state_keys, relation, strict=True))
    remaining = set(state_keys)
    cycles: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        cycle: list[int] = []
        cursor = start
        while cursor not in cycle:
            if cursor not in remaining:
                raise CycleLanguageError("action cycles are not disjoint")
            cycle.append(cursor)
            remaining.remove(cursor)
            cursor = destination[cursor]
        if cursor != start:
            raise CycleLanguageError("action relation is not a permutation")
        cycles.append(tuple(cycle))
    return tuple(cycles)


def _encode_tables(tables: _Tables) -> bytes:
    state_program = ",".join(_state_token(key) for key in tables.state_keys)
    lines = [
        MAGIC,
        f"states = [{state_program}];",
        "actions = {",
    ]
    for action, relation in zip(
        tables.action_keys,
        tables.transitions,
        strict=True,
    ):
        cycles = _canonical_cycles(tables.state_keys, relation)
        program = " * ".join(
            "cycle[" + ",".join(_state_token(state) for state in cycle) + "]"
            for cycle in cycles
        )
        lines.append(f"  a#{action:016x} := {program};")
    lines.extend(("};", "observers = {"))
    for observer, answers in zip(
        tables.observer_keys,
        tables.observations,
        strict=True,
    ):
        classes: dict[int, list[int]] = {}
        for state, answer in zip(tables.state_keys, answers, strict=True):
            classes.setdefault(answer, []).append(state)
        partition = " + ".join(
            f"class[{answer}]"
            + "{"
            + ",".join(_state_token(state) for state in classes[answer])
            + "}"
            for answer in sorted(classes)
        )
        lines.append(f"  o#{observer:016x} := {partition};")
    lines.extend(("};", "halt."))
    payload = ("\n".join(lines) + "\n").encode("ascii")
    _preflight(payload)
    return payload


def encode_cycle_language(row: Mapping[str, object]) -> bytes:
    """Encode normalized raw-world semantics as one canonical cycle program."""

    return _encode_tables(_tables_from_row(row))


def _preflight(payload: bytes) -> list[str]:
    if not isinstance(payload, bytes):
        raise CycleLanguageError("cycle program must be bytes")
    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise CycleLanguageError("cycle program byte length is out of bounds")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CycleLanguageError("cycle program must be strict ASCII") from exc
    if not text.endswith("\n") or "\r" in text or "\t" in text or "\x00" in text:
        raise CycleLanguageError("cycle program has noncanonical text framing")
    lines = text[:-1].split("\n")
    if (
        not lines
        or len(lines) > MAX_LINES
        or any(not line for line in lines)
        or any(len(line.encode("ascii")) > MAX_LINE_BYTES for line in lines)
    ):
        raise CycleLanguageError("cycle program line bound is violated")
    if len(_LEXICAL_TOKEN.findall(text)) > MAX_TOKEN_COUNT:
        raise CycleLanguageError("cycle program token bound is violated")
    return lines


def _parse_state_list(source: str) -> tuple[int, ...]:
    parts = source.split(",")
    if not parts or len(parts) > MAX_STATES:
        raise CycleLanguageError("state declaration count is out of bounds")
    values: list[int] = []
    for part in parts:
        match = _STATE_TOKEN.fullmatch(part)
        if match is None:
            raise CycleLanguageError("state key spelling is invalid")
        value = int(match.group(1), 16)
        if value == 0:
            raise CycleLanguageError("state key must be nonzero")
        values.append(value)
    if len(set(values)) != len(values):
        raise CycleLanguageError("state declaration contains duplicates")
    return tuple(values)


def _parse_action_program(
    source: str,
    state_keys: tuple[int, ...],
) -> tuple[int, ...]:
    clauses = source.split(" * ")
    if not clauses or len(clauses) > len(state_keys):
        raise CycleLanguageError("action cycle count is out of bounds")
    destination: dict[int, int] = {}
    known_states = set(state_keys)
    for clause in clauses:
        match = _CYCLE.fullmatch(clause)
        if match is None:
            raise CycleLanguageError("action cycle spelling is invalid")
        cycle = _parse_state_list(match.group(1))
        if any(state not in known_states for state in cycle):
            raise CycleLanguageError("action cycle references an unknown state")
        for index, source_state in enumerate(cycle):
            if source_state in destination:
                raise CycleLanguageError("action cycles are not disjoint")
            destination[source_state] = cycle[(index + 1) % len(cycle)]
    if set(destination) != known_states:
        raise CycleLanguageError("action cycles do not cover every state")
    return tuple(destination[state] for state in state_keys)


def _parse_observer_program(
    source: str,
    state_keys: tuple[int, ...],
) -> tuple[int, ...]:
    clauses = source.split(" + ")
    if not clauses or len(clauses) > len(state_keys):
        raise CycleLanguageError("observer class count is out of bounds")
    answer_for_state: dict[int, int] = {}
    seen_answers: set[int] = set()
    known_states = set(state_keys)
    for clause in clauses:
        match = _CLASS.fullmatch(clause)
        if match is None:
            raise CycleLanguageError("observer class spelling is invalid")
        answer = int(match.group(1))
        if answer > MAX_U64:
            raise CycleLanguageError("observer answer exceeds uint64")
        if answer in seen_answers:
            raise CycleLanguageError("observer answer class is duplicated")
        seen_answers.add(answer)
        members = _parse_state_list(match.group(2))
        if any(state not in known_states for state in members):
            raise CycleLanguageError("observer class references an unknown state")
        for state in members:
            if state in answer_for_state:
                raise CycleLanguageError("observer classes are not disjoint")
            answer_for_state[state] = answer
    if set(answer_for_state) != known_states:
        raise CycleLanguageError("observer classes do not partition every state")
    return tuple(answer_for_state[state] for state in state_keys)


def _row_from_tables(tables: _Tables) -> dict[str, object]:
    demonstrations = [
        {
            "action_key": action,
            "source_key": state,
            "target_key": relation[state_slot],
        }
        for action, relation in zip(
            tables.action_keys,
            tables.transitions,
            strict=True,
        )
        for state_slot, state in enumerate(tables.state_keys)
    ]
    observations = [
        {
            "answer": answers[state_slot],
            "observer_key": observer,
            "state_key": state,
        }
        for observer, answers in zip(
            tables.observer_keys,
            tables.observations,
            strict=True,
        )
        for state_slot, state in enumerate(tables.state_keys)
    ]
    return {
        "demonstrations": demonstrations,
        "observations": observations,
        "renderer_choice": RENDERER_CHOICE,
        "schema": NORMALIZED_SCHEMA,
    }


def decode_cycle_language(payload: bytes) -> dict[str, object]:
    """Decode one strict canonical cycle program into normalized semantics."""

    lines = _preflight(payload)
    if len(lines) < 9 or lines[0] != MAGIC:
        raise CycleLanguageError("cycle program header is invalid")
    state_match = _STATES_LINE.fullmatch(lines[1])
    if state_match is None:
        raise CycleLanguageError("state declaration is invalid")
    state_keys = _parse_state_list(state_match.group(1))
    if lines[2] != "actions = {":
        raise CycleLanguageError("action section header is invalid")

    try:
        action_end = lines.index("};", 3)
    except ValueError as exc:
        raise CycleLanguageError("action section is unterminated") from exc
    action_lines = lines[3:action_end]
    if not action_lines or len(action_lines) > MAX_ACTIONS:
        raise CycleLanguageError("action declaration count is out of bounds")
    if action_end + 1 >= len(lines) or lines[action_end + 1] != "observers = {":
        raise CycleLanguageError("observer section header is invalid")
    try:
        observer_end = lines.index("};", action_end + 2)
    except ValueError as exc:
        raise CycleLanguageError("observer section is unterminated") from exc
    if observer_end != len(lines) - 2 or lines[-1] != "halt.":
        raise CycleLanguageError("cycle program trailer is invalid")
    observer_lines = lines[action_end + 2 : observer_end]
    if not observer_lines or len(observer_lines) > MAX_OBSERVERS:
        raise CycleLanguageError("observer declaration count is out of bounds")

    action_keys: list[int] = []
    transitions: list[tuple[int, ...]] = []
    for line in action_lines:
        match = _ACTION_LINE.fullmatch(line)
        if match is None:
            raise CycleLanguageError("action declaration is invalid")
        key = int(match.group(1), 16)
        if key == 0 or key in action_keys:
            raise CycleLanguageError("action key is zero or duplicated")
        action_keys.append(key)
        transitions.append(_parse_action_program(match.group(2), state_keys))

    observer_keys: list[int] = []
    observations: list[tuple[int, ...]] = []
    for line in observer_lines:
        match = _OBSERVER_LINE.fullmatch(line)
        if match is None:
            raise CycleLanguageError("observer declaration is invalid")
        key = int(match.group(1), 16)
        if key == 0 or key in observer_keys:
            raise CycleLanguageError("observer key is zero or duplicated")
        observer_keys.append(key)
        observations.append(_parse_observer_program(match.group(2), state_keys))

    tables = _Tables(
        state_keys=state_keys,
        action_keys=tuple(action_keys),
        observer_keys=tuple(observer_keys),
        transitions=tuple(transitions),
        observations=tuple(observations),
    )
    canonical_tables = _tables_from_row(_row_from_tables(tables))
    if _encode_tables(canonical_tables) != payload:
        raise CycleLanguageError("cycle program is not canonical")
    return _row_from_tables(canonical_tables)


__all__ = [
    "CycleLanguageError",
    "MAGIC",
    "MAX_ACTIONS",
    "MAX_LINE_BYTES",
    "MAX_LINES",
    "MAX_OBSERVERS",
    "MAX_PAYLOAD_BYTES",
    "MAX_STATES",
    "MAX_TOKEN_COUNT",
    "NORMALIZED_SCHEMA",
    "RENDERER_CHOICE",
    "decode_cycle_language",
    "encode_cycle_language",
]
