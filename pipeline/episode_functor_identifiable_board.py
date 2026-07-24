#!/usr/bin/env python3
"""Identifiable partial-evidence board for a learned EPISODE compiler.

This is a CPU mechanics generator, not a neural dataset authorization.  Every
source hides one cell from each action permutation and balanced observer.  The
public laws make the omitted cells uniquely recoverable.  Renderer factors are
composed without emitting a renderer or family identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import random
import re
from typing import Iterable, Sequence


STATE_COUNT = 8
ACTION_COUNT = 3
OBSERVER_COUNT = 2
ANSWER_COUNT = 4
ANSWER_MULTIPLICITY = 2
SOURCE_FACTOR_COMBINATIONS = tuple(itertools.product(range(2), repeat=3))
TRAIN_FACTOR_COMBINATIONS = tuple(
    factors for factors in SOURCE_FACTOR_COMBINATIONS if sum(factors) % 2 == 0
)
HELD_FACTOR_COMBINATIONS = tuple(
    factors for factors in SOURCE_FACTOR_COMBINATIONS if sum(factors) % 2 == 1
)
FAMILIES = (
    "random-s8",
    "affine-f2-3",
    "dihedral-vertices",
    "dihedral-regular",
    "quaternion-regular",
    "cube-rotations",
)
_KEY_HEX = re.compile(r"h([0-9a-f]{16})")
_KEY_DECIMAL = re.compile(r"d([1-9][0-9]{0,19})")


class IdentifiableBoardError(ValueError):
    """A board, renderer, solver, or mechanics invariant failed."""


@dataclass(frozen=True, slots=True)
class GrammarFactors:
    framing: int
    organization: int
    codec: int

    def __post_init__(self) -> None:
        if (self.framing, self.organization, self.codec) not in (
            SOURCE_FACTOR_COMBINATIONS
        ):
            raise IdentifiableBoardError("grammar factors leave the binary cube")

    @property
    def values(self) -> tuple[int, int, int]:
        return self.framing, self.organization, self.codec


@dataclass(frozen=True, slots=True)
class IdentifiableMachine:
    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    observer_keys: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.state_keys) != STATE_COUNT
            or len(self.action_keys) != ACTION_COUNT
            or len(self.observer_keys) != OBSERVER_COUNT
            or len(set(self.state_keys)) != STATE_COUNT
            or len(set(self.action_keys)) != ACTION_COUNT
            or len(set(self.observer_keys)) != OBSERVER_COUNT
        ):
            raise IdentifiableBoardError("machine key geometry differs")
        if any(key <= 0 or key >= 1 << 64 for key in (
            *self.state_keys,
            *self.action_keys,
            *self.observer_keys,
        )):
            raise IdentifiableBoardError("machine key is not nonzero uint64")
        if len(self.transitions) != ACTION_COUNT or any(
            sorted(row) != list(range(STATE_COUNT)) for row in self.transitions
        ):
            raise IdentifiableBoardError("machine action is not a permutation")
        if _reachable_count(self.transitions) != STATE_COUNT:
            raise IdentifiableBoardError("machine actions are not rooted-transitive")
        if len(self.observations) != OBSERVER_COUNT or any(
            len(row) != STATE_COUNT
            or sorted(row)
            != [
                answer
                for answer in range(ANSWER_COUNT)
                for _ in range(ANSWER_MULTIPLICITY)
            ]
            for row in self.observations
        ):
            raise IdentifiableBoardError("machine observer is not balanced")
        if _future_class_count(self.transitions, self.observations) != STATE_COUNT:
            raise IdentifiableBoardError("machine future behavior is not separating")

    def canonical_structural_bytes(self) -> bytes:
        """Canonicalize state/action/observer and global answer recodings."""

        candidates: list[bytes] = []
        for action_order in itertools.permutations(range(ACTION_COUNT)):
            ordered_actions = tuple(self.transitions[index] for index in action_order)
            for root in range(STATE_COUNT):
                old_for_new = _bfs_state_order(ordered_actions, root)
                old_to_new = {
                    old: new for new, old in enumerate(old_for_new)
                }
                canonical_transitions = tuple(
                    tuple(old_to_new[row[old]] for old in old_for_new)
                    for row in ordered_actions
                )
                for observer_order in itertools.permutations(
                    range(OBSERVER_COUNT)
                ):
                    ordered_observations = tuple(
                        tuple(self.observations[index][old] for old in old_for_new)
                        for index in observer_order
                    )
                    answer_map: dict[int, int] = {}
                    normalized_observations: list[tuple[int, ...]] = []
                    for row in ordered_observations:
                        normalized_row: list[int] = []
                        for answer in row:
                            if answer not in answer_map:
                                answer_map[answer] = len(answer_map)
                            normalized_row.append(answer_map[answer])
                        normalized_observations.append(tuple(normalized_row))
                    payload = {
                        "observations": normalized_observations,
                        "transitions": canonical_transitions,
                    }
                    candidates.append(
                        json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    )
        return min(candidates)


@dataclass(frozen=True, slots=True)
class PartialEvidence:
    state_keys: tuple[int, ...]
    transition_events: tuple[tuple[int, int, int], ...]
    observation_events: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.state_keys) != STATE_COUNT or len(set(self.state_keys)) != STATE_COUNT:
            raise IdentifiableBoardError("partial evidence state inventory differs")
        if len(self.transition_events) != ACTION_COUNT * (STATE_COUNT - 1):
            raise IdentifiableBoardError("partial transition evidence count differs")
        if len(self.observation_events) != OBSERVER_COUNT * (STATE_COUNT - 1):
            raise IdentifiableBoardError("partial observation evidence count differs")


@dataclass(frozen=True, slots=True)
class LateQuery:
    start_key: int
    action_keys: tuple[int, ...]
    observer_key: int

    def __post_init__(self) -> None:
        if not 0 <= len(self.action_keys) <= 32:
            raise IdentifiableBoardError("late query path length differs")
        if any(key <= 0 or key >= 1 << 64 for key in (
            self.start_key,
            *self.action_keys,
            self.observer_key,
        )):
            raise IdentifiableBoardError("late query key is not nonzero uint64")


@dataclass(frozen=True, slots=True)
class PilotRow:
    world_id: str
    split: str
    family: str
    factors: GrammarFactors
    source: bytes
    machine: IdentifiableMachine
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateSource:
    """The only source object admissible to a neural forward path."""

    source: bytes

    def __post_init__(self) -> None:
        if not self.source:
            raise IdentifiableBoardError("candidate source is empty")
        if any(
            marker in self.source.lower()
            for marker in (b"renderer", b"family", b"split")
        ):
            raise IdentifiableBoardError(
                "candidate source contains forbidden metadata"
            )


def _domain_rng(seed: str, *parts: object) -> random.Random:
    framed = "\0".join((seed, *(str(part) for part in parts))).encode("utf-8")
    return random.Random(int.from_bytes(sha256(framed).digest(), "big"))


def _shuffle(rng: random.Random, values: Iterable[int]) -> tuple[int, ...]:
    result = list(values)
    rng.shuffle(result)
    return tuple(result)


def _compose(
    first: Sequence[int],
    second: Sequence[int],
) -> tuple[int, ...]:
    return tuple(second[first[state]] for state in range(STATE_COUNT))


def _identity() -> tuple[int, ...]:
    return tuple(range(STATE_COUNT))


def _closure(generators: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    seen = {_identity()}
    frontier = [_identity()]
    while frontier:
        left = frontier.pop()
        for right in generators:
            for candidate in (_compose(left, right), _compose(right, left)):
                if candidate not in seen:
                    seen.add(candidate)
                    frontier.append(candidate)
        if len(seen) > 40_320:
            raise IdentifiableBoardError("permutation closure exceeds S8")
    return tuple(sorted(seen))


def _closure_exceeds(
    generators: Sequence[tuple[int, ...]],
    limit: int,
) -> bool:
    seen = {_identity()}
    frontier = [_identity()]
    while frontier:
        left = frontier.pop()
        for right in generators:
            for candidate in (_compose(left, right), _compose(right, left)):
                if candidate not in seen:
                    seen.add(candidate)
                    if len(seen) > limit:
                        return True
                    frontier.append(candidate)
    return False


def _reachable_count(actions: Sequence[Sequence[int]]) -> int:
    seen = {0}
    frontier = [0]
    while frontier:
        state = frontier.pop()
        for action in actions:
            destination = action[state]
            if destination not in seen:
                seen.add(destination)
                frontier.append(destination)
    return len(seen)


def _noncommuting(actions: Sequence[Sequence[int]]) -> bool:
    return any(
        _compose(actions[left], actions[right])
        != _compose(actions[right], actions[left])
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    )


def _affine_bases() -> tuple[tuple[int, ...], ...]:
    def rotate(value: int) -> int:
        x0, x1, x2 = tuple((value >> bit) & 1 for bit in range(3))
        return x1 | (x2 << 1) | (x0 << 2)

    def shear(value: int) -> int:
        x0, x1, x2 = tuple((value >> bit) & 1 for bit in range(3))
        return (x0 ^ x1) | (x1 << 1) | (x2 << 2)

    return (
        tuple(rotate(value) for value in range(STATE_COUNT)),
        tuple(shear(value) for value in range(STATE_COUNT)),
        tuple(value ^ 1 for value in range(STATE_COUNT)),
    )


def _dihedral_vertex_bases() -> tuple[tuple[int, ...], ...]:
    return (
        tuple((value + 1) % 8 for value in range(8)),
        tuple((-value) % 8 for value in range(8)),
    )


def _dihedral_regular_bases() -> tuple[tuple[int, ...], ...]:
    elements = tuple((rotation, reflection) for reflection in range(2) for rotation in range(4))
    index = {element: slot for slot, element in enumerate(elements)}

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        i, j = left
        k, ell = right
        return (i + (-1 if j else 1) * k) % 4, j ^ ell

    def left_action(element: tuple[int, int]) -> tuple[int, ...]:
        return tuple(index[multiply(element, state)] for state in elements)

    return left_action((1, 0)), left_action((0, 1))


def _quaternion_regular_bases() -> tuple[tuple[int, ...], ...]:
    # (sign, basis), with basis 0=1, 1=i, 2=j, 3=k.
    elements = tuple((sign, basis) for sign in (1, -1) for basis in range(4))
    index = {element: slot for slot, element in enumerate(elements)}
    table = (
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((1, 1), (-1, 0), (1, 3), (-1, 2)),
        ((1, 2), (-1, 3), (-1, 0), (1, 1)),
        ((1, 3), (1, 2), (-1, 1), (-1, 0)),
    )

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        sign, basis = table[left[1]][right[1]]
        return left[0] * right[0] * sign, basis

    def left_action(element: tuple[int, int]) -> tuple[int, ...]:
        return tuple(index[multiply(element, state)] for state in elements)

    return left_action((1, 1)), left_action((1, 2))


def _cube_rotation_bases() -> tuple[tuple[int, ...], ...]:
    vertices = tuple(
        (x, y, z)
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    )
    index = {vertex: slot for slot, vertex in enumerate(vertices)}

    def permutation(function) -> tuple[int, ...]:
        return tuple(index[function(*vertex)] for vertex in vertices)

    return (
        permutation(lambda x, y, z: (-y, x, z)),
        permutation(lambda x, y, z: (x, -z, y)),
        permutation(lambda x, y, z: (y, z, x)),
    )


def _family_actions(
    family: str,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    if family == "random-s8":
        for _ in range(4_096):
            actions = tuple(_shuffle(rng, range(STATE_COUNT)) for _ in range(ACTION_COUNT))
            if (
                len(set(actions)) == ACTION_COUNT
                and _reachable_count(actions) == STATE_COUNT
                and _noncommuting(actions)
                and _closure_exceeds(actions, 1_344)
            ):
                return actions
        raise IdentifiableBoardError("random S8 family did not admit a triple")
    base_by_family = {
        "affine-f2-3": _affine_bases,
        "dihedral-vertices": _dihedral_vertex_bases,
        "dihedral-regular": _dihedral_regular_bases,
        "quaternion-regular": _quaternion_regular_bases,
        "cube-rotations": _cube_rotation_bases,
    }
    try:
        group = _closure(base_by_family[family]())
    except KeyError as exc:
        raise IdentifiableBoardError("unknown action family") from exc
    candidates = tuple(element for element in group if element != _identity())
    for _ in range(4_096):
        actions = tuple(rng.sample(candidates, ACTION_COUNT))
        if (
            _reachable_count(actions) == STATE_COUNT
            and _noncommuting(actions)
            and len(_closure(actions)) == len(group)
        ):
            return actions
    raise IdentifiableBoardError(f"{family} did not admit an action triple")


def _future_class_count(
    transitions: Sequence[Sequence[int]],
    observations: Sequence[Sequence[int]],
) -> int:
    signatures = tuple(
        tuple(observer[state] for observer in observations)
        for state in range(STATE_COUNT)
    )
    classes = _class_ids(signatures)
    while True:
        refined = _class_ids(
            tuple(
                (
                    signatures[state],
                    tuple(classes[action[state]] for action in transitions),
                )
                for state in range(STATE_COUNT)
            )
        )
        if all(
            (refined[left] == refined[right])
            == (classes[left] == classes[right])
            for left in range(STATE_COUNT)
            for right in range(STATE_COUNT)
        ):
            return len(set(refined))
        classes = refined


def _class_ids(values: Sequence[object]) -> tuple[int, ...]:
    mapping: dict[object, int] = {}
    result: list[int] = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return tuple(result)


def _bfs_state_order(
    actions: Sequence[Sequence[int]],
    root: int,
) -> tuple[int, ...]:
    seen = {root}
    order = [root]
    for state in order:
        for action in actions:
            destination = action[state]
            if destination not in seen:
                seen.add(destination)
                order.append(destination)
    if len(order) != STATE_COUNT:
        raise IdentifiableBoardError("action family is not rooted-transitive")
    return tuple(order)


def canonical_action_bytes(
    transitions: Sequence[Sequence[int]],
) -> bytes:
    checked = tuple(tuple(int(cell) for cell in row) for row in transitions)
    if (
        len(checked) != ACTION_COUNT
        or any(sorted(row) != list(range(STATE_COUNT)) for row in checked)
        or _reachable_count(checked) != STATE_COUNT
    ):
        raise IdentifiableBoardError("action canonicalizer input differs")
    candidates: list[bytes] = []
    for action_order in itertools.permutations(range(ACTION_COUNT)):
        ordered = tuple(checked[index] for index in action_order)
        for root in range(STATE_COUNT):
            old_for_new = _bfs_state_order(ordered, root)
            old_to_new = {old: new for new, old in enumerate(old_for_new)}
            canonical = tuple(
                tuple(old_to_new[row[old]] for old in old_for_new)
                for row in ordered
            )
            candidates.append(bytes(cell for row in canonical for cell in row))
    return min(candidates)


def generate_machine(
    *,
    seed: str,
    split: str,
    index: int,
    family: str,
) -> IdentifiableMachine:
    rng = _domain_rng(seed, "machine", split, index, family)
    actions = _family_actions(family, rng)
    state_recode = _shuffle(rng, range(STATE_COUNT))
    inverse_state = {old: new for new, old in enumerate(state_recode)}
    actions = tuple(
        tuple(inverse_state[action[state_recode[new]]] for new in range(STATE_COUNT))
        for action in actions
    )
    actions = tuple(actions[index] for index in _shuffle(rng, range(ACTION_COUNT)))
    observations: tuple[tuple[int, ...], ...] | None = None
    for _ in range(4_096):
        rows = tuple(
            _shuffle(
                rng,
                (
                    answer
                    for answer in range(ANSWER_COUNT)
                    for _ in range(ANSWER_MULTIPLICITY)
                ),
            )
            for _ in range(OBSERVER_COUNT)
        )
        if _future_class_count(actions, rows) == STATE_COUNT:
            observations = rows
            break
    if observations is None:
        raise IdentifiableBoardError("observer generator did not separate states")
    keys: list[int] = []
    while len(keys) < STATE_COUNT + ACTION_COUNT + OBSERVER_COUNT:
        key = rng.getrandbits(64)
        if key and key not in keys:
            keys.append(key)
    return IdentifiableMachine(
        state_keys=tuple(keys[:STATE_COUNT]),
        action_keys=tuple(keys[STATE_COUNT : STATE_COUNT + ACTION_COUNT]),
        observer_keys=tuple(keys[-OBSERVER_COUNT:]),
        transitions=actions,
        observations=observations,
    )


def hide_one_cell_per_relation(
    machine: IdentifiableMachine,
    *,
    seed: str,
    split: str,
    index: int,
) -> PartialEvidence:
    rng = _domain_rng(seed, "partial", split, index)
    transition_events: list[tuple[int, int, int]] = []
    for action, action_key in enumerate(machine.action_keys):
        hidden = rng.randrange(STATE_COUNT)
        transition_events.extend(
            (action_key, machine.state_keys[state], machine.state_keys[destination])
            for state, destination in enumerate(machine.transitions[action])
            if state != hidden
        )
    observation_events: list[tuple[int, int, int]] = []
    for observer, observer_key in enumerate(machine.observer_keys):
        hidden = rng.randrange(STATE_COUNT)
        observation_events.extend(
            (observer_key, machine.state_keys[state], answer)
            for state, answer in enumerate(machine.observations[observer])
            if state != hidden
        )
    rng.shuffle(transition_events)
    rng.shuffle(observation_events)
    return PartialEvidence(
        state_keys=machine.state_keys,
        transition_events=tuple(transition_events),
        observation_events=tuple(observation_events),
    )


def _encode_key(key: int, codec: int) -> str:
    if codec == 0:
        return f"h{key:016x}"
    return f"d{key}"


def _decode_key(token: str) -> int:
    match = _KEY_HEX.fullmatch(token)
    if match is not None:
        value = int(match.group(1), 16)
    else:
        match = _KEY_DECIMAL.fullmatch(token)
        if match is None:
            raise IdentifiableBoardError("opaque key token is malformed")
        value = int(match.group(1))
    if value <= 0 or value >= 1 << 64:
        raise IdentifiableBoardError("opaque key token leaves uint64")
    return value


def encode_source(
    evidence: PartialEvidence,
    factors: GrammarFactors,
) -> bytes:
    def key(value: int) -> str:
        return _encode_key(value, factors.codec)

    if factors.organization == 0:
        records = [
            *(f"S {key(value)}" for value in evidence.state_keys),
            "LAW-A PERMUTATION",
            "LAW-O BALANCED 2 EACH 0 1 2 3",
            *(
                f"T {key(action)} {key(source)} {key(target)}"
                for action, source, target in evidence.transition_events
            ),
            *(
                f"O {key(observer)} {key(state)} {answer}"
                for observer, state, answer in evidence.observation_events
            ),
        ]
    else:
        records = [
            *(f"S key={key(value)}" for value in evidence.state_keys),
            "LAW-A kind=permutation",
            "LAW-O multiplicity=2 answers=0,1,2,3",
            *(
                f"T dst={key(target)} action={key(action)} src={key(source)}"
                for action, source, target in evidence.transition_events
            ),
            *(
                f"O answer={answer} state={key(state)} observer={key(observer)}"
                for observer, state, answer in evidence.observation_events
            ),
        ]
    if factors.framing == 0:
        return ("BEGIN-EFC\n" + "\n".join(records) + "\nEND-EFC\n").encode("ascii")
    return ("EFC{ " + " ; ".join(records) + " }\n").encode("ascii")


def _source_records(payload: bytes) -> tuple[str, ...]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise IdentifiableBoardError("source is not ASCII") from exc
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise IdentifiableBoardError("source framing is noncanonical")
    if text.startswith("BEGIN-EFC\n") and text.endswith("END-EFC\n"):
        records = tuple(text[len("BEGIN-EFC\n") : -len("END-EFC\n")].splitlines())
    elif text.startswith("EFC{ ") and text.endswith(" }\n"):
        records = tuple(text[len("EFC{ ") : -len(" }\n")].split(" ; "))
    else:
        raise IdentifiableBoardError("source wrapper is unknown")
    if not records or any(not record for record in records):
        raise IdentifiableBoardError("source has an empty record")
    return records


def decode_source(payload: bytes) -> PartialEvidence:
    states: list[int] = []
    transitions: list[tuple[int, int, int]] = []
    observations: list[tuple[int, int, int]] = []
    action_law = False
    observer_law = False
    for record in _source_records(payload):
        fields = record.split()
        if fields[0] == "S":
            if len(fields) == 2 and "=" not in fields[1]:
                states.append(_decode_key(fields[1]))
            elif len(fields) == 2 and fields[1].startswith("key="):
                states.append(_decode_key(fields[1][4:]))
            else:
                raise IdentifiableBoardError("state record is malformed")
        elif record in ("LAW-A PERMUTATION", "LAW-A kind=permutation"):
            action_law = True
        elif record in (
            "LAW-O BALANCED 2 EACH 0 1 2 3",
            "LAW-O multiplicity=2 answers=0,1,2,3",
        ):
            observer_law = True
        elif fields[0] == "T":
            if len(fields) == 4 and "=" not in record:
                transitions.append(tuple(_decode_key(value) for value in fields[1:]))
            elif len(fields) == 4:
                mapping = _assignment_fields(
                    fields[1:],
                    expected=("dst", "action", "src"),
                )
                transitions.append(
                    (
                        _decode_key(mapping["action"]),
                        _decode_key(mapping["src"]),
                        _decode_key(mapping["dst"]),
                    )
                )
            else:
                raise IdentifiableBoardError("transition record is malformed")
        elif fields[0] == "O":
            if len(fields) == 4 and "=" not in record:
                observations.append(
                    (_decode_key(fields[1]), _decode_key(fields[2]), _answer(fields[3]))
                )
            elif len(fields) == 4:
                mapping = _assignment_fields(
                    fields[1:],
                    expected=("answer", "state", "observer"),
                )
                observations.append(
                    (
                        _decode_key(mapping["observer"]),
                        _decode_key(mapping["state"]),
                        _answer(mapping["answer"]),
                    )
                )
            else:
                raise IdentifiableBoardError("observation record is malformed")
        else:
            raise IdentifiableBoardError("source record kind is unknown")
    if not action_law or not observer_law:
        raise IdentifiableBoardError("source omits a completion law")
    return PartialEvidence(
        state_keys=tuple(states),
        transition_events=tuple(transitions),
        observation_events=tuple(observations),
    )


def _assignment_fields(
    fields: Sequence[str],
    *,
    expected: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    names: list[str] = []
    for field in fields:
        if field.count("=") != 1:
            raise IdentifiableBoardError("assignment field is malformed")
        name, value = field.split("=")
        if not name or not value or name in result:
            raise IdentifiableBoardError("assignment field is duplicated")
        names.append(name)
        result[name] = value
    if tuple(names) != tuple(expected):
        raise IdentifiableBoardError("assignment field order or names differ")
    return result


def _answer(token: str) -> int:
    if token not in tuple(str(value) for value in range(ANSWER_COUNT)):
        raise IdentifiableBoardError("answer token leaves the frozen alphabet")
    return int(token)


def solve_unique_completion(evidence: PartialEvidence) -> IdentifiableMachine:
    states = tuple(sorted(evidence.state_keys))
    state_set = set(states)
    action_keys = tuple(sorted({event[0] for event in evidence.transition_events}))
    observer_keys = tuple(sorted({event[0] for event in evidence.observation_events}))
    if len(action_keys) != ACTION_COUNT or len(observer_keys) != OBSERVER_COUNT:
        raise IdentifiableBoardError("partial evidence role cardinality differs")
    state_index = {key: index for index, key in enumerate(states)}
    transitions: list[tuple[int, ...]] = []
    for action_key in action_keys:
        cells = {
            source: target
            for action, source, target in evidence.transition_events
            if action == action_key
        }
        if (
            len(cells) != STATE_COUNT - 1
            or not set(cells).issubset(state_set)
            or not set(cells.values()).issubset(state_set)
            or len(set(cells.values())) != STATE_COUNT - 1
        ):
            raise IdentifiableBoardError("action evidence is not uniquely completable")
        missing_source = (state_set - set(cells)).pop()
        missing_target = (state_set - set(cells.values())).pop()
        cells[missing_source] = missing_target
        transitions.append(tuple(state_index[cells[state]] for state in states))
    observations: list[tuple[int, ...]] = []
    for observer_key in observer_keys:
        cells = {
            state: answer
            for observer, state, answer in evidence.observation_events
            if observer == observer_key
        }
        if len(cells) != STATE_COUNT - 1 or not set(cells).issubset(state_set):
            raise IdentifiableBoardError("observer evidence is not uniquely completable")
        counts = {answer: 0 for answer in range(ANSWER_COUNT)}
        for answer in cells.values():
            if answer not in counts:
                raise IdentifiableBoardError("observer answer leaves alphabet")
            counts[answer] += 1
        missing_answers = [
            answer
            for answer, count in counts.items()
            if count == ANSWER_MULTIPLICITY - 1
        ]
        if (
            len(missing_answers) != 1
            or any(
                count not in (ANSWER_MULTIPLICITY - 1, ANSWER_MULTIPLICITY)
                for count in counts.values()
            )
        ):
            raise IdentifiableBoardError("observer balance does not identify one answer")
        missing_state = (state_set - set(cells)).pop()
        cells[missing_state] = missing_answers[0]
        observations.append(tuple(cells[state] for state in states))
    return IdentifiableMachine(
        state_keys=states,
        action_keys=action_keys,
        observer_keys=observer_keys,
        transitions=tuple(transitions),
        observations=tuple(observations),
    )


def encode_query(query: LateQuery, factors: GrammarFactors) -> bytes:
    def key(value: int) -> str:
        return _encode_key(value, factors.codec)

    if factors.organization == 0:
        records = (
            f"START {key(query.start_key)}",
            *(f"STEP {key(action)}" for action in query.action_keys),
            f"READ {key(query.observer_key)}",
        )
    else:
        path = ",".join(key(action) for action in query.action_keys) or "-"
        records = (
            f"start={key(query.start_key)}",
            f"path={path}",
            f"observer={key(query.observer_key)}",
        )
    if factors.framing == 0:
        return ("BEGIN-Q\n" + "\n".join(records) + "\nEND-Q\n").encode("ascii")
    return ("Q{ " + " ; ".join(records) + " }\n").encode("ascii")


def decode_query(payload: bytes) -> LateQuery:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise IdentifiableBoardError("query is not ASCII") from exc
    if text.startswith("BEGIN-Q\n") and text.endswith("END-Q\n"):
        records = tuple(text[len("BEGIN-Q\n") : -len("END-Q\n")].splitlines())
    elif text.startswith("Q{ ") and text.endswith(" }\n"):
        records = tuple(text[len("Q{ ") : -len(" }\n")].split(" ; "))
    else:
        raise IdentifiableBoardError("query wrapper is unknown")
    if not records:
        raise IdentifiableBoardError("query is empty")
    if records[0].startswith("START "):
        if not records[-1].startswith("READ "):
            raise IdentifiableBoardError("positional query omits observer")
        start = _decode_key(records[0][6:])
        actions = tuple(
            _decode_key(record[5:])
            for record in records[1:-1]
            if record.startswith("STEP ")
        )
        if len(actions) != len(records) - 2:
            raise IdentifiableBoardError("positional query has unknown record")
        observer = _decode_key(records[-1][5:])
    else:
        if len(records) != 3:
            raise IdentifiableBoardError("organized query field count differs")
        mapping = _assignment_fields(
            records,
            expected=("start", "path", "observer"),
        )
        start = _decode_key(mapping["start"])
        actions = (
            ()
            if mapping["path"] == "-"
            else tuple(_decode_key(token) for token in mapping["path"].split(","))
        )
        observer = _decode_key(mapping["observer"])
    return LateQuery(start_key=start, action_keys=actions, observer_key=observer)


def execute_query(machine: IdentifiableMachine, query: LateQuery) -> int:
    state_index = {key: index for index, key in enumerate(machine.state_keys)}
    action_index = {key: index for index, key in enumerate(machine.action_keys)}
    observer_index = {key: index for index, key in enumerate(machine.observer_keys)}
    try:
        state = state_index[query.start_key]
        for action_key in query.action_keys:
            state = machine.transitions[action_index[action_key]][state]
        return machine.observations[observer_index[query.observer_key]][state]
    except KeyError as exc:
        raise IdentifiableBoardError("late query references an unknown key") from exc


def generate_pilot_rows(
    *,
    seed: str,
    counts: dict[str, int],
) -> tuple[PilotRow, ...]:
    allowed_splits = ("train", "mechanics", "development", "confirmation")
    if set(counts) != set(allowed_splits) or any(counts[split] < 1 for split in counts):
        raise IdentifiableBoardError("pilot split counts differ")
    rows: list[PilotRow] = []
    structural_seen: set[str] = set()
    split_families = {
        "train": FAMILIES[:3],
        "mechanics": FAMILIES,
        "development": FAMILIES[:4],
        "confirmation": FAMILIES,
    }
    split_factors = {
        "train": TRAIN_FACTOR_COMBINATIONS,
        "mechanics": SOURCE_FACTOR_COMBINATIONS,
        "development": HELD_FACTOR_COMBINATIONS[:3],
        "confirmation": (HELD_FACTOR_COMBINATIONS[-1],),
    }
    for split in allowed_splits:
        accepted = 0
        attempt = 0
        while accepted < counts[split]:
            family = split_families[split][attempt % len(split_families[split])]
            machine = generate_machine(
                seed=seed,
                split=split,
                index=attempt,
                family=family,
            )
            canonical = sha256(machine.canonical_structural_bytes()).hexdigest()
            attempt += 1
            if canonical in structural_seen:
                continue
            evidence = hide_one_cell_per_relation(
                machine,
                seed=seed,
                split=split,
                index=attempt - 1,
            )
            world_id = f"{split}-{accepted:08d}-{canonical[:16]}"
            for factor_values in split_factors[split]:
                factors = GrammarFactors(*factor_values)
                source = encode_source(evidence, factors)
                decoded = decode_source(source)
                solved = solve_unique_completion(decoded)
                if (
                    solved.canonical_structural_bytes()
                    != machine.canonical_structural_bytes()
                ):
                    raise IdentifiableBoardError(
                        "direct solver changed world semantics"
                    )
                rows.append(
                    PilotRow(
                        world_id=world_id,
                        split=split,
                        family=family,
                        factors=factors,
                        source=source,
                        machine=machine,
                        canonical_sha256=canonical,
                    )
                )
            structural_seen.add(canonical)
            accepted += 1
            if attempt > 100_000:
                raise IdentifiableBoardError("pilot split exhausted unique worlds")
    return tuple(rows)


def project_candidate_sources(
    rows: Sequence[PilotRow],
    *,
    split: str,
) -> tuple[CandidateSource, ...]:
    selected = tuple(
        CandidateSource(row.source)
        for row in rows
        if row.split == split
    )
    if not selected:
        raise IdentifiableBoardError("candidate projection split is empty")
    return selected


def resource_receipt(max_depth: int) -> dict[str, int | float]:
    if not 0 <= max_depth <= 32:
        raise IdentifiableBoardError("resource depth leaves support")
    coordinates = (
        STATE_COUNT
        * OBSERVER_COUNT
        * sum(ACTION_COUNT**depth for depth in range(max_depth + 1))
    )
    answer_bits = coordinates * 2
    machine_bits = 1_536 * 8
    return {
        "answer_bits": answer_bits,
        "coordinates": coordinates,
        "deployed_machine_bits": machine_bits,
        "answer_to_machine_ratio": answer_bits / machine_bits,
    }


__all__ = [
    "ACTION_COUNT",
    "ANSWER_COUNT",
    "CandidateSource",
    "FAMILIES",
    "GrammarFactors",
    "HELD_FACTOR_COMBINATIONS",
    "IdentifiableBoardError",
    "IdentifiableMachine",
    "LateQuery",
    "OBSERVER_COUNT",
    "PartialEvidence",
    "PilotRow",
    "SOURCE_FACTOR_COMBINATIONS",
    "STATE_COUNT",
    "TRAIN_FACTOR_COMBINATIONS",
    "canonical_action_bytes",
    "decode_query",
    "decode_source",
    "encode_query",
    "encode_source",
    "execute_query",
    "generate_machine",
    "generate_pilot_rows",
    "hide_one_cell_per_relation",
    "project_candidate_sources",
    "resource_receipt",
    "solve_unique_completion",
]
