"""Exact mechanics for closure-tied categorical action algebras.

The module defines a finite reference system for the neural CTAA hypothesis. It
contains no learned executor and is not a capability implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable, Sequence


State = tuple[int, ...]
CopyAction = tuple[int, ...]


def all_states(width: int, alphabet: int) -> tuple[State, ...]:
    if width < 1 or alphabet < 2:
        raise ValueError("CTAA state geometry differs")
    return tuple(product(range(alphabet), repeat=width))


def all_copy_actions(width: int) -> tuple[CopyAction, ...]:
    if width < 1:
        raise ValueError("CTAA action width differs")
    return tuple(product(range(width), repeat=width))


def validate_action(action: CopyAction, width: int) -> None:
    if len(action) != width or any(index < 0 or index >= width for index in action):
        raise ValueError("CTAA copy action differs")


def apply_action(action: CopyAction, state: State) -> State:
    validate_action(action, len(state))
    return tuple(state[index] for index in action)


def compose_actions(after: CopyAction, before: CopyAction) -> CopyAction:
    """Return the action equivalent to applying ``before`` then ``after``."""
    if len(after) != len(before):
        raise ValueError("CTAA action composition width differs")
    validate_action(after, len(before))
    validate_action(before, len(before))
    return tuple(before[index] for index in after)


def identity_action(width: int) -> CopyAction:
    if width < 1:
        raise ValueError("CTAA identity width differs")
    return tuple(range(width))


def relabel_values(state: State, permutation: Sequence[int]) -> State:
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("CTAA value permutation differs")
    if any(value < 0 or value >= len(permutation) for value in state):
        raise ValueError("CTAA state value leaves permutation alphabet")
    return tuple(permutation[value] for value in state)


def reindex_state(state: State, permutation: Sequence[int]) -> State:
    if sorted(permutation) != list(range(len(state))):
        raise ValueError("CTAA position permutation differs")
    return tuple(state[index] for index in permutation)


def reindex_action(action: CopyAction, permutation: Sequence[int]) -> CopyAction:
    """Conjugate an action into a permuted storage coordinate system."""
    width = len(action)
    if sorted(permutation) != list(range(width)):
        raise ValueError("CTAA action reindex permutation differs")
    inverse = [0] * width
    for new_index, old_index in enumerate(permutation):
        inverse[old_index] = new_index
    return tuple(inverse[action[old_index]] for old_index in permutation)


def behavioral_signature(
    state: State,
    continuations: Sequence[CopyAction],
    *,
    query_indices: Sequence[int] = (0,),
) -> tuple[int, ...]:
    """Read only frozen query coordinates after each continuation."""
    if not query_indices or any(
        index < 0 or index >= len(state) for index in query_indices
    ):
        raise ValueError("CTAA query basis differs")
    return tuple(
        output[index]
        for action in continuations
        for output in (apply_action(action, state),)
        for index in query_indices
    )


@dataclass(frozen=True)
class ActionPacket:
    actions: tuple[CopyAction, ...]
    halt_at: int

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("CTAA packet is empty")
        width = len(self.actions[0])
        if width < 1:
            raise ValueError("CTAA packet action width differs")
        if any(len(action) != width for action in self.actions):
            raise ValueError("CTAA packet action widths differ")
        for action in self.actions:
            validate_action(action, width)
        if self.halt_at < 0 or self.halt_at > len(self.actions):
            raise ValueError("CTAA halt boundary differs")


@dataclass(frozen=True)
class ExecutionTrace:
    states: tuple[State, ...]
    halted: tuple[bool, ...]


def execute_packet(
    packet: ActionPacket,
    initial: State,
    *,
    suffix: Iterable[CopyAction] = (),
) -> ExecutionTrace:
    """Execute hard actions with an absorbing interpreter halt."""
    suffix_actions = tuple(suffix)
    width = len(packet.actions[0])
    if len(initial) != width:
        raise ValueError("CTAA initial state width differs")
    for action in suffix_actions:
        validate_action(action, width)
    state = initial
    states = [state]
    halted = [packet.halt_at == 0]
    schedule = packet.actions + suffix_actions
    for step, action in enumerate(schedule):
        is_halted = step >= packet.halt_at
        if not is_halted:
            state = apply_action(action, state)
        states.append(state)
        halted.append(is_halted or step + 1 >= packet.halt_at)
    return ExecutionTrace(states=tuple(states), halted=tuple(halted))


def value_permutations(alphabet: int) -> tuple[tuple[int, ...], ...]:
    if alphabet < 2:
        raise ValueError("CTAA alphabet differs")
    return tuple(permutations(range(alphabet)))


def position_permutations(width: int) -> tuple[tuple[int, ...], ...]:
    if width < 1:
        raise ValueError("CTAA position width differs")
    return tuple(permutations(range(width)))
