"""Exact finite Hankel-shift mechanics for the identifiable EFC board.

This module is an oracle/mechanics implementation, not a candidate neural
boundary.  It turns each anonymous state into the finite table of observations
reachable after every action word through a chosen depth.  Applying an action
is represented by prefixing that action to every probe word.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import random
from typing import Sequence

from pipeline.episode_functor_identifiable_board import (
    ACTION_COUNT,
    ANSWER_COUNT,
    IdentifiableMachine,
    OBSERVER_COUNT,
    STATE_COUNT,
)


ActionWord = tuple[int, ...]
StateSignature = tuple[tuple[tuple[int, ...], ...], ...]
ShiftSignature = tuple[tuple[StateSignature, ...], ...]
ShiftIncidence = tuple[tuple[int, ...], ...]


class HankelShiftError(ValueError):
    """A finite behavioral code or shift-decoding invariant failed."""


@dataclass(frozen=True, slots=True)
class HankelCodebook:
    """Base and action-derivative behavioral signatures for one machine."""

    depth: int
    words: tuple[ActionWord, ...]
    base: tuple[StateSignature, ...]
    derivative: ShiftSignature
    incidence: ShiftIncidence

    def __post_init__(self) -> None:
        expected_words = sum(ACTION_COUNT**length for length in range(self.depth + 1))
        if (
            not 0 <= self.depth <= 8
            or self.words != enumerate_action_words(self.depth)
            or len(self.words) != expected_words
            or len(self.base) != STATE_COUNT
            or len(self.derivative) != ACTION_COUNT
            or any(len(row) != STATE_COUNT for row in self.derivative)
            or len(self.incidence) != ACTION_COUNT
            or any(len(row) != expected_words for row in self.incidence)
        ):
            raise HankelShiftError("Hankel codebook geometry differs")
        _validate_incidence(self.incidence, self.depth)
        for signature in (
            *self.base,
            *(
                signature
                for action_row in self.derivative
                for signature in action_row
            ),
        ):
            _validate_signature(signature, expected_words)

    @property
    def coordinate_count(self) -> int:
        return len(self.words) * OBSERVER_COUNT


@dataclass(frozen=True, slots=True)
class HankelDecode:
    """Decoded categorical machine tables and their selected syndromes."""

    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]
    distances: tuple[tuple[tuple[int, ...], ...], ...]
    syndromes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.transitions) != ACTION_COUNT
            or any(len(row) != STATE_COUNT for row in self.transitions)
            or len(self.observations) != OBSERVER_COUNT
            or any(len(row) != STATE_COUNT for row in self.observations)
            or len(self.distances) != ACTION_COUNT
            or any(len(row) != STATE_COUNT for row in self.distances)
            or any(
                len(candidates) != STATE_COUNT
                for row in self.distances
                for candidates in row
            )
            or len(self.syndromes) != ACTION_COUNT
            or any(len(row) != STATE_COUNT for row in self.syndromes)
        ):
            raise HankelShiftError("Hankel decode geometry differs")


def enumerate_action_words(max_depth: int) -> tuple[ActionWord, ...]:
    """Enumerate words by length, then lexicographically within each length."""

    if not 0 <= max_depth <= 8:
        raise HankelShiftError("Hankel depth leaves finite support")
    return tuple(
        word
        for length in range(max_depth + 1)
        for word in itertools.product(range(ACTION_COUNT), repeat=length)
    )


def execute_action_word(
    machine: IdentifiableMachine,
    state: int,
    word: Sequence[int],
) -> int:
    """Execute actions left-to-right, matching ``execute_query`` semantics."""

    if state not in range(STATE_COUNT):
        raise HankelShiftError("Hankel start state leaves support")
    result = state
    for action in word:
        if action not in range(ACTION_COUNT):
            raise HankelShiftError("Hankel action leaves support")
        result = machine.transitions[action][result]
    return result


def prefix_shift_incidence(max_depth: int) -> ShiftIncidence:
    """Map ``(action, word)`` to the extended word ``(action, *word)``."""

    words = enumerate_action_words(max_depth)
    extended = enumerate_action_words(max_depth + 1)
    index = {word: position for position, word in enumerate(extended)}
    return tuple(
        tuple(index[(action, *word)] for word in words)
        for action in range(ACTION_COUNT)
    )


def random_shift_incidence(
    max_depth: int,
    *,
    seed: str,
) -> ShiftIncidence:
    """Seeded position-scramble control preserving action equivariance."""

    words = enumerate_action_words(max_depth)
    extended = enumerate_action_words(max_depth + 1)
    index = {word: position for position, word in enumerate(extended)}
    rng = random.Random(
        int.from_bytes(
            sha256(f"{seed}\0{max_depth}".encode("utf-8")).digest(),
            "big",
        )
    )
    position_permutations: dict[int, tuple[int, ...]] = {}
    for length in range(max_depth + 1):
        permutation = list(range(length + 1))
        rng.shuffle(permutation)
        position_permutations[length] = tuple(permutation)
    incidence = tuple(
        tuple(
            index[
                tuple(
                    (action, *word)[position]
                    for position in position_permutations[len(word)]
                )
            ]
            for word in words
        )
        for action in range(ACTION_COUNT)
    )
    _validate_incidence(incidence, max_depth)
    return incidence


def commutative_bag_incidence(max_depth: int) -> ShiftIncidence:
    """Equivariant stable-bag control that removes repeated-symbol order."""

    words = enumerate_action_words(max_depth)
    extended = enumerate_action_words(max_depth + 1)
    index = {word: position for position, word in enumerate(extended)}
    def stable_bag(word: ActionWord) -> ActionWord:
        order = tuple(dict.fromkeys(word))
        return tuple(
            symbol
            for first in order
            for symbol in word
            if symbol == first
        )

    return tuple(
        tuple(index[stable_bag((action, *word))] for word in words)
        for action in range(ACTION_COUNT)
    )


def _validate_incidence(
    incidence: ShiftIncidence,
    max_depth: int,
) -> None:
    words = enumerate_action_words(max_depth)
    extended = enumerate_action_words(max_depth + 1)
    if (
        len(incidence) != ACTION_COUNT
        or any(len(row) != len(words) for row in incidence)
    ):
        raise HankelShiftError("shift incidence geometry differs")
    for action_row in incidence:
        for word, target in zip(words, action_row, strict=True):
            if (
                target not in range(len(extended))
                or len(extended[target]) != len(word) + 1
            ):
                raise HankelShiftError(
                    "shift incidence violates word-length stratum"
                )


def _state_signature(
    machine: IdentifiableMachine,
    state: int,
    words: Sequence[ActionWord],
) -> StateSignature:
    return tuple(
        tuple(
            machine.observations[observer][
                execute_action_word(machine, state, word)
            ]
            for observer in range(OBSERVER_COUNT)
        )
        for word in words
    )


def build_hankel_codebook(
    machine: IdentifiableMachine,
    *,
    max_depth: int = 3,
    incidence: ShiftIncidence | None = None,
) -> HankelCodebook:
    """Build a base code and an incidence-selected derivative code."""

    words = enumerate_action_words(max_depth)
    extended_words = enumerate_action_words(max_depth + 1)
    selected_incidence = (
        prefix_shift_incidence(max_depth)
        if incidence is None
        else incidence
    )
    _validate_incidence(selected_incidence, max_depth)
    extended = tuple(
        _state_signature(machine, state, extended_words)
        for state in range(STATE_COUNT)
    )
    base = tuple(
        tuple(signature[: len(words)])
        for signature in extended
    )
    derivative = tuple(
        tuple(
            tuple(extended[state][coordinate] for coordinate in action_row)
            for state in range(STATE_COUNT)
        )
        for action_row in selected_incidence
    )
    return HankelCodebook(
        depth=max_depth,
        words=words,
        base=base,
        derivative=derivative,
        incidence=selected_incidence,
    )


def _validate_signature(
    signature: StateSignature,
    word_count: int,
) -> None:
    if (
        len(signature) != word_count
        or any(len(row) != OBSERVER_COUNT for row in signature)
        or any(
            answer not in range(ANSWER_COUNT)
            for row in signature
            for answer in row
        )
    ):
        raise HankelShiftError("state signature geometry or answer differs")


def signature_hamming(
    left: StateSignature,
    right: StateSignature,
) -> int:
    if len(left) != len(right):
        raise HankelShiftError("signature word inventories differ")
    _validate_signature(left, len(left))
    _validate_signature(right, len(right))
    return sum(
        left_answer != right_answer
        for left_row, right_row in zip(left, right, strict=True)
        for left_answer, right_answer in zip(left_row, right_row, strict=True)
    )


def minimum_signature_distance(codebook: HankelCodebook) -> int:
    return min(
        signature_hamming(codebook.base[left], codebook.base[right])
        for left in range(STATE_COUNT)
        for right in range(left + 1, STATE_COUNT)
    )


def derivative_only_correction_radius(codebook: HankelCodebook) -> int:
    """Guaranteed radius when only a derivative signature is corrupted."""

    return (minimum_signature_distance(codebook) - 1) // 2


def joint_codebook_correction_radius(codebook: HankelCodebook) -> int:
    """Guaranteed per-code radius when derivative and base can both err.

    If every compared base code and the derivative code differ from truth in
    at most ``e`` coordinates, the true match is within ``2e`` and every false
    match is at least ``d_min - 2e`` away.  Strict nearest-neighbor recovery
    therefore requires ``4e < d_min``.
    """

    return (minimum_signature_distance(codebook) - 1) // 4


def decode_hankel_shifts(
    codebook: HankelCodebook,
    *,
    require_unique: bool = True,
) -> HankelDecode:
    """Decode transitions by nearest future signature and observers at epsilon."""

    distance_rows: list[tuple[tuple[int, ...], ...]] = []
    transition_rows: list[tuple[int, ...]] = []
    syndrome_rows: list[tuple[int, ...]] = []
    for action in range(ACTION_COUNT):
        action_distances: list[tuple[int, ...]] = []
        action_transitions: list[int] = []
        action_syndromes: list[int] = []
        for state in range(STATE_COUNT):
            candidates = tuple(
                signature_hamming(
                    codebook.derivative[action][state],
                    codebook.base[target],
                )
                for target in range(STATE_COUNT)
            )
            minimum = min(candidates)
            winners = tuple(
                target
                for target, distance in enumerate(candidates)
                if distance == minimum
            )
            if require_unique and len(winners) != 1:
                raise HankelShiftError(
                    "shift decoder has a coordinate-dependent tie"
                )
            action_distances.append(candidates)
            action_transitions.append(winners[0])
            action_syndromes.append(minimum)
        distance_rows.append(tuple(action_distances))
        transition_rows.append(tuple(action_transitions))
        syndrome_rows.append(tuple(action_syndromes))
    observations = tuple(
        tuple(codebook.base[state][0][observer] for state in range(STATE_COUNT))
        for observer in range(OBSERVER_COUNT)
    )
    return HankelDecode(
        transitions=tuple(transition_rows),
        observations=observations,
        distances=tuple(distance_rows),
        syndromes=tuple(syndrome_rows),
    )


def exact_codebook_receipt(
    machines: Sequence[IdentifiableMachine],
    *,
    max_depth: int = 3,
) -> dict[str, int]:
    """Summarize exact separation and corruption margins over frozen worlds."""

    frozen = tuple(machines)
    if not frozen:
        raise HankelShiftError("Hankel receipt has no machines")
    distances = tuple(
        minimum_signature_distance(
            build_hankel_codebook(machine, max_depth=max_depth)
        )
        for machine in frozen
    )
    return {
        "coordinate_count": (
            len(enumerate_action_words(max_depth)) * OBSERVER_COUNT
        ),
        "derivative_only_radius": (min(distances) - 1) // 2,
        "joint_codebook_radius": (min(distances) - 1) // 4,
        "maximum_minimum_distance": max(distances),
        "minimum_distance": min(distances),
        "separated_machines": sum(distance > 0 for distance in distances),
        "world_count": len(frozen),
    }


__all__ = [
    "ActionWord",
    "HankelCodebook",
    "HankelDecode",
    "HankelShiftError",
    "ShiftIncidence",
    "build_hankel_codebook",
    "commutative_bag_incidence",
    "decode_hankel_shifts",
    "derivative_only_correction_radius",
    "enumerate_action_words",
    "exact_codebook_receipt",
    "execute_action_word",
    "joint_codebook_correction_radius",
    "minimum_signature_distance",
    "prefix_shift_incidence",
    "random_shift_incidence",
    "signature_hamming",
]
