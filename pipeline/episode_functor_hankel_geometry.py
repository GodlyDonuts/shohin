"""Board-free finite word geometry for neural Hankel-shift arms."""

from __future__ import annotations

from hashlib import sha256
import itertools
import random


ACTION_COUNT = 3
ActionWord = tuple[int, ...]
ShiftIncidence = tuple[tuple[int, ...], ...]


class HankelGeometryError(ValueError):
    """A finite action word or shift incidence left frozen support."""


def enumerate_action_words(max_depth: int) -> tuple[ActionWord, ...]:
    """Enumerate words by length, then lexicographically within each length."""

    if not 0 <= max_depth <= 8:
        raise HankelGeometryError("Hankel depth leaves finite support")
    return tuple(
        word
        for length in range(max_depth + 1)
        for word in itertools.product(range(ACTION_COUNT), repeat=length)
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
        raise HankelGeometryError("shift incidence geometry differs")
    for action_row in incidence:
        for word, target in zip(words, action_row, strict=True):
            if (
                target not in range(len(extended))
                or len(extended[target]) != len(word) + 1
            ):
                raise HankelGeometryError(
                    "shift incidence violates word-length stratum"
                )


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


__all__ = [
    "HankelGeometryError",
    "commutative_bag_incidence",
    "enumerate_action_words",
    "prefix_shift_incidence",
    "random_shift_incidence",
]
