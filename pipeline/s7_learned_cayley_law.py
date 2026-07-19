"""Exact mechanics for S7 learned-Cayley contextual law compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from s6_contextual_affine_law import AffineLaw, pop_insert, validate_state


PRIMARY_MODULI = (5, 7, 11)
DIAGNOSTIC_MODULUS = 13


@dataclass(frozen=True)
class SymbolBinding:
    """An arbitrary observed-symbol to latent-residue bijection."""

    modulus: int
    observed_to_latent: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.observed_to_latent) != self.modulus:
            raise ValueError("S7 binding length does not match modulus")
        if set(self.observed_to_latent) != set(range(self.modulus)):
            raise ValueError("S7 binding must be a complete permutation")

    @property
    def latent_to_observed(self) -> tuple[int, ...]:
        inverse = [0] * self.modulus
        for observed, latent in enumerate(self.observed_to_latent):
            inverse[latent] = observed
        return tuple(inverse)

    @property
    def zero_symbol(self) -> int:
        return self.latent_to_observed[0]

    @property
    def successor(self) -> tuple[int, ...]:
        inverse = self.latent_to_observed
        result = []
        for observed, latent in enumerate(self.observed_to_latent):
            next_latent = 0 if latent + 1 == self.modulus else latent + 1
            result.append(inverse[next_latent])
        return tuple(result)

    def encode(self, latent: int) -> int:
        if not 0 <= latent < self.modulus:
            raise ValueError("S7 latent residue outside binding")
        return self.latent_to_observed[latent]

    def decode(self, observed: int) -> int:
        if not 0 <= observed < self.modulus:
            raise ValueError("S7 observed symbol outside binding")
        return self.observed_to_latent[observed]

    def card(self, law: AffineLaw) -> tuple[int, int]:
        if law.modulus != self.modulus:
            raise ValueError("S7 law/binding modulus mismatch")
        y0, y1 = law.card
        return self.encode(y0), self.encode(y1)

    def destination(self, law: AffineLaw, observed_position: int) -> int:
        if law.modulus != self.modulus:
            raise ValueError("S7 law/binding modulus mismatch")
        latent_position = self.decode(observed_position)
        return self.encode(law.destination(latent_position))


def validate_successor(successor: Sequence[int], zero_symbol: int) -> tuple[int, ...]:
    """Validate that successor is one directed cycle rooted at zero_symbol."""

    result = tuple(int(value) for value in successor)
    modulus = len(result)
    if set(result) != set(range(modulus)):
        raise ValueError("S7 successor must be a permutation")
    if not 0 <= zero_symbol < modulus:
        raise ValueError("S7 zero symbol outside successor")
    seen: list[int] = []
    cursor = zero_symbol
    for _ in result:
        if cursor in seen:
            raise ValueError("S7 successor closes before visiting every symbol")
        seen.append(cursor)
        cursor = result[cursor]
    if cursor != zero_symbol or len(seen) != modulus:
        raise ValueError("S7 successor is not one complete cycle")
    return result


def compile_destination(
    successor: Sequence[int],
    zero_symbol: int,
    card_y0: int,
    card_y1: int,
    current_symbol: int,
) -> int:
    """Compile and execute a law using only successor reuse and equality."""

    cycle = validate_successor(successor, zero_symbol)
    modulus = len(cycle)
    for symbol in (card_y0, card_y1, current_symbol):
        if not 0 <= symbol < modulus:
            raise ValueError("S7 compiler input outside successor")
    if card_y0 == card_y1:
        raise ValueError("S7 bijective law requires two distinct witnesses")

    probe = card_y0
    slope_symbol = zero_symbol
    for _ in cycle:
        probe = cycle[probe]
        slope_symbol = cycle[slope_symbol]
        if probe == card_y1:
            break
    else:
        raise ValueError("S7 card witnesses are disconnected")

    cursor = zero_symbol
    destination = card_y0
    for _ in cycle:
        if cursor == current_symbol:
            return destination
        inner_cursor = zero_symbol
        for _ in cycle:
            if inner_cursor == slope_symbol:
                break
            inner_cursor = cycle[inner_cursor]
            destination = cycle[destination]
        else:
            raise ValueError("S7 slope walk failed to terminate")
        cursor = cycle[cursor]
    if cursor == current_symbol:
        return destination
    raise ValueError("S7 current-symbol walk failed to terminate")


def stride_two_successor(successor: Sequence[int], zero_symbol: int) -> tuple[int, ...]:
    """Return the matched false generator S^2 for odd-cycle controls."""

    cycle = validate_successor(successor, zero_symbol)
    return tuple(cycle[cycle[symbol]] for symbol in range(len(cycle)))


def apply_compiled_law(
    state: Sequence[int],
    identity: int,
    successor: Sequence[int],
    zero_symbol: int,
    card_y0: int,
    card_y1: int,
) -> tuple[int, ...]:
    current = validate_state(state, len(successor))
    source = current.index(identity)
    destination = compile_destination(
        successor, zero_symbol, card_y0, card_y1, source
    )
    return pop_insert(current, identity, destination)


def execute_compiled_program(
    initial_state: Sequence[int],
    events: Iterable[tuple[int, int, int]],
    successor: Sequence[int],
    zero_symbol: int,
) -> tuple[int, ...]:
    state = tuple(initial_state)
    for identity, card_y0, card_y1 in events:
        state = apply_compiled_law(
            state, identity, successor, zero_symbol, card_y0, card_y1
        )
    return state
