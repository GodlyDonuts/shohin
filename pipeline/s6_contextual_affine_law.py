"""Exact mechanics for S6 contextual affine law induction.

This module contains no neural model.  It defines the finite-field law family,
the source-deleted categorical state update, and the immutable law split used by
the S6 falsifier and later board builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence


ADMITTED_MODULI = (5, 7, 11)
DIAGNOSTIC_MODULUS = 13
ALL_AUDIT_MODULI = ADMITTED_MODULI + (DIAGNOSTIC_MODULUS,)
SPLIT_PERSON = "s6-law-v1"


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


@dataclass(frozen=True, order=True)
class AffineLaw:
    """A bijective affine destination law over the prime field Z_m."""

    modulus: int
    slope: int
    intercept: int

    def __post_init__(self) -> None:
        if not is_prime(self.modulus):
            raise ValueError("S6 modulus must be prime")
        if not 0 < self.slope < self.modulus:
            raise ValueError("S6 slope must be nonzero and canonical")
        if not 0 <= self.intercept < self.modulus:
            raise ValueError("S6 intercept must be canonical")

    @property
    def card(self) -> tuple[int, int]:
        return self.destination(0), self.destination(1)

    @property
    def key(self) -> str:
        return f"m{self.modulus}_a{self.slope}_b{self.intercept}"

    def destination(self, position: int) -> int:
        if not 0 <= position < self.modulus:
            raise ValueError("position outside S6 modulus")
        return (self.slope * position + self.intercept) % self.modulus


def infer_affine_law(modulus: int, card_y0: int, card_y1: int) -> AffineLaw:
    """Recover the unique law identified by witnesses at positions zero and one."""

    if not is_prime(modulus):
        raise ValueError("S6 modulus must be prime")
    if not 0 <= card_y0 < modulus or not 0 <= card_y1 < modulus:
        raise ValueError("law-card value outside modulus")
    slope = (card_y1 - card_y0) % modulus
    if slope == 0:
        raise ValueError("equal witnesses do not define a bijective affine law")
    return AffineLaw(modulus=modulus, slope=slope, intercept=card_y0)


def enumerate_laws(modulus: int) -> tuple[AffineLaw, ...]:
    if not is_prime(modulus):
        raise ValueError("S6 modulus must be prime")
    return tuple(
        AffineLaw(modulus, slope, intercept)
        for slope in range(1, modulus)
        for intercept in range(modulus)
    )


def law_bucket(law: AffineLaw) -> int:
    payload = (
        f"{SPLIT_PERSON}|{law.modulus}|{law.slope}|{law.intercept}"
    ).encode("ascii")
    return hashlib.sha256(payload).digest()[0] % 5


def law_split(law: AffineLaw) -> str:
    bucket = law_bucket(law)
    if bucket == 0:
        return "development"
    if bucket == 1:
        return "confirmation"
    return "train"


def raw_split_laws(modulus: int) -> dict[str, tuple[AffineLaw, ...]]:
    split: dict[str, list[AffineLaw]] = {
        "train": [],
        "development": [],
        "confirmation": [],
    }
    for law in enumerate_laws(modulus):
        split[law_split(law)].append(law)
    return {name: tuple(values) for name, values in split.items()}


def _coverage_values(laws: Sequence[AffineLaw], field: str) -> set[int]:
    if field == "card_y0":
        return {law.card[0] for law in laws}
    if field == "card_y1":
        return {law.card[1] for law in laws}
    if field == "destination":
        return {
            law.destination(position)
            for law in laws
            for position in range(law.modulus)
        }
    raise ValueError(f"unknown S6 coverage field: {field}")


def repaired_split_laws(
    modulus: int,
) -> tuple[dict[str, tuple[AffineLaw, ...]], tuple[dict[str, object], ...]]:
    """Apply the frozen v1.1 coordinate-coverage repair to raw hash buckets."""

    raw = raw_split_laws(modulus)
    split = {name: list(values) for name, values in raw.items()}
    promotions: list[dict[str, object]] = []
    for field in ("card_y0", "card_y1", "destination"):
        for value in range(modulus):
            if value in _coverage_values(split["train"], field):
                continue
            selected: tuple[str, AffineLaw] | None = None
            for source in ("confirmation", "development"):
                if len(split[source]) <= 1:
                    continue
                candidates = [
                    law
                    for law in split[source]
                    if value in _coverage_values((law,), field)
                ]
                if candidates:
                    selected = source, min(candidates)
                    break
            if selected is None:
                raise ValueError(
                    f"cannot repair S6 {field}={value} coverage at modulus {modulus}"
                )
            source, law = selected
            split[source].remove(law)
            split["train"].append(law)
            promotions.append(
                {
                    "modulus": modulus,
                    "field": field,
                    "value": value,
                    "law": law.key,
                    "source": source,
                    "destination": "train",
                }
            )
    frozen = {name: tuple(sorted(values)) for name, values in split.items()}
    return frozen, tuple(promotions)


def split_laws(modulus: int) -> dict[str, tuple[AffineLaw, ...]]:
    split, _ = repaired_split_laws(modulus)
    return split


def one_witness_candidates(modulus: int, card_y0: int) -> tuple[AffineLaw, ...]:
    """Return every law consistent with only the zero-position witness."""

    if not 0 <= card_y0 < modulus:
        raise ValueError("law-card value outside modulus")
    return tuple(
        AffineLaw(modulus, slope, card_y0) for slope in range(1, modulus)
    )


def validate_state(state: Sequence[int], modulus: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in state)
    if len(result) != modulus or set(result) != set(range(modulus)):
        raise ValueError("S6 state must be a complete identity permutation")
    return result


def pop_insert(
    state: Sequence[int], identity: int, destination: int
) -> tuple[int, ...]:
    """Move one identity to an absolute destination while preserving all others."""

    current = list(validate_state(state, len(state)))
    if identity not in current:
        raise ValueError("identity missing from S6 state")
    if not 0 <= destination < len(current):
        raise ValueError("destination outside S6 state")
    current.insert(destination, current.pop(current.index(identity)))
    return tuple(current)


def apply_law(
    state: Sequence[int], identity: int, law: AffineLaw
) -> tuple[int, ...]:
    current = validate_state(state, law.modulus)
    source = current.index(identity)
    return pop_insert(current, identity, law.destination(source))


def execute_program(
    initial_state: Sequence[int], events: Iterable[tuple[int, AffineLaw]]
) -> tuple[int, ...]:
    state = tuple(initial_state)
    for identity, law in events:
        state = apply_law(state, identity, law)
    return state


def treatment_input(
    law: AffineLaw, current_location: int
) -> dict[str, int]:
    """The complete information legally visible to the learned S6 unit."""

    card_y0, card_y1 = law.card
    return {
        "modulus": law.modulus,
        "card_y0": card_y0,
        "card_y1": card_y1,
        "current_location": current_location,
    }
