"""Dual-code reversible scratchpad protocol for isolated causal research.

The protocol represents one digitwise machine state through two independently
permuted, per-episode token codes.  Dataset generation may use the semantic
solver below, but a runtime controller may only transport model text, parse its
grammar, and compare an exact recovered predecessor with its prior input.
It may not call ``apply_microstep`` or ``invert_microstep`` during rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from digitwise_protocol import apply_microstep, canonical_state


FIELDS = ("op", "w", "p", "c", "a", "b", "r", "z")
OPERATIONS = ("add", "sub")
FIELD_STEMS = ("ki", "lo", "mu", "ne", "pa", "ri", "so", "tu", "ve", "xa", "yo", "ze")
DIGIT_STEMS = ("ba", "ce", "di", "fo", "gu", "ha", "jo", "qu", "we", "xy", "ra", "si", "to", "vu")
OP_STEMS = ("nav", "sol", "ter", "via")


@dataclass(frozen=True)
class Codebook:
    """One rendering code for a semantic DWS state.

    Prefixes make A and B surface vocabularies disjoint.  Permutations, rather
    than the prefixes, carry the codebook-binding challenge: the same semantic
    field or decimal digit maps to a fresh alias on every episode.
    """

    channel: str
    field_order: tuple[str, ...]
    field_aliases: tuple[str, ...]
    digit_aliases: tuple[str, ...]
    operation_aliases: tuple[str, ...]

    def __post_init__(self):
        if self.channel not in {"A", "B"}:
            raise ValueError("channel must be A or B")
        if set(self.field_order) != set(FIELDS) or len(self.field_order) != len(FIELDS):
            raise ValueError("field order must be a permutation of canonical fields")
        if len(set(self.field_aliases)) != len(FIELDS) or len(self.field_aliases) != len(FIELDS):
            raise ValueError("field aliases must be unique")
        if len(set(self.digit_aliases)) != 10 or len(self.digit_aliases) != 10:
            raise ValueError("digit aliases must be a ten-symbol permutation")
        if len(set(self.operation_aliases)) != len(OPERATIONS) or len(self.operation_aliases) != len(OPERATIONS):
            raise ValueError("operation aliases must be unique")
        if any(not alias.isalpha() or alias.lower() != alias for alias in self.aliases):
            raise ValueError("aliases must be lowercase alphabetic tokens")

    @property
    def aliases(self):
        return self.field_aliases + self.digit_aliases + self.operation_aliases

    @property
    def field_to_alias(self):
        return dict(zip(FIELDS, self.field_aliases))

    @property
    def alias_to_field(self):
        return dict(zip(self.field_aliases, FIELDS))

    @property
    def digit_to_alias(self):
        return {str(index): alias for index, alias in enumerate(self.digit_aliases)}

    @property
    def alias_to_digit(self):
        return {alias: str(index) for index, alias in enumerate(self.digit_aliases)}

    @property
    def operation_to_alias(self):
        return dict(zip(OPERATIONS, self.operation_aliases))

    @property
    def alias_to_operation(self):
        return dict(zip(self.operation_aliases, OPERATIONS))


def make_codebook(seed: int | str, channel: str) -> Codebook:
    """Create a deterministic, episode-local codebook without global templates."""
    if channel not in {"A", "B"}:
        raise ValueError("channel must be A or B")
    rng = random.Random("dcr-v1:{}:{}".format(seed, channel))
    prefix = channel.lower()
    order = list(FIELDS)
    rng.shuffle(order)
    fields = rng.sample(FIELD_STEMS, len(FIELDS))
    digits = rng.sample(DIGIT_STEMS, 10)
    operations = rng.sample(OP_STEMS, len(OPERATIONS))
    return Codebook(
        channel=channel,
        field_order=tuple(order),
        field_aliases=tuple(prefix + stem for stem in fields),
        digit_aliases=tuple(prefix + stem for stem in digits),
        operation_aliases=tuple(prefix + stem for stem in operations),
    )


def codebook_prompt(book: Codebook) -> str:
    """Render the static key retained alongside every source-dropped state."""
    fields = ", ".join("{}={}".format(field, book.field_to_alias[field]) for field in FIELDS)
    operations = ", ".join("{}={}".format(op, book.operation_to_alias[op]) for op in OPERATIONS)
    digits = ", ".join("{}={}".format(index, book.digit_to_alias[str(index)]) for index in range(10))
    order = ",".join(book.field_to_alias[field] for field in book.field_order)
    return "DCR {} key. fields [{}]. order [{}]. ops [{}]. digits [{}].".format(
        book.channel, fields, order, operations, digits,
    )


def _encode_decimal(value: object, book: Codebook) -> str:
    text = str(value)
    if not text or not text.isdigit():
        raise ValueError("state decimal fields must be unsigned decimal strings")
    return ".".join(book.digit_to_alias[digit] for digit in text)


def _decode_decimal(value: str, book: Codebook) -> str:
    tokens = str(value).split(".")
    if not tokens or any(not token or token not in book.alias_to_digit for token in tokens):
        raise ValueError("invalid coded decimal")
    return "".join(book.alias_to_digit[token] for token in tokens)


def encode_state(state: Mapping[str, object], book: Codebook) -> str:
    """Serialize a verified semantic state in one non-copying code channel."""
    normalized = {
        "op": str(state["op"]), "w": int(state["w"]), "p": int(state["p"]), "c": int(state["c"]),
        "a": str(state["a"]), "b": str(state["b"]), "r": str(state["r"]), "z": int(state["z"]),
    }
    canonical_state(normalized)
    fields = []
    for field in book.field_order:
        alias = book.field_to_alias[field]
        if field == "op":
            value = book.operation_to_alias[normalized[field]]
        else:
            value = _encode_decimal(normalized[field], book)
        fields.append("{}={}".format(alias, value))
    return "dcr:{}|{}".format(book.channel, ";".join(fields))


def parse_state(text: object, book: Codebook):
    """Extract exactly one fully valid state encoded in ``book`` or return None."""
    candidates = [line.strip() for line in str(text).splitlines() if line.strip().startswith("dcr:")]
    if len(candidates) != 1:
        return None
    prefix = "dcr:{}|".format(book.channel)
    line = candidates[0]
    if not line.startswith(prefix):
        return None
    parts = line[len(prefix):].split(";")
    expected_aliases = [book.field_to_alias[field] for field in book.field_order]
    if len(parts) != len(expected_aliases):
        return None
    decoded = {}
    try:
        for part, alias in zip(parts, expected_aliases):
            key, value = part.split("=", 1)
            if key != alias:
                return None
            field = book.alias_to_field[key]
            decoded[field] = book.alias_to_operation[value] if field == "op" else _decode_decimal(value, book)
        state = {
            "op": decoded["op"], "w": int(decoded["w"]), "p": int(decoded["p"]),
            "c": int(decoded["c"]), "a": decoded["a"], "b": decoded["b"], "r": decoded["r"],
            "z": int(decoded["z"]),
        }
        canonical_state(state)
        return state
    except (KeyError, TypeError, ValueError):
        return None


def invert_microstep(next_state: Mapping[str, object]):
    """Solver-only inverse for data construction and post-hoc scoring.

    The previous carry is the sole ambiguity.  Enumerating its two legal values
    and replaying the semantic forward rule makes inverse targets exact without
    embedding an inverse arithmetic shortcut in the runtime controller.
    """
    current = {
        "op": str(next_state["op"]), "w": int(next_state["w"]), "p": int(next_state["p"]),
        "c": int(next_state["c"]), "a": str(next_state["a"]), "b": str(next_state["b"]),
        "r": str(next_state["r"]), "z": int(next_state["z"]),
    }
    canonical_state(current)
    if current["p"] <= 0:
        raise ValueError("initial state has no predecessor")
    position = current["p"] - 1
    tape = list(current["r"])
    tape[position] = "0"
    for prior_carry in (0, 1):
        previous = dict(current)
        previous.update({"p": position, "c": prior_carry, "r": "".join(tape), "z": 0})
        try:
            canonical_state(previous)
        except ValueError:
            continue
        if apply_microstep(previous) == current:
            return previous
    raise ValueError("state has no unique valid predecessor")


def forward_prompt(book: Codebook, state_line: str) -> str:
    return (
        "Advance one local decimal machine transition in DCR {}. {}\n"
        "State: {}\nReturn exactly one dcr:{} state line.\nAnswer:"
    ).format(book.channel, codebook_prompt(book), state_line, book.channel)


def transcode_prompt(source: Codebook, target: Codebook, state_line: str) -> str:
    return (
        "Rewrite the identical machine state from DCR {} to DCR {}. {} {}\n"
        "Input: {}\nReturn exactly one dcr:{} state line.\nAnswer:"
    ).format(source.channel, target.channel, codebook_prompt(source), codebook_prompt(target), state_line, target.channel)


def reverse_prompt(book: Codebook, state_line: str) -> str:
    return (
        "Recover exactly the immediately preceding DCR {} machine state. {}\n"
        "Current state: {}\nReturn exactly one dcr:{} state line.\nAnswer:"
    ).format(book.channel, codebook_prompt(book), state_line, book.channel)
