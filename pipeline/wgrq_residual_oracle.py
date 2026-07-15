#!/usr/bin/env python3
"""Exact Stage-A oracle for the delayed-witness edge-parity ring.

States are packed integers whose bit ``i`` is physical coordinate ``x_i``.
Words are applied from left to right.  The implementation deliberately keeps
the physical simulator separate from the canonical residual code so callers
cannot accidentally treat a physical state ID as a learned target.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


ROTATE = "R"
FLIP = "F"
EVENT_ALPHABET = (ROTATE, FLIP)
EVENT_CODE_BITS = {ROTATE: (1, 0), FLIP: (0, 1)}
SYMBOLIC_GATE_SCALES = (3, 6)


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("DWEPR requires integer n >= 3")
    return n


def state_mask(n: int) -> int:
    return (1 << _validate_n(n)) - 1


def _validate_state(state: int, n: int) -> int:
    _validate_n(n)
    if isinstance(state, bool) or not isinstance(state, int):
        raise TypeError("state must be a packed integer")
    if state < 0 or state > state_mask(n):
        raise ValueError("state is outside GF(2)^n")
    return state


def rotate(state: int, n: int) -> int:
    """Apply ``(R x)_i = x_(i+1 mod n)``."""
    state = _validate_state(state, n)
    return (state >> 1) | ((state & 1) << (n - 1))


def inverse_rotate(state: int, n: int) -> int:
    """Apply the inverse physical rotation."""
    state = _validate_state(state, n)
    return ((state << 1) & state_mask(n)) | (state >> (n - 1))


def flip(state: int, n: int) -> int:
    """Toggle physical coordinate zero."""
    return _validate_state(state, n) ^ 1


def read(state: int, n: int) -> int:
    """Return the sole ordinary oracle answer, ``x_0 xor x_1``."""
    state = _validate_state(state, n)
    return (state & 1) ^ ((state >> 1) & 1)


observe = read


def apply_event(state: int, event: str, n: int) -> int:
    if event == ROTATE:
        return rotate(state, n)
    if event == FLIP:
        return flip(state, n)
    raise ValueError("unknown DWEPR event: {!r}".format(event))


def apply_word(state: int, word: Iterable[str], n: int) -> int:
    state = _validate_state(state, n)
    for event in word:
        state = apply_event(state, event, n)
    return state


def rotation_word(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("rotation count must be a nonnegative integer")
    return (ROTATE,) * count


def event_counts(word: Iterable[str]) -> dict[str, int]:
    counts = Counter(word)
    unknown = set(counts) - set(EVENT_ALPHABET)
    if unknown:
        raise ValueError("unknown DWEPR events: {}".format(sorted(unknown)))
    return {ROTATE: counts[ROTATE], FLIP: counts[FLIP]}


def edge_vector(state: int, n: int) -> tuple[int, ...]:
    state = _validate_state(state, n)
    return tuple(
        ((state >> i) & 1) ^ ((state >> ((i + 1) % n)) & 1)
        for i in range(n)
    )


def serialize_edge_vector(edges: Sequence[int]) -> str:
    """Serialize an even-parity edge vector using its first ``n-1`` bits."""
    if len(edges) < 3 or any(bit not in (0, 1) for bit in edges):
        raise ValueError("edge vector must contain at least three binary bits")
    if sum(edges) % 2:
        raise ValueError("a ring edge vector must have even parity")
    return "".join(str(int(bit)) for bit in edges[:-1])


def deserialize_edge_vector(code: str | bytes) -> tuple[int, ...]:
    if isinstance(code, bytes):
        code = code.decode("ascii")
    if not isinstance(code, str) or len(code) < 2 or set(code) - {"0", "1"}:
        raise ValueError("canonical edge code must be an (n-1)-bit string")
    prefix = tuple(int(bit) for bit in code)
    return prefix + (sum(prefix) % 2,)


def canonical_code(state: int, n: int) -> str:
    """Return the canonical serialized residual class code."""
    return serialize_edge_vector(edge_vector(state, n))


residual_code = canonical_code


def canonical_representative(code: str | bytes) -> int:
    """Return the unique representative with physical coordinate zero set."""
    edges = deserialize_edge_vector(code)
    state = 0
    coordinate = 0
    for i, edge in enumerate(edges[:-1]):
        coordinate ^= edge
        state |= coordinate << (i + 1)
    return state


def residual_equivalent(left: int, right: int, n: int) -> bool:
    left = _validate_state(left, n)
    right = _validate_state(right, n)
    return right == left or right == (left ^ state_mask(n))


equivalent = residual_equivalent


def answer_after_rotations(state: int, rotations: int, n: int) -> int:
    return read(apply_word(state, rotation_word(rotations), n), n)


def determining_signature(state: int, n: int) -> tuple[int, ...]:
    """Return answers for the determining continuations R^0 through R^(n-2)."""
    _validate_state(state, n)
    return tuple(answer_after_rotations(state, k, n) for k in range(n - 1))


def shortest_witness_depth(left: int, right: int, n: int) -> int | None:
    """Return the least distinguishing rotation count, or ``None`` if merged."""
    left_edges = edge_vector(left, n)
    right_edges = edge_vector(right, n)
    for index, (left_bit, right_bit) in enumerate(zip(left_edges, right_edges)):
        if left_bit != right_bit:
            if index == n - 1:
                raise AssertionError("a nonzero even-parity difference cannot start at n-1")
            return index
    return None


def shortest_witness_word(left: int, right: int, n: int) -> tuple[str, ...] | None:
    depth = shortest_witness_depth(left, right, n)
    return None if depth is None else rotation_word(depth)


def quotient_transition(code: str | bytes, event: str) -> str:
    edges = list(deserialize_edge_vector(code))
    if event == ROTATE:
        edges = edges[1:] + edges[:1]
    elif event == FLIP:
        edges[0] ^= 1
        edges[-1] ^= 1
    else:
        raise ValueError("unknown DWEPR event: {!r}".format(event))
    return serialize_edge_vector(edges)


def quotient_read(code: str | bytes) -> int:
    return deserialize_edge_vector(code)[0]


def canonical_access_word(state: int, n: int) -> tuple[str, ...]:
    """Construct a direct zero-to-state word with n rotations.

    Cycle ``i`` optionally flips coordinate zero and then rotates.  After all
    ``n`` cycles, that flip lands at final physical coordinate ``i``.
    """
    state = _validate_state(state, n)
    word: list[str] = []
    for i in range(n):
        if (state >> i) & 1:
            word.append(FLIP)
        word.append(ROTATE)
    result = tuple(word)
    if apply_word(0, result, n) != state:
        raise AssertionError("canonical access construction is inconsistent")
    return result


def cancellation_gadgets(n: int) -> dict[str, tuple[str, ...]]:
    """Return the preregistered cancellation words in frozen order."""
    _validate_n(n)
    gadgets = {
        "ff_rn_identity": (FLIP, FLIP) + rotation_word(n),
        "frf_rn_minus_1_nonidentity": (FLIP, ROTATE, FLIP) + rotation_word(n - 1),
        "fr_order": (FLIP, ROTATE),
        "rf_order": (ROTATE, FLIP),
        "global_complement": (FLIP, ROTATE) * n,
    }
    if n % 2 == 0:
        gadgets["equal_count_identity"] = (FLIP, FLIP) * (n // 2) + rotation_word(n)
    return gadgets


def cancellation_gadget_report(n: int) -> dict[str, object]:
    """Exhaustively certify the cancellation controls for one scale."""
    gadgets = cancellation_gadgets(n)
    mask = state_mask(n)
    identity = gadgets["ff_rn_identity"]
    nonidentity = gadgets["frf_rn_minus_1_nonidentity"]
    fr = gadgets["fr_order"]
    rf = gadgets["rf_order"]
    global_complement = gadgets["global_complement"]
    identity_ok = all(apply_word(state, identity, n) == state for state in range(mask + 1))
    nonidentity_ok = all(apply_word(state, nonidentity, n) != state for state in range(mask + 1))
    order_ok = all(apply_word(state, fr, n) != apply_word(state, rf, n) for state in range(mask + 1))
    complement_ok = all(
        apply_word(state, global_complement, n) == (state ^ mask)
        and canonical_code(apply_word(state, global_complement, n), n) == canonical_code(state, n)
        for state in range(mask + 1)
    )
    equal_count_ok: bool | None = None
    if "equal_count_identity" in gadgets:
        equal_count_ok = all(
            apply_word(state, gadgets["equal_count_identity"], n) == state
            for state in range(mask + 1)
        )
    same_counts = event_counts(identity) == event_counts(nonidentity)
    global_equal_counts = (
        event_counts(global_complement) == event_counts(gadgets["equal_count_identity"])
        if "equal_count_identity" in gadgets
        else None
    )
    passed = all((identity_ok, nonidentity_ok, order_ok, complement_ok, same_counts))
    if equal_count_ok is not None:
        passed = passed and equal_count_ok and bool(global_equal_counts)
    return {
        "n": n,
        "passed": bool(passed),
        "words": {name: "".join(word) for name, word in gadgets.items()},
        "event_counts": {name: event_counts(word) for name, word in gadgets.items()},
        "ff_rn_identity": identity_ok,
        "same_count_nonidentity": nonidentity_ok and same_counts,
        "fr_ne_rf": order_ok,
        "global_complement_observationally_null": complement_ok,
        "equal_count_identity": equal_count_ok,
        "global_complement_equal_counts": global_equal_counts,
    }


def _expected_rotate(state: int, n: int) -> int:
    return sum(((state >> ((i + 1) % n)) & 1) << i for i in range(n))


def exhaustive_symbolic_check(n: int) -> dict[str, object]:
    """Run every frozen Stage-A symbolic gate at one scale."""
    _validate_n(n)
    states = 1 << n
    minimum_check_count = (n - 1) * (1 << (2 * n)) + 3 * states
    checks = 0
    core_checks = 0

    def check(condition: bool, message: str, *, core: bool = False) -> None:
        nonlocal checks, core_checks
        checks += 1
        if core:
            core_checks += 1
        if not condition:
            raise AssertionError("DWEPR_{} symbolic gate failed: {}".format(n, message))

    classes: dict[str, list[int]] = defaultdict(list)
    for state in range(states):
        rotated = rotate(state, n)
        check(
            rotated == _expected_rotate(state, n)
            and inverse_rotate(rotated, n) == state
            and rotate(inverse_rotate(state, n), n) == state,
            "physical rotation or inverse",
            core=True,
        )
        check(
            flip(state, n) == (state ^ 1) and flip(flip(state, n), n) == state,
            "physical flip or inverse",
            core=True,
        )
        edges = edge_vector(state, n)
        code = canonical_code(state, n)
        check(
            sum(edges) % 2 == 0
            and read(state, n) == edges[0]
            and deserialize_edge_vector(code) == edges,
            "edge/read/serialization mechanics",
            core=True,
        )
        check(len(code) == n - 1, "fixed-width canonical serialization")
        check(canonical_code(canonical_representative(code), n) == code, "canonical round trip")
        classes[code].append(state)

    depth_counts: Counter[int] = Counter()
    for left in range(states):
        left_edges = edge_vector(left, n)
        left_signature = determining_signature(left, n)
        for right in range(states):
            right_edges = edge_vector(right, n)
            difference = tuple(a ^ b for a, b in zip(left_edges, right_edges))
            for rotations in range(n - 1):
                check(
                    answer_after_rotations(left, rotations, n)
                    ^ answer_after_rotations(right, rotations, n)
                    == difference[rotations],
                    "residual theorem",
                    core=True,
                )
            equivalent_by_answers = left_signature == determining_signature(right, n)
            equivalent_by_physics = residual_equivalent(left, right, n)
            check(equivalent_by_answers == equivalent_by_physics, "determining continuations")
            check(
                (canonical_code(left, n) == canonical_code(right, n)) == equivalent_by_physics,
                "canonical collision or over-splitting",
            )
            witness = shortest_witness_depth(left, right, n)
            if equivalent_by_physics:
                check(witness is None, "equivalent states received a witness")
            else:
                check(witness is not None and 0 <= witness <= n - 2, "witness depth bound")
                assert witness is not None
                check(
                    all(
                        answer_after_rotations(left, earlier, n)
                        == answer_after_rotations(right, earlier, n)
                        for earlier in range(witness)
                    ),
                    "witness was not shortest",
                )
                check(
                    answer_after_rotations(left, witness, n)
                    != answer_after_rotations(right, witness, n),
                    "witness did not distinguish",
                )
                depth_counts[witness] += 1

    check(len(classes) == 1 << (n - 1), "residual class count")
    for code, members in classes.items():
        check(len(members) == 2, "residual class size")
        check(members[0] ^ members[1] == state_mask(n), "class is not a complement pair")
        for event in EVENT_ALPHABET:
            next_codes = {canonical_code(apply_event(state, event, n), n) for state in members}
            check(len(next_codes) == 1, "representative-dependent quotient transition")
            check(next(iter(next_codes)) == quotient_transition(code, event), "quotient transition mismatch")
        check(quotient_read(code) == read(members[0], n), "quotient read mismatch")

    check(set(depth_counts) == set(range(n - 1)), "missing shortest-witness depth")
    check(max(depth_counts) == n - 2, "n-2 witness bound is not tight")
    cancellation = cancellation_gadget_report(n)
    check(bool(cancellation["passed"]), "cancellation controls")
    check(core_checks == minimum_check_count, "minimum check ledger")
    check(checks >= minimum_check_count, "total check ledger")
    return {
        "schema": "dwepr_symbolic_gate_v1",
        "n": n,
        "passed": True,
        "physical_states": states,
        "residual_classes": len(classes),
        "class_size": 2,
        "determining_rotations": list(range(n - 1)),
        "shortest_witness_depth_counts": {
            str(depth): depth_counts[depth] for depth in range(n - 1)
        },
        "maximum_shortest_witness_depth": max(depth_counts),
        "minimum_check_count": minimum_check_count,
        "core_check_count": core_checks,
        "check_count": checks,
        "cancellation_controls": cancellation,
    }


def run_stage_a_symbolic_gates() -> dict[str, object]:
    scales = [exhaustive_symbolic_check(n) for n in SYMBOLIC_GATE_SCALES]
    return {
        "schema": "dwepr_stage_a_symbolic_gates_v1",
        "passed": all(bool(scale["passed"]) for scale in scales),
        "scales": scales,
        "total_minimum_check_count": sum(int(scale["minimum_check_count"]) for scale in scales),
        "total_core_check_count": sum(int(scale["core_check_count"]) for scale in scales),
        "total_check_count": sum(int(scale["check_count"]) for scale in scales),
    }


@dataclass(frozen=True, slots=True)
class DWEPR:
    """Small immutable facade over the exact packed-state functions."""

    n: int

    def __post_init__(self) -> None:
        _validate_n(self.n)

    @property
    def initial_state(self) -> int:
        return 0

    @property
    def mask(self) -> int:
        return state_mask(self.n)

    def rotate(self, state: int) -> int:
        return rotate(state, self.n)

    def inverse_rotate(self, state: int) -> int:
        return inverse_rotate(state, self.n)

    def flip(self, state: int) -> int:
        return flip(state, self.n)

    def read(self, state: int) -> int:
        return read(state, self.n)

    def apply(self, state: int, word: Iterable[str]) -> int:
        return apply_word(state, word, self.n)

    def edges(self, state: int) -> tuple[int, ...]:
        return edge_vector(state, self.n)

    def code(self, state: int) -> str:
        return canonical_code(state, self.n)

    def equivalent(self, left: int, right: int) -> bool:
        return residual_equivalent(left, right, self.n)

    def shortest_witness_depth(self, left: int, right: int) -> int | None:
        return shortest_witness_depth(left, right, self.n)
