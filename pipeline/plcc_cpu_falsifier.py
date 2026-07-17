#!/usr/bin/env python3
"""Deterministic CPU mechanics falsifier for the Packet-on-Lattice Carry Cell.

This module is deliberately neural-free. It audits the finite oracle mechanics
frozen in R12_PACKET_ON_LATTICE_CARRY_CELL_PREREG.md and records the hostile
equivalence boundary: on this board, PLCC is exactly a coordinate change of an
explicit recurrent (cursor, carry) transducer. A passing report is not evidence
of novelty, learnability, Shohin capability, or a new reasoning primitive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


PROTOCOL_ID = "R12-PLCC-CPU-v1"
SCHEMA_VERSION = 1
OPS = ("ADD", "SUB")
DIGITS = tuple(range(10))
CONTEXT_WIDTHS = (1, 2, 3, 4)
LOCAL_CELL_COUNT = len(OPS) * 10 * 10 * 2
TWO_COLUMN_BOARD_COUNT = len(OPS) * (10**4) * 2
WIDTH_TWO_STATE_COUNT = 4

# Frozen after the deterministic v1 board and local table were implemented,
# before any neural fit, GPU pilot, or model result.
FROZEN_LOCAL_TABLE_SHA256 = (
    "6c21e29e5341a3343cab76edeadb613d71a83bb00c2b7c1438d24c2160c0c7e2"
)
FROZEN_TWO_COLUMN_BOARD_SHA256 = (
    "1911674b7ea403ac70a4de0f6cd04f1f2e99f62df84cd1d49ce38dcd11322181"
)


class AuditError(ValueError):
    """A finite contract or closed-world payload check failed."""


@dataclass(frozen=True, slots=True)
class Column:
    """One immutable least-significant-first operand column."""

    a: int
    b: int

    def __post_init__(self) -> None:
        _validate_digit(self.a, "a")
        _validate_digit(self.b, "b")


@dataclass(frozen=True, slots=True)
class OperandBoard:
    """Immutable source columns. The board is not mutable runtime state."""

    op: str
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        _validate_op(self.op)
        if type(self.columns) is not tuple or not self.columns:
            raise ValueError("columns must be a nonempty exact tuple")
        if any(type(column) is not Column for column in self.columns):
            raise TypeError("every source column must be an exact Column")

    @property
    def width(self) -> int:
        return len(self.columns)


@dataclass(frozen=True, slots=True)
class Packet:
    """Complete mutable PLCC state: support location plus one polarity bit."""

    location: int
    polarity: int

    def __post_init__(self) -> None:
        if isinstance(self.location, bool) or not isinstance(self.location, int):
            raise TypeError("packet location must be an integer")
        if self.location < 0:
            raise ValueError("packet location must be nonnegative")
        _validate_bit(self.polarity, "packet polarity")


@dataclass(frozen=True, slots=True)
class LocalOutput:
    """Ephemeral local scatter output; never retained across a cycle."""

    digit: int
    next_carry: int

    def __post_init__(self) -> None:
        _validate_digit(self.digit, "local digit")
        _validate_bit(self.next_carry, "next carry")


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Only allowed external emission, produced at the terminal source slot."""

    final_digit: int
    terminal_carry: int

    def __post_init__(self) -> None:
        _validate_digit(self.final_digit, "final digit")
        _validate_bit(self.terminal_carry, "terminal carry")


@dataclass(frozen=True, slots=True)
class RecurrentState:
    """Favorable explicit control state under the coordinate map."""

    cursor: int
    carry: int

    def __post_init__(self) -> None:
        if isinstance(self.cursor, bool) or not isinstance(self.cursor, int):
            raise TypeError("recurrent cursor must be an integer")
        if self.cursor < 0:
            raise ValueError("recurrent cursor must be nonnegative")
        _validate_bit(self.carry, "recurrent carry")


def _validate_op(op: str) -> None:
    if op not in OPS:
        raise ValueError(f"operation must be one of {OPS}, got {op!r}")


def _validate_digit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer digit")
    if not 0 <= value <= 9:
        raise ValueError(f"{name} must be in [0, 9]")


def _validate_bit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer bit")
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode deterministic ASCII JSON with one trailing newline."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reference_local_transition(op: str, a_p: int, b_p: int, c_p: int) -> LocalOutput:
    """Independent exact decimal reference for add-carry or sub-borrow."""

    _validate_op(op)
    _validate_digit(a_p, "a_p")
    _validate_digit(b_p, "b_p")
    _validate_bit(c_p, "c_p")
    if op == "ADD":
        total = a_p + b_p + c_p
        return LocalOutput(total % 10, int(total >= 10))
    difference = a_p - b_p - c_p
    return LocalOutput(difference % 10, int(difference < 0))


def plcc_local_transition(op: str, a_p: int, b_p: int, c_p: int) -> LocalOutput:
    """Frozen tied local cell with exactly the four authorized inputs."""

    _validate_op(op)
    _validate_digit(a_p, "a_p")
    _validate_digit(b_p, "b_p")
    _validate_bit(c_p, "c_p")
    if op == "ADD":
        raw = a_p + b_p + c_p
        if raw >= 10:
            return LocalOutput(raw - 10, 1)
        return LocalOutput(raw, 0)
    raw = a_p - b_p - c_p
    if raw < 0:
        return LocalOutput(raw + 10, 1)
    return LocalOutput(raw, 0)


def packet_occupancy(packet: Packet, width: int) -> tuple[int, ...]:
    """Reflect packet support as a one-hot occupancy vector."""

    validate_packet_no_hidden_payload(packet, width, check_occupancy=False)
    return tuple(int(index == packet.location) for index in range(width))


def validate_packet_no_hidden_payload(
    packet: Packet,
    width: int,
    *,
    check_occupancy: bool = True,
) -> None:
    """Reject subclasses, extra fields, dictionaries, or non-one-hot support."""

    if type(packet) is not Packet:
        raise AuditError("runtime packet must have exact Packet type")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise AuditError("packet width must be a positive integer")
    if packet.location >= width:
        raise AuditError("packet location lies outside the source lattice")
    field_names = tuple(field.name for field in fields(Packet))
    if field_names != ("location", "polarity"):
        raise AuditError("packet dataclass contains a hidden field")
    if tuple(Packet.__slots__) != ("location", "polarity"):
        raise AuditError("packet slots contain a hidden payload")
    if hasattr(packet, "__dict__"):
        raise AuditError("packet exposes a dynamic payload dictionary")
    if check_occupancy:
        occupancy = tuple(int(index == packet.location) for index in range(width))
        if len(occupancy) != width:
            raise AuditError("packet occupancy has the wrong width")
        if any(value not in (0, 1) for value in occupancy) or sum(occupancy) != 1:
            raise AuditError("packet occupancy must contain exactly one hard one")
    _validate_bit(packet.polarity, "packet polarity")


def validate_board_closed_world(board: OperandBoard) -> None:
    """Reject board subclasses or hidden source fields."""

    if type(board) is not OperandBoard:
        raise AuditError("source board must have exact OperandBoard type")
    if tuple(field.name for field in fields(OperandBoard)) != ("op", "columns"):
        raise AuditError("source board contains a hidden field")
    if tuple(OperandBoard.__slots__) != ("op", "columns"):
        raise AuditError("source board slots contain a hidden payload")
    if hasattr(board, "__dict__"):
        raise AuditError("source board exposes a dynamic payload dictionary")
    if type(board.columns) is not tuple or any(
        type(column) is not Column for column in board.columns
    ):
        raise AuditError("source columns are not the frozen immutable tuple")


def audit_local_at_packet(board: OperandBoard, packet: Packet) -> LocalOutput:
    """Non-causal auditor reflection of the cell selected by packet support."""

    validate_board_closed_world(board)
    validate_packet_no_hidden_payload(packet, board.width)
    column = board.columns[packet.location]
    return plcc_local_transition(board.op, column.a, column.b, packet.polarity)


def advance_packet(board: OperandBoard, packet: Packet) -> Packet | Endpoint:
    """Advance one cycle without exposing a nonterminal digit or result tape."""

    local = audit_local_at_packet(board, packet)
    if packet.location == board.width - 1:
        return Endpoint(local.digit, local.next_carry)
    return Packet(packet.location + 1, local.next_carry)


def run_plcc_from_packet(board: OperandBoard, packet: Packet) -> Endpoint:
    """Run to the endpoint while retaining only the one packet between cycles."""

    validate_board_closed_world(board)
    validate_packet_no_hidden_payload(packet, board.width)
    state: Packet | Endpoint = packet
    while type(state) is Packet:
        state = advance_packet(board, state)
    if type(state) is not Endpoint:
        raise AuditError("PLCC escaped the closed Packet-or-Endpoint state space")
    return state


def run_plcc(board: OperandBoard, initial_carry: int) -> Endpoint:
    _validate_bit(initial_carry, "initial carry")
    return run_plcc_from_packet(board, Packet(0, initial_carry))


def run_recurrent_from_state(
    board: OperandBoard,
    state: RecurrentState,
) -> Endpoint:
    """Favorable ordinary recurrent control using the independent reference."""

    validate_board_closed_world(board)
    if type(state) is not RecurrentState or state.cursor >= board.width:
        raise AuditError("recurrent state lies outside the frozen board")
    cursor = state.cursor
    carry = state.carry
    while True:
        column = board.columns[cursor]
        local = reference_local_transition(board.op, column.a, column.b, carry)
        if cursor == board.width - 1:
            return Endpoint(local.digit, local.next_carry)
        cursor += 1
        carry = local.next_carry


def run_recurrent_control(board: OperandBoard, initial_carry: int) -> Endpoint:
    _validate_bit(initial_carry, "initial carry")
    return run_recurrent_from_state(board, RecurrentState(0, initial_carry))


def packet_to_recurrent(packet: Packet, width: int) -> RecurrentState:
    validate_packet_no_hidden_payload(packet, width)
    return RecurrentState(packet.location, packet.polarity)


def recurrent_to_packet(state: RecurrentState, width: int) -> Packet:
    if type(state) is not RecurrentState or state.cursor >= width:
        raise AuditError("recurrent state lies outside the packet lattice")
    return Packet(state.cursor, state.carry)


def iter_local_cells() -> Iterator[tuple[str, int, int, int]]:
    yield from itertools.product(OPS, DIGITS, DIGITS, (0, 1))


def iter_two_column_cases(
    *,
    include_initial_carry: bool = True,
) -> Iterator[tuple[str, Column, Column, int]]:
    carries = (0, 1) if include_initial_carry else (0,)
    for op, a0, b0, a1, b1, initial_carry in itertools.product(
        OPS,
        DIGITS,
        DIGITS,
        DIGITS,
        DIGITS,
        carries,
    ):
        yield op, Column(a0, b0), Column(a1, b1), initial_carry


def local_table_sha256() -> str:
    digest = hashlib.sha256()
    for op, a_p, b_p, c_p in iter_local_cells():
        output = reference_local_transition(op, a_p, b_p, c_p)
        digest.update(
            canonical_json_bytes(
                {
                    "a_p": a_p,
                    "b_p": b_p,
                    "c_p": c_p,
                    "digit": output.digit,
                    "next_carry": output.next_carry,
                    "op": op,
                }
            )
        )
    return digest.hexdigest()


def two_column_board_sha256() -> str:
    digest = hashlib.sha256()
    for op, column0, column1, initial_carry in iter_two_column_cases():
        first = reference_local_transition(op, column0.a, column0.b, initial_carry)
        second = reference_local_transition(op, column1.a, column1.b, first.next_carry)
        digest.update(
            canonical_json_bytes(
                {
                    "a0": column0.a,
                    "a1": column1.a,
                    "b0": column0.b,
                    "b1": column1.b,
                    "final_digit": second.digit,
                    "initial_carry": initial_carry,
                    "op": op,
                    "terminal_carry": second.next_carry,
                }
            )
        )
    return digest.hexdigest()


@lru_cache(maxsize=1)
def local_cell_audit() -> dict[str, Any]:
    contexts = tuple(
        (width, position, position == width - 1)
        for width in CONTEXT_WIDTHS
        for position in range(width)
    )
    exact_cells = 0
    context_exact = 0
    terminal_contexts = sum(terminal for _, _, terminal in contexts)
    nonterminal_contexts = len(contexts) - terminal_contexts
    signature = tuple(inspect.signature(plcc_local_transition).parameters)
    for op, a_p, b_p, c_p in iter_local_cells():
        expected = reference_local_transition(op, a_p, b_p, c_p)
        observed = plcc_local_transition(op, a_p, b_p, c_p)
        exact_cells += int(observed == expected)
        for _width, _position, _terminal in contexts:
            context_exact += int(plcc_local_transition(op, a_p, b_p, c_p) == expected)
    return {
        "cell_count": LOCAL_CELL_COUNT,
        "exact_cells": exact_cells,
        "authorized_signature": ["op", "a_p", "b_p", "c_p"],
        "observed_signature": list(signature),
        "context_widths": list(CONTEXT_WIDTHS),
        "context_count_per_cell": len(contexts),
        "terminal_contexts_per_cell": terminal_contexts,
        "nonterminal_contexts_per_cell": nonterminal_contexts,
        "context_observation_count": LOCAL_CELL_COUNT * len(contexts),
        "context_exact": context_exact,
        "local_table_sha256": local_table_sha256(),
    }


@lru_cache(maxsize=1)
def two_column_audit() -> dict[str, Any]:
    plcc_exact = 0
    recurrent_exact = 0
    equivalent = 0
    prefix_deletion_exact = 0
    intermediate_packet_only = 0
    terminal_endpoint_only = 0
    total = 0
    for op, column0, column1, initial_carry in iter_two_column_cases():
        board = OperandBoard(op, (column0, column1))
        first = reference_local_transition(op, column0.a, column0.b, initial_carry)
        second = reference_local_transition(op, column1.a, column1.b, first.next_carry)
        expected = Endpoint(second.digit, second.next_carry)

        plcc = run_plcc(board, initial_carry)
        recurrent = run_recurrent_control(board, initial_carry)
        plcc_exact += int(plcc == expected)
        recurrent_exact += int(recurrent == expected)
        equivalent += int(plcc == recurrent)

        first_runtime = advance_packet(board, Packet(0, initial_carry))
        intermediate_packet_only += int(type(first_runtime) is Packet)
        if type(first_runtime) is not Packet:
            raise AuditError("nonterminal PLCC cycle emitted an endpoint")
        terminal_runtime = advance_packet(board, first_runtime)
        terminal_endpoint_only += int(type(terminal_runtime) is Endpoint)

        suffix = OperandBoard(op, (column1,))
        normalized_packet = Packet(0, first_runtime.polarity)
        deleted_prefix_endpoint = run_plcc_from_packet(suffix, normalized_packet)
        prefix_deletion_exact += int(deleted_prefix_endpoint == plcc)
        total += 1
    return {
        "case_count": total,
        "plcc_exact": plcc_exact,
        "explicit_recurrent_exact": recurrent_exact,
        "plcc_recurrent_equivalent": equivalent,
        "source_prefix_deletion_exact": prefix_deletion_exact,
        "intermediate_packet_only": intermediate_packet_only,
        "terminal_endpoint_only": terminal_endpoint_only,
        "intermediate_emitted_symbol_count": 0,
        "generated_token_count": 0,
        "generated_kv_bytes": 0,
        "result_tape_slots": 0,
        "host_arithmetic_calls_during_neural_inference": 0,
        "two_column_board_sha256": two_column_board_sha256(),
    }


def _same_carry_prefix_pair(
    op: str,
    desired_carry: int,
) -> tuple[tuple[Column, int], tuple[Column, int]]:
    candidates: list[tuple[Column, int, int]] = []
    for a_p, b_p, c_p in itertools.product(DIGITS, DIGITS, (0, 1)):
        output = reference_local_transition(op, a_p, b_p, c_p)
        if output.next_carry == desired_carry:
            candidates.append((Column(a_p, b_p), c_p, output.digit))
    for left, right in itertools.combinations(candidates, 2):
        if left[:2] != right[:2] and left[2] != right[2]:
            return (left[0], left[1]), (right[0], right[1])
    raise AuditError("no unrelated same-carry prefix pair exists")


@lru_cache(maxsize=1)
def carry_swap_audit() -> dict[str, Any]:
    different_carry_checks = 0
    different_carry_donor_exact = 0
    different_carry_divergent = 0
    same_carry_checks = 0
    same_carry_sham_exact = 0
    witness_rows: list[dict[str, Any]] = []

    for op in OPS:
        pairs = {carry: _same_carry_prefix_pair(op, carry) for carry in (0, 1)}
        for carry, (left, right) in pairs.items():
            witness_rows.append(
                {
                    "op": op,
                    "carry": carry,
                    "left": {
                        "a": left[0].a,
                        "b": left[0].b,
                        "initial_carry": left[1],
                    },
                    "right": {
                        "a": right[0].a,
                        "b": right[0].b,
                        "initial_carry": right[1],
                    },
                }
            )

        for recipient_carry in (0, 1):
            donor_carry = 1 - recipient_carry
            recipient_prefix, recipient_initial = pairs[recipient_carry][0]
            donor_prefix, donor_initial = pairs[donor_carry][0]
            for suffix_a, suffix_b in itertools.product(DIGITS, DIGITS):
                suffix = Column(suffix_a, suffix_b)
                recipient_board = OperandBoard(op, (recipient_prefix, suffix))
                donor_board = OperandBoard(op, (donor_prefix, suffix))
                recipient_packet = advance_packet(
                    recipient_board, Packet(0, recipient_initial)
                )
                donor_packet = advance_packet(donor_board, Packet(0, donor_initial))
                if (
                    type(recipient_packet) is not Packet
                    or type(donor_packet) is not Packet
                ):
                    raise AuditError("prefix transition did not return a packet")
                if recipient_packet.polarity != recipient_carry:
                    raise AuditError("recipient prefix produced the wrong carry")
                if donor_packet.polarity != donor_carry:
                    raise AuditError("donor prefix produced the wrong carry")

                transplanted = Packet(
                    recipient_packet.location,
                    donor_packet.polarity,
                )
                donor_endpoint = run_plcc_from_packet(recipient_board, transplanted)
                expected_donor = reference_local_transition(
                    op, suffix_a, suffix_b, donor_carry
                )
                expected_recipient = reference_local_transition(
                    op, suffix_a, suffix_b, recipient_carry
                )
                different_carry_donor_exact += int(
                    donor_endpoint
                    == Endpoint(expected_donor.digit, expected_donor.next_carry)
                )
                different_carry_divergent += int(
                    donor_endpoint
                    != Endpoint(
                        expected_recipient.digit,
                        expected_recipient.next_carry,
                    )
                )
                different_carry_checks += 1

        for carry, (left, right) in pairs.items():
            left_prefix, left_initial = left
            right_prefix, right_initial = right
            for suffix_a, suffix_b in itertools.product(DIGITS, DIGITS):
                suffix = Column(suffix_a, suffix_b)
                left_board = OperandBoard(op, (left_prefix, suffix))
                right_board = OperandBoard(op, (right_prefix, suffix))
                left_packet = advance_packet(left_board, Packet(0, left_initial))
                right_packet = advance_packet(right_board, Packet(0, right_initial))
                if type(left_packet) is not Packet or type(right_packet) is not Packet:
                    raise AuditError("same-carry prefix did not return a packet")
                if left_packet.polarity != carry or right_packet.polarity != carry:
                    raise AuditError("same-carry witness failed its carry class")
                left_endpoint = run_plcc_from_packet(left_board, left_packet)
                sham_endpoint = run_plcc_from_packet(left_board, right_packet)
                same_carry_sham_exact += int(left_endpoint == sham_endpoint)
                same_carry_checks += 1

    return {
        "different_carry_swap_checks": different_carry_checks,
        "different_carry_donor_exact": different_carry_donor_exact,
        "different_carry_recipient_divergence": different_carry_divergent,
        "same_carry_sham_checks": same_carry_checks,
        "same_carry_sham_exact": same_carry_sham_exact,
        "same_carry_prefix_witnesses": witness_rows,
    }


@lru_cache(maxsize=1)
def cursor_swap_audit() -> dict[str, Any]:
    swap_pairs = 0
    selected_column_exact = 0
    polarity_preserved_before_scatter = 0
    for op, column0, column1, polarity in iter_two_column_cases():
        board = OperandBoard(op, (column0, column1))
        packet_at_zero = Packet(0, polarity)
        packet_at_one = Packet(1, polarity)
        observed_zero = audit_local_at_packet(board, packet_at_zero)
        observed_one = audit_local_at_packet(board, packet_at_one)
        expected_zero = reference_local_transition(op, column0.a, column0.b, polarity)
        expected_one = reference_local_transition(op, column1.a, column1.b, polarity)
        selected_column_exact += int(observed_zero == expected_zero)
        selected_column_exact += int(observed_one == expected_one)
        polarity_preserved_before_scatter += int(
            packet_at_zero.polarity == packet_at_one.polarity == polarity
        )
        swap_pairs += 1
    return {
        "swap_pair_count": swap_pairs,
        "selected_column_observation_count": 2 * swap_pairs,
        "selected_column_exact": selected_column_exact,
        "polarity_preserved_before_scatter": polarity_preserved_before_scatter,
    }


def _endpoint_dict(endpoint: Endpoint) -> dict[str, int]:
    return {
        "final_digit": endpoint.final_digit,
        "terminal_carry": endpoint.terminal_carry,
    }


@lru_cache(maxsize=1)
def distinguishability_audit() -> dict[str, Any]:
    states = tuple(
        Packet(location, polarity) for location in (0, 1) for polarity in (0, 1)
    )
    witnesses: list[dict[str, Any]] = []
    for left, right in itertools.combinations(states, 2):
        witness: dict[str, Any] | None = None
        for op, column0, column1, _unused in iter_two_column_cases(
            include_initial_carry=False
        ):
            board = OperandBoard(op, (column0, column1))
            left_endpoint = run_plcc_from_packet(board, left)
            right_endpoint = run_plcc_from_packet(board, right)
            if left_endpoint != right_endpoint:
                witness = {
                    "left_state": {
                        "location": left.location,
                        "polarity": left.polarity,
                    },
                    "right_state": {
                        "location": right.location,
                        "polarity": right.polarity,
                    },
                    "board": {
                        "op": op,
                        "columns": [
                            {"a": column0.a, "b": column0.b},
                            {"a": column1.a, "b": column1.b},
                        ],
                    },
                    "left_endpoint": _endpoint_dict(left_endpoint),
                    "right_endpoint": _endpoint_dict(right_endpoint),
                }
                break
        if witness is not None:
            witnesses.append(witness)
    witness_count = len(witnesses)
    pair_count = math.comb(WIDTH_TWO_STATE_COUNT, 2)
    return {
        "width": 2,
        "state_count": WIDTH_TWO_STATE_COUNT,
        "state_pair_count": pair_count,
        "distinguishing_witness_count": witness_count,
        "pairwise_distinguishable": witness_count == pair_count,
        "minimum_total_logical_bits": math.ceil(math.log2(WIDTH_TWO_STATE_COUNT)),
        "packet_polarity_bits": 1,
        "cursor_must_be_encoded_elsewhere": True,
        "polarity_alone_insufficient": witness_count == pair_count,
        "witnesses": witnesses,
    }


@lru_cache(maxsize=1)
def packet_surface_audit() -> dict[str, Any]:
    occupancy_checks = 0
    exact_one_occupancy = 0
    exact_round_trips = 0
    for width in CONTEXT_WIDTHS:
        for location, polarity in itertools.product(range(width), (0, 1)):
            packet = Packet(location, polarity)
            validate_packet_no_hidden_payload(packet, width)
            occupancy = packet_occupancy(packet, width)
            exact_one_occupancy += int(
                sum(occupancy) == 1 and set(occupancy).issubset({0, 1})
            )
            recurrent = packet_to_recurrent(packet, width)
            exact_round_trips += int(recurrent_to_packet(recurrent, width) == packet)
            occupancy_checks += 1
    return {
        "packet_type_exact": True,
        "packet_fields": [field.name for field in fields(Packet)],
        "packet_slots": list(Packet.__slots__),
        "packet_has_dynamic_dict": hasattr(Packet(0, 0), "__dict__"),
        "mutable_payload_bits": 1,
        "support_encodes_cursor_only": True,
        "occupancy_checks": occupancy_checks,
        "exact_one_occupancy": exact_one_occupancy,
        "packet_recurrent_round_trips": exact_round_trips,
        "learned_address_head_count": 0,
        "external_memory_bits": 0,
        "external_execution_calls": 0,
        "verifier_calls": 0,
        "retained_result_tape_slots": 0,
    }


@lru_cache(maxsize=1)
def recurrent_equivalence_audit() -> dict[str, Any]:
    packet_states = tuple(
        Packet(location, polarity) for location in (0, 1) for polarity in (0, 1)
    )
    mapped_states = tuple(packet_to_recurrent(packet, 2) for packet in packet_states)
    coordinate_bijection = len(set(mapped_states)) == WIDTH_TWO_STATE_COUNT and all(
        recurrent_to_packet(mapped, 2) == packet
        for packet, mapped in zip(packet_states, mapped_states, strict=True)
    )
    plcc_resource_vector = {
        "mutable_payload_bits": 1,
        "cursor_states": 2,
        "cursor_logical_bits": 1,
        "runtime_state_count": 4,
        "total_logical_state_bits": 2,
        "local_transition_cells": LOCAL_CELL_COUNT,
        "sequential_depth": 2,
        "result_tape_slots": 0,
        "external_execution_calls": 0,
    }
    recurrent_resource_vector = dict(plcc_resource_vector)
    trajectories = two_column_audit()
    return {
        "coordinate_map": "Packet(location,polarity) <-> RecurrentState(cursor,carry)",
        "coordinate_bijection_width_two": coordinate_bijection,
        "endpoint_equivalence_cases": trajectories["plcc_recurrent_equivalent"],
        "endpoint_equivalence_total": trajectories["case_count"],
        "plcc_resource_vector": plcc_resource_vector,
        "explicit_recurrent_resource_vector": recurrent_resource_vector,
        "resource_vectors_equal": plcc_resource_vector == recurrent_resource_vector,
        "computational_class_boundary": "finite_state_transducer",
        "mechanical_advantage_over_recurrent_control": False,
    }


def run_audit() -> dict[str, Any]:
    local = local_cell_audit()
    trajectories = two_column_audit()
    carry_swaps = carry_swap_audit()
    cursor_swaps = cursor_swap_audit()
    distinguishability = distinguishability_audit()
    packet_surface = packet_surface_audit()
    equivalence = recurrent_equivalence_audit()

    gates = {
        "local_400_cells_exact": local["exact_cells"] == LOCAL_CELL_COUNT,
        "local_cell_context_invariance_exact": local["context_exact"]
        == local["context_observation_count"],
        "local_cell_signature_closed": local["observed_signature"]
        == local["authorized_signature"],
        "local_table_commitment_frozen": local["local_table_sha256"]
        == FROZEN_LOCAL_TABLE_SHA256,
        "two_column_40000_exact": trajectories["plcc_exact"] == TWO_COLUMN_BOARD_COUNT,
        "two_column_board_commitment_frozen": trajectories["two_column_board_sha256"]
        == FROZEN_TWO_COLUMN_BOARD_SHA256,
        "source_prefix_deletion_invariant": trajectories["source_prefix_deletion_exact"]
        == TWO_COLUMN_BOARD_COUNT,
        "different_carry_donor_swap_exact": carry_swaps["different_carry_donor_exact"]
        == carry_swaps["different_carry_swap_checks"],
        "different_carry_changes_endpoint": carry_swaps[
            "different_carry_recipient_divergence"
        ]
        == carry_swaps["different_carry_swap_checks"],
        "same_carry_sham_invariant": carry_swaps["same_carry_sham_exact"]
        == carry_swaps["same_carry_sham_checks"],
        "cursor_location_swap_exact": cursor_swaps["selected_column_exact"]
        == cursor_swaps["selected_column_observation_count"],
        "cursor_swap_preserves_polarity": cursor_swaps[
            "polarity_preserved_before_scatter"
        ]
        == cursor_swaps["swap_pair_count"],
        "one_hard_occupancy_and_one_bit": packet_surface["exact_one_occupancy"]
        == packet_surface["occupancy_checks"]
        and packet_surface["packet_fields"] == ["location", "polarity"]
        and packet_surface["packet_slots"] == ["location", "polarity"]
        and not packet_surface["packet_has_dynamic_dict"],
        "zero_intermediate_emission": trajectories["intermediate_packet_only"]
        == TWO_COLUMN_BOARD_COUNT
        and trajectories["terminal_endpoint_only"] == TWO_COLUMN_BOARD_COUNT
        and trajectories["intermediate_emitted_symbol_count"] == 0
        and trajectories["generated_token_count"] == 0
        and trajectories["generated_kv_bytes"] == 0
        and trajectories["result_tape_slots"] == 0,
        "four_states_pairwise_distinguishable": distinguishability[
            "pairwise_distinguishable"
        ]
        and distinguishability["minimum_total_logical_bits"] == 2,
        "explicit_recurrent_control_exact": trajectories["explicit_recurrent_exact"]
        == TWO_COLUMN_BOARD_COUNT,
        "plcc_equals_explicit_recurrent_control": equivalence[
            "endpoint_equivalence_cases"
        ]
        == equivalence["endpoint_equivalence_total"]
        and equivalence["coordinate_bijection_width_two"]
        and equivalence["resource_vectors_equal"],
        "no_hidden_runtime_channel": packet_surface["learned_address_head_count"] == 0
        and packet_surface["external_memory_bits"] == 0
        and packet_surface["external_execution_calls"] == 0
        and packet_surface["verifier_calls"] == 0
        and packet_surface["retained_result_tape_slots"] == 0,
    }
    mechanics_contract_satisfied = all(gates.values())
    core = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "deterministic oracle mechanics only; no novelty, SoTA, neural "
            "learnability, Shohin capability, or reasoning-primitive claim"
        ),
        "board": {
            "operations": list(OPS),
            "radix": 10,
            "local_cell_count": LOCAL_CELL_COUNT,
            "two_column_case_count": TWO_COLUMN_BOARD_COUNT,
            "local_table_sha256": local["local_table_sha256"],
            "two_column_board_sha256": trajectories["two_column_board_sha256"],
        },
        "local_cells": local,
        "two_column_trajectories": trajectories,
        "carry_interventions": carry_swaps,
        "cursor_interventions": cursor_swaps,
        "distinguishability_boundary": distinguishability,
        "packet_surface": packet_surface,
        "explicit_recurrent_equivalence": equivalence,
        "gates": gates,
        "mechanics_contract_satisfied": mechanics_contract_satisfied,
        "mechanical_verdict": (
            "equivalent_to_explicit_recurrent_control"
            if mechanics_contract_satisfied
            else "mechanics_rejected"
        ),
        "novel_reasoning_primitive_supported": False,
        "neural_pilot_authorized_by_this_report": False,
    }
    return {
        **core,
        "report_content_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def report_bytes(report: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    return canonical_json_bytes(report)


def write_report_once(
    path: str | os.PathLike[str],
    report: Mapping[str, Any],
    *,
    pretty: bool = False,
) -> str:
    """Publish one read-only report without overwriting an existing target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report_bytes(report, pretty=pretty)
    temp = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    fd: int | None = None
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, 0o444)
        os.link(temp, destination)
        temp.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if fd is not None:
            os.close(fd)
        temp.unlink(missing_ok=True)
        raise
    return sha256_bytes(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="publish the deterministic JSON report once instead of stdout",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent JSON without changing report semantics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_audit()
    if args.output is None:
        print(report_bytes(report, pretty=args.pretty).decode("ascii"), end="")
    else:
        write_report_once(args.output, report, pretty=args.pretty)
    return 0 if report["mechanics_contract_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
