#!/usr/bin/env python3
"""Fail-closed calibrated CPU audit for Last-Reset Witness Attention.

This module has two deliberately separate surfaces:

* exhaustive finite mechanics gates over oracle K/P/G transition words; and
* a small endpoint-supervised learning board whose models receive only raw
  operation and decimal-digit features.

Calibration occurred before this contract was frozen.  The mechanics and
negative learning result are exploratory audit evidence only, not
preregistered or canonical scientific evidence.  The scaled board never loads Shohin,
launches a GPU job, performs host arithmetic inside model inference, exposes a
result tape, or consumes generated-token KV state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import stat
from typing import Iterable, Iterator, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn


PROTOCOL_ID = "R12-LRWA-CPU-v2-calibrated"
REPORT_STATUS = "CALIBRATED_EXPLORATORY_REJECTION"
CALIBRATION_DISCLOSURE = (
    "implementation and outcome calibration occurred before contract freeze; "
    "this report is exploratory audit evidence and must not be represented as "
    "preregistered, outcome-naive, canonical, or promotion evidence"
)
CLAIM_BOUNDARY = (
    "known carry-lookahead/reset-monoid routing; calibrated exploratory mechanics "
    "and negative learning audit only; not a new primitive; no GPU or promotion authority"
)
STATUS_ALPHABET = ("K", "P", "G")
TOGGLE_EVENT = "T"
OPERATIONS = ("ADD", "SUB")

BASE_PARAMETER_COUNT = 125_081_664
COMPILER_PARAMETER_COUNT = 21_524_484
MOTOR_PARAMETER_COUNT = 1_195_020
ADDED_PARAMETER_COUNT = 22_719_504
TOTAL_PARAMETER_COUNT = 147_801_168
STRICT_PARAMETER_CAP = 150_000_000
REMAINING_PARAMETER_BUDGET = 2_198_832

COMPILER_SHAPES = ((1153, 4096), (4096, 4096), (4096, 4))
MOTOR_SHAPES = ((1154, 1024), (1024, 12))

RESET_WORD_MAX_LENGTH = 10
POSITION_CONTROL_MAX_LENGTH = 8

RAW_FEATURE_DIMENSION = 24
SCALED_HIDDEN_DIMENSION = 32
SCALED_PARAMETER_COUNT = 2_396

SOURCE_BINDING_PATHS = (
    "R12_LAST_RESET_WITNESS_ATTENTION_PREREG.md",
    "pipeline/last_reset_witness_cpu_falsifier.py",
    "pipeline/test_last_reset_witness_cpu_falsifier.py",
)

# These commitments are filled from the canonical generators and then frozen
# in both code and the preregistration.  A mismatch aborts before a report can
# be published.
FROZEN_RESET_WORD_BOARD_SHA256 = (
    "c5ca6f3ddb6a5192c37527240424563891c0b092ec96593353de9805bda983e7"
)
FROZEN_LOCAL_CELL_BOARD_SHA256 = (
    "d4fd99e9deae46c6d32ac993a0236cebf7d896d8316bf018d55b9f5a0baf076b"
)
FROZEN_TOGGLE_BOARD_SHA256 = (
    "620abd2eec5eff480d42be47ebd04e563a34e28f15d693f7d7c9de826b328461"
)
FROZEN_INTERVENTION_BOARD_SHA256 = (
    "6801acdd5e4319e439f8b776d5d6e431e6dc6e5b1c00d858b28b4a2177326463"
)
FROZEN_TRAIN_SPLIT_SHA256 = (
    "f1f2f4a1e4d15425a5b6d323813c99f4af5de4f17ca15dbe1f716433a3c355bd"
)
FROZEN_EVAL_SPLIT_SHA256: dict[int, str] = {
    4: "986bb8c02cf9ca4e3b16bb59b7e125d4f6d55af634b506891108d27f25dc1b79",
    8: "8154eb246435c8b746224a37906b39f802397f21f0f11c34636e50b9f80ef425",
    16: "f74e70d420e42d8be4ba9091b133233ce8410f0eec08a694417896c97849d69f",
    32: "2b2d8f1fbc207f39d04e5b8eccaf2c7a6de0b541adc1c3c5b5a5a6f883bd573a",
}


class FalsifierError(RuntimeError):
    """Raised when a frozen contract or evidence check fails closed."""


ArmName = Literal["witness", "serial", "dense"]


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Frozen bounded CPU learning board."""

    data_seed: int = 20_260_717
    train_size: int = 1_536
    eval_size_per_width: int = 256
    train_widths: tuple[int, ...] = (4, 8)
    eval_widths: tuple[int, ...] = (4, 8, 16, 32)
    model_seeds: tuple[int, ...] = (1_701, 1_702, 1_703)
    updates: int = 96
    batch_size: int = 128
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    hidden_dimension: int = SCALED_HIDDEN_DIMENSION


@dataclass(frozen=True, slots=True)
class LearningExample:
    """One raw-input endpoint query.

    ``target`` is produced by the offline dataset oracle.  Neither it nor any
    K/P/G status is part of the model input tensor.
    """

    example_id: str
    operation: str
    a_digits: tuple[int, ...]
    b_digits: tuple[int, ...]
    initial_carry: int
    query_position: int
    target: int


DEFAULT_LEARNING_CONFIG = LearningConfig()


def _normalize_json_value(value: object) -> object:
    """Normalize containers so canonical bytes survive a JSON reopen."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise FalsifierError("canonical JSON keys must be strings or integers")
            normalized_key = str(key)
            if normalized_key in normalized:
                raise FalsifierError("canonical JSON key normalization collided")
            normalized[normalized_key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise FalsifierError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the one accepted reopen-stable JSON encoding."""

    return (
        json.dumps(
            _normalize_json_value(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_bindings() -> dict[str, object]:
    """Bind the current three-file audit bytes without attestation claims."""

    root = Path(__file__).resolve().parents[1]
    files: dict[str, dict[str, object]] = {}
    for relative_name in SOURCE_BINDING_PATHS:
        path = root / relative_name
        if path.is_symlink() or not path.is_file():
            raise FalsifierError(
                f"source binding is not a regular file: {relative_name}"
            )
        payload = path.read_bytes()
        files[relative_name] = {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    return {
        "binding_scope": "current_local_file_bytes_at_report_generation",
        "trusted_timestamp_claimed": False,
        "immutable_source_claimed": False,
        "files": files,
    }


def _row_stream_hash(rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def _validate_bit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise FalsifierError(f"{name} must be an exact integer bit")


def _validate_digit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
        raise FalsifierError(f"{name} must be an exact decimal digit")


def _linear_parameter_count(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def deployment_budget() -> dict[str, object]:
    """Independently derive the exact strict-under-150M deployment budget."""

    compiler_layers = [
        {
            "shape": [in_features, out_features],
            "parameters": _linear_parameter_count(in_features, out_features),
        }
        for in_features, out_features in COMPILER_SHAPES
    ]
    motor_layers = [
        {
            "shape": [in_features, out_features],
            "parameters": _linear_parameter_count(in_features, out_features),
        }
        for in_features, out_features in MOTOR_SHAPES
    ]
    compiler = sum(int(layer["parameters"]) for layer in compiler_layers)
    motor = sum(int(layer["parameters"]) for layer in motor_layers)
    added = compiler + motor
    total = BASE_PARAMETER_COUNT + added
    remaining = STRICT_PARAMETER_CAP - total
    if (
        compiler != COMPILER_PARAMETER_COUNT
        or motor != MOTOR_PARAMETER_COUNT
        or added != ADDED_PARAMETER_COUNT
        or total != TOTAL_PARAMETER_COUNT
        or remaining != REMAINING_PARAMETER_BUDGET
        or total >= STRICT_PARAMETER_CAP
    ):
        raise FalsifierError(
            "deployment parameter budget does not match frozen constants"
        )
    control_counts = {
        "last_reset_witness": added,
        "serial_recurrent": added,
        "dense_attention": added,
    }
    if len(set(control_counts.values())) != 1:
        raise FalsifierError("deployment controls are not parameter matched")
    return {
        "base_parameters": BASE_PARAMETER_COUNT,
        "compiler_layers": compiler_layers,
        "compiler_parameters": compiler,
        "motor_layers": motor_layers,
        "motor_parameters": motor,
        "added_parameters": added,
        "total_parameters": total,
        "strict_cap": STRICT_PARAMETER_CAP,
        "strictly_below_cap": total < STRICT_PARAMETER_CAP,
        "remaining_parameters": remaining,
        "control_parameter_counts": control_counts,
        "controls_exactly_parameter_matched": len(set(control_counts.values())) == 1,
    }


def status_for_raw_cell(operation: str, a_digit: int, b_digit: int) -> str:
    """Offline oracle status for the separately labeled finite mechanics gate."""

    _validate_digit(a_digit, "a_digit")
    _validate_digit(b_digit, "b_digit")
    if operation == "ADD":
        total = a_digit + b_digit
        if total <= 8:
            return "K"
        if total == 9:
            return "P"
        return "G"
    if operation == "SUB":
        difference = a_digit - b_digit
        if difference >= 1:
            return "K"
        if difference == 0:
            return "P"
        return "G"
    raise FalsifierError(f"unknown operation {operation!r}")


def apply_status(status: str, carry: int) -> int:
    """Apply one reset-monoid event to a carry/borrow bit."""

    _validate_bit(carry, "carry")
    if status == "K":
        return 0
    if status == "P":
        return carry
    if status == "G":
        return 1
    if status == TOGGLE_EVENT:
        return 1 - carry
    raise FalsifierError(f"unknown status {status!r}")


def raw_local_transition(
    operation: str, a_digit: int, b_digit: int, incoming: int
) -> tuple[int, int]:
    """Offline decimal oracle used only to create labels and finite evidence."""

    _validate_bit(incoming, "incoming")
    _validate_digit(a_digit, "a_digit")
    _validate_digit(b_digit, "b_digit")
    if operation == "ADD":
        value = a_digit + b_digit + incoming
        return value % 10, int(value >= 10)
    if operation == "SUB":
        value = a_digit - b_digit - incoming
        return value % 10, int(value < 0)
    raise FalsifierError(f"unknown operation {operation!r}")


def serial_trace(word: str, initial_carry: int) -> tuple[int, ...]:
    """Carry before query positions 0 through len(word), inclusive."""

    _validate_bit(initial_carry, "initial_carry")
    carry = initial_carry
    trace = [carry]
    for status in word:
        carry = apply_status(status, carry)
        trace.append(carry)
    return tuple(trace)


def witness_carry(prefix: str, initial_carry: int) -> int:
    """Retrieve the value of the last reset before a late query."""

    _validate_bit(initial_carry, "initial_carry")
    for status in reversed(prefix):
        if status == "K":
            return 0
        if status == "G":
            return 1
        if status != "P":
            raise FalsifierError("witness mechanics accepts only K/P/G words")
    return initial_carry


def witness_trace(word: str, initial_carry: int) -> tuple[int, ...]:
    return tuple(
        witness_carry(word[:query], initial_carry) for query in range(len(word) + 1)
    )


def _bits_to_text(bits: Sequence[int]) -> str:
    return "".join(str(bit) for bit in bits)


def _text_to_bits(text: str) -> tuple[int, ...]:
    if not text or any(char not in "01" for char in text):
        raise FalsifierError("trace must be a non-empty bit string")
    return tuple(int(char) for char in text)


def iter_reset_word_cases() -> Iterator[list[object]]:
    """Exhaust K/P/G words length 1..10, both starts, all query traces."""

    for length in range(1, RESET_WORD_MAX_LENGTH + 1):
        for symbols in product(STATUS_ALPHABET, repeat=length):
            word = "".join(symbols)
            for initial in (0, 1):
                yield [
                    word,
                    initial,
                    _bits_to_text(serial_trace(word, initial)),
                    _bits_to_text(witness_trace(word, initial)),
                ]


def build_reset_word_evidence() -> list[list[object]]:
    return list(iter_reset_word_cases())


def iter_local_cell_cases() -> Iterator[list[object]]:
    """Exhaust the 400 operation/digit/incoming-bit local cells."""

    for operation in OPERATIONS:
        for a_digit in range(10):
            for b_digit in range(10):
                status = status_for_raw_cell(operation, a_digit, b_digit)
                for incoming in (0, 1):
                    digit, outgoing = raw_local_transition(
                        operation, a_digit, b_digit, incoming
                    )
                    yield [
                        operation,
                        a_digit,
                        b_digit,
                        incoming,
                        status,
                        digit,
                        outgoing,
                    ]


def build_local_cell_evidence() -> list[list[object]]:
    return list(iter_local_cell_cases())


def iter_toggle_cases() -> Iterator[list[object]]:
    """Complete toggle truth table under every available K/P/G alias.

    A reset-monoid-only witness has no toggle action.  For each fixed alias it
    is tested on both possible incoming bits.  The favorable four-function
    recurrent control receives the real T transition.
    """

    for alias in STATUS_ALPHABET:
        for incoming in (0, 1):
            candidate = apply_status(alias, incoming)
            recurrent = apply_status(TOGGLE_EVENT, incoming)
            yield [alias, incoming, candidate, recurrent]


def build_toggle_evidence() -> list[list[object]]:
    return list(iter_toggle_cases())


def _first_raw_pair(operation: str, status: str, ordinal: int = 0) -> tuple[int, int]:
    pairs = [
        (a_digit, b_digit)
        for a_digit in range(10)
        for b_digit in range(10)
        if status_for_raw_cell(operation, a_digit, b_digit) == status
    ]
    if ordinal >= len(pairs):
        raise FalsifierError("raw status class lacks the requested donor")
    return pairs[ordinal]


def _rotate_text(text: str) -> str:
    return text[1:] + text[:1] if text else text


def _gate_value_endpoint(word: str, initial: int, *, shuffle: str) -> int:
    gates = [int(status in ("K", "G")) for status in word]
    values = [int(status == "G") for status in word]
    if shuffle == "gate":
        gates = gates[1:] + gates[:1]
    elif shuffle == "value":
        reset_indices = [index for index, gate in enumerate(gates) if gate]
        if len(reset_indices) >= 2:
            rotated = [values[index] for index in reset_indices]
            rotated = rotated[1:] + rotated[:1]
            for index, value in zip(reset_indices, rotated):
                values[index] = value
        elif len(reset_indices) == 1:
            index = reset_indices[0]
            values[index] = 1 - values[index]
    else:
        raise FalsifierError(f"unknown shuffle {shuffle!r}")
    carry = initial
    for gate, value in zip(gates, values):
        if gate:
            carry = value
    return carry


def build_intervention_evidence() -> dict[str, list[list[object]]]:
    """Build raw mechanics evidence for all frozen causal/sham controls."""

    position_rows: list[list[object]] = []
    shuffle_rows: list[list[object]] = []
    for length in range(2, POSITION_CONTROL_MAX_LENGTH + 1):
        for symbols in product(STATUS_ALPHABET, repeat=length):
            word = "".join(symbols)
            reversed_word = word[::-1]
            for initial in (0, 1):
                baseline = serial_trace(word, initial)[-1]
                position_rows.append(
                    [
                        word,
                        initial,
                        reversed_word,
                        baseline,
                        serial_trace(reversed_word, initial)[-1],
                    ]
                )
                shuffle_rows.append(
                    [
                        word,
                        initial,
                        baseline,
                        _gate_value_endpoint(word, initial, shuffle="gate"),
                        _gate_value_endpoint(word, initial, shuffle="value"),
                    ]
                )

    donor_rows: list[list[object]] = []
    same_status_rows: list[list[object]] = []
    for operation in OPERATIONS:
        k_a, k_b = _first_raw_pair(operation, "K")
        g_a, g_b = _first_raw_pair(operation, "G")
        for suffix_a in range(10):
            for suffix_b in range(10):
                for initial in (0, 1):
                    _, carry_k = raw_local_transition(operation, k_a, k_b, initial)
                    _, carry_g = raw_local_transition(operation, g_a, g_b, initial)
                    digit_k, out_k = raw_local_transition(
                        operation, suffix_a, suffix_b, carry_k
                    )
                    digit_g, out_g = raw_local_transition(
                        operation, suffix_a, suffix_b, carry_g
                    )
                    donor_rows.append(
                        [
                            operation,
                            [k_a, k_b],
                            [g_a, g_b],
                            [suffix_a, suffix_b],
                            initial,
                            [digit_k, out_k],
                            [digit_g, out_g],
                        ]
                    )
        for status in STATUS_ALPHABET:
            first = _first_raw_pair(operation, status, 0)
            second = _first_raw_pair(operation, status, 1)
            for incoming in (0, 1):
                _, first_out = raw_local_transition(operation, *first, incoming)
                _, second_out = raw_local_transition(operation, *second, incoming)
                for suffix_a in range(10):
                    for suffix_b in range(10):
                        first_endpoint = raw_local_transition(
                            operation, suffix_a, suffix_b, first_out
                        )
                        second_endpoint = raw_local_transition(
                            operation, suffix_a, suffix_b, second_out
                        )
                        same_status_rows.append(
                            [
                                operation,
                                status,
                                list(first),
                                list(second),
                                incoming,
                                [suffix_a, suffix_b],
                                list(first_endpoint),
                                list(second_endpoint),
                            ]
                        )

    shadowed_rows: list[list[object]] = []
    for early in ("K", "G"):
        for late in ("K", "G"):
            for gap in range(7):
                for initial in (0, 1):
                    baseline_word = early + ("P" * gap) + late
                    sham_word = ("G" if early == "K" else "K") + ("P" * gap) + late
                    shadowed_rows.append(
                        [
                            baseline_word,
                            sham_word,
                            initial,
                            serial_trace(baseline_word, initial)[-1],
                            serial_trace(sham_word, initial)[-1],
                        ]
                    )

    generated_prefix_rows: list[list[object]] = []
    corruptions = ("", "0", "999999", "<think>wrong</think>", "KPGT")
    selected_words = ("P", "KPG", "GPPPK", "PPPPPP", "KPGPGPGP")
    for word in selected_words:
        for initial in (0, 1):
            endpoint = witness_trace(word, initial)[-1]
            for corruption in corruptions:
                generated_prefix_rows.append(
                    [word, initial, corruption, endpoint, endpoint]
                )

    return {
        "position_shuffle": position_rows,
        "gate_value_shuffle": shuffle_rows,
        "kg_donor_selectivity": donor_rows,
        "same_status_sham": same_status_rows,
        "shadowed_witness_sham": shadowed_rows,
        "generated_prefix_invariance": generated_prefix_rows,
    }


def _independent_reset_summary(rows: Sequence[Sequence[object]]) -> dict[str, object]:
    expected = iter_reset_word_cases()
    case_count = 0
    query_count = 0
    exact = 0
    for observed, expected_row in zip(rows, expected, strict=True):
        if list(observed[:2]) != expected_row[:2]:
            raise FalsifierError("reset-word evidence order/input changed")
        word = observed[0]
        initial = observed[1]
        if not isinstance(word, str) or not isinstance(initial, int):
            raise FalsifierError("reset-word evidence has invalid input types")
        serial_observed = _text_to_bits(str(observed[2]))
        witness_observed = _text_to_bits(str(observed[3]))
        serial_expected = serial_trace(word, initial)
        witness_expected = witness_trace(word, initial)
        if serial_observed != serial_expected or witness_observed != witness_expected:
            raise FalsifierError("reset-word raw evidence does not recompute")
        case_count += 1
        query_count += len(serial_expected)
        exact += sum(
            int(serial_value == witness_value)
            for serial_value, witness_value in zip(
                serial_observed, witness_observed, strict=True
            )
        )
    expected_count = 2 * sum(3**length for length in range(1, 11))
    if case_count != expected_count or len(rows) != expected_count:
        raise FalsifierError("reset-word evidence is incomplete")
    return {
        "word_case_count": case_count,
        "query_observation_count": query_count,
        "witness_serial_exact": exact,
        "all_query_positions_exact": exact == query_count,
        "board_sha256": _row_stream_hash(rows),
    }


def _independent_local_summary(rows: Sequence[Sequence[object]]) -> dict[str, object]:
    expected = iter_local_cell_cases()
    exact = 0
    status_exact = 0
    count = 0
    for observed, expected_row in zip(rows, expected, strict=True):
        if list(observed[:4]) != expected_row[:4]:
            raise FalsifierError("local-cell evidence order/input changed")
        operation, a_digit, b_digit, incoming = observed[:4]
        status = status_for_raw_cell(str(operation), int(a_digit), int(b_digit))
        digit, outgoing = raw_local_transition(
            str(operation), int(a_digit), int(b_digit), int(incoming)
        )
        status_exact += int(observed[4] == status)
        exact += int([digit, outgoing] == list(observed[5:7]))
        count += 1
    if count != 400 or len(rows) != 400:
        raise FalsifierError("local-cell evidence is incomplete")
    return {
        "cell_count": count,
        "status_exact": status_exact,
        "transition_exact": exact,
        "all_400_exact": status_exact == exact == 400,
        "board_sha256": _row_stream_hash(rows),
        "status_scope": "finite_mechanics_label_only",
    }


def _independent_toggle_summary(rows: Sequence[Sequence[object]]) -> dict[str, object]:
    expected = iter_toggle_cases()
    by_alias: dict[str, list[int]] = {status: [] for status in STATUS_ALPHABET}
    recurrent_exact = 0
    count = 0
    for observed, expected_row in zip(rows, expected, strict=True):
        if list(observed[:2]) != expected_row[:2]:
            raise FalsifierError("toggle evidence order/input changed")
        alias = str(observed[0])
        incoming = int(observed[1])
        candidate = apply_status(alias, incoming)
        recurrent = apply_status(TOGGLE_EVENT, incoming)
        if observed[2] != candidate or observed[3] != recurrent:
            raise FalsifierError("toggle raw evidence does not recompute")
        by_alias[alias].append(int(candidate == recurrent))
        recurrent_exact += int(recurrent == 1 - incoming)
        count += 1
    if count != 6 or len(rows) != 6:
        raise FalsifierError("toggle truth table is incomplete")
    alias_accuracy = {
        alias: sum(values) / len(values) for alias, values in by_alias.items()
    }
    return {
        "truth_table_cases": count,
        "candidate_alias_accuracy": alias_accuracy,
        "best_candidate_accuracy": max(alias_accuracy.values()),
        "four_function_recurrent_accuracy": recurrent_exact / count,
        "candidate_at_most_75_percent": max(alias_accuracy.values()) <= 0.75,
        "recurrent_exact": recurrent_exact == count,
        "board_sha256": _row_stream_hash(rows),
    }


def _independent_intervention_summary(
    evidence: Mapping[str, Sequence[Sequence[object]]],
) -> dict[str, object]:
    required = {
        "position_shuffle",
        "gate_value_shuffle",
        "kg_donor_selectivity",
        "same_status_sham",
        "shadowed_witness_sham",
        "generated_prefix_invariance",
    }
    if set(evidence) != required:
        raise FalsifierError("intervention evidence has a non-canonical key set")

    position_changed = 0
    for row in evidence["position_shuffle"]:
        word, initial, shuffled, baseline, observed_shuffled = row
        if shuffled != str(word)[::-1]:
            raise FalsifierError("position shuffle is not the frozen reversal")
        expected_base = serial_trace(str(word), int(initial))[-1]
        expected_shuffle = serial_trace(str(shuffled), int(initial))[-1]
        if baseline != expected_base or observed_shuffled != expected_shuffle:
            raise FalsifierError("position-shuffle evidence does not recompute")
        position_changed += int(expected_base != expected_shuffle)

    gate_changed = 0
    value_changed = 0
    for row in evidence["gate_value_shuffle"]:
        word, initial, baseline, gate_endpoint, value_endpoint = row
        expected_base = serial_trace(str(word), int(initial))[-1]
        expected_gate = _gate_value_endpoint(str(word), int(initial), shuffle="gate")
        expected_value = _gate_value_endpoint(str(word), int(initial), shuffle="value")
        if [baseline, gate_endpoint, value_endpoint] != [
            expected_base,
            expected_gate,
            expected_value,
        ]:
            raise FalsifierError("gate/value-shuffle evidence does not recompute")
        gate_changed += int(expected_base != expected_gate)
        value_changed += int(expected_base != expected_value)

    donor_exact = 0
    donor_selective = 0
    for row in evidence["kg_donor_selectivity"]:
        operation, k_pair, g_pair, suffix, initial, k_endpoint, g_endpoint = row
        _, k_carry = raw_local_transition(
            str(operation), int(k_pair[0]), int(k_pair[1]), int(initial)
        )
        _, g_carry = raw_local_transition(
            str(operation), int(g_pair[0]), int(g_pair[1]), int(initial)
        )
        expected_k = raw_local_transition(
            str(operation), int(suffix[0]), int(suffix[1]), k_carry
        )
        expected_g = raw_local_transition(
            str(operation), int(suffix[0]), int(suffix[1]), g_carry
        )
        donor_exact += int(
            list(expected_k) == k_endpoint and list(expected_g) == g_endpoint
        )
        donor_selective += int(expected_k != expected_g)

    same_status_exact = 0
    for row in evidence["same_status_sham"]:
        operation, status, first, second, incoming, suffix, first_out, second_out = row
        if status_for_raw_cell(str(operation), int(first[0]), int(first[1])) != status:
            raise FalsifierError("same-status first donor has the wrong status")
        if (
            status_for_raw_cell(str(operation), int(second[0]), int(second[1]))
            != status
        ):
            raise FalsifierError("same-status second donor has the wrong status")
        _, first_carry = raw_local_transition(
            str(operation), int(first[0]), int(first[1]), int(incoming)
        )
        _, second_carry = raw_local_transition(
            str(operation), int(second[0]), int(second[1]), int(incoming)
        )
        expected_first = raw_local_transition(
            str(operation), int(suffix[0]), int(suffix[1]), first_carry
        )
        expected_second = raw_local_transition(
            str(operation), int(suffix[0]), int(suffix[1]), second_carry
        )
        if list(expected_first) != first_out or list(expected_second) != second_out:
            raise FalsifierError("same-status sham evidence does not recompute")
        same_status_exact += int(expected_first == expected_second)

    shadowed_exact = 0
    for row in evidence["shadowed_witness_sham"]:
        baseline_word, sham_word, initial, baseline, sham = row
        expected_base = serial_trace(str(baseline_word), int(initial))[-1]
        expected_sham = serial_trace(str(sham_word), int(initial))[-1]
        if baseline != expected_base or sham != expected_sham:
            raise FalsifierError("shadowed sham evidence does not recompute")
        shadowed_exact += int(expected_base == expected_sham)

    generated_exact = 0
    for row in evidence["generated_prefix_invariance"]:
        word, initial, _corruption, baseline, corrupted = row
        expected = witness_trace(str(word), int(initial))[-1]
        if baseline != expected or corrupted != expected:
            raise FalsifierError("generated-prefix evidence does not recompute")
        generated_exact += 1

    flat_rows = [[name, row] for name in sorted(evidence) for row in evidence[name]]
    return {
        "position_cases": len(evidence["position_shuffle"]),
        "position_changed": position_changed,
        "gate_shuffle_changed": gate_changed,
        "value_shuffle_changed": value_changed,
        "kg_donor_cases": len(evidence["kg_donor_selectivity"]),
        "kg_donor_exact": donor_exact,
        "kg_donor_selective": donor_selective,
        "same_status_sham_cases": len(evidence["same_status_sham"]),
        "same_status_sham_exact": same_status_exact,
        "shadowed_sham_cases": len(evidence["shadowed_witness_sham"]),
        "shadowed_sham_exact": shadowed_exact,
        "generated_prefix_cases": len(evidence["generated_prefix_invariance"]),
        "generated_prefix_exact": generated_exact,
        "board_sha256": _row_stream_hash(flat_rows),
    }


def build_mechanics_evidence() -> dict[str, object]:
    return {
        "reset_words": build_reset_word_evidence(),
        "local_cells": build_local_cell_evidence(),
        "toggle_truth_table": build_toggle_evidence(),
        "interventions": build_intervention_evidence(),
    }


def recompute_mechanics_from_evidence(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    if set(evidence) != {
        "reset_words",
        "local_cells",
        "toggle_truth_table",
        "interventions",
    }:
        raise FalsifierError("mechanics evidence has a non-canonical key set")
    reset = _independent_reset_summary(evidence["reset_words"])  # type: ignore[arg-type]
    local = _independent_local_summary(evidence["local_cells"])  # type: ignore[arg-type]
    toggle = _independent_toggle_summary(evidence["toggle_truth_table"])  # type: ignore[arg-type]
    interventions = _independent_intervention_summary(  # type: ignore[arg-type]
        evidence["interventions"]
    )
    commitments = {
        "reset_word_board_sha256": reset["board_sha256"],
        "local_cell_board_sha256": local["board_sha256"],
        "toggle_board_sha256": toggle["board_sha256"],
        "intervention_board_sha256": interventions["board_sha256"],
    }
    gates = {
        "exhaustive_reset_identity": bool(reset["all_query_positions_exact"]),
        "all_400_raw_local_cells_exact": bool(local["all_400_exact"]),
        "toggle_rejects_reset_only_candidate": bool(
            toggle["candidate_at_most_75_percent"]
        ),
        "toggle_four_function_recurrent_exact": bool(toggle["recurrent_exact"]),
        "position_shuffle_is_non_vacuous": int(interventions["position_changed"]) > 0,
        "gate_shuffle_is_non_vacuous": int(interventions["gate_shuffle_changed"]) > 0,
        "value_shuffle_is_non_vacuous": int(interventions["value_shuffle_changed"]) > 0,
        "kg_donor_selectivity_exact": interventions["kg_donor_exact"]
        == interventions["kg_donor_cases"]
        == interventions["kg_donor_selective"],
        "same_status_sham_exact": interventions["same_status_sham_exact"]
        == interventions["same_status_sham_cases"],
        "shadowed_witness_sham_exact": interventions["shadowed_sham_exact"]
        == interventions["shadowed_sham_cases"],
        "generated_prefix_invariance_exact": interventions["generated_prefix_exact"]
        == interventions["generated_prefix_cases"],
    }
    return {
        "reset_words": reset,
        "local_cells": local,
        "toggle_negative_control": toggle,
        "interventions": interventions,
        "commitments": commitments,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def _offline_target(
    operation: str,
    a_digits: Sequence[int],
    b_digits: Sequence[int],
    initial_carry: int,
    query_position: int,
) -> int:
    """Create one endpoint label; this function is never called by a model."""

    if len(a_digits) != len(b_digits) or not a_digits:
        raise FalsifierError("operand widths must match and be nonzero")
    if not 0 <= query_position <= len(a_digits):
        raise FalsifierError("query position lies outside the operand")
    carry = initial_carry
    for index in range(query_position):
        _, carry = raw_local_transition(
            operation, a_digits[index], b_digits[index], carry
        )
    if query_position == len(a_digits):
        return 10 + carry
    digit, _ = raw_local_transition(
        operation,
        a_digits[query_position],
        b_digits[query_position],
        carry,
    )
    return digit


def build_learning_split(
    *, split: str, widths: Sequence[int], size: int, seed: int
) -> tuple[LearningExample, ...]:
    if not split or size <= 0 or not widths or any(width <= 0 for width in widths):
        raise FalsifierError("invalid learning split configuration")
    rng = random.Random(seed)
    examples: list[LearningExample] = []
    for index in range(size):
        width = int(widths[index % len(widths)])
        operation = OPERATIONS[(index // len(widths)) % len(OPERATIONS)]
        initial = (index // (len(widths) * len(OPERATIONS))) % 2
        a_digits = tuple(rng.randrange(10) for _ in range(width))
        b_digits = tuple(rng.randrange(10) for _ in range(width))
        query = rng.randrange(width + 1)
        target = _offline_target(operation, a_digits, b_digits, initial, query)
        examples.append(
            LearningExample(
                example_id=f"{split}-{index:06d}",
                operation=operation,
                a_digits=a_digits,
                b_digits=b_digits,
                initial_carry=initial,
                query_position=query,
                target=target,
            )
        )
    return tuple(examples)


def _example_record(example: LearningExample) -> dict[str, object]:
    record = asdict(example)
    record["a_digits"] = list(example.a_digits)
    record["b_digits"] = list(example.b_digits)
    return record


def learning_split_sha256(examples: Sequence[LearningExample]) -> str:
    return _row_stream_hash(_example_record(example) for example in examples)


def build_frozen_learning_splits(
    config: LearningConfig = DEFAULT_LEARNING_CONFIG,
) -> tuple[tuple[LearningExample, ...], dict[int, tuple[LearningExample, ...]]]:
    train = build_learning_split(
        split="train",
        widths=config.train_widths,
        size=config.train_size,
        seed=config.data_seed,
    )
    evaluations = {
        width: build_learning_split(
            split=f"eval-w{width}",
            widths=(width,),
            size=config.eval_size_per_width,
            seed=config.data_seed + 10_000 + width,
        )
        for width in config.eval_widths
    }
    return train, evaluations


def _raw_feature(
    operation: str, a_digit: int | None, b_digit: int | None
) -> list[float]:
    """Encode only operation, raw digits, terminal flag, and a constant."""

    values = [0.0] * RAW_FEATURE_DIMENSION
    if operation not in OPERATIONS:
        raise FalsifierError("raw feature has an unknown operation")
    values[OPERATIONS.index(operation)] = 1.0
    if a_digit is None or b_digit is None:
        if a_digit is not None or b_digit is not None:
            raise FalsifierError("terminal raw feature must omit both digits")
        values[22] = 1.0
    else:
        _validate_digit(a_digit, "a_digit")
        _validate_digit(b_digit, "b_digit")
        values[2 + a_digit] = 1.0
        values[12 + b_digit] = 1.0
    values[23] = 1.0
    return values


def tensorize_examples(examples: Sequence[LearningExample]) -> dict[str, Tensor]:
    if not examples:
        raise FalsifierError("cannot tensorize an empty learning split")
    maximum_width = max(len(example.a_digits) for example in examples)
    features = torch.zeros(
        (len(examples), maximum_width + 1, RAW_FEATURE_DIMENSION),
        dtype=torch.float32,
        device="cpu",
    )
    queries = torch.empty(len(examples), dtype=torch.long, device="cpu")
    initial = torch.empty(len(examples), dtype=torch.float32, device="cpu")
    targets = torch.empty(len(examples), dtype=torch.long, device="cpu")
    for row, example in enumerate(examples):
        for position, (a_digit, b_digit) in enumerate(
            zip(example.a_digits, example.b_digits, strict=True)
        ):
            features[row, position] = torch.tensor(
                _raw_feature(example.operation, a_digit, b_digit),
                dtype=torch.float32,
            )
        terminal_position = len(example.a_digits)
        features[row, terminal_position] = torch.tensor(
            _raw_feature(example.operation, None, None), dtype=torch.float32
        )
        queries[row] = example.query_position
        initial[row] = float(example.initial_carry)
        targets[row] = example.target
    return {
        "features": features,
        "queries": queries,
        "initial": initial,
        "targets": targets,
    }


class ScaledRoutingArm(nn.Module):
    """Exactly matched sidecar tensors under three routing topologies."""

    def __init__(self, arm: ArmName, hidden_dimension: int) -> None:
        super().__init__()
        if arm not in ("witness", "serial", "dense"):
            raise FalsifierError(f"unknown learning arm {arm!r}")
        self.arm = arm
        self.encoder_in = nn.Linear(RAW_FEATURE_DIMENSION, hidden_dimension)
        self.encoder_hidden = nn.Linear(hidden_dimension, hidden_dimension)
        self.router = nn.Linear(hidden_dimension, 4)
        self.motor = nn.Linear(hidden_dimension + 1, 12)

    def _encode(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = torch.tanh(self.encoder_in(features))
        hidden = torch.tanh(self.encoder_hidden(hidden))
        return hidden, self.router(hidden)

    @staticmethod
    def _history_mask(queries: Tensor, positions: Tensor) -> Tensor:
        return positions.unsqueeze(0) < queries.unsqueeze(1)

    def _witness_route(
        self, routes: Tensor, queries: Tensor, initial: Tensor
    ) -> Tensor:
        batch, time, _ = routes.shape
        positions = torch.arange(time, dtype=routes.dtype, device=routes.device)
        history = self._history_mask(queries, positions)
        reset_logit = torch.logsumexp(routes[..., (0, 2)], dim=-1) - torch.logsumexp(
            routes[..., (1, 3)], dim=-1
        )
        recency = 0.5 * positions.unsqueeze(0) / max(time - 1, 1)
        scores = reset_logit + recency
        scores = scores.masked_fill(~history, -1.0e9)
        sentinel_score = torch.zeros(
            (batch, 1), dtype=routes.dtype, device=routes.device
        )
        all_scores = torch.cat((sentinel_score, scores), dim=1)
        weights = torch.softmax(all_scores, dim=1)
        reset_values = torch.sigmoid(routes[..., 2] - routes[..., 0])
        values = torch.cat((initial.unsqueeze(1), reset_values), dim=1)
        return (weights * values).sum(dim=1)

    def _serial_route(self, routes: Tensor, queries: Tensor, initial: Tensor) -> Tensor:
        probabilities = torch.softmax(routes, dim=-1)
        carry = initial
        for position in range(routes.shape[1]):
            updated = (
                probabilities[:, position, 2]
                + (probabilities[:, position, 1] + probabilities[:, position, 3])
                * carry
            )
            carry = torch.where(position < queries, updated, carry)
        return carry

    def _dense_route(self, routes: Tensor, queries: Tensor, initial: Tensor) -> Tensor:
        batch, time, _ = routes.shape
        row = torch.arange(batch, device=routes.device)
        current = routes[row, queries]
        scores = torch.einsum("btd,bd->bt", routes, current) / 2.0
        positions = torch.arange(time, dtype=routes.dtype, device=routes.device)
        scores = scores + 0.5 * positions.unsqueeze(0) / max(time - 1, 1)
        history = self._history_mask(queries, positions)
        scores = scores.masked_fill(~history, -1.0e9)
        sentinel_score = torch.zeros(
            (batch, 1), dtype=routes.dtype, device=routes.device
        )
        weights = torch.softmax(torch.cat((sentinel_score, scores), dim=1), dim=1)
        values = torch.sigmoid(routes[..., 2] - routes[..., 0])
        return (weights * torch.cat((initial.unsqueeze(1), values), dim=1)).sum(dim=1)

    def forward(self, features: Tensor, queries: Tensor, initial: Tensor) -> Tensor:
        """Infer endpoint logits from raw source features only."""

        if features.device.type != "cpu":
            raise FalsifierError("scaled falsifier refuses non-CPU tensors")
        hidden, routes = self._encode(features)
        if self.arm == "witness":
            carry = self._witness_route(routes, queries, initial)
        elif self.arm == "serial":
            carry = self._serial_route(routes, queries, initial)
        else:
            carry = self._dense_route(routes, queries, initial)
        row = torch.arange(features.shape[0], device=features.device)
        current = hidden[row, queries]
        return self.motor(torch.cat((current, carry.unsqueeze(1)), dim=1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def scaled_parameter_audit(
    hidden_dimension: int = SCALED_HIDDEN_DIMENSION,
) -> dict[str, object]:
    counts = {}
    for arm in ("witness", "serial", "dense"):
        torch.manual_seed(0)
        model = ScaledRoutingArm(arm, hidden_dimension)  # type: ignore[arg-type]
        counts[arm] = count_trainable_parameters(model)
    expected = (
        _linear_parameter_count(RAW_FEATURE_DIMENSION, hidden_dimension)
        + _linear_parameter_count(hidden_dimension, hidden_dimension)
        + _linear_parameter_count(hidden_dimension, 4)
        + _linear_parameter_count(hidden_dimension + 1, 12)
    )
    if any(count != expected for count in counts.values()):
        raise FalsifierError("scaled arm parameter count mismatch")
    if (
        hidden_dimension == SCALED_HIDDEN_DIMENSION
        and expected != SCALED_PARAMETER_COUNT
    ):
        raise FalsifierError("frozen scaled parameter count changed")
    return {
        "hidden_dimension": hidden_dimension,
        "expected_parameters": expected,
        "arm_parameter_counts": counts,
        "exactly_matched": len(set(counts.values())) == 1,
        "all_parameters_used_by_each_forward": True,
    }


def useful_forward_flops(arm: ArmName, width: int, query: int) -> int:
    """Heuristic static operation count for one scaled example.

    This is not an executed-graph profiler measurement and must not be reported
    as measured hardware FLOPs.
    """

    positions = width + 1
    common_per_position = 2 * (
        RAW_FEATURE_DIMENSION * SCALED_HIDDEN_DIMENSION
        + SCALED_HIDDEN_DIMENSION * SCALED_HIDDEN_DIMENSION
        + SCALED_HIDDEN_DIMENSION * 4
    )
    motor = 2 * (SCALED_HIDDEN_DIMENSION + 1) * 12
    routing_per_history = {"witness": 24, "serial": 20, "dense": 28}[arm]
    return positions * common_per_position + motor + query * routing_per_history


def scaled_flop_audit(maximum_width: int = 32) -> dict[str, object]:
    totals = {
        arm: sum(
            useful_forward_flops(arm, maximum_width, query)
            for query in range(maximum_width + 1)
        )
        for arm in ("witness", "serial", "dense")
    }
    ratio = max(totals.values()) / min(totals.values())
    return {
        "width": maximum_width,
        "query_positions": maximum_width + 1,
        "heuristic_forward_operation_counts": totals,
        "max_min_ratio": ratio,
        "within_one_percent": ratio <= 1.01,
        "measurement_kind": "static_heuristic_multiply_add_and_routing_estimate",
        "executed_graph_measurement": False,
        "hardware_flops_claimed": False,
        "wall_clock_measurement": False,
        "decision_use": "exploratory_nomination_expression_only",
    }


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batch_indices(
    size: int, batch_size: int, updates: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    pool = list(range(size))
    cursor = len(pool)
    batches: list[list[int]] = []
    for _ in range(updates):
        batch: list[int] = []
        while len(batch) < batch_size:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            take = min(batch_size - len(batch), len(pool) - cursor)
            batch.extend(pool[cursor : cursor + take])
            cursor += take
        batches.append(batch)
    return batches


def _select_batch(
    tensors: Mapping[str, Tensor], indices: Sequence[int]
) -> dict[str, Tensor]:
    index_tensor = torch.tensor(indices, dtype=torch.long, device="cpu")
    return {
        name: tensor.index_select(0, index_tensor) for name, tensor in tensors.items()
    }


def train_scaled_arm(
    arm: ArmName,
    seed: int,
    train_examples: Sequence[LearningExample],
    config: LearningConfig,
) -> tuple[ScaledRoutingArm, list[float]]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = ScaledRoutingArm(arm, config.hidden_dimension).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tensors = tensorize_examples(train_examples)
    batches = _batch_indices(
        len(train_examples), config.batch_size, config.updates, seed + 91_000
    )
    losses: list[float] = []
    model.train()
    for indices in batches:
        batch = _select_batch(tensors, indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"], batch["queries"], batch["initial"])
        loss = nn.functional.cross_entropy(logits, batch["targets"])
        if not torch.isfinite(loss):
            raise FalsifierError("scaled learning produced a non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


def evaluate_scaled_arm(
    model: ScaledRoutingArm, examples: Sequence[LearningExample]
) -> tuple[float, list[list[object]]]:
    tensors = tensorize_examples(examples)
    model.eval()
    with torch.no_grad():
        logits = model(tensors["features"], tensors["queries"], tensors["initial"])
        predictions = logits.argmax(dim=1).tolist()
    rows = [
        [example.example_id, example.target, int(prediction)]
        for example, prediction in zip(examples, predictions, strict=True)
    ]
    accuracy = sum(int(target == prediction) for _, target, prediction in rows) / len(
        rows
    )
    return accuracy, rows


def _median(values: Sequence[float]) -> float:
    if not values:
        raise FalsifierError("cannot take an empty median")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def run_scaled_learning(
    config: LearningConfig = DEFAULT_LEARNING_CONFIG,
) -> dict[str, object]:
    """Rerun the calibrated bounded endpoint-only CPU comparison.

    Calibration preceded this contract.  The result is exploratory rejection
    evidence only and cannot authorize architecture promotion or a GPU run.
    """

    train_examples, eval_splits = build_frozen_learning_splits(config)
    train_hash = learning_split_sha256(train_examples)
    eval_hashes = {
        width: learning_split_sha256(examples)
        for width, examples in eval_splits.items()
    }
    if config == DEFAULT_LEARNING_CONFIG:
        if train_hash != FROZEN_TRAIN_SPLIT_SHA256:
            raise FalsifierError("frozen training split hash mismatch")
        if eval_hashes != FROZEN_EVAL_SPLIT_SHA256:
            raise FalsifierError("frozen evaluation split hash mismatch")

    raw_rows: list[list[object]] = []
    runs: list[dict[str, object]] = []
    parameter_audit = scaled_parameter_audit(config.hidden_dimension)
    accuracies: dict[str, dict[int, list[float]]] = {
        arm: {width: [] for width in config.eval_widths}
        for arm in ("witness", "serial", "dense")
    }
    for seed in config.model_seeds:
        initial_hashes: dict[str, str] = {}
        for arm in ("witness", "serial", "dense"):
            torch.manual_seed(seed)
            initial_hashes[arm] = _state_dict_sha256(
                ScaledRoutingArm(arm, config.hidden_dimension)  # type: ignore[arg-type]
            )
        if len(set(initial_hashes.values())) != 1:
            raise FalsifierError("same-seed scaled arms do not start identically")

        for arm in ("witness", "serial", "dense"):
            model, losses = train_scaled_arm(
                arm,
                seed,
                train_examples,
                config,  # type: ignore[arg-type]
            )
            width_scores: dict[int, float] = {}
            run_rows: list[list[object]] = []
            for width, examples in eval_splits.items():
                accuracy, rows = evaluate_scaled_arm(model, examples)
                width_scores[width] = accuracy
                accuracies[arm][width].append(accuracy)
                labeled_rows = [[arm, seed, width, *row] for row in rows]
                run_rows.extend(labeled_rows)
                raw_rows.extend(labeled_rows)
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "trainable_parameter_count": count_trainable_parameters(model),
                    "initial_state_sha256": initial_hashes[arm],
                    "final_state_sha256": _state_dict_sha256(model),
                    "first_loss": losses[0],
                    "final_loss": losses[-1],
                    "loss_trace": losses,
                    "loss_trace_sha256": _row_stream_hash(losses),
                    "accuracy_by_width": width_scores,
                    "prediction_rows_sha256": _row_stream_hash(run_rows),
                }
            )

    medians = {
        arm: {width: _median(values) for width, values in width_scores.items()}
        for arm, width_scores in accuracies.items()
    }
    primary_width = max(config.eval_widths)
    witness_primary = medians["witness"][primary_width]
    serial_primary = medians["serial"][primary_width]
    dense_primary = medians["dense"][primary_width]
    per_seed_advantage = [
        witness - serial
        for witness, serial in zip(
            accuracies["witness"][primary_width],
            accuracies["serial"][primary_width],
            strict=True,
        )
    ]
    dense_within_two_points = dense_primary >= witness_primary - 0.02
    serial_wins = serial_primary >= witness_primary
    flop_audit = scaled_flop_audit(primary_width)
    decision_inputs = {
        "witness_primary_at_least_95_percent": witness_primary >= 0.95,
        "at_least_two_seeds_15pp_over_serial": sum(
            advantage >= 0.15 for advantage in per_seed_advantage
        )
        >= 2,
        "dense_not_within_two_percentage_points": not dense_within_two_points,
        "serial_does_not_win_or_tie": not serial_wins,
        "exact_parameter_matching": bool(parameter_audit["exactly_matched"])
        and all(
            count == parameter_audit["expected_parameters"]
            for count in parameter_audit["arm_parameter_counts"].values()
        ),
        "heuristic_flop_ratio_at_most_1_01": bool(flop_audit["within_one_percent"])
        and float(flop_audit["max_min_ratio"]) <= 1.01,
    }
    scaled_candidate_gate = all(decision_inputs.values())
    recomputed_rows = recompute_learning_rows(raw_rows, config)
    return {
        "scope": "calibrated_exploratory_audit_not_preregistered_evidence",
        "calibration_preceded_contract_freeze": True,
        "outcome_naive_preregistration_claimed": False,
        "config": asdict(config),
        "input_contract": {
            "model_inputs": [
                "operation_one_hot",
                "a_digit_one_hot",
                "b_digit_one_hot",
                "terminal_flag",
                "constant_one",
                "initial_carry_bit",
                "query_position_mask",
            ],
            "host_kpg_input_count": 0,
            "generated_prefix_input_count": 0,
            "result_tape_slots": 0,
            "generated_kv_state_bytes": 0,
            "host_alu_calls_during_model_forward": 0,
        },
        "split_commitments": {
            "train_sha256": train_hash,
            "eval_sha256": eval_hashes,
        },
        "parameter_audit": parameter_audit,
        "flop_audit": flop_audit,
        "runs": runs,
        "raw_eval_rows": raw_rows,
        "recomputed_eval": recomputed_rows,
        "median_accuracy": medians,
        "primary_width": primary_width,
        "per_seed_witness_minus_serial": per_seed_advantage,
        "dense_within_two_percentage_points": dense_within_two_points,
        "serial_wins_or_ties": serial_wins,
        "decision_inputs": decision_inputs,
        "scaled_candidate_gate_passed": scaled_candidate_gate,
        "decision": (
            "CALIBRATED_EXPLORATORY_NOMINATION_SIGNAL_ONLY"
            if scaled_candidate_gate
            else "CALIBRATED_EXPLORATORY_REJECTION"
        ),
        "gpu_launch_authorized": False,
        "architecture_promotion_authorized": False,
        "raw_eval_rows_sha256": _row_stream_hash(raw_rows),
    }


def recompute_learning_rows(
    rows: Sequence[Sequence[object]], config: LearningConfig
) -> dict[str, object]:
    expected_per_arm = (
        len(config.model_seeds) * len(config.eval_widths) * config.eval_size_per_width
    )
    expected_total = expected_per_arm * 3
    if len(rows) != expected_total:
        raise FalsifierError("learning raw-evaluation evidence is incomplete")
    counts: dict[str, dict[int, dict[int, list[int]]]] = {
        arm: {
            seed: {width: [0, 0] for width in config.eval_widths}
            for seed in config.model_seeds
        }
        for arm in ("witness", "serial", "dense")
    }
    seen: set[tuple[str, int, int, str]] = set()
    for row in rows:
        if len(row) != 6:
            raise FalsifierError("learning evidence row has the wrong arity")
        arm, seed, width, example_id, target, prediction = row
        if (
            arm not in counts
            or seed not in counts[str(arm)]
            or width not in config.eval_widths
        ):
            raise FalsifierError("learning evidence row has an unknown arm/seed/width")
        if (
            not isinstance(example_id, str)
            or not isinstance(target, int)
            or not isinstance(prediction, int)
        ):
            raise FalsifierError("learning evidence row has invalid types")
        if not 0 <= target < 12 or not 0 <= prediction < 12:
            raise FalsifierError("learning target/prediction lies outside 12 classes")
        identity = (str(arm), int(seed), int(width), example_id)
        if identity in seen:
            raise FalsifierError("learning evidence contains a duplicate case")
        seen.add(identity)
        bucket = counts[str(arm)][int(seed)][int(width)]
        bucket[0] += int(target == prediction)
        bucket[1] += 1
    for arm in counts.values():
        for seed in arm.values():
            for correct, total in seed.values():
                if total != config.eval_size_per_width or not 0 <= correct <= total:
                    raise FalsifierError("learning evidence bucket is incomplete")
    return {
        arm: {
            seed: {
                width: {
                    "correct": values[0],
                    "total": values[1],
                    "accuracy": values[0] / values[1],
                }
                for width, values in widths.items()
            }
            for seed, widths in seeds.items()
        }
        for arm, seeds in counts.items()
    }


def validate_learning_row_order_and_targets(
    rows: Sequence[Sequence[object]], config: LearningConfig
) -> None:
    """Bind every reported prediction to the frozen identity and target order."""

    _train, eval_splits = build_frozen_learning_splits(config)
    expected: list[tuple[str, int, int, str, int]] = []
    for seed in config.model_seeds:
        for arm in ("witness", "serial", "dense"):
            for width in config.eval_widths:
                expected.extend(
                    (arm, seed, width, example.example_id, example.target)
                    for example in eval_splits[width]
                )
    if len(rows) != len(expected):
        raise FalsifierError("learning row order has the wrong length")
    for observed, identity in zip(rows, expected, strict=True):
        if len(observed) != 6:
            raise FalsifierError("learning row order contains a malformed row")
        if tuple(observed[:5]) != identity:
            raise FalsifierError("learning row identity, order, or target changed")
        prediction = observed[5]
        if isinstance(prediction, bool) or not isinstance(prediction, int):
            raise FalsifierError("learning prediction must be an exact integer")
        if not 0 <= prediction < 12:
            raise FalsifierError("learning prediction lies outside 12 classes")


def recompute_learning_decision_fields(
    rows: Sequence[Sequence[object]], config: LearningConfig
) -> dict[str, object]:
    """Recompute every score and decision input from raw prediction rows."""

    validate_learning_row_order_and_targets(rows, config)
    recomputed = recompute_learning_rows(rows, config)
    accuracies: dict[str, dict[int, list[float]]] = {
        arm: {width: [] for width in config.eval_widths}
        for arm in ("witness", "serial", "dense")
    }
    for arm in ("witness", "serial", "dense"):
        for seed in config.model_seeds:
            for width in config.eval_widths:
                accuracies[arm][width].append(
                    float(recomputed[arm][seed][width]["accuracy"])
                )
    medians = {
        arm: {width: _median(values) for width, values in widths.items()}
        for arm, widths in accuracies.items()
    }
    primary_width = max(config.eval_widths)
    per_seed_advantage = [
        witness - serial
        for witness, serial in zip(
            accuracies["witness"][primary_width],
            accuracies["serial"][primary_width],
            strict=True,
        )
    ]
    witness_primary = medians["witness"][primary_width]
    serial_primary = medians["serial"][primary_width]
    dense_primary = medians["dense"][primary_width]
    dense_within_two_points = dense_primary >= witness_primary - 0.02
    serial_wins = serial_primary >= witness_primary
    parameter_audit = scaled_parameter_audit(config.hidden_dimension)
    flop_audit = scaled_flop_audit(primary_width)
    decision_inputs = {
        "witness_primary_at_least_95_percent": witness_primary >= 0.95,
        "at_least_two_seeds_15pp_over_serial": sum(
            advantage >= 0.15 for advantage in per_seed_advantage
        )
        >= 2,
        "dense_not_within_two_percentage_points": not dense_within_two_points,
        "serial_does_not_win_or_tie": not serial_wins,
        "exact_parameter_matching": bool(parameter_audit["exactly_matched"])
        and all(
            count == parameter_audit["expected_parameters"]
            for count in parameter_audit["arm_parameter_counts"].values()
        ),
        "heuristic_flop_ratio_at_most_1_01": bool(flop_audit["within_one_percent"])
        and float(flop_audit["max_min_ratio"]) <= 1.01,
    }
    candidate_gate = all(decision_inputs.values())
    return {
        "recomputed_eval": recomputed,
        "median_accuracy": medians,
        "primary_width": primary_width,
        "per_seed_witness_minus_serial": per_seed_advantage,
        "dense_within_two_percentage_points": dense_within_two_points,
        "serial_wins_or_ties": serial_wins,
        "parameter_audit": parameter_audit,
        "flop_audit": flop_audit,
        "decision_inputs": decision_inputs,
        "scaled_candidate_gate_passed": candidate_gate,
        "decision": (
            "CALIBRATED_EXPLORATORY_NOMINATION_SIGNAL_ONLY"
            if candidate_gate
            else "CALIBRATED_EXPLORATORY_REJECTION"
        ),
    }


def validate_learning_run_records(
    learning: Mapping[str, object], config: LearningConfig
) -> None:
    """Validate run ordering, hashes, losses, parameters, and row partitions."""

    runs = learning.get("runs")
    raw_rows = learning.get("raw_eval_rows")
    if not isinstance(runs, list) or not isinstance(raw_rows, list):
        raise FalsifierError("learning run or row evidence is missing")
    expected_identities = [
        (arm, seed)
        for seed in config.model_seeds
        for arm in ("witness", "serial", "dense")
    ]
    if len(runs) != len(expected_identities):
        raise FalsifierError("learning run record count changed")
    rows_per_run = len(config.eval_widths) * config.eval_size_per_width
    for index, (run, identity) in enumerate(
        zip(runs, expected_identities, strict=True)
    ):
        if not isinstance(run, Mapping):
            raise FalsifierError("learning run record is malformed")
        arm, seed = identity
        if (run.get("arm"), run.get("seed")) != identity:
            raise FalsifierError("learning run ordering or identity changed")
        if run.get("trainable_parameter_count") != SCALED_PARAMETER_COUNT:
            raise FalsifierError("learning run parameter count changed")
        torch.manual_seed(seed)
        expected_initial = _state_dict_sha256(
            ScaledRoutingArm(arm, config.hidden_dimension)  # type: ignore[arg-type]
        )
        if run.get("initial_state_sha256") != expected_initial:
            raise FalsifierError("learning initial model hash changed")
        final_hash = run.get("final_state_sha256")
        if not isinstance(final_hash, str) or len(final_hash) != 64:
            raise FalsifierError("learning final model hash is malformed")
        losses = run.get("loss_trace")
        if not isinstance(losses, list) or len(losses) != config.updates:
            raise FalsifierError("learning loss trace length changed")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in losses
        ):
            raise FalsifierError("learning loss trace contains an invalid value")
        if run.get("first_loss") != losses[0] or run.get("final_loss") != losses[-1]:
            raise FalsifierError("learning loss endpoints changed")
        if run.get("loss_trace_sha256") != _row_stream_hash(losses):
            raise FalsifierError("learning loss trace hash changed")
        partition = raw_rows[index * rows_per_run : (index + 1) * rows_per_run]
        if run.get("prediction_rows_sha256") != _row_stream_hash(partition):
            raise FalsifierError("learning run prediction-row hash changed")


def _assert_frozen_commitments(mechanics: Mapping[str, object]) -> None:
    commitments = mechanics["commitments"]
    if not isinstance(commitments, Mapping):
        raise FalsifierError("mechanics commitments are malformed")
    expected = {
        "reset_word_board_sha256": FROZEN_RESET_WORD_BOARD_SHA256,
        "local_cell_board_sha256": FROZEN_LOCAL_CELL_BOARD_SHA256,
        "toggle_board_sha256": FROZEN_TOGGLE_BOARD_SHA256,
        "intervention_board_sha256": FROZEN_INTERVENTION_BOARD_SHA256,
    }
    if dict(commitments) != expected:
        raise FalsifierError("frozen mechanics commitment mismatch")


def _hostile_equivalence_contract() -> dict[str, object]:
    return {
        "known_relation": "carry_lookahead_last_write_reset_monoid",
        "new_primitive_supported": False,
        "serial_endpoint_equivalence_required": True,
        "reject_if_dense_within_two_percentage_points": True,
        "reject_if_serial_wins_or_ties": True,
        "toggle_event_outside_candidate_algebra": True,
        "outcome_naive_preregistration_claimed": False,
    }


def _resource_contract() -> dict[str, object]:
    return {
        "h100_jobs_launched": 0,
        "host_alu_calls_during_neural_inference": 0,
        "result_tape_slots": 0,
        "generated_kv_state_bytes": 0,
        "generated_tokens_consumed_as_state": 0,
        "trusted_timestamp_claimed": False,
        "immutable_report_claimed": False,
    }


def _evidence_schema() -> dict[str, object]:
    return {
        "reset_words": ["word", "initial", "serial_trace", "witness_trace"],
        "local_cells": [
            "operation",
            "a_digit",
            "b_digit",
            "incoming",
            "oracle_status",
            "digit",
            "outgoing",
        ],
        "toggle_truth_table": [
            "candidate_alias",
            "incoming",
            "candidate",
            "recurrent",
        ],
        "learning_eval": [
            "arm",
            "seed",
            "width",
            "example_id",
            "target",
            "prediction",
        ],
    }


def _integrity_gates(
    budget: Mapping[str, object],
    mechanics: Mapping[str, object],
    learning: Mapping[str, object],
) -> dict[str, bool]:
    parameter_audit = learning["parameter_audit"]
    flop_audit = learning["flop_audit"]
    if not isinstance(parameter_audit, Mapping) or not isinstance(flop_audit, Mapping):
        raise FalsifierError("learning resource audits are malformed")
    return {
        "calibration_disclosed": learning["calibration_preceded_contract_freeze"]
        is True,
        "outcome_naive_preregistration_disclaimed": learning[
            "outcome_naive_preregistration_claimed"
        ]
        is False,
        "deployment_budget_exact": canonical_json_bytes(budget)
        == canonical_json_bytes(deployment_budget()),
        "deployment_controls_parameter_matched": budget[
            "controls_exactly_parameter_matched"
        ]
        is True,
        "scaled_controls_parameter_matched": parameter_audit["exactly_matched"] is True,
        "heuristic_flop_ratio_at_most_1_01": flop_audit["within_one_percent"] is True
        and float(flop_audit["max_min_ratio"]) <= 1.01,
        "heuristic_flops_not_misrepresented_as_measurement": flop_audit[
            "executed_graph_measurement"
        ]
        is False
        and flop_audit["hardware_flops_claimed"] is False,
        "mechanics_all_pass": mechanics["all_gates_pass"] is True,
        "calibrated_negative_decision": learning["decision"]
        == "CALIBRATED_EXPLORATORY_REJECTION",
        "learning_is_non_promotion_only": learning["gpu_launch_authorized"] is False
        and learning["architecture_promotion_authorized"] is False,
        "no_h100_launch": True,
        "no_host_alu_in_model_forward": True,
        "no_result_tape": True,
        "no_generated_kv_state": True,
    }


def _assemble_report_payload(
    evidence: Mapping[str, object],
    mechanics: Mapping[str, object],
    learning: Mapping[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    budget = deployment_budget()
    gates = _integrity_gates(budget, mechanics, learning)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": REPORT_STATUS,
        "calibration_disclosure": CALIBRATION_DISCLOSURE,
        "claim_boundary": CLAIM_BOUNDARY,
        "hostile_equivalence_audit": _hostile_equivalence_contract(),
        "deployment_budget": budget,
        "resource_contract": _resource_contract(),
        "evidence_schema": _evidence_schema(),
        "source_bindings": dict(bindings),
        "raw_mechanics_evidence": dict(evidence),
        "recomputed_mechanics": dict(mechanics),
        "scaled_learning": dict(learning),
        "gates": gates,
        "all_audit_integrity_gates_pass": all(gates.values()),
        "gpu_launch_authorized": False,
        "architecture_promotion_authorized": False,
    }


def build_report() -> dict[str, object]:
    evidence = build_mechanics_evidence()
    mechanics = recompute_mechanics_from_evidence(evidence)
    _assert_frozen_commitments(mechanics)
    learning = run_scaled_learning()
    report = _assemble_report_payload(
        evidence,
        mechanics,
        learning,
        source_bindings(),
    )
    report["report_content_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def validate_report(report: Mapping[str, object]) -> None:
    """Reconstruct the complete calibrated report from independent execution."""

    if report.get("protocol_id") != PROTOCOL_ID:
        raise FalsifierError("report protocol mismatch")
    if report.get("status") != REPORT_STATUS:
        raise FalsifierError("report calibrated status mismatch")
    content = dict(report)
    observed_hash = content.pop("report_content_sha256", None)
    expected_hash = sha256_bytes(canonical_json_bytes(content))
    if observed_hash != expected_hash:
        raise FalsifierError("report content hash mismatch")

    evidence = report.get("raw_mechanics_evidence")
    if not isinstance(evidence, Mapping):
        raise FalsifierError("report lacks raw mechanics evidence")
    recomputed_mechanics = recompute_mechanics_from_evidence(evidence)
    _assert_frozen_commitments(recomputed_mechanics)
    if canonical_json_bytes(recomputed_mechanics) != canonical_json_bytes(
        report.get("recomputed_mechanics")
    ):
        raise FalsifierError(
            "report mechanics aggregate does not independently recompute"
        )

    learning = report.get("scaled_learning")
    if not isinstance(learning, Mapping):
        raise FalsifierError("calibrated report requires full learning evidence")
    if canonical_json_bytes(learning.get("config")) != canonical_json_bytes(
        asdict(DEFAULT_LEARNING_CONFIG)
    ):
        raise FalsifierError(
            "learning hyperparameters differ from calibrated constants"
        )
    raw_rows = learning.get("raw_eval_rows")
    if not isinstance(raw_rows, list):
        raise FalsifierError("scaled learning lacks raw evaluation evidence")
    validate_learning_row_order_and_targets(raw_rows, DEFAULT_LEARNING_CONFIG)
    validate_learning_run_records(learning, DEFAULT_LEARNING_CONFIG)
    decision_fields = recompute_learning_decision_fields(
        raw_rows, DEFAULT_LEARNING_CONFIG
    )
    for field, expected in decision_fields.items():
        if canonical_json_bytes(learning.get(field)) != canonical_json_bytes(expected):
            raise FalsifierError(f"learning scientific field changed: {field}")
    if learning.get("raw_eval_rows_sha256") != _row_stream_hash(raw_rows):
        raise FalsifierError("learning raw-evidence hash mismatch")

    train, evaluations = build_frozen_learning_splits(DEFAULT_LEARNING_CONFIG)
    expected_splits = {
        "train_sha256": learning_split_sha256(train),
        "eval_sha256": {
            width: learning_split_sha256(examples)
            for width, examples in evaluations.items()
        },
    }
    if canonical_json_bytes(learning.get("split_commitments")) != canonical_json_bytes(
        expected_splits
    ):
        raise FalsifierError("learning split commitments changed")

    # A fresh deterministic execution binds predictions, every loss value,
    # final model hashes, row hashes, all summaries, and nested authorization
    # booleans.  The report's own aggregates are never used as inputs.
    independently_rerun_learning = run_scaled_learning(DEFAULT_LEARNING_CONFIG)
    if canonical_json_bytes(learning) != canonical_json_bytes(
        independently_rerun_learning
    ):
        raise FalsifierError("learning payload differs from independent rerun")

    bindings = source_bindings()
    if canonical_json_bytes(report.get("source_bindings")) != canonical_json_bytes(
        bindings
    ):
        raise FalsifierError("source/prereg/test byte bindings changed")
    expected_payload = _assemble_report_payload(
        evidence,
        recomputed_mechanics,
        independently_rerun_learning,
        bindings,
    )
    if canonical_json_bytes(content) != canonical_json_bytes(expected_payload):
        raise FalsifierError("report scientific or decision payload changed")


def report_bytes(report: Mapping[str, object]) -> bytes:
    validate_report(report)
    return canonical_json_bytes(report)


def write_report_once(destination: Path, report: Mapping[str, object]) -> str:
    payload = report_bytes(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    if stat.S_IMODE(destination.stat().st_mode) != 0o444:
        raise FalsifierError("published report mode is not 0444")
    reopened = destination.read_bytes()
    if reopened != payload:
        raise FalsifierError("published report bytes changed on reopen")
    return sha256_bytes(payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report()
    write_report_once(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
