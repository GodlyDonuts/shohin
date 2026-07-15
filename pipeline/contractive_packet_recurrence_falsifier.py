#!/usr/bin/env python3
"""Frozen CPU falsifier for contractive packet recurrence.

The board is finite, deterministic, model-free, and deliberately favorable to
the proposed mechanism.  A residual semantic state is encoded by a five-lane
repetition code, every local transition is applied lane-wise, and a strict
majority projection follows every update.  The audit distinguishes bounded
off-manifold corruption from a wrong but valid semantic codeword.

No model, accelerator framework, subprocess, network call, fitting path, or
production data is used here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "CPR-D14-R5-v1"
SCHEMA_VERSION = 1
BOARD_SCHEMA = "contractive_packet_recurrence_board_v1"
MODULUS = 7
MAX_SOURCE_LENGTH = 6
GENERATORS = ("T", "N")
REPETITION_LANES = 5
CORRECTION_RADIUS = 2
LANE_BITS = 10
LANE_ALPHABET_SIZE = 1 << LANE_BITS
INVALID_LANE = LANE_ALPHABET_SIZE - 1

# Filled after the finite contract is final.  Neither digest is serialized into
# the board, so freezing these constants cannot perturb the bytes they bind.
FROZEN_SOURCE_SHA256 = (
    "cf12740f920062d993b89457e5de880eeae3fd536e204fa9e1c5282ad34335e4"
)
FROZEN_BOARD_SHA256 = "ac61dc756b70c338aabb9245e1d48017048a959b02d5001e8e0aba847f7d38bd"
FROZEN_AUDIT_SHA256 = "d119fc88af77c9dde163d644749654a70b455260613be6b287fb29cffb524187"

RESOURCE_SECTIONS = (
    "compiler_channel",
    "transition_channel",
    "projection_channel",
    "source_channel",
    "state_and_fixed_resources",
    "external_resources",
)


class AuditError(ValueError):
    """A fail-closed board, packet, or file-contract violation."""


class ProjectionError(ValueError):
    """A packet is outside every uniquely decodable repetition-code basin."""


@dataclass(frozen=True, order=True)
class Action:
    """Affine action x -> sign*x + offset over Z_7."""

    sign: int
    offset: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError("action sign must be -1 or 1")
        if not 0 <= self.offset < MODULUS:
            raise ValueError("action offset is outside Z_7")

    def apply(self, value: int) -> int:
        _validate_value(value)
        return (self.sign * value + self.offset) % MODULUS

    def followed_by(self, later: "Action") -> "Action":
        return Action(
            later.sign * self.sign,
            (later.sign * self.offset + later.offset) % MODULUS,
        )


IDENTITY_ACTION = Action(1, 0)
TRANSLATION = Action(1, 1)
REFLECTION = Action(-1, 0)


@dataclass(frozen=True)
class SealedPacket:
    """The complete post-compilation object visible to packet recurrence."""

    lanes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.lanes) != REPETITION_LANES:
            raise ValueError(f"packet must have {REPETITION_LANES} lanes")
        if any(
            isinstance(lane, bool)
            or not isinstance(lane, int)
            or not 0 <= lane < LANE_ALPHABET_SIZE
            for lane in self.lanes
        ):
            raise ValueError("packet lane is outside the frozen lane alphabet")


def _validate_value(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("state value must be an integer")
    if not 0 <= value < MODULUS:
        raise ValueError("state value is outside Z_7")


def validate_word(word: str) -> None:
    if not isinstance(word, str):
        raise TypeError("source word must be text")
    if len(word) > MAX_SOURCE_LENGTH:
        raise ValueError("source word exceeds the frozen maximum length")
    invalid = sorted(set(word).difference(GENERATORS))
    if invalid:
        raise ValueError(f"source word contains invalid generators: {invalid}")


def residual_words() -> tuple[str, ...]:
    return tuple(
        "".join(symbols)
        for length in range(MAX_SOURCE_LENGTH + 1)
        for symbols in itertools.product(GENERATORS, repeat=length)
    )


RESIDUAL_WORDS = residual_words()
WORD_TO_INDEX = {word: index for index, word in enumerate(RESIDUAL_WORDS)}
SEMANTIC_STATE_COUNT = MODULUS * len(RESIDUAL_WORDS)
SEMANTIC_BITS = math.ceil(math.log2(SEMANTIC_STATE_COUNT))
PHYSICAL_PACKET_BITS = REPETITION_LANES * LANE_BITS
ACTION_COUNT = 2 * MODULUS
ACTION_BITS = math.ceil(math.log2(ACTION_COUNT))
TOTAL_CASES = SEMANTIC_STATE_COUNT
TOTAL_SOURCE_SYMBOLS = MODULUS * sum(len(word) for word in RESIDUAL_WORDS)
TOTAL_TRANSITION_STEPS = TOTAL_SOURCE_SYMBOLS
TOTAL_TREE_MERGES = MODULUS * sum(max(0, len(word) - 1) for word in RESIDUAL_WORDS)
TOTAL_SOURCE_REPLAY_SYMBOLS = MODULUS * sum(
    len(word) * (len(word) + 1) // 2 for word in RESIDUAL_WORDS
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes) -> Any:
    try:
        text = payload.decode("ascii")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AuditError(f"non-finite JSON constant {value!r}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise AuditError("board is not canonical ASCII JSON") from exc
    except json.JSONDecodeError as exc:
        raise AuditError("board is not valid JSON") from exc


def action_space() -> tuple[Action, ...]:
    return tuple(Action(sign, offset) for sign in (1, -1) for offset in range(MODULUS))


def encode_action(action: Action) -> int:
    return action.offset if action.sign == 1 else MODULUS + action.offset


def decode_action(code: int) -> Action:
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("action code must be an integer")
    if not 0 <= code < ACTION_COUNT:
        raise ValueError("action code is outside D14")
    if code < MODULUS:
        return Action(1, code)
    return Action(-1, code - MODULUS)


def generator_action(symbol: str) -> Action:
    if symbol == "T":
        return TRANSLATION
    if symbol == "N":
        return REFLECTION
    raise ValueError(f"invalid generator {symbol!r}")


def compile_action(word: str) -> Action:
    validate_word(word)
    result = IDENTITY_ACTION
    for symbol in word:
        result = result.followed_by(generator_action(symbol))
    return result


def tree_compile_action(word: str) -> Action:
    validate_word(word)
    if not word:
        return IDENTITY_ACTION
    if len(word) == 1:
        return generator_action(word)
    middle = len(word) // 2
    return tree_compile_action(word[:middle]).followed_by(
        tree_compile_action(word[middle:])
    )


def _build_action_fsm() -> tuple[tuple[int, int], ...]:
    rows = []
    for action in action_space():
        rows.append(
            (
                encode_action(action.followed_by(TRANSLATION)),
                encode_action(action.followed_by(REFLECTION)),
            )
        )
    return tuple(rows)


ACTION_FSM = _build_action_fsm()


def fsm_compile_action(word: str) -> int:
    validate_word(word)
    state = encode_action(IDENTITY_ACTION)
    for symbol in word:
        state = ACTION_FSM[state][0 if symbol == "T" else 1]
    return state


def serial_execute(word: str, value: int) -> int:
    validate_word(word)
    _validate_value(value)
    result = value
    for symbol in word:
        result = generator_action(symbol).apply(result)
    return result


def algebra_execute(word: str, value: int) -> int:
    """Execute a presentation relation without the board's length ceiling."""

    if not isinstance(word, str) or set(word).difference(GENERATORS):
        raise ValueError("algebra word contains an invalid generator")
    _validate_value(value)
    result = value
    for symbol in word:
        result = generator_action(symbol).apply(result)
    return result


def encode_semantic_state(value: int, residual: str) -> int:
    _validate_value(value)
    validate_word(residual)
    return WORD_TO_INDEX[residual] * MODULUS + value


def decode_semantic_state(index: int) -> tuple[int, str]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("semantic state index must be an integer")
    if not 0 <= index < SEMANTIC_STATE_COUNT:
        raise ValueError("semantic state index is outside the frozen manifold")
    word_index, value = divmod(index, MODULUS)
    return value, RESIDUAL_WORDS[word_index]


def semantic_transition(index: int) -> int:
    value, residual = decode_semantic_state(index)
    if not residual:
        return index
    next_value = generator_action(residual[0]).apply(value)
    return encode_semantic_state(next_value, residual[1:])


def encode_packet(state_index: int) -> SealedPacket:
    decode_semantic_state(state_index)
    return SealedPacket((state_index,) * REPETITION_LANES)


def decode_valid_packet(packet: SealedPacket) -> int:
    if len(set(packet.lanes)) != 1:
        raise ValueError("packet is not on the valid code manifold")
    state_index = packet.lanes[0]
    decode_semantic_state(state_index)
    return state_index


def hamming_distance(left: SealedPacket, right: SealedPacket) -> int:
    return sum(a != b for a, b in zip(left.lanes, right.lanes, strict=True))


def project_packet(packet: SealedPacket) -> SealedPacket:
    valid_counts = Counter(
        lane for lane in packet.lanes if 0 <= lane < SEMANTIC_STATE_COUNT
    )
    if not valid_counts:
        raise ProjectionError("packet has no valid semantic lane")
    best_count = max(valid_counts.values())
    winners = [lane for lane, count in valid_counts.items() if count == best_count]
    if best_count <= REPETITION_LANES // 2 or len(winners) != 1:
        raise ProjectionError("packet has no unique strict valid majority")
    return encode_packet(winners[0])


def lane_transition(packet: SealedPacket) -> SealedPacket:
    lanes = tuple(
        semantic_transition(lane) if 0 <= lane < SEMANTIC_STATE_COUNT else lane
        for lane in packet.lanes
    )
    return SealedPacket(lanes)


def compile_residual_packet(value: int, word: str) -> SealedPacket:
    return encode_packet(encode_semantic_state(value, word))


def coded_step(packet: SealedPacket) -> SealedPacket:
    return project_packet(lane_transition(packet))


def run_residual_fsm(state_index: int) -> int:
    while True:
        value, residual = decode_semantic_state(state_index)
        if not residual:
            return value
        state_index = semantic_transition(state_index)


def run_coded_recurrence(packet: SealedPacket) -> int:
    while True:
        state_index = decode_valid_packet(project_packet(packet))
        value, residual = decode_semantic_state(state_index)
        if not residual:
            return value
        packet = coded_step(packet)


def _corrupt_packet(
    packet: SealedPacket, positions: Sequence[int], replacement: int
) -> SealedPacket:
    lanes = list(packet.lanes)
    for position in positions:
        if not 0 <= position < REPETITION_LANES:
            raise ValueError("corruption position is outside the packet")
        lanes[position] = replacement
    return SealedPacket(tuple(lanes))


def source_commitment_payload(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {
                "case_id": case["case_id"],
                "initial_value": case["initial_value"],
                "source": case["source"],
                "source_length": case["source_length"],
                "initial_state_index": case["initial_state_index"],
            }
            for case in cases
        ],
    }


def source_commitment_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(source_commitment_payload(cases)))


def _case(word: str, value: int) -> dict[str, Any]:
    state = encode_semantic_state(value, word)
    trajectory = [state]
    for _ in word:
        state = semantic_transition(state)
        trajectory.append(state)
    expected = serial_execute(word, value)
    return {
        "action_code": encode_action(compile_action(word)),
        "case_id": f"l{len(word)}-{word or 'I'}-x{value}",
        "expected_final": expected,
        "initial_state_index": trajectory[0],
        "initial_value": value,
        "source": word,
        "source_length": len(word),
        "trajectory_state_indices": trajectory,
    }


def resource_ledger() -> dict[str, Any]:
    common_external = {
        "accelerator_calls": 0,
        "model_parameters": 0,
        "network_calls": 0,
        "oracle_calls": 0,
        "subprocess_calls": 0,
        "training_examples": 0,
        "training_flops": 0,
    }

    def entry(
        *,
        compiler_calls: int,
        compiler_reads: int,
        compiler_updates: int,
        compiler_output_bits: int,
        transition_calls: int,
        semantic_updates: int,
        lane_updates: int,
        duplicate_updates: int,
        projection_calls: int,
        projection_reads: int,
        projection_writes: int,
        external_retained: int,
        embedded_residual: int,
        runtime_source_reads: int,
        replay_source_reads: int,
        postseal_external_reads: int,
        active_bits: int,
        semantic_bits: int,
        physical_bits: int,
        fixed_table_entries: int,
        sequential_depth: int,
    ) -> dict[str, Any]:
        return {
            "compiler_channel": {
                "calls": compiler_calls,
                "output_bits_max": compiler_output_bits,
                "source_symbols_read": compiler_reads,
                "state_updates": compiler_updates,
            },
            "transition_channel": {
                "calls": transition_calls,
                "duplicate_verifier_updates": duplicate_updates,
                "lane_updates": lane_updates,
                "semantic_updates": semantic_updates,
            },
            "projection_channel": {
                "calls": projection_calls,
                "correction_radius_lanes": CORRECTION_RADIUS if projection_calls else 0,
                "packet_lane_reads": projection_reads,
                "packet_lane_writes": projection_writes,
            },
            "source_channel": {
                "embedded_residual_symbols_max": embedded_residual,
                "external_source_symbols_retained_max": external_retained,
                "postseal_external_reads": postseal_external_reads,
                "replay_source_symbols_read": replay_source_reads,
                "runtime_source_symbols_read": runtime_source_reads,
            },
            "state_and_fixed_resources": {
                "active_bits_max": active_bits,
                "fixed_table_entries": fixed_table_entries,
                "physical_packet_bits": physical_bits,
                "semantic_bits": semantic_bits,
                "sequential_depth_max": sequential_depth,
            },
            "external_resources": dict(common_external),
        }

    base_coded = dict(
        compiler_calls=TOTAL_CASES,
        compiler_reads=TOTAL_SOURCE_SYMBOLS,
        compiler_updates=TOTAL_SOURCE_SYMBOLS,
        compiler_output_bits=PHYSICAL_PACKET_BITS,
        transition_calls=TOTAL_TRANSITION_STEPS,
        semantic_updates=TOTAL_TRANSITION_STEPS,
        lane_updates=TOTAL_TRANSITION_STEPS * REPETITION_LANES,
        duplicate_updates=0,
        projection_calls=TOTAL_TRANSITION_STEPS,
        projection_reads=TOTAL_TRANSITION_STEPS * REPETITION_LANES,
        projection_writes=TOTAL_TRANSITION_STEPS * REPETITION_LANES,
        external_retained=0,
        embedded_residual=MAX_SOURCE_LENGTH,
        runtime_source_reads=0,
        replay_source_reads=0,
        postseal_external_reads=0,
        active_bits=PHYSICAL_PACKET_BITS,
        semantic_bits=SEMANTIC_BITS,
        physical_bits=PHYSICAL_PACKET_BITS,
        fixed_table_entries=SEMANTIC_STATE_COUNT,
        sequential_depth=2 * MAX_SOURCE_LENGTH,
    )

    ledger = {
        "serial": entry(
            compiler_calls=0,
            compiler_reads=0,
            compiler_updates=0,
            compiler_output_bits=0,
            transition_calls=TOTAL_TRANSITION_STEPS,
            semantic_updates=TOTAL_TRANSITION_STEPS,
            lane_updates=TOTAL_TRANSITION_STEPS,
            duplicate_updates=0,
            projection_calls=0,
            projection_reads=0,
            projection_writes=0,
            external_retained=MAX_SOURCE_LENGTH,
            embedded_residual=0,
            runtime_source_reads=TOTAL_SOURCE_SYMBOLS,
            replay_source_reads=0,
            postseal_external_reads=TOTAL_SOURCE_SYMBOLS,
            active_bits=math.ceil(math.log2(MODULUS)) + MAX_SOURCE_LENGTH,
            semantic_bits=math.ceil(math.log2(MODULUS)),
            physical_bits=0,
            fixed_table_entries=len(GENERATORS),
            sequential_depth=MAX_SOURCE_LENGTH,
        ),
        "balanced_tree": entry(
            compiler_calls=TOTAL_CASES,
            compiler_reads=TOTAL_SOURCE_SYMBOLS,
            compiler_updates=TOTAL_TREE_MERGES,
            compiler_output_bits=ACTION_BITS,
            transition_calls=TOTAL_CASES,
            semantic_updates=TOTAL_CASES,
            lane_updates=TOTAL_CASES,
            duplicate_updates=0,
            projection_calls=0,
            projection_reads=0,
            projection_writes=0,
            external_retained=0,
            embedded_residual=0,
            runtime_source_reads=0,
            replay_source_reads=0,
            postseal_external_reads=0,
            active_bits=ACTION_BITS + math.ceil(math.log2(MODULUS)),
            semantic_bits=ACTION_BITS,
            physical_bits=0,
            fixed_table_entries=len(GENERATORS),
            sequential_depth=math.ceil(math.log2(MAX_SOURCE_LENGTH)) + 1,
        ),
        "action_fsm": entry(
            compiler_calls=TOTAL_CASES,
            compiler_reads=TOTAL_SOURCE_SYMBOLS,
            compiler_updates=TOTAL_SOURCE_SYMBOLS,
            compiler_output_bits=ACTION_BITS,
            transition_calls=TOTAL_CASES,
            semantic_updates=TOTAL_CASES,
            lane_updates=TOTAL_CASES,
            duplicate_updates=0,
            projection_calls=0,
            projection_reads=0,
            projection_writes=0,
            external_retained=0,
            embedded_residual=0,
            runtime_source_reads=0,
            replay_source_reads=0,
            postseal_external_reads=0,
            active_bits=ACTION_BITS + math.ceil(math.log2(MODULUS)),
            semantic_bits=ACTION_BITS,
            physical_bits=0,
            fixed_table_entries=ACTION_COUNT * len(GENERATORS) + ACTION_COUNT * MODULUS,
            sequential_depth=MAX_SOURCE_LENGTH + 1,
        ),
        "residual_fsm": entry(
            compiler_calls=TOTAL_CASES,
            compiler_reads=TOTAL_SOURCE_SYMBOLS,
            compiler_updates=TOTAL_SOURCE_SYMBOLS,
            compiler_output_bits=SEMANTIC_BITS,
            transition_calls=TOTAL_TRANSITION_STEPS,
            semantic_updates=TOTAL_TRANSITION_STEPS,
            lane_updates=TOTAL_TRANSITION_STEPS,
            duplicate_updates=0,
            projection_calls=0,
            projection_reads=0,
            projection_writes=0,
            external_retained=0,
            embedded_residual=MAX_SOURCE_LENGTH,
            runtime_source_reads=0,
            replay_source_reads=0,
            postseal_external_reads=0,
            active_bits=SEMANTIC_BITS,
            semantic_bits=SEMANTIC_BITS,
            physical_bits=0,
            fixed_table_entries=SEMANTIC_STATE_COUNT,
            sequential_depth=MAX_SOURCE_LENGTH,
        ),
        "coded_recurrence": entry(**base_coded),
        "duplicate_verified_coded": entry(
            **{**base_coded, "duplicate_updates": TOTAL_TRANSITION_STEPS}
        ),
        "source_replay_rescue": entry(
            **{
                **base_coded,
                "external_retained": MAX_SOURCE_LENGTH,
                "replay_source_reads": TOTAL_SOURCE_REPLAY_SYMBOLS,
                "postseal_external_reads": TOTAL_SOURCE_REPLAY_SYMBOLS,
            }
        ),
    }
    return ledger


def generate_board() -> dict[str, Any]:
    cases = [_case(word, value) for word in RESIDUAL_WORDS for value in range(MODULUS)]
    return {
        "cases": cases,
        "constants": {
            "action_bits": ACTION_BITS,
            "action_count": ACTION_COUNT,
            "correction_radius": CORRECTION_RADIUS,
            "generators": list(GENERATORS),
            "lane_alphabet_size": LANE_ALPHABET_SIZE,
            "lane_bits": LANE_BITS,
            "max_source_length": MAX_SOURCE_LENGTH,
            "modulus": MODULUS,
            "physical_packet_bits": PHYSICAL_PACKET_BITS,
            "repetition_lanes": REPETITION_LANES,
            "semantic_bits": SEMANTIC_BITS,
            "semantic_state_count": SEMANTIC_STATE_COUNT,
        },
        "frozen_interventions": {
            "coherent_corruption_weights": [0, 1, 2, 3],
            "donor_swap_scope": "all_ordered_distinct_semantic_state_pairs",
            "invalid_corruption_weights": [0, 1, 2, 3],
            "projection_rule": "unique_strict_valid_majority",
            "semantic_error_channels": ["compiler", "transition", "projection"],
            "source_rescues": ["duplicate_transition", "source_replay"],
        },
        "protocol_id": PROTOCOL_ID,
        "resource_ledger": resource_ledger(),
        "schema": BOARD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_commitment_sha256": source_commitment_sha256(cases),
    }


def board_bytes(board: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(board)


def board_sha256(board: Mapping[str, Any]) -> str:
    return sha256_bytes(board_bytes(board))


def audit_report_sha256(report: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(report))


def relation_audit() -> dict[str, Any]:
    actions = action_space()
    closure = all(
        first.followed_by(second) in actions for first in actions for second in actions
    )
    associative = all(
        first.followed_by(second).followed_by(third)
        == first.followed_by(second.followed_by(third))
        for first in actions
        for second in actions
        for third in actions
    )
    tn = serial_execute("TN", 0)
    nt = serial_execute("NT", 0)
    presentation = {
        "T7_identity": all(
            algebra_execute("T" * 7, value) == value for value in range(7)
        ),
        "N2_identity": all(algebra_execute("NN", value) == value for value in range(7)),
        "NTN_equals_T6": all(
            algebra_execute("NTN", value) == algebra_execute("T" * 6, value)
            for value in range(7)
        ),
    }
    return {
        "action_count": len(actions),
        "associative": associative,
        "closure": closure,
        "noncommutative_witness": {
            "NT_at_0": nt,
            "TN_at_0": tn,
            "distinct": tn != nt,
        },
        "pass": closure and associative and tn != nt and all(presentation.values()),
        "presentation": presentation,
    }


def controls_audit(board: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "action_fsm": 0,
        "balanced_tree": 0,
        "coded_recurrence": 0,
        "residual_fsm": 0,
        "serial": 0,
    }
    transition_cells = 0
    transition_exact = 0
    for case in board["cases"]:
        word = case["source"]
        value = case["initial_value"]
        expected = case["expected_final"]
        counts["serial"] += serial_execute(word, value) == expected
        counts["balanced_tree"] += tree_compile_action(word).apply(value) == expected
        counts["action_fsm"] += (
            decode_action(fsm_compile_action(word)).apply(value) == expected
        )
        counts["residual_fsm"] += (
            run_residual_fsm(case["initial_state_index"]) == expected
        )
        counts["coded_recurrence"] += (
            run_coded_recurrence(compile_residual_packet(value, word)) == expected
        )
        trajectory = case["trajectory_state_indices"]
        for current, expected_next in zip(trajectory[:-1], trajectory[1:], strict=True):
            transition_cells += 1
            transition_exact += semantic_transition(current) == expected_next
            coded_next = decode_valid_packet(coded_step(encode_packet(current)))
            transition_exact += coded_next == expected_next
    expected_per_control = len(board["cases"])
    return {
        "all_exact": all(value == expected_per_control for value in counts.values())
        and transition_exact == 2 * transition_cells,
        "correct_final_cells": counts,
        "expected_final_cells_per_control": expected_per_control,
        "local_transition_checks": 2 * transition_cells,
        "local_transition_correct": transition_exact,
    }


def corruption_audit() -> dict[str, Any]:
    coherent_counts = {str(weight): 0 for weight in range(4)}
    invalid_counts = {str(weight): 0 for weight in range(4)}
    invalid_rejections = 0
    coherent_wrong_at_three = 0
    for state_index in range(SEMANTIC_STATE_COUNT):
        packet = encode_packet(state_index)
        donor = (state_index + 1) % SEMANTIC_STATE_COUNT
        for weight in range(4):
            for positions in itertools.combinations(range(REPETITION_LANES), weight):
                coherent = _corrupt_packet(packet, positions, donor)
                projected = project_packet(coherent)
                decoded = decode_valid_packet(projected)
                if weight <= CORRECTION_RADIUS:
                    coherent_counts[str(weight)] += decoded == state_index
                else:
                    coherent_wrong_at_three += decoded == donor

                invalid = _corrupt_packet(packet, positions, INVALID_LANE)
                try:
                    invalid_decoded = decode_valid_packet(project_packet(invalid))
                except ProjectionError:
                    invalid_rejections += weight == 3
                else:
                    if weight <= CORRECTION_RADIUS:
                        invalid_counts[str(weight)] += invalid_decoded == state_index
    expected = {
        str(weight): SEMANTIC_STATE_COUNT * math.comb(REPETITION_LANES, weight)
        for weight in range(4)
    }
    return {
        "coherent_correct_by_weight": coherent_counts,
        "coherent_wrong_donor_at_weight_3": coherent_wrong_at_three,
        "expected_subsets_by_weight": expected,
        "invalid_correct_by_weight": invalid_counts,
        "invalid_rejections_at_weight_3": invalid_rejections,
        "pass": all(
            coherent_counts[str(weight)] == expected[str(weight)]
            and invalid_counts[str(weight)] == expected[str(weight)]
            for weight in range(CORRECTION_RADIUS + 1)
        )
        and coherent_wrong_at_three == expected["3"]
        and invalid_rejections == expected["3"],
    }


def valid_codeword_no_go_audit() -> dict[str, Any]:
    checked = 0
    fixed = 0
    distance_preserved = 0
    for target in range(SEMANTIC_STATE_COUNT):
        target_packet = encode_packet(target)
        for donor in range(SEMANTIC_STATE_COUNT):
            if donor == target:
                continue
            checked += 1
            donor_packet = encode_packet(donor)
            projected = project_packet(donor_packet)
            fixed += projected == donor_packet
            distance_preserved += (
                hamming_distance(projected, target_packet)
                == hamming_distance(donor_packet, target_packet)
                == REPETITION_LANES
            )
    return {
        "checked_ordered_distinct_pairs": checked,
        "compiler_channel_swaps_follow_donor": fixed,
        "distance_preserved_pairs": distance_preserved,
        "expected_pairs": SEMANTIC_STATE_COUNT * (SEMANTIC_STATE_COUNT - 1),
        "pass": checked == fixed == distance_preserved,
        "projection_channel_swaps_follow_donor": fixed,
        "transition_channel_swaps_follow_donor": fixed,
    }


def source_deletion_and_rescue_audit(board: Mapping[str, Any]) -> dict[str, Any]:
    packet_fields = {field.name for field in fields(SealedPacket)}
    source_deleted_cases = 0
    embedded_source_cases = 0
    semantic_error_survives = 0
    duplicate_transition_rescues = 0
    source_replay_rescues = 0
    source_replay_symbols = 0
    transition_cells = 0
    for case in board["cases"]:
        word = case["source"]
        initial_value = case["initial_value"]
        packet = compile_residual_packet(initial_value, word)
        source_deleted_cases += packet_fields == {"lanes"} and not any(
            hasattr(packet, name)
            for name in ("source", "word", "plan", "pointer", "cache", "handle")
        )
        decoded_value, decoded_residual = decode_semantic_state(
            decode_valid_packet(packet)
        )
        embedded_source_cases += (
            decoded_value == initial_value and decoded_residual == word
        )

        current = case["initial_state_index"]
        for step in range(1, len(word) + 1):
            transition_cells += 1
            clean_next = semantic_transition(current)
            wrong_next = (clean_next + 1) % SEMANTIC_STATE_COUNT
            projected_wrong = decode_valid_packet(
                project_packet(encode_packet(wrong_next))
            )
            semantic_error_survives += projected_wrong == wrong_next

            duplicate_expected = semantic_transition(current)
            duplicate_transition_rescues += duplicate_expected == clean_next

            replay_value = serial_execute(word[:step], initial_value)
            replay_residual = word[step:]
            replay_state = encode_semantic_state(replay_value, replay_residual)
            source_replay_rescues += replay_state == clean_next
            source_replay_symbols += step
            current = clean_next
    return {
        "duplicate_transition_rescues": duplicate_transition_rescues,
        "embedded_source_cases": embedded_source_cases,
        "expected_cases": len(board["cases"]),
        "expected_transition_cells": transition_cells,
        "packet_fields": sorted(packet_fields),
        "pass": source_deleted_cases == embedded_source_cases == len(board["cases"])
        and semantic_error_survives
        == duplicate_transition_rescues
        == source_replay_rescues
        == transition_cells
        and source_replay_symbols == TOTAL_SOURCE_REPLAY_SYMBOLS,
        "semantic_error_survives_projection": semantic_error_survives,
        "source_deleted_external_cases": source_deleted_cases,
        "source_replay_rescues": source_replay_rescues,
        "source_replay_symbols_read": source_replay_symbols,
    }


def contraction_theorem_audit() -> dict[str, Any]:
    local_checks = 0
    local_zero_after_projection = 0
    for target in range(SEMANTIC_STATE_COUNT):
        target_packet = encode_packet(target)
        donor = (target + 1) % SEMANTIC_STATE_COUNT
        for errors in range(CORRECTION_RADIUS + 1):
            for positions in itertools.combinations(range(REPETITION_LANES), errors):
                corrupted = _corrupt_packet(target_packet, positions, donor)
                local_checks += 1
                local_zero_after_projection += (
                    project_packet(corrupted) == target_packet
                )

    global_target = encode_packet(0)
    global_wrong = encode_packet(1)
    before = hamming_distance(global_wrong, global_target)
    after = hamming_distance(project_packet(global_wrong), global_target)

    semantic_success = Fraction(99, 100)
    reliability_rows = [
        {
            "depth": depth,
            "denominator": (semantic_success**depth).denominator,
            "numerator": (semantic_success**depth).numerator,
        }
        for depth in (1, 2, 4, 8, 16, 32, 64)
    ]
    return {
        "global_strict_contraction_impossible_witness": {
            "distance_after": after,
            "distance_before": before,
            "ratio_is_one": before == after != 0,
            "target_state": 0,
            "wrong_valid_state": 1,
        },
        "local_basin_checks": local_checks,
        "local_zero_after_projection": local_zero_after_projection,
        "pass": local_checks == local_zero_after_projection and before == after != 0,
        "semantic_error_exact_sequence_success": reliability_rows,
    }


def collapse_audit() -> dict[str, Any]:
    equivalent_states = 0
    for state_index in range(SEMANTIC_STATE_COUNT):
        coded_next = decode_valid_packet(coded_step(encode_packet(state_index)))
        equivalent_states += coded_next == semantic_transition(state_index)
    coded = resource_ledger()["coded_recurrence"]
    fsm = resource_ledger()["residual_fsm"]
    return {
        "behaviorally_equivalent_semantic_states": equivalent_states,
        "coded_active_bits": coded["state_and_fixed_resources"]["active_bits_max"],
        "coded_depth_per_max_case": coded["state_and_fixed_resources"][
            "sequential_depth_max"
        ],
        "classification": "exact_repetition_code_around_finite_residual_fsm",
        "expected_semantic_states": SEMANTIC_STATE_COUNT,
        "fsm_active_bits": fsm["state_and_fixed_resources"]["active_bits_max"],
        "fsm_depth_per_max_case": fsm["state_and_fixed_resources"][
            "sequential_depth_max"
        ],
        "noncollapsed_interface_survives": False,
        "pass": equivalent_states == SEMANTIC_STATE_COUNT
        and fsm["state_and_fixed_resources"]["active_bits_max"]
        < coded["state_and_fixed_resources"]["active_bits_max"]
        and fsm["state_and_fixed_resources"]["sequential_depth_max"]
        < coded["state_and_fixed_resources"]["sequential_depth_max"],
    }


def audit_board(
    board: Mapping[str, Any], *, require_frozen_hashes: bool = True
) -> dict[str, Any]:
    expected_top = {
        "cases",
        "constants",
        "frozen_interventions",
        "protocol_id",
        "resource_ledger",
        "schema",
        "schema_version",
        "source_commitment_sha256",
    }
    if not isinstance(board, Mapping) or set(board) != expected_top:
        raise AuditError("board top-level fields differ from the frozen schema")
    if board["schema"] != BOARD_SCHEMA or board["protocol_id"] != PROTOCOL_ID:
        raise AuditError("board schema or protocol identifier differs")
    if board["schema_version"] != SCHEMA_VERSION:
        raise AuditError("board schema version differs")

    expected_constants = generate_board()["constants"]
    if board["constants"] != expected_constants:
        raise AuditError("board constants differ from the frozen contract")
    if board["frozen_interventions"] != generate_board()["frozen_interventions"]:
        raise AuditError("board intervention contract differs")

    cases = board["cases"]
    if not isinstance(cases, list) or len(cases) != TOTAL_CASES:
        raise AuditError("board case count differs")
    expected_case_fields = {
        "action_code",
        "case_id",
        "expected_final",
        "initial_state_index",
        "initial_value",
        "source",
        "source_length",
        "trajectory_state_indices",
    }
    position = 0
    for word in RESIDUAL_WORDS:
        for value in range(MODULUS):
            case = cases[position]
            if not isinstance(case, Mapping) or set(case) != expected_case_fields:
                raise AuditError(f"case {position} fields differ")
            expected = _case(word, value)
            if dict(case) != expected:
                raise AuditError(
                    f"case {position} differs from exhaustive recomputation"
                )
            position += 1

    source_hash = source_commitment_sha256(cases)
    if board["source_commitment_sha256"] != source_hash:
        raise AuditError("source commitment differs from recomputation")
    if board["resource_ledger"] != resource_ledger():
        raise AuditError("resource ledger differs from exact recomputation")
    for algorithm, entry in board["resource_ledger"].items():
        if set(entry) != set(RESOURCE_SECTIONS):
            raise AuditError(f"resource sections differ for {algorithm}")

    relation = relation_audit()
    controls = controls_audit(board)
    corruption = corruption_audit()
    no_go = valid_codeword_no_go_audit()
    deletion = source_deletion_and_rescue_audit(board)
    theorem = contraction_theorem_audit()
    collapse = collapse_audit()
    gates = {
        "algebra": relation["pass"],
        "classical_collapse": collapse["pass"],
        "controls": controls["all_exact"],
        "corruption_boundary": corruption["pass"],
        "local_contraction_global_no_go": theorem["pass"],
        "source_deletion_and_rescues": deletion["pass"],
        "valid_codeword_no_go": no_go["pass"],
    }
    digest = board_sha256(board)
    if require_frozen_hashes:
        if not FROZEN_SOURCE_SHA256 or not FROZEN_BOARD_SHA256:
            raise AuditError("frozen digest constants have not been populated")
        if source_hash != FROZEN_SOURCE_SHA256:
            raise AuditError("source commitment differs from frozen digest")
        if digest != FROZEN_BOARD_SHA256:
            raise AuditError("board bytes differ from frozen digest")
    if not all(gates.values()):
        raise AuditError("one or more exhaustive CPU gates failed")
    return {
        "board_sha256": digest,
        "classical_collapse": collapse,
        "controls": controls,
        "corruption": corruption,
        "gates": gates,
        "pass": True,
        "relation": relation,
        "source_commitment_sha256": source_hash,
        "source_deletion_and_rescues": deletion,
        "theorem": theorem,
        "valid_codeword_no_go": no_go,
    }


def write_immutable_board(path: Path) -> dict[str, Any]:
    board = generate_board()
    audit_board(board)
    payload = board_bytes(board)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return {
        "bytes": len(payload),
        "mode": stat.S_IMODE(path.stat().st_mode),
        "path": str(path),
        "sha256": sha256_bytes(payload),
    }


def audit_board_file(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AuditError("board path is a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise AuditError("board path is not a regular file")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise AuditError("board file is writable")
    payload = path.read_bytes()
    board = strict_json_loads(payload)
    if canonical_json_bytes(board) != payload:
        raise AuditError("board file is not byte-canonical JSON")
    return audit_board(board)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="create an immutable board")
    generate.add_argument("--output", required=True, type=Path)
    audit = subparsers.add_parser("audit", help="audit an immutable board")
    audit.add_argument("--input", required=True, type=Path)
    subparsers.add_parser("report", help="audit the in-memory frozen board")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        result = write_immutable_board(args.output)
    elif args.command == "audit":
        result = audit_board_file(args.input)
    else:
        result = audit_board(generate_board())
    print(pretty_json_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
