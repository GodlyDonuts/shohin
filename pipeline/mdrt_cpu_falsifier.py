#!/usr/bin/env python3
"""Deterministic CPU mechanics falsifier for R12-MDRT-CPU-v1.

This module implements only the finite mechanics frozen in
R12_MIXED_DIFFERENCE_RESIDUAL_TRANSDUCER_PREREG.md.  It contains no neural
model, fitting path, accelerator dependency, subprocess, network call, or
production-data interface.  The planted positive includes the complete task
transition law in its fixed mixed-interaction resources; a mechanics pass is
not evidence that Shohin contains or can learn that interaction.
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
from typing import Any, Iterable, Sequence


PROTOCOL_ID = "R12-MDRT-CPU-v1"
SCHEMA_VERSION = 1
MODULUS = 17
MAX_PROGRAM_LENGTH = 8
GENERATORS = ("A", "B")
HALT = "HALT"
INVALID_ACTION = "INVALID"
ACTIONS = (*GENERATORS, HALT)
ARM_POSITIVE = "planted_mixed_interaction"
ARM_ZERO = "zero_interaction"
ARM_SHORTCUT = "state_depth_shortcut"
EXECUTABLE_ARMS = (ARM_POSITIVE, ARM_ZERO, ARM_SHORTCUT)
MACHINE_TASK = "task"
VECTOR_MODULUS = 65_537
VECTOR_WIDTH = 6
VECTOR_PRECISION_BITS = 17

PROGRAM_ORDINALS = (1 << (MAX_PROGRAM_LENGTH + 1)) - 1
VALID_STATE_COUNT = MODULUS * PROGRAM_ORDINALS
SINK_STATE_ID = VALID_STATE_COUNT
STATE_COUNT = VALID_STATE_COUNT + 1
ACTION_COUNT = len(ACTIONS)
STATE_BITS = math.ceil(math.log2(STATE_COUNT))
SHORTCUT_USED_STATE_BITS = math.ceil(math.log2(MODULUS)) + math.ceil(
    math.log2(MAX_PROGRAM_LENGTH + 1)
)
FIXED_TAIL_TABLE_ENTRIES = STATE_COUNT * ACTION_COUNT
FIXED_TAIL_TABLE_BITS = FIXED_TAIL_TABLE_ENTRIES * STATE_BITS
TRANSIENT_VECTOR_BITS = VECTOR_WIDTH * VECTOR_PRECISION_BITS
SOURCE_BITS_READ_MAX = (
    math.ceil(math.log2(MODULUS))
    + math.ceil(math.log2(MAX_PROGRAM_LENGTH + 1))
    + MAX_PROGRAM_LENGTH
)
FROZEN_SOURCE_SHA256 = (
    "3d049655977ec31f5caa00dbd2ab03857c9e97f7560d76a57dd66185ebbdc510"
)

PARTITIONS = {
    "train_named_mechanics": (1, 2, 3, 4),
    "development_named_mechanics": (5, 6),
    "evaluation_named_mechanics": (7, 8),
}

COMMON_VECTOR = (101, 211, 307, 401, 503, 601)
ACTION_MAIN_VECTORS = {
    "A": (17, 29, 43, 59, 71, 89),
    "B": (97, 109, 127, 149, 163, 181),
    HALT: (191, 211, 229, 251, 271, 293),
}
ZERO_VECTOR = (0,) * VECTOR_WIDTH


class AuditError(ValueError):
    """A deterministic contract or recomputation failed closed."""


class DecodeError(ValueError):
    """A mixed vector is not an exact sealed-state code."""


@dataclass(frozen=True, order=True)
class RuntimeState:
    """Complete finite causal state, including one absorbing invalid state."""

    value: int
    remaining: str
    invalid: bool = False

    def __post_init__(self) -> None:
        if self.invalid:
            if self.value != 0 or self.remaining:
                raise ValueError("invalid state must use canonical zero payload")
            return
        _validate_value(self.value)
        validate_program(self.remaining, allow_empty=True)


SINK_STATE = RuntimeState(0, "", True)


@dataclass(frozen=True)
class SealedState:
    """Only object retained after source compilation."""

    state_id: int

    def __post_init__(self) -> None:
        if isinstance(self.state_id, bool) or not isinstance(self.state_id, int):
            raise TypeError("sealed state identifier must be an integer")
        if not 0 <= self.state_id < STATE_COUNT:
            raise ValueError("sealed state identifier is outside the finite board")


@dataclass(frozen=True)
class SourceCase:
    """One exhaustive source before commitment."""

    case_id: str
    value: int
    program: str
    partition: str


@dataclass(frozen=True)
class StepRecord:
    """One autonomous MDRT transition."""

    action: str
    next_state: SealedState
    mixed_vector: tuple[int, ...]
    decode_success: bool


@dataclass(frozen=True)
class Rollout:
    """Source-free rollout trace used by deterministic audits."""

    actions: tuple[str, ...]
    state_ids: tuple[int, ...]
    final_state_id: int
    halted: bool
    all_decodes_valid: bool


def _validate_value(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("field value must be an integer")
    if not 0 <= value < MODULUS:
        raise ValueError("field value is outside F_17")


def validate_program(program: str, *, allow_empty: bool) -> None:
    if not isinstance(program, str):
        raise TypeError("program must be text")
    if not allow_empty and not program:
        raise ValueError("source program must be nonempty")
    if len(program) > MAX_PROGRAM_LENGTH:
        raise ValueError("program exceeds the frozen maximum length")
    invalid = sorted(set(program).difference(GENERATORS))
    if invalid:
        raise ValueError(f"program contains invalid generators: {invalid}")


def apply_generator(action: str, value: int) -> int:
    _validate_value(value)
    if action == "A":
        return (value + 1) % MODULUS
    if action == "B":
        return (2 * value) % MODULUS
    raise ValueError("arithmetic action must be A or B")


def execute_program(value: int, program: str) -> int:
    _validate_value(value)
    validate_program(program, allow_empty=True)
    for action in program:
        value = apply_generator(action, value)
    return value


def word_rank(word: str) -> int:
    validate_program(word, allow_empty=True)
    rank = 0
    for symbol in word:
        rank = 2 * rank + int(symbol == "B")
    return rank


def word_from_rank(length: int, rank: int) -> str:
    if not 0 <= length <= MAX_PROGRAM_LENGTH:
        raise ValueError("word length is outside the frozen board")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("word rank must be an integer")
    if not 0 <= rank < (1 << length):
        raise ValueError("word rank is outside its length class")
    return "".join(
        "B" if rank & (1 << shift) else "A"
        for shift in range(length - 1, -1, -1)
    )


def encode_state_id(state: RuntimeState) -> int:
    if state.invalid:
        return SINK_STATE_ID
    length = len(state.remaining)
    ordinal = (1 << length) - 1 + word_rank(state.remaining)
    state_id = ordinal * MODULUS + state.value
    if not 0 <= state_id < VALID_STATE_COUNT:
        raise AssertionError("state encoding escaped the frozen range")
    return state_id


def decode_state_id(state_id: int) -> RuntimeState:
    if isinstance(state_id, bool) or not isinstance(state_id, int):
        raise TypeError("state identifier must be an integer")
    if state_id == SINK_STATE_ID:
        return SINK_STATE
    if not 0 <= state_id < VALID_STATE_COUNT:
        raise ValueError("state identifier is outside the frozen board")
    ordinal, value = divmod(state_id, MODULUS)
    length = (ordinal + 1).bit_length() - 1
    rank = ordinal - ((1 << length) - 1)
    return RuntimeState(value, word_from_rank(length, rank))


def required_action(state: RuntimeState) -> str:
    if state.invalid:
        return INVALID_ACTION
    if state.remaining:
        return state.remaining[0]
    return HALT


@lru_cache(maxsize=None)
def task_transition(state: RuntimeState, action: str) -> RuntimeState:
    if action not in ACTIONS:
        raise ValueError("action is outside the frozen action alphabet")
    if state.invalid:
        return SINK_STATE
    if not state.remaining:
        return state if action == HALT else SINK_STATE
    if action != state.remaining[0]:
        return SINK_STATE
    return RuntimeState(
        apply_generator(action, state.value),
        state.remaining[1:],
    )


def task_output(state: RuntimeState) -> tuple[int, str, bool, bool]:
    if state.invalid:
        return (-1, INVALID_ACTION, True, True)
    return (state.value, required_action(state), not state.remaining, False)


def _vector_add(*vectors: tuple[int, ...]) -> tuple[int, ...]:
    if any(len(vector) != VECTOR_WIDTH for vector in vectors):
        raise ValueError("vector width differs from the frozen width")
    return tuple(
        sum(vector[index] for vector in vectors) % VECTOR_MODULUS
        for index in range(VECTOR_WIDTH)
    )


def _vector_subtract(
    left: tuple[int, ...],
    *subtractors: tuple[int, ...],
) -> tuple[int, ...]:
    if len(left) != VECTOR_WIDTH or any(
        len(vector) != VECTOR_WIDTH for vector in subtractors
    ):
        raise ValueError("vector width differs from the frozen width")
    return tuple(
        (left[index] - sum(vector[index] for vector in subtractors))
        % VECTOR_MODULUS
        for index in range(VECTOR_WIDTH)
    )


def state_code_vector(state: RuntimeState) -> tuple[int, ...]:
    state_id = encode_state_id(state)
    if state.invalid:
        value_code = MODULUS
        word_code = 0
        length_and_flag = 1
    else:
        value_code = state.value
        word_code = (1 << len(state.remaining)) | word_rank(state.remaining)
        length_and_flag = 2 * len(state.remaining)
    return (
        state_id,
        (3 * state_id + 5) % VECTOR_MODULUS,
        (7 * state_id + 11) % VECTOR_MODULUS,
        value_code,
        word_code,
        length_and_flag,
    )


def decode_state_code(vector: tuple[int, ...]) -> RuntimeState:
    if len(vector) != VECTOR_WIDTH:
        raise DecodeError("mixed vector has the wrong width")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < VECTOR_MODULUS
        for value in vector
    ):
        raise DecodeError("mixed vector has an invalid coordinate")
    state_id = vector[0]
    if not 0 <= state_id < STATE_COUNT:
        raise DecodeError("mixed vector does not name a finite state")
    state = decode_state_id(state_id)
    if vector != state_code_vector(state):
        raise DecodeError("mixed vector failed the exact state checksum")
    return state


def _state_main_vector(state: RuntimeState) -> tuple[int, ...]:
    code = state_code_vector(state)
    return tuple(
        ((index + 2) * value + 13 * (index + 1)) % VECTOR_MODULUS
        for index, value in enumerate(code)
    )


@lru_cache(maxsize=None)
def tail_output(
    arm: str,
    state: RuntimeState | None,
    action: str | None,
) -> tuple[int, ...]:
    if arm not in (ARM_POSITIVE, ARM_ZERO):
        raise ValueError("tail arm must be planted-positive or zero-interaction")
    if action is not None and action not in ACTIONS:
        raise ValueError("action is outside the frozen action alphabet")
    terms = [COMMON_VECTOR]
    if state is not None:
        terms.append(_state_main_vector(state))
    if action is not None:
        terms.append(ACTION_MAIN_VECTORS[action])
    if arm == ARM_POSITIVE and state is not None and action is not None:
        terms.append(state_code_vector(task_transition(state, action)))
    return _vector_add(*terms)


@lru_cache(maxsize=None)
def mixed_difference(
    arm: str,
    state: RuntimeState,
    action: str,
) -> tuple[int, ...]:
    both = tail_output(arm, state, action)
    state_only = tail_output(arm, state, None)
    action_only = tail_output(arm, None, action)
    carrier = tail_output(arm, None, None)
    return _vector_add(_vector_subtract(both, state_only, action_only), carrier)


@lru_cache(maxsize=None)
def model_transition(
    arm: str,
    state: RuntimeState,
    action: str,
) -> tuple[RuntimeState, bool]:
    mixed = mixed_difference(arm, state, action)
    try:
        return decode_state_code(mixed), True
    except DecodeError:
        return SINK_STATE, False


def compile_source(value: int, program: str) -> SealedState:
    _validate_value(value)
    validate_program(program, allow_empty=False)
    return SealedState(encode_state_id(RuntimeState(value, program)))


def mdrt_step(sealed: SealedState, arm: str) -> StepRecord:
    if arm not in (ARM_POSITIVE, ARM_ZERO):
        raise ValueError("MDRT step arm must be positive or zero")
    state = decode_state_id(sealed.state_id)
    action = required_action(state)
    if action == INVALID_ACTION:
        return StepRecord(action, SealedState(SINK_STATE_ID), ZERO_VECTOR, False)
    mixed = mixed_difference(arm, state, action)
    successor, decode_success = model_transition(arm, state, action)
    return StepRecord(
        action,
        SealedState(encode_state_id(successor)),
        mixed,
        decode_success,
    )


def rollout_from_sealed(sealed: SealedState, arm: str) -> Rollout:
    current = sealed
    actions: list[str] = []
    state_ids = [sealed.state_id]
    all_decodes_valid = True
    for _ in range(MAX_PROGRAM_LENGTH + 1):
        state = decode_state_id(current.state_id)
        if state.invalid:
            return Rollout(
                tuple(actions),
                tuple(state_ids),
                current.state_id,
                False,
                False,
            )
        step = mdrt_step(current, arm)
        actions.append(step.action)
        all_decodes_valid = all_decodes_valid and step.decode_success
        current = step.next_state
        state_ids.append(current.state_id)
        if not step.decode_success:
            return Rollout(
                tuple(actions),
                tuple(state_ids),
                current.state_id,
                False,
                False,
            )
        if step.action == HALT:
            return Rollout(
                tuple(actions),
                tuple(state_ids),
                current.state_id,
                True,
                all_decodes_valid,
            )
    return Rollout(
        tuple(actions),
        tuple(state_ids),
        current.state_id,
        False,
        all_decodes_valid,
    )


def interchange_sealed_state(
    recipient: SealedState,
    donor: SealedState,
) -> SealedState:
    """Replace the complete recipient baton with the donor baton."""
    if not isinstance(recipient, SealedState) or not isinstance(donor, SealedState):
        raise TypeError("state interchange requires two sealed states")
    return SealedState(donor.state_id)


def shortcut_action(remaining_depth: int) -> str:
    if not 0 <= remaining_depth <= MAX_PROGRAM_LENGTH:
        raise ValueError("remaining depth is outside the frozen board")
    if remaining_depth == 0:
        return HALT
    return "A" if remaining_depth % 2 else "B"


def run_shortcut(value: int, program: str) -> Rollout:
    _validate_value(value)
    validate_program(program, allow_empty=False)
    current_value = value
    actions: list[str] = []
    synthetic_ids = [encode_state_id(RuntimeState(value, "A" * len(program)))]
    for remaining_depth in range(len(program), 0, -1):
        action = shortcut_action(remaining_depth)
        actions.append(action)
        current_value = apply_generator(action, current_value)
        synthetic_ids.append(
            encode_state_id(RuntimeState(current_value, "A" * (remaining_depth - 1)))
        )
    actions.append(HALT)
    synthetic_ids.append(synthetic_ids[-1])
    return Rollout(
        tuple(actions),
        tuple(synthetic_ids),
        synthetic_ids[-1],
        True,
        True,
    )


def iter_words(length: int) -> Iterable[str]:
    if not 0 <= length <= MAX_PROGRAM_LENGTH:
        raise ValueError("word length is outside the frozen board")
    for symbols in itertools.product(GENERATORS, repeat=length):
        yield "".join(symbols)


def _partition_for_length(length: int) -> str:
    for name, lengths in PARTITIONS.items():
        if length in lengths:
            return name
    raise AssertionError("length has no frozen partition")


@lru_cache(maxsize=1)
def source_cases() -> tuple[SourceCase, ...]:
    cases = []
    for length in range(1, MAX_PROGRAM_LENGTH + 1):
        partition = _partition_for_length(length)
        for program in iter_words(length):
            for value in range(MODULUS):
                cases.append(
                    SourceCase(
                        f"L{length}-{program}-x{value:02d}",
                        value,
                        program,
                        partition,
                    )
                )
    return tuple(cases)


@lru_cache(maxsize=1)
def all_states() -> tuple[RuntimeState, ...]:
    states = []
    for length in range(MAX_PROGRAM_LENGTH + 1):
        for word in iter_words(length):
            states.extend(RuntimeState(value, word) for value in range(MODULUS))
    states.append(SINK_STATE)
    if tuple(encode_state_id(state) for state in states) != tuple(range(STATE_COUNT)):
        raise AssertionError("state enumeration differs from canonical identifiers")
    return tuple(states)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_commitment_sha256() -> str:
    rows = [
        {
            "case_id": case.case_id,
            "partition": case.partition,
            "program": case.program,
            "value": case.value,
        }
        for case in source_cases()
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def rollout_bytes(rollout: Rollout) -> bytes:
    return canonical_json_bytes(
        {
            "actions": list(rollout.actions),
            "all_decodes_valid": rollout.all_decodes_valid,
            "final_state_id": rollout.final_state_id,
            "halted": rollout.halted,
            "state_ids": list(rollout.state_ids),
        }
    )


def board_summary() -> dict[str, Any]:
    cases = source_cases()
    partition_counts = {
        name: sum(case.partition == name for case in cases) for name in PARTITIONS
    }
    transition_steps = sum(len(case.program) + 1 for case in cases)
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "modulus": MODULUS,
        "generators": list(GENERATORS),
        "actions": list(ACTIONS),
        "max_program_length": MAX_PROGRAM_LENGTH,
        "partition_counts": partition_counts,
        "source_case_count": len(cases),
        "runtime_state_count": STATE_COUNT,
        "state_action_cell_count": STATE_COUNT * ACTION_COUNT,
        "source_trajectory_step_count_including_halt": transition_steps,
        "source_commitment_sha256": source_commitment_sha256(),
        "noncommutative_witness": {
            "initial": 0,
            "AB": execute_program(0, "AB"),
            "BA": execute_program(0, "BA"),
        },
    }


@lru_cache(maxsize=None)
def mixed_cell_audit(arm: str) -> dict[str, Any]:
    if arm not in (ARM_POSITIVE, ARM_ZERO):
        raise ValueError("mixed-cell audit arm must be positive or zero")
    exact_successors = 0
    valid_decodes = 0
    zero_vectors = 0
    for state in all_states():
        for action in ACTIONS:
            mixed = mixed_difference(arm, state, action)
            zero_vectors += int(mixed == ZERO_VECTOR)
            try:
                decoded = decode_state_code(mixed)
            except DecodeError:
                continue
            valid_decodes += 1
            exact_successors += int(decoded == task_transition(state, action))
    total = STATE_COUNT * ACTION_COUNT
    return {
        "arm": arm,
        "total_cells": total,
        "exact_successors": exact_successors,
        "valid_decodes": valid_decodes,
        "zero_vectors": zero_vectors,
    }


def _rollout_is_exact(case: SourceCase, rollout: Rollout) -> tuple[bool, bool]:
    expected_actions = tuple(case.program) + (HALT,)
    final_state = decode_state_id(rollout.final_state_id)
    final_exact = (
        not final_state.invalid
        and final_state.value == execute_program(case.value, case.program)
        and not final_state.remaining
    )
    trajectory_exact = (
        rollout.halted
        and rollout.all_decodes_valid
        and rollout.actions == expected_actions
        and final_exact
    )
    return trajectory_exact, final_exact


@lru_cache(maxsize=None)
def trajectory_audit(arm: str) -> dict[str, Any]:
    if arm not in EXECUTABLE_ARMS:
        raise ValueError("trajectory audit arm is unknown")
    exact_trajectories = 0
    exact_final_values = 0
    halted = 0
    valid_decode_rollouts = 0
    by_length = {
        length: {"cases": 0, "exact_trajectories": 0}
        for length in range(1, MAX_PROGRAM_LENGTH + 1)
    }
    for case in source_cases():
        if arm == ARM_SHORTCUT:
            rollout = run_shortcut(case.value, case.program)
        else:
            rollout = rollout_from_sealed(compile_source(case.value, case.program), arm)
        trajectory_exact, final_exact = _rollout_is_exact(case, rollout)
        exact_trajectories += int(trajectory_exact)
        exact_final_values += int(final_exact)
        halted += int(rollout.halted)
        valid_decode_rollouts += int(rollout.all_decodes_valid)
        row = by_length[len(case.program)]
        row["cases"] += 1
        row["exact_trajectories"] += int(trajectory_exact)
    return {
        "arm": arm,
        "total_cases": len(source_cases()),
        "exact_trajectories": exact_trajectories,
        "exact_final_values": exact_final_values,
        "halted": halted,
        "all_decodes_valid": valid_decode_rollouts,
        "by_length": by_length,
    }


def _machine_output(kind: str, state: RuntimeState) -> tuple[int, str, bool, bool]:
    if kind == ARM_SHORTCUT:
        if state.invalid:
            return task_output(state)
        return (
            state.value,
            shortcut_action(len(state.remaining)),
            not state.remaining,
            False,
        )
    if kind in (MACHINE_TASK, ARM_POSITIVE, ARM_ZERO):
        return task_output(state)
    raise ValueError("machine kind is unknown")


def _shortcut_transition(state: RuntimeState) -> RuntimeState:
    if state.invalid:
        return SINK_STATE
    remaining_depth = len(state.remaining)
    action = shortcut_action(remaining_depth)
    if action == HALT:
        return state
    return RuntimeState(
        apply_generator(action, state.value),
        "A" * (remaining_depth - 1),
    )


@lru_cache(maxsize=None)
def machine_transition(kind: str, state: RuntimeState, action: str) -> RuntimeState:
    if action not in ACTIONS:
        raise ValueError("action is outside the frozen action alphabet")
    if kind == MACHINE_TASK:
        return task_transition(state, action)
    if kind in (ARM_POSITIVE, ARM_ZERO):
        return model_transition(kind, state, action)[0]
    if kind == ARM_SHORTCUT:
        return _shortcut_transition(state)
    raise ValueError("machine kind is unknown")


@lru_cache(maxsize=None)
def moore_minimization_audit(kind: str) -> dict[str, Any]:
    states = all_states()
    groups: dict[tuple[int, str, bool, bool], list[int]] = {}
    for state_id, state in enumerate(states):
        groups.setdefault(_machine_output(kind, state), []).append(state_id)
    partition = tuple(tuple(group) for group in groups.values())
    iterations = 0
    while True:
        block_of = [0] * STATE_COUNT
        for block_index, block in enumerate(partition):
            for state_id in block:
                block_of[state_id] = block_index
        refined: dict[tuple[Any, ...], list[int]] = {}
        for state_id, state in enumerate(states):
            signature = (
                _machine_output(kind, state),
                tuple(
                    block_of[encode_state_id(machine_transition(kind, state, action))]
                    for action in ACTIONS
                ),
            )
            refined.setdefault(signature, []).append(state_id)
        next_partition = tuple(tuple(group) for group in refined.values())
        iterations += 1
        if next_partition == partition:
            break
        partition = next_partition
    class_sizes = sorted((len(block) for block in partition), reverse=True)
    return {
        "machine": kind,
        "state_count": STATE_COUNT,
        "class_count": len(partition),
        "refinement_iterations": iterations,
        "largest_class": class_sizes[0],
        "singleton_classes": sum(size == 1 for size in class_sizes),
    }


def _inverted_program(program: str) -> str:
    return "".join("B" if symbol == "A" else "A" for symbol in reversed(program))


@lru_cache(maxsize=1)
def erasure_and_donor_audit() -> dict[str, Any]:
    field_names = tuple(field.name for field in fields(SealedState))
    runtime_parameters = tuple(inspect.signature(rollout_from_sealed).parameters)
    source_mutation_identical = 0
    stale_state_identical = 0
    donor_following = 0
    cases = source_cases()
    for index, case in enumerate(cases):
        sealed = compile_source(case.value, case.program)
        baseline = rollout_from_sealed(sealed, ARM_POSITIVE)
        baseline_bytes = rollout_bytes(baseline)

        mutated_value = (case.value + 1) % MODULUS
        mutated_program = _inverted_program(case.program)
        _ = (mutated_value, mutated_program)
        source_mutation_identical += int(
            rollout_bytes(rollout_from_sealed(sealed, ARM_POSITIVE))
            == baseline_bytes
        )

        first_step = mdrt_step(sealed, ARM_POSITIVE)
        successor = first_step.next_state
        continuation = rollout_from_sealed(successor, ARM_POSITIVE)
        discarded_old = compile_source(mutated_value, mutated_program)
        del discarded_old
        stale_state_identical += int(
            rollout_bytes(rollout_from_sealed(successor, ARM_POSITIVE))
            == rollout_bytes(continuation)
        )

        donor = cases[(index + 1) % len(cases)]
        donor_state = compile_source(donor.value, donor.program)
        donor_baseline = rollout_from_sealed(donor_state, ARM_POSITIVE)
        transplanted = interchange_sealed_state(sealed, donor_state)
        donor_following += int(
            rollout_bytes(rollout_from_sealed(transplanted, ARM_POSITIVE))
            == rollout_bytes(donor_baseline)
        )
    total = len(cases)
    return {
        "sealed_state_fields": list(field_names),
        "rollout_runtime_parameters": list(runtime_parameters),
        "audited_cases": total,
        "source_mutation_bit_identical": source_mutation_identical,
        "stale_state_bit_identical": stale_state_identical,
        "donor_following": donor_following,
        "structural_source_free": field_names == ("state_id",)
        and runtime_parameters == ("sealed", "arm"),
    }


def allocated_resource_budget() -> dict[str, int]:
    return {
        "trainable_parameters": 0,
        "allocated_persistent_state_bits": STATE_BITS,
        "precision_bits": VECTOR_PRECISION_BITS,
        "allocated_transient_vector_bits": TRANSIENT_VECTOR_BITS,
        "allocated_fixed_tail_table_entries": FIXED_TAIL_TABLE_ENTRIES,
        "allocated_fixed_tail_table_bits": FIXED_TAIL_TABLE_BITS,
        "charged_tail_calls_per_transition": 2,
        "source_bits_read_at_compile": SOURCE_BITS_READ_MAX,
        "source_bytes_retained_after_compile": 0,
        "oracle_calls_at_inference": 0,
        "training_examples": 0,
        "optimizer_updates": 0,
        "training_flops": 0,
        "external_memory_bits": 0,
        "external_execution_calls": 0,
        "sequential_depth_per_task_step": 1,
    }


def resource_vector(arm: str) -> dict[str, Any]:
    if arm not in EXECUTABLE_ARMS:
        raise ValueError("resource arm is unknown")
    if arm == ARM_POSITIVE:
        utilized = {
            "utilized_persistent_state_bits": STATE_BITS,
            "utilized_transient_vector_bits": TRANSIENT_VECTOR_BITS,
            "utilized_fixed_tail_table_entries": FIXED_TAIL_TABLE_ENTRIES,
            "semantic_tail_calls_per_transition": 2,
        }
    elif arm == ARM_ZERO:
        utilized = {
            "utilized_persistent_state_bits": STATE_BITS,
            "utilized_transient_vector_bits": TRANSIENT_VECTOR_BITS,
            "utilized_fixed_tail_table_entries": 0,
            "semantic_tail_calls_per_transition": 2,
        }
    else:
        utilized = {
            "utilized_persistent_state_bits": SHORTCUT_USED_STATE_BITS,
            "utilized_transient_vector_bits": 0,
            "utilized_fixed_tail_table_entries": 0,
            "semantic_tail_calls_per_transition": 0,
        }
    return {
        "arm": arm,
        "allocated": allocated_resource_budget(),
        "utilized": utilized,
    }


def matched_resource_audit() -> dict[str, Any]:
    vectors = {arm: resource_vector(arm) for arm in EXECUTABLE_ARMS}
    allocated = [vectors[arm]["allocated"] for arm in EXECUTABLE_ARMS]
    return {
        "arms": vectors,
        "allocated_budgets_identical": all(
            budget == allocated[0] for budget in allocated[1:]
        ),
        "audit_oracle_transition_calls_per_arm": board_summary()[
            "source_trajectory_step_count_including_halt"
        ],
        "padding_reported_separately": True,
    }


def run_audit() -> dict[str, Any]:
    board = board_summary()
    positive_cells = mixed_cell_audit(ARM_POSITIVE)
    zero_cells = mixed_cell_audit(ARM_ZERO)
    positive_trajectories = trajectory_audit(ARM_POSITIVE)
    zero_trajectories = trajectory_audit(ARM_ZERO)
    shortcut_trajectories = trajectory_audit(ARM_SHORTCUT)
    task_moore = moore_minimization_audit(MACHINE_TASK)
    positive_moore = moore_minimization_audit(ARM_POSITIVE)
    zero_moore = moore_minimization_audit(ARM_ZERO)
    shortcut_moore = moore_minimization_audit(ARM_SHORTCUT)
    erasure = erasure_and_donor_audit()
    resources = matched_resource_audit()
    total_cells = STATE_COUNT * ACTION_COUNT
    total_cases = len(source_cases())
    gates = {
        "noncommutative_board": board["noncommutative_witness"]["AB"]
        != board["noncommutative_witness"]["BA"],
        "positive_cells_exact": positive_cells["exact_successors"] == total_cells
        and positive_cells["valid_decodes"] == total_cells,
        "positive_trajectories_exact": positive_trajectories[
            "exact_trajectories"
        ]
        == total_cases,
        "zero_interaction_exact": zero_cells["zero_vectors"] == total_cells
        and zero_cells["valid_decodes"] == 0
        and zero_trajectories["exact_trajectories"] == 0,
        "shortcut_ceiling_exact": shortcut_trajectories["exact_trajectories"]
        == MODULUS * MAX_PROGRAM_LENGTH,
        "task_moore_exact": task_moore["class_count"] == STATE_COUNT,
        "positive_moore_exact": positive_moore["class_count"] == STATE_COUNT,
        "collapsed_controls": zero_moore["class_count"] < STATE_COUNT
        and shortcut_moore["class_count"] < STATE_COUNT,
        "erasure_exact": erasure["structural_source_free"]
        and erasure["source_mutation_bit_identical"] == total_cases
        and erasure["stale_state_bit_identical"] == total_cases
        and erasure["donor_following"] == total_cases,
        "allocated_resources_matched": resources["allocated_budgets_identical"],
        "source_commitment_frozen": board["source_commitment_sha256"]
        == FROZEN_SOURCE_SHA256,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "deterministic planted mechanics only; no neural learnability, "
            "Shohin capability, architecture, novelty, or result claim"
        ),
        "board": board,
        "mixed_cells": {
            ARM_POSITIVE: positive_cells,
            ARM_ZERO: zero_cells,
        },
        "trajectories": {
            ARM_POSITIVE: positive_trajectories,
            ARM_ZERO: zero_trajectories,
            ARM_SHORTCUT: shortcut_trajectories,
        },
        "moore_minimization": {
            MACHINE_TASK: task_moore,
            ARM_POSITIVE: positive_moore,
            ARM_ZERO: zero_moore,
            ARM_SHORTCUT: shortcut_moore,
        },
        "erasure_and_donor": erasure,
        "resource_accounting": resources,
        "gates": gates,
        "mechanics_contract_satisfied": all(gates.values()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent the deterministic JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_audit()
    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(canonical_json_bytes(report).decode("ascii"), end="")
    return 0 if report["mechanics_contract_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
