#!/usr/bin/env python3
"""CPU-only full-program falsifier for the bounded VAMT v3 machine.

The candidate consumes explicit categorical executor and serializer tables. An
independent Python reference interpreter scores complete programs. All table
lookups are external symbolic execution: this artifact proves no model-owned
reasoning and authorizes no neural fit or accelerator work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

PROTOCOL = "R12-VAMT-FULL-MACHINE-FALSIFIER-v3"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
BASE_PARAMETERS = 125_081_664
STRICT_TOTAL_MAXIMUM = 149_999_999
T_MAX = 256
W = 16
D = W + 1
L = 8
PAD_CURSOR = T_MAX
EXECUTOR_CYCLES = L * D
SERIALIZER_CYCLES = D
OPS = ("ADD", "SUB")
PROGRAM_OPS = ("LOAD", "ADD", "SUB", "HALT")
DIGITS = tuple(range(10))

# Candidate and reference codebooks are separately constructed immutable maps.
# Their values are frozen by the tokenizer artifact above, not learned here.
CANDIDATE_DIGIT_BY_TOKEN = MappingProxyType(
    {28: 0, 29: 1, 30: 2, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 9}
)
CANDIDATE_TOKEN_BY_DIGIT = MappingProxyType(
    {0: 28, 1: 29, 2: 30, 3: 31, 4: 32, 5: 33, 6: 34, 7: 35, 8: 36, 9: 37}
)
REFERENCE_DIGIT_BY_TOKEN = MappingProxyType(
    {28: 0, 29: 1, 30: 2, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8, 37: 9}
)
REFERENCE_TOKEN_BY_DIGIT = MappingProxyType(
    {0: 28, 1: 29, 2: 30, 3: 31, 4: 32, 5: 33, 6: 34, 7: 35, 8: 36, 9: 37}
)

ExecutorContext = tuple[str, int, int, int]
ExecutorOutcome = tuple[int, int]
SerializerContext = tuple[int, int, int]
SerializerOutcome = tuple[int, int, int, int]


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def executor_contexts() -> frozenset[ExecutorContext]:
    return frozenset(
        (op, left, right, carry)
        for op in OPS
        for left in DIGITS
        for right in DIGITS
        for carry in (0, 1)
    )


def serializer_contexts() -> frozenset[SerializerContext]:
    return frozenset(
        (seen, digit, at_last)
        for seen in (0, 1)
        for digit in DIGITS
        for at_last in (0, 1)
    )


def _candidate_executor_outcome(context: ExecutorContext) -> ExecutorOutcome:
    op, left, right, carry = context
    if op == "ADD":
        value = left + right + carry
        return value % 10, value // 10
    value = left - right - carry
    return (value + 10, 1) if value < 0 else (value, 0)


def canonical_candidate_executor_table() -> dict[ExecutorContext, ExecutorOutcome]:
    """Oracle-injected candidate table; never used as the scoring oracle."""
    return {context: _candidate_executor_outcome(context) for context in executor_contexts()}


def _candidate_serializer_outcome(context: SerializerContext) -> SerializerOutcome:
    seen, digit, at_last = context
    emit = int(bool(seen or digit != 0 or at_last))
    return emit, int(bool(seen or digit != 0)), at_last, digit


def canonical_candidate_serializer_table() -> dict[SerializerContext, SerializerOutcome]:
    """Oracle-injected candidate table; never used as the scoring oracle."""
    return {
        context: _candidate_serializer_outcome(context)
        for context in serializer_contexts()
    }


@dataclass(frozen=True)
class Instruction:
    opcode: str
    start: int = 0
    end: int = 0


@dataclass
class MachineState:
    pc: int = 0
    phase: int = 0
    source_cursor: int = PAD_CURSOR
    carry_or_borrow: int = 0
    accumulator: list[int] = field(default_factory=lambda: [0] * D)
    invalid: bool = False
    halted: bool = False


@dataclass
class SerializerState:
    read_cursor: int = D - 1
    seen_nonzero: int = 0
    serializer_halted: bool = False
    write_cursor: int = 0
    status: str = "RUN"
    output_token_ids: list[int] = field(default_factory=lambda: [0] * D)


@dataclass
class RuntimeLedger:
    executor_cycles: int = 0
    active_executor_cycles: int = 0
    masked_executor_cycles: int = 0
    transition_lookups: int = 0
    add_transition_lookups: int = 0
    sub_transition_lookups: int = 0
    pointer_reads: int = 0
    load_writes: int = 0
    categorical_writes: int = 0
    halt_actions: int = 0
    invalidations: int = 0
    serializer_cycles: int = 0
    active_serializer_cycles: int = 0
    masked_serializer_cycles: int = 0
    serializer_lookups: int = 0
    external_execution_calls: int = 0
    semantic_host_arithmetic_calls: int = 0
    parser_repairs: int = 0
    verifier_calls: int = 0
    retries: int = 0


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    output_token_ids: tuple[int, ...]
    output_digits: tuple[int, ...]
    state: MachineState
    serializer_state: SerializerState
    ledger: RuntimeLedger


@dataclass(frozen=True)
class ReferenceResult:
    status: str
    output_token_ids: tuple[int, ...]
    output_digits: tuple[int, ...]


@dataclass(frozen=True)
class ProgramCase:
    name: str
    category: str
    source_tokens: tuple[int, ...]
    program: tuple[Instruction, ...]
    negative_subtraction: bool = False


def _initial_cursor(instruction: Instruction) -> int:
    return instruction.end if 0 <= instruction.end < T_MAX else PAD_CURSOR


def _expected_cursor(instruction: Instruction, phase: int) -> int:
    if phase >= W:
        return PAD_CURSOR
    position = instruction.end - phase
    return position if position >= instruction.start else PAD_CURSOR


def _candidate_valid_instruction_shape(
    instruction: Instruction, source_tokens: Sequence[int]
) -> bool:
    return (
        instruction.opcode in ("LOAD", "ADD", "SUB")
        and type(instruction.start) is int
        and type(instruction.end) is int
        and 0 <= instruction.start <= instruction.end < len(source_tokens) <= T_MAX
        and instruction.end - instruction.start + 1 <= W
    )


class FullProgramMachine:
    """Fixed-cycle bounded interpreter driven only by explicit candidate tables."""

    def __init__(
        self,
        executor_table: Mapping[ExecutorContext, ExecutorOutcome],
        serializer_table: Mapping[SerializerContext, SerializerOutcome],
    ) -> None:
        if set(executor_table) != executor_contexts():
            raise ValueError("executor table must cover exactly 400 contexts")
        if set(serializer_table) != serializer_contexts():
            raise ValueError("serializer table must cover exactly 40 contexts")
        self.executor_table = dict(executor_table)
        self.serializer_table = dict(serializer_table)

    @staticmethod
    def _invalidate(state: MachineState, ledger: RuntimeLedger, *, halt: bool = False) -> None:
        if not state.invalid:
            ledger.invalidations += 1
        state.invalid = True
        if halt:
            state.halted = True

    def _step_executor(
        self,
        state: MachineState,
        source_tokens: Sequence[int],
        program: Sequence[Instruction],
        ledger: RuntimeLedger,
    ) -> None:
        ledger.executor_cycles += 1
        if state.invalid or state.halted:
            ledger.masked_executor_cycles += 1
            return
        ledger.active_executor_cycles += 1
        instruction = program[state.pc]

        if instruction.opcode == "HALT":
            if state.phase != 0:
                self._invalidate(state, ledger, halt=True)
                return
            state.halted = True
            ledger.halt_actions += 1
            return

        if state.phase == 0 and not _candidate_valid_instruction_shape(
            instruction, source_tokens
        ):
            self._invalidate(state, ledger)
            return
        if instruction.opcode not in ("LOAD", "ADD", "SUB"):
            self._invalidate(state, ledger)
            return
        if state.source_cursor != _expected_cursor(instruction, state.phase):
            self._invalidate(state, ledger)
            return

        if state.phase < W:
            ledger.pointer_reads += 1
            if state.source_cursor == PAD_CURSOR:
                right = 0
            else:
                token_id = source_tokens[state.source_cursor]
                right = CANDIDATE_DIGIT_BY_TOKEN.get(token_id, -1)
                if right not in DIGITS:
                    self._invalidate(state, ledger)
                    return
        else:
            right = 0

        if instruction.opcode == "LOAD":
            state.accumulator[state.phase] = right
            ledger.load_writes += 1
            ledger.categorical_writes += 1
        else:
            context = (
                instruction.opcode,
                state.accumulator[state.phase],
                right,
                state.carry_or_borrow,
            )
            outcome = self.executor_table[context]
            ledger.transition_lookups += 1
            ledger.external_execution_calls += 1
            if instruction.opcode == "ADD":
                ledger.add_transition_lookups += 1
            else:
                ledger.sub_transition_lookups += 1
            if (
                not isinstance(outcome, tuple)
                or len(outcome) != 2
                or outcome[0] not in DIGITS
                or outcome[1] not in (0, 1)
            ):
                self._invalidate(state, ledger)
                return
            state.accumulator[state.phase], state.carry_or_borrow = outcome
            ledger.categorical_writes += 1

        if state.phase < W:
            state.phase += 1
            state.source_cursor = _expected_cursor(instruction, state.phase)
            return

        if state.carry_or_borrow != 0:
            self._invalidate(state, ledger)
            return
        state.carry_or_borrow = 0
        if state.pc == L - 1:
            self._invalidate(state, ledger, halt=True)
            return
        state.pc += 1
        state.phase = 0
        state.source_cursor = _initial_cursor(program[state.pc])

    def _run_serializer(
        self, machine_state: MachineState, ledger: RuntimeLedger
    ) -> SerializerState:
        state = SerializerState()
        for _ in range(SERIALIZER_CYCLES):
            ledger.serializer_cycles += 1
            if state.serializer_halted:
                ledger.masked_serializer_cycles += 1
                continue
            ledger.active_serializer_cycles += 1
            if machine_state.invalid or not machine_state.halted:
                state.status = "REJECT"
                state.serializer_halted = True
                continue

            digit = machine_state.accumulator[state.read_cursor]
            at_last = int(state.read_cursor == 0)
            context = (state.seen_nonzero, digit, at_last)
            outcome = self.serializer_table[context]
            ledger.serializer_lookups += 1
            ledger.external_execution_calls += 1
            if (
                not isinstance(outcome, tuple)
                or len(outcome) != 4
                or outcome[0] not in (0, 1)
                or outcome[1] not in (0, 1)
                or outcome[2] not in (0, 1)
                or outcome[3] not in DIGITS
            ):
                state.status = "REJECT"
                state.serializer_halted = True
                continue
            emit, next_seen, halt, symbol = outcome
            if emit:
                if state.write_cursor >= D:
                    state.status = "REJECT"
                    state.serializer_halted = True
                    continue
                state.output_token_ids[state.write_cursor] = CANDIDATE_TOKEN_BY_DIGIT[symbol]
                state.write_cursor += 1
                ledger.categorical_writes += 1
            state.seen_nonzero = next_seen
            if halt:
                state.status = "ACCEPT"
                state.serializer_halted = True
            else:
                state.read_cursor -= 1

        if state.status == "RUN":
            state.status = "REJECT"
            state.serializer_halted = True
            state.write_cursor = 0
        if state.status == "REJECT":
            state.write_cursor = 0
        return state

    def run(
        self, source_tokens: Sequence[int], program: Sequence[Instruction]
    ) -> ExecutionResult:
        if len(program) != L:
            raise ValueError("program must contain exactly eight instruction slots")
        if len(source_tokens) > T_MAX:
            raise ValueError("source exceeds the frozen 256-token bound")
        state = MachineState(source_cursor=_initial_cursor(program[0]))
        ledger = RuntimeLedger()
        for _ in range(EXECUTOR_CYCLES):
            self._step_executor(state, source_tokens, program, ledger)
        serializer_state = self._run_serializer(state, ledger)
        token_ids = tuple(serializer_state.output_token_ids[: serializer_state.write_cursor])
        digits = tuple(CANDIDATE_DIGIT_BY_TOKEN[token] for token in token_ids)
        return ExecutionResult(
            status=serializer_state.status,
            output_token_ids=token_ids,
            output_digits=digits,
            state=state,
            serializer_state=serializer_state,
            ledger=ledger,
        )


def _reference_span_value(
    source_tokens: Sequence[int], instruction: Instruction
) -> int | None:
    # Intentionally duplicate the structural rule instead of calling the
    # candidate validator. A defect or poison in one path cannot validate the
    # other path by construction.
    reference_shape_valid = (
        instruction.opcode in ("LOAD", "ADD", "SUB")
        and type(instruction.start) is int
        and type(instruction.end) is int
        and 0 <= instruction.start
        and instruction.start <= instruction.end
        and instruction.end < len(source_tokens)
        and len(source_tokens) <= 256
        and instruction.end - instruction.start + 1 <= 16
    )
    if not reference_shape_valid:
        return None
    digits = []
    for position in range(instruction.start, instruction.end + 1):
        digit = REFERENCE_DIGIT_BY_TOKEN.get(source_tokens[position])
        if digit is None:
            return None
        digits.append(str(digit))
    return int("".join(digits))


def reference_execute(
    source_tokens: Sequence[int], program: Sequence[Instruction]
) -> ReferenceResult:
    """Independent host oracle; it never reads or constructs candidate tables."""
    if len(program) != L or len(source_tokens) > T_MAX:
        return ReferenceResult("REJECT", (), ())
    accumulator = 0
    for instruction in program:
        if instruction.opcode == "HALT":
            digits = tuple(int(char) for char in str(accumulator))
            tokens = tuple(REFERENCE_TOKEN_BY_DIGIT[digit] for digit in digits)
            return ReferenceResult("ACCEPT", tokens, digits)
        value = _reference_span_value(source_tokens, instruction)
        if value is None:
            return ReferenceResult("REJECT", (), ())
        if instruction.opcode == "LOAD":
            accumulator = value
        elif instruction.opcode == "ADD":
            accumulator += value
        elif instruction.opcode == "SUB":
            accumulator -= value
        else:
            return ReferenceResult("REJECT", (), ())
        if accumulator < 0 or accumulator >= 10**D:
            return ReferenceResult("REJECT", (), ())
    return ReferenceResult("REJECT", (), ())


def _padded_program(instructions: Sequence[Instruction]) -> tuple[Instruction, ...]:
    if len(instructions) > L:
        raise ValueError("too many instructions")
    return tuple(instructions) + (Instruction("HALT"),) * (L - len(instructions))


def _source_with_operands(width: int) -> tuple[tuple[int, ...], dict[str, Instruction]]:
    maximum = 10**width - 1
    values = {
        "a": maximum,
        "b": int("1" * width),
        "c": int("2" * width),
        "z": 0,
    }
    source: list[int] = []
    spans: dict[str, Instruction] = {}
    for index, (name, value) in enumerate(values.items()):
        if index:
            source.append(1_000 + index)
        start = len(source)
        for char in f"{value:0{width}d}":
            source.append(REFERENCE_TOKEN_BY_DIGIT[int(char)])
        spans[name] = Instruction("LOAD", start, len(source) - 1)
    return tuple(source), spans


def _as_op(opcode: str, span: Instruction) -> Instruction:
    return Instruction(opcode, span.start, span.end)


def build_full_program_board() -> tuple[ProgramCase, ...]:
    cases: list[ProgramCase] = []
    for width in range(1, W + 1):
        source, spans = _source_with_operands(width)
        programs = (
            ("load_add", (spans["a"], _as_op("ADD", spans["b"]), Instruction("HALT")), False),
            ("nonnegative_sub", (spans["a"], _as_op("SUB", spans["b"]), Instruction("HALT")), False),
            ("negative_sub_a", (spans["b"], _as_op("SUB", spans["a"]), Instruction("HALT")), True),
            ("negative_sub_b", (spans["z"], _as_op("SUB", spans["b"]), Instruction("HALT")), True),
            ("add_then_sub", (spans["a"], _as_op("ADD", spans["b"]), _as_op("SUB", spans["c"]), Instruction("HALT")), False),
            ("three_addends", (spans["b"], _as_op("ADD", spans["c"]), _as_op("ADD", spans["a"]), Instruction("HALT")), False),
            ("sub_then_add", (spans["a"], _as_op("SUB", spans["c"]), _as_op("ADD", spans["b"]), Instruction("HALT")), False),
            ("load_only", (spans["c"], Instruction("HALT")), False),
        )
        for name, instructions, negative in programs:
            cases.append(
                ProgramCase(
                    name=f"width_{width:02d}_{name}",
                    category="width_sweep",
                    source_tokens=source,
                    program=_padded_program(instructions),
                    negative_subtraction=negative,
                )
            )

    one_digit = (REFERENCE_TOKEN_BY_DIGIT[7],)
    for index in range(16):
        trailing = tuple(
            Instruction(
                ("BOGUS", "ADD", "SUB", "LOAD")[(index + offset) % 4],
                -1 if offset % 2 == 0 else 300,
                300 if offset % 3 == 0 else -2,
            )
            for offset in range(6)
        )
        cases.append(
            ProgramCase(
                name=f"post_halt_mask_{index:02d}",
                category="post_halt_masking",
                source_tokens=one_digit,
                program=(Instruction("LOAD", 0, 0), Instruction("HALT"), *trailing),
            )
        )

    seventeen_digits = tuple(REFERENCE_TOKEN_BY_DIGIT[index % 10] for index in range(17))
    malformed = (
        ("over_width", seventeen_digits, Instruction("LOAD", 0, 16)),
        ("reversed", one_digit, Instruction("LOAD", 1, 0)),
        ("negative_start", one_digit, Instruction("LOAD", -1, 0)),
        ("high_end", one_digit, Instruction("LOAD", 0, 255)),
        ("nondigit", (REFERENCE_TOKEN_BY_DIGIT[1], 9_999, REFERENCE_TOKEN_BY_DIGIT[2]), Instruction("LOAD", 0, 2)),
    )
    for name, source, instruction in malformed:
        cases.append(
            ProgramCase(
                name=f"malformed_{name}",
                category="malformed_span",
                source_tokens=source,
                program=_padded_program((instruction, Instruction("HALT"))),
            )
        )

    cases.append(
        ProgramCase(
            name="missing_halt",
            category="missing_halt",
            source_tokens=(REFERENCE_TOKEN_BY_DIGIT[1],),
            program=tuple(Instruction("ADD", 0, 0) for _ in range(L)),
        )
    )

    maximum_source = tuple(REFERENCE_TOKEN_BY_DIGIT[9] for _ in range(W))
    cases.append(
        ProgramCase(
            name="seven_add_maximum_bound",
            category="seven_add_bound",
            source_tokens=maximum_source,
            program=tuple(Instruction("ADD", 0, W - 1) for _ in range(7))
            + (Instruction("HALT"),),
        )
    )

    cases.append(
        ProgramCase(
            name="terminal_carry_reuse",
            category="terminal_carry_reuse",
            source_tokens=(REFERENCE_TOKEN_BY_DIGIT[9],),
            program=_padded_program(
                (
                    Instruction("LOAD", 0, 0),
                    Instruction("ADD", 0, 0),
                    Instruction("ADD", 0, 0),
                    Instruction("HALT"),
                )
            ),
        )
    )
    if len(cases) != 152:
        raise AssertionError(f"full board must contain 152 cases, got {len(cases)}")
    return tuple(cases)


def _reference_local_outcome(context: ExecutorContext) -> ExecutorOutcome:
    op, left, right, carry = context
    if op == "ADD":
        value = left + right + carry
        return value % 10, int(value >= 10)
    value = left - right - carry
    return value % 10, int(value < 0)


def local_executor_certificate(
    table: Mapping[ExecutorContext, ExecutorOutcome]
) -> dict:
    mismatches = []
    for context in sorted(executor_contexts()):
        expected = _reference_local_outcome(context)
        actual = table.get(context)
        if actual != expected:
            mismatches.append(
                {"context": list(context), "expected": list(expected), "actual": actual}
            )
    return {
        "contexts": 400,
        "correct": 400 - len(mismatches),
        "mismatches": mismatches[:16],
        "reference_uses_candidate_table": False,
        "pass": not mismatches,
    }


def _reference_serializer_outcome(context: SerializerContext) -> SerializerOutcome:
    seen, digit, at_last = context
    emit = int(seen == 1 or digit != 0 or at_last == 1)
    next_seen = int(seen == 1 or digit != 0)
    return emit, next_seen, at_last, digit


def serializer_context_certificate(
    table: Mapping[SerializerContext, SerializerOutcome]
) -> dict:
    mismatches = []
    for context in sorted(serializer_contexts()):
        expected = _reference_serializer_outcome(context)
        actual = table.get(context)
        if actual != expected:
            mismatches.append(
                {"context": list(context), "expected": list(expected), "actual": actual}
            )
    return {
        "contexts": 40,
        "correct": 40 - len(mismatches),
        "mismatches": mismatches[:16],
        "reference_uses_candidate_table": False,
        "pass": not mismatches,
    }


def full_program_certificate(
    executor_table: Mapping[ExecutorContext, ExecutorOutcome],
    serializer_table: Mapping[SerializerContext, SerializerOutcome],
) -> dict:
    machine = FullProgramMachine(executor_table, serializer_table)
    cases = build_full_program_board()
    failures = []
    category_counts = Counter(case.category for case in cases)
    executor_cycles = 0
    serializer_cycles = 0
    negative_sub_cycles_exact = True
    for case in cases:
        expected = reference_execute(case.source_tokens, case.program)
        actual = machine.run(case.source_tokens, case.program)
        executor_cycles += actual.ledger.executor_cycles
        serializer_cycles += actual.ledger.serializer_cycles
        if case.negative_subtraction:
            negative_sub_cycles_exact &= actual.ledger.sub_transition_lookups == D
        exact = (
            actual.status == expected.status
            and actual.output_token_ids == expected.output_token_ids
            and actual.output_digits == expected.output_digits
        )
        if not exact and len(failures) < 16:
            failures.append(
                {
                    "name": case.name,
                    "expected_status": expected.status,
                    "actual_status": actual.status,
                    "expected_digits": list(expected.output_digits),
                    "actual_digits": list(actual.output_digits),
                }
            )
    expected_categories = {
        "width_sweep": 128,
        "post_halt_masking": 16,
        "malformed_span": 5,
        "missing_halt": 1,
        "seven_add_bound": 1,
        "terminal_carry_reuse": 1,
    }
    return {
        "executions": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "expected_category_counts": expected_categories,
        "negative_subtractions": sum(case.negative_subtraction for case in cases),
        "negative_subtractions_execute_all_17_phases": negative_sub_cycles_exact,
        "executor_cycles": executor_cycles,
        "expected_executor_cycles": len(cases) * EXECUTOR_CYCLES,
        "serializer_cycles": serializer_cycles,
        "expected_serializer_cycles": len(cases) * SERIALIZER_CYCLES,
        "failures": failures,
        "candidate_reference_tables_shared": False,
        "pass": (
            not failures
            and dict(category_counts) == expected_categories
            and negative_sub_cycles_exact
            and executor_cycles == len(cases) * EXECUTOR_CYCLES
            and serializer_cycles == len(cases) * SERIALIZER_CYCLES
        ),
    }


def poison_independence_certificate() -> dict:
    executor = canonical_candidate_executor_table()
    serializer = canonical_candidate_serializer_table()
    canonical = full_program_certificate(executor, serializer)

    executor_poison = dict(executor)
    executor_poison[("ADD", 9, 9, 0)] = (9, 0)
    executor_rejected = not full_program_certificate(executor_poison, serializer)["pass"]

    serializer_poison = dict(serializer)
    serializer_poison[(0, 0, 0)] = (1, 0, 0, 9)
    serializer_rejected = not full_program_certificate(executor, serializer_poison)["pass"]

    joint_executor_poison = dict(executor)
    joint_executor_poison[("SUB", 9, 1, 0)] = (9, 0)
    joint_serializer_poison = dict(serializer)
    joint_serializer_poison[(0, 2, 1)] = (1, 1, 1, 3)
    joint_rejected = not full_program_certificate(
        joint_executor_poison, joint_serializer_poison
    )["pass"]

    return {
        "canonical_pass": canonical["pass"],
        "executor_poison_rejected": executor_rejected,
        "serializer_poison_rejected": serializer_rejected,
        "joint_poison_rejected": joint_rejected,
        "mutable_global_truth_table_exists": False,
        "pass": canonical["pass"] and executor_rejected and serializer_rejected and joint_rejected,
    }


def parameter_ledger() -> dict:
    components = {
        "slot_embeddings_8x128": 8 * 128,
        "global_projection_576x128_plus_bias": 576 * 128 + 128,
        "source_key_576x128": 576 * 128,
        "start_end_query_projections": 2 * 128 * 128,
        "opcode_head_128x4_plus_bias": 128 * 4 + 4,
        "executor_factorized_logits": 400 * 10 + 400 * 2,
        "serializer_factorized_logits": 40 * (2 + 2 + 2 + 10),
    }
    additional = sum(components.values())
    total = BASE_PARAMETERS + additional
    return {
        "base_parameters": BASE_PARAMETERS,
        "components": components,
        "additional_parameters": additional,
        "total_parameters": total,
        "strict_total_maximum": STRICT_TOTAL_MAXIMUM,
        "strict_headroom": STRICT_TOTAL_MAXIMUM - total,
        "minimal_only_within_declared_factorized_full_logit_family": True,
        "globally_minimal_claim": False,
        "pass": additional == 187_332 and total == 125_268_996,
    }


def target_information_ledger() -> dict:
    executor_context_bits = 400 * (1 + 4 + 4 + 1)
    executor_target_bits = 400 * (4 + 1)
    serializer_context_bits = 40 * (1 + 4 + 1)
    serializer_target_bits = 40 * (1 + 1 + 1 + 4)
    return {
        "compiler_target_bits_per_program": L * (2 + 8 + 8),
        "executor_context_bits": executor_context_bits,
        "executor_target_bits": executor_target_bits,
        "serializer_context_bits": serializer_context_bits,
        "serializer_target_bits": serializer_target_bits,
        "executor_serializer_target_bits": executor_target_bits + serializer_target_bits,
        "context_and_target_bits": (
            executor_context_bits
            + executor_target_bits
            + serializer_context_bits
            + serializer_target_bits
        ),
        "pass": (
            L * (2 + 8 + 8) == 144
            and executor_target_bits + serializer_target_bits == 2_280
            and executor_context_bits
            + executor_target_bits
            + serializer_context_bits
            + serializer_target_bits
            == 6_520
        ),
    }


def bounded_resource_ledger() -> dict:
    packed_bits = {
        "program": L * (2 + 8 + 8),
        "machine": 3 + 5 + 9 + 1 + D * 4 + 1 + 1,
        "serializer": 5 + 1 + 1 + 5 + 2,
    }
    byte_addressed = {
        "program_opcodes_uint8": 8,
        "program_starts_uint8": 8,
        "program_ends_uint8": 8,
        "pc_uint8": 1,
        "phase_uint8": 1,
        "source_cursor_uint16": 2,
        "carry_uint8": 1,
        "accumulator_uint8": 17,
        "invalid_uint8": 1,
        "halted_uint8": 1,
        "serializer_private_uint8": 5,
    }
    output = {"token_ids_uint16": 34, "length_uint8": 1, "status_uint8": 1}
    executor_temp = {
        "factorized_logits_float32": 48,
        "argmax_categories_uint8": 2,
        "source_token_uint16": 2,
        "phase_flags_uint8": 1,
    }
    serializer_temp = {
        "factorized_logits_float32": 64,
        "decisions_uint8": 4,
        "input_digit_uint8": 1,
    }
    compiler_temp = {
        "global_projection_float32": 512,
        "source_keys_float32": 131_072,
        "start_end_scores_float32": 16_384,
        "source_hidden_copy_float32": 589_824,
        "opcode_logits_float32": 128,
        "slot_embeddings_float32": 4_096,
        "start_end_queries_float32": 8_192,
        "scalar_scratch": 24,
    }
    packed_total = sum(packed_bits.values())
    compiler_peak = sum(compiler_temp.values())
    return {
        "bounds": {"source_tokens": T_MAX, "source_width": W, "accumulator_digits": D, "instructions": L},
        "packed_program_private_bits": packed_bits,
        "packed_program_private_bytes": math.ceil(packed_total / 8),
        "byte_addressed_program_private_components": byte_addressed,
        "byte_addressed_program_private_bytes": sum(byte_addressed.values()),
        "immutable_source_bytes": T_MAX * 2,
        "output_components": output,
        "output_length_status_bytes": sum(output.values()),
        "digit_codebook_bytes": 10 * 2,
        "executor_temporary_components": executor_temp,
        "executor_temporary_bytes": sum(executor_temp.values()),
        "serializer_temporary_components": serializer_temp,
        "serializer_temporary_bytes": sum(serializer_temp.values()),
        "post_base_compiler_temporary_components": compiler_temp,
        "post_base_compiler_temporary_peak_bytes": compiler_peak,
        "compiler_phase_including_source_codebook_bytes": compiler_peak + T_MAX * 2 + 10 * 2,
        "post_compiler_serializer_live_bytes": 688,
        "base_activation_allocation_peak": "UNKNOWN_MUST_BE_MEASURED",
        "exact_full_peak_claim_allowed": False,
        "pass": (
            packed_total == 246
            and math.ceil(packed_total / 8) == 31
            and sum(byte_addressed.values()) == 53
            and sum(output.values()) == 36
            and sum(executor_temp.values()) == 53
            and sum(serializer_temp.values()) == 69
            and compiler_peak == 750_232
            and compiler_peak + 532 == 750_764
        ),
    }


def compute_ledger() -> dict:
    compiler = {
        "global_projection": 576 * 128,
        "source_keys": T_MAX * 576 * 128,
        "start_end_queries": 2 * L * 128 * 128,
        "opcode_head": L * 128 * 4,
        "pointer_scores": 2 * L * T_MAX * 128,
    }
    compiler_macs = sum(compiler.values())
    executor_dense = EXECUTOR_CYCLES * (400 * 10 + 400 * 2)
    serializer_dense = SERIALIZER_CYCLES * (40 * (2 + 2 + 2 + 10))
    return {
        "compiler_components": compiler,
        "compiler_matrix_macs": compiler_macs,
        "executor_dense_one_hot_equivalent_macs": executor_dense,
        "serializer_dense_one_hot_equivalent_macs": serializer_dense,
        "executor_serializer_dense_one_hot_equivalent_macs": executor_dense + serializer_dense,
        "total_non_base_dense_equivalent_macs": compiler_macs + executor_dense + serializer_dense,
        "fixed_executor_cycles": EXECUTOR_CYCLES,
        "fixed_serializer_cycles": SERIALIZER_CYCLES,
        "pass": (
            compiler_macs == 19_738_624
            and executor_dense + serializer_dense == 663_680
            and compiler_macs + executor_dense + serializer_dense == 20_402_304
        ),
    }


def finite_machine_and_control_boundary() -> dict:
    packed_states_upper = 2 ** 246
    return {
        "machine_class": "bounded deterministic finite-state transducer",
        "packed_program_private_bits": 246,
        "state_count_upper_bound": packed_states_upper,
        "fixed_executor_depth": EXECUTOR_CYCLES,
        "fixed_serializer_depth": SERIALIZER_CYCLES,
        "favorable_control": "pointer network plus tied Mealy/NPI controller",
        "fixed_digit_permutation_isomorphism": True,
        "control_receives_same_cases_labels_state_cycles_parameters_macs": True,
        "novel_primitive_claim_allowed": False,
        "surviving_claim_if_neural_gate_later_passes": "optimization/data-efficiency/vocabulary-alignment only",
        "pass": packed_states_upper.bit_length() == 247,
    }


def symbolic_runtime_accounting() -> dict:
    machine = FullProgramMachine(
        canonical_candidate_executor_table(), canonical_candidate_serializer_table()
    )
    case = next(case for case in build_full_program_board() if case.name == "terminal_carry_reuse")
    result = machine.run(case.source_tokens, case.program)
    forbidden = (
        "semantic_host_arithmetic_calls",
        "parser_repairs",
        "verifier_calls",
        "retries",
    )
    ledger = asdict(result.ledger)
    return {
        "case": case.name,
        "status": result.status,
        "output_digits": list(result.output_digits),
        "ledger": ledger,
        "forbidden_runtime_calls_zero": {name: ledger[name] == 0 for name in forbidden},
        "all_executor_cycles_charged": ledger["executor_cycles"] == EXECUTOR_CYCLES,
        "all_serializer_cycles_charged": ledger["serializer_cycles"] == SERIALIZER_CYCLES,
        "external_symbolic_execution_counted": ledger["external_execution_calls"] > 0,
        "oracle_injected_external_execution": True,
        "autonomous_capability": False,
        "pass": (
            result.status == "ACCEPT"
            and result.output_digits == (2, 7)
            and all(ledger[name] == 0 for name in forbidden)
            and ledger["executor_cycles"] == EXECUTOR_CYCLES
            and ledger["serializer_cycles"] == SERIALIZER_CYCLES
            and ledger["external_execution_calls"] > 0
        ),
    }


def build_report() -> dict:
    executor = canonical_candidate_executor_table()
    serializer = canonical_candidate_serializer_table()
    sections = {
        "local_executor_contexts": local_executor_certificate(executor),
        "serializer_contexts": serializer_context_certificate(serializer),
        "full_program_board": full_program_certificate(executor, serializer),
        "poison_independence": poison_independence_certificate(),
        "parameter_ledger": parameter_ledger(),
        "target_information_ledger": target_information_ledger(),
        "bounded_resource_ledger": bounded_resource_ledger(),
        "compute_ledger": compute_ledger(),
        "finite_machine_and_control_boundary": finite_machine_and_control_boundary(),
        "symbolic_runtime_accounting": symbolic_runtime_accounting(),
    }
    report = {
        "protocol": PROTOCOL,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "digit_token_ids": {str(digit): token for digit, token in CANDIDATE_TOKEN_BY_DIGIT.items()},
        "claim_boundary": (
            "A pass proves only deterministic bounded mechanics, independent scoring, exact ledgers, "
            "and poison sensitivity for this external symbolic machine. It does not authorize neural "
            "code, fitting, H100 work, autonomous arithmetic, reasoning, novelty, or a Shohin claim."
        ),
        "oracle_injected_external_execution": True,
        "autonomous_capability": False,
        "neural_preregistration_authorized": False,
        "sections": sections,
        "all_pass": all(section["pass"] for section in sections.values()),
    }
    report["payload_sha256"] = payload_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = canonical_json_bytes(report)
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
    print(payload.decode(), end="")
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
