#!/usr/bin/env python3
"""CPU structural falsifier for the factorized counterfactual residual cycle.

This is an oracle mechanics board, not a neural model or a training path.  The
hard-coded decimal operator proves that the proposed interfaces can express the
target transition while enforcing the frozen dependency graph.  Every oracle
call is reported as external execution, so a pass cannot support a capability
or learnability claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable


PROTOCOL_ID = "R12-FCRC-CPU-v2"
SCHEMA_VERSION = 2
OPERATIONS = ("add", "sub")
PHASE_RUN = "run"
PHASE_FINAL = "final"
PHASE_HALT = "halt"
PHASES = (PHASE_RUN, PHASE_FINAL, PHASE_HALT)
PACKET_FIELDS = ("cursor", "carry", "phase")
LOCAL_OPERATOR_PARAMETERS = ("operation", "left_digit", "right_digit", "carry")
ADDRESS_SOURCE_PARAMETERS = ("source", "cursor")
LATE_ACTUATOR_PARAMETERS = ("local_result",)
TERMINAL_ACTUATOR_PARAMETERS = ("packet",)
MAX_MECHANICS_WIDTH = 8
LOCAL_TABLE_ENTRIES = len(OPERATIONS) * 10 * 10 * 2
CASES_PER_REGIME = 100
REQUIRED_NEURAL_CONTROLS = (
    "token_sft",
    "generic_recurrent",
    "learned_400_entry_table",
    "carry_only_rank8_writer_reader",
)

REGIME_SPECS = {
    "fit_w4": {
        "width": 4,
        "scalar_intervals": ((1_000, 3_999), (6_000, 8_999)),
    },
    "fit_w6": {
        "width": 6,
        "scalar_intervals": ((100_000, 399_999), (600_000, 899_999)),
    },
    "value_ood_w4": {
        "width": 4,
        "scalar_intervals": ((4_000, 5_999), (9_000, 9_999)),
    },
    "value_ood_w6": {
        "width": 6,
        "scalar_intervals": ((400_000, 599_999), (900_000, 999_999)),
    },
    "width_ood_w8": {
        "width": 8,
        "scalar_intervals": ((40_000_000, 59_999_999), (90_000_000, 99_999_999)),
    },
}

CONTEXT_VARIANTS = (
    "cursor",
    "terminal",
    "width_6",
    "width_8",
    "result_prefix",
    "history",
)
CANONICAL_R3_SHA256 = "0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a"
CANONICAL_R3_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/evals/drs_causal_cycle_post_drs_r3.json"
)


class ContractError(ValueError):
    """The frozen mechanics contract was violated."""


def require_plain_int(value: Any, label: str) -> int:
    """Accept only a built-in scalar; subclasses can carry hidden payloads."""
    if type(value) is not int:
        raise TypeError(f"{label} must be a plain built-in integer")
    return value


def require_plain_str(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a plain built-in string")
    return value


@dataclass(frozen=True, slots=True)
class Packet:
    """The complete mutable state allowed to cross an FCRC cycle boundary."""

    cursor: int
    carry: int
    phase: str

    def __post_init__(self) -> None:
        require_plain_int(self.cursor, "cursor")
        require_plain_int(self.carry, "carry")
        require_plain_str(self.phase, "phase")
        if self.cursor < 0:
            raise ValueError("cursor must be nonnegative")
        if self.carry not in (0, 1):
            raise ValueError("carry must be zero or one")
        if self.phase not in PHASES:
            raise ValueError("phase is outside the frozen alphabet")


@dataclass(frozen=True, slots=True)
class GrowingPacket:
    """Negative control: an illegal packet that accumulates a result tape."""

    cursor: int
    carry: int
    phase: str
    result_tape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Source:
    """Read-only operand memory.  It is not retained inside the hard packet."""

    operation: str
    left_digits: tuple[int, ...]
    right_digits: tuple[int, ...]

    def __post_init__(self) -> None:
        require_plain_str(self.operation, "operation")
        if type(self.left_digits) is not tuple or type(self.right_digits) is not tuple:
            raise TypeError("source digit containers must be plain tuples")
        if self.operation not in OPERATIONS:
            raise ValueError("operation must be add or sub")
        if len(self.left_digits) != len(self.right_digits):
            raise ValueError("operand widths differ")
        if not 2 <= len(self.left_digits) <= MAX_MECHANICS_WIDTH:
            raise ValueError("source width is outside the mechanics board")
        for digit in (*self.left_digits, *self.right_digits):
            require_plain_int(digit, "source digit")
            if not 0 <= digit <= 9:
                raise ValueError("source digit is outside decimal range")

    @property
    def width(self) -> int:
        return len(self.left_digits)


@dataclass(frozen=True, slots=True)
class AddressRead:
    """One source symbol selected by the learned-address interface contract."""

    operation: str
    left_digit: int
    right_digit: int

    def __post_init__(self) -> None:
        require_plain_str(self.operation, "address operation")
        require_plain_int(self.left_digit, "address left digit")
        require_plain_int(self.right_digit, "address right digit")
        if self.operation not in OPERATIONS:
            raise ValueError("address operation must be add or sub")
        if not 0 <= self.left_digit <= 9 or not 0 <= self.right_digit <= 9:
            raise ValueError("address digit is outside decimal range")


@dataclass(frozen=True, slots=True)
class LocalResult:
    """Position-blind local arithmetic result."""

    digit: int
    next_carry: int

    def __post_init__(self) -> None:
        require_plain_int(self.digit, "local result digit")
        require_plain_int(self.next_carry, "next carry")
        if not 0 <= self.digit <= 9:
            raise ValueError("local result digit is outside decimal range")
        if self.next_carry not in (0, 1):
            raise ValueError("next carry must be zero or one")


@dataclass(frozen=True, slots=True)
class Emission:
    """One write-only symbol emitted by the late actuator."""

    kind: str
    value: int

    def __post_init__(self) -> None:
        require_plain_str(self.kind, "emission kind")
        require_plain_int(self.value, "emission value")
        if self.kind not in ("digit", "terminal_carry"):
            raise ValueError("unknown emission kind")
        if self.kind == "digit" and not 0 <= self.value <= 9:
            raise ValueError("digit emission is outside decimal range")
        if self.kind == "terminal_carry" and self.value not in (0, 1):
            raise ValueError("terminal carry emission must be zero or one")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One autonomous packet transition and write-only emission."""

    before: Packet
    after: Packet
    emission: Emission

    def __post_init__(self) -> None:
        if type(self.before) is not Packet or type(self.after) is not Packet:
            raise TypeError("step packets must use the exact Packet type")
        if type(self.emission) is not Emission:
            raise TypeError("step emission must use the exact Emission type")


@dataclass(frozen=True, slots=True)
class ProbeContext:
    """Superset context used only to detect forbidden local dependencies."""

    operation: str
    left_digit: int
    right_digit: int
    carry: int
    width: int
    cursor: int
    terminal: bool
    result_prefix: tuple[int, ...]
    generated_history: tuple[int, ...]
    variant: str

    def __post_init__(self) -> None:
        require_plain_str(self.operation, "probe operation")
        for label, value in (
            ("probe left digit", self.left_digit),
            ("probe right digit", self.right_digit),
            ("probe carry", self.carry),
            ("probe width", self.width),
            ("probe cursor", self.cursor),
        ):
            require_plain_int(value, label)
        if type(self.terminal) is not bool:
            raise TypeError("probe terminal flag must be a plain bool")
        if (
            type(self.result_prefix) is not tuple
            or type(self.generated_history) is not tuple
        ):
            raise TypeError("probe histories must be plain tuples")
        for value in (*self.result_prefix, *self.generated_history):
            require_plain_int(value, "probe history value")
        require_plain_str(self.variant, "probe variant")

    @property
    def local_key(self) -> tuple[str, int, int, int]:
        return (self.operation, self.left_digit, self.right_digit, self.carry)


LocalOperator = Callable[[str, int, int, int], LocalResult]
ContextOperator = Callable[[ProbeContext], LocalResult]
Observer = Callable[[int, Emission], Any]


def packet_bits(width: int) -> int:
    """Logical packet capacity, including unavoidable logarithmic cursor bits."""
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    cursor_bits = math.ceil(math.log2(width + 1))
    phase_bits = math.ceil(math.log2(len(PHASES)))
    return cursor_bits + 1 + phase_bits


def digits_lsf(value: int, width: int) -> tuple[int, ...]:
    require_plain_int(value, "value")
    require_plain_int(width, "width")
    if not 0 <= value < 10**width:
        raise ValueError("value does not fit source width")
    return tuple((value // (10**position)) % 10 for position in range(width))


def digits_value(digits: tuple[int, ...]) -> int:
    return sum(digit * (10**position) for position, digit in enumerate(digits))


def source_from_values(operation: str, left: int, right: int, width: int) -> Source:
    return Source(operation, digits_lsf(left, width), digits_lsf(right, width))


def address_source(source: Source, cursor: int) -> AddressRead:
    """Oracle realization of the interface a future learned address head must fit."""
    if type(source) is not Source:
        raise TypeError("address source must use the exact Source type")
    require_plain_int(cursor, "cursor")
    if not 0 <= cursor < source.width:
        raise ContractError("address cursor is outside the operand source")
    return AddressRead(
        source.operation,
        source.left_digits[cursor],
        source.right_digits[cursor],
    )


def position_blind_local_operator(
    operation: str,
    left_digit: int,
    right_digit: int,
    carry: int,
) -> LocalResult:
    """The complete 400-cell decimal law with no position or history input."""
    require_plain_str(operation, "operation")
    if operation not in OPERATIONS:
        raise ValueError("operation must be add or sub")
    if any(type(value) is not int for value in (left_digit, right_digit, carry)):
        raise TypeError("local arithmetic inputs must be integers")
    if not 0 <= left_digit <= 9 or not 0 <= right_digit <= 9:
        raise ValueError("operand digit is outside decimal range")
    if carry not in (0, 1):
        raise ValueError("carry must be zero or one")
    if operation == "add":
        total = left_digit + right_digit + carry
        return LocalResult(total % 10, total // 10)
    total = left_digit - right_digit - carry
    return LocalResult((total + 10) % 10, int(total < 0))


def late_residual_actuator(local_result: LocalResult) -> Emission:
    """Write one local digit; the write is never retained as future state."""
    return Emission("digit", local_result.digit)


def terminal_actuator(packet: Packet) -> Emission:
    """Emit the carried bit from the packet without reading width or a result tape."""
    if packet.phase != PHASE_FINAL:
        raise ContractError("terminal actuator requires final phase")
    return Emission("terminal_carry", packet.carry)


def fcrc_step(
    source: Source,
    packet: Packet,
    local_operator: LocalOperator = position_blind_local_operator,
) -> StepRecord:
    """Advance only from read-only source plus the three-field hard packet."""
    if packet.cursor > source.width:
        raise ContractError("packet cursor escaped source width")
    if packet.phase == PHASE_HALT:
        raise ContractError("halted packet cannot advance")
    if packet.phase == PHASE_FINAL:
        if packet.cursor != source.width:
            raise ContractError("final phase requires cursor at source end")
        emission = terminal_actuator(packet)
        after = Packet(packet.cursor, packet.carry, PHASE_HALT)
        return StepRecord(packet, after, emission)
    if packet.cursor >= source.width:
        raise ContractError("run phase cannot address source end")
    addressed = address_source(source, packet.cursor)
    local = local_operator(
        addressed.operation,
        addressed.left_digit,
        addressed.right_digit,
        packet.carry,
    )
    emission = late_residual_actuator(local)
    next_cursor = packet.cursor + 1
    next_phase = PHASE_FINAL if next_cursor == source.width else PHASE_RUN
    after = Packet(next_cursor, local.next_carry, next_phase)
    return StepRecord(packet, after, emission)


def rollout(
    source: Source,
    *,
    observer: Observer | None = None,
    local_operator: LocalOperator = position_blind_local_operator,
) -> tuple[StepRecord, ...]:
    """Run to HALT; observer output is deliberately discarded and cannot feed back."""
    packet = Packet(0, 0, PHASE_RUN)
    records = []
    for step_index in range(source.width + 1):
        record = fcrc_step(source, packet, local_operator)
        records.append(record)
        if observer is not None:
            observer(step_index, record.emission)
        packet = record.after
    if packet.phase != PHASE_HALT:
        raise ContractError("rollout did not reach halt at the frozen depth")
    return tuple(records)


def emission_trace(records: tuple[StepRecord, ...]) -> tuple[tuple[str, int], ...]:
    return tuple((record.emission.kind, record.emission.value) for record in records)


def oracle_trace(source: Source) -> tuple[tuple[str, int], ...]:
    """Independent integer-arithmetic trace for mechanics scoring."""
    left = digits_value(source.left_digits)
    right = digits_value(source.right_digits)
    modulus = 10**source.width
    if source.operation == "add":
        total = left + right
        body, terminal = total % modulus, total // modulus
    else:
        difference = left - right
        body, terminal = difference % modulus, int(difference < 0)
    trace = [("digit", digit) for digit in digits_lsf(body, source.width)]
    trace.append(("terminal_carry", terminal))
    return tuple(trace)


def iter_local_keys() -> Iterable[tuple[str, int, int, int]]:
    return itertools.product(OPERATIONS, range(10), range(10), range(2))


def _context_variants(
    operation: str,
    left_digit: int,
    right_digit: int,
    carry: int,
) -> tuple[ProbeContext, ...]:
    common = (operation, left_digit, right_digit, carry)
    return (
        ProbeContext(*common, 4, 1, False, (0,), (), "base"),
        ProbeContext(*common, 4, 2, False, (0, 0), (), "cursor"),
        ProbeContext(*common, 4, 3, True, (0, 0, 0), (), "terminal"),
        ProbeContext(*common, 6, 1, False, (0,), (), "width_6"),
        ProbeContext(*common, 8, 1, False, (0,), (), "width_8"),
        ProbeContext(*common, 4, 1, False, (9,), (), "result_prefix"),
        ProbeContext(*common, 4, 1, False, (0,), (7, 3, 9), "history"),
    )


def reference_context_operator(context: ProbeContext) -> LocalResult:
    return position_blind_local_operator(
        context.operation,
        context.left_digit,
        context.right_digit,
        context.carry,
    )


# These aliases bind admission to the original in-module functions.  Public
# entry points may be monkey-patched in tests, but an alternate callable must
# never become its own expected identity merely because it has the same name.
_EXPECTED_REFERENCE_CONTEXT_OPERATOR = reference_context_operator
_EXPECTED_POSITION_BLIND_LOCAL_OPERATOR = position_blind_local_operator
_EXPECTED_ADDRESS_SOURCE = address_source
_EXPECTED_LATE_RESIDUAL_ACTUATOR = late_residual_actuator
_EXPECTED_TERMINAL_ACTUATOR = terminal_actuator


def terminal_zero_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if context.terminal and context.operation == "add":
        return LocalResult(result.digit, 0)
    return result


def cursor_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if context.cursor == 2:
        return LocalResult((result.digit + 1) % 10, result.next_carry)
    return result


def width_6_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if context.width == 6:
        return LocalResult((result.digit + 1) % 10, result.next_carry)
    return result


def width_8_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if context.width == 8:
        return LocalResult((result.digit + 1) % 10, result.next_carry)
    return result


def result_prefix_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if any(context.result_prefix):
        return LocalResult((result.digit + 1) % 10, result.next_carry)
    return result


def generated_history_leak(context: ProbeContext) -> LocalResult:
    result = reference_context_operator(context)
    if context.generated_history:
        return LocalResult((result.digit + 1) % 10, result.next_carry)
    return result


def context_invariance_audit(operator: ContextOperator) -> dict[str, Any]:
    violations = {variant: 0 for variant in CONTEXT_VARIANTS}
    witnesses: dict[str, dict[str, Any]] = {}
    groups = 0
    for key in iter_local_keys():
        contexts = _context_variants(*key)
        baseline = operator(contexts[0])
        groups += 1
        for context in contexts[1:]:
            observed = operator(context)
            if observed == baseline:
                continue
            violations[context.variant] += 1
            witnesses.setdefault(
                context.variant,
                {
                    "local_key": list(key),
                    "baseline": [baseline.digit, baseline.next_carry],
                    "observed": [observed.digit, observed.next_carry],
                },
            )
    return {
        "local_equivalence_groups": groups,
        "violations_by_variant": violations,
        "total_violations": sum(violations.values()),
        "witnesses": witnesses,
    }


def callable_dependency_audit(
    operator: Callable[..., Any],
    *,
    expected: Callable[..., Any],
    allowed_globals: set[str],
    allowed_unbound: set[str] | None = None,
) -> dict[str, Any]:
    """Fail closed on alternate callables, closures, or mutable global state.

    This is a source-bound audit for the fixed CPU witness, not a proof of
    arbitrary Python purity.  A future neural implementation needs an explicit
    tensor data-flow audit rather than inheriting this result.
    """
    is_plain_function = inspect.isfunction(operator) and not inspect.ismethod(operator)
    closure = inspect.getclosurevars(operator) if is_plain_function else None
    allowed_unbound = set(allowed_unbound or ())
    referenced_globals = set(closure.globals) if closure is not None else set()
    mutable_global_names = []
    if closure is not None:
        for name, value in closure.globals.items():
            if isinstance(value, (bytearray, dict, list, set)):
                mutable_global_names.append(name)
    try:
        source = inspect.getsource(operator) if is_plain_function else ""
    except (OSError, TypeError):
        source = ""
    checks = {
        "exact_callable_identity": operator is expected,
        "plain_unbound_function": is_plain_function,
        "no_closure_cells": getattr(operator, "__closure__", None) is None,
        "no_defaults": getattr(operator, "__defaults__", None) in (None, ()),
        "no_keyword_defaults": not getattr(operator, "__kwdefaults__", None),
        "no_function_attributes": not getattr(operator, "__dict__", None),
        "no_nonlocals": closure is not None and not closure.nonlocals,
        "unbound_names_exactly_allowlisted": (
            closure is not None and set(closure.unbound) == allowed_unbound
        ),
        "globals_exactly_allowlisted": referenced_globals == allowed_globals,
        "no_mutable_globals": not mutable_global_names,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "referenced_globals": sorted(referenced_globals),
        "mutable_global_names": sorted(mutable_global_names),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "boundary": (
            "Exact in-module callable and static dependency audit under a frozen source "
            "hash; not a general proof of Python purity."
        ),
    }


def fixed_operator_dependency_audit() -> dict[str, Any]:
    context = callable_dependency_audit(
        reference_context_operator,
        expected=_EXPECTED_REFERENCE_CONTEXT_OPERATOR,
        allowed_globals={"position_blind_local_operator"},
        allowed_unbound={"operation", "left_digit", "right_digit", "carry"},
    )
    local = callable_dependency_audit(
        position_blind_local_operator,
        expected=_EXPECTED_POSITION_BLIND_LOCAL_OPERATOR,
        allowed_globals={"LocalResult", "OPERATIONS", "require_plain_str"},
    )
    address = callable_dependency_audit(
        address_source,
        expected=_EXPECTED_ADDRESS_SOURCE,
        allowed_globals={"AddressRead", "ContractError", "Source", "require_plain_int"},
        allowed_unbound={"left_digits", "operation", "right_digits", "width"},
    )
    late_actuator = callable_dependency_audit(
        late_residual_actuator,
        expected=_EXPECTED_LATE_RESIDUAL_ACTUATOR,
        allowed_globals={"Emission"},
        allowed_unbound={"digit"},
    )
    terminal_actuator_audit = callable_dependency_audit(
        terminal_actuator,
        expected=_EXPECTED_TERMINAL_ACTUATOR,
        allowed_globals={"ContractError", "Emission", "PHASE_FINAL"},
        allowed_unbound={"carry", "phase"},
    )
    return {
        "valid": all(
            row["valid"]
            for row in (
                context,
                local,
                address,
                late_actuator,
                terminal_actuator_audit,
            )
        ),
        "context": context,
        "local": local,
        "address": address,
        "late_actuator": late_actuator,
        "terminal_actuator": terminal_actuator_audit,
    }


def packet_schema_audit(packet_type: type[Any]) -> dict[str, Any]:
    if not is_dataclass(packet_type):
        return {"valid": False, "reason": "not_dataclass", "fields": []}
    names = tuple(field.name for field in fields(packet_type))
    has_slots = hasattr(packet_type, "__slots__")
    annotations = tuple(getattr(packet_type, "__annotations__", {}).items())
    expected_annotations = (("cursor", "int"), ("carry", "int"), ("phase", "str"))
    exact_type = packet_type is Packet
    valid = (
        names == PACKET_FIELDS
        and has_slots
        and annotations == expected_annotations
        and exact_type
    )
    return {
        "valid": valid,
        "fields": list(names),
        "expected_fields": list(PACKET_FIELDS),
        "slots": has_slots,
        "exact_type": exact_type,
        "annotations_exact": annotations == expected_annotations,
        "reason": None if valid else "schema_or_slots_mismatch",
    }


def terminal_carry_no_go_audit() -> dict[str, Any]:
    """Finite witness for observational equivalence of T and terminal-zero T0."""
    train_support = 0
    train_agreements = 0
    omitted_support = 0
    omitted_disagreements = 0
    for left, right, carry in itertools.product(range(10), range(10), range(2)):
        target = position_blind_local_operator("add", left, right, carry)
        alternative = LocalResult(target.digit, 0)
        if target.next_carry == 0:
            train_support += 1
            train_agreements += int(target == alternative)
        else:
            omitted_support += 1
            omitted_disagreements += int(target != alternative)
    return {
        "training_terminal_zero_cells": train_support,
        "training_agreements": train_agreements,
        "omitted_terminal_one_cells": omitted_support,
        "omitted_disagreements": omitted_disagreements,
        "observationally_indistinguishable_on_training_support": (
            train_agreements == train_support
        ),
        "different_on_every_omitted_terminal_carry_cell": (
            omitted_disagreements == omitted_support and omitted_support > 0
        ),
    }


def terminal_update_audit() -> dict[str, Any]:
    terminal_exact = 0
    nonterminal_exact = 0
    emission_equal = 0
    for operation, left, right, carry in iter_local_keys():
        terminal_source = Source(
            operation,
            (0, left),
            (0, right),
        )
        nonterminal_source = Source(
            operation,
            (0, left, 0),
            (0, right, 0),
        )
        terminal = fcrc_step(terminal_source, Packet(1, carry, PHASE_RUN))
        nonterminal = fcrc_step(nonterminal_source, Packet(1, carry, PHASE_RUN))
        expected = position_blind_local_operator(operation, left, right, carry)
        terminal_exact += int(terminal.after.carry == expected.next_carry)
        nonterminal_exact += int(nonterminal.after.carry == expected.next_carry)
        emission_equal += int(terminal.emission == nonterminal.emission)
    return {
        "cells": LOCAL_TABLE_ENTRIES,
        "terminal_carry_updates_exact": terminal_exact,
        "nonterminal_carry_updates_exact": nonterminal_exact,
        "terminal_nonterminal_emissions_equal": emission_equal,
    }


def autonomous_two_step_audit() -> dict[str, Any]:
    exact = 0
    carry_boundary_cases = 0
    total = 0
    for operation in OPERATIONS:
        for left, right in itertools.product(range(100), repeat=2):
            source = source_from_values(operation, left, right, 2)
            packet = Packet(0, 0, PHASE_RUN)
            first = fcrc_step(source, packet)
            second = fcrc_step(source, first.after)
            observed = emission_trace((first, second))
            expected = oracle_trace(source)[:2]
            exact += int(observed == expected)
            carry_boundary_cases += int(first.after.carry == 1)
            total += 1
    return {
        "cases": total,
        "exact": exact,
        "carry_boundary_cases": carry_boundary_cases,
    }


def _lcg(value: int) -> int:
    return (1_103_515_245 * value + 12_345) & 0x7FFFFFFF


def _lcg_range(state: int, low: int, high: int) -> tuple[int, int]:
    if low > high:
        raise ValueError("empty deterministic sample range")
    state = _lcg(state)
    return state, low + state % (high - low + 1)


def _sample_scalar(
    state: int,
    scalar_intervals: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    state = _lcg(state)
    interval_index = (state >> 8) % len(scalar_intervals)
    low, high = scalar_intervals[interval_index]
    return _lcg_range(state, low, high)


def scalar_in_declared_support(regime: str, value: int) -> bool:
    require_plain_str(regime, "regime")
    require_plain_int(value, "support value")
    intervals = REGIME_SPECS[regime]["scalar_intervals"]
    return any(low <= value <= high for low, high in intervals)


def _addition_case(
    state: int,
    *,
    width: int,
    scalar_intervals: tuple[tuple[int, int], ...],
    terminal_carry: int,
) -> tuple[int, int, int]:
    """Construct one addition with an exact terminal-carry class."""
    modulus = 10**width
    for _attempt in range(100_000):
        state, left = _sample_scalar(state, scalar_intervals)
        state, right = _sample_scalar(state, scalar_intervals)
        if int(left + right >= modulus) == terminal_carry:
            return state, left, right
    raise AssertionError("addition support cannot realize the requested carry class")


def _values_have_intermediate_borrow(left: int, right: int, width: int) -> bool:
    left_digits = digits_lsf(left, width)
    right_digits = digits_lsf(right, width)
    borrow = 0
    observed = False
    for position, (left_digit, right_digit) in enumerate(
        zip(left_digits, right_digits, strict=True)
    ):
        borrow = int(left_digit - right_digit - borrow < 0)
        if position < width - 1:
            observed |= bool(borrow)
    if borrow:
        raise AssertionError("valid subtraction ended in a final borrow")
    return observed


def _subtraction_case(
    state: int,
    *,
    width: int,
    scalar_intervals: tuple[tuple[int, int], ...],
    intermediate_borrow: int,
) -> tuple[int, int, int]:
    """Construct valid subtraction with or without an interior borrow."""
    for _attempt in range(1_000_000):
        state, left = _sample_scalar(state, scalar_intervals)
        state, right = _sample_scalar(state, scalar_intervals)
        if left < right:
            continue
        observed = _values_have_intermediate_borrow(left, right, width)
        if observed == bool(intermediate_borrow):
            return state, left, right
    raise AssertionError(
        "subtraction support cannot realize the requested borrow class"
    )


def has_intermediate_borrow(source: Source) -> bool:
    if source.operation != "sub":
        raise ValueError("borrow audit requires subtraction")
    carry = 0
    observed = False
    for position in range(source.width):
        result = position_blind_local_operator(
            "sub",
            source.left_digits[position],
            source.right_digits[position],
            carry,
        )
        carry = result.next_carry
        if position < source.width - 1:
            observed |= bool(carry)
    if carry:
        raise AssertionError("mechanics subtraction ended in a final borrow")
    return observed


def mechanics_cases() -> tuple[tuple[str, int, int, Source], ...]:
    cases = []
    state = 0x5F3759DF
    for regime, spec in REGIME_SPECS.items():
        width = spec["width"]
        for index in range(CASES_PER_REGIME):
            operation = OPERATIONS[index % len(OPERATIONS)]
            class_index = index // len(OPERATIONS)
            if operation == "add":
                state, left, right = _addition_case(
                    state,
                    width=width,
                    scalar_intervals=spec["scalar_intervals"],
                    terminal_carry=class_index % 2,
                )
            else:
                state, left, right = _subtraction_case(
                    state,
                    width=width,
                    scalar_intervals=spec["scalar_intervals"],
                    intermediate_borrow=class_index % 2,
                )
            cases.append(
                (regime, left, right, source_from_values(operation, left, right, width))
            )
    return tuple(cases)


def mechanics_balance_audit() -> dict[str, Any]:
    by_regime = {
        regime: {
            "add_carry_0": 0,
            "add_carry_1": 0,
            "sub_no_intermediate_borrow": 0,
            "sub_with_intermediate_borrow": 0,
        }
        for regime in REGIME_SPECS
    }
    for regime, _left, _right, source in mechanics_cases():
        row = by_regime[regime]
        if source.operation == "add":
            terminal = oracle_trace(source)[-1][1]
            row[f"add_carry_{terminal}"] += 1
        else:
            key = (
                "sub_with_intermediate_borrow"
                if has_intermediate_borrow(source)
                else "sub_no_intermediate_borrow"
            )
            row[key] += 1
    expected = {
        "add_carry_0": 25,
        "add_carry_1": 25,
        "sub_no_intermediate_borrow": 25,
        "sub_with_intermediate_borrow": 25,
    }
    return {
        "by_regime": by_regime,
        "expected_per_regime": expected,
        "balanced": all(row == expected for row in by_regime.values()),
    }


def _interval_intersections(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> list[list[int]]:
    intersections = []
    for first_low, first_high in first:
        for second_low, second_high in second:
            low = max(first_low, second_low)
            high = min(first_high, second_high)
            if low <= high:
                intersections.append([low, high])
    return intersections


def mechanics_scalar_support_audit() -> dict[str, Any]:
    """Prove declared and sampled fit/value-OOD scalar supports are disjoint."""
    fit_regimes = tuple(name for name in REGIME_SPECS if name.startswith("fit_"))
    value_ood_regimes = tuple(
        name for name in REGIME_SPECS if name.startswith("value_ood_")
    )
    declared_intersections: dict[str, list[list[int]]] = {}
    for fit_regime in fit_regimes:
        for ood_regime in value_ood_regimes:
            key = f"{fit_regime}__{ood_regime}"
            declared_intersections[key] = _interval_intersections(
                REGIME_SPECS[fit_regime]["scalar_intervals"],
                REGIME_SPECS[ood_regime]["scalar_intervals"],
            )

    declared_validity = {}
    for regime, spec in REGIME_SPECS.items():
        intervals = spec["scalar_intervals"]
        modulus = 10 ** spec["width"]
        declared_validity[regime] = (
            bool(intervals)
            and all(0 <= low <= high < modulus for low, high in intervals)
            and not any(
                _interval_intersections((first,), (second,))
                for index, first in enumerate(intervals)
                for second in intervals[index + 1 :]
            )
        )

    observed = {regime: set() for regime in REGIME_SPECS}
    membership_violations = []
    for regime, left, right, _source in mechanics_cases():
        for role, value in (("left", left), ("right", right)):
            observed[regime].add(value)
            if not scalar_in_declared_support(regime, value):
                membership_violations.append(
                    {"regime": regime, "role": role, "value": value}
                )
    observed_fit = set().union(*(observed[name] for name in fit_regimes))
    observed_value_ood = set().union(*(observed[name] for name in value_ood_regimes))
    observed_intersection = sorted(observed_fit & observed_value_ood)
    declared_disjoint = all(not rows for rows in declared_intersections.values())
    return {
        "declared_intersections": declared_intersections,
        "declared_regime_supports_valid": declared_validity,
        "declared_fit_value_ood_disjoint": declared_disjoint,
        "observed_fit_scalar_count": len(observed_fit),
        "observed_value_ood_scalar_count": len(observed_value_ood),
        "observed_fit_value_ood_intersection": observed_intersection,
        "operand_membership_violations": membership_violations,
        "valid": (
            declared_disjoint
            and all(declared_validity.values())
            and not observed_intersection
            and not membership_violations
        ),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def drs_r3_localization_audit(path: Path = CANONICAL_R3_PATH) -> dict[str, Any]:
    """Bind post-hoc site-reach localization to the immutable r3 bytes."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError("canonical r3 artifact is not a regular non-symlink file")
    digest = sha256_file(path)
    if digest != CANONICAL_R3_SHA256:
        raise ContractError("canonical r3 artifact hash mismatch")
    document = json.loads(path.read_bytes())
    records = document.get("records")
    if (
        document.get("audit") != "drs_causal_cycle_post_drs_v3"
        or not isinstance(records, list)
        or len(records) != 50
        or document.get("decision", {}).get("mechanically_valid") is not True
    ):
        raise ContractError("canonical r3 artifact structure mismatch")
    reached = [
        row
        for row in records
        if any(
            event.get("field") == "digit"
            for event in row.get("arms", {}).get("both", {}).get("fired", [])
        )
    ]
    reached_exact = sum(row["arms"]["both"]["exact"] is True for row in reached)
    carry_reached = sum(
        any(event.get("field") == "carry" for event in row["arms"]["both"]["fired"])
        for row in records
    )
    return {
        "artifact_sha256": digest,
        "records": len(records),
        "carry_site_reached": carry_reached,
        "digit_site_reached": len(reached),
        "digit_site_reached_and_full_state_exact": reached_exact,
        "conditional_exact_rate": reached_exact / len(reached) if reached else None,
        "post_hoc": True,
        "selection_biased": True,
        "claim_boundary": (
            "Secondary localization only; it cannot alter the locked r3 decisions or "
            "establish carry as the sole defect."
        ),
    }


def mechanics_board_sha256() -> str:
    rows = [
        {
            "left": left,
            "operation": source.operation,
            "regime": regime,
            "right": right,
            "width": source.width,
        }
        for regime, left, right, source in mechanics_cases()
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def ood_rollout_audit() -> dict[str, Any]:
    by_regime = {regime: {"cases": 0, "exact": 0} for regime in REGIME_SPECS}
    history_independent = 0
    packet_schema_stable = 0
    for regime, _left, _right, source in mechanics_cases():
        observed_a: list[tuple[int, str, int]] = []
        observed_b: list[tuple[int, str, int]] = []

        def observer_a(index: int, emission: Emission) -> tuple[str, int]:
            observed_a.append((index, emission.kind, emission.value))
            return ("ignored-a", index)

        def observer_b(index: int, emission: Emission) -> dict[str, Any]:
            observed_b.append((index, emission.kind, (emission.value + 7) % 10))
            return {"fake_kv": [9] * (index + 1)}

        rollout_a = rollout(source, observer=observer_a)
        rollout_b = rollout(source, observer=observer_b)
        trace_a = emission_trace(rollout_a)
        trace_b = emission_trace(rollout_b)
        expected = oracle_trace(source)
        row = by_regime[regime]
        row["cases"] += 1
        row["exact"] += int(trace_a == expected)
        history_independent += int(trace_a == trace_b)
        packet_schema_stable += int(
            all(
                packet_schema_audit(type(record.after))["valid"] for record in rollout_a
            )
        )
    return {
        "board_sha256": mechanics_board_sha256(),
        "balance": mechanics_balance_audit(),
        "scalar_support": mechanics_scalar_support_audit(),
        "cases": len(mechanics_cases()),
        "exact": sum(row["exact"] for row in by_regime.values()),
        "history_independent": history_independent,
        "packet_schema_stable": packet_schema_stable,
        "by_regime": by_regime,
        "observer_outputs_are_not_causal": True,
    }


def local_table_collapse_audit() -> dict[str, Any]:
    table = {}
    exact = 0
    for key in iter_local_keys():
        result = position_blind_local_operator(*key)
        name = ":".join(map(str, key))
        table[name] = [result.digit, result.next_carry]
    for key in iter_local_keys():
        result = table[":".join(map(str, key))]
        exact += int(LocalResult(*result) == position_blind_local_operator(*key))
    return {
        "entries": len(table),
        "lookup_exact": exact,
        "table_sha256": hashlib.sha256(canonical_json_bytes(table)).hexdigest(),
        "finite_lookup_table_extensionally_equivalent": exact == LOCAL_TABLE_ENTRIES,
        "minimum_raw_output_bits": LOCAL_TABLE_ENTRIES * math.ceil(math.log2(20)),
    }


def minimal_packet_witnesses() -> dict[str, Any]:
    carry_zero = position_blind_local_operator("add", 0, 0, 0)
    carry_one = position_blind_local_operator("add", 0, 0, 1)
    source = Source("add", (1, 2), (0, 0))
    cursor_zero = address_source(source, 0)
    cursor_one = address_source(source, 1)
    final = Packet(source.width, 1, PHASE_FINAL)
    final_record = fcrc_step(source, final)
    halt_rejected = False
    try:
        fcrc_step(source, Packet(source.width, 1, PHASE_HALT))
    except ContractError:
        halt_rejected = True
    return {
        "carry_required": carry_zero != carry_one,
        "cursor_required": cursor_zero != cursor_one,
        "phase_required": (
            final_record.emission == Emission("terminal_carry", 1)
            and final_record.after.phase == PHASE_HALT
            and halt_rejected
        ),
        "carry_witness": {
            "carry_0": [carry_zero.digit, carry_zero.next_carry],
            "carry_1": [carry_one.digit, carry_one.next_carry],
        },
        "phase_witness": {
            "final_emission": [
                final_record.emission.kind,
                final_record.emission.value,
            ],
            "final_transitions_to_halt": final_record.after.phase == PHASE_HALT,
            "halted_step_rejected": halt_rejected,
        },
    }


def resource_accounting() -> dict[str, Any]:
    widths = (2, 4, 6, 8)
    return {
        "cpu_positive_is_hardcoded_oracle": True,
        "cpu_trainable_parameters": 0,
        "packet_fields": list(PACKET_FIELDS),
        "packet_field_count": len(PACKET_FIELDS),
        "packet_bits_by_width": {str(width): packet_bits(width) for width in widths},
        "packet_state_cardinality_by_width": {
            str(width): (width + 1) * 2 * len(PHASES) for width in widths
        },
        "read_only_source_digit_symbols_by_width": {
            str(width): 2 * width for width in widths
        },
        "read_only_source_control_symbols": 2,
        "result_tape_bits": 0,
        "generated_token_kv_causal_bits": 0,
        "hardcoded_address_calls_by_width": {str(width): width for width in widths},
        "hardcoded_decimal_calls_by_width": {str(width): width for width in widths},
        "hardcoded_actuator_calls_by_width": {
            str(width): width + 1 for width in widths
        },
        "hardcoded_control_transitions_by_width": {
            str(width): width + 1 for width in widths
        },
        "hardcoded_learned_module_substitutes_by_width": {
            str(width): 3 * width + 1 for width in widths
        },
        "external_execution_calls_by_width": {
            str(width): 4 * width + 2 for width in widths
        },
        "emitted_symbols_by_width": {str(width): width + 1 for width in widths},
        "sequential_steps_by_width": {str(width): width + 1 for width in widths},
        "fixed_width_conditional_fst": True,
        "ordinary_rnn_simulable": True,
        "lookup_table_simulable": True,
        "cpu_positive_disqualified_as_learned_reasoning": True,
        "required_neural_controls": list(REQUIRED_NEURAL_CONTROLS),
    }


def structural_surface_audit() -> dict[str, Any]:
    address_parameters = tuple(inspect.signature(address_source).parameters)
    local_parameters = tuple(
        inspect.signature(position_blind_local_operator).parameters
    )
    late_actuator_parameters = tuple(
        inspect.signature(late_residual_actuator).parameters
    )
    terminal_actuator_parameters = tuple(
        inspect.signature(terminal_actuator).parameters
    )
    step_parameters = tuple(inspect.signature(fcrc_step).parameters)
    forbidden = {
        "generated_history",
        "history",
        "kv",
        "prefix",
        "result_prefix",
        "result_tape",
        "terminal",
        "width",
    }
    return {
        "address_parameters": list(address_parameters),
        "address_exact_surface": address_parameters == ADDRESS_SOURCE_PARAMETERS,
        "local_operator_parameters": list(local_parameters),
        "local_operator_exact_surface": local_parameters == LOCAL_OPERATOR_PARAMETERS,
        "late_actuator_parameters": list(late_actuator_parameters),
        "late_actuator_exact_surface": late_actuator_parameters
        == LATE_ACTUATOR_PARAMETERS,
        "terminal_actuator_parameters": list(terminal_actuator_parameters),
        "terminal_actuator_exact_surface": terminal_actuator_parameters
        == TERMINAL_ACTUATOR_PARAMETERS,
        "step_parameters": list(step_parameters),
        "step_has_no_generated_or_result_state": forbidden.isdisjoint(step_parameters),
        "packet_schema": packet_schema_audit(Packet),
        "growing_packet_schema": packet_schema_audit(GrowingPacket),
        "fixed_operator_dependencies": fixed_operator_dependency_audit(),
    }


def run_audit() -> dict[str, Any]:
    positive = context_invariance_audit(reference_context_operator)
    terminal_negative = context_invariance_audit(terminal_zero_leak)
    cursor_negative = context_invariance_audit(cursor_leak)
    width_6_negative = context_invariance_audit(width_6_leak)
    width_8_negative = context_invariance_audit(width_8_leak)
    prefix_negative = context_invariance_audit(result_prefix_leak)
    history_negative = context_invariance_audit(generated_history_leak)
    no_go = terminal_carry_no_go_audit()
    terminal_update = terminal_update_audit()
    two_step = autonomous_two_step_audit()
    ood = ood_rollout_audit()
    collapse = local_table_collapse_audit()
    surface = structural_surface_audit()
    minimality = minimal_packet_witnesses()
    resources = resource_accounting()
    r3 = drs_r3_localization_audit()

    gates = {
        "packet_schema_exact": surface["packet_schema"]["valid"],
        "packet_growth_negative_rejected": not surface["growing_packet_schema"][
            "valid"
        ],
        "address_surface_exact": surface["address_exact_surface"],
        "local_operator_surface_exact": surface["local_operator_exact_surface"],
        "actuator_surfaces_exact": surface["late_actuator_exact_surface"]
        and surface["terminal_actuator_exact_surface"],
        "no_generated_or_result_state_surface": surface[
            "step_has_no_generated_or_result_state"
        ],
        "fixed_operator_dependency_surface_exact": surface[
            "fixed_operator_dependencies"
        ]["valid"],
        "positive_context_invariance_exact": positive["total_violations"] == 0,
        "terminal_leak_negative_detected": terminal_negative["violations_by_variant"][
            "terminal"
        ]
        > 0,
        "cursor_leak_negative_detected": cursor_negative["violations_by_variant"][
            "cursor"
        ]
        > 0,
        "width_6_leak_negative_detected": width_6_negative["violations_by_variant"][
            "width_6"
        ]
        > 0,
        "width_8_leak_negative_detected": width_8_negative["violations_by_variant"][
            "width_8"
        ]
        > 0,
        "result_prefix_leak_negative_detected": prefix_negative[
            "violations_by_variant"
        ]["result_prefix"]
        > 0,
        "generated_history_leak_negative_detected": history_negative[
            "violations_by_variant"
        ]["history"]
        > 0,
        "terminal_no_go_witness_exact": no_go[
            "observationally_indistinguishable_on_training_support"
        ]
        and no_go["different_on_every_omitted_terminal_carry_cell"],
        "terminal_update_is_position_blind": all(
            terminal_update[name] == LOCAL_TABLE_ENTRIES
            for name in (
                "terminal_carry_updates_exact",
                "nonterminal_carry_updates_exact",
                "terminal_nonterminal_emissions_equal",
            )
        ),
        "autonomous_two_step_exact": two_step["exact"] == two_step["cases"],
        "width_value_ood_rollout_exact": ood["exact"] == ood["cases"]
        and all(row["exact"] == row["cases"] for row in ood["by_regime"].values()),
        "mechanics_regimes_balanced": ood["balance"]["balanced"],
        "fit_value_ood_scalar_supports_disjoint": ood["scalar_support"]["valid"],
        "generated_history_sham_invariant": ood["history_independent"] == ood["cases"],
        "packet_schema_stable_over_rollout": ood["packet_schema_stable"]
        == ood["cases"],
        "lookup_collapse_admitted": collapse[
            "finite_lookup_table_extensionally_equivalent"
        ],
        "three_packet_fields_are_behaviorally_required": all(
            minimality[name]
            for name in ("carry_required", "cursor_required", "phase_required")
        ),
        "cpu_oracle_is_disqualified": resources[
            "cpu_positive_disqualified_as_learned_reasoning"
        ],
        "carry_only_control_declared": "carry_only_rank8_writer_reader"
        in resources["required_neural_controls"],
        "no_result_tape_or_generated_kv_accounted": resources["result_tape_bits"] == 0
        and resources["generated_token_kv_causal_bits"] == 0,
        "canonical_r3_localization_bound": r3["carry_site_reached"] == 50
        and r3["digit_site_reached"] == 16
        and r3["digit_site_reached_and_full_state_exact"] == 14,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "mechanics_contract_satisfied": all(gates.values()),
        "gates": gates,
        "surface": surface,
        "terminal_carry_identifiability_no_go": no_go,
        "context_invariance": {
            "positive": positive,
            "terminal_leak_negative": terminal_negative,
            "cursor_leak_negative": cursor_negative,
            "width_6_leak_negative": width_6_negative,
            "width_8_leak_negative": width_8_negative,
            "result_prefix_leak_negative": prefix_negative,
            "generated_history_leak_negative": history_negative,
        },
        "terminal_update": terminal_update,
        "autonomous_two_step": two_step,
        "width_value_ood": ood,
        "minimal_packet_witnesses": minimality,
        "collapse": collapse,
        "resources": resources,
        "r3_post_hoc_localization": r3,
        "go_boundary": (
            "GO authorizes only an isolated learned pilot after every structural gate "
            "passes and its resource/control board is frozen; CPU GO is not a neural, "
            "capability, reasoning, or novelty result. Any failed gate is NO-GO."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(canonical_json_bytes(report).decode("ascii"), end="")
    return 0 if report["mechanics_contract_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
