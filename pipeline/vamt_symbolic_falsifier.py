#!/usr/bin/env python3
"""CPU-only mechanics falsifier for the R12 VAMT resource hypothesis.

This module contains no neural implementation. Arithmetic appears only in
offline oracle/audit functions. The exact symbolic realization consumes frozen
categorical lookup tables, copies addressed symbols, advances a fixed cursor,
and records those lookups as external symbolic execution. It cannot qualify as
model-owned reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROTOCOL = "R12-VAMT-SYMBOLIC-FALSIFIER-v2"
OPS = ("add", "sub")
DIGITS = tuple(range(10))
CARRIES = (0, 1)
BASE_PARAMETERS = 125_081_664
STRICT_TOTAL_MAXIMUM = 149_999_999

Context = tuple[str, int, int, int]
Outcome = tuple[int, int]


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def audit_local_oracle(operation: str, left: int, right: int, carry: int) -> Outcome:
    """Offline arithmetic oracle. This function is forbidden in candidate replay."""
    if operation not in OPS or left not in DIGITS or right not in DIGITS:
        raise ValueError("invalid local arithmetic context")
    if carry not in CARRIES:
        raise ValueError("carry must be zero or one")
    if operation == "add":
        total = left + right + carry
        return total % 10, total // 10
    total = left - right - carry
    return (total + 10, 1) if total < 0 else (total, 0)


def build_truth_table() -> dict[Context, Outcome]:
    return {
        (operation, left, right, carry): audit_local_oracle(
            operation, left, right, carry
        )
        for operation in OPS
        for left in DIGITS
        for right in DIGITS
        for carry in CARRIES
    }


TRUTH_TABLE = build_truth_table()


@dataclass
class RuntimeLedger:
    transition_lookups: int = 0
    cursor_shifts: int = 0
    categorical_writes: int = 0
    serializer_lookups: int = 0
    pointer_reads: int = 0
    semantic_host_arithmetic_calls: int = 0
    parser_repairs: int = 0
    verifier_calls: int = 0
    retries: int = 0
    external_execution_calls: int = 0


@dataclass(frozen=True)
class MachineResult:
    digits_lsd: tuple[int, ...]
    terminal_carry_or_borrow: int
    ledger: RuntimeLedger


class TiedDigitMachine:
    """One categorical transition table reused at every tape position."""

    def __init__(self, table: Mapping[Context, Outcome]):
        if set(table) != set(TRUTH_TABLE):
            raise ValueError("transition table must cover exactly all 400 contexts")
        self.table = dict(table)

    def run(
        self,
        operation: str,
        left_lsd: Sequence[int],
        right_lsd: Sequence[int],
    ) -> MachineResult:
        if operation not in OPS or not left_lsd or len(left_lsd) != len(right_lsd):
            raise ValueError("malformed categorical tapes")
        if any(value not in DIGITS for value in (*left_lsd, *right_lsd)):
            raise ValueError("tapes must contain decimal digit categories")
        ledger = RuntimeLedger()
        carry = 0
        output = []
        for left, right in zip(left_lsd, right_lsd, strict=True):
            ledger.pointer_reads += 2
            digit, carry = self.table[(operation, left, right, carry)]
            ledger.transition_lookups += 1
            ledger.external_execution_calls += 1
            output.append(digit)
            ledger.categorical_writes += 1
            ledger.cursor_shifts += 1
        return MachineResult(tuple(output), carry, ledger)


def audit_int_to_digits(value: int, width: int) -> tuple[int, ...]:
    """Offline conversion used only to generate exhaustive audit inputs."""
    if value < 0 or width <= 0 or value >= 10**width:
        raise ValueError("value does not fit audit width")
    return tuple((value // (10**position)) % 10 for position in range(width))


def audit_digits_to_int(digits_lsd: Sequence[int]) -> int:
    """Offline conversion used only to score a categorical candidate result."""
    return sum(int(digit) * (10**position) for position, digit in enumerate(digits_lsd))


SerializerContext = tuple[int, int, int]
SerializerOutcome = tuple[int | None, int, int]


def audit_serializer_oracle(
    seen_nonzero: int, digit: int, is_last: int
) -> SerializerOutcome:
    """Offline oracle for canonical leading-zero suppression."""
    if seen_nonzero not in (0, 1) or digit not in DIGITS or is_last not in (0, 1):
        raise ValueError("invalid serializer context")
    emit = bool(seen_nonzero or digit != 0 or is_last)
    return (digit if emit else None), int(emit), is_last


def build_serializer_table() -> dict[SerializerContext, SerializerOutcome]:
    return {
        (seen, digit, is_last): audit_serializer_oracle(seen, digit, is_last)
        for seen in (0, 1)
        for digit in DIGITS
        for is_last in (0, 1)
    }


SERIALIZER_TABLE = build_serializer_table()


class TiedSerializer:
    """Copy a categorical result tape through one shared serializer table."""

    def __init__(self, table: Mapping[SerializerContext, SerializerOutcome]):
        if set(table) != set(SERIALIZER_TABLE):
            raise ValueError("serializer table must cover exactly all 40 contexts")
        self.table = dict(table)

    def run(self, digits_lsd: Sequence[int]) -> tuple[str, RuntimeLedger]:
        if not digits_lsd:
            raise ValueError("malformed result register")
        if any(digit not in DIGITS for digit in digits_lsd):
            raise ValueError("register must contain decimal digit categories")
        ledger = RuntimeLedger()
        seen = 0
        emitted = []
        digits_msd = tuple(reversed(digits_lsd))
        for index, digit in enumerate(digits_msd):
            is_last = int(index + 1 == len(digits_msd))
            symbol, seen, halt = self.table[(seen, digit, is_last)]
            ledger.serializer_lookups += 1
            ledger.external_execution_calls += 1
            ledger.cursor_shifts += 1
            if symbol is not None:
                emitted.append(str(symbol))
                ledger.categorical_writes += 1
            if halt != is_last:
                raise ValueError("serializer halt does not match terminal relation")
        return "".join(emitted), ledger


def addressed_copy(tokens: Sequence[str], positions: Sequence[int]) -> tuple[str, ...]:
    """Pointer semantics only: select source symbols without interpreting them."""
    if any(position < 0 or position >= len(tokens) for position in positions):
        raise ValueError("pointer outside source")
    return tuple(tokens[position] for position in positions)


@dataclass(frozen=True)
class OperandSpan:
    start: int
    end: int


def read_operand_lsd(
    tokens: Sequence[str], span: OperandSpan, width: int
) -> tuple[str, ...]:
    """Fixed endpoint/cursor semantics; no integer conversion occurs."""
    if width <= 0 or not 0 <= span.start <= span.end < len(tokens):
        raise ValueError("invalid inclusive operand span")
    if any(tokens[position] not in tuple(map(str, DIGITS)) for position in range(span.start, span.end + 1)):
        raise ValueError("operand span contains a nondigit symbol")
    cursor = span.end
    output = []
    for _ in range(width):
        if cursor >= span.start:
            output.append(tokens[cursor])
            cursor -= 1
        else:
            output.append("0")
    return tuple(output)


def exact_local_coverage(table: Mapping[Context, Outcome]) -> dict:
    mismatches = []
    for context, expected in TRUTH_TABLE.items():
        actual = table.get(context)
        if actual != expected:
            mismatches.append({"context": list(context), "expected": expected, "actual": actual})
    return {
        "contexts": len(TRUTH_TABLE),
        "correct": len(TRUTH_TABLE) - len(mismatches),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def exhaustive_small_width_replay(machine: TiedDigitMachine) -> dict:
    checks = 0
    rejected_negative_subtractions = 0
    failures = []
    for width in (1, 2):
        limit = 10**width
        for left in range(limit):
            left_digits = audit_int_to_digits(left, width)
            for right in range(limit):
                right_digits = audit_int_to_digits(right, width)
                for operation in OPS:
                    if operation == "sub" and left < right:
                        rejected_negative_subtractions += 1
                        continue
                    result = machine.run(operation, left_digits, right_digits)
                    observed = audit_digits_to_int(result.digits_lsd)
                    if operation == "add":
                        expected = left + right
                        observed += result.terminal_carry_or_borrow * limit
                        correct = observed == expected
                    else:
                        expected = left - right
                        correct = (
                            observed == expected
                            and result.terminal_carry_or_borrow == 0
                        )
                    checks += 1
                    if not correct and len(failures) < 16:
                        failures.append(
                            {
                                "width": width,
                                "operation": operation,
                                "left": left,
                                "right": right,
                                "observed": observed,
                                "carry_or_borrow": result.terminal_carry_or_borrow,
                            }
                        )
    return {
        "admitted_checks": checks,
        "rejected_negative_subtractions": rejected_negative_subtractions,
        "failures": failures,
        "signed_subtraction_supported": False,
        "pass": not failures and rejected_negative_subtractions > 0,
    }


def induction_cell_certificate(table: Mapping[Context, Outcome], maximum_width: int = 64) -> dict:
    """Check every local context at every position; tying makes position irrelevant."""
    if maximum_width <= 0:
        raise ValueError("maximum width must be positive")
    checks = 0
    for width in range(1, maximum_width + 1):
        for _position in range(width):
            for context, expected in TRUTH_TABLE.items():
                if table[context] != expected:
                    return {
                        "maximum_width": maximum_width,
                        "checks": checks + 1,
                        "failing_context": list(context),
                        "pass": False,
                    }
                checks += 1
    return {
        "maximum_width": maximum_width,
        "checks": checks,
        "expected_checks": 400 * maximum_width * (maximum_width + 1) // 2,
        "pass": True,
    }


def untied_nonidentifiability_witness(observed_positions: int = 4) -> dict:
    """Construct a position-untied machine that is perfect on train positions only."""
    if observed_positions <= 0:
        raise ValueError("observed positions must be positive")
    tables = [dict(TRUTH_TABLE) for _ in range(observed_positions + 1)]
    poisoned_context: Context = ("add", 0, 0, 0)
    tables[-1][poisoned_context] = (1, 0)
    observed_correct = all(
        tables[position][context] == expected
        for position in range(observed_positions)
        for context, expected in TRUTH_TABLE.items()
    )
    unseen_fails = tables[-1][poisoned_context] != TRUTH_TABLE[poisoned_context]
    return {
        "observed_positions": observed_positions,
        "observed_contexts": observed_positions * len(TRUTH_TABLE),
        "untied_context_parameters_at_confirmation_width": (
            observed_positions + 1
        )
        * len(TRUTH_TABLE),
        "tied_context_parameters": len(TRUTH_TABLE),
        "poisoned_unseen_context": list(poisoned_context),
        "observed_positions_exact": observed_correct,
        "first_unseen_position_fails": unseen_fails,
        "pass": observed_correct and unseen_fails,
    }


def pointer_equivariance_certificate() -> dict:
    source = ("Alice", "has", "1", "2", "3", ";", "Bob", "adds", "4", "5")
    span = OperandSpan(2, 4)
    base = read_operand_lsd(source, span, width=5)

    renamed = ("Carol",) + source[1:6] + ("Dana",) + source[7:]
    rename_ok = read_operand_lsd(renamed, span, width=5) == base

    replacement = source[:2] + ("9", "8", "7", "6") + source[5:]
    replacement_span = OperandSpan(2, 5)
    replacement_ok = read_operand_lsd(replacement, replacement_span, width=5) == (
        "6", "7", "8", "9", "0"
    )

    inserted = ("Note", ":", "irrelevant", ".") + source
    shifted = OperandSpan(span.start + 4, span.end + 4)
    insertion_ok = read_operand_lsd(inserted, shifted, width=5) == base

    second_span = OperandSpan(8, 9)
    unequal_width_ok = read_operand_lsd(source, second_span, width=5) == (
        "5", "4", "0", "0", "0"
    )

    return {
        "base_copy": list(base),
        "inclusive_span": asdict(span),
        "read_direction": "end_to_start",
        "post_span_symbol": "0",
        "entity_rename_invariant": rename_ok,
        "literal_replacement_redirects_content": replacement_ok,
        "neutral_prefix_relocates_pointer": insertion_ok,
        "unequal_width_zero_padding": unequal_width_ok,
        "pass": rename_ok and replacement_ok and insertion_ok and unequal_width_ok,
    }


def serializer_certificate(serializer: TiedSerializer, maximum_exhaustive_width: int = 4) -> dict:
    atomic = all(
        serializer.table[context] == expected
        for context, expected in SERIALIZER_TABLE.items()
    )
    checks = 0
    failures = []
    for width in range(1, maximum_exhaustive_width + 1):
        for value in range(10**width):
            digits = audit_int_to_digits(value, width)
            text, _ledger = serializer.run(digits)
            expected = str(value)
            checks += 1
            if text != expected and len(failures) < 16:
                failures.append({"width": width, "value": value, "text": text})
    terminal_carry_text, _ = serializer.run((3, 9, 5, 7, 0, 1))
    terminal_carry_ok = terminal_carry_text == "107593"
    return {
        "atomic_contexts": len(SERIALIZER_TABLE),
        "atomic_exact": atomic,
        "exhaustive_checks": checks,
        "failures": failures,
        "terminal_carry_example": terminal_carry_text,
        "terminal_carry_exact": terminal_carry_ok,
        "pass": atomic and not failures and terminal_carry_ok,
    }


def parameter_ledger() -> dict:
    lora = 12 * 64 * 10_176
    compiler_front = (
        2 * 576
        + (576 * 1024 + 1024)
        + (1024 * 1024 + 1024)
        + (1024 * 16 + 16)
        + (1024 * 8 + 8)
        + 1024 * 256
    )
    compiler_decoder = (
        (3 * 512 * 1024 + 3 * 512 * 512 + 6 * 512)
        + (1024 * 512 + 512)
        + (512 * 16 + 16)
        + (512 * 8 + 8)
        + (512 * 2 + 2)
        + 512 * 256
    )
    executor = (
        576 * 128
        + (16 + 2 + 8 + 32) * 128
        + 2 * 512
        + (512 * 2048 + 2048)
        + (2048 * 2048 + 2048)
        + (2048 * 32 + 32)
    )
    serializer = (
        13 * 256
        + (3 * 512 * 256 + 3 * 512 * 512 + 6 * 512)
        + (512 * 576 + 576)
        + (512 * 2 + 2)
        + (1024 * 512 + 512)
    )
    maximum_components = {
        "late_lora": lora,
        "compiler_front": compiler_front,
        "compiler_decoder": compiler_decoder,
        "executor": executor,
        "serializer": serializer,
    }
    maximum_additional = sum(maximum_components.values())
    maximum_total = BASE_PARAMETERS + maximum_additional
    minimal_components = {
        "r4_style_compiler_and_transition_table": 300_493,
        "boundary_head": 771,
        "digit_key": 32_768,
        "slot_start_end_queries": 32_768,
        "event_start_end_queries": 65_536,
        "serializer": 1_741,
    }
    minimal_additional = sum(minimal_components.values())
    minimal_total = BASE_PARAMETERS + minimal_additional
    return {
        "base_parameters": BASE_PARAMETERS,
        "strict_total_maximum": STRICT_TOTAL_MAXIMUM,
        "minimal": {
            "components": minimal_components,
            "additional_parameters": minimal_additional,
            "total_parameters": minimal_total,
            "strict_headroom": STRICT_TOTAL_MAXIMUM - minimal_total,
        },
        "maximum": {
            "components": maximum_components,
            "additional_parameters": maximum_additional,
            "total_parameters": maximum_total,
            "strict_headroom": STRICT_TOTAL_MAXIMUM - maximum_total,
        },
        "pass": (
            minimal_total <= STRICT_TOTAL_MAXIMUM
            and maximum_total <= STRICT_TOTAL_MAXIMUM
        ),
    }


def bounded_resource_ledger() -> dict:
    retained = {
        "program_opcodes_uint8": 8,
        "program_starts_uint16": 16,
        "program_ends_uint16": 16,
        "pc_phase_uint8": 2,
        "source_cursor_uint16": 2,
        "carry_invalid_halt_uint8": 3,
        "accumulator_uint8": 17,
        "serializer_cursor_seen_halt_uint8": 3,
    }
    output = {"token_ids_uint16": 34, "emit_mask_uint8": 17}
    return {
        "bounds": {"source_tokens": 256, "instructions": 8, "width": 16},
        "immutable_source_bytes": 512,
        "retained_components": retained,
        "retained_program_and_private_state_bytes": sum(retained.values()),
        "output_components": output,
        "output_bytes": sum(output.values()),
        "fixed_executor_cycles": 8 * 16,
        "minimal_serializer_matrix_macs": 17 * (13 * 64 + 64 * 13),
        "maximum_executor_matrix_macs": 8
        * 16
        * (512 * 2048 + 2048 * 2048 + 2048 * 32),
        "structured_target_bits_must_be_charged": True,
        "peak_temporary_bytes_deferred_to_executable_preregistration": True,
        "pass": sum(retained.values()) == 67 and sum(output.values()) == 51,
    }


def finite_mealy_certificate(maximum_width: int = 64) -> dict:
    rows = []
    for width in (1, 2, 4, 8, 16, 32, maximum_width):
        if width <= 0:
            continue
        # Upper bound: cursor, carry, halt, and an 11-symbol result tape
        # (ten digits plus blank). Source/program are immutable inputs.
        state_upper = (width + 1) * 2 * 2 * (11**width)
        retained_bits = math.ceil(math.log2(state_upper))
        rows.append(
            {
                "width": width,
                "state_upper_bound": state_upper,
                "retained_bits_upper_bound": retained_bits,
                "transition_depth": width,
                "serializer_depth": width + 1,
            }
        )
    return {
        "machine_class": "finite deterministic Mealy transducer",
        "bounded_unrolling": "finite acyclic circuit",
        "rows": rows,
        "novel_primitive_claim_allowed": False,
        "pass": True,
    }


def symbolic_runtime_accounting(machine: TiedDigitMachine) -> dict:
    result = machine.run("add", (9, 9, 0), (8, 0, 0))
    ledger = asdict(result.ledger)
    semantic_forbidden = (
        "semantic_host_arithmetic_calls",
        "parser_repairs",
        "verifier_calls",
        "retries",
    )
    return {
        "ledger": ledger,
        "semantic_forbidden_zero": {
            name: ledger[name] == 0 for name in semantic_forbidden
        },
        "fixed_runtime_nonzero": {
            "transition_lookups": ledger["transition_lookups"] > 0,
            "cursor_shifts": ledger["cursor_shifts"] > 0,
            "categorical_writes": ledger["categorical_writes"] > 0,
            "pointer_reads": ledger["pointer_reads"] > 0,
        },
        "external_symbolic_execution_counted": (
            ledger["external_execution_calls"] == ledger["transition_lookups"]
            and ledger["external_execution_calls"] > 0
        ),
        "future_neural_host_boundary_proven": False,
        "claim_boundary": (
            "The Python truth-table lookup is external symbolic execution. "
            "This accounting proves no model-owned inference property."
        ),
        "pass": (
            all(ledger[name] == 0 for name in semantic_forbidden)
            and ledger["external_execution_calls"] == ledger["transition_lookups"]
            and ledger["external_execution_calls"] > 0
        ),
    }


def build_report() -> dict:
    machine = TiedDigitMachine(TRUTH_TABLE)
    serializer = TiedSerializer(SERIALIZER_TABLE)
    sections = {
        "local_coverage": exact_local_coverage(TRUTH_TABLE),
        "small_width_replay": exhaustive_small_width_replay(machine),
        "induction_cells": induction_cell_certificate(TRUTH_TABLE),
        "untied_witness": untied_nonidentifiability_witness(),
        "pointer_equivariance": pointer_equivariance_certificate(),
        "serializer": serializer_certificate(serializer),
        "parameter_ledger": parameter_ledger(),
        "bounded_resource_ledger": bounded_resource_ledger(),
        "mealy_collapse": finite_mealy_certificate(),
        "symbolic_runtime_accounting": symbolic_runtime_accounting(machine),
    }
    report = {
        "protocol": PROTOCOL,
        "claim_boundary": (
            "A pass proves bounded categorical mechanics and the named tied-vs-untied "
            "identification witness only. It does not authorize neural code, fitting, "
            "reasoning, novelty, or accelerator work."
        ),
        "schema_v1_positive_authority": False,
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
            raise SystemExit("refusing to overwrite {}".format(args.out))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
    print(payload.decode(), end="")
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
