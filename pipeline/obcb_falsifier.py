#!/usr/bin/env python3
"""Exact CPU falsifier for Operator-Balanced Commit Bisimulation.

This module is an exhaustive oracle mechanics board, not a neural learner. It
checks the two-state decimal carry quotient, a one-bit source-deletion boundary,
and named shortcut controls. The arithmetic oracle is verification machinery
and is forbidden from any future Shohin inference path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import itertools
import json
import os
from pathlib import Path
import stat
from typing import Any, Protocol


PROTOCOL_ID = "R12-OBCB-1-CPU-v1"
SCHEMA_VERSION = 1
OPERATIONS = ("add", "sub")
LOCAL_EVENT_COUNT = 200
LOCAL_CELL_COUNT = 400
TWO_STEP_EDGE_COUNT = 40_000
BALANCED_PAIR_REPETITIONS = {"K0": 2, "I": 9, "K1": 2}
BALANCED_PAIRS_PER_OPERATOR = 90
BALANCED_ROWS_PER_OPERATOR = 180
BALANCED_TOTAL_ROWS = 1_080
RESOURCE_VECTOR = {
    "added_trainable_parameters": 0,
    "retained_dynamic_bits": 1,
    "source_bytes_after_commit": 0,
    "result_history_symbols": 0,
    "hidden_step_bits": 0,
    "external_memory_bytes": 0,
    "external_execution_calls_at_inference": 0,
    "additional_inference_steps": 0,
}


class ContractError(ValueError):
    """The frozen OBCB mechanics contract was violated."""


class CarryMap(str, Enum):
    """The complete transformation monoid on the decimal carry bit."""

    K0 = "K0"
    IDENTITY = "I"
    K1 = "K1"

    def apply(self, bit: int) -> int:
        require_bit(bit, "carry-map input")
        if self is CarryMap.K0:
            return 0
        if self is CarryMap.K1:
            return 1
        return bit


@dataclass(frozen=True, slots=True)
class DigitEvent:
    """One visible local decimal event; it is deleted after commitment."""

    operation: str
    left_digit: int
    right_digit: int

    def __post_init__(self) -> None:
        if type(self.operation) is not str or self.operation not in OPERATIONS:
            raise ContractError("operation must be the plain string add or sub")
        require_digit(self.left_digit, "left digit")
        require_digit(self.right_digit, "right digit")


@dataclass(frozen=True, slots=True)
class CommitBit:
    """The complete semantic object permitted to cross a cycle boundary."""

    bit: bool

    def __post_init__(self) -> None:
        if type(self.bit) is not bool:
            raise ContractError("committed state must be one plain boolean")


@dataclass(frozen=True, slots=True)
class ResultHistoryPacket:
    """Negative control: correct carry plus an illegal prior result digit."""

    bit: bool
    prior_digit: int


@dataclass(frozen=True, slots=True)
class HiddenStepPacket:
    """Negative control: correct carry plus an illegal hidden step counter."""

    bit: bool
    step: int


@dataclass(frozen=True, slots=True)
class LocalResult:
    digit: int
    next_carry: int

    def __post_init__(self) -> None:
        require_digit(self.digit, "result digit")
        require_bit(self.next_carry, "next carry")


@dataclass(frozen=True, slots=True)
class StepOutput:
    digit: int
    packet: object

    def __post_init__(self) -> None:
        require_digit(self.digit, "emitted digit")


class Machine(Protocol):
    name: str

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        """Apply one visible event to one committed state."""


MachineFactory = Callable[[], Machine]


def require_digit(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ContractError(f"{label} must be a plain decimal digit")
    return value


def require_bit(value: object, label: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ContractError(f"{label} must be a plain integer bit")
    return value


def packet_bit(packet: object) -> int:
    bit = getattr(packet, "bit", None)
    if type(bit) is not bool:
        raise ContractError("packet does not expose one plain boolean bit")
    return int(bit)


def exact_packet_contract(packet: object) -> bool:
    """Inspect structure; do not trust a self-attested resource ledger."""

    return (
        type(packet) is CommitBit
        and tuple(field.name for field in fields(packet)) == ("bit",)
        and CommitBit.__slots__ == ("bit",)
        and not hasattr(packet, "__dict__")
        and type(packet.bit) is bool
    )


def machine_retained_fields(machine: object) -> tuple[str, ...]:
    """Return populated instance slots or dict keys after a commit."""

    populated: list[str] = []
    instance_dict = getattr(machine, "__dict__", None)
    if isinstance(instance_dict, dict):
        populated.extend(sorted(instance_dict))
    for cls in type(machine).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"}:
                continue
            if hasattr(machine, name) and name not in populated:
                populated.append(name)
    return tuple(sorted(populated))


def iter_events(operation: str | None = None) -> Iterable[DigitEvent]:
    operations = OPERATIONS if operation is None else (operation,)
    for op, left, right in itertools.product(operations, range(10), range(10)):
        yield DigitEvent(op, left, right)


def iter_local_cells() -> Iterable[tuple[DigitEvent, int]]:
    for event in iter_events():
        for carry in (0, 1):
            yield event, carry


def classify_event(event: DigitEvent) -> CarryMap:
    if event.operation == "add":
        total = event.left_digit + event.right_digit
        if total <= 8:
            return CarryMap.K0
        if total == 9:
            return CarryMap.IDENTITY
        return CarryMap.K1
    if event.left_digit > event.right_digit:
        return CarryMap.K0
    if event.left_digit == event.right_digit:
        return CarryMap.IDENTITY
    return CarryMap.K1


def oracle_transition(event: DigitEvent, carry: int) -> LocalResult:
    """Independent exact decimal oracle used only by the CPU falsifier."""

    require_bit(carry, "oracle carry")
    if event.operation == "add":
        total = event.left_digit + event.right_digit + carry
        return LocalResult(total % 10, int(total >= 10))
    total = event.left_digit - event.right_digit - carry
    return LocalResult(total % 10, int(total < 0))


def compose_maps(outer: CarryMap, inner: CarryMap) -> CarryMap:
    truth_table = tuple(outer.apply(inner.apply(bit)) for bit in (0, 1))
    by_truth_table = {
        (0, 0): CarryMap.K0,
        (0, 1): CarryMap.IDENTITY,
        (1, 1): CarryMap.K1,
    }
    try:
        return by_truth_table[truth_table]
    except KeyError as error:
        raise ContractError("carry maps escaped the frozen monoid") from error


class OBCBMachine:
    """Exact stateless realization of the one-bit commit boundary."""

    __slots__ = ()
    name = "obcb_1"

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        result = oracle_transition(event, packet_bit(packet))
        return StepOutput(result.digit, CommitBit(bool(result.next_carry)))


class CommitIgnoringMachine:
    """Negative control that never reads the committed bit."""

    __slots__ = ()
    name = "commit_ignoring"

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        del packet
        result = oracle_transition(event, 0)
        return StepOutput(result.digit, CommitBit(bool(result.next_carry)))


class ShuffledStateMachine:
    """Negative control with deterministic pairwise label swaps."""

    __slots__ = ()
    name = "shuffled_state"

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        result = oracle_transition(event, packet_bit(packet))
        swap = (event.left_digit * 10 + event.right_digit) % 2
        committed = result.next_carry ^ swap
        return StepOutput(result.digit, CommitBit(bool(committed)))


class ResultHistoryMachine:
    """Favorable exact control that illegally retains the prior digit."""

    __slots__ = ()
    name = "result_history"

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        result = oracle_transition(event, packet_bit(packet))
        retained = ResultHistoryPacket(bool(result.next_carry), result.digit)
        return StepOutput(result.digit, retained)


class HiddenStepMachine:
    """Favorable exact control that illegally retains a hidden counter."""

    __slots__ = ()
    name = "hidden_step"

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        prior_step = packet.step if type(packet) is HiddenStepPacket else 0
        result = oracle_transition(event, packet_bit(packet))
        retained = HiddenStepPacket(bool(result.next_carry), prior_step + 1)
        return StepOutput(result.digit, retained)


class StaleSourceReplayMachine:
    """Favorable exact control that recomputes state from retained source."""

    __slots__ = ("calls", "source_carry", "source_event")
    name = "stale_source_replay"

    def __init__(self) -> None:
        self.calls = 0
        self.source_carry = 0
        self.source_event: DigitEvent | None = None

    def step(self, event: DigitEvent, packet: object) -> StepOutput:
        incoming = packet_bit(packet)
        if self.calls:
            if self.source_event is None:
                raise ContractError("stale-source control lost its retained source")
            incoming = oracle_transition(
                self.source_event, self.source_carry
            ).next_carry
        result = oracle_transition(event, incoming)
        if not self.calls:
            self.source_event = event
            self.source_carry = packet_bit(packet)
        self.calls += 1
        return StepOutput(result.digit, CommitBit(bool(result.next_carry)))


CONTROL_FACTORIES: dict[str, MachineFactory] = {
    "commit_ignoring": CommitIgnoringMachine,
    "stale_source_replay": StaleSourceReplayMachine,
    "shuffled_state": ShuffledStateMachine,
    "result_history": ResultHistoryMachine,
    "hidden_step": HiddenStepMachine,
}


def monoid_audit() -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for operation in OPERATIONS:
        operation_counts = Counter(
            classify_event(event).value for event in iter_events(operation)
        )
        counts[operation] = {
            carry_map.value: operation_counts[carry_map.value] for carry_map in CarryMap
        }

    table = {
        outer.value: {
            inner.value: compose_maps(outer, inner).value for inner in CarryMap
        }
        for outer in CarryMap
    }
    closure = all(
        value in {member.value for member in CarryMap}
        for row in table.values()
        for value in row.values()
    )
    identity = all(
        compose_maps(CarryMap.IDENTITY, member) is member
        and compose_maps(member, CarryMap.IDENTITY) is member
        for member in CarryMap
    )
    associative = all(
        compose_maps(compose_maps(left, middle), right)
        is compose_maps(left, compose_maps(middle, right))
        for left, middle, right in itertools.product(CarryMap, repeat=3)
    )
    return {
        "event_counts": counts,
        "composition_table_outer_after_inner": table,
        "closure": closure,
        "identity": identity,
        "associativity_triples_checked": 27,
        "associative": associative,
        "pass": counts
        == {
            "add": {"K0": 45, "I": 10, "K1": 45},
            "sub": {"K0": 45, "I": 10, "K1": 45},
        }
        and closure
        and identity
        and associative,
    }


def balanced_allocation_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paired_counts: Counter[tuple[str, str]] = Counter()
    integrity = True
    for event in iter_events():
        operator = classify_event(event).value
        repeats = BALANCED_PAIR_REPETITIONS[operator]
        for repetition in range(repeats):
            pair_id = (
                f"{event.operation}:{event.left_digit}:{event.right_digit}:{repetition}"
            )
            pair_rows = []
            for carry in (0, 1):
                expected = oracle_transition(event, carry)
                row = {
                    "pair_id": pair_id,
                    "operation": event.operation,
                    "left_digit": event.left_digit,
                    "right_digit": event.right_digit,
                    "incoming_carry": carry,
                    "digit": expected.digit,
                    "next_carry": expected.next_carry,
                    "carry_map": operator,
                }
                rows.append(row)
                pair_rows.append(row)
            integrity &= (
                len(pair_rows) == 2
                and {row["incoming_carry"] for row in pair_rows} == {0, 1}
                and len(
                    {
                        (
                            row["operation"],
                            row["left_digit"],
                            row["right_digit"],
                            row["carry_map"],
                        )
                        for row in pair_rows
                    }
                )
                == 1
            )
            paired_counts[(event.operation, operator)] += 1

    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        for row in rows
    )
    pairs_by_operation = {
        operation: {
            carry_map.value: paired_counts[(operation, carry_map.value)]
            for carry_map in CarryMap
        }
        for operation in OPERATIONS
    }
    rows_by_operation = {
        operation: {
            carry_map.value: 2 * paired_counts[(operation, carry_map.value)]
            for carry_map in CarryMap
        }
        for operation in OPERATIONS
    }
    expected = {
        operation: {
            carry_map.value: BALANCED_PAIRS_PER_OPERATOR for carry_map in CarryMap
        }
        for operation in OPERATIONS
    }
    return {
        "underlying_pair_counts": monoid_audit()["event_counts"],
        "pair_repetitions": BALANCED_PAIR_REPETITIONS,
        "sampled_pairs_by_operation": pairs_by_operation,
        "sampled_rows_by_operation": rows_by_operation,
        "total_paired_examples": len(rows) // 2,
        "total_rows": len(rows),
        "counterfactual_pairs_inseparable": integrity,
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
        "pass": integrity
        and pairs_by_operation == expected
        and len(rows) == BALANCED_TOTAL_ROWS,
    }


def expected_flip_signature(event: DigitEvent) -> tuple[int, int, int, int]:
    zero = oracle_transition(event, 0)
    one = oracle_transition(event, 1)
    digit_delta = 1 if event.operation == "add" else 9
    if one.digit != (zero.digit + digit_delta) % 10:
        raise AssertionError("decimal oracle violated its carry-flip digit law")
    carry_map = classify_event(event)
    if (zero.next_carry, one.next_carry) != (
        carry_map.apply(0),
        carry_map.apply(1),
    ):
        raise AssertionError("decimal oracle violated its carry-map law")
    return zero.digit, one.digit, zero.next_carry, one.next_carry


def opposite_source(event: DigitEvent, incoming: int) -> DigitEvent:
    """Choose a same-operation source whose output carry is the opposite bit."""

    output = oracle_transition(event, incoming).next_carry
    if event.operation == "add":
        return DigitEvent("add", 0, 0) if output else DigitEvent("add", 9, 9)
    return DigitEvent("sub", 9, 0) if output else DigitEvent("sub", 0, 9)


def poison_retained_source(machine: object, replacement: DigitEvent) -> bool:
    """Mutate only an explicitly retained source slot in a negative control."""

    if not hasattr(machine, "source_event"):
        return False
    setattr(machine, "source_event", replacement)
    return True


def _run_pair(
    factory: MachineFactory,
    first_event: DigitEvent,
    second_event: DigitEvent,
    initial_carry: int,
    *,
    poison_source: bool,
) -> tuple[StepOutput, StepOutput, tuple[str, ...], bool]:
    machine = factory()
    first = machine.step(first_event, CommitBit(bool(initial_carry)))
    retained = machine_retained_fields(machine)
    poisoned = False
    if poison_source:
        poisoned = poison_retained_source(
            machine, opposite_source(first_event, initial_carry)
        )
    second = machine.step(second_event, first.packet)
    return first, second, retained, poisoned


def evaluate_machine(factory: MachineFactory) -> dict[str, Any]:
    local_exact = 0
    local_packet_exact = 0
    local_stateless = 0
    for event, carry in iter_local_cells():
        machine = factory()
        observed = machine.step(event, CommitBit(bool(carry)))
        expected = oracle_transition(event, carry)
        if (
            observed.digit == expected.digit
            and packet_bit(observed.packet) == expected.next_carry
        ):
            local_exact += 1
        local_packet_exact += int(exact_packet_contract(observed.packet))
        local_stateless += int(not machine_retained_fields(machine))

    flip_exact = 0
    for event in iter_events():
        zero_machine = factory()
        one_machine = factory()
        zero = zero_machine.step(event, CommitBit(False))
        one = one_machine.step(event, CommitBit(True))
        expected = expected_flip_signature(event)
        observed = (
            zero.digit,
            one.digit,
            packet_bit(zero.packet),
            packet_bit(one.packet),
        )
        flip_exact += int(observed == expected)

    edges = 0
    factual_exact = 0
    flipped_exact = 0
    source_poison_invariant = 0
    first_packet_exact = 0
    second_packet_exact = 0
    edge_stateless_after_commit = 0
    poisoned_controls = 0
    for operation in OPERATIONS:
        events = tuple(iter_events(operation))
        for first_event, second_event, initial_carry in itertools.product(
            events, events, (0, 1)
        ):
            edges += 1
            first, second, retained, _ = _run_pair(
                factory,
                first_event,
                second_event,
                initial_carry,
                poison_source=False,
            )
            expected_first = oracle_transition(first_event, initial_carry)
            expected_second = oracle_transition(second_event, expected_first.next_carry)
            factual_exact += int(
                first.digit == expected_first.digit
                and packet_bit(first.packet) == expected_first.next_carry
                and second.digit == expected_second.digit
                and packet_bit(second.packet) == expected_second.next_carry
            )
            first_packet_exact += int(exact_packet_contract(first.packet))
            second_packet_exact += int(exact_packet_contract(second.packet))
            edge_stateless_after_commit += int(not retained)

            flipped = 1 - initial_carry
            flip_first, flip_second, _, _ = _run_pair(
                factory,
                first_event,
                second_event,
                flipped,
                poison_source=False,
            )
            expected_flip_first = oracle_transition(first_event, flipped)
            expected_flip_second = oracle_transition(
                second_event, expected_flip_first.next_carry
            )
            flipped_exact += int(
                flip_first.digit == expected_flip_first.digit
                and packet_bit(flip_first.packet) == expected_flip_first.next_carry
                and flip_second.digit == expected_flip_second.digit
                and packet_bit(flip_second.packet) == expected_flip_second.next_carry
            )

            poison_first, poison_second, _, poisoned = _run_pair(
                factory,
                first_event,
                second_event,
                initial_carry,
                poison_source=True,
            )
            poisoned_controls += int(poisoned)
            source_poison_invariant += int(
                poison_first.digit == first.digit
                and packet_bit(poison_first.packet) == packet_bit(first.packet)
                and poison_second.digit == second.digit
                and packet_bit(poison_second.packet) == packet_bit(second.packet)
            )

    checks = {
        "all_400_local_cells_exact": local_exact == LOCAL_CELL_COUNT,
        "all_200_carry_flip_pairs_exact": flip_exact == LOCAL_EVENT_COUNT,
        "local_packets_exactly_one_bit": local_packet_exact == LOCAL_CELL_COUNT,
        "local_machine_has_no_retained_state": local_stateless == LOCAL_CELL_COUNT,
        "all_40000_factual_edges_exact": factual_exact == TWO_STEP_EDGE_COUNT,
        "all_40000_flipped_edges_exact": flipped_exact == TWO_STEP_EDGE_COUNT,
        "all_40000_source_poison_edges_invariant": source_poison_invariant
        == TWO_STEP_EDGE_COUNT,
        "all_edge_packets_exactly_one_bit": first_packet_exact == TWO_STEP_EDGE_COUNT
        and second_packet_exact == TWO_STEP_EDGE_COUNT,
        "machine_has_no_postcommit_state": edge_stateless_after_commit
        == TWO_STEP_EDGE_COUNT,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    machine_name = factory().name
    return {
        "name": machine_name,
        "local_cells_total": LOCAL_CELL_COUNT,
        "local_cells_exact": local_exact,
        "carry_flip_pairs_total": LOCAL_EVENT_COUNT,
        "carry_flip_pairs_exact": flip_exact,
        "local_packets_exact": local_packet_exact,
        "local_stateless_commits": local_stateless,
        "two_step_edges_total": edges,
        "factual_edges_exact": factual_exact,
        "flipped_edges_exact": flipped_exact,
        "source_poison_edges_invariant": source_poison_invariant,
        "poisoned_control_edges": poisoned_controls,
        "first_packets_exact": first_packet_exact,
        "second_packets_exact": second_packet_exact,
        "stateless_edge_commits": edge_stateless_after_commit,
        "checks": checks,
        "failed_checks": failed_checks,
        "pass": all(checks.values()),
    }


def build_report() -> dict[str, Any]:
    monoid = monoid_audit()
    allocation = balanced_allocation_audit()
    treatment = evaluate_machine(OBCBMachine)
    controls = {
        name: evaluate_machine(factory) for name, factory in CONTROL_FACTORIES.items()
    }
    for result in controls.values():
        result["rejected"] = not result["pass"]

    gates = {
        "three_element_monoid": monoid["pass"],
        "operator_balanced_counterfactual_allocation": allocation["pass"],
        "obcb_local_exactness": treatment["checks"]["all_400_local_cells_exact"],
        "obcb_carry_flip_bisimulation": treatment["checks"][
            "all_200_carry_flip_pairs_exact"
        ],
        "obcb_exact_iteration_closure": treatment["checks"][
            "all_40000_factual_edges_exact"
        ]
        and treatment["checks"]["all_40000_flipped_edges_exact"],
        "obcb_one_bit_source_deletion": treatment["checks"][
            "local_packets_exactly_one_bit"
        ]
        and treatment["checks"]["all_edge_packets_exactly_one_bit"]
        and treatment["checks"]["machine_has_no_postcommit_state"]
        and treatment["checks"]["all_40000_source_poison_edges_invariant"],
        "all_named_controls_rejected": all(
            result["rejected"] for result in controls.values()
        ),
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": {
            "cpu_structural_falsifier_only": True,
            "neural_capability_evidence": False,
            "h100_authorized": False,
            "blocked_until_live_carry_only_result": True,
            "oracle_available_at_neural_inference": False,
        },
        "resource_vector": RESOURCE_VECTOR,
        "monoid": monoid,
        "balanced_allocation": allocation,
        "obcb_treatment": treatment,
        "negative_controls": controls,
        "gates": gates,
        "pass": all(gates.values()),
    }
    return report


def verify_report(report: dict[str, Any]) -> None:
    if report.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("unexpected OBCB protocol ID")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("unexpected OBCB schema version")
    if report.get("resource_vector") != RESOURCE_VECTOR:
        raise ContractError("resource vector drifted")
    treatment = report.get("obcb_treatment", {})
    if treatment.get("local_cells_exact") != LOCAL_CELL_COUNT:
        raise ContractError("not all 400 local cells are exact")
    if treatment.get("carry_flip_pairs_exact") != LOCAL_EVENT_COUNT:
        raise ContractError("carry-flip bisimulation is incomplete")
    if treatment.get("two_step_edges_total") != TWO_STEP_EDGE_COUNT:
        raise ContractError("two-step board does not contain 40,000 edges")
    for key in (
        "factual_edges_exact",
        "flipped_edges_exact",
        "source_poison_edges_invariant",
        "first_packets_exact",
        "second_packets_exact",
        "stateless_edge_commits",
    ):
        if treatment.get(key) != TWO_STEP_EDGE_COUNT:
            raise ContractError(f"treatment failed frozen count: {key}")
    allocation = report.get("balanced_allocation", {})
    if allocation.get("total_rows") != BALANCED_TOTAL_ROWS:
        raise ContractError("operator-balanced board does not contain 1,080 rows")
    controls = report.get("negative_controls", {})
    if set(controls) != set(CONTROL_FACTORIES):
        raise ContractError("negative-control set drifted")
    if not all(result.get("rejected") is True for result in controls.values()):
        raise ContractError("a named negative control was admitted")
    gates = report.get("gates", {})
    if set(gates) != {
        "three_element_monoid",
        "operator_balanced_counterfactual_allocation",
        "obcb_local_exactness",
        "obcb_carry_flip_bisimulation",
        "obcb_exact_iteration_closure",
        "obcb_one_bit_source_deletion",
        "all_named_controls_rejected",
    }:
        raise ContractError("gate set drifted")
    if not all(value is True for value in gates.values()):
        raise ContractError("one or more frozen OBCB gates failed")
    if report.get("pass") is not True:
        raise ContractError("report did not receive structural PASS")


def canonical_report_bytes(report: dict[str, Any] | None = None) -> bytes:
    value = build_report() if report is None else report
    verify_report(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def write_immutable_report(path: Path, report: dict[str, Any]) -> None:
    payload = canonical_report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write one immutable canonical JSON report instead of stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    verify_report(report)
    if args.output is None:
        print(canonical_report_bytes(report).decode("ascii"), end="")
    else:
        write_immutable_report(args.output, report)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
