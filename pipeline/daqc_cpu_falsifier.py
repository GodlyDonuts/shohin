#!/usr/bin/env python3
"""Frozen CPU falsifiers for Deferred-Argument Quotient Compilation (DAQC).

This module is deliberately model-free.  It constructs a finite, score-blind
board for the faithful action of D_34 on Z_17, commits source programs before
late inputs are attached, and evaluates exact symbolic controls.  It does not
fit Shohin, import torch, or request a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "DAQC-D34-v1"
SCHEMA_VERSION = 1
MODULUS = 17
ACTION_COUNT = 34
VARIANTS_PER_ACTION = 4
LATE_INPUTS_PER_CASE = MODULUS

# These seeds are part of the preregistration.  Changing one requires a new
# protocol identifier and new frozen hashes.
SOURCE_SEED = 2026071501
LATE_INPUT_SEED = 2026071502
CONTROL_SEED = 2026071503

VARIANT_NAMES = (
    "canonical",
    "involution_padding",
    "conjugacy_padding",
    "cycle_padding",
)

RESOURCE_VECTOR_FIELDS = (
    "parameters",
    "retained_bits",
    "precision",
    "source_bytes",
    "training_examples",
    "oracle_calls",
    "training_flops",
    "inference_flops",
    "sequential_depth",
    "external_memory",
    "external_execution",
)

# Filled only after the generator and audit tests agree.  These values freeze
# the v1 board bytes; they are not inputs to board construction.
CANONICAL_SOURCE_SHA256 = "10d012c974431249fe9517983e0a9e6760d64bc08998c1bd5e3f1082e0130142"
CANONICAL_BOARD_SHA256 = "5a9e533757384ecdfe541f9b13ce0997b736fb5e4acc398e490d2815093839c7"


class AuditError(ValueError):
    """Raised when a purported frozen board violates a preregistered gate."""


@dataclass(frozen=True, order=True)
class Action:
    """Affine D_34 action x -> sign*x + offset over Z_17."""

    sign: int
    offset: int

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or 1, got {self.sign}")
        if not 0 <= self.offset < MODULUS:
            raise ValueError(f"offset must be in [0, {MODULUS}), got {self.offset}")

    def apply(self, value: int) -> int:
        if not 0 <= value < MODULUS:
            raise ValueError(f"value must be in [0, {MODULUS}), got {value}")
        return (self.sign * value + self.offset) % MODULUS

    def followed_by(self, later: "Action") -> "Action":
        """Compose in source order: apply self, then apply later."""

        return Action(
            sign=later.sign * self.sign,
            offset=(later.sign * self.offset + later.offset) % MODULUS,
        )


IDENTITY = Action(1, 0)
TRANSLATION = Action(1, 1)
REFLECTION = Action(-1, 0)
GENERATORS = {"T": TRANSLATION, "N": REFLECTION}


@dataclass(frozen=True)
class SealedCode:
    """The complete object visible to the post-commitment executor."""

    code_index: int

    def execute(self, late_input: int) -> int:
        return decode_action(self.code_index).apply(late_input)


@dataclass(frozen=True)
class CodeAuditEnvelope:
    """Non-executable provenance retained by the CPU auditor, not the executor."""

    case_id: str
    code: SealedCode
    source_commitment_sha256: str
    code_provenance_case_id: str


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_rank(seed: int, *parts: object) -> int:
    material = "|".join((str(seed), *(str(part) for part in parts))).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def action_space() -> tuple[Action, ...]:
    return tuple(
        Action(sign, offset)
        for sign in (1, -1)
        for offset in range(MODULUS)
    )


def encode_action(action: Action) -> int:
    return action.offset if action.sign == 1 else MODULUS + action.offset


def decode_action(code_index: int) -> Action:
    if not 0 <= code_index < ACTION_COUNT:
        raise ValueError(f"code index must be in [0, {ACTION_COUNT}), got {code_index}")
    if code_index < MODULUS:
        return Action(1, code_index)
    return Action(-1, code_index - MODULUS)


def code_bits(code_index: int) -> str:
    decode_action(code_index)
    return format(code_index, "06b")


def generator_action(token: str) -> Action:
    try:
        return GENERATORS[token]
    except KeyError as exc:
        raise ValueError(f"invalid generator {token!r}; expected 'T' or 'N'") from exc


def validate_word(word: str) -> None:
    invalid = sorted(set(word).difference(GENERATORS))
    if invalid:
        raise ValueError(f"word contains invalid generators: {invalid}")


def word_action(word: str) -> Action:
    """Compile a source word by exact affine composition."""

    validate_word(word)
    state = IDENTITY
    for token in word:
        state = state.followed_by(generator_action(token))
    return state


def sequential_execute(word: str, late_input: int) -> int:
    """Favorable source-retaining serial interpreter control."""

    validate_word(word)
    if not 0 <= late_input < MODULUS:
        raise ValueError("late input is outside Z_17")
    value = late_input
    for token in word:
        if token == "T":
            value = (value + 1) % MODULUS
        else:
            value = (-value) % MODULUS
    return value


def tree_compile(word: str) -> Action:
    """Favorable balanced exact product-tree control."""

    validate_word(word)
    if not word:
        return IDENTITY
    if len(word) == 1:
        return generator_action(word)
    middle = len(word) // 2
    left = tree_compile(word[:middle])
    right = tree_compile(word[middle:])
    return left.followed_by(right)


def _build_fst_transitions() -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    for action in action_space():
        rows.append(
            (
                encode_action(action.followed_by(TRANSLATION)),
                encode_action(action.followed_by(REFLECTION)),
            )
        )
    return tuple(rows)


FST_TRANSITIONS = _build_fst_transitions()


def fst_compile(word: str) -> int:
    """Favorable exact 34-state finite-state compiler control."""

    validate_word(word)
    state = encode_action(IDENTITY)
    for token in word:
        state = FST_TRANSITIONS[state][0 if token == "T" else 1]
    return state


def direct_execute(code_index: int, late_input: int) -> int:
    """Favorable direct-oracle control from the preregistered action label."""

    return decode_action(code_index).apply(late_input)


def canonical_word(action: Action) -> str:
    """Return a deterministic nonempty representative of an action class."""

    if action.sign == 1:
        return "T" * (action.offset or MODULUS)
    return "N" + ("T" * action.offset)


def _insert_identity_block(base: str, block: str, case_key: str) -> str:
    if word_action(block) != IDENTITY:
        raise AssertionError("padding block is not an identity word")
    position = _hash_rank(SOURCE_SEED, case_key, "insert") % (len(base) + 1)
    return base[:position] + block + base[position:]


def equivalent_variant(action: Action, action_index: int, variant: str) -> str:
    base = canonical_word(action)
    if variant == "canonical":
        word = base
    elif variant == "involution_padding":
        repetitions = 1 + (action_index % 5)
        word = _insert_identity_block(base, "NN" * repetitions, f"{action_index}:{variant}")
    elif variant == "conjugacy_padding":
        repetitions = 1 + (action_index % 4)
        word = _insert_identity_block(base, "NTNT" * repetitions, f"{action_index}:{variant}")
    elif variant == "cycle_padding":
        repetitions = 1 + (action_index % 3)
        word = _insert_identity_block(
            base, "T" * (MODULUS * repetitions), f"{action_index}:{variant}"
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if word_action(word) != action:
        raise AssertionError("equivalence-preserving variant changed the action")
    return word


SOURCE_CASE_KEYS = (
    "case_id",
    "action_index",
    "action_sign",
    "action_offset",
    "variant",
    "word",
    "word_length",
    "source_sha256",
    "committed_code_bits",
)

BOARD_CASE_KEYS = SOURCE_CASE_KEYS + ("late_inputs", "expected_outputs")


def _source_case(action: Action, action_index: int, variant: str) -> dict[str, Any]:
    word = equivalent_variant(action, action_index, variant)
    case_id = f"a{action_index:02d}-{variant}"
    source_sha = sha256_bytes(f"{case_id}\0{word}".encode("ascii"))
    return {
        "case_id": case_id,
        "action_index": action_index,
        "action_sign": action.sign,
        "action_offset": action.offset,
        "variant": variant,
        "word": word,
        "word_length": len(word),
        "source_sha256": source_sha,
        "committed_code_bits": code_bits(action_index),
    }


def generate_source_cases() -> list[dict[str, Any]]:
    cases = [
        _source_case(action, action_index, variant)
        for action_index, action in enumerate(action_space())
        for variant in VARIANT_NAMES
    ]
    words = [case["word"] for case in cases]
    if len(words) != len(set(words)):
        raise AssertionError("source board contains duplicate words")
    return cases


def source_commitment_payload(source_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for case in source_cases:
        missing = set(SOURCE_CASE_KEYS).difference(case)
        if missing:
            raise AuditError(f"source case is missing fields: {sorted(missing)}")
        normalized.append({key: case[key] for key in SOURCE_CASE_KEYS})
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "modulus": MODULUS,
        "source_seed": SOURCE_SEED,
        "variant_names": list(VARIANT_NAMES),
        "source_cases": normalized,
    }


def source_commitment_sha256(source_cases: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(_canonical_json_bytes(source_commitment_payload(source_cases)))


def _late_input_order(case_id: str) -> list[int]:
    return sorted(
        range(MODULUS),
        key=lambda value: _hash_rank(LATE_INPUT_SEED, case_id, value),
    )


def generate_board() -> dict[str, Any]:
    """Generate the immutable v1 board in two explicit phases."""

    # Phase 1: source programs and compiled action codes are committed.
    source_cases = generate_source_cases()
    source_sha = source_commitment_sha256(source_cases)

    # Phase 2: every possible late input is attached in independently seeded
    # order.  The source commitment above cannot depend on this phase.
    board_cases: list[dict[str, Any]] = []
    for source_case in source_cases:
        late_inputs = _late_input_order(source_case["case_id"])
        action = decode_action(source_case["action_index"])
        board_case = dict(source_case)
        board_case["late_inputs"] = late_inputs
        board_case["expected_outputs"] = [action.apply(value) for value in late_inputs]
        board_cases.append(board_case)

    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "modulus": MODULUS,
        "action_count": ACTION_COUNT,
        "variants_per_action": VARIANTS_PER_ACTION,
        "late_inputs_per_case": LATE_INPUTS_PER_CASE,
        "seeds": {
            "source": SOURCE_SEED,
            "late_input": LATE_INPUT_SEED,
            "control": CONTROL_SEED,
        },
        "source_commitment_sha256": source_sha,
        "cases": board_cases,
    }


def board_bytes(board: Mapping[str, Any] | None = None) -> bytes:
    return _canonical_json_bytes(generate_board() if board is None else board)


def board_sha256(board: Mapping[str, Any] | None = None) -> str:
    return sha256_bytes(board_bytes(board))


def _assert_frozen_hash(actual: str, expected: str, label: str) -> None:
    if expected.startswith("__"):
        raise AuditError(f"{label} has not been frozen in source")
    if actual != expected:
        raise AuditError(f"{label} mismatch: expected {expected}, got {actual}")


def seal_source_case(case: Mapping[str, Any], source_commitment: str) -> CodeAuditEnvelope:
    word = str(case["word"])
    compiled = encode_action(word_action(word))
    claimed = int(case["action_index"])
    if compiled != claimed:
        raise AuditError(
            f"case {case.get('case_id')} compiles to {compiled}, claims {claimed}"
        )
    return CodeAuditEnvelope(
        case_id=str(case["case_id"]),
        code=SealedCode(code_index=compiled),
        source_commitment_sha256=source_commitment,
        code_provenance_case_id=str(case["case_id"]),
    )


def interchange_code(
    recipient: CodeAuditEnvelope, donor: CodeAuditEnvelope
) -> CodeAuditEnvelope:
    """Swap only the committed quotient code, retaining recipient identity."""

    return CodeAuditEnvelope(
        case_id=recipient.case_id,
        code=donor.code,
        source_commitment_sha256=recipient.source_commitment_sha256,
        code_provenance_case_id=donor.code_provenance_case_id,
    )


def behavior_signature(action: Action) -> tuple[int, ...]:
    return tuple(action.apply(value) for value in range(MODULUS))


def relation_audit() -> dict[str, Any]:
    relations = {
        "T^17=e": ("T" * MODULUS, ""),
        "N^2=e": ("NN", ""),
        "NTN=T^-1": ("NTN", "T" * (MODULUS - 1)),
        "NTNT=e": ("NTNT", ""),
    }
    results: dict[str, Any] = {}
    for name, (left, right) in relations.items():
        left_action = word_action(left)
        right_action = word_action(right)
        outputs_equal = all(
            sequential_execute(left, value) == sequential_execute(right, value)
            for value in range(MODULUS)
        )
        results[name] = {
            "left_code": encode_action(left_action),
            "right_code": encode_action(right_action),
            "all_17_inputs_equal": outputs_equal,
            "pass": left_action == right_action and outputs_equal,
        }

    tn = word_action("TN")
    nt = word_action("NT")
    witness_input = next(
        value for value in range(MODULUS) if tn.apply(value) != nt.apply(value)
    )
    noncommutative = {
        "left_word": "TN",
        "right_word": "NT",
        "witness_input": witness_input,
        "left_output": tn.apply(witness_input),
        "right_output": nt.apply(witness_input),
        "pass": tn != nt,
    }
    return {
        "relations": results,
        "noncommutative_translation_reflection_witness": noncommutative,
        "pass": all(item["pass"] for item in results.values())
        and noncommutative["pass"],
    }


def collision_audit() -> dict[str, Any]:
    signatures: dict[tuple[int, ...], int] = {}
    collisions: list[tuple[int, int]] = []
    for index, action in enumerate(action_space()):
        signature = behavior_signature(action)
        if signature in signatures:
            collisions.append((signatures[signature], index))
        else:
            signatures[signature] = index
    return {
        "action_count": ACTION_COUNT,
        "unique_behavior_signatures": len(signatures),
        "cross_action_collisions": [list(pair) for pair in collisions],
        "pass": len(signatures) == ACTION_COUNT and not collisions,
    }


def minimum_injective_bits(cardinality: int) -> int:
    if cardinality <= 0:
        raise ValueError("cardinality must be positive")
    return (cardinality - 1).bit_length()


def finite_code_capacity_audit(bit_width: int) -> dict[str, Any]:
    if bit_width < 0:
        raise ValueError("bit width must be nonnegative")
    capacity = 1 << bit_width
    buckets: dict[int, list[int]] = {}
    for action_index in range(ACTION_COUNT):
        buckets.setdefault(action_index % capacity, []).append(action_index)

    witnesses: list[dict[str, int]] = []
    for bucket, members in sorted(buckets.items()):
        for position in range(1, len(members)):
            left_index = members[0]
            right_index = members[position]
            left = decode_action(left_index)
            right = decode_action(right_index)
            separating_input = next(
                value
                for value in range(MODULUS)
                if left.apply(value) != right.apply(value)
            )
            witnesses.append(
                {
                    "bucket": bucket,
                    "left_action": left_index,
                    "right_action": right_index,
                    "separating_input": separating_input,
                    "left_output": left.apply(separating_input),
                    "right_output": right.apply(separating_input),
                }
            )
    return {
        "bit_width": bit_width,
        "capacity": capacity,
        "required_bits": minimum_injective_bits(ACTION_COUNT),
        "collision_witnesses": witnesses,
        "injective_possible": capacity >= ACTION_COUNT,
    }


def free_semigroup_linear_bit_no_go(
    alphabet_size: int = 2, max_word_length: int = 64
) -> dict[str, Any]:
    """Exact counting no-go for semantics with no quotient collisions."""

    if alphabet_size < 2:
        raise ValueError("alphabet size must be at least two")
    if max_word_length < 1:
        raise ValueError("max word length must be positive")
    rows = []
    for length in range(1, max_word_length + 1):
        distinct_semantics = alphabet_size**length
        rows.append(
            {
                "word_length": length,
                "distinct_semantics": distinct_semantics,
                "minimum_injective_bits": minimum_injective_bits(distinct_semantics),
            }
        )
    binary_exact_linear = alphabet_size != 2 or all(
        row["minimum_injective_bits"] == row["word_length"] for row in rows
    )
    return {
        "alphabet_size": alphabet_size,
        "max_word_length": max_word_length,
        "rows": rows,
        "binary_bits_equal_length": binary_exact_linear,
        "pass": binary_exact_linear,
    }


def serial_reliability(per_atomic_success: Fraction, source_length: int) -> Fraction:
    if not 0 <= per_atomic_success <= 1:
        raise ValueError("success probability must lie in [0, 1]")
    if source_length < 0:
        raise ValueError("source length must be nonnegative")
    return per_atomic_success**source_length


def compiled_reliability(
    compiler_success: Fraction,
    per_runtime_success: Fraction,
    runtime_calls: int,
) -> Fraction:
    if not 0 <= compiler_success <= 1 or not 0 <= per_runtime_success <= 1:
        raise ValueError("success probabilities must lie in [0, 1]")
    if runtime_calls < 0:
        raise ValueError("runtime calls must be nonnegative")
    return compiler_success * (per_runtime_success**runtime_calls)


def tree_reliability(per_merge_success: Fraction, leaves: int) -> Fraction:
    if not 0 <= per_merge_success <= 1:
        raise ValueError("success probability must lie in [0, 1]")
    if leaves < 1:
        raise ValueError("tree must have at least one leaf")
    return per_merge_success ** (leaves - 1)


def reliability_audit() -> dict[str, Any]:
    atomic = Fraction(99, 100)
    compiler = Fraction(999, 1000)
    runtime = Fraction(99, 100)
    source_length = 64
    runtime_calls = 1
    serial = serial_reliability(atomic, source_length)
    compiled = compiled_reliability(compiler, runtime, runtime_calls)
    tree = tree_reliability(atomic, source_length)
    return {
        "fixed_exact_inputs": {
            "atomic_success": f"{atomic.numerator}/{atomic.denominator}",
            "compiler_success": f"{compiler.numerator}/{compiler.denominator}",
            "runtime_success": f"{runtime.numerator}/{runtime.denominator}",
            "source_length": source_length,
            "runtime_calls": runtime_calls,
        },
        "serial_success": f"{serial.numerator}/{serial.denominator}",
        "compiled_success": f"{compiled.numerator}/{compiled.denominator}",
        "tree_success": f"{tree.numerator}/{tree.denominator}",
        "compiled_exceeds_serial_on_fixed_witness": compiled > serial,
        "tree_has_l_minus_one_fallible_merges": True,
        "pass": compiled > serial and tree == atomic ** (source_length - 1),
    }


def controls_audit(board: Mapping[str, Any]) -> dict[str, Any]:
    control_names = ("exact_fst", "sequential", "balanced_tree", "direct")
    correct = {name: 0 for name in control_names}
    total = 0
    for case in board["cases"]:
        word = str(case["word"])
        code = int(case["action_index"])
        fst_code = fst_compile(word)
        tree_code = encode_action(tree_compile(word))
        for late_input, expected in zip(
            case["late_inputs"], case["expected_outputs"], strict=True
        ):
            total += 1
            if direct_execute(fst_code, late_input) == expected:
                correct["exact_fst"] += 1
            if sequential_execute(word, late_input) == expected:
                correct["sequential"] += 1
            if direct_execute(tree_code, late_input) == expected:
                correct["balanced_tree"] += 1
            if direct_execute(code, late_input) == expected:
                correct["direct"] += 1
    return {
        "total_cells_per_control": total,
        "correct_cells": correct,
        "all_exact": all(value == total for value in correct.values()),
        "pass": all(value == total for value in correct.values()),
    }


def code_interchange_audit(board: Mapping[str, Any]) -> dict[str, Any]:
    representatives: dict[int, CodeAuditEnvelope] = {}
    source_commitment = str(board["source_commitment_sha256"])
    for case in board["cases"]:
        code = int(case["action_index"])
        representatives.setdefault(code, seal_source_case(case, source_commitment))

    pairs = [
        (recipient, donor)
        for recipient in range(ACTION_COUNT)
        for donor in range(ACTION_COUNT)
        if recipient != donor
    ]
    pairs.sort(key=lambda pair: _hash_rank(CONTROL_SEED, "swap", *pair))

    checked_cells = 0
    separating_pairs = 0
    for recipient_index, donor_index in pairs:
        recipient = representatives[recipient_index]
        donor = representatives[donor_index]
        swapped = interchange_code(recipient, donor)
        donor_action = decode_action(donor_index)
        recipient_action = decode_action(recipient_index)
        pair_separates = False
        for late_input in range(MODULUS):
            checked_cells += 1
            if swapped.code.execute(late_input) != donor_action.apply(late_input):
                raise AuditError("interchanged code did not follow donor behavior")
            pair_separates |= (
                swapped.code.execute(late_input) != recipient_action.apply(late_input)
            )
        separating_pairs += int(pair_separates)

    expected_pairs = ACTION_COUNT * (ACTION_COUNT - 1)
    expected_cells = expected_pairs * MODULUS
    return {
        "ordered_distinct_action_pairs": len(pairs),
        "checked_late_input_cells": checked_cells,
        "pairs_with_recipient_donor_separation": separating_pairs,
        "expected_pairs": expected_pairs,
        "expected_cells": expected_cells,
        "pass": len(pairs) == expected_pairs
        and checked_cells == expected_cells
        and separating_pairs == expected_pairs,
    }


def resource_vector(board: Mapping[str, Any]) -> dict[str, Any]:
    serialized_bytes = len(board_bytes(board))
    source_bytes = sum(len(case["word"].encode("ascii")) for case in board["cases"])
    case_count = len(board["cases"])
    lengths = [int(case["word_length"]) for case in board["cases"]]
    audit_envelope_ascii_bytes = sum(
        len(case["case_id"].encode("ascii")) * 2 + 64 for case in board["cases"]
    )
    vector = {
        "parameters": 0,
        "retained_bits": {
            "per_sealed_code": minimum_injective_bits(ACTION_COUNT),
            "all_sealed_codes": case_count * minimum_injective_bits(ACTION_COUNT),
            "retained_source_after_sealing": 0,
        },
        "precision": {
            "kind": "exact_integer_mod_17",
            "floating_point_used": False,
            "logical_input_output_bits": 5,
            "logical_action_code_bits": 6,
        },
        "source_bytes": source_bytes,
        "training_examples": 0,
        "oracle_calls": 0,
        "training_flops": 0,
        "inference_flops": 0,
        "sequential_depth": {
            "direct_post_commit": 1,
            "sequential_min": min(lengths),
            "sequential_max": max(lengths),
            "fst_compile_min": min(lengths),
            "fst_compile_max": max(lengths),
            "balanced_tree_compile_max": max(math.ceil(math.log2(n)) for n in lengths),
            "post_compile_execute": 1,
        },
        "external_memory": {
            "canonical_board_bytes": serialized_bytes,
            "logical_sealed_code_bits": case_count
            * minimum_injective_bits(ACTION_COUNT),
            "non_executable_audit_envelope_ascii_bytes": audit_envelope_ascii_bytes,
        },
        "external_execution": {
            "used": True,
            "kind": "stdlib_cpu_exact_symbolic_controls",
            "integer_operations_counted_separately_from_flops": True,
        },
    }
    if tuple(vector) != RESOURCE_VECTOR_FIELDS:
        raise AssertionError("resource vector field order drifted")
    return vector


EXPECTED_TOP_LEVEL_KEYS = {
    "protocol_id",
    "schema_version",
    "modulus",
    "action_count",
    "variants_per_action",
    "late_inputs_per_case",
    "seeds",
    "source_commitment_sha256",
    "cases",
}


def _validate_board_structure(board: Mapping[str, Any]) -> None:
    if set(board) != EXPECTED_TOP_LEVEL_KEYS:
        raise AuditError("board top-level schema differs from frozen v1 schema")
    expected_scalars = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "modulus": MODULUS,
        "action_count": ACTION_COUNT,
        "variants_per_action": VARIANTS_PER_ACTION,
        "late_inputs_per_case": LATE_INPUTS_PER_CASE,
        "seeds": {
            "source": SOURCE_SEED,
            "late_input": LATE_INPUT_SEED,
            "control": CONTROL_SEED,
        },
    }
    for key, expected in expected_scalars.items():
        if board[key] != expected:
            raise AuditError(f"board field {key!r} differs from frozen value")

    cases = board["cases"]
    expected_case_count = ACTION_COUNT * VARIANTS_PER_ACTION
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        raise AuditError(f"expected {expected_case_count} cases")
    case_ids: set[str] = set()
    words: set[str] = set()
    action_variant_pairs: set[tuple[int, str]] = set()
    for case in cases:
        if set(case) != set(BOARD_CASE_KEYS):
            raise AuditError("case schema differs from frozen v1 schema")
        case_id = str(case["case_id"])
        word = str(case["word"])
        action_index = int(case["action_index"])
        variant = str(case["variant"])
        if case_id in case_ids or word in words:
            raise AuditError("case ids and source words must both be unique")
        case_ids.add(case_id)
        words.add(word)
        action_variant_pairs.add((action_index, variant))
        action = decode_action(action_index)
        if action.sign != case["action_sign"] or action.offset != case["action_offset"]:
            raise AuditError(f"action fields disagree in {case_id}")
        if variant not in VARIANT_NAMES:
            raise AuditError(f"unknown variant in {case_id}")
        if len(word) != case["word_length"] or word_action(word) != action:
            raise AuditError(f"source word does not match committed action in {case_id}")
        expected_source_sha = sha256_bytes(f"{case_id}\0{word}".encode("ascii"))
        if case["source_sha256"] != expected_source_sha:
            raise AuditError(f"source hash mismatch in {case_id}")
        if case["committed_code_bits"] != code_bits(action_index):
            raise AuditError(f"committed code mismatch in {case_id}")
        late_inputs = case["late_inputs"]
        outputs = case["expected_outputs"]
        if late_inputs != _late_input_order(case_id):
            raise AuditError(f"late-input order mismatch in {case_id}")
        if sorted(late_inputs) != list(range(MODULUS)):
            raise AuditError(f"late inputs are not exhaustive in {case_id}")
        if outputs != [action.apply(value) for value in late_inputs]:
            raise AuditError(f"expected outputs mismatch in {case_id}")

    expected_pairs = {
        (action_index, variant)
        for action_index in range(ACTION_COUNT)
        for variant in VARIANT_NAMES
    }
    if action_variant_pairs != expected_pairs:
        raise AuditError("action-by-variant board is incomplete")


def audit_board(board: Mapping[str, Any], require_frozen_hashes: bool = True) -> dict[str, Any]:
    """Audit structure, commitments, controls, and finite falsification gates."""

    _validate_board_structure(board)
    cases = board["cases"]
    source_sha = source_commitment_sha256(cases)
    if source_sha != board["source_commitment_sha256"]:
        raise AuditError("source commitment does not match source-only payload")
    full_sha = board_sha256(board)
    if require_frozen_hashes:
        _assert_frozen_hash(source_sha, CANONICAL_SOURCE_SHA256, "source commitment")
        _assert_frozen_hash(full_sha, CANONICAL_BOARD_SHA256, "board hash")

    relations = relation_audit()
    collisions = collision_audit()
    controls = controls_audit(board)
    interchange = code_interchange_audit(board)
    five_bit = finite_code_capacity_audit(5)
    six_bit = finite_code_capacity_audit(6)
    linear_no_go = free_semigroup_linear_bit_no_go()
    reliability = reliability_audit()
    gates = {
        "frozen_source_commitment": not require_frozen_hashes
        or source_sha == CANONICAL_SOURCE_SHA256,
        "frozen_board_hash": not require_frozen_hashes
        or full_sha == CANONICAL_BOARD_SHA256,
        "all_34_actions_four_variants": len(cases)
        == ACTION_COUNT * VARIANTS_PER_ACTION,
        "all_17_late_inputs": all(
            sorted(case["late_inputs"]) == list(range(MODULUS)) for case in cases
        ),
        "relations_and_noncommutativity": relations["pass"],
        "faithful_no_cross_action_collisions": collisions["pass"],
        "four_favorable_controls_exact": controls["pass"],
        "code_interchange_exact": interchange["pass"],
        "five_bits_impossible": bool(five_bit["collision_witnesses"])
        and not five_bit["injective_possible"],
        "six_bits_sufficient": six_bit["injective_possible"]
        and not six_bit["collision_witnesses"],
        "free_binary_semigroup_linear_bit_no_go": linear_no_go["pass"],
        "fixed_reliability_witness": reliability["pass"],
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "source_commitment_sha256": source_sha,
        "board_sha256": full_sha,
        "case_count": len(cases),
        "evaluation_cells": len(cases) * MODULUS,
        "relations": relations,
        "collisions": collisions,
        "controls": controls,
        "code_interchange": interchange,
        "five_bit_falsifier": five_bit,
        "six_bit_control": six_bit,
        "linear_bit_no_go": linear_no_go,
        "reliability": reliability,
        "resource_vector": resource_vector(board),
        "gates": gates,
        "pass": all(gates.values()),
    }


def write_immutable_board(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Create the canonical board once, read-only; never overwrite a path."""

    target = Path(path)
    payload = board_bytes()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, 0o444)
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(target),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": stat.S_IMODE(target.stat().st_mode),
    }


def audit_board_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    mode = stat.S_IMODE(target.stat().st_mode)
    if mode != 0o444:
        raise AuditError(f"board file mode must be 0444, got {mode:04o}")
    raw = target.read_bytes()
    try:
        board = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("board is not valid canonical JSON") from exc
    if raw != board_bytes(board):
        raise AuditError("board file bytes are not canonical JSON bytes")
    return audit_board(board)


def sealed_code_has_no_source_fields() -> bool:
    forbidden = {"word", "source", "tokens", "program"}
    field_names = {field.name for field in fields(SealedCode)}
    return field_names == {"code_index"} and not forbidden.intersection(field_names)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": report["protocol_id"],
        "source_commitment_sha256": report["source_commitment_sha256"],
        "board_sha256": report["board_sha256"],
        "case_count": report["case_count"],
        "evaluation_cells": report["evaluation_cells"],
        "gates": report["gates"],
        "pass": report["pass"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="write frozen board once")
    generate_parser.add_argument("board", type=Path)

    audit_parser = subparsers.add_parser("audit", help="audit an existing board")
    audit_parser.add_argument("board", type=Path)

    subparsers.add_parser("self-check", help="generate in memory and run every gate")
    args = parser.parse_args(argv)

    if args.command == "generate":
        result = write_immutable_board(args.board)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "audit":
        report = audit_board_file(args.board)
    else:
        report = audit_board(generate_board())
    print(json.dumps(_summary(report), sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
