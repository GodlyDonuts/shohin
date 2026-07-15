#!/usr/bin/env python3
"""Frozen raw-260k teacher-forced next-operation likelihood diagnostic.

This evaluator reuses the immutable operation-cursor source board and its
score-blind first-16-per-family subset.  It performs one inference-only forward
per transition and arm, then reads only four frozen one-token candidate logits.
It contains no generation, training, retry, repair, or adaptive search path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCHEMA = "source_scheduled_reasoning_confirmation_v1"
RESULT_SCHEMA = "raw260k_operation_selection_likelihood_v1"
CANDIDATE_MANIFEST_SCHEMA = "raw260k_operation_candidate_manifest_v1"
PROMPT_MANIFEST_SCHEMA = "raw260k_operation_selection_prompt_manifest_v1"
TOKENIZED_MANIFEST_SCHEMA = (
    "raw260k_operation_selection_tokenized_prompt_manifest_v1"
)

SOURCE_SEED = 2026071502
SOURCE_PER_FAMILY = 64
DIAGNOSTIC_PER_FAMILY = 16
SOURCE_CASE_COUNT = 256
SOURCE_TRANSITION_COUNT = 704
DIAGNOSTIC_CASE_COUNT = 64
DIAGNOSTIC_TRANSITION_COUNT = 176
EXPECTED_CHECKPOINT_STEP = 260000
EXPECTED_CONTEXT_LENGTH = 2048
EXPECTED_MODEL_CALLS = 528
EXPECTED_CANDIDATE_LOGIT_VALUES = 2112
FROZEN_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)
OPERATIONS = ("add", "subtract", "multiply", "remainder")

FULL_SOURCE_CURSOR = "full_source_cursor"
RESIDUAL_SUFFIX_HEAD = "residual_suffix_head"
RESIDUAL_SUFFIX_ORACLE_STATE = "residual_suffix_oracle_state"
ARMS = (
    FULL_SOURCE_CURSOR,
    RESIDUAL_SUFFIX_HEAD,
    RESIDUAL_SUFFIX_ORACLE_STATE,
)

EXPECTED_SOURCE_SHA256 = (
    "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474"
)
EXPECTED_SOURCE_ROWS_SHA256 = (
    "4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4"
)
EXPECTED_SUBSET_ROWS_SHA256 = (
    "c48ad18103b7971e7cd3c29be172ed40baccaa10d5d255011a22d3c023dc17e6"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
)
EXPECTED_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)

# Filled from canonical ASCII JSON manifests after the source/prompt contract is
# constructed.  The values are mirrored in R12_OPERATION_SELECTION_LIKELIHOOD_PREREG.md.
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "1e25e46807928f7ca9af1a3ea601181513e9cfe677e21f08632b295a62a75b89"
)
EXPECTED_PROMPT_MANIFEST_SHA256 = (
    "0a85a49f6ba818d51f6b74a48129f48e9e9d6bcd02dde152c31d626c68a472d6"
)
EXPECTED_TOKENIZED_MANIFEST_SHA256 = (
    "2e910eb13ad2200cee82d713174f43c95d5c8a8544b9089b5b6de2fff976aaa6"
)

# These exact tokenizer-derived counts are also frozen in the preregistration.
EXPECTED_PROMPT_TOKENS_BY_ARM = {
    FULL_SOURCE_CURSOR: 10868,
    RESIDUAL_SUFFIX_HEAD: 9731,
    RESIDUAL_SUFFIX_ORACLE_STATE: 12561,
}
EXPECTED_PROMPT_TOKENS_TOTAL = 33160
EXPECTED_MAX_PROMPT_TOKENS = 79

EXPECTED_TRANSITIONS_BY_FAMILY = {
    "multiply_subtract": 32,
    "base_conversion": 64,
    "sequential_state": 48,
    "modular_update": 32,
}
EXPECTED_TRANSITIONS_BY_INDEX = {0: 64, 1: 64, 2: 32, 3: 16}
EXPECTED_TRANSITIONS_BY_OPERATION = {
    "add": 64,
    "subtract": 32,
    "multiply": 64,
    "remainder": 16,
}

SOURCE_KEYS = {
    "schema",
    "seed",
    "per_family",
    "case_count",
    "family_order",
    "cases_sha256",
    "rows",
}
SOURCE_ROW_KEYS = {
    "id",
    "family",
    "question",
    "initial_state",
    "schedule",
    "answer",
    "stratum",
}

_MULTIPLY = re.compile(r"Compute (\d+) times (\d+), then subtract (\d+)\.")
_BASE = re.compile(r"Convert the base-(\d+) numeral ([0-9]{3}) to base 10\.")
_SEQUENTIAL = re.compile(
    r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\."
)
_MODULAR = re.compile(
    r"Add (\d+) and (\d+), then give the remainder after division by (\d+)\."
)


@dataclass(frozen=True)
class CandidateSpec:
    operation: str
    text: str
    token_id: int


CANDIDATES = (
    CandidateSpec("add", " add", 820),
    CandidateSpec("subtract", " subtract", 5498),
    CandidateSpec("multiply", " multiply", 4307),
    CandidateSpec("remainder", " remainder", 7486),
)


@dataclass(frozen=True)
class FrozenTransition:
    row_id: str
    family: str
    question: str
    index: int
    gold_operation: str
    current_state: int
    residual_suffix: tuple[tuple[str, int], ...]
    prompts: tuple[tuple[str, str], ...]

    def prompt_for(self, arm: str) -> str:
        return dict(self.prompts)[arm]


@dataclass(frozen=True)
class PreparedPrompt:
    arm: str
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedTransition:
    frozen: FrozenTransition
    prompts: tuple[PreparedPrompt, ...]

    def prompt_for(self, arm: str) -> PreparedPrompt:
        for prompt in self.prompts:
            if prompt.arm == arm:
                return prompt
        raise KeyError(arm)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(rows))


def hash_regular_file(
    path: str | Path, *, require_read_only: bool = False
) -> str:
    source_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {source_path}")
        if require_read_only and metadata.st_mode & 0o222:
            raise PermissionError(f"frozen input has write bits: {source_path}")
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file_sha256(
    path: str | Path,
    expected: str,
    label: str,
    *,
    require_read_only: bool = False,
) -> str:
    actual = hash_regular_file(path, require_read_only=require_read_only)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def read_regular_file_bytes(
    path: str | Path,
    *,
    require_read_only: bool = False,
) -> bytes:
    """Read one regular-file snapshot through a no-follow descriptor."""
    source_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {source_path}")
        if require_read_only and metadata.st_mode & 0o222:
            raise PermissionError(f"frozen input has write bits: {source_path}")
        payload = source.read()
    return payload


def require_file_bytes_sha256(
    path: str | Path,
    expected: str,
    label: str,
    *,
    require_read_only: bool = False,
) -> tuple[bytes, str]:
    """Read one regular-file snapshot and bind the exact bytes later consumed."""
    payload = read_regular_file_bytes(path, require_read_only=require_read_only)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return payload, actual


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def strict_json_loads(payload: bytes) -> Any:
    return json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )


def parse_question(
    family: str, question: str
) -> tuple[int, list[tuple[str, int]], dict[str, Any]]:
    """Reconstruct the schedule from source prose, never from a result file."""
    if family == "multiply_subtract":
        match = _MULTIPLY.fullmatch(question)
        _require(match is not None, f"unparsed multiply_subtract question: {question}")
        start, multiplier, subtractor = map(int, match.groups())
        return (
            start,
            [("multiply", multiplier), ("subtract", subtractor)],
            {"start": start, "multiplier": multiplier, "subtractor": subtractor},
        )
    if family == "base_conversion":
        match = _BASE.fullmatch(question)
        _require(match is not None, f"unparsed base_conversion question: {question}")
        base = int(match.group(1))
        digits = [int(value) for value in match.group(2)]
        return (
            digits[0],
            [
                ("multiply", base),
                ("add", digits[1]),
                ("multiply", base),
                ("add", digits[2]),
            ],
            {"base": base, "digits": digits},
        )
    if family == "sequential_state":
        match = _SEQUENTIAL.fullmatch(question)
        _require(match is not None, f"unparsed sequential_state question: {question}")
        start, addend, multiplier, subtractor = map(int, match.groups())
        return (
            start,
            [
                ("add", addend),
                ("multiply", multiplier),
                ("subtract", subtractor),
            ],
            {
                "start": start,
                "addend": addend,
                "multiplier": multiplier,
                "subtractor": subtractor,
            },
        )
    if family == "modular_update":
        match = _MODULAR.fullmatch(question)
        _require(match is not None, f"unparsed modular_update question: {question}")
        start, addend, modulus = map(int, match.groups())
        return (
            start,
            [("add", addend), ("remainder", modulus)],
            {"start": start, "addend": addend, "modulus": modulus},
        )
    raise ValueError(f"unknown family: {family}")


def apply_operation(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "remainder":
        return value % operand
    raise ValueError(f"unknown operation: {operation}")


def _in_range(value: int, lower: int, upper: int) -> bool:
    return lower <= value <= upper


def _expected_stratum(
    family: str, family_index: int, details: Mapping[str, Any], answer: int
) -> str:
    first_half = family_index < 32
    if family == "multiply_subtract":
        multiplier_range = (2, 9) if first_half else (10, 19)
        valid = (
            _in_range(details["start"], 20, 99)
            and _in_range(details["multiplier"], *multiplier_range)
            and details["subtractor"] > 0
            and answer > 0
        )
        stratum = "small_multiplier" if first_half else "two_digit_multiplier"
    elif family == "base_conversion":
        base_range = (2, 9) if first_half else (10, 12)
        digits = details["digits"]
        valid = (
            _in_range(details["base"], *base_range)
            and digits[0] > 0
            and all(0 <= digit <= 9 and digit < details["base"] for digit in digits)
        )
        stratum = "base_2_9" if first_half else "base_10_12"
    elif family == "sequential_state":
        multiplier_range = (2, 5) if first_half else (6, 7)
        valid = (
            _in_range(details["start"], 5, 50)
            and _in_range(details["addend"], 1, 25)
            and _in_range(details["multiplier"], *multiplier_range)
            and details["subtractor"] > 0
            and answer > 0
        )
        stratum = "multiplier_2_5" if first_half else "multiplier_6_7"
    else:
        modulus_range = (3, 14) if first_half else (15, 25)
        valid = (
            _in_range(details["start"], 10, 99)
            and _in_range(details["addend"], 10, 99)
            and _in_range(details["modulus"], *modulus_range)
        )
        stratum = "modulus_3_14" if first_half else "modulus_15_25"
    _require(valid, f"row violates frozen {family} ranges")
    return stratum


def reconstruct_schedule(row: Mapping[str, Any]) -> tuple[int, list[tuple[str, int]]]:
    family = row.get("family")
    question = row.get("question")
    _require(isinstance(family, str), "source family is not a string")
    _require(isinstance(question, str), "source question is not a string")
    start, schedule, _ = parse_question(family, question)
    raw_schedule = row.get("schedule")
    _require(isinstance(raw_schedule, list), "source schedule is not a list")
    normalized: list[tuple[str, int]] = []
    for step in raw_schedule:
        _require(
            isinstance(step, list)
            and len(step) == 2
            and type(step[0]) is str
            and type(step[1]) is int,
            "invalid source schedule step",
        )
        normalized.append((step[0], step[1]))
    _require(
        type(row.get("initial_state")) is int and row["initial_state"] == start,
        "question and source initial state disagree",
    )
    _require(normalized == schedule, "question and source schedule disagree")
    return start, schedule


def audit_source(source: Any) -> tuple[list[dict[str, Any]], int]:
    _require(type(source) is dict and set(source) == SOURCE_KEYS, "wrong source schema")
    _require(
        source.get("schema") == SOURCE_SCHEMA and source.get("seed") == SOURCE_SEED,
        "wrong source schema id or seed",
    )
    _require(
        source.get("per_family") == SOURCE_PER_FAMILY
        and source.get("case_count") == SOURCE_CASE_COUNT
        and tuple(source.get("family_order", ())) == FAMILIES,
        "wrong source family or cardinality contract",
    )
    _require(
        source.get("cases_sha256") == EXPECTED_SOURCE_ROWS_SHA256,
        "wrong frozen source rows hash metadata",
    )
    rows = source.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == SOURCE_CASE_COUNT,
        "wrong source row count",
    )
    _require(digest_rows(rows) == EXPECTED_SOURCE_ROWS_SHA256, "source row hash drift")

    identifiers: set[str] = set()
    questions: set[str] = set()
    transition_count = 0
    for position, row in enumerate(rows):
        _require(
            type(row) is dict and set(row) == SOURCE_ROW_KEYS,
            "wrong source row schema",
        )
        family = FAMILIES[position // SOURCE_PER_FAMILY]
        family_index = position % SOURCE_PER_FAMILY
        expected_id = f"{family}_{family_index:03d}"
        _require(
            row.get("family") == family and row.get("id") == expected_id,
            "source rows left frozen family/id order",
        )
        _require(expected_id not in identifiers, "duplicate source row id")
        identifiers.add(expected_id)
        question = row.get("question")
        _require(
            isinstance(question, str) and question not in questions,
            "invalid or duplicate source question",
        )
        questions.add(question)
        _require(type(row.get("answer")) is int, "source answer is not an integer")

        start, schedule, details = parse_question(family, question)
        _require(
            row.get("stratum")
            == _expected_stratum(family, family_index, details, row["answer"]),
            "wrong frozen source stratum",
        )
        verified_start, verified_schedule = reconstruct_schedule(row)
        _require(
            verified_start == start and verified_schedule == schedule,
            "independent source reconstruction mismatch",
        )
        state = start
        for operation, operand in schedule:
            state = apply_operation(state, operation, operand)
            transition_count += 1
        _require(state == row["answer"], "source replay does not match answer")

    _require(
        transition_count == SOURCE_TRANSITION_COUNT,
        "wrong frozen source transition count",
    )
    return rows, transition_count


def load_frozen_source(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, str]:
    source_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    with os.fdopen(descriptor, "rb") as source_file:
        metadata = os.fstat(source_file.fileno())
        _require(stat.S_ISREG(metadata.st_mode), "source artifact is not regular")
        if metadata.st_mode & 0o222:
            raise PermissionError("frozen source artifact must have no write bits")
        payload = source_file.read()
    source_sha256 = sha256_bytes(payload)
    _require(
        source_sha256 == EXPECTED_SOURCE_SHA256,
        "source artifact hash does not match frozen source",
    )
    source = strict_json_loads(payload)
    rows, transitions = audit_source(source)
    return source, rows, transitions, source_sha256


def select_frozen_subset(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int]:
    _require(len(rows) == SOURCE_CASE_COUNT, "subset selection needs all source rows")
    selected: list[Mapping[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        start = family_index * SOURCE_PER_FAMILY
        block = rows[start : start + SOURCE_PER_FAMILY]
        _require(
            all(row.get("family") == family for row in block),
            "source family block changed before subset selection",
        )
        selected.extend(block[:DIAGNOSTIC_PER_FAMILY])
    expected_ids = [
        f"{family}_{index:03d}"
        for family in FAMILIES
        for index in range(DIAGNOSTIC_PER_FAMILY)
    ]
    _require(len(selected) == DIAGNOSTIC_CASE_COUNT, "wrong subset case count")
    _require(
        [row.get("id") for row in selected] == expected_ids,
        "subset is not first 16 rows per frozen family block",
    )
    _require(
        digest_rows(selected) == EXPECTED_SUBSET_ROWS_SHA256,
        "frozen subset row hash mismatch",
    )
    transitions = sum(len(reconstruct_schedule(row)[1]) for row in selected)
    _require(
        transitions == DIAGNOSTIC_TRANSITION_COUNT,
        "wrong diagnostic transition count",
    )
    return selected, transitions


def render_residual_suffix(schedule: Sequence[tuple[str, int]]) -> str:
    return json.dumps(
        [[operation, operand] for operation, operand in schedule],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def full_source_cursor_prompt(question: str, index: int) -> str:
    return (
        "Task: Select the operation at the supplied cursor.\n"
        f"Source: {question}\n"
        f"Step index (zero-based): {index}\n"
        "Candidate operations: add, subtract, multiply, remainder.\n"
        "The operation at that cursor is"
    )


def residual_suffix_head_prompt(residual_suffix: Sequence[tuple[str, int]]) -> str:
    return (
        "Task: Select the first operation in the supplied residual suffix.\n"
        f"Residual suffix (read-only JSON): {render_residual_suffix(residual_suffix)}\n"
        "Candidate operations: add, subtract, multiply, remainder.\n"
        "The residual head operation is"
    )


def residual_suffix_oracle_state_prompt(
    current_state: int, residual_suffix: Sequence[tuple[str, int]]
) -> str:
    return (
        "Task: Select the first operation in the supplied residual suffix.\n"
        f"Current state (oracle-supplied for this arm): {current_state}\n"
        f"Residual suffix (read-only JSON): {render_residual_suffix(residual_suffix)}\n"
        "Candidate operations: add, subtract, multiply, remainder.\n"
        "The residual head operation is"
    )


def build_frozen_transitions(
    subset_rows: Sequence[Mapping[str, Any]],
) -> tuple[FrozenTransition, ...]:
    transitions: list[FrozenTransition] = []
    for row in subset_rows:
        current_state, schedule = reconstruct_schedule(row)
        for index, (operation, operand) in enumerate(schedule):
            residual_suffix = tuple(schedule[index:])
            prompts = (
                (
                    FULL_SOURCE_CURSOR,
                    full_source_cursor_prompt(row["question"], index),
                ),
                (
                    RESIDUAL_SUFFIX_HEAD,
                    residual_suffix_head_prompt(residual_suffix),
                ),
                (
                    RESIDUAL_SUFFIX_ORACLE_STATE,
                    residual_suffix_oracle_state_prompt(current_state, residual_suffix),
                ),
            )
            transitions.append(
                FrozenTransition(
                    row_id=row["id"],
                    family=row["family"],
                    question=row["question"],
                    index=index,
                    gold_operation=operation,
                    current_state=current_state,
                    residual_suffix=residual_suffix,
                    prompts=prompts,
                )
            )
            current_state = apply_operation(current_state, operation, operand)

    _require(
        len(transitions) == DIAGNOSTIC_TRANSITION_COUNT,
        "wrong frozen transition construction count",
    )
    _require(
        Counter(item.family for item in transitions) == EXPECTED_TRANSITIONS_BY_FAMILY,
        "frozen family transition geometry changed",
    )
    _require(
        Counter(item.index for item in transitions) == EXPECTED_TRANSITIONS_BY_INDEX,
        "frozen index transition geometry changed",
    )
    _require(
        Counter(item.gold_operation for item in transitions)
        == EXPECTED_TRANSITIONS_BY_OPERATION,
        "frozen operation transition geometry changed",
    )
    return tuple(transitions)


def candidate_manifest_value() -> dict[str, Any]:
    return {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "candidate_order": list(OPERATIONS),
        "candidates": [
            {
                "operation": candidate.operation,
                "text": candidate.text,
                "token_id": candidate.token_id,
            }
            for candidate in CANDIDATES
        ],
        "selection_rule": "unique_largest_logit_among_four_candidates",
        "ties": "incorrect_no_prediction",
        "candidate_tokens_appended_to_model_input": 0,
    }


def candidate_manifest_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(candidate_manifest_value()))


def prompt_manifest_value(
    transitions: Sequence[FrozenTransition],
) -> dict[str, Any]:
    return {
        "schema": PROMPT_MANIFEST_SCHEMA,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
        "subset_rows_sha256": EXPECTED_SUBSET_ROWS_SHA256,
        "candidate_manifest_sha256": candidate_manifest_sha256(),
        "arm_order": list(ARMS),
        "case_count": DIAGNOSTIC_CASE_COUNT,
        "transition_count": DIAGNOSTIC_TRANSITION_COUNT,
        "rows": [
            {
                "id": transition.row_id,
                "family": transition.family,
                "index": transition.index,
                "gold_operation": transition.gold_operation,
                "current_state": transition.current_state,
                "residual_suffix": [list(step) for step in transition.residual_suffix],
                "prompts": {
                    arm: transition.prompt_for(arm)
                    for arm in ARMS
                },
            }
            for transition in transitions
        ],
    }


def prompt_manifest_sha256(transitions: Sequence[FrozenTransition]) -> str:
    return sha256_bytes(canonical_json_bytes(prompt_manifest_value(transitions)))


def verify_untokenized_manifests(
    transitions: Sequence[FrozenTransition],
) -> tuple[str, str]:
    candidate_sha256 = candidate_manifest_sha256()
    _require(
        candidate_sha256 == EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "candidate manifest SHA-256 mismatch",
    )
    prompt_sha256 = prompt_manifest_sha256(transitions)
    _require(
        prompt_sha256 == EXPECTED_PROMPT_MANIFEST_SHA256,
        "prompt manifest SHA-256 mismatch",
    )
    return candidate_sha256, prompt_sha256


def _verify_candidate_tokenizer_contract(tokenizer: Any) -> None:
    _require(tuple(candidate.operation for candidate in CANDIDATES) == OPERATIONS, "candidate order drift")
    token_ids: set[int] = set()
    for candidate in CANDIDATES:
        encoded = tuple(tokenizer.encode(candidate.text).ids)
        _require(
            encoded == (candidate.token_id,),
            f"candidate is not its frozen one token: {candidate.operation}",
        )
        _require(
            tokenizer.decode([candidate.token_id]) == candidate.text,
            f"candidate token decode drift: {candidate.operation}",
        )
        _require(candidate.token_id not in token_ids, "duplicate candidate token id")
        token_ids.add(candidate.token_id)


def prepare_transitions(
    transitions: Sequence[FrozenTransition], tokenizer: Any
) -> tuple[PreparedTransition, ...]:
    _verify_candidate_tokenizer_contract(tokenizer)
    prepared: list[PreparedTransition] = []
    for transition in transitions:
        prompts: list[PreparedPrompt] = []
        for arm in ARMS:
            text = transition.prompt_for(arm)
            prompt_token_ids = tuple(tokenizer.encode(text).ids)
            _require(bool(prompt_token_ids), "prompt tokenized empty")
            _require(
                len(prompt_token_ids) < EXPECTED_CONTEXT_LENGTH,
                "prompt reaches or exceeds frozen context length",
            )
            for candidate in CANDIDATES:
                combined = tuple(tokenizer.encode(text + candidate.text).ids)
                _require(
                    combined[: len(prompt_token_ids)] == prompt_token_ids,
                    f"candidate retokenized prompt boundary at {transition.row_id}/{transition.index}/{arm}/{candidate.operation}",
                )
                _require(
                    combined[len(prompt_token_ids) :] == (candidate.token_id,),
                    f"candidate suffix is not one frozen token at {transition.row_id}/{transition.index}/{arm}/{candidate.operation}",
                )
            prompts.append(PreparedPrompt(arm, text, prompt_token_ids))
        prepared.append(PreparedTransition(transition, tuple(prompts)))
    _require(len(prepared) == DIAGNOSTIC_TRANSITION_COUNT, "prepared transition count drift")
    return tuple(prepared)


def tokenized_manifest_value(
    prepared: Sequence[PreparedTransition],
) -> dict[str, Any]:
    return {
        "schema": TOKENIZED_MANIFEST_SCHEMA,
        "prompt_manifest_sha256": prompt_manifest_sha256(
            [item.frozen for item in prepared]
        ),
        "candidate_manifest_sha256": candidate_manifest_sha256(),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "candidate_token_ids": {
            candidate.operation: candidate.token_id for candidate in CANDIDATES
        },
        "rows": [
            {
                "id": item.frozen.row_id,
                "index": item.frozen.index,
                "prompts": {
                    prompt.arm: {
                        "prompt_token_ids": list(prompt.token_ids),
                        "candidate_suffix_token_ids": {
                            candidate.operation: [candidate.token_id]
                            for candidate in CANDIDATES
                        },
                    }
                    for prompt in item.prompts
                },
            }
            for item in prepared
        ],
    }


def tokenized_manifest_sha256(prepared: Sequence[PreparedTransition]) -> str:
    return sha256_bytes(canonical_json_bytes(tokenized_manifest_value(prepared)))


def prompt_token_counts(
    prepared: Sequence[PreparedTransition],
) -> tuple[dict[str, int], int, int]:
    by_arm = {
        arm: sum(len(item.prompt_for(arm).token_ids) for item in prepared)
        for arm in ARMS
    }
    total = sum(by_arm.values())
    maximum = max(
        len(item.prompt_for(arm).token_ids) for item in prepared for arm in ARMS
    )
    return by_arm, total, maximum


def verify_tokenized_manifest(prepared: Sequence[PreparedTransition]) -> str:
    actual = tokenized_manifest_sha256(prepared)
    _require(
        actual == EXPECTED_TOKENIZED_MANIFEST_SHA256,
        "tokenized prompt manifest SHA-256 mismatch",
    )
    by_arm, total, maximum = prompt_token_counts(prepared)
    _require(by_arm == EXPECTED_PROMPT_TOKENS_BY_ARM, "prompt token counts by arm drift")
    _require(total == EXPECTED_PROMPT_TOKENS_TOTAL, "total prompt token count drift")
    _require(maximum == EXPECTED_MAX_PROMPT_TOKENS, "maximum prompt token count drift")
    return actual


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def score_candidate_logits(
    gold_operation: str, candidate_logits: Sequence[float]
) -> dict[str, Any]:
    _require(gold_operation in OPERATIONS, "gold operation is not a candidate")
    _require(
        len(candidate_logits) == len(CANDIDATES),
        "candidate logit count must be exactly four",
    )
    logits = [
        _finite_float(value, f"candidate logit {index}")
        for index, value in enumerate(candidate_logits)
    ]
    maximum = max(logits)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    log_probabilities = [value - normalizer for value in logits]
    probabilities = [math.exp(value) for value in log_probabilities]
    top_operations = [
        candidate.operation
        for candidate, value in zip(CANDIDATES, logits, strict=True)
        if value == maximum
    ]
    unique_top1 = len(top_operations) == 1
    prediction = top_operations[0] if unique_top1 else None
    gold_index = OPERATIONS.index(gold_operation)
    best_incorrect = max(
        value for index, value in enumerate(logits) if index != gold_index
    )
    return {
        "candidates": [
            {
                "operation": candidate.operation,
                "text": candidate.text,
                "token_id": candidate.token_id,
                "logit": logit,
                "restricted_log_probability": log_probability,
                "restricted_probability": probability,
            }
            for candidate, logit, log_probability, probability in zip(
                CANDIDATES,
                logits,
                log_probabilities,
                probabilities,
                strict=True,
            )
        ],
        "top_operations": top_operations,
        "unique_top1": unique_top1,
        "prediction": prediction,
        "correct": prediction == gold_operation,
        "gold_operation": gold_operation,
        "gold_logit_margin_to_best_incorrect": logits[gold_index] - best_incorrect,
        "gold_restricted_log_probability": log_probabilities[gold_index],
        "gold_restricted_probability": probabilities[gold_index],
    }


def build_arm_record(
    prompt: PreparedPrompt,
    gold_operation: str,
    candidate_logits: Sequence[float],
) -> dict[str, Any]:
    return {
        "prompt": prompt.text,
        "prompt_token_ids": list(prompt.token_ids),
        "prompt_token_count": len(prompt.token_ids),
        **score_candidate_logits(gold_operation, candidate_logits),
    }


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": int(numerator), "denominator": int(denominator)}


def _flatten_entries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        for step in row["steps"]:
            entries.append(
                {
                    "family": row["family"],
                    "index": step["index"],
                    "operation": step["gold_operation"],
                    "step": step,
                }
            )
    return entries


def _arm_accuracy(entries: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    records = [entry["step"][arm] for entry in entries]
    denominator = len(records)
    unique = sum(record["unique_top1"] for record in records)
    correct = sum(record["correct"] for record in records)
    return {
        "calls": denominator,
        "unique_top1": _ratio(unique, denominator),
        "ties": _ratio(denominator - unique, denominator),
        "accuracy": _ratio(correct, denominator),
    }


def _group_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "transition_count": len(entries),
        "model_forward_calls": len(entries) * len(ARMS),
        "candidate_logit_values_scored": len(entries) * len(ARMS) * len(CANDIDATES),
        "by_arm": {arm: _arm_accuracy(entries, arm) for arm in ARMS},
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = _flatten_entries(rows)
    _require(len(entries) == DIAGNOSTIC_TRANSITION_COUNT, "summary transition count drift")
    by_family = {
        family: {
            "case_count": DIAGNOSTIC_PER_FAMILY,
            **_group_summary([entry for entry in entries if entry["family"] == family]),
        }
        for family in FAMILIES
    }
    by_index = {
        str(index): _group_summary(
            [entry for entry in entries if entry["index"] == index]
        )
        for index in EXPECTED_TRANSITIONS_BY_INDEX
    }
    by_operation = {
        operation: _group_summary(
            [entry for entry in entries if entry["operation"] == operation]
        )
        for operation in OPERATIONS
    }
    _require(
        {family: value["transition_count"] for family, value in by_family.items()}
        == EXPECTED_TRANSITIONS_BY_FAMILY,
        "summary family denominators drift",
    )
    _require(
        {int(index): value["transition_count"] for index, value in by_index.items()}
        == EXPECTED_TRANSITIONS_BY_INDEX,
        "summary index denominators drift",
    )
    _require(
        {operation: value["transition_count"] for operation, value in by_operation.items()}
        == EXPECTED_TRANSITIONS_BY_OPERATION,
        "summary operation denominators drift",
    )
    return {
        "case_count": DIAGNOSTIC_CASE_COUNT,
        **_group_summary(entries),
        "by_family": by_family,
        "by_step_index": by_index,
        "by_operation": by_operation,
    }


def resource_ledger(prepared: Sequence[PreparedTransition]) -> dict[str, Any]:
    by_arm, total_prompt_tokens, maximum = prompt_token_counts(prepared)
    ledger = {
        "source_rows_read": SOURCE_CASE_COUNT,
        "source_transitions_audited": SOURCE_TRANSITION_COUNT,
        "subset_cases": DIAGNOSTIC_CASE_COUNT,
        "subset_transitions": DIAGNOSTIC_TRANSITION_COUNT,
        "arms_per_transition": len(ARMS),
        "prompts_scored": EXPECTED_MODEL_CALLS,
        "model_forward_calls": EXPECTED_MODEL_CALLS,
        "candidate_classes_per_prompt": len(CANDIDATES),
        "candidate_logit_values_scored": EXPECTED_CANDIDATE_LOGIT_VALUES,
        "teacher_forced_candidate_targets": EXPECTED_CANDIDATE_LOGIT_VALUES,
        "prompt_tokens_by_arm": by_arm,
        "prompt_tokens_replayed": total_prompt_tokens,
        "maximum_prompt_tokens": maximum,
        "model_input_token_positions": total_prompt_tokens,
        "candidate_tokens_appended_to_model_input": 0,
        "checkpoint_hash_passes": 4,
        "tokenizer_hash_passes": 4,
        "source_hash_passes": 4,
        "implementation_hash_passes": 5,
        "tokenizer_loads": 2,
        "h100_preflight_allocations": 1,
        "model_loads": 1,
        "generated_tokens": 0,
        "sampled_tokens": 0,
        "training_tokens": 0,
        "retries": 0,
        "repairs": 0,
        "searches": 0,
        "threshold_searches": 0,
        "verifier_feedback_calls": 0,
        "external_generation_calls": 0,
        "quarantine_result_files_created": 1,
        "preserved_result_copies": 2,
        "read_only_receipt_files": 1,
        "mutable_sidecars": 0,
        "authenticated_git_fetches": 1,
        "scheduler_log_files": 1,
        "mutable_scheduler_log_files": 1,
    }
    _require(by_arm == EXPECTED_PROMPT_TOKENS_BY_ARM, "ledger arm token counts drift")
    _require(total_prompt_tokens == EXPECTED_PROMPT_TOKENS_TOTAL, "ledger token total drift")
    _require(maximum == EXPECTED_MAX_PROMPT_TOKENS, "ledger max prompt length drift")
    return ledger


def evaluate_prepared_transitions(
    prepared: Sequence[PreparedTransition],
    scorer: Callable[[PreparedTransition, PreparedPrompt], Sequence[float]],
    *,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    _require(
        len(prepared) == DIAGNOSTIC_TRANSITION_COUNT,
        "evaluation transition count must be frozen",
    )
    rows: list[dict[str, Any]] = []
    observed_calls: Counter[str] = Counter()
    current_row: dict[str, Any] | None = None
    for offset, item in enumerate(prepared, 1):
        frozen = item.frozen
        if current_row is None or current_row["id"] != frozen.row_id:
            current_row = {"id": frozen.row_id, "family": frozen.family, "steps": []}
            rows.append(current_row)
        _require(current_row["family"] == frozen.family, "row family changed in evaluation")
        step: dict[str, Any] = {
            "index": frozen.index,
            "gold_operation": frozen.gold_operation,
            "current_state": frozen.current_state,
            "residual_suffix": [list(value) for value in frozen.residual_suffix],
        }
        for arm in ARMS:
            prompt = item.prompt_for(arm)
            logits = scorer(item, prompt)
            observed_calls[arm] += 1
            step[arm] = build_arm_record(prompt, frozen.gold_operation, logits)
        current_row["steps"].append(step)
        if progress and offset % 16 == 0:
            print(
                f"[operation-selection-likelihood] {offset}/{len(prepared)} transitions",
                flush=True,
            )

    _require(len(rows) == DIAGNOSTIC_CASE_COUNT, "evaluation row count drift")
    expected_calls = Counter({arm: DIAGNOSTIC_TRANSITION_COUNT for arm in ARMS})
    _require(observed_calls == expected_calls, "model forward count by arm drift")
    _require(sum(observed_calls.values()) == EXPECTED_MODEL_CALLS, "total model call drift")
    return rows, observed_calls


def audit_execution(
    rows: Any,
    prepared: Sequence[PreparedTransition],
    observed_calls: Mapping[str, int],
) -> None:
    _require(isinstance(rows, list) and len(rows) == DIAGNOSTIC_CASE_COUNT, "wrong result row count")
    flat_steps: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    prepared_offset = 0
    for row in rows:
        _require(
            type(row) is dict and set(row) == {"id", "family", "steps"},
            "wrong result row schema",
        )
        _require(prepared_offset < len(prepared), "result has extra rows")
        first = prepared[prepared_offset].frozen
        _require(
            row["id"] == first.row_id and row["family"] == first.family,
            "result row identity drift",
        )
        steps = row["steps"]
        expected_step_count = EXPECTED_TRANSITIONS_BY_FAMILY[first.family] // DIAGNOSTIC_PER_FAMILY
        _require(
            isinstance(steps, list) and len(steps) == expected_step_count,
            "result row step count drift",
        )
        for step in steps:
            _require(prepared_offset < len(prepared), "result has extra steps")
            item = prepared[prepared_offset]
            frozen = item.frozen
            _require(frozen.row_id == row["id"], "result steps crossed source rows")
            expected_step_keys = {
                "index",
                "gold_operation",
                "current_state",
                "residual_suffix",
                *ARMS,
            }
            _require(
                type(step) is dict and set(step) == expected_step_keys,
                "wrong result step schema",
            )
            expected_fixed = {
                "index": frozen.index,
                "gold_operation": frozen.gold_operation,
                "current_state": frozen.current_state,
                "residual_suffix": [list(value) for value in frozen.residual_suffix],
            }
            _require(
                all(step[key] == value for key, value in expected_fixed.items()),
                "result step geometry drift",
            )
            for arm in ARMS:
                record = step[arm]
                _require(isinstance(record, dict), "arm score record is not an object")
                candidate_rows = record.get("candidates")
                _require(
                    isinstance(candidate_rows, list)
                    and len(candidate_rows) == len(CANDIDATES),
                    "arm candidate score rows drift",
                )
                logits = []
                for candidate_row, candidate in zip(
                    candidate_rows, CANDIDATES, strict=True
                ):
                    _require(
                        isinstance(candidate_row, dict)
                        and candidate_row.get("operation") == candidate.operation
                        and candidate_row.get("text") == candidate.text
                        and candidate_row.get("token_id") == candidate.token_id,
                        "arm candidate identity drift",
                    )
                    logits.append(candidate_row.get("logit"))
                expected_record = build_arm_record(
                    item.prompt_for(arm), frozen.gold_operation, logits
                )
                _require(record == expected_record, "arm prompt or derived score drift")
            flat_steps.append((row, step))
            prepared_offset += 1
    _require(prepared_offset == len(prepared), "result omitted prepared transitions")
    expected_calls = {arm: DIAGNOSTIC_TRANSITION_COUNT for arm in ARMS}
    _require(dict(observed_calls) == expected_calls, "observed call ledger drift")
    _require(len(flat_steps) * len(ARMS) == EXPECTED_MODEL_CALLS, "preserved call count drift")


def diagnostic_scope() -> dict[str, Any]:
    return {
        "status": "adaptive_exploratory_teacher_forced_decomposition_only",
        "question": "restricted_next_operation_preference",
        "held_out_confirmation": False,
        "reasoning_claim": "none",
        "free_decoding_claim": "none",
        "promotion_decision": "none",
        "training_action": False,
        "production_submission": False,
    }


def candidate_contract() -> dict[str, Any]:
    return {
        **candidate_manifest_value(),
        "metric": "four_candidate_restricted_next_token_likelihood",
        "model_forward_per_prompt": 1,
        "candidate_logits_reused_from_same_forward": True,
        "full_vocabulary_rank_reported": False,
    }


def prompt_contract() -> dict[str, Any]:
    return {
        "arm_order": list(ARMS),
        "prompt_manifest_sha256": EXPECTED_PROMPT_MANIFEST_SHA256,
        "tokenized_prompt_manifest_sha256": EXPECTED_TOKENIZED_MANIFEST_SHA256,
        "prompt_truncation": "forbidden",
        "completion_boundary": "prompt_ends_with_is_candidate_includes_one_leading_space",
        "exposure": {
            FULL_SOURCE_CURSOR: ["full_source_question", "zero_based_cursor"],
            RESIDUAL_SUFFIX_HEAD: ["oracle_residual_schedule_suffix"],
            RESIDUAL_SUFFIX_ORACLE_STATE: [
                "oracle_residual_schedule_suffix",
                "oracle_current_numeric_state",
            ],
        },
        "arm_interpretation": {
            FULL_SOURCE_CURSOR: "source_conditioned_operation_preference_only",
            RESIDUAL_SUFFIX_HEAD: "literal_label_copy_control_only",
            RESIDUAL_SUFFIX_ORACLE_STATE: "literal_label_copy_plus_state_control_only",
        },
        "excluded_from_all_prompts": [
            "gold_final_answer",
            "prior_model_response",
            "candidate_logit_or_score",
            "verifier_feedback",
        ],
    }


def implementation_source_paths() -> dict[str, Path]:
    train_dir = Path(__file__).resolve().parent
    return {
        "preregistration": ROOT / "R12_OPERATION_SELECTION_LIKELIHOOD_PREREG.md",
        "evaluator": Path(__file__).resolve(),
        "tests": train_dir / "test_probe_operation_selection_likelihood.py",
        "job": train_dir / "jobs/probe_operation_selection_likelihood.sbatch",
        "model_loader": train_dir / "model.py",
        "inherited_operation_cursor_contract": ROOT / "R12_OPERATION_CURSOR_DIAGNOSTIC.md",
        "inherited_operation_cursor_geometry": train_dir / "eval_operation_cursor.py",
    }


def hash_implementation(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: hash_regular_file(path) for name, path in paths.items()}


def snapshot_implementation(
    paths: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, bytes]]:
    payloads = {
        name: read_regular_file_bytes(path)
        for name, path in paths.items()
    }
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    return hashes, payloads


def implementation_manifest_sha256(hashes: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(hashes)))


def build_result(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    source_sha256: str,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    candidate_sha256: str,
    prompt_sha256: str,
    tokenized_sha256: str,
    implementation_hashes: Mapping[str, str],
    frozen_commit: str,
    device: Mapping[str, Any],
    prepared: Sequence[PreparedTransition],
    rows: list[dict[str, Any]],
    observed_calls: Mapping[str, int],
) -> dict[str, Any]:
    _require(
        isinstance(frozen_commit, str) and FROZEN_COMMIT_RE.fullmatch(frozen_commit),
        "frozen commit is not a full lowercase Git SHA-1",
    )
    audit_execution(rows, prepared, observed_calls)
    ledger = resource_ledger(prepared)
    summary = build_summary(rows)
    result = {
        "schema": RESULT_SCHEMA,
        "diagnostic_scope": diagnostic_scope(),
        "bindings": {
            "frozen_commit": frozen_commit,
            "source": str(Path(source_path).resolve()),
            "source_sha256": source_sha256,
            "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
            "subset_rows_sha256": EXPECTED_SUBSET_ROWS_SHA256,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "tokenizer": str(Path(tokenizer_path).resolve()),
            "tokenizer_sha256": tokenizer_sha256,
            "candidate_manifest_sha256": candidate_sha256,
            "prompt_manifest_sha256": prompt_sha256,
            "tokenized_prompt_manifest_sha256": tokenized_sha256,
            "implementation_sha256": dict(implementation_hashes),
            "implementation_manifest_sha256": implementation_manifest_sha256(
                implementation_hashes
            ),
        },
        "candidate_contract": candidate_contract(),
        "prompt_contract": prompt_contract(),
        "device": dict(device),
        "resource_ledger": ledger,
        "summary": summary,
        "integrity": {
            "source_reconstructed_from_natural_language": True,
            "source_schedule_and_answers_replayed": True,
            "score_blind_subset_reconstructed": True,
            "all_prompts_reconstructed": True,
            "candidate_boundaries_verified_one_token": True,
            "input_hashes_stable_before_and_after": True,
            "implementation_hashes_stable_before_and_after": True,
            "all_scores_recomputed_from_preserved_candidate_logits": True,
            "model_call_accounting_exact": True,
            "no_generation_training_retry_or_repair": True,
            "exclusive_read_only_output": True,
        },
        "rows": rows,
    }
    audit_preserved_result(
        result,
        source_path=source_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        candidate_sha256=candidate_sha256,
        prompt_sha256=prompt_sha256,
        tokenized_sha256=tokenized_sha256,
        implementation_hashes=implementation_hashes,
        frozen_commit=frozen_commit,
        device=device,
        prepared=prepared,
    )
    return result


def audit_preserved_result(
    result: Any,
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    source_sha256: str,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    candidate_sha256: str,
    prompt_sha256: str,
    tokenized_sha256: str,
    implementation_hashes: Mapping[str, str],
    frozen_commit: str,
    device: Mapping[str, Any],
    prepared: Sequence[PreparedTransition],
) -> bool:
    expected_keys = {
        "schema",
        "diagnostic_scope",
        "bindings",
        "candidate_contract",
        "prompt_contract",
        "device",
        "resource_ledger",
        "summary",
        "integrity",
        "rows",
    }
    _require(type(result) is dict and set(result) == expected_keys, "wrong result schema")
    _require(result["schema"] == RESULT_SCHEMA, "wrong result schema id")
    _require(
        isinstance(frozen_commit, str) and FROZEN_COMMIT_RE.fullmatch(frozen_commit),
        "frozen commit is not a full lowercase Git SHA-1",
    )
    expected_bindings = {
        "frozen_commit": frozen_commit,
        "source": str(Path(source_path).resolve()),
        "source_sha256": source_sha256,
        "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
        "subset_rows_sha256": EXPECTED_SUBSET_ROWS_SHA256,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "tokenizer": str(Path(tokenizer_path).resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "candidate_manifest_sha256": candidate_sha256,
        "prompt_manifest_sha256": prompt_sha256,
        "tokenized_prompt_manifest_sha256": tokenized_sha256,
        "implementation_sha256": dict(implementation_hashes),
        "implementation_manifest_sha256": implementation_manifest_sha256(
            implementation_hashes
        ),
    }
    _require(result["diagnostic_scope"] == diagnostic_scope(), "scope contract drift")
    _require(result["bindings"] == expected_bindings, "binding contract drift")
    _require(result["candidate_contract"] == candidate_contract(), "candidate contract drift")
    _require(result["prompt_contract"] == prompt_contract(), "prompt contract drift")
    _require(result["device"] == dict(device), "device record drift")
    expected_calls = {arm: DIAGNOSTIC_TRANSITION_COUNT for arm in ARMS}
    audit_execution(result["rows"], prepared, expected_calls)
    _require(result["resource_ledger"] == resource_ledger(prepared), "resource ledger drift")
    _require(result["summary"] == build_summary(result["rows"]), "summary drift")
    expected_integrity = {
        "source_reconstructed_from_natural_language": True,
        "source_schedule_and_answers_replayed": True,
        "score_blind_subset_reconstructed": True,
        "all_prompts_reconstructed": True,
        "candidate_boundaries_verified_one_token": True,
        "input_hashes_stable_before_and_after": True,
        "implementation_hashes_stable_before_and_after": True,
        "all_scores_recomputed_from_preserved_candidate_logits": True,
        "model_call_accounting_exact": True,
        "no_generation_training_retry_or_repair": True,
        "exclusive_read_only_output": True,
    }
    _require(result["integrity"] == expected_integrity, "integrity record drift")
    return True


@torch.inference_mode()
def next_operation_candidate_logits(
    model: Any, prompt: PreparedPrompt, device: str
) -> tuple[float, ...]:
    tokens = torch.tensor([prompt.token_ids], dtype=torch.long, device=device)
    logits, _ = model(tokens)
    _require(
        logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[1] == len(prompt.token_ids),
        "model returned wrong logit shape",
    )
    candidate_ids = torch.tensor(
        [candidate.token_id for candidate in CANDIDATES],
        dtype=torch.long,
        device=device,
    )
    selected = logits[0, -1].index_select(0, candidate_ids).float()
    _require(selected.numel() == len(CANDIDATES), "candidate logit gather drift")
    values = tuple(float(value) for value in selected.cpu().tolist())
    for index, value in enumerate(values):
        _finite_float(value, f"model candidate logit {index}")
    return values


def preflight_h100() -> tuple[str, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; canonical diagnostic requires H100")
    visible = torch.cuda.device_count()
    _require(visible == 1, "canonical diagnostic requires exactly one visible GPU")
    device = "cuda:0"
    try:
        allocation = torch.zeros(1, device=device)
        del allocation
    except Exception as error:
        raise RuntimeError("H100 allocation preflight failed") from error
    name = torch.cuda.get_device_name(0)
    _require("H100" in name.upper(), f"canonical diagnostic requires H100, got {name}")
    major, minor = torch.cuda.get_device_capability(0)
    return device, {
        "type": "cuda",
        "index": 0,
        "name": name,
        "compute_capability": f"{major}.{minor}",
        "visible_device_count": visible,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "autocast": False,
        "candidate_logit_dtype": "float32",
    }


def validate_checkpoint_metadata(checkpoint: Any) -> Mapping[str, Any]:
    _require(isinstance(checkpoint, dict), "checkpoint must be a dictionary")
    _require(
        type(checkpoint.get("step")) is int
        and checkpoint["step"] == EXPECTED_CHECKPOINT_STEP,
        "checkpoint step differs from frozen raw-260k step",
    )
    config = checkpoint.get("cfg")
    _require(isinstance(config, dict), "checkpoint config is missing")
    expected_config = {
        "seq_len": EXPECTED_CONTEXT_LENGTH,
        "n_layer": 30,
        "d_model": 576,
        "n_loop": 1,
    }
    for key, expected in expected_config.items():
        _require(config.get(key) == expected, f"checkpoint config {key} drift")
    _require(isinstance(checkpoint.get("model"), dict), "checkpoint model state is missing")
    return config


def model_classes_from_source(
    model_source_payload: bytes, source_path: str | Path
) -> tuple[type[Any], type[Any]]:
    """Compile the attested source snapshot directly, never through import caches."""
    module_name = "_shohin_attested_operation_likelihood_model"
    _require(module_name not in sys.modules, "attested model module name is already loaded")
    module = types.ModuleType(module_name)
    module.__file__ = str(Path(source_path).resolve())
    sys.modules[module_name] = module
    try:
        code = compile(model_source_payload, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        gpt = module.__dict__.get("GPT")
        config = module.__dict__.get("GPTConfig")
        _require(isinstance(gpt, type) and isinstance(config, type), "model classes are missing")
        return gpt, config
    finally:
        del sys.modules[module_name]


def load_model(
    checkpoint_payload: bytes,
    model_source_payload: bytes,
    model_source_path: str | Path,
    device: str,
) -> tuple[int, Any]:
    GPT, GPTConfig = model_classes_from_source(model_source_payload, model_source_path)

    checkpoint = torch.load(
        io.BytesIO(checkpoint_payload), map_location="cpu", weights_only=False
    )
    config = validate_checkpoint_metadata(checkpoint)
    model = GPT(GPTConfig(**config))
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    model = model.to(device).eval()
    step = int(checkpoint["step"])
    del checkpoint
    return step, model


OUTPUT_PREFIX = "raw260k_operation_selection_likelihood_"


def validate_output_path(path: str | Path) -> Path:
    destination = Path(path)
    _require(
        destination.name.startswith(OUTPUT_PREFIX) and destination.suffix == ".json",
        "output filename does not match frozen pattern",
    )
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing existing output: {destination}")
    _require(destination.parent.is_dir(), "output parent directory does not exist")
    _require(not destination.parent.is_symlink(), "output parent may not be a symlink")
    return destination


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    destination = validate_output_path(path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    metadata = destination.lstat()
    _require(stat.S_ISREG(metadata.st_mode), "output is not a regular file")
    if metadata.st_mode & 0o222:
        raise PermissionError("diagnostic output remained writable")
    try:
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError:
        pass
    return sha256_bytes(payload)


def audit_existing_result_file(
    *,
    result_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    source_path: str | Path,
    frozen_commit: str,
) -> str:
    """Reconstruct and fully audit a completed result without loading the model."""
    result_payload = read_regular_file_bytes(result_path, require_read_only=True)
    result = strict_json_loads(result_payload)

    _, source_rows, source_transitions, source_sha256 = load_frozen_source(
        source_path
    )
    _require(source_transitions == SOURCE_TRANSITION_COUNT, "source transition audit drift")
    subset_rows, subset_transitions = select_frozen_subset(source_rows)
    _require(subset_transitions == DIAGNOSTIC_TRANSITION_COUNT, "subset transition audit drift")
    frozen_transitions = build_frozen_transitions(subset_rows)
    candidate_sha256, prompt_sha256 = verify_untokenized_manifests(
        frozen_transitions
    )

    checkpoint_sha256 = require_file_sha256(
        checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
    )
    tokenizer_payload, tokenizer_sha256 = require_file_bytes_sha256(
        tokenizer_path, EXPECTED_TOKENIZER_SHA256, "tokenizer"
    )
    implementation_hashes = hash_implementation(implementation_source_paths())
    tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    prepared = prepare_transitions(frozen_transitions, tokenizer)
    tokenized_sha256 = verify_tokenized_manifest(prepared)

    _require(type(result) is dict, "result is not an object")
    device = result.get("device")
    _require(type(device) is dict, "result device record is missing")
    audit_preserved_result(
        result,
        source_path=source_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        candidate_sha256=candidate_sha256,
        prompt_sha256=prompt_sha256,
        tokenized_sha256=tokenized_sha256,
        implementation_hashes=implementation_hashes,
        frozen_commit=frozen_commit,
        device=device,
        prepared=prepared,
    )
    return sha256_bytes(result_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-result")
    parser.add_argument("--frozen-commit", required=True)
    args = parser.parse_args()
    _require(
        FROZEN_COMMIT_RE.fullmatch(args.frozen_commit) is not None,
        "frozen commit is not a full lowercase Git SHA-1",
    )

    if args.audit_result is not None:
        result_sha256 = audit_existing_result_file(
            result_path=args.audit_result,
            checkpoint_path=args.ckpt,
            tokenizer_path=args.tokenizer,
            source_path=args.source,
            frozen_commit=args.frozen_commit,
        )
        print(
            json.dumps(
                {
                    "schema": RESULT_SCHEMA,
                    "audited_result": str(Path(args.audit_result).resolve()),
                    "sha256": result_sha256,
                    "full_preserved_result_audit": True,
                    "model_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    output = validate_output_path(args.out)
    source, source_rows, source_transitions, source_sha256 = load_frozen_source(
        args.source
    )
    _require(source_transitions == SOURCE_TRANSITION_COUNT, "source transition audit drift")
    subset_rows, subset_transitions = select_frozen_subset(source_rows)
    _require(subset_transitions == DIAGNOSTIC_TRANSITION_COUNT, "subset transition audit drift")
    frozen_transitions = build_frozen_transitions(subset_rows)
    candidate_sha256, prompt_sha256 = verify_untokenized_manifests(
        frozen_transitions
    )

    checkpoint_payload, checkpoint_sha256 = require_file_bytes_sha256(
        args.ckpt, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
    )
    tokenizer_payload, tokenizer_sha256 = require_file_bytes_sha256(
        args.tokenizer, EXPECTED_TOKENIZER_SHA256, "tokenizer"
    )
    implementation_paths = implementation_source_paths()
    implementation_hashes, implementation_payloads = snapshot_implementation(
        implementation_paths
    )

    tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    prepared = prepare_transitions(frozen_transitions, tokenizer)
    tokenized_sha256 = verify_tokenized_manifest(prepared)

    device_name, device_record = preflight_h100()
    checkpoint_step, model = load_model(
        checkpoint_payload,
        implementation_payloads["model_loader"],
        implementation_paths["model_loader"],
        device_name,
    )
    del checkpoint_payload
    del implementation_payloads
    _require(checkpoint_step == EXPECTED_CHECKPOINT_STEP, "loaded checkpoint step drift")
    rows, observed_calls = evaluate_prepared_transitions(
        prepared,
        lambda _item, prompt: next_operation_candidate_logits(
            model, prompt, device_name
        ),
        progress=True,
    )

    post_hashes = {
        "checkpoint": hash_regular_file(args.ckpt),
        "tokenizer": hash_regular_file(args.tokenizer),
        "source": hash_regular_file(args.source, require_read_only=True),
    }
    expected_post_hashes = {
        "checkpoint": checkpoint_sha256,
        "tokenizer": tokenizer_sha256,
        "source": source_sha256,
    }
    _require(post_hashes == expected_post_hashes, "bound input changed during evaluation")
    _require(
        hash_implementation(implementation_paths) == implementation_hashes,
        "implementation changed during evaluation",
    )

    result = build_result(
        source_path=args.source,
        checkpoint_path=args.ckpt,
        tokenizer_path=args.tokenizer,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        candidate_sha256=candidate_sha256,
        prompt_sha256=prompt_sha256,
        tokenized_sha256=tokenized_sha256,
        implementation_hashes=implementation_hashes,
        frozen_commit=args.frozen_commit,
        device=device_record,
        prepared=prepared,
        rows=rows,
        observed_calls=observed_calls,
    )
    output_sha256 = write_immutable_json(output, result)
    print(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "out": str(output),
                "sha256": output_sha256,
                "model_forward_calls": result["resource_ledger"][
                    "model_forward_calls"
                ],
                "score_quarantined_pending_full_audit": True,
                "reasoning_claim": "none",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
