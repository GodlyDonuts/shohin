#!/usr/bin/env python3
"""Independent fail-closed assessment of the frozen scheduled confirmation.

This module intentionally does not import the generator or evaluator. It accepts
only the admitted frozen inputs and reconstructs the board, prompts, parses,
scores, paired test, resource ledger, and gates from preserved call records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
BOARD_SCHEMA = "source_scheduled_reasoning_confirmation_v1"
RESULT_SCHEMA = "source_scheduled_reasoning_confirmation_result_v1"
ASSESSMENT_SCHEMA = "source_scheduled_reasoning_confirmation_assessment_v1"
SEED = 2026071502
PER_FAMILY = 64
FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)
EXPECTED_CASE_COUNT = 256
EXPECTED_TRANSITION_COUNT = 704
EXPECTED_CHECKPOINT_STEP = 260000
MAX_NEW_FULL = 128
MAX_NEW_ATOMIC = 48

EXPECTED_BOARD_SHA256 = (
    "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474"
)
EXPECTED_CASES_SHA256 = (
    "4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
)
EXPECTED_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
EXPECTED_IMPLEMENTATION_SHA256 = {
    "contract": "cdca05d9a99a0e341661534433eae5f1351049b4b97f7bf235dcc2cc29edb39d",
    "generator": "4817472dc3ba0e31b3aed6f91ff01c0fe06cba827828cf5b46cd2b0ed79ccf29",
    "evaluator": "acd0b895bf73ceeaab37e70330b5a80097027dc3bb6365b848e63567f93247ca",
    "job": "a34b06ec4da24c5267856f5347a2893fdb64d210a3b56b81df3cb23492122c15",
    "model_loader": "3f0d092fed269a2ca7556a878fbcf12ebbc0a901911c79fdbc37b6b08dab9284",
}

BOARD_KEYS = {
    "schema",
    "seed",
    "per_family",
    "case_count",
    "family_order",
    "cases_sha256",
    "rows",
}
BOARD_ROW_KEYS = {
    "id",
    "family",
    "question",
    "initial_state",
    "schedule",
    "answer",
    "stratum",
}
CALL_KEYS = {
    "prompt",
    "max_new",
    "response",
    "prompt_token_count",
    "untruncated_prompt_token_count",
    "prompt_truncated",
    "sampled_token_count",
    "decoded_token_count",
    "stop_reason",
}
SCORED_CALL_KEYS = CALL_KEYS | {
    "answer_segment",
    "predicted_answer",
    "correct",
}
ATOMIC_CALL_KEYS = CALL_KEYS | {
    "index",
    "operation",
    "operand",
    "input_state",
    "expected_state",
    "predicted_state",
    "correct",
}
SCHEDULED_CALL_KEYS = CALL_KEYS | {
    "index",
    "operation",
    "operand",
    "input_state",
    "predicted_state",
}
RESULT_ROW_KEYS = {
    "id",
    "family",
    "stratum",
    "question",
    "answer",
    "direct",
    "whole_problem_work",
    "atomic_oracle_state",
    "source_scheduled",
}
RESULT_KEYS = {
    "schema",
    "board",
    "board_sha256",
    "cases_sha256",
    "checkpoint_step",
    "checkpoint_sha256",
    "tokenizer_sha256",
    "implementation_sha256",
    "device",
    "max_new_full",
    "max_new_atomic",
    "resource_ledger",
    "summary",
    "gates",
    "integrity_gates",
    "advance_to_internalization",
    "rows",
}
SUMMARY_KEYS = {
    "case_count",
    "transition_count",
    "direct_correct",
    "whole_correct",
    "scheduled_correct",
    "atomic_correct",
    "atomic_total",
    "scheduler_only_correct",
    "direct_only_correct",
    "mcnemar_exact_p",
    "by_family",
}
FAMILY_SUMMARY_KEYS = {
    "count",
    "direct_correct",
    "whole_correct",
    "scheduled_correct",
    "atomic_correct",
    "atomic_total",
}
GATE_KEYS = {
    "scheduled_absolute",
    "scheduled_advantage",
    "paired_significance",
    "family_nonregression",
    "sequential_absolute",
    "atomic_ceiling",
}
INTEGRITY_KEYS = {
    "frozen_board_artifact",
    "independent_board_structure",
    "input_hashes_stable",
    "implementation_hashes_stable",
    "decode_caps_frozen",
    "immutable_output",
    "transcripts_complete",
    "renderers_exact",
    "model_call_accounting_exact",
    "scheduled_carry_is_model_only",
    "parse_failure_termination_exact",
    "no_unregistered_early_stop",
}

HEADER = re.compile(r"(?:^|\n)\s*(?:Question|Problem)(?:\s+\d+)?\s*:", re.IGNORECASE)
INTEGER = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)" r"(?![A-Za-z0-9_,]|\.\d)"
)


class AssessmentError(ValueError):
    """The evidence failed a frozen assessor requirement."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssessmentError(message)


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_bool(value: object) -> bool:
    return type(value) is bool


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _reject_duplicate_keys(pairs: Iterable[tuple[str, object]]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise AssessmentError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise AssessmentError(f"non-finite JSON number: {value}")


def strict_json_loads(payload: bytes, label: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssessmentError(f"{label} is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AssessmentError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssessmentError(f"invalid {label} JSON") from exc


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_paths() -> dict[str, Path]:
    return {
        "contract": ROOT / "R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION.md",
        "generator": ROOT / "train/generate_scheduled_reasoning_confirmation.py",
        "evaluator": ROOT / "train/eval_scheduled_reasoning_confirmation.py",
        "job": ROOT / "train/jobs/eval_scheduled_reasoning_confirmation.sbatch",
        "model_loader": ROOT / "train/model.py",
    }


def hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _parse_question(family: str, question: str):
    if family == "multiply_subtract":
        match = re.fullmatch(
            r"Compute (\d+) times (\d+), then subtract (\d+)\.", question
        )
        _require(match is not None, "unparsed multiply_subtract question")
        start, multiplier, subtractor = map(int, match.groups())
        return (
            start,
            [("multiply", multiplier), ("subtract", subtractor)],
            {"start": start, "multiplier": multiplier, "subtractor": subtractor},
        )
    if family == "base_conversion":
        match = re.fullmatch(
            r"Convert the base-(\d+) numeral ([0-9]{3}) to base 10\.", question
        )
        _require(match is not None, "unparsed base_conversion question")
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
        match = re.fullmatch(
            r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\.",
            question,
        )
        _require(match is not None, "unparsed sequential_state question")
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
        match = re.fullmatch(
            r"Add (\d+) and (\d+), then give the remainder after division by (\d+)\.",
            question,
        )
        _require(match is not None, "unparsed modular_update question")
        start, addend, modulus = map(int, match.groups())
        return (
            start,
            [("add", addend), ("remainder", modulus)],
            {"start": start, "addend": addend, "modulus": modulus},
        )
    raise AssessmentError(f"unknown family: {family}")


def apply_operation(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "remainder":
        _require(operand > 0, "remainder operand must be positive")
        return value % operand
    raise AssessmentError(f"unknown operation: {operation}")


def _validate_ranges(family: str, family_index: int, details: dict, answer: int) -> str:
    first_half = family_index < 32
    if family == "multiply_subtract":
        lower, upper = (2, 9) if first_half else (10, 19)
        valid = (
            20 <= details["start"] <= 99
            and lower <= details["multiplier"] <= upper
            and details["subtractor"] > 0
            and answer > 0
        )
        stratum = "small_multiplier" if first_half else "two_digit_multiplier"
    elif family == "base_conversion":
        lower, upper = (2, 9) if first_half else (10, 12)
        digits = details["digits"]
        valid = (
            lower <= details["base"] <= upper
            and digits[0] > 0
            and all(0 <= digit <= 9 and digit < details["base"] for digit in digits)
        )
        stratum = "base_2_9" if first_half else "base_10_12"
    elif family == "sequential_state":
        lower, upper = (2, 5) if first_half else (6, 7)
        valid = (
            5 <= details["start"] <= 50
            and 1 <= details["addend"] <= 25
            and lower <= details["multiplier"] <= upper
            and details["subtractor"] > 0
            and answer > 0
        )
        stratum = "multiplier_2_5" if first_half else "multiplier_6_7"
    elif family == "modular_update":
        lower, upper = (3, 14) if first_half else (15, 25)
        valid = (
            10 <= details["start"] <= 99
            and 10 <= details["addend"] <= 99
            and lower <= details["modulus"] <= upper
        )
        stratum = "modulus_3_14" if first_half else "modulus_15_25"
    else:
        raise AssessmentError(f"unknown family: {family}")
    _require(valid, f"row violates frozen {family} ranges")
    return stratum


def audit_board(board: object) -> tuple[list[dict], int]:
    _require(isinstance(board, dict), "board must be an object")
    _require(set(board) == BOARD_KEYS, "wrong board schema fields")
    _require(board["schema"] == BOARD_SCHEMA, "wrong board schema")
    _require(board["seed"] == SEED, "wrong board seed")
    _require(board["per_family"] == PER_FAMILY, "wrong per-family count")
    _require(board["case_count"] == EXPECTED_CASE_COUNT, "wrong case count")
    _require(board["family_order"] == list(FAMILIES), "wrong family order")
    _require(board["cases_sha256"] == EXPECTED_CASES_SHA256, "wrong cases hash")
    rows = board["rows"]
    _require(isinstance(rows, list), "board rows must be a list")
    _require(len(rows) == EXPECTED_CASE_COUNT, "wrong board row count")
    _require(
        sha256_bytes(canonical_bytes(rows)) == EXPECTED_CASES_SHA256,
        "board rows do not match frozen cases hash",
    )

    identifiers = set()
    questions = set()
    counts = Counter()
    total_steps = 0
    for position, row in enumerate(rows):
        _require(isinstance(row, dict), "board row must be an object")
        _require(set(row) == BOARD_ROW_KEYS, "wrong board row fields")
        family = FAMILIES[position // PER_FAMILY]
        family_index = position % PER_FAMILY
        expected_id = f"{family}_{family_index:03d}"
        _require(row["family"] == family, "board family order mismatch")
        _require(row["id"] == expected_id, "board id mismatch")
        _require(expected_id not in identifiers, "duplicate board id")
        identifiers.add(expected_id)
        _require(isinstance(row["question"], str), "question must be text")
        _require(row["question"] not in questions, "duplicate question")
        questions.add(row["question"])
        _require(_is_int(row["initial_state"]), "initial state must be an integer")
        _require(_is_int(row["answer"]), "answer must be an integer")

        start, schedule, details = _parse_question(family, row["question"])
        expected_stratum = _validate_ranges(
            family, family_index, details, row["answer"]
        )
        _require(row["stratum"] == expected_stratum, "board stratum mismatch")
        _require(isinstance(row["schedule"], list), "schedule must be a list")
        normalized = []
        for step in row["schedule"]:
            _require(
                isinstance(step, list)
                and len(step) == 2
                and isinstance(step[0], str)
                and _is_int(step[1]),
                "invalid schedule step",
            )
            normalized.append((step[0], step[1]))
        _require(start == row["initial_state"], "question/initial state mismatch")
        _require(normalized == schedule, "question/schedule mismatch")
        state = start
        for operation, operand in schedule:
            state = apply_operation(state, operation, operand)
            total_steps += 1
        _require(state == row["answer"], "schedule replay/answer mismatch")
        counts[family] += 1

    _require(
        counts == Counter({family: PER_FAMILY for family in FAMILIES}),
        "family balance mismatch",
    )
    _require(total_steps == EXPECTED_TRANSITION_COUNT, "transition count mismatch")
    return rows, total_steps


def answer_segment(response: str) -> str:
    return HEADER.split(response, maxsplit=1)[0].strip()


def last_integer(text: str) -> int | None:
    values = INTEGER.findall(text)
    return int(values[-1].replace(",", "")) if values else None


def parse_full_response(response: str) -> tuple[str, int | None]:
    segment = answer_segment(response)
    return segment, last_integer(segment)


def parse_first_line(response: str) -> int | None:
    line = next((line.strip() for line in response.splitlines() if line.strip()), "")
    return last_integer(line)


def operation_clause(value: int, operation: str, operand: int) -> str:
    if operation == "add":
        return f"Compute {value} plus {operand}."
    if operation == "subtract":
        return f"Compute {value} minus {operand}."
    if operation == "multiply":
        return f"Compute {value} times {operand}."
    if operation == "remainder":
        return f"Give the remainder after dividing {value} by {operand}."
    raise AssessmentError(f"unknown operation: {operation}")


def format_atomic_prompt(value: int, operation: str, operand: int) -> str:
    return f"Problem: {operation_clause(value, operation, operand)}\nWork:"


def direct_prompt(question: str) -> str:
    return f"Question: {question} Return only the final integer.\nAnswer:"


def whole_prompt(question: str) -> str:
    return f"Problem: {question}\nWork:"


def _audit_call(
    record: object,
    expected_keys: set[str],
    expected_prompt: str,
    expected_max_new: int,
    prompt_token_counter: Callable[[str], int],
) -> None:
    _require(isinstance(record, dict), "call record must be an object")
    _require(set(record) == expected_keys, "wrong call record fields")
    _require(record["prompt"] == expected_prompt, "wrong prompt renderer")
    _require(record["max_new"] == expected_max_new, "wrong decode cap")
    _require(isinstance(record["response"], str), "missing response transcript")
    for field in (
        "prompt_token_count",
        "untruncated_prompt_token_count",
        "sampled_token_count",
        "decoded_token_count",
    ):
        _require(_is_int(record[field]) and record[field] >= 0, f"invalid {field}")
    _require(_is_bool(record["prompt_truncated"]), "invalid truncation flag")
    expected_prompt_tokens = prompt_token_counter(expected_prompt)
    _require(
        _is_int(expected_prompt_tokens) and expected_prompt_tokens > 0,
        "tokenizer returned an invalid prompt count",
    )
    _require(record["prompt_truncated"] is False, "confirmation prompt was truncated")
    _require(
        record["prompt_token_count"] == expected_prompt_tokens,
        "prompt token count does not match frozen tokenizer",
    )
    _require(
        record["untruncated_prompt_token_count"] == expected_prompt_tokens,
        "untruncated prompt token count mismatch",
    )
    stop_reason = record["stop_reason"]
    _require(
        stop_reason in {"eos", "max_new", "context_limit"},
        "unregistered decode stop",
    )
    expected_sampled = record["decoded_token_count"] + (stop_reason == "eos")
    _require(
        record["sampled_token_count"] == expected_sampled,
        "sampled/decoded token accounting mismatch",
    )
    _require(
        record["sampled_token_count"] <= expected_max_new,
        "call exceeded frozen token cap",
    )
    if stop_reason == "max_new":
        _require(
            record["sampled_token_count"] == expected_max_new,
            "max_new stop occurred before frozen cap",
        )


def exact_mcnemar(scheduler_only: int, direct_only: int) -> Fraction:
    _require(
        _is_int(scheduler_only) and scheduler_only >= 0, "invalid discordant count"
    )
    _require(_is_int(direct_only) and direct_only >= 0, "invalid discordant count")
    discordant = scheduler_only + direct_only
    if discordant == 0:
        return Fraction(1, 1)
    smaller = min(scheduler_only, direct_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(Fraction(1, 1), Fraction(2 * tail, 2**discordant))


def _validate_reported_summary(summary: object) -> None:
    _require(
        isinstance(summary, dict) and set(summary) == SUMMARY_KEYS,
        "wrong summary schema",
    )
    for key in SUMMARY_KEYS - {"mcnemar_exact_p", "by_family"}:
        _require(
            _is_int(summary[key]) and summary[key] >= 0, f"invalid summary field {key}"
        )
    _require(
        type(summary["mcnemar_exact_p"]) is float, "McNemar report must be a float"
    )
    by_family = summary["by_family"]
    _require(
        isinstance(by_family, dict) and set(by_family) == set(FAMILIES),
        "wrong family summary schema",
    )
    for family in FAMILIES:
        item = by_family[family]
        _require(
            isinstance(item, dict) and set(item) == FAMILY_SUMMARY_KEYS,
            "wrong family summary fields",
        )
        for key in FAMILY_SUMMARY_KEYS:
            _require(
                _is_int(item[key]) and item[key] >= 0, "invalid family summary count"
            )


def _resource_item(records: list[dict]) -> dict:
    return {
        "model_calls": len(records),
        "prompt_token_count": sum(record["prompt_token_count"] for record in records),
        "sampled_token_count": sum(record["sampled_token_count"] for record in records),
        "decoded_token_count": sum(record["decoded_token_count"] for record in records),
    }


def assess_payload(
    result: object,
    board: object,
    prompt_token_counter: Callable[[str], int],
) -> dict:
    """Recompute all evidence from a decoded result and frozen board."""

    board_rows, transition_count = audit_board(board)
    _require(isinstance(result, dict), "result must be an object")
    _require(set(result) == RESULT_KEYS, "wrong result schema fields")
    _require(result["schema"] == RESULT_SCHEMA, "wrong result schema")
    _require(
        isinstance(result["board"], str) and result["board"],
        "missing board path record",
    )
    _require(
        result["board_sha256"] == EXPECTED_BOARD_SHA256, "wrong recorded board hash"
    )
    _require(
        result["cases_sha256"] == EXPECTED_CASES_SHA256, "wrong recorded cases hash"
    )
    _require(
        result["checkpoint_step"] == EXPECTED_CHECKPOINT_STEP, "wrong checkpoint step"
    )
    _require(
        result["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "wrong checkpoint hash",
    )
    _require(
        result["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA256, "wrong tokenizer hash"
    )
    _require(
        _exact_json_equal(
            result["implementation_sha256"], EXPECTED_IMPLEMENTATION_SHA256
        ),
        "wrong admitted implementation hashes",
    )
    _require(result["device"] == "cuda", "confirmation did not run on CUDA")
    _require(result["max_new_full"] == MAX_NEW_FULL, "wrong full decode cap")
    _require(result["max_new_atomic"] == MAX_NEW_ATOMIC, "wrong atomic decode cap")
    rows = result["rows"]
    _require(isinstance(rows, list), "result rows must be a list")
    _require(len(rows) == EXPECTED_CASE_COUNT, "missing or extra result rows")

    arm_records = {
        "direct_qa": [],
        "whole_problem_work": [],
        "atomic_oracle_state": [],
        "source_scheduled": [],
    }
    computed = []
    scheduled_parse_failures = 0
    scheduled_unissued_calls = 0

    for result_row, board_row in zip(rows, board_rows):
        _require(isinstance(result_row, dict), "result row must be an object")
        _require(set(result_row) == RESULT_ROW_KEYS, "wrong result row fields")
        for key in ("id", "family", "stratum", "question", "answer"):
            _require(result_row[key] == board_row[key], "result row/board mismatch")
        start = board_row["initial_state"]
        schedule = [(step[0], step[1]) for step in board_row["schedule"]]

        direct = result_row["direct"]
        _audit_call(
            direct,
            SCORED_CALL_KEYS,
            direct_prompt(board_row["question"]),
            MAX_NEW_FULL,
            prompt_token_counter,
        )
        direct_segment, direct_prediction = parse_full_response(direct["response"])
        direct_correct = direct_prediction == board_row["answer"]
        _require(
            direct["answer_segment"] == direct_segment, "tampered direct answer segment"
        )
        _require(
            direct["predicted_answer"] == direct_prediction,
            "tampered direct prediction",
        )
        _require(
            _is_bool(direct["correct"]) and direct["correct"] == direct_correct,
            "tampered direct correctness",
        )
        arm_records["direct_qa"].append(direct)

        whole = result_row["whole_problem_work"]
        _audit_call(
            whole,
            SCORED_CALL_KEYS,
            whole_prompt(board_row["question"]),
            MAX_NEW_FULL,
            prompt_token_counter,
        )
        whole_segment, whole_prediction = parse_full_response(whole["response"])
        whole_correct = whole_prediction == board_row["answer"]
        _require(
            whole["answer_segment"] == whole_segment,
            "tampered whole-work answer segment",
        )
        _require(
            whole["predicted_answer"] == whole_prediction,
            "tampered whole-work prediction",
        )
        _require(
            _is_bool(whole["correct"]) and whole["correct"] == whole_correct,
            "tampered whole-work correctness",
        )
        arm_records["whole_problem_work"].append(whole)

        atomic = result_row["atomic_oracle_state"]
        _require(
            isinstance(atomic, list) and len(atomic) == len(schedule),
            "atomic call count mismatch",
        )
        true_state = start
        atomic_correct = 0
        for index, ((operation, operand), record) in enumerate(zip(schedule, atomic)):
            expected_state = apply_operation(true_state, operation, operand)
            _audit_call(
                record,
                ATOMIC_CALL_KEYS,
                format_atomic_prompt(true_state, operation, operand),
                MAX_NEW_ATOMIC,
                prompt_token_counter,
            )
            predicted_state = parse_first_line(record["response"])
            correct = predicted_state == expected_state
            expected_metadata = {
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": true_state,
                "expected_state": expected_state,
                "predicted_state": predicted_state,
                "correct": correct,
            }
            for key, value in expected_metadata.items():
                _require(
                    type(record[key]) is type(value) and record[key] == value,
                    "tampered atomic transcript metadata",
                )
            atomic_correct += correct
            arm_records["atomic_oracle_state"].append(record)
            true_state = expected_state

        scheduled = result_row["source_scheduled"]
        _require(
            isinstance(scheduled, dict)
            and set(scheduled) == {"predicted_answer", "correct", "steps"},
            "wrong scheduled result fields",
        )
        steps = scheduled["steps"]
        _require(
            isinstance(steps, list) and 1 <= len(steps) <= len(schedule),
            "scheduled call count mismatch",
        )
        model_state = start
        parse_failed = False
        for index, record in enumerate(steps):
            operation, operand = schedule[index]
            _audit_call(
                record,
                SCHEDULED_CALL_KEYS,
                format_atomic_prompt(model_state, operation, operand),
                MAX_NEW_ATOMIC,
                prompt_token_counter,
            )
            predicted_state = parse_first_line(record["response"])
            expected_metadata = {
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": model_state,
                "predicted_state": predicted_state,
            }
            for key, value in expected_metadata.items():
                _require(
                    type(record[key]) is type(value) and record[key] == value,
                    "tampered scheduled carry metadata",
                )
            arm_records["source_scheduled"].append(record)
            if predicted_state is None:
                _require(index == len(steps) - 1, "chain continued after parse failure")
                parse_failed = True
                break
            model_state = predicted_state

        if parse_failed:
            scheduled_parse_failures += 1
            scheduled_unissued_calls += len(schedule) - len(steps)
            scheduled_prediction = None
        else:
            _require(
                len(steps) == len(schedule),
                "scheduled chain ended without parse failure",
            )
            scheduled_prediction = model_state
        scheduled_correct = scheduled_prediction == board_row["answer"]
        _require(
            type(scheduled["predicted_answer"]) is type(scheduled_prediction)
            and scheduled["predicted_answer"] == scheduled_prediction,
            "tampered scheduled final prediction",
        )
        _require(
            _is_bool(scheduled["correct"])
            and scheduled["correct"] == scheduled_correct,
            "tampered scheduled final correctness",
        )
        computed.append(
            {
                "family": board_row["family"],
                "direct_correct": direct_correct,
                "whole_correct": whole_correct,
                "scheduled_correct": scheduled_correct,
                "atomic_correct": atomic_correct,
                "atomic_total": len(schedule),
            }
        )

    by_family = {}
    for family in FAMILIES:
        selected = [row for row in computed if row["family"] == family]
        by_family[family] = {
            "count": len(selected),
            "direct_correct": sum(row["direct_correct"] for row in selected),
            "whole_correct": sum(row["whole_correct"] for row in selected),
            "scheduled_correct": sum(row["scheduled_correct"] for row in selected),
            "atomic_correct": sum(row["atomic_correct"] for row in selected),
            "atomic_total": sum(row["atomic_total"] for row in selected),
        }
    scheduler_only = sum(
        row["scheduled_correct"] and not row["direct_correct"] for row in computed
    )
    direct_only = sum(
        row["direct_correct"] and not row["scheduled_correct"] for row in computed
    )
    mcnemar = exact_mcnemar(scheduler_only, direct_only)
    summary = {
        "case_count": len(computed),
        "transition_count": transition_count,
        "direct_correct": sum(row["direct_correct"] for row in computed),
        "whole_correct": sum(row["whole_correct"] for row in computed),
        "scheduled_correct": sum(row["scheduled_correct"] for row in computed),
        "atomic_correct": sum(row["atomic_correct"] for row in computed),
        "atomic_total": transition_count,
        "scheduler_only_correct": scheduler_only,
        "direct_only_correct": direct_only,
        "mcnemar_exact_p": float(mcnemar),
        "by_family": by_family,
    }
    gates = {
        "scheduled_absolute": summary["scheduled_correct"] * 100
        >= 35 * EXPECTED_CASE_COUNT,
        "scheduled_advantage": (
            (summary["scheduled_correct"] - summary["direct_correct"]) * 100
            >= 10 * EXPECTED_CASE_COUNT
        ),
        "paired_significance": mcnemar < Fraction(1, 100),
        "family_nonregression": all(
            item["scheduled_correct"] >= item["direct_correct"]
            for item in by_family.values()
        ),
        "sequential_absolute": (
            by_family["sequential_state"]["scheduled_correct"] * 100
            >= 70 * by_family["sequential_state"]["count"]
        ),
        "atomic_ceiling": summary["atomic_correct"] * 100 >= 70 * transition_count,
    }
    advance = all(gates.values())

    expected_counts = {
        "direct_qa": EXPECTED_CASE_COUNT,
        "whole_problem_work": EXPECTED_CASE_COUNT,
        "atomic_oracle_state": transition_count,
        "source_scheduled": transition_count - scheduled_unissued_calls,
    }
    actual_counts = {arm: len(records) for arm, records in arm_records.items()}
    _require(actual_counts == expected_counts, "model-call accounting mismatch")
    by_arm = {arm: _resource_item(records) for arm, records in arm_records.items()}
    resource_ledger = {
        "model_calls": sum(item["model_calls"] for item in by_arm.values()),
        "maximum_model_calls_without_parse_failures": 2 * EXPECTED_CASE_COUNT
        + 2 * transition_count,
        "prompt_token_count": sum(
            item["prompt_token_count"] for item in by_arm.values()
        ),
        "sampled_token_count": sum(
            item["sampled_token_count"] for item in by_arm.values()
        ),
        "decoded_token_count": sum(
            item["decoded_token_count"] for item in by_arm.values()
        ),
        "by_arm": by_arm,
        "scheduled_parse_failure_chains": scheduled_parse_failures,
        "scheduled_calls_not_issued_after_parse_failure": scheduled_unissued_calls,
        "early_stop_policy": "eos_or_frozen_token_cap_or_context_limit_only",
        "external_schedule_parser": True,
        "external_horner_schedule": True,
        "gold_intermediates_in_scheduled_arm": 0,
        "retries": 0,
        "repair_calls": 0,
        "search_calls": 0,
        "verifier_feedback_calls": 0,
    }
    integrity_gates = {key: True for key in INTEGRITY_KEYS}

    _validate_reported_summary(result["summary"])
    _require(
        _exact_json_equal(result["summary"], summary),
        "reported summary differs from raw records",
    )
    _require(
        isinstance(result["gates"], dict)
        and set(result["gates"]) == GATE_KEYS
        and all(_is_bool(value) for value in result["gates"].values()),
        "wrong reported gate schema",
    )
    _require(
        _exact_json_equal(result["gates"], gates),
        "reported gates differ from recomputation",
    )
    _require(
        isinstance(result["integrity_gates"], dict)
        and set(result["integrity_gates"]) == INTEGRITY_KEYS
        and all(_is_bool(value) for value in result["integrity_gates"].values()),
        "wrong integrity gate schema",
    )
    _require(
        _exact_json_equal(result["integrity_gates"], integrity_gates),
        "reported integrity gates differ from independent checks",
    )
    _require(
        _exact_json_equal(result["resource_ledger"], resource_ledger),
        "reported resource ledger differs from raw call records",
    )
    _require(
        _is_bool(result["advance_to_internalization"])
        and result["advance_to_internalization"] == advance,
        "reported advance boolean differs from recomputation",
    )

    exact_summary = dict(summary)
    exact_summary.pop("mcnemar_exact_p")
    exact_summary["mcnemar_exact_two_sided"] = {
        "numerator": mcnemar.numerator,
        "denominator": mcnemar.denominator,
    }
    exact_summary["mcnemar_decimal"] = format(float(mcnemar), ".17g")
    return {
        "summary": exact_summary,
        "gates": gates,
        "advance_to_internalization": advance,
        "resource_ledger": resource_ledger,
        "independent_integrity": {
            "board_reparsed_and_replayed": True,
            "raw_responses_reparsed": True,
            "reported_scores_ignored_then_compared": True,
            "prompts_reconstructed_exactly": True,
            "prompt_tokens_recounted_with_frozen_tokenizer": True,
            "model_calls_recounted": True,
            "scheduled_carry_replayed": True,
            "mcnemar_recomputed_as_exact_rational": True,
        },
    }


def write_immutable_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
            os.fchmod(sink.fileno(), 0o444)
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _require(path.stat().st_mode & 0o222 == 0, "assessment output is writable")
    return sha256_bytes(payload)


def _require_regular(path: Path, label: str) -> None:
    _require(path.exists(), f"missing {label}")
    _require(path.is_file(), f"{label} is not a regular file")
    _require(not path.is_symlink(), f"{label} must not be a symlink")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--board",
        type=Path,
        default=ROOT
        / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "train/flagship_out/ckpt_0260000.pt"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "artifacts/shohin-tok-32k.json"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    _require(not args.out.exists(), "refusing to overwrite assessment output")
    for path, label in (
        (args.result, "source result"),
        (args.board, "frozen board"),
        (args.checkpoint, "frozen checkpoint"),
        (args.tokenizer, "frozen tokenizer"),
    ):
        _require_regular(path, label)
    for name, path in implementation_paths().items():
        _require_regular(path, f"implementation {name}")
    _require(args.result.stat().st_mode & 0o222 == 0, "source result is writable")
    _require(args.board.stat().st_mode & 0o222 == 0, "frozen board is writable")

    result_payload = args.result.read_bytes()
    board_payload = args.board.read_bytes()
    start_hashes = {
        "source_result": sha256_bytes(result_payload),
        "board": sha256_bytes(board_payload),
        "checkpoint": sha256_file(args.checkpoint),
        "tokenizer": sha256_file(args.tokenizer),
    }
    _require(
        start_hashes["board"] == EXPECTED_BOARD_SHA256, "board artifact hash mismatch"
    )
    _require(
        start_hashes["checkpoint"] == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint artifact hash mismatch",
    )
    _require(
        start_hashes["tokenizer"] == EXPECTED_TOKENIZER_SHA256,
        "tokenizer artifact hash mismatch",
    )
    start_implementation = hash_paths(implementation_paths())
    _require(
        start_implementation == EXPECTED_IMPLEMENTATION_SHA256,
        "current implementation does not match admitted hashes",
    )

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    def token_counter(prompt: str) -> int:
        return len(tokenizer.encode(prompt).ids)

    result = strict_json_loads(result_payload, "source result")
    board = strict_json_loads(board_payload, "board")
    assessment = assess_payload(result, board, token_counter)

    end_hashes = {
        "source_result": sha256_file(args.result),
        "board": sha256_file(args.board),
        "checkpoint": sha256_file(args.checkpoint),
        "tokenizer": sha256_file(args.tokenizer),
    }
    _require(end_hashes == start_hashes, "source evidence changed during assessment")
    end_implementation = hash_paths(implementation_paths())
    _require(
        end_implementation == start_implementation,
        "admitted implementation changed during assessment",
    )

    output = {
        "schema": ASSESSMENT_SCHEMA,
        "source_result": str(args.result),
        "source_result_sha256": start_hashes["source_result"],
        "frozen_input_sha256": {
            "board": EXPECTED_BOARD_SHA256,
            "cases": EXPECTED_CASES_SHA256,
            "checkpoint": EXPECTED_CHECKPOINT_SHA256,
            "tokenizer": EXPECTED_TOKENIZER_SHA256,
        },
        "admitted_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "custody": {
            "source_result_read_only": True,
            "source_result_stable_during_assessment": True,
            "frozen_inputs_stable_during_assessment": True,
            "implementation_stable_during_assessment": True,
            "exclusive_immutable_assessment_output": True,
        },
        **assessment,
    }
    output_sha256 = write_immutable_json(args.out, output)
    print(
        json.dumps(
            {
                "assessment": str(args.out),
                "assessment_sha256": output_sha256,
                "summary": assessment["summary"],
                "gates": assessment["gates"],
                "advance_to_internalization": assessment["advance_to_internalization"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
