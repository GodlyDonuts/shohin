#!/usr/bin/env python3
"""Frozen raw-260k operation-cursor diagnostic with exact transcript auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer


SOURCE_SCHEMA = "source_scheduled_reasoning_confirmation_v1"
RESULT_SCHEMA = "raw260k_operation_cursor_diagnostic_v1"
SOURCE_SEED = 2026071502
SOURCE_PER_FAMILY = 64
DIAGNOSTIC_PER_FAMILY = 16
SOURCE_CASE_COUNT = 256
SOURCE_TRANSITION_COUNT = 704
DIAGNOSTIC_CASE_COUNT = 64
DIAGNOSTIC_TRANSITION_COUNT = 176
EXPECTED_MODEL_CALLS = 528
EXPECTED_CHECKPOINT_STEP = 260000
MAX_NEW = 32

FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)
OPERATIONS = ("add", "subtract", "multiply", "remainder")

SOURCE_STEP_SELECTOR = "source_step_selector"
RESIDUAL_SUFFIX_SELECTOR = "residual_suffix_selector"
RESIDUAL_SUFFIX_STATE_UPDATE = "residual_suffix_state_update"
ARMS = (
    SOURCE_STEP_SELECTOR,
    RESIDUAL_SUFFIX_SELECTOR,
    RESIDUAL_SUFFIX_STATE_UPDATE,
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
SCORE_KEYS = {
    "parse_success",
    "parse_error",
    "parsed",
    "operation_correct",
    "operand_correct",
    "selection_correct",
}
SELECTOR_RECORD_KEYS = CALL_KEYS | SCORE_KEYS
STATE_RECORD_KEYS = SELECTOR_RECORD_KEYS | {"next_state_correct", "joint_correct"}
STEP_RESULT_KEYS = {"index", *ARMS}
ROW_RESULT_KEYS = {"id", "family", "steps"}
RESULT_KEYS = {
    "schema",
    "diagnostic_scope",
    "source_contract",
    "checkpoint_step",
    "input_sha256",
    "code_sha256",
    "device",
    "decode_contract",
    "prompt_exposure",
    "resource_ledger",
    "summary",
    "integrity",
    "rows",
}

_MULTIPLY = re.compile(r"Compute (\d+) times (\d+), then subtract (\d+)\.")
_BASE = re.compile(r"Convert the base-(\d+) numeral ([0-9]{3}) to base 10\.")
_SEQUENTIAL = re.compile(
    r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\."
)
_MODULAR = re.compile(
    r"Add (\d+) and (\d+), then give the remainder after division by (\d+)\."
)


class DuplicateJSONKey(ValueError):
    pass


class InvalidJSONConstant(ValueError):
    pass


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def digest_rows(rows):
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostic_source_paths():
    train_dir = Path(__file__).resolve().parent
    root = train_dir.parent
    return {
        "contract": root / "R12_OPERATION_CURSOR_DIAGNOSTIC.md",
        "evaluator": Path(__file__).resolve(),
        "tests": train_dir / "test_eval_operation_cursor.py",
        "job": train_dir / "jobs/eval_operation_cursor.sbatch",
        "model_loader": train_dir / "model.py",
    }


def hash_paths(paths):
    return {name: sha256_file(path) for name, path in paths.items()}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJSONKey(key)
        value[key] = item
    return value


def _reject_constant(value):
    raise InvalidJSONConstant(value)


def _strict_json_loads(text):
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def parse_structured_response(response, include_next_state=False):
    """Parse exactly one arm-specific JSON object, with no salvage behavior."""
    if not isinstance(response, str) or not response.strip():
        return None, "empty_response"
    try:
        value = _strict_json_loads(response)
    except DuplicateJSONKey:
        return None, "duplicate_key"
    except InvalidJSONConstant:
        return None, "invalid_json_constant"
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "invalid_json"

    if type(value) is not dict:
        return None, "not_object"
    expected_keys = {"operation", "operand"}
    if include_next_state:
        expected_keys.add("next_state")
    if set(value) != expected_keys:
        return None, "wrong_keys"
    if type(value["operation"]) is not str:
        return None, "operation_not_string"
    if value["operation"] not in OPERATIONS:
        return None, "operation_not_allowed"
    if type(value["operand"]) is not int:
        return None, "operand_not_integer"
    if include_next_state and type(value["next_state"]) is not int:
        return None, "next_state_not_integer"

    parsed = {
        "operation": value["operation"],
        "operand": value["operand"],
    }
    if include_next_state:
        parsed["next_state"] = value["next_state"]
    return parsed, None


def parse_question(family, question):
    """Reconstruct a schedule from source prose without trusting board schedule fields."""
    if family == "multiply_subtract":
        match = _MULTIPLY.fullmatch(question)
        if not match:
            raise ValueError(f"unparsed multiply_subtract question: {question}")
        start, multiplier, subtractor = map(int, match.groups())
        return (
            start,
            [("multiply", multiplier), ("subtract", subtractor)],
            {"start": start, "multiplier": multiplier, "subtractor": subtractor},
        )
    if family == "base_conversion":
        match = _BASE.fullmatch(question)
        if not match:
            raise ValueError(f"unparsed base_conversion question: {question}")
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
        if not match:
            raise ValueError(f"unparsed sequential_state question: {question}")
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
        if not match:
            raise ValueError(f"unparsed modular_update question: {question}")
        start, addend, modulus = map(int, match.groups())
        return (
            start,
            [("add", addend), ("remainder", modulus)],
            {"start": start, "addend": addend, "modulus": modulus},
        )
    raise ValueError(f"unknown family: {family}")


def apply_operation(value, operation, operand):
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "remainder":
        return value % operand
    raise ValueError(f"unknown operation: {operation}")


def _in_range(value, lower, upper):
    return lower <= value <= upper


def _expected_stratum(family, family_index, details, answer):
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
    if not valid:
        raise ValueError(f"row violates frozen {family} ranges")
    return stratum


def reconstruct_schedule(row):
    start, schedule, _ = parse_question(row.get("family"), row.get("question"))
    raw_schedule = row.get("schedule")
    if not isinstance(raw_schedule, list):
        raise ValueError("source schedule is not a list")
    normalized = []
    for step in raw_schedule:
        if (
            not isinstance(step, list)
            or len(step) != 2
            or type(step[0]) is not str
            or type(step[1]) is not int
        ):
            raise ValueError("invalid source schedule step")
        normalized.append((step[0], step[1]))
    if type(row.get("initial_state")) is not int or row["initial_state"] != start:
        raise ValueError("question and source initial state disagree")
    if normalized != schedule:
        raise ValueError("question and source schedule disagree")
    return start, schedule


def audit_source(source):
    if type(source) is not dict or set(source) != SOURCE_KEYS:
        raise ValueError("wrong source artifact schema")
    if source.get("schema") != SOURCE_SCHEMA or source.get("seed") != SOURCE_SEED:
        raise ValueError("wrong source schema or seed")
    if (
        source.get("per_family") != SOURCE_PER_FAMILY
        or source.get("case_count") != SOURCE_CASE_COUNT
        or tuple(source.get("family_order", ())) != FAMILIES
    ):
        raise ValueError("wrong source family or cardinality contract")
    if source.get("cases_sha256") != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("wrong frozen source rows hash metadata")
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != SOURCE_CASE_COUNT:
        raise ValueError("wrong source row count")
    if digest_rows(rows) != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("source rows do not match frozen hash")

    identifiers = set()
    questions = set()
    total_transitions = 0
    for position, row in enumerate(rows):
        if type(row) is not dict or set(row) != SOURCE_ROW_KEYS:
            raise ValueError("wrong source row schema")
        family = FAMILIES[position // SOURCE_PER_FAMILY]
        family_index = position % SOURCE_PER_FAMILY
        expected_id = f"{family}_{family_index:03d}"
        if row.get("family") != family or row.get("id") != expected_id:
            raise ValueError("source rows are outside frozen family/id order")
        if expected_id in identifiers:
            raise ValueError("duplicate source row id")
        identifiers.add(expected_id)
        question = row.get("question")
        if not isinstance(question, str) or question in questions:
            raise ValueError("invalid or duplicate source question")
        questions.add(question)
        if type(row.get("answer")) is not int:
            raise ValueError("source answer is not an integer")

        start, schedule, details = parse_question(family, question)
        if row.get("stratum") != _expected_stratum(
            family, family_index, details, row["answer"]
        ):
            raise ValueError("wrong frozen source stratum")
        verified_start, verified_schedule = reconstruct_schedule(row)
        if verified_start != start or verified_schedule != schedule:
            raise ValueError("independent source reconstruction mismatch")
        state = start
        for operation, operand in schedule:
            state = apply_operation(state, operation, operand)
            total_transitions += 1
        if state != row["answer"]:
            raise ValueError("source schedule replay does not match answer")

    if total_transitions != SOURCE_TRANSITION_COUNT:
        raise ValueError("wrong frozen source transition count")
    return rows, total_transitions


def load_frozen_source(path):
    path = Path(path)
    payload = path.read_bytes()
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    if artifact_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError("source artifact hash does not match frozen source")
    if path.stat().st_mode & 0o222:
        raise PermissionError("frozen source artifact must be read-only")
    source = _strict_json_loads(payload.decode("ascii"))
    rows, transitions = audit_source(source)
    return source, rows, transitions, artifact_sha256


def select_frozen_subset(rows):
    if not isinstance(rows, list) or len(rows) != SOURCE_CASE_COUNT:
        raise ValueError("subset selection requires all frozen source rows")
    selected = []
    for family_index, family in enumerate(FAMILIES):
        start = family_index * SOURCE_PER_FAMILY
        block = rows[start : start + SOURCE_PER_FAMILY]
        if any(row.get("family") != family for row in block):
            raise ValueError("source family block changed before subset selection")
        selected.extend(block[:DIAGNOSTIC_PER_FAMILY])
    if len(selected) != DIAGNOSTIC_CASE_COUNT:
        raise ValueError("wrong diagnostic subset row count")
    expected_ids = [
        f"{family}_{index:03d}"
        for family in FAMILIES
        for index in range(DIAGNOSTIC_PER_FAMILY)
    ]
    if [row.get("id") for row in selected] != expected_ids:
        raise ValueError("diagnostic subset is not first 16 per family")
    if digest_rows(selected) != EXPECTED_SUBSET_ROWS_SHA256:
        raise ValueError("diagnostic subset hash mismatch")
    transitions = sum(len(reconstruct_schedule(row)[1]) for row in selected)
    if transitions != DIAGNOSTIC_TRANSITION_COUNT:
        raise ValueError("wrong diagnostic transition count")
    return selected, transitions


def render_residual_suffix(schedule):
    return json.dumps(
        [[operation, operand] for operation, operand in schedule],
        separators=(",", ":"),
    )


def source_step_prompt(question, index):
    return (
        "Task: Select the operation and operand at the supplied cursor.\n"
        f"Source: {question}\n"
        f"Step index (zero-based): {index}\n"
        "Output schema: operation is one of add, subtract, multiply, remainder; "
        "operand is an integer.\n"
        "Return only one JSON object with exactly the keys operation and operand, "
        "and no other text.\n"
        "JSON:"
    )


def residual_suffix_prompt(residual_schedule):
    return (
        "Task: Select the first operation and operand from the supplied residual suffix.\n"
        f"Residual suffix (read-only JSON): {render_residual_suffix(residual_schedule)}\n"
        "Do not execute an operation. Do not return or update the residual suffix.\n"
        "Output schema: operation is one of add, subtract, multiply, remainder; "
        "operand is an integer.\n"
        "Return only one JSON object with exactly the keys operation and operand, "
        "and no other text.\n"
        "JSON:"
    )


def residual_state_prompt(current_state, residual_schedule):
    return (
        "Task: Select the first operation and operand from the supplied residual suffix, "
        "then apply it once to the supplied current state.\n"
        f"Current state (oracle-supplied for this arm): {current_state}\n"
        f"Residual suffix (read-only JSON): {render_residual_suffix(residual_schedule)}\n"
        "Do not return or update the residual suffix.\n"
        "Output schema: operation is one of add, subtract, multiply, remainder; "
        "operand and next_state are integers.\n"
        "Return only one JSON object with exactly the keys operation, operand, and "
        "next_state, and no other text.\n"
        "JSON:"
    )


def _decode_tokens(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _autocast(device):
    return torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
    )


@torch.no_grad()
def greedy_completion(model, tokenizer, prompt, device, max_new):
    """Greedy decode with only EOS, context, or the frozen token cap as stops."""
    if max_new != MAX_NEW:
        raise ValueError("operation-cursor decode cap is frozen")
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("prompt encoded to no tokens")
    if len(prompt_ids) >= cap:
        raise ValueError("operation-cursor prompt would reach or exceed context limit")
    with _autocast(device):
        logits, cache = model(
            torch.tensor([prompt_ids], device=device), return_cache=True, pos=0
        )

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    generated = []
    sampled_token_count = 0
    position = len(prompt_ids)
    stop_reason = "max_new"
    for _ in range(max_new):
        token = int(logits[:, -1].argmax(dim=-1).item())
        sampled_token_count += 1
        if eos_id is not None and token == eos_id:
            stop_reason = "eos"
            break
        generated.append(token)
        if position >= cap:
            stop_reason = "context_limit"
            break
        with _autocast(device):
            logits, cache = model(
                torch.tensor([[token]], device=device),
                cache=cache,
                pos=position,
                return_cache=True,
            )
        position += 1

    return {
        "response": _decode_tokens(tokenizer, generated),
        "prompt_token_count": len(prompt_ids),
        "untruncated_prompt_token_count": len(prompt_ids),
        "prompt_truncated": False,
        "sampled_token_count": sampled_token_count,
        "decoded_token_count": len(generated),
        "stop_reason": stop_reason,
    }


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def load_model(path, device):
    # Import only after main has captured the model-loader hash.
    from model import GPT, GPTConfig

    checkpoint = torch.load(path, map_location="cpu")
    if type(checkpoint) is not dict or type(checkpoint.get("cfg")) is not dict:
        raise ValueError("checkpoint does not contain the expected model payload")
    if (
        type(checkpoint.get("step")) is not int
        or checkpoint["step"] != EXPECTED_CHECKPOINT_STEP
    ):
        raise ValueError("checkpoint metadata is not exact raw step 260000")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def call_model(model, tokenizer, prompt, device, arm, observed_call_counts):
    if arm not in ARMS:
        raise ValueError("unregistered operation-cursor arm")
    record = {
        "prompt": prompt,
        "max_new": MAX_NEW,
        **greedy_completion(model, tokenizer, prompt, device, MAX_NEW),
    }
    observed_call_counts[arm] += 1
    return record


def score_call_record(
    record,
    expected_operation,
    expected_operand,
    expected_next_state=None,
    include_next_state=False,
):
    parsed, parse_error = parse_structured_response(
        record.get("response"), include_next_state=include_next_state
    )
    operation_correct = bool(
        parsed is not None and parsed["operation"] == expected_operation
    )
    operand_correct = bool(parsed is not None and parsed["operand"] == expected_operand)
    scored = {
        **record,
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "parsed": parsed,
        "operation_correct": operation_correct,
        "operand_correct": operand_correct,
        "selection_correct": operation_correct and operand_correct,
    }
    if include_next_state:
        next_state_correct = bool(
            parsed is not None and parsed["next_state"] == expected_next_state
        )
        scored["next_state_correct"] = next_state_correct
        scored["joint_correct"] = scored["selection_correct"] and next_state_correct
    return scored


def evaluate_rows(model, tokenizer, device, subset_rows, progress=False):
    results = []
    observed_call_counts = Counter()
    for offset, row in enumerate(subset_rows, 1):
        current_state, schedule = reconstruct_schedule(row)
        step_results = []
        for index, (operation, operand) in enumerate(schedule):
            residual = schedule[index:]
            next_state = apply_operation(current_state, operation, operand)

            source_record = call_model(
                model,
                tokenizer,
                source_step_prompt(row["question"], index),
                device,
                SOURCE_STEP_SELECTOR,
                observed_call_counts,
            )
            suffix_record = call_model(
                model,
                tokenizer,
                residual_suffix_prompt(residual),
                device,
                RESIDUAL_SUFFIX_SELECTOR,
                observed_call_counts,
            )
            state_record = call_model(
                model,
                tokenizer,
                residual_state_prompt(current_state, residual),
                device,
                RESIDUAL_SUFFIX_STATE_UPDATE,
                observed_call_counts,
            )
            step_results.append(
                {
                    "index": index,
                    SOURCE_STEP_SELECTOR: score_call_record(
                        source_record, operation, operand
                    ),
                    RESIDUAL_SUFFIX_SELECTOR: score_call_record(
                        suffix_record, operation, operand
                    ),
                    RESIDUAL_SUFFIX_STATE_UPDATE: score_call_record(
                        state_record,
                        operation,
                        operand,
                        expected_next_state=next_state,
                        include_next_state=True,
                    ),
                }
            )
            current_state = next_state
        results.append(
            {"id": row["id"], "family": row["family"], "steps": step_results}
        )
        if progress and offset % 8 == 0:
            print(f"[operation-cursor] {offset}/{len(subset_rows)} cases", flush=True)
    return results, observed_call_counts


def _audit_decode_record(record, expected_prompt):
    if not isinstance(record, dict) or record.get("prompt") != expected_prompt:
        raise ValueError("operation-cursor prompt transcript mismatch")
    if record.get("max_new") != MAX_NEW:
        raise ValueError("operation-cursor decode cap mismatch")
    if not isinstance(record.get("response"), str):
        raise ValueError("missing operation-cursor response transcript")
    integer_fields = (
        "prompt_token_count",
        "untruncated_prompt_token_count",
        "sampled_token_count",
        "decoded_token_count",
    )
    if any(
        type(record.get(field)) is not int or record[field] < 0
        for field in integer_fields
    ):
        raise ValueError("invalid operation-cursor token accounting")
    if record["prompt_token_count"] <= 0:
        raise ValueError("empty operation-cursor prompt tokenization")
    if record.get("prompt_truncated") is not False:
        raise ValueError("operation-cursor prompt was truncated")
    if record["prompt_token_count"] != record["untruncated_prompt_token_count"]:
        raise ValueError("operation-cursor prompt token counts disagree")
    sampled = record["sampled_token_count"]
    decoded = record["decoded_token_count"]
    stop = record.get("stop_reason")
    if stop not in {"eos", "max_new", "context_limit"}:
        raise ValueError("unregistered operation-cursor decode stop")
    if sampled > MAX_NEW or decoded > sampled:
        raise ValueError("operation-cursor call exceeded token cap")
    if stop == "eos" and sampled != decoded + 1:
        raise ValueError("EOS token accounting mismatch")
    if stop == "max_new" and (sampled != MAX_NEW or decoded != sampled):
        raise ValueError("max_new token accounting mismatch")
    if stop == "context_limit" and (sampled != decoded or sampled == 0):
        raise ValueError("context-limit token accounting mismatch")


def _audit_scored_record(
    record,
    expected_prompt,
    expected_operation,
    expected_operand,
    expected_next_state=None,
    include_next_state=False,
):
    expected_keys = STATE_RECORD_KEYS if include_next_state else SELECTOR_RECORD_KEYS
    if type(record) is not dict or set(record) != expected_keys:
        raise ValueError("wrong scored operation-cursor call schema")
    _audit_decode_record(record, expected_prompt)
    parsed, parse_error = parse_structured_response(
        record["response"], include_next_state=include_next_state
    )
    operation_correct = bool(
        parsed is not None and parsed["operation"] == expected_operation
    )
    operand_correct = bool(parsed is not None and parsed["operand"] == expected_operand)
    expected = {
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "parsed": parsed,
        "operation_correct": operation_correct,
        "operand_correct": operand_correct,
        "selection_correct": operation_correct and operand_correct,
    }
    if include_next_state:
        next_state_correct = bool(
            parsed is not None and parsed["next_state"] == expected_next_state
        )
        expected["next_state_correct"] = next_state_correct
        expected["joint_correct"] = expected["selection_correct"] and next_state_correct
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("preserved parse or score does not match response")


def audit_execution(results, subset_rows, observed_call_counts):
    if not isinstance(results, list) or len(results) != len(subset_rows):
        raise ValueError("wrong operation-cursor result row count")
    arm_records = {arm: [] for arm in ARMS}
    for result, row in zip(results, subset_rows):
        if type(result) is not dict or set(result) != ROW_RESULT_KEYS:
            raise ValueError("wrong operation-cursor result row schema")
        if result.get("id") != row["id"] or result.get("family") != row["family"]:
            raise ValueError("operation-cursor result row identity mismatch")
        current_state, schedule = reconstruct_schedule(row)
        steps = result.get("steps")
        if not isinstance(steps, list) or len(steps) != len(schedule):
            raise ValueError("operation-cursor step count mismatch")
        for index, ((operation, operand), step) in enumerate(zip(schedule, steps)):
            if type(step) is not dict or set(step) != STEP_RESULT_KEYS:
                raise ValueError("wrong operation-cursor step schema")
            if step.get("index") != index:
                raise ValueError("operation-cursor step index mismatch")
            residual = schedule[index:]
            next_state = apply_operation(current_state, operation, operand)
            _audit_scored_record(
                step[SOURCE_STEP_SELECTOR],
                source_step_prompt(row["question"], index),
                operation,
                operand,
            )
            _audit_scored_record(
                step[RESIDUAL_SUFFIX_SELECTOR],
                residual_suffix_prompt(residual),
                operation,
                operand,
            )
            _audit_scored_record(
                step[RESIDUAL_SUFFIX_STATE_UPDATE],
                residual_state_prompt(current_state, residual),
                operation,
                operand,
                expected_next_state=next_state,
                include_next_state=True,
            )
            for arm in ARMS:
                arm_records[arm].append(step[arm])
            current_state = next_state

    actual_counts = {arm: len(records) for arm, records in arm_records.items()}
    expected_counts = {arm: DIAGNOSTIC_TRANSITION_COUNT for arm in ARMS}
    if actual_counts != expected_counts:
        raise ValueError("operation-cursor model-call transcript count mismatch")
    if dict(observed_call_counts) != actual_counts:
        raise ValueError("observed model calls do not match preserved transcripts")

    by_arm = {}
    for arm, records in arm_records.items():
        by_arm[arm] = {
            "model_calls": len(records),
            "prompt_token_count": sum(
                record["prompt_token_count"] for record in records
            ),
            "sampled_token_count": sum(
                record["sampled_token_count"] for record in records
            ),
            "decoded_token_count": sum(
                record["decoded_token_count"] for record in records
            ),
            "parse_success": sum(record["parse_success"] for record in records),
            "parse_failure": sum(not record["parse_success"] for record in records),
        }
    resource_ledger = {
        "model_calls": sum(item["model_calls"] for item in by_arm.values()),
        "expected_model_calls": EXPECTED_MODEL_CALLS,
        "calls_per_transition": len(ARMS),
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
        "calls_not_issued_after_parse_failure": 0,
        "retries": 0,
        "repair_calls": 0,
        "search_calls": 0,
        "verifier_feedback_calls": 0,
        "early_stop_policy": "eos_or_frozen_token_cap_or_context_limit_only",
    }
    if resource_ledger["model_calls"] != EXPECTED_MODEL_CALLS:
        raise ValueError("operation-cursor total model-call accounting mismatch")
    integrity = {
        "transcripts_complete": True,
        "prompts_reconstructed_exactly": True,
        "schedules_reconstructed_from_source": True,
        "strict_parses_recomputed": True,
        "scores_recomputed": True,
        "model_call_accounting_exact": True,
        "token_accounting_exact": True,
        "no_conditional_call_suppression": True,
        "no_retries_or_repairs": True,
    }
    return resource_ledger, integrity


def _ratio(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator)}


def _flatten_entries(results, subset_rows):
    entries = []
    for result, row in zip(results, subset_rows):
        _, schedule = reconstruct_schedule(row)
        for (operation, _), step in zip(schedule, result["steps"]):
            entries.append(
                {
                    "family": row["family"],
                    "index": step["index"],
                    "operation": operation,
                    "step": step,
                }
            )
    return entries


def _arm_summary(entries, arm):
    records = [entry["step"][arm] for entry in entries]
    total = len(records)
    parse_success = sum(record["parse_success"] for record in records)
    errors = Counter(
        record["parse_error"] for record in records if not record["parse_success"]
    )
    summary = {
        "calls": total,
        "parse_success": _ratio(parse_success, total),
        "parse_failure": _ratio(total - parse_success, total),
        "parse_error_counts": dict(sorted(errors.items())),
        "operation_correct": _ratio(
            sum(record["operation_correct"] for record in records), total
        ),
        "operand_correct": _ratio(
            sum(record["operand_correct"] for record in records), total
        ),
        "selection_correct": _ratio(
            sum(record["selection_correct"] for record in records), total
        ),
    }
    if arm == RESIDUAL_SUFFIX_STATE_UPDATE:
        selection_correct = sum(record["selection_correct"] for record in records)
        joint_correct = sum(record["joint_correct"] for record in records)
        summary.update(
            {
                "next_state_correct": _ratio(
                    sum(record["next_state_correct"] for record in records), total
                ),
                "joint_correct": _ratio(joint_correct, total),
                "next_state_correct_given_selection_correct": _ratio(
                    joint_correct, selection_correct
                ),
            }
        )
    return summary


def _paired_cells(entries, left_arm, left_metric, right_arm, right_metric):
    pairs = [
        (
            bool(entry["step"][left_arm][left_metric]),
            bool(entry["step"][right_arm][right_metric]),
        )
        for entry in entries
    ]
    return {
        "left_metric": f"{left_arm}.{left_metric}",
        "right_metric": f"{right_arm}.{right_metric}",
        "both_correct": sum(left and right for left, right in pairs),
        "left_only_correct": sum(left and not right for left, right in pairs),
        "right_only_correct": sum(not left and right for left, right in pairs),
        "neither_correct": sum(not left and not right for left, right in pairs),
        "total": len(pairs),
    }


def _group_summary(entries):
    state_records = [entry["step"][RESIDUAL_SUFFIX_STATE_UPDATE] for entry in entries]
    state_selection = sum(record["selection_correct"] for record in state_records)
    state_joint = sum(record["joint_correct"] for record in state_records)
    return {
        "transition_count": len(entries),
        "model_calls": len(entries) * len(ARMS),
        "by_arm": {arm: _arm_summary(entries, arm) for arm in ARMS},
        "paired_comparisons": {
            "source_step_selection_vs_residual_suffix_selection": _paired_cells(
                entries,
                SOURCE_STEP_SELECTOR,
                "selection_correct",
                RESIDUAL_SUFFIX_SELECTOR,
                "selection_correct",
            ),
            "residual_suffix_selection_vs_state_update_selection": _paired_cells(
                entries,
                RESIDUAL_SUFFIX_SELECTOR,
                "selection_correct",
                RESIDUAL_SUFFIX_STATE_UPDATE,
                "selection_correct",
            ),
            "residual_suffix_selection_vs_joint_state_update": _paired_cells(
                entries,
                RESIDUAL_SUFFIX_SELECTOR,
                "selection_correct",
                RESIDUAL_SUFFIX_STATE_UPDATE,
                "joint_correct",
            ),
        },
        "state_update_conditionals": {
            "selection_correct": _ratio(state_selection, len(entries)),
            "next_state_correct": _ratio(
                sum(record["next_state_correct"] for record in state_records),
                len(entries),
            ),
            "joint_correct": _ratio(state_joint, len(entries)),
            "next_state_correct_given_selection_correct": _ratio(
                state_joint, state_selection
            ),
        },
    }


def build_summary(results, subset_rows):
    entries = _flatten_entries(results, subset_rows)
    global_summary = _group_summary(entries)
    by_family = {}
    for family in FAMILIES:
        selected = [entry for entry in entries if entry["family"] == family]
        by_family[family] = {
            "case_count": DIAGNOSTIC_PER_FAMILY,
            **_group_summary(selected),
        }
    by_step_index = {}
    for index in range(4):
        selected = [entry for entry in entries if entry["index"] == index]
        if selected:
            by_step_index[str(index)] = _group_summary(selected)
    by_operation = {}
    for operation in OPERATIONS:
        selected = [entry for entry in entries if entry["operation"] == operation]
        if selected:
            by_operation[operation] = _group_summary(selected)
    return {
        "case_count": len(subset_rows),
        **global_summary,
        "by_family": by_family,
        "by_step_index": by_step_index,
        "by_operation": by_operation,
    }


def source_contract(source, subset_rows, transitions):
    return {
        "schema": SOURCE_SCHEMA,
        "selection": "first_16_in_each_frozen_family_block",
        "family_order": list(FAMILIES),
        "per_family": DIAGNOSTIC_PER_FAMILY,
        "case_count": len(subset_rows),
        "transition_count": transitions,
        "source_rows_sha256": source["cases_sha256"],
        "subset_rows_sha256": digest_rows(subset_rows),
        "row_ids": [row["id"] for row in subset_rows],
    }


def diagnostic_scope():
    return {
        "diagnostic_only": True,
        "reasoning_claim": "none",
        "promotion_decision": "none",
        "production_submission": False,
        "training_action": False,
        "reported_values": "exact_structured_parse_and_correctness_counts_only",
    }


def decode_contract():
    return {
        "strategy": "greedy_argmax",
        "max_new": MAX_NEW,
        "prompt_truncation": "forbidden",
        "retries": 0,
        "repair_calls": 0,
        "response_parser": "whole_response_strict_json_exact_keys",
    }


def prompt_exposure_contract():
    return {
        SOURCE_STEP_SELECTOR: ["full_source_question", "zero_based_step_index"],
        RESIDUAL_SUFFIX_SELECTOR: ["oracle_residual_schedule_suffix"],
        RESIDUAL_SUFFIX_STATE_UPDATE: [
            "oracle_residual_schedule_suffix",
            "oracle_current_numeric_state",
        ],
        "excluded_from_all_prompts": [
            "gold_final_answer",
            "expected_next_state",
            "prior_model_response",
            "score",
            "verifier_feedback",
        ],
    }


def build_result(
    source,
    subset_rows,
    transitions,
    checkpoint_step,
    input_hashes,
    code_hashes,
    device,
    results,
    observed_call_counts,
):
    resource_ledger, execution_integrity = audit_execution(
        results, subset_rows, observed_call_counts
    )
    integrity = {
        "frozen_source_artifact_hash": True,
        "frozen_source_rows_hash": True,
        "frozen_subset_rows_hash": True,
        "input_hashes_stable": True,
        "code_hashes_stable": True,
        "immutable_output": True,
        "arm_exposure_contract_exact": True,
        **execution_integrity,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "diagnostic_scope": diagnostic_scope(),
        "source_contract": source_contract(source, subset_rows, transitions),
        "checkpoint_step": checkpoint_step,
        "input_sha256": input_hashes,
        "code_sha256": code_hashes,
        "device": device,
        "decode_contract": decode_contract(),
        "prompt_exposure": prompt_exposure_contract(),
        "resource_ledger": resource_ledger,
        "summary": build_summary(results, subset_rows),
        "integrity": integrity,
        "rows": results,
    }
    audit_preserved_result(
        result,
        source,
        subset_rows,
        transitions,
        input_hashes,
        code_hashes,
        device,
    )
    return result


def audit_preserved_result(
    result,
    source,
    subset_rows,
    transitions,
    expected_input_hashes,
    expected_code_hashes,
    expected_device,
):
    if type(result) is not dict or set(result) != RESULT_KEYS:
        raise ValueError("wrong preserved operation-cursor result schema")
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("wrong operation-cursor result schema id")
    fixed_fields = {
        "diagnostic_scope": diagnostic_scope(),
        "source_contract": source_contract(source, subset_rows, transitions),
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "input_sha256": expected_input_hashes,
        "code_sha256": expected_code_hashes,
        "device": expected_device,
        "decode_contract": decode_contract(),
        "prompt_exposure": prompt_exposure_contract(),
    }
    for key, expected in fixed_fields.items():
        if result.get(key) != expected:
            raise ValueError(f"preserved operation-cursor {key} mismatch")
    expected_counts = Counter({arm: transitions for arm in ARMS})
    ledger, execution_integrity = audit_execution(
        result.get("rows"), subset_rows, expected_counts
    )
    expected_integrity = {
        "frozen_source_artifact_hash": True,
        "frozen_source_rows_hash": True,
        "frozen_subset_rows_hash": True,
        "input_hashes_stable": True,
        "code_hashes_stable": True,
        "immutable_output": True,
        "arm_exposure_contract_exact": True,
        **execution_integrity,
    }
    if result.get("resource_ledger") != ledger:
        raise ValueError("preserved operation-cursor resource ledger mismatch")
    if result.get("summary") != build_summary(result["rows"], subset_rows):
        raise ValueError("preserved operation-cursor summary mismatch")
    if result.get("integrity") != expected_integrity:
        raise ValueError("preserved operation-cursor integrity mismatch")
    return True


def write_immutable_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError("refusing existing operation-cursor output")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    if path.stat().st_mode & 0o222:
        raise PermissionError("operation-cursor output remained writable")
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    args = parser.parse_args()

    if os.path.lexists(args.out):
        raise FileExistsError("refusing existing operation-cursor output")
    source, source_rows, source_transitions, source_sha256 = load_frozen_source(
        args.source
    )
    if source_transitions != SOURCE_TRANSITION_COUNT:
        raise ValueError("frozen source transition audit changed")
    subset_rows, transitions = select_frozen_subset(source_rows)
    input_hashes = {
        "checkpoint": sha256_file(args.ckpt),
        "tokenizer": sha256_file(args.tokenizer),
        "source": source_sha256,
    }
    expected_input_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
        "source": EXPECTED_SOURCE_SHA256,
    }
    if input_hashes != expected_input_hashes:
        raise ValueError("operation-cursor input hash mismatch")
    source_paths = diagnostic_source_paths()
    code_hashes = hash_paths(source_paths)

    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    results, observed_call_counts = evaluate_rows(
        model, tokenizer, device, subset_rows, progress=True
    )

    current_input_hashes = {
        "checkpoint": sha256_file(args.ckpt),
        "tokenizer": sha256_file(args.tokenizer),
        "source": sha256_file(args.source),
    }
    if current_input_hashes != input_hashes:
        raise RuntimeError("operation-cursor input changed during evaluation")
    if hash_paths(source_paths) != code_hashes:
        raise RuntimeError("operation-cursor implementation changed during evaluation")

    result = build_result(
        source,
        subset_rows,
        transitions,
        checkpoint["step"],
        input_hashes,
        code_hashes,
        device,
        results,
        observed_call_counts,
    )
    output_sha256 = write_immutable_json(args.out, result)
    print(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "case_count": result["summary"]["case_count"],
                "transition_count": result["summary"]["transition_count"],
                "model_calls": result["resource_ledger"]["model_calls"],
                "by_arm": result["summary"]["by_arm"],
                "paired_comparisons": result["summary"]["paired_comparisons"],
                "output_sha256": output_sha256,
                "reasoning_claim": "none",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
