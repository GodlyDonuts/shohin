#!/usr/bin/env python3
"""Evaluate the frozen Source-Scheduled Reasoning confirmation board."""

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


SCHEMA = "source_scheduled_reasoning_confirmation_v1"
RESULT_SCHEMA = "source_scheduled_reasoning_confirmation_result_v1"
SEED = 2026071502
PER_FAMILY = 64
FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)
EXPECTED_CASES_SHA256 = (
    "4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4"
)
EXPECTED_BOARD_SHA256 = (
    "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474"
)
MAX_NEW_FULL = 128
MAX_NEW_ATOMIC = 48
ROW_KEYS = {
    "id",
    "family",
    "question",
    "initial_state",
    "schedule",
    "answer",
    "stratum",
}
CALL_RECORD_KEYS = {
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
SCORED_CALL_KEYS = CALL_RECORD_KEYS | {"answer_segment", "predicted_answer", "correct"}
ATOMIC_STEP_KEYS = CALL_RECORD_KEYS | {
    "index",
    "operation",
    "operand",
    "input_state",
    "expected_state",
    "predicted_state",
    "correct",
}
SCHEDULED_STEP_KEYS = CALL_RECORD_KEYS | {
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
HEADER = re.compile(r"(?:^|\n)\s*(?:Question|Problem)(?:\s+\d+)?\s*:", re.IGNORECASE)
INTEGER = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?![A-Za-z0-9_,]|\.\d)"
)


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


def confirmation_source_paths():
    train_dir = Path(__file__).resolve().parent
    root = train_dir.parent
    return {
        "contract": root / "R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION.md",
        "generator": train_dir / "generate_scheduled_reasoning_confirmation.py",
        "evaluator": Path(__file__).resolve(),
        "job": train_dir / "jobs/eval_scheduled_reasoning_confirmation.sbatch",
        "model_loader": train_dir / "model.py",
    }


def hash_paths(paths):
    return {name: sha256_file(path) for name, path in paths.items()}


def answer_segment(response):
    return HEADER.split(response, maxsplit=1)[0].strip()


def last_integer(text):
    values = INTEGER.findall(text)
    return int(values[-1].replace(",", "")) if values else None


def first_nonempty_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def parse_first_line_final(text):
    return last_integer(first_nonempty_line(text))


def parse_full_response(response):
    segment = answer_segment(response)
    return segment, last_integer(segment)


def _parse_question(family, question):
    if family == "multiply_subtract":
        match = re.fullmatch(
            r"Compute (\d+) times (\d+), then subtract (\d+)\.", question
        )
        if not match:
            raise ValueError(f"unparsed multiply question: {question}")
        start, multiplier, subtractor = map(int, match.groups())
        return (
            start,
            [("multiply", multiplier), ("subtract", subtractor)],
            {
                "start": start,
                "multiplier": multiplier,
                "subtractor": subtractor,
            },
        )
    if family == "base_conversion":
        match = re.fullmatch(
            r"Convert the base-(\d+) numeral ([0-9]{3}) to base 10\.", question
        )
        if not match:
            raise ValueError(f"unparsed base question: {question}")
        base = int(match.group(1))
        digits = [int(value) for value in match.group(2)]
        schedule = [
            ("multiply", base),
            ("add", digits[1]),
            ("multiply", base),
            ("add", digits[2]),
        ]
        return digits[0], schedule, {"base": base, "digits": digits}
    if family == "sequential_state":
        match = re.fullmatch(
            r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\.",
            question,
        )
        if not match:
            raise ValueError(f"unparsed state question: {question}")
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
        if not match:
            raise ValueError(f"unparsed modular question: {question}")
        start, addend, modulus = map(int, match.groups())
        return (
            start,
            [("add", addend), ("remainder", modulus)],
            {
                "start": start,
                "addend": addend,
                "modulus": modulus,
            },
        )
    raise ValueError(f"unknown family: {family}")


def parse_schedule(row):
    start, schedule, _ = _parse_question(row.get("family"), row.get("question"))
    return start, schedule


def apply_operation(value, operation, operand):
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "remainder":
        return value % operand
    raise ValueError(operation)


def _in_range(value, lower, upper):
    return lower <= value <= upper


def _validate_row_ranges(family, family_index, details, answer):
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


def audit_board(board):
    if not isinstance(board, dict):
        raise ValueError("board must be an object")
    if board.get("schema") != SCHEMA or board.get("seed") != SEED:
        raise ValueError("wrong board schema or seed")
    rows = board.get("rows")
    expected_count = len(FAMILIES) * PER_FAMILY
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("wrong board row count")
    if board.get("case_count") != expected_count:
        raise ValueError("wrong board case_count metadata")
    if (
        board.get("per_family") != PER_FAMILY
        or tuple(board.get("family_order", ())) != FAMILIES
    ):
        raise ValueError("wrong board family contract")
    if board.get("cases_sha256") != EXPECTED_CASES_SHA256:
        raise ValueError("wrong frozen cases hash")
    if digest_rows(rows) != EXPECTED_CASES_SHA256:
        raise ValueError("board rows do not match the frozen cases hash")

    counts = Counter()
    questions = set()
    identifiers = set()
    total_steps = 0
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("wrong board row schema")
        expected_family = FAMILIES[position // PER_FAMILY]
        family_index = position % PER_FAMILY
        if row.get("family") != expected_family:
            raise ValueError("rows are not in frozen family order")
        expected_id = f"{expected_family}_{family_index:03d}"
        if row.get("id") != expected_id or expected_id in identifiers:
            raise ValueError("invalid or duplicate row id")
        identifiers.add(expected_id)
        question = row.get("question")
        if not isinstance(question, str) or question in questions:
            raise ValueError("invalid or duplicate question")
        questions.add(question)
        if (
            type(row.get("initial_state")) is not int
            or type(row.get("answer")) is not int
        ):
            raise ValueError("state and answer must be integers")

        start, schedule, details = _parse_question(expected_family, question)
        expected_stratum = _validate_row_ranges(
            expected_family, family_index, details, row["answer"]
        )
        if row.get("stratum") != expected_stratum:
            raise ValueError("row has wrong frozen stratum")
        raw_schedule = row.get("schedule")
        if not isinstance(raw_schedule, list):
            raise ValueError("schedule must be a list")
        normalized = []
        for step in raw_schedule:
            if (
                not isinstance(step, list)
                or len(step) != 2
                or not isinstance(step[0], str)
                or type(step[1]) is not int
            ):
                raise ValueError("invalid schedule step")
            normalized.append((step[0], step[1]))
        if start != row["initial_state"] or normalized != schedule:
            raise ValueError("question and public schedule disagree")

        state = start
        for operation, operand in schedule:
            state = apply_operation(state, operation, operand)
            total_steps += 1
        if state != row["answer"]:
            raise ValueError("schedule replay does not match final answer")
        counts[expected_family] += 1

    if counts != Counter({family: PER_FAMILY for family in FAMILIES}):
        raise ValueError("family balance mismatch")
    if total_steps != 704:
        raise ValueError("frozen transition count mismatch")
    return rows, total_steps


def load_frozen_board(path):
    path = Path(path)
    payload = path.read_bytes()
    board_sha256 = hashlib.sha256(payload).hexdigest()
    if board_sha256 != EXPECTED_BOARD_SHA256:
        raise ValueError("board artifact hash does not match the frozen board")
    if path.stat().st_mode & 0o222:
        raise PermissionError("frozen board must not have writable mode bits")
    board = json.loads(payload)
    rows, total_steps = audit_board(board)
    return board, rows, total_steps, board_sha256


def operation_clause(value, operation, operand):
    if operation == "add":
        return f"Compute {value} plus {operand}."
    if operation == "subtract":
        return f"Compute {value} minus {operand}."
    if operation == "multiply":
        return f"Compute {value} times {operand}."
    if operation == "remainder":
        return f"Give the remainder after dividing {value} by {operand}."
    raise ValueError(operation)


def format_atomic_prompt(value, operation, operand):
    return f"Problem: {operation_clause(value, operation, operand)}\nWork:"


def direct_prompt(question):
    return f"Question: {question} Return only the final integer.\nAnswer:"


def whole_prompt(question):
    return f"Problem: {question}\nWork:"


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
    if max_new <= 0:
        raise ValueError("max_new must be positive")
    cap = int(model.cfg.seq_len)
    full_prompt_ids = tokenizer.encode(prompt).ids
    prompt_ids = full_prompt_ids[-cap:]
    if not prompt_ids:
        raise ValueError("prompt encoded to no tokens")
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
        "untruncated_prompt_token_count": len(full_prompt_ids),
        "prompt_truncated": len(prompt_ids) != len(full_prompt_ids),
        "sampled_token_count": sampled_token_count,
        "decoded_token_count": len(generated),
        "stop_reason": stop_reason,
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def call(model, tokenizer, prompt, device, max_new, call_counts=None, arm=None):
    record = {
        "prompt": prompt,
        "max_new": max_new,
        **greedy_completion(model, tokenizer, prompt, device, max_new),
    }
    if call_counts is not None:
        if arm not in {
            "direct_qa",
            "whole_problem_work",
            "atomic_oracle_state",
            "source_scheduled",
        }:
            raise ValueError("counted model call requires a registered arm")
        call_counts[arm] += 1
    return record


def run_scheduled(
    model, tokenizer, device, initial_state, schedule, max_new, call_counts=None
):
    state = int(initial_state)
    steps = []
    for index, (operation, operand) in enumerate(schedule):
        prompt = format_atomic_prompt(state, operation, operand)
        if call_counts is None:
            record = call(model, tokenizer, prompt, device, max_new)
        else:
            record = call(
                model,
                tokenizer,
                prompt,
                device,
                max_new,
                call_counts,
                "source_scheduled",
            )
        predicted = parse_first_line_final(record["response"])
        record.update(
            {
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": state,
                "predicted_state": predicted,
            }
        )
        steps.append(record)
        if predicted is None:
            return None, steps
        state = predicted
    return state, steps


def exact_mcnemar_p(scheduler_only, direct_only):
    if (
        type(scheduler_only) is not int
        or type(direct_only) is not int
        or scheduler_only < 0
        or direct_only < 0
    ):
        raise ValueError("McNemar discordant counts must be nonnegative integers")
    discordant = scheduler_only + direct_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(scheduler_only, direct_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def decide(summary):
    case_count = summary["case_count"]
    atomic_total = summary["atomic_total"]
    sequential = summary["by_family"]["sequential_state"]
    gates = {
        "scheduled_absolute": summary["scheduled_correct"] * 100 >= 35 * case_count,
        "scheduled_advantage": (
            summary["scheduled_correct"] - summary["direct_correct"]
        )
        * 100
        >= 10 * case_count,
        "paired_significance": summary["mcnemar_exact_p"] < 0.01,
        "family_nonregression": all(
            item["scheduled_correct"] >= item["direct_correct"]
            for item in summary["by_family"].values()
        ),
        "sequential_absolute": (
            sequential["scheduled_correct"] * 100 >= 70 * sequential["count"]
        ),
        "atomic_ceiling": summary["atomic_correct"] * 100 >= 70 * atomic_total,
    }
    return gates, all(gates.values())


def _audit_call_record(record, expected_prompt, expected_max_new):
    if not isinstance(record, dict) or record.get("prompt") != expected_prompt:
        raise ValueError("renderer deviation in call transcript")
    if record.get("max_new") != expected_max_new:
        raise ValueError("decode cap deviation in call transcript")
    if not isinstance(record.get("response"), str):
        raise ValueError("missing response transcript")
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
        raise ValueError("invalid token accounting")
    if record["prompt_token_count"] <= 0 or record.get("prompt_truncated") is not False:
        raise ValueError("confirmation prompt was truncated")
    if record["prompt_token_count"] != record["untruncated_prompt_token_count"]:
        raise ValueError("prompt token accounting mismatch")
    stop_reason = record.get("stop_reason")
    if stop_reason not in {"eos", "max_new", "context_limit"}:
        raise ValueError("unregistered or premature decode stop")
    expected_sampled = record["decoded_token_count"] + (stop_reason == "eos")
    if record["sampled_token_count"] != expected_sampled:
        raise ValueError("sampled token accounting mismatch")
    if stop_reason == "max_new" and record["sampled_token_count"] != expected_max_new:
        raise ValueError("max_new stop occurred before the frozen token cap")
    if record["sampled_token_count"] > expected_max_new:
        raise ValueError("call exceeded the frozen token cap")


def audit_execution(results, board_rows, total_steps, observed_call_counts):
    if len(results) != len(board_rows):
        raise ValueError("missing result rows")
    arm_records = {
        "direct_qa": [],
        "whole_problem_work": [],
        "atomic_oracle_state": [],
        "source_scheduled": [],
    }
    scheduled_parse_failures = 0
    scheduled_unissued_calls = 0

    for result, row in zip(results, board_rows):
        if not isinstance(result, dict) or set(result) != RESULT_ROW_KEYS:
            raise ValueError("wrong result row schema")
        if any(
            result.get(key) != row[key]
            for key in ("id", "family", "question", "answer", "stratum")
        ):
            raise ValueError("result row does not match board row")
        start, schedule = parse_schedule(row)

        direct = result.get("direct")
        _audit_call_record(direct, direct_prompt(row["question"]), MAX_NEW_FULL)
        if set(direct) != SCORED_CALL_KEYS:
            raise ValueError("wrong direct transcript schema")
        direct_segment, direct_prediction = parse_full_response(direct["response"])
        if (
            direct.get("answer_segment") != direct_segment
            or direct.get("predicted_answer") != direct_prediction
            or direct.get("correct") != (direct_prediction == row["answer"])
        ):
            raise ValueError("direct score does not match transcript")
        arm_records["direct_qa"].append(direct)

        whole = result.get("whole_problem_work")
        _audit_call_record(whole, whole_prompt(row["question"]), MAX_NEW_FULL)
        if set(whole) != SCORED_CALL_KEYS:
            raise ValueError("wrong whole-work transcript schema")
        whole_segment, whole_prediction = parse_full_response(whole["response"])
        if (
            whole.get("answer_segment") != whole_segment
            or whole.get("predicted_answer") != whole_prediction
            or whole.get("correct") != (whole_prediction == row["answer"])
        ):
            raise ValueError("whole-work score does not match transcript")
        arm_records["whole_problem_work"].append(whole)

        atomic = result.get("atomic_oracle_state")
        if not isinstance(atomic, list) or len(atomic) != len(schedule):
            raise ValueError("atomic arm call count mismatch")
        true_state = start
        for index, ((operation, operand), record) in enumerate(zip(schedule, atomic)):
            expected_state = apply_operation(true_state, operation, operand)
            _audit_call_record(
                record,
                format_atomic_prompt(true_state, operation, operand),
                MAX_NEW_ATOMIC,
            )
            if set(record) != ATOMIC_STEP_KEYS:
                raise ValueError("wrong atomic transcript schema")
            predicted = parse_first_line_final(record["response"])
            expected_metadata = {
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": true_state,
                "expected_state": expected_state,
                "predicted_state": predicted,
                "correct": predicted == expected_state,
            }
            if any(
                record.get(key) != value for key, value in expected_metadata.items()
            ):
                raise ValueError("atomic transcript or score mismatch")
            arm_records["atomic_oracle_state"].append(record)
            true_state = expected_state

        scheduled = result.get("source_scheduled")
        if not isinstance(scheduled, dict) or set(scheduled) != {
            "predicted_answer",
            "correct",
            "steps",
        }:
            raise ValueError("wrong scheduled result schema")
        steps = scheduled.get("steps") if isinstance(scheduled, dict) else None
        if not isinstance(steps, list) or not steps or len(steps) > len(schedule):
            raise ValueError("scheduled arm call count mismatch")
        model_state = start
        parse_failed = False
        for index, record in enumerate(steps):
            operation, operand = schedule[index]
            _audit_call_record(
                record,
                format_atomic_prompt(model_state, operation, operand),
                MAX_NEW_ATOMIC,
            )
            if set(record) != SCHEDULED_STEP_KEYS:
                raise ValueError("scheduled transcript has non-controller fields")
            predicted = parse_first_line_final(record["response"])
            expected_metadata = {
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": model_state,
                "predicted_state": predicted,
            }
            if any(
                record.get(key) != value for key, value in expected_metadata.items()
            ):
                raise ValueError("scheduled carry or transcript mismatch")
            arm_records["source_scheduled"].append(record)
            if predicted is None:
                if index != len(steps) - 1:
                    raise ValueError("scheduled chain continued after parse failure")
                parse_failed = True
                break
            model_state = predicted

        if parse_failed:
            scheduled_parse_failures += 1
            scheduled_unissued_calls += len(schedule) - len(steps)
            expected_final = None
        else:
            if len(steps) != len(schedule):
                raise ValueError("scheduled chain terminated without parse failure")
            expected_final = model_state
        if scheduled.get("predicted_answer") != expected_final or scheduled.get(
            "correct"
        ) != (expected_final == row["answer"]):
            raise ValueError("scheduled final score mismatch")

    expected_counts = {
        "direct_qa": len(board_rows),
        "whole_problem_work": len(board_rows),
        "atomic_oracle_state": total_steps,
        "source_scheduled": total_steps - scheduled_unissued_calls,
    }
    actual_counts = {arm: len(records) for arm, records in arm_records.items()}
    if actual_counts != expected_counts:
        raise ValueError("model-call accounting mismatch")
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
        }
    resource_ledger = {
        "model_calls": sum(item["model_calls"] for item in by_arm.values()),
        "maximum_model_calls_without_parse_failures": 2 * len(board_rows)
        + 2 * total_steps,
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
    integrity = {
        "transcripts_complete": True,
        "renderers_exact": True,
        "model_call_accounting_exact": True,
        "scheduled_carry_is_model_only": True,
        "parse_failure_termination_exact": True,
        "no_unregistered_early_stop": True,
    }
    return resource_ledger, integrity


def write_immutable_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    if path.stat().st_mode & 0o222:
        raise PermissionError("confirmation output remained writable")
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if Path(args.out).exists():
        raise FileExistsError("refusing to overwrite confirmation output")
    board, rows, total_steps, board_sha256 = load_frozen_board(args.board)
    input_hashes = {
        "board": board_sha256,
        "checkpoint": sha256_file(args.ckpt),
        "tokenizer": sha256_file(args.tokenizer),
    }
    source_paths = confirmation_source_paths()
    implementation_hashes = hash_paths(source_paths)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)

    results = []
    observed_call_counts = Counter()
    for offset, row in enumerate(rows, 1):
        direct = call(
            model,
            tokenizer,
            direct_prompt(row["question"]),
            device,
            MAX_NEW_FULL,
            observed_call_counts,
            "direct_qa",
        )
        direct["answer_segment"], direct["predicted_answer"] = parse_full_response(
            direct["response"]
        )
        direct["correct"] = direct["predicted_answer"] == row["answer"]

        whole = call(
            model,
            tokenizer,
            whole_prompt(row["question"]),
            device,
            MAX_NEW_FULL,
            observed_call_counts,
            "whole_problem_work",
        )
        whole["answer_segment"], whole["predicted_answer"] = parse_full_response(
            whole["response"]
        )
        whole["correct"] = whole["predicted_answer"] == row["answer"]

        start, schedule = parse_schedule(row)
        true_state = start
        atomic = []
        for index, (operation, operand) in enumerate(schedule):
            expected = apply_operation(true_state, operation, operand)
            record = call(
                model,
                tokenizer,
                format_atomic_prompt(true_state, operation, operand),
                device,
                MAX_NEW_ATOMIC,
                observed_call_counts,
                "atomic_oracle_state",
            )
            predicted = parse_first_line_final(record["response"])
            record.update(
                {
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": true_state,
                    "expected_state": expected,
                    "predicted_state": predicted,
                    "correct": predicted == expected,
                }
            )
            atomic.append(record)
            true_state = expected

        scheduled_answer, scheduled_steps = run_scheduled(
            model,
            tokenizer,
            device,
            start,
            schedule,
            MAX_NEW_ATOMIC,
            observed_call_counts,
        )
        results.append(
            {
                "id": row["id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "question": row["question"],
                "answer": row["answer"],
                "direct": direct,
                "whole_problem_work": whole,
                "atomic_oracle_state": atomic,
                "source_scheduled": {
                    "predicted_answer": scheduled_answer,
                    "correct": scheduled_answer == row["answer"],
                    "steps": scheduled_steps,
                },
            }
        )
        if offset % 8 == 0:
            print(f"[scheduled-confirm] {offset}/{len(rows)}", flush=True)

    by_family = {}
    for family in FAMILIES:
        selected = [row for row in results if row["family"] == family]
        by_family[family] = {
            "count": len(selected),
            "direct_correct": sum(row["direct"]["correct"] for row in selected),
            "whole_correct": sum(
                row["whole_problem_work"]["correct"] for row in selected
            ),
            "scheduled_correct": sum(
                row["source_scheduled"]["correct"] for row in selected
            ),
            "atomic_correct": sum(
                step["correct"]
                for row in selected
                for step in row["atomic_oracle_state"]
            ),
            "atomic_total": sum(len(row["atomic_oracle_state"]) for row in selected),
        }
    scheduler_only = sum(
        row["source_scheduled"]["correct"] and not row["direct"]["correct"]
        for row in results
    )
    direct_only = sum(
        row["direct"]["correct"] and not row["source_scheduled"]["correct"]
        for row in results
    )
    summary = {
        "case_count": len(results),
        "transition_count": total_steps,
        "direct_correct": sum(row["direct"]["correct"] for row in results),
        "whole_correct": sum(row["whole_problem_work"]["correct"] for row in results),
        "scheduled_correct": sum(row["source_scheduled"]["correct"] for row in results),
        "atomic_correct": sum(
            step["correct"] for row in results for step in row["atomic_oracle_state"]
        ),
        "atomic_total": total_steps,
        "scheduler_only_correct": scheduler_only,
        "direct_only_correct": direct_only,
        "mcnemar_exact_p": exact_mcnemar_p(scheduler_only, direct_only),
        "by_family": by_family,
    }
    score_gates, score_advance = decide(summary)
    resource_ledger, execution_integrity = audit_execution(
        results, rows, total_steps, observed_call_counts
    )

    current_hashes = {
        "board": sha256_file(args.board),
        "checkpoint": sha256_file(args.ckpt),
        "tokenizer": sha256_file(args.tokenizer),
    }
    if current_hashes != input_hashes:
        raise RuntimeError("confirmation input changed during evaluation")
    if hash_paths(source_paths) != implementation_hashes:
        raise RuntimeError("confirmation implementation changed during evaluation")
    integrity_gates = {
        "frozen_board_artifact": True,
        "independent_board_structure": True,
        "input_hashes_stable": True,
        "implementation_hashes_stable": True,
        "decode_caps_frozen": True,
        "immutable_output": True,
        **execution_integrity,
    }
    advance = score_advance and all(integrity_gates.values())
    result = {
        "schema": RESULT_SCHEMA,
        "board": args.board,
        "board_sha256": input_hashes["board"],
        "cases_sha256": board["cases_sha256"],
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": input_hashes["checkpoint"],
        "tokenizer_sha256": input_hashes["tokenizer"],
        "implementation_sha256": implementation_hashes,
        "device": device,
        "max_new_full": MAX_NEW_FULL,
        "max_new_atomic": MAX_NEW_ATOMIC,
        "resource_ledger": resource_ledger,
        "summary": summary,
        "gates": score_gates,
        "integrity_gates": integrity_gates,
        "advance_to_internalization": advance,
        "rows": results,
    }
    output_sha256 = write_immutable_json(args.out, result)
    print(
        json.dumps(
            {
                "summary": summary,
                "gates": score_gates,
                "integrity_gates": integrity_gates,
                "advance": advance,
                "output_sha256": output_sha256,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
