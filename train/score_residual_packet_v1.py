#!/usr/bin/env python3
"""Fail-closed independent scorer for the frozen RSP-C1 experiment.

The scorer accepts only a hash-bound ``residual_packet_v1_score_manifest``.  It
does not import the evaluator and does not consume evaluator-produced metrics or
correctness fields.  Every packet, answer, executor line, prompt, token record,
trajectory, resource count, gate, and exact McNemar probability is reconstructed
from frozen artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer


MANIFEST_SCHEMA = "residual_packet_v1_score_manifest"
TRANSCRIPT_SCHEMA = "residual_packet_v1_raw_transcript"
DECISION_SCHEMA = "residual_packet_v1_decision"
BOARD_SCHEMA = "residual_packet_board_v1"
TRAINING_MANIFEST_SCHEMA = "residual_packet_generation_manifest_v1"
FIT_SEEDS = (2026071511, 2026071512)
STRATA = ("renderer_ood", "value_ood", "order_ood", "length_ood")
PER_STRATUM = 64
CASE_COUNT = 256
MAX_TRANSITIONS = 5
MAX_NEW_CONTROLLER = 80
MAX_NEW_EXECUTOR = 48
SWAP_COUNT = 64
RAW_EXECUTOR_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
PREREQUISITE_BOARD_SHA256 = "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474"
PREREQUISITE_CASES_SHA256 = "4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4"
UPDATER_SEED = 2026071505
EXPECTED_BOARD_ROWS_SHA256 = (
    "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
)
EXPECTED_BOARD_SHA256 = (
    "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2"
)

ASCII_WHITESPACE = " \t\n\r\f\v"
INTEGER_PATTERN = r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)"
POSITIVE_PATTERN = r"[1-9][0-9]*"
OPERATION_RE = re.compile(
    rf"\A(add|multiply|subtract) ({POSITIVE_PATTERN})\Z", re.ASCII
)
PACKET_RE = re.compile(
    rf"\AState: ({INTEGER_PATTERN})\nPlan: ("
    rf"(?:add|multiply|subtract) {POSITIVE_PATTERN}"
    rf"(?:; (?:add|multiply|subtract) {POSITIVE_PATTERN})*)\Z",
    re.ASCII,
)
ANSWER_RE = re.compile(rf"\AAnswer: ({INTEGER_PATTERN})\Z", re.ASCII)
EXECUTOR_INTEGER_RE = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?![A-Za-z0-9_,]|\.\d)"
)
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
HEADER_RE = re.compile(r"(?:^|\n)\s*(?:Question|Problem)(?:\s+\d+)?\s*:", re.I)

SOURCE_TEMPLATES = {
    "train_0": "Begin with {state}. Execute this sequence from left to right: {clauses}.",
    "train_1": "Take {state} as the running value. Perform, in sequence: {clauses}.",
    "train_2": "The starting number is {state}. Make these changes in order: {clauses}.",
    "train_3": "Set the running total to {state}. Follow the listed commands: {clauses}.",
    "reserved": "Initialize the value to {state}. Apply these instructions in order: {clauses}.",
}
HELD_OUT_BIGRAMS = (("multiply", "add"), ("subtract", "multiply"))

PROTOCOL_PATH = Path(__file__).resolve().with_name("residual_packet_protocol.py")


def _load_exact_protocol() -> Any:
    payload = PROTOCOL_PATH.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_PROTOCOL_SHA256:
        raise ImportError(
            "residual packet protocol hash mismatch: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(
        "_rsp_c1_scorer_protocol_e011e838", PROTOCOL_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load residual packet protocol at {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != PROTOCOL_PATH:
        raise ImportError("residual packet protocol loaded from an unexpected path")
    return module


ADMITTED_PROTOCOL = _load_exact_protocol()


class EvidenceError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(code, detail)


def _strict_int(value: object, label: str) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int),
        "invalid_integer",
        f"{label} must be an integer",
    )
    return int(value)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError("duplicate_json_key", f"{label} repeats {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvidenceError("nonfinite_json_number", f"{label} contains {item}")
            ),
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("invalid_json", f"cannot parse {label}: {error}") from error
    _require(isinstance(value, dict), "invalid_json_shape", f"{label} must be an object")
    return value


def _read_regular(path: Path, capture: bool = True) -> tuple[bytes | None, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceError("artifact_unreadable", f"cannot open {path}: {error}") from error
    chunks = [] if capture else None
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "artifact_not_regular", str(path))
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(
            all(getattr(before, name) == getattr(after, name) for name in stable),
            "artifact_changed_during_read",
            str(path),
        )
    finally:
        os.close(descriptor)
    return (b"".join(chunks) if chunks is not None else None), digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _read_regular(Path(path), capture=False)[1]


def _hash(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and HASH_RE.fullmatch(value) is not None,
        "invalid_sha256",
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _reference(
    value: object, label: str, directory: Path, *, json_artifact: bool
) -> tuple[dict[str, Any] | None, dict[str, str], Path]:
    _require(isinstance(value, dict), "invalid_artifact_reference", label)
    _require(set(value) == {"path", "sha256"}, "invalid_artifact_reference", label)
    raw_path = value["path"]
    expected = _hash(value["sha256"], f"{label}.sha256")
    _require(isinstance(raw_path, str) and raw_path, "invalid_artifact_path", label)
    path = Path(raw_path)
    if not path.is_absolute():
        path = directory / path
    path = Path(os.path.abspath(path))
    raw, observed = _read_regular(path, capture=json_artifact)
    _require(
        observed == expected,
        "artifact_hash_mismatch",
        f"{label}: expected {expected}, observed {observed}",
    )
    document = _parse_json(raw, label) if json_artifact and raw is not None else None
    return document, {"path": str(raw_path), "sha256": observed}, path


def parse_operation(text: object) -> tuple[str, int] | None:
    if not isinstance(text, str):
        return None
    match = OPERATION_RE.fullmatch(text)
    return (match.group(1), int(match.group(2))) if match is not None else None


def parse_packet(text: object) -> tuple[int, tuple[tuple[str, int], ...]] | None:
    if not isinstance(text, str):
        return None
    body = text.strip(ASCII_WHITESPACE)
    match = PACKET_RE.fullmatch(body)
    if match is None:
        return None
    plan = tuple(parse_operation(item) for item in match.group(2).split("; "))
    if any(operation is None for operation in plan):
        return None
    return int(match.group(1)), tuple(plan)  # type: ignore[arg-type]


def parse_answer(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    match = ANSWER_RE.fullmatch(text.strip(ASCII_WHITESPACE))
    return int(match.group(1)) if match is not None else None


def canonical_packet(state: int, plan: Sequence[tuple[str, int]]) -> str:
    _strict_int(state, "packet state")
    _require(bool(plan), "empty_packet_plan", "packet plan may not be empty")
    operations = []
    for operation, operand in plan:
        rendered = f"{operation} {operand}"
        _require(parse_operation(rendered) == (operation, operand), "invalid_operation", rendered)
        operations.append(rendered)
    return f"State: {state}\nPlan: " + "; ".join(operations)


def canonical_answer(value: int) -> str:
    return f"Answer: {_strict_int(value, 'answer')}"


def compiler_prompt(source: str) -> str:
    return f"Problem: {source}\nCompile only the execution packet.\nPacket:"


def update_prompt(packet: str, observed: int) -> str:
    _require(parse_packet(packet) is not None, "invalid_packet", "updater packet")
    return f"Packet:\n{packet}\nObserved result: {observed}\nNext packet:"


def executor_prompt(state: int, operation: str, operand: int) -> str:
    if operation == "add":
        clause = f"Compute {state} plus {operand}."
    elif operation == "multiply":
        clause = f"Compute {state} times {operand}."
    elif operation == "subtract":
        clause = f"Compute {state} minus {operand}."
    else:
        raise EvidenceError("invalid_operation", operation)
    return f"Problem: {clause}\nWork:"


def parse_executor_result(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    values = EXECUTOR_INTEGER_RE.findall(line)
    return int(values[-1].replace(",", "")) if values else None


def apply_operation(state: int, operation: str, operand: int) -> int:
    if operation == "add":
        return state + operand
    if operation == "multiply":
        return state * operand
    if operation == "subtract":
        return state - operand
    raise EvidenceError("invalid_operation", operation)


def replay(initial: int, operations: Sequence[tuple[str, int]]) -> tuple[int, ...]:
    states = [initial]
    for operation, operand in operations:
        states.append(apply_operation(states[-1], operation, operand))
    return tuple(states)


def _normalize_operations(value: object) -> tuple[tuple[str, int], ...]:
    _require(isinstance(value, list) and value, "invalid_board_program", "operations")
    result = []
    for item in value:
        _require(isinstance(item, list) and len(item) == 2, "invalid_board_program", str(item))
        parsed = parse_operation(f"{item[0]} {item[1]}")
        _require(parsed is not None, "invalid_board_program", str(item))
        result.append(parsed)
    return tuple(result)


def _source(initial: int, operations: Sequence[tuple[str, int]], template: str) -> str:
    clauses = "; ".join(
        f"multiply by {operand}" if operation == "multiply" else f"{operation} {operand}"
        for operation, operand in operations
    )
    return SOURCE_TEMPLATES[template].format(state=initial, clauses=clauses)


def audit_board(board: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    _require(board.get("schema") == BOARD_SCHEMA, "board_schema_mismatch", BOARD_SCHEMA)
    _require(board.get("seed") == 2026071503, "board_seed_mismatch", "board seed")
    _require(board.get("case_count") == CASE_COUNT, "board_count_mismatch", "case_count")
    _require(board.get("per_stratum") == PER_STRATUM, "board_count_mismatch", "per_stratum")
    _require(tuple(board.get("stratum_order", ())) == STRATA, "board_order_mismatch", "strata")
    rows = board.get("rows")
    _require(isinstance(rows, list) and len(rows) == CASE_COUNT, "board_count_mismatch", "rows")
    expected_keys = {
        "answer", "id", "initial_state", "operations", "packet", "source",
        "stratum", "template_id", "trajectory",
    }
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_programs: set[tuple[Any, ...]] = set()
    seen_trajectories: set[tuple[int, ...]] = set()
    seen_answers: set[int] = set()
    counts = Counter()
    normalized = []
    for position, raw in enumerate(rows):
        _require(isinstance(raw, dict) and set(raw) == expected_keys, "board_row_schema", str(position))
        stratum = STRATA[position // PER_STRATUM]
        identifier = f"{stratum}_{position % PER_STRATUM:03d}"
        _require(raw["stratum"] == stratum and raw["id"] == identifier, "board_order_mismatch", identifier)
        initial = _strict_int(raw["initial_state"], f"{identifier}.initial_state")
        operations = _normalize_operations(raw["operations"])
        _require(2 < len(operations) <= MAX_TRANSITIONS, "board_length_mismatch", identifier)
        trajectory = replay(initial, operations)
        _require(list(trajectory) == raw["trajectory"], "board_trajectory_mismatch", identifier)
        _require(trajectory[-1] == raw["answer"] and min(trajectory) > 0, "board_answer_mismatch", identifier)
        _require(raw["packet"] == canonical_packet(initial, operations), "board_packet_mismatch", identifier)
        template = raw["template_id"]
        _require(template in SOURCE_TEMPLATES, "board_template_mismatch", identifier)
        _require(raw["source"] == _source(initial, operations, template), "board_source_mismatch", identifier)
        bigrams = tuple(zip((item[0] for item in operations), (item[0] for item in operations[1:])))
        held = [pair for pair in bigrams if pair in HELD_OUT_BIGRAMS]
        if stratum == "renderer_ood":
            valid = len(operations) == 3 and template == "reserved" and not held
        elif stratum == "value_ood":
            valid = (
                len(operations) == 3 and template != "reserved" and not held
                and 100 <= initial <= 299
                and all((8 <= operand <= 12) if op == "multiply" else (26 <= operand <= 75) for op, operand in operations)
            )
        elif stratum == "order_ood":
            expected_held = HELD_OUT_BIGRAMS[0 if position % PER_STRATUM < 32 else 1]
            valid = len(operations) in (3, 4) and template != "reserved" and held == [expected_held]
        else:
            valid = len(operations) == 5 and template != "reserved" and not held
        _require(valid, "board_stratum_contract", identifier)
        if stratum != "value_ood":
            _require(10 <= initial <= 99, "board_value_contract", identifier)
            _require(
                all((2 <= operand <= 7) if op == "multiply" else (2 <= operand <= 25) for op, operand in operations),
                "board_value_contract",
                identifier,
            )
        signature = (initial, operations)
        _require(identifier not in seen_ids and raw["source"] not in seen_sources, "board_duplicate", identifier)
        _require(signature not in seen_programs and trajectory not in seen_trajectories, "board_duplicate", identifier)
        _require(trajectory[-1] not in seen_answers, "board_duplicate", identifier)
        seen_ids.add(identifier)
        seen_sources.add(raw["source"])
        seen_programs.add(signature)
        seen_trajectories.add(trajectory)
        seen_answers.add(trajectory[-1])
        counts[stratum] += 1
        normalized.append({**raw, "operations": operations, "trajectory": trajectory})
    _require(counts == Counter({item: PER_STRATUM for item in STRATA}), "board_balance_mismatch", str(counts))
    rows_digest = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    _require(
        rows_digest == EXPECTED_BOARD_ROWS_SHA256
        and board.get("rows_sha256") == rows_digest,
        "board_rows_hash_mismatch",
        rows_digest,
    )
    return normalized, rows_digest


def exact_mcnemar(treatment_only: int, sham_only: int) -> dict[str, Any]:
    for value in (treatment_only, sham_only):
        _strict_int(value, "McNemar discordance")
        _require(value >= 0, "invalid_mcnemar", "negative discordance")
    discordant = treatment_only + sham_only
    if discordant == 0:
        numerator, denominator = 1, 1
    else:
        denominator = 2**discordant
        numerator = min(
            denominator,
            2 * sum(math.comb(discordant, index) for index in range(min(treatment_only, sham_only) + 1)),
        )
        divisor = math.gcd(numerator, denominator)
        numerator //= divisor
        denominator //= divisor
    return {
        "treatment_only": treatment_only,
        "sham_only": sham_only,
        "numerator": numerator,
        "denominator": denominator,
        "p": numerator / denominator,
    }


class CallAuditor:
    CALL_KEYS = {
        "call_index", "model", "arm", "prompt", "max_new", "response",
        "prompt_token_count", "sampled_token_ids", "sampled_token_count",
        "decoded_token_ids", "decoded_token_count", "stop_reason",
    }

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []

    def check(self, value: object, model: str, arm: str, prompt: str, max_new: int) -> dict[str, Any]:
        _require(isinstance(value, dict) and set(value) == self.CALL_KEYS, "call_schema_mismatch", f"{model}/{arm}")
        call = dict(value)
        _require(call["model"] == model and call["arm"] == arm, "call_identity_mismatch", f"{model}/{arm}")
        _require(call["prompt"] == prompt and call["max_new"] == max_new, "call_prompt_mismatch", f"{model}/{arm}")
        _strict_int(call["call_index"], "call_index")
        prompt_ids = list(self.tokenizer.encode(prompt).ids)
        _require(call["prompt_token_count"] == len(prompt_ids) > 0, "prompt_token_mismatch", f"{model}/{arm}")
        sampled = call["sampled_token_ids"]
        decoded = call["decoded_token_ids"]
        _require(
            isinstance(sampled, list) and isinstance(decoded, list)
            and all(type(token) is int and token >= 0 for token in sampled + decoded),
            "token_id_mismatch",
            f"{model}/{arm}",
        )
        _require(call["sampled_token_count"] == len(sampled), "sampled_token_mismatch", f"{model}/{arm}")
        _require(call["decoded_token_count"] == len(decoded), "decoded_token_mismatch", f"{model}/{arm}")
        stop = call["stop_reason"]
        _require(stop in {"eos", "max_new", "context_limit"}, "stop_reason_mismatch", str(stop))
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if stop == "eos":
            _require(eos_id is not None and sampled == decoded + [eos_id], "eos_accounting_mismatch", f"{model}/{arm}")
        else:
            _require(sampled == decoded, "sampled_decoded_mismatch", f"{model}/{arm}")
        if stop == "max_new":
            _require(len(sampled) == max_new, "decode_cap_mismatch", f"{model}/{arm}")
        _require(len(sampled) <= max_new, "decode_cap_mismatch", f"{model}/{arm}")
        try:
            decoded_text = self.tokenizer.decode(decoded, skip_special_tokens=True)
        except TypeError:
            decoded_text = self.tokenizer.decode(decoded)
        _require(call["response"] == decoded_text, "response_token_mismatch", f"{model}/{arm}")
        self.calls.append(call)
        return call

    def finish(self, expected_count: int) -> None:
        indexes = sorted(call["call_index"] for call in self.calls)
        _require(indexes == list(range(expected_count)), "call_order_mismatch", "call indexes")


def _runtime_score(
    runtime: object,
    initial_packet: str,
    controller: str,
    arm: str,
    auditor: CallAuditor,
) -> dict[str, Any]:
    _require(isinstance(runtime, dict) and set(runtime) == {"termination", "steps"}, "runtime_schema_mismatch", arm)
    parsed = parse_packet(initial_packet)
    expected_termination = "initial_packet_invalid" if parsed is None else None
    steps = runtime["steps"]
    _require(isinstance(steps, list), "runtime_schema_mismatch", arm)
    observations = []
    update_exact = []
    packet = parsed
    for step_index, step in enumerate(steps):
        _require(packet is not None and step_index < MAX_TRANSITIONS, "runtime_extra_call", arm)
        _require(isinstance(step, dict) and set(step) in ({"executor"}, {"executor", "updater"}), "runtime_step_schema", arm)
        state, plan = packet
        operation, operand = plan[0]
        executor = auditor.check(
            step["executor"], "raw_260k_executor", arm,
            executor_prompt(state, operation, operand), MAX_NEW_EXECUTOR,
        )
        observed = parse_executor_result(executor["response"])
        if observed is None:
            _require("updater" not in step and step_index == len(steps) - 1, "runtime_continued_after_failure", arm)
            expected_termination = "executor_result_invalid"
            packet = None
            break
        observations.append(observed)
        _require("updater" in step, "missing_updater_call", arm)
        canonical = canonical_packet(state, plan)
        updater = auditor.check(
            step["updater"], controller, arm,
            update_prompt(canonical, observed), MAX_NEW_CONTROLLER,
        )
        expected_response = canonical_answer(observed) if len(plan) == 1 else canonical_packet(observed, plan[1:])
        exact = updater["response"].strip(ASCII_WHITESPACE) == expected_response
        update_exact.append(exact)
        if not exact:
            _require(step_index == len(steps) - 1, "runtime_continued_after_failure", arm)
            expected_termination = "updater_output_invalid"
            packet = None
            break
        if len(plan) == 1:
            _require(step_index == len(steps) - 1, "runtime_continued_after_answer", arm)
            expected_termination = "answer"
            packet = None
            break
        packet = (observed, plan[1:])
    if parsed is not None and expected_termination is None:
        expected_termination = "transition_limit" if len(steps) == MAX_TRANSITIONS else "updater_output_invalid"
    _require(runtime["termination"] == expected_termination, "termination_mismatch", arm)
    return {
        "observations": observations,
        "update_exact": update_exact,
        "complete": expected_termination == "answer",
        "executor_calls": len(steps),
        "updater_calls": sum(isinstance(step, dict) and "updater" in step for step in steps),
    }


def _teacher_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answers = {int(row["answer"]) for row in rows}
    result = []
    for row in rows:
        for index in range(len(row["operations"])):
            nonce = 0
            while True:
                label = f"{UPDATER_SEED}:{row['id']}:{index}:state:{nonce}".encode("ascii")
                state = 10 + int.from_bytes(hashlib.sha256(label).digest()[:8], "big") % 990
                if state not in answers:
                    break
                nonce += 1
            nonce = 0
            while True:
                label = f"{UPDATER_SEED}:{row['id']}:{index}:observed:{nonce}".encode("ascii")
                observed = 10 + int.from_bytes(hashlib.sha256(label).digest()[:8], "big") % 990
                operation, operand = row["operations"][index]
                if observed not in answers and observed != apply_operation(state, operation, operand):
                    break
                nonce += 1
            result.append({
                "id": row["id"], "step_index": index,
                "packet": canonical_packet(state, row["operations"][index:]),
                "observed": observed,
            })
    return result


def _swap_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result = []
    for stratum in STRATA:
        selected = [row for row in rows if row["stratum"] == stratum][:16]
        for index, row in enumerate(selected):
            result.append((row, selected[(index + 1) % 16]))
    return result


def _assert_no_booleans(value: object, path: str = "$") -> None:
    _require(not isinstance(value, bool), "trusted_boolean_in_transcript", path)
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_booleans(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_booleans(item, f"{path}[{index}]")


def score_transcript(
    transcript: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    seed: int,
    hashes: Mapping[str, str],
    checkpoint_hashes: Mapping[str, str],
    expected_training_resources: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    _assert_no_booleans(transcript)
    expected_top = {
        "schema", "seed", "protocol_module", "input_hashes", "decode_caps", "models",
        "external_scheduler", "controllers", "resource_ledger", "call_count",
    }
    _require(set(transcript) == expected_top, "transcript_schema_mismatch", str(seed))
    _require(transcript["schema"] == TRANSCRIPT_SCHEMA and transcript["seed"] == seed, "transcript_identity_mismatch", str(seed))
    _require(transcript["decode_caps"] == {"controller": 80, "executor": 48, "maximum_transitions": 5}, "decode_cap_mismatch", str(seed))
    input_hashes = transcript["input_hashes"]
    _require(isinstance(input_hashes, dict), "input_hash_mismatch", str(seed))
    _require(set(input_hashes) == set(hashes), "input_hash_mismatch", f"{seed}/artifact set")
    for name, digest in hashes.items():
        _require(input_hashes.get(name) == digest, "input_hash_mismatch", f"{seed}/{name}")
    models = transcript["models"]
    _require(isinstance(models, dict) and set(models) == {"treatment", "sham", "raw_260k_executor"}, "model_manifest_mismatch", str(seed))
    for name, digest in checkpoint_hashes.items():
        _require(models[name].get("checkpoint_sha256") == digest, "checkpoint_binding_mismatch", f"{seed}/{name}")

    auditor = CallAuditor(tokenizer)
    external_rows = transcript["external_scheduler"]
    _require(isinstance(external_rows, list) and len(external_rows) == CASE_COUNT, "external_row_count", str(seed))
    external_trajectories: dict[str, tuple[int, ...] | None] = {}
    external_gold = 0
    for row, record in zip(rows, external_rows):
        _require(isinstance(record, dict) and set(record) == {"id", "runtime"} and record["id"] == row["id"], "external_row_mismatch", row["id"])
        runtime = record["runtime"]
        _require(isinstance(runtime, dict) and set(runtime) == {"termination", "steps"}, "runtime_schema_mismatch", "external")
        state = row["initial_state"]
        observed = []
        for index, call in enumerate(runtime["steps"]):
            _require(index < len(row["operations"]), "runtime_extra_call", "external")
            operation, operand = row["operations"][index]
            checked = auditor.check(call, "raw_260k_executor", "external_scheduler", executor_prompt(state, operation, operand), MAX_NEW_EXECUTOR)
            state = parse_executor_result(checked["response"])
            if state is None:
                break
            observed.append(state)
        complete = len(observed) == len(row["operations"])
        expected_term = "complete" if complete else "executor_result_invalid"
        _require(runtime["termination"] == expected_term, "termination_mismatch", "external")
        trajectory = (row["initial_state"], *observed) if complete else None
        external_trajectories[row["id"]] = trajectory
        external_gold += int(complete and observed[-1] == row["answer"])

    controllers = transcript["controllers"]
    _require(isinstance(controllers, dict) and set(controllers) == {"treatment", "sham"}, "controller_set_mismatch", str(seed))
    arm_scores: dict[str, Any] = {}
    expected_teacher = _teacher_cases(rows)
    expected_swaps = _swap_pairs(rows)
    unissued: defaultdict[tuple[str, str], int] = defaultdict(int)
    unissued[("raw_260k_executor", "external_scheduler")] = sum(
        len(row["operations"]) - (len(external_trajectories[row["id"]]) - 1 if external_trajectories[row["id"]] else len(transcript["external_scheduler"][index]["runtime"]["steps"]))
        for index, row in enumerate(rows)
    )
    for controller in ("treatment", "sham"):
        payload = controllers[controller]
        _require(isinstance(payload, dict) and set(payload) == {"strict_closed_loop", "oracle_packet_loop", "teacher_forced_updater", "packet_swaps"}, "controller_schema_mismatch", controller)
        strict_records = payload["strict_closed_loop"]
        oracle_records = payload["oracle_packet_loop"]
        _require(len(strict_records) == CASE_COUNT and len(oracle_records) == CASE_COUNT, "controller_row_count", controller)
        compile_bits = []
        strict_bits = []
        oracle_bits = []
        trajectory_bits = []
        gold_bits = []
        strict_by_stratum = Counter()
        compile_by_stratum = Counter()
        strict_lengths = Counter()
        length_counts = Counter()
        for row, strict_record, oracle_record in zip(rows, strict_records, oracle_records):
            _require(isinstance(strict_record, dict) and set(strict_record) == {"id", "compiler", "runtime"} and strict_record["id"] == row["id"], "strict_row_mismatch", row["id"])
            compiler = auditor.check(strict_record["compiler"], controller, "strict_closed_loop", compiler_prompt(row["source"]), MAX_NEW_CONTROLLER)
            compiled = parse_packet(compiler["response"])
            compile_exact = compiled == (row["initial_state"], row["operations"])
            strict_runtime = _runtime_score(strict_record["runtime"], compiler["response"], controller, "strict_closed_loop", auditor)
            strict_exact = bool(compile_exact and strict_runtime["complete"] and all(strict_runtime["update_exact"]) and len(strict_runtime["observations"]) == len(row["operations"]))
            external = external_trajectories[row["id"]]
            emitted = (row["initial_state"], *strict_runtime["observations"]) if compile_exact and strict_runtime["complete"] else None
            trajectory_match = emitted is not None and emitted == external
            final_gold = strict_exact and bool(strict_runtime["observations"]) and strict_runtime["observations"][-1] == row["answer"]
            compile_bits.append(compile_exact)
            strict_bits.append(strict_exact)
            trajectory_bits.append(trajectory_match)
            gold_bits.append(final_gold)
            compile_by_stratum[row["stratum"]] += int(compile_exact)
            strict_by_stratum[row["stratum"]] += int(strict_exact)
            length = len(row["operations"])
            length_counts[length] += 1
            strict_lengths[length] += int(strict_exact)
            unissued[(controller, "strict_closed_loop")] += len(row["operations"]) - strict_runtime["updater_calls"]
            unissued[("raw_260k_executor", "strict_closed_loop")] += len(row["operations"]) - strict_runtime["executor_calls"]

            _require(isinstance(oracle_record, dict) and set(oracle_record) == {"id", "runtime"} and oracle_record["id"] == row["id"], "oracle_row_mismatch", row["id"])
            oracle_runtime = _runtime_score(oracle_record["runtime"], row["packet"], controller, "oracle_packet_loop", auditor)
            oracle_exact = bool(oracle_runtime["complete"] and all(oracle_runtime["update_exact"]) and len(oracle_runtime["observations"]) == len(row["operations"]))
            oracle_bits.append(oracle_exact)
            unissued[(controller, "oracle_packet_loop")] += len(row["operations"]) - oracle_runtime["updater_calls"]
            unissued[("raw_260k_executor", "oracle_packet_loop")] += len(row["operations"]) - oracle_runtime["executor_calls"]

        teacher_records = payload["teacher_forced_updater"]
        _require(isinstance(teacher_records, list) and len(teacher_records) == len(expected_teacher), "teacher_row_count", controller)
        teacher_exact = []
        teacher_halt = []
        for expected, record in zip(expected_teacher, teacher_records):
            _require(isinstance(record, dict) and set(record) == {"id", "step_index", "packet", "observed", "call"}, "teacher_row_schema", controller)
            _require({key: record[key] for key in expected} == expected, "teacher_case_mismatch", expected["id"])
            call = auditor.check(record["call"], controller, "teacher_forced_updater", update_prompt(expected["packet"], expected["observed"]), MAX_NEW_CONTROLLER)
            _, plan = parse_packet(expected["packet"])  # type: ignore[misc]
            expected_response = canonical_answer(expected["observed"]) if len(plan) == 1 else canonical_packet(expected["observed"], plan[1:])
            exact = call["response"].strip(ASCII_WHITESPACE) == expected_response
            teacher_exact.append(exact)
            if len(plan) == 1:
                teacher_halt.append(exact)

        swap_records = payload["packet_swaps"]
        _require(isinstance(swap_records, list) and len(swap_records) == SWAP_COUNT, "swap_count_mismatch", controller)
        swap_follow = []
        for (original, donor), record in zip(expected_swaps, swap_records):
            _require(isinstance(record, dict) and set(record) == {"original_id", "donor_id", "compiler", "intervened_packet", "runtime"}, "swap_row_schema", controller)
            _require(record["original_id"] == original["id"] and record["donor_id"] == donor["id"], "swap_pair_mismatch", controller)
            auditor.check(record["compiler"], controller, "packet_swap", compiler_prompt(original["source"]), MAX_NEW_CONTROLLER)
            _require(record["intervened_packet"] == donor["packet"], "swap_packet_mismatch", controller)
            swap_runtime = _runtime_score(record["runtime"], donor["packet"], controller, "packet_swap", auditor)
            emitted = (donor["initial_state"], *swap_runtime["observations"]) if swap_runtime["complete"] else None
            donor_external = external_trajectories[donor["id"]]
            original_external = external_trajectories[original["id"]]
            swap_follow.append(emitted is not None and emitted == donor_external and emitted != original_external)
            unissued[(controller, "packet_swap")] += len(donor["operations"]) - swap_runtime["updater_calls"]
            unissued[("raw_260k_executor", "packet_swap")] += len(donor["operations"]) - swap_runtime["executor_calls"]

        update_rate = sum(teacher_exact) / len(teacher_exact)
        halt_rate = sum(teacher_halt) / len(teacher_halt)
        compile_rate = sum(compile_bits) / CASE_COUNT
        length_curve = {}
        for length in sorted(length_counts):
            predicted = compile_rate * halt_rate * (update_rate**length)
            length_curve[str(length)] = {
                "correct": strict_lengths[length], "total": length_counts[length],
                "observed": strict_lengths[length] / length_counts[length],
                "stationary_prediction": predicted,
            }
        arm_scores[controller] = {
            "compile_bits": compile_bits,
            "strict_bits": strict_bits,
            "compile_exact": sum(compile_bits),
            "strict_closed_loop": sum(strict_bits),
            "oracle_packet_loop": sum(oracle_bits),
            "teacher_update_exact": sum(teacher_exact),
            "teacher_update_total": len(teacher_exact),
            "external_trajectory_match": sum(trajectory_bits),
            "external_trajectory_mismatches": CASE_COUNT - sum(trajectory_bits),
            "gold_answer": sum(gold_bits),
            "packet_swap_follow": sum(swap_follow),
            "by_stratum": {
                stratum: {"compile_exact": compile_by_stratum[stratum], "strict_closed_loop": strict_by_stratum[stratum]}
                for stratum in STRATA
            },
            "length_curve": length_curve,
        }

    auditor.finish(_strict_int(transcript["call_count"], "call_count"))
    _verify_ledger(
        transcript["resource_ledger"],
        auditor.calls,
        unissued,
        expected_training_resources or {},
    )
    treatment = arm_scores["treatment"]
    sham = arm_scores["sham"]
    compile_mc = exact_mcnemar(
        sum(t and not s for t, s in zip(treatment["compile_bits"], sham["compile_bits"])),
        sum(s and not t for t, s in zip(treatment["compile_bits"], sham["compile_bits"])),
    )
    strict_mc = exact_mcnemar(
        sum(t and not s for t, s in zip(treatment["strict_bits"], sham["strict_bits"])),
        sum(s and not t for t, s in zip(treatment["strict_bits"], sham["strict_bits"])),
    )
    gates = {
        "raw_external_scheduler_gold": external_gold >= 128,
        "oracle_packet_loop": treatment["oracle_packet_loop"] >= 230,
        "initial_compilation": treatment["compile_exact"] >= 224,
        "conditional_packet_update": 20 * treatment["teacher_update_exact"] >= 19 * treatment["teacher_update_total"],
        "strict_source_deleted_closed_loop": treatment["strict_closed_loop"] >= 192,
        "per_stratum_compilation": all(treatment["by_stratum"][item]["compile_exact"] >= 52 for item in STRATA),
        "per_stratum_strict": all(treatment["by_stratum"][item]["strict_closed_loop"] >= 40 for item in STRATA),
        "external_trajectory_mismatches": treatment["external_trajectory_mismatches"] <= 8,
        "treatment_sham_compilation_gap": 100 * (treatment["compile_exact"] - sham["compile_exact"]) >= 30 * CASE_COUNT,
        "treatment_sham_strict_gap": 100 * (treatment["strict_closed_loop"] - sham["strict_closed_loop"]) >= 25 * CASE_COUNT,
        "treatment_beats_sham_every_stratum": all(
            treatment["by_stratum"][item]["compile_exact"] > sham["by_stratum"][item]["compile_exact"]
            and treatment["by_stratum"][item]["strict_closed_loop"] > sham["by_stratum"][item]["strict_closed_loop"]
            for item in STRATA
        ),
        "compilation_mcnemar": compile_mc["numerator"] * 100 < compile_mc["denominator"],
        "strict_mcnemar": strict_mc["numerator"] * 100 < strict_mc["denominator"],
        "packet_swap": treatment["packet_swap_follow"] >= 60,
    }
    for arm in arm_scores.values():
        arm.pop("compile_bits")
        arm.pop("strict_bits")
    return {
        "seed": seed, "external_scheduler_gold": external_gold, "arms": arm_scores,
        "mcnemar": {"compilation": compile_mc, "strict_closed_loop": strict_mc},
        "gates": gates, "passed": all(gates.values()),
    }


def _verify_ledger(
    value: object,
    calls: Sequence[Mapping[str, Any]],
    unissued: Mapping[tuple[str, str], int],
    training_resources: Mapping[str, Mapping[str, int]],
) -> None:
    _require(isinstance(value, dict) and set(value) == {"by_model"}, "ledger_schema_mismatch", "resource ledger")
    by_model = value["by_model"]
    _require(isinstance(by_model, dict) and set(by_model) == {"treatment", "sham", "raw_260k_executor"}, "ledger_model_mismatch", "models")
    for model in by_model:
        payload = by_model[model]
        _require(isinstance(payload, dict) and set(payload) == {"by_arm"}, "ledger_schema_mismatch", model)
        by_arm = payload["by_arm"]
        _require(isinstance(by_arm, dict) and "training" in by_arm, "ledger_schema_mismatch", model)
        expected_arms = {call["arm"] for call in calls if call["model"] == model} | {arm for name, arm in unissued if name == model} | {"training"}
        _require(set(by_arm) == expected_arms, "ledger_arm_mismatch", model)
        for arm, reported in by_arm.items():
            _require(isinstance(reported, dict), "ledger_schema_mismatch", f"{model}/{arm}")
            selected = [call for call in calls if call["model"] == model and call["arm"] == arm]
            training = training_resources.get(model, {})
            expected = {
                "model_calls": len(selected),
                "prompt_tokens": sum(call["prompt_token_count"] for call in selected),
                "sampled_tokens": sum(call["sampled_token_count"] for call in selected),
                "decoded_tokens": sum(call["decoded_token_count"] for call in selected),
                "supervised_completion_tokens": int(
                    training.get("supervised_completion_tokens", 0)
                )
                if arm == "training"
                else 0,
                "packed_forward_token_positions": int(
                    training.get("packed_forward_token_positions", 0)
                )
                if arm == "training"
                else 0,
                "calls_not_issued_after_parse_failure": unissued.get((model, arm), 0),
                "retries": 0, "repairs": 0, "searches": 0, "verifier_feedback_calls": 0,
            }
            _require(reported == expected, "ledger_count_mismatch", f"{model}/{arm}")


def training_resource_counts(
    value: Mapping[str, Any], seed: int
) -> dict[str, dict[str, int]]:
    _require(
        value.get("paired_seed") in (None, seed),
        "training_resource_seed_mismatch",
        str(seed),
    )
    result = {}
    for arm in ("treatment", "sham"):
        item = value.get(arm)
        _require(isinstance(item, dict), "training_resource_schema", arm)
        _require(
            {
                "supervised_completion_tokens",
                "packed_forward_token_positions",
            }.issubset(item),
            "training_resource_schema",
            f"{arm} lacks exact consumed token counts",
        )
        supervised = _strict_int(
            item["supervised_completion_tokens"], f"{arm}.supervised"
        )
        forward = _strict_int(
            item["packed_forward_token_positions"], f"{arm}.forward"
        )
        _require(
            supervised >= 0 and forward >= 0,
            "training_resource_schema",
            f"{arm} has negative token counts",
        )
        result[arm] = {
            "supervised_completion_tokens": supervised,
            "packed_forward_token_positions": forward,
        }
    _require(result["treatment"] == result["sham"], "training_resource_arm_mismatch", str(seed))
    result["raw_260k_executor"] = {
        "supervised_completion_tokens": 0,
        "packed_forward_token_positions": 0,
    }
    return result


def _prerequisite_schedule(
    family: object, question: object
) -> tuple[int, tuple[tuple[str, int], ...]]:
    _require(isinstance(question, str), "prerequisite_question_mismatch", str(question))
    patterns = {
        "multiply_subtract": re.compile(
            r"Compute (\d+) times (\d+), then subtract (\d+)\.\Z"
        ),
        "base_conversion": re.compile(
            r"Convert the base-(\d+) numeral ([0-9]{3}) to base 10\.\Z"
        ),
        "sequential_state": re.compile(
            r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\.\Z"
        ),
        "modular_update": re.compile(
            r"Add (\d+) and (\d+), then give the remainder after division by (\d+)\.\Z"
        ),
    }
    _require(family in patterns, "prerequisite_family_mismatch", str(family))
    match = patterns[str(family)].fullmatch(question)
    _require(match is not None, "prerequisite_question_mismatch", question)
    if family == "multiply_subtract":
        start, multiplier, subtractor = map(int, match.groups())
        return start, (("multiply", multiplier), ("subtract", subtractor))
    if family == "base_conversion":
        base = int(match.group(1))
        digits = tuple(int(value) for value in match.group(2))
        return digits[0], (
            ("multiply", base),
            ("add", digits[1]),
            ("multiply", base),
            ("add", digits[2]),
        )
    if family == "sequential_state":
        start, addend, multiplier, subtractor = map(int, match.groups())
        return start, (
            ("add", addend),
            ("multiply", multiplier),
            ("subtract", subtractor),
        )
    start, addend, modulus = map(int, match.groups())
    return start, (("add", addend), ("remainder", modulus))


def _prerequisite_apply(state: int, operation: str, operand: int) -> int:
    if operation == "remainder":
        return state % operand
    return apply_operation(state, operation, operand)


def _prerequisite_prompt(state: int, operation: str, operand: int) -> str:
    if operation == "remainder":
        return f"Problem: Give the remainder after dividing {state} by {operand}.\nWork:"
    return executor_prompt(state, operation, operand)


def _prerequisite_call(record: object, prompt: str, max_new: int) -> Mapping[str, Any]:
    _require(isinstance(record, dict), "prerequisite_call_schema", prompt)
    _require(record.get("prompt") == prompt and record.get("max_new") == max_new, "prerequisite_prompt_mismatch", prompt)
    _require(isinstance(record.get("response"), str), "prerequisite_call_schema", prompt)
    for field in (
        "prompt_token_count",
        "untruncated_prompt_token_count",
        "sampled_token_count",
        "decoded_token_count",
    ):
        value = record.get(field)
        _require(type(value) is int and value >= 0, "prerequisite_token_accounting", field)
    _require(
        record["prompt_token_count"] == record["untruncated_prompt_token_count"] > 0
        and record.get("prompt_truncated") is False,
        "prerequisite_token_accounting",
        "prompt truncation",
    )
    stop = record.get("stop_reason")
    _require(stop in {"eos", "max_new", "context_limit"}, "prerequisite_stop_reason", str(stop))
    expected_sampled = record["decoded_token_count"] + int(stop == "eos")
    _require(record["sampled_token_count"] == expected_sampled, "prerequisite_token_accounting", "sampled")
    _require(record["sampled_token_count"] <= max_new, "prerequisite_decode_cap", prompt)
    if stop == "max_new":
        _require(record["sampled_token_count"] == max_new, "prerequisite_decode_cap", prompt)
    return record


def _full_response_integer(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    segment = HEADER_RE.split(text, maxsplit=1)[0].strip()
    values = EXECUTOR_INTEGER_RE.findall(segment)
    return int(values[-1].replace(",", "")) if values else None


def _prerequisite_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    _require(result.get("board_sha256") == PREREQUISITE_BOARD_SHA256, "prerequisite_hash_mismatch", "board")
    _require(result.get("cases_sha256") == PREREQUISITE_CASES_SHA256, "prerequisite_hash_mismatch", "cases")
    _require(result.get("checkpoint_sha256") == RAW_EXECUTOR_SHA256, "prerequisite_hash_mismatch", "checkpoint")
    _require(result.get("tokenizer_sha256") == TOKENIZER_SHA256, "prerequisite_hash_mismatch", "tokenizer")
    rows = result.get("rows")
    _require(isinstance(rows, list) and len(rows) == 256, "prerequisite_row_count", "rows")
    direct = []
    whole = []
    scheduled = []
    atomic_correct = atomic_total = 0
    by_family: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "count": 0,
            "direct_correct": 0,
            "whole_correct": 0,
            "scheduled_correct": 0,
            "atomic_correct": 0,
            "atomic_total": 0,
        }
    )
    arm_calls: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = _strict_int(row.get("answer"), "prerequisite answer")
        family = row.get("family")
        question = row.get("question")
        start, schedule = _prerequisite_schedule(family, question)
        mathematical = start
        for operation, operand in schedule:
            mathematical = _prerequisite_apply(mathematical, operation, operand)
        _require(mathematical == answer, "prerequisite_answer_mismatch", str(row.get("id")))

        direct_call = _prerequisite_call(
            row.get("direct"),
            f"Question: {question} Return only the final integer.\nAnswer:",
            128,
        )
        arm_calls["direct_qa"].append(direct_call)
        direct_prediction = _full_response_integer(direct_call["response"])
        direct_bit = direct_prediction == answer
        whole_call = _prerequisite_call(
            row.get("whole_problem_work"), f"Problem: {question}\nWork:", 128
        )
        arm_calls["whole_problem_work"].append(whole_call)
        whole_bit = _full_response_integer(whole_call["response"]) == answer

        atomic = row.get("atomic_oracle_state")
        _require(
            isinstance(atomic, list) and len(atomic) == len(schedule),
            "prerequisite_transcript_mismatch",
            "atomic",
        )
        true_state = start
        family_atomic = 0
        for (operation, operand), step in zip(schedule, atomic):
            expected_state = _prerequisite_apply(true_state, operation, operand)
            call = _prerequisite_call(
                step,
                _prerequisite_prompt(true_state, operation, operand),
                48,
            )
            arm_calls["atomic_oracle_state"].append(call)
            prediction = parse_executor_result(call["response"])
            family_atomic += int(prediction == expected_state)
            atomic_correct += int(prediction == expected_state)
            atomic_total += 1
            true_state = expected_state

        scheduled_steps = row.get("source_scheduled", {}).get("steps")
        _require(isinstance(scheduled_steps, list) and scheduled_steps, "prerequisite_transcript_mismatch", "scheduled")
        _require(len(scheduled_steps) <= len(schedule), "prerequisite_transcript_mismatch", "scheduled calls")
        model_state = start
        valid = True
        for index, step in enumerate(scheduled_steps):
            operation, operand = schedule[index]
            call = _prerequisite_call(
                step,
                _prerequisite_prompt(model_state, operation, operand),
                48,
            )
            arm_calls["source_scheduled"].append(call)
            parsed = parse_executor_result(call["response"])
            if parsed is None:
                _require(index == len(scheduled_steps) - 1, "prerequisite_continued_after_failure", str(row.get("id")))
                model_state = None
                valid = False
                break
            model_state = parsed
        if valid:
            _require(len(scheduled_steps) == len(schedule), "prerequisite_missing_call", str(row.get("id")))
        final = model_state
        if final is None:
            valid = False
        scheduled_bit = valid and final == answer
        _require(isinstance(family, str), "prerequisite_transcript_mismatch", "family")
        by_family[family]["count"] += 1
        by_family[family]["direct_correct"] += int(direct_bit)
        by_family[family]["whole_correct"] += int(whole_bit)
        by_family[family]["scheduled_correct"] += int(scheduled_bit)
        by_family[family]["atomic_correct"] += family_atomic
        by_family[family]["atomic_total"] += len(schedule)
        direct.append(direct_bit)
        whole.append(whole_bit)
        scheduled.append(scheduled_bit)
    mc = exact_mcnemar(
        sum(s and not d for s, d in zip(scheduled, direct)),
        sum(d and not s for s, d in zip(scheduled, direct)),
    )
    summary = {
        "case_count": 256,
        "transition_count": atomic_total,
        "direct_correct": sum(direct),
        "whole_correct": sum(whole),
        "scheduled_correct": sum(scheduled),
        "atomic_correct": atomic_correct,
        "atomic_total": atomic_total,
        "scheduler_only_correct": mc["treatment_only"],
        "direct_only_correct": mc["sham_only"],
        "mcnemar_exact_p": mc["p"],
        "by_family": dict(by_family),
    }
    gates = {
        "scheduled_absolute": 100 * summary["scheduled_correct"] >= 35 * 256,
        "scheduled_advantage": 100 * (summary["scheduled_correct"] - summary["direct_correct"]) >= 10 * 256,
        "paired_significance": mc["numerator"] * 100 < mc["denominator"],
        "family_nonregression": all(item["scheduled_correct"] >= item["direct_correct"] for item in by_family.values()),
        "sequential_absolute": 100 * by_family["sequential_state"]["scheduled_correct"] >= 70 * by_family["sequential_state"]["count"],
        "atomic_ceiling": 100 * atomic_correct >= 70 * atomic_total,
    }
    reported = result.get("summary")
    _require(reported == summary, "prerequisite_summary_mismatch", "full summary")
    _require(result.get("gates") == gates, "prerequisite_gate_mismatch", "score gates")
    ledger = result.get("resource_ledger")
    _require(isinstance(ledger, dict), "prerequisite_ledger_mismatch", "ledger")
    reported_arms = ledger.get("by_arm")
    _require(isinstance(reported_arms, dict), "prerequisite_ledger_mismatch", "by_arm")
    for arm, calls in arm_calls.items():
        reported_arm = reported_arms.get(arm)
        _require(
            isinstance(reported_arm, dict)
            and reported_arm.get("model_calls") == len(calls)
            and reported_arm.get("prompt_token_count")
            == sum(call["prompt_token_count"] for call in calls)
            and reported_arm.get("sampled_token_count")
            == sum(call["sampled_token_count"] for call in calls)
            and reported_arm.get("decoded_token_count")
            == sum(call["decoded_token_count"] for call in calls),
            "prerequisite_ledger_mismatch",
            arm,
        )
    integrity = result.get("integrity_gates")
    _require(
        isinstance(integrity, dict)
        and integrity
        and all(value is True for value in integrity.values()),
        "prerequisite_integrity_gate_failed",
        str(integrity),
    )
    _require(all(gates.values()), "prerequisite_gate_failed", str(gates))
    _require(result.get("advance_to_internalization") is True, "prerequisite_report_disagrees", "advance")
    return {"summary": summary, "gates": gates, "passed": True}


def score_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest_hash = None
    try:
        raw, manifest_hash = _read_regular(manifest_path)
        assert raw is not None
        manifest = _parse_json(raw, "manifest")
        required = {
            "schema", "frozen", "preregistration", "protocol", "evaluator",
            "board", "tokenizer", "raw_executor_checkpoint",
            "treatment_data", "sham_data", "training_manifest", "admission_audit",
            "prerequisite_confirmation", "runs",
        }
        _require(set(manifest) == required, "manifest_schema_mismatch", "keys")
        _require(manifest["schema"] == MANIFEST_SCHEMA and manifest["frozen"] is True, "manifest_schema_mismatch", MANIFEST_SCHEMA)
        directory = manifest_path.parent
        _, preregistration_ref, _ = _reference(
            manifest["preregistration"],
            "preregistration",
            directory,
            json_artifact=False,
        )
        _, protocol_ref, _ = _reference(
            manifest["protocol"], "protocol", directory, json_artifact=False
        )
        _require(
            protocol_ref["sha256"] == EXPECTED_PROTOCOL_SHA256
            and Path(ADMITTED_PROTOCOL.__file__).resolve() == PROTOCOL_PATH,
            "protocol_implementation_mismatch",
            protocol_ref["sha256"],
        )
        _, evaluator_ref, _ = _reference(
            manifest["evaluator"], "evaluator", directory, json_artifact=False
        )
        live_evaluator = Path(__file__).with_name("eval_residual_packet_v1.py")
        _require(
            evaluator_ref["sha256"] == sha256_file(live_evaluator),
            "evaluator_implementation_mismatch",
            evaluator_ref["sha256"],
        )
        board, board_ref, _ = _reference(manifest["board"], "board", directory, json_artifact=True)
        assert board is not None
        _require(
            board_ref["sha256"] == EXPECTED_BOARD_SHA256,
            "board_artifact_hash_mismatch",
            board_ref["sha256"],
        )
        rows, rows_hash = audit_board(board)
        _, tokenizer_ref, tokenizer_path = _reference(manifest["tokenizer"], "tokenizer", directory, json_artifact=False)
        _require(tokenizer_ref["sha256"] == TOKENIZER_SHA256, "tokenizer_hash_mismatch", tokenizer_ref["sha256"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _, executor_ref, _ = _reference(manifest["raw_executor_checkpoint"], "raw executor", directory, json_artifact=False)
        _require(executor_ref["sha256"] == RAW_EXECUTOR_SHA256, "executor_hash_mismatch", executor_ref["sha256"])
        _, treatment_data_ref, _ = _reference(manifest["treatment_data"], "treatment data", directory, json_artifact=False)
        _, sham_data_ref, _ = _reference(manifest["sham_data"], "sham data", directory, json_artifact=False)
        training, training_ref, _ = _reference(manifest["training_manifest"], "training manifest", directory, json_artifact=True)
        audit, audit_ref, _ = _reference(manifest["admission_audit"], "admission audit", directory, json_artifact=True)
        prerequisite, prerequisite_ref, _ = _reference(manifest["prerequisite_confirmation"], "prerequisite confirmation", directory, json_artifact=True)
        assert training is not None and audit is not None and prerequisite is not None
        _require(training.get("schema") == TRAINING_MANIFEST_SCHEMA, "training_manifest_schema", TRAINING_MANIFEST_SCHEMA)
        artifacts = training.get("artifacts", {})
        _require(artifacts.get("board_sha256") == board_ref["sha256"] and artifacts.get("board_rows_sha256") == rows_hash, "training_board_binding", "training manifest")
        _require(artifacts.get("treatment_sha256") == treatment_data_ref["sha256"] and artifacts.get("sham_sha256") == sham_data_ref["sha256"], "training_data_binding", "training manifest")
        _require(training.get("tokenizer_sha256") == tokenizer_ref["sha256"], "training_tokenizer_binding", "training manifest")
        serialized_audit = json.dumps(audit, sort_keys=True)
        for digest in (board_ref["sha256"], tokenizer_ref["sha256"], treatment_data_ref["sha256"], sham_data_ref["sha256"]):
            _require(digest in serialized_audit, "admission_audit_binding", digest)
        _require(
            audit.get("schema") == "residual_packet_admission_audit_v1"
            and audit.get("admitted") is True
            and audit.get("failures") == [],
            "admission_audit_failed",
            "admission audit is not passing",
        )
        audit_hashes = audit.get("artifact_sha256", {})
        _require(
            audit_hashes.get("board") == board_ref["sha256"]
            and audit_hashes.get("tokenizer") == tokenizer_ref["sha256"]
            and audit_hashes.get("treatment") == treatment_data_ref["sha256"]
            and audit_hashes.get("sham") == sham_data_ref["sha256"]
            and audit_hashes.get("manifest") == training_ref["sha256"],
            "admission_audit_binding",
            "artifact hashes",
        )
        prerequisite_score = _prerequisite_summary(prerequisite)
        runs = manifest["runs"]
        _require(isinstance(runs, list) and len(runs) == 2, "run_grid_mismatch", "runs")
        scores = []
        seen = set()
        verified_runs = []
        for run in runs:
            _require(
                isinstance(run, dict)
                and set(run)
                == {
                    "seed",
                    "treatment_checkpoint",
                    "sham_checkpoint",
                    "training_resources",
                    "transcript",
                },
                "run_schema_mismatch",
                str(run),
            )
            seed = _strict_int(run["seed"], "run seed")
            _require(seed in FIT_SEEDS and seed not in seen, "run_grid_mismatch", str(seed))
            seen.add(seed)
            _, treatment_checkpoint, _ = _reference(run["treatment_checkpoint"], f"treatment checkpoint {seed}", directory, json_artifact=False)
            _, sham_checkpoint, _ = _reference(run["sham_checkpoint"], f"sham checkpoint {seed}", directory, json_artifact=False)
            training_resources, training_resources_ref, _ = _reference(
                run["training_resources"],
                f"training resources {seed}",
                directory,
                json_artifact=True,
            )
            transcript, transcript_ref, _ = _reference(run["transcript"], f"transcript {seed}", directory, json_artifact=True)
            assert training_resources is not None and transcript is not None
            resource_counts = training_resource_counts(training_resources, seed)
            expected_hashes = {
                "board": board_ref["sha256"], "tokenizer": tokenizer_ref["sha256"],
                "raw_260k_executor": executor_ref["sha256"],
                "treatment_checkpoint": treatment_checkpoint["sha256"],
                "sham_checkpoint": sham_checkpoint["sha256"],
                "training_resources": training_resources_ref["sha256"],
                "protocol": protocol_ref["sha256"],
                "evaluator": evaluator_ref["sha256"],
            }
            score = score_transcript(
                transcript, rows, tokenizer, seed, expected_hashes,
                {"treatment": treatment_checkpoint["sha256"], "sham": sham_checkpoint["sha256"], "raw_260k_executor": executor_ref["sha256"]},
                resource_counts,
            )
            scores.append(score)
            verified_runs.append({
                "seed": seed, "treatment_checkpoint": treatment_checkpoint,
                "sham_checkpoint": sham_checkpoint,
                "training_resources": training_resources_ref,
                "transcript": transcript_ref,
            })
        _require(seen == set(FIT_SEEDS), "run_grid_mismatch", str(seen))
        passed = prerequisite_score["passed"] and all(score["passed"] for score in scores)
        return {
            "schema": DECISION_SCHEMA, "decision": "GO" if passed else "NO_GO", "go": passed,
            "reasons": [] if passed else ["one_or_more_locked_gates_failed"],
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "verification": {
                "board": board_ref, "tokenizer": tokenizer_ref, "raw_executor": executor_ref,
                "preregistration": preregistration_ref,
                "protocol": protocol_ref,
                "evaluator": evaluator_ref,
                "treatment_data": treatment_data_ref, "sham_data": sham_data_ref,
                "training_manifest": training_ref, "admission_audit": audit_ref,
                "prerequisite_confirmation": prerequisite_ref, "runs": verified_runs,
                "admitted_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            },
            "prerequisite": prerequisite_score, "runs": scores,
        }
    except EvidenceError as error:
        return {
            "schema": DECISION_SCHEMA, "decision": "NO_GO", "go": False,
            "reasons": ["evidence_contract_failed", error.code], "failure_detail": error.detail,
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        }
    except Exception as error:
        return {
            "schema": DECISION_SCHEMA, "decision": "NO_GO", "go": False,
            "reasons": ["scoring_execution_failed"],
            "failure_detail": f"{type(error).__name__}: {error}",
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        }


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable decision write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    raise SystemExit(
        "RSP-C1 is permanently closed; preserve this module as audit evidence only"
    )
    args = build_parser().parse_args()
    decision = score_manifest(args.manifest)
    digest = write_immutable_json(args.out, decision)
    print(json.dumps({"decision": decision["decision"], "out": args.out, "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
