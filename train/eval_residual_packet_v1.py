#!/usr/bin/env python3
"""Acquire raw transcripts for the frozen RSP-C1 evaluation.

This module deliberately does not score its output.  The source-bearing compiler
wrapper and the source-blind interpreter are separate functions, every model call
creates a fresh KV cache, and the emitted JSON contains no booleans or aggregate
success metrics.  ``score_residual_packet_v1.py`` independently reconstructs all
derived values from the frozen board and these raw call records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tokenizers import Tokenizer

try:
    from model import GPT, GPTConfig
except ModuleNotFoundError:
    from train.model import GPT, GPTConfig

TRANSCRIPT_SCHEMA = "residual_packet_v1_raw_transcript"
BOARD_SCHEMA = "residual_packet_board_v1"
BOARD_SEED = 2026071503
UPDATER_SEED = 2026071505
FIT_SEEDS = (2026071511, 2026071512)
STRATA = ("renderer_ood", "value_ood", "order_ood", "length_ood")
MAX_TRANSITIONS = 5
MAX_NEW_CONTROLLER = 80
MAX_NEW_EXECUTOR = 48
SWAP_COUNT = 64
MODEL_NAMES = ("treatment", "sham", "raw_260k_executor")
EXPECTED_BOARD_ROWS_SHA256 = (
    "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e"
)
EXPECTED_BOARD_SHA256 = (
    "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2"
)

INTEGER_PATTERN = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)"
POSITIVE_PATTERN = r"[1-9][0-9]*"
OPERATION_RE = re.compile(
    rf"(?P<operation>add|multiply|subtract) (?P<operand>{POSITIVE_PATTERN})\Z"
)
PACKET_RE = re.compile(
    rf"State: (?P<state>{INTEGER_PATTERN})\n"
    rf"Plan: (?P<plan>(?:add|multiply|subtract) {POSITIVE_PATTERN}"
    rf"(?:; (?:add|multiply|subtract) {POSITIVE_PATTERN})*)\Z"
)
ANSWER_RE = re.compile(rf"Answer: (?P<answer>{INTEGER_PATTERN})\Z")
EXECUTOR_INTEGER_RE = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?![A-Za-z0-9_,]|\.\d)"
)


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
        "_rsp_c1_protocol_e011e838", PROTOCOL_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load residual packet protocol at {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != PROTOCOL_PATH:
        raise ImportError("residual packet protocol loaded from an unexpected path")
    return module


PROTOCOL = _load_exact_protocol()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _without_surrounding_whitespace(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    return text.strip(" \t\n\r\f\v")


def parse_operation(text: object) -> tuple[str, int] | None:
    if not isinstance(text, str):
        return None
    match = OPERATION_RE.fullmatch(text)
    if match is None:
        return None
    return match.group("operation"), int(match.group("operand"))


def parse_packet(text: object) -> tuple[int, tuple[tuple[str, int], ...]] | None:
    """Parse only the locked two-line packet grammar."""

    stripped = _without_surrounding_whitespace(text)
    if stripped is None:
        return None
    match = PACKET_RE.fullmatch(stripped)
    if match is None:
        return None
    operations = tuple(parse_operation(item) for item in match.group("plan").split("; "))
    if any(item is None for item in operations):
        return None
    return int(match.group("state")), tuple(operations)  # type: ignore[arg-type]


def parse_answer(text: object) -> int | None:
    stripped = _without_surrounding_whitespace(text)
    if stripped is None:
        return None
    match = ANSWER_RE.fullmatch(stripped)
    return int(match.group("answer")) if match is not None else None


def format_operation(operation: str, operand: int) -> str:
    rendered = f"{operation} {operand}"
    if parse_operation(rendered) != (operation, operand):
        raise ValueError("operation is outside the frozen packet grammar")
    return rendered


def format_packet(state: int, operations: Sequence[tuple[str, int]]) -> str:
    if type(state) is not int or not operations:
        raise ValueError("a packet requires an integer state and nonempty plan")
    rendered = "State: {}\nPlan: {}".format(
        state, "; ".join(format_operation(*item) for item in operations)
    )
    if parse_packet(rendered) != (state, tuple(operations)):
        raise AssertionError("packet formatter and parser disagree")
    return rendered


def format_answer(value: int) -> str:
    if type(value) is not int:
        raise ValueError("answer must be an integer")
    rendered = f"Answer: {value}"
    if parse_answer(rendered) != value:
        raise AssertionError("answer formatter and parser disagree")
    return rendered


def _check_protocol_renderer(name: str, arguments: tuple[Any, ...], rendered: str) -> str:
    function = getattr(PROTOCOL, name, None)
    if function is not None:
        shared = function(*arguments)
        if shared != rendered:
            raise RuntimeError(f"shared protocol {name} disagrees with frozen renderer")
    return rendered


def compiler_prompt(source: str) -> str:
    if not isinstance(source, str) or not source:
        raise ValueError("compiler source must be nonempty text")
    rendered = f"Problem: {source}\nCompile only the execution packet.\nPacket:"
    return _check_protocol_renderer("compiler_prompt", (source,), rendered)


def updater_prompt(packet: str, observed: int) -> str:
    if parse_packet(packet) is None or type(observed) is not int:
        raise ValueError("updater input is not a canonical source-free state")
    rendered = f"Packet:\n{packet}\nObserved result: {observed}\nNext packet:"
    return _check_protocol_renderer("update_prompt", (packet, observed), rendered)


def executor_prompt(state: int, operation: str, operand: int) -> str:
    if type(state) is not int or parse_operation(f"{operation} {operand}") is None:
        raise ValueError("executor input is outside the frozen grammar")
    if operation == "add":
        clause = f"Compute {state} plus {operand}."
    elif operation == "multiply":
        clause = f"Compute {state} times {operand}."
    else:
        clause = f"Compute {state} minus {operand}."
    rendered = f"Problem: {clause}\nWork:"
    return _check_protocol_renderer(
        "format_atomic_prompt", (state, operation, operand), rendered
    )


def first_nonempty_line(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def parse_executor_result(text: object) -> int | None:
    values = EXECUTOR_INTEGER_RE.findall(first_nonempty_line(text))
    return int(values[-1].replace(",", "")) if values else None


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(token_ids))


def _autocast(device: str):
    return torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
    )


@torch.no_grad()
def greedy_completion(
    model: Any, tokenizer: Any, prompt: str, device: str, max_new: int
) -> dict[str, Any]:
    """Greedy completion whose cache is born and destroyed inside one call."""

    if type(max_new) is not int or max_new <= 0:
        raise ValueError("max_new must be a positive integer")
    prompt_ids = list(tokenizer.encode(prompt).ids)
    context_limit = int(model.cfg.seq_len)
    if not prompt_ids or len(prompt_ids) >= context_limit:
        raise ValueError("prompt is empty or does not fit the model context")

    with _autocast(device):
        logits, cache = model(
            torch.tensor([prompt_ids], device=device), return_cache=True, pos=0
        )
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    sampled_ids: list[int] = []
    decoded_ids: list[int] = []
    position = len(prompt_ids)
    stop_reason = "max_new"
    try:
        for _ in range(max_new):
            token = int(logits[:, -1].argmax(dim=-1).item())
            sampled_ids.append(token)
            if eos_id is not None and token == eos_id:
                stop_reason = "eos"
                break
            decoded_ids.append(token)
            position += 1
            if position >= context_limit:
                stop_reason = "context_limit"
                break
            with _autocast(device):
                logits, cache = model(
                    torch.tensor([[token]], device=device),
                    cache=cache,
                    pos=position - 1,
                    return_cache=True,
                )
    finally:
        del cache

    return {
        "response": _decode_tokens(tokenizer, decoded_ids),
        "prompt_token_count": len(prompt_ids),
        "sampled_token_ids": sampled_ids,
        "sampled_token_count": len(sampled_ids),
        "decoded_token_ids": decoded_ids,
        "decoded_token_count": len(decoded_ids),
        "stop_reason": stop_reason,
    }


class ModelEndpoint:
    """One immutable checkpoint/tokenizer pair; calls never retain decode state."""

    def __init__(
        self,
        name: str,
        checkpoint_path: str | Path,
        model: Any,
        tokenizer: Any,
        device: str,
        checkpoint_step: int | None,
    ) -> None:
        if name not in MODEL_NAMES:
            raise ValueError("unregistered model endpoint")
        self.name = name
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.checkpoint_step = checkpoint_step
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def complete(self, prompt: str, max_new: int) -> dict[str, Any]:
        return greedy_completion(self.model, self.tokenizer, prompt, self.device, max_new)


class CallRecorder:
    """Append-only call order and resource accounting."""

    def __init__(self) -> None:
        self.next_call_index = 0
        self.records: list[dict[str, Any]] = []
        self.unissued: defaultdict[tuple[str, str], int] = defaultdict(int)

    def call(self, endpoint: Any, arm: str, prompt: str, max_new: int) -> dict[str, Any]:
        complete = getattr(endpoint, "complete", endpoint)
        raw = complete(prompt, max_new)
        required = {
            "response",
            "prompt_token_count",
            "sampled_token_ids",
            "sampled_token_count",
            "decoded_token_ids",
            "decoded_token_count",
            "stop_reason",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("model endpoint returned an incomplete raw call record")
        name = getattr(endpoint, "name", None)
        if name not in MODEL_NAMES:
            raise ValueError("model endpoint has no frozen identity")
        record = {
            "call_index": self.next_call_index,
            "model": name,
            "arm": arm,
            "prompt": prompt,
            "max_new": max_new,
            **dict(raw),
        }
        self.next_call_index += 1
        self.records.append(record)
        return record

    def note_unissued(self, model: str, arm: str, count: int) -> None:
        if model not in MODEL_NAMES or type(count) is not int or count < 0:
            raise ValueError("invalid unissued-call accounting")
        self.unissued[(model, arm)] += count

    def runtime_sink(self) -> "RuntimeCallSink":
        return RuntimeCallSink(self.call)

    def ledger(self, training_resources: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
        by_model: dict[str, Any] = {}
        for model in MODEL_NAMES:
            model_records = [record for record in self.records if record["model"] == model]
            arms = sorted(
                {record["arm"] for record in model_records}
                | {arm for (name, arm) in self.unissued if name == model}
            )
            by_arm = {}
            for arm in arms:
                selected = [record for record in model_records if record["arm"] == arm]
                by_arm[arm] = {
                    "model_calls": len(selected),
                    "prompt_tokens": sum(record["prompt_token_count"] for record in selected),
                    "sampled_tokens": sum(record["sampled_token_count"] for record in selected),
                    "decoded_tokens": sum(record["decoded_token_count"] for record in selected),
                    "supervised_completion_tokens": 0,
                    "packed_forward_token_positions": 0,
                    "calls_not_issued_after_parse_failure": self.unissued[(model, arm)],
                    "retries": 0,
                    "repairs": 0,
                    "searches": 0,
                    "verifier_feedback_calls": 0,
                }
            training = training_resources.get(model, {})
            by_arm["training"] = {
                "model_calls": 0,
                "prompt_tokens": 0,
                "sampled_tokens": 0,
                "decoded_tokens": 0,
                "supervised_completion_tokens": int(
                    training.get("supervised_completion_tokens", 0)
                ),
                "packed_forward_token_positions": int(
                    training.get("packed_forward_token_positions", 0)
                ),
                "calls_not_issued_after_parse_failure": 0,
                "retries": 0,
                "repairs": 0,
                "searches": 0,
                "verifier_feedback_calls": 0,
            }
            by_model[model] = {"by_arm": by_arm}
        return {"by_model": by_model}


class RuntimeCallSink:
    """Write-only facade: a blind runtime cannot inspect earlier transcripts."""

    __slots__ = ("__append_call",)

    def __init__(self, append_call: Any) -> None:
        self.__append_call = append_call

    def call(self, endpoint: Any, arm: str, prompt: str, max_new: int) -> dict[str, Any]:
        return self.__append_call(endpoint, arm, prompt, max_new)


def normalize_operations(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("program must be a nonempty list")
    result = []
    for item in value:
        if isinstance(item, str):
            parsed = parse_operation(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parsed = parse_operation(f"{item[0]} {item[1]}")
        elif isinstance(item, Mapping):
            operation = item.get("operation", item.get("op"))
            operand = item.get("operand", item.get("value"))
            parsed = parse_operation(f"{operation} {operand}")
        else:
            parsed = None
        if parsed is None:
            raise ValueError("program contains a noncanonical operation")
        result.append(parsed)
    return tuple(result)


def row_fields(row: Mapping[str, Any]) -> tuple[str, str, str, int, tuple[tuple[str, int], ...]]:
    identifier = row.get("id")
    stratum = row.get("stratum")
    source = row.get("source", row.get("question"))
    initial = row.get("initial_state", row.get("state"))
    operations = row.get("program", row.get("operations", row.get("schedule")))
    if not all(isinstance(value, str) and value for value in (identifier, stratum, source)):
        raise ValueError("board row lacks id, stratum, or source")
    if type(initial) is not int:
        raise ValueError("board row initial state must be an integer")
    return identifier, stratum, source, initial, normalize_operations(operations)


def board_rows(board: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = board.get("rows", board.get("cases"))
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("board must contain a row list")
    return list(rows)


def load_board(path: str | Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise PermissionError("frozen RSP-C1 board must be a read-only regular file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_BOARD_SHA256:
        raise ValueError("RSP-C1 board artifact hash does not match the frozen digest")
    board = json.loads(raw)
    if not isinstance(board, dict):
        raise ValueError("board must be a JSON object")
    if (
        board.get("schema") != BOARD_SCHEMA
        or board.get("seed") != BOARD_SEED
        or board.get("case_count") != 256
        or board.get("per_stratum") != 64
        or tuple(board.get("stratum_order", ())) != STRATA
    ):
        raise ValueError("board metadata differs from the frozen RSP-C1 contract")
    rows = board_rows(board)
    if len(rows) != 256:
        raise ValueError("RSP-C1 board must contain exactly 256 cases")
    seen: set[str] = set()
    strata = defaultdict(int)
    for row in rows:
        identifier, stratum, rendered_source, initial, operations = row_fields(row)
        if identifier in seen or stratum not in STRATA or len(operations) > MAX_TRANSITIONS:
            raise ValueError("board identity, stratum, or length contract failed")
        if row.get("packet") != PROTOCOL.canonical_packet(initial, operations):
            raise ValueError("board packet differs from the admitted protocol")
        if row.get("source") != PROTOCOL.render_source(
            initial, operations, row.get("template_id")
        ) or row.get("source") != rendered_source:
            raise ValueError("board source differs from the admitted protocol")
        trajectory = tuple(PROTOCOL.trajectory(initial, operations))
        if row.get("trajectory") != list(trajectory) or row.get("answer") != trajectory[-1]:
            raise ValueError("board arithmetic replay failed")
        seen.add(identifier)
        strata[stratum] += 1
    if dict(strata) != {stratum: 64 for stratum in STRATA}:
        raise ValueError("board strata are not exactly balanced")
    rows_digest = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    if rows_digest != EXPECTED_BOARD_ROWS_SHA256 or board.get("rows_sha256") != rows_digest:
        raise ValueError("board canonical-row hash does not match the frozen digest")
    return board, rows


def _strict_updater_transition(
    prior: tuple[int, tuple[tuple[str, int], ...]], observed: int, response: str
) -> tuple[str, tuple[int, tuple[tuple[str, int], ...]] | None]:
    _, operations = prior
    answer = parse_answer(response)
    packet = parse_packet(response)
    if len(operations) == 1 and answer == observed:
        return "answer", None
    expected = (observed, operations[1:])
    if len(operations) > 1 and packet == expected:
        return "packet", packet
    return "invalid", None


def run_source_blind_loop(
    initial_packet_response: str,
    controller: Any,
    executor: Any,
    recorder: Any,
    arm: str,
) -> dict[str, Any]:
    """Run with only a model-authored packet; no source or gold enters here."""

    packet = parse_packet(initial_packet_response)
    if packet is None:
        return {"termination": "initial_packet_invalid", "steps": []}
    canonical_packet = format_packet(*packet)
    steps = []
    for _ in range(MAX_TRANSITIONS):
        state, operations = packet
        operation, operand = operations[0]
        executor_call = recorder.call(
            executor,
            arm,
            executor_prompt(state, operation, operand),
            MAX_NEW_EXECUTOR,
        )
        observed = parse_executor_result(executor_call["response"])
        step: dict[str, Any] = {"executor": executor_call}
        steps.append(step)
        if observed is None:
            return {"termination": "executor_result_invalid", "steps": steps}

        update_call = recorder.call(
            controller,
            arm,
            updater_prompt(canonical_packet, observed),
            MAX_NEW_CONTROLLER,
        )
        step["updater"] = update_call
        disposition, next_packet = _strict_updater_transition(
            packet, observed, update_call["response"]
        )
        if disposition == "answer":
            return {"termination": "answer", "steps": steps}
        if disposition == "invalid" or next_packet is None:
            return {"termination": "updater_output_invalid", "steps": steps}
        packet = next_packet
        canonical_packet = format_packet(*packet)
    return {"termination": "transition_limit", "steps": steps}


def _account_blind_unissued(
    recorder: CallRecorder,
    controller_name: str,
    arm: str,
    expected_transitions: int,
    runtime: Mapping[str, Any],
) -> None:
    steps = runtime["steps"]
    executor_calls = len(steps)
    updater_calls = sum("updater" in step for step in steps)
    recorder.note_unissued(
        "raw_260k_executor", arm, expected_transitions - executor_calls
    )
    recorder.note_unissued(controller_name, arm, expected_transitions - updater_calls)


def run_compiled_case(
    source: str,
    expected_transitions: int,
    controller: Any,
    executor: Any,
    recorder: CallRecorder,
    arm: str = "strict_closed_loop",
) -> dict[str, Any]:
    compiler_call = recorder.call(
        controller, arm, compiler_prompt(source), MAX_NEW_CONTROLLER
    )
    packet_text = compiler_call["response"]
    source = ""
    del source
    runtime = run_source_blind_loop(
        packet_text, controller, executor, recorder.runtime_sink(), arm
    )
    _account_blind_unissued(
        recorder, controller.name, arm, expected_transitions, runtime
    )
    return {"compiler": compiler_call, "runtime": runtime}


def run_oracle_packet_case(
    initial_state: int,
    operations: Sequence[tuple[str, int]],
    controller: Any,
    executor: Any,
    recorder: CallRecorder,
) -> dict[str, Any]:
    runtime = run_source_blind_loop(
        format_packet(initial_state, operations),
        controller,
        executor,
        recorder.runtime_sink(),
        "oracle_packet_loop",
    )
    _account_blind_unissued(
        recorder,
        controller.name,
        "oracle_packet_loop",
        len(operations),
        runtime,
    )
    return runtime


def run_external_scheduler(
    initial_state: int,
    operations: Sequence[tuple[str, int]],
    executor: Any,
    recorder: CallRecorder,
) -> dict[str, Any]:
    state = initial_state
    steps = []
    for operation, operand in operations:
        call = recorder.call(
            executor,
            "external_scheduler",
            executor_prompt(state, operation, operand),
            MAX_NEW_EXECUTOR,
        )
        steps.append(call)
        parsed = parse_executor_result(call["response"])
        if parsed is None:
            recorder.note_unissued(
                "raw_260k_executor",
                "external_scheduler",
                len(operations) - len(steps),
            )
            return {"termination": "executor_result_invalid", "steps": steps}
        state = parsed
    return {"termination": "complete", "steps": steps}


def _apply(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "multiply":
        return value * operand
    if operation == "subtract":
        return value - operand
    raise ValueError(operation)


def _prf_integer(label: str, lower: int = 10, span: int = 990) -> int:
    digest = hashlib.sha256(f"{UPDATER_SEED}:{label}".encode("ascii")).digest()
    return lower + int.from_bytes(digest[:8], "big") % span


def build_teacher_forced_cases(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    answers = {
        int(row["answer"])
        for row in rows
        if type(row.get("answer")) is int
    }
    result = []
    for row in rows:
        identifier, _, _, _, operations = row_fields(row)
        for index in range(len(operations)):
            state_nonce = 0
            while True:
                state = _prf_integer(f"{identifier}:{index}:state:{state_nonce}")
                if state not in answers:
                    break
                state_nonce += 1
            observed_nonce = 0
            while True:
                observed = _prf_integer(
                    f"{identifier}:{index}:observed:{observed_nonce}"
                )
                operation, operand = operations[index]
                if observed not in answers and observed != _apply(state, operation, operand):
                    break
                observed_nonce += 1
            result.append(
                {
                    "id": identifier,
                    "step_index": index,
                    "packet": format_packet(state, operations[index:]),
                    "observed": observed,
                }
            )
    return result


def build_packet_swaps(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    selected = []
    by_stratum = {stratum: [] for stratum in STRATA}
    for row in rows:
        by_stratum[row_fields(row)[1]].append(row)
    for stratum in STRATA:
        candidates = by_stratum[stratum][:16]
        if len(candidates) != 16:
            raise ValueError("packet swaps require 16 cases per stratum")
        for index, original in enumerate(candidates):
            donor = candidates[(index + 1) % len(candidates)]
            selected.append((original, donor))
    if len(selected) != SWAP_COUNT:
        raise AssertionError("wrong packet-swap count")
    return selected


def run_packet_swap(
    original_source: str,
    donor_state: int,
    donor_operations: Sequence[tuple[str, int]],
    controller: Any,
    executor: Any,
    recorder: CallRecorder,
) -> dict[str, Any]:
    compiler_call = recorder.call(
        controller,
        "packet_swap",
        compiler_prompt(original_source),
        MAX_NEW_CONTROLLER,
    )
    original_source = ""
    del original_source
    intervened_packet = format_packet(donor_state, donor_operations)
    runtime = run_source_blind_loop(
        intervened_packet,
        controller,
        executor,
        recorder.runtime_sink(),
        "packet_swap",
    )
    _account_blind_unissued(
        recorder, controller.name, "packet_swap", len(donor_operations), runtime
    )
    return {
        "compiler": compiler_call,
        "intervened_packet": intervened_packet,
        "runtime": runtime,
    }


def acquire_transcript(
    board: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    treatment: Any,
    sham: Any,
    executor: Any,
    seed: int,
    input_hashes: Mapping[str, str],
    training_resources: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    if seed not in FIT_SEEDS:
        raise ValueError("evaluation seed is not one of the two frozen fit seeds")
    recorder = CallRecorder()
    external = []
    for row in rows:
        identifier, _, _, initial, operations = row_fields(row)
        external.append(
            {
                "id": identifier,
                "runtime": run_external_scheduler(initial, operations, executor, recorder),
            }
        )

    controllers = {}
    teacher_cases = build_teacher_forced_cases(rows)
    swaps = build_packet_swaps(rows)
    for controller in (treatment, sham):
        strict = []
        oracle = []
        for row in rows:
            identifier, _, source, initial, operations = row_fields(row)
            strict.append(
                {
                    "id": identifier,
                    **run_compiled_case(
                        source,
                        len(operations),
                        controller,
                        executor,
                        recorder,
                    ),
                }
            )
            oracle.append(
                {
                    "id": identifier,
                    "runtime": run_oracle_packet_case(
                        initial, operations, controller, executor, recorder
                    ),
                }
            )

        teacher = []
        for case in teacher_cases:
            call = recorder.call(
                controller,
                "teacher_forced_updater",
                updater_prompt(case["packet"], case["observed"]),
                MAX_NEW_CONTROLLER,
            )
            teacher.append({**case, "call": call})

        packet_swaps = []
        for original, donor in swaps:
            original_id, _, source, _, _ = row_fields(original)
            donor_id, _, _, donor_state, donor_operations = row_fields(donor)
            packet_swaps.append(
                {
                    "original_id": original_id,
                    "donor_id": donor_id,
                    **run_packet_swap(
                        source,
                        donor_state,
                        donor_operations,
                        controller,
                        executor,
                        recorder,
                    ),
                }
            )
        controllers[controller.name] = {
            "strict_closed_loop": strict,
            "oracle_packet_loop": oracle,
            "teacher_forced_updater": teacher,
            "packet_swaps": packet_swaps,
        }

    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "seed": seed,
        "protocol_module": PROTOCOL.__name__,
        "input_hashes": dict(input_hashes),
        "decode_caps": {
            "controller": MAX_NEW_CONTROLLER,
            "executor": MAX_NEW_EXECUTOR,
            "maximum_transitions": MAX_TRANSITIONS,
        },
        "models": {
            endpoint.name: {
                "checkpoint_path": endpoint.checkpoint_path,
                "checkpoint_sha256": endpoint.checkpoint_sha256,
                "checkpoint_step": endpoint.checkpoint_step,
            }
            for endpoint in (treatment, sham, executor)
        },
        "external_scheduler": external,
        "controllers": controllers,
        "resource_ledger": recorder.ledger(training_resources or {}),
        "call_count": recorder.next_call_index,
    }
    assert_no_booleans(transcript)
    return transcript


def assert_no_booleans(value: object, path: str = "$") -> None:
    if isinstance(value, bool):
        raise ValueError(f"raw transcript contains forbidden boolean at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_booleans(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_booleans(item, f"{path}[{index}]")


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    assert_no_booleans(value)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable transcript write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if destination.stat().st_mode & 0o222:
        raise PermissionError("raw transcript remained writable")
    return hashlib.sha256(payload).hexdigest()


def load_model(path: str | Path, device: str) -> tuple[dict[str, Any], Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def _load_training_resources(path: str | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("training-resource JSON must be an object")
    paired_seed = value.get("paired_seed")
    if paired_seed is not None and paired_seed not in FIT_SEEDS:
        raise ValueError("training-resource JSON has a non-frozen paired seed")
    result = {}
    for model in ("treatment", "sham"):
        item = value.get(model)
        if not isinstance(item, dict):
            raise ValueError("training-resource JSON lacks a controller arm")
        if not {
            "supervised_completion_tokens",
            "packed_forward_token_positions",
        }.issubset(item):
            raise ValueError(
                "training-resource JSON must contain exact consumed supervised "
                "and packed-forward token counts"
            )
        supervised = int(item["supervised_completion_tokens"])
        forward = int(item["packed_forward_token_positions"])
        if supervised < 0 or forward < 0:
            raise ValueError("training-resource counts must be nonnegative")
        result[model] = {
            "supervised_completion_tokens": supervised,
            "packed_forward_token_positions": forward,
        }
    if result["treatment"] != result["sham"]:
        raise ValueError("paired training-resource counts differ between arms")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--sham", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=FIT_SEEDS)
    parser.add_argument("--training-resources")
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    raise SystemExit(
        "RSP-C1 is permanently closed; preserve this module as audit evidence only"
    )
    args = build_parser().parse_args()
    if Path(args.out).exists():
        raise FileExistsError("refusing to overwrite raw transcript")
    board, rows = load_board(args.board)
    paths = {
        "board": args.board,
        "tokenizer": args.tokenizer,
        "raw_260k_executor": args.executor,
        "treatment_checkpoint": args.treatment,
        "sham_checkpoint": args.sham,
        "protocol": Path(PROTOCOL.__file__).resolve(),
        "evaluator": Path(__file__).resolve(),
    }
    if args.training_resources:
        paths["training_resources"] = args.training_resources
    initial_hashes = {name: sha256_file(path) for name, path in paths.items()}
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    endpoints = []
    for name, path in (
        ("treatment", args.treatment),
        ("sham", args.sham),
        ("raw_260k_executor", args.executor),
    ):
        checkpoint, model = load_model(path, device)
        endpoints.append(
            ModelEndpoint(name, path, model, tokenizer, device, checkpoint.get("step"))
        )
    transcript = acquire_transcript(
        board,
        rows,
        endpoints[0],
        endpoints[1],
        endpoints[2],
        args.seed,
        initial_hashes,
        _load_training_resources(args.training_resources),
    )
    final_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if final_hashes != initial_hashes:
        raise RuntimeError("an immutable evaluation input changed during acquisition")
    digest = write_immutable_json(args.out, transcript)
    print(json.dumps({"out": args.out, "sha256": digest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
