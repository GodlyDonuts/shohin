#!/usr/bin/env python3
"""Clean-room, fail-closed scorer for the frozen RSP-C1 experiment.

This scorer intentionally shares no code with the acquisition evaluator or the
other RSP-C1 scorer.  It treats the board and transcript as hostile evidence:
all arithmetic, protocol transitions, call order, token accounting, resource
accounting, paired tests, and gates are reconstructed from raw fields.

The post-generation hashes in ``PRODUCTION_BINDINGS`` are deliberately invalid
freeze sentinels until the conditional RSP-C1 prerequisite passes and those
artifacts exist.  The CLI refuses to run while any sentinel remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


BOARD_SCHEMA = "residual_packet_board_v1"
TRANSCRIPT_SCHEMA = "residual_packet_v1_raw_transcript"
SCORE_SCHEMA = "residual_packet_v1_independent_score_v1"
FIT_SEEDS = (2026071511, 2026071512)
BOARD_SEED = 2026071503
UPDATER_SEED = 2026071505
STRATA = ("renderer_ood", "value_ood", "order_ood", "length_ood")
MODEL_NAMES = ("treatment", "sham", "raw_260k_executor")
CONTROLLER_NAMES = ("treatment", "sham")
MAX_CONTROLLER_TOKENS = 80
MAX_EXECUTOR_TOKENS = 48
MAX_TRANSITIONS = 5

INTEGER = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)"
POSITIVE = r"[1-9][0-9]*"
OP_RE = re.compile(rf"(?P<kind>add|multiply|subtract) (?P<value>{POSITIVE})\Z")
PACKET_RE = re.compile(
    rf"State: (?P<state>{INTEGER})\n"
    rf"Plan: (?P<plan>(?:add|multiply|subtract) {POSITIVE}"
    rf"(?:; (?:add|multiply|subtract) {POSITIVE})*)\Z"
)
ANSWER_RE = re.compile(rf"Answer: (?P<answer>{INTEGER})\Z")
EXECUTOR_INT_RE = re.compile(
    r"(?<![A-Za-z0-9_,])(?<!\d\.)-?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?![A-Za-z0-9_,]|\.\d)"
)
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")

CALL_KEYS = {
    "call_index",
    "model",
    "arm",
    "prompt",
    "max_new",
    "response",
    "prompt_token_count",
    "sampled_token_ids",
    "sampled_token_count",
    "decoded_token_ids",
    "decoded_token_count",
    "stop_reason",
}
RESOURCE_KEYS = {
    "model_calls",
    "prompt_tokens",
    "sampled_tokens",
    "decoded_tokens",
    "supervised_completion_tokens",
    "packed_forward_token_positions",
    "calls_not_issued_after_parse_failure",
    "retries",
    "repairs",
    "searches",
    "verifier_feedback_calls",
}

REQUIRED_ARTIFACTS = (
    "prereg",
    "prerequisite_confirmation",
    "protocol",
    "sft_encoding",
    "evaluator",
    "board",
    "tokenizer",
    "executor_checkpoint",
    "treatment_data",
    "sham_data",
    "training_manifest",
    "audit",
    "training_resources",
    "controller_manifest",
    "executor_manifest",
    "treatment_checkpoint",
    "sham_checkpoint",
    "transcript",
)


class IntegrityError(ValueError):
    """Raised before any score is admitted."""


class TokenCodec(Protocol):
    eos_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class HuggingFaceTokenCodec:
    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(path))
        eos_id = self._tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            raise IntegrityError("tokenizer has no <|endoftext|> token")
        self.eos_id = int(eos_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        try:
            return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)
        except TypeError:
            return self._tokenizer.decode(list(token_ids))


@dataclass(frozen=True)
class ScoreContract:
    board_seed: int = BOARD_SEED
    fit_seeds: tuple[int, ...] = FIT_SEEDS
    strata: tuple[str, ...] = STRATA
    case_count: int = 256
    per_stratum: int = 64
    swaps_per_stratum: int = 16
    max_transitions: int = MAX_TRANSITIONS
    external_gold_min: int = 128
    oracle_min: int = 230
    compile_min: int = 224
    strict_min: int = 192
    per_stratum_compile_min: int = 52
    per_stratum_strict_min: int = 40
    external_mismatch_max: int = 8
    swap_min: int = 60
    update_min: Fraction = field(default_factory=lambda: Fraction(95, 100))
    compile_delta_min: Fraction = field(default_factory=lambda: Fraction(30, 100))
    strict_delta_min: Fraction = field(default_factory=lambda: Fraction(25, 100))
    mcnemar_max: Fraction = field(default_factory=lambda: Fraction(1, 100))


@dataclass(frozen=True)
class FrozenBindings:
    seed: int
    sha256: Mapping[str, str]
    board_rows_sha256: str

    def validate(self) -> None:
        if self.seed not in FIT_SEEDS and self.seed <= 0:
            raise IntegrityError("binding seed is invalid")
        if set(self.sha256) != set(REQUIRED_ARTIFACTS):
            missing = sorted(set(REQUIRED_ARTIFACTS) - set(self.sha256))
            extra = sorted(set(self.sha256) - set(REQUIRED_ARTIFACTS))
            raise IntegrityError(f"binding keys differ: missing={missing} extra={extra}")
        for label, digest in self.sha256.items():
            if not isinstance(digest, str) or HEX64_RE.fullmatch(digest) is None:
                raise IntegrityError(f"{label} digest has not been frozen")
        if HEX64_RE.fullmatch(self.board_rows_sha256) is None:
            raise IntegrityError("board row digest has not been frozen")


def _sentinel(label: str) -> str:
    return f"TO_BE_FROZEN_{label.upper()}"


def _production_binding(seed: int) -> FrozenBindings:
    known = {
        "prereg": _sentinel("prereg_sha256"),
        "prerequisite_confirmation": _sentinel("confirmation_sha256"),
        "protocol": "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2",
        "sft_encoding": "33d07db19afb7155567cef922dbab616916864eafaeba7fa26f12d2203c0d5b8",
        "evaluator": _sentinel("evaluator_sha256"),
        "board": "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7",
        "tokenizer": "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4",
        "executor_checkpoint": "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d",
        "treatment_data": _sentinel("treatment_data_sha256"),
        "sham_data": _sentinel("sham_data_sha256"),
        "training_manifest": _sentinel("training_manifest_sha256"),
        "audit": _sentinel("audit_sha256"),
        "training_resources": _sentinel(f"training_resources_{seed}_sha256"),
        "controller_manifest": _sentinel(f"controller_manifest_{seed}_sha256"),
        "executor_manifest": _sentinel("executor_manifest_sha256"),
        "treatment_checkpoint": _sentinel(f"treatment_checkpoint_{seed}_sha256"),
        "sham_checkpoint": _sentinel(f"sham_checkpoint_{seed}_sha256"),
        "transcript": _sentinel(f"transcript_{seed}_sha256"),
    }
    return FrozenBindings(
        seed=seed,
        sha256=known,
        board_rows_sha256="fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e",
    )


PRODUCTION_BINDINGS = {seed: _production_binding(seed) for seed in FIT_SEEDS}


def _is_int(value: object) -> bool:
    return type(value) is int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _exact_keys(value: object, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    expected = set(keys)
    observed = set(value)
    _require(observed == expected, f"{label} keys differ: {sorted(observed ^ expected)}")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"{label} is not valid unique-key JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_regular(path: str | Path, label: str, immutable: bool = False) -> bytes:
    source = Path(path)
    info = source.lstat()
    _require(stat.S_ISREG(info.st_mode), f"{label} is not a regular file")
    _require(not source.is_symlink(), f"{label} may not be a symlink")
    if immutable:
        _require(info.st_mode & 0o222 == 0, f"{label} retained write bits")
    return source.read_bytes()


def assert_no_booleans(value: object, path: str = "$") -> None:
    if isinstance(value, bool):
        raise IntegrityError(f"raw transcript contains a boolean at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_booleans(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_booleans(child, f"{path}[{index}]")


Operation = tuple[str, int]
Packet = tuple[int, tuple[Operation, ...]]


def parse_operation(value: object) -> Operation | None:
    if not isinstance(value, str):
        return None
    match = OP_RE.fullmatch(value)
    if match is None:
        return None
    return match.group("kind"), int(match.group("value"))


def parse_operations(value: object) -> tuple[Operation, ...]:
    _require(isinstance(value, list) and value, "operation list is empty or malformed")
    result: list[Operation] = []
    for item in value:
        _require(isinstance(item, list) and len(item) == 2, "operation row is malformed")
        kind, operand = item
        _require(isinstance(kind, str) and _is_int(operand), "operation types are malformed")
        parsed = parse_operation(f"{kind} {operand}")
        _require(parsed is not None, "operation is outside the frozen grammar")
        result.append(parsed)
    return tuple(result)


def parse_packet(value: object) -> Packet | None:
    if not isinstance(value, str):
        return None
    match = PACKET_RE.fullmatch(value.strip(" \t\r\n\f\v"))
    if match is None:
        return None
    operations = tuple(parse_operation(item) for item in match.group("plan").split("; "))
    if any(item is None for item in operations):
        return None
    return int(match.group("state")), tuple(operations)  # type: ignore[arg-type]


def parse_answer(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = ANSWER_RE.fullmatch(value.strip(" \t\r\n\f\v"))
    return int(match.group("answer")) if match is not None else None


def format_operation(operation: Operation) -> str:
    kind, operand = operation
    rendered = f"{kind} {operand}"
    _require(parse_operation(rendered) == operation, "cannot render operation")
    return rendered


def format_packet(state: int, operations: Sequence[Operation]) -> str:
    _require(_is_int(state) and bool(operations), "cannot render empty packet")
    rendered = f"State: {state}\nPlan: " + "; ".join(map(format_operation, operations))
    _require(parse_packet(rendered) == (state, tuple(operations)), "packet render failed")
    return rendered


def format_answer(value: int) -> str:
    _require(_is_int(value), "answer is not an integer")
    return f"Answer: {value}"


def compiler_prompt(source: str) -> str:
    return f"Problem: {source}\nCompile only the execution packet.\nPacket:"


def update_prompt(packet: str, observed: int) -> str:
    return f"Packet:\n{packet}\nObserved result: {observed}\nNext packet:"


def executor_prompt(state: int, operation: Operation) -> str:
    kind, operand = operation
    if kind == "add":
        clause = f"Compute {state} plus {operand}."
    elif kind == "multiply":
        clause = f"Compute {state} times {operand}."
    else:
        clause = f"Compute {state} minus {operand}."
    return f"Problem: {clause}\nWork:"


def parse_executor_result(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    matches = EXECUTOR_INT_RE.findall(line)
    return int(matches[-1].replace(",", "")) if matches else None


def apply_operation(state: int, operation: Operation) -> int:
    kind, operand = operation
    if kind == "add":
        return state + operand
    if kind == "multiply":
        return state * operand
    if kind == "subtract":
        return state - operand
    raise IntegrityError(f"unknown operation {kind}")


def replay_math(initial: int, operations: Sequence[Operation]) -> tuple[int, ...]:
    states = [initial]
    for operation in operations:
        states.append(apply_operation(states[-1], operation))
    return tuple(states)


@dataclass(frozen=True)
class BoardRow:
    identifier: str
    stratum: str
    source: str
    initial: int
    operations: tuple[Operation, ...]
    packet: str
    trajectory: tuple[int, ...]
    answer: int


def validate_board(
    board: object,
    bindings: FrozenBindings,
    contract: ScoreContract,
) -> list[BoardRow]:
    board = _exact_keys(
        board,
        {
            "case_count",
            "per_stratum",
            "protocol_sha256",
            "rows",
            "rows_sha256",
            "schema",
            "seed",
            "stratum_order",
        },
        "board",
    )
    _require(board["schema"] == BOARD_SCHEMA, "board schema mismatch")
    _require(board["seed"] == contract.board_seed, "board seed mismatch")
    _require(board["case_count"] == contract.case_count, "board case count mismatch")
    _require(board["per_stratum"] == contract.per_stratum, "board balance mismatch")
    _require(board["stratum_order"] == list(contract.strata), "board stratum order mismatch")
    _require(board["protocol_sha256"] == bindings.sha256["protocol"], "board protocol binding mismatch")
    rows_raw = board["rows"]
    _require(isinstance(rows_raw, list) and len(rows_raw) == contract.case_count, "board rows mismatch")
    rows_digest = sha256_bytes(canonical_json_bytes(rows_raw))
    _require(rows_digest == bindings.board_rows_sha256, "board row hash mismatch")
    _require(board["rows_sha256"] == rows_digest, "embedded board row hash mismatch")

    rows: list[BoardRow] = []
    identities: set[str] = set()
    sources: set[str] = set()
    packets: set[str] = set()
    trajectories: set[tuple[int, ...]] = set()
    answers: set[int] = set()
    programs: set[tuple[Any, ...]] = set()
    strata = Counter()
    for index, raw in enumerate(rows_raw):
        row = _exact_keys(
            raw,
            {
                "answer",
                "id",
                "initial_state",
                "operations",
                "packet",
                "source",
                "stratum",
                "template_id",
                "trajectory",
            },
            f"board.rows[{index}]",
        )
        _require(isinstance(row["id"], str) and row["id"], "board id malformed")
        _require(row["stratum"] in contract.strata, "board stratum malformed")
        _require(isinstance(row["source"], str) and row["source"], "board source malformed")
        _require(isinstance(row["template_id"], str) and row["template_id"], "template id malformed")
        _require(_is_int(row["initial_state"]), "board initial state malformed")
        _require(_is_int(row["answer"]), "board answer malformed")
        operations = parse_operations(row["operations"])
        _require(1 <= len(operations) <= contract.max_transitions, "board program length malformed")
        trajectory = replay_math(row["initial_state"], operations)
        _require(all(state > 0 for state in trajectory), "board has nonpositive state")
        _require(row["trajectory"] == list(trajectory), "board trajectory replay mismatch")
        _require(row["answer"] == trajectory[-1], "board answer replay mismatch")
        packet = format_packet(row["initial_state"], operations)
        _require(row["packet"] == packet, "board packet mismatch")
        signature = (row["initial_state"], operations)
        for value, seen, label in (
            (row["id"], identities, "id"),
            (row["source"], sources, "source"),
            (packet, packets, "packet"),
            (trajectory, trajectories, "trajectory"),
            (row["answer"], answers, "answer"),
            (signature, programs, "program"),
        ):
            _require(value not in seen, f"duplicate board {label}")
            seen.add(value)
        strata[row["stratum"]] += 1
        rows.append(
            BoardRow(
                identifier=row["id"],
                stratum=row["stratum"],
                source=row["source"],
                initial=row["initial_state"],
                operations=operations,
                packet=packet,
                trajectory=trajectory,
                answer=row["answer"],
            )
        )
    _require(
        strata == Counter({stratum: contract.per_stratum for stratum in contract.strata}),
        "board strata are not exactly balanced",
    )
    expected_order = [stratum for stratum in contract.strata for _ in range(contract.per_stratum)]
    _require([row.stratum for row in rows] == expected_order, "board rows are out of stratum order")
    return rows


def _collect_string_leaves(value: object) -> set[str]:
    leaves: set[str] = set()
    if isinstance(value, str):
        leaves.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            leaves.update(_collect_string_leaves(child))
    elif isinstance(value, list):
        for child in value:
            leaves.update(_collect_string_leaves(child))
    return leaves


def validate_support_artifacts(
    loaded: Mapping[str, Any], bindings: FrozenBindings
) -> dict[str, dict[str, int]]:
    confirmation = loaded["prerequisite_confirmation"]
    _require(isinstance(confirmation, Mapping), "prerequisite confirmation is not an object")

    def find_advance(value: object) -> list[bool]:
        found: list[bool] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key == "advance_to_internalization":
                    found.append(child is True)
                else:
                    found.extend(find_advance(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(find_advance(child))
        return found

    advance = find_advance(confirmation)
    _require(advance == [True], "prerequisite confirmation is not uniquely passing")

    manifest = loaded["training_manifest"]
    _require(isinstance(manifest, Mapping), "training manifest is not an object")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, Mapping), "training manifest lacks artifacts")
    expected_manifest_hashes = {
        "board_sha256": bindings.sha256["board"],
        "board_rows_sha256": bindings.board_rows_sha256,
        "treatment_sha256": bindings.sha256["treatment_data"],
        "sham_sha256": bindings.sha256["sham_data"],
    }
    for key, expected in expected_manifest_hashes.items():
        _require(artifacts.get(key) == expected, f"training manifest {key} mismatch")
    _require(manifest.get("tokenizer_sha256") == bindings.sha256["tokenizer"], "manifest tokenizer mismatch")
    _require(manifest.get("protocol_sha256") == bindings.sha256["protocol"], "manifest protocol mismatch")
    _require(
        manifest.get("sft_encoding_sha256") == bindings.sha256["sft_encoding"],
        "manifest SFT encoding mismatch",
    )

    audit = loaded["audit"]
    _require(isinstance(audit, Mapping), "audit is not an object")
    _require(audit.get("failures") == [], "admission audit contains failures")
    _require(audit.get("admitted") is True, "admission audit is not admitted")
    audit_hashes = audit.get("artifact_sha256")
    _require(isinstance(audit_hashes, Mapping), "audit lacks artifact hashes")
    for key, expected in {
        "board": bindings.sha256["board"],
        "manifest": bindings.sha256["training_manifest"],
        "protocol": bindings.sha256["protocol"],
        "sft_encoding": bindings.sha256["sft_encoding"],
        "sham": bindings.sha256["sham_data"],
        "treatment": bindings.sha256["treatment_data"],
        "tokenizer": bindings.sha256["tokenizer"],
    }.items():
        _require(audit_hashes.get(key) == expected, f"audit {key} hash mismatch")

    resources = loaded["training_resources"]
    resources = _exact_keys(resources, {"paired_seed", "treatment", "sham"}, "training resources")
    _require(resources["paired_seed"] == bindings.seed, "training resource seed mismatch")
    parsed_resources: dict[str, dict[str, int]] = {}
    for arm in CONTROLLER_NAMES:
        item = _exact_keys(
            resources[arm],
            {"supervised_completion_tokens", "packed_forward_token_positions"},
            f"training resources.{arm}",
        )
        parsed: dict[str, int] = {}
        for key, value in item.items():
            _require(_is_int(value) and value >= 0, f"training resource {arm}.{key} invalid")
            parsed[key] = value
        parsed_resources[arm] = parsed
    _require(parsed_resources["treatment"] == parsed_resources["sham"], "paired resources differ")

    controller_leaves = _collect_string_leaves(loaded["controller_manifest"])
    for label in (
        "treatment_checkpoint",
        "sham_checkpoint",
        "tokenizer",
        "training_resources",
        "board",
        "audit",
        "treatment_data",
        "sham_data",
        "training_manifest",
        "sft_encoding",
    ):
        _require(bindings.sha256[label] in controller_leaves, f"controller manifest does not bind {label}")
    executor_leaves = _collect_string_leaves(loaded["executor_manifest"])
    for label in ("executor_checkpoint", "tokenizer"):
        _require(bindings.sha256[label] in executor_leaves, f"executor manifest does not bind {label}")
    return parsed_resources


@dataclass
class CallCollector:
    codec: TokenCodec
    records: list[Mapping[str, Any]] = field(default_factory=list)
    unissued: defaultdict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def call(
        self,
        value: object,
        *,
        model: str,
        arm: str,
        prompt: str,
        max_new: int,
        label: str,
    ) -> Mapping[str, Any]:
        call = _exact_keys(value, CALL_KEYS, label)
        _require(call["model"] == model, f"{label} model mismatch")
        _require(call["arm"] == arm, f"{label} arm mismatch")
        _require(call["prompt"] == prompt, f"{label} prompt mismatch")
        _require(call["max_new"] == max_new, f"{label} decode cap mismatch")
        _require(_is_int(call["call_index"]) and call["call_index"] >= 0, f"{label} call index invalid")
        _require(isinstance(call["response"], str), f"{label} response invalid")
        for count_key in ("prompt_token_count", "sampled_token_count", "decoded_token_count"):
            _require(_is_int(call[count_key]) and call[count_key] >= 0, f"{label} {count_key} invalid")
        sampled = call["sampled_token_ids"]
        decoded = call["decoded_token_ids"]
        _require(isinstance(sampled, list) and all(_is_int(token) and token >= 0 for token in sampled), f"{label} sampled ids invalid")
        _require(isinstance(decoded, list) and all(_is_int(token) and token >= 0 for token in decoded), f"{label} decoded ids invalid")
        _require(call["sampled_token_count"] == len(sampled), f"{label} sampled count mismatch")
        _require(call["decoded_token_count"] == len(decoded), f"{label} decoded count mismatch")
        _require(call["prompt_token_count"] == len(self.codec.encode(prompt)), f"{label} prompt token count mismatch")
        _require(self.codec.decode(decoded) == call["response"], f"{label} decoded response mismatch")
        stop = call["stop_reason"]
        _require(stop in {"eos", "max_new", "context_limit"}, f"{label} stop reason invalid")
        if stop == "eos":
            _require(sampled == decoded + [self.codec.eos_id], f"{label} EOS accounting mismatch")
        else:
            _require(sampled == decoded, f"{label} sampled/decoded ids mismatch")
        if stop == "max_new":
            _require(len(sampled) == max_new, f"{label} max_new stop count mismatch")
        _require(len(sampled) <= max_new, f"{label} sampled beyond decode cap")
        self.records.append(call)
        return call

    def note_unissued(self, model: str, arm: str, count: int, label: str) -> None:
        _require(_is_int(count) and count >= 0, f"{label} negative unissued calls")
        self.unissued[(model, arm)] += count

    def finalize_order(self, call_count: object) -> None:
        _require(_is_int(call_count), "transcript call_count is not an integer")
        indices = [record["call_index"] for record in self.records]
        _require(indices == list(range(len(indices))), "call indices do not match issue order")
        _require(call_count == len(indices), "transcript call_count mismatch")

    def expected_ledger(
        self, resources: Mapping[str, Mapping[str, int]]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"by_model": {}}
        for model in MODEL_NAMES:
            records = [record for record in self.records if record["model"] == model]
            arms = sorted(
                {record["arm"] for record in records}
                | {arm for (name, arm), count in self.unissued.items() if name == model and count >= 0}
            )
            by_arm: dict[str, Any] = {}
            for arm in arms:
                selected = [record for record in records if record["arm"] == arm]
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
            training = resources.get(model, {})
            by_arm["training"] = {
                "model_calls": 0,
                "prompt_tokens": 0,
                "sampled_tokens": 0,
                "decoded_tokens": 0,
                "supervised_completion_tokens": int(training.get("supervised_completion_tokens", 0)),
                "packed_forward_token_positions": int(training.get("packed_forward_token_positions", 0)),
                "calls_not_issued_after_parse_failure": 0,
                "retries": 0,
                "repairs": 0,
                "searches": 0,
                "verifier_feedback_calls": 0,
            }
            result["by_model"][model] = {"by_arm": by_arm}
        return result


@dataclass(frozen=True)
class RuntimeOutcome:
    exact: bool
    initial_packet: Packet | None
    emitted_states: tuple[int, ...]
    executor_states: tuple[int, ...]
    final_answer: int | None
    inferred_termination: str


def replay_runtime(
    runtime_value: object,
    initial_packet_text: object,
    *,
    controller: str,
    arm: str,
    expected_transitions: int,
    collector: CallCollector,
    label: str,
) -> RuntimeOutcome:
    runtime = _exact_keys(runtime_value, {"termination", "steps"}, label)
    _require(isinstance(runtime["termination"], str), f"{label} termination malformed")
    steps = runtime["steps"]
    _require(isinstance(steps, list), f"{label} steps malformed")
    packet = parse_packet(initial_packet_text)
    if packet is None:
        _require(not steps, f"{label} issued calls after invalid initial packet")
        inferred = "initial_packet_invalid"
        collector.note_unissued("raw_260k_executor", arm, expected_transitions, label)
        collector.note_unissued(controller, arm, expected_transitions, label)
        _require(runtime["termination"] == inferred, f"{label} trusted termination mismatch")
        return RuntimeOutcome(False, None, (), (), None, inferred)

    _require(len(steps) <= MAX_TRANSITIONS, f"{label} exceeds transition cap")
    current = packet
    emitted = [current[0]]
    observed_states: list[int] = []
    exact = True
    final_answer: int | None = None
    inferred: str | None = None
    updater_calls = 0
    for index, step_value in enumerate(steps):
        step = _exact_keys(step_value, {"executor", "updater"} if isinstance(step_value, Mapping) and "updater" in step_value else {"executor"}, f"{label}.steps[{index}]")
        state, operations = current
        _require(operations, f"{label} has an empty residual program")
        operation = operations[0]
        executor = collector.call(
            step["executor"],
            model="raw_260k_executor",
            arm=arm,
            prompt=executor_prompt(state, operation),
            max_new=MAX_EXECUTOR_TOKENS,
            label=f"{label}.steps[{index}].executor",
        )
        observed = parse_executor_result(executor["response"])
        if observed is None:
            _require("updater" not in step, f"{label} called updater after executor parse failure")
            _require(index == len(steps) - 1, f"{label} continued after executor parse failure")
            exact = False
            inferred = "executor_result_invalid"
            break
        observed_states.append(observed)
        _require("updater" in step, f"{label} omitted updater after parsed executor result")
        canonical = format_packet(state, operations)
        updater = collector.call(
            step["updater"],
            model=controller,
            arm=arm,
            prompt=update_prompt(canonical, observed),
            max_new=MAX_CONTROLLER_TOKENS,
            label=f"{label}.steps[{index}].updater",
        )
        updater_calls += 1
        response = updater["response"]
        if len(operations) == 1 and parse_answer(response) == observed:
            emitted.append(observed)
            final_answer = observed
            inferred = "answer"
            _require(index == len(steps) - 1, f"{label} continued after final answer")
            break
        expected_packet = (observed, operations[1:])
        parsed_next = parse_packet(response)
        if len(operations) > 1 and parsed_next == expected_packet:
            emitted.append(observed)
            current = expected_packet
            continue
        exact = False
        inferred = "updater_output_invalid"
        _require(index == len(steps) - 1, f"{label} continued after updater failure")
        break
    if inferred is None:
        _require(len(steps) == MAX_TRANSITIONS, f"{label} ended without a terminal event")
        exact = False
        inferred = "transition_limit"
    if inferred != "answer":
        exact = False
    _require(len(steps) <= expected_transitions, f"{label} issued more executor calls than admitted")
    _require(updater_calls <= expected_transitions, f"{label} issued more updater calls than admitted")
    collector.note_unissued("raw_260k_executor", arm, expected_transitions - len(steps), label)
    collector.note_unissued(controller, arm, expected_transitions - updater_calls, label)
    _require(runtime["termination"] == inferred, f"{label} trusted termination mismatch")
    return RuntimeOutcome(
        exact,
        packet,
        tuple(emitted),
        tuple(observed_states),
        final_answer,
        inferred,
    )


def replay_external(
    value: object,
    row: BoardRow,
    collector: CallCollector,
    label: str,
) -> RuntimeOutcome:
    runtime = _exact_keys(value, {"termination", "steps"}, label)
    steps = runtime["steps"]
    _require(isinstance(steps, list), f"{label} steps malformed")
    _require(len(steps) <= len(row.operations), f"{label} has excess calls")
    state = row.initial
    emitted = [state]
    inferred = "complete"
    for index, call_value in enumerate(steps):
        operation = row.operations[index]
        call = collector.call(
            call_value,
            model="raw_260k_executor",
            arm="external_scheduler",
            prompt=executor_prompt(state, operation),
            max_new=MAX_EXECUTOR_TOKENS,
            label=f"{label}.steps[{index}]",
        )
        parsed = parse_executor_result(call["response"])
        if parsed is None:
            inferred = "executor_result_invalid"
            _require(index == len(steps) - 1, f"{label} continued after parse failure")
            break
        state = parsed
        emitted.append(state)
    if inferred == "complete":
        _require(len(steps) == len(row.operations), f"{label} silently omitted calls")
    collector.note_unissued(
        "raw_260k_executor",
        "external_scheduler",
        len(row.operations) - len(steps),
        label,
    )
    _require(runtime["termination"] == inferred, f"{label} trusted termination mismatch")
    complete = inferred == "complete"
    return RuntimeOutcome(
        complete,
        (row.initial, row.operations),
        tuple(emitted),
        tuple(emitted[1:]),
        emitted[-1] if complete else None,
        inferred,
    )


def _prf_integer(label: str) -> int:
    digest = hashlib.sha256(f"{UPDATER_SEED}:{label}".encode("ascii")).digest()
    return 10 + int.from_bytes(digest[:8], "big") % 990


def expected_teacher_cases(rows: Sequence[BoardRow]) -> list[tuple[BoardRow, int, str, int, str]]:
    board_answers = {row.answer for row in rows}
    result = []
    for row in rows:
        for index, operation in enumerate(row.operations):
            nonce = 0
            while True:
                state = _prf_integer(f"{row.identifier}:{index}:state:{nonce}")
                if state not in board_answers:
                    break
                nonce += 1
            nonce = 0
            while True:
                observed = _prf_integer(f"{row.identifier}:{index}:observed:{nonce}")
                if observed not in board_answers and observed != apply_operation(state, operation):
                    break
                nonce += 1
            packet = format_packet(state, row.operations[index:])
            if len(row.operations[index:]) == 1:
                expected = format_answer(observed)
            else:
                expected = format_packet(observed, row.operations[index + 1 :])
            result.append((row, index, packet, observed, expected))
    return result


def expected_swaps(rows: Sequence[BoardRow], contract: ScoreContract) -> list[tuple[BoardRow, BoardRow]]:
    result: list[tuple[BoardRow, BoardRow]] = []
    for stratum in contract.strata:
        selected = [row for row in rows if row.stratum == stratum][: contract.swaps_per_stratum]
        _require(len(selected) == contract.swaps_per_stratum, "insufficient packet-swap cases")
        for index, original in enumerate(selected):
            result.append((original, selected[(index + 1) % len(selected)]))
    return result


def _metric(correct: int, total: int) -> dict[str, Any]:
    _require(0 <= correct <= total and total > 0, "invalid metric count")
    fraction = Fraction(correct, total)
    return {
        "correct": correct,
        "total": total,
        "rate": float(fraction),
        "rate_exact": f"{fraction.numerator}/{fraction.denominator}",
    }


def _stratified_metric(values: Sequence[bool], rows: Sequence[BoardRow], contract: ScoreContract) -> dict[str, Any]:
    _require(len(values) == len(rows), "stratified metric length mismatch")
    result = _metric(sum(values), len(values))
    result["by_stratum"] = {
        stratum: _metric(
            sum(value for value, row in zip(values, rows) if row.stratum == stratum),
            sum(row.stratum == stratum for row in rows),
        )
        for stratum in contract.strata
    }
    return result


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    _require(len(left) == len(right) and len(left) > 0, "McNemar pairs malformed")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        probability = Fraction(1, 1)
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        probability = min(Fraction(1, 1), Fraction(2 * tail, 2**discordant))
    return {
        "treatment_only": left_only,
        "sham_only": right_only,
        "discordant": discordant,
        "p_value": float(probability),
        "p_value_decimal": format(float(probability), ".17g"),
        "p_value_exact": f"{probability.numerator}/{probability.denominator}",
    }


def validate_transcript_models(
    value: object, bindings: FrozenBindings
) -> None:
    models = _exact_keys(value, MODEL_NAMES, "transcript.models")
    expected = {
        "treatment": bindings.sha256["treatment_checkpoint"],
        "sham": bindings.sha256["sham_checkpoint"],
        "raw_260k_executor": bindings.sha256["executor_checkpoint"],
    }
    for model, digest in expected.items():
        metadata = _exact_keys(
            models[model],
            {"checkpoint_path", "checkpoint_sha256", "checkpoint_step"},
            f"transcript.models.{model}",
        )
        _require(isinstance(metadata["checkpoint_path"], str) and metadata["checkpoint_path"], "checkpoint path malformed")
        _require(metadata["checkpoint_sha256"] == digest, f"{model} checkpoint hash mismatch")
        _require(metadata["checkpoint_step"] is None or _is_int(metadata["checkpoint_step"]), "checkpoint step malformed")
    _require(models["raw_260k_executor"]["checkpoint_step"] == 260000, "executor step is not 260000")


def score_transcript(
    board: object,
    transcript: object,
    *,
    bindings: FrozenBindings,
    contract: ScoreContract,
    codec: TokenCodec,
    training_resources: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    rows = validate_board(board, bindings, contract)
    assert_no_booleans(transcript)
    transcript = _exact_keys(
        transcript,
        {
            "schema",
            "seed",
            "protocol_module",
            "input_hashes",
            "decode_caps",
            "models",
            "external_scheduler",
            "controllers",
            "resource_ledger",
            "call_count",
        },
        "transcript",
    )
    _require(transcript["schema"] == TRANSCRIPT_SCHEMA, "transcript schema mismatch")
    _require(transcript["seed"] == bindings.seed, "transcript seed mismatch")
    _require(bindings.seed in contract.fit_seeds, "transcript seed is outside contract")
    _require(isinstance(transcript["protocol_module"], str) and transcript["protocol_module"].endswith("residual_packet_protocol"), "protocol module identity mismatch")
    _require(
        transcript["decode_caps"]
        == {
            "controller": MAX_CONTROLLER_TOKENS,
            "executor": MAX_EXECUTOR_TOKENS,
            "maximum_transitions": contract.max_transitions,
        },
        "decode caps mismatch",
    )
    expected_input_hashes = {
        "board": bindings.sha256["board"],
        "tokenizer": bindings.sha256["tokenizer"],
        "raw_260k_executor": bindings.sha256["executor_checkpoint"],
        "treatment_checkpoint": bindings.sha256["treatment_checkpoint"],
        "sham_checkpoint": bindings.sha256["sham_checkpoint"],
        "protocol": bindings.sha256["protocol"],
        "evaluator": bindings.sha256["evaluator"],
        "training_resources": bindings.sha256["training_resources"],
    }
    _require(transcript["input_hashes"] == expected_input_hashes, "transcript input hashes mismatch")
    validate_transcript_models(transcript["models"], bindings)

    collector = CallCollector(codec)
    external_raw = transcript["external_scheduler"]
    _require(isinstance(external_raw, list) and len(external_raw) == len(rows), "external scheduler rows mismatch")
    external_outcomes: dict[str, RuntimeOutcome] = {}
    external_gold: list[bool] = []
    external_gold_trajectories: list[bool] = []
    external_step_values: list[bool] = []
    for index, (item_value, row) in enumerate(zip(external_raw, rows)):
        item = _exact_keys(item_value, {"id", "runtime"}, f"external[{index}]")
        _require(item["id"] == row.identifier, "external row id mismatch")
        outcome = replay_external(item["runtime"], row, collector, f"external[{index}].runtime")
        external_outcomes[row.identifier] = outcome
        external_gold.append(outcome.final_answer == row.answer)
        external_gold_trajectories.append(outcome.emitted_states == row.trajectory)
        external_step_values.extend(
            index < len(outcome.executor_states)
            and outcome.executor_states[index] == row.trajectory[index + 1]
            for index in range(len(row.operations))
        )

    controllers = _exact_keys(transcript["controllers"], CONTROLLER_NAMES, "controllers")
    arm_results: dict[str, Any] = {}
    arm_vectors: dict[str, dict[str, list[bool]]] = {}
    teacher_expectations = expected_teacher_cases(rows)
    swaps = expected_swaps(rows, contract)

    for controller in CONTROLLER_NAMES:
        arm = _exact_keys(
            controllers[controller],
            {"strict_closed_loop", "oracle_packet_loop", "teacher_forced_updater", "packet_swaps"},
            f"controllers.{controller}",
        )
        strict_raw = arm["strict_closed_loop"]
        oracle_raw = arm["oracle_packet_loop"]
        _require(isinstance(strict_raw, list) and len(strict_raw) == len(rows), f"{controller} strict rows mismatch")
        _require(isinstance(oracle_raw, list) and len(oracle_raw) == len(rows), f"{controller} oracle rows mismatch")
        compile_values: list[bool] = []
        strict_values: list[bool] = []
        oracle_values: list[bool] = []
        oracle_gold_values: list[bool] = []
        trajectory_values: list[bool] = []
        gold_values: list[bool] = []
        gold_trajectory_values: list[bool] = []
        strict_outcomes: list[RuntimeOutcome] = []

        # Acquisition interleaves strict and oracle calls per board row.
        for index, (strict_value, oracle_value, row) in enumerate(zip(strict_raw, oracle_raw, rows)):
            strict_item = _exact_keys(strict_value, {"id", "compiler", "runtime"}, f"{controller}.strict[{index}]")
            _require(strict_item["id"] == row.identifier, f"{controller} strict id mismatch")
            compiler = collector.call(
                strict_item["compiler"],
                model=controller,
                arm="strict_closed_loop",
                prompt=compiler_prompt(row.source),
                max_new=MAX_CONTROLLER_TOKENS,
                label=f"{controller}.strict[{index}].compiler",
            )
            compile_exact = parse_packet(compiler["response"]) == (row.initial, row.operations)
            strict_outcome = replay_runtime(
                strict_item["runtime"],
                compiler["response"],
                controller=controller,
                arm="strict_closed_loop",
                expected_transitions=len(row.operations),
                collector=collector,
                label=f"{controller}.strict[{index}].runtime",
            )
            strict_exact = compile_exact and strict_outcome.exact
            external = external_outcomes[row.identifier]
            trajectory_match = (
                strict_exact
                and external.exact
                and strict_outcome.emitted_states == external.emitted_states
            )
            compile_values.append(compile_exact)
            strict_values.append(strict_exact)
            trajectory_values.append(trajectory_match)
            gold_values.append(strict_outcome.final_answer == row.answer)
            gold_trajectory_values.append(strict_outcome.emitted_states == row.trajectory)
            strict_outcomes.append(strict_outcome)

            oracle_item = _exact_keys(oracle_value, {"id", "runtime"}, f"{controller}.oracle[{index}]")
            _require(oracle_item["id"] == row.identifier, f"{controller} oracle id mismatch")
            oracle_outcome = replay_runtime(
                oracle_item["runtime"],
                row.packet,
                controller=controller,
                arm="oracle_packet_loop",
                expected_transitions=len(row.operations),
                collector=collector,
                label=f"{controller}.oracle[{index}].runtime",
            )
            oracle_values.append(oracle_outcome.exact)
            oracle_gold_values.append(oracle_outcome.emitted_states == row.trajectory)

        teacher_raw = arm["teacher_forced_updater"]
        _require(isinstance(teacher_raw, list) and len(teacher_raw) == len(teacher_expectations), f"{controller} teacher rows mismatch")
        update_values: list[bool] = []
        update_nonterminal: list[bool] = []
        halt_values: list[bool] = []
        for index, (item_value, expected) in enumerate(zip(teacher_raw, teacher_expectations)):
            row, step_index, packet, observed, response = expected
            item = _exact_keys(item_value, {"id", "step_index", "packet", "observed", "call"}, f"{controller}.teacher[{index}]")
            _require(
                (item["id"], item["step_index"], item["packet"], item["observed"])
                == (row.identifier, step_index, packet, observed),
                f"{controller} teacher case reconstruction mismatch",
            )
            call = collector.call(
                item["call"],
                model=controller,
                arm="teacher_forced_updater",
                prompt=update_prompt(packet, observed),
                max_new=MAX_CONTROLLER_TOKENS,
                label=f"{controller}.teacher[{index}].call",
            )
            exact = call["response"].strip(" \t\r\n\f\v") == response
            update_values.append(exact)
            if parse_answer(response) is None:
                update_nonterminal.append(exact)
            else:
                halt_values.append(exact)

        swaps_raw = arm["packet_swaps"]
        _require(isinstance(swaps_raw, list) and len(swaps_raw) == len(swaps), f"{controller} swap rows mismatch")
        swap_values: list[bool] = []
        for index, (item_value, (original, donor)) in enumerate(zip(swaps_raw, swaps)):
            item = _exact_keys(
                item_value,
                {"original_id", "donor_id", "compiler", "intervened_packet", "runtime"},
                f"{controller}.swap[{index}]",
            )
            _require(
                (item["original_id"], item["donor_id"])
                == (original.identifier, donor.identifier),
                f"{controller} swap pairing mismatch",
            )
            _require(item["intervened_packet"] == donor.packet, f"{controller} swap packet mismatch")
            collector.call(
                item["compiler"],
                model=controller,
                arm="packet_swap",
                prompt=compiler_prompt(original.source),
                max_new=MAX_CONTROLLER_TOKENS,
                label=f"{controller}.swap[{index}].compiler",
            )
            outcome = replay_runtime(
                item["runtime"],
                donor.packet,
                controller=controller,
                arm="packet_swap",
                expected_transitions=len(donor.operations),
                collector=collector,
                label=f"{controller}.swap[{index}].runtime",
            )
            donor_external = external_outcomes[donor.identifier]
            original_external = external_outcomes[original.identifier]
            follows = (
                outcome.exact
                and donor_external.exact
                and outcome.emitted_states == donor_external.emitted_states
                and outcome.emitted_states != original_external.emitted_states
            )
            swap_values.append(follows)

        length_curve: dict[str, Any] = {}
        update_rate = Fraction(sum(update_values), len(update_values))
        halt_rate = Fraction(sum(halt_values), len(halt_values))
        for length in sorted({len(row.operations) for row in rows}):
            indices = [index for index, row in enumerate(rows) if len(row.operations) == length]
            compile_rate = Fraction(sum(compile_values[index] for index in indices), len(indices))
            observed_rate = Fraction(sum(strict_values[index] for index in indices), len(indices))
            predicted = compile_rate * halt_rate * (update_rate**length)
            length_curve[str(length)] = {
                "cases": len(indices),
                "compile_rate": float(compile_rate),
                "observed_strict_rate": float(observed_rate),
                "predicted_c_h_u_pow_l": float(predicted),
                "predicted_exact": f"{predicted.numerator}/{predicted.denominator}",
            }

        arm_results[controller] = {
            "compile_exact": _stratified_metric(compile_values, rows, contract),
            "strict_closed_loop": _stratified_metric(strict_values, rows, contract),
            "oracle_packet_loop": _stratified_metric(oracle_values, rows, contract),
            "oracle_gold_trajectory": _stratified_metric(oracle_gold_values, rows, contract),
            "external_trajectory_match": _stratified_metric(trajectory_values, rows, contract),
            "gold_answer": _stratified_metric(gold_values, rows, contract),
            "gold_trajectory": _stratified_metric(gold_trajectory_values, rows, contract),
            "conditional_update_exact": _metric(sum(update_values), len(update_values)),
            "nonterminal_update_exact": _metric(sum(update_nonterminal), len(update_nonterminal)),
            "halt_copy_exact": _metric(sum(halt_values), len(halt_values)),
            "packet_swap_follows_donor": _metric(sum(swap_values), len(swap_values)),
            "external_trajectory_mismatches": len(rows) - sum(trajectory_values),
            "length_curve": length_curve,
        }
        arm_vectors[controller] = {
            "compile": compile_values,
            "strict": strict_values,
        }

    collector.finalize_order(transcript["call_count"])
    expected_ledger = collector.expected_ledger(
        {
            "treatment": training_resources["treatment"],
            "sham": training_resources["sham"],
        }
    )
    _require(transcript["resource_ledger"] == expected_ledger, "resource ledger reconstruction mismatch")

    paired = {
        "compile_mcnemar": exact_mcnemar(
            arm_vectors["treatment"]["compile"], arm_vectors["sham"]["compile"]
        ),
        "strict_mcnemar": exact_mcnemar(
            arm_vectors["treatment"]["strict"], arm_vectors["sham"]["strict"]
        ),
    }
    treatment_compile = Fraction(
        arm_results["treatment"]["compile_exact"]["correct"], len(rows)
    )
    sham_compile = Fraction(arm_results["sham"]["compile_exact"]["correct"], len(rows))
    treatment_strict = Fraction(
        arm_results["treatment"]["strict_closed_loop"]["correct"], len(rows)
    )
    sham_strict = Fraction(
        arm_results["sham"]["strict_closed_loop"]["correct"], len(rows)
    )
    compile_delta = treatment_compile - sham_compile
    strict_delta = treatment_strict - sham_strict
    paired["compile_delta"] = {
        "value": float(compile_delta),
        "exact": f"{compile_delta.numerator}/{compile_delta.denominator}",
    }
    paired["strict_delta"] = {
        "value": float(strict_delta),
        "exact": f"{strict_delta.numerator}/{strict_delta.denominator}",
    }

    treatment_beats_every_stratum = all(
        arm_results["treatment"][metric]["by_stratum"][stratum]["correct"]
        > arm_results["sham"][metric]["by_stratum"][stratum]["correct"]
        for metric in ("compile_exact", "strict_closed_loop")
        for stratum in contract.strata
    )
    compile_p = Fraction(
        int(paired["compile_mcnemar"]["p_value_exact"].split("/")[0]),
        int(paired["compile_mcnemar"]["p_value_exact"].split("/")[1]),
    )
    strict_p = Fraction(
        int(paired["strict_mcnemar"]["p_value_exact"].split("/")[0]),
        int(paired["strict_mcnemar"]["p_value_exact"].split("/")[1]),
    )
    external_gold_count = sum(external_gold)
    gates = {
        "external_scheduler_gold": external_gold_count >= contract.external_gold_min,
        "oracle_packet_loop": arm_results["treatment"]["oracle_packet_loop"]["correct"] >= contract.oracle_min,
        "compile_exact": arm_results["treatment"]["compile_exact"]["correct"] >= contract.compile_min,
        "conditional_update": Fraction(
            arm_results["treatment"]["conditional_update_exact"]["correct"],
            arm_results["treatment"]["conditional_update_exact"]["total"],
        ) >= contract.update_min,
        "strict_closed_loop": arm_results["treatment"]["strict_closed_loop"]["correct"] >= contract.strict_min,
        "per_stratum_compile": all(
            arm_results["treatment"]["compile_exact"]["by_stratum"][stratum]["correct"]
            >= contract.per_stratum_compile_min
            for stratum in contract.strata
        ),
        "per_stratum_strict": all(
            arm_results["treatment"]["strict_closed_loop"]["by_stratum"][stratum]["correct"]
            >= contract.per_stratum_strict_min
            for stratum in contract.strata
        ),
        "external_trajectory_mismatches": arm_results["treatment"]["external_trajectory_mismatches"] <= contract.external_mismatch_max,
        "compile_delta": compile_delta >= contract.compile_delta_min,
        "strict_delta": strict_delta >= contract.strict_delta_min,
        "treatment_beats_sham_every_stratum": treatment_beats_every_stratum,
        "compile_mcnemar": compile_p < contract.mcnemar_max,
        "strict_mcnemar": strict_p < contract.mcnemar_max,
        "packet_swap": arm_results["treatment"]["packet_swap_follows_donor"]["correct"] >= contract.swap_min,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema": SCORE_SCHEMA,
        "seed": bindings.seed,
        "board_rows_sha256": bindings.board_rows_sha256,
        "external_scheduler_gold_answer": _metric(external_gold_count, len(rows)),
        "external_scheduler_gold_trajectory": _metric(
            sum(external_gold_trajectories), len(rows)
        ),
        "external_scheduler_atomic_transition": _metric(
            sum(external_step_values), len(external_step_values)
        ),
        "arms": arm_results,
        "paired": paired,
        "gates": gates,
        "resource_ledger_recomputed": expected_ledger,
    }


def score_artifacts(
    paths: Mapping[str, str | Path],
    *,
    bindings: FrozenBindings,
    contract: ScoreContract = ScoreContract(),
    codec: TokenCodec | None = None,
) -> dict[str, Any]:
    bindings.validate()
    _require(set(paths) == set(REQUIRED_ARTIFACTS), "artifact path set differs from frozen binding set")
    immutable = {
        "board",
        "treatment_data",
        "sham_data",
        "training_manifest",
        "audit",
        "training_resources",
        "controller_manifest",
        "executor_manifest",
        "transcript",
    }
    raw: dict[str, bytes] = {}
    for label in REQUIRED_ARTIFACTS:
        raw[label] = read_regular(paths[label], label, immutable=label in immutable)
        observed = sha256_bytes(raw[label])
        _require(observed == bindings.sha256[label], f"{label} hash mismatch")

    json_labels = {
        "prerequisite_confirmation",
        "board",
        "training_manifest",
        "audit",
        "training_resources",
        "controller_manifest",
        "executor_manifest",
        "transcript",
    }
    loaded = {label: load_json_bytes(raw[label], label) for label in json_labels}
    training_resources = validate_support_artifacts(loaded, bindings)
    if codec is None:
        codec = HuggingFaceTokenCodec(paths["tokenizer"])
    result = score_transcript(
        loaded["board"],
        loaded["transcript"],
        bindings=bindings,
        contract=contract,
        codec=codec,
        training_resources=training_resources,
    )
    result["artifact_sha256"] = dict(sorted(bindings.sha256.items()))
    result["scorer_sha256"] = sha256_file(Path(__file__).resolve())
    return result


def write_exclusive_immutable(path: str | Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating independent score")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require(destination.stat().st_mode & 0o222 == 0, "score output retained write bits")
    return sha256_bytes(payload)


def validate_production_source_paths(paths: Mapping[str, str | Path]) -> None:
    """Prevent package resolution or a same-hash alternate path from selecting code."""

    source_dir = Path(__file__).resolve().parent
    expected = {
        "protocol": source_dir / "residual_packet_protocol.py",
        "sft_encoding": source_dir / "sft_encoding.py",
        "evaluator": source_dir / "eval_residual_packet_v1.py",
    }
    for label, exact_path in expected.items():
        observed = Path(paths[label]).resolve(strict=True)
        _require(observed == exact_path, f"{label} is not the exact admitted source path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=FIT_SEEDS, required=True)
    for label in REQUIRED_ARTIFACTS:
        parser.add_argument("--" + label.replace("_", "-"), dest=label, required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    raise SystemExit(
        "RSP-C1 is permanently closed; preserve this module as audit evidence only"
    )
    args = build_parser().parse_args()
    bindings = PRODUCTION_BINDINGS[args.seed]
    # This is intentionally before artifact reads: no CLI hash can substitute
    # for a digest frozen in this independent scorer's source.
    bindings.validate()
    paths = {label: getattr(args, label) for label in REQUIRED_ARTIFACTS}
    validate_production_source_paths(paths)
    score = score_artifacts(paths, bindings=bindings)
    digest = write_exclusive_immutable(args.out, score)
    print(json.dumps({"out": args.out, "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
