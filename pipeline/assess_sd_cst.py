#!/usr/bin/env python3
"""Independently assess immutable SD-CST development evidence.

The assessor consumes JSON only.  It does not import model or training code,
trust pre-aggregated scores, or infer missing evidence.  Malformed, duplicate,
incomplete, unbound, or confirmation-contaminated evidence is rejected before
an assessment artifact is written.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVAL_SCHEMA = "r12_sd_cst_development_eval_v1"
CONFIG_SCHEMA = "r12_sd_cst_development_gate_config_v1"
ASSESSMENT_SCHEMA = "r12_sd_cst_development_assessment_v1"
DEVELOPMENT_SPLIT = "sd_cst_development"
EXPECTED_VARIANTS = (
    "canonical",
    "query_swap",
    "paraphrase",
    "binding_recode",
    "order_counterfactual",
    "stop_shift",
    "storage_order_shuffle",
    "post_halt_suffix",
)
DEPTHS = tuple(range(1, 7))
STATE_COUNT = 6
EVENT_COUNT = 8
ENTITY_COUNT = 3
KIND_COUNT = 3
AMOUNT_COUNT = 2
PERMUTATIONS = tuple(itertools.permutations(range(ENTITY_COUNT)))
SHA256_KEYS = (
    "architecture_sha256",
    "board_sha256",
    "checkpoint_sha256",
    "evaluator_sha256",
)
THRESHOLD_KEYS = (
    "compiler_initial_overall",
    "compiler_initial_min_variant",
    "compiler_initial_min_depth",
    "compiler_event_kind_overall",
    "compiler_event_kind_min_variant",
    "compiler_event_kind_min_depth",
    "compiler_event_identity_overall",
    "compiler_event_identity_min_variant",
    "compiler_event_identity_min_depth",
    "compiler_event_amount_overall",
    "compiler_event_amount_min_variant",
    "compiler_event_amount_min_depth",
    "compiler_exact_tape_overall",
    "compiler_exact_tape_min_variant",
    "compiler_exact_tape_min_depth",
    "autonomous_graph_overall",
    "autonomous_state_overall",
    "autonomous_answer_overall",
    "autonomous_graph_depth6",
    "autonomous_state_depth6",
    "autonomous_answer_depth6",
    "exact_tape_conditional_execution",
    "query_swap_state_invariance",
    "query_swap_answer_follow_query_conditional",
    "state_swap_separating_effect",
    "post_stop_suffix_invariance",
    "force_alive_suffix_oracle",
    "variant_graph_min",
    "variant_state_min",
    "variant_answer_min",
    "source_poison_bit_identity",
)


class AssessmentError(ValueError):
    """Evidence or configuration cannot support a valid assessment."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssessmentError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AssessmentError(f"non-finite JSON constant: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssessmentError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssessmentError(f"top-level JSON must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_ids_sha256(row_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(row_ids)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssessmentError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssessmentError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssessmentError(f"{path} must be a nonempty string")
    return value


def _integer(value: Any, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssessmentError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise AssessmentError(f"{path} must be at least {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise AssessmentError(f"{path} must be boolean")
    return value


def _category(value: Any, path: str, size: int) -> int:
    result = _integer(value, path)
    if not 0 <= result < size:
        raise AssessmentError(f"{path} must be in [0,{size - 1}]")
    return result


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssessmentError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AssessmentError(f"{path} must be a finite probability")
    return result


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AssessmentError(f"{path} must be a lowercase SHA-256 digest")
    return text


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise AssessmentError(f"{path} keys mismatch; missing={missing}, extra={extra}")


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": (float(correct) / total) if total else None,
    }


def _gate(value: float | None, floor: float) -> dict[str, Any]:
    return {
        "value": value,
        "floor": floor,
        "pass": value is not None and value >= floor,
    }


def _minimum_metric(groups: Mapping[str, Mapping[str, Any]], field: str) -> float | None:
    values = [
        stats[field]["accuracy"]
        for stats in groups.values()
        if stats[field]["accuracy"] is not None
    ]
    return min(values) if values and len(values) == len(groups) else None


def _decode_b64(value: Any, path: str) -> bytes:
    text = _string(value, path)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AssessmentError(f"{path} is not canonical base64") from error


def _apply_action(state_id: int, kind_id: int, role: int, amount_id: int) -> int:
    state = list(PERMUTATIONS[state_id])
    source = state.index(role)
    amount = amount_id + 1
    destination = max(0, source - amount) if kind_id == 0 else min(2, source + amount)
    value = state.pop(source)
    state.insert(destination, value)
    return PERMUTATIONS.index(tuple(state))


def _execute_program(initial_state_id: int, slots: Sequence[Mapping[str, Any]], force_alive: bool) -> int:
    state_id = initial_state_id
    alive = True
    for slot in slots:
        kind = int(slot["kind_id"])
        if kind == 2:
            if not force_alive:
                alive = False
            continue
        if alive:
            state_id = _apply_action(
                state_id,
                kind,
                int(slot["entity_role"]),
                int(slot["amount_id"]),
            )
    return state_id


def _execute_state_swap(
    receiver: Mapping[str, Any],
    donor: Mapping[str, Any],
    after_step: int,
) -> int:
    receiver_state = int(receiver["initial_state_id"])
    donor_state = int(donor["initial_state_id"])
    receiver_alive = donor_alive = True
    receiver_slots = receiver["event_slots"]
    donor_slots = donor["event_slots"]
    if not 0 <= after_step < EVENT_COUNT:
        raise AssessmentError("state-swap step is outside the fixed tape")
    for index in range(after_step + 1):
        for side in ("receiver", "donor"):
            slots = receiver_slots if side == "receiver" else donor_slots
            state = receiver_state if side == "receiver" else donor_state
            alive = receiver_alive if side == "receiver" else donor_alive
            kind = int(slots[index]["kind_id"])
            if kind == 2:
                alive = False
            elif alive:
                state = _apply_action(
                    state,
                    kind,
                    int(slots[index]["entity_role"]),
                    int(slots[index]["amount_id"]),
                )
            if side == "receiver":
                receiver_state, receiver_alive = state, alive
            else:
                donor_state, donor_alive = state, alive
    if not receiver_alive or not donor_alive:
        raise AssessmentError("state swap must occur before STOP in both programs")
    receiver_state = donor_state
    for slot in receiver_slots[after_step + 1:]:
        kind = int(slot["kind_id"])
        if kind == 2:
            receiver_alive = False
        elif receiver_alive:
            receiver_state = _apply_action(
                receiver_state,
                kind,
                int(slot["entity_role"]),
                int(slot["amount_id"]),
            )
    return receiver_state


def _answer(state_id: int, query_position: int) -> int:
    return int(PERMUTATIONS[state_id][query_position])


def _validate_slot(slot: Any, path: str, *, gold: bool) -> dict[str, Any]:
    value = _mapping(slot, path)
    required = {"kind_id", "entity_role", "amount_id"}
    if gold:
        required.add("identity_and_amount_scored")
    if not required <= set(value):
        raise AssessmentError(f"{path} lacks fields {sorted(required - set(value))}")
    result = {
        "kind_id": _category(value["kind_id"], f"{path}.kind_id", KIND_COUNT),
        "entity_role": _category(
            value["entity_role"], f"{path}.entity_role", ENTITY_COUNT
        ),
        "amount_id": _category(value["amount_id"], f"{path}.amount_id", AMOUNT_COUNT),
    }
    if gold:
        result["identity_and_amount_scored"] = _boolean(
            value["identity_and_amount_scored"],
            f"{path}.identity_and_amount_scored",
        )
        expected_scored = result["kind_id"] != 2
        if result["identity_and_amount_scored"] != expected_scored:
            raise AssessmentError(f"{path} has inconsistent STOP scoring mask")
    return result


def _validate_tape(value: Any, path: str, *, gold: bool) -> dict[str, Any]:
    tape = _mapping(value, path)
    if "initial_state_id" not in tape or "event_slots" not in tape:
        raise AssessmentError(f"{path} lacks initial_state_id/event_slots")
    slots = _list(tape["event_slots"], f"{path}.event_slots")
    if len(slots) != EVENT_COUNT:
        raise AssessmentError(f"{path}.event_slots must contain exactly {EVENT_COUNT} slots")
    result = {
        "initial_state_id": _category(
            tape["initial_state_id"], f"{path}.initial_state_id", STATE_COUNT
        ),
        "event_slots": [
            _validate_slot(slot, f"{path}.event_slots[{index}]", gold=gold)
            for index, slot in enumerate(slots)
        ],
    }
    if gold:
        stops = [index for index, slot in enumerate(result["event_slots"]) if slot["kind_id"] == 2]
        if len(stops) != 1:
            raise AssessmentError(f"{path} must contain exactly one STOP")
    return result


def _same_program(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["initial_state_id"] == right["initial_state_id"] and left["event_slots"] == right["event_slots"]


def _active_prefix(slots: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    result = []
    for slot in slots:
        if slot["kind_id"] == 2:
            break
        result.append((slot["kind_id"], slot["entity_role"], slot["amount_id"]))
    return tuple(result)


def _suffix(slots: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    seen_stop = False
    result = []
    for slot in slots:
        if slot["kind_id"] == 2:
            seen_stop = True
        elif seen_stop:
            result.append((slot["kind_id"], slot["entity_role"], slot["amount_id"]))
    return tuple(result)


def _actions(slots: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (slot["kind_id"], slot["entity_role"], slot["amount_id"])
        for slot in slots
        if slot["kind_id"] != 2
    )


def _stop_index(slots: Sequence[Mapping[str, Any]]) -> int:
    return next(index for index, slot in enumerate(slots) if slot["kind_id"] == 2)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA:
        raise AssessmentError("gate config schema mismatch")
    expected = _mapping(config.get("expected"), "config.expected")
    required_expected = {
        "eval_schema", "protocol", "split", "row_count", "family_count",
        "family_size", "row_ids_sha256", "depth_counts", "variants",
    }
    if not required_expected <= set(expected):
        raise AssessmentError("config.expected is incomplete")
    if expected["eval_schema"] != EVAL_SCHEMA:
        raise AssessmentError("config expected eval schema mismatch")
    variants = tuple(_list(expected["variants"], "config.expected.variants"))
    if variants != EXPECTED_VARIANTS:
        raise AssessmentError("config variants must match the frozen SD-CST order")
    depth_counts_raw = _mapping(expected["depth_counts"], "config.expected.depth_counts")
    if set(depth_counts_raw) != {str(depth) for depth in DEPTHS}:
        raise AssessmentError("config depth_counts must cover depths 1..6 exactly")
    depth_counts = {
        depth: _integer(depth_counts_raw[str(depth)], f"config.expected.depth_counts.{depth}", 1)
        for depth in DEPTHS
    }
    row_count = _integer(expected["row_count"], "config.expected.row_count", 1)
    if sum(depth_counts.values()) != row_count:
        raise AssessmentError("config depth counts do not sum to row_count")
    family_count = _integer(expected["family_count"], "config.expected.family_count", 1)
    family_size = _integer(expected["family_size"], "config.expected.family_size", 1)
    if family_size != len(EXPECTED_VARIANTS) or family_count * family_size != row_count:
        raise AssessmentError("config family shape does not match row_count")

    thresholds_raw = _mapping(config.get("thresholds"), "config.thresholds")
    _exact_keys(thresholds_raw, THRESHOLD_KEYS, "config.thresholds")
    thresholds = {
        key: _probability(thresholds_raw[key], f"config.thresholds.{key}")
        for key in THRESHOLD_KEYS
    }
    if thresholds["source_poison_bit_identity"] != 1.0:
        raise AssessmentError(
            "source_poison_bit_identity must preregister exact byte identity"
        )
    controls_raw = _mapping(config.get("controls"), "config.controls")
    if not controls_raw:
        raise AssessmentError("config.controls cannot be empty")
    controls: dict[str, dict[str, Any]] = {}
    for name, raw in controls_raw.items():
        rule = _mapping(raw, f"config.controls.{name}")
        _exact_keys(rule, ("direction", "threshold", "min_cases"), f"config.controls.{name}")
        direction = rule["direction"]
        if direction not in ("at_least", "at_most"):
            raise AssessmentError(f"config.controls.{name}.direction is invalid")
        controls[name] = {
            "direction": direction,
            "threshold": _probability(rule["threshold"], f"config.controls.{name}.threshold"),
            "min_cases": _integer(rule["min_cases"], f"config.controls.{name}.min_cases", 1),
        }

    expected_hashes_raw = _mapping(config.get("expected_artifact_hashes"), "config.expected_artifact_hashes")
    _exact_keys(expected_hashes_raw, SHA256_KEYS, "config.expected_artifact_hashes")
    expected_hashes = {
        key: _sha256(expected_hashes_raw[key], f"config.expected_artifact_hashes.{key}")
        for key in SHA256_KEYS
    }
    expected_ledger_hash = _sha256(
        config.get("expected_access_ledger_sha256"),
        "config.expected_access_ledger_sha256",
    )
    cap = _integer(config.get("parameter_cap"), "config.parameter_cap", 1)
    if cap > 150_000_000:
        raise AssessmentError("parameter cap cannot exceed 150,000,000")
    if _integer(config.get("confirmation_accesses"), "config.confirmation_accesses", 0) != 0:
        raise AssessmentError("gate config must preregister zero confirmation access")
    split = _string(expected["split"], "config.expected.split")
    if split != DEVELOPMENT_SPLIT:
        raise AssessmentError("SD-CST development assessor requires sd_cst_development")
    return {
        "protocol": _string(expected["protocol"], "config.expected.protocol"),
        "split": split,
        "row_count": row_count,
        "family_count": family_count,
        "family_size": family_size,
        "row_ids_sha256": _sha256(expected["row_ids_sha256"], "config.expected.row_ids_sha256"),
        "depth_counts": depth_counts,
        "thresholds": thresholds,
        "controls": controls,
        "expected_hashes": expected_hashes,
        "expected_ledger_hash": expected_ledger_hash,
        "parameter_cap": cap,
    }


def _validate_certificates(raw: Any) -> tuple[dict[str, Any], dict[str, bool]]:
    certs = _mapping(raw, "eval.certificates")
    required = ("motor_state_action", "motor_stop", "dead_invariance", "reader")
    _exact_keys(certs, required, "eval.certificates")

    state_action_rows = _list(certs["motor_state_action"], "eval.certificates.motor_state_action")
    expected_actions = {
        (state, kind, role, amount)
        for state in range(STATE_COUNT)
        for kind in range(2)
        for role in range(ENTITY_COUNT)
        for amount in range(AMOUNT_COUNT)
    }
    seen_actions: set[tuple[int, int, int, int]] = set()
    state_action_correct = 0
    for index, raw_row in enumerate(state_action_rows):
        path = f"eval.certificates.motor_state_action[{index}]"
        row = _mapping(raw_row, path)
        key = (
            _category(row.get("state_id"), f"{path}.state_id", STATE_COUNT),
            _category(row.get("kind_id"), f"{path}.kind_id", 2),
            _category(row.get("entity_role"), f"{path}.entity_role", ENTITY_COUNT),
            _category(row.get("amount_id"), f"{path}.amount_id", AMOUNT_COUNT),
        )
        if key in seen_actions:
            raise AssessmentError(f"duplicate motor state-action certificate: {key}")
        seen_actions.add(key)
        predicted = _category(row.get("predicted_state_id"), f"{path}.predicted_state_id", STATE_COUNT)
        alive = _boolean(row.get("predicted_alive"), f"{path}.predicted_alive")
        state_action_correct += int(predicted == _apply_action(*key) and alive)
    if seen_actions != expected_actions:
        raise AssessmentError("motor state-action certificate is not the exact 72-cell product")

    stop_rows = _list(certs["motor_stop"], "eval.certificates.motor_stop")
    seen_stop: set[int] = set()
    stop_correct = 0
    for index, raw_row in enumerate(stop_rows):
        path = f"eval.certificates.motor_stop[{index}]"
        row = _mapping(raw_row, path)
        state = _category(row.get("state_id"), f"{path}.state_id", STATE_COUNT)
        if state in seen_stop:
            raise AssessmentError(f"duplicate STOP certificate state: {state}")
        seen_stop.add(state)
        predicted = _category(row.get("predicted_state_id"), f"{path}.predicted_state_id", STATE_COUNT)
        alive = _boolean(row.get("predicted_alive"), f"{path}.predicted_alive")
        stop_correct += int(predicted == state and not alive)
    if seen_stop != set(range(STATE_COUNT)):
        raise AssessmentError("STOP certificate is not the exact six-state product")

    action_domain = {
        (kind, role, amount)
        for kind in range(2)
        for role in range(ENTITY_COUNT)
        for amount in range(AMOUNT_COUNT)
    } | {(2, 0, 0)}
    expected_dead = {
        (state, kind, role, amount)
        for state in range(STATE_COUNT)
        for kind, role, amount in action_domain
    }
    dead_rows = _list(certs["dead_invariance"], "eval.certificates.dead_invariance")
    seen_dead: set[tuple[int, int, int, int]] = set()
    dead_correct = 0
    for index, raw_row in enumerate(dead_rows):
        path = f"eval.certificates.dead_invariance[{index}]"
        row = _mapping(raw_row, path)
        key = (
            _category(row.get("state_id"), f"{path}.state_id", STATE_COUNT),
            _category(row.get("kind_id"), f"{path}.kind_id", KIND_COUNT),
            _category(row.get("entity_role"), f"{path}.entity_role", ENTITY_COUNT),
            _category(row.get("amount_id"), f"{path}.amount_id", AMOUNT_COUNT),
        )
        if key not in expected_dead:
            raise AssessmentError(f"dead-invariance action is outside the 13-action domain: {key}")
        if key in seen_dead:
            raise AssessmentError(f"duplicate dead-invariance certificate: {key}")
        seen_dead.add(key)
        predicted = _category(row.get("predicted_state_id"), f"{path}.predicted_state_id", STATE_COUNT)
        alive = _boolean(row.get("predicted_alive"), f"{path}.predicted_alive")
        dead_correct += int(predicted == key[0] and not alive)
    if seen_dead != expected_dead:
        raise AssessmentError("dead-invariance certificate is not the exact 78-cell product")

    reader_rows = _list(certs["reader"], "eval.certificates.reader")
    expected_reader = {
        (state, query)
        for state in range(STATE_COUNT)
        for query in range(ENTITY_COUNT)
    }
    seen_reader: set[tuple[int, int]] = set()
    reader_correct = 0
    for index, raw_row in enumerate(reader_rows):
        path = f"eval.certificates.reader[{index}]"
        row = _mapping(raw_row, path)
        key = (
            _category(row.get("state_id"), f"{path}.state_id", STATE_COUNT),
            _category(row.get("query_position"), f"{path}.query_position", ENTITY_COUNT),
        )
        if key in seen_reader:
            raise AssessmentError(f"duplicate reader certificate: {key}")
        seen_reader.add(key)
        predicted = _category(row.get("predicted_answer_role"), f"{path}.predicted_answer_role", ENTITY_COUNT)
        reader_correct += int(predicted == _answer(*key))
    if seen_reader != expected_reader:
        raise AssessmentError("reader certificate is not the exact 18-cell product")

    metrics = {
        "motor_state_action": _metric(state_action_correct, 72),
        "motor_stop": _metric(stop_correct, 6),
        "dead_invariance": _metric(dead_correct, 78),
        "reader": _metric(reader_correct, 18),
    }
    gates = {name: metric["correct"] == metric["total"] for name, metric in metrics.items()}
    return metrics, gates


def assess(eval_payload: Mapping[str, Any], gate_config: Mapping[str, Any]) -> dict[str, Any]:
    config = _validate_config(gate_config)
    if eval_payload.get("schema") != EVAL_SCHEMA:
        raise AssessmentError("evaluation schema mismatch")
    if eval_payload.get("protocol") != config["protocol"]:
        raise AssessmentError("evaluation protocol mismatch")
    if eval_payload.get("split") != config["split"]:
        raise AssessmentError("evaluation split mismatch")

    custody = _mapping(eval_payload.get("custody"), "eval.custody")
    development_accesses = _integer(
        custody.get("development_accesses"), "eval.custody.development_accesses", 0
    )
    confirmation_accesses = _integer(
        custody.get("confirmation_accesses"), "eval.custody.confirmation_accesses", 0
    )
    confirmation_opened = _boolean(
        custody.get("confirmation_opened"), "eval.custody.confirmation_opened"
    )
    if confirmation_accesses != 0 or confirmation_opened:
        raise AssessmentError("evaluation accessed sealed confirmation evidence")
    if development_accesses != 1:
        raise AssessmentError("evaluation must consume development exactly once")
    access_ledger = _mapping(custody.get("access_ledger"), "eval.custody.access_ledger")
    access_ledger_hash = _sha256(
        access_ledger.get("sha256"), "eval.custody.access_ledger.sha256",
    )
    if access_ledger_hash != config["expected_ledger_hash"]:
        raise AssessmentError("development access ledger does not match preregistration")

    artifact_hashes_raw = _mapping(eval_payload.get("artifact_hashes"), "eval.artifact_hashes")
    _exact_keys(artifact_hashes_raw, SHA256_KEYS, "eval.artifact_hashes")
    artifact_hashes = {
        key: _sha256(artifact_hashes_raw[key], f"eval.artifact_hashes.{key}")
        for key in SHA256_KEYS
    }
    if artifact_hashes != config["expected_hashes"]:
        raise AssessmentError("evaluation artifact hashes do not match preregistration")
    if artifact_hashes["evaluator_sha256"] != sha256_file(Path(__file__).resolve()):
        raise AssessmentError("evaluator hash does not bind this assessor source")

    parameters = _mapping(eval_payload.get("parameters"), "eval.parameters")
    required_parameters = (
        "base", "compiler", "motor", "reader", "total",
        "excluded_trainable_parameters", "complete_system",
    )
    _exact_keys(parameters, required_parameters, "eval.parameters")
    components = {
        name: _integer(parameters[name], f"eval.parameters.{name}", 0)
        for name in ("base", "compiler", "motor", "reader")
    }
    total_parameters = _integer(parameters["total"], "eval.parameters.total", 1)
    if sum(components.values()) != total_parameters:
        raise AssessmentError("complete-system parameter total does not equal component sum")
    excluded_parameters = _integer(
        parameters["excluded_trainable_parameters"],
        "eval.parameters.excluded_trainable_parameters",
        0,
    )
    complete_system = _boolean(parameters["complete_system"], "eval.parameters.complete_system")

    raw_rows = _list(eval_payload.get("rows"), "eval.rows")
    if len(raw_rows) != config["row_count"]:
        raise AssessmentError(
            f"evaluation row count mismatch: {len(raw_rows)} != {config['row_count']}"
        )
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    family_members: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    depth_counts: Counter[int] = Counter()
    for index, raw_row in enumerate(raw_rows):
        path = f"eval.rows[{index}]"
        row = _mapping(raw_row, path)
        row_id = _string(row.get("id"), f"{path}.id")
        if row_id in ids:
            raise AssessmentError(f"duplicate evaluation row id: {row_id}")
        ids.add(row_id)
        family_id = _string(row.get("family_id"), f"{path}.family_id")
        variant = _string(row.get("variant"), f"{path}.variant")
        if variant not in EXPECTED_VARIANTS:
            raise AssessmentError(f"{path}.variant is not registered")
        if variant in family_members[family_id]:
            raise AssessmentError(f"duplicate {variant} row in family {family_id}")
        depth = _integer(row.get("depth"), f"{path}.depth")
        if depth not in DEPTHS:
            raise AssessmentError(f"{path}.depth must be in 1..6")
        gold = _validate_tape(row.get("compiler_gold"), f"{path}.compiler_gold", gold=True)
        prediction = _validate_tape(
            row.get("compiler_prediction"), f"{path}.compiler_prediction", gold=False
        )
        stop_index = next(
            slot_index
            for slot_index, slot in enumerate(gold["event_slots"])
            if slot["kind_id"] == 2
        )
        if stop_index != depth:
            raise AssessmentError(f"{path} depth does not equal the STOP prefix length")
        query_gold = _category(
            row.get("late_query_gold"), f"{path}.late_query_gold", ENTITY_COUNT
        )
        query_prediction = _category(
            row.get("late_query_prediction"),
            f"{path}.late_query_prediction",
            ENTITY_COUNT,
        )
        oracle = _mapping(row.get("oracle"), f"{path}.oracle")
        oracle_state = _category(
            oracle.get("final_state_id"), f"{path}.oracle.final_state_id", STATE_COUNT
        )
        oracle_answer = _category(
            oracle.get("answer_role"), f"{path}.oracle.answer_role", ENTITY_COUNT
        )
        independently_executed = _execute_program(
            gold["initial_state_id"], gold["event_slots"], force_alive=False
        )
        if oracle_state != independently_executed or oracle_answer != _answer(oracle_state, query_gold):
            raise AssessmentError(f"{path} oracle disagrees with the independent executor")
        autonomous = _mapping(row.get("autonomous"), f"{path}.autonomous")
        predicted_state = _category(
            autonomous.get("final_state_id"),
            f"{path}.autonomous.final_state_id",
            STATE_COUNT,
        )
        predicted_answer = _category(
            autonomous.get("answer_role"),
            f"{path}.autonomous.answer_role",
            ENTITY_COUNT,
        )
        interventions = _mapping(row.get("interventions"), f"{path}.interventions")
        normalized = {
            "id": row_id,
            "family_id": family_id,
            "variant": variant,
            "depth": depth,
            "gold": gold,
            "prediction": prediction,
            "query_gold": query_gold,
            "query_prediction": query_prediction,
            "oracle_state": oracle_state,
            "oracle_answer": oracle_answer,
            "predicted_state": predicted_state,
            "predicted_answer": predicted_answer,
            "interventions": interventions,
        }
        rows.append(normalized)
        family_members[family_id][variant] = normalized
        depth_counts[depth] += 1

    if row_ids_sha256(ids) != config["row_ids_sha256"]:
        raise AssessmentError("evaluation row IDs do not match preregistration")
    if dict(depth_counts) != config["depth_counts"]:
        raise AssessmentError("evaluation depth counts do not match preregistration")
    if len(family_members) != config["family_count"]:
        raise AssessmentError("evaluation family count does not match preregistration")
    for family_id, members in family_members.items():
        if set(members) != set(EXPECTED_VARIANTS):
            missing = sorted(set(EXPECTED_VARIANTS) - set(members))
            extra = sorted(set(members) - set(EXPECTED_VARIANTS))
            raise AssessmentError(
                f"family {family_id} is incomplete; missing={missing}, extra={extra}"
            )

    field_names = ("initial", "event_kind", "event_identity", "event_amount", "exact_tape", "query")
    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_variant: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_depth: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    row_flags: dict[str, dict[str, bool]] = {}
    for row in rows:
        gold = row["gold"]
        prediction = row["prediction"]
        initial_exact = gold["initial_state_id"] == prediction["initial_state_id"]
        slot_exact: list[bool] = []
        for gold_slot, predicted_slot in zip(gold["event_slots"], prediction["event_slots"]):
            kind_exact = gold_slot["kind_id"] == predicted_slot["kind_id"]
            scored = gold_slot["identity_and_amount_scored"]
            identity_exact = gold_slot["entity_role"] == predicted_slot["entity_role"]
            amount_exact = gold_slot["amount_id"] == predicted_slot["amount_id"]
            values = {
                "event_kind": kind_exact,
                "event_identity": identity_exact,
                "event_amount": amount_exact,
            }
            for field, exact in values.items():
                if field != "event_kind" and not scored:
                    continue
                for target in (
                    aggregate[field],
                    by_family[row["family_id"]][field],
                    by_variant[row["variant"]][field],
                    by_depth[row["depth"]][field],
                ):
                    target["total"] += 1
                    target["correct"] += int(exact)
            slot_exact.append(kind_exact and (not scored or (identity_exact and amount_exact)))
        tape_exact = initial_exact and all(slot_exact)
        query_exact = row["query_gold"] == row["query_prediction"]
        scalar_values = {
            "initial": initial_exact,
            "exact_tape": tape_exact,
            "query": query_exact,
        }
        for field, exact in scalar_values.items():
            for target in (
                aggregate[field],
                by_family[row["family_id"]][field],
                by_variant[row["variant"]][field],
                by_depth[row["depth"]][field],
            ):
                target["total"] += 1
                target["correct"] += int(exact)
        state_exact = row["predicted_state"] == row["oracle_state"]
        answer_exact = row["predicted_answer"] == row["oracle_answer"]
        row_flags[row["id"]] = {
            "graph": tape_exact,
            "state": state_exact,
            "answer": answer_exact,
            "query": query_exact,
        }

    def compiler_group(source: Mapping[Any, Mapping[str, Counter[str]]]) -> dict[str, Any]:
        return {
            str(group): {
                field: _metric(stats[field]["correct"], stats[field]["total"])
                for field in field_names
            }
            for group, stats in sorted(source.items(), key=lambda item: str(item[0]))
        }

    compiler_overall = {
        field: _metric(aggregate[field]["correct"], aggregate[field]["total"])
        for field in field_names
    }
    compiler_family = compiler_group(by_family)
    compiler_variant = compiler_group(by_variant)
    compiler_depth = compiler_group(by_depth)

    auto_overall = {
        field: _metric(sum(flags[field] for flags in row_flags.values()), len(rows))
        for field in ("graph", "state", "answer")
    }
    auto_variant = {
        variant: {
            field: _metric(
                sum(row_flags[row["id"]][field] for row in rows if row["variant"] == variant),
                sum(row["variant"] == variant for row in rows),
            )
            for field in ("graph", "state", "answer")
        }
        for variant in EXPECTED_VARIANTS
    }
    auto_depth = {
        str(depth): {
            field: _metric(
                sum(row_flags[row["id"]][field] for row in rows if row["depth"] == depth),
                depth_counts[depth],
            )
            for field in ("graph", "state", "answer")
        }
        for depth in DEPTHS
    }
    exact_tape_rows = [
        row for row in rows
        if row_flags[row["id"]]["graph"] and row_flags[row["id"]]["query"]
    ]
    exact_tape_execution = _metric(
        sum(
            row_flags[row["id"]]["state"] and row_flags[row["id"]]["answer"]
            for row in exact_tape_rows
        ),
        len(exact_tape_rows),
    )

    query_invariant = 0
    query_conditional_correct = 0
    query_conditional_total = 0
    post_stop_invariant = 0
    post_stop_total = 0
    force_alive_correct = 0
    force_alive_total = 0
    state_swap_correct = 0
    state_swap_total = 0
    for family_id, members in family_members.items():
        canonical = members["canonical"]
        query_swap = members["query_swap"]
        if not _same_program(canonical["gold"], query_swap["gold"]):
            raise AssessmentError(f"family {family_id} query-swap programs differ")
        if canonical["query_gold"] == query_swap["query_gold"]:
            raise AssessmentError(f"family {family_id} query swap does not change query")
        if canonical["oracle_state"] != query_swap["oracle_state"] or canonical["oracle_answer"] == query_swap["oracle_answer"]:
            raise AssessmentError(f"family {family_id} query swap is not separating")

        for variant in ("paraphrase", "binding_recode", "storage_order_shuffle"):
            equivalent = members[variant]
            if (
                not _same_program(canonical["gold"], equivalent["gold"])
                or canonical["query_gold"] != equivalent["query_gold"]
                or canonical["oracle_state"] != equivalent["oracle_state"]
                or canonical["oracle_answer"] != equivalent["oracle_answer"]
            ):
                raise AssessmentError(
                    f"family {family_id} {variant} abstract semantics differ"
                )

        order = members["order_counterfactual"]
        canonical_actions = _actions(canonical["gold"]["event_slots"])
        order_actions = _actions(order["gold"]["event_slots"])
        if (
            canonical["gold"]["initial_state_id"] != order["gold"]["initial_state_id"]
            or Counter(canonical_actions) != Counter(order_actions)
            or canonical_actions == order_actions
            or _stop_index(canonical["gold"]["event_slots"])
            != _stop_index(order["gold"]["event_slots"])
            or canonical["query_gold"] != order["query_gold"]
            or canonical["oracle_answer"] == order["oracle_answer"]
        ):
            raise AssessmentError(f"family {family_id} order counterfactual is invalid")

        stop_shift = members["stop_shift"]
        if (
            canonical["gold"]["initial_state_id"]
            != stop_shift["gold"]["initial_state_id"]
            or canonical_actions != _actions(stop_shift["gold"]["event_slots"])
            or _stop_index(canonical["gold"]["event_slots"])
            == _stop_index(stop_shift["gold"]["event_slots"])
            or canonical["query_gold"] != stop_shift["query_gold"]
            or canonical["oracle_answer"] == stop_shift["oracle_answer"]
        ):
            raise AssessmentError(f"family {family_id} STOP shift is invalid")
        query_invariant += int(canonical["predicted_state"] == query_swap["predicted_state"])
        conditional = all(
            (
                row_flags[canonical["id"]]["graph"],
                row_flags[query_swap["id"]]["graph"],
                row_flags[canonical["id"]]["query"],
                row_flags[query_swap["id"]]["query"],
            )
        )
        if conditional:
            query_conditional_total += 1
            query_conditional_correct += int(
                row_flags[canonical["id"]]["answer"]
                and row_flags[query_swap["id"]]["answer"]
            )

        suffix = members["post_halt_suffix"]
        if (
            canonical["gold"]["initial_state_id"] != suffix["gold"]["initial_state_id"]
            or _active_prefix(canonical["gold"]["event_slots"])
            != _active_prefix(suffix["gold"]["event_slots"])
            or _suffix(canonical["gold"]["event_slots"])
            == _suffix(suffix["gold"]["event_slots"])
            or canonical["query_gold"] != suffix["query_gold"]
            or canonical["oracle_state"] != suffix["oracle_state"]
            or canonical["oracle_answer"] != suffix["oracle_answer"]
        ):
            raise AssessmentError(f"family {family_id} post-STOP pair is invalid")
        suffix_eligible = all((
            row_flags[canonical["id"]]["graph"],
            row_flags[suffix["id"]]["graph"],
            row_flags[canonical["id"]]["query"],
            row_flags[suffix["id"]]["query"],
        ))
        if suffix_eligible:
            post_stop_total += 1
            post_stop_invariant += int(
                canonical["predicted_state"] == suffix["predicted_state"]
                and canonical["predicted_answer"] == suffix["predicted_answer"]
            )
        force_alive = _mapping(
            suffix["interventions"].get("force_alive"),
            f"row {suffix['id']}.interventions.force_alive",
        )
        force_state = _category(
            force_alive.get("final_state_id"),
            f"row {suffix['id']}.interventions.force_alive.final_state_id",
            STATE_COUNT,
        )
        force_answer = _category(
            force_alive.get("answer_role"),
            f"row {suffix['id']}.interventions.force_alive.answer_role",
            ENTITY_COUNT,
        )
        full_state = _execute_program(
            suffix["gold"]["initial_state_id"],
            suffix["gold"]["event_slots"],
            force_alive=True,
        )
        full_answer = _answer(full_state, suffix["query_gold"])
        if full_state == suffix["oracle_state"]:
            raise AssessmentError(f"family {family_id} force-alive suffix is not separating")
        if suffix_eligible:
            force_alive_total += 1
            force_alive_correct += int(
                force_state == full_state and force_answer == full_answer
            )

        state_swap = _mapping(
            canonical["interventions"].get("state_swap"),
            f"row {canonical['id']}.interventions.state_swap",
        )
        donor_id = _string(
            state_swap.get("donor_id"),
            f"row {canonical['id']}.interventions.state_swap.donor_id",
        )
        donor = next((row for row in rows if row["id"] == donor_id), None)
        if donor is None or donor["variant"] != "canonical":
            raise AssessmentError(f"family {family_id} state-swap donor is missing/noncanonical")
        after_step = _integer(
            state_swap.get("after_step"),
            f"row {canonical['id']}.interventions.state_swap.after_step",
        )
        expected_state = _execute_state_swap(
            canonical["gold"], donor["gold"], after_step,
        )
        expected_answer = _answer(expected_state, canonical["query_gold"])
        if expected_state == canonical["oracle_state"] or expected_answer == canonical["oracle_answer"]:
            raise AssessmentError(f"family {family_id} state swap is not separating")
        swap_state = _category(
            state_swap.get("final_state_id"),
            f"row {canonical['id']}.interventions.state_swap.final_state_id",
            STATE_COUNT,
        )
        swap_answer = _category(
            state_swap.get("answer_role"),
            f"row {canonical['id']}.interventions.state_swap.answer_role",
            ENTITY_COUNT,
        )
        state_swap_eligible = all((
            row_flags[canonical["id"]]["graph"],
            row_flags[canonical["id"]]["query"],
            row_flags[donor["id"]]["graph"],
        ))
        if state_swap_eligible:
            state_swap_total += 1
            state_swap_correct += int(
                swap_state == expected_state and swap_answer == expected_answer
            )

    family_count = len(family_members)
    causal = {
        "query_swap_state_invariance": _metric(query_invariant, family_count),
        "query_swap_answer_follow_query_conditional": _metric(
            query_conditional_correct, query_conditional_total
        ),
        "state_swap_separating_effect": _metric(state_swap_correct, state_swap_total),
        "post_stop_suffix_invariance": _metric(post_stop_invariant, post_stop_total),
        "force_alive_suffix_oracle": _metric(force_alive_correct, force_alive_total),
    }
    causal["query_swap_answer_follow_query_conditional"]["eligible_fraction"] = (
        query_conditional_total / family_count
    )
    causal["state_swap_separating_effect"]["eligible_fraction"] = (
        state_swap_total / family_count
    )
    causal["post_stop_suffix_invariance"]["eligible_fraction"] = (
        post_stop_total / family_count
    )
    causal["force_alive_suffix_oracle"]["eligible_fraction"] = (
        force_alive_total / family_count
    )

    poison_rows = _list(eval_payload.get("source_poison"), "eval.source_poison")
    poison_seen: set[str] = set()
    poison_correct = 0
    for index, raw in enumerate(poison_rows):
        path = f"eval.source_poison[{index}]"
        record = _mapping(raw, path)
        row_id = _string(record.get("id"), f"{path}.id")
        if row_id not in ids:
            raise AssessmentError(f"source-poison record has unknown row id: {row_id}")
        if row_id in poison_seen:
            raise AssessmentError(f"duplicate source-poison row id: {row_id}")
        poison_seen.add(row_id)
        equal = True
        for field in ("program_tape", "late_query", "rollout"):
            clean = _decode_b64(record.get(f"clean_{field}_b64"), f"{path}.clean_{field}_b64")
            poisoned = _decode_b64(record.get(f"poisoned_{field}_b64"), f"{path}.poisoned_{field}_b64")
            equal &= clean == poisoned
        poison_correct += int(equal)
    if poison_seen != ids:
        raise AssessmentError("source-poison evidence does not cover every evaluation row exactly once")
    poison_metric = _metric(poison_correct, len(rows))

    certificate_metrics, certificate_gates = _validate_certificates(
        eval_payload.get("certificates")
    )

    eval_controls = _mapping(eval_payload.get("controls"), "eval.controls")
    if set(eval_controls) != set(config["controls"]):
        raise AssessmentError("evaluation controls do not match preregistration")
    control_metrics: dict[str, Any] = {}
    control_gates: dict[str, bool] = {}
    for name, rule in config["controls"].items():
        value = _mapping(eval_controls[name], f"eval.controls.{name}")
        _exact_keys(value, ("cases", "correct"), f"eval.controls.{name}")
        cases = _integer(value["cases"], f"eval.controls.{name}.cases", 1)
        correct = _integer(value["correct"], f"eval.controls.{name}.correct", 0)
        if correct > cases or cases < rule["min_cases"]:
            raise AssessmentError(f"eval.controls.{name} has invalid/insufficient counts")
        accuracy = correct / cases
        passed = accuracy >= rule["threshold"] if rule["direction"] == "at_least" else accuracy <= rule["threshold"]
        control_metrics[name] = _metric(correct, cases) | dict(rule)
        control_metrics[name]["pass"] = passed
        control_gates[name] = passed

    thresholds = config["thresholds"]
    gates: dict[str, Any] = {
        "compiler_initial_overall": _gate(compiler_overall["initial"]["accuracy"], thresholds["compiler_initial_overall"]),
        "compiler_initial_min_variant": _gate(_minimum_metric(compiler_variant, "initial"), thresholds["compiler_initial_min_variant"]),
        "compiler_initial_min_depth": _gate(_minimum_metric(compiler_depth, "initial"), thresholds["compiler_initial_min_depth"]),
        "compiler_event_kind_overall": _gate(compiler_overall["event_kind"]["accuracy"], thresholds["compiler_event_kind_overall"]),
        "compiler_event_kind_min_variant": _gate(_minimum_metric(compiler_variant, "event_kind"), thresholds["compiler_event_kind_min_variant"]),
        "compiler_event_kind_min_depth": _gate(_minimum_metric(compiler_depth, "event_kind"), thresholds["compiler_event_kind_min_depth"]),
        "compiler_event_identity_overall": _gate(compiler_overall["event_identity"]["accuracy"], thresholds["compiler_event_identity_overall"]),
        "compiler_event_identity_min_variant": _gate(_minimum_metric(compiler_variant, "event_identity"), thresholds["compiler_event_identity_min_variant"]),
        "compiler_event_identity_min_depth": _gate(_minimum_metric(compiler_depth, "event_identity"), thresholds["compiler_event_identity_min_depth"]),
        "compiler_event_amount_overall": _gate(compiler_overall["event_amount"]["accuracy"], thresholds["compiler_event_amount_overall"]),
        "compiler_event_amount_min_variant": _gate(_minimum_metric(compiler_variant, "event_amount"), thresholds["compiler_event_amount_min_variant"]),
        "compiler_event_amount_min_depth": _gate(_minimum_metric(compiler_depth, "event_amount"), thresholds["compiler_event_amount_min_depth"]),
        "compiler_exact_tape_overall": _gate(compiler_overall["exact_tape"]["accuracy"], thresholds["compiler_exact_tape_overall"]),
        "compiler_exact_tape_min_variant": _gate(_minimum_metric(compiler_variant, "exact_tape"), thresholds["compiler_exact_tape_min_variant"]),
        "compiler_exact_tape_min_depth": _gate(_minimum_metric(compiler_depth, "exact_tape"), thresholds["compiler_exact_tape_min_depth"]),
        "autonomous_graph_overall": _gate(auto_overall["graph"]["accuracy"], thresholds["autonomous_graph_overall"]),
        "autonomous_state_overall": _gate(auto_overall["state"]["accuracy"], thresholds["autonomous_state_overall"]),
        "autonomous_answer_overall": _gate(auto_overall["answer"]["accuracy"], thresholds["autonomous_answer_overall"]),
        "autonomous_graph_depth6": _gate(auto_depth["6"]["graph"]["accuracy"], thresholds["autonomous_graph_depth6"]),
        "autonomous_state_depth6": _gate(auto_depth["6"]["state"]["accuracy"], thresholds["autonomous_state_depth6"]),
        "autonomous_answer_depth6": _gate(auto_depth["6"]["answer"]["accuracy"], thresholds["autonomous_answer_depth6"]),
        "exact_tape_conditional_execution": _gate(exact_tape_execution["accuracy"], thresholds["exact_tape_conditional_execution"]),
        "query_swap_state_invariance": _gate(causal["query_swap_state_invariance"]["accuracy"], thresholds["query_swap_state_invariance"]),
        "query_swap_answer_follow_query_conditional": _gate(causal["query_swap_answer_follow_query_conditional"]["accuracy"], thresholds["query_swap_answer_follow_query_conditional"]),
        "state_swap_separating_effect": _gate(causal["state_swap_separating_effect"]["accuracy"], thresholds["state_swap_separating_effect"]),
        "post_stop_suffix_invariance": _gate(causal["post_stop_suffix_invariance"]["accuracy"], thresholds["post_stop_suffix_invariance"]),
        "force_alive_suffix_oracle": _gate(causal["force_alive_suffix_oracle"]["accuracy"], thresholds["force_alive_suffix_oracle"]),
        "variant_graph_min": _gate(min(stats["graph"]["accuracy"] for stats in auto_variant.values()), thresholds["variant_graph_min"]),
        "variant_state_min": _gate(min(stats["state"]["accuracy"] for stats in auto_variant.values()), thresholds["variant_state_min"]),
        "variant_answer_min": _gate(min(stats["answer"]["accuracy"] for stats in auto_variant.values()), thresholds["variant_answer_min"]),
        "source_poison_bit_identity": _gate(poison_metric["accuracy"], thresholds["source_poison_bit_identity"]),
    }
    if query_conditional_total < math.ceil(0.85 * family_count):
        gates["query_swap_answer_follow_query_conditional"]["pass"] = False
    if state_swap_total < math.ceil(0.85 * family_count):
        gates["state_swap_separating_effect"]["pass"] = False
    if post_stop_total < math.ceil(0.85 * family_count):
        gates["post_stop_suffix_invariance"]["pass"] = False
        gates["force_alive_suffix_oracle"]["pass"] = False
    gates["motor_72_of_72"] = {"pass": certificate_gates["motor_state_action"], **certificate_metrics["motor_state_action"]}
    gates["stop_6_of_6"] = {"pass": certificate_gates["motor_stop"], **certificate_metrics["motor_stop"]}
    gates["dead_invariance_78_of_78"] = {"pass": certificate_gates["dead_invariance"], **certificate_metrics["dead_invariance"]}
    gates["reader_18_of_18"] = {"pass": certificate_gates["reader"], **certificate_metrics["reader"]}
    gates["controls"] = {"pass": all(control_gates.values()), "arms": control_gates}
    gates["source_poison_complete"] = {"pass": poison_metric["total"] == len(rows)}
    gates["complete_system_parameter_cap"] = {
        "value": total_parameters,
        "ceiling_exclusive": config["parameter_cap"],
        "pass": complete_system and excluded_parameters == 0 and total_parameters < config["parameter_cap"],
    }
    gates["hash_binding"] = {"pass": artifact_hashes == config["expected_hashes"]}
    gates["confirmation_access_zero"] = {
        "value": confirmation_accesses,
        "pass": confirmation_accesses == 0 and not confirmation_opened,
    }
    gates["development_access_one"] = {
        "value": development_accesses,
        "pass": development_accesses == 1,
    }
    gates["access_ledger_binding"] = {
        "value": access_ledger_hash,
        "expected": config["expected_ledger_hash"],
        "pass": access_ledger_hash == config["expected_ledger_hash"],
    }

    all_gates_pass = all(bool(value["pass"]) for value in gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "decision": (
            "authorize_one_sealed_confirmation"
            if all_gates_pass
            else "reject_sd_cst_development_candidate_keep_confirmation_sealed"
        ),
        "all_gates_pass": all_gates_pass,
        "confirmation_authorized": all_gates_pass,
        "compiler": {
            "overall": compiler_overall,
            "per_family": compiler_family,
            "per_variant": compiler_variant,
            "per_depth": compiler_depth,
        },
        "autonomous": {
            "overall": auto_overall,
            "per_variant": auto_variant,
            "per_depth": auto_depth,
            "exact_tape_conditional_execution": exact_tape_execution,
        },
        "causal": causal,
        "certificates": certificate_metrics,
        "controls": control_metrics,
        "source_poison_bit_identity": poison_metric,
        "parameters": {
            "components": components,
            "total": total_parameters,
            "cap_exclusive": config["parameter_cap"],
            "complete_system": complete_system,
            "excluded_trainable_parameters": excluded_parameters,
        },
        "artifact_hashes": artifact_hashes,
        "custody": {
            "split": config["split"],
            "development_accesses": development_accesses,
            "confirmation_accesses": confirmation_accesses,
            "confirmation_opened": confirmation_opened,
            "access_ledger_sha256": access_ledger_hash,
        },
        "gates": gates,
        "claim_boundary": (
            "A pass authorizes one sealed confirmation of bounded SD-CST on the frozen "
            "three-entity finite-state board. It is not evidence of general reasoning, "
            "open-domain transfer, or a flagship promotion condition."
        ),
    }


def assess_files(eval_path: Path, config_path: Path) -> dict[str, Any]:
    result = assess(load_json(eval_path), load_json(config_path))
    result["evidence_sha256"] = {
        "evaluation": sha256_file(eval_path),
        "gate_config": sha256_file(config_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing assessment output: {args.out}")
    try:
        result = assess_files(args.eval, args.config)
    except AssessmentError as error:
        print(f"sd-cst assessment refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": result["all_gates_pass"],
        "confirmation_authorized": result["confirmation_authorized"],
        "decision": result["decision"],
        "out": str(args.out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
