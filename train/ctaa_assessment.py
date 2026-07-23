"""Independent oracle-side scoring for committed CTAA raw evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable

from commit_ctaa_raw_evidence import (
    RAW_EVIDENCE_RECEIPT_SCHEMA,
    RAW_EVIDENCE_SCHEMA,
)

EVIDENCE_KEYS = {
    "schema",
    "family_id",
    "source_index",
    "packet_valid",
    "predicted_action_cards",
    "predicted_opcode_to_card",
    "predicted_initial_state",
    "predicted_opcode_schedule",
    "predicted_schedule",
    "predicted_query_position",
    "state_route",
    "halted",
    "composed_states",
    "route_agreement",
    "answer",
}
ORACLE_KEYS = {
    "family_id",
    "partition",
    "factorial_cell",
    "program_class",
    "depth",
    "action_cards",
    "opcode_to_card",
    "initial_state",
    "opcode_schedule",
    "schedule",
    "query_position",
    "prefix_states",
    "terminal_state",
    "answer",
    "map_deletion_depth",
    "state_deletion_depth",
    "answer_deletion_depth",
    "shortest_equivalent_length",
    "max_run_length",
    "normalized_event_entropy",
    "renderer",
}
INTERVENTION_KEYS = {
    "parent_family_id",
    "relation",
    "invariant_terminal",
    "invariant_trace",
}
BOOLEAN_METRICS = (
    "packet_valid",
    "cards_exact",
    "independent_binding_exact",
    "initial_exact",
    "stop_exact",
    "opcode_schedule_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
)


def _resolve_committed_schedule(
    binding: object,
    opcode_schedule: object,
    *,
    label: str,
) -> list[int]:
    if (
        not isinstance(binding, list)
        or len(binding) != 4
        or any(type(value) is not int for value in binding)
        or sorted(binding) != [0, 1, 2, 3]
        or not isinstance(opcode_schedule, list)
        or len(opcode_schedule) != 41
        or any(
            type(value) is not int or not 0 <= value <= 4
            for value in opcode_schedule
        )
    ):
        raise ValueError(f"CTAA {label} opcode program differs")
    return [4 if opcode == 4 else int(binding[opcode]) for opcode in opcode_schedule]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"CTAA assessment duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_immutable_bytes(path: Path, label: str) -> bytes:
    raw_path = os.path.abspath(os.fspath(path))
    if "\x00" in raw_path or raw_path == "/":
        raise ValueError(f"CTAA assessment {label} path differs")
    components = raw_path.split("/")[1:]
    if any(component in ("", ".", "..") for component in components):
        raise ValueError(f"CTAA assessment {label} path differs")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_descriptor = os.open("/", directory_flags)
        try:
            for component in components[:-1]:
                child = os.open(
                    component, directory_flags, dir_fd=parent_descriptor
                )
                os.close(parent_descriptor)
                parent_descriptor = child
        except BaseException:
            os.close(parent_descriptor)
            raise
    except OSError as error:
        raise ValueError(
            f"CTAA assessment {label} parent is missing or symlinked"
        ) from error
    try:
        metadata = os.stat(
            components[-1], dir_fd=parent_descriptor, follow_symlinks=False
        )
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} is not a single-link immutable file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(components[-1], flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA assessment {label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    if (
        before.st_dev != metadata.st_dev
        or before.st_ino != metadata.st_ino
        or before.st_size != metadata.st_size
        or before.st_mtime_ns != metadata.st_mtime_ns
        or before.st_ctime_ns != metadata.st_ctime_ns
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise ValueError(f"CTAA assessment {label} changed while being read")
    return b"".join(chunks)


def _decode_object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CTAA assessment {label} JSON differs") from error
    if not isinstance(value, dict):
        raise ValueError(f"CTAA assessment {label} root differs")
    return value


def _load_jsonl_bytes(data: bytes, label: str) -> list[dict[str, object]]:
    rows = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"CTAA assessment {label} encoding differs") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {item}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"CTAA assessment {label} row {line_number} JSON differs"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"CTAA assessment row {line_number} differs")
        rows.append(value)
    if not rows:
        raise ValueError("CTAA assessment input is empty")
    return rows


def load_committed_evidence_bundle(
    directory: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    receipt_path = directory / "receipt.json"
    evidence_path = directory / "evidence.jsonl"
    receipt = _decode_object(
        _read_immutable_bytes(receipt_path, "evidence receipt"), "evidence receipt"
    )
    evidence_bytes = _read_immutable_bytes(evidence_path, "committed evidence")
    if (
        receipt.get("schema") != RAW_EVIDENCE_RECEIPT_SCHEMA
        or receipt.get("evidence_sha256") != hashlib.sha256(evidence_bytes).hexdigest()
    ):
        raise ValueError("CTAA committed-evidence receipt differs")
    rows = _load_jsonl_bytes(evidence_bytes, "committed evidence")
    if receipt.get("rows") != len(rows):
        raise ValueError("CTAA committed-evidence row count differs")
    seen = set()
    for index, row in enumerate(rows):
        if (
            set(row) != EVIDENCE_KEYS
            or row.get("schema") != RAW_EVIDENCE_SCHEMA
            or row.get("source_index") != index
            or not isinstance(row.get("family_id"), str)
            or row["family_id"] in seen
        ):
            raise ValueError("CTAA committed-evidence row schema differs")
        seen.add(row["family_id"])
    return receipt, rows


def load_committed_evidence_receipt(directory: Path) -> dict[str, object]:
    receipt, _ = load_committed_evidence_bundle(directory)
    return receipt


def load_committed_evidence(directory: Path) -> list[dict[str, object]]:
    _, rows = load_committed_evidence_bundle(directory)
    return rows


def load_oracle(
    path: Path, partition: str, *, expected_sha256: str | None = None
) -> list[dict[str, object]]:
    oracle_bytes = _read_immutable_bytes(path, "sealed oracle")
    if (
        expected_sha256 is not None
        and hashlib.sha256(oracle_bytes).hexdigest() != expected_sha256
    ):
        raise ValueError("CTAA assessed oracle hash differs from sealed board")
    rows = _load_jsonl_bytes(oracle_bytes, "sealed oracle")
    seen = set()
    for row in rows:
        allowed = ORACLE_KEYS | (
            INTERVENTION_KEYS if "parent_family_id" in row else set()
        )
        if (
            set(row) != allowed
            or row.get("partition") != partition
            or not isinstance(row.get("family_id"), str)
            or row["family_id"] in seen
            or "program_source" in row
            or "query_source" in row
        ):
            raise ValueError("CTAA oracle row schema differs")
        seen.add(row["family_id"])
    return rows


def _halt_valid(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 42
        or any(type(item) is not bool for item in value)
    ):
        return False
    transitions = [int(value[index + 1]) - int(value[index]) for index in range(41)]
    return (
        not value[0]
        and value[-1]
        and transitions.count(1) == 1
        and min(transitions) >= 0
    )


def _mean(values: Iterable[bool]) -> float:
    result = list(values)
    return sum(result) / len(result) if result else 0.0


def _aggregate(row_scores: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": len(row_scores),
        **{
            metric: _mean(bool(row[metric]) for row in row_scores)
            for metric in BOOLEAN_METRICS
        },
        "active_prefix_step_accuracy": (
            sum(int(row["active_steps_correct"]) for row in row_scores)
            / sum(int(row["active_steps_total"]) for row in row_scores)
            if sum(int(row["active_steps_total"]) for row in row_scores)
            else 0.0
        ),
    }


def _strata(row_scores: list[dict[str, object]], key: str) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in row_scores:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(values) for name, values in sorted(grouped.items())}


def _factorial_effects(row_scores: list[dict[str, object]]) -> dict[str, object]:
    axes = {"semantic": 0, "renderer": 1, "lexical": 2}
    effects: dict[str, object] = {}
    for name, position in axes.items():
        inherited = [
            bool(row["prefix_exact"])
            for row in row_scores
            if str(row["factorial_cell"])[position] == "i"
        ]
        held_out = [
            bool(row["prefix_exact"])
            for row in row_scores
            if str(row["factorial_cell"])[position] == "h"
        ]
        effects[name] = {
            "inherited": _mean(inherited),
            "held_out": _mean(held_out),
            "held_out_minus_inherited": _mean(held_out) - _mean(inherited),
            "inherited_families": len(inherited),
            "held_out_families": len(held_out),
        }
    return effects


def score_evidence(
    evidence_rows: list[dict[str, object]],
    oracle_rows: list[dict[str, object]],
    *,
    parent_evidence_rows: list[dict[str, object]] | None = None,
    parent_oracle_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    evidence = {row["family_id"]: row for row in evidence_rows}
    oracle = {row["family_id"]: row for row in oracle_rows}
    if set(evidence) != set(oracle):
        raise ValueError("CTAA evidence/oracle family set differs")
    parent_evidence = (
        {row["family_id"]: row for row in parent_evidence_rows}
        if parent_evidence_rows is not None
        else {}
    )
    parent_oracle = (
        {row["family_id"]: row for row in parent_oracle_rows}
        if parent_oracle_rows is not None
        else {}
    )
    row_scores: list[dict[str, object]] = []
    action_correct: dict[str, list[bool]] = defaultdict(list)
    semantic_action_correct: dict[str, list[bool]] = defaultdict(list)
    action_rank_correct: dict[str, list[bool]] = defaultdict(list)
    quartile_correct: dict[str, list[bool]] = defaultdict(list)
    relation_correct: dict[str, list[bool]] = defaultdict(list)
    for family_id in [row["family_id"] for row in oracle_rows]:
        predicted = evidence[family_id]
        target = oracle[family_id]
        packet_valid = bool(predicted["packet_valid"])
        cards_exact = predicted["predicted_action_cards"] == target["action_cards"]
        independent_binding_exact = (
            predicted["predicted_opcode_to_card"] == target["opcode_to_card"]
        )
        initial_exact = predicted["predicted_initial_state"] == target["initial_state"]
        opcode_schedule_exact = (
            predicted["predicted_opcode_schedule"] == target["opcode_schedule"]
        )
        reconstructed_schedule = _resolve_committed_schedule(
            predicted["predicted_opcode_to_card"],
            predicted["predicted_opcode_schedule"],
            label="committed prediction",
        )
        if predicted["predicted_schedule"] != reconstructed_schedule:
            raise ValueError("CTAA committed resolved schedule differs")
        oracle_schedule = _resolve_committed_schedule(
            target["opcode_to_card"],
            target["opcode_schedule"],
            label="oracle",
        )
        if target["schedule"] != oracle_schedule:
            raise ValueError("CTAA oracle resolved schedule differs")
        schedule_exact = reconstructed_schedule == oracle_schedule
        predicted_stop = [
            index
            for index, event in enumerate(reconstructed_schedule)
            if event == 4
        ]
        target_stop = oracle_schedule.index(4)
        stop_exact = packet_valid and predicted_stop == [target_stop]
        program_exact = (
            packet_valid
            and cards_exact
            and independent_binding_exact
            and initial_exact
            and opcode_schedule_exact
            and schedule_exact
        )
        query_exact = (
            predicted["predicted_query_position"] is not None
            and predicted["predicted_query_position"] == target["query_position"]
        )
        halt_valid = _halt_valid(predicted["halted"])
        route_agreement = (
            bool(predicted["route_agreement"])
            and predicted["state_route"] is not None
            and predicted["state_route"] == predicted["composed_states"]
        )
        prefix_exact = (
            program_exact
            and halt_valid
            and route_agreement
            and predicted["state_route"] == target["prefix_states"]
        )
        terminal_exact = (
            program_exact
            and isinstance(predicted["state_route"], list)
            and len(predicted["state_route"]) == 42
            and predicted["state_route"][-1] == target["terminal_state"]
        )
        answer_exact = (
            program_exact
            and query_exact
            and route_agreement
            and predicted["answer"] is not None
            and predicted["answer"] == target["answer"]
        )
        depth = int(target["depth"])
        active_correct = 0
        active_step_outcomes: list[dict[str, object]] = []
        state_route = predicted["state_route"]
        for step in range(depth):
            correct = (
                program_exact
                and isinstance(state_route, list)
                and len(state_route) == 42
                and state_route[step + 1] == target["prefix_states"][step + 1]
            )
            opcode = int(target["opcode_schedule"][step])
            card_address = int(oracle_schedule[step])
            action_correct[str(opcode)].append(correct)
            action = target["action_cards"][card_address]
            semantic_key = json.dumps(action, separators=(",", ":"))
            action_rank = len(set(action))
            semantic_action_correct[semantic_key].append(correct)
            action_rank_correct[str(action_rank)].append(correct)
            quartile = min(3, (4 * step) // depth)
            quartile_correct[str(quartile + 1)].append(correct)
            active_correct += int(correct)
            active_step_outcomes.append(
                {
                    "step": step,
                    "opcode": opcode,
                    "semantic_action": list(action),
                    "action_rank": action_rank,
                    "quartile": quartile + 1,
                    "correct": bool(correct),
                }
            )
        parent_family_id = target.get("parent_family_id")
        relation = target.get("relation")
        expected_trace_equal = target.get("invariant_trace")
        expected_terminal_equal = target.get("invariant_terminal")
        observed_trace_equal: bool | None = None
        observed_terminal_equal: bool | None = None
        relation_correct_value: bool | None = None
        if parent_family_id is not None:
            parent = parent_evidence.get(str(parent_family_id))
            parent_target = parent_oracle.get(str(parent_family_id))
            if parent is None or parent_target is None:
                relation_correct_value = False
            else:
                observed_trace_equal = (
                    parent.get("state_route") is not None
                    and predicted["state_route"] is not None
                    and parent["state_route"] == predicted["state_route"]
                )
                observed_terminal_equal = (
                    isinstance(parent.get("state_route"), list)
                    and isinstance(predicted["state_route"], list)
                    and parent["state_route"][-1] == predicted["state_route"][-1]
                )
                parent_correct = (
                    bool(parent.get("packet_valid"))
                    and _halt_valid(parent.get("halted"))
                    and bool(parent.get("route_agreement"))
                    and parent.get("state_route") == parent_target["prefix_states"]
                )
                relation_correct_value = (
                    parent_correct
                    and prefix_exact
                    and observed_trace_equal == bool(expected_trace_equal)
                    and observed_terminal_equal == bool(expected_terminal_equal)
                )
            relation_correct[str(relation)].append(bool(relation_correct_value))
        row_score = {
            "family_id": family_id,
            "cluster_family_id": parent_family_id or family_id,
            "parent_family_id": parent_family_id,
            "relation": relation,
            "expected_trace_equal": expected_trace_equal,
            "expected_terminal_equal": expected_terminal_equal,
            "observed_trace_equal": observed_trace_equal,
            "observed_terminal_equal": observed_terminal_equal,
            "relation_correct": relation_correct_value,
            "factorial_cell": target["factorial_cell"],
            "shift_order": str(target["factorial_cell"]).count("h"),
            "program_class": target["program_class"],
            "depth": target["depth"],
            "renderer": target["renderer"],
            "packet_valid": packet_valid,
            "cards_exact": cards_exact,
            "independent_binding_exact": independent_binding_exact,
            "initial_exact": initial_exact,
            "stop_exact": stop_exact,
            "opcode_schedule_exact": opcode_schedule_exact,
            "schedule_exact": schedule_exact,
            "program_exact": program_exact,
            "query_exact": query_exact,
            "halt_valid": halt_valid,
            "route_agreement": route_agreement,
            "prefix_exact": prefix_exact,
            "terminal_exact": terminal_exact,
            "answer_exact": answer_exact,
            "active_steps_correct": active_correct,
            "active_steps_total": depth,
            "active_step_outcomes": active_step_outcomes,
        }
        row_scores.append(row_score)
    return {
        "overall": _aggregate(row_scores),
        "by_factorial_cell": _strata(row_scores, "factorial_cell"),
        "by_program_class": _strata(row_scores, "program_class"),
        "by_depth": _strata(row_scores, "depth"),
        "by_renderer": _strata(row_scores, "renderer"),
        "by_shift_order": _strata(row_scores, "shift_order"),
        "factorial_main_effects": _factorial_effects(row_scores),
        "by_action_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(action_correct.items())
        },
        "by_semantic_action_active_prefix_accuracy": {
            key: _mean(values)
            for key, values in sorted(semantic_action_correct.items())
        },
        "by_action_rank_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(action_rank_correct.items())
        },
        "by_step_quartile_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(quartile_correct.items())
        },
        "intervention_relation_correct": {
            key: _mean(values) for key, values in sorted(relation_correct.items())
        },
        "family_scores": row_scores,
    }
