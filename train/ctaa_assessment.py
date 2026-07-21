"""Independent oracle-side scoring for committed CTAA raw evidence."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

from commit_ctaa_raw_evidence import (
    RAW_EVIDENCE_RECEIPT_SCHEMA,
    RAW_EVIDENCE_SCHEMA,
)
from ctaa_evaluation_io import sha256_file


EVIDENCE_KEYS = {
    "schema",
    "family_id",
    "source_index",
    "packet_valid",
    "predicted_action_cards",
    "predicted_initial_state",
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
    "initial_state",
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
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"CTAA assessment row {line_number} differs")
            rows.append(value)
    if not rows:
        raise ValueError("CTAA assessment input is empty")
    return rows


def load_committed_evidence_receipt(directory: Path) -> dict[str, object]:
    receipt_path = directory / "receipt.json"
    evidence_path = directory / "evidence.jsonl"
    receipt = json.loads(receipt_path.read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != RAW_EVIDENCE_RECEIPT_SCHEMA
        or receipt.get("evidence_sha256") != sha256_file(evidence_path)
    ):
        raise ValueError("CTAA committed-evidence receipt differs")
    return receipt


def load_committed_evidence(directory: Path) -> list[dict[str, object]]:
    receipt = load_committed_evidence_receipt(directory)
    evidence_path = directory / "evidence.jsonl"
    rows = _load_jsonl(evidence_path)
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
    return rows


def load_oracle(path: Path, partition: str) -> list[dict[str, object]]:
    rows = _load_jsonl(path)
    seen = set()
    for row in rows:
        allowed = ORACLE_KEYS | (INTERVENTION_KEYS if "parent_family_id" in row else set())
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
    if not isinstance(value, list) or len(value) != 42 or any(type(item) is not bool for item in value):
        return False
    transitions = [int(value[index + 1]) - int(value[index]) for index in range(41)]
    return not value[0] and value[-1] and transitions.count(1) == 1 and min(transitions) >= 0


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


def score_evidence(
    evidence_rows: list[dict[str, object]],
    oracle_rows: list[dict[str, object]],
    *,
    parent_evidence_rows: list[dict[str, object]] | None = None,
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
    row_scores: list[dict[str, object]] = []
    action_correct: dict[str, list[bool]] = defaultdict(list)
    quartile_correct: dict[str, list[bool]] = defaultdict(list)
    relation_correct: dict[str, list[bool]] = defaultdict(list)
    for family_id in [row["family_id"] for row in oracle_rows]:
        predicted = evidence[family_id]
        target = oracle[family_id]
        packet_valid = bool(predicted["packet_valid"])
        cards_exact = predicted["predicted_action_cards"] == target["action_cards"]
        binding_exact = cards_exact
        initial_exact = predicted["predicted_initial_state"] == target["initial_state"]
        schedule_exact = predicted["predicted_schedule"] == target["schedule"]
        predicted_stop = [
            index
            for index, event in enumerate(predicted["predicted_schedule"])
            if event == 4
        ]
        target_stop = target["schedule"].index(4)
        stop_exact = packet_valid and predicted_stop == [target_stop]
        program_exact = packet_valid and cards_exact and initial_exact and schedule_exact
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
        state_route = predicted["state_route"]
        for step in range(depth):
            correct = (
                program_exact
                and isinstance(state_route, list)
                and len(state_route) == 42
                and state_route[step + 1] == target["prefix_states"][step + 1]
            )
            action_correct[str(target["schedule"][step])].append(correct)
            quartile = min(3, (4 * step) // depth)
            quartile_correct[str(quartile + 1)].append(correct)
            active_correct += int(correct)
        row_score = {
            "family_id": family_id,
            "factorial_cell": target["factorial_cell"],
            "program_class": target["program_class"],
            "depth": target["depth"],
            "renderer": target["renderer"],
            "packet_valid": packet_valid,
            "cards_exact": cards_exact,
            "binding_exact": binding_exact,
            "initial_exact": initial_exact,
            "stop_exact": stop_exact,
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
        }
        row_scores.append(row_score)
        if "parent_family_id" in target and parent_evidence:
            parent = parent_evidence.get(target["parent_family_id"])
            if parent is None:
                relation_correct[str(target["relation"])].append(False)
            else:
                expected_trace_equal = bool(target["invariant_trace"])
                expected_terminal_equal = bool(target["invariant_terminal"])
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
                relation_correct[str(target["relation"])].append(
                    observed_trace_equal == expected_trace_equal
                    and observed_terminal_equal == expected_terminal_equal
                )
    return {
        "overall": _aggregate(row_scores),
        "by_factorial_cell": _strata(row_scores, "factorial_cell"),
        "by_program_class": _strata(row_scores, "program_class"),
        "by_depth": _strata(row_scores, "depth"),
        "by_renderer": _strata(row_scores, "renderer"),
        "by_action_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(action_correct.items())
        },
        "by_step_quartile_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(quartile_correct.items())
        },
        "intervention_relation_correct": {
            key: _mean(values) for key, values in sorted(relation_correct.items())
        },
        "family_scores": row_scores,
    }
