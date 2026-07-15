#!/usr/bin/env python3
"""Independently reconstruct and audit the cursor-action mechanics board."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOARD_SCHEMA = "counterfactual_cursor_action_board_v1"
BOARD_ID = "ccaa-mechanics-600-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
LABELS = OPERATIONS + ("DONE",)
ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "pipeline/generate_counterfactual_cursor_action_board.py"

# This source deliberately does not import the generator. The literal renderer
# contract is duplicated so a malformed board cannot certify itself.
RENDERERS = (
    (17, "Start with 17. Execute these clauses in order: ", "; ", ".", {
        "add": (7, "add 7"), "subtract": (5, "subtract 5"),
        "multiply": (3, "multiply by 3"),
        "remainder": (11, "take the remainder modulo 11")}),
    (23, "Initial value: 23. Ordered procedure: ", ", then ", ".", {
        "add": (13, "increase the value by 13"),
        "subtract": (4, "decrease the value by 4"),
        "multiply": (3, "triple the value"),
        "remainder": (17, "reduce the value modulo 17")}),
    (31, "Let n = 31. Apply from left to right: ", " | ", ".", {
        "add": (9, "n <- n + 9"), "subtract": (6, "n <- n - 6"),
        "multiply": (2, "n <- n * 2"),
        "remainder": (13, "n <- n mod 13")}),
    (29, "The register begins at 29. Its ordered program is: ", "; next, ", ".", {
        "add": (8, "add 8 to the register"),
        "subtract": (3, "subtract 3 from the register"),
        "multiply": (5, "multiply the register by 5"),
        "remainder": (17, "replace the register by its remainder modulo 17")}),
    (37, "STATE(37) runs this sequence: ", " -> ", ".", {
        "add": (12, "ADD(12)"), "subtract": (7, "SUBTRACT(7)"),
        "multiply": (4, "MULTIPLY(4)"),
        "remainder": (19, "REMAINDER(19)")}),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_source(renderer_id: int, order: tuple[str, ...]) -> tuple[str, list[dict[str, Any]]]:
    start_value, prefix, joiner, suffix, clauses = RENDERERS[renderer_id]
    texts = [clauses[operation][1] for operation in order]
    source = prefix + joiner.join(texts) + suffix
    spans = []
    scan = len(prefix)
    for operation, text in zip(order, texts):
        start = source.find(text, scan)
        require(start >= 0, "auditor could not reconstruct clause")
        end = start + len(text)
        spans.append({
            "operation": operation,
            "operand": clauses[operation][0],
            "text": text,
            "start": start,
            "end": end,
        })
        scan = end
    return source, spans


def recover_order(renderer_id: int, source: str) -> tuple[int, tuple[str, ...], list[dict[str, Any]]]:
    matches = []
    for permutation_id, order in enumerate(itertools.permutations(OPERATIONS)):
        expected, spans = expected_source(renderer_id, order)
        if expected == source:
            matches.append((permutation_id, order, spans))
    require(len(matches) == 1, "source does not recover exactly one operation order")
    return matches[0]


def best_group_score(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> int:
    groups: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[tuple(row[key] for key in keys)][row["target_action"]] += 1
    return sum(max(counts.values()) for counts in groups.values())


def audit_collapse() -> dict[str, Any]:
    phases = ("SELECT", "EXECUTE")
    states: list[tuple[int | str, str]] = [
        (cursor, phase) for cursor in range(5) for phase in phases
    ] + [("HALT", "HALT")]
    events = OPERATIONS + ("COMMIT", "DONE", "OTHER")

    def explicit_transition(state: tuple[int | str, str], event: str) -> tuple[int | str, str]:
        cursor, phase = state
        if phase == "HALT":
            return state
        require(type(cursor) is int, "non-halt state has noninteger cursor")
        if phase == "SELECT" and cursor < 4 and event in OPERATIONS:
            return (min(cursor + 1, 4), "EXECUTE")
        if phase == "EXECUTE" and event == "COMMIT":
            return (cursor, "SELECT")
        if phase == "SELECT" and cursor == 4 and event == "DONE":
            return ("HALT", "HALT")
        return state

    matrices = {}
    transition_assertions = 0
    for event in events:
        matrix = [[0 for _ in states] for _ in states]
        for source_index, state in enumerate(states):
            target = explicit_transition(state, event)
            target_index = states.index(target)
            matrix[target_index][source_index] = 1
        matrices[event] = matrix
        for source_index, state in enumerate(states):
            output = [sum(matrix[row][column] * (1 if column == source_index else 0)
                          for column in range(len(states))) for row in range(len(states))]
            require(sum(output) == 1, "event matrix did not preserve one-hot state")
            recovered = states[output.index(1)]
            require(recovered == explicit_transition(state, event), "event matrix transition mismatch")
            transition_assertions += 1

    trace = (
        "add", "COMMIT", "subtract", "COMMIT", "multiply", "COMMIT",
        "remainder", "COMMIT", "DONE",
    )
    event_state: tuple[int | str, str] = (0, "SELECT")
    event_trace = [event_state]
    for event in trace:
        event_state = explicit_transition(event_state, event)
        event_trace.append(event_state)
    require(event_state == ("HALT", "HALT"), "valid event trace did not halt")

    # At fixed one-controller-step duration, the event cursor degenerates to
    # the ordinary clamped position table. This is a narrower collapse than the
    # event-triggered finite-state reduction above.
    state = [1, 0, 0, 0, 0]
    sequence = []
    for step in range(11):
        lookup = min(step, 4)
        recurrence = state.index(1)
        pointer = next(index for index, value in enumerate(state) if value)
        position = min(step, 4)
        require(lookup == recurrence == pointer == position, "cursor realizations disagree")
        sequence.append(lookup)
        next_state = [0, 0, 0, 0, 0]
        next_state[min(recurrence + 1, 4)] = 1
        state = next_state

    # A fixed cursor code injected into Q is exactly a larger ordinary linear
    # projection on augmented features. Integer arithmetic keeps this check
    # exact rather than tolerance-based.
    cursor_codes = [
        tuple(1 if (cursor >> bit) & 1 else -1 for bit in range(3))
        for cursor in range(5)
    ]
    adapter = [[((row + 3) * (column + 5)) % 11 - 5 for column in range(3)] for row in range(64)]
    base = [[((row + 7) * (column + 2)) % 13 - 6 for column in range(4)] for row in range(64)]
    hidden = (3, -2, 5, 1)
    projection_assertions = 0
    for code in cursor_codes:
        separate = [
            sum(base[row][column] * hidden[column] for column in range(4))
            + sum(adapter[row][column] * code[column] for column in range(3))
            for row in range(64)
        ]
        augmented_weights = [base[row] + adapter[row] for row in range(64)]
        augmented_input = hidden + code
        folded = [
            sum(augmented_weights[row][column] * augmented_input[column]
                for column in range(7))
            for row in range(64)
        ]
        require(separate == folded, "cursor Q injection did not fold into augmented projection")
        projection_assertions += len(separate)
    return {
        "event_fsm_realizations_equal": True,
        "event_fsm_states": len(states),
        "event_classes": len(events),
        "event_transition_assertions": transition_assertions,
        "valid_event_trace": [list(value) for value in event_trace],
        "fixed_duration_realizations_equal": True,
        "fixed_duration_checked_steps": len(sequence),
        "fixed_duration_cursor_sequence": sequence,
        "query_projection_folded": True,
        "query_projection_assertions": projection_assertions,
        "primitive_novelty_rejected": True,
    }


def audit_document(document: dict[str, Any], board_file_sha256: str | None = None) -> dict[str, Any]:
    require(set(document) == {
        "schema", "board_id", "generator_sha256", "geometry", "rows_sha256",
        "rows", "adjacent_order_pairs", "renderer_groups",
    }, "board top-level fields mismatch")
    require(document["schema"] == BOARD_SCHEMA, "board schema mismatch")
    require(document["board_id"] == BOARD_ID, "board ID mismatch")
    require(document["generator_sha256"] == file_sha256(GENERATOR), "generator hash mismatch")
    rows = document["rows"]
    require(isinstance(rows, list) and len(rows) == 600, "board must contain 600 rows")
    require(document["rows_sha256"] == sha256_bytes(canonical_json(rows)), "row hash mismatch")

    expected_geometry = {
        "operations": list(OPERATIONS), "renderers": 5,
        "permutations_per_renderer": 24, "cursor_states": 5,
        "sources": 120, "cells": 600,
    }
    require(document["geometry"] == expected_geometry, "geometry mismatch")

    expected_row_fields = {
        "schema", "row_id", "source_id", "renderer_id", "permutation_id",
        "source", "start_value", "operation_order", "clause_spans", "cursor",
        "target_action", "target_index",
    }
    row_ids = set()
    source_cursor_keys = set()
    source_texts: dict[str, set[str]] = defaultdict(set)
    recovered_sources: dict[str, tuple[int, tuple[str, ...]]] = {}
    target_counts = Counter()
    operation_by_renderer_cursor: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_row_fields, "row fields mismatch")
        require(row["schema"] == "counterfactual_cursor_action_cell_v1", "row schema mismatch")
        require(row["row_id"] not in row_ids, "duplicate row ID")
        row_ids.add(row["row_id"])
        renderer_id = row["renderer_id"]
        cursor = row["cursor"]
        require(type(renderer_id) is int and 0 <= renderer_id < 5, "invalid renderer")
        require(type(cursor) is int and 0 <= cursor < 5, "invalid cursor")
        permutation_id, order, spans = recover_order(renderer_id, row["source"])
        source_id = f"r{renderer_id:02d}-p{permutation_id:02d}"
        require(row["source_id"] == source_id, "source ID mismatch")
        require(row["row_id"] == f"{source_id}-c{cursor}", "row ID mismatch")
        require(row["permutation_id"] == permutation_id, "permutation ID mismatch")
        require(row["operation_order"] == list(order), "operation order mismatch")
        require(row["clause_spans"] == spans, "clause spans mismatch")
        require(row["start_value"] == RENDERERS[renderer_id][0], "start value mismatch")
        target = order[cursor] if cursor < 4 else "DONE"
        require(row["target_action"] == target, "target action mismatch")
        require(row["target_index"] == LABELS.index(target), "target index mismatch")
        key = (source_id, cursor)
        require(key not in source_cursor_keys, "duplicate source/cursor key")
        source_cursor_keys.add(key)
        source_texts[source_id].add(row["source"])
        recovered_sources[source_id] = (renderer_id, order)
        target_counts[target] += 1
        operation_by_renderer_cursor[(renderer_id, cursor)][target] += 1

    require(len(source_texts) == 120, "source count mismatch")
    require(all(len(values) == 1 for values in source_texts.values()), "cursor leaked into source text")
    require(all(sum(source_id == row["source_id"] for row in rows) == 5 for source_id in source_texts),
            "source cursor group is incomplete")
    require(target_counts == Counter({label: 120 for label in LABELS}), "global targets are unbalanced")
    for renderer_id in range(5):
        for cursor in range(4):
            require(operation_by_renderer_cursor[(renderer_id, cursor)] ==
                    Counter({operation: 6 for operation in OPERATIONS}),
                    "renderer/cursor operations are unbalanced")
        require(operation_by_renderer_cursor[(renderer_id, 4)] == Counter({"DONE": 24}),
                "DONE geometry mismatch")

    permutations = list(itertools.permutations(OPERATIONS))
    permutation_index = {value: index for index, value in enumerate(permutations)}
    expected_pairs = []
    for renderer_id in range(5):
        for permutation_id, order in enumerate(permutations):
            for swap_index in range(3):
                swapped = list(order)
                swapped[swap_index], swapped[swap_index + 1] = swapped[swap_index + 1], swapped[swap_index]
                other_id = permutation_index[tuple(swapped)]
                if permutation_id < other_id:
                    expected_pairs.append({
                        "renderer_id": renderer_id, "swap_index": swap_index,
                        "left_source_id": f"r{renderer_id:02d}-p{permutation_id:02d}",
                        "right_source_id": f"r{renderer_id:02d}-p{other_id:02d}",
                    })
    require(document["adjacent_order_pairs"] == expected_pairs, "adjacent-order pair map mismatch")
    expected_groups = [{
        "permutation_id": permutation_id,
        "source_ids": [f"r{renderer_id:02d}-p{permutation_id:02d}" for renderer_id in range(5)],
    } for permutation_id in range(24)]
    require(document["renderer_groups"] == expected_groups, "renderer group map mismatch")

    symbolic_scores = {
        "oracle_source_cursor": 600,
        "global_constant": max(target_counts.values()),
        "best_source_only": best_group_score(rows, ("source_id",)),
        "best_renderer_only": best_group_score(rows, ("renderer_id",)),
        "best_cursor_only": best_group_score(rows, ("cursor",)),
        "best_renderer_cursor": best_group_score(rows, ("renderer_id", "cursor")),
    }
    by_source_cursor = {(row["source_id"], row["cursor"]): row["target_action"] for row in rows}
    symbolic_scores["oracle_cursor_clamped_zero"] = sum(
        by_source_cursor[(row["source_id"], 0)] == row["target_action"] for row in rows
    )
    symbolic_scores["oracle_cursor_five_cycle"] = sum(
        by_source_cursor[(row["source_id"], (row["cursor"] + 1) % 5)] == row["target_action"]
        for row in rows
    )
    expected_scores = {
        "oracle_source_cursor": 600, "global_constant": 120,
        "best_source_only": 120, "best_renderer_only": 120,
        "best_cursor_only": 240, "best_renderer_cursor": 240,
        "oracle_cursor_clamped_zero": 120, "oracle_cursor_five_cycle": 0,
    }
    require(symbolic_scores == expected_scores, "symbolic shortcut scores mismatch")

    return {
        "schema": "counterfactual_cursor_action_board_audit_v1",
        "board_id": BOARD_ID,
        "board_canonical_sha256": sha256_bytes(canonical_json(document)),
        "board_file_sha256": board_file_sha256,
        "board_rows_sha256": document["rows_sha256"],
        "auditor_sha256": file_sha256(Path(__file__).resolve()),
        "counts": {
            "rows": len(rows), "sources": len(source_texts),
            "adjacent_order_pairs": len(expected_pairs), "renderer_groups": len(expected_groups),
        },
        "target_counts": dict(sorted(target_counts.items())),
        "symbolic_scores": symbolic_scores,
        "collapse": audit_collapse(),
        "all_passed": True,
        "claim_boundary": (
            "A pass establishes finite board geometry and shortcut rejection only; it does not "
            "establish a novel primitive, neural learnability, autonomous execution, or reasoning."
        ),
    }


def write_exclusive_read_only(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.board.open("r", encoding="utf-8") as handle:
        board = json.load(handle)
    report = audit_document(board, board_file_sha256=file_sha256(arguments.board))
    write_exclusive_read_only(arguments.out, report)
    print(
        f"[ccaa-audit] passed rows={report['counts']['rows']} "
        f"board_rows_sha256={report['board_rows_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
