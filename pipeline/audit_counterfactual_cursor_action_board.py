#!/usr/bin/env python3
"""Independently reconstruct and audit the cursor-action mechanics board."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import stat
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOARD_SCHEMA = "counterfactual_cursor_action_board_v1"
BOARD_ID = "ccaa-mechanics-600-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
LABELS = OPERATIONS + ("DONE",)
ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "pipeline/generate_counterfactual_cursor_action_board.py"
CONTRACT = ROOT / "pipeline/counterfactual_cursor_action_contract_v1.json"
CONTRACT_SHA256 = "a7061e553b13189e91d25a19f164ccdecfe49404591f1fe0ecfa83b72b690a3c"
IMPLEMENTATION_PATHS = (
    "R12_COUNTERFACTUAL_CURSOR_ACTION_THEORY.md",
    "R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md",
    "pipeline/counterfactual_cursor_action_contract_v1.json",
    "pipeline/generate_counterfactual_cursor_action_board.py",
    "pipeline/audit_counterfactual_cursor_action_board.py",
    "pipeline/test_counterfactual_cursor_action_board.py",
)
EXPECTED_EXPOSURE_CONTRACT = {
    "selector_model_visible_row_fields": ["source"],
    "selector_model_visible_side_state": ["cursor"],
    "one_call_model_visible_row_fields": ["source"],
    "one_call_initial_side_state": {"cursor": 0, "phase": "SELECT"},
    "forbidden_model_visible_fields": [
        "row_id", "source_id", "renderer_id", "permutation_id", "start_value",
        "operation_order", "clause_spans", "target_action", "target_index",
    ],
}


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


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    return json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)


def load_contract() -> dict[str, Any]:
    require(file_sha256(CONTRACT) == CONTRACT_SHA256, "contract hash mismatch")
    contract = load_json_strict(CONTRACT)
    require(contract.get("schema") == "counterfactual_cursor_action_contract_v1",
            "contract schema mismatch")
    require(tuple(contract.get("operations", ())) == OPERATIONS, "contract operations mismatch")
    require(tuple(contract.get("labels", ())) == LABELS, "contract labels mismatch")
    renderers = contract.get("renderers")
    require(isinstance(renderers, list) and len(renderers) == 5, "contract renderer mismatch")
    for renderer_id, renderer in enumerate(renderers):
        require(isinstance(renderer, dict), "contract renderer is not an object")
        require(renderer.get("renderer_id") == renderer_id, "contract renderer ID mismatch")
        require(type(renderer.get("start_value")) is int, "contract start value is not an integer")
        clauses = renderer.get("clauses")
        require(isinstance(clauses, list) and len(clauses) == 4, "contract clauses mismatch")
        require(tuple(clause.get("operation") for clause in clauses) == OPERATIONS,
                "contract clause order mismatch")
        for clause in clauses:
            require(type(clause.get("operand")) is int, "contract operand is not an integer")
            require(isinstance(clause.get("text"), str) and clause["text"], "contract clause text missing")
    semantic_tuples = [
        tuple((clause["operation"], clause["operand"]) for clause in renderer["clauses"])
        for renderer in renderers
    ]
    require(len(set(semantic_tuples)) == 1, "renderer groups are not content-matched")
    return contract


def verify_implementation_identity(identity: dict[str, Any]) -> None:
    require(isinstance(identity, dict) and set(identity) == {"git_commit", "file_sha256"},
            "implementation identity fields mismatch")
    commit = identity["git_commit"]
    hashes = identity["file_sha256"]
    require(isinstance(commit, str) and len(commit) == 40, "implementation commit mismatch")
    require(isinstance(hashes, dict) and set(hashes) == set(IMPLEMENTATION_PATHS),
            "implementation file ledger mismatch")
    for relative in IMPLEMENTATION_PATHS:
        expected = hashes[relative]
        require(expected == file_sha256(ROOT / relative), f"live implementation hash mismatch: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True, capture_output=True,
        ).stdout
        require(expected == sha256_bytes(committed), f"committed implementation hash mismatch: {relative}")


def expected_source(
    renderers: list[dict[str, Any]], renderer_id: int, order: tuple[str, ...]
) -> tuple[str, list[dict[str, Any]]]:
    renderer = renderers[renderer_id]
    clauses = {clause["operation"]: clause for clause in renderer["clauses"]}
    prefix, joiner, suffix = renderer["prefix"], renderer["joiner"], renderer["suffix"]
    texts = [clauses[operation]["text"] for operation in order]
    source = prefix + joiner.join(texts) + suffix
    spans = []
    scan = len(prefix)
    for operation, text in zip(order, texts):
        start = source.find(text, scan)
        require(start >= 0, "auditor could not reconstruct clause")
        end = start + len(text)
        spans.append({
            "operation": operation,
            "operand": clauses[operation]["operand"],
            "text": text,
            "start": start,
            "end": end,
        })
        scan = end
    return source, spans


def recover_order(
    renderers: list[dict[str, Any]], renderer_id: int, source: str
) -> tuple[int, tuple[str, ...], list[dict[str, Any]]]:
    matches = []
    for permutation_id, order in enumerate(itertools.permutations(OPERATIONS)):
        expected, spans = expected_source(renderers, renderer_id, order)
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
    ] + [("HALT_PENDING", "HALT_PENDING"), ("HALT", "HALT")]
    events = OPERATIONS + ("COMMIT", "DONE", "EOS", "OTHER")

    def explicit_transition(state: tuple[int | str, str], event: str) -> tuple[int | str, str]:
        cursor, phase = state
        if phase == "HALT":
            return state
        if phase == "HALT_PENDING":
            return ("HALT", "HALT") if event == "EOS" else state
        require(type(cursor) is int, "non-halt state has noninteger cursor")
        if phase == "SELECT" and cursor < 4 and event in OPERATIONS:
            return (min(cursor + 1, 4), "EXECUTE")
        if phase == "EXECUTE" and event == "COMMIT":
            return (cursor, "SELECT")
        if phase == "SELECT" and cursor == 4 and event == "DONE":
            return ("HALT_PENDING", "HALT_PENDING")
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
        "remainder", "COMMIT", "DONE", "EOS",
    )
    event_state: tuple[int | str, str] = (0, "SELECT")
    event_trace = [event_state]
    for event in trace:
        event_state = explicit_transition(event_state, event)
        event_trace.append(event_state)
    require(event_state == ("HALT", "HALT"), "valid event trace did not halt")
    require(explicit_transition((0, "SELECT"), "DONE") == (0, "SELECT"),
            "premature DONE was not rejected")
    require(explicit_transition((4, "SELECT"), "add") == (4, "SELECT"),
            "terminal operation event was not rejected")

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


def audit_document(
    document: dict[str, Any], board_file_sha256: str | None = None,
    verify_commit: bool = False,
) -> dict[str, Any]:
    contract = load_contract()
    renderers = contract["renderers"]
    require(set(document) == {
        "schema", "board_id", "contract_sha256", "generator_sha256",
        "implementation_identity", "exposure_contract", "geometry", "rows_sha256",
        "rows", "adjacent_order_pairs", "renderer_groups",
    }, "board top-level fields mismatch")
    require(document["schema"] == BOARD_SCHEMA, "board schema mismatch")
    require(document["board_id"] == BOARD_ID, "board ID mismatch")
    require(document["contract_sha256"] == CONTRACT_SHA256, "board contract hash mismatch")
    require(document["generator_sha256"] == file_sha256(GENERATOR), "generator hash mismatch")
    require(document["exposure_contract"] == EXPECTED_EXPOSURE_CONTRACT,
            "model exposure contract mismatch")
    if verify_commit:
        verify_implementation_identity(document["implementation_identity"])
    else:
        identity = document["implementation_identity"]
        require(isinstance(identity, dict) and set(identity) == {"git_commit", "file_sha256"},
                "implementation identity fields mismatch")
        require(set(identity["file_sha256"]) == set(IMPLEMENTATION_PATHS),
                "implementation file ledger mismatch")
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
    observed_row_order = []
    source_cursor_keys = set()
    source_texts: dict[str, set[str]] = defaultdict(set)
    recovered_sources: dict[str, tuple[int, tuple[str, ...]]] = {}
    target_counts = Counter()
    operation_by_renderer_cursor: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for row in rows:
        require(isinstance(row, dict) and set(row) == expected_row_fields, "row fields mismatch")
        require(row["schema"] == "counterfactual_cursor_action_cell_v1", "row schema mismatch")
        require(isinstance(row["row_id"], str), "row ID is not a string")
        require(isinstance(row["source_id"], str), "source ID is not a string")
        require(isinstance(row["source"], str), "source is not a string")
        require(row["row_id"] not in row_ids, "duplicate row ID")
        row_ids.add(row["row_id"])
        observed_row_order.append(row["row_id"])
        renderer_id = row["renderer_id"]
        cursor = row["cursor"]
        require(type(renderer_id) is int and 0 <= renderer_id < 5, "invalid renderer")
        require(type(cursor) is int and 0 <= cursor < 5, "invalid cursor")
        require(type(row["permutation_id"]) is int, "permutation ID is not an integer")
        require(type(row["start_value"]) is int, "start value is not an integer")
        require(type(row["target_index"]) is int, "target index is not an integer")
        permutation_id, order, spans = recover_order(renderers, renderer_id, row["source"])
        source_id = f"r{renderer_id:02d}-p{permutation_id:02d}"
        require(row["source_id"] == source_id, "source ID mismatch")
        require(row["row_id"] == f"{source_id}-c{cursor}", "row ID mismatch")
        require(row["permutation_id"] == permutation_id, "permutation ID mismatch")
        require(row["operation_order"] == list(order), "operation order mismatch")
        require(row["clause_spans"] == spans, "clause spans mismatch")
        require(row["start_value"] == renderers[renderer_id]["start_value"], "start value mismatch")
        for span in row["clause_spans"]:
            require(type(span.get("operand")) is int, "span operand is not an integer")
            require(type(span.get("start")) is int, "span start is not an integer")
            require(type(span.get("end")) is int, "span end is not an integer")
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
    expected_row_order = [
        f"r{renderer_id:02d}-p{permutation_id:02d}-c{cursor}"
        for renderer_id in range(5) for permutation_id in range(24) for cursor in range(5)
    ]
    require(observed_row_order == expected_row_order, "canonical row ordering mismatch")
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
        "contract_sha256": CONTRACT_SHA256,
        "implementation_identity": document["implementation_identity"],
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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    board_path = arguments.board.resolve()
    original = arguments.board.lstat()
    require(not stat.S_ISLNK(original.st_mode), "board must not be a symlink")
    require(stat.S_ISREG(original.st_mode), "board must be a regular file")
    require(original.st_mode & 0o222 == 0, "board must be read-only")
    board = load_json_strict(board_path)
    report = audit_document(
        board, board_file_sha256=file_sha256(board_path), verify_commit=True,
    )
    write_exclusive_read_only(arguments.out, report)
    print(
        f"[ccaa-audit] passed rows={report['counts']['rows']} "
        f"board_rows_sha256={report['board_rows_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
