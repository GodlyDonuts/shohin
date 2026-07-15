#!/usr/bin/env python3
"""Generate the finite, score-free counterfactual cursor-action mechanics board."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "counterfactual_cursor_action_board_v1"
BOARD_ID = "ccaa-mechanics-600-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
DONE = "DONE"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path(__file__).resolve().with_name("counterfactual_cursor_action_contract_v1.json")
CONTRACT_SHA256 = "a7061e553b13189e91d25a19f164ccdecfe49404591f1fe0ecfa83b72b690a3c"
IMPLEMENTATION_PATHS = (
    "R12_COUNTERFACTUAL_CURSOR_ACTION_THEORY.md",
    "R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md",
    "pipeline/counterfactual_cursor_action_contract_v1.json",
    "pipeline/generate_counterfactual_cursor_action_board.py",
    "pipeline/audit_counterfactual_cursor_action_board.py",
    "pipeline/test_counterfactual_cursor_action_board.py",
)
EXPOSURE_CONTRACT = {
    "selector_model_visible_row_fields": ["source"],
    "selector_model_visible_side_state": ["cursor"],
    "one_call_model_visible_row_fields": ["source"],
    "one_call_initial_side_state": {"cursor": 0, "phase": "SELECT"},
    "forbidden_model_visible_fields": [
        "row_id", "source_id", "renderer_id", "permutation_id", "start_value",
        "operation_order", "clause_spans", "target_action", "target_index",
    ],
}


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
            raise ValueError(f"duplicate JSON key in contract: {key}")
        result[key] = value
    return result


def load_contract() -> dict[str, Any]:
    if file_sha256(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("cursor-action contract hash mismatch")
    contract = json.loads(CONTRACT_PATH.read_text(), object_pairs_hook=reject_duplicate_keys)
    if contract.get("schema") != "counterfactual_cursor_action_contract_v1":
        raise ValueError("cursor-action contract schema mismatch")
    if tuple(contract.get("operations", ())) != OPERATIONS:
        raise ValueError("cursor-action operation alphabet mismatch")
    if tuple(contract.get("labels", ())) != OPERATIONS + (DONE,):
        raise ValueError("cursor-action label alphabet mismatch")
    renderers = contract.get("renderers")
    if not isinstance(renderers, list) or len(renderers) != 5:
        raise ValueError("cursor-action renderer count mismatch")
    return contract


def implementation_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "file_sha256": {
            relative: file_sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
    }


def require_clean_implementation() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *IMPLEMENTATION_PATHS],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("refusing persistent board from a dirty implementation surface")


def render_source(
    renderers: list[dict[str, Any]], renderer_id: int, order: tuple[str, ...]
) -> tuple[str, list[dict[str, Any]]]:
    renderer = renderers[renderer_id]
    clauses = {clause["operation"]: clause for clause in renderer["clauses"]}
    texts = [clauses[operation]["text"] for operation in order]
    source = renderer["prefix"] + renderer["joiner"].join(texts) + renderer["suffix"]
    spans = []
    scan = len(renderer["prefix"])
    for operation, text in zip(order, texts):
        start = source.find(text, scan)
        if start < 0:
            raise AssertionError("rendered clause is absent")
        end = start + len(text)
        operand = clauses[operation]["operand"]
        spans.append(
            {
                "operation": operation,
                "operand": operand,
                "text": text,
                "start": start,
                "end": end,
            }
        )
        scan = end
    return source, spans


def generate_document() -> dict[str, Any]:
    contract = load_contract()
    renderers = contract["renderers"]
    permutations = list(itertools.permutations(OPERATIONS))
    permutation_index = {value: index for index, value in enumerate(permutations)}
    rows = []
    source_rows: dict[str, dict[str, Any]] = {}

    for renderer_id in range(len(renderers)):
        for permutation_id, order in enumerate(permutations):
            source_id = f"r{renderer_id:02d}-p{permutation_id:02d}"
            source, spans = render_source(renderers, renderer_id, order)
            source_rows[source_id] = {
                "source_id": source_id,
                "renderer_id": renderer_id,
                "permutation_id": permutation_id,
                "operation_order": list(order),
                "source": source,
            }
            for cursor in range(5):
                target = order[cursor] if cursor < 4 else DONE
                rows.append(
                    {
                        "schema": "counterfactual_cursor_action_cell_v1",
                        "row_id": f"{source_id}-c{cursor}",
                        "source_id": source_id,
                        "renderer_id": renderer_id,
                        "permutation_id": permutation_id,
                        "source": source,
                        "start_value": renderers[renderer_id]["start_value"],
                        "operation_order": list(order),
                        "clause_spans": spans,
                        "cursor": cursor,
                        "target_action": target,
                        "target_index": OPERATIONS.index(target) if target in OPERATIONS else 4,
                    }
                )

    adjacent_order_pairs = []
    for renderer_id in range(len(renderers)):
        for permutation_id, order in enumerate(permutations):
            for swap_index in range(3):
                swapped = list(order)
                swapped[swap_index], swapped[swap_index + 1] = (
                    swapped[swap_index + 1],
                    swapped[swap_index],
                )
                other_id = permutation_index[tuple(swapped)]
                if permutation_id < other_id:
                    adjacent_order_pairs.append(
                        {
                            "renderer_id": renderer_id,
                            "swap_index": swap_index,
                            "left_source_id": f"r{renderer_id:02d}-p{permutation_id:02d}",
                            "right_source_id": f"r{renderer_id:02d}-p{other_id:02d}",
                        }
                    )

    renderer_groups = [
        {
            "permutation_id": permutation_id,
            "source_ids": [
                f"r{renderer_id:02d}-p{permutation_id:02d}"
                for renderer_id in range(len(renderers))
            ],
        }
        for permutation_id in range(len(permutations))
    ]

    document = {
        "schema": SCHEMA,
        "board_id": BOARD_ID,
        "contract_sha256": CONTRACT_SHA256,
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_identity": implementation_identity(),
        "exposure_contract": EXPOSURE_CONTRACT,
        "geometry": {
            "operations": list(OPERATIONS),
            "renderers": len(renderers),
            "permutations_per_renderer": len(permutations),
            "cursor_states": 5,
            "sources": len(source_rows),
            "cells": len(rows),
        },
        "rows_sha256": sha256_bytes(canonical_json(rows)),
        "rows": rows,
        "adjacent_order_pairs": adjacent_order_pairs,
        "renderer_groups": renderer_groups,
    }
    return document


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
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    require_clean_implementation()
    document = generate_document()
    write_exclusive_read_only(arguments.out, document)
    print(
        f"[ccaa-board] wrote {arguments.out} cells={len(document['rows'])} "
        f"rows_sha256={document['rows_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
