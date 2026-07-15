#!/usr/bin/env python3
"""Generate the finite, score-free counterfactual cursor-action mechanics board."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "counterfactual_cursor_action_board_v1"
BOARD_ID = "ccaa-mechanics-600-v1"
OPERATIONS = ("add", "subtract", "multiply", "remainder")
DONE = "DONE"
ROOT = Path(__file__).resolve().parents[1]

RENDERERS = (
    {
        "start": 17,
        "prefix": "Start with 17. Execute these clauses in order: ",
        "joiner": "; ",
        "suffix": ".",
        "clauses": {
            "add": (7, "add 7"),
            "subtract": (5, "subtract 5"),
            "multiply": (3, "multiply by 3"),
            "remainder": (11, "take the remainder modulo 11"),
        },
    },
    {
        "start": 23,
        "prefix": "Initial value: 23. Ordered procedure: ",
        "joiner": ", then ",
        "suffix": ".",
        "clauses": {
            "add": (13, "increase the value by 13"),
            "subtract": (4, "decrease the value by 4"),
            "multiply": (3, "triple the value"),
            "remainder": (17, "reduce the value modulo 17"),
        },
    },
    {
        "start": 31,
        "prefix": "Let n = 31. Apply from left to right: ",
        "joiner": " | ",
        "suffix": ".",
        "clauses": {
            "add": (9, "n <- n + 9"),
            "subtract": (6, "n <- n - 6"),
            "multiply": (2, "n <- n * 2"),
            "remainder": (13, "n <- n mod 13"),
        },
    },
    {
        "start": 29,
        "prefix": "The register begins at 29. Its ordered program is: ",
        "joiner": "; next, ",
        "suffix": ".",
        "clauses": {
            "add": (8, "add 8 to the register"),
            "subtract": (3, "subtract 3 from the register"),
            "multiply": (5, "multiply the register by 5"),
            "remainder": (17, "replace the register by its remainder modulo 17"),
        },
    },
    {
        "start": 37,
        "prefix": "STATE(37) runs this sequence: ",
        "joiner": " -> ",
        "suffix": ".",
        "clauses": {
            "add": (12, "ADD(12)"),
            "subtract": (7, "SUBTRACT(7)"),
            "multiply": (4, "MULTIPLY(4)"),
            "remainder": (19, "REMAINDER(19)"),
        },
    },
)


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


def render_source(renderer_id: int, order: tuple[str, ...]) -> tuple[str, list[dict[str, Any]]]:
    renderer = RENDERERS[renderer_id]
    texts = [renderer["clauses"][operation][1] for operation in order]
    source = renderer["prefix"] + renderer["joiner"].join(texts) + renderer["suffix"]
    spans = []
    scan = len(renderer["prefix"])
    for operation, text in zip(order, texts):
        start = source.find(text, scan)
        if start < 0:
            raise AssertionError("rendered clause is absent")
        end = start + len(text)
        operand = renderer["clauses"][operation][0]
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
    permutations = list(itertools.permutations(OPERATIONS))
    permutation_index = {value: index for index, value in enumerate(permutations)}
    rows = []
    source_rows: dict[str, dict[str, Any]] = {}

    for renderer_id in range(len(RENDERERS)):
        for permutation_id, order in enumerate(permutations):
            source_id = f"r{renderer_id:02d}-p{permutation_id:02d}"
            source, spans = render_source(renderer_id, order)
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
                        "start_value": RENDERERS[renderer_id]["start"],
                        "operation_order": list(order),
                        "clause_spans": spans,
                        "cursor": cursor,
                        "target_action": target,
                        "target_index": OPERATIONS.index(target) if target in OPERATIONS else 4,
                    }
                )

    adjacent_order_pairs = []
    for renderer_id in range(len(RENDERERS)):
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
                for renderer_id in range(len(RENDERERS))
            ],
        }
        for permutation_id in range(len(permutations))
    ]

    document = {
        "schema": SCHEMA,
        "board_id": BOARD_ID,
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "geometry": {
            "operations": list(OPERATIONS),
            "renderers": len(RENDERERS),
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
    document = generate_document()
    write_exclusive_read_only(arguments.out, document)
    print(
        f"[ccaa-board] wrote {arguments.out} cells={len(document['rows'])} "
        f"rows_sha256={document['rows_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
