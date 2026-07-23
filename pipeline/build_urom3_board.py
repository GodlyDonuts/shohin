#!/usr/bin/env python3
"""Build immutable UROM-3 train/development rows without opening confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer

from urom3_board import canonical_json, generate_rows, validate_row


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def serialize_rows(rows: Sequence[dict[str, object]]) -> bytes:
    return b"".join(
        canonical_json(row).encode("utf-8") + b"\n"
        for row in rows
    )


def write_once(path: Path, value: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable UROM artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale UROM temporary artifact exists: {temporary}")
    temporary.write_bytes(value)
    temporary.replace(path)
    return sha256_bytes(value)


def distribution(rows: Sequence[dict[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def summarize(
    rows: Sequence[dict[str, object]],
    tokenizer: Tokenizer | None = None,
) -> dict[str, object]:
    targets = [row["compiler_targets"] for row in rows]
    oracles = [row["oracle"] for row in rows]
    report: dict[str, object] = {
        "rows": len(rows),
        "axis_cells": distribution(rows, "axis_cell"),
        "families": distribution(rows, "family"),
        "renderers": distribution(rows, "renderer"),
        "cardinalities": dict(
            sorted(
                Counter(
                    str(target["cardinality"])  # type: ignore[index]
                    for target in targets
                ).items()
            )
        ),
        "program_bytes": {
            "minimum": min(len(str(row["program_text"]).encode("ascii")) for row in rows),
            "maximum": max(len(str(row["program_text"]).encode("ascii")) for row in rows),
        },
        "query_bytes": {
            "minimum": min(len(str(row["query_text"]).encode("ascii")) for row in rows),
            "maximum": max(len(str(row["query_text"]).encode("ascii")) for row in rows),
        },
        "answer_set_size": dict(
            sorted(
                Counter(
                    str(sum(int(value) for value in oracle["answer_bits"]))  # type: ignore[index]
                    for oracle in oracles
                ).items()
            )
        ),
        "semantic_worlds": len({str(row["semantic_sha256"]) for row in rows}),
        "row_hashes": len({str(row["row_sha256"]) for row in rows}),
    }
    if tokenizer is not None:
        program_tokens = [
            len(tokenizer.encode(str(row["program_text"])).ids)
            for row in rows
        ]
        query_tokens = [
            len(tokenizer.encode(str(row["query_text"])).ids)
            for row in rows
        ]
        report["token_lengths"] = {
            "program_minimum": min(program_tokens),
            "program_maximum": max(program_tokens),
            "query_minimum": min(query_tokens),
            "query_maximum": max(query_tokens),
        }
    return report


def build(
    *,
    out_dir: Path,
    seed: int,
    train_count: int,
    development_count: int,
    tokenizer_path: Path | None = None,
    max_tokens: int = 2_048,
) -> dict[str, object]:
    train = generate_rows(split="train", count=train_count, seed=seed)
    development = generate_rows(
        split="development",
        count=development_count,
        seed=seed + 1,
    )
    for row in (*train, *development):
        validate_row(row)
    train_semantics = {str(row["semantic_sha256"]) for row in train}
    development_semantics = {
        str(row["semantic_sha256"]) for row in development
    }
    if train_semantics.intersection(development_semantics):
        raise ValueError("UROM train/development semantic overlap")
    tokenizer = (
        Tokenizer.from_file(str(tokenizer_path))
        if tokenizer_path is not None
        else None
    )
    if tokenizer is not None:
        for label, rows in (("train", train), ("development", development)):
            lengths = [
                len(tokenizer.encode(str(row["program_text"])).ids)
                for row in rows
            ]
            if max(lengths) > max_tokens:
                raise ValueError(f"UROM {label} source exceeds token context")

    train_bytes = serialize_rows(train)
    development_bytes = serialize_rows(development)
    train_path = out_dir / "urom3_train.jsonl"
    development_path = out_dir / "urom3_development.jsonl"
    train_sha256 = write_once(train_path, train_bytes)
    development_sha256 = write_once(development_path, development_bytes)
    report: dict[str, object] = {
        "schema": "urom3_board_manifest_v2",
        "claim_boundary": (
            "Train/development board construction only. Confirmation remains "
            "unopened. This is not a neural result or reasoning claim."
        ),
        "seed": seed,
        "confirmation_accesses": 0,
        "confirmation_generated": False,
        "tokenizer": (
            None
            if tokenizer_path is None
            else {
                "path": str(tokenizer_path),
                "sha256": sha256_bytes(tokenizer_path.read_bytes()),
                "maximum_tokens": max_tokens,
            }
        ),
        "source_deleted_execution_required": True,
        "host_execution_forbidden": True,
        "generated_token_feedback_forbidden": True,
        "retry_repair_forbidden": True,
        "splits": {
            "train": {
                "path": train_path.name,
                "sha256": train_sha256,
                **summarize(train, tokenizer),
            },
            "development": {
                "path": development_path.name,
                "sha256": development_sha256,
                **summarize(development, tokenizer),
            },
        },
        "overlap": {
            "semantic_worlds": 0,
            "row_hashes": len(
                {str(row["row_sha256"]) for row in train}.intersection(
                    str(row["row_sha256"]) for row in development
                )
            ),
        },
    }
    manifest_bytes = (
        json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    report["manifest_sha256"] = sha256_bytes(manifest_bytes)
    final_manifest = (
        json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    write_once(out_dir / "manifest.json", final_manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train-count", type=int, default=12_000)
    parser.add_argument("--development-count", type=int, default=2_048)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=2_048)
    args = parser.parse_args()
    report = build(
        out_dir=args.out_dir,
        seed=args.seed,
        train_count=args.train_count,
        development_count=args.development_count,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
    )
    print(canonical_json(report))


if __name__ == "__main__":
    main()
