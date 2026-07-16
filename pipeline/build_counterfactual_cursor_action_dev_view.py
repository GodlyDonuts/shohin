#!/usr/bin/env python3
"""Build a deterministic confirmation-free view of the frozen R12 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from counterfactual_cursor_action_data import ModelInputMode, load_canary  # noqa: E402
from counterfactual_cursor_action_dev_view import SCHEMA, canonical_json  # noqa: E402


SOURCE_CANARY_SHA256 = "baf985855c396f63dffba1e09733a7372bd8b29c852cb5b9f482b4d59de714a1"
SOURCE_AUDIT_SHA256 = "5deb9dc396e3c8d99f32b9f0e14482d288cff9d82145582665569c911a802e5d"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
IMPLEMENTATION_FILES = (
    "pipeline/build_counterfactual_cursor_action_dev_view.py",
    "pipeline/audit_counterfactual_cursor_action_dev_view.py",
    "pipeline/test_counterfactual_cursor_action_dev_view.py",
    "train/counterfactual_cursor_action_dev_view.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_implementation(commit: str) -> dict[str, str]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("implementation commit is malformed")
    result = {}
    for relative in IMPLEMENTATION_FILES:
        live = ROOT / relative
        observed = sha256_file(live)
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT, check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != observed:
            raise ValueError(f"live implementation differs from commit: {relative}")
        result[relative] = observed
    return result


def write_exclusive_read_only(path: Path, payload: dict[str, Any]) -> str:
    if os.path.lexists(path):
        raise FileExistsError(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if sha256_file(arguments.canary) != SOURCE_CANARY_SHA256:
        raise ValueError("source canary SHA-256 mismatch")
    if sha256_file(arguments.audit) != SOURCE_AUDIT_SHA256:
        raise ValueError("source audit SHA-256 mismatch")
    if sha256_file(arguments.tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer SHA-256 mismatch")
    implementation_hashes = verify_implementation(arguments.implementation_commit)
    dataset = load_canary(
        arguments.canary,
        arguments.audit,
        arguments.tokenizer,
        requested_model_fields={ModelInputMode.SIDECAR: ("prompt_token_ids", "cursor")},
    )
    splits = {}
    for name in ("train", "development"):
        split = dataset.split(name)
        splits[name] = {
            "sources": [
                {
                    "source_id": source.source_id,
                    "renderer_id": source.renderer_id,
                    "pack_id": source.pack_id,
                    "permutation_id": source.permutation_id,
                    "prompt_token_ids": list(source.prompt_token_ids),
                }
                for source in split.sources
            ],
            "cells": [
                {
                    "source_id": cell.source_id,
                    "cursor": cell.cursor,
                    "target_index": cell.target_index,
                    "target_token_id": cell.target_token_id,
                }
                for cell in split.cells
            ],
        }
    payload = {
        "schema": SCHEMA,
        "source_canary_sha256": SOURCE_CANARY_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "implementation_commit": arguments.implementation_commit,
        "implementation_file_sha256": implementation_hashes,
        "label_token_ids": [820, 5498, 4307, 7486, 2165],
        "splits": splits,
    }
    payload["payload_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    digest = write_exclusive_read_only(arguments.out, payload)
    print(json.dumps({"out": str(arguments.out), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
