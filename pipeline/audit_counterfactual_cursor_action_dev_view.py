#!/usr/bin/env python3
"""Independently compare the R12 development view with its frozen source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "counterfactual_cursor_action_dev_view_v1"
AUDIT_SCHEMA = "counterfactual_cursor_action_dev_view_audit_v1"
SOURCE_CANARY_SHA256 = "baf985855c396f63dffba1e09733a7372bd8b29c852cb5b9f482b4d59de714a1"
SOURCE_AUDIT_SHA256 = "5deb9dc396e3c8d99f32b9f0e14482d288cff9d82145582665569c911a802e5d"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_read_only(path: Path, payload: object) -> str:
    if os.path.lexists(path):
        raise FileExistsError(path)
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


def project(source: dict[str, object], name: str) -> dict[str, object]:
    split = source["splits"][name]
    return {
        "sources": [
            {
                "source_id": item["source_id"],
                "renderer_id": item["renderer_id"],
                "pack_id": item["pack_id"],
                "permutation_id": item["permutation_id"],
                "prompt_token_ids": item["prompt_token_ids"],
            }
            for item in split["sources"]
        ],
        "cells": [
            {
                "source_id": item["source_id"],
                "cursor": item["cursor"],
                "target_index": item["target_index"],
                "target_token_id": item["target_token_id"],
            }
            for item in split["cells"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if sha256_file(arguments.canary) != SOURCE_CANARY_SHA256:
        raise ValueError("source canary SHA-256 mismatch")
    if sha256_file(arguments.source_audit) != SOURCE_AUDIT_SHA256:
        raise ValueError("source audit SHA-256 mismatch")
    if sha256_file(arguments.tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer SHA-256 mismatch")
    source = json.loads(arguments.canary.read_bytes())
    view = json.loads(arguments.view.read_bytes())
    if view["schema"] != SCHEMA or list(view["splits"]) != ["train", "development"]:
        raise ValueError("view split boundary mismatch")
    for name in ("train", "development"):
        if view["splits"][name] != project(source, name):
            raise ValueError(f"{name} projection mismatch")
    serialized = arguments.view.read_bytes()
    if b'"confirmation"' in serialized:
        raise ValueError("confirmation key leaked into development view")
    counts = {
        name: {
            "sources": len(view["splits"][name]["sources"]),
            "cells": len(view["splits"][name]["cells"]),
        }
        for name in ("train", "development")
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "view_sha256": sha256_file(arguments.view),
        "source_canary_sha256": SOURCE_CANARY_SHA256,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "split_counts": counts,
        "confirmation_absent": True,
        "status": "pass",
    }
    digest = write_exclusive_read_only(arguments.out, report)
    print(json.dumps({"out": str(arguments.out), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
