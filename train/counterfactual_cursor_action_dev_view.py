"""Strict loader for the confirmation-free R12 development view."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "counterfactual_cursor_action_dev_view_v1"
AUDIT_SCHEMA = "counterfactual_cursor_action_dev_view_audit_v1"
SPLIT_NAMES = ("train", "development")
LABEL_TOKEN_IDS = (820, 5498, 4307, 7486, 2165)
EXPECTED = {
    "train": {"sources": 1152, "cells": 5760, "renderers": tuple(range(6)), "packs": 8},
    "development": {"sources": 192, "cells": 960, "renderers": (6, 7), "packs": 4},
}


@dataclass(frozen=True)
class DevelopmentSource:
    source_id: str
    renderer_id: int
    pack_id: int
    permutation_id: int
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class DevelopmentCell:
    source_id: str
    cursor: int
    target_index: int
    target_token_id: int


@dataclass(frozen=True)
class DevelopmentSplit:
    name: str
    sources: tuple[DevelopmentSource, ...]
    cells: tuple[DevelopmentCell, ...]
    source_index_by_id: dict[str, int]


@dataclass(frozen=True)
class DevelopmentView:
    file_sha256: str
    audit_file_sha256: str
    source_canary_sha256: str
    source_audit_sha256: str
    tokenizer_sha256: str
    implementation_commit: str
    splits: tuple[DevelopmentSplit, ...]

    def split(self, name: str) -> DevelopmentSplit:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(name)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular_read_only(path: str | os.PathLike[str], label: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"{label} has a symlink path component")
    metadata = absolute.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o222:
        raise ValueError(f"{label} must be a read-only regular file")
    return absolute.read_bytes()


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _build_split(name: str, raw: Any) -> DevelopmentSplit:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} split must be an object")
    _exact_keys(raw, {"sources", "cells"}, f"{name} split")
    if not isinstance(raw["sources"], list) or not isinstance(raw["cells"], list):
        raise ValueError(f"{name} rows must be arrays")
    expected = EXPECTED[name]
    if len(raw["sources"]) != expected["sources"] or len(raw["cells"]) != expected["cells"]:
        raise ValueError(f"{name} row counts mismatch")

    sources = []
    observed_coordinates = set()
    for item in raw["sources"]:
        if not isinstance(item, dict):
            raise ValueError("source must be an object")
        _exact_keys(
            item,
            {"source_id", "renderer_id", "pack_id", "permutation_id", "prompt_token_ids"},
            "source",
        )
        source_id = item["source_id"]
        renderer_id = _integer(item["renderer_id"], "renderer_id")
        pack_id = _integer(item["pack_id"], "pack_id")
        permutation_id = _integer(item["permutation_id"], "permutation_id")
        tokens = item["prompt_token_ids"]
        if not isinstance(source_id, str) or not source_id.startswith(f"{name}-"):
            raise ValueError("source_id is invalid")
        if renderer_id not in expected["renderers"] or not 0 <= pack_id < expected["packs"]:
            raise ValueError("source coordinate is outside the split")
        if not 0 <= permutation_id < 24:
            raise ValueError("permutation_id is outside [0,23]")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("prompt_token_ids must be a nonempty list")
        if any(type(token) is not int or not 0 <= token < 32000 for token in tokens):
            raise ValueError("prompt_token_ids contains an invalid token")
        coordinate = (renderer_id, pack_id, permutation_id)
        if coordinate in observed_coordinates:
            raise ValueError("duplicate source coordinate")
        observed_coordinates.add(coordinate)
        sources.append(DevelopmentSource(
            source_id, renderer_id, pack_id, permutation_id, tuple(tokens),
        ))
    expected_coordinates = {
        (renderer, pack, permutation)
        for renderer in expected["renderers"]
        for pack in range(expected["packs"])
        for permutation in range(24)
    }
    if observed_coordinates != expected_coordinates:
        raise ValueError(f"{name} source geometry mismatch")
    source_index = {source.source_id: index for index, source in enumerate(sources)}
    if len(source_index) != len(sources):
        raise ValueError("duplicate source_id")

    cells = []
    for index, item in enumerate(raw["cells"]):
        if not isinstance(item, dict):
            raise ValueError("cell must be an object")
        _exact_keys(
            item, {"source_id", "cursor", "target_index", "target_token_id"}, "cell",
        )
        source_id = item["source_id"]
        cursor = _integer(item["cursor"], "cursor")
        target_index = _integer(item["target_index"], "target_index")
        target_token_id = _integer(item["target_token_id"], "target_token_id")
        expected_source = sources[index // 5].source_id
        if source_id != expected_source or cursor != index % 5:
            raise ValueError("cells are not canonical source-major cursor rows")
        if not 0 <= target_index < 5 or target_token_id != LABEL_TOKEN_IDS[target_index]:
            raise ValueError("cell target is invalid")
        cells.append(DevelopmentCell(source_id, cursor, target_index, target_token_id))
    return DevelopmentSplit(name, tuple(sources), tuple(cells), source_index)


def load_development_view(
    view_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    *,
    expected_view_sha256: str,
    expected_audit_sha256: str,
) -> DevelopmentView:
    view_bytes = _read_regular_read_only(view_path, "development view")
    audit_bytes = _read_regular_read_only(audit_path, "development view audit")
    if sha256_bytes(view_bytes) != expected_view_sha256:
        raise ValueError("development view SHA-256 mismatch")
    if sha256_bytes(audit_bytes) != expected_audit_sha256:
        raise ValueError("development view audit SHA-256 mismatch")
    document = json.loads(view_bytes, object_pairs_hook=_pairs)
    audit = json.loads(audit_bytes, object_pairs_hook=_pairs)
    if not isinstance(document, dict) or not isinstance(audit, dict):
        raise ValueError("development view documents must be objects")
    _exact_keys(
        document,
        {
            "schema", "source_canary_sha256", "source_audit_sha256", "tokenizer_sha256",
            "implementation_commit", "implementation_file_sha256", "label_token_ids",
            "splits", "payload_sha256",
        },
        "development view",
    )
    if document["schema"] != SCHEMA or document["label_token_ids"] != list(LABEL_TOKEN_IDS):
        raise ValueError("development view identity mismatch")
    payload = {key: value for key, value in document.items() if key != "payload_sha256"}
    if document["payload_sha256"] != sha256_bytes(canonical_json(payload)):
        raise ValueError("development view payload hash mismatch")
    if (
        not isinstance(document["splits"], dict)
        or set(document["splits"]) != set(SPLIT_NAMES)
    ):
        raise ValueError("development view must contain only train and development")
    _exact_keys(
        audit,
        {
            "schema", "view_sha256", "source_canary_sha256", "source_audit_sha256",
            "tokenizer_sha256", "split_counts", "confirmation_absent", "status",
        },
        "development view audit",
    )
    if (
        audit["schema"] != AUDIT_SCHEMA
        or audit["view_sha256"] != expected_view_sha256
        or audit["source_canary_sha256"] != document["source_canary_sha256"]
        or audit["source_audit_sha256"] != document["source_audit_sha256"]
        or audit["tokenizer_sha256"] != document["tokenizer_sha256"]
        or audit["confirmation_absent"] is not True
        or audit["status"] != "pass"
    ):
        raise ValueError("development view audit binding mismatch")
    splits = tuple(_build_split(name, document["splits"][name]) for name in SPLIT_NAMES)
    expected_counts = {
        split.name: {"sources": len(split.sources), "cells": len(split.cells)}
        for split in splits
    }
    if audit["split_counts"] != expected_counts:
        raise ValueError("development view audit counts mismatch")
    commit = document["implementation_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("development view implementation commit is invalid")
    return DevelopmentView(
        expected_view_sha256,
        expected_audit_sha256,
        document["source_canary_sha256"],
        document["source_audit_sha256"],
        document["tokenizer_sha256"],
        commit,
        splits,
    )
