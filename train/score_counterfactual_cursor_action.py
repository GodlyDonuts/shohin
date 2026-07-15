#!/usr/bin/env python3
"""Independent release scorer for R12 cursor-action inference artifacts.

This module deliberately does not import the evaluator.  It requires the exact
receipt SHA-256, re-hashes every bound file, validates the score-free receipt
and raw record schemas, joins cell IDs to the immutable canary gold, and then
computes the preregistered selector and causal-relation statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RAW_SCHEMA = "counterfactual_cursor_action_raw_inference_v1"
RECEIPT_SCHEMA = "counterfactual_cursor_action_inference_receipt_v1"
SCORE_SCHEMA = "counterfactual_cursor_action_independent_score_v1"
CANARY_SCHEMA = "counterfactual_cursor_action_canary_v1"
AUDIT_SCHEMA = "counterfactual_cursor_action_canary_audit_v1"
MANIFEST_SCHEMA = "counterfactual_cursor_action_training_manifest_v1"
CANARY_ID = "ccaa-neural-canary-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ARM_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
BOOTSTRAP_SEED = 2026071504
BOOTSTRAP_REPLICATES = 20_000
ARMS = (
    "orbit_interchange",
    "ordinary_loss",
    "relation_sham",
    "source_only",
    "cursor_table",
    "text_cursor_lora",
)
MATCHED_ARMS = ARMS[:4]
FROZEN_BASE_STEP = 260000
FROZEN_UPDATES = 1_152
EXPECTED_PARAMETERS = {
    "orbit_interchange": 192,
    "ordinary_loss": 192,
    "relation_sham": 192,
    "source_only": 192,
    "cursor_table": 512,
    "text_cursor_lora": 640,
}
EXPECTED_RELATION_COEFFICIENTS = {
    "orbit_interchange": 1.0,
    "ordinary_loss": 0.0,
    "relation_sham": 1.0,
    "source_only": 0.0,
    "cursor_table": 0.0,
    "text_cursor_lora": 0.0,
}

ROOT = Path(__file__).resolve().parents[1]
LIVE_CODE_PATHS = {
    "evaluator": ROOT / "train/eval_counterfactual_cursor_action.py",
    "model": ROOT / "train/model.py",
    "cursor_sidecar": ROOT / "train/counterfactual_cursor_action.py",
    "adapter_factory": ROOT / "train/counterfactual_cursor_action_training.py",
}
SCORER_PATH = Path(__file__).resolve()

CONDITION_LIBRARY = {
    "canonical": (0, 1, 2, 3, 4),
    "clamped_zero": (0, 0, 0, 0, 0),
    "deranged_cycle": (1, 2, 3, 4, 0),
}

CANARY_TOP_KEYS = {
    "schema", "canary_id", "contract_sha256", "tokenizer_sha256",
    "generator_sha256", "implementation_identity", "exposure_contract",
    "label_order", "label_token_ids", "splits", "payload_sha256",
}
SPLIT_KEYS = {
    "geometry", "sources_sha256", "cells_sha256", "sources", "cells",
    "content_groups", "adjacent_pairs", "training_units",
}
SOURCE_KEYS = {
    "schema", "source_id", "split", "renderer_id", "pack_id",
    "permutation_id", "source_text", "prompt", "prompt_token_ids",
    "operation_order", "clause_spans",
}
CELL_KEYS = {
    "schema", "cell_id", "source_id", "cursor", "text_prompt",
    "text_prompt_token_ids", "target_action", "target_index", "target_token_id",
}
AUDIT_KEYS = {
    "schema", "canary_id", "canary_file_sha256", "canary_payload_sha256",
    "contract_sha256", "tokenizer_sha256", "evalgrams_sha256",
    "auditor_sha256", "implementation_identity", "split_summary",
    "cross_split_13gram_counts", "all_checks_pass",
}
BINDING_KEYS = {
    "canary_file_sha256", "canary_payload_sha256", "canary_contract_sha256",
    "canary_audit_file_sha256", "base_checkpoint_sha256",
    "base_checkpoint_step", "adapter_sha256", "adapter_implementation_commit",
    "tokenizer_sha256", "confirmation_sources_sha256",
    "confirmation_cells_sha256", "training_manifest_sha256", "code_sha256",
}
MANIFEST_KEYS = {
    "schema", "arms", "arm_order", "bindings", "all_arms_complete",
    "score_bearing_evaluation_performed",
}
MANIFEST_ARM_KEYS = {
    "arm", "artifact", "artifact_sha256", "initial_adapter_sha256",
    "final_adapter_sha256", "trainable_scalars", "updates",
    "relation_coefficient", "fixed_training_compute_proxy",
}
MANIFEST_BINDING_KEYS = {
    "base_sha256", "base_step", "canary_sha256", "canary_payload_sha256",
    "audit_sha256", "tokenizer_sha256", "implementation_commit",
}
RAW_KEYS = {
    "schema", "arm_name", "adapter_contract", "bindings",
    "condition_contract", "restricted_token_ids", "row_count",
    "condition_count", "forward_count", "inference_records",
}
RECEIPT_KEYS = {
    "schema", "job_identity", "arm_name", "bindings", "row_count",
    "condition_count", "forward_count", "raw_result_sha256", "raw_result_bytes",
}
RAW_RECORD_KEYS = {"cell_id", "conditions"}
INFERENCE_KEYS = {
    "full_vocab_argmax_token_id", "full_vocab_argmax_logit",
    "full_vocab_top_count", "full_vocab_unique_top1",
    "full_vocab_prediction_token_id", "restricted_argmax_index",
    "restricted_argmax_token_id", "restricted_top_count",
    "restricted_unique_top1", "restricted_prediction_token_id",
    "restricted_logits",
}
RAW_FORBIDDEN_KEY_PARTS = ("correct", "accuracy", "score", "target", "gold")
RECEIPT_FORBIDDEN_KEY_PARTS = RAW_FORBIDDEN_KEY_PARTS + (
    "prediction", "argmax", "logit",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def strict_json_loads(payload: bytes | str) -> Any:
    if isinstance(payload, bytes):
        payload = payload.decode("ascii")
    return json.loads(
        payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )


def _absolute(path: str | os.PathLike[str]) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    aliases = {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
        Path("/etc"): Path("/private/etc"),
    }
    for alias, target in aliases.items():
        if (
            alias.is_symlink()
            and alias.resolve() == target
            and (absolute == alias or alias in absolute.parents)
        ):
            return target / absolute.relative_to(alias)
    return absolute


def reject_symlink_components(path: str | os.PathLike[str]) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
    return absolute


def read_regular_file(
    path: str | os.PathLike[str], *, require_read_only: bool = True,
) -> bytes:
    absolute = reject_symlink_components(path)
    metadata = absolute.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"input is not a regular file: {absolute}")
    if require_read_only:
        require(metadata.st_mode & 0o222 == 0, f"input is writable: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


def hash_regular_file(
    path: str | os.PathLike[str], *, require_read_only: bool = True,
) -> str:
    return sha256_bytes(read_regular_file(path, require_read_only=require_read_only))


def validate_sha256(value: str, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def load_json_file(path: str | os.PathLike[str]) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path)
    value = strict_json_loads(payload)
    require(type(value) is dict, f"JSON input is not an object: {path}")
    return value, payload


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def reject_forbidden_keys(value: Any, parts: Sequence[str], label: str) -> None:
    for key in _walk_keys(value):
        lowered = key.lower()
        if any(part in lowered for part in parts):
            raise ValueError(f"{label} contains forbidden key: {key}")


def write_exclusive_read_only_json(
    path: str | os.PathLike[str], value: Mapping[str, Any],
) -> str:
    destination = reject_symlink_components(path)
    require(not os.path.lexists(destination), f"refusing existing output: {destination}")
    parent = reject_symlink_components(destination.parent)
    require(parent.is_dir(), f"output parent is not a directory: {parent}")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = parent / f".{destination.name}.{os.getpid()}.tmp"
    require(not os.path.lexists(temporary), f"temporary output exists: {temporary}")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
            os.fchmod(sink.fileno(), 0o444)
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = destination.lstat()
    require(stat.S_ISREG(metadata.st_mode), "score output is not a regular file")
    require(metadata.st_mode & 0o222 == 0, "score output remained writable")
    return sha256_bytes(payload)


@dataclass(frozen=True)
class GoldIndex:
    canary: dict[str, Any]
    audit: dict[str, Any]
    canary_sha256: str
    audit_sha256: str
    sources: dict[str, dict[str, Any]]
    cells: dict[str, dict[str, Any]]
    source_cells: dict[str, tuple[str, ...]]
    adjacent_pairs: tuple[dict[str, Any], ...]
    content_groups: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrainingManifest:
    sha256: str
    base_sha256: str
    implementation_commit: str
    entries: Mapping[str, Mapping[str, Any]]
    artifact_paths: Mapping[str, Path]


def load_gold_index(
    canary_path: str | os.PathLike[str], audit_path: str | os.PathLike[str],
) -> GoldIndex:
    canary, canary_payload = load_json_file(canary_path)
    audit, audit_payload = load_json_file(audit_path)
    canary_sha256 = sha256_bytes(canary_payload)
    audit_sha256 = sha256_bytes(audit_payload)
    require(set(canary) == CANARY_TOP_KEYS and canary["schema"] == CANARY_SCHEMA,
            "canary schema changed")
    require(canary["canary_id"] == CANARY_ID, "canary ID mismatch")
    require(canary["payload_sha256"] == sha256_bytes(canonical_json_bytes({
        key: value for key, value in canary.items() if key != "payload_sha256"
    })), "canary payload hash mismatch")
    confirmation = canary.get("splits", {}).get("confirmation")
    require(type(confirmation) is dict and set(confirmation) == SPLIT_KEYS,
            "confirmation split schema changed")
    require(confirmation["sources_sha256"] == sha256_bytes(
        canonical_json_bytes(confirmation["sources"])), "source payload hash mismatch")
    require(confirmation["cells_sha256"] == sha256_bytes(
        canonical_json_bytes(confirmation["cells"])), "cell payload hash mismatch")
    require(set(audit) == AUDIT_KEYS and audit["schema"] == AUDIT_SCHEMA,
            "audit schema changed")
    require(audit["canary_id"] == CANARY_ID and audit["all_checks_pass"] is True,
            "audit did not admit this canary")
    require(audit["canary_file_sha256"] == canary_sha256,
            "audit/canary file binding mismatch")
    require(audit["canary_payload_sha256"] == canary["payload_sha256"],
            "audit/canary payload binding mismatch")
    require(audit["contract_sha256"] == canary["contract_sha256"],
            "audit/canary contract mismatch")
    require(audit["tokenizer_sha256"] == canary["tokenizer_sha256"],
            "audit/canary tokenizer mismatch")

    sources: dict[str, dict[str, Any]] = {}
    for source in confirmation["sources"]:
        require(type(source) is dict and set(source) == SOURCE_KEYS,
                "gold source schema changed")
        source_id = source["source_id"]
        require(isinstance(source_id, str) and source_id not in sources,
                "invalid or duplicate gold source")
        sources[source_id] = source
    cells: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for cell in confirmation["cells"]:
        require(type(cell) is dict and set(cell) == CELL_KEYS,
                "gold cell schema changed")
        cell_id = cell["cell_id"]
        source_id = cell["source_id"]
        cursor = cell["cursor"]
        require(source_id in sources and type(cursor) is int and 0 <= cursor < 5,
                "gold cell source/cursor is invalid")
        require(cell_id == f"{source_id}-c{cursor}" and cell_id not in cells,
                "invalid or duplicate gold cell")
        require(cell["target_token_id"] == canary["label_token_ids"][cell["target_index"]],
                "gold token/index mismatch")
        cells[cell_id] = cell
        grouped[source_id].append((cursor, cell_id))
    source_cells = {}
    for source_id, values in grouped.items():
        values.sort()
        require([cursor for cursor, _ in values] == list(range(5)),
                "source does not have exactly five cursor cells")
        source_cells[source_id] = tuple(cell_id for _, cell_id in values)
    require(set(source_cells) == set(sources), "gold source/cell coverage mismatch")
    require(len(sources) == 960 and len(cells) == 4_800,
            "confirmation source/cell geometry changed")
    require(len(confirmation["adjacent_pairs"]) == 1_440,
            "confirmation adjacent-pair geometry changed")
    require(len(confirmation["content_groups"]) == 192,
            "confirmation renderer-group geometry changed")
    return GoldIndex(
        canary=canary,
        audit=audit,
        canary_sha256=canary_sha256,
        audit_sha256=audit_sha256,
        sources=sources,
        cells=cells,
        source_cells=source_cells,
        adjacent_pairs=tuple(confirmation["adjacent_pairs"]),
        content_groups=tuple(confirmation["content_groups"]),
    )


def load_training_manifest(
    manifest_path: str | os.PathLike[str], gold: GoldIndex,
) -> TrainingManifest:
    """Independently verify the completed six-arm run and artifact hashes."""
    absolute_manifest = reject_symlink_components(manifest_path)
    document, payload = load_json_file(absolute_manifest)
    manifest_sha256 = sha256_bytes(payload)
    require(type(document) is dict and set(document) == MANIFEST_KEYS,
            "training manifest schema changed")
    require(document["schema"] == MANIFEST_SCHEMA,
            "training manifest schema mismatch")
    require(document["arm_order"] == list(ARMS),
            "training manifest arm order changed")
    require(document["all_arms_complete"] is True,
            "training manifest is incomplete")
    require(document["score_bearing_evaluation_performed"] is False,
            "training manifest claims prior score-bearing evaluation")

    bindings = document["bindings"]
    require(type(bindings) is dict and set(bindings) == MANIFEST_BINDING_KEYS,
            "training manifest binding schema changed")
    for key in (
        "base_sha256", "canary_sha256", "canary_payload_sha256", "audit_sha256",
        "tokenizer_sha256",
    ):
        validate_sha256(bindings[key], f"training manifest binding {key}")
    require(bindings["base_step"] == FROZEN_BASE_STEP,
            "training manifest base step changed")
    identity = gold.canary.get("implementation_identity")
    require(type(identity) is dict and set(identity) == {"git_commit", "file_sha256"},
            "canary implementation identity is missing")
    implementation_commit = identity["git_commit"]
    require(isinstance(implementation_commit, str)
            and COMMIT_RE.fullmatch(implementation_commit) is not None,
            "canary implementation commit is invalid")
    require(bindings == {
        "base_sha256": bindings["base_sha256"],
        "base_step": FROZEN_BASE_STEP,
        "canary_sha256": gold.canary_sha256,
        "canary_payload_sha256": gold.canary["payload_sha256"],
        "audit_sha256": gold.audit_sha256,
        "tokenizer_sha256": gold.canary["tokenizer_sha256"],
        "implementation_commit": implementation_commit,
    }, "training manifest immutable bindings mismatch")

    arms = document["arms"]
    require(isinstance(arms, list) and len(arms) == len(ARMS),
            "training manifest arm count mismatch")
    entries: dict[str, Mapping[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for expected_arm, entry in zip(ARMS, arms, strict=True):
        require(type(entry) is dict and set(entry) == MANIFEST_ARM_KEYS,
                "training manifest arm schema changed")
        arm = entry["arm"]
        require(arm == expected_arm and arm not in entries,
                "training manifest arm identity mismatch")
        require(entry["artifact"] == f"{arm}/adapter.pt",
                "training manifest artifact path changed")
        require(entry["trainable_scalars"] == EXPECTED_PARAMETERS[arm],
                "training manifest parameter count mismatch")
        require(entry["updates"] == FROZEN_UPDATES,
                "training manifest update count mismatch")
        require(entry["relation_coefficient"] == EXPECTED_RELATION_COEFFICIENTS[arm],
                "training manifest relation coefficient mismatch")
        require(type(entry["fixed_training_compute_proxy"]) is dict,
                "training manifest compute proxy is malformed")
        for key in (
            "artifact_sha256", "initial_adapter_sha256", "final_adapter_sha256",
        ):
            validate_sha256(entry[key], f"training manifest {arm} {key}")
        artifact = reject_symlink_components(
            absolute_manifest.parent / entry["artifact"]
        )
        require(artifact.parent.parent == absolute_manifest.parent,
                "training manifest artifact escaped its output root")
        require(hash_regular_file(artifact) == entry["artifact_sha256"],
                "training manifest arm artifact hash mismatch")
        entries[arm] = entry
        artifact_paths[arm] = artifact

    require(len({entries[arm]["initial_adapter_sha256"] for arm in MATCHED_ARMS}) == 1,
            "information-matched arms did not share one adapter initialization")
    matched_compute = [entries[arm]["fixed_training_compute_proxy"] for arm in MATCHED_ARMS]
    require(all(proxy == matched_compute[0] for proxy in matched_compute[1:]),
            "information-matched arm compute ledgers differ")
    return TrainingManifest(
        sha256=manifest_sha256,
        base_sha256=bindings["base_sha256"],
        implementation_commit=implementation_commit,
        entries=entries,
        artifact_paths=artifact_paths,
    )


def _finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)),
            f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is non-finite")
    return result


def validate_inference_record(record: Mapping[str, Any], restricted_ids: Sequence[int]) -> None:
    require(type(record) is dict and set(record) == INFERENCE_KEYS,
            "raw inference record schema changed")
    full_argmax = record["full_vocab_argmax_token_id"]
    full_count = record["full_vocab_top_count"]
    full_unique = record["full_vocab_unique_top1"]
    require(type(full_argmax) is int and full_argmax >= 0, "invalid full-vocab argmax")
    _finite_number(record["full_vocab_argmax_logit"], "full-vocab maximum logit")
    require(type(full_count) is int and full_count >= 1, "invalid full-vocab tie count")
    require(type(full_unique) is bool and full_unique == (full_count == 1),
            "full-vocab unique-top-1 flag mismatch")
    require(record["full_vocab_prediction_token_id"] == (
        full_argmax if full_unique else None
    ), "full-vocab tie prediction did not fail")

    logits = record["restricted_logits"]
    require(isinstance(logits, list) and len(logits) == 5,
            "restricted logit vector must have length five")
    values = [_finite_number(value, "restricted logit") for value in logits]
    maximum = max(values)
    top_indices = [index for index, value in enumerate(values) if value == maximum]
    expected_index = top_indices[0]
    unique = len(top_indices) == 1
    require(record["restricted_argmax_index"] == expected_index,
            "restricted argmax index mismatch")
    require(record["restricted_argmax_token_id"] == restricted_ids[expected_index],
            "restricted argmax token mismatch")
    require(record["restricted_top_count"] == len(top_indices),
            "restricted tie count mismatch")
    require(record["restricted_unique_top1"] is unique,
            "restricted unique-top-1 flag mismatch")
    require(record["restricted_prediction_token_id"] == (
        restricted_ids[expected_index] if unique else None
    ), "restricted tie prediction did not fail")


@dataclass(frozen=True)
class ChainArtifact:
    arm_name: str
    raw: dict[str, Any]
    receipt: dict[str, Any]
    receipt_sha256: str
    records: dict[str, dict[str, Any]]


def validate_chain(
    *,
    gold: GoldIndex,
    training_manifest: TrainingManifest,
    base_path: str | os.PathLike[str],
    adapter_path: str | os.PathLike[str],
    raw_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    expected_receipt_sha256: str,
) -> ChainArtifact:
    raw, raw_payload = load_json_file(raw_path)
    receipt, receipt_payload = load_json_file(receipt_path)
    raw_sha256 = sha256_bytes(raw_payload)
    receipt_sha256 = sha256_bytes(receipt_payload)
    require(receipt_sha256 == validate_sha256(expected_receipt_sha256, "receipt"),
            "exact receipt SHA-256 mismatch")
    reject_forbidden_keys(raw, RAW_FORBIDDEN_KEY_PARTS, "raw artifact")
    reject_forbidden_keys(receipt, RECEIPT_FORBIDDEN_KEY_PARTS, "receipt")
    require(set(raw) == RAW_KEYS and raw["schema"] == RAW_SCHEMA,
            "raw artifact schema changed")
    require(set(receipt) == RECEIPT_KEYS and receipt["schema"] == RECEIPT_SCHEMA,
            "receipt schema changed")
    arm_name = raw["arm_name"]
    require(arm_name in ARMS and ARM_RE.fullmatch(arm_name) is not None,
            "raw arm name is invalid")
    require(receipt["arm_name"] == arm_name, "receipt/raw arm mismatch")
    require(type(raw["adapter_contract"]) is dict, "adapter contract is not an object")
    require(type(raw["bindings"]) is dict and set(raw["bindings"]) == BINDING_KEYS,
            "raw binding schema changed")
    require(receipt["bindings"] == raw["bindings"], "receipt/raw binding mismatch")
    bindings = raw["bindings"]
    for key in (
        "canary_file_sha256", "canary_payload_sha256",
        "canary_contract_sha256", "canary_audit_file_sha256",
        "base_checkpoint_sha256", "adapter_sha256", "tokenizer_sha256",
        "confirmation_sources_sha256", "confirmation_cells_sha256",
        "training_manifest_sha256",
    ):
        validate_sha256(bindings[key], f"binding {key}")
    require(
        type(bindings["base_checkpoint_step"]) is int
        and bindings["base_checkpoint_step"] == FROZEN_BASE_STEP,
        "bound base checkpoint step is invalid",
    )
    require(
        isinstance(bindings["adapter_implementation_commit"], str)
        and COMMIT_RE.fullmatch(bindings["adapter_implementation_commit"]) is not None,
        "bound adapter implementation commit is invalid",
    )
    require(
        type(bindings["code_sha256"]) is dict
        and set(bindings["code_sha256"]) == set(LIVE_CODE_PATHS),
        "bound code hash schema changed",
    )
    for name, digest in bindings["code_sha256"].items():
        validate_sha256(digest, f"bound code {name}")
    require(bindings["canary_file_sha256"] == gold.canary_sha256,
            "raw does not bind supplied canary")
    require(bindings["canary_payload_sha256"] == gold.canary["payload_sha256"],
            "raw does not bind canary payload")
    require(bindings["canary_contract_sha256"] == gold.canary["contract_sha256"],
            "raw does not bind canary contract")
    require(bindings["canary_audit_file_sha256"] == gold.audit_sha256,
            "raw does not bind supplied audit")
    require(bindings["tokenizer_sha256"] == gold.canary["tokenizer_sha256"],
            "raw tokenizer binding mismatch")
    confirmation = gold.canary["splits"]["confirmation"]
    require(bindings["confirmation_sources_sha256"] == confirmation["sources_sha256"],
            "raw confirmation source binding mismatch")
    require(bindings["confirmation_cells_sha256"] == confirmation["cells_sha256"],
            "raw confirmation cell binding mismatch")
    require(bindings["training_manifest_sha256"] == training_manifest.sha256,
            "raw training-manifest binding mismatch")
    require(hash_regular_file(base_path) == bindings["base_checkpoint_sha256"],
            "bound base checkpoint hash mismatch")
    require(hash_regular_file(adapter_path) == bindings["adapter_sha256"],
            "bound adapter hash mismatch")
    require(_absolute(adapter_path) == training_manifest.artifact_paths[arm_name],
            "adapter path is not the manifest-bound arm artifact")
    require(
        bindings["adapter_sha256"]
        == training_manifest.entries[arm_name]["artifact_sha256"],
        "adapter hash is not the manifest-bound arm artifact",
    )
    require(bindings["base_checkpoint_sha256"] == training_manifest.base_sha256,
            "raw base hash differs from training manifest")
    require(
        bindings["adapter_implementation_commit"]
        == training_manifest.implementation_commit,
        "raw implementation commit differs from training manifest",
    )
    require(
        bindings["code_sha256"] == {
            name: hash_regular_file(path, require_read_only=False)
            for name, path in LIVE_CODE_PATHS.items()
        },
        "live evaluation code differs from receipt binding",
    )

    require(type(receipt["job_identity"]) is dict and set(receipt["job_identity"]) == {
        "scheduler", "job_id", "array_task_id", "attempt_id"
    }, "receipt job identity schema changed")
    require(all(isinstance(value, str) and value
                for value in receipt["job_identity"].values()),
            "receipt job identity is incomplete")
    require(receipt["raw_result_sha256"] == raw_sha256,
            "receipt does not bind raw artifact bytes")
    require(receipt["raw_result_bytes"] == len(raw_payload),
            "receipt raw byte count mismatch")
    for key in ("row_count", "condition_count", "forward_count"):
        require(type(raw[key]) is int and raw[key] > 0, f"raw {key} is invalid")
        require(receipt[key] == raw[key], f"receipt/raw {key} mismatch")

    condition_contract = raw["condition_contract"]
    require(isinstance(condition_contract, list) and bool(condition_contract),
            "condition contract is empty")
    names = []
    for condition in condition_contract:
        require(type(condition) is dict and set(condition) == {"name", "cursor_map"},
                "condition entry schema changed")
        name = condition["name"]
        require(name in CONDITION_LIBRARY, "unknown raw inference condition")
        require(tuple(condition["cursor_map"]) == CONDITION_LIBRARY[name],
                "raw condition cursor map changed")
        names.append(name)
    require(names[0] == "canonical" and len(names) == len(set(names)),
            "condition ordering or uniqueness changed")
    require(raw["condition_count"] == len(names), "condition count mismatch")
    restricted_ids = raw["restricted_token_ids"]
    require(restricted_ids == gold.canary["label_token_ids"],
            "restricted token order differs from canary")

    inference_records = raw["inference_records"]
    require(isinstance(inference_records, list)
            and raw["row_count"] == len(inference_records) == len(gold.cells),
            "raw row count mismatch")
    expected_order = list(gold.cells)
    observed_order = []
    records = {}
    for row in inference_records:
        require(type(row) is dict and set(row) == RAW_RECORD_KEYS,
                "raw row schema changed")
        cell_id = row["cell_id"]
        require(cell_id in gold.cells and cell_id not in records,
                "unknown or duplicate raw cell ID")
        require(type(row["conditions"]) is dict and set(row["conditions"]) == set(names),
                "raw row condition coverage mismatch")
        for name in names:
            validate_inference_record(row["conditions"][name], restricted_ids)
        records[cell_id] = row
        observed_order.append(cell_id)
    require(observed_order == expected_order, "raw cell ordering differs from canary")
    return ChainArtifact(arm_name, raw, receipt, receipt_sha256, records)


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    require(type(numerator) is int and type(denominator) is int and denominator >= 0,
            "invalid ratio")
    require(0 <= numerator <= denominator, "ratio numerator is outside denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "proportion": numerator / denominator if denominator else None,
    }


def prediction_token(
    artifact: ChainArtifact, cell_id: str, condition: str, mode: str,
) -> int | None:
    record = artifact.records[cell_id]["conditions"][condition]
    if mode == "restricted":
        return record["restricted_prediction_token_id"]
    if mode == "full_vocab":
        return record["full_vocab_prediction_token_id"]
    raise ValueError(f"unknown prediction mode: {mode}")


def exact_group_flags(
    gold: GoldIndex, artifact: ChainArtifact, mode: str, condition: str = "canonical",
) -> dict[str, bool]:
    return {
        source_id: all(
            prediction_token(artifact, cell_id, condition, mode)
            == gold.cells[cell_id]["target_token_id"]
            for cell_id in cell_ids
        )
        for source_id, cell_ids in gold.source_cells.items()
    }


def _cell_metrics(
    gold: GoldIndex, artifact: ChainArtifact, mode: str, condition: str,
    cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = list(gold.cells) if cell_ids is None else list(cell_ids)
    correct = 0
    ties = 0
    for cell_id in selected:
        prediction = prediction_token(artifact, cell_id, condition, mode)
        correct += prediction == gold.cells[cell_id]["target_token_id"]
        ties += prediction is None
    return {
        "accuracy": ratio(correct, len(selected)),
        "unique_top1_ties": ratio(ties, len(selected)),
    }


def _group_metrics(flags: Mapping[str, bool]) -> dict[str, Any]:
    return {"exact_five_action_groups": ratio(sum(flags.values()), len(flags))}


def directed_cursor_switch(
    gold: GoldIndex, artifact: ChainArtifact, mode: str,
) -> dict[str, Any]:
    donor_target = 0
    exact_switch = 0
    denominator = 0
    for source_id, cell_ids in gold.source_cells.items():
        del source_id
        for source_cursor in range(5):
            for donor_cursor in range(5):
                if source_cursor == donor_cursor:
                    continue
                denominator += 1
                source_cell = cell_ids[source_cursor]
                donor_cell = cell_ids[donor_cursor]
                source_prediction = prediction_token(
                    artifact, source_cell, "canonical", mode
                )
                donor_prediction = prediction_token(
                    artifact, donor_cell, "canonical", mode
                )
                donor_ok = donor_prediction == gold.cells[donor_cell]["target_token_id"]
                donor_target += donor_ok
                exact_switch += (
                    donor_ok
                    and source_prediction == gold.cells[source_cell]["target_token_id"]
                    and donor_prediction != source_prediction
                )
    return {
        "donor_target": ratio(donor_target, denominator),
        "exact_source_to_donor_switch": ratio(exact_switch, denominator),
    }


def adjacent_equivariance(
    gold: GoldIndex, artifact: ChainArtifact, mode: str,
) -> dict[str, Any]:
    affected_relation = 0
    affected_exact = 0
    unaffected_relation = 0
    unaffected_exact = 0
    affected_count = 0
    unaffected_count = 0
    for pair in gold.adjacent_pairs:
        left = pair["left_source_id"]
        right = pair["right_source_id"]
        swap = pair["swap_index"]
        require(left in gold.source_cells and right in gold.source_cells,
                "adjacent pair references an unknown source")
        for left_cursor, right_cursor in ((swap, swap + 1), (swap + 1, swap)):
            affected_count += 1
            left_cell = gold.source_cells[left][left_cursor]
            right_cell = gold.source_cells[right][right_cursor]
            left_prediction = prediction_token(artifact, left_cell, "canonical", mode)
            right_prediction = prediction_token(artifact, right_cell, "canonical", mode)
            relation = left_prediction is not None and left_prediction == right_prediction
            affected_relation += relation
            affected_exact += (
                relation
                and left_prediction == gold.cells[left_cell]["target_token_id"]
                and right_prediction == gold.cells[right_cell]["target_token_id"]
            )
        for cursor in range(5):
            if cursor in {swap, swap + 1}:
                continue
            unaffected_count += 1
            left_cell = gold.source_cells[left][cursor]
            right_cell = gold.source_cells[right][cursor]
            left_prediction = prediction_token(artifact, left_cell, "canonical", mode)
            right_prediction = prediction_token(artifact, right_cell, "canonical", mode)
            relation = left_prediction is not None and left_prediction == right_prediction
            unaffected_relation += relation
            unaffected_exact += (
                relation
                and left_prediction == gold.cells[left_cell]["target_token_id"]
                and right_prediction == gold.cells[right_cell]["target_token_id"]
            )
    return {
        "affected_cross_position_relation": ratio(affected_relation, affected_count),
        "affected_cross_position_relation_and_correct": ratio(
            affected_exact, affected_count
        ),
        "unaffected_same_position_invariance": ratio(
            unaffected_relation, unaffected_count
        ),
        "unaffected_same_position_invariance_and_correct": ratio(
            unaffected_exact, unaffected_count
        ),
    }


def renderer_invariance(
    gold: GoldIndex, artifact: ChainArtifact, mode: str,
) -> dict[str, Any]:
    invariant = 0
    invariant_exact = 0
    denominator = 0
    for group in gold.content_groups:
        source_ids = group["source_ids"]
        require(isinstance(source_ids, list) and len(source_ids) >= 2,
                "renderer content group is malformed")
        for left, right in itertools.combinations(source_ids, 2):
            require(left in gold.source_cells and right in gold.source_cells,
                    "renderer group references an unknown source")
            for cursor in range(5):
                denominator += 1
                left_cell = gold.source_cells[left][cursor]
                right_cell = gold.source_cells[right][cursor]
                left_prediction = prediction_token(artifact, left_cell, "canonical", mode)
                right_prediction = prediction_token(artifact, right_cell, "canonical", mode)
                relation = left_prediction is not None and left_prediction == right_prediction
                invariant += relation
                invariant_exact += (
                    relation
                    and left_prediction == gold.cells[left_cell]["target_token_id"]
                    and right_prediction == gold.cells[right_cell]["target_token_id"]
                )
    return {
        "prediction_invariance": ratio(invariant, denominator),
        "prediction_invariance_and_correct": ratio(invariant_exact, denominator),
    }


def score_mode(
    gold: GoldIndex, artifact: ChainArtifact, mode: str,
) -> dict[str, Any]:
    canonical_flags = exact_group_flags(gold, artifact, mode)
    by_renderer = {}
    renderer_sources: dict[int, list[str]] = defaultdict(list)
    for source_id, source in gold.sources.items():
        renderer_sources[source["renderer_id"]].append(source_id)
    for renderer_id, source_ids in sorted(renderer_sources.items()):
        renderer_cells = [
            cell_id for source_id in source_ids for cell_id in gold.source_cells[source_id]
        ]
        renderer_flags = {source_id: canonical_flags[source_id] for source_id in source_ids}
        by_renderer[str(renderer_id)] = {
            **_cell_metrics(gold, artifact, mode, "canonical", renderer_cells),
            **_group_metrics(renderer_flags),
        }
    result = {
        "canonical": {
            **_cell_metrics(gold, artifact, mode, "canonical"),
            **_group_metrics(canonical_flags),
            "per_renderer": by_renderer,
            "directed_cursor_switch": directed_cursor_switch(gold, artifact, mode),
            "adjacent_equivariance": adjacent_equivariance(gold, artifact, mode),
            "renderer_invariance": renderer_invariance(gold, artifact, mode),
        },
        "ablations": {},
    }
    condition_names = [item["name"] for item in artifact.raw["condition_contract"]]
    canonical_exact_sources = {
        source_id for source_id, exact in canonical_flags.items() if exact
    }
    conditioned_cells = [
        cell_id for source_id in canonical_exact_sources
        for cell_id in gold.source_cells[source_id]
    ]
    expected = {"clamped_zero": 0.20, "deranged_cycle": 0.0}
    for condition in condition_names:
        if condition == "canonical":
            continue
        all_cells = _cell_metrics(gold, artifact, mode, condition)
        conditioned = _cell_metrics(
            gold, artifact, mode, condition, conditioned_cells
        ) if conditioned_cells else {
            "accuracy": ratio(0, 0), "unique_top1_ties": ratio(0, 0)
        }
        expected_proportion = expected.get(condition)
        observed = conditioned["accuracy"]["proportion"]
        result["ablations"][condition] = {
            "all_cells": all_cells,
            "conditioned_on_canonical_exact_groups": conditioned,
            "canonical_exact_group_count": len(canonical_exact_sources),
            "symbolic_expected_proportion": expected_proportion,
            "absolute_deviation_from_symbolic": (
                abs(observed - expected_proportion)
                if observed is not None and expected_proportion is not None else None
            ),
        }
    return result


def score_arm(gold: GoldIndex, artifact: ChainArtifact) -> dict[str, Any]:
    return {
        "arm_name": artifact.arm_name,
        "receipt_sha256": artifact.receipt_sha256,
        "restricted": score_mode(gold, artifact, "restricted"),
        "full_vocab": score_mode(gold, artifact, "full_vocab"),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values) and 0.0 <= probability <= 1.0, "invalid quantile request")
    ordered = sorted(values)
    index = math.floor(probability * (len(ordered) - 1))
    return ordered[index]


def paired_bootstrap_comparisons(
    gold: GoldIndex,
    treatment: ChainArtifact,
    controls: Mapping[str, ChainArtifact],
    mode: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    require(replicates > 0 and controls, "paired bootstrap requires controls")
    treatment_flags = exact_group_flags(gold, treatment, mode)
    control_flags = {
        name: exact_group_flags(gold, artifact, mode)
        for name, artifact in controls.items()
    }
    require(all(set(flags) == set(treatment_flags) for flags in control_flags.values()),
            "bootstrap source-group alignment mismatch")
    content_clusters = []
    covered_sources = []
    for group in gold.content_groups:
        source_ids = tuple(group["source_ids"])
        require(bool(source_ids) and all(source_id in treatment_flags for source_id in source_ids),
                "bootstrap content group references an unknown source")
        content_clusters.append(source_ids)
        covered_sources.extend(source_ids)
    require(
        len(covered_sources) == len(set(covered_sources)) == len(treatment_flags)
        and set(covered_sources) == set(treatment_flags),
        "bootstrap content groups do not partition confirmation sources",
    )
    source_count = len(treatment_flags)
    observed = {
        name: sum(
            int(treatment_flags[source_id]) - int(flags[source_id])
            for source_id in treatment_flags
        ) / source_count
        for name, flags in control_flags.items()
    }
    distributions = {name: [] for name in controls}
    joint_minimum = []
    rng = random.Random(seed)
    for _ in range(replicates):
        totals = {name: 0 for name in controls}
        sampled = 0
        for _index in range(len(content_clusters)):
            source_ids = content_clusters[rng.randrange(len(content_clusters))]
            for source_id in source_ids:
                sampled += 1
                treatment_value = int(treatment_flags[source_id])
                for name, flags in control_flags.items():
                    totals[name] += treatment_value - int(flags[source_id])
        require(sampled == source_count, "bootstrap stratum accounting drift")
        replicate_values = {name: total / source_count for name, total in totals.items()}
        for name, value in replicate_values.items():
            distributions[name].append(value)
        joint_minimum.append(min(replicate_values.values()))
    control_count = len(controls)
    return {
        "mode": mode,
        "unit": "exact_five_action_source_group",
        "cluster_unit": "content_matched_pack_permutation_across_all_renderers",
        "paired_within_content_cluster": True,
        "seed": seed,
        "replicates": replicates,
        "source_group_count": source_count,
        "content_cluster_count": len(content_clusters),
        "controls": {
            name: {
                "observed_difference": observed[name],
                "one_sided_95_percent_lower_bound": _quantile(
                    distributions[name], 0.05
                ),
                "bonferroni_simultaneous_95_percent_lower_bound": _quantile(
                    distributions[name], 0.05 / control_count
                ),
            }
            for name in controls
        },
        "joint_minimum_difference_one_sided_95_percent_lower_bound": _quantile(
            joint_minimum, 0.05
        ),
    }


def selector_gate_decision(
    treatment_score: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen selector gate to full-vocabulary confirmation metrics."""
    primary = treatment_score["full_vocab"]
    canonical = primary["canonical"]
    checks: dict[str, dict[str, Any]] = {}
    renderer_values = {
        renderer: metrics["accuracy"]["proportion"]
        for renderer, metrics in canonical["per_renderer"].items()
    }
    checks["per_renderer_cell_accuracy"] = {
        "observed": renderer_values,
        "threshold": 0.95,
        "passed": len(renderer_values) == 5
        and all(value is not None and value >= 0.95 for value in renderer_values.values()),
    }
    exact_groups = canonical["exact_five_action_groups"]["proportion"]
    checks["exact_five_action_groups"] = {
        "observed": exact_groups,
        "threshold": 0.90,
        "passed": exact_groups is not None and exact_groups >= 0.90,
    }
    directed = canonical["directed_cursor_switch"][
        "exact_source_to_donor_switch"
    ]["proportion"]
    checks["directed_cursor_switch"] = {
        "observed": directed,
        "threshold": 0.95,
        "passed": directed is not None and directed >= 0.95,
    }
    affected = canonical["adjacent_equivariance"][
        "affected_cross_position_relation_and_correct"
    ]["proportion"]
    unaffected = canonical["adjacent_equivariance"][
        "unaffected_same_position_invariance_and_correct"
    ]["proportion"]
    checks["adjacent_equivariance"] = {
        "observed": {"affected": affected, "unaffected": unaffected},
        "threshold": 0.95,
        "passed": affected is not None and affected >= 0.95
        and unaffected is not None and unaffected >= 0.95,
    }
    renderer = canonical["renderer_invariance"][
        "prediction_invariance_and_correct"
    ]["proportion"]
    checks["renderer_invariance"] = {
        "observed": renderer,
        "threshold": 0.99,
        "passed": renderer is not None and renderer >= 0.99,
    }

    control_checks = {}
    for arm in ("ordinary_loss", "relation_sham"):
        values = comparison["controls"][arm]
        observed = values["observed_difference"]
        lower = values["bonferroni_simultaneous_95_percent_lower_bound"]
        control_checks[arm] = {
            "observed_difference": observed,
            "minimum_difference": 0.10,
            "simultaneous_lower_bound": lower,
            "passed": observed >= 0.10 and lower > 0.0,
        }
    checks["matched_control_advantage"] = {
        "observed": control_checks,
        "passed": all(value["passed"] for value in control_checks.values()),
    }

    ablation_checks = {}
    for condition in ("clamped_zero", "deranged_cycle"):
        metrics = primary["ablations"][condition]
        deviation = metrics["absolute_deviation_from_symbolic"]
        ablation_checks[condition] = {
            "observed": metrics[
                "conditioned_on_canonical_exact_groups"
            ]["accuracy"]["proportion"],
            "expected": metrics["symbolic_expected_proportion"],
            "absolute_deviation": deviation,
            "maximum_absolute_deviation": 0.02,
            "canonical_exact_group_count": metrics["canonical_exact_group_count"],
            "passed": metrics["canonical_exact_group_count"] > 0
            and deviation is not None and deviation <= 0.02,
        }
    checks["cursor_ablations"] = {
        "observed": ablation_checks,
        "passed": all(value["passed"] for value in ablation_checks.values()),
    }
    selector_passed = all(check["passed"] for check in checks.values())
    return {
        "primary_prediction_mode": "full_vocab",
        "checks": checks,
        "selector_checks_passed": selector_passed,
        "decision": (
            "selector_go_executor_pending" if selector_passed else "selector_no_go"
        ),
        "atomic_executor_gate_pending": True,
        "one_call_done_eos_gate_pending": True,
        "reasoning_claim_authorized": False,
    }


def build_score_report(
    gold: GoldIndex,
    treatment: ChainArtifact,
    controls: Mapping[str, ChainArtifact],
) -> dict[str, Any]:
    arms = {"treatment": score_arm(gold, treatment)}
    arms.update({name: score_arm(gold, artifact) for name, artifact in controls.items()})
    primary_controls = {
        name: controls[name] for name in ("ordinary_loss", "relation_sham")
    }
    comparisons = {
        mode: paired_bootstrap_comparisons(gold, treatment, primary_controls, mode)
        for mode in ("restricted", "full_vocab")
    }
    report = {
        "schema": SCORE_SCHEMA,
        "bindings": {
            "canary_sha256": gold.canary_sha256,
            "canary_payload_sha256": gold.canary["payload_sha256"],
            "audit_sha256": gold.audit_sha256,
            "base_sha256": treatment.raw["bindings"]["base_checkpoint_sha256"],
            "training_manifest_sha256": treatment.raw["bindings"][
                "training_manifest_sha256"
            ],
            "scorer_sha256": hash_regular_file(SCORER_PATH, require_read_only=False),
            "treatment_receipt_sha256": treatment.receipt_sha256,
            "control_receipt_sha256": {
                name: artifact.receipt_sha256 for name, artifact in controls.items()
            },
        },
        "scoring_contract": {
            "prediction_modes": ["restricted", "full_vocab"],
            "tie_rule": "non_unique_top1_is_incorrect",
            "directed_switch_rule": "both_endpoints_correct_and_predictions_change",
            "affected_equivariance_rule": "cross_swapped_cursor_predictions_match",
            "unaffected_equivariance_rule": "same_cursor_predictions_match",
            "renderer_invariance_rule": "content_matched_predictions_match",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_cluster": "content_matched_pack_permutation_across_renderers",
        },
        "label_order": gold.canary["label_order"],
        "label_token_ids": gold.canary["label_token_ids"],
        "arms": arms,
        "paired_bootstrap_comparisons": comparisons,
    }
    report["selector_gate"] = selector_gate_decision(
        arms["treatment"], comparisons["full_vocab"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--control",
        action="append",
        nargs=5,
        metavar=("ARM", "ADAPTER", "RAW", "RECEIPT", "RECEIPT_SHA256"),
        default=[],
        help="repeat for each matched control artifact",
    )
    arguments = parser.parse_args()
    destination = reject_symlink_components(arguments.out)
    require(not os.path.lexists(destination), f"refusing existing output: {destination}")

    gold = load_gold_index(arguments.canary, arguments.audit)
    training_manifest = load_training_manifest(arguments.training_manifest, gold)
    treatment = validate_chain(
        gold=gold,
        training_manifest=training_manifest,
        base_path=arguments.base,
        adapter_path=arguments.adapter,
        raw_path=arguments.raw,
        receipt_path=arguments.receipt,
        expected_receipt_sha256=arguments.receipt_sha256,
    )
    controls = {}
    for arm, adapter, raw, receipt, receipt_sha256 in arguments.control:
        require(arm not in controls and arm != "treatment", "duplicate control arm")
        artifact = validate_chain(
            gold=gold,
            training_manifest=training_manifest,
            base_path=arguments.base,
            adapter_path=adapter,
            raw_path=raw,
            receipt_path=receipt,
            expected_receipt_sha256=receipt_sha256,
        )
        require(artifact.arm_name == arm, "control name does not match artifact arm")
        controls[arm] = artifact
    require(treatment.arm_name == "orbit_interchange",
            "paired comparison treatment must be orbit_interchange")
    require(set(controls) == set(ARMS[1:]),
            "release score requires all five preregistered control arms")
    require(
        all(
            artifact.raw["bindings"]["code_sha256"]
            == treatment.raw["bindings"]["code_sha256"]
            for artifact in controls.values()
        ),
        "arm evaluations did not use one identical code identity",
    )
    report = build_score_report(gold, treatment, controls)
    output_sha256 = write_exclusive_read_only_json(arguments.out, report)
    print(json.dumps({
        "schema": SCORE_SCHEMA,
        "out": str(arguments.out),
        "sha256": output_sha256,
        "arms": [treatment.arm_name, *controls],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES if controls else 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
