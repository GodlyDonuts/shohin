#!/usr/bin/env python3
"""Frozen, fail-closed adjudicator for the R12 ACW CPU experiment.

The input is a hash-bound manifest containing one record for every frozen run.
Each record binds a real checkpoint, evaluation-domain root and manifest,
trainer-bundle root and manifest, evaluator report, and second evaluator output.
The adjudicator opens and hashes those artifacts, checks their transitive
bindings, and executes a third evaluator replay in a fresh interpreter before
it accepts any metric.
Confirmation records contain only their public commitment and index; this module
never retrieves or accepts confirmation seed material.

The required run matrix is eight scored arms by three development and three
confirmation identities, plus one direct-state diagnostic for each development
identity.  There are no CLI overrides for identities, thresholds, arms, label
checkpoints, resource fields, or the claim boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

MANIFEST_SCHEMA = "r12_acw_adjudication_manifest_v3"
MANIFEST_PROTOCOL = "R12-ACW-ADJUDICATION-MANIFEST-v3"
DECISION_SCHEMA = "r12_acw_adjudication_decision_v3"
DECISION_PROTOCOL = "R12-ACW-ADJUDICATION-DECISION-v3"
EVALUATION_PROTOCOL = "R12-ACW-CAUSAL-EVALUATION-v2"
GENERATOR_PROTOCOL = "R12-ACW-HIDDEN-BASIS-v2"
TRAINING_PROTOCOL = "R12-ACW-TRAINER-v2"
TRAINER_BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v4"
PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v3"
PILOT_COMPARISON_PROTOCOL = "R12-ACW-PILOT-REPLAY-COMPARISON-v3"
TRAIN_LEDGER_PROTOCOL = "R12-ACW-TRAIN-RESOURCE-LEDGER-v2"
INFERENCE_LEDGER_PROTOCOL = "R12-ACW-INFERENCE-RESOURCE-LEDGER-v2"

DEVELOPMENT_SEEDS = (2026071601, 2026071602, 2026071603)
CONFIRMATION_COMMITMENTS = (
    "35102b3974877e8547b9b9c74156c63b71d467820f752301be21721b0f58e9a1",
    "737a6d6a76c3cdbfd07d84c83cfec5491cf13afeb8e077421af789cb652baa7f",
    "0e60eb70f2193ea57710db1f2cf9d6f93cf9b8e310b1b2cf5f4ea2694851854d",
)

SCORED_ARMS = (
    "acw",
    "dense_categorical",
    "addressed_continuous",
    "gru",
    "packet_token_transformer",
    "uniform_query_acw",
    "answer_motor",
    "source_retained",
)
DIRECT_STATE_ARM = "direct_state_acw"
VALID_EQUAL_LABEL_ARCHITECTURE_CONTROLS = (
    "dense_categorical",
    "addressed_continuous",
    "gru",
    "packet_token_transformer",
    "answer_motor",
)

EVALUATION_DEPTHS = (8, 16, 32, 64, 65)
LABEL_CHECKPOINTS = tuple(8192 + 4096 * round_index for round_index in range(13))
FINAL_SCALAR_LABELS = 57_344
OPTIMIZER_UPDATES = 3_400
PUBLIC_HISTORIES = 2_048
PUBLIC_QUERIES = 24
NEW_READER_QUERIES = 8
EVENT_WORD_HISTORIES = 256
MAX_ORACLE_CANDIDATE_EVALUATIONS = 147_456
MAX_WITNESS_SELECTIONS = 6_144

DEPTH_THRESHOLDS = {
    8: (0.99, 0.95),
    32: (0.99, 0.92),
    64: (0.98, 0.90),
    65: (0.97, 0.85),
}
DIRECT_STATE_SCALAR_FLOOR = 0.99
DIRECT_STATE_STATE_FLOOR = 0.95
DONOR_SCALAR_FLOOR = 0.99
SHUFFLE_SCALAR_CEILING = 1.0 / 17.0 + 0.02
NEW_READER_SCALAR_FLOOR = 0.98
NEW_READER_STATE_FLOOR = 0.90
PRIMARY_STATE_FLOOR = 0.90
CONTROL_MARGIN_FLOOR = 0.10

EVALUATOR_CLAIM_BOUNDARY = (
    "Frozen synthetic state-transport evaluation only; no language, autonomous "
    "controller, novelty, or reasoning claim."
)
SOURCE_SWAP_BASIS = (
    "Structural: the frozen ACW reader accepts only packet and query; "
    "source bytes are absent from its callable interface."
)
INFERENCE_LEDGER_SCOPE = (
    "all required public depths, post-freeze reader evaluations, and applicable "
    "causal interventions"
)
TRAINING_PROFILE_SCOPE = "one complete forward+backward+AdamW answer-loss update"
DIRECT_TRAINING_PROFILE_SCOPE = (
    "one complete direct-state forward+backward+AdamW update"
)
INFERENCE_PROFILE_SCOPE = (
    "one source-deleted literal-state inference batch including all events and reader"
)
FLOP_COUNTING_CONTRACT = (
    "PyTorch CPU profiler operator-reported FLOPs; unsupported operators are "
    "listed as uncounted rather than imputed."
)
TRANSIENT_MEMORY_CONTRACT = (
    "Runtime CPU profiler allocations; largest operator and self-operator "
    "allocations are reported without claiming allocator-wide liveness."
)
BOUNDED_CLAIM = (
    "A GO supports only the preregistered R12 synthetic hidden-basis result: a "
    "scheduled, source-deleted three-register ACW transports state across the "
    "frozen CPU family and clears the stated equal-label architecture controls. "
    "It is not evidence for an autonomous controller, language transfer, general "
    "reasoning, novelty, CGBR optimality, or a Shohin sidecar result."
)

HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_BINARY_BYTES = 4 * 1024 * 1024 * 1024
EVALUATOR_TIMEOUT_SECONDS = 4 * 60 * 60

ARM_PARAMETERS = {
    "acw": 26_008,
    "dense_categorical": 26_250,
    "addressed_continuous": 26_008,
    "gru": 26_036,
    "packet_token_transformer": 25_872,
    "uniform_query_acw": 26_008,
    "answer_motor": 25_939,
    "source_retained": 166_801,
    "direct_state_acw": 26_008,
}

_CATEGORICAL_BITS = 3.0 * math.log2(17)
ARM_RESOURCES = {
    "acw": (_CATEGORICAL_BITS, 3, "uint8", 204, 0, 0, True),
    "dense_categorical": (_CATEGORICAL_BITS, 3, "uint8", 204, 0, 0, True),
    "addressed_continuous": (96.0, 12, "float32", 12, 0, 0, True),
    "gru": (1248.0, 156, "float32", 156, 0, 0, True),
    "packet_token_transformer": (
        _CATEGORICAL_BITS,
        3,
        "uint8",
        204,
        672,
        0,
        True,
    ),
    "uniform_query_acw": (_CATEGORICAL_BITS, 3, "uint8", 204, 0, 0, True),
    "answer_motor": (6144.0, 768, "float32", 768, 0, 384, True),
    "source_retained": (7168.0, 896, "float32", 896, 0, 384, False),
    "direct_state_acw": (_CATEGORICAL_BITS, 3, "uint8", 204, 0, 0, True),
}

RUN_KEYS = {
    "arm",
    "checkpoint",
    "dataset",
    "trainer_bundle",
    "evaluation_report",
    "replay_report",
}
REFERENCE_KEYS = {"path", "sha256"}
ROOTED_MANIFEST_KEYS = {"root", "manifest"}
DATASET_KEYS = {
    "protocol",
    "seed_identity",
    "seed_fingerprint",
    "field_size",
    "dimension",
    "source_dim",
    "event_dim",
    "event_count",
    "event_address_counts",
    "public_queries",
    "new_queries",
    "counts",
    "evaluation_depths",
    "visited_buckets",
    "depth_counts",
    "arrays",
    "payload_sha256",
}
ARRAY_RECORD_KEYS = {"bytes", "dtype", "shape", "sha256"}
BUNDLE_KEYS = {
    "protocol",
    "source_manifest_payload_sha256",
    "seed_identity",
    "data_replay_verification",
    "query_schedule_sha256",
    "query_schedule_kind",
    "pilot_report_payload_sha256",
    "pilot_report_sha256",
    "pilot_replay_comparison_payload_sha256",
    "pilot_replay_comparison_sha256",
    "pilot_artifacts",
    "arrays",
    "files",
    "oracle_paths_exported",
    "payload_sha256",
}
BUNDLE_DATA_REPLAY_KEYS = {
    "protocol",
    "seed_identity",
    "seed_fingerprint",
    "source_manifest_payload_sha256",
    "regenerated_manifest_payload_sha256",
    "array_registry_sha256",
    "arrays_verified",
    "public_arrays_verified",
    "oracle_arrays_verified",
}
BUNDLE_FILE_KEYS = {"bytes", "rows", "sha256"}
BUNDLE_ARTIFACT_RECORD_KEYS = {"bytes", "sha256"}
BUNDLE_PILOT_ARTIFACTS = (
    "pilot/report.json",
    "pilot/replay_comparison.json",
    "pilot/cgb_schedule.jsonl",
    "pilot/uniform_schedule.jsonl",
)
BUNDLE_ARRAYS = (
    "public/event_features.npy",
    "public/event_addresses.npy",
    "public/train/source_features.npy",
    "public/train/event_ids.npy",
    "public/train/lengths.npy",
    "public/train/initial_queries.npy",
    "public/train/initial_answers.npy",
)
CHECKPOINT_KEYS = {
    "protocol",
    "arm",
    "seed",
    "dataset_manifest_payload_sha256",
    "source_manifest_payload_sha256",
    "curriculum_sha256",
    "query_schedule_sha256",
    "query_schedule_kind",
    "pilot_report_payload_sha256",
    "parameters",
    "training_report",
    "label_efficiency_models",
    "scientific_identity",
    "model",
}
ACCURACY_KEYS = {
    "scalar_correct",
    "scalar_total",
    "scalar_accuracy",
    "state_exact",
    "state_total",
    "state_exactness",
}
EVALUATION_BASE_KEYS = {
    "protocol",
    "checkpoint_sha256",
    "checkpoint_arm",
    "model_arm",
    "parameters",
    "dataset_manifest_payload_sha256",
    "seed_identity",
    "optimizer_seed",
    "query_schedule_kind",
    "pilot_report_payload_sha256",
    "training_evidence",
    "scientific_identity",
    "public_depths",
    "new_reader",
    "compiled_sparse_control",
    "claim_boundary",
    "payload_sha256",
}
EVALUATION_ACW_KEYS = {
    "packet_interventions",
    "write_legality",
    "event_words",
}
NEW_READER_KEYS = {
    "updates",
    "state_dim",
    "reader_parameters",
    "loss_first",
    "loss_last",
    "depths",
}
INTERVENTION_KEYS = {
    "donor_following",
    "shuffled_against_original",
    "held_packet_source_swap_predictions_identical",
    "source_swap_basis",
    "donor_different_truth_fraction",
}
WRITE_LEGALITY_KEYS = {"unaddressed_registers_checked", "illegal_writes"}
EVENT_WORD_KEYS = {
    "histories",
    "equivalent_prediction_query_equivalence",
    "equivalent_a",
    "equivalent_b",
    "non_equivalent_target_separator_rate",
    "non_equivalent_prediction_separator_rate",
    "non_equivalent_a",
    "non_equivalent_b",
}
TRAIN_LEDGER_KEYS = {
    "protocol",
    "checkpoint_sha256",
    "dataset_manifest_payload_sha256",
    "curriculum_sha256",
    "scalar_labels",
    "state_auxiliary_labels",
    "optimizer_updates",
    "optimizer_evaluations",
    "oracle_candidate_evaluations",
    "witness_selections",
    "trainable_parameters",
    "semantic_state_bits",
    "persistent_training_state_bytes",
    "declared_transient_token_bytes",
    "parameter_matched_primary",
    "mixed_precision",
    "extra_source_bytes",
    "oracle_access",
    "confirmation_preimage_access",
    "train_flops",
    "train_wall_seconds",
    "flop_measurement_complete",
}
INFERENCE_LEDGER_KEYS = {
    "protocol",
    "checkpoint_sha256",
    "dataset_manifest_payload_sha256",
    "scope",
    "trainable_parameters",
    "semantic_state_bits",
    "persistent_state_bytes",
    "persistent_state_dtype",
    "declared_transient_token_bytes",
    "peak_transient_bytes",
    "extra_source_bytes",
    "kv_cache_bytes",
    "mixed_precision",
    "event_updates",
    "query_reads",
    "inference_flops",
    "inference_wall_seconds",
    "flop_measurement_complete",
}
LABEL_EFFICIENCY_KEYS = {
    "round",
    "labels",
    "optimizer_updates",
    "model_tensor_sha256",
    "depth_64",
}
TRAINING_EVIDENCE_KEYS = {
    "trainer_bundle_manifest_payload_sha256",
    "curriculum_sha256",
    "query_schedule_sha256",
    "updates",
    "labels",
    "resource_ledger",
    "resource_measurements",
}
NATIVE_RESOURCE_LEDGER_KEYS = {
    "trainable_parameters",
    "semantic_state_bits",
    "persistent_evaluation_bytes",
    "persistent_evaluation_dtype",
    "persistent_training_state_bytes",
    "declared_transient_token_bytes",
    "parameter_matched_primary",
}
RESOURCE_MEASUREMENTS_KEYS = {"training", "inference"}
PROFILE_MEASUREMENT_KEYS = {
    "scope",
    "batch_size",
    "active_events",
    "wall_seconds",
    "process_peak_rss_bytes",
    "profiler_event_count",
    "operator_inventory",
    "uncounted_operator_names",
    "operator_inventory_complete",
    "operator_reported_flops",
    "largest_operator_allocation_bytes",
    "largest_self_operator_allocation_bytes",
    "total_positive_operator_allocations_bytes",
    "flop_counting_contract",
    "transient_memory_contract",
}
OPERATOR_INVENTORY_ENTRY_KEYS = {
    "name",
    "calls",
    "operator_reported_flops",
    "positive_allocation_bytes",
    "positive_self_allocation_bytes",
}
TRAINING_PROFILE_EXTRA_KEYS = {"optimizer_included"}
DIRECT_TRAINING_PROFILE_EXTRA_KEYS = {
    "optimizer_included",
    "state_auxiliary_weight",
}
COMPILED_CONTROL_KEYS = {
    "depths",
    "external_event_updates",
    "event_arithmetic",
    "external_query_reads",
    "query_arithmetic",
    "resource_ledger",
    "claim_boundary",
}
COMPILED_DEPTH_EXTRA_KEYS = {
    "transition_state_exact",
    "transition_state_total",
    "transition_state_exactness",
}
COMPILED_RESOURCE_KEYS = {
    "trainable_parameters",
    "persistent_state_bytes",
    "event_table_bytes",
    "query_table_bytes",
    "runtime",
}


class EvidenceError(ValueError):
    """A frozen evidence-contract violation."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EvidenceError(
            "noncanonical_json", f"value is not canonical JSON: {exc}"
        ) from exc


def with_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy carrying SHA-256 over canonical JSON without the hash field."""

    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EvidenceError(
            "schema_mismatch",
            f"{label} keys differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("invalid_json_shape", f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError("invalid_json_shape", f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError("invalid_integer", f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError("integer_out_of_range", f"{label} must be >= {minimum}")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("invalid_number", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError("nonfinite_number", f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError("number_out_of_range", f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise EvidenceError("number_out_of_range", f"{label} must be <= {maximum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceError("invalid_boolean", f"{label} must be boolean")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise EvidenceError("invalid_sha256", f"{label} must be lowercase SHA-256 hex")
    return value


def _verify_payload_hash(value: dict[str, Any], label: str) -> str:
    recorded = _hash(value.get("payload_sha256"), f"{label}.payload_sha256")
    payload = dict(value)
    payload.pop("payload_sha256")
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != recorded:
        raise EvidenceError(
            "payload_hash_mismatch",
            f"{label} payload hash mismatch: expected {recorded}, got {observed}",
        )
    return recorded


def _read_regular_file(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot open {path}: {exc}"
        ) from exc
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(
                "artifact_not_regular", f"artifact is not regular: {path}"
            )
        size = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            size += len(block)
            if size > MAX_JSON_BYTES:
                raise EvidenceError(
                    "json_artifact_too_large", f"artifact exceeds limit: {path}"
                )
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise EvidenceError(
                "artifact_changed_during_read", f"artifact changed: {path}"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _hash_regular_file(Path(path), "artifact")[1]


def _hash_regular_file(path: Path, label: str) -> tuple[int, str]:
    """Hash one stable regular file without materializing it in memory."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot open {label}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(
                "artifact_not_regular", f"{label} is not a regular file"
            )
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            size += len(block)
            if size > MAX_BINARY_BYTES:
                raise EvidenceError(
                    "binary_artifact_too_large", f"{label} exceeds the limit"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise EvidenceError("artifact_changed_during_read", f"{label} changed")
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(
                    "duplicate_json_key", f"{label} repeats key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceError("nonfinite_json_number", f"{label} contains {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid_json", f"cannot parse {label}: {exc}") from exc
    return _object(parsed, label)


def _reference(value: Any, label: str) -> tuple[str, str]:
    reference = _object(value, label)
    _expect_keys(reference, REFERENCE_KEYS, label)
    raw_path = reference["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError("invalid_artifact_path", f"{label}.path must be nonempty")
    return raw_path, _hash(reference["sha256"], f"{label}.sha256")


def _resolve_artifact_path(raw_path: str, base: Path, label: str) -> Path:
    candidate = Path(raw_path)
    if "\x00" in raw_path or any(part == ".." for part in candidate.parts):
        raise EvidenceError("invalid_artifact_path", f"{label} contains traversal")
    if not candidate.is_absolute():
        candidate = base / candidate
    if candidate.is_symlink():
        raise EvidenceError("artifact_symlink_forbidden", f"{label} is a symlink")
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot resolve {label}: {exc}"
        ) from exc


def _verify_file_reference(
    value: Any,
    label: str,
    base: Path,
) -> tuple[dict[str, str | int], Path]:
    raw_path, expected_hash = _reference(value, label)
    resolved = _resolve_artifact_path(raw_path, base, label)
    size, actual_hash = _hash_regular_file(resolved, label)
    if actual_hash != expected_hash:
        raise EvidenceError(
            "artifact_hash_mismatch",
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
    return {"path": raw_path, "sha256": actual_hash, "bytes": size}, resolved


def _verify_json_reference(
    value: Any,
    label: str,
    base: Path,
) -> tuple[dict[str, Any], dict[str, str], Path]:
    raw_path, expected_hash = _reference(value, label)
    resolved = _resolve_artifact_path(raw_path, base, label)
    raw, actual_hash = _read_regular_file(resolved)
    if actual_hash != expected_hash:
        raise EvidenceError(
            "artifact_hash_mismatch",
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
    return _parse_json(raw, label), {"path": raw_path, "sha256": actual_hash}, resolved


def _verify_rooted_json_reference(
    value: Any,
    label: str,
    base: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    rooted = _object(value, label)
    _expect_keys(rooted, ROOTED_MANIFEST_KEYS, label)
    raw_root = rooted["root"]
    if not isinstance(raw_root, str) or not raw_root:
        raise EvidenceError("invalid_artifact_path", f"{label}.root must be nonempty")
    root = _resolve_artifact_path(raw_root, base, f"{label}.root")
    if not root.is_dir():
        raise EvidenceError(
            "artifact_not_directory", f"{label}.root is not a directory"
        )
    manifest, manifest_binding, manifest_path = _verify_json_reference(
        rooted["manifest"], f"{label}.manifest", base
    )
    if manifest_path != root / "manifest.json":
        raise EvidenceError(
            "root_manifest_mismatch",
            f"{label}.manifest must be the manifest.json inside its bound root",
        )
    return (
        manifest,
        {
            "root": raw_root,
            "manifest": manifest_binding,
        },
        root,
    )


def _safe_child(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise EvidenceError(
            "invalid_artifact_path", f"{label} is not a safe relative path"
        )
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot resolve {label} root"
        ) from exc
    unresolved = resolved_root / candidate
    if unresolved.is_symlink():
        raise EvidenceError("artifact_symlink_forbidden", f"{label} is a symlink")
    resolved = _resolve_artifact_path(str(unresolved), resolved_root, label)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceError(
            "artifact_path_escape", f"{label} escapes its root"
        ) from exc
    return resolved


def _load_bound_array(
    root: Path,
    relative: str,
    raw_record: Any,
    label: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
) -> np.ndarray:
    record = _object(raw_record, f"{label}.record")
    _expect_keys(record, ARRAY_RECORD_KEYS, f"{label}.record")
    expected_bytes = _integer(record["bytes"], f"{label}.record.bytes", minimum=1)
    recorded_hash = _hash(record["sha256"], f"{label}.record.sha256")
    recorded_dtype = record["dtype"]
    if not isinstance(recorded_dtype, str) or not recorded_dtype:
        raise EvidenceError("array_schema_mismatch", f"{label} has no dtype")
    recorded_shape = _list(record["shape"], f"{label}.record.shape")
    if not recorded_shape:
        raise EvidenceError("array_schema_mismatch", f"{label} has no shape")
    for dimension in recorded_shape:
        _integer(dimension, f"{label}.record.shape", minimum=1)
    path = _safe_child(root, relative, label)
    observed_bytes, observed_hash = _hash_regular_file(path, label)
    if observed_bytes != expected_bytes or observed_hash != recorded_hash:
        raise EvidenceError(
            "array_artifact_mismatch",
            f"{label} bytes or SHA-256 differ from its manifest record",
        )
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise EvidenceError(
            "invalid_npy_artifact", f"cannot load {label}: {exc}"
        ) from exc
    if list(array.shape) != recorded_shape or str(array.dtype) != recorded_dtype:
        raise EvidenceError(
            "array_schema_mismatch",
            f"{label} NumPy header differs from its manifest record",
        )
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        raise EvidenceError(
            "dataset_array_shape_mismatch",
            f"{label} shape must be {list(expected_shape)}, got {list(array.shape)}",
        )
    if expected_dtype is not None and str(array.dtype) != expected_dtype:
        raise EvidenceError(
            "dataset_array_dtype_mismatch",
            f"{label} dtype must be {expected_dtype}, got {array.dtype}",
        )
    return array


def _registered_identity(identity: Any, label: str) -> tuple[str, int]:
    value = _object(identity, label)
    kind = value.get("kind")
    if kind == "pilot":
        raise EvidenceError("pilot_seed_forbidden", f"{label} is the non-scored pilot")
    if kind == "development":
        _expect_keys(value, {"kind", "seed"}, label)
        seed = _integer(value["seed"], f"{label}.seed")
        if seed not in DEVELOPMENT_SEEDS:
            raise EvidenceError(
                "unregistered_seed_identity", f"{label} seed is not registered"
            )
        return "development", DEVELOPMENT_SEEDS.index(seed)
    if kind == "confirmation":
        _expect_keys(value, {"kind", "index", "commitment"}, label)
        index = _integer(value["index"], f"{label}.index", minimum=0)
        if index >= len(CONFIRMATION_COMMITMENTS):
            raise EvidenceError(
                "unregistered_seed_identity", f"{label} index is not registered"
            )
        commitment = _hash(value["commitment"], f"{label}.commitment")
        if commitment != CONFIRMATION_COMMITMENTS[index]:
            raise EvidenceError(
                "confirmation_commitment_mismatch",
                f"{label} commitment does not match registered index {index}",
            )
        return "confirmation", index
    raise EvidenceError(
        "unregistered_seed_identity", f"{label} has unknown kind {kind!r}"
    )


def _required_dataset_arrays() -> set[str]:
    return set(_required_dataset_specs())


def _validate_dataset_manifest(
    manifest: dict[str, Any], label: str
) -> tuple[tuple[str, int], dict[str, Any]]:
    _expect_keys(manifest, DATASET_KEYS, label)
    payload_hash = _verify_payload_hash(manifest, label)
    if manifest["protocol"] != GENERATOR_PROTOCOL:
        raise EvidenceError("dataset_protocol_mismatch", f"{label} has wrong protocol")
    identity_key = _registered_identity(
        manifest["seed_identity"], f"{label}.seed_identity"
    )
    _hash(manifest["seed_fingerprint"], f"{label}.seed_fingerprint")
    frozen_values = {
        "field_size": 17,
        "dimension": 3,
        "source_dim": 96,
        "event_dim": 96,
        "event_count": 48,
        "public_queries": 24,
        "new_queries": 8,
    }
    for field, expected in frozen_values.items():
        if _integer(manifest[field], f"{label}.{field}") != expected:
            raise EvidenceError(
                "dataset_schema_mismatch", f"{label}.{field} must be {expected}"
            )
    if manifest["event_address_counts"] != {"0": 16, "1": 16, "2": 16}:
        raise EvidenceError("dataset_schema_mismatch", f"{label} address counts differ")
    if manifest["counts"] != {
        "train": 4096,
        "adaptation": 1024,
        "evaluation_per_depth": 2048,
    }:
        raise EvidenceError("dataset_schema_mismatch", f"{label} split counts differ")
    if manifest["evaluation_depths"] != list(EVALUATION_DEPTHS):
        raise EvidenceError(
            "dataset_schema_mismatch", f"{label} evaluation depths differ"
        )
    _object(manifest["visited_buckets"], f"{label}.visited_buckets")
    _object(manifest["depth_counts"], f"{label}.depth_counts")
    arrays = _object(manifest["arrays"], f"{label}.arrays")
    missing = _required_dataset_arrays() - set(arrays)
    if missing:
        raise EvidenceError(
            "dataset_schema_mismatch", f"{label} lacks arrays {sorted(missing)}"
        )
    for path, raw_record in arrays.items():
        record = _object(raw_record, f"{label}.arrays[{path!r}]")
        _expect_keys(record, ARRAY_RECORD_KEYS, f"{label}.arrays[{path!r}]")
        _integer(record["bytes"], f"{label}.arrays[{path!r}].bytes", minimum=1)
        if not isinstance(record["dtype"], str) or not record["dtype"]:
            raise EvidenceError(
                "dataset_schema_mismatch", f"{label} array dtype is empty"
            )
        shape = _list(record["shape"], f"{label}.arrays[{path!r}].shape")
        if not shape:
            raise EvidenceError(
                "dataset_schema_mismatch", f"{label} array shape is empty"
            )
        for dimension in shape:
            _integer(dimension, f"{label}.arrays[{path!r}].shape", minimum=1)
        _hash(record["sha256"], f"{label}.arrays[{path!r}].sha256")
    return identity_key, {
        "payload_sha256": payload_hash,
        "seed_identity": manifest["seed_identity"],
    }


def _required_dataset_specs() -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {
        "public/event_features.npy": ((48, 96), "float32"),
        "public/event_addresses.npy": ((48,), "int8"),
        "public/train/source_features.npy": ((4096, 96), "float32"),
        "public/train/event_ids.npy": ((4096, 8), "int16"),
        "public/train/lengths.npy": ((4096,), "int16"),
        "public/train/initial_queries.npy": ((4096, 2), "int8"),
        "public/train/initial_answers.npy": ((4096, 2), "int8"),
        "oracle/train/source_states.npy": ((4096, 3), "int8"),
        "oracle/train/trajectory_states.npy": ((4096, 9, 3), "int8"),
        "oracle/train/final_states.npy": ((4096, 3), "int8"),
        "oracle/train/public_answers.npy": ((4096, 24), "int8"),
        "oracle/domain/basis.npy": ((3, 3), "int8"),
        "oracle/domain/events.npy": ((48, 5), "int8"),
        "oracle/domain/query_coefficients.npy": ((24, 3), "int8"),
        "oracle/domain/query_offsets.npy": ((24,), "int8"),
        "oracle/domain/query_permutations.npy": ((24, 17), "int8"),
        "oracle/domain/new_query_coefficients.npy": ((8, 3), "int8"),
        "oracle/domain/new_query_offsets.npy": ((8,), "int8"),
        "oracle/domain/new_query_permutations.npy": ((8, 17), "int8"),
    }
    split_specs = {
        "source_features.npy": (96, "float32"),
        "event_ids.npy": (None, "int16"),
        "lengths.npy": (None, "int16"),
        "final_states.npy": (3, "int8"),
        "public_answers.npy": (24, "int8"),
        "new_answers.npy": (8, "int8"),
    }
    for prefix, histories, depth in (
        ("oracle/adaptation", 1024, 8),
        *(
            (f"oracle/evaluation/depth_{value:03d}", 2048, value)
            for value in EVALUATION_DEPTHS
        ),
    ):
        for name, (width, dtype) in split_specs.items():
            if name == "event_ids.npy":
                shape = (histories, depth)
            elif name == "lengths.npy":
                shape = (histories,)
            else:
                assert width is not None
                shape = (histories, width)
            specs[f"{prefix}/{name}"] = (shape, dtype)
        specs[f"{prefix}/source_states.npy"] = ((histories, 3), "int8")
        specs[f"{prefix}/trajectory_states.npy"] = (
            (histories, depth + 1, 3),
            "int8",
        )
    return specs


def _answers_from_state(
    states: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    raw = (
        states.astype(np.int16) @ coefficients.astype(np.int16).T
        + offsets.astype(np.int16)[None, :]
    ) % 17
    query_ids = np.arange(coefficients.shape[0], dtype=np.int64)[None, :]
    return permutations[query_ids, raw].astype(np.int8)


def _validate_history_truth(
    loaded: dict[str, np.ndarray],
    *,
    data_prefix: str,
    truth_prefix: str,
    events: np.ndarray,
    query_coefficients: np.ndarray,
    query_offsets: np.ndarray,
    query_permutations: np.ndarray,
    new_query_coefficients: np.ndarray | None,
    new_query_offsets: np.ndarray | None,
    new_query_permutations: np.ndarray | None,
    exact_depth: int | None,
    label: str,
) -> None:
    event_ids = np.asarray(loaded[f"{data_prefix}/event_ids.npy"])
    lengths = np.asarray(loaded[f"{data_prefix}/lengths.npy"])
    source_states = np.asarray(loaded[f"{truth_prefix}/source_states.npy"])
    trajectory = np.asarray(loaded[f"{truth_prefix}/trajectory_states.npy"])
    final_states = np.asarray(loaded[f"{truth_prefix}/final_states.npy"])
    if bool(((source_states < 0) | (source_states >= 17)).any()):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} source state leaves F_17"
        )
    if not np.array_equal(trajectory[:, 0], source_states):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} trajectory origin differs"
        )
    if exact_depth is None:
        if bool(((lengths < 0) | (lengths > event_ids.shape[1])).any()):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} lengths are out of range"
            )
    elif not bool((lengths == exact_depth).all()):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} lengths are not exact"
        )
    state = source_states.copy()
    row_ids = np.arange(len(state))
    for step in range(event_ids.shape[1]):
        active = step < lengths
        ids = event_ids[:, step]
        if bool(((ids[active] < 0) | (ids[active] >= len(events))).any()):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} active event ID is invalid"
            )
        if bool((ids[~active] != -1).any()):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} inactive event ID is not -1"
            )
        active_rows = row_ids[active]
        selected = events[ids[active].astype(np.int64)]
        destination = selected[:, 0].astype(np.int64)
        source = selected[:, 1].astype(np.int64)
        updated = state.copy()
        updated[active_rows, destination] = (
            selected[:, 2].astype(np.int16)
            * state[active_rows, destination].astype(np.int16)
            + selected[:, 3].astype(np.int16)
            * state[active_rows, source].astype(np.int16)
            + selected[:, 4].astype(np.int16)
        ) % 17
        expected_trajectory = np.full((len(state), 3), -1, dtype=np.int8)
        expected_trajectory[active] = updated[active]
        if not np.array_equal(trajectory[:, step + 1], expected_trajectory):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} trajectory replay differs"
            )
        state = updated
    if not np.array_equal(state, final_states):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} final state replay differs"
        )
    public_answers = np.asarray(loaded[f"{truth_prefix}/public_answers.npy"])
    expected_public = _answers_from_state(
        state, query_coefficients, query_offsets, query_permutations
    )
    if not np.array_equal(public_answers, expected_public):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} public answers differ"
        )
    if new_query_coefficients is not None:
        assert new_query_offsets is not None and new_query_permutations is not None
        new_answers = np.asarray(loaded[f"{truth_prefix}/new_answers.npy"])
        expected_new = _answers_from_state(
            state,
            new_query_coefficients,
            new_query_offsets,
            new_query_permutations,
        )
        if not np.array_equal(new_answers, expected_new):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} new answers differ"
            )


def _validate_dataset_tree(
    root: Path,
    manifest: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Open, hash, and validate every array in an evaluation-domain root."""

    arrays = _object(manifest["arrays"], f"{label}.arrays")
    specs = _required_dataset_specs()
    if not set(specs).issubset(arrays):
        raise EvidenceError(
            "dataset_schema_mismatch",
            f"{label} lacks required arrays {sorted(set(specs) - set(arrays))}",
        )
    loaded: dict[str, np.ndarray] = {}
    for relative, record in sorted(arrays.items()):
        expected = specs.get(relative)
        array = _load_bound_array(
            root,
            relative,
            record,
            f"{label}.arrays[{relative!r}]",
            expected_shape=None if expected is None else expected[0],
            expected_dtype=None if expected is None else expected[1],
        )
        if relative in specs:
            loaded[relative] = array

    addresses = np.asarray(loaded["public/event_addresses.npy"])
    if not np.array_equal(
        np.bincount(addresses.astype(np.int64), minlength=3),
        np.asarray([16, 16, 16]),
    ):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} event-address bank is not 16/16/16"
        )
    events = np.asarray(loaded["oracle/domain/events.npy"])
    if (
        bool(((events < 0) | (events >= 17)).any())
        or bool((events[:, :2] >= 3).any())
        or not np.array_equal(addresses, events[:, 0])
    ):
        raise EvidenceError("dataset_semantic_mismatch", f"{label} events leave F_17")
    query_coefficients = np.asarray(loaded["oracle/domain/query_coefficients.npy"])
    query_offsets = np.asarray(loaded["oracle/domain/query_offsets.npy"])
    query_permutations = np.asarray(loaded["oracle/domain/query_permutations.npy"])
    new_query_coefficients = np.asarray(
        loaded["oracle/domain/new_query_coefficients.npy"]
    )
    new_query_offsets = np.asarray(loaded["oracle/domain/new_query_offsets.npy"])
    new_query_permutations = np.asarray(
        loaded["oracle/domain/new_query_permutations.npy"]
    )
    for bank_label, coefficients, offsets, permutations in (
        ("public", query_coefficients, query_offsets, query_permutations),
        (
            "new",
            new_query_coefficients,
            new_query_offsets,
            new_query_permutations,
        ),
    ):
        expected_permutation = np.arange(17, dtype=permutations.dtype)
        if (
            bool(((coefficients < 0) | (coefficients >= 17)).any())
            or bool(((offsets < 0) | (offsets >= 17)).any())
            or any(
                not np.array_equal(np.sort(row), expected_permutation)
                for row in permutations
            )
        ):
            raise EvidenceError(
                "dataset_semantic_mismatch",
                f"{label} {bank_label} query bank is invalid",
            )

    _validate_history_truth(
        loaded,
        data_prefix="public/train",
        truth_prefix="oracle/train",
        events=events,
        query_coefficients=query_coefficients,
        query_offsets=query_offsets,
        query_permutations=query_permutations,
        new_query_coefficients=None,
        new_query_offsets=None,
        new_query_permutations=None,
        exact_depth=None,
        label=f"{label}.train",
    )
    initial_queries = np.asarray(loaded["public/train/initial_queries.npy"])
    initial_answers = np.asarray(loaded["public/train/initial_answers.npy"])
    train_answers = np.asarray(loaded["oracle/train/public_answers.npy"])
    if (
        bool(((initial_queries < 0) | (initial_queries >= 24)).any())
        or bool((initial_queries[:, 0] == initial_queries[:, 1]).any())
        or not np.array_equal(
            initial_answers,
            np.take_along_axis(train_answers, initial_queries.astype(np.int64), axis=1),
        )
    ):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} public initial labels differ"
        )

    for prefix, exact_depth in (
        ("oracle/adaptation", None),
        *(
            (f"oracle/evaluation/depth_{depth:03d}", depth)
            for depth in EVALUATION_DEPTHS
        ),
    ):
        _validate_history_truth(
            loaded,
            data_prefix=prefix,
            truth_prefix=prefix,
            events=events,
            query_coefficients=query_coefficients,
            query_offsets=query_offsets,
            query_permutations=query_permutations,
            new_query_coefficients=new_query_coefficients,
            new_query_offsets=new_query_offsets,
            new_query_permutations=new_query_permutations,
            exact_depth=exact_depth,
            label=f"{label}.{prefix}",
        )
        source = np.asarray(loaded[f"{prefix}/source_features.npy"])
        if not bool(np.isfinite(source).all()):
            raise EvidenceError(
                "dataset_semantic_mismatch", f"{label} source features are non-finite"
            )
    if not bool(np.isfinite(loaded["public/train/source_features.npy"]).all()):
        raise EvidenceError(
            "dataset_semantic_mismatch", f"{label} train source features are non-finite"
        )
    return {
        "arrays_hashed_and_opened": len(arrays),
        "required_array_shapes_verified": len(specs),
    }


def _validate_curriculum(
    root: Path,
    files: dict[str, Any],
    initial_queries: np.ndarray,
    initial_answers: np.ndarray,
    label: str,
) -> tuple[str, str]:
    if set(files) != {"curriculum.jsonl"}:
        raise EvidenceError(
            "bundle_schema_mismatch", f"{label}.files must contain curriculum.jsonl"
        )
    record = _object(files["curriculum.jsonl"], f"{label}.files.curriculum.jsonl")
    _expect_keys(record, BUNDLE_FILE_KEYS, f"{label}.files.curriculum.jsonl")
    expected_bytes = _integer(
        record["bytes"], f"{label}.files.curriculum.jsonl.bytes", minimum=1
    )
    expected_rows = _integer(
        record["rows"], f"{label}.files.curriculum.jsonl.rows", minimum=1
    )
    if expected_rows != FINAL_SCALAR_LABELS:
        raise EvidenceError(
            "label_count_mismatch", f"{label} curriculum row count differs"
        )
    expected_hash = _hash(record["sha256"], f"{label}.files.curriculum.jsonl.sha256")
    path = _safe_child(root, "curriculum.jsonl", f"{label}.curriculum")
    raw, actual_hash = _read_regular_file(path)
    if len(raw) != expected_bytes or actual_hash != expected_hash:
        raise EvidenceError(
            "bundle_file_mismatch", f"{label} curriculum bytes or hash differ"
        )
    lines = raw.splitlines()
    if len(lines) != expected_rows or not raw.endswith(b"\n"):
        raise EvidenceError(
            "bundle_file_mismatch", f"{label} curriculum framing differs"
        )
    pairs: set[tuple[int, int]] = set()
    per_history = np.zeros(4096, dtype=np.int16)
    round_counts = np.zeros(13, dtype=np.int64)
    round_zero: list[list[tuple[int, int]]] = [[] for _ in range(4096)]
    schedule_digest = hashlib.sha256()
    for index, line in enumerate(lines):
        row = _parse_json(line, f"{label}.curriculum[{index}]")
        _expect_keys(
            row,
            {"history_id", "query_id", "answer", "round"},
            f"{label}.curriculum[{index}]",
        )
        if canonical_json_bytes(row) != line:
            raise EvidenceError(
                "bundle_curriculum_mismatch",
                f"{label} curriculum row is not canonical JSON",
            )
        schedule_digest.update(
            canonical_json_bytes(
                {
                    "history_id": row["history_id"],
                    "query_id": row["query_id"],
                    "round": row["round"],
                }
            )
            + b"\n"
        )
        history_id = _integer(
            row["history_id"], f"{label}.curriculum[{index}].history_id", minimum=0
        )
        query_id = _integer(
            row["query_id"], f"{label}.curriculum[{index}].query_id", minimum=0
        )
        answer = _integer(
            row["answer"], f"{label}.curriculum[{index}].answer", minimum=0
        )
        round_index = _integer(
            row["round"], f"{label}.curriculum[{index}].round", minimum=0
        )
        if history_id >= 4096 or query_id >= 24 or answer >= 17 or round_index >= 13:
            raise EvidenceError(
                "bundle_curriculum_mismatch",
                f"{label} curriculum value is out of range",
            )
        pair = (history_id, query_id)
        if pair in pairs:
            raise EvidenceError(
                "bundle_curriculum_mismatch", f"{label} repeats a history/query pair"
            )
        pairs.add(pair)
        per_history[history_id] += 1
        round_counts[round_index] += 1
        if round_index == 0:
            round_zero[history_id].append((query_id, answer))
    expected_round_counts = np.full(13, 4096, dtype=np.int64)
    expected_round_counts[0] = 8192
    if not bool((per_history == 14).all()) or not np.array_equal(
        round_counts, expected_round_counts
    ):
        raise EvidenceError(
            "bundle_curriculum_mismatch", f"{label} curriculum allocation differs"
        )
    for history_id, entries in enumerate(round_zero):
        if len(entries) != 2:
            raise EvidenceError(
                "bundle_curriculum_mismatch", f"{label} round zero is incomplete"
            )
        ordered = sorted(entries)
        if [entry[0] for entry in ordered] != sorted(
            initial_queries[history_id].tolist()
        ):
            raise EvidenceError(
                "bundle_curriculum_mismatch", f"{label} initial queries differ"
            )
        observed = {query: answer for query, answer in entries}
        for query, answer in zip(
            initial_queries[history_id], initial_answers[history_id], strict=True
        ):
            if observed[int(query)] != int(answer):
                raise EvidenceError(
                    "bundle_curriculum_mismatch", f"{label} initial answers differ"
                )
    return actual_hash, schedule_digest.hexdigest()


def _open_bundle_artifact(
    root: Path,
    relative: str,
    record: Any,
    label: str,
) -> tuple[bytes, str]:
    record = _object(record, f"{label}.{relative}")
    _expect_keys(
        record,
        BUNDLE_ARTIFACT_RECORD_KEYS,
        f"{label}.{relative}",
    )
    expected_bytes = _integer(record["bytes"], f"{label}.{relative}.bytes", minimum=1)
    expected_hash = _hash(record["sha256"], f"{label}.{relative}.sha256")
    raw, observed_hash = _read_regular_file(
        _safe_child(root, relative, f"{label}.{relative}")
    )
    if len(raw) != expected_bytes or observed_hash != expected_hash:
        raise EvidenceError(
            "bundle_pilot_artifact_mismatch",
            f"{label} differs from opened artifact {relative}",
        )
    return raw, observed_hash


def _validate_bundle_pilot_artifacts(
    root: Path,
    manifest: dict[str, Any],
    *,
    schedule_kind: str,
    schedule_hash: str,
    label: str,
) -> dict[str, Any]:
    registry = _object(manifest["pilot_artifacts"], f"{label}.pilot_artifacts")
    if set(registry) != set(BUNDLE_PILOT_ARTIFACTS):
        raise EvidenceError(
            "bundle_schema_mismatch",
            f"{label} pilot-artifact registry differs",
        )
    opened = {
        relative: _open_bundle_artifact(
            root,
            relative,
            registry[relative],
            f"{label}.pilot_artifacts",
        )
        for relative in BUNDLE_PILOT_ARTIFACTS
    }
    report = _parse_json(opened["pilot/report.json"][0], f"{label}.pilot_report")
    comparison = _parse_json(
        opened["pilot/replay_comparison.json"][0],
        f"{label}.pilot_comparison",
    )
    report_payload = _verify_payload_hash(report, f"{label}.pilot_report")
    comparison_payload = _verify_payload_hash(comparison, f"{label}.pilot_comparison")
    if (
        report.get("protocol") != PILOT_PROTOCOL
        or report_payload != manifest["pilot_report_payload_sha256"]
        or opened["pilot/report.json"][1] != manifest["pilot_report_sha256"]
    ):
        raise EvidenceError(
            "bundle_pilot_binding_mismatch",
            f"{label} pilot report binding differs",
        )
    if (
        comparison.get("protocol") != PILOT_COMPARISON_PROTOCOL
        or comparison_payload != manifest["pilot_replay_comparison_payload_sha256"]
        or opened["pilot/replay_comparison.json"][1]
        != manifest["pilot_replay_comparison_sha256"]
    ):
        raise EvidenceError(
            "bundle_pilot_binding_mismatch",
            f"{label} pilot comparison binding differs",
        )

    pilot_identity = _validate_scientific_identity(
        report.get("scientific_identity"), f"{label}.pilot_report.scientific_identity"
    )

    schedule_names = {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}
    schedules = _object(report.get("schedules"), f"{label}.pilot_report.schedules")
    if set(schedules) != schedule_names:
        raise EvidenceError(
            "bundle_pilot_binding_mismatch",
            f"{label} pilot schedule registry differs",
        )
    for name in schedule_names:
        record = _object(schedules[name], f"{label}.pilot_report.schedules.{name}")
        _expect_keys(
            record,
            BUNDLE_FILE_KEYS,
            f"{label}.pilot_report.schedules.{name}",
        )
        raw, observed_hash = opened[f"pilot/{name}"]
        if (
            _integer(
                record["bytes"],
                f"{label}.pilot_report.schedules.{name}.bytes",
                minimum=1,
            )
            != len(raw)
            or _integer(
                record["rows"],
                f"{label}.pilot_report.schedules.{name}.rows",
                minimum=1,
            )
            != len(raw.splitlines())
            or _hash(
                record["sha256"],
                f"{label}.pilot_report.schedules.{name}.sha256",
            )
            != observed_hash
        ):
            raise EvidenceError(
                "bundle_pilot_binding_mismatch",
                f"{label} pilot schedule differs: {name}",
            )
    if (
        schedules[schedule_kind]["sha256"] != schedule_hash
        or opened[f"pilot/{schedule_kind}"][1] != schedule_hash
    ):
        raise EvidenceError(
            "bundle_schedule_binding_mismatch",
            f"{label} selected pilot schedule differs from consumed curriculum",
        )

    expected_common = {"report.json", *schedule_names}
    common_files = _object(
        comparison.get("common_files"), f"{label}.pilot_comparison.common_files"
    )
    recomputation = _object(
        comparison.get("independent_recomputation_sha256"),
        f"{label}.pilot_comparison.independent_recomputation_sha256",
    )
    if set(common_files) != expected_common or set(recomputation) != expected_common:
        raise EvidenceError(
            "bundle_pilot_binding_mismatch",
            f"{label} pilot comparison registry differs",
        )
    for name in expected_common:
        record = _object(
            common_files[name], f"{label}.pilot_comparison.common_files.{name}"
        )
        _expect_keys(
            record,
            BUNDLE_ARTIFACT_RECORD_KEYS,
            f"{label}.pilot_comparison.common_files.{name}",
        )
        raw, observed_hash = opened[f"pilot/{name}"]
        if (
            _integer(
                record["bytes"],
                f"{label}.pilot_comparison.common_files.{name}.bytes",
                minimum=1,
            )
            != len(raw)
            or _hash(
                record["sha256"],
                f"{label}.pilot_comparison.common_files.{name}.sha256",
            )
            != observed_hash
            or _hash(
                recomputation[name],
                f"{label}.pilot_comparison.independent_recomputation_sha256.{name}",
            )
            != observed_hash
        ):
            raise EvidenceError(
                "bundle_pilot_binding_mismatch",
                f"{label} pilot comparison differs: {name}",
            )
    if (
        comparison.get("reports_byte_identical") is not True
        or comparison.get("schedules_byte_identical") is not True
        or comparison.get("independently_recomputed") is not True
        or comparison.get("dataset_manifest_payload_sha256")
        != report.get("dataset_manifest_payload_sha256")
        or comparison.get("scientific_identity") != pilot_identity
    ):
        raise EvidenceError(
            "bundle_pilot_binding_mismatch",
            f"{label} pilot comparison differs from its report",
        )
    return {
        "pilot_report_payload_sha256": report_payload,
        "pilot_replay_comparison_payload_sha256": comparison_payload,
        "pilot_replay_comparison_sha256": opened["pilot/replay_comparison.json"][1],
        "pilot_scientific_identity": pilot_identity,
        "pilot_artifacts_opened": len(opened),
    }


def _validate_trainer_bundle(
    root: Path,
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_summary: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    del root, manifest, dataset_manifest, dataset_summary, label
    raise EvidenceError(
        "pilot_anchor_required",
        "canonical trainer evidence is disabled until the verified public pilot "
        "artifact registry is committed and pushed as an external anchor",
    )


def _validate_unanchored_trainer_bundle_structure(
    root: Path,
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_summary: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    _expect_keys(manifest, BUNDLE_KEYS, label)
    payload_hash = _verify_payload_hash(manifest, label)
    if manifest["protocol"] != TRAINER_BUNDLE_PROTOCOL:
        raise EvidenceError("bundle_protocol_mismatch", f"{label} has wrong protocol")
    if manifest["source_manifest_payload_sha256"] != dataset_summary["payload_sha256"]:
        raise EvidenceError(
            "bundle_dataset_mismatch", f"{label} is bound to another source dataset"
        )
    if manifest["seed_identity"] != dataset_summary["seed_identity"]:
        raise EvidenceError(
            "bundle_seed_identity_mismatch", f"{label} seed identity differs"
        )
    schedule_hash = _hash(
        manifest["query_schedule_sha256"], f"{label}.query_schedule_sha256"
    )
    schedule_kind = manifest["query_schedule_kind"]
    if schedule_kind not in {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
        raise EvidenceError(
            "query_schedule_kind_mismatch", f"{label} schedule kind is not registered"
        )
    _hash(
        manifest["pilot_report_payload_sha256"], f"{label}.pilot_report_payload_sha256"
    )
    _hash(manifest["pilot_report_sha256"], f"{label}.pilot_report_sha256")
    _hash(
        manifest["pilot_replay_comparison_payload_sha256"],
        f"{label}.pilot_replay_comparison_payload_sha256",
    )
    _hash(
        manifest["pilot_replay_comparison_sha256"],
        f"{label}.pilot_replay_comparison_sha256",
    )
    replay = manifest["data_replay_verification"]
    if dataset_summary["seed_identity"].get("kind") == "development":
        replay = _object(replay, f"{label}.data_replay_verification")
        _expect_keys(
            replay, BUNDLE_DATA_REPLAY_KEYS, f"{label}.data_replay_verification"
        )
        if (
            replay["protocol"] != "R12-ACW-DATA-REPLAY-v1"
            or replay["seed_identity"] != dataset_summary["seed_identity"]
            or replay["source_manifest_payload_sha256"]
            != dataset_summary["payload_sha256"]
            or replay["regenerated_manifest_payload_sha256"]
            != dataset_summary["payload_sha256"]
            or min(
                _integer(
                    replay[name],
                    f"{label}.data_replay_verification.{name}",
                    minimum=1,
                )
                for name in (
                    "arrays_verified",
                    "public_arrays_verified",
                    "oracle_arrays_verified",
                )
            )
            <= 0
        ):
            raise EvidenceError(
                "bundle_data_replay_mismatch",
                f"{label} deterministic data replay differs from the dataset",
            )
        _hash(
            replay["seed_fingerprint"],
            f"{label}.data_replay_verification.seed_fingerprint",
        )
        _hash(
            replay["array_registry_sha256"],
            f"{label}.data_replay_verification.array_registry_sha256",
        )
    elif replay is not None:
        raise EvidenceError(
            "bundle_data_replay_mismatch",
            f"{label} confirmation bundle may not claim public-seed replay",
        )
    if (
        _integer(
            manifest["oracle_paths_exported"],
            f"{label}.oracle_paths_exported",
            minimum=0,
        )
        != 0
    ):
        raise EvidenceError("bundle_oracle_exposure", f"{label} exports oracle paths")
    if any(
        "oracle" in part.lower()
        for path in root.rglob("*")
        for part in path.relative_to(root).parts
    ):
        raise EvidenceError(
            "bundle_oracle_exposure", f"{label} contains an oracle-named path"
        )
    arrays = _object(manifest["arrays"], f"{label}.arrays")
    if set(arrays) != set(BUNDLE_ARRAYS):
        raise EvidenceError(
            "bundle_schema_mismatch", f"{label} public array set differs"
        )
    bundle_specs = {
        "public/event_features.npy": ((48, 96), "float32"),
        "public/event_addresses.npy": ((48,), "int8"),
        "public/train/source_features.npy": ((4096, 96), "float32"),
        "public/train/event_ids.npy": ((4096, 8), "int16"),
        "public/train/lengths.npy": ((4096,), "int16"),
        "public/train/initial_queries.npy": ((4096, 2), "int8"),
        "public/train/initial_answers.npy": ((4096, 2), "int8"),
    }
    loaded = {
        relative: _load_bound_array(
            root,
            relative,
            arrays[relative],
            f"{label}.arrays[{relative!r}]",
            expected_shape=bundle_specs[relative][0],
            expected_dtype=bundle_specs[relative][1],
        )
        for relative in BUNDLE_ARRAYS
    }
    copied = BUNDLE_ARRAYS[:5]
    source_arrays = _object(dataset_manifest["arrays"], f"{label}.source_arrays")
    for relative in copied:
        if arrays[relative] != source_arrays.get(relative):
            raise EvidenceError(
                "bundle_array_source_mismatch",
                f"{label} changes copied array {relative}",
            )
    initial_queries = np.asarray(loaded["public/train/initial_queries.npy"])
    initial_answers = np.asarray(loaded["public/train/initial_answers.npy"])
    if bool(((initial_queries < 0) | (initial_queries >= 24)).any()) or bool(
        ((initial_answers < 0) | (initial_answers >= 17)).any()
    ):
        raise EvidenceError(
            "bundle_curriculum_mismatch", f"{label} initial labels are out of range"
        )
    if bool((initial_queries[:, 0] == initial_queries[:, 1]).any()):
        raise EvidenceError(
            "bundle_curriculum_mismatch", f"{label} repeats an initial query"
        )
    curriculum_hash, derived_schedule_hash = _validate_curriculum(
        root,
        _object(manifest["files"], f"{label}.files"),
        initial_queries,
        initial_answers,
        label,
    )
    if derived_schedule_hash != schedule_hash:
        raise EvidenceError(
            "bundle_schedule_binding_mismatch",
            f"{label} curriculum-derived schedule hash differs",
        )
    pilot_artifacts = _validate_bundle_pilot_artifacts(
        root,
        manifest,
        schedule_kind=schedule_kind,
        schedule_hash=derived_schedule_hash,
        label=label,
    )
    return {
        "payload_sha256": payload_hash,
        "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
        "seed_identity": dataset_summary["seed_identity"],
        "query_schedule_sha256": schedule_hash,
        "query_schedule_kind": schedule_kind,
        "pilot_report_payload_sha256": pilot_artifacts["pilot_report_payload_sha256"],
        "pilot_replay_comparison_payload_sha256": pilot_artifacts[
            "pilot_replay_comparison_payload_sha256"
        ],
        "pilot_replay_comparison_sha256": pilot_artifacts[
            "pilot_replay_comparison_sha256"
        ],
        "pilot_scientific_identity": pilot_artifacts["pilot_scientific_identity"],
        "curriculum_sha256": curriculum_hash,
        "arrays_hashed_and_opened": len(arrays),
        "pilot_artifacts_opened": pilot_artifacts["pilot_artifacts_opened"],
    }


def _validate_accuracy(
    value: Any,
    label: str,
    *,
    histories: int,
    queries: int,
) -> dict[str, float | int]:
    metric = _object(value, label)
    _expect_keys(metric, ACCURACY_KEYS, label)
    scalar_total = _integer(metric["scalar_total"], f"{label}.scalar_total", minimum=1)
    state_total = _integer(metric["state_total"], f"{label}.state_total", minimum=1)
    if state_total != histories or scalar_total != histories * queries:
        raise EvidenceError(
            "metric_denominator_mismatch",
            f"{label} must contain {histories} histories and {queries} queries each",
        )
    scalar_correct = _integer(
        metric["scalar_correct"], f"{label}.scalar_correct", minimum=0
    )
    state_exact = _integer(metric["state_exact"], f"{label}.state_exact", minimum=0)
    if scalar_correct > scalar_total or state_exact > state_total:
        raise EvidenceError(
            "metric_count_mismatch", f"{label} correct count exceeds total"
        )
    scalar_accuracy = _number(
        metric["scalar_accuracy"], f"{label}.scalar_accuracy", minimum=0.0, maximum=1.0
    )
    state_exactness = _number(
        metric["state_exactness"], f"{label}.state_exactness", minimum=0.0, maximum=1.0
    )
    observed_scalar = scalar_correct / scalar_total
    observed_state = state_exact / state_total
    if not math.isclose(scalar_accuracy, observed_scalar, rel_tol=0.0, abs_tol=1e-6):
        raise EvidenceError(
            "metric_count_mismatch", f"{label} scalar ratio disagrees with counts"
        )
    if not math.isclose(state_exactness, observed_state, rel_tol=0.0, abs_tol=1e-6):
        raise EvidenceError(
            "metric_count_mismatch", f"{label} state ratio disagrees with counts"
        )
    return {
        "scalar_correct": scalar_correct,
        "scalar_total": scalar_total,
        "scalar_accuracy": observed_scalar,
        "state_exact": state_exact,
        "state_total": state_total,
        "state_exactness": observed_state,
    }


def _validate_scientific_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _object(value, label)
    _expect_keys(identity, {"scientific_commit", "scientific_path_sha256"}, label)
    commit = identity["scientific_commit"]
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise EvidenceError(
            "scientific_identity_mismatch", f"{label} commit is invalid"
        )
    paths = _object(
        identity["scientific_path_sha256"], f"{label}.scientific_path_sha256"
    )
    if not paths:
        raise EvidenceError(
            "scientific_identity_mismatch", f"{label} path set is empty"
        )
    for path, digest in paths.items():
        if not isinstance(path, str) or not path:
            raise EvidenceError(
                "scientific_identity_mismatch", f"{label} has an empty path"
            )
        _hash(digest, f"{label}.scientific_path_sha256[{path!r}]")
    return identity


def _checkpoint_arms(logical_arm: str) -> tuple[str, str]:
    if logical_arm == "uniform_query_acw":
        return "acw", "acw"
    if logical_arm == DIRECT_STATE_ARM:
        return DIRECT_STATE_ARM, "acw"
    return logical_arm, logical_arm


def _expected_optimizer_seed(seed_identity: dict[str, Any]) -> int:
    if seed_identity["kind"] == "development":
        return seed_identity["seed"]
    material = b"R12-ACW-OPT-v1\x00" + seed_identity["commitment"].encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def _validate_checkpoint_artifact(
    path: Path,
    file_sha256: str,
    *,
    logical_arm: str,
    dataset_summary: dict[str, Any],
    bundle_summary: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Load a real checkpoint and verify its model and artifact bindings."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise EvidenceError(
            "checkpoint_unreadable",
            f"cannot load {label} with weights_only=True: {type(exc).__name__}: {exc}",
        ) from exc
    checkpoint = _object(checkpoint, label)
    _expect_keys(checkpoint, CHECKPOINT_KEYS, label)
    if checkpoint["protocol"] != TRAINING_PROTOCOL:
        raise EvidenceError(
            "checkpoint_protocol_mismatch", f"{label} has wrong protocol"
        )
    checkpoint_arm, model_arm = _checkpoint_arms(logical_arm)
    if checkpoint["arm"] != checkpoint_arm:
        raise EvidenceError("checkpoint_arm_mismatch", f"{label} arm differs")
    if _integer(checkpoint["seed"], f"{label}.seed") != _expected_optimizer_seed(
        dataset_summary["seed_identity"]
    ):
        raise EvidenceError("optimizer_seed_mismatch", f"{label} seed differs")
    bindings = {
        "dataset_manifest_payload_sha256": bundle_summary["payload_sha256"],
        "source_manifest_payload_sha256": dataset_summary["payload_sha256"],
        "curriculum_sha256": bundle_summary["curriculum_sha256"],
        "query_schedule_sha256": bundle_summary["query_schedule_sha256"],
        "query_schedule_kind": bundle_summary["query_schedule_kind"],
        "pilot_report_payload_sha256": bundle_summary["pilot_report_payload_sha256"],
    }
    for field, expected in bindings.items():
        if checkpoint[field] != expected:
            raise EvidenceError(
                "checkpoint_bundle_binding_mismatch",
                f"{label}.{field} differs from the opened dataset or trainer bundle",
            )
    parameters = _integer(checkpoint["parameters"], f"{label}.parameters", minimum=1)
    if parameters != ARM_PARAMETERS[logical_arm]:
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} parameter count differs"
        )
    model_state = _object(checkpoint["model"], f"{label}.model")
    try:
        from pipeline.acw_hidden_basis_training import model_for_arm
        from pipeline.addressed_categorical_workspace import trainable_parameters

        model = model_for_arm(model_arm)
        model.load_state_dict(model_state, strict=True)
    except Exception as exc:
        raise EvidenceError(
            "checkpoint_model_mismatch",
            f"{label} model tensors do not load into {model_arm}: {type(exc).__name__}: {exc}",
        ) from exc
    if trainable_parameters(model) != parameters:
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} tensor parameter count differs"
        )
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise EvidenceError(
                "checkpoint_model_mismatch", f"{label} has a non-tensor state entry"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise EvidenceError(
                "checkpoint_model_mismatch", f"{label}.{name} is non-finite"
            )
    scientific_identity = _validate_scientific_identity(
        checkpoint["scientific_identity"], f"{label}.scientific_identity"
    )
    training_report = _object(checkpoint["training_report"], f"{label}.training_report")
    training_evidence = _validate_training_evidence(
        {
            "trainer_bundle_manifest_payload_sha256": checkpoint[
                "dataset_manifest_payload_sha256"
            ],
            "curriculum_sha256": checkpoint["curriculum_sha256"],
            "query_schedule_sha256": checkpoint["query_schedule_sha256"],
            "updates": training_report.get("updates"),
            "labels": training_report.get("labels"),
            "resource_ledger": training_report.get("resource_ledger"),
            "resource_measurements": training_report.get("resource_measurements"),
        },
        arm=logical_arm,
        label=f"{label}.training_evidence",
    )
    snapshots = checkpoint["label_efficiency_models"]
    if logical_arm == DIRECT_STATE_ARM:
        if snapshots is not None:
            raise EvidenceError(
                "checkpoint_label_snapshot_mismatch",
                f"{label} direct-state diagnostic must not carry scored snapshots",
            )
    elif not isinstance(snapshots, list) or len(snapshots) != len(LABEL_CHECKPOINTS):
        raise EvidenceError(
            "checkpoint_label_snapshot_mismatch",
            f"{label} must carry all {len(LABEL_CHECKPOINTS)} label snapshots",
        )
    return {
        "sha256": file_sha256,
        "checkpoint_arm": checkpoint_arm,
        "model_arm": model_arm,
        "parameters": parameters,
        "scientific_identity": scientific_identity,
        "training_evidence": training_evidence,
        **bindings,
    }


def _independent_evaluator_replay(
    checkpoint_path: Path,
    dataset_root: Path,
    expected_bytes: bytes,
    label: str,
) -> dict[str, Any]:
    """Execute the frozen evaluator in a new interpreter and compare raw bytes."""

    repository = Path(__file__).resolve().parents[1]
    evaluator = repository / "pipeline" / "evaluate_acw_hidden_basis.py"
    if not evaluator.is_file():
        raise EvidenceError(
            "independent_evaluator_missing", "frozen evaluator is absent"
        )
    with tempfile.TemporaryDirectory(prefix="acw-adjudicator-replay-") as temporary:
        output = Path(temporary) / "evaluation.json"
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.evaluate_acw_hidden_basis",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset",
                    str(dataset_root),
                    "--out",
                    str(output),
                ],
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=EVALUATOR_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvidenceError(
                "independent_evaluator_failed",
                f"{label} could not execute the evaluator: {type(exc).__name__}: {exc}",
            ) from exc
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "no diagnostic").strip()
            raise EvidenceError(
                "independent_evaluator_failed",
                f"{label} evaluator exit {completed.returncode}: {detail[-2000:]}",
            )
        replay_bytes, replay_sha256 = _read_regular_file(output)
        if replay_bytes != expected_bytes:
            raise EvidenceError(
                "independent_evaluator_mismatch",
                f"{label} fresh evaluator output is not byte-identical to the two submitted outputs",
            )
        replay = _parse_json(replay_bytes, f"{label}.independent_replay")
        return {
            "sha256": replay_sha256,
            "payload_sha256": _verify_payload_hash(
                replay, f"{label}.independent_replay"
            ),
            "byte_identical": True,
            "process_isolation": True,
        }


def _validate_native_resource_ledger(
    value: Any,
    *,
    arm: str,
    label: str,
) -> dict[str, Any]:
    ledger = _object(value, label)
    _expect_keys(ledger, NATIVE_RESOURCE_LEDGER_KEYS, label)
    (
        semantic_bits,
        persistent_bytes,
        dtype,
        training_bytes,
        transient_bytes,
        _,
        matched,
    ) = ARM_RESOURCES[arm]
    if (
        _integer(ledger["trainable_parameters"], f"{label}.trainable_parameters")
        != ARM_PARAMETERS[arm]
    ):
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} parameter count differs"
        )
    observed_bits = _number(
        ledger["semantic_state_bits"], f"{label}.semantic_state_bits", minimum=0.0
    )
    if not math.isclose(observed_bits, semantic_bits, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError("resource_ledger_mismatch", f"{label} semantic bits differ")
    exact = {
        "persistent_evaluation_bytes": persistent_bytes,
        "persistent_training_state_bytes": training_bytes,
        "declared_transient_token_bytes": transient_bytes,
    }
    for field, expected in exact.items():
        if _integer(ledger[field], f"{label}.{field}", minimum=0) != expected:
            raise EvidenceError("resource_ledger_mismatch", f"{label}.{field} differs")
    if ledger["persistent_evaluation_dtype"] != dtype:
        raise EvidenceError("resource_ledger_mismatch", f"{label} state dtype differs")
    if (
        _boolean(
            ledger["parameter_matched_primary"], f"{label}.parameter_matched_primary"
        )
        != matched
    ):
        raise EvidenceError("resource_ledger_mismatch", f"{label} match flag differs")
    return ledger


def _validate_resource_profile(
    value: Any,
    *,
    label: str,
    expected_scope: str,
    direct_training: bool = False,
    training: bool = False,
) -> dict[str, Any]:
    profile = _object(value, label)
    extras = (
        DIRECT_TRAINING_PROFILE_EXTRA_KEYS
        if direct_training
        else TRAINING_PROFILE_EXTRA_KEYS
        if training
        else set()
    )
    _expect_keys(profile, PROFILE_MEASUREMENT_KEYS | extras, label)
    if profile["scope"] != expected_scope:
        raise EvidenceError("incomplete_resource_ledger", f"{label} scope differs")
    if _integer(profile["batch_size"], f"{label}.batch_size") != 256:
        raise EvidenceError("resource_ledger_mismatch", f"{label} batch size differs")
    _integer(profile["active_events"], f"{label}.active_events", minimum=1)
    wall_seconds = _number(
        profile["wall_seconds"], f"{label}.wall_seconds", minimum=0.0
    )
    if wall_seconds <= 0.0:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} wall time is not positive"
        )
    _integer(
        profile["process_peak_rss_bytes"], f"{label}.process_peak_rss_bytes", minimum=1
    )
    event_count = _integer(
        profile["profiler_event_count"], f"{label}.profiler_event_count", minimum=1
    )
    if not _boolean(
        profile["operator_inventory_complete"],
        f"{label}.operator_inventory_complete",
    ):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} operator inventory is incomplete"
        )
    inventory = _list(profile["operator_inventory"], f"{label}.operator_inventory")
    if not inventory:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} operator inventory is empty"
        )
    normalized_inventory = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(inventory):
        entry_label = f"{label}.operator_inventory[{index}]"
        entry = _object(raw_entry, entry_label)
        _expect_keys(entry, OPERATOR_INVENTORY_ENTRY_KEYS, entry_label)
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise EvidenceError("schema_mismatch", f"{entry_label}.name is invalid")
        if name in seen_names:
            raise EvidenceError(
                "incomplete_resource_ledger", f"{label} repeats operator {name!r}"
            )
        seen_names.add(name)
        normalized_inventory.append(
            {
                "name": name,
                "calls": _integer(entry["calls"], f"{entry_label}.calls", minimum=1),
                "operator_reported_flops": _integer(
                    entry["operator_reported_flops"],
                    f"{entry_label}.operator_reported_flops",
                    minimum=0,
                ),
                "positive_allocation_bytes": _integer(
                    entry["positive_allocation_bytes"],
                    f"{entry_label}.positive_allocation_bytes",
                    minimum=0,
                ),
                "positive_self_allocation_bytes": _integer(
                    entry["positive_self_allocation_bytes"],
                    f"{entry_label}.positive_self_allocation_bytes",
                    minimum=0,
                ),
            }
        )
    if [entry["name"] for entry in normalized_inventory] != sorted(seen_names):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} operator inventory is not sorted"
        )
    if sum(entry["calls"] for entry in normalized_inventory) != event_count:
        raise EvidenceError(
            "incomplete_resource_ledger",
            f"{label} profiler event count is inconsistent",
        )
    uncounted = profile["uncounted_operator_names"]
    if not isinstance(uncounted, list) or not all(
        isinstance(name, str) and name for name in uncounted
    ):
        raise EvidenceError(
            "schema_mismatch", f"{label}.uncounted_operator_names is invalid"
        )
    expected_uncounted = [
        entry["name"]
        for entry in normalized_inventory
        if entry["operator_reported_flops"] == 0
    ]
    if uncounted != expected_uncounted:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} uncounted inventory is inconsistent"
        )
    for field in (
        "operator_reported_flops",
        "largest_operator_allocation_bytes",
        "largest_self_operator_allocation_bytes",
        "total_positive_operator_allocations_bytes",
    ):
        _integer(profile[field], f"{label}.{field}", minimum=0)
    if (
        sum(entry["operator_reported_flops"] for entry in normalized_inventory)
        != profile["operator_reported_flops"]
    ):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} reported FLOPs do not reconcile"
        )
    if (
        sum(entry["positive_allocation_bytes"] for entry in normalized_inventory)
        != profile["total_positive_operator_allocations_bytes"]
    ):
        raise EvidenceError(
            "incomplete_resource_ledger",
            f"{label} allocation inventory does not reconcile",
        )
    if profile["flop_counting_contract"] != FLOP_COUNTING_CONTRACT:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} FLOP contract differs"
        )
    if profile["transient_memory_contract"] != TRANSIENT_MEMORY_CONTRACT:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} memory contract differs"
        )
    if training and not _boolean(
        profile["optimizer_included"], f"{label}.optimizer_included"
    ):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} omits optimizer resources"
        )
    if direct_training and not math.isclose(
        _number(
            profile["state_auxiliary_weight"],
            f"{label}.state_auxiliary_weight",
            minimum=0.0,
        ),
        4.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise EvidenceError(
            "direct_state_supervision_missing", f"{label} weight differs"
        )
    return profile


def _validate_training_evidence(
    value: Any,
    *,
    arm: str,
    label: str,
) -> dict[str, Any]:
    evidence = _object(value, label)
    _expect_keys(evidence, TRAINING_EVIDENCE_KEYS, label)
    trainer_bundle_manifest_payload_sha256 = _hash(
        evidence["trainer_bundle_manifest_payload_sha256"],
        f"{label}.trainer_bundle_manifest_payload_sha256",
    )
    curriculum_sha256 = _hash(
        evidence["curriculum_sha256"], f"{label}.curriculum_sha256"
    )
    query_schedule_sha256 = _hash(
        evidence["query_schedule_sha256"], f"{label}.query_schedule_sha256"
    )
    if (
        len(
            {
                trainer_bundle_manifest_payload_sha256,
                curriculum_sha256,
                query_schedule_sha256,
            }
        )
        != 3
    ):
        raise EvidenceError(
            "training_artifact_hash_reused",
            f"{label} reuses one hash for distinct frozen artifacts",
        )
    if _integer(evidence["updates"], f"{label}.updates") != OPTIMIZER_UPDATES:
        raise EvidenceError("optimizer_update_mismatch", f"{label} updates differ")
    if _integer(evidence["labels"], f"{label}.labels") != FINAL_SCALAR_LABELS:
        raise EvidenceError("label_count_mismatch", f"{label} label count differs")
    ledger = _validate_native_resource_ledger(
        evidence["resource_ledger"], arm=arm, label=f"{label}.resource_ledger"
    )
    measurements = _object(
        evidence["resource_measurements"], f"{label}.resource_measurements"
    )
    _expect_keys(
        measurements, RESOURCE_MEASUREMENTS_KEYS, f"{label}.resource_measurements"
    )
    direct = arm == DIRECT_STATE_ARM
    training_profile = _validate_resource_profile(
        measurements["training"],
        label=f"{label}.resource_measurements.training",
        expected_scope=DIRECT_TRAINING_PROFILE_SCOPE
        if direct
        else TRAINING_PROFILE_SCOPE,
        direct_training=direct,
        training=True,
    )
    inference_profile = _validate_resource_profile(
        measurements["inference"],
        label=f"{label}.resource_measurements.inference",
        expected_scope=INFERENCE_PROFILE_SCOPE,
    )
    return {
        "trainer_bundle_manifest_payload_sha256": (
            trainer_bundle_manifest_payload_sha256
        ),
        "curriculum_sha256": curriculum_sha256,
        "query_schedule_sha256": query_schedule_sha256,
        "updates": OPTIMIZER_UPDATES,
        "labels": FINAL_SCALAR_LABELS,
        "resource_ledger": ledger,
        "resource_measurements": {
            "training": training_profile,
            "inference": inference_profile,
        },
    }


def _validate_native_label_efficiency(value: Any, label: str) -> list[dict[str, Any]]:
    records = _list(value, label)
    if len(records) != len(LABEL_CHECKPOINTS):
        raise EvidenceError(
            "label_efficiency_checkpoint_mismatch",
            f"{label} must contain exactly {len(LABEL_CHECKPOINTS)} records",
        )
    normalized = []
    model_hashes: set[str] = set()
    for round_index, (raw, labels) in enumerate(zip(records, LABEL_CHECKPOINTS)):
        record_label = f"{label}[{round_index}]"
        record = _object(raw, record_label)
        _expect_keys(record, LABEL_EFFICIENCY_KEYS, record_label)
        if _integer(record["round"], f"{record_label}.round") != round_index:
            raise EvidenceError(
                "label_efficiency_checkpoint_mismatch", f"{record_label} round differs"
            )
        if _integer(record["labels"], f"{record_label}.labels") != labels:
            raise EvidenceError(
                "label_efficiency_checkpoint_mismatch", f"{record_label} labels differ"
            )
        expected_updates = (
            OPTIMIZER_UPDATES
            if round_index == len(LABEL_CHECKPOINTS) - 1
            else 200 * (round_index + 1)
        )
        if (
            _integer(record["optimizer_updates"], f"{record_label}.optimizer_updates")
            != expected_updates
        ):
            raise EvidenceError(
                "label_efficiency_checkpoint_mismatch", f"{record_label} updates differ"
            )
        model_hash = _hash(
            record["model_tensor_sha256"], f"{record_label}.model_tensor_sha256"
        )
        if model_hash in model_hashes:
            raise EvidenceError(
                "duplicate_label_checkpoint", f"{label} repeats model hash"
            )
        model_hashes.add(model_hash)
        metric = _validate_accuracy(
            record["depth_64"],
            f"{record_label}.depth_64",
            histories=PUBLIC_HISTORIES,
            queries=PUBLIC_QUERIES,
        )
        normalized.append(
            {
                "round": round_index,
                "labels": labels,
                "optimizer_updates": expected_updates,
                "model_tensor_sha256": model_hash,
                "depth_64_scalar_accuracy": metric["scalar_accuracy"],
                "depth_64_state_exactness": metric["state_exactness"],
            }
        )
    return normalized


def _validate_compiled_sparse_control(value: Any, label: str) -> dict[str, Any]:
    control = _object(value, label)
    _expect_keys(control, COMPILED_CONTROL_KEYS, label)
    depths = _object(control["depths"], f"{label}.depths")
    if set(depths) != {str(depth) for depth in EVALUATION_DEPTHS}:
        raise EvidenceError("evaluation_depth_set_mismatch", f"{label} depths differ")
    normalized_depths = {}
    for depth in EVALUATION_DEPTHS:
        depth_label = f"{label}.depths.{depth}"
        metric = _object(depths[str(depth)], depth_label)
        _expect_keys(metric, ACCURACY_KEYS | COMPILED_DEPTH_EXTRA_KEYS, depth_label)
        accuracy = _validate_accuracy(
            {key: metric[key] for key in ACCURACY_KEYS},
            depth_label,
            histories=PUBLIC_HISTORIES,
            queries=PUBLIC_QUERIES,
        )
        transition_total = _integer(
            metric["transition_state_total"], f"{depth_label}.transition_state_total"
        )
        transition_exact = _integer(
            metric["transition_state_exact"], f"{depth_label}.transition_state_exact"
        )
        transition_rate = _number(
            metric["transition_state_exactness"],
            f"{depth_label}.transition_state_exactness",
            minimum=0.0,
            maximum=1.0,
        )
        if transition_total != PUBLIC_HISTORIES or transition_exact != transition_total:
            raise EvidenceError(
                "compiled_sparse_control_failed",
                f"{depth_label} transition replay failed",
            )
        if not math.isclose(transition_rate, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise EvidenceError(
                "compiled_sparse_control_failed",
                f"{depth_label} transition rate failed",
            )
        if accuracy["scalar_accuracy"] != 1.0 or accuracy["state_exactness"] != 1.0:
            raise EvidenceError(
                "compiled_sparse_control_failed", f"{depth_label} answers failed"
            )
        normalized_depths[str(depth)] = {
            **_metric_summary(accuracy),
            "transition_state_exactness": transition_rate,
        }

    event_updates = _integer(
        control["external_event_updates"], f"{label}.external_event_updates"
    )
    expected_event_updates = PUBLIC_HISTORIES * sum(EVALUATION_DEPTHS)
    if event_updates != expected_event_updates:
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} event count differs"
        )
    query_reads = _integer(
        control["external_query_reads"], f"{label}.external_query_reads"
    )
    expected_query_reads = PUBLIC_HISTORIES * PUBLIC_QUERIES * len(EVALUATION_DEPTHS)
    if query_reads != expected_query_reads:
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} query count differs"
        )
    event_arithmetic = _object(control["event_arithmetic"], f"{label}.event_arithmetic")
    _expect_keys(
        event_arithmetic,
        {"multiplications", "additions", "modulo"},
        f"{label}.event_arithmetic",
    )
    if event_arithmetic != {
        "multiplications": 2 * event_updates,
        "additions": 2 * event_updates,
        "modulo": event_updates,
    }:
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} event arithmetic differs"
        )
    query_arithmetic = _object(control["query_arithmetic"], f"{label}.query_arithmetic")
    _expect_keys(
        query_arithmetic,
        {"multiplications", "additions", "modulo", "permutation_lookups"},
        f"{label}.query_arithmetic",
    )
    if query_arithmetic != {
        "multiplications": 3 * query_reads,
        "additions": 3 * query_reads,
        "modulo": query_reads,
        "permutation_lookups": query_reads,
    }:
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} query arithmetic differs"
        )
    resources = _object(control["resource_ledger"], f"{label}.resource_ledger")
    _expect_keys(resources, COMPILED_RESOURCE_KEYS, f"{label}.resource_ledger")
    if (
        _integer(
            resources["trainable_parameters"],
            f"{label}.resource_ledger.trainable_parameters",
        )
        != 0
    ):
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} has trainable parameters"
        )
    if (
        _integer(
            resources["persistent_state_bytes"],
            f"{label}.resource_ledger.persistent_state_bytes",
        )
        != 3
    ):
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} state bytes differ"
        )
    for field in ("event_table_bytes", "query_table_bytes"):
        _integer(resources[field], f"{label}.resource_ledger.{field}", minimum=1)
    if resources["runtime"] != "NumPy/Python exact F_17 replay":
        raise EvidenceError(
            "compiled_resource_ledger_mismatch", f"{label} runtime differs"
        )
    if (
        control["claim_boundary"]
        != "Known exact compilation; not neural learnability evidence."
    ):
        raise EvidenceError(
            "evaluation_claim_boundary_mismatch", f"{label} claim differs"
        )
    return {"depths": normalized_depths, "resource_ledger": resources}


def _validate_evaluation_report(
    report: dict[str, Any],
    *,
    logical_arm: str,
    dataset_payload_sha256: str,
    dataset_seed_identity: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    checkpoint_arm, model_arm = _checkpoint_arms(logical_arm)
    expected_keys = EVALUATION_BASE_KEYS | (
        EVALUATION_ACW_KEYS if model_arm == "acw" else set()
    )
    if logical_arm != DIRECT_STATE_ARM:
        expected_keys = expected_keys | {"label_efficiency"}
    _expect_keys(report, expected_keys, label)
    payload_hash = _verify_payload_hash(report, label)
    if report["protocol"] != EVALUATION_PROTOCOL:
        raise EvidenceError(
            "evaluation_protocol_mismatch", f"{label} has wrong protocol"
        )
    checkpoint_sha256 = _hash(report["checkpoint_sha256"], f"{label}.checkpoint_sha256")
    if report["checkpoint_arm"] != checkpoint_arm or report["model_arm"] != model_arm:
        raise EvidenceError("evaluation_arm_mismatch", f"{label} arm identity differs")
    parameters = _integer(report["parameters"], f"{label}.parameters", minimum=1)
    if parameters != ARM_PARAMETERS[logical_arm]:
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} parameter count differs"
        )
    if report["dataset_manifest_payload_sha256"] != dataset_payload_sha256:
        raise EvidenceError(
            "evaluation_dataset_mismatch", f"{label} is bound to another dataset"
        )
    if report["seed_identity"] != dataset_seed_identity:
        raise EvidenceError(
            "evaluation_seed_identity_mismatch",
            f"{label} seed identity differs from its generator manifest",
        )
    optimizer_seed = _integer(report["optimizer_seed"], f"{label}.optimizer_seed")
    if optimizer_seed != _expected_optimizer_seed(dataset_seed_identity):
        raise EvidenceError(
            "optimizer_seed_mismatch", f"{label} optimizer seed is not registered"
        )
    expected_schedule = (
        "uniform_schedule.jsonl"
        if logical_arm == "uniform_query_acw"
        else "cgb_schedule.jsonl"
    )
    if report["query_schedule_kind"] != expected_schedule:
        raise EvidenceError(
            "query_schedule_kind_mismatch", f"{label} schedule kind differs"
        )
    pilot_report_payload_sha256 = _hash(
        report["pilot_report_payload_sha256"],
        f"{label}.pilot_report_payload_sha256",
    )
    training_evidence = _validate_training_evidence(
        report["training_evidence"], arm=logical_arm, label=f"{label}.training_evidence"
    )
    label_efficiency = (
        []
        if logical_arm == DIRECT_STATE_ARM
        else _validate_native_label_efficiency(
            report["label_efficiency"], f"{label}.label_efficiency"
        )
    )
    compiled_control = _validate_compiled_sparse_control(
        report["compiled_sparse_control"], f"{label}.compiled_sparse_control"
    )
    scientific_identity = _validate_scientific_identity(
        report["scientific_identity"], f"{label}.scientific_identity"
    )
    if report["claim_boundary"] != EVALUATOR_CLAIM_BOUNDARY:
        raise EvidenceError(
            "evaluation_claim_boundary_mismatch", f"{label} claim changed"
        )

    public_raw = _object(report["public_depths"], f"{label}.public_depths")
    if set(public_raw) != {str(depth) for depth in EVALUATION_DEPTHS}:
        raise EvidenceError(
            "evaluation_depth_set_mismatch", f"{label} public depths differ"
        )
    public = {
        depth: _validate_accuracy(
            public_raw[str(depth)],
            f"{label}.public_depths.{depth}",
            histories=PUBLIC_HISTORIES,
            queries=PUBLIC_QUERIES,
        )
        for depth in EVALUATION_DEPTHS
    }
    if label_efficiency:
        final_efficiency = label_efficiency[-1]
        final_primary = public[64]
        if not math.isclose(
            final_efficiency["depth_64_scalar_accuracy"],
            final_primary["scalar_accuracy"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            final_efficiency["depth_64_state_exactness"],
            final_primary["state_exactness"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise EvidenceError(
                "label_efficiency_final_metric_mismatch",
                f"{label} final 57,344-label metric differs from the primary report",
            )

    reader = _object(report["new_reader"], f"{label}.new_reader")
    _expect_keys(reader, NEW_READER_KEYS, f"{label}.new_reader")
    if _integer(reader["updates"], f"{label}.new_reader.updates") != 500:
        raise EvidenceError(
            "new_reader_schema_mismatch", f"{label} new reader updates differ"
        )
    _integer(reader["state_dim"], f"{label}.new_reader.state_dim", minimum=1)
    _integer(
        reader["reader_parameters"], f"{label}.new_reader.reader_parameters", minimum=1
    )
    _number(reader["loss_first"], f"{label}.new_reader.loss_first", minimum=0.0)
    _number(reader["loss_last"], f"{label}.new_reader.loss_last", minimum=0.0)
    reader_depths_raw = _object(reader["depths"], f"{label}.new_reader.depths")
    if set(reader_depths_raw) != {str(depth) for depth in EVALUATION_DEPTHS}:
        raise EvidenceError(
            "evaluation_depth_set_mismatch", f"{label} new-reader depths differ"
        )
    reader_depths = {
        depth: _validate_accuracy(
            reader_depths_raw[str(depth)],
            f"{label}.new_reader.depths.{depth}",
            histories=PUBLIC_HISTORIES,
            queries=NEW_READER_QUERIES,
        )
        for depth in EVALUATION_DEPTHS
    }

    normalized: dict[str, Any] = {
        "payload_sha256": payload_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_payload_sha256": dataset_payload_sha256,
        "seed_identity": dataset_seed_identity,
        "optimizer_seed": optimizer_seed,
        "query_schedule_kind": expected_schedule,
        "pilot_report_payload_sha256": pilot_report_payload_sha256,
        "training_evidence": training_evidence,
        "label_efficiency": label_efficiency,
        "compiled_sparse_control": compiled_control,
        "scientific_identity": scientific_identity,
        "public_depths": public,
        "new_reader_depths": reader_depths,
    }
    if model_arm == "acw":
        interventions = _object(
            report["packet_interventions"], f"{label}.packet_interventions"
        )
        _expect_keys(interventions, INTERVENTION_KEYS, f"{label}.packet_interventions")
        donor = _validate_accuracy(
            interventions["donor_following"],
            f"{label}.packet_interventions.donor_following",
            histories=PUBLIC_HISTORIES,
            queries=PUBLIC_QUERIES,
        )
        shuffled = _validate_accuracy(
            interventions["shuffled_against_original"],
            f"{label}.packet_interventions.shuffled_against_original",
            histories=PUBLIC_HISTORIES,
            queries=PUBLIC_QUERIES,
        )
        source_identical = _boolean(
            interventions["held_packet_source_swap_predictions_identical"],
            f"{label}.packet_interventions.held_packet_source_swap_predictions_identical",
        )
        if interventions["source_swap_basis"] != SOURCE_SWAP_BASIS:
            raise EvidenceError(
                "source_swap_schema_mismatch", f"{label} source-swap basis differs"
            )
        donor_difference = _number(
            interventions["donor_different_truth_fraction"],
            f"{label}.packet_interventions.donor_different_truth_fraction",
            minimum=0.0,
            maximum=1.0,
        )

        write_legality = _object(report["write_legality"], f"{label}.write_legality")
        _expect_keys(write_legality, WRITE_LEGALITY_KEYS, f"{label}.write_legality")
        checked = _integer(
            write_legality["unaddressed_registers_checked"],
            f"{label}.write_legality.unaddressed_registers_checked",
            minimum=1,
        )
        illegal = _integer(
            write_legality["illegal_writes"],
            f"{label}.write_legality.illegal_writes",
            minimum=0,
        )
        if illegal > checked:
            raise EvidenceError(
                "write_ledger_mismatch", f"{label} illegal writes exceed checks"
            )

        words = _object(report["event_words"], f"{label}.event_words")
        _expect_keys(words, EVENT_WORD_KEYS, f"{label}.event_words")
        histories = _integer(words["histories"], f"{label}.event_words.histories")
        if histories != EVENT_WORD_HISTORIES:
            raise EvidenceError(
                "event_word_schema_mismatch", f"{label} event-word count differs"
            )
        event_word_rates = {
            name: _number(
                words[name], f"{label}.event_words.{name}", minimum=0.0, maximum=1.0
            )
            for name in (
                "equivalent_prediction_query_equivalence",
                "non_equivalent_target_separator_rate",
                "non_equivalent_prediction_separator_rate",
            )
        }
        event_word_accuracy = {
            name: _validate_accuracy(
                words[name],
                f"{label}.event_words.{name}",
                histories=EVENT_WORD_HISTORIES,
                queries=PUBLIC_QUERIES,
            )
            for name in (
                "equivalent_a",
                "equivalent_b",
                "non_equivalent_a",
                "non_equivalent_b",
            )
        }
        normalized.update(
            {
                "packet_interventions": {
                    "donor_following": donor,
                    "shuffled_against_original": shuffled,
                    "held_packet_source_swap_predictions_identical": source_identical,
                    "donor_different_truth_fraction": donor_difference,
                },
                "write_legality": {
                    "unaddressed_registers_checked": checked,
                    "illegal_writes": illegal,
                },
                "event_words": {**event_word_rates, **event_word_accuracy},
            }
        )
    return normalized


def _validate_training_ledger(
    value: Any,
    *,
    arm: str,
    checkpoint_sha256: str,
    dataset_payload_sha256: str,
    label: str,
) -> dict[str, Any]:
    ledger = _object(value, label)
    _expect_keys(ledger, TRAIN_LEDGER_KEYS, label)
    if ledger["protocol"] != TRAIN_LEDGER_PROTOCOL:
        raise EvidenceError(
            "training_ledger_protocol_mismatch", f"{label} protocol differs"
        )
    if ledger["checkpoint_sha256"] != checkpoint_sha256:
        raise EvidenceError(
            "training_checkpoint_mismatch", f"{label} checkpoint differs"
        )
    if ledger["dataset_manifest_payload_sha256"] != dataset_payload_sha256:
        raise EvidenceError("training_dataset_mismatch", f"{label} dataset differs")
    _hash(ledger["curriculum_sha256"], f"{label}.curriculum_sha256")
    labels = _integer(ledger["scalar_labels"], f"{label}.scalar_labels", minimum=0)
    if labels != FINAL_SCALAR_LABELS:
        raise EvidenceError(
            "label_count_mismatch", f"{label} must report {FINAL_SCALAR_LABELS} labels"
        )
    auxiliary = _integer(
        ledger["state_auxiliary_labels"], f"{label}.state_auxiliary_labels", minimum=0
    )
    oracle_access = _boolean(ledger["oracle_access"], f"{label}.oracle_access")
    if arm == DIRECT_STATE_ARM:
        if auxiliary <= 0 or not oracle_access:
            raise EvidenceError(
                "direct_state_supervision_missing",
                f"{label} lacks diagnostic supervision",
            )
    elif auxiliary != 0 or oracle_access:
        raise EvidenceError(
            "scored_arm_oracle_access", f"{label} exposes hidden supervision"
        )
    if _boolean(
        ledger["confirmation_preimage_access"], f"{label}.confirmation_preimage_access"
    ):
        raise EvidenceError(
            "confirmation_preimage_leak", f"{label} reports seed preimage access"
        )
    if (
        _integer(ledger["optimizer_updates"], f"{label}.optimizer_updates")
        != OPTIMIZER_UPDATES
    ):
        raise EvidenceError("optimizer_update_mismatch", f"{label} updates differ")
    _integer(
        ledger["optimizer_evaluations"], f"{label}.optimizer_evaluations", minimum=1
    )
    candidates = _integer(
        ledger["oracle_candidate_evaluations"],
        f"{label}.oracle_candidate_evaluations",
        minimum=0,
    )
    selections = _integer(
        ledger["witness_selections"], f"{label}.witness_selections", minimum=0
    )
    if (
        candidates > MAX_ORACLE_CANDIDATE_EVALUATIONS
        or selections > MAX_WITNESS_SELECTIONS
    ):
        raise EvidenceError(
            "oracle_resource_cap_exceeded", f"{label} exceeds frozen oracle caps"
        )

    (
        semantic_bits,
        _,
        _,
        training_bytes,
        transient_bytes,
        extra_source,
        parameter_matched,
    ) = ARM_RESOURCES[arm]
    if (
        _integer(ledger["trainable_parameters"], f"{label}.trainable_parameters")
        != ARM_PARAMETERS[arm]
    ):
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} parameter count differs"
        )
    observed_bits = _number(
        ledger["semantic_state_bits"], f"{label}.semantic_state_bits", minimum=0.0
    )
    if not math.isclose(observed_bits, semantic_bits, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError("resource_ledger_mismatch", f"{label} semantic bits differ")
    exact_resources = {
        "persistent_training_state_bytes": training_bytes,
        "declared_transient_token_bytes": transient_bytes,
        "extra_source_bytes": extra_source,
    }
    for field, expected in exact_resources.items():
        if _integer(ledger[field], f"{label}.{field}", minimum=0) != expected:
            raise EvidenceError("resource_ledger_mismatch", f"{label}.{field} differs")
    if (
        _boolean(
            ledger["parameter_matched_primary"], f"{label}.parameter_matched_primary"
        )
        != parameter_matched
    ):
        raise EvidenceError(
            "resource_ledger_mismatch", f"{label} parameter-match flag differs"
        )
    if _boolean(ledger["mixed_precision"], f"{label}.mixed_precision"):
        raise EvidenceError(
            "mixed_precision_forbidden", f"{label} must be float32 training"
        )
    _integer(ledger["train_flops"], f"{label}.train_flops", minimum=1)
    _number(ledger["train_wall_seconds"], f"{label}.train_wall_seconds", minimum=0.0)
    if not _boolean(
        ledger["flop_measurement_complete"], f"{label}.flop_measurement_complete"
    ):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} FLOP measurement is incomplete"
        )
    return ledger


def _validate_inference_ledger(
    value: Any,
    *,
    arm: str,
    checkpoint_sha256: str,
    dataset_payload_sha256: str,
    label: str,
) -> dict[str, Any]:
    ledger = _object(value, label)
    _expect_keys(ledger, INFERENCE_LEDGER_KEYS, label)
    if ledger["protocol"] != INFERENCE_LEDGER_PROTOCOL:
        raise EvidenceError(
            "inference_ledger_protocol_mismatch", f"{label} protocol differs"
        )
    if ledger["checkpoint_sha256"] != checkpoint_sha256:
        raise EvidenceError(
            "inference_checkpoint_mismatch", f"{label} checkpoint differs"
        )
    if ledger["dataset_manifest_payload_sha256"] != dataset_payload_sha256:
        raise EvidenceError("inference_dataset_mismatch", f"{label} dataset differs")
    if ledger["scope"] != INFERENCE_LEDGER_SCOPE:
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} scope is incomplete"
        )
    if (
        _integer(ledger["trainable_parameters"], f"{label}.trainable_parameters")
        != ARM_PARAMETERS[arm]
    ):
        raise EvidenceError(
            "parameter_count_mismatch", f"{label} parameter count differs"
        )
    semantic_bits, persistent_bytes, dtype, _, transient_bytes, extra_source, _ = (
        ARM_RESOURCES[arm]
    )
    observed_bits = _number(
        ledger["semantic_state_bits"], f"{label}.semantic_state_bits", minimum=0.0
    )
    if not math.isclose(observed_bits, semantic_bits, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError("resource_ledger_mismatch", f"{label} semantic bits differ")
    exact_resources = {
        "persistent_state_bytes": persistent_bytes,
        "declared_transient_token_bytes": transient_bytes,
        "extra_source_bytes": extra_source,
        "kv_cache_bytes": 0,
    }
    for field, expected in exact_resources.items():
        if _integer(ledger[field], f"{label}.{field}", minimum=0) != expected:
            raise EvidenceError("resource_ledger_mismatch", f"{label}.{field} differs")
    if ledger["persistent_state_dtype"] != dtype:
        raise EvidenceError("resource_ledger_mismatch", f"{label} state dtype differs")
    peak = _integer(
        ledger["peak_transient_bytes"], f"{label}.peak_transient_bytes", minimum=0
    )
    if peak < transient_bytes:
        raise EvidenceError(
            "resource_ledger_mismatch", f"{label} peak transient bytes are too low"
        )
    if _boolean(ledger["mixed_precision"], f"{label}.mixed_precision"):
        raise EvidenceError(
            "mixed_precision_forbidden", f"{label} mixed precision is forbidden"
        )
    for field in ("event_updates", "query_reads", "inference_flops"):
        _integer(ledger[field], f"{label}.{field}", minimum=1)
    _number(
        ledger["inference_wall_seconds"], f"{label}.inference_wall_seconds", minimum=0.0
    )
    if not _boolean(
        ledger["flop_measurement_complete"], f"{label}.flop_measurement_complete"
    ):
        raise EvidenceError(
            "incomplete_resource_ledger", f"{label} FLOP measurement is incomplete"
        )
    return ledger


def _validate_label_efficiency(
    value: Any,
    *,
    arm: str,
    final_checkpoint_sha256: str,
    final_metric: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    records = _list(value, label)
    if arm == DIRECT_STATE_ARM:
        if records:
            raise EvidenceError(
                "diagnostic_label_efficiency_forbidden", f"{label} must be empty"
            )
        return []
    if len(records) != len(LABEL_CHECKPOINTS):
        raise EvidenceError(
            "label_efficiency_checkpoint_mismatch",
            f"{label} must contain exactly {len(LABEL_CHECKPOINTS)} records",
        )
    normalized = []
    seen_hashes: set[str] = set()
    for position, (raw, expected_labels) in enumerate(zip(records, LABEL_CHECKPOINTS)):
        record_label = f"{label}[{position}]"
        record = _object(raw, record_label)
        _expect_keys(record, LABEL_EFFICIENCY_KEYS, record_label)
        if _integer(record["labels"], f"{record_label}.labels") != expected_labels:
            raise EvidenceError(
                "label_efficiency_checkpoint_mismatch",
                f"{record_label} label count differs",
            )
        checkpoint = _hash(
            record["checkpoint_sha256"], f"{record_label}.checkpoint_sha256"
        )
        if checkpoint in seen_hashes:
            raise EvidenceError(
                "duplicate_label_checkpoint", f"{label} repeats checkpoint hash"
            )
        seen_hashes.add(checkpoint)
        scalar = _number(
            record["depth_64_scalar_accuracy"],
            f"{record_label}.depth_64_scalar_accuracy",
            minimum=0.0,
            maximum=1.0,
        )
        state = _number(
            record["depth_64_state_exactness"],
            f"{record_label}.depth_64_state_exactness",
            minimum=0.0,
            maximum=1.0,
        )
        normalized.append(
            {
                "labels": expected_labels,
                "checkpoint_sha256": checkpoint,
                "depth_64_scalar_accuracy": scalar,
                "depth_64_state_exactness": state,
            }
        )
    final = normalized[-1]
    if final["checkpoint_sha256"] != final_checkpoint_sha256:
        raise EvidenceError(
            "label_efficiency_final_binding_mismatch",
            f"{label} final checkpoint differs",
        )
    if not math.isclose(
        final["depth_64_scalar_accuracy"], final_metric["scalar_accuracy"], abs_tol=1e-6
    ) or not math.isclose(
        final["depth_64_state_exactness"], final_metric["state_exactness"], abs_tol=1e-6
    ):
        raise EvidenceError(
            "label_efficiency_final_metric_mismatch", f"{label} final metric differs"
        )
    return normalized


def _expected_run_keys() -> set[tuple[str, str, int]]:
    expected = {
        (arm, split, index)
        for arm in SCORED_ARMS
        for split in ("development", "confirmation")
        for index in range(3)
    }
    expected.update((DIRECT_STATE_ARM, "development", index) for index in range(3))
    return expected


def verify_evidence(
    manifest: dict[str, Any], base: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _expect_keys(
        manifest, {"schema", "protocol", "reports", "payload_sha256"}, "manifest"
    )
    manifest_payload_sha256 = _verify_payload_hash(manifest, "manifest")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise EvidenceError(
            "manifest_schema_mismatch", f"manifest schema must be {MANIFEST_SCHEMA}"
        )
    if manifest["protocol"] != MANIFEST_PROTOCOL:
        raise EvidenceError(
            "manifest_protocol_mismatch",
            f"manifest protocol must be {MANIFEST_PROTOCOL}",
        )
    reports = _list(manifest["reports"], "manifest.reports")
    expected_keys = _expected_run_keys()
    if len(reports) != len(expected_keys):
        raise EvidenceError(
            "report_count_mismatch",
            f"manifest must contain exactly {len(expected_keys)} reports, got {len(reports)}",
        )

    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    dataset_by_identity: dict[tuple[str, int], tuple[str, str]] = {}
    identity_by_dataset: dict[str, tuple[str, int]] = {}
    dataset_cache: dict[tuple[Path, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    bundle_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    primary_report_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    checkpoint_paths: set[Path] = set()
    report_paths: set[Path] = set()
    scientific_identity: dict[str, Any] | None = None
    pilot_report_payload_sha256: str | None = None
    query_schedule_hashes = {
        "cgb_schedule.jsonl": set(),
        "uniform_schedule.jsonl": set(),
    }
    trainer_bundle_by_group: dict[tuple[str, int, str], str] = {}
    trainer_bundle_root_by_group: dict[tuple[str, int, str], str] = {}
    curriculum_by_group: dict[tuple[str, int, str], str] = {}
    group_by_trainer_bundle: dict[str, tuple[str, int, str]] = {}
    group_by_curriculum: dict[str, tuple[str, int, str]] = {}
    artifact_bindings = []

    for position, raw_run in enumerate(reports):
        label = f"manifest.reports[{position}]"
        run = _object(raw_run, label)
        _expect_keys(run, RUN_KEYS, label)
        arm = run["arm"]
        if not isinstance(arm, str) or arm not in (*SCORED_ARMS, DIRECT_STATE_ARM):
            raise EvidenceError("unknown_arm", f"{label}.arm is not frozen")

        dataset, dataset_binding, dataset_root = _verify_rooted_json_reference(
            run["dataset"], f"{label}.dataset", base
        )
        identity_key, dataset_summary = _validate_dataset_manifest(
            dataset, f"{label}.dataset.manifest"
        )
        dataset_cache_key = (
            dataset_root,
            dataset_summary["payload_sha256"],
        )
        cached_dataset = dataset_cache.get(dataset_cache_key)
        if cached_dataset is None:
            dataset_tree = _validate_dataset_tree(
                dataset_root, dataset, f"{label}.dataset"
            )
            dataset_cache[dataset_cache_key] = (dataset_summary, dataset_tree)
        else:
            cached_summary, dataset_tree = cached_dataset
            if cached_summary != dataset_summary:
                raise EvidenceError(
                    "dataset_cache_binding_mismatch",
                    f"{label} changes a previously opened dataset binding",
                )
        split, index = identity_key
        key = (arm, split, index)
        if key in indexed:
            raise EvidenceError(
                "duplicate_seed_identity", f"duplicate report for {key}"
            )
        if arm == DIRECT_STATE_ARM and split != "development":
            raise EvidenceError(
                "direct_state_confirmation_forbidden",
                "direct-state diagnostics are frozen to the three public development identities",
            )

        dataset_payload = dataset_summary["payload_sha256"]
        dataset_identity_binding = (dataset_payload, str(dataset_root))
        prior_dataset = dataset_by_identity.setdefault(
            identity_key, dataset_identity_binding
        )
        if prior_dataset != dataset_identity_binding:
            raise EvidenceError(
                "dataset_identity_fork",
                f"identity {identity_key} has multiple datasets or roots",
            )
        prior_identity = identity_by_dataset.setdefault(dataset_payload, identity_key)
        if prior_identity != identity_key:
            raise EvidenceError(
                "dataset_identity_collision", "one dataset has multiple identities"
            )

        bundle, bundle_binding, bundle_root = _verify_rooted_json_reference(
            run["trainer_bundle"], f"{label}.trainer_bundle", base
        )
        bundle_payload = _verify_payload_hash(
            bundle, f"{label}.trainer_bundle.manifest"
        )
        bundle_cache_key = (bundle_root, bundle_payload)
        bundle_summary = bundle_cache.get(bundle_cache_key)
        if bundle_summary is None:
            bundle_summary = _validate_trainer_bundle(
                bundle_root,
                bundle,
                dataset,
                dataset_summary,
                f"{label}.trainer_bundle.manifest",
            )
            bundle_cache[bundle_cache_key] = bundle_summary
        elif (
            bundle_summary["source_manifest_payload_sha256"] != dataset_payload
            or bundle_summary["seed_identity"] != dataset_summary["seed_identity"]
        ):
            raise EvidenceError(
                "bundle_dataset_mismatch",
                f"{label} reuses a trainer bundle against another dataset",
            )
        expected_schedule_kind = (
            "uniform_schedule.jsonl"
            if arm == "uniform_query_acw"
            else "cgb_schedule.jsonl"
        )
        if bundle_summary["query_schedule_kind"] != expected_schedule_kind:
            raise EvidenceError(
                "query_schedule_kind_mismatch",
                f"{label} trainer bundle uses the wrong schedule family",
            )

        checkpoint_binding, checkpoint_path = _verify_file_reference(
            run["checkpoint"], f"{label}.checkpoint", base
        )
        if checkpoint_path in checkpoint_paths:
            raise EvidenceError(
                "checkpoint_reused", f"{label} reuses another run checkpoint path"
            )
        checkpoint_paths.add(checkpoint_path)
        checkpoint_summary = _validate_checkpoint_artifact(
            checkpoint_path,
            str(checkpoint_binding["sha256"]),
            logical_arm=arm,
            dataset_summary=dataset_summary,
            bundle_summary=bundle_summary,
            label=f"{label}.checkpoint",
        )

        evaluation, evaluation_binding, evaluation_path = _verify_json_reference(
            run["evaluation_report"], f"{label}.evaluation_report", base
        )
        replay, replay_binding, replay_path = _verify_json_reference(
            run["replay_report"], f"{label}.replay_report", base
        )
        if evaluation_path == replay_path:
            raise EvidenceError(
                "replay_path_reused", f"{label} replay must be a distinct artifact"
            )
        if evaluation_binding["sha256"] != replay_binding["sha256"]:
            raise EvidenceError(
                "replay_hash_mismatch", f"{label} replay is not byte-identical"
            )
        evaluation_bytes, _ = _read_regular_file(evaluation_path)
        replay_bytes, _ = _read_regular_file(replay_path)
        if evaluation_bytes != replay_bytes:
            raise EvidenceError(
                "replay_byte_mismatch",
                f"{label} evaluator outputs differ at the byte level",
            )
        if evaluation_path in report_paths or replay_path in report_paths:
            raise EvidenceError(
                "evaluation_artifact_reused", f"{label} reuses an evaluator artifact"
            )
        report_paths.update((evaluation_path, replay_path))
        if evaluation_binding["sha256"] in primary_report_hashes:
            raise EvidenceError(
                "evaluation_payload_reused", f"{label} reuses another run report"
            )
        primary_report_hashes.add(evaluation_binding["sha256"])

        parsed = _validate_evaluation_report(
            evaluation,
            logical_arm=arm,
            dataset_payload_sha256=dataset_payload,
            dataset_seed_identity=dataset_summary["seed_identity"],
            label=f"{label}.evaluation_report",
        )
        replay_parsed = _validate_evaluation_report(
            replay,
            logical_arm=arm,
            dataset_payload_sha256=dataset_payload,
            dataset_seed_identity=dataset_summary["seed_identity"],
            label=f"{label}.replay_report",
        )
        if replay_parsed != parsed:
            raise EvidenceError(
                "replay_semantic_mismatch", f"{label} replay content differs"
            )

        if parsed["checkpoint_sha256"] != checkpoint_binding["sha256"]:
            raise EvidenceError(
                "evaluation_checkpoint_mismatch",
                f"{label} report does not bind the opened checkpoint file",
            )
        if parsed["training_evidence"] != checkpoint_summary["training_evidence"]:
            raise EvidenceError(
                "evaluation_checkpoint_training_mismatch",
                f"{label} report training evidence differs from the checkpoint",
            )
        if parsed["scientific_identity"] != checkpoint_summary["scientific_identity"]:
            raise EvidenceError(
                "evaluation_checkpoint_identity_mismatch",
                f"{label} report scientific identity differs from the checkpoint",
            )
        independent_replay = _independent_evaluator_replay(
            checkpoint_path,
            dataset_root,
            evaluation_bytes,
            label,
        )
        if independent_replay["payload_sha256"] != parsed["payload_sha256"]:
            raise EvidenceError(
                "independent_evaluator_mismatch",
                f"{label} fresh evaluator payload hash differs",
            )

        current_identity = parsed["scientific_identity"]
        if bundle_summary["pilot_scientific_identity"] != current_identity:
            raise EvidenceError(
                "pilot_scientific_identity_mismatch",
                f"{label} pilot and trained evidence use different scientific code",
            )
        if scientific_identity is None:
            scientific_identity = current_identity
        elif current_identity != scientific_identity:
            raise EvidenceError(
                "scientific_identity_fork", f"{label} uses different scientific code"
            )

        checkpoint_sha256 = str(checkpoint_binding["sha256"])
        if checkpoint_sha256 in checkpoint_hashes:
            raise EvidenceError(
                "checkpoint_reused", f"{label} reuses another run checkpoint"
            )
        checkpoint_hashes.add(checkpoint_sha256)
        current_pilot = parsed["pilot_report_payload_sha256"]
        if pilot_report_payload_sha256 is None:
            pilot_report_payload_sha256 = current_pilot
        elif current_pilot != pilot_report_payload_sha256:
            raise EvidenceError(
                "pilot_report_binding_fork",
                f"{label} uses a different frozen pilot report",
            )

        training_evidence = parsed["training_evidence"]
        schedule_kind = parsed["query_schedule_kind"]
        if schedule_kind != bundle_summary["query_schedule_kind"]:
            raise EvidenceError(
                "evaluation_bundle_binding_mismatch",
                f"{label} report schedule kind differs from the opened bundle",
            )
        expected_training_bindings = {
            "trainer_bundle_manifest_payload_sha256": bundle_summary["payload_sha256"],
            "curriculum_sha256": bundle_summary["curriculum_sha256"],
            "query_schedule_sha256": bundle_summary["query_schedule_sha256"],
        }
        for field, expected in expected_training_bindings.items():
            if training_evidence[field] != expected:
                raise EvidenceError(
                    "evaluation_bundle_binding_mismatch",
                    f"{label} report {field} differs from the opened bundle",
                )
        if current_pilot != bundle_summary["pilot_report_payload_sha256"]:
            raise EvidenceError(
                "evaluation_bundle_binding_mismatch",
                f"{label} report pilot binding differs from the opened bundle",
            )
        query_schedule_hashes[schedule_kind].add(
            training_evidence["query_schedule_sha256"]
        )
        artifact_group = (split, index, schedule_kind)
        trainer_bundle_hash = training_evidence[
            "trainer_bundle_manifest_payload_sha256"
        ]
        curriculum_hash = training_evidence["curriculum_sha256"]

        prior_bundle = trainer_bundle_by_group.setdefault(
            artifact_group, trainer_bundle_hash
        )
        if prior_bundle != trainer_bundle_hash:
            raise EvidenceError(
                "trainer_bundle_binding_fork",
                f"{label} changes trainer bundle within {artifact_group}",
            )
        prior_bundle_root = trainer_bundle_root_by_group.setdefault(
            artifact_group, str(bundle_root)
        )
        if prior_bundle_root != str(bundle_root):
            raise EvidenceError(
                "trainer_bundle_root_fork",
                f"{label} changes trainer-bundle root within {artifact_group}",
            )
        prior_curriculum = curriculum_by_group.setdefault(
            artifact_group, curriculum_hash
        )
        if prior_curriculum != curriculum_hash:
            raise EvidenceError(
                "curriculum_binding_fork",
                f"{label} changes curriculum within {artifact_group}",
            )

        prior_bundle_group = group_by_trainer_bundle.setdefault(
            trainer_bundle_hash, artifact_group
        )
        if prior_bundle_group != artifact_group:
            raise EvidenceError(
                "trainer_bundle_reused_across_domains",
                f"{label} reuses a trainer bundle across distinct seed/schedule domains",
            )
        prior_curriculum_group = group_by_curriculum.setdefault(
            curriculum_hash, artifact_group
        )
        if prior_curriculum_group != artifact_group:
            raise EvidenceError(
                "curriculum_reused_across_domains",
                f"{label} reuses a curriculum where answers or schedule differ",
            )

        indexed[key] = {
            "arm": arm,
            "split": split,
            "index": index,
            "seed_identity": dataset_summary["seed_identity"],
            "evaluation": parsed,
            "training": parsed["training_evidence"],
            "inference": parsed["training_evidence"]["resource_measurements"][
                "inference"
            ],
            "label_efficiency": parsed["label_efficiency"],
            "bindings": {
                "dataset": dataset_binding,
                "trainer_bundle": bundle_binding,
                "checkpoint": checkpoint_binding,
                "evaluation_report": evaluation_binding,
                "replay_report": replay_binding,
                "independent_evaluator_replay": independent_replay,
            },
        }
        artifact_bindings.append(
            {
                "arm": arm,
                "seed_identity": dataset_summary["seed_identity"],
                "dataset": {
                    **dataset_binding,
                    "arrays_hashed_and_opened": dataset_tree[
                        "arrays_hashed_and_opened"
                    ],
                    "required_array_shapes_verified": dataset_tree[
                        "required_array_shapes_verified"
                    ],
                },
                "trainer_bundle": {
                    **bundle_binding,
                    "arrays_hashed_and_opened": bundle_summary[
                        "arrays_hashed_and_opened"
                    ],
                    "curriculum_sha256": bundle_summary["curriculum_sha256"],
                },
                "checkpoint": checkpoint_binding,
                "evaluation_report": evaluation_binding,
                "replay_report": replay_binding,
                "independent_evaluator_replay": independent_replay,
            }
        )

    if any(len(hashes) != 1 for hashes in query_schedule_hashes.values()):
        raise EvidenceError(
            "query_schedule_binding_fork",
            "all non-uniform runs must share one CGB schedule hash and all uniform-query runs one uniform schedule hash",
        )
    frozen_schedule_hashes = {
        kind: next(iter(hashes)) for kind, hashes in query_schedule_hashes.items()
    }
    if len(set(frozen_schedule_hashes.values())) != 2:
        raise EvidenceError(
            "query_schedule_hash_reused",
            "CGB and uniform schedules must bind distinct artifacts",
        )

    missing = sorted(expected_keys - set(indexed))
    extra = sorted(set(indexed) - expected_keys)
    if missing or extra:
        raise EvidenceError("run_matrix_mismatch", f"missing={missing}, extra={extra}")
    assert scientific_identity is not None
    ordered = [
        indexed[key]
        for key in sorted(
            indexed,
            key=lambda item: (
                (*SCORED_ARMS, DIRECT_STATE_ARM).index(item[0]),
                0 if item[1] == "development" else 1,
                item[2],
            ),
        )
    ]
    verification = {
        "status": "verified",
        "manifest_payload_sha256": manifest_payload_sha256,
        "exact_run_matrix": True,
        "scored_runs_verified": len(SCORED_ARMS) * 6,
        "direct_state_runs_verified": 3,
        "evaluation_reports_verified": len(ordered),
        "byte_identical_replays_verified": len(ordered),
        "independent_evaluator_replays_verified": len(ordered),
        "actual_checkpoints_opened_and_hashed": len(checkpoint_hashes),
        "unique_dataset_manifests_verified": len(identity_by_dataset),
        "unique_dataset_roots_opened": len(dataset_cache),
        "unique_trainer_bundles_opened": len(bundle_cache),
        "dataset_arrays_hashed_and_opened": sum(
            tree["arrays_hashed_and_opened"] for _, tree in dataset_cache.values()
        ),
        "trainer_bundle_arrays_hashed_and_opened": sum(
            summary["arrays_hashed_and_opened"] for summary in bundle_cache.values()
        ),
        "trainer_bundle_pilot_artifacts_opened": sum(
            summary["pilot_artifacts_opened"] for summary in bundle_cache.values()
        ),
        "resource_ledgers_complete": True,
        "label_efficiency_records_complete": True,
        "compiled_sparse_controls_passed": len(ordered),
        "pilot_report_payload_sha256": pilot_report_payload_sha256,
        "query_schedule_sha256": frozen_schedule_hashes,
        "trainer_bundle_bindings": [
            {
                "split": split,
                "index": index,
                "query_schedule_kind": schedule,
                "trainer_bundle_manifest_payload_sha256": (
                    trainer_bundle_by_group[(split, index, schedule)]
                ),
                "trainer_bundle_root": trainer_bundle_root_by_group[
                    (split, index, schedule)
                ],
                "curriculum_sha256": curriculum_by_group[(split, index, schedule)],
            }
            for split, index, schedule in sorted(trainer_bundle_by_group)
        ],
        "scientific_identity": scientific_identity,
        "artifact_bindings": artifact_bindings,
    }
    return ordered, verification


def _acw_gate(run: dict[str, Any]) -> dict[str, Any]:
    report = run["evaluation"]
    failures = []
    for depth, (scalar_floor, state_floor) in DEPTH_THRESHOLDS.items():
        metric = report["public_depths"][depth]
        if metric["scalar_accuracy"] < scalar_floor:
            failures.append(f"depth_{depth}_scalar_below_{scalar_floor}")
        if metric["state_exactness"] < state_floor:
            failures.append(f"depth_{depth}_state_below_{state_floor}")
    packet = report["packet_interventions"]
    if packet["donor_following"]["scalar_accuracy"] < DONOR_SCALAR_FLOOR:
        failures.append("donor_following_scalar_below_0.99")
    if packet["shuffled_against_original"]["scalar_accuracy"] > SHUFFLE_SCALAR_CEILING:
        failures.append("shuffled_scalar_above_chance_plus_0.02")
    if not packet["held_packet_source_swap_predictions_identical"]:
        failures.append("held_packet_source_swap_changed_predictions")
    if packet["donor_different_truth_fraction"] <= 0.0:
        failures.append("donor_map_has_no_truth_change")
    for depth in EVALUATION_DEPTHS:
        metric = report["new_reader_depths"][depth]
        if metric["scalar_accuracy"] < NEW_READER_SCALAR_FLOOR:
            failures.append(f"new_reader_depth_{depth}_scalar_below_0.98")
        if metric["state_exactness"] < NEW_READER_STATE_FLOOR:
            failures.append(f"new_reader_depth_{depth}_state_below_0.90")
    if report["write_legality"]["illegal_writes"] != 0:
        failures.append("illegal_multi_register_write")
    words = report["event_words"]
    if words["equivalent_prediction_query_equivalence"] != 1.0:
        failures.append("equivalent_event_words_not_query_equivalent")
    if words["non_equivalent_target_separator_rate"] != 1.0:
        failures.append("non_equivalent_event_words_lack_target_separator")
    if words["non_equivalent_prediction_separator_rate"] != 1.0:
        failures.append("non_equivalent_event_words_lack_prediction_separator")
    return {"passed": not failures, "failures": failures}


def _direct_state_gate(run: dict[str, Any]) -> dict[str, Any]:
    metric = run["evaluation"]["public_depths"][8]
    failures = []
    if metric["scalar_accuracy"] < DIRECT_STATE_SCALAR_FLOOR:
        failures.append("depth_8_scalar_below_0.99")
    if metric["state_exactness"] < DIRECT_STATE_STATE_FLOOR:
        failures.append("depth_8_state_below_0.95")
    return {"passed": not failures, "failures": failures}


def _metric_summary(metric: dict[str, Any]) -> dict[str, float]:
    return {
        "scalar_accuracy": metric["scalar_accuracy"],
        "state_exactness": metric["state_exactness"],
    }


def _seed_result(run: dict[str, Any], gate: dict[str, Any] | None) -> dict[str, Any]:
    report = run["evaluation"]
    result = {
        "arm": run["arm"],
        "role": (
            "direct_state_diagnostic" if run["arm"] == DIRECT_STATE_ARM else "scored"
        ),
        "split": run["split"],
        "index": run["index"],
        "seed_identity": run["seed_identity"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifact_bindings": run["bindings"],
        "optimizer_seed": report["optimizer_seed"],
        "query_schedule_kind": report["query_schedule_kind"],
        "pilot_report_payload_sha256": report["pilot_report_payload_sha256"],
        "public_depths": {
            str(depth): _metric_summary(report["public_depths"][depth])
            for depth in EVALUATION_DEPTHS
        },
        "new_reader_depths": {
            str(depth): _metric_summary(report["new_reader_depths"][depth])
            for depth in EVALUATION_DEPTHS
        },
        "training_evidence": run["training"],
        "label_efficiency": run["label_efficiency"],
        "compiled_sparse_control": report["compiled_sparse_control"],
        "frozen_gate": gate,
    }
    if "packet_interventions" in report:
        result["causal_diagnostics"] = {
            "donor_following": _metric_summary(
                report["packet_interventions"]["donor_following"]
            ),
            "shuffled_against_original": _metric_summary(
                report["packet_interventions"]["shuffled_against_original"]
            ),
            "held_packet_source_swap_predictions_identical": report[
                "packet_interventions"
            ]["held_packet_source_swap_predictions_identical"],
            "donor_different_truth_fraction": report["packet_interventions"][
                "donor_different_truth_fraction"
            ],
            "write_legality": report["write_legality"],
            "event_word_rates": {
                name: report["event_words"][name]
                for name in (
                    "equivalent_prediction_query_equivalence",
                    "non_equivalent_target_separator_rate",
                    "non_equivalent_prediction_separator_rate",
                )
            },
        }
    return result


def _confirmation_medians(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in SCORED_ARMS:
        selected = [
            run for run in runs if run["arm"] == arm and run["split"] == "confirmation"
        ]
        assert len(selected) == 3
        public = {}
        reader = {}
        for depth in EVALUATION_DEPTHS:
            public[str(depth)] = {
                name: float(
                    median(
                        run["evaluation"]["public_depths"][depth][name]
                        for run in selected
                    )
                )
                for name in ("scalar_accuracy", "state_exactness")
            }
            reader[str(depth)] = {
                name: float(
                    median(
                        run["evaluation"]["new_reader_depths"][depth][name]
                        for run in selected
                    )
                )
                for name in ("scalar_accuracy", "state_exactness")
            }
        efficiency = []
        for position, labels in enumerate(LABEL_CHECKPOINTS):
            efficiency.append(
                {
                    "labels": labels,
                    "depth_64_scalar_accuracy": float(
                        median(
                            run["label_efficiency"][position][
                                "depth_64_scalar_accuracy"
                            ]
                            for run in selected
                        )
                    ),
                    "depth_64_state_exactness": float(
                        median(
                            run["label_efficiency"][position][
                                "depth_64_state_exactness"
                            ]
                            for run in selected
                        )
                    ),
                }
            )
        result[arm] = {
            "public_depths": public,
            "new_reader_depths": reader,
            "label_efficiency": efficiency,
        }
        if "packet_interventions" in selected[0]["evaluation"]:
            result[arm]["causal_diagnostics"] = {
                "donor_following_scalar_accuracy": float(
                    median(
                        run["evaluation"]["packet_interventions"]["donor_following"][
                            "scalar_accuracy"
                        ]
                        for run in selected
                    )
                ),
                "shuffled_against_original_scalar_accuracy": float(
                    median(
                        run["evaluation"]["packet_interventions"][
                            "shuffled_against_original"
                        ]["scalar_accuracy"]
                        for run in selected
                    )
                ),
                "all_source_swaps_identical": all(
                    run["evaluation"]["packet_interventions"][
                        "held_packet_source_swap_predictions_identical"
                    ]
                    for run in selected
                ),
                "donor_different_truth_fraction": float(
                    median(
                        run["evaluation"]["packet_interventions"][
                            "donor_different_truth_fraction"
                        ]
                        for run in selected
                    )
                ),
                "illegal_writes_median": float(
                    median(
                        run["evaluation"]["write_legality"]["illegal_writes"]
                        for run in selected
                    )
                ),
            }
    return result


def _label_efficiency_summary(confirmation: dict[str, Any]) -> dict[str, Any]:
    crossings: dict[str, int | None] = {}
    for arm in SCORED_ARMS:
        crossing = next(
            (
                row["labels"]
                for row in confirmation[arm]["label_efficiency"]
                if row["depth_64_state_exactness"] >= PRIMARY_STATE_FLOOR
            ),
            None,
        )
        crossings[arm] = crossing
    acw_crossing = crossings["acw"]
    uniform_crossing = crossings["uniform_query_acw"]
    ratio = None
    statement_allowed = acw_crossing is not None and uniform_crossing is not None
    if statement_allowed:
        ratio = uniform_crossing / acw_crossing
    return {
        "first_confirmation_median_crossing_labels": crossings,
        "cgbr_efficiency_statement_allowed": statement_allowed,
        "uniform_to_cgbr_first_crossing_label_ratio": ratio,
        "secondary_only_cannot_rescue_primary": True,
    }


def _requirements() -> dict[str, Any]:
    return {
        "scored_arms": list(SCORED_ARMS),
        "valid_equal_label_architecture_controls": list(
            VALID_EQUAL_LABEL_ARCHITECTURE_CONTROLS
        ),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_indices": [0, 1, 2],
        "confirmation_commitments": list(CONFIRMATION_COMMITMENTS),
        "direct_state_development_diagnostics": 3,
        "evaluation_depths": list(EVALUATION_DEPTHS),
        "depth_thresholds": {
            str(depth): {"scalar_accuracy": scalar, "state_exactness": state}
            for depth, (scalar, state) in DEPTH_THRESHOLDS.items()
        },
        "direct_state_gate": {
            "depth": 8,
            "scalar_accuracy": DIRECT_STATE_SCALAR_FLOOR,
            "state_exactness": DIRECT_STATE_STATE_FLOOR,
        },
        "donor_scalar_accuracy": DONOR_SCALAR_FLOOR,
        "shuffle_scalar_ceiling": SHUFFLE_SCALAR_CEILING,
        "new_reader_all_depths": {
            "scalar_accuracy": NEW_READER_SCALAR_FLOOR,
            "state_exactness": NEW_READER_STATE_FLOOR,
        },
        "illegal_writes": 0,
        "scalar_labels": FINAL_SCALAR_LABELS,
        "label_efficiency_checkpoints": list(LABEL_CHECKPOINTS),
        "acw_development_passes_required": 3,
        "acw_confirmation_passes_required": 2,
        "primary_confirmation_depth_64_state_floor": PRIMARY_STATE_FLOOR,
        "primary_control_margin": CONTROL_MARGIN_FLOOR,
        "byte_identical_replay_required": True,
        "complete_train_and_inference_resource_ledgers_required": True,
    }


def _bind_decision(payload: dict[str, Any]) -> dict[str, Any]:
    return with_payload_hash(payload)


def _evidence_rejection(
    manifest_path: Path,
    manifest_file_sha256: str | None,
    error: EvidenceError,
) -> dict[str, Any]:
    return _bind_decision(
        {
            "schema": DECISION_SCHEMA,
            "protocol": DECISION_PROTOCOL,
            "decision": "NO_GO",
            "go": False,
            "reasons": ["evidence_contract_failed", error.code],
            "failure_detail": error.detail,
            "manifest": {"path": str(manifest_path), "sha256": manifest_file_sha256},
            "requirements": _requirements(),
            "verification": {"status": "failed"},
            "bounded_claim": BOUNDED_CLAIM,
            "output_contract": {
                "exclusive_create": True,
                "overwrite": False,
                "mode": "0444",
            },
        }
    )


def adjudicate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify all evidence and return a payload-hashed GO/NO_GO decision."""

    path = Path(manifest_path)
    manifest_file_sha256: str | None = None
    try:
        raw, manifest_file_sha256 = _read_regular_file(path)
        manifest = _parse_json(raw, "manifest")
        runs, verification = verify_evidence(manifest, path.parent)
    except EvidenceError as exc:
        return _evidence_rejection(path, manifest_file_sha256, exc)
    except Exception as exc:
        return _evidence_rejection(
            path,
            manifest_file_sha256,
            EvidenceError(
                "adjudication_execution_failed",
                f"adjudication failed without fallback ({type(exc).__name__}): {exc}",
            ),
        )

    seed_results = []
    acw_gates: dict[tuple[str, int], dict[str, Any]] = {}
    direct_gates: dict[int, dict[str, Any]] = {}
    for run in runs:
        gate = None
        if run["arm"] == "acw":
            gate = _acw_gate(run)
            acw_gates[(run["split"], run["index"])] = gate
        elif run["arm"] == DIRECT_STATE_ARM:
            gate = _direct_state_gate(run)
            direct_gates[run["index"]] = gate
        seed_results.append(_seed_result(run, gate))

    confirmation = _confirmation_medians(runs)
    acw_median = confirmation["acw"]["public_depths"]["64"]["state_exactness"]
    control_medians = {
        arm: confirmation[arm]["public_depths"]["64"]["state_exactness"]
        for arm in VALID_EQUAL_LABEL_ARCHITECTURE_CONTROLS
    }
    strongest_control = max(control_medians, key=control_medians.__getitem__)
    strongest_control_median = control_medians[strongest_control]
    margin = acw_median - strongest_control_median

    development_passes = sum(
        acw_gates[("development", index)]["passed"] for index in range(3)
    )
    confirmation_passes = sum(
        acw_gates[("confirmation", index)]["passed"] for index in range(3)
    )
    direct_passes = sum(direct_gates[index]["passed"] for index in range(3))

    reasons = []
    if direct_passes != 3:
        reasons.append("direct_state_diagnostic_gate_failed")
    if development_passes != 3:
        reasons.append("acw_all_development_seed_rule_failed")
    if confirmation_passes < 2:
        reasons.append("acw_two_of_three_confirmation_rule_failed")
    if acw_median < PRIMARY_STATE_FLOOR:
        reasons.append("primary_confirmation_depth_64_state_below_0.90")
    if margin < CONTROL_MARGIN_FLOOR:
        reasons.append("primary_equal_label_control_margin_below_0.10")

    go = not reasons
    return _bind_decision(
        {
            "schema": DECISION_SCHEMA,
            "protocol": DECISION_PROTOCOL,
            "decision": "GO" if go else "NO_GO",
            "go": go,
            "reasons": reasons,
            "manifest": {
                "path": str(path),
                "sha256": manifest_file_sha256,
                "payload_sha256": verification["manifest_payload_sha256"],
            },
            "requirements": _requirements(),
            "verification": verification,
            "direct_state_diagnostic": {
                "passed": direct_passes == 3,
                "passes": direct_passes,
                "required": 3,
                "seeds": [
                    {"index": index, **direct_gates[index]} for index in range(3)
                ],
            },
            "acw_seed_rule": {
                "development_passes": development_passes,
                "development_required": 3,
                "confirmation_passes": confirmation_passes,
                "confirmation_required": 2,
            },
            "seed_results": seed_results,
            "confirmation_medians": confirmation,
            "primary_endpoint": {
                "labels": FINAL_SCALAR_LABELS,
                "metric": "depth_64_state_exactness",
                "acw_confirmation_median": acw_median,
                "acw_floor": PRIMARY_STATE_FLOOR,
                "acw_floor_passed": acw_median >= PRIMARY_STATE_FLOOR,
                "valid_equal_label_control_confirmation_medians": control_medians,
                "strongest_valid_equal_label_control": strongest_control,
                "strongest_control_confirmation_median": strongest_control_median,
                "absolute_margin": margin,
                "required_absolute_margin": CONTROL_MARGIN_FLOOR,
                "control_margin_passed": margin >= CONTROL_MARGIN_FLOOR,
            },
            "label_efficiency": _label_efficiency_summary(confirmation),
            "bounded_claim": BOUNDED_CLAIM,
            "output_contract": {
                "exclusive_create": True,
                "overwrite": False,
                "mode": "0444",
            },
        }
    )


def write_immutable_json(path: str | Path, payload: dict[str, Any]) -> str:
    """Exclusively create, fsync, and remove write bits from one decision JSON."""

    decision = _object(payload, "decision")
    _verify_payload_hash(decision, "decision")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("protocol") != DECISION_PROTOCOL
    ):
        raise EvidenceError(
            "decision_protocol_mismatch", "decision schema or protocol is not frozen v2"
        )
    go = _boolean(decision.get("go"), "decision.go")
    expected_decision = "GO" if go else "NO_GO"
    if decision.get("decision") != expected_decision:
        raise EvidenceError(
            "decision_state_mismatch", "decision and go fields disagree"
        )
    if decision.get("bounded_claim") != BOUNDED_CLAIM:
        raise EvidenceError("decision_claim_mismatch", "decision bounded claim differs")
    if decision.get("output_contract") != {
        "exclusive_create": True,
        "overwrite": False,
        "mode": "0444",
    }:
        raise EvidenceError(
            "decision_output_contract_mismatch", "decision output contract differs"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(decision) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating decision")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Frozen R12 ACW evidence manifest"
    )
    parser.add_argument("--out", required=True, help="New immutable decision JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing decision: {args.out}")
    decision = adjudicate_manifest(args.manifest)
    file_sha256 = write_immutable_json(args.out, decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "decision_file_sha256": file_sha256,
                "decision_payload_sha256": decision["payload_sha256"],
                "reasons": decision["reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
