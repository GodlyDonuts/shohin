#!/usr/bin/env python3
"""Frozen, fail-closed adjudicator for the R12 ACW CPU experiment.

The input is a hash-bound manifest containing one record for every frozen run.
Each record binds a real checkpoint, evaluation-domain root and manifest,
trainer-bundle root and manifest, evaluator report, and second evaluator output.
Before any confirmation artifact is opened, the adjudicator independently
replays an immutable development-only baseline and reconstructs its selection.
The full manifest must bind those exact baseline bytes.  The adjudicator then
opens and hashes all artifacts, checks their transitive bindings, and executes
a third evaluator replay in a fresh interpreter before it accepts any metric.
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
import io
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

MANIFEST_SCHEMA = "r12_acw_adjudication_manifest_v4"
MANIFEST_PROTOCOL = "R12-ACW-ADJUDICATION-MANIFEST-v4"
DEVELOPMENT_MANIFEST_SCHEMA = "r12_acw_development_baseline_manifest_v2"
DEVELOPMENT_MANIFEST_PROTOCOL = "R12-ACW-DEVELOPMENT-BASELINE-MANIFEST-v2"
DIRECT_STATE_MANIFEST_SCHEMA = "r12_acw_direct_state_manifest_v1"
DIRECT_STATE_MANIFEST_PROTOCOL = "R12-ACW-DIRECT-STATE-MANIFEST-v1"
DIRECT_STATE_DECISION_SCHEMA = "r12_acw_direct_state_decision_v1"
DIRECT_STATE_DECISION_PROTOCOL = "R12-ACW-DIRECT-STATE-DECISION-v1"
PHASE2_AUTHORIZATION_SCHEMA = "r12_acw_phase2_authorization_v1"
PHASE2_AUTHORIZATION_PROTOCOL = "R12-ACW-PHASE2-AUTHORIZATION-v1"
ATTEMPT_START_SCHEMA = "r12_acw_development_attempt_start_v1"
ATTEMPT_START_PROTOCOL = "R12-ACW-DEVELOPMENT-ATTEMPT-START-v1"
DEVELOPMENT_BASELINE_SCHEMA = "r12_acw_development_baseline_v3"
DEVELOPMENT_BASELINE_PROTOCOL = "R12-ACW-DEVELOPMENT-BASELINE-v3"
CONFIRMATION_AUTHORIZATION_PROTOCOL = "R12-ACW-CONFIRMATION-AUTHORIZATION-v1"
DECISION_SCHEMA = "r12_acw_adjudication_decision_v4"
DECISION_PROTOCOL = "R12-ACW-ADJUDICATION-DECISION-v4"
EVALUATION_PROTOCOL = "R12-ACW-CAUSAL-EVALUATION-v2"
GENERATOR_PROTOCOL = "R12-ACW-HIDDEN-BASIS-v3"
TRAINING_PROTOCOL = "R12-ACW-TRAINER-v3"
TRAINER_BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v4"
PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v6"
PILOT_COMPARISON_PROTOCOL = "R12-ACW-PILOT-REPLAY-COMPARISON-v6"
PILOT_ARTIFACT_REGISTRY_PROTOCOL = "R12-ACW-PILOT-ARTIFACT-REGISTRY-v2"
PILOT_INDEPENDENT_VERIFICATION_PROTOCOL = "R12-ACW-PILOT-INDEPENDENT-VERIFICATION-v3"
TRAIN_LEDGER_PROTOCOL = "R12-ACW-TRAIN-RESOURCE-LEDGER-v2"
INFERENCE_LEDGER_PROTOCOL = "R12-ACW-INFERENCE-RESOURCE-LEDGER-v2"

PILOT_SCIENTIFIC_COMMIT = "5f5e3cd0d69da67335ad1f1f485c6e3d8f00ff8e"
PILOT_ANCHOR_COMMIT = "02c9d4ae57093b6c60d90580503e2a01c7c81619"
PILOT_EXECUTION_COMMIT = "38ebad21cf9c4ef98b172394891c2a35ef671b12"
PILOT_CUSTODY_COMMIT = "7433062211c4ad0371a975019c37625f7d811b27"
PILOT_REGISTRY_RAW_SHA256 = (
    "66597cf5381fdc11d4ecd73a93d9bbd2fa68417a77b09c1330ecfeb73652451c"
)
PILOT_REGISTRY_PATH = "R12_ACW_PILOT_ARTIFACT_REGISTRY_V2.json"
PILOT_ANCHORED_FILES = 81
PILOT_CANONICAL_REMOTE_URL = "https://github.com/GodlyDonuts/shohin.git"
PILOT_OFFLINE_BUNDLE_TEMPLATE = "/home/sa305415/shohin_acw_{commit8}.bundle"
PILOT_ACTIVATION_ALLOWLIST = (
    "AGENT_RUNBOOK.md",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/freeze_acw_curriculum.py",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
    "pipeline/test_freeze_acw_curriculum.py",
)
PILOT_CUSTODY_ALLOWLIST = (
    "AGENT_RUNBOOK.md",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
)
PILOT_DEVELOPMENT_ALLOWLIST = (
    "AGENT_RUNBOOK.md",
    "R12_ACW_DEVELOPMENT_PLAN_V1.json",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/build_acw_development_manifest.py",
    "pipeline/evaluate_acw_hidden_basis.py",
    "pipeline/jobs/run_acw_development_stokes.sbatch",
    "pipeline/jobs/run_acw_terminal_monitor_stokes.sbatch",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
    "pipeline/test_build_acw_development_manifest.py",
    "pipeline/test_acw_g_custody.py",
)
PILOT_CANONICAL_PATHS = {
    "dataset": "artifacts/r12/acw_pilot_domain_v3_runtime_v2",
    "pilot": "artifacts/r12/acw_cgbr_pilot_v6",
    "replay_a": "artifacts/r12/acw_cgbr_pilot_v6_replay_a",
    "replay_b": "artifacts/r12/acw_cgbr_pilot_v6_replay_b",
    "verification": "artifacts/r12/acw_cgbr_pilot_v6_independent_verification",
}
PILOT_REGISTRY_CLAIM = (
    "Byte registry for one non-scored Track S pilot and its independent replay. "
    "It authorizes no scored arm, Shohin fit, or reasoning claim."
)
PILOT_INDEPENDENT_VERIFICATION_CLAIM = (
    "Different-node deterministic replay of the non-scored Track S pilot. "
    "This is not a scored architecture or reasoning result."
)
PILOT_SCIENTIFIC_PATHS = (
    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md",
    "R12_GOAL_CONDITIONED_VERSION_SPACE_CONTROLLER_PREREG.md",
    "pipeline/addressed_categorical_workspace.py",
    "pipeline/audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/generate_acw_hidden_basis.py",
    "pipeline/acw_nist_beacon.py",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/freeze_acw_curriculum.py",
    "pipeline/evaluate_acw_hidden_basis.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/test_addressed_categorical_workspace.py",
    "pipeline/test_audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/test_generate_acw_hidden_basis.py",
    "pipeline/test_acw_nist_beacon.py",
    "pipeline/testdata/acw_nist_beacon_snapshot.json",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_freeze_acw_curriculum.py",
    "pipeline/test_evaluate_acw_hidden_basis.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
    "pipeline/jobs/run_acw_pilot_stokes.sbatch",
    "pipeline/jobs/verify_acw_pilot_stokes.sbatch",
)
DEVELOPMENT_PLAN_PATH = "R12_ACW_DEVELOPMENT_PLAN_V1.json"
DEVELOPMENT_PLAN_RAW_SHA256 = (
    "39f91a28f4ac0a593ecabd19942e598a4474d777baccb7367d0bbfd73128335d"
)
CANONICAL_DEVELOPMENT_RUNTIME_SHA256 = (
    "0e91de0e3dbca24ea4f04b9b03398a91486b93b31eff5a3ba4574dd43eaa677f"
)
DEVELOPMENT_EXECUTION_PATHS = (
    DEVELOPMENT_PLAN_PATH,
    "pipeline/build_acw_development_manifest.py",
    "pipeline/jobs/run_acw_development_stokes.sbatch",
    "pipeline/jobs/run_acw_terminal_monitor_stokes.sbatch",
)
ACW_SCIENTIFIC_PATHS = (*PILOT_SCIENTIFIC_PATHS, *DEVELOPMENT_EXECUTION_PATHS)

DEVELOPMENT_SEEDS = (2026071601, 2026071602, 2026071603)
CONFIRMATION_COMMITMENTS: tuple[str, ...] = ()

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
BASELINE_ELIGIBLE_ARMS = tuple(arm for arm in SCORED_ARMS if arm != "source_retained")

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
DEVELOPMENT_BASELINE_CLAIM = (
    "This is the strongest deployable checkpoint under the frozen development-only "
    "R12 protocol. It is retained even when later adjudication is NO_GO, but it "
    "cannot override promotion gates and is not a reasoning or generalization claim."
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
    "canonical_runtime_sha256",
    "development_plan_sha256",
    "execution_receipt",
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


def _read_regular_file_with_stat(path: Path) -> tuple[bytes, str, os.stat_result]:
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
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise EvidenceError(
                "artifact_changed_during_read", f"artifact changed: {path}"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest(), after


def _read_regular_file(path: Path) -> tuple[bytes, str]:
    raw, digest, _ = _read_regular_file_with_stat(path)
    return raw, digest


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


def _read_binary_regular_file(path: Path, label: str) -> tuple[bytes, str]:
    """Read and hash one stable regular binary artifact through one descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(
            "artifact_unreadable", f"cannot open {label}: {exc}"
        ) from exc
    chunks: list[bytes] = []
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
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise EvidenceError("artifact_changed_during_read", f"{label} changed")
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest()


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


def _anchor_git_text(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise EvidenceError(
            "pilot_anchor_invalid",
            f"pilot anchor Git command failed: {' '.join(arguments)}",
        ) from exc


def _anchor_git_blob(root: Path, revision: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "show",
                f"{revision}:{relative}",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise EvidenceError(
            "pilot_anchor_invalid",
            f"pilot anchor lacks {relative} at {revision}",
        ) from exc


def _anchor_commit_parents(root: Path, commit: str) -> list[str]:
    return _anchor_git_text(root, "show", "-s", "--format=%P", commit).stdout.split()


def _anchor_diff(root: Path, before: str, after: str) -> list[tuple[str, str]]:
    records = []
    output = _anchor_git_text(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        before,
        after,
    ).stdout
    for line in output.splitlines():
        status, relative = line.split("\t", 1)
        records.append((status, relative))
    return records


def _adjudicator_activation_commit(root: Path) -> str:
    replacement_refs = _anchor_git_text(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    ).stdout.strip()
    grafts_raw = _anchor_git_text(
        root,
        "rev-parse",
        "--git-path",
        "info/grafts",
    ).stdout.strip()
    grafts_path = Path(grafts_raw)
    if not grafts_path.is_absolute():
        grafts_path = root / grafts_path
    if replacement_refs or grafts_path.exists():
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation forbids Git replacements and grafts",
        )
    activation_commit = _anchor_git_text(root, "rev-parse", "HEAD").stdout.strip()
    if activation_commit in {PILOT_SCIENTIFIC_COMMIT, PILOT_ANCHOR_COMMIT}:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation requires a distinct E commit",
        )
    if _anchor_commit_parents(root, PILOT_ANCHOR_COMMIT) != [PILOT_SCIENTIFIC_COMMIT]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot anchor A must have S as its sole parent",
        )
    if _anchor_diff(root, PILOT_SCIENTIFIC_COMMIT, PILOT_ANCHOR_COMMIT) != [
        ("A", PILOT_REGISTRY_PATH)
    ]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot anchor A must add only the registry",
        )
    if _anchor_commit_parents(root, PILOT_EXECUTION_COMMIT) != [PILOT_ANCHOR_COMMIT]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation E must have A as its sole parent",
        )
    if _anchor_diff(root, PILOT_ANCHOR_COMMIT, PILOT_EXECUTION_COMMIT) != [
        ("M", relative) for relative in sorted(PILOT_ACTIVATION_ALLOWLIST)
    ]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation E differs from its exact allowlist",
        )
    if _anchor_commit_parents(root, PILOT_CUSTODY_COMMIT) != [PILOT_EXECUTION_COMMIT]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot custody F must have E as its sole parent",
        )
    if _anchor_diff(root, PILOT_EXECUTION_COMMIT, PILOT_CUSTODY_COMMIT) != [
        ("M", relative) for relative in sorted(PILOT_CUSTODY_ALLOWLIST)
    ]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot custody F differs from its exact allowlist",
        )
    if activation_commit == PILOT_EXECUTION_COMMIT:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation requires custody successor F or G",
        )
    if activation_commit != PILOT_CUSTODY_COMMIT:
        if _anchor_commit_parents(root, activation_commit) != [PILOT_CUSTODY_COMMIT]:
            raise EvidenceError(
                "pilot_anchor_invalid",
                "development custody G must have F as its sole parent",
            )
        added = {
            DEVELOPMENT_PLAN_PATH,
            "pipeline/build_acw_development_manifest.py",
            "pipeline/jobs/run_acw_development_stokes.sbatch",
            "pipeline/test_build_acw_development_manifest.py",
            "pipeline/test_acw_g_custody.py",
        }
        expected = [
            ("A" if relative in added else "M", relative)
            for relative in sorted(PILOT_DEVELOPMENT_ALLOWLIST)
        ]
        if _anchor_diff(root, PILOT_CUSTODY_COMMIT, activation_commit) != expected:
            raise EvidenceError(
                "pilot_anchor_invalid",
                "development custody G differs from its exact allowlist",
            )
    absent = _anchor_git_text(
        root,
        "cat-file",
        "-e",
        f"{PILOT_SCIENTIFIC_COMMIT}:{PILOT_REGISTRY_PATH}",
        check=False,
    )
    if absent.returncode == 0:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot registry already existed in S",
        )

    namespace_init = root / "pipeline" / "__init__.py"
    committed_namespace_init = _anchor_git_text(
        root,
        "cat-file",
        "-e",
        f"{activation_commit}:pipeline/__init__.py",
        check=False,
    )
    if os.path.lexists(namespace_init) or committed_namespace_init.returncode == 0:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation requires an unshadowed pipeline namespace",
        )

    remote_url = _anchor_git_text(root, "remote", "get-url", "origin").stdout.strip()
    expected_bundle = PILOT_OFFLINE_BUNDLE_TEMPLATE.format(
        commit8=activation_commit[:8]
    )
    if remote_url == PILOT_CANONICAL_REMOTE_URL:
        remote = _anchor_git_text(
            root,
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/main",
        ).stdout.split()
    elif remote_url == expected_bundle:
        bundle = Path(remote_url)
        if not bundle.is_file() or bundle.is_symlink():
            raise EvidenceError(
                "pilot_anchor_invalid",
                "pilot activation offline bundle is not a regular file",
            )
        _anchor_git_text(root, "bundle", "verify", remote_url)
        remote = _anchor_git_text(
            root,
            "bundle",
            "list-heads",
            remote_url,
            "refs/heads/main",
        ).stdout.split()
    else:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation origin is not an approved publication route",
        )
    if not remote or remote[0] != activation_commit:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation HEAD must equal pushed origin/main",
        )
    protected = sorted(
        set(PILOT_SCIENTIFIC_PATHS)
        | set(PILOT_ACTIVATION_ALLOWLIST)
        | set(PILOT_CUSTODY_ALLOWLIST)
        | set(PILOT_DEVELOPMENT_ALLOWLIST)
        | {PILOT_REGISTRY_PATH}
    )
    status = _anchor_git_text(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *protected,
    ).stdout.strip()
    if status:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot activation paths are not clean in Git",
        )
    for relative in protected:
        path = root / relative
        raw, _ = _read_regular_file(path)
        if raw != _anchor_git_blob(root, activation_commit, relative):
            raise EvidenceError(
                "pilot_anchor_invalid",
                f"pilot activation path differs from HEAD: {relative}",
            )
    return activation_commit


def _load_anchor_json(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _read_regular_file(path)
    parsed = _parse_json(raw, label)
    if raw != canonical_json_bytes(parsed) + b"\n":
        raise EvidenceError(
            "noncanonical_json",
            f"{label} is not canonical newline-framed JSON",
        )
    _verify_payload_hash(parsed, label)
    return parsed, raw, digest


def _anchor_artifact_record(value: Any, label: str) -> dict[str, Any]:
    record = _object(value, label)
    _expect_keys(record, {"bytes", "sha256"}, label)
    size = _integer(record["bytes"], f"{label}.bytes", minimum=0)
    return {"bytes": size, "sha256": _hash(record["sha256"], f"{label}.sha256")}


def _anchor_artifact_path(root: Path, relative: str, record: Any) -> Path:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != relative
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            f"unsafe registered pilot path: {relative}",
        )
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceError(
                "pilot_anchor_invalid",
                f"registered pilot path is a symlink: {relative}",
            )
    checked = _anchor_artifact_record(record, f"pilot_registry[{relative!r}]")
    size, digest = _hash_regular_file(current, f"pilot_registry[{relative!r}]")
    if size != checked["bytes"] or digest != checked["sha256"]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            f"registered pilot artifact differs: {relative}",
        )
    return current


def _load_adjudicator_pilot_anchor() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    activation_commit = _adjudicator_activation_commit(root)
    registry, registry_raw, registry_digest = _load_anchor_json(
        root / PILOT_REGISTRY_PATH,
        "pilot artifact registry",
    )
    if registry_digest != PILOT_REGISTRY_RAW_SHA256:
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot artifact registry raw hash differs",
        )
    if registry_raw != _anchor_git_blob(
        root, PILOT_ANCHOR_COMMIT, PILOT_REGISTRY_PATH
    ) or registry_raw != _anchor_git_blob(root, activation_commit, PILOT_REGISTRY_PATH):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot registry bytes differ across A, E, and the working tree",
        )
    expected_keys = {
        "protocol",
        "scientific_identity",
        "canonical_paths",
        "dataset_manifest_payload_sha256",
        "pilot_report_payload_sha256",
        "pilot_report_sha256",
        "pilot_replay_comparison_payload_sha256",
        "pilot_replay_comparison_sha256",
        "independent_verification_payload_sha256",
        "independent_verification_sha256",
        "artifact_files",
        "artifact_file_count",
        "artifact_files_payload_sha256",
        "activation_allowlist",
        "claim_boundary",
        "payload_sha256",
    }
    _expect_keys(registry, expected_keys, "pilot artifact registry")
    identity = _validate_scientific_identity(
        registry["scientific_identity"],
        "pilot artifact registry scientific_identity",
    )
    if (
        registry["protocol"] != PILOT_ARTIFACT_REGISTRY_PROTOCOL
        or registry["canonical_paths"] != PILOT_CANONICAL_PATHS
        or registry["activation_allowlist"] != list(PILOT_ACTIVATION_ALLOWLIST)
        or registry["claim_boundary"] != PILOT_REGISTRY_CLAIM
        or identity["scientific_commit"] != PILOT_SCIENTIFIC_COMMIT
        or activation_commit == PILOT_SCIENTIFIC_COMMIT
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot artifact registry activation binding differs",
        )
    if set(identity["scientific_path_sha256"]) != set(PILOT_SCIENTIFIC_PATHS):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot registry scientific path set differs",
        )
    for relative in PILOT_SCIENTIFIC_PATHS:
        expected = hashlib.sha256(
            _anchor_git_blob(root, PILOT_SCIENTIFIC_COMMIT, relative)
        ).hexdigest()
        if identity["scientific_path_sha256"].get(relative) != expected:
            raise EvidenceError(
                "pilot_anchor_invalid",
                f"pilot registry scientific hash differs: {relative}",
            )
    activation_identity = {
        "scientific_commit": activation_commit,
        "scientific_path_sha256": {
            relative: hashlib.sha256(
                _anchor_git_blob(root, activation_commit, relative)
            ).hexdigest()
            for relative in ACW_SCIENTIFIC_PATHS
        },
    }
    artifact_files = _object(
        registry["artifact_files"],
        "pilot artifact registry artifact_files",
    )
    if (
        len(artifact_files) != PILOT_ANCHORED_FILES
        or registry["artifact_file_count"] != PILOT_ANCHORED_FILES
        or registry["artifact_files_payload_sha256"]
        != hashlib.sha256(canonical_json_bytes(artifact_files)).hexdigest()
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot anchored artifact registry differs",
        )
    prefixes = tuple(f"{value}/" for value in PILOT_CANONICAL_PATHS.values())
    for relative, record in artifact_files.items():
        if not isinstance(relative, str) or not relative.startswith(prefixes):
            raise EvidenceError(
                "pilot_anchor_invalid",
                "pilot registry contains an out-of-tree artifact",
            )
        _anchor_artifact_record(record, f"pilot_registry[{relative!r}]")

    pilot_root = PILOT_CANONICAL_PATHS["pilot"]
    bundle_sources = {
        "pilot/report.json": f"{pilot_root}/report.json",
        "pilot/replay_comparison.json": f"{pilot_root}/replay_comparison.json",
        "pilot/cgb_schedule.jsonl": f"{pilot_root}/cgb_schedule.jsonl",
        "pilot/uniform_schedule.jsonl": f"{pilot_root}/uniform_schedule.jsonl",
    }
    bundle_paths = {
        bundle_relative: _anchor_artifact_path(
            root,
            source_relative,
            artifact_files[source_relative],
        )
        for bundle_relative, source_relative in bundle_sources.items()
    }
    report, report_raw, report_digest = _load_anchor_json(
        bundle_paths["pilot/report.json"],
        "anchored pilot report",
    )
    comparison, comparison_raw, comparison_digest = _load_anchor_json(
        bundle_paths["pilot/replay_comparison.json"],
        "anchored pilot comparison",
    )
    if (
        report.get("protocol") != PILOT_PROTOCOL
        or comparison.get("protocol") != PILOT_COMPARISON_PROTOCOL
        or report.get("scientific_identity") != identity
        or comparison.get("scientific_identity") != identity
        or registry["pilot_report_payload_sha256"] != report.get("payload_sha256")
        or registry["pilot_report_sha256"] != report_digest
        or registry["pilot_replay_comparison_payload_sha256"]
        != comparison.get("payload_sha256")
        or registry["pilot_replay_comparison_sha256"] != comparison_digest
        or hashlib.sha256(report_raw).hexdigest() != report_digest
        or hashlib.sha256(comparison_raw).hexdigest() != comparison_digest
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot registry differs from its frozen report or comparison",
        )

    actual = set()
    for prefix in PILOT_CANONICAL_PATHS.values():
        tree = root / prefix
        if not tree.is_dir() or tree.is_symlink():
            raise EvidenceError(
                "pilot_anchor_invalid",
                f"pilot artifact tree is invalid: {prefix}",
            )
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise EvidenceError(
                    "pilot_anchor_invalid",
                    f"pilot artifact tree contains a symlink: {path}",
                )
            if path.is_file():
                actual.add(str(path.relative_to(root)))
            elif not path.is_dir():
                raise EvidenceError(
                    "pilot_anchor_invalid",
                    f"pilot artifact tree contains a special file: {path}",
                )
    if actual != set(artifact_files):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "pilot artifact tree differs from its exact registry",
        )
    for relative, record in artifact_files.items():
        _anchor_artifact_path(root, relative, record)

    verification_relative = f"{PILOT_CANONICAL_PATHS['verification']}/verification.json"
    receipt, receipt_raw, receipt_digest = _load_anchor_json(
        root / verification_relative,
        "independent pilot verification",
    )
    producer_files = {
        relative: record
        for relative, record in artifact_files.items()
        if relative != verification_relative
    }
    producer = receipt.get("producer")
    verifier_start = receipt.get("verifier_slurm_snapshot_start")
    verifier_finish = receipt.get("verifier_slurm_snapshot_finish")
    if not all(
        isinstance(value, dict) for value in (producer, verifier_start, verifier_finish)
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "independent pilot verification identity is invalid",
        )
    producer_snapshot = producer.get("slurm_snapshot")
    producer_allocation = (
        producer_snapshot.get("allocation")
        if isinstance(producer_snapshot, dict)
        else None
    )
    verifier_start_allocation = verifier_start.get("allocation")
    verifier_finish_allocation = verifier_finish.get("allocation")
    if not all(
        isinstance(value, dict)
        for value in (
            producer_allocation,
            verifier_start_allocation,
            verifier_finish_allocation,
        )
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "independent pilot verification allocation is invalid",
        )
    producer_job = str(producer_allocation.get("job_id", ""))
    verifier_job = str(verifier_start_allocation.get("job_id", ""))
    producer_node = str(producer_allocation.get("node_list", ""))
    verifier_node = str(verifier_start_allocation.get("node_list", ""))
    if (
        receipt.get("protocol") != PILOT_INDEPENDENT_VERIFICATION_PROTOCOL
        or receipt.get("claim_boundary") != PILOT_INDEPENDENT_VERIFICATION_CLAIM
        or registry["independent_verification_payload_sha256"]
        != receipt.get("payload_sha256")
        or registry["independent_verification_sha256"] != receipt_digest
        or hashlib.sha256(receipt_raw).hexdigest() != receipt_digest
        or receipt.get("artifact_files") != producer_files
        or receipt.get("artifact_file_count") != len(producer_files) == 80
        or receipt.get("artifact_files_payload_sha256")
        != hashlib.sha256(canonical_json_bytes(producer_files)).hexdigest()
        or receipt.get("fresh_recomputation_complete") is not True
        or receipt.get("scientific_identity") != identity
        or receipt.get("dataset_manifest_payload_sha256")
        != registry["dataset_manifest_payload_sha256"]
        or receipt.get("pilot_report_payload_sha256") != report.get("payload_sha256")
        or receipt.get("pilot_report_sha256") != report_digest
        or producer.get("comparison_payload_sha256") != comparison.get("payload_sha256")
        or verifier_start_allocation != verifier_finish_allocation
        or not producer_job
        or not verifier_job
        or producer_job == verifier_job
        or not producer_node
        or not verifier_node
        or producer_node == verifier_node
        or str(producer.get("hostname", "")).split(".", 1)[0] != producer_node
        or str(receipt.get("hostname", "")).split(".", 1)[0] != verifier_node
    ):
        raise EvidenceError(
            "pilot_anchor_invalid",
            "independent pilot verification differs from the anchor",
        )
    return {
        "activation_commit": activation_commit,
        "anchor_commit": PILOT_ANCHOR_COMMIT,
        "scientific_identity": identity,
        "activation_scientific_identity": activation_identity,
        "registry_raw_sha256": PILOT_REGISTRY_RAW_SHA256,
        "artifact_files": artifact_files,
        "bundle_sources": bundle_sources,
    }


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


def _verify_binary_reference(
    value: Any,
    label: str,
    base: Path,
) -> tuple[dict[str, str | int], Path, bytes]:
    raw_path, expected_hash = _reference(value, label)
    resolved = _resolve_artifact_path(raw_path, base, label)
    raw, actual_hash = _read_binary_regular_file(resolved, label)
    if actual_hash != expected_hash:
        raise EvidenceError(
            "artifact_hash_mismatch",
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
    return (
        {"path": raw_path, "sha256": actual_hash, "bytes": len(raw)},
        resolved,
        raw,
    )


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
    raw, observed_hash = _read_binary_regular_file(path, label)
    if len(raw) != expected_bytes or observed_hash != recorded_hash:
        raise EvidenceError(
            "array_artifact_mismatch",
            f"{label} bytes or SHA-256 differ from its manifest record",
        )
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
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
        raise EvidenceError(
            "confirmation_not_authorized",
            f"{label} requires a future commit-bound beacon schema",
        )
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
    seed_fingerprint = _hash(manifest["seed_fingerprint"], f"{label}.seed_fingerprint")
    if identity_key[0] == "development":
        from pipeline.generate_acw_hidden_basis import development_seed_material

        registered_seed = DEVELOPMENT_SEEDS[identity_key[1]]
        expected_fingerprint = hashlib.sha256(
            development_seed_material(registered_seed)
        ).hexdigest()
        if seed_fingerprint != expected_fingerprint:
            raise EvidenceError(
                "dataset_seed_fingerprint_mismatch",
                f"{label} fingerprint is not derived from its registered seed",
            )
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


def _freeze_private_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise EvidenceError(
                "private_replay_tree_mismatch",
                f"private regeneration contains symlink {path}",
            )
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise EvidenceError(
                "private_replay_tree_mismatch",
                f"private regeneration contains special entry {path}",
            )
    root.chmod(0o555)


def _tree_bytes(
    root: Path, label: str
) -> tuple[dict[str, tuple[bytes, int]], dict[str, int]]:
    """Read one regular, symlink-free artifact tree through stable file opens."""

    if not root.is_dir() or root.is_symlink():
        raise EvidenceError(
            "private_replay_tree_mismatch", f"{label} is not a directory"
        )
    files: dict[str, tuple[bytes, int]] = {}
    directories: dict[str, int] = {".": stat.S_IMODE(root.stat().st_mode)}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise EvidenceError(
                "private_replay_tree_mismatch", f"{label} contains symlink {relative}"
            )
        if path.is_dir():
            directories[relative] = stat.S_IMODE(path.stat().st_mode)
            continue
        if not path.is_file():
            raise EvidenceError(
                "private_replay_tree_mismatch",
                f"{label} contains non-regular entry {relative}",
            )
        raw, _ = _read_binary_regular_file(path, f"{label}.{relative}")
        metadata = path.stat()
        if metadata.st_nlink != 1:
            raise EvidenceError(
                "private_replay_tree_mismatch",
                f"{label} contains hard-linked file {relative}",
            )
        files[relative] = (raw, stat.S_IMODE(metadata.st_mode))
    return files, directories


def _require_byte_identical_tree(
    submitted: Path,
    regenerated: Path,
    label: str,
) -> dict[str, Any]:
    """Require complete byte equality against a verifier-private regeneration."""

    submitted_files, submitted_directories = _tree_bytes(
        submitted, f"{label}.submitted"
    )
    regenerated_files, regenerated_directories = _tree_bytes(
        regenerated, f"{label}.regenerated"
    )
    if (
        set(submitted_files) != set(regenerated_files)
        or submitted_directories != regenerated_directories
    ):
        raise EvidenceError(
            "private_replay_tree_mismatch",
            f"{label} submitted tree registry differs from private regeneration",
        )
    mismatches = [
        relative
        for relative in sorted(submitted_files)
        if submitted_files[relative] != regenerated_files[relative]
    ]
    if mismatches:
        raise EvidenceError(
            "private_replay_byte_mismatch",
            f"{label} differs from private regeneration at {mismatches[0]}",
        )
    digest = hashlib.sha256()
    for relative, (raw, mode) in sorted(regenerated_files.items()):
        record = canonical_json_bytes(
            {
                "path": relative,
                "bytes": len(raw),
                "mode": f"{mode:04o}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return {
        "files": len(regenerated_files),
        "directories": len(regenerated_directories),
        "tree_sha256": digest.hexdigest(),
        "byte_identical": True,
    }


def _regenerate_private_development_dataset(
    destination: Path,
    *,
    identity_key: tuple[str, int],
    submitted_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Regenerate a registered public domain without consuming submitted bytes."""

    if identity_key[0] != "development":
        raise EvidenceError(
            "private_replay_identity_mismatch",
            f"{label} is not a registered development identity",
        )
    from pipeline.generate_acw_hidden_basis import (
        development_seed_material,
        generate_dataset,
    )

    seed = DEVELOPMENT_SEEDS[identity_key[1]]
    seed_identity = {"kind": "development", "seed": seed}
    manifest = generate_dataset(
        destination,
        development_seed_material(seed),
        seed_identity=seed_identity,
    )
    _freeze_private_tree(destination)
    tree = _require_byte_identical_tree(
        submitted_root,
        destination,
        f"{label}.dataset",
    )
    regenerated_identity, summary = _validate_dataset_manifest(
        manifest, f"{label}.private_dataset.manifest"
    )
    if regenerated_identity != identity_key:
        raise EvidenceError(
            "private_replay_identity_mismatch",
            f"{label} private dataset identity differs",
        )
    _validate_dataset_tree(destination, manifest, f"{label}.private_dataset")
    return manifest, summary, tree


def _regenerate_private_trainer_bundle(
    destination: Path,
    *,
    private_dataset_root: Path,
    private_dataset_manifest: dict[str, Any],
    private_dataset_summary: dict[str, Any],
    submitted_root: Path,
    schedule_kind: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild one trainer bundle from private truth and anchored schedule bytes."""

    from pipeline.acw_hidden_basis_training import load_committed_pilot_anchor
    from pipeline.freeze_acw_curriculum import build_trainer_bundle

    anchor = load_committed_pilot_anchor(verify_all_artifacts=True)
    schedule_key = f"pilot/{schedule_kind}"
    schedule_path = Path(anchor["bundle_paths"][schedule_key])
    pilot_report_path = Path(anchor["bundle_paths"]["pilot/report.json"])
    manifest = build_trainer_bundle(
        private_dataset_root,
        schedule_path,
        destination,
        canonical=True,
        pilot_report_path=pilot_report_path,
    )
    _freeze_private_tree(destination)
    tree = _require_byte_identical_tree(
        submitted_root,
        destination,
        f"{label}.trainer_bundle",
    )
    summary = _validate_trainer_bundle(
        destination,
        manifest,
        private_dataset_manifest,
        private_dataset_summary,
        f"{label}.private_trainer_bundle.manifest",
        dataset_root=private_dataset_root,
    )
    return manifest, summary, tree


def _validate_curriculum(
    root: Path,
    files: dict[str, Any],
    initial_queries: np.ndarray,
    initial_answers: np.ndarray,
    label: str,
    *,
    oracle_answers: np.ndarray | None = None,
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
    if oracle_answers is not None and oracle_answers.shape != (4096, 24):
        raise EvidenceError(
            "bundle_curriculum_mismatch",
            f"{label} oracle answer bank has the wrong shape",
        )
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
        if oracle_answers is not None and answer != int(
            oracle_answers[history_id, query_id]
        ):
            raise EvidenceError(
                "bundle_curriculum_answer_mismatch",
                f"{label} curriculum answer differs from replayed oracle truth",
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
    *,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    anchor = _load_adjudicator_pilot_anchor()
    summary = _validate_unanchored_trainer_bundle_structure(
        root,
        manifest,
        dataset_manifest,
        dataset_summary,
        label,
        dataset_root=dataset_root,
    )
    if summary["pilot_scientific_identity"] != anchor["scientific_identity"]:
        raise EvidenceError(
            "pilot_anchor_invalid",
            f"{label} pilot identity differs from anchored S",
        )
    pilot_artifacts = _object(manifest["pilot_artifacts"], f"{label}.pilot_artifacts")
    for bundle_relative, source_relative in anchor["bundle_sources"].items():
        if pilot_artifacts.get(bundle_relative) != anchor["artifact_files"].get(
            source_relative
        ):
            raise EvidenceError(
                "pilot_anchor_invalid",
                f"{label} artifact differs from pilot anchor: {bundle_relative}",
            )
    return {
        **summary,
        "pilot_anchor_commit": anchor["anchor_commit"],
        "pilot_registry_raw_sha256": anchor["registry_raw_sha256"],
        "activation_commit": anchor["activation_commit"],
        "activation_scientific_identity": anchor["activation_scientific_identity"],
    }


def _validate_unanchored_trainer_bundle_structure(
    root: Path,
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_summary: dict[str, Any],
    label: str,
    *,
    dataset_root: Path | None = None,
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
    oracle_answers = None
    if dataset_root is not None:
        oracle_record = _object(
            dataset_manifest["arrays"].get("oracle/train/public_answers.npy"),
            f"{label}.source_oracle_answers",
        )
        oracle_answers = np.asarray(
            _load_bound_array(
                dataset_root,
                "oracle/train/public_answers.npy",
                oracle_record,
                f"{label}.source_oracle_answers",
                expected_shape=(4096, 24),
                expected_dtype="int8",
            )
        )
    curriculum_hash, derived_schedule_hash = _validate_curriculum(
        root,
        _object(manifest["files"], f"{label}.files"),
        initial_queries,
        initial_answers,
        label,
        oracle_answers=oracle_answers,
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
    raise EvidenceError(
        "confirmation_not_authorized",
        "confirmation optimizer derivation requires a future commit-bound beacon schema",
    )


def _validate_checkpoint_artifact(
    path: Path,
    file_sha256: str,
    *,
    logical_arm: str,
    dataset_summary: dict[str, Any],
    bundle_summary: dict[str, Any],
    label: str,
    checkpoint_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Load a real checkpoint and verify its model and artifact bindings."""

    try:
        if checkpoint_bytes is None:
            checkpoint_bytes, observed = _read_binary_regular_file(path, label)
            if observed != file_sha256:
                raise EvidenceError(
                    "artifact_hash_mismatch",
                    f"{label} changed before checkpoint deserialization",
                )
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except EvidenceError:
        raise
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
            "canonical_runtime_sha256": training_report.get("canonical_runtime_sha256"),
            "development_plan_sha256": training_report.get("development_plan_sha256"),
            "execution_receipt": training_report.get("execution_receipt"),
        },
        arm=logical_arm,
        label=f"{label}.training_evidence",
    )
    if (
        training_evidence["execution_receipt"]["scientific_commit"]
        != scientific_identity["scientific_commit"]
    ):
        raise EvidenceError(
            "training_execution_commit_mismatch",
            f"{label} execution receipt and checkpoint identity differ",
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


def _write_snapshot_file(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating evidence snapshot")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot_dataset_tree(
    source_root: Path,
    manifest: dict[str, Any],
    destination: Path,
) -> None:
    destination.mkdir(mode=0o700)
    _write_snapshot_file(
        destination / "manifest.json",
        canonical_json_bytes(manifest) + b"\n",
    )
    arrays = _object(manifest.get("arrays"), "snapshot dataset arrays")
    for relative, raw_record in arrays.items():
        if not isinstance(relative, str):
            raise EvidenceError(
                "dataset_snapshot_failed", "dataset array path is not a string"
            )
        record = _object(raw_record, f"snapshot dataset arrays[{relative!r}]")
        _expect_keys(
            record,
            ARRAY_RECORD_KEYS,
            f"snapshot dataset arrays[{relative!r}]",
        )
        expected_bytes = _integer(
            record["bytes"],
            f"snapshot dataset arrays[{relative!r}].bytes",
            minimum=1,
        )
        expected_hash = _hash(
            record["sha256"],
            f"snapshot dataset arrays[{relative!r}].sha256",
        )
        source = _safe_child(source_root, relative, f"snapshot dataset {relative}")
        raw, digest = _read_binary_regular_file(source, f"snapshot dataset {relative}")
        if len(raw) != expected_bytes or digest != expected_hash:
            raise EvidenceError(
                "dataset_snapshot_failed",
                f"dataset array changed before snapshot: {relative}",
            )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvidenceError(
                "dataset_snapshot_failed", f"unsafe dataset array path: {relative}"
            )
        _write_snapshot_file(destination / relative_path, raw)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    destination.chmod(0o555)


def _independent_evaluator_replay(
    checkpoint_path: Path,
    dataset_root: Path,
    expected_bytes: bytes,
    label: str,
    *,
    checkpoint_bytes: bytes | None = None,
    dataset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the frozen evaluator in a new interpreter and compare raw bytes."""

    repository = Path(__file__).resolve().parents[1]
    evaluator = repository / "pipeline" / "evaluate_acw_hidden_basis.py"
    if not evaluator.is_file():
        raise EvidenceError(
            "independent_evaluator_missing", "frozen evaluator is absent"
        )
    with tempfile.TemporaryDirectory(prefix="acw-adjudicator-replay-") as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / "evaluation.json"
        checkpoint_snapshot = temporary_root / "checkpoint.pt"
        dataset_snapshot = temporary_root / "dataset"
        if checkpoint_bytes is None:
            checkpoint_bytes, _ = _read_binary_regular_file(
                checkpoint_path, f"{label}.checkpoint replay snapshot"
            )
        if dataset_manifest is None:
            manifest_raw, _ = _read_regular_file(dataset_root / "manifest.json")
            dataset_manifest = _parse_json(
                manifest_raw, f"{label}.dataset replay manifest"
            )
        _write_snapshot_file(checkpoint_snapshot, checkpoint_bytes)
        _snapshot_dataset_tree(dataset_root, dataset_manifest, dataset_snapshot)
        bootstrap = (
            "import os,runpy,sys,sysconfig;"
            "repo,script=sys.argv[1:3];del sys.argv[1:3];"
            "marker=os.path.join(repo,'pipeline','__init__.py');"
            "assert not os.path.lexists(marker),marker;"
            "paths=[repo,sysconfig.get_paths()['purelib'],"
            "sysconfig.get_paths()['platlib']];"
            "sys.path[:0]=list(dict.fromkeys(paths));"
            "runpy.run_path(script,run_name='__main__')"
        )
        environment = _canonical_development_subprocess_environment()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    "-S",
                    "-c",
                    bootstrap,
                    str(repository),
                    str(evaluator),
                    "--checkpoint",
                    str(checkpoint_snapshot),
                    "--dataset",
                    str(dataset_snapshot),
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


def _canonical_development_subprocess_environment() -> dict[str, str]:
    from pipeline.freeze_acw_curriculum import CANONICAL_PILOT_STATIC_ENV

    dynamic_keys = (
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_NODELIST",
        "SLURM_SUBMIT_DIR",
    )
    missing = [key for key in dynamic_keys if not os.environ.get(key)]
    if missing:
        raise EvidenceError(
            "training_runtime_mismatch",
            f"canonical development Slurm environment is incomplete: {missing}",
        )
    environment = dict(CANONICAL_PILOT_STATIC_ENV)
    environment.update({key: str(os.environ[key]) for key in dynamic_keys})
    return environment


def _tensor_state_sha256(state: dict[str, torch.Tensor], label: str) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise EvidenceError(
                "trainer_replay_mismatch", f"{label} contains a non-tensor entry"
            )
        value = tensor.detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_semantic_fingerprint(raw: bytes, label: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise EvidenceError(
            "trainer_replay_mismatch",
            f"{label} cannot be loaded: {type(exc).__name__}: {exc}",
        ) from exc
    checkpoint = _object(checkpoint, label)
    _expect_keys(checkpoint, CHECKPOINT_KEYS, label)
    report = dict(_object(checkpoint["training_report"], f"{label}.training_report"))
    report.pop("wall_seconds", None)
    report.pop("resource_measurements", None)
    report.pop("execution_receipt", None)
    snapshots = checkpoint["label_efficiency_models"]
    if snapshots is None:
        snapshot_hashes = None
    else:
        snapshot_hashes = [
            _tensor_state_sha256(
                _object(state, f"{label}.label_efficiency_models[{index}]"),
                f"{label}.label_efficiency_models[{index}]",
            )
            for index, state in enumerate(_list(snapshots, f"{label}.snapshots"))
        ]
    return {
        "protocol": checkpoint["protocol"],
        "arm": checkpoint["arm"],
        "seed": checkpoint["seed"],
        "dataset_manifest_payload_sha256": checkpoint[
            "dataset_manifest_payload_sha256"
        ],
        "source_manifest_payload_sha256": checkpoint["source_manifest_payload_sha256"],
        "curriculum_sha256": checkpoint["curriculum_sha256"],
        "query_schedule_sha256": checkpoint["query_schedule_sha256"],
        "query_schedule_kind": checkpoint["query_schedule_kind"],
        "pilot_report_payload_sha256": checkpoint["pilot_report_payload_sha256"],
        "parameters": checkpoint["parameters"],
        "training_report": report,
        "label_efficiency_model_sha256": snapshot_hashes,
        "scientific_identity": checkpoint["scientific_identity"],
        "model_tensor_sha256": _tensor_state_sha256(
            _object(checkpoint["model"], f"{label}.model"), f"{label}.model"
        ),
    }


def _independent_trainer_replay(
    checkpoint_bytes: bytes,
    *,
    logical_arm: str,
    dataset_root: Path,
    bundle_root: Path,
    optimizer_seed: int,
    dataset_summary: dict[str, Any],
    bundle_summary: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Refit independently and require the submitted semantic checkpoint exactly."""

    repository = Path(__file__).resolve().parents[1]
    trainer = repository / "pipeline" / "acw_hidden_basis_training.py"
    trainer_arm = "acw" if logical_arm == "uniform_query_acw" else logical_arm
    try:
        development_index = DEVELOPMENT_SEEDS.index(optimizer_seed)
    except ValueError as exc:
        raise EvidenceError(
            "trainer_replay_mismatch", f"{label} optimizer seed is not registered"
        ) from exc
    private_name = (
        "acw_development_g1_direct_verifier"
        if logical_arm == DIRECT_STATE_ARM
        else "acw_development_g1_final_verifier"
    )
    persistent = (
        Path("/lustre/fs1/home/sa305415/shohin_acw/artifacts/r12")
        / private_name
        / "runs"
        / f"{development_index:02d}_{logical_arm}"
        / "checkpoint.pt"
    )

    if persistent.is_file() and not persistent.is_symlink():
        output = persistent
        replay_bytes, replay_sha256 = _read_binary_regular_file(
            output, f"{label}.trainer_replay"
        )
        replay_source = "dedicated_verifier_job"
    else:
        with tempfile.TemporaryDirectory(
            prefix="acw-adjudicator-trainer-"
        ) as temporary:
            output = Path(temporary) / "checkpoint.pt"
            command = [
                sys.executable,
                "-P",
                "-S",
                str(trainer),
                "--bundle",
                str(bundle_root),
                "--curriculum",
                str(bundle_root / "curriculum.jsonl"),
                "--arm",
                trainer_arm,
                "--seed",
                str(optimizer_seed),
                "--attempt-id",
                f"{logical_arm}__{optimizer_seed}",
                "--verification-replay",
                "--out",
                str(output),
            ]
            if logical_arm == DIRECT_STATE_ARM:
                command.extend(("--oracle-dataset", str(dataset_root)))
            try:
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    env=_canonical_development_subprocess_environment(),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=7_200,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise EvidenceError(
                    "trainer_replay_failed",
                    f"{label} canonical refit failed: {type(exc).__name__}: {exc}",
                ) from exc
            if completed.returncode != 0 or not output.is_file():
                detail = (
                    completed.stderr or completed.stdout or "no diagnostic"
                ).strip()
                raise EvidenceError(
                    "trainer_replay_failed",
                    f"{label} canonical refit exit {completed.returncode}: {detail[-2000:]}",
                )
            replay_bytes, replay_sha256 = _read_binary_regular_file(
                output, f"{label}.trainer_replay"
            )
            replay_source = "ephemeral_verifier_refit"
            _validate_checkpoint_artifact(
                output,
                replay_sha256,
                logical_arm=logical_arm,
                dataset_summary=dataset_summary,
                bundle_summary=bundle_summary,
                label=f"{label}.trainer_replay",
                checkpoint_bytes=replay_bytes,
            )
    if replay_source == "dedicated_verifier_job":
        _validate_checkpoint_artifact(
            output,
            replay_sha256,
            logical_arm=logical_arm,
            dataset_summary=dataset_summary,
            bundle_summary=bundle_summary,
            label=f"{label}.trainer_replay",
            checkpoint_bytes=replay_bytes,
        )
    submitted = _checkpoint_semantic_fingerprint(checkpoint_bytes, f"{label}.submitted")
    replayed = _checkpoint_semantic_fingerprint(replay_bytes, f"{label}.trainer_replay")
    if submitted != replayed:
        raise EvidenceError(
            "trainer_replay_mismatch",
            f"{label} checkpoint is not the exact canonical fit",
        )
    return {
        "semantic_fingerprint_sha256": hashlib.sha256(
            canonical_json_bytes(submitted)
        ).hexdigest(),
        "replay_checkpoint_sha256": replay_sha256,
        "semantic_match": True,
        "canonical_runtime_sha256": CANONICAL_DEVELOPMENT_RUNTIME_SHA256,
        "source": replay_source,
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


def _validate_development_execution_receipt(
    value: Any, *, label: str
) -> dict[str, Any]:
    receipt = _object(value, label)
    _expect_keys(
        receipt,
        {
            "schema",
            "protocol",
            "scientific_commit",
            "canonical_runtime_sha256",
            "development_plan_sha256",
            "environment_sha256",
            "batch_script_sha256",
            "slurm",
            "process_membership",
            "role",
            "attempt_id",
            "verification_replay",
        },
        label,
    )
    scientific_commit = receipt["scientific_commit"]
    runtime_sha256 = _hash(
        receipt["canonical_runtime_sha256"], f"{label}.canonical_runtime_sha256"
    )
    plan_sha256 = _hash(
        receipt["development_plan_sha256"], f"{label}.development_plan_sha256"
    )
    environment_sha256 = _hash(
        receipt["environment_sha256"], f"{label}.environment_sha256"
    )
    batch_script_sha256 = _hash(
        receipt["batch_script_sha256"], f"{label}.batch_script_sha256"
    )
    slurm = _object(receipt["slurm"], f"{label}.slurm")
    _expect_keys(
        slurm,
        {"job_id", "job_name", "node_list", "cpus_per_task"},
        f"{label}.slurm",
    )
    membership = _object(receipt["process_membership"], f"{label}.process_membership")
    _expect_keys(
        membership,
        {"cpu_list", "memory_list", "task_cgroup"},
        f"{label}.process_membership",
    )
    repository = Path(__file__).resolve().parents[1]
    plan_path = repository / DEVELOPMENT_PLAN_PATH
    plan_raw, observed_plan_sha256 = _read_regular_file(plan_path)
    plan = _parse_json(plan_raw, "committed development plan")
    job_id = str(slurm["job_id"])
    stages = plan.get("custody_stages")
    stage_matches = (
        [
            stage
            for stage in stages
            if isinstance(stage, dict) and str(stage.get("held_slurm_job_id")) == job_id
        ]
        if isinstance(stages, list)
        else []
    )
    stage = stage_matches[0] if len(stage_matches) == 1 else None
    role = receipt["role"]
    attempt_id = receipt["attempt_id"]
    attempts = plan.get("attempt_table")
    attempt_matches = (
        [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("attempt_id") == attempt_id
        ]
        if isinstance(attempts, list)
        else []
    )
    attempt = attempt_matches[0] if len(attempt_matches) == 1 else None
    side_roles = (
        {
            side.get("job_role")
            for side in (attempt.get("producer"), attempt.get("verifier"))
            if isinstance(side, dict)
        }
        if isinstance(attempt, dict)
        else set()
    )
    from pipeline.freeze_acw_curriculum import (
        CANONICAL_PILOT_STATIC_ENV,
        CANONICAL_PILOT_UID,
        _cpu_list_members,
    )

    expected_environment = dict(CANONICAL_PILOT_STATIC_ENV)
    expected_environment.update(
        {
            "SLURM_CPUS_PER_TASK": str(slurm["cpus_per_task"]),
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NAME": str(slurm["job_name"]),
            "SLURM_JOB_NODELIST": str(slurm["node_list"]),
            "SLURM_NODELIST": str(slurm["node_list"]),
            "SLURM_SUBMIT_DIR": "/lustre/fs1/home/sa305415/shohin_acw",
        }
    )
    try:
        cpu_members = _cpu_list_members(str(membership["cpu_list"]))
    except (TypeError, ValueError) as exc:
        raise EvidenceError(
            "training_runtime_mismatch", f"{label} CPU membership is malformed"
        ) from exc
    if (
        receipt["schema"] != "r12_acw_development_execution_receipt_v1"
        or receipt["protocol"] != "R12-ACW-DEVELOPMENT-EXECUTION-v1"
        or not isinstance(scientific_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", scientific_commit) is None
        or runtime_sha256 != CANONICAL_DEVELOPMENT_RUNTIME_SHA256
        or plan_sha256 != DEVELOPMENT_PLAN_RAW_SHA256
        or observed_plan_sha256 != DEVELOPMENT_PLAN_RAW_SHA256
        or stage is None
        or role != stage.get("role")
        or role not in side_roles
        or attempt is None
        or not isinstance(attempt_id, str)
        or slurm["job_name"] != stage.get("job_name")
        or slurm["node_list"] != stage.get("expected_node")
        or not isinstance(slurm["node_list"], str)
        or not slurm["node_list"]
        or slurm["cpus_per_task"] != "4"
        or environment_sha256
        != hashlib.sha256(canonical_json_bytes(expected_environment)).hexdigest()
        or batch_script_sha256
        != sha256_file(repository / str(stage.get("script", {}).get("path")))
        or len(cpu_members) != 4
        or not str(membership["memory_list"])
        or membership["task_cgroup"]
        != (f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{job_id}/step_batch/task_0")
    ):
        raise EvidenceError(
            "training_runtime_mismatch",
            f"{label} execution receipt differs from its top-level batch cgroup or held job",
        )
    return {
        "schema": receipt["schema"],
        "protocol": receipt["protocol"],
        "scientific_commit": scientific_commit,
        "canonical_runtime_sha256": runtime_sha256,
        "development_plan_sha256": plan_sha256,
        "environment_sha256": environment_sha256,
        "batch_script_sha256": batch_script_sha256,
        "slurm": dict(slurm),
        "process_membership": dict(membership),
        "role": role,
        "attempt_id": attempt_id,
        "verification_replay": _boolean(
            receipt["verification_replay"], f"{label}.verification_replay"
        ),
    }


def _validate_attempt_receipt_reference(
    value: Any,
    *,
    base: Path,
    arm: str,
    index: int,
    run: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    receipt, binding, path = _verify_json_reference(value, label, base)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "attempt_receipt_mutable", f"{label} must be a mode-0444 regular file"
        )
    _expect_keys(
        receipt,
        {
            "schema",
            "protocol",
            "attempt_id",
            "role",
            "logical_arm",
            "trainer_arm",
            "seed",
            "development_plan_sha256",
            "artifact_root",
            "task_root",
            "slurm",
            "outputs",
            "completed_once",
            "confirmation_authorized",
            "payload_sha256",
        },
        label,
    )
    payload_sha256 = _verify_payload_hash(receipt, label)
    attempt_id = f"{arm}__{DEVELOPMENT_SEEDS[index]}"
    expected_role = "phase1_producer" if arm == DIRECT_STATE_ARM else "phase2_producer"
    expected_trainer_arm = "acw" if arm == "uniform_query_acw" else arm
    expected_task = f"runs/{index:02d}_{arm}"
    if path != base / expected_task / "attempt.json":
        raise EvidenceError(
            "attempt_receipt_path_mismatch", f"{label} is outside its claimed task"
        )
    outputs = _object(receipt["outputs"], f"{label}.outputs")
    _expect_keys(outputs, {"checkpoint", "evaluation", "replay"}, f"{label}.outputs")
    expected_outputs = {
        "checkpoint": run["checkpoint"],
        "evaluation": run["evaluation_report"],
        "replay": run["replay_report"],
    }
    for name, expected in expected_outputs.items():
        observed = _object(outputs[name], f"{label}.outputs.{name}")
        _expect_keys(observed, REFERENCE_KEYS, f"{label}.outputs.{name}")
        if observed != expected:
            raise EvidenceError(
                "attempt_receipt_output_mismatch",
                f"{label} does not bind the manifest {name} artifact",
            )
    slurm = _object(receipt["slurm"], f"{label}.slurm")
    _expect_keys(
        slurm,
        {
            "job_id",
            "job_name",
            "node",
            "cpus_per_task",
            "dependency",
            "script",
            "spool_script_sha256",
            "scontrol_snapshot_sha256",
            "process_membership",
        },
        f"{label}.slurm",
    )
    repository = Path(__file__).resolve().parents[1]
    plan_raw, plan_sha256 = _read_regular_file(repository / DEVELOPMENT_PLAN_PATH)
    plan = _parse_json(plan_raw, "committed development plan")
    stages = plan.get("custody_stages")
    stage_matches = (
        [
            stage
            for stage in stages
            if isinstance(stage, dict) and stage.get("role") == expected_role
        ]
        if isinstance(stages, list)
        else []
    )
    stage = stage_matches[0] if len(stage_matches) == 1 else None
    membership = _object(slurm["process_membership"], f"{label}.process_membership")
    _expect_keys(
        membership,
        {"cpu_list", "memory_list", "task_cgroup"},
        f"{label}.process_membership",
    )
    from pipeline.freeze_acw_curriculum import CANONICAL_PILOT_UID

    if (
        receipt["schema"] != "r12_acw_development_attempt_receipt_v1"
        or receipt["protocol"] != "R12-ACW-DEVELOPMENT-ATTEMPT-v1"
        or receipt["attempt_id"] != attempt_id
        or receipt["logical_arm"] != arm
        or receipt["trainer_arm"] != expected_trainer_arm
        or receipt["seed"] != DEVELOPMENT_SEEDS[index]
        or receipt["role"] != expected_role
        or receipt["development_plan_sha256"] != DEVELOPMENT_PLAN_RAW_SHA256
        or plan_sha256 != DEVELOPMENT_PLAN_RAW_SHA256
        or receipt["artifact_root"] != str(base.resolve(strict=True))
        or receipt["task_root"] != expected_task
        or receipt["completed_once"] is not True
        or receipt["confirmation_authorized"] is not False
        or stage is None
        or str(slurm["job_id"]) != str(stage.get("held_slurm_job_id"))
        or slurm["job_name"] != stage.get("job_name")
        or slurm["node"] != stage.get("expected_node")
        or slurm["cpus_per_task"] != "4"
        or slurm["script"] != stage.get("script")
        or slurm["spool_script_sha256"] != stage.get("script", {}).get("sha256")
        or membership["task_cgroup"]
        != (
            f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{stage.get('held_slurm_job_id')}"
            "/step_batch/task_0"
        )
    ):
        raise EvidenceError(
            "attempt_receipt_mismatch", f"{label} differs from the frozen attempt"
        )
    return {**binding, "payload_sha256": payload_sha256, "attempt_id": attempt_id}


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
    runtime_sha256 = _hash(
        evidence["canonical_runtime_sha256"],
        f"{label}.canonical_runtime_sha256",
    )
    plan_sha256 = _hash(
        evidence["development_plan_sha256"],
        f"{label}.development_plan_sha256",
    )
    if runtime_sha256 != CANONICAL_DEVELOPMENT_RUNTIME_SHA256:
        raise EvidenceError(
            "training_runtime_mismatch", f"{label} runtime identity differs"
        )
    if plan_sha256 != DEVELOPMENT_PLAN_RAW_SHA256:
        raise EvidenceError(
            "development_plan_mismatch", f"{label} development plan differs"
        )
    execution_receipt = _validate_development_execution_receipt(
        evidence["execution_receipt"], label=f"{label}.execution_receipt"
    )
    receipt_attempt_id = execution_receipt.get("attempt_id")
    if receipt_attempt_id is not None and receipt_attempt_id.rsplit("__", 1)[0] != arm:
        raise EvidenceError(
            "training_runtime_mismatch",
            f"{label} attempt ID does not bind arm {arm}",
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
        "canonical_runtime_sha256": runtime_sha256,
        "development_plan_sha256": plan_sha256,
        "execution_receipt": execution_receipt,
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
    if (
        training_evidence["execution_receipt"]["scientific_commit"]
        != scientific_identity["scientific_commit"]
    ):
        raise EvidenceError(
            "training_execution_commit_mismatch",
            f"{label} execution receipt and evaluation identity differ",
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


def _expected_run_keys(scope: str = "full") -> set[tuple[str, str, int]]:
    if scope not in {"direct_state", "development", "full"}:
        raise ValueError(f"unknown evidence scope: {scope}")
    if scope == "direct_state":
        return {(DIRECT_STATE_ARM, "development", index) for index in range(3)}
    splits = (
        ("development",)
        if scope == "development"
        else (
            "development",
            "confirmation",
        )
    )
    expected = {
        (arm, split, index)
        for arm in SCORED_ARMS
        for split in splits
        for index in range(3)
    }
    expected.update((DIRECT_STATE_ARM, "development", index) for index in range(3))
    return expected


def _verify_evidence_with_private_workspace(
    manifest: dict[str, Any],
    base: Path,
    *,
    private_root: Path,
    scope: str = "full",
    expected_development_baseline_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scope not in {"direct_state", "development", "full"}:
        raise ValueError(f"unknown evidence scope: {scope}")
    manifest_keys = {"schema", "protocol", "reports", "payload_sha256"}
    if scope in {"direct_state", "development"}:
        manifest_keys.update(
            {
                "development_plan",
                "attempt_claim",
                "attempt_start",
                "stage_receipts",
            }
        )
    if scope == "direct_state":
        manifest_keys.add("private_refit_verification")
    if scope == "development":
        manifest_keys.update(
            {
                "phase2_authorization",
                "direct_refit_verification",
                "private_refit_verification",
            }
        )
    if scope == "full":
        manifest_keys.add("development_baseline")
    _expect_keys(manifest, manifest_keys, "manifest")
    manifest_payload_sha256 = _verify_payload_hash(manifest, "manifest")
    expected_schema, expected_protocol = {
        "direct_state": (DIRECT_STATE_MANIFEST_SCHEMA, DIRECT_STATE_MANIFEST_PROTOCOL),
        "development": (DEVELOPMENT_MANIFEST_SCHEMA, DEVELOPMENT_MANIFEST_PROTOCOL),
        "full": (MANIFEST_SCHEMA, MANIFEST_PROTOCOL),
    }[scope]
    if manifest["schema"] != expected_schema:
        raise EvidenceError(
            "manifest_schema_mismatch", f"manifest schema must be {expected_schema}"
        )
    if manifest["protocol"] != expected_protocol:
        raise EvidenceError(
            "manifest_protocol_mismatch",
            f"manifest protocol must be {expected_protocol}",
        )
    development_baseline_binding = None
    development_plan_binding = None
    attempt_start_binding = None
    attempt_claim_binding = None
    stage_receipt_bindings = None
    private_refit_bindings = None
    phase2_authorization = None
    if scope in {"direct_state", "development"}:
        development_plan_binding = _validate_development_plan_reference(
            manifest["development_plan"], base
        )
        attempt_start_binding = _validate_attempt_start_reference(
            manifest["attempt_start"],
            base,
            expected_plan=development_plan_binding,
        )
        attempt_claim_binding = _validate_attempt_claim_reference(
            manifest["attempt_claim"],
            base,
            expected_plan=development_plan_binding,
        )
        stage_receipt_bindings = _validate_stage_receipts(
            manifest["stage_receipts"],
            base,
            scope=scope,
            expected_plan=development_plan_binding,
        )
        private_refit_bindings = {
            "direct": _validate_private_refit_verification_reference(
                (
                    manifest["private_refit_verification"]
                    if scope == "direct_state"
                    else manifest["direct_refit_verification"]
                ),
                base,
                scope="direct",
                expected_plan=development_plan_binding,
            )
        }
        if scope == "development":
            private_refit_bindings["final"] = (
                _validate_private_refit_verification_reference(
                    manifest["private_refit_verification"],
                    base,
                    scope="final",
                    expected_plan=development_plan_binding,
                )
            )
    if scope == "development":
        phase2_authorization = _validate_phase2_authorization(
            manifest["phase2_authorization"],
            base,
            expected_plan=development_plan_binding,
        )
    if scope == "full":
        if expected_development_baseline_record is None:
            raise EvidenceError(
                "development_baseline_binding_required",
                "full evidence cannot be opened without a validated baseline binding",
            )
        expected_record = _object(
            expected_development_baseline_record,
            "validated development baseline record",
        )
        _expect_keys(
            expected_record,
            {"path", "sha256", "payload_sha256"},
            "validated development baseline record",
        )
        development_baseline_binding = _object(
            manifest["development_baseline"], "manifest.development_baseline"
        )
        _expect_keys(
            development_baseline_binding,
            {"path", "sha256", "payload_sha256"},
            "manifest.development_baseline",
        )
        if (
            not isinstance(development_baseline_binding["path"], str)
            or not development_baseline_binding["path"]
            or _hash(
                development_baseline_binding["sha256"],
                "manifest.development_baseline.sha256",
            )
            != expected_record["sha256"]
            or _hash(
                development_baseline_binding["payload_sha256"],
                "manifest.development_baseline.payload_sha256",
            )
            != expected_record["payload_sha256"]
            or development_baseline_binding["path"] != expected_record["path"]
        ):
            raise EvidenceError(
                "development_baseline_binding_mismatch",
                "full manifest does not bind the independently validated baseline",
            )
    reports = _list(manifest["reports"], "manifest.reports")
    expected_keys = _expected_run_keys(scope)
    if len(reports) != len(expected_keys):
        raise EvidenceError(
            "report_count_mismatch",
            f"manifest must contain exactly {len(expected_keys)} reports, got {len(reports)}",
        )
    ordered_expected = None
    if scope == "direct_state":
        ordered_expected = [
            (DIRECT_STATE_ARM, "development", index) for index in range(3)
        ]
    elif scope == "development":
        ordered_expected = [
            (arm, "development", index)
            for arm in (DIRECT_STATE_ARM, *SCORED_ARMS)
            for index in range(3)
        ]

    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    dataset_by_identity: dict[tuple[str, int], tuple[str, str]] = {}
    identity_by_dataset: dict[str, tuple[str, int]] = {}
    dataset_cache: dict[tuple[Path, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    bundle_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    private_dataset_cache: dict[
        tuple[str, int],
        tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    private_bundle_cache: dict[
        tuple[tuple[str, int], str],
        tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
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
    run_keys = RUN_KEYS
    if scope in {"direct_state", "development"}:
        run_keys = RUN_KEYS | {"attempt_id", "attempt_receipt"}

    for position, raw_run in enumerate(reports):
        label = f"manifest.reports[{position}]"
        run = _object(raw_run, label)
        _expect_keys(run, run_keys, label)
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
        private_dataset = private_dataset_cache.get(identity_key)
        if private_dataset is None:
            private_dataset_root = private_root / f"dataset_{split}_{index}"
            (
                private_dataset_manifest,
                private_dataset_summary,
                private_dataset_tree,
            ) = _regenerate_private_development_dataset(
                private_dataset_root,
                identity_key=identity_key,
                submitted_root=dataset_root,
                label=label,
            )
            if private_dataset_summary != dataset_summary:
                raise EvidenceError(
                    "private_replay_dataset_mismatch",
                    f"{label} submitted dataset summary differs from regeneration",
                )
            private_dataset = (
                private_dataset_root,
                private_dataset_manifest,
                private_dataset_summary,
                private_dataset_tree,
            )
            private_dataset_cache[identity_key] = private_dataset
        (
            private_dataset_root,
            private_dataset_manifest,
            private_dataset_summary,
            private_dataset_tree,
        ) = private_dataset
        key = (arm, split, index)
        if ordered_expected is not None and key != ordered_expected[position]:
            raise EvidenceError(
                "attempt_order_mismatch",
                f"{label} is outside the frozen direct-first attempt order",
            )
        attempt_receipt_binding = None
        if scope in {"direct_state", "development"}:
            expected_attempt_id = f"{arm}__{DEVELOPMENT_SEEDS[index]}"
            if run["attempt_id"] != expected_attempt_id:
                raise EvidenceError(
                    "attempt_id_mismatch",
                    f"{label} attempt ID differs from its seed",
                )
            attempt_receipt_binding = _validate_attempt_receipt_reference(
                run["attempt_receipt"],
                base=base,
                arm=arm,
                index=index,
                run=run,
                label=f"{label}.attempt_receipt",
            )
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
                dataset_root=dataset_root,
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
        private_bundle_key = (identity_key, expected_schedule_kind)
        private_bundle = private_bundle_cache.get(private_bundle_key)
        if private_bundle is None:
            private_bundle_root = private_root / (
                f"bundle_{split}_{index}_{expected_schedule_kind.removesuffix('.jsonl')}"
            )
            (
                private_bundle_manifest,
                private_bundle_summary,
                private_bundle_tree,
            ) = _regenerate_private_trainer_bundle(
                private_bundle_root,
                private_dataset_root=private_dataset_root,
                private_dataset_manifest=private_dataset_manifest,
                private_dataset_summary=private_dataset_summary,
                submitted_root=bundle_root,
                schedule_kind=expected_schedule_kind,
                label=label,
            )
            if private_bundle_summary != bundle_summary:
                raise EvidenceError(
                    "private_replay_bundle_mismatch",
                    f"{label} submitted bundle summary differs from regeneration",
                )
            private_bundle = (
                private_bundle_root,
                private_bundle_manifest,
                private_bundle_summary,
                private_bundle_tree,
            )
            private_bundle_cache[private_bundle_key] = private_bundle
        (
            private_bundle_root,
            private_bundle_manifest,
            private_bundle_summary,
            private_bundle_tree,
        ) = private_bundle

        checkpoint_binding, checkpoint_path, checkpoint_bytes = (
            _verify_binary_reference(run["checkpoint"], f"{label}.checkpoint", base)
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
            checkpoint_bytes=checkpoint_bytes,
        )
        trainer_replay = _independent_trainer_replay(
            checkpoint_bytes,
            logical_arm=arm,
            dataset_root=private_dataset_root,
            bundle_root=private_bundle_root,
            optimizer_seed=_expected_optimizer_seed(
                private_dataset_summary["seed_identity"]
            ),
            dataset_summary=private_dataset_summary,
            bundle_summary=private_bundle_summary,
            label=label,
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
            private_dataset_root,
            evaluation_bytes,
            label,
            checkpoint_bytes=checkpoint_bytes,
            dataset_manifest=private_dataset_manifest,
        )
        if independent_replay["payload_sha256"] != parsed["payload_sha256"]:
            raise EvidenceError(
                "independent_evaluator_mismatch",
                f"{label} fresh evaluator payload hash differs",
            )

        current_identity = parsed["scientific_identity"]
        if current_identity != bundle_summary["activation_scientific_identity"]:
            raise EvidenceError(
                "pilot_scientific_identity_mismatch",
                f"{label} must bind every scientific path to activation E",
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
                "independent_trainer_replay": trainer_replay,
                "private_dataset_replay": private_dataset_tree,
                "private_bundle_replay": private_bundle_tree,
            },
            "_checkpoint_path": checkpoint_path,
            "_checkpoint_bytes": checkpoint_bytes,
        }
        artifact_binding = {
            "arm": arm,
            "seed_identity": dataset_summary["seed_identity"],
            "dataset": {
                **dataset_binding,
                "arrays_hashed_and_opened": dataset_tree["arrays_hashed_and_opened"],
                "required_array_shapes_verified": dataset_tree[
                    "required_array_shapes_verified"
                ],
            },
            "trainer_bundle": {
                **bundle_binding,
                "arrays_hashed_and_opened": bundle_summary["arrays_hashed_and_opened"],
                "curriculum_sha256": bundle_summary["curriculum_sha256"],
            },
            "checkpoint": checkpoint_binding,
            "evaluation_report": evaluation_binding,
            "replay_report": replay_binding,
            "independent_evaluator_replay": independent_replay,
            "independent_trainer_replay": trainer_replay,
            "private_dataset_replay": private_dataset_tree,
            "private_bundle_replay": private_bundle_tree,
        }
        if attempt_receipt_binding is not None:
            artifact_binding["attempt_receipt"] = attempt_receipt_binding
        artifact_bindings.append(artifact_binding)

    required_schedule_kinds = (
        {"cgb_schedule.jsonl"}
        if scope == "direct_state"
        else {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}
    )
    if any(
        len(query_schedule_hashes[kind]) != 1 for kind in required_schedule_kinds
    ) or any(
        query_schedule_hashes[kind]
        for kind in set(query_schedule_hashes) - required_schedule_kinds
    ):
        raise EvidenceError(
            "query_schedule_binding_fork",
            "all non-uniform runs must share one CGB schedule hash and all uniform-query runs one uniform schedule hash",
        )
    frozen_schedule_hashes = {
        kind: next(iter(query_schedule_hashes[kind]))
        for kind in sorted(required_schedule_kinds)
    }
    if scope != "direct_state" and len(set(frozen_schedule_hashes.values())) != 2:
        raise EvidenceError(
            "query_schedule_hash_reused",
            "CGB and uniform schedules must bind distinct artifacts",
        )

    missing = sorted(expected_keys - set(indexed))
    extra = sorted(set(indexed) - expected_keys)
    if missing or extra:
        raise EvidenceError("run_matrix_mismatch", f"missing={missing}, extra={extra}")
    assert scientific_identity is not None
    if (
        attempt_start_binding is not None
        and attempt_start_binding["scientific_commit"]
        != scientific_identity["scientific_commit"]
    ):
        raise EvidenceError(
            "attempt_start_commit_mismatch",
            "attempt start and fitted checkpoints bind different G commits",
        )
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
        "scope": scope,
        "confirmation_evidence_opened": scope == "full",
        "development_baseline_binding": development_baseline_binding,
        "development_plan_binding": development_plan_binding,
        "attempt_start_binding": attempt_start_binding,
        "attempt_claim_binding": attempt_claim_binding,
        "stage_receipt_bindings": stage_receipt_bindings,
        "private_refit_bindings": private_refit_bindings,
        "phase2_authorization": phase2_authorization,
        "confirmation_artifacts_transitively_bound_to_baseline": scope == "full",
        "manifest_payload_sha256": manifest_payload_sha256,
        "exact_run_matrix": True,
        "scored_runs_verified": (
            0
            if scope == "direct_state"
            else len(SCORED_ARMS) * (3 if scope == "development" else 6)
        ),
        "direct_state_runs_verified": 3,
        "evaluation_reports_verified": len(ordered),
        "byte_identical_replays_verified": len(ordered),
        "independent_evaluator_replays_verified": len(ordered),
        "independent_trainer_replays_verified": len(ordered),
        "private_dataset_replays_verified": len(private_dataset_cache),
        "private_bundle_replays_verified": len(private_bundle_cache),
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


def verify_evidence(
    manifest: dict[str, Any],
    base: Path,
    *,
    scope: str = "full",
    expected_development_baseline_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify evidence using inputs regenerated in an ephemeral private tree."""

    with tempfile.TemporaryDirectory(prefix="acw-private-verifier-") as temporary:
        return _verify_evidence_with_private_workspace(
            manifest,
            base,
            private_root=Path(temporary),
            scope=scope,
            expected_development_baseline_record=(expected_development_baseline_record),
        )


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


def _validate_development_plan_reference(value: Any, base: Path) -> dict[str, Any]:
    plan, binding, path = _verify_json_reference(value, "development_plan", base)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "development_plan_mutable", "development plan must be mode 0444"
        )
    raw, digest = _read_regular_file(path)
    if digest != DEVELOPMENT_PLAN_RAW_SHA256:
        raise EvidenceError(
            "development_plan_mismatch", "development plan raw bytes differ"
        )
    if raw != canonical_json_bytes(plan) + b"\n":
        raise EvidenceError(
            "development_plan_mismatch", "development plan is not canonical JSON"
        )
    payload_sha256 = _verify_payload_hash(plan, "development plan")
    try:
        from pipeline.build_acw_development_manifest import validate_plan

        validate_plan(plan, require_ready=True)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise EvidenceError(
            "development_plan_mismatch",
            f"development plan contract differs: {exc}",
        ) from exc
    return {
        **binding,
        "path": str(path.resolve(strict=True)),
        "payload_sha256": payload_sha256,
    }


def _validate_attempt_start_reference(
    value: Any,
    base: Path,
    *,
    expected_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    attempt, binding, path = _verify_json_reference(value, "attempt_start", base)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "attempt_start_mutable", "attempt start must be a mode-0444 regular file"
        )
    raw, _ = _read_regular_file(path)
    if raw != canonical_json_bytes(attempt) + b"\n":
        raise EvidenceError(
            "attempt_start_noncanonical",
            "attempt start is not canonical newline-framed JSON",
        )
    _expect_keys(
        attempt,
        {
            "schema",
            "protocol",
            "scientific_commit",
            "development_plan",
            "artifact_root",
            "slurm",
            "created_before_scoring",
            "checkpoint_count_at_creation",
            "one_attempt",
            "overwrite",
            "attempt_ids",
            "payload_sha256",
        },
        "attempt start",
    )
    payload_sha256 = _verify_payload_hash(attempt, "attempt start")
    nested_plan = _validate_development_plan_reference(
        attempt["development_plan"], base
    )
    slurm = _object(attempt["slurm"], "attempt start Slurm identity")
    _expect_keys(
        slurm,
        {"job_id", "job_name", "node_list", "cpus_per_task"},
        "attempt start Slurm identity",
    )
    plan, _, _ = _verify_json_reference(
        attempt["development_plan"], "attempt start development plan", base
    )
    attempt_registry = _object(
        plan.get("attempt_registry"), "development plan attempt registry"
    )
    stages = _list(plan.get("custody_stages"), "development plan custody stages")
    phase1 = _object(stages[0], "development plan phase-1 stage")
    held_job_id = str(phase1.get("held_slurm_job_id", ""))
    scientific_commit = attempt["scientific_commit"]
    if (
        attempt["schema"] != ATTEMPT_START_SCHEMA
        or attempt["protocol"] != ATTEMPT_START_PROTOCOL
        or nested_plan != expected_plan
        or not isinstance(scientific_commit, str)
        or len(scientific_commit) != 40
        or any(character not in "0123456789abcdef" for character in scientific_commit)
        or attempt["artifact_root"] != str(base.resolve(strict=True))
        or not held_job_id.isdigit()
        or str(slurm["job_id"]) != held_job_id
        or slurm["job_name"] != phase1.get("job_name")
        or slurm["node_list"] != phase1.get("expected_node")
        or not isinstance(slurm["node_list"], str)
        or not slurm["node_list"]
        or slurm["cpus_per_task"] != "4"
        or attempt["created_before_scoring"] is not True
        or attempt["checkpoint_count_at_creation"] != 0
        or attempt["one_attempt"] is not True
        or attempt["overwrite"] is not False
        or attempt["attempt_ids"] != attempt_registry.get("attempt_ids")
    ):
        raise EvidenceError(
            "attempt_start_mismatch", "attempt start differs from the committed plan"
        )
    return {
        **binding,
        "path": str(path.resolve(strict=True)),
        "payload_sha256": payload_sha256,
        "scientific_commit": scientific_commit,
        "slurm": slurm,
    }


def _validate_attempt_claim_reference(
    value: Any,
    base: Path,
    *,
    expected_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    claim, binding, path = _verify_json_reference(value, "attempt_claim", base)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "attempt_claim_mutable", "attempt claim must be a mode-0444 regular file"
        )
    _expect_keys(
        claim,
        {
            "schema",
            "protocol",
            "development_plan",
            "artifact_root",
            "attempt_count",
            "attempt_table_sha256",
            "jobs_sha256",
            "all_argv_and_paths_claimed_before_scoring",
            "checkpoint_count_at_creation",
            "confirmation_authorized",
            "claim_boundary",
            "payload_sha256",
        },
        "attempt claim",
    )
    payload_sha256 = _verify_payload_hash(claim, "attempt claim")
    nested_plan = _validate_development_plan_reference(claim["development_plan"], base)
    plan, _, _ = _verify_json_reference(
        claim["development_plan"], "attempt claim development plan", base
    )
    if (
        claim["schema"] != "r12_acw_development_attempt_claim_v1"
        or claim["protocol"] != "R12-ACW-DEVELOPMENT-ATTEMPT-CLAIM-v1"
        or nested_plan != expected_plan
        or claim["artifact_root"] != str(base.resolve(strict=True))
        or claim["attempt_count"] != 27
        or claim["attempt_table_sha256"]
        != hashlib.sha256(canonical_json_bytes(plan["attempt_table"])).hexdigest()
        or claim["jobs_sha256"]
        != hashlib.sha256(canonical_json_bytes(plan["custody_stages"])).hexdigest()
        or claim["all_argv_and_paths_claimed_before_scoring"] is not True
        or claim["checkpoint_count_at_creation"] != 0
        or claim["confirmation_authorized"] is not False
        or claim["claim_boundary"] != plan["claim_boundary"]
    ):
        raise EvidenceError(
            "attempt_claim_mismatch", "attempt claim differs from the committed plan"
        )
    return {
        **binding,
        "path": str(path.resolve(strict=True)),
        "payload_sha256": payload_sha256,
    }


def _validate_stage_job_binding(
    value: Any,
    *,
    stage: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    slurm = _object(value, label)
    _expect_keys(
        slurm,
        {
            "job_id",
            "job_name",
            "node",
            "cpus_per_task",
            "dependency",
            "script",
            "spool_script_sha256",
            "scontrol_snapshot_sha256",
            "process_membership",
        },
        label,
    )
    membership = _object(slurm["process_membership"], f"{label}.process_membership")
    _expect_keys(
        membership,
        {"cpu_list", "memory_list", "task_cgroup"},
        f"{label}.process_membership",
    )
    from pipeline.freeze_acw_curriculum import CANONICAL_PILOT_UID, _cpu_list_members

    try:
        cpu_members = _cpu_list_members(str(membership["cpu_list"]))
    except (TypeError, ValueError) as exc:
        raise EvidenceError(
            "stage_receipt_mismatch", f"{label} CPU membership is malformed"
        ) from exc
    job_id = str(stage["held_slurm_job_id"])
    if (
        str(slurm["job_id"]) != job_id
        or slurm["job_name"] != stage["job_name"]
        or slurm["node"] != stage["expected_node"]
        or slurm["cpus_per_task"] != "4"
        or slurm["dependency"] != stage["dependency"]
        or slurm["script"] != stage["script"]
        or slurm["spool_script_sha256"] != stage["script"]["sha256"]
        or _hash(
            slurm["scontrol_snapshot_sha256"],
            f"{label}.scontrol_snapshot_sha256",
        )
        != slurm["scontrol_snapshot_sha256"]
        or len(cpu_members) != 4
        or not str(membership["memory_list"])
        or membership["task_cgroup"]
        != f"/slurm/uid_{CANONICAL_PILOT_UID}/job_{job_id}/step_batch/task_0"
    ):
        raise EvidenceError(
            "stage_receipt_mismatch", f"{label} differs from the committed stage"
        )
    return dict(slurm)


def _stable_stage_job_binding(value: dict[str, Any]) -> dict[str, Any]:
    """Return only allocation identity fields that cannot drift during a job."""

    return {
        key: value[key]
        for key in (
            "job_id",
            "job_name",
            "node",
            "cpus_per_task",
            "dependency",
            "script",
            "spool_script_sha256",
            "process_membership",
        )
    }


def _registered_closed_world_tree_files(tree: Path, *, kind: str) -> set[Path]:
    if not tree.is_dir() or tree.is_symlink():
        raise EvidenceError(
            "stage_receipt_mismatch", f"{kind} closed-world tree is unavailable"
        )
    manifest_path = tree / "manifest.json"
    raw, _ = _read_regular_file(manifest_path)
    manifest = _parse_json(raw, f"{kind} closed-world manifest")
    registries = (
        (manifest.get("arrays"),)
        if kind == "dataset"
        else (
            manifest.get("arrays"),
            manifest.get("files"),
            manifest.get("pilot_artifacts"),
        )
    )
    relatives = {"manifest.json"}
    for registry in registries:
        if not isinstance(registry, dict):
            raise EvidenceError(
                "stage_receipt_mismatch",
                f"{kind} closed-world manifest registry is missing",
            )
        for raw_relative in registry:
            relative = Path(str(raw_relative))
            if relative.is_absolute() or ".." in relative.parts:
                raise EvidenceError(
                    "stage_receipt_mismatch",
                    f"{kind} closed-world manifest path is unsafe",
                )
            relatives.add(relative.as_posix())
    expected = {tree / relative for relative in relatives}
    actual = {
        path for path in tree.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise EvidenceError(
            "stage_receipt_mismatch",
            f"{kind} closed-world tree differs from its manifest registry",
        )
    return expected


def _expected_stage_closed_world_inventory(
    plan: dict[str, Any],
    *,
    attempt_root: Path,
    scan_root: Path,
    role: str,
    include_current_terminal_receipts: bool = False,
) -> tuple[list[str], int]:
    roles = (
        "phase1_producer",
        "phase1_verifier",
        "phase2_producer",
        "phase2_verifier",
    )
    if role not in roles:
        raise EvidenceError("stage_receipt_mismatch", "closed-world role is unknown")
    current_ordinal = roles.index(role)
    starts = {item: f"custody/stages/{item}_start.json" for item in roles}
    completions = {item: f"custody/stages/{item}_completion.json" for item in roles}
    accounting = {item: f"custody/stages/{item}_accounting.json" for item in roles}
    outputs = {
        "phase1_producer": ("direct_state_producer_manifest.json",),
        "phase1_verifier": (
            "direct_refit_verification.json",
            "direct_state_manifest.json",
            "direct_state_decision.json",
            "phase2_authorization.json",
        ),
        "phase2_producer": ("development_producer_manifest.json",),
        "phase2_verifier": (
            "final_refit_verification.json",
            "development_manifest.json",
        ),
    }
    expected: set[Path] = set()
    if scan_root == attempt_root:
        expected.update(
            attempt_root / name
            for name in (
                "development_plan.json",
                "attempt_claim.json",
                "attempt_start.json",
            )
        )
        for ordinal, prior_role in enumerate(roles[: current_ordinal + 1]):
            expected.add(attempt_root / starts[prior_role])
            if ordinal < current_ordinal or include_current_terminal_receipts:
                expected.add(attempt_root / completions[prior_role])
                expected.add(attempt_root / accounting[prior_role])
            expected.update(attempt_root / name for name in outputs[prior_role])

    input_table = _list(plan.get("input_table"), "development plan input table")
    for raw_record in input_table:
        record = _object(raw_record, "development plan input record")
        input_role = str(record.get("role"))
        if input_role not in roles or roles.index(input_role) > current_ordinal:
            continue
        paths = _object(record.get("paths"), "development plan input paths")
        for name, raw_path in paths.items():
            tree = Path(str(raw_path))
            if tree.is_relative_to(scan_root):
                expected.update(
                    _registered_closed_world_tree_files(
                        tree, kind="dataset" if name == "dataset" else "bundle"
                    )
                )

    attempt_table = _list(plan.get("attempt_table"), "development plan attempt table")
    for raw_attempt in attempt_table:
        attempt = _object(raw_attempt, "development plan attempt record")
        for side_name in ("producer", "verifier"):
            side = _object(attempt.get(side_name), f"attempt {side_name}")
            side_role = str(side.get("job_role"))
            if side_role not in roles or roles.index(side_role) > current_ordinal:
                continue
            paths = _object(side.get("paths"), f"attempt {side_name} paths")
            task_root = Path(str(paths.get("task_root")))
            if task_root.is_relative_to(scan_root):
                task_files = {
                    task_root / name
                    for name in (
                        "attempt.json",
                        "checkpoint.pt",
                        "evaluation.json",
                        "replay.json",
                    )
                }
                actual_task_files = {
                    path
                    for path in task_root.rglob("*")
                    if path.is_file() and not path.is_symlink()
                }
                if actual_task_files != task_files:
                    raise EvidenceError(
                        "stage_receipt_mismatch",
                        "attempt task differs from its exact four-file registry",
                    )
                expected.update(task_files)

    relatives = sorted(path.relative_to(scan_root).as_posix() for path in expected)
    directories = {scan_root}
    for path in expected:
        parent = path.parent
        while parent != scan_root:
            if not parent.is_relative_to(scan_root):
                raise EvidenceError(
                    "stage_receipt_mismatch", "closed-world file escapes its root"
                )
            directories.add(parent)
            parent = parent.parent
    return relatives, len(directories)


def _validate_predecessor_handoff(
    value: Any,
    *,
    plan: dict[str, Any],
    base: Path,
    role: str,
) -> dict[str, Any] | None:
    roles = (
        "phase1_producer",
        "phase1_verifier",
        "phase2_producer",
        "phase2_verifier",
    )
    ordinal = roles.index(role)
    if ordinal == 0:
        if value is not None:
            raise EvidenceError(
                "stage_receipt_mismatch", "first role has a predecessor handoff"
            )
        return None
    handoff = _object(value, f"{role} predecessor handoff")
    _expect_keys(
        handoff,
        {
            "predecessor_role",
            "predecessor_stage",
            "predecessor_completion",
            "predecessor_terminal_accounting",
            "live_closed_world_before_consumer_scoring",
            "consumer_observed_before_role_scoring",
        },
        f"{role} predecessor handoff",
    )
    predecessor = roles[ordinal - 1]
    predecessor_stage = (
        "phase1",
        "direct_verified",
        "phase2",
        "final",
    )[ordinal - 1]
    completion_name = f"custody/stages/{predecessor}_completion.json"
    accounting_name = f"custody/stages/{predecessor}_accounting.json"
    _, completion_path = _verify_file_reference(
        handoff["predecessor_completion"],
        f"{role} predecessor completion",
        base,
    )
    _, accounting_path = _verify_file_reference(
        handoff["predecessor_terminal_accounting"],
        f"{role} predecessor accounting",
        base,
    )
    scans = _object(
        handoff["live_closed_world_before_consumer_scoring"],
        f"{role} predecessor handoff scans",
    )
    expected_scan_keys = {"main", "direct_verifier"} if ordinal >= 2 else {"main"}
    _expect_keys(scans, expected_scan_keys, f"{role} predecessor handoff scans")
    main_paths, main_directory_count = _expected_stage_closed_world_inventory(
        plan,
        attempt_root=base.resolve(strict=True),
        scan_root=base.resolve(strict=True),
        role=predecessor,
        include_current_terminal_receipts=True,
    )
    _validate_closed_world_summary(
        scans["main"],
        expected_root=base.resolve(strict=True),
        expected_stage=predecessor_stage,
        expected_paths=main_paths,
        expected_directory_count=main_directory_count,
        label=f"{role} predecessor handoff scans.main",
    )
    if ordinal >= 2:
        private_root = (base.parent / "acw_development_g1_direct_verifier").resolve(
            strict=True
        )
        private_paths, private_directory_count = _expected_stage_closed_world_inventory(
            plan,
            attempt_root=base.resolve(strict=True),
            scan_root=private_root,
            role=predecessor,
            include_current_terminal_receipts=True,
        )
        _validate_closed_world_summary(
            scans["direct_verifier"],
            expected_root=private_root,
            expected_stage=predecessor_stage,
            expected_paths=private_paths,
            expected_directory_count=private_directory_count,
            label=f"{role} predecessor handoff scans.direct_verifier",
        )
    if (
        handoff["predecessor_role"] != predecessor
        or handoff["predecessor_stage"] != predecessor_stage
        or completion_path != (base / completion_name).resolve(strict=True)
        or accounting_path != (base / accounting_name).resolve(strict=True)
        or handoff["consumer_observed_before_role_scoring"] is not True
    ):
        raise EvidenceError(
            "stage_receipt_mismatch", f"{role} predecessor handoff differs"
        )
    return dict(handoff)


def _validate_closed_world_summary(
    value: Any,
    *,
    expected_root: Path,
    expected_stage: str,
    expected_paths: list[str],
    expected_directory_count: int,
    label: str,
) -> dict[str, Any]:
    summary = _object(value, label)
    if (
        not expected_root.is_dir()
        or expected_root.is_symlink()
        or expected_root.resolve(strict=True) != expected_root
    ):
        raise EvidenceError("stage_receipt_mismatch", f"{label} root is not canonical")
    _expect_keys(
        summary,
        {
            "stage",
            "root",
            "file_count",
            "directory_count",
            "files",
            "tree_sha256",
            "exact_file_set",
            "exact_directory_set",
            "symlinks",
            "special_files",
        },
        label,
    )
    files = _list(summary["files"], f"{label}.files")
    digest = hashlib.sha256()
    observed_paths: list[str] = []
    for index, raw_record in enumerate(files):
        record_label = f"{label}.files[{index}]"
        record = _object(raw_record, record_label)
        _expect_keys(record, {"path", "bytes", "mode", "sha256"}, record_label)
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or record["mode"] != "0444"
            or _integer(record["bytes"], f"{record_label}.bytes") < 0
            or _hash(record["sha256"], f"{record_label}.sha256") != record["sha256"]
        ):
            raise EvidenceError(
                "stage_receipt_mismatch", f"{record_label} is malformed"
            )
        observed_paths.append(relative)
        artifact = expected_root / relative
        try:
            resolved = artifact.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise EvidenceError(
                "stage_receipt_mismatch", f"{record_label} is unavailable"
            ) from exc
        if (
            artifact.is_symlink()
            or not resolved.is_relative_to(expected_root)
            or not resolved.is_file()
            or stat.S_IMODE(resolved.stat().st_mode) != 0o444
        ):
            raise EvidenceError(
                "stage_receipt_mismatch", f"{record_label} is not immutable and rooted"
            )
        size, observed_sha256 = _hash_regular_file(resolved, record_label)
        if size != record["bytes"] or observed_sha256 != record["sha256"]:
            raise EvidenceError(
                "stage_receipt_mismatch", f"{record_label} bytes differ"
            )
        digest.update(canonical_json_bytes(record) + b"\n")
    if (
        summary["stage"] != expected_stage
        or summary["root"] != str(expected_root)
        or summary["file_count"] != len(files)
        or summary["directory_count"] != expected_directory_count
        or observed_paths != sorted(observed_paths)
        or len(set(observed_paths)) != len(observed_paths)
        or observed_paths != expected_paths
        or _hash(summary["tree_sha256"], f"{label}.tree_sha256") != digest.hexdigest()
        or summary["exact_file_set"] is not True
        or summary["exact_directory_set"] is not True
        or summary["symlinks"] != 0
        or summary["special_files"] != 0
    ):
        raise EvidenceError(
            "stage_receipt_mismatch", f"{label} closed-world scan differs"
        )
    return dict(summary)


def _validate_stage_receipts(
    value: Any,
    base: Path,
    *,
    scope: str,
    expected_plan: dict[str, Any] | None,
    require_all_closed: bool = False,
) -> dict[str, Any]:
    receipts = _object(value, "stage receipts")
    roles = (
        ("phase1_producer", "phase1_verifier")
        if scope == "direct_state"
        else (
            "phase1_producer",
            "phase1_verifier",
            "phase2_producer",
            "phase2_verifier",
        )
    )
    _expect_keys(receipts, set(roles), "stage receipts")
    plan, _, _ = _verify_json_reference(
        {"path": "development_plan.json", "sha256": DEVELOPMENT_PLAN_RAW_SHA256},
        "stage receipt development plan",
        base,
    )
    stages = {
        stage["role"]: stage
        for stage in _list(plan["custody_stages"], "custody stages")
    }
    bindings: dict[str, Any] = {}
    for position, role in enumerate(roles):
        stage = _object(stages.get(role), f"stage plan {role}")
        record = _object(receipts[role], f"stage receipts.{role}")
        open_role = position == len(roles) - 1 and not require_all_closed
        expected_record_keys = (
            {"start"} if open_role else {"start", "completion", "terminal_accounting"}
        )
        _expect_keys(record, expected_record_keys, f"stage receipts.{role}")
        start, start_binding, start_path = _verify_json_reference(
            record["start"], f"stage receipts.{role}.start", base
        )
        if stat.S_IMODE(start_path.stat().st_mode) != 0o444:
            raise EvidenceError("stage_receipt_mutable", f"{role} start is mutable")
        _expect_keys(
            start,
            {
                "schema",
                "protocol",
                "role",
                "development_plan",
                "attempt_claim",
                "scientific_commit",
                "slurm",
                "planned_work",
                "predecessor_handoff",
                "created_before_role_scoring",
                "confirmation_authorized",
                "payload_sha256",
            },
            f"{role} start",
        )
        start_payload = _verify_payload_hash(start, f"{role} start")
        nested_plan = _validate_development_plan_reference(
            start["development_plan"], base
        )
        _validate_attempt_claim_reference(
            start["attempt_claim"], base, expected_plan=expected_plan
        )
        start_slurm = _validate_stage_job_binding(
            start["slurm"], stage=stage, label=f"{role} start.slurm"
        )
        _validate_predecessor_handoff(
            start["predecessor_handoff"], plan=plan, base=base, role=role
        )
        if (
            start["schema"] != "r12_acw_development_stage_start_v1"
            or start["protocol"] != "R12-ACW-DEVELOPMENT-STAGE-START-v1"
            or start["role"] != role
            or nested_plan != expected_plan
            or re.fullmatch(r"[0-9a-f]{40}", str(start["scientific_commit"])) is None
            or start["planned_work"] != stage["work"]
            or start["created_before_role_scoring"] is not True
            or start["confirmation_authorized"] is not False
        ):
            raise EvidenceError("stage_receipt_mismatch", f"{role} start differs")
        role_binding: dict[str, Any] = {
            "start": {**start_binding, "payload_sha256": start_payload},
            "slurm": start_slurm,
        }
        if not open_role:
            completion, completion_binding, completion_path = _verify_json_reference(
                record["completion"], f"stage receipts.{role}.completion", base
            )
            if stat.S_IMODE(completion_path.stat().st_mode) != 0o444:
                raise EvidenceError(
                    "stage_receipt_mutable", f"{role} completion is mutable"
                )
            _expect_keys(
                completion,
                {
                    "schema",
                    "protocol",
                    "role",
                    "development_plan",
                    "stage_start",
                    "slurm",
                    "completed_work",
                    "outputs",
                    "closed_world",
                    "all_outputs_immutable",
                    "normal_slurm_steps_used",
                    "confirmation_authorized",
                    "payload_sha256",
                },
                f"{role} completion",
            )
            completion_payload = _verify_payload_hash(completion, f"{role} completion")
            completion_slurm = _validate_stage_job_binding(
                completion["slurm"], stage=stage, label=f"{role} completion.slurm"
            )
            expected_output_names = {
                "phase1_producer": ("direct_state_producer_manifest.json",),
                "phase1_verifier": (
                    "direct_refit_verification.json",
                    "direct_state_manifest.json",
                    "direct_state_decision.json",
                    "phase2_authorization.json",
                ),
                "phase2_producer": ("development_producer_manifest.json",),
                "phase2_verifier": (
                    "final_refit_verification.json",
                    "development_manifest.json",
                ),
            }[role]
            outputs = _object(completion["outputs"], f"{role} completion.outputs")
            _expect_keys(
                outputs,
                set(expected_output_names),
                f"{role} completion.outputs",
            )
            for output_name in expected_output_names:
                _, output_path = _verify_file_reference(
                    outputs[output_name],
                    f"{role} completion.outputs.{output_name}",
                    base,
                )
                if (
                    output_path != (base / output_name).resolve(strict=True)
                    or stat.S_IMODE(output_path.stat().st_mode) != 0o444
                ):
                    raise EvidenceError(
                        "stage_receipt_mismatch",
                        f"{role} completion output differs: {output_name}",
                    )
            closed_world = _object(
                completion["closed_world"], f"{role} completion.closed_world"
            )
            expected_closed_world_keys = (
                {"main", "private"}
                if role in {"phase1_verifier", "phase2_verifier"}
                else {"main"}
            )
            _expect_keys(
                closed_world,
                expected_closed_world_keys,
                f"{role} completion.closed_world",
            )
            expected_stage = {
                "phase1_producer": "phase1",
                "phase1_verifier": "direct_verified",
                "phase2_producer": "phase2",
                "phase2_verifier": "final",
            }[role]
            main_paths, main_directory_count = _expected_stage_closed_world_inventory(
                plan,
                attempt_root=base.resolve(strict=True),
                scan_root=base.resolve(strict=True),
                role=role,
            )
            _validate_closed_world_summary(
                closed_world["main"],
                expected_root=base.resolve(strict=True),
                expected_stage=expected_stage,
                expected_paths=main_paths,
                expected_directory_count=main_directory_count,
                label=f"{role} completion.closed_world.main",
            )
            if "private" in closed_world:
                private_name = (
                    "acw_development_g1_direct_verifier"
                    if role == "phase1_verifier"
                    else "acw_development_g1_final_verifier"
                )
                private_root = (base.parent / private_name).resolve(strict=True)
                private_paths, private_directory_count = (
                    _expected_stage_closed_world_inventory(
                        plan,
                        attempt_root=base.resolve(strict=True),
                        scan_root=private_root,
                        role=role,
                    )
                )
                _validate_closed_world_summary(
                    closed_world["private"],
                    expected_root=private_root,
                    expected_stage=expected_stage,
                    expected_paths=private_paths,
                    expected_directory_count=private_directory_count,
                    label=f"{role} completion.closed_world.private",
                )
            if (
                completion["schema"] != "r12_acw_development_stage_completion_v1"
                or completion["protocol"] != "R12-ACW-DEVELOPMENT-STAGE-COMPLETION-v1"
                or completion["role"] != role
                or completion["development_plan"] != start["development_plan"]
                or completion["stage_start"] != record["start"]
                or _stable_stage_job_binding(completion_slurm)
                != _stable_stage_job_binding(start_slurm)
                or completion["completed_work"] != stage["work"]
                or completion["all_outputs_immutable"] is not True
                or completion["normal_slurm_steps_used"] != 0
                or completion["confirmation_authorized"] is not False
            ):
                raise EvidenceError(
                    "stage_receipt_mismatch", f"{role} completion differs"
                )
            accounting, accounting_binding, accounting_path = _verify_json_reference(
                record["terminal_accounting"],
                f"stage receipts.{role}.terminal_accounting",
                base,
            )
            if stat.S_IMODE(accounting_path.stat().st_mode) != 0o444:
                raise EvidenceError(
                    "stage_receipt_mutable", f"{role} accounting is mutable"
                )
            _expect_keys(
                accounting,
                {
                    "schema",
                    "protocol",
                    "role",
                    "development_plan",
                    "observed_by",
                    "terminal_rows",
                    "normal_slurm_steps",
                    "terminal_completed",
                    "resource_values_are_diagnostic_only",
                    "confirmation_authorized",
                    "payload_sha256",
                },
                f"{role} accounting",
            )
            _verify_payload_hash(accounting, f"{role} accounting")
            accounting_plan = _validate_development_plan_reference(
                accounting["development_plan"], base
            )
            if position + 1 < len(roles):
                observer_role = roles[position + 1]
                observer_stage = _object(
                    stages.get(observer_role), f"stage plan {observer_role}"
                )
                _validate_stage_job_binding(
                    accounting["observed_by"],
                    stage=observer_stage,
                    label=f"{role} accounting.observed_by",
                )
            else:
                monitor_stage = _object(
                    plan.get("accounting", {}).get("monitor_stage"),
                    "terminal monitor stage",
                )
                _validate_stage_job_binding(
                    accounting["observed_by"],
                    stage=monitor_stage,
                    label=f"{role} accounting.observed_by",
                )
            terminal_rows = _list(
                accounting["terminal_rows"], f"{role} accounting.terminal_rows"
            )
            job_id = str(stage["held_slurm_job_id"])
            expected_row_names = {
                job_id: stage["job_name"],
                f"{job_id}.batch": "batch",
                f"{job_id}.extern": "extern",
            }
            observed_rows: dict[str, dict[str, Any]] = {}
            for row_index, raw_row in enumerate(terminal_rows):
                row_label = f"{role} accounting.terminal_rows[{row_index}]"
                row = _object(raw_row, row_label)
                _expect_keys(
                    row,
                    {
                        "job_id_raw",
                        "job_name",
                        "state",
                        "exit_code",
                        "node_list",
                        "cpus",
                        "elapsed_raw",
                        "max_rss",
                    },
                    row_label,
                )
                row_id = row["job_id_raw"]
                if not isinstance(row_id, str) or row_id in observed_rows:
                    raise EvidenceError(
                        "stage_accounting_mismatch",
                        f"{role} accounting rows are duplicated",
                    )
                observed_rows[row_id] = row
            rows_valid = set(observed_rows) == set(expected_row_names)
            if rows_valid:
                for row_id, expected_name in expected_row_names.items():
                    row = observed_rows[row_id]
                    if (
                        row["job_name"] != expected_name
                        or row["state"] != "COMPLETED"
                        or row["exit_code"] != "0:0"
                        or row["node_list"] != stage["expected_node"]
                        or row["cpus"] != "4"
                        or not isinstance(row["elapsed_raw"], str)
                        or not isinstance(row["max_rss"], str)
                    ):
                        rows_valid = False
                        break
            if (
                accounting.get("schema") != "r12_acw_development_stage_accounting_v1"
                or accounting.get("protocol")
                != "R12-ACW-DEVELOPMENT-STAGE-ACCOUNTING-v1"
                or accounting.get("role") != role
                or accounting.get("terminal_completed") is not True
                or accounting.get("normal_slurm_steps") != []
                or accounting_plan != expected_plan
                or not rows_valid
                or accounting.get("resource_values_are_diagnostic_only") is not True
                or accounting.get("confirmation_authorized") is not False
            ):
                raise EvidenceError(
                    "stage_accounting_mismatch", f"{role} accounting differs"
                )
            role_binding["completion"] = {
                **completion_binding,
                "payload_sha256": completion_payload,
            }
            role_binding["terminal_accounting"] = accounting_binding
        bindings[role] = role_binding
    return bindings


def _validate_private_refit_verification_reference(
    value: Any,
    base: Path,
    *,
    scope: str,
    expected_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    report, binding, path = _verify_json_reference(
        value, f"{scope} private refit verification", base
    )
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "private_refit_verification_mutable",
            f"{scope} private refit verification must be mode 0444",
        )
    _expect_keys(
        report,
        {
            "schema",
            "protocol",
            "scope",
            "development_plan",
            "attempt_count",
            "comparisons",
            "datasets_regenerated_privately",
            "curricula_regenerated_privately",
            "models_refit_from_private_copies",
            "model_tensors_byte_identical",
            "normalized_evaluations_identical",
            "confirmation_authorized",
            "payload_sha256",
        },
        f"{scope} private refit verification",
    )
    payload_sha256 = _verify_payload_hash(report, f"{scope} private refit verification")
    nested_plan = _validate_development_plan_reference(report["development_plan"], base)
    expected_arms = (DIRECT_STATE_ARM,) if scope == "direct" else SCORED_ARMS
    expected_attempt_ids = [
        f"{arm}__{seed}" for arm in expected_arms for seed in DEVELOPMENT_SEEDS
    ]
    comparisons = _list(report["comparisons"], f"{scope} private refit comparisons")
    observed_attempt_ids = []
    for index, value in enumerate(comparisons):
        comparison = _object(value, f"{scope} comparison[{index}]")
        _expect_keys(
            comparison,
            {
                "attempt_id",
                "model_tensor_sha256",
                "stable_checkpoint_payload_sha256",
                "producer_checkpoint_sha256",
                "verifier_checkpoint_sha256",
                "producer_evaluation",
                "verifier_evaluation",
                "producer_stage",
                "verifier_stage",
            },
            f"{scope} comparison[{index}]",
        )
        observed_attempt_ids.append(comparison["attempt_id"])
        for key in (
            "model_tensor_sha256",
            "stable_checkpoint_payload_sha256",
            "producer_checkpoint_sha256",
            "verifier_checkpoint_sha256",
        ):
            _hash(comparison[key], f"{scope} comparison[{index}].{key}")
        for side in ("producer_evaluation", "verifier_evaluation"):
            evaluation = _object(
                comparison[side], f"{scope} comparison[{index}].{side}"
            )
            _expect_keys(
                evaluation,
                {"raw_sha256", "normalized_payload_sha256"},
                f"{scope} comparison[{index}].{side}",
            )
            _hash(
                evaluation["raw_sha256"],
                f"{scope} comparison[{index}].{side}.raw_sha256",
            )
            _hash(
                evaluation["normalized_payload_sha256"],
                f"{scope} comparison[{index}].{side}.normalized_payload_sha256",
            )
        if (
            comparison["producer_evaluation"]["normalized_payload_sha256"]
            != (comparison["verifier_evaluation"]["normalized_payload_sha256"])
        ):
            raise EvidenceError(
                "private_refit_evaluation_mismatch",
                f"{scope} comparison[{index}] evaluator semantics differ",
            )
    expected_roles = (
        ("phase1_producer", "phase1_verifier")
        if scope == "direct"
        else ("phase2_producer", "phase2_verifier")
    )
    if (
        report["schema"] != "r12_acw_development_private_refit_verification_v1"
        or report["protocol"] != "R12-ACW-DEVELOPMENT-PRIVATE-REFIT-VERIFICATION-v1"
        or report["scope"] != scope
        or nested_plan != expected_plan
        or report["attempt_count"] != len(expected_attempt_ids)
        or observed_attempt_ids != expected_attempt_ids
        or any(
            (comparison["producer_stage"], comparison["verifier_stage"])
            != expected_roles
            for comparison in comparisons
        )
        or report["datasets_regenerated_privately"] is not True
        or report["curricula_regenerated_privately"] is not True
        or report["models_refit_from_private_copies"] is not True
        or report["model_tensors_byte_identical"] is not True
        or report["normalized_evaluations_identical"] is not True
        or report["confirmation_authorized"] is not False
    ):
        raise EvidenceError(
            "private_refit_verification_mismatch",
            f"{scope} private refit verification differs from the frozen matrix",
        )
    return {
        **binding,
        "path": str(path.resolve(strict=True)),
        "payload_sha256": payload_sha256,
        "attempt_ids": observed_attempt_ids,
    }


def _direct_state_decision_payload(
    manifest_path: Path,
    manifest_sha256: str,
    runs: list[dict[str, Any]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        {"index": index, **_direct_state_gate(run)} for index, run in enumerate(runs)
    ]
    passed = len(gates) == 3 and all(gate["passed"] for gate in gates)
    return with_payload_hash(
        {
            "schema": DIRECT_STATE_DECISION_SCHEMA,
            "protocol": DIRECT_STATE_DECISION_PROTOCOL,
            "decision": "PASS" if passed else "NO_GO",
            "passed": passed,
            "development_plan": verification["development_plan_binding"],
            "direct_state_manifest": {
                "path": str(manifest_path.resolve(strict=True)),
                "sha256": manifest_sha256,
                "payload_sha256": verification["manifest_payload_sha256"],
            },
            "seed_gates": gates,
            "verification": verification,
            "phase2_authorized": passed,
            "claim_boundary": (
                "Positive-control qualification only; this is not a scored "
                "architecture or reasoning result."
            ),
            "output_contract": {
                "exclusive_create": True,
                "overwrite": False,
                "mode": "0444",
            },
        }
    )


def qualify_direct_state(
    manifest_path: str | Path,
    decision_out: str | Path,
    authorization_out: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = Path(manifest_path)
    raw, manifest_sha256, manifest_stat = _read_regular_file_with_stat(path)
    if stat.S_IMODE(manifest_stat.st_mode) != 0o444:
        raise EvidenceError(
            "direct_state_manifest_mutable",
            "direct-state manifest must be an immutable mode-0444 regular file",
        )
    manifest = _parse_json(raw, "direct-state manifest")
    runs, verification = verify_evidence(manifest, path.parent, scope="direct_state")
    decision = _direct_state_decision_payload(path, manifest_sha256, runs, verification)
    decision_record = _write_immutable_binary(
        decision_out, canonical_json_bytes(decision) + b"\n"
    )
    if not decision["passed"]:
        return decision, None
    authorization = with_payload_hash(
        {
            "schema": PHASE2_AUTHORIZATION_SCHEMA,
            "protocol": PHASE2_AUTHORIZATION_PROTOCOL,
            "authorized": True,
            "development_plan": verification["development_plan_binding"],
            "direct_state_manifest": decision["direct_state_manifest"],
            "direct_state_decision": {
                "path": str(Path(decision_record["path"]).resolve(strict=True)),
                "sha256": decision_record["sha256"],
                "payload_sha256": decision["payload_sha256"],
            },
            "scored_arms": list(SCORED_ARMS),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "one_attempt": True,
            "confirmation_authorized": False,
            "claim_boundary": (
                "Authorization to execute only the fixed public development matrix; "
                "confirmation and reasoning claims remain closed."
            ),
        }
    )
    _write_immutable_binary(
        authorization_out, canonical_json_bytes(authorization) + b"\n"
    )
    return decision, authorization


def _validate_phase2_authorization(
    value: Any,
    base: Path,
    *,
    expected_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    authorization, binding, path = _verify_json_reference(
        value, "phase2_authorization", base
    )
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise EvidenceError(
            "phase2_authorization_mutable", "phase-2 authorization must be mode 0444"
        )
    _expect_keys(
        authorization,
        {
            "schema",
            "protocol",
            "authorized",
            "development_plan",
            "direct_state_manifest",
            "direct_state_decision",
            "scored_arms",
            "development_seeds",
            "one_attempt",
            "confirmation_authorized",
            "claim_boundary",
            "payload_sha256",
        },
        "phase2 authorization",
    )
    payload_sha256 = _verify_payload_hash(authorization, "phase2 authorization")
    if (
        authorization["schema"] != PHASE2_AUTHORIZATION_SCHEMA
        or authorization["protocol"] != PHASE2_AUTHORIZATION_PROTOCOL
        or authorization["authorized"] is not True
        or authorization["development_plan"] != expected_plan
        or authorization["scored_arms"] != list(SCORED_ARMS)
        or authorization["development_seeds"] != list(DEVELOPMENT_SEEDS)
        or authorization["one_attempt"] is not True
        or authorization["confirmation_authorized"] is not False
        or authorization["claim_boundary"]
        != (
            "Authorization to execute only the fixed public development matrix; "
            "confirmation and reasoning claims remain closed."
        )
    ):
        raise EvidenceError(
            "phase2_authorization_mismatch", "phase-2 authorization contract differs"
        )
    direct_manifest, direct_binding, direct_path = _verify_json_reference(
        authorization["direct_state_manifest"],
        "phase2_authorization.direct_state_manifest",
        base,
    )
    direct_runs, direct_verification = verify_evidence(
        direct_manifest, direct_path.parent, scope="direct_state"
    )
    direct_raw, direct_sha256 = _read_regular_file(direct_path)
    del direct_raw
    expected_decision = _direct_state_decision_payload(
        direct_path, direct_sha256, direct_runs, direct_verification
    )
    if not expected_decision["passed"]:
        raise EvidenceError(
            "direct_state_gate_failed", "direct-state qualification did not pass"
        )
    decision, decision_binding, decision_path = _verify_json_reference(
        authorization["direct_state_decision"],
        "phase2_authorization.direct_state_decision",
        base,
    )
    if (
        stat.S_IMODE(decision_path.stat().st_mode) != 0o444
        or decision != expected_decision
        or authorization["direct_state_manifest"]
        != {
            "path": direct_binding["path"],
            "sha256": direct_binding["sha256"],
            "payload_sha256": direct_verification["manifest_payload_sha256"],
        }
        or authorization["direct_state_decision"]
        != {
            "path": decision_binding["path"],
            "sha256": decision_binding["sha256"],
            "payload_sha256": expected_decision["payload_sha256"],
        }
    ):
        raise EvidenceError(
            "phase2_authorization_mismatch",
            "phase-2 authorization does not bind the independently replayed gate",
        )
    return {
        **binding,
        "path": str(path.resolve(strict=True)),
        "payload_sha256": payload_sha256,
        "direct_state_reverified": True,
    }


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


def _development_baseline(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Select and retain the strongest verified development checkpoint."""

    candidates = []
    arm_medians = {}
    for arm in SCORED_ARMS:
        selected = [
            run for run in runs if run["arm"] == arm and run["split"] == "development"
        ]
        assert len(selected) == 3
        state_median = float(
            median(
                run["evaluation"]["public_depths"][64]["state_exactness"]
                for run in selected
            )
        )
        scalar_median = float(
            median(
                run["evaluation"]["public_depths"][64]["scalar_accuracy"]
                for run in selected
            )
        )
        arm_medians[arm] = {
            "depth_64_state_exactness": state_median,
            "depth_64_scalar_accuracy": scalar_median,
        }
        for run in selected:
            metric = run["evaluation"]["public_depths"][64]
            candidates.append(
                {
                    "arm": arm,
                    "index": run["index"],
                    "seed_identity": run["seed_identity"],
                    "checkpoint": run["bindings"]["checkpoint"],
                    "depth_64_state_exactness": metric["state_exactness"],
                    "depth_64_scalar_accuracy": metric["scalar_accuracy"],
                }
            )

    ranked_arms = sorted(
        BASELINE_ELIGIBLE_ARMS,
        key=lambda arm: (
            -arm_medians[arm]["depth_64_state_exactness"],
            -arm_medians[arm]["depth_64_scalar_accuracy"],
            arm,
        ),
    )
    selected_arm = ranked_arms[0]
    selected_checkpoint = sorted(
        (row for row in candidates if row["arm"] == selected_arm),
        key=lambda row: (
            -row["depth_64_state_exactness"],
            -row["depth_64_scalar_accuracy"],
            row["index"],
        ),
    )[0]
    return {
        "status": "retained_baseline",
        "scope": "verified_development_runs_only",
        "eligible_arms": list(BASELINE_ELIGIBLE_ARMS),
        "ineligible_upper_bound_controls": ["source_retained"],
        "labels": FINAL_SCALAR_LABELS,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "selection_order": [
            "arm_median_depth_64_state_exactness_desc",
            "arm_median_depth_64_scalar_accuracy_desc",
            "arm_id_lexical_asc",
            "checkpoint_state_exactness_desc",
            "checkpoint_scalar_accuracy_desc",
            "development_index_asc",
        ],
        "selected_arm": selected_arm,
        "selected_arm_metrics": arm_medians[selected_arm],
        "selected_checkpoint": selected_checkpoint,
        "arm_medians": arm_medians,
        "candidate_registry": candidates,
        "candidate_count": len(candidates),
        "retention_independent_of_promotion": True,
        "can_override_promotion_gates": False,
        "claim_boundary": (
            "This records the strongest verified development baseline under the frozen "
            "protocol; it does not authorize a reasoning or generalization claim."
        ),
    }


def _development_run_bindings(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arm": run["arm"],
            "index": run["index"],
            "seed_identity": run["seed_identity"],
            "bindings": run["bindings"],
        }
        for run in runs
        if run["split"] == "development"
    ]


def _selected_development_run(
    runs: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = _object(selection["selected_checkpoint"], "selected checkpoint")
    matches = [
        run
        for run in runs
        if run["split"] == "development"
        and run["arm"] == selection["selected_arm"]
        and run["index"] == checkpoint["index"]
    ]
    if (
        len(matches) != 1
        or matches[0]["bindings"]["checkpoint"] != checkpoint["checkpoint"]
    ):
        raise EvidenceError(
            "development_baseline_selection_mismatch",
            "selected development checkpoint is not uniquely bound",
        )
    return matches[0]


def _write_immutable_binary(path: str | Path, raw: bytes) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while preserving development checkpoint")
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
    return {
        "path": str(destination.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mode": "0444",
    }


def _confirmation_authorization() -> dict[str, Any]:
    return {
        "protocol": CONFIRMATION_AUTHORIZATION_PROTOCOL,
        "authorized": False,
        "status": "pending_future_nist_beacon",
        "full_manifest_schema": MANIFEST_SCHEMA,
        "full_manifest_protocol": MANIFEST_PROTOCOL,
        "scored_arms": list(SCORED_ARMS),
        "confirmation_indices": [0, 1, 2],
        "confirmation_commitments": [],
        "direct_state_confirmation_authorized": False,
        "immutable_baseline_required_before_confirmation": True,
        "full_manifest_must_bind_baseline": True,
        "future_beacon_required": True,
    }


def freeze_development_baseline(
    manifest_path: str | Path, checkpoint_out: str | Path
) -> dict[str, Any]:
    """Verify development only and preserve its strongest deployable checkpoint."""

    path = Path(manifest_path)
    manifest_raw, manifest_sha256, manifest_stat = _read_regular_file_with_stat(path)
    if stat.S_IMODE(manifest_stat.st_mode) != 0o444:
        raise EvidenceError(
            "development_manifest_mutable",
            "development manifest must be an immutable mode-0444 regular file",
        )
    manifest = _parse_json(manifest_raw, "development manifest")
    runs, verification = verify_evidence(manifest, path.parent, scope="development")
    selection = _development_baseline(runs)
    selected_run = _selected_development_run(runs, selection)
    source_path = selected_run["_checkpoint_path"]
    destination = Path(checkpoint_out)
    try:
        if source_path.resolve(strict=True) == destination.resolve(strict=False):
            raise EvidenceError(
                "development_baseline_output_reused",
                "baseline copy must not reuse its source checkpoint path",
            )
    except OSError as exc:
        raise EvidenceError(
            "development_baseline_output_invalid",
            f"cannot resolve baseline checkpoint output: {exc}",
        ) from exc
    copied = _write_immutable_binary(destination, selected_run["_checkpoint_bytes"])
    if copied["sha256"] != selected_run["bindings"]["checkpoint"]["sha256"]:
        raise EvidenceError(
            "development_baseline_copy_mismatch",
            "preserved checkpoint differs from selected source bytes",
        )
    return with_payload_hash(
        {
            "schema": DEVELOPMENT_BASELINE_SCHEMA,
            "protocol": DEVELOPMENT_BASELINE_PROTOCOL,
            "status": "retained_baseline",
            "development_manifest": {
                "path": str(path.resolve(strict=True)),
                "sha256": manifest_sha256,
                "payload_sha256": verification["manifest_payload_sha256"],
            },
            "verification": verification,
            "selection": selection,
            "development_run_bindings": _development_run_bindings(runs),
            "source_checkpoint": selected_run["bindings"]["checkpoint"],
            "copied_checkpoint": copied,
            "activation_scientific_identity": verification["scientific_identity"],
            "confirmation_authorization": _confirmation_authorization(),
            "confirmation_evidence_opened": False,
            "retention_independent_of_promotion": True,
            "can_override_promotion_gates": False,
            "claim_boundary": DEVELOPMENT_BASELINE_CLAIM,
            "output_contract": {
                "exclusive_create": True,
                "overwrite": False,
                "mode": "0444",
            },
        }
    )


def _validate_frozen_development_baseline(
    baseline_path: str | Path,
) -> dict[str, Any]:
    path = Path(baseline_path)
    raw, file_sha256, baseline_stat = _read_regular_file_with_stat(path)
    if stat.S_IMODE(baseline_stat.st_mode) != 0o444:
        raise EvidenceError(
            "development_baseline_mutable",
            "development baseline must be an immutable mode-0444 regular file",
        )
    baseline = _parse_json(raw, "development baseline")
    if raw != canonical_json_bytes(baseline) + b"\n":
        raise EvidenceError(
            "development_baseline_noncanonical",
            "development baseline is not canonical newline-framed JSON",
        )
    payload_sha256 = _verify_payload_hash(baseline, "development baseline")
    _expect_keys(
        baseline,
        {
            "schema",
            "protocol",
            "status",
            "development_manifest",
            "verification",
            "selection",
            "development_run_bindings",
            "source_checkpoint",
            "copied_checkpoint",
            "activation_scientific_identity",
            "confirmation_authorization",
            "confirmation_evidence_opened",
            "retention_independent_of_promotion",
            "can_override_promotion_gates",
            "claim_boundary",
            "output_contract",
            "payload_sha256",
        },
        "development baseline",
    )
    if (
        baseline["schema"] != DEVELOPMENT_BASELINE_SCHEMA
        or baseline["protocol"] != DEVELOPMENT_BASELINE_PROTOCOL
        or baseline["status"] != "retained_baseline"
        or baseline["confirmation_evidence_opened"] is not False
        or baseline["retention_independent_of_promotion"] is not True
        or baseline["can_override_promotion_gates"] is not False
        or baseline["claim_boundary"] != DEVELOPMENT_BASELINE_CLAIM
        or baseline["confirmation_authorization"] != _confirmation_authorization()
        or baseline["output_contract"]
        != {"exclusive_create": True, "overwrite": False, "mode": "0444"}
    ):
        raise EvidenceError(
            "development_baseline_contract_mismatch",
            "development baseline retention contract differs",
        )
    development_manifest = _object(
        baseline["development_manifest"], "development baseline manifest"
    )
    _expect_keys(
        development_manifest,
        {"path", "sha256", "payload_sha256"},
        "development baseline manifest",
    )
    frozen_manifest, _, frozen_manifest_path = _verify_json_reference(
        {
            "path": development_manifest["path"],
            "sha256": development_manifest["sha256"],
        },
        "development baseline manifest",
        path.parent,
    )
    if (
        _verify_payload_hash(frozen_manifest, "development baseline manifest")
        != development_manifest["payload_sha256"]
    ):
        raise EvidenceError(
            "development_baseline_manifest_mismatch",
            "frozen development manifest binding differs",
        )
    development_runs, development_verification = verify_evidence(
        frozen_manifest,
        frozen_manifest_path.parent,
        scope="development",
    )
    baseline_verification = _object(
        baseline["verification"], "development baseline verification"
    )
    if (
        baseline_verification != development_verification
        or development_verification.get("status") != "verified"
        or development_verification.get("scope") != "development"
        or development_verification.get("confirmation_evidence_opened") is not False
        or development_verification.get("manifest_payload_sha256")
        != development_manifest["payload_sha256"]
        or development_verification.get("scientific_identity")
        != baseline["activation_scientific_identity"]
    ):
        raise EvidenceError(
            "development_baseline_verification_mismatch",
            "frozen development verification contract differs",
        )
    expected_selection = _development_baseline(development_runs)
    expected_bindings = _development_run_bindings(development_runs)
    if (
        baseline["selection"] != expected_selection
        or baseline["development_run_bindings"] != expected_bindings
    ):
        raise EvidenceError(
            "development_baseline_evidence_mismatch",
            "frozen baseline differs from full manifest development evidence",
        )
    selected_run = _selected_development_run(development_runs, expected_selection)
    if baseline["source_checkpoint"] != selected_run["bindings"]["checkpoint"]:
        raise EvidenceError(
            "development_baseline_source_mismatch",
            "frozen baseline source checkpoint differs",
        )
    copied_record = _object(
        baseline["copied_checkpoint"], "development baseline copied checkpoint"
    )
    _expect_keys(
        copied_record,
        {"path", "sha256", "bytes", "mode"},
        "development baseline copied checkpoint",
    )
    copied_binding, copied_path, copied_bytes = _verify_binary_reference(
        {"path": copied_record["path"], "sha256": copied_record["sha256"]},
        "development baseline copied checkpoint",
        path.parent,
    )
    copied_size = _integer(
        copied_record["bytes"],
        "development baseline copied checkpoint.bytes",
        minimum=1,
    )
    if (
        copied_record["mode"] != "0444"
        or stat.S_IMODE(copied_path.stat().st_mode) != 0o444
        or copied_binding["sha256"] != selected_run["bindings"]["checkpoint"]["sha256"]
        or copied_binding["bytes"] != selected_run["bindings"]["checkpoint"]["bytes"]
        or copied_size != copied_binding["bytes"]
        or copied_bytes != selected_run["_checkpoint_bytes"]
    ):
        raise EvidenceError(
            "development_baseline_copy_mismatch",
            "frozen baseline checkpoint copy differs from selected bytes",
        )
    return {
        **expected_selection,
        "selection": expected_selection,
        "record": {
            "path": str(path.resolve(strict=True)),
            "sha256": file_sha256,
            "payload_sha256": payload_sha256,
        },
        "development_manifest": baseline["development_manifest"],
        "development_verification": development_verification,
        "development_run_bindings": expected_bindings,
        "activation_scientific_identity": baseline["activation_scientific_identity"],
        "confirmation_authorization": baseline["confirmation_authorization"],
        "source_checkpoint": baseline["source_checkpoint"],
        "copied_checkpoint": baseline["copied_checkpoint"],
        "confirmation_evidence_opened_when_frozen": False,
        "retention_independent_of_promotion": True,
        "can_override_promotion_gates": False,
        "claim_boundary": DEVELOPMENT_BASELINE_CLAIM,
    }


def _validate_full_evidence_against_development_baseline(
    baseline: dict[str, Any],
    runs: list[dict[str, Any]],
    verification: dict[str, Any],
) -> None:
    expected_selection = _development_baseline(runs)
    if (
        baseline.get("selection") != expected_selection
        or baseline.get("development_run_bindings") != _development_run_bindings(runs)
        or baseline.get("activation_scientific_identity")
        != verification.get("scientific_identity")
        or verification.get("development_baseline_binding") != baseline.get("record")
        or verification.get("confirmation_artifacts_transitively_bound_to_baseline")
        is not True
    ):
        raise EvidenceError(
            "development_baseline_evidence_mismatch",
            "full evidence differs from the independently replayed development baseline",
        )


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
        "best_valid_development_baseline_retained_even_on_no_go": True,
        "baseline_eligible_arms": list(BASELINE_ELIGIBLE_ARMS),
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


def adjudicate_manifest(
    manifest_path: str | Path,
    development_baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all evidence and return a payload-hashed GO/NO_GO decision."""

    path = Path(manifest_path)
    manifest_file_sha256: str | None = None
    try:
        if _confirmation_authorization()["authorized"] is not True:
            raise EvidenceError(
                "confirmation_not_authorized",
                "full adjudication is closed pending a future commit-bound NIST Beacon pulse",
            )
        if development_baseline_path is None:
            raise EvidenceError(
                "development_baseline_required",
                "full adjudication requires a frozen development-only baseline",
            )
        development_baseline = _validate_frozen_development_baseline(
            development_baseline_path,
        )
        raw, manifest_file_sha256 = _read_regular_file(path)
        manifest = _parse_json(raw, "manifest")
        runs, verification = verify_evidence(
            manifest,
            path.parent,
            expected_development_baseline_record=development_baseline["record"],
        )
        _validate_full_evidence_against_development_baseline(
            development_baseline,
            runs,
            verification,
        )
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
            "development_baseline": development_baseline,
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


def write_immutable_development_baseline(
    path: str | Path, payload: dict[str, Any]
) -> str:
    baseline = _object(payload, "development baseline")
    _verify_payload_hash(baseline, "development baseline")
    if (
        baseline.get("schema") != DEVELOPMENT_BASELINE_SCHEMA
        or baseline.get("protocol") != DEVELOPMENT_BASELINE_PROTOCOL
        or baseline.get("status") != "retained_baseline"
        or baseline.get("confirmation_evidence_opened") is not False
        or baseline.get("retention_independent_of_promotion") is not True
        or baseline.get("can_override_promotion_gates") is not False
        or baseline.get("claim_boundary") != DEVELOPMENT_BASELINE_CLAIM
        or baseline.get("confirmation_authorization") != _confirmation_authorization()
        or baseline.get("output_contract")
        != {"exclusive_create": True, "overwrite": False, "mode": "0444"}
    ):
        raise EvidenceError(
            "development_baseline_contract_mismatch",
            "development baseline retention contract differs",
        )
    copied = _object(
        baseline.get("copied_checkpoint"), "development baseline copied checkpoint"
    )
    _expect_keys(
        copied,
        {"path", "sha256", "bytes", "mode"},
        "development baseline copied checkpoint",
    )
    copied_binding, copied_path, _ = _verify_binary_reference(
        {"path": copied["path"], "sha256": copied["sha256"]},
        "development baseline copied checkpoint",
        Path(path).parent,
    )
    copied_size = _integer(
        copied["bytes"],
        "development baseline copied checkpoint.bytes",
        minimum=1,
    )
    if (
        copied["mode"] != "0444"
        or copied_size != copied_binding["bytes"]
        or stat.S_IMODE(copied_path.stat().st_mode) != 0o444
    ):
        raise EvidenceError(
            "development_baseline_copy_mismatch",
            "development baseline checkpoint copy differs",
        )
    encoded = canonical_json_bytes(baseline) + b"\n"
    return str(_write_immutable_binary(path, encoded)["sha256"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Frozen R12 ACW evidence manifest"
    )
    parser.add_argument("--out", required=True, help="New immutable decision JSON")
    parser.add_argument(
        "--development-baseline",
        help="Frozen development baseline required for full adjudication",
    )
    parser.add_argument(
        "--baseline-checkpoint-out",
        help="Freeze development-only evidence and copy the selected checkpoint here",
    )
    parser.add_argument(
        "--qualify-direct-state",
        action="store_true",
        help="Verify only the three direct-state diagnostics before scored fitting",
    )
    parser.add_argument(
        "--phase2-authorization-out",
        help="Immutable phase-2 authorization emitted only when all diagnostics pass",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.lexists(args.out):
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if args.qualify_direct_state:
        if (
            not args.phase2_authorization_out
            or args.baseline_checkpoint_out
            or args.development_baseline
        ):
            raise SystemExit(
                "--qualify-direct-state requires --phase2-authorization-out and "
                "forbids baseline arguments"
            )
        if os.path.lexists(args.phase2_authorization_out):
            raise SystemExit(
                "refusing to overwrite existing phase-2 authorization: "
                f"{args.phase2_authorization_out}"
            )
        decision, authorization = qualify_direct_state(
            args.manifest, args.out, args.phase2_authorization_out
        )
        print(
            json.dumps(
                {
                    "decision": decision["decision"],
                    "decision_payload_sha256": decision["payload_sha256"],
                    "phase2_authorized": authorization is not None,
                },
                sort_keys=True,
            )
        )
        return 0 if authorization is not None else 2
    if args.baseline_checkpoint_out:
        if args.phase2_authorization_out:
            raise SystemExit(
                "--phase2-authorization-out is valid only with --qualify-direct-state"
            )
        if args.development_baseline:
            raise SystemExit(
                "--development-baseline and --baseline-checkpoint-out are exclusive"
            )
        if os.path.lexists(args.baseline_checkpoint_out):
            raise SystemExit(
                "refusing to overwrite existing baseline checkpoint: "
                f"{args.baseline_checkpoint_out}"
            )
        baseline = freeze_development_baseline(
            args.manifest, args.baseline_checkpoint_out
        )
        file_sha256 = write_immutable_development_baseline(args.out, baseline)
        print(
            json.dumps(
                {
                    "status": baseline["status"],
                    "selected_arm": baseline["selection"]["selected_arm"],
                    "baseline_file_sha256": file_sha256,
                    "baseline_payload_sha256": baseline["payload_sha256"],
                    "checkpoint_sha256": baseline["copied_checkpoint"]["sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.development_baseline:
        raise SystemExit(
            "full adjudication requires --development-baseline frozen before confirmation"
        )
    decision = adjudicate_manifest(args.manifest, args.development_baseline)
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
