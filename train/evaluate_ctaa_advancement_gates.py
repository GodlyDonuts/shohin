#!/usr/bin/env python3
"""Audit CTAA gate inputs and reject unresolved provenance contracts.

Revision 1 of this file accepted a caller-authored sidecar containing family
labels, parent mappings, pass bits, and a bootstrap seed.  That is an invalid
custody boundary: all of those values can be selected after outcomes are
visible. Assessment revision 2 retains the per-family oracle provenance and
typed active-step outcomes needed to reconstruct them independently.

This revision therefore does two things only:

1. validate and recompute every claim that current immutable artifacts can
   support; and
2. fail with ``UnresolvedContractError`` before any advancement statistic is
   computed.

The evaluator must remain closed until upstream producers commit the missing
provenance before outcome access.  It never accepts a metadata sidecar or a
caller-selected bootstrap seed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ctaa_access_registry import (
    ACCESS_SPEND,
    ASSESSMENT_COMMIT,
    canonical_json_bytes,
    verify_registry,
    verify_registry_events,
)
from ctaa_bootstrap_seed_receipt import validate_receipt as validate_bootstrap_receipt
from ctaa_assessment_source_bundle import validate_assessment_source_bundle
from ctaa_intervention_protocol import MANDATORY_OPERATIONS, RUNTIME_PANEL_SIZE
from ctaa_runtime_bundle import (
    RuntimeBundleError,
    make_runtime_bundle,
    read_runtime_plan,
    validate_runtime_bundle,
)
from ctaa_runtime_evidence import read_runtime_evidence
from ctaa_runtime_execution_set import read_runtime_execution_set_with_replay
from ctaa_statistical_gate_spec import (
    StatisticalGateBindings,
    StatisticalGateSpecError,
    read_signed_statistical_gate_spec_with_sha,
)
from ctaa_runtime_plan_replay import load_runtime_replay_rows, replay_runtime_plan


ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v2"
FINITE_AUDIT_SCHEMA = "r12_ctaa_v2_finite_core_evaluation_v1"
RESOURCE_PROFILE_SCHEMA = "r12_ctaa_v2_resource_profile_v2"
CAPACITY_AUDIT_SCHEMA = "ctaa_matched_core_preflight_v1"
IMMUTABLE_PREFLIGHT_SCHEMA = "r12_ctaa_v2_immutable_artifact_preflight_v1"
ASSESSMENT_CLAIM_SCHEMA = "r12_ctaa_signed_assessment_claim_v5"
ASSESSMENT_CLAIM_KEYS = {"payload", "signature"}
ASSESSMENT_CLAIM_PAYLOAD_KEYS = {
    "schema",
    "registry_id",
    "access_id",
    "spend_event_id",
    "commit_event_id",
    "partition",
    "manifest_sha256",
    "board_sha256",
    "run_plan_sha256",
    "run_contract_sha256",
    "bootstrap_seed_receipt_sha256",
    "bootstrap_seed",
    "runtime_bundle_sha256",
    "execution_set_file_sha256",
    "execution_set_sha256",
    "statistical_gate_spec_file_sha256",
    "gate_spec_sha256",
    "assessment_source_bundle_sha256",
    "assessment_source_manifest_sha256",
    "python_interpreter_sha256",
    "bwrap_executable_sha256",
    "expected_previous_hash",
    "assessment_output",
    "assessor_argv_sha256",
    "signing_public_key",
}

ARMS = (
    "ctaa_closure",
    "oprc_closure",
    "ctaa_no_closure",
    "ctaa_shuffled_closure",
)
DATASETS = ("base", "intervention")
SEMANTIC_AXES = ("train", "development", "confirmation")
FACTORIAL_CELLS = frozenset({"iii", "iih", "ihi", "ihh", "hii", "hih", "hhi", "hhh"})
PROGRAM_CLASSES = frozenset(
    {"stable_rank_two", "implicit_final_collapse", "explicit_final_collapse"}
)
DEPTHS = (16, 32)
RENDERERS = frozenset(range(16))
EXPECTED_PER_FACTORIAL_CLASS_DEPTH = 576
EXPECTED_BASE_FAMILIES = 27_648
EXPECTED_INTERVENTION_FAMILIES = 12_960
STRICT_SYSTEM_PARAMETER_LIMIT = 149_999_999
RESOURCE_ARMS = ("closure_feature", "outer_product_control")
RESOURCE_PHASES = (
    "curriculum_selection",
    "forward",
    "backward",
    "optimizer_step",
    "compiler_training",
    "inference",
)
RESOURCE_DEPTHS = (1, 16, 32, 39)
RESOURCE_OBSERVATION_SCHEMA = "r12_ctaa_v2_resource_observation_v1"
RESOURCE_COMPARISON_SCHEMA = "r12_ctaa_v2_matched_resource_comparison_v1"
RESOURCE_SHARED_BINDINGS = (
    "trunk_checkpoint_sha256",
    "qualified_compiler_checkpoint_sha256",
    "compiler_initial_adapter_sha256",
    "tokenizer_sha256",
    "compiler_training_source_sha256",
    "atomic_training_source_sha256",
    "closure_training_source_sha256",
    "curriculum_selection_plan_sha256",
    "admission_device",
)

ASSESSMENT_KEYS = {
    "schema",
    "partition",
    "manifest_sha256",
    "access",
    "oracle_sha256",
    "runs",
    "capability_gate_computed",
}
ASSESSMENT_ACCESS_KEYS = {
    "schema",
    "registry_id",
    "registry_head_receipt_sha256",
    "registry_head_entry_hash",
    "access_event_payload_sha256",
    "access_id",
    "partition",
    "manifest_sha256",
    "board_sha256",
    "run_contract_sha256",
    "runtime_bundle_sha256",
    "assessment_claim_sha256",
    "execution_set_file_sha256",
    "execution_set_sha256",
    "statistical_gate_spec_file_sha256",
    "gate_spec_sha256",
    "bootstrap_seed_receipt_sha256",
    "bootstrap_seed",
    "access",
}
RUN_KEYS = {"seed", "arm", "dataset", "evidence_commitment", "scores"}
SCORE_KEYS = {
    "overall",
    "by_factorial_cell",
    "by_program_class",
    "by_depth",
    "by_renderer",
    "by_shift_order",
    "factorial_main_effects",
    "by_action_active_prefix_accuracy",
    "by_semantic_action_active_prefix_accuracy",
    "by_action_rank_active_prefix_accuracy",
    "by_step_quartile_active_prefix_accuracy",
    "intervention_relation_correct",
    "family_scores",
}
FAMILY_KEYS = {
    "family_id",
    "cluster_family_id",
    "parent_family_id",
    "relation",
    "expected_trace_equal",
    "expected_terminal_equal",
    "observed_trace_equal",
    "observed_terminal_equal",
    "relation_correct",
    "factorial_cell",
    "shift_order",
    "program_class",
    "depth",
    "renderer",
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
    "active_steps_correct",
    "active_steps_total",
    "active_step_outcomes",
}
BOOLEAN_FAMILY_KEYS = {
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
    "halt_valid",
    "route_agreement",
    "prefix_exact",
    "terminal_exact",
    "answer_exact",
}
COMMITMENT_SHA_KEYS = {
    "program_predictions_sha256",
    "compiler_sha256",
    "program_source_sha256",
    "query_source_sha256",
    "packet_index_sha256",
    "execution_sha256",
    "core_sha256",
    "query_predictions_sha256",
    "answers_sha256",
    "evidence_sha256",
}
SHARED_COMMITMENT_KEYS = {
    "program_predictions_sha256",
    "compiler_sha256",
    "program_source_sha256",
    "query_source_sha256",
    "packet_index_sha256",
}
SHARED_FRONTEND_KEYS = {
    "packet_valid",
    "cards_exact",
    "binding_exact",
    "initial_exact",
    "stop_exact",
    "schedule_exact",
    "program_exact",
    "query_exact",
}

UNRESOLVED_CONTRACTS = (
    "the assessor's binding_exact field aliases action-card equality instead of independently measuring opcode-to-card binding as required by the signed statistical specification",
    "the final gate validates sealed-assessor family rows but does not independently reconstruct all forty raw evidence/oracle scores",
    "no capability-time signed receipt binds the measured six-phase resource matrix and complete intervention panel to the assessed manifest and frozen cores",
    "the unmocked execution-set-to-final-gate integration and Linux bubblewrap custody smoke remain unverified",
)


class UnresolvedContractError(RuntimeError):
    """The available immutable schemas cannot safely support advancement."""

    def __init__(self, audit: Mapping[str, object]):
        self.audit = dict(audit)
        blockers = self.audit.get("unresolved_contracts", [])
        super().__init__(
            "CTAA advancement contract is unresolved: " + " | ".join(map(str, blockers))
        )


def _open_parent_directory(path: Path, label: str) -> tuple[int, str]:
    raw = os.path.abspath(os.fspath(path))
    if "\x00" in raw or raw == "/":
        raise ValueError(f"CTAA {label} path differs")
    components = raw.split("/")[1:]
    if any(component in ("", ".", "..") for component in components):
        raise ValueError(f"CTAA {label} path differs")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open("/", flags)
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                raise
            except OSError as error:
                raise ValueError(
                    f"CTAA {label} parent is missing or symlinked"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ValueError(f"CTAA {label} parent is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, components[-1]
    except Exception:
        os.close(descriptor)
        raise


def _read_file_once(
    path: Path,
    label: str,
    *,
    require_read_only: bool = True,
) -> bytes:
    parent_descriptor, name = _open_parent_directory(Path(path), label)
    descriptor = -1
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        os.close(parent_descriptor)
        raise
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA {label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (require_read_only and metadata.st_mode & 0o222)
        or metadata.st_nlink != 1
    ):
        os.close(parent_descriptor)
        raise ValueError(f"CTAA {label} is not a single-link immutable file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        raise ValueError(f"CTAA {label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    if (
        before.st_dev != metadata.st_dev
        or before.st_ino != metadata.st_ino
        or before.st_size != metadata.st_size
        or before.st_mtime_ns != metadata.st_mtime_ns
        or before.st_ctime_ns != metadata.st_ctime_ns
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or (require_read_only and after.st_mode & 0o222)
        or after.st_nlink != 1
    ):
        raise ValueError(f"CTAA {label} changed while being read")
    return b"".join(chunks)


def _load_object_with_raw(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"CTAA {label} contains duplicate JSON key {key}")
            result[key] = item
        return result

    raw = _read_file_once(path, label)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CTAA {label} JSON differs") from error
    if not isinstance(value, dict):
        raise ValueError(f"CTAA {label} is not a JSON object")
    return value, raw


def _load_object(path: Path, label: str) -> dict[str, object]:
    value, _ = _load_object_with_raw(path, label)
    return value


@contextmanager
def _immutable_snapshot(raw: bytes, label: str):
    """Expose one captured byte string to path-only legacy verifiers."""

    snapshot_root = os.path.realpath(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(
        prefix="ctaa-gate-snapshot-", dir=snapshot_root
    ) as directory:
        path = Path(directory) / label
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short CTAA snapshot write")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        yield path


def _load_registry_public_key(path: Path) -> bytes:
    raw = _read_file_once(path, "registry public key")
    if len(raw) == 32:
        return raw
    try:
        text = raw.decode("ascii").strip()
        key = bytes.fromhex(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("CTAA registry public key differs") from error
    if len(key) != 32 or text != key.hex():
        raise ValueError("CTAA registry public key differs")
    return key


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"CTAA {label} SHA-256 commitment differs")
    return str(value)


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"CTAA {label} Boolean differs")
    return bool(value)


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise ValueError(f"CTAA {label} integer differs")
    return int(value)


def _mean(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_steps = sum(int(row["active_steps_total"]) for row in rows)
    return {
        "rows": len(rows),
        **{
            metric: _mean([bool(row[metric]) for row in rows])
            for metric in BOOLEAN_FAMILY_KEYS
        },
        "active_prefix_step_accuracy": (
            sum(int(row["active_steps_correct"]) for row in rows) / total_steps
            if total_steps
            else 0.0
        ),
    }


def _strata(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(values) for name, values in sorted(grouped.items())}


def _factorial_effects(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, position in {"semantic": 0, "renderer": 1, "lexical": 2}.items():
        inherited = [
            bool(row["prefix_exact"])
            for row in rows
            if str(row["factorial_cell"])[position] == "i"
        ]
        held_out = [
            bool(row["prefix_exact"])
            for row in rows
            if str(row["factorial_cell"])[position] == "h"
        ]
        result[name] = {
            "inherited": _mean(inherited),
            "held_out": _mean(held_out),
            "held_out_minus_inherited": _mean(held_out) - _mean(inherited),
            "inherited_families": len(inherited),
            "held_out_families": len(held_out),
        }
    return result


def _recompute_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_opcode: dict[str, list[bool]] = defaultdict(list)
    by_semantic: dict[str, list[bool]] = defaultdict(list)
    by_rank: dict[str, list[bool]] = defaultdict(list)
    by_quartile: dict[str, list[bool]] = defaultdict(list)
    by_relation: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for outcome in row["active_step_outcomes"]:
            by_opcode[str(outcome["opcode"])].append(bool(outcome["correct"]))
            semantic = json.dumps(outcome["semantic_action"], separators=(",", ":"))
            by_semantic[semantic].append(bool(outcome["correct"]))
            by_rank[str(outcome["action_rank"])].append(bool(outcome["correct"]))
            by_quartile[str(outcome["quartile"])].append(bool(outcome["correct"]))
        if row["relation"] is not None:
            by_relation[str(row["relation"])].append(bool(row["relation_correct"]))
    return {
        "overall": _aggregate(rows),
        "by_factorial_cell": _strata(rows, "factorial_cell"),
        "by_program_class": _strata(rows, "program_class"),
        "by_depth": _strata(rows, "depth"),
        "by_renderer": _strata(rows, "renderer"),
        "by_shift_order": _strata(rows, "shift_order"),
        "factorial_main_effects": _factorial_effects(rows),
        "by_action_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(by_opcode.items())
        },
        "by_semantic_action_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(by_semantic.items())
        },
        "by_action_rank_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(by_rank.items())
        },
        "by_step_quartile_active_prefix_accuracy": {
            key: _mean(values) for key, values in sorted(by_quartile.items())
        },
        "intervention_relation_correct": {
            key: _mean(values) for key, values in sorted(by_relation.items())
        },
        "family_scores": list(rows),
    }


def _validate_commitment(value: object, rows: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("CTAA evidence commitment differs")
    for key in COMMITMENT_SHA_KEYS:
        _require_sha256(value.get(key), f"evidence {key}")
    if not isinstance(value.get("core_kind"), str) or not value["core_kind"]:
        raise ValueError("CTAA evidence core kind differs")
    if value.get("rows") != rows or value.get("oracle_access") != 0:
        raise ValueError("CTAA evidence rows or oracle-access custody differs")
    valid = _require_int(value.get("valid_packets"), "valid packet count")
    if valid > rows or any(
        value.get(key) != valid
        for key in ("executed_rows", "queried_rows", "answered_rows")
    ):
        raise ValueError("CTAA evidence stage counts differ")
    return value


def _validate_family_rows(
    value: object,
    *,
    dataset: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("CTAA family outcomes are missing")
    expected = (
        EXPECTED_BASE_FAMILIES if dataset == "base" else EXPECTED_INTERVENTION_FAMILIES
    )
    if len(value) != expected:
        raise ValueError(f"CTAA {dataset} family count differs")
    rows: dict[str, dict[str, object]] = {}
    geometry: Counter[tuple[str, str, int]] = Counter()
    for row in value:
        # Extra caller-authored mappings, labels, or bootstrap fields are not
        # extensions of assessment_v2; they are rejected as forged metadata.
        if not isinstance(row, dict) or set(row) != FAMILY_KEYS:
            raise ValueError(
                "CTAA assessment_v2 family schema differs or contains forged labels"
            )
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id or family_id in rows:
            raise ValueError("CTAA family ID identity differs")
        if row.get("factorial_cell") not in FACTORIAL_CELLS:
            raise ValueError("CTAA factorial-cell stratum differs")
        shift_order = _require_int(row.get("shift_order"), "family shift order")
        if shift_order != str(row["factorial_cell"]).count("h"):
            raise ValueError("CTAA family shift order differs")
        if row.get("program_class") not in PROGRAM_CLASSES:
            raise ValueError("CTAA program-class stratum differs")
        depth = _require_int(row.get("depth"), "family depth", minimum=1)
        renderer = _require_int(row.get("renderer"), "family renderer")
        if depth not in DEPTHS or renderer not in RENDERERS:
            raise ValueError("CTAA depth or renderer stratum differs")
        for key in BOOLEAN_FAMILY_KEYS:
            _require_bool(row.get(key), f"family {key}")
        total = _require_int(
            row.get("active_steps_total"), "active-step total", minimum=1
        )
        correct = _require_int(row.get("active_steps_correct"), "active-step correct")
        if total != depth or correct > total:
            raise ValueError("CTAA active-prefix family geometry differs")
        outcomes = row.get("active_step_outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != depth:
            raise ValueError("CTAA active-step outcome geometry differs")
        outcome_correct = 0
        for step, outcome in enumerate(outcomes):
            if not isinstance(outcome, dict) or set(outcome) != {
                "step",
                "opcode",
                "semantic_action",
                "action_rank",
                "quartile",
                "correct",
            }:
                raise ValueError("CTAA active-step outcome schema differs")
            opcode = _require_int(outcome.get("opcode"), "active-step opcode")
            action = outcome.get("semantic_action")
            rank = _require_int(outcome.get("action_rank"), "active-step action rank")
            quartile = _require_int(
                outcome.get("quartile"), "active-step quartile", minimum=1
            )
            if (
                outcome.get("step") != step
                or opcode not in range(4)
                or not isinstance(action, list)
                or len(action) != 3
                or any(type(item) is not int or item not in range(3) for item in action)
                or rank != len(set(action))
                or quartile != min(3, (4 * step) // depth) + 1
                or type(outcome.get("correct")) is not bool
            ):
                raise ValueError("CTAA active-step outcome value differs")
            outcome_correct += int(outcome["correct"])
        if outcome_correct != correct:
            raise ValueError("CTAA active-step aggregate differs")
        cluster = row.get("cluster_family_id")
        parent = row.get("parent_family_id")
        relation = row.get("relation")
        relation_fields = (
            row.get("expected_trace_equal"),
            row.get("expected_terminal_equal"),
            row.get("observed_trace_equal"),
            row.get("observed_terminal_equal"),
            row.get("relation_correct"),
        )
        if dataset == "base":
            if (
                cluster != family_id
                or parent is not None
                or relation is not None
                or any(value is not None for value in relation_fields)
            ):
                raise ValueError("CTAA base family intervention provenance differs")
        elif (
            not isinstance(cluster, str)
            or not isinstance(parent, str)
            or not parent
            or cluster != parent
            or not isinstance(relation, str)
            or not relation
            or any(type(value) is not bool for value in relation_fields)
        ):
            raise ValueError("CTAA intervention family provenance differs")
        rows[family_id] = row
        geometry[(str(row["factorial_cell"]), str(row["program_class"]), depth)] += 1
    if dataset == "base":
        expected_geometry = {
            (cell, program_class, depth): EXPECTED_PER_FACTORIAL_CLASS_DEPTH
            for cell in FACTORIAL_CELLS
            for program_class in PROGRAM_CLASSES
            for depth in DEPTHS
        }
        if dict(geometry) != expected_geometry:
            raise ValueError("CTAA factorial/class/depth family geometry differs")
    return rows


def _validate_assessment(
    report: dict[str, object],
) -> tuple[
    list[int],
    dict[tuple[int, str, str], dict[str, object]],
    dict[tuple[int, str, str], dict[str, dict[str, object]]],
]:
    if set(report) != ASSESSMENT_KEYS or report.get("schema") != ASSESSMENT_SCHEMA:
        raise ValueError(
            "CTAA assessment_v2 schema differs or contains outcome-aware metadata"
        )
    if report.get("partition") not in {"development", "confirmation"}:
        raise ValueError("CTAA assessment partition differs")
    _require_sha256(report.get("manifest_sha256"), "assessment manifest")
    if report.get("capability_gate_computed") is not False:
        raise ValueError("CTAA assessment already asserts a capability gate")
    access = report.get("access")
    if (
        not isinstance(access, dict)
        or set(access) != ASSESSMENT_ACCESS_KEYS
        or access.get("schema") != "r12_ctaa_v2_assessment_access_v7"
        or access.get("partition") != report["partition"]
        or access.get("manifest_sha256") != report["manifest_sha256"]
        or access.get("access") != 1
    ):
        raise ValueError("CTAA assessment access receipt differs")
    for key in (
        "registry_head_receipt_sha256",
        "registry_head_entry_hash",
        "access_event_payload_sha256",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_bundle_sha256",
        "assessment_claim_sha256",
        "execution_set_file_sha256",
        "execution_set_sha256",
        "bootstrap_seed_receipt_sha256",
    ):
        _require_sha256(access.get(key), f"assessment access {key}")
    _require_int(access.get("bootstrap_seed"), "assessment access bootstrap seed")
    for key in ("registry_id", "access_id"):
        if not isinstance(access.get(key), str) or not access[key]:
            raise ValueError("CTAA assessment access identity differs")
    runs = report.get("runs")
    if not isinstance(runs, dict) or len(runs) != 40:
        raise ValueError(
            "CTAA assessment requires exactly five paired seeds by four arms by two datasets"
        )

    indexed: dict[tuple[int, str, str], dict[str, object]] = {}
    families: dict[tuple[int, str, str], dict[str, dict[str, object]]] = {}
    for name, run in runs.items():
        if (
            not isinstance(name, str)
            or not isinstance(run, dict)
            or set(run) != RUN_KEYS
        ):
            raise ValueError("CTAA assessment run schema differs")
        seed = _require_int(run.get("seed"), "paired seed")
        arm = run.get("arm")
        dataset = run.get("dataset")
        if arm not in ARMS or dataset not in DATASETS:
            raise ValueError("CTAA assessment arm or dataset differs")
        key = (seed, str(arm), str(dataset))
        if key in indexed:
            raise ValueError("CTAA assessment duplicates a paired run")
        scores = run.get("scores")
        if not isinstance(scores, dict) or set(scores) != SCORE_KEYS:
            raise ValueError(
                "CTAA assessment score schema differs or contains forged metadata"
            )
        rows = _validate_family_rows(scores["family_scores"], dataset=str(dataset))
        recomputed_scores = _recompute_scores(list(rows.values()))
        if scores != recomputed_scores:
            raise ValueError("CTAA assessment aggregates differ from family outcomes")
        commitment = _validate_commitment(run.get("evidence_commitment"), len(rows))
        indexed[key] = {**run, "evidence_commitment": commitment}
        families[key] = rows

    seeds = sorted({seed for seed, _, _ in indexed})
    expected_lattice = {
        (seed, arm, dataset) for seed in seeds for arm in ARMS for dataset in DATASETS
    }
    if len(seeds) != 5 or set(indexed) != expected_lattice:
        raise ValueError("CTAA paired 5x4x2 run lattice differs")

    reference_ids = {
        dataset: set(families[(seeds[0], ARMS[0], dataset)]) for dataset in DATASETS
    }
    for dataset in DATASETS:
        for seed in seeds:
            commitments = []
            for arm in ARMS:
                key = (seed, arm, dataset)
                if set(families[key]) != reference_ids[dataset]:
                    raise ValueError("CTAA paired family ID identity differs")
                commitments.append(indexed[key]["evidence_commitment"])
            for commitment_key in SHARED_COMMITMENT_KEYS:
                if len({value[commitment_key] for value in commitments}) != 1:
                    raise ValueError(f"CTAA paired {commitment_key} identity differs")
            for family_id in reference_ids[dataset]:
                signatures = {
                    tuple(
                        families[(seed, arm, dataset)][family_id][field]
                        for field in SHARED_FRONTEND_KEYS
                    )
                    for arm in ARMS
                }
                if len(signatures) != 1:
                    raise ValueError("CTAA shared compiler outcomes differ across arms")

    for seed in seeds:
        for arm in ARMS:
            base_rows = families[(seed, arm, "base")]
            intervention_rows = families[(seed, arm, "intervention")]
            for child in intervention_rows.values():
                parent = base_rows.get(str(child["parent_family_id"]))
                if parent is None:
                    raise ValueError("CTAA intervention parent family is absent")
                expected_relation = (
                    bool(parent["prefix_exact"])
                    and bool(child["prefix_exact"])
                    and child["observed_trace_equal"] == child["expected_trace_equal"]
                    and child["observed_terminal_equal"]
                    == child["expected_terminal_equal"]
                )
                if child["relation_correct"] is not expected_relation:
                    raise ValueError("CTAA intervention relation outcome is forged")

    compiler_hashes = set()
    core_hashes = set()
    for seed in seeds:
        per_seed_compilers = set()
        for arm in ARMS:
            base = indexed[(seed, arm, "base")]["evidence_commitment"]
            intervention = indexed[(seed, arm, "intervention")]["evidence_commitment"]
            if (
                base["core_sha256"] != intervention["core_sha256"]
                or base["core_kind"] != intervention["core_kind"]
            ):
                raise ValueError("CTAA base/intervention frozen-core identity differs")
            core_hashes.add(str(base["core_sha256"]))
            per_seed_compilers.add(str(base["compiler_sha256"]))
            per_seed_compilers.add(str(intervention["compiler_sha256"]))
        if len(per_seed_compilers) != 1:
            raise ValueError("CTAA base/intervention compiler identity differs")
        compiler_hashes.update(per_seed_compilers)
    if len(compiler_hashes) != 5 or len(core_hashes) != 20:
        raise ValueError(
            "CTAA independently initialized compiler/core identity differs"
        )

    for dataset in DATASETS:
        for source_key in ("program_source_sha256", "query_source_sha256"):
            values = {
                indexed[(seed, arm, dataset)]["evidence_commitment"][source_key]
                for seed in seeds
                for arm in ARMS
            }
            if len(values) != 1:
                raise ValueError(f"CTAA sealed {dataset} {source_key} differs")
    return seeds, indexed, families


def _validate_finite_audits(
    paths: Sequence[Path],
    *,
    indexed: Mapping[tuple[int, str, str], dict[str, object]],
) -> dict[str, object]:
    expected_by_core = {
        str(run["evidence_commitment"]["core_sha256"]): (seed, arm)
        for (seed, arm, dataset), run in indexed.items()
        if dataset == "base"
    }
    if len(paths) != 20 or len(expected_by_core) != 20:
        raise ValueError(
            "CTAA requires exactly twenty uniquely core-bound finite audits"
        )
    audited = {}
    for path in paths:
        value, raw = _load_object_with_raw(path, "finite-domain audit")
        if value.get("schema") != FINITE_AUDIT_SCHEMA or value.get("board_access") != 0:
            raise ValueError("CTAA finite-domain audit schema or custody differs")
        core_sha = _require_sha256(value.get("core_sha256"), "finite-audit core")
        if core_sha not in expected_by_core or core_sha in audited:
            raise ValueError(
                "CTAA finite audit is duplicated or not bound to an assessed core"
            )
        seed, arm = expected_by_core[core_sha]
        commitment = indexed[(seed, arm, "base")]["evidence_commitment"]
        if value.get("core_kind") != commitment["core_kind"]:
            raise ValueError("CTAA finite-audit core kind differs")
        axes = value.get("axes")
        gates = value.get("gates")
        if not isinstance(axes, dict) or not isinstance(gates, dict):
            raise ValueError("CTAA finite-domain axis receipt differs")
        if set(axes) != set(SEMANTIC_AXES) or set(gates) != set(SEMANTIC_AXES):
            raise ValueError("CTAA finite-domain axis coverage differs")
        recomputed = {}
        for axis in SEMANTIC_AXES:
            result = axes[axis]
            if (
                not isinstance(result, dict)
                or result.get("atomic_cases") != 243
                or result.get("two_action_cases") != 2_187
            ):
                raise ValueError("CTAA finite-domain case geometry differs")
            passed = all(
                result.get(metric) == 1.0
                for metric in (
                    "atomic_exact",
                    "two_action_exact",
                    "composition_exact",
                    "route_agreement",
                )
            )
            if gates[axis] is not passed:
                raise ValueError("CTAA finite-domain producer pass bit is forged")
            recomputed[axis] = passed
        all_pass = all(recomputed.values())
        if value.get("all_gates_pass") is not all_pass:
            raise ValueError("CTAA finite-domain aggregate pass bit is forged")
        audited[core_sha] = {
            "path_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": seed,
            "arm": arm,
            "axis_pass": recomputed,
            "all_pass": all_pass,
        }
    if set(audited) != set(expected_by_core):
        raise ValueError("CTAA finite-domain audit coverage differs")
    return audited


def _resource_canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("CTAA resource receipt is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _validate_resource_profile(value: dict[str, object]) -> dict[str, object]:
    expected_report_keys = {
        "schema",
        "base_sha256",
        "base_step",
        "qualified_compiler_sha256",
        "qualified_memory_tensors",
        "artifact_bindings",
        "parameter_ledger",
        "core_parameters",
        "transition_flops",
        "state_contract",
        "evaluation_charge",
        "runtime",
        "measurements",
        "matched_arm_comparisons",
        "required_phases",
        "profile_depths",
        "board_seed_generated",
        "oracle_access",
        "all_static_gates_pass",
        "all_resource_gates_pass",
        "all_gates_pass",
    }
    if (
        set(value) != expected_report_keys
        or value.get("schema") != RESOURCE_PROFILE_SCHEMA
    ):
        raise ValueError("CTAA resource-profile schema differs")
    ledger = value.get("parameter_ledger")
    cores = value.get("core_parameters")
    flops = value.get("transition_flops")
    state = value.get("state_contract")
    charge = value.get("evaluation_charge")
    if not all(
        isinstance(item, dict) for item in (ledger, cores, flops, state, charge)
    ):
        raise ValueError("CTAA resource-profile structure differs")
    ledger_values = {
        key: _require_int(ledger.get(key), f"parameter {key}")
        for key in ("trunk", "compiler_adapter", "core", "total", "headroom")
    }
    ledger_exact = (
        ledger_values["total"]
        == ledger_values["trunk"]
        + ledger_values["compiler_adapter"]
        + ledger_values["core"]
        and ledger_values["headroom"]
        == STRICT_SYSTEM_PARAMETER_LIMIT - ledger_values["total"]
        and ledger_values["total"] <= STRICT_SYSTEM_PARAMETER_LIMIT
    )
    core_exact = (
        cores.get("closure_feature") == cores.get("outer_product_control") == 107_753
        and cores.get("exactly_matched") is True
        and ledger_values["core"] == 107_753
    )
    charged = max(
        _require_int(flops.get("closure_feature_analytic"), "treatment FLOPs"),
        _require_int(flops.get("outer_product_control_analytic"), "control FLOPs"),
    )
    flop_exact = (
        flops.get("charged_per_call") == charged
        and flops.get("treatment_padding_charge")
        == charged - flops.get("closure_feature_analytic")
        and flops.get("control_padding_charge")
        == charged - flops.get("outer_product_control_analytic")
        and charge.get("dual_route_core_calls_per_row") == 123
        and charge.get("charged_core_flops_per_row") == 123 * charged
    )
    state_exact = (
        state.get("matched_across_arms") is True
        and state.get("hard_packet_bytes_per_row") == 56
        and state.get("semantic_recurrent_state_bytes") == 3
        and state.get("implementation_recurrent_state_int64_bytes") == 24
        and state.get("halt_state_bytes") == 1
    )
    static_pass = (
        ledger_exact
        and core_exact
        and flop_exact
        and state_exact
        and value.get("qualified_memory_tensors") == 63
    )
    if value.get("all_static_gates_pass") is not static_pass:
        raise ValueError("CTAA resource-profile producer pass bit is forged")
    if (
        value.get("board_seed_generated") is not False
        or value.get("oracle_access") != 0
    ):
        raise ValueError("CTAA resource-profile custody differs")

    bindings = value.get("artifact_bindings")
    expected_binding_keys = {
        *RESOURCE_SHARED_BINDINGS,
        "core_checkpoint_sha256",
        "core_kind",
    }
    if not isinstance(bindings, dict) or set(bindings) != set(RESOURCE_ARMS):
        raise ValueError("CTAA resource artifact-binding arms differ")
    normalized_bindings: dict[str, dict[str, object]] = {}
    for arm in RESOURCE_ARMS:
        arm_bindings = bindings.get(arm)
        if (
            not isinstance(arm_bindings, dict)
            or set(arm_bindings) != expected_binding_keys
        ):
            raise ValueError("CTAA resource artifact-binding schema differs")
        if arm_bindings.get("core_kind") != arm:
            raise ValueError("CTAA resource core binding differs")
        for key, item in arm_bindings.items():
            if key.endswith("_sha256"):
                _require_sha256(item, f"resource binding {key}")
        if (
            not isinstance(arm_bindings.get("admission_device"), str)
            or not arm_bindings["admission_device"]
        ):
            raise ValueError("CTAA resource admission device differs")
        normalized_bindings[arm] = dict(arm_bindings)
    if any(
        normalized_bindings[RESOURCE_ARMS[0]][key]
        != normalized_bindings[RESOURCE_ARMS[1]][key]
        for key in RESOURCE_SHARED_BINDINGS
    ):
        raise ValueError("CTAA resource matched-arm artifact binding differs")
    if normalized_bindings[RESOURCE_ARMS[0]]["trunk_checkpoint_sha256"] != value.get(
        "base_sha256"
    ) or normalized_bindings[RESOURCE_ARMS[0]][
        "qualified_compiler_checkpoint_sha256"
    ] != value.get("qualified_compiler_sha256"):
        raise ValueError("CTAA resource top-level artifact binding differs")

    if value.get("required_phases") != list(RESOURCE_PHASES) or value.get(
        "profile_depths"
    ) != list(RESOURCE_DEPTHS):
        raise ValueError("CTAA resource measurement plan differs")
    observation_keys = {
        "schema",
        "arm",
        "phase",
        "active_depth",
        "device",
        "batch_size",
        "repeats",
        "warmup_count",
        "elapsed_ns",
        "milliseconds_per_iteration",
        "rows_per_second",
        "peak_allocated_bytes",
        "work_units_per_iteration",
        "bindings",
        "observation_sha256",
    }
    measurements = value.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("CTAA resource observations differ")
    expected_identities = {
        (arm, phase, depth)
        for arm in RESOURCE_ARMS
        for phase in RESOURCE_PHASES
        for depth in RESOURCE_DEPTHS
    }
    observed: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in measurements:
        if not isinstance(row, dict) or set(row) != observation_keys:
            raise ValueError("CTAA resource observation schema differs")
        identity = (row.get("arm"), row.get("phase"), row.get("active_depth"))
        if identity not in expected_identities or identity in observed:
            raise ValueError("CTAA resource observation identity differs")
        arm, phase, depth = identity
        assert (
            isinstance(arm, str) and isinstance(phase, str) and isinstance(depth, int)
        )
        if row.get("schema") != RESOURCE_OBSERVATION_SCHEMA:
            raise ValueError("CTAA resource observation version differs")
        if row.get("bindings") != normalized_bindings[arm]:
            raise ValueError("CTAA resource observation artifact binding differs")
        unhashed = {
            key: item for key, item in row.items() if key != "observation_sha256"
        }
        if row.get("observation_sha256") != _resource_canonical_sha256(unhashed):
            raise ValueError("CTAA resource observation hash differs")
        integers = {
            key: _require_int(row.get(key), f"resource observation {key}", minimum=1)
            for key in (
                "batch_size",
                "repeats",
                "warmup_count",
                "elapsed_ns",
                "work_units_per_iteration",
            )
        }
        expected_ms = integers["elapsed_ns"] / integers["repeats"] / 1_000_000.0
        expected_rows = (
            integers["batch_size"]
            * integers["repeats"]
            * 1_000_000_000.0
            / integers["elapsed_ns"]
        )
        for key, expected in (
            ("milliseconds_per_iteration", expected_ms),
            ("rows_per_second", expected_rows),
        ):
            item = row.get(key)
            if (
                type(item) not in {int, float}
                or not math.isfinite(float(item))
                or not math.isclose(float(item), expected, rel_tol=1e-12, abs_tol=0.0)
            ):
                raise ValueError("CTAA resource observation derived timing differs")
        peak = _require_int(
            row.get("peak_allocated_bytes"),
            "resource observation peak memory",
            minimum=0,
        )
        device = row.get("device")
        admission_device = normalized_bindings[arm]["admission_device"]
        if (
            not isinstance(device, str)
            or (
                phase == "curriculum_selection"
                and device not in {"cpu", admission_device}
            )
            or (phase != "curriculum_selection" and device != admission_device)
        ):
            raise ValueError("CTAA resource observation device differs")
        if device.startswith("cuda") and peak == 0:
            raise ValueError("CTAA CUDA resource observation memory is absent")
        observed[(arm, phase, depth)] = row
    if set(observed) != expected_identities:
        raise ValueError("CTAA resource observation matrix is incomplete")

    comparison_keys = {
        "schema",
        "phase",
        "active_depth",
        "treatment_observation_sha256",
        "control_observation_sha256",
        "shared_bindings_exact",
        "work_units_exact",
        "batch_size_exact",
        "repeats_exact",
        "warmup_count_exact",
        "elapsed_ratio_control_over_treatment",
        "peak_bytes_ratio_control_over_treatment",
    }
    comparisons = value.get("matched_arm_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(
        RESOURCE_PHASES
    ) * len(RESOURCE_DEPTHS):
        raise ValueError("CTAA resource comparison matrix is incomplete")
    expected_comparisons = []
    for phase in RESOURCE_PHASES:
        for depth in RESOURCE_DEPTHS:
            treatment = observed[(RESOURCE_ARMS[0], phase, depth)]
            control = observed[(RESOURCE_ARMS[1], phase, depth)]
            treatment_peak = int(treatment["peak_allocated_bytes"])
            control_peak = int(control["peak_allocated_bytes"])
            expected_comparisons.append(
                {
                    "schema": RESOURCE_COMPARISON_SCHEMA,
                    "phase": phase,
                    "active_depth": depth,
                    "treatment_observation_sha256": treatment["observation_sha256"],
                    "control_observation_sha256": control["observation_sha256"],
                    "shared_bindings_exact": all(
                        treatment["bindings"][key] == control["bindings"][key]  # type: ignore[index]
                        for key in RESOURCE_SHARED_BINDINGS
                    ),
                    "work_units_exact": treatment["work_units_per_iteration"]
                    == control["work_units_per_iteration"],
                    "batch_size_exact": treatment["batch_size"]
                    == control["batch_size"],
                    "repeats_exact": treatment["repeats"] == control["repeats"],
                    "warmup_count_exact": treatment["warmup_count"]
                    == control["warmup_count"],
                    "elapsed_ratio_control_over_treatment": float(control["elapsed_ns"])
                    / float(treatment["elapsed_ns"]),
                    "peak_bytes_ratio_control_over_treatment": (
                        float(control_peak) / treatment_peak
                        if treatment_peak
                        else (1.0 if control_peak == 0 else None)
                    ),
                }
            )
    for actual, expected in zip(comparisons, expected_comparisons, strict=True):
        if (
            not isinstance(actual, dict)
            or set(actual) != comparison_keys
            or actual != expected
            or not all(
                actual[key] is True
                for key in (
                    "shared_bindings_exact",
                    "work_units_exact",
                    "batch_size_exact",
                    "repeats_exact",
                    "warmup_count_exact",
                )
            )
        ):
            raise ValueError("CTAA resource matched-arm comparison differs")

    expected_runtime = {
        arm: {
            str(depth): observed[(arm, "inference", depth)] for depth in RESOURCE_DEPTHS
        }
        for arm in RESOURCE_ARMS
    }
    if value.get("runtime") != expected_runtime:
        raise ValueError("CTAA resource runtime projection differs")
    resource_pass = True
    if value.get("all_resource_gates_pass") is not resource_pass:
        raise ValueError("CTAA resource-profile producer resource pass bit is forged")
    if value.get("all_gates_pass") is not (static_pass and resource_pass):
        raise ValueError("CTAA resource-profile aggregate pass bit is forged")
    return {
        "base_sha256": _require_sha256(value.get("base_sha256"), "resource base"),
        "qualified_compiler_sha256": _require_sha256(
            value.get("qualified_compiler_sha256"), "qualified compiler"
        ),
        "parameter_ledger": ledger_values,
        "closure_feature_flops": flops["closure_feature_analytic"],
        "outer_product_flops": flops["outer_product_control_analytic"],
        "all_static_gates_pass": static_pass,
        "all_resource_gates_pass": resource_pass,
        "artifact_bindings": normalized_bindings,
    }


def _validate_capacity_audit(value: dict[str, object]) -> dict[str, object]:
    if value.get("schema") != CAPACITY_AUDIT_SCHEMA:
        raise ValueError("CTAA matched-capacity audit schema differs")
    treatment = value.get("treatment")
    control = value.get("control")
    gates = value.get("gates")
    if not all(isinstance(item, dict) for item in (treatment, control, gates)):
        raise ValueError("CTAA matched-capacity audit structure differs")
    recomputed = {
        "parameters_exactly_matched": (
            value.get("treatment_parameters")
            == value.get("control_parameters")
            == 107_753
        ),
        "control_features_separate_all_pairs": value.get("unique_control_features")
        == 729,
        "closure_treatment_optimizes_exactly": treatment.get("exact_accuracy") == 1.0,
        "arbitrary_control_table_optimizes_exactly": control.get("exact_accuracy")
        == 1.0,
    }
    if gates != recomputed or value.get("all_gates_pass") is not all(
        recomputed.values()
    ):
        raise ValueError("CTAA matched-capacity producer pass bits are forged")
    return {
        "treatment_parameters": value["treatment_parameters"],
        "control_parameters": value["control_parameters"],
        "all_gates_pass": all(recomputed.values()),
    }


def _validate_immutable_preflight(
    value: dict[str, object],
    *,
    resources: Mapping[str, object],
    capacity: Mapping[str, object],
) -> dict[str, object]:
    if value.get("schema") != IMMUTABLE_PREFLIGHT_SCHEMA:
        raise ValueError("CTAA immutable-preflight schema differs")
    base = value.get("base")
    compiler = value.get("qualified_compiler")
    core = value.get("core_match")
    if not all(isinstance(item, dict) for item in (base, compiler, core)):
        raise ValueError("CTAA immutable-preflight structure differs")
    gates = (
        base.get("sha256") == resources["base_sha256"]
        and base.get("strict_missing_keys") == []
        and base.get("strict_unexpected_keys") == []
        and compiler.get("sha256") == resources["qualified_compiler_sha256"]
        and compiler.get("memory_tensors_loaded")
        == compiler.get("memory_tensors_present")
        == 63
        and value.get("parameter_ledger") == resources["parameter_ledger"]
        and core.get("treatment_parameters") == capacity["treatment_parameters"]
        and core.get("control_parameters") == capacity["control_parameters"]
        and core.get("treatment_flops") == resources["closure_feature_flops"]
        and core.get("control_flops") == resources["outer_product_flops"]
        and value.get("board_artifact_written") is False
        and value.get("jobs_launched") is False
        and value.get("production_seed_generated") is False
    )
    if value.get("all_gates_pass") is not gates:
        raise ValueError("CTAA immutable-preflight producer pass bit is forged")
    return {
        "base_sha256": base["sha256"],
        "qualified_compiler_sha256": compiler["sha256"],
        "all_gates_pass": gates,
    }


def _verification_key_bytes(
    key: bytes | Ed25519PublicKey,
) -> tuple[bytes, Ed25519PublicKey]:
    if isinstance(key, bytes):
        if len(key) != 32:
            raise ValueError("CTAA registry verification key differs")
        try:
            return key, Ed25519PublicKey.from_public_bytes(key)
        except ValueError as error:
            raise ValueError("CTAA registry verification key differs") from error
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("CTAA registry verification key differs")
    return (
        key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ),
        key,
    )


def _validate_signed_assessment_claim(
    *,
    path: Path,
    assessment_path: Path,
    assessment: Mapping[str, object],
    access: Mapping[str, object],
    spend_payload: Mapping[str, object],
    commit_payload: Mapping[str, object],
    registry_verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    claim, raw = _load_object_with_raw(path, "signed assessment claim")
    if set(claim) != ASSESSMENT_CLAIM_KEYS:
        raise ValueError("CTAA signed assessment claim schema differs")
    payload = claim.get("payload")
    signature = claim.get("signature")
    if (
        not isinstance(payload, dict)
        or set(payload) != ASSESSMENT_CLAIM_PAYLOAD_KEYS
        or payload.get("schema") != ASSESSMENT_CLAIM_SCHEMA
        or not isinstance(signature, str)
        or len(signature) != 128
        or any(character not in "0123456789abcdef" for character in signature)
        or raw != canonical_json_bytes(claim) + b"\n"
    ):
        raise ValueError("CTAA signed assessment claim schema differs")
    key_bytes, public_key = _verification_key_bytes(registry_verification_key)
    if payload.get("signing_public_key") != key_bytes.hex():
        raise ValueError("CTAA signed assessment claim key differs")
    try:
        public_key.verify(bytes.fromhex(signature), canonical_json_bytes(payload))
    except (InvalidSignature, ValueError) as error:
        raise ValueError("CTAA signed assessment claim signature differs") from error
    for key in (
        "manifest_sha256",
        "board_sha256",
        "run_plan_sha256",
        "run_contract_sha256",
        "bootstrap_seed_receipt_sha256",
        "runtime_bundle_sha256",
        "execution_set_file_sha256",
        "execution_set_sha256",
        "statistical_gate_spec_file_sha256",
        "gate_spec_sha256",
        "assessment_source_bundle_sha256",
        "assessment_source_manifest_sha256",
        "python_interpreter_sha256",
        "bwrap_executable_sha256",
        "expected_previous_hash",
        "assessor_argv_sha256",
    ):
        _require_sha256(payload.get(key), f"signed assessment claim {key}")
    expected = {
        "registry_id": access.get("registry_id"),
        "access_id": access.get("access_id"),
        "spend_event_id": spend_payload.get("event_id"),
        "commit_event_id": commit_payload.get("event_id"),
        "partition": assessment.get("partition"),
        "manifest_sha256": assessment.get("manifest_sha256"),
        "board_sha256": access.get("board_sha256"),
        "run_contract_sha256": access.get("run_contract_sha256"),
        "bootstrap_seed_receipt_sha256": access.get("bootstrap_seed_receipt_sha256"),
        "bootstrap_seed": access.get("bootstrap_seed"),
        "runtime_bundle_sha256": access.get("runtime_bundle_sha256"),
        "execution_set_file_sha256": access.get("execution_set_file_sha256"),
        "execution_set_sha256": access.get("execution_set_sha256"),
        "statistical_gate_spec_file_sha256": access.get(
            "statistical_gate_spec_file_sha256"
        ),
        "gate_spec_sha256": access.get("gate_spec_sha256"),
        "expected_previous_hash": spend_payload.get("previous_hash"),
        "assessment_output": str(Path(os.path.abspath(assessment_path))),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("CTAA signed assessment claim binding differs")
    return {
        "claim_sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }


def _validate_loaded_runtime_bundle_with_replay(
    bundle_value: Mapping[str, object],
    bundle_path: Path,
    *,
    run_contract: Mapping[str, object],
    program_path: Path,
    query_path: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    """Validate one captured bundle object and replay each member once."""

    bundle = validate_runtime_bundle(bundle_value, run_contract=run_contract)
    root = Path(os.path.abspath(bundle_path)).parent
    artifacts = []
    expected_attempts = RUNTIME_PANEL_SIZE * len(MANDATORY_OPERATIONS)
    for entry in bundle["entries"]:
        if not isinstance(entry, Mapping):
            raise RuntimeBundleError("CTAA runtime replay entry differs")
        plan_path = root / str(entry["runtime_plan_filename"])
        evidence_path = root / str(entry["runtime_evidence_filename"])
        if plan_path.parent != root or evidence_path.parent != root:
            raise RuntimeBundleError("CTAA runtime bundle member escapes package root")
        plan, plan_file_sha = read_runtime_plan(plan_path)
        if (
            plan_file_sha != entry["runtime_plan_file_sha256"]
            or plan.plan_sha256 != entry["runtime_plan_sha256"]
            or plan.bindings.training_seed != entry["training_seed"]
        ):
            raise RuntimeBundleError("CTAA runtime plan member differs")
        evidence = read_runtime_evidence(
            evidence_path,
            plan,
            expected_file_sha256=str(entry["runtime_evidence_file_sha256"]),
        )
        if evidence.get("evidence_sha256") != entry["runtime_evidence_sha256"]:
            raise RuntimeBundleError("CTAA runtime evidence member differs")
        try:
            replay_rows = load_runtime_replay_rows(
                plan=plan,
                program_path=program_path,
                query_path=query_path,
                tokenizer_path=tokenizer_path,
            )
            replay = replay_runtime_plan(plan, replay_rows)
        except ValueError as error:
            raise RuntimeBundleError(
                "CTAA runtime plan semantic replay failed"
            ) from error
        if (
            replay.plan_sha256 != entry["runtime_plan_sha256"]
            or replay.attempt_count != expected_attempts
        ):
            raise RuntimeBundleError("CTAA runtime plan semantic replay differs")
        artifacts.append(
            (
                plan,
                evidence,
                str(entry["runtime_plan_filename"]),
                plan_file_sha,
                str(entry["runtime_evidence_filename"]),
                str(entry["runtime_evidence_file_sha256"]),
            )
        )
    rebuilt = make_runtime_bundle(run_contract=run_contract, artifacts=artifacts)
    if rebuilt != bundle:
        raise RuntimeBundleError("CTAA runtime bundle member recomputation differs")
    return bundle


def _validate_signed_assessment_access(
    *,
    assessment: Mapping[str, object],
    assessment_path: Path,
    assessment_sha256: str,
    assessment_claim_path: Path,
    access_registry_path: Path,
    access_spend_head_receipt_path: Path,
    assessment_commit_head_receipt_path: Path,
    registry_verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    access = assessment.get("access")
    if not isinstance(access, dict):
        raise ValueError("CTAA signed assessment access differs")
    spend_receipt, spend_receipt_raw = _load_object_with_raw(
        access_spend_head_receipt_path, "access-spend head receipt"
    )
    commit_receipt, _commit_receipt_raw = _load_object_with_raw(
        assessment_commit_head_receipt_path, "assessment-commit head receipt"
    )
    if hashlib.sha256(spend_receipt_raw).hexdigest() != access.get(
        "registry_head_receipt_sha256"
    ):
        raise ValueError("CTAA assessment retained spend receipt differs")
    registry_raw = _read_file_once(
        access_registry_path,
        "access registry",
        require_read_only=False,
    )
    with _immutable_snapshot(registry_raw, "access-registry.jsonl") as registry_path:
        verify_registry(
            registry_path,
            registry_verification_key,
            expected_head_receipt=spend_receipt,
            allow_extensions=True,
        )
        state = verify_registry(
            registry_path,
            registry_verification_key,
            expected_head_receipt=commit_receipt,
        )
        events = verify_registry_events(
            registry_path,
            registry_verification_key,
            expected_head_receipt=commit_receipt,
        )
    if len(events) < 2:
        raise ValueError("CTAA assessment registry lacks spend/commit events")
    spend, commit = events[-2:]
    spend_payload = spend.payload
    commit_payload = commit.payload
    claim = _validate_signed_assessment_claim(
        path=assessment_claim_path,
        assessment_path=assessment_path,
        assessment=assessment,
        access=access,
        spend_payload=spend_payload,
        commit_payload=commit_payload,
        registry_verification_key=registry_verification_key,
    )
    assessment_claim_sha256 = claim["claim_sha256"]
    if assessment_claim_sha256 != access.get("assessment_claim_sha256"):
        raise ValueError("CTAA signed assessment claim binding differs")
    expected_spend = {
        "event_type": ACCESS_SPEND,
        "access_id": access.get("access_id"),
        "partition": assessment.get("partition"),
        "manifest_sha256": assessment.get("manifest_sha256"),
        "board_sha256": access.get("board_sha256"),
        "run_contract_sha256": access.get("run_contract_sha256"),
        "runtime_bundle_sha256": access.get("runtime_bundle_sha256"),
        "assessment_claim_sha256": assessment_claim_sha256,
        "bootstrap_seed_receipt_sha256": access.get("bootstrap_seed_receipt_sha256"),
        "bootstrap_seed": access.get("bootstrap_seed"),
        "statistical_gate_spec_file_sha256": access.get(
            "statistical_gate_spec_file_sha256"
        ),
        "gate_spec_sha256": access.get("gate_spec_sha256"),
    }
    if (
        spend.entry_hash != access.get("registry_head_entry_hash")
        or hashlib.sha256(spend.canonical_payload).hexdigest()
        != access.get("access_event_payload_sha256")
        or any(spend_payload.get(key) != item for key, item in expected_spend.items())
        or spend_payload.get("registry_id") != access.get("registry_id")
    ):
        raise ValueError("CTAA signed access-spend binding differs")
    if (
        commit_payload.get("event_type") != ASSESSMENT_COMMIT
        or commit_payload.get("access_id") != access.get("access_id")
        or commit_payload.get("assessment_sha256") != assessment_sha256
        or commit_payload.get("statistical_gate_spec_file_sha256")
        != access.get("statistical_gate_spec_file_sha256")
        or commit_payload.get("gate_spec_sha256") != access.get("gate_spec_sha256")
        or state.head_hash != commit.entry_hash
        or state.head_event_type != ASSESSMENT_COMMIT
        or state.open_access_id is not None
        or state.registry_id != access.get("registry_id")
    ):
        raise ValueError("CTAA signed assessment-commit binding differs")
    return {
        "registry_id": state.registry_id,
        "access_id": access["access_id"],
        "spend_entry_hash": spend.entry_hash,
        "assessment_commit_entry_hash": commit.entry_hash,
        "assessment_sha256": assessment_sha256,
        "run_contract_sha256": access["run_contract_sha256"],
        "runtime_bundle_sha256": access["runtime_bundle_sha256"],
        "execution_set_file_sha256": access["execution_set_file_sha256"],
        "execution_set_sha256": access["execution_set_sha256"],
        "statistical_gate_spec_file_sha256": access[
            "statistical_gate_spec_file_sha256"
        ],
        "gate_spec_sha256": access["gate_spec_sha256"],
        "assessment_claim_sha256": assessment_claim_sha256,
        "assessment_claim_payload": claim["payload"],
        "bootstrap_seed_receipt_sha256": access["bootstrap_seed_receipt_sha256"],
        "bootstrap_seed": access["bootstrap_seed"],
    }


def audit_current_contract(
    *,
    assessment_path: Path,
    assessment_claim_path: Path,
    access_registry_path: Path,
    access_spend_head_receipt_path: Path,
    assessment_commit_head_receipt_path: Path,
    registry_verification_key: bytes | Ed25519PublicKey,
    finite_audit_paths: Sequence[Path],
    resource_profile_path: Path,
    capacity_audit_path: Path,
    immutable_preflight_path: Path,
    bootstrap_seed_receipt_path: Path,
    run_contract_path: Path,
    runtime_bundle_path: Path,
    runtime_program_source_path: Path,
    runtime_query_source_path: Path,
    runtime_tokenizer_path: Path,
    runtime_execution_set_path: Path,
    assessment_source_bundle_path: Path,
    assessment_source_manifest_path: Path,
    statistical_gate_spec_path: Path,
    python_executable_path: Path,
    bwrap_executable_path: Path,
) -> dict[str, object]:
    assessment, assessment_raw = _load_object_with_raw(assessment_path, "assessment")
    assessment_sha256 = hashlib.sha256(assessment_raw).hexdigest()
    seeds, indexed, _families = _validate_assessment(assessment)
    signed_access = _validate_signed_assessment_access(
        assessment=assessment,
        assessment_path=assessment_path,
        assessment_sha256=assessment_sha256,
        assessment_claim_path=assessment_claim_path,
        access_registry_path=access_registry_path,
        access_spend_head_receipt_path=access_spend_head_receipt_path,
        assessment_commit_head_receipt_path=assessment_commit_head_receipt_path,
        registry_verification_key=registry_verification_key,
    )
    run_contract, _run_contract_raw = _load_object_with_raw(
        run_contract_path, "run contract"
    )
    assessment_claim_payload = signed_access["assessment_claim_payload"]
    if not isinstance(
        assessment_claim_payload, Mapping
    ) or assessment_claim_payload.get("run_plan_sha256") != run_contract.get(
        "run_plan_sha256"
    ):
        raise ValueError("CTAA signed assessment claim run-plan binding differs")

    runtime_execution_set, execution_set_file_sha256 = (
        read_runtime_execution_set_with_replay(
            runtime_execution_set_path,
            runtime_bundle_path=runtime_bundle_path,
            run_contract=run_contract,
            verification_key=registry_verification_key,
        )
    )
    if (
        execution_set_file_sha256 != signed_access["execution_set_file_sha256"]
        or runtime_execution_set.get("execution_set_sha256")
        != signed_access["execution_set_sha256"]
        or runtime_execution_set.get("run_contract_sha256")
        != signed_access["run_contract_sha256"]
    ):
        raise ValueError("CTAA signed runtime execution set binding differs")

    source_manifest_value, source_manifest_raw = _load_object_with_raw(
        assessment_source_manifest_path, "assessment source manifest"
    )
    validated_source_manifest = validate_assessment_source_bundle(
        source_root=Path(__file__).resolve().parent,
        bundle_path=assessment_source_bundle_path,
        manifest_path=assessment_source_manifest_path,
        python_executable=python_executable_path,
        bwrap_executable=bwrap_executable_path,
    )
    source_bundle_raw = _read_file_once(
        assessment_source_bundle_path, "assessment source bundle"
    )
    source_bindings = {
        "assessment_source_bundle_sha256": hashlib.sha256(
            source_bundle_raw
        ).hexdigest(),
        "assessment_source_manifest_sha256": hashlib.sha256(
            source_manifest_raw
        ).hexdigest(),
        "python_interpreter_sha256": validated_source_manifest.get(
            "python_interpreter", {}
        ).get("sha256"),
        "bwrap_executable_sha256": validated_source_manifest.get(
            "bwrap_executable", {}
        ).get("sha256"),
    }
    if source_manifest_value != validated_source_manifest or any(
        assessment_claim_payload.get(key) != value
        for key, value in source_bindings.items()
    ):
        raise ValueError("CTAA signed assessor source binding differs")
    runtime_bundle_value, runtime_bundle_raw = _load_object_with_raw(
        runtime_bundle_path, "runtime bundle"
    )
    canonical_runtime_bundle = (
        json.dumps(
            runtime_bundle_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if runtime_bundle_raw != canonical_runtime_bundle:
        raise ValueError("CTAA runtime bundle is not canonical JSON")
    runtime_program_raw = _read_file_once(
        runtime_program_source_path, "runtime program source"
    )
    runtime_query_raw = _read_file_once(
        runtime_query_source_path, "runtime query source"
    )
    runtime_tokenizer_raw = _read_file_once(runtime_tokenizer_path, "runtime tokenizer")
    with (
        _immutable_snapshot(
            runtime_program_raw, "program-source.jsonl"
        ) as program_path,
        _immutable_snapshot(runtime_query_raw, "query-source.jsonl") as query_path,
        _immutable_snapshot(runtime_tokenizer_raw, "tokenizer.json") as tokenizer_path,
    ):
        runtime_bundle = _validate_loaded_runtime_bundle_with_replay(
            runtime_bundle_value,
            runtime_bundle_path,
            run_contract=run_contract,
            program_path=program_path,
            query_path=query_path,
            tokenizer_path=tokenizer_path,
        )
    runtime_bundle_sha256 = hashlib.sha256(runtime_bundle_raw).hexdigest()
    if (
        runtime_bundle_sha256 != signed_access["runtime_bundle_sha256"]
        or runtime_bundle["partition"] != assessment["partition"]
        or runtime_bundle["manifest_sha256"] != assessment["manifest_sha256"]
        or runtime_bundle["run_contract_sha256"] != signed_access["run_contract_sha256"]
    ):
        raise ValueError("CTAA signed runtime bundle binding differs")
    finite = _validate_finite_audits(finite_audit_paths, indexed=indexed)
    resource_value, resource_raw = _load_object_with_raw(
        resource_profile_path, "resource profile"
    )
    resource = _validate_resource_profile(resource_value)
    capacity_value, capacity_raw = _load_object_with_raw(
        capacity_audit_path, "capacity audit"
    )
    capacity = _validate_capacity_audit(capacity_value)
    immutable_value, immutable_raw = _load_object_with_raw(
        immutable_preflight_path, "immutable preflight"
    )
    immutable = _validate_immutable_preflight(
        immutable_value,
        resources=resource,
        capacity=capacity,
    )
    bootstrap_value, bootstrap_raw = _load_object_with_raw(
        bootstrap_seed_receipt_path, "bootstrap seed receipt"
    )
    gate_source_sha256 = hashlib.sha256(
        _read_file_once(
            Path(__file__), "advancement gate source", require_read_only=False
        )
    ).hexdigest()
    statistics_source_sha256 = hashlib.sha256(
        _read_file_once(
            Path(__file__).with_name("ctaa_gate_statistics.py"),
            "gate statistics source",
            require_read_only=False,
        )
    ).hexdigest()
    bootstrap = validate_bootstrap_receipt(
        bootstrap_value,
        manifest_sha256=str(assessment["manifest_sha256"]),
        gate_source_sha256=gate_source_sha256,
        statistics_source_sha256=statistics_source_sha256,
    )
    bootstrap_receipt_sha256 = hashlib.sha256(bootstrap_raw).hexdigest()
    if (
        signed_access["bootstrap_seed_receipt_sha256"] != bootstrap_receipt_sha256
        or signed_access["bootstrap_seed"] != bootstrap["bootstrap_seed"]
    ):
        raise ValueError("CTAA signed bootstrap commitment differs")
    training_seeds = run_contract.get("training_seeds")
    if (
        not isinstance(training_seeds, list)
        or len(training_seeds) != 5
        or any(type(seed) is not int or seed < 0 for seed in training_seeds)
    ):
        raise ValueError(
            "CTAA statistical gate requires exactly five ordered training seeds"
        )
    gate_bindings = StatisticalGateBindings(
        manifest_sha256=str(assessment["manifest_sha256"]),
        board_sha256=str(assessment_claim_payload["board_sha256"]),
        run_plan_sha256=str(assessment_claim_payload["run_plan_sha256"]),
        run_contract_sha256=str(signed_access["run_contract_sha256"]),
        runtime_bundle_file_sha256=runtime_bundle_sha256,
        runtime_bundle_sha256=str(runtime_bundle["bundle_sha256"]),
        runtime_execution_set_file_sha256=execution_set_file_sha256,
        runtime_execution_set_sha256=str(
            runtime_execution_set["execution_set_sha256"]
        ),
        assessment_source_bundle_sha256=str(
            source_bindings["assessment_source_bundle_sha256"]
        ),
        assessment_source_manifest_sha256=str(
            source_bindings["assessment_source_manifest_sha256"]
        ),
        bootstrap_seed_receipt_sha256=bootstrap_receipt_sha256,
        bootstrap_seed=int(bootstrap["bootstrap_seed"]),
        training_seeds=tuple(training_seeds),
    )
    try:
        statistical_gate_spec, statistical_gate_spec_file_sha256 = (
            read_signed_statistical_gate_spec_with_sha(
                statistical_gate_spec_path,
                verification_key=registry_verification_key,
                expected_bindings=gate_bindings,
            )
        )
    except (StatisticalGateSpecError, OSError, ValueError) as error:
        raise ValueError("CTAA signed statistical gate specification differs") from error
    gate_spec_sha256 = statistical_gate_spec.get("gate_spec_sha256")
    if (
        statistical_gate_spec_file_sha256
        != signed_access["statistical_gate_spec_file_sha256"]
        or gate_spec_sha256 != signed_access["gate_spec_sha256"]
    ):
        raise ValueError("CTAA signed statistical gate specification binding differs")
    return {
        "schema": "r12_ctaa_v2_advancement_contract_audit_v3",
        "assessment_schema": assessment["schema"],
        "assessment_sha256": assessment_sha256,
        "signed_access_recomputation": signed_access,
        "manifest_sha256": assessment["manifest_sha256"],
        "seeds": seeds,
        "finite_audits": finite,
        "resource_profile_sha256": hashlib.sha256(resource_raw).hexdigest(),
        "capacity_audit_sha256": hashlib.sha256(capacity_raw).hexdigest(),
        "immutable_preflight_sha256": hashlib.sha256(immutable_raw).hexdigest(),
        "bootstrap_seed_receipt_sha256": bootstrap_receipt_sha256,
        "bootstrap_seed": bootstrap["bootstrap_seed"],
        "statistical_gate_spec_file_sha256": statistical_gate_spec_file_sha256,
        "gate_spec_sha256": gate_spec_sha256,
        "statistical_gate_spec_recomputation": statistical_gate_spec,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "runtime_bundle_commitment_sha256": runtime_bundle["bundle_sha256"],
        "runtime_bundle_recomputation": runtime_bundle,
        "execution_set_file_sha256": execution_set_file_sha256,
        "execution_set_sha256": runtime_execution_set["execution_set_sha256"],
        "runtime_execution_set_recomputation": runtime_execution_set,
        "resource_recomputation": resource,
        "capacity_recomputation": capacity,
        "immutable_recomputation": immutable,
        "caller_metadata_accepted": False,
        "caller_bootstrap_seed_accepted": False,
        "committed_bootstrap_seed_accepted": True,
        "advancement_statistics_computed": False,
        "all_advancement_gates_pass": False,
        "contract_resolved": False,
        "unresolved_contracts": list(UNRESOLVED_CONTRACTS),
    }


def evaluate_advancement_gates(
    *,
    assessment_path: Path,
    assessment_claim_path: Path,
    access_registry_path: Path,
    access_spend_head_receipt_path: Path,
    assessment_commit_head_receipt_path: Path,
    registry_verification_key: bytes | Ed25519PublicKey,
    finite_audit_paths: Sequence[Path],
    resource_profile_path: Path,
    capacity_audit_path: Path,
    immutable_preflight_path: Path,
    bootstrap_seed_receipt_path: Path,
    run_contract_path: Path,
    runtime_bundle_path: Path,
    runtime_program_source_path: Path,
    runtime_query_source_path: Path,
    runtime_tokenizer_path: Path,
    runtime_execution_set_path: Path,
    assessment_source_bundle_path: Path,
    assessment_source_manifest_path: Path,
    statistical_gate_spec_path: Path,
    python_executable_path: Path,
    bwrap_executable_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"refusing existing CTAA advancement output: {output_path}"
        )
    audit = audit_current_contract(
        assessment_path=assessment_path,
        assessment_claim_path=assessment_claim_path,
        access_registry_path=access_registry_path,
        access_spend_head_receipt_path=access_spend_head_receipt_path,
        assessment_commit_head_receipt_path=assessment_commit_head_receipt_path,
        registry_verification_key=registry_verification_key,
        finite_audit_paths=finite_audit_paths,
        resource_profile_path=resource_profile_path,
        capacity_audit_path=capacity_audit_path,
        immutable_preflight_path=immutable_preflight_path,
        bootstrap_seed_receipt_path=bootstrap_seed_receipt_path,
        run_contract_path=run_contract_path,
        runtime_bundle_path=runtime_bundle_path,
        runtime_program_source_path=runtime_program_source_path,
        runtime_query_source_path=runtime_query_source_path,
        runtime_tokenizer_path=runtime_tokenizer_path,
        runtime_execution_set_path=runtime_execution_set_path,
        assessment_source_bundle_path=assessment_source_bundle_path,
        assessment_source_manifest_path=assessment_source_manifest_path,
        statistical_gate_spec_path=statistical_gate_spec_path,
        python_executable_path=python_executable_path,
        bwrap_executable_path=bwrap_executable_path,
    )
    # Deliberately do not write a development-gate-shaped receipt.  A rejection
    # artifact could be mistaken for an authorization by downstream code.
    raise UnresolvedContractError(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--assessment-claim", type=Path, required=True)
    parser.add_argument("--access-registry", type=Path, required=True)
    parser.add_argument("--access-spend-head-receipt", type=Path, required=True)
    parser.add_argument("--assessment-commit-head-receipt", type=Path, required=True)
    parser.add_argument("--registry-public-key", type=Path, required=True)
    parser.add_argument("--finite-audit", type=Path, action="append", required=True)
    parser.add_argument("--resource-profile", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--immutable-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-seed-receipt", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--runtime-program-source", type=Path, required=True)
    parser.add_argument("--runtime-query-source", type=Path, required=True)
    parser.add_argument("--runtime-tokenizer", type=Path, required=True)
    parser.add_argument("--runtime-execution-set", type=Path, required=True)
    parser.add_argument("--assessment-source-bundle", type=Path, required=True)
    parser.add_argument("--assessment-source-manifest", type=Path, required=True)
    parser.add_argument("--statistical-gate-spec", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--bwrap-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate_advancement_gates(
            assessment_path=args.assessment,
            assessment_claim_path=args.assessment_claim,
            access_registry_path=args.access_registry,
            access_spend_head_receipt_path=args.access_spend_head_receipt,
            assessment_commit_head_receipt_path=args.assessment_commit_head_receipt,
            registry_verification_key=_load_registry_public_key(
                args.registry_public_key
            ),
            finite_audit_paths=args.finite_audit,
            resource_profile_path=args.resource_profile,
            capacity_audit_path=args.capacity_audit,
            immutable_preflight_path=args.immutable_preflight,
            bootstrap_seed_receipt_path=args.bootstrap_seed_receipt,
            run_contract_path=args.run_contract,
            runtime_bundle_path=args.runtime_bundle,
            runtime_program_source_path=args.runtime_program_source,
            runtime_query_source_path=args.runtime_query_source,
            runtime_tokenizer_path=args.runtime_tokenizer,
            runtime_execution_set_path=args.runtime_execution_set,
            assessment_source_bundle_path=args.assessment_source_bundle,
            assessment_source_manifest_path=args.assessment_source_manifest,
            statistical_gate_spec_path=args.statistical_gate_spec,
            python_executable_path=args.python_executable,
            bwrap_executable_path=args.bwrap_executable,
            output_path=args.output,
        )
    except UnresolvedContractError as error:
        print(json.dumps(error.audit, sort_keys=True))
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
