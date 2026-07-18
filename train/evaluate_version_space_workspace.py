#!/usr/bin/env python3
"""CPU-only R10 static version-space and ACAW confirmation evaluator.

One deterministic pooled quantile is computed from the provenance-bound
calibration score report. A disjoint finite confirmation board is then evaluated
without tuning. All source eviction in this module means reversible HOT-context
eviction backed by an immutable retrieval pointer; no source is irreversibly
deleted.
"""

from __future__ import annotations

import argparse
import collections
import hmac
import hashlib
import json
import math
import operator
import os
import platform
import random
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import tokenizers
import torch

from affine_ambiguity_workspace import (
    AffineAmbiguityWorkspace,
    compose_workspaces,
    workspace_from_operations,
)
from categorical_microcode import (
    OPCODE_TO_ID,
    OPCODES,
    QUERIES,
    QUERY_TO_ID,
    opcode_for,
    operation_value,
    query_for,
)
from version_space_accounting import account_version_space_tree
from version_space_product_tree import (
    NUMERIC_OPCODES,
    ExactAffineTransform,
    VersionSpaceProductTree,
    build_tree,
    chronological_compose,
    compact_frontier,
    operation_transform,
)

AUDIT = "referential_version_space_workspace_confirmation_r10"
SCORE_AUDIT = "referential_version_scores_r10"
SCORE_SCHEMA_VERSION = 2
CHAIN_AUDIT = "referential_version_score_to_decision_chain_r10"
CHAIN_SCHEMA_VERSION = 1
AUTHENTICATED_SCORE_AUDIT = "referential_version_scores_r10_in_memory_v1"
BOARD_SCHEMA = "r10_workspace_board_v2"
GATE_MANIFEST_BUILD = "r10_workspace_boards_v2"
GATE_ADMISSION_AUDIT = "r10_workspace_boards_independent_admission_v2"
FROZEN_GATE_MANIFEST = "r10_workspace_frozen_score_gate_v2"
CALIBRATION_REGIMES = ("fit_iid", "depth_ood")
CONFIRMATION_REGIMES = ("language_ood", "full_ood")
EXPECTED_DEPTHS = {
    "fit_iid": (4, 8),
    "depth_ood": (16, 32),
    "language_ood": (4, 8),
    "full_ood": (16, 32),
}
EXPECTED_FAMILIES = {
    "calibration": (
        "glass studio",
        "map room",
        "print works",
        "signal bureau",
    ),
    "confirmation": (
        "binding hall",
        "ceramic works",
        "forecast office",
        "rail control",
    ),
}
EXPECTED_BOARD_ROWS = {"calibration": 800, "confirmation": 1840}
EXPECTED_CELL_ROWS = {"calibration": 10, "confirmation": 23}
EXPECTED_PARTITION_ROWS = {
    "fit_iid": 400,
    "depth_ood": 400,
    "language_ood": 920,
    "full_ood": 920,
}
CALIBRATION_QUANTILE = 0.97
MIN_PARTITION_SELECTIVE_ACCURACY = 0.99
MIN_PARTITION_SELECTIVE_COVERAGE = 0.40
MIN_EXACT_CELL_CERTIFICATES = 10
MIN_PARTITION_CERTIFICATES = 400
MIN_RETRIEVAL_BACKED_HOT_REMOVAL = 0.75
MIN_BASELINE_IMPROVEMENT = 0.01
COMPLETE_PROGRAM_COVERAGE_FLOOR = 0.95
EVENT_COVERAGE_FLOOR = 0.97
QUERY_COVERAGE_FLOOR = 0.97
VSPT_CAP = 32
OVERFLOW_SIZE = VSPT_CAP + 1
MAX_AFFINE_AMBIGUITY_RANK = 6
EXPECTED_ADAPTER_SHA256 = (
    "bf07d65075a42142c34bfc510cbef95290a9b8a0f7ed96ac1d4abc5f175a6480"
)
EXPECTED_EXTRACTOR_SEED = 20260714
FROZEN_BATCH_SIZE = 16
FROZEN_BOARD_SEEDS = {
    "calibration": 2026071401,
    "confirmation": 2026071402,
}
CANONICAL_R5_NOVELTY_BOARD_SHA256 = (
    "d85f16ff374b0c650cf3603826cc5f3b377842818db62bada3b84e71308b9473"
)
FROZEN_BOARD_ORDER = ("calibration", "confirmation")
FROZEN_DETERMINISM = {
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
    "float32_matmul_precision": "highest",
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}
FROZEN_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "4",
}
EXPECTED_DEVICE_CLASS = {
    "type": "cuda",
    "name": "NVIDIA H100 PCIe",
    "compute_capability": [9, 0],
}
R9C_PROTOCOL = "referential_bidirectional_syndrome_microcode_r9c"
POINTER_PROTOCOL = "causal_microcode_referential_slots_v4"
STRUCTURAL_ADMISSION_AUDIT = "role_equivariant_microcode_v3"
LABEL_ADMISSION_AUDIT = "referential_slot_label_admission_v1"
NO_SYNDROME_CONFIG = {
    "conditioning": "directional",
    "use_syndrome": False,
    "shuffle_goal": False,
}
PROBABILITY_TOLERANCE = 2e-5
JOINT_TOLERANCE = 5e-5
UNCAPPED_DISTINCT_LIMIT = 65536
UNCAPPED_PAIR_BUDGET = 200000
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_IDENTITY_KEYS = frozenset({"git_revision", "files", "aggregate_sha256", "runtime"})
CODE_IDENTITY_RUNTIME_KEYS = frozenset({"python", "torch", "tokenizers"})
REQUIRED_CODE_IDENTITY_FILES = frozenset(
    {
        "pipeline/audit_categorical_microcode_v1.py",
        "pipeline/audit_r10_workspace_boards.py",
        "pipeline/audit_referential_slot_labels.py",
        "pipeline/audit_role_equivariant_microcode_v3.py",
        "pipeline/generate_r10_workspace_boards.py",
        "pipeline/jobs/build_r10_workspace_boards_stokes.sbatch",
        "train/evaluate_version_space_workspace.py",
        "train/extract_referential_version_scores.py",
        "train/jobs/evaluate_version_space_workspace.sbatch",
        "train/jobs/extract_referential_version_scores.sbatch",
    }
)
FORBIDDEN_GATE_FIELD_FRAGMENTS = (
    "alpha",
    "bonferroni",
    "clopper",
    "confidence",
    "pearson",
    "population",
    "required_zero_error_cases",
    "sample_size_formula",
    "selective_accuracy_target",
    "simultaneous",
    "target_success_probability",
)
FORBIDDEN_GATE_TEXT = (
    "bonferroni",
    "clopper-pearson",
    "clopper_pearson",
    "conformal",
    "population guarantee",
    "population-guarantee",
    "simultaneous confidence",
    "simultaneous-confidence",
)


class EvaluationContractError(ValueError):
    """Raised before publication when an input or mechanics contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise EvaluationContractError("{} must be an integer, not bool".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise EvaluationContractError(
            "{} must be an exact integer".format(name)
        ) from error


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_file_once(path, description: str) -> tuple[str, bytes, os.stat_result]:
    """Read one canonical regular file through a no-follow descriptor."""
    _require(
        isinstance(path, (str, os.PathLike)),
        "{} path must be path-like".format(description),
    )
    raw_path = os.fspath(path)
    _require(isinstance(raw_path, str), "{} path must be text".format(description))
    real_path = os.path.realpath(raw_path)
    _require(
        os.path.abspath(raw_path) == real_path,
        "{} path must be absolute and canonical".format(description),
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(real_path, flags)
    except OSError as error:
        raise EvaluationContractError(
            "cannot open {}: {}".format(description, error)
        ) from error
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode) and opened.st_size > 0,
            "{} must be a non-empty regular file".format(description),
        )
        linked = os.stat(real_path, follow_symlinks=False)
        _require(
            (opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino),
            "{} changed while it was opened".format(description),
        )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        _require(
            len(raw) == opened.st_size,
            "{} size changed while it was read".format(description),
        )
        closed_over = os.fstat(descriptor)
        _require(
            (
                closed_over.st_dev,
                closed_over.st_ino,
                closed_over.st_size,
                closed_over.st_mtime_ns,
                closed_over.st_ctime_ns,
            )
            == (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ),
            "{} changed while it was read".format(description),
        )
        return real_path, raw, opened
    finally:
        os.close(descriptor)


def _json_object_from_bytes(raw: bytes, description: str) -> dict:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(
            "invalid {}: {}".format(description, error)
        ) from error
    _require(isinstance(payload, dict), "{} must be a JSON object".format(description))
    return payload


def _jsonl_from_bytes(raw: bytes, description: str = "board JSONL") -> tuple[dict, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvaluationContractError(
            "invalid {}: {}".format(description, error)
        ) from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        _require(
            line.strip() != "",
            "{} contains a blank line at {}".format(description, line_number),
        )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationContractError(
                "invalid {} line {}: {}".format(description, line_number, error)
            ) from error
        _require(
            isinstance(row, dict),
            "{} row {} must be a JSON object".format(description, line_number),
        )
        rows.append(row)
    _require(rows, "{} is empty".format(description))
    return tuple(rows)


def _sha256_cached(path, cache: dict[str, str] | None) -> str:
    real_path = os.path.realpath(path)
    if cache is None:
        return sha256_file(real_path)
    if real_path not in cache:
        cache[real_path] = sha256_file(real_path)
    return cache[real_path]


def canonical_json_bytes(payload) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _reject_invalid_confidence_claims(value, path: str = "$") -> None:
    if isinstance(value, dict):
        if path == "$.code_identity.files":
            return
        for key, child in value.items():
            _require(
                isinstance(key, str), "gate key is not a string at {}".format(path)
            )
            lowered = key.lower()
            _require(
                not any(
                    fragment in lowered for fragment in FORBIDDEN_GATE_FIELD_FRAGMENTS
                ),
                "forbidden population-confidence field at {}.{}".format(path, key),
            )
            _reject_invalid_confidence_claims(child, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_invalid_confidence_claims(child, "{}[{}]".format(path, index))
    elif isinstance(value, str):
        lowered = value.lower()
        _require(
            not any(fragment in lowered for fragment in FORBIDDEN_GATE_TEXT),
            "forbidden population-confidence text at {}".format(path),
        )


def current_runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "tokenizers": str(tokenizers.__version__),
    }


def code_identity_aggregate(
    git_revision: str,
    files: dict[str, str],
    runtime: dict[str, str],
) -> str:
    """Hash the complete committed source, revision, and runtime identity."""
    _require(_is_git_revision(git_revision), "aggregate git revision is invalid")
    _require(
        isinstance(runtime, dict) and set(runtime) == CODE_IDENTITY_RUNTIME_KEYS,
        "aggregate runtime schema changed",
    )
    return canonical_sha256(
        {"git_revision": git_revision, "files": files, "runtime": runtime}
    )


def _live_git_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvaluationContractError(
            "cannot resolve live git revision: {}".format(error)
        ) from error
    revision = completed.stdout.strip()
    _require(_is_git_revision(revision), "live git revision is not a full SHA")
    return revision


def _committed_blob_sha256(repo_root: Path, revision: str, relative: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", "{}:{}".format(revision, relative)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvaluationContractError(
            "cannot read committed source {}: {}".format(relative, error)
        ) from error
    return hashlib.sha256(completed.stdout).hexdigest()


def _repo_relative_source(path, repo_root: Path, description: str) -> str:
    real_path = Path(path).resolve()
    try:
        relative = real_path.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise EvaluationContractError(
            "{} must be inside the repository".format(description)
        ) from error
    _require(relative != "", "{} cannot be the repository root".format(description))
    return relative


def validate_code_identity(
    identity,
    *,
    repo_root,
    expected_git_revision: str,
    required_sources: dict[str, str],
    expected_runtime: dict[str, str] | None = None,
    live_git_revision: str | None = None,
    hash_cache: dict[str, str] | None = None,
    committed_source_hashes: dict[str, str] | None = None,
) -> dict:
    """Verify the complete shared code identity against this checkout/runtime."""
    _require(isinstance(identity, dict), "gate lacks code_identity")
    _require(
        set(identity) == CODE_IDENTITY_KEYS,
        "code_identity schema changed",
    )
    revision = identity.get("git_revision")
    _require(_is_git_revision(revision), "code_identity git_revision is invalid")
    _require(
        _is_git_revision(expected_git_revision), "expected code revision is invalid"
    )
    _require(
        revision == expected_git_revision,
        "code_identity git revision differs from the frozen revision",
    )

    root = Path(repo_root).resolve()
    _require(root.is_dir(), "repository root is missing")
    actual_revision = live_git_revision or _live_git_revision(root)
    _require(
        _is_git_revision(actual_revision) and actual_revision == revision,
        "live git revision differs from code_identity",
    )

    files = identity.get("files")
    _require(isinstance(files, dict) and files, "code_identity files are empty")
    _require(
        REQUIRED_CODE_IDENTITY_FILES.issubset(files),
        "code_identity omits required R10 source or job files",
    )
    for relative, expected_hash in files.items():
        _require(
            isinstance(relative, str) and relative != "" and "\\" not in relative,
            "code_identity source path is invalid",
        )
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute()
            and pure.as_posix() == relative
            and all(part not in ("", ".", "..") for part in pure.parts),
            "code_identity source path is not canonical: {}".format(relative),
        )
        _require(
            _is_sha256(expected_hash),
            "code_identity source hash is invalid: {}".format(relative),
        )
        source = root.joinpath(*pure.parts)
        real_source = Path(os.path.realpath(source))
        _require(
            source.absolute() == real_source,
            "code_identity source is not a canonical regular path: {}".format(relative),
        )
        _require(
            real_source.is_file() and real_source.stat().st_size > 0,
            "code_identity source is missing: {}".format(relative),
        )
        _require(
            _sha256_cached(real_source, hash_cache) == expected_hash,
            "code_identity source hash mismatch: {}".format(relative),
        )
        committed_hash = (
            committed_source_hashes.get(relative)
            if committed_source_hashes is not None
            else _committed_blob_sha256(root, revision, relative)
        )
        _require(
            committed_hash == expected_hash,
            "code_identity source is not the committed blob: {}".format(relative),
        )

    _require(
        isinstance(required_sources, dict) and required_sources,
        "required code sources are empty",
    )
    for relative, expected_hash in required_sources.items():
        _require(
            files.get(relative) == expected_hash,
            "code_identity does not bind required source {}".format(relative),
        )

    runtime = identity.get("runtime")
    _require(
        isinstance(runtime, dict)
        and set(runtime) == CODE_IDENTITY_RUNTIME_KEYS
        and all(isinstance(value, str) and value for value in runtime.values()),
        "code_identity runtime schema changed",
    )
    actual_runtime = expected_runtime or current_runtime_identity()
    _require(
        set(actual_runtime) == CODE_IDENTITY_RUNTIME_KEYS,
        "live runtime identity schema changed",
    )
    for name, expected_version in runtime.items():
        _require(
            actual_runtime.get(name) == expected_version,
            "code_identity runtime mismatch: {}".format(name),
        )

    aggregate = identity.get("aggregate_sha256")
    _require(_is_sha256(aggregate), "code_identity aggregate_sha256 is invalid")
    _require(
        aggregate == code_identity_aggregate(revision, files, runtime),
        "code_identity aggregate mismatch",
    )
    return identity


def atomic_write_json_no_overwrite(payload, path) -> None:
    """Publish complete JSON with fsync and a no-overwrite hard-link commit."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError("refusing existing output: {}".format(output))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(output.name),
        suffix=".tmp",
        dir=str(output.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing existing output: {}".format(output)
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path, description: str) -> dict:
    _, raw, _ = _read_regular_file_once(path, description)
    return _json_object_from_bytes(raw, description)


def _load_jsonl(path) -> tuple[dict, ...]:
    _, raw, _ = _read_regular_file_once(path, "board JSONL")
    return _jsonl_from_bytes(raw)


def _validate_probability_row(raw, width: int, name: str) -> tuple[float, ...]:
    _require(
        isinstance(raw, list) and len(raw) == width,
        "{} has the wrong width".format(name),
    )
    values = []
    for index, value in enumerate(raw):
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            "{}[{}] is not numeric".format(name, index),
        )
        value = float(value)
        _require(math.isfinite(value), "{}[{}] is non-finite".format(name, index))
        _require(0.0 <= value <= 1.0, "{}[{}] is outside [0,1]".format(name, index))
        values.append(value)
    total = math.fsum(values)
    _require(
        abs(total - 1.0) <= PROBABILITY_TOLERANCE,
        "{} does not sum to one (got {!r})".format(name, total),
    )
    return tuple(values)


def _validate_joint_distribution(
    joint: Sequence[float],
    forward: Sequence[float],
    backward: Sequence[float],
    name: str,
) -> None:
    weights = tuple(math.sqrt(left * right) for left, right in zip(forward, backward))
    normalizer = math.fsum(weights)
    _require(
        normalizer > 0.0,
        "{} directional probabilities have zero geometric mass".format(name),
    )
    expected = tuple(value / normalizer for value in weights)
    _require(
        all(
            abs(actual - target) <= JOINT_TOLERANCE
            for actual, target in zip(joint, expected)
        ),
        "{} is not softmax(average(forward_logits, backward_logits))".format(name),
    )


def _event_lines(question, depth: int) -> tuple[str, ...]:
    _require(isinstance(question, str), "board question must be a string")
    lines = tuple(
        line for line in question.splitlines() if line.startswith(("Step ", "Event "))
    )
    _require(
        len(lines) == depth, "board question event-line count differs from operations"
    )
    return lines


def _event_source_payloads(row: dict, reference: str) -> tuple[dict, ...]:
    operations = row.get("operations")
    _require(isinstance(operations, list) and operations, "board row lacks operations")
    lines = _event_lines(row.get("question"), len(operations))
    return tuple(
        {
            "event_index": index,
            "operation": operation,
            "reference": reference,
            "text": lines[index],
        }
        for index, operation in enumerate(operations)
    )


@dataclass(frozen=True)
class ScoreRecord:
    index: int
    reference: str
    regime: str
    family: str
    operation_targets: tuple[int, ...]
    operation_values: tuple[int, ...]
    initial_state: tuple[int, int]
    query_target: int
    answer: int
    joint_probabilities: tuple[tuple[float, ...], ...]
    forward_probabilities: tuple[tuple[float, ...], ...]
    backward_probabilities: tuple[tuple[float, ...], ...]
    query_probabilities: tuple[float, ...]
    source_record_sha256: str
    event_source_payloads: tuple[dict, ...]

    @property
    def depth(self) -> int:
        return len(self.operation_targets)

    @property
    def source_bytes(self) -> int:
        return sum(
            len(canonical_json_bytes(payload)) for payload in self.event_source_payloads
        )


@dataclass(frozen=True)
class BoundScoreReport:
    path: str
    sha256: str
    report: dict
    records: tuple[ScoreRecord, ...]


@dataclass(frozen=True)
class AuthenticatedScoreBytes:
    """Extractor-created score bytes authenticated only for the live invocation."""

    board_name: str
    payload: bytes
    sha256: str
    mac: str


@dataclass(frozen=True)
class GateBundle:
    manifest_path: str
    manifest_sha256: str
    manifest: dict
    admission_path: str
    admission_sha256: str
    admission: dict
    build_path: str
    build_sha256: str
    build: dict
    code_identity: dict


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    quantile: float
    programs: int
    events: int
    order_statistic: int
    calibration_digest: str

    @property
    def minimum_probability(self) -> float:
        return 0.0 if math.isinf(self.threshold) else math.exp(-self.threshold)


def _board_regimes(board_name: str) -> tuple[str, str]:
    _require(board_name in EXPECTED_BOARD_ROWS, "unknown board role")
    return CALIBRATION_REGIMES if board_name == "calibration" else CONFIRMATION_REGIMES


def _expected_cell_counter(board_name: str) -> collections.Counter:
    return collections.Counter(
        {
            (regime, depth, query, family): EXPECTED_CELL_ROWS[board_name]
            for regime in _board_regimes(board_name)
            for depth in EXPECTED_DEPTHS[regime]
            for query in QUERIES
            for family in EXPECTED_FAMILIES[board_name]
        }
    )


def _cell_key(regime: str, depth: int, query: str, family: str) -> str:
    return "{}|depth={}|query={}|family={}".format(regime, depth, query, family)


def validate_record_geometry(records: Sequence[ScoreRecord], board_name: str) -> dict:
    """Require every frozen v2 regime/depth/query/family cell exactly once-sized."""
    records = tuple(records)
    _require(
        len(records) == EXPECTED_BOARD_ROWS[board_name],
        "{} score report must contain exactly {} rows".format(
            board_name, EXPECTED_BOARD_ROWS[board_name]
        ),
    )
    expected_regimes = set(_board_regimes(board_name))
    _require(
        {record.regime for record in records} == expected_regimes,
        "{} regimes must be exactly {}".format(board_name, sorted(expected_regimes)),
    )
    for record in records:
        _require(
            record.depth in EXPECTED_DEPTHS.get(record.regime, ()),
            "{} has forbidden depth {} for regime {}".format(
                record.reference, record.depth, record.regime
            ),
        )
        _require(
            record.family in EXPECTED_FAMILIES[board_name],
            "{} has unknown {} family {}".format(
                record.reference, board_name, record.family
            ),
        )
    actual_cells = collections.Counter(
        (
            record.regime,
            record.depth,
            QUERIES[record.query_target],
            record.family,
        )
        for record in records
    )
    expected_cells = _expected_cell_counter(board_name)
    missing = sorted(set(expected_cells) - set(actual_cells))
    unexpected = sorted(set(actual_cells) - set(expected_cells))
    undersized = sorted(
        (cell, actual_cells[cell], expected_cells[cell])
        for cell in expected_cells
        if actual_cells[cell] < expected_cells[cell]
    )
    oversized = sorted(
        (cell, actual_cells[cell], expected_cells[cell])
        for cell in expected_cells
        if actual_cells[cell] > expected_cells[cell]
    )
    _require(not missing, "{} board is missing frozen cells".format(board_name))
    _require(not unexpected, "{} board has unexpected cells".format(board_name))
    _require(
        not undersized,
        "{} board has undersized frozen cells".format(board_name),
    )
    _require(not oversized, "{} board has oversized frozen cells".format(board_name))
    regimes = collections.Counter(record.regime for record in records)
    depths = collections.Counter((record.regime, record.depth) for record in records)
    queries = collections.Counter(QUERIES[record.query_target] for record in records)
    families = collections.Counter(record.family for record in records)
    return {
        "board": board_name,
        "rows": len(records),
        "regimes": dict(sorted(regimes.items())),
        "regime_depths": {
            "{}:{}".format(regime, depth): count
            for (regime, depth), count in sorted(depths.items())
        },
        "queries": dict(sorted(queries.items())),
        "families": dict(sorted(families.items())),
        "cells": {
            _cell_key(*cell): actual_cells[cell] for cell in sorted(actual_cells)
        },
        "cell_rows": EXPECTED_CELL_ROWS[board_name],
        "all_cells_exact": True,
    }


def _expected_summary(board_name: str) -> dict:
    rows = EXPECTED_BOARD_ROWS[board_name]
    regimes = {
        regime: EXPECTED_PARTITION_ROWS[regime] for regime in _board_regimes(board_name)
    }
    return {
        "rows": rows,
        "depths": {
            str(depth): rows // 4
            for regime in _board_regimes(board_name)
            for depth in EXPECTED_DEPTHS[regime]
        },
        "regimes": regimes,
        "queries": {query: rows // len(QUERIES) for query in QUERIES},
        "families": {
            family: rows // len(EXPECTED_FAMILIES[board_name])
            for family in EXPECTED_FAMILIES[board_name]
        },
        "expected_cell_count": len(_expected_cell_counter(board_name)),
        "rows_per_exact_cell": EXPECTED_CELL_ROWS[board_name],
        "exact_cells": {
            _cell_key(*cell): count
            for cell, count in sorted(_expected_cell_counter(board_name).items())
        },
    }


def _validate_frozen_board_summary(
    summary: dict,
    board_name: str,
    description: str,
    *,
    require_capacity: bool = False,
) -> None:
    _require(isinstance(summary, dict), "{} is not an object".format(description))
    expected = _expected_summary(board_name)
    for key, value in expected.items():
        _require(
            summary.get(key) == value,
            "{} {} differs from the frozen v2 contract".format(description, key),
        )
    if board_name == "confirmation" and require_capacity:
        _require(
            summary.get("accepted_capacity_at_40_percent_coverage_by_partition")
            == {
                partition: math.floor(
                    EXPECTED_PARTITION_ROWS[partition]
                    * MIN_PARTITION_SELECTIVE_COVERAGE
                )
                for partition in CONFIRMATION_REGIMES
            },
            "confirmation manifest 40 percent capacity changed",
        )


def _expected_build_schedule() -> dict:
    families = {
        "calibration": ["map room", "print works", "signal bureau", "glass studio"],
        "confirmation": [
            "binding hall",
            "rail control",
            "ceramic works",
            "forecast office",
        ],
    }
    ranges = {
        "fit_iid": ([3, 29], [1, 9], "in_range"),
        "depth_ood": ([211, 499], [11, 29], "shifted"),
        "language_ood": ([3, 29], [1, 9], "in_range"),
        "full_ood": ([701, 1099], [31, 53], "shifted"),
    }
    return {
        board_name: {
            "cells": len(_expected_cell_counter(board_name)),
            "regimes": {
                regime: {
                    "depths": list(EXPECTED_DEPTHS[regime]),
                    "numeric_profile": ranges[regime][2],
                    "initial_range_inclusive": ranges[regime][0],
                    "event_value_range_inclusive": ranges[regime][1],
                    "families": families[board_name],
                    "queries": list(QUERIES),
                }
                for regime in _board_regimes(board_name)
            },
        }
        for board_name in FROZEN_BOARD_ORDER
    }


def _recompute_pass_checks(checks, description: str) -> bool:
    _require(
        isinstance(checks, dict) and checks,
        "{} checks are missing".format(description),
    )
    _require(
        all(isinstance(key, str) and isinstance(value, bool) for key, value in checks.items()),
        "{} checks have an invalid schema".format(description),
    )
    return all(checks.values())


def _source_scans_are_substantively_zero(source_reports) -> bool:
    if not isinstance(source_reports, list) or not source_reports:
        return False
    roles = []
    for report in source_reports:
        if not isinstance(report, dict):
            return False
        role = report.get("role")
        path = report.get("path")
        if role not in {"training_data", "r5_fresh_board"}:
            return False
        if (
            not isinstance(path, str)
            or not os.path.isabs(path)
            or os.path.realpath(path) != path
            or not _is_sha256(report.get("sha256"))
            or not isinstance(report.get("rows_scanned"), int)
            or isinstance(report.get("rows_scanned"), bool)
            or report["rows_scanned"] <= 0
        ):
            return False
        boards = report.get("boards")
        if not isinstance(boards, dict) or set(boards) != set(FROZEN_BOARD_ORDER):
            return False
        for board_report in boards.values():
            if not isinstance(board_report, dict):
                return False
            for key in (
                "exact_prompt_rows",
                "ngram13_rows",
                "program_rows",
                "novel_domain_phrase_rows",
            ):
                if board_report.get(key) != 0:
                    return False
            if board_report.get("sample_hits") != []:
                return False
        roles.append(role)
    r5 = [item for item in source_reports if item["role"] == "r5_fresh_board"]
    return (
        roles.count("r5_fresh_board") == 1
        and roles.count("training_data") >= 1
        and r5[0]["sha256"] == CANONICAL_R5_NOVELTY_BOARD_SHA256
    )


def _validate_build_and_admission_content(
    *,
    manifest: dict,
    admission: dict,
    build: dict,
    build_path: str,
    build_sha256: str,
    expected_board_bindings: dict[str, dict[str, str]],
) -> None:
    """Recompute the decision-relevant build/admission invariants."""
    required_build_fields = {
        "build",
        "schema",
        "cpu_only",
        "score_outputs_read",
        "score_artifacts",
        "ready_for_r10_score_run",
        "ngram_width",
        "executor_width",
        "generation_contract",
        "schedule_contract",
        "tokenizer",
        "inputs",
        "outputs",
        "cross_board_scan",
        "claim_boundary",
    }
    _require(
        set(build) == required_build_fields,
        "build manifest fields differ from the frozen contract",
    )
    _require(
        build.get("build") == GATE_MANIFEST_BUILD
        and build.get("schema") == BOARD_SCHEMA
        and build.get("cpu_only") is True
        and build.get("score_outputs_read") is False
        and build.get("score_artifacts") == []
        and build.get("ready_for_r10_score_run") is False
        and build.get("ngram_width") == 13
        and build.get("executor_width") == 8,
        "build manifest execution contract changed",
    )
    _require(
        build.get("generation_contract")
        == {
            "generator_seeds": FROZEN_BOARD_SEEDS,
            "r5_novelty_board_sha256": CANONICAL_R5_NOVELTY_BOARD_SHA256,
            "seed_variation_forbidden": True,
            "r5_variation_forbidden": True,
        },
        "build generation contract changed",
    )
    _require(
        build.get("schedule_contract") == _expected_build_schedule(),
        "build schedule contract changed",
    )
    _require(
        build.get("cross_board_scan")
        == {"exact_prompt_hits": 0, "ngram13_hits": 0, "program_hits": 0},
        "build cross-board scan is not zero",
    )

    outputs = build.get("outputs")
    _require(
        isinstance(outputs, dict) and set(outputs) == set(FROZEN_BOARD_ORDER),
        "build outputs changed",
    )
    gate_boards = manifest.get("boards")
    admitted_boards = admission.get("boards")
    _require(isinstance(gate_boards, dict), "frozen gate lacks boards")
    _require(isinstance(admitted_boards, dict), "admission lacks boards")
    for board_name in FROZEN_BOARD_ORDER:
        output = outputs[board_name]
        binding = expected_board_bindings[board_name]
        _validate_frozen_board_summary(
            output,
            board_name,
            "build output {}".format(board_name),
            require_capacity=True,
        )
        _require(
            output.get("seed") == FROZEN_BOARD_SEEDS[board_name],
            "{} generation seed changed".format(board_name),
        )
        _require(
            isinstance(output.get("path"), str)
            and os.path.isabs(output["path"])
            and os.path.realpath(output["path"]) == output["path"],
            "{} build output path is not canonical".format(board_name),
        )
        _require(
            output.get("sha256") == binding["data_sha256"],
            "{} build output hash mismatch".format(board_name),
        )
        if "data_path" in binding:
            _require(
                output["path"] == binding["data_path"],
                "{} build output path mismatch".format(board_name),
            )

        gate_board = gate_boards.get(board_name)
        _require(isinstance(gate_board, dict), "frozen gate board is missing")
        _require(
            gate_board.get("path") == output["path"]
            and gate_board.get("sha256") == output["sha256"],
            "frozen gate does not bind the substantive {} build output".format(
                board_name
            ),
        )

        admitted = admitted_boards.get(board_name)
        _validate_frozen_board_summary(
            admitted, board_name, "gate admission {}".format(board_name)
        )
        _require(
            admitted.get("path") == output["path"]
            and admitted.get("sha256") == output["sha256"]
            and admitted.get("generation_seeds") == [FROZEN_BOARD_SEEDS[board_name]],
            "admission does not reproduce the {} build output".format(board_name),
        )
        recomputed_checks_pass = _recompute_pass_checks(
            admitted.get("checks"), "{} admission".format(board_name)
        )
        _require(
            admitted.get("all_checks_pass") is recomputed_checks_pass
            and recomputed_checks_pass,
            "{} admission checks do not substantively pass".format(board_name),
        )

    build_tokenizer = build.get("tokenizer")
    admitted_tokenizer = admission.get("tokenizer")
    _require(
        isinstance(build_tokenizer, dict)
        and admitted_tokenizer == build_tokenizer
        and isinstance(build_tokenizer.get("path"), str)
        and os.path.isabs(build_tokenizer["path"])
        and os.path.realpath(build_tokenizer["path"]) == build_tokenizer["path"]
        and _is_sha256(build_tokenizer.get("sha256"))
        and build_tokenizer.get("max_tokens") == 2048,
        "build/admission tokenizer binding changed",
    )

    source_reports = build.get("inputs")
    _require(
        _source_scans_are_substantively_zero(source_reports),
        "build source novelty scans are not substantively zero",
    )
    hard_scan = admission.get("hard_scan")
    _require(isinstance(hard_scan, dict), "admission hard scan is missing")
    _require(
        hard_scan.get("source_reports") == source_reports
        and hard_scan.get("all_source_scans_zero")
        is _source_scans_are_substantively_zero(source_reports)
        and hard_scan.get("cross_board", {}).get("exact_prompt_hits") == 0
        and hard_scan.get("cross_board", {}).get("ngram13_hits") == 0
        and hard_scan.get("cross_board", {}).get("program_hits") == 0
        and hard_scan.get("cross_board_zero") is True,
        "admission hard scan does not reproduce the build scans",
    )
    _require(
        admission.get("deterministic_distinct_seeds")
        is (FROZEN_BOARD_SEEDS["calibration"] != FROZEN_BOARD_SEEDS["confirmation"]),
        "admission seed-separation invariant changed",
    )
    _require(
        admission.get("calibration_confirmation_regimes_disjoint")
        is set(CALIBRATION_REGIMES).isdisjoint(CONFIRMATION_REGIMES),
        "admission regime-separation invariant changed",
    )
    quota = admission.get("confirmation_empirical_quota")
    _require(
        isinstance(quota, dict)
        and quota.get("scope") == "frozen confirmation rows only"
        and quota.get("all_exact_cells_required") is True
        and quota.get("extrapolation_beyond_frozen_board_forbidden") is True
        and quota.get("check_required") is True
        and quota.get("passes") is True,
        "admission confirmation quota changed",
    )
    expected_quota_partitions = {
        partition: {
            "rows": EXPECTED_PARTITION_ROWS[partition],
            "exact_cells": 40,
            "rows_per_exact_cell": EXPECTED_CELL_ROWS["confirmation"],
            "acceptance_quota_fraction_per_exact_cell": 0.40,
            "minimum_accepted_per_exact_cell": MIN_EXACT_CELL_CERTIFICATES,
            "minimum_accepted": MIN_PARTITION_CERTIFICATES,
            "maximum_false_certificates": 0,
        }
        for partition in CONFIRMATION_REGIMES
    }
    _require(
        quota.get("partitions") == expected_quota_partitions,
        "admission confirmation quota partitions changed",
    )

    admitted_build = admission.get("build_manifest")
    _require(isinstance(admitted_build, dict), "admission build binding is missing")
    recomputed_build_checks_pass = _recompute_pass_checks(
        admitted_build.get("checks"), "build admission"
    )
    _require(
        admitted_build.get("path") == build_path
        and admitted_build.get("sha256") == build_sha256
        and admitted_build.get("all_checks_pass") is recomputed_build_checks_pass
        and recomputed_build_checks_pass,
        "admission does not substantively validate the loaded build manifest",
    )
    _require(
        admission.get("all_checks_pass") is True
        and admission.get("r10_score_run_precondition_satisfied") is True,
        "admission aggregate precondition failed",
    )


def _load_hash_bound_json(
    path, expected_sha256: str, description: str
) -> tuple[str, dict]:
    _require(_is_sha256(expected_sha256), "{} hash is invalid".format(description))
    real_path, raw, _ = _read_regular_file_once(path, description)
    _require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "{} hash mismatch".format(description),
    )
    return real_path, _json_object_from_bytes(raw, description)


def validate_gate_bundle(
    manifest_path,
    *,
    expected_manifest_sha256: str,
    admission_path,
    expected_admission_sha256: str,
    expected_board_bindings: dict[str, dict[str, str]],
    expected_evaluator_sha256: str,
    expected_extractor_sha256: str,
    evaluator_path,
    extractor_path,
    expected_code_revision: str,
    expected_adapter_sha256: str,
    expected_seed: int,
    repo_root=REPO_ROOT,
    expected_runtime: dict[str, str] | None = None,
    live_git_revision: str | None = None,
    committed_source_hashes: dict[str, str] | None = None,
    manifest_bytes: bytes | None = None,
    admission_bytes: bytes | None = None,
    build_bytes: bytes | None = None,
    trusted_source_hashes: dict[str, str] | None = None,
) -> GateBundle:
    """Validate the score-blind gate, code identity, and every hash edge."""
    manifest_real = os.path.realpath(manifest_path)
    admission_real = os.path.realpath(admission_path)
    _require(
        os.path.abspath(os.fspath(manifest_path)) == manifest_real
        and os.path.abspath(os.fspath(admission_path)) == admission_real,
        "gate and admission paths must be absolute and canonical",
    )
    if manifest_bytes is None:
        manifest_real, manifest = _load_hash_bound_json(
            manifest_path, expected_manifest_sha256, "gate manifest"
        )
    else:
        _require(
            hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha256,
            "gate manifest hash mismatch",
        )
        manifest = _json_object_from_bytes(manifest_bytes, "gate manifest")
    if admission_bytes is None:
        admission_real, admission = _load_hash_bound_json(
            admission_path, expected_admission_sha256, "gate admission"
        )
    else:
        _require(
            hashlib.sha256(admission_bytes).hexdigest() == expected_admission_sha256,
            "gate admission hash mismatch",
        )
        admission = _json_object_from_bytes(admission_bytes, "gate admission")
    _reject_invalid_confidence_claims(manifest)
    _require(
        manifest.get("manifest") == FROZEN_GATE_MANIFEST,
        "invalid frozen gate manifest",
    )
    _require(manifest.get("schema") == BOARD_SCHEMA, "invalid gate board schema")
    _require(
        manifest.get("frozen_before_scores") is True
        and manifest.get("required_before_any_r10_score_run") is True
        and manifest.get("board_gate_satisfied") is True
        and manifest.get("score_outputs_read") is False
        and manifest.get("score_artifacts") == [],
        "frozen gate manifest is not a score-blind precondition",
    )
    admission_binding = manifest.get("admission_report")
    _require(
        isinstance(admission_binding, dict)
        and admission_binding.get("audit") == GATE_ADMISSION_AUDIT
        and admission_binding.get("path") == admission_real
        and admission_binding.get("sha256") == expected_admission_sha256,
        "frozen gate does not bind the exact admission report",
    )

    root = Path(repo_root).resolve()
    evaluator_real = Path(evaluator_path).resolve()
    extractor_real = Path(extractor_path).resolve()
    evaluator_relative = _repo_relative_source(evaluator_real, root, "evaluator source")
    extractor_relative = _repo_relative_source(extractor_real, root, "extractor source")
    identity = validate_code_identity(
        manifest.get("code_identity"),
        repo_root=root,
        expected_git_revision=expected_code_revision,
        required_sources={
            evaluator_relative: expected_evaluator_sha256,
            extractor_relative: expected_extractor_sha256,
        },
        expected_runtime=expected_runtime,
        live_git_revision=live_git_revision,
        hash_cache=trusted_source_hashes,
        committed_source_hashes=committed_source_hashes,
    )
    admission_identity = admission.get("code_identity")
    admission_identity_digest = admission.get("code_identity_aggregate_sha256")
    _require(
        admission_identity is None or admission_identity == identity,
        "gate admission code_identity differs from the frozen gate",
    )
    _require(
        admission_identity_digest is None
        or admission_identity_digest == identity["aggregate_sha256"],
        "gate admission code identity aggregate mismatch",
    )
    frozen_build = manifest.get("build_manifest")
    _require(
        isinstance(frozen_build, dict)
        and frozen_build.get("build") == GATE_MANIFEST_BUILD
        and isinstance(frozen_build.get("path"), str)
        and _is_sha256(frozen_build.get("sha256")),
        "frozen gate does not bind a v2 build manifest",
    )
    if build_bytes is None:
        build_real, build = _load_hash_bound_json(
            frozen_build["path"], frozen_build["sha256"], "build manifest"
        )
    else:
        build_real = os.path.realpath(frozen_build["path"])
        _require(
            hashlib.sha256(build_bytes).hexdigest() == frozen_build["sha256"],
            "build manifest hash mismatch",
        )
        build = _json_object_from_bytes(build_bytes, "build manifest")
    _require(
        build_real == frozen_build["path"],
        "frozen gate build path is not canonical",
    )
    _validate_build_and_admission_content(
        manifest=manifest,
        admission=admission,
        build=build,
        build_path=build_real,
        build_sha256=frozen_build["sha256"],
        expected_board_bindings=expected_board_bindings,
    )
    gate_boards = manifest.get("boards")
    _require(isinstance(gate_boards, dict), "frozen gate lacks boards")
    for board_name in ("calibration", "confirmation"):
        board = gate_boards.get(board_name)
        expected = _expected_summary(board_name)
        _require(isinstance(board, dict), "frozen gate board is missing")
        for key in ("rows", "regimes", "expected_cell_count", "rows_per_exact_cell"):
            _require(
                board.get(key) == expected[key],
                "frozen gate {} {} changed".format(board_name, key),
            )
        _require(
            board.get("sha256") == expected_board_bindings[board_name]["data_sha256"],
            "frozen gate {} board hash mismatch".format(board_name),
        )
        _require(
            board.get("structural_admission", {}).get("sha256")
            == expected_board_bindings[board_name]["structural_admission_sha256"],
            "frozen gate {} structural admission mismatch".format(board_name),
        )
        _require(
            board.get("referential_label_admission", {}).get("sha256")
            == expected_board_bindings[board_name]["label_admission_sha256"],
            "frozen gate {} label admission mismatch".format(board_name),
        )
    expected_partitions = {
        "calibration": {
            "fit_iid": {
                "rows": 400,
                "depths": [4, 8],
                "numeric_profile": "in_range",
                "exact_cells": 40,
                "rows_per_cell": 10,
                "rows_per_exact_cell": 10,
            },
            "depth_ood": {
                "rows": 400,
                "depths": [16, 32],
                "numeric_profile": "shifted",
                "exact_cells": 40,
                "rows_per_cell": 10,
                "rows_per_exact_cell": 10,
            },
        },
        "confirmation": {
            "language_ood": {
                "rows": 920,
                "depths": [4, 8],
                "numeric_profile": "in_range",
                "exact_cells": 40,
                "rows_per_cell": 23,
                "rows_per_exact_cell": 23,
                "minimum_accepted_per_exact_cell": MIN_EXACT_CELL_CERTIFICATES,
                "minimum_accepted": MIN_PARTITION_CERTIFICATES,
                "maximum_false_certificates": 0,
            },
            "full_ood": {
                "rows": 920,
                "depths": [16, 32],
                "numeric_profile": "shifted",
                "exact_cells": 40,
                "rows_per_cell": 23,
                "rows_per_exact_cell": 23,
                "minimum_accepted_per_exact_cell": MIN_EXACT_CELL_CERTIFICATES,
                "minimum_accepted": MIN_PARTITION_CERTIFICATES,
                "maximum_false_certificates": 0,
            },
        },
    }
    actual_partitions = manifest.get("partitions")
    _require(isinstance(actual_partitions, dict), "frozen gate partitions are missing")
    for board_name, partitions in expected_partitions.items():
        actual_board = actual_partitions.get(board_name)
        _require(
            isinstance(actual_board, dict) and set(actual_board) == set(partitions),
            "frozen gate {} partitions changed".format(board_name),
        )
        for partition, expected in partitions.items():
            actual = actual_board.get(partition)
            _require(
                isinstance(actual, dict) and actual == expected,
                "frozen gate partition geometry changed: {}".format(partition),
            )
    calibration_gate = manifest.get("calibration_threshold", {})
    _require(
        calibration_gate.get("scope") == "frozen calibration rows only"
        and calibration_gate.get("quantile") == CALIBRATION_QUANTILE
        and calibration_gate.get("partitions_pooled_only_for_calibration")
        == list(CALIBRATION_REGIMES)
        and calibration_gate.get("threshold_count") == 1
        and calibration_gate.get("confirmation_rows_must_not_influence_threshold")
        is True
        and calibration_gate.get("post_score_tuning_forbidden") is True
        and calibration_gate.get("extrapolation_beyond_frozen_board_forbidden") is True,
        "frozen calibration threshold contract changed",
    )
    confirmation_gate = manifest.get("confirmation_thresholds", {})
    _require(
        confirmation_gate.get("scope") == "frozen confirmation rows only"
        and confirmation_gate.get("minimum_selective_coverage_each_partition")
        == MIN_PARTITION_SELECTIVE_COVERAGE
        and confirmation_gate.get("acceptance_quota_fraction_per_exact_cell")
        == MIN_PARTITION_SELECTIVE_COVERAGE
        and confirmation_gate.get("acceptance_quota_rounding") == "ceil(0.40 * 23) = 10"
        and confirmation_gate.get("exact_cells_each_partition") == 40
        and confirmation_gate.get("rows_each_exact_cell")
        == EXPECTED_CELL_ROWS["confirmation"]
        and confirmation_gate.get("minimum_accepted_each_exact_cell")
        == MIN_EXACT_CELL_CERTIFICATES
        and confirmation_gate.get("minimum_accepted_each_partition")
        == MIN_PARTITION_CERTIFICATES
        and confirmation_gate.get("maximum_false_certificates_each_exact_cell") == 0
        and confirmation_gate.get("maximum_false_certificates_each_partition") == 0
        and confirmation_gate.get("minimum_empirical_selective_accuracy_each_partition")
        == MIN_PARTITION_SELECTIVE_ACCURACY
        and confirmation_gate.get("all_exact_cells_required") is True
        and confirmation_gate.get("pooled_partition_substitution_forbidden") is True
        and confirmation_gate.get("extrapolation_beyond_frozen_board_forbidden")
        is True,
        "frozen confirmation threshold contract changed",
    )
    implementations = manifest.get("implementations", {})
    _require(
        implementations.get("evaluator", {}).get("identifier") == AUDIT
        and implementations.get("evaluator", {}).get("path") == str(evaluator_real)
        and implementations.get("evaluator", {}).get("sha256")
        == expected_evaluator_sha256,
        "frozen gate evaluator hash mismatch",
    )
    _require(
        implementations.get("extractor", {}).get("identifier") == SCORE_AUDIT
        and implementations.get("extractor", {}).get("path") == str(extractor_real)
        and implementations.get("extractor", {}).get("sha256")
        == expected_extractor_sha256
        and implementations.get("extractor", {}).get("expected_seed") == expected_seed,
        "frozen gate extractor binding mismatch",
    )
    _require(
        implementations.get("expected_adapter_sha256") == expected_adapter_sha256,
        "frozen gate adapter hash mismatch",
    )

    _require(
        manifest.get("score_outputs_read") is False
        and manifest.get("score_artifacts") == [],
        "gate manifest is not score-blind",
    )

    _require(
        admission.get("audit") == GATE_ADMISSION_AUDIT,
        "invalid independent gate admission audit",
    )
    _require(admission.get("schema") == BOARD_SCHEMA, "admission schema changed")
    _require(admission.get("cpu_only") is True, "gate admission is not CPU-only")
    _require(
        admission.get("score_outputs_read") is False
        and admission.get("score_artifacts") == [],
        "gate admission is not score-blind",
    )
    _require(admission.get("all_checks_pass") is True, "gate admission failed")
    _require(
        admission.get("r10_score_run_precondition_satisfied") is True,
        "gate admission does not authorize score extraction",
    )
    manifest_binding = admission.get("build_manifest")
    _require(
        isinstance(manifest_binding, dict)
        and manifest_binding.get("path") == frozen_build.get("path")
        and manifest_binding.get("sha256") == frozen_build.get("sha256")
        and manifest_binding.get("all_checks_pass") is True,
        "gate admission does not bind the frozen build manifest",
    )
    admitted_boards = admission.get("boards")
    _require(isinstance(admitted_boards, dict), "gate admission lacks boards")
    for board_name in ("calibration", "confirmation"):
        board = admitted_boards.get(board_name)
        _validate_frozen_board_summary(
            board, board_name, "gate admission {}".format(board_name)
        )
        _require(
            board.get("sha256") == expected_board_bindings[board_name]["data_sha256"],
            "gate admission {} board hash mismatch".format(board_name),
        )
        _require(
            board.get("all_checks_pass") is True,
            "gate admission {} checks failed".format(board_name),
        )
    compatibility = admission.get("extractor_compatibility_admissions")
    _require(
        isinstance(compatibility, dict)
        and compatibility.get("enabled") is True
        and compatibility.get("all_checks_pass") is True,
        "gate admission lacks passing extractor admissions",
    )
    compatible_boards = compatibility.get("boards")
    _require(isinstance(compatible_boards, dict), "compatibility boards are missing")
    for board_name in ("calibration", "confirmation"):
        board = compatible_boards.get(board_name)
        _require(
            isinstance(board, dict) and board.get("all_checks_pass") is True,
            "{} compatibility admissions failed".format(board_name),
        )
        structural = board.get("structural")
        labels = board.get("referential_labels")
        _require(
            isinstance(structural, dict)
            and structural.get("sha256")
            == expected_board_bindings[board_name]["structural_admission_sha256"],
            "{} structural admission is not gate-bound".format(board_name),
        )
        _require(
            isinstance(labels, dict)
            and labels.get("sha256")
            == expected_board_bindings[board_name]["label_admission_sha256"],
            "{} label admission is not gate-bound".format(board_name),
        )
    return GateBundle(
        manifest_path=manifest_real,
        manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        admission_path=admission_real,
        admission_sha256=expected_admission_sha256,
        admission=admission,
        build_path=build_real,
        build_sha256=frozen_build["sha256"],
        build=build,
        code_identity=identity,
    )


def _artifact_path_and_hash(
    report: dict,
    path_key: str,
    hash_key: str,
    hash_cache: dict[str, str] | None = None,
    *,
    payload: bytes | None = None,
    trusted_artifact_hashes: dict[str, str] | None = None,
) -> tuple[str, str]:
    path = report.get(path_key)
    expected = report.get(hash_key)
    _require(
        isinstance(path, str) and os.path.isabs(path),
        "{} must be absolute".format(path_key),
    )
    _require(os.path.realpath(path) == path, "{} must be a real path".format(path_key))
    _require(_is_sha256(expected), "{} is not a SHA-256".format(hash_key))
    if payload is not None:
        _require(payload, "{} payload is empty".format(path_key))
        actual = hashlib.sha256(payload).hexdigest()
    elif trusted_artifact_hashes is not None and path in trusted_artifact_hashes:
        actual = trusted_artifact_hashes[path]
    else:
        _require(
            Path(path).is_file() and Path(path).stat().st_size > 0,
            "missing artifact: {}".format(path),
        )
        actual = _sha256_cached(path, hash_cache)
    _require(actual == expected, "{} hash mismatch".format(path_key))
    return path, actual


def _validate_admissions(
    report: dict,
    artifacts: dict[str, tuple[str, str]],
    artifact_payloads: dict[str, bytes] | None = None,
) -> None:
    artifact_payloads = artifact_payloads or {}
    structural = (
        _json_object_from_bytes(
            artifact_payloads["structural_admission"], "structural admission"
        )
        if "structural_admission" in artifact_payloads
        else _load_json_object(
            artifacts["structural_admission"][0], "structural admission"
        )
    )
    labels = (
        _json_object_from_bytes(
            artifact_payloads["referential_label_admission"],
            "referential-label admission",
        )
        if "referential_label_admission" in artifact_payloads
        else _load_json_object(
            artifacts["referential_label_admission"][0],
            "referential-label admission",
        )
    )
    metadata = report["adapter_metadata"]
    training_sha256 = report["r9c_training_data_sha256"]

    _require(
        structural.get("audit") == STRUCTURAL_ADMISSION_AUDIT,
        "invalid structural admission audit",
    )
    _require(structural.get("all_checks_pass") is True, "structural admission failed")
    _require(
        structural.get("eval_sha256") == report["data_sha256"],
        "structural admission board mismatch",
    )
    _require(
        structural.get("train_sha256") == training_sha256,
        "structural admission training mismatch",
    )
    _require(
        structural.get("tokenizer_sha256") == report["tokenizer_sha256"],
        "structural admission tokenizer mismatch",
    )

    _require(
        labels.get("audit") == LABEL_ADMISSION_AUDIT, "invalid label admission audit"
    )
    _require(
        labels.get("all_checks_pass") is True, "referential-label admission failed"
    )
    _require(
        labels.get("tokenizer_sha256") == report["tokenizer_sha256"],
        "label admission tokenizer mismatch",
    )
    datasets = labels.get("datasets")
    _require(isinstance(datasets, dict), "label admission lacks datasets")
    evaluation = datasets.get("eval")
    training = datasets.get("train")
    _require(
        isinstance(evaluation, dict) and evaluation.get("all_checks_pass") is True,
        "evaluation labels were not admitted",
    )
    _require(
        isinstance(training, dict) and training.get("all_checks_pass") is True,
        "training labels were not admitted",
    )
    _require(
        evaluation.get("sha256") == report["data_sha256"],
        "label admission board mismatch",
    )
    _require(
        training.get("sha256") == training_sha256, "label admission training mismatch"
    )

    _require(
        metadata.get("data_sha256") == training_sha256, "adapter training hash mismatch"
    )
    _require(
        report.get("r9c_training_structural_admission_sha256")
        == metadata.get("admission_sha256"),
        "report does not bind the adapter's training structural admission",
    )
    _require(
        report.get("r9c_training_referential_label_admission_sha256")
        == metadata.get("label_admission_sha256"),
        "report does not bind the adapter's training label admission",
    )


def _validate_metadata(
    report: dict,
    artifacts: dict[str, tuple[str, str]],
    hash_cache: dict[str, str] | None = None,
    trusted_artifact_hashes: dict[str, str] | None = None,
) -> None:
    metadata = report.get("adapter_metadata")
    pointer = report.get("pointer_adapter_metadata")
    _require(isinstance(metadata, dict), "score report lacks adapter metadata")
    _require(isinstance(pointer, dict), "score report lacks pointer metadata")
    _require(metadata.get("protocol") == R9C_PROTOCOL, "invalid R9c protocol")
    _require(metadata.get("arm") == "no_syndrome", "R10 requires no_syndrome scores")
    _require(
        metadata.get("arm_config") == NO_SYNDROME_CONFIG,
        "no_syndrome arm config changed",
    )
    _require(
        metadata.get("pointer_protocol") == POINTER_PROTOCOL,
        "invalid pointer protocol binding",
    )
    _require(
        metadata.get("pointer_parameters_trainable") == 0, "pointer was not frozen"
    )
    rounds = metadata.get("rounds")
    _require(
        isinstance(rounds, int) and not isinstance(rounds, bool) and rounds > 0,
        "invalid replay rounds",
    )
    _require(
        metadata.get("base_sha256") == report.get("base_sha256"),
        "adapter/base mismatch",
    )
    _require(
        metadata.get("pointer_adapter_sha256") == report.get("pointer_adapter_sha256"),
        "adapter/pointer mismatch",
    )
    _require(
        metadata.get("tokenizer_sha256") == report.get("tokenizer_sha256"),
        "adapter/tokenizer mismatch",
    )
    _require(
        metadata.get("final_adapter_sha256") == report.get("adapter_state_sha256"),
        "adapter state hash mismatch",
    )
    _require(
        report.get("adapter_step") == "syndrome_adapter_ep1", "adapter step changed"
    )
    _require(
        report.get("r9c_training_data") == metadata.get("data"),
        "training data path mismatch",
    )
    _require(
        report.get("r9c_training_data_sha256") == metadata.get("data_sha256"),
        "training data hash mismatch",
    )

    _require(
        pointer.get("protocol") == POINTER_PROTOCOL, "invalid pointer metadata protocol"
    )
    _require(pointer.get("role_mode") == "pointer", "R10 requires pointer role mode")
    _require(
        pointer.get("base_parameters_trainable") == 0, "pointer base was not frozen"
    )
    _require(
        pointer.get("base_sha256") == report.get("base_sha256"), "pointer/base mismatch"
    )
    _require(
        pointer.get("data_sha256") == report.get("r9c_training_data_sha256"),
        "pointer training mismatch",
    )
    _require(
        pointer.get("admission_sha256") == metadata.get("admission_sha256"),
        "pointer structural admission mismatch",
    )
    _require(
        pointer.get("label_admission_sha256") == metadata.get("label_admission_sha256"),
        "pointer label admission mismatch",
    )
    _require(
        int(pointer.get("hidden", -1)) == int(metadata.get("pointer_hidden", -2)),
        "pointer width mismatch",
    )

    for key, hash_key in (
        ("base_checkpoint", "base_sha256"),
        ("data", "data_sha256"),
        ("admission", "admission_sha256"),
        ("label_admission", "label_admission_sha256"),
    ):
        path = pointer.get(key)
        expected = pointer.get(hash_key)
        _require(
            isinstance(path, str)
            and (
                (trusted_artifact_hashes is not None and path in trusted_artifact_hashes)
                or Path(path).is_file()
            ),
            "pointer metadata missing {}".format(key),
        )
        actual = (
            trusted_artifact_hashes[path]
            if trusted_artifact_hashes is not None and path in trusted_artifact_hashes
            else _sha256_cached(path, hash_cache)
        )
        _require(
            _is_sha256(expected) and actual == expected,
            "pointer metadata {} hash mismatch".format(key),
        )

    replay = report.get("replay")
    _require(isinstance(replay, dict), "score report lacks replay contract")
    _require(
        replay
        == {
            "arm": "no_syndrome",
            "mode": "fixed",
            "adaptive": False,
            "rounds": rounds,
            "conditioning": "directional",
            "use_syndrome": False,
            "shuffle_goal": False,
        },
        "score replay contract changed",
    )


def _expected_board_fields(row: dict) -> dict:
    keys = row.get("keys")
    operations = row.get("operations")
    initial = row.get("initial")
    query = row.get("query")
    _require(row.get("schema") == BOARD_SCHEMA, "board schema changed")
    _require(
        isinstance(keys, list) and len(keys) == 2 and len(set(keys)) == 2,
        "board keys are invalid",
    )
    _require(
        isinstance(operations, list) and operations, "board operations are invalid"
    )
    depth = _exact_int(row.get("depth"), "board depth")
    _require(depth == len(operations), "board depth differs from operations")
    _require(isinstance(initial, dict), "board initial state is invalid")
    _require(isinstance(query, dict), "board query is invalid")
    family = row.get("family")
    _require(isinstance(family, str) and family, "board family is invalid")
    try:
        operation_targets = tuple(
            OPCODE_TO_ID[opcode_for(item, keys)] for item in operations
        )
        operation_values = tuple(operation_value(item) for item in operations)
        initial_state = tuple(int(initial[key]) for key in keys)
        query_target = QUERY_TO_ID[query_for(query, keys)]
        answer = int(row["answer"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationContractError(
            "board structured program is invalid: {}".format(error)
        ) from error
    return {
        "reference": str(row.get("reference", "")),
        "regime": str(row.get("eval_regime", "")),
        "family": family,
        "operation_targets": operation_targets,
        "operation_values": operation_values,
        "initial_state": initial_state,
        "query_target": query_target,
        "answer": answer,
    }


def _normalized_candidate_value(opcode_id: int, event_value: int) -> int:
    name = OPCODES[opcode_id]
    return event_value if name in NUMERIC_OPCODES else 0


def _program_transform(
    opcodes: Sequence[int], values: Sequence[int]
) -> ExactAffineTransform:
    _require(len(opcodes) == len(values), "program opcode/value length mismatch")
    return chronological_compose(
        operation_transform(opcode, _normalized_candidate_value(opcode, value))
        for opcode, value in zip(opcodes, values)
    )


def _validate_score_record_preflight(
    raw: dict, board_row: dict, expected_index: int
) -> None:
    """Validate all non-probability row content before any score is inspected."""
    expected_keys = {
        "index",
        "reference",
        "regime",
        "operation_targets",
        "operation_values",
        "initial_state",
        "query_target",
        "answer",
        "joint_probabilities",
        "forward_probabilities",
        "backward_probabilities",
        "query_probabilities",
    }
    _require(
        isinstance(raw, dict) and set(raw) == expected_keys,
        "score record schema changed",
    )
    index = _exact_int(raw["index"], "record index")
    _require(index == expected_index, "record indices are not contiguous and ordered")
    _require(
        isinstance(raw["reference"], str) and raw["reference"],
        "record reference is empty",
    )
    _require(
        isinstance(raw["regime"], str) and raw["regime"],
        "record regime is invalid",
    )
    targets_raw = raw["operation_targets"]
    values_raw = raw["operation_values"]
    _require(
        isinstance(targets_raw, list) and targets_raw,
        "record has no operation targets",
    )
    _require(
        isinstance(values_raw, list) and len(values_raw) == len(targets_raw),
        "operation values have wrong length",
    )
    targets = tuple(_exact_int(value, "operation target") for value in targets_raw)
    values = tuple(_exact_int(value, "operation value") for value in values_raw)
    _require(
        all(0 <= value < len(OPCODES) for value in targets),
        "operation target is out of range",
    )
    for target, value in zip(targets, values):
        if OPCODES[target] not in NUMERIC_OPCODES:
            _require(value == 0, "true structural operations must have event value zero")
    initial_raw = raw["initial_state"]
    _require(
        isinstance(initial_raw, list) and len(initial_raw) == 2,
        "initial state must have two integers",
    )
    initial = tuple(_exact_int(value, "initial state") for value in initial_raw)
    query_target = _exact_int(raw["query_target"], "query target")
    answer = _exact_int(raw["answer"], "answer")
    _require(0 <= query_target < len(QUERIES), "query target is out of range")
    expected = _expected_board_fields(board_row)
    actual = {
        "reference": raw["reference"],
        "regime": raw["regime"],
        "family": expected["family"],
        "operation_targets": targets,
        "operation_values": values,
        "initial_state": initial,
        "query_target": query_target,
        "answer": answer,
    }
    _require(
        actual == expected,
        "record {}/reference {} does not bind its board row".format(
            index, raw["reference"]
        ),
    )
    _require(
        _program_transform(targets, values).answer(initial, query_target) == answer,
        "record {} structured oracle does not reproduce its answer".format(index),
    )
    _event_source_payloads(board_row, raw["reference"])


def _validate_score_record(
    raw: dict, board_row: dict, expected_index: int
) -> ScoreRecord:
    expected_keys = {
        "index",
        "reference",
        "regime",
        "operation_targets",
        "operation_values",
        "initial_state",
        "query_target",
        "answer",
        "joint_probabilities",
        "forward_probabilities",
        "backward_probabilities",
        "query_probabilities",
    }
    _require(
        isinstance(raw, dict) and set(raw) == expected_keys,
        "score record schema changed",
    )
    index = _exact_int(raw["index"], "record index")
    _require(index == expected_index, "record indices are not contiguous and ordered")
    reference = raw["reference"]
    regime = raw["regime"]
    _require(isinstance(reference, str) and reference, "record reference is empty")
    _require(isinstance(regime, str) and regime, "record regime is invalid")

    targets_raw = raw["operation_targets"]
    values_raw = raw["operation_values"]
    _require(
        isinstance(targets_raw, list) and targets_raw, "record has no operation targets"
    )
    _require(
        isinstance(values_raw, list) and len(values_raw) == len(targets_raw),
        "operation values have wrong length",
    )
    targets = tuple(_exact_int(value, "operation target") for value in targets_raw)
    values = tuple(_exact_int(value, "operation value") for value in values_raw)
    _require(
        all(0 <= value < len(OPCODES) for value in targets),
        "operation target is out of range",
    )
    for target, value in zip(targets, values):
        if OPCODES[target] not in NUMERIC_OPCODES:
            _require(
                value == 0, "true structural operations must have event value zero"
            )

    initial_raw = raw["initial_state"]
    _require(
        isinstance(initial_raw, list) and len(initial_raw) == 2,
        "initial state must have two integers",
    )
    initial = tuple(_exact_int(value, "initial state") for value in initial_raw)
    query_target = _exact_int(raw["query_target"], "query target")
    answer = _exact_int(raw["answer"], "answer")
    _require(0 <= query_target < len(QUERIES), "query target is out of range")

    board_expected = _expected_board_fields(board_row)
    actual_fields = {
        "reference": reference,
        "regime": regime,
        "family": board_expected["family"],
        "operation_targets": targets,
        "operation_values": values,
        "initial_state": initial,
        "query_target": query_target,
        "answer": answer,
    }
    _require(
        actual_fields == board_expected,
        "record {}/reference {} does not bind its board row".format(index, reference),
    )
    oracle = _program_transform(targets, values)
    _require(
        oracle.answer(initial, query_target) == answer,
        "record {} structured oracle does not reproduce its answer".format(index),
    )
    event_payloads = _event_source_payloads(board_row, reference)

    joint_rows = raw["joint_probabilities"]
    forward_rows = raw["forward_probabilities"]
    backward_rows = raw["backward_probabilities"]
    depth = len(targets)
    for name, rows in (
        ("joint", joint_rows),
        ("forward", forward_rows),
        ("backward", backward_rows),
    ):
        _require(
            isinstance(rows, list) and len(rows) == depth,
            "{} probability depth mismatch".format(name),
        )
    joint = tuple(
        _validate_probability_row(row, len(OPCODES), "joint[{}]".format(offset))
        for offset, row in enumerate(joint_rows)
    )
    forward = tuple(
        _validate_probability_row(row, len(OPCODES), "forward[{}]".format(offset))
        for offset, row in enumerate(forward_rows)
    )
    backward = tuple(
        _validate_probability_row(row, len(OPCODES), "backward[{}]".format(offset))
        for offset, row in enumerate(backward_rows)
    )
    for offset, (joint_row, forward_row, backward_row) in enumerate(
        zip(joint, forward, backward)
    ):
        _validate_joint_distribution(
            joint_row, forward_row, backward_row, "event {}".format(offset)
        )
    query = _validate_probability_row(raw["query_probabilities"], len(QUERIES), "query")

    return ScoreRecord(
        index=index,
        reference=reference,
        regime=regime,
        family=board_expected["family"],
        operation_targets=targets,
        operation_values=values,
        initial_state=initial,  # type: ignore[arg-type]
        query_target=query_target,
        answer=answer,
        joint_probabilities=joint,
        forward_probabilities=forward,
        backward_probabilities=backward,
        query_probabilities=query,
        source_record_sha256=canonical_sha256(board_row),
        event_source_payloads=event_payloads,
    )


def validate_score_report(
    path,
    *,
    board_name: str,
    expected_report_sha256: str,
    expected_data_sha256: str,
    expected_structural_admission_sha256: str,
    expected_label_admission_sha256: str,
    extractor_path,
    expected_extractor_sha256: str,
    expected_evaluator_sha256: str,
    gate_bundle: GateBundle,
    expected_code_revision: str,
    expected_adapter_sha256: str = EXPECTED_ADAPTER_SHA256,
    expected_seed: int = EXPECTED_EXTRACTOR_SEED,
    hash_cache: dict[str, str] | None = None,
    report_bytes: bytes | None = None,
    artifact_payloads: dict[str, bytes] | None = None,
    trusted_artifact_hashes: dict[str, str] | None = None,
    expected_chain_receipt: dict | None = None,
) -> BoundScoreReport:
    """Validate one complete score report and every artifact/provenance edge."""
    for name, value in (
        ("report", expected_report_sha256),
        ("data", expected_data_sha256),
        ("structural admission", expected_structural_admission_sha256),
        ("label admission", expected_label_admission_sha256),
        ("adapter", expected_adapter_sha256),
        ("extractor", expected_extractor_sha256),
        ("evaluator", expected_evaluator_sha256),
        ("gate manifest", gate_bundle.manifest_sha256),
        ("gate admission", gate_bundle.admission_sha256),
    ):
        _require(_is_sha256(value), "expected {} hash is invalid".format(name))
    _require(
        _is_git_revision(expected_code_revision), "expected code revision is invalid"
    )
    _require(
        gate_bundle.code_identity.get("git_revision") == expected_code_revision,
        "gate code identity revision mismatch",
    )
    extractor_path = os.path.realpath(extractor_path)
    _require(Path(extractor_path).is_file(), "missing extractor source")
    _require(
        _sha256_cached(extractor_path, hash_cache) == expected_extractor_sha256,
        "extractor source hash mismatch",
    )
    if report_bytes is None:
        report_path, raw_report, _ = _read_regular_file_once(path, "score report")
    else:
        _require(isinstance(report_bytes, bytes) and report_bytes, "score bytes are empty")
        report_path = "memory://{}/score-report".format(board_name)
        raw_report = report_bytes
    report_sha256 = hashlib.sha256(raw_report).hexdigest()
    _require(report_sha256 == expected_report_sha256, "score report hash mismatch")
    report = _json_object_from_bytes(raw_report, "score report")
    _require(report.get("audit") == SCORE_AUDIT, "invalid score report audit")
    _require(
        report.get("schema_version") == SCORE_SCHEMA_VERSION,
        "score schema version changed",
    )
    if expected_chain_receipt is not None:
        _require(
            report.get("chain_receipt") == expected_chain_receipt,
            "score report execution receipt mismatch",
        )
    expected_code_bindings = {
        "board_name": board_name,
        "code_identity_aggregate_sha256": gate_bundle.code_identity["aggregate_sha256"],
        "evaluator": os.path.realpath(__file__),
        "evaluator_sha256": expected_evaluator_sha256,
        "extractor": extractor_path,
        "extractor_sha256": expected_extractor_sha256,
        "gate_manifest": gate_bundle.manifest_path,
        "gate_manifest_sha256": gate_bundle.manifest_sha256,
        "gate_admission": gate_bundle.admission_path,
        "gate_admission_sha256": gate_bundle.admission_sha256,
    }
    _require(
        all(report.get(key) == value for key, value in expected_code_bindings.items()),
        "score report lacks the frozen evaluator/extractor/gate bindings",
    )
    _require(
        report.get("code_identity") == gate_bundle.code_identity,
        "score report code_identity mismatch",
    )
    if "code_revision" in report:
        _require(
            report["code_revision"] == expected_code_revision,
            "score code revision mismatch",
        )
    _require(report.get("seed") == expected_seed, "extractor seed changed")
    _require(
        report.get("categorical_order")
        == {
            "operations": list(OPCODES),
            "queries": list(QUERIES),
        },
        "categorical order changed",
    )

    artifact_keys = {
        "base": ("base", "base_sha256"),
        "pointer_adapter": ("pointer_adapter", "pointer_adapter_sha256"),
        "adapter": ("adapter", "adapter_sha256"),
        "data": ("data", "data_sha256"),
        "tokenizer": ("tokenizer", "tokenizer_sha256"),
        "structural_admission": ("structural_admission", "structural_admission_sha256"),
        "referential_label_admission": (
            "referential_label_admission",
            "referential_label_admission_sha256",
        ),
        "r9c_training_data": ("r9c_training_data", "r9c_training_data_sha256"),
    }
    artifact_payloads = artifact_payloads or {}
    artifacts = {
        name: _artifact_path_and_hash(
            report,
            path_key,
            hash_key,
            hash_cache,
            payload=artifact_payloads.get(name),
            trusted_artifact_hashes=trusted_artifact_hashes,
        )
        for name, (path_key, hash_key) in artifact_keys.items()
    }
    _require(
        report["adapter_sha256"] == expected_adapter_sha256,
        "frozen adapter hash mismatch",
    )
    _require(
        report["data_sha256"] == expected_data_sha256, "frozen board hash mismatch"
    )
    _require(
        report["structural_admission_sha256"] == expected_structural_admission_sha256,
        "frozen structural admission hash mismatch",
    )
    _require(
        report["referential_label_admission_sha256"] == expected_label_admission_sha256,
        "frozen label admission hash mismatch",
    )
    _validate_metadata(
        report,
        artifacts,
        hash_cache,
        trusted_artifact_hashes=trusted_artifact_hashes,
    )
    _validate_admissions(report, artifacts, artifact_payloads)

    board_rows = (
        _jsonl_from_bytes(artifact_payloads["data"])
        if "data" in artifact_payloads
        else _load_jsonl(report["data"])
    )
    raw_records = report.get("records")
    _require(
        isinstance(raw_records, list) and raw_records, "score report has no records"
    )
    _require(
        len(raw_records) == len(board_rows), "score rows do not cover the board exactly"
    )
    for index, (raw, board_row) in enumerate(zip(raw_records, board_rows)):
        _validate_score_record_preflight(raw, board_row, index)
    records = tuple(
        _validate_score_record(raw, board_rows[index], index)
        for index, raw in enumerate(raw_records)
    )
    references = tuple(record.reference for record in records)
    _require(
        len(set(references)) == len(references), "record references are not unique"
    )
    regimes = {record.regime for record in records}
    allowed = set(_board_regimes(board_name))
    _require(
        regimes == allowed,
        "score report regimes {} do not equal declared {}".format(
            sorted(regimes), sorted(allowed)
        ),
    )
    _require(
        all(any(record.regime == regime for record in records) for regime in allowed),
        "a declared regime is empty",
    )
    validate_record_geometry(records, board_name)
    _require(report.get("cases") == len(records), "score case count mismatch")
    _require(
        report.get("events") == sum(record.depth for record in records),
        "score event count mismatch",
    )
    batches = report.get("batches")
    _require(
        isinstance(batches, int)
        and not isinstance(batches, bool)
        and 0 < batches <= len(records),
        "invalid score batch count",
    )
    return BoundScoreReport(report_path, report_sha256, report, records)


def authenticate_score_bytes(
    board_name: str, payload: bytes, authentication_key: bytes
) -> AuthenticatedScoreBytes:
    _require(board_name in FROZEN_BOARD_ORDER, "authenticated board name changed")
    _require(
        isinstance(payload, bytes) and payload, "authenticated score payload is empty"
    )
    _require(
        isinstance(authentication_key, bytes) and len(authentication_key) == 32,
        "score authentication key must be 32 bytes",
    )
    digest = hashlib.sha256(payload).hexdigest()
    mac = hmac.new(
        authentication_key,
        board_name.encode("ascii") + b"\0" + payload,
        hashlib.sha256,
    ).hexdigest()
    return AuthenticatedScoreBytes(board_name, payload, digest, mac)


def _verify_authenticated_score_bytes(
    envelope: AuthenticatedScoreBytes,
    authentication_key: bytes,
    expected_board_name: str,
) -> None:
    _require(
        isinstance(envelope, AuthenticatedScoreBytes),
        "score input was not produced by the extractor envelope",
    )
    _require(
        isinstance(authentication_key, bytes) and len(authentication_key) == 32,
        "score authentication key must be 32 bytes",
    )
    _require(
        envelope.board_name == expected_board_name,
        "authenticated score board order changed",
    )
    actual_sha256 = hashlib.sha256(envelope.payload).hexdigest()
    _require(
        _is_sha256(envelope.sha256) and envelope.sha256 == actual_sha256,
        "authenticated score byte hash mismatch",
    )
    expected_mac = hmac.new(
        authentication_key,
        expected_board_name.encode("ascii") + b"\0" + envelope.payload,
        hashlib.sha256,
    ).hexdigest()
    _require(
        _is_sha256(envelope.mac) and hmac.compare_digest(envelope.mac, expected_mac),
        "authenticated score MAC mismatch",
    )


def chain_basis(
    gate_bundle: GateBundle,
    *,
    expected_adapter_sha256: str,
    expected_extractor_sha256: str,
    expected_evaluator_sha256: str,
    board_bindings: dict[str, dict[str, str]],
) -> dict:
    return {
        "audit": CHAIN_AUDIT,
        "schema_version": CHAIN_SCHEMA_VERSION,
        "adapter_sha256": expected_adapter_sha256,
        "batch_size": FROZEN_BATCH_SIZE,
        "board_order": list(FROZEN_BOARD_ORDER),
        "boards": board_bindings,
        "code_identity_aggregate_sha256": gate_bundle.code_identity[
            "aggregate_sha256"
        ],
        "determinism": FROZEN_DETERMINISM,
        "evaluator_sha256": expected_evaluator_sha256,
        "extractor_sha256": expected_extractor_sha256,
        "gate_admission_sha256": gate_bundle.admission_sha256,
        "gate_manifest_sha256": gate_bundle.manifest_sha256,
        "build_manifest_sha256": gate_bundle.build_sha256,
        "seed": EXPECTED_EXTRACTOR_SEED,
    }


def validate_chain_receipt(
    receipt: dict,
    *,
    gate_bundle: GateBundle,
    expected_adapter_sha256: str,
    expected_extractor_sha256: str,
    expected_evaluator_sha256: str,
    board_bindings: dict[str, dict[str, str]],
) -> None:
    _require(isinstance(receipt, dict), "score report lacks a chain receipt")
    expected_keys = {
        "audit",
        "schema_version",
        "attempt",
        "chain_basis",
        "chain_id",
        "device",
        "environment",
        "output_namespace",
        "runtime",
        "selection_contract",
    }
    _require(set(receipt) == expected_keys, "chain receipt schema changed")
    expected_basis = chain_basis(
        gate_bundle,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_extractor_sha256=expected_extractor_sha256,
        expected_evaluator_sha256=expected_evaluator_sha256,
        board_bindings=board_bindings,
    )
    expected_chain_id = canonical_sha256(expected_basis)
    _require(
        receipt.get("audit") == CHAIN_AUDIT
        and receipt.get("schema_version") == CHAIN_SCHEMA_VERSION
        and receipt.get("attempt") == 1
        and receipt.get("chain_basis") == expected_basis
        and receipt.get("chain_id") == expected_chain_id,
        "chain receipt does not bind the frozen score inputs",
    )
    output_namespace = receipt.get("output_namespace")
    expected_namespace = str(
        (REPO_ROOT / "train" / "r10_score_chains" / expected_chain_id).resolve()
    )
    _require(
        output_namespace == expected_namespace,
        "chain output namespace is not the deterministic one-run namespace",
    )
    _require(
        receipt.get("environment") == FROZEN_ENVIRONMENT,
        "chain environment differs from the deterministic contract",
    )
    runtime = receipt.get("runtime")
    _require(
        isinstance(runtime, dict)
        and set(runtime)
        == {"python", "torch", "tokenizers", "cuda", "cudnn"}
        and runtime["python"] == gate_bundle.code_identity["runtime"]["python"]
        and runtime["torch"] == gate_bundle.code_identity["runtime"]["torch"]
        and runtime["tokenizers"]
        == gate_bundle.code_identity["runtime"]["tokenizers"]
        and isinstance(runtime["cuda"], str)
        and runtime["cuda"]
        and isinstance(runtime["cudnn"], str)
        and runtime["cudnn"],
        "chain runtime identity changed",
    )
    device = receipt.get("device")
    _require(isinstance(device, dict), "chain device identity is missing")
    _require(
        device.get("type") == EXPECTED_DEVICE_CLASS["type"]
        and device.get("name") == EXPECTED_DEVICE_CLASS["name"]
        and device.get("compute_capability")
        == EXPECTED_DEVICE_CLASS["compute_capability"]
        and isinstance(device.get("uuid"), str)
        and device["uuid"]
        and isinstance(device.get("pci_bus_id"), str)
        and device["pci_bus_id"]
        and isinstance(device.get("total_memory"), int)
        and not isinstance(device.get("total_memory"), bool)
        and device["total_memory"] > 0
        and isinstance(device.get("multi_processor_count"), int)
        and not isinstance(device.get("multi_processor_count"), bool)
        and device["multi_processor_count"] > 0,
        "chain device identity is not the frozen H100 class",
    )
    _require(
        receipt.get("selection_contract")
        == {
            "calibration_runs": 1,
            "confirmation_runs": 1,
            "confirmation_selection": "single_in_process_run_only",
            "alternate_batch_forbidden": True,
            "namespace_reuse_forbidden": True,
        },
        "chain confirmation-selection contract changed",
    )


def evaluate_authenticated_score_chain(
    *,
    calibration_envelope: AuthenticatedScoreBytes,
    confirmation_envelope: AuthenticatedScoreBytes,
    authentication_key: bytes,
    gate_bundle: GateBundle,
    board_bindings: dict[str, dict[str, str]],
    artifact_payloads: dict[str, dict[str, bytes]],
    trusted_artifact_hashes: dict[str, str],
    extractor_path,
    extractor_sha256: str,
    evaluator_sha256: str,
    code_revision: str,
    expected_adapter_sha256: str = EXPECTED_ADAPTER_SHA256,
) -> dict:
    """Evaluate only extractor-authenticated bytes from this live process."""
    _verify_authenticated_score_bytes(
        calibration_envelope, authentication_key, "calibration"
    )
    _verify_authenticated_score_bytes(
        confirmation_envelope, authentication_key, "confirmation"
    )
    calibration_raw = _json_object_from_bytes(
        calibration_envelope.payload, "authenticated calibration scores"
    )
    confirmation_raw = _json_object_from_bytes(
        confirmation_envelope.payload, "authenticated confirmation scores"
    )
    calibration_receipt = calibration_raw.get("chain_receipt")
    _require(
        confirmation_raw.get("chain_receipt") == calibration_receipt,
        "calibration and confirmation receipts differ",
    )
    validate_chain_receipt(
        calibration_receipt,
        gate_bundle=gate_bundle,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_extractor_sha256=extractor_sha256,
        expected_evaluator_sha256=evaluator_sha256,
        board_bindings=board_bindings,
    )

    reports = {}
    for board_name, envelope in (
        ("calibration", calibration_envelope),
        ("confirmation", confirmation_envelope),
    ):
        binding = board_bindings[board_name]
        reports[board_name] = validate_score_report(
            None,
            board_name=board_name,
            expected_report_sha256=envelope.sha256,
            expected_data_sha256=binding["data_sha256"],
            expected_structural_admission_sha256=binding[
                "structural_admission_sha256"
            ],
            expected_label_admission_sha256=binding["label_admission_sha256"],
            extractor_path=extractor_path,
            expected_extractor_sha256=extractor_sha256,
            expected_evaluator_sha256=evaluator_sha256,
            gate_bundle=gate_bundle,
            expected_code_revision=code_revision,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_seed=EXPECTED_EXTRACTOR_SEED,
            report_bytes=envelope.payload,
            artifact_payloads=artifact_payloads[board_name],
            trusted_artifact_hashes=trusted_artifact_hashes,
            expected_chain_receipt=calibration_receipt,
        )

    calibration = calibrate_program_threshold(reports["calibration"].records)
    result = assess_static_confirmation(
        reports["calibration"],
        reports["confirmation"],
        calibration,
        CONFIRMATION_REGIMES,
        control_seed=EXPECTED_EXTRACTOR_SEED,
    )
    result["chain_receipt"] = calibration_receipt
    result["authenticated_score_inputs"] = {
        board_name: {
            "sha256": reports[board_name].sha256,
            "mac": envelope.mac,
            "data_sha256": reports[board_name].report["data_sha256"],
        }
        for board_name, envelope in (
            ("calibration", calibration_envelope),
            ("confirmation", confirmation_envelope),
        )
    }
    result["board_contract"] = {
        "schema": BOARD_SCHEMA,
        "calibration": validate_record_geometry(
            reports["calibration"].records, "calibration"
        ),
        "confirmation": validate_record_geometry(
            reports["confirmation"].records, "confirmation"
        ),
    }
    result["hard_argmax_parity"] = {
        "calibration": {
            "enabled": False,
            "reason": "external old-report inputs are forbidden in the one-shot chain",
        },
        "test": {
            "enabled": False,
            "reason": "external old-report inputs are forbidden in the one-shot chain",
        },
    }
    return result


def _nonconformity(probability: float) -> float:
    return math.inf if probability == 0.0 else -math.log(probability)


def calibrate_program_threshold(records: Sequence[ScoreRecord]) -> CalibrationResult:
    """Calibrate one max-over-program threshold without consulting test records."""
    records = tuple(records)
    _require(records, "calibration report contains no programs")
    _require(
        {record.regime for record in records} == set(CALIBRATION_REGIMES),
        "calibration records must contain only fit_iid and depth_ood",
    )
    scores = []
    digest_rows = []
    for record in records:
        event_scores = tuple(
            _nonconformity(probabilities[target])
            for probabilities, target in zip(
                record.joint_probabilities, record.operation_targets
            )
        )
        query_score = _nonconformity(record.query_probabilities[record.query_target])
        score = max((*event_scores, query_score))
        scores.append(score)
        digest_rows.append(
            {
                "index": record.index,
                "joint_probabilities": record.joint_probabilities,
                "operation_targets": record.operation_targets,
                "query_probabilities": record.query_probabilities,
                "query_target": record.query_target,
                "reference": record.reference,
                "regime": record.regime,
                "family": record.family,
            }
        )
    programs = len(scores)
    order_statistic = math.ceil((programs + 1) * CALIBRATION_QUANTILE)
    threshold = (
        math.inf if order_statistic > programs else sorted(scores)[order_statistic - 1]
    )
    return CalibrationResult(
        threshold=threshold,
        quantile=CALIBRATION_QUANTILE,
        programs=programs,
        events=sum(record.depth for record in records),
        order_statistic=order_statistic,
        calibration_digest=canonical_sha256(digest_rows),
    )


def candidate_ids(probabilities: Sequence[float], threshold: float) -> tuple[int, ...]:
    """Return the complete threshold set; an empty set remains empty."""
    return tuple(
        index
        for index, probability in enumerate(probabilities)
        if _nonconformity(probability) <= threshold
    )


def _argmax(probabilities: Sequence[float]) -> int:
    _require(bool(probabilities), "argmax requires a nonempty probability vector")
    return max(range(len(probabilities)), key=lambda index: probabilities[index])


def _entropy(probabilities: Sequence[float]) -> float:
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)


def _top1_margin(probabilities: Sequence[float]) -> float:
    first, second = sorted(probabilities, reverse=True)[:2]
    return first - second


def top1_analysis(record: ScoreRecord) -> dict:
    operations = tuple(_argmax(row) for row in record.joint_probabilities)
    query = _argmax(record.query_probabilities)
    transform = _program_transform(operations, record.operation_values)
    answer = transform.answer(record.initial_state, query)
    rows = (*record.joint_probabilities, record.query_probabilities)
    return {
        "answer": answer,
        "answer_correct": answer == record.answer,
        "operation_correct": sum(
            predicted == target
            for predicted, target in zip(operations, record.operation_targets)
        ),
        "operation_predictions": list(operations),
        "program_exact": operations == record.operation_targets
        and query == record.query_target,
        "query_correct": query == record.query_target,
        "query_prediction": query,
        "selection_scores": {
            "max_probability": math.fsum(math.log(max(row)) for row in rows),
            "minimum_top1_margin": min(_top1_margin(row) for row in rows),
            "maximum_entropy": max(_entropy(row) for row in rows),
        },
    }


def _operation_candidate_tuples(
    record: ScoreRecord,
    operation_candidates: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    return tuple(
        tuple(
            (
                OPCODES[opcode],
                _normalized_candidate_value(opcode, record.operation_values[offset]),
            )
            for opcode in candidates
        )
        for offset, candidates in enumerate(operation_candidates)
    )


def _workspace_tree(
    candidate_operations: Sequence[Sequence[tuple[str, int]]],
) -> tuple[AffineAmbiguityWorkspace, dict[tuple[int, int], AffineAmbiguityWorkspace]]:
    level = []
    workspaces = {}
    for index, candidates in enumerate(candidate_operations):
        workspace = workspace_from_operations(candidates, source_index=index)
        level.append((index, index + 1, workspace))
        workspaces[(index, index + 1)] = workspace
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
                continue
            left_start, left_end, left = level[offset]
            right_start, right_end, right = level[offset + 1]
            _require(left_end == right_start, "ACAW ranges are not contiguous")
            workspace = compose_workspaces(left, right)
            workspaces[(left_start, right_end)] = workspace
            next_level.append((left_start, right_end, workspace))
        level = next_level
    return level[0][2], workspaces


def _tree_nodes(node: VersionSpaceProductTree) -> Iterable[VersionSpaceProductTree]:
    yield node
    if node.left is not None:
        yield from _tree_nodes(node.left)
        yield from _tree_nodes(node.right)  # type: ignore[arg-type]


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _rational_matrix_payload(matrix) -> list[list[list[int]]]:
    return [[_fraction_payload(Fraction(value)) for value in row] for row in matrix]


def _transform_payload(transform: ExactAffineTransform) -> list[list[int]]:
    return [list(row) for row in transform.rows]


def _transform_sha256(transform: ExactAffineTransform) -> str:
    return canonical_sha256(_transform_payload(transform))


def _rational_transform_sha256(matrix) -> str:
    return canonical_sha256(_rational_matrix_payload(matrix))


def _rational_equals_transform(matrix, transform: ExactAffineTransform) -> bool:
    return all(
        Fraction(matrix[row][column]) == transform.rows[row][column]
        for row in range(3)
        for column in range(3)
    )


def _oracle_segment_transform(
    record: ScoreRecord, start: int, end: int
) -> ExactAffineTransform:
    return _program_transform(
        record.operation_targets[start:end],
        record.operation_values[start:end],
    )


def _retrieval_pointer(
    record: ScoreRecord,
    start: int,
    end: int,
    score_report_sha256: str,
    board_sha256: str,
) -> dict:
    return {
        "board_sha256": board_sha256,
        "end": end,
        "record_index": record.index,
        "reference": record.reference,
        "score_report_sha256": score_report_sha256,
        "source_record_sha256": record.source_record_sha256,
        "start": start,
    }


def _source_hot_item(record: ScoreRecord, index: int) -> dict:
    return {"kind": "source", "payload": record.event_source_payloads[index]}


def _storage_summary(
    items: Sequence[dict],
    pointers: Sequence[dict],
    source_bytes: int,
    provenance_items: Sequence[dict] = (),
    accounting_schema: str = "r10-acaw-accounting-v1",
) -> dict:
    hot_bytes = sum(len(canonical_json_bytes(item)) for item in items)
    retrieval_bytes = sum(len(canonical_json_bytes(pointer)) for pointer in pointers)
    factorized_bytes = sum(len(canonical_json_bytes(item)) for item in provenance_items)
    retrieval_provenance_bytes = retrieval_bytes + factorized_bytes
    retained_events = sum(
        item["end"] - item["start"] for item in items if item["kind"] == "source"
    )
    evicted_events = sum(
        item["end"] - item["start"] for item in items if item["kind"] == "transform"
    )
    false_segments = sum(bool(item.get("false_hot_eviction")) for item in items)
    false_events = sum(
        item["end"] - item["start"]
        for item in items
        if item["kind"] == "transform" and item.get("false_hot_eviction")
    )
    pointer_ranges = collections.Counter(
        (pointer.get("start"), pointer.get("end")) for pointer in pointers
    )
    retrieval_bound_events = 0
    unbound_events = 0
    for item in items:
        if item["kind"] != "transform":
            continue
        key = (item["start"], item["end"])
        events = item["end"] - item["start"]
        if pointer_ranges[key] > 0:
            pointer_ranges[key] -= 1
            retrieval_bound_events += events
        else:
            unbound_events += events
    orphan_pointers = sum(pointer_ranges.values())
    reader_family_events = sum(
        item["end"] - item["start"]
        for item in items
        if item["kind"] == "transform" and item.get("eviction_basis") == "reader_family"
    )
    query_driven_events = sum(
        item["end"] - item["start"]
        for item in items
        if item["kind"] == "transform"
        and item.get("eviction_basis") == "query_agreement"
    )
    return {
        "accounting_schema": accounting_schema,
        "canonical_hot_bytes": hot_bytes,
        "evicted_source_events": evicted_events,
        "external_binding_pointer_bytes": retrieval_bytes,
        "factorized_node_count": len(provenance_items),
        "factorized_provenance_bytes": factorized_bytes,
        "false_hot_eviction_events": false_events,
        "false_hot_eviction_segments": false_segments,
        "hot_plus_retrieval_provenance_bytes": (hot_bytes + retrieval_provenance_bytes),
        "irreversible_source_deletions": unbound_events,
        "orphan_retrieval_pointers": orphan_pointers,
        "query_driven_evicted_source_events": query_driven_events,
        "reader_family_evicted_source_events": reader_family_events,
        "retained_source_events": retained_events,
        "retrieval_bound_source_events": retrieval_bound_events,
        "retrieval_pointer_count": len(pointers),
        "retrieval_provenance_bytes": retrieval_provenance_bytes,
        "retrieval_reference_bytes": retrieval_bytes,
        "source_bytes": source_bytes,
        "total_canonical_bytes": (
            hot_bytes + factorized_bytes + source_bytes + retrieval_bytes
        ),
        "transform_segments": sum(item["kind"] == "transform" for item in items),
        "unbound_evicted_source_events": unbound_events,
    }


def _exact_storage(
    tree: VersionSpaceProductTree,
    record: ScoreRecord,
    score_report_sha256: str,
    board_sha256: str,
) -> tuple[dict, list[dict]]:
    serialized_items = []
    pointers = []
    segments = []
    for item in compact_frontier(tree):
        start = int(item["start"])
        end = int(item["end"])
        if item["kind"] == "source":
            for index in range(start, end):
                serialized_items.append(
                    {
                        "end": index + 1,
                        "kind": "source",
                        "payload": record.event_source_payloads[index],
                        "start": index,
                    }
                )
            continue
        transform = item["transform"]
        oracle = _oracle_segment_transform(record, start, end)
        false_eviction = transform != oracle
        pointer = _retrieval_pointer(
            record, start, end, score_report_sha256, board_sha256
        )
        pointers.append(pointer)
        serialized_items.append(
            {
                "end": end,
                "eviction_basis": "operation_transform",
                "false_hot_eviction": false_eviction,
                "kind": "transform",
                "start": start,
                "transform": _transform_payload(transform),
            }
        )
        segments.append(
            {
                "candidate_transform_sha256": _transform_sha256(transform),
                "end": end,
                "false_hot_eviction": false_eviction,
                "oracle_transform_sha256": _transform_sha256(oracle),
                "retrieval_pointer": pointer,
                "start": start,
            }
        )
    custom = _storage_summary(serialized_items, pointers, record.source_bytes)
    accounting = account_version_space_tree(tree, record.event_source_payloads)
    _require(
        accounting.retrieval_reference_count == len(segments),
        "canonical VSPT retrieval accounting differs from audited eviction segments",
    )
    _require(
        accounting.evicted_source_events == custom["evicted_source_events"]
        and accounting.retained_source_events == custom["retained_source_events"],
        "canonical VSPT source accounting differs from its compact frontier",
    )
    canonical = accounting.as_dict()
    external_binding_bytes = custom["retrieval_reference_bytes"]
    retrieval_provenance_bytes = (
        accounting.factorized_provenance_bytes
        + accounting.retrieval_reference_bytes
        + external_binding_bytes
    )
    storage = {
        "accounting_schema": canonical["accounting_schema"],
        "canonical_hot_bytes": accounting.active_hot_frontier_bytes,
        "canonical_stores": canonical["stores"],
        "evicted_source_events": accounting.evicted_source_events,
        "external_binding_pointer_bytes": external_binding_bytes,
        "factorized_node_count": accounting.factorized_node_count,
        "factorized_provenance_bytes": accounting.factorized_provenance_bytes,
        "false_hot_eviction_events": custom["false_hot_eviction_events"],
        "false_hot_eviction_segments": custom["false_hot_eviction_segments"],
        "hot_plus_retrieval_provenance_bytes": (
            accounting.active_hot_frontier_bytes + retrieval_provenance_bytes
        ),
        "irreversible_source_deletions": custom["irreversible_source_deletions"],
        "orphan_retrieval_pointers": custom["orphan_retrieval_pointers"],
        "query_driven_evicted_source_events": custom[
            "query_driven_evicted_source_events"
        ],
        "reader_family_evicted_source_events": custom[
            "reader_family_evicted_source_events"
        ],
        "retained_source_events": accounting.retained_source_events,
        "retrieval_bound_source_events": custom["retrieval_bound_source_events"],
        "retrieval_pointer_count": accounting.retrieval_reference_count,
        "retrieval_provenance_bytes": retrieval_provenance_bytes,
        "retrieval_reference_bytes": accounting.retrieval_reference_bytes,
        "source_bytes": accounting.external_source_bytes,
        "total_canonical_bytes": (
            accounting.total_canonical_bytes + external_binding_bytes
        ),
        "total_integer_growth": canonical["total_integer_growth"],
        "transform_integer_growth": canonical["transform_integer_growth"],
        "transform_segments": len(segments),
        "unbound_evicted_source_events": custom["unbound_evicted_source_events"],
    }
    return storage, segments


def _acaw_frontier(
    tree: VersionSpaceProductTree,
    workspaces: dict[tuple[int, int], AffineAmbiguityWorkspace],
    record: ScoreRecord,
    score_report_sha256: str,
    board_sha256: str,
) -> tuple[dict, list[dict]]:
    root_workspace = workspaces[(tree.start, tree.end)]
    serialized_items = [
        {
            "anchor": _rational_matrix_payload(root_workspace.anchor),
            "basis": [
                _rational_matrix_payload(direction)
                for direction in root_workspace.basis
            ],
            "end": tree.end,
            "kind": "workspace",
            "start": tree.start,
        }
    ]
    pointers = []
    segments = []
    provenance_items = [
        {
            "anchor": _rational_matrix_payload(workspace.anchor),
            "basis": [
                _rational_matrix_payload(direction) for direction in workspace.basis
            ],
            "end": end,
            "kind": "workspace_node",
            "retained_source_indices": list(workspace.retained_source_indices),
            "start": start,
        }
        for (start, end), workspace in sorted(workspaces.items())
    ]

    def visit(node: VersionSpaceProductTree) -> None:
        workspace = workspaces[(node.start, node.end)]
        if workspace.source_droppable:
            oracle = _oracle_segment_transform(record, node.start, node.end)
            false_eviction = not _rational_equals_transform(workspace.anchor, oracle)
            pointer = _retrieval_pointer(
                record,
                node.start,
                node.end,
                score_report_sha256,
                board_sha256,
            )
            pointers.append(pointer)
            serialized_items.append(
                {
                    "end": node.end,
                    "eviction_basis": "operation_transform",
                    "false_hot_eviction": false_eviction,
                    "kind": "transform",
                    "start": node.start,
                    "transform": _rational_matrix_payload(workspace.anchor),
                }
            )
            segments.append(
                {
                    "candidate_transform_sha256": _rational_transform_sha256(
                        workspace.anchor
                    ),
                    "end": node.end,
                    "false_hot_eviction": false_eviction,
                    "oracle_transform_sha256": _transform_sha256(oracle),
                    "retrieval_pointer": pointer,
                    "start": node.start,
                }
            )
            return
        if node.left is None:
            serialized_items.append(
                {
                    "end": node.end,
                    "kind": "source",
                    "payload": record.event_source_payloads[node.start],
                    "start": node.start,
                }
            )
            return
        visit(node.left)
        visit(node.right)  # type: ignore[arg-type]

    visit(tree)
    return (
        _storage_summary(
            serialized_items,
            pointers,
            record.source_bytes,
            provenance_items=provenance_items,
        ),
        segments,
    )


def _all_source_storage(record: ScoreRecord, accounting_schema: str) -> dict:
    items = [
        {
            "end": index + 1,
            "kind": "source",
            "payload": payload,
            "start": index,
        }
        for index, payload in enumerate(record.event_source_payloads)
    ]
    return _storage_summary(
        items, (), record.source_bytes, accounting_schema=accounting_schema
    )


def _exact_query_certificate(
    tree: VersionSpaceProductTree,
    record: ScoreRecord,
    query_candidates: Sequence[int],
) -> tuple[bool, int | None, tuple[int, ...]]:
    if tree.overflow or not query_candidates:
        return False, None, ()
    answers = tuple(
        sorted(
            {
                candidate.transform.answer(record.initial_state, query)
                for candidate in tree.candidates
                for query in query_candidates
            }
        )
    )
    return len(answers) == 1, answers[0] if len(answers) == 1 else None, answers


def _acaw_query_certificate(
    workspace: AffineAmbiguityWorkspace,
    record: ScoreRecord,
    query_candidates: Sequence[int],
) -> tuple[bool, int | None, bool]:
    if not query_candidates:
        return False, None, False
    certificates = tuple(
        workspace.query_certificate(record.initial_state, query)
        for query in query_candidates
    )
    annihilator_proof = all(certificate.certified for certificate in certificates)
    if not annihilator_proof:
        return False, None, False
    answers = tuple(certificate.integer_answer for certificate in certificates)
    integer_and_equal = (
        all(answer is not None for answer in answers) and len(set(answers)) == 1
    )
    return integer_and_equal, answers[0] if integer_and_equal else None, True


def _uncapped_root_size(
    record: ScoreRecord,
    operation_candidates: Sequence[Sequence[int]],
) -> int | None:
    current = {ExactAffineTransform.identity()}
    remaining_budget = UNCAPPED_PAIR_BUDGET
    for offset, candidates in enumerate(operation_candidates):
        transforms = {
            operation_transform(
                opcode,
                _normalized_candidate_value(opcode, record.operation_values[offset]),
            )
            for opcode in candidates
        }
        if not transforms:
            return 0
        pairs = len(current) * len(transforms)
        if pairs > remaining_budget:
            return None
        remaining_budget -= pairs
        current = {
            earlier.followed_by(later) for earlier in current for later in transforms
        }
        if len(current) > UNCAPPED_DISTINCT_LIMIT:
            return None
    return len(current)


def analyze_candidate_program(
    record: ScoreRecord,
    operation_candidates: Sequence[Sequence[int]],
    query_candidates: Sequence[int],
    *,
    score_report_sha256: str,
    board_sha256: str,
) -> dict:
    """Run complete categorical sets through cap-32 VSPT and exact-rational ACAW."""
    operation_candidates = tuple(
        tuple(candidates) for candidates in operation_candidates
    )
    query_candidates = tuple(query_candidates)
    _require(len(operation_candidates) == record.depth, "candidate depth mismatch")
    _require(
        all(
            len(set(candidates)) == len(candidates)
            and all(0 <= item < len(OPCODES) for item in candidates)
            for candidates in operation_candidates
        ),
        "operation candidates contain duplicate or invalid ids",
    )
    _require(
        len(set(query_candidates)) == len(query_candidates)
        and all(0 <= item < len(QUERIES) for item in query_candidates),
        "query candidates contain duplicate or invalid ids",
    )
    operation_sequence_count = math.prod(
        len(candidates) for candidates in operation_candidates
    )
    categorical_program_count = operation_sequence_count * len(query_candidates)
    categorical = {
        "candidate_query_count": len(query_candidates),
        "candidate_query_ids": list(query_candidates),
        "candidate_sequence_count": operation_sequence_count,
        "candidate_sequence_query_count": categorical_program_count,
        "empty_operation_candidate_sets": sum(
            not candidates for candidates in operation_candidates
        ),
        "empty_query_candidate_set": not query_candidates,
        "operation_candidate_counts": [
            len(candidates) for candidates in operation_candidates
        ],
        "operation_candidate_ids": [
            list(candidates) for candidates in operation_candidates
        ],
    }
    if operation_sequence_count == 0:
        vspt_storage = _all_source_storage(record, "r10-no-vspt-tree-accounting-v1")
        acaw_storage = _all_source_storage(
            record, "r10-no-acaw-workspace-accounting-v1"
        )
        return {
            "categorical": categorical,
            "vspt": {
                "available": False,
                "full_transform_certificate": False,
                "overflow": False,
                "query_answer": None,
                "query_certificate": False,
                "query_certificate_correct": False,
                "query_only_certificate": False,
                "root_size": None,
                "root_size_gate_value": OVERFLOW_SIZE,
                "storage": vspt_storage,
                "subtree_evictions": [],
                "uncapped_root_size": 0,
            },
            "acaw": {
                "ambiguity_rank": None,
                "annihilator_proof": False,
                "available": False,
                "full_transform_certificate": False,
                "query_answer": None,
                "query_certificate": False,
                "query_certificate_correct": False,
                "query_only_certificate": False,
                "storage": acaw_storage,
                "subtree_evictions": [],
            },
            "mechanics": {
                "acaw_nonoverflow_certificates_without_exact": 0,
                "ambiguity_rank_excess": 0,
                "exact_transform_containment_checks": 0,
                "exact_transforms_outside_acaw_hulls": 0,
                "max_acaw_ambiguity_rank": None,
            },
        }

    candidate_operations = _operation_candidate_tuples(record, operation_candidates)
    sources = tuple(
        "{}:event:{}".format(record.reference, index) for index in range(record.depth)
    )
    tree = build_tree(candidate_operations, sources=sources, cap=VSPT_CAP)
    workspace, workspaces = _workspace_tree(candidate_operations)
    containment_checks = 0
    containment_violations = 0
    max_rank = 0
    for node in _tree_nodes(tree):
        node_workspace = workspaces[(node.start, node.end)]
        max_rank = max(max_rank, node_workspace.ambiguity_rank)
        if not node.overflow:
            containment_checks += len(node.candidates)
            containment_violations += sum(
                not node_workspace.contains(candidate.transform)
                for candidate in node.candidates
            )

    exact_certified, exact_answer, exact_answers = _exact_query_certificate(
        tree,
        record,
        query_candidates,
    )
    acaw_certified, acaw_answer, annihilator_proof = _acaw_query_certificate(
        workspace,
        record,
        query_candidates,
    )
    exact_storage, exact_segments = _exact_storage(
        tree,
        record,
        score_report_sha256,
        board_sha256,
    )
    acaw_storage, acaw_segments = _acaw_frontier(
        tree,
        workspaces,
        record,
        score_report_sha256,
        board_sha256,
    )
    uncapped = tree.version_space_size
    if tree.overflow:
        uncapped = _uncapped_root_size(record, operation_candidates)
    return {
        "categorical": categorical,
        "vspt": {
            "available": True,
            "complete_query_answers": list(exact_answers),
            "false_full_transform_certificate": (
                tree.source_droppable
                and tree.unique_transform
                != _oracle_segment_transform(record, 0, record.depth)
            ),
            "full_transform_certificate": tree.source_droppable,
            "overflow": tree.overflow,
            "query_answer": exact_answer,
            "query_certificate": exact_certified,
            "query_certificate_correct": exact_certified
            and exact_answer == record.answer,
            "query_only_certificate": exact_certified and not tree.source_droppable,
            "root_size": tree.version_space_size,
            "root_size_gate_value": (
                tree.version_space_size
                if tree.version_space_size is not None
                else OVERFLOW_SIZE
            ),
            "storage": exact_storage,
            "subtree_evictions": exact_segments,
            "uncapped_root_size": uncapped,
        },
        "acaw": {
            "ambiguity_rank": workspace.ambiguity_rank,
            "annihilator_proof": annihilator_proof,
            "available": True,
            "false_full_transform_certificate": (
                workspace.source_droppable
                and not _rational_equals_transform(
                    workspace.anchor,
                    _oracle_segment_transform(record, 0, record.depth),
                )
            ),
            "full_transform_certificate": workspace.source_droppable,
            "query_answer": acaw_answer,
            "query_certificate": acaw_certified,
            "query_certificate_correct": acaw_certified
            and acaw_answer == record.answer,
            "query_only_certificate": acaw_certified and not workspace.source_droppable,
            "storage": acaw_storage,
            "subtree_evictions": acaw_segments,
        },
        "mechanics": {
            "acaw_nonoverflow_certificates_without_exact": int(
                not tree.overflow and acaw_certified and not exact_certified
            ),
            "ambiguity_rank_excess": max(0, max_rank - MAX_AFFINE_AMBIGUITY_RANK),
            "exact_transform_containment_checks": containment_checks,
            "exact_transforms_outside_acaw_hulls": containment_violations,
            "max_acaw_ambiguity_rank": max_rank,
        },
    }


@dataclass(frozen=True)
class CaseEvaluation:
    record: ScoreRecord
    operation_candidates: tuple[tuple[int, ...], ...]
    query_candidates: tuple[int, ...]
    top1: dict
    analysis: dict

    @property
    def event_coverage(self) -> tuple[bool, ...]:
        return tuple(
            target in candidates
            for target, candidates in zip(
                self.record.operation_targets, self.operation_candidates
            )
        )

    @property
    def query_covered(self) -> bool:
        return self.record.query_target in self.query_candidates

    @property
    def complete_program_covered(self) -> bool:
        return all(self.event_coverage) and self.query_covered

    def output_record(self) -> dict:
        return {
            "acaw": self.analysis["acaw"],
            "candidate_coverage": {
                "complete_program": self.complete_program_covered,
                "events": list(self.event_coverage),
                "query": self.query_covered,
            },
            "categorical_candidates": self.analysis["categorical"],
            "depth": self.record.depth,
            "family": self.record.family,
            "index": self.record.index,
            "mechanics": self.analysis["mechanics"],
            "query_target": self.record.query_target,
            "reference": self.record.reference,
            "regime": self.record.regime,
            "source_record_sha256": self.record.source_record_sha256,
            "top1": self.top1,
            "vspt": self.analysis["vspt"],
        }


def evaluate_case(
    record: ScoreRecord,
    operation_candidates: Sequence[Sequence[int]],
    query_candidates: Sequence[int],
    *,
    score_report_sha256: str,
    board_sha256: str,
) -> CaseEvaluation:
    operations = tuple(tuple(candidates) for candidates in operation_candidates)
    queries = tuple(query_candidates)
    return CaseEvaluation(
        record=record,
        operation_candidates=operations,
        query_candidates=queries,
        top1=top1_analysis(record),
        analysis=analyze_candidate_program(
            record,
            operations,
            queries,
            score_report_sha256=score_report_sha256,
            board_sha256=board_sha256,
        ),
    )


def _primary_candidates(
    record: ScoreRecord,
    threshold: float,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    operations = tuple(
        candidate_ids(row, threshold) for row in record.joint_probabilities
    )
    queries = candidate_ids(record.query_probabilities, threshold)
    return operations, queries


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _at_least(value: float | None, floor: float) -> bool:
    return value is not None and value + 1e-12 >= floor


def _nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _method_summary(cases: Sequence[CaseEvaluation], method: str) -> dict:
    items = [case.analysis[method] for case in cases]
    certificates = sum(item["query_certificate"] for item in items)
    correct = sum(item["query_certificate_correct"] for item in items)
    full = sum(item["full_transform_certificate"] for item in items)
    storage = [item["storage"] for item in items]
    total_source_events = sum(
        item["retained_source_events"] + item["evicted_source_events"]
        for item in storage
    )
    retrieval_bound_events = sum(
        item["retrieval_bound_source_events"] for item in storage
    )
    summary = {
        "accounting_schemas": sorted({item["accounting_schema"] for item in storage}),
        "canonical_hot_bytes": sum(item["canonical_hot_bytes"] for item in storage),
        "evicted_source_events": sum(item["evicted_source_events"] for item in storage),
        "external_binding_pointer_bytes": sum(
            item["external_binding_pointer_bytes"] for item in storage
        ),
        "factorized_node_count": sum(item["factorized_node_count"] for item in storage),
        "factorized_provenance_bytes": sum(
            item["factorized_provenance_bytes"] for item in storage
        ),
        "false_full_transform_certificates": sum(
            bool(item.get("false_full_transform_certificate")) for item in items
        ),
        "false_hot_eviction_events": sum(
            item["false_hot_eviction_events"] for item in storage
        ),
        "false_hot_eviction_segments": sum(
            item["false_hot_eviction_segments"] for item in storage
        ),
        "false_query_certificates": certificates - correct,
        "full_transform_certificates": full,
        "hot_plus_retrieval_provenance_bytes": sum(
            item["hot_plus_retrieval_provenance_bytes"] for item in storage
        ),
        "irreversible_source_deletions": sum(
            item["irreversible_source_deletions"] for item in storage
        ),
        "orphan_retrieval_pointers": sum(
            item["orphan_retrieval_pointers"] for item in storage
        ),
        "query_driven_evicted_source_events": sum(
            item["query_driven_evicted_source_events"] for item in storage
        ),
        "query_certificates": certificates,
        "query_only_certificates": sum(
            item["query_only_certificate"] for item in items
        ),
        "retained_source_events": sum(
            item["retained_source_events"] for item in storage
        ),
        "reader_family_evicted_source_events": sum(
            item["reader_family_evicted_source_events"] for item in storage
        ),
        "retrieval_backed_hot_removal": _fraction(
            retrieval_bound_events, total_source_events
        ),
        "retrieval_bound_source_events": retrieval_bound_events,
        "retrieval_pointer_count": sum(
            item["retrieval_pointer_count"] for item in storage
        ),
        "retrieval_provenance_bytes": sum(
            item["retrieval_provenance_bytes"] for item in storage
        ),
        "retrieval_reference_bytes": sum(
            item["retrieval_reference_bytes"] for item in storage
        ),
        "selective_accuracy": _fraction(correct, certificates),
        "selective_coverage": _fraction(certificates, len(cases)),
        "source_bytes": sum(item["source_bytes"] for item in storage),
        "total_canonical_bytes": sum(item["total_canonical_bytes"] for item in storage),
        "transform_segments": sum(item["transform_segments"] for item in storage),
        "unbound_evicted_source_events": sum(
            item["unbound_evicted_source_events"] for item in storage
        ),
    }
    if method == "vspt":
        encoded = [item["root_size_gate_value"] for item in items]
        overflow = sum(item["overflow"] for item in items)
        uncapped = [item["uncapped_root_size"] for item in items]
        summary.update(
            {
                "overflow_cases": overflow,
                "overflow_rate": _fraction(overflow, len(cases)),
                "root_size_encoding": "exact size when available; overflow or empty candidate set is 33",
                "root_size_p50": _nearest_rank(encoded, 0.50),
                "root_size_p90": _nearest_rank(encoded, 0.90),
                "uncapped_feasible_cases": sum(value is not None for value in uncapped),
                "uncapped_root_size_max": max(
                    (value for value in uncapped if value is not None),
                    default=None,
                ),
            }
        )
    else:
        ranks = [
            item["ambiguity_rank"]
            for item in items
            if item["ambiguity_rank"] is not None
        ]
        summary["ambiguity_rank_max"] = max(ranks, default=None)
    return summary


def summarize_cases(cases: Sequence[CaseEvaluation]) -> dict:
    cases = tuple(cases)
    events = sum(case.record.depth for case in cases)
    event_covered = sum(sum(case.event_coverage) for case in cases)
    queries_covered = sum(case.query_covered for case in cases)
    programs_covered = sum(case.complete_program_covered for case in cases)
    operation_candidates = [
        len(candidates) for case in cases for candidates in case.operation_candidates
    ]
    query_candidates = [len(case.query_candidates) for case in cases]
    top1_operation_correct = sum(case.top1["operation_correct"] for case in cases)
    families = collections.Counter(case.record.family for case in cases)
    return {
        "acaw": _method_summary(cases, "acaw"),
        "candidate_coverage": {
            "complete_program": _fraction(programs_covered, len(cases)),
            "complete_programs_covered": programs_covered,
            "event": _fraction(event_covered, events),
            "events_covered": event_covered,
            "query": _fraction(queries_covered, len(cases)),
            "queries_covered": queries_covered,
        },
        "cases": len(cases),
        "categorical_candidate_counts": {
            "event_candidates_max": max(operation_candidates, default=0),
            "event_candidates_mean": _fraction(
                sum(operation_candidates), len(operation_candidates)
            ),
            "query_candidates_max": max(query_candidates, default=0),
            "query_candidates_mean": _fraction(
                sum(query_candidates), len(query_candidates)
            ),
        },
        "events": events,
        "families": dict(sorted(families.items())),
        "top1": {
            "answer_accuracy": _fraction(
                sum(case.top1["answer_correct"] for case in cases), len(cases)
            ),
            "answers_correct": sum(case.top1["answer_correct"] for case in cases),
            "operation_accuracy": _fraction(top1_operation_correct, events),
            "operations_correct": top1_operation_correct,
            "program_exact_accuracy": _fraction(
                sum(case.top1["program_exact"] for case in cases), len(cases)
            ),
            "query_accuracy": _fraction(
                sum(case.top1["query_correct"] for case in cases), len(cases)
            ),
        },
        "vspt": _method_summary(cases, "vspt"),
    }


def _class_coverage(cases: Sequence[CaseEvaluation]) -> dict:
    operation = {}
    for opcode, name in enumerate(OPCODES):
        selected = [
            target in candidates
            for case in cases
            for target, candidates in zip(
                case.record.operation_targets, case.operation_candidates
            )
            if target == opcode
        ]
        operation[name] = {
            "covered": sum(selected),
            "coverage": _fraction(sum(selected), len(selected)),
            "total": len(selected),
        }
    query = {}
    for query_id, name in enumerate(QUERIES):
        selected = [
            case.record.query_target in case.query_candidates
            for case in cases
            if case.record.query_target == query_id
        ]
        query[name] = {
            "covered": sum(selected),
            "coverage": _fraction(sum(selected), len(selected)),
            "total": len(selected),
        }
    return {"operations": operation, "queries": query}


def _stratum_summaries(cases: Sequence[CaseEvaluation]) -> dict:
    by_depth = {}
    for depth in sorted({case.record.depth for case in cases}):
        selected = [case for case in cases if case.record.depth == depth]
        by_depth[str(depth)] = summarize_cases(selected)
    by_query = {}
    for query_id, name in enumerate(QUERIES):
        selected = [case for case in cases if case.record.query_target == query_id]
        if selected:
            by_query[name] = summarize_cases(selected)
    by_family = {}
    for family in sorted({case.record.family for case in cases}):
        selected = [case for case in cases if case.record.family == family]
        by_family[family] = summarize_cases(selected)
    by_depth_query = {}
    for depth in sorted({case.record.depth for case in cases}):
        for query_id, name in enumerate(QUERIES):
            selected = [
                case
                for case in cases
                if case.record.depth == depth and case.record.query_target == query_id
            ]
            if selected:
                by_depth_query["depth={}:query={}".format(depth, name)] = (
                    summarize_cases(selected)
                )
    by_exact_cell = {}
    for regime in sorted({case.record.regime for case in cases}):
        for depth in EXPECTED_DEPTHS[regime]:
            for query_id, query_name in enumerate(QUERIES):
                for family in sorted({case.record.family for case in cases}):
                    selected = [
                        case
                        for case in cases
                        if case.record.regime == regime
                        and case.record.depth == depth
                        and case.record.query_target == query_id
                        and case.record.family == family
                    ]
                    if selected:
                        by_exact_cell[_cell_key(regime, depth, query_name, family)] = (
                            summarize_cases(selected)
                        )
    return {
        "depth": by_depth,
        "depth_query": by_depth_query,
        "exact_regime_depth_query_family": by_exact_cell,
        "family": by_family,
        "query": by_query,
    }


def _class_priors(
    records: Sequence[ScoreRecord],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    operation_counts = [0] * len(OPCODES)
    query_counts = [0] * len(QUERIES)
    for record in records:
        for target in record.operation_targets:
            operation_counts[target] += 1
        query_counts[record.query_target] += 1
    operation_total = sum(operation_counts)
    query_total = sum(query_counts)
    _require(
        operation_total > 0 and query_total > 0, "calibration class priors are empty"
    )
    return (
        tuple(count / operation_total for count in operation_counts),
        tuple(count / query_total for count in query_counts),
    )


def _stratified_shuffle_candidates(
    records: Sequence[ScoreRecord],
    threshold: float,
    seed: int,
) -> tuple[dict[int, tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]], dict]:
    groups: dict[tuple[str, int, str, str], list[ScoreRecord]] = {}
    for record in records:
        groups.setdefault(
            (
                record.regime,
                record.depth,
                QUERIES[record.query_target],
                record.family,
            ),
            [],
        ).append(record)
    mapping = {}
    strata = {}
    supported = True
    for stratum, selected in sorted(groups.items()):
        selected = sorted(selected, key=lambda record: record.index)
        if len(selected) < 2:
            supported = False
            strata[_cell_key(*stratum)] = {
                "cases": len(selected),
                "deranged": False,
            }
            continue
        key_seed = int.from_bytes(
            hashlib.sha256(
                ":".join(
                    (str(seed), stratum[0], str(stratum[1]), stratum[2], stratum[3])
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        ordered = list(selected)
        random.Random(key_seed).shuffle(ordered)
        donors = ordered[1:] + ordered[:1]
        for destination, donor in zip(ordered, donors):
            mapping[destination.index] = _primary_candidates(donor, threshold)
        strata[_cell_key(*stratum)] = {
            "cases": len(selected),
            "deranged": True,
        }
    return mapping, {
        "definition": (
            "Deterministic cyclic score-vector derangement within each exact partition, depth, "
            "query, and family cell; event positions and destination event values are preserved."
        ),
        "strata": strata,
        "supported": supported,
    }


def _evaluate_policy(
    records: Sequence[ScoreRecord],
    candidates_by_index: dict[int, tuple[Sequence[Sequence[int]], Sequence[int]]],
    *,
    score_report_sha256: str,
    board_sha256: str,
) -> tuple[CaseEvaluation, ...]:
    return tuple(
        evaluate_case(
            record,
            candidates_by_index[record.index][0],
            candidates_by_index[record.index][1],
            score_report_sha256=score_report_sha256,
            board_sha256=board_sha256,
        )
        for record in records
    )


def _partitioned_summary(
    cases: Sequence[CaseEvaluation],
    partitions: Sequence[str],
) -> dict:
    return {
        "combined": summarize_cases(cases),
        "partitions": {
            partition: summarize_cases(
                [case for case in cases if case.record.regime == partition]
            )
            for partition in partitions
        },
    }


def _matched_coverage_baselines(
    cases: Sequence[CaseEvaluation],
    partitions: Sequence[str],
) -> dict:
    _require(
        tuple(partitions) == CONFIRMATION_REGIMES,
        "baseline partitions must be the frozen confirmation regimes",
    )
    specifications = {
        "max_probability": ("max_probability", True),
        "minimum_top1_margin": ("minimum_top1_margin", True),
        "maximum_entropy": ("maximum_entropy", False),
    }
    output = {}
    for baseline, (score_name, descending) in specifications.items():
        partition_results = {}
        selected_all = []
        for partition in partitions:
            chosen = []
            cells = {}
            for depth in EXPECTED_DEPTHS[partition]:
                for query_id, query_name in enumerate(QUERIES):
                    for family in EXPECTED_FAMILIES["confirmation"]:
                        selected = [
                            case
                            for case in cases
                            if case.record.regime == partition
                            and case.record.depth == depth
                            and case.record.query_target == query_id
                            and case.record.family == family
                        ]
                        expected_rows = EXPECTED_CELL_ROWS["confirmation"]
                        _require(
                            len(selected) == expected_rows,
                            "baseline exact cell {} has {} rows, expected {}".format(
                                _cell_key(partition, depth, query_name, family),
                                len(selected),
                                expected_rows,
                            ),
                        )
                        accepted = sum(
                            case.analysis["acaw"]["query_certificate"]
                            for case in selected
                        )
                        ordered = sorted(
                            selected,
                            key=lambda case: (
                                (
                                    -case.top1["selection_scores"][score_name]
                                    if descending
                                    else case.top1["selection_scores"][score_name]
                                ),
                                hashlib.sha256(
                                    case.record.reference.encode("utf-8")
                                ).hexdigest(),
                            ),
                        )
                        cell_chosen = ordered[:accepted]
                        chosen.extend(cell_chosen)
                        cells[_cell_key(partition, depth, query_name, family)] = {
                            "rows": len(selected),
                            "accepted": accepted,
                            "accuracy": _fraction(
                                sum(
                                    case.top1["answer_correct"] for case in cell_chosen
                                ),
                                accepted,
                            ),
                            "family": family,
                            "selected_reference_sha256": [
                                hashlib.sha256(
                                    case.record.reference.encode("utf-8")
                                ).hexdigest()
                                for case in cell_chosen
                            ],
                        }
            selected_all.extend(chosen)
            accepted = len(chosen)
            partition_results[partition] = {
                "accepted": accepted,
                "accuracy": _fraction(
                    sum(case.top1["answer_correct"] for case in chosen), accepted
                ),
                "cells": cells,
            }
        total = len(selected_all)
        output[baseline] = {
            "accepted": total,
            "accuracy": _fraction(
                sum(case.top1["answer_correct"] for case in selected_all), total
            ),
            "partitions": partition_results,
            "ranking": (
                "descending top1 program log probability"
                if baseline == "max_probability"
                else (
                    "descending minimum top1 categorical margin"
                    if baseline == "minimum_top1_margin"
                    else "ascending maximum categorical entropy"
                )
            ),
        }
    best = max(
        (item["accuracy"] for item in output.values() if item["accuracy"] is not None),
        default=None,
    )
    acaw_certificates = sum(
        case.analysis["acaw"]["query_certificate"] for case in cases
    )
    acaw = _fraction(
        sum(case.analysis["acaw"]["query_certificate_correct"] for case in cases),
        acaw_certificates,
    )
    partition_comparison = {}
    for partition in partitions:
        selected = [case for case in cases if case.record.regime == partition]
        accepted = sum(case.analysis["acaw"]["query_certificate"] for case in selected)
        acaw_accuracy = _fraction(
            sum(
                case.analysis["acaw"]["query_certificate_correct"] for case in selected
            ),
            accepted,
        )
        partition_baselines = {
            name: item["partitions"][partition]["accuracy"]
            for name, item in output.items()
        }
        best_partition = max(
            (value for value in partition_baselines.values() if value is not None),
            default=None,
        )
        partition_comparison[partition] = {
            "acaw_accuracy": acaw_accuracy,
            "baseline_accuracies": partition_baselines,
            "best_baseline_accuracy": best_partition,
            "acaw_over_best_baseline": (
                None
                if acaw_accuracy is None or best_partition is None
                else acaw_accuracy - best_partition
            ),
        }
    return {
        "baselines": output,
        "best_baseline_accuracy": best,
        "acaw_accuracy": acaw,
        "acaw_over_best_baseline": (
            None if best is None or acaw is None else acaw - best
        ),
        "partitions": partition_comparison,
        "matching_rule": (
            "Each baseline is selected independently inside every frozen "
            "partition x depth x query x family cell to exactly the ACAW certificate count in "
            "that cell. Family pooling is forbidden. Selection scores are tied by "
            "SHA-256(reference); labels are opened only afterward."
        ),
    }


def _control_reports(
    calibration_records: Sequence[ScoreRecord],
    test_records: Sequence[ScoreRecord],
    primary: Sequence[CaseEvaluation],
    calibration: CalibrationResult,
    partitions: Sequence[str],
    *,
    score_report_sha256: str,
    board_sha256: str,
    seed: int,
) -> dict:
    primary_by_index = {
        case.record.index: (case.operation_candidates, case.query_candidates)
        for case in primary
    }
    all_operations = tuple(range(len(OPCODES)))
    all_queries = tuple(range(len(QUERIES)))

    full_opset_candidates = {
        record.index: (
            tuple(all_operations for _ in range(record.depth)),
            all_queries,
        )
        for record in test_records
    }
    operation_prior, query_prior = _class_priors(calibration_records)
    prior_operations = candidate_ids(operation_prior, calibration.threshold)
    prior_queries = candidate_ids(query_prior, calibration.threshold)
    class_prior_candidates = {
        record.index: (
            tuple(prior_operations for _ in range(record.depth)),
            prior_queries,
        )
        for record in test_records
    }
    shuffled_candidates, shuffled_contract = _stratified_shuffle_candidates(
        test_records,
        calibration.threshold,
        seed,
    )
    oracle_query_candidates = {
        record.index: (primary_by_index[record.index][0], (record.query_target,))
        for record in test_records
    }
    oracle_singleton_candidates = {
        record.index: (
            tuple((target,) for target in record.operation_targets),
            (record.query_target,),
        )
        for record in test_records
    }

    def run(candidates):
        cases = _evaluate_policy(
            test_records,
            candidates,
            score_report_sha256=score_report_sha256,
            board_sha256=board_sha256,
        )
        return _partitioned_summary(cases, partitions), cases

    full_summary, _ = run(full_opset_candidates)
    prior_summary, _ = run(class_prior_candidates)
    if shuffled_contract["supported"]:
        shuffled_summary, _ = run(shuffled_candidates)
    else:
        shuffled_summary = None
    oracle_query_summary, _ = run(oracle_query_candidates)
    oracle_singleton_summary, oracle_singleton_cases = run(oracle_singleton_candidates)
    primary_summary = _partitioned_summary(primary, partitions)
    return {
        "class_prior": {
            "candidate_ids": {
                "operations": list(prior_operations),
                "queries": list(prior_queries),
            },
            "definition": (
                "Empirical operation-event and query-program class frequencies from the calibration "
                "file, thresholded by the same frozen q."
            ),
            "priors": {
                "operations": list(operation_prior),
                "queries": list(query_prior),
            },
            "summary": prior_summary,
            "supported": True,
        },
        "full_opset": {
            "definition": "Every operation opcode and every query category is retained.",
            "summary": full_summary,
            "supported": True,
        },
        "oracle_numeral": {
            "definition": (
                "Schema-degenerate control: the extractor supplies one immutable lexical event value "
                "to every policy, and structural candidates force value zero. No learned numeral "
                "prediction channel exists in schema version 1."
            ),
            "schema_degenerate": True,
            "summary": primary_summary,
            "supported": True,
        },
        "oracle_query": {
            "definition": (
                "Primary neural operation candidate sets with the query candidate set replaced by "
                "the singleton oracle query."
            ),
            "summary": oracle_query_summary,
            "supported": True,
        },
        "oracle_singleton": {
            "definition": "Singleton true operation at every event and singleton true query.",
            "sanity": {
                "all_acaw_answers_correct": all(
                    case.analysis["acaw"]["query_certificate_correct"]
                    for case in oracle_singleton_cases
                ),
                "all_acaw_full_transform_certified": all(
                    case.analysis["acaw"]["full_transform_certificate"]
                    for case in oracle_singleton_cases
                ),
                "all_vspt_answers_correct": all(
                    case.analysis["vspt"]["query_certificate_correct"]
                    for case in oracle_singleton_cases
                ),
                "zero_false_hot_evictions": all(
                    case.analysis[method]["storage"]["false_hot_eviction_segments"] == 0
                    for case in oracle_singleton_cases
                    for method in ("vspt", "acaw")
                ),
                "zero_overflow": all(
                    not case.analysis["vspt"]["overflow"]
                    for case in oracle_singleton_cases
                ),
            },
            "summary": oracle_singleton_summary,
            "supported": True,
        },
        "stratified_score_shuffle": {
            **shuffled_contract,
            "summary": shuffled_summary,
        },
    }


def validate_report_separation(
    calibration: BoundScoreReport,
    test: BoundScoreReport,
) -> None:
    _require(
        calibration.sha256 != test.sha256,
        "calibration and test score reports are identical",
    )
    _require(
        calibration.report["data_sha256"] != test.report["data_sha256"],
        "calibration and test boards are identical",
    )
    calibration_references = {record.reference for record in calibration.records}
    test_references = {record.reference for record in test.records}
    _require(
        calibration_references.isdisjoint(test_references),
        "calibration and test references overlap",
    )
    invariant_keys = (
        "base_sha256",
        "pointer_adapter_sha256",
        "adapter_sha256",
        "adapter_state_sha256",
        "tokenizer_sha256",
        "r9c_training_data_sha256",
        "r9c_training_structural_admission_sha256",
        "r9c_training_referential_label_admission_sha256",
        "code_identity",
        "evaluator_sha256",
        "extractor_sha256",
        "gate_manifest_sha256",
        "gate_admission_sha256",
        "categorical_order",
        "replay",
    )
    for key in invariant_keys:
        _require(
            calibration.report.get(key) == test.report.get(key),
            "calibration/test provenance differs on {}".format(key),
        )
    _require(
        calibration.report.get("adapter_metadata")
        == test.report.get("adapter_metadata"),
        "calibration/test adapter metadata differs",
    )
    _require(
        calibration.report.get("pointer_adapter_metadata")
        == test.report.get("pointer_adapter_metadata"),
        "calibration/test pointer metadata differs",
    )


def validate_old_fixed_parity(
    old_report_path,
    old_report_sha256: str,
    scores: BoundScoreReport,
) -> dict:
    """Require hard argmax parity with a hash-bound old fixed report."""
    _require(_is_sha256(old_report_sha256), "old fixed report hash is invalid")
    _require(Path(old_report_path).is_file(), "old fixed report is missing")
    _require(
        sha256_file(old_report_path) == old_report_sha256,
        "old fixed report hash mismatch",
    )
    old = _load_json_object(old_report_path, "old fixed report")
    _require(
        old.get("audit") == "referential_bidirectional_syndrome_microcode_eval_r9c",
        "old fixed report has the wrong audit",
    )
    metadata = old.get("adapter_metadata")
    _require(
        isinstance(metadata, dict) and metadata.get("arm") == "no_syndrome",
        "old report is not no_syndrome",
    )
    for key in (
        "base_sha256",
        "pointer_adapter_sha256",
        "adapter_sha256",
        "data_sha256",
    ):
        _require(
            old.get(key) == scores.report.get(key),
            "old fixed parity differs on {}".format(key),
        )
    raw_records = old.get("records")
    _require(
        isinstance(raw_records, list) and len(raw_records) == len(scores.records),
        "old fixed row count mismatch",
    )
    mismatches = []
    for expected, raw in zip(scores.records, raw_records):
        _require(isinstance(raw, dict), "old fixed row is not an object")
        bindings = {
            "index": expected.index,
            "reference": expected.reference,
            "regime": expected.regime,
            "depth": expected.depth,
            "operation_targets": list(expected.operation_targets),
            "operation_values": list(expected.operation_values),
            "query_target": expected.query_target,
            "expected_answer": expected.answer,
        }
        for key, value in bindings.items():
            _require(
                raw.get(key) == value,
                "old fixed row binding mismatch on {}".format(key),
            )
        fixed = raw.get("fixed")
        _require(isinstance(fixed, dict), "old fixed row lacks fixed predictions")
        old_operations = fixed.get("joint_operations")
        old_query = fixed.get("query_prediction")
        current_operations = [_argmax(row) for row in expected.joint_probabilities]
        current_query = _argmax(expected.query_probabilities)
        if old_operations != current_operations or old_query != current_query:
            mismatches.append(expected.index)
    _require(
        not mismatches, "hard argmax parity failed at rows {}".format(mismatches[:20])
    )
    return {
        "enabled": True,
        "path": os.path.realpath(old_report_path),
        "rows": len(scores.records),
        "sha256": old_report_sha256,
        "zero_mismatches": True,
    }


def _optional_old_parity(path, expected_sha256, scores: BoundScoreReport) -> dict:
    _require(
        (path is None) == (expected_sha256 is None),
        "old fixed report path and hash must be supplied together",
    )
    if path is None:
        return {"enabled": False, "reason": "no hash-bound old fixed report supplied"}
    return validate_old_fixed_parity(path, expected_sha256, scores)


def _all_control_support(controls: dict) -> bool:
    return all(control.get("supported") is True for control in controls.values())


def _exact_cell_empirical_evidence(
    cases: Sequence[CaseEvaluation],
) -> dict[str, dict]:
    evidence = {}
    for partition in CONFIRMATION_REGIMES:
        for depth in EXPECTED_DEPTHS[partition]:
            for query_id, query_name in enumerate(QUERIES):
                for family in EXPECTED_FAMILIES["confirmation"]:
                    selected = [
                        case
                        for case in cases
                        if case.record.regime == partition
                        and case.record.depth == depth
                        and case.record.query_target == query_id
                        and case.record.family == family
                    ]
                    key = _cell_key(partition, depth, query_name, family)
                    _require(
                        len(selected) == EXPECTED_CELL_ROWS["confirmation"],
                        "empirical exact cell {} has the wrong row count".format(key),
                    )
                    accepted = sum(
                        case.analysis["acaw"]["query_certificate"] for case in selected
                    )
                    correct = sum(
                        case.analysis["acaw"]["query_certificate_correct"]
                        for case in selected
                    )
                    evidence[key] = {
                        "accepted": accepted,
                        "correct": correct,
                        "depth": depth,
                        "empirical_selective_accuracy": _fraction(correct, accepted),
                        "empirical_selective_coverage": _fraction(
                            accepted, len(selected)
                        ),
                        "false_certificates": accepted - correct,
                        "family": family,
                        "query": query_name,
                        "regime": partition,
                        "rows": len(selected),
                    }
    _require(
        len(evidence)
        == len(CONFIRMATION_REGIMES)
        * 2
        * len(QUERIES)
        * len(EXPECTED_FAMILIES["confirmation"]),
        "empirical exact-cell evidence is incomplete",
    )
    return evidence


def _confirmation_partition_evidence(
    partitioned: dict,
    baselines: dict,
    exact_cells: dict[str, dict],
) -> dict:
    evidence = {}
    for partition in CONFIRMATION_REGIMES:
        summary = partitioned["partitions"][partition]
        acaw = summary["acaw"]
        accepted = acaw["query_certificates"]
        false_certificates = acaw["false_query_certificates"]
        correct = accepted - false_certificates
        comparison = baselines["partitions"][partition]
        partition_cells = {
            key: value
            for key, value in exact_cells.items()
            if value["regime"] == partition
        }
        _require(
            len(partition_cells) == 40,
            "partition {} must contain 40 empirical exact cells".format(partition),
        )
        evidence[partition] = {
            "rows": summary["cases"],
            "accepted": accepted,
            "correct": correct,
            "false_certificates": false_certificates,
            "empirical_selective_accuracy": acaw["selective_accuracy"],
            "empirical_selective_coverage": acaw["selective_coverage"],
            "exact_cells": partition_cells,
            "candidate_coverage": summary["candidate_coverage"],
            "retrieval_backed_hot_removal": acaw["retrieval_backed_hot_removal"],
            "retrieval_bound_source_events": acaw["retrieval_bound_source_events"],
            "source_events": summary["events"],
            "best_stratified_matched_coverage_baseline_accuracy": comparison[
                "best_baseline_accuracy"
            ],
            "acaw_over_best_stratified_matched_coverage_baseline": comparison[
                "acaw_over_best_baseline"
            ],
        }
    return evidence


def _partition_empirical_gates(evidence: dict) -> dict:
    _require(
        set(evidence) == set(CONFIRMATION_REGIMES),
        "partition evidence must cover both confirmation regimes",
    )
    return {
        "acaw_at_least_10_certificates_each_exact_cell": all(
            cell["accepted"] >= MIN_EXACT_CELL_CERTIFICATES
            for item in evidence.values()
            for cell in item["exact_cells"].values()
        ),
        "acaw_at_least_400_certificates_each_partition": all(
            item["accepted"] >= MIN_PARTITION_CERTIFICATES for item in evidence.values()
        ),
        "acaw_beats_best_stratified_matched_coverage_baseline_by_1pp_each_partition": all(
            _at_least(
                item["acaw_over_best_stratified_matched_coverage_baseline"],
                MIN_BASELINE_IMPROVEMENT,
            )
            for item in evidence.values()
        ),
        "acaw_complete_program_candidate_coverage_at_least_95pct_each_partition": all(
            _at_least(
                item["candidate_coverage"]["complete_program"],
                COMPLETE_PROGRAM_COVERAGE_FLOOR,
            )
            for item in evidence.values()
        ),
        "acaw_event_candidate_coverage_at_least_97pct_each_partition": all(
            _at_least(item["candidate_coverage"]["event"], EVENT_COVERAGE_FLOOR)
            for item in evidence.values()
        ),
        "acaw_query_candidate_coverage_at_least_97pct_each_partition": all(
            _at_least(item["candidate_coverage"]["query"], QUERY_COVERAGE_FLOOR)
            for item in evidence.values()
        ),
        "acaw_retrieval_backed_hot_removal_at_least_75pct_each_partition": all(
            _at_least(
                item["retrieval_backed_hot_removal"],
                MIN_RETRIEVAL_BACKED_HOT_REMOVAL,
            )
            for item in evidence.values()
        ),
        "acaw_empirical_selective_accuracy_at_least_99pct_each_partition": all(
            _at_least(
                item["empirical_selective_accuracy"],
                MIN_PARTITION_SELECTIVE_ACCURACY,
            )
            for item in evidence.values()
        ),
        "acaw_empirical_selective_coverage_at_least_40pct_each_partition": all(
            _at_least(
                item["empirical_selective_coverage"],
                MIN_PARTITION_SELECTIVE_COVERAGE,
            )
            for item in evidence.values()
        ),
        "acaw_zero_false_certificates_each_partition": all(
            item["false_certificates"] == 0 for item in evidence.values()
        ),
    }


def assess_static_confirmation(
    calibration_scores: BoundScoreReport,
    test_scores: BoundScoreReport,
    calibration: CalibrationResult,
    test_partitions: Sequence[str],
    *,
    control_seed: int = EXPECTED_EXTRACTOR_SEED,
) -> dict:
    _require(
        tuple(test_partitions) == CONFIRMATION_REGIMES,
        "confirmation partitions must be exactly language_ood and full_ood",
    )
    validate_record_geometry(calibration_scores.records, "calibration")
    validate_record_geometry(test_scores.records, "confirmation")
    validate_report_separation(calibration_scores, test_scores)
    primary_candidates = {
        record.index: _primary_candidates(record, calibration.threshold)
        for record in test_scores.records
    }
    primary = _evaluate_policy(
        test_scores.records,
        primary_candidates,
        score_report_sha256=test_scores.sha256,
        board_sha256=test_scores.report["data_sha256"],
    )
    partitioned = _partitioned_summary(primary, test_partitions)
    combined = partitioned["combined"]
    classes = _class_coverage(primary)
    strata = _stratum_summaries(primary)
    baselines = _matched_coverage_baselines(primary, test_partitions)
    exact_cell_evidence = _exact_cell_empirical_evidence(primary)
    partition_evidence = _confirmation_partition_evidence(
        partitioned, baselines, exact_cell_evidence
    )
    controls = _control_reports(
        calibration_scores.records,
        test_scores.records,
        primary,
        calibration,
        test_partitions,
        score_report_sha256=test_scores.sha256,
        board_sha256=test_scores.report["data_sha256"],
        seed=control_seed,
    )

    storage_records = [
        case.analysis[method]["storage"]
        for case in primary
        for method in ("vspt", "acaw")
    ]
    observed_mechanics = {
        "acaw_certificates_without_annihilator_proof": sum(
            case.analysis["acaw"]["query_certificate"]
            and not case.analysis["acaw"]["annihilator_proof"]
            for case in primary
        ),
        "acaw_nonoverflow_certificates_without_exact": sum(
            case.analysis["mechanics"]["acaw_nonoverflow_certificates_without_exact"]
            for case in primary
        ),
        "ambiguity_rank_excess": sum(
            case.analysis["mechanics"]["ambiguity_rank_excess"] for case in primary
        ),
        "candidate_empty_certificates": sum(
            bool(
                case.analysis["categorical"]["empty_operation_candidate_sets"]
                or case.analysis["categorical"]["empty_query_candidate_set"]
            )
            and bool(
                case.analysis["vspt"]["query_certificate"]
                or case.analysis["acaw"]["query_certificate"]
            )
            for case in primary
        ),
        "exact_transform_containment_checks": sum(
            case.analysis["mechanics"]["exact_transform_containment_checks"]
            for case in primary
        ),
        "exact_transforms_outside_acaw_hulls": sum(
            case.analysis["mechanics"]["exact_transforms_outside_acaw_hulls"]
            for case in primary
        ),
        "irreversible_source_deletions": sum(
            item["irreversible_source_deletions"] for item in storage_records
        ),
        "orphan_retrieval_pointers": sum(
            item["orphan_retrieval_pointers"] for item in storage_records
        ),
        "query_driven_evicted_source_events": sum(
            item["query_driven_evicted_source_events"] for item in storage_records
        ),
        "reader_family_evicted_source_events": sum(
            item["reader_family_evicted_source_events"] for item in storage_records
        ),
        "retrieval_segment_count_mismatches": sum(
            item["retrieval_pointer_count"] != item["transform_segments"]
            for item in storage_records
        ),
        "score_reports_with_adaptive_replay": sum(
            report.report.get("replay", {}).get("adaptive") is not False
            for report in (calibration_scores, test_scores)
        ),
        "source_event_accounting_mismatches": sum(
            item["retained_source_events"] + item["evicted_source_events"]
            != case.record.depth
            for case in primary
            for item in (
                case.analysis["vspt"]["storage"],
                case.analysis["acaw"]["storage"],
            )
        ),
        "unbound_evicted_source_events": sum(
            item["unbound_evicted_source_events"] for item in storage_records
        ),
    }
    mechanics_contracts = {
        key: value == 0
        for key, value in observed_mechanics.items()
        if key != "exact_transform_containment_checks"
    }
    mechanics = {
        "observed_counts": observed_mechanics,
        "limits": {"max_affine_ambiguity_rank": MAX_AFFINE_AMBIGUITY_RANK},
        "contracts": mechanics_contracts,
        "all_contracts_hold": all(mechanics_contracts.values()),
    }

    class_operation_gate = all(
        item["total"] > 0 and _at_least(item["coverage"], 0.90)
        for item in classes["operations"].values()
    )
    class_query_gate = all(
        item["total"] > 0 and _at_least(item["coverage"], 0.90)
        for item in classes["queries"].values()
    )
    partition_gates = _partition_empirical_gates(partition_evidence)
    depth_coverage_gate = all(
        _at_least(summary["acaw"]["selective_coverage"], 0.25)
        for summary in strata["depth"].values()
    )
    query_coverage_gate = set(strata["query"]) == set(QUERIES) and all(
        _at_least(summary["acaw"]["selective_coverage"], 0.25)
        for summary in strata["query"].values()
    )
    depth_query_coverage_gate = all(
        _at_least(summary["acaw"]["selective_coverage"], 0.25)
        for summary in strata["depth_query"].values()
    )
    oracle_sanity = controls["oracle_singleton"]["sanity"]
    gates = {
        **partition_gates,
        "acaw_combined_empirical_selective_accuracy_at_least_99pct": _at_least(
            combined["acaw"]["selective_accuracy"],
            MIN_PARTITION_SELECTIVE_ACCURACY,
        ),
        "acaw_selective_coverage_at_least_25pct_each_depth": depth_coverage_gate,
        "acaw_selective_coverage_at_least_25pct_each_depth_query_stratum": (
            depth_query_coverage_gate
        ),
        "acaw_selective_coverage_at_least_25pct_each_query": query_coverage_gate,
        "all_named_controls_supported": _all_control_support(controls),
        "mechanics_contracts_hold": mechanics["all_contracts_hold"],
        "operation_candidate_coverage_at_least_90pct_each_opcode": class_operation_gate,
        "oracle_singleton_control_is_exact": all(oracle_sanity.values()),
        "query_candidate_coverage_at_least_90pct_each_query": class_query_gate,
        "root_size_p50_at_most_16": (
            combined["vspt"]["root_size_p50"] is not None
            and combined["vspt"]["root_size_p50"] <= 16
        ),
        "root_size_p90_at_most_32": (
            combined["vspt"]["root_size_p90"] is not None
            and combined["vspt"]["root_size_p90"] <= 32
        ),
        "total_overflow_rate_at_most_10pct": (
            combined["vspt"]["overflow_rate"] is not None
            and combined["vspt"]["overflow_rate"] <= 0.10 + 1e-12
        ),
        "zero_false_full_transform_certificates": (
            combined["vspt"]["false_full_transform_certificates"] == 0
            and combined["acaw"]["false_full_transform_certificates"] == 0
        ),
        "zero_false_hot_evictions": (
            combined["vspt"]["false_hot_eviction_segments"] == 0
            and combined["acaw"]["false_hot_eviction_segments"] == 0
        ),
        "zero_false_query_certificates": (
            combined["vspt"]["false_query_certificates"] == 0
            and combined["acaw"]["false_query_certificates"] == 0
        ),
    }
    passed = all(gates.values())
    return {
        "audit": AUDIT,
        "advance_r10_static_path": passed,
        "calibration": {
            "calibration_digest": calibration.calibration_digest,
            "events_reported_not_used_for_k": calibration.events,
            "minimum_probability": calibration.minimum_probability,
            "order_statistic_k": calibration.order_statistic,
            "programs_n": calibration.programs,
            "quantile": calibration.quantile,
            "regimes": list(CALIBRATION_REGIMES),
            "score": "max(-log p_true) across every true operation and the true query",
            "interpretation": (
                "Deterministic pooled calibration rule on the frozen fit_iid and depth_ood "
                "rows; it is not a population coverage theorem."
            ),
            "threshold": (
                None if math.isinf(calibration.threshold) else calibration.threshold
            ),
            "threshold_is_infinite": math.isinf(calibration.threshold),
            "unit": "program",
        },
        "candidate_construction": {
            "empty_set_policy": "abstain; never inject argmax",
            "joint_probability_contract": "softmax(average(forward_logits, backward_logits))",
            "operation_source": "joint_probabilities",
            "query_source": "query_probabilities; query_target appears only in coverage and named oracle controls",
            "structural_value_contract": "merge and swap candidates force value=0",
            "thresholds": 1,
        },
        "claim_boundary": (
            "A pass supports only static, candidate-set-conditional ACAW certificates and reversible "
            "HOT-context eviction on this frozen confirmation board. language_ood and full_ood "
            "results are finite-board empirical evidence. It does not establish population "
            "coverage, "
            "learned replay, irreversible deletion, broad reasoning, or context scaling."
        ),
        "class_candidate_coverage": classes,
        "controls": controls,
        "decision": (
            "pass_r10_static_confirmation"
            if passed
            else "reject_r10_static_confirmation"
        ),
        "gates": gates,
        "hot_storage_contract": {
            "canonical_encoding": (
                "UTF-8 JSON with sorted keys, ASCII escaping, no insignificant whitespace, and "
                "rational entries serialized as [numerator,denominator]"
            ),
            "eviction_semantics": (
                "Candidate-set-conditional removal from HOT context only; every transform segment "
                "retains a score-report/board/row/range retrieval pointer."
            ),
            "hot_bytes": (
                "Sum of canonical item bytes for the compact frontier: retained event-source "
                "payloads plus rank-zero transform summaries, with the root anchor/basis also "
                "included for ACAW."
            ),
            "provenance_bytes": (
                "Sum of canonical factorized-node records. Exact VSPT accounting uses "
                "r10-version-space-accounting-v1 commitments; ACAW records every composed hull."
            ),
            "retrieval_provenance_bytes": (
                "Factorized provenance plus canonical range references and external "
                "score-report/board binding pointers."
            ),
            "source_bytes": (
                "Sum of canonical bytes for all event text/structured-operation source payloads."
            ),
        },
        "learned_replay": {
            "attempted": False,
            "supported": False,
            "reason": "No second frozen evidence source exists for a learned or active refinement claim.",
        },
        "matched_coverage_selective_baselines": baselines,
        "mechanics": mechanics,
        "partition_empirical_evidence": partition_evidence,
        "records": [case.output_record() for case in primary],
        "strata": strata,
        "summary": partitioned,
        "test_partitions": list(test_partitions),
    }


def run_evaluation(
    *,
    calibration_scores_path,
    calibration_scores_sha256: str,
    calibration_data_sha256: str,
    calibration_structural_admission_sha256: str,
    calibration_label_admission_sha256: str,
    test_scores_path,
    test_scores_sha256: str,
    test_data_sha256: str,
    test_structural_admission_sha256: str,
    test_label_admission_sha256: str,
    gate_manifest_path,
    gate_manifest_sha256: str,
    gate_admission_path,
    gate_admission_sha256: str,
    extractor_path,
    extractor_sha256: str,
    evaluator_sha256: str,
    code_revision: str,
    calibration_old_fixed_report=None,
    calibration_old_fixed_report_sha256=None,
    test_old_fixed_report=None,
    test_old_fixed_report_sha256=None,
    expected_adapter_sha256: str = EXPECTED_ADAPTER_SHA256,
    expected_seed: int = EXPECTED_EXTRACTOR_SEED,
) -> dict:
    _require(
        False,
        "standalone caller-supplied score reports are forbidden; run the committed extractor chain",
    )
    _require(_is_sha256(evaluator_sha256), "evaluator hash is invalid")
    _require(_is_sha256(extractor_sha256), "extractor hash is invalid")
    _require(_is_git_revision(code_revision), "code revision is invalid")
    actual_evaluator_sha256 = sha256_file(__file__)
    _require(
        actual_evaluator_sha256 == evaluator_sha256,
        "executing evaluator differs from the frozen evaluator hash",
    )
    extractor_real = os.path.realpath(extractor_path)
    _require(Path(extractor_real).is_file(), "missing extractor source")
    _require(
        sha256_file(extractor_real) == extractor_sha256,
        "extractor source differs from the frozen extractor hash",
    )
    repo_root = Path(__file__).resolve().parents[1]
    gate_bundle = validate_gate_bundle(
        gate_manifest_path,
        expected_manifest_sha256=gate_manifest_sha256,
        admission_path=gate_admission_path,
        expected_admission_sha256=gate_admission_sha256,
        expected_board_bindings={
            "calibration": {
                "data_sha256": calibration_data_sha256,
                "structural_admission_sha256": (
                    calibration_structural_admission_sha256
                ),
                "label_admission_sha256": calibration_label_admission_sha256,
            },
            "confirmation": {
                "data_sha256": test_data_sha256,
                "structural_admission_sha256": test_structural_admission_sha256,
                "label_admission_sha256": test_label_admission_sha256,
            },
        },
        expected_evaluator_sha256=evaluator_sha256,
        expected_extractor_sha256=extractor_sha256,
        evaluator_path=__file__,
        extractor_path=extractor_real,
        expected_code_revision=code_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_seed=expected_seed,
        repo_root=repo_root,
    )

    # This preflight completes before either score-report path is opened.
    hash_cache: dict[str, str] = {}
    calibration_scores = validate_score_report(
        calibration_scores_path,
        board_name="calibration",
        expected_report_sha256=calibration_scores_sha256,
        expected_data_sha256=calibration_data_sha256,
        expected_structural_admission_sha256=calibration_structural_admission_sha256,
        expected_label_admission_sha256=calibration_label_admission_sha256,
        extractor_path=extractor_real,
        expected_extractor_sha256=extractor_sha256,
        expected_evaluator_sha256=evaluator_sha256,
        gate_bundle=gate_bundle,
        expected_code_revision=code_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_seed=expected_seed,
        hash_cache=hash_cache,
    )
    calibration = calibrate_program_threshold(calibration_scores.records)

    # Rehash shared artifacts for the test report so a between-report change fails closed.
    hash_cache = {}
    test_scores = validate_score_report(
        test_scores_path,
        board_name="confirmation",
        expected_report_sha256=test_scores_sha256,
        expected_data_sha256=test_data_sha256,
        expected_structural_admission_sha256=test_structural_admission_sha256,
        expected_label_admission_sha256=test_label_admission_sha256,
        extractor_path=extractor_real,
        expected_extractor_sha256=extractor_sha256,
        expected_evaluator_sha256=evaluator_sha256,
        gate_bundle=gate_bundle,
        expected_code_revision=code_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_seed=expected_seed,
        hash_cache=hash_cache,
    )
    result = assess_static_confirmation(
        calibration_scores,
        test_scores,
        calibration,
        CONFIRMATION_REGIMES,
        control_seed=expected_seed,
    )
    result["inputs"] = {
        "calibration_scores": {
            "data_sha256": calibration_scores.report["data_sha256"],
            "path": calibration_scores.path,
            "sha256": calibration_scores.sha256,
            "structural_admission_sha256": calibration_scores.report[
                "structural_admission_sha256"
            ],
            "referential_label_admission_sha256": calibration_scores.report[
                "referential_label_admission_sha256"
            ],
        },
        "test_scores": {
            "data_sha256": test_scores.report["data_sha256"],
            "path": test_scores.path,
            "sha256": test_scores.sha256,
            "structural_admission_sha256": test_scores.report[
                "structural_admission_sha256"
            ],
            "referential_label_admission_sha256": test_scores.report[
                "referential_label_admission_sha256"
            ],
        },
        "shared": {
            "adapter_sha256": test_scores.report["adapter_sha256"],
            "base_sha256": test_scores.report["base_sha256"],
            "code_identity": gate_bundle.code_identity,
            "evaluator_sha256": evaluator_sha256,
            "extractor_sha256": extractor_sha256,
            "gate_admission": gate_bundle.admission_path,
            "gate_admission_sha256": gate_bundle.admission_sha256,
            "gate_manifest": gate_bundle.manifest_path,
            "gate_manifest_sha256": gate_bundle.manifest_sha256,
            "pointer_adapter_sha256": test_scores.report["pointer_adapter_sha256"],
            "tokenizer_sha256": test_scores.report["tokenizer_sha256"],
        },
    }
    result["board_contract"] = {
        "schema": BOARD_SCHEMA,
        "calibration": validate_record_geometry(
            calibration_scores.records, "calibration"
        ),
        "confirmation": validate_record_geometry(test_scores.records, "confirmation"),
    }
    result["hard_argmax_parity"] = {
        "calibration": _optional_old_parity(
            calibration_old_fixed_report,
            calibration_old_fixed_report_sha256,
            calibration_scores,
        ),
        "test": _optional_old_parity(
            test_old_fixed_report,
            test_old_fixed_report_sha256,
            test_scores,
        ),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-scores", required=True)
    parser.add_argument("--calibration-scores-sha256", required=True)
    parser.add_argument("--calibration-data-sha256", required=True)
    parser.add_argument("--calibration-structural-admission-sha256", required=True)
    parser.add_argument("--calibration-label-admission-sha256", required=True)
    parser.add_argument("--test-scores", required=True)
    parser.add_argument("--test-scores-sha256", required=True)
    parser.add_argument("--test-data-sha256", required=True)
    parser.add_argument("--test-structural-admission-sha256", required=True)
    parser.add_argument("--test-label-admission-sha256", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--gate-admission", required=True)
    parser.add_argument("--gate-admission-sha256", required=True)
    parser.add_argument(
        "--extractor",
        default=str(Path(__file__).with_name("extract_referential_version_scores.py")),
    )
    parser.add_argument("--extractor-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--calibration-old-fixed-report")
    parser.add_argument("--calibration-old-fixed-report-sha256")
    parser.add_argument("--test-old-fixed-report")
    parser.add_argument("--test-old-fixed-report-sha256")
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if os.path.lexists(args.out):
        print("refusing existing output: {}".format(args.out), file=sys.stderr)
        raise SystemExit(2)
    try:
        result = run_evaluation(
            calibration_scores_path=args.calibration_scores,
            calibration_scores_sha256=args.calibration_scores_sha256,
            calibration_data_sha256=args.calibration_data_sha256,
            calibration_structural_admission_sha256=(
                args.calibration_structural_admission_sha256
            ),
            calibration_label_admission_sha256=args.calibration_label_admission_sha256,
            test_scores_path=args.test_scores,
            test_scores_sha256=args.test_scores_sha256,
            test_data_sha256=args.test_data_sha256,
            test_structural_admission_sha256=args.test_structural_admission_sha256,
            test_label_admission_sha256=args.test_label_admission_sha256,
            gate_manifest_path=args.gate_manifest,
            gate_manifest_sha256=args.gate_manifest_sha256,
            gate_admission_path=args.gate_admission,
            gate_admission_sha256=args.gate_admission_sha256,
            extractor_path=args.extractor,
            extractor_sha256=args.extractor_sha256,
            evaluator_sha256=args.evaluator_sha256,
            code_revision=args.code_revision,
            calibration_old_fixed_report=args.calibration_old_fixed_report,
            calibration_old_fixed_report_sha256=(
                args.calibration_old_fixed_report_sha256
            ),
            test_old_fixed_report=args.test_old_fixed_report,
            test_old_fixed_report_sha256=args.test_old_fixed_report_sha256,
        )
        atomic_write_json_no_overwrite(result, args.out)
    except (EvaluationContractError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                "advance_r10_static_path": result["advance_r10_static_path"],
                "decision": result["decision"],
                "gates": result["gates"],
                "out": os.path.realpath(args.out),
            },
            sort_keys=True,
        )
    )
    if not result["advance_r10_static_path"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
