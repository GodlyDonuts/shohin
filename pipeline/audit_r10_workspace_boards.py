#!/usr/bin/env python3
"""Independently admit score-unseen R10 workspace boards before scoring.

The auditor does not import the board generator and accepts no model, adapter,
probability, or score path.  It independently validates surface semantics,
lexical extraction, event order, compiler labels, exact execution, tokenizer
length, balance, novelty, and every hash in the build manifest.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import collections
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tokenizers
import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from categorical_microcode import (  # noqa: E402
    OPCODES,
    QUERIES,
    compile_example,
    execute_program,
    transition_basis_targets,
)
from referential_slot_microcode import compile_referential_example  # noqa: E402


SCHEMA = "r10_workspace_board_v2"
NGRAM_WIDTH = 13
EXECUTOR_WIDTH = 8
EXECUTOR_LIMIT = 10 ** EXECUTOR_WIDTH
WORD = re.compile(r"\w+")
STANDALONE_INTEGER = re.compile(r"(?<![\w-])\d+(?![\w-])")
GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
OPCODE_TO_ID = {name: index for index, name in enumerate(OPCODES)}
QUERY_TO_ID = {name: index for index, name in enumerate(QUERIES)}
ADMISSION_AUDIT = "r10_workspace_boards_independent_admission_v2"
FROZEN_GATE_MANIFEST = "r10_workspace_frozen_score_gate_v2"
EVALUATOR_IDENTIFIER = "referential_version_space_workspace_confirmation_r10"
EXTRACTOR_IDENTIFIER = "referential_version_scores_r10"
EXPECTED_EXTRACTOR_SEED = 20260714
EXPECTED_ADAPTER_SHA256 = "bf07d65075a42142c34bfc510cbef95290a9b8a0f7ed96ac1d4abc5f175a6480"
CANONICAL_GENERATOR_SEEDS = {
    "calibration": 2026071401,
    "confirmation": 2026071402,
}
CANONICAL_R5_NOVELTY_BOARD_SHA256 = (
    "d85f16ff374b0c650cf3603826cc5f3b377842818db62bada3b84e71308b9473"
)
BUILD_MANIFEST_CLAIM_BOUNDARY = (
    "This v2 build proves deterministic score-blind generation on the frozen exact-cell "
    "schedule, compiler/executor preflight, and lexical/program novelty scans only. It "
    "contains no model score and does not authorize an R10 score run until the independent "
    "admission report and frozen gate manifest bind these exact hashes."
)
BUILD_MANIFEST_FIELDS = {
    "build", "schema", "cpu_only", "score_outputs_read", "score_artifacts",
    "ready_for_r10_score_run", "ngram_width", "executor_width",
    "generation_contract", "schedule_contract", "tokenizer", "inputs", "outputs",
    "cross_board_scan", "claim_boundary",
}
BOARD_ADMISSION_CHECK_FIELDS = {
    "minimum_rows", "exact_row_count", "strata_divisible",
    "all_rows_structurally_valid", "regimes_exactly_balanced",
    "numeric_profiles_exactly_balanced", "depths_exactly_balanced",
    "regime_depths_exactly_balanced", "queries_exactly_balanced",
    "depth_query_cells_exactly_balanced", "regime_depth_query_family_cells_exact",
    "cell_indices_exact", "domains_exactly_balanced", "all_operation_labels_covered",
    "operation_labels_globally_balanced",
    "operation_labels_balanced_within_each_regime_depth", "all_operation_kinds_covered",
    "all_intro_templates_covered", "all_operation_templates_covered",
    "all_query_templates_covered", "zero_normalized_prompt_duplicates",
    "zero_program_duplicates", "zero_reference_duplicates", "one_generation_seed",
    "canonical_generation_seed", "tokenizer_limit", "zero_oracle_errors",
    "zero_lexical_errors", "zero_event_order_errors", "zero_semantic_errors",
}
COMPATIBILITY_ADMISSION_CHECK_FIELDS = {
    "structural_audit_kind", "structural_passed", "structural_training_bound",
    "structural_board_bound", "structural_tokenizer_bound", "label_audit_kind",
    "label_passed", "label_training_passed", "label_training_bound",
    "label_board_passed", "label_board_bound", "label_tokenizer_bound",
    "admissions_share_training", "admissions_share_board",
}
DEFAULT_BOARD_ROWS = {"calibration": 800, "confirmation": 1840}
DEFAULT_CELL_ROWS = {"calibration": 10, "confirmation": 23}
DEFAULT_CONFIRMATION_PARTITION_ROWS = 920
CONFIRMATION_CELLS_PER_PARTITION = 40
CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL = 0.40
CONFIRMATION_MIN_EMPIRICAL_ACCURACY = 0.99
CONFIRMATION_MIN_ACCEPTED_PER_CELL = math.ceil(
    CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL * DEFAULT_CELL_ROWS["confirmation"]
)
CONFIRMATION_MIN_ACCEPTED_PER_PARTITION = (
    CONFIRMATION_CELLS_PER_PARTITION * CONFIRMATION_MIN_ACCEPTED_PER_CELL
)
CONFIRMATION_MAX_FALSE_CERTIFICATES = 0
EVALUATOR_REPO_PATH = "train/evaluate_version_space_workspace.py"
EXTRACTOR_REPO_PATH = "train/extract_referential_version_scores.py"
CODE_IDENTITY_PIPELINE_FILES = (
    "pipeline/audit_categorical_microcode_v1.py",
    "pipeline/audit_r10_workspace_boards.py",
    "pipeline/audit_referential_slot_labels.py",
    "pipeline/audit_role_equivariant_microcode_v3.py",
    "pipeline/generate_r10_workspace_boards.py",
)
CODE_IDENTITY_R10_JOB_FILES = (
    "pipeline/jobs/build_r10_workspace_boards_stokes.sbatch",
    "train/jobs/evaluate_version_space_workspace.sbatch",
    "train/jobs/extract_referential_version_scores.sbatch",
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
    "simultaneous confidence",
    "simultaneous-confidence",
)


@dataclass(frozen=True)
class RegimeAuditSpec:
    depths: tuple[int, int]
    numeric_profile: str
    initial_range: tuple[int, int]
    value_range: tuple[int, int]


@dataclass(frozen=True)
class AuditSpec:
    name: str
    reference_prefix: str
    regimes: dict[str, RegimeAuditSpec]
    domains: dict[str, tuple[tuple[str, str], str]]
    event_prefix: str
    query_prefix: str
    answer_marker: str
    intro_templates: tuple[str, ...]
    operation_templates: dict[str, tuple[str, ...]]
    query_templates: dict[str, tuple[str, ...]]


CALIBRATION_OPERATION_TEMPLATES = {
    "add": (
        "Post an inbound lot of {value} {unit} to {target}.",
        "Credit {target} using a newly recorded {value} {unit}.",
        "Augment the balance named {target} by exactly {value} {unit}.",
    ),
    "sub": (
        "Post an outbound lot of {value} {unit} against {target}.",
        "Debit {target} by a recorded {value} {unit}.",
        "Reduce the balance named {target} by exactly {value} {unit}.",
    ),
    "move": (
        "Reclassify {value} {unit} from {source} under {target}.",
        "Post a paired adjustment of {value} {unit} out of {source} and into {target}.",
        "Debit {source} and credit {target} for the same {value} {unit}.",
    ),
    "merge": (
        "Accumulate the whole balance named {source} into {target} while preserving {source}.",
        "Use the current {source} balance as an additional credit to {target}.",
        "Increase {target} by the entire amount presently recorded under {source}.",
    ),
    "swap": (
        "Transpose the balances attached to {left} and {right}.",
        "Give {left} the prior {right} balance and give {right} the prior {left} balance.",
        "Exchange only the ledger balances named {left} and {right}.",
    ),
}
CONFIRMATION_OPERATION_TEMPLATES = {
    "add": (
        "Apply a positive adjustment of {value} {unit} to {target}.",
        "Enter {value} incoming {unit} on the account for {target}.",
        "Raise {target}'s recorded amount through an addition of {value} {unit}.",
    ),
    "sub": (
        "Apply a negative adjustment of {value} {unit} to {target}.",
        "Enter {value} outgoing {unit} on the account for {target}.",
        "Lower {target}'s recorded amount through a deduction of {value} {unit}.",
    ),
    "move": (
        "Route {value} {unit} away from {source} and onward to {target}.",
        "Record one relocation: {source} loses {value} {unit} as {target} receives them.",
        "Shift a quantity of {value} {unit}; subtract it from {source} and add it to {target}.",
    ),
    "merge": (
        "Append the present amount in {source} to {target}, leaving {source} unchanged.",
        "Treat all of {source}'s current balance as an extra amount for {target}.",
        "Combine {source} into {target} additively without resetting either named account.",
    ),
    "swap": (
        "Let the two accounts {left} and {right} inherit one another's previous amounts.",
        "Replace {left}'s amount with old {right}, and replace {right}'s amount with old {left}.",
        "Perform a two-way balance transposition between {left} and {right}.",
    ),
}
CALIBRATION_QUERY_TEMPLATES = {
    "read": (
        "Return the closing balance associated with {key}.",
        "What closing quantity is posted under {key}?",
        "State the terminal balance for {key}.",
    ),
    "sum": (
        "Return the aggregate closing balance of {left} together with {right}.",
        "What total results when the closing {left} and {right} balances are combined?",
        "State the joint terminal quantity across {left} and {right}.",
    ),
    "difference": (
        "Return the nonnegative closing margin of {high} over {low}.",
        "By what quantity does closing {high} stand above closing {low}?",
        "State the closing {high} balance minus the closing {low} balance.",
    ),
}
CONFIRMATION_QUERY_TEMPLATES = {
    "read": (
        "Provide the final account amount for {key}.",
        "Which ending quantity belongs to {key}?",
        "Report the amount left on {key} after every step.",
    ),
    "sum": (
        "Provide the final combined amount across {left} plus {right}.",
        "Which ending total is obtained by adding {left} and {right}?",
        "Report the sum of the two completed accounts {left} and {right}.",
    ),
    "difference": (
        "Provide the final excess of {high} relative to {low}.",
        "Which ending gap remains when {low} is taken from {high}?",
        "Report the completed {high} amount less the completed {low} amount.",
    ),
}


SPECS = {
    "calibration": AuditSpec(
        name="calibration",
        reference_prefix="R10-CAL",
        regimes={
            "fit_iid": RegimeAuditSpec((4, 8), "in_range", (3, 29), (1, 9)),
            "depth_ood": RegimeAuditSpec((16, 32), "shifted", (211, 499), (11, 29)),
        },
        domains={
            "map room": (("contour slips", "bearing folios"), "survey entries"),
            "print works": (("proof bundles", "plate packets"), "production records"),
            "signal bureau": (("relay frames", "beacon logs"), "telemetry records"),
            "glass studio": (("anneal trays", "mould tickets"), "work orders"),
        },
        event_prefix="Event",
        query_prefix="Request",
        answer_marker="Result:",
        intro_templates=(
            "Calibration ledger at {family} opens {left} with {left_value} {unit} and opens "
            "{right} with {right_value} {unit}; every written label is nonnumeric.",
            "For the {family} calibration, the opening balance is {left_value} {unit} under "
            "{left}, while {right} begins at {right_value} {unit}; label words carry no quantity.",
            "A sealed {family} record starts with {left_value} {unit} assigned to {left} and "
            "{right_value} {unit} assigned to {right}; names are textual only.",
        ),
        operation_templates=CALIBRATION_OPERATION_TEMPLATES,
        query_templates=CALIBRATION_QUERY_TEMPLATES,
    ),
    "confirmation": AuditSpec(
        name="confirmation",
        reference_prefix="R10-CON",
        regimes={
            "language_ood": RegimeAuditSpec((4, 8), "in_range", (3, 29), (1, 9)),
            "full_ood": RegimeAuditSpec((16, 32), "shifted", (701, 1099), (31, 53)),
        },
        domains={
            "binding hall": (("quire stacks", "cover lots"), "binding materials"),
            "rail control": (("routing chits", "coupler tallies"), "dispatch records"),
            "ceramic works": (("bisque racks", "glaze batches"), "studio pieces"),
            "forecast office": (("radar frames", "pressure charts"), "weather observations"),
        },
        event_prefix="Step",
        query_prefix="Inquiry",
        answer_marker="Answer:",
        intro_templates=(
            "Independent ledger for {family} assigns {left_value} {unit} initially to {left}; "
            "separately, {right} is assigned {right_value} {unit}. All names are text.",
            "At {family}, begin the verification account with {left} holding {left_value} {unit} "
            "and {right} holding {right_value} {unit}; no label denotes a number.",
            "The untouched {family} account records an initial {left_value} {unit} for {left} "
            "versus {right_value} {unit} for {right}; words in names have no numeric role.",
        ),
        operation_templates=CONFIRMATION_OPERATION_TEMPLATES,
        query_templates=CONFIRMATION_QUERY_TEMPLATES,
    ),
}


def all_depths(spec: AuditSpec) -> tuple[int, ...]:
    return tuple(depth for regime in spec.regimes.values() for depth in regime.depths)


def cell_id(regime: str, depth: int, query: str, family: str) -> str:
    return "{}|depth={}|query={}|family={}".format(regime, depth, query, family)


def expected_cell_keys(spec: AuditSpec) -> tuple[tuple[str, int, str, str], ...]:
    return tuple(
        (regime_name, depth, query, family)
        for regime_name, regime in spec.regimes.items()
        for depth in regime.depths
        for query in QUERIES
        for family in spec.domains
    )


class RowAuditError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def require(condition: bool, category: str, message: str) -> None:
    if not condition:
        raise RowAuditError(category, message)


def exact_int(value, name: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), "structured_semantics", name)
    return int(value)


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text: str, width: int = NGRAM_WIDTH) -> set[str]:
    words = normalized(text).split()
    return {
        " ".join(words[index:index + width])
        for index in range(max(0, len(words) - width + 1))
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode())


def canonical_generation_contract() -> dict:
    return {
        "generator_seeds": dict(CANONICAL_GENERATOR_SEEDS),
        "r5_novelty_board_sha256": CANONICAL_R5_NOVELTY_BOARD_SHA256,
        "seed_variation_forbidden": True,
        "r5_variation_forbidden": True,
    }


def validate_git_revision(value: str) -> str:
    if not isinstance(value, str) or GIT_REVISION.fullmatch(value) is None:
        raise ValueError("code revision must be a full lowercase 40-hex git revision")
    return value


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "tokenizers": str(tokenizers.__version__),
    }


def code_identity_relative_paths(repo_root=ROOT) -> tuple[str, ...]:
    repo_root = Path(repo_root).resolve()
    train_root = repo_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError("missing code identity source directory {}".format(train_root))
    paths = {
        path
        for path in train_root.rglob("*.py")
        if path.is_file()
    }
    paths.update(
        repo_root / relative
        for relative in (*CODE_IDENTITY_PIPELINE_FILES, *CODE_IDENTITY_R10_JOB_FILES)
    )
    missing = sorted(
        str(path.relative_to(repo_root))
        for path in paths
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError("missing code identity files: {}".format(", ".join(missing)))
    symlinks = sorted(
        str(path.relative_to(repo_root))
        for path in paths
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError("code identity files must not be symlinks: {}".format(", ".join(symlinks)))
    return tuple(sorted(path.relative_to(repo_root).as_posix() for path in paths))


def build_code_identity(code_revision: str, repo_root=ROOT) -> dict:
    code_revision = validate_git_revision(code_revision)
    repo_root = Path(repo_root).resolve()
    files = {
        relative: sha256_file(repo_root / relative)
        for relative in code_identity_relative_paths(repo_root)
    }
    runtime = runtime_versions()
    return {
        "git_revision": code_revision,
        "files": files,
        "aggregate_sha256": code_identity_aggregate(code_revision, files, runtime),
        "runtime": runtime,
    }


def code_identity_aggregate(git_revision: str, files: dict, runtime: dict) -> str:
    validate_git_revision(git_revision)
    if runtime != runtime_versions():
        raise ValueError("code identity aggregate runtime differs")
    return canonical_hash({
        "git_revision": git_revision,
        "files": files,
        "runtime": runtime,
    })


def validate_code_identity(
    identity,
    *,
    repo_root=ROOT,
    expected_revision: str | None = None,
) -> None:
    if not isinstance(identity, dict) or set(identity) != {
        "git_revision", "files", "aggregate_sha256", "runtime",
    }:
        raise ValueError("code identity fields differ from the frozen contract")
    revision = validate_git_revision(identity["git_revision"])
    if expected_revision is not None and revision != validate_git_revision(expected_revision):
        raise ValueError("code identity git revision differs")
    files = identity["files"]
    expected_paths = code_identity_relative_paths(repo_root)
    if not isinstance(files, dict) or tuple(files) != expected_paths:
        raise ValueError("code identity source closure is missing, extra, or unsorted")
    repo_root = Path(repo_root).resolve()
    for relative, expected_sha256 in files.items():
        if not isinstance(expected_sha256, str) or SHA256.fullmatch(expected_sha256) is None:
            raise ValueError("invalid code identity hash for {}".format(relative))
        if sha256_file(repo_root / relative) != expected_sha256:
            raise ValueError("code identity hash mismatch for {}".format(relative))
    aggregate = identity["aggregate_sha256"]
    if not isinstance(aggregate, str) or SHA256.fullmatch(aggregate) is None:
        raise ValueError("invalid code identity aggregate hash")
    if aggregate != code_identity_aggregate(revision, files, identity["runtime"]):
        raise ValueError("code identity aggregate hash mismatch")
    if identity["runtime"] != runtime_versions():
        raise ValueError("code identity runtime versions differ")


def _git_output(repo_root, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError("git checkout verification failed: {}".format(detail))
    return result.stdout


def capture_clean_committed_code_identity(code_revision: str, repo_root=ROOT) -> dict:
    """Capture code identity only from an exact, clean, fully committed checkout."""
    code_revision = validate_git_revision(code_revision)
    repo_root = Path(repo_root).resolve()
    top_level = Path(_git_output(repo_root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top_level != repo_root:
        raise ValueError("repo_root must be the exact Git checkout root")
    head = _git_output(repo_root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if head != code_revision:
        raise ValueError("clean checkout HEAD differs from the frozen code revision")
    status = _git_output(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all",
    )
    if status:
        sample = ", ".join(line for line in status.splitlines()[:8])
        raise ValueError("code checkout is not clean and committed: {}".format(sample))
    tracked = set(
        _git_output(repo_root, "ls-files", "-z").rstrip("\0").split("\0")
    )
    expected_paths = code_identity_relative_paths(repo_root)
    missing = [relative for relative in expected_paths if relative not in tracked]
    if missing:
        raise ValueError(
            "code identity source is not committed: {}".format(", ".join(missing))
        )
    identity = build_code_identity(code_revision, repo_root)
    validate_code_identity(
        identity, repo_root=repo_root, expected_revision=code_revision,
    )
    return identity


def require_unchanged_code_identity(
    before: dict,
    *,
    code_revision: str,
    repo_root=ROOT,
) -> dict:
    try:
        after = capture_clean_committed_code_identity(code_revision, repo_root)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError("code checkout changed during score-blind admission") from error
    if after != before:
        raise RuntimeError("code identity changed during score-blind admission")
    return after


def run_admission_with_code_custody(
    *,
    code_revision: str,
    admission_work,
    repo_root=ROOT,
):
    before = capture_clean_committed_code_identity(code_revision, repo_root)
    report = admission_work()
    require_unchanged_code_identity(
        before, code_revision=code_revision, repo_root=repo_root,
    )
    return report, before


def repo_implementation_path(path, expected_relative: str, repo_root=ROOT) -> str:
    repo_root = Path(repo_root).resolve()
    supplied = Path(path)
    if supplied.is_symlink() or supplied.resolve() != (repo_root / expected_relative).resolve():
        raise ValueError("implementation must be the repo source {}".format(expected_relative))
    if not supplied.is_file():
        raise FileNotFoundError("missing implementation {}".format(supplied))
    return expected_relative


def reject_legacy_statistical_claims(value, path="$") -> None:
    if isinstance(value, dict):
        if path == "$.code_identity.files":
            return
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("gate manifest key at {} is not a string".format(path))
            lowered = key.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_GATE_FIELD_FRAGMENTS):
                raise ValueError("legacy statistical gate field is forbidden: {}.{}".format(path, key))
            reject_legacy_statistical_claims(child, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_legacy_statistical_claims(child, "{}[{}]".format(path, index))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_GATE_TEXT):
            raise ValueError("legacy statistical gate claim is forbidden at {}".format(path))


def operation_opcode(operation, keys) -> str:
    require(isinstance(operation, dict), "structured_semantics", "operation must be an object")
    kind = operation.get("kind")
    key_index = {key: index for index, key in enumerate(keys)}
    expected_fields = {
        "add": {"kind", "target", "value"},
        "sub": {"kind", "target", "value"},
        "move": {"kind", "source", "target", "value"},
        "merge": {"kind", "source", "target"},
        "swap": {"kind", "left", "right"},
    }
    require(kind in expected_fields, "structured_semantics", "unknown operation kind")
    require(set(operation) == expected_fields[kind], "structured_semantics", "operation fields differ")
    if kind in {"add", "sub"}:
        require(operation["target"] in key_index, "structured_semantics", "invalid operation target")
        return "{}_{}".format(kind, key_index[operation["target"]])
    if kind in {"move", "merge"}:
        require(operation["source"] in key_index, "structured_semantics", "invalid operation source")
        require(operation["target"] in key_index, "structured_semantics", "invalid operation target")
        require(operation["source"] != operation["target"], "structured_semantics", "identical roles")
        return "{}_{}_{}".format(
            kind, key_index[operation["source"]], key_index[operation["target"]],
        )
    require(
        operation["left"] == keys[0] and operation["right"] == keys[1],
        "structured_semantics", "swap role order differs",
    )
    return "swap"


def apply_structured_operation(values, operation, keys, regime: RegimeAuditSpec):
    opcode = operation_opcode(operation, keys)
    result = dict(values)
    kind = operation["kind"]
    if kind in {"add", "sub", "move"}:
        value = exact_int(operation["value"], "operation value must be an exact integer")
        require(
            regime.value_range[0] <= value <= regime.value_range[1],
            "structured_semantics", "operation value leaves frozen board range",
        )
    else:
        value = 0
    if kind == "add":
        result[operation["target"]] += value
    elif kind == "sub":
        result[operation["target"]] -= value
    elif kind == "move":
        result[operation["source"]] -= value
        result[operation["target"]] += value
    elif kind == "merge":
        result[operation["target"]] += result[operation["source"]]
    else:
        result[keys[0]], result[keys[1]] = result[keys[1]], result[keys[0]]
    require(
        all(0 <= value < EXECUTOR_LIMIT for value in result.values()),
        "executor_safety", "register leaves exact executor range",
    )
    return result, opcode, value


def query_semantics(query, values, keys):
    require(isinstance(query, dict), "structured_semantics", "query must be an object")
    kind = query.get("kind")
    key_index = {key: index for index, key in enumerate(keys)}
    expected_fields = {
        "read": {"kind", "key", "answer", "text"},
        "sum": {"kind", "answer", "text"},
        "difference": {"kind", "high", "low", "answer", "text"},
    }
    require(kind in expected_fields, "structured_semantics", "unknown query kind")
    require(set(query) == expected_fields[kind], "structured_semantics", "query fields differ")
    if kind == "read":
        require(query["key"] in key_index, "structured_semantics", "invalid read key")
        opcode = "read_{}".format(key_index[query["key"]])
        expected_answer = values[query["key"]]
    elif kind == "sum":
        opcode = "sum"
        expected_answer = values[keys[0]] + values[keys[1]]
    else:
        require(query["high"] in key_index, "structured_semantics", "invalid high key")
        require(query["low"] in key_index, "structured_semantics", "invalid low key")
        require(query["high"] != query["low"], "structured_semantics", "difference roles identical")
        require(
            values[query["high"]] >= values[query["low"]],
            "executor_safety", "difference orientation is negative",
        )
        opcode = "difference_{}_{}".format(
            key_index[query["high"]], key_index[query["low"]],
        )
        expected_answer = values[query["high"]] - values[query["low"]]
    answer = exact_int(query["answer"], "query answer must be an exact integer")
    require(answer == expected_answer, "structured_semantics", "query answer disagrees with replay")
    require(0 <= answer < EXECUTOR_LIMIT, "executor_safety", "answer leaves exact executor range")
    require(isinstance(query["text"], str), "structured_semantics", "query text must be a string")
    return opcode, answer


def program_signature(row) -> tuple | None:
    try:
        keys = tuple(row["keys"])
        if len(keys) != 2 or set(row["initial"]) != set(keys):
            return None
        operations = tuple(
            (operation_opcode(operation, keys), int(operation.get("value", 0)))
            for operation in row["operations"]
        )
        query = row["query"]
        if query["kind"] == "read":
            target = "read_{}".format(keys.index(query["key"]))
        elif query["kind"] == "sum":
            target = "sum"
        else:
            target = "difference_{}_{}".format(
                keys.index(query["high"]), keys.index(query["low"]),
            )
        return (
            tuple(int(row["initial"][key]) for key in keys),
            operations,
            target,
        )
    except (KeyError, RowAuditError, TypeError, ValueError):
        return None


class PredecodedExactTable:
    """Expose the exact ALU argmax without recomputing it for every operation."""

    def __init__(self):
        self.targets = transition_basis_targets()

    def argmax(self, dim=-1):
        if dim != -1:
            raise ValueError("exact transition table only supports the categorical axis")
        return self.targets


def exact_table():
    return PredecodedExactTable()


def identify_template(actual: str, candidates: list[str], category: str) -> int:
    matches = [index for index, candidate in enumerate(candidates) if actual == candidate]
    require(len(matches) == 1, category, "surface does not match exactly one frozen template")
    return matches[0]


def audit_row(row, line_number: int, spec: AuditSpec, tokenizer, table, max_tokens: int) -> dict:
    required_fields = {
        "schema", "board", "question", "response", "answer", "source", "training_group",
        "family", "unit", "depth", "heldout", "eval_regime", "numeric_profile",
        "cell_id", "cell_index", "reference",
        "generation_seed", "initial", "keys", "operations", "query", "surface",
        "prompt_sha256", "program_sha256",
    }
    require(isinstance(row, dict), "schema", "row must be an object")
    require(set(row) == required_fields, "schema", "row fields differ from frozen schema")
    require(row["schema"] == SCHEMA, "schema", "schema identifier differs")
    require(row["board"] == spec.name, "schema", "board identifier differs")
    require(row["source"] == "r10_workspace_{}_v2".format(spec.name), "schema", "source differs")
    require(row["training_group"] == "r10_workspace_score_unseen", "schema", "training group differs")
    require(row["heldout"] is True, "schema", "heldout flag must be true")
    regime_name = row["eval_regime"]
    require(regime_name in spec.regimes, "schema", "eval regime differs")
    regime = spec.regimes[regime_name]
    require(
        row["numeric_profile"] == regime.numeric_profile,
        "schema", "numeric profile differs from regime",
    )
    cell_index = exact_int(row["cell_index"], "cell index must be an exact integer")
    require(cell_index >= 0, "balance", "cell index must be nonnegative")
    require(
        row["reference"] == "{}-{:06d}".format(spec.reference_prefix, line_number - 1),
        "reference", "reference is not stable line order",
    )
    exact_int(row["generation_seed"], "generation seed must be an exact integer")
    require(isinstance(row["question"], str), "schema", "question must be a string")
    require(isinstance(row["response"], str), "schema", "response must be a string")

    family = row["family"]
    require(family in spec.domains, "domain", "family is not in frozen new-domain registry")
    expected_keys, expected_unit = spec.domains[family]
    require(tuple(row["keys"]) == expected_keys, "domain", "domain keys differ")
    require(row["unit"] == expected_unit, "domain", "domain unit differs")
    keys = expected_keys
    require(isinstance(row["initial"], dict), "structured_semantics", "initial must be an object")
    require(set(row["initial"]) == set(keys), "structured_semantics", "initial keys differ")
    initial = {key: exact_int(row["initial"][key], "initial value must be exact") for key in keys}
    require(
        all(regime.initial_range[0] <= value <= regime.initial_range[1] for value in initial.values()),
        "structured_semantics", "initial value leaves frozen regime range",
    )
    depth = exact_int(row["depth"], "depth must be an exact integer")
    require(depth in regime.depths, "balance", "depth is outside frozen regime")
    require(isinstance(row["operations"], list), "structured_semantics", "operations must be a list")
    require(len(row["operations"]) == depth, "event_order", "depth and operation count differ")

    values = dict(initial)
    opcodes = []
    operation_values = []
    traces = [tuple(values[key] for key in keys)]
    for operation in row["operations"]:
        values, opcode, value = apply_structured_operation(values, operation, keys, regime)
        opcodes.append(opcode)
        operation_values.append(value)
        traces.append(tuple(values[key] for key in keys))
    target_query, answer = query_semantics(row["query"], values, keys)
    expected_cell = cell_id(regime_name, depth, target_query, family)
    require(row["cell_id"] == expected_cell, "balance", "cell identifier differs")
    require(row["answer"] == str(answer), "structured_semantics", "row answer differs")
    require(row["response"] == "The answer is {}.".format(answer), "structured_semantics", "response differs")

    question = row["question"]
    lines = question.splitlines()
    require(len(lines) == depth + 3, "event_order", "question line count differs")
    intro_candidates = [
        template.format(
            family=family,
            left=keys[0],
            right=keys[1],
            left_value=initial[keys[0]],
            right_value=initial[keys[1]],
            unit=expected_unit,
        )
        for template in spec.intro_templates
    ]
    intro_template = identify_template(lines[0], intro_candidates, "structured_semantics")
    lexical_initial = [int(value) for value in STANDALONE_INTEGER.findall(lines[0])]
    require(
        lexical_initial == [initial[keys[0]], initial[keys[1]]],
        "lexical_extraction", "intro numeral extraction differs",
    )

    operation_template_ids = []
    for offset, operation in enumerate(row["operations"]):
        line = lines[offset + 1]
        prefix = "{} {}: ".format(spec.event_prefix, offset + 1)
        require(line.startswith(prefix), "event_order", "event numbering or order differs")
        rendered = [
            prefix + template.format(unit=expected_unit, **operation)
            for template in spec.operation_templates[operation["kind"]]
        ]
        template_id = identify_template(line, rendered, "structured_semantics")
        operation_template_ids.append(template_id)
        lexical = [int(value) for value in STANDALONE_INTEGER.findall(line.split(":", 1)[-1])]
        expected_lexical = [] if operation["kind"] in {"merge", "swap"} else [operation["value"]]
        require(lexical == expected_lexical, "lexical_extraction", "event numeral extraction differs")

    query = row["query"]
    query_candidates = [
        template.format(left=keys[0], right=keys[1], **query)
        for template in spec.query_templates[query["kind"]]
    ]
    query_prefix = "{}: ".format(spec.query_prefix)
    require(lines[-2].startswith(query_prefix), "structured_semantics", "query prefix differs")
    query_surface = lines[-2][len(query_prefix):]
    query_template = identify_template(query_surface, query_candidates, "structured_semantics")
    require(query["text"] == query_surface, "structured_semantics", "query text differs from surface")
    require(lines[-1] == spec.answer_marker, "schema", "answer marker differs")
    surface = row["surface"]
    require(
        isinstance(surface, dict)
        and set(surface) == {"intro_template", "operation_templates", "query_template"},
        "schema", "surface metadata fields differ",
    )
    require(surface["intro_template"] == intro_template, "structured_semantics", "intro template id differs")
    require(
        surface["operation_templates"] == operation_template_ids,
        "structured_semantics", "operation template ids differ",
    )
    require(surface["query_template"] == query_template, "structured_semantics", "query template id differs")

    signature = (
        tuple(initial[key] for key in keys),
        tuple(zip(opcodes, operation_values)),
        target_query,
    )
    require(row["prompt_sha256"] == sha256_bytes(question.encode()), "hash", "prompt hash differs")
    require(row["program_sha256"] == canonical_hash(signature), "hash", "program hash differs")
    example = compile_example(row, tokenizer)
    require(len(example.ids) <= max_tokens, "tokenizer_length", "row exceeds tokenizer limit")
    require(
        tuple(example.operation_targets) == tuple(OPCODE_TO_ID[opcode] for opcode in opcodes),
        "compiler_contract", "compiled operation labels differ",
    )
    require(
        tuple(example.operation_values) == tuple(operation_values),
        "compiler_contract", "compiled operation values differ",
    )
    require(
        example.query_target == QUERY_TO_ID[target_query],
        "compiler_contract", "compiled query label differs",
    )
    require(example.initial_values == tuple(initial[key] for key in keys), "compiler_contract", "compiled initial differs")
    require(example.answer == answer, "compiler_contract", "compiled answer differs")
    referential = compile_referential_example(row, tokenizer)
    require(referential.compiled == example, "compiler_contract", "referential compiler differs")
    require(
        all(referential.intro_slot_targets)
        and not set(referential.intro_slot_targets[0]) & set(referential.intro_slot_targets[1]),
        "compiler_contract", "introductory referential targets are empty or overlap",
    )
    for operation, targets in zip(row["operations"], referential.operation_mention_targets):
        require(
            bool(targets) == (operation["kind"] != "swap"),
            "compiler_contract", "operation referential target shape differs",
        )
    require(
        bool(referential.query_mention_target) == (query["kind"] != "sum"),
        "compiler_contract", "query referential target shape differs",
    )
    oracle = execute_program(
        example.initial_values,
        example.operation_targets,
        example.operation_values,
        example.query_target,
        table,
        width=EXECUTOR_WIDTH,
    )
    require(oracle == answer, "oracle", "exact executor answer differs")
    return {
        "reference": row["reference"],
        "seed": row["generation_seed"],
        "regime": regime_name,
        "numeric_profile": row["numeric_profile"],
        "depth": depth,
        "family": family,
        "query": target_query,
        "cell_key": (regime_name, depth, target_query, family),
        "cell_id": expected_cell,
        "cell_index": cell_index,
        "query_kind": query["kind"],
        "opcodes": opcodes,
        "operation_kinds": [operation["kind"] for operation in row["operations"]],
        "token_length": len(example.ids),
        "intro_template": intro_template,
        "operation_templates": list(zip(
            [operation["kind"] for operation in row["operations"]], operation_template_ids,
        )),
        "query_template": (query["kind"], query_template),
        "signature": signature,
        "normalized_prompt": normalized(question),
        "traces": traces,
    }


def read_jsonl(path):
    rows = []
    parse_errors = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                parse_errors.append({"line": line_number, "category": "json", "error": str(error)})
    return rows, parse_errors


def balanced_counter(counter, labels, expected) -> bool:
    return set(counter) == set(labels) and all(counter[label] == expected for label in labels)


def audit_board(path, name: str, tokenizer, max_tokens: int, minimum_rows: int) -> dict:
    spec = SPECS[name]
    rows_with_lines, errors = read_jsonl(path)
    table = exact_table()
    valid = []
    for line_number, row in rows_with_lines:
        try:
            valid.append(audit_row(row, line_number, spec, tokenizer, table, max_tokens))
        except (IndexError, KeyError, RowAuditError, TypeError, ValueError) as error:
            category = error.category if isinstance(error, RowAuditError) else "exception"
            if len(errors) < 50:
                errors.append({"line": line_number, "category": category, "error": str(error)})

    total_rows = len(rows_with_lines)
    regimes = collections.Counter(item["regime"] for item in valid)
    numeric_profiles = collections.Counter(item["numeric_profile"] for item in valid)
    depths = collections.Counter(item["depth"] for item in valid)
    regime_depths = collections.Counter((item["regime"], item["depth"]) for item in valid)
    queries = collections.Counter(item["query"] for item in valid)
    depth_queries = collections.Counter((item["depth"], item["query"]) for item in valid)
    families = collections.Counter(item["family"] for item in valid)
    cells = collections.Counter(item["cell_key"] for item in valid)
    cell_indices = collections.defaultdict(list)
    for item in valid:
        cell_indices[item["cell_key"]].append(item["cell_index"])
    opcodes = collections.Counter(opcode for item in valid for opcode in item["opcodes"])
    operation_kinds = collections.Counter(kind for item in valid for kind in item["operation_kinds"])
    stratum_opcodes = collections.defaultdict(collections.Counter)
    for item in valid:
        stratum_opcodes[(item["regime"], item["depth"])].update(item["opcodes"])
    intro_templates = collections.Counter(item["intro_template"] for item in valid)
    operation_templates = collections.Counter(
        template for item in valid for template in item["operation_templates"]
    )
    query_templates = collections.Counter(item["query_template"] for item in valid)
    prompt_keys = [normalized(row.get("question", "")) for _, row in rows_with_lines if isinstance(row, dict)]
    program_keys = [program_signature(row) for _, row in rows_with_lines if isinstance(row, dict)]
    program_keys = [signature for signature in program_keys if signature is not None]
    references = [row.get("reference") for _, row in rows_with_lines if isinstance(row, dict)]
    seeds = sorted({item["seed"] for item in valid})
    token_lengths = [item["token_length"] for item in valid]
    frozen_cells = expected_cell_keys(spec)
    frozen_cell_set = set(frozen_cells)
    expected_cell = (
        minimum_rows // len(frozen_cells)
        if minimum_rows % len(frozen_cells) == 0 else -1
    )
    expected_regime = minimum_rows // len(spec.regimes)
    depths_allowed = all_depths(spec)
    expected_depth = minimum_rows // len(depths_allowed)
    expected_query = minimum_rows // len(QUERIES)
    expected_family = minimum_rows // len(spec.domains)
    expected_stratum = expected_cell * len(QUERIES) * len(spec.domains) if expected_cell >= 0 else -1
    expected_cell_indices = set(range(expected_cell)) if expected_cell >= 0 else set()
    missing_cells = sorted(frozen_cell_set - set(cells))
    unexpected_cells = sorted(set(cells) - frozen_cell_set)
    undersized_cells = sorted(
        (key, cells[key], expected_cell)
        for key in frozen_cells
        if expected_cell >= 0 and cells[key] < expected_cell
    )
    oversized_cells = sorted(
        (key, cells[key], expected_cell)
        for key in frozen_cells
        if expected_cell >= 0 and cells[key] > expected_cell
    )
    opcode_balance = True
    for regime_name, regime in spec.regimes.items():
        for depth in regime.depths:
            values = [stratum_opcodes[(regime_name, depth)][opcode] for opcode in OPCODES]
            opcode_balance &= (
                min(values, default=0) > 0
                and max(values, default=0) - min(values, default=0) <= 1
            )
    expected_operation_templates = {
        (kind, template)
        for kind, templates in spec.operation_templates.items()
        for template in range(len(templates))
    }
    expected_query_templates = {
        (kind, template)
        for kind, templates in spec.query_templates.items()
        for template in range(len(templates))
    }
    expected_profile_counts = collections.Counter({
        regime.numeric_profile: expected_regime for regime in spec.regimes.values()
    })
    exact_cells_pass = (
        expected_cell >= 0
        and not missing_cells
        and not unexpected_cells
        and not undersized_cells
        and not oversized_cells
    )
    cell_indices_pass = exact_cells_pass and all(
        len(cell_indices[key]) == expected_cell
        and set(cell_indices[key]) == expected_cell_indices
        for key in frozen_cells
    )
    checks = {
        "minimum_rows": total_rows >= minimum_rows,
        "exact_row_count": total_rows == minimum_rows,
        "strata_divisible": total_rows % len(frozen_cells) == 0,
        "all_rows_structurally_valid": not errors and len(valid) == total_rows,
        "regimes_exactly_balanced": balanced_counter(regimes, spec.regimes, expected_regime),
        "numeric_profiles_exactly_balanced": numeric_profiles == expected_profile_counts,
        "depths_exactly_balanced": balanced_counter(depths, depths_allowed, expected_depth),
        "regime_depths_exactly_balanced": (
            set(regime_depths) == {
                (regime_name, depth)
                for regime_name, regime in spec.regimes.items()
                for depth in regime.depths
            }
            and all(count == expected_stratum for count in regime_depths.values())
        ),
        "queries_exactly_balanced": balanced_counter(queries, QUERIES, expected_query),
        "depth_query_cells_exactly_balanced": (
            set(depth_queries) == {(depth, query) for depth in depths_allowed for query in QUERIES}
            and all(
                count == expected_cell * len(spec.domains)
                for count in depth_queries.values()
            )
        ),
        "regime_depth_query_family_cells_exact": exact_cells_pass,
        "cell_indices_exact": cell_indices_pass,
        "domains_exactly_balanced": balanced_counter(families, spec.domains, expected_family),
        "all_operation_labels_covered": set(opcodes) == set(OPCODES),
        "operation_labels_globally_balanced": (
            set(opcodes) == set(OPCODES)
            and max(opcodes.values(), default=0) - min(opcodes.values(), default=0) <= 1
        ),
        "operation_labels_balanced_within_each_regime_depth": opcode_balance,
        "all_operation_kinds_covered": set(operation_kinds) == {"add", "sub", "move", "merge", "swap"},
        "all_intro_templates_covered": set(intro_templates) == set(range(len(spec.intro_templates))),
        "all_operation_templates_covered": set(operation_templates) == expected_operation_templates,
        "all_query_templates_covered": set(query_templates) == expected_query_templates,
        "zero_normalized_prompt_duplicates": len(prompt_keys) == total_rows == len(set(prompt_keys)),
        "zero_program_duplicates": len(program_keys) == total_rows == len(set(program_keys)),
        "zero_reference_duplicates": len(references) == total_rows == len(set(references)),
        "one_generation_seed": len(seeds) == 1,
        "canonical_generation_seed": seeds == [CANONICAL_GENERATOR_SEEDS[name]],
        "tokenizer_limit": bool(token_lengths) and max(token_lengths) <= max_tokens,
        "zero_oracle_errors": not any(error.get("category") == "oracle" for error in errors),
        "zero_lexical_errors": not any(error.get("category") == "lexical_extraction" for error in errors),
        "zero_event_order_errors": not any(error.get("category") == "event_order" for error in errors),
        "zero_semantic_errors": not any(
            error.get("category") in {"structured_semantics", "compiler_contract", "executor_safety"}
            for error in errors
        ),
    }
    return {
        "board": name,
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "rows": total_rows,
        "valid_rows": len(valid),
        "generation_seeds": seeds,
        "regimes": dict(sorted(regimes.items())),
        "numeric_profiles": dict(sorted(numeric_profiles.items())),
        "depths": {str(key): value for key, value in sorted(depths.items())},
        "regime_depths": {
            "{}:depth={}".format(regime, depth): regime_depths[(regime, depth)]
            for regime, depth in sorted(regime_depths)
        },
        "queries": dict(sorted(queries.items())),
        "depth_query_cells": {
            "{}:{}".format(depth, query): depth_queries[(depth, query)]
            for depth in depths_allowed for query in QUERIES
        },
        "expected_cell_count": len(frozen_cells),
        "rows_per_exact_cell": expected_cell,
        "exact_cells": {
            cell_id(*key): cells[key] for key in frozen_cells
        },
        "cell_failures": {
            "missing": [cell_id(*key) for key in missing_cells],
            "unexpected": [cell_id(*key) for key in unexpected_cells],
            "undersized": [
                {"cell": cell_id(*key), "actual": actual, "expected": expected}
                for key, actual, expected in undersized_cells
            ],
            "oversized": [
                {"cell": cell_id(*key), "actual": actual, "expected": expected}
                for key, actual, expected in oversized_cells
            ],
        },
        "families": dict(sorted(families.items())),
        "opcodes": dict(sorted(opcodes.items())),
        "opcodes_by_regime_depth": {
            "{}:depth={}".format(regime_name, depth): {
                opcode: stratum_opcodes[(regime_name, depth)][opcode] for opcode in OPCODES
            }
            for regime_name, regime in spec.regimes.items()
            for depth in regime.depths
        },
        "operation_kinds": dict(sorted(operation_kinds.items())),
        "template_coverage": {
            "intro": {str(key): value for key, value in sorted(intro_templates.items())},
            "operations": {
                "{}:{}".format(kind, template): count
                for (kind, template), count in sorted(operation_templates.items())
            },
            "queries": {
                "{}:{}".format(kind, template): count
                for (kind, template), count in sorted(query_templates.items())
            },
        },
        "tokenizer_lengths": {
            "min": min(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
            "mean": sum(token_lengths) / len(token_lengths) if token_lengths else None,
            "limit": max_tokens,
        },
        "duplicate_counts": {
            "normalized_prompts": total_rows - len(set(prompt_keys)),
            "programs": total_rows - len(set(program_keys)),
            "references": total_rows - len(set(references)),
        },
        "errors": errors,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "rows_data": [row for _, row in rows_with_lines],
    }


def board_index(rows, spec: AuditSpec) -> dict:
    exact = set()
    grams = set()
    programs = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            continue
        exact.add(normalized(row["question"]))
        grams.update(ngrams(row["question"]))
        signature = program_signature(row)
        if signature is not None:
            programs.add(signature)
    phrases = {
        normalized(phrase)
        for family, (keys, unit) in spec.domains.items()
        for phrase in (family, *keys, unit)
    }
    return {"exact": exact, "grams": grams, "programs": programs, "phrases": phrases}


def phrase_hits(words: list[str], phrases: set[str]) -> set[str]:
    padded = " {} ".format(" ".join(words))
    return {phrase for phrase in phrases if " {} ".format(phrase) in padded}


def scan_source(path, role: str, indices) -> dict:
    reports = {
        name: {
            "exact_prompt_rows": 0,
            "ngram13_rows": 0,
            "program_rows": 0,
            "novel_domain_phrase_rows": 0,
            "sample_hits": [],
        }
        for name in indices
    }
    rows = 0
    errors = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
                question = row["question"]
                if not isinstance(question, str):
                    raise TypeError("question is not a string")
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                if len(errors) < 20:
                    errors.append({"line": line_number, "error": str(error)})
                continue
            key = normalized(question)
            words = key.split()
            source_grams = {
                " ".join(words[index:index + NGRAM_WIDTH])
                for index in range(max(0, len(words) - NGRAM_WIDTH + 1))
            }
            signature = program_signature(row)
            for name, index in indices.items():
                exact = key in index["exact"]
                gram = bool(source_grams & index["grams"])
                program = signature is not None and signature in index["programs"]
                phrases = phrase_hits(words, index["phrases"])
                reports[name]["exact_prompt_rows"] += int(exact)
                reports[name]["ngram13_rows"] += int(gram)
                reports[name]["program_rows"] += int(program)
                reports[name]["novel_domain_phrase_rows"] += int(bool(phrases))
                if (exact or gram or program or phrases) and len(reports[name]["sample_hits"]) < 12:
                    reports[name]["sample_hits"].append({
                        "line": line_number,
                        "exact": exact,
                        "ngram13": gram,
                        "program": program,
                        "phrases": sorted(phrases),
                    })
    return {
        "role": role,
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "rows_scanned": rows,
        "errors": errors,
        "boards": reports,
    }


def source_scans_pass(source_reports) -> bool:
    for source in source_reports:
        if source["errors"]:
            return False
        for report in source["boards"].values():
            if any(report[key] for key in (
                "exact_prompt_rows", "ngram13_rows", "program_rows", "novel_domain_phrase_rows",
            )):
                return False
    return True


def expected_schedule_contract() -> dict:
    return {
        name: {
            "cells": len(expected_cell_keys(spec)),
            "regimes": {
                regime_name: {
                    "depths": list(regime.depths),
                    "numeric_profile": regime.numeric_profile,
                    "initial_range_inclusive": list(regime.initial_range),
                    "event_value_range_inclusive": list(regime.value_range),
                    "families": list(spec.domains),
                    "queries": list(QUERIES),
                }
                for regime_name, regime in spec.regimes.items()
            },
        }
        for name, spec in SPECS.items()
    }


def expected_build_manifest_check_fields(manifest) -> set[str]:
    fields = {
        "exact_fields", "build_kind", "schema", "cpu_only", "score_outputs_not_read",
        "build_not_score_authorizing", "ngram_width", "executor_width",
        "generation_contract", "claim_boundary", "schedule_contract", "tokenizer_path",
        "tokenizer_sha256", "tokenizer_limit", "input_count", "canonical_r5_input",
        "build_cross_scan_zero",
    }
    for name in CANONICAL_GENERATOR_SEEDS:
        fields.update({
            "{}_path".format(name),
            "{}_sha256".format(name),
            "{}_rows".format(name),
            "{}_regimes".format(name),
            "{}_exact_cell_count".format(name),
            "{}_rows_per_cell".format(name),
            "{}_exact_cells".format(name),
            "{}_seed".format(name),
            "{}_r5_binding".format(name),
        })
    inputs = manifest.get("inputs", []) if isinstance(manifest, dict) else []
    fields.update("input_{}_binding".format(index) for index in range(len(inputs)))
    return fields


def audit_manifest_binding(
    manifest_path,
    manifest_bytes,
    manifest,
    tokenizer_path,
    max_tokens,
    board_reports,
    source_reports,
) -> dict:
    checks = {}
    checks["exact_fields"] = isinstance(manifest, dict) and set(manifest) == BUILD_MANIFEST_FIELDS
    checks["build_kind"] = manifest.get("build") == "r10_workspace_boards_v2"
    checks["schema"] = manifest.get("schema") == SCHEMA
    checks["cpu_only"] = manifest.get("cpu_only") is True
    checks["score_outputs_not_read"] = (
        manifest.get("score_outputs_read") is False and manifest.get("score_artifacts") == []
    )
    checks["build_not_score_authorizing"] = manifest.get("ready_for_r10_score_run") is False
    checks["ngram_width"] = manifest.get("ngram_width") == NGRAM_WIDTH
    checks["executor_width"] = manifest.get("executor_width") == EXECUTOR_WIDTH
    checks["generation_contract"] = (
        manifest.get("generation_contract") == canonical_generation_contract()
    )
    checks["claim_boundary"] = (
        manifest.get("claim_boundary") == BUILD_MANIFEST_CLAIM_BOUNDARY
    )
    checks["schedule_contract"] = (
        manifest.get("schedule_contract") == expected_schedule_contract()
    )
    tokenizer = manifest.get("tokenizer", {})
    checks["tokenizer_path"] = tokenizer.get("path") == str(Path(tokenizer_path).resolve())
    checks["tokenizer_sha256"] = tokenizer.get("sha256") == sha256_file(tokenizer_path)
    checks["tokenizer_limit"] = tokenizer.get("max_tokens") == max_tokens
    outputs = manifest.get("outputs", {})
    for name, report in board_reports.items():
        record = outputs.get(name, {})
        checks["{}_path".format(name)] = record.get("path") == report["path"]
        checks["{}_sha256".format(name)] = record.get("sha256") == report["sha256"]
        checks["{}_rows".format(name)] = record.get("rows") == report["rows"]
        checks["{}_regimes".format(name)] = record.get("regimes") == report["regimes"]
        checks["{}_exact_cell_count".format(name)] = (
            record.get("expected_cell_count") == report["expected_cell_count"]
        )
        checks["{}_rows_per_cell".format(name)] = (
            record.get("rows_per_exact_cell") == report["rows_per_exact_cell"]
        )
        checks["{}_exact_cells".format(name)] = (
            record.get("exact_cells") == report["exact_cells"]
        )
        checks["{}_seed".format(name)] = (
            report["generation_seeds"] == [CANONICAL_GENERATOR_SEEDS[name]]
            and record.get("seed") == CANONICAL_GENERATOR_SEEDS[name]
        )
        checks["{}_r5_binding".format(name)] = (
            record.get("r5_novelty_board_sha256")
            == CANONICAL_R5_NOVELTY_BOARD_SHA256
        )
    manifest_inputs = manifest.get("inputs", [])
    checks["input_count"] = len(manifest_inputs) == len(source_reports)
    for index, report in enumerate(source_reports):
        if index >= len(manifest_inputs):
            checks["input_{}_binding".format(index)] = False
            continue
        record = manifest_inputs[index]
        checks["input_{}_binding".format(index)] = (
            record.get("role") == report["role"]
            and record.get("path") == report["path"]
            and record.get("sha256") == report["sha256"]
            and record.get("rows_scanned") == report["rows_scanned"]
        )
    r5_inputs = [
        record for record in manifest_inputs
        if isinstance(record, dict) and record.get("role") == "r5_fresh_board"
    ]
    checks["canonical_r5_input"] = (
        len(r5_inputs) == 1
        and r5_inputs[0].get("sha256") == CANONICAL_R5_NOVELTY_BOARD_SHA256
    )
    cross = manifest.get("cross_board_scan", {})
    checks["build_cross_scan_zero"] = cross == {
        "exact_prompt_hits": 0, "ngram13_hits": 0, "program_hits": 0,
    }
    return {
        "build": manifest.get("build"),
        "path": str(Path(manifest_path).resolve()),
        "sha256": sha256_bytes(manifest_bytes),
        "byte_length": len(manifest_bytes),
        "bytes_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        "content": manifest,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def verify_build_manifest_binding(
    binding,
    *,
    require_checks: bool,
    require_external_bytes: bool,
) -> dict:
    base_fields = {
        "build", "path", "sha256", "byte_length", "bytes_base64", "content",
    }
    expected_fields = base_fields | ({"checks", "all_checks_pass"} if require_checks else set())
    if not isinstance(binding, dict) or set(binding) != expected_fields:
        raise ValueError("build manifest binding fields differ from the frozen contract")
    if binding.get("build") != "r10_workspace_boards_v2":
        raise ValueError("build manifest binding kind differs")
    path = binding.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("build manifest binding path must be absolute")
    expected_sha256 = binding.get("sha256")
    if not isinstance(expected_sha256, str) or SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("build manifest binding SHA256 is invalid")
    encoded = binding.get("bytes_base64")
    if not isinstance(encoded, str):
        raise ValueError("build manifest binding lacks exact bytes")
    try:
        manifest_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ValueError("build manifest exact bytes are not valid base64") from error
    if binding.get("byte_length") != len(manifest_bytes):
        raise ValueError("build manifest exact byte length differs")
    if sha256_bytes(manifest_bytes) != expected_sha256:
        raise ValueError("build manifest exact byte hash differs")
    try:
        content = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("build manifest exact bytes are not valid JSON") from error
    if content != binding.get("content"):
        raise ValueError("build manifest parsed content differs from its exact bytes")
    if not isinstance(content, dict) or set(content) != BUILD_MANIFEST_FIELDS:
        raise ValueError("embedded build manifest content fields differ")
    if content.get("build") != "r10_workspace_boards_v2":
        raise ValueError("embedded build manifest kind differs")
    if not (
        content.get("schema") == SCHEMA
        and content.get("cpu_only") is True
        and content.get("ngram_width") == NGRAM_WIDTH
        and content.get("executor_width") == EXECUTOR_WIDTH
        and content.get("schedule_contract") == expected_schedule_contract()
        and content.get("cross_board_scan") == {
            "exact_prompt_hits": 0,
            "ngram13_hits": 0,
            "program_hits": 0,
        }
    ):
        raise ValueError("embedded build manifest core contract differs")
    if content.get("generation_contract") != canonical_generation_contract():
        raise ValueError("embedded build manifest generation contract differs")
    if content.get("claim_boundary") != BUILD_MANIFEST_CLAIM_BOUNDARY:
        raise ValueError("embedded build manifest claim boundary differs")
    if not (
        content.get("score_outputs_read") is False
        and content.get("score_artifacts") == []
        and content.get("ready_for_r10_score_run") is False
    ):
        raise ValueError("embedded build manifest is not score-blind")
    if require_checks:
        checks = binding.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or set(checks) != expected_build_manifest_check_fields(content)
            or not all(value is True for value in checks.values())
            or binding.get("all_checks_pass") is not True
        ):
            raise ValueError("build manifest admission checks did not all pass")
    if require_external_bytes:
        external_path = Path(path)
        if not external_path.is_file() or external_path.read_bytes() != manifest_bytes:
            raise ValueError("external build manifest bytes differ from the frozen binding")
    return content


def audit_compatibility_admissions(paths, board_reports, source_reports, tokenizer_path) -> dict:
    """Bind the legacy admission shapes consumed by the frozen score extractor."""
    if paths is None:
        return {"enabled": False, "boards": {}, "all_checks_pass": True}
    training_hashes = {
        report["sha256"] for report in source_reports if report["role"] == "training_data"
    }
    tokenizer_sha256 = sha256_file(tokenizer_path)
    output = {}
    for name, admission_paths in paths.items():
        structural_path = admission_paths["structural"]
        label_path = admission_paths["label"]
        with open(structural_path) as source:
            structural = json.load(source)
        with open(label_path) as source:
            labels = json.load(source)
        datasets = labels.get("datasets", {}) if isinstance(labels, dict) else {}
        training = datasets.get("train", {}) if isinstance(datasets, dict) else {}
        evaluation = datasets.get("eval", {}) if isinstance(datasets, dict) else {}
        checks = {
            "structural_audit_kind": structural.get("audit") == "role_equivariant_microcode_v3",
            "structural_passed": structural.get("all_checks_pass") is True,
            "structural_training_bound": structural.get("train_sha256") in training_hashes,
            "structural_board_bound": structural.get("eval_sha256") == board_reports[name]["sha256"],
            "structural_tokenizer_bound": structural.get("tokenizer_sha256") == tokenizer_sha256,
            "label_audit_kind": labels.get("audit") == "referential_slot_label_admission_v1",
            "label_passed": labels.get("all_checks_pass") is True,
            "label_training_passed": training.get("all_checks_pass") is True,
            "label_training_bound": training.get("sha256") in training_hashes,
            "label_board_passed": evaluation.get("all_checks_pass") is True,
            "label_board_bound": evaluation.get("sha256") == board_reports[name]["sha256"],
            "label_tokenizer_bound": labels.get("tokenizer_sha256") == tokenizer_sha256,
            "admissions_share_training": structural.get("train_sha256") == training.get("sha256"),
            "admissions_share_board": structural.get("eval_sha256") == evaluation.get("sha256"),
        }
        output[name] = {
            "structural": {
                "path": str(Path(structural_path).resolve()),
                "sha256": sha256_file(structural_path),
            },
            "referential_labels": {
                "path": str(Path(label_path).resolve()),
                "sha256": sha256_file(label_path),
            },
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        }
    return {
        "enabled": True,
        "boards": output,
        "all_checks_pass": set(output) == set(board_reports) and all(
            item["all_checks_pass"] for item in output.values()
        ),
    }


def audit_bundle(
    *,
    training_data,
    r5_board,
    tokenizer_path,
    calibration_path,
    confirmation_path,
    build_manifest_path,
    max_tokens=2048,
    minimum_calibration=800,
    minimum_confirmation=1840,
    require_confirmation_capacity=True,
    compatibility_admissions=None,
) -> dict:
    training_data = list(training_data)
    paths = [
        *training_data, r5_board, tokenizer_path, calibration_path,
        confirmation_path, build_manifest_path,
    ]
    if compatibility_admissions is not None:
        paths.extend(
            path
            for admissions in compatibility_admissions.values()
            for path in admissions.values()
        )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing inputs: {}".format(", ".join(missing)))
    if sha256_file(r5_board) != CANONICAL_R5_NOVELTY_BOARD_SHA256:
        raise ValueError("R5 novelty board SHA256 differs from the frozen canonical artifact")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    board_reports = {
        "calibration": audit_board(
            calibration_path, "calibration", tokenizer, max_tokens, minimum_calibration,
        ),
        "confirmation": audit_board(
            confirmation_path, "confirmation", tokenizer, max_tokens, minimum_confirmation,
        ),
    }
    indices = {
        name: board_index(report["rows_data"], SPECS[name])
        for name, report in board_reports.items()
    }
    cross_exact = indices["calibration"]["exact"] & indices["confirmation"]["exact"]
    cross_grams = indices["calibration"]["grams"] & indices["confirmation"]["grams"]
    cross_programs = indices["calibration"]["programs"] & indices["confirmation"]["programs"]
    source_reports = [
        scan_source(path, "training_data", indices) for path in training_data
    ] + [scan_source(r5_board, "r5_fresh_board", indices)]
    build_manifest_bytes = Path(build_manifest_path).read_bytes()
    build_manifest = json.loads(build_manifest_bytes)
    binding = audit_manifest_binding(
        build_manifest_path,
        build_manifest_bytes,
        build_manifest,
        tokenizer_path,
        max_tokens,
        board_reports,
        source_reports,
    )
    compatibility = audit_compatibility_admissions(
        compatibility_admissions, board_reports, source_reports, tokenizer_path,
    )
    calibration_seeds = board_reports["calibration"]["generation_seeds"]
    confirmation_seeds = board_reports["confirmation"]["generation_seeds"]
    seeds_distinct = (
        len(calibration_seeds) == 1
        and len(confirmation_seeds) == 1
        and calibration_seeds[0] != confirmation_seeds[0]
    )
    seeds_canonical = (
        calibration_seeds == [CANONICAL_GENERATOR_SEEDS["calibration"]]
        and confirmation_seeds == [CANONICAL_GENERATOR_SEEDS["confirmation"]]
    )
    for name, board in board_reports.items():
        board["generation_seed"] = (
            board["generation_seeds"][0]
            if len(board["generation_seeds"]) == 1 else None
        )
        board["r5_novelty_board_sha256"] = CANONICAL_R5_NOVELTY_BOARD_SHA256
    confirmation_cell_counts = collections.Counter(
        regime for regime, _, _, _ in expected_cell_keys(SPECS["confirmation"])
    )
    confirmation_rows_per_cell = board_reports["confirmation"]["rows_per_exact_cell"]
    confirmation_minimum_per_cell = (
        math.ceil(
            CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL * confirmation_rows_per_cell
        )
        if confirmation_rows_per_cell >= 0 else -1
    )
    confirmation_partitions = {
        regime: {
            "rows": count,
            "exact_cells": confirmation_cell_counts[regime],
            "rows_per_exact_cell": confirmation_rows_per_cell,
            "acceptance_quota_fraction_per_exact_cell": (
                CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL
            ),
            "minimum_accepted_per_exact_cell": confirmation_minimum_per_cell,
            "minimum_accepted": (
                confirmation_cell_counts[regime] * confirmation_minimum_per_cell
            ),
            "maximum_false_certificates": CONFIRMATION_MAX_FALSE_CERTIFICATES,
        }
        for regime, count in board_reports["confirmation"]["regimes"].items()
    }
    capacity_pass = not require_confirmation_capacity or (
        set(confirmation_partitions) == set(SPECS["confirmation"].regimes)
        and all(
            partition["rows"] == DEFAULT_CONFIRMATION_PARTITION_ROWS
            and partition["exact_cells"] == CONFIRMATION_CELLS_PER_PARTITION
            and partition["rows_per_exact_cell"] == DEFAULT_CELL_ROWS["confirmation"]
            and partition["minimum_accepted_per_exact_cell"]
            == CONFIRMATION_MIN_ACCEPTED_PER_CELL
            and partition["minimum_accepted"]
            == CONFIRMATION_MIN_ACCEPTED_PER_PARTITION
            and partition["maximum_false_certificates"]
            == CONFIRMATION_MAX_FALSE_CERTIFICATES
            for partition in confirmation_partitions.values()
        )
    )
    regime_sets_disjoint = (
        set(board_reports["calibration"]["regimes"])
        .isdisjoint(board_reports["confirmation"]["regimes"])
    )
    hard_scan = {
        "source_reports": source_reports,
        "all_source_scans_zero": source_scans_pass(source_reports),
        "cross_board": {
            "exact_prompt_hits": len(cross_exact),
            "ngram13_hits": len(cross_grams),
            "program_hits": len(cross_programs),
            "sample_grams": sorted(cross_grams)[:12],
        },
        "cross_board_zero": not cross_exact and not cross_grams and not cross_programs,
    }
    report = {
        "audit": ADMISSION_AUDIT,
        "schema": SCHEMA,
        "cpu_only": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "tokenizer": {
            "path": str(Path(tokenizer_path).resolve()),
            "sha256": sha256_file(tokenizer_path),
            "max_tokens": max_tokens,
        },
        "build_manifest": binding,
        "extractor_compatibility_admissions": compatibility,
        "boards": {
            name: {key: value for key, value in board.items() if key != "rows_data"}
            for name, board in board_reports.items()
        },
        "hard_scan": hard_scan,
        "deterministic_distinct_seeds": seeds_distinct,
        "canonical_generator_seeds": seeds_canonical,
        "generation_contract": canonical_generation_contract(),
        "calibration_confirmation_regimes_disjoint": regime_sets_disjoint,
        "confirmation_empirical_quota": {
            "partitions": confirmation_partitions,
            "scope": "frozen confirmation rows only",
            "all_exact_cells_required": True,
            "extrapolation_beyond_frozen_board_forbidden": True,
            "check_required": require_confirmation_capacity,
            "passes": capacity_pass,
        },
        "claim_boundary": (
            "This score-blind v2 admission proves only immutable hashes, exact scheduled cells, "
            "lexical and structured compiler semantics, exact CPU oracle safety, frozen finite-board "
            "quotas, and prompt/program novelty. It contains no neural score and makes no claim "
            "beyond the admitted rows."
        ),
    }
    report["all_checks_pass"] = (
        all(board["all_checks_pass"] for board in board_reports.values())
        and hard_scan["all_source_scans_zero"]
        and hard_scan["cross_board_zero"]
        and binding["all_checks_pass"]
        and compatibility["all_checks_pass"]
        and seeds_distinct
        and seeds_canonical
        and regime_sets_disjoint
        and capacity_pass
    )
    report["r10_score_run_precondition_satisfied"] = (
        report["all_checks_pass"] and compatibility["enabled"]
    )
    return report


def frozen_partition_contract() -> dict:
    return {
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
                "rows": DEFAULT_CONFIRMATION_PARTITION_ROWS,
                "depths": [4, 8],
                "numeric_profile": "in_range",
                "exact_cells": CONFIRMATION_CELLS_PER_PARTITION,
                "rows_per_cell": DEFAULT_CELL_ROWS["confirmation"],
                "rows_per_exact_cell": DEFAULT_CELL_ROWS["confirmation"],
                "minimum_accepted_per_exact_cell": CONFIRMATION_MIN_ACCEPTED_PER_CELL,
                "minimum_accepted": CONFIRMATION_MIN_ACCEPTED_PER_PARTITION,
                "maximum_false_certificates": CONFIRMATION_MAX_FALSE_CERTIFICATES,
            },
            "full_ood": {
                "rows": DEFAULT_CONFIRMATION_PARTITION_ROWS,
                "depths": [16, 32],
                "numeric_profile": "shifted",
                "exact_cells": CONFIRMATION_CELLS_PER_PARTITION,
                "rows_per_cell": DEFAULT_CELL_ROWS["confirmation"],
                "rows_per_exact_cell": DEFAULT_CELL_ROWS["confirmation"],
                "minimum_accepted_per_exact_cell": CONFIRMATION_MIN_ACCEPTED_PER_CELL,
                "minimum_accepted": CONFIRMATION_MIN_ACCEPTED_PER_PARTITION,
                "maximum_false_certificates": CONFIRMATION_MAX_FALSE_CERTIFICATES,
            },
        },
    }


def frozen_confirmation_thresholds() -> dict:
    return {
        "scope": "frozen confirmation rows only",
        "minimum_selective_coverage_each_partition": (
            CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL
        ),
        "acceptance_quota_fraction_per_exact_cell": (
            CONFIRMATION_ACCEPTANCE_FRACTION_PER_CELL
        ),
        "acceptance_quota_rounding": "ceil(0.40 * 23) = 10",
        "exact_cells_each_partition": CONFIRMATION_CELLS_PER_PARTITION,
        "rows_each_exact_cell": DEFAULT_CELL_ROWS["confirmation"],
        "minimum_accepted_each_exact_cell": CONFIRMATION_MIN_ACCEPTED_PER_CELL,
        "minimum_accepted_each_partition": CONFIRMATION_MIN_ACCEPTED_PER_PARTITION,
        "maximum_false_certificates_each_exact_cell": CONFIRMATION_MAX_FALSE_CERTIFICATES,
        "maximum_false_certificates_each_partition": CONFIRMATION_MAX_FALSE_CERTIFICATES,
        "minimum_empirical_selective_accuracy_each_partition": (
            CONFIRMATION_MIN_EMPIRICAL_ACCURACY
        ),
        "all_exact_cells_required": True,
        "pooled_partition_substitution_forbidden": True,
        "extrapolation_beyond_frozen_board_forbidden": True,
    }


def _all_true_checks(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _nonempty_line_count(path: Path) -> int:
    with open(path, "rb") as source:
        return sum(1 for line in source if line.strip())


def validate_admission_report_for_freeze(
    report,
    *,
    admission_report_path,
    expected_code_identity: dict,
) -> dict:
    admission_path = Path(admission_report_path)
    if not admission_path.is_file():
        raise FileNotFoundError("missing gate input {}".format(admission_path))
    try:
        admission_content = json.loads(admission_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("admission report is not valid JSON") from error
    if admission_content != report:
        raise ValueError("caller report differs from the exact admission report bytes")
    if not (
        report.get("audit") == ADMISSION_AUDIT
        and report.get("schema") == SCHEMA
        and report.get("cpu_only") is True
        and report.get("score_outputs_read") is False
        and report.get("score_artifacts") == []
        and report.get("generation_contract") == canonical_generation_contract()
        and report.get("canonical_generator_seeds") is True
        and report.get("deterministic_distinct_seeds") is True
        and report.get("calibration_confirmation_regimes_disjoint") is True
        and report.get("all_checks_pass") is True
        and report.get("r10_score_run_precondition_satisfied") is True
    ):
        raise ValueError("admission report does not satisfy the frozen score-blind contract")
    if report.get("code_identity") != expected_code_identity:
        raise ValueError("admission report code identity differs from the clean checkout")
    if (
        report.get("code_identity_aggregate_sha256")
        != expected_code_identity["aggregate_sha256"]
    ):
        raise ValueError("admission report code identity aggregate differs")

    build_binding = report.get("build_manifest")
    build_content = verify_build_manifest_binding(
        build_binding,
        require_checks=True,
        require_external_bytes=True,
    )
    tokenizer = report.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer != build_content.get("tokenizer"):
        raise ValueError("admission tokenizer differs from the exact build manifest")
    tokenizer_path = Path(tokenizer.get("path", ""))
    if (
        not tokenizer_path.is_file()
        or tokenizer.get("sha256") != sha256_file(tokenizer_path)
    ):
        raise ValueError("admission tokenizer bytes differ")

    inputs = build_content.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("embedded build manifest inputs are invalid")
    r5_inputs = []
    training_inputs = []
    for record in inputs:
        if not isinstance(record, dict):
            raise ValueError("embedded build manifest input record is invalid")
        input_path = Path(record.get("path", ""))
        if not input_path.is_file() or record.get("sha256") != sha256_file(input_path):
            raise ValueError("build input bytes differ: {}".format(input_path))
        if record.get("rows_scanned") != _nonempty_line_count(input_path):
            raise ValueError("build input row count differs: {}".format(input_path))
        if record.get("role") == "r5_fresh_board":
            r5_inputs.append(record)
        elif record.get("role") == "training_data":
            training_inputs.append(record)
        else:
            raise ValueError("embedded build manifest input role differs")
    if (
        not training_inputs
        or len(r5_inputs) != 1
        or r5_inputs[0].get("sha256") != CANONICAL_R5_NOVELTY_BOARD_SHA256
    ):
        raise ValueError("embedded build manifest does not bind the canonical R5 input")

    boards = report.get("boards")
    outputs = build_content.get("outputs")
    if not isinstance(boards, dict) or set(boards) != set(CANONICAL_GENERATOR_SEEDS):
        raise ValueError("admission board set differs")
    if not isinstance(outputs, dict) or set(outputs) != set(boards):
        raise ValueError("embedded build manifest output set differs")
    for name, expected_seed in CANONICAL_GENERATOR_SEEDS.items():
        board = boards[name]
        output = outputs[name]
        if not isinstance(board, dict) or not isinstance(output, dict):
            raise ValueError("{} board binding is invalid".format(name))
        if (
            board.get("all_checks_pass") is not True
            or not _all_true_checks(board.get("checks"))
            or set(board["checks"]) != BOARD_ADMISSION_CHECK_FIELDS
        ):
            raise ValueError("{} board admission checks did not all pass".format(name))
        if (
            board.get("generation_seeds") != [expected_seed]
            or board.get("generation_seed") != expected_seed
            or board.get("r5_novelty_board_sha256")
            != CANONICAL_R5_NOVELTY_BOARD_SHA256
        ):
            raise ValueError("{} board custody binding differs".format(name))
        for key in (
            "path", "sha256", "rows", "regimes", "expected_cell_count",
            "rows_per_exact_cell", "exact_cells",
        ):
            if output.get(key) != board.get(key):
                raise ValueError("{} build/admission {} binding differs".format(name, key))
        if (
            output.get("seed") != expected_seed
            or output.get("r5_novelty_board_sha256")
            != CANONICAL_R5_NOVELTY_BOARD_SHA256
        ):
            raise ValueError("{} build custody binding differs".format(name))
        board_path = Path(board.get("path", ""))
        if not board_path.is_file() or board.get("sha256") != sha256_file(board_path):
            raise ValueError("{} board bytes differ".format(name))
        if board.get("rows") != _nonempty_line_count(board_path):
            raise ValueError("{} board row count differs".format(name))

    hard_scan = report.get("hard_scan")
    if not (
        isinstance(hard_scan, dict)
        and hard_scan.get("all_source_scans_zero") is True
        and hard_scan.get("cross_board_zero") is True
    ):
        raise ValueError("admission novelty scans did not all pass")
    empirical = report.get("confirmation_empirical_quota")
    if not isinstance(empirical, dict) or empirical.get("passes") is not True:
        raise ValueError("admission finite-board empirical quota did not pass")

    compatibility = report.get("extractor_compatibility_admissions")
    compatible_boards = compatibility.get("boards") if isinstance(compatibility, dict) else None
    if not (
        isinstance(compatibility, dict)
        and compatibility.get("enabled") is True
        and compatibility.get("all_checks_pass") is True
        and isinstance(compatible_boards, dict)
        and set(compatible_boards) == set(boards)
    ):
        raise ValueError("extractor compatibility admissions did not all pass")
    for name, item in compatible_boards.items():
        if (
            item.get("all_checks_pass") is not True
            or not _all_true_checks(item.get("checks"))
            or set(item["checks"]) != COMPATIBILITY_ADMISSION_CHECK_FIELDS
        ):
            raise ValueError("{} compatibility checks did not all pass".format(name))
        for key in ("structural", "referential_labels"):
            binding = item.get(key)
            path = Path(binding.get("path", "")) if isinstance(binding, dict) else Path("")
            if not path.is_file() or binding.get("sha256") != sha256_file(path):
                raise ValueError("{} {} admission bytes differ".format(name, key))
    return build_content


def validate_gate_build_manifest_contract(manifest) -> dict:
    build_content = verify_build_manifest_binding(
        manifest.get("build_manifest"),
        require_checks=False,
        require_external_bytes=False,
    )
    if manifest.get("generation_contract") != canonical_generation_contract():
        raise ValueError("frozen gate generation contract differs")
    if build_content.get("generation_contract") != manifest["generation_contract"]:
        raise ValueError("frozen gate and build generation contracts differ")
    if build_content.get("tokenizer") != manifest.get("tokenizer"):
        raise ValueError("frozen gate tokenizer differs from embedded build content")
    inputs = build_content.get("inputs", [])
    r5_inputs = [
        record for record in inputs
        if isinstance(record, dict) and record.get("role") == "r5_fresh_board"
    ]
    if (
        len(r5_inputs) != 1
        or r5_inputs[0].get("sha256") != CANONICAL_R5_NOVELTY_BOARD_SHA256
    ):
        raise ValueError("frozen gate embedded build lacks the canonical R5 hash")
    outputs = build_content.get("outputs", {})
    boards = manifest.get("boards", {})
    for name, expected_seed in CANONICAL_GENERATOR_SEEDS.items():
        output = outputs.get(name, {})
        board = boards.get(name, {})
        if (
            output.get("seed") != expected_seed
            or board.get("generation_seed") != expected_seed
            or output.get("r5_novelty_board_sha256")
            != CANONICAL_R5_NOVELTY_BOARD_SHA256
            or board.get("r5_novelty_board_sha256")
            != CANONICAL_R5_NOVELTY_BOARD_SHA256
        ):
            raise ValueError("frozen {} custody binding differs".format(name))
        for key in ("path", "sha256", "rows", "regimes", "expected_cell_count", "rows_per_exact_cell"):
            if output.get(key) != board.get(key):
                raise ValueError("frozen {} build/board {} binding differs".format(name, key))
    return build_content


def validate_frozen_gate_manifest(
    manifest,
    *,
    repo_root=ROOT,
    expected_code_revision: str | None = None,
) -> None:
    expected_fields = {
        "manifest", "schema", "frozen_before_scores", "required_before_any_r10_score_run",
        "score_outputs_read", "score_artifacts", "board_gate_satisfied", "admission_report",
        "build_manifest", "tokenizer", "boards", "partitions", "calibration_threshold",
        "confirmation_thresholds", "implementations", "code_identity", "generation_contract",
        "claim_boundary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("frozen gate manifest fields differ from the frozen contract")
    reject_legacy_statistical_claims(manifest)
    if manifest.get("manifest") != FROZEN_GATE_MANIFEST or manifest.get("schema") != SCHEMA:
        raise ValueError("frozen gate manifest identity differs")
    if not (
        manifest.get("frozen_before_scores") is True
        and manifest.get("required_before_any_r10_score_run") is True
        and manifest.get("board_gate_satisfied") is True
        and manifest.get("score_outputs_read") is False
        and manifest.get("score_artifacts") == []
    ):
        raise ValueError("frozen gate is not a score-blind precondition")
    if manifest.get("partitions") != frozen_partition_contract():
        raise ValueError("frozen partition geometry or empirical quotas differ")
    if manifest.get("confirmation_thresholds") != frozen_confirmation_thresholds():
        raise ValueError("frozen confirmation empirical gate differs")
    validate_gate_build_manifest_contract(manifest)
    validate_code_identity(
        manifest.get("code_identity"),
        repo_root=repo_root,
        expected_revision=expected_code_revision,
    )
    identity_files = manifest["code_identity"]["files"]
    repo_root = Path(repo_root).resolve()
    implementations = manifest.get("implementations")
    expected_implementations = {
        "evaluator": {
            "identifier": EVALUATOR_IDENTIFIER,
            "path": str((repo_root / EVALUATOR_REPO_PATH).resolve()),
            "sha256": identity_files[EVALUATOR_REPO_PATH],
        },
        "extractor": {
            "identifier": EXTRACTOR_IDENTIFIER,
            "path": str((repo_root / EXTRACTOR_REPO_PATH).resolve()),
            "sha256": identity_files[EXTRACTOR_REPO_PATH],
            "expected_seed": EXPECTED_EXTRACTOR_SEED,
        },
        "expected_adapter_sha256": EXPECTED_ADAPTER_SHA256,
    }
    if implementations != expected_implementations:
        raise ValueError("frozen evaluator/extractor bindings differ")


def build_frozen_gate_manifest(
    *,
    report,
    admission_report_path,
    evaluator_path,
    extractor_path,
    code_revision,
    repo_root=ROOT,
) -> dict:
    code_revision = validate_git_revision(code_revision)
    code_identity = capture_clean_committed_code_identity(code_revision, repo_root)
    if report.get("audit") != ADMISSION_AUDIT:
        raise ValueError("cannot freeze an unknown admission report")
    if report.get("all_checks_pass") is not True:
        raise ValueError("cannot freeze a failed admission report")
    if report.get("r10_score_run_precondition_satisfied") is not True:
        raise ValueError("compatibility admissions are required before freezing gates")
    if not Path(admission_report_path).is_file():
        raise FileNotFoundError("missing gate input {}".format(admission_report_path))
    validate_admission_report_for_freeze(
        report,
        admission_report_path=admission_report_path,
        expected_code_identity=code_identity,
    )
    evaluator_relative = repo_implementation_path(
        evaluator_path, EVALUATOR_REPO_PATH, repo_root,
    )
    extractor_relative = repo_implementation_path(
        extractor_path, EXTRACTOR_REPO_PATH, repo_root,
    )
    evaluator_source = Path(evaluator_path).read_text(encoding="utf-8")
    extractor_source = Path(extractor_path).read_text(encoding="utf-8")
    evaluator_markers = (
        'AUDIT = "{}"'.format(EVALUATOR_IDENTIFIER),
        'GATE_ADMISSION_AUDIT = "{}"'.format(ADMISSION_AUDIT),
        'BOARD_SCHEMA = "{}"'.format(SCHEMA),
        'GATE_MANIFEST_BUILD = "r10_workspace_boards_v2"',
    )
    if not all(marker in evaluator_source for marker in evaluator_markers):
        raise ValueError("evaluator source does not declare the frozen v2 identifiers")
    extractor_markers = (
        'SCORE_AUDIT = "{}"'.format(EXTRACTOR_IDENTIFIER),
        '"audit": "{}"'.format(EXTRACTOR_IDENTIFIER),
    )
    if not any(marker in extractor_source for marker in extractor_markers):
        raise ValueError("extractor source does not declare the frozen score identifier")

    expected_geometry = {
        "calibration": {
            "rows": DEFAULT_BOARD_ROWS["calibration"],
            "rows_per_exact_cell": DEFAULT_CELL_ROWS["calibration"],
            "regimes": {"depth_ood": 400, "fit_iid": 400},
        },
        "confirmation": {
            "rows": DEFAULT_BOARD_ROWS["confirmation"],
            "rows_per_exact_cell": DEFAULT_CELL_ROWS["confirmation"],
            "regimes": {
                "full_ood": DEFAULT_CONFIRMATION_PARTITION_ROWS,
                "language_ood": DEFAULT_CONFIRMATION_PARTITION_ROWS,
            },
        },
    }
    for name, expected in expected_geometry.items():
        actual = report["boards"].get(name, {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("{} geometry is not the frozen default".format(name))
        if actual.get("expected_cell_count") != 80:
            raise ValueError("{} must have exactly 80 scheduled cells".format(name))
    confirmation_cells = report["boards"]["confirmation"].get("exact_cells", {})
    for partition in SPECS["confirmation"].regimes:
        partition_cells = {
            name: count
            for name, count in confirmation_cells.items()
            if name.startswith(partition + "|")
        }
        if (
            len(partition_cells) != CONFIRMATION_CELLS_PER_PARTITION
            or set(partition_cells.values()) != {DEFAULT_CELL_ROWS["confirmation"]}
        ):
            raise ValueError("{} exact-cell geometry differs".format(partition))

    admission_identity = report.get("code_identity")
    admission_identity_digest = report.get("code_identity_aggregate_sha256")
    if admission_identity is not None:
        validate_code_identity(
            admission_identity,
            repo_root=repo_root,
            expected_revision=code_revision,
        )
        if admission_identity != code_identity:
            raise ValueError("admission and gate code identities differ")
    if (
        admission_identity_digest is not None
        and admission_identity_digest != code_identity["aggregate_sha256"]
    ):
        raise ValueError("admission and gate code identity aggregates differ")
    compatibility = report["extractor_compatibility_admissions"]["boards"]
    boards = {
        name: {
            "path": report["boards"][name]["path"],
            "sha256": report["boards"][name]["sha256"],
            "rows": report["boards"][name]["rows"],
            "regimes": report["boards"][name]["regimes"],
            "expected_cell_count": report["boards"][name]["expected_cell_count"],
            "rows_per_exact_cell": report["boards"][name]["rows_per_exact_cell"],
            "generation_seed": report["boards"][name]["generation_seed"],
            "r5_novelty_board_sha256": (
                report["boards"][name]["r5_novelty_board_sha256"]
            ),
            "structural_admission": compatibility[name]["structural"],
            "referential_label_admission": compatibility[name]["referential_labels"],
        }
        for name in ("calibration", "confirmation")
    }
    manifest = {
        "manifest": FROZEN_GATE_MANIFEST,
        "schema": SCHEMA,
        "frozen_before_scores": True,
        "required_before_any_r10_score_run": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "board_gate_satisfied": True,
        "admission_report": {
            "audit": ADMISSION_AUDIT,
            "path": str(Path(admission_report_path).resolve()),
            "sha256": sha256_file(admission_report_path),
        },
        "build_manifest": {
            key: report["build_manifest"][key]
            for key in (
                "build", "path", "sha256", "byte_length", "bytes_base64", "content",
            )
        },
        "tokenizer": report["tokenizer"],
        "boards": boards,
        "partitions": frozen_partition_contract(),
        "calibration_threshold": {
            "unit": "program",
            "scope": "frozen calibration rows only",
            "partitions_pooled_only_for_calibration": ["fit_iid", "depth_ood"],
            "quantile": 0.97,
            "nonconformity": "max(-log(p_true)) across all true operations and the true query",
            "order_statistic": "ceil((n_programs + 1) * 0.97)",
            "threshold_count": 1,
            "confirmation_rows_must_not_influence_threshold": True,
            "post_score_tuning_forbidden": True,
            "extrapolation_beyond_frozen_board_forbidden": True,
        },
        "confirmation_thresholds": frozen_confirmation_thresholds(),
        "implementations": {
            "evaluator": {
                "identifier": EVALUATOR_IDENTIFIER,
                "path": str(Path(evaluator_path).resolve()),
                "sha256": code_identity["files"][evaluator_relative],
            },
            "extractor": {
                "identifier": EXTRACTOR_IDENTIFIER,
                "path": str(Path(extractor_path).resolve()),
                "sha256": code_identity["files"][extractor_relative],
                "expected_seed": EXPECTED_EXTRACTOR_SEED,
            },
            "expected_adapter_sha256": EXPECTED_ADAPTER_SHA256,
        },
        "code_identity": code_identity,
        "generation_contract": canonical_generation_contract(),
        "claim_boundary": (
            "This frozen manifest admits only hash-bound score-unseen inputs and fixes finite-board "
            "empirical gates. It contains no model score, makes no claim beyond the frozen rows, "
            "and forbids post-result gate changes."
        ),
    }
    validate_frozen_gate_manifest(
        manifest,
        repo_root=repo_root,
        expected_code_revision=code_revision,
    )
    validate_admission_report_for_freeze(
        report,
        admission_report_path=admission_report_path,
        expected_code_identity=code_identity,
    )
    require_unchanged_code_identity(
        code_identity, code_revision=code_revision, repo_root=repo_root,
    )
    return manifest


def exclusive_write(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as destination:
        destination.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", action="append", required=True)
    parser.add_argument("--r5-board", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--calibration-structural-admission", required=True)
    parser.add_argument("--calibration-label-admission", required=True)
    parser.add_argument("--confirmation-structural-admission", required=True)
    parser.add_argument("--confirmation-label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate-manifest-out", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--code-revision", required=True, type=validate_git_revision)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--minimum-calibration", type=int, default=800)
    parser.add_argument("--minimum-confirmation", type=int, default=1840)
    args = parser.parse_args()
    output_paths = [Path(args.out).resolve(), Path(args.gate_manifest_out).resolve()]
    if len(set(output_paths)) != len(output_paths):
        raise SystemExit("admission and gate manifest outputs must be distinct")
    for path in output_paths:
        if path.exists():
            raise SystemExit("refusing existing output {}".format(path))
    try:
        repo_implementation_path(args.evaluator, EVALUATOR_REPO_PATH)
        repo_implementation_path(args.extractor, EXTRACTOR_REPO_PATH)
        def admission_work():
            return audit_bundle(
                training_data=args.training_data,
                r5_board=args.r5_board,
                tokenizer_path=args.tokenizer,
                calibration_path=args.calibration,
                confirmation_path=args.confirmation,
                build_manifest_path=args.build_manifest,
                max_tokens=args.max_tokens,
                minimum_calibration=args.minimum_calibration,
                minimum_confirmation=args.minimum_confirmation,
                compatibility_admissions={
                    "calibration": {
                        "structural": args.calibration_structural_admission,
                        "label": args.calibration_label_admission,
                    },
                    "confirmation": {
                        "structural": args.confirmation_structural_admission,
                        "label": args.confirmation_label_admission,
                    },
                },
            )

        report, frozen_code_identity = run_admission_with_code_custody(
            code_revision=args.code_revision,
            admission_work=admission_work,
        )
        report["code_identity"] = frozen_code_identity
        report["code_identity_aggregate_sha256"] = (
            frozen_code_identity["aggregate_sha256"]
        )
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        exclusive_write(args.out, report_bytes)
        if not report["all_checks_pass"]:
            raise SystemExit(1)
        gate_manifest = build_frozen_gate_manifest(
            report=report,
            admission_report_path=args.out,
            evaluator_path=args.evaluator,
            extractor_path=args.extractor,
            code_revision=args.code_revision,
        )
        gate_bytes = (json.dumps(gate_manifest, indent=2, sort_keys=True) + "\n").encode()
        exclusive_write(args.gate_manifest_out, gate_bytes)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "audit": report["audit"],
        "all_checks_pass": report["all_checks_pass"],
        "r10_score_run_precondition_satisfied": report["r10_score_run_precondition_satisfied"],
        "build_manifest_sha256": report["build_manifest"]["sha256"],
        "admission_report_sha256": sha256_file(args.out),
        "frozen_gate_manifest": gate_manifest["manifest"],
        "frozen_gate_manifest_sha256": sha256_file(args.gate_manifest_out),
        "git_revision": gate_manifest["code_identity"]["git_revision"],
        "code_identity_aggregate_sha256": (
            gate_manifest["code_identity"]["aggregate_sha256"]
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
