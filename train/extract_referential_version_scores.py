#!/usr/bin/env python3
"""Extract provenance-bound R10 categorical scores from frozen R9c no-syndrome."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import stat
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import tokenizers
import torch

from categorical_microcode import OPCODES, QUERIES, sha256_file


R9C_PROTOCOL = "referential_bidirectional_syndrome_microcode_r9c"
POINTER_PROTOCOL = "causal_microcode_referential_slots_v4"
STRUCTURAL_ADMISSION_AUDIT = "role_equivariant_microcode_v3"
LABEL_ADMISSION_AUDIT = "referential_slot_label_admission_v1"
SCORE_AUDIT = "referential_version_scores_r10"
SCORE_SCHEMA_VERSION = 2
EVALUATOR_AUDIT = "referential_version_space_workspace_confirmation_r10"
BOARD_SCHEMA = "r10_workspace_board_v2"
GATE_MANIFEST_BUILD = "r10_workspace_boards_v2"
GATE_ADMISSION_AUDIT = "r10_workspace_boards_independent_admission_v2"
FROZEN_GATE_MANIFEST = "r10_workspace_frozen_score_gate_v2"
BOARD_NAMES = ("calibration", "confirmation")
FROZEN_BATCH_SIZE = 16
FROZEN_SEED = 20260714
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
EXPECTED_DEVICE_NAME = "NVIDIA H100 PCIe"
EXPECTED_DEVICE_CAPABILITY = (9, 0)
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
NO_SYNDROME_CONFIG = {
    "conditioning": "directional",
    "use_syndrome": False,
    "shuffle_goal": False,
}


@dataclass(frozen=True)
class PreflightBundle:
    board_name: str
    seed: int
    repo_root: str
    paths: dict[str, str]
    hashes: dict[str, str]
    gate_manifest: dict
    gate_admission: dict
    structural_admission: dict
    label_admission: dict
    code_identity: dict


@dataclass
class FrozenArtifact:
    """A path-bound descriptor retained through decision publication."""

    name: str
    path: str
    sha256: str
    descriptor: int
    identity: tuple[int, int, int, int, int]
    raw: bytes | None = None

    @classmethod
    def open(
        cls,
        name,
        path,
        *,
        expected_sha256=None,
        capture_bytes=False,
        canonical=False,
    ):
        if expected_sha256 is not None:
            _require(_is_sha256(expected_sha256), "{} hash is invalid".format(name))
        real_path = _real_input_path(path, name, canonical=canonical)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(real_path, flags)
        except OSError as error:
            raise SystemExit("cannot open {}: {}".format(name, error)) from error
        try:
            opened = os.fstat(descriptor)
            _require(
                stat.S_ISREG(opened.st_mode) and opened.st_size > 0,
                "{} must be a non-empty regular file".format(name),
            )
            linked = os.stat(real_path, follow_symlinks=False)
            _require(
                (opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino),
                "{} changed while it was opened".format(name),
            )
            digest = hashlib.sha256()
            chunks = [] if capture_bytes else None
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            actual_sha256 = digest.hexdigest()
            _require(
                expected_sha256 is None or actual_sha256 == expected_sha256,
                "{} hash mismatch".format(name),
            )
            closed_over = os.fstat(descriptor)
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            _require(
                (
                    closed_over.st_dev,
                    closed_over.st_ino,
                    closed_over.st_size,
                    closed_over.st_mtime_ns,
                    closed_over.st_ctime_ns,
                )
                == identity,
                "{} changed while it was hashed".format(name),
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            return cls(
                name=name,
                path=real_path,
                sha256=actual_sha256,
                descriptor=descriptor,
                identity=identity,
                raw=b"".join(chunks) if chunks is not None else None,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def reader(self):
        duplicate = os.dup(self.descriptor)
        os.lseek(duplicate, 0, os.SEEK_SET)
        return os.fdopen(duplicate, "rb")

    def json_object(self):
        _require(self.raw is not None, "{} was not captured as bytes".format(self.name))
        try:
            payload = json.loads(self.raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit("invalid {}: {}".format(self.name, error)) from error
        _require(isinstance(payload, dict), "{} must be a JSON object".format(self.name))
        return payload

    def recheck(self):
        opened = os.fstat(self.descriptor)
        current_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(
            current_identity == self.identity,
            "frozen input changed before publication: {}".format(self.name),
        )
        try:
            linked = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise SystemExit(
                "frozen input path disappeared before publication: {}".format(self.name)
            ) from error
        _require(
            (linked.st_dev, linked.st_ino) == self.identity[:2],
            "frozen input path was swapped before publication: {}".format(self.name),
        )
        if self.raw is None:
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while True:
                chunk = os.read(self.descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            _require(
                digest.hexdigest() == self.sha256,
                "frozen input bytes changed before publication: {}".format(self.name),
            )
            os.lseek(self.descriptor, 0, os.SEEK_SET)

    def close(self):
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass
class ChainPreflightBundle:
    repo_root: str
    artifacts: dict[str, FrozenArtifact]
    gate_manifest: dict
    gate_admission: dict
    gate_bundle: object
    code_identity: dict
    board_bindings: dict[str, dict[str, str]]
    structural_admissions: dict[str, dict]
    label_admissions: dict[str, dict]
    chain_id: str
    output_namespace: str

    def recheck(self):
        for artifact in self.artifacts.values():
            artifact.recheck()

    def close(self):
        for artifact in self.artifacts.values():
            artifact.close()


def categorical_probabilities(forward_logits, backward_logits, query_logits):
    """Return float32 probabilities for one fixed-replay batch."""
    if forward_logits.ndim != 3 or backward_logits.shape != forward_logits.shape:
        raise ValueError(
            "directional logits must share [batch,events,categories] shape"
        )
    if forward_logits.shape[-1] != len(OPCODES):
        raise ValueError("directional logits have the wrong categorical width")
    if query_logits.ndim != 2 or query_logits.shape != (
        forward_logits.shape[0],
        len(QUERIES),
    ):
        raise ValueError("query logits must have shape [batch,queries]")
    tensors = (forward_logits, backward_logits, query_logits)
    if not all(torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("all logits must be floating point")
    if not all(tensor.device == forward_logits.device for tensor in tensors):
        raise ValueError("all logits must share a device")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("all logits must be finite")

    forward = forward_logits.float()
    backward = backward_logits.float()
    query = query_logits.float()
    return {
        "joint": (0.5 * (forward + backward)).softmax(dim=-1),
        "forward": forward.softmax(dim=-1),
        "backward": backward.softmax(dim=-1),
        "query": query.softmax(dim=-1),
    }


def _require(condition, message):
    if not condition:
        raise SystemExit(message)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_code_identity(
    code_revision, expected_extractor_sha256, actual_extractor_sha256
):
    _require(
        _is_git_revision(code_revision),
        "code revision must be a full lowercase git SHA",
    )
    _require(
        _is_sha256(expected_extractor_sha256), "expected extractor hash must be SHA-256"
    )
    _require(
        _is_sha256(actual_extractor_sha256), "actual extractor hash must be SHA-256"
    )
    _require(
        expected_extractor_sha256 == actual_extractor_sha256,
        "executing extractor differs from the frozen extractor hash",
    )


def canonical_json_bytes(payload):
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def code_identity_aggregate_sha256(git_revision, files, runtime):
    """Hash the complete committed source, revision, and runtime identity."""
    _require(_is_git_revision(git_revision), "aggregate git revision is invalid")
    _require(
        isinstance(runtime, dict) and set(runtime) == {"python", "torch", "tokenizers"},
        "aggregate runtime schema changed",
    )
    return hashlib.sha256(
        canonical_json_bytes(
            {"git_revision": git_revision, "files": files, "runtime": runtime}
        )
    ).hexdigest()


def current_runtime_identity():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "tokenizers": str(tokenizers.__version__),
    }


def _real_input_path(path, description, *, canonical=False):
    _require(
        isinstance(path, (str, os.PathLike)),
        "{} path must be path-like".format(description),
    )
    raw_path = os.fspath(path)
    _require(isinstance(raw_path, str), "{} path must be text".format(description))
    real_path = os.path.realpath(raw_path)
    if canonical:
        _require(
            os.path.abspath(raw_path) == real_path,
            "{} path must be absolute and canonical".format(description),
        )
    _require(
        Path(real_path).is_file() and Path(real_path).stat().st_size > 0,
        "missing {}: {}".format(description, real_path),
    )
    return real_path


def _hash_bound_file(path, expected_sha256, description, *, canonical=False):
    _require(_is_sha256(expected_sha256), "{} hash is invalid".format(description))
    real_path = _real_input_path(path, description, canonical=canonical)
    actual_sha256 = sha256_file(real_path)
    _require(actual_sha256 == expected_sha256, "{} hash mismatch".format(description))
    return real_path, actual_sha256


def _load_hash_bound_json(path, expected_sha256, description, *, canonical=False):
    _require(_is_sha256(expected_sha256), "{} hash is invalid".format(description))
    real_path = _real_input_path(path, description, canonical=canonical)
    raw = Path(real_path).read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "{} hash mismatch".format(description),
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            "{} is not valid JSON: {}".format(description, error)
        ) from error
    _require(isinstance(payload, dict), "{} must be a JSON object".format(description))
    return real_path, expected_sha256, payload


def _discover_repo_context(extractor_path, repo_root=None):
    fallback = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(extractor_path).resolve().parents[1]
    )
    command = ["git", "-C", str(fallback), "rev-parse", "--show-toplevel", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        _require(
            not (fallback / ".git").exists(),
            "git metadata exists but HEAD could not be read: {}".format(error),
        )
        return fallback, None
    if completed.returncode != 0:
        _require(
            not (fallback / ".git").exists(),
            "git metadata exists but HEAD could not be read: {}".format(
                completed.stderr.strip()
            ),
        )
        return fallback, None
    lines = completed.stdout.splitlines()
    _require(len(lines) == 2, "git returned an invalid repository identity")
    discovered_root = Path(lines[0]).resolve()
    git_head = lines[1].strip()
    _require(_is_git_revision(git_head), "git returned an invalid HEAD revision")
    return discovered_root, git_head


def _committed_blob_sha256(repo_root, revision, relative_path):
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                "{}:{}".format(revision, relative_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "cannot read committed source {}: {}".format(relative_path, error)
        ) from error
    return hashlib.sha256(completed.stdout).hexdigest()


def validate_manifest_code_identity(
    code_identity,
    *,
    code_revision,
    repo_root,
    git_head,
    source_paths,
    source_hashes,
    runtime_identity=None,
    committed_source_hashes=None,
):
    """Validate the manifest's complete source/runtime identity before model use."""
    _require(isinstance(code_identity, dict), "gate manifest lacks code_identity")
    _require(
        set(code_identity) == {"git_revision", "files", "aggregate_sha256", "runtime"},
        "gate code_identity fields changed",
    )
    _require(
        code_identity.get("git_revision") == code_revision,
        "gate code revision mismatch",
    )
    _require(
        _is_git_revision(code_revision),
        "code revision must be a full lowercase git SHA",
    )
    if git_head is not None:
        _require(
            git_head == code_revision,
            "actual git HEAD differs from the frozen revision",
        )

    files = code_identity.get("files")
    _require(isinstance(files, dict) and files, "gate code_identity lacks source files")
    _require(
        REQUIRED_CODE_IDENTITY_FILES.issubset(files),
        "gate code_identity omits required R10 source or job files",
    )
    root = Path(repo_root).resolve()
    code_paths = {}
    code_hashes = {}
    for relative_path, expected_sha256 in files.items():
        _require(
            isinstance(relative_path, str) and relative_path,
            "code_identity file path is invalid",
        )
        pure_path = PurePosixPath(relative_path)
        _require(
            not pure_path.is_absolute()
            and pure_path.as_posix() == relative_path
            and all(part not in ("", ".", "..") for part in pure_path.parts),
            "code_identity paths must be canonical repo-relative paths",
        )
        _require(
            _is_sha256(expected_sha256),
            "code_identity hash is invalid for {}".format(relative_path),
        )
        source = root.joinpath(*pure_path.parts)
        source_path = Path(os.path.realpath(source))
        _require(
            source.absolute() == source_path,
            "code_identity source is not canonical: {}".format(relative_path),
        )
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise SystemExit(
                "code_identity path escapes the repository: {}".format(relative_path)
            ) from error
        _require(
            source_path.is_file() and source_path.stat().st_size > 0,
            "missing code_identity source: {}".format(relative_path),
        )
        actual_sha256 = sha256_file(source_path)
        _require(
            actual_sha256 == expected_sha256,
            "code_identity source hash mismatch: {}".format(relative_path),
        )
        committed_sha256 = (
            committed_source_hashes.get(relative_path)
            if committed_source_hashes is not None
            else _committed_blob_sha256(root, code_revision, relative_path)
        )
        _require(
            committed_sha256 == expected_sha256,
            "code_identity source is not the committed blob: {}".format(
                relative_path
            ),
        )
        key = "code_identity:{}".format(relative_path)
        code_paths[key] = str(source_path)
        code_hashes[key] = actual_sha256

    for source_name in ("extractor", "evaluator"):
        try:
            relative_path = (
                Path(source_paths[source_name]).resolve().relative_to(root).as_posix()
            )
        except ValueError as error:
            raise SystemExit(
                "{} source is outside the code_identity repository".format(source_name)
            ) from error
        _require(
            files.get(relative_path) == source_hashes[source_name],
            "code_identity does not bind the exact {} source".format(source_name),
        )

    runtime = code_identity.get("runtime")
    _require(
        isinstance(runtime, dict)
        and set(runtime) == {"python", "torch", "tokenizers"}
        and all(isinstance(value, str) and value for value in runtime.values()),
        "gate code_identity runtime fields changed",
    )
    actual_runtime = runtime_identity or current_runtime_identity()
    _require(
        actual_runtime == runtime,
        "runtime differs from gate code_identity",
    )
    aggregate_sha256 = code_identity.get("aggregate_sha256")
    _require(_is_sha256(aggregate_sha256), "code_identity aggregate hash is invalid")
    _require(
        aggregate_sha256
        == code_identity_aggregate_sha256(code_revision, files, runtime),
        "code_identity aggregate hash mismatch",
    )
    return copy.deepcopy(code_identity), code_paths, code_hashes


def _binding_sha256(binding, description):
    _require(isinstance(binding, dict), "{} binding is missing".format(description))
    value = binding.get("sha256")
    _require(_is_sha256(value), "{} binding hash is invalid".format(description))
    return value


def validate_gate_bindings(args, paths, hashes, manifest, admission):
    """Validate both frozen gate documents and the requested board hash chain."""
    _require(
        manifest.get("manifest") == FROZEN_GATE_MANIFEST, "invalid frozen gate manifest"
    )
    _require(manifest.get("schema") == BOARD_SCHEMA, "invalid gate board schema")
    _require(
        manifest.get("frozen_before_scores") is True
        and manifest.get("required_before_any_r10_score_run") is True
        and manifest.get("board_gate_satisfied") is True
        and manifest.get("score_outputs_read") is False
        and manifest.get("score_artifacts") == [],
        "frozen gate is not a score-blind extraction precondition",
    )

    admission_binding = manifest.get("admission_report")
    _require(
        isinstance(admission_binding, dict)
        and admission_binding.get("audit") == GATE_ADMISSION_AUDIT
        and admission_binding.get("path") == paths["gate_admission"]
        and admission_binding.get("sha256") == hashes["gate_admission"],
        "frozen gate does not bind the exact independent admission",
    )
    identity = manifest.get("code_identity")
    admission_identity = admission.get("code_identity")
    admission_identity_sha256 = admission.get("code_identity_aggregate_sha256")
    _require(
        admission_identity is None or admission_identity == identity,
        "gate admission code_identity differs from the frozen gate",
    )
    _require(
        admission_identity_sha256 is None
        or (
            isinstance(identity, dict)
            and admission_identity_sha256 == identity.get("aggregate_sha256")
        ),
        "gate admission code identity aggregate mismatch",
    )
    frozen_build = manifest.get("build_manifest")
    admitted_build = admission.get("build_manifest")
    _require(
        isinstance(frozen_build, dict)
        and frozen_build.get("build") == GATE_MANIFEST_BUILD
        and isinstance(frozen_build.get("path"), str)
        and _is_sha256(frozen_build.get("sha256")),
        "frozen gate does not bind its build manifest",
    )
    _require(
        isinstance(admitted_build, dict)
        and admitted_build.get("path") == frozen_build.get("path")
        and admitted_build.get("sha256") == frozen_build.get("sha256")
        and admitted_build.get("all_checks_pass") is True,
        "independent admission does not bind the frozen build manifest",
    )

    _require(
        admission.get("audit") == GATE_ADMISSION_AUDIT, "invalid gate admission audit"
    )
    _require(admission.get("schema") == BOARD_SCHEMA, "invalid gate admission schema")
    _require(
        admission.get("cpu_only") is True
        and admission.get("score_outputs_read") is False
        and admission.get("score_artifacts") == []
        and admission.get("all_checks_pass") is True
        and admission.get("r10_score_run_precondition_satisfied") is True,
        "independent gate admission does not authorize score extraction",
    )

    gate_boards = manifest.get("boards")
    admitted_boards = admission.get("boards")
    compatibility = admission.get("extractor_compatibility_admissions")
    _require(isinstance(gate_boards, dict), "frozen gate lacks board bindings")
    _require(isinstance(admitted_boards, dict), "gate admission lacks board bindings")
    _require(
        isinstance(compatibility, dict)
        and compatibility.get("enabled") is True
        and compatibility.get("all_checks_pass") is True
        and isinstance(compatibility.get("boards"), dict),
        "gate admission lacks passing extractor compatibility admissions",
    )
    for board_name in BOARD_NAMES:
        gate_board = gate_boards.get(board_name)
        admitted_board = admitted_boards.get(board_name)
        compatible_board = compatibility["boards"].get(board_name)
        _require(
            isinstance(gate_board, dict),
            "frozen gate lacks {} board".format(board_name),
        )
        _require(
            isinstance(admitted_board, dict)
            and admitted_board.get("all_checks_pass") is True,
            "gate admission rejects {} board".format(board_name),
        )
        _require(
            isinstance(compatible_board, dict)
            and compatible_board.get("all_checks_pass") is True,
            "extractor compatibility rejects {} board".format(board_name),
        )
        gate_board_sha256 = gate_board.get("sha256")
        _require(
            _is_sha256(gate_board_sha256),
            "{} gate board hash is invalid".format(board_name),
        )
        _require(
            admitted_board.get("sha256") == gate_board_sha256,
            "gate and admission disagree on {} board".format(board_name),
        )
        structural_sha256 = _binding_sha256(
            gate_board.get("structural_admission"),
            "{} structural admission".format(board_name),
        )
        label_sha256 = _binding_sha256(
            gate_board.get("referential_label_admission"),
            "{} referential-label admission".format(board_name),
        )
        _require(
            _binding_sha256(
                compatible_board.get("structural"),
                "{} compatible structural admission".format(board_name),
            )
            == structural_sha256,
            "gate and compatibility admission disagree on {} structural hash".format(
                board_name
            ),
        )
        _require(
            _binding_sha256(
                compatible_board.get("referential_labels"),
                "{} compatible label admission".format(board_name),
            )
            == label_sha256,
            "gate and compatibility admission disagree on {} label hash".format(
                board_name
            ),
        )
        if board_name == args.board_name:
            _require(
                gate_board_sha256 == hashes["data"],
                "frozen gate does not bind the requested board",
            )
            _require(
                structural_sha256 == hashes["structural_admission"],
                "frozen gate does not bind the requested structural admission",
            )
            _require(
                label_sha256 == hashes["referential_label_admission"],
                "frozen gate does not bind the requested label admission",
            )

    implementations = manifest.get("implementations")
    _require(isinstance(implementations, dict), "frozen gate lacks implementations")
    evaluator = implementations.get("evaluator")
    extractor = implementations.get("extractor")
    _require(
        isinstance(evaluator, dict)
        and evaluator.get("identifier") == EVALUATOR_AUDIT
        and evaluator.get("path") == paths["evaluator"]
        and evaluator.get("sha256") == hashes["evaluator"],
        "frozen gate evaluator binding mismatch",
    )
    _require(
        isinstance(extractor, dict)
        and extractor.get("identifier") == SCORE_AUDIT
        and extractor.get("path") == paths["extractor"]
        and extractor.get("sha256") == hashes["extractor"]
        and extractor.get("expected_seed") == args.seed,
        "frozen gate extractor binding mismatch",
    )
    _require(
        implementations.get("expected_adapter_sha256") == hashes["adapter"],
        "frozen gate adapter binding mismatch",
    )


def validate_board_admissions(hashes, admission, label_admission):
    """Validate board-facing admission content without opening a checkpoint."""
    _require(
        admission.get("audit") == STRUCTURAL_ADMISSION_AUDIT,
        "invalid structural admission audit",
    )
    _require(admission.get("all_checks_pass") is True, "structural admission failed")
    _require(
        admission.get("eval_sha256") == hashes["data"],
        "structural admission does not bind the evaluation JSONL",
    )
    _require(
        admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "structural admission does not bind the tokenizer",
    )
    _require(
        _is_sha256(admission.get("train_sha256")),
        "structural admission training hash is invalid",
    )
    _require(
        label_admission.get("audit") == LABEL_ADMISSION_AUDIT,
        "invalid referential-label admission audit",
    )
    _require(
        label_admission.get("all_checks_pass") is True,
        "referential-label admission failed",
    )
    _require(
        label_admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "referential-label admission does not bind the tokenizer",
    )
    datasets = label_admission.get("datasets")
    _require(isinstance(datasets, dict), "referential-label admission lacks datasets")
    evaluation = datasets.get("eval")
    training = datasets.get("train")
    _require(
        isinstance(evaluation, dict)
        and evaluation.get("all_checks_pass") is True
        and evaluation.get("sha256") == hashes["data"],
        "referential evaluation labels do not bind the admitted board",
    )
    _require(
        isinstance(training, dict)
        and training.get("all_checks_pass") is True
        and _is_sha256(training.get("sha256")),
        "referential training labels were not admitted",
    )
    _require(
        training.get("sha256") == admission.get("train_sha256"),
        "structural and referential-label admissions disagree on training data",
    )


def validate_preflight(
    args,
    *,
    extractor_path=__file__,
    repo_root=None,
    runtime_identity=None,
):
    """Perform every gate, source, runtime, and artifact check before CUDA."""
    _require(
        args.board_name in BOARD_NAMES, "board name must be calibration or confirmation"
    )
    _require(
        isinstance(args.seed, int) and not isinstance(args.seed, bool),
        "seed must be an integer",
    )
    for description, value in (
        ("adapter", args.adapter_sha256),
        ("board", args.data_sha256),
        ("structural admission", args.admission_sha256),
        ("referential-label admission", args.label_admission_sha256),
        ("gate manifest", args.gate_manifest_sha256),
        ("gate admission", args.gate_admission_sha256),
        ("evaluator", args.evaluator_sha256),
        ("extractor", args.extractor_sha256),
    ):
        _require(_is_sha256(value), "expected {} hash is invalid".format(description))

    paths = {}
    hashes = {}
    extractor_real, extractor_sha256 = _hash_bound_file(
        Path(extractor_path).resolve(),
        args.extractor_sha256,
        "extractor",
        canonical=True,
    )
    paths["extractor"] = extractor_real
    hashes["extractor"] = extractor_sha256
    validate_code_identity(args.code_revision, args.extractor_sha256, extractor_sha256)

    gate_real, gate_sha256, gate_manifest = _load_hash_bound_json(
        args.gate_manifest,
        args.gate_manifest_sha256,
        "gate manifest",
        canonical=True,
    )
    paths["gate_manifest"] = gate_real
    hashes["gate_manifest"] = gate_sha256
    admission_real, admission_sha256, gate_admission = _load_hash_bound_json(
        args.gate_admission,
        args.gate_admission_sha256,
        "gate admission",
        canonical=True,
    )
    paths["gate_admission"] = admission_real
    hashes["gate_admission"] = admission_sha256
    evaluator_real, evaluator_sha256 = _hash_bound_file(
        args.evaluator,
        args.evaluator_sha256,
        "evaluator",
        canonical=True,
    )
    paths["evaluator"] = evaluator_real
    hashes["evaluator"] = evaluator_sha256

    json_inputs = (
        (
            "structural_admission",
            args.admission,
            args.admission_sha256,
            "structural admission",
        ),
        (
            "referential_label_admission",
            args.label_admission,
            args.label_admission_sha256,
            "referential-label admission",
        ),
    )
    loaded_json = {}
    for name, path, expected_sha256, description in json_inputs:
        real_path, actual_sha256, payload = _load_hash_bound_json(
            path, expected_sha256, description
        )
        paths[name] = real_path
        hashes[name] = actual_sha256
        loaded_json[name] = payload

    bound_files = (
        ("data", args.data, args.data_sha256, "board data"),
        ("adapter", args.adapter, args.adapter_sha256, "adapter"),
    )
    for name, path, expected_sha256, description in bound_files:
        real_path, actual_sha256 = _hash_bound_file(path, expected_sha256, description)
        paths[name] = real_path
        hashes[name] = actual_sha256

    tokenizer_real = _real_input_path(args.tokenizer, "tokenizer")
    paths["tokenizer"] = tokenizer_real
    hashes["tokenizer"] = sha256_file(tokenizer_real)

    discovered_root, git_head = _discover_repo_context(extractor_real, repo_root)
    code_identity, code_paths, code_hashes = validate_manifest_code_identity(
        gate_manifest.get("code_identity"),
        code_revision=args.code_revision,
        repo_root=discovered_root,
        git_head=git_head,
        source_paths=paths,
        source_hashes=hashes,
        runtime_identity=runtime_identity,
    )
    paths.update(code_paths)
    hashes.update(code_hashes)
    validate_gate_bindings(args, paths, hashes, gate_manifest, gate_admission)
    validate_board_admissions(
        hashes,
        loaded_json["structural_admission"],
        loaded_json["referential_label_admission"],
    )

    for name, path, description in (
        ("base", args.base, "base checkpoint"),
        ("pointer_adapter", args.pointer_adapter, "pointer adapter"),
    ):
        real_path = _real_input_path(path, description)
        paths[name] = real_path
        hashes[name] = sha256_file(real_path)

    return PreflightBundle(
        board_name=args.board_name,
        seed=int(args.seed),
        repo_root=str(discovered_root),
        paths=paths,
        hashes=hashes,
        gate_manifest=gate_manifest,
        gate_admission=gate_admission,
        structural_admission=loaded_json["structural_admission"],
        label_admission=loaded_json["referential_label_admission"],
        code_identity=code_identity,
    )


def _chain_board_arg(args, board_name, field):
    return getattr(args, "{}_{}".format(board_name, field))


def _load_generator_from_frozen_bytes(artifact):
    _require(artifact.raw is not None, "generator source was not captured")
    module_name = "_r10_frozen_generator_{}".format(artifact.sha256[:16])
    module = types.ModuleType(module_name)
    module.__file__ = artifact.path
    sys.modules[module_name] = module
    try:
        code = compile(artifact.raw, artifact.path, "exec")
        exec(code, module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _validate_frozen_board_bytes(artifacts, code_identity):
    generator_relative = "pipeline/generate_r10_workspace_boards.py"
    generator_artifact = artifacts["code_identity:{}".format(generator_relative)]
    _require(
        code_identity["files"].get(generator_relative) == generator_artifact.sha256,
        "frozen generator source is not code-identity bound",
    )
    generator = _load_generator_from_frozen_bytes(generator_artifact)
    for board_name, count in (("calibration", 800), ("confirmation", 1840)):
        expected = generator.serialize_jsonl(
            generator.build_board(
                board_name,
                count,
                generator.CANONICAL_GENERATOR_SEEDS[board_name],
            )
        )
        actual = artifacts["{}_data".format(board_name)].raw
        _require(
            actual == expected,
            "{} board bytes are not the committed deterministic build".format(
                board_name
            ),
        )


def validate_chain_preflight(
    args,
    *,
    extractor_path=__file__,
    repo_root=None,
    runtime_identity=None,
    committed_source_hashes=None,
    regenerate_boards=True,
):
    """Freeze and substantively validate the complete two-board chain."""
    _require(
        getattr(args, "batch_size", FROZEN_BATCH_SIZE) == FROZEN_BATCH_SIZE,
        "R10 batch size is frozen at {}".format(FROZEN_BATCH_SIZE),
    )
    _require(
        getattr(args, "seed", FROZEN_SEED) == FROZEN_SEED,
        "R10 extractor seed is frozen at {}".format(FROZEN_SEED),
    )
    expected_hashes = {
        "adapter": args.adapter_sha256,
        "gate_manifest": args.gate_manifest_sha256,
        "gate_admission": args.gate_admission_sha256,
        "evaluator": args.evaluator_sha256,
        "extractor": args.extractor_sha256,
    }
    board_bindings = {}
    for board_name in FROZEN_BOARD_ORDER:
        board_bindings[board_name] = {
            "data_path": os.path.realpath(_chain_board_arg(args, board_name, "data")),
            "data_sha256": _chain_board_arg(args, board_name, "data_sha256"),
            "structural_admission_sha256": _chain_board_arg(
                args, board_name, "admission_sha256"
            ),
            "label_admission_sha256": _chain_board_arg(
                args, board_name, "label_admission_sha256"
            ),
        }
        expected_hashes["{}_data".format(board_name)] = board_bindings[board_name][
            "data_sha256"
        ]
        expected_hashes["{}_structural_admission".format(board_name)] = (
            board_bindings[board_name]["structural_admission_sha256"]
        )
        expected_hashes["{}_label_admission".format(board_name)] = board_bindings[
            board_name
        ]["label_admission_sha256"]
    for description, value in expected_hashes.items():
        _require(_is_sha256(value), "expected {} hash is invalid".format(description))

    artifacts = {}
    try:
        artifacts["extractor"] = FrozenArtifact.open(
            "extractor",
            Path(extractor_path).resolve(),
            expected_sha256=args.extractor_sha256,
            capture_bytes=True,
            canonical=True,
        )
        artifacts["evaluator"] = FrozenArtifact.open(
            "evaluator",
            args.evaluator,
            expected_sha256=args.evaluator_sha256,
            capture_bytes=True,
            canonical=True,
        )
        artifacts["gate_manifest"] = FrozenArtifact.open(
            "gate manifest",
            args.gate_manifest,
            expected_sha256=args.gate_manifest_sha256,
            capture_bytes=True,
            canonical=True,
        )
        artifacts["gate_admission"] = FrozenArtifact.open(
            "gate admission",
            args.gate_admission,
            expected_sha256=args.gate_admission_sha256,
            capture_bytes=True,
            canonical=True,
        )
        artifacts["adapter"] = FrozenArtifact.open(
            "adapter",
            args.adapter,
            expected_sha256=args.adapter_sha256,
            capture_bytes=False,
        )
        artifacts["tokenizer"] = FrozenArtifact.open(
            "tokenizer", args.tokenizer, capture_bytes=True
        )
        for board_name in FROZEN_BOARD_ORDER:
            artifacts["{}_data".format(board_name)] = FrozenArtifact.open(
                "{} board".format(board_name),
                _chain_board_arg(args, board_name, "data"),
                expected_sha256=board_bindings[board_name]["data_sha256"],
                capture_bytes=True,
            )
            artifacts[
                "{}_structural_admission".format(board_name)
            ] = FrozenArtifact.open(
                "{} structural admission".format(board_name),
                _chain_board_arg(args, board_name, "admission"),
                expected_sha256=board_bindings[board_name][
                    "structural_admission_sha256"
                ],
                capture_bytes=True,
            )
            artifacts["{}_label_admission".format(board_name)] = FrozenArtifact.open(
                "{} referential-label admission".format(board_name),
                _chain_board_arg(args, board_name, "label_admission"),
                expected_sha256=board_bindings[board_name]["label_admission_sha256"],
                capture_bytes=True,
            )

        gate_manifest = artifacts["gate_manifest"].json_object()
        gate_admission = artifacts["gate_admission"].json_object()
        frozen_build = gate_manifest.get("build_manifest")
        _require(
            isinstance(frozen_build, dict)
            and frozen_build.get("build") == GATE_MANIFEST_BUILD
            and isinstance(frozen_build.get("path"), str)
            and _is_sha256(frozen_build.get("sha256")),
            "frozen gate does not bind a build manifest",
        )
        artifacts["build_manifest"] = FrozenArtifact.open(
            "build manifest",
            frozen_build["path"],
            expected_sha256=frozen_build["sha256"],
            capture_bytes=True,
            canonical=True,
        )

        discovered_root, git_head = _discover_repo_context(extractor_path, repo_root)
        _require(git_head == args.code_revision, "live git revision changed")
        identity = gate_manifest.get("code_identity")
        _require(isinstance(identity, dict), "gate lacks code_identity")
        _require(
            set(identity) == {"git_revision", "files", "aggregate_sha256", "runtime"},
            "gate code_identity fields changed",
        )
        _require(identity.get("git_revision") == args.code_revision, "revision mismatch")
        files = identity.get("files")
        _require(
            isinstance(files, dict) and REQUIRED_CODE_IDENTITY_FILES.issubset(files),
            "gate code_identity omits required R10 source or job files",
        )
        trusted_source_hashes = {
            artifacts["extractor"].path: artifacts["extractor"].sha256,
            artifacts["evaluator"].path: artifacts["evaluator"].sha256,
        }
        committed_hashes = {}
        for relative_path, expected_sha256 in files.items():
            _require(
                isinstance(relative_path, str)
                and relative_path
                and _is_sha256(expected_sha256),
                "code_identity source binding is invalid",
            )
            source_path = str((discovered_root / PurePosixPath(relative_path)).resolve())
            existing = next(
                (
                    item
                    for item in artifacts.values()
                    if item.path == source_path and item.sha256 == expected_sha256
                ),
                None,
            )
            if existing is None:
                key = "code_identity:{}".format(relative_path)
                artifacts[key] = FrozenArtifact.open(
                    key,
                    source_path,
                    expected_sha256=expected_sha256,
                    capture_bytes=True,
                    canonical=True,
                )
                existing = artifacts[key]
            else:
                artifacts[
                    "code_identity:{}".format(relative_path)
                ] = existing
            trusted_source_hashes[source_path] = existing.sha256
            committed_hashes[relative_path] = (
                committed_source_hashes.get(relative_path)
                if committed_source_hashes is not None
                else _committed_blob_sha256(
                    discovered_root, args.code_revision, relative_path
                )
            )
            _require(
                committed_hashes[relative_path] == expected_sha256,
                "code_identity source is not committed: {}".format(relative_path),
            )
        actual_runtime = runtime_identity or current_runtime_identity()
        _require(identity.get("runtime") == actual_runtime, "runtime identity changed")
        _require(
            identity.get("aggregate_sha256")
            == code_identity_aggregate_sha256(args.code_revision, files, actual_runtime),
            "code identity aggregate mismatch",
        )

        build = artifacts["build_manifest"].json_object()
        for source in build.get("inputs", []):
            _require(isinstance(source, dict), "build input is not an object")
            name = "build_input:{}".format(len(artifacts))
            artifacts[name] = FrozenArtifact.open(
                name,
                source.get("path"),
                expected_sha256=source.get("sha256"),
                capture_bytes=False,
                canonical=True,
            )

        import evaluate_version_space_workspace as evaluator

        gate_bundle = evaluator.validate_gate_bundle(
            artifacts["gate_manifest"].path,
            expected_manifest_sha256=artifacts["gate_manifest"].sha256,
            admission_path=artifacts["gate_admission"].path,
            expected_admission_sha256=artifacts["gate_admission"].sha256,
            expected_board_bindings=board_bindings,
            expected_evaluator_sha256=artifacts["evaluator"].sha256,
            expected_extractor_sha256=artifacts["extractor"].sha256,
            evaluator_path=artifacts["evaluator"].path,
            extractor_path=artifacts["extractor"].path,
            expected_code_revision=args.code_revision,
            expected_adapter_sha256=artifacts["adapter"].sha256,
            expected_seed=FROZEN_SEED,
            repo_root=discovered_root,
            expected_runtime=actual_runtime,
            live_git_revision=git_head,
            committed_source_hashes=committed_hashes,
            manifest_bytes=artifacts["gate_manifest"].raw,
            admission_bytes=artifacts["gate_admission"].raw,
            build_bytes=artifacts["build_manifest"].raw,
            trusted_source_hashes=trusted_source_hashes,
        )
        _require(
            build["tokenizer"]["path"] == artifacts["tokenizer"].path
            and build["tokenizer"]["sha256"] == artifacts["tokenizer"].sha256,
            "tokenizer does not match the verified build manifest",
        )
        structural_admissions = {}
        label_admissions = {}
        for board_name in FROZEN_BOARD_ORDER:
            structural_admissions[board_name] = artifacts[
                "{}_structural_admission".format(board_name)
            ].json_object()
            label_admissions[board_name] = artifacts[
                "{}_label_admission".format(board_name)
            ].json_object()
            hashes = {
                "data": artifacts["{}_data".format(board_name)].sha256,
                "tokenizer": artifacts["tokenizer"].sha256,
            }
            validate_board_admissions(
                hashes,
                structural_admissions[board_name],
                label_admissions[board_name],
            )
        if regenerate_boards:
            _validate_frozen_board_bytes(artifacts, identity)

        artifacts["base"] = FrozenArtifact.open(
            "base checkpoint", args.base, capture_bytes=False
        )
        artifacts["pointer_adapter"] = FrozenArtifact.open(
            "pointer adapter", args.pointer_adapter, capture_bytes=False
        )

        basis = evaluator.chain_basis(
            gate_bundle,
            expected_adapter_sha256=artifacts["adapter"].sha256,
            expected_extractor_sha256=artifacts["extractor"].sha256,
            expected_evaluator_sha256=artifacts["evaluator"].sha256,
            board_bindings=board_bindings,
        )
        chain_id = evaluator.canonical_sha256(basis)
        output_namespace = str(
            (discovered_root / "train" / "r10_score_chains" / chain_id).resolve()
        )
        return ChainPreflightBundle(
            repo_root=str(discovered_root),
            artifacts=artifacts,
            gate_manifest=gate_manifest,
            gate_admission=gate_admission,
            gate_bundle=gate_bundle,
            code_identity=copy.deepcopy(identity),
            board_bindings=board_bindings,
            structural_admissions=structural_admissions,
            label_admissions=label_admissions,
            chain_id=chain_id,
            output_namespace=output_namespace,
        )
    except BaseException:
        closed = set()
        for artifact in artifacts.values():
            if id(artifact) not in closed:
                artifact.close()
                closed.add(id(artifact))
        raise


def score_binding_metadata(preflight):
    """Return the schema-v2 bindings consumed by the CPU evaluator."""
    return {
        "audit": SCORE_AUDIT,
        "schema_version": SCORE_SCHEMA_VERSION,
        "board_name": preflight.board_name,
        "code_revision": preflight.code_identity["git_revision"],
        "code_identity": copy.deepcopy(preflight.code_identity),
        "code_identity_aggregate_sha256": preflight.code_identity["aggregate_sha256"],
        "seed": preflight.seed,
        "evaluator": preflight.paths["evaluator"],
        "evaluator_sha256": preflight.hashes["evaluator"],
        "extractor": preflight.paths["extractor"],
        "extractor_sha256": preflight.hashes["extractor"],
        "gate_manifest": preflight.paths["gate_manifest"],
        "gate_manifest_sha256": preflight.hashes["gate_manifest"],
        "gate_admission": preflight.paths["gate_admission"],
        "gate_admission_sha256": preflight.hashes["gate_admission"],
    }


def validate_hash_bindings(metadata, hashes, admission, label_admission):
    """Fail closed unless the adapter and both admissions form one hash chain."""
    for key in (
        "base_sha256",
        "pointer_adapter_sha256",
        "data_sha256",
        "tokenizer_sha256",
        "admission_sha256",
        "label_admission_sha256",
        "final_adapter_sha256",
    ):
        _require(
            _is_sha256(metadata.get(key)), "R9c metadata lacks a valid {}".format(key)
        )
    _require(metadata.get("protocol") == R9C_PROTOCOL, "invalid R9c adapter protocol")
    _require(metadata.get("arm") == "no_syndrome", "R10 requires arm=no_syndrome")
    _require(
        metadata.get("arm_config") == NO_SYNDROME_CONFIG,
        "R9c no_syndrome metadata differs from the frozen runtime contract",
    )
    _require(
        metadata.get("pointer_protocol") == POINTER_PROTOCOL,
        "invalid R9c pointer protocol",
    )
    _require(
        metadata.get("pointer_parameters_trainable") == 0,
        "R9c metadata does not freeze the pointer adapter",
    )
    rounds = metadata.get("rounds")
    _require(
        isinstance(rounds, int) and not isinstance(rounds, bool) and rounds > 0,
        "R9c metadata has an invalid fixed replay count",
    )
    _require(
        metadata.get("base_sha256") == hashes["base"],
        "R9c adapter does not bind the supplied base",
    )
    _require(
        metadata.get("pointer_adapter_sha256") == hashes["pointer_adapter"],
        "R9c adapter does not bind the supplied pointer",
    )
    _require(
        metadata.get("tokenizer_sha256") == hashes["tokenizer"],
        "R9c adapter does not bind the supplied tokenizer",
    )
    training_sha256 = metadata["data_sha256"]
    _require(
        admission.get("audit") == STRUCTURAL_ADMISSION_AUDIT,
        "invalid structural admission audit",
    )
    _require(admission.get("all_checks_pass") is True, "structural admission failed")
    _require(
        admission.get("eval_sha256") == hashes["data"],
        "structural admission does not bind the evaluation JSONL",
    )
    _require(
        admission.get("train_sha256") == training_sha256,
        "structural admission does not bind the R9c training data",
    )
    _require(
        admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "structural admission does not bind the tokenizer",
    )

    _require(
        label_admission.get("audit") == LABEL_ADMISSION_AUDIT,
        "invalid referential-label admission audit",
    )
    _require(
        label_admission.get("all_checks_pass") is True,
        "referential-label admission failed",
    )
    datasets = label_admission.get("datasets")
    _require(isinstance(datasets, dict), "referential-label admission lacks datasets")
    evaluation = datasets.get("eval")
    training = datasets.get("train")
    _require(
        isinstance(evaluation, dict) and evaluation.get("all_checks_pass") is True,
        "referential evaluation labels were not admitted",
    )
    _require(
        isinstance(training, dict) and training.get("all_checks_pass") is True,
        "referential training labels were not admitted",
    )
    _require(
        evaluation.get("sha256") == hashes["data"],
        "referential-label admission does not bind the evaluation JSONL",
    )
    _require(
        training.get("sha256") == training_sha256,
        "referential-label admission does not bind the R9c training data",
    )
    _require(
        label_admission.get("tokenizer_sha256") == hashes["tokenizer"],
        "referential-label admission does not bind the tokenizer",
    )
    _require(
        admission.get("eval_sha256") == evaluation.get("sha256")
        and admission.get("train_sha256") == training.get("sha256")
        and admission.get("tokenizer_sha256")
        == label_admission.get("tokenizer_sha256"),
        "structural and referential-label admissions do not describe one artifact set",
    )


def validate_pointer_metadata(pointer_metadata, r9c_metadata):
    _require(
        pointer_metadata.get("protocol") == POINTER_PROTOCOL,
        "invalid pointer adapter protocol",
    )
    _require(
        pointer_metadata.get("role_mode") == "pointer",
        "R10 requires the R4 pointer bridge",
    )
    _require(
        pointer_metadata.get("base_parameters_trainable") == 0,
        "pointer metadata does not freeze the base",
    )
    bindings = (
        ("base_sha256", "base_sha256"),
        ("data_sha256", "data_sha256"),
        ("admission_sha256", "admission_sha256"),
        ("label_admission_sha256", "label_admission_sha256"),
    )
    for pointer_key, r9c_key in bindings:
        _require(
            pointer_metadata.get(pointer_key) == r9c_metadata.get(r9c_key),
            "pointer and R9c metadata disagree on {}".format(pointer_key),
        )
    _require(
        int(pointer_metadata.get("hidden", -1))
        == int(r9c_metadata.get("pointer_hidden", -2)),
        "pointer and R9c metadata disagree on hidden width",
    )


def serialize_record(index, wrapped, probabilities, local_index):
    example = wrapped.compiled
    depth = len(example.operation_targets)
    for name in ("joint", "forward", "backward"):
        if probabilities[name].shape[1] != depth:
            raise ValueError(
                "{} score depth differs from compiled example".format(name)
            )
    return {
        "index": int(index),
        "reference": example.reference,
        "regime": example.regime,
        "operation_targets": [int(value) for value in example.operation_targets],
        "operation_values": [int(value) for value in example.operation_values],
        "initial_state": [int(value) for value in example.initial_values],
        "query_target": int(example.query_target),
        "answer": int(example.answer),
        "joint_probabilities": probabilities["joint"][local_index].tolist(),
        "forward_probabilities": probabilities["forward"][local_index].tolist(),
        "backward_probabilities": probabilities["backward"][local_index].tolist(),
        "query_probabilities": probabilities["query"][local_index].tolist(),
    }


def atomic_write_json_no_overwrite(payload, path):
    """Publish complete JSON atomically without an overwrite race."""
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


def atomic_write_bytes_no_overwrite(payload, path):
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
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
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


def _configure_frozen_determinism():
    for name, expected in FROZEN_ENVIRONMENT.items():
        _require(
            os.environ.get(name) == expected,
            "deterministic environment mismatch: {}".format(name),
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _require(
        torch.are_deterministic_algorithms_enabled()
        and torch.backends.cudnn.benchmark is False
        and torch.backends.cudnn.deterministic is True
        and torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False
        and torch.get_float32_matmul_precision() == "highest",
        "PyTorch deterministic settings did not hold",
    )


def _runtime_receipt():
    runtime = current_runtime_identity()
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    _require(isinstance(cuda_version, str) and cuda_version, "CUDA runtime is missing")
    _require(cudnn_version is not None, "cuDNN runtime is missing")
    return {
        **runtime,
        "cuda": cuda_version,
        "cudnn": str(cudnn_version),
    }


def _device_receipt():
    _require(torch.cuda.device_count() == 1, "R10 requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    _require(
        properties.name == EXPECTED_DEVICE_NAME
        and capability == EXPECTED_DEVICE_CAPABILITY,
        "R10 requires the frozen NVIDIA H100 PCIe device class",
    )
    uuid = str(getattr(properties, "uuid", "") or "")
    _require(uuid, "CUDA device UUID is unavailable")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("cannot bind CUDA PCI identity: {}".format(error)) from error
    pci_by_uuid = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            pci_by_uuid[parts[0]] = parts[1]
    pci_bus_id = pci_by_uuid.get(uuid)
    _require(pci_bus_id, "CUDA UUID is absent from nvidia-smi inventory")
    return {
        "type": "cuda",
        "name": properties.name,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
        "compute_capability": list(capability),
        "total_memory": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
    }


def _claim_one_run_namespace(preflight):
    output = Path(preflight.output_namespace)
    expected = Path(preflight.repo_root) / "train" / "r10_score_chains" / preflight.chain_id
    _require(output == expected.resolve(), "one-run namespace path changed")
    parent = output.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require(not parent.is_symlink(), "one-run namespace root must not be a symlink")
    try:
        os.mkdir(output, mode=0o700)
    except FileExistsError as error:
        raise SystemExit(
            "R10 chain was already claimed; repeat/alternate confirmation is forbidden: {}".format(
                output
            )
        ) from error
    claim = {
        "audit": "referential_version_score_to_decision_claim_r10",
        "attempt": 1,
        "chain_id": preflight.chain_id,
        "output_namespace": str(output),
        "state": "claimed_before_cuda_and_scores",
    }
    atomic_write_json_no_overwrite(claim, output / "claim.json")
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _require_unchanged(paths, hashes):
    for name, path in sorted(paths.items()):
        _require(
            Path(path).is_file()
            and Path(path).stat().st_size > 0
            and sha256_file(path) == hashes[name],
            "input changed during extraction: {} ({})".format(path, name),
        )


def _require_preflight_unchanged(preflight):
    _require_unchanged(preflight.paths, preflight.hashes)
    _require(
        current_runtime_identity() == preflight.code_identity["runtime"],
        "runtime changed during extraction",
    )
    repo_root, git_head = _discover_repo_context(
        preflight.paths["extractor"], preflight.repo_root
    )
    _require(
        str(repo_root) == preflight.repo_root,
        "repository root changed during extraction",
    )
    if git_head is not None:
        _require(
            git_head == preflight.code_identity["git_revision"],
            "git HEAD changed during extraction",
        )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-name", choices=BOARD_NAMES, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pointer-adapter", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--label-admission-sha256", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--gate-manifest-sha256", required=True)
    parser.add_argument("--gate-admission", required=True)
    parser.add_argument("--gate-admission-sha256", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--extractor-sha256", required=True)
    return parser


def run(args):
    _require(args.batch_size > 0, "batch size must be positive")
    _require(
        not os.path.lexists(args.out), "refusing existing output: {}".format(args.out)
    )
    preflight = validate_preflight(args)
    return _extract_and_publish(args, preflight)


def _extract_and_publish(args, preflight):
    paths = preflight.paths
    hashes = preflight.hashes
    _require(torch.cuda.is_available(), "R10 score extraction requires CUDA")
    try:
        torch.empty(1024, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as error:
        raise SystemExit(
            "R10 score extraction requires a usable CUDA allocation: {}".format(error)
        )
    torch.manual_seed(preflight.seed)
    torch.cuda.manual_seed_all(preflight.seed)

    checkpoint = torch.load(paths["adapter"], map_location="cpu")
    metadata = checkpoint.get("referential_syndrome_microcode")
    _require(isinstance(metadata, dict), "adapter lacks complete R9c metadata")
    _require(
        checkpoint.get("step") == "syndrome_adapter_ep1", "invalid R9c adapter step"
    )
    _require(
        isinstance(checkpoint.get("microcode_state"), dict),
        "adapter lacks microcode state",
    )
    validate_hash_bindings(
        metadata,
        hashes,
        preflight.structural_admission,
        preflight.label_admission,
    )

    from tokenizers import Tokenizer

    from eval_referential_slot_microcode import load_examples
    from eval_referential_syndrome_microcode import matched_batches
    from referential_syndrome_microcode import ReferentialSyndromeBridge
    from train_referential_slot_microcode import pad_ids
    from train_referential_syndrome_microcode import hash_microcode_state, load_pointer

    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(paths["tokenizer"])
    base_checkpoint, pointer_metadata, pointer = load_pointer(
        paths["base"],
        paths["pointer_adapter"],
        "cuda",
    )
    validate_pointer_metadata(pointer_metadata, metadata)
    examples = load_examples(paths["data"], tokenizer, pointer.model.cfg.seq_len)
    del base_checkpoint
    batches = matched_batches(examples, args.batch_size)

    bridge = (
        ReferentialSyndromeBridge(
            pointer,
            pointer_hidden=int(pointer_metadata["hidden"]),
            memory_dim=int(metadata["memory_dim"]),
        )
        .to("cuda")
        .eval()
    )
    incompatible = bridge.microcode.load_state_dict(
        checkpoint["microcode_state"], strict=True
    )
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "R9c microcode state is incompatible with the frozen bridge",
    )
    bridge.requires_grad_(False)
    _require(
        bridge.adapter_num_params() == int(metadata.get("adapter_parameters", -1)),
        "R9c adapter parameter count differs from metadata",
    )
    _require(
        hash_microcode_state(bridge) == metadata.get("final_adapter_sha256"),
        "R9c microcode state does not match its final adapter hash",
    )

    records = [None] * len(examples)
    rounds = int(metadata["rounds"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected = [examples[index] for index in indices]
            encoded = bridge.encode_examples(pad_ids(selected, "cuda"), selected)
            run = bridge.microcode(
                encoded.event_features,
                encoded.values,
                encoded.initial_values,
                encoded.query_goals,
                rounds=rounds,
                conditioning="directional",
                use_syndrome=False,
                adaptive=False,
            )
            probabilities = {
                name: tensor.cpu()
                for name, tensor in categorical_probabilities(
                    run.forward_logits,
                    run.backward_logits,
                    encoded.query_logits,
                ).items()
            }
            for local_index, index in enumerate(indices):
                records[index] = serialize_record(
                    index,
                    examples[index],
                    probabilities,
                    local_index,
                )
            if batch_number % 20 == 0 or batch_number == len(batches):
                print(
                    "[r10-scores] {}/{} batches".format(batch_number, len(batches)),
                    flush=True,
                )

    _require(
        all(record is not None for record in records),
        "R10 extraction left unscored rows",
    )
    _require_preflight_unchanged(preflight)
    result = score_binding_metadata(preflight)
    result.update(
        {
            "base": paths["base"],
            "base_sha256": hashes["base"],
            "pointer_adapter": paths["pointer_adapter"],
            "pointer_adapter_sha256": hashes["pointer_adapter"],
            "adapter": paths["adapter"],
            "adapter_sha256": hashes["adapter"],
            "adapter_step": checkpoint["step"],
            "adapter_state_sha256": metadata["final_adapter_sha256"],
            "adapter_metadata": metadata,
            "pointer_adapter_metadata": pointer_metadata,
            "data": paths["data"],
            "data_sha256": hashes["data"],
            "tokenizer": paths["tokenizer"],
            "tokenizer_sha256": hashes["tokenizer"],
            "structural_admission": paths["structural_admission"],
            "structural_admission_sha256": hashes["structural_admission"],
            "referential_label_admission": paths["referential_label_admission"],
            "referential_label_admission_sha256": hashes["referential_label_admission"],
            "r9c_training_data": metadata["data"],
            "r9c_training_data_sha256": metadata["data_sha256"],
            "r9c_training_structural_admission_sha256": metadata["admission_sha256"],
            "r9c_training_referential_label_admission_sha256": metadata[
                "label_admission_sha256"
            ],
            "categorical_order": {
                "operations": list(OPCODES),
                "queries": list(QUERIES),
            },
            "replay": {
                "arm": "no_syndrome",
                "mode": "fixed",
                "adaptive": False,
                "rounds": rounds,
                "conditioning": "directional",
                "use_syndrome": False,
                "shuffle_goal": False,
            },
            "cases": len(records),
            "events": sum(len(record["operation_targets"]) for record in records),
            "batches": len(batches),
            "records": records,
            "claim_boundary": (
                "Read-only categorical score extraction from the frozen rejected R9c no-syndrome control. "
                "These probabilities do not establish reasoning, certification, or fresh transfer."
            ),
        }
    )
    try:
        atomic_write_json_no_overwrite(result, args.out)
    except FileExistsError as error:
        raise SystemExit(str(error))
    print(
        "[r10-scores] saved {} cases to {}".format(len(records), args.out), flush=True
    )
    return result


def main():
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
