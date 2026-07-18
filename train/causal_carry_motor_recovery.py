#!/usr/bin/env python3
"""Dual-provenance recovery for the sealed a0c258e carry feature shards.

This module never rewrites or publishes into the upstream canonical root.  It
treats that root as immutable scientific input and records the independently
reviewed recovery executor as a second provenance domain.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
import tokenizers as tokenizers_module
from tokenizers import Tokenizer

import causal_carry_motor as upstream
import model as model_module


UPSTREAM_SOURCE_COMMIT = "a0c258e6709766c643cf127a429a7d6ef4a4211b"
UPSTREAM_SOURCE_MANIFEST_SHA256 = (
    "9ae61e1a3e8f672a71a01edc16e6a5f1f8f3c69f49afd5e97f41c6cde15350a9"
)
UPSTREAM_PLAN_SHA256 = (
    "1b845d47f6875df571169efb5adb0716dfbc5d266a2499e4a92451351a262b6d"
)
UPSTREAM_CONFIRMATION_COMMITMENT_SHA256 = (
    "1ee32e4e2e8f9eb56026b7b8de1fdff207e9fd3694e0ae354f103d58ebb820da"
)
UPSTREAM_BOARD_ROWS_SHA256 = (
    "6517b1ff3aa557e449a2eef9c5540c3d5f8699482d933d5c320b606adb4a0f1b"
)
UPSTREAM_CANONICAL_BOARD_SHA256 = (
    "d6282610ba845b23ebe849efe574233bf657a50aea0a7edb901e9e1d95b24391"
)
NORMALIZATION_MISMATCH_LEDGER_SHA256 = (
    "b43cb4a6fbfab97c659e8658f63185ae8b3dc1d8cce34089958d3b09df0593b6"
)
UPSTREAM_SHARD_SHA256 = (
    "4affa12434513ebe9587464ff38656abaaf7e47904d9db6ced252c3adea52a96",
    "4731c1644703e26c1978ca1ec1ba80af7c173c5d9676ae68fbd04368f3b54c2c",
    "e81639e68a838bfa6695be92f7c1333d100b2317c48fb2cf0d995f22a6e50a43",
    "ae86ec1b70dca21d67849fc4be17ffec682472851735c3b9523292836a74e70f",
    "ce5a151f89e20e774c7d37afc446ea026ec14a587c70fa614414f060f10a2144",
    "f02d8221bf3a393566c279e27bf888fcbd1ef9ea17bdd33262472c898950ea83",
    "009b83f0c2a70362654e3e3e4cad27d30f79f93f3bdd32d6ce3064695dd2b9db",
    "8214d356288c56a116a3de753a8948a35f731d52c520fa906f4e31c1b0f14fb4",
)

DATA_ROOT = Path("/lustre/fs1/home/sa305415/shohin")
UPSTREAM_ROOT = (
    DATA_ROOT / "artifacts" / "carry_motor" / f"canonical_{UPSTREAM_SOURCE_COMMIT}"
)
UPSTREAM_PLAN_PATH = UPSTREAM_ROOT / "plan.json"
UPSTREAM_CONFIRMATION_PATH = (
    DATA_ROOT
    / "artifacts"
    / "carry_motor"
    / "confirmation_commitments"
    / f"commitment_{UPSTREAM_SOURCE_COMMIT}"
    / "commitment.json"
)
RECOVERY_PARENT = DATA_ROOT / "artifacts" / "carry_motor" / "recoveries"
REVIEW_PARENT = DATA_ROOT / "artifacts" / "carry_motor" / "recovery_reviews"
PINNED_PYTHON_LAUNCHER = DATA_ROOT / "miniforge3" / "bin" / "python"
PINNED_GIT = Path("/usr/bin/git")

RECOVERY_PLAN_AUDIT = "causal_carry_motor_recovery_plan_v2"
RECOVERY_FIT_AUDIT = "causal_carry_motor_fit_v9_recovery"
RECOVERY_REVIEW_AUDIT = "causal_carry_motor_recovery_hostile_review_v2"
RECOVERY_EXECUTOR_SOURCE_SCHEMA = "carry_motor_recovery_executor_source_v2"
RECOVERY_EXECUTOR_RUNTIME_SCHEMA = "carry_motor_recovery_executor_runtime_v1"
UPSTREAM_CUSTODY_SCHEMA = "carry_motor_upstream_custody_snapshot_v1"
NORMALIZATION_SCHEMA = "carry_motor_fit_board_strict_json_normalization_v1"
DESERIALIZATION_SCHEMA = "bound_weights_only_torchversion_allowlist_v1"

RECOVERY_SOURCE_PATHS = (
    "R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md",
    "train/causal_carry_motor_recovery.py",
    "train/test_causal_carry_motor_recovery.py",
    "train/jobs/causal_carry_motor_recovery.sbatch",
)
RECOVERY_NAME_STATUS_DIFF = tuple(
    f"A\t{name}" for name in sorted(RECOVERY_SOURCE_PATHS)
)
EXECUTOR_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "LANG": "C",
    "LC_ALL": "C",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
FORBIDDEN_EXECUTOR_ENVIRONMENT = (
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONSAFEPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
)

RECOVERY_PLAN_KEYS = frozenset(
    {
        "audit",
        "recovery",
        "recovery_plan_path",
        "recovery_executor_source_contract",
        "executor_runtime_contract",
        "hostile_review_binding",
        "upstream_protocol",
        "normalization_proof",
        "allowed_transformation",
        "fit_contract",
        "output_contract",
        "deserialization_contract",
        "claim_boundary",
    }
)
RECOVERY_FIT_KEYS = frozenset(
    {
        "audit",
        "recovery",
        "recovery_plan_sha256",
        "recovery_executor_source_contract",
        "executor_runtime_contract",
        "upstream_protocol_source_contract",
        "upstream_plan_binding",
        "upstream_shard_receipts",
        "normalization_proof",
        "allowed_transformation",
        "deserialization_contract",
        "fit_payload",
        "claim_boundary",
    }
)
LEGACY_PAYLOAD_KEYS = upstream.CANONICAL_FIT_KEYS - {"audit", "canonical"}

EXPECTED_NORMALIZATION_MISMATCHES = (
    {
        "path": "board.prompt_length_histogram",
        "generated_key_type": "int",
        "sealed_key_type": "str",
        "generated_keys": [97, 99, 103, 105],
        "sealed_keys": ["97", "99", "103", "105"],
    },
    {
        "path": "board.token_length_histogram",
        "generated_key_type": "int",
        "sealed_key_type": "str",
        "generated_keys": [114, 116, 120, 122],
        "sealed_keys": ["114", "116", "120", "122"],
    },
)

ALLOWED_TRANSFORMATION = {
    "schema": NORMALIZATION_SCHEMA,
    "operation": "strict_json_round_trip_of_complete_generated_fit_board",
    "changed_paths": [item["path"] for item in EXPECTED_NORMALIZATION_MISMATCHES],
    "permitted_semantic_changes": 0,
    "permitted_additional_transformations": 0,
}
DESERIALIZATION_CONTRACT = {
    "schema": DESERIALIZATION_SCHEMA,
    "weights_only": True,
    "safe_globals": ["torch.torch_version.TorchVersion"],
    "bind_before_deserialize": True,
    "ambient_override_environment_forbidden": [
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
    ],
    "fallback_to_unrestricted_pickle": False,
}
RECOVERY_PLAN_CLAIM_BOUNDARY = (
    "This recovery plan binds an already sealed upstream feature lineage and one "
    "reviewed mechanical normalization executor. It establishes no fitted motor, "
    "evaluation result, mechanism, capability, or reasoning claim."
)
RECOVERY_FIT_CLAIM_BOUNDARY = (
    "This v9 recovery fit is not a v8 canonical artifact and establishes no reasoning "
    "result. It reuses exact sealed a0c258e features under explicit dual provenance; "
    "heldout development and separately reviewed confirmation recovery remain required."
)
REVIEW_CLAIM_BOUNDARY = (
    "GO attests only that the exact recovery source and normalization amendment are "
    "eligible to publish the frozen fit. It is not a capability or evaluation result."
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value):
    return sha256_bytes(canonical_json_payload(value).encode("ascii"))


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_exact_json(text, label):
    try:
        return json.loads(
            text,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def canonical_json_payload(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "value is not finite or contains non-finite strict JSON"
        ) from exc


def canonical_json_document(value):
    return load_exact_json(canonical_json_payload(value), "canonical JSON document")


def type_strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            type_strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _typed_tree(value):
    if value is None:
        return ["none", None]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("typed tree contains non-finite float")
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is list:
        return ["list", [_typed_tree(item) for item in value]]
    if type(value) is dict:
        entries = []
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise ValueError("typed tree contains unsupported mapping key")
            entries.append([_typed_tree(key), _typed_tree(item)])
        entries.sort(key=canonical_json_payload)
        return ["dict", entries]
    raise ValueError(f"typed tree contains unsupported type: {type(value).__name__}")


def typed_tree_sha256(value):
    return stable_json_sha256(_typed_tree(value))


def normalization_contract():
    return {
        "schema": NORMALIZATION_SCHEMA,
        "upstream_board_rows_sha256": UPSTREAM_BOARD_ROWS_SHA256,
        "canonical_board_sha256": UPSTREAM_CANONICAL_BOARD_SHA256,
        "mismatch_ledger_sha256": NORMALIZATION_MISMATCH_LEDGER_SHA256,
        "expected_mismatches": list(EXPECTED_NORMALIZATION_MISMATCHES),
        "allowed_transformation": ALLOWED_TRANSFORMATION,
    }


def build_normalization_proof(generated_board, sealed_board, rows):
    """Prove that strict JSON key normalization is the sole board difference."""
    generated = copy.deepcopy(generated_board)
    generated["rows_sha256"] = upstream.stable_json_sha256(rows)
    if generated["rows_sha256"] != UPSTREAM_BOARD_ROWS_SHA256:
        raise ValueError("generated fit rows differ from sealed upstream identity")
    if stable_json_sha256(sealed_board) != UPSTREAM_CANONICAL_BOARD_SHA256:
        raise ValueError("sealed upstream board identity mismatch")
    if set(generated) != set(sealed_board):
        raise ValueError("generated and sealed board schemas differ")

    histogram_fields = {
        "prompt_length_histogram",
        "token_length_histogram",
    }
    for key in sorted(set(generated) - histogram_fields):
        if not type_strict_equal(generated[key], sealed_board[key]):
            raise ValueError(f"non-histogram board difference: {key}")

    observed = []
    for field in ("prompt_length_histogram", "token_length_histogram"):
        raw = generated[field]
        sealed = sealed_board[field]
        if type(raw) is not dict or type(sealed) is not dict:
            raise ValueError(f"{field} is not a mapping")
        if not raw or any(type(key) is not int for key in raw):
            raise ValueError(f"{field} generated keys are not all integers")
        if not sealed or any(type(key) is not str for key in sealed):
            raise ValueError(f"{field} sealed keys are not all strings")
        if any(type(value) is not int for value in (*raw.values(), *sealed.values())):
            raise ValueError(f"{field} counts are not exact integers")
        if {str(key): value for key, value in raw.items()} != sealed:
            raise ValueError(f"{field} differs beyond JSON key typing")
        observed.append(
            {
                "path": f"board.{field}",
                "generated_key_type": "int",
                "sealed_key_type": "str",
                "generated_keys": sorted(raw),
                "sealed_keys": sorted(sealed, key=int),
            }
        )
    if not type_strict_equal(observed, list(EXPECTED_NORMALIZATION_MISMATCHES)):
        raise ValueError(
            "normalization mismatch ledger is not the frozen two-entry ledger"
        )
    if stable_json_sha256(observed) != NORMALIZATION_MISMATCH_LEDGER_SHA256:
        raise ValueError("normalization mismatch ledger hash mismatch")

    normalized = canonical_json_document(generated)
    if not type_strict_equal(normalized, sealed_board):
        raise ValueError("strict JSON normalization does not reproduce sealed board")
    if stable_json_sha256(normalized) != UPSTREAM_CANONICAL_BOARD_SHA256:
        raise ValueError("normalized board hash mismatch")
    return {
        "schema": NORMALIZATION_SCHEMA,
        "generated_rows": len(rows),
        "generated_rows_sha256": generated["rows_sha256"],
        "generated_board_typed_sha256": typed_tree_sha256(generated),
        "canonical_board_sha256": stable_json_sha256(normalized),
        "mismatch_count": len(observed),
        "mismatches": observed,
        "mismatch_ledger_sha256": stable_json_sha256(observed),
        "canonical_board_equal": True,
        "allowed_transformation": ALLOWED_TRANSFORMATION,
    }, normalized


class BoundFile:
    """Exact lexical path plus open-descriptor identity and digest binding."""

    def __init__(
        self,
        path,
        expected_path,
        expected_sha256,
        label,
        *,
        required_mode=None,
        required_parent_mode=None,
    ):
        raw = os.fspath(path)
        expected = Path(expected_path)
        if raw != str(expected) or not expected.is_absolute():
            raise ValueError(f"{label} path aliases or differs from frozen path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
            raise ValueError(f"{label} receipt is invalid")
        parent_stat = os.lstat(expected.parent)
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
            raise ValueError(f"{label} parent is not a regular directory")
        if required_parent_mode is not None and stat.S_IMODE(
            parent_stat.st_mode
        ) != int(required_parent_mode):
            raise ValueError(f"{label} parent mode mismatch")
        path_stat = os.lstat(expected)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or path_stat.st_nlink != 1
        ):
            raise ValueError(f"{label} is not a one-link regular file")
        if required_mode is not None and stat.S_IMODE(path_stat.st_mode) != int(
            required_mode
        ):
            raise ValueError(f"{label} mode mismatch")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(expected, flags)
        self.handle = os.fdopen(descriptor, "rb")
        opened = os.fstat(self.handle.fileno())
        self.path = expected
        self.label = label
        self.identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if (opened.st_dev, opened.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            self.close()
            raise ValueError(f"{label} changed during binding")
        self.sha256 = self._hash_handle()
        if self.sha256 != expected_sha256:
            self.close()
            raise ValueError(f"{label} artifact hash mismatch")

    def _hash_handle(self):
        digest = hashlib.sha256()
        self.handle.seek(0)
        for block in iter(lambda: self.handle.read(1024 * 1024), b""):
            digest.update(block)
        self.handle.seek(0)
        return digest.hexdigest()

    def bytes(self):
        self.handle.seek(0)
        payload = self.handle.read()
        self.handle.seek(0)
        return payload

    def text(self):
        return self.bytes().decode("utf-8")

    def verify(self):
        observed = os.stat(self.path, follow_symlinks=False)
        identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if identity != self.identity or self._hash_handle() != self.sha256:
            raise RuntimeError(f"{self.label} changed after binding")

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def safe_torch_load(bound):
    for name in DESERIALIZATION_CONTRACT["ambient_override_environment_forbidden"]:
        if name in os.environ:
            raise RuntimeError(
                f"ambient torch deserialization override is forbidden: {name}"
            )
    torch_version_type = torch.torch_version.TorchVersion
    bound.handle.seek(0)
    with torch.serialization.safe_globals([torch_version_type]):
        value = torch.load(bound.handle, map_location="cpu", weights_only=True)
    bound.handle.seek(0)
    bound.verify()
    return value


def _git(root, *arguments):
    binary = PINNED_GIT
    if not binary.is_file() or binary.is_symlink():
        raise ValueError("recovery git executable is not the pinned regular file")
    return subprocess.check_output(
        [str(binary), "-C", str(root), *arguments], stderr=subprocess.STDOUT
    )


def _git_text(root, *arguments):
    return _git(root, *arguments).decode().strip()


def validate_loaded_module_paths(repo_root):
    root = Path(repo_root)
    expected = {
        "recovery": root / "train" / "causal_carry_motor_recovery.py",
        "upstream": root / "train" / "causal_carry_motor.py",
        "model": root / "train" / "model.py",
    }
    observed = {
        "recovery": Path(__file__),
        "upstream": Path(upstream.__file__),
        "model": Path(model_module.__file__),
    }
    for name in expected:
        if (
            os.fspath(observed[name]) != str(expected[name])
            or observed[name].resolve(strict=True) != expected[name]
        ):
            raise ValueError(f"loaded {name} module is shadowed or aliased")
    return {name: str(path) for name, path in expected.items()}


def validate_recovery_commit_topology(recovery_source_commit, *, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    lineage = _git_text(
        root, "rev-list", "--parents", "-n", "1", recovery_source_commit
    ).split()
    if lineage != [recovery_source_commit, UPSTREAM_SOURCE_COMMIT]:
        raise ValueError("recovery commit must have a0c258e as its sole direct parent")
    observed = tuple(
        _git(
            root,
            "diff",
            "--name-status",
            "--no-renames",
            UPSTREAM_SOURCE_COMMIT,
            recovery_source_commit,
        )
        .decode()
        .splitlines()
    )
    if observed != RECOVERY_NAME_STATUS_DIFF:
        raise ValueError(
            "recovery commit diff must be exactly four added recovery files"
        )
    return {
        "parent_commit": UPSTREAM_SOURCE_COMMIT,
        "name_status_diff": list(observed),
    }


def validate_checkout_closed_world(repo_root):
    root = Path(repo_root)
    tracked_payload = _git(root, "ls-files", "-z").decode("utf-8")
    tracked = {name for name in tracked_payload.split("\0") if name}
    observed = set()
    for directory, child_directories, files in os.walk(root, followlinks=False):
        current = Path(directory)
        if current == root and ".git" in child_directories:
            child_directories.remove(".git")
        for name in tuple(child_directories):
            path = current / name
            if path.is_symlink():
                observed.add(path.relative_to(root).as_posix())
                child_directories.remove(name)
        for name in files:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                continue
            observed.add(relative)
    if observed != tracked:
        extras = sorted(observed - tracked)
        missing = sorted(tracked - observed)
        raise ValueError(
            "recovery checkout is not closed-world: "
            f"extra={extras!r} missing={missing!r}"
        )
    return {
        "tracked_file_count": len(tracked),
        "tracked_paths_sha256": stable_json_sha256(sorted(tracked)),
    }


def _capture_executable(path, label):
    expected = Path(path)
    observed = os.lstat(expected)
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
            raise ValueError(f"{label} changed during binding")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        current = os.stat(expected, follow_symlinks=False)
        identity = {
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "mode": stat.S_IMODE(opened.st_mode),
            "links": opened.st_nlink,
            "uid": opened.st_uid,
            "gid": opened.st_gid,
            "size": opened.st_size,
            "mtime_ns": opened.st_mtime_ns,
            "ctime_ns": opened.st_ctime_ns,
        }
        current_identity = {
            "device": current.st_dev,
            "inode": current.st_ino,
            "mode": stat.S_IMODE(current.st_mode),
            "links": current.st_nlink,
            "uid": current.st_uid,
            "gid": current.st_gid,
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "ctime_ns": current.st_ctime_ns,
        }
        if identity != current_identity:
            raise RuntimeError(f"{label} changed while hashing")
        return {
            "path": str(expected),
            "sha256": digest.hexdigest(),
            "identity": identity,
        }
    finally:
        os.close(descriptor)


def capture_executor_runtime_contract(*, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    launcher = PINNED_PYTHON_LAUNCHER
    if not root.is_absolute() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("recovery source root is not an exact physical directory")
    for name, value in EXECUTOR_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise ValueError(f"recovery executor environment mismatch: {name}")
    for name in FORBIDDEN_EXECUTOR_ENVIRONMENT:
        if name in os.environ:
            raise ValueError(f"forbidden recovery executor environment: {name}")
    expected_pythonpath = str(root / "train")
    if os.environ.get("PYTHONPATH") != expected_pythonpath:
        raise ValueError("recovery PYTHONPATH is not the sole reviewed train path")
    resolved_launcher = launcher.resolve(strict=True)
    resolved_executable = Path(sys.executable).resolve(strict=True)
    if resolved_launcher != resolved_executable:
        raise ValueError("running Python does not resolve from the pinned launcher")
    expected_flags = {
        "dont_write_bytecode": 1,
        "hash_randomization": 0,
        "no_user_site": 1,
        "ignore_environment": 0,
        "isolated": 0,
        "optimize": 0,
    }
    observed_flags = {name: getattr(sys.flags, name) for name in expected_flags}
    if observed_flags != expected_flags:
        raise ValueError("recovery Python startup flags mismatch")
    module_paths = validate_loaded_module_paths(root)
    package_files = {}
    for name, module in (
        ("torch", torch),
        ("tokenizers", tokenizers_module),
    ):
        module_path = Path(module.__file__).resolve(strict=True)
        if Path(sys.prefix).resolve(strict=True) not in module_path.parents:
            raise ValueError(f"loaded {name} package escapes the pinned Python prefix")
        package_files[name] = _capture_executable(
            module_path, f"loaded {name} package entrypoint"
        )
    return canonical_json_document(
        {
            "schema": RECOVERY_EXECUTOR_RUNTIME_SCHEMA,
            "source_root": str(root),
            "launcher_path": str(launcher),
            "resolved_executable": _capture_executable(
                resolved_executable, "resolved Python interpreter"
            ),
            "git_executable": _capture_executable(PINNED_GIT, "pinned git"),
            "python": {
                "version": sys.version,
                "implementation": sys.implementation.name,
                "cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
                "executable": sys.executable,
                "sys_path": list(sys.path),
                "flags": observed_flags,
            },
            "packages": {
                "torch": {
                    "version": str(torch.__version__),
                    "entrypoint": package_files["torch"],
                },
                "tokenizers": {
                    "version": str(tokenizers_module.__version__),
                    "entrypoint": package_files["tokenizers"],
                },
            },
            "module_paths": module_paths,
            "environment": {
                **EXECUTOR_ENVIRONMENT,
                "PYTHONPATH": expected_pythonpath,
            },
            "forbidden_environment": list(FORBIDDEN_EXECUTOR_ENVIRONMENT),
        }
    )


def build_recovery_executor_source_contract(
    recovery_source_commit,
    expected_manifest_sha256,
    *,
    repo_root=None,
):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    if not re.fullmatch(r"[0-9a-f]{40}", str(recovery_source_commit)):
        raise ValueError("recovery source commit must be lowercase 40-hex")
    if recovery_source_commit == UPSTREAM_SOURCE_COMMIT:
        raise ValueError("recovery executor may not alias the upstream source identity")
    if _git_text(root, "rev-parse", "HEAD") != recovery_source_commit:
        raise ValueError("reviewed recovery source commit is not checked out")
    topology = validate_recovery_commit_topology(recovery_source_commit, repo_root=root)
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("recovery executor checkout is not clean")
    checkout = validate_checkout_closed_world(root)
    module_paths = validate_loaded_module_paths(root)
    sources = {}
    source_files = {}
    for name in RECOVERY_SOURCE_PATHS:
        tree_entry = _git_text(root, "ls-tree", recovery_source_commit, "--", name)
        if not re.fullmatch(
            rf"100644 blob [0-9a-f]{{40}}\t{re.escape(name)}", tree_entry
        ):
            raise ValueError(f"recovery source Git mode mismatch: {name}")
        committed = _git(root, "show", f"{recovery_source_commit}:{name}")
        path = root / name
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o644
            or before.st_nlink != 1
        ):
            raise ValueError(f"recovery source file identity mismatch: {name}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            with os.fdopen(os.dup(descriptor), "rb") as source:
                working = source.read()
            after = os.stat(path, follow_symlinks=False)
        finally:
            os.close(descriptor)
        if not (
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after)
        ):
            raise RuntimeError(f"recovery source changed during binding: {name}")
        if working != committed:
            raise ValueError(f"recovery executor source differs from commit: {name}")
        sources[name] = sha256_bytes(committed)
        source_files[name] = {
            "git_mode": "100644",
            "checkout_mode": "0644",
            "links": 1,
        }
    manifest = stable_json_sha256(sources)
    if manifest != expected_manifest_sha256:
        raise ValueError("recovery executor source manifest mismatch")
    return {
        "schema": RECOVERY_EXECUTOR_SOURCE_SCHEMA,
        "git_commit": recovery_source_commit,
        **topology,
        "checkout": checkout,
        "sources": sources,
        "source_files": source_files,
        "manifest_sha256": manifest,
        "loaded_module_paths": module_paths,
    }


def verify_upstream_source_snapshot(plan, *, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    expected_contract = {
        "git_commit": UPSTREAM_SOURCE_COMMIT,
        "manifest_sha256": UPSTREAM_SOURCE_MANIFEST_SHA256,
    }
    if not type_strict_equal(plan.get("source_contract"), expected_contract):
        raise ValueError("upstream plan source contract mismatch")
    plan_sources = plan.get("scientific_source_sha256")
    if type(plan_sources) is not dict or set(plan_sources) != set(
        upstream.SCIENTIFIC_SOURCE_PATHS
    ):
        raise ValueError("upstream plan scientific source schema mismatch")
    observed = {}
    for name in upstream.SCIENTIFIC_SOURCE_PATHS:
        committed = _git(root, "show", f"{UPSTREAM_SOURCE_COMMIT}:{name}")
        digest = sha256_bytes(committed)
        if plan_sources[name] != digest:
            raise ValueError(f"upstream plan source hash mismatch: {name}")
        if (root / name).read_bytes() != committed:
            raise ValueError(f"loaded upstream dependency differs from a0c258e: {name}")
        observed[name] = digest
    if stable_json_sha256(observed) != UPSTREAM_SOURCE_MANIFEST_SHA256:
        raise ValueError("upstream scientific source manifest mismatch")
    return expected_contract, observed


def validate_confirmation_generator_contract(plan, source_contract, source_hashes):
    commitment = plan.get("confirmation_commitment")
    document = commitment.get("document") if isinstance(commitment, dict) else None
    generator = (
        document.get("generator_source_contract")
        if isinstance(document, dict)
        else None
    )
    expected_sources = {
        name: source_hashes[name]
        for name in upstream.CANONICAL_CONFIRMATION_GENERATOR_SOURCES
    }
    expected_generator = {
        "schema": upstream.CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
        "entrypoint": upstream.CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
        "sources": expected_sources,
        "manifest_sha256": upstream.stable_json_sha256(expected_sources),
    }
    if (
        not type_strict_equal(generator, expected_generator)
        or document.get("source_contract") != source_contract
    ):
        raise ValueError("upstream confirmation generator substitution detected")
    return generator


def recovery_root(recovery_source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", str(recovery_source_commit)):
        raise ValueError("recovery source commit must be lowercase 40-hex")
    if recovery_source_commit == UPSTREAM_SOURCE_COMMIT:
        raise ValueError("recovery root cannot alias the upstream commit")
    return RECOVERY_PARENT / (
        f"upstream_{UPSTREAM_PLAN_SHA256}_executor_{recovery_source_commit}"
    )


def recovery_review_path(recovery_source_commit):
    return REVIEW_PARENT / f"review_{recovery_source_commit}" / "hostile_review.json"


def _require_directory(path, mode, label, children=None):
    observed = os.lstat(path)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != mode
    ):
        raise ValueError(f"{label} directory identity or mode mismatch")
    if children is not None and {item.name for item in Path(path).iterdir()} != set(
        children
    ):
        raise ValueError(f"{label} directory is not closed-world")


def _stat_identity(observed):
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "links": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _capture_directory(path, mode, children, label):
    expected = Path(path)
    before = os.lstat(expected)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise ValueError(f"{label} directory identity or mode mismatch")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(expected, flags)
    try:
        opened = os.fstat(descriptor)
        observed_children = sorted(os.listdir(descriptor))
        if observed_children != sorted(children):
            raise ValueError(f"{label} directory is not closed-world")
        after = os.stat(expected, follow_symlinks=False)
        if not (
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after)
        ):
            raise RuntimeError(f"{label} directory changed during capture")
        return {
            "path": str(expected),
            "kind": "directory",
            "identity": _stat_identity(opened),
            "children": observed_children,
        }
    finally:
        os.close(descriptor)


def _capture_custody_file(path, mode, expected_sha256, label):
    expected = Path(path)
    before = os.lstat(expected)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
        or before.st_nlink != 1
    ):
        raise ValueError(f"{label} file identity or mode mismatch")
    descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = os.stat(expected, follow_symlinks=False)
        if not (
            _stat_identity(before) == _stat_identity(opened) == _stat_identity(after)
        ):
            raise RuntimeError(f"{label} changed during capture")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        return {
            "path": str(expected),
            "kind": "file",
            "identity": _stat_identity(opened),
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def capture_upstream_custody_snapshot():
    root_children = {
        "plan.json",
        "fit",
        "development_eval",
        "confirmation_eval",
        *(f"shard_{index:02d}" for index in range(upstream.CANONICAL_FEATURE_SHARDS)),
    }
    entries = [
        _capture_directory(
            UPSTREAM_ROOT, 0o555, root_children, "upstream canonical root"
        ),
        _capture_custody_file(
            UPSTREAM_PLAN_PATH,
            0o444,
            UPSTREAM_PLAN_SHA256,
            "upstream canonical plan",
        ),
    ]
    for name in ("fit", "development_eval", "confirmation_eval"):
        entries.append(
            _capture_directory(UPSTREAM_ROOT / name, 0o700, (), f"upstream {name}")
        )
    for index, expected_sha256 in enumerate(UPSTREAM_SHARD_SHA256):
        directory = UPSTREAM_ROOT / f"shard_{index:02d}"
        entries.append(
            _capture_directory(
                directory, 0o555, ("features.pt",), f"upstream shard {index}"
            )
        )
        entries.append(
            _capture_custody_file(
                directory / "features.pt",
                0o444,
                expected_sha256,
                f"upstream shard {index} artifact",
            )
        )
    entries.extend(
        (
            _capture_directory(
                UPSTREAM_CONFIRMATION_PATH.parent,
                0o555,
                ("commitment.json",),
                "upstream confirmation commitment",
            ),
            _capture_custody_file(
                UPSTREAM_CONFIRMATION_PATH,
                0o444,
                UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
                "upstream confirmation commitment",
            ),
        )
    )
    return canonical_json_document(
        {"schema": UPSTREAM_CUSTODY_SCHEMA, "entries": entries}
    )


def assert_upstream_custody_unchanged(expected, phase):
    observed = capture_upstream_custody_snapshot()
    if not type_strict_equal(observed, expected):
        raise RuntimeError(f"upstream custody changed {phase}")
    return observed


def validate_upstream_layout():
    return capture_upstream_custody_snapshot()


def load_upstream_plan(*, repo_root=None):
    upstream_custody_snapshot = validate_upstream_layout()
    plan_bound = BoundFile(
        str(UPSTREAM_PLAN_PATH),
        UPSTREAM_PLAN_PATH,
        UPSTREAM_PLAN_SHA256,
        "upstream canonical plan",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    _require_directory(
        UPSTREAM_CONFIRMATION_PATH.parent,
        0o555,
        "upstream confirmation commitment",
        ("commitment.json",),
    )
    confirmation_bound = BoundFile(
        str(UPSTREAM_CONFIRMATION_PATH),
        UPSTREAM_CONFIRMATION_PATH,
        UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
        "upstream confirmation commitment",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    try:
        plan = load_exact_json(plan_bound.text(), "upstream canonical plan")
        if (
            plan.get("audit") != upstream.CANONICAL_PLAN_AUDIT
            or plan.get("canonical") is not True
            or plan.get("plan_path") != str(UPSTREAM_PLAN_PATH)
        ):
            raise ValueError("upstream canonical plan header mismatch")
        source_contract, source_hashes = verify_upstream_source_snapshot(
            plan, repo_root=repo_root
        )
        if plan.get("board_rows_sha256") != UPSTREAM_BOARD_ROWS_SHA256:
            raise ValueError("upstream board-row identity mismatch")
        if stable_json_sha256(plan.get("board")) != UPSTREAM_CANONICAL_BOARD_SHA256:
            raise ValueError("upstream canonical board hash mismatch")
        commitment = plan.get("confirmation_commitment")
        if (
            type(commitment) is not dict
            or commitment.get("path") != str(UPSTREAM_CONFIRMATION_PATH)
            or commitment.get("sha256") != UPSTREAM_CONFIRMATION_COMMITMENT_SHA256
        ):
            raise ValueError("upstream confirmation commitment binding mismatch")
        commitment_document = load_exact_json(
            confirmation_bound.text(), "upstream confirmation commitment"
        )
        if not type_strict_equal(commitment.get("document"), commitment_document):
            raise ValueError("upstream confirmation commitment bytes differ from plan")
        validate_confirmation_generator_contract(plan, source_contract, source_hashes)
        expected_fit = {
            "seed": upstream.FIT_SEED,
            "rank": upstream.RANK,
            "quota": upstream.FIT_QUOTA,
            "updates": upstream.CANONICAL_UPDATES,
            "batch_size": upstream.CANONICAL_BATCH,
            "lr": upstream.CANONICAL_LR,
            "weight_decay": upstream.CANONICAL_WEIGHT_DECAY,
        }
        for name, value in expected_fit.items():
            if not type_strict_equal(plan.get("fit_budget", {}).get(name), value):
                raise ValueError(f"upstream fit budget changed: {name}")
        if (
            plan.get("shard_count") != upstream.CANONICAL_FEATURE_SHARDS
            or len(plan.get("shards", ())) != upstream.CANONICAL_FEATURE_SHARDS
            or plan.get("fit_artifact") != str(UPSTREAM_ROOT / "fit" / "motor.pt")
        ):
            raise ValueError("upstream plan shard or output contract mismatch")
        plan_bound.verify()
        confirmation_bound.verify()
        return (
            plan_bound,
            confirmation_bound,
            plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
        )
    except Exception:
        plan_bound.close()
        confirmation_bound.close()
        raise


def load_hostile_review(
    recovery_source_commit,
    recovery_source_contract,
    executor_runtime_contract,
    review_path,
    review_sha256,
):
    expected_path = recovery_review_path(recovery_source_commit)
    _require_directory(
        expected_path.parent,
        0o555,
        "hostile review receipt",
        ("hostile_review.json",),
    )
    bound = BoundFile(
        review_path,
        expected_path,
        review_sha256,
        "hostile review receipt",
        required_mode=0o444,
        required_parent_mode=0o555,
    )
    try:
        review = load_exact_json(bound.text(), "hostile review receipt")
        expected_keys = {
            "audit",
            "decision",
            "recovery_executor_source_contract",
            "executor_runtime_contract",
            "upstream_plan_sha256",
            "normalization_contract",
            "allowed_transformation",
            "claim_boundary",
        }
        if set(review) != expected_keys:
            raise ValueError("hostile review receipt schema mismatch")
        if (
            review["audit"] != RECOVERY_REVIEW_AUDIT
            or review["decision"] != "GO"
            or not type_strict_equal(
                review["recovery_executor_source_contract"], recovery_source_contract
            )
            or not type_strict_equal(
                review["executor_runtime_contract"], executor_runtime_contract
            )
            or review["upstream_plan_sha256"] != UPSTREAM_PLAN_SHA256
            or not type_strict_equal(
                review["normalization_contract"], normalization_contract()
            )
            or not type_strict_equal(
                review["allowed_transformation"], ALLOWED_TRANSFORMATION
            )
            or review["claim_boundary"] != REVIEW_CLAIM_BOUNDARY
        ):
            raise ValueError("hostile review receipt does not authorize exact recovery")
        bound.verify()
        return bound, review
    except Exception:
        bound.close()
        raise


def _bind_frozen_inputs(plan):
    bounds = {}
    try:
        for name in ("checkpoint", "tokenizer", "episodes", "cycle"):
            descriptor = plan["frozen_inputs"][name]
            bounds[name] = BoundFile(
                descriptor["path"],
                Path(descriptor["path"]),
                descriptor["sha256"],
                f"upstream frozen {name}",
            )
        return bounds
    except Exception:
        for bound in bounds.values():
            bound.close()
        raise


def reconstruct_board(plan, frozen_bounds):
    tokenizer = Tokenizer.from_str(frozen_bounds["tokenizer"].text())
    rows, generated_board = upstream.generate_fit_rows(
        tokenizer,
        frozen_bounds["episodes"].text(),
        upstream.FIT_SEED,
        upstream.FIT_QUOTA,
    )
    proof, normalized_board = build_normalization_proof(
        generated_board, plan["board"], rows
    )
    if normalized_board != plan["board"]:
        raise ValueError("normalized board differs from upstream plan")
    if upstream.stable_json_sha256([row["prefix_sha256"] for row in rows]) != plan.get(
        "extraction_order_sha256"
    ):
        raise ValueError("upstream row order changed")
    control_labels, control = upstream.permuted_control_labels(rows)
    if not type_strict_equal(control, plan["fit_budget"]["control"]):
        raise ValueError("upstream shuffled control changed")
    _schedule, schedule_sha256 = upstream._batch_schedule(
        len(rows),
        plan["fit_budget"]["batch_size"],
        plan["fit_budget"]["updates"],
        plan["fit_budget"]["seed"],
    )
    if schedule_sha256 != plan["fit_budget"]["schedule_sha256"]:
        raise ValueError("upstream fit schedule changed")
    initial_state, initial_sha256 = upstream.initial_motor_state(plan["d_model"])
    if initial_sha256 != plan["fit_budget"]["initial_state_sha256"]:
        raise ValueError("upstream initial motor state changed")
    for bound in frozen_bounds.values():
        bound.verify()
    return {
        "tokenizer": tokenizer,
        "rows": rows,
        "normalization_proof": proof,
        "normalized_board": normalized_board,
        "control_labels": control_labels,
        "control": control,
        "initial_state": initial_state,
        "initial_state_sha256": initial_sha256,
    }


def bind_and_merge_upstream_shards(plan, rows, source_contract, source_hashes):
    bounds = []
    payloads = []
    try:
        for index, (descriptor, expected_sha256) in enumerate(
            zip(plan["shards"], UPSTREAM_SHARD_SHA256)
        ):
            expected_path = UPSTREAM_ROOT / f"shard_{index:02d}" / "features.pt"
            if descriptor.get("artifact") != str(expected_path):
                raise ValueError(f"upstream shard {index} path substitution")
            bound = BoundFile(
                descriptor["artifact"],
                expected_path,
                expected_sha256,
                f"upstream shard {index}",
                required_mode=0o444,
                required_parent_mode=0o555,
            )
            bounds.append(bound)
        for bound in bounds:
            payloads.append((bound.sha256, str(bound.path), safe_torch_load(bound)))
        expected_bindings = {
            "base_checkpoint_sha256": plan["frozen_inputs"]["checkpoint"]["sha256"],
            "tokenizer_sha256": plan["frozen_inputs"]["tokenizer"]["sha256"],
            "episodes_sha256": plan["frozen_inputs"]["episodes"]["sha256"],
            "cycle_sha256": plan["frozen_inputs"]["cycle"]["sha256"],
            "confirmation_commitment_sha256": UPSTREAM_CONFIRMATION_COMMITMENT_SHA256,
            "scientific_source_sha256": source_hashes,
        }
        features, merge = upstream.merge_feature_shards(
            payloads,
            rows,
            expected_bindings,
            source_contract,
            plan,
            UPSTREAM_PLAN_SHA256,
        )
        for bound in bounds:
            bound.verify()
        receipts = []
        for index, (bound, descriptor, payload) in enumerate(
            zip(bounds, plan["shards"], (item[2] for item in payloads))
        ):
            receipts.append(
                {
                    "shard_index": index,
                    "path": str(bound.path),
                    "sha256": bound.sha256,
                    "bytes": bound.identity[2],
                    "descriptor": descriptor,
                    "feature_payload_sha256": payload["feature_payload_sha256"],
                    "sentinel_payload_sha256": payload["sentinel_payload_sha256"],
                }
            )
        return bounds, features, merge, receipts, expected_bindings
    except Exception:
        for bound in bounds:
            bound.close()
        raise


def _review_binding(bound, review):
    return {"path": str(bound.path), "sha256": bound.sha256, "document": review}


def build_recovery_plan_document(
    recovery_source_commit,
    recovery_source_contract,
    executor_runtime_contract,
    review_bound,
    review,
    upstream_plan,
    upstream_source_contract,
    upstream_source_hashes,
    upstream_custody_snapshot,
    context,
    feature_merge,
    shard_receipts,
):
    root = recovery_root(recovery_source_commit)
    plan_path = root / "recovery_plan.json"
    generator_contract = upstream_plan["confirmation_commitment"]["document"][
        "generator_source_contract"
    ]
    document = {
        "audit": RECOVERY_PLAN_AUDIT,
        "recovery": True,
        "recovery_plan_path": str(plan_path),
        "recovery_executor_source_contract": recovery_source_contract,
        "executor_runtime_contract": executor_runtime_contract,
        "hostile_review_binding": _review_binding(review_bound, review),
        "upstream_protocol": {
            "source_contract": upstream_source_contract,
            "scientific_source_sha256": upstream_source_hashes,
            "custody_snapshot": upstream_custody_snapshot,
            "plan_binding": {
                "path": str(UPSTREAM_PLAN_PATH),
                "sha256": UPSTREAM_PLAN_SHA256,
                "audit": upstream_plan["audit"],
            },
            "confirmation_commitment_binding": {
                "path": upstream_plan["confirmation_commitment"]["path"],
                "sha256": upstream_plan["confirmation_commitment"]["sha256"],
                "generator_source_contract": generator_contract,
            },
            "frozen_inputs": upstream_plan["frozen_inputs"],
            "shard_receipts": shard_receipts,
            "feature_merge": feature_merge,
        },
        "normalization_proof": context["normalization_proof"],
        "allowed_transformation": ALLOWED_TRANSFORMATION,
        "fit_contract": {
            "checkpoint_step": upstream_plan["checkpoint_step"],
            "d_model": upstream_plan["d_model"],
            "vocab_size": upstream_plan["vocab_size"],
            "zero_id": upstream_plan["zero_id"],
            "one_id": upstream_plan["one_id"],
            "board": upstream_plan["board"],
            "board_rows_sha256": upstream_plan["board_rows_sha256"],
            "extraction_order_sha256": upstream_plan["extraction_order_sha256"],
            "fit_budget": upstream_plan["fit_budget"],
            "runtime_contract": upstream_plan["runtime_contract"],
        },
        "output_contract": {
            "root": str(root),
            "fit_artifact": str(root / "fit" / "motor.pt"),
            "development_eval_artifact": str(
                root / "development_eval" / "evaluation.json"
            ),
            "confirmation_eval_artifact": str(
                root / "confirmation_eval" / "evaluation.json"
            ),
            "upstream_root_must_remain_untouched": str(UPSTREAM_ROOT),
        },
        "deserialization_contract": DESERIALIZATION_CONTRACT,
        "claim_boundary": RECOVERY_PLAN_CLAIM_BOUNDARY,
    }
    return canonical_json_document(document)


def validate_recovery_plan_document(
    observed,
    expected,
    recovery_source_commit,
):
    if type(observed) is not dict or set(observed) != RECOVERY_PLAN_KEYS:
        raise ValueError("recovery plan schema mismatch")
    if (
        observed.get("audit") != RECOVERY_PLAN_AUDIT
        or observed.get("recovery") is not True
        or not type_strict_equal(observed, expected)
    ):
        raise ValueError(
            "recovery plan differs from independently reconstructed contract"
        )
    root = recovery_root(recovery_source_commit)
    if observed["output_contract"]["root"] != str(root):
        raise ValueError("recovery output root mismatch")
    for name in (
        "fit_artifact",
        "development_eval_artifact",
        "confirmation_eval_artifact",
    ):
        path = Path(observed["output_contract"][name])
        if path == UPSTREAM_ROOT or UPSTREAM_ROOT in path.parents:
            raise ValueError("recovery output aliases the old canonical root")
        if root not in path.parents:
            raise ValueError("recovery output escapes immutable recovery root")
    if not type_strict_equal(
        observed["allowed_transformation"], ALLOWED_TRANSFORMATION
    ):
        raise ValueError("recovery plan admits extra transformations")


def _publish_recovery_plan(root, document):
    if root.exists() or root.is_symlink() or not root.is_absolute():
        raise FileExistsError("recovery root must be a new exact absolute path")
    root.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(root.parent)
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise ValueError("recovery parent is not a regular non-symlink directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(root.parent, directory_flags)
    try:
        opened_parent = os.fstat(parent_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise ValueError("recovery parent changed during exclusive publication")
        try:
            os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                "recovery root raced or already exists; refusing replacement"
            ) from exc
        root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        opened_root = os.fstat(root_fd)
        linked_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(linked_root.st_mode)
            or stat.S_ISLNK(linked_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino)
            != (linked_root.st_dev, linked_root.st_ino)
        ):
            raise ValueError("exclusive recovery root identity mismatch")
    except Exception:
        os.close(parent_fd)
        raise
    try:
        for name in ("fit", "development_eval", "confirmation_eval"):
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        )
        plan_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        plan_flags |= getattr(os, "O_NOFOLLOW", 0)
        plan_fd = os.open("recovery_plan.json", plan_flags, 0o600, dir_fd=root_fd)
        with os.fdopen(plan_fd, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
            os.fchmod(sink.fileno(), 0o444)
            os.fsync(sink.fileno())
        if set(os.listdir(root_fd)) != {
            "recovery_plan.json",
            "fit",
            "development_eval",
            "confirmation_eval",
        }:
            raise ValueError("exclusive recovery root gained unexpected children")
        os.fsync(root_fd)
        os.fchmod(root_fd, 0o555)
        os.fsync(root_fd)
        current_parent = os.lstat(root.parent)
        current_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current_parent.st_dev, current_parent.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) or (current_root.st_dev, current_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise RuntimeError("recovery publication path changed before sealing")
        os.fsync(parent_fd)
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def recovery_fit_state(root):
    directory = Path(root) / "fit"
    observed = os.lstat(directory)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise ValueError("recovery fit is not a regular directory")
    mode = stat.S_IMODE(observed.st_mode)
    children = {item.name for item in directory.iterdir()}
    if mode == 0o700 and not children:
        return "empty"
    if children != {"motor.pt"}:
        raise ValueError("recovery fit directory is not closed-world")
    artifact = directory / "motor.pt"
    artifact_stat = os.lstat(artifact)
    if (
        not stat.S_ISREG(artifact_stat.st_mode)
        or stat.S_ISLNK(artifact_stat.st_mode)
        or artifact_stat.st_nlink != 1
    ):
        raise ValueError("recovery fit artifact identity mismatch")
    artifact_mode = stat.S_IMODE(artifact_stat.st_mode)
    if mode == 0o700 and artifact_mode == 0o600:
        return "interrupted"
    if artifact_mode != 0o444:
        raise ValueError("recovery fit artifact mode mismatch")
    if mode == 0o700:
        return "recoverable"
    if mode == 0o555:
        return "sealed"
    raise ValueError("recovery fit directory mode mismatch")


def _open_recovery_fit_directory(directory, required_mode):
    path = Path(directory)
    before = os.lstat(path)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != required_mode
    ):
        raise ValueError("recovery fit directory identity or mode mismatch")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(opened):
        os.close(descriptor)
        raise RuntimeError("recovery fit directory changed during binding")
    return descriptor, _stat_identity(opened)


def discard_interrupted_recovery_fit(out):
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("interrupted recovery artifact path is not exact")
    directory_fd, directory_identity = _open_recovery_fit_directory(path.parent, 0o700)
    try:
        if os.listdir(directory_fd) != [path.name]:
            raise ValueError("interrupted recovery fit is not closed-world")
        observed = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_nlink != 1
        ):
            raise ValueError("interrupted recovery fit artifact identity mismatch")
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        if os.listdir(directory_fd):
            raise RuntimeError("interrupted recovery fit cleanup did not become empty")
        current = os.stat(path.parent, follow_symlinks=False)
        if (
            current.st_dev != directory_identity["device"]
            or current.st_ino != directory_identity["inode"]
            or stat.S_IMODE(current.st_mode) != 0o700
        ):
            raise RuntimeError("recovery fit directory was substituted during cleanup")
    finally:
        os.close(directory_fd)


def publish_recovery_torch(out, value):
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("recovery artifact path is not exact")
    directory_fd, directory_identity = _open_recovery_fit_directory(path.parent, 0o700)
    try:
        if os.listdir(directory_fd):
            raise FileExistsError("recovery fit directory is not empty")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        artifact_fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(artifact_fd, "w+b", closefd=False) as sink:
                created = os.fstat(artifact_fd)
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_nlink != 1
                    or stat.S_IMODE(created.st_mode) != 0o600
                ):
                    raise RuntimeError("exclusive recovery artifact creation failed")
                torch.save(value, sink)
                sink.flush()
                os.fsync(artifact_fd)
                linked = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    _stat_identity(linked) != _stat_identity(os.fstat(artifact_fd))
                    or linked.st_nlink != 1
                    or os.listdir(directory_fd) != [path.name]
                ):
                    raise RuntimeError("recovery artifact changed before publication")
                sink.seek(0)
                digest = hashlib.sha256()
                for block in iter(lambda: sink.read(1024 * 1024), b""):
                    digest.update(block)
                os.fchmod(artifact_fd, 0o444)
                os.fsync(artifact_fd)
                published = os.stat(
                    path.name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    published.st_dev != created.st_dev
                    or published.st_ino != created.st_ino
                    or published.st_nlink != 1
                    or stat.S_IMODE(published.st_mode) != 0o444
                    or os.listdir(directory_fd) != [path.name]
                ):
                    raise RuntimeError(
                        "recovery artifact publication identity mismatch"
                    )
            os.fsync(directory_fd)
        finally:
            os.close(artifact_fd)
        current_directory = os.stat(path.parent, follow_symlinks=False)
        if (
            current_directory.st_dev != directory_identity["device"]
            or current_directory.st_ino != directory_identity["inode"]
            or stat.S_IMODE(current_directory.st_mode) != 0o700
        ):
            raise RuntimeError("recovery fit directory changed during publication")
        return digest.hexdigest()
    finally:
        os.close(directory_fd)


def seal_recovery_fit(out):
    path = Path(out)
    if not path.is_absolute() or path.name != "motor.pt":
        raise ValueError("recovery artifact path is not exact")
    directory_fd, directory_identity = _open_recovery_fit_directory(path.parent, 0o700)
    try:
        if os.listdir(directory_fd) != [path.name]:
            raise ValueError("recovery fit cannot seal a non-closed-world directory")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        artifact_fd = os.open(path.name, flags, dir_fd=directory_fd)
        try:
            artifact = os.fstat(artifact_fd)
            linked = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                _stat_identity(artifact) != _stat_identity(linked)
                or not stat.S_ISREG(artifact.st_mode)
                or artifact.st_nlink != 1
                or stat.S_IMODE(artifact.st_mode) != 0o444
            ):
                raise ValueError("recovery fit artifact cannot be sealed")
            os.fsync(artifact_fd)
        finally:
            os.close(artifact_fd)
        os.fsync(directory_fd)
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
        current = os.stat(path.parent, follow_symlinks=False)
        current_artifact = os.stat(
            path.name, dir_fd=directory_fd, follow_symlinks=False
        )
        if (
            current.st_dev != directory_identity["device"]
            or current.st_ino != directory_identity["inode"]
            or stat.S_IMODE(current.st_mode) != 0o555
            or os.listdir(directory_fd) != [path.name]
            or current_artifact.st_dev != artifact.st_dev
            or current_artifact.st_ino != artifact.st_ino
            or current_artifact.st_nlink != 1
            or stat.S_IMODE(current_artifact.st_mode) != 0o444
        ):
            raise RuntimeError("recovery fit directory changed while sealing")
    finally:
        os.close(directory_fd)


def validate_recovery_layout(root, *, fit_state="empty"):
    _require_directory(
        root,
        0o555,
        "recovery root",
        ("recovery_plan.json", "fit", "development_eval", "confirmation_eval"),
    )
    plan = root / "recovery_plan.json"
    observed = os.lstat(plan)
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o444
        or observed.st_nlink != 1
    ):
        raise ValueError("recovery plan file identity mismatch")
    observed_fit_state = recovery_fit_state(root)
    if observed_fit_state != fit_state:
        raise ValueError(
            f"recovery fit state mismatch: expected {fit_state}, got {observed_fit_state}"
        )
    _require_directory(root / "development_eval", 0o700, "recovery development", ())
    _require_directory(root / "confirmation_eval", 0o700, "recovery confirmation", ())


def require_recovery_cuda_runtime(upstream_plan):
    device = upstream.require_canonical_cuda_runtime()
    observed = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    expected = upstream_plan["runtime_contract"]["artifact_runtime"]
    if not type_strict_equal(observed, expected):
        raise RuntimeError("recovery H100 runtime differs from frozen upstream runtime")
    return device


def prepare_context(args):
    executor_runtime_contract = capture_executor_runtime_contract()
    recovery_source_contract = build_recovery_executor_source_contract(
        args.recovery_source_commit,
        args.recovery_source_manifest_sha256,
    )
    review_bound, review = load_hostile_review(
        args.recovery_source_commit,
        recovery_source_contract,
        executor_runtime_contract,
        args.hostile_review,
        args.hostile_review_sha256,
    )
    upstream_plan_bound = None
    confirmation_bound = None
    frozen_bounds = {}
    shard_bounds = []
    try:
        (
            upstream_plan_bound,
            confirmation_bound,
            upstream_plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
        ) = load_upstream_plan()
        frozen_bounds = _bind_frozen_inputs(upstream_plan)
        board_context = reconstruct_board(upstream_plan, frozen_bounds)
        shard_bounds, features, feature_merge, receipts, expected_bindings = (
            bind_and_merge_upstream_shards(
                upstream_plan,
                board_context["rows"],
                source_contract,
                source_hashes,
            )
        )
        expected_plan = build_recovery_plan_document(
            args.recovery_source_commit,
            recovery_source_contract,
            executor_runtime_contract,
            review_bound,
            review,
            upstream_plan,
            source_contract,
            source_hashes,
            upstream_custody_snapshot,
            board_context,
            feature_merge,
            receipts,
        )
        for bound in (
            review_bound,
            upstream_plan_bound,
            confirmation_bound,
            *frozen_bounds.values(),
            *shard_bounds,
        ):
            bound.verify()
        return {
            "recovery_source_contract": recovery_source_contract,
            "executor_runtime_contract": executor_runtime_contract,
            "review_bound": review_bound,
            "review": review,
            "upstream_plan_bound": upstream_plan_bound,
            "confirmation_bound": confirmation_bound,
            "upstream_plan": upstream_plan,
            "upstream_source_contract": source_contract,
            "upstream_source_hashes": source_hashes,
            "upstream_custody_snapshot": upstream_custody_snapshot,
            "frozen_bounds": frozen_bounds,
            "board_context": board_context,
            "shard_bounds": shard_bounds,
            "features": features,
            "feature_merge": feature_merge,
            "shard_receipts": receipts,
            "expected_bindings": expected_bindings,
            "expected_recovery_plan": expected_plan,
        }
    except Exception:
        if upstream_plan_bound is not None:
            upstream_plan_bound.close()
        if confirmation_bound is not None:
            confirmation_bound.close()
        review_bound.close()
        for bound in frozen_bounds.values():
            bound.close()
        for bound in shard_bounds:
            bound.close()
        raise


def close_context(context):
    context["review_bound"].close()
    context["upstream_plan_bound"].close()
    context["confirmation_bound"].close()
    for bound in context["frozen_bounds"].values():
        bound.close()
    for bound in context["shard_bounds"]:
        bound.close()


def verify_context_bindings(context, recovery_plan_bound=None):
    bounds = [
        context["review_bound"],
        context["upstream_plan_bound"],
        context["confirmation_bound"],
        *context["frozen_bounds"].values(),
        *context["shard_bounds"],
    ]
    if recovery_plan_bound is not None:
        bounds.insert(0, recovery_plan_bound)
    for bound in bounds:
        bound.verify()
    observed_runtime = capture_executor_runtime_contract()
    if not type_strict_equal(observed_runtime, context["executor_runtime_contract"]):
        raise RuntimeError("recovery executor runtime changed after binding")


def _plan(args):
    context = prepare_context(args)
    try:
        root = recovery_root(args.recovery_source_commit)
        raw_root = os.fspath(args.recovery_root)
        raw_plan = os.fspath(args.recovery_plan)
        if raw_root != str(root) or raw_plan != str(root / "recovery_plan.json"):
            raise ValueError(
                "caller recovery root or plan aliases immutable derived path"
            )
        validate_recovery_plan_document(
            context["expected_recovery_plan"],
            context["expected_recovery_plan"],
            args.recovery_source_commit,
        )
        _publish_recovery_plan(root, context["expected_recovery_plan"])
        validate_recovery_layout(root, fit_state="empty")
        print(
            json.dumps(
                {
                    "audit": RECOVERY_PLAN_AUDIT,
                    "plan": str(root / "recovery_plan.json"),
                    "sha256": sha256_file(root / "recovery_plan.json"),
                    "gpu_status": "NO-GO-until-this-exact-plan-and-source-remain-reviewed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        close_context(context)


def _legacy_candidate_from_fit_payload(fit_payload):
    if type(fit_payload) is not dict or set(fit_payload) != LEGACY_PAYLOAD_KEYS:
        raise ValueError("recovery fit payload schema mismatch")
    return {
        "audit": upstream.CANONICAL_FIT_AUDIT,
        "canonical": True,
        **fit_payload,
    }


def _require_type_strict(value, expected, label):
    if not type_strict_equal(value, expected):
        raise ValueError(f"legacy payload type-strict mismatch: {label}")


def _validate_legacy_state_types(state, label):
    expected_keys = {"down.weight", "down.bias", "up.weight", "up.bias"}
    if type(state) is not collections.OrderedDict or set(state) != expected_keys:
        raise ValueError(f"legacy payload state type mismatch: {label}")
    for name, tensor in state.items():
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"legacy payload tensor type mismatch: {label}.{name}")


def _validate_legacy_fit_report_types(report, fit_budget, label):
    expected_keys = {
        "updates",
        "batch_size",
        "lr",
        "weight_decay",
        "schedule_sha256",
        "first_loss",
        "final_loss",
        "min_loss",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError(f"legacy payload fit report type mismatch: {label}")
    for name in ("updates", "batch_size", "lr", "weight_decay"):
        _require_type_strict(report[name], fit_budget[name], f"{label}.{name}")
    _require_type_strict(
        report["schedule_sha256"], fit_budget["schedule_sha256"], f"{label}.schedule"
    )
    for name in ("first_loss", "final_loss", "min_loss"):
        if type(report[name]) is not float or not math.isfinite(report[name]):
            raise ValueError(f"legacy payload loss type mismatch: {label}.{name}")


def _validate_legacy_linear_types(report):
    expected_keys = {
        "train_rows",
        "test_rows",
        "test_correct",
        "test_accuracy",
        "schedule_sha256",
        "claim_boundary",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError("legacy payload linear diagnostic type mismatch")
    for name in ("train_rows", "test_rows", "test_correct"):
        if type(report[name]) is not int:
            raise ValueError(f"legacy payload linear count type mismatch: {name}")
    if type(report["test_accuracy"]) is not float or not math.isfinite(
        report["test_accuracy"]
    ):
        raise ValueError("legacy payload linear accuracy type mismatch")
    for name in ("schedule_sha256", "claim_boundary"):
        if type(report[name]) is not str:
            raise ValueError(f"legacy payload linear string type mismatch: {name}")


def _expected_teacher_evidence(fit_payload, context, device):
    plan = context["upstream_plan"]
    treatment = upstream.CarryMotor(plan["d_model"], plan["fit_budget"]["rank"]).to(
        device
    )
    shuffled = upstream.CarryMotor(plan["d_model"], plan["fit_budget"]["rank"]).to(
        device
    )
    treatment.load_state_dict(fit_payload["treatment"], strict=True)
    shuffled.load_state_dict(fit_payload["shuffled"], strict=True)
    treatment.eval()
    shuffled.eval()
    return upstream.canonical_fit_teacher_forced_evidence(
        context["features"],
        context["board_context"]["rows"],
        context["board_context"]["control_labels"],
        treatment,
        shuffled,
        context["feature_merge"]["teacher_metric_feature_payload_sha256"],
        device,
    )


def validate_legacy_payload_type_strict(fit_payload, context, device):
    if type(fit_payload) is not dict or set(fit_payload) != LEGACY_PAYLOAD_KEYS:
        raise ValueError("legacy recovery fit payload schema mismatch")
    plan = context["upstream_plan"]
    fit_budget = plan["fit_budget"]
    expected_bindings = context["expected_bindings"]
    parameter_count = (
        plan["d_model"] * fit_budget["rank"]
        + fit_budget["rank"]
        + 2 * fit_budget["rank"]
        + 2
    )
    expected_exact = {
        "plan_sha256": UPSTREAM_PLAN_SHA256,
        "base_checkpoint_sha256": expected_bindings["base_checkpoint_sha256"],
        "tokenizer_sha256": expected_bindings["tokenizer_sha256"],
        "episodes_sha256": expected_bindings["episodes_sha256"],
        "cycle_sha256": expected_bindings["cycle_sha256"],
        "confirmation_commitment_sha256": expected_bindings[
            "confirmation_commitment_sha256"
        ],
        "scientific_source_sha256": context["upstream_source_hashes"],
        "source_contract": context["upstream_source_contract"],
        "checkpoint_step": plan["checkpoint_step"],
        "d_model": plan["d_model"],
        "rank": fit_budget["rank"],
        "parameter_count": parameter_count,
        "extract_batch": plan["runtime_contract"]["extract_batch"],
        "feature_shard_merge": context["feature_merge"],
        "deployment_logit_dtype": context["features"]["deployment_logit_dtype"],
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "initial_state_sha256": fit_budget["initial_state_sha256"],
        "board": context["board_context"]["normalized_board"],
        "control": context["board_context"]["control"],
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }
    for name, expected in expected_exact.items():
        _require_type_strict(fit_payload[name], expected, name)
    for arm in ("treatment", "shuffled"):
        _validate_legacy_state_types(fit_payload[arm], arm)
        state_sha256 = fit_payload[f"{arm}_state_sha256"]
        if type(state_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", state_sha256
        ):
            raise ValueError(f"legacy payload state receipt type mismatch: {arm}")
        _validate_legacy_fit_report_types(fit_payload[f"{arm}_fit"], fit_budget, arm)
    _validate_legacy_linear_types(fit_payload["linear_diagnostic"])
    expected_evidence = _expected_teacher_evidence(fit_payload, context, device)
    _require_type_strict(
        fit_payload["fit_feature_metrics"],
        expected_evidence,
        "fit_feature_metrics",
    )
    covered = set(expected_exact) | {
        "treatment",
        "shuffled",
        "treatment_state_sha256",
        "shuffled_state_sha256",
        "treatment_fit",
        "shuffled_fit",
        "linear_diagnostic",
        "fit_feature_metrics",
    }
    if covered != LEGACY_PAYLOAD_KEYS:
        raise AssertionError("legacy payload type validator coverage drift")


def validate_recovery_fit_bundle(
    bundle,
    recovery_plan,
    recovery_plan_sha256,
    context,
    device,
):
    if type(bundle) is not dict or set(bundle) != RECOVERY_FIT_KEYS:
        raise ValueError("v9 recovery fit schema mismatch")
    if bundle.get("audit") != RECOVERY_FIT_AUDIT or bundle.get("recovery") is not True:
        raise ValueError("fit is not a v9 recovery artifact")
    if "canonical" in bundle or bundle.get("audit") == upstream.CANONICAL_FIT_AUDIT:
        raise ValueError("recovery fit may never publish as v8 canonical")
    expected = {
        "recovery_plan_sha256": recovery_plan_sha256,
        "recovery_executor_source_contract": recovery_plan[
            "recovery_executor_source_contract"
        ],
        "executor_runtime_contract": recovery_plan["executor_runtime_contract"],
        "upstream_protocol_source_contract": recovery_plan["upstream_protocol"][
            "source_contract"
        ],
        "upstream_plan_binding": recovery_plan["upstream_protocol"]["plan_binding"],
        "upstream_shard_receipts": recovery_plan["upstream_protocol"]["shard_receipts"],
        "normalization_proof": recovery_plan["normalization_proof"],
        "allowed_transformation": ALLOWED_TRANSFORMATION,
        "deserialization_contract": DESERIALIZATION_CONTRACT,
        "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
    }
    for name, value in expected.items():
        if not type_strict_equal(bundle.get(name), value):
            raise ValueError(f"recovery fit provenance mismatch: {name}")
    validate_legacy_payload_type_strict(bundle["fit_payload"], context, device)
    legacy = _legacy_candidate_from_fit_payload(bundle["fit_payload"])
    upstream._validate_motor_bundle_against_replayed_features(
        legacy,
        context["expected_bindings"],
        context["upstream_source_hashes"],
        context["upstream_source_contract"],
        UPSTREAM_PLAN_SHA256,
        context["upstream_plan"],
        context["features"],
        context["feature_merge"],
        device,
    )


def _build_fit_payload(context, checkpoint, device):
    plan = context["upstream_plan"]
    board_context = context["board_context"]
    features = context["features"]
    fit_budget = plan["fit_budget"]
    cfg = model_module.GPTConfig(**checkpoint["cfg"])
    if (
        checkpoint.get("step") != plan["checkpoint_step"]
        or int(cfg.n_loop) != 1
        or int(cfg.d_model) != plan["d_model"]
        or int(cfg.vocab_size) != plan["vocab_size"]
    ):
        raise ValueError("checkpoint differs from frozen upstream fit contract")
    torch.manual_seed(fit_budget["seed"])
    torch.cuda.manual_seed_all(fit_budget["seed"])
    initial_state, initial_state_sha256 = upstream.initial_motor_state(plan["d_model"])
    if (
        initial_state_sha256 != board_context["initial_state_sha256"]
        or initial_state_sha256 != fit_budget["initial_state_sha256"]
    ):
        raise ValueError("fit-time initial motor state differs from frozen plan")
    treatment, treatment_fit = upstream.fit_motor(
        features,
        features["labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
    )
    shuffled, shuffled_fit = upstream.fit_motor(
        features,
        board_context["control_labels"],
        initial_state,
        device,
        fit_budget["updates"],
        fit_budget["batch_size"],
        fit_budget["lr"],
        fit_budget["weight_decay"],
        fit_budget["seed"],
    )
    if treatment_fit["schedule_sha256"] != shuffled_fit["schedule_sha256"]:
        raise RuntimeError("recovery arms used different fit schedules")
    linear = upstream.fit_linear_diagnostic(features, features["labels"], device)
    treatment = treatment.to(device).eval()
    shuffled = shuffled.to(device).eval()
    evidence = upstream.canonical_fit_teacher_forced_evidence(
        features,
        board_context["rows"],
        board_context["control_labels"],
        treatment,
        shuffled,
        context["feature_merge"]["teacher_metric_feature_payload_sha256"],
        device,
    )
    treatment = treatment.cpu()
    shuffled = shuffled.cpu()
    expected_bindings = {
        key: value
        for key, value in context["expected_bindings"].items()
        if key != "scientific_source_sha256"
    }
    return {
        "plan_sha256": UPSTREAM_PLAN_SHA256,
        **expected_bindings,
        "scientific_source_sha256": context["upstream_source_hashes"],
        "source_contract": context["upstream_source_contract"],
        "checkpoint_step": checkpoint.get("step"),
        "d_model": int(cfg.d_model),
        "rank": fit_budget["rank"],
        "parameter_count": treatment.parameter_count(),
        "extract_batch": plan["runtime_contract"]["extract_batch"],
        "feature_shard_merge": context["feature_merge"],
        "deployment_logit_dtype": features["deployment_logit_dtype"],
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "initial_state_sha256": board_context["initial_state_sha256"],
        "treatment": treatment.state_dict(),
        "shuffled": shuffled.state_dict(),
        "treatment_state_sha256": upstream.tensor_state_sha256(treatment.state_dict()),
        "shuffled_state_sha256": upstream.tensor_state_sha256(shuffled.state_dict()),
        "board": board_context["normalized_board"],
        "control": board_context["control"],
        "treatment_fit": treatment_fit,
        "shuffled_fit": shuffled_fit,
        "linear_diagnostic": linear,
        "fit_feature_metrics": evidence,
        "claim_boundary": upstream.CANONICAL_FIT_CLAIM_BOUNDARY,
    }


def _fit(args):
    context = prepare_context(args)
    recovery_plan_bound = None
    try:
        root = recovery_root(args.recovery_source_commit)
        if os.fspath(args.recovery_root) != str(root):
            raise ValueError("fit recovery root aliases immutable derived path")
        fit_state = recovery_fit_state(root)
        validate_recovery_layout(root, fit_state=fit_state)
        expected_plan_path = root / "recovery_plan.json"
        recovery_plan_bound = BoundFile(
            args.recovery_plan,
            expected_plan_path,
            args.recovery_plan_sha256,
            "recovery plan",
            required_mode=0o444,
            required_parent_mode=0o555,
        )
        recovery_plan = load_exact_json(recovery_plan_bound.text(), "recovery plan")
        validate_recovery_plan_document(
            recovery_plan,
            context["expected_recovery_plan"],
            args.recovery_source_commit,
        )
        out = Path(recovery_plan["output_contract"]["fit_artifact"])
        if out != root / "fit" / "motor.pt" or UPSTREAM_ROOT in out.parents:
            raise ValueError("recovery fit output aliases forbidden upstream root")
        device = require_recovery_cuda_runtime(context["upstream_plan"])
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"], "before recovery fit handling"
        )
        if fit_state == "interrupted":
            discard_interrupted_recovery_fit(out)
            assert_upstream_custody_unchanged(
                context["upstream_custody_snapshot"],
                "after interrupted recovery cleanup",
            )
            fit_state = "empty"
            validate_recovery_layout(root, fit_state="empty")
        if fit_state in {"recoverable", "sealed"}:
            published = BoundFile(
                str(out),
                out,
                sha256_file(out),
                "existing v9 recovery fit",
                required_mode=0o444,
                required_parent_mode=0o700 if fit_state == "recoverable" else 0o555,
            )
            try:
                replay = safe_torch_load(published)
                validate_recovery_fit_bundle(
                    replay,
                    recovery_plan,
                    recovery_plan_bound.sha256,
                    context,
                    device,
                )
                verify_context_bindings(context, recovery_plan_bound)
                published.verify()
            finally:
                published.close()
            if fit_state == "recoverable":
                assert_upstream_custody_unchanged(
                    context["upstream_custody_snapshot"],
                    "immediately before recovered publication sealing",
                )
                seal_recovery_fit(out)
                assert_upstream_custody_unchanged(
                    context["upstream_custody_snapshot"],
                    "immediately after recovered publication sealing",
                )
            validate_recovery_layout(root, fit_state="sealed")
            print(
                json.dumps(
                    {
                        "audit": RECOVERY_FIT_AUDIT,
                        "artifact": str(out),
                        "sha256": sha256_file(out),
                        "existing_validated": True,
                        "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if fit_state != "empty":
            raise ValueError("recovery fit state is not publishable")
        checkpoint = safe_torch_load(context["frozen_bounds"]["checkpoint"])
        fit_payload = _build_fit_payload(context, checkpoint, device)
        bundle = {
            "audit": RECOVERY_FIT_AUDIT,
            "recovery": True,
            "recovery_plan_sha256": recovery_plan_bound.sha256,
            "recovery_executor_source_contract": context["recovery_source_contract"],
            "executor_runtime_contract": context["executor_runtime_contract"],
            "upstream_protocol_source_contract": context["upstream_source_contract"],
            "upstream_plan_binding": recovery_plan["upstream_protocol"]["plan_binding"],
            "upstream_shard_receipts": context["shard_receipts"],
            "normalization_proof": context["board_context"]["normalization_proof"],
            "allowed_transformation": ALLOWED_TRANSFORMATION,
            "deserialization_contract": DESERIALIZATION_CONTRACT,
            "fit_payload": fit_payload,
            "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
        }
        validate_recovery_fit_bundle(
            bundle,
            recovery_plan,
            recovery_plan_bound.sha256,
            context,
            device,
        )
        verify_context_bindings(context, recovery_plan_bound)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately before recovery artifact publication",
        )
        published_sha256 = publish_recovery_torch(out, bundle)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately after recovery artifact publication",
        )
        validate_recovery_layout(root, fit_state="recoverable")
        published = BoundFile(
            str(out),
            out,
            published_sha256,
            "published v9 recovery fit",
            required_mode=0o444,
            required_parent_mode=0o700,
        )
        try:
            replay = safe_torch_load(published)
            validate_recovery_fit_bundle(
                replay,
                recovery_plan,
                recovery_plan_bound.sha256,
                context,
                device,
            )
            verify_context_bindings(context, recovery_plan_bound)
            published.verify()
        finally:
            published.close()
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately before recovery artifact sealing",
        )
        seal_recovery_fit(out)
        assert_upstream_custody_unchanged(
            context["upstream_custody_snapshot"],
            "immediately after recovery artifact sealing",
        )
        validate_recovery_layout(root, fit_state="sealed")
        final = BoundFile(
            str(out),
            out,
            published_sha256,
            "sealed v9 recovery fit",
            required_mode=0o444,
            required_parent_mode=0o555,
        )
        try:
            final.verify()
        finally:
            final.close()
        print(
            json.dumps(
                {
                    "audit": RECOVERY_FIT_AUDIT,
                    "artifact": str(out),
                    "sha256": published_sha256,
                    "claim_boundary": RECOVERY_FIT_CLAIM_BOUNDARY,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if recovery_plan_bound is not None:
            recovery_plan_bound.close()
        close_context(context)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--recovery-source-commit", required=True)
    common.add_argument("--recovery-source-manifest-sha256", required=True)
    common.add_argument("--hostile-review", required=True)
    common.add_argument("--hostile-review-sha256", required=True)
    common.add_argument("--recovery-root", required=True)
    common.add_argument("--recovery-plan", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", parents=[common])
    plan.set_defaults(func=_plan)
    fit = commands.add_parser("fit", parents=[common])
    fit.add_argument("--recovery-plan-sha256", required=True)
    fit.set_defaults(func=_fit)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
