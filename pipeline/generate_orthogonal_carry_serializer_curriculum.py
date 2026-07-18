#!/usr/bin/env python3
"""Build and verify the CPU-only OCSC curriculum bundle.

The bundle contains a 24,000-row orthogonal carry/serializer curriculum, a
matched 24,000-row IID control, lossless tokenizer receipts, pair-preserving
packs, exact six-cell schedules, prompt commitments, and a complete audit. It
does not launch training or evaluate a model.
"""

from __future__ import annotations

if (
    __name__ == "__main__"
    and globals().get("_OCSC_BOOTSTRAP_EXECUTION_CONTEXT") is None
):
    raise SystemExit(
        "OCSC contract rejected: direct generator execution is forbidden; "
        "use the externally hash-verified runner"
    )

import argparse
import base64
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
import errno
import fcntl
from fractions import Fraction
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import stat
import struct
import sys
import sysconfig
import time
from typing import Any, Callable, Iterable
import zlib


_BOOTSTRAP_EXECUTION_CONTEXT = globals().get("_OCSC_BOOTSTRAP_EXECUTION_CONTEXT")
ROOT = Path(
    _BOOTSTRAP_EXECUTION_CONTEXT["checkout_root_path"]
    if isinstance(_BOOTSTRAP_EXECUTION_CONTEXT, dict)
    else os.path.abspath(os.fspath(Path(__file__).parent.parent))
)


SCHEMA = "shohin-ocsc-v1"
WIDTHS = (3, 4, 5, 6, 7)
CONTEXT_COUNT = 5
TRANSITION_CELLS_PER_WIDTH = 900
TRANSITION_ROWS_PER_WIDTH = 4_500
TRANSITION_ROWS = 22_500
SERIALIZER_ROWS_PER_WIDTH = 300
SERIALIZER_ROWS = 1_500
CORPUS_ROWS = 24_000
BATCH_SLOTS = 8
SEQUENCE_LENGTH = 256
MAIN_POSITIONS_PER_UPDATE = BATCH_SLOTS * SEQUENCE_LENGTH
PACKS_PER_CORPUS = 4_875
UPDATES_PER_ARM = 5_120
REPEATED_PACKS = UPDATES_PER_ARM - PACKS_PER_CORPUS
RUN_CELLS = ("A", "B", "M00", "M10", "M01", "M11")
PAIRED_SEEDS = (2026071801, 2026071802, 2026071803)
SERIALIZER_SLICES = ("add_c0", "add_c1", "sub_c0")
ROLE_CELL_COUNTS = {
    "initial": 200,
    "interior": 400,
    "terminal_add": 200,
    "terminal_sub": 100,
}
ROLE_ROW_COUNTS = {
    name: count * CONTEXT_COUNT for name, count in ROLE_CELL_COUNTS.items()
}
REGISTRY_COUNTS = {
    ("replay", "drs"): 640,
    ("replay", "non_dws"): 640,
    ("development", "drs"): 704,
    ("development", "non_dws"): 704,
    ("secret_confirmation", "drs"): 704,
    ("secret_confirmation", "non_dws"): 704,
}
REPLAY_ROWS = 1_280
EVALUATION_ROWS = 2_816
PROMPT_REGISTRY_ROWS = REPLAY_ROWS + EVALUATION_ROWS
KNOWN_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
KNOWN_TOKENIZER_BYTES = 2_309_567
FACTORIAL_DEVELOPMENT_HELDOUT_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
FACTORIAL_DEVELOPMENT_BOARD_SHA256 = (
    "f2fcfcae41b55aa82dd360036bd8c9c00ed6e4ca442debec1c85ed282e50dfe1"
)
PARENT_CHECKPOINT_SHA256 = (
    "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
)
REPLAY_V5_PATH = (
    "artifacts/eval_history/"
    "digitwise_factorial_v4_four_arm_replay_v5_de45ace_20260718.json"
)
REPLAY_V5_SHA256 = "d08b17a4fdaf031205ca445bb01f72a2983010e5eb929e6f13ab46409fa5c42f"
WIDTH_SWEEP_V2_PATH = (
    "artifacts/eval_history/drs_terminal_width_sweep_v2_w2_w10_20260718_mps.json"
)
WIDTH_SWEEP_V2_SHA256 = (
    "db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520"
)
RESIDUAL_SWAP_PATH = (
    "artifacts/eval_history/drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json"
)
RESIDUAL_SWAP_SHA256 = (
    "4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f"
)
DIGITWISE_PROTOCOL_SHA256 = (
    "37cd76751eb4146f85268d6c0e44d946eb353ee03605ceb25f4bda97e4c00813"
)
DIGITWISE_PROTOCOL_BYTES = 8_690
PUBLICATION_SIGNATURE_DOMAIN = b"R12-OCSC-PREPUBLICATION-COMMITMENT-v1\x00"
INDEPENDENT_REVIEW_SIGNATURE_DOMAIN = b"R12-OCSC-INDEPENDENT-REVIEW-v1\x00"
STAGING_IDENTITY_DOMAIN = b"R12-OCSC-STAGING-IDENTITY-v1\x00"
PUBLICATION_LEASE_DOMAIN = b"R12-OCSC-PUBLICATION-LEASE-v1\x00"
LINUX_LUSTRE_QUALIFICATION_SIGNATURE_DOMAIN = (
    b"R12-OCSC-LINUX-LUSTRE-QUALIFICATION-v3\x00"
)
QUALIFICATION_EVENT_SIGNATURE_DOMAIN = b"R12-OCSC-QUALIFICATION-EVENT-v1\x00"
QUALIFICATION_BROKER_REQUEST_SIGNATURE_DOMAIN = (
    b"R12-OCSC-QUALIFICATION-BROKER-REQUEST-v1\x00"
)
QUALIFICATION_BROKER_RECEIPT_SIGNATURE_DOMAIN = (
    b"R12-OCSC-QUALIFICATION-BROKER-RECEIPT-v1\x00"
)
TRUSTED_PUBLICATION_KEYS = {
    "production": "4d67229fe6b9c62f95ae9208284735fcb4c410e2efb2e4f8a6d935b762887e08",
    "test": "ff0be94c519b0bb590e73c26e618dbf52a850c48928d4fedc615af5d96549217",
}
TRUSTED_INDEPENDENT_REVIEW_KEYS = {
    # Production stays fail-closed until a real external review key is frozen.
    "production": None,
    "test": "9453be42182991d6980d9ee5280c2e95f8514d9b76f52ff33f3b879f747c7e70",
}
TRUSTED_LINUX_LUSTRE_QUALIFICATION_KEYS = {
    # Production stays fail-closed until an external Lustre reviewer key is frozen.
    "production": None,
    "test": "43046bfe4092b3e94994eada15dcc20d8aaa07b658fd3954eb8e0efb8bdca5de",
}
LINUX_LUSTRE_QUALIFICATION_CHECKS = (
    "external_bootstrap_source_bound",
    "all_consumed_source_bytes_pinned",
    "runtime_closure_complete_before_action",
    "real_tokenizer_registry_consumed",
    "cross_node_distinct_hosts",
    "same_lustre_mount_and_output_inode",
    "production_broker_transfer_complete",
    "publication_path_complete",
    "renameat2_noreplace_real",
    "descriptor_relative_io",
    "file_fsync_after_chmod",
    "stage_and_parent_fsync",
    "kernel_flock_live_exclusion",
    "stale_lease_observed_without_mutation",
    "all_crash_evidence_permanently_retained",
    "canonical_death_recovery",
    "collision_rejected",
    "path_substitution_rejected",
    "coherent_forgery_foreign_child_preserved",
    "partial_child_preserved",
    "foreign_replacement_preserved",
    "runtime_shadow_import_rejected",
    "injected_io_fail_closed",
    "strict_full_readback",
    "report_marker_receipt_event_derived",
    "permanent_evidence_inventory_recorded",
)
QUALIFICATION_CRASH_POINTS = (
    "stage-created-before-journal",
    "journal-durable-before-first-artifact",
    "partial-artifact-write",
    "stage-fsync-before-rename",
    "canonical-before-parent-fsync",
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX128_RE = re.compile(r"^[0-9a-f]{128}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
FIELD_RE = re.compile(r"(?:^|;)(op|w|p|c|a|b|r|z)=([^;]+)")
INDEPENDENT_STATE_RE = re.compile(
    r"^dws:op=(add|sub);w=([0-9]+);p=([0-9]+);c=([01]);"
    r"a=([0-9]+);b=([0-9]+);r=([0-9]+);z=([01])$"
)
ARTIFACT_NAMES = (
    "ocsc_train.jsonl",
    "iid_control_train.jsonl",
    "relational_pairs.jsonl",
    "replay_prompts.jsonl",
    "tokenization_receipt.jsonl",
    "packs.jsonl",
    "schedule.jsonl",
    "commitments.json",
    "audit_report.json",
    "manifest.json",
)
SOURCE_PATHS = (
    "R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md",
    "pipeline/generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/test_generate_orthogonal_carry_serializer_curriculum.py",
    "pipeline/run_orthogonal_carry_serializer_curriculum.py",
    "train/digitwise_protocol.py",
)
REVIEWED_SOURCE_PATHS = SOURCE_PATHS[:4]
ORACLE_SOURCE_PATH = SOURCE_PATHS[4]
RUNTIME_DISTRIBUTIONS = ("cryptography", "tokenizers")
RUNTIME_STDLIB_MODULES = (
    "_ctypes",
    "_hashlib",
    "_json",
    "_random",
    "_struct",
    "argparse",
    "base64",
    "collections",
    "ctypes",
    "dataclasses",
    "errno",
    "fcntl",
    "fractions",
    "hashlib",
    "importlib",
    "importlib.machinery",
    "importlib.metadata",
    "json",
    "math",
    "os",
    "pathlib",
    "platform",
    "random",
    "re",
    "socket",
    "stat",
    "struct",
    "sysconfig",
    "time",
    "typing",
    "zlib",
)
RUN_CELL_CONTRACT = {
    "A": {
        "corpus": "iid_control",
        "field_weights": "uniform",
        "local_relation": False,
        "serializer_relation": False,
    },
    "B": {
        "corpus": "ocsc",
        "field_weights": "uniform",
        "local_relation": False,
        "serializer_relation": False,
    },
    "M00": {
        "corpus": "ocsc",
        "field_weights": "carry_serializer_v1",
        "local_relation": False,
        "serializer_relation": False,
    },
    "M10": {
        "corpus": "ocsc",
        "field_weights": "carry_serializer_v1",
        "local_relation": True,
        "serializer_relation": False,
    },
    "M01": {
        "corpus": "ocsc",
        "field_weights": "carry_serializer_v1",
        "local_relation": False,
        "serializer_relation": True,
    },
    "M11": {
        "corpus": "ocsc",
        "field_weights": "carry_serializer_v1",
        "local_relation": True,
        "serializer_relation": True,
    },
}
FIELD_WEIGHT_UNITS = {
    "default": 4,
    "op": 2,
    "w": 2,
    "p": 5,
    "c": 8,
    "a": 2,
    "b": 2,
    "r": 6,
    "z": 5,
    "answer": 8,
}
FIELD_IDS = {
    "padding": 0,
    "prompt": 1,
    "default": 2,
    "op": 3,
    "w": 4,
    "p": 5,
    "c": 6,
    "a": 7,
    "b": 8,
    "r": 9,
    "z": 10,
    "answer": 11,
}
SLOT_PAYLOAD_LAYOUT = (
    ("token_ids", 4),
    ("attention_mask", 1),
    ("completion_mask", 1),
    ("field_ids", 1),
    ("raw_weight_units", 1),
    ("position_ids", 2),
)
SLOT_PAYLOAD_BYTES = SEQUENCE_LENGTH * sum(width for _, width in SLOT_PAYLOAD_LAYOUT)
HIDDEN_LEAF_DOMAIN = b"R12-OCSC-HIDDEN-LEAF-v1\x00"
HIDDEN_NODE_DOMAIN = b"R12-OCSC-HIDDEN-NODE-v1\x00"
HIDDEN_ROOT_DOMAIN = b"R12-OCSC-HIDDEN-ROOT-v1\x00"
HIDDEN_CUSTODIAN_DOMAIN = b"R12-OCSC-CUSTODIAN-OPENING-v1\x00"
HIDDEN_INITIAL_SITES_PER_WIDTH = 50
HIDDEN_NONINITIAL_SITE_COUNTS = {
    "interior": 40,
    "terminal_add": 25,
    "terminal_sub": 15,
}
HIDDEN_PREFIX_VARIANTS = ("anchor", "intervention")


class ContractError(ValueError):
    """Raised when an input or artifact violates the frozen contract."""


@dataclass(frozen=True)
class FileSnapshot:
    """Bytes and filesystem identity captured through one pinned directory fd."""

    payload: bytes
    resolved_path: str
    metadata: os.stat_result
    parent_resolved_path: str
    parent_metadata: os.stat_result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def _bootstrap_source_snapshot(relative: str) -> FileSnapshot | None:
    """Return bytes already pinned by the external runner, never a reopened file."""

    context = _BOOTSTRAP_EXECUTION_CONTEXT
    if context is None:
        return None
    snapshots = context.get("source_snapshots") if isinstance(context, dict) else None
    source_fds = context.get("source_fds") if isinstance(context, dict) else None
    checkout_root_fd = (
        context.get("checkout_root_fd") if isinstance(context, dict) else None
    )
    checkout_root_path = (
        context.get("checkout_root_path") if isinstance(context, dict) else None
    )
    if (
        not isinstance(snapshots, dict)
        or not isinstance(source_fds, dict)
        or type(checkout_root_fd) is not int
        or checkout_root_fd < 0
        or not isinstance(checkout_root_path, str)
        or relative not in snapshots
        or relative not in source_fds
    ):
        raise ContractError("bootstrap source snapshot is missing: " + relative)
    record = snapshots[relative]
    if (
        not isinstance(record, dict)
        or set(record) != {"payload", "contract"}
        or not isinstance(record["payload"], bytes)
        or not isinstance(record["contract"], dict)
    ):
        raise ContractError("bootstrap source snapshot mismatch: " + relative)
    contract = record["contract"]
    metadata = os.fstat(source_fds[relative])
    live_fd = None
    parent_fd = None
    try:
        live_fd = _open_relative_components(
            checkout_root_fd,
            relative,
            "bootstrap source " + relative,
            directory=False,
        )
        path_metadata = os.fstat(live_fd)
        parent_relative = relative.rpartition("/")[0]
        parent_fd = (
            _open_relative_components(
                checkout_root_fd,
                parent_relative,
                "bootstrap source parent " + relative,
                directory=True,
            )
            if parent_relative
            else os.dup(checkout_root_fd)
        )
        parent_metadata = os.fstat(parent_fd)
    except (KeyError, OSError, TypeError, ContractError) as error:
        raise ContractError("bootstrap source parent mismatch: " + relative) from error
    finally:
        if live_fd is not None:
            os.close(live_fd)
        if parent_fd is not None:
            os.close(parent_fd)
    expected_path = checkout_root_path + "/" + relative
    snapshot = FileSnapshot(
        payload=record["payload"],
        resolved_path=contract["resolved_path"],
        metadata=metadata,
        parent_resolved_path=(
            checkout_root_path + "/" + parent_relative
            if parent_relative
            else checkout_root_path
        ),
        parent_metadata=parent_metadata,
    )
    live_contract = {
        "resolved_path": snapshot.resolved_path,
        "bytes": len(snapshot.payload),
        "sha256": snapshot.sha256,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "hard_links": metadata.st_nlink,
    }
    if (
        live_contract != contract
        or contract.get("resolved_path") != expected_path
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_size != len(snapshot.payload)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
        or path_metadata.st_mode != metadata.st_mode
        or path_metadata.st_nlink != metadata.st_nlink
        or path_metadata.st_uid != metadata.st_uid
    ):
        raise ContractError("bootstrap source identity mismatch: " + relative)
    return snapshot


@dataclass
class PublicationLease:
    """A deterministic lease inode held under an exclusive kernel lock."""

    name: str
    descriptor: int
    metadata: os.stat_result
    record: dict
    created: bool


def _file_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def recursively_type_strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            recursively_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            recursively_type_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _lexical_absolute_path(path: Path, label: str) -> str:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.startswith("/") or raw == "/":
        raise ContractError("{} must be one non-root absolute path".format(label))
    try:
        raw.encode("ascii")
    except UnicodeEncodeError as error:
        raise ContractError("{} path must be ASCII".format(label)) from error
    if raw.endswith("/") or any(
        component in {"", ".", ".."} for component in raw.split("/")[1:]
    ):
        raise ContractError("{} path contains an empty or dot component".format(label))
    return raw


def _open_absolute_components(
    path: Path,
    label: str,
    *,
    directory: bool,
) -> tuple[int, str]:
    absolute = _lexical_absolute_path(Path(path), label)
    try:
        descriptor = os.open(
            "/",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ContractError(
            "{} filesystem root is not readable".format(label)
        ) from error
    try:
        components = absolute.split("/")[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            if not final or directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                held = os.fstat(next_descriptor)
                entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if (
                    held.st_dev,
                    held.st_ino,
                    held.st_mode,
                ) != (
                    entry.st_dev,
                    entry.st_ino,
                    entry.st_mode,
                ):
                    raise ContractError(
                        "{} component changed during traversal".format(label)
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, absolute
    except OSError as error:
        os.close(descriptor)
        raise ContractError(
            "{} has a symlink, missing, or unreadable path component".format(label)
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_components(
    root_fd: int,
    relative: str,
    label: str,
    *,
    directory: bool,
) -> int:
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise ContractError("{} must be one relative path".format(label))
    components = tuple(relative.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise ContractError("{} contains an empty or dot component".format(label))
    descriptor = os.dup(root_fd)
    try:
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            if not final or directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                held = os.fstat(next_descriptor)
                entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if (
                    held.st_dev,
                    held.st_ino,
                    held.st_mode,
                ) != (
                    entry.st_dev,
                    entry.st_ino,
                    entry.st_mode,
                ):
                    raise ContractError(
                        "{} component changed during traversal".format(label)
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ContractError(
            "{} has a symlink, missing, or unreadable component".format(label)
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _validated_absolute_file_path(path: Path, label: str) -> Path:
    descriptor, absolute = _open_absolute_components(Path(path), label, directory=False)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("{} must be a regular file".format(label))
    finally:
        os.close(descriptor)
    return Path(absolute)


def _validated_absolute_directory_path(path: Path, label: str) -> Path:
    descriptor, absolute, _ = _open_pinned_directory(Path(path), label)
    os.close(descriptor)
    return Path(absolute)


def _open_pinned_directory(
    path: Path,
    label: str,
    *,
    exact_mode: int | None = None,
    reject_other_writes: bool = False,
) -> tuple[int, str, os.stat_result]:
    descriptor, absolute = _open_absolute_components(Path(path), label, directory=True)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractError("{} must be a directory".format(label))
        mode = stat.S_IMODE(metadata.st_mode)
        if exact_mode is not None and mode != exact_mode:
            raise ContractError("{} must be mode {:04o}".format(label, exact_mode))
        if reject_other_writes and mode & 0o022:
            raise ContractError(
                "{} must not be writable by group or other".format(label)
            )
        return descriptor, absolute, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_path_matches_fd(
    requested_path: Path,
    resolved_path: str,
    descriptor: int,
    label: str,
    *,
    exact_mode: int | None = None,
) -> os.stat_result:
    fresh_descriptor = None
    try:
        pinned = os.fstat(descriptor)
        requested_absolute = _lexical_absolute_path(requested_path, label)
        if requested_absolute != resolved_path:
            raise ContractError("{} lexical path changed".format(label))
        fresh_descriptor, fresh_absolute = _open_absolute_components(
            requested_path, label, directory=True
        )
        fresh = os.fstat(fresh_descriptor)
    except (OSError, ContractError) as error:
        raise ContractError("{} changed during the operation".format(label)) from error
    finally:
        if fresh_descriptor is not None:
            os.close(fresh_descriptor)
    if (
        not stat.S_ISDIR(pinned.st_mode)
        or fresh_absolute != resolved_path
        or not _same_inode(pinned, fresh)
    ):
        raise ContractError("{} changed during the operation".format(label))
    if exact_mode is not None and stat.S_IMODE(pinned.st_mode) != exact_mode:
        raise ContractError("{} mode changed during the operation".format(label))
    return pinned


def read_file_snapshot(
    path: Path,
    label: str,
    *,
    exact_mode: int | None = None,
    custody_root: bool = False,
) -> FileSnapshot:
    """Read one stable inode through a pinned parent directory descriptor."""

    path = Path(path)
    if path.name in {"", ".", ".."}:
        raise ContractError("{} path mismatch".format(label))
    parent_fd, parent_resolved, parent_before = _open_pinned_directory(
        path.parent,
        "{} custody root directory".format(label)
        if custody_root
        else "{} parent".format(label),
        exact_mode=0o555 if custody_root else None,
    )
    file_fd = None
    try:
        if custody_root:
            try:
                inventory_before = sorted(os.listdir(parent_fd))
            except OSError as error:
                raise ContractError(
                    "{} custody root cannot be listed".format(label)
                ) from error
            if inventory_before != [path.name]:
                raise ContractError(
                    "{} custody root must contain exactly one file".format(label)
                )
        try:
            file_fd = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise ContractError("{} is not readable".format(label)) from error
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractError(
                "{} must be a regular file with exactly one hard link".format(label)
            )
        mode = stat.S_IMODE(before.st_mode)
        if exact_mode is not None and mode != exact_mode:
            raise ContractError("{} mode must be {:04o}".format(label, exact_mode))
        if exact_mode is None and (mode & 0o022 or mode & 0o7000):
            raise ContractError("{} has unsafe mode {:04o}".format(label, mode))
        blocks = []
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(file_fd)
        try:
            entry_after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ContractError("{} changed during snapshot".format(label)) from error
        if (
            _file_state(before) != _file_state(after)
            or _file_state(before) != _file_state(entry_after)
            or len(payload) != before.st_size
        ):
            raise ContractError("{} changed during snapshot".format(label))
        if custody_root:
            try:
                inventory_after = sorted(os.listdir(parent_fd))
            except OSError as error:
                raise ContractError(
                    "{} custody root changed during snapshot".format(label)
                ) from error
            if inventory_after != [path.name]:
                raise ContractError(
                    "{} custody root changed during snapshot".format(label)
                )
        parent_after = _assert_directory_path_matches_fd(
            path.parent,
            parent_resolved,
            parent_fd,
            "{} custody root directory".format(label)
            if custody_root
            else "{} parent".format(label),
            exact_mode=0o555 if custody_root else None,
        )
        if _directory_state(parent_before) != _directory_state(parent_after):
            raise ContractError("{} parent changed during snapshot".format(label))
        return FileSnapshot(
            payload=payload,
            resolved_path=str(Path(parent_resolved) / path.name),
            metadata=before,
            parent_resolved_path=parent_resolved,
            parent_metadata=parent_before,
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


class PinnedBundle:
    """Keep one immutable bundle directory and every artifact inode pinned."""

    def __init__(self, path: Path, *, directory_fd: int | None = None):
        self.requested_path = Path(path)
        self.directory_fd = -1
        self.file_fds: dict[str, int] = {}
        self.files: dict[str, FileSnapshot] = {}
        try:
            if directory_fd is None:
                (
                    self.directory_fd,
                    self.resolved_path,
                    self.metadata,
                ) = _open_pinned_directory(
                    self.requested_path,
                    "bundle directory",
                    exact_mode=0o555,
                )
            else:
                self.directory_fd = os.dup(directory_fd)
                self.resolved_path = _lexical_absolute_path(
                    self.requested_path, "bundle directory"
                )
                self.metadata = _assert_directory_path_matches_fd(
                    self.requested_path,
                    self.resolved_path,
                    self.directory_fd,
                    "bundle directory",
                    exact_mode=0o555,
                )
            inventory = sorted(os.listdir(self.directory_fd))
            if inventory != sorted(ARTIFACT_NAMES):
                raise ContractError("bundle inventory mismatch")
            for name in ARTIFACT_NAMES:
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=self.directory_fd,
                )
                self.file_fds[name] = descriptor
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) != 0o444
                ):
                    raise ContractError(
                        "bundle artifact identity/mode mismatch: " + name
                    )
                blocks = []
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    blocks.append(block)
                payload = b"".join(blocks)
                if len(payload) != before.st_size:
                    raise ContractError(
                        "bundle artifact changed during snapshot: " + name
                    )
                self.files[name] = FileSnapshot(
                    payload=payload,
                    resolved_path=str(Path(self.resolved_path) / name),
                    metadata=before,
                    parent_resolved_path=self.resolved_path,
                    parent_metadata=self.metadata,
                )
            self.assert_unchanged()
        except BaseException:
            self.close()
            raise

    def assert_unchanged(self) -> None:
        if self.directory_fd < 0:
            raise ContractError("bundle snapshot is closed")
        current_directory = _assert_directory_path_matches_fd(
            self.requested_path,
            self.resolved_path,
            self.directory_fd,
            "bundle directory",
            exact_mode=0o555,
        )
        if _directory_state(current_directory) != _directory_state(self.metadata):
            raise ContractError("bundle directory changed during verification")
        if sorted(os.listdir(self.directory_fd)) != sorted(ARTIFACT_NAMES):
            raise ContractError("bundle inventory changed during verification")
        for name in ARTIFACT_NAMES:
            try:
                descriptor_state = os.fstat(self.file_fds[name])
                entry_state = os.stat(
                    name,
                    dir_fd=self.directory_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ContractError(
                    "bundle artifact changed during verification: " + name
                ) from error
            expected = self.files[name].metadata
            if _file_state(descriptor_state) != _file_state(expected) or _file_state(
                entry_state
            ) != _file_state(expected):
                raise ContractError(
                    "bundle artifact changed during verification: " + name
                )

    def assert_full_readback(self) -> None:
        """Reread every byte through the descriptors retained since snapshot."""

        self.assert_unchanged()
        for name in ARTIFACT_NAMES:
            descriptor = self.file_fds[name]
            expected = self.files[name]
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                blocks = []
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    blocks.append(block)
                payload = b"".join(blocks)
                descriptor_state = os.fstat(descriptor)
                entry_state = os.stat(
                    name,
                    dir_fd=self.directory_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ContractError(
                    "bundle artifact full readback failed: " + name
                ) from error
            if (
                payload != expected.payload
                or _file_state(descriptor_state) != _file_state(expected.metadata)
                or _file_state(entry_state) != _file_state(expected.metadata)
            ):
                raise ContractError(
                    "bundle artifact changed during full readback: " + name
                )
        self.assert_unchanged()

    def close(self) -> None:
        for descriptor in self.file_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.file_fds.clear()
        if self.directory_fd >= 0:
            try:
                os.close(self.directory_fd)
            except OSError:
                pass
            self.directory_fd = -1

    def __enter__(self) -> PinnedBundle:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def custody_snapshot_contract(snapshot: FileSnapshot) -> dict:
    return {
        "resolved_path": snapshot.resolved_path,
        "bytes": len(snapshot.payload),
        "sha256": snapshot.sha256,
        "file_device": snapshot.metadata.st_dev,
        "file_inode": snapshot.metadata.st_ino,
        "custody_root_path": snapshot.parent_resolved_path,
        "custody_root_device": snapshot.parent_metadata.st_dev,
        "custody_root_inode": snapshot.parent_metadata.st_ino,
    }


def custody_input_snapshots(
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    *,
    snapshots: dict[str, FileSnapshot] | None = None,
    label_prefix: str,
) -> dict[str, FileSnapshot]:
    """Capture or validate the one snapshot set used by a complete operation."""

    paths = {
        "tokenizer": Path(tokenizer_path),
        "prompt_registry": Path(prompt_registry_path),
        "secret_confirmation_commitment": Path(confirmation_path),
    }
    if snapshots is None:
        snapshots = {
            label: read_file_snapshot(
                path,
                "{} {}".format(label_prefix, label),
                exact_mode=0o444,
                custody_root=True,
            )
            for label, path in paths.items()
        }
    if not isinstance(snapshots, dict) or set(snapshots) != set(paths):
        raise ContractError("custody input snapshot inventory mismatch")
    validated = {}
    for label, path in paths.items():
        snapshot = snapshots[label]
        if not isinstance(snapshot, FileSnapshot):
            raise ContractError("custody input snapshot type mismatch: " + label)
        resolved_path = _lexical_absolute_path(path, "custody input path " + label)
        if (
            snapshot.resolved_path != resolved_path
            or not stat.S_ISREG(snapshot.metadata.st_mode)
            or snapshot.metadata.st_nlink != 1
            or stat.S_IMODE(snapshot.metadata.st_mode) != 0o444
            or not stat.S_ISDIR(snapshot.parent_metadata.st_mode)
            or stat.S_IMODE(snapshot.parent_metadata.st_mode) != 0o555
            or snapshot.parent_resolved_path != str(Path(resolved_path).parent)
        ):
            raise ContractError("custody input snapshot identity mismatch: " + label)
        validated[label] = snapshot
    return validated


def manifest_custody_fields(prefix: str, contract: dict) -> dict:
    return {
        "{}_path".format(prefix): contract["resolved_path"],
        "{}_bytes".format(prefix): contract["bytes"],
        "{}_sha256".format(prefix): contract["sha256"],
        "{}_file_device".format(prefix): contract["file_device"],
        "{}_file_inode".format(prefix): contract["file_inode"],
        "{}_custody_root_path".format(prefix): contract["custody_root_path"],
        "{}_custody_root_device".format(prefix): contract["custody_root_device"],
        "{}_custody_root_inode".format(prefix): contract["custody_root_inode"],
    }


def _load_reviewed_digitwise_protocol() -> tuple[FileSnapshot, dict[str, Any]]:
    """Authenticate the exact oracle bytes before compiling any of them."""

    snapshot = _bootstrap_source_snapshot(ORACLE_SOURCE_PATH) or read_file_snapshot(
        ROOT / "train" / "digitwise_protocol.py",
        "reviewed digitwise protocol source",
    )
    if len(snapshot.payload) != DIGITWISE_PROTOCOL_BYTES or snapshot.sha256 != (
        DIGITWISE_PROTOCOL_SHA256
    ):
        raise ContractError("reviewed digitwise protocol source identity mismatch")
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": snapshot.resolved_path,
        "__name__": "_ocsc_reviewed_digitwise_protocol",
        "__package__": None,
    }
    code = compile(snapshot.payload, snapshot.resolved_path, "exec", dont_inherit=True)
    exec(code, namespace)
    required = {
        name: namespace.get(name)
        for name in (
            "apply_microstep",
            "canonical_state",
            "initial_state",
            "parse_state",
            "state_answer",
        )
    }
    if any(not callable(value) for value in required.values()):
        raise ContractError("reviewed digitwise protocol export mismatch")
    return snapshot, required


_REVIEWED_DIGITWISE_PROTOCOL_SNAPSHOT, _REVIEWED_ORACLE = (
    _load_reviewed_digitwise_protocol()
)
apply_microstep = _REVIEWED_ORACLE["apply_microstep"]
canonical_state = _REVIEWED_ORACLE["canonical_state"]
initial_state = _REVIEWED_ORACLE["initial_state"]
parse_state = _REVIEWED_ORACLE["parse_state"]
state_answer = _REVIEWED_ORACLE["state_answer"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return read_file_snapshot(path, "hashed file").sha256


def assert_finite_json(value: Any, label: str = "JSON value", path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("{} contains a nonfinite number at {}".format(label, path))
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError(
                    "{} contains a non-string key at {}".format(label, path)
                )
            assert_finite_json(child, label, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_json(child, label, "{}[{}]".format(path, index))


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    assert_finite_json(value)
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join(canonical_json_bytes(row, newline=True) for row in rows)


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def with_payload_hash(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = hash_json(result)
    return result


def _resolved_runtime_snapshot(path: Path, label: str) -> dict:
    resolved = Path(_lexical_absolute_path(Path(path), label))
    parent_fd, parent_resolved, _ = _open_pinned_directory(
        resolved.parent, label + " parent"
    )
    descriptor = None
    try:
        descriptor = os.open(
            resolved.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        entry = os.stat(resolved.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _file_state(before) != _file_state(
            entry
        ):
            raise ContractError(label + " runtime file identity mismatch")
        blocks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        final_entry = os.stat(resolved.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            len(payload) != before.st_size
            or _file_state(before) != _file_state(after)
            or _file_state(before) != _file_state(final_entry)
        ):
            raise ContractError(label + " changed during runtime snapshot")
        _assert_directory_path_matches_fd(
            resolved.parent, parent_resolved, parent_fd, label + " parent"
        )
        result = {
            "resolved_path": str(Path(parent_resolved) / resolved.name),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "mode": stat.S_IMODE(before.st_mode),
            "owner_uid": before.st_uid,
            "device": before.st_dev,
            "inode": before.st_ino,
            "hard_links": before.st_nlink,
        }
        if isinstance(_BOOTSTRAP_EXECUTION_CONTEXT, dict):
            execution = _BOOTSTRAP_EXECUTION_CONTEXT.get("contract")
            inventory = (
                execution.get("external_runtime_inventory")
                if isinstance(execution, dict)
                else None
            )
            external = (
                inventory.get("files", {}).get(result["resolved_path"])
                if isinstance(inventory, dict)
                else None
            )
            observed = {
                "path": result["resolved_path"],
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "mode": result["mode"],
                "owner_uid": result["owner_uid"],
                "device": result["device"],
                "inode": result["inode"],
                "hard_links": result["hard_links"],
            }
            if (
                not isinstance(external, dict)
                or {key: external.get(key) for key in observed} != observed
            ):
                raise ContractError(
                    label + " was not authenticated by the external runtime manifest"
                )
        return result
    except OSError as error:
        raise ContractError(label + " runtime file is not readable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _execution_file_contract(snapshot: FileSnapshot) -> dict:
    return {
        "resolved_path": snapshot.resolved_path,
        "bytes": len(snapshot.payload),
        "sha256": snapshot.sha256,
        "mode": stat.S_IMODE(snapshot.metadata.st_mode),
        "owner_uid": snapshot.metadata.st_uid,
        "device": snapshot.metadata.st_dev,
        "inode": snapshot.metadata.st_ino,
        "hard_links": snapshot.metadata.st_nlink,
    }


def _read_execution_descriptor(
    descriptor: int,
    expected: dict,
    label: str,
    *,
    checkout_root_fd: int | None = None,
    relative_path: str | None = None,
) -> dict:
    expected_keys = {
        "resolved_path",
        "bytes",
        "sha256",
        "mode",
        "owner_uid",
        "device",
        "inode",
        "hard_links",
    }
    if (
        type(descriptor) is not int
        or descriptor < 0
        or not isinstance(expected, dict)
        or set(expected) != expected_keys
        or not isinstance(expected["resolved_path"], str)
        or not expected["resolved_path"].startswith("/")
        or type(expected["bytes"]) is not int
        or expected["bytes"] <= 0
        or not isinstance(expected["sha256"], str)
        or not HEX64_RE.fullmatch(expected["sha256"])
        or any(
            type(expected[field]) is not int
            for field in (
                "mode",
                "owner_uid",
                "device",
                "inode",
                "hard_links",
            )
        )
    ):
        raise ContractError(label + " execution identity contract mismatch")
    try:
        before = os.fstat(descriptor)
        offset = 0
        blocks = []
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not block:
                break
            blocks.append(block)
            offset += len(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ContractError(label + " pinned execution bytes are unreadable") from error
    live_fd = None
    try:
        live_fd = (
            _open_relative_components(
                checkout_root_fd,
                relative_path,
                label + " live path",
                directory=False,
            )
            if checkout_root_fd is not None and relative_path is not None
            else _open_absolute_components(
                Path(expected["resolved_path"]), label + " live path", directory=False
            )[0]
        )
        live = os.fstat(live_fd)
    except (OSError, ContractError) as error:
        raise ContractError(label + " live path is unavailable") from error
    finally:
        if live_fd is not None:
            os.close(live_fd)
    descriptor_contract = {
        "resolved_path": expected["resolved_path"],
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": stat.S_IMODE(before.st_mode),
        "owner_uid": before.st_uid,
        "device": before.st_dev,
        "inode": before.st_ino,
        "hard_links": before.st_nlink,
    }
    if (
        _file_state(before) != _file_state(after)
        or descriptor_contract != expected
        or not stat.S_ISREG(live.st_mode)
        or live.st_nlink != 1
        or _file_state(live) != _file_state(before)
    ):
        raise ContractError(label + " source/runtime identity drifted")
    return descriptor_contract


def bootstrap_execution_contract(
    argv: list[str] | None = None,
    *,
    required: bool = False,
) -> dict:
    context = _BOOTSTRAP_EXECUTION_CONTEXT
    if context is None:
        if required:
            raise ContractError("source-bound CLI requires the external bootstrap")
        local = {
            "schema": "shohin-ocsc-bootstrap-execution-v2",
            "profile": "nonclaim-local-pytest",
            "source_bound": False,
            "qualification_authority": False,
            "production_authority": False,
        }
        return with_payload_hash(local, "payload_sha256")
    if not isinstance(context, dict) or set(context) != {
        "contract",
        "checkout_root_fd",
        "checkout_root_path",
        "source_fds",
        "source_snapshots",
        "runtime_fds",
        "runtime_snapshots",
    }:
        raise ContractError("bootstrap execution context mismatch")
    contract = context["contract"]
    expected_keys = {
        "schema",
        "profile",
        "source_bound",
        "qualification_authority",
        "production_authority",
        "bootstrap",
        "sources",
        "sources_sha256",
        "interpreter",
        "external_runtime_inventory",
        "generator_argv",
        "generator_argv_sha256",
        "payload_sha256",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != expected_keys
        or contract.get("schema") != "shohin-ocsc-bootstrap-execution-v2"
        or contract.get("profile") not in {"test", "qualification", "production"}
        or contract.get("source_bound") is not True
        or contract.get("qualification_authority") is not False
        or contract.get("production_authority") is not False
        or not isinstance(contract.get("bootstrap"), dict)
        or contract["bootstrap"].get("schema")
        != "shohin-ocsc-external-bootstrap-identity-v1"
        or contract["bootstrap"].get("authority") is not False
        or contract["bootstrap"].get("payload_sha256")
        != hash_json(
            {
                key: value
                for key, value in contract["bootstrap"].items()
                if key != "payload_sha256"
            }
        )
        or not isinstance(contract.get("sources"), dict)
        or set(contract.get("sources", {})) != set(SOURCE_PATHS)
        or contract.get("sources_sha256") != hash_json(contract.get("sources"))
        or not isinstance(contract.get("external_runtime_inventory"), dict)
        or contract["external_runtime_inventory"].get("schema")
        != "shohin-ocsc-external-runtime-inventory-v1"
        or contract["external_runtime_inventory"].get("payload_sha256")
        != hash_json(
            {
                key: value
                for key, value in contract["external_runtime_inventory"].items()
                if key != "payload_sha256"
            }
        )
        or not isinstance(contract.get("generator_argv"), list)
        or any(not isinstance(value, str) for value in contract["generator_argv"])
        or contract.get("generator_argv_sha256")
        != hash_json(contract["generator_argv"])
        or contract.get("payload_sha256")
        != hash_json(
            {key: value for key, value in contract.items() if key != "payload_sha256"}
        )
    ):
        raise ContractError("bootstrap execution contract mismatch")
    if (
        type(context["checkout_root_fd"]) is not int
        or context["checkout_root_fd"] < 0
        or not isinstance(context["checkout_root_path"], str)
        or not isinstance(context["source_fds"], dict)
        or not isinstance(context["source_snapshots"], dict)
        or set(context["source_fds"]) != set(SOURCE_PATHS)
        or set(context["source_snapshots"]) != set(SOURCE_PATHS)
    ):
        raise ContractError("bootstrap source inventory mismatch")
    if (
        not isinstance(context["runtime_fds"], dict)
        or not isinstance(context["runtime_snapshots"], dict)
        or not {"bootstrap", "interpreter", "runtime_manifest"}.issubset(
            context["runtime_fds"]
        )
        or set(context["runtime_snapshots"])
        != {"bootstrap", "interpreter", "runtime_manifest"}
    ):
        raise ContractError("bootstrap runtime inventory mismatch")
    checkout_root_fd = context["checkout_root_fd"]
    checkout_root_path = _lexical_absolute_path(
        Path(context["checkout_root_path"]), "bootstrap checkout root"
    )
    root_metadata = os.fstat(checkout_root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or checkout_root_path != contract["bootstrap"].get("checkout_root_path")
        or root_metadata.st_dev != contract["bootstrap"].get("checkout_root_device")
        or root_metadata.st_ino != contract["bootstrap"].get("checkout_root_inode")
    ):
        raise ContractError("bootstrap checkout root identity mismatch")
    for relative in SOURCE_PATHS:
        expected_path = checkout_root_path + "/" + relative
        source = contract["sources"].get(relative)
        snapshot = context["source_snapshots"].get(relative)
        if (
            not isinstance(source, dict)
            or source.get("resolved_path") != expected_path
            or not isinstance(snapshot, dict)
            or set(snapshot) != {"payload", "contract"}
            or not isinstance(snapshot.get("payload"), bytes)
            or snapshot.get("contract") != source
        ):
            raise ContractError("bootstrap reviewed source path mismatch: " + relative)
        descriptor_contract = _read_execution_descriptor(
            context["source_fds"][relative],
            source,
            "source " + relative,
            checkout_root_fd=checkout_root_fd,
            relative_path=relative,
        )
        if (
            descriptor_contract != source
            or sha256_bytes(snapshot["payload"]) != source["sha256"]
        ):
            raise ContractError("bootstrap source snapshot mismatch: " + relative)
    runtime_contracts = {
        "bootstrap": contract["bootstrap"]["executable"],
        "interpreter": contract["interpreter"],
        "runtime_manifest": contract["bootstrap"]["runtime_manifest"],
    }
    for runtime_name, runtime_contract in runtime_contracts.items():
        runtime_snapshot = context["runtime_snapshots"][runtime_name]
        if (
            not isinstance(runtime_snapshot, dict)
            or not isinstance(runtime_contract, dict)
            or set(runtime_snapshot) != {"payload", "contract"}
            or not isinstance(runtime_snapshot["payload"], bytes)
            or runtime_snapshot["contract"] != runtime_contract
            or sha256_bytes(runtime_snapshot["payload"])
            != runtime_contract.get("sha256")
            or _read_execution_descriptor(
                context["runtime_fds"][runtime_name],
                runtime_contract,
                "bootstrap runtime " + runtime_name,
            )
            != runtime_contract
        ):
            raise ContractError("bootstrap runtime identity drifted: " + runtime_name)
    if _lexical_absolute_path(Path(sys.executable), "Python executable") != contract[
        "interpreter"
    ].get("resolved_path"):
        raise ContractError("bootstrap Python executable path drifted")
    if argv is not None and list(argv) != contract["generator_argv"]:
        raise ContractError("bootstrap generator argv drifted")
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"),
        "bootstrap execution contract",
    )


def bootstrap_source_identity_contract() -> dict:
    execution = bootstrap_execution_contract()
    identity = {
        "schema": "shohin-ocsc-bootstrap-source-identity-v2",
        "profile": execution["profile"],
        "source_bound": execution["source_bound"],
        "qualification_authority": False,
        "production_authority": False,
    }
    if execution["source_bound"]:
        identity.update(
            {
                "sources": execution["sources"],
                "sources_sha256": execution["sources_sha256"],
                "external_bootstrap": execution["bootstrap"],
                "external_runtime_inventory": execution["external_runtime_inventory"],
                "interpreter": execution["interpreter"],
            }
        )
    return with_payload_hash(identity, "payload_sha256")


def _distribution_import_identity(distribution_name: str) -> tuple[Any, Path, Path]:
    """Resolve a package from its installed distribution, never a search-path shadow."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError(
            "required runtime distribution is unavailable: " + distribution_name
        ) from error
    distribution_root = _validated_absolute_directory_path(
        Path(distribution.locate_file("")),
        "runtime distribution root " + distribution_name,
    )
    trusted_spec = importlib.machinery.PathFinder.find_spec(
        distribution_name, [str(distribution_root)]
    )
    ambient_spec = importlib.machinery.PathFinder.find_spec(
        distribution_name, list(sys.path)
    )
    if (
        trusted_spec is None
        or trusted_spec.origin in {None, "built-in", "frozen"}
        or ambient_spec is None
        or ambient_spec.origin in {None, "built-in", "frozen"}
    ):
        raise ContractError(
            "runtime distribution import identity is unavailable: " + distribution_name
        )
    trusted_origin = _validated_absolute_file_path(
        Path(trusted_spec.origin),
        "trusted runtime import origin " + distribution_name,
    )
    ambient_origin = _validated_absolute_file_path(
        Path(ambient_spec.origin),
        "ambient runtime import origin " + distribution_name,
    )
    if ambient_origin != trusted_origin:
        raise ContractError(
            "runtime distribution shadow import rejected: " + distribution_name
        )
    existing = sys.modules.get(distribution_name)
    if existing is not None:
        existing_origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if existing_origin in {None, "built-in", "frozen"}:
            raise ContractError(
                "loaded runtime distribution identity mismatch: " + distribution_name
            )
        loaded_origin = _validated_absolute_file_path(
            Path(existing_origin),
            "loaded runtime import origin " + distribution_name,
        )
        if loaded_origin != trusted_origin:
            raise ContractError(
                "loaded runtime distribution shadow rejected: " + distribution_name
            )
    return distribution, distribution_root, trusted_origin


def _trusted_distribution_module(distribution_name: str) -> Any:
    _, _, trusted_origin = _distribution_import_identity(distribution_name)
    try:
        module = importlib.import_module(distribution_name)
    except ImportError as error:
        raise ContractError(
            "required runtime distribution cannot be imported: " + distribution_name
        ) from error
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    try:
        loaded_origin = _validated_absolute_file_path(
            Path(origin),
            "loaded runtime distribution " + distribution_name,
        )
    except (OSError, TypeError, ContractError) as error:
        raise ContractError(
            "loaded runtime distribution identity mismatch: " + distribution_name
        ) from error
    if loaded_origin != trusted_origin:
        raise ContractError(
            "loaded runtime distribution shadow rejected: " + distribution_name
        )
    return module


def _runtime_distribution_contract(distribution_name: str) -> dict:
    distribution, distribution_root, trusted_origin = _distribution_import_identity(
        distribution_name
    )
    files = distribution.files
    if files is None:
        raise ContractError(
            "runtime distribution has no installed file inventory: " + distribution_name
        )
    records = {}
    package_record_paths = set()
    native_files = {}
    for relative_entry in sorted(files, key=lambda entry: str(entry)):
        relative = Path(str(relative_entry))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "__pycache__" in relative.parts
            or relative.suffix == ".pyc"
        ):
            continue
        first = relative.parts[0]
        if first != distribution_name and not first.endswith(".dist-info"):
            continue
        candidate = Path(distribution.locate_file(relative_entry))
        try:
            resolved = _validated_absolute_file_path(
                candidate,
                "runtime distribution file {}:{}".format(distribution_name, relative),
            )
            resolved.relative_to(distribution_root)
        except (OSError, ValueError, ContractError) as error:
            raise ContractError(
                "runtime distribution file escapes its root: " + str(relative)
            ) from error
        if not resolved.is_file():
            continue
        record = _resolved_runtime_snapshot(
            resolved,
            "runtime distribution file {}:{}".format(distribution_name, relative),
        )
        relative_key = relative.as_posix()
        records[relative_key] = record
        if first == distribution_name:
            package_record_paths.add(relative_key)
        if resolved.suffix.lower() in {".a", ".dll", ".dylib", ".pyd", ".so"}:
            native_files[relative_key] = record
    package_root = distribution_root / distribution_name
    if not package_root.is_dir():
        raise ContractError(
            "runtime distribution package directory is unavailable: "
            + distribution_name
        )
    physical_package_paths = {
        path.relative_to(distribution_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if physical_package_paths != package_record_paths:
        raise ContractError(
            "runtime distribution file inventory drifted: " + distribution_name
        )
    if not records or trusted_origin.as_posix() not in {
        value["resolved_path"] for value in records.values()
    }:
        raise ContractError(
            "runtime distribution import origin is outside its inventory: "
            + distribution_name
        )
    contract = {
        "distribution": distribution_name,
        "version": distribution.version,
        "distribution_root": str(distribution_root),
        "trusted_import_origin": str(trusted_origin),
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records.values()),
        "native_files": native_files,
        "files_sha256": hash_json(records),
    }
    return with_payload_hash(contract, "payload_sha256")


def _runtime_module_contract(module_name: str, stdlib_root: Path) -> dict:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ContractError(
            "required stdlib module is unavailable: " + module_name
        ) from error
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin in {None, "built-in", "frozen"}:
        return {"kind": str(origin or "built-in"), "origin": str(origin)}
    try:
        resolved = _validated_absolute_file_path(
            Path(origin), "stdlib module " + module_name
        )
        resolved.relative_to(stdlib_root)
    except (OSError, ValueError, ContractError) as error:
        raise ContractError(
            "stdlib module resolved outside the bound stdlib: " + module_name
        ) from error
    return {
        "kind": "file",
        **_resolved_runtime_snapshot(resolved, "runtime stdlib module " + module_name),
    }


def _preload_consumed_distributions() -> None:
    InvalidSignature, Ed25519PublicKey = _trusted_cryptography_symbols()
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(TRUSTED_PUBLICATION_KEYS["test"])
        ).verify(b"\x00" * 64, b"OCSC runtime-closure preload")
    except InvalidSignature:
        pass
    try:
        ed25519 = importlib.import_module(
            "cryptography.hazmat.primitives.asymmetric.ed25519"
        )
        serialization = importlib.import_module(
            "cryptography.hazmat.primitives.serialization"
        )
        preload_private_key = ed25519.Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)
        preload_private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        preload_private_key.sign(b"OCSC runtime-closure signing preload")
    except (ImportError, ValueError) as error:
        raise ContractError(
            "trusted cryptography signing runtime is incomplete"
        ) from error
    host_fqdn = socket.getfqdn()
    if not host_fqdn:
        raise ContractError("qualification host identity is unavailable")
    _trusted_distribution_module("tokenizers")
    for module_name in (
        "tokenizers.decoders",
        "tokenizers.models",
        "tokenizers.pre_tokenizers",
    ):
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            raise ContractError(
                "trusted tokenizers runtime is incomplete: " + module_name
            ) from error


def _loaded_module_closure(
    stdlib_root: Path,
    distribution_roots: dict[str, Path],
) -> dict:
    modules = {}
    for module_name, module in sorted(sys.modules.items()):
        if module is None:
            modules[module_name] = {"kind": "none"}
            continue
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            modules[module_name] = {"kind": str(origin)}
            continue
        if origin is None:
            locations = getattr(spec, "submodule_search_locations", None)
            modules[module_name] = {
                "kind": "namespace-or-executed",
                "search_locations": (
                    sorted(
                        str(
                            _validated_absolute_directory_path(
                                Path(value),
                                "namespace search location " + module_name,
                            )
                        )
                        for value in locations
                    )
                    if locations is not None
                    else []
                ),
            }
            continue
        try:
            resolved = _validated_absolute_file_path(
                Path(origin), "loaded module " + module_name
            )
        except (OSError, ContractError) as error:
            raise ContractError(
                "loaded module origin is not resolvable: " + module_name
            ) from error
        owner = None
        try:
            relative = resolved.relative_to(stdlib_root)
            owner = {"kind": "stdlib", "relative_path": relative.as_posix()}
        except ValueError:
            for distribution_name, distribution_root in distribution_roots.items():
                try:
                    relative = resolved.relative_to(distribution_root)
                except ValueError:
                    continue
                owner = {
                    "kind": "distribution",
                    "distribution": distribution_name,
                    "relative_path": relative.as_posix(),
                }
                break
        if owner is None:
            bootstrap = bootstrap_execution_contract()
            reviewed = {
                source.get("resolved_path")
                for source in bootstrap.get("sources", {}).values()
                if isinstance(source, dict)
            }
            if str(resolved) in reviewed:
                owner = {"kind": "reviewed-execution-source"}
            else:
                raise ContractError(
                    "loaded module is outside the closed runtime: " + module_name
                )
        modules[module_name] = {
            **owner,
            "file": _resolved_runtime_snapshot(
                resolved, "loaded module " + module_name
            ),
        }
    return {
        "modules": modules,
        "module_count": len(modules),
        "modules_sha256": hash_json(modules),
    }


def _parse_linux_native_image_mappings(lines: list[str]) -> dict[str, dict[str, int]]:
    mappings = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        mapped = fields[5]
        if mapped.endswith(" (deleted)"):
            raise ContractError("a consumed native image was deleted")
        if not mapped.startswith("/"):
            continue
        device_parts = fields[3].split(":")
        try:
            if len(device_parts) != 2:
                raise ValueError
            mapped_device = os.makedev(
                int(device_parts[0], 16), int(device_parts[1], 16)
            )
            mapped_inode = int(fields[4], 10)
        except (OSError, ValueError) as error:
            raise ContractError("native image mapping identity is malformed") from error
        if mapped_inode <= 0:
            raise ContractError("native image mapping has no file identity")
        identity = {
            "mapped_device": mapped_device,
            "mapped_inode": mapped_inode,
        }
        previous = mappings.get(mapped)
        if previous is not None and previous != identity:
            raise ContractError("native image path maps multiple file identities")
        mappings[mapped] = identity
    return mappings


def _bound_linux_native_image(path: str, mapping: dict[str, int]) -> dict:
    snapshot = _resolved_runtime_snapshot(Path(path), "native image " + path)
    if (
        snapshot["device"] != mapping["mapped_device"]
        or snapshot["inode"] != mapping["mapped_inode"]
    ):
        raise ContractError(
            "native image path does not name the executed mapping identity"
        )
    return {**snapshot, **mapping}


def _linux_native_image_closure() -> dict:
    maps_path = Path("/proc/self/maps")
    try:
        lines = maps_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError("Linux native image inventory is unavailable") from error
    mappings = _parse_linux_native_image_mappings(lines)
    if not mappings:
        raise ContractError("Linux native image inventory is empty")
    files = {
        path: _bound_linux_native_image(path, mappings[path])
        for path in sorted(mappings)
    }
    return {
        "authority": "linux-proc-self-maps-executable-files",
        "complete_for_production": True,
        "files": files,
        "file_count": len(files),
        "files_sha256": hash_json(files),
    }


def _native_image_closure(source_bound: bool, profile: str) -> dict:
    if source_bound and sys.platform.startswith("linux"):
        return _linux_native_image_closure()
    if profile == "production":
        raise ContractError(
            "production runtime requires authoritative Linux native image closure"
        )
    return {
        "authority": "nonclaim-nonlinux-or-inprocess",
        "complete_for_production": False,
        "files": {},
        "file_count": 0,
        "files_sha256": hash_json({}),
    }


def runtime_closure_contract() -> dict:
    """Bind the executable Python, stdlib modules, and third-party code in use."""

    execution = bootstrap_execution_contract()
    try:
        interpreter_path = _validated_absolute_file_path(
            Path(
                execution["interpreter"]["resolved_path"]
                if execution["source_bound"]
                else getattr(sys, "_base_executable", sys.executable)
            ),
            "runtime Python interpreter",
        )
        stdlib_root = _validated_absolute_directory_path(
            Path(sysconfig.get_path("stdlib")), "runtime stdlib root"
        )
    except (OSError, TypeError, ContractError) as error:
        raise ContractError(
            "runtime interpreter or stdlib path is not resolvable"
        ) from error
    if not stdlib_root.is_dir():
        raise ContractError("runtime stdlib root is not a directory")
    source_bound = execution["source_bound"] is True
    if source_bound:
        _preload_consumed_distributions()
    distributions = {
        name: _runtime_distribution_contract(name) for name in RUNTIME_DISTRIBUTIONS
    }
    distribution_roots = {
        name: Path(contract["distribution_root"])
        for name, contract in distributions.items()
    }
    stdlib_modules = {
        name: _runtime_module_contract(name, stdlib_root)
        for name in RUNTIME_STDLIB_MODULES
    }
    loaded_modules = (
        _loaded_module_closure(stdlib_root, distribution_roots)
        if source_bound
        else {
            "modules": {},
            "module_count": 0,
            "modules_sha256": hash_json({}),
        }
    )
    native_images = _native_image_closure(source_bound, execution["profile"])
    libc_name, libc_version = platform.libc_ver()
    contract = {
        "schema": "shohin-ocsc-runtime-closure-v3",
        "execution_profile": execution["profile"],
        "source_bound": source_bound,
        "interpreter": (
            execution["interpreter"]
            if source_bound
            else _resolved_runtime_snapshot(
                interpreter_path, "runtime Python interpreter"
            )
        ),
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "hexversion": sys.hexversion,
            "isolated": bool(sys.flags.isolated),
            "safe_path": bool(sys.flags.safe_path),
            "no_user_site": bool(sys.flags.no_user_site),
            "no_site": bool(sys.flags.no_site),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "sys_platform": sys.platform,
            "machine": platform.machine(),
            "release": platform.release(),
            "version": platform.version(),
            "libc_name": libc_name,
            "libc_version": libc_version,
        },
        "stdlib": {
            "resolved_root": str(stdlib_root),
            "modules": stdlib_modules,
            "modules_sha256": hash_json(stdlib_modules),
        },
        "loaded_modules": loaded_modules,
        "distributions": distributions,
        "native_images": native_images,
        "external_preattestation": (
            execution["external_runtime_inventory"] if source_bound else None
        ),
        "pre_attestation_complete": source_bound,
        "production_closure_complete": (
            source_bound
            and execution["profile"] in {"qualification", "production"}
            and execution["external_runtime_inventory"]["held_file_count"] > 0
            and native_images["complete_for_production"] is True
        ),
        "isolated_cli_required": True,
    }
    return with_payload_hash(contract, "payload_sha256")


def validate_runtime_closure_contract(contract: dict) -> dict:
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "shohin-ocsc-runtime-closure-v3"
        or contract.get("isolated_cli_required") is not True
        or contract.get("payload_sha256")
        != hash_json(
            {key: value for key, value in contract.items() if key != "payload_sha256"}
        )
    ):
        raise ContractError("runtime closure contract mismatch")
    live = runtime_closure_contract()
    if not recursively_type_strict_equal(contract, live):
        raise ContractError("runtime closure drifted or was forged")
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"), "runtime closure contract"
    )


def _trusted_cryptography_symbols() -> tuple[Any, Any]:
    _trusted_distribution_module("cryptography")
    try:
        exceptions = importlib.import_module("cryptography.exceptions")
        ed25519 = importlib.import_module(
            "cryptography.hazmat.primitives.asymmetric.ed25519"
        )
    except ImportError as error:
        raise ContractError("trusted cryptography runtime is incomplete") from error
    return exceptions.InvalidSignature, ed25519.Ed25519PublicKey


def assert_ascii(value: str, label: str) -> None:
    try:
        str(value).encode("ascii")
    except UnicodeEncodeError as error:
        raise ContractError("{} must be ASCII".format(label)) from error


def normalized_prompt(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def normalized_prompt_sha256(text: str) -> str:
    return sha256_bytes(normalized_prompt(text).encode("ascii"))


def transition_prompt(state: dict) -> str:
    return "OCSC transition.\nState: {}\nNext state:".format(canonical_state(state))


def serializer_prompt(state: dict) -> str:
    canonical = canonical_state(state)
    if not state["z"]:
        raise ContractError("serializer prompt requires a terminal state")
    return "OCSC serializer.\nState: {}\nAnswer:".format(canonical)


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "big")


def digits_lsf(value: int, width: int) -> str:
    return "".join(str((value // (10**position)) % 10) for position in range(width))


def value_lsf(tape: str) -> int:
    return sum(int(digit) * (10**position) for position, digit in enumerate(str(tape)))


def tape_value(digits: list[int]) -> int:
    return sum(digit * (10**position) for position, digit in enumerate(digits))


def independent_value_lsf(tape: str) -> int:
    if (
        not isinstance(tape, str)
        or not tape
        or not tape.isascii()
        or not tape.isdigit()
    ):
        raise ContractError("independent arithmetic received an invalid digit tape")
    return sum(int(digit) * (10**position) for position, digit in enumerate(tape))


def independent_validate_state(state: dict) -> dict:
    required = {"op", "w", "p", "c", "a", "b", "r", "z"}
    if not isinstance(state, dict) or set(state) != required:
        raise ContractError("independent arithmetic state key mismatch")
    if state["op"] not in {"add", "sub"}:
        raise ContractError("independent arithmetic operation mismatch")
    if any(type(state[name]) is not int for name in ("w", "p", "c", "z")):
        raise ContractError("independent arithmetic scalar type mismatch")
    width, position, carry, terminal = (
        state["w"],
        state["p"],
        state["c"],
        state["z"],
    )
    if (
        width <= 0
        or position < 0
        or position > width
        or carry not in (0, 1)
        or terminal not in (0, 1)
        or terminal != int(position == width)
    ):
        raise ContractError("independent arithmetic scalar value mismatch")
    for field in ("a", "b", "r"):
        tape = state[field]
        if (
            not isinstance(tape, str)
            or not tape.isascii()
            or not tape.isdigit()
            or len(tape) != width
        ):
            raise ContractError("independent arithmetic tape mismatch")
    if any(digit != "0" for digit in state["r"][position:]):
        raise ContractError("independent arithmetic found a written suffix")
    if position == 0 and carry != 0:
        raise ContractError("independent arithmetic found initial carry")
    if state["op"] == "sub" and independent_value_lsf(
        state["a"]
    ) < independent_value_lsf(state["b"]):
        raise ContractError("independent subtraction is negative")
    return dict(state)


def independent_canonical_state(state: dict) -> str:
    state = independent_validate_state(state)
    return ("dws:op={op};w={w};p={p};c={c};a={a};b={b};r={r};z={z}").format(**state)


def independent_parse_state(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    match = INDEPENDENT_STATE_RE.fullmatch(text)
    if match is None:
        return None
    operation, width, position, carry, a_tape, b_tape, result_tape, terminal = (
        match.groups()
    )
    state = {
        "op": operation,
        "w": int(width),
        "p": int(position),
        "c": int(carry),
        "a": a_tape,
        "b": b_tape,
        "r": result_tape,
        "z": int(terminal),
    }
    try:
        canonical = independent_canonical_state(state)
    except ContractError:
        return None
    if text != canonical:
        return None
    return state


def independent_apply_microstep(state: dict) -> dict:
    state = independent_validate_state(state)
    if state["z"]:
        raise ContractError("independent arithmetic cannot step a terminal state")
    position = state["p"]
    left = ord(state["a"][position]) - ord("0")
    right = ord(state["b"][position]) - ord("0")
    carry = state["c"]
    if state["op"] == "add":
        raw = left + right + carry
        digit = raw - 10 if raw >= 10 else raw
        next_carry = int(raw >= 10)
    else:
        raw = left - right - carry
        digit = raw + 10 if raw < 0 else raw
        next_carry = int(raw < 0)
    result = state["r"][:position] + str(digit) + state["r"][position + 1 :]
    next_position = position + 1
    next_state = {
        "op": state["op"],
        "w": state["w"],
        "p": next_position,
        "c": next_carry,
        "a": state["a"],
        "b": state["b"],
        "r": result,
        "z": int(next_position == state["w"]),
    }
    return independent_validate_state(next_state)


def independent_state_at(
    operation: str, left: int, right: int, width: int, position: int
) -> dict:
    if (
        operation not in {"add", "sub"}
        or type(left) is not int
        or type(right) is not int
        or type(width) is not int
        or type(position) is not int
        or width <= 0
        or not 0 <= position <= width
        or not 0 <= left < 10**width
        or not 0 <= right < 10**width
        or (operation == "sub" and left < right)
    ):
        raise ContractError("independent state replay input mismatch")
    state = {
        "op": operation,
        "w": width,
        "p": 0,
        "c": 0,
        "a": digits_lsf(left, width),
        "b": digits_lsf(right, width),
        "r": "0" * width,
        "z": 0,
    }
    independent_validate_state(state)
    for _ in range(position):
        state = independent_apply_microstep(state)
    return state


def independent_state_answer(state: dict) -> int:
    state = independent_validate_state(state)
    if not state["z"]:
        raise ContractError("independent answer requested before terminal state")
    result = independent_value_lsf(state["r"])
    if state["op"] == "add":
        return result + state["c"] * (10 ** state["w"])
    if state["c"]:
        raise ContractError("independent terminal subtraction borrowed")
    return result


def state_at(operation: str, left: int, right: int, width: int, position: int) -> dict:
    state = initial_state(operation, left, right, width)
    for _ in range(position):
        state = apply_microstep(state)
    return state


def semantic_signature(state: dict, kind: str) -> str:
    if kind == "transition":
        payload = {
            "kind": kind,
            "op": str(state["op"]),
            "w": int(state["w"]),
            "p": int(state["p"]),
            "c": int(state["c"]),
            "a": str(state["a"]),
            "b": str(state["b"]),
            "r": str(state["r"]),
            "z": int(state["z"]),
        }
    elif kind == "serializer":
        payload = {
            "kind": kind,
            "width": int(state["w"]),
            "r": str(state["r"]),
            "c": int(state["c"]),
            "op": str(state["op"]),
        }
    else:
        raise ContractError("unknown semantic-signature kind")
    return hash_json(payload)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    width: int
    role: str
    operation: str
    position: int
    incoming_carry: int
    left_digit: int
    right_digit: int
    role_index: int


def cells_for_width(width: int) -> list[Cell]:
    if width not in WIDTHS:
        raise ContractError("unsupported width")
    cells: list[Cell] = []
    role_indices = Counter()

    def append(
        role: str, operation: str, position: int, carry: int, a: int, b: int
    ) -> None:
        role_index = role_indices[role]
        role_indices[role] += 1
        cell_id = "ocsc-cell-w{}-{}-{:03d}".format(
            width, role.replace("_", "-"), role_index
        )
        cells.append(
            Cell(
                cell_id=cell_id,
                width=width,
                role=role,
                operation=operation,
                position=position,
                incoming_carry=carry,
                left_digit=a,
                right_digit=b,
                role_index=role_index,
            )
        )

    for operation in ("add", "sub"):
        for left_digit in range(10):
            for right_digit in range(10):
                append("initial", operation, 0, 0, left_digit, right_digit)

    interior_index = 0
    for operation in ("add", "sub"):
        for carry in (0, 1):
            for left_digit in range(10):
                for right_digit in range(10):
                    position = 1 + (interior_index % (width - 2))
                    append(
                        "interior",
                        operation,
                        position,
                        carry,
                        left_digit,
                        right_digit,
                    )
                    interior_index += 1

    for carry in (0, 1):
        for left_digit in range(10):
            for right_digit in range(10):
                append(
                    "terminal_add",
                    "add",
                    width - 1,
                    carry,
                    left_digit,
                    right_digit,
                )

    for left_digit in range(10):
        for right_digit in range(left_digit + 1):
            append(
                "terminal_sub",
                "sub",
                width - 1,
                0,
                left_digit,
                right_digit,
            )
    for left_digit in range(1, 10):
        for right_digit in range(left_digit):
            append(
                "terminal_sub",
                "sub",
                width - 1,
                1,
                left_digit,
                right_digit,
            )

    counts = Counter(cell.role for cell in cells)
    if len(cells) != TRANSITION_CELLS_PER_WIDTH or counts != Counter(ROLE_CELL_COUNTS):
        raise AssertionError("local cell basis construction drifted")
    return cells


def reachable_context_state(
    cell: Cell, context_index: int, context_nonce: int = 0
) -> dict:
    rng = random.Random(
        stable_seed(
            "{}|context={}|nonce={}".format(cell.cell_id, context_index, context_nonce)
        )
    )
    for _ in range(100_000):
        a_digits = [rng.randrange(10) for _ in range(cell.width)]
        b_digits = [rng.randrange(10) for _ in range(cell.width)]
        a_digits[cell.position] = cell.left_digit
        b_digits[cell.position] = cell.right_digit
        if cell.operation == "sub" and cell.position < cell.width - 1:
            a_digits[-1], b_digits[-1] = 9, 0
        left, right = tape_value(a_digits), tape_value(b_digits)
        if cell.operation == "sub" and left < right:
            continue
        state = state_at(cell.operation, left, right, cell.width, cell.position)
        if (
            int(state["c"]) == cell.incoming_carry
            and int(state["a"][cell.position]) == cell.left_digit
            and int(state["b"][cell.position]) == cell.right_digit
        ):
            return state
    raise AssertionError("failed to realize reachable local context")


def intervention_state(reachable: dict, pair_index: int) -> tuple[dict, int]:
    if int(reachable["p"]) == 0:
        raise ContractError("initial state has no writable intervention prefix")
    state = dict(reachable)
    result = list(state["r"])
    position = pair_index % int(state["p"])
    delta = 1 + 2 * pair_index
    result[position] = str((int(result[position]) + delta) % 10)
    state["r"] = "".join(result)
    canonical_state(state)
    return state, position


def initial_suffix_variant(reachable: dict, pair_index: int) -> tuple[dict, str, int]:
    if int(reachable["p"]) != 0:
        raise ContractError("initial suffix variant requires p=0")
    state = dict(reachable)
    position = 1 + (pair_index % (int(state["w"]) - 1))
    field = "a"
    digits = list(state[field])
    if state["op"] == "sub":
        digits[position] = str(max(0, int(digits[position]) - 1 - pair_index))
    else:
        digits[position] = str((int(digits[position]) + 1 + pair_index) % 10)
    if digits[position] == state[field][position]:
        digits[position] = str((int(digits[position]) + 1) % 10)
    state[field] = "".join(digits)
    canonical_state(state)
    return state, field, position


def transition_row_from_state(
    cell: Cell,
    context_index: int,
    context_nonce: int,
    state: dict,
    *,
    reachability: str,
    pair_anchor_context_index: int | None,
    perturbation_field: str | None,
    perturbation_position: int | None,
) -> dict:
    expected = apply_microstep(state)
    prompt = transition_prompt(state)
    row_id = "{}-ctx{}".format(cell.cell_id, context_index)
    local_target = {
        "digit": int(expected["r"][cell.position]),
        "outgoing_carry": int(expected["c"]),
    }
    row = {
        "schema": SCHEMA,
        "dataset": "ocsc",
        "split": "train",
        "row_id": row_id,
        "kind": "transition",
        "skeleton_id": "skeleton-{}".format(cell.cell_id),
        "cell_id": cell.cell_id,
        "context_id": "{}-context-{}".format(cell.cell_id, context_index),
        "context_index": context_index,
        "context_nonce": context_nonce,
        "pair_anchor_context_index": pair_anchor_context_index,
        "perturbation_field": perturbation_field,
        "perturbation_position": perturbation_position,
        "reachability": reachability,
        "intervention": (
            "written_result_prefix" if reachability == "interventional" else "none"
        ),
        "role": cell.role,
        "width": cell.width,
        "operation": cell.operation,
        "position": cell.position,
        "incoming_carry": cell.incoming_carry,
        "left_digit": cell.left_digit,
        "right_digit": cell.right_digit,
        "outgoing_carry": int(expected["c"]),
        "local_target": local_target,
        "local_target_sha256": hash_json(local_target),
        "state": canonical_state(state),
        "expected_state": canonical_state(expected),
        "question": prompt,
        "completion_prompt": prompt,
        "response": canonical_state(expected),
        "source": "ocsc_transition_matched_v2",
        "training_group": "orthogonal_carry_serializer_curriculum",
        "prompt_sha256": sha256_bytes(prompt.encode("ascii")),
        "normalized_prompt_sha256": normalized_prompt_sha256(prompt),
        "semantic_signature_sha256": semantic_signature(state, "transition"),
    }
    return with_payload_hash(row, "row_sha256")


def assert_matched_local_pair(left: dict, right: dict, relation: str) -> None:
    left_state = parse_state(left["state"])
    right_state = parse_state(right["state"])
    left_target = parse_state(left["expected_state"])
    right_target = parse_state(right["expected_state"])
    if None in (left_state, right_state, left_target, right_target):
        raise AssertionError("matched pair contains an invalid state")
    field = str(right["perturbation_field"])
    position = int(right["perturbation_position"])
    if relation == "local_prefix_intervention":
        if field != "r" or position >= int(left_state["p"]):
            raise AssertionError("prefix intervention position is invalid")
    elif relation == "initial_suffix_context_invariance":
        if field not in {"a", "b"} or position <= int(left_state["p"]):
            raise AssertionError("initial suffix variation position is invalid")
    else:
        raise AssertionError("unknown local relation")
    for source_left, source_right in (
        (left_state, right_state),
        (left_target, right_target),
    ):
        for name in source_left:
            if name == field:
                left_tape = str(source_left[name])
                right_tape = str(source_right[name])
                differences = [
                    index
                    for index, (a, b) in enumerate(zip(left_tape, right_tape))
                    if a != b
                ]
                if differences != [position]:
                    raise AssertionError("matched pair tape delta is not singleton")
            elif source_left[name] != source_right[name]:
                raise AssertionError("matched pair changed a nonperturbed state field")
    if left["local_target_sha256"] != right["local_target_sha256"]:
        raise AssertionError("matched pair local target mismatch")


def generate_ocsc_transition_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    relations: list[dict] = []
    seen_prompts: set[str] = set()
    for width in WIDTHS:
        for cell in cells_for_width(width):
            base_states: list[tuple[dict, int]] = []
            for context in range(3):
                for context_nonce in range(100_000):
                    state = reachable_context_state(cell, context, context_nonce)
                    prompt_hash = normalized_prompt_sha256(transition_prompt(state))
                    if prompt_hash in seen_prompts:
                        continue
                    seen_prompts.add(prompt_hash)
                    base_states.append((state, context_nonce))
                    break
                else:
                    raise AssertionError("failed to construct a unique OCSC context")
            cell_rows = [
                transition_row_from_state(
                    cell,
                    context,
                    context_nonce,
                    state,
                    reachability="reachable",
                    pair_anchor_context_index=context if context < 2 else None,
                    perturbation_field=None,
                    perturbation_position=None,
                )
                for context, (state, context_nonce) in enumerate(base_states)
            ]
            variants = []
            for pair_index in range(2):
                anchor_state, context_nonce = base_states[pair_index]
                if cell.role == "initial":
                    variant, field, position = initial_suffix_variant(
                        anchor_state, pair_index
                    )
                    reachability = "reachable"
                else:
                    variant, position = intervention_state(anchor_state, pair_index)
                    field = "r"
                    reachability = "interventional"
                variant_hash = normalized_prompt_sha256(transition_prompt(variant))
                if variant_hash in seen_prompts:
                    raise AssertionError("matched variant prompt collision")
                seen_prompts.add(variant_hash)
                variants.append(
                    transition_row_from_state(
                        cell,
                        3 + pair_index,
                        context_nonce,
                        variant,
                        reachability=reachability,
                        pair_anchor_context_index=pair_index,
                        perturbation_field=field,
                        perturbation_position=position,
                    )
                )
            cell_rows.extend(variants)
            rows.extend(cell_rows)
            for relation_index, (left_context, right_context) in enumerate(
                ((0, 3), (1, 4))
            ):
                pair = {
                    "schema": SCHEMA,
                    "pair_id": "ocsc-rel-local-{}-{}".format(
                        cell.cell_id, relation_index
                    ),
                    "relation": (
                        "initial_suffix_context_invariance"
                        if cell.role == "initial"
                        else "local_prefix_intervention"
                    ),
                    "cell_id": cell.cell_id,
                    "role": cell.role,
                    "incoming_carry": cell.incoming_carry,
                    "left_row_id": cell_rows[left_context]["row_id"],
                    "right_row_id": cell_rows[right_context]["row_id"],
                    "target_sha256": cell_rows[left_context]["local_target_sha256"],
                    "perturbation_field": cell_rows[right_context][
                        "perturbation_field"
                    ],
                    "perturbation_position": cell_rows[right_context][
                        "perturbation_position"
                    ],
                    "factorial_active": cell.role != "initial"
                    and not (
                        cell.role == "terminal_sub"
                        and cell.incoming_carry == 0
                        and cell.role_index >= 45
                    ),
                    "shared_main_forward_required": True,
                }
                assert_matched_local_pair(
                    cell_rows[left_context], cell_rows[right_context], pair["relation"]
                )
                relations.append(with_payload_hash(pair, "pair_sha256"))
    if len(rows) != TRANSITION_ROWS or len(relations) != 9_000:
        raise AssertionError("OCSC transition construction count mismatch")
    return rows, relations


def normalized_reversal(pattern: tuple[int, ...]) -> tuple[int, ...]:
    reversed_pattern = tuple(reversed(pattern))
    offset = reversed_pattern[0]
    return tuple((digit - offset) % 10 for digit in reversed_pattern)


def translated_orbit(pattern: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        tuple((digit + translation) % 10 for digit in base)
        for base in (pattern, tuple(reversed(pattern)))
        for translation in range(10)
    ]


def hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def serializer_pattern_metrics(pattern: tuple[int, ...]) -> dict[str, int | bool]:
    differences = tuple(
        (pattern[index + 1] - pattern[index]) % 10 for index in range(len(pattern) - 1)
    )
    maximum_frequency = max(Counter(pattern).values())
    return {
        "distinct_digits": len(set(pattern)),
        "distinct_adjacent_differences": len(set(differences)),
        "non_affine": len(set(differences)) >= 2,
        "maximum_digit_frequency": maximum_frequency,
        "constant_except_one": maximum_frequency >= len(pattern) - 1,
    }


def serializer_min_hamming(width: int) -> int:
    return {3: 1, 4: 2, 5: 2, 6: 3, 7: 3}[width]


def _serializer_patterns_for_domain(
    width: int, seed_label: str, blocked_tapes: set[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    if width not in WIDTHS:
        raise ContractError("unsupported serializer width")
    selected: list[tuple[int, ...]] = []
    excluded: set[tuple[int, ...]] = set()
    rng = random.Random(stable_seed(seed_label))
    minimum_hamming = serializer_min_hamming(width)
    for _ in range(100_000):
        pattern = (0,) + tuple(rng.randrange(10) for _ in range(width - 1))
        reverse = normalized_reversal(pattern)
        orbit = set(translated_orbit(pattern))
        if (
            reverse == pattern
            or pattern in excluded
            or reverse in excluded
            or orbit & blocked_tapes
        ):
            continue
        metrics = serializer_pattern_metrics(pattern)
        if (
            metrics["distinct_digits"] < min(width, 3)
            or metrics["distinct_adjacent_differences"] < 2
            or not metrics["non_affine"]
            or metrics["constant_except_one"]
        ):
            continue
        if any(
            min(hamming(pattern, variant) for variant in translated_orbit(prior))
            < minimum_hamming
            for prior in selected
        ):
            continue
        selected.append(pattern)
        excluded.add(pattern)
        excluded.add(reverse)
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise AssertionError("failed to construct serializer pattern orbits")
    return selected


def serializer_patterns(width: int) -> list[tuple[int, ...]]:
    return _serializer_patterns_for_domain(
        width,
        "ocsc-serializer-diverse-orbits-v2-w{}".format(width),
        set(),
    )


def hidden_serializer_patterns(width: int) -> list[tuple[int, ...]]:
    blocked = {
        tape
        for pattern in serializer_patterns(width)
        for tape in translated_orbit(pattern)
    }
    return _serializer_patterns_for_domain(
        width,
        "ocsc-hidden-serializer-diverse-orbits-v2-w{}".format(width),
        blocked,
    )


def serializer_operands(
    width: int, pair_index: int, tapes: tuple[str, str]
) -> tuple[int, int]:
    rng = random.Random(
        stable_seed("serializer-operands-w{}-p{}".format(width, pair_index))
    )
    for _ in range(100_000):
        a_digits = [rng.randrange(10) for _ in range(width)]
        b_digits = [rng.randrange(10) for _ in range(width)]
        a_digits[-1], b_digits[-1] = 9, 0
        left, right = tape_value(a_digits), tape_value(b_digits)
        add_state = state_at("add", left, right, width, width)
        sub_state = state_at("sub", left, right, width, width)
        if all(tape not in {add_state["r"], sub_state["r"]} for tape in tapes):
            return left, right
    raise AssertionError("failed to select non-recomputable serializer operands")


def serializer_state(
    width: int,
    operation: str,
    carry: int,
    left: int,
    right: int,
    tape: str,
) -> dict:
    state = {
        "op": operation,
        "w": width,
        "p": width,
        "c": carry,
        "a": digits_lsf(left, width),
        "b": digits_lsf(right, width),
        "r": tape,
        "z": 1,
    }
    canonical_state(state)
    state_answer(state)
    return state


def generate_serializer_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    relations: list[dict] = []
    slice_contract = {
        "add_c0": ("add", 0),
        "add_c1": ("add", 1),
        "sub_c0": ("sub", 0),
    }
    for width in WIDTHS:
        pair_index = 0
        for pattern_index, pattern in enumerate(serializer_patterns(width)):
            for translation in range(10):
                forward = "".join(str((digit + translation) % 10) for digit in pattern)
                reverse = forward[::-1]
                if forward == reverse:
                    raise AssertionError("palindromic serializer tape")
                left, right = serializer_operands(width, pair_index, (forward, reverse))
                operand_signature = hash_json(
                    {"width": width, "left": left, "right": right}
                )
                for slice_name, (operation, carry) in slice_contract.items():
                    endpoint_rows = []
                    for orientation, tape in (
                        ("forward", forward),
                        ("reverse", reverse),
                    ):
                        state = serializer_state(
                            width, operation, carry, left, right, tape
                        )
                        prompt = serializer_prompt(state)
                        answer = state_answer(state)
                        row = {
                            "schema": SCHEMA,
                            "dataset": "shared_serializer",
                            "split": "train",
                            "row_id": ("ocsc-ser-w{}-p{:02d}-{}-{}").format(
                                width,
                                pair_index,
                                slice_name.replace("_", "-"),
                                orientation,
                            ),
                            "kind": "serializer",
                            "pair_base_id": "ocsc-tape-w{}-p{:02d}".format(
                                width, pair_index
                            ),
                            "pattern_index": pattern_index,
                            "translation": translation,
                            "orientation": orientation,
                            "serializer_slice": slice_name,
                            "reachability": "interventional",
                            "intervention": "terminal_result_tape",
                            "width": width,
                            "operation": operation,
                            "incoming_carry": carry,
                            "tape": tape,
                            "tape_sha256": sha256_bytes(tape.encode("ascii")),
                            "operand_signature_sha256": operand_signature,
                            "state": canonical_state(state),
                            "expected_answer": answer,
                            "answer_length": len(str(answer)),
                            "question": prompt,
                            "completion_prompt": prompt,
                            "response": "answer={}".format(answer),
                            "source": "ocsc_serializer_v1",
                            "training_group": (
                                "orthogonal_carry_serializer_curriculum"
                            ),
                            "prompt_sha256": sha256_bytes(prompt.encode("ascii")),
                            "normalized_prompt_sha256": (
                                normalized_prompt_sha256(prompt)
                            ),
                            "semantic_signature_sha256": semantic_signature(
                                state, "serializer"
                            ),
                        }
                        endpoint_rows.append(with_payload_hash(row, "row_sha256"))
                    rows.extend(endpoint_rows)
                    pair = {
                        "schema": SCHEMA,
                        "pair_id": ("ocsc-rel-ser-w{}-p{:02d}-{}").format(
                            width,
                            pair_index,
                            slice_name.replace("_", "-"),
                        ),
                        "relation": "serializer_reversal",
                        "pair_base_id": endpoint_rows[0]["pair_base_id"],
                        "serializer_slice": slice_name,
                        "left_row_id": endpoint_rows[0]["row_id"],
                        "right_row_id": endpoint_rows[1]["row_id"],
                        "operand_signature_sha256": operand_signature,
                        "transform": ("reverse_lsf_tape_then_render_ordinary_integer"),
                        "shared_main_forward_required": True,
                    }
                    relations.append(with_payload_hash(pair, "pair_sha256"))
                pair_index += 1
        if pair_index != 50:
            raise AssertionError("serializer pair count drifted")
    if len(rows) != SERIALIZER_ROWS or len(relations) != 750:
        raise AssertionError("serializer construction count mismatch")
    return rows, relations


def iid_transition_row(target: dict, draw: int) -> dict:
    width = int(target["width"])
    role = str(target["role"])
    position = int(target["position"])
    operation = str(target["operation"])
    rng = random.Random(
        stable_seed("iid-slot-v2|{}|draw={}".format(target["row_id"], draw))
    )
    for subdraw in range(100):
        left, right = rng.randrange(10**width), rng.randrange(10**width)
        if operation == "sub" and left < right:
            left, right = right, left
        state = state_at(operation, left, right, width, position)
        expected = apply_microstep(state)
        if role == "initial" and int(state["p"]) != 0:
            continue
        if role == "terminal_sub" and int(expected["c"]) != 0:
            continue
        prompt = transition_prompt(state)
        prompt_hash = normalized_prompt_sha256(prompt)
        local_target = {
            "digit": int(expected["r"][position]),
            "outgoing_carry": int(expected["c"]),
        }
        row = {
            "schema": SCHEMA,
            "dataset": "iid_control",
            "split": "train",
            "row_id": "iid-{}-ctx{}".format(
                target["cell_id"].removeprefix("ocsc-cell-"),
                target["context_index"],
            ),
            "kind": "transition",
            "skeleton_id": target["skeleton_id"],
            "matched_ocsc_row_id": target["row_id"],
            "context_index": target["context_index"],
            "iid_draw": draw * 100 + subdraw,
            "reachability": "reachable",
            "intervention": "none",
            "role": role,
            "width": width,
            "operation": operation,
            "position": position,
            "incoming_carry": int(state["c"]),
            "left_digit": int(state["a"][position]),
            "right_digit": int(state["b"][position]),
            "outgoing_carry": int(expected["c"]),
            "local_target": local_target,
            "local_target_sha256": hash_json(local_target),
            "state": canonical_state(state),
            "expected_state": canonical_state(expected),
            "question": prompt,
            "completion_prompt": prompt,
            "response": canonical_state(expected),
            "source": "ocsc_iid_transition_control_v1",
            "training_group": "orthogonal_carry_serializer_iid_control",
            "prompt_sha256": sha256_bytes(prompt.encode("ascii")),
            "normalized_prompt_sha256": prompt_hash,
            "semantic_signature_sha256": semantic_signature(state, "transition"),
        }
        return with_payload_hash(row, "row_sha256")
    raise AssertionError("failed to draw a feasible IID control row")


def generate_iid_transition_rows(
    ocsc_rows: list[dict], tokenizer: FrozenTokenizer | None = None
) -> list[dict]:
    forbidden = {row["normalized_prompt_sha256"] for row in ocsc_rows}
    rows: list[dict] = []
    for target in ocsc_rows:
        target_layout = (
            training_layout_signature(tokenizer, target)
            if tokenizer is not None
            else None
        )
        for draw in range(100_000):
            candidate = iid_transition_row(target, draw)
            if candidate["normalized_prompt_sha256"] in forbidden:
                continue
            if (
                tokenizer is not None
                and training_layout_signature(tokenizer, candidate) != target_layout
            ):
                continue
            forbidden.add(candidate["normalized_prompt_sha256"])
            rows.append(candidate)
            break
        else:
            raise ContractError(
                "failed to token-layout-match IID row to {}".format(target["row_id"])
            )
    if len(rows) != TRANSITION_ROWS:
        raise AssertionError("IID transition construction count mismatch")
    return rows


def strict_json_loads(line: str, label: str) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate JSON key in {}: {}".format(label, key))
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError("nonfinite JSON constant in {}: {}".format(label, token))
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ContractError("invalid JSON in {}".format(label)) from error
    if not isinstance(value, dict):
        raise ContractError("{} must contain JSON objects".format(label))
    assert_finite_json(value, label)
    return value


def validate_regular_file(
    path: Path, label: str, *, exact_mode: int | None = None
) -> os.stat_result:
    return read_file_snapshot(path, label, exact_mode=exact_mode).metadata


def validate_custody_root_file(path: Path, label: str) -> os.stat_result:
    return read_file_snapshot(
        path,
        label,
        exact_mode=0o444,
        custody_root=True,
    ).metadata


def read_ascii_file(path: Path, label: str) -> bytes:
    payload = read_file_snapshot(path, label).payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("{} must be ASCII".format(label)) from error
    return payload


def load_prompt_registry(
    path: Path, snapshot: FileSnapshot | None = None
) -> tuple[list[dict], dict]:
    snapshot = snapshot or read_file_snapshot(path, "prompt registry")
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("prompt registry must be ASCII") from error
    records = []
    seen_ids: set[str] = set()
    seen_normalized: set[str] = set()
    seen_semantic: set[str] = set()
    counts = Counter()
    replay_keys = {
        "prompt_id",
        "family",
        "use",
        "prompt",
        "prompt_sha256",
        "normalized_prompt_sha256",
        "semantic_signature_sha256",
        "source_commitment",
    }
    commitment_keys = replay_keys - {"prompt"}
    for line_number, raw_line in enumerate(payload.decode("ascii").splitlines(), 1):
        if not raw_line:
            raise ContractError("prompt registry contains a blank line")
        row = strict_json_loads(raw_line, "prompt registry line {}".format(line_number))
        use, family = row.get("use"), row.get("family")
        if (use, family) not in REGISTRY_COUNTS:
            raise ContractError("prompt registry use/family mismatch")
        expected_keys = replay_keys if use == "replay" else commitment_keys
        if set(row) != expected_keys:
            raise ContractError("prompt registry key mismatch")
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or not ID_RE.fullmatch(prompt_id):
            raise ContractError("invalid prompt registry ID")
        if prompt_id in seen_ids:
            raise ContractError("duplicate prompt registry ID")
        seen_ids.add(prompt_id)
        for field in (
            "prompt_sha256",
            "normalized_prompt_sha256",
            "semantic_signature_sha256",
            "source_commitment",
        ):
            if not isinstance(row[field], str) or not HEX64_RE.fullmatch(row[field]):
                raise ContractError("invalid prompt registry hash")
        if row["normalized_prompt_sha256"] in seen_normalized:
            raise ContractError("duplicate normalized prompt commitment")
        seen_normalized.add(row["normalized_prompt_sha256"])
        if row["semantic_signature_sha256"] in seen_semantic:
            raise ContractError("duplicate semantic signature commitment")
        seen_semantic.add(row["semantic_signature_sha256"])
        if use == "replay":
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt:
                raise ContractError("replay prompt must be nonempty")
            assert_ascii(prompt, "replay prompt")
            if sha256_bytes(prompt.encode("ascii")) != row["prompt_sha256"]:
                raise ContractError("replay prompt hash mismatch")
            if normalized_prompt_sha256(prompt) != row["normalized_prompt_sha256"]:
                raise ContractError("replay normalized prompt hash mismatch")
        counts[(use, family)] += 1
        records.append(row)
    if len(records) != PROMPT_REGISTRY_ROWS or counts != Counter(REGISTRY_COUNTS):
        raise ContractError("prompt registry count contract mismatch")
    roots = {}
    for use in ("replay", "development", "secret_confirmation"):
        selected = sorted(
            (row for row in records if row["use"] == use),
            key=lambda row: row["prompt_id"],
        )
        roots[use] = sha256_bytes(jsonl_bytes(selected))
    summary = {
        "schema": "shohin-ocsc-prompt-registry-commitment-v1",
        "resolved_path": snapshot.resolved_path,
        "physical_sha256": sha256_bytes(payload),
        "physical_bytes": len(payload),
        "rows": len(records),
        "counts": {
            "{}:{}".format(use, family): count
            for (use, family), count in sorted(counts.items())
        },
        "roots": roots,
        "evaluation_rows": sum(row["use"] != "replay" for row in records),
        "secret_rows_republished": False,
    }
    return records, summary


def expected_serializer_slice_counts() -> dict[str, int]:
    return {
        "w{}:{}".format(width, slice_name): 100
        for width in WIDTHS
        for slice_name in SERIALIZER_SLICES
    }


def hidden_transition_semantic_contract() -> dict:
    return {
        "site_counts_per_width_role": {
            "initial": HIDDEN_INITIAL_SITES_PER_WIDTH,
            **HIDDEN_NONINITIAL_SITE_COUNTS,
        },
        "required_operations_by_role": {
            "initial": ["add", "sub"],
            "interior": ["add", "sub"],
            "terminal_add": ["add"],
            "terminal_sub": ["sub"],
        },
        "minimum_sites_per_required_operation": {
            "initial": 16,
            "interior": 13,
            "terminal_add": HIDDEN_NONINITIAL_SITE_COUNTS["terminal_add"],
            "terminal_sub": HIDDEN_NONINITIAL_SITE_COUNTS["terminal_sub"],
        },
        "required_active_left_digits": {
            "initial": list(range(10)),
            "interior": list(range(10)),
            "terminal_add": list(range(10)),
            "terminal_sub": list(range(1, 10)),
        },
        "required_active_right_digits": {
            "initial": list(range(10)),
            "interior": list(range(10)),
            "terminal_add": list(range(10)),
            "terminal_sub": list(range(9)),
        },
        "minimum_sites_per_required_active_digit": {
            "initial": 2,
            "interior": 2,
            "terminal_add": 1,
            "terminal_sub": 1,
        },
        "interior_all_legal_positions_required": True,
        "interior_position_count_max_delta": 1,
        "unique_natural_local_tuple_required_per_site": True,
        "unique_anchor_endpoint_transition_tuple_required": True,
    }


def hidden_geometry_contract() -> dict:
    transition_slices = {
        "w{}:initial:c0:{}".format(width, endpoint): HIDDEN_INITIAL_SITES_PER_WIDTH
        for width in WIDTHS
        for endpoint in ("anchor", "variant")
    }
    for width in WIDTHS:
        for role, count in HIDDEN_NONINITIAL_SITE_COUNTS.items():
            for carry in (0, 1):
                for prefix_variant in HIDDEN_PREFIX_VARIANTS:
                    transition_slices[
                        "w{}:{}:c{}:{}".format(width, role, carry, prefix_variant)
                    ] = count
    return {
        "direct_cases": 3_600,
        "transition_cases": 2_100,
        "serializer_cases": 1_500,
        "initial_invariance_sites": len(WIDTHS) * HIDDEN_INITIAL_SITES_PER_WIDTH,
        "noninitial_paired_carry_sites": len(WIDTHS)
        * sum(HIDDEN_NONINITIAL_SITE_COUNTS.values()),
        "noninitial_rows_per_site": 4,
        "transition_slices": transition_slices,
        "serializer_slices": expected_serializer_slice_counts(),
        "semantic_coverage": hidden_transition_semantic_contract(),
    }


def hidden_merkle_algorithm() -> dict:
    return {
        "canonical_row": "ascii-json-sort-keys-no-whitespace-no-newline",
        "hash": "sha256",
        "index_encoding": "unsigned-64-bit-big-endian",
        "leaf_domain_ascii": "R12-OCSC-HIDDEN-LEAF-v1\\0",
        "length_encoding": "unsigned-64-bit-big-endian",
        "node_domain_ascii": "R12-OCSC-HIDDEN-NODE-v1\\0",
        "node_level_encoding": "unsigned-64-bit-big-endian",
        "odd_node": "duplicate-last",
        "root_domain_ascii": "R12-OCSC-HIDDEN-ROOT-v1\\0",
        "root_encoding": "lowercase-hex",
    }


def custodian_commitment_algorithm() -> dict:
    return {
        "canonical_opening": "ascii-json-sort-keys-no-whitespace-no-newline",
        "commitment_domain_ascii": "R12-OCSC-CUSTODIAN-OPENING-v1\\0",
        "length_encoding": "unsigned-64-bit-big-endian",
        "hash": "sha256",
        "encoding": "lowercase-hex",
    }


def custodian_opening_commitment(document: dict) -> str:
    canonical = canonical_json_bytes(document)
    return hashlib.sha256(
        HIDDEN_CUSTODIAN_DOMAIN + struct.pack(">Q", len(canonical)) + canonical
    ).hexdigest()


def load_confirmation_commitment(
    path: Path, snapshot: FileSnapshot | None = None
) -> tuple[dict, str]:
    snapshot = snapshot or read_file_snapshot(path, "secret confirmation commitment")
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("secret confirmation commitment must be ASCII") from error
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ContractError("secret confirmation commitment must be one JSON line")
    document = strict_json_loads(lines[0], "secret confirmation commitment")
    expected_keys = {
        "schema",
        "board_id",
        "leaf_count",
        "merkle_root",
        "geometry",
        "geometry_sha256",
        "merkle_algorithm",
        "custodian_commitment_algorithm",
        "custodian_commitment",
        "secret_rows_in_document",
    }
    if set(document) != expected_keys:
        raise ContractError("secret confirmation commitment key mismatch")
    if document["schema"] != "shohin-ocsc-hidden-merkle-commitment-v2":
        raise ContractError("secret confirmation commitment schema mismatch")
    if not isinstance(document["board_id"], str) or not ID_RE.fullmatch(
        document["board_id"]
    ):
        raise ContractError("invalid secret confirmation board ID")
    geometry = hidden_geometry_contract()
    if (
        type(document["leaf_count"]) is not int
        or document["leaf_count"] != 3_600
        or document["geometry"] != geometry
        or document["geometry_sha256"] != hash_json(geometry)
        or document["merkle_algorithm"] != hidden_merkle_algorithm()
        or document["custodian_commitment_algorithm"]
        != custodian_commitment_algorithm()
    ):
        raise ContractError("secret confirmation geometry mismatch")
    if document["secret_rows_in_document"] is not False:
        raise ContractError("secret confirmation document must not contain rows")
    for field in ("merkle_root", "custodian_commitment"):
        if not isinstance(document[field], str) or not HEX64_RE.fullmatch(
            document[field]
        ):
            raise ContractError("invalid secret confirmation root")
    return document, sha256_bytes(payload)


def load_custodian_opening(
    path: Path,
    commitment: dict,
    snapshot: FileSnapshot | None = None,
) -> tuple[dict, str]:
    snapshot = snapshot or read_file_snapshot(path, "custodian opening")
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("custodian opening must be ASCII") from error
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ContractError("custodian opening must be one JSON line")
    document = strict_json_loads(lines[0], "custodian opening")
    if set(document) != {"schema", "board_id", "custodian_id", "nonce_hex"}:
        raise ContractError("custodian opening key mismatch")
    if (
        document["schema"] != "shohin-ocsc-custodian-opening-v1"
        or document["board_id"] != commitment["board_id"]
        or not isinstance(document["custodian_id"], str)
        or not ID_RE.fullmatch(document["custodian_id"])
        or not isinstance(document["nonce_hex"], str)
        or not HEX64_RE.fullmatch(document["nonce_hex"])
    ):
        raise ContractError("custodian opening identity mismatch")
    computed = custodian_opening_commitment(document)
    if computed != commitment["custodian_commitment"]:
        raise ContractError("custodian opening authentication failed")
    return document, sha256_bytes(payload)


class FrozenTokenizer:
    def __init__(
        self,
        path: Path,
        mode: str,
        snapshot: FileSnapshot | None = None,
    ):
        path = Path(path)
        self.path = path
        snapshot = snapshot or read_file_snapshot(path, "tokenizer")
        payload = snapshot.payload
        self.resolved_path = snapshot.resolved_path
        self.payload_sha256 = sha256_bytes(payload)
        self.payload_bytes = len(payload)
        if mode == "production" and (
            self.payload_sha256 != KNOWN_TOKENIZER_SHA256
            or self.payload_bytes != KNOWN_TOKENIZER_BYTES
        ):
            raise ContractError("production tokenizer commitment mismatch")
        tokenizer_module = _trusted_distribution_module("tokenizers")
        Tokenizer = tokenizer_module.Tokenizer
        try:
            self.tokenizer = Tokenizer.from_str(payload.decode("utf-8"))
        except Exception as error:
            raise ContractError("failed to load tokenizer") from error
        self.vocab_size = int(self.tokenizer.get_vocab_size())

    def encode(self, text: str):
        assert_ascii(text, "tokenized text")
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if decoded != text:
            raise ContractError("tokenizer is not lossless for an OCSC frame")
        return encoding


def int_array_sha256(values: Iterable[int], width: int = 4) -> str:
    digest = hashlib.sha256()
    packer = struct.Struct("<I" if width == 4 else "<Q")
    for value in values:
        digest.update(packer.pack(int(value)))
    return digest.hexdigest()


def field_spans(response: str, response_start: int) -> list[tuple[int, int, str]]:
    if response.startswith("dws:"):
        spans = []
        for match in FIELD_RE.finditer(response[4:]):
            start, end = match.span(2)
            spans.append(
                (
                    response_start + 4 + start,
                    response_start + 4 + end,
                    match.group(1),
                )
            )
        if {field for _, _, field in spans} != {
            "op",
            "w",
            "p",
            "c",
            "a",
            "b",
            "r",
            "z",
        }:
            raise AssertionError("transition response field parse failed")
        return spans
    if response.startswith("answer="):
        return [
            (
                response_start + len("answer="),
                response_start + len(response),
                "answer",
            )
        ]
    raise ContractError("unsupported response serialization")


@dataclass
class TokenizedRow:
    corpus: str
    row_id: str
    kind: str
    payload: bytes
    token_count: int
    supervised_tokens: int
    raw_weight_units: int
    activation_indices: dict
    receipt: dict


def frozen_digit_token_ids(tokenizer: FrozenTokenizer) -> list[int]:
    cached = getattr(tokenizer, "_ocsc_digit_token_ids", None)
    if cached is not None:
        return list(cached)
    result = []
    for digit in range(10):
        encoding = tokenizer.encode(str(digit))
        if len(encoding.ids) != 1 or encoding.offsets != [(0, 1)]:
            raise ContractError("each decimal digit must be one exact tokenizer token")
        result.append(int(encoding.ids[0]))
    if len(set(result)) != 10:
        raise ContractError("decimal digit token IDs must be distinct")
    tokenizer._ocsc_digit_token_ids = tuple(result)
    return result


def training_frame(row: dict) -> tuple[str, int, int]:
    prompt = row["completion_prompt"]
    response = row["response"]
    frame = prompt + "\n" + response + "\n"
    response_start = len(prompt) + 1
    return frame, response_start, response_start + len(response)


def _exact_character_token(
    encoding: Any, character_offset: int, expected_token_id: int, label: str
) -> int:
    matches = [
        index
        for index, (start, end) in enumerate(encoding.offsets)
        if start == character_offset and end == character_offset + 1
    ]
    if len(matches) != 1 or int(encoding.ids[matches[0]]) != expected_token_id:
        raise ContractError("{} is not aligned to one frozen digit token".format(label))
    if matches[0] == 0:
        raise ContractError("{} has no causal prediction position".format(label))
    return matches[0]


def _slot_vectors(
    tokenizer: FrozenTokenizer,
    row: dict,
    pad_token_id: int,
    digit_token_ids: list[int],
) -> tuple[dict[str, list[int]], dict, Counter]:
    frame, response_start, response_end = training_frame(row)
    encoding = tokenizer.encode(frame)
    token_count = len(encoding.ids)
    if token_count > SEQUENCE_LENGTH:
        raise ContractError(
            "row exceeds the frozen 256-token slot: {}".format(row["row_id"])
        )
    spans = field_spans(row["response"], response_start)
    span_by_field = {field: (start, end) for start, end, field in spans}
    attention = [1] * token_count + [0] * (SEQUENCE_LENGTH - token_count)
    completion = [0] * SEQUENCE_LENGTH
    field_ids = [FIELD_IDS["prompt"]] * token_count + [0] * (
        SEQUENCE_LENGTH - token_count
    )
    raw_units = [0] * SEQUENCE_LENGTH
    field_counts = Counter()
    for index, (start, end) in enumerate(encoding.offsets):
        if start < response_start < end or start < response_end < end:
            raise ContractError("token crosses a completion boundary")
        if end <= response_start or start >= response_end or start == end:
            continue
        completion[index] = 1
        matching = [
            field for left, right, field in spans if end > left and start < right
        ]
        field = matching[0] if matching else "default"
        field_ids[index] = FIELD_IDS[field]
        raw_units[index] = FIELD_WEIGHT_UNITS[field]
        field_counts[field] += 1
    if not any(completion):
        raise ContractError("training frame has no supervised tokenizer positions")
    activation: dict[str, Any]
    if row["kind"] == "transition":
        carry_offset = span_by_field["c"][0]
        result_offset = span_by_field["r"][0] + int(row["position"])
        carry_value = int(row["outgoing_carry"])
        digit_value = int(row["local_target"]["digit"])
        carry_target = _exact_character_token(
            encoding,
            carry_offset,
            digit_token_ids[carry_value],
            "transition carry",
        )
        digit_target = _exact_character_token(
            encoding,
            result_offset,
            digit_token_ids[digit_value],
            "transition result digit",
        )
        activation = {
            "kind": "transition",
            "carry": {
                "prediction_index": carry_target - 1,
                "target_token_index": carry_target,
                "target_value": carry_value,
            },
            "result_digit": {
                "prediction_index": digit_target - 1,
                "target_token_index": digit_target,
                "target_value": digit_value,
            },
        }
    elif row["kind"] == "serializer":
        answer = str(row["expected_answer"])
        answer_start = span_by_field["answer"][0]
        answer_targets = [
            _exact_character_token(
                encoding,
                answer_start + index,
                digit_token_ids[int(character)],
                "serializer answer digit",
            )
            for index, character in enumerate(answer)
        ]
        tape_targets: list[int | None] = []
        prefix = 1 if row["serializer_slice"] == "add_c1" else 0
        for tape_position in range(int(row["width"])):
            ordinary_index = prefix + int(row["width"]) - 1 - tape_position
            if prefix == 0:
                ordinary_index = len(answer) - 1 - tape_position
            if ordinary_index < prefix or ordinary_index >= len(answer):
                tape_targets.append(None)
            else:
                tape_targets.append(answer_targets[ordinary_index])
        activation = {
            "kind": "serializer",
            "answer_prediction_indices": [index - 1 for index in answer_targets],
            "answer_target_token_indices": answer_targets,
            "tape_position_prediction_indices": [
                None if index is None else index - 1 for index in tape_targets
            ],
            "tape_position_target_token_indices": tape_targets,
        }
    else:
        raise ContractError("unknown training row kind")
    vectors = {
        "token_ids": [int(value) for value in encoding.ids]
        + [pad_token_id] * (SEQUENCE_LENGTH - token_count),
        "attention_mask": attention,
        "completion_mask": completion,
        "field_ids": field_ids,
        "raw_weight_units": raw_units,
        "position_ids": list(range(token_count))
        + [0] * (SEQUENCE_LENGTH - token_count),
    }
    return vectors, activation, field_counts


def slot_payload_bytes(vectors: dict[str, list[int]]) -> bytes:
    if set(vectors) != {name for name, _ in SLOT_PAYLOAD_LAYOUT}:
        raise ContractError("slot vector key mismatch")
    if any(len(vectors[name]) != SEQUENCE_LENGTH for name, _ in SLOT_PAYLOAD_LAYOUT):
        raise ContractError("slot vector length mismatch")
    payload = struct.pack("<{}I".format(SEQUENCE_LENGTH), *vectors["token_ids"])
    payload += bytes(vectors["attention_mask"])
    payload += bytes(vectors["completion_mask"])
    payload += bytes(vectors["field_ids"])
    payload += bytes(vectors["raw_weight_units"])
    payload += struct.pack("<{}H".format(SEQUENCE_LENGTH), *vectors["position_ids"])
    if len(payload) != SLOT_PAYLOAD_BYTES:
        raise AssertionError("slot payload byte layout drifted")
    return payload


def slot_payload_receipt(payload: bytes) -> dict:
    compressed = zlib.compress(payload, level=9)
    return {
        "codec": "zlib-level-9-then-base64",
        "compressed_bytes": len(compressed),
        "compressed_sha256": sha256_bytes(compressed),
        "data_base64": base64.b64encode(compressed).decode("ascii"),
        "layout": "256xu32,256xu8,256xu8,256xu8,256xu8,256xu16-le",
        "uncompressed_bytes": len(payload),
        "uncompressed_sha256": sha256_bytes(payload),
    }


def decode_slot_payload(receipt: dict) -> dict[str, list[int]]:
    payload_record = receipt["slot_payload"]
    expected_keys = {
        "codec",
        "compressed_bytes",
        "compressed_sha256",
        "data_base64",
        "layout",
        "uncompressed_bytes",
        "uncompressed_sha256",
    }
    if set(payload_record) != expected_keys:
        raise ContractError("slot payload receipt key mismatch")
    if (
        payload_record["codec"] != "zlib-level-9-then-base64"
        or payload_record["layout"] != "256xu32,256xu8,256xu8,256xu8,256xu8,256xu16-le"
        or payload_record["uncompressed_bytes"] != SLOT_PAYLOAD_BYTES
    ):
        raise ContractError("slot payload receipt contract mismatch")
    try:
        compressed = base64.b64decode(payload_record["data_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ContractError("invalid slot payload base64") from error
    if (
        len(compressed) != payload_record["compressed_bytes"]
        or sha256_bytes(compressed) != payload_record["compressed_sha256"]
    ):
        raise ContractError("compressed slot payload hash mismatch")
    decompressor = zlib.decompressobj()
    try:
        payload = decompressor.decompress(compressed) + decompressor.flush()
    except zlib.error as error:
        raise ContractError("invalid compressed slot payload") from error
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise ContractError("noncanonical compressed slot payload")
    if (
        len(payload) != SLOT_PAYLOAD_BYTES
        or sha256_bytes(payload) != payload_record["uncompressed_sha256"]
    ):
        raise ContractError("uncompressed slot payload hash mismatch")
    cursor = 0
    token_bytes = 4 * SEQUENCE_LENGTH
    tokens = list(struct.unpack("<{}I".format(SEQUENCE_LENGTH), payload[:token_bytes]))
    cursor += token_bytes
    result = {"token_ids": tokens}
    for name in (
        "attention_mask",
        "completion_mask",
        "field_ids",
        "raw_weight_units",
    ):
        result[name] = list(payload[cursor : cursor + SEQUENCE_LENGTH])
        cursor += SEQUENCE_LENGTH
    result["position_ids"] = list(
        struct.unpack("<{}H".format(SEQUENCE_LENGTH), payload[cursor:])
    )
    return result


def training_layout_signature(
    tokenizer: FrozenTokenizer, row: dict
) -> tuple[int, bytes]:
    digit_ids = frozen_digit_token_ids(tokenizer)
    vectors, _, _ = _slot_vectors(tokenizer, row, 0, digit_ids)
    return sum(vectors["attention_mask"]), bytes(vectors["completion_mask"])


def tokenize_training_row(
    tokenizer: FrozenTokenizer,
    corpus: str,
    row: dict,
    pad_token_id: int,
    digit_token_ids: list[int],
) -> TokenizedRow:
    frame, _, _ = training_frame(row)
    encoding = tokenizer.encode(frame)
    vectors, activation, field_counts = _slot_vectors(
        tokenizer, row, pad_token_id, digit_token_ids
    )
    payload = slot_payload_bytes(vectors)
    supervised_tokens = sum(vectors["completion_mask"])
    raw_weight_units = sum(vectors["raw_weight_units"])
    decoded = tokenizer.tokenizer.decode(encoding.ids, skip_special_tokens=False)
    receipt = {
        "schema": "shohin-ocsc-slot-receipt-v2",
        "record_kind": "training_row",
        "corpus": corpus,
        "row_id": row["row_id"],
        "frame_sha256": sha256_bytes(frame.encode("ascii")),
        "decoded_frame_sha256": sha256_bytes(decoded.encode("ascii")),
        "lossless": True,
        "token_count": len(encoding.ids),
        "token_ids_sha256": int_array_sha256(encoding.ids),
        "offsets_sha256": hash_json([list(offset) for offset in encoding.offsets]),
        "supervised_tokens": supervised_tokens,
        "completion_mask_sha256": sha256_bytes(bytes(vectors["completion_mask"])),
        "field_token_counts": dict(sorted(field_counts.items())),
        "raw_weight_units": raw_weight_units,
        "raw_weight_vector_sha256": sha256_bytes(bytes(vectors["raw_weight_units"])),
        "activation_indices": activation,
        "slot_payload": slot_payload_receipt(payload),
    }
    return TokenizedRow(
        corpus=corpus,
        row_id=row["row_id"],
        kind=row["kind"],
        payload=payload,
        token_count=len(encoding.ids),
        supervised_tokens=supervised_tokens,
        raw_weight_units=raw_weight_units,
        activation_indices=activation,
        receipt=with_payload_hash(receipt, "receipt_sha256"),
    )


def tokenize_replay_prompt(tokenizer: FrozenTokenizer, row: dict) -> tuple[dict, dict]:
    encoding = tokenizer.encode(row["prompt"])
    if len(encoding.ids) < 2 or len(encoding.ids) > 128:
        raise ContractError("replay prompt token length must be in [2, 128]")
    decoded = tokenizer.tokenizer.decode(encoding.ids, skip_special_tokens=False)
    receipt = {
        "schema": "shohin-ocsc-replay-receipt-v2",
        "record_kind": "replay_prompt",
        "prompt_id": row["prompt_id"],
        "family": row["family"],
        "frame_sha256": row["prompt_sha256"],
        "decoded_frame_sha256": sha256_bytes(decoded.encode("ascii")),
        "lossless": True,
        "token_count": len(encoding.ids),
        "token_ids": [int(value) for value in encoding.ids],
        "token_ids_sha256": int_array_sha256(encoding.ids),
    }
    replay = {
        "schema": SCHEMA,
        "replay_id": row["prompt_id"],
        "registry_row_sha256": hash_json(row),
        "family": row["family"],
        "prompt": row["prompt"],
        "prompt_sha256": row["prompt_sha256"],
        "normalized_prompt_sha256": row["normalized_prompt_sha256"],
        "semantic_signature_sha256": row["semantic_signature_sha256"],
        "source_commitment": row["source_commitment"],
        "token_count": len(encoding.ids),
        "token_ids": [int(value) for value in encoding.ids],
        "token_ids_sha256": int_array_sha256(encoding.ids),
    }
    return (
        with_payload_hash(replay, "row_sha256"),
        with_payload_hash(receipt, "receipt_sha256"),
    )


def tokenize_corpora(
    tokenizer: FrozenTokenizer,
    ocsc_rows: list[dict],
    iid_rows: list[dict],
    registry: list[dict],
    pad_token_id: int,
) -> tuple[
    dict[str, dict[str, TokenizedRow]],
    list[dict],
    list[dict],
    TokenizedRow,
    list[int],
]:
    tokenized: dict[str, dict[str, TokenizedRow]] = {
        "ocsc": {},
        "iid_control": {},
    }
    receipts = []
    digit_token_ids = frozen_digit_token_ids(tokenizer)
    for corpus, rows in (
        ("ocsc", [row for row in ocsc_rows if row["kind"] == "transition"]),
        (
            "iid_control",
            [row for row in iid_rows if row["kind"] == "transition"],
        ),
        (
            "shared_serializer",
            [row for row in ocsc_rows if row["kind"] == "serializer"],
        ),
    ):
        target_corpora = (
            ("ocsc", "iid_control") if corpus == "shared_serializer" else (corpus,)
        )
        for row in rows:
            item = tokenize_training_row(
                tokenizer, corpus, row, pad_token_id, digit_token_ids
            )
            if any(item.row_id in tokenized[target] for target in target_corpora):
                raise AssertionError("duplicate tokenized row ID")
            for target_corpus in target_corpora:
                tokenized[target_corpus][item.row_id] = item
            receipts.append(item.receipt)
    replay_rows = []
    for row in sorted(
        (record for record in registry if record["use"] == "replay"),
        key=lambda record: (
            record["family"],
            record["prompt_id"],
        ),
    ):
        replay, receipt = tokenize_replay_prompt(tokenizer, row)
        replay_rows.append(replay)
        receipts.append(receipt)
    dummy_vectors = {
        "token_ids": [pad_token_id] * SEQUENCE_LENGTH,
        "attention_mask": [0] * SEQUENCE_LENGTH,
        "completion_mask": [0] * SEQUENCE_LENGTH,
        "field_ids": [0] * SEQUENCE_LENGTH,
        "raw_weight_units": [0] * SEQUENCE_LENGTH,
        "position_ids": [0] * SEQUENCE_LENGTH,
    }
    dummy_payload = slot_payload_bytes(dummy_vectors)
    dummy_receipt = with_payload_hash(
        {
            "schema": "shohin-ocsc-slot-receipt-v2",
            "record_kind": "dummy_slot",
            "corpus": "padding",
            "row_id": "ocsc-canonical-dummy-slot",
            "lossless": True,
            "token_count": 0,
            "supervised_tokens": 0,
            "raw_weight_units": 0,
            "activation_indices": {"kind": "dummy"},
            "slot_payload": slot_payload_receipt(dummy_payload),
        },
        "receipt_sha256",
    )
    receipts.append(dummy_receipt)
    dummy = TokenizedRow(
        corpus="padding",
        row_id="ocsc-canonical-dummy-slot",
        kind="dummy",
        payload=dummy_payload,
        token_count=0,
        supervised_tokens=0,
        raw_weight_units=0,
        activation_indices={"kind": "dummy"},
        receipt=dummy_receipt,
    )
    return tokenized, replay_rows, receipts, dummy, digit_token_ids


def generate_serializer_counterfactual_relations(
    serializer_rows: list[dict], serializer_relations: list[dict]
) -> list[dict]:
    by_id = {row["row_id"]: row for row in serializer_rows}
    grouped = defaultdict(list)
    for relation in serializer_relations:
        left = by_id[relation["left_row_id"]]
        grouped[(left["width"], left["serializer_slice"])].append(relation)
    counterfactuals = []
    for (width, slice_name), positives in sorted(grouped.items()):
        positives = sorted(positives, key=lambda row: row["pair_id"])
        if len(positives) != 50:
            raise AssertionError("serializer positive-pair geometry drifted")
        for group_index in range(0, 50, 2):
            first, second = positives[group_index : group_index + 2]
            endpoints = (
                (first["left_row_id"], second["right_row_id"], 0),
                (second["left_row_id"], first["right_row_id"], 1),
            )
            for left_id, right_id, edge_index in endpoints:
                left_tape = by_id[left_id]["tape"]
                right_tape = by_id[right_id]["tape"]
                differing = sum(
                    left_tape[position] != right_tape[width - 1 - position]
                    for position in range(width)
                )
                if differing < serializer_min_hamming(width):
                    raise AssertionError("serializer mismatch lacks Hamming separation")
                relation = {
                    "schema": SCHEMA,
                    "pair_id": "ocsc-rel-ser-cf-w{}-{}-{:02d}-{}".format(
                        width,
                        slice_name.replace("_", "-"),
                        group_index // 2,
                        edge_index,
                    ),
                    "relation": "serializer_counterfactual_mismatch",
                    "serializer_slice": slice_name,
                    "width": width,
                    "left_row_id": left_id,
                    "right_row_id": right_id,
                    "source_positive_pair_ids": [first["pair_id"], second["pair_id"]],
                    "aligned_differing_tape_positions": differing,
                    "shared_main_forward_required": True,
                }
                counterfactuals.append(with_payload_hash(relation, "pair_sha256"))
    if len(counterfactuals) != 750:
        raise AssertionError("serializer counterfactual count mismatch")
    return counterfactuals


def _slot_descriptor(index: int, item: TokenizedRow) -> dict:
    return {
        "slot_index": index,
        "real": item.kind != "dummy",
        "row_id": None if item.kind == "dummy" else item.row_id,
        "corpus": item.corpus,
        "receipt_sha256": item.receipt["receipt_sha256"],
        "slot_payload_sha256": sha256_bytes(item.payload),
        "token_count": item.token_count,
        "padding_tokens": SEQUENCE_LENGTH - item.token_count,
        "supervised_positions": item.supervised_tokens,
        "field_raw_weight_units": item.raw_weight_units,
        "activation_indices": item.activation_indices,
    }


def _pair_activation_map(
    relation: dict,
    row_ids: list[str],
    tokenized: dict[str, TokenizedRow],
    rows_by_id: dict[str, dict],
    digit_token_ids: list[int],
) -> dict:
    left_slot = row_ids.index(relation["left_row_id"])
    right_slot = row_ids.index(relation["right_row_id"])
    left_item = tokenized[relation["left_row_id"]]
    right_item = tokenized[relation["right_row_id"]]
    if relation["relation"] in {
        "local_prefix_intervention",
        "initial_suffix_context_invariance",
    }:
        package = (
            "initial_invariance"
            if relation["relation"] == "initial_suffix_context_invariance"
            else "local"
        )
        return {
            "pair_id": relation["pair_id"],
            "package": package,
            "factorial_active": relation["factorial_active"],
            "role": relation["role"],
            "incoming_carry": relation["incoming_carry"],
            "polarity": "positive",
            "left_slot": left_slot,
            "right_slot": right_slot,
            "carry_candidate_token_ids": digit_token_ids[:2],
            "digit_candidate_token_ids": digit_token_ids,
            "left": left_item.activation_indices,
            "right": right_item.activation_indices,
        }
    if relation["relation"] not in {
        "serializer_reversal",
        "serializer_counterfactual_mismatch",
    }:
        raise ContractError("unknown relation activation type")
    left_row = rows_by_id[relation["left_row_id"]]
    right_row = rows_by_id[relation["right_row_id"]]
    width = int(left_row["width"])
    alignments = []
    left_predictions = left_item.activation_indices["tape_position_prediction_indices"]
    right_predictions = right_item.activation_indices[
        "tape_position_prediction_indices"
    ]
    counterfactual = relation["relation"] == "serializer_counterfactual_mismatch"
    for left_position in range(width):
        right_position = width - 1 - left_position
        left_prediction = left_predictions[left_position]
        right_prediction = right_predictions[right_position]
        if left_prediction is None or right_prediction is None:
            continue
        left_digit = int(left_row["tape"][left_position])
        right_digit = int(right_row["tape"][right_position])
        if counterfactual and left_digit == right_digit:
            continue
        if not counterfactual and left_digit != right_digit:
            raise AssertionError("positive serializer alignment target mismatch")
        alignments.append(
            {
                "left_prediction_index": left_prediction,
                "left_tape_position": left_position,
                "left_target_digit": left_digit,
                "right_prediction_index": right_prediction,
                "right_tape_position": right_position,
                "right_target_digit": right_digit,
            }
        )
    if not alignments:
        raise ContractError("serializer relation has no visible aligned digit")
    return {
        "pair_id": relation["pair_id"],
        "package": "serializer",
        "polarity": "counterfactual" if counterfactual else "positive",
        "left_slot": left_slot,
        "right_slot": right_slot,
        "digit_candidate_token_ids": digit_token_ids,
        "aligned_positions": alignments,
        "counterfactual_js_margin": (
            {"numerator": 1, "denominator": 10} if counterfactual else None
        ),
    }


def make_batch_pack(
    *,
    pack_id: str,
    skeleton_id: str,
    variant: str,
    corpus: str,
    geometry: dict,
    row_ids: list[str],
    relations: list[dict],
    tokenized: dict[str, TokenizedRow],
    rows_by_id: dict[str, dict],
    dummy: TokenizedRow,
    digit_token_ids: list[int],
) -> dict:
    expected_real = 5 if geometry["kind"] == "transition" else 4
    if len(row_ids) != expected_real:
        raise AssertionError("real slot geometry mismatch")
    items = [tokenized[row_id] for row_id in row_ids]
    items.extend([dummy] * (BATCH_SLOTS - len(items)))
    slots = [_slot_descriptor(index, item) for index, item in enumerate(items)]
    pair_maps = [
        _pair_activation_map(relation, row_ids, tokenized, rows_by_id, digit_token_ids)
        for relation in relations
    ]
    pack = {
        "schema": "shohin-ocsc-independent-batch-v2",
        "pack_id": pack_id,
        "skeleton_id": skeleton_id,
        "variant": variant,
        "corpus": corpus,
        "geometry": geometry,
        "batch_shape": [BATCH_SLOTS, SEQUENCE_LENGTH],
        "real_slots": expected_real,
        "dummy_slots": BATCH_SLOTS - expected_real,
        "row_ids": row_ids,
        "row_presentations": expected_real,
        "nonpadding_tokens": sum(item.token_count for item in items),
        "supervised_positions": sum(item.supervised_tokens for item in items),
        "field_raw_weight_units": sum(item.raw_weight_units for item in items),
        "main_positions": MAIN_POSITIONS_PER_UPDATE,
        "attention": {
            "topology": "eight-independent-block-diagonal-causal-lanes",
            "cross_slot_attention": False,
            "causal_mask_reset_per_slot": True,
            "position_ids_reset_per_slot": True,
            "kv_cache_shared_across_slots": False,
        },
        "slots": slots,
        "resident_pair_ids": [relation["pair_id"] for relation in relations],
        "pair_activation_maps": pair_maps,
        "batch_payload_bytes": BATCH_SLOTS * SLOT_PAYLOAD_BYTES,
        "batch_payload_sha256": sha256_bytes(b"".join(item.payload for item in items)),
    }
    return with_payload_hash(pack, "pack_sha256")


def stratified_skeleton_order(
    skeletons: list[dict], order_domain: str = "ocsc-shared-stratum-v2"
) -> list[dict]:
    grouped = defaultdict(list)
    for skeleton in skeletons:
        geometry = skeleton["geometry"]
        if geometry["kind"] == "transition":
            key = (
                "transition",
                geometry["width"],
                geometry["role"],
                geometry["position"],
                geometry["operation"],
                geometry["incoming_carry"],
            )
        else:
            key = ("serializer", geometry["width"], geometry["serializer_slice"])
        grouped[key].append(skeleton)
    for key in grouped:
        grouped[key].sort(
            key=lambda row: hashlib.sha256(
                (order_domain + "|" + row["skeleton_id"]).encode("ascii")
            ).digest()
        )
    ordered = []
    keys = sorted(grouped, key=lambda value: tuple(str(part) for part in value))
    offset = 0
    while len(ordered) < len(skeletons):
        progressed = False
        for key in keys:
            if offset < len(grouped[key]):
                ordered.append(grouped[key][offset])
                progressed = True
        if not progressed:
            raise AssertionError("stratified skeleton round-robin stalled")
        offset += 1
    if len(ordered) != PACKS_PER_CORPUS:
        raise AssertionError("shared skeleton count mismatch")
    return ordered


def replicate_skeleton_order(skeletons: list[dict], seed: int) -> list[dict]:
    return stratified_skeleton_order(
        skeletons,
        "ocsc-paired-replicate-stratum-v1|seed={}".format(seed),
    )


def build_shared_packs(
    ocsc_rows: list[dict],
    iid_rows: list[dict],
    local_relations: list[dict],
    serializer_relations: list[dict],
    tokenized: dict[str, dict[str, TokenizedRow]],
    dummy: TokenizedRow,
    digit_token_ids: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    serializer_rows = [row for row in ocsc_rows if row["kind"] == "serializer"]
    counterfactuals = generate_serializer_counterfactual_relations(
        serializer_rows, serializer_relations
    )
    all_relations = local_relations + serializer_relations + counterfactuals
    rows_by_id = {row["row_id"]: row for row in ocsc_rows + iid_rows}
    relation_by_cell = defaultdict(list)
    for relation in local_relations:
        relation_by_cell[relation["cell_id"]].append(relation)
    ocsc_by_skeleton = defaultdict(list)
    iid_by_skeleton = defaultdict(list)
    for row in ocsc_rows:
        if row["kind"] == "transition":
            ocsc_by_skeleton[row["skeleton_id"]].append(row)
    for row in iid_rows:
        if row["kind"] == "transition":
            iid_by_skeleton[row["skeleton_id"]].append(row)
    packs = []
    skeletons = []
    for skeleton_id in sorted(ocsc_by_skeleton):
        ocsc_group = sorted(
            ocsc_by_skeleton[skeleton_id], key=lambda row: row["context_index"]
        )
        iid_group = sorted(
            iid_by_skeleton[skeleton_id], key=lambda row: row["context_index"]
        )
        if len(ocsc_group) != 5 or len(iid_group) != 5:
            raise AssertionError("matched transition skeleton geometry mismatch")
        first = ocsc_group[0]
        cell_relations = sorted(
            relation_by_cell[first["cell_id"]], key=lambda row: row["pair_id"]
        )
        geometry = {
            "kind": "transition",
            "width": first["width"],
            "role": first["role"],
            "position": first["position"],
            "operation": first["operation"],
            "incoming_carry": first["incoming_carry"],
            "local_factorial_active": all(
                relation["factorial_active"] for relation in cell_relations
            ),
            "serializer_slice": None,
        }
        for ocsc_row, iid_row in zip(ocsc_group, iid_group):
            left = tokenized["ocsc"][ocsc_row["row_id"]].receipt
            right = tokenized["iid_control"][iid_row["row_id"]].receipt
            if (
                left["token_count"] != right["token_count"]
                or left["completion_mask_sha256"] != right["completion_mask_sha256"]
                or left["supervised_tokens"] != right["supervised_tokens"]
            ):
                raise ContractError("A/B transition slot token geometry mismatch")
        pack_a = make_batch_pack(
            pack_id="pack-a-" + skeleton_id,
            skeleton_id=skeleton_id,
            variant="A",
            corpus="iid_control",
            geometry=geometry,
            row_ids=[row["row_id"] for row in iid_group],
            relations=[],
            tokenized=tokenized["iid_control"],
            rows_by_id=rows_by_id,
            dummy=dummy,
            digit_token_ids=digit_token_ids,
        )
        pack_b = make_batch_pack(
            pack_id="pack-ocsc-" + skeleton_id,
            skeleton_id=skeleton_id,
            variant="OCSC",
            corpus="ocsc",
            geometry=geometry,
            row_ids=[row["row_id"] for row in ocsc_group],
            relations=cell_relations,
            tokenized=tokenized["ocsc"],
            rows_by_id=rows_by_id,
            dummy=dummy,
            digit_token_ids=digit_token_ids,
        )
        for field in (
            "row_presentations",
            "nonpadding_tokens",
            "supervised_positions",
            "main_positions",
        ):
            if pack_a[field] != pack_b[field]:
                raise ContractError("A/B transition resource geometry mismatch")
        packs.extend((pack_a, pack_b))
        skeletons.append(
            {
                "skeleton_id": skeleton_id,
                "geometry": geometry,
                "pack_ids": {
                    "A": pack_a["pack_id"],
                    **{cell: pack_b["pack_id"] for cell in RUN_CELLS if cell != "A"},
                },
            }
        )
    serializer_by_id = {row["row_id"]: row for row in serializer_rows}
    positives_by_slice = defaultdict(list)
    counter_by_source = defaultdict(list)
    for relation in serializer_relations:
        left = serializer_by_id[relation["left_row_id"]]
        positives_by_slice[(left["width"], left["serializer_slice"])].append(relation)
    for relation in counterfactuals:
        counter_by_source[tuple(relation["source_positive_pair_ids"])].append(relation)
    for (width, slice_name), positives in sorted(positives_by_slice.items()):
        positives = sorted(positives, key=lambda row: row["pair_id"])
        for start in range(0, 50, 2):
            selected = positives[start : start + 2]
            source_key = tuple(relation["pair_id"] for relation in selected)
            relations = selected + sorted(
                counter_by_source[source_key], key=lambda row: row["pair_id"]
            )
            row_ids = [
                endpoint
                for relation in selected
                for endpoint in (relation["left_row_id"], relation["right_row_id"])
            ]
            skeleton_id = "skeleton-ser-w{}-{}-{:02d}".format(
                width, slice_name.replace("_", "-"), start // 2
            )
            geometry = {
                "kind": "serializer",
                "width": width,
                "role": None,
                "position": width,
                "operation": serializer_by_id[row_ids[0]]["operation"],
                "incoming_carry": serializer_by_id[row_ids[0]]["incoming_carry"],
                "local_factorial_active": False,
                "serializer_slice": slice_name,
            }
            pack = make_batch_pack(
                pack_id="pack-shared-" + skeleton_id,
                skeleton_id=skeleton_id,
                variant="SHARED",
                corpus="shared_serializer",
                geometry=geometry,
                row_ids=row_ids,
                relations=relations,
                tokenized=tokenized["ocsc"],
                rows_by_id=rows_by_id,
                dummy=dummy,
                digit_token_ids=digit_token_ids,
            )
            packs.append(pack)
            skeletons.append(
                {
                    "skeleton_id": skeleton_id,
                    "geometry": geometry,
                    "pack_ids": {cell: pack["pack_id"] for cell in RUN_CELLS},
                }
            )
    if len(packs) != 9_375 or len(skeletons) != PACKS_PER_CORPUS:
        raise AssertionError("shared pack inventory mismatch")
    return packs, stratified_skeleton_order(skeletons), all_relations


def replay_for_updates(replay_rows: list[dict]) -> dict[int, dict]:
    by_family = {
        family: sorted(
            (row for row in replay_rows if row["family"] == family),
            key=lambda row: row["replay_id"],
        )
        for family in ("drs", "non_dws")
    }
    if any(len(rows) != 640 for rows in by_family.values()):
        raise AssertionError("replay family count mismatch")
    result = {}
    for replay_slot in range(REPLAY_ROWS):
        family = "drs" if replay_slot % 2 == 0 else "non_dws"
        result[replay_slot * 4] = by_family[family][replay_slot // 2]
    return result


def paired_update_rng_seed(seed: int, update: int) -> int:
    return stable_seed("ocsc-paired-update-rng-v1|{}|{}".format(seed, update))


def select_repeat_skeletons(skeleton_order: list[dict], seed: int) -> list[dict]:
    categories = defaultdict(list)
    for skeleton in skeleton_order:
        geometry = skeleton["geometry"]
        if geometry["kind"] == "serializer":
            category = "serializer"
        elif geometry["role"] == "initial":
            category = "initial"
        elif not geometry["local_factorial_active"]:
            category = "inactive_noninitial"
        else:
            category = "active_c{}".format(geometry["incoming_carry"])
        categories[category].append(skeleton)
    target_counts = {
        "serializer": 19,
        "initial": 50,
        "inactive_noninitial": 4,
        "active_c0": 86,
        "active_c1": 86,
    }
    selected = []
    for category, count in target_counts.items():
        candidates = sorted(
            categories[category],
            key=lambda row: hashlib.sha256(
                (
                    "ocsc-balanced-repeat-selection-v2|{}|{}".format(
                        "{}:seed={}".format(category, seed), row["skeleton_id"]
                    )
                ).encode("ascii")
            ).digest(),
        )
        if len(candidates) < count:
            raise AssertionError("insufficient balanced repeat candidates")
        selected.extend(candidates[:count])
    selected.sort(
        key=lambda row: hashlib.sha256(
            (
                "ocsc-balanced-repeat-order-v2|seed={}|{}".format(
                    seed, row["skeleton_id"]
                )
            ).encode("ascii")
        ).digest()
    )
    if (
        len(selected) != REPEATED_PACKS
        or len({row["skeleton_id"] for row in selected}) != REPEATED_PACKS
    ):
        raise AssertionError("balanced repeat selection mismatch")
    return selected


def assert_distinct_repeat_sets(repeat_sets_by_seed: dict[str, set[str]]) -> dict:
    expected_seeds = {str(seed) for seed in PAIRED_SEEDS}
    if set(repeat_sets_by_seed) != expected_seeds:
        raise ContractError("repeat-set seed inventory mismatch")
    hashes = {}
    for seed, repeat_set in repeat_sets_by_seed.items():
        if not isinstance(repeat_set, set) or len(repeat_set) != REPEATED_PACKS:
            raise ContractError("repeat-set cardinality mismatch")
        hashes[seed] = hash_json(sorted(repeat_set))
    if len(set(hashes.values())) != len(PAIRED_SEEDS):
        raise ContractError("paired seeds reuse one repeat-set identity")
    return hashes


def build_schedule(
    packs: list[dict], skeleton_order: list[dict], replay_rows: list[dict]
) -> tuple[list[dict], dict]:
    replay_map = replay_for_updates(replay_rows)
    pack_by_id = {pack["pack_id"]: pack for pack in packs}
    if len(pack_by_id) != len(packs):
        raise AssertionError("duplicate pack ID")
    schedule = []
    run_stats = {}
    repeat_sets_by_seed = {}
    for paired_seed_index, seed in enumerate(PAIRED_SEEDS):
        replicate_order = replicate_skeleton_order(skeleton_order, seed)
        repeat_skeletons = select_repeat_skeletons(replicate_order, seed)
        repeat_set = {row["skeleton_id"] for row in repeat_skeletons}
        repeat_sets_by_seed[str(seed)] = repeat_set
        repeat_set_sha256 = hash_json(sorted(repeat_set))
        cycle = replicate_order + repeat_skeletons
        replicate_cycle_sha256 = hash_json(
            [skeleton["skeleton_id"] for skeleton in cycle]
        )
        replicate_rng_stream_sha256 = hash_json(
            [paired_update_rng_seed(seed, update) for update in range(UPDATES_PER_ARM)]
        )
        for run_cell in RUN_CELLS:
            contract = RUN_CELL_CONTRACT[run_cell]
            occurrences = Counter()
            family_counts = Counter()
            totals = Counter()
            active_pair_occurrences = 0
            active_noninitial_carry_occurrences = Counter()
            for update, skeleton in enumerate(cycle):
                pack = pack_by_id[skeleton["pack_ids"][run_cell]]
                occurrence = occurrences[pack["pack_id"]]
                occurrences[pack["pack_id"]] += 1
                replay = replay_map.get(update)
                if replay is not None:
                    family_counts[replay["family"]] += 1
                active_pairs = [
                    pair_map["pair_id"]
                    for pair_map in pack["pair_activation_maps"]
                    if (
                        pair_map["package"] == "local"
                        and contract["local_relation"]
                        and pair_map["factorial_active"]
                    )
                    or (
                        pair_map["package"] == "serializer"
                        and contract["serializer_relation"]
                    )
                ]
                active_pair_occurrences += len(active_pairs)
                active_ids = set(active_pairs)
                active_noninitial_carry_occurrences.update(
                    pair_map["incoming_carry"]
                    for pair_map in pack["pair_activation_maps"]
                    if pair_map["pair_id"] in active_ids
                    and pair_map["package"] == "local"
                    and pair_map["role"] != "initial"
                )
                supervised = int(pack["supervised_positions"])
                raw_units = (
                    int(pack["field_raw_weight_units"])
                    if contract["field_weights"] == "carry_serializer_v1"
                    else supervised
                )
                totals.update(
                    {
                        "row_presentations": pack["row_presentations"],
                        "nonpadding_tokens": pack["nonpadding_tokens"],
                        "supervised_positions": supervised,
                        "raw_weight_units": raw_units,
                        "main_positions": pack["main_positions"],
                    }
                )
                row = {
                    "schema": "shohin-ocsc-paired-schedule-v2",
                    "paired_seed_index": paired_seed_index,
                    "paired_seed": seed,
                    "replicate_cycle_sha256": replicate_cycle_sha256,
                    "replicate_repeat_set_sha256": repeat_set_sha256,
                    "run_cell": run_cell,
                    "update": update,
                    "update_rng_seed": paired_update_rng_seed(seed, update),
                    "skeleton_id": skeleton["skeleton_id"],
                    "pack_id": pack["pack_id"],
                    "pack_occurrence": occurrence,
                    "cycle": update // PACKS_PER_CORPUS,
                    "corpus": contract["corpus"],
                    "row_presentations": pack["row_presentations"],
                    "nonpadding_tokens": pack["nonpadding_tokens"],
                    "supervised_positions": supervised,
                    "raw_weight_units": raw_units,
                    "main_positions": pack["main_positions"],
                    "resident_pair_ids": pack["resident_pair_ids"],
                    "active_pair_ids": active_pairs,
                    "local_relation_active": contract["local_relation"],
                    "serializer_relation_active": contract["serializer_relation"],
                    "replay_id": replay["replay_id"] if replay else None,
                    "replay_family": replay["family"] if replay else None,
                    "field_weight_profile": contract["field_weights"],
                    "replay_kl_coefficient": {"numerator": 1, "denominator": 10},
                    "relational_coefficient": {"numerator": 1, "denominator": 4},
                }
                schedule.append(with_payload_hash(row, "schedule_row_sha256"))
            repetitions = Counter(occurrences.values())
            if repetitions != Counter({1: 4_630, 2: 245}):
                raise AssertionError("schedule repetition contract mismatch")
            if family_counts != Counter({"drs": 640, "non_dws": 640}):
                raise AssertionError("replay family schedule mismatch")
            expected_carry_dose = (
                Counter({0: 3_622, 1: 3_622})
                if contract["local_relation"]
                else Counter()
            )
            if active_noninitial_carry_occurrences != expected_carry_dose:
                raise AssertionError("scheduled active carry dose is not balanced")
            scale = Fraction(totals["supervised_positions"], totals["raw_weight_units"])
            key = "seed{}:{}".format(paired_seed_index, run_cell)
            run_stats[key] = {
                "paired_seed": seed,
                "replicate_cycle_sha256": replicate_cycle_sha256,
                "replicate_repeat_set_sha256": repeat_set_sha256,
                "replicate_rng_stream_sha256": replicate_rng_stream_sha256,
                "run_cell": run_cell,
                "updates": UPDATES_PER_ARM,
                "unique_skeletons": PACKS_PER_CORPUS,
                "unique_packs": PACKS_PER_CORPUS,
                "every_pack_before_repeat": True,
                "pack_repetition_histogram": {"1": 4_630, "2": 245},
                "replay_prompts": REPLAY_ROWS,
                "replay_family_counts": {"drs": 640, "non_dws": 640},
                "active_pair_occurrences": active_pair_occurrences,
                "active_noninitial_local_pair_presentations_by_incoming_carry": {
                    str(carry): count
                    for carry, count in sorted(
                        active_noninitial_carry_occurrences.items()
                    )
                },
                **dict(totals),
                "normalized_token_weight": {
                    "formula": "raw_unit*scale_numerator/scale_denominator",
                    "scale_numerator": scale.numerator,
                    "scale_denominator": scale.denominator,
                    "scheduled_weight_sum_numerator": totals["supervised_positions"],
                    "scheduled_weight_sum_denominator": 1,
                    "scheduled_mean_numerator": 1,
                    "scheduled_mean_denominator": 1,
                },
            }
    assert_distinct_repeat_sets(repeat_sets_by_seed)
    return schedule, run_stats


def audit_reachability(row: dict) -> None:
    state = parse_state(row["state"])
    expected = parse_state(row["expected_state"])
    if (
        state is None
        or expected is None
        or canonical_state(apply_microstep(state)) != row["expected_state"]
    ):
        raise ContractError("invalid transition solver witness")
    left, right = value_lsf(state["a"]), value_lsf(state["b"])
    natural = state_at(state["op"], left, right, state["w"], state["p"])
    if row["reachability"] == "reachable" and canonical_state(natural) != row["state"]:
        raise ContractError("reachable row is not solver-reachable")
    if row["reachability"] == "interventional":
        if canonical_state(natural) == row["state"]:
            raise ContractError("interventional row is naturally reachable")
        for field in ("op", "w", "p", "c", "a", "b", "z"):
            if natural[field] != state[field]:
                raise ContractError("intervention changed a forbidden state field")


def validate_independent_arithmetic_rows(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        kind = row.get("kind")
        state = independent_parse_state(row.get("state"))
        if state is None or row["state"] != independent_canonical_state(state):
            raise ContractError("independent arithmetic rejected a row state")
        if kind == "transition":
            expected = independent_apply_microstep(state)
            expected_text = independent_canonical_state(expected)
            imported_expected = apply_microstep(dict(state))
            if canonical_state(imported_expected) != expected_text:
                raise ContractError("reviewed arithmetic oracle disagrees with replay")
            expected_target = {
                "digit": int(expected["r"][state["p"]]),
                "outgoing_carry": expected["c"],
            }
            if (
                row.get("expected_state") != expected_text
                or row.get("response") != expected_text
                or row.get("local_target") != expected_target
                or row.get("local_target_sha256") != hash_json(expected_target)
            ):
                raise ContractError("independent transition replay mismatch")
        elif kind == "serializer":
            answer = independent_state_answer(state)
            if state_answer(dict(state)) != answer:
                raise ContractError("reviewed serializer oracle disagrees with replay")
            if row.get("response") != "answer={}".format(answer):
                raise ContractError("independent serializer replay mismatch")
        else:
            raise ContractError("independent arithmetic row kind mismatch")
        counts[kind] += 1
    return {
        "schema": "shohin-ocsc-independent-arithmetic-replay-v1",
        "transition_rows": counts["transition"],
        "serializer_rows": counts["serializer"],
        "reviewed_oracle_sha256": DIGITWISE_PROTOCOL_SHA256,
        "reviewed_oracle_bytes": DIGITWISE_PROTOCOL_BYTES,
        "all_rows_exact": True,
    }


def audit_corpus(
    ocsc_rows: list[dict],
    iid_rows: list[dict],
    relations: list[dict],
    registry: list[dict],
) -> dict:
    independent_replay = validate_independent_arithmetic_rows(ocsc_rows + iid_rows)
    if len(ocsc_rows) != CORPUS_ROWS or len(iid_rows) != CORPUS_ROWS:
        raise ContractError("corpus row count mismatch")
    for corpus, rows in (
        ("ocsc", ocsc_rows),
        ("iid_control", iid_rows),
    ):
        ids = [row["row_id"] for row in rows]
        prompts = [row["normalized_prompt_sha256"] for row in rows]
        if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
            raise ContractError(
                "duplicate row ID or normalized prompt in {}".format(corpus)
            )
        for row in rows:
            payload = dict(row)
            row_hash = payload.pop("row_sha256")
            if hash_json(payload) != row_hash:
                raise ContractError("row payload hash mismatch")
            if row["question"] != row["completion_prompt"] or not row["response"]:
                raise ContractError("invalid SFT boundary")
            assert_ascii(
                canonical_json_bytes(row).decode("ascii"),
                "corpus row",
            )
            if row["kind"] == "transition":
                audit_reachability(row)
            else:
                state = parse_state(row["state"])
                if (
                    state is None
                    or not state["z"]
                    or state_answer(state) != row["expected_answer"]
                ):
                    raise ContractError("invalid serializer witness")

    ocsc_transitions = [row for row in ocsc_rows if row["kind"] == "transition"]
    serializers = [row for row in ocsc_rows if row["kind"] == "serializer"]
    iid_transitions = [row for row in iid_rows if row["kind"] == "transition"]
    if len(ocsc_transitions) != TRANSITION_ROWS or len(serializers) != SERIALIZER_ROWS:
        raise ContractError("OCSC kind count mismatch")
    if len(iid_transitions) != TRANSITION_ROWS:
        raise ContractError("IID kind count mismatch")
    structural_fields = ("width", "role", "position", "operation")
    ocsc_geometry = Counter(
        tuple(row[field] for field in structural_fields) for row in ocsc_transitions
    )
    iid_geometry = Counter(
        tuple(row[field] for field in structural_fields) for row in iid_transitions
    )
    if iid_geometry != ocsc_geometry:
        raise ContractError("IID operation-position geometry is not slot matched")
    overlap = {row["row_sha256"] for row in ocsc_rows} & {
        row["row_sha256"] for row in iid_rows
    }
    if overlap != {row["row_sha256"] for row in serializers}:
        raise ContractError("cross-corpus overlap differs from shared serializer rows")

    cells = defaultdict(list)
    for row in ocsc_transitions:
        cells[row["cell_id"]].append(row)
    if len(cells) != len(WIDTHS) * TRANSITION_CELLS_PER_WIDTH:
        raise ContractError("cell count mismatch")
    for cell_rows in cells.values():
        if sorted(row["context_index"] for row in cell_rows) != list(range(5)):
            raise ContractError("cell context coverage mismatch")
        if len({row["local_target_sha256"] for row in cell_rows}) != 1:
            raise ContractError("cell local target differs across contexts")
        by_context = {row["context_index"]: row for row in cell_rows}
        for left_context, right_context in ((0, 3), (1, 4)):
            relation = (
                "initial_suffix_context_invariance"
                if by_context[left_context]["role"] == "initial"
                else "local_prefix_intervention"
            )
            assert_matched_local_pair(
                by_context[left_context], by_context[right_context], relation
            )
    role_cells = Counter((rows[0]["width"], rows[0]["role"]) for rows in cells.values())
    expected_role_cells = Counter(
        {
            (width, role): count
            for width in WIDTHS
            for role, count in ROLE_CELL_COUNTS.items()
        }
    )
    if role_cells != expected_role_cells:
        raise ContractError("per-width role cell count mismatch")

    terminal_sub_cells = Counter()
    impossible_initial_c1 = 0
    impossible_terminal_borrow = 0
    reachability = Counter()
    interior_positions = Counter()
    for cell_rows in cells.values():
        first = cell_rows[0]
        if first["role"] == "terminal_sub":
            terminal_sub_cells[(first["width"], first["incoming_carry"])] += 1
        if first["role"] == "initial" and first["incoming_carry"] == 1:
            impossible_initial_c1 += 1
        if first["role"] == "terminal_sub" and any(
            row["outgoing_carry"] for row in cell_rows
        ):
            impossible_terminal_borrow += 1
        if first["role"] == "interior":
            interior_positions[(first["width"], first["position"])] += 1
        reachability.update(row["reachability"] for row in cell_rows)
    expected_terminal_sub = Counter()
    for width in WIDTHS:
        expected_terminal_sub[(width, 0)] = 55
        expected_terminal_sub[(width, 1)] = 45
    if terminal_sub_cells != expected_terminal_sub:
        raise ContractError("terminal subtraction feasible split mismatch")
    if impossible_initial_c1 or impossible_terminal_borrow:
        raise ContractError("impossible carry slice was generated")
    for width in WIDTHS:
        counts = [
            interior_positions[(width, position)] for position in range(1, width - 1)
        ]
        if max(counts) - min(counts) > 1 or sum(counts) != 400:
            raise ContractError("interior position round-robin mismatch")

    serializer_stats = {}
    for width in WIDTHS:
        width_rows = [row for row in serializers if row["width"] == width]
        tapes = {row["tape"] for row in width_rows}
        if len(width_rows) != 300 or len(tapes) != 100:
            raise ContractError("serializer width count mismatch")
        by_pair = defaultdict(list)
        for row in width_rows:
            by_pair[row["pair_base_id"]].append(row)
        if len(by_pair) != 50:
            raise ContractError("serializer tape pair count mismatch")
        for pair_rows in by_pair.values():
            pair_tapes = {row["tape"] for row in pair_rows}
            if len(pair_tapes) != 2:
                raise ContractError("serializer pair tape count mismatch")
            first, second = sorted(pair_tapes)
            if first == first[::-1] or second != first[::-1]:
                raise ContractError("serializer reversal contract mismatch")
            if len({row["operand_signature_sha256"] for row in pair_rows}) != 1:
                raise ContractError("serializer pair operands differ")
        marginals = {
            str(position): dict(
                sorted(
                    Counter(tape[position] for tape in tapes).items(),
                    key=lambda item: item[0],
                )
            )
            for position in range(width)
        }
        if any(
            count != 10
            for position in marginals.values()
            for count in position.values()
        ):
            raise ContractError("serializer digit marginal mismatch")
        slice_counts = Counter(row["serializer_slice"] for row in width_rows)
        orientation_counts = Counter(row["orientation"] for row in width_rows)
        if slice_counts != Counter({name: 100 for name in SERIALIZER_SLICES}):
            raise ContractError("serializer slice count mismatch")
        if orientation_counts != Counter({"forward": 150, "reverse": 150}):
            raise ContractError("serializer orientation count mismatch")
        answer_lengths = {
            slice_name: {
                str(length): count
                for length, count in sorted(
                    Counter(
                        row["answer_length"]
                        for row in width_rows
                        if row["serializer_slice"] == slice_name
                    ).items()
                )
            }
            for slice_name in SERIALIZER_SLICES
        }
        if answer_lengths["add_c1"] != {str(width + 1): 100}:
            raise ContractError("carry-one serializer answer length mismatch")
        patterns = serializer_patterns(width)
        pattern_metrics = [serializer_pattern_metrics(pattern) for pattern in patterns]
        minimum_hamming = min(
            min(hamming(left, variant) for variant in translated_orbit(right))
            for index, left in enumerate(patterns)
            for right in patterns[index + 1 :]
        )
        if (
            any(metric["constant_except_one"] for metric in pattern_metrics)
            or any(not metric["non_affine"] for metric in pattern_metrics)
            or any(
                metric["distinct_digits"] < min(width, 3) for metric in pattern_metrics
            )
            or minimum_hamming < serializer_min_hamming(width)
        ):
            raise ContractError("serializer anti-shortcut audit failed")
        serializer_stats[str(width)] = {
            "rows": len(width_rows),
            "tape_pairs": len(by_pair),
            "unique_tapes": len(tapes),
            "non_palindromic_tapes": sum(tape != tape[::-1] for tape in tapes),
            "orientation_counts": dict(sorted(orientation_counts.items())),
            "slice_counts": dict(sorted(slice_counts.items())),
            "lsf_zero_tapes": sum(tape[0] == "0" for tape in tapes),
            "most_significant_zero_tapes": sum(tape[-1] == "0" for tape in tapes),
            "answer_length_histograms": answer_lengths,
            "digit_marginals_by_lsf_position": marginals,
            "orbit_pattern_metrics": pattern_metrics,
            "pairwise_orbit_min_hamming": minimum_hamming,
            "required_pairwise_orbit_min_hamming": serializer_min_hamming(width),
        }

    pair_ids = [pair["pair_id"] for pair in relations]
    if len(relations) != 10_500 or len(set(pair_ids)) != len(pair_ids):
        raise ContractError("relational pair count or ID mismatch")
    by_relation = Counter(pair["relation"] for pair in relations)
    if by_relation != Counter(
        {
            "initial_suffix_context_invariance": 2_000,
            "local_prefix_intervention": 7_000,
            "serializer_reversal": 750,
            "serializer_counterfactual_mismatch": 750,
        }
    ):
        raise ContractError("relational type count mismatch")
    active_noninitial_carry = Counter(
        pair["incoming_carry"]
        for pair in relations
        if pair["relation"] == "local_prefix_intervention" and pair["factorial_active"]
    )
    if active_noninitial_carry != Counter({0: 3_450, 1: 3_450}):
        raise ContractError("active noninitial local carry balance mismatch")
    positive_endpoints = Counter(
        endpoint
        for pair in relations
        if pair["relation"] != "serializer_counterfactual_mismatch"
        for endpoint in (pair["left_row_id"], pair["right_row_id"])
    )
    if any(count != 1 for count in positive_endpoints.values()):
        raise ContractError("positive relational endpoint reuse")
    endpoints = {
        endpoint
        for pair in relations
        for endpoint in (pair["left_row_id"], pair["right_row_id"])
    }
    row_ids = {row["row_id"] for row in ocsc_rows}
    if not endpoints <= row_ids:
        raise ContractError("relational endpoint is missing")

    train_prompt_hashes = {
        row["normalized_prompt_sha256"] for row in ocsc_rows + iid_rows
    }
    train_signatures = {
        row["semantic_signature_sha256"] for row in ocsc_rows + iid_rows
    }
    replay = [row for row in registry if row["use"] == "replay"]
    evaluation = [row for row in registry if row["use"] != "replay"]
    replay_hashes = {row["normalized_prompt_sha256"] for row in replay}
    eval_hashes = {row["normalized_prompt_sha256"] for row in evaluation}
    replay_signatures = {row["semantic_signature_sha256"] for row in replay}
    eval_signatures = {row["semantic_signature_sha256"] for row in evaluation}
    if train_prompt_hashes & (replay_hashes | eval_hashes):
        raise ContractError("training prompt overlaps replay or evaluation")
    if replay_hashes & eval_hashes:
        raise ContractError("replay prompt overlaps evaluation")
    if train_signatures & (replay_signatures | eval_signatures):
        raise ContractError("training semantic signature overlaps replay or evaluation")
    if replay_signatures & eval_signatures:
        raise ContractError("replay semantic signature overlaps evaluation")

    return {
        "independent_arithmetic_replay": independent_replay,
        "corpora": {
            "ocsc": {
                "rows": len(ocsc_rows),
                "transition_rows": len(ocsc_transitions),
                "serializer_rows": len(serializers),
                "cells": len(cells),
                "rows_by_width": dict(
                    sorted(
                        (str(width), count)
                        for width, count in Counter(
                            row["width"] for row in ocsc_rows
                        ).items()
                    )
                ),
                "transition_cells_by_width_role": {
                    "w{}:{}".format(width, role): count
                    for (width, role), count in sorted(role_cells.items())
                },
                "reachability": dict(sorted(reachability.items())),
                "interior_position_cells": {
                    "w{}:p{}".format(width, position): count
                    for (width, position), count in sorted(interior_positions.items())
                },
                "terminal_sub_feasible_cells": {
                    "w{}:c{}".format(width, carry): count
                    for (width, carry), count in sorted(terminal_sub_cells.items())
                },
                "impossible_slices": {
                    "initial_c1": "N/A",
                    "terminal_sub_outgoing_borrow_1": "N/A",
                },
            },
            "iid_control": {
                "rows": len(iid_rows),
                "transition_rows": len(iid_transitions),
                "serializer_rows": sum(row["kind"] == "serializer" for row in iid_rows),
                "rows_by_width": dict(
                    sorted(
                        (str(width), count)
                        for width, count in Counter(
                            row["width"] for row in iid_rows
                        ).items()
                    )
                ),
                "all_transition_rows_reachable": all(
                    row["reachability"] == "reachable" for row in iid_transitions
                ),
            },
            "shared_rows": SERIALIZER_ROWS,
            "unexpected_cross_corpus_rows": 0,
        },
        "serializer": serializer_stats,
        "relations": {
            "pairs": len(relations),
            **dict(sorted(by_relation.items())),
            "active_noninitial_local_pairs_by_incoming_carry": {
                str(carry): count
                for carry, count in sorted(active_noninitial_carry.items())
            },
            "positive_endpoint_reuse": 0,
            "counterfactual_endpoint_reuse_is_preregistered": True,
        },
        "duplicates": {
            "ocsc_row_ids": 0,
            "ocsc_normalized_prompts": 0,
            "iid_row_ids": 0,
            "iid_normalized_prompts": 0,
            "relation_ids": 0,
        },
        "leakage": {
            "train_replay_exact_normalized_prompt_hits": 0,
            "train_evaluation_exact_normalized_prompt_hits": 0,
            "replay_evaluation_exact_normalized_prompt_hits": 0,
            "train_replay_semantic_signature_hits": 0,
            "train_evaluation_semantic_signature_hits": 0,
            "replay_evaluation_semantic_signature_hits": 0,
        },
    }


def audit_packing_and_schedule(
    receipts: list[dict],
    packs: list[dict],
    schedule: list[dict],
    run_stats: dict,
) -> dict:
    if len(receipts) != 46_500 + REPLAY_ROWS + 1:
        raise ContractError("tokenization receipt count mismatch")
    if not all(receipt["lossless"] for receipt in receipts):
        raise ContractError("non-lossless tokenization receipt")
    training_receipts = [
        receipt for receipt in receipts if receipt["record_kind"] == "training_row"
    ]
    for receipt in training_receipts:
        vectors = decode_slot_payload(receipt)
        if (
            sum(vectors["attention_mask"]) != receipt["token_count"]
            or sum(vectors["completion_mask"]) != receipt["supervised_tokens"]
            or sum(vectors["raw_weight_units"]) != receipt["raw_weight_units"]
        ):
            raise ContractError("emitted slot vector totals mismatch")
    if len(packs) != 9_375:
        raise ContractError("pack inventory count mismatch")
    pack_by_id = {pack["pack_id"]: pack for pack in packs}
    if len(pack_by_id) != len(packs):
        raise ContractError("duplicate pack ID")
    for pack in packs:
        expected_real = 5 if pack["geometry"]["kind"] == "transition" else 4
        if (
            pack["batch_shape"] != [8, 256]
            or pack["real_slots"] != expected_real
            or pack["dummy_slots"] != 8 - expected_real
            or len(pack["slots"]) != 8
            or [slot["slot_index"] for slot in pack["slots"]] != list(range(8))
            or any(
                slot["real"] != (index < expected_real)
                for index, slot in enumerate(pack["slots"])
            )
            or pack["attention"]["cross_slot_attention"] is not False
            or pack["attention"]["causal_mask_reset_per_slot"] is not True
            or pack["attention"]["position_ids_reset_per_slot"] is not True
        ):
            raise ContractError("independent [8,256] pack contract mismatch")
    if len(schedule) != len(PAIRED_SEEDS) * len(RUN_CELLS) * UPDATES_PER_ARM:
        raise ContractError("schedule row count mismatch")
    schedule_by_run = defaultdict(list)
    for row in schedule:
        schedule_by_run[(row["paired_seed_index"], row["run_cell"])].append(row)
    serializer_timing_by_seed = {}
    replicate_cycle_hashes = {}
    replicate_rng_hashes = {}
    repeat_sets_by_seed = {}
    ab_resource_equality = True
    for seed_index, seed in enumerate(PAIRED_SEEDS):
        by_cell = {cell: schedule_by_run[(seed_index, cell)] for cell in RUN_CELLS}
        for cell, rows in by_cell.items():
            if len(rows) != UPDATES_PER_ARM:
                raise ContractError("paired run schedule length mismatch")
            if any(row["paired_seed"] != seed for row in rows):
                raise ContractError("paired seed value mismatch")
            if [row["update"] for row in rows] != list(range(UPDATES_PER_ARM)):
                raise ContractError("schedule update sequence mismatch")
            first_repeat = next(
                (index for index, row in enumerate(rows) if row["pack_occurrence"] > 0),
                len(rows),
            )
            if first_repeat != PACKS_PER_CORPUS:
                raise ContractError("a pack repeated before every pack was presented")
            counts = Counter(row["pack_id"] for row in rows)
            if Counter(counts.values()) != Counter({1: 4_630, 2: 245}):
                raise ContractError("schedule floor/ceil repetition mismatch")
            if Counter(
                row["replay_family"] for row in rows if row["replay_id"] is not None
            ) != Counter({"drs": 640, "non_dws": 640}):
                raise ContractError("schedule replay mismatch")
            for row in rows:
                pack = pack_by_id[row["pack_id"]]
                for field in (
                    "row_presentations",
                    "nonpadding_tokens",
                    "supervised_positions",
                    "main_positions",
                ):
                    if row[field] != pack[field]:
                        raise ContractError("schedule resource receipt mismatch")
                if row["update_rng_seed"] != paired_update_rng_seed(
                    seed, row["update"]
                ):
                    raise ContractError("paired update RNG mismatch")
                active = set(row["active_pair_ids"])
                maps = {item["pair_id"]: item for item in pack["pair_activation_maps"]}
                if not active <= set(maps):
                    raise ContractError("active relation is absent from main forward")
                contract = RUN_CELL_CONTRACT[cell]
                expected_active = {
                    pair_id
                    for pair_id, pair_map in maps.items()
                    if (
                        pair_map["package"] == "local"
                        and contract["local_relation"]
                        and pair_map["factorial_active"]
                    )
                    or (
                        pair_map["package"] == "serializer"
                        and contract["serializer_relation"]
                    )
                }
                if active != expected_active:
                    raise ContractError("relation factorial activation mismatch")
        reference_skeletons = [row["skeleton_id"] for row in by_cell["A"]]
        cycle_sha256 = hash_json(reference_skeletons)
        repeat_set = {
            row["skeleton_id"] for row in by_cell["A"] if row["pack_occurrence"] > 0
        }
        repeat_sets_by_seed[str(seed)] = repeat_set
        repeat_set_sha256 = hash_json(sorted(repeat_set))
        replicate_cycle_hashes[str(seed)] = cycle_sha256
        replicate_rng_hashes[str(seed)] = run_stats["seed{}:A".format(seed_index)][
            "replicate_rng_stream_sha256"
        ]
        if any(
            row["replicate_cycle_sha256"] != cycle_sha256
            for rows in by_cell.values()
            for row in rows
        ):
            raise ContractError("replicate cycle commitment mismatch")
        if any(
            row["replicate_repeat_set_sha256"] != repeat_set_sha256
            for rows in by_cell.values()
            for row in rows
        ):
            raise ContractError("replicate repeat-set commitment mismatch")
        for cell in RUN_CELLS:
            stats = run_stats["seed{}:{}".format(seed_index, cell)]
            if stats["replicate_repeat_set_sha256"] != repeat_set_sha256:
                raise ContractError("run-stat repeat-set commitment mismatch")
        for cell in RUN_CELLS[1:]:
            if [row["skeleton_id"] for row in by_cell[cell]] != reference_skeletons:
                raise ContractError("run cells do not share one schedule skeleton")
        for left, right in zip(by_cell["A"], by_cell["B"]):
            if any(
                left[field] != right[field]
                for field in (
                    "row_presentations",
                    "nonpadding_tokens",
                    "supervised_positions",
                    "main_positions",
                )
            ):
                ab_resource_equality = False
            pack = pack_by_id[left["pack_id"]]
            if (
                pack["geometry"]["kind"] == "serializer"
                and left["pack_id"] != right["pack_id"]
            ):
                raise ContractError("A/B serializer pack bytes or timing differ")
        for update in range(UPDATES_PER_ARM):
            reference = by_cell["M00"][update]
            for cell in ("M10", "M01", "M11"):
                candidate = by_cell[cell][update]
                for field in (
                    "skeleton_id",
                    "pack_id",
                    "update_rng_seed",
                    "replay_id",
                    "row_presentations",
                    "nonpadding_tokens",
                    "supervised_positions",
                    "raw_weight_units",
                    "main_positions",
                ):
                    if candidate[field] != reference[field]:
                        raise ContractError("relation factorial resource drift")
        timing = [
            row["update"]
            for row in by_cell["A"]
            if pack_by_id[row["pack_id"]]["geometry"]["kind"] == "serializer"
        ]
        serializer_timing_by_seed[str(seed)] = timing
    repeat_set_hashes = assert_distinct_repeat_sets(repeat_sets_by_seed)
    if len(set(replicate_cycle_hashes.values())) != len(PAIRED_SEEDS) or len(
        set(replicate_rng_hashes.values())
    ) != len(PAIRED_SEEDS):
        raise ContractError("paired seeds reuse one batch permutation or RNG stream")
    if not ab_resource_equality:
        raise ContractError("A/B per-update resource equality failed")

    return {
        "tokenization": {
            "receipt_rows": len(receipts),
            "training_receipts": len(training_receipts),
            "replay_receipts": REPLAY_ROWS,
            "dummy_receipts": 1,
            "lossless_receipts": sum(receipt["lossless"] for receipt in receipts),
            "slot_payload_layout": "256xu32,256xu8,256xu8,256xu8,256xu8,256xu16-le",
            "tokenizer_decode_mismatches": 0,
        },
        "packing": {
            "physical_pack_records": len(packs),
            "transition_A": sum(pack["variant"] == "A" for pack in packs),
            "transition_OCSC": sum(pack["variant"] == "OCSC" for pack in packs),
            "serializer_shared": sum(pack["variant"] == "SHARED" for pack in packs),
            "batch_shape": [8, 256],
            "transition_slot_geometry": {"real": 5, "dummy": 3},
            "serializer_slot_geometry": {"real": 4, "dummy": 4},
            "block_diagonal_causal_attention": True,
            "cross_row_attention": False,
        },
        "schedules": run_stats,
        "shared_skeleton": {
            "unique_slots": PACKS_PER_CORPUS,
            "updates": UPDATES_PER_ARM,
            "repeated_prefix_slots": REPEATED_PACKS,
            "serializer_timing_updates_sha256_by_seed": {
                seed: hash_json(timing)
                for seed, timing in serializer_timing_by_seed.items()
            },
            "serializer_pack_presentations_per_seed": {
                seed: len(timing) for seed, timing in serializer_timing_by_seed.items()
            },
            "replicate_cycle_sha256_by_seed": replicate_cycle_hashes,
            "replicate_rng_stream_sha256_by_seed": replicate_rng_hashes,
            "replicate_repeat_set_sha256_by_seed": repeat_set_hashes,
            "replicate_batch_permutations_distinct": True,
            "replicate_repeat_sets_distinct": True,
            "ab_per_update_row_presentations_equal": True,
            "ab_per_update_nonpadding_tokens_equal": True,
            "ab_per_update_supervised_positions_equal": True,
            "ab_per_update_main_positions_equal": True,
            "serializer_bytes_identical_all_cells": True,
        },
        "contrasts": {
            "B-A": "slot-matched OCSC curriculum versus reachable IID control",
            "M00-B": "field token weights only",
            "relation_factorial": {
                "M00": "neither relation",
                "M10": "local only",
                "M01": "serializer positive-plus-counterfactual only",
                "M11": "both relations",
                "paired_seeds": list(PAIRED_SEEDS),
            },
            "execution_or_promotion_authorized": False,
        },
    }


def source_manifest() -> dict:
    execution = bootstrap_source_identity_contract()
    sources = {}
    snapshots = {}
    for relative in SOURCE_PATHS:
        snapshot = (
            _bootstrap_source_snapshot(relative)
            if execution["source_bound"]
            else read_file_snapshot(
                ROOT / relative,
                "nonclaim local source {}".format(relative),
            )
        )
        if snapshot is None:
            raise ContractError("bound source snapshot is unavailable: " + relative)
        snapshots[relative] = snapshot
        sources[relative] = {
            "bytes": len(snapshot.payload),
            "sha256": snapshot.sha256,
        }
    oracle = sources["train/digitwise_protocol.py"]
    if (
        oracle
        != {
            "bytes": DIGITWISE_PROTOCOL_BYTES,
            "sha256": DIGITWISE_PROTOCOL_SHA256,
        }
        or _REVIEWED_DIGITWISE_PROTOCOL_SNAPSHOT.payload
        != snapshots[ORACLE_SOURCE_PATH].payload
    ):
        raise ContractError("reviewed digitwise protocol source identity mismatch")
    runtime_closure = runtime_closure_contract()
    payload = {
        "sources": sources,
        "runtime_closure": runtime_closure,
        "bootstrap_source_identity": execution,
    }
    return {
        "schema": "shohin-ocsc-source-manifest-v4",
        "sources": sources,
        "runtime_closure": runtime_closure,
        "bootstrap_source_identity": execution,
        "payload_sha256": hash_json(payload),
    }


def validate_source_manifest_contract(contract: dict) -> dict:
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {
            "schema",
            "sources",
            "runtime_closure",
            "bootstrap_source_identity",
            "payload_sha256",
        }
        or contract["schema"] != "shohin-ocsc-source-manifest-v4"
        or not isinstance(contract["sources"], dict)
        or set(contract["sources"]) != set(SOURCE_PATHS)
        or contract["payload_sha256"]
        != hash_json(
            {
                "sources": contract["sources"],
                "runtime_closure": contract["runtime_closure"],
                "bootstrap_source_identity": contract["bootstrap_source_identity"],
            }
        )
    ):
        raise ContractError("source manifest contract mismatch")
    for relative, source in contract["sources"].items():
        if (
            not isinstance(source, dict)
            or set(source) != {"bytes", "sha256"}
            or type(source["bytes"]) is not int
            or source["bytes"] < 0
            or not isinstance(source["sha256"], str)
            or not HEX64_RE.fullmatch(source["sha256"])
        ):
            raise ContractError("source manifest entry mismatch: " + relative)
    if contract["sources"]["train/digitwise_protocol.py"] != {
        "bytes": DIGITWISE_PROTOCOL_BYTES,
        "sha256": DIGITWISE_PROTOCOL_SHA256,
    }:
        raise ContractError("reviewed digitwise protocol source identity mismatch")
    live_execution = bootstrap_source_identity_contract()
    if not recursively_type_strict_equal(
        contract["bootstrap_source_identity"], live_execution
    ):
        raise ContractError("bootstrap execution identity drifted or was forged")
    validate_runtime_closure_contract(contract["runtime_closure"])
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"), "source manifest contract"
    )


def output_parent_contract(output_dir: Path) -> tuple[str, dict]:
    output_dir = Path(output_dir)
    if output_dir.name in {"", ".", ".."} or output_dir.parent == output_dir:
        raise ContractError("publication request output path mismatch")
    parent_fd, parent_resolved, metadata = _open_pinned_directory(
        output_dir.parent,
        "publication output parent",
        reject_other_writes=True,
    )
    try:
        _assert_directory_path_matches_fd(
            output_dir.parent,
            parent_resolved,
            parent_fd,
            "publication output parent",
        )
    finally:
        os.close(parent_fd)
    return str(Path(parent_resolved) / output_dir.name), {
        "resolved_path": parent_resolved,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
    }


def validate_signed_output_identity(
    request: dict, *, require_unpublished: bool
) -> None:
    output_path = request.get("output_dir")
    parent_contract = request.get("output_parent")
    if (
        not isinstance(output_path, str)
        or not output_path.startswith("/")
        or not isinstance(parent_contract, dict)
        or set(parent_contract)
        != {"resolved_path", "device", "inode", "mode", "owner_uid"}
        or not isinstance(parent_contract["resolved_path"], str)
        or type(parent_contract["device"]) is not int
        or type(parent_contract["inode"]) is not int
        or type(parent_contract["mode"]) is not int
        or type(parent_contract["owner_uid"]) is not int
        or Path(output_path).parent != Path(parent_contract["resolved_path"])
    ):
        raise ContractError("signed output identity contract mismatch")
    parent_fd, parent_resolved, metadata = _open_pinned_directory(
        Path(parent_contract["resolved_path"]),
        "signed output parent",
        reject_other_writes=True,
    )
    try:
        live_contract = {
            "resolved_path": parent_resolved,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "owner_uid": metadata.st_uid,
        }
        if live_contract != parent_contract:
            raise ContractError("signed output parent identity drifted")
        if require_unpublished:
            try:
                os.stat(Path(output_path).name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ContractError("signed output path cannot be inspected") from error
            else:
                raise ContractError(
                    "prepublication commitment was not consumed before output"
                )
        _assert_directory_path_matches_fd(
            Path(parent_contract["resolved_path"]),
            parent_resolved,
            parent_fd,
            "signed output parent",
        )
    finally:
        os.close(parent_fd)


def publication_commitment_request(
    mode: str,
    output_dir: Path,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    pad_token_id: int,
    *,
    input_snapshots: dict[str, FileSnapshot] | None = None,
    source_manifest_contract: dict | None = None,
) -> dict:
    if mode not in {"production", "test"}:
        raise ContractError("publication request mode mismatch")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ContractError("publication request pad-token mismatch")
    resolved_output, output_parent = output_parent_contract(output_dir)
    snapshots = custody_input_snapshots(
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        snapshots=input_snapshots,
        label_prefix="publication request",
    )
    inputs = {
        label: custody_snapshot_contract(snapshot)
        for label, snapshot in snapshots.items()
    }
    bound_sources = validate_source_manifest_contract(
        source_manifest()
        if source_manifest_contract is None
        else source_manifest_contract
    )
    request = {
        "schema": "shohin-ocsc-prepublication-request-v1",
        "mode": mode,
        "output_dir": resolved_output,
        "output_parent": output_parent,
        "pad_token_id": pad_token_id,
        "artifact_inventory": list(ARTIFACT_NAMES),
        "deterministic_builder": "build_artifacts-v3-independent-replay",
        "source_manifest": bound_sources,
        "inputs": inputs,
        "independent_review_contract": {
            "required_before_publication": True,
            "receipt_schema": "shohin-ocsc-independent-review-receipt-v1",
            "review_request_schema": "shohin-ocsc-independent-review-request-v1",
            "expected_output_schema": "shohin-ocsc-expected-output-v1",
            "reviewed_source_paths": list(REVIEWED_SOURCE_PATHS),
            "oracle_source_path": ORACLE_SOURCE_PATH,
            "test_signer_has_production_authority": False,
            "production_trust_root_configured": (
                TRUSTED_INDEPENDENT_REVIEW_KEYS["production"] is not None
            ),
        },
        "linux_lustre_qualification_source_contract": {
            "required_before_any_external_qualification": True,
            "receipt_schema": ("shohin-ocsc-linux-lustre-qualification-receipt-v3"),
            "current_status": "NO-GO-pending-fresh-independent-source-review",
            "test_signer_has_production_authority": False,
            "qualification_authority_from_author_tests": False,
        },
        "bundle_publication_authorized": False,
        "missing_reviewed_implementations": [
            "consumer",
            "trainer",
            "evaluator",
            "report-validator",
            "parameter-ledger",
            "train-eval-exclusion",
        ],
        "gpu_execution_authorized": False,
        "promotion_authorized": False,
    }
    return with_payload_hash(request, "payload_sha256")


def publication_signing_payload(unsigned_document: dict) -> bytes:
    return PUBLICATION_SIGNATURE_DOMAIN + canonical_json_bytes(unsigned_document)


def load_publication_commitment(
    path: Path,
    expected_request: dict,
    mode: str,
    *,
    require_unpublished: bool,
    snapshot: FileSnapshot | None = None,
) -> dict:
    path = Path(path)
    snapshot = snapshot or read_file_snapshot(
        path,
        "prepublication commitment",
        exact_mode=0o444,
        custody_root=True,
    )
    resolved_path = _lexical_absolute_path(path, "prepublication commitment")
    if (
        not isinstance(snapshot, FileSnapshot)
        or snapshot.resolved_path != resolved_path
        or not stat.S_ISREG(snapshot.metadata.st_mode)
        or snapshot.metadata.st_nlink != 1
        or stat.S_IMODE(snapshot.metadata.st_mode) != 0o444
        or not stat.S_ISDIR(snapshot.parent_metadata.st_mode)
        or stat.S_IMODE(snapshot.parent_metadata.st_mode) != 0o555
        or snapshot.parent_resolved_path != str(Path(resolved_path).parent)
    ):
        raise ContractError("prepublication commitment snapshot identity mismatch")
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("prepublication commitment must be ASCII") from error
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("prepublication commitment must use one final LF")
    document = strict_json_loads(payload.decode("ascii"), "prepublication commitment")
    if payload != canonical_json_bytes(document, newline=True):
        raise ContractError("prepublication commitment is not canonical JSON")
    expected_keys = {
        "schema",
        "custodian_id",
        "sequence",
        "nonce_hex",
        "request",
        "request_sha256",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    if set(document) != expected_keys:
        raise ContractError("prepublication commitment key mismatch")
    if (
        document["schema"] != "shohin-ocsc-prepublication-commitment-v1"
        or not isinstance(document["custodian_id"], str)
        or not ID_RE.fullmatch(document["custodian_id"])
        or type(document["sequence"]) is not int
        or document["sequence"] <= 0
        or not isinstance(document["nonce_hex"], str)
        or not HEX64_RE.fullmatch(document["nonce_hex"])
        or document["signature_algorithm"] != "ed25519"
        or not isinstance(document["signer_public_key_hex"], str)
        or not HEX64_RE.fullmatch(document["signer_public_key_hex"])
        or not isinstance(document["signature_hex"], str)
        or not HEX128_RE.fullmatch(document["signature_hex"])
    ):
        raise ContractError("prepublication commitment identity mismatch")
    trusted_key = TRUSTED_PUBLICATION_KEYS.get(mode)
    if trusted_key is None or document["signer_public_key_hex"] != trusted_key:
        raise ContractError("prepublication commitment signer is not trusted")
    if not recursively_type_strict_equal(
        document["request"], expected_request
    ) or document["request_sha256"] != hash_json(expected_request):
        raise ContractError("prepublication commitment request/source mismatch")
    unsigned = dict(document)
    signature_hex = unsigned.pop("signature_hex")
    InvalidSignature, Ed25519PublicKey = _trusted_cryptography_symbols()
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_key)).verify(
            bytes.fromhex(signature_hex), publication_signing_payload(unsigned)
        )
    except (InvalidSignature, ValueError) as error:
        raise ContractError("prepublication commitment signature failed") from error
    validate_signed_output_identity(
        expected_request,
        require_unpublished=require_unpublished,
    )
    return {
        "schema": "shohin-ocsc-prepublication-receipt-v1",
        "resolved_path": snapshot.resolved_path,
        "physical_sha256": sha256_bytes(payload),
        "physical_bytes": len(payload),
        "physical_file_device": snapshot.metadata.st_dev,
        "physical_file_inode": snapshot.metadata.st_ino,
        "custody_root_path": snapshot.parent_resolved_path,
        "custody_root_device": snapshot.parent_metadata.st_dev,
        "custody_root_inode": snapshot.parent_metadata.st_ino,
        "custodian_id": document["custodian_id"],
        "sequence": document["sequence"],
        "nonce_hex": document["nonce_hex"],
        "signer_public_key_hex": trusted_key,
        "request_sha256": document["request_sha256"],
        "request": expected_request,
        "signature_verified": True,
    }


def revalidate_publication_receipt(
    publication_receipt: dict,
    mode: str,
    output_dir: Path,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    pad_token_id: int,
    *,
    require_unpublished: bool,
    input_snapshots: dict[str, FileSnapshot] | None = None,
    commitment_snapshot: FileSnapshot | None = None,
    source_manifest_contract: dict | None = None,
) -> dict:
    if not isinstance(publication_receipt, dict):
        raise ContractError("prepublication receipt type mismatch")
    commitment_path = publication_receipt.get("resolved_path")
    if not isinstance(commitment_path, str) or not commitment_path:
        raise ContractError("prepublication receipt path mismatch")
    request = publication_commitment_request(
        mode,
        output_dir,
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        pad_token_id,
        input_snapshots=input_snapshots,
        source_manifest_contract=source_manifest_contract,
    )
    verified = load_publication_commitment(
        Path(commitment_path),
        request,
        mode,
        require_unpublished=require_unpublished,
        snapshot=commitment_snapshot,
    )
    if not recursively_type_strict_equal(publication_receipt, verified):
        raise ContractError("prepublication receipt was forged or drifted")
    return verified


def expected_output_identity(
    publication_request: dict,
    artifacts: dict[str, bytes],
) -> dict:
    """Bind the canonical destination and every byte proposed for publication."""

    if (
        not isinstance(publication_request, dict)
        or publication_request.get("schema") != "shohin-ocsc-prepublication-request-v1"
        or publication_request.get("payload_sha256")
        != hash_json(
            {
                key: value
                for key, value in publication_request.items()
                if key != "payload_sha256"
            }
        )
    ):
        raise ContractError("independent review publication request mismatch")
    if set(artifacts) != set(ARTIFACT_NAMES) or any(
        not isinstance(payload, bytes) for payload in artifacts.values()
    ):
        raise ContractError("independent review artifact inventory/type mismatch")
    files = {
        name: {"bytes": len(artifacts[name]), "sha256": sha256_bytes(artifacts[name])}
        for name in ARTIFACT_NAMES
    }
    identity = {
        "schema": "shohin-ocsc-expected-output-v1",
        "output_dir": publication_request.get("output_dir"),
        "output_parent": publication_request.get("output_parent"),
        "files": files,
        "artifact_inventory_sha256": hash_json(files),
    }
    return with_payload_hash(identity, "payload_sha256")


def independent_review_request(
    publication_request: dict,
    artifacts: dict[str, bytes],
) -> dict:
    """Construct the exact material an independent reviewer must authorize."""

    source_manifest_contract = validate_source_manifest_contract(
        publication_request.get("source_manifest")
        if isinstance(publication_request, dict)
        else None
    )
    inputs = publication_request.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "tokenizer",
        "prompt_registry",
        "secret_confirmation_commitment",
    }:
        raise ContractError("independent review input contract mismatch")
    reviewed_sources = {
        name: source_manifest_contract["sources"][name]
        for name in REVIEWED_SOURCE_PATHS
    }
    request = {
        "schema": "shohin-ocsc-independent-review-request-v1",
        "mode": publication_request.get("mode"),
        "reviewed_source_bytes": reviewed_sources,
        "reviewed_source_bytes_sha256": hash_json(reviewed_sources),
        "reviewed_oracle": {
            "path": ORACLE_SOURCE_PATH,
            **source_manifest_contract["sources"][ORACLE_SOURCE_PATH],
        },
        "tokenizer": inputs["tokenizer"],
        "prompt_registry": inputs["prompt_registry"],
        "hidden_commitment": inputs["secret_confirmation_commitment"],
        "publication_request": publication_request,
        "publication_request_sha256": hash_json(publication_request),
        "expected_output_identity": expected_output_identity(
            publication_request, artifacts
        ),
        "authority": {
            "cpu_bundle_publication_only": True,
            "gpu_execution_authorized": False,
            "promotion_authorized": False,
            "test_signer_has_production_authority": False,
        },
    }
    if request["mode"] not in {"production", "test"}:
        raise ContractError("independent review mode mismatch")
    return with_payload_hash(request, "payload_sha256")


def independent_review_signing_payload(unsigned_document: dict) -> bytes:
    return INDEPENDENT_REVIEW_SIGNATURE_DOMAIN + canonical_json_bytes(unsigned_document)


def load_independent_review_receipt(
    path: Path,
    expected_request: dict,
    mode: str,
    *,
    snapshot: FileSnapshot | None = None,
) -> dict:
    path = Path(path)
    snapshot = snapshot or read_file_snapshot(
        path,
        "independent review receipt",
        exact_mode=0o444,
        custody_root=True,
    )
    resolved_path = _lexical_absolute_path(path, "independent review receipt")
    if (
        not isinstance(snapshot, FileSnapshot)
        or snapshot.resolved_path != resolved_path
        or not stat.S_ISREG(snapshot.metadata.st_mode)
        or snapshot.metadata.st_nlink != 1
        or stat.S_IMODE(snapshot.metadata.st_mode) != 0o444
        or not stat.S_ISDIR(snapshot.parent_metadata.st_mode)
        or stat.S_IMODE(snapshot.parent_metadata.st_mode) != 0o555
        or snapshot.parent_resolved_path != str(Path(resolved_path).parent)
    ):
        raise ContractError("independent review receipt snapshot identity mismatch")
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("independent review receipt must be ASCII") from error
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("independent review receipt must use one final LF")
    document = strict_json_loads(payload.decode("ascii"), "independent review receipt")
    if payload != canonical_json_bytes(document, newline=True):
        raise ContractError("independent review receipt is not canonical JSON")
    expected_keys = {
        "schema",
        "reviewer_id",
        "sequence",
        "nonce_hex",
        "decision",
        "review_request",
        "review_request_sha256",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    if set(document) != expected_keys:
        raise ContractError("independent review receipt key mismatch")
    if (
        document["schema"] != "shohin-ocsc-independent-review-receipt-v1"
        or not isinstance(document["reviewer_id"], str)
        or not ID_RE.fullmatch(document["reviewer_id"])
        or type(document["sequence"]) is not int
        or document["sequence"] <= 0
        or not isinstance(document["nonce_hex"], str)
        or not HEX64_RE.fullmatch(document["nonce_hex"])
        or document["decision"] != "approve-cpu-publication-contract-only"
        or document["signature_algorithm"] != "ed25519"
        or not isinstance(document["signer_public_key_hex"], str)
        or not HEX64_RE.fullmatch(document["signer_public_key_hex"])
        or not isinstance(document["signature_hex"], str)
        or not HEX128_RE.fullmatch(document["signature_hex"])
    ):
        raise ContractError("independent review receipt identity mismatch")
    trusted_key = TRUSTED_INDEPENDENT_REVIEW_KEYS.get(mode)
    if trusted_key is None:
        raise ContractError(
            "production independent-review trust root is not configured"
        )
    if document["signer_public_key_hex"] != trusted_key:
        raise ContractError("independent review signer is not trusted")
    if not recursively_type_strict_equal(
        document["review_request"], expected_request
    ) or document["review_request_sha256"] != hash_json(expected_request):
        raise ContractError("independent review receipt request/output mismatch")
    unsigned = dict(document)
    signature_hex = unsigned.pop("signature_hex")
    InvalidSignature, Ed25519PublicKey = _trusted_cryptography_symbols()
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_key)).verify(
            bytes.fromhex(signature_hex),
            independent_review_signing_payload(unsigned),
        )
    except (InvalidSignature, ValueError) as error:
        raise ContractError("independent review receipt signature failed") from error
    return {
        "schema": "shohin-ocsc-independent-review-verification-v1",
        "resolved_path": snapshot.resolved_path,
        "physical_sha256": snapshot.sha256,
        "physical_bytes": len(snapshot.payload),
        "physical_file_device": snapshot.metadata.st_dev,
        "physical_file_inode": snapshot.metadata.st_ino,
        "custody_root_path": snapshot.parent_resolved_path,
        "custody_root_device": snapshot.parent_metadata.st_dev,
        "custody_root_inode": snapshot.parent_metadata.st_ino,
        "reviewer_id": document["reviewer_id"],
        "sequence": document["sequence"],
        "nonce_hex": document["nonce_hex"],
        "signer_public_key_hex": trusted_key,
        "review_request_sha256": document["review_request_sha256"],
        "review_request": expected_request,
        "signature_verified": True,
    }


def revalidate_independent_review_receipt(
    review_receipt: dict,
    path: Path,
    publication_request: dict,
    artifacts: dict[str, bytes],
    mode: str,
    *,
    snapshot: FileSnapshot | None = None,
) -> dict:
    if not isinstance(review_receipt, dict):
        raise ContractError("independent review receipt type mismatch")
    verified = load_independent_review_receipt(
        path,
        independent_review_request(publication_request, artifacts),
        mode,
        snapshot=snapshot,
    )
    if not recursively_type_strict_equal(review_receipt, verified):
        raise ContractError("independent review receipt was forged or drifted")
    return verified


def frozen_diagnostic_gate_contract() -> dict:
    """Return the exact noncompensatory diagnostic board contract."""

    return {
        "schema": "shohin-ocsc-frozen-diagnostic-gates-v1",
        "replay_v5": {
            "artifact": {"path": REPLAY_V5_PATH, "sha256": REPLAY_V5_SHA256},
            "comparison": {
                "parent_arm": "width",
                "candidate_arm": "term_width",
                "terminal_carry_class_semantics": (
                    "carry_or_borrow_before_final_digit->after_final_digit"
                ),
            },
            "required_report_slices": {
                "terminal_carry_classes": ["00", "01", "10", "11"],
                "operations": ["add", "sub"],
                "widths": [4, 6, 8],
                "outcomes": ["exact_state_gain", "exact_state_loss"],
                "first_mismatch_field_sets": ["c", "c+r", "r"],
                "first_mismatch_before_terminal_digit": True,
                "same_carry_other_field_outcome_reported_separately": True,
                "effective_main_supervision_counts_and_weights": [
                    "matched_feasible_carry_positive",
                    "matched_feasible_carry_negative",
                ],
            },
            "noncompensatory_gates": {
                "class_10_gains_strictly_greater_than_losses_overall": True,
                "class_00_gains_at_least_losses_overall": True,
                "class_00_gains_at_least_losses_within_subtraction": True,
                "expected_carry_one_parent_1_to_0_repairs_strictly_greater_than_new_errors": (
                    True
                ),
                "expected_carry_zero_parent_0_to_1_repairs_strictly_greater_than_new_errors": (
                    True
                ),
                "class_00_first_mismatch_c_involved_overall": {
                    "candidate_strictly_below": 196,
                    "frozen_denominator": 200,
                },
                "class_00_first_mismatch_c_involved_subtraction": {
                    "candidate_strictly_below": 155,
                    "frozen_denominator": 158,
                },
                "positive_only_carry_supplement_forbidden": True,
                "same_carry_other_field_compensation_forbidden": True,
                "class_10_compensation_for_class_00_forbidden": True,
                "aggregate_exactness_compensation_forbidden": True,
            },
        },
        "width_sweep_v2": {
            "artifact": {
                "path": WIDTH_SWEEP_V2_PATH,
                "sha256": WIDTH_SWEEP_V2_SHA256,
            },
            "widths": list(range(2, 11)),
            "polarities": ["positive_final_carry", "negative_no_final_carry"],
            "tolerant_raw_fields": ["p", "c", "r", "z"],
            "positive_raw_failure_fields_by_width": {
                "w2": ["z"],
                "w3": ["z"],
                "w4": ["c"],
                "w5": ["c"],
                "w6": ["c", "r"],
                "w7": ["c"],
                "w8": ["c", "r"],
                "w9": ["c", "r"],
                "w10": ["p", "c", "r", "z"],
            },
            "noncompensatory_gates": {
                "negative_raw_carry_exact": {"minimum": 9, "denominator": 9},
                "strict_terminal_state_positive": {
                    "minimum": 9,
                    "denominator": 9,
                    "required_fields": ["r", "c", "p", "z"],
                },
                "strict_terminal_state_negative": {
                    "minimum": 9,
                    "denominator": 9,
                    "required_fields": ["r", "c", "p", "z"],
                },
                "positive_serializer_terminal_carry_included": True,
                "positive_serializer_all_widths": {
                    "widths": list(range(2, 11)),
                    "minimum": 9,
                    "denominator": 9,
                },
                "negative_serializer_preservation": {
                    "widths": list(range(2, 7)),
                    "minimum": 5,
                    "denominator": 5,
                },
                "negative_serializer_transfer": {
                    "widths": list(range(7, 11)),
                    "minimum": 4,
                    "denominator": 4,
                    "reported_separately_from_preservation": True,
                },
                "strict_parse_failure_may_null_raw_fields": False,
                "width_or_polarity_pooling_authorized": False,
                "raw_field_compensation_authorized": False,
                "transition_serializer_compensation_authorized": False,
            },
        },
        "residual_swap": {
            "artifact": {
                "path": RESIDUAL_SWAP_PATH,
                "sha256": RESIDUAL_SWAP_SHA256,
            },
            "teacher_forced_layer": 29,
            "frozen_parent_observation": {
                "positive_separation_widths": [2, 3, 4, 5, 7, 8, 9, 10],
                "inverted_separation_widths": [6],
                "absolute_c1_logit_exceeds_c0_widths": [2, 3],
            },
            "candidate_calibration_gates": {
                "signed_positive_minus_negative_separation_strictly_positive_widths": (
                    list(range(2, 11))
                ),
                "width_6_residual_inversion_repair_required": True,
                "positive_arm_absolute_c1_minus_c0_strictly_positive_widths": (
                    list(range(2, 11))
                ),
                "autonomous_strict_state_gate_also_required": True,
                "autonomous_serializer_gate_also_required": True,
                "teacher_forced_swap_has_independent_promotion_authority": False,
                "width_pooling_authorized": False,
                "residual_compensation_for_state_or_serializer_authorized": False,
            },
        },
        "decision_rule": {
            "all_required": True,
            "per_seed_required": True,
            "per_width_required": True,
            "per_polarity_required": True,
            "row_pooling_authorized": False,
            "slice_pooling_authorized": False,
            "cross_gate_compensation_authorized": False,
            "aggregate_score_override_authorized": False,
        },
    }


def validate_frozen_diagnostic_gate_contract(contract: dict) -> dict:
    expected = frozen_diagnostic_gate_contract()
    if not recursively_type_strict_equal(contract, expected):
        raise ContractError("frozen diagnostic evaluation contract mismatch")
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"),
        "frozen diagnostic evaluation contract",
    )


def hidden_transition_gates() -> dict[str, dict[str, int]]:
    return {
        key: {
            "minimum": (99 * denominator + 99) // 100,
            "denominator": denominator,
        }
        for key, denominator in hidden_geometry_contract()["transition_slices"].items()
    }


def evaluation_gate_contract() -> dict:
    contract = {
        "status": "frozen-for-future-separate-gpu-prereg-only",
        "frozen_diagnostics": frozen_diagnostic_gate_contract(),
        "development": {
            "replication_policy": {
                "unit": "paired seed",
                "seed_count": 3,
                "distinct_stratified_batch_permutation_required": True,
                "row_level_pooling_as_independent_replication_forbidden": True,
                "report_each_seed_before_any_descriptive_summary": True,
            },
            "per_seed_direct_integer_gates": {
                "B_minus_A_direct_correct_minimum": 60,
                "M00_minus_B_direct_correct_minimum": -6,
                "M00_minus_B_serializer_correct_minimum": 10,
                "M10_minus_M00_direct_correct_minimum": 0,
                "M11_minus_M01_direct_correct_minimum": 0,
                "M01_minus_M00_serializer_correct_minimum": 10,
                "M11_minus_M10_serializer_correct_minimum": 10,
            },
            "primary_local_effect_each_seed": {
                "metric": "hidden noninitial paired carry target-switch exact sites",
                "denominator": 400,
                "M11_absolute_minimum": 392,
                "M11_minus_M01_minimum": 8,
                "M11_minus_M00_minimum": 8,
                "M10_minus_M00_minimum": 0,
                "initial_suffix_invariance_rows_in_numerator": 0,
                "all_three_seed_gates_required": True,
                "pooled_row_numerator_authorized": False,
            },
            "carry_canonicalization_package_board": {
                "base_sites": 170,
                "carry_pairs_per_package_condition": 170,
                "prompt_presentations": 680,
                "slice_geometry": {
                    "w{}:true_c{}".format(width, carry): 17
                    for width in WIDTHS
                    for carry in (0, 1)
                },
                "site_geometry_per_width_and_true_carry": {
                    "interior": 10,
                    "terminal_add": 5,
                    "terminal_sub": 2,
                },
                "compound_canonicalization_package": {
                    "history_retained_package": (
                        "one-cache EOS-suppressed S0 then byte-identical generated/current S1"
                    ),
                    "fresh_current_state_package": (
                        "fresh prompt containing only the same S1 bytes and no S0 bytes"
                    ),
                    "carry_factor": "nominal incoming carry versus flipped incoming carry",
                    "four_prompts_per_base_site": True,
                    "simultaneously_changes": [
                        "old-source presence",
                        "cache reset",
                        "absolute token positions",
                        "prompt framing",
                    ],
                    "source_only_effect_identified": False,
                    "source_specific_attribution_deferred_to": (
                        "separate SCERT source-by-cache-by-position-by-framing factorial"
                    ),
                },
                "required_opening_row_contract": {
                    "site_id": "exact stable ASCII ID",
                    "endpoint": "nominal or carry-flipped",
                    "package_condition": (
                        "history-retained-package or fresh-current-state-package"
                    ),
                    "prompt_bytes": "exact ASCII bytes",
                    "endpoint_token_bytes_hex": "ordered exact target bytes",
                    "endpoint_token_ids": "ordered frozen-tokenizer IDs",
                    "target_token_positions": "absolute per-presentation positions",
                    "prediction_positions": "causal predecessor indices",
                    "local_target": "ordered active digit and outgoing carry",
                    "scoring_contract": "parsed target-switch and full-response exact",
                },
                "reported_separately": {
                    "nominal_full_target_exact_by_package_condition": (
                        "descriptive integer counts"
                    ),
                    "counterfactual_full_target_exact_by_package_condition": (
                        "descriptive integer counts only"
                    ),
                    "raw_output_changed_by_package_condition": (
                        "descriptive integer counts"
                    ),
                    "paired_local_target_switch_exact": (
                        "within each source condition, both parsed outputs match their own ordered (active_digit,outgoing_carry) targets"
                    ),
                    "canonicalization_package_joint_exact": (
                        "all four parsed outputs match the S1-local target assigned to their carry endpoint"
                    ),
                },
                "noncompensatory_gates_each_M11_seed": {
                    "history_retained_package_target_switch": {
                        "minimum": 166,
                        "denominator": 170,
                    },
                    "fresh_current_state_package_target_switch": {
                        "minimum": 166,
                        "denominator": 170,
                    },
                    "canonicalization_package_joint_exact": {
                        "minimum": 163,
                        "denominator": 170,
                    },
                    "each_width_true_carry_slice": {
                        "minimum": 16,
                        "denominator": 17,
                    },
                },
                "counterfactual_accuracy_can_satisfy_target_switch_gate": False,
                "raw_counterfactual_accuracy_causal_authority": False,
                "local_relation_interpretation_requires_all_package_gates": True,
                "source_specific_claim_authorized": False,
            },
        },
        "replay_regression_each_seed_and_cell": {
            "top1_parent_match_overall": {"minimum": 1_268, "denominator": 1_280},
            "top1_parent_match_drs": {"minimum": 634, "denominator": 640},
            "top1_parent_match_non_dws": {"minimum": 634, "denominator": 640},
        },
        "hidden": {
            "direct_overall": {"minimum": 3_564, "denominator": 3_600},
            "noninitial_paired_carry_target_switch_each_M11_seed": {
                "overall": {"minimum": 392, "denominator": 400},
                "each_width": {"minimum": 79, "denominator": 80},
                "interior": {"minimum": 198, "denominator": 200},
                "terminal_add": {"minimum": 124, "denominator": 125},
                "terminal_sub": {"minimum": 75, "denominator": 75},
                "noncompensatory": True,
            },
            "initial_suffix_invariance_pairs": {
                "minimum": 248,
                "denominator": 250,
                "carry_effect_numerator_contribution": 0,
            },
            "transition_slices": hidden_transition_gates(),
            "serializer_slices": {
                key: {"minimum": 99, "denominator": 100}
                for key in expected_serializer_slice_counts()
            },
            "impossible_slices": {
                "initial_c1": "N/A",
                "terminal_sub_outgoing_borrow_1": "N/A",
            },
        },
        "natural_language": "fully_deferred",
        "all_required_no_pooling_no_compensation": {
            "all_required": True,
            "row_pooling_authorized": False,
            "slice_pooling_authorized": False,
            "cross_gate_compensation_authorized": False,
        },
        "execution_authorized": False,
        "promotion_authorized": False,
    }
    return with_payload_hash(contract, "payload_sha256")


def validate_evaluation_gate_contract(contract: dict) -> dict:
    expected = evaluation_gate_contract()
    if not recursively_type_strict_equal(contract, expected):
        raise ContractError("evaluation gate contract mismatch")
    validate_frozen_diagnostic_gate_contract(contract["frozen_diagnostics"])
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"),
        "evaluation gate contract",
    )


def nonexecuting_consumer_contract() -> dict:
    """Describe the future consumer boundary without consuming any data."""

    contract = {
        "schema": "shohin-ocsc-nonexecuting-consumer-contract-v1",
        "status": (
            "schema-only-no-consumer-trainer-evaluator-report-validator-"
            "parameter-ledger-or-train-eval-exclusion-implemented"
        ),
        "accepted_action": "validate-contract-reference-only",
        "required_bundle_contract": {
            "manifest_schema": "shohin-ocsc-bundle-manifest-v2",
            "closed_artifact_inventory": list(ARTIFACT_NAMES),
            "commitments_schema": "shohin-ocsc-commitments-v2",
            "audit_schema": "shohin-ocsc-complete-audit-v2",
            "source_manifest_schema": "shohin-ocsc-source-manifest-v4",
            "evaluation_gate_contract_sha256": evaluation_gate_contract()[
                "payload_sha256"
            ],
        },
        "future_batch_interface": {
            "batch_slots": BATCH_SLOTS,
            "sequence_length": SEQUENCE_LENGTH,
            "updates": UPDATES_PER_ARM,
            "run_cells": list(RUN_CELLS),
            "paired_seeds": list(PAIRED_SEEDS),
            "slot_payload": [
                {"field": field, "bytes_per_value": width}
                for field, width in SLOT_PAYLOAD_LAYOUT
            ],
            "counts_are_json_integers_not_booleans": True,
        },
        "request_exact_fields": {
            "schema": "shohin-ocsc-nonexecuting-consumer-request-v1",
            "action": "validate-contract-reference-only",
            "consumer_contract_sha256": "lowercase-sha256",
            "bundle_manifest_sha256": "lowercase-sha256",
            "source_manifest_sha256": "lowercase-sha256",
            "evaluation_gate_contract_sha256": "lowercase-sha256",
            "run_cell": "one-frozen-run-cell",
            "paired_seed": "one-frozen-json-integer-seed",
            "updates": UPDATES_PER_ARM,
            "batch_slots": BATCH_SLOTS,
            "sequence_length": SEQUENCE_LENGTH,
            "training_requested": False,
            "evaluation_requested": False,
            "publication_requested": False,
        },
        "implemented_outputs": ["validated canonical request copy"],
        "forbidden_outputs": [
            "model tensors",
            "optimizer state",
            "training metrics",
            "evaluation metrics",
            "promotion decision",
        ],
        "trainer_implemented": False,
        "evaluator_implemented": False,
        "report_validator_implemented": False,
        "parameter_ledger_implemented": False,
        "train_eval_exclusion_implemented": False,
        "execution_authorized": False,
        "publication_authorized": False,
        "promotion_authorized": False,
        "claim_authorized": False,
    }
    return with_payload_hash(contract, "payload_sha256")


def validate_nonexecuting_consumer_contract(contract: dict) -> dict:
    expected = nonexecuting_consumer_contract()
    if not recursively_type_strict_equal(contract, expected):
        raise ContractError("nonexecuting consumer contract mismatch")
    return strict_json_loads(
        canonical_json_bytes(contract).decode("ascii"),
        "nonexecuting consumer contract",
    )


def validate_nonexecuting_consumer_request(request: dict) -> dict:
    """Validate one reference request; never fit, evaluate, or publish."""

    exact_fields = {
        "schema",
        "action",
        "consumer_contract_sha256",
        "bundle_manifest_sha256",
        "source_manifest_sha256",
        "evaluation_gate_contract_sha256",
        "run_cell",
        "paired_seed",
        "updates",
        "batch_slots",
        "sequence_length",
        "training_requested",
        "evaluation_requested",
        "publication_requested",
    }
    if not isinstance(request, dict) or set(request) != exact_fields:
        raise ContractError("nonexecuting consumer request field mismatch")
    hashes = (
        "consumer_contract_sha256",
        "bundle_manifest_sha256",
        "source_manifest_sha256",
        "evaluation_gate_contract_sha256",
    )
    if any(
        not isinstance(request[field], str) or not HEX64_RE.fullmatch(request[field])
        for field in hashes
    ):
        raise ContractError("nonexecuting consumer request hash mismatch")
    contract = nonexecuting_consumer_contract()
    if (
        request["schema"] != "shohin-ocsc-nonexecuting-consumer-request-v1"
        or request["action"] != "validate-contract-reference-only"
        or request["consumer_contract_sha256"] != contract["payload_sha256"]
        or request["evaluation_gate_contract_sha256"]
        != evaluation_gate_contract()["payload_sha256"]
        or request["run_cell"] not in RUN_CELLS
        or type(request["paired_seed"]) is not int
        or request["paired_seed"] not in PAIRED_SEEDS
        or type(request["updates"]) is not int
        or request["updates"] != UPDATES_PER_ARM
        or type(request["batch_slots"]) is not int
        or request["batch_slots"] != BATCH_SLOTS
        or type(request["sequence_length"]) is not int
        or request["sequence_length"] != SEQUENCE_LENGTH
        or request["training_requested"] is not False
        or request["evaluation_requested"] is not False
        or request["publication_requested"] is not False
    ):
        raise ContractError("nonexecuting consumer request mismatch")
    return strict_json_loads(
        canonical_json_bytes(request).decode("ascii"),
        "nonexecuting consumer request",
    )


def consumer_interface_contract(source_manifest_contract: dict) -> dict:
    source_manifest_contract = validate_source_manifest_contract(
        source_manifest_contract
    )
    evaluation_contract = evaluation_gate_contract()
    contract = {
        "schema": "shohin-ocsc-consumer-interface-v1",
        "status": "interface-only-consumers-unimplemented",
        "execution_authorized": False,
        "consumer_compatibility_claimed": False,
        "source_binding": {
            "ocsc_source_manifest_sha256": source_manifest_contract["payload_sha256"],
            "evaluation_gate_contract_sha256": evaluation_contract["payload_sha256"],
            "trainer_source": {
                "implementation_status": "unimplemented",
                "reviewed_repo_relative_path": None,
                "sha256": None,
                "must_be_frozen_by_fresh_review_before_execution": True,
            },
            "evaluator_source": {
                "implementation_status": "unimplemented",
                "reviewed_repo_relative_path": None,
                "sha256": None,
                "must_be_frozen_by_fresh_review_before_execution": True,
            },
            "report_validator_source": {
                "implementation_status": "unimplemented",
                "reviewed_repo_relative_path": None,
                "sha256": None,
                "must_be_frozen_by_fresh_review_before_execution": True,
            },
            "parameter_ledger_source": {
                "implementation_status": "unimplemented",
                "reviewed_repo_relative_path": None,
                "sha256": None,
                "must_be_frozen_by_fresh_review_before_execution": True,
            },
            "train_eval_exclusion_source": {
                "implementation_status": "unimplemented",
                "reviewed_repo_relative_path": None,
                "sha256": None,
                "must_be_frozen_by_fresh_review_before_execution": True,
            },
        },
        "trainer_request": {
            "schema": "shohin-ocsc-trainer-request-v1",
            "required_fields": [
                "bundle_manifest_path",
                "bundle_manifest_sha256",
                "ocsc_source_manifest_sha256",
                "trainer_source_path",
                "trainer_source_sha256",
                "gpu_preregistration_path",
                "gpu_preregistration_sha256",
                "linux_lustre_qualification_receipt_path",
                "linux_lustre_qualification_receipt_sha256",
                "parent_checkpoint_path",
                "parent_checkpoint_sha256",
                "run_cell",
                "paired_seed",
                "output_dir",
            ],
            "run_cells": list(RUN_CELLS),
            "paired_seeds": list(PAIRED_SEEDS),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "consumed_bundle_files": list(ARTIFACT_NAMES),
            "descriptor_pinned_full_inventory_required": True,
            "path_reopen_after_authentication_forbidden": True,
            "unknown_fields_rejected": True,
        },
        "trainer_receipt": {
            "schema": "shohin-ocsc-trainer-receipt-v1",
            "required_fields": [
                "trainer_request_sha256",
                "trainer_source_sha256",
                "bundle_manifest_sha256",
                "source_manifest_sha256",
                "parent_checkpoint_sha256",
                "run_cell",
                "paired_seed",
                "checkpoint_path",
                "checkpoint_bytes",
                "checkpoint_sha256",
                "optimizer_state_sha256",
                "schedule_sha256",
                "updates_completed",
                "nonfinite_or_skipped_updates",
                "runtime_closure",
                "payload_sha256",
            ],
            "updates_completed_required": UPDATES_PER_ARM,
            "nonfinite_or_skipped_updates_required": 0,
            "self_attested_source_substitution_authorized": False,
        },
        "evaluator_request": {
            "schema": "shohin-ocsc-evaluator-request-v1",
            "required_fields": [
                "trainer_receipt_path",
                "trainer_receipt_sha256",
                "candidate_checkpoint_sha256",
                "parent_checkpoint_sha256",
                "evaluator_source_path",
                "evaluator_source_sha256",
                "report_validator_source_path",
                "report_validator_source_sha256",
                "evaluation_gate_contract_sha256",
                "frozen_diagnostic_artifacts",
                "development_opening_identity",
                "hidden_opening_identity",
                "output_report_path",
            ],
            "frozen_diagnostic_artifacts": {
                name: board["artifact"]
                for name, board in evaluation_contract["frozen_diagnostics"].items()
                if isinstance(board, dict) and "artifact" in board
            },
            "candidate_and_parent_loaded_from_pinned_bytes": True,
            "unknown_fields_rejected": True,
        },
        "evaluation_report": {
            "schema": "shohin-ocsc-evaluation-report-v1",
            "required_bindings": [
                "evaluator_request_sha256",
                "evaluator_source_sha256",
                "report_validator_source_sha256",
                "trainer_receipt_sha256",
                "candidate_checkpoint_sha256",
                "parent_checkpoint_sha256",
                "bundle_manifest_sha256",
                "source_manifest_sha256",
                "evaluation_gate_contract_sha256",
                "frozen_diagnostic_artifact_sha256s",
                "tokenizer_sha256",
                "prompt_registry_sha256",
                "runtime_closure",
                "payload_sha256",
            ],
            "required_sections": [
                "per_seed_per_cell_development_integer_counts",
                "per_seed_hidden_integer_counts",
                "replay_v5_class_operation_width_gain_loss_slices",
                "width_sweep_raw_field_sets_by_polarity_and_width",
                "width_sweep_strict_state_by_polarity_and_width",
                "serializer_preservation_widths_2_to_6",
                "serializer_transfer_widths_7_to_10",
                "residual_calibration_each_width_including_width_6",
                "all_required_noncompensatory_decision",
            ],
            "all_integer_numerators_and_denominators_required": True,
            "per_seed_reporting_required": True,
            "pooled_rows_authorized": False,
            "missing_slice_authorized": False,
            "cross_gate_compensation_authorized": False,
            "consumer_compatibility_claim_authorized": False,
        },
        "enablement": {
            "separately_reviewed_consumer_required": True,
            "separately_reviewed_trainer_required": True,
            "separately_reviewed_evaluator_required": True,
            "separately_reviewed_report_validator_required": True,
            "separately_reviewed_parameter_ledger_required": True,
            "separately_reviewed_train_eval_exclusion_required": True,
            "source_hashes_must_be_non_null_and_exact": True,
            "tested_real_bundle_consumer_required": True,
            "gpu_preregistration_required": True,
            "linux_lustre_qualification_required": True,
            "current_gate": "NO-GO",
        },
    }
    return with_payload_hash(contract, "payload_sha256")


def future_execution_contract(source_manifest_contract: dict) -> dict:
    return {
        "status": (
            "nonexecutable-until-separate-consumer-trainer-evaluator-report-"
            "validator-parameter-ledger-train-eval-exclusion-and-gpu-prereg-exist"
        ),
        "execution_authorized": False,
        "causal_claim_authorized": False,
        "promotion_authorized": False,
        "consumer_interface": consumer_interface_contract(source_manifest_contract),
        "parent_checkpoint": {
            "sha256": PARENT_CHECKPOINT_SHA256,
            "load": "exact model tensors and exact serialized model config",
            "optimizer_state": "fresh-zero-state",
            "starts": "independent model and optimizer objects per seed and run cell",
        },
        "model": {
            "architecture": "exact architecture serialized by parent checkpoint",
            "trainable_parameters": "all parent model parameters",
            "batch_shape": [8, 256],
            "attention": "independent block-diagonal causal lane per batch row",
            "position_ids": "0..token_count-1 independently in every lane",
            "parameter_and_activation_dtype": "bfloat16",
            "gradient_and_loss_accumulation_dtype": "float32",
            "hardware": "one NVIDIA H100 per independent run",
            "gradient_accumulation_steps": 1,
        },
        "optimizer": {
            "group_assignment": {
                "Muon": (
                    "requires_grad and ndim==2 and name contains neither 'tok' nor 'head'"
                ),
                "AdamW": "every other requires_grad parameter",
                "overlap_or_omission": "fatal",
            },
            "Muon": {
                "learning_rate": {"numerator": 1, "denominator": 50},
                "momentum": {"numerator": 19, "denominator": 20},
                "nesterov": True,
                "newton_schulz_steps": 5,
                "newton_schulz_coefficients_decimal": ["3.4445", "-4.7750", "2.0315"],
                "weight_decay": 0,
            },
            "AdamW": {
                "learning_rate": {"numerator": 3, "denominator": 1_000},
                "betas": [
                    {"numerator": 9, "denominator": 10},
                    {"numerator": 19, "denominator": 20},
                ],
                "epsilon_decimal": "0.00000001",
                "weight_decay": 0,
            },
            "learning_rate_schedule": {
                "kind": "linear-warmup-then-flat-then-linear-decay",
                "warmup_updates": 200,
                "flat_through_update_exclusive": 4_096,
                "decay_updates": 1_024,
                "final_multiplier": {"numerator": 1, "denominator": 10},
            },
            "global_gradient_norm_clip": {"numerator": 1, "denominator": 1},
            "zero_grad_set_to_none": True,
            "skipped_updates": "forbidden; any nonfinite loss or gradient invalidates run",
        },
        "randomness": {
            "paired_seeds": list(PAIRED_SEEDS),
            "per_update_seed_formula": "sha256_u64be('ocsc-paired-update-rng-v1|seed|update')",
            "same_rng_state_across_run_cells_within_seed": True,
            "batch_permutation": (
                "seed-specific stratified sha256 order plus seed-specific balanced repeats"
            ),
            "batch_permutation_identical_across_run_cells_within_seed": True,
            "batch_permutation_must_differ_across_seeds": True,
            "replication_unit": "paired seed, never rows or updates",
            "pooled_pseudoreplication_forbidden": True,
            "dropout_probability": {"numerator": 0, "denominator": 1},
            "torch_compile": False,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "tf32": False,
        },
        "main_loss": {
            "formula": "sum_i(normalized_weight_i*CE_i)/sum_i(normalized_weight_i)",
            "targets": "completion_mask positions only",
            "normalization": "exact emitted scheduled rational scale",
        },
        "replay_kl": {
            "direction": "KL(parent||run_cell)",
            "reference": "frozen parent checkpoint in eval mode",
            "vocabulary": "full tokenizer vocabulary",
            "mask": "positions t where replay attention[t]==1 and attention[t+1]==1",
            "formula": "sum_v p_parent(v)*(log_p_parent(v)-log_p_run(v))",
            "numeric_dtype": "float32",
            "reduction": "sum over valid positions and vocabulary, divided by valid positions",
            "coefficient": {"numerator": 1, "denominator": 10},
        },
        "relational_loss": {
            "source_forward": "main [8,256] forward only; no extra endpoint forward",
            "layer": "final transformer output after final norm and before LM head",
            "candidate_distribution": (
                "softmax over emitted candidate token logits at emitted causal prediction index"
            ),
            "js": ("0.5*KL(p||0.5*(p+q))+0.5*KL(q||0.5*(p+q)), natural log, float32"),
            "local_pair": (
                "0.5*(JS(carry candidates {0,1})+JS(result digit candidates {0..9}))"
            ),
            "local_eligible_relation": (
                "factorial_active noninitial local_prefix_intervention only"
            ),
            "initial_suffix_invariance_activation": "always zero and excluded",
            "local_reduction": (
                "arithmetic mean over eligible active noninitial pairs in the update"
            ),
            "serializer_positive": (
                "arithmetic mean JS over emitted reversal-aligned visible tape positions"
            ),
            "serializer_counterfactual": (
                "arithmetic mean max(0,1/10-JS) over emitted mismatched aligned positions"
            ),
            "serializer_reduction": "positive mean plus counterfactual mean",
            "factorial_cells": {
                "M00": "zero relational loss",
                "M10": "local only",
                "M01": "serializer positive plus counterfactual only",
                "M11": "local plus serializer",
            },
            "coefficient": {"numerator": 1, "denominator": 4},
        },
        "paired_carry_intervention_readout": {
            "scope": "development and hidden active noninitial sites only",
            "true_incoming_carry_balance": "exactly 17 c0 and 17 c1 base sites per width",
            "compound_canonicalization_package": {
                "history_retained_package": (
                    "S0 plus byte-identical S1 in one EOS-suppressed cache"
                ),
                "fresh_current_state_package": "fresh cache with the same S1 and no S0",
                "source_only_effect_identified": False,
                "source_specific_attribution": "deferred to separate SCERT factorial",
            },
            "nominal_full_target_exact": (
                "output under nominal state equals nominal full target"
            ),
            "counterfactual_full_target_exact": (
                "output under carry-flipped state equals counterfactual full target"
            ),
            "raw_output_changed": "nominal output bytes differ from counterfactual output bytes",
            "paired_local_target_switch_exact": (
                "within each package condition, both carry-endpoint outputs parse and match their own local target tuple"
            ),
            "canonicalization_package_joint_exact": (
                "all carry-by-package outputs match the local target encoded by current S1"
            ),
            "stored_per_row_fields": [
                "site_id",
                "endpoint",
                "prompt bytes",
                "endpoint token bytes and IDs",
                "target and prediction positions",
                "ordered local target",
                "exact scoring contract",
            ],
            "counterfactual_accuracy_is_not_a_causal_response_metric": True,
            "noncompensatory_gate": True,
            "initial_suffix_invariance_contributes_to_carry_numerator": False,
            "primary_effect": (
                "per-seed hidden target-switch margin for M11 versus M01 and M00"
            ),
            "local_supervision_alone_does_not_establish_package_robustness": True,
        },
        "total_loss": {
            "formula": (
                "L_main+(1/10)*L_KL+(1/4)*(I_local*L_local+I_serializer*L_serializer)"
            ),
            "coefficient_sum_renormalization": False,
            "missing_component_value": 0,
        },
        "mandatory_before_any_gpu_run": (
            "a separate exact GPU preregistration and tested consumer bound by source hashes"
        ),
    }


def build_artifacts(
    mode: str,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    pad_token_id: int,
    publication_receipt: dict,
    *,
    expected_output_dir: Path,
    require_unpublished: bool,
    input_snapshots: dict[str, FileSnapshot] | None = None,
    publication_commitment_snapshot: FileSnapshot | None = None,
    source_manifest_contract: dict | None = None,
) -> dict[str, bytes]:
    if mode not in {"production", "test"}:
        raise ContractError("mode must be production or test")
    input_snapshots = custody_input_snapshots(
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        snapshots=input_snapshots,
        label_prefix="build",
    )
    live_source_manifest = validate_source_manifest_contract(
        source_manifest()
        if source_manifest_contract is None
        else source_manifest_contract
    )
    commitment_path = (
        publication_receipt.get("resolved_path")
        if isinstance(publication_receipt, dict)
        else None
    )
    if not isinstance(commitment_path, str) or not commitment_path:
        raise ContractError("prepublication receipt path mismatch")
    publication_commitment_snapshot = publication_commitment_snapshot or (
        read_file_snapshot(
            Path(commitment_path),
            "build prepublication commitment",
            exact_mode=0o444,
            custody_root=True,
        )
    )
    publication_receipt = revalidate_publication_receipt(
        publication_receipt,
        mode,
        expected_output_dir,
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        pad_token_id,
        require_unpublished=require_unpublished,
        input_snapshots=input_snapshots,
        commitment_snapshot=publication_commitment_snapshot,
        source_manifest_contract=live_source_manifest,
    )
    tokenizer = FrozenTokenizer(tokenizer_path, mode, input_snapshots["tokenizer"])
    if pad_token_id < 0 or pad_token_id >= tokenizer.vocab_size:
        raise ContractError("pad token ID is outside the tokenizer vocabulary")
    registry, registry_summary = load_prompt_registry(
        prompt_registry_path,
        input_snapshots["prompt_registry"],
    )
    confirmation, confirmation_sha256 = load_confirmation_commitment(
        confirmation_path,
        input_snapshots["secret_confirmation_commitment"],
    )
    signed_request = publication_receipt["request"]
    loaded_inputs = {
        label: custody_snapshot_contract(snapshot)
        for label, snapshot in input_snapshots.items()
    }
    if (
        publication_receipt.get("schema") != "shohin-ocsc-prepublication-receipt-v1"
        or publication_receipt.get("signature_verified") is not True
        or signed_request.get("source_manifest") != live_source_manifest
        or signed_request.get("inputs") != loaded_inputs
    ):
        raise ContractError(
            "prepublication source or input closure drifted before build"
        )

    ocsc_transitions, local_relations = generate_ocsc_transition_rows()
    serializer_rows, serializer_relations = generate_serializer_rows()
    ocsc_rows = ocsc_transitions + serializer_rows
    iid_rows = (
        generate_iid_transition_rows(ocsc_transitions, tokenizer) + serializer_rows
    )

    tokenized, replay_rows, receipts, dummy, digit_token_ids = tokenize_corpora(
        tokenizer, ocsc_rows, iid_rows, registry, pad_token_id
    )
    packs, skeleton_order, relations = build_shared_packs(
        ocsc_rows,
        iid_rows,
        local_relations,
        serializer_relations,
        tokenized,
        dummy,
        digit_token_ids,
    )
    corpus_audit = audit_corpus(ocsc_rows, iid_rows, relations, registry)
    schedule, run_stats = build_schedule(packs, skeleton_order, replay_rows)
    schedule_audit = audit_packing_and_schedule(receipts, packs, schedule, run_stats)
    evaluation_contract = evaluation_gate_contract()
    execution_contract = future_execution_contract(live_source_manifest)
    reference_consumer_contract = nonexecuting_consumer_contract()

    commitments = {
        "schema": "shohin-ocsc-commitments-v2",
        "prompt_registry": registry_summary,
        "development": {
            "reserved_prompt_count": 1_408,
            "commitment_root": registry_summary["roots"]["development"],
            "existing_factorial_board_use": ("development_only"),
            "factorial_heldout_sha256": (FACTORIAL_DEVELOPMENT_HELDOUT_SHA256),
            "factorial_board_sha256": (FACTORIAL_DEVELOPMENT_BOARD_SHA256),
            "promotion_authority": False,
        },
        "secret_confirmation": {
            "reserved_prompt_count": 1_408,
            "registry_commitment_root": registry_summary["roots"][
                "secret_confirmation"
            ],
            "direct_board_commitment": confirmation,
            "direct_board_commitment_sha256": (confirmation_sha256),
            "secret_rows_republished": False,
        },
        "claim_boundary": (
            "Commitments and geometry only. No secret prompt, answer, "
            "pair preimage, model score, training result, or "
            "natural-language claim is present."
        ),
        "future_execution_contract": execution_contract,
        "evaluation_gates": evaluation_contract,
        "nonexecuting_consumer_contract": reference_consumer_contract,
    }
    commitments = with_payload_hash(commitments, "payload_sha256")

    audit = {
        "schema": "shohin-ocsc-complete-audit-v2",
        "mode": mode,
        "cpu_preregistration_eligible": False,
        "cpu_preregistration_review_status": "NO-GO-pending-independent-hostile-review",
        "production_training_eligible": False,
        "contract": {
            "widths": list(WIDTHS),
            "contexts_per_cell": CONTEXT_COUNT,
            "transition_cells_per_width": (TRANSITION_CELLS_PER_WIDTH),
            "transition_rows_per_width": (TRANSITION_ROWS_PER_WIDTH),
            "transition_rows": TRANSITION_ROWS,
            "serializer_rows": SERIALIZER_ROWS,
            "corpus_rows": CORPUS_ROWS,
            "iid_control_rows": CORPUS_ROWS,
            "updates_per_run": UPDATES_PER_ARM,
            "batch_slots": BATCH_SLOTS,
            "sequence_length": SEQUENCE_LENGTH,
            "main_positions_per_update": MAIN_POSITIONS_PER_UPDATE,
            "shared_skeleton_slots": PACKS_PER_CORPUS,
            "run_cells": list(RUN_CELLS),
            "paired_seeds": list(PAIRED_SEEDS),
            "replay_prompts_per_run": REPLAY_ROWS,
            "reserved_evaluation_prompts": EVALUATION_ROWS,
        },
        **corpus_audit,
        **schedule_audit,
        "evaluation_gates": evaluation_contract,
        "future_execution_contract": execution_contract,
        "nonexecuting_consumer_contract": reference_consumer_contract,
        "cpu_only": True,
        "jobs_or_training_launched": False,
        "gpu_execution_authorized": False,
        "promotion_authorized": False,
        "source_manifest": live_source_manifest,
        "prepublication_custody": {
            "receipt_sha256": publication_receipt["physical_sha256"],
            "request_sha256": publication_receipt["request_sha256"],
            "custodian_id": publication_receipt["custodian_id"],
            "sequence": publication_receipt["sequence"],
            "signer_public_key_hex": publication_receipt["signer_public_key_hex"],
            "signature_verified_before_publication": True,
            "postpublication_self_attestation_accepted": False,
        },
        "tokenizer": {
            "sha256": tokenizer.payload_sha256,
            "bytes": tokenizer.payload_bytes,
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": pad_token_id,
            "production_commitment_required": {
                "sha256": KNOWN_TOKENIZER_SHA256,
                "bytes": KNOWN_TOKENIZER_BYTES,
            },
        },
    }
    audit = with_payload_hash(audit, "payload_sha256")

    artifacts = {
        "ocsc_train.jsonl": jsonl_bytes(ocsc_rows),
        "iid_control_train.jsonl": jsonl_bytes(iid_rows),
        "relational_pairs.jsonl": jsonl_bytes(relations),
        "replay_prompts.jsonl": jsonl_bytes(replay_rows),
        "tokenization_receipt.jsonl": jsonl_bytes(receipts),
        "packs.jsonl": jsonl_bytes(packs),
        "schedule.jsonl": jsonl_bytes(schedule),
        "commitments.json": pretty_json_bytes(commitments),
        "audit_report.json": pretty_json_bytes(audit),
    }
    manifest = {
        "schema": "shohin-ocsc-bundle-manifest-v2",
        "mode": mode,
        "files": {
            name: {
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            for name, payload in sorted(artifacts.items())
        },
        "inputs": {
            **manifest_custody_fields("tokenizer", loaded_inputs["tokenizer"]),
            **manifest_custody_fields(
                "prompt_registry",
                loaded_inputs["prompt_registry"],
            ),
            **manifest_custody_fields(
                "secret_confirmation_commitment",
                loaded_inputs["secret_confirmation_commitment"],
            ),
            "prepublication_commitment_sha256": publication_receipt["physical_sha256"],
            "prepublication_commitment_bytes": publication_receipt["physical_bytes"],
            "prepublication_commitment_path": publication_receipt["resolved_path"],
            "prepublication_commitment_file_device": publication_receipt[
                "physical_file_device"
            ],
            "prepublication_commitment_file_inode": publication_receipt[
                "physical_file_inode"
            ],
            "prepublication_commitment_custody_root_path": publication_receipt[
                "custody_root_path"
            ],
            "prepublication_commitment_custody_root_device": publication_receipt[
                "custody_root_device"
            ],
            "prepublication_commitment_custody_root_inode": publication_receipt[
                "custody_root_inode"
            ],
            "prepublication_request_sha256": publication_receipt["request_sha256"],
            "prepublication_custodian_id": publication_receipt["custodian_id"],
            "prepublication_sequence": publication_receipt["sequence"],
            "prepublication_signer_public_key_hex": publication_receipt[
                "signer_public_key_hex"
            ],
            "pad_token_id": pad_token_id,
        },
        "source_manifest": audit["source_manifest"],
        "artifact_inventory_closed": True,
    }
    manifest = with_payload_hash(manifest, "payload_sha256")
    artifacts["manifest.json"] = pretty_json_bytes(manifest)
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise AssertionError("artifact inventory mismatch")
    revalidate_publication_receipt(
        publication_receipt,
        mode,
        expected_output_dir,
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        pad_token_id,
        require_unpublished=require_unpublished,
        input_snapshots=input_snapshots,
        commitment_snapshot=publication_commitment_snapshot,
        source_manifest_contract=live_source_manifest,
    )
    return artifacts


HIDDEN_OPENING_KEYS = {
    "schema",
    "board_id",
    "ordinal",
    "row_id",
    "kind",
    "width",
    "role",
    "position",
    "operation",
    "incoming_carry",
    "reachability",
    "serializer_slice",
    "site_id",
    "pair_id",
    "carry_pair_id",
    "prefix_pair_id",
    "endpoint",
    "prefix_variant",
    "intervention_field",
    "intervention_position",
    "orientation",
    "state",
    "prompt",
    "response",
    "prompt_sha256",
    "normalized_prompt_sha256",
    "semantic_signature_sha256",
    "local_target",
    "scoring_contract",
}


def hidden_leaf_digest(index: int, canonical_row: bytes) -> bytes:
    if type(index) is not int or index < 0 or index >= 2**64:
        raise ContractError("hidden leaf index is outside uint64")
    return hashlib.sha256(
        HIDDEN_LEAF_DOMAIN
        + struct.pack(">Q", index)
        + struct.pack(">Q", len(canonical_row))
        + canonical_row
    ).digest()


def hidden_merkle_root(canonical_rows: list[bytes]) -> str:
    if not canonical_rows:
        raise ContractError("hidden Merkle tree cannot be empty")
    nodes = [hidden_leaf_digest(index, row) for index, row in enumerate(canonical_rows)]
    level = 1
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(
                HIDDEN_NODE_DOMAIN
                + struct.pack(">Q", level)
                + nodes[index]
                + nodes[index + 1]
            ).digest()
            for index in range(0, len(nodes), 2)
        ]
        level += 1
    return hashlib.sha256(
        HIDDEN_ROOT_DOMAIN + struct.pack(">Q", len(canonical_rows)) + nodes[0]
    ).hexdigest()


def read_canonical_jsonl(
    path: Path,
    label: str,
    snapshot: FileSnapshot | None = None,
) -> tuple[list[dict], list[bytes]]:
    snapshot = snapshot or read_file_snapshot(path, label)
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("{} must be ASCII".format(label)) from error
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("{} must end in LF and contain no CR".format(label))
    rows = []
    canonical_rows = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw:
            raise ContractError("{} contains a blank line".format(label))
        text = raw.decode("ascii")
        row = strict_json_loads(text, "{} line {}".format(label, line_number))
        canonical = canonical_json_bytes(row)
        if raw != canonical:
            raise ContractError("{} contains a noncanonical JSON line".format(label))
        rows.append(row)
        canonical_rows.append(canonical)
    return rows, canonical_rows


def hidden_scoring_contract(tokenizer: FrozenTokenizer, state: dict, kind: str) -> dict:
    prompt = (
        transition_prompt(state) if kind == "transition" else serializer_prompt(state)
    )
    response = (
        canonical_state(apply_microstep(state))
        if kind == "transition"
        else "answer={}".format(state_answer(state))
    )
    frame = prompt + "\n" + response + "\n"
    response_start = len(prompt) + 1
    encoding = tokenizer.encode(frame)
    digit_token_ids = frozen_digit_token_ids(tokenizer)
    spans = {
        field: (start, end)
        for start, end, field in field_spans(response, response_start)
    }
    if kind == "transition":
        expected = apply_microstep(state)
        targets = [
            (
                "active_result_digit",
                int(expected["r"][int(state["p"])]),
                spans["r"][0] + int(state["p"]),
            ),
            ("outgoing_carry", int(expected["c"]), spans["c"][0]),
        ]
        metric = "parsed-local-target-and-full-response-exact"
    elif kind == "serializer":
        answer = str(state_answer(state))
        targets = [
            (
                "answer_digit_msd_{}".format(index),
                int(character),
                spans["answer"][0] + index,
            )
            for index, character in enumerate(answer)
        ]
        metric = "ordinary-answer-token-and-full-response-exact"
    else:
        raise ContractError("unknown hidden scoring kind")
    target_token_positions = [
        _exact_character_token(
            encoding,
            character_offset,
            digit_token_ids[value],
            "hidden {} target".format(kind),
        )
        for _, value, character_offset in targets
    ]
    values = [value for _, value, _ in targets]
    return {
        "schema": "shohin-ocsc-hidden-row-score-v1",
        "kind": kind,
        "metric": metric,
        "target_names": [name for name, _, _ in targets],
        "target_values": values,
        "endpoint_token_bytes_hex": [
            str(value).encode("ascii").hex() for value in values
        ],
        "endpoint_token_ids": [digit_token_ids[value] for value in values],
        "target_token_positions": target_token_positions,
        "prediction_positions": [index - 1 for index in target_token_positions],
        "frame_sha256": sha256_bytes(frame.encode("ascii")),
        "frame_token_ids_sha256": int_array_sha256(encoding.ids),
        "parse_contract": "ASCII exact response plus ordered endpoint targets",
    }


def _single_byte_delta(left: str, right: str, label: str) -> None:
    if len(left) != len(right) or sum(a != b for a, b in zip(left, right)) != 1:
        raise ContractError("{} must differ in exactly one byte".format(label))


def _state_singleton_delta(
    left: dict, right: dict, field: str, position: int, label: str
) -> None:
    if field not in {"a", "b", "r"}:
        raise ContractError("{} has an invalid intervention field".format(label))
    for name in left:
        if name == field:
            differences = [
                index
                for index, (a, b) in enumerate(zip(left[name], right[name]))
                if a != b
            ]
            if differences != [position]:
                raise ContractError("{} tape delta is not singleton".format(label))
        elif left[name] != right[name]:
            raise ContractError("{} changed another state field".format(label))


def _local_target(state: dict) -> dict:
    expected = independent_apply_microstep(state)
    return {
        "digit": int(expected["r"][int(state["p"])]),
        "outgoing_carry": int(expected["c"]),
    }


def _validate_hidden_transition_row(
    row: dict, state: dict, tokenizer: FrozenTokenizer
) -> None:
    if row["role"] not in {"initial", *HIDDEN_NONINITIAL_SITE_COUNTS}:
        raise ContractError("hidden transition role mismatch")
    if row["serializer_slice"] is not None or row["orientation"] is not None:
        raise ContractError("hidden transition has serializer-only fields")
    if row["position"] != state["p"] or row["incoming_carry"] != state["c"]:
        raise ContractError("hidden transition state metadata mismatch")
    if row["operation"] != state["op"] or row["width"] != state["w"]:
        raise ContractError("hidden transition operation/width mismatch")
    role = row["role"]
    if (
        (role == "initial" and row["position"] != 0)
        or (role == "interior" and not 0 < row["position"] < row["width"] - 1)
        or (
            role in {"terminal_add", "terminal_sub"}
            and row["position"] != row["width"] - 1
        )
        or (role == "terminal_add" and row["operation"] != "add")
        or (role == "terminal_sub" and row["operation"] != "sub")
    ):
        raise ContractError("hidden transition role geometry mismatch")
    expected = independent_apply_microstep(state)
    if row["response"] != independent_canonical_state(expected):
        raise ContractError("hidden transition response mismatch")
    if row["local_target"] != _local_target(state):
        raise ContractError("hidden transition local target mismatch")
    if row["scoring_contract"] != hidden_scoring_contract(
        tokenizer, state, "transition"
    ):
        raise ContractError("hidden transition scoring contract mismatch")
    if role == "terminal_sub" and expected["c"] != 0:
        raise ContractError("hidden terminal subtraction has outgoing borrow")
    if row["reachability"] not in {
        "reachable",
        "carry_interventional",
        "prefix_interventional",
        "prefix_and_carry_interventional",
    }:
        raise ContractError("hidden reachability label mismatch")


def _validate_hidden_serializer_row(
    row: dict, state: dict, tokenizer: FrozenTokenizer
) -> None:
    if (
        row["role"] is not None
        or row["position"] != row["width"]
        or row["reachability"] != "interventional"
        or row["serializer_slice"] not in SERIALIZER_SLICES
        or row["orientation"] not in {"forward", "reverse"}
        or not isinstance(row["pair_id"], str)
        or not ID_RE.fullmatch(row["pair_id"])
        or not isinstance(row["site_id"], str)
        or not ID_RE.fullmatch(row["site_id"])
        or row["carry_pair_id"] is not None
        or row["prefix_pair_id"] is not None
        or row["prefix_variant"] is not None
        or row["intervention_field"] is not None
        or row["intervention_position"] is not None
        or row["endpoint"] != row["orientation"]
        or row["local_target"] is not None
    ):
        raise ContractError("hidden serializer geometry mismatch")
    expected_operation, expected_carry = {
        "add_c0": ("add", 0),
        "add_c1": ("add", 1),
        "sub_c0": ("sub", 0),
    }[row["serializer_slice"]]
    if (
        not state["z"]
        or state["p"] != state["w"]
        or row["operation"] != state["op"]
        or state["op"] != expected_operation
        or row["incoming_carry"] != state["c"]
        or state["c"] != expected_carry
        or row["width"] != state["w"]
    ):
        raise ContractError("hidden serializer state mismatch")
    if row["response"] != "answer={}".format(independent_state_answer(state)):
        raise ContractError("hidden serializer response mismatch")
    if state["r"] == state["r"][::-1]:
        raise ContractError("hidden serializer palindrome")
    natural = independent_state_at(
        state["op"],
        independent_value_lsf(state["a"]),
        independent_value_lsf(state["b"]),
        state["w"],
        state["w"],
    )
    if state["r"] == natural["r"]:
        raise ContractError("hidden serializer tape equals the natural result")
    if row["scoring_contract"] != hidden_scoring_contract(
        tokenizer, state, "serializer"
    ):
        raise ContractError("hidden serializer scoring contract mismatch")


def _validate_initial_hidden_sites(rows: list[dict]) -> Counter:
    by_site = defaultdict(list)
    for row in rows:
        if row["role"] == "initial":
            by_site[row["site_id"]].append(row)
    if len(by_site) != len(WIDTHS) * HIDDEN_INITIAL_SITES_PER_WIDTH:
        raise ContractError("hidden initial site count mismatch")
    geometry = Counter()
    seen_pair_ids = set()
    for site_id, pair_rows in by_site.items():
        if len(pair_rows) != 2:
            raise ContractError("hidden initial site must contain two rows")
        by_endpoint = {row["endpoint"]: row for row in pair_rows}
        if set(by_endpoint) != {"anchor", "variant"}:
            raise ContractError("hidden initial endpoint mismatch")
        anchor, variant = by_endpoint["anchor"], by_endpoint["variant"]
        pair_id = anchor["pair_id"]
        if (
            not isinstance(pair_id, str)
            or not ID_RE.fullmatch(pair_id)
            or pair_id in seen_pair_ids
            or variant["pair_id"] != pair_id
            or anchor["prefix_pair_id"] != pair_id
            or variant["prefix_pair_id"] != pair_id
            or anchor["carry_pair_id"] is not None
            or variant["carry_pair_id"] is not None
            or anchor["prefix_variant"] is not None
            or variant["prefix_variant"] is not None
        ):
            raise ContractError("hidden initial pair identity mismatch")
        seen_pair_ids.add(pair_id)
        field = anchor["intervention_field"]
        position = anchor["intervention_position"]
        if (
            field not in {"a", "b"}
            or type(position) is not int
            or not 0 < position < anchor["width"]
            or variant["intervention_field"] != field
            or variant["intervention_position"] != position
            or any(row["incoming_carry"] != 0 for row in pair_rows)
            or any(row["reachability"] != "reachable" for row in pair_rows)
        ):
            raise ContractError("hidden initial suffix intervention mismatch")
        anchor_state = parse_state(anchor["state"])
        variant_state = parse_state(variant["state"])
        anchor_target = parse_state(anchor["response"])
        variant_target = parse_state(variant["response"])
        for state in (anchor_state, variant_state):
            natural = independent_state_at(
                state["op"],
                independent_value_lsf(state["a"]),
                independent_value_lsf(state["b"]),
                state["w"],
                0,
            )
            if independent_canonical_state(natural) != independent_canonical_state(
                state
            ):
                raise ContractError("hidden initial row is not solver-reachable")
        _state_singleton_delta(
            anchor_state, variant_state, field, position, "hidden initial source"
        )
        _state_singleton_delta(
            anchor_target, variant_target, field, position, "hidden initial target"
        )
        _single_byte_delta(anchor["state"], variant["state"], "hidden initial state")
        _single_byte_delta(anchor["prompt"], variant["prompt"], "hidden initial prompt")
        _single_byte_delta(
            anchor["response"], variant["response"], "hidden initial response"
        )
        if anchor["local_target"] != variant["local_target"]:
            raise ContractError("hidden initial suffix changed the local target")
        for endpoint, row in by_endpoint.items():
            geometry[(row["width"], "initial", 0, endpoint)] += 1
    return geometry


def _validate_noninitial_hidden_sites(rows: list[dict]) -> Counter:
    by_site = defaultdict(list)
    for row in rows:
        if row["role"] != "initial":
            by_site[row["site_id"]].append(row)
    expected_sites = len(WIDTHS) * sum(HIDDEN_NONINITIAL_SITE_COUNTS.values())
    if len(by_site) != expected_sites:
        raise ContractError("hidden noninitial site count mismatch")
    geometry = Counter()
    seen_carry_pairs = set()
    seen_prefix_pairs = set()
    for site_id, site_rows in by_site.items():
        if len(site_rows) != 4:
            raise ContractError("hidden noninitial site must contain four rows")
        cells = {(row["prefix_variant"], row["endpoint"]): row for row in site_rows}
        expected_cells = {
            (prefix, "c{}".format(carry))
            for prefix in HIDDEN_PREFIX_VARIANTS
            for carry in (0, 1)
        }
        if set(cells) != expected_cells:
            raise ContractError("hidden carry/prefix site cell mismatch")
        first = site_rows[0]
        field = first["intervention_field"]
        position = first["intervention_position"]
        if (
            first["role"] not in HIDDEN_NONINITIAL_SITE_COUNTS
            or field != "r"
            or type(position) is not int
            or not 0 <= position < first["position"]
            or any(row["pair_id"] is not None for row in site_rows)
            or any(row["intervention_field"] != field for row in site_rows)
            or any(row["intervention_position"] != position for row in site_rows)
            or len(
                {
                    (row["width"], row["role"], row["position"], row["operation"])
                    for row in site_rows
                }
            )
            != 1
        ):
            raise ContractError("hidden noninitial site metadata mismatch")
        parsed = {key: parse_state(row["state"]) for key, row in cells.items()}
        targets = {key: parse_state(row["response"]) for key, row in cells.items()}
        for prefix in HIDDEN_PREFIX_VARIANTS:
            c0, c1 = cells[(prefix, "c0")], cells[(prefix, "c1")]
            c0_state, c1_state = parsed[(prefix, "c0")], parsed[(prefix, "c1")]
            for name in c0_state:
                if name == "c":
                    if (c0_state[name], c1_state[name]) != (0, 1):
                        raise ContractError("hidden carry endpoints are not c0/c1")
                elif c0_state[name] != c1_state[name]:
                    raise ContractError("hidden carry pair changed another field")
            _single_byte_delta(c0["state"], c1["state"], "hidden carry state")
            _single_byte_delta(c0["prompt"], c1["prompt"], "hidden carry prompt")
            carry_pair_id = c0["carry_pair_id"]
            if (
                not isinstance(carry_pair_id, str)
                or not ID_RE.fullmatch(carry_pair_id)
                or c1["carry_pair_id"] != carry_pair_id
                or carry_pair_id in seen_carry_pairs
                or c0["local_target"] == c1["local_target"]
                or c0["local_target"]["digit"] == c1["local_target"]["digit"]
            ):
                raise ContractError("hidden carry target-switch semantics mismatch")
            seen_carry_pairs.add(carry_pair_id)
        for carry in (0, 1):
            anchor = cells[("anchor", "c{}".format(carry))]
            intervention = cells[("intervention", "c{}".format(carry))]
            anchor_state = parsed[("anchor", "c{}".format(carry))]
            intervention_state = parsed[("intervention", "c{}".format(carry))]
            anchor_target = targets[("anchor", "c{}".format(carry))]
            intervention_target = targets[("intervention", "c{}".format(carry))]
            _state_singleton_delta(
                anchor_state,
                intervention_state,
                "r",
                position,
                "hidden prefix source",
            )
            _state_singleton_delta(
                anchor_target,
                intervention_target,
                "r",
                position,
                "hidden prefix target",
            )
            _single_byte_delta(
                anchor["state"], intervention["state"], "hidden prefix state"
            )
            _single_byte_delta(
                anchor["prompt"], intervention["prompt"], "hidden prefix prompt"
            )
            _single_byte_delta(
                anchor["response"],
                intervention["response"],
                "hidden prefix response",
            )
            prefix_pair_id = anchor["prefix_pair_id"]
            if (
                not isinstance(prefix_pair_id, str)
                or not ID_RE.fullmatch(prefix_pair_id)
                or intervention["prefix_pair_id"] != prefix_pair_id
                or prefix_pair_id in seen_prefix_pairs
                or anchor["local_target"] != intervention["local_target"]
            ):
                raise ContractError("hidden prefix intervention semantics mismatch")
            seen_prefix_pairs.add(prefix_pair_id)
        natural = independent_state_at(
            first["operation"],
            independent_value_lsf(parsed[("anchor", "c0")]["a"]),
            independent_value_lsf(parsed[("anchor", "c0")]["b"]),
            first["width"],
            first["position"],
        )
        natural_carry = int(natural["c"])
        natural_anchor = parsed[("anchor", "c{}".format(natural_carry))]
        if independent_canonical_state(natural_anchor) != independent_canonical_state(
            natural
        ):
            raise ContractError("hidden noninitial anchor is not solver-reachable")
        for (prefix, endpoint), row in cells.items():
            carry = int(endpoint[-1])
            expected_reachability = {
                ("anchor", True): "reachable",
                ("anchor", False): "carry_interventional",
                ("intervention", True): "prefix_interventional",
                ("intervention", False): "prefix_and_carry_interventional",
            }[(prefix, carry == natural_carry)]
            if row["reachability"] != expected_reachability:
                raise ContractError("hidden noninitial reachability mismatch")
            geometry[(row["width"], row["role"], carry, prefix)] += 1
    return geometry


def _validate_hidden_transition_semantic_coverage(rows: list[dict]) -> dict:
    by_width_role = defaultdict(list)
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site_id"]].append(row)
    for site_id, site_rows in by_site.items():
        role = site_rows[0]["role"]
        reachable = [row for row in site_rows if row["reachability"] == "reachable"]
        if role == "initial":
            natural_candidates = [
                row for row in reachable if row["endpoint"] == "anchor"
            ]
        else:
            natural_candidates = reachable
        if len(natural_candidates) != 1:
            raise ContractError("hidden site lacks one solver-reachable anchor")
        natural_row = natural_candidates[0]
        natural_state = independent_parse_state(natural_row["state"])
        if natural_state is None:
            raise ContractError("hidden semantic coverage state is invalid")
        by_width_role[(natural_row["width"], role)].append(
            {
                "site_id": site_id,
                "operation": natural_state["op"],
                "position": natural_state["p"],
                "left_digit": int(natural_state["a"][natural_state["p"]]),
                "right_digit": int(natural_state["b"][natural_state["p"]]),
                "natural_tuple": (
                    natural_state["op"],
                    natural_state["p"],
                    natural_state["c"],
                    int(natural_state["a"][natural_state["p"]]),
                    int(natural_state["b"][natural_state["p"]]),
                ),
                "anchor_endpoint_tuples": [
                    (
                        endpoint_state["op"],
                        endpoint_state["p"],
                        endpoint_state["c"],
                        int(endpoint_state["a"][endpoint_state["p"]]),
                        int(endpoint_state["b"][endpoint_state["p"]]),
                        independent_apply_microstep(endpoint_state)["r"][
                            endpoint_state["p"]
                        ],
                        independent_apply_microstep(endpoint_state)["c"],
                    )
                    for endpoint_state in (
                        independent_parse_state(row["state"])
                        for row in site_rows
                        if (
                            (role == "initial" and row["endpoint"] == "anchor")
                            or (role != "initial" and row["prefix_variant"] == "anchor")
                        )
                    )
                ],
            }
        )

    return _validate_hidden_transition_semantic_record_map(by_width_role)


def _validate_hidden_transition_semantic_record_map(
    by_width_role: dict[tuple[int, str], list[dict]],
) -> dict:
    contract = hidden_transition_semantic_contract()
    summary = {}
    for width in WIDTHS:
        for role, expected_count in contract["site_counts_per_width_role"].items():
            records = by_width_role[(width, role)]
            if len(records) != expected_count:
                raise ContractError("hidden semantic site-count mismatch")
            operation_counts = Counter(row["operation"] for row in records)
            required_operations = set(contract["required_operations_by_role"][role])
            if set(operation_counts) != required_operations or any(
                operation_counts[operation]
                < contract["minimum_sites_per_required_operation"][role]
                for operation in required_operations
            ):
                raise ContractError("hidden operation coverage collapsed")
            positions = Counter(row["position"] for row in records)
            if role == "initial":
                expected_positions = {0}
            elif role == "interior":
                expected_positions = set(range(1, width - 1))
            else:
                expected_positions = {width - 1}
            if set(positions) != expected_positions:
                raise ContractError("hidden position coverage collapsed")
            if (
                role == "interior"
                and max(positions.values()) - min(positions.values())
                > contract["interior_position_count_max_delta"]
            ):
                raise ContractError("hidden interior position balance collapsed")
            minimum_digit_count = contract["minimum_sites_per_required_active_digit"][
                role
            ]
            left_counts = Counter(row["left_digit"] for row in records)
            right_counts = Counter(row["right_digit"] for row in records)
            if any(
                left_counts[digit] < minimum_digit_count
                for digit in contract["required_active_left_digits"][role]
            ) or any(
                right_counts[digit] < minimum_digit_count
                for digit in contract["required_active_right_digits"][role]
            ):
                raise ContractError("hidden active-operand coverage collapsed")
            natural_tuples = {row["natural_tuple"] for row in records}
            endpoint_tuples = {
                item for row in records for item in row["anchor_endpoint_tuples"]
            }
            expected_endpoint_count = expected_count * (1 if role == "initial" else 2)
            if len(natural_tuples) != expected_count:
                raise ContractError("hidden natural local-transition tuples collapsed")
            if len(endpoint_tuples) != expected_endpoint_count:
                raise ContractError("hidden endpoint local-transition tuples collapsed")
            key = "w{}:{}".format(width, role)
            summary[key] = {
                "sites": expected_count,
                "operation_counts": dict(sorted(operation_counts.items())),
                "position_counts": {
                    str(position): count
                    for position, count in sorted(positions.items())
                },
                "active_left_digits": sorted(left_counts),
                "active_right_digits": sorted(right_counts),
                "unique_natural_local_tuples": len(natural_tuples),
                "unique_anchor_endpoint_transition_tuples": len(endpoint_tuples),
            }
    return summary


def _normalized_translation(tape: str) -> tuple[int, ...]:
    digits = tuple(int(character) for character in tape)
    offset = digits[0]
    return tuple((digit - offset) % 10 for digit in digits)


def _serializer_orbit_key(tape: str) -> tuple[int, ...]:
    return min(_normalized_translation(tape), _normalized_translation(tape[::-1]))


def _validate_hidden_serializer_sites(rows: list[dict]) -> Counter:
    pair_rows = defaultdict(list)
    site_rows = defaultdict(list)
    for row in rows:
        pair_rows[(row["width"], row["serializer_slice"], row["pair_id"])].append(row)
        site_rows[(row["width"], row["site_id"])].append(row)
    if len(pair_rows) != 750 or len(site_rows) != len(WIDTHS) * 50:
        raise ContractError("hidden serializer pair/site count mismatch")
    geometry = Counter()
    for (width, slice_name, _), endpoints in pair_rows.items():
        by_orientation = {row["orientation"]: row for row in endpoints}
        if len(endpoints) != 2 or set(by_orientation) != {"forward", "reverse"}:
            raise ContractError("hidden serializer pair opening mismatch")
        forward, reverse = by_orientation["forward"], by_orientation["reverse"]
        forward_state = parse_state(forward["state"])
        reverse_state = parse_state(reverse["state"])
        if (
            forward["site_id"] != reverse["site_id"]
            or forward_state["r"] != reverse_state["r"][::-1]
            or forward_state["r"] == forward_state["r"][::-1]
            or any(
                forward_state[field] != reverse_state[field]
                for field in ("op", "w", "p", "c", "a", "b", "z")
            )
        ):
            raise ContractError("hidden serializer pair operands or reversal mismatch")
        geometry[(width, slice_name)] += 2
    seen_site_signatures = set()
    for (width, _), rows_at_site in site_rows.items():
        if len(rows_at_site) != 6:
            raise ContractError("hidden serializer site must contain six rows")
        states = [parse_state(row["state"]) for row in rows_at_site]
        if (
            {row["serializer_slice"] for row in rows_at_site} != set(SERIALIZER_SLICES)
            or Counter(row["orientation"] for row in rows_at_site)
            != Counter({"forward": 3, "reverse": 3})
            or len({(state["a"], state["b"]) for state in states}) != 1
            or len({state["r"] for state in states}) != 2
        ):
            raise ContractError("hidden serializer site contract mismatch")
        signature = hash_json(
            {
                "width": width,
                "a": states[0]["a"],
                "b": states[0]["b"],
                "tapes": sorted({state["r"] for state in states}),
            }
        )
        if signature in seen_site_signatures:
            raise ContractError("duplicate hidden serializer site signature")
        seen_site_signatures.add(signature)
    for width in WIDTHS:
        width_rows = [row for row in rows if row["width"] == width]
        tapes = {parse_state(row["state"])["r"] for row in width_rows}
        if len(width_rows) != 300 or len(tapes) != 100:
            raise ContractError("hidden serializer width/tape count mismatch")
        if any(tape == tape[::-1] for tape in tapes):
            raise ContractError("hidden serializer palindrome")
        for tape in tapes:
            metrics = serializer_pattern_metrics(tuple(int(digit) for digit in tape))
            if (
                metrics["constant_except_one"]
                or metrics["distinct_digits"] < min(width, 3)
                or metrics["distinct_adjacent_differences"] < 2
                or not metrics["non_affine"]
            ):
                raise ContractError("hidden serializer tape diversity mismatch")
        for position in range(width):
            if Counter(tape[position] for tape in tapes) != Counter(
                {str(digit): 10 for digit in range(10)}
            ):
                raise ContractError("hidden serializer digit marginal mismatch")
        if (
            sum(tape[0] == "0" for tape in tapes) != 10
            or sum(tape[-1] == "0" for tape in tapes) != 10
        ):
            raise ContractError("hidden serializer zero marginal mismatch")
        orbit_counts = Counter(_serializer_orbit_key(tape) for tape in tapes)
        if len(orbit_counts) != 5 or set(orbit_counts.values()) != {20}:
            raise ContractError("hidden serializer orbit collapse")
        patterns = list(orbit_counts)
        minimum_hamming = min(
            min(hamming(left, variant) for variant in translated_orbit(right))
            for index, left in enumerate(patterns)
            for right in patterns[index + 1 :]
        )
        if minimum_hamming < serializer_min_hamming(width):
            raise ContractError("hidden serializer orbit Hamming separation mismatch")
    return geometry


def validate_hidden_opening_rows(
    rows: list[dict], tokenizer: FrozenTokenizer, commitment: dict
) -> dict:
    seen_ids = set()
    seen_prompts = set()
    seen_signatures = set()
    transitions = []
    serializers = []
    for ordinal, row in enumerate(rows):
        if set(row) != HIDDEN_OPENING_KEYS:
            raise ContractError("hidden opening row key mismatch")
        if (
            row["schema"] != "shohin-ocsc-hidden-opening-row-v2"
            or row["board_id"] != commitment["board_id"]
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or not isinstance(row["row_id"], str)
            or not ID_RE.fullmatch(row["row_id"])
            or row["row_id"] in seen_ids
            or not isinstance(row["site_id"], str)
            or not ID_RE.fullmatch(row["site_id"])
            or type(row["width"]) is not int
            or row["width"] not in WIDTHS
            or type(row["position"]) is not int
            or type(row["incoming_carry"]) is not int
            or row["incoming_carry"] not in (0, 1)
            or row["operation"] not in {"add", "sub"}
        ):
            raise ContractError("hidden opening row identity/type mismatch")
        seen_ids.add(row["row_id"])
        for field in (
            "prompt_sha256",
            "normalized_prompt_sha256",
            "semantic_signature_sha256",
        ):
            if not isinstance(row[field], str) or not HEX64_RE.fullmatch(row[field]):
                raise ContractError("hidden opening hash field mismatch")
        state = independent_parse_state(row["state"])
        if state is None:
            raise ContractError("hidden opening state is invalid or noncanonical")
        if row["kind"] == "transition":
            _validate_hidden_transition_row(row, state, tokenizer)
            expected_prompt = transition_prompt(state)
            transitions.append(row)
        elif row["kind"] == "serializer":
            _validate_hidden_serializer_row(row, state, tokenizer)
            expected_prompt = serializer_prompt(state)
            serializers.append(row)
        else:
            raise ContractError("hidden opening kind mismatch")
        if (
            row["prompt"] != expected_prompt
            or row["prompt_sha256"] != sha256_bytes(expected_prompt.encode("ascii"))
            or row["normalized_prompt_sha256"]
            != normalized_prompt_sha256(expected_prompt)
            or row["semantic_signature_sha256"]
            != semantic_signature(state, row["kind"])
        ):
            raise ContractError("hidden opening prompt or semantic hash mismatch")
        if row["semantic_signature_sha256"] in seen_signatures:
            raise ContractError("duplicate hidden semantic signature")
        if row["normalized_prompt_sha256"] in seen_prompts:
            raise ContractError("duplicate hidden normalized prompt")
        seen_prompts.add(row["normalized_prompt_sha256"])
        seen_signatures.add(row["semantic_signature_sha256"])
    if len(transitions) != 2_100 or len(serializers) != 1_500:
        raise ContractError("hidden opening kind count mismatch")
    transition_geometry = _validate_initial_hidden_sites(transitions)
    transition_geometry.update(_validate_noninitial_hidden_sites(transitions))
    semantic_coverage = _validate_hidden_transition_semantic_coverage(transitions)
    serializer_geometry = _validate_hidden_serializer_sites(serializers)
    expected_transition = Counter(
        {
            tuple(key.split(":")): count
            for key, count in commitment["geometry"]["transition_slices"].items()
        }
    )
    actual_transition = Counter(
        {
            ("w{}".format(width), role, "c{}".format(carry), endpoint): count
            for (width, role, carry, endpoint), count in transition_geometry.items()
        }
    )
    expected_serializer = Counter(
        {
            (int(key.split(":")[0][1:]), key.split(":")[1]): count
            for key, count in commitment["geometry"]["serializer_slices"].items()
        }
    )
    if (
        actual_transition != expected_transition
        or serializer_geometry != expected_serializer
    ):
        raise ContractError("hidden opening geometry mismatch")
    return {
        "seen_prompts": seen_prompts,
        "seen_signatures": seen_signatures,
        "transition_sites": commitment["geometry"]["initial_invariance_sites"]
        + commitment["geometry"]["noninitial_paired_carry_sites"],
        "serializer_sites": len(WIDTHS) * 50,
        "semantic_coverage": semantic_coverage,
    }


def assert_hidden_disjoint_from_committed_sets(
    hidden_audit: dict, comparison: list[dict]
) -> None:
    blocked_prompts = {row["normalized_prompt_sha256"] for row in comparison}
    blocked_signatures = {row["semantic_signature_sha256"] for row in comparison}
    if (
        hidden_audit["seen_prompts"] & blocked_prompts
        or hidden_audit["seen_signatures"] & blocked_signatures
    ):
        raise ContractError(
            "hidden opening overlaps train, replay, development, or confirmation"
        )


def _verify_hidden_opening_snapshot(
    opening_path: Path,
    commitment_path: Path,
    bundle_snapshot: PinnedBundle,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    custodian_opening_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
) -> dict:
    custody_paths = {
        "hidden opening": Path(opening_path),
        "secret confirmation commitment": Path(commitment_path),
        "tokenizer": Path(tokenizer_path),
        "prompt registry": Path(prompt_registry_path),
        "custodian opening": Path(custodian_opening_path),
        "prepublication commitment": Path(publication_commitment_path),
        "independent review receipt": Path(independent_review_receipt_path),
    }
    custody_snapshots = {
        label: read_file_snapshot(
            path,
            label,
            exact_mode=0o444,
            custody_root=True,
        )
        for label, path in custody_paths.items()
    }
    bundle_verification, manifest = _verify_bundle_snapshot(
        bundle_snapshot,
        tokenizer_path,
        prompt_registry_path,
        commitment_path,
        publication_commitment_path,
        independent_review_receipt_path,
        verification_snapshots={
            "tokenizer": custody_snapshots["tokenizer"],
            "prompt_registry": custody_snapshots["prompt registry"],
            "secret_confirmation_commitment": custody_snapshots[
                "secret confirmation commitment"
            ],
        },
        publication_commitment_snapshot=custody_snapshots["prepublication commitment"],
        independent_review_snapshot=custody_snapshots["independent review receipt"],
    )
    commitment, commitment_sha256 = load_confirmation_commitment(
        commitment_path,
        custody_snapshots["secret confirmation commitment"],
    )
    _, custodian_opening_sha256 = load_custodian_opening(
        custodian_opening_path,
        commitment,
        custody_snapshots["custodian opening"],
    )
    rows, canonical_rows = read_canonical_jsonl(
        opening_path,
        "hidden opening",
        custody_snapshots["hidden opening"],
    )
    if len(rows) != commitment["leaf_count"]:
        raise ContractError("hidden opening leaf count mismatch")
    tokenizer = FrozenTokenizer(
        tokenizer_path,
        manifest["mode"],
        custody_snapshots["tokenizer"],
    )
    hidden_audit = validate_hidden_opening_rows(rows, tokenizer, commitment)
    root = hidden_merkle_root(canonical_rows)
    if root != commitment["merkle_root"]:
        raise ContractError("hidden Merkle root mismatch")

    if manifest["inputs"]["secret_confirmation_commitment_sha256"] != commitment_sha256:
        raise ContractError("hidden opening commitment is not the bundle commitment")
    train_rows = []
    for name in ("ocsc_train.jsonl", "iid_control_train.jsonl"):
        contract = manifest["files"][name]
        artifact_snapshot = bundle_snapshot.files[name]
        if (
            len(artifact_snapshot.payload) != contract["bytes"]
            or artifact_snapshot.sha256 != contract["sha256"]
        ):
            raise ContractError("bundle artifact hash mismatch during hidden opening")
        loaded, _ = read_canonical_jsonl(
            Path(artifact_snapshot.resolved_path),
            name,
            artifact_snapshot,
        )
        train_rows.extend(loaded)
    replay_rows, _ = read_canonical_jsonl(
        Path(bundle_snapshot.files["replay_prompts.jsonl"].resolved_path),
        "replay prompts",
        bundle_snapshot.files["replay_prompts.jsonl"],
    )
    registry, _ = load_prompt_registry(
        prompt_registry_path,
        custody_snapshots["prompt registry"],
    )
    replay_registry = {
        row["prompt_id"]: row for row in registry if row["use"] == "replay"
    }
    if len(replay_rows) != REPLAY_ROWS or set(replay_registry) != {
        row["replay_id"] for row in replay_rows
    }:
        raise ContractError("replay rows do not match the committed registry")
    for replay in replay_rows:
        source = replay_registry[replay["replay_id"]]
        if replay["registry_row_sha256"] != hash_json(source) or any(
            replay[field] != source[field]
            for field in (
                "family",
                "prompt",
                "prompt_sha256",
                "normalized_prompt_sha256",
                "semantic_signature_sha256",
                "source_commitment",
            )
        ):
            raise ContractError("replay row bytes are not registry-bound")
    comparison = (
        train_rows + replay_rows + [row for row in registry if row["use"] != "replay"]
    )
    assert_hidden_disjoint_from_committed_sets(hidden_audit, comparison)
    for label, path in custody_paths.items():
        final_snapshot = read_file_snapshot(
            path,
            "final " + label,
            exact_mode=0o444,
            custody_root=True,
        )
        if custody_snapshot_contract(final_snapshot) != custody_snapshot_contract(
            custody_snapshots[label]
        ):
            raise ContractError("hidden-opening custody input drifted: " + label)
    bundle_snapshot.assert_unchanged()
    return {
        "schema": "shohin-ocsc-hidden-opening-verification-v2",
        "verified": True,
        "board_id": commitment["board_id"],
        "rows": len(rows),
        "merkle_root": root,
        "commitment_sha256": commitment_sha256,
        "custodian_opening_sha256": custodian_opening_sha256,
        "prepublication_commitment_sha256": bundle_verification[
            "prepublication_commitment_sha256"
        ],
        "independent_review_receipt_sha256": bundle_verification[
            "independent_review_receipt_sha256"
        ],
        "full_bundle_verified": True,
        "transition_sites": hidden_audit["transition_sites"],
        "serializer_sites": hidden_audit["serializer_sites"],
        "train_replay_development_confirmation_overlap": 0,
    }


def verify_hidden_opening(
    opening_path: Path,
    commitment_path: Path,
    bundle: Path,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    custodian_opening_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
) -> dict:
    with PinnedBundle(bundle) as bundle_snapshot:
        result = _verify_hidden_opening_snapshot(
            opening_path,
            commitment_path,
            bundle_snapshot,
            tokenizer_path,
            prompt_registry_path,
            custodian_opening_path,
            publication_commitment_path,
            independent_review_receipt_path,
        )
        bundle_snapshot.assert_unchanged()
        return result


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short publication write")
        offset += written


def _entry_state_or_none(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ContractError("publication entry cannot be inspected: " + name) from error


def publication_staging_names(
    publication_receipt: dict,
    independent_review_receipt: dict,
) -> tuple[str, str]:
    request = publication_receipt.get("request")
    material = {
        "output_dir": request.get("output_dir") if isinstance(request, dict) else None,
        "output_parent": (
            request.get("output_parent") if isinstance(request, dict) else None
        ),
        "prepublication_commitment_sha256": publication_receipt.get("physical_sha256"),
        "prepublication_request_sha256": publication_receipt.get("request_sha256"),
        "prepublication_nonce_hex": publication_receipt.get("nonce_hex"),
        "independent_review_receipt_sha256": independent_review_receipt.get(
            "physical_sha256"
        ),
        "independent_review_request_sha256": independent_review_receipt.get(
            "review_request_sha256"
        ),
        "independent_review_nonce_hex": independent_review_receipt.get("nonce_hex"),
    }
    encoded = canonical_json_bytes(material)
    staging_id = hashlib.sha256(STAGING_IDENTITY_DOMAIN + encoded).hexdigest()
    stage_name = ".ocsc.partial." + staging_id
    return stage_name, stage_name + ".recovery.json"


def publication_lease_name(
    publication_receipt: dict,
    independent_review_receipt: dict,
) -> str:
    stage_name, _ = publication_staging_names(
        publication_receipt, independent_review_receipt
    )
    return stage_name + ".lease"


def _publication_host_identity_sha256() -> str:
    boot_id = None
    if sys.platform.startswith("linux"):
        try:
            boot_id = (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
            )
        except (OSError, UnicodeError):
            boot_id = None
    identity = {
        "boot_id": boot_id,
        "machine": platform.machine(),
        "node": platform.node(),
        "release": platform.release(),
        "sys_platform": sys.platform,
    }
    return sha256_bytes(PUBLICATION_LEASE_DOMAIN + canonical_json_bytes(identity))


def publication_lease_record(
    lease_name: str,
    metadata: os.stat_result,
    *,
    nonce_hex: str | None = None,
) -> dict:
    nonce_hex = nonce_hex or os.urandom(32).hex()
    if not HEX64_RE.fullmatch(nonce_hex):
        raise ContractError("publication lease nonce mismatch")
    record = {
        "schema": "shohin-ocsc-publication-lease-v1",
        "lease_name": lease_name,
        "lease_file_device": metadata.st_dev,
        "lease_file_inode": metadata.st_ino,
        "lease_file_owner_uid": metadata.st_uid,
        "publisher_pid": os.getpid(),
        "publisher_nonce_hex": nonce_hex,
        "host_kernel_identity_sha256": _publication_host_identity_sha256(),
        "liveness_primitive": "fcntl.flock-LOCK_EX-LOCK_NB-held-fd",
    }
    return with_payload_hash(record, "payload_sha256")


def _read_publication_lease_record(
    parent_fd: int,
    lease_name: str,
    descriptor: int,
) -> tuple[dict, os.stat_result]:
    before = os.fstat(descriptor)
    entry = os.stat(lease_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size > 1024 * 1024
        or _file_state(before) != _file_state(entry)
    ):
        raise ContractError("publication lease identity mismatch")
    os.lseek(descriptor, 0, os.SEEK_SET)
    blocks = []
    while True:
        block = os.read(descriptor, 64 * 1024)
        if not block:
            break
        blocks.append(block)
    payload = b"".join(blocks)
    after = os.fstat(descriptor)
    final_entry = os.stat(lease_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        len(payload) != before.st_size
        or _file_state(before) != _file_state(after)
        or _file_state(before) != _file_state(final_entry)
    ):
        raise ContractError("publication lease changed during read")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("publication lease must be ASCII") from error
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("publication lease must use one final LF")
    record = strict_json_loads(text, "publication lease")
    expected_keys = {
        "schema",
        "lease_name",
        "lease_file_device",
        "lease_file_inode",
        "lease_file_owner_uid",
        "publisher_pid",
        "publisher_nonce_hex",
        "host_kernel_identity_sha256",
        "liveness_primitive",
        "payload_sha256",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or payload != canonical_json_bytes(record, newline=True)
        or record["schema"] != "shohin-ocsc-publication-lease-v1"
        or record["lease_name"] != lease_name
        or type(record["lease_file_device"]) is not int
        or type(record["lease_file_inode"]) is not int
        or type(record["lease_file_owner_uid"]) is not int
        or type(record["publisher_pid"]) is not int
        or record["publisher_pid"] <= 0
        or not isinstance(record["publisher_nonce_hex"], str)
        or not HEX64_RE.fullmatch(record["publisher_nonce_hex"])
        or not isinstance(record["host_kernel_identity_sha256"], str)
        or not HEX64_RE.fullmatch(record["host_kernel_identity_sha256"])
        or record["liveness_primitive"] != "fcntl.flock-LOCK_EX-LOCK_NB-held-fd"
        or record["lease_file_device"] != before.st_dev
        or record["lease_file_inode"] != before.st_ino
        or record["lease_file_owner_uid"] != before.st_uid
    ):
        raise ContractError("publication lease contract mismatch")
    claimed = record["payload_sha256"]
    unhashed = dict(record)
    unhashed.pop("payload_sha256")
    if claimed != hash_json(unhashed):
        raise ContractError("publication lease payload hash mismatch")
    return record, before


def _write_publication_lease_record(
    parent_fd: int,
    lease_name: str,
    descriptor: int,
    record: dict,
) -> os.stat_result:
    payload = canonical_json_bytes(record, newline=True)
    before = os.fstat(descriptor)
    entry_before = os.stat(lease_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or not _same_inode(before, entry_before)
    ):
        raise ContractError("publication lease identity mismatch")
    os.fchmod(descriptor, 0o600)
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    _write_all(descriptor, payload)
    os.fsync(descriptor)
    metadata = os.fstat(descriptor)
    entry = os.stat(lease_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size != len(payload)
        or _file_state(metadata) != _file_state(entry)
    ):
        raise ContractError("publication lease identity mismatch")
    os.fsync(parent_fd)
    return metadata


def _acquire_publication_lease(
    parent_fd: int,
    lease_name: str,
    event_hook: Callable[[str, dict], None] | None = None,
) -> PublicationLease:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    created = False
    try:
        descriptor = os.open(
            lease_name,
            flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
    except FileExistsError:
        try:
            descriptor = os.open(lease_name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise ContractError("publication lease is not safely openable") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                if event_hook is not None:
                    observed_record, observed_metadata = _read_publication_lease_record(
                        parent_fd,
                        lease_name,
                        descriptor,
                    )
                    event_hook(
                        "live-lease-rejected",
                        {
                            "lease_name": lease_name,
                            "lease_record": observed_record,
                            "lease_file_device": observed_metadata.st_dev,
                            "lease_file_inode": observed_metadata.st_ino,
                        },
                    )
                raise ContractError(
                    "live concurrent publisher holds the publication lease"
                ) from error
            raise ContractError(
                "publication liveness cannot be established by kernel lock"
            ) from error
        metadata = os.fstat(descriptor)
        entry = os.stat(lease_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not _same_inode(metadata, entry)
        ):
            raise ContractError("publication lease identity mismatch")
        if created:
            record = publication_lease_record(lease_name, metadata)
            metadata = _write_publication_lease_record(
                parent_fd, lease_name, descriptor, record
            )
        else:
            record, metadata = _read_publication_lease_record(
                parent_fd, lease_name, descriptor
            )
        lease = PublicationLease(
            name=lease_name,
            descriptor=descriptor,
            metadata=metadata,
            record=record,
            created=created,
        )
        if event_hook is not None:
            event_hook(
                "lease-acquired",
                {
                    "lease_name": lease.name,
                    "lease_record": lease.record,
                    "lease_file_device": lease.metadata.st_dev,
                    "lease_file_inode": lease.metadata.st_ino,
                },
            )
        return lease
    except BaseException:
        os.close(descriptor)
        raise


def _release_publication_lease(lease: PublicationLease) -> None:
    try:
        fcntl.flock(lease.descriptor, fcntl.LOCK_UN)
    finally:
        os.close(lease.descriptor)


def qualification_publisher_identity(
    publisher_id: str,
    sequence: int,
    nonce_hex: str,
) -> dict:
    if (
        not isinstance(publisher_id, str)
        or not ID_RE.fullmatch(publisher_id)
        or type(sequence) is not int
        or sequence <= 0
        or not isinstance(nonce_hex, str)
        or not HEX64_RE.fullmatch(nonce_hex)
    ):
        raise ContractError("qualification publisher identity mismatch")
    return {
        "publisher_id": publisher_id,
        "sequence": sequence,
        "nonce_hex": nonce_hex,
    }


def qualification_publisher_receipt(
    identity: dict,
    event: str,
    details: dict,
    output_dir: Path,
    source_manifest_contract: dict,
) -> dict:
    identity = qualification_publisher_identity(
        identity.get("publisher_id") if isinstance(identity, dict) else None,
        identity.get("sequence") if isinstance(identity, dict) else None,
        identity.get("nonce_hex") if isinstance(identity, dict) else None,
    )
    if event not in {
        "lease-acquired",
        "live-lease-rejected",
        "publication-verified",
    }:
        raise ContractError("qualification publisher event mismatch")
    if not isinstance(details, dict) or set(details) != {
        "lease_name",
        "lease_record",
        "lease_file_device",
        "lease_file_inode",
    }:
        raise ContractError("qualification lease event detail mismatch")
    lease_record = details["lease_record"]
    if (
        not isinstance(lease_record, dict)
        or lease_record.get("lease_name") != details["lease_name"]
        or lease_record.get("lease_file_device") != details["lease_file_device"]
        or lease_record.get("lease_file_inode") != details["lease_file_inode"]
    ):
        raise ContractError("qualification lease receipt identity mismatch")
    source_manifest_contract = validate_source_manifest_contract(
        source_manifest_contract
    )
    source_identity = source_manifest_contract["bootstrap_source_identity"]
    receipt = {
        "schema": "shohin-ocsc-qualification-publisher-receipt-v1",
        **identity,
        "event": event,
        "output_dir": _lexical_absolute_path(
            Path(output_dir), "qualification output directory"
        ),
        "source_manifest_sha256": source_manifest_contract["payload_sha256"],
        "bootstrap_source_identity_sha256": source_identity["payload_sha256"],
        "host_fqdn": socket.getfqdn(),
        "host_kernel_identity_sha256": _publication_host_identity_sha256(),
        "publisher_pid": os.getpid(),
        "lease_name": details["lease_name"],
        "lease_record": lease_record,
        "lease_file_device": details["lease_file_device"],
        "lease_file_inode": details["lease_file_inode"],
        "claim_boundary": (
            "qualification_filesystem_event_only_no_publication_or_gpu_authority"
        ),
    }
    return with_payload_hash(receipt, "payload_sha256")


def validate_qualification_publisher_receipt(receipt: dict) -> dict:
    expected_keys = {
        "schema",
        "publisher_id",
        "sequence",
        "nonce_hex",
        "event",
        "output_dir",
        "source_manifest_sha256",
        "bootstrap_source_identity_sha256",
        "host_fqdn",
        "host_kernel_identity_sha256",
        "publisher_pid",
        "lease_name",
        "lease_record",
        "lease_file_device",
        "lease_file_inode",
        "claim_boundary",
        "payload_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("schema") != "shohin-ocsc-qualification-publisher-receipt-v1"
        or receipt.get("event")
        not in {"lease-acquired", "live-lease-rejected", "publication-verified"}
        or not isinstance(receipt.get("output_dir"), str)
        or not receipt["output_dir"].startswith("/")
        or any(
            not isinstance(receipt.get(field), str)
            or not HEX64_RE.fullmatch(receipt[field])
            for field in (
                "source_manifest_sha256",
                "bootstrap_source_identity_sha256",
                "host_kernel_identity_sha256",
                "payload_sha256",
            )
        )
        or not isinstance(receipt.get("host_fqdn"), str)
        or not receipt["host_fqdn"]
        or type(receipt.get("publisher_pid")) is not int
        or receipt["publisher_pid"] <= 0
        or type(receipt.get("lease_file_device")) is not int
        or type(receipt.get("lease_file_inode")) is not int
        or receipt.get("claim_boundary")
        != "qualification_filesystem_event_only_no_publication_or_gpu_authority"
        or receipt["payload_sha256"]
        != hash_json(
            {key: value for key, value in receipt.items() if key != "payload_sha256"}
        )
    ):
        raise ContractError("qualification publisher receipt mismatch")
    qualification_publisher_identity(
        receipt["publisher_id"], receipt["sequence"], receipt["nonce_hex"]
    )
    lease_record = receipt["lease_record"]
    if (
        not isinstance(lease_record, dict)
        or receipt["lease_name"] != lease_record.get("lease_name")
        or receipt["lease_file_device"] != lease_record.get("lease_file_device")
        or receipt["lease_file_inode"] != lease_record.get("lease_file_inode")
    ):
        raise ContractError("qualification publisher lease binding mismatch")
    return strict_json_loads(
        canonical_json_bytes(receipt).decode("ascii"),
        "qualification publisher receipt",
    )


def linux_lustre_qualification_signing_payload(unsigned_document: dict) -> bytes:
    return LINUX_LUSTRE_QUALIFICATION_SIGNATURE_DOMAIN + canonical_json_bytes(
        unsigned_document
    )


def validate_qualification_summary(
    _summary: dict,
    _retained_evidence_count: int,
) -> dict:
    raise ContractError(
        "caller-supplied qualification summaries have no authority; derive from events"
    )


def qualification_event_signing_payload(unsigned_document: dict) -> bytes:
    return QUALIFICATION_EVENT_SIGNATURE_DOMAIN + canonical_json_bytes(
        unsigned_document
    )


def qualification_broker_request_signing_payload(unsigned_document: dict) -> bytes:
    return QUALIFICATION_BROKER_REQUEST_SIGNATURE_DOMAIN + canonical_json_bytes(
        unsigned_document
    )


def qualification_broker_receipt_signing_payload(unsigned_document: dict) -> bytes:
    return QUALIFICATION_BROKER_RECEIPT_SIGNATURE_DOMAIN + canonical_json_bytes(
        unsigned_document
    )


def _reject_boolean_tree(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ContractError(label + " may not contain booleans")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError(label + " keys must be strings")
            _reject_boolean_tree(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_boolean_tree(child, label)


def _qualification_trusted_key(mode: str) -> str:
    if mode not in {"production", "test"}:
        raise ContractError("Linux/Lustre qualification mode mismatch")
    trusted_key = TRUSTED_LINUX_LUSTRE_QUALIFICATION_KEYS.get(mode)
    if trusted_key is None:
        raise ContractError(
            "production Linux/Lustre qualification trust root is not configured"
        )
    return trusted_key


def _verify_qualification_signature(
    document: dict,
    mode: str,
    domain_payload: Callable[[dict], bytes],
    label: str,
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    if (
        document.get("signature_algorithm") != "ed25519"
        or document.get("signer_public_key_hex") != trusted_key
        or not isinstance(document.get("signature_hex"), str)
        or not HEX128_RE.fullmatch(document["signature_hex"])
    ):
        raise ContractError(label + " signature contract mismatch")
    unsigned = dict(document)
    signature = unsigned.pop("signature_hex")
    InvalidSignature, Ed25519PublicKey = _trusted_cryptography_symbols()
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_key)).verify(
            bytes.fromhex(signature), domain_payload(unsigned)
        )
    except (InvalidSignature, ValueError) as error:
        raise ContractError(label + " signature failed") from error
    return unsigned


def _sign_qualification_document(
    unsigned: dict,
    private_key_hex: str,
    mode: str,
    domain_payload: Callable[[dict], bytes],
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    if not isinstance(private_key_hex, str) or not HEX64_RE.fullmatch(private_key_hex):
        raise ContractError("qualification private key bytes mismatch")
    _trusted_distribution_module("cryptography")
    try:
        ed25519 = importlib.import_module(
            "cryptography.hazmat.primitives.asymmetric.ed25519"
        )
        serialization = importlib.import_module(
            "cryptography.hazmat.primitives.serialization"
        )
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(private_key_hex)
        )
        public_key_hex = (
            private_key.public_key()
            .public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            .hex()
        )
    except (ImportError, ValueError) as error:
        raise ContractError("qualification signing runtime is unavailable") from error
    if public_key_hex != trusted_key:
        raise ContractError("qualification signing key is not trusted")
    document = dict(unsigned)
    document["signature_hex"] = private_key.sign(domain_payload(unsigned)).hex()
    return document


def _validate_qualification_event_details(event_type: str, details: dict) -> None:
    base_fields = {"evidence_sha256", "operation", "outcome"}
    extra_fields: set[str] = set()
    if event_type == "publication_path_complete":
        extra_fields = {
            "path_steps",
            "publication_receipt_sha256",
            "bundle_manifest_sha256",
            "output_device",
            "output_inode",
        }
    elif event_type in {
        "all_crash_evidence_permanently_retained",
        "permanent_evidence_inventory_recorded",
    }:
        extra_fields = {"retained_evidence"}
    if not isinstance(details, dict) or set(details) != base_fields | extra_fields:
        raise ContractError("qualification event detail mismatch")
    if (
        not isinstance(details.get("evidence_sha256"), str)
        or not HEX64_RE.fullmatch(details["evidence_sha256"])
        or details["evidence_sha256"]
        != hash_json(
            {key: value for key, value in details.items() if key != "evidence_sha256"}
        )
        or details.get("operation") != event_type
        or details.get("outcome") != "observed-from-raw-evidence"
    ):
        raise ContractError("qualification event evidence mismatch")
    if event_type == "publication_path_complete":
        if (
            details["path_steps"]
            != [
                "production-broker-transfer",
                "publish_bundle-stage-no-replace",
                "file-fsync-after-chmod",
                "stage-fsync",
                "rename-noreplace",
                "parent-fsync",
                "descriptor-inode-readback",
            ]
            or any(
                not isinstance(details[field], str)
                or not HEX64_RE.fullmatch(details[field])
                for field in (
                    "publication_receipt_sha256",
                    "bundle_manifest_sha256",
                )
            )
            or any(
                type(details[field]) is not int or details[field] <= 0
                for field in ("output_device", "output_inode")
            )
        ):
            raise ContractError("qualification publication-path evidence mismatch")
    if event_type in {
        "all_crash_evidence_permanently_retained",
        "permanent_evidence_inventory_recorded",
    }:
        inventory = details["retained_evidence"]
        evidence_fields = {
            "crash_point",
            "custody_path",
            "tree_device",
            "tree_inode",
            "tree_inventory_sha256",
            "journal_sha256",
            "lease_sha256",
            "stage_state",
            "canonical_state",
            "journal_state",
            "lease_state",
        }
        if (
            not isinstance(inventory, list)
            or len(inventory) != len(QUALIFICATION_CRASH_POINTS)
            or {record.get("crash_point") for record in inventory}
            != set(QUALIFICATION_CRASH_POINTS)
        ):
            raise ContractError("qualification retained-evidence inventory mismatch")
        for record in inventory:
            if (
                not isinstance(record, dict)
                or set(record) != evidence_fields
                or not isinstance(record["custody_path"], str)
                or not record["custody_path"].startswith("/")
                or any(
                    type(record[field]) is not int or record[field] <= 0
                    for field in ("tree_device", "tree_inode")
                )
                or any(
                    not isinstance(record[field], str)
                    or not HEX64_RE.fullmatch(record[field])
                    for field in ("tree_inventory_sha256", "lease_sha256")
                )
                or record["journal_state"] not in {"absent", "retained"}
                or (
                    record["journal_state"] == "retained"
                    and (
                        not isinstance(record["journal_sha256"], str)
                        or not HEX64_RE.fullmatch(record["journal_sha256"])
                    )
                )
                or (
                    record["journal_state"] == "absent"
                    and record["journal_sha256"] is not None
                )
                or record["stage_state"] not in {"retained", "renamed-canonical"}
                or record["canonical_state"] not in {"absent", "retained"}
                or record["lease_state"] != "retained"
            ):
                raise ContractError("qualification retained-evidence record mismatch")


def qualification_event_unsigned(
    *,
    qualification_id: str,
    host_id: str,
    host_fqdn: str,
    host_kernel_identity_sha256: str,
    sequence: int,
    previous_event_sha256: str,
    event_type: str,
    nonce_hex: str,
    source_manifest_sha256: str,
    lustre_mount_source: str,
    lustre_mountpoint: str,
    output_parent_device: int,
    output_parent_inode: int,
    details: dict,
    mode: str,
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    if (
        not isinstance(qualification_id, str)
        or not ID_RE.fullmatch(qualification_id)
        or not isinstance(host_id, str)
        or not ID_RE.fullmatch(host_id)
        or not isinstance(host_fqdn, str)
        or not host_fqdn
        or type(sequence) is not int
        or sequence <= 0
        or event_type not in LINUX_LUSTRE_QUALIFICATION_CHECKS
        or not isinstance(lustre_mount_source, str)
        or not lustre_mount_source
        or not isinstance(lustre_mountpoint, str)
        or not lustre_mountpoint.startswith("/")
        or any(
            not isinstance(value, str) or not HEX64_RE.fullmatch(value)
            for value in (
                host_kernel_identity_sha256,
                previous_event_sha256,
                nonce_hex,
                source_manifest_sha256,
            )
        )
        or type(output_parent_device) is not int
        or output_parent_device <= 0
        or type(output_parent_inode) is not int
        or output_parent_inode <= 0
    ):
        raise ContractError("qualification event identity mismatch")
    _reject_boolean_tree(details, "qualification event details")
    _validate_qualification_event_details(event_type, details)
    identity = {
        "qualification_id": qualification_id,
        "host_id": host_id,
        "host_fqdn": host_fqdn,
        "host_kernel_identity_sha256": host_kernel_identity_sha256,
        "sequence": sequence,
        "previous_event_sha256": previous_event_sha256,
        "event_type": event_type,
        "nonce_hex": nonce_hex,
        "source_manifest_sha256": source_manifest_sha256,
        "lustre_mount_source": lustre_mount_source,
        "lustre_mountpoint": lustre_mountpoint,
        "output_parent_device": output_parent_device,
        "output_parent_inode": output_parent_inode,
        "details": details,
    }
    return {
        "schema": "shohin-ocsc-qualification-event-v1",
        "event_id": hash_json(identity),
        **identity,
        "claim_boundary": "signed-raw-filesystem-event-no-independent-authority",
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": trusted_key,
    }


def sign_qualification_event(unsigned: dict, private_key_hex: str, mode: str) -> dict:
    return _sign_qualification_document(
        unsigned,
        private_key_hex,
        mode,
        qualification_event_signing_payload,
    )


def validate_qualification_event(
    event: dict,
    expected_source_manifest_sha256: str,
    mode: str,
) -> dict:
    expected_keys = {
        "schema",
        "event_id",
        "qualification_id",
        "host_id",
        "host_fqdn",
        "host_kernel_identity_sha256",
        "sequence",
        "previous_event_sha256",
        "event_type",
        "nonce_hex",
        "source_manifest_sha256",
        "lustre_mount_source",
        "lustre_mountpoint",
        "output_parent_device",
        "output_parent_inode",
        "details",
        "claim_boundary",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    if not isinstance(event, dict) or set(event) != expected_keys:
        raise ContractError("qualification event contract mismatch")
    _reject_boolean_tree(event, "qualification event")
    unsigned = _verify_qualification_signature(
        event,
        mode,
        qualification_event_signing_payload,
        "qualification event",
    )
    rebuilt = qualification_event_unsigned(
        qualification_id=event["qualification_id"],
        host_id=event["host_id"],
        host_fqdn=event["host_fqdn"],
        host_kernel_identity_sha256=event["host_kernel_identity_sha256"],
        sequence=event["sequence"],
        previous_event_sha256=event["previous_event_sha256"],
        event_type=event["event_type"],
        nonce_hex=event["nonce_hex"],
        source_manifest_sha256=event["source_manifest_sha256"],
        lustre_mount_source=event["lustre_mount_source"],
        lustre_mountpoint=event["lustre_mountpoint"],
        output_parent_device=event["output_parent_device"],
        output_parent_inode=event["output_parent_inode"],
        details=event["details"],
        mode=mode,
    )
    if (
        unsigned != rebuilt
        or event["source_manifest_sha256"] != expected_source_manifest_sha256
    ):
        raise ContractError("qualification event source or identity mismatch")
    return strict_json_loads(
        canonical_json_bytes(event).decode("ascii"), "qualification event"
    )


def qualification_broker_request_unsigned(
    event: dict,
    *,
    broker_id: str,
    sequence: int,
    previous_request_sha256: str,
    destination_path: str,
    mode: str,
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    if (
        not isinstance(broker_id, str)
        or not ID_RE.fullmatch(broker_id)
        or type(sequence) is not int
        or sequence <= 0
        or not isinstance(previous_request_sha256, str)
        or not HEX64_RE.fullmatch(previous_request_sha256)
        or not isinstance(destination_path, str)
        or not destination_path.startswith("/")
    ):
        raise ContractError("qualification broker request identity mismatch")
    source_event_sha256 = hash_json(event)
    identity = {
        "qualification_id": event.get("qualification_id"),
        "broker_id": broker_id,
        "sequence": sequence,
        "previous_request_sha256": previous_request_sha256,
        "source_event_id": event.get("event_id"),
        "source_event_sha256": source_event_sha256,
        "source_manifest_sha256": event.get("source_manifest_sha256"),
        "destination_path": destination_path,
        "operation": "production-broker-descriptor-copy-no-replace",
    }
    return {
        "schema": "shohin-ocsc-qualification-broker-request-v1",
        "request_id": hash_json(identity),
        **identity,
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": trusted_key,
    }


def _validate_broker_request(
    request: dict,
    events: dict[str, dict],
    expected_source_manifest_sha256: str,
    mode: str,
) -> dict:
    expected_keys = {
        "schema",
        "request_id",
        "qualification_id",
        "broker_id",
        "sequence",
        "previous_request_sha256",
        "source_event_id",
        "source_event_sha256",
        "source_manifest_sha256",
        "destination_path",
        "operation",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    if not isinstance(request, dict) or set(request) != expected_keys:
        raise ContractError("qualification broker request mismatch")
    _reject_boolean_tree(request, "qualification broker request")
    unsigned = _verify_qualification_signature(
        request,
        mode,
        qualification_broker_request_signing_payload,
        "qualification broker request",
    )
    event = events.get(request["source_event_id"])
    if event is None:
        raise ContractError("qualification broker request event is missing")
    rebuilt = qualification_broker_request_unsigned(
        event,
        broker_id=request["broker_id"],
        sequence=request["sequence"],
        previous_request_sha256=request["previous_request_sha256"],
        destination_path=request["destination_path"],
        mode=mode,
    )
    if (
        unsigned != rebuilt
        or request["source_manifest_sha256"] != expected_source_manifest_sha256
    ):
        raise ContractError("qualification broker request binding mismatch")
    return request


def qualification_broker_receipt_unsigned(
    request: dict,
    destination: FileSnapshot,
    *,
    broker_sequence: int,
    previous_receipt_sha256: str,
    mode: str,
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    if (
        type(broker_sequence) is not int
        or broker_sequence <= 0
        or not isinstance(previous_receipt_sha256, str)
        or not HEX64_RE.fullmatch(previous_receipt_sha256)
    ):
        raise ContractError("qualification broker receipt sequence mismatch")
    transfer = {
        "destination_path": destination.resolved_path,
        "physical_bytes": len(destination.payload),
        "physical_sha256": destination.sha256,
        "file_device": destination.metadata.st_dev,
        "file_inode": destination.metadata.st_ino,
        "file_mode": stat.S_IMODE(destination.metadata.st_mode),
        "hard_links": destination.metadata.st_nlink,
        "no_replace_primitive": "openat-O_CREAT-O_EXCL-O_NOFOLLOW",
        "file_fsync_after_chmod": "observed",
        "parent_fsync": "observed",
        "descriptor_inode_readback": "observed",
        "retention_policy": "permanent-no-delete-no-rewrite",
    }
    identity = {
        "qualification_id": request.get("qualification_id"),
        "broker_id": request.get("broker_id"),
        "sequence": broker_sequence,
        "previous_receipt_sha256": previous_receipt_sha256,
        "request_id": request.get("request_id"),
        "request_sha256": hash_json(request),
        "source_event_id": request.get("source_event_id"),
        "source_event_sha256": request.get("source_event_sha256"),
        "source_manifest_sha256": request.get("source_manifest_sha256"),
        "transfer": transfer,
    }
    return {
        "schema": "shohin-ocsc-qualification-broker-receipt-v1",
        "receipt_id": hash_json(identity),
        **identity,
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": trusted_key,
    }


def _validate_broker_receipt(
    receipt: dict,
    requests: dict[str, dict],
    expected_source_manifest_sha256: str,
    mode: str,
) -> dict:
    expected_keys = {
        "schema",
        "receipt_id",
        "qualification_id",
        "broker_id",
        "sequence",
        "previous_receipt_sha256",
        "request_id",
        "request_sha256",
        "source_event_id",
        "source_event_sha256",
        "source_manifest_sha256",
        "transfer",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    transfer_fields = {
        "destination_path",
        "physical_bytes",
        "physical_sha256",
        "file_device",
        "file_inode",
        "file_mode",
        "hard_links",
        "no_replace_primitive",
        "file_fsync_after_chmod",
        "parent_fsync",
        "descriptor_inode_readback",
        "retention_policy",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or not isinstance(receipt.get("transfer"), dict)
        or set(receipt["transfer"]) != transfer_fields
    ):
        raise ContractError("qualification broker receipt mismatch")
    _reject_boolean_tree(receipt, "qualification broker receipt")
    _verify_qualification_signature(
        receipt,
        mode,
        qualification_broker_receipt_signing_payload,
        "qualification broker receipt",
    )
    request = requests.get(receipt["request_id"])
    transfer = receipt["transfer"]
    if (
        request is None
        or receipt["qualification_id"] != request["qualification_id"]
        or receipt["broker_id"] != request["broker_id"]
        or receipt["request_sha256"] != hash_json(request)
        or receipt["source_event_id"] != request["source_event_id"]
        or receipt["source_event_sha256"] != request["source_event_sha256"]
        or receipt["source_manifest_sha256"] != expected_source_manifest_sha256
        or transfer["destination_path"] != request["destination_path"]
        or type(transfer["physical_bytes"]) is not int
        or transfer["physical_bytes"] <= 0
        or not isinstance(transfer["physical_sha256"], str)
        or not HEX64_RE.fullmatch(transfer["physical_sha256"])
        or any(
            type(transfer[field]) is not int or transfer[field] <= 0
            for field in ("file_device", "file_inode", "hard_links")
        )
        or transfer["file_mode"] != 0o444
        or transfer["hard_links"] != 1
        or transfer["no_replace_primitive"] != "openat-O_CREAT-O_EXCL-O_NOFOLLOW"
        or transfer["file_fsync_after_chmod"] != "observed"
        or transfer["parent_fsync"] != "observed"
        or transfer["descriptor_inode_readback"] != "observed"
        or transfer["retention_policy"] != "permanent-no-delete-no-rewrite"
    ):
        raise ContractError("qualification broker receipt binding mismatch")
    identity = {
        key: receipt[key]
        for key in (
            "qualification_id",
            "broker_id",
            "sequence",
            "previous_receipt_sha256",
            "request_id",
            "request_sha256",
            "source_event_id",
            "source_event_sha256",
            "source_manifest_sha256",
            "transfer",
        )
    }
    if receipt["receipt_id"] != hash_json(identity):
        raise ContractError("qualification broker receipt identity mismatch")
    return receipt


def derive_qualification_report(
    raw_events: list[dict],
    broker_requests: list[dict],
    broker_receipts: list[dict],
    expected_source_manifest_sha256: str,
    mode: str,
) -> dict:
    if (
        not isinstance(raw_events, list)
        or not isinstance(broker_requests, list)
        or not isinstance(broker_receipts, list)
        or not isinstance(expected_source_manifest_sha256, str)
        or not HEX64_RE.fullmatch(expected_source_manifest_sha256)
    ):
        raise ContractError("qualification raw evidence inventory mismatch")
    events = [
        validate_qualification_event(event, expected_source_manifest_sha256, mode)
        for event in raw_events
    ]
    by_event_id = {event["event_id"]: event for event in events}
    if (
        len(by_event_id) != len(events)
        or len(events) != len(LINUX_LUSTRE_QUALIFICATION_CHECKS)
        or {event["event_type"] for event in events}
        != set(LINUX_LUSTRE_QUALIFICATION_CHECKS)
    ):
        raise ContractError("qualification event/check inventory mismatch")
    qualification_ids = {event["qualification_id"] for event in events}
    mount_identities = {
        (
            event["lustre_mount_source"],
            event["lustre_mountpoint"],
            event["output_parent_device"],
            event["output_parent_inode"],
        )
        for event in events
    }
    host_identities = {
        (
            event["host_id"],
            event["host_fqdn"],
            event["host_kernel_identity_sha256"],
        )
        for event in events
    }
    if (
        len(qualification_ids) != 1
        or len(mount_identities) != 1
        or len(host_identities) != 2
        or len({host[0] for host in host_identities}) != 2
        or len({host[1] for host in host_identities}) != 2
        or len({host[2] for host in host_identities}) != 2
    ):
        raise ContractError("qualification two-host Lustre identity mismatch")
    for host_id in {event["host_id"] for event in events}:
        chain = sorted(
            (event for event in events if event["host_id"] == host_id),
            key=lambda event: event["sequence"],
        )
        if [event["sequence"] for event in chain] != list(range(1, len(chain) + 1)):
            raise ContractError("qualification host event sequence mismatch")
        previous = "0" * 64
        for event in chain:
            if event["previous_event_sha256"] != previous:
                raise ContractError("qualification host event chain mismatch")
            previous = hash_json(event)
    requests = [
        _validate_broker_request(
            request,
            by_event_id,
            expected_source_manifest_sha256,
            mode,
        )
        for request in broker_requests
    ]
    by_request_id = {request["request_id"]: request for request in requests}
    if len(by_request_id) != len(requests) or len(requests) != 1:
        raise ContractError("qualification broker request inventory mismatch")
    receipts = [
        _validate_broker_receipt(
            receipt,
            by_request_id,
            expected_source_manifest_sha256,
            mode,
        )
        for receipt in broker_receipts
    ]
    if len(receipts) != 1:
        raise ContractError("qualification broker receipt inventory mismatch")
    request = requests[0]
    receipt = receipts[0]
    broker_event = next(
        event
        for event in events
        if event["event_type"] == "production_broker_transfer_complete"
    )
    if (
        request["source_event_id"] != broker_event["event_id"]
        or receipt["request_id"] != request["request_id"]
        or receipt["transfer"]["physical_sha256"]
        != sha256_bytes(canonical_json_bytes(broker_event, newline=True))
        or request["sequence"] != 1
        or request["previous_request_sha256"] != "0" * 64
        or receipt["sequence"] != 1
        or receipt["previous_receipt_sha256"] != "0" * 64
    ):
        raise ContractError("qualification production broker path mismatch")
    check_evidence = {event["event_type"]: [event["event_id"]] for event in events}
    check_evidence["production_broker_transfer_complete"].extend(
        [request["request_id"], receipt["receipt_id"]]
    )
    retained_event = next(
        event
        for event in events
        if event["event_type"] == "all_crash_evidence_permanently_retained"
    )
    summary = {
        "raw_event_count": len(events),
        "broker_request_count": len(requests),
        "broker_receipt_count": len(receipts),
        "derived_check_count": len(check_evidence),
        "retained_evidence_count": len(retained_event["details"]["retained_evidence"]),
        "distinct_host_count": len(host_identities),
    }
    report = {
        "schema": "shohin-ocsc-qualification-derived-report-v1",
        "qualification_id": next(iter(qualification_ids)),
        "source_manifest_sha256": expected_source_manifest_sha256,
        "check_evidence": {
            name: check_evidence[name] for name in LINUX_LUSTRE_QUALIFICATION_CHECKS
        },
        "summary": summary,
        "raw_event_chain_root_sha256": hash_json(
            sorted(hash_json(event) for event in events)
        ),
        "broker_chain_root_sha256": hash_json([hash_json(request), hash_json(receipt)]),
        "claim_boundary": "event-derived-filesystem-qualification-source-only",
    }
    _reject_boolean_tree(report, "qualification derived report")
    return with_payload_hash(report, "payload_sha256")


def qualification_marker(report: dict) -> dict:
    expected_summary = {
        "raw_event_count": len(LINUX_LUSTRE_QUALIFICATION_CHECKS),
        "broker_request_count": 1,
        "broker_receipt_count": 1,
        "derived_check_count": len(LINUX_LUSTRE_QUALIFICATION_CHECKS),
        "retained_evidence_count": len(QUALIFICATION_CRASH_POINTS),
        "distinct_host_count": 2,
    }
    if (
        not isinstance(report, dict)
        or report.get("schema") != "shohin-ocsc-qualification-derived-report-v1"
        or report.get("payload_sha256")
        != hash_json(
            {key: value for key, value in report.items() if key != "payload_sha256"}
        )
        or report.get("summary") != expected_summary
        or not isinstance(report.get("check_evidence"), dict)
        or set(report["check_evidence"]) != set(LINUX_LUSTRE_QUALIFICATION_CHECKS)
        or any(
            not isinstance(report["check_evidence"][name], list)
            or not report["check_evidence"][name]
            or any(
                not isinstance(evidence_id, str) or not HEX64_RE.fullmatch(evidence_id)
                for evidence_id in report["check_evidence"][name]
            )
            for name in LINUX_LUSTRE_QUALIFICATION_CHECKS
        )
    ):
        raise ContractError("qualification report cannot derive marker")
    summary = report["summary"]
    marker = {
        "schema": "shohin-ocsc-qualification-marker-v1",
        "qualification_id": report["qualification_id"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "report_sha256": report["payload_sha256"],
        "raw_event_chain_root_sha256": report["raw_event_chain_root_sha256"],
        "broker_chain_root_sha256": report["broker_chain_root_sha256"],
        "derived_check_count": summary["derived_check_count"],
        "retained_evidence_count": summary["retained_evidence_count"],
        "authority": "qualification-source-review-only",
    }
    _reject_boolean_tree(marker, "qualification marker")
    return with_payload_hash(marker, "payload_sha256")


def load_linux_lustre_qualification_receipt(
    path: Path,
    expected_source_manifest: dict,
    mode: str,
    *,
    snapshot: FileSnapshot | None = None,
) -> dict:
    trusted_key = _qualification_trusted_key(mode)
    expected_source_manifest = validate_source_manifest_contract(
        expected_source_manifest
    )
    snapshot = snapshot or read_file_snapshot(
        Path(path),
        "Linux/Lustre qualification receipt",
        exact_mode=0o444,
        custody_root=True,
    )
    document = strict_json_loads(
        snapshot.payload.decode("ascii"), "Linux/Lustre qualification receipt"
    )
    expected_keys = {
        "schema",
        "qualification_id",
        "reviewer_id",
        "sequence",
        "nonce_hex",
        "command",
        "command_sha256",
        "source_manifest",
        "source_manifest_sha256",
        "raw_events",
        "broker_requests",
        "broker_receipts",
        "derived_report",
        "marker",
        "claim_boundary",
        "signature_algorithm",
        "signer_public_key_hex",
        "signature_hex",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or snapshot.payload != canonical_json_bytes(document, newline=True)
        or document.get("schema") != "shohin-ocsc-linux-lustre-qualification-receipt-v3"
        or not isinstance(document.get("qualification_id"), str)
        or not ID_RE.fullmatch(document["qualification_id"])
        or not isinstance(document.get("reviewer_id"), str)
        or not ID_RE.fullmatch(document["reviewer_id"])
        or type(document.get("sequence")) is not int
        or document["sequence"] <= 0
        or not isinstance(document.get("nonce_hex"), str)
        or not HEX64_RE.fullmatch(document["nonce_hex"])
        or not isinstance(document.get("command"), list)
        or any(not isinstance(value, str) for value in document["command"])
        or document.get("command_sha256") != hash_json(document["command"])
        or document.get("source_manifest_sha256")
        != expected_source_manifest["payload_sha256"]
        or not recursively_type_strict_equal(
            document.get("source_manifest"), expected_source_manifest
        )
        or document.get("claim_boundary")
        != "linux_lustre_qualification_source_evidence_only_no_bundle_consumer_fit_eval_gpu_or_scientific_authority"
        or document.get("signature_algorithm") != "ed25519"
        or document.get("signer_public_key_hex") != trusted_key
    ):
        raise ContractError("Linux/Lustre qualification receipt mismatch")
    _verify_qualification_signature(
        document,
        mode,
        linux_lustre_qualification_signing_payload,
        "Linux/Lustre qualification receipt",
    )
    report = derive_qualification_report(
        document["raw_events"],
        document["broker_requests"],
        document["broker_receipts"],
        expected_source_manifest["payload_sha256"],
        mode,
    )
    marker = qualification_marker(report)
    if (
        not recursively_type_strict_equal(document["derived_report"], report)
        or not recursively_type_strict_equal(document["marker"], marker)
        or document["qualification_id"] != report["qualification_id"]
    ):
        raise ContractError(
            "Linux/Lustre qualification report or marker was not event-derived"
        )
    return {
        "schema": "shohin-ocsc-linux-lustre-qualification-verification-v2",
        "resolved_path": snapshot.resolved_path,
        "physical_bytes": len(snapshot.payload),
        "physical_sha256": snapshot.sha256,
        "signer_public_key_hex": trusted_key,
        "qualification_id": document["qualification_id"],
        "source_manifest_sha256": expected_source_manifest["payload_sha256"],
        "signature_verified": True,
        "cross_node_verified": True,
        "event_derivation_verified": True,
        "qualification_authority": False,
        "claim_boundary": document["claim_boundary"],
    }


def _write_qualification_control_document(path: Path, document: dict) -> FileSnapshot:
    path = Path(path)
    if path.name in {"", ".", ".."} or path.parent == path:
        raise ContractError("qualification control path mismatch")
    parent_fd, parent_resolved, parent_metadata = _open_pinned_directory(
        path.parent,
        "qualification control parent",
        reject_other_writes=True,
    )
    try:
        if stat.S_IMODE(parent_metadata.st_mode) != 0o700:
            raise ContractError("qualification control parent must be mode 0700")
        snapshot = _write_qualification_control_document_at(
            parent_fd,
            parent_resolved,
            path.name,
            document,
        )
        _assert_directory_path_matches_fd(
            path.parent,
            parent_resolved,
            parent_fd,
            "qualification control parent",
            exact_mode=0o700,
        )
        return snapshot
    finally:
        os.close(parent_fd)


def _write_qualification_control_document_at(
    parent_fd: int,
    parent_resolved: str,
    name: str,
    document: dict,
) -> FileSnapshot:
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or name != Path(name).name
        or not isinstance(parent_resolved, str)
        or not parent_resolved.startswith("/")
    ):
        raise ContractError("qualification control entry mismatch")
    payload = canonical_json_bytes(document, newline=True)
    descriptor = None
    try:
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        readback = os.pread(descriptor, len(payload) + 1, 0)
        after = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            readback != payload
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size != len(payload)
            or _file_state(before) != _file_state(after)
            or _file_state(before) != _file_state(entry)
        ):
            raise ContractError("qualification control descriptor readback mismatch")
        os.fsync(parent_fd)
        parent_metadata = os.fstat(parent_fd)
        return FileSnapshot(
            payload=readback,
            resolved_path=str(Path(parent_resolved) / name),
            metadata=before,
            parent_resolved_path=parent_resolved,
            parent_metadata=parent_metadata,
        )
    except FileExistsError as error:
        raise ContractError("qualification control receipt already exists") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_qualification_custody_document(
    evidence_dir: Path,
    custody_name: str,
    file_name: str,
    document: dict,
) -> FileSnapshot:
    if any(
        not isinstance(name, str) or name in {"", ".", ".."} or name != Path(name).name
        for name in (custody_name, file_name)
    ):
        raise ContractError("qualification custody entry mismatch")
    evidence_fd, evidence_resolved, _ = _open_pinned_directory(
        Path(evidence_dir),
        "qualification evidence package",
        exact_mode=0o700,
    )
    custody_fd = None
    try:
        try:
            os.mkdir(custody_name, mode=0o700, dir_fd=evidence_fd)
        except FileExistsError as error:
            raise ContractError(
                "qualification custody root already exists and was retained"
            ) from error
        custody_fd, custody_metadata = _open_bundle_directory_at(
            evidence_fd,
            custody_name,
            "qualification custody root",
        )
        if stat.S_IMODE(custody_metadata.st_mode) != 0o700:
            raise ContractError("qualification custody root mode mismatch")
        _write_qualification_control_document_at(
            custody_fd,
            str(Path(evidence_resolved) / custody_name),
            file_name,
            document,
        )
        os.fchmod(custody_fd, 0o555)
        os.fsync(custody_fd)
        os.fsync(evidence_fd)
        _assert_directory_path_matches_fd(
            Path(evidence_dir),
            evidence_resolved,
            evidence_fd,
            "qualification evidence package",
            exact_mode=0o700,
        )
        _assert_directory_path_matches_fd(
            Path(evidence_resolved) / custody_name,
            str(Path(evidence_resolved) / custody_name),
            custody_fd,
            "qualification custody root",
            exact_mode=0o555,
        )
    finally:
        if custody_fd is not None:
            os.close(custody_fd)
        os.close(evidence_fd)
    return read_file_snapshot(
        Path(evidence_resolved) / custody_name / file_name,
        "qualification custody document",
        exact_mode=0o444,
        custody_root=True,
    )


def _retained_entry_contract_at(
    parent_fd: int,
    name: str,
    label: str,
    allowed_modes: set[int],
) -> dict:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise ContractError(label + " is not safely readable") from error
    try:
        before = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) not in allowed_modes
            or _file_state(before) != _file_state(entry)
        ):
            raise ContractError(label + " retained identity mismatch")
        payload = os.pread(descriptor, before.st_size, 0)
        after = os.fstat(descriptor)
        final_entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            len(payload) != before.st_size
            or _file_state(before) != _file_state(after)
            or _file_state(before) != _file_state(final_entry)
        ):
            raise ContractError(label + " changed during retained readback")
        return {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "mode": stat.S_IMODE(before.st_mode),
            "device": before.st_dev,
            "inode": before.st_ino,
        }
    finally:
        os.close(descriptor)


def capture_retained_publication_evidence(
    output_parent: Path,
    *,
    crash_point: str,
    stage_name: str,
    canonical_name: str,
    journal_name: str,
    lease_name: str,
) -> dict:
    """Read one crash state by descriptor without mutating retained evidence."""

    if crash_point not in QUALIFICATION_CRASH_POINTS or any(
        not isinstance(name, str) or name in {"", ".", ".."} or name != Path(name).name
        for name in (stage_name, canonical_name, journal_name, lease_name)
    ):
        raise ContractError("retained publication evidence identity mismatch")
    parent_fd, parent_path, _ = _open_pinned_directory(
        Path(output_parent),
        "retained publication evidence parent",
        reject_other_writes=True,
    )
    tree_fd = None
    try:
        stage = _entry_state_or_none(parent_fd, stage_name)
        canonical = _entry_state_or_none(parent_fd, canonical_name)
        expect_canonical = crash_point == "canonical-before-parent-fsync"
        if (stage is not None) == (canonical is not None) or expect_canonical != (
            canonical is not None
        ):
            raise ContractError("retained publication tree state mismatch")
        tree_name = canonical_name if expect_canonical else stage_name
        tree_fd, tree = _open_bundle_directory_at(
            parent_fd, tree_name, "retained publication tree"
        )
        children = {}
        for child in sorted(os.listdir(tree_fd)):
            children[child] = _retained_entry_contract_at(
                tree_fd,
                child,
                "retained publication child " + child,
                {0o600, 0o444},
            )
        tree_after = os.fstat(tree_fd)
        tree_entry = os.stat(tree_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _file_state(tree) != _file_state(tree_after)
            or _file_state(tree) != _file_state(tree_entry)
            or sorted(os.listdir(tree_fd)) != sorted(children)
        ):
            raise ContractError("retained publication tree changed during readback")
        journal_state = _entry_state_or_none(parent_fd, journal_name)
        if journal_state is None:
            if crash_point != "stage-created-before-journal":
                raise ContractError("retained publication journal is missing")
            journal_sha256 = None
            journal_status = "absent"
        else:
            journal_sha256 = _retained_entry_contract_at(
                parent_fd,
                journal_name,
                "retained publication journal",
                {0o444},
            )["sha256"]
            journal_status = "retained"
        lease_sha256 = _retained_entry_contract_at(
            parent_fd,
            lease_name,
            "retained publication lease",
            {0o600},
        )["sha256"]
        _assert_directory_path_matches_fd(
            Path(output_parent),
            parent_path,
            parent_fd,
            "retained publication evidence parent",
        )
        return {
            "crash_point": crash_point,
            "custody_path": str(Path(parent_path) / tree_name),
            "tree_device": tree.st_dev,
            "tree_inode": tree.st_ino,
            "tree_inventory_sha256": hash_json(children),
            "journal_sha256": journal_sha256,
            "lease_sha256": lease_sha256,
            "stage_state": "renamed-canonical" if expect_canonical else "retained",
            "canonical_state": "retained" if expect_canonical else "absent",
            "journal_state": journal_status,
            "lease_state": "retained",
        }
    finally:
        if tree_fd is not None:
            os.close(tree_fd)
        os.close(parent_fd)


def _trigger_qualification_crash(
    selected_crash_point: str | None,
    reached_crash_point: str,
) -> None:
    if reached_crash_point not in QUALIFICATION_CRASH_POINTS or (
        selected_crash_point is not None
        and selected_crash_point not in QUALIFICATION_CRASH_POINTS
    ):
        raise ContractError("qualification crash-point identity mismatch")
    if selected_crash_point == reached_crash_point:
        os.kill(os.getpid(), signal.SIGKILL)
        raise ContractError("qualification hard-crash signal unexpectedly returned")


def execute_qualification_broker_transfer(
    source_event_path: Path,
    broker_record_dir: Path,
    publication_event_dir: Path,
    *,
    broker_id: str,
    sequence: int,
    previous_request_sha256: str,
    previous_receipt_sha256: str,
    expected_source_manifest_sha256: str,
    private_key_hex: str,
    mode: str,
) -> dict:
    """Persist one signed event through the production no-replace broker path."""

    source = read_file_snapshot(
        Path(source_event_path),
        "qualification broker source event",
        exact_mode=0o444,
    )
    event = strict_json_loads(
        source.payload.decode("ascii"), "qualification broker source event"
    )
    if source.payload != canonical_json_bytes(event, newline=True):
        raise ContractError("qualification broker source event is not canonical")
    event = validate_qualification_event(event, expected_source_manifest_sha256, mode)
    destination_path = Path(publication_event_dir) / (event["event_id"] + ".event.json")
    request_unsigned = qualification_broker_request_unsigned(
        event,
        broker_id=broker_id,
        sequence=sequence,
        previous_request_sha256=previous_request_sha256,
        destination_path=_lexical_absolute_path(
            destination_path, "qualification broker destination"
        ),
        mode=mode,
    )
    request = _sign_qualification_document(
        request_unsigned,
        private_key_hex,
        mode,
        qualification_broker_request_signing_payload,
    )
    request_path = Path(broker_record_dir) / (
        request["request_id"] + ".broker-request.json"
    )
    request_snapshot = _write_qualification_control_document(request_path, request)
    destination_snapshot = _write_qualification_control_document(
        destination_path, event
    )
    if (
        destination_snapshot.payload != source.payload
        or destination_snapshot.sha256 != source.sha256
        or stat.S_IMODE(destination_snapshot.metadata.st_mode) != 0o444
        or destination_snapshot.metadata.st_nlink != 1
    ):
        raise ContractError("qualification broker descriptor readback mismatch")
    receipt_unsigned = qualification_broker_receipt_unsigned(
        request,
        destination_snapshot,
        broker_sequence=sequence,
        previous_receipt_sha256=previous_receipt_sha256,
        mode=mode,
    )
    receipt = _sign_qualification_document(
        receipt_unsigned,
        private_key_hex,
        mode,
        qualification_broker_receipt_signing_payload,
    )
    receipt_path = Path(broker_record_dir) / (
        receipt["receipt_id"] + ".broker-receipt.json"
    )
    receipt_snapshot = _write_qualification_control_document(receipt_path, receipt)
    return {
        "schema": "shohin-ocsc-qualification-broker-transfer-result-v1",
        "request": request,
        "request_path": request_snapshot.resolved_path,
        "event": event,
        "destination_path": destination_snapshot.resolved_path,
        "receipt": receipt,
        "receipt_path": receipt_snapshot.resolved_path,
        "retention_policy": "permanent-no-delete-no-rewrite",
        "authority": "qualification-source-review-only",
    }


def write_qualification_evidence_package(
    evidence_dir: Path,
    *,
    reviewer_id: str,
    sequence: int,
    nonce_hex: str,
    command: list[str],
    source_manifest_contract: dict,
    raw_events: list[dict],
    broker_requests: list[dict],
    broker_receipts: list[dict],
    private_key_hex: str,
    mode: str,
) -> dict:
    """Derive and persist the report, marker, and final receipt without overwrite."""

    source_manifest_contract = validate_source_manifest_contract(
        source_manifest_contract
    )
    if (
        not isinstance(reviewer_id, str)
        or not ID_RE.fullmatch(reviewer_id)
        or type(sequence) is not int
        or sequence <= 0
        or not isinstance(nonce_hex, str)
        or not HEX64_RE.fullmatch(nonce_hex)
        or not isinstance(command, list)
        or any(not isinstance(value, str) for value in command)
    ):
        raise ContractError("qualification evidence package identity mismatch")
    source_hash = source_manifest_contract["payload_sha256"]
    report = derive_qualification_report(
        raw_events,
        broker_requests,
        broker_receipts,
        source_hash,
        mode,
    )
    marker = qualification_marker(report)
    trusted_key = _qualification_trusted_key(mode)
    unsigned_receipt = {
        "schema": "shohin-ocsc-linux-lustre-qualification-receipt-v3",
        "qualification_id": report["qualification_id"],
        "reviewer_id": reviewer_id,
        "sequence": sequence,
        "nonce_hex": nonce_hex,
        "command": command,
        "command_sha256": hash_json(command),
        "source_manifest": source_manifest_contract,
        "source_manifest_sha256": source_hash,
        "raw_events": raw_events,
        "broker_requests": broker_requests,
        "broker_receipts": broker_receipts,
        "derived_report": report,
        "marker": marker,
        "claim_boundary": (
            "linux_lustre_qualification_source_evidence_only_no_bundle_consumer_"
            "fit_eval_gpu_or_scientific_authority"
        ),
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": trusted_key,
    }
    receipt = _sign_qualification_document(
        unsigned_receipt,
        private_key_hex,
        mode,
        linux_lustre_qualification_signing_payload,
    )
    prefix = report["qualification_id"]
    report_snapshot = _write_qualification_custody_document(
        Path(evidence_dir),
        prefix + ".derived-report.custody",
        "derived-report.json",
        report,
    )
    marker_snapshot = _write_qualification_custody_document(
        Path(evidence_dir),
        prefix + ".marker.custody",
        "marker.json",
        marker,
    )
    receipt_snapshot = _write_qualification_custody_document(
        Path(evidence_dir),
        prefix + ".receipt.custody",
        "receipt.json",
        receipt,
    )
    verification = load_linux_lustre_qualification_receipt(
        Path(receipt_snapshot.resolved_path),
        source_manifest_contract,
        mode,
        snapshot=receipt_snapshot,
    )
    return {
        "schema": "shohin-ocsc-qualification-evidence-package-v1",
        "report": report,
        "report_path": report_snapshot.resolved_path,
        "marker": marker,
        "marker_path": marker_snapshot.resolved_path,
        "receipt": receipt,
        "receipt_path": receipt_snapshot.resolved_path,
        "verification": verification,
        "retention_policy": "permanent-no-delete-no-rewrite",
        "authority": "qualification-source-review-only",
    }


def _load_qualification_signing_key(path: Path) -> str:
    snapshot = read_file_snapshot(
        Path(path),
        "qualification signing key",
        exact_mode=0o400,
        custody_root=True,
    )
    if (
        len(snapshot.payload) != 65
        or not snapshot.payload.endswith(b"\n")
        or b"\r" in snapshot.payload
    ):
        raise ContractError("qualification signing key file framing mismatch")
    try:
        private_key_hex = snapshot.payload[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("qualification signing key must be ASCII") from error
    if not HEX64_RE.fullmatch(private_key_hex):
        raise ContractError("qualification signing key bytes mismatch")
    return private_key_hex


def load_qualification_evidence_request(path: Path) -> dict:
    snapshot = read_file_snapshot(
        Path(path),
        "qualification raw-evidence request",
        exact_mode=0o444,
        custody_root=True,
    )
    document = strict_json_loads(
        snapshot.payload.decode("ascii"), "qualification raw-evidence request"
    )
    expected_keys = {
        "schema",
        "reviewer_id",
        "sequence",
        "nonce_hex",
        "command",
        "raw_events",
        "broker_requests",
        "broker_receipts",
        "payload_sha256",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or snapshot.payload != canonical_json_bytes(document, newline=True)
        or document.get("schema") != "shohin-ocsc-qualification-raw-evidence-request-v1"
        or not isinstance(document.get("reviewer_id"), str)
        or not ID_RE.fullmatch(document["reviewer_id"])
        or type(document.get("sequence")) is not int
        or document["sequence"] <= 0
        or not isinstance(document.get("nonce_hex"), str)
        or not HEX64_RE.fullmatch(document["nonce_hex"])
        or not isinstance(document.get("command"), list)
        or any(not isinstance(value, str) for value in document["command"])
        or not isinstance(document.get("raw_events"), list)
        or not isinstance(document.get("broker_requests"), list)
        or not isinstance(document.get("broker_receipts"), list)
        or document.get("payload_sha256")
        != hash_json(
            {key: value for key, value in document.items() if key != "payload_sha256"}
        )
    ):
        raise ContractError("qualification raw-evidence request mismatch")
    return document


def qualification_release_contract(identity: dict, lease_record: dict) -> dict:
    identity = qualification_publisher_identity(
        identity.get("publisher_id") if isinstance(identity, dict) else None,
        identity.get("sequence") if isinstance(identity, dict) else None,
        identity.get("nonce_hex") if isinstance(identity, dict) else None,
    )
    if not isinstance(lease_record, dict):
        raise ContractError("qualification release lease mismatch")
    document = {
        "schema": "shohin-ocsc-qualification-release-v1",
        **identity,
        "lease_name": lease_record.get("lease_name"),
        "lease_record_sha256": lease_record.get("payload_sha256"),
        "decision": "release-held-qualification-publisher",
    }
    return with_payload_hash(document, "payload_sha256")


def wait_for_qualification_release(
    path: Path,
    identity: dict,
    lease_record: dict,
    timeout_seconds: int,
) -> None:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3_600:
        raise ContractError("qualification release timeout mismatch")
    expected = qualification_release_contract(identity, lease_record)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            snapshot = read_file_snapshot(
                path,
                "qualification release receipt",
                exact_mode=0o444,
            )
        except ContractError as error:
            if not Path(path).exists():
                time.sleep(0.05)
                continue
            raise error
        document = strict_json_loads(
            snapshot.payload.decode("ascii"),
            "qualification release receipt",
        )
        if snapshot.payload != canonical_json_bytes(
            document, newline=True
        ) or not recursively_type_strict_equal(document, expected):
            raise ContractError("qualification release receipt mismatch")
        return
    raise ContractError("qualification release receipt timed out")


def publication_recovery_record(
    publication_receipt: dict,
    independent_review_receipt: dict,
    artifacts: dict[str, bytes],
    stage_device: int,
    stage_inode: int,
    stage_owner_uid: int,
    lease_record: dict,
) -> dict:
    stage_name, journal_name = publication_staging_names(
        publication_receipt, independent_review_receipt
    )
    request = publication_receipt["request"]
    record = {
        "schema": "shohin-ocsc-publication-recovery-v2",
        "stage_name": stage_name,
        "journal_name": journal_name,
        "canonical_name": Path(request["output_dir"]).name,
        "output_dir": request["output_dir"],
        "output_parent": request["output_parent"],
        "stage_directory_device": stage_device,
        "stage_directory_inode": stage_inode,
        "stage_directory_owner_uid": stage_owner_uid,
        "publication_lease": lease_record,
        "prepublication_commitment_sha256": publication_receipt["physical_sha256"],
        "prepublication_request_sha256": publication_receipt["request_sha256"],
        "prepublication_signer_public_key_hex": publication_receipt[
            "signer_public_key_hex"
        ],
        "independent_review_receipt_sha256": independent_review_receipt[
            "physical_sha256"
        ],
        "independent_review_request_sha256": independent_review_receipt[
            "review_request_sha256"
        ],
        "independent_review_signer_public_key_hex": independent_review_receipt[
            "signer_public_key_hex"
        ],
        "expected_output_identity": expected_output_identity(request, artifacts),
    }
    return with_payload_hash(record, "payload_sha256")


def _read_recovery_record_at(
    parent_fd: int,
    journal_name: str,
) -> tuple[dict, os.stat_result]:
    try:
        descriptor = os.open(
            journal_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise ContractError("publication recovery journal is not readable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size > 1024 * 1024
        ):
            raise ContractError("publication recovery journal identity mismatch")
        blocks = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        entry = os.stat(journal_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _file_state(before) != _file_state(after)
            or _file_state(before) != _file_state(entry)
            or len(payload) != before.st_size
        ):
            raise ContractError("publication recovery journal changed during read")
    finally:
        os.close(descriptor)
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("publication recovery journal must be ASCII") from error
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("publication recovery journal must use one final LF")
    document = strict_json_loads(text, "publication recovery journal")
    if payload != canonical_json_bytes(document, newline=True):
        raise ContractError("publication recovery journal is not canonical JSON")
    claimed = document.get("payload_sha256") if isinstance(document, dict) else None
    unhashed = dict(document) if isinstance(document, dict) else {}
    unhashed.pop("payload_sha256", None)
    if claimed != hash_json(unhashed):
        raise ContractError("publication recovery journal payload hash mismatch")
    return document, before


def _publish_recovery_record_at(
    parent_fd: int,
    journal_name: str,
    record: dict,
) -> os.stat_result:
    payload = canonical_json_bytes(record, newline=True)
    descriptor = os.open(
        journal_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        entry = os.stat(journal_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_size != len(payload)
            or _file_state(metadata) != _file_state(entry)
        ):
            raise ContractError("publication recovery journal identity mismatch")
        os.fsync(parent_fd)
        return metadata
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(
    source_name: str,
    destination_name: str,
    source_parent_fd: int,
    destination_parent_fd: int,
) -> None:
    """Atomically publish without replacing a late-created destination."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rename = libc.renameatx_np
            flags = 0x00000004  # RENAME_EXCL from sys/stdio.h.
        elif sys.platform.startswith("linux"):
            rename = libc.renameat2
            flags = 0x00000001  # RENAME_NOREPLACE from linux/fs.h.
        else:
            raise ContractError(
                "atomic no-replace publication is unsupported on this platform"
            )
    except AttributeError as error:
        raise ContractError(
            "atomic no-replace publication primitive is unavailable"
        ) from error
    except OSError as error:
        raise ContractError(
            "atomic no-replace publication primitive cannot be loaded"
        ) from error

    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    if error_number in {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }:
        raise ContractError(
            "atomic no-replace publication is unsupported by this filesystem"
        )
    raise ContractError(
        "atomic no-replace publication rename failed: {}".format(
            os.strerror(error_number)
        )
    )


def _open_bundle_directory_at(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise ContractError(label + " is not a readable directory") from error
    try:
        metadata = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or not _same_inode(metadata, entry)
        ):
            raise ContractError(label + " identity mismatch")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _strict_verify_publication_tree(
    path: Path,
    directory_fd: int,
    *,
    expected_output_dir: Path,
    require_unpublished: bool,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
    input_snapshots: dict[str, FileSnapshot],
    publication_commitment_snapshot: FileSnapshot,
    independent_review_snapshot: FileSnapshot,
    publication_receipt: dict,
    independent_review_receipt: dict,
    source_manifest_contract: dict,
    pinned_finalizer: Callable[[PinnedBundle], None] | None = None,
) -> dict:
    with PinnedBundle(path, directory_fd=directory_fd) as bundle_snapshot:
        result, _ = _verify_bundle_snapshot(
            bundle_snapshot,
            tokenizer_path,
            prompt_registry_path,
            confirmation_path,
            publication_commitment_path,
            independent_review_receipt_path,
            verification_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            independent_review_snapshot=independent_review_snapshot,
            expected_publication_receipt=publication_receipt,
            expected_independent_review_receipt=independent_review_receipt,
            source_manifest_contract=source_manifest_contract,
            expected_output_dir=expected_output_dir,
            require_unpublished=require_unpublished,
        )
        if pinned_finalizer is not None:
            pinned_finalizer(bundle_snapshot)
        bundle_snapshot.assert_unchanged()
        return result


def _recover_interrupted_publication(
    parent_fd: int,
    parent_resolved: str,
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
    input_snapshots: dict[str, FileSnapshot],
    publication_commitment_snapshot: FileSnapshot,
    independent_review_snapshot: FileSnapshot,
    publication_receipt: dict,
    independent_review_receipt: dict,
    source_manifest_contract: dict,
    lease_record: dict,
    lease_was_created: bool,
) -> bool:
    """Recover under a kernel lease and deterministic output-content authority."""

    stage_name, journal_name = publication_staging_names(
        publication_receipt, independent_review_receipt
    )
    canonical_state = _entry_state_or_none(parent_fd, output_dir.name)
    stage_state = _entry_state_or_none(parent_fd, stage_name)
    journal_state = _entry_state_or_none(parent_fd, journal_name)
    if journal_state is None:
        if stage_state is not None:
            raise ContractError(
                "unauthenticated publication stage collision; foreign tree retained"
            )
        if canonical_state is None:
            if not lease_was_created:
                raise ContractError(
                    "orphan publication lease is non-authoritative and retained"
                )
            return False
        canonical_fd, canonical_metadata = _open_bundle_directory_at(
            parent_fd, output_dir.name, "existing canonical bundle"
        )
        try:

            def finalize_existing(bundle_snapshot: PinnedBundle) -> None:
                current = os.stat(
                    output_dir.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if not _same_inode(current, canonical_metadata):
                    raise ContractError("canonical bundle changed during recovery")
                os.fsync(parent_fd)
                bundle_snapshot.assert_full_readback()
                _assert_directory_path_matches_fd(
                    output_dir.parent,
                    parent_resolved,
                    parent_fd,
                    "publication output parent",
                )

            _strict_verify_publication_tree(
                output_dir,
                canonical_fd,
                expected_output_dir=output_dir,
                require_unpublished=False,
                tokenizer_path=tokenizer_path,
                prompt_registry_path=prompt_registry_path,
                confirmation_path=confirmation_path,
                publication_commitment_path=publication_commitment_path,
                independent_review_receipt_path=independent_review_receipt_path,
                input_snapshots=input_snapshots,
                publication_commitment_snapshot=publication_commitment_snapshot,
                independent_review_snapshot=independent_review_snapshot,
                publication_receipt=publication_receipt,
                independent_review_receipt=independent_review_receipt,
                source_manifest_contract=source_manifest_contract,
                pinned_finalizer=finalize_existing,
            )
            return True
        finally:
            os.close(canonical_fd)

    record, authenticated_journal_state = _read_recovery_record_at(
        parent_fd, journal_name
    )
    if not _same_inode(journal_state, authenticated_journal_state):
        raise ContractError("publication recovery journal was substituted")
    if stage_state is not None and canonical_state is not None:
        raise ContractError("ambiguous publication recovery state; no tree was deleted")
    live_tree_state = stage_state if stage_state is not None else canonical_state
    if live_tree_state is None:
        raise ContractError(
            "orphan publication recovery metadata is non-authoritative and retained"
        )
    expected_record = publication_recovery_record(
        publication_receipt,
        independent_review_receipt,
        artifacts,
        live_tree_state.st_dev,
        live_tree_state.st_ino,
        live_tree_state.st_uid,
        lease_record,
    )
    if not recursively_type_strict_equal(record, expected_record):
        raise ContractError("publication recovery journal authentication failed")
    expected_inode = (live_tree_state.st_dev, live_tree_state.st_ino)
    if stage_state is not None:
        if (stage_state.st_dev, stage_state.st_ino) != expected_inode:
            raise ContractError(
                "publication recovery stage was substituted; foreign tree retained"
            )
        raise ContractError(
            "authenticated partial publication stage, journal, and lease are "
            "permanently retained; a new output identity is required"
        )
    if canonical_state is not None:
        if (canonical_state.st_dev, canonical_state.st_ino) != expected_inode:
            raise ContractError(
                "publication recovery canonical path was substituted; no tree was deleted"
            )
        canonical_fd, canonical_metadata = _open_bundle_directory_at(
            parent_fd, output_dir.name, "recoverable canonical bundle"
        )
        try:
            if (canonical_metadata.st_dev, canonical_metadata.st_ino) != expected_inode:
                raise ContractError(
                    "publication recovery canonical path was substituted"
                )

            def finalize_recovered(bundle_snapshot: PinnedBundle) -> None:
                current = os.stat(
                    output_dir.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if not _same_inode(current, canonical_metadata):
                    raise ContractError("canonical bundle changed during recovery")
                current_journal = os.stat(
                    journal_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if not _same_inode(current_journal, authenticated_journal_state):
                    raise ContractError("publication recovery journal changed")
                os.fsync(parent_fd)
                bundle_snapshot.assert_full_readback()
                _assert_directory_path_matches_fd(
                    output_dir.parent,
                    parent_resolved,
                    parent_fd,
                    "publication output parent",
                )

            _strict_verify_publication_tree(
                output_dir,
                canonical_fd,
                expected_output_dir=output_dir,
                require_unpublished=False,
                tokenizer_path=tokenizer_path,
                prompt_registry_path=prompt_registry_path,
                confirmation_path=confirmation_path,
                publication_commitment_path=publication_commitment_path,
                independent_review_receipt_path=independent_review_receipt_path,
                input_snapshots=input_snapshots,
                publication_commitment_snapshot=publication_commitment_snapshot,
                independent_review_snapshot=independent_review_snapshot,
                publication_receipt=publication_receipt,
                independent_review_receipt=independent_review_receipt,
                source_manifest_contract=source_manifest_contract,
                pinned_finalizer=finalize_recovered,
            )
            return True
        finally:
            os.close(canonical_fd)
    raise ContractError(
        "publication recovery state is non-authoritative and was retained"
    )


def publish_bundle(
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    mode: str,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    pad_token_id: int,
    publication_receipt: dict,
    independent_review_receipt: dict,
    input_snapshots: dict[str, FileSnapshot] | None = None,
    publication_commitment_snapshot: FileSnapshot | None = None,
    independent_review_snapshot: FileSnapshot | None = None,
    source_manifest_contract: dict | None = None,
    publication_event_hook: Callable[[str, dict], None] | None = None,
    qualification_crash_point: str | None = None,
    linux_lustre_qualification_verification: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    if set(artifacts) != set(ARTIFACT_NAMES) or any(
        not isinstance(payload, bytes) for payload in artifacts.values()
    ):
        raise ContractError("publication artifact inventory/type mismatch")
    if output_dir.name in {"", ".", ".."} or output_dir.parent == output_dir:
        raise ContractError("invalid output directory path")
    input_snapshots = custody_input_snapshots(
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        snapshots=input_snapshots,
        label_prefix="publication",
    )
    live_source_manifest = validate_source_manifest_contract(
        source_manifest()
        if source_manifest_contract is None
        else source_manifest_contract
    )
    if mode == "production":
        raise ContractError(
            "production publication is NO-GO until separately reviewed consumer, "
            "trainer, evaluator, report-validator, parameter-ledger, and "
            "train-eval-exclusion implementations exist"
        )
    if mode != "test":
        raise ContractError("publication mode mismatch")
    if (
        qualification_crash_point is not None
        and qualification_crash_point not in QUALIFICATION_CRASH_POINTS
    ):
        raise ContractError("qualification crash-point identity mismatch")
    if linux_lustre_qualification_verification is not None:
        raise ContractError("test publication cannot consume production qualification")
    commitment_path = (
        publication_receipt.get("resolved_path")
        if isinstance(publication_receipt, dict)
        else None
    )
    if not isinstance(commitment_path, str) or not commitment_path:
        raise ContractError("prepublication receipt path mismatch")
    independent_review_path = (
        independent_review_receipt.get("resolved_path")
        if isinstance(independent_review_receipt, dict)
        else None
    )
    if not isinstance(independent_review_path, str) or not independent_review_path:
        raise ContractError("independent review receipt path mismatch")
    publication_commitment_snapshot = publication_commitment_snapshot or (
        read_file_snapshot(
            Path(commitment_path),
            "publication prepublication commitment",
            exact_mode=0o444,
            custody_root=True,
        )
    )
    independent_review_snapshot = independent_review_snapshot or read_file_snapshot(
        Path(independent_review_path),
        "publication independent review receipt",
        exact_mode=0o444,
        custody_root=True,
    )
    publication_receipt = revalidate_publication_receipt(
        publication_receipt,
        mode,
        output_dir,
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        pad_token_id,
        require_unpublished=False,
        input_snapshots=input_snapshots,
        commitment_snapshot=publication_commitment_snapshot,
        source_manifest_contract=live_source_manifest,
    )
    request = publication_receipt["request"]
    independent_review_receipt = revalidate_independent_review_receipt(
        independent_review_receipt,
        Path(independent_review_path),
        request,
        artifacts,
        mode,
        snapshot=independent_review_snapshot,
    )
    parent_fd, parent_resolved, parent_metadata = _open_pinned_directory(
        output_dir.parent,
        "publication output parent",
        reject_other_writes=True,
    )
    signed_parent = request["output_parent"]
    live_parent = {
        "resolved_path": parent_resolved,
        "device": parent_metadata.st_dev,
        "inode": parent_metadata.st_ino,
        "mode": stat.S_IMODE(parent_metadata.st_mode),
        "owner_uid": parent_metadata.st_uid,
    }
    if live_parent != signed_parent or request["output_dir"] != str(
        Path(parent_resolved) / output_dir.name
    ):
        os.close(parent_fd)
        raise ContractError("publication output identity is not the signed identity")
    stage_name, journal_name = publication_staging_names(
        publication_receipt, independent_review_receipt
    )
    lease_name = publication_lease_name(publication_receipt, independent_review_receipt)
    stage_fd = None
    journal_state = None
    lease = None
    try:
        lease = _acquire_publication_lease(
            parent_fd,
            lease_name,
            event_hook=publication_event_hook,
        )
        if _recover_interrupted_publication(
            parent_fd,
            parent_resolved,
            output_dir,
            artifacts,
            tokenizer_path=tokenizer_path,
            prompt_registry_path=prompt_registry_path,
            confirmation_path=confirmation_path,
            publication_commitment_path=Path(commitment_path),
            independent_review_receipt_path=Path(independent_review_path),
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            independent_review_snapshot=independent_review_snapshot,
            publication_receipt=publication_receipt,
            independent_review_receipt=independent_review_receipt,
            source_manifest_contract=live_source_manifest,
            lease_record=lease.record,
            lease_was_created=lease.created,
        ):
            return
        publication_receipt = revalidate_publication_receipt(
            publication_receipt,
            mode,
            output_dir,
            tokenizer_path,
            prompt_registry_path,
            confirmation_path,
            pad_token_id,
            require_unpublished=True,
            input_snapshots=input_snapshots,
            commitment_snapshot=publication_commitment_snapshot,
            source_manifest_contract=live_source_manifest,
        )
        independent_review_receipt = revalidate_independent_review_receipt(
            independent_review_receipt,
            Path(independent_review_path),
            publication_receipt["request"],
            artifacts,
            mode,
            snapshot=independent_review_snapshot,
        )
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise ContractError(
                "authenticated publication stage name collided; foreign tree retained"
            ) from error
        stage_fd, stage_metadata = _open_bundle_directory_at(
            parent_fd, stage_name, "new publication stage"
        )
        if stat.S_IMODE(stage_metadata.st_mode) != 0o700:
            raise ContractError("new publication stage mode mismatch")
        _trigger_qualification_crash(
            qualification_crash_point,
            "stage-created-before-journal",
        )
        recovery_record = publication_recovery_record(
            publication_receipt,
            independent_review_receipt,
            artifacts,
            stage_metadata.st_dev,
            stage_metadata.st_ino,
            stage_metadata.st_uid,
            lease.record,
        )
        journal_state = _publish_recovery_record_at(
            parent_fd, journal_name, recovery_record
        )
        _trigger_qualification_crash(
            qualification_crash_point,
            "journal-durable-before-first-artifact",
        )
        for artifact_index, name in enumerate(ARTIFACT_NAMES):
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=stage_fd,
            )
            try:
                payload = artifacts[name]
                if (
                    artifact_index == 0
                    and qualification_crash_point == "partial-artifact-write"
                ):
                    if len(payload) < 2:
                        raise ContractError(
                            "qualification partial artifact is too short"
                        )
                    _write_all(descriptor, payload[: max(1, len(payload) // 2)])
                    os.fsync(descriptor)
                    _trigger_qualification_crash(
                        qualification_crash_point,
                        "partial-artifact-write",
                    )
                _write_all(descriptor, payload)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            metadata = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
                or metadata.st_size != len(artifacts[name])
            ):
                raise ContractError("staged artifact identity mismatch")
        os.fchmod(stage_fd, 0o555)
        os.fsync(stage_fd)
        _trigger_qualification_crash(
            qualification_crash_point,
            "stage-fsync-before-rename",
        )

        def finalize_staged(bundle_snapshot: PinnedBundle) -> None:
            final_receipt = revalidate_publication_receipt(
                publication_receipt,
                mode,
                output_dir,
                tokenizer_path,
                prompt_registry_path,
                confirmation_path,
                pad_token_id,
                require_unpublished=True,
                input_snapshots=input_snapshots,
                commitment_snapshot=publication_commitment_snapshot,
                source_manifest_contract=live_source_manifest,
            )
            if not recursively_type_strict_equal(final_receipt, publication_receipt):
                raise ContractError("prepublication receipt drifted before publication")
            final_review = revalidate_independent_review_receipt(
                independent_review_receipt,
                Path(independent_review_path),
                publication_receipt["request"],
                artifacts,
                mode,
                snapshot=independent_review_snapshot,
            )
            if not recursively_type_strict_equal(
                final_review, independent_review_receipt
            ):
                raise ContractError(
                    "independent review receipt drifted before publication"
                )
            pinned_parent = _assert_directory_path_matches_fd(
                output_dir.parent,
                parent_resolved,
                parent_fd,
                "publication output parent",
            )
            if (
                pinned_parent.st_dev != signed_parent["device"]
                or pinned_parent.st_ino != signed_parent["inode"]
                or stat.S_IMODE(pinned_parent.st_mode) != signed_parent["mode"]
                or pinned_parent.st_uid != signed_parent["owner_uid"]
            ):
                raise ContractError("signed output parent drifted before publication")
            bundle_snapshot.assert_full_readback()

        _strict_verify_publication_tree(
            Path(parent_resolved) / stage_name,
            stage_fd,
            expected_output_dir=output_dir,
            require_unpublished=True,
            tokenizer_path=tokenizer_path,
            prompt_registry_path=prompt_registry_path,
            confirmation_path=confirmation_path,
            publication_commitment_path=Path(commitment_path),
            independent_review_receipt_path=Path(independent_review_path),
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            independent_review_snapshot=independent_review_snapshot,
            publication_receipt=publication_receipt,
            independent_review_receipt=independent_review_receipt,
            source_manifest_contract=live_source_manifest,
            pinned_finalizer=finalize_staged,
        )
        try:
            _rename_directory_noreplace(
                stage_name,
                output_dir.name,
                parent_fd,
                parent_fd,
            )
        except FileExistsError as error:
            raise ContractError(
                "refusing to overwrite existing output directory"
            ) from error
        _trigger_qualification_crash(
            qualification_crash_point,
            "canonical-before-parent-fsync",
        )
        os.fsync(parent_fd)
        _assert_directory_path_matches_fd(
            output_dir.parent,
            parent_resolved,
            parent_fd,
            "publication output parent",
        )

        def commit_published(bundle_snapshot: PinnedBundle) -> None:
            _assert_directory_path_matches_fd(
                output_dir,
                str(Path(parent_resolved) / output_dir.name),
                stage_fd,
                "published bundle directory",
                exact_mode=0o555,
            )
            if journal_state is None:
                raise ContractError("publication recovery journal is missing")
            current_journal = os.stat(
                journal_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if not _same_inode(current_journal, journal_state):
                raise ContractError(
                    "publication recovery journal changed before commit"
                )
            bundle_snapshot.assert_full_readback()
            _assert_directory_path_matches_fd(
                output_dir.parent,
                parent_resolved,
                parent_fd,
                "publication output parent",
            )
            _assert_directory_path_matches_fd(
                output_dir,
                str(Path(parent_resolved) / output_dir.name),
                stage_fd,
                "published bundle directory",
                exact_mode=0o555,
            )

        _strict_verify_publication_tree(
            output_dir,
            stage_fd,
            expected_output_dir=output_dir,
            require_unpublished=False,
            tokenizer_path=tokenizer_path,
            prompt_registry_path=prompt_registry_path,
            confirmation_path=confirmation_path,
            publication_commitment_path=Path(commitment_path),
            independent_review_receipt_path=Path(independent_review_path),
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            independent_review_snapshot=independent_review_snapshot,
            publication_receipt=publication_receipt,
            independent_review_receipt=independent_review_receipt,
            source_manifest_contract=live_source_manifest,
            pinned_finalizer=commit_published,
        )
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        try:
            if lease is not None:
                _release_publication_lease(lease)
        finally:
            os.close(parent_fd)


def load_manifest(
    bundle: Path,
    snapshot: FileSnapshot | None = None,
) -> dict:
    manifest_path = Path(bundle) / "manifest.json"
    snapshot = snapshot or read_file_snapshot(
        manifest_path,
        "bundle manifest",
        exact_mode=0o444,
    )
    payload = snapshot.payload
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("bundle manifest must be ASCII") from error
    manifest = strict_json_loads(payload.decode("ascii"), "bundle manifest")
    expected_keys = {
        "schema",
        "mode",
        "files",
        "inputs",
        "source_manifest",
        "artifact_inventory_closed",
        "payload_sha256",
    }
    if set(manifest) != expected_keys:
        raise ContractError("bundle manifest key mismatch")
    if manifest["schema"] != "shohin-ocsc-bundle-manifest-v2":
        raise ContractError("bundle manifest schema mismatch")
    if manifest["mode"] not in {"production", "test"}:
        raise ContractError("bundle manifest mode mismatch")
    if manifest["artifact_inventory_closed"] is not True:
        raise ContractError("bundle manifest inventory is not closed")
    expected_files = set(ARTIFACT_NAMES) - {"manifest.json"}
    if (
        not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != expected_files
    ):
        raise ContractError("bundle manifest file inventory mismatch")
    for name, contract in manifest["files"].items():
        if name != Path(name).name or "/" in name or "\\" in name:
            raise ContractError("bundle manifest contains an unsafe path")
        if (
            not isinstance(contract, dict)
            or set(contract) != {"bytes", "sha256"}
            or type(contract["bytes"]) is not int
            or contract["bytes"] < 0
            or not isinstance(contract["sha256"], str)
            or not HEX64_RE.fullmatch(contract["sha256"])
        ):
            raise ContractError("bundle manifest file contract mismatch")
    custody_prefixes = (
        "tokenizer",
        "prompt_registry",
        "secret_confirmation_commitment",
        "prepublication_commitment",
    )
    expected_inputs = {
        "{}_{}".format(prefix, suffix)
        for prefix in custody_prefixes
        for suffix in (
            "sha256",
            "bytes",
            "path",
            "file_device",
            "file_inode",
            "custody_root_path",
            "custody_root_device",
            "custody_root_inode",
        )
    } | {
        "prepublication_request_sha256",
        "prepublication_custodian_id",
        "prepublication_sequence",
        "prepublication_signer_public_key_hex",
        "pad_token_id",
    }
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ContractError("bundle manifest input key mismatch")
    for field in (
        *("{}_sha256".format(prefix) for prefix in custody_prefixes),
        "prepublication_request_sha256",
        "prepublication_signer_public_key_hex",
    ):
        if not isinstance(inputs[field], str) or not HEX64_RE.fullmatch(inputs[field]):
            raise ContractError("bundle manifest input hash mismatch")
    if (
        any(
            type(inputs["{}_{}".format(prefix, suffix)]) is not int
            or inputs["{}_{}".format(prefix, suffix)] <= 0
            for prefix in custody_prefixes
            for suffix in (
                "bytes",
                "file_device",
                "file_inode",
                "custody_root_device",
                "custody_root_inode",
            )
        )
        or type(inputs["prepublication_sequence"]) is not int
        or inputs["prepublication_sequence"] <= 0
        or type(inputs["pad_token_id"]) is not int
        or inputs["pad_token_id"] < 0
    ):
        raise ContractError("bundle manifest input integer mismatch")
    for field in (
        field
        for prefix in custody_prefixes
        for field in (
            "{}_path".format(prefix),
            "{}_custody_root_path".format(prefix),
        )
    ):
        value = inputs[field]
        if (
            not isinstance(value, str)
            or not value.startswith("/")
            or ".." in Path(value).parts
            or str(Path(value)) != value
        ):
            raise ContractError("bundle manifest input path mismatch")
    if not isinstance(
        inputs["prepublication_custodian_id"], str
    ) or not ID_RE.fullmatch(inputs["prepublication_custodian_id"]):
        raise ContractError("bundle manifest custodian identity mismatch")
    source = manifest["source_manifest"]
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "schema",
            "sources",
            "runtime_closure",
            "bootstrap_source_identity",
            "payload_sha256",
        }
        or source["schema"] != "shohin-ocsc-source-manifest-v4"
        or not isinstance(source["sources"], dict)
        or set(source["sources"]) != set(SOURCE_PATHS)
        or source["payload_sha256"]
        != hash_json(
            {
                "sources": source["sources"],
                "runtime_closure": source["runtime_closure"],
                "bootstrap_source_identity": source["bootstrap_source_identity"],
            }
        )
    ):
        raise ContractError("bundle source manifest mismatch")
    for relative, contract in source["sources"].items():
        if (
            relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(contract, dict)
            or set(contract) != {"bytes", "sha256"}
            or type(contract["bytes"]) is not int
            or contract["bytes"] < 0
            or not isinstance(contract["sha256"], str)
            or not HEX64_RE.fullmatch(contract["sha256"])
        ):
            raise ContractError("bundle source path or contract mismatch")
    validate_runtime_closure_contract(source["runtime_closure"])
    claimed = manifest.get("payload_sha256")
    unhashed = dict(manifest)
    unhashed.pop("payload_sha256", None)
    if claimed != hash_json(unhashed):
        raise ContractError("bundle manifest payload hash mismatch")
    return manifest


def _verify_bundle_snapshot(
    bundle_snapshot: PinnedBundle,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
    *,
    verification_snapshots: dict[str, FileSnapshot] | None = None,
    publication_commitment_snapshot: FileSnapshot | None = None,
    independent_review_snapshot: FileSnapshot | None = None,
    expected_publication_receipt: dict | None = None,
    expected_independent_review_receipt: dict | None = None,
    source_manifest_contract: dict | None = None,
    expected_output_dir: Path | None = None,
    require_unpublished: bool = False,
) -> tuple[dict, dict]:
    bundle_snapshot.assert_unchanged()
    bundle = Path(bundle_snapshot.resolved_path)
    request_output_dir = (
        bundle if expected_output_dir is None else Path(expected_output_dir)
    )
    manifest = load_manifest(
        request_output_dir,
        bundle_snapshot.files["manifest.json"],
    )
    for name, contract in manifest["files"].items():
        file_snapshot = bundle_snapshot.files[name]
        if (
            len(file_snapshot.payload) != contract["bytes"]
            or file_snapshot.sha256 != contract["sha256"]
        ):
            raise ContractError("bundle artifact hash mismatch: {}".format(name))
    verification_snapshots = custody_input_snapshots(
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        snapshots=verification_snapshots,
        label_prefix="verification",
    )
    publication_commitment_snapshot = publication_commitment_snapshot or (
        read_file_snapshot(
            publication_commitment_path,
            "verification prepublication commitment",
            exact_mode=0o444,
            custody_root=True,
        )
    )
    live_source_manifest = validate_source_manifest_contract(
        source_manifest()
        if source_manifest_contract is None
        else source_manifest_contract
    )
    request = publication_commitment_request(
        manifest["mode"],
        request_output_dir,
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        int(manifest["inputs"]["pad_token_id"]),
        input_snapshots=verification_snapshots,
        source_manifest_contract=live_source_manifest,
    )
    publication_receipt = load_publication_commitment(
        publication_commitment_path,
        request,
        manifest["mode"],
        require_unpublished=require_unpublished,
        snapshot=publication_commitment_snapshot,
    )
    if expected_publication_receipt is not None and not recursively_type_strict_equal(
        publication_receipt, expected_publication_receipt
    ):
        raise ContractError("verification prepublication receipt drifted")
    artifact_payloads = {
        name: bundle_snapshot.files[name].payload for name in ARTIFACT_NAMES
    }
    independent_review_snapshot = independent_review_snapshot or read_file_snapshot(
        independent_review_receipt_path,
        "verification independent review receipt",
        exact_mode=0o444,
        custody_root=True,
    )
    independent_review_receipt = load_independent_review_receipt(
        independent_review_receipt_path,
        independent_review_request(request, artifact_payloads),
        manifest["mode"],
        snapshot=independent_review_snapshot,
    )
    if (
        expected_independent_review_receipt is not None
        and not recursively_type_strict_equal(
            independent_review_receipt, expected_independent_review_receipt
        )
    ):
        raise ContractError("verification independent review receipt drifted")
    input_contracts = (
        (
            "tokenizer",
            "tokenizer",
            verification_snapshots["tokenizer"],
        ),
        (
            "prompt registry",
            "prompt_registry",
            verification_snapshots["prompt_registry"],
        ),
        (
            "confirmation commitment",
            "secret_confirmation_commitment",
            verification_snapshots["secret_confirmation_commitment"],
        ),
    )
    for label, prefix, snapshot in input_contracts:
        live_fields = manifest_custody_fields(
            prefix,
            custody_snapshot_contract(snapshot),
        )
        if any(
            manifest["inputs"].get(key) != value for key, value in live_fields.items()
        ):
            raise ContractError("verification {} commitment mismatch".format(label))
    receipt_fields = {
        "prepublication_commitment_path": publication_receipt["resolved_path"],
        "prepublication_commitment_bytes": publication_receipt["physical_bytes"],
        "prepublication_commitment_sha256": publication_receipt["physical_sha256"],
        "prepublication_commitment_file_device": publication_receipt[
            "physical_file_device"
        ],
        "prepublication_commitment_file_inode": publication_receipt[
            "physical_file_inode"
        ],
        "prepublication_commitment_custody_root_path": publication_receipt[
            "custody_root_path"
        ],
        "prepublication_commitment_custody_root_device": publication_receipt[
            "custody_root_device"
        ],
        "prepublication_commitment_custody_root_inode": publication_receipt[
            "custody_root_inode"
        ],
    }
    if (
        any(
            manifest["inputs"].get(key) != value
            for key, value in receipt_fields.items()
        )
        or manifest["inputs"]["prepublication_request_sha256"]
        != publication_receipt["request_sha256"]
        or manifest["inputs"]["prepublication_custodian_id"]
        != publication_receipt["custodian_id"]
        or manifest["inputs"]["prepublication_sequence"]
        != publication_receipt["sequence"]
        or manifest["inputs"]["prepublication_signer_public_key_hex"]
        != publication_receipt["signer_public_key_hex"]
    ):
        raise ContractError("verification prepublication receipt mismatch")
    if (
        manifest["inputs"]["tokenizer_sha256"]
        != verification_snapshots["tokenizer"].sha256
    ):
        raise ContractError("verification tokenizer hash mismatch")
    if (
        manifest["inputs"]["prompt_registry_sha256"]
        != verification_snapshots["prompt_registry"].sha256
    ):
        raise ContractError("verification prompt registry hash mismatch")
    if (
        manifest["inputs"]["secret_confirmation_commitment_sha256"]
        != verification_snapshots["secret_confirmation_commitment"].sha256
    ):
        raise ContractError("verification confirmation commitment hash mismatch")
    expected = build_artifacts(
        manifest["mode"],
        tokenizer_path,
        prompt_registry_path,
        confirmation_path,
        int(manifest["inputs"]["pad_token_id"]),
        publication_receipt,
        expected_output_dir=request_output_dir,
        require_unpublished=require_unpublished,
        input_snapshots=verification_snapshots,
        publication_commitment_snapshot=publication_commitment_snapshot,
        source_manifest_contract=live_source_manifest,
    )
    for name in ARTIFACT_NAMES:
        if bundle_snapshot.files[name].payload != expected[name]:
            raise ContractError(
                "bundle differs from deterministic reconstruction: " + name
            )
    for label, path in (
        ("tokenizer", Path(tokenizer_path)),
        ("prompt_registry", Path(prompt_registry_path)),
        ("secret_confirmation_commitment", Path(confirmation_path)),
    ):
        final_snapshot = read_file_snapshot(
            path,
            "final verification " + label,
            exact_mode=0o444,
            custody_root=True,
        )
        if custody_snapshot_contract(final_snapshot) != custody_snapshot_contract(
            verification_snapshots[label]
        ):
            raise ContractError("verification custody input drifted: " + label)
    final_publication_snapshot = read_file_snapshot(
        publication_commitment_path,
        "final verification prepublication commitment",
        exact_mode=0o444,
        custody_root=True,
    )
    if custody_snapshot_contract(
        final_publication_snapshot
    ) != custody_snapshot_contract(publication_commitment_snapshot):
        raise ContractError(
            "verification custody input drifted: prepublication_commitment"
        )
    final_independent_review_snapshot = read_file_snapshot(
        independent_review_receipt_path,
        "final verification independent review receipt",
        exact_mode=0o444,
        custody_root=True,
    )
    if custody_snapshot_contract(
        final_independent_review_snapshot
    ) != custody_snapshot_contract(independent_review_snapshot):
        raise ContractError(
            "verification custody input drifted: independent_review_receipt"
        )
    bundle_snapshot.assert_full_readback()
    result = {
        "schema": "shohin-ocsc-verification-v1",
        "verified": True,
        "bundle_manifest_sha256": bundle_snapshot.files["manifest.json"].sha256,
        "bundle_directory_device": bundle_snapshot.metadata.st_dev,
        "bundle_directory_inode": bundle_snapshot.metadata.st_ino,
        "prepublication_commitment_sha256": publication_receipt["physical_sha256"],
        "independent_review_receipt_sha256": independent_review_receipt[
            "physical_sha256"
        ],
        "artifact_count": len(ARTIFACT_NAMES),
    }
    return result, manifest


def verify_bundle(
    bundle: Path,
    tokenizer_path: Path,
    prompt_registry_path: Path,
    confirmation_path: Path,
    publication_commitment_path: Path,
    independent_review_receipt_path: Path,
) -> dict:
    with PinnedBundle(bundle) as bundle_snapshot:
        result, _ = _verify_bundle_snapshot(
            bundle_snapshot,
            tokenizer_path,
            prompt_registry_path,
            confirmation_path,
            publication_commitment_path,
            independent_review_receipt_path,
        )
        bundle_snapshot.assert_unchanged()
        return result


def bootstrap_cli(argv: list[str]) -> tuple[dict, dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path)
    action.add_argument("--qualification-output-dir", type=Path)
    action.add_argument("--prepare-publication-request", type=Path)
    action.add_argument("--prepare-independent-review-request", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--verify-hidden-opening", type=Path)
    action.add_argument("--verify-linux-lustre-qualification-receipt", type=Path)
    action.add_argument("--print-source-manifest", action="store_true")
    action.add_argument("--qualification-broker-transfer-event", type=Path)
    action.add_argument("--qualification-write-evidence-package", type=Path)
    parser.add_argument("--mode", choices=("production", "test"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--custodian-opening", type=Path)
    parser.add_argument("--publication-commitment", type=Path)
    parser.add_argument("--independent-review-receipt", type=Path)
    parser.add_argument("--linux-lustre-qualification-receipt", type=Path)
    parser.add_argument("--prompt-registry", type=Path)
    parser.add_argument("--secret-confirmation-commitment", type=Path)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--qualification-publisher-id")
    parser.add_argument("--qualification-publisher-sequence", type=int)
    parser.add_argument("--qualification-publisher-nonce-hex")
    parser.add_argument("--qualification-lease-receipt-out", type=Path)
    parser.add_argument("--qualification-result-receipt-out", type=Path)
    parser.add_argument("--qualification-release-receipt", type=Path)
    parser.add_argument("--qualification-signing-key", type=Path)
    parser.add_argument("--qualification-broker-record-dir", type=Path)
    parser.add_argument("--qualification-publication-event-dir", type=Path)
    parser.add_argument("--qualification-broker-id")
    parser.add_argument("--qualification-broker-sequence", type=int)
    parser.add_argument("--qualification-previous-request-sha256")
    parser.add_argument("--qualification-previous-receipt-sha256")
    parser.add_argument("--qualification-raw-evidence-request", type=Path)
    parser.add_argument(
        "--qualification-crash-point",
        choices=QUALIFICATION_CRASH_POINTS,
    )
    parser.add_argument(
        "--qualification-release-timeout-seconds",
        type=int,
        default=300,
    )
    args = parser.parse_args(argv)
    if not (
        sys.flags.isolated
        and sys.flags.safe_path
        and sys.flags.no_user_site
        and sys.flags.no_site
        and sys.dont_write_bytecode
    ):
        raise ContractError("source-bound CLI execution requires Python -I -S -B")
    execution = bootstrap_execution_contract(argv, required=True)
    if (
        args.qualification_crash_point is not None
        and args.qualification_output_dir is None
    ):
        raise ContractError(
            "qualification crash points require --qualification-output-dir"
        )
    if args.print_source_manifest:
        bound_source_manifest = source_manifest()
        return (
            {
                "schema": "shohin-ocsc-source-manifest-inspection-v1",
                "source_manifest": bound_source_manifest,
                "qualification_authority": False,
                "publication_authority": False,
            },
            bound_source_manifest,
        )
    qualification_source_action = (
        args.qualification_broker_transfer_event is not None
        or args.qualification_write_evidence_package is not None
    )
    if qualification_source_action:
        if (
            execution["profile"] != "qualification"
            or args.mode not in {"production", "test"}
            or args.qualification_signing_key is None
        ):
            raise ContractError(
                "qualification source actions require the qualification profile, "
                "mode, and pinned signing key"
            )
        bound_source_manifest = source_manifest()
        private_key_hex = _load_qualification_signing_key(
            args.qualification_signing_key
        )
        if args.qualification_broker_transfer_event is not None:
            broker_arguments = (
                args.qualification_broker_record_dir,
                args.qualification_publication_event_dir,
                args.qualification_broker_id,
                args.qualification_broker_sequence,
                args.qualification_previous_request_sha256,
                args.qualification_previous_receipt_sha256,
            )
            if any(value is None for value in broker_arguments):
                raise ContractError(
                    "qualification broker transfer arguments are incomplete"
                )
            result = execute_qualification_broker_transfer(
                args.qualification_broker_transfer_event,
                args.qualification_broker_record_dir,
                args.qualification_publication_event_dir,
                broker_id=args.qualification_broker_id,
                sequence=args.qualification_broker_sequence,
                previous_request_sha256=args.qualification_previous_request_sha256,
                previous_receipt_sha256=args.qualification_previous_receipt_sha256,
                expected_source_manifest_sha256=bound_source_manifest["payload_sha256"],
                private_key_hex=private_key_hex,
                mode=args.mode,
            )
            return result, bound_source_manifest
        if args.qualification_raw_evidence_request is None:
            raise ContractError(
                "qualification evidence package requires one raw-evidence request"
            )
        evidence = load_qualification_evidence_request(
            args.qualification_raw_evidence_request
        )
        result = write_qualification_evidence_package(
            args.qualification_write_evidence_package,
            reviewer_id=evidence["reviewer_id"],
            sequence=evidence["sequence"],
            nonce_hex=evidence["nonce_hex"],
            command=evidence["command"],
            source_manifest_contract=bound_source_manifest,
            raw_events=evidence["raw_events"],
            broker_requests=evidence["broker_requests"],
            broker_receipts=evidence["broker_receipts"],
            private_key_hex=private_key_hex,
            mode=args.mode,
        )
        return result, bound_source_manifest
    if args.prompt_registry is None or args.secret_confirmation_commitment is None:
        raise ContractError(
            "CLI execution requires prompt registry and secret confirmation commitment"
        )
    bound_source_manifest = source_manifest()

    request_output = (
        args.prepare_publication_request
        or args.prepare_independent_review_request
        or args.qualification_output_dir
        or args.output_dir
    )
    if request_output is not None:
        if args.mode is None or args.tokenizer is None:
            raise ContractError("request construction requires mode and tokenizer")
        input_snapshots = custody_input_snapshots(
            args.tokenizer,
            args.prompt_registry,
            args.secret_confirmation_commitment,
            label_prefix="source-bound CLI",
        )
        request = publication_commitment_request(
            args.mode,
            request_output,
            args.tokenizer,
            args.prompt_registry,
            args.secret_confirmation_commitment,
            args.pad_token_id,
            input_snapshots=input_snapshots,
            source_manifest_contract=bound_source_manifest,
        )
    if args.prepare_publication_request is not None:
        if (
            args.publication_commitment is not None
            or args.independent_review_receipt is not None
        ):
            raise ContractError("publication request preparation accepts no receipts")
        result = request
    elif args.verify_linux_lustre_qualification_receipt is not None:
        if args.mode is None or args.mode not in {"test", "production"}:
            raise ContractError("qualification receipt verification requires mode")
        result = load_linux_lustre_qualification_receipt(
            args.verify_linux_lustre_qualification_receipt,
            bound_source_manifest,
            args.mode,
        )
    elif args.prepare_independent_review_request is not None:
        if (
            args.publication_commitment is None
            or args.independent_review_receipt is not None
        ):
            raise ContractError(
                "review preparation requires only a prepublication commitment"
            )
        publication_commitment_snapshot = read_file_snapshot(
            args.publication_commitment,
            "review preparation prepublication commitment",
            exact_mode=0o444,
            custody_root=True,
        )
        publication_receipt = load_publication_commitment(
            args.publication_commitment,
            request,
            args.mode,
            require_unpublished=False,
            snapshot=publication_commitment_snapshot,
        )
        artifacts = build_artifacts(
            args.mode,
            args.tokenizer,
            args.prompt_registry,
            args.secret_confirmation_commitment,
            args.pad_token_id,
            publication_receipt,
            expected_output_dir=args.prepare_independent_review_request,
            require_unpublished=False,
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            source_manifest_contract=bound_source_manifest,
        )
        result = independent_review_request(request, artifacts)
    elif args.output_dir is not None or args.qualification_output_dir is not None:
        publication_output = args.output_dir or args.qualification_output_dir
        qualification_mode = args.qualification_output_dir is not None
        if (
            args.publication_commitment is None
            or args.independent_review_receipt is None
        ):
            raise ContractError("generation requires both external receipts")
        qualification_arguments = (
            args.qualification_publisher_id,
            args.qualification_publisher_sequence,
            args.qualification_publisher_nonce_hex,
            args.qualification_lease_receipt_out,
            args.qualification_result_receipt_out,
        )
        if qualification_mode:
            if execution["profile"] != "qualification" or any(
                value is None for value in qualification_arguments
            ):
                raise ContractError(
                    "qualification publication requires the qualification profile, publisher identity, and both receipt hooks"
                )
            publisher_identity = qualification_publisher_identity(
                args.qualification_publisher_id,
                args.qualification_publisher_sequence,
                args.qualification_publisher_nonce_hex,
            )
        elif any(value is not None for value in qualification_arguments) or (
            args.qualification_release_receipt is not None
        ):
            raise ContractError(
                "qualification publisher hooks require --qualification-output-dir"
            )
        publication_commitment_snapshot = read_file_snapshot(
            args.publication_commitment,
            "generation prepublication commitment",
            exact_mode=0o444,
            custody_root=True,
        )
        publication_receipt = load_publication_commitment(
            args.publication_commitment,
            request,
            args.mode,
            require_unpublished=False,
            snapshot=publication_commitment_snapshot,
        )
        artifacts = build_artifacts(
            args.mode,
            args.tokenizer,
            args.prompt_registry,
            args.secret_confirmation_commitment,
            args.pad_token_id,
            publication_receipt,
            expected_output_dir=publication_output,
            require_unpublished=False,
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            source_manifest_contract=bound_source_manifest,
        )
        independent_review_snapshot = read_file_snapshot(
            args.independent_review_receipt,
            "generation independent review receipt",
            exact_mode=0o444,
            custody_root=True,
        )
        independent_review_receipt = load_independent_review_receipt(
            args.independent_review_receipt,
            independent_review_request(request, artifacts),
            args.mode,
            snapshot=independent_review_snapshot,
        )
        if args.mode == "production":
            raise ContractError(
                "production publication is NO-GO until separately reviewed consumer, "
                "trainer, evaluator, report-validator, parameter-ledger, and "
                "train-eval-exclusion implementations exist"
            )
        if args.linux_lustre_qualification_receipt is not None:
            raise ContractError(
                "qualification receipts cannot authorize bundle publication"
            )
        qualification_verification = None
        acquired_details = None

        def qualification_event_hook(event: str, details: dict) -> None:
            nonlocal acquired_details
            if not qualification_mode:
                raise ContractError("unexpected qualification event hook")
            receipt = qualification_publisher_receipt(
                publisher_identity,
                event,
                details,
                publication_output,
                bound_source_manifest,
            )
            _write_qualification_control_document(
                args.qualification_lease_receipt_out,
                receipt,
            )
            if event == "lease-acquired":
                acquired_details = details
                if args.qualification_release_receipt is not None:
                    wait_for_qualification_release(
                        args.qualification_release_receipt,
                        publisher_identity,
                        details["lease_record"],
                        args.qualification_release_timeout_seconds,
                    )

        publish_bundle(
            publication_output,
            artifacts,
            mode=args.mode,
            tokenizer_path=args.tokenizer,
            prompt_registry_path=args.prompt_registry,
            confirmation_path=args.secret_confirmation_commitment,
            pad_token_id=args.pad_token_id,
            publication_receipt=publication_receipt,
            independent_review_receipt=independent_review_receipt,
            input_snapshots=input_snapshots,
            publication_commitment_snapshot=publication_commitment_snapshot,
            independent_review_snapshot=independent_review_snapshot,
            source_manifest_contract=bound_source_manifest,
            publication_event_hook=(
                qualification_event_hook if qualification_mode else None
            ),
            qualification_crash_point=(
                args.qualification_crash_point if qualification_mode else None
            ),
            linux_lustre_qualification_verification=qualification_verification,
        )
        if qualification_mode:
            if acquired_details is None:
                raise ContractError("qualification publisher acquired no lease")
            completion = qualification_publisher_receipt(
                publisher_identity,
                "publication-verified",
                acquired_details,
                publication_output,
                bound_source_manifest,
            )
            _write_qualification_control_document(
                args.qualification_result_receipt_out,
                completion,
            )
        result = {
            "generated": True,
            "qualification_mode": qualification_mode,
            "output_dir": _lexical_absolute_path(
                publication_output, "publication output directory"
            ),
            "manifest_sha256": sha256_file(publication_output / "manifest.json"),
            "rows": {"ocsc": CORPUS_ROWS, "iid_control": CORPUS_ROWS},
            "updates_per_arm": UPDATES_PER_ARM,
        }
    elif args.verify is not None:
        if (
            args.mode is not None
            or args.tokenizer is None
            or args.publication_commitment is None
            or args.independent_review_receipt is None
        ):
            raise ContractError(
                "verification reads mode from the bundle and requires custody inputs"
            )
        result = verify_bundle(
            args.verify,
            args.tokenizer,
            args.prompt_registry,
            args.secret_confirmation_commitment,
            args.publication_commitment,
            args.independent_review_receipt,
        )
    else:
        if (
            args.mode is not None
            or args.bundle is None
            or args.tokenizer is None
            or args.custodian_opening is None
            or args.publication_commitment is None
            or args.independent_review_receipt is None
        ):
            raise ContractError(
                "hidden opening verification requires every custody input and no mode"
            )
        result = verify_hidden_opening(
            args.verify_hidden_opening,
            args.secret_confirmation_commitment,
            args.bundle,
            args.tokenizer,
            args.prompt_registry,
            args.custodian_opening,
            args.publication_commitment,
            args.independent_review_receipt,
        )
    final_execution = bootstrap_execution_contract(argv, required=True)
    if not recursively_type_strict_equal(execution, final_execution):
        raise ContractError("bootstrap execution identity drifted during CLI operation")
    validate_source_manifest_contract(bound_source_manifest)
    return result, bound_source_manifest


def main() -> None:
    try:
        result, _ = bootstrap_cli(sys.argv[1:])
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    except ContractError as error:
        raise SystemExit("OCSC contract rejected: {}".format(error)) from error


if __name__ == "__main__":
    main()
