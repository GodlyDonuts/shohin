#!/usr/bin/env python3
"""Development-only EOS-suppressed DWS trace and field-use screen."""

from __future__ import annotations

import argparse
import array
import base64
import csv
import ctypes
import errno
import fcntl
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import secrets
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import types
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


PROTOCOL_ID = "R12-DWS-EOS-SUPPRESSED-FIELD-SCREEN-DEV-v4"
OUTPUT_SCHEMA = "r12_dws_eos_suppressed_field_screen_dev_v4"
WRAPPER_ACCEPTANCE_SCHEMA = "r12_dws_eos_wrapper_acceptance_v8"
GENERATOR_ATTESTATION_SCHEMA = "r12_dws_eos_generator_attestation_v2"
RUNTIME_OBSERVATION_SCHEMA = "r12_dws_eos_runtime_observation_v2"
ACCEPTED_BUNDLE_SCHEMA = "r12_dws_eos_canonical_report_bundle_v5"
ACCEPTANCE_MARKER_SCHEMA = "r12_dws_eos_post_publication_marker_v5"
DURABLE_ACCEPTANCE_RECEIPT_SCHEMA = (
    "r12_dws_eos_durable_post_fsync_acceptance_receipt_v1"
)
INDEPENDENT_VERIFIER_SCHEMA = "r12_dws_eos_independent_verifier_v2"
RUN_AUTHORIZATION_SCHEMA = "r12_dws_eos_external_run_authorization_v5"
LINUX_QUALIFICATION_SCHEMA = "r12_dws_eos_linux_publication_qualification_v6"
LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_SCHEMA = (
    "r12_dws_eos_linux_qualification_authorization_receipt_v1"
)
LINUX_QUALIFICATION_REPORT_SCHEMA = "r12_dws_eos_linux_receipt_report_v3"
LINUX_QUALIFICATION_MARKER_SCHEMA = "r12_dws_eos_linux_receipt_marker_v3"
LINUX_QUALIFICATION_RECEIPT_SCHEMA = "r12_dws_eos_linux_receipt_mechanics_v3"
LINUX_RECEIPT_CRASH_STAGES = (
    ("before_publication_parent_fsync", 91, "rejected"),
    ("during_partial_receipt_write", 92, "rejected"),
    ("after_complete_receipt_write", 93, "accepted"),
)
LINUX_QUALIFICATION_REPORT_NAME = "qualification-report.json"
LINUX_QUALIFICATION_MARKER_NAME = "qualification-marker.json"
LINUX_QUALIFICATION_RECEIPT_NAME = "qualification-receipt.json"
LINUX_QUALIFICATION_RECEIPT_STATUS = (
    "linux_receipt_mechanics_complete_after_required_fsyncs"
)
LINUX_QUALIFICATION_AUTHORITY_BOUNDARY = (
    "mechanics_only_ephemeral_broker_key_no_production_or_test_authority"
)
LINUX_QUALIFICATION_STATUS = (
    "linux_lustre_mechanics_exercised_for_exact_byte_review_only"
)
LINUX_QUALIFICATION_CLAIM_BOUNDARY = (
    "not_production_authority_not_h100_scientific_evidence"
)
LINUX_QUALIFICATION_REQUIRED_CHECKS = (
    "publisher_lease_acquired",
    "concurrent_flock_rejected",
    "production_scm_rights_broker_transfer",
    "ephemeral_signed_exact_stale_cleanup_quarantined",
    "foreign_inode_after_quarantine_rename_preserved",
    "renameat2_noreplace",
    "directory_fsync_failure_rollback",
    "o_sync_receipt_before_parent_fsync_rejected",
    "o_sync_partial_receipt_rejected",
    "o_sync_complete_receipt_independent_replay_accepted",
    "held_evaluator_pathname_substitution_exercised",
    "symlink_rejected",
    "hardlink_rejected",
    "random_temp_crash_cleanup",
    "held_directory_pathname_substitution_rejected",
    "lustre_file_rename_fsync_reopen",
)
PRIVATE_CANDIDATE_STATE = "private_candidate_requires_live_wrapper_acceptance"
CLAIM_BOUNDARY = (
    "Development-only field-use diagnostic. Fixed token overrides supply an external halt "
    "veto and decode clock; they are not autonomous halting or reasoning. Full-state "
    "recurrence is NO-GO. Nominal trace accuracy and counterfactual full-target accuracy "
    "cannot compensate for failed paired carry target-switch response; that failure is a "
    "noncompensatory carry-use veto. Fresh latest-state core-prompt replay is an external "
    "compound intervention that jointly removes prior context, re-encodes position, and "
    "canonicalizes the state surface. Recovery identifies only that compound package; it "
    "does not isolate a stale-source mechanism or autonomous recurrence. "
    "The Slurm controller, accounting database, configured device cgroup, and externally "
    "pinned production authorization key are explicit operational trust roots rather "
    "than model evidence. No promotion, hidden confirmation, verifier-backed capability, "
    "or H100 claim is authorized. CPU mechanics are development-GO; Linux publication, "
    "Newton deployment, and H100 launch remain NO-GO pending fresh hostile review and a "
    "real Linux/Lustre qualification run."
)

ROOT = Path(
    os.environ.get("R12_REVIEWED_SOURCE_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
CHECKPOINT_PATH = ROOT / "train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt"
TOKENIZER_PATH = ROOT / "artifacts/shohin-tok-32k.json"
HELDOUT_PATH = ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl"
PREREG_PATH = ROOT / "R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md"

RUNTIME_SOURCE_MANIFEST_SCHEMA = "r12_dws_eos_suppressed_runtime_sources_v7"
RUNTIME_IDENTITY_SCHEMA = "r12_dws_eos_runtime_identity_v4"
SOURCE_EXECUTION_MODE = "sealed_memfd_evaluator_and_model_verified_bytes"
PYTHON_STARTUP_MODE = "exec_absolute_pinned_interpreter_-I_-S_-B"
DELEGATED_KEY_BROKER_REQUEST_SCHEMA = "r12_delegated_key_broker_request_v1"
DELEGATED_KEY_BROKER_RESPONSE_SCHEMA = "r12_delegated_key_broker_response_v1"
DELEGATED_KEY_BROKER_REQUEST_KEYS = {
    "schema",
    "process_id",
    "python_startup_mode",
    "python_executable",
    "python_startup",
    "run_authorization_sha256",
    "source_manifest_sha256",
    "evaluator_sha256",
    "wrapper_sha256",
    "delegated_public_key_hex",
    "delegated_private_key_sha256",
}
RUNTIME_SOURCE_PATHS = (
    "R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md",
    "train/eval_dws_eos_suppressed_trace.py",
    "train/jobs/eval_dws_eos_suppressed_trace.sbatch",
    "train/model.py",
    "train/test_eval_dws_eos_suppressed_trace.py",
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SDPA_BACKEND = "math"
REQUIRED_MEMFD_SEALS = 0x000F
SLURM_PARTITION = "normal"
SLURM_NODE_COUNT = 1
SLURM_TASK_COUNT = 1
SLURM_CPUS_PER_TASK = 4
SLURM_MEMORY = "64G"
SLURM_MEMORY_BYTES = 64 * 1024**3
SLURM_TIME_LIMIT = "08:00:00"
SLURM_TIME_LIMIT_SECONDS = 8 * 60 * 60
SLURM_GPU_TYPE = "nvidia_h100_pcie"
SLURM_GPU_COUNT = 1
SLURM_GRES = f"gpu:{SLURM_GPU_TYPE}:{SLURM_GPU_COUNT}"
SLURM_TYPED_GPU_TRES = f"gres/gpu:{SLURM_GPU_TYPE}"
REQUIRED_CUDA_DEVICE_NAME = "NVIDIA H100 PCIe"
REQUIRED_CUDA_DEVICE_CAPABILITY = (9, 0)
REQUIRED_CUDA_MEMORY_MIN_BYTES = 75 * 1024**3
REQUIRED_CUDA_MEMORY_MAX_BYTES = 85 * 1024**3
NVIDIA_CONTROL_DEVICE_SPECS = (
    ("nvidiactl", "physical_major", 255),
    ("nvidia-modeset", "physical_major", 254),
    ("nvidia-uvm", "uvm_major", 0),
    ("nvidia-uvm-tools", "uvm_major", 1),
)
ACCEPTANCE_COMMIT_SUFFIX = ".r12-acceptance-commit.json"
DURABLE_ACCEPTANCE_RECEIPT_SUFFIX = ".r12-durable-acceptance.json"
DELEGATED_PUBLICATION_SCOPES = (
    "post_publication_commit_marker",
    "durable_post_fsync_acceptance_receipt",
)

# This is a public-only review anchor. The corresponding production private key is not
# present in this repository. Replacing the anchor requires a reviewed source change.
PRODUCTION_AUTHORITY_KEY_ID = "r12-production-authority-2026-07-v1"
PRODUCTION_AUTHORITY_PUBLIC_KEY_HEX = (
    "3805039655eef59153ba2b148551df2376d7c2cfa550ee5f6386745d4d0ed857"
)
PRODUCTION_AUTHORITY_PUBLIC_KEY_SHA256 = (
    "b47cb1db3d3ef97ad5cf9f80405e979f4be58f5ea1d570f092825f272bd978dc"
)
PRODUCTION_AUTHORITY_FILE_SHA256 = (
    "09b3f77db76b1277c79f82fa28920fff1d749925231de1ba1f85e73e2ae7e642"
)
PRODUCTION_AUTHORITY_SCOPE = "production_external_authority"
TEST_AUTHORITY_SCOPE = "test_only_no_production_authority"
ALLOW_TEST_AUTHORITY = False
TEST_AUTHORITY_PUBLIC_KEY_HEX: str | None = None

torch: Any = None
F: Any = None
Tokenizer: Any = None

EXPECTED_SHA256 = {
    "checkpoint": "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459",
    "tokenizer": "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4",
    "heldout": "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646",
}
EXPECTED_CHECKPOINT_STEP = "sft_ep1"
EXPECTED_REGIMES = (
    "fit_w4",
    "fit_w6",
    "value_ood_w4",
    "value_ood_w6",
    "width_ood_w8",
)
OPERATIONS = ("add", "sub")
SELECTION_DOMAIN = b"R12-DWS-EOS-SUPPRESSED-FIELD-SCREEN-DEV-v1\x00"
CASES_PER_CELL = 10
CASE_COUNT = 100
ORDERED_CASE_IDS_SHA256 = (
    "c83796c32fdc69efd99bff579103b0a6e2be9812cbc94b91a061bcbb24a1ad7b"
)
REPLICATION_CASE_IDS = (
    "fit_w4-00258",
    "value_ood_w4-00217",
    "fit_w4-00261",
    "fit_w4-00196",
    "fit_w6-00122",
    "value_ood_w6-00028",
    "value_ood_w6-00280",
    "value_ood_w6-00067",
    "width_ood_w8-00120",
    "width_ood_w8-00176",
    "width_ood_w8-00180",
    "width_ood_w8-00103",
)
REPLICATION_CASE_IDS_SHA256 = (
    "1dc75ec7995e61a85f7bec9ae1fa62aa1adaf71bd46172e880aea901482396b9"
)
FRESH_REENCODING_GATES = {
    "intact_adjacent_exact_overall_min": 0.90,
    "intact_adjacent_exact_each_width_min": 0.80,
    "carry_full_target_exact_overall_min": 0.75,
    "carry_full_target_exact_each_width_min": 0.65,
    "carry_output_switch_overall_min": 0.90,
    "carry_output_switch_each_width_min": 0.80,
    "carry_paired_switch_overall_min": 0.75,
    "carry_paired_switch_each_width_min": 0.70,
    "paired_recovery_vs_full_history_lf_overall_min": 0.40,
    "paired_recovery_vs_full_history_lf_each_width_min": 0.30,
}

PROMPT_PREFIX = (
    "Microstate update. Digits in a, b, and r are least-significant first. "
    "Use the digit at p with c, write only r[p], then advance p by one.\nState: "
)
PROMPT_SUFFIX = "\nReturn exactly one dws state line.\nAnswer:"
PROMPT_ENCODING = "ascii"
PROMPT_PREFIX_SHA256 = (
    "875d5f9e27adefcd06c06be7e68f177f3fc6e7d5865e4c14003782871b8a96a1"
)
PROMPT_SUFFIX_SHA256 = (
    "60307c5d8511691fe05a0d9c346714c304503a28da707c598a7533b41b5967f6"
)
PROMPT_TEMPLATE_SHA256 = (
    "90713da16e103d71e8ff70806a355bacf89815d637f670ffd04c62cc1c0d5814"
)

MAX_NEW_TOKENS = 768
DEVICE = "cuda"
PRECISION = "cuda_bfloat16_autocast"
EOS_TOKEN_ID = 0
REPLACEMENT_TOKENS = {
    "eos_to_lf_211": (211, "\n"),
    "eos_to_space_233": (233, " "),
    "eos_to_semicolon_39": (39, ";"),
    "eos_to_nonformat_x_100": (100, "x"),
}


class ContractError(RuntimeError):
    """Raised when any frozen mechanical contract is violated."""


class DecodeMode(str, Enum):
    ORDINARY_EOS_STOP = "ordinary_eos_stop"
    EOS_MASKED_ARGMAX = "eos_masked_argmax"
    EOS_TO_LF = "eos_to_lf_211"
    EOS_TO_SPACE = "eos_to_space_233"
    EOS_TO_SEMICOLON = "eos_to_semicolon_39"
    EOS_TO_NONFORMAT_X = "eos_to_nonformat_x_100"


PRIMARY_ARM_ORDER = tuple(mode.value for mode in DecodeMode)
FIXED_BUDGET_ARMS = frozenset(PRIMARY_ARM_ORDER[1:])
FIELD_CLOCK_ARMS = PRIMARY_ARM_ORDER[1:]
HISTORY_BRANCHES = (
    "intact",
    "carry_flip",
    "written_result_r0_flip",
    "active_operand_digit_perturbation",
    "equal_token_length_destroyed_history",
)
FRESH_REENCODING_BRANCHES = HISTORY_BRANCHES[:4]


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    """The complete token-only input visible to one decode."""

    model: Any
    prompt_token_ids: tuple[int, ...]
    device: str
    max_new_tokens: int
    mode: DecodeMode
    eos_token_id: int


@dataclass(frozen=True, slots=True)
class TokenDecision:
    logits_after_intervention: torch.Tensor
    raw_argmax_id: int
    selected_token_id: int
    eos_was_raw_argmax: bool
    eos_logit: float
    selected_raw_logit: float
    override_applied: bool


@dataclass(frozen=True, slots=True)
class RawEosEvent:
    generated_index: int
    absolute_token_position: int
    eos_logit: float
    non_eos_argmax_token_id: int
    non_eos_argmax_logit: float
    replacement_token_id: int | None
    replacement_raw_logit: float | None


@dataclass(frozen=True, slots=True)
class RawDecode:
    mode: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    eos_mask_applied_positions: tuple[int, ...]
    eos_events: tuple[RawEosEvent, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if type(token_id) is not int or token_id < 0:
            raise ContractError("token IDs must be nonnegative integers")
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _required_memfd_seals() -> int:
    return REQUIRED_MEMFD_SEALS


def read_sealed_memfd_bytes(descriptor: int, byte_count: int, label: str) -> bytes:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 0
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise ContractError(f"{label} must be an anonymous regular inode")
    if not hasattr(fcntl, "F_GET_SEALS"):
        raise ContractError("memfd seal inspection is unavailable")
    required_seals = _required_memfd_seals()
    seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    payload, observed_info = _read_exact_descriptor_bytes(
        descriptor, label, expected_info=info
    )
    if len(payload) != byte_count or observed_info.st_size != byte_count:
        raise ContractError(f"{label} byte count differs")
    if seals != required_seals:
        raise ContractError(f"{label} seals differ")
    return payload


def read_sealed_source_descriptor(
    descriptor: int, expected_sha256: str, label: str
) -> tuple[bytes, dict[str, Any]]:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0 or info.st_size <= 0:
        raise ContractError(f"{label} must be a nonempty anonymous regular inode")
    payload = read_sealed_memfd_bytes(descriptor, info.st_size, label)
    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise ContractError(f"{label} hash differs from sealed source manifest")
    return payload, {
        "descriptor_kind": "sealed_memfd",
        "sha256": observed_sha256,
        "byte_count": len(payload),
        "seals": fcntl.fcntl(descriptor, fcntl.F_GET_SEALS),
    }


_ED25519_Q = 2**255 - 19
_ED25519_L = 2**252 + 27742317777372353535851937790883648493
_ED25519_D = (-121665 * pow(121666, _ED25519_Q - 2, _ED25519_Q)) % _ED25519_Q
_ED25519_I = pow(2, (_ED25519_Q - 1) // 4, _ED25519_Q)


def _ed25519_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED25519_D * y * y + 1, _ED25519_Q - 2, _ED25519_Q)
    x = pow(xx % _ED25519_Q, (_ED25519_Q + 3) // 8, _ED25519_Q)
    if (x * x - xx) % _ED25519_Q:
        x = (x * _ED25519_I) % _ED25519_Q
    if (x * x - xx) % _ED25519_Q:
        raise ValueError("Ed25519 point is not on the curve")
    if x & 1:
        x = _ED25519_Q - x
    return x


_ED25519_BY = (4 * pow(5, _ED25519_Q - 2, _ED25519_Q)) % _ED25519_Q
_ED25519_B = (_ed25519_xrecover(_ED25519_BY), _ED25519_BY)
_ED25519_IDENTITY = (0, 1)


def _ed25519_extended(point: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = point
    return x, y, 1, (x * y) % _ED25519_Q


def _ed25519_extended_add(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = ((y1 - x1) * (y2 - x2)) % _ED25519_Q
    b = ((y1 + x1) * (y2 + x2)) % _ED25519_Q
    c = (2 * _ED25519_D * t1 * t2) % _ED25519_Q
    d = (2 * z1 * z2) % _ED25519_Q
    e = (b - a) % _ED25519_Q
    f = (d - c) % _ED25519_Q
    g = (d + c) % _ED25519_Q
    h = (b + a) % _ED25519_Q
    return (
        e * f % _ED25519_Q,
        g * h % _ED25519_Q,
        f * g % _ED25519_Q,
        e * h % _ED25519_Q,
    )


def _ed25519_extended_double(
    point: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, z, _t = point
    a = x * x % _ED25519_Q
    b = y * y % _ED25519_Q
    c = 2 * z * z % _ED25519_Q
    d = -a % _ED25519_Q
    e = ((x + y) * (x + y) - a - b) % _ED25519_Q
    g = (d + b) % _ED25519_Q
    f = (g - c) % _ED25519_Q
    h = (d - b) % _ED25519_Q
    return (
        e * f % _ED25519_Q,
        g * h % _ED25519_Q,
        f * g % _ED25519_Q,
        e * h % _ED25519_Q,
    )


def _ed25519_affine(point: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, z, _t = point
    inverse = pow(z, _ED25519_Q - 2, _ED25519_Q)
    return x * inverse % _ED25519_Q, y * inverse % _ED25519_Q


def _ed25519_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return _ed25519_affine(
        _ed25519_extended_add(_ed25519_extended(left), _ed25519_extended(right))
    )


def _ed25519_scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _ed25519_extended(_ED25519_IDENTITY)
    addend = _ed25519_extended(point)
    while scalar:
        if scalar & 1:
            result = _ed25519_extended_add(result, addend)
        addend = _ed25519_extended_double(addend)
        scalar >>= 1
    return _ed25519_affine(result)


def _ed25519_encode_point(point: tuple[int, int]) -> bytes:
    x, y = point
    if not (0 <= x < _ED25519_Q and 0 <= y < _ED25519_Q):
        raise ValueError("Ed25519 affine coordinates are noncanonical")
    return int(y | ((x & 1) << 255)).to_bytes(32, "little")


def _ed25519_decode_point(encoded: bytes) -> tuple[int, int]:
    if len(encoded) != 32:
        raise ValueError("Ed25519 point length differs")
    encoded_integer = int.from_bytes(encoded, "little")
    x_sign = encoded_integer >> 255
    y = encoded_integer & ((1 << 255) - 1)
    if y >= _ED25519_Q:
        raise ValueError("Ed25519 point encoding is noncanonical")
    x = _ed25519_xrecover(y)
    if x == 0 and x_sign:
        raise ValueError("Ed25519 point encoding has a noncanonical zero sign")
    if (x & 1) != x_sign:
        x = _ED25519_Q - x
    point = (x, y)
    if _ed25519_encode_point(point) != encoded:
        raise ValueError("Ed25519 point encoding differs")
    if point == _ED25519_IDENTITY:
        raise ValueError("Ed25519 identity point is forbidden")
    if _ed25519_scalar_mult(point, _ED25519_L) != _ED25519_IDENTITY:
        raise ValueError("Ed25519 point is not in the prime-order subgroup")
    return point


def _ed25519_decode_scalar(encoded: bytes) -> int:
    if len(encoded) != 32:
        raise ValueError("Ed25519 scalar length differs")
    scalar = int.from_bytes(encoded, "little")
    if scalar >= _ED25519_L:
        raise ValueError("Ed25519 scalar encoding is out of range")
    return scalar


def _ed25519_expanded_private_key(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("Ed25519 seed length differs")
    digest = hashlib.sha512(seed).digest()
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    return int.from_bytes(scalar_bytes, "little"), digest[32:]


def _ed25519_public_key(seed: bytes) -> bytes:
    scalar, _prefix = _ed25519_expanded_private_key(seed)
    return _ed25519_encode_point(_ed25519_scalar_mult(_ED25519_B, scalar))


def _ed25519_sign(seed: bytes, message: bytes) -> bytes:
    scalar, prefix = _ed25519_expanded_private_key(seed)
    public_key = _ed25519_encode_point(_ed25519_scalar_mult(_ED25519_B, scalar))
    nonce = (
        int.from_bytes(hashlib.sha512(prefix + message).digest(), "little") % _ED25519_L
    )
    encoded_r = _ed25519_encode_point(_ed25519_scalar_mult(_ED25519_B, nonce))
    challenge = (
        int.from_bytes(
            hashlib.sha512(encoded_r + public_key + message).digest(), "little"
        )
        % _ED25519_L
    )
    encoded_s = ((nonce + challenge * scalar) % _ED25519_L).to_bytes(32, "little")
    return encoded_r + encoded_s


def _ed25519_verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
    try:
        if len(public_key) != 32 or len(signature) != 64:
            return False
        encoded_r = signature[:32]
        scalar = _ed25519_decode_scalar(signature[32:])
        authority_point = _ed25519_decode_point(public_key)
        r_point = _ed25519_decode_point(encoded_r)
        challenge = (
            int.from_bytes(
                hashlib.sha512(encoded_r + public_key + message).digest(), "little"
            )
            % _ED25519_L
        )
        return _ed25519_scalar_mult(_ED25519_B, scalar) == _ed25519_add(
            r_point, _ed25519_scalar_mult(authority_point, challenge)
        )
    except (OverflowError, ValueError):
        return False


def signing_key_record(private_key_bytes: bytes) -> dict[str, Any]:
    if len(private_key_bytes) != 32:
        raise ContractError("Ed25519 private key must contain exactly 32 bytes")
    public_key_bytes = _ed25519_public_key(private_key_bytes)
    return {
        "descriptor_kind": "sealed_memfd",
        "byte_count": 32,
        "private_key_sha256": sha256_bytes(private_key_bytes),
        "public_key_hex": public_key_bytes.hex(),
        "seals": _required_memfd_seals(),
    }


def read_sealed_signing_key(descriptor: int, expected: dict[str, Any]) -> bytes:
    private_key_bytes = read_sealed_memfd_bytes(descriptor, 32, "signing key")
    observed = {
        **signing_key_record(private_key_bytes),
        "seals": fcntl.fcntl(descriptor, fcntl.F_GET_SEALS),
    }
    if observed != expected:
        raise ContractError("signing-key descriptor identity differs")
    return private_key_bytes


def delegated_signing_key_record_from_authorization(
    authorization: dict[str, Any],
) -> dict[str, Any]:
    public_key_hex = authorization["delegated_marker_public_key_hex"]
    private_key_sha256 = authorization["delegated_marker_private_key_sha256"]
    if (
        not isinstance(public_key_hex, str)
        or re.fullmatch(r"[0-9a-f]{64}", public_key_hex) is None
        or not isinstance(private_key_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", private_key_sha256) is None
    ):
        raise ContractError("delegated signing-key authorization encoding differs")
    try:
        _ed25519_decode_point(bytes.fromhex(public_key_hex))
    except ValueError as error:
        raise ContractError("delegated signing public key is invalid") from error
    return {
        "descriptor_kind": "sealed_memfd",
        "byte_count": 32,
        "private_key_sha256": private_key_sha256,
        "public_key_hex": public_key_hex,
        "seals": _required_memfd_seals(),
    }


def build_delegated_key_broker_request(
    *,
    authorization_sha256: str,
    source_manifest_sha256: str,
    evaluator_sha256: str,
    wrapper_sha256: str,
    runtime_identity: dict[str, Any],
    expected_signing_key: dict[str, Any],
    process_id: int | None = None,
) -> dict[str, Any]:
    for label, digest in (
        ("authorization", authorization_sha256),
        ("source manifest", source_manifest_sha256),
        ("evaluator", evaluator_sha256),
        ("wrapper", wrapper_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ContractError(f"delegated-key broker {label} hash differs")
    _require_exact_keys(
        expected_signing_key,
        SIGNING_KEY_RECORD_KEYS,
        "delegated-key broker expected signing key",
    )
    requester_pid = os.getpid() if process_id is None else process_id
    if type(requester_pid) is not int or requester_pid <= 0:
        raise ContractError("delegated-key broker process ID differs")
    request = {
        "schema": DELEGATED_KEY_BROKER_REQUEST_SCHEMA,
        "process_id": requester_pid,
        "python_startup_mode": PYTHON_STARTUP_MODE,
        "python_executable": runtime_identity["python"],
        "python_startup": runtime_identity["python_startup"],
        "run_authorization_sha256": authorization_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "evaluator_sha256": evaluator_sha256,
        "wrapper_sha256": wrapper_sha256,
        "delegated_public_key_hex": expected_signing_key["public_key_hex"],
        "delegated_private_key_sha256": expected_signing_key["private_key_sha256"],
    }
    _require_exact_keys(
        request, DELEGATED_KEY_BROKER_REQUEST_KEYS, "delegated-key broker request"
    )
    return request


def receive_delegated_signing_key_from_broker(
    broker_descriptor: int,
    request: dict[str, Any],
    expected_signing_key: dict[str, Any],
    *,
    key_reader: Any = None,
) -> tuple[int, bytes]:
    """Receive authority only after the exec'd, non-dumpable process is attested."""
    broker_info = os.fstat(broker_descriptor)
    broker_identity = (
        broker_info.st_dev,
        broker_info.st_ino,
        broker_info.st_mode,
        broker_info.st_nlink,
        broker_info.st_uid,
    )
    if broker_descriptor < 3 or not stat.S_ISSOCK(broker_info.st_mode):
        raise ContractError("delegated-key broker descriptor is not a socket")
    request_payload = stable_json_bytes(request)
    expected_response = stable_json_bytes(
        {
            "schema": DELEGATED_KEY_BROKER_RESPONSE_SCHEMA,
            "request_sha256": sha256_bytes(request_payload),
        }
    )
    received_descriptors: list[int] = []
    duplicate = os.dup(broker_descriptor)
    try:
        duplicate_info = os.fstat(duplicate)
        if (
            duplicate_info.st_dev,
            duplicate_info.st_ino,
            duplicate_info.st_mode,
            duplicate_info.st_nlink,
            duplicate_info.st_uid,
        ) != broker_identity:
            os.close(duplicate)
            raise ContractError("delegated-key broker descriptor changed before use")
        with socket.socket(fileno=duplicate) as broker:
            socket_type = broker.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
            if broker.family != socket.AF_UNIX or socket_type not in {
                socket.SOCK_SEQPACKET,
                socket.SOCK_DGRAM,
            }:
                raise ContractError(
                    "delegated-key broker must be a connected message-boundary Unix socket"
                )
            broker.settimeout(30.0)
            broker.sendall(request_payload)
            ancillary_size = socket.CMSG_SPACE(array.array("i", [0]).itemsize)
            response, ancillary, flags, _address = broker.recvmsg(
                len(expected_response) + 1, ancillary_size
            )
            if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
                raise ContractError("delegated-key broker response was truncated")
            for level, message_type, data in ancillary:
                if level != socket.SOL_SOCKET or message_type != socket.SCM_RIGHTS:
                    raise ContractError("delegated-key broker ancillary data differs")
                descriptors = array.array("i")
                if len(data) % descriptors.itemsize:
                    raise ContractError("delegated-key broker descriptor data differs")
                descriptors.frombytes(data)
                received_descriptors.extend(descriptors)
            after_info = os.fstat(broker.fileno())
            if (
                after_info.st_dev,
                after_info.st_ino,
                after_info.st_mode,
                after_info.st_nlink,
                after_info.st_uid,
            ) != broker_identity:
                raise ContractError(
                    "delegated-key broker descriptor changed during use"
                )
    except (OSError, TimeoutError) as error:
        for descriptor in received_descriptors:
            os.close(descriptor)
        raise ContractError("delegated-key broker exchange failed") from error
    except BaseException:
        for descriptor in received_descriptors:
            os.close(descriptor)
        raise
    if response != expected_response or len(received_descriptors) != 1:
        for descriptor in received_descriptors:
            os.close(descriptor)
        raise ContractError("delegated-key broker response identity differs")
    delegated_descriptor = received_descriptors[0]
    os.set_inheritable(delegated_descriptor, False)
    if os.get_inheritable(delegated_descriptor):
        os.close(delegated_descriptor)
        raise ContractError("delegated signing-key descriptor remained inheritable")
    try:
        reader = read_sealed_signing_key if key_reader is None else key_reader
        private_key = reader(delegated_descriptor, expected_signing_key)
    except BaseException:
        os.close(delegated_descriptor)
        raise
    return delegated_descriptor, private_key


def external_file_record(
    path: Path, payload: bytes, info: os.stat_result
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "size": info.st_size,
    }


def slurm_authorization_projection(slurm: dict[str, Any]) -> dict[str, Any]:
    _validate_slurm_identity(slurm)
    return {key: slurm[key] for key in SLURM_IDENTITY_KEYS}


def run_authorization_payload(authorization: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(authorization, RUN_AUTHORIZATION_KEYS, "run authorization")
    return {
        key: value for key, value in authorization.items() if key != "signature_hex"
    }


def sign_run_authorization(
    payload: dict[str, Any], authority_private_key_bytes: bytes
) -> dict[str, Any]:
    """Test/custodian helper; production root private material must remain external."""
    if set(payload) != RUN_AUTHORIZATION_KEYS - {"signature_hex"}:
        raise ContractError("run-authorization payload keys differ")
    if len(authority_private_key_bytes) != 32:
        raise ContractError("authority private key length differs")
    return {
        **payload,
        "signature_hex": _ed25519_sign(
            authority_private_key_bytes, stable_json_bytes(payload)
        ).hex(),
    }


def _trusted_authority_public_key(authorization: dict[str, Any]) -> str:
    scope = authorization["authority_scope"]
    if scope == PRODUCTION_AUTHORITY_SCOPE:
        return PRODUCTION_AUTHORITY_PUBLIC_KEY_HEX
    if (
        scope == TEST_AUTHORITY_SCOPE
        and ALLOW_TEST_AUTHORITY
        and TEST_AUTHORITY_PUBLIC_KEY_HEX is not None
    ):
        return TEST_AUTHORITY_PUBLIC_KEY_HEX
    raise ContractError("run authorization is not rooted in production authority")


def validate_run_authorization(
    authorization: dict[str, Any],
    *,
    expected_slurm: dict[str, Any],
    expected_output_directory: dict[str, Any],
    expected_accepted_name: str,
    expected_source_manifest_sha256: str,
    expected_delegated_marker_key: dict[str, Any],
    require_current: bool,
    now: datetime | None = None,
) -> None:
    _require_plain_json_tree(authorization, "run authorization")
    payload = run_authorization_payload(authorization)
    _require_exact_keys(
        authorization["frozen_inputs"],
        AUTHORIZED_FROZEN_INPUT_KEYS,
        "run authorization frozen inputs",
    )
    _require_exact_keys(
        authorization["slurm_allocation"],
        AUTHORIZED_SLURM_KEYS,
        "run authorization Slurm allocation",
    )
    validate_replayable_linux_qualification_receipt(
        authorization["linux_qualification_receipt"]
    )
    _validate_stale_cleanup_entries(
        authorization["stale_cleanup_entries"],
        accepted_name=expected_accepted_name,
        job_id=expected_slurm["job_id"],
        authorization_nonce=authorization["authorization_nonce"],
    )
    authority_public_key_hex = _trusted_authority_public_key(authorization)
    authority_public_key_sha256 = sha256_bytes(bytes.fromhex(authority_public_key_hex))
    expected_key_id = (
        PRODUCTION_AUTHORITY_KEY_ID
        if authorization["authority_scope"] == PRODUCTION_AUTHORITY_SCOPE
        else "r12-test-authority-no-production-authority"
    )
    if (
        authorization["schema"] != RUN_AUTHORIZATION_SCHEMA
        or authorization["authority_key_id"] != expected_key_id
        or authorization["authority_public_key_sha256"] != authority_public_key_sha256
        or type(authorization["authorization_sequence"]) is not int
        or authorization["authorization_sequence"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", authorization["authorization_nonce"]) is None
        or re.fullmatch(r"[0-9a-f]{40}", authorization["source_commit"]) is None
        or authorization["source_manifest_sha256"] != expected_source_manifest_sha256
        or authorization["output_directory"] != expected_output_directory
        or authorization["accepted_name"] != expected_accepted_name
        or authorization["output_path"]
        != str(Path(expected_output_directory["path"]) / expected_accepted_name)
        or not _strict_equal(
            authorization["slurm_allocation"],
            slurm_authorization_projection(expected_slurm),
        )
        or authorization["delegated_marker_public_key_hex"]
        != expected_delegated_marker_key["public_key_hex"]
        or authorization["delegated_marker_private_key_sha256"]
        != expected_delegated_marker_key["private_key_sha256"]
        or authorization["delegated_publication_scopes"]
        != list(DELEGATED_PUBLICATION_SCOPES)
        or authorization["ordered_case_ids_sha256"] != ORDERED_CASE_IDS_SHA256
    ):
        raise ContractError("run authorization cross-binding differs")
    for key in ("source_manifest_path", "output_path"):
        if (
            type(authorization[key]) is not str
            or not Path(authorization[key]).is_absolute()
        ):
            raise ContractError(f"run authorization path differs at {key}")
    frozen = authorization["frozen_inputs"]
    expected_frozen = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": EXPECTED_SHA256["checkpoint"],
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": EXPECTED_SHA256["tokenizer"],
        "heldout_path": str(HELDOUT_PATH),
        "heldout_sha256": EXPECTED_SHA256["heldout"],
        "prereg_path": str(PREREG_PATH),
    }
    if any(frozen[key] != value for key, value in expected_frozen.items()):
        raise ContractError("run authorization frozen input path/hash differs")
    if re.fullmatch(r"[0-9a-f]{64}", frozen["prereg_sha256"]) is None:
        raise ContractError("run authorization prereg hash differs")
    issued = _require_utc_timestamp(
        authorization["issued_at_utc"], "authorization issued"
    )
    not_before = _require_utc_timestamp(
        authorization["not_before_utc"], "authorization not-before"
    )
    expires = _require_utc_timestamp(
        authorization["expires_at_utc"], "authorization expiry"
    )
    if not (issued <= not_before < expires):
        raise ContractError("run authorization timestamp ordering differs")
    if require_current:
        observed_now = now or datetime.now(timezone.utc)
        if not_before > observed_now or observed_now >= expires:
            raise ContractError("run authorization is not currently valid")
    if (
        type(authorization["signature_hex"]) is not str
        or re.fullmatch(r"[0-9a-f]{128}", authorization["signature_hex"]) is None
    ):
        raise ContractError("run authorization signature encoding differs")
    if not _ed25519_verify(
        bytes.fromhex(authority_public_key_hex),
        bytes.fromhex(authorization["signature_hex"]),
        stable_json_bytes(payload),
    ):
        raise ContractError("run authorization signature does not verify")


def _generator_attestation_payload(
    report_body: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": GENERATOR_ATTESTATION_SCHEMA,
        "scheme": "ed25519_over_canonical_attestation_payload",
        "report_body_sha256": sha256_bytes(stable_json_bytes(report_body)),
        "generator_public_key_hex": context["generator_signing_key"]["public_key_hex"],
        "sealed_evaluator_sha256": context["sealed_generator"]["evaluator_sha256"],
        "sealed_model_sha256": context["sealed_generator"]["model_sha256"],
        "wrapper_sha256": context["wrapper_sha256"],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "slurm_job_id": context["slurm_identity"]["job_id"],
        "nonce": context["nonce"],
    }


def attach_generator_attestation(
    report_body: dict[str, Any], context: dict[str, Any], private_key_bytes: bytes
) -> dict[str, Any]:
    if signing_key_record(private_key_bytes) != context["generator_signing_key"]:
        raise ContractError("generator signing key differs from context")
    if "generator_attestation" in report_body:
        raise ContractError("report body already contains a generator attestation")
    payload = _generator_attestation_payload(report_body, context)
    attestation = {
        **payload,
        "signature_hex": _ed25519_sign(
            private_key_bytes, stable_json_bytes(payload)
        ).hex(),
    }
    return {**report_body, "generator_attestation": attestation}


def validate_generator_attestation(
    report: dict[str, Any], context: dict[str, Any]
) -> None:
    attestation = report["generator_attestation"]
    _require_exact_keys(
        attestation, GENERATOR_ATTESTATION_KEYS, "generator_attestation"
    )
    report_body = {
        key: value for key, value in report.items() if key != "generator_attestation"
    }
    payload = _generator_attestation_payload(report_body, context)
    if {key: attestation[key] for key in payload} != payload:
        raise ContractError("generator attestation does not verify")
    signature_hex = attestation["signature_hex"]
    public_key_hex = context["generator_signing_key"]["public_key_hex"]
    if (
        not isinstance(signature_hex, str)
        or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None
        or re.fullmatch(r"[0-9a-f]{64}", public_key_hex) is None
    ):
        raise ContractError("generator attestation signature encoding differs")
    if not _ed25519_verify(
        bytes.fromhex(public_key_hex),
        bytes.fromhex(signature_hex),
        stable_json_bytes(payload),
    ):
        raise ContractError("generator attestation does not verify")


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
    )


def _inode_policy_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
    )


def _read_exact_descriptor_bytes(
    descriptor: int,
    label: str,
    *,
    expected_info: os.stat_result | None = None,
    expected_payload: bytes | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one held regular inode and retain full stat and byte identity."""
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
        raise ContractError(f"{label} descriptor is not a regular file")
    if expected_info is not None and _file_identity(before) != _file_identity(
        expected_info
    ):
        raise ContractError(f"{label} descriptor identity differs")
    payload = os.pread(descriptor, before.st_size + 1, 0)
    after = os.fstat(descriptor)
    if len(payload) != before.st_size or _file_identity(after) != _file_identity(
        before
    ):
        raise ContractError(f"{label} descriptor changed while reading")
    if expected_payload is not None and payload != expected_payload:
        raise ContractError(f"{label} descriptor bytes differ")
    return payload, after


def _read_held_directory_entry_descriptor(
    directory_fd: int,
    name: str,
    descriptor: int,
    label: str,
    *,
    expected_info: os.stat_result | None = None,
    expected_payload: bytes | None = None,
) -> tuple[bytes, os.stat_result]:
    """Consume a directory entry only through its already-held descriptor."""
    _validate_directory_entry_name(name, label)
    path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    payload, info = _read_exact_descriptor_bytes(
        descriptor,
        label,
        expected_info=expected_info,
        expected_payload=expected_payload,
    )
    if _file_identity(path_before) != _file_identity(info):
        raise ContractError(f"{label} pathname does not name the held inode")
    path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _file_identity(path_after) != _file_identity(info):
        raise ContractError(f"{label} pathname changed while reading")
    return payload, info


def _open_regular_single_link(path: Path) -> tuple[BinaryIO, tuple[int, ...]]:
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ContractError(f"input must be a regular single-link file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    source = os.fdopen(descriptor, "rb")
    after = os.fstat(source.fileno())
    identity = _file_identity(before)
    if _file_identity(after) != identity:
        source.close()
        raise ContractError(f"input identity changed while opening: {path}")
    return source, identity


def _verify_open_identity(
    source: BinaryIO, identity: tuple[int, ...], path: Path
) -> None:
    if _file_identity(os.fstat(source.fileno())) != identity:
        raise ContractError(f"input identity changed while reading: {path}")


def read_verified_bytes(path: Path, expected_sha256: str) -> bytes:
    source, identity = _open_regular_single_link(path)
    with source:
        data = source.read()
        _verify_open_identity(source, identity, path)
    observed = sha256_bytes(data)
    if observed != expected_sha256:
        raise ContractError(f"frozen input hash mismatch for {path}: {observed}")
    return data


def load_verified_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    source, identity = _open_regular_single_link(path)
    with source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise ContractError(f"frozen checkpoint hash mismatch: {observed}")
        source.seek(0)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        _verify_open_identity(source, identity, path)
        source.seek(0)
        second = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            second.update(block)
        if second.hexdigest() != expected_sha256:
            raise ContractError("checkpoint changed while loading")
    if not isinstance(checkpoint, dict):
        raise ContractError("checkpoint root must be an object")
    return checkpoint


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object_bytes(data: bytes, label: str) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ContractError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as error:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ContractError(f"{label} root must be an object")
    _require_plain_json_tree(value, label)
    return value


def _require_normalized_absolute_path(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ContractError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} cannot be resolved: {path}") from error
    if resolved != path:
        raise ContractError(f"{label} must be normalized and contain no symlinks")
    return path


def _directory_record(path: Path, info: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def open_owned_output_directory(path: Path) -> tuple[int, dict[str, Any]]:
    """Open one current-UID mode-0700 directory and retain its inode custody."""
    path = _require_normalized_absolute_path(path, "output directory")
    before = path.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise ContractError("output directory must be current-UID-owned mode 0700")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise ContractError("output directory identity changed while opening")
        after = path.lstat()
        if _file_identity(after) != _file_identity(opened):
            raise ContractError("output directory pathname changed while opening")
        return descriptor, _directory_record(path, opened)
    except BaseException:
        os.close(descriptor)
        raise


def validate_output_directory_fd(
    descriptor: int,
    expected: dict[str, Any],
    *,
    require_path_identity: bool,
) -> os.stat_result:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or _directory_record(Path(expected["path"]), info) != expected
    ):
        raise ContractError("output directory descriptor identity differs")
    if require_path_identity:
        path_info = Path(expected["path"]).lstat()
        if _file_identity(path_info) != _file_identity(info):
            raise ContractError("output directory pathname no longer names held inode")
    return info


def _validate_directory_entry_name(name: str, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ContractError(f"{label} must be one plain directory-entry name")
    return name


def _openat_regular_single_link(
    directory_fd: int, name: str, label: str
) -> tuple[BinaryIO, tuple[int, ...]]:
    _validate_directory_entry_name(name, label)
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ContractError(f"{label} must be a regular single-link file")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    source = os.fdopen(descriptor, "rb")
    opened = os.fstat(source.fileno())
    identity = _file_identity(before)
    if _file_identity(opened) != identity:
        source.close()
        raise ContractError(f"{label} identity changed while opening")
    return source, identity


def read_directory_entry_bytes(
    directory_fd: int, name: str, label: str
) -> tuple[bytes, os.stat_result]:
    source, identity = _openat_regular_single_link(directory_fd, name, label)
    with source:
        data = source.read()
        _verify_open_identity(source, identity, Path(name))
        info = os.fstat(source.fileno())
    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _file_identity(after) != identity:
        raise ContractError(f"{label} pathname changed while reading")
    return data, info


def rename_noreplace_at(
    old_directory_fd: int,
    old_name: str,
    new_directory_fd: int,
    new_name: str,
) -> None:
    """Linux atomic rename with no replacement, relative to held dirfds."""
    _validate_directory_entry_name(old_name, "rename source")
    _validate_directory_entry_name(new_name, "rename destination")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ContractError("renameat2 is unavailable; no-overwrite rename refused")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        old_directory_fd,
        os.fsencode(old_name),
        new_directory_fd,
        os.fsencode(new_name),
        1,
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(new_name)
        raise OSError(error_number, os.strerror(error_number))


def _descriptor_exec_path(descriptor: int) -> str:
    if not sys.platform.startswith("linux"):
        raise ContractError("descriptor execution requires Linux /proc")
    os.fstat(descriptor)
    return f"/proc/self/fd/{descriptor}"


def _run_git(
    git_bin: Path,
    source_root: Path,
    *arguments: str,
    executable_descriptor: int | None = None,
) -> bytes:
    executable = (
        _descriptor_exec_path(executable_descriptor)
        if executable_descriptor is not None
        else str(git_bin)
    )
    process = subprocess.run(
        [
            executable,
            "--no-pager",
            "--no-optional-locks",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(source_root),
            *arguments,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(() if executable_descriptor is None else (executable_descriptor,)),
        env=_sealed_git_environment(),
    )
    if process.returncode:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(f"git {' '.join(arguments)} failed: {stderr}")
    return process.stdout


def _sealed_git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "LC_ALL": "C",
        "LANG": "C",
        "PATH": "",
    }


def _run_exact_command(
    executable: str, *arguments: str, executable_descriptor: int | None = None
) -> bytes:
    execution_path = (
        _descriptor_exec_path(executable_descriptor)
        if executable_descriptor is not None
        else executable
    )
    process = subprocess.run(
        [execution_path, *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(() if executable_descriptor is None else (executable_descriptor,)),
    )
    if process.returncode:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"runtime command failed: {Path(executable).name} {' '.join(arguments)}: "
            f"{stderr}"
        )
    if not process.stdout:
        raise ContractError("runtime command returned empty output")
    return process.stdout


def _parse_scontrol_record(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ContractError("scontrol job output is not UTF-8") from error
    if not text or "\n" in text:
        raise ContractError("scontrol job output must be exactly one nonempty record")
    fields: dict[str, str] = {}
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as error:
        raise ContractError("scontrol job output is not shell-tokenizable") from error
    for token in tokens:
        if "=" not in token:
            raise ContractError("scontrol job output contains a field without equals")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ContractError(
                "scontrol job output contains an invalid duplicate field"
            )
        fields[key] = value
    return fields


def _parse_cluster_name(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("scontrol config output is not UTF-8") from error
    matches = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*ClusterName\s*=\s*(\S+)\s*", line)
        if match:
            matches.append(match.group(1))
    if len(matches) != 1:
        raise ContractError("scontrol config must contain exactly one ClusterName")
    return matches[0]


def _parse_tres_field(value: str, label: str) -> dict[str, str]:
    if not isinstance(value, str) or not value or value == "(null)":
        raise ContractError(f"{label} is empty")
    entries: dict[str, str] = {}
    for item in value.split(","):
        if item.count("=") != 1:
            raise ContractError(f"{label} contains a malformed TRES entry")
        key, amount = item.split("=", 1)
        if (
            re.fullmatch(r"[A-Za-z0-9_./:-]+", key) is None
            or re.fullmatch(r"[0-9]+(?:[KMGTP])?", amount) is None
            or key in entries
        ):
            raise ContractError(f"{label} contains an invalid or duplicate TRES entry")
        entries[key] = amount
    return dict(sorted(entries.items()))


def _validate_h100_tres(entries: dict[str, str], label: str) -> None:
    required = {
        "cpu": str(SLURM_CPUS_PER_TASK),
        "mem": SLURM_MEMORY,
        "node": str(SLURM_NODE_COUNT),
        "gres/gpu": str(SLURM_GPU_COUNT),
        SLURM_TYPED_GPU_TRES: str(SLURM_GPU_COUNT),
    }
    if any(entries.get(key) != amount for key, amount in required.items()):
        raise ContractError(f"{label} differs from the frozen allocation contract")
    gpu_entries = {
        key: amount for key, amount in entries.items() if key.startswith("gres/gpu")
    }
    if gpu_entries != {
        "gres/gpu": str(SLURM_GPU_COUNT),
        SLURM_TYPED_GPU_TRES: str(SLURM_GPU_COUNT),
    }:
        raise ContractError(f"{label} contains an additional or mistyped GPU resource")


def _parse_sacct_record(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ContractError("sacct output is not UTF-8") from error
    lines = [line for line in text.splitlines() if line]
    if len(lines) != 1:
        raise ContractError("sacct must return exactly one base-job record")
    names = (
        "job_id_raw",
        "job_name",
        "partition",
        "state",
        "alloc_cpus",
        "req_memory",
        "time_limit",
        "req_tres",
        "alloc_tres",
        "node_list",
    )
    fields = lines[0].split("|")
    if len(fields) != len(names) or any(not field for field in fields):
        raise ContractError("sacct record has an invalid exact field count")
    return dict(zip(names, fields, strict=True))


def _canonical_single_node_name(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value) is None
        or "[" in value
        or "]" in value
        or "," in value
        or "." in value
    ):
        raise ContractError(f"{label} is not one exact canonical node name")
    return value


def _parse_devices_list(data: bytes) -> tuple[str, ...]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("device-cgroup rules are not ASCII") from error
    rules = []
    for line in text.splitlines():
        if not line:
            continue
        if re.fullmatch(r"[ac] (?:[0-9]+|\*):(?:[0-9]+|\*) rwm", line) is None:
            raise ContractError("device-cgroup rule has an invalid canonical form")
        rules.append(line)
    if not rules or len(rules) != len(set(rules)):
        raise ContractError("device-cgroup rules are empty or duplicated")
    return tuple(sorted(rules))


def _parse_nvidia_smi_gpu(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").strip().splitlines()
    except UnicodeDecodeError as error:
        raise ContractError("nvidia-smi query is not UTF-8") from error
    if len(lines) != 1:
        raise ContractError("exactly one physical GPU must be reported")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 6:
        raise ContractError("nvidia-smi query field count differs")
    index, minor_number, name, gpu_uuid, pci_bus_id, mig_mode = fields
    if (
        re.fullmatch(r"[0-9]+", index) is None
        or re.fullmatch(r"[0-9]+", minor_number) is None
        or name != REQUIRED_CUDA_DEVICE_NAME
        or re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid) is None
        or re.fullmatch(
            r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", pci_bus_id
        )
        is None
        or mig_mode != "Disabled"
    ):
        raise ContractError("nvidia-smi full H100 PCIe identity differs")
    return {
        "index": index,
        "minor_number": minor_number,
        "name": name,
        "uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id.lower(),
        "mig_mode": mig_mode,
    }


def _resolve_single_gpu_selector(selector: str, gpu: dict[str, str], label: str) -> str:
    if not selector or "," in selector or selector.strip() != selector:
        raise ContractError(f"{label} must contain exactly one canonical selector")
    if selector not in {gpu["index"], gpu["uuid"]}:
        raise ContractError(f"{label} does not resolve to the allocated GPU")
    return gpu["uuid"]


def _validate_nvidia_control_device_numbers(
    control_device_numbers: dict[str, tuple[int, int]],
    gpu_major: int,
    gpu_minor: int,
) -> None:
    specs = {
        name: (major_kind, canonical_minor)
        for name, major_kind, canonical_minor in NVIDIA_CONTROL_DEVICE_SPECS
    }
    if set(control_device_numbers) - set(specs):
        raise ContractError("unknown NVIDIA control-device name")
    observed_numbers: set[tuple[int, int]] = set()
    uvm_majors: set[int] = set()
    for name, device_number in control_device_numbers.items():
        if (
            type(device_number) is not tuple
            or len(device_number) != 2
            or any(type(value) is not int or value < 0 for value in device_number)
        ):
            raise ContractError("NVIDIA control-device number is malformed")
        if device_number in observed_numbers:
            raise ContractError("named NVIDIA control devices alias one device number")
        observed_numbers.add(device_number)
        major, minor = device_number
        major_kind, canonical_minor = specs[name]
        if device_number == (gpu_major, gpu_minor) or (
            major == gpu_major and minor != canonical_minor
        ):
            raise ContractError(
                "NVIDIA control device collides with a physical-GPU device number"
            )
        if minor != canonical_minor or (
            major_kind == "physical_major" and major != gpu_major
        ):
            raise ContractError("NVIDIA control device has a noncanonical identity")
        if major_kind == "uvm_major":
            if major == gpu_major:
                raise ContractError(
                    "NVIDIA control device collides with a physical-GPU device number"
                )
            uvm_majors.add(major)
    if len(uvm_majors) > 1:
        raise ContractError("NVIDIA UVM control devices use inconsistent majors")


def _observe_nvidia_control_device_numbers(
    dev_root: Path, gpu_major: int, gpu_minor: int
) -> dict[str, tuple[int, int]]:
    control_device_numbers: dict[str, tuple[int, int]] = {}
    for control_name, _major_kind, _canonical_minor in NVIDIA_CONTROL_DEVICE_SPECS:
        control_path = dev_root / control_name
        try:
            control_info = control_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ContractError(
                f"NVIDIA control path is not inspectable: {control_name}"
            ) from error
        if not stat.S_ISCHR(control_info.st_mode):
            raise ContractError(
                f"NVIDIA control path is not a character device: {control_name}"
            )
        control_device_numbers[control_name] = (
            os.major(control_info.st_rdev),
            os.minor(control_info.st_rdev),
        )
    _validate_nvidia_control_device_numbers(
        control_device_numbers, gpu_major, gpu_minor
    )
    return control_device_numbers


def _nvidia_control_device_records(
    control_device_numbers: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        {"name": name, "major": major, "minor": minor}
        for name, (major, minor) in sorted(control_device_numbers.items())
    ]


def _validate_nvidia_control_device_records(
    records: Any, gpu_major: int, gpu_minor: int
) -> None:
    if type(records) is not list:
        raise ContractError("NVIDIA control-device records are not a list")
    control_device_numbers: dict[str, tuple[int, int]] = {}
    for record in records:
        if type(record) is not dict or set(record) != {"name", "major", "minor"}:
            raise ContractError("NVIDIA control-device record shape differs")
        name = record["name"]
        if type(name) is not str or name in control_device_numbers:
            raise ContractError("NVIDIA control-device record name differs")
        control_device_numbers[name] = (record["major"], record["minor"])
    if records != _nvidia_control_device_records(control_device_numbers):
        raise ContractError("NVIDIA control-device records are not canonical")
    _validate_nvidia_control_device_numbers(
        control_device_numbers, gpu_major, gpu_minor
    )


def observe_gpu_allocation_binding(
    nvidia_smi: str,
    job_id: str,
    *,
    nvidia_smi_descriptor: int | None = None,
    proc_self_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_devices_root: Path = Path("/sys/fs/cgroup/devices"),
    dev_root: Path = Path("/dev"),
    pci_devices_root: Path = Path("/sys/bus/pci/devices"),
) -> dict[str, Any]:
    query = _run_exact_command(
        nvidia_smi,
        "--query-gpu=index,minor_number,name,uuid,pci.bus_id,mig.mode.current",
        "--format=csv,noheader,nounits",
        executable_descriptor=nvidia_smi_descriptor,
    )
    listing = _run_exact_command(
        nvidia_smi, "-L", executable_descriptor=nvidia_smi_descriptor
    )
    gpu = _parse_nvidia_smi_gpu(query)
    expected_listing = (
        f"GPU {gpu['index']}: {gpu['name']} (UUID: {gpu['uuid']})\n".encode()
    )
    if listing != expected_listing or b"MIG" in listing:
        raise ContractError("nvidia-smi reports MIG or a noncanonical GPU inventory")

    pci_bdf = gpu["pci_bus_id"]
    if pci_bdf.startswith("00000000:"):
        pci_bdf = "0000:" + pci_bdf.removeprefix("00000000:")
    pci_path = pci_devices_root / pci_bdf
    pci_fields = {}
    for name in ("vendor", "device", "class"):
        try:
            value = (pci_path / name).read_text(encoding="ascii").strip().lower()
        except OSError as error:
            raise ContractError(
                f"PCI sysfs identity is unavailable at {name}"
            ) from error
        if re.fullmatch(r"0x[0-9a-f]+", value) is None:
            raise ContractError(f"PCI sysfs identity is malformed at {name}")
        pci_fields[name] = value
    if pci_fields["vendor"] != "0x10de" or not pci_fields["class"].startswith("0x03"):
        raise ContractError("PCI sysfs device is not an NVIDIA display/compute device")
    pci_identity_sha256 = sha256_bytes(
        stable_json_bytes({"bdf": pci_bdf, **pci_fields})
    )

    cgroup_data = proc_self_cgroup.read_bytes()
    device_entries = []
    for line in cgroup_data.decode("ascii").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            raise ContractError("/proc/self/cgroup contains a malformed record")
        if "devices" in parts[1].split(","):
            device_entries.append(parts[2])
    if len(device_entries) != 1:
        raise ContractError(
            "one inspectable cgroup-v1 devices controller is required; cgroup-v2 "
            "device-BPF evidence is not silently inferred"
        )
    cgroup_path = device_entries[0]
    components = [part for part in cgroup_path.split("/") if part]
    if f"job_{job_id}" not in components:
        raise ContractError("devices cgroup is not exactly nested under this Slurm job")
    devices_path = cgroup_devices_root.joinpath(*components, "devices.list")
    devices_data = devices_path.read_bytes()
    rules = _parse_devices_list(devices_data)

    nvidia_smi_minor = int(gpu["minor_number"])
    gpu_device = dev_root / f"nvidia{nvidia_smi_minor}"
    device_info = gpu_device.stat()
    if not stat.S_ISCHR(device_info.st_mode):
        raise ContractError("allocated NVIDIA GPU path is not a character device")
    gpu_major = os.major(device_info.st_rdev)
    gpu_minor = os.minor(device_info.st_rdev)
    if gpu_minor != nvidia_smi_minor:
        raise ContractError(
            "nvidia-smi minor_number differs from the allocated GPU device minor"
        )
    if f"c {gpu_major}:{gpu_minor} rwm" not in rules:
        raise ContractError(
            "allocated GPU major/minor is absent from job device cgroup"
        )
    if (
        any(rule.startswith("a ") for rule in rules)
        or any(rule.startswith("c *:") for rule in rules)
        or f"c {gpu_major}:* rwm" in rules
    ):
        raise ContractError("device cgroup grants an untyped GPU wildcard")

    named_control_device_numbers = _observe_nvidia_control_device_numbers(
        dev_root, gpu_major, gpu_minor
    )
    control_device_numbers = set(named_control_device_numbers.values())

    concrete_physical_permissions = set()
    for rule in rules:
        match = re.fullmatch(r"c ([0-9]+):([0-9]+) rwm", rule)
        if match is None:
            continue
        device_number = (int(match.group(1)), int(match.group(2)))
        if device_number[0] != gpu_major or device_number in control_device_numbers:
            continue
        concrete_physical_permissions.add(device_number)
    if concrete_physical_permissions != {(gpu_major, gpu_minor)}:
        raise ContractError("device cgroup grants an extra concrete physical-GPU minor")

    cuda_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    slurm_selector = os.environ.get("SLURM_JOB_GPUS", "")
    cuda_uuid = _resolve_single_gpu_selector(cuda_selector, gpu, "CUDA_VISIBLE_DEVICES")
    slurm_uuid = _resolve_single_gpu_selector(slurm_selector, gpu, "SLURM_JOB_GPUS")
    if cuda_uuid != slurm_uuid:
        raise ContractError("CUDA and Slurm GPU selectors resolve differently")
    return {
        "cgroup_version": "v1_devices_controller",
        "cgroup_path": cgroup_path,
        "devices_list_path": str(devices_path),
        "devices_list_sha256": sha256_bytes(devices_data),
        "allowed_device_rules": list(rules),
        "allocated_gpu_device": str(gpu_device),
        "pci_bus_id": gpu["pci_bus_id"],
        "pci_sysfs_path": str(pci_path),
        "pci_vendor_id": pci_fields["vendor"],
        "pci_device_id": pci_fields["device"],
        "pci_class_id": pci_fields["class"],
        "pci_identity_sha256": pci_identity_sha256,
        "gpu_uuid": gpu["uuid"],
        "gpu_name": gpu["name"],
        "nvidia_smi_index": int(gpu["index"]),
        "nvidia_smi_minor_number": nvidia_smi_minor,
        "gpu_minor": gpu_minor,
        "gpu_major": gpu_major,
        "nvidia_control_devices": _nvidia_control_device_records(
            named_control_device_numbers
        ),
        "concrete_physical_gpu_permissions": [
            {"major": major, "minor": minor}
            for major, minor in sorted(concrete_physical_permissions)
        ],
        "mig_mode": gpu["mig_mode"],
        "mig_devices_present": False,
        "nvidia_smi_query_sha256": sha256_bytes(query),
        "nvidia_smi_list_sha256": sha256_bytes(listing),
        "cuda_visible_devices": cuda_selector,
        "slurm_job_gpus": slurm_selector,
        "selector_mapping": {
            "cuda_uuid": cuda_uuid,
            "slurm_uuid": slurm_uuid,
        },
    }


def _validate_slurm_identity(value: dict[str, Any]) -> None:
    _require_exact_keys(value, SLURM_IDENTITY_KEYS, "Slurm identity")
    string_fields = (
        "job_id",
        "job_name",
        "job_state",
        "cluster_name",
        "user_name",
        "command",
        "command_sha256",
        "batch_host",
        "node_list",
        "observed_hostname",
        "partition",
        "min_memory_node",
        "time_limit",
        "gres",
        "gpu_type",
        "job_record_sha256",
        "cluster_config_sha256",
    )
    integer_fields = (
        "user_uid",
        "batch_flag",
        "num_nodes",
        "num_cpus",
        "num_tasks",
        "cpus_per_task",
        "min_cpus_node",
        "memory_bytes",
        "time_limit_seconds",
        "requeue",
        "gpu_count",
    )
    if any(not isinstance(value[key], str) for key in string_fields) or any(
        type(value[key]) is not int for key in integer_fields
    ):
        raise ContractError("parsed Slurm allocation field types differ")
    if (
        re.fullmatch(r"[1-9][0-9]*", value["job_id"]) is None
        or value["job_name"] != "shohin-r12-dws-eos-dev"
        or value["job_state"] != "RUNNING"
        or not isinstance(value["cluster_name"], str)
        or not value["cluster_name"]
        or value["user_uid"] != os.getuid()
        or value["batch_flag"] != 1
        or value["partition"] != SLURM_PARTITION
        or value["num_nodes"] != SLURM_NODE_COUNT
        or value["num_cpus"] != SLURM_CPUS_PER_TASK
        or value["num_tasks"] != SLURM_TASK_COUNT
        or value["cpus_per_task"] != SLURM_CPUS_PER_TASK
        or value["min_cpus_node"] != SLURM_CPUS_PER_TASK
        or value["min_memory_node"] != SLURM_MEMORY
        or value["memory_bytes"] != SLURM_MEMORY_BYTES
        or value["time_limit"] != SLURM_TIME_LIMIT
        or value["time_limit_seconds"] != SLURM_TIME_LIMIT_SECONDS
        or value["requeue"] != 0
        or value["gres"] != SLURM_GRES
        or value["gpu_type"] != SLURM_GPU_TYPE
        or value["gpu_count"] != SLURM_GPU_COUNT
        or any(
            not isinstance(value[key], str) or not value[key]
            for key in (
                "user_name",
                "command",
                "batch_host",
                "node_list",
                "observed_hostname",
            )
        )
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
            for key in (
                "command_sha256",
                "job_record_sha256",
                "cluster_config_sha256",
            )
        )
    ):
        raise ContractError("parsed Slurm allocation identity differs")
    canonical_node = _canonical_single_node_name(value["node_list"], "Slurm NodeList")
    if any(
        _canonical_single_node_name(value[key], f"Slurm {key}") != canonical_node
        for key in ("batch_host", "observed_hostname")
    ):
        raise ContractError("Slurm node identity cross-binding differs")
    for key in ("req_tres", "alloc_tres"):
        entries = value[key]
        if not isinstance(entries, dict) or any(
            not isinstance(name, str) or not isinstance(amount, str)
            for name, amount in entries.items()
        ):
            raise ContractError(f"parsed Slurm {key} identity differs")
        _validate_h100_tres(entries, key)
    if value["tres_per_node"] != {SLURM_TYPED_GPU_TRES: str(SLURM_GPU_COUNT)}:
        raise ContractError("parsed Slurm per-node GPU TRES differs")
    sacct = value["sacct_identity"]
    _require_exact_keys(sacct, SACCT_IDENTITY_KEYS, "Slurm sacct identity")
    if (
        sacct["job_id_raw"] != value["job_id"]
        or sacct["job_name"] != value["job_name"]
        or sacct["partition"] != value["partition"]
        or sacct["state"] != "RUNNING"
        or sacct["alloc_cpus"] != value["num_cpus"]
        or sacct["req_memory"] != SLURM_MEMORY
        or sacct["time_limit"] != SLURM_TIME_LIMIT
        or sacct["req_tres"] != value["req_tres"]
        or sacct["alloc_tres"] != value["alloc_tres"]
        or _canonical_single_node_name(sacct["node_list"], "sacct NodeList")
        != canonical_node
        or re.fullmatch(r"[0-9a-f]{64}", sacct["record_sha256"]) is None
    ):
        raise ContractError("sacct identity differs from scontrol allocation")
    gpu = value["gpu_binding"]
    _require_exact_keys(gpu, GPU_BINDING_KEYS, "Slurm GPU binding")
    if (
        gpu["cgroup_version"] != "v1_devices_controller"
        or type(gpu["cgroup_path"]) is not str
        or f"job_{value['job_id']}" not in gpu["cgroup_path"].split("/")
        or gpu["gpu_name"] != REQUIRED_CUDA_DEVICE_NAME
        or gpu["pci_vendor_id"] != "0x10de"
        or not gpu["pci_class_id"].startswith("0x03")
        or type(gpu["pci_sysfs_path"]) is not str
        or not Path(gpu["pci_sysfs_path"]).is_absolute()
        or gpu["mig_mode"] != "Disabled"
        or gpu["mig_devices_present"] is not False
        or type(gpu["gpu_major"]) is not int
        or type(gpu["gpu_minor"]) is not int
        or type(gpu["nvidia_smi_index"]) is not int
        or type(gpu["nvidia_smi_minor_number"]) is not int
        or min(
            gpu["gpu_major"],
            gpu["gpu_minor"],
            gpu["nvidia_smi_index"],
            gpu["nvidia_smi_minor_number"],
        )
        < 0
        or gpu["nvidia_smi_minor_number"] != gpu["gpu_minor"]
        or not _strict_equal(
            gpu["concrete_physical_gpu_permissions"],
            [{"major": gpu["gpu_major"], "minor": gpu["gpu_minor"]}],
        )
        or type(gpu["allowed_device_rules"]) is not list
        or not gpu["allowed_device_rules"]
        or gpu["selector_mapping"]
        != {"cuda_uuid": gpu["gpu_uuid"], "slurm_uuid": gpu["gpu_uuid"]}
        or any(
            re.fullmatch(r"[0-9a-f]{64}", gpu[key]) is None
            for key in (
                "devices_list_sha256",
                "nvidia_smi_query_sha256",
                "nvidia_smi_list_sha256",
                "pci_identity_sha256",
            )
        )
    ):
        raise ContractError("Slurm device-cgroup/physical-GPU binding differs")
    _validate_nvidia_control_device_records(
        gpu["nvidia_control_devices"], gpu["gpu_major"], gpu["gpu_minor"]
    )


def observe_slurm_identity(
    scontrol: str,
    sacct: str,
    nvidia_smi: str,
    job_id: str,
    *,
    scontrol_descriptor: int | None = None,
    sacct_descriptor: int | None = None,
    nvidia_smi_descriptor: int | None = None,
) -> dict[str, Any]:
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ContractError("Slurm job ID selector is not a positive decimal integer")
    job_output = _run_exact_command(
        scontrol,
        "show",
        "job",
        "-o",
        job_id,
        executable_descriptor=scontrol_descriptor,
    )
    fields = _parse_scontrol_record(job_output)
    required = {
        "JobId",
        "JobName",
        "JobState",
        "UserId",
        "BatchFlag",
        "Command",
        "BatchHost",
        "NodeList",
        "Partition",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "CPUs/Task",
        "MinCPUsNode",
        "MinMemoryNode",
        "TimeLimit",
        "Requeue",
        "ReqTRES",
        "AllocTRES",
        "TresPerNode",
        "Gres",
    }
    if not required.issubset(fields):
        raise ContractError("scontrol job record lacks a required exact identity field")
    user_match = re.fullmatch(r"([^()]+)\(([0-9]+)\)", fields["UserId"])
    if user_match is None:
        raise ContractError("scontrol UserId field has an invalid form")
    parsed_uid = int(user_match.group(2))
    if (
        fields["JobId"] != job_id
        or fields["JobName"] != "shohin-r12-dws-eos-dev"
        or fields["JobState"] != "RUNNING"
        or fields["BatchFlag"] != "1"
        or parsed_uid != os.getuid()
        or not fields["BatchHost"]
    ):
        raise ContractError(
            "parsed Slurm job identity differs from the frozen contract"
        )
    command_path = _require_normalized_absolute_path(
        Path(fields["Command"]), "parsed Slurm command"
    )
    node_list = _canonical_single_node_name(fields["NodeList"], "scontrol NodeList")
    batch_host = _canonical_single_node_name(fields["BatchHost"], "scontrol BatchHost")
    observed_hostname = _canonical_single_node_name(
        socket.gethostname(), "observed current hostname"
    )
    if batch_host != node_list or observed_hostname != node_list:
        raise ContractError("scontrol and observed node identities differ")
    decimal_fields = {}
    for key in (
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "CPUs/Task",
        "MinCPUsNode",
        "Requeue",
    ):
        if re.fullmatch(r"[0-9]+", fields[key]) is None:
            raise ContractError(f"parsed Slurm decimal field is invalid: {key}")
        decimal_fields[key] = int(fields[key])
    req_tres = _parse_tres_field(fields["ReqTRES"], "ReqTRES")
    alloc_tres = _parse_tres_field(fields["AllocTRES"], "AllocTRES")
    tres_per_node = _parse_tres_field(fields["TresPerNode"], "TresPerNode")
    if fields["Gres"] != SLURM_GRES:
        raise ContractError("scontrol Gres differs from the typed H100 contract")
    config_output = _run_exact_command(
        scontrol, "show", "config", executable_descriptor=scontrol_descriptor
    )
    sacct_output = _run_exact_command(
        sacct,
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobIDRaw,JobName,Partition,State,AllocCPUS,ReqMem,Timelimit,ReqTRES,AllocTRES,NodeList",
        executable_descriptor=sacct_descriptor,
    )
    sacct_fields = _parse_sacct_record(sacct_output)
    if re.fullmatch(r"[0-9]+", sacct_fields["alloc_cpus"]) is None:
        raise ContractError("sacct AllocCPUS is not decimal")
    sacct_identity = {
        **sacct_fields,
        "alloc_cpus": int(sacct_fields["alloc_cpus"]),
        "req_tres": _parse_tres_field(sacct_fields["req_tres"], "sacct ReqTRES"),
        "alloc_tres": _parse_tres_field(sacct_fields["alloc_tres"], "sacct AllocTRES"),
        "record_sha256": sha256_bytes(sacct_output),
    }
    gpu_binding = observe_gpu_allocation_binding(
        nvidia_smi,
        job_id,
        nvidia_smi_descriptor=nvidia_smi_descriptor,
    )
    observed = {
        "job_id": fields["JobId"],
        "job_name": fields["JobName"],
        "job_state": fields["JobState"],
        "cluster_name": _parse_cluster_name(config_output),
        "user_name": user_match.group(1),
        "user_uid": parsed_uid,
        "batch_flag": 1,
        "command": str(command_path),
        "command_sha256": sha256_regular_file(command_path),
        "batch_host": batch_host,
        "node_list": node_list,
        "observed_hostname": observed_hostname,
        "partition": fields["Partition"],
        "num_nodes": decimal_fields["NumNodes"],
        "num_cpus": decimal_fields["NumCPUs"],
        "num_tasks": decimal_fields["NumTasks"],
        "cpus_per_task": decimal_fields["CPUs/Task"],
        "min_cpus_node": decimal_fields["MinCPUsNode"],
        "min_memory_node": fields["MinMemoryNode"],
        "memory_bytes": SLURM_MEMORY_BYTES,
        "time_limit": fields["TimeLimit"],
        "time_limit_seconds": SLURM_TIME_LIMIT_SECONDS,
        "requeue": decimal_fields["Requeue"],
        "gres": SLURM_GRES,
        "gpu_type": SLURM_GPU_TYPE,
        "gpu_count": SLURM_GPU_COUNT,
        "req_tres": req_tres,
        "alloc_tres": alloc_tres,
        "tres_per_node": tres_per_node,
        "job_record_sha256": sha256_bytes(job_output),
        "cluster_config_sha256": sha256_bytes(config_output),
        "sacct_identity": sacct_identity,
        "gpu_binding": gpu_binding,
    }
    _validate_slurm_identity(observed)
    return observed


def _executable_identity(
    path: Path,
    version_arguments: tuple[str, ...],
    descriptor: int | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    path = _require_normalized_absolute_path(path, "runtime executable")
    if not os.access(path, os.X_OK):
        raise ContractError(f"runtime executable is not executable: {path}")
    pass_fds: tuple[int, ...] = ()
    execution_path = str(path)
    held_source: BinaryIO | None = None
    if descriptor is not None:
        descriptor_info = os.fstat(descriptor)
        descriptor_bytes, descriptor_info = _read_exact_descriptor_bytes(
            descriptor, f"held executable {path}", expected_info=descriptor_info
        )
        if _file_identity(descriptor_info) != _file_identity(path.lstat()):
            raise ContractError(f"held executable descriptor differs for {path}")
        execution_path = _descriptor_exec_path(descriptor)
        pass_fds = (descriptor,)
    else:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"runtime executable is not regular: {path}")
        held_source = os.fdopen(
            os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb"
        )
        held_identity = _file_identity(before)
        if _file_identity(os.fstat(held_source.fileno())) != held_identity:
            held_source.close()
            raise ContractError(f"runtime executable changed while opening: {path}")
        descriptor_bytes = held_source.read()
        _verify_open_identity(held_source, held_identity, path)
        descriptor_info = os.fstat(held_source.fileno())
    before_sha256 = sha256_bytes(descriptor_bytes)
    try:
        process = subprocess.run(
            [execution_path, *version_arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
            env=environment,
        )
        if process.returncode:
            raise ContractError(f"runtime version command failed for {path}")
        version = (process.stdout or process.stderr).decode("utf-8").strip()
        if not version:
            raise ContractError(f"runtime version is empty for {path}")
        held_descriptor = descriptor if descriptor is not None else held_source.fileno()
        _read_exact_descriptor_bytes(
            held_descriptor,
            f"held executable {path}",
            expected_info=descriptor_info,
            expected_payload=descriptor_bytes,
        )
        if _file_identity(path.lstat()) != _file_identity(descriptor_info):
            raise ContractError(
                f"runtime executable pathname changed during identity probe: {path}"
            )
    finally:
        if held_source is not None:
            held_source.close()
    return {
        "path": str(path),
        "sha256": before_sha256,
        "version": version,
    }


def _strict_distribution_relative_path(
    value: str, label: str, *, allow_leading_parents: bool = True
) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ContractError(f"{label} path is not canonical POSIX-relative")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts:
        raise ContractError(f"{label} path is not canonical POSIX-relative")
    reached_file_component = False
    for part in relative.parts:
        if part in {"", "."}:
            raise ContractError(f"{label} path is not canonical POSIX-relative")
        if part == "..":
            if not allow_leading_parents or reached_file_component:
                raise ContractError(f"{label} path is not canonical POSIX-relative")
        else:
            reached_file_component = True
    if not reached_file_component:
        raise ContractError(f"{label} path is not canonical POSIX-relative")
    canonical = relative.as_posix()
    if canonical != value:
        raise ContractError(f"{label} path is not canonical POSIX-relative")
    return canonical


def _locate_distribution_file(
    distribution: Any,
    package_path: Any,
    relative_path: str,
    distribution_root: Path,
    installation_root: Path,
    label: str,
) -> Path:
    relative = PurePosixPath(relative_path)
    expected = Path(os.path.normpath(str(distribution_root.joinpath(*relative.parts))))
    try:
        expected.relative_to(installation_root)
    except ValueError as error:
        raise ContractError(f"{label} escapes the Python installation root") from error
    located = Path(os.path.normpath(str(distribution.locate_file(package_path))))
    if located != expected:
        raise ContractError(f"{label} location differs from RECORD")
    return _require_normalized_absolute_path(expected, label)


def _regular_file_closure_record(
    path: Path, *, relative_path: str | None, label: str
) -> tuple[dict[str, Any], bytes]:
    path = _require_normalized_absolute_path(path, label)
    before = path.lstat()
    mode = stat.S_IMODE(before.st_mode)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 0
        or mode & 0o022
        or mode & ~0o777
    ):
        raise ContractError(
            f"{label} must be one single-link regular file without group/other write bits"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise ContractError(f"{label} changed while opening")
        payload = os.pread(descriptor, opened.st_size + 1, 0)
        if len(payload) != opened.st_size:
            raise ContractError(f"{label} changed while reading")
        after_descriptor = os.fstat(descriptor)
        after_path = path.lstat()
        if _file_identity(after_descriptor) != _file_identity(opened) or _file_identity(
            after_path
        ) != _file_identity(opened):
            raise ContractError(f"{label} identity changed while hashing")
    finally:
        os.close(descriptor)
    record = {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "byte_count": len(payload),
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "mode": mode,
        "nlink": before.st_nlink,
    }
    if relative_path is not None:
        record = {"relative_path": relative_path, **record}
    return record, payload


def _validate_record_digest(
    relative_path: str,
    hash_specification: str,
    size_specification: str,
    file_record: dict[str, Any],
) -> None:
    if hash_specification:
        if not hash_specification.startswith("sha256="):
            raise ContractError(f"RECORD hash algorithm differs at {relative_path}")
        encoded = hash_specification.removeprefix("sha256=")
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
            raise ContractError(f"RECORD SHA-256 encoding differs at {relative_path}")
        try:
            declared = base64.urlsafe_b64decode(encoded + "=")
        except ValueError as error:
            raise ContractError(
                f"RECORD SHA-256 encoding differs at {relative_path}"
            ) from error
        if len(declared) != 32 or declared.hex() != file_record["sha256"]:
            raise ContractError(f"RECORD SHA-256 differs at {relative_path}")
    if size_specification:
        if re.fullmatch(r"0|[1-9][0-9]*", size_specification) is None:
            raise ContractError(f"RECORD size encoding differs at {relative_path}")
        if int(size_specification) != file_record["byte_count"]:
            raise ContractError(f"RECORD size differs at {relative_path}")


def _distribution_identity(
    distribution_name: str, module_relative_path: str
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError(
            f"required package is absent: {distribution_name}"
        ) from error
    metadata_files = distribution.files
    if metadata_files is None:
        raise ContractError(
            f"package has no installed-file manifest: {distribution_name}"
        )
    distribution_root = _require_normalized_absolute_path(
        Path(distribution.locate_file("")), f"{distribution_name} distribution root"
    )
    installation_root = Path(sys.prefix).resolve(strict=True)
    if not distribution_root.is_dir():
        raise ContractError(f"package root is not a directory: {distribution_name}")
    try:
        distribution_root.relative_to(installation_root)
    except ValueError as error:
        raise ContractError(
            f"package root escapes the Python installation: {distribution_name}"
        ) from error
    metadata_by_name: dict[str, Any] = {}
    for package_path in metadata_files:
        relative_path = _strict_distribution_relative_path(
            str(package_path), f"{distribution_name} metadata"
        )
        if relative_path in metadata_by_name:
            raise ContractError(
                f"package manifest has duplicate path: {distribution_name}"
            )
        metadata_by_name[relative_path] = package_path
    module_relative_path = _strict_distribution_relative_path(
        module_relative_path,
        f"{distribution_name} module",
        allow_leading_parents=False,
    )
    record_names = [
        name for name in metadata_by_name if name.endswith(".dist-info/RECORD")
    ]
    if module_relative_path not in metadata_by_name or len(record_names) != 1:
        raise ContractError(f"package identity files differ: {distribution_name}")
    record_relative_path = record_names[0]
    record_path = _locate_distribution_file(
        distribution,
        metadata_by_name[record_relative_path],
        record_relative_path,
        distribution_root,
        installation_root,
        f"{distribution_name} RECORD",
    )
    _record_file, record_payload = _regular_file_closure_record(
        record_path,
        relative_path=record_relative_path,
        label=f"{distribution_name} RECORD",
    )
    try:
        record_rows = list(csv.reader(io.StringIO(record_payload.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ContractError(
            f"package RECORD is not strict CSV: {distribution_name}"
        ) from error
    record_by_name: dict[str, tuple[str, str]] = {}
    for row in record_rows:
        if len(row) != 3:
            raise ContractError(
                f"package RECORD row width differs: {distribution_name}"
            )
        relative_path = _strict_distribution_relative_path(
            row[0], f"{distribution_name} RECORD"
        )
        if relative_path in record_by_name:
            raise ContractError(
                f"package RECORD has duplicate path: {distribution_name}"
            )
        record_by_name[relative_path] = (row[1], row[2])
    if set(record_by_name) != set(metadata_by_name):
        raise ContractError(
            f"package RECORD/metadata path closure differs: {distribution_name}"
        )

    file_manifest: list[dict[str, Any]] = []
    for relative_path in sorted(record_by_name):
        located = _locate_distribution_file(
            distribution,
            metadata_by_name[relative_path],
            relative_path,
            distribution_root,
            installation_root,
            f"{distribution_name} installed file",
        )
        file_record, _payload = _regular_file_closure_record(
            located,
            relative_path=relative_path,
            label=f"{distribution_name} installed file",
        )
        _validate_record_digest(
            relative_path, *record_by_name[relative_path], file_record
        )
        file_manifest.append(file_record)
    closure_payload = stable_json_bytes(
        {
            "distribution_name": distribution_name,
            "distribution_root": str(distribution_root),
            "installation_root": str(installation_root),
            "files": file_manifest,
            "version": distribution.version,
        }
    )
    module_path = next(
        entry["path"]
        for entry in file_manifest
        if entry["relative_path"] == module_relative_path
    )
    return {
        "distribution_name": distribution_name,
        "version": distribution.version,
        "distribution_root": str(distribution_root),
        "installation_root": str(installation_root),
        "module_path": module_path,
        "record_path": str(record_path),
        "file_count": len(file_manifest),
        "files": file_manifest,
        "closure_sha256": sha256_bytes(closure_payload),
    }


def _rehash_manifest_distribution(expected: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        expected, DISTRIBUTION_IDENTITY_KEYS, "manifest package distribution"
    )
    if not isinstance(expected["files"], list) or expected["file_count"] != len(
        expected["files"]
    ):
        raise ContractError("manifest package file count differs")
    observed_files = []
    for expected_file in expected["files"]:
        _require_exact_keys(
            expected_file, DISTRIBUTION_FILE_KEYS, "manifest package file"
        )
        observed, _payload = _regular_file_closure_record(
            Path(expected_file["path"]),
            relative_path=expected_file["relative_path"],
            label="manifest package file",
        )
        if observed != expected_file:
            raise ContractError("manifest package file identity or bytes differ")
        observed_files.append(observed)
    observed = {**expected, "files": observed_files}
    closure_payload = stable_json_bytes(
        {
            "distribution_name": observed["distribution_name"],
            "distribution_root": observed["distribution_root"],
            "installation_root": observed["installation_root"],
            "files": observed_files,
            "version": observed["version"],
        }
    )
    if sha256_bytes(closure_payload) != expected["closure_sha256"]:
        raise ContractError("manifest package closure digest differs")
    return observed


def _python_component_record(
    reference: str, *, require_immutable: bool = True
) -> dict[str, Any]:
    if not isinstance(reference, str) or not reference:
        raise ContractError("pre-authorization Python component reference differs")
    descriptor_match = re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", reference)
    if descriptor_match is not None and sys.platform.startswith("linux"):
        descriptor = int(descriptor_match.group(1))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0 or info.st_size <= 0:
            raise ContractError("pre-authorization Python descriptor component differs")
        payload = os.pread(descriptor, info.st_size + 1, 0)
        if len(payload) != info.st_size:
            raise ContractError("pre-authorization Python descriptor component changed")
        seals = (
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
            if hasattr(fcntl, "F_GET_SEALS")
            else None
        )
        return {
            "reference": reference,
            "kind": "anonymous_descriptor",
            "sha256": sha256_bytes(payload),
            "byte_count": len(payload),
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
            "seals": seals,
        }
    path = Path(reference)
    if not path.is_absolute():
        raise ContractError("pre-authorization Python component is not absolute")
    if require_immutable:
        record, _payload = _regular_file_closure_record(
            path,
            relative_path=None,
            label="pre-authorization Python component",
        )
    else:
        path = _require_normalized_absolute_path(path, "observed test Python component")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("observed test Python component is not regular")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(before):
                raise ContractError("observed test Python component changed")
            payload = os.pread(descriptor, opened.st_size + 1, 0)
            if len(payload) != opened.st_size:
                raise ContractError("observed test Python component changed")
        finally:
            os.close(descriptor)
        record = {
            "path": str(path),
            "sha256": sha256_bytes(payload),
            "byte_count": len(payload),
            "device": before.st_dev,
            "inode": before.st_ino,
            "uid": before.st_uid,
            "mode": stat.S_IMODE(before.st_mode),
            "nlink": before.st_nlink,
        }
    return {
        "reference": reference,
        "kind": "regular_file",
        **record,
        "seals": None,
    }


def _python_search_path_record(
    value: str, *, require_immutable: bool = True
) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ContractError("isolated Python search path is not absolute")
    if not require_immutable:
        path = Path(value)
        if path.is_absolute() and path.exists():
            info = path.stat()
            return {
                "path": value,
                "kind": "observed_unsealed_path",
                "device": info.st_dev,
                "inode": info.st_ino,
                "uid": info.st_uid,
                "mode": stat.S_IMODE(info.st_mode),
                "ancestor": None,
                "sha256": None,
                "byte_count": None,
            }
        return {
            "path": value,
            "kind": "observed_unsealed_entry",
            "device": 0,
            "inode": 0,
            "uid": os.getuid(),
            "mode": 0,
            "ancestor": None,
            "sha256": None,
            "byte_count": None,
        }
    if not Path(value).is_absolute():
        raise ContractError("isolated Python search path is not absolute")
    path = Path(value)
    try:
        info = path.lstat()
    except FileNotFoundError:
        ancestor = path.parent
        while not ancestor.exists():
            if ancestor == ancestor.parent:
                raise ContractError("isolated Python search path has no ancestor")
            ancestor = ancestor.parent
        resolved_ancestor = ancestor.resolve(strict=True)
        ancestor_info = resolved_ancestor.stat()
        if not stat.S_ISDIR(ancestor_info.st_mode) or ancestor_info.st_mode & 0o022:
            raise ContractError(
                "absent isolated Python search path has a writable ancestor"
            )
        return {
            "path": value,
            "kind": "absent_under_nonwritable_ancestor",
            "device": ancestor_info.st_dev,
            "inode": ancestor_info.st_ino,
            "uid": ancestor_info.st_uid,
            "mode": stat.S_IMODE(ancestor_info.st_mode),
            "ancestor": str(resolved_ancestor),
            "sha256": None,
            "byte_count": None,
        }
    if stat.S_ISDIR(info.st_mode):
        resolved = path.resolve(strict=True)
        if resolved != path or info.st_mode & 0o022:
            raise ContractError(
                "isolated Python search directory is mutable or aliased"
            )
        return {
            "path": value,
            "kind": "nonwritable_directory",
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "mode": stat.S_IMODE(info.st_mode),
            "ancestor": None,
            "sha256": None,
            "byte_count": None,
        }
    if stat.S_ISREG(info.st_mode):
        record, payload = _regular_file_closure_record(
            path, relative_path=None, label="isolated Python archive search path"
        )
        return {
            "path": value,
            "kind": "verified_archive",
            "device": record["device"],
            "inode": record["inode"],
            "uid": record["uid"],
            "mode": record["mode"],
            "ancestor": None,
            "sha256": record["sha256"],
            "byte_count": len(payload),
        }
    raise ContractError("isolated Python search path has an unsupported type")


def _preauthorization_python_runtime_identity(
    *, require_sealed_startup: bool
) -> dict[str, Any]:
    flags = {
        "isolated": bool(sys.flags.isolated),
        "no_site": bool(sys.flags.no_site),
        "no_user_site": bool(sys.flags.no_user_site),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
    }
    startup_environment = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "PYTHONINSPECT",
            "PYTHONWARNINGS",
        )
    }
    site_modules = sorted(
        name
        for name in ("site", "sitecustomize", "usercustomize")
        if name in sys.modules
    )
    module_origins = []
    component_references: set[str] = set()
    for name in sorted(sys.modules):
        module = sys.modules[name]
        if module is None:
            module_origins.append(
                {"name": name, "origin": None, "file": None, "cached": None}
            )
            continue
        specification = getattr(module, "__spec__", None)
        origin = getattr(specification, "origin", None)
        module_file = getattr(module, "__file__", None)
        cached = getattr(module, "__cached__", None)
        if isinstance(cached, str) and not Path(cached).exists():
            cached = None
        for field, reference in (
            ("origin", origin),
            ("file", module_file),
            ("cached", cached),
        ):
            if not isinstance(reference, str) or not reference:
                continue
            if reference in {"built-in", "frozen"} or reference.startswith("<"):
                continue
            if reference.startswith("sealed-memfd:"):
                continue
            if field == "cached" and not Path(reference).exists():
                continue
            if not require_sealed_startup and not Path(reference).is_absolute():
                continue
            component_references.add(reference)
        module_origins.append(
            {
                "name": name,
                "origin": origin if isinstance(origin, str) else None,
                "file": module_file if isinstance(module_file, str) else None,
                "cached": cached if isinstance(cached, str) else None,
            }
        )
    components = [
        _python_component_record(reference, require_immutable=require_sealed_startup)
        for reference in sorted(component_references)
    ]
    search_path = [
        _python_search_path_record(value, require_immutable=require_sealed_startup)
        for value in sys.path
    ]
    identity = {
        "mode": PYTHON_STARTUP_MODE
        if require_sealed_startup
        else "observed_unsealed_test_runtime",
        "flags": flags,
        "startup_environment": startup_environment,
        "site_modules_loaded": site_modules,
        "processed_pth_files": [],
        "module_origins": module_origins,
        "components": components,
        "search_path": search_path,
    }
    if require_sealed_startup:
        required_flags = {
            "isolated": True,
            "no_site": True,
            "no_user_site": True,
            "ignore_environment": True,
            "dont_write_bytecode": True,
        }
        if any(
            flags[name] is not expected for name, expected in required_flags.items()
        ):
            raise ContractError("Python did not start through -I -S -B")
        if site_modules or any(startup_environment.values()):
            raise ContractError("Python site/startup customization surface is active")
        if any(
            "site-packages" in record["path"] or "dist-packages" in record["path"]
            for record in search_path
        ):
            raise ContractError("site-package path was active before authorization")
    identity["closure_sha256"] = sha256_bytes(stable_json_bytes(identity))
    return identity


def _decode_proc_maps_path(value: str, label: str) -> Path:
    if value.endswith(" (deleted)"):
        raise ContractError(f"{label} names a deleted file")

    def decode_escape(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    decoded = re.sub(r"\\([0-7]{3})", decode_escape, value)
    if "\x00" in decoded or not decoded.startswith("/"):
        raise ContractError(f"{label} pathname differs")
    return Path(decoded)


def _proc_self_maps_bindings(
    maps_payload: bytes, label: str
) -> dict[Path, tuple[int, int, int]]:
    try:
        lines = maps_payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise ContractError(f"{label} is not strict UTF-8") from error
    bindings: dict[Path, tuple[int, int, int]] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        device_match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+)", fields[3])
        if device_match is None or re.fullmatch(r"0|[1-9][0-9]*", fields[4]) is None:
            raise ContractError(f"{label} device/inode encoding differs")
        path = _decode_proc_maps_path(fields[5], label)
        binding = (
            int(device_match.group(1), 16),
            int(device_match.group(2), 16),
            int(fields[4]),
        )
        if binding[2] == 0:
            raise ContractError(f"{label} absolute file mapping has zero inode")
        previous = bindings.setdefault(path, binding)
        if previous != binding:
            raise ContractError(f"{label} pathname maps more than one inode")
    return bindings


def _mapped_regular_file_closure_record(
    path: Path,
    mapped_identity: tuple[int, int, int],
    *,
    label: str,
) -> dict[str, Any]:
    record, _payload = _regular_file_closure_record(
        path, relative_path=None, label=label
    )
    observed_identity = (
        os.major(record["device"]),
        os.minor(record["device"]),
        record["inode"],
    )
    if observed_identity != mapped_identity:
        raise ContractError(f"{label} pathname no longer names mapped device/inode")
    return {
        **record,
        "mapped_device_major": mapped_identity[0],
        "mapped_device_minor": mapped_identity[1],
        "mapped_inode": mapped_identity[2],
    }


def _preauthorization_native_library_identity() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        files: list[dict[str, Any]] = []
        return {
            "platform": sys.platform,
            "source": "non_linux_development_only_no_proc_self_maps",
            "files": files,
            "closure_sha256": sha256_bytes(stable_json_bytes(files)),
        }
    try:
        maps_payload = Path("/proc/self/maps").read_bytes()
    except OSError as error:
        raise ContractError(
            "pre-authorization native mapping table is unavailable"
        ) from error
    bindings = _proc_self_maps_bindings(
        maps_payload, "pre-authorization native mapping"
    )
    if not bindings:
        raise ContractError("pre-authorization native mapping closure is empty")
    files = [
        _mapped_regular_file_closure_record(
            path,
            bindings[path],
            label="pre-authorization native mapping",
        )
        for path in sorted(bindings, key=str)
    ]
    return {
        "platform": sys.platform,
        "source": "proc_self_maps_bound_device_inode_and_path",
        "files": files,
        "closure_sha256": sha256_bytes(stable_json_bytes(files)),
    }


def observe_runtime_identity(
    python_bin: Path,
    git_bin: Path,
    scontrol_bin: Path,
    sacct_bin: Path,
    nvidia_smi_bin: Path,
    *,
    python_descriptor: int | None = None,
    git_descriptor: int | None = None,
    scontrol_descriptor: int | None = None,
    sacct_descriptor: int | None = None,
    nvidia_smi_descriptor: int | None = None,
    expected_packages: dict[str, Any] | None = None,
    require_sealed_startup: bool = False,
) -> dict[str, Any]:
    python_bin = _require_normalized_absolute_path(python_bin, "PYTHON_BIN")
    if python_descriptor is None and Path(sys.executable).resolve() != python_bin:
        raise ContractError("running interpreter differs from pinned PYTHON_BIN")
    if python_descriptor is not None:
        running_info = Path("/proc/self/exe").stat()
        descriptor_info = os.fstat(python_descriptor)
        if _inode_policy_identity(running_info) != _inode_policy_identity(
            descriptor_info
        ):
            raise ContractError(
                "running interpreter differs from held Python descriptor"
            )
    packages = (
        {
            "torch": _distribution_identity("torch", "torch/__init__.py"),
            "tokenizers": _distribution_identity(
                "tokenizers", "tokenizers/__init__.py"
            ),
        }
        if expected_packages is None
        else {
            name: _rehash_manifest_distribution(expected_packages[name])
            for name in ("torch", "tokenizers")
        }
    )
    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "python": _executable_identity(python_bin, ("--version",), python_descriptor),
        "git": _executable_identity(
            git_bin,
            ("--version",),
            git_descriptor,
            environment=_sealed_git_environment(),
        ),
        "scontrol": _executable_identity(
            scontrol_bin, ("--version",), scontrol_descriptor
        ),
        "sacct": _executable_identity(sacct_bin, ("--version",), sacct_descriptor),
        "nvidia_smi": _executable_identity(
            nvidia_smi_bin, ("--version",), nvidia_smi_descriptor
        ),
        "python_startup": _preauthorization_python_runtime_identity(
            require_sealed_startup=require_sealed_startup
        ),
        "packages": packages,
        "backend": {
            "device": DEVICE,
            "precision": PRECISION,
            "sdpa_backend": SDPA_BACKEND,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "deterministic_algorithms": True,
            "ld_preload": os.environ.get("LD_PRELOAD") or None,
            "dyld_insert_libraries": os.environ.get("DYLD_INSERT_LIBRARIES") or None,
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH") or None,
            "preauthorization_native_libraries": (
                _preauthorization_native_library_identity()
            ),
            "coverage_claim": (
                "sealed_python_startup_full_record_distributions_and_"
                "mapped_device_inode_bound_native_maps"
            ),
        },
    }


def revalidate_runtime_identity(
    expected: dict[str, Any],
    python_bin: Path,
    git_bin: Path,
    scontrol_bin: Path,
    sacct_bin: Path,
    nvidia_smi_bin: Path,
    *,
    python_descriptor: int | None = None,
    git_descriptor: int | None = None,
    scontrol_descriptor: int | None = None,
    sacct_descriptor: int | None = None,
    nvidia_smi_descriptor: int | None = None,
) -> None:
    if expected.get("schema") != RUNTIME_IDENTITY_SCHEMA:
        raise ContractError("runtime revalidation schema differs")
    executable_observations = {
        "python": _executable_identity(python_bin, ("--version",), python_descriptor),
        "git": _executable_identity(
            git_bin,
            ("--version",),
            git_descriptor,
            environment=_sealed_git_environment(),
        ),
        "scontrol": _executable_identity(
            scontrol_bin, ("--version",), scontrol_descriptor
        ),
        "sacct": _executable_identity(sacct_bin, ("--version",), sacct_descriptor),
        "nvidia_smi": _executable_identity(
            nvidia_smi_bin, ("--version",), nvidia_smi_descriptor
        ),
    }
    if any(
        expected[name] != record for name, record in executable_observations.items()
    ):
        raise ContractError("runtime executable changed after authorization")
    if {
        name: _rehash_manifest_distribution(expected["packages"][name])
        for name in ("torch", "tokenizers")
    } != expected["packages"]:
        raise ContractError("runtime package changed after authorization")
    for component in expected["python_startup"]["components"]:
        if _python_component_record(component["reference"]) != component:
            raise ContractError("Python startup component changed after authorization")
    for search_entry in expected["python_startup"]["search_path"]:
        if _python_search_path_record(search_entry["path"]) != search_entry:
            raise ContractError(
                "Python startup search path changed after authorization"
            )
    native = expected["backend"]["preauthorization_native_libraries"]
    if sys.platform.startswith("linux"):
        maps_payload = Path("/proc/self/maps").read_bytes()
        current_bindings = _proc_self_maps_bindings(
            maps_payload, "runtime revalidation mapping"
        )
        for expected_file in native["files"]:
            path = Path(expected_file["path"])
            binding = current_bindings.get(path)
            if binding is None:
                raise ContractError("pre-authorization mapping disappeared")
            if (
                _mapped_regular_file_closure_record(
                    path, binding, label="runtime revalidation mapping"
                )
                != expected_file
            ):
                raise ContractError("pre-authorization mapping changed")
    backend = expected["backend"]
    if (
        (os.environ.get("LD_PRELOAD") or None) != backend["ld_preload"]
        or (os.environ.get("DYLD_INSERT_LIBRARIES") or None)
        != backend["dyld_insert_libraries"]
        or (os.environ.get("LD_LIBRARY_PATH") or None) != backend["ld_library_path"]
    ):
        raise ContractError("runtime loader environment changed after authorization")


def activate_pinned_runtime_packages(runtime_identity: dict[str, Any]) -> None:
    global F, Tokenizer, torch
    observed = runtime_identity["packages"]
    current_startup = _preauthorization_python_runtime_identity(
        require_sealed_startup=runtime_identity["python_startup"]["mode"]
        == PYTHON_STARTUP_MODE
    )
    if current_startup != runtime_identity["python_startup"]:
        raise ContractError("Python startup closure changed before package activation")
    rehashed_before_import = {
        name: _rehash_manifest_distribution(observed[name])
        for name in ("torch", "tokenizers")
    }
    if rehashed_before_import != observed:
        raise ContractError("runtime package bytes changed before import")
    for package_root in dict.fromkeys(
        observed[name]["distribution_root"] for name in ("torch", "tokenizers")
    ):
        if package_root not in sys.path:
            sys.path.append(package_root)
    torch_module = importlib.import_module("torch")
    functional_module = importlib.import_module("torch.nn.functional")
    tokenizers_module = importlib.import_module("tokenizers")
    if (
        Path(torch_module.__file__).resolve() != Path(observed["torch"]["module_path"])
        or Path(tokenizers_module.__file__).resolve()
        != Path(observed["tokenizers"]["module_path"])
        or importlib.metadata.version("torch") != observed["torch"]["version"]
        or importlib.metadata.version("tokenizers") != observed["tokenizers"]["version"]
    ):
        raise ContractError("imported package identity differs from sealed runtime")
    rehashed = {
        "torch": _distribution_identity("torch", "torch/__init__.py"),
        "tokenizers": _distribution_identity("tokenizers", "tokenizers/__init__.py"),
    }
    if rehashed != observed:
        raise ContractError("imported package bytes differ from sealed runtime")
    torch = torch_module
    F = functional_module
    Tokenizer = tokenizers_module.Tokenizer


def verify_runtime_source_manifest(
    source_root: Path,
    source_commit: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    python_bin: Path,
    git_bin: Path,
    scontrol_bin: Path,
    sacct_bin: Path,
    nvidia_smi_bin: Path,
    *,
    manifest_descriptor: int | None = None,
    python_descriptor: int | None = None,
    git_descriptor: int | None = None,
    scontrol_descriptor: int | None = None,
    sacct_descriptor: int | None = None,
    nvidia_smi_descriptor: int | None = None,
    require_sealed_startup: bool = False,
    revalidate_existing_runtime: bool = False,
) -> dict[str, Any]:
    """Bind clean current source bytes to one externally sealed commit manifest."""
    source_root = _require_normalized_absolute_path(source_root, "SOURCE_ROOT")
    manifest_path = _require_normalized_absolute_path(
        manifest_path, "runtime-source manifest"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ContractError("SOURCE_COMMIT must be exactly 40 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ContractError("runtime-source manifest SHA-256 is not lowercase hex")
    try:
        manifest_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ContractError("runtime-source manifest must be outside SOURCE_ROOT")

    manifest_stat = manifest_path.lstat()
    if manifest_stat.st_mode & 0o222:
        raise ContractError(
            "runtime-source manifest must have no write permission bits"
        )
    if manifest_descriptor is None:
        manifest_bytes = read_verified_bytes(manifest_path, expected_manifest_sha256)
    else:
        opened = os.fstat(manifest_descriptor)
        if (
            _file_identity(opened) != _file_identity(manifest_stat)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise ContractError("held runtime-source manifest descriptor differs")
        manifest_bytes = os.pread(manifest_descriptor, opened.st_size + 1, 0)
        if (
            len(manifest_bytes) != opened.st_size
            or sha256_bytes(manifest_bytes) != expected_manifest_sha256
            or _file_identity(os.fstat(manifest_descriptor)) != _file_identity(opened)
            or _file_identity(manifest_path.lstat()) != _file_identity(opened)
        ):
            raise ContractError("held runtime-source manifest changed while reading")
    manifest = _parse_json_object_bytes(manifest_bytes, "runtime-source manifest")
    if stable_json_bytes(manifest) != manifest_bytes:
        raise ContractError("runtime-source manifest is not canonical stable JSON")
    _require_exact_keys(
        manifest,
        {"schema", "source_root", "source_commit", "files", "runtime"},
        "runtime-source manifest",
    )
    if manifest["schema"] != RUNTIME_SOURCE_MANIFEST_SCHEMA:
        raise ContractError("runtime-source manifest schema differs")
    if manifest["source_root"] != str(source_root):
        raise ContractError("runtime-source manifest SOURCE_ROOT differs")
    if manifest["source_commit"] != source_commit:
        raise ContractError("runtime-source manifest SOURCE_COMMIT differs")
    files = manifest["files"]
    if not isinstance(files, dict) or tuple(files) != RUNTIME_SOURCE_PATHS:
        raise ContractError("runtime-source manifest file order or closure differs")
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in files.values()
    ):
        raise ContractError("runtime-source manifest contains an invalid file SHA-256")

    expected_runtime = manifest["runtime"]
    if (
        not isinstance(expected_runtime, dict)
        or not isinstance(expected_runtime.get("packages"), dict)
        or set(expected_runtime["packages"]) != {"torch", "tokenizers"}
    ):
        raise ContractError("runtime/package manifest shape differs")
    if revalidate_existing_runtime:
        revalidate_runtime_identity(
            expected_runtime,
            python_bin,
            git_bin,
            scontrol_bin,
            sacct_bin,
            nvidia_smi_bin,
            python_descriptor=python_descriptor,
            git_descriptor=git_descriptor,
            scontrol_descriptor=scontrol_descriptor,
            sacct_descriptor=sacct_descriptor,
            nvidia_smi_descriptor=nvidia_smi_descriptor,
        )
        observed_runtime = expected_runtime
    else:
        observed_runtime = observe_runtime_identity(
            python_bin,
            git_bin,
            scontrol_bin,
            sacct_bin,
            nvidia_smi_bin,
            python_descriptor=python_descriptor,
            git_descriptor=git_descriptor,
            scontrol_descriptor=scontrol_descriptor,
            sacct_descriptor=sacct_descriptor,
            nvidia_smi_descriptor=nvidia_smi_descriptor,
            expected_packages=expected_runtime["packages"],
            require_sealed_startup=require_sealed_startup,
        )
    if expected_runtime != observed_runtime:
        raise ContractError("runtime/package/backend identity differs from manifest")

    top_level = _run_git(
        git_bin,
        source_root,
        "rev-parse",
        "--show-toplevel",
        executable_descriptor=git_descriptor,
    ).rstrip(b"\n")
    if top_level != os.fsencode(source_root):
        raise ContractError("SOURCE_ROOT is not the exact Git top level")
    head = (
        _run_git(
            git_bin,
            source_root,
            "rev-parse",
            "HEAD",
            executable_descriptor=git_descriptor,
        )
        .decode("ascii")
        .strip()
    )
    if head != source_commit:
        raise ContractError("SOURCE_COMMIT is not the checked-out HEAD")
    if (
        _run_git(
            git_bin,
            source_root,
            "cat-file",
            "-t",
            source_commit,
            executable_descriptor=git_descriptor,
        )
        != b"commit\n"
    ):
        raise ContractError("SOURCE_COMMIT does not identify a commit object")
    status = _run_git(
        git_bin,
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        executable_descriptor=git_descriptor,
    )
    if status:
        raise ContractError("SOURCE_ROOT Git state is not clean")

    for relative_path, expected_sha256 in files.items():
        current_path = source_root / relative_path
        current_bytes = read_verified_bytes(current_path, expected_sha256)
        committed_bytes = _run_git(
            git_bin,
            source_root,
            "show",
            f"{source_commit}:{relative_path}",
            executable_descriptor=git_descriptor,
        )
        if current_bytes != committed_bytes:
            raise ContractError(
                f"current bytes differ from git show for runtime source {relative_path}"
            )
    return {
        "schema": RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "path": str(manifest_path),
        "sha256": expected_manifest_sha256,
        "manifest_file": {
            "device": manifest_stat.st_dev,
            "inode": manifest_stat.st_ino,
            "uid": manifest_stat.st_uid,
            "mode": stat.S_IMODE(manifest_stat.st_mode),
            "size": manifest_stat.st_size,
        },
        "source_root": str(source_root),
        "source_commit": source_commit,
        "files": files,
        "runtime": observed_runtime,
        "git_status": "clean",
        "git_show_byte_equality": True,
    }


def parse_heldout_bytes(data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("heldout is not UTF-8") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ContractError(f"blank heldout line at {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_json_object_without_duplicates)
        except (json.JSONDecodeError, ContractError) as error:
            raise ContractError(
                f"invalid heldout JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ContractError(f"heldout line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ContractError("heldout is empty")
    return rows


DWS_RE = re.compile(
    r"dws:op=(add|sub);w=(0|[1-9][0-9]*);p=(0|[1-9][0-9]*);"
    r"c=([01]);a=([0-9]+);b=([0-9]+);r=([0-9]+);z=([01])"
)
ANSWER_RE = re.compile(r"answer=(-?(?:0|[1-9][0-9]*))")


def _require_plain_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _value_lsf(digits: str) -> int:
    return sum(int(digit) * 10**index for index, digit in enumerate(digits))


def validate_dws_state(state: dict[str, Any]) -> None:
    if set(state) != {"op", "w", "p", "c", "a", "b", "r", "z"}:
        raise ValueError("invalid DWS state keys")
    if state["op"] not in OPERATIONS:
        raise ValueError("invalid operation")
    width = _require_plain_int(state["w"], "w")
    position = _require_plain_int(state["p"], "p")
    carry = _require_plain_int(state["c"], "c")
    terminal = _require_plain_int(state["z"], "z")
    if width <= 0 or not 0 <= position <= width:
        raise ValueError("invalid width or position")
    if carry not in (0, 1) or terminal not in (0, 1):
        raise ValueError("invalid binary field")
    if terminal != int(position == width):
        raise ValueError("terminal flag does not match position")
    for name in ("a", "b", "r"):
        tape = state[name]
        if not isinstance(tape, str) or not re.fullmatch(r"[0-9]+", tape):
            raise ValueError(f"invalid {name} tape")
        if len(tape) != width:
            raise ValueError(f"wrong {name} tape width")
    if any(digit != "0" for digit in state["r"][position:]):
        raise ValueError("unwritten result suffix is not zero")
    if position == 0 and carry:
        raise ValueError("initial state cannot carry")
    if state["op"] == "sub" and _value_lsf(state["a"]) < _value_lsf(state["b"]):
        raise ValueError("subtraction state has negative operands")


def canonical_dws_state(state: dict[str, Any]) -> str:
    validate_dws_state(state)
    return ("dws:op={op};w={w};p={p};c={c};a={a};b={b};r={r};z={z}").format(**state)


def parse_dws_line(line: str) -> dict[str, Any] | None:
    match = DWS_RE.fullmatch(line)
    if match is None:
        return None
    operation, width, position, carry, left, right, result, terminal = match.groups()
    state: dict[str, Any] = {
        "op": operation,
        "w": int(width),
        "p": int(position),
        "c": int(carry),
        "a": left,
        "b": right,
        "r": result,
        "z": int(terminal),
    }
    try:
        if canonical_dws_state(state) != line:
            return None
    except ValueError:
        return None
    return state


def apply_microstep_posthoc(state: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one oracle transition; never called by token decoding."""
    validate_dws_state(state)
    if state["z"]:
        raise ValueError("cannot step a terminal state")
    position = state["p"]
    left = int(state["a"][position])
    right = int(state["b"][position])
    carry = state["c"]
    if state["op"] == "add":
        total = left + right + carry
        digit, next_carry = total % 10, total // 10
    else:
        total = left - right - carry
        digit, next_carry = (total + 10) % 10, int(total < 0)
    result = list(state["r"])
    result[position] = str(digit)
    next_position = position + 1
    next_state = dict(state)
    next_state.update(
        p=next_position,
        c=next_carry,
        r="".join(result),
        z=int(next_position == state["w"]),
    )
    validate_dws_state(next_state)
    return next_state


def reconstruct_oracle_posthoc(initial_state: dict[str, Any]) -> list[dict[str, Any]]:
    validate_dws_state(initial_state)
    if initial_state["p"] != 0 or initial_state["c"] != 0 or initial_state["z"] != 0:
        raise ContractError("selected episode does not start at the initial DWS state")
    states = []
    current = dict(initial_state)
    while not current["z"]:
        current = apply_microstep_posthoc(current)
        states.append(current)
    return states


def state_answer_posthoc(state: dict[str, Any]) -> int:
    validate_dws_state(state)
    if not state["z"]:
        raise ValueError("answer requires a terminal state")
    result = _value_lsf(state["r"])
    if state["op"] == "add":
        return result + state["c"] * 10 ** state["w"]
    if state["c"]:
        raise ValueError("terminal subtraction retains a borrow")
    return result


def render_core_prompt_bytes(state_line: str) -> bytes:
    state = parse_dws_line(state_line)
    if state is None:
        raise ContractError("core prompt requires one canonical DWS state")
    return (PROMPT_PREFIX + state_line + PROMPT_SUFFIX).encode(PROMPT_ENCODING)


def render_initial_prompt_bytes(initial_state_line: str) -> bytes:
    state = parse_dws_line(initial_state_line)
    if state is None or state["p"] != 0:
        raise ContractError("prompt requires one canonical initial DWS state")
    return render_core_prompt_bytes(initial_state_line)


def selection_digest(case_id: str) -> str:
    try:
        case_bytes = case_id.encode("ascii")
    except UnicodeEncodeError as error:
        raise ContractError("case ID is not ASCII") from error
    return sha256_bytes(SELECTION_DOMAIN + case_bytes)


def select_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ContractError("heldout case IDs must be unique strings")
        by_id[case_id] = row
    selected = []
    for regime in EXPECTED_REGIMES:
        for operation in OPERATIONS:
            candidates = [
                row
                for row in rows
                if row.get("split") == regime and row.get("operation") == operation
            ]
            candidates.sort(key=lambda row: (selection_digest(row["id"]), row["id"]))
            if len(candidates) < CASES_PER_CELL:
                raise ContractError(f"insufficient cases in {(regime, operation)}")
            selected.extend(candidates[:CASES_PER_CELL])
    if len(selected) != CASE_COUNT:
        raise ContractError("frozen selector did not produce 100 cases")
    ordered_ids = [row["id"] for row in selected]
    ids_hash = sha256_bytes(("\n".join(ordered_ids) + "\n").encode("ascii"))
    if ids_hash != ORDERED_CASE_IDS_SHA256:
        raise ContractError(f"ordered case selection hash mismatch: {ids_hash}")
    for row in selected:
        if row.get("prompt_style") != "heldout":
            raise ContractError("frozen heldout row changed prompt-style metadata")
        state = parse_dws_line(row.get("initial_state", ""))
        if state is None or state["p"] != 0:
            raise ContractError(f"invalid initial state in {row['id']}")
    replication = []
    for width in (4, 6, 8):
        for operation in OPERATIONS:
            candidates = [
                row
                for row in selected
                if row.get("width") == width and row.get("operation") == operation
            ]
            candidates.sort(
                key=lambda row: (
                    sha256_bytes(
                        SELECTION_DOMAIN
                        + b"replication\x00"
                        + row["id"].encode("ascii")
                    ),
                    row["id"],
                )
            )
            replication.extend(row["id"] for row in candidates[:2])
    if tuple(replication) != REPLICATION_CASE_IDS:
        raise ContractError("frozen 12-case replication selection changed")
    replication_hash = sha256_bytes(("\n".join(replication) + "\n").encode("ascii"))
    if replication_hash != REPLICATION_CASE_IDS_SHA256:
        raise ContractError("frozen replication ID hash changed")
    return selected


def validate_report_case_heldout_identity(
    case: dict[str, Any], heldout_row: dict[str, Any], path: str
) -> dict[str, Any]:
    """Bind one report case to the exact selected heldout row."""
    initial_state = parse_dws_line(case.get("initial_state", ""))
    if initial_state is None or initial_state["p"] != 0:
        raise ContractError(f"case initial state is invalid at {path}")
    if (
        case.get("case_id") != heldout_row.get("id")
        or case.get("split") != heldout_row.get("split")
        or case.get("operation") != heldout_row.get("operation")
        or case.get("width") != heldout_row.get("width")
        or case.get("initial_state") != heldout_row.get("initial_state")
        or case.get("operation") != initial_state["op"]
        or case.get("width") != initial_state["w"]
        or case.get("selection_sha256") != selection_digest(case["case_id"])
    ):
        raise ContractError(f"case frozen heldout identity differs at {path}")
    return initial_state


def replacement_token_id(mode: DecodeMode) -> int | None:
    if mode == DecodeMode.EOS_TO_LF:
        return 211
    if mode == DecodeMode.EOS_TO_SPACE:
        return 233
    if mode == DecodeMode.EOS_TO_SEMICOLON:
        return 39
    if mode == DecodeMode.EOS_TO_NONFORMAT_X:
        return 100
    return None


def apply_decode_intervention(
    next_logits: torch.Tensor, mode: DecodeMode, eos_token_id: int
) -> TokenDecision:
    """Apply the arm's sole token-level rule to one next-token logit vector."""
    if next_logits.ndim != 1 or next_logits.numel() <= eos_token_id:
        raise ContractError("next-token logits have invalid shape")
    if not bool(torch.isfinite(next_logits).all()):
        raise ContractError("model produced non-finite logits before intervention")
    raw_argmax = int(next_logits.argmax().item())
    eos_selected = raw_argmax == eos_token_id
    eos_logit = float(next_logits[eos_token_id].float().item())
    altered = next_logits
    selected = raw_argmax
    override = False
    if mode == DecodeMode.EOS_MASKED_ARGMAX:
        altered = next_logits.clone()
        altered[eos_token_id] = float("-inf")
        if not bool(torch.isfinite(altered).any()):
            raise ContractError("EOS masking left no finite replacement token")
        selected = int(altered.argmax().item())
        override = eos_selected
    elif replacement_token_id(mode) is not None and eos_selected:
        selected = int(replacement_token_id(mode))
        override = True
    selected_raw_logit = float(next_logits[selected].float().item())
    return TokenDecision(
        logits_after_intervention=altered,
        raw_argmax_id=raw_argmax,
        selected_token_id=selected,
        eos_was_raw_argmax=eos_selected,
        eos_logit=eos_logit,
        selected_raw_logit=selected_raw_logit,
        override_applied=override,
    )


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
        raise ContractError("model logits must have shape [1, tokens, vocabulary]")
    return logits[0, -1]


def decode_cached_greedy(request: DecodeRequest) -> RawDecode:
    """Run one greedy KV-cached token decode under a frozen EOS rule."""
    if request.max_new_tokens <= 0:
        raise ContractError("decode budget must be positive")
    prompt_ids = list(request.prompt_token_ids)
    if not prompt_ids or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in prompt_ids
    ):
        raise ContractError("decode prompt token IDs are invalid")
    cap = int(request.model.cfg.seq_len)
    if len(prompt_ids) + request.max_new_tokens > cap:
        raise ContractError("frozen decode would exceed the model context")
    logits, cache = request.model(
        torch.tensor([prompt_ids], dtype=torch.long, device=request.device),
        return_cache=True,
        pos=0,
    )
    generated: list[int] = []
    mask_positions: list[int] = []
    eos_events: list[RawEosEvent] = []
    stop_reason = "fixed_budget"
    for generated_index in range(request.max_new_tokens):
        decision = apply_decode_intervention(
            _last_token_logits(logits), request.mode, request.eos_token_id
        )
        if request.mode == DecodeMode.EOS_MASKED_ARGMAX:
            mask_positions.append(generated_index)
        if decision.eos_was_raw_argmax:
            non_eos_logits = _last_token_logits(logits).clone()
            non_eos_logits[request.eos_token_id] = float("-inf")
            non_eos_argmax = int(non_eos_logits.argmax().item())
            eos_events.append(
                RawEosEvent(
                    generated_index=generated_index,
                    absolute_token_position=len(prompt_ids) + generated_index,
                    eos_logit=decision.eos_logit,
                    non_eos_argmax_token_id=non_eos_argmax,
                    non_eos_argmax_logit=float(
                        _last_token_logits(logits)[non_eos_argmax].float().item()
                    ),
                    replacement_token_id=(
                        decision.selected_token_id
                        if decision.override_applied
                        else None
                    ),
                    replacement_raw_logit=(
                        decision.selected_raw_logit
                        if decision.override_applied
                        else None
                    ),
                )
            )
        generated.append(decision.selected_token_id)
        if (
            request.mode == DecodeMode.ORDINARY_EOS_STOP
            and decision.selected_token_id == request.eos_token_id
        ):
            stop_reason = "eos"
            break
        if generated_index + 1 == request.max_new_tokens:
            break
        logits, cache = request.model(
            torch.tensor(
                [[decision.selected_token_id]],
                dtype=torch.long,
                device=request.device,
            ),
            cache=cache,
            pos=len(prompt_ids) + generated_index,
            return_cache=True,
        )
    if (
        request.mode.value in FIXED_BUDGET_ARMS
        and len(generated) != request.max_new_tokens
    ):
        raise ContractError("fixed-budget arm stopped before its token budget")
    if request.mode == DecodeMode.ORDINARY_EOS_STOP and stop_reason != "eos":
        stop_reason = "max_new_tokens"
    return RawDecode(
        mode=request.mode.value,
        prompt_token_ids=tuple(prompt_ids),
        prompt_token_count=len(prompt_ids),
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        eos_mask_applied_positions=tuple(mask_positions),
        eos_events=tuple(eos_events),
    )


def validate_static_contract() -> None:
    prefix = PROMPT_PREFIX.encode(PROMPT_ENCODING)
    suffix = PROMPT_SUFFIX.encode(PROMPT_ENCODING)
    template = prefix + b"{initial_state}" + suffix
    observed = (
        sha256_bytes(prefix),
        sha256_bytes(suffix),
        sha256_bytes(template),
    )
    expected = (
        PROMPT_PREFIX_SHA256,
        PROMPT_SUFFIX_SHA256,
        PROMPT_TEMPLATE_SHA256,
    )
    if observed != expected:
        raise ContractError(f"prompt byte contract changed: {observed}")
    if tuple(REPLACEMENT_TOKENS) != FIELD_CLOCK_ARMS[1:]:
        raise ContractError("replacement-token arm order changed")
    if MAX_NEW_TOKENS != 768 or CASE_COUNT != 100:
        raise ContractError("frozen screen size or token budget changed")


def validate_tokenizer_contract(tokenizer: Any) -> None:
    if tokenizer.token_to_id("<|endoftext|>") != EOS_TOKEN_ID:
        raise ContractError("tokenizer EOS ID changed")
    for arm, (token_id, text) in REPLACEMENT_TOKENS.items():
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        encoded = list(tokenizer.encode(text).ids)
        if decoded != text or encoded != [token_id]:
            raise ContractError(f"replacement token contract changed for {arm}")


def response_content_token_ids(raw: RawDecode) -> tuple[int, ...]:
    tokens = raw.generated_token_ids
    if raw.stop_reason == "eos":
        if not tokens or tokens[-1] != EOS_TOKEN_ID:
            raise ContractError("EOS-stopped decode lacks its terminal EOS token")
        return tokens[:-1]
    return tokens


def decode_text_posthoc(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def score_trace_posthoc(
    response_text: str,
    initial_state: dict[str, Any],
    oracle_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse and score all complete DWS lines after generation is finished."""
    lines = response_text.split("\n")
    dws_lines: list[dict[str, Any]] = []
    answer_lines: list[dict[str, Any]] = []
    nonprotocol_line_indices: list[int] = []
    seen: set[str] = set()
    first_terminal_line_index: int | None = None
    for line_index, line in enumerate(lines):
        if "dws:" in line:
            state = parse_dws_line(line)
            canonical = canonical_dws_state(state) if state is not None else None
            repeated = canonical in seen if canonical is not None else False
            post_terminal = first_terminal_line_index is not None
            candidate_index = len(dws_lines)
            exact_at_oracle_index = bool(
                state is not None
                and candidate_index < len(oracle_states)
                and state == oracle_states[candidate_index]
            )
            dws_lines.append(
                {
                    "line_index": line_index,
                    "candidate_index": candidate_index,
                    "text": line,
                    "grammar_valid": state is not None,
                    "canonical_state": canonical,
                    "repeated": repeated,
                    "post_terminal": post_terminal,
                    "exact_at_oracle_index": exact_at_oracle_index,
                }
            )
            if canonical is not None:
                seen.add(canonical)
            if state is not None and state["z"] and first_terminal_line_index is None:
                first_terminal_line_index = line_index
            continue
        answer_match = ANSWER_RE.fullmatch(line)
        if answer_match is not None:
            answer_lines.append(
                {
                    "line_index": line_index,
                    "text": line,
                    "value": int(answer_match.group(1)),
                }
            )
        elif line:
            nonprotocol_line_indices.append(line_index)

    longest_exact_prefix = 0
    for record in dws_lines:
        if not record["exact_at_oracle_index"]:
            break
        longest_exact_prefix += 1

    transition_grammar_valid = bool(dws_lines)
    current = dict(initial_state)
    for record in dws_lines:
        state = (
            parse_dws_line(record["canonical_state"])
            if record["canonical_state"] is not None
            else None
        )
        if (
            state is None
            or current["z"]
            or record["repeated"]
            or record["post_terminal"]
            or apply_microstep_posthoc(current) != state
        ):
            transition_grammar_valid = False
            break
        current = state

    parsed_terminal = next(
        (
            parse_dws_line(record["canonical_state"])
            for record in dws_lines
            if record["canonical_state"] is not None
            and parse_dws_line(record["canonical_state"])["z"]
        ),
        None,
    )
    oracle_terminal = oracle_states[-1]
    observed_final_tape = (
        {"r": parsed_terminal["r"], "c": parsed_terminal["c"]}
        if parsed_terminal is not None
        else None
    )
    oracle_final_tape = {"r": oracle_terminal["r"], "c": oracle_terminal["c"]}
    exact_final_tape = observed_final_tape == oracle_final_tape
    terminal_answer: int | None = None
    if parsed_terminal is not None:
        try:
            terminal_answer = state_answer_posthoc(parsed_terminal)
        except ValueError:
            terminal_answer = None
    oracle_answer = state_answer_posthoc(oracle_terminal)
    generated_answer_values = [record["value"] for record in answer_lines]
    full_trace = bool(
        longest_exact_prefix == len(oracle_states)
        and first_terminal_line_index == dws_lines[len(oracle_states) - 1]["line_index"]
    )
    malformed = [
        record["line_index"] for record in dws_lines if not record["grammar_valid"]
    ]
    repeated = [record["line_index"] for record in dws_lines if record["repeated"]]
    post_terminal = [
        record["line_index"] for record in dws_lines if record["post_terminal"]
    ]
    all_dws_canonical = bool(dws_lines) and not malformed
    answer_placement_valid = all(
        first_terminal_line_index is not None
        and record["line_index"] > first_terminal_line_index
        for record in answer_lines
    )
    response_grammar_valid = bool(
        all_dws_canonical
        and transition_grammar_valid
        and not nonprotocol_line_indices
        and not repeated
        and not post_terminal
        and len(answer_lines) <= 1
        and answer_placement_valid
    )
    return {
        "dws_lines": dws_lines,
        "dws_candidate_count": len(dws_lines),
        "valid_dws_line_count": sum(record["grammar_valid"] for record in dws_lines),
        "malformed_dws_line_indices": malformed,
        "repeated_dws_line_indices": repeated,
        "post_terminal_dws_line_indices": post_terminal,
        "nonprotocol_line_indices": nonprotocol_line_indices,
        "answer_lines": answer_lines,
        "longest_exact_prefix": longest_exact_prefix,
        "oracle_trace_length": len(oracle_states),
        "first_state_exact": bool(dws_lines and dws_lines[0]["exact_at_oracle_index"]),
        "full_exact_trace_through_first_terminal": full_trace,
        "first_terminal_line_index": first_terminal_line_index,
        "exact_terminal_state": parsed_terminal == oracle_terminal,
        "observed_final_tape": observed_final_tape,
        "oracle_final_tape": oracle_final_tape,
        "exact_final_tape": exact_final_tape,
        "terminal_answer": terminal_answer,
        "oracle_answer": oracle_answer,
        "terminal_answer_exact": terminal_answer == oracle_answer,
        "generated_answer_values": generated_answer_values,
        "generated_answer_exact": generated_answer_values == [oracle_answer],
        "all_dws_lines_canonical": all_dws_canonical,
        "transition_grammar_valid": transition_grammar_valid,
        "response_grammar_valid": response_grammar_valid,
    }


def raw_decode_report_posthoc(
    raw: RawDecode,
    tokenizer: Any,
    initial_state: dict[str, Any],
    oracle_states: list[dict[str, Any]],
) -> dict[str, Any]:
    content_ids = response_content_token_ids(raw)
    response_text = decode_text_posthoc(tokenizer, content_ids)
    events = []
    for event in raw.eos_events:
        replacement_text = (
            tokenizer.decode([event.replacement_token_id], skip_special_tokens=False)
            if event.replacement_token_id is not None
            else None
        )
        events.append(
            {
                "generated_index": event.generated_index,
                "absolute_token_position": event.absolute_token_position,
                "eos_logit": event.eos_logit,
                "non_eos_argmax_token_id": event.non_eos_argmax_token_id,
                "non_eos_argmax_token_text": tokenizer.decode(
                    [event.non_eos_argmax_token_id], skip_special_tokens=False
                ),
                "non_eos_argmax_logit": event.non_eos_argmax_logit,
                "replacement_token_id": event.replacement_token_id,
                "replacement_token_text": replacement_text,
                "replacement_raw_logit": event.replacement_raw_logit,
            }
        )
    raw_eos_positions = [event.generated_index for event in raw.eos_events]
    override_positions = [
        event.generated_index
        for event in raw.eos_events
        if event.replacement_token_id is not None
    ]
    return {
        "mode": raw.mode,
        "decode_prompt_token_ids": list(raw.prompt_token_ids),
        "decode_prompt_token_ids_sha256": token_ids_sha256(raw.prompt_token_ids),
        "decode_prompt_token_count": raw.prompt_token_count,
        "generated_token_ids": list(raw.generated_token_ids),
        "generated_token_count": len(raw.generated_token_ids),
        "content_token_count": len(content_ids),
        "generated_token_ids_sha256": token_ids_sha256(raw.generated_token_ids),
        "stop_reason": raw.stop_reason,
        "fixed_budget": raw.mode in FIXED_BUDGET_ARMS,
        "eos_mask_applied_positions": list(raw.eos_mask_applied_positions),
        "model_selected_eos_positions": raw_eos_positions,
        "override_positions": override_positions,
        "override_count": len(override_positions),
        "eos_events": events,
        "response_text": response_text,
        "response_sha256": sha256_bytes(response_text.encode("utf-8")),
        "trace_score": score_trace_posthoc(response_text, initial_state, oracle_states),
    }


def extract_first_emitted_state_posthoc(
    ordinary_report: dict[str, Any], tokenizer: Any
) -> tuple[dict[str, Any] | None, tuple[int, ...] | None, str | None]:
    """Derive field-screen availability from ordinary token evidence only."""
    if (
        ordinary_report["mode"] != DecodeMode.ORDINARY_EOS_STOP.value
        or ordinary_report["stop_reason"] != "eos"
        or ordinary_report["fixed_budget"] is not False
        or ordinary_report["override_count"] != 0
        or len(ordinary_report["eos_events"]) != 1
        or not ordinary_report["generated_token_ids"]
        or ordinary_report["generated_token_ids"][-1] != EOS_TOKEN_ID
    ):
        return None, None, "ordinary_decode_has_no_exact_first_eos_boundary"
    line = ordinary_report["response_text"]
    if not line or "\n" in line or "\r" in line:
        return None, None, "ordinary_first_response_missing_or_malformed"
    state = parse_dws_line(line)
    if (
        state is None
        or canonical_dws_state(state) != line
        or state["z"]
        or state["p"] != 1
    ):
        return None, None, "ordinary_first_response_is_not_one_nonterminal_step"
    content_ids = tuple(ordinary_report["generated_token_ids"][:-1])
    encoded = tuple(tokenizer.encode(line).ids)
    if (
        content_ids != encoded
        or ordinary_report["content_token_count"] != len(content_ids)
        or ordinary_report["eos_events"][0]["generated_index"] != len(content_ids)
    ):
        return None, None, "ordinary_first_response_is_not_token_exact_canonical_line"
    return state, content_ids, None


def _flip_digit(value: str) -> str:
    return str((int(value) + 9) % 10)


def active_operand_branch_posthoc(
    emitted_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    nominal_next = apply_microstep_posthoc(emitted_state)
    position = emitted_state["p"]
    for tape_name in ("a", "b"):
        for delta in (1, -1):
            candidate = dict(emitted_state)
            tape = list(candidate[tape_name])
            old = int(tape[position])
            new = (old + delta) % 10
            tape[position] = str(new)
            candidate[tape_name] = "".join(tape)
            try:
                validate_dws_state(candidate)
                candidate_next = apply_microstep_posthoc(candidate)
            except ValueError:
                continue
            if candidate_next["r"][position] != nominal_next["r"][position]:
                return candidate, {
                    "field": f"{tape_name}[{position}]",
                    "tape": tape_name,
                    "position": position,
                    "from": str(old),
                    "to": str(new),
                    "candidate_order": "a_then_b__plus1_then_minus1_mod10",
                }
    raise ContractError("no valid active-operand perturbation exists")


def build_history_branches_posthoc(
    emitted_state: dict[str, Any], emitted_token_ids: tuple[int, ...], tokenizer: Any
) -> dict[str, dict[str, Any]]:
    validate_dws_state(emitted_state)
    if emitted_state["p"] != 1 or emitted_state["z"]:
        raise ContractError("field screen requires a first emitted nonterminal state")
    branches: dict[str, dict[str, Any]] = {}

    def state_branch(
        name: str, state: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        line = canonical_dws_state(state)
        token_ids = tuple(tokenizer.encode(line).ids)
        branches[name] = {
            "state": state,
            "state_line": line,
            "history_token_ids": token_ids,
            "history_token_count": len(token_ids),
            "metadata": metadata,
        }

    state_branch("intact", dict(emitted_state), {"intervention": "none"})

    carry = dict(emitted_state)
    carry["c"] = 1 - carry["c"]
    state_branch(
        "carry_flip",
        carry,
        {"field": "c", "from": emitted_state["c"], "to": carry["c"]},
    )

    result = dict(emitted_state)
    result_tape = list(result["r"])
    result_tape[0] = _flip_digit(result_tape[0])
    result["r"] = "".join(result_tape)
    state_branch(
        "written_result_r0_flip",
        result,
        {
            "field": "r[0]",
            "from": emitted_state["r"][0],
            "to": result["r"][0],
        },
    )

    operand_state, operand_metadata = active_operand_branch_posthoc(emitted_state)
    state_branch("active_operand_digit_perturbation", operand_state, operand_metadata)

    branches["equal_token_length_destroyed_history"] = {
        "state": None,
        "state_line": None,
        "history_token_ids": (100,) * len(emitted_token_ids),
        "history_token_count": len(emitted_token_ids),
        "metadata": {
            "intervention": "replace_each_emitted_history_token_with_id_100",
            "source_token_count": len(emitted_token_ids),
        },
    }
    if tuple(branches) != HISTORY_BRANCHES:
        raise ContractError("history branch order changed")
    return branches


def _common_decode_fields_posthoc(raw: RawDecode, tokenizer: Any) -> dict[str, Any]:
    content_ids = response_content_token_ids(raw)
    response_text = decode_text_posthoc(tokenizer, content_ids)
    events = []
    for event in raw.eos_events:
        replacement_text = (
            tokenizer.decode([event.replacement_token_id], skip_special_tokens=False)
            if event.replacement_token_id is not None
            else None
        )
        events.append(
            {
                "generated_index": event.generated_index,
                "absolute_token_position": event.absolute_token_position,
                "eos_logit": event.eos_logit,
                "non_eos_argmax_token_id": event.non_eos_argmax_token_id,
                "non_eos_argmax_token_text": tokenizer.decode(
                    [event.non_eos_argmax_token_id], skip_special_tokens=False
                ),
                "non_eos_argmax_logit": event.non_eos_argmax_logit,
                "replacement_token_id": event.replacement_token_id,
                "replacement_token_text": replacement_text,
                "replacement_raw_logit": event.replacement_raw_logit,
            }
        )
    return {
        "mode": raw.mode,
        "decode_prompt_token_ids": list(raw.prompt_token_ids),
        "decode_prompt_token_ids_sha256": token_ids_sha256(raw.prompt_token_ids),
        "decode_prompt_token_count": raw.prompt_token_count,
        "generated_token_ids": list(raw.generated_token_ids),
        "generated_token_count": len(raw.generated_token_ids),
        "content_token_count": len(content_ids),
        "generated_token_ids_sha256": token_ids_sha256(raw.generated_token_ids),
        "stop_reason": raw.stop_reason,
        "fixed_budget": raw.mode in FIXED_BUDGET_ARMS,
        "eos_mask_applied_positions": list(raw.eos_mask_applied_positions),
        "model_selected_eos_positions": [
            event.generated_index for event in raw.eos_events
        ],
        "override_positions": [
            event.generated_index
            for event in raw.eos_events
            if event.replacement_token_id is not None
        ],
        "override_count": sum(
            event.replacement_token_id is not None for event in raw.eos_events
        ),
        "eos_events": events,
        "response_text": response_text,
        "response_sha256": sha256_bytes(response_text.encode("utf-8")),
    }


def first_adjacent_state_posthoc(response_text: str) -> dict[str, Any] | None:
    """Accept only a canonical DWS line as the first nonempty continuation line."""
    for line in response_text.split("\n"):
        if not line:
            continue
        return parse_dws_line(line)
    return None


def field_branch_report_posthoc(
    raw: RawDecode,
    tokenizer: Any,
    target_state: dict[str, Any] | None,
) -> dict[str, Any]:
    common = _common_decode_fields_posthoc(raw, tokenizer)
    observed = first_adjacent_state_posthoc(common["response_text"])
    common.update(
        {
            "first_line_serialization_valid": observed is not None,
            "observed_first_state": (
                canonical_dws_state(observed) if observed is not None else None
            ),
            "target_state": (
                canonical_dws_state(target_state) if target_state is not None else None
            ),
            "full_target_exact": bool(
                observed is not None
                and target_state is not None
                and observed == target_state
            ),
        }
    )
    return common


def _state_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    value = report["observed_first_state"]
    return parse_dws_line(value) if value is not None else None


def _endpoint_value(
    state: dict[str, Any] | None,
    endpoint: str,
    source_state: dict[str, Any],
    metadata: dict[str, Any],
) -> str | int | None:
    if state is None:
        return None
    if endpoint == "active_result_digit":
        return state["r"][source_state["p"]]
    if endpoint == "next_carry":
        return state["c"]
    if endpoint == "written_result_r0":
        return state["r"][0]
    if endpoint == "active_operand_digit":
        return state[metadata["tape"]][metadata["position"]]
    raise ContractError(f"unknown field endpoint: {endpoint}")


def paired_endpoint_score(
    endpoint: str,
    source_state: dict[str, Any],
    metadata: dict[str, Any],
    intact_target: dict[str, Any],
    counterfactual_target: dict[str, Any],
    intact_observed: dict[str, Any] | None,
    counterfactual_observed: dict[str, Any] | None,
) -> dict[str, Any]:
    intact_target_value = _endpoint_value(
        intact_target, endpoint, source_state, metadata
    )
    counterfactual_target_value = _endpoint_value(
        counterfactual_target, endpoint, source_state, metadata
    )
    intact_observed_value = _endpoint_value(
        intact_observed, endpoint, source_state, metadata
    )
    counterfactual_observed_value = _endpoint_value(
        counterfactual_observed, endpoint, source_state, metadata
    )
    target_changed = intact_target_value != counterfactual_target_value
    output_changed = (
        intact_observed_value is not None
        and counterfactual_observed_value is not None
        and intact_observed_value != counterfactual_observed_value
    )
    paired_exact = bool(
        target_changed
        and output_changed
        and intact_observed_value == intact_target_value
        and counterfactual_observed_value == counterfactual_target_value
    )
    return {
        "endpoint": endpoint,
        "intact_target": intact_target_value,
        "counterfactual_target": counterfactual_target_value,
        "intact_observed": intact_observed_value,
        "counterfactual_observed": counterfactual_observed_value,
        "target_changed": target_changed,
        "output_changed": output_changed,
        "intact_target_exact": intact_observed_value == intact_target_value,
        "counterfactual_target_exact": (
            counterfactual_observed_value == counterfactual_target_value
        ),
        "paired_target_switch_exact": paired_exact,
    }


def score_field_clock_posthoc(
    clock_arm: str,
    boundary_token_id: int,
    boundary_token_text: str,
    branch_reports: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, Any]],
    emitted_state: dict[str, Any],
    episode_oracle: list[dict[str, Any]],
) -> dict[str, Any]:
    intact_target = apply_microstep_posthoc(emitted_state)
    intact_observed = _state_from_report(branch_reports["intact"])
    interventions: dict[str, Any] = {}
    endpoint_sets = {
        "carry_flip": ("active_result_digit", "next_carry"),
        "written_result_r0_flip": ("written_result_r0",),
        "active_operand_digit_perturbation": (
            "active_result_digit",
            "active_operand_digit",
        ),
    }
    for branch_name, endpoints in endpoint_sets.items():
        branch_state = branches[branch_name]["state"]
        if branch_state is None:
            raise ContractError("semantic field branch lacks a state")
        target = apply_microstep_posthoc(branch_state)
        observed = _state_from_report(branch_reports[branch_name])
        endpoint_scores = {
            endpoint: paired_endpoint_score(
                endpoint,
                emitted_state,
                branches[branch_name]["metadata"],
                intact_target,
                target,
                intact_observed,
                observed,
            )
            for endpoint in endpoints
        }
        interventions[branch_name] = {
            "metadata": branches[branch_name]["metadata"],
            "target_state": canonical_dws_state(target),
            "full_target_exact": branch_reports[branch_name]["full_target_exact"],
            "whole_output_changed": bool(
                intact_observed is not None
                and observed is not None
                and intact_observed != observed
            ),
            "endpoints": endpoint_scores,
        }

    destroyed_observed = _state_from_report(
        branch_reports["equal_token_length_destroyed_history"]
    )
    intact_adjacent_exact = intact_observed == intact_target
    destroyed_adjacent_exact = destroyed_observed == intact_target
    history = {
        "intact_adjacent_transition_exact": intact_adjacent_exact,
        "destroyed_matches_intact_target": destroyed_adjacent_exact,
        "whole_output_changed": bool(
            intact_observed is not None
            and destroyed_observed is not None
            and intact_observed != destroyed_observed
        ),
        "history_destruction_paired_loss": bool(
            intact_adjacent_exact and not destroyed_adjacent_exact
        ),
    }
    nominal_second_target = episode_oracle[1] if len(episode_oracle) > 1 else None
    return {
        "clock_arm": clock_arm,
        "boundary_token_id": boundary_token_id,
        "boundary_token_text": boundary_token_text,
        "nominal_second_exact": bool(
            intact_observed is not None
            and nominal_second_target is not None
            and intact_observed == nominal_second_target
        ),
        "adjacent_transition_of_emitted_state_exact": intact_adjacent_exact,
        "intact_target_state": canonical_dws_state(intact_target),
        "branch_contracts": {
            name: {
                "state_line": branch["state_line"],
                "history_token_count": branch["history_token_count"],
                "metadata": branch["metadata"],
            }
            for name, branch in branches.items()
        },
        "branch_reports": branch_reports,
        "interventions": interventions,
        "history_destruction": history,
    }


def boundary_token_for_clock(clock_arm: str, first_event: dict[str, Any]) -> int:
    if clock_arm == DecodeMode.EOS_MASKED_ARGMAX.value:
        return int(first_event["non_eos_argmax_token_id"])
    if clock_arm in REPLACEMENT_TOKENS:
        return REPLACEMENT_TOKENS[clock_arm][0]
    raise ContractError(f"invalid field clock arm: {clock_arm}")


def prepare_field_requests_posthoc(
    model: Any,
    device: str,
    prompt_ids: tuple[int, ...],
    ordinary_report: dict[str, Any],
    tokenizer: Any,
) -> tuple[
    dict[str, dict[str, DecodeRequest]],
    dict[str, DecodeRequest],
    dict[str, dict[str, Any]] | None,
    dict[str, Any],
]:
    emitted_state, emitted_ids, failure = extract_first_emitted_state_posthoc(
        ordinary_report, tokenizer
    )
    if emitted_state is None or emitted_ids is None:
        return {}, {}, None, {"available": False, "failure": failure}
    events = ordinary_report["eos_events"]
    if len(events) != 1 or ordinary_report["stop_reason"] != "eos":
        return (
            {},
            {},
            None,
            {
                "available": False,
                "failure": "ordinary_first_decode_did_not_stop_at_exactly_one_eos",
            },
        )
    branches = build_history_branches_posthoc(emitted_state, emitted_ids, tokenizer)
    requests: dict[str, dict[str, DecodeRequest]] = {}
    for clock_arm in FIELD_CLOCK_ARMS:
        boundary_token = boundary_token_for_clock(clock_arm, events[0])
        mode = DecodeMode(clock_arm)
        requests[clock_arm] = {}
        for branch_name in HISTORY_BRANCHES:
            history_ids = branches[branch_name]["history_token_ids"]
            continuation_prefix = (*prompt_ids, *history_ids, boundary_token)
            requests[clock_arm][branch_name] = DecodeRequest(
                model=model,
                prompt_token_ids=tuple(continuation_prefix),
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
                mode=mode,
                eos_token_id=EOS_TOKEN_ID,
            )
    fresh_reencoding_requests = {}
    for branch_name in FRESH_REENCODING_BRANCHES:
        state_line = branches[branch_name]["state_line"]
        if state_line is None:
            raise ContractError("fresh re-encoding branch lacks a state")
        fresh_reencoding_requests[branch_name] = DecodeRequest(
            model=model,
            prompt_token_ids=_tokenize_prompt(
                tokenizer, render_core_prompt_bytes(state_line)
            ),
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            mode=DecodeMode.ORDINARY_EOS_STOP,
            eos_token_id=EOS_TOKEN_ID,
        )
    return (
        requests,
        fresh_reencoding_requests,
        branches,
        {
            "available": True,
            "failure": None,
            "emitted_state": canonical_dws_state(emitted_state),
            "emitted_history_token_count": len(emitted_ids),
        },
    )


def unavailable_field_screen(reason: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "failure": reason or "unknown_field_screen_failure",
        "emitted_state": None,
        "emitted_history_token_count": None,
        "by_clock": {
            clock_arm: {
                "clock_arm": clock_arm,
                "missing_or_malformed_failure": True,
                "nominal_second_exact": False,
                "adjacent_transition_of_emitted_state_exact": False,
                "carry_full_target_exact": False,
                "carry_target_switch_exact": False,
                "result_full_target_exact": False,
                "result_target_switch_exact": False,
                "operand_full_target_exact": False,
                "operand_target_switch_exact": False,
                "history_destruction_paired_loss": False,
                "detail": None,
            }
            for clock_arm in FIELD_CLOCK_ARMS
        },
        "fresh_latest_reencoding": unavailable_fresh_reencoding(reason),
    }


def compact_field_clock_score(detail: dict[str, Any]) -> dict[str, Any]:
    carry = detail["interventions"]["carry_flip"]
    result = detail["interventions"]["written_result_r0_flip"]
    operand = detail["interventions"]["active_operand_digit_perturbation"]
    return {
        "clock_arm": detail["clock_arm"],
        "missing_or_malformed_failure": False,
        "nominal_second_exact": detail["nominal_second_exact"],
        "adjacent_transition_of_emitted_state_exact": detail[
            "adjacent_transition_of_emitted_state_exact"
        ],
        "carry_full_target_exact": carry["full_target_exact"],
        "carry_target_switch_exact": carry["endpoints"]["active_result_digit"][
            "paired_target_switch_exact"
        ],
        "result_full_target_exact": result["full_target_exact"],
        "result_target_switch_exact": result["endpoints"]["written_result_r0"][
            "paired_target_switch_exact"
        ],
        "operand_full_target_exact": operand["full_target_exact"],
        "operand_target_switch_exact": operand["endpoints"]["active_result_digit"][
            "paired_target_switch_exact"
        ],
        "history_destruction_paired_loss": detail["history_destruction"][
            "history_destruction_paired_loss"
        ],
        "detail": detail,
    }


def score_fresh_reencoding_posthoc(
    branch_reports: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, Any]],
    emitted_state: dict[str, Any],
    episode_oracle: list[dict[str, Any]],
    full_history_lf: dict[str, Any],
) -> dict[str, Any]:
    intact_target = apply_microstep_posthoc(emitted_state)
    intact_observed = _state_from_report(branch_reports["intact"])
    endpoint_sets = {
        "carry_flip": ("active_result_digit", "next_carry"),
        "written_result_r0_flip": ("written_result_r0",),
        "active_operand_digit_perturbation": (
            "active_result_digit",
            "active_operand_digit",
        ),
    }
    interventions = {}
    for branch_name, endpoints in endpoint_sets.items():
        branch_state = branches[branch_name]["state"]
        if branch_state is None:
            raise ContractError("fresh re-encoding branch lacks semantic state")
        target = apply_microstep_posthoc(branch_state)
        observed = _state_from_report(branch_reports[branch_name])
        interventions[branch_name] = {
            "metadata": branches[branch_name]["metadata"],
            "target_state": canonical_dws_state(target),
            "full_target_exact": branch_reports[branch_name]["full_target_exact"],
            "whole_output_changed": bool(
                intact_observed is not None
                and observed is not None
                and intact_observed != observed
            ),
            "endpoints": {
                endpoint: paired_endpoint_score(
                    endpoint,
                    emitted_state,
                    branches[branch_name]["metadata"],
                    intact_target,
                    target,
                    intact_observed,
                    observed,
                )
                for endpoint in endpoints
            },
        }
    carry = interventions["carry_flip"]
    result = interventions["written_result_r0_flip"]
    operand = interventions["active_operand_digit_perturbation"]
    carry_endpoint = carry["endpoints"]["active_result_digit"]
    nominal_second = episode_oracle[1] if len(episode_oracle) > 1 else None
    return {
        "mode": "compound_fresh_core_prompt_latest_state_reencoding_canonicalization",
        "external_reencoding": True,
        "nominal_second_exact": bool(
            intact_observed is not None
            and nominal_second is not None
            and intact_observed == nominal_second
        ),
        "adjacent_transition_of_emitted_state_exact": intact_observed == intact_target,
        "carry_full_target_exact": carry["full_target_exact"],
        "carry_output_switch": carry_endpoint["output_changed"],
        "carry_target_switch_exact": carry_endpoint["paired_target_switch_exact"],
        "result_full_target_exact": result["full_target_exact"],
        "result_target_switch_exact": result["endpoints"]["written_result_r0"][
            "paired_target_switch_exact"
        ],
        "operand_full_target_exact": operand["full_target_exact"],
        "operand_target_switch_exact": operand["endpoints"]["active_result_digit"][
            "paired_target_switch_exact"
        ],
        "carry_target_switch_recovery_vs_full_history_lf": bool(
            carry_endpoint["paired_target_switch_exact"]
            and not full_history_lf["carry_target_switch_exact"]
        ),
        "branch_reports": branch_reports,
        "interventions": interventions,
    }


def compact_fresh_reencoding_score(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "failure": None,
        "mode": detail["mode"],
        "external_reencoding": detail["external_reencoding"],
        "nominal_second_exact": detail["nominal_second_exact"],
        "adjacent_transition_of_emitted_state_exact": detail[
            "adjacent_transition_of_emitted_state_exact"
        ],
        "carry_full_target_exact": detail["carry_full_target_exact"],
        "carry_output_switch": detail["carry_output_switch"],
        "carry_target_switch_exact": detail["carry_target_switch_exact"],
        "result_full_target_exact": detail["result_full_target_exact"],
        "result_target_switch_exact": detail["result_target_switch_exact"],
        "operand_full_target_exact": detail["operand_full_target_exact"],
        "operand_target_switch_exact": detail["operand_target_switch_exact"],
        "carry_target_switch_recovery_vs_full_history_lf": detail[
            "carry_target_switch_recovery_vs_full_history_lf"
        ],
        "detail": detail,
    }


def unavailable_fresh_reencoding(reason: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "failure": reason or "unknown_fresh_reencoding_failure",
        "mode": "compound_fresh_core_prompt_latest_state_reencoding_canonicalization",
        "external_reencoding": True,
        "nominal_second_exact": False,
        "adjacent_transition_of_emitted_state_exact": False,
        "carry_full_target_exact": False,
        "carry_output_switch": False,
        "carry_target_switch_exact": False,
        "result_full_target_exact": False,
        "result_target_switch_exact": False,
        "operand_full_target_exact": False,
        "operand_target_switch_exact": False,
        "carry_target_switch_recovery_vs_full_history_lf": False,
        "detail": None,
    }


def validate_heldout_oracle_posthoc(
    row: dict[str, Any], oracle_states: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_states = [canonical_dws_state(state) for state in oracle_states]
    if row.get("expected_states") != expected_states:
        raise ContractError(f"heldout expected-state mismatch in {row['id']}")
    expected_answer = state_answer_posthoc(oracle_states[-1])
    if row.get("expected_answer") != expected_answer:
        raise ContractError(f"heldout expected-answer mismatch in {row['id']}")
    return {
        "case_id": row["id"],
        "row_sha256": sha256_bytes(stable_json_bytes(row)),
        "expected_states_sha256": sha256_bytes(
            stable_json_bytes(row["expected_states"])
        ),
        "expected_answer": row["expected_answer"],
    }


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def carry_target_switch_veto(
    records: list[dict[str, Any]], expected_count: int = CASE_COUNT
) -> bool:
    """Apply the noncompensatory all-planned-case carry-use veto."""
    return len(records) != expected_count or not all(
        record["carry_target_switch_exact"] for record in records
    )


def _fresh_reencoding_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    records = [case["field_screen"]["fresh_latest_reencoding"] for case in cases]
    metrics = (
        "nominal_second_exact",
        "adjacent_transition_of_emitted_state_exact",
        "carry_full_target_exact",
        "carry_output_switch",
        "carry_target_switch_exact",
        "result_full_target_exact",
        "result_target_switch_exact",
        "operand_full_target_exact",
        "operand_target_switch_exact",
        "carry_target_switch_recovery_vs_full_history_lf",
    )

    def summarize(subcases: list[dict[str, Any]]) -> dict[str, Any]:
        subrecords = [
            case["field_screen"]["fresh_latest_reencoding"] for case in subcases
        ]
        lf_records = [
            case["field_screen"]["by_clock"][DecodeMode.EOS_TO_LF.value]
            for case in subcases
        ]
        result: dict[str, Any] = {
            "cases": len(subrecords),
            "missing_or_malformed_failures": sum(
                not record["available"] for record in subrecords
            ),
        }
        for metric in metrics:
            count = sum(record[metric] for record in subrecords)
            result[metric] = {"count": count, "rate": rate(count, len(subrecords))}
        full_lf_count = sum(
            record["carry_target_switch_exact"] for record in lf_records
        )
        result["full_history_lf_carry_target_switch_exact"] = {
            "count": full_lf_count,
            "rate": rate(full_lf_count, len(lf_records)),
        }
        result["paired_switch_rate_delta_vs_full_history_lf"] = (
            result["carry_target_switch_exact"]["rate"]
            - result["full_history_lf_carry_target_switch_exact"]["rate"]
        )
        return result

    overall = summarize(cases)
    by_width = {
        str(width): summarize([case for case in cases if case["width"] == width])
        for width in (4, 6, 8)
    }
    gates = FRESH_REENCODING_GATES
    gate_checks = {
        "intact_adjacent_exact_overall": overall[
            "adjacent_transition_of_emitted_state_exact"
        ]["rate"]
        >= gates["intact_adjacent_exact_overall_min"],
        "intact_adjacent_exact_each_width": all(
            width["adjacent_transition_of_emitted_state_exact"]["rate"]
            >= gates["intact_adjacent_exact_each_width_min"]
            for width in by_width.values()
        ),
        "carry_full_target_exact_overall": overall["carry_full_target_exact"]["rate"]
        >= gates["carry_full_target_exact_overall_min"],
        "carry_full_target_exact_each_width": all(
            width["carry_full_target_exact"]["rate"]
            >= gates["carry_full_target_exact_each_width_min"]
            for width in by_width.values()
        ),
        "carry_output_switch_overall": overall["carry_output_switch"]["rate"]
        >= gates["carry_output_switch_overall_min"],
        "carry_output_switch_each_width": all(
            width["carry_output_switch"]["rate"]
            >= gates["carry_output_switch_each_width_min"]
            for width in by_width.values()
        ),
        "carry_paired_switch_overall": overall["carry_target_switch_exact"]["rate"]
        >= gates["carry_paired_switch_overall_min"],
        "carry_paired_switch_each_width": all(
            width["carry_target_switch_exact"]["rate"]
            >= gates["carry_paired_switch_each_width_min"]
            for width in by_width.values()
        ),
        "paired_recovery_overall": overall[
            "paired_switch_rate_delta_vs_full_history_lf"
        ]
        >= gates["paired_recovery_vs_full_history_lf_overall_min"],
        "paired_recovery_each_width": all(
            width["paired_switch_rate_delta_vs_full_history_lf"]
            >= gates["paired_recovery_vs_full_history_lf_each_width_min"]
            for width in by_width.values()
        ),
    }
    return {
        "cases": len(records),
        "overall": overall,
        "by_width": by_width,
        "gates": gates,
        "gate_checks": gate_checks,
        "compound_fresh_reencoding_screen_pass": all(gate_checks.values()),
    }


def aggregate_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    primary_by_arm: dict[str, Any] = {}
    for arm in PRIMARY_ARM_ORDER:
        records = [case["primary_arms"][arm] for case in cases]
        primary_by_arm[arm] = {
            "cases": len(records),
            "generated_tokens_total": sum(
                record["generated_token_count"] for record in records
            ),
            "generated_tokens_min": min(
                record["generated_token_count"] for record in records
            ),
            "generated_tokens_max": max(
                record["generated_token_count"] for record in records
            ),
            "model_selected_eos_events": sum(
                len(record["model_selected_eos_positions"]) for record in records
            ),
            "override_count": sum(record["override_count"] for record in records),
            "first_state_exact": sum(
                record["trace_score"]["first_state_exact"] for record in records
            ),
            "longest_exact_prefix_total": sum(
                record["trace_score"]["longest_exact_prefix"] for record in records
            ),
            "full_exact_trace": sum(
                record["trace_score"]["full_exact_trace_through_first_terminal"]
                for record in records
            ),
            "exact_final_tape": sum(
                record["trace_score"]["exact_final_tape"] for record in records
            ),
            "terminal_answer_exact": sum(
                record["trace_score"]["terminal_answer_exact"] for record in records
            ),
            "response_grammar_valid": sum(
                record["trace_score"]["response_grammar_valid"] for record in records
            ),
        }

    field_by_clock: dict[str, Any] = {}
    for clock_arm in FIELD_CLOCK_ARMS:
        records = [case["field_screen"]["by_clock"][clock_arm] for case in cases]
        metrics = (
            "nominal_second_exact",
            "adjacent_transition_of_emitted_state_exact",
            "carry_full_target_exact",
            "carry_target_switch_exact",
            "result_full_target_exact",
            "result_target_switch_exact",
            "operand_full_target_exact",
            "operand_target_switch_exact",
            "history_destruction_paired_loss",
        )
        summary = {
            "cases": len(records),
            "missing_or_malformed_failures": sum(
                record["missing_or_malformed_failure"] for record in records
            ),
        }
        for metric in metrics:
            count = sum(record[metric] for record in records)
            summary[metric] = {"count": count, "rate": rate(count, len(records))}
        by_true_first_carry = {}
        for carry in (0, 1):
            stratum = [
                record
                for case, record in zip(cases, records)
                if case["oracle"]["first_state_carry"] == carry
            ]
            switch_count = sum(
                record["carry_target_switch_exact"] for record in stratum
            )
            full_count = sum(record["carry_full_target_exact"] for record in stratum)
            by_true_first_carry[str(carry)] = {
                "cases": len(stratum),
                "carry_full_target_exact": {
                    "count": full_count,
                    "rate": rate(full_count, len(stratum)),
                },
                "carry_target_switch_exact": {
                    "count": switch_count,
                    "rate": rate(switch_count, len(stratum)),
                },
            }
        summary["by_true_first_carry"] = by_true_first_carry
        summary["carry_target_switch_noncompensatory_veto"] = carry_target_switch_veto(
            records
        )
        field_by_clock[clock_arm] = summary

    cell_counts = Counter((case["split"], case["operation"]) for case in cases)
    fresh_reencoding = _fresh_reencoding_summary(cases)
    replication_cases = [
        case
        for case_id in REPLICATION_CASE_IDS
        for case in cases
        if case["case_id"] == case_id
    ]
    if len(replication_cases) != len(REPLICATION_CASE_IDS):
        raise ContractError("frozen replication cases are missing")
    replication = _fresh_reencoding_summary(replication_cases)
    return {
        "case_count": len(cases),
        "primary_widths_4_6_count": sum(case["width"] in (4, 6) for case in cases),
        "width_8_extrapolation_count": sum(case["width"] == 8 for case in cases),
        "cell_counts": {
            f"{regime}:{operation}": cell_counts[(regime, operation)]
            for regime in EXPECTED_REGIMES
            for operation in OPERATIONS
        },
        "primary_by_arm": primary_by_arm,
        "field_by_clock": field_by_clock,
        "fresh_latest_reencoding": fresh_reencoding,
        "frozen_replication_12": {
            "ordered_case_ids": list(REPLICATION_CASE_IDS),
            "ordered_case_ids_sha256": REPLICATION_CASE_IDS_SHA256,
            "fresh_latest_reencoding": replication,
        },
        "carry_target_switch_global_veto": any(
            summary["carry_target_switch_noncompensatory_veto"]
            for summary in field_by_clock.values()
        ),
    }


def frozen_contract() -> dict[str, Any]:
    return {
        "input_sha256": dict(EXPECTED_SHA256),
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "case_selection": {
            "domain_hex": SELECTION_DOMAIN.hex(),
            "method": "per_cell_ascending_sha256(domain_nul_plus_ascii_case_id)_then_case_id",
            "regime_order": list(EXPECTED_REGIMES),
            "operation_order": list(OPERATIONS),
            "cases_per_cell": CASES_PER_CELL,
            "case_count": CASE_COUNT,
            "ordered_case_ids_sha256": ORDERED_CASE_IDS_SHA256,
            "primary_widths_4_6_count": 80,
            "width_8_extrapolation_count": 20,
            "replication_12": {
                "method": (
                    "within_frozen_100_per_width_operation_take_two_by_ascending_"
                    "sha256(selection_domain_plus_replication_nul_plus_case_id)"
                ),
                "ordered_case_ids": list(REPLICATION_CASE_IDS),
                "ordered_case_ids_sha256": REPLICATION_CASE_IDS_SHA256,
            },
        },
        "prompt": {
            "style": "core",
            "encoding": PROMPT_ENCODING,
            "prefix": PROMPT_PREFIX,
            "suffix": PROMPT_SUFFIX,
            "prefix_sha256": PROMPT_PREFIX_SHA256,
            "suffix_sha256": PROMPT_SUFFIX_SHA256,
            "template_sha256": PROMPT_TEMPLATE_SHA256,
            "exact_token_ids_and_canonical_token_id_hash_recorded": True,
        },
        "decode": {
            "greedy": True,
            "kv_cached": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "primary_arm_order": list(PRIMARY_ARM_ORDER),
            "fixed_budget_arms": sorted(FIXED_BUDGET_ARMS),
            "field_clock_arms": list(FIELD_CLOCK_ARMS),
            "history_branches": list(HISTORY_BRANCHES),
            "fresh_reencoding_branches": list(FRESH_REENCODING_BRANCHES),
            "eos_token_id": EOS_TOKEN_ID,
            "replacement_tokens": {
                arm: {"token_id": value[0], "text": value[1]}
                for arm, value in REPLACEMENT_TOKENS.items()
            },
            "ordinary_only_content_dependent_stop": "model_selected_eos",
            "online_parse_solver_or_semantic_stop": False,
        },
        "device": DEVICE,
        "precision": PRECISION,
        "allocation": {
            "partition": SLURM_PARTITION,
            "nodes": SLURM_NODE_COUNT,
            "tasks": SLURM_TASK_COUNT,
            "cpus_per_task": SLURM_CPUS_PER_TASK,
            "memory": SLURM_MEMORY,
            "memory_bytes": SLURM_MEMORY_BYTES,
            "time_limit": SLURM_TIME_LIMIT,
            "time_limit_seconds": SLURM_TIME_LIMIT_SECONDS,
            "requeue": False,
            "gres": SLURM_GRES,
            "gpu_type": SLURM_GPU_TYPE,
            "gpu_count": SLURM_GPU_COUNT,
            "scontrol_and_sacct_exact_identity_required": True,
            "typed_requested_and_allocated_tres_required": True,
            "device_cgroup_major_minor_required": True,
            "named_nvidia_control_devices_canonical_and_collision_free": True,
            "cuda_slurm_uuid_pci_one_to_one_required": True,
            "full_gpu_mig_disabled_required": True,
            "trust_root": "slurm_controller_accounting_and_device_cgroup",
        },
        "cuda_device": {
            "exact_name": REQUIRED_CUDA_DEVICE_NAME,
            "compute_capability": list(REQUIRED_CUDA_DEVICE_CAPABILITY),
            "total_memory_min_bytes": REQUIRED_CUDA_MEMORY_MIN_BYTES,
            "total_memory_max_bytes": REQUIRED_CUDA_MEMORY_MAX_BYTES,
            "visible_device_count": 1,
            "uuid_required": True,
            "cuda_visible_devices_required": True,
        },
        "determinism": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "python_startup_flags": ["-I", "-S", "-B"],
            "python_isolated_mode": True,
            "python_site_import_disabled": True,
            "python_dont_write_bytecode": "1",
            "python_path_environment_unset": True,
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "cuda_matmul_tf32_allowed": False,
            "cudnn_tf32_allowed": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
            "sdpa_backend": SDPA_BACKEND,
            "sdpa_exclusive": True,
            "bf16_probe_bitwise_equal_required": True,
        },
        "runtime_source_custody": {
            "manifest_schema": RUNTIME_SOURCE_MANIFEST_SCHEMA,
            "ordered_file_closure": list(RUNTIME_SOURCE_PATHS),
            "manifest_external_to_source_root": True,
            "manifest_single_link_no_write_bits": True,
            "source_commit_exact_lowercase_40_hex": True,
            "clean_git_and_git_show_byte_equality": True,
            "git_environment_config_fsmonitor_and_hooks_neutralized": True,
            "runtime_identity_schema": RUNTIME_IDENTITY_SCHEMA,
            "absolute_executable_path_hash_and_version": True,
            "package_version_import_hash_and_record_hash": True,
            "package_identity_rehashed_after_import": True,
            "actual_running_wrapper_bytes_equal_external_seal": True,
            "held_executable_and_manifest_descriptors": True,
            "source_execution": SOURCE_EXECUTION_MODE,
            "sealed_evaluator_and_model_memfd_exact_bytes": True,
            "generator_verifier_delegated_marker_ed25519_keys_distinct": True,
            "production_authority_public_key_source_pinned": True,
            "production_private_key_in_repository": False,
            "predecode_external_authorization_required": True,
            "authorization_binds_full_live_slurm_and_output_identity": True,
            "authorization_binds_replayable_linux_qualification_receipt": True,
            "delegated_marker_key_post_exec_brokered_sealed_memfd": True,
            "bash_receives_no_authority_bearing_secret": True,
            "bash_parent_replaced_by_exec_before_key_delivery": True,
            "python_startup_stdlib_pyc_and_search_surface_accounted": True,
            "preauthorization_maps_bind_mapped_device_inode_to_path": True,
            "delegated_marker_private_key_has_no_repository_or_runtime_path": True,
            "test_authority_scope_has_no_production_authority": True,
            "generator_receives_output_directory_descriptor": False,
            "verifier_receives_output_directory_descriptor": False,
            "wrapper_verification": (
                "before_generation_after_generation_after_independent_verifier"
            ),
            "evaluator_verification": (
                "before_cuda_after_generation_and_each_candidate_validation"
            ),
            "executed_runtime_coverage": (
                "loaded_file_snapshots_include_shared_objects_cuda_driver_libc_"
                "loader_and_preload_state_but_are_not_an_immutable_runtime_seal"
            ),
        },
        "publication": {
            "generator_output_state": PRIVATE_CANDIDATE_STATE,
            "generator_cannot_publish_accepted_path": True,
            "generator_output_transport": "anonymous_wrapper_owned_descriptor",
            "output_directory_current_uid_mode_octal": "0700",
            "stable_output_directory_fd_and_inode_custody": True,
            "same_directory_exclusive_temp": True,
            "temp_flush_and_fsync": True,
            "temp_readback_full_schema_validation": True,
            "private_candidate_mode_octal": "0400",
            "candidate_no_overwrite_atomic_rename": "renameat2_RENAME_NOREPLACE",
            "candidate_creation_descriptor_retained_through_consumption": True,
            "candidate_readonly_verifier_descriptor_bound_before_rename": True,
            "candidate_consumption_rename_only_retained_quarantine": True,
            "candidate_stat_mode_nlink_size_and_readback_validation": True,
            "independent_full_verifier_process": True,
            "signed_generator_report": "ed25519",
            "signed_independent_verifier_receipt": "distinct_ed25519_key",
            "signed_post_publication_marker": "delegated_ed25519_key",
            "signed_durable_acceptance_receipt": (
                "externally_authorized_delegated_ed25519_key_after_reopen_fsync"
            ),
            "live_slurm_and_actual_wrapper_identity_required": True,
            "wrapper_only_canonical_acceptance": True,
            "accepted_mode_octal": "0444",
            "accepted_no_overwrite_atomic_rename": "renameat2_RENAME_NOREPLACE",
            "accepted_parent_fsync_and_final_readback": True,
            "canonical_bundle_contains_completion_authority": False,
            "commit_marker_suffix": ACCEPTANCE_COMMIT_SUFFIX,
            "durable_acceptance_receipt_suffix": DURABLE_ACCEPTANCE_RECEIPT_SUFFIX,
            "commit_marker_created_after_canonical_final_checks": True,
            "receipt_slot_mode_octal": "0444",
            "receipt_slot_parent_fsynced_while_empty": True,
            "marker_pre_parent_fsync_pair_is_not_accepted": True,
            "report_marker_and_parent_reopened_fsynced_before_receipt_write": True,
            "replay_requires_bundle_marker_and_durable_acceptance_receipt": True,
            "accepted_receipt_bound_to_final_inode_and_report_hash": True,
            "post_rename_failure_retains_exact_inode_in_quarantine": True,
            "offline_receipt_replay_uses_public_keys": True,
            "marker_signature_chains_to_external_run_authorization": True,
            "receipt_signature_chains_to_external_run_authorization": True,
            "canonical_paths_reopened_after_marker_and_wrapper_checks": True,
            "replacement_race_requires_final_inventory_mismatch": True,
            "signed_stale_cleanup_is_rename_only_durable_quarantine": True,
            "stale_cleanup_never_unlinks_quarantine_pathnames": True,
            "quarantine_recovery_accepts_fresh_signed_exact_inode_record": True,
        },
        "linux_publication_qualification": {
            "schema": LINUX_QUALIFICATION_SCHEMA,
            "status": "required_not_executed_on_this_host",
            "requires_lustre": True,
            "requires_real_renameat2_memfd_prctl_and_directory_fsync": True,
            "receipt_protocol": "retained_o_sync_descriptor_after_required_fsyncs",
            "subprocess_crash_stages": [
                stage
                for stage, _exit_code, _expected_replay in LINUX_RECEIPT_CRASH_STAGES
            ],
            "independent_reopen_replay_after_each_child_exit": True,
            "only_complete_post_fsync_receipt_is_accepted": True,
            "first_child_evaluator_pathname_substitution_exercised": True,
            "replayable_authorization_receipt_schema": (
                LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_SCHEMA
            ),
            "publisher_lease_and_concurrent_process_flock_exercised": True,
            "ephemeral_mechanics_signature_stale_cleanup_exercised": True,
            "foreign_inode_and_directory_path_substitution_exercised": True,
            "scientific_decode": False,
        },
        "external_controller_ceiling": {
            "status": "descriptive_not_executed",
            "description": (
                "A future external controller may feed each oracle state into a fresh "
                "canonical one-step prompt. This evaluator does not execute that ceiling "
                "and no autonomous comparison may be inferred from it."
            ),
        },
        "fresh_latest_reencoding": {
            "mode": (
                "compound_fresh_core_prompt_latest_state_reencoding_canonicalization"
            ),
            "resource_boundary": (
                "compound_context_removal_position_reset_and_surface_canonicalization"
            ),
            "gates": dict(FRESH_REENCODING_GATES),
        },
        "posthoc_kv_slicing_negative_control": {
            "status": "descriptive_supplied_development_negative_not_reexecuted",
            "variants": [
                "latest_generated_state_kv_only",
                "immutable_prefix_plus_latest_state_kv",
                "drop_only_stale_s0_keys_keep_contextualized_suffix_and_latest",
            ],
            "boundary": (
                "Post-hoc KV slicing cannot decontextualize suffix or state representations "
                "already constructed with stale S0 attention. It is not a sufficient zero-"
                "weight test of a context-isolation mechanism."
            ),
        },
        "output_schema": OUTPUT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
    }


TOP_LEVEL_KEYS = {
    "schema",
    "protocol",
    "development_only",
    "claim_boundary",
    "frozen_contract",
    "execution",
    "aggregate",
    "cases",
    "adjudication",
    "wrapper_acceptance",
    "generator_attestation",
}
WRAPPER_ACCEPTANCE_KEYS = {
    "schema",
    "publication_state",
    "slurm_identity",
    "wrapper_sha256",
    "source_manifest_sha256",
    "runtime_identity_sha256",
    "nonce",
    "output_directory",
    "candidate_name",
    "accepted_name",
    "production_authority_key_file",
    "run_authorization_file",
    "run_authorization_sha256",
    "run_authorization",
    "generator_signing_key",
    "verifier_signing_key",
    "delegated_marker_signing_key",
    "sealed_generator",
}
OUTPUT_DIRECTORY_KEYS = {"path", "device", "inode", "uid", "mode"}
SLURM_IDENTITY_KEYS = {
    "job_id",
    "job_name",
    "job_state",
    "cluster_name",
    "user_name",
    "user_uid",
    "batch_flag",
    "command",
    "command_sha256",
    "batch_host",
    "node_list",
    "observed_hostname",
    "partition",
    "num_nodes",
    "num_cpus",
    "num_tasks",
    "cpus_per_task",
    "min_cpus_node",
    "min_memory_node",
    "memory_bytes",
    "time_limit",
    "time_limit_seconds",
    "requeue",
    "gres",
    "gpu_type",
    "gpu_count",
    "req_tres",
    "alloc_tres",
    "tres_per_node",
    "job_record_sha256",
    "cluster_config_sha256",
    "sacct_identity",
    "gpu_binding",
}
SACCT_IDENTITY_KEYS = {
    "job_id_raw",
    "job_name",
    "partition",
    "state",
    "alloc_cpus",
    "req_memory",
    "time_limit",
    "req_tres",
    "alloc_tres",
    "node_list",
    "record_sha256",
}
GPU_BINDING_KEYS = {
    "cgroup_version",
    "cgroup_path",
    "devices_list_path",
    "devices_list_sha256",
    "allowed_device_rules",
    "allocated_gpu_device",
    "pci_bus_id",
    "pci_sysfs_path",
    "pci_vendor_id",
    "pci_device_id",
    "pci_class_id",
    "pci_identity_sha256",
    "gpu_uuid",
    "gpu_name",
    "nvidia_smi_index",
    "nvidia_smi_minor_number",
    "gpu_minor",
    "gpu_major",
    "nvidia_control_devices",
    "concrete_physical_gpu_permissions",
    "mig_mode",
    "mig_devices_present",
    "nvidia_smi_query_sha256",
    "nvidia_smi_list_sha256",
    "cuda_visible_devices",
    "slurm_job_gpus",
    "selector_mapping",
}
SIGNING_KEY_RECORD_KEYS = {
    "descriptor_kind",
    "byte_count",
    "private_key_sha256",
    "public_key_hex",
    "seals",
}
EXTERNAL_FILE_RECORD_KEYS = {
    "path",
    "sha256",
    "device",
    "inode",
    "uid",
    "mode",
    "nlink",
    "size",
}
RUN_AUTHORIZATION_KEYS = {
    "schema",
    "authority_scope",
    "authority_key_id",
    "authority_public_key_sha256",
    "authorization_sequence",
    "authorization_nonce",
    "issued_at_utc",
    "not_before_utc",
    "expires_at_utc",
    "source_commit",
    "source_manifest_path",
    "source_manifest_sha256",
    "output_path",
    "output_directory",
    "accepted_name",
    "frozen_inputs",
    "ordered_case_ids_sha256",
    "slurm_allocation",
    "delegated_marker_public_key_hex",
    "delegated_marker_private_key_sha256",
    "delegated_publication_scopes",
    "linux_qualification_receipt",
    "stale_cleanup_entries",
    "signature_hex",
}
AUTHORIZED_FROZEN_INPUT_KEYS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "tokenizer_path",
    "tokenizer_sha256",
    "heldout_path",
    "heldout_sha256",
    "prereg_path",
    "prereg_sha256",
}
AUTHORIZED_SLURM_KEYS = set(SLURM_IDENTITY_KEYS)
STALE_CLEANUP_ENTRY_KEYS = {
    "name",
    "device",
    "inode",
    "uid",
    "mode",
    "nlink",
    "size",
    "sha256",
}
SEALED_GENERATOR_KEYS = {"evaluator_sha256", "model_sha256"}
GENERATOR_ATTESTATION_KEYS = {
    "schema",
    "scheme",
    "report_body_sha256",
    "generator_public_key_hex",
    "sealed_evaluator_sha256",
    "sealed_model_sha256",
    "wrapper_sha256",
    "source_manifest_sha256",
    "slurm_job_id",
    "nonce",
    "signature_hex",
}
CASE_KEYS = {
    "case_id",
    "split",
    "operation",
    "width",
    "selection_sha256",
    "initial_state",
    "prompt",
    "oracle",
    "primary_arms",
    "field_screen",
}
PRIMARY_RECORD_KEYS = {
    "mode",
    "decode_prompt_token_ids",
    "decode_prompt_token_ids_sha256",
    "decode_prompt_token_count",
    "generated_token_ids",
    "generated_token_count",
    "content_token_count",
    "generated_token_ids_sha256",
    "stop_reason",
    "fixed_budget",
    "eos_mask_applied_positions",
    "model_selected_eos_positions",
    "override_positions",
    "override_count",
    "eos_events",
    "response_text",
    "response_sha256",
    "trace_score",
}
PROMPT_RECORD_KEYS = {
    "utf8",
    "byte_count",
    "sha256",
    "token_ids",
    "token_count",
    "token_ids_sha256",
}
ORACLE_RECORD_KEYS = {
    "states",
    "trace_length",
    "first_state_carry",
    "final_tape",
    "answer",
    "heldout_binding",
}
HELDOUT_BINDING_KEYS = {
    "case_id",
    "row_sha256",
    "expected_states_sha256",
    "expected_answer",
}
EOS_EVENT_KEYS = {
    "generated_index",
    "absolute_token_position",
    "eos_logit",
    "non_eos_argmax_token_id",
    "non_eos_argmax_token_text",
    "non_eos_argmax_logit",
    "replacement_token_id",
    "replacement_token_text",
    "replacement_raw_logit",
}
DWS_LINE_RECORD_KEYS = {
    "line_index",
    "candidate_index",
    "text",
    "grammar_valid",
    "canonical_state",
    "repeated",
    "post_terminal",
    "exact_at_oracle_index",
}
ANSWER_LINE_RECORD_KEYS = {"line_index", "text", "value"}
TRACE_SCORE_KEYS = {
    "dws_lines",
    "dws_candidate_count",
    "valid_dws_line_count",
    "malformed_dws_line_indices",
    "repeated_dws_line_indices",
    "post_terminal_dws_line_indices",
    "nonprotocol_line_indices",
    "answer_lines",
    "longest_exact_prefix",
    "oracle_trace_length",
    "first_state_exact",
    "full_exact_trace_through_first_terminal",
    "first_terminal_line_index",
    "exact_terminal_state",
    "observed_final_tape",
    "oracle_final_tape",
    "exact_final_tape",
    "terminal_answer",
    "oracle_answer",
    "terminal_answer_exact",
    "generated_answer_values",
    "generated_answer_exact",
    "all_dws_lines_canonical",
    "transition_grammar_valid",
    "response_grammar_valid",
}
FIELD_SCREEN_KEYS = {
    "available",
    "failure",
    "emitted_state",
    "emitted_history_token_count",
    "by_clock",
    "fresh_latest_reencoding",
}
CLOCK_SCORE_KEYS = {
    "clock_arm",
    "missing_or_malformed_failure",
    "nominal_second_exact",
    "adjacent_transition_of_emitted_state_exact",
    "carry_full_target_exact",
    "carry_target_switch_exact",
    "result_full_target_exact",
    "result_target_switch_exact",
    "operand_full_target_exact",
    "operand_target_switch_exact",
    "history_destruction_paired_loss",
    "detail",
}
COMMON_DECODE_REPORT_KEYS = PRIMARY_RECORD_KEYS - {"trace_score"}
FIELD_BRANCH_REPORT_KEYS = COMMON_DECODE_REPORT_KEYS | {
    "first_line_serialization_valid",
    "observed_first_state",
    "target_state",
    "full_target_exact",
}
FIELD_DETAIL_KEYS = {
    "clock_arm",
    "boundary_token_id",
    "boundary_token_text",
    "nominal_second_exact",
    "adjacent_transition_of_emitted_state_exact",
    "intact_target_state",
    "branch_contracts",
    "branch_reports",
    "interventions",
    "history_destruction",
}
BRANCH_CONTRACT_KEYS = {"state_line", "history_token_count", "metadata"}
INTERVENTION_SCORE_KEYS = {
    "metadata",
    "target_state",
    "full_target_exact",
    "whole_output_changed",
    "endpoints",
}
ENDPOINT_SCORE_KEYS = {
    "endpoint",
    "intact_target",
    "counterfactual_target",
    "intact_observed",
    "counterfactual_observed",
    "target_changed",
    "output_changed",
    "intact_target_exact",
    "counterfactual_target_exact",
    "paired_target_switch_exact",
}
FRESH_REENCODING_KEYS = {
    "available",
    "failure",
    "mode",
    "external_reencoding",
    "nominal_second_exact",
    "adjacent_transition_of_emitted_state_exact",
    "carry_full_target_exact",
    "carry_output_switch",
    "carry_target_switch_exact",
    "result_full_target_exact",
    "result_target_switch_exact",
    "operand_full_target_exact",
    "operand_target_switch_exact",
    "carry_target_switch_recovery_vs_full_history_lf",
    "detail",
}
FRESH_REENCODING_DETAIL_KEYS = {
    "mode",
    "external_reencoding",
    "nominal_second_exact",
    "adjacent_transition_of_emitted_state_exact",
    "carry_full_target_exact",
    "carry_output_switch",
    "carry_target_switch_exact",
    "result_full_target_exact",
    "result_target_switch_exact",
    "operand_full_target_exact",
    "operand_target_switch_exact",
    "carry_target_switch_recovery_vs_full_history_lf",
    "branch_reports",
    "interventions",
}
RUNTIME_CUSTODY_KEYS = {
    "schema",
    "path",
    "sha256",
    "manifest_file",
    "source_root",
    "source_commit",
    "files",
    "runtime",
    "git_status",
    "git_show_byte_equality",
}
MANIFEST_FILE_KEYS = {"device", "inode", "uid", "mode", "size"}
RUNTIME_IDENTITY_KEYS = {
    "schema",
    "python",
    "git",
    "scontrol",
    "sacct",
    "nvidia_smi",
    "python_startup",
    "packages",
    "backend",
}
EXECUTABLE_IDENTITY_KEYS = {"path", "sha256", "version"}
PYTHON_STARTUP_IDENTITY_KEYS = {
    "mode",
    "flags",
    "startup_environment",
    "site_modules_loaded",
    "processed_pth_files",
    "module_origins",
    "components",
    "search_path",
    "closure_sha256",
}
PYTHON_STARTUP_FLAG_KEYS = {
    "isolated",
    "no_site",
    "no_user_site",
    "ignore_environment",
    "dont_write_bytecode",
    "safe_path",
}
PYTHON_STARTUP_ENVIRONMENT_KEYS = {
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONINSPECT",
    "PYTHONWARNINGS",
}
PYTHON_MODULE_ORIGIN_KEYS = {"name", "origin", "file", "cached"}
PYTHON_SEARCH_PATH_KEYS = {
    "path",
    "kind",
    "device",
    "inode",
    "uid",
    "mode",
    "ancestor",
    "sha256",
    "byte_count",
}
DISTRIBUTION_IDENTITY_KEYS = {
    "distribution_name",
    "version",
    "distribution_root",
    "installation_root",
    "module_path",
    "record_path",
    "file_count",
    "files",
    "closure_sha256",
}
DISTRIBUTION_FILE_KEYS = {
    "relative_path",
    "path",
    "sha256",
    "byte_count",
    "device",
    "inode",
    "uid",
    "mode",
    "nlink",
}
NATIVE_LIBRARY_CLOSURE_KEYS = {"platform", "source", "files", "closure_sha256"}
NATIVE_LIBRARY_FILE_KEYS = DISTRIBUTION_FILE_KEYS - {"relative_path"} | {
    "mapped_device_major",
    "mapped_device_minor",
    "mapped_inode",
}
RUNTIME_BACKEND_KEYS = {
    "device",
    "precision",
    "sdpa_backend",
    "cublas_workspace_config",
    "deterministic_algorithms",
    "ld_preload",
    "dyld_insert_libraries",
    "ld_library_path",
    "preauthorization_native_libraries",
    "coverage_claim",
}
SOURCE_EXECUTION_KEYS = {
    "mode",
    "python_startup_mode",
    "evaluator",
    "model",
}
SEALED_SOURCE_DESCRIPTOR_KEYS = {
    "descriptor_kind",
    "sha256",
    "byte_count",
    "seals",
}
RUNTIME_OBSERVATION_KEYS = {
    "schema",
    "phase",
    "coverage",
    "python_executable",
    "platform",
    "libc_version",
    "ld_preload",
    "dyld_insert_libraries",
    "ld_library_path",
    "mapping_source_sha256",
    "loaded_files",
    "loaded_files_sha256",
    "shared_objects",
    "shared_objects_sha256",
    "libc_objects",
    "loader_objects",
    "cuda_objects",
    "cuda_driver_version",
    "cuda_runtime_version",
    "cuda_visible_devices",
    "cuda_device_name",
    "cuda_device_capability",
    "cuda_device_total_memory_bytes",
    "cuda_device_uuid",
    "nvidia_driver_file",
}
RUNTIME_FILE_KEYS = {
    "path",
    "sha256",
    "device",
    "inode",
    "size",
    "mapping_identity",
}
EXECUTION_KEYS = {
    "started_at_utc",
    "finished_at_utc",
    "input_paths",
    "verified_input_sha256",
    "checkpoint_step",
    "ordered_case_ids",
    "runtime_source_manifest",
    "source_execution",
    "runtime_observation",
    "device",
    "precision",
    "python",
    "torch",
    "cuda_runtime",
    "cuda_visible_devices",
    "device_name",
    "device_capability",
    "device_total_memory_bytes",
    "device_uuid",
    "visible_cuda_device_count",
    "cublas_workspace_config",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "cuda_matmul_tf32_allowed",
    "cudnn_tf32_allowed",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "float32_matmul_precision",
    "sdpa_backend",
    "sdpa_math_enabled",
    "sdpa_flash_enabled",
    "sdpa_mem_efficient_enabled",
    "sdpa_cudnn_enabled",
    "sdpa_bf16_probe_bitwise_equal",
    "seed",
}
ACCEPTED_BUNDLE_KEYS = {"schema", "report"}
FINAL_INODE_KEYS = {"device", "inode", "uid", "mode", "nlink", "size"}
COMMIT_MARKER_INODE_KEYS = {"device", "inode", "uid"}
FROZEN_INPUT_OBSERVATION_KEYS = {"paths", "sha256"}
INDEPENDENT_VERIFIER_KEYS = {
    "schema",
    "candidate_sha256",
    "candidate_inode",
    "report_body_sha256",
    "generator_attestation_sha256",
    "validation_runtime_observation",
    "validation_runtime_observation_sha256",
    "frozen_input_observation",
    "verifier_public_key_hex",
    "signature_hex",
}
CANDIDATE_INODE_KEYS = {"device", "inode", "uid", "mode", "nlink", "size"}
ACCEPTANCE_MARKER_KEYS = {
    "schema",
    "status",
    "committed_at_utc",
    "commit_nonce",
    "accepted_name",
    "commit_marker_name",
    "output_directory",
    "final_inode",
    "accepted_bundle_sha256",
    "commit_marker_inode",
    "report_sha256",
    "report_body_sha256",
    "generator_attestation_sha256",
    "independent_verifier",
    "independent_verifier_sha256",
    "slurm_identity",
    "wrapper_sha256",
    "source_manifest_sha256",
    "runtime_identity_sha256",
    "frozen_input_observation",
    "runtime_coverage_boundary",
    "post_publication_checks",
    "run_authorization_sha256",
    "authorization_sequence",
    "authority_key_id",
    "authority_public_key_sha256",
    "delegated_marker_public_key_hex",
    "delegated_publication_scope",
    "signature_hex",
}
POST_PUBLICATION_MARKER_CHECK_KEYS = {
    "canonical_rename_complete",
    "canonical_parent_fsync_complete",
    "canonical_readback_complete",
    "canonical_full_replay_complete",
    "wrapper_pre_marker_checks_complete",
}
DURABLE_ACCEPTANCE_RECEIPT_INODE_KEYS = {"device", "inode", "uid"}
DURABLE_ACCEPTANCE_RECEIPT_KEYS = {
    "schema",
    "status",
    "witnessed_at_utc",
    "witness_nonce",
    "accepted_name",
    "commit_marker_name",
    "durable_acceptance_receipt_name",
    "output_directory",
    "final_inode",
    "commit_marker_inode",
    "durable_acceptance_receipt_inode",
    "accepted_bundle_sha256",
    "commit_marker_sha256",
    "report_sha256",
    "run_authorization_sha256",
    "authorization_sequence",
    "authority_key_id",
    "authority_public_key_sha256",
    "delegated_marker_public_key_hex",
    "delegated_publication_scope",
    "durability_checks",
    "signature_hex",
}
DURABLE_ACCEPTANCE_CHECK_KEYS = {
    "receipt_slot_parent_fsync_complete",
    "canonical_report_reopen_fsync_complete",
    "commit_marker_reopen_fsync_complete",
    "publication_parent_fsync_complete",
    "wrapper_final_checks_complete",
    "receipt_o_sync_write_complete",
}
LINUX_QUALIFICATION_REPORT_KEYS = {
    "schema",
    "stage",
    "evaluator_source",
    "broker_request_sha256",
    "brokered_signing_key",
    "authority_boundary",
}
LINUX_QUALIFICATION_MARKER_KEYS = {
    "schema",
    "stage",
    "report_name",
    "report_sha256",
    "evaluator_sha256",
    "broker_request_sha256",
    "brokered_public_key_hex",
    "authority_boundary",
    "signature_hex",
}
LINUX_QUALIFICATION_RECEIPT_KEYS = {
    "schema",
    "status",
    "stage",
    "report_name",
    "marker_name",
    "receipt_name",
    "evaluator_source",
    "report_sha256",
    "marker_sha256",
    "report_inode",
    "marker_inode",
    "receipt_inode",
    "durability_checks",
    "broker_request_sha256",
    "brokered_public_key_hex",
    "authority_boundary",
    "signature_hex",
}
LINUX_QUALIFICATION_RECEIPT_CHECK_KEYS = {
    "report_file_fsync_complete",
    "marker_file_fsync_complete",
    "receipt_slot_file_fsync_complete",
    "receipt_slot_parent_fsync_complete",
    "report_held_descriptor_fsync_complete",
    "marker_held_descriptor_fsync_complete",
    "publication_parent_fsync_complete",
    "receipt_o_sync_write_complete",
}
LINUX_QUALIFICATION_RESULT_KEYS = {
    "schema",
    "scientific_decode_executed",
    "gpu_required",
    "filesystem",
    "evaluator_sha256",
    "checks",
    "qualification_evidence",
    "receipt_crash_cases",
    "status",
    "claim_boundary",
}
LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_KEYS = {
    "schema",
    "qualification_result",
    "accepted_publication",
    "signature_hex",
}
LINUX_QUALIFICATION_ACCEPTED_PUBLICATION_KEYS = {
    "broker_request",
    "report",
    "marker",
    "receipt",
}
LINUX_QUALIFICATION_CRASH_CASE_KEYS = {
    "stage",
    "child_exit_code",
    "expected_replay",
    "observed_replay",
    "independent_report_marker_replay",
    "receipt_size",
    "evaluator_source",
    "broker_request_sha256",
}
LINUX_QUALIFICATION_EVIDENCE_KEYS = {
    "delegated_key_broker_transfer",
    "publisher_lease",
    "signed_stale_cleanup",
    "foreign_inode_substitution",
    "held_evaluator_path_substitution",
    "directory_path_substitution",
}
LINUX_QUALIFICATION_EVALUATOR_SUBSTITUTION_KEYS = {
    "source_name",
    "retained_name",
    "replacement_name",
    "source_inode",
    "retained_inode",
    "replacement_inode",
    "held_sha256",
    "retained_sha256",
    "replacement_sha256",
    "substitution_before_first_child",
    "original_inode_retained",
}


def _require_exact_keys(value: Any, keys: set[str], path: str) -> None:
    _require_plain_json_tree(value, path)
    if type(value) is not dict or set(value) != keys:
        observed = sorted(value) if type(value) is dict else type(value).__name__
        raise ContractError(f"schema keys differ at {path}: {observed}")


def _require_plain_json_tree(value: Any, path: str = "value") -> None:
    """Reject Python subclass and numeric-alias ambiguity before semantic replay."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"non-finite JSON number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_plain_json_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ContractError(f"non-string JSON object key at {path}")
            _require_plain_json_tree(item, f"{path}.{key}")
        return
    raise ContractError(f"non-plain JSON value at {path}: {type(value).__name__}")


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_nonnegative_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"expected a nonnegative integer at {path}")
    return value


def _require_finite_number(value: Any, path: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ContractError(f"expected a finite float at {path}")
    return value


def _require_utc_timestamp(value: Any, path: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContractError(f"expected a nonempty UTC timestamp at {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractError(f"invalid timestamp at {path}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"timestamp is not UTC at {path}")
    return parsed


def _validate_decode_common(
    record: dict[str, Any],
    path: str,
    *,
    expected_mode: str | None = None,
    expected_prompt_token_count: int | None = None,
    expected_prompt_token_ids: tuple[int, ...] | None = None,
    tokenizer: Any | None = None,
) -> None:
    mode = record["mode"]
    if mode not in PRIMARY_ARM_ORDER:
        raise ContractError(f"unknown decode mode at {path}")
    if expected_mode is not None and mode != expected_mode:
        raise ContractError(f"decode arm identity mismatch at {path}")
    prompt_token_count = _require_nonnegative_int(
        record["decode_prompt_token_count"], f"{path}.decode_prompt_token_count"
    )
    if prompt_token_count == 0:
        raise ContractError(f"decode prompt is empty at {path}")
    if (
        expected_prompt_token_count is not None
        and prompt_token_count != expected_prompt_token_count
    ):
        raise ContractError(f"decode prompt-token count mismatch at {path}")
    if type(record["decode_prompt_token_ids"]) is not list:
        raise ContractError(f"decode prompt token IDs are not a list at {path}")
    prompt_ids = record["decode_prompt_token_ids"]
    if len(prompt_ids) != prompt_token_count or record[
        "decode_prompt_token_ids_sha256"
    ] != token_ids_sha256(prompt_ids):
        raise ContractError(f"decode prompt token identity mismatch at {path}")
    if expected_prompt_token_ids is not None and tuple(prompt_ids) != tuple(
        expected_prompt_token_ids
    ):
        raise ContractError(f"decode prompt token sequence mismatch at {path}")
    if type(record["generated_token_ids"]) is not list:
        raise ContractError(f"generated token IDs are not a list at {path}")
    generated_count = _require_nonnegative_int(
        record["generated_token_count"], f"{path}.generated_token_count"
    )
    if record["generated_token_count"] != len(record["generated_token_ids"]):
        raise ContractError(f"generated-token accounting mismatch at {path}")
    if record["generated_token_ids_sha256"] != token_ids_sha256(
        record["generated_token_ids"]
    ):
        raise ContractError(f"generated-token hash mismatch at {path}")
    content_count = _require_nonnegative_int(
        record["content_token_count"], f"{path}.content_token_count"
    )
    override_count = _require_nonnegative_int(
        record["override_count"], f"{path}.override_count"
    )
    if type(record["override_positions"]) is not list:
        raise ContractError(f"override positions are not a list at {path}")
    for index, position in enumerate(record["override_positions"]):
        _require_nonnegative_int(position, f"{path}.override_positions[{index}]")
    if record["override_count"] != len(record["override_positions"]):
        raise ContractError(f"override accounting mismatch at {path}")
    if not isinstance(record["response_text"], str):
        raise ContractError(f"response text is not a string at {path}")
    if record["response_sha256"] != sha256_bytes(
        record["response_text"].encode("utf-8")
    ):
        raise ContractError(f"response hash mismatch at {path}")
    if type(record["eos_mask_applied_positions"]) is not list:
        raise ContractError(f"EOS mask positions are not a list at {path}")
    for index, position in enumerate(record["eos_mask_applied_positions"]):
        _require_nonnegative_int(
            position, f"{path}.eos_mask_applied_positions[{index}]"
        )
    if tokenizer is not None:
        content_ids = record["generated_token_ids"]
        if record["stop_reason"] == "eos":
            content_ids = content_ids[:-1]
        decoded = tokenizer.decode(content_ids, skip_special_tokens=False)
        if decoded != record["response_text"]:
            raise ContractError(f"response text does not replay from tokens at {path}")
    if type(record["eos_events"]) is not list:
        raise ContractError(f"EOS events are not a list at {path}")
    for event_index, event in enumerate(record["eos_events"]):
        _require_exact_keys(event, EOS_EVENT_KEYS, f"{path}.eos_events[{event_index}]")
    event_positions = [event["generated_index"] for event in record["eos_events"]]
    if event_positions != sorted(set(event_positions)):
        raise ContractError(
            f"EOS event positions are not strictly increasing at {path}"
        )
    if type(record["model_selected_eos_positions"]) is not list:
        raise ContractError(f"model-selected EOS positions are not a list at {path}")
    for index, position in enumerate(record["model_selected_eos_positions"]):
        _require_nonnegative_int(
            position, f"{path}.model_selected_eos_positions[{index}]"
        )
    if record["model_selected_eos_positions"] != event_positions:
        raise ContractError(f"EOS-position accounting mismatch at {path}")
    if record["override_positions"] != [
        event["generated_index"]
        for event in record["eos_events"]
        if event["replacement_token_id"] is not None
    ]:
        raise ContractError(f"override-position accounting mismatch at {path}")
    fixed_replacement = replacement_token_id(DecodeMode(mode))
    for event_index, event in enumerate(record["eos_events"]):
        event_path = f"{path}.eos_events[{event_index}]"
        generated_index = _require_nonnegative_int(
            event["generated_index"], f"{event_path}.generated_index"
        )
        if generated_index >= generated_count:
            raise ContractError(
                f"EOS event lies outside generated tokens at {event_path}"
            )
        absolute_position = _require_nonnegative_int(
            event["absolute_token_position"],
            f"{event_path}.absolute_token_position",
        )
        if absolute_position != prompt_token_count + generated_index:
            raise ContractError(f"absolute EOS position mismatch at {event_path}")
        _require_finite_number(event["eos_logit"], f"{event_path}.eos_logit")
        non_eos_id = _require_nonnegative_int(
            event["non_eos_argmax_token_id"],
            f"{event_path}.non_eos_argmax_token_id",
        )
        if non_eos_id == EOS_TOKEN_ID:
            raise ContractError(f"non-EOS argmax is EOS at {event_path}")
        _require_finite_number(
            event["non_eos_argmax_logit"],
            f"{event_path}.non_eos_argmax_logit",
        )
        if not isinstance(event["non_eos_argmax_token_text"], str):
            raise ContractError(f"non-EOS token text is invalid at {event_path}")
        if tokenizer is not None and event["non_eos_argmax_token_text"] != (
            tokenizer.decode([non_eos_id], skip_special_tokens=False)
        ):
            raise ContractError(f"non-EOS token text does not replay at {event_path}")
        replacement_id = event["replacement_token_id"]
        if mode == DecodeMode.ORDINARY_EOS_STOP.value:
            if any(
                event[key] is not None
                for key in (
                    "replacement_token_id",
                    "replacement_token_text",
                    "replacement_raw_logit",
                )
            ):
                raise ContractError(
                    f"ordinary EOS event was overridden at {event_path}"
                )
        else:
            _require_nonnegative_int(
                replacement_id, f"{event_path}.replacement_token_id"
            )
            expected_replacement = (
                non_eos_id
                if mode == DecodeMode.EOS_MASKED_ARGMAX.value
                else fixed_replacement
            )
            if replacement_id != expected_replacement:
                raise ContractError(
                    f"EOS replacement identity mismatch at {event_path}"
                )
            if record["generated_token_ids"][generated_index] != replacement_id:
                raise ContractError(
                    f"EOS replacement token was not fed at {event_path}"
                )
            _require_finite_number(
                event["replacement_raw_logit"],
                f"{event_path}.replacement_raw_logit",
            )
            if mode == DecodeMode.EOS_MASKED_ARGMAX.value:
                if (
                    event["replacement_token_text"]
                    != event["non_eos_argmax_token_text"]
                    or event["replacement_raw_logit"] != event["non_eos_argmax_logit"]
                ):
                    raise ContractError(
                        f"masked-EOS runner-up evidence mismatch at {event_path}"
                    )
            else:
                expected_text = REPLACEMENT_TOKENS[mode][1]
                if event["replacement_token_text"] != expected_text:
                    raise ContractError(
                        f"fixed replacement text mismatch at {event_path}"
                    )
            if tokenizer is not None and event["replacement_token_text"] != (
                tokenizer.decode([replacement_id], skip_special_tokens=False)
            ):
                raise ContractError(
                    f"replacement token text does not replay at {event_path}"
                )

    if mode == DecodeMode.ORDINARY_EOS_STOP.value:
        if record["fixed_budget"] is not False:
            raise ContractError(f"ordinary decode is labeled fixed-budget at {path}")
        if record["eos_mask_applied_positions"] != []:
            raise ContractError(
                f"ordinary decode contains EOS mask positions at {path}"
            )
        if record["override_positions"] != [] or record["override_count"] != 0:
            raise ContractError(f"ordinary decode contains overrides at {path}")
        if record["stop_reason"] == "eos":
            if (
                generated_count == 0
                or generated_count > MAX_NEW_TOKENS
                or record["generated_token_ids"][-1] != EOS_TOKEN_ID
                or EOS_TOKEN_ID in record["generated_token_ids"][:-1]
                or event_positions != [generated_count - 1]
                or content_count != generated_count - 1
            ):
                raise ContractError(f"ordinary EOS-stop accounting mismatch at {path}")
        elif record["stop_reason"] == "max_new_tokens":
            if (
                generated_count != MAX_NEW_TOKENS
                or event_positions
                or EOS_TOKEN_ID in record["generated_token_ids"]
                or content_count != generated_count
            ):
                raise ContractError(f"ordinary cap-stop accounting mismatch at {path}")
        else:
            raise ContractError(f"ordinary decode has an invalid stop reason at {path}")
    else:
        if (
            record["fixed_budget"] is not True
            or record["stop_reason"] != "fixed_budget"
            or generated_count != MAX_NEW_TOKENS
            or content_count != generated_count
            or EOS_TOKEN_ID in record["generated_token_ids"]
        ):
            raise ContractError(f"fixed-budget stop/accounting mismatch at {path}")
        expected_masks = (
            list(range(MAX_NEW_TOKENS))
            if mode == DecodeMode.EOS_MASKED_ARGMAX.value
            else []
        )
        if record["eos_mask_applied_positions"] != expected_masks:
            raise ContractError(f"EOS mask positions mismatch at {path}")
        if record["override_positions"] != event_positions:
            raise ContractError(f"fixed-budget override positions mismatch at {path}")
        if override_count != len(event_positions):
            raise ContractError(f"fixed-budget override count mismatch at {path}")


def _validate_interventions(value: dict[str, Any], path: str) -> None:
    endpoint_sets = {
        "carry_flip": {"active_result_digit", "next_carry"},
        "written_result_r0_flip": {"written_result_r0"},
        "active_operand_digit_perturbation": {
            "active_result_digit",
            "active_operand_digit",
        },
    }
    if not isinstance(value, dict) or set(value) != set(endpoint_sets):
        raise ContractError(f"intervention set differs at {path}")
    for name, intervention in value.items():
        _require_exact_keys(intervention, INTERVENTION_SCORE_KEYS, f"{path}.{name}")
        if (
            not isinstance(intervention["endpoints"], dict)
            or set(intervention["endpoints"]) != endpoint_sets[name]
        ):
            raise ContractError(f"endpoint set differs at {path}.{name}")
        for endpoint, score in intervention["endpoints"].items():
            _require_exact_keys(
                score, ENDPOINT_SCORE_KEYS, f"{path}.{name}.endpoints.{endpoint}"
            )


def _validate_branch_contracts(
    contracts: dict[str, Any],
    emitted_state: dict[str, Any],
    emitted_history_token_count: int,
    tokenizer: Any,
    path: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(contracts, dict) or set(contracts) != set(HISTORY_BRANCHES):
        raise ContractError(f"field branch contracts differ at {path}")
    carry = dict(emitted_state)
    carry["c"] = 1 - carry["c"]
    result = dict(emitted_state)
    result_tape = list(result["r"])
    result_tape[0] = _flip_digit(result_tape[0])
    result["r"] = "".join(result_tape)
    operand, operand_metadata = active_operand_branch_posthoc(emitted_state)
    expected = {
        "intact": (dict(emitted_state), {"intervention": "none"}),
        "carry_flip": (
            carry,
            {"field": "c", "from": emitted_state["c"], "to": carry["c"]},
        ),
        "written_result_r0_flip": (
            result,
            {
                "field": "r[0]",
                "from": emitted_state["r"][0],
                "to": result["r"][0],
            },
        ),
        "active_operand_digit_perturbation": (operand, operand_metadata),
    }
    reconstructed: dict[str, dict[str, Any]] = {}
    for name in HISTORY_BRANCHES:
        contract = contracts[name]
        contract_path = f"{path}.{name}"
        _require_exact_keys(contract, BRANCH_CONTRACT_KEYS, contract_path)
        token_count = _require_nonnegative_int(
            contract["history_token_count"], f"{contract_path}.history_token_count"
        )
        if name == "equal_token_length_destroyed_history":
            expected_metadata = {
                "intervention": "replace_each_emitted_history_token_with_id_100",
                "source_token_count": emitted_history_token_count,
            }
            if (
                contract["state_line"] is not None
                or token_count != emitted_history_token_count
                or contract["metadata"] != expected_metadata
            ):
                raise ContractError(
                    f"destroyed-history contract differs at {contract_path}"
                )
            reconstructed[name] = {
                "state": None,
                "state_line": None,
                "history_token_count": token_count,
                "metadata": contract["metadata"],
            }
            continue
        expected_state, expected_metadata = expected[name]
        expected_line = canonical_dws_state(expected_state)
        expected_token_count = len(tokenizer.encode(expected_line).ids)
        if (
            token_count != expected_token_count
            or (name == "intact" and token_count != emitted_history_token_count)
            or contract["state_line"] != expected_line
            or contract["metadata"] != expected_metadata
        ):
            raise ContractError(f"semantic branch contract differs at {contract_path}")
        reconstructed[name] = {
            "state": expected_state,
            "state_line": expected_line,
            "history_token_count": token_count,
            "metadata": contract["metadata"],
        }
    return reconstructed


def _validate_branch_reports(
    reports: dict[str, Any],
    expected_branches: tuple[str, ...],
    path: str,
    *,
    expected_mode: str,
    branches: dict[str, dict[str, Any]],
    tokenizer: Any,
    expected_prompt_token_counts: dict[str, int] | None = None,
    expected_prompt_token_ids: dict[str, tuple[int, ...]] | None = None,
) -> None:
    if type(reports) is not dict or set(reports) != set(expected_branches):
        raise ContractError(f"branch set differs at {path}")
    for name, branch in reports.items():
        branch_path = f"{path}.{name}"
        _require_exact_keys(branch, FIELD_BRANCH_REPORT_KEYS, branch_path)
        prompt_count = (
            expected_prompt_token_counts[name]
            if expected_prompt_token_counts is not None
            else None
        )
        _validate_decode_common(
            branch,
            branch_path,
            expected_mode=expected_mode,
            expected_prompt_token_count=prompt_count,
            expected_prompt_token_ids=(
                expected_prompt_token_ids[name]
                if expected_prompt_token_ids is not None
                else None
            ),
            tokenizer=tokenizer,
        )
        observed = first_adjacent_state_posthoc(branch["response_text"])
        observed_line = canonical_dws_state(observed) if observed is not None else None
        target_state = branches[name]["state"]
        target = (
            apply_microstep_posthoc(target_state) if target_state is not None else None
        )
        target_line = canonical_dws_state(target) if target is not None else None
        if branch["first_line_serialization_valid"] is not (observed is not None):
            raise ContractError(
                f"first-line serialization flag mismatch at {branch_path}"
            )
        if branch["observed_first_state"] != observed_line:
            raise ContractError(f"observed first state mismatch at {branch_path}")
        if branch["target_state"] != target_line:
            raise ContractError(f"branch target state mismatch at {branch_path}")
        if branch["full_target_exact"] is not bool(
            observed is not None and target is not None and observed == target
        ):
            raise ContractError(f"branch full-target score mismatch at {branch_path}")


def _validate_python_startup_identity(value: dict[str, Any]) -> None:
    _require_exact_keys(
        value, PYTHON_STARTUP_IDENTITY_KEYS, "pre-authorization Python startup"
    )
    if value["mode"] != PYTHON_STARTUP_MODE:
        raise ContractError("pre-authorization Python startup mode differs")
    flags = value["flags"]
    _require_exact_keys(flags, PYTHON_STARTUP_FLAG_KEYS, "Python startup flags")
    for name in PYTHON_STARTUP_FLAG_KEYS:
        if type(flags[name]) is not bool:
            raise ContractError("Python startup flag type differs")
    if any(
        flags[name] is not True
        for name in (
            "isolated",
            "no_site",
            "no_user_site",
            "ignore_environment",
            "dont_write_bytecode",
        )
    ):
        raise ContractError("Python startup was not sealed with -I -S -B")
    startup_environment = value["startup_environment"]
    _require_exact_keys(
        startup_environment,
        PYTHON_STARTUP_ENVIRONMENT_KEYS,
        "Python startup environment",
    )
    if any(item is not None for item in startup_environment.values()):
        raise ContractError("Python startup environment was not neutralized")
    if value["site_modules_loaded"] != [] or value["processed_pth_files"] != []:
        raise ContractError("Python site/customization processing was not disabled")

    origins = value["module_origins"]
    if not isinstance(origins, list) or not origins:
        raise ContractError("pre-authorization Python module closure is empty")
    previous_name = None
    referenced_components: set[str] = set()
    for origin in origins:
        _require_exact_keys(
            origin, PYTHON_MODULE_ORIGIN_KEYS, "pre-authorization Python module"
        )
        name = origin["name"]
        if (
            not isinstance(name, str)
            or not name
            or (previous_name is not None and name <= previous_name)
        ):
            raise ContractError("pre-authorization Python module order differs")
        previous_name = name
        for field in ("origin", "file", "cached"):
            reference = origin[field]
            if reference is not None and not isinstance(reference, str):
                raise ContractError("pre-authorization Python module origin differs")
            if (
                isinstance(reference, str)
                and reference
                and reference not in {"built-in", "frozen"}
                and not reference.startswith("<")
                and not reference.startswith("sealed-memfd:")
            ):
                referenced_components.add(reference)

    components = value["components"]
    if not isinstance(components, list) or not components:
        raise ContractError("pre-authorization Python file closure is empty")
    component_references = []
    for component in components:
        kind = component.get("kind") if isinstance(component, dict) else None
        keys = {
            "reference",
            "kind",
            "sha256",
            "byte_count",
            "device",
            "inode",
            "uid",
            "mode",
            "nlink",
            "seals",
        }
        if kind == "regular_file":
            keys.add("path")
        elif kind != "anonymous_descriptor":
            raise ContractError("pre-authorization Python component kind differs")
        _require_exact_keys(component, keys, "pre-authorization Python component")
        if (
            not isinstance(component["reference"], str)
            or not component["reference"]
            or re.fullmatch(r"[0-9a-f]{64}", component["sha256"]) is None
            or any(
                type(component[name]) is not int or component[name] < 0
                for name in (
                    "byte_count",
                    "device",
                    "inode",
                    "uid",
                    "mode",
                    "nlink",
                )
            )
            or component["mode"] & 0o022
        ):
            raise ContractError("pre-authorization Python component differs")
        if kind == "regular_file":
            if (
                component["path"] != component["reference"]
                or not Path(component["path"]).is_absolute()
                or component["nlink"] != 1
                or component["seals"] is not None
            ):
                raise ContractError("pre-authorization Python regular file differs")
        elif (
            re.fullmatch(r"/proc/self/fd/[1-9][0-9]*", component["reference"]) is None
            or component["nlink"] != 0
            or component["seals"] != _required_memfd_seals()
        ):
            raise ContractError("pre-authorization Python descriptor differs")
        component_references.append(component["reference"])
    if (
        component_references != sorted(set(component_references))
        or set(component_references) != referenced_components
    ):
        raise ContractError("pre-authorization Python component closure differs")

    search_path = value["search_path"]
    if not isinstance(search_path, list) or not search_path:
        raise ContractError("isolated Python search path closure is empty")
    for entry in search_path:
        _require_exact_keys(entry, PYTHON_SEARCH_PATH_KEYS, "Python search path")
        if (
            not isinstance(entry["path"], str)
            or not Path(entry["path"]).is_absolute()
            or "site-packages" in entry["path"]
            or "dist-packages" in entry["path"]
            or entry["kind"]
            not in {
                "absent_under_nonwritable_ancestor",
                "nonwritable_directory",
                "verified_archive",
            }
            or any(
                type(entry[name]) is not int or entry[name] < 0
                for name in ("device", "inode", "uid", "mode")
            )
            or entry["mode"] & 0o022
        ):
            raise ContractError("isolated Python search path differs")
        if entry["kind"] == "verified_archive":
            if (
                re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                or type(entry["byte_count"]) is not int
                or entry["byte_count"] <= 0
                or entry["ancestor"] is not None
            ):
                raise ContractError("verified Python archive path differs")
        elif entry["sha256"] is not None or entry["byte_count"] is not None:
            raise ContractError("Python directory search path hash differs")
    closure = {key: item for key, item in value.items() if key != "closure_sha256"}
    if value["closure_sha256"] != sha256_bytes(stable_json_bytes(closure)):
        raise ContractError("pre-authorization Python closure digest differs")


def _validate_runtime_custody(value: dict[str, Any]) -> None:
    _require_exact_keys(
        value, RUNTIME_CUSTODY_KEYS, "execution.runtime_source_manifest"
    )
    if value["schema"] != RUNTIME_SOURCE_MANIFEST_SCHEMA:
        raise ContractError("execution runtime-source manifest schema differs")
    if (
        not isinstance(value["source_root"], str)
        or not Path(value["source_root"]).is_absolute()
    ):
        raise ContractError("execution SOURCE_ROOT is not absolute")
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise ContractError("execution manifest path is not absolute")
    if (
        not isinstance(value["source_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]) is None
    ):
        raise ContractError("execution SOURCE_COMMIT is invalid")
    if (
        not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ContractError("execution manifest hash is invalid")
    manifest_file = value["manifest_file"]
    _require_exact_keys(
        manifest_file, MANIFEST_FILE_KEYS, "execution runtime manifest file"
    )
    if manifest_file["mode"] & 0o222 or any(
        not isinstance(manifest_file[key], int)
        or isinstance(manifest_file[key], bool)
        or manifest_file[key] < 0
        for key in ("device", "inode", "uid", "size")
    ):
        raise ContractError("execution runtime manifest inode identity differs")
    files = value["files"]
    if not isinstance(files, dict) or tuple(files) != RUNTIME_SOURCE_PATHS:
        raise ContractError("execution runtime-source closure differs")
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in files.values()
    ):
        raise ContractError("execution runtime-source file hash is invalid")
    runtime = value["runtime"]
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema") != RUNTIME_IDENTITY_SCHEMA
        or set(runtime) != RUNTIME_IDENTITY_KEYS
    ):
        raise ContractError("execution runtime identity schema differs")
    for name in ("python", "git", "scontrol", "sacct", "nvidia_smi"):
        executable = runtime[name]
        _require_exact_keys(
            executable, EXECUTABLE_IDENTITY_KEYS, f"runtime executable {name}"
        )
        if (
            not isinstance(executable["path"], str)
            or not Path(executable["path"]).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", executable["sha256"]) is None
            or not isinstance(executable["version"], str)
            or not executable["version"]
        ):
            raise ContractError(f"runtime executable identity differs at {name}")
    _validate_python_startup_identity(runtime["python_startup"])
    packages = runtime["packages"]
    if not isinstance(packages, dict) or set(packages) != {"torch", "tokenizers"}:
        raise ContractError("runtime package closure differs")
    for name, package in packages.items():
        _require_exact_keys(
            package, DISTRIBUTION_IDENTITY_KEYS, f"runtime package {name}"
        )
        if (
            package["distribution_name"] != name
            or not isinstance(package["version"], str)
            or not package["version"]
            or any(
                not isinstance(package[key], str)
                or not Path(package[key]).is_absolute()
                for key in (
                    "distribution_root",
                    "installation_root",
                    "module_path",
                    "record_path",
                )
            )
            or type(package["file_count"]) is not int
            or package["file_count"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", package["closure_sha256"]) is None
            or not isinstance(package["files"], list)
            or len(package["files"]) != package["file_count"]
        ):
            raise ContractError(f"runtime package identity differs at {name}")
        previous_path = None
        installation_root = Path(package["installation_root"])
        distribution_root = Path(package["distribution_root"])
        try:
            distribution_root.relative_to(installation_root)
        except ValueError as error:
            raise ContractError(
                f"runtime package root escapes installation at {name}"
            ) from error
        for entry in package["files"]:
            _require_exact_keys(
                entry, DISTRIBUTION_FILE_KEYS, f"runtime package file {name}"
            )
            relative_path = _strict_distribution_relative_path(
                entry["relative_path"], f"runtime package file {name}"
            )
            if previous_path is not None and relative_path <= previous_path:
                raise ContractError(f"runtime package file order differs at {name}")
            previous_path = relative_path
            expected_path = Path(
                os.path.normpath(
                    str(distribution_root.joinpath(*PurePosixPath(relative_path).parts))
                )
            )
            if not isinstance(entry["path"], str):
                raise ContractError(f"runtime package file identity differs at {name}")
            entry_path = Path(entry["path"])
            try:
                entry_path.relative_to(installation_root)
            except ValueError as error:
                raise ContractError(
                    f"runtime package file escapes installation at {name}"
                ) from error
            if (
                not entry_path.is_absolute()
                or entry_path != expected_path
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                or any(
                    type(entry[key]) is not int or entry[key] < 0
                    for key in ("byte_count", "device", "inode", "uid", "mode", "nlink")
                )
                or entry["nlink"] != 1
                or entry["mode"] & 0o022
            ):
                raise ContractError(f"runtime package file identity differs at {name}")
        file_paths = {
            entry["relative_path"]: entry["path"] for entry in package["files"]
        }
        record_paths = [
            path
            for relative_path, path in file_paths.items()
            if relative_path.endswith(".dist-info/RECORD")
        ]
        if file_paths.get(f"{name}/__init__.py") != package[
            "module_path"
        ] or record_paths != [package["record_path"]]:
            raise ContractError(f"runtime package entry points differ at {name}")
        closure_payload = stable_json_bytes(
            {
                "distribution_name": package["distribution_name"],
                "distribution_root": package["distribution_root"],
                "installation_root": package["installation_root"],
                "files": package["files"],
                "version": package["version"],
            }
        )
        if sha256_bytes(closure_payload) != package["closure_sha256"]:
            raise ContractError(f"runtime package closure digest differs at {name}")
    backend = runtime["backend"]
    _require_exact_keys(backend, RUNTIME_BACKEND_KEYS, "runtime backend")
    expected_backend = {
        "device": DEVICE,
        "precision": PRECISION,
        "sdpa_backend": SDPA_BACKEND,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "ld_preload": None,
        "dyld_insert_libraries": None,
        "coverage_claim": (
            "sealed_python_startup_full_record_distributions_and_"
            "mapped_device_inode_bound_native_maps"
        ),
    }
    for key, expected in expected_backend.items():
        if backend[key] != expected:
            raise ContractError(f"runtime backend identity differs at {key}")
    if backend["ld_library_path"] is not None and not isinstance(
        backend["ld_library_path"], str
    ):
        raise ContractError("runtime backend LD_LIBRARY_PATH differs")
    native = backend["preauthorization_native_libraries"]
    _require_exact_keys(
        native, NATIVE_LIBRARY_CLOSURE_KEYS, "pre-authorization native closure"
    )
    if (
        native["platform"] != sys.platform
        or not isinstance(native["source"], str)
        or not isinstance(native["files"], list)
        or re.fullmatch(r"[0-9a-f]{64}", native["closure_sha256"]) is None
        or sha256_bytes(stable_json_bytes(native["files"])) != native["closure_sha256"]
        or (
            sys.platform.startswith("linux")
            and (
                native["source"] != "proc_self_maps_bound_device_inode_and_path"
                or not native["files"]
            )
        )
        or (
            not sys.platform.startswith("linux")
            and native["source"] != "non_linux_development_only_no_proc_self_maps"
        )
    ):
        raise ContractError("pre-authorization native closure differs")
    previous_native_path = None
    for entry in native["files"]:
        _require_exact_keys(
            entry, NATIVE_LIBRARY_FILE_KEYS, "pre-authorization native file"
        )
        if (
            not isinstance(entry["path"], str)
            or not Path(entry["path"]).is_absolute()
            or (
                previous_native_path is not None
                and entry["path"] <= previous_native_path
            )
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or any(
                type(entry[key]) is not int or entry[key] < 0
                for key in (
                    "byte_count",
                    "device",
                    "inode",
                    "uid",
                    "mode",
                    "nlink",
                    "mapped_device_major",
                    "mapped_device_minor",
                    "mapped_inode",
                )
            )
            or entry["nlink"] != 1
            or entry["mode"] & 0o022
            or (
                os.major(entry["device"]),
                os.minor(entry["device"]),
                entry["inode"],
            )
            != (
                entry["mapped_device_major"],
                entry["mapped_device_minor"],
                entry["mapped_inode"],
            )
        ):
            raise ContractError("pre-authorization native file identity differs")
        previous_native_path = entry["path"]
    if value["git_status"] != "clean" or value["git_show_byte_equality"] is not True:
        raise ContractError(
            "execution source custody was not clean and commit-identical"
        )


def _validate_runtime_observation(
    value: dict[str, Any], expected_phase: str, python_path: str
) -> None:
    _require_exact_keys(value, RUNTIME_OBSERVATION_KEYS, "runtime observation")
    if (
        value["schema"] != RUNTIME_OBSERVATION_SCHEMA
        or value["phase"] != expected_phase
        or value["coverage"]
        != "loaded_file_snapshot_not_complete_immutable_executed_runtime_seal"
        or value["python_executable"] != python_path
        or value["ld_preload"] is not None
        or value["dyld_insert_libraries"] is not None
    ):
        raise ContractError("runtime observation identity differs")
    if set(value["platform"]) != {"system", "release", "machine"} or any(
        not isinstance(item, str) or not item for item in value["platform"].values()
    ):
        raise ContractError("runtime platform observation differs")
    if (
        not isinstance(value["libc_version"], list)
        or len(value["libc_version"]) != 2
        or any(not isinstance(item, str) for item in value["libc_version"])
    ):
        raise ContractError("runtime libc observation differs")
    if value["ld_library_path"] is not None and not isinstance(
        value["ld_library_path"], str
    ):
        raise ContractError("runtime LD_LIBRARY_PATH observation differs")
    if (
        not isinstance(value["mapping_source_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["mapping_source_sha256"]) is None
    ):
        raise ContractError("runtime mapping-source hash differs")
    for list_name in (
        "loaded_files",
        "shared_objects",
        "libc_objects",
        "loader_objects",
        "cuda_objects",
    ):
        records = value[list_name]
        if not isinstance(records, list):
            raise ContractError(f"runtime object list differs at {list_name}")
        observed_paths = []
        for index, record in enumerate(records):
            _require_exact_keys(
                record, RUNTIME_FILE_KEYS, f"runtime.{list_name}[{index}]"
            )
            if (
                not isinstance(record["path"], str)
                or not Path(record["path"]).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                or any(
                    not isinstance(record[key], int)
                    or isinstance(record[key], bool)
                    or record[key] < 0
                    for key in ("device", "inode", "size")
                )
            ):
                raise ContractError(f"runtime object record differs at {list_name}")
            mapping_identity = record["mapping_identity"]
            if mapping_identity is not None:
                _require_exact_keys(
                    mapping_identity,
                    {"device_major", "device_minor", "inode"},
                    f"runtime.{list_name}[{index}].mapping_identity",
                )
                if any(
                    type(mapping_identity[key]) is not int or mapping_identity[key] < 0
                    for key in ("device_major", "device_minor", "inode")
                ) or (
                    os.major(record["device"]),
                    os.minor(record["device"]),
                    record["inode"],
                ) != (
                    mapping_identity["device_major"],
                    mapping_identity["device_minor"],
                    mapping_identity["inode"],
                ):
                    raise ContractError(
                        f"runtime mapped object identity differs at {list_name}"
                    )
            observed_paths.append(record["path"])
        if observed_paths != sorted(set(observed_paths)):
            raise ContractError(f"runtime object order differs at {list_name}")
    if value["loaded_files_sha256"] != sha256_bytes(
        stable_json_bytes(value["loaded_files"])
    ) or value["shared_objects_sha256"] != sha256_bytes(
        stable_json_bytes(value["shared_objects"])
    ):
        raise ContractError("runtime object-closure hash differs")
    if any(
        not value[name]
        for name in ("shared_objects", "libc_objects", "loader_objects", "cuda_objects")
    ):
        raise ContractError("runtime shared-object evidence is incomplete")
    expected_shared = [
        record
        for record in value["loaded_files"]
        if ".so" in Path(record["path"]).name or Path(record["path"]).suffix == ".dylib"
    ]
    expected_libc = [
        record
        for record in expected_shared
        if re.search(r"(^|/)libc(?:[-.]|\.so)", record["path"])
    ]
    expected_loader = [
        record
        for record in expected_shared
        if re.search(r"ld-linux|ld-musl|/dyld$", record["path"])
    ]
    expected_cuda = [
        record
        for record in expected_shared
        if re.search(
            r"cuda|cudnn|cublas|cufft|curand|cusolver|cusparse|nccl|nvidia",
            record["path"],
            re.IGNORECASE,
        )
    ]
    if (
        value["shared_objects"] != expected_shared
        or value["libc_objects"] != expected_libc
        or value["loader_objects"] != expected_loader
        or value["cuda_objects"] != expected_cuda
    ):
        raise ContractError("runtime shared-object classification differs")
    if value["platform"]["system"] == "Linux" and any(
        record["mapping_identity"] is None for record in expected_shared
    ):
        raise ContractError("Linux shared object lacks mapped device/inode binding")
    if (
        not isinstance(value["cuda_driver_version"], int)
        or isinstance(value["cuda_driver_version"], bool)
        or value["cuda_driver_version"] <= 0
        or not isinstance(value["cuda_runtime_version"], str)
        or not value["cuda_runtime_version"]
        or not isinstance(value["cuda_device_uuid"], str)
        or not value["cuda_device_uuid"]
        or not isinstance(value["cuda_visible_devices"], str)
        or not value["cuda_visible_devices"]
        or value["cuda_device_name"] != REQUIRED_CUDA_DEVICE_NAME
        or value["cuda_device_capability"] != list(REQUIRED_CUDA_DEVICE_CAPABILITY)
        or not isinstance(value["cuda_device_total_memory_bytes"], int)
        or isinstance(value["cuda_device_total_memory_bytes"], bool)
        or not (
            REQUIRED_CUDA_MEMORY_MIN_BYTES
            <= value["cuda_device_total_memory_bytes"]
            <= REQUIRED_CUDA_MEMORY_MAX_BYTES
        )
    ):
        raise ContractError("runtime CUDA observation differs")
    driver_file = value["nvidia_driver_file"]
    if (
        not isinstance(driver_file, dict)
        or set(driver_file) != {"path", "sha256", "text_sha256"}
        or driver_file["path"] != "/proc/driver/nvidia/version"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", driver_file[key]) is None
            for key in ("sha256", "text_sha256")
        )
    ):
        raise ContractError("runtime NVIDIA driver-file observation differs")


def frozen_input_paths() -> dict[str, str]:
    return {
        "checkpoint": str(CHECKPOINT_PATH),
        "tokenizer": str(TOKENIZER_PATH),
        "heldout": str(HELDOUT_PATH),
    }


def observe_frozen_inputs() -> dict[str, Any]:
    observed = {
        "checkpoint": sha256_regular_file(CHECKPOINT_PATH, require_single_link=True),
        "tokenizer": sha256_regular_file(TOKENIZER_PATH, require_single_link=True),
        "heldout": sha256_regular_file(HELDOUT_PATH, require_single_link=True),
    }
    if observed != EXPECTED_SHA256:
        raise ContractError("validation-time frozen input hashes differ")
    return {"paths": frozen_input_paths(), "sha256": observed}


def _validate_frozen_input_observation(value: dict[str, Any]) -> None:
    _require_exact_keys(
        value, FROZEN_INPUT_OBSERVATION_KEYS, "frozen input observation"
    )
    if not _strict_equal(
        value, {"paths": frozen_input_paths(), "sha256": dict(EXPECTED_SHA256)}
    ):
        raise ContractError("frozen input observation differs")


def _validate_source_execution(
    value: dict[str, Any], custody: dict[str, Any], *, live_custody: bool
) -> None:
    _require_exact_keys(value, SOURCE_EXECUTION_KEYS, "execution.source_execution")
    del live_custody
    if (
        value["mode"] != SOURCE_EXECUTION_MODE
        or value["python_startup_mode"] != PYTHON_STARTUP_MODE
    ):
        raise ContractError("source execution mode differs")
    expected_hashes = {
        "evaluator": custody["files"]["train/eval_dws_eos_suppressed_trace.py"],
        "model": custody["files"]["train/model.py"],
    }
    for name, expected_sha256 in expected_hashes.items():
        record = value[name]
        _require_exact_keys(
            record, SEALED_SOURCE_DESCRIPTOR_KEYS, f"source execution {name}"
        )
        if (
            record["descriptor_kind"] != "sealed_memfd"
            or record["sha256"] != expected_sha256
            or not isinstance(record["byte_count"], int)
            or isinstance(record["byte_count"], bool)
            or record["byte_count"] <= 0
            or record["seals"] != _required_memfd_seals()
        ):
            raise ContractError(f"sealed source descriptor differs at {name}")


def _validate_execution(value: dict[str, Any], *, live_custody: bool) -> None:
    _require_exact_keys(value, EXECUTION_KEYS, "execution")
    if value["input_paths"] != frozen_input_paths():
        raise ContractError("execution input paths differ from exact frozen paths")
    if value["verified_input_sha256"] != EXPECTED_SHA256:
        raise ContractError("execution input hashes differ")
    if value["checkpoint_step"] != EXPECTED_CHECKPOINT_STEP:
        raise ContractError("execution checkpoint step differs")
    custody = value["runtime_source_manifest"]
    _validate_runtime_custody(custody)
    if live_custody:
        replayed_custody = verify_runtime_source_manifest(
            Path(custody["source_root"]),
            custody["source_commit"],
            Path(custody["path"]),
            custody["sha256"],
            Path(custody["runtime"]["python"]["path"]),
            Path(custody["runtime"]["git"]["path"]),
            Path(custody["runtime"]["scontrol"]["path"]),
            Path(custody["runtime"]["sacct"]["path"]),
            Path(custody["runtime"]["nvidia_smi"]["path"]),
        )
        if replayed_custody != custody:
            raise ContractError("execution runtime-source custody does not replay")
    _validate_source_execution(
        value["source_execution"], custody, live_custody=live_custody
    )
    runtime_observation = value["runtime_observation"]
    _validate_runtime_observation(
        runtime_observation,
        "generator_post_decode",
        custody["runtime"]["python"]["path"],
    )
    if runtime_observation["ld_library_path"] != custody["runtime"]["backend"].get(
        "ld_library_path"
    ):
        raise ContractError("generator LD_LIBRARY_PATH differs from runtime seal")
    if runtime_observation["cuda_runtime_version"] != value["cuda_runtime"]:
        raise ContractError("generator CUDA runtime observation differs")
    for execution_key, observation_key in (
        ("cuda_visible_devices", "cuda_visible_devices"),
        ("device_name", "cuda_device_name"),
        ("device_capability", "cuda_device_capability"),
        ("device_total_memory_bytes", "cuda_device_total_memory_bytes"),
        ("device_uuid", "cuda_device_uuid"),
    ):
        if value[execution_key] != runtime_observation[observation_key]:
            raise ContractError(
                f"generator device/runtime observation differs at {execution_key}"
            )
    exact = {
        "device": DEVICE,
        "precision": PRECISION,
        "visible_cuda_device_count": 1,
        "device_name": REQUIRED_CUDA_DEVICE_NAME,
        "device_capability": list(REQUIRED_CUDA_DEVICE_CAPABILITY),
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cuda_matmul_tf32_allowed": False,
        "cudnn_tf32_allowed": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
        "sdpa_backend": SDPA_BACKEND,
        "sdpa_math_enabled": True,
        "sdpa_flash_enabled": False,
        "sdpa_mem_efficient_enabled": False,
        "sdpa_cudnn_enabled": False,
        "sdpa_bf16_probe_bitwise_equal": True,
        "seed": 0,
    }
    for key, expected in exact.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ContractError(f"execution determinism contract differs at {key}")
    started_at = _require_utc_timestamp(value["started_at_utc"], "execution start")
    finished_at = _require_utc_timestamp(value["finished_at_utc"], "execution finish")
    if finished_at < started_at:
        raise ContractError("execution finish precedes execution start")
    for key in ("python", "torch"):
        if not isinstance(value[key], str) or not value[key]:
            raise ContractError(f"execution metadata is invalid at {key}")
    if not isinstance(value["cuda_runtime"], str) or not value["cuda_runtime"]:
        raise ContractError("execution CUDA runtime metadata is invalid")
    if (
        not isinstance(value["cuda_visible_devices"], str)
        or not value["cuda_visible_devices"]
        or not isinstance(value["device_uuid"], str)
        or not value["device_uuid"]
        or not isinstance(value["device_total_memory_bytes"], int)
        or isinstance(value["device_total_memory_bytes"], bool)
        or not (
            REQUIRED_CUDA_MEMORY_MIN_BYTES
            <= value["device_total_memory_bytes"]
            <= REQUIRED_CUDA_MEMORY_MAX_BYTES
        )
    ):
        raise ContractError("execution H100 PCIe device metadata is invalid")
    gpu_binding = value.get("_validated_slurm_gpu_binding")
    if gpu_binding is not None:
        raise ContractError("execution contains an internal-only GPU binding field")


def _validate_wrapper_acceptance(
    value: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    _require_plain_json_tree(value, "wrapper_acceptance")
    _require_exact_keys(value, WRAPPER_ACCEPTANCE_KEYS, "wrapper_acceptance")
    if expected is None or not _strict_equal(value, expected):
        raise ContractError("live wrapper acceptance context is absent or differs")
    if (
        value["schema"] != WRAPPER_ACCEPTANCE_SCHEMA
        or value["publication_state"] != PRIVATE_CANDIDATE_STATE
    ):
        raise ContractError("wrapper acceptance identity differs")
    slurm = value["slurm_identity"]
    _validate_slurm_identity(slurm)
    for key in (
        "wrapper_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
        "nonce",
    ):
        if (
            not isinstance(value[key], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
        ):
            raise ContractError(f"wrapper acceptance hash differs at {key}")
    output_directory = value["output_directory"]
    _require_exact_keys(
        output_directory, OUTPUT_DIRECTORY_KEYS, "wrapper_acceptance.output_directory"
    )
    if (
        not isinstance(output_directory["path"], str)
        or not Path(output_directory["path"]).is_absolute()
        or output_directory["uid"] != os.getuid()
        or output_directory["mode"] != 0o700
        or any(
            not isinstance(output_directory[key], int)
            or isinstance(output_directory[key], bool)
            or output_directory[key] < 0
            for key in ("device", "inode")
        )
    ):
        raise ContractError("wrapper output-directory identity differs")
    candidate_name = _validate_directory_entry_name(
        value["candidate_name"], "candidate name"
    )
    accepted_name = _validate_directory_entry_name(
        value["accepted_name"], "accepted name"
    )
    if candidate_name != (
        f".{accepted_name}.r12-candidate-{slurm['job_id']}-{value['nonce']}"
    ):
        raise ContractError("wrapper candidate/final name identity differs")
    for key in (
        "generator_signing_key",
        "verifier_signing_key",
        "delegated_marker_signing_key",
    ):
        signing_key = value[key]
        _require_exact_keys(
            signing_key,
            SIGNING_KEY_RECORD_KEYS,
            f"wrapper_acceptance.{key}",
        )
        if (
            signing_key["descriptor_kind"] != "sealed_memfd"
            or signing_key["byte_count"] != 32
            or re.fullmatch(r"[0-9a-f]{64}", signing_key["private_key_sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", signing_key["public_key_hex"]) is None
            or signing_key["seals"] != _required_memfd_seals()
        ):
            raise ContractError(f"{key} identity differs")
    public_keys = {
        value[key]["public_key_hex"]
        for key in (
            "generator_signing_key",
            "verifier_signing_key",
            "delegated_marker_signing_key",
        )
    }
    if len(public_keys) != 3:
        raise ContractError(
            "generator, verifier, and delegated marker keys must be distinct"
        )
    for key in ("production_authority_key_file", "run_authorization_file"):
        record = value[key]
        _require_exact_keys(
            record, EXTERNAL_FILE_RECORD_KEYS, f"wrapper_acceptance.{key}"
        )
        if (
            type(record["path"]) is not str
            or not Path(record["path"]).is_absolute()
            or type(record["device"]) is not int
            or type(record["inode"]) is not int
            or type(record["uid"]) is not int
            or type(record["mode"]) is not int
            or type(record["nlink"]) is not int
            or type(record["size"]) is not int
            or record["nlink"] != 1
            or record["mode"] & 0o222
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ContractError(
                f"external authorization file identity differs at {key}"
            )
    authorization = value["run_authorization"]
    if (
        value["run_authorization_sha256"]
        != sha256_bytes(stable_json_bytes(authorization))
        or value["run_authorization_file"]["sha256"]
        != value["run_authorization_sha256"]
    ):
        raise ContractError("run authorization file/hash binding differs")
    if authorization["authority_scope"] == PRODUCTION_AUTHORITY_SCOPE:
        if value["production_authority_key_file"]["sha256"] != (
            PRODUCTION_AUTHORITY_FILE_SHA256
        ):
            raise ContractError("production authority key file differs from source pin")
    validate_run_authorization(
        authorization,
        expected_slurm=slurm,
        expected_output_directory=output_directory,
        expected_accepted_name=accepted_name,
        expected_source_manifest_sha256=value["source_manifest_sha256"],
        expected_delegated_marker_key=value["delegated_marker_signing_key"],
        require_current=False,
    )
    _require_exact_keys(
        value["sealed_generator"],
        SEALED_GENERATOR_KEYS,
        "wrapper_acceptance.sealed_generator",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value["sealed_generator"][key]) is None
        for key in SEALED_GENERATOR_KEYS
    ):
        raise ContractError("sealed generator source identity differs")
    qualification_evaluator_sha256 = authorization["linux_qualification_receipt"][
        "qualification_result"
    ]["evaluator_sha256"]
    if qualification_evaluator_sha256 != value["sealed_generator"]["evaluator_sha256"]:
        raise ContractError(
            "Linux qualification receipt evaluator/source binding differs"
        )


def load_wrapper_acceptance_context_descriptor(
    descriptor: int, expected_sha256: str
) -> dict[str, Any]:
    info = os.fstat(descriptor)
    context_bytes = read_sealed_memfd_bytes(
        descriptor, info.st_size, "wrapper acceptance context"
    )
    if sha256_bytes(context_bytes) != expected_sha256:
        raise ContractError("wrapper acceptance context descriptor hash differs")
    context = _parse_json_object_bytes(
        context_bytes, "wrapper acceptance context descriptor"
    )
    if stable_json_bytes(context) != context_bytes:
        raise ContractError("wrapper acceptance context is not canonical JSON")
    _validate_wrapper_acceptance(context, context)
    return context


def write_anonymous_report_descriptor(descriptor: int, payload: bytes) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 0
        or info.st_uid != os.getuid()
        or info.st_size != 0
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise ContractError("anonymous report output descriptor identity differs")
    _write_all(descriptor, payload)
    os.fsync(descriptor)
    after = os.fstat(descriptor)
    if _inode_policy_identity(after) != _inode_policy_identity(
        info
    ) or after.st_size != len(payload):
        raise ContractError("anonymous report output descriptor readback differs")
    _read_exact_descriptor_bytes(
        descriptor,
        "anonymous report output",
        expected_info=after,
        expected_payload=payload,
    )


def verify_live_slurm_context(
    context: dict[str, Any],
    runtime_custody: dict[str, Any],
    *,
    scontrol_descriptor: int | None = None,
    sacct_descriptor: int | None = None,
    nvidia_smi_descriptor: int | None = None,
) -> None:
    scontrol = runtime_custody["runtime"]["scontrol"]["path"]
    sacct = runtime_custody["runtime"]["sacct"]["path"]
    nvidia_smi = runtime_custody["runtime"]["nvidia_smi"]["path"]
    observed = observe_slurm_identity(
        scontrol,
        sacct,
        nvidia_smi,
        context["slurm_identity"]["job_id"],
        scontrol_descriptor=scontrol_descriptor,
        sacct_descriptor=sacct_descriptor,
        nvidia_smi_descriptor=nvidia_smi_descriptor,
    )
    if observed != context["slurm_identity"]:
        raise ContractError(
            "parsed live Slurm identity differs from acceptance context"
        )


def validate_report_schema(
    report: dict[str, Any],
    expected_acceptance_context: dict[str, Any] | None = None,
    *,
    live_custody: bool = True,
) -> None:
    _require_plain_json_tree(report, "report")
    _require_exact_keys(report, TOP_LEVEL_KEYS, "report")
    if report["schema"] != OUTPUT_SCHEMA or report["protocol"] != PROTOCOL_ID:
        raise ContractError("report identity differs from frozen schema")
    if (
        report["development_only"] is not True
        or report["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise ContractError("report escaped the development claim boundary")
    if not _strict_equal(report["frozen_contract"], frozen_contract()):
        raise ContractError("report frozen contract differs")
    _validate_wrapper_acceptance(
        report["wrapper_acceptance"], expected_acceptance_context
    )
    validate_generator_attestation(report, report["wrapper_acceptance"])
    _validate_execution(report["execution"], live_custody=live_custody)
    custody = report["execution"]["runtime_source_manifest"]
    acceptance = report["wrapper_acceptance"]
    gpu_binding = acceptance["slurm_identity"]["gpu_binding"]
    if (
        report["execution"]["device_uuid"] != gpu_binding["gpu_uuid"]
        or report["execution"]["device_name"] != gpu_binding["gpu_name"]
        or report["execution"]["cuda_visible_devices"]
        != gpu_binding["cuda_visible_devices"]
        or report["execution"]["runtime_observation"]["cuda_device_uuid"]
        != gpu_binding["gpu_uuid"]
    ):
        raise ContractError("CUDA runtime device is not the scheduler/cgroup-bound GPU")
    if (
        acceptance["wrapper_sha256"]
        != custody["files"]["train/jobs/eval_dws_eos_suppressed_trace.sbatch"]
        or acceptance["slurm_identity"]["command_sha256"]
        != acceptance["wrapper_sha256"]
        or acceptance["source_manifest_sha256"] != custody["sha256"]
        or acceptance["runtime_identity_sha256"]
        != sha256_bytes(stable_json_bytes(custody["runtime"]))
        or acceptance["sealed_generator"]["evaluator_sha256"]
        != custody["files"]["train/eval_dws_eos_suppressed_trace.py"]
        or acceptance["sealed_generator"]["model_sha256"]
        != custody["files"]["train/model.py"]
    ):
        raise ContractError("wrapper acceptance is not bound to sealed source custody")
    authorization = acceptance["run_authorization"]
    authorized_inputs = authorization["frozen_inputs"]
    if (
        authorization["source_commit"] != custody["source_commit"]
        or authorization["source_manifest_path"] != custody["path"]
        or authorized_inputs["prereg_sha256"]
        != custody["files"]["R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md"]
        or authorization["output_directory"] != acceptance["output_directory"]
        or authorization["output_path"]
        != str(
            Path(acceptance["output_directory"]["path"]) / acceptance["accepted_name"]
        )
        or authorized_inputs["checkpoint_path"]
        != report["execution"]["input_paths"]["checkpoint"]
        or authorized_inputs["tokenizer_path"]
        != report["execution"]["input_paths"]["tokenizer"]
        or authorized_inputs["heldout_path"]
        != report["execution"]["input_paths"]["heldout"]
    ):
        raise ContractError("run authorization is not cross-bound to report custody")
    if live_custody:
        verify_live_slurm_context(acceptance, custody)
    try:
        Path(acceptance["output_directory"]["path"]).relative_to(
            Path(custody["source_root"])
        )
    except ValueError:
        pass
    else:
        raise ContractError("accepted output directory must be outside SOURCE_ROOT")
    tokenizer_bytes = read_verified_bytes(TOKENIZER_PATH, EXPECTED_SHA256["tokenizer"])
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    validate_tokenizer_contract(tokenizer)
    heldout_bytes = read_verified_bytes(HELDOUT_PATH, EXPECTED_SHA256["heldout"])
    selected_heldout_rows = select_cases(parse_heldout_bytes(heldout_bytes))

    cases = report["cases"]
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ContractError("report must contain exactly 100 cases")
    ordered_ids = []
    for index, case in enumerate(cases):
        case_path = f"cases[{index}]"
        _require_exact_keys(case, CASE_KEYS, case_path)
        heldout_row = selected_heldout_rows[index]
        ordered_ids.append(case["case_id"])
        initial_state = validate_report_case_heldout_identity(
            case, heldout_row, case_path
        )

        prompt = case["prompt"]
        _require_exact_keys(prompt, PROMPT_RECORD_KEYS, f"{case_path}.prompt")
        expected_prompt_bytes = render_initial_prompt_bytes(case["initial_state"])
        expected_prompt_ids = _tokenize_prompt(tokenizer, expected_prompt_bytes)
        if (
            prompt["utf8"] != expected_prompt_bytes.decode(PROMPT_ENCODING)
            or prompt["byte_count"] != len(expected_prompt_bytes)
            or prompt["sha256"] != sha256_bytes(expected_prompt_bytes)
            or prompt["token_ids"] != list(expected_prompt_ids)
            or prompt["token_count"] != len(expected_prompt_ids)
            or prompt["token_ids_sha256"] != token_ids_sha256(expected_prompt_ids)
        ):
            raise ContractError(f"case prompt contract differs at {case_path}")

        oracle = case["oracle"]
        _require_exact_keys(oracle, ORACLE_RECORD_KEYS, f"{case_path}.oracle")
        oracle_states = reconstruct_oracle_posthoc(initial_state)
        expected_heldout_binding = validate_heldout_oracle_posthoc(
            heldout_row, oracle_states
        )
        _require_exact_keys(
            oracle["heldout_binding"],
            HELDOUT_BINDING_KEYS,
            f"{case_path}.oracle.heldout_binding",
        )
        expected_oracle = {
            "states": [canonical_dws_state(state) for state in oracle_states],
            "trace_length": len(oracle_states),
            "first_state_carry": oracle_states[0]["c"],
            "final_tape": {
                "r": oracle_states[-1]["r"],
                "c": oracle_states[-1]["c"],
            },
            "answer": state_answer_posthoc(oracle_states[-1]),
            "heldout_binding": expected_heldout_binding,
        }
        if not _strict_equal(oracle, expected_oracle):
            raise ContractError(f"case oracle replay differs at {case_path}")

        primary = case["primary_arms"]
        if not isinstance(primary, dict) or set(primary) != set(PRIMARY_ARM_ORDER):
            raise ContractError(f"primary arm identity differs at {case_path}")
        for arm, record in primary.items():
            record_path = f"{case_path}.primary_arms.{arm}"
            _require_exact_keys(record, PRIMARY_RECORD_KEYS, record_path)
            _validate_decode_common(
                record,
                record_path,
                expected_mode=arm,
                expected_prompt_token_count=prompt["token_count"],
                expected_prompt_token_ids=expected_prompt_ids,
                tokenizer=tokenizer,
            )
            trace_path = f"{record_path}.trace_score"
            _require_exact_keys(record["trace_score"], TRACE_SCORE_KEYS, trace_path)
            for line_index, line in enumerate(record["trace_score"]["dws_lines"]):
                _require_exact_keys(
                    line,
                    DWS_LINE_RECORD_KEYS,
                    f"{trace_path}.dws_lines[{line_index}]",
                )
            for answer_index, answer in enumerate(
                record["trace_score"]["answer_lines"]
            ):
                _require_exact_keys(
                    answer,
                    ANSWER_LINE_RECORD_KEYS,
                    f"{trace_path}.answer_lines[{answer_index}]",
                )
            expected_trace = score_trace_posthoc(
                record["response_text"], initial_state, oracle_states
            )
            if not _strict_equal(record["trace_score"], expected_trace):
                raise ContractError(f"trace-score replay differs at {record_path}")

        field_screen = case["field_screen"]
        field_path = f"{case_path}.field_screen"
        _require_exact_keys(field_screen, FIELD_SCREEN_KEYS, field_path)
        if not isinstance(field_screen["by_clock"], dict) or set(
            field_screen["by_clock"]
        ) != set(FIELD_CLOCK_ARMS):
            raise ContractError(f"field clock identity differs at {field_path}")
        ordinary = primary[DecodeMode.ORDINARY_EOS_STOP.value]
        derived_state, derived_ids, derived_failure = (
            extract_first_emitted_state_posthoc(ordinary, tokenizer)
        )
        if derived_failure is not None:
            if field_screen != unavailable_field_screen(derived_failure):
                raise ContractError(f"unavailable field screen differs at {field_path}")
            continue
        if field_screen["available"] is not True:
            raise ContractError(
                f"field work is missing despite an available ordinary boundary at {field_path}"
            )
        if field_screen["failure"] is not None:
            raise ContractError(f"available field screen has a failure at {field_path}")
        emitted_state = parse_dws_line(field_screen["emitted_state"])
        emitted_count = _require_nonnegative_int(
            field_screen["emitted_history_token_count"],
            f"{field_path}.emitted_history_token_count",
        )
        ordinary_trace = ordinary["trace_score"]
        if (
            emitted_state is None
            or emitted_state != derived_state
            or tuple(derived_ids or ())
            != tuple(tokenizer.encode(field_screen["emitted_state"]).ids)
            or emitted_state["p"] != 1
            or emitted_state["z"]
            or emitted_count == 0
            or emitted_count != ordinary["content_token_count"]
            or ordinary["stop_reason"] != "eos"
            or len(ordinary["eos_events"]) != 1
            or ordinary_trace["dws_candidate_count"] != 1
            or ordinary_trace["valid_dws_line_count"] != 1
            or ordinary_trace["nonprotocol_line_indices"]
            or ordinary_trace["dws_lines"][0]["canonical_state"]
            != field_screen["emitted_state"]
            or tuple(ordinary["generated_token_ids"][:-1])
            != tuple(tokenizer.encode(field_screen["emitted_state"]).ids)
        ):
            raise ContractError(f"field-screen first boundary differs at {field_path}")

        replay_branches: dict[str, dict[str, Any]] | None = None
        ordinary_event = ordinary["eos_events"][0]
        for clock_arm, clock in field_screen["by_clock"].items():
            clock_path = f"{field_path}.by_clock.{clock_arm}"
            _require_exact_keys(clock, CLOCK_SCORE_KEYS, clock_path)
            detail = clock["detail"]
            if detail is None:
                raise ContractError(
                    f"available field screen lacks detail at {clock_path}"
                )
            detail_path = f"{clock_path}.detail"
            _require_exact_keys(detail, FIELD_DETAIL_KEYS, detail_path)
            branches = _validate_branch_contracts(
                detail["branch_contracts"],
                emitted_state,
                emitted_count,
                tokenizer,
                f"{detail_path}.branch_contracts",
            )
            if replay_branches is None:
                replay_branches = branches
            elif (
                detail["branch_contracts"]
                != field_screen["by_clock"][FIELD_CLOCK_ARMS[0]]["detail"][
                    "branch_contracts"
                ]
            ):
                raise ContractError(f"branch contracts vary by clock at {clock_path}")
            if clock_arm == DecodeMode.EOS_MASKED_ARGMAX.value:
                expected_boundary = (
                    ordinary_event["non_eos_argmax_token_id"],
                    ordinary_event["non_eos_argmax_token_text"],
                )
            else:
                expected_boundary = REPLACEMENT_TOKENS[clock_arm]
            if (
                detail["clock_arm"] != clock_arm
                or (detail["boundary_token_id"], detail["boundary_token_text"])
                != expected_boundary
            ):
                raise ContractError(f"field clock boundary differs at {clock_path}")
            boundary_id = expected_boundary[0]
            prompt_ids_by_branch = {}
            for name, branch in branches.items():
                history_ids = (
                    (100,) * emitted_count
                    if name == "equal_token_length_destroyed_history"
                    else tuple(tokenizer.encode(branch["state_line"]).ids)
                )
                prompt_ids_by_branch[name] = (
                    *expected_prompt_ids,
                    *history_ids,
                    boundary_id,
                )
            prompt_counts = {
                name: len(ids) for name, ids in prompt_ids_by_branch.items()
            }
            _validate_branch_reports(
                detail["branch_reports"],
                HISTORY_BRANCHES,
                f"{detail_path}.branch_reports",
                expected_mode=clock_arm,
                branches=branches,
                tokenizer=tokenizer,
                expected_prompt_token_counts=prompt_counts,
                expected_prompt_token_ids=prompt_ids_by_branch,
            )
            _validate_interventions(
                detail["interventions"], f"{detail_path}.interventions"
            )
            if set(detail["history_destruction"]) != {
                "intact_adjacent_transition_exact",
                "destroyed_matches_intact_target",
                "whole_output_changed",
                "history_destruction_paired_loss",
            }:
                raise ContractError(
                    f"history-destruction schema differs at {clock_path}"
                )
            replay_detail = score_field_clock_posthoc(
                clock_arm,
                expected_boundary[0],
                expected_boundary[1],
                detail["branch_reports"],
                branches,
                emitted_state,
                oracle_states,
            )
            if not _strict_equal(detail, replay_detail) or not _strict_equal(
                clock, compact_field_clock_score(replay_detail)
            ):
                raise ContractError(f"field semantic replay differs at {clock_path}")

        if replay_branches is None:
            raise ContractError(f"field branch replay is absent at {field_path}")
        source = field_screen["fresh_latest_reencoding"]
        source_path = f"{field_path}.fresh_latest_reencoding"
        _require_exact_keys(source, FRESH_REENCODING_KEYS, source_path)
        if source["available"] is not True or source["failure"] is not None:
            raise ContractError(f"fresh re-encoding is unavailable at {source_path}")
        detail = source["detail"]
        if detail is None:
            raise ContractError(f"fresh re-encoding detail is absent at {source_path}")
        detail_path = f"{source_path}.detail"
        _require_exact_keys(detail, FRESH_REENCODING_DETAIL_KEYS, detail_path)
        expected_source_mode = (
            "compound_fresh_core_prompt_latest_state_reencoding_canonicalization"
        )
        if (
            source["mode"] != expected_source_mode
            or detail["mode"] != expected_source_mode
            or source["external_reencoding"] is not True
            or detail["external_reencoding"] is not True
        ):
            raise ContractError(
                f"fresh re-encoding resource identity differs at {source_path}"
            )
        source_branches = {
            name: replay_branches[name] for name in FRESH_REENCODING_BRANCHES
        }
        source_prompt_ids = {
            name: _tokenize_prompt(
                tokenizer,
                render_core_prompt_bytes(branch["state_line"]),
            )
            for name, branch in source_branches.items()
        }
        source_prompt_counts = {
            name: len(ids) for name, ids in source_prompt_ids.items()
        }
        _validate_branch_reports(
            detail["branch_reports"],
            FRESH_REENCODING_BRANCHES,
            f"{detail_path}.branch_reports",
            expected_mode=DecodeMode.ORDINARY_EOS_STOP.value,
            branches=source_branches,
            tokenizer=tokenizer,
            expected_prompt_token_counts=source_prompt_counts,
            expected_prompt_token_ids=source_prompt_ids,
        )
        _validate_interventions(detail["interventions"], f"{detail_path}.interventions")
        replay_source = score_fresh_reencoding_posthoc(
            detail["branch_reports"],
            source_branches,
            emitted_state,
            oracle_states,
            field_screen["by_clock"][DecodeMode.EOS_TO_LF.value],
        )
        if not _strict_equal(detail, replay_source) or not _strict_equal(
            source, compact_fresh_reencoding_score(replay_source)
        ):
            raise ContractError(
                f"fresh-reencoding semantic replay differs at {source_path}"
            )

    ids_hash = sha256_bytes(("\n".join(ordered_ids) + "\n").encode("ascii"))
    if ids_hash != ORDERED_CASE_IDS_SHA256:
        raise ContractError("report case order differs from frozen selection")
    if report["execution"]["ordered_case_ids"] != ordered_ids:
        raise ContractError("execution case order differs from report cases")
    recomputed = aggregate_report(cases)
    if not _strict_equal(report["aggregate"], recomputed):
        raise ContractError("report aggregate is not supported by case records")
    expected_adjudication = {
        "field_screen_execution": "development_go",
        "full_state_recurrence": "no_go",
        "carry_target_switch_noncompensatory_veto": recomputed[
            "carry_target_switch_global_veto"
        ],
        "compound_fresh_reencoding_screen_pass": recomputed["fresh_latest_reencoding"][
            "compound_fresh_reencoding_screen_pass"
        ],
        "promotion_authorized": False,
    }
    if not _strict_equal(report["adjudication"], expected_adjudication):
        raise ContractError("report adjudication differs from frozen rule")


def _readback_and_validate_report_at(
    directory_fd: int,
    name: str,
    descriptor: int,
    expected_info: os.stat_result,
    expected_payload: bytes,
    expected_acceptance_context: dict[str, Any],
) -> None:
    observed_payload, _ = _read_held_directory_entry_descriptor(
        directory_fd,
        name,
        descriptor,
        "published report",
        expected_info=expected_info,
        expected_payload=expected_payload,
    )
    report = _parse_json_object_bytes(observed_payload, "published report")
    if stable_json_bytes(report) != observed_payload:
        raise ContractError("published report is not canonical stable JSON")
    validate_report_schema(report, expected_acceptance_context, live_custody=False)


def _entry_stat_or_none(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def cleanup_candidate_transaction_entries(
    directory_fd: int,
    candidate_name: str,
    nonce: str,
    *,
    remove_candidate: bool,
) -> None:
    _validate_directory_entry_name(candidate_name, "candidate name")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ContractError("candidate cleanup nonce differs")
    prefix = f".{candidate_name}.tmp-{nonce}-"
    for name in os.listdir(directory_fd):
        if name.startswith(prefix):
            raise ContractError(
                "candidate transaction residue requires external cleanup authorization"
            )
    if remove_candidate:
        info = _entry_stat_or_none(directory_fd, candidate_name)
        if info is not None:
            raise ContractError(
                "candidate residue requires external cleanup authorization"
            )


def _stale_protocol_patterns(
    accepted_name: str, job_id: str, authorization_nonce: str
) -> tuple[re.Pattern[str], ...]:
    _validate_directory_entry_name(accepted_name, "accepted name")
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ContractError("stale cleanup job ID differs")
    if re.fullmatch(r"[0-9a-f]{64}", authorization_nonce) is None:
        raise ContractError("stale cleanup authorization nonce differs")
    escaped = re.escape(accepted_name)
    escaped_commit = re.escape(acceptance_commit_marker_name(accepted_name))
    any_job = r"[1-9][0-9]*"
    any_nonce = r"[0-9a-f]{64}"
    return (
        re.compile(rf"\.{escaped}\.r12-candidate-{any_job}-{any_nonce}"),
        re.compile(
            rf"\.\.{escaped}\.r12-candidate-{any_job}-{any_nonce}"
            rf"\.tmp-{any_nonce}-[0-9a-f]{{32}}"
        ),
        re.compile(rf"\.{escaped}\.r12-report-{any_nonce}-[0-9a-f]{{32}}"),
        re.compile(rf"\.{escaped_commit}\.r12-commit-{any_nonce}-[0-9a-f]{{32}}"),
        re.compile(rf"\.{escaped}\.r12-rollback-quarantine-{any_nonce}-[0-9a-f]{{64}}"),
        re.compile(
            rf"\.{escaped}\.r12-cleanup-quarantine-{any_job}-"
            rf"{any_nonce}-[0-9a-f]{{64}}"
        ),
    )


def _stale_quarantine_name(
    accepted_name: str,
    job_id: str,
    authorization_nonce: str,
    entry: dict[str, Any],
) -> str:
    _validate_directory_entry_name(accepted_name, "accepted name")
    digest = sha256_bytes(stable_json_bytes(entry))
    return (
        f".{accepted_name}.r12-cleanup-quarantine-{job_id}-"
        f"{authorization_nonce}-{digest}"
    )


def _rollback_quarantine_name(
    accepted_name: str,
    nonce: str,
    source_name: str,
    expected_info: os.stat_result,
    expected_payload: bytes,
) -> str:
    _validate_directory_entry_name(accepted_name, "rollback accepted name")
    _validate_directory_entry_name(source_name, "rollback source name")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ContractError("rollback quarantine nonce differs")
    record = {
        "source_name": source_name,
        "device": expected_info.st_dev,
        "inode": expected_info.st_ino,
        "mode": expected_info.st_mode,
        "nlink": expected_info.st_nlink,
        "uid": expected_info.st_uid,
        "size": expected_info.st_size,
        "mtime_ns": expected_info.st_mtime_ns,
        "sha256": sha256_bytes(expected_payload),
    }
    return (
        f".{accepted_name}.r12-rollback-quarantine-{nonce}-"
        f"{sha256_bytes(stable_json_bytes(record))}"
    )


def _is_stale_quarantine_name(
    name: str, accepted_name: str, job_id: str, authorization_nonce: str
) -> bool:
    pattern = _stale_protocol_patterns(accepted_name, job_id, authorization_nonce)[-1]
    return pattern.fullmatch(name) is not None


def _validate_stale_cleanup_entries(
    entries: Any,
    *,
    accepted_name: str,
    job_id: str,
    authorization_nonce: str,
) -> None:
    if not isinstance(entries, list):
        raise ContractError("stale cleanup authorization must be a list")
    patterns = _stale_protocol_patterns(accepted_name, job_id, authorization_nonce)
    previous_name = None
    for entry in entries:
        _require_exact_keys(entry, STALE_CLEANUP_ENTRY_KEYS, "stale cleanup entry")
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not any(pattern.fullmatch(name) for pattern in patterns)
            or (previous_name is not None and name <= previous_name)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or any(
                type(entry[key]) is not int or entry[key] < 0
                for key in ("device", "inode", "uid", "mode", "nlink", "size")
            )
            or entry["uid"] != os.getuid()
            or entry["nlink"] != 1
            or entry["mode"] & 0o022
        ):
            raise ContractError("stale cleanup entry identity differs")
        previous_name = name


def acquire_publisher_lease(
    directory_fd: int,
    output_directory: dict[str, Any],
    accepted_name: str,
    job_id: str,
    authorization_nonce: str,
) -> dict[str, Any]:
    """Hold the output-directory inode exclusively before inspecting residue."""
    validate_output_directory_fd(
        directory_fd, output_directory, require_path_identity=True
    )
    _validate_directory_entry_name(accepted_name, "accepted name")
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ContractError("publisher lease job ID differs")
    if re.fullmatch(r"[0-9a-f]{64}", authorization_nonce) is None:
        raise ContractError("publisher lease authorization nonce differs")
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ContractError(
            "another live publisher owns the output directory"
        ) from error
    info = os.fstat(directory_fd)
    return {
        "accepted_name": accepted_name,
        "authorization_nonce": authorization_nonce,
        "directory_device": info.st_dev,
        "directory_inode": info.st_ino,
        "job_id": job_id,
        "owner_pid": os.getpid(),
    }


def release_publisher_lease(directory_fd: int, lease: dict[str, Any]) -> None:
    _validate_publisher_lease(directory_fd, lease)
    fcntl.flock(directory_fd, fcntl.LOCK_UN)


def _validate_publisher_lease(
    directory_fd: int, lease: dict[str, Any]
) -> os.stat_result:
    _require_exact_keys(
        lease,
        {
            "accepted_name",
            "authorization_nonce",
            "directory_device",
            "directory_inode",
            "job_id",
            "owner_pid",
        },
        "publisher lease",
    )
    info = os.fstat(directory_fd)
    if (
        lease["directory_device"] != info.st_dev
        or lease["directory_inode"] != info.st_ino
        or lease["owner_pid"] != os.getpid()
    ):
        raise ContractError("publisher lease identity differs")
    return info


def _stale_cleanup_record(
    directory_fd: int, name: str, label: str
) -> tuple[dict[str, Any], bytes]:
    payload, info = read_directory_entry_bytes(directory_fd, name, label)
    return {
        "name": name,
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "size": info.st_size,
        "sha256": sha256_bytes(payload),
    }, payload


def cleanup_stale_publication_entries(
    directory_fd: int,
    accepted_name: str,
    authorization: dict[str, Any],
    lease: dict[str, Any],
    *,
    failure_injector: Any = None,
) -> list[str]:
    """Move signed stale inodes to durable quarantine without pathname unlink."""
    _validate_directory_entry_name(accepted_name, "accepted name")
    _validate_publisher_lease(directory_fd, lease)
    if (
        lease["accepted_name"] != accepted_name
        or lease["job_id"] != authorization["slurm_allocation"]["job_id"]
        or lease["authorization_nonce"] != authorization["authorization_nonce"]
    ):
        raise ContractError("stale cleanup publisher lease differs from authorization")
    entries = authorization["stale_cleanup_entries"]
    _validate_stale_cleanup_entries(
        entries,
        accepted_name=accepted_name,
        job_id=lease["job_id"],
        authorization_nonce=lease["authorization_nonce"],
    )
    commit_marker_name = acceptance_commit_marker_name(accepted_name)
    receipt_name = durable_acceptance_receipt_name(accepted_name)
    for name in (accepted_name, commit_marker_name, receipt_name):
        if _entry_stat_or_none(directory_fd, name) is not None:
            raise FileExistsError(f"refusing stale cleanup beside publication {name}")
    patterns = _stale_protocol_patterns(
        accepted_name, lease["job_id"], lease["authorization_nonce"]
    )
    observed_names = sorted(
        name
        for name in os.listdir(directory_fd)
        if any(pattern.fullmatch(name) for pattern in patterns)
    )
    authorized_names = [entry["name"] for entry in entries]
    if observed_names != authorized_names:
        raise ContractError(
            "protocol residue differs from externally authorized stale cleanup set"
        )
    for entry in entries:
        observed, _payload = _stale_cleanup_record(
            directory_fd, entry["name"], "authorized stale cleanup entry"
        )
        if observed != entry:
            raise ContractError("authorized stale cleanup inode or bytes differ")
    quarantined_names = []
    for entry in entries:
        if _is_stale_quarantine_name(
            entry["name"],
            accepted_name,
            lease["job_id"],
            lease["authorization_nonce"],
        ):
            quarantined_names.append(entry["name"])
            continue
        quarantine_name = _stale_quarantine_name(
            accepted_name,
            lease["job_id"],
            lease["authorization_nonce"],
            entry,
        )
        if _entry_stat_or_none(directory_fd, quarantine_name) is not None:
            raise ContractError("stale cleanup quarantine destination already exists")
        rename_noreplace_at(directory_fd, entry["name"], directory_fd, quarantine_name)
        os.fsync(directory_fd)
        if failure_injector is not None:
            failure_injector("after_quarantine_rename", entry["name"], quarantine_name)
        quarantined, _payload = _stale_cleanup_record(
            directory_fd, quarantine_name, "quarantined stale cleanup entry"
        )
        quarantined["name"] = entry["name"]
        if quarantined != entry:
            raise ContractError(
                "stale cleanup substitution was quarantined and preserved"
            )
        quarantined_names.append(quarantine_name)
    os.fsync(directory_fd)
    return quarantined_names


def publish_private_candidate_exclusive(
    directory_fd: int,
    candidate_name: str,
    report: dict[str, Any],
    expected_acceptance_context: dict[str, Any],
) -> tuple[str, int, int, os.stat_result]:
    """Publish a candidate and transfer creation plus bound read-only FDs."""
    validate_output_directory_fd(
        directory_fd,
        expected_acceptance_context["output_directory"],
        require_path_identity=True,
    )
    if candidate_name != expected_acceptance_context["candidate_name"]:
        raise ContractError("candidate name differs from acceptance context")
    _validate_directory_entry_name(candidate_name, "candidate name")
    validate_report_schema(report, expected_acceptance_context, live_custody=False)
    payload = stable_json_bytes(report)
    if _entry_stat_or_none(directory_fd, candidate_name) is not None:
        raise FileExistsError(candidate_name)
    nonce = expected_acceptance_context["nonce"]
    cleanup_candidate_transaction_entries(
        directory_fd, candidate_name, nonce, remove_candidate=False
    )
    temp_name = f".{candidate_name}.tmp-{nonce}-{secrets.token_hex(16)}"
    descriptor = os.open(
        temp_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    published_info: os.stat_result | None = None
    verifier_descriptor: int | None = None
    descriptor_transferred = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("candidate temp write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if os.pread(descriptor, len(payload) + 1, 0) != payload:
            raise ContractError("candidate temp descriptor readback differs")
        temp_report = _parse_json_object_bytes(payload, "candidate temp report")
        validate_report_schema(
            temp_report, expected_acceptance_context, live_custody=False
        )
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        temp_info = os.fstat(descriptor)
        published_info = temp_info
        verifier_descriptor = os.open(
            temp_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        os.set_inheritable(verifier_descriptor, False)
        if (
            fcntl.fcntl(verifier_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
            != os.O_RDONLY
            or os.get_inheritable(verifier_descriptor)
            or _file_identity(os.fstat(verifier_descriptor))
            != _file_identity(temp_info)
        ):
            raise ContractError("private candidate verifier descriptor differs")
        _read_exact_descriptor_bytes(
            verifier_descriptor,
            "private candidate verifier descriptor",
            expected_info=temp_info,
            expected_payload=payload,
        )
        rename_noreplace_at(directory_fd, temp_name, directory_fd, candidate_name)
        os.fsync(directory_fd)
        validate_output_directory_fd(
            directory_fd,
            expected_acceptance_context["output_directory"],
            require_path_identity=True,
        )
        final_info = os.stat(candidate_name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _file_identity(final_info) != _file_identity(temp_info)
            or not stat.S_ISREG(final_info.st_mode)
            or stat.S_IMODE(final_info.st_mode) != 0o400
            or final_info.st_nlink != 1
        ):
            raise ContractError("private candidate final stat contract differs")
        _readback_and_validate_report_at(
            directory_fd,
            candidate_name,
            descriptor,
            temp_info,
            payload,
            expected_acceptance_context,
        )
        held_info = os.fstat(descriptor)
        if _file_identity(held_info) != _file_identity(temp_info):
            raise ContractError("private candidate creation descriptor changed")
        descriptor_transferred = True
        return sha256_bytes(payload), descriptor, verifier_descriptor, held_info
    except BaseException as error:
        rollback_refused = False
        for name, label in (
            (candidate_name, "candidate rollback"),
            (temp_name, "candidate temp rollback"),
        ):
            rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
                directory_fd,
                expected_acceptance_context["accepted_name"],
                nonce,
                name,
                descriptor,
                published_info,
                payload,
                label,
            )
            if not rollback_ok:
                rollback_refused = True
        os.fsync(directory_fd)
        if rollback_refused:
            raise ContractError(
                "candidate rollback quarantine refused an inode-substituted or "
                "byte-mutated entry"
            ) from error
        raise
    finally:
        if not descriptor_transferred:
            os.close(descriptor)
            if verifier_descriptor is not None:
                os.close(verifier_descriptor)


def verify_private_candidate(
    directory_fd: int,
    candidate_name: str,
    expected_acceptance_context: dict[str, Any],
) -> str:
    validate_output_directory_fd(
        directory_fd,
        expected_acceptance_context["output_directory"],
        require_path_identity=True,
    )
    if candidate_name != expected_acceptance_context["candidate_name"]:
        raise ContractError("candidate verifier name differs from context")
    payload, info = read_directory_entry_bytes(
        directory_fd, candidate_name, "private candidate"
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_nlink != 1
    ):
        raise ContractError("private candidate stat contract differs")
    report = _parse_json_object_bytes(payload, "private candidate")
    if stable_json_bytes(report) != payload:
        raise ContractError("private candidate is not canonical stable JSON")
    validate_report_schema(report, expected_acceptance_context, live_custody=False)
    return sha256_bytes(payload)


def read_private_candidate_descriptor(
    descriptor: int,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or before.st_size <= 0
    ):
        raise ContractError("private candidate descriptor stat contract differs")
    payload = os.pread(descriptor, before.st_size + 1, 0)
    after = os.fstat(descriptor)
    if len(payload) != before.st_size or _file_identity(after) != _file_identity(
        before
    ):
        raise ContractError("private candidate descriptor changed while reading")
    return payload, after


def build_independent_verifier_receipt(
    candidate_payload: bytes,
    candidate_info: os.stat_result,
    report: dict[str, Any],
    validation_runtime_observation: dict[str, Any],
    frozen_input_observation: dict[str, Any],
    verifier_private_key_bytes: bytes,
) -> dict[str, Any]:
    _validate_runtime_observation(
        validation_runtime_observation,
        "independent_validator",
        report["execution"]["runtime_source_manifest"]["runtime"]["python"]["path"],
    )
    _validate_frozen_input_observation(frozen_input_observation)
    context = report["wrapper_acceptance"]
    if (
        signing_key_record(verifier_private_key_bytes)
        != context["verifier_signing_key"]
    ):
        raise ContractError("independent verifier signing key differs from context")
    payload = {
        "schema": INDEPENDENT_VERIFIER_SCHEMA,
        "candidate_sha256": sha256_bytes(candidate_payload),
        "candidate_inode": {
            "device": candidate_info.st_dev,
            "inode": candidate_info.st_ino,
            "uid": candidate_info.st_uid,
            "mode": stat.S_IMODE(candidate_info.st_mode),
            "nlink": candidate_info.st_nlink,
            "size": candidate_info.st_size,
        },
        "report_body_sha256": report["generator_attestation"]["report_body_sha256"],
        "generator_attestation_sha256": sha256_bytes(
            stable_json_bytes(report["generator_attestation"])
        ),
        "validation_runtime_observation": validation_runtime_observation,
        "validation_runtime_observation_sha256": sha256_bytes(
            stable_json_bytes(validation_runtime_observation)
        ),
        "frozen_input_observation": frozen_input_observation,
        "verifier_public_key_hex": context["verifier_signing_key"]["public_key_hex"],
    }
    return {
        **payload,
        "signature_hex": _ed25519_sign(
            verifier_private_key_bytes, stable_json_bytes(payload)
        ).hex(),
    }


def _validate_independent_verifier_receipt(
    value: dict[str, Any], report: dict[str, Any]
) -> None:
    _require_exact_keys(
        value, INDEPENDENT_VERIFIER_KEYS, "acceptance independent verifier"
    )
    payload = {key: item for key, item in value.items() if key != "signature_hex"}
    candidate_inode = value["candidate_inode"]
    _require_exact_keys(
        candidate_inode, CANDIDATE_INODE_KEYS, "independent verifier candidate inode"
    )
    if (
        value["schema"] != INDEPENDENT_VERIFIER_SCHEMA
        or value["candidate_sha256"] != sha256_bytes(stable_json_bytes(report))
        or value["report_body_sha256"]
        != report["generator_attestation"]["report_body_sha256"]
        or value["generator_attestation_sha256"]
        != sha256_bytes(stable_json_bytes(report["generator_attestation"]))
        or value["validation_runtime_observation_sha256"]
        != sha256_bytes(stable_json_bytes(value["validation_runtime_observation"]))
        or value["verifier_public_key_hex"]
        != report["wrapper_acceptance"]["verifier_signing_key"]["public_key_hex"]
        or candidate_inode["uid"] != os.getuid()
        or candidate_inode["mode"] != 0o400
        or candidate_inode["nlink"] != 1
        or candidate_inode["size"] != len(stable_json_bytes(report))
        or any(
            type(candidate_inode[key]) is not int or candidate_inode[key] < 0
            for key in CANDIDATE_INODE_KEYS
        )
    ):
        raise ContractError("independent verifier receipt differs")
    if (
        not isinstance(value["signature_hex"], str)
        or re.fullmatch(r"[0-9a-f]{128}", value["signature_hex"]) is None
    ):
        raise ContractError("independent verifier signature encoding differs")
    if not _ed25519_verify(
        bytes.fromhex(value["verifier_public_key_hex"]),
        bytes.fromhex(value["signature_hex"]),
        stable_json_bytes(payload),
    ):
        raise ContractError("independent verifier signature does not verify")
    custody = report["execution"]["runtime_source_manifest"]
    _validate_runtime_observation(
        value["validation_runtime_observation"],
        "independent_validator",
        custody["runtime"]["python"]["path"],
    )
    if value["validation_runtime_observation"]["ld_library_path"] != custody["runtime"][
        "backend"
    ].get("ld_library_path"):
        raise ContractError("validator LD_LIBRARY_PATH differs from runtime seal")
    generator_runtime = report["execution"]["runtime_observation"]
    validator_runtime = value["validation_runtime_observation"]
    for key in (
        "platform",
        "libc_version",
        "ld_library_path",
        "cuda_driver_version",
        "cuda_runtime_version",
        "cuda_visible_devices",
        "cuda_device_name",
        "cuda_device_capability",
        "cuda_device_total_memory_bytes",
        "cuda_device_uuid",
        "nvidia_driver_file",
    ):
        if validator_runtime[key] != generator_runtime[key]:
            raise ContractError(f"validator/generator runtime differs at {key}")
    _validate_frozen_input_observation(value["frozen_input_observation"])


def acceptance_commit_marker_name(accepted_name: str) -> str:
    accepted_name = _validate_directory_entry_name(accepted_name, "accepted name")
    return _validate_directory_entry_name(
        f"{accepted_name}{ACCEPTANCE_COMMIT_SUFFIX}", "acceptance commit marker name"
    )


def durable_acceptance_receipt_name(accepted_name: str) -> str:
    accepted_name = _validate_directory_entry_name(accepted_name, "accepted name")
    return _validate_directory_entry_name(
        f"{accepted_name}{DURABLE_ACCEPTANCE_RECEIPT_SUFFIX}",
        "durable acceptance receipt name",
    )


def _published_file_record(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "size": info.st_size,
    }


def build_accepted_bundle(report: dict[str, Any]) -> dict[str, Any]:
    context = report["wrapper_acceptance"]
    validate_generator_attestation(report, context)
    return {"schema": ACCEPTED_BUNDLE_SCHEMA, "report": report}


def _validate_canonical_report_bundle(
    bundle: dict[str, Any],
    final_info: os.stat_result,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
) -> None:
    _require_exact_keys(bundle, ACCEPTED_BUNDLE_KEYS, "canonical report bundle")
    if bundle["schema"] != ACCEPTED_BUNDLE_SCHEMA:
        raise ContractError("canonical report bundle schema differs")
    payload = stable_json_bytes(bundle)
    if (
        not stat.S_ISREG(final_info.st_mode)
        or stat.S_IMODE(final_info.st_mode) != 0o444
        or final_info.st_nlink != 1
        or final_info.st_uid != os.getuid()
        or final_info.st_size != len(payload)
    ):
        raise ContractError("canonical report bundle final stat contract differs")
    report = bundle["report"]
    context = report["wrapper_acceptance"]
    custody = report["execution"]["runtime_source_manifest"]
    if sha256_bytes(external_manifest_bytes) != context["source_manifest_sha256"]:
        raise ContractError("acceptance external manifest hash differs")
    manifest = _parse_json_object_bytes(
        external_manifest_bytes, "acceptance external manifest"
    )
    observed_manifest_file = {
        "device": external_manifest_info.st_dev,
        "inode": external_manifest_info.st_ino,
        "uid": external_manifest_info.st_uid,
        "mode": stat.S_IMODE(external_manifest_info.st_mode),
        "size": external_manifest_info.st_size,
    }
    if observed_manifest_file != custody["manifest_file"]:
        raise ContractError("acceptance external manifest inode identity differs")
    expected_manifest = {
        "schema": RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "source_root": custody["source_root"],
        "source_commit": custody["source_commit"],
        "files": custody["files"],
        "runtime": custody["runtime"],
    }
    if (
        manifest != expected_manifest
        or stable_json_bytes(manifest) != external_manifest_bytes
    ):
        raise ContractError("acceptance external manifest content differs")
    validate_report_schema(report, context, live_custody=False)


def build_acceptance_commit_marker(
    directory_fd: int,
    canonical_report_descriptor: int,
    bundle: dict[str, Any],
    bundle_payload: bytes,
    final_info: os.stat_result,
    independent_verifier: dict[str, Any],
    commit_marker_info: os.stat_result,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
    committed_at_utc: str,
    commit_nonce: str,
    delegated_marker_private_key_bytes: bytes,
) -> dict[str, Any]:
    report = bundle["report"]
    context = report["wrapper_acceptance"]
    validate_output_directory_fd(
        directory_fd, context["output_directory"], require_path_identity=True
    )
    try:
        canonical_payload, canonical_info = _read_held_directory_entry_descriptor(
            directory_fd,
            context["accepted_name"],
            canonical_report_descriptor,
            "commit-authority canonical report",
            expected_info=final_info,
            expected_payload=bundle_payload,
        )
    except FileNotFoundError as error:
        raise ContractError(
            "commit authority cannot precede canonical report publication"
        ) from error
    _validate_canonical_report_bundle(
        bundle,
        canonical_info,
        external_manifest_bytes,
        external_manifest_info,
    )
    _validate_independent_verifier_receipt(independent_verifier, report)
    if (
        signing_key_record(delegated_marker_private_key_bytes)
        != context["delegated_marker_signing_key"]
    ):
        raise ContractError("delegated marker signing key differs from context")
    _require_utc_timestamp(committed_at_utc, "post-publication commit timestamp")
    if re.fullmatch(r"[0-9a-f]{64}", commit_nonce) is None:
        raise ContractError("post-publication commit nonce differs")
    marker_inode = {
        "device": commit_marker_info.st_dev,
        "inode": commit_marker_info.st_ino,
        "uid": commit_marker_info.st_uid,
    }
    if marker_inode["uid"] != os.getuid() or any(
        not isinstance(marker_inode[key], int)
        or isinstance(marker_inode[key], bool)
        or marker_inode[key] < 0
        for key in ("device", "inode")
    ):
        raise ContractError("post-publication commit marker inode differs")
    receipt_payload = {
        "schema": ACCEPTANCE_MARKER_SCHEMA,
        "status": "wrapper_post_publication_marker_complete",
        "committed_at_utc": committed_at_utc,
        "commit_nonce": commit_nonce,
        "accepted_name": context["accepted_name"],
        "commit_marker_name": acceptance_commit_marker_name(context["accepted_name"]),
        "output_directory": context["output_directory"],
        "final_inode": _published_file_record(final_info),
        "accepted_bundle_sha256": sha256_bytes(bundle_payload),
        "commit_marker_inode": marker_inode,
        "report_sha256": sha256_bytes(stable_json_bytes(report)),
        "report_body_sha256": report["generator_attestation"]["report_body_sha256"],
        "generator_attestation_sha256": sha256_bytes(
            stable_json_bytes(report["generator_attestation"])
        ),
        "independent_verifier": independent_verifier,
        "independent_verifier_sha256": sha256_bytes(
            stable_json_bytes(independent_verifier)
        ),
        "slurm_identity": context["slurm_identity"],
        "wrapper_sha256": context["wrapper_sha256"],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "runtime_identity_sha256": context["runtime_identity_sha256"],
        "frozen_input_observation": independent_verifier["frozen_input_observation"],
        "runtime_coverage_boundary": (
            "generator_and_validator_loaded_file_snapshots_not_a_complete_"
            "immutable_executed_runtime_seal"
        ),
        "post_publication_checks": {
            "canonical_rename_complete": True,
            "canonical_parent_fsync_complete": True,
            "canonical_readback_complete": True,
            "canonical_full_replay_complete": True,
            "wrapper_pre_marker_checks_complete": True,
        },
        "run_authorization_sha256": context["run_authorization_sha256"],
        "authorization_sequence": context["run_authorization"][
            "authorization_sequence"
        ],
        "authority_key_id": context["run_authorization"]["authority_key_id"],
        "authority_public_key_sha256": context["run_authorization"][
            "authority_public_key_sha256"
        ],
        "delegated_marker_public_key_hex": context["delegated_marker_signing_key"][
            "public_key_hex"
        ],
        "delegated_publication_scope": DELEGATED_PUBLICATION_SCOPES[0],
    }
    return {
        **receipt_payload,
        "signature_hex": _ed25519_sign(
            delegated_marker_private_key_bytes, stable_json_bytes(receipt_payload)
        ).hex(),
    }


def validate_acceptance_commit_marker(
    bundle: dict[str, Any],
    final_info: os.stat_result,
    commit_marker: dict[str, Any],
    commit_marker_info: os.stat_result,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
) -> None:
    _validate_canonical_report_bundle(
        bundle, final_info, external_manifest_bytes, external_manifest_info
    )
    _require_exact_keys(
        commit_marker, ACCEPTANCE_MARKER_KEYS, "post-publication commit marker"
    )
    if (
        commit_marker["schema"] != ACCEPTANCE_MARKER_SCHEMA
        or commit_marker["status"] != "wrapper_post_publication_marker_complete"
        or commit_marker["runtime_coverage_boundary"]
        != "generator_and_validator_loaded_file_snapshots_not_a_complete_immutable_"
        "executed_runtime_seal"
    ):
        raise ContractError("post-publication commit marker identity differs")
    _require_utc_timestamp(
        commit_marker["committed_at_utc"], "post-publication commit timestamp"
    )
    if re.fullmatch(r"[0-9a-f]{64}", commit_marker["commit_nonce"]) is None:
        raise ContractError("post-publication commit nonce differs")
    report = bundle["report"]
    context = report["wrapper_acceptance"]
    bundle_payload = stable_json_bytes(bundle)
    marker_payload = {
        key: item for key, item in commit_marker.items() if key != "signature_hex"
    }
    _require_exact_keys(
        commit_marker["post_publication_checks"],
        POST_PUBLICATION_MARKER_CHECK_KEYS,
        "post-publication checks",
    )
    if any(
        value is not True for value in commit_marker["post_publication_checks"].values()
    ):
        raise ContractError("post-publication checks are incomplete")
    if (
        not _strict_equal(
            commit_marker["output_directory"], context["output_directory"]
        )
        or not _strict_equal(commit_marker["slurm_identity"], context["slurm_identity"])
        or not _strict_equal(
            commit_marker["frozen_input_observation"],
            commit_marker["independent_verifier"]["frozen_input_observation"],
        )
    ):
        raise ContractError("post-publication typed custody binding differs")
    if (
        commit_marker["accepted_name"] != context["accepted_name"]
        or commit_marker["commit_marker_name"]
        != acceptance_commit_marker_name(context["accepted_name"])
        or commit_marker["output_directory"] != context["output_directory"]
        or commit_marker["slurm_identity"] != context["slurm_identity"]
        or commit_marker["wrapper_sha256"] != context["wrapper_sha256"]
        or commit_marker["source_manifest_sha256"] != context["source_manifest_sha256"]
        or commit_marker["runtime_identity_sha256"]
        != context["runtime_identity_sha256"]
        or commit_marker["accepted_bundle_sha256"] != sha256_bytes(bundle_payload)
        or commit_marker["report_sha256"] != sha256_bytes(stable_json_bytes(report))
        or commit_marker["report_body_sha256"]
        != report["generator_attestation"]["report_body_sha256"]
        or commit_marker["generator_attestation_sha256"]
        != sha256_bytes(stable_json_bytes(report["generator_attestation"]))
        or commit_marker["independent_verifier_sha256"]
        != sha256_bytes(stable_json_bytes(commit_marker["independent_verifier"]))
        or commit_marker["run_authorization_sha256"]
        != context["run_authorization_sha256"]
        or type(commit_marker["authorization_sequence"]) is not int
        or commit_marker["authorization_sequence"]
        != context["run_authorization"]["authorization_sequence"]
        or commit_marker["authority_key_id"]
        != context["run_authorization"]["authority_key_id"]
        or commit_marker["authority_public_key_sha256"]
        != context["run_authorization"]["authority_public_key_sha256"]
        or commit_marker["delegated_marker_public_key_hex"]
        != context["delegated_marker_signing_key"]["public_key_hex"]
        or commit_marker["delegated_publication_scope"]
        != DELEGATED_PUBLICATION_SCOPES[0]
    ):
        raise ContractError("post-publication commit marker cross-binding differs")
    if (
        not isinstance(commit_marker["signature_hex"], str)
        or re.fullmatch(r"[0-9a-f]{128}", commit_marker["signature_hex"]) is None
    ):
        raise ContractError("post-publication commit signature encoding differs")
    if not _ed25519_verify(
        bytes.fromhex(commit_marker["delegated_marker_public_key_hex"]),
        bytes.fromhex(commit_marker["signature_hex"]),
        stable_json_bytes(marker_payload),
    ):
        raise ContractError("post-publication commit signature does not verify")
    _require_exact_keys(commit_marker["final_inode"], FINAL_INODE_KEYS, "final inode")
    if not _strict_equal(
        commit_marker["final_inode"], _published_file_record(final_info)
    ):
        raise ContractError("post-publication commit final inode differs")
    _require_exact_keys(
        commit_marker["commit_marker_inode"],
        COMMIT_MARKER_INODE_KEYS,
        "commit marker inode",
    )
    expected_marker_inode = {
        "device": commit_marker_info.st_dev,
        "inode": commit_marker_info.st_ino,
        "uid": commit_marker_info.st_uid,
    }
    if not _strict_equal(commit_marker["commit_marker_inode"], expected_marker_inode):
        raise ContractError("post-publication commit marker inode differs")
    if (
        not stat.S_ISREG(commit_marker_info.st_mode)
        or stat.S_IMODE(commit_marker_info.st_mode) != 0o444
        or commit_marker_info.st_nlink != 1
        or commit_marker_info.st_uid != os.getuid()
        or commit_marker_info.st_size != len(stable_json_bytes(commit_marker))
    ):
        raise ContractError("post-publication commit marker stat contract differs")
    _validate_independent_verifier_receipt(
        commit_marker["independent_verifier"], report
    )
    _validate_frozen_input_observation(commit_marker["frozen_input_observation"])


def build_durable_acceptance_receipt(
    directory_fd: int,
    canonical_report_descriptor: int,
    commit_marker_descriptor: int,
    bundle: dict[str, Any],
    bundle_payload: bytes,
    final_info: os.stat_result,
    commit_marker: dict[str, Any],
    commit_marker_payload: bytes,
    commit_marker_info: os.stat_result,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
    receipt_slot_info: os.stat_result,
    witnessed_at_utc: str,
    witness_nonce: str,
    delegated_marker_private_key_bytes: bytes,
) -> dict[str, Any]:
    report = bundle["report"]
    context = report["wrapper_acceptance"]
    validate_output_directory_fd(
        directory_fd, context["output_directory"], require_path_identity=True
    )
    _read_held_directory_entry_descriptor(
        directory_fd,
        context["accepted_name"],
        canonical_report_descriptor,
        "durability receipt canonical report",
        expected_info=final_info,
        expected_payload=bundle_payload,
    )
    _read_held_directory_entry_descriptor(
        directory_fd,
        acceptance_commit_marker_name(context["accepted_name"]),
        commit_marker_descriptor,
        "durability receipt commit marker",
        expected_info=commit_marker_info,
        expected_payload=commit_marker_payload,
    )
    validate_acceptance_commit_marker(
        bundle,
        final_info,
        commit_marker,
        commit_marker_info,
        external_manifest_bytes,
        external_manifest_info,
    )
    if bundle_payload != stable_json_bytes(bundle):
        raise ContractError("durability receipt canonical bundle bytes differ")
    if commit_marker_payload != stable_json_bytes(commit_marker):
        raise ContractError("durability receipt commit marker bytes differ")
    if (
        not stat.S_ISREG(receipt_slot_info.st_mode)
        or stat.S_IMODE(receipt_slot_info.st_mode) != 0o444
        or receipt_slot_info.st_nlink != 1
        or receipt_slot_info.st_uid != os.getuid()
        or receipt_slot_info.st_size != 0
    ):
        raise ContractError("durable acceptance receipt slot is not empty and sealed")
    if (
        signing_key_record(delegated_marker_private_key_bytes)
        != context["delegated_marker_signing_key"]
    ):
        raise ContractError("durable acceptance receipt signing key differs")
    _require_utc_timestamp(witnessed_at_utc, "durable acceptance witness timestamp")
    if re.fullmatch(r"[0-9a-f]{64}", witness_nonce) is None:
        raise ContractError("durable acceptance witness nonce differs")
    receipt_inode = {
        "device": receipt_slot_info.st_dev,
        "inode": receipt_slot_info.st_ino,
        "uid": receipt_slot_info.st_uid,
    }
    receipt_payload = {
        "schema": DURABLE_ACCEPTANCE_RECEIPT_SCHEMA,
        "status": "wrapper_durable_post_fsync_acceptance_complete",
        "witnessed_at_utc": witnessed_at_utc,
        "witness_nonce": witness_nonce,
        "accepted_name": context["accepted_name"],
        "commit_marker_name": acceptance_commit_marker_name(context["accepted_name"]),
        "durable_acceptance_receipt_name": durable_acceptance_receipt_name(
            context["accepted_name"]
        ),
        "output_directory": context["output_directory"],
        "final_inode": _published_file_record(final_info),
        "commit_marker_inode": _published_file_record(commit_marker_info),
        "durable_acceptance_receipt_inode": receipt_inode,
        "accepted_bundle_sha256": sha256_bytes(bundle_payload),
        "commit_marker_sha256": sha256_bytes(commit_marker_payload),
        "report_sha256": commit_marker["report_sha256"],
        "run_authorization_sha256": context["run_authorization_sha256"],
        "authorization_sequence": context["run_authorization"][
            "authorization_sequence"
        ],
        "authority_key_id": context["run_authorization"]["authority_key_id"],
        "authority_public_key_sha256": context["run_authorization"][
            "authority_public_key_sha256"
        ],
        "delegated_marker_public_key_hex": context["delegated_marker_signing_key"][
            "public_key_hex"
        ],
        "delegated_publication_scope": DELEGATED_PUBLICATION_SCOPES[1],
        "durability_checks": {
            "receipt_slot_parent_fsync_complete": True,
            "canonical_report_reopen_fsync_complete": True,
            "commit_marker_reopen_fsync_complete": True,
            "publication_parent_fsync_complete": True,
            "wrapper_final_checks_complete": True,
            "receipt_o_sync_write_complete": True,
        },
    }
    return {
        **receipt_payload,
        "signature_hex": _ed25519_sign(
            delegated_marker_private_key_bytes, stable_json_bytes(receipt_payload)
        ).hex(),
    }


def validate_durable_acceptance_receipt(
    bundle: dict[str, Any],
    final_info: os.stat_result,
    commit_marker: dict[str, Any],
    commit_marker_info: os.stat_result,
    durable_receipt: dict[str, Any],
    durable_receipt_info: os.stat_result,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
) -> None:
    validate_acceptance_commit_marker(
        bundle,
        final_info,
        commit_marker,
        commit_marker_info,
        external_manifest_bytes,
        external_manifest_info,
    )
    _require_exact_keys(
        durable_receipt,
        DURABLE_ACCEPTANCE_RECEIPT_KEYS,
        "durable post-fsync acceptance receipt",
    )
    if (
        durable_receipt["schema"] != DURABLE_ACCEPTANCE_RECEIPT_SCHEMA
        or durable_receipt["status"] != "wrapper_durable_post_fsync_acceptance_complete"
    ):
        raise ContractError("durable acceptance receipt identity differs")
    _require_utc_timestamp(
        durable_receipt["witnessed_at_utc"],
        "durable acceptance witness timestamp",
    )
    if re.fullmatch(r"[0-9a-f]{64}", durable_receipt["witness_nonce"]) is None:
        raise ContractError("durable acceptance witness nonce differs")
    _require_exact_keys(
        durable_receipt["durability_checks"],
        DURABLE_ACCEPTANCE_CHECK_KEYS,
        "durable acceptance checks",
    )
    if any(
        value is not True for value in durable_receipt["durability_checks"].values()
    ):
        raise ContractError("durable acceptance checks are incomplete")
    report = bundle["report"]
    context = report["wrapper_acceptance"]
    if (
        durable_receipt["accepted_name"] != context["accepted_name"]
        or durable_receipt["commit_marker_name"]
        != acceptance_commit_marker_name(context["accepted_name"])
        or durable_receipt["durable_acceptance_receipt_name"]
        != durable_acceptance_receipt_name(context["accepted_name"])
        or not _strict_equal(
            durable_receipt["output_directory"], context["output_directory"]
        )
        or durable_receipt["accepted_bundle_sha256"]
        != sha256_bytes(stable_json_bytes(bundle))
        or durable_receipt["commit_marker_sha256"]
        != sha256_bytes(stable_json_bytes(commit_marker))
        or durable_receipt["report_sha256"] != commit_marker["report_sha256"]
        or durable_receipt["run_authorization_sha256"]
        != context["run_authorization_sha256"]
        or type(durable_receipt["authorization_sequence"]) is not int
        or durable_receipt["authorization_sequence"]
        != context["run_authorization"]["authorization_sequence"]
        or durable_receipt["authority_key_id"]
        != context["run_authorization"]["authority_key_id"]
        or durable_receipt["authority_public_key_sha256"]
        != context["run_authorization"]["authority_public_key_sha256"]
        or durable_receipt["delegated_marker_public_key_hex"]
        != context["delegated_marker_signing_key"]["public_key_hex"]
        or durable_receipt["delegated_publication_scope"]
        != DELEGATED_PUBLICATION_SCOPES[1]
    ):
        raise ContractError("durable acceptance receipt cross-binding differs")
    _require_exact_keys(
        durable_receipt["final_inode"], FINAL_INODE_KEYS, "durable final inode"
    )
    _require_exact_keys(
        durable_receipt["commit_marker_inode"],
        FINAL_INODE_KEYS,
        "durable commit marker inode",
    )
    _require_exact_keys(
        durable_receipt["durable_acceptance_receipt_inode"],
        DURABLE_ACCEPTANCE_RECEIPT_INODE_KEYS,
        "durable acceptance receipt inode",
    )
    expected_receipt_inode = {
        "device": durable_receipt_info.st_dev,
        "inode": durable_receipt_info.st_ino,
        "uid": durable_receipt_info.st_uid,
    }
    if (
        not _strict_equal(
            durable_receipt["final_inode"], _published_file_record(final_info)
        )
        or not _strict_equal(
            durable_receipt["commit_marker_inode"],
            _published_file_record(commit_marker_info),
        )
        or not _strict_equal(
            durable_receipt["durable_acceptance_receipt_inode"],
            expected_receipt_inode,
        )
    ):
        raise ContractError("durable acceptance receipt inode binding differs")
    if (
        not stat.S_ISREG(durable_receipt_info.st_mode)
        or stat.S_IMODE(durable_receipt_info.st_mode) != 0o444
        or durable_receipt_info.st_nlink != 1
        or durable_receipt_info.st_uid != os.getuid()
        or durable_receipt_info.st_size != len(stable_json_bytes(durable_receipt))
    ):
        raise ContractError("durable acceptance receipt stat contract differs")
    signature_payload = {
        key: item for key, item in durable_receipt.items() if key != "signature_hex"
    }
    if (
        not isinstance(durable_receipt["signature_hex"], str)
        or re.fullmatch(r"[0-9a-f]{128}", durable_receipt["signature_hex"]) is None
    ):
        raise ContractError("durable acceptance receipt signature encoding differs")
    if not _ed25519_verify(
        bytes.fromhex(durable_receipt["delegated_marker_public_key_hex"]),
        bytes.fromhex(durable_receipt["signature_hex"]),
        stable_json_bytes(signature_payload),
    ):
        raise ContractError("durable acceptance receipt signature does not verify")


def _read_and_validate_marker_publication_from_descriptors(
    directory_fd: int,
    accepted_name: str,
    report_descriptor: int,
    marker_descriptor: int,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
    *,
    expected_report_payload: bytes | None = None,
    expected_report_info: os.stat_result | None = None,
    expected_marker_payload: bytes | None = None,
    expected_marker_info: os.stat_result | None = None,
) -> tuple[
    bytes, os.stat_result, bytes, os.stat_result, dict[str, Any], dict[str, Any]
]:
    accepted_name = _validate_directory_entry_name(accepted_name, "accepted name")
    payload, final_info = _read_held_directory_entry_descriptor(
        directory_fd,
        accepted_name,
        report_descriptor,
        "canonical report bundle",
        expected_info=expected_report_info,
        expected_payload=expected_report_payload,
    )
    bundle = _parse_json_object_bytes(payload, "canonical report bundle")
    if stable_json_bytes(bundle) != payload:
        raise ContractError("canonical report bundle is not canonical JSON")
    marker_name = acceptance_commit_marker_name(accepted_name)
    marker_payload, marker_info = _read_held_directory_entry_descriptor(
        directory_fd,
        marker_name,
        marker_descriptor,
        "post-publication commit marker",
        expected_info=expected_marker_info,
        expected_payload=expected_marker_payload,
    )
    commit_marker = _parse_json_object_bytes(
        marker_payload, "post-publication commit marker"
    )
    if stable_json_bytes(commit_marker) != marker_payload:
        raise ContractError("post-publication commit marker is not canonical JSON")
    validate_acceptance_commit_marker(
        bundle,
        final_info,
        commit_marker,
        marker_info,
        external_manifest_bytes,
        external_manifest_info,
    )
    return (
        payload,
        final_info,
        marker_payload,
        marker_info,
        bundle,
        commit_marker,
    )


def _read_and_validate_marker_publication(
    directory_fd: int,
    accepted_name: str,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
) -> tuple[
    bytes, os.stat_result, bytes, os.stat_result, dict[str, Any], dict[str, Any]
]:
    accepted_name = _validate_directory_entry_name(accepted_name, "accepted name")
    marker_name = acceptance_commit_marker_name(accepted_name)
    report_descriptor = os.open(
        accepted_name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    marker_descriptor: int | None = None
    try:
        try:
            marker_descriptor = os.open(
                marker_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError as error:
            raise ContractError(
                "canonical report has no durable post-publication commit marker"
            ) from error
        return _read_and_validate_marker_publication_from_descriptors(
            directory_fd,
            accepted_name,
            report_descriptor,
            marker_descriptor,
            external_manifest_bytes,
            external_manifest_info,
        )
    finally:
        os.close(report_descriptor)
        if marker_descriptor is not None:
            os.close(marker_descriptor)


def _fsync_held_canonical_publication(
    directory_fd: int,
    report_name: str,
    marker_name: str,
    report_descriptor: int,
    marker_descriptor: int,
    expected_report_payload: bytes,
    expected_marker_payload: bytes,
    expected_final_info: os.stat_result,
    expected_marker_info: os.stat_result,
) -> None:
    entries = (
        (
            report_name,
            report_descriptor,
            "canonical report durability hold",
            expected_final_info,
            expected_report_payload,
        ),
        (
            marker_name,
            marker_descriptor,
            "commit marker durability hold",
            expected_marker_info,
            expected_marker_payload,
        ),
    )
    for name, descriptor, label, expected_info, expected_payload in entries:
        _read_held_directory_entry_descriptor(
            directory_fd,
            name,
            descriptor,
            label,
            expected_info=expected_info,
            expected_payload=expected_payload,
        )
        os.fsync(descriptor)
        _read_held_directory_entry_descriptor(
            directory_fd,
            name,
            descriptor,
            label,
            expected_info=expected_info,
            expected_payload=expected_payload,
        )
    os.fsync(directory_fd)


def _read_and_validate_held_publication_triplet(
    directory_fd: int,
    report_name: str,
    marker_name: str,
    receipt_name: str,
    report_descriptor: int,
    marker_descriptor: int,
    receipt_descriptor: int,
    parse_report_artifact: Any,
    parse_marker_artifact: Any,
    parse_receipt_artifact: Any,
    *,
    expected_report_payload: bytes | None = None,
    expected_report_info: os.stat_result | None = None,
    expected_marker_payload: bytes | None = None,
    expected_marker_info: os.stat_result | None = None,
    expected_receipt_payload: bytes | None = None,
    expected_receipt_info: os.stat_result | None = None,
) -> tuple[
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    report_payload, report_info = _read_held_directory_entry_descriptor(
        directory_fd,
        report_name,
        report_descriptor,
        "publication report replay",
        expected_info=expected_report_info,
        expected_payload=expected_report_payload,
    )
    report = parse_report_artifact(report_payload, report_info)
    marker_payload, marker_info = _read_held_directory_entry_descriptor(
        directory_fd,
        marker_name,
        marker_descriptor,
        "publication marker replay",
        expected_info=expected_marker_info,
        expected_payload=expected_marker_payload,
    )
    marker = parse_marker_artifact(report, report_info, marker_payload, marker_info)
    receipt_payload, receipt_info = _read_held_directory_entry_descriptor(
        directory_fd,
        receipt_name,
        receipt_descriptor,
        "publication receipt replay",
        expected_info=expected_receipt_info,
        expected_payload=expected_receipt_payload,
    )
    if not receipt_payload:
        raise ContractError("publication has no valid complete receipt")
    receipt = parse_receipt_artifact(
        report,
        report_info,
        marker,
        marker_info,
        receipt_payload,
        receipt_info,
    )
    for name, descriptor, label, info, payload in (
        (
            report_name,
            report_descriptor,
            "publication report final hold",
            report_info,
            report_payload,
        ),
        (
            marker_name,
            marker_descriptor,
            "publication marker final hold",
            marker_info,
            marker_payload,
        ),
        (
            receipt_name,
            receipt_descriptor,
            "publication receipt final hold",
            receipt_info,
            receipt_payload,
        ),
    ):
        _read_held_directory_entry_descriptor(
            directory_fd,
            name,
            descriptor,
            label,
            expected_info=info,
            expected_payload=payload,
        )
    return (
        report_payload,
        report_info,
        marker_payload,
        marker_info,
        receipt_payload,
        receipt_info,
        report,
        marker,
        receipt,
    )


def _read_and_validate_accepted_publication_from_descriptors(
    directory_fd: int,
    accepted_name: str,
    report_descriptor: int,
    marker_descriptor: int,
    receipt_descriptor: int,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
    *,
    expected_report_payload: bytes | None = None,
    expected_report_info: os.stat_result | None = None,
    expected_marker_payload: bytes | None = None,
    expected_marker_info: os.stat_result | None = None,
    expected_receipt_payload: bytes | None = None,
    expected_receipt_info: os.stat_result | None = None,
) -> tuple[
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    def parse_report(payload: bytes, info: os.stat_result) -> dict[str, Any]:
        bundle = _parse_json_object_bytes(payload, "canonical report bundle")
        if stable_json_bytes(bundle) != payload:
            raise ContractError("canonical report bundle is not canonical JSON")
        _validate_canonical_report_bundle(
            bundle, info, external_manifest_bytes, external_manifest_info
        )
        return bundle

    def parse_marker(
        bundle: dict[str, Any],
        report_info: os.stat_result,
        payload: bytes,
        info: os.stat_result,
    ) -> dict[str, Any]:
        marker = _parse_json_object_bytes(payload, "post-publication commit marker")
        if stable_json_bytes(marker) != payload:
            raise ContractError("post-publication commit marker is not canonical JSON")
        validate_acceptance_commit_marker(
            bundle,
            report_info,
            marker,
            info,
            external_manifest_bytes,
            external_manifest_info,
        )
        return marker

    def parse_receipt(
        bundle: dict[str, Any],
        report_info: os.stat_result,
        marker: dict[str, Any],
        marker_info: os.stat_result,
        payload: bytes,
        info: os.stat_result,
    ) -> dict[str, Any]:
        try:
            receipt = _parse_json_object_bytes(
                payload, "durable post-fsync acceptance receipt"
            )
        except ContractError as error:
            raise ContractError(
                "publication has no valid durable post-fsync acceptance receipt"
            ) from error
        if stable_json_bytes(receipt) != payload:
            raise ContractError("durable acceptance receipt is not canonical JSON")
        validate_durable_acceptance_receipt(
            bundle,
            report_info,
            marker,
            marker_info,
            receipt,
            info,
            external_manifest_bytes,
            external_manifest_info,
        )
        return receipt

    try:
        return _read_and_validate_held_publication_triplet(
            directory_fd,
            accepted_name,
            acceptance_commit_marker_name(accepted_name),
            durable_acceptance_receipt_name(accepted_name),
            report_descriptor,
            marker_descriptor,
            receipt_descriptor,
            parse_report,
            parse_marker,
            parse_receipt,
            expected_report_payload=expected_report_payload,
            expected_report_info=expected_report_info,
            expected_marker_payload=expected_marker_payload,
            expected_marker_info=expected_marker_info,
            expected_receipt_payload=expected_receipt_payload,
            expected_receipt_info=expected_receipt_info,
        )
    except ContractError as error:
        if str(error) == "publication has no valid complete receipt":
            raise ContractError(
                "publication has no valid durable post-fsync acceptance receipt"
            ) from error
        raise


def read_and_validate_accepted_publication(
    directory_fd: int,
    accepted_name: str,
    external_manifest_bytes: bytes,
    external_manifest_info: os.stat_result,
) -> tuple[
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    accepted_name = _validate_directory_entry_name(accepted_name, "accepted name")
    marker_name = acceptance_commit_marker_name(accepted_name)
    receipt_name = durable_acceptance_receipt_name(accepted_name)
    descriptors: list[int] = []
    try:
        for name in (accepted_name, marker_name, receipt_name):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except FileNotFoundError as error:
                if name == marker_name:
                    message = (
                        "canonical report has no durable post-publication commit marker"
                    )
                elif name == receipt_name:
                    message = "publication has no durable post-fsync acceptance receipt"
                else:
                    raise
                raise ContractError(message) from error
            descriptors.append(descriptor)
        return _read_and_validate_accepted_publication_from_descriptors(
            directory_fd,
            accepted_name,
            descriptors[0],
            descriptors[1],
            descriptors[2],
            external_manifest_bytes,
            external_manifest_info,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("publication write made no progress")
        view = view[written:]


def _open_durable_o_sync_receipt_slot(
    directory_fd: int, receipt_name: str
) -> tuple[int, os.stat_result]:
    _validate_directory_entry_name(receipt_name, "durable receipt slot name")
    sync_open_flag = getattr(os, "O_SYNC", None)
    if type(sync_open_flag) is not int or sync_open_flag == 0:
        raise ContractError("durable acceptance receipt requires O_SYNC")
    descriptor = os.open(
        receipt_name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | sync_open_flag,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        slot_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(slot_info.st_mode)
            or stat.S_IMODE(slot_info.st_mode) != 0o444
            or slot_info.st_nlink != 1
            or slot_info.st_uid != os.getuid()
            or slot_info.st_size != 0
        ):
            raise ContractError("durable acceptance receipt slot contract differs")
        return descriptor, slot_info
    except BaseException:
        os.close(descriptor)
        raise


def _write_complete_o_sync_receipt(
    descriptor: int, payload: bytes, *, after_sync_write: Any
) -> os.stat_result:
    if not payload or after_sync_write is None:
        raise ContractError("complete O_SYNC receipt write contract differs")
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or before.st_size != 0
    ):
        raise ContractError("O_SYNC receipt descriptor is not an empty sealed slot")
    _write_all(descriptor, payload)
    written = os.fstat(descriptor)
    if _inode_policy_identity(written) != _inode_policy_identity(
        before
    ) or written.st_size != len(payload):
        raise ContractError("O_SYNC receipt inode changed during write")
    after_sync_write()
    os.fsync(descriptor)
    final_info = os.fstat(descriptor)
    if _inode_policy_identity(final_info) != _inode_policy_identity(
        before
    ) or final_info.st_size != len(payload):
        raise ContractError("O_SYNC receipt inode changed after fsync")
    _read_exact_descriptor_bytes(
        descriptor,
        "durable acceptance receipt",
        expected_info=final_info,
        expected_payload=payload,
    )
    return final_info


def cleanup_acceptance_transaction_entries(
    directory_fd: int, accepted_name: str, nonce: str
) -> None:
    _validate_directory_entry_name(accepted_name, "accepted name")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ContractError("acceptance cleanup nonce differs")
    prefixes = (
        f".{accepted_name}.r12-report-{nonce}-",
        f".{acceptance_commit_marker_name(accepted_name)}.r12-commit-{nonce}-",
    )
    for name in os.listdir(directory_fd):
        if name.startswith(prefixes):
            raise ContractError(
                "acceptance residue requires external cleanup authorization"
            )


def _quarantine_held_entry_if_exact(
    directory_fd: int,
    accepted_name: str,
    nonce: str,
    name: str,
    descriptor: int | None,
    expected_info: os.stat_result | None,
    expected_payload: bytes,
    label: str,
) -> tuple[bool, str | None]:
    """Rename a held exact inode to quarantine without any pathname unlink."""
    if _entry_stat_or_none(directory_fd, name) is None:
        return True, None
    if descriptor is None or expected_info is None:
        return False, None
    quarantine_name = _rollback_quarantine_name(
        accepted_name, nonce, name, expected_info, expected_payload
    )
    if _entry_stat_or_none(directory_fd, quarantine_name) is not None:
        return False, quarantine_name
    try:
        _read_held_directory_entry_descriptor(
            directory_fd,
            name,
            descriptor,
            label,
            expected_info=expected_info,
            expected_payload=expected_payload,
        )
    except (ContractError, FileNotFoundError):
        return False, quarantine_name
    try:
        rename_noreplace_at(directory_fd, name, directory_fd, quarantine_name)
    except OSError:
        # A no-replace implementation may report an error after moving the entry.
        # The held post-rename check below is the sole success criterion.
        pass
    try:
        os.fsync(directory_fd)
        _read_held_directory_entry_descriptor(
            directory_fd,
            quarantine_name,
            descriptor,
            f"{label} quarantine",
            expected_info=expected_info,
            expected_payload=expected_payload,
        )
    except (ContractError, FileNotFoundError, OSError):
        return False, quarantine_name
    if _entry_stat_or_none(directory_fd, name) is not None:
        return False, quarantine_name
    return True, quarantine_name


def publish_accepted_bundle_exclusive(
    directory_fd: int,
    report: dict[str, Any],
    independent_verifier: dict[str, Any] | None,
    external_manifest_bytes: bytes | None,
    external_manifest_info: os.stat_result | None,
    delegated_marker_private_key_bytes: bytes,
    *,
    failure_injector: Any,
    qualification_contract: dict[str, Any] | None = None,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    """Publish report and marker, then durably witness acceptance in a preflushed inode."""
    if failure_injector is None:
        raise ContractError("wrapper final-verification callback is mandatory")
    if qualification_contract is None:
        if (
            independent_verifier is None
            or external_manifest_bytes is None
            or external_manifest_info is None
        ):
            raise ContractError("production publication custody inputs are missing")
        context = report["wrapper_acceptance"]
        validate_report_schema(report, context, live_custody=False)
        _validate_independent_verifier_receipt(independent_verifier, report)
        accepted_name = _validate_directory_entry_name(
            context["accepted_name"], "accepted name"
        )
        commit_marker_name = acceptance_commit_marker_name(accepted_name)
        durable_receipt_name = durable_acceptance_receipt_name(accepted_name)
        nonce = context["nonce"]
        bundle = build_accepted_bundle(report)
        initial_payload = stable_json_bytes(bundle)

        def parse_report_artifact(
            payload: bytes, info: os.stat_result
        ) -> dict[str, Any]:
            observed = _parse_json_object_bytes(payload, "canonical report bundle")
            if stable_json_bytes(observed) != payload:
                raise ContractError("canonical report bundle is not canonical JSON")
            _validate_canonical_report_bundle(
                observed,
                info,
                external_manifest_bytes,
                external_manifest_info,
            )
            return observed

        def build_marker_artifact(
            report_descriptor: int,
            observed_bundle: dict[str, Any],
            payload: bytes,
            info: os.stat_result,
            marker_slot_info: os.stat_result,
        ) -> tuple[dict[str, Any], bytes]:
            marker = build_acceptance_commit_marker(
                directory_fd,
                report_descriptor,
                observed_bundle,
                payload,
                info,
                independent_verifier,
                marker_slot_info,
                external_manifest_bytes,
                external_manifest_info,
                datetime.now(timezone.utc).isoformat(),
                secrets.token_hex(32),
                delegated_marker_private_key_bytes,
            )
            return marker, stable_json_bytes(marker)

        def parse_marker_artifact(
            observed_bundle: dict[str, Any],
            report_info: os.stat_result,
            payload: bytes,
            marker_info: os.stat_result,
        ) -> dict[str, Any]:
            marker = _parse_json_object_bytes(payload, "post-publication commit marker")
            if stable_json_bytes(marker) != payload:
                raise ContractError(
                    "post-publication commit marker is not canonical JSON"
                )
            validate_acceptance_commit_marker(
                observed_bundle,
                report_info,
                marker,
                marker_info,
                external_manifest_bytes,
                external_manifest_info,
            )
            return marker

        def build_receipt_artifact(
            report_descriptor: int,
            marker_descriptor: int,
            observed_bundle: dict[str, Any],
            report_payload: bytes,
            report_info: os.stat_result,
            marker: dict[str, Any],
            marker_payload: bytes,
            marker_info: os.stat_result,
            receipt_slot_info: os.stat_result,
        ) -> tuple[dict[str, Any], bytes]:
            receipt = build_durable_acceptance_receipt(
                directory_fd,
                report_descriptor,
                marker_descriptor,
                observed_bundle,
                report_payload,
                report_info,
                marker,
                marker_payload,
                marker_info,
                external_manifest_bytes,
                external_manifest_info,
                receipt_slot_info,
                datetime.now(timezone.utc).isoformat(),
                secrets.token_hex(32),
                delegated_marker_private_key_bytes,
            )
            return receipt, stable_json_bytes(receipt)

        def parse_receipt_artifact(
            observed_bundle: dict[str, Any],
            report_info: os.stat_result,
            marker: dict[str, Any],
            marker_info: os.stat_result,
            payload: bytes,
            receipt_info: os.stat_result,
        ) -> dict[str, Any]:
            receipt = _parse_json_object_bytes(
                payload, "durable post-fsync acceptance receipt"
            )
            if stable_json_bytes(receipt) != payload:
                raise ContractError("durable acceptance receipt is not canonical JSON")
            validate_durable_acceptance_receipt(
                observed_bundle,
                report_info,
                marker,
                marker_info,
                receipt,
                receipt_info,
                external_manifest_bytes,
                external_manifest_info,
            )
            return receipt

        def before_receipt_write(_descriptor: int, _payload: bytes) -> None:
            return None

        transaction_rename_noreplace = rename_noreplace_at

    else:
        required_contract_keys = {
            "accepted_name",
            "commit_marker_name",
            "durable_receipt_name",
            "nonce",
            "output_directory",
            "initial_payload",
            "parse_report_artifact",
            "build_marker_artifact",
            "parse_marker_artifact",
            "build_receipt_artifact",
            "parse_receipt_artifact",
            "before_receipt_write",
            "rename_noreplace",
        }
        if set(qualification_contract) != required_contract_keys:
            raise ContractError("qualification publication contract keys differ")
        context = {"output_directory": qualification_contract["output_directory"]}
        accepted_name = _validate_directory_entry_name(
            qualification_contract["accepted_name"], "accepted name"
        )
        commit_marker_name = _validate_directory_entry_name(
            qualification_contract["commit_marker_name"], "commit marker name"
        )
        durable_receipt_name = _validate_directory_entry_name(
            qualification_contract["durable_receipt_name"], "durable receipt name"
        )
        nonce = qualification_contract["nonce"]
        initial_payload = qualification_contract["initial_payload"]
        parse_report_artifact = qualification_contract["parse_report_artifact"]
        build_marker_artifact = qualification_contract["build_marker_artifact"]
        parse_marker_artifact = qualification_contract["parse_marker_artifact"]
        build_receipt_artifact = qualification_contract["build_receipt_artifact"]
        parse_receipt_artifact = qualification_contract["parse_receipt_artifact"]
        before_receipt_write = qualification_contract["before_receipt_write"]
        transaction_rename_noreplace = qualification_contract["rename_noreplace"]
        if not isinstance(initial_payload, bytes) or not initial_payload:
            raise ContractError("qualification publication payload differs")
    validate_output_directory_fd(
        directory_fd, context["output_directory"], require_path_identity=True
    )
    for name in (accepted_name, commit_marker_name, durable_receipt_name):
        if _entry_stat_or_none(directory_fd, name) is not None:
            raise FileExistsError(f"refusing to overwrite accepted output: {name}")
    if re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise ContractError("publication transaction nonce differs")
    prefixes = (
        f".{accepted_name}.r12-report-{nonce}-",
        f".{commit_marker_name}.r12-commit-{nonce}-",
    )
    if any(name.startswith(prefixes) for name in os.listdir(directory_fd)):
        raise ContractError(
            "publication residue requires external cleanup authorization"
        )
    temp_name = f".{accepted_name}.r12-report-{nonce}-{secrets.token_hex(16)}"
    report_descriptor = os.open(
        temp_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    published_info: os.stat_result | None = None
    commit_marker_info: os.stat_result | None = None
    durable_receipt_info: os.stat_result | None = None
    report_owned_payload = b""
    commit_owned_payload = b""
    receipt_owned_payload = b""
    commit_temp_name: str | None = None
    commit_descriptor: int | None = None
    durable_receipt_descriptor: int | None = None
    try:
        payload = initial_payload
        _write_all(report_descriptor, payload)
        os.fsync(report_descriptor)
        if os.pread(report_descriptor, len(payload) + 1, 0) != payload:
            raise ContractError("acceptance temp descriptor readback differs")
        os.fchmod(report_descriptor, 0o444)
        os.fsync(report_descriptor)
        sealed_info = os.fstat(report_descriptor)
        published_info = sealed_info
        report_owned_payload = payload
        if (
            not stat.S_ISREG(sealed_info.st_mode)
            or stat.S_IMODE(sealed_info.st_mode) != 0o444
            or sealed_info.st_nlink != 1
            or sealed_info.st_uid != os.getuid()
        ):
            raise ContractError("acceptance temp stat contract differs")
        parse_report_artifact(payload, sealed_info)
        failure_injector("before_report_rename")
        transaction_rename_noreplace(
            directory_fd, temp_name, directory_fd, accepted_name
        )
        failure_injector("after_report_rename")
        os.fsync(directory_fd)
        failure_injector("after_report_parent_fsync")
        validate_output_directory_fd(
            directory_fd, context["output_directory"], require_path_identity=True
        )
        final_payload, final_info = _read_held_directory_entry_descriptor(
            directory_fd,
            accepted_name,
            report_descriptor,
            "accepted bundle",
            expected_info=sealed_info,
            expected_payload=payload,
        )
        final_bundle = parse_report_artifact(final_payload, final_info)
        failure_injector("after_final_readback")
        failure_injector("before_durable_acceptance_receipt_slot_create")
        durable_receipt_descriptor, receipt_slot_info = (
            _open_durable_o_sync_receipt_slot(directory_fd, durable_receipt_name)
        )
        durable_receipt_info = receipt_slot_info
        os.fsync(directory_fd)
        failure_injector("after_durable_acceptance_receipt_slot_parent_fsync")
        failure_injector("before_commit_marker_create")
        commit_temp_name = (
            f".{commit_marker_name}.r12-commit-{nonce}-{secrets.token_hex(16)}"
        )
        commit_descriptor = os.open(
            commit_temp_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        commit_temp_info = os.fstat(commit_descriptor)
        commit_marker, commit_payload = build_marker_artifact(
            report_descriptor,
            final_bundle,
            final_payload,
            final_info,
            commit_temp_info,
        )
        _write_all(commit_descriptor, commit_payload)
        os.fsync(commit_descriptor)
        if os.pread(commit_descriptor, len(commit_payload) + 1, 0) != commit_payload:
            raise ContractError("acceptance commit temp descriptor readback differs")
        os.fchmod(commit_descriptor, 0o444)
        os.fsync(commit_descriptor)
        sealed_commit_info = os.fstat(commit_descriptor)
        commit_marker_info = sealed_commit_info
        commit_owned_payload = commit_payload
        parse_marker_artifact(
            final_bundle,
            final_info,
            commit_payload,
            sealed_commit_info,
        )
        failure_injector("before_commit_marker_rename")
        transaction_rename_noreplace(
            directory_fd,
            commit_temp_name,
            directory_fd,
            commit_marker_name,
        )
        failure_injector("before_commit_marker_parent_fsync")
        os.fsync(directory_fd)
        failure_injector("after_commit_marker_parent_fsync")
        final_commit_payload, final_commit_info = _read_held_directory_entry_descriptor(
            directory_fd,
            commit_marker_name,
            commit_descriptor,
            "post-publication commit marker",
            expected_info=sealed_commit_info,
            expected_payload=commit_payload,
        )
        final_commit_marker = parse_marker_artifact(
            final_bundle,
            final_info,
            final_commit_payload,
            final_commit_info,
        )
        failure_injector("after_commit_marker_readback")
        marker_report_payload, marker_report_info = (
            _read_held_directory_entry_descriptor(
                directory_fd,
                accepted_name,
                report_descriptor,
                "final marker replay report",
                expected_info=final_info,
                expected_payload=final_payload,
            )
        )
        marker_report = parse_report_artifact(marker_report_payload, marker_report_info)
        marker_replay_payload, marker_replay_info = (
            _read_held_directory_entry_descriptor(
                directory_fd,
                commit_marker_name,
                commit_descriptor,
                "final marker replay marker",
                expected_info=final_commit_info,
                expected_payload=final_commit_payload,
            )
        )
        marker_replay = parse_marker_artifact(
            marker_report,
            marker_report_info,
            marker_replay_payload,
            marker_replay_info,
        )
        marker_reopen = (
            marker_report_payload,
            marker_report_info,
            marker_replay_payload,
            marker_replay_info,
            marker_report,
            marker_replay,
        )
        failure_injector("after_final_canonical_path_validation")
        _fsync_held_canonical_publication(
            directory_fd,
            accepted_name,
            commit_marker_name,
            report_descriptor,
            commit_descriptor,
            final_payload,
            final_commit_payload,
            final_info,
            final_commit_info,
        )
        failure_injector("after_canonical_report_marker_parent_fsync")
        validate_output_directory_fd(
            directory_fd, context["output_directory"], require_path_identity=True
        )
        receipt_slot_payload, receipt_slot_info = _read_held_directory_entry_descriptor(
            directory_fd,
            durable_receipt_name,
            durable_receipt_descriptor,
            "durable acceptance receipt slot",
            expected_info=durable_receipt_info,
            expected_payload=b"",
        )
        if receipt_slot_payload:
            raise ContractError("durable acceptance receipt slot changed before use")
        durable_receipt, durable_receipt_payload = build_receipt_artifact(
            report_descriptor,
            commit_descriptor,
            final_bundle,
            final_payload,
            final_info,
            final_commit_marker,
            final_commit_payload,
            final_commit_info,
            receipt_slot_info,
        )
        receipt_owned_payload = durable_receipt_payload
        before_receipt_write(durable_receipt_descriptor, durable_receipt_payload)
        final_receipt_info = _write_complete_o_sync_receipt(
            durable_receipt_descriptor,
            durable_receipt_payload,
            after_sync_write=lambda: failure_injector(
                "after_durable_acceptance_receipt_sync_write"
            ),
        )
        durable_receipt_info = final_receipt_info
        parse_receipt_artifact(
            final_bundle,
            final_info,
            final_commit_marker,
            final_commit_info,
            durable_receipt_payload,
            final_receipt_info,
        )
        failure_injector("after_durable_acceptance_receipt_fsync")

        def replay_held_artifacts() -> tuple[
            bytes,
            os.stat_result,
            bytes,
            os.stat_result,
            bytes,
            os.stat_result,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]:
            return _read_and_validate_held_publication_triplet(
                directory_fd,
                accepted_name,
                commit_marker_name,
                durable_receipt_name,
                report_descriptor,
                commit_descriptor,
                durable_receipt_descriptor,
                parse_report_artifact,
                parse_marker_artifact,
                parse_receipt_artifact,
                expected_report_payload=payload,
                expected_report_info=final_info,
                expected_marker_payload=final_commit_payload,
                expected_marker_info=final_commit_info,
                expected_receipt_payload=durable_receipt_payload,
                expected_receipt_info=final_receipt_info,
            )

        first_reopen = replay_held_artifacts()
        first_inventory = {
            "report_sha256": sha256_bytes(first_reopen[0]),
            "report_inode": _published_file_record(first_reopen[1]),
            "marker_sha256": sha256_bytes(first_reopen[2]),
            "marker_inode": _published_file_record(first_reopen[3]),
            "receipt_sha256": sha256_bytes(first_reopen[4]),
            "receipt_inode": _published_file_record(first_reopen[5]),
        }
        failure_injector("after_final_acceptance_replay")
        second_reopen = replay_held_artifacts()
        second_inventory = {
            "report_sha256": sha256_bytes(second_reopen[0]),
            "report_inode": _published_file_record(second_reopen[1]),
            "marker_sha256": sha256_bytes(second_reopen[2]),
            "marker_inode": _published_file_record(second_reopen[3]),
            "receipt_sha256": sha256_bytes(second_reopen[4]),
            "receipt_inode": _published_file_record(second_reopen[5]),
        }
        expected_inventory = {
            "report_sha256": sha256_bytes(payload),
            "report_inode": _published_file_record(final_info),
            "marker_sha256": sha256_bytes(final_commit_payload),
            "marker_inode": _published_file_record(final_commit_info),
            "receipt_sha256": sha256_bytes(durable_receipt_payload),
            "receipt_inode": _published_file_record(final_receipt_info),
        }
        marker_inventory = {
            key: expected_inventory[key]
            for key in (
                "report_sha256",
                "report_inode",
                "marker_sha256",
                "marker_inode",
            )
        }
        observed_marker_inventory = {
            "report_sha256": sha256_bytes(marker_reopen[0]),
            "report_inode": _published_file_record(marker_reopen[1]),
            "marker_sha256": sha256_bytes(marker_reopen[2]),
            "marker_inode": _published_file_record(marker_reopen[3]),
        }
        if (
            not _strict_equal(observed_marker_inventory, marker_inventory)
            or not _strict_equal(first_inventory, expected_inventory)
            or not _strict_equal(second_inventory, expected_inventory)
            or first_reopen[0] != payload
            or first_reopen[2] != final_commit_payload
            or first_reopen[4] != durable_receipt_payload
            or second_reopen[0] != payload
            or second_reopen[2] != final_commit_payload
            or second_reopen[4] != durable_receipt_payload
        ):
            raise ContractError(
                "canonical publication changed during final path replay"
            )
        validate_output_directory_fd(
            directory_fd, context["output_directory"], require_path_identity=True
        )
        return second_reopen[0], second_reopen[2], second_reopen[4], second_reopen[8]
    except BaseException as error:
        rollback_refused = False
        for name, descriptor, expected_info, expected_payload, label in (
            (
                durable_receipt_name,
                durable_receipt_descriptor,
                durable_receipt_info,
                receipt_owned_payload,
                "acceptance receipt rollback",
            ),
            (
                commit_marker_name,
                commit_descriptor,
                commit_marker_info,
                commit_owned_payload,
                "acceptance marker rollback",
            ),
            (
                accepted_name,
                report_descriptor,
                published_info,
                report_owned_payload,
                "acceptance report rollback",
            ),
        ):
            rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
                directory_fd,
                accepted_name,
                nonce,
                name,
                descriptor,
                expected_info,
                expected_payload,
                label,
            )
            if not rollback_ok:
                rollback_refused = True
        if commit_temp_name is not None:
            rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
                directory_fd,
                accepted_name,
                nonce,
                commit_temp_name,
                commit_descriptor,
                commit_marker_info,
                commit_owned_payload,
                "acceptance marker temp rollback",
            )
            if not rollback_ok:
                rollback_refused = True
        rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
            directory_fd,
            accepted_name,
            nonce,
            temp_name,
            report_descriptor,
            published_info,
            report_owned_payload,
            "acceptance report temp rollback",
        )
        if not rollback_ok:
            rollback_refused = True
        os.fsync(directory_fd)
        if rollback_refused:
            raise ContractError(
                "acceptance rollback quarantine refused an inode-substituted or "
                "byte-mutated publication entry"
            ) from error
        raise
    finally:
        os.close(report_descriptor)
        if commit_descriptor is not None:
            os.close(commit_descriptor)
        if durable_receipt_descriptor is not None:
            os.close(durable_receipt_descriptor)


def _tokenize_prompt(tokenizer: Any, prompt_bytes: bytes) -> tuple[int, ...]:
    try:
        text = prompt_bytes.decode(PROMPT_ENCODING)
    except UnicodeDecodeError as error:
        raise ContractError("prompt escaped the frozen ASCII encoding") from error
    token_ids = tuple(tokenizer.encode(text).ids)
    if not token_ids:
        raise ContractError("prompt tokenization is empty")
    return token_ids


def _load_model(
    checkpoint: dict[str, Any], model_source_bytes: bytes, source_label: str
) -> Any:
    module_name = "_r12_sealed_dws_model"
    model_module = types.ModuleType(module_name)
    model_module.__file__ = source_label
    sys.modules[module_name] = model_module
    try:
        exec(
            compile(model_source_bytes, source_label, "exec"),
            model_module.__dict__,
        )
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    GPT = model_module.GPT
    GPTConfig = model_module.GPTConfig
    if checkpoint.get("step") != EXPECTED_CHECKPOINT_STEP:
        raise ContractError("checkpoint step identity is not exact string sft_ep1")
    if not isinstance(checkpoint.get("cfg"), dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ContractError("checkpoint lacks cfg or model state")
    model = GPT(GPTConfig(**checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    if int(model.cfg.n_loop) != 1:
        raise ContractError("field screen requires frozen DRS n_loop=1")
    return model.eval()


def _sdpa_backend_flags() -> dict[str, bool]:
    names = {
        "sdpa_math_enabled": "math_sdp_enabled",
        "sdpa_flash_enabled": "flash_sdp_enabled",
        "sdpa_mem_efficient_enabled": "mem_efficient_sdp_enabled",
        "sdpa_cudnn_enabled": "cudnn_sdp_enabled",
    }
    values = {}
    for output_name, attribute_name in names.items():
        getter = getattr(torch.backends.cuda, attribute_name, None)
        if getter is None or not callable(getter):
            raise ContractError(
                f"required CUDA SDPA flag is unavailable: {attribute_name}"
            )
        values[output_name] = bool(getter())
    return values


@contextmanager
def pinned_math_sdpa() -> Any:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as error:
        raise ContractError("PyTorch math SDPA pinning API is unavailable") from error
    try:
        context = sdpa_kernel(backends=[SDPBackend.MATH])
    except (RuntimeError, TypeError, ValueError) as error:
        raise ContractError("PyTorch math SDPA backend cannot be selected") from error
    with context:
        flags = _sdpa_backend_flags()
        expected = {
            "sdpa_math_enabled": True,
            "sdpa_flash_enabled": False,
            "sdpa_mem_efficient_enabled": False,
            "sdpa_cudnn_enabled": False,
        }
        if flags != expected:
            raise ContractError(
                f"math SDPA backend was not exclusively pinned: {flags}"
            )
        yield flags


def _validate_h100_pcie_device_identity(
    device_name: str,
    properties_name: str | None,
    device_capability: tuple[int, int] | list[int],
    device_total_memory_bytes: Any,
    device_uuid: str,
) -> None:
    if (
        device_name != REQUIRED_CUDA_DEVICE_NAME
        or properties_name != REQUIRED_CUDA_DEVICE_NAME
        or tuple(device_capability) != REQUIRED_CUDA_DEVICE_CAPABILITY
        or not device_uuid
        or not isinstance(device_total_memory_bytes, int)
        or isinstance(device_total_memory_bytes, bool)
        or not (
            REQUIRED_CUDA_MEMORY_MIN_BYTES
            <= device_total_memory_bytes
            <= REQUIRED_CUDA_MEMORY_MAX_BYTES
        )
    ):
        raise ContractError(
            "frozen development screen requires one full NVIDIA H100 PCIe device"
        )


def _device_preflight(
    expected_gpu_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("frozen development screen requires CUDA BF16 support")
    if torch.cuda.device_count() != 1:
        raise ContractError("exactly one CUDA device must be visible")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise ContractError(
            f"CUBLAS_WORKSPACE_CONFIG must equal {CUBLAS_WORKSPACE_CONFIG}"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ContractError("PYTHONDONTWRITEBYTECODE must equal 1")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        raise ContractError(
            "CUDA_VISIBLE_DEVICES must identify the Slurm-allocated GPU"
        )
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)
    device_capability = tuple(torch.cuda.get_device_capability(0))
    device_uuid = str(getattr(properties, "uuid", ""))
    device_total_memory_bytes = getattr(properties, "total_memory", None)
    _validate_h100_pcie_device_identity(
        device_name,
        getattr(properties, "name", None),
        device_capability,
        device_total_memory_bytes,
        device_uuid,
    )
    if expected_gpu_binding is not None and (
        device_uuid != expected_gpu_binding["gpu_uuid"]
        or device_name != expected_gpu_binding["gpu_name"]
        or cuda_visible_devices != expected_gpu_binding["cuda_visible_devices"]
        or expected_gpu_binding["mig_mode"] != "Disabled"
        or expected_gpu_binding["mig_devices_present"] is not False
    ):
        raise ContractError("visible CUDA device differs from Slurm/cgroup GPU binding")
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    with pinned_math_sdpa() as sdpa_flags:
        probe = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(
            1, 2, 4, 16
        )
        query = (probe / 127).to(torch.bfloat16)
        key = ((probe + 3) / 131).to(torch.bfloat16)
        value = ((probe + 7) / 137).to(torch.bfloat16)
        first = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        second = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        torch.cuda.synchronize()
        probe_equal = bool(torch.equal(first, second))
    if not probe_equal:
        raise ContractError("pinned BF16 math SDPA probe was not bitwise deterministic")
    deterministic_warn_only = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    if deterministic_warn_only is None:
        raise ContractError("deterministic-algorithms warn-only query is unavailable")
    return {
        "device": DEVICE,
        "precision": PRECISION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": cuda_visible_devices,
        "device_name": device_name,
        "device_capability": list(device_capability),
        "device_total_memory_bytes": device_total_memory_bytes,
        "device_uuid": device_uuid,
        "visible_cuda_device_count": torch.cuda.device_count(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(deterministic_warn_only()),
        "cuda_matmul_tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32_allowed": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa_backend": SDPA_BACKEND,
        **sdpa_flags,
        "sdpa_bf16_probe_bitwise_equal": probe_equal,
        "seed": 0,
    }


def sha256_regular_file(path: Path, *, require_single_link: bool = False) -> str:
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (
        require_single_link and before.st_nlink != 1
    ):
        raise ContractError(f"input must be a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    source = os.fdopen(descriptor, "rb")
    identity = _file_identity(before)
    with source:
        if _file_identity(os.fstat(source.fileno())) != identity:
            raise ContractError(f"input identity changed while opening: {path}")
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        _verify_open_identity(source, identity, path)
    if _file_identity(path.lstat()) != identity:
        raise ContractError(f"input pathname changed while hashing: {path}")
    return digest.hexdigest()


def _loaded_runtime_paths() -> tuple[
    dict[Path, tuple[int, int, int] | None], str | None
]:
    executable_path = (
        Path("/proc/self/exe").resolve()
        if sys.platform.startswith("linux")
        else Path(sys.executable).resolve()
    )
    paths: dict[Path, tuple[int, int, int] | None] = {executable_path: None}
    source_hash: str | None = None
    if sys.platform.startswith("linux"):
        maps_path = Path("/proc/self/maps")
        maps_bytes = maps_path.read_bytes()
        source_hash = sha256_bytes(maps_bytes)
        bindings = _proc_self_maps_bindings(maps_bytes, "loaded runtime mapping")
        for path, binding in bindings.items():
            existing = paths.setdefault(path, binding)
            if existing is not None and existing != binding:
                raise ContractError("loaded runtime pathname maps more than one inode")
            paths[path] = binding
    elif sys.platform == "darwin":
        process = ctypes.CDLL(None)
        image_count = process._dyld_image_count
        image_count.restype = ctypes.c_uint32
        image_name = process._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        names = []
        for index in range(image_count()):
            value = image_name(index)
            if value:
                names.append(value.decode("utf-8"))
        source_hash = sha256_bytes(stable_json_bytes(sorted(names)))
        for name in names:
            path = Path(name)
            if path.is_absolute() and path.exists():
                paths.setdefault(path.resolve(strict=True), None)
    else:
        raise ContractError(
            "runtime object observation is unsupported on this platform"
        )
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if module_path:
            path = Path(module_path)
            if str(path).startswith("/proc/self/fd/"):
                continue
            if path.is_absolute() and path.exists():
                paths.setdefault(path.resolve(strict=True), None)
    return dict(sorted(paths.items(), key=lambda item: str(item[0]))), source_hash


def _cuda_driver_version() -> int:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise ContractError("libcuda.so.1 cannot be loaded") from error
    function = driver.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    value = ctypes.c_int()
    result = function(ctypes.byref(value))
    if result != 0 or value.value <= 0:
        raise ContractError("CUDA driver version query failed")
    return value.value


def observe_executed_runtime(phase: str, *, require_cuda: bool) -> dict[str, Any]:
    if not isinstance(phase, str) or not phase:
        raise ContractError("runtime observation phase is empty")
    if os.environ.get("LD_PRELOAD") or os.environ.get("DYLD_INSERT_LIBRARIES"):
        raise ContractError("preloaded shared libraries are forbidden")
    cuda_driver_version = _cuda_driver_version() if require_cuda else None
    loaded_paths, mapping_source_sha256 = _loaded_runtime_paths()
    records = []
    for path, mapped_identity in loaded_paths.items():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            continue
        if mapped_identity is not None:
            mapped_record = _mapped_regular_file_closure_record(
                path, mapped_identity, label="loaded runtime mapping"
            )
            mapping_identity = {
                "device_major": mapped_record["mapped_device_major"],
                "device_minor": mapped_record["mapped_device_minor"],
                "inode": mapped_record["mapped_inode"],
            }
        else:
            mapping_identity = None
        records.append(
            {
                "path": str(path),
                "sha256": sha256_regular_file(path),
                "device": info.st_dev,
                "inode": info.st_ino,
                "size": info.st_size,
                "mapping_identity": mapping_identity,
            }
        )
    shared = [
        record
        for record in records
        if ".so" in Path(record["path"]).name or Path(record["path"]).suffix == ".dylib"
    ]
    libc_records = [
        record
        for record in shared
        if re.search(r"(^|/)libc(?:[-.]|\.so)", record["path"])
    ]
    loader_records = [
        record
        for record in shared
        if re.search(r"ld-linux|ld-musl|/dyld$", record["path"])
    ]
    cuda_records = [
        record
        for record in shared
        if re.search(
            r"cuda|cudnn|cublas|cufft|curand|cusolver|cusparse|nccl|nvidia",
            record["path"],
            re.IGNORECASE,
        )
    ]
    if require_cuda and (
        not shared or not libc_records or not loader_records or not cuda_records
    ):
        raise ContractError(
            "runtime snapshot lacks shared-object, libc, loader, or CUDA evidence"
        )
    driver_path = Path("/proc/driver/nvidia/version")
    driver_file = None
    if require_cuda:
        if not driver_path.is_file():
            raise ContractError("NVIDIA driver version file is unavailable")
        driver_bytes = driver_path.read_bytes()
        driver_file = {
            "path": str(driver_path),
            "sha256": sha256_bytes(driver_bytes),
            "text_sha256": sha256_bytes(driver_bytes.strip()),
        }
    cuda_visible_devices = None
    device_name = None
    device_capability = None
    device_total_memory_bytes = None
    device_uuid = None
    if require_cuda:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cuda_visible_devices:
            raise ContractError(
                "CUDA_VISIBLE_DEVICES is absent from runtime observation"
            )
        properties = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        device_capability = list(torch.cuda.get_device_capability(0))
        device_total_memory_bytes = getattr(properties, "total_memory", None)
        raw_uuid = getattr(properties, "uuid", None)
        device_uuid = str(raw_uuid) if raw_uuid is not None else None
        _validate_h100_pcie_device_identity(
            device_name,
            getattr(properties, "name", None),
            device_capability,
            device_total_memory_bytes,
            device_uuid,
        )
    return {
        "schema": RUNTIME_OBSERVATION_SCHEMA,
        "phase": phase,
        "coverage": (
            "loaded_file_snapshot_not_complete_immutable_executed_runtime_seal"
        ),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libc_version": list(platform.libc_ver()),
        "ld_preload": None,
        "dyld_insert_libraries": None,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH") or None,
        "mapping_source_sha256": mapping_source_sha256,
        "loaded_files": records,
        "loaded_files_sha256": sha256_bytes(stable_json_bytes(records)),
        "shared_objects": shared,
        "shared_objects_sha256": sha256_bytes(stable_json_bytes(shared)),
        "libc_objects": libc_records,
        "loader_objects": loader_records,
        "cuda_objects": cuda_records,
        "cuda_driver_version": cuda_driver_version,
        "cuda_runtime_version": torch.version.cuda if require_cuda else None,
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_name": device_name,
        "cuda_device_capability": device_capability,
        "cuda_device_total_memory_bytes": device_total_memory_bytes,
        "cuda_device_uuid": device_uuid,
        "nvidia_driver_file": driver_file,
    }


def _prepare_cases(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    prepared = []
    for row in select_cases(rows):
        initial_line = row["initial_state"]
        initial_state = parse_dws_line(initial_line)
        if initial_state is None:
            raise ContractError(f"selected case has invalid state: {row['id']}")
        if (
            row["operation"] != initial_state["op"]
            or row["width"] != initial_state["w"]
        ):
            raise ContractError(f"selected case metadata disagrees: {row['id']}")
        prompt_bytes = render_initial_prompt_bytes(initial_line)
        prompt_ids = _tokenize_prompt(tokenizer, prompt_bytes)
        prepared.append(
            {
                "row": row,
                "initial_state": initial_state,
                "prompt_bytes": prompt_bytes,
                "prompt_ids": prompt_ids,
            }
        )
    return prepared


def _run_primary_decodes(
    model: Any, prepared: list[dict[str, Any]]
) -> dict[str, dict[str, RawDecode]]:
    results: dict[str, dict[str, RawDecode]] = {}
    for index, case in enumerate(prepared, 1):
        case_id = case["row"]["id"]
        results[case_id] = {}
        for arm in PRIMARY_ARM_ORDER:
            results[case_id][arm] = decode_cached_greedy(
                DecodeRequest(
                    model=model,
                    prompt_token_ids=case["prompt_ids"],
                    device=DEVICE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    mode=DecodeMode(arm),
                    eos_token_id=EOS_TOKEN_ID,
                )
            )
        print(f"[dws-eos] primary {index}/{len(prepared)} {case_id}", flush=True)
    return results


def _posthoc_primary_and_prepare_fields(
    model: Any,
    prepared: list[dict[str, Any]],
    primary_raw: dict[str, dict[str, RawDecode]],
    tokenizer: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, DecodeRequest]]],
    dict[str, dict[str, DecodeRequest]],
    dict[str, dict[str, dict[str, Any]]],
]:
    case_reports = []
    all_requests: dict[str, dict[str, dict[str, DecodeRequest]]] = {}
    all_source_requests: dict[str, dict[str, DecodeRequest]] = {}
    all_branches: dict[str, dict[str, dict[str, Any]]] = {}
    for case in prepared:
        row = case["row"]
        case_id = row["id"]
        oracle_states = reconstruct_oracle_posthoc(case["initial_state"])
        heldout_binding = validate_heldout_oracle_posthoc(row, oracle_states)
        primary_reports = {
            arm: raw_decode_report_posthoc(
                primary_raw[case_id][arm],
                tokenizer,
                case["initial_state"],
                oracle_states,
            )
            for arm in PRIMARY_ARM_ORDER
        }
        requests, source_requests, branches, field_status = (
            prepare_field_requests_posthoc(
                model,
                DEVICE,
                case["prompt_ids"],
                primary_reports[DecodeMode.ORDINARY_EOS_STOP.value],
                tokenizer,
            )
        )
        if requests and branches is not None:
            all_requests[case_id] = requests
            all_source_requests[case_id] = source_requests
            all_branches[case_id] = branches
        prompt_text = case["prompt_bytes"].decode(PROMPT_ENCODING)
        case_reports.append(
            {
                "case_id": case_id,
                "split": row["split"],
                "operation": row["operation"],
                "width": row["width"],
                "selection_sha256": selection_digest(case_id),
                "initial_state": row["initial_state"],
                "prompt": {
                    "utf8": prompt_text,
                    "byte_count": len(case["prompt_bytes"]),
                    "sha256": sha256_bytes(case["prompt_bytes"]),
                    "token_ids": list(case["prompt_ids"]),
                    "token_count": len(case["prompt_ids"]),
                    "token_ids_sha256": token_ids_sha256(case["prompt_ids"]),
                },
                "oracle": {
                    "states": [canonical_dws_state(state) for state in oracle_states],
                    "trace_length": len(oracle_states),
                    "first_state_carry": oracle_states[0]["c"],
                    "final_tape": {
                        "r": oracle_states[-1]["r"],
                        "c": oracle_states[-1]["c"],
                    },
                    "answer": state_answer_posthoc(oracle_states[-1]),
                    "heldout_binding": heldout_binding,
                },
                "primary_arms": primary_reports,
                "field_screen": field_status,
            }
        )
    return case_reports, all_requests, all_source_requests, all_branches


def _run_field_decodes(
    requests: dict[str, dict[str, dict[str, DecodeRequest]]],
) -> dict[str, dict[str, dict[str, RawDecode]]]:
    results: dict[str, dict[str, dict[str, RawDecode]]] = {}
    case_ids = list(requests)
    for case_index, case_id in enumerate(case_ids, 1):
        results[case_id] = {}
        for clock_arm in FIELD_CLOCK_ARMS:
            results[case_id][clock_arm] = {}
            for branch_name in HISTORY_BRANCHES:
                results[case_id][clock_arm][branch_name] = decode_cached_greedy(
                    requests[case_id][clock_arm][branch_name]
                )
        print(
            f"[dws-eos] field {case_index}/{len(case_ids)} {case_id}",
            flush=True,
        )
    return results


def _run_fresh_reencoding_decodes(
    requests: dict[str, dict[str, DecodeRequest]],
) -> dict[str, dict[str, RawDecode]]:
    results: dict[str, dict[str, RawDecode]] = {}
    case_ids = list(requests)
    for case_index, case_id in enumerate(case_ids, 1):
        results[case_id] = {
            branch_name: decode_cached_greedy(requests[case_id][branch_name])
            for branch_name in FRESH_REENCODING_BRANCHES
        }
        print(
            f"[dws-eos] fresh-reencoding {case_index}/{len(case_ids)} {case_id}",
            flush=True,
        )
    return results


def _finish_field_scores(
    case_reports: list[dict[str, Any]],
    field_raw: dict[str, dict[str, dict[str, RawDecode]]],
    fresh_reencoding_raw: dict[str, dict[str, RawDecode]],
    all_branches: dict[str, dict[str, dict[str, Any]]],
    tokenizer: Any,
) -> None:
    for case in case_reports:
        case_id = case["case_id"]
        status = case["field_screen"]
        if not status["available"]:
            case["field_screen"] = unavailable_field_screen(status["failure"])
            continue
        emitted_state = parse_dws_line(status["emitted_state"])
        if emitted_state is None:
            raise ContractError("available field screen lacks emitted state")
        episode_oracle = [parse_dws_line(line) for line in case["oracle"]["states"]]
        if any(state is None for state in episode_oracle):
            raise ContractError("case oracle failed to round-trip")
        branches = all_branches[case_id]
        by_clock = {}
        ordinary_event = case["primary_arms"][DecodeMode.ORDINARY_EOS_STOP.value][
            "eos_events"
        ][0]
        for clock_arm in FIELD_CLOCK_ARMS:
            branch_reports = {}
            for branch_name in HISTORY_BRANCHES:
                branch_state = branches[branch_name]["state"]
                target = (
                    apply_microstep_posthoc(branch_state)
                    if branch_state is not None
                    else None
                )
                branch_reports[branch_name] = field_branch_report_posthoc(
                    field_raw[case_id][clock_arm][branch_name], tokenizer, target
                )
            boundary_token = boundary_token_for_clock(clock_arm, ordinary_event)
            detail = score_field_clock_posthoc(
                clock_arm,
                boundary_token,
                tokenizer.decode([boundary_token], skip_special_tokens=False),
                branch_reports,
                branches,
                emitted_state,
                episode_oracle,
            )
            by_clock[clock_arm] = compact_field_clock_score(detail)
        source_branch_reports = {}
        for branch_name in FRESH_REENCODING_BRANCHES:
            branch_state = branches[branch_name]["state"]
            if branch_state is None:
                raise ContractError("fresh re-encoding branch lacks state")
            source_branch_reports[branch_name] = field_branch_report_posthoc(
                fresh_reencoding_raw[case_id][branch_name],
                tokenizer,
                apply_microstep_posthoc(branch_state),
            )
        source_detail = score_fresh_reencoding_posthoc(
            source_branch_reports,
            branches,
            emitted_state,
            episode_oracle,
            by_clock[DecodeMode.EOS_TO_LF.value],
        )
        case["field_screen"] = {
            "available": True,
            "failure": None,
            "emitted_state": status["emitted_state"],
            "emitted_history_token_count": status["emitted_history_token_count"],
            "by_clock": by_clock,
            "fresh_latest_reencoding": compact_fresh_reencoding_score(source_detail),
        }


def _decimal_descriptor(value: str, label: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ContractError(f"{label} is not a decimal descriptor")
    descriptor = int(value)
    if descriptor < 3:
        raise ContractError(f"{label} must not alias a standard stream")
    os.fstat(descriptor)
    return descriptor


def _strict_decimal_int(value: str, label: str, *, minimum: int) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ContractError(f"{label} is not canonical decimal")
    observed = int(value)
    if observed < minimum:
        raise ContractError(f"{label} is below its minimum")
    return observed


def _linux_mount_fstype(path: Path) -> str:
    path_text = str(path)
    matches: list[tuple[int, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split(" ")
        try:
            separator = fields.index("-")
        except ValueError as error:
            raise ContractError("Linux mountinfo record lacks a separator") from error
        if separator < 6 or separator + 1 >= len(fields):
            raise ContractError("Linux mountinfo record is malformed")
        mountpoint = fields[4].replace("\\040", " ")
        if path_text == mountpoint or path_text.startswith(
            mountpoint.rstrip("/") + "/"
        ):
            matches.append((len(mountpoint), fields[separator + 1]))
    if not matches:
        raise ContractError("qualification root has no parsed Linux mount")
    return max(matches)[1]


def _linux_receipt_crash_stage(stage: str) -> tuple[int, str]:
    for candidate, exit_code, expected_replay in LINUX_RECEIPT_CRASH_STAGES:
        if stage == candidate:
            return exit_code, expected_replay
    raise ContractError("Linux receipt crash stage differs")


def _create_held_evaluator_image(
    evaluator_path: Path, expected_evaluator_sha256: str
) -> tuple[int, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_evaluator_sha256) is None:
        raise ContractError("Linux receipt evaluator hash is not lowercase SHA-256")
    _record, payload = _regular_file_closure_record(
        evaluator_path,
        relative_path=None,
        label="Linux qualification evaluator source",
    )
    if sha256_bytes(payload) != expected_evaluator_sha256:
        raise ContractError("Linux receipt evaluator bytes differ from reviewed hash")
    if hasattr(os, "memfd_create") and sys.platform.startswith("linux"):
        descriptor = os.memfd_create("r12-linux-held-evaluator", os.MFD_ALLOW_SEALING)
        descriptor_kind = "sealed_memfd"
        seals = _required_memfd_seals()
    else:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix="r12-linux-held-evaluator-"
        )
        os.unlink(temporary_path)
        descriptor_kind = "unlinked_held_copy_development_only"
        seals = 0
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        if descriptor_kind == "sealed_memfd":
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        source = {
            "sha256": expected_evaluator_sha256,
            "byte_count": len(payload),
            "descriptor_kind": descriptor_kind,
            "seals": seals,
        }
        _read_held_evaluator_image(descriptor, source)
        return descriptor, source
    except BaseException:
        os.close(descriptor)
        raise


def _read_held_evaluator_image(
    descriptor: int, expected_source: dict[str, Any]
) -> bytes:
    _require_exact_keys(
        expected_source,
        {"sha256", "byte_count", "descriptor_kind", "seals"},
        "held Linux qualification evaluator source",
    )
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 0
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_size != expected_source["byte_count"]
    ):
        raise ContractError("held Linux qualification evaluator inode differs")
    if expected_source["descriptor_kind"] == "sealed_memfd":
        if (
            not hasattr(fcntl, "F_GET_SEALS")
            or expected_source["seals"] != _required_memfd_seals()
            or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != expected_source["seals"]
        ):
            raise ContractError("held Linux qualification evaluator seals differ")
    elif (
        expected_source["descriptor_kind"] != "unlinked_held_copy_development_only"
        or expected_source["seals"] != 0
        or sys.platform.startswith("linux")
    ):
        raise ContractError("held Linux qualification evaluator kind differs")
    payload = os.pread(descriptor, info.st_size + 1, 0)
    if (
        len(payload) != info.st_size
        or sha256_bytes(payload) != expected_source["sha256"]
        or _file_identity(os.fstat(descriptor)) != _file_identity(info)
    ):
        raise ContractError("held Linux qualification evaluator bytes differ")
    return payload


def _held_descriptor_execution_path(descriptor: int) -> str:
    prefix = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
    return f"{prefix}/{descriptor}"


HELD_EVALUATOR_DESCRIPTOR_BOOTSTRAP = """\
import hashlib
import os
import sys

descriptor = int(sys.argv[1])
expected_sha256 = sys.argv[2]
expected_byte_count = int(sys.argv[3])
source_name = sys.argv[4]
payload = os.pread(descriptor, expected_byte_count + 1, 0)
if (
    len(payload) != expected_byte_count
    or hashlib.sha256(payload).hexdigest() != expected_sha256
):
    raise SystemExit("held evaluator descriptor bytes differ")
sys.argv = [source_name, *sys.argv[5:]]
namespace = {
    "__name__": "__main__",
    "__file__": source_name,
    "__package__": None,
    "__cached__": None,
}
exec(compile(payload, source_name, "exec"), namespace, namespace)
"""


def _linux_qualification_evaluator_source(
    expected_source: dict[str, Any], evaluator_descriptor: int
) -> dict[str, Any]:
    _read_held_evaluator_image(evaluator_descriptor, expected_source)
    return {
        "sha256": expected_source["sha256"],
        "byte_count": expected_source["byte_count"],
        "descriptor_kind": expected_source["descriptor_kind"],
        "seals": expected_source["seals"],
    }


def _validate_linux_qualification_evaluator_source(
    value: Any, expected_source: dict[str, Any]
) -> None:
    _require_exact_keys(
        value,
        {"sha256", "byte_count", "descriptor_kind", "seals"},
        "Linux qualification evaluator source",
    )
    if not _strict_equal(value, expected_source):
        raise ContractError("Linux qualification evaluator source differs")


def _validate_linux_qualification_file(
    info: os.stat_result, payload: bytes, label: str
) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_size != len(payload)
    ):
        raise ContractError(f"{label} stat contract differs")


def _validate_linux_qualification_report_marker_payloads(
    report: dict[str, Any],
    report_payload: bytes,
    marker: dict[str, Any],
    marker_payload: bytes,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
) -> None:
    _linux_receipt_crash_stage(stage)
    _require_exact_keys(
        report, LINUX_QUALIFICATION_REPORT_KEYS, "Linux qualification report"
    )
    _require_exact_keys(
        marker, LINUX_QUALIFICATION_MARKER_KEYS, "Linux qualification marker"
    )
    if stable_json_bytes(report) != report_payload:
        raise ContractError("Linux qualification report is not canonical JSON")
    if stable_json_bytes(marker) != marker_payload:
        raise ContractError("Linux qualification marker is not canonical JSON")
    _validate_linux_qualification_evaluator_source(
        report["evaluator_source"], expected_evaluator_source
    )
    _require_exact_keys(
        report["brokered_signing_key"],
        SIGNING_KEY_RECORD_KEYS,
        "Linux qualification brokered signing key",
    )
    if (
        report["schema"] != LINUX_QUALIFICATION_REPORT_SCHEMA
        or report["stage"] != stage
        or report["broker_request_sha256"] != expected_broker_request_sha256
        or not _strict_equal(report["brokered_signing_key"], expected_signing_key)
        or report["authority_boundary"] != LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
    ):
        raise ContractError("Linux qualification report identity differs")
    if (
        marker["schema"] != LINUX_QUALIFICATION_MARKER_SCHEMA
        or marker["stage"] != stage
        or marker["report_name"] != LINUX_QUALIFICATION_REPORT_NAME
        or marker["report_sha256"] != sha256_bytes(report_payload)
        or marker["evaluator_sha256"] != expected_evaluator_source["sha256"]
        or marker["broker_request_sha256"] != expected_broker_request_sha256
        or marker["brokered_public_key_hex"] != expected_signing_key["public_key_hex"]
        or marker["authority_boundary"] != LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
    ):
        raise ContractError("Linux qualification marker binding differs")
    signature_payload = {
        key: value for key, value in marker.items() if key != "signature_hex"
    }
    if (
        not isinstance(marker["signature_hex"], str)
        or re.fullmatch(r"[0-9a-f]{128}", marker["signature_hex"]) is None
        or not _ed25519_verify(
            bytes.fromhex(expected_signing_key["public_key_hex"]),
            bytes.fromhex(marker["signature_hex"]),
            stable_json_bytes(signature_payload),
        )
    ):
        raise ContractError("Linux qualification marker signature differs")


def _validate_linux_qualification_report_marker(
    report: dict[str, Any],
    report_payload: bytes,
    report_info: os.stat_result,
    marker: dict[str, Any],
    marker_payload: bytes,
    marker_info: os.stat_result,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
) -> None:
    _validate_linux_qualification_report_marker_payloads(
        report,
        report_payload,
        marker,
        marker_payload,
        stage,
        expected_evaluator_source,
        expected_broker_request_sha256,
        expected_signing_key,
    )
    _validate_linux_qualification_file(
        report_info, report_payload, "Linux qualification report"
    )
    _validate_linux_qualification_file(
        marker_info, marker_payload, "Linux qualification marker"
    )


def _read_and_validate_linux_qualification_report_marker(
    directory_fd: int,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
) -> tuple[
    bytes,
    os.stat_result,
    bytes,
    os.stat_result,
    dict[str, Any],
    dict[str, Any],
]:
    descriptors: list[int] = []
    try:
        for name in (
            LINUX_QUALIFICATION_REPORT_NAME,
            LINUX_QUALIFICATION_MARKER_NAME,
        ):
            descriptors.append(
                os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            )
        report_payload, report_info = _read_held_directory_entry_descriptor(
            directory_fd,
            LINUX_QUALIFICATION_REPORT_NAME,
            descriptors[0],
            "Linux qualification report",
        )
        marker_payload, marker_info = _read_held_directory_entry_descriptor(
            directory_fd,
            LINUX_QUALIFICATION_MARKER_NAME,
            descriptors[1],
            "Linux qualification marker",
        )
        report = _parse_json_object_bytes(report_payload, "Linux qualification report")
        marker = _parse_json_object_bytes(marker_payload, "Linux qualification marker")
        _validate_linux_qualification_report_marker(
            report,
            report_payload,
            report_info,
            marker,
            marker_payload,
            marker_info,
            stage,
            expected_evaluator_source,
            expected_broker_request_sha256,
            expected_signing_key,
        )
        _read_held_directory_entry_descriptor(
            directory_fd,
            LINUX_QUALIFICATION_REPORT_NAME,
            descriptors[0],
            "Linux qualification report final hold",
            expected_info=report_info,
            expected_payload=report_payload,
        )
        _read_held_directory_entry_descriptor(
            directory_fd,
            LINUX_QUALIFICATION_MARKER_NAME,
            descriptors[1],
            "Linux qualification marker final hold",
            expected_info=marker_info,
            expected_payload=marker_payload,
        )
        return (
            report_payload,
            report_info,
            marker_payload,
            marker_info,
            report,
            marker,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _build_linux_qualification_receipt(
    report: dict[str, Any],
    report_payload: bytes,
    report_info: os.stat_result,
    marker: dict[str, Any],
    marker_payload: bytes,
    marker_info: os.stat_result,
    receipt_slot_info: os.stat_result,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
    private_key: bytes,
) -> dict[str, Any]:
    _validate_linux_qualification_report_marker(
        report,
        report_payload,
        report_info,
        marker,
        marker_payload,
        marker_info,
        stage,
        expected_evaluator_source,
        expected_broker_request_sha256,
        expected_signing_key,
    )
    if (
        not stat.S_ISREG(receipt_slot_info.st_mode)
        or stat.S_IMODE(receipt_slot_info.st_mode) != 0o444
        or receipt_slot_info.st_nlink != 1
        or receipt_slot_info.st_uid != os.getuid()
        or receipt_slot_info.st_size != 0
    ):
        raise ContractError("Linux qualification receipt slot differs")
    payload = {
        "schema": LINUX_QUALIFICATION_RECEIPT_SCHEMA,
        "status": LINUX_QUALIFICATION_RECEIPT_STATUS,
        "stage": stage,
        "report_name": LINUX_QUALIFICATION_REPORT_NAME,
        "marker_name": LINUX_QUALIFICATION_MARKER_NAME,
        "receipt_name": LINUX_QUALIFICATION_RECEIPT_NAME,
        "evaluator_source": report["evaluator_source"],
        "report_sha256": sha256_bytes(report_payload),
        "marker_sha256": sha256_bytes(marker_payload),
        "report_inode": _published_file_record(report_info),
        "marker_inode": _published_file_record(marker_info),
        "receipt_inode": {
            "device": receipt_slot_info.st_dev,
            "inode": receipt_slot_info.st_ino,
            "uid": receipt_slot_info.st_uid,
        },
        "durability_checks": {
            "report_file_fsync_complete": True,
            "marker_file_fsync_complete": True,
            "receipt_slot_file_fsync_complete": True,
            "receipt_slot_parent_fsync_complete": True,
            "report_held_descriptor_fsync_complete": True,
            "marker_held_descriptor_fsync_complete": True,
            "publication_parent_fsync_complete": True,
            "receipt_o_sync_write_complete": True,
        },
        "broker_request_sha256": expected_broker_request_sha256,
        "brokered_public_key_hex": expected_signing_key["public_key_hex"],
        "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
    }
    return {
        **payload,
        "signature_hex": _ed25519_sign(private_key, stable_json_bytes(payload)).hex(),
    }


def _validate_linux_qualification_receipt_payloads(
    report: dict[str, Any],
    report_payload: bytes,
    marker: dict[str, Any],
    marker_payload: bytes,
    receipt: dict[str, Any],
    receipt_payload: bytes,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
) -> None:
    _validate_linux_qualification_report_marker_payloads(
        report,
        report_payload,
        marker,
        marker_payload,
        stage,
        expected_evaluator_source,
        expected_broker_request_sha256,
        expected_signing_key,
    )
    _require_exact_keys(
        receipt, LINUX_QUALIFICATION_RECEIPT_KEYS, "Linux qualification receipt"
    )
    _require_exact_keys(
        receipt["durability_checks"],
        LINUX_QUALIFICATION_RECEIPT_CHECK_KEYS,
        "Linux qualification receipt durability checks",
    )
    if stable_json_bytes(receipt) != receipt_payload:
        raise ContractError("Linux qualification receipt is not canonical JSON")
    if any(value is not True for value in receipt["durability_checks"].values()):
        raise ContractError("Linux qualification receipt durability checks differ")
    if (
        receipt["schema"] != LINUX_QUALIFICATION_RECEIPT_SCHEMA
        or receipt["status"] != LINUX_QUALIFICATION_RECEIPT_STATUS
        or receipt["stage"] != stage
        or receipt["report_name"] != LINUX_QUALIFICATION_REPORT_NAME
        or receipt["marker_name"] != LINUX_QUALIFICATION_MARKER_NAME
        or receipt["receipt_name"] != LINUX_QUALIFICATION_RECEIPT_NAME
        or not _strict_equal(receipt["evaluator_source"], report["evaluator_source"])
        or receipt["report_sha256"] != sha256_bytes(report_payload)
        or receipt["marker_sha256"] != sha256_bytes(marker_payload)
        or receipt["broker_request_sha256"] != expected_broker_request_sha256
        or receipt["brokered_public_key_hex"] != expected_signing_key["public_key_hex"]
        or receipt["authority_boundary"] != LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
    ):
        raise ContractError("Linux qualification receipt binding differs")
    _require_exact_keys(
        receipt["report_inode"], FINAL_INODE_KEYS, "Linux qualification report inode"
    )
    _require_exact_keys(
        receipt["marker_inode"], FINAL_INODE_KEYS, "Linux qualification marker inode"
    )
    _require_exact_keys(
        receipt["receipt_inode"],
        DURABLE_ACCEPTANCE_RECEIPT_INODE_KEYS,
        "Linux qualification receipt inode",
    )
    for label, record, payload in (
        ("report", receipt["report_inode"], report_payload),
        ("marker", receipt["marker_inode"], marker_payload),
    ):
        if (
            any(type(record[key]) is not int or record[key] < 0 for key in record)
            or record["mode"] != 0o444
            or record["nlink"] != 1
            or record["size"] != len(payload)
        ):
            raise ContractError(
                f"Linux qualification {label} replay inode binding differs"
            )
    if any(
        type(receipt["receipt_inode"][key]) is not int
        or receipt["receipt_inode"][key] < 0
        for key in receipt["receipt_inode"]
    ):
        raise ContractError("Linux qualification receipt replay inode differs")
    signature_payload = {
        key: value for key, value in receipt.items() if key != "signature_hex"
    }
    if (
        not isinstance(receipt["signature_hex"], str)
        or re.fullmatch(r"[0-9a-f]{128}", receipt["signature_hex"]) is None
        or not _ed25519_verify(
            bytes.fromhex(expected_signing_key["public_key_hex"]),
            bytes.fromhex(receipt["signature_hex"]),
            stable_json_bytes(signature_payload),
        )
    ):
        raise ContractError("Linux qualification receipt signature differs")


def _validate_linux_qualification_receipt(
    report: dict[str, Any],
    report_payload: bytes,
    report_info: os.stat_result,
    marker: dict[str, Any],
    marker_payload: bytes,
    marker_info: os.stat_result,
    receipt: dict[str, Any],
    receipt_payload: bytes,
    receipt_info: os.stat_result,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    expected_broker_request_sha256: str,
    expected_signing_key: dict[str, Any],
) -> None:
    _validate_linux_qualification_receipt_payloads(
        report,
        report_payload,
        marker,
        marker_payload,
        receipt,
        receipt_payload,
        stage,
        expected_evaluator_source,
        expected_broker_request_sha256,
        expected_signing_key,
    )
    expected_receipt_inode = {
        "device": receipt_info.st_dev,
        "inode": receipt_info.st_ino,
        "uid": receipt_info.st_uid,
    }
    if (
        not _strict_equal(receipt["report_inode"], _published_file_record(report_info))
        or not _strict_equal(
            receipt["marker_inode"], _published_file_record(marker_info)
        )
        or not _strict_equal(receipt["receipt_inode"], expected_receipt_inode)
    ):
        raise ContractError("Linux qualification receipt inode binding differs")
    _validate_linux_qualification_file(
        report_info, report_payload, "Linux qualification report"
    )
    _validate_linux_qualification_file(
        marker_info, marker_payload, "Linux qualification marker"
    )
    _validate_linux_qualification_file(
        receipt_info, receipt_payload, "Linux qualification receipt"
    )


def _linux_qualification_publication_contract(
    output_directory: dict[str, Any],
    stage: str,
    expected_evaluator_source: dict[str, Any],
    broker_request_sha256: str,
    private_key: bytes,
    *,
    before_receipt_write: Any = None,
) -> dict[str, Any]:
    _linux_receipt_crash_stage(stage)
    expected_signing_key = signing_key_record(private_key)
    if re.fullmatch(r"[0-9a-f]{64}", broker_request_sha256) is None:
        raise ContractError("Linux qualification broker request hash differs")
    report = {
        "schema": LINUX_QUALIFICATION_REPORT_SCHEMA,
        "stage": stage,
        "evaluator_source": expected_evaluator_source,
        "broker_request_sha256": broker_request_sha256,
        "brokered_signing_key": expected_signing_key,
        "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
    }
    report_payload = stable_json_bytes(report)

    def qualification_rename_noreplace(
        old_directory_fd: int,
        old_name: str,
        new_directory_fd: int,
        new_name: str,
    ) -> None:
        if sys.platform.startswith("linux"):
            rename_noreplace_at(
                old_directory_fd,
                old_name,
                new_directory_fd,
                new_name,
            )
            return
        if _entry_stat_or_none(new_directory_fd, new_name) is not None:
            raise FileExistsError(new_name)
        os.rename(
            old_name,
            new_name,
            src_dir_fd=old_directory_fd,
            dst_dir_fd=new_directory_fd,
        )

    def parse_report(payload: bytes, info: os.stat_result) -> dict[str, Any]:
        observed = _parse_json_object_bytes(payload, "Linux qualification report")
        _require_exact_keys(
            observed, LINUX_QUALIFICATION_REPORT_KEYS, "Linux qualification report"
        )
        if stable_json_bytes(observed) != payload or not _strict_equal(
            observed, report
        ):
            raise ContractError("Linux qualification report identity differs")
        _validate_linux_qualification_file(info, payload, "Linux qualification report")
        return observed

    def build_marker(
        _report_descriptor: int,
        observed_report: dict[str, Any],
        observed_report_payload: bytes,
        report_info: os.stat_result,
        _marker_slot_info: os.stat_result,
    ) -> tuple[dict[str, Any], bytes]:
        parse_report(observed_report_payload, report_info)
        marker_payload = {
            "schema": LINUX_QUALIFICATION_MARKER_SCHEMA,
            "stage": stage,
            "report_name": LINUX_QUALIFICATION_REPORT_NAME,
            "report_sha256": sha256_bytes(observed_report_payload),
            "evaluator_sha256": expected_evaluator_source["sha256"],
            "broker_request_sha256": broker_request_sha256,
            "brokered_public_key_hex": expected_signing_key["public_key_hex"],
            "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
        }
        marker = {
            **marker_payload,
            "signature_hex": _ed25519_sign(
                private_key, stable_json_bytes(marker_payload)
            ).hex(),
        }
        return marker, stable_json_bytes(marker)

    def parse_marker(
        observed_report: dict[str, Any],
        report_info: os.stat_result,
        payload: bytes,
        info: os.stat_result,
    ) -> dict[str, Any]:
        marker = _parse_json_object_bytes(payload, "Linux qualification marker")
        _validate_linux_qualification_report_marker(
            observed_report,
            stable_json_bytes(observed_report),
            report_info,
            marker,
            payload,
            info,
            stage,
            expected_evaluator_source,
            broker_request_sha256,
            expected_signing_key,
        )
        return marker

    def build_receipt(
        _report_descriptor: int,
        _marker_descriptor: int,
        observed_report: dict[str, Any],
        observed_report_payload: bytes,
        report_info: os.stat_result,
        marker: dict[str, Any],
        marker_payload: bytes,
        marker_info: os.stat_result,
        receipt_slot_info: os.stat_result,
    ) -> tuple[dict[str, Any], bytes]:
        receipt = _build_linux_qualification_receipt(
            observed_report,
            observed_report_payload,
            report_info,
            marker,
            marker_payload,
            marker_info,
            receipt_slot_info,
            stage,
            expected_evaluator_source,
            broker_request_sha256,
            expected_signing_key,
            private_key,
        )
        return receipt, stable_json_bytes(receipt)

    def parse_receipt(
        observed_report: dict[str, Any],
        report_info: os.stat_result,
        marker: dict[str, Any],
        marker_info: os.stat_result,
        payload: bytes,
        info: os.stat_result,
    ) -> dict[str, Any]:
        receipt = _parse_json_object_bytes(payload, "Linux qualification receipt")
        _validate_linux_qualification_receipt(
            observed_report,
            stable_json_bytes(observed_report),
            report_info,
            marker,
            stable_json_bytes(marker),
            marker_info,
            receipt,
            payload,
            info,
            stage,
            expected_evaluator_source,
            broker_request_sha256,
            expected_signing_key,
        )
        return receipt

    return {
        "accepted_name": LINUX_QUALIFICATION_REPORT_NAME,
        "commit_marker_name": LINUX_QUALIFICATION_MARKER_NAME,
        "durable_receipt_name": LINUX_QUALIFICATION_RECEIPT_NAME,
        "nonce": broker_request_sha256,
        "output_directory": output_directory,
        "initial_payload": report_payload,
        "parse_report_artifact": parse_report,
        "build_marker_artifact": build_marker,
        "parse_marker_artifact": parse_marker,
        "build_receipt_artifact": build_receipt,
        "parse_receipt_artifact": parse_receipt,
        "before_receipt_write": before_receipt_write
        if before_receipt_write is not None
        else (lambda _descriptor, _payload: None),
        "rename_noreplace": qualification_rename_noreplace,
    }


def _read_and_validate_linux_qualification_receipt(
    directory_fd: int,
    output_directory: dict[str, Any],
    stage: str,
    expected_evaluator_source: dict[str, Any],
    broker_process_id: int,
) -> dict[str, Any]:
    private_key = _linux_qualification_private_key(
        stage, expected_evaluator_source["sha256"]
    )
    expected_signing_key = signing_key_record(private_key)
    request = _linux_qualification_broker_request(
        stage,
        expected_evaluator_source["sha256"],
        expected_signing_key,
        process_id=broker_process_id,
    )
    request_sha256 = sha256_bytes(stable_json_bytes(request))
    contract = _linux_qualification_publication_contract(
        output_directory,
        stage,
        expected_evaluator_source,
        request_sha256,
        private_key,
    )
    descriptors: list[int] = []
    try:
        for name in (
            LINUX_QUALIFICATION_REPORT_NAME,
            LINUX_QUALIFICATION_MARKER_NAME,
            LINUX_QUALIFICATION_RECEIPT_NAME,
        ):
            descriptors.append(
                os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            )
        publication = _read_and_validate_held_publication_triplet(
            directory_fd,
            LINUX_QUALIFICATION_REPORT_NAME,
            LINUX_QUALIFICATION_MARKER_NAME,
            LINUX_QUALIFICATION_RECEIPT_NAME,
            descriptors[0],
            descriptors[1],
            descriptors[2],
            contract["parse_report_artifact"],
            contract["parse_marker_artifact"],
            contract["parse_receipt_artifact"],
        )
    except (FileNotFoundError, ContractError, OSError) as error:
        raise ContractError(
            "Linux qualification publication has no valid complete O_SYNC receipt"
        ) from error
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return publication[8]


def _run_linux_receipt_crash_worker(
    root: Path,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    evaluator_descriptor: int,
) -> None:
    exit_code, _expected_replay = _linux_receipt_crash_stage(stage)
    evaluator_source = _linux_qualification_evaluator_source(
        expected_evaluator_source, evaluator_descriptor
    )
    root = _require_normalized_absolute_path(root, "Linux receipt crash worker root")
    directory_fd, directory_record = open_owned_output_directory(root)
    brokered_key_descriptor: int | None = None
    lease: dict[str, Any] | None = None
    try:
        if os.listdir(directory_fd):
            raise ContractError("Linux receipt crash worker directory is not empty")
        (
            brokered_key_descriptor,
            private_key,
            broker_request,
            _broker_evidence,
        ) = _linux_qualification_broker_exchange(
            stage, expected_evaluator_source["sha256"]
        )
        broker_request_sha256 = sha256_bytes(stable_json_bytes(broker_request))
        _read_linux_qualification_broker_key(
            brokered_key_descriptor, signing_key_record(private_key)
        )
        lease = acquire_publisher_lease(
            directory_fd,
            directory_record,
            LINUX_QUALIFICATION_REPORT_NAME,
            "1",
            broker_request_sha256,
        )

        def before_receipt_write(descriptor: int, payload: bytes) -> None:
            if stage != "during_partial_receipt_write":
                return
            partial_payload = payload[: len(payload) // 2]
            if not partial_payload or len(partial_payload) >= len(payload):
                raise ContractError("Linux qualification partial receipt size differs")
            _write_all(descriptor, partial_payload)
            if os.fstat(descriptor).st_size != len(partial_payload):
                raise ContractError("Linux qualification partial receipt write differs")
            os._exit(exit_code)

        contract = _linux_qualification_publication_contract(
            directory_record,
            stage,
            evaluator_source,
            broker_request_sha256,
            private_key,
            before_receipt_write=before_receipt_write,
        )

        def inject_failure(publication_stage: str) -> None:
            if (
                stage == "before_publication_parent_fsync"
                and publication_stage == "before_commit_marker_parent_fsync"
            ) or (
                stage == "after_complete_receipt_write"
                and publication_stage == "after_durable_acceptance_receipt_sync_write"
            ):
                os._exit(exit_code)

        publish_accepted_bundle_exclusive(
            directory_fd,
            {},
            None,
            None,
            None,
            private_key,
            failure_injector=inject_failure,
            qualification_contract=contract,
        )
        raise ContractError("Linux qualification crash injection did not occur")
    finally:
        if lease is not None:
            release_publisher_lease(directory_fd, lease)
        if brokered_key_descriptor is not None:
            os.close(brokered_key_descriptor)
        os.close(directory_fd)


def _run_linux_receipt_crash_subprocess(
    root: Path,
    stage: str,
    expected_evaluator_source: dict[str, Any],
    evaluator_descriptor: int,
    *,
    before_spawn: Any = None,
) -> tuple[int, int]:
    expected_exit_code, _expected_replay = _linux_receipt_crash_stage(stage)
    python_path = Path(sys.executable).resolve(strict=True)
    _read_held_evaluator_image(evaluator_descriptor, expected_evaluator_source)
    if before_spawn is not None:
        before_spawn()
    _read_held_evaluator_image(evaluator_descriptor, expected_evaluator_source)
    environment = {
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONWARNINGS": "error",
    }
    process = subprocess.Popen(
        [
            str(python_path),
            "-I",
            "-S",
            "-B",
            "-W",
            "error",
            "-c",
            HELD_EVALUATOR_DESCRIPTOR_BOOTSTRAP,
            str(evaluator_descriptor),
            expected_evaluator_source["sha256"],
            str(expected_evaluator_source["byte_count"]),
            _held_descriptor_execution_path(evaluator_descriptor),
            "linux-receipt-crash-worker",
            "--linux-smoke-root",
            str(root),
            "--linux-receipt-crash-stage",
            stage,
            "--expected-evaluator-sha256",
            expected_evaluator_source["sha256"],
            "--expected-evaluator-byte-count",
            str(expected_evaluator_source["byte_count"]),
            "--expected-evaluator-descriptor-kind",
            expected_evaluator_source["descriptor_kind"],
            "--expected-evaluator-seals",
            str(expected_evaluator_source["seals"]),
            "--held-evaluator-fd",
            str(evaluator_descriptor),
            "--development-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        pass_fds=(evaluator_descriptor,),
    )
    child_pid = process.pid
    stdout, stderr_bytes = process.communicate()
    if process.returncode != expected_exit_code or stdout or stderr_bytes:
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise ContractError(
            "Linux receipt crash subprocess differed at "
            f"{stage}: exit={process.returncode}, stderr={stderr!r}"
        )
    return process.returncode, child_pid


def _cleanup_linux_qualification_scenario(
    transaction_fd: int, scenario_name: str, scenario_path: Path
) -> None:
    try:
        scenario_fd, _scenario_record = open_owned_output_directory(scenario_path)
    except FileNotFoundError:
        return
    try:
        for name in os.listdir(scenario_fd):
            info = os.stat(name, dir_fd=scenario_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ContractError("Linux qualification scenario residue differs")
            os.unlink(name, dir_fd=scenario_fd)
        os.fsync(scenario_fd)
    finally:
        os.close(scenario_fd)
    os.rmdir(scenario_name, dir_fd=transaction_fd)
    os.fsync(transaction_fd)


def _run_linux_receipt_crash_consistency_cases(
    transaction_fd: int,
    transaction_path: Path,
    expected_evaluator_sha256: str,
    *,
    evaluator_path: Path | None = None,
    first_child_before_spawn: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transaction_path = _require_normalized_absolute_path(
        transaction_path, "Linux receipt qualification transaction"
    )
    cases: list[dict[str, Any]] = []
    receipt_sizes: dict[str, int] = {}
    accepted_publication: dict[str, Any] | None = None
    evaluator_descriptor, evaluator_source = _create_held_evaluator_image(
        Path(__file__).resolve(strict=True)
        if evaluator_path is None
        else evaluator_path,
        expected_evaluator_sha256,
    )
    try:
        for index, (
            stage,
            expected_exit_code,
            expected_replay,
        ) in enumerate(LINUX_RECEIPT_CRASH_STAGES):
            scenario_name = f"receipt-crash-{stage}"
            scenario_path = transaction_path / scenario_name
            os.mkdir(scenario_name, 0o700, dir_fd=transaction_fd)
            os.fsync(transaction_fd)
            scenario_fd: int | None = None
            try:
                observed_exit_code, broker_process_id = (
                    _run_linux_receipt_crash_subprocess(
                        scenario_path,
                        stage,
                        evaluator_source,
                        evaluator_descriptor,
                        before_spawn=first_child_before_spawn if index == 0 else None,
                    )
                )
                if observed_exit_code != expected_exit_code:
                    raise ContractError("Linux receipt crash exit code differs")
                scenario_fd, scenario_record = open_owned_output_directory(
                    scenario_path
                )
                expected_signing_key = signing_key_record(
                    _linux_qualification_private_key(stage, expected_evaluator_sha256)
                )
                broker_request = _linux_qualification_broker_request(
                    stage,
                    expected_evaluator_sha256,
                    expected_signing_key,
                    process_id=broker_process_id,
                )
                broker_request_sha256 = sha256_bytes(stable_json_bytes(broker_request))
                (
                    _report_payload,
                    _report_info,
                    _marker_payload,
                    _marker_info,
                    report,
                    marker,
                ) = _read_and_validate_linux_qualification_report_marker(
                    scenario_fd,
                    stage,
                    evaluator_source,
                    broker_request_sha256,
                    expected_signing_key,
                )
                receipt_info = os.stat(
                    LINUX_QUALIFICATION_RECEIPT_NAME,
                    dir_fd=scenario_fd,
                    follow_symlinks=False,
                )
                receipt_sizes[stage] = receipt_info.st_size
                try:
                    receipt = _read_and_validate_linux_qualification_receipt(
                        scenario_fd,
                        scenario_record,
                        stage,
                        evaluator_source,
                        broker_process_id,
                    )
                except ContractError as error:
                    if (
                        expected_replay != "rejected"
                        or "no valid complete O_SYNC receipt" not in str(error)
                    ):
                        raise
                    observed_replay = "rejected"
                else:
                    if expected_replay != "accepted":
                        raise ContractError(
                            "Linux qualification accepted an incomplete receipt"
                        )
                    if receipt["status"] != LINUX_QUALIFICATION_RECEIPT_STATUS:
                        raise ContractError(
                            "Linux qualification accepted receipt differs"
                        )
                    observed_replay = "accepted"
                    accepted_publication = {
                        "broker_request": broker_request,
                        "report": report,
                        "marker": marker,
                        "receipt": receipt,
                    }
                cases.append(
                    {
                        "stage": stage,
                        "child_exit_code": observed_exit_code,
                        "expected_replay": expected_replay,
                        "observed_replay": observed_replay,
                        "independent_report_marker_replay": "validated",
                        "receipt_size": receipt_info.st_size,
                        "evaluator_source": evaluator_source,
                        "broker_request_sha256": broker_request_sha256,
                    }
                )
            finally:
                if scenario_fd is not None:
                    os.close(scenario_fd)
                _cleanup_linux_qualification_scenario(
                    transaction_fd, scenario_name, scenario_path
                )
    finally:
        os.close(evaluator_descriptor)
    before_size = receipt_sizes["before_publication_parent_fsync"]
    partial_size = receipt_sizes["during_partial_receipt_write"]
    complete_size = receipt_sizes["after_complete_receipt_write"]
    if before_size != 0 or not (0 < partial_size < complete_size):
        raise ContractError("Linux receipt crash-consistency sizes differ")
    if accepted_publication is None:
        raise ContractError("Linux qualification accepted publication is absent")
    return cases, accepted_publication


def _linux_smoke_publish_with_injected_fsync_failure(
    directory_fd: int, name: str, fail_stage: str
) -> None:
    temp_name = f".{name}.tmp-{secrets.token_hex(16)}"
    descriptor = os.open(
        temp_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    published_info: os.stat_result | None = None
    payload = b""
    try:
        payload = b"linux-publication-qualification\n"
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        published_info = info
        rename_noreplace_at(directory_fd, temp_name, directory_fd, name)
        if fail_stage == "before_directory_fsync":
            raise OSError(errno.EIO, "injected pre-directory-fsync failure")
        os.fsync(directory_fd)
        if fail_stage == "after_directory_fsync":
            raise OSError(errno.EIO, "injected post-directory-fsync failure")
        raise ContractError("qualification fsync failure was not injected")
    except OSError as error:
        if error.errno != errno.EIO:
            raise
        rollback_nonce = sha256_bytes(f"{name}:{fail_stage}".encode("ascii"))
        rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
            directory_fd,
            name,
            rollback_nonce,
            name,
            descriptor,
            published_info,
            payload,
            "qualification fsync rollback",
        )
        temp_rollback_ok, _quarantine_name = _quarantine_held_entry_if_exact(
            directory_fd,
            name,
            rollback_nonce,
            temp_name,
            descriptor,
            published_info,
            payload,
            "qualification fsync temp rollback",
        )
        rollback_ok = temp_rollback_ok and rollback_ok
        if not rollback_ok:
            raise ContractError(
                "qualification rollback saw an inode or byte substitution"
            )
        os.fsync(directory_fd)
        if _entry_stat_or_none(directory_fd, name) is not None:
            raise ContractError("qualification fsync rollback left canonical residue")
    finally:
        os.close(descriptor)


def _linux_smoke_concurrent_publisher_lease(
    directory_path: Path,
    held_directory_fd: int,
    output_record: dict[str, Any],
    accepted_name: str,
    job_id: str,
    nonce: str,
) -> dict[str, Any]:
    read_descriptor, write_descriptor = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_descriptor)
        child_fd: int | None = None
        try:
            os.close(held_directory_fd)
            child_fd, child_record = open_owned_output_directory(directory_path)
            if child_record != output_record:
                raise ContractError("concurrent lease child directory differs")
            try:
                acquire_publisher_lease(
                    child_fd, output_record, accepted_name, job_id, nonce
                )
            except ContractError as error:
                if "another live publisher" not in str(error):
                    raise
                payload = stable_json_bytes(
                    {"result": "exclusive_flock_rejected_concurrent_process"}
                )
                _write_all(write_descriptor, payload)
                os._exit(0)
            raise ContractError("concurrent lease child acquired held flock")
        except BaseException as error:
            _write_all(
                write_descriptor,
                stable_json_bytes(
                    {"result": "error", "detail": f"{type(error).__name__}: {error}"}
                ),
            )
            os._exit(97)
        finally:
            if child_fd is not None:
                os.close(child_fd)
    os.close(write_descriptor)
    try:
        payload = b""
        while True:
            block = os.read(read_descriptor, 4096)
            if not block:
                break
            payload += block
    finally:
        os.close(read_descriptor)
    waited_pid, status = os.waitpid(child_pid, 0)
    if waited_pid != child_pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
        raise ContractError("linux-smoke concurrent lease child failed")
    result = _parse_json_object_bytes(payload, "linux-smoke concurrent lease result")
    if result != {"result": "exclusive_flock_rejected_concurrent_process"}:
        raise ContractError("linux-smoke concurrent lease result differs")
    return {"child_pid": child_pid, **result}


def _linux_qualification_private_key(stage: str, evaluator_sha256: str) -> bytes:
    _linux_receipt_crash_stage(stage)
    if re.fullmatch(r"[0-9a-f]{64}", evaluator_sha256) is None:
        raise ContractError("Linux qualification evaluator hash differs")
    return hashlib.sha256(
        b"r12-linux-qualification-ephemeral-broker-key-v1\x00"
        + stage.encode("ascii")
        + bytes.fromhex(evaluator_sha256)
    ).digest()


def _canonical_python_executable_identity() -> dict[str, str]:
    path = Path(sys.executable).resolve(strict=True)
    path = _require_normalized_absolute_path(path, "qualification Python executable")
    return {"path": str(path), "sha256": sha256_regular_file(path)}


def _linux_qualification_broker_request(
    stage: str,
    evaluator_sha256: str,
    expected_signing_key: dict[str, Any],
    *,
    process_id: int | None = None,
) -> dict[str, Any]:
    _linux_receipt_crash_stage(stage)
    return build_delegated_key_broker_request(
        authorization_sha256=sha256_bytes(
            f"linux-qualification-authorization:{stage}".encode("ascii")
        ),
        source_manifest_sha256=sha256_bytes(b"linux-qualification-no-source-manifest"),
        evaluator_sha256=evaluator_sha256,
        wrapper_sha256=sha256_bytes(b"linux-qualification-no-wrapper-authority"),
        runtime_identity={
            "python": _canonical_python_executable_identity(),
            "python_startup": {"mode": PYTHON_STARTUP_MODE},
        },
        expected_signing_key=expected_signing_key,
        process_id=process_id,
    )


def _validate_linux_qualification_result(result: Any) -> dict[str, Any]:
    _require_exact_keys(
        result, LINUX_QUALIFICATION_RESULT_KEYS, "Linux qualification result"
    )
    if (
        result["schema"] != LINUX_QUALIFICATION_SCHEMA
        or result["scientific_decode_executed"] is not False
        or result["gpu_required"] is not False
        or result["filesystem"] != "lustre"
        or result["status"] != LINUX_QUALIFICATION_STATUS
        or result["claim_boundary"] != LINUX_QUALIFICATION_CLAIM_BOUNDARY
        or re.fullmatch(r"[0-9a-f]{64}", result["evaluator_sha256"]) is None
        or result["checks"] != list(LINUX_QUALIFICATION_REQUIRED_CHECKS)
    ):
        raise ContractError("Linux qualification result identity differs")

    evidence = result["qualification_evidence"]
    _require_exact_keys(
        evidence, LINUX_QUALIFICATION_EVIDENCE_KEYS, "Linux qualification evidence"
    )
    broker = evidence["delegated_key_broker_transfer"]
    _require_exact_keys(
        broker,
        {
            "authority_boundary",
            "request_sha256",
            "request_process_id",
            "brokered_signing_key",
            "scm_rights_descriptor_received",
        },
        "Linux qualification broker evidence",
    )
    _require_exact_keys(
        broker["brokered_signing_key"],
        SIGNING_KEY_RECORD_KEYS,
        "Linux qualification evidence signing key",
    )
    expected_mechanics_signing_key = signing_key_record(
        _linux_qualification_private_key(
            "after_complete_receipt_write", result["evaluator_sha256"]
        )
    )
    if (
        broker["authority_boundary"] != LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
        or re.fullmatch(r"[0-9a-f]{64}", broker["request_sha256"]) is None
        or type(broker["request_process_id"]) is not int
        or broker["request_process_id"] <= 0
        or broker["scm_rights_descriptor_received"] is not True
        or not _strict_equal(
            broker["brokered_signing_key"], expected_mechanics_signing_key
        )
    ):
        raise ContractError("Linux qualification broker evidence differs")

    publisher = evidence["publisher_lease"]
    _require_exact_keys(
        publisher, {"lease", "concurrent_attempt"}, "Linux qualification lease evidence"
    )
    _require_exact_keys(
        publisher["lease"],
        {
            "accepted_name",
            "authorization_nonce",
            "directory_device",
            "directory_inode",
            "job_id",
            "owner_pid",
        },
        "Linux qualification publisher lease",
    )
    _require_exact_keys(
        publisher["concurrent_attempt"],
        {"child_pid", "result"},
        "Linux qualification concurrent lease evidence",
    )
    lease = publisher["lease"]
    concurrent = publisher["concurrent_attempt"]
    if (
        lease["accepted_name"] != "qualification-output.json"
        or re.fullmatch(r"[0-9a-f]{64}", lease["authorization_nonce"]) is None
        or lease["job_id"] != "1"
        or any(
            type(lease[key]) is not int or lease[key] < 0
            for key in ("directory_device", "directory_inode", "owner_pid")
        )
        or lease["owner_pid"] <= 0
        or type(concurrent["child_pid"]) is not int
        or concurrent["child_pid"] <= 0
        or concurrent["result"] != "exclusive_flock_rejected_concurrent_process"
    ):
        raise ContractError("Linux qualification publisher lease evidence differs")

    cleanup = evidence["signed_stale_cleanup"]
    _require_exact_keys(
        cleanup,
        {
            "authorization",
            "source_name",
            "quarantine_name",
            "source_inode",
            "quarantine_inode",
            "pathname_unlink_used",
        },
        "Linux qualification stale-cleanup evidence",
    )
    foreign = evidence["foreign_inode_substitution"]
    _require_exact_keys(
        foreign,
        {
            "authorization",
            "substituted_path",
            "foreign_inode",
            "foreign_payload_sha256",
            "foreign_inode_preserved",
        },
        "Linux qualification foreign-inode evidence",
    )
    for label, authorization in (
        ("cleanup", cleanup["authorization"]),
        ("foreign", foreign["authorization"]),
    ):
        _require_exact_keys(
            authorization,
            {
                "schema",
                "authority_boundary",
                "payload_sha256",
                "ephemeral_public_key_hex",
                "signature_hex",
                "signature_verified",
            },
            f"Linux qualification {label} authorization evidence",
        )
        if (
            authorization["schema"] != "r12_linux_smoke_stale_cleanup_authorization_v1"
            or authorization["authority_boundary"]
            != LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
            or re.fullmatch(r"[0-9a-f]{64}", authorization["payload_sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", authorization["ephemeral_public_key_hex"])
            is None
            or re.fullmatch(r"[0-9a-f]{128}", authorization["signature_hex"]) is None
            or authorization["signature_verified"] is not True
        ):
            raise ContractError(
                f"Linux qualification {label} authorization evidence differs"
            )
    if (
        not all(
            isinstance(cleanup[key], str) for key in ("source_name", "quarantine_name")
        )
        or type(cleanup["source_inode"]) is not int
        or cleanup["source_inode"] < 0
        or type(cleanup["quarantine_inode"]) is not int
        or cleanup["quarantine_inode"] < 0
        or cleanup["source_inode"] != cleanup["quarantine_inode"]
        or cleanup["pathname_unlink_used"] is not False
        or not isinstance(foreign["substituted_path"], str)
        or type(foreign["foreign_inode"]) is not int
        or foreign["foreign_inode"] < 0
        or re.fullmatch(r"[0-9a-f]{64}", foreign["foreign_payload_sha256"]) is None
        or foreign["foreign_inode_preserved"] is not True
    ):
        raise ContractError("Linux qualification retained-inode evidence differs")

    substitution = evidence["held_evaluator_path_substitution"]
    _require_exact_keys(
        substitution,
        LINUX_QUALIFICATION_EVALUATOR_SUBSTITUTION_KEYS,
        "Linux qualification evaluator substitution evidence",
    )
    if (
        not all(
            isinstance(substitution[key], str)
            for key in ("source_name", "retained_name", "replacement_name")
        )
        or substitution["source_name"] != substitution["replacement_name"]
        or any(
            type(substitution[key]) is not int or substitution[key] < 0
            for key in ("source_inode", "retained_inode", "replacement_inode")
        )
        or substitution["source_inode"] != substitution["retained_inode"]
        or substitution["replacement_inode"] == substitution["retained_inode"]
        or substitution["held_sha256"] != result["evaluator_sha256"]
        or substitution["retained_sha256"] != result["evaluator_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", substitution["replacement_sha256"]) is None
        or substitution["replacement_sha256"] == result["evaluator_sha256"]
        or substitution["substitution_before_first_child"] is not True
        or substitution["original_inode_retained"] is not True
    ):
        raise ContractError("Linux qualification evaluator substitution differs")

    directory = evidence["directory_path_substitution"]
    _require_exact_keys(
        directory,
        {"held_device", "held_inode", "replacement_rejected"},
        "Linux qualification directory substitution evidence",
    )
    if (
        type(directory["held_device"]) is not int
        or directory["held_device"] < 0
        or type(directory["held_inode"]) is not int
        or directory["held_inode"] < 0
        or directory["replacement_rejected"] is not True
    ):
        raise ContractError("Linux qualification directory substitution differs")

    cases = result["receipt_crash_cases"]
    if type(cases) is not list or len(cases) != len(LINUX_RECEIPT_CRASH_STAGES):
        raise ContractError("Linux qualification crash-case count differs")
    expected_source: dict[str, Any] | None = None
    for case, (stage, exit_code, replay) in zip(
        cases, LINUX_RECEIPT_CRASH_STAGES, strict=True
    ):
        _require_exact_keys(
            case, LINUX_QUALIFICATION_CRASH_CASE_KEYS, "Linux qualification crash case"
        )
        source = case["evaluator_source"]
        _require_exact_keys(
            source,
            {"sha256", "byte_count", "descriptor_kind", "seals"},
            "Linux qualification crash evaluator source",
        )
        if expected_source is None:
            expected_source = source
        if (
            case["stage"] != stage
            or case["child_exit_code"] != exit_code
            or case["expected_replay"] != replay
            or case["observed_replay"] != replay
            or case["independent_report_marker_replay"] != "validated"
            or type(case["receipt_size"]) is not int
            or case["receipt_size"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", case["broker_request_sha256"]) is None
            or not _strict_equal(source, expected_source)
            or source["sha256"] != result["evaluator_sha256"]
            or type(source["byte_count"]) is not int
            or source["byte_count"] <= 0
            or source["descriptor_kind"] != "sealed_memfd"
            or source["seals"] != _required_memfd_seals()
        ):
            raise ContractError("Linux qualification crash-case identity differs")
    sizes = [case["receipt_size"] for case in cases]
    if sizes[0] != 0 or not (0 < sizes[1] < sizes[2]):
        raise ContractError("Linux qualification crash-case receipt sizes differ")
    assert expected_source is not None
    return expected_source


def _validate_linux_qualification_accepted_publication(
    value: Any, result: dict[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        LINUX_QUALIFICATION_ACCEPTED_PUBLICATION_KEYS,
        "accepted Linux qualification publication",
    )
    request = value["broker_request"]
    _require_exact_keys(
        request, DELEGATED_KEY_BROKER_REQUEST_KEYS, "Linux qualification broker request"
    )
    expected_source = _validate_linux_qualification_result(result)
    stage = "after_complete_receipt_write"
    evaluator_sha256 = result["evaluator_sha256"]
    private_key = _linux_qualification_private_key(stage, evaluator_sha256)
    expected_signing_key = signing_key_record(private_key)
    if type(request["process_id"]) is not int or request["process_id"] <= 0:
        raise ContractError("Linux qualification broker process ID differs")
    expected_request = _linux_qualification_broker_request(
        stage,
        evaluator_sha256,
        expected_signing_key,
        process_id=request["process_id"],
    )
    if not _strict_equal(request, expected_request):
        raise ContractError("Linux qualification broker request replay differs")
    request_sha256 = sha256_bytes(stable_json_bytes(request))
    report = value["report"]
    marker = value["marker"]
    receipt = value["receipt"]
    report_payload = stable_json_bytes(report)
    marker_payload = stable_json_bytes(marker)
    receipt_payload = stable_json_bytes(receipt)
    _validate_linux_qualification_receipt_payloads(
        report,
        report_payload,
        marker,
        marker_payload,
        receipt,
        receipt_payload,
        stage,
        expected_source,
        request_sha256,
        expected_signing_key,
    )
    final_case = result["receipt_crash_cases"][-1]
    if final_case["broker_request_sha256"] != request_sha256:
        raise ContractError("Linux qualification final crash-case request differs")
    return expected_signing_key


def build_replayable_linux_qualification_receipt(
    qualification_result: dict[str, Any],
    accepted_publication: dict[str, Any],
    private_key: bytes,
) -> dict[str, Any]:
    expected_signing_key = _validate_linux_qualification_accepted_publication(
        accepted_publication, qualification_result
    )
    if signing_key_record(private_key) != expected_signing_key:
        raise ContractError("Linux qualification receipt signing key differs")
    payload = {
        "schema": LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_SCHEMA,
        "qualification_result": qualification_result,
        "accepted_publication": accepted_publication,
    }
    receipt = {
        **payload,
        "signature_hex": _ed25519_sign(private_key, stable_json_bytes(payload)).hex(),
    }
    validate_replayable_linux_qualification_receipt(receipt)
    return receipt


def validate_replayable_linux_qualification_receipt(value: Any) -> None:
    _require_exact_keys(
        value,
        LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_KEYS,
        "replayable Linux qualification receipt",
    )
    if value["schema"] != LINUX_QUALIFICATION_AUTHORIZATION_RECEIPT_SCHEMA:
        raise ContractError("replayable Linux qualification receipt schema differs")
    expected_signing_key = _validate_linux_qualification_accepted_publication(
        value["accepted_publication"], value["qualification_result"]
    )
    payload = {key: item for key, item in value.items() if key != "signature_hex"}
    if (
        type(value["signature_hex"]) is not str
        or re.fullmatch(r"[0-9a-f]{128}", value["signature_hex"]) is None
        or not _ed25519_verify(
            bytes.fromhex(expected_signing_key["public_key_hex"]),
            bytes.fromhex(value["signature_hex"]),
            stable_json_bytes(payload),
        )
    ):
        raise ContractError("replayable Linux qualification receipt signature differs")


def _create_linux_qualification_signing_key_descriptor(private_key: bytes) -> int:
    if sys.platform.startswith("linux") and hasattr(os, "memfd_create"):
        descriptor = os.memfd_create(
            "r12-linux-qualification-broker-key", os.MFD_ALLOW_SEALING
        )
        sealed = True
    else:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix="r12-linux-qualification-broker-key-test-"
        )
        os.unlink(temporary_path)
        sealed = False
    try:
        _write_all(descriptor, private_key)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        if sealed:
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _required_memfd_seals())
            read_sealed_signing_key(descriptor, signing_key_record(private_key))
        else:
            _read_linux_qualification_broker_key(
                descriptor, signing_key_record(private_key)
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_linux_qualification_broker_key(
    descriptor: int, expected_signing_key: dict[str, Any]
) -> bytes:
    if sys.platform.startswith("linux"):
        return read_sealed_signing_key(descriptor, expected_signing_key)
    payload, info = _read_exact_descriptor_bytes(
        descriptor, "non-Linux qualification broker key"
    )
    if (
        info.st_nlink != 0
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o400
        or signing_key_record(payload) != expected_signing_key
    ):
        raise ContractError("non-Linux qualification broker key differs")
    return payload


def _linux_qualification_broker_exchange(
    stage: str, evaluator_sha256: str
) -> tuple[int, bytes, dict[str, Any], dict[str, Any]]:
    """Exercise the production request/SCM_RIGHTS receiver with zero authority."""
    private_key = _linux_qualification_private_key(stage, evaluator_sha256)
    expected_signing_key = signing_key_record(private_key)
    request = _linux_qualification_broker_request(
        stage, evaluator_sha256, expected_signing_key
    )
    request_payload = stable_json_bytes(request)
    key_descriptor = _create_linux_qualification_signing_key_descriptor(private_key)
    try:
        client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    except OSError as error:
        unsupported_socket_errors = {
            errno.EPROTONOSUPPORT,
            getattr(errno, "ESOCKTNOSUPPORT", errno.EPROTONOSUPPORT),
        }
        if error.errno not in unsupported_socket_errors:
            raise
        client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    child_pid = os.fork()
    if child_pid == 0:
        client.close()
        try:
            observed_request = server.recv(len(request_payload) + 1)
            if observed_request != request_payload:
                os._exit(96)
            response = stable_json_bytes(
                {
                    "schema": DELEGATED_KEY_BROKER_RESPONSE_SCHEMA,
                    "request_sha256": sha256_bytes(observed_request),
                }
            )
            descriptors = array.array("i", [key_descriptor])
            server.sendmsg(
                [response],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())],
            )
            os._exit(0)
        except BaseException:
            os._exit(97)
    server.close()
    received_descriptor: int | None = None
    try:
        try:
            received_descriptor, received_key = (
                receive_delegated_signing_key_from_broker(
                    client.fileno(),
                    request,
                    expected_signing_key,
                    key_reader=_read_linux_qualification_broker_key,
                )
            )
        except BaseException:
            os.waitpid(child_pid, 0)
            raise
    finally:
        client.close()
        os.close(key_descriptor)
    waited_pid, status = os.waitpid(child_pid, 0)
    if (
        waited_pid != child_pid
        or not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
    ):
        if received_descriptor is not None:
            os.close(received_descriptor)
        raise ContractError("Linux qualification broker child failed")
    if received_descriptor is None or received_key != private_key:
        if received_descriptor is not None:
            os.close(received_descriptor)
        raise ContractError("Linux qualification brokered key differs")
    evidence = {
        "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
        "request_sha256": sha256_bytes(request_payload),
        "request_process_id": request["process_id"],
        "brokered_signing_key": expected_signing_key,
        "scm_rights_descriptor_received": True,
    }
    return received_descriptor, received_key, request, evidence


def _linux_smoke_signed_cleanup_authorization(
    job_id: str, nonce: str, entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema": "r12_linux_smoke_stale_cleanup_authorization_v1",
        "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
        "job_id": job_id,
        "authorization_nonce": nonce,
        "stale_cleanup_entries": entries,
    }
    private_key = os.urandom(32)
    public_key = _ed25519_public_key(private_key)
    signature = _ed25519_sign(private_key, stable_json_bytes(payload))
    if not _ed25519_verify(public_key, signature, stable_json_bytes(payload)):
        raise ContractError("linux-smoke cleanup signature did not verify")
    authorization = {
        "slurm_allocation": {"job_id": job_id},
        "authorization_nonce": nonce,
        "stale_cleanup_entries": entries,
    }
    evidence = {
        "schema": payload["schema"],
        "authority_boundary": LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
        "payload_sha256": sha256_bytes(stable_json_bytes(payload)),
        "ephemeral_public_key_hex": public_key.hex(),
        "signature_hex": signature.hex(),
        "signature_verified": True,
    }
    return authorization, evidence


def run_linux_publication_qualification(
    root: Path, expected_evaluator_sha256: str
) -> dict[str, Any]:
    """Exercise Linux/Lustre custody mechanics without CUDA or scientific decoding."""
    if not sys.platform.startswith("linux"):
        raise ContractError("linux-smoke requires Linux")
    if re.fullmatch(r"[0-9a-f]{64}", expected_evaluator_sha256) is None:
        raise ContractError("linux-smoke evaluator hash is not lowercase SHA-256")
    if sha256_regular_file(Path(__file__)) != expected_evaluator_sha256:
        raise ContractError("linux-smoke evaluator bytes differ from reviewed hash")
    root = _require_normalized_absolute_path(root, "linux-smoke root")
    if _linux_mount_fstype(root) != "lustre":
        raise ContractError("linux-smoke root must reside on a Lustre mount")

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
        raise ContractError("linux-smoke could not enforce PR_SET_DUMPABLE=0")
    memfd = os.memfd_create("r12-linux-smoke", os.MFD_ALLOW_SEALING)
    try:
        _write_all(memfd, b"sealed-linux-smoke\n")
        os.fsync(memfd)
        fcntl.fcntl(memfd, fcntl.F_ADD_SEALS, _required_memfd_seals())
        if (
            fcntl.fcntl(memfd, fcntl.F_GET_SEALS) != _required_memfd_seals()
            or os.pread(memfd, 64, 0) != b"sealed-linux-smoke\n"
        ):
            raise ContractError("linux-smoke memfd seal/readback differs")
    finally:
        os.close(memfd)

    qualification_broker_fd, qualification_broker_key, _request, broker_evidence = (
        _linux_qualification_broker_exchange(
            "after_complete_receipt_write", expected_evaluator_sha256
        )
    )
    try:
        _read_linux_qualification_broker_key(
            qualification_broker_fd,
            signing_key_record(qualification_broker_key),
        )
    finally:
        os.close(qualification_broker_fd)

    root_fd, root_record = open_owned_output_directory(root)
    transaction_name = f".r12-linux-smoke-{secrets.token_hex(16)}"
    transaction_path = root / transaction_name
    os.mkdir(transaction_name, 0o700, dir_fd=root_fd)
    os.fsync(root_fd)
    transaction_fd, transaction_record = open_owned_output_directory(transaction_path)
    checks: list[str] = []
    receipt_crash_cases: list[dict[str, Any]] = []
    accepted_publication: dict[str, Any] | None = None
    qualification_evidence: dict[str, Any] = {}
    qualification_evidence["delegated_key_broker_transfer"] = broker_evidence
    qualification_job_id = "1"
    qualification_nonce = secrets.token_hex(32)
    qualification_accepted_name = "qualification-output.json"
    publisher_lease: dict[str, Any] | None = None
    try:
        publisher_lease = acquire_publisher_lease(
            transaction_fd,
            transaction_record,
            qualification_accepted_name,
            qualification_job_id,
            qualification_nonce,
        )
        qualification_evidence["publisher_lease"] = {
            "lease": publisher_lease,
            "concurrent_attempt": _linux_smoke_concurrent_publisher_lease(
                transaction_path,
                transaction_fd,
                transaction_record,
                qualification_accepted_name,
                qualification_job_id,
                qualification_nonce,
            ),
        }
        checks.extend(("publisher_lease_acquired", "concurrent_flock_rejected"))
        checks.append("production_scm_rights_broker_transfer")

        stale_name = (
            f".{qualification_accepted_name}.r12-candidate-"
            f"{qualification_job_id}-{qualification_nonce}"
        )
        stale_descriptor = os.open(
            stale_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=transaction_fd,
        )
        try:
            _write_all(stale_descriptor, b"linux-smoke-authorized-stale\n")
            os.fsync(stale_descriptor)
        finally:
            os.close(stale_descriptor)
        os.fsync(transaction_fd)
        stale_entry, _stale_payload = _stale_cleanup_record(
            transaction_fd, stale_name, "linux-smoke authorized stale entry"
        )
        cleanup_authorization, cleanup_signature = (
            _linux_smoke_signed_cleanup_authorization(
                qualification_job_id, qualification_nonce, [stale_entry]
            )
        )
        quarantined = cleanup_stale_publication_entries(
            transaction_fd,
            qualification_accepted_name,
            cleanup_authorization,
            publisher_lease,
        )
        if (
            len(quarantined) != 1
            or _entry_stat_or_none(transaction_fd, stale_name) is not None
        ):
            raise ContractError("linux-smoke signed stale cleanup did not quarantine")
        quarantine_name = quarantined[0]
        quarantine_entry, _quarantine_payload = _stale_cleanup_record(
            transaction_fd, quarantine_name, "linux-smoke stale quarantine"
        )
        rebound_quarantine = {**quarantine_entry, "name": stale_name}
        if rebound_quarantine != stale_entry:
            raise ContractError("linux-smoke stale quarantine identity differs")
        qualification_evidence["signed_stale_cleanup"] = {
            "authorization": cleanup_signature,
            "source_name": stale_name,
            "quarantine_name": quarantine_name,
            "source_inode": stale_entry["inode"],
            "quarantine_inode": quarantine_entry["inode"],
            "pathname_unlink_used": False,
        }
        checks.append("ephemeral_signed_exact_stale_cleanup_quarantined")

        hostile_name = (
            f".{qualification_accepted_name}.r12-report-"
            f"{qualification_nonce}-{secrets.token_hex(16)}"
        )
        hostile_descriptor = os.open(
            hostile_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=transaction_fd,
        )
        try:
            _write_all(hostile_descriptor, b"linux-smoke-authorized-before-swap\n")
            os.fsync(hostile_descriptor)
        finally:
            os.close(hostile_descriptor)
        os.fsync(transaction_fd)
        existing_quarantine, _payload = _stale_cleanup_record(
            transaction_fd, quarantine_name, "existing linux-smoke quarantine"
        )
        hostile_entry, _payload = _stale_cleanup_record(
            transaction_fd, hostile_name, "linux-smoke hostile stale entry"
        )
        hostile_entries = sorted(
            (existing_quarantine, hostile_entry), key=lambda entry: entry["name"]
        )
        hostile_authorization, hostile_signature = (
            _linux_smoke_signed_cleanup_authorization(
                qualification_job_id, qualification_nonce, hostile_entries
            )
        )
        displaced_authorized_name = ".linux-smoke-displaced-authorized-inode"
        substituted_quarantine_name: str | None = None

        def substitute_quarantine(
            stage: str, _source_name: str, destination_name: str
        ) -> None:
            nonlocal substituted_quarantine_name
            if stage != "after_quarantine_rename":
                raise ContractError("linux-smoke substitution stage differs")
            os.rename(
                destination_name,
                displaced_authorized_name,
                src_dir_fd=transaction_fd,
                dst_dir_fd=transaction_fd,
            )
            replacement = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=transaction_fd,
            )
            try:
                _write_all(replacement, b"foreign-inode-must-survive\n")
                os.fsync(replacement)
            finally:
                os.close(replacement)
            os.fsync(transaction_fd)
            substituted_quarantine_name = destination_name

        try:
            cleanup_stale_publication_entries(
                transaction_fd,
                qualification_accepted_name,
                hostile_authorization,
                publisher_lease,
                failure_injector=substitute_quarantine,
            )
        except ContractError as error:
            if "substitution was quarantined and preserved" not in str(error):
                raise
        else:
            raise ContractError("linux-smoke accepted stale inode substitution")
        if substituted_quarantine_name is None:
            raise ContractError("linux-smoke substitution injector did not run")
        foreign_payload, foreign_info = read_directory_entry_bytes(
            transaction_fd,
            substituted_quarantine_name,
            "linux-smoke substituted foreign inode",
        )
        if foreign_payload != b"foreign-inode-must-survive\n":
            raise ContractError("linux-smoke foreign inode was not preserved")
        qualification_evidence["foreign_inode_substitution"] = {
            "authorization": hostile_signature,
            "substituted_path": substituted_quarantine_name,
            "foreign_inode": foreign_info.st_ino,
            "foreign_payload_sha256": sha256_bytes(foreign_payload),
            "foreign_inode_preserved": True,
        }
        checks.append("foreign_inode_after_quarantine_rename_preserved")

        for name in ("rename-source", "rename-collision"):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=transaction_fd,
            )
            os.write(descriptor, name.encode("ascii"))
            os.fsync(descriptor)
            os.close(descriptor)
        rename_noreplace_at(
            transaction_fd, "rename-source", transaction_fd, "rename-final"
        )
        os.fsync(transaction_fd)
        try:
            rename_noreplace_at(
                transaction_fd,
                "rename-collision",
                transaction_fd,
                "rename-final",
            )
        except FileExistsError:
            checks.append("renameat2_noreplace")
        else:
            raise ContractError("renameat2 no-replace overwrote an existing entry")

        _linux_smoke_publish_with_injected_fsync_failure(
            transaction_fd, "pre-fsync-failure", "before_directory_fsync"
        )
        _linux_smoke_publish_with_injected_fsync_failure(
            transaction_fd, "post-fsync-failure", "after_directory_fsync"
        )
        checks.append("directory_fsync_failure_rollback")

        _evaluator_record, evaluator_payload = _regular_file_closure_record(
            Path(__file__).resolve(strict=True),
            relative_path=None,
            label="linux-smoke evaluator substitution source",
        )
        if sha256_bytes(evaluator_payload) != expected_evaluator_sha256:
            raise ContractError("linux-smoke substitution source hash differs")
        evaluator_probe_name = ".linux-smoke-held-evaluator-source.py"
        evaluator_retained_name = ".linux-smoke-held-evaluator-retained.py"
        evaluator_probe_path = transaction_path / evaluator_probe_name
        evaluator_probe_descriptor = os.open(
            evaluator_probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=transaction_fd,
        )
        try:
            _write_all(evaluator_probe_descriptor, evaluator_payload)
            os.fchmod(evaluator_probe_descriptor, 0o444)
            os.fsync(evaluator_probe_descriptor)
        finally:
            os.close(evaluator_probe_descriptor)
        os.fsync(transaction_fd)
        evaluator_probe_entry, _probe_payload = _stale_cleanup_record(
            transaction_fd,
            evaluator_probe_name,
            "linux-smoke held-evaluator substitution source",
        )
        evaluator_substitution_evidence: dict[str, Any] | None = None

        def substitute_evaluator_path() -> None:
            nonlocal evaluator_substitution_evidence
            rename_noreplace_at(
                transaction_fd,
                evaluator_probe_name,
                transaction_fd,
                evaluator_retained_name,
            )
            os.fsync(transaction_fd)
            retained_entry, _retained_payload = _stale_cleanup_record(
                transaction_fd,
                evaluator_retained_name,
                "linux-smoke retained evaluator source",
            )
            hostile_payload = b"raise SystemExit('substituted evaluator pathname')\n"
            replacement_descriptor = os.open(
                evaluator_probe_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=transaction_fd,
            )
            try:
                _write_all(replacement_descriptor, hostile_payload)
                os.fchmod(replacement_descriptor, 0o444)
                os.fsync(replacement_descriptor)
            finally:
                os.close(replacement_descriptor)
            os.fsync(transaction_fd)
            replacement_entry, _replacement_payload = _stale_cleanup_record(
                transaction_fd,
                evaluator_probe_name,
                "linux-smoke substituted evaluator pathname",
            )
            if {
                **retained_entry,
                "name": evaluator_probe_name,
            } != evaluator_probe_entry:
                raise ContractError(
                    "linux-smoke retained evaluator source identity differs"
                )
            evaluator_substitution_evidence = {
                "source_name": evaluator_probe_name,
                "retained_name": evaluator_retained_name,
                "replacement_name": evaluator_probe_name,
                "source_inode": evaluator_probe_entry["inode"],
                "retained_inode": retained_entry["inode"],
                "replacement_inode": replacement_entry["inode"],
                "held_sha256": expected_evaluator_sha256,
                "retained_sha256": retained_entry["sha256"],
                "replacement_sha256": replacement_entry["sha256"],
                "substitution_before_first_child": True,
                "original_inode_retained": True,
            }

        receipt_crash_cases, accepted_publication = (
            _run_linux_receipt_crash_consistency_cases(
                transaction_fd,
                transaction_path,
                expected_evaluator_sha256,
                evaluator_path=evaluator_probe_path,
                first_child_before_spawn=substitute_evaluator_path,
            )
        )
        if evaluator_substitution_evidence is None:
            raise ContractError(
                "linux-smoke evaluator substitution callback did not execute"
            )
        qualification_evidence["held_evaluator_path_substitution"] = (
            evaluator_substitution_evidence
        )
        checks.extend(
            (
                "o_sync_receipt_before_parent_fsync_rejected",
                "o_sync_partial_receipt_rejected",
                "o_sync_complete_receipt_independent_replay_accepted",
            )
        )
        checks.append("held_evaluator_pathname_substitution_exercised")

        os.symlink("rename-final", "hostile-symlink", dir_fd=transaction_fd)
        try:
            read_directory_entry_bytes(
                transaction_fd, "hostile-symlink", "linux-smoke symlink"
            )
        except ContractError:
            checks.append("symlink_rejected")
        else:
            raise ContractError("linux-smoke accepted a symlink")
        os.link(
            "rename-final",
            "hostile-hardlink",
            src_dir_fd=transaction_fd,
            dst_dir_fd=transaction_fd,
        )
        try:
            read_directory_entry_bytes(
                transaction_fd, "rename-final", "linux-smoke hardlink"
            )
        except ContractError:
            checks.append("hardlink_rejected")
        else:
            raise ContractError("linux-smoke accepted a multiply linked file")

        crash_prefix = ".random-crash-temp-"
        crash_names = [f"{crash_prefix}{secrets.token_hex(16)}" for _ in range(3)]
        for name in crash_names:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=transaction_fd,
            )
            os.close(descriptor)
        for name in os.listdir(transaction_fd):
            if name.startswith(crash_prefix):
                os.unlink(name, dir_fd=transaction_fd)
        os.fsync(transaction_fd)
        if any(name.startswith(crash_prefix) for name in os.listdir(transaction_fd)):
            raise ContractError("linux-smoke random crash-temp cleanup failed")
        checks.append("random_temp_crash_cleanup")

        displaced_name = f"{transaction_name}.displaced"
        replacement_name = f"{transaction_name}.replacement"
        os.rename(
            transaction_name, displaced_name, src_dir_fd=root_fd, dst_dir_fd=root_fd
        )
        os.mkdir(replacement_name, 0o700, dir_fd=root_fd)
        os.rename(
            replacement_name, transaction_name, src_dir_fd=root_fd, dst_dir_fd=root_fd
        )
        try:
            validate_output_directory_fd(
                transaction_fd, transaction_record, require_path_identity=True
            )
        except ContractError:
            checks.append("held_directory_pathname_substitution_rejected")
            qualification_evidence["directory_path_substitution"] = {
                "held_device": transaction_record["device"],
                "held_inode": transaction_record["inode"],
                "replacement_rejected": True,
            }
        else:
            raise ContractError("linux-smoke accepted directory pathname substitution")
        os.rmdir(transaction_name, dir_fd=root_fd)
        os.rename(
            displaced_name, transaction_name, src_dir_fd=root_fd, dst_dir_fd=root_fd
        )
        os.fsync(root_fd)
        validate_output_directory_fd(
            transaction_fd, transaction_record, require_path_identity=True
        )
        checks.append("lustre_file_rename_fsync_reopen")
    finally:
        if publisher_lease is not None:
            release_publisher_lease(transaction_fd, publisher_lease)
        for name in os.listdir(transaction_fd):
            os.unlink(name, dir_fd=transaction_fd)
        os.fsync(transaction_fd)
        os.close(transaction_fd)
        os.rmdir(transaction_name, dir_fd=root_fd)
        os.fsync(root_fd)
        validate_output_directory_fd(root_fd, root_record, require_path_identity=True)
        os.close(root_fd)
    if (
        sha256_regular_file(Path(__file__).resolve(strict=True))
        != expected_evaluator_sha256
    ):
        raise ContractError("linux-smoke evaluator changed during qualification")
    if accepted_publication is None:
        raise ContractError("linux-smoke accepted publication receipt is absent")
    qualification_result = {
        "schema": LINUX_QUALIFICATION_SCHEMA,
        "scientific_decode_executed": False,
        "gpu_required": False,
        "filesystem": "lustre",
        "evaluator_sha256": expected_evaluator_sha256,
        "checks": checks,
        "qualification_evidence": qualification_evidence,
        "receipt_crash_cases": receipt_crash_cases,
        "status": LINUX_QUALIFICATION_STATUS,
        "claim_boundary": LINUX_QUALIFICATION_CLAIM_BOUNDARY,
    }
    authorization_receipt = build_replayable_linux_qualification_receipt(
        qualification_result,
        accepted_publication,
        qualification_broker_key,
    )
    return {
        **qualification_result,
        "authorization_receipt": authorization_receipt,
    }


def _live_runtime_custody(args: argparse.Namespace) -> dict[str, Any]:
    python_fd = _decimal_descriptor(args.python_source_fd, "Python source FD")
    git_fd = _decimal_descriptor(args.git_source_fd, "Git source FD")
    scontrol_fd = _decimal_descriptor(args.scontrol_source_fd, "scontrol source FD")
    sacct_fd = _decimal_descriptor(args.sacct_source_fd, "sacct source FD")
    nvidia_smi_fd = _decimal_descriptor(
        args.nvidia_smi_source_fd, "nvidia-smi source FD"
    )
    custody = verify_runtime_source_manifest(
        Path(args.source_root),
        args.source_commit,
        Path(args.runtime_source_manifest),
        args.runtime_source_manifest_sha256,
        Path(args.python_bin),
        Path(args.git_bin),
        Path(args.scontrol_bin),
        Path(args.sacct_bin),
        Path(args.nvidia_smi_bin),
        manifest_descriptor=_decimal_descriptor(
            args.runtime_source_manifest_fd, "runtime-source manifest FD"
        ),
        python_descriptor=python_fd,
        git_descriptor=git_fd,
        scontrol_descriptor=scontrol_fd,
        sacct_descriptor=sacct_fd,
        nvidia_smi_descriptor=nvidia_smi_fd,
        require_sealed_startup=True,
        revalidate_existing_runtime=torch is not None,
    )
    if custody["source_root"] != str(ROOT):
        raise ContractError("evaluator SOURCE_ROOT differs from reviewed custody")
    return custody


def _verify_running_evaluator_descriptor(
    descriptor: int, expected_sha256: str
) -> tuple[bytes, dict[str, Any]]:
    payload, record = read_sealed_source_descriptor(
        descriptor, expected_sha256, "sealed evaluator source"
    )
    running_info = os.stat(__file__)
    descriptor_info = os.fstat(descriptor)
    if _file_identity(running_info) != _file_identity(descriptor_info):
        raise ContractError("running evaluator is not the inherited sealed descriptor")
    _read_exact_descriptor_bytes(
        descriptor,
        "running sealed evaluator",
        expected_info=descriptor_info,
        expected_payload=payload,
    )
    return payload, record


def _main_replay_accepted_bundle(args: argparse.Namespace) -> None:
    accepted = _require_normalized_absolute_path(Path(args.accepted), "accepted bundle")
    directory_fd, directory_record = open_owned_output_directory(accepted.parent)
    manifest_source, manifest_identity = _open_regular_single_link(
        Path(args.runtime_source_manifest)
    )
    publication_descriptors: list[int] = []
    running_source: BinaryIO | None = None
    try:
        external_manifest_bytes = manifest_source.read()
        _verify_open_identity(
            manifest_source, manifest_identity, Path(args.runtime_source_manifest)
        )
        external_manifest_info = os.fstat(manifest_source.fileno())
        if sha256_bytes(external_manifest_bytes) != args.runtime_source_manifest_sha256:
            raise ContractError("replay external manifest hash differs")
        marker_name = acceptance_commit_marker_name(accepted.name)
        receipt_name = durable_acceptance_receipt_name(accepted.name)
        for name in (accepted.name, marker_name, receipt_name):
            publication_descriptors.append(
                os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            )
        payload, final_info = _read_held_directory_entry_descriptor(
            directory_fd,
            accepted.name,
            publication_descriptors[0],
            "accepted bundle",
        )
        bundle = _parse_json_object_bytes(payload, "accepted bundle")
        if stable_json_bytes(bundle) != payload:
            raise ContractError("accepted bundle is not canonical stable JSON")
        replay_context = bundle["report"]["wrapper_acceptance"]
        if (
            accepted.name != replay_context["accepted_name"]
            or directory_record != replay_context["output_directory"]
        ):
            raise ContractError("accepted bundle pathname or parent inode differs")
        custody = bundle["report"]["execution"]["runtime_source_manifest"]
        running_evaluator = Path(__file__).resolve(strict=True)
        running_source, running_identity = _open_regular_single_link(running_evaluator)
        running_bytes = running_source.read()
        _verify_open_identity(running_source, running_identity, running_evaluator)
        if (
            sha256_bytes(running_bytes)
            != custody["files"]["train/eval_dws_eos_suppressed_trace.py"]
        ):
            raise ContractError("replay evaluator bytes differ from source manifest")
        observed_runtime = observe_runtime_identity(
            Path(args.python_bin),
            Path(args.git_bin),
            Path(args.scontrol_bin),
            Path(args.sacct_bin),
            Path(args.nvidia_smi_bin),
            expected_packages=custody["runtime"]["packages"],
            require_sealed_startup=True,
        )
        if observed_runtime != custody["runtime"]:
            raise ContractError("replay runtime identity differs from acceptance")
        activate_pinned_runtime_packages(observed_runtime)
        publication = _read_and_validate_accepted_publication_from_descriptors(
            directory_fd,
            accepted.name,
            publication_descriptors[0],
            publication_descriptors[1],
            publication_descriptors[2],
            external_manifest_bytes,
            external_manifest_info,
            expected_report_payload=payload,
            expected_report_info=final_info,
        )
        if publication[0] != payload or _file_identity(
            publication[1]
        ) != _file_identity(final_info):
            raise ContractError("accepted bundle changed during replay")
        marker_payload = publication[2]
        receipt_payload = publication[4]
        durable_receipt = publication[8]
        _read_exact_descriptor_bytes(
            manifest_source.fileno(),
            "replay external manifest",
            expected_info=external_manifest_info,
            expected_payload=external_manifest_bytes,
        )
        _verify_open_identity(
            manifest_source, manifest_identity, Path(args.runtime_source_manifest)
        )
        _read_exact_descriptor_bytes(
            running_source.fileno(),
            "replay evaluator source",
            expected_payload=running_bytes,
        )
        _verify_open_identity(running_source, running_identity, running_evaluator)
        validate_output_directory_fd(
            directory_fd, directory_record, require_path_identity=True
        )
        sys.stdout.buffer.write(
            stable_json_bytes(
                {
                    "accepted_path": str(accepted),
                    "accepted_bundle_sha256": sha256_bytes(payload),
                    "commit_marker_path": str(accepted.parent / marker_name),
                    "commit_marker_sha256": sha256_bytes(marker_payload),
                    "durable_acceptance_receipt_path": str(
                        accepted.parent / receipt_name
                    ),
                    "durable_acceptance_receipt_sha256": sha256_bytes(receipt_payload),
                    "final_inode": durable_receipt["final_inode"],
                    "report_sha256": durable_receipt["report_sha256"],
                    "status": (
                        "accepted_bundle_replay_verified_with_durable_post_fsync_receipt"
                    ),
                }
            )
        )
    finally:
        if running_source is not None:
            running_source.close()
        for descriptor in publication_descriptors:
            os.close(descriptor)
        manifest_source.close()
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "generate-sealed-report",
            "verify-private-candidate",
            "replay-accepted-bundle",
            "linux-smoke",
            "linux-receipt-crash-worker",
        ),
    )
    parser.add_argument("--acceptance-context-fd")
    parser.add_argument("--acceptance-context-sha256")
    parser.add_argument("--source-root")
    parser.add_argument("--source-commit")
    parser.add_argument("--runtime-source-manifest")
    parser.add_argument("--runtime-source-manifest-fd")
    parser.add_argument("--runtime-source-manifest-sha256")
    parser.add_argument("--python-bin")
    parser.add_argument("--git-bin")
    parser.add_argument("--scontrol-bin")
    parser.add_argument("--sacct-bin")
    parser.add_argument("--nvidia-smi-bin")
    parser.add_argument("--evaluator-source-fd")
    parser.add_argument("--model-source-fd")
    parser.add_argument("--python-source-fd")
    parser.add_argument("--git-source-fd")
    parser.add_argument("--scontrol-source-fd")
    parser.add_argument("--sacct-source-fd")
    parser.add_argument("--nvidia-smi-source-fd")
    parser.add_argument("--generator-signing-key-fd")
    parser.add_argument("--verifier-signing-key-fd")
    parser.add_argument("--report-output-fd")
    parser.add_argument("--candidate-fd")
    parser.add_argument("--accepted")
    parser.add_argument("--linux-smoke-root")
    parser.add_argument(
        "--linux-receipt-crash-stage",
        choices=tuple(
            stage for stage, _exit_code, _replay in LINUX_RECEIPT_CRASH_STAGES
        ),
    )
    parser.add_argument("--expected-evaluator-sha256")
    parser.add_argument("--expected-evaluator-byte-count")
    parser.add_argument("--expected-evaluator-descriptor-kind")
    parser.add_argument("--expected-evaluator-seals")
    parser.add_argument("--held-evaluator-fd")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    if not args.development_only:
        raise SystemExit("explicit --development-only acknowledgement is required")
    if args.command == "linux-receipt-crash-worker":
        if (
            args.linux_smoke_root is None
            or args.linux_receipt_crash_stage is None
            or args.expected_evaluator_sha256 is None
            or args.expected_evaluator_byte_count is None
            or args.expected_evaluator_descriptor_kind is None
            or args.expected_evaluator_seals is None
            or args.held_evaluator_fd is None
        ):
            raise ContractError(
                "Linux receipt crash worker requires root, stage, and held evaluator identity"
            )
        expected_source = {
            "sha256": args.expected_evaluator_sha256,
            "byte_count": _strict_decimal_int(
                args.expected_evaluator_byte_count,
                "held evaluator byte count",
                minimum=1,
            ),
            "descriptor_kind": args.expected_evaluator_descriptor_kind,
            "seals": _strict_decimal_int(
                args.expected_evaluator_seals,
                "held evaluator seals",
                minimum=0,
            ),
        }
        _run_linux_receipt_crash_worker(
            Path(args.linux_smoke_root),
            args.linux_receipt_crash_stage,
            expected_source,
            _decimal_descriptor(args.held_evaluator_fd, "held evaluator FD"),
        )
        raise ContractError("Linux receipt crash worker returned without interruption")
    if args.command == "linux-smoke":
        if args.linux_smoke_root is None or args.expected_evaluator_sha256 is None:
            raise ContractError(
                "linux-smoke requires --linux-smoke-root and reviewed evaluator hash"
            )
        sys.stdout.buffer.write(
            stable_json_bytes(
                run_linux_publication_qualification(
                    Path(args.linux_smoke_root), args.expected_evaluator_sha256
                )
            )
        )
        return
    if args.command == "replay-accepted-bundle":
        replay_required = (
            "accepted",
            "runtime_source_manifest",
            "runtime_source_manifest_sha256",
            "python_bin",
            "git_bin",
            "scontrol_bin",
            "sacct_bin",
            "nvidia_smi_bin",
        )
        if any(getattr(args, name) is None for name in replay_required):
            raise ContractError("accepted-bundle replay arguments are missing")
        _main_replay_accepted_bundle(args)
        return
    required = (
        "acceptance_context_fd",
        "acceptance_context_sha256",
        "source_root",
        "source_commit",
        "runtime_source_manifest",
        "runtime_source_manifest_sha256",
        "python_bin",
        "git_bin",
        "scontrol_bin",
        "sacct_bin",
        "nvidia_smi_bin",
        "runtime_source_manifest_fd",
        "evaluator_source_fd",
        "python_source_fd",
        "git_source_fd",
        "scontrol_source_fd",
        "sacct_source_fd",
        "nvidia_smi_source_fd",
    )
    if any(getattr(args, name) is None for name in required):
        raise ContractError(
            "live generation/verification descriptor arguments are missing"
        )

    context = load_wrapper_acceptance_context_descriptor(
        _decimal_descriptor(args.acceptance_context_fd, "acceptance context FD"),
        args.acceptance_context_sha256,
    )
    source_custody = _live_runtime_custody(args)
    evaluator_fd = _decimal_descriptor(
        args.evaluator_source_fd, "sealed evaluator source FD"
    )
    _, evaluator_record = _verify_running_evaluator_descriptor(
        evaluator_fd,
        source_custody["files"]["train/eval_dws_eos_suppressed_trace.py"],
    )
    activate_pinned_runtime_packages(source_custody["runtime"])
    verify_live_slurm_context(
        context,
        source_custody,
        scontrol_descriptor=_decimal_descriptor(
            args.scontrol_source_fd, "scontrol source FD"
        ),
        sacct_descriptor=_decimal_descriptor(args.sacct_source_fd, "sacct source FD"),
        nvidia_smi_descriptor=_decimal_descriptor(
            args.nvidia_smi_source_fd, "nvidia-smi source FD"
        ),
    )

    if args.command == "verify-private-candidate":
        if args.candidate_fd is None or args.verifier_signing_key_fd is None:
            raise ContractError("candidate verification descriptors are missing")
        candidate_fd = _decimal_descriptor(args.candidate_fd, "private candidate FD")
        candidate_payload, candidate_info = read_private_candidate_descriptor(
            candidate_fd
        )
        report = _parse_json_object_bytes(candidate_payload, "private candidate")
        if stable_json_bytes(report) != candidate_payload:
            raise ContractError("private candidate is not canonical stable JSON")
        _device_preflight(context["slurm_identity"]["gpu_binding"])
        validate_report_schema(report, context, live_custody=False)
        frozen_input_observation = observe_frozen_inputs()
        validation_runtime_observation = observe_executed_runtime(
            "independent_validator", require_cuda=True
        )
        if _file_identity(os.fstat(candidate_fd)) != _file_identity(candidate_info):
            raise ContractError("private candidate descriptor changed after validation")
        post_custody = _live_runtime_custody(args)
        if post_custody != source_custody:
            raise ContractError("runtime-source custody changed during verification")
        verifier_private_key = read_sealed_signing_key(
            _decimal_descriptor(
                args.verifier_signing_key_fd, "verifier signing-key FD"
            ),
            context["verifier_signing_key"],
        )
        receipt = build_independent_verifier_receipt(
            candidate_payload,
            candidate_info,
            report,
            validation_runtime_observation,
            frozen_input_observation,
            verifier_private_key,
        )
        sys.stdout.buffer.write(stable_json_bytes(receipt))
        return

    if (
        args.model_source_fd is None
        or args.generator_signing_key_fd is None
        or args.report_output_fd is None
    ):
        raise ContractError("generation descriptors are missing")
    model_source_bytes, model_record = read_sealed_source_descriptor(
        _decimal_descriptor(args.model_source_fd, "sealed model source FD"),
        source_custody["files"]["train/model.py"],
        "sealed model source",
    )
    generator_private_key = read_sealed_signing_key(
        _decimal_descriptor(args.generator_signing_key_fd, "generator signing-key FD"),
        context["generator_signing_key"],
    )
    started_at = datetime.now(timezone.utc).isoformat()
    validate_static_contract()
    tokenizer_bytes = read_verified_bytes(TOKENIZER_PATH, EXPECTED_SHA256["tokenizer"])
    heldout_bytes = read_verified_bytes(HELDOUT_PATH, EXPECTED_SHA256["heldout"])
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    validate_tokenizer_contract(tokenizer)
    prepared = _prepare_cases(parse_heldout_bytes(heldout_bytes), tokenizer)
    checkpoint = load_verified_checkpoint(
        CHECKPOINT_PATH, EXPECTED_SHA256["checkpoint"]
    )
    model = _load_model(checkpoint, model_source_bytes, "sealed-memfd:train/model.py")
    device_record = _device_preflight(context["slurm_identity"]["gpu_binding"])
    model = model.to(DEVICE).eval()

    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    with pinned_math_sdpa() as active_sdpa_flags:
        if any(device_record[key] != value for key, value in active_sdpa_flags.items()):
            raise ContractError("generation SDPA backend differs from preflight")
        with autocast:
            primary_raw = _run_primary_decodes(model, prepared)
            case_reports, field_requests, source_requests, all_branches = (
                _posthoc_primary_and_prepare_fields(
                    model, prepared, primary_raw, tokenizer
                )
            )
            field_raw = _run_field_decodes(field_requests)
            fresh_reencoding_raw = _run_fresh_reencoding_decodes(source_requests)
    _finish_field_scores(
        case_reports,
        field_raw,
        fresh_reencoding_raw,
        all_branches,
        tokenizer,
    )

    aggregate = aggregate_report(case_reports)
    post_execution_custody = _live_runtime_custody(args)
    if post_execution_custody != source_custody:
        raise ContractError("runtime-source custody changed during execution")
    execution = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": frozen_input_paths(),
        "verified_input_sha256": dict(EXPECTED_SHA256),
        "checkpoint_step": checkpoint["step"],
        "ordered_case_ids": [case["case_id"] for case in case_reports],
        "runtime_source_manifest": post_execution_custody,
        "source_execution": {
            "mode": SOURCE_EXECUTION_MODE,
            "python_startup_mode": PYTHON_STARTUP_MODE,
            "evaluator": evaluator_record,
            "model": model_record,
        },
        "runtime_observation": observe_executed_runtime(
            "generator_post_decode", require_cuda=True
        ),
        **device_record,
    }
    report_body = {
        "schema": OUTPUT_SCHEMA,
        "protocol": PROTOCOL_ID,
        "development_only": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "frozen_contract": frozen_contract(),
        "execution": execution,
        "aggregate": aggregate,
        "cases": case_reports,
        "adjudication": {
            "field_screen_execution": "development_go",
            "full_state_recurrence": "no_go",
            "carry_target_switch_noncompensatory_veto": aggregate[
                "carry_target_switch_global_veto"
            ],
            "compound_fresh_reencoding_screen_pass": aggregate[
                "fresh_latest_reencoding"
            ]["compound_fresh_reencoding_screen_pass"],
            "promotion_authorized": False,
        },
        "wrapper_acceptance": context,
    }
    report = attach_generator_attestation(report_body, context, generator_private_key)
    validate_report_schema(report, context, live_custody=False)
    write_anonymous_report_descriptor(
        _decimal_descriptor(args.report_output_fd, "anonymous report output FD"),
        stable_json_bytes(report),
    )
    print("sealed report generated; wrapper has not accepted it", flush=True)


if __name__ == "__main__":
    main()
