#!/usr/bin/env python3
"""Materialize deferred-seed RSP-C2 board and matched training arms.

Production entry points accept seeds only through an immutable
``rsp_c2_seed_receipt_v2``.  There is deliberately no CLI seed override.  The
board and data phases each consume an exclusive one-shot invocation path; once
that receipt exists, a failed or partial phase cannot be retried under C2.

The implementation is self-contained.  It does not import or execute any C1
generator, auditor, fixture, case, seed, or frozen board digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "R12_SOURCE_DELETED_RESIDUAL_PACKET_C2_PREREG.md"
C1_CLOSURE_PATH = ROOT / "R12_SOURCE_DELETED_RESIDUAL_PACKET_C1_CLOSURE.md"
SEED_DERIVATION_PATH = ROOT / "pipeline" / "derive_residual_packet_c2_seeds.py"
AUDITOR_PATH = ROOT / "pipeline" / "audit_residual_packet_c2.py"

# These bind the frozen research-integrity documents, not any quarantined C1
# board, row, case, seed, or data digest.
FROZEN_PREREGISTRATION_SHA256 = (
    "8f5c904664da3191a88ba85bddcb883c0df79da0416b3dd0ceb2f2a9ec8aaa5e"
)
FROZEN_C1_CLOSURE_SHA256 = (
    "bbeb081c3b2defbb866e39acb2f022477d8824bbbe651d08c7f0472745a3911e"
)

SEED_RECEIPT_SCHEMA = "rsp_c2_seed_receipt_v2"
PREREQUISITE_SCHEMA = "source_scheduled_reasoning_confirmation_pass_v1"
FREEZE_PUSH_SCHEMA = "rsp_c2_freeze_push_receipt_v1"
BEACON_VERIFICATION_SCHEMA = "rsp_c2_nist_beacon_verification_v1"
PROVENANCE_SCHEMA = "rsp_c2_frozen_provenance_v1"
INVOCATION_SCHEMA = "rsp_c2_generation_invocation_v1"
BOARD_SCHEMA = "rsp_c2_board_v1"
BOARD_AUDIT_SCHEMA = "rsp_c2_board_admission_audit_v1"
ROW_SCHEMA = "rsp_c2_sft_row_v1"
MANIFEST_SCHEMA = "rsp_c2_generation_manifest_v1"
SEED_SCHEME = "sha256_domain_separated_nist_receipts_v2"
BASE_DOMAIN = b"SHOHIN-RSP-C2-SEED-BASE-v2\0"
SEED_DOMAIN = b"SHOHIN-RSP-C2-SEED-v2\0"
SEED_LABELS = ("board", "training", "observation", "sham", "fit-a", "fit-b")

PRODUCTION_PROFILE = "production_rsp_c2_v1"
TOY_PREFIX = "TOY_ONLY_NEVER_PRODUCTION_"
PACK_LENGTH = 128
PER_STRATUM = 64
STRATUM_ORDER = ("renderer_ood", "value_ood", "operation_order_ood", "length_ood")
TRAIN_LENGTH_COUNTS = {2: 1024, 3: 2048, 4: 1024}
OPERATION_TYPES = ("add", "multiply", "subtract")
HELD_OUT_BIGRAMS = (("add", "multiply"), ("multiply", "subtract"))
TRAIN_TEMPLATE_IDS = ("c2_train_a", "c2_train_b", "c2_train_c", "c2_train_d")
RESERVED_TEMPLATE_ID = "c2_reserved_ledger"

HEX_64_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
HEX_128_RE = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
UTC_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(?P<millis>\d{3})Z\Z",
    re.ASCII,
)
REMOTE_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*\.git\Z",
    re.ASCII,
)
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}\Z", re.ASCII)
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,126}[A-Za-z0-9]\Z", re.ASCII)
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]", re.ASCII)
NGRAM_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+", re.ASCII)
INTEGER_RE = re.compile(r"(?<![A-Za-z0-9])[0-9]+(?![A-Za-z0-9])", re.ASCII)

PREREQUISITE_KEYS = {
    "advance_to_internalization",
    "all_locked_gates_pass",
    "confirmation_contract_sha256",
    "confirmation_result_sha256",
    "independent_recomputation_complete",
    "independent_score_receipt_sha256",
    "independent_scorer_sha256",
    "primary_score_receipt_sha256",
    "primary_scorer_sha256",
    "result_immutable",
    "schema",
    "scorers_agree",
}
PROVENANCE_KEYS = {
    "auditor_sha256",
    "c1_closure_sha256",
    "freeze_commit",
    "freeze_push_receipt_sha256",
    "generator_sha256",
    "prerequisite_receipt_sha256",
    "preregistration_sha256",
    "runtime_receipt_sha256",
    "schema",
    "seed_derivation_sha256",
    "tokenizer_sha256",
}


@dataclass(frozen=True)
class Geometry:
    profile: str
    per_stratum: int
    training_length_counts: Mapping[int, int]
    normal_initial: tuple[int, int] = (20, 98)
    normal_add_sub_operand: tuple[int, int] = (3, 24)
    normal_multiply_operand: tuple[int, int] = (2, 6)
    ood_initial: tuple[int, int] = (300, 599)
    ood_add_sub_operand: tuple[int, int] = (40, 90)
    ood_multiply_operand: tuple[int, int] = (8, 13)


PRODUCTION_GEOMETRY = Geometry(
    profile=PRODUCTION_PROFILE,
    per_stratum=PER_STRATUM,
    training_length_counts=TRAIN_LENGTH_COUNTS,
)


@dataclass(frozen=True)
class Custody:
    prerequisite: Mapping[str, Any]
    prerequisite_sha256: str
    provenance: Mapping[str, Any]
    provenance_sha256: str
    seed_receipt: Mapping[str, Any]
    seed_receipt_sha256: str
    seed_integers: Mapping[str, int]
    seed_digests: Mapping[str, str]
    tokenizer_bytes: bytes
    tokenizer_sha256: str
    runtime_receipt_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _read_regular_bytes(path: str | Path, label: str, *, read_only: bool) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if read_only and before.st_mode & 0o222:
            raise PermissionError(f"{label} must be read-only")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_immutable_json(
    path: str | Path, label: str, *, canonical: bool
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label, read_only=True)
    payload = parse_json_bytes(raw, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    if canonical and raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload, raw


def sha256_file(path: str | Path, label: str = "file") -> str:
    return sha256_bytes(_read_regular_bytes(path, label, read_only=False))


def _lower_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical lowercase SHA-256")
    return value


def _git_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git OID")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _timestamp_ms(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    match = UTC_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} must use YYYY-MM-DDTHH:MM:SS.mmmZ")
    try:
        parsed = datetime.strptime(match.group("date"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{label} is not a valid UTC timestamp") from error
    result = int(parsed.timestamp()) * 1000 + int(match.group("millis"))
    rendered = datetime.fromtimestamp(result // 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if value != f"{rendered}.{result % 1000:03d}Z":
        raise ValueError(f"{label} is not canonical")
    return result


def validate_prerequisite_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != PREREQUISITE_KEYS:
        missing = sorted(PREREQUISITE_KEYS - set(payload))
        extra = sorted(set(payload) - PREREQUISITE_KEYS)
        raise ValueError(
            f"prerequisite receipt keys differ: missing={missing}, extra={extra}"
        )
    if payload["schema"] != PREREQUISITE_SCHEMA:
        raise ValueError("prerequisite receipt schema mismatch")
    for key in (
        "advance_to_internalization",
        "all_locked_gates_pass",
        "independent_recomputation_complete",
        "result_immutable",
        "scorers_agree",
    ):
        if payload[key] is not True:
            raise ValueError(f"prerequisite receipt requires {key}=true")
    normalized = dict(payload)
    for key in (
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
    ):
        normalized[key] = _lower_hash(payload[key], key)
    if normalized["primary_scorer_sha256"] == normalized["independent_scorer_sha256"]:
        raise ValueError("primary and independent scorer implementations must differ")
    if (
        normalized["primary_score_receipt_sha256"]
        == normalized["independent_score_receipt_sha256"]
    ):
        raise ValueError("primary and independent score receipts must differ")
    return normalized


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical nonempty identifier")
    return value


def _field(value: str | int) -> bytes:
    return str(value).encode("ascii") + b"\0"


def validate_seed_receipt(
    payload: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    top_keys = {
        "base_commitment_sha256",
        "beacon",
        "freeze",
        "prerequisite",
        "schema",
        "seed_labels",
        "seed_scheme",
        "seeds",
    }
    if set(payload) != top_keys:
        raise ValueError("seed receipt top-level schema mismatch")
    if (
        payload["schema"] != SEED_RECEIPT_SCHEMA
        or payload["seed_scheme"] != SEED_SCHEME
    ):
        raise ValueError("seed receipt schema or scheme mismatch")
    if payload["seed_labels"] != list(SEED_LABELS):
        raise ValueError("seed receipt labels or label order mismatch")
    freeze = payload["freeze"]
    freeze_keys = {
        "branch",
        "commit_oid",
        "observed_at",
        "observer_id",
        "observer_implementation_sha256",
        "pushed_at",
        "raw_sha256",
        "ref",
        "remote_ref_evidence_sha256",
        "remote_ref_oid",
        "remote_url",
    }
    if not isinstance(freeze, dict) or set(freeze) != freeze_keys:
        raise ValueError("seed receipt freeze schema mismatch")
    commit_oid = _git_oid(freeze["commit_oid"], "seed receipt freeze commit")
    remote_ref_oid = _git_oid(
        freeze["remote_ref_oid"], "seed receipt remote ref commit"
    )
    if commit_oid != remote_ref_oid:
        raise ValueError("seed receipt freeze commit and remote ref differ")
    branch = freeze["branch"]
    if (
        not isinstance(branch, str)
        or BRANCH_RE.fullmatch(branch) is None
        or ".." in branch
        or "@{" in branch
        or "//" in branch
        or branch.endswith((".", "/", ".lock"))
    ):
        raise ValueError("seed receipt freeze branch is not canonical")
    if freeze["ref"] != f"refs/heads/{branch}":
        raise ValueError("seed receipt freeze ref does not match branch")
    remote_url = freeze["remote_url"]
    if not isinstance(remote_url, str) or REMOTE_URL_RE.fullmatch(remote_url) is None:
        raise ValueError("seed receipt freeze remote URL is not canonical")
    pushed_at_ms = _timestamp_ms(freeze["pushed_at"], "seed receipt pushed_at")
    observed_at_ms = _timestamp_ms(freeze["observed_at"], "seed receipt observed_at")
    if not 0 <= observed_at_ms - pushed_at_ms <= 300_000:
        raise ValueError("seed receipt freeze observation lag is invalid")
    _identifier(freeze["observer_id"], "seed receipt freeze observer")
    for key in (
        "observer_implementation_sha256",
        "raw_sha256",
        "remote_ref_evidence_sha256",
    ):
        _lower_hash(freeze[key], f"seed receipt freeze {key}")
    reconstructed_freeze = {
        **{key: value for key, value in freeze.items() if key != "raw_sha256"},
        "schema": FREEZE_PUSH_SCHEMA,
    }
    if sha256_bytes(canonical_json_bytes(reconstructed_freeze)) != freeze["raw_sha256"]:
        raise ValueError("seed receipt freeze raw SHA-256 does not replay")

    prerequisite = payload["prerequisite"]
    prerequisite_keys = {
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
        "raw_sha256",
    }
    if not isinstance(prerequisite, dict) or set(prerequisite) != prerequisite_keys:
        raise ValueError("seed receipt prerequisite schema mismatch")
    for key in prerequisite_keys:
        _lower_hash(prerequisite[key], f"seed receipt prerequisite {key}")
    if (
        prerequisite["primary_scorer_sha256"]
        == prerequisite["independent_scorer_sha256"]
    ):
        raise ValueError("seed receipt scorer implementations are not independent")
    if (
        prerequisite["primary_score_receipt_sha256"]
        == prerequisite["independent_score_receipt_sha256"]
    ):
        raise ValueError("seed receipt score receipts are not independent")
    reconstructed_prerequisite = {
        "advance_to_internalization": True,
        "all_locked_gates_pass": True,
        "confirmation_contract_sha256": prerequisite["confirmation_contract_sha256"],
        "confirmation_result_sha256": prerequisite["confirmation_result_sha256"],
        "independent_recomputation_complete": True,
        "independent_score_receipt_sha256": prerequisite[
            "independent_score_receipt_sha256"
        ],
        "independent_scorer_sha256": prerequisite["independent_scorer_sha256"],
        "primary_score_receipt_sha256": prerequisite["primary_score_receipt_sha256"],
        "primary_scorer_sha256": prerequisite["primary_scorer_sha256"],
        "result_immutable": True,
        "schema": PREREQUISITE_SCHEMA,
        "scorers_agree": True,
    }
    if (
        sha256_bytes(canonical_json_bytes(reconstructed_prerequisite))
        != prerequisite["raw_sha256"]
    ):
        raise ValueError("seed receipt prerequisite raw SHA-256 does not replay")

    beacon = payload["beacon"]
    beacon_keys = {
        "certificate_id",
        "chain_index",
        "first_pulse_target_ms",
        "output_value",
        "output_value_sha256",
        "period_ms",
        "pulse_index",
        "raw_sha256",
        "signature_value_sha256",
        "status_code",
        "time_stamp",
        "time_stamp_ms",
        "verification",
    }
    if not isinstance(beacon, dict) or set(beacon) != beacon_keys:
        raise ValueError("seed receipt beacon schema mismatch")
    for key in (
        "output_value_sha256",
        "raw_sha256",
        "signature_value_sha256",
    ):
        _lower_hash(beacon[key], f"seed receipt beacon {key}")
    output = beacon["output_value"]
    if not isinstance(output, str) or HEX_128_RE.fullmatch(output) is None:
        raise ValueError(
            "seed receipt beacon output must be 512-bit lowercase hexadecimal"
        )
    if sha256_bytes(bytes.fromhex(output)) != beacon["output_value_sha256"]:
        raise ValueError("seed receipt beacon output hash mismatch")
    if beacon["period_ms"] != 60_000 or isinstance(beacon["period_ms"], bool):
        raise ValueError("seed receipt beacon period mismatch")
    if beacon["status_code"] != 0 or isinstance(beacon["status_code"], bool):
        raise ValueError("seed receipt beacon status mismatch")
    _positive_integer(beacon["chain_index"], "seed receipt beacon chain index")
    _positive_integer(beacon["pulse_index"], "seed receipt beacon pulse index")
    timestamp_ms = _timestamp_ms(beacon["time_stamp"], "seed receipt beacon timestamp")
    if beacon["time_stamp_ms"] != timestamp_ms or isinstance(
        beacon["time_stamp_ms"], bool
    ):
        raise ValueError("seed receipt beacon timestamp integer mismatch")
    target = pushed_at_ms + 3_600_000
    if beacon["first_pulse_target_ms"] != target:
        raise ValueError("seed receipt beacon first-pulse target mismatch")
    if not 0 <= timestamp_ms - target < 60_000:
        raise ValueError("seed receipt beacon is outside the first eligible slot")
    if not isinstance(beacon["certificate_id"], str) or not beacon["certificate_id"]:
        raise ValueError("seed receipt beacon certificate is missing")

    verification = beacon["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "certificate_sha256",
        "raw_sha256",
        "validators",
        "validators_agree",
    }:
        raise ValueError("seed receipt beacon verification schema mismatch")
    _lower_hash(
        verification["certificate_sha256"],
        "seed receipt beacon certificate SHA-256",
    )
    _lower_hash(
        verification["raw_sha256"],
        "seed receipt beacon verification receipt SHA-256",
    )
    if verification["validators_agree"] is not True:
        raise ValueError("seed receipt beacon validators do not agree")
    validators = verification["validators"]
    validator_keys = {
        "certificate_valid",
        "chain_valid",
        "evidence_sha256",
        "implementation_sha256",
        "raw_beacon_sha256",
        "signature_valid",
        "validated_at",
        "validator_id",
    }
    if not isinstance(validators, list) or len(validators) != 2:
        raise ValueError("seed receipt requires exactly two beacon validators")
    normalized_validators = []
    for index, validator in enumerate(validators):
        if not isinstance(validator, dict) or set(validator) != validator_keys:
            raise ValueError("seed receipt beacon validator schema mismatch")
        for gate in ("certificate_valid", "chain_valid", "signature_valid"):
            if validator[gate] is not True:
                raise ValueError(f"seed receipt beacon validator {index} failed {gate}")
        validator_id = _identifier(
            validator["validator_id"], f"seed receipt beacon validator {index} id"
        )
        implementation = _lower_hash(
            validator["implementation_sha256"],
            f"seed receipt beacon validator {index} implementation",
        )
        evidence = _lower_hash(
            validator["evidence_sha256"],
            f"seed receipt beacon validator {index} evidence",
        )
        raw_beacon = _lower_hash(
            validator["raw_beacon_sha256"],
            f"seed receipt beacon validator {index} raw beacon",
        )
        if raw_beacon != beacon["raw_sha256"]:
            raise ValueError("seed receipt beacon validator raw hash mismatch")
        _timestamp_ms(
            validator["validated_at"],
            f"seed receipt beacon validator {index} validated_at",
        )
        normalized_validators.append((validator_id, implementation, evidence))
    for field_index, label in (
        (0, "identifier"),
        (1, "implementation"),
        (2, "evidence"),
    ):
        if (
            normalized_validators[0][field_index]
            == normalized_validators[1][field_index]
        ):
            raise ValueError(f"seed receipt beacon validator {label}s must differ")
    reconstructed_verification = {
        "beacon_raw_sha256": beacon["raw_sha256"],
        "certificate_id": beacon["certificate_id"],
        "certificate_sha256": verification["certificate_sha256"],
        "chain_index": beacon["chain_index"],
        "output_value_sha256": beacon["output_value_sha256"],
        "period_ms": beacon["period_ms"],
        "pulse_index": beacon["pulse_index"],
        "schema": BEACON_VERIFICATION_SCHEMA,
        "signature_value_sha256": beacon["signature_value_sha256"],
        "status_code": beacon["status_code"],
        "time_stamp": beacon["time_stamp"],
        "time_stamp_ms": beacon["time_stamp_ms"],
        "validators": validators,
        "validators_agree": True,
    }
    if (
        sha256_bytes(canonical_json_bytes(reconstructed_verification))
        != verification["raw_sha256"]
    ):
        raise ValueError("seed receipt beacon verification raw SHA-256 does not replay")

    base_preimage = b"".join(
        (
            BASE_DOMAIN,
            _field(commit_oid),
            _field(remote_url),
            _field(freeze["ref"]),
            _field(freeze["pushed_at"]),
            _field(freeze["observed_at"]),
            _field(freeze["raw_sha256"]),
            _field(prerequisite["raw_sha256"]),
            _field(prerequisite["confirmation_result_sha256"]),
            _field(beacon["raw_sha256"]),
            _field(beacon["chain_index"]),
            _field(beacon["pulse_index"]),
            _field(beacon["time_stamp_ms"]),
            _field(beacon["output_value"]),
            _field(verification["raw_sha256"]),
        )
    )
    base_digest = hashlib.sha256(base_preimage).digest()
    base_hex = _lower_hash(payload["base_commitment_sha256"], "base commitment")
    if base_hex != base_digest.hex():
        raise ValueError("seed receipt base commitment does not replay")

    seeds = payload["seeds"]
    if not isinstance(seeds, dict) or set(seeds) != set(SEED_LABELS):
        raise ValueError("seed receipt seed map mismatch")
    integers: dict[str, int] = {}
    digests: dict[str, str] = {}
    for label in SEED_LABELS:
        item = seeds[label]
        if not isinstance(item, dict) or set(item) != {"integer_decimal", "sha256"}:
            raise ValueError(f"seed receipt {label} entry schema mismatch")
        digest = hashlib.sha256(
            SEED_DOMAIN + base_digest + b"\0" + label.encode("ascii")
        ).digest()
        digest_hex = _lower_hash(item["sha256"], f"{label} seed digest")
        decimal = item["integer_decimal"]
        if not isinstance(decimal, str) or DECIMAL_RE.fullmatch(decimal) is None:
            raise ValueError(f"{label} seed integer is not canonical decimal")
        integer = int(decimal)
        if digest_hex != digest.hex() or integer != int.from_bytes(digest, "big"):
            raise ValueError(f"{label} seed does not derive from the base commitment")
        integers[label] = integer
        digests[label] = digest_hex
    if len(set(digests.values())) != len(SEED_LABELS):
        raise ValueError("seed receipt contains duplicate derived seeds")
    return integers, digests


def validate_provenance_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != PROVENANCE_KEYS:
        missing = sorted(PROVENANCE_KEYS - set(payload))
        extra = sorted(set(payload) - PROVENANCE_KEYS)
        raise ValueError(f"provenance keys differ: missing={missing}, extra={extra}")
    if payload["schema"] != PROVENANCE_SCHEMA:
        raise ValueError("provenance receipt schema mismatch")
    normalized = dict(payload)
    normalized["freeze_commit"] = _git_oid(payload["freeze_commit"], "freeze commit")
    for key in PROVENANCE_KEYS - {"schema", "freeze_commit"}:
        normalized[key] = _lower_hash(payload[key], f"provenance {key}")
    if normalized["preregistration_sha256"] != FROZEN_PREREGISTRATION_SHA256:
        raise ValueError("provenance does not bind the frozen C2 preregistration")
    if normalized["c1_closure_sha256"] != FROZEN_C1_CLOSURE_SHA256:
        raise ValueError("provenance does not bind the C1 closure document")
    return normalized


def load_production_custody(
    *,
    prerequisite_path: str | Path,
    prerequisite_sha256: str,
    seed_receipt_path: str | Path,
    seed_receipt_sha256: str,
    provenance_path: str | Path,
    provenance_sha256: str,
    tokenizer_path: str | Path,
    runtime_receipt_path: str | Path,
) -> Custody:
    prerequisite_sha256 = _lower_hash(
        prerequisite_sha256, "expected prerequisite SHA-256"
    )
    seed_receipt_sha256 = _lower_hash(
        seed_receipt_sha256, "expected seed receipt SHA-256"
    )
    provenance_sha256 = _lower_hash(provenance_sha256, "expected provenance SHA-256")

    prerequisite_payload, prerequisite_raw = read_immutable_json(
        prerequisite_path, "prerequisite pass receipt", canonical=True
    )
    if sha256_bytes(prerequisite_raw) != prerequisite_sha256:
        raise ValueError("prerequisite receipt SHA-256 mismatch")
    prerequisite = validate_prerequisite_receipt(prerequisite_payload)

    seed_payload, seed_raw = read_immutable_json(
        seed_receipt_path, "C2 seed receipt", canonical=True
    )
    if sha256_bytes(seed_raw) != seed_receipt_sha256:
        raise ValueError("seed receipt SHA-256 mismatch")
    seed_integers, seed_digests = validate_seed_receipt(seed_payload)

    provenance_payload, provenance_raw = read_immutable_json(
        provenance_path, "C2 frozen provenance receipt", canonical=True
    )
    if sha256_bytes(provenance_raw) != provenance_sha256:
        raise ValueError("provenance receipt SHA-256 mismatch")
    provenance = validate_provenance_receipt(provenance_payload)

    embedded_prerequisite = seed_payload["prerequisite"]
    if embedded_prerequisite["raw_sha256"] != prerequisite_sha256:
        raise ValueError("seed receipt does not bind the supplied prerequisite receipt")
    for key in (
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
    ):
        if embedded_prerequisite[key] != prerequisite[key]:
            raise ValueError(f"seed receipt prerequisite field mismatch: {key}")
    if provenance["prerequisite_receipt_sha256"] != prerequisite_sha256:
        raise ValueError("provenance prerequisite receipt SHA-256 mismatch")
    if seed_payload["freeze"]["commit_oid"] != provenance["freeze_commit"]:
        raise ValueError("seed receipt freeze commit does not match provenance")
    if seed_payload["freeze"]["raw_sha256"] != provenance["freeze_push_receipt_sha256"]:
        raise ValueError("seed receipt freeze push receipt does not match provenance")

    tokenizer_raw = _read_regular_bytes(tokenizer_path, "tokenizer", read_only=True)
    source_hashes = {
        "preregistration_sha256": sha256_file(
            PREREGISTRATION_PATH, "C2 preregistration"
        ),
        "c1_closure_sha256": sha256_file(C1_CLOSURE_PATH, "C1 closure document"),
        "seed_derivation_sha256": sha256_file(
            SEED_DERIVATION_PATH, "C2 seed derivation"
        ),
        "generator_sha256": sha256_file(__file__, "C2 generator"),
        "auditor_sha256": sha256_file(AUDITOR_PATH, "C2 auditor"),
        "tokenizer_sha256": sha256_bytes(tokenizer_raw),
    }
    for key, observed in source_hashes.items():
        if provenance[key] != observed:
            raise ValueError(f"frozen provenance mismatch for {key}")
    runtime_raw = _read_regular_bytes(
        runtime_receipt_path, "runtime identity receipt", read_only=True
    )
    runtime_sha256 = sha256_bytes(runtime_raw)
    if provenance["runtime_receipt_sha256"] != runtime_sha256:
        raise ValueError("runtime receipt SHA-256 does not match provenance")

    return Custody(
        prerequisite=prerequisite,
        prerequisite_sha256=prerequisite_sha256,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
        seed_receipt=seed_payload,
        seed_receipt_sha256=seed_receipt_sha256,
        seed_integers=seed_integers,
        seed_digests=seed_digests,
        tokenizer_bytes=tokenizer_raw,
        tokenizer_sha256=source_hashes["tokenizer_sha256"],
        runtime_receipt_sha256=runtime_sha256,
    )


def _operation_phrase(operation: Sequence[Any]) -> str:
    kind, operand = str(operation[0]), int(operation[1])
    if kind == "add":
        return f"add {operand}"
    if kind == "multiply":
        return f"multiply by {operand}"
    if kind == "subtract":
        return f"subtract {operand}"
    raise ValueError(f"unknown operation {kind!r}")


def render_source(
    initial_state: int, operations: Sequence[Sequence[Any]], template_id: str
) -> str:
    phrases = [_operation_phrase(operation) for operation in operations]
    if template_id == "c2_train_a":
        return (
            f"Set register R to {initial_state}. Then "
            + ", then ".join(phrases)
            + ". Return R."
        )
    if template_id == "c2_train_b":
        numbered = " ".join(
            f"{index + 1}) {phrase}." for index, phrase in enumerate(phrases)
        )
        return f"Initial integer: {initial_state}. Execute in order. {numbered} Report the result."
    if template_id == "c2_train_c":
        return (
            f"Start={initial_state}; program="
            + " -> ".join(phrases)
            + "; emit final integer."
        )
    if template_id == "c2_train_d":
        return (
            f"A counter begins at {initial_state}. Apply this ordered instruction list: "
            + " | ".join(phrases)
            + ". Give the counter value."
        )
    if template_id == RESERVED_TEMPLATE_ID:
        codes = {"add": "A", "multiply": "M", "subtract": "S"}
        encoded = "/".join(
            f"{codes[str(kind)]}{int(operand)}" for kind, operand in operations
        )
        return f"C2-LDG::I={initial_state}::OPS={encoded}::HALT=VALUE"
    raise ValueError(f"unknown C2 template {template_id!r}")


def render_packet(state: int, operations: Sequence[Sequence[Any]]) -> str:
    codes = {"add": "ADD", "multiply": "MUL", "subtract": "SUB"}
    plan = ",".join(
        f"{codes[str(kind)]}:{int(operand)}" for kind, operand in operations
    )
    return f"<C2P|S={int(state)}|R={plan}>"


def render_answer(value: int) -> str:
    return f"<C2A|V={int(value)}>"


def compiler_prompt(source: str) -> str:
    return (
        "Compile this arithmetic source into the exact C2 packet grammar.\n"
        f"Source:\n{source}\nPacket:\n"
    )


def updater_prompt(packet: str, observed_result: int) -> str:
    return (
        "Advance exactly one C2 packet step using the supplied observation.\n"
        f"Packet:\n{packet}\nObservation: {int(observed_result)}\nNext:\n"
    )


def apply_operation(state: int, operation: Sequence[Any]) -> int:
    kind, operand = str(operation[0]), int(operation[1])
    if kind == "add":
        return state + operand
    if kind == "multiply":
        return state * operand
    if kind == "subtract":
        return state - operand
    raise ValueError(f"unknown operation {kind!r}")


def trajectory(
    initial_state: int, operations: Sequence[Sequence[Any]]
) -> tuple[int, ...]:
    states = [int(initial_state)]
    for operation in operations:
        states.append(apply_operation(states[-1], operation))
    return tuple(states)


def semantic_signature(
    initial_state: int, operations: Sequence[Sequence[Any]]
) -> tuple[Any, ...]:
    return (int(initial_state),) + tuple(
        (str(kind), int(operand)) for kind, operand in operations
    )


def operation_types(operations: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(operation[0]) for operation in operations)


def digit_widths(
    initial_state: int, operations: Sequence[Sequence[Any]]
) -> tuple[int, ...]:
    return (len(str(initial_state)),) + tuple(
        len(str(int(operation[1]))) for operation in operations
    )


def integer_occurrences(text: str) -> tuple[int, ...]:
    return tuple(int(match.group(0)) for match in INTEGER_RE.finditer(text))


def source_ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    tokens = tuple(NGRAM_TOKEN_RE.findall(text.lower()))
    return {tokens[index : index + width] for index in range(len(tokens) - width + 1)}


def _sequence_pool(
    length: int, required_holdout: tuple[str, str] | None = None
) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = [()]
    for _ in range(length):
        sequences = [
            prefix + (kind,) for prefix in sequences for kind in OPERATION_TYPES
        ]
    accepted = []
    for sequence in sequences:
        held = [
            pair for pair in zip(sequence, sequence[1:]) if pair in HELD_OUT_BIGRAMS
        ]
        if required_holdout is None and not held:
            accepted.append(sequence)
        elif required_holdout is not None and held == [required_holdout]:
            accepted.append(sequence)
    if not accepted:
        raise RuntimeError("no operation sequence satisfies C2 geometry")
    return tuple(accepted)


def _operand(
    rng: random.Random,
    kind: str,
    geometry: Geometry,
    *,
    value_ood: bool,
    width: int | None = None,
) -> int:
    if value_ood:
        low, high = (
            geometry.ood_multiply_operand
            if kind == "multiply"
            else geometry.ood_add_sub_operand
        )
    else:
        low, high = (
            geometry.normal_multiply_operand
            if kind == "multiply"
            else geometry.normal_add_sub_operand
        )
    if width is not None:
        low = max(low, 10 ** (width - 1) if width > 1 else 0)
        high = min(high, 10**width - 1)
        if low > high:
            raise ValueError(f"no {kind} operand with width {width}")
    return rng.randint(low, high)


def _sample_program(
    rng: random.Random,
    types: Sequence[str],
    geometry: Geometry,
    *,
    value_ood: bool = False,
    widths: Sequence[int] | None = None,
) -> tuple[int, list[list[Any]]]:
    if widths is None:
        low, high = geometry.ood_initial if value_ood else geometry.normal_initial
        initial = rng.randint(low, high)
        operations = [
            [kind, _operand(rng, kind, geometry, value_ood=value_ood)] for kind in types
        ]
    else:
        initial_low = max(geometry.normal_initial[0], 10 ** (int(widths[0]) - 1))
        initial_high = min(geometry.normal_initial[1], 10 ** int(widths[0]) - 1)
        if initial_low > initial_high:
            raise ValueError(f"no initial state with width {widths[0]}")
        initial = rng.randint(initial_low, initial_high)
        operations = [
            [kind, _operand(rng, kind, geometry, value_ood=False, width=int(width))]
            for kind, width in zip(types, widths[1:])
        ]
    return initial, operations


def geometry_payload(geometry: Geometry) -> dict[str, Any]:
    return {
        "held_out_bigrams": [list(pair) for pair in HELD_OUT_BIGRAMS],
        "normal_ranges": {
            "add_sub_operand": list(geometry.normal_add_sub_operand),
            "initial_state": list(geometry.normal_initial),
            "multiply_operand": list(geometry.normal_multiply_operand),
        },
        "ood_ranges": {
            "add_sub_operand": list(geometry.ood_add_sub_operand),
            "initial_state": list(geometry.ood_initial),
            "multiply_operand": list(geometry.ood_multiply_operand),
        },
        "operation_types": list(OPERATION_TYPES),
        "per_stratum": geometry.per_stratum,
        "reserved_template_id": RESERVED_TEMPLATE_ID,
        "stratum_lengths": {
            "length_ood": [5],
            "operation_order_ood": [3, 4],
            "renderer_ood": [3],
            "value_ood": [3],
        },
        "stratum_order": list(STRATUM_ORDER),
        "training_length_counts": {
            str(key): value
            for key, value in sorted(geometry.training_length_counts.items())
        },
        "training_template_ids": list(TRAIN_TEMPLATE_IDS),
    }


def build_board_rows(
    *, seed: int, geometry: Geometry, id_prefix: str
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen_programs: set[tuple[Any, ...]] = set()
    seen_sources: set[str] = set()
    seen_packets: set[str] = set()
    seen_trajectories: set[tuple[int, ...]] = set()
    seen_answers: set[int] = set()
    for stratum in STRATUM_ORDER:
        for local_index in range(geometry.per_stratum):
            for _attempt in range(100_000):
                if stratum == "renderer_ood":
                    types = rng.choice(_sequence_pool(3))
                    template = RESERVED_TEMPLATE_ID
                    initial, operations = _sample_program(rng, types, geometry)
                elif stratum == "value_ood":
                    types = rng.choice(_sequence_pool(3))
                    template = TRAIN_TEMPLATE_IDS[local_index % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = _sample_program(
                        rng, types, geometry, value_ood=True
                    )
                elif stratum == "operation_order_ood":
                    half = geometry.per_stratum // 2
                    held = HELD_OUT_BIGRAMS[0 if local_index < half else 1]
                    within_half = (
                        local_index if local_index < half else local_index - half
                    )
                    length = 3 if within_half % 2 == 0 else 4
                    types = rng.choice(_sequence_pool(length, held))
                    template = TRAIN_TEMPLATE_IDS[local_index % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = _sample_program(rng, types, geometry)
                else:
                    types = rng.choice(_sequence_pool(5))
                    template = TRAIN_TEMPLATE_IDS[local_index % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = _sample_program(rng, types, geometry)
                states = trajectory(initial, operations)
                if min(states) <= 0:
                    continue
                source = render_source(initial, operations, template)
                packet = render_packet(initial, operations)
                signature = semantic_signature(initial, operations)
                if (
                    signature in seen_programs
                    or source in seen_sources
                    or packet in seen_packets
                    or states in seen_trajectories
                    or states[-1] in seen_answers
                ):
                    continue
                row = {
                    "answer": states[-1],
                    "id": f"{id_prefix}{stratum}_{local_index:03d}",
                    "initial_state": initial,
                    "operations": [list(operation) for operation in operations],
                    "packet": packet,
                    "source": source,
                    "stratum": stratum,
                    "template_id": template,
                    "trajectory": list(states),
                }
                rows.append(row)
                seen_programs.add(signature)
                seen_sources.add(source)
                seen_packets.add(packet)
                seen_trajectories.add(states)
                seen_answers.add(states[-1])
                break
            else:
                raise RuntimeError(
                    f"could not generate C2 board row {stratum}/{local_index}"
                )
    return rows


def production_board_payload(custody: Custody) -> dict[str, Any]:
    rows = build_board_rows(
        seed=custody.seed_integers["board"],
        geometry=PRODUCTION_GEOMETRY,
        id_prefix="rsp_c2_",
    )
    geometry = geometry_payload(PRODUCTION_GEOMETRY)
    return {
        "case_count": len(rows),
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "prerequisite_receipt_sha256": custody.prerequisite_sha256,
            "provenance_receipt_sha256": custody.provenance_sha256,
            "seed_receipt_sha256": custody.seed_receipt_sha256,
        },
        "geometry": geometry,
        "geometry_sha256": sha256_bytes(canonical_json_bytes(geometry)),
        "per_stratum": PER_STRATUM,
        "profile": PRODUCTION_PROFILE,
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "schema": BOARD_SCHEMA,
        "seed_commitment_sha256": custody.seed_digests["board"],
        "stratum_order": list(STRATUM_ORDER),
    }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    ids = getattr(encoded, "ids", None)
    if not isinstance(ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in ids
    ):
        raise ValueError("tokenizer encode() must return a nonnegative integer id list")
    return list(ids)


def _program_record(
    *,
    initial: int,
    operations: Sequence[Sequence[Any]],
    template: str,
    tokenizer: Any,
    pair_id: int,
) -> dict[str, Any]:
    states = trajectory(initial, operations)
    source = render_source(initial, operations, template)
    packet = render_packet(initial, operations)
    return {
        "final_answer": states[-1],
        "initial_state": initial,
        "operations": [list(operation) for operation in operations],
        "packet": packet,
        "packet_token_count": len(_token_ids(tokenizer, packet)),
        "pair_id": pair_id,
        "source": source,
        "template_id": template,
        "trajectory": list(states),
    }


def _program_allowed(
    program: Mapping[str, Any],
    *,
    board_answers: set[int],
    board_programs: set[tuple[Any, ...]],
    board_sources: set[str],
    board_packets: set[str],
    board_trajectories: set[tuple[int, ...]],
    board_grams: set[tuple[str, ...]],
    used_programs: set[tuple[Any, ...]],
    used_sources: set[str],
    used_packets: set[str],
    used_trajectories: set[tuple[int, ...]],
) -> bool:
    signature = semantic_signature(program["initial_state"], program["operations"])
    states = tuple(int(value) for value in program["trajectory"])
    packet_fields = {int(program["initial_state"])} | {
        int(operation[1]) for operation in program["operations"]
    }
    return (
        min(states) > 0
        and int(program["final_answer"]) not in packet_fields
        and not (packet_fields & board_answers)
        and signature not in board_programs
        and str(program["source"]) not in board_sources
        and str(program["packet"]) not in board_packets
        and states not in board_trajectories
        and int(program["final_answer"]) not in board_answers
        and not (source_ngrams(str(program["source"])) & board_grams)
        and signature not in used_programs
        and str(program["source"]) not in used_sources
        and str(program["packet"]) not in used_packets
        and states not in used_trajectories
    )


def build_training_programs(
    *,
    board_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    seed: int,
    geometry: Geometry,
    id_prefix: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    board_answers = {int(row["answer"]) for row in board_rows}
    board_programs = {
        semantic_signature(row["initial_state"], row["operations"])
        for row in board_rows
    }
    board_sources = {str(row["source"]) for row in board_rows}
    board_packets = {str(row["packet"]) for row in board_rows}
    board_trajectories = {
        tuple(int(value) for value in row["trajectory"]) for row in board_rows
    }
    board_grams: set[tuple[str, ...]] = set()
    for source in board_sources:
        board_grams.update(source_ngrams(source))
    used_programs: set[tuple[Any, ...]] = set()
    used_sources: set[str] = set()
    used_packets: set[str] = set()
    used_trajectories: set[tuple[int, ...]] = set()
    programs: list[dict[str, Any]] = []
    pair_id = 0
    for length, count in sorted(geometry.training_length_counts.items()):
        if count <= 0 or count % 2:
            raise ValueError("every C2 training length count must be positive and even")
        for local_pair in range(count // 2):
            for _base_attempt in range(20_000):
                types = rng.choice(_sequence_pool(length))
                template = TRAIN_TEMPLATE_IDS[local_pair % len(TRAIN_TEMPLATE_IDS)]
                initial, operations = _sample_program(rng, types, geometry)
                first = _program_record(
                    initial=initial,
                    operations=operations,
                    template=template,
                    tokenizer=tokenizer,
                    pair_id=pair_id,
                )
                if not _program_allowed(
                    first,
                    board_answers=board_answers,
                    board_programs=board_programs,
                    board_sources=board_sources,
                    board_packets=board_packets,
                    board_trajectories=board_trajectories,
                    board_grams=board_grams,
                    used_programs=used_programs,
                    used_sources=used_sources,
                    used_packets=used_packets,
                    used_trajectories=used_trajectories,
                ):
                    continue
                widths = digit_widths(first["initial_state"], first["operations"])
                first_signature = semantic_signature(
                    first["initial_state"], first["operations"]
                )
                for _partner_attempt in range(20_000):
                    second_initial, second_operations = _sample_program(
                        rng, types, geometry, widths=widths
                    )
                    second = _program_record(
                        initial=second_initial,
                        operations=second_operations,
                        template=template,
                        tokenizer=tokenizer,
                        pair_id=pair_id,
                    )
                    if (
                        second["packet_token_count"] != first["packet_token_count"]
                        or len(str(second["final_answer"]))
                        != len(str(first["final_answer"]))
                        or second["final_answer"] == first["final_answer"]
                        or int(first["final_answer"])
                        in integer_occurrences(str(second["packet"]))
                        or int(second["final_answer"])
                        in integer_occurrences(str(first["packet"]))
                        or semantic_signature(second_initial, second_operations)
                        == first_signature
                    ):
                        continue
                    if not _program_allowed(
                        second,
                        board_answers=board_answers,
                        board_programs=board_programs,
                        board_sources=board_sources,
                        board_packets=board_packets,
                        board_trajectories=board_trajectories,
                        board_grams=board_grams,
                        used_programs=used_programs | {first_signature},
                        used_sources=used_sources | {str(first["source"])},
                        used_packets=used_packets | {str(first["packet"])},
                        used_trajectories=used_trajectories
                        | {tuple(int(value) for value in first["trajectory"])},
                    ):
                        continue
                    for program in (first, second):
                        signature = semantic_signature(
                            program["initial_state"], program["operations"]
                        )
                        used_programs.add(signature)
                        used_sources.add(str(program["source"]))
                        used_packets.add(str(program["packet"]))
                        used_trajectories.add(
                            tuple(int(value) for value in program["trajectory"])
                        )
                        programs.append(program)
                    pair_id += 1
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError(
                    f"could not generate matched C2 training pair length={length}"
                )
    for index, program in enumerate(programs):
        program["id"] = f"{id_prefix}train_{index:04d}"
    return programs


def sham_stratum(program: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        len(program["operations"]),
        operation_types(program["operations"]),
        str(program["template_id"]),
        digit_widths(program["initial_state"], program["operations"]),
        int(program["packet_token_count"]),
        len(str(program["final_answer"])),
    )


def build_sham_permutation(
    programs: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[int, ...]:
    rng = random.Random(seed)
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, program in enumerate(programs):
        strata[sham_stratum(program)].append(index)
    mapping = [-1] * len(programs)
    for key in sorted(strata):
        recipients = list(strata[key])
        if len(recipients) < 2:
            raise RuntimeError(f"singleton C2 sham stratum {key!r}")
        rng.shuffle(recipients)
        choices: dict[int, list[int]] = {}
        for recipient in recipients:
            donors = [
                donor
                for donor in recipients
                if donor != recipient
                and programs[recipient]["final_answer"]
                != programs[donor]["final_answer"]
                and int(programs[recipient]["final_answer"])
                not in integer_occurrences(str(programs[donor]["packet"]))
            ]
            rng.shuffle(donors)
            choices[recipient] = donors
        owners: dict[int, int] = {}

        def assign(recipient: int, visited: set[int]) -> bool:
            for donor in choices[recipient]:
                if donor in visited:
                    continue
                visited.add(donor)
                previous = owners.get(donor)
                if previous is None or assign(previous, visited):
                    owners[donor] = recipient
                    return True
            return False

        if any(not assign(recipient, set()) for recipient in recipients):
            raise RuntimeError(f"no valid C2 sham derangement for {key!r}")
        for donor, recipient in owners.items():
            mapping[recipient] = donor
    if sorted(mapping) != list(range(len(programs))) or any(
        recipient == donor for recipient, donor in enumerate(mapping)
    ):
        raise RuntimeError("C2 sham mapping is not a complete derangement")
    return tuple(mapping)


def _false_observation(
    rng: random.Random,
    operation: Sequence[Any],
    *,
    forbidden_values: set[int],
) -> tuple[int, int]:
    for _attempt in range(100_000):
        state = rng.randint(1_000, 9_999)
        observed = rng.randint(1_000, 9_999)
        if state in forbidden_values or observed in forbidden_values:
            continue
        if observed == apply_operation(state, operation):
            continue
        return state, observed
    raise RuntimeError("could not sample an arithmetic-false C2 observation")


def build_training_arms(
    *,
    programs: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    sham_mapping: Sequence[int],
    observation_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(observation_seed)
    board_answers = {int(row["answer"]) for row in board_rows}
    treatment: list[dict[str, Any]] = []
    sham: list[dict[str, Any]] = []
    for index, program in enumerate(programs):
        prompt = compiler_prompt(str(program["source"]))
        common = {
            "completion_prompt": prompt,
            "id": f"{program['id']}_compiler",
            "kind": "compiler",
            "program_id": program["id"],
            "question": prompt,
            "schema": ROW_SCHEMA,
            "training_group": "rsp_c2",
        }
        treatment.append(
            {
                **common,
                "response": program["packet"],
                "response_program_id": program["id"],
            }
        )
        donor = programs[sham_mapping[index]]
        sham.append(
            {
                **common,
                "response": donor["packet"],
                "response_program_id": donor["id"],
            }
        )
        source_values = {int(value) for value in program["trajectory"]}
        source_values.update(int(operation[1]) for operation in program["operations"])
        forbidden = board_answers | source_values
        for step, operation in enumerate(program["operations"]):
            state, observed = _false_observation(
                rng, operation, forbidden_values=forbidden
            )
            packet = render_packet(state, program["operations"][step:])
            prompt = updater_prompt(packet, observed)
            remaining = program["operations"][step + 1 :]
            response = (
                render_packet(observed, remaining)
                if remaining
                else render_answer(observed)
            )
            row = {
                "completion_prompt": prompt,
                "id": f"{program['id']}_updater_{step:02d}",
                "kind": "updater",
                "program_id": program["id"],
                "question": prompt,
                "response": response,
                "schema": ROW_SCHEMA,
                "step": step,
                "training_group": "rsp_c2",
            }
            treatment.append(row)
            sham.append(dict(row))
    return treatment, sham


def token_accounting(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, eos_id: int
) -> dict[str, Any]:
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("EOS id must be a nonnegative integer")
    all_masks = bytearray()
    row_lengths: list[int] = []
    prompt_order: list[str] = []
    response_token_counts: Counter[int] = Counter()
    prompt_tokens = 0
    response_tokens = 0
    supervised_tokens = 0
    for row in rows:
        prompt = str(row["completion_prompt"])
        response = str(row["response"]).rstrip()
        prompt_ids = _token_ids(tokenizer, prompt)
        response_ids = _token_ids(tokenizer, response)
        full_length = len(prompt_ids) + len(response_ids) + 1
        if full_length > PACK_LENGTH:
            raise RuntimeError("a C2 example exceeds the frozen pack length")
        row_lengths.append(full_length)
        prompt_order.append(prompt)
        prompt_tokens += len(prompt_ids)
        response_tokens += len(response_ids)
        supervised_tokens += len(response_ids) + 1
        all_masks.extend(b"\0" * len(prompt_ids))
        all_masks.extend(b"\1" * (len(response_ids) + 1))
        response_token_counts.update(response_ids)
    full_tokens = len(all_masks)
    packed_sequences = max(0, (full_tokens - 2) // PACK_LENGTH)
    forward_tokens = packed_sequences * PACK_LENGTH
    packed_target_mask = bytes(all_masks[1 : forward_tokens + 1])
    response_multiset = [
        [token_id, count] for token_id, count in sorted(response_token_counts.items())
    ]
    return {
        "compiler_rows": sum(row["kind"] == "compiler" for row in rows),
        "discarded_token_count": full_tokens - forward_tokens,
        "example_count": len(rows),
        "full_token_count": full_tokens,
        "packed_forward_positions_sha256": sha256_bytes(
            canonical_json_bytes([0, forward_tokens, PACK_LENGTH])
        ),
        "packed_sequence_count": packed_sequences,
        "packed_supervision_geometry_sha256": sha256_bytes(packed_target_mask),
        "prompt_order_sha256": sha256_bytes(canonical_json_bytes(prompt_order)),
        "prompt_token_count": prompt_tokens,
        "response_token_count": response_tokens,
        "response_token_multiset_sha256": sha256_bytes(
            canonical_json_bytes(response_multiset)
        ),
        "row_encoded_lengths_sha256": sha256_bytes(canonical_json_bytes(row_lengths)),
        "supervised_target_token_count": supervised_tokens,
        "updater_rows": sum(row["kind"] == "updater" for row in rows),
    }


def _data_products(
    *,
    board: Mapping[str, Any],
    tokenizer: Any,
    training_seed: int,
    observation_seed: int,
    sham_seed: int,
    geometry: Geometry,
    id_prefix: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[int, ...],
    dict[str, Any],
    dict[str, Any],
]:
    programs = build_training_programs(
        board_rows=board["rows"],
        tokenizer=tokenizer,
        seed=training_seed,
        geometry=geometry,
        id_prefix=id_prefix,
    )
    mapping = build_sham_permutation(programs, seed=sham_seed)
    treatment, sham = build_training_arms(
        programs=programs,
        board_rows=board["rows"],
        sham_mapping=mapping,
        observation_seed=observation_seed,
    )
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    treatment_tokens = token_accounting(treatment, tokenizer, int(eos_id))
    sham_tokens = token_accounting(sham, tokenizer, int(eos_id))
    if treatment_tokens != sham_tokens:
        raise RuntimeError("C2 treatment and sham token contracts differ")
    treatment_updaters = [row for row in treatment if row["kind"] == "updater"]
    sham_updaters = [row for row in sham if row["kind"] == "updater"]
    if jsonl_bytes(treatment_updaters) != jsonl_bytes(sham_updaters):
        raise RuntimeError("C2 updater rows are not byte-identical")
    return programs, treatment, sham, mapping, treatment_tokens, sham_tokens


def _exclusive_immutable_write(path: str | Path, payload: bytes) -> str:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {destination.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating immutable C2 output")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    info = destination.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise PermissionError("C2 output is not a regular read-only file")
    return sha256_bytes(payload)


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    return _exclusive_immutable_write(path, canonical_json_bytes(payload))


def _preflight_outputs(paths: Sequence[str | Path]) -> list[Path]:
    destinations = [Path(path).resolve(strict=False) for path in paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError("all C2 output paths must be distinct")
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to reuse C2 output path: {destination}")
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"output parent does not exist: {destination.parent}"
            )
    return destinations


def _invocation_payload(
    *,
    phase: str,
    custody: Custody,
    outputs: Sequence[Path],
    extra_inputs: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "prerequisite_receipt_sha256": custody.prerequisite_sha256,
            "provenance_receipt_sha256": custody.provenance_sha256,
            "runtime_receipt_sha256": custody.runtime_receipt_sha256,
            "seed_receipt_sha256": custody.seed_receipt_sha256,
            "tokenizer_sha256": custody.tokenizer_sha256,
        },
        "extra_inputs": dict(sorted(extra_inputs.items())),
        "output_paths": [str(path) for path in outputs],
        "phase": phase,
        "profile": PRODUCTION_PROFILE,
        "schema": INVOCATION_SCHEMA,
    }


def _validate_board_shape(board: Mapping[str, Any], custody: Custody) -> None:
    expected_keys = {
        "case_count",
        "custody",
        "geometry",
        "geometry_sha256",
        "per_stratum",
        "profile",
        "rows",
        "rows_sha256",
        "schema",
        "seed_commitment_sha256",
        "stratum_order",
    }
    if set(board) != expected_keys or board.get("schema") != BOARD_SCHEMA:
        raise ValueError("board schema mismatch")
    if board.get("profile") != PRODUCTION_PROFILE:
        raise ValueError("board profile mismatch")
    if board.get("case_count") != 256 or board.get("per_stratum") != 64:
        raise ValueError("board production count mismatch")
    if board.get("stratum_order") != list(STRATUM_ORDER):
        raise ValueError("board stratum order mismatch")
    if board.get("geometry") != geometry_payload(PRODUCTION_GEOMETRY):
        raise ValueError("board geometry mismatch")
    if board.get("geometry_sha256") != sha256_bytes(
        canonical_json_bytes(board["geometry"])
    ):
        raise ValueError("board geometry SHA-256 mismatch")
    rows = board.get("rows")
    if not isinstance(rows, list) or len(rows) != 256:
        raise ValueError("board row count mismatch")
    if board.get("rows_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise ValueError("board canonical-row SHA-256 mismatch")
    if board.get("seed_commitment_sha256") != custody.seed_digests["board"]:
        raise ValueError("board seed commitment mismatch")
    expected_custody = {
        "freeze_commit": custody.provenance["freeze_commit"],
        "prerequisite_receipt_sha256": custody.prerequisite_sha256,
        "provenance_receipt_sha256": custody.provenance_sha256,
        "seed_receipt_sha256": custody.seed_receipt_sha256,
    }
    if board.get("custody") != expected_custody:
        raise ValueError("board custody block mismatch")
    strata = Counter(row.get("stratum") for row in rows if isinstance(row, dict))
    if strata != Counter({name: 64 for name in STRATUM_ORDER}):
        raise ValueError("board stratum counts mismatch")
    if [row.get("stratum") for row in rows] != [
        name for name in STRATUM_ORDER for _ in range(64)
    ]:
        raise ValueError("board stratum ordering mismatch")


def _read_board(
    path: str | Path, expected_sha256: str, custody: Custody
) -> dict[str, Any]:
    expected_sha256 = _lower_hash(expected_sha256, "expected board SHA-256")
    board, raw = read_immutable_json(path, "C2 board", canonical=True)
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("board artifact SHA-256 mismatch")
    _validate_board_shape(board, custody)
    return board


def _read_board_audit(
    path: str | Path,
    expected_sha256: str,
    *,
    board_sha256: str,
    board_rows_sha256: str,
    custody: Custody,
) -> dict[str, Any]:
    expected_sha256 = _lower_hash(expected_sha256, "expected board audit SHA-256")
    audit, raw = read_immutable_json(path, "C2 board audit", canonical=True)
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("board audit SHA-256 mismatch")
    expected_keys = {"admitted", "artifacts", "custody", "failures", "replay", "schema"}
    if set(audit) != expected_keys or audit.get("schema") != BOARD_AUDIT_SCHEMA:
        raise ValueError("board audit schema mismatch")
    if audit.get("admitted") is not True or audit.get("failures") != []:
        raise ValueError("board audit did not admit the board")
    if audit.get("artifacts") != {
        "board_rows_sha256": board_rows_sha256,
        "board_sha256": board_sha256,
    }:
        raise ValueError("board audit does not bind the supplied board")
    expected_custody = {
        "prerequisite_receipt_sha256": custody.prerequisite_sha256,
        "provenance_receipt_sha256": custody.provenance_sha256,
        "seed_receipt_sha256": custody.seed_receipt_sha256,
    }
    if audit.get("custody") != expected_custody:
        raise ValueError("board audit custody mismatch")
    expected_replay = {
        "case_count": 256,
        "expected_board_sha256": board_sha256,
        "expected_rows_sha256": board_rows_sha256,
        "strata": {name: 64 for name in sorted(STRATUM_ORDER)},
    }
    if audit.get("replay") != expected_replay:
        raise ValueError("board audit replay summary mismatch")
    return audit


def _load_frozen_tokenizer(custody: Custody) -> Any:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(custody.tokenizer_bytes.decode("utf-8"))
    if tokenizer.token_to_id("<|endoftext|>") is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    return tokenizer


def run_board_phase(args: argparse.Namespace) -> dict[str, Any]:
    outputs = _preflight_outputs((args.invocation_out, args.board_out))
    custody = load_production_custody(
        prerequisite_path=args.prerequisite_receipt,
        prerequisite_sha256=args.prerequisite_receipt_sha256,
        seed_receipt_path=args.seed_receipt,
        seed_receipt_sha256=args.seed_receipt_sha256,
        provenance_path=args.provenance_receipt,
        provenance_sha256=args.provenance_receipt_sha256,
        tokenizer_path=args.tokenizer,
        runtime_receipt_path=args.runtime_receipt,
    )
    _load_frozen_tokenizer(custody)
    invocation = _invocation_payload(
        phase="board",
        custody=custody,
        outputs=outputs,
        extra_inputs={},
    )
    invocation_sha256 = write_immutable_json(outputs[0], invocation)
    board = production_board_payload(custody)
    board_sha256 = write_immutable_json(outputs[1], board)
    return {
        "board_sha256": board_sha256,
        "case_count": board["case_count"],
        "invocation_sha256": invocation_sha256,
        "rows_sha256": board["rows_sha256"],
        "schema": BOARD_SCHEMA,
    }


def run_data_phase(args: argparse.Namespace) -> dict[str, Any]:
    outputs = _preflight_outputs(
        (args.invocation_out, args.treatment_out, args.sham_out, args.manifest_out)
    )
    custody = load_production_custody(
        prerequisite_path=args.prerequisite_receipt,
        prerequisite_sha256=args.prerequisite_receipt_sha256,
        seed_receipt_path=args.seed_receipt,
        seed_receipt_sha256=args.seed_receipt_sha256,
        provenance_path=args.provenance_receipt,
        provenance_sha256=args.provenance_receipt_sha256,
        tokenizer_path=args.tokenizer,
        runtime_receipt_path=args.runtime_receipt,
    )
    board_sha256 = _lower_hash(args.board_sha256, "expected board SHA-256")
    board_audit_sha256 = _lower_hash(
        args.board_audit_sha256, "expected board audit SHA-256"
    )
    board = _read_board(args.board, board_sha256, custody)
    _read_board_audit(
        args.board_audit,
        board_audit_sha256,
        board_sha256=board_sha256,
        board_rows_sha256=board["rows_sha256"],
        custody=custody,
    )
    tokenizer = _load_frozen_tokenizer(custody)
    invocation = _invocation_payload(
        phase="data",
        custody=custody,
        outputs=outputs,
        extra_inputs={
            "board_audit_sha256": board_audit_sha256,
            "board_sha256": board_sha256,
        },
    )
    invocation_sha256 = write_immutable_json(outputs[0], invocation)
    programs, treatment, sham, mapping, treatment_tokens, sham_tokens = _data_products(
        board=board,
        tokenizer=tokenizer,
        training_seed=custody.seed_integers["training"],
        observation_seed=custody.seed_integers["observation"],
        sham_seed=custody.seed_integers["sham"],
        geometry=PRODUCTION_GEOMETRY,
        id_prefix="rsp_c2_",
    )
    treatment_raw = jsonl_bytes(treatment)
    sham_raw = jsonl_bytes(sham)
    treatment_sha256 = _exclusive_immutable_write(outputs[1], treatment_raw)
    sham_sha256 = _exclusive_immutable_write(outputs[2], sham_raw)
    strata = Counter(sham_stratum(program) for program in programs)
    manifest = {
        "artifacts": {
            "board_audit_sha256": board_audit_sha256,
            "board_rows_sha256": board["rows_sha256"],
            "board_sha256": board_sha256,
            "sham_rows": len(sham),
            "sham_sha256": sham_sha256,
            "treatment_rows": len(treatment),
            "treatment_sha256": treatment_sha256,
        },
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "generation_invocation_sha256": invocation_sha256,
            "prerequisite_receipt_sha256": custody.prerequisite_sha256,
            "provenance_receipt_sha256": custody.provenance_sha256,
            "runtime_receipt_sha256": custody.runtime_receipt_sha256,
            "seed_commitments": {
                label: custody.seed_digests[label]
                for label in ("board", "training", "observation", "sham")
            },
            "seed_receipt_sha256": custody.seed_receipt_sha256,
            "tokenizer_sha256": custody.tokenizer_sha256,
        },
        "encoding_contract": {
            "completion_only": True,
            "eos_token": "<|endoftext|>",
            "pack_length": PACK_LENGTH,
            "prompt_field": "completion_prompt",
            "response_field": "response",
            "response_rstrip": True,
        },
        "geometry_sha256": board["geometry_sha256"],
        "length_counts": {
            str(key): value for key, value in sorted(TRAIN_LENGTH_COUNTS.items())
        },
        "profile": PRODUCTION_PROFILE,
        "program_count": len(programs),
        "programs_sha256": sha256_bytes(canonical_json_bytes(programs)),
        "schema": MANIFEST_SCHEMA,
        "sham_contract": {
            "mapping_sha256": sha256_bytes(canonical_json_bytes(list(mapping))),
            "minimum_stratum_size": min(strata.values()),
            "stratum_count": len(strata),
        },
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "token_parity": {
            "all_locked_fields_equal": treatment_tokens == sham_tokens,
            "updater_rows_byte_identical": jsonl_bytes(
                [row for row in treatment if row["kind"] == "updater"]
            )
            == jsonl_bytes([row for row in sham if row["kind"] == "updater"]),
        },
    }
    manifest_sha256 = write_immutable_json(outputs[3], manifest)
    return {
        "manifest_sha256": manifest_sha256,
        "program_count": len(programs),
        "schema": MANIFEST_SCHEMA,
        "sham_sha256": sham_sha256,
        "treatment_sha256": treatment_sha256,
    }


class ToyHashTokenizer:
    """Deterministic tokenizer used only by explicitly labelled toy fixtures."""

    class _Encoding:
        def __init__(self, ids: list[int]):
            self.ids = ids

    def encode(self, text: str) -> _Encoding:
        ids = [
            int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big")
            for token in TOKEN_RE.findall(text)
        ]
        return self._Encoding(ids)

    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<|endoftext|>" else None


def toy_geometry(
    label: str,
    *,
    per_stratum: int = 4,
    training_length_counts: Mapping[int, int] | None = None,
) -> Geometry:
    if not isinstance(label, str) or not label.startswith(TOY_PREFIX):
        raise ValueError(f"toy label must begin with {TOY_PREFIX}")
    counts = dict(training_length_counts or {2: 4, 3: 8, 4: 4})
    if per_stratum <= 0 or per_stratum >= PER_STRATUM or per_stratum % 2:
        raise ValueError(
            "toy per-stratum count must be positive, even, and nonproduction"
        )
    if not counts or any(
        count <= 0 or count >= TRAIN_LENGTH_COUNTS.get(length, 0) or count % 2
        for length, count in counts.items()
    ):
        raise ValueError("toy training counts must be even and strictly sub-production")
    return Geometry(
        profile=label,
        per_stratum=per_stratum,
        training_length_counts=counts,
    )


def build_toy_fixture_bundle(
    label: str,
    *,
    tokenizer: Any | None = None,
    per_stratum: int = 4,
    training_length_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Build a tiny, unmistakably nonproduction fixture without production custody."""

    geometry = toy_geometry(
        label,
        per_stratum=per_stratum,
        training_length_counts=training_length_counts,
    )
    tokenizer = ToyHashTokenizer() if tokenizer is None else tokenizer

    def toy_seed(seed_label: str) -> int:
        payload = f"{label}:{seed_label}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest(), "big")

    rows = build_board_rows(
        seed=toy_seed("board"), geometry=geometry, id_prefix=f"{label}_"
    )
    board_geometry = geometry_payload(geometry)
    board = {
        "case_count": len(rows),
        "fixture_label": label,
        "geometry": board_geometry,
        "geometry_sha256": sha256_bytes(canonical_json_bytes(board_geometry)),
        "profile": label,
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "schema": "toy_only_rsp_c2_board_fixture_v1",
    }
    programs, treatment, sham, mapping, treatment_tokens, sham_tokens = _data_products(
        board=board,
        tokenizer=tokenizer,
        training_seed=toy_seed("training"),
        observation_seed=toy_seed("observation"),
        sham_seed=toy_seed("sham"),
        geometry=geometry,
        id_prefix=f"{label}_",
    )
    manifest = {
        "fixture_label": label,
        "length_counts": {
            str(key): value
            for key, value in sorted(geometry.training_length_counts.items())
        },
        "mapping_sha256": sha256_bytes(canonical_json_bytes(list(mapping))),
        "program_count": len(programs),
        "programs_sha256": sha256_bytes(canonical_json_bytes(programs)),
        "schema": "toy_only_rsp_c2_generation_manifest_fixture_v1",
        "sham_sha256": sha256_bytes(jsonl_bytes(sham)),
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "treatment_sha256": sha256_bytes(jsonl_bytes(treatment)),
    }
    return {
        "board": board,
        "fixture_label": label,
        "manifest": manifest,
        "schema": "toy_only_rsp_c2_bundle_fixture_v1",
        "sham": sham,
        "treatment": treatment,
    }


def _add_custody_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prerequisite-receipt", required=True)
    parser.add_argument("--prerequisite-receipt-sha256", required=True)
    parser.add_argument("--seed-receipt", required=True)
    parser.add_argument("--seed-receipt-sha256", required=True)
    parser.add_argument("--provenance-receipt", required=True)
    parser.add_argument("--provenance-receipt-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--runtime-receipt", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser(
        "board", help="one-shot production board materialization"
    )
    _add_custody_arguments(board)
    board.add_argument("--invocation-out", required=True)
    board.add_argument("--board-out", required=True)
    data = commands.add_parser("data", help="one-shot production arm materialization")
    _add_custody_arguments(data)
    data.add_argument("--board", required=True)
    data.add_argument("--board-sha256", required=True)
    data.add_argument("--board-audit", required=True)
    data.add_argument("--board-audit-sha256", required=True)
    data.add_argument("--invocation-out", required=True)
    data.add_argument("--treatment-out", required=True)
    data.add_argument("--sham-out", required=True)
    data.add_argument("--manifest-out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_board_phase(args) if args.command == "board" else run_data_phase(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
