#!/usr/bin/env python3
"""Independently replay and admit deferred-seed RSP-C2 artifacts.

This auditor is intentionally self-contained and never imports the C2
generator.  It reconstructs board rows, semantic programs, observations, sham
assignment, encoded token geometry, and canonical artifact bytes from the
immutable seed receipt and frozen provenance.
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
GENERATOR_PATH = ROOT / "pipeline" / "generate_residual_packet_c2.py"

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
GENERATION_INVOCATION_SCHEMA = "rsp_c2_generation_invocation_v1"
AUDIT_INVOCATION_SCHEMA = "rsp_c2_audit_invocation_v1"
BOARD_SCHEMA = "rsp_c2_board_v1"
BOARD_AUDIT_SCHEMA = "rsp_c2_board_admission_audit_v1"
DATA_AUDIT_SCHEMA = "rsp_c2_data_admission_audit_v1"
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
LEXEME_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]", re.ASCII)
WORD_NUMBER_RE = re.compile(r"[a-z]+|[0-9]+", re.ASCII)
STANDALONE_INTEGER_RE = re.compile(r"(?<![A-Za-z0-9])[0-9]+(?![A-Za-z0-9])", re.ASCII)
PACKET_RE = re.compile(
    r"<C2P\|S=(?P<state>[0-9]+)\|R=(?P<plan>(?:ADD|MUL|SUB):[0-9]+"
    r"(?:,(?:ADD|MUL|SUB):[0-9]+)*)>\Z",
    re.ASCII,
)
UPDATER_PROMPT_RE = re.compile(
    r"Advance exactly one C2 packet step using the supplied observation\.\n"
    r"Packet:\n(?P<packet><C2P\|S=[0-9]+\|R=.*>)\n"
    r"Observation: (?P<observed>[0-9]+)\nNext:\n\Z",
    re.ASCII,
)

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
    length_counts: Mapping[int, int]
    normal_initial: tuple[int, int] = (20, 98)
    normal_add_sub: tuple[int, int] = (3, 24)
    normal_multiply: tuple[int, int] = (2, 6)
    ood_initial: tuple[int, int] = (300, 599)
    ood_add_sub: tuple[int, int] = (40, 90)
    ood_multiply: tuple[int, int] = (8, 13)


PRODUCTION_GEOMETRY = Geometry(PRODUCTION_PROFILE, PER_STRATUM, TRAIN_LENGTH_COUNTS)


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


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def decode_json(raw: bytes, label: str) -> Any:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    try:
        return json.loads(
            source,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _read_file(path: str | Path, label: str, *, immutable: bool) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        first = os.fstat(descriptor)
        if not stat.S_ISREG(first.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if immutable and first.st_mode & 0o222:
            raise PermissionError(f"{label} must be read-only")
        parts: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            parts.append(block)
        second = os.fstat(descriptor)
        before = (
            first.st_dev,
            first.st_ino,
            first.st_size,
            first.st_mtime_ns,
            first.st_ctime_ns,
        )
        after = (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
            second.st_ctime_ns,
        )
        if before != after:
            raise RuntimeError(f"{label} changed during audit read")
        return b"".join(parts)
    finally:
        os.close(descriptor)


def read_json_artifact(
    path: str | Path, label: str, *, canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    raw = _read_file(path, label, immutable=True)
    value = decode_json(raw, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    if canonical and raw != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value, raw


def read_jsonl_artifact(
    path: str | Path, label: str
) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_file(path, label, immutable=True)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{label} must be nonempty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if line == b"\n":
            raise ValueError(f"{label} contains a blank row at {index}")
        value = decode_json(line, f"{label} row {index}")
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise ValueError(f"{label} row {index} is not a canonical object")
        rows.append(value)
    return rows, raw


def file_digest(path: str | Path, label: str) -> str:
    return digest(_read_file(path, label, immutable=False))


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_64_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical lowercase SHA-256")
    return value


def require_oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git OID")
    return value


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def utc_millis(value: Any, label: str) -> int:
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
        raise ValueError(f"{label} is not a valid timestamp") from error
    result = int(parsed.timestamp()) * 1000 + int(match.group("millis"))
    rerendered = datetime.fromtimestamp(result // 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if value != f"{rerendered}.{result % 1000:03d}Z":
        raise ValueError(f"{label} is not canonical")
    return result


def inspect_prerequisite(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        set(payload) != PREREQUISITE_KEYS
        or payload.get("schema") != PREREQUISITE_SCHEMA
    ):
        raise ValueError("prerequisite receipt schema mismatch")
    gates = (
        "advance_to_internalization",
        "all_locked_gates_pass",
        "independent_recomputation_complete",
        "result_immutable",
        "scorers_agree",
    )
    for gate in gates:
        if payload[gate] is not True:
            raise ValueError(f"prerequisite receipt requires {gate}=true")
    output = dict(payload)
    hashes = (
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
    )
    for key in hashes:
        output[key] = require_hash(payload[key], key)
    if output["primary_scorer_sha256"] == output["independent_scorer_sha256"]:
        raise ValueError("prerequisite scorers are not independent")
    if (
        output["primary_score_receipt_sha256"]
        == output["independent_score_receipt_sha256"]
    ):
        raise ValueError("prerequisite score receipts are not independent")
    return output


def require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical nonempty identifier")
    return value


def receipt_field(value: str | int) -> bytes:
    return str(value).encode("ascii") + b"\0"


def inspect_seed_receipt(
    payload: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    expected_top = {
        "base_commitment_sha256",
        "beacon",
        "freeze",
        "prerequisite",
        "schema",
        "seed_labels",
        "seed_scheme",
        "seeds",
    }
    if set(payload) != expected_top:
        raise ValueError("seed receipt top-level schema mismatch")
    if (
        payload.get("schema") != SEED_RECEIPT_SCHEMA
        or payload.get("seed_scheme") != SEED_SCHEME
    ):
        raise ValueError("seed receipt identity mismatch")
    if payload.get("seed_labels") != list(SEED_LABELS):
        raise ValueError("seed receipt label set or order mismatch")
    freeze = payload.get("freeze")
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
    commit_oid = require_oid(freeze["commit_oid"], "seed receipt freeze commit")
    remote_ref_oid = require_oid(
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
    pushed_at = utc_millis(freeze["pushed_at"], "seed receipt freeze pushed_at")
    observed_at = utc_millis(freeze["observed_at"], "seed receipt freeze observed_at")
    if not 0 <= observed_at - pushed_at <= 300_000:
        raise ValueError("seed receipt freeze observation lag is invalid")
    require_identifier(freeze["observer_id"], "seed receipt freeze observer")
    for key in (
        "observer_implementation_sha256",
        "raw_sha256",
        "remote_ref_evidence_sha256",
    ):
        require_hash(freeze[key], f"seed receipt freeze {key}")
    freeze_receipt = {
        **{key: value for key, value in freeze.items() if key != "raw_sha256"},
        "schema": FREEZE_PUSH_SCHEMA,
    }
    if digest(canonical_json_bytes(freeze_receipt)) != freeze["raw_sha256"]:
        raise ValueError("seed receipt freeze raw SHA-256 does not replay")

    prerequisite = payload.get("prerequisite")
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
        require_hash(prerequisite[key], f"seed receipt prerequisite {key}")
    if (
        prerequisite["primary_scorer_sha256"]
        == prerequisite["independent_scorer_sha256"]
    ):
        raise ValueError("seed receipt scorers are not independent")
    if (
        prerequisite["primary_score_receipt_sha256"]
        == prerequisite["independent_score_receipt_sha256"]
    ):
        raise ValueError("seed receipt score receipts are not independent")
    prerequisite_receipt = {
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
    if digest(canonical_json_bytes(prerequisite_receipt)) != prerequisite["raw_sha256"]:
        raise ValueError("seed receipt prerequisite raw SHA-256 does not replay")

    beacon = payload.get("beacon")
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
        require_hash(beacon[key], f"seed receipt beacon {key}")
    output_value = beacon["output_value"]
    if not isinstance(output_value, str) or HEX_128_RE.fullmatch(output_value) is None:
        raise ValueError("seed receipt beacon output is not lowercase 512-bit hex")
    if digest(bytes.fromhex(output_value)) != beacon["output_value_sha256"]:
        raise ValueError("seed receipt beacon output hash mismatch")
    if beacon["period_ms"] != 60_000 or isinstance(beacon["period_ms"], bool):
        raise ValueError("seed receipt beacon period mismatch")
    if beacon["status_code"] != 0 or isinstance(beacon["status_code"], bool):
        raise ValueError("seed receipt beacon status mismatch")
    positive_int(beacon["chain_index"], "seed receipt chain index")
    positive_int(beacon["pulse_index"], "seed receipt pulse index")
    timestamp = utc_millis(beacon["time_stamp"], "seed receipt beacon timestamp")
    if beacon["time_stamp_ms"] != timestamp or isinstance(
        beacon["time_stamp_ms"], bool
    ):
        raise ValueError("seed receipt beacon timestamp mismatch")
    target = pushed_at + 3_600_000
    if (
        beacon["first_pulse_target_ms"] != target
        or not 0 <= timestamp - target < 60_000
    ):
        raise ValueError("seed receipt beacon is not in the first eligible slot")
    if not isinstance(beacon["certificate_id"], str) or not beacon["certificate_id"]:
        raise ValueError("seed receipt beacon certificate is absent")

    verification = beacon["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "certificate_sha256",
        "raw_sha256",
        "validators",
        "validators_agree",
    }:
        raise ValueError("seed receipt beacon verification schema mismatch")
    require_hash(
        verification["certificate_sha256"],
        "seed receipt beacon certificate SHA-256",
    )
    require_hash(
        verification["raw_sha256"],
        "seed receipt beacon verification receipt SHA-256",
    )
    if verification["validators_agree"] is not True:
        raise ValueError("seed receipt beacon validators disagree")
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
        raise ValueError("seed receipt requires two beacon validators")
    identities = []
    for index, validator in enumerate(validators):
        if not isinstance(validator, dict) or set(validator) != validator_keys:
            raise ValueError("seed receipt beacon validator schema mismatch")
        for gate in ("certificate_valid", "chain_valid", "signature_valid"):
            if validator[gate] is not True:
                raise ValueError(f"seed receipt beacon validator {index} failed {gate}")
        validator_id = require_identifier(
            validator["validator_id"], f"seed receipt beacon validator {index} id"
        )
        implementation = require_hash(
            validator["implementation_sha256"],
            f"seed receipt beacon validator {index} implementation",
        )
        evidence = require_hash(
            validator["evidence_sha256"],
            f"seed receipt beacon validator {index} evidence",
        )
        raw_beacon = require_hash(
            validator["raw_beacon_sha256"],
            f"seed receipt beacon validator {index} raw beacon",
        )
        if raw_beacon != beacon["raw_sha256"]:
            raise ValueError("seed receipt beacon validator raw hash mismatch")
        utc_millis(
            validator["validated_at"],
            f"seed receipt beacon validator {index} validated_at",
        )
        identities.append((validator_id, implementation, evidence))
    for field_index, label in (
        (0, "identifier"),
        (1, "implementation"),
        (2, "evidence"),
    ):
        if identities[0][field_index] == identities[1][field_index]:
            raise ValueError(f"seed receipt beacon validator {label}s must differ")
    verification_receipt = {
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
    if digest(canonical_json_bytes(verification_receipt)) != verification["raw_sha256"]:
        raise ValueError("seed receipt beacon verification raw SHA-256 does not replay")

    base_preimage = b"".join(
        (
            BASE_DOMAIN,
            receipt_field(commit_oid),
            receipt_field(remote_url),
            receipt_field(freeze["ref"]),
            receipt_field(freeze["pushed_at"]),
            receipt_field(freeze["observed_at"]),
            receipt_field(freeze["raw_sha256"]),
            receipt_field(prerequisite["raw_sha256"]),
            receipt_field(prerequisite["confirmation_result_sha256"]),
            receipt_field(beacon["raw_sha256"]),
            receipt_field(beacon["chain_index"]),
            receipt_field(beacon["pulse_index"]),
            receipt_field(beacon["time_stamp_ms"]),
            receipt_field(beacon["output_value"]),
            receipt_field(verification["raw_sha256"]),
        )
    )
    base = hashlib.sha256(base_preimage).digest()
    if require_hash(payload["base_commitment_sha256"], "base commitment") != base.hex():
        raise ValueError("seed receipt base commitment does not replay")

    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != set(SEED_LABELS):
        raise ValueError("seed receipt seed map mismatch")
    integers: dict[str, int] = {}
    commitments: dict[str, str] = {}
    for label in SEED_LABELS:
        record = seeds[label]
        if not isinstance(record, dict) or set(record) != {"integer_decimal", "sha256"}:
            raise ValueError(f"seed receipt {label} schema mismatch")
        expected = hashlib.sha256(
            SEED_DOMAIN + base + b"\0" + label.encode("ascii")
        ).digest()
        observed_digest = require_hash(record["sha256"], f"{label} seed digest")
        decimal = record["integer_decimal"]
        if not isinstance(decimal, str) or DECIMAL_RE.fullmatch(decimal) is None:
            raise ValueError(f"{label} seed decimal is not canonical")
        observed_integer = int(decimal)
        if observed_digest != expected.hex() or observed_integer != int.from_bytes(
            expected, "big"
        ):
            raise ValueError(f"{label} seed derivation mismatch")
        integers[label] = observed_integer
        commitments[label] = observed_digest
    if len(set(commitments.values())) != len(SEED_LABELS):
        raise ValueError("seed receipt derived seeds are not unique")
    return integers, commitments


def inspect_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != PROVENANCE_KEYS or payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("frozen provenance receipt schema mismatch")
    output = dict(payload)
    output["freeze_commit"] = require_oid(payload["freeze_commit"], "freeze commit")
    for key in PROVENANCE_KEYS - {"schema", "freeze_commit"}:
        output[key] = require_hash(payload[key], f"provenance {key}")
    if output["preregistration_sha256"] != FROZEN_PREREGISTRATION_SHA256:
        raise ValueError("provenance C2 preregistration hash mismatch")
    if output["c1_closure_sha256"] != FROZEN_C1_CLOSURE_SHA256:
        raise ValueError("provenance C1 closure hash mismatch")
    return output


def load_custody(
    *,
    prerequisite_path: str | Path,
    prerequisite_sha256: str,
    seed_path: str | Path,
    seed_sha256: str,
    provenance_path: str | Path,
    provenance_sha256: str,
    tokenizer_path: str | Path,
    runtime_path: str | Path,
) -> Custody:
    prerequisite_sha256 = require_hash(
        prerequisite_sha256, "expected prerequisite SHA-256"
    )
    seed_sha256 = require_hash(seed_sha256, "expected seed receipt SHA-256")
    provenance_sha256 = require_hash(provenance_sha256, "expected provenance SHA-256")

    prerequisite_payload, prerequisite_raw = read_json_artifact(
        prerequisite_path, "prerequisite pass receipt", canonical=True
    )
    if digest(prerequisite_raw) != prerequisite_sha256:
        raise ValueError("prerequisite receipt SHA-256 mismatch")
    prerequisite = inspect_prerequisite(prerequisite_payload)

    seed_payload, seed_raw = read_json_artifact(seed_path, "C2 seed receipt")
    if digest(seed_raw) != seed_sha256:
        raise ValueError("seed receipt SHA-256 mismatch")
    seed_integers, seed_digests = inspect_seed_receipt(seed_payload)

    provenance_payload, provenance_raw = read_json_artifact(
        provenance_path, "C2 frozen provenance receipt"
    )
    if digest(provenance_raw) != provenance_sha256:
        raise ValueError("provenance receipt SHA-256 mismatch")
    provenance = inspect_provenance(provenance_payload)

    embedded = seed_payload["prerequisite"]
    if embedded["raw_sha256"] != prerequisite_sha256:
        raise ValueError("seed receipt prerequisite raw hash mismatch")
    for key in (
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
    ):
        if embedded[key] != prerequisite[key]:
            raise ValueError(f"seed receipt prerequisite mismatch: {key}")
    if provenance["prerequisite_receipt_sha256"] != prerequisite_sha256:
        raise ValueError("provenance prerequisite hash mismatch")
    if seed_payload["freeze"]["commit_oid"] != provenance["freeze_commit"]:
        raise ValueError("seed receipt freeze commit mismatch")
    if seed_payload["freeze"]["raw_sha256"] != provenance["freeze_push_receipt_sha256"]:
        raise ValueError("seed receipt push receipt mismatch")

    tokenizer_raw = _read_file(tokenizer_path, "tokenizer", immutable=True)
    actual = {
        "auditor_sha256": file_digest(__file__, "C2 auditor"),
        "c1_closure_sha256": file_digest(C1_CLOSURE_PATH, "C1 closure document"),
        "generator_sha256": file_digest(GENERATOR_PATH, "C2 generator"),
        "preregistration_sha256": file_digest(
            PREREGISTRATION_PATH, "C2 preregistration"
        ),
        "seed_derivation_sha256": file_digest(
            SEED_DERIVATION_PATH, "C2 seed derivation"
        ),
        "tokenizer_sha256": digest(tokenizer_raw),
    }
    for key, value in actual.items():
        if provenance[key] != value:
            raise ValueError(f"frozen provenance mismatch for {key}")
    runtime_raw = _read_file(runtime_path, "runtime identity receipt", immutable=True)
    runtime_digest = digest(runtime_raw)
    if runtime_digest != provenance["runtime_receipt_sha256"]:
        raise ValueError("runtime receipt hash mismatch")
    return Custody(
        prerequisite=prerequisite,
        prerequisite_sha256=prerequisite_sha256,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
        seed_receipt=seed_payload,
        seed_receipt_sha256=seed_sha256,
        seed_integers=seed_integers,
        seed_digests=seed_digests,
        tokenizer_bytes=tokenizer_raw,
        tokenizer_sha256=actual["tokenizer_sha256"],
        runtime_receipt_sha256=runtime_digest,
    )


def execute(state: int, operation: Sequence[Any]) -> int:
    kind = str(operation[0])
    operand = int(operation[1])
    if kind == "add":
        return state + operand
    if kind == "multiply":
        return state * operand
    if kind == "subtract":
        return state - operand
    raise ValueError(f"unsupported C2 operation {kind!r}")


def replay_states(initial: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    values = [int(initial)]
    for operation in operations:
        values.append(execute(values[-1], operation))
    return tuple(values)


def signature(initial: int, operations: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    return (int(initial),) + tuple(
        (str(operation[0]), int(operation[1])) for operation in operations
    )


def kinds(operations: Sequence[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(operation[0]) for operation in operations)


def widths(initial: int, operations: Sequence[Sequence[Any]]) -> tuple[int, ...]:
    return (len(str(initial)),) + tuple(
        len(str(int(operation[1]))) for operation in operations
    )


def integers_in(text: str) -> tuple[int, ...]:
    return tuple(int(match.group(0)) for match in STANDALONE_INTEGER_RE.finditer(text))


def grams(text: str, size: int = 13) -> set[tuple[str, ...]]:
    tokens = tuple(WORD_NUMBER_RE.findall(text.lower()))
    return {tokens[offset : offset + size] for offset in range(len(tokens) - size + 1)}


def operation_text(operation: Sequence[Any]) -> str:
    kind = str(operation[0])
    operand = int(operation[1])
    if kind == "add":
        return f"add {operand}"
    if kind == "multiply":
        return f"multiply by {operand}"
    if kind == "subtract":
        return f"subtract {operand}"
    raise ValueError(f"unsupported C2 operation {kind!r}")


def source_text(
    initial: int, operations: Sequence[Sequence[Any]], template: str
) -> str:
    phrases = [operation_text(operation) for operation in operations]
    if template == "c2_train_a":
        return (
            f"Set register R to {initial}. Then "
            + ", then ".join(phrases)
            + ". Return R."
        )
    if template == "c2_train_b":
        numbered = " ".join(
            f"{index + 1}) {phrase}." for index, phrase in enumerate(phrases)
        )
        return f"Initial integer: {initial}. Execute in order. {numbered} Report the result."
    if template == "c2_train_c":
        return (
            f"Start={initial}; program="
            + " -> ".join(phrases)
            + "; emit final integer."
        )
    if template == "c2_train_d":
        return (
            f"A counter begins at {initial}. Apply this ordered instruction list: "
            + " | ".join(phrases)
            + ". Give the counter value."
        )
    if template == RESERVED_TEMPLATE_ID:
        opcode = {"add": "A", "multiply": "M", "subtract": "S"}
        encoded = "/".join(
            f"{opcode[str(operation[0])]}{int(operation[1])}"
            for operation in operations
        )
        return f"C2-LDG::I={initial}::OPS={encoded}::HALT=VALUE"
    raise ValueError(f"unsupported C2 source template {template!r}")


def packet_text(state: int, operations: Sequence[Sequence[Any]]) -> str:
    opcode = {"add": "ADD", "multiply": "MUL", "subtract": "SUB"}
    encoded = ",".join(
        f"{opcode[str(operation[0])]}:{int(operation[1])}" for operation in operations
    )
    return f"<C2P|S={int(state)}|R={encoded}>"


def answer_text(value: int) -> str:
    return f"<C2A|V={int(value)}>"


def compiler_text(source: str) -> str:
    return (
        "Compile this arithmetic source into the exact C2 packet grammar.\n"
        f"Source:\n{source}\nPacket:\n"
    )


def updater_text(packet: str, observed: int) -> str:
    return (
        "Advance exactly one C2 packet step using the supplied observation.\n"
        f"Packet:\n{packet}\nObservation: {int(observed)}\nNext:\n"
    )


def sequence_candidates(
    length: int, required: tuple[str, str] | None = None
) -> tuple[tuple[str, ...], ...]:
    candidates: list[tuple[str, ...]] = [()]
    for _ in range(length):
        candidates = [
            prefix + (kind,) for prefix in candidates for kind in OPERATION_TYPES
        ]
    result = []
    for candidate in candidates:
        held = [
            pair for pair in zip(candidate, candidate[1:]) if pair in HELD_OUT_BIGRAMS
        ]
        if required is None and not held:
            result.append(candidate)
        elif required is not None and held == [required]:
            result.append(candidate)
    if not result:
        raise RuntimeError("auditor found no valid C2 operation sequences")
    return tuple(result)


def random_operand(
    rng: random.Random,
    kind: str,
    geometry: Geometry,
    *,
    ood: bool,
    width: int | None = None,
) -> int:
    if ood:
        low, high = (
            geometry.ood_multiply if kind == "multiply" else geometry.ood_add_sub
        )
    else:
        low, high = (
            geometry.normal_multiply if kind == "multiply" else geometry.normal_add_sub
        )
    if width is not None:
        low = max(low, 10 ** (width - 1) if width > 1 else 0)
        high = min(high, 10**width - 1)
        if low > high:
            raise ValueError(f"auditor cannot sample {kind} at width {width}")
    return rng.randint(low, high)


def random_program(
    rng: random.Random,
    operation_kinds: Sequence[str],
    geometry: Geometry,
    *,
    ood: bool = False,
    exact_widths: Sequence[int] | None = None,
) -> tuple[int, list[list[Any]]]:
    if exact_widths is None:
        low, high = geometry.ood_initial if ood else geometry.normal_initial
        initial = rng.randint(low, high)
        operations = [
            [kind, random_operand(rng, kind, geometry, ood=ood)]
            for kind in operation_kinds
        ]
    else:
        low = max(geometry.normal_initial[0], 10 ** (int(exact_widths[0]) - 1))
        high = min(geometry.normal_initial[1], 10 ** int(exact_widths[0]) - 1)
        if low > high:
            raise ValueError("auditor cannot sample requested initial width")
        initial = rng.randint(low, high)
        operations = [
            [
                kind,
                random_operand(
                    rng, kind, geometry, ood=False, width=int(operand_width)
                ),
            ]
            for kind, operand_width in zip(operation_kinds, exact_widths[1:])
        ]
    return initial, operations


def geometry_record(geometry: Geometry) -> dict[str, Any]:
    return {
        "held_out_bigrams": [list(pair) for pair in HELD_OUT_BIGRAMS],
        "normal_ranges": {
            "add_sub_operand": list(geometry.normal_add_sub),
            "initial_state": list(geometry.normal_initial),
            "multiply_operand": list(geometry.normal_multiply),
        },
        "ood_ranges": {
            "add_sub_operand": list(geometry.ood_add_sub),
            "initial_state": list(geometry.ood_initial),
            "multiply_operand": list(geometry.ood_multiply),
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
            str(key): value for key, value in sorted(geometry.length_counts.items())
        },
        "training_template_ids": list(TRAIN_TEMPLATE_IDS),
    }


def replay_board_rows(
    *, seed: int, geometry: Geometry, id_prefix: str
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    accepted: list[dict[str, Any]] = []
    program_seen: set[tuple[Any, ...]] = set()
    source_seen: set[str] = set()
    packet_seen: set[str] = set()
    states_seen: set[tuple[int, ...]] = set()
    answer_seen: set[int] = set()
    for stratum in STRATUM_ORDER:
        for local in range(geometry.per_stratum):
            for _ in range(100_000):
                if stratum == "renderer_ood":
                    selected_kinds = rng.choice(sequence_candidates(3))
                    template = RESERVED_TEMPLATE_ID
                    initial, operations = random_program(rng, selected_kinds, geometry)
                elif stratum == "value_ood":
                    selected_kinds = rng.choice(sequence_candidates(3))
                    template = TRAIN_TEMPLATE_IDS[local % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = random_program(
                        rng, selected_kinds, geometry, ood=True
                    )
                elif stratum == "operation_order_ood":
                    half = geometry.per_stratum // 2
                    held = HELD_OUT_BIGRAMS[0 if local < half else 1]
                    within_half = local if local < half else local - half
                    length = 3 if within_half % 2 == 0 else 4
                    selected_kinds = rng.choice(sequence_candidates(length, held))
                    template = TRAIN_TEMPLATE_IDS[local % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = random_program(rng, selected_kinds, geometry)
                else:
                    selected_kinds = rng.choice(sequence_candidates(5))
                    template = TRAIN_TEMPLATE_IDS[local % len(TRAIN_TEMPLATE_IDS)]
                    initial, operations = random_program(rng, selected_kinds, geometry)
                states = replay_states(initial, operations)
                if min(states) <= 0:
                    continue
                rendered_source = source_text(initial, operations, template)
                rendered_packet = packet_text(initial, operations)
                program_key = signature(initial, operations)
                if (
                    program_key in program_seen
                    or rendered_source in source_seen
                    or rendered_packet in packet_seen
                    or states in states_seen
                    or states[-1] in answer_seen
                ):
                    continue
                accepted.append(
                    {
                        "answer": states[-1],
                        "id": f"{id_prefix}{stratum}_{local:03d}",
                        "initial_state": initial,
                        "operations": [list(operation) for operation in operations],
                        "packet": rendered_packet,
                        "source": rendered_source,
                        "stratum": stratum,
                        "template_id": template,
                        "trajectory": list(states),
                    }
                )
                program_seen.add(program_key)
                source_seen.add(rendered_source)
                packet_seen.add(rendered_packet)
                states_seen.add(states)
                answer_seen.add(states[-1])
                break
            else:
                raise RuntimeError(
                    f"auditor could not replay board row {stratum}/{local}"
                )
    return accepted


def expected_production_board(custody: Custody) -> dict[str, Any]:
    rows = replay_board_rows(
        seed=custody.seed_integers["board"],
        geometry=PRODUCTION_GEOMETRY,
        id_prefix="rsp_c2_",
    )
    geometry = geometry_record(PRODUCTION_GEOMETRY)
    return {
        "case_count": len(rows),
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "prerequisite_receipt_sha256": custody.prerequisite_sha256,
            "provenance_receipt_sha256": custody.provenance_sha256,
            "seed_receipt_sha256": custody.seed_receipt_sha256,
        },
        "geometry": geometry,
        "geometry_sha256": digest(canonical_json_bytes(geometry)),
        "per_stratum": PER_STRATUM,
        "profile": PRODUCTION_PROFILE,
        "rows": rows,
        "rows_sha256": digest(canonical_json_bytes(rows)),
        "schema": BOARD_SCHEMA,
        "seed_commitment_sha256": custody.seed_digests["board"],
        "stratum_order": list(STRATUM_ORDER),
    }


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    values = getattr(encoded, "ids", None)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("tokenizer returned an invalid token id sequence")
    return list(values)


def program_record(
    initial: int,
    operations: Sequence[Sequence[Any]],
    template: str,
    tokenizer: Any,
    pair_id: int,
) -> dict[str, Any]:
    states = replay_states(initial, operations)
    source = source_text(initial, operations, template)
    packet = packet_text(initial, operations)
    return {
        "final_answer": states[-1],
        "initial_state": initial,
        "operations": [list(operation) for operation in operations],
        "packet": packet,
        "packet_token_count": len(tokenizer_ids(tokenizer, packet)),
        "pair_id": pair_id,
        "source": source,
        "template_id": template,
        "trajectory": list(states),
    }


def candidate_is_disjoint(
    candidate: Mapping[str, Any],
    *,
    board_answers: set[int],
    board_signatures: set[tuple[Any, ...]],
    board_sources: set[str],
    board_packets: set[str],
    board_trajectories: set[tuple[int, ...]],
    board_grams: set[tuple[str, ...]],
    used_signatures: set[tuple[Any, ...]],
    used_sources: set[str],
    used_packets: set[str],
    used_trajectories: set[tuple[int, ...]],
) -> bool:
    candidate_signature = signature(candidate["initial_state"], candidate["operations"])
    candidate_trajectory = tuple(int(value) for value in candidate["trajectory"])
    packet_values = {int(candidate["initial_state"])} | {
        int(operation[1]) for operation in candidate["operations"]
    }
    return (
        min(candidate_trajectory) > 0
        and int(candidate["final_answer"]) not in packet_values
        and not (packet_values & board_answers)
        and candidate_signature not in board_signatures
        and str(candidate["source"]) not in board_sources
        and str(candidate["packet"]) not in board_packets
        and candidate_trajectory not in board_trajectories
        and int(candidate["final_answer"]) not in board_answers
        and not (grams(str(candidate["source"])) & board_grams)
        and candidate_signature not in used_signatures
        and str(candidate["source"]) not in used_sources
        and str(candidate["packet"]) not in used_packets
        and candidate_trajectory not in used_trajectories
    )


def replay_programs(
    *,
    board_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    seed: int,
    geometry: Geometry,
    id_prefix: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    board_answers = {int(row["answer"]) for row in board_rows}
    board_signatures = {
        signature(row["initial_state"], row["operations"]) for row in board_rows
    }
    board_sources = {str(row["source"]) for row in board_rows}
    board_packets = {str(row["packet"]) for row in board_rows}
    board_trajectories = {
        tuple(int(value) for value in row["trajectory"]) for row in board_rows
    }
    board_grams: set[tuple[str, ...]] = set()
    for board_source in board_sources:
        board_grams.update(grams(board_source))
    used_signatures: set[tuple[Any, ...]] = set()
    used_sources: set[str] = set()
    used_packets: set[str] = set()
    used_trajectories: set[tuple[int, ...]] = set()
    programs: list[dict[str, Any]] = []
    pair_id = 0
    for length, count in sorted(geometry.length_counts.items()):
        if count <= 0 or count % 2:
            raise ValueError("auditor requires positive even training counts")
        for local_pair in range(count // 2):
            for _ in range(20_000):
                selected_kinds = rng.choice(sequence_candidates(length))
                template = TRAIN_TEMPLATE_IDS[local_pair % len(TRAIN_TEMPLATE_IDS)]
                initial, operations = random_program(rng, selected_kinds, geometry)
                first = program_record(
                    initial, operations, template, tokenizer, pair_id
                )
                if not candidate_is_disjoint(
                    first,
                    board_answers=board_answers,
                    board_signatures=board_signatures,
                    board_sources=board_sources,
                    board_packets=board_packets,
                    board_trajectories=board_trajectories,
                    board_grams=board_grams,
                    used_signatures=used_signatures,
                    used_sources=used_sources,
                    used_packets=used_packets,
                    used_trajectories=used_trajectories,
                ):
                    continue
                first_widths = widths(first["initial_state"], first["operations"])
                first_signature = signature(first["initial_state"], first["operations"])
                for _ in range(20_000):
                    partner_initial, partner_operations = random_program(
                        rng,
                        selected_kinds,
                        geometry,
                        exact_widths=first_widths,
                    )
                    second = program_record(
                        partner_initial,
                        partner_operations,
                        template,
                        tokenizer,
                        pair_id,
                    )
                    if (
                        second["packet_token_count"] != first["packet_token_count"]
                        or len(str(second["final_answer"]))
                        != len(str(first["final_answer"]))
                        or second["final_answer"] == first["final_answer"]
                        or int(first["final_answer"])
                        in integers_in(str(second["packet"]))
                        or int(second["final_answer"])
                        in integers_in(str(first["packet"]))
                        or signature(partner_initial, partner_operations)
                        == first_signature
                    ):
                        continue
                    if not candidate_is_disjoint(
                        second,
                        board_answers=board_answers,
                        board_signatures=board_signatures,
                        board_sources=board_sources,
                        board_packets=board_packets,
                        board_trajectories=board_trajectories,
                        board_grams=board_grams,
                        used_signatures=used_signatures | {first_signature},
                        used_sources=used_sources | {str(first["source"])},
                        used_packets=used_packets | {str(first["packet"])},
                        used_trajectories=used_trajectories
                        | {tuple(int(value) for value in first["trajectory"])},
                    ):
                        continue
                    for accepted in (first, second):
                        used_signatures.add(
                            signature(accepted["initial_state"], accepted["operations"])
                        )
                        used_sources.add(str(accepted["source"]))
                        used_packets.add(str(accepted["packet"]))
                        used_trajectories.add(
                            tuple(int(value) for value in accepted["trajectory"])
                        )
                        programs.append(accepted)
                    pair_id += 1
                    break
                else:
                    continue
                break
            else:
                raise RuntimeError(
                    f"auditor could not replay matched training pair length={length}"
                )
    for index, program in enumerate(programs):
        program["id"] = f"{id_prefix}train_{index:04d}"
    return programs


def matching_key(program: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        len(program["operations"]),
        kinds(program["operations"]),
        str(program["template_id"]),
        widths(program["initial_state"], program["operations"]),
        int(program["packet_token_count"]),
        len(str(program["final_answer"])),
    )


def replay_sham_mapping(
    programs: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[int, ...]:
    rng = random.Random(seed)
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, program in enumerate(programs):
        buckets[matching_key(program)].append(index)
    mapping = [-1] * len(programs)
    for key in sorted(buckets):
        recipients = list(buckets[key])
        if len(recipients) < 2:
            raise RuntimeError(f"auditor found singleton sham stratum {key!r}")
        rng.shuffle(recipients)
        preferences: dict[int, list[int]] = {}
        for recipient in recipients:
            choices = [
                donor
                for donor in recipients
                if donor != recipient
                and programs[recipient]["final_answer"]
                != programs[donor]["final_answer"]
                and int(programs[recipient]["final_answer"])
                not in integers_in(str(programs[donor]["packet"]))
            ]
            rng.shuffle(choices)
            preferences[recipient] = choices
        donor_owner: dict[int, int] = {}

        def place(recipient: int, visited: set[int]) -> bool:
            for donor in preferences[recipient]:
                if donor in visited:
                    continue
                visited.add(donor)
                previous = donor_owner.get(donor)
                if previous is None or place(previous, visited):
                    donor_owner[donor] = recipient
                    return True
            return False

        if any(not place(recipient, set()) for recipient in recipients):
            raise RuntimeError(f"auditor cannot construct sham stratum {key!r}")
        for donor, recipient in donor_owner.items():
            mapping[recipient] = donor
    if sorted(mapping) != list(range(len(programs))) or any(
        recipient == donor for recipient, donor in enumerate(mapping)
    ):
        raise RuntimeError("auditor sham mapping is not a complete derangement")
    return tuple(mapping)


def random_false_pair(
    rng: random.Random, operation: Sequence[Any], forbidden: set[int]
) -> tuple[int, int]:
    for _ in range(100_000):
        state = rng.randint(1_000, 9_999)
        observed = rng.randint(1_000, 9_999)
        if state in forbidden or observed in forbidden:
            continue
        if observed == execute(state, operation):
            continue
        return state, observed
    raise RuntimeError("auditor could not replay a false observation")


def replay_training_rows(
    *,
    programs: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    mapping: Sequence[int],
    observation_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(observation_seed)
    board_answers = {int(row["answer"]) for row in board_rows}
    treatment: list[dict[str, Any]] = []
    sham: list[dict[str, Any]] = []
    for recipient, program in enumerate(programs):
        prompt = compiler_text(str(program["source"]))
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
        donor = programs[mapping[recipient]]
        sham.append(
            {
                **common,
                "response": donor["packet"],
                "response_program_id": donor["id"],
            }
        )
        forbidden = board_answers | {int(value) for value in program["trajectory"]}
        forbidden.update(int(operation[1]) for operation in program["operations"])
        for step, operation in enumerate(program["operations"]):
            state, observed = random_false_pair(rng, operation, forbidden)
            prompt = updater_text(
                packet_text(state, program["operations"][step:]), observed
            )
            remaining = program["operations"][step + 1 :]
            response = (
                packet_text(observed, remaining) if remaining else answer_text(observed)
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


def token_ledger(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, eos_id: int
) -> dict[str, Any]:
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("auditor received an invalid EOS id")
    masks = bytearray()
    row_lengths: list[int] = []
    prompts: list[str] = []
    response_counts: Counter[int] = Counter()
    prompt_total = 0
    response_total = 0
    supervised_total = 0
    for row in rows:
        prompt = str(row["completion_prompt"])
        response = str(row["response"]).rstrip()
        prompt_ids = tokenizer_ids(tokenizer, prompt)
        response_ids = tokenizer_ids(tokenizer, response)
        length = len(prompt_ids) + len(response_ids) + 1
        if length > PACK_LENGTH:
            raise RuntimeError("auditor found a row beyond the fixed pack length")
        row_lengths.append(length)
        prompts.append(prompt)
        prompt_total += len(prompt_ids)
        response_total += len(response_ids)
        supervised_total += len(response_ids) + 1
        masks.extend(b"\0" * len(prompt_ids))
        masks.extend(b"\1" * (len(response_ids) + 1))
        response_counts.update(response_ids)
    full_count = len(masks)
    packed_count = max(0, (full_count - 2) // PACK_LENGTH)
    forward_count = packed_count * PACK_LENGTH
    packed_mask = bytes(masks[1 : forward_count + 1])
    multiset = [[token, count] for token, count in sorted(response_counts.items())]
    return {
        "compiler_rows": sum(row["kind"] == "compiler" for row in rows),
        "discarded_token_count": full_count - forward_count,
        "example_count": len(rows),
        "full_token_count": full_count,
        "packed_forward_positions_sha256": digest(
            canonical_json_bytes([0, forward_count, PACK_LENGTH])
        ),
        "packed_sequence_count": packed_count,
        "packed_supervision_geometry_sha256": digest(packed_mask),
        "prompt_order_sha256": digest(canonical_json_bytes(prompts)),
        "prompt_token_count": prompt_total,
        "response_token_count": response_total,
        "response_token_multiset_sha256": digest(canonical_json_bytes(multiset)),
        "row_encoded_lengths_sha256": digest(canonical_json_bytes(row_lengths)),
        "supervised_target_token_count": supervised_total,
        "updater_rows": sum(row["kind"] == "updater" for row in rows),
    }


def replay_data(
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
    programs = replay_programs(
        board_rows=board["rows"],
        tokenizer=tokenizer,
        seed=training_seed,
        geometry=geometry,
        id_prefix=id_prefix,
    )
    mapping = replay_sham_mapping(programs, seed=sham_seed)
    treatment, sham = replay_training_rows(
        programs=programs,
        board_rows=board["rows"],
        mapping=mapping,
        observation_seed=observation_seed,
    )
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    treatment_ledger = token_ledger(treatment, tokenizer, int(eos))
    sham_ledger = token_ledger(sham, tokenizer, int(eos))
    return programs, treatment, sham, mapping, treatment_ledger, sham_ledger


class ToyHashTokenizer:
    class Encoding:
        def __init__(self, ids: list[int]):
            self.ids = ids

    def encode(self, text: str) -> Encoding:
        return self.Encoding(
            [
                int.from_bytes(
                    hashlib.sha256(token.encode("ascii")).digest()[:4], "big"
                )
                for token in LEXEME_RE.findall(text)
            ]
        )

    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<|endoftext|>" else None


def toy_geometry(
    label: str,
    *,
    per_stratum: int,
    length_counts: Mapping[int, int],
) -> Geometry:
    if not isinstance(label, str) or not label.startswith(TOY_PREFIX):
        raise ValueError(f"toy label must begin with {TOY_PREFIX}")
    counts = dict(length_counts)
    if per_stratum <= 0 or per_stratum >= PER_STRATUM or per_stratum % 2:
        raise ValueError("toy per-stratum count is not safely sub-production")
    if not counts or any(
        count <= 0 or count >= TRAIN_LENGTH_COUNTS.get(length, 0) or count % 2
        for length, count in counts.items()
    ):
        raise ValueError("toy training counts are not safely sub-production")
    return Geometry(label, per_stratum, counts)


def toy_seed(label: str, purpose: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{label}:{purpose}".encode("ascii")).digest(), "big"
    )


def parse_packet(value: str) -> tuple[int, list[list[Any]]]:
    match = PACKET_RE.fullmatch(value)
    if match is None:
        raise ValueError("packet grammar mismatch")
    inverse = {"ADD": "add", "MUL": "multiply", "SUB": "subtract"}
    operations = []
    for item in match.group("plan").split(","):
        opcode, operand = item.split(":", 1)
        operations.append([inverse[opcode], int(operand)])
    return int(match.group("state")), operations


def board_failures(board: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
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
        failures.append("board_schema")
    if board.get("profile") != PRODUCTION_PROFILE:
        failures.append("board_profile")
    rows = board.get("rows")
    if not isinstance(rows, list):
        return failures + ["board_rows_schema"]
    if (
        board.get("case_count") != 256
        or len(rows) != 256
        or board.get("per_stratum") != 64
    ):
        failures.append("board_count")
    if board.get("stratum_order") != list(STRATUM_ORDER):
        failures.append("board_stratum_order")
    expected_geometry = geometry_record(PRODUCTION_GEOMETRY)
    if board.get("geometry") != expected_geometry:
        failures.append("board_geometry")
    if board.get("geometry_sha256") != digest(
        canonical_json_bytes(board.get("geometry"))
    ):
        failures.append("board_geometry_hash")
    if board.get("rows_sha256") != digest(canonical_json_bytes(rows)):
        failures.append("board_rows_hash")
    failures.extend(
        board_row_failures(rows, expected_per_stratum=64, id_prefix="rsp_c2_")
    )
    return failures


def board_row_failures(
    rows: Sequence[Any], *, expected_per_stratum: int, id_prefix: str
) -> list[str]:
    failures: set[str] = set()
    row_keys = {
        "answer",
        "id",
        "initial_state",
        "operations",
        "packet",
        "source",
        "stratum",
        "template_id",
        "trajectory",
    }
    signatures: list[tuple[Any, ...]] = []
    sources: list[str] = []
    packets: list[str] = []
    trajectories: list[tuple[int, ...]] = []
    answers: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != row_keys:
            failures.add("board_row_schema")
            continue
        try:
            initial = row["initial_state"]
            if (
                isinstance(initial, bool)
                or not isinstance(initial, int)
                or initial <= 0
            ):
                raise ValueError("initial")
            if isinstance(row["answer"], bool) or not isinstance(row["answer"], int):
                raise ValueError("answer")
            if not isinstance(row["trajectory"], list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in row["trajectory"]
            ):
                raise ValueError("trajectory")
            if any(
                not isinstance(row[key], str)
                for key in ("id", "packet", "source", "stratum", "template_id")
            ):
                raise ValueError("string fields")
            operations = row["operations"]
            if not isinstance(operations, list) or not operations:
                raise ValueError("operations")
            for operation in operations:
                if (
                    not isinstance(operation, list)
                    or len(operation) != 2
                    or operation[0] not in OPERATION_TYPES
                    or isinstance(operation[1], bool)
                    or not isinstance(operation[1], int)
                    or operation[1] <= 0
                ):
                    raise ValueError("operation")
            states = replay_states(initial, operations)
            if min(states) <= 0 or row["trajectory"] != list(states):
                failures.add("board_trajectory")
            if row["answer"] != states[-1]:
                failures.add("board_answer")
            if row["packet"] != packet_text(initial, operations):
                failures.add("board_packet")
            if row["source"] != source_text(
                initial, operations, str(row["template_id"])
            ):
                failures.add("board_source")
            stratum = str(row["stratum"])
            local = index % expected_per_stratum
            stratum_index = index // expected_per_stratum
            expected_stratum = (
                STRATUM_ORDER[stratum_index]
                if stratum_index < len(STRATUM_ORDER)
                else None
            )
            if (
                expected_stratum is None
                or stratum != expected_stratum
                or row["id"] != (f"{id_prefix}{expected_stratum}_{local:03d}")
            ):
                failures.add("board_case_order")
            operation_kinds = kinds(operations)
            held = [
                pair
                for pair in zip(operation_kinds, operation_kinds[1:])
                if pair in HELD_OUT_BIGRAMS
            ]
            if stratum == "renderer_ood":
                valid = (
                    len(operations) == 3
                    and row["template_id"] == RESERVED_TEMPLATE_ID
                    and not held
                    and PRODUCTION_GEOMETRY.normal_initial[0]
                    <= initial
                    <= PRODUCTION_GEOMETRY.normal_initial[1]
                )
            elif stratum == "value_ood":
                valid = (
                    len(operations) == 3
                    and row["template_id"] in TRAIN_TEMPLATE_IDS
                    and not held
                    and PRODUCTION_GEOMETRY.ood_initial[0]
                    <= initial
                    <= PRODUCTION_GEOMETRY.ood_initial[1]
                )
            elif stratum == "operation_order_ood":
                valid = (
                    len(operations) in (3, 4)
                    and row["template_id"] in TRAIN_TEMPLATE_IDS
                    and len(held) == 1
                )
            elif stratum == "length_ood":
                valid = (
                    len(operations) == 5
                    and row["template_id"] in TRAIN_TEMPLATE_IDS
                    and not held
                )
            else:
                valid = False
            if not valid:
                failures.add("board_stratum_contract")
            if stratum != "value_ood":
                for kind, operand in operations:
                    bounds = (
                        PRODUCTION_GEOMETRY.normal_multiply
                        if kind == "multiply"
                        else PRODUCTION_GEOMETRY.normal_add_sub
                    )
                    if not bounds[0] <= operand <= bounds[1]:
                        failures.add("board_value_range")
            else:
                for kind, operand in operations:
                    bounds = (
                        PRODUCTION_GEOMETRY.ood_multiply
                        if kind == "multiply"
                        else PRODUCTION_GEOMETRY.ood_add_sub
                    )
                    if not bounds[0] <= operand <= bounds[1]:
                        failures.add("board_value_range")
            signatures.append(signature(initial, operations))
            sources.append(str(row["source"]))
            packets.append(str(row["packet"]))
            trajectories.append(states)
            answers.append(int(row["answer"]))
        except (KeyError, TypeError, ValueError):
            failures.add("board_row_semantics")
    if any(
        len(values) != len(set(values))
        for values in (signatures, sources, packets, trajectories, answers)
    ):
        failures.add("board_uniqueness")
    counts = Counter(row.get("stratum") for row in rows if isinstance(row, dict))
    if counts != Counter({name: expected_per_stratum for name in STRATUM_ORDER}):
        failures.add("board_stratum_counts")
    return sorted(failures)


def training_disjointness_failures(
    board_rows: Sequence[Mapping[str, Any]], programs: Sequence[Mapping[str, Any]]
) -> list[str]:
    board_answers = {int(row["answer"]) for row in board_rows}
    board_signatures = {
        signature(row["initial_state"], row["operations"]) for row in board_rows
    }
    board_sources = {str(row["source"]) for row in board_rows}
    board_packets = {str(row["packet"]) for row in board_rows}
    board_trajectories = {
        tuple(int(value) for value in row["trajectory"]) for row in board_rows
    }
    board_grams: set[tuple[str, ...]] = set()
    for source in board_sources:
        board_grams.update(grams(source))
    program_signatures: list[tuple[Any, ...]] = []
    program_sources: list[str] = []
    program_packets: list[str] = []
    program_trajectories: list[tuple[int, ...]] = []
    failures: set[str] = set()
    for program in programs:
        program_signature = signature(program["initial_state"], program["operations"])
        program_trajectory = tuple(int(value) for value in program["trajectory"])
        packet_values = {int(program["initial_state"])} | {
            int(operation[1]) for operation in program["operations"]
        }
        if (
            program_signature in board_signatures
            or str(program["source"]) in board_sources
            or str(program["packet"]) in board_packets
            or program_trajectory in board_trajectories
            or int(program["final_answer"]) in board_answers
            or packet_values & board_answers
            or grams(str(program["source"])) & board_grams
        ):
            failures.add("board_training_disjointness")
        if (
            int(program["final_answer"]) in packet_values
            or min(program_trajectory) <= 0
        ):
            failures.add("training_semantics")
        program_signatures.append(program_signature)
        program_sources.append(str(program["source"]))
        program_packets.append(str(program["packet"]))
        program_trajectories.append(program_trajectory)
    if any(
        len(values) != len(set(values))
        for values in (
            program_signatures,
            program_sources,
            program_packets,
            program_trajectories,
        )
    ):
        failures.add("training_uniqueness")
    return sorted(failures)


def row_schema_failures(rows: Sequence[Any]) -> list[str]:
    failures: set[str] = set()
    compiler_keys = {
        "completion_prompt",
        "id",
        "kind",
        "program_id",
        "question",
        "response",
        "response_program_id",
        "schema",
        "training_group",
    }
    updater_keys = {
        "completion_prompt",
        "id",
        "kind",
        "program_id",
        "question",
        "response",
        "schema",
        "step",
        "training_group",
    }
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            failures.add("row_schema")
            continue
        expected = compiler_keys if row.get("kind") == "compiler" else updater_keys
        if (
            set(row) != expected
            or row.get("schema") != ROW_SCHEMA
            or row.get("training_group") != "rsp_c2"
            or row.get("question") != row.get("completion_prompt")
        ):
            failures.add("row_schema")
        if row.get("kind") not in {"compiler", "updater"}:
            failures.add("row_kind")
        if isinstance(row.get("id"), str):
            ids.append(row["id"])
    if len(ids) != len(set(ids)):
        failures.add("row_id_uniqueness")
    return sorted(failures)


def sham_failures(
    treatment: Sequence[Mapping[str, Any]],
    sham: Sequence[Mapping[str, Any]],
    programs: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures: set[str] = set()
    if len(treatment) != len(sham):
        return ["row_count_parity", "sham_derangement"]
    treatment_compilers = [row for row in treatment if row.get("kind") == "compiler"]
    sham_compilers = [row for row in sham if row.get("kind") == "compiler"]
    if len(treatment_compilers) != len(programs) or len(sham_compilers) != len(
        programs
    ):
        failures.add("sham_derangement")
        return sorted(failures)
    by_id = {str(program["id"]): index for index, program in enumerate(programs)}
    donors: list[int] = []
    for recipient, (treatment_row, sham_row) in enumerate(
        zip(treatment_compilers, sham_compilers)
    ):
        recipient_program = programs[recipient]
        donor_id = sham_row.get("response_program_id")
        donor_index = by_id.get(str(donor_id), -1)
        if (
            treatment_row.get("program_id") != recipient_program["id"]
            or treatment_row.get("response_program_id") != recipient_program["id"]
            or treatment_row.get("response") != recipient_program["packet"]
            or sham_row.get("program_id") != recipient_program["id"]
            or donor_index < 0
        ):
            failures.add("sham_derangement")
            continue
        donor = programs[donor_index]
        donors.append(donor_index)
        if (
            donor_index == recipient
            or sham_row.get("response") != donor["packet"]
            or matching_key(donor) != matching_key(recipient_program)
            or donor["final_answer"] == recipient_program["final_answer"]
            or int(recipient_program["final_answer"])
            in integers_in(str(donor["packet"]))
        ):
            failures.add("sham_derangement")
    if sorted(donors) != list(range(len(programs))):
        failures.add("sham_derangement")
    treatment_updaters = [row for row in treatment if row.get("kind") == "updater"]
    sham_updaters = [row for row in sham if row.get("kind") == "updater"]
    if jsonl_bytes(treatment_updaters) != jsonl_bytes(sham_updaters):
        failures.add("updater_byte_parity")
    if [row.get("completion_prompt") for row in treatment] != [
        row.get("completion_prompt") for row in sham
    ]:
        failures.add("prompt_order_parity")
    return sorted(failures)


def updater_observation_failures(
    rows: Sequence[Mapping[str, Any]],
    programs: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    by_id = {str(program["id"]): program for program in programs}
    board_answers = {int(row["answer"]) for row in board_rows}
    for row in rows:
        if row.get("kind") != "updater":
            continue
        try:
            program = by_id[str(row["program_id"])]
            step = row["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                raise ValueError("step")
            match = UPDATER_PROMPT_RE.fullmatch(str(row["completion_prompt"]))
            if match is None:
                raise ValueError("prompt")
            state, operations = parse_packet(match.group("packet"))
            observed = int(match.group("observed"))
            if operations != program["operations"][step:]:
                raise ValueError("residual plan")
            if observed == execute(state, operations[0]):
                raise ValueError("correct transition")
            forbidden = board_answers | {int(value) for value in program["trajectory"]}
            forbidden.update(int(operation[1]) for operation in program["operations"])
            if state in forbidden or observed in forbidden:
                raise ValueError("forbidden value")
            remaining = operations[1:]
            expected_response = (
                packet_text(observed, remaining) if remaining else answer_text(observed)
            )
            if row["response"] != expected_response:
                raise ValueError("response")
        except (IndexError, KeyError, TypeError, ValueError):
            return ["updater_observation_contract"]
    return []


def expected_toy_bundle(
    label: str,
    *,
    tokenizer: Any,
    per_stratum: int,
    length_counts: Mapping[int, int],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    geometry = toy_geometry(label, per_stratum=per_stratum, length_counts=length_counts)
    rows = replay_board_rows(
        seed=toy_seed(label, "board"),
        geometry=geometry,
        id_prefix=f"{label}_",
    )
    geometry_payload = geometry_record(geometry)
    board = {
        "case_count": len(rows),
        "fixture_label": label,
        "geometry": geometry_payload,
        "geometry_sha256": digest(canonical_json_bytes(geometry_payload)),
        "profile": label,
        "rows": rows,
        "rows_sha256": digest(canonical_json_bytes(rows)),
        "schema": "toy_only_rsp_c2_board_fixture_v1",
    }
    programs, treatment, sham, mapping, treatment_tokens, sham_tokens = replay_data(
        board=board,
        tokenizer=tokenizer,
        training_seed=toy_seed(label, "training"),
        observation_seed=toy_seed(label, "observation"),
        sham_seed=toy_seed(label, "sham"),
        geometry=geometry,
        id_prefix=f"{label}_",
    )
    manifest = {
        "fixture_label": label,
        "length_counts": {
            str(key): value for key, value in sorted(geometry.length_counts.items())
        },
        "mapping_sha256": digest(canonical_json_bytes(list(mapping))),
        "program_count": len(programs),
        "programs_sha256": digest(canonical_json_bytes(programs)),
        "schema": "toy_only_rsp_c2_generation_manifest_fixture_v1",
        "sham_sha256": digest(jsonl_bytes(sham)),
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "treatment_sha256": digest(jsonl_bytes(treatment)),
    }
    return board, treatment, sham, manifest, programs


def audit_toy_fixture_bundle(
    bundle: Mapping[str, Any],
    label: str,
    *,
    tokenizer: Any | None = None,
    per_stratum: int = 4,
    length_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Audit only a tiny fixture whose label and counts cannot be production."""

    tokenizer = ToyHashTokenizer() if tokenizer is None else tokenizer
    counts = dict(length_counts or {2: 4, 3: 8, 4: 4})
    failures: set[str] = set()
    if not label.startswith(TOY_PREFIX):
        raise ValueError(f"toy label must begin with {TOY_PREFIX}")
    if (
        set(bundle)
        != {"board", "fixture_label", "manifest", "schema", "sham", "treatment"}
        or bundle.get("schema") != "toy_only_rsp_c2_bundle_fixture_v1"
        or bundle.get("fixture_label") != label
    ):
        failures.add("bundle_schema")
    expected_board, expected_treatment, expected_sham, expected_manifest, programs = (
        expected_toy_bundle(
            label,
            tokenizer=tokenizer,
            per_stratum=per_stratum,
            length_counts=counts,
        )
    )
    board = bundle.get("board")
    treatment = bundle.get("treatment")
    sham = bundle.get("sham")
    manifest = bundle.get("manifest")
    if not isinstance(board, dict):
        failures.add("board_schema")
        board = {}
    if not isinstance(treatment, list) or not all(
        isinstance(row, dict) for row in treatment
    ):
        failures.add("treatment_schema")
        treatment = []
    if not isinstance(sham, list) or not all(isinstance(row, dict) for row in sham):
        failures.add("sham_schema")
        sham = []
    if not isinstance(manifest, dict):
        failures.add("manifest_schema")
        manifest = {}
    if board.get("schema") != "toy_only_rsp_c2_board_fixture_v1":
        failures.add("board_schema")
    if board != expected_board:
        failures.add("board_replay")
    if board.get("rows_sha256") != digest(canonical_json_bytes(board.get("rows"))):
        failures.add("board_rows_hash")
    if treatment != expected_treatment:
        failures.add("treatment_replay")
    if sham != expected_sham:
        failures.add("sham_replay")
    if manifest.get("schema") != "toy_only_rsp_c2_generation_manifest_fixture_v1":
        failures.add("manifest_schema")
    if manifest != expected_manifest:
        failures.add("manifest_replay")
    if manifest.get("treatment_sha256") != digest(jsonl_bytes(treatment)):
        failures.add("treatment_hash")
    if manifest.get("sham_sha256") != digest(jsonl_bytes(sham)):
        failures.add("sham_hash")
    failures.update(row_schema_failures(treatment))
    failures.update(row_schema_failures(sham))
    failures.update(sham_failures(treatment, sham, programs))
    failures.update(
        updater_observation_failures(treatment, programs, expected_board["rows"])
    )
    actual_board_rows = board.get("rows")
    if isinstance(actual_board_rows, list):
        failures.update(
            board_row_failures(
                actual_board_rows,
                expected_per_stratum=per_stratum,
                id_prefix=f"{label}_",
            )
        )
        try:
            failures.update(training_disjointness_failures(actual_board_rows, programs))
        except (KeyError, TypeError, ValueError):
            failures.add("board_training_disjointness")
    else:
        failures.add("board_rows_schema")
    try:
        eos = tokenizer.token_to_id("<|endoftext|>")
        if eos is None:
            raise ValueError("missing EOS")
        treatment_tokens = token_ledger(treatment, tokenizer, int(eos))
        sham_tokens = token_ledger(sham, tokenizer, int(eos))
        if treatment_tokens != sham_tokens:
            failures.add("token_parity")
        if manifest.get("token_accounting") != {
            "sham": sham_tokens,
            "treatment": treatment_tokens,
        }:
            failures.add("token_accounting")
    except (KeyError, RuntimeError, TypeError, ValueError):
        failures.add("token_accounting")
    return {
        "admitted": not failures,
        "failures": sorted(failures),
        "fixture_label": label,
        "schema": "toy_only_rsp_c2_admission_audit_fixture_v1",
    }


def custody_block(custody: Custody) -> dict[str, str]:
    return {
        "prerequisite_receipt_sha256": custody.prerequisite_sha256,
        "provenance_receipt_sha256": custody.provenance_sha256,
        "seed_receipt_sha256": custody.seed_receipt_sha256,
    }


def audit_board_payload(
    board: Mapping[str, Any],
    raw: bytes,
    *,
    expected_artifact_sha256: str,
    custody: Custody,
) -> dict[str, Any]:
    expected_artifact_sha256 = require_hash(
        expected_artifact_sha256, "expected board artifact SHA-256"
    )
    failures = set(board_failures(board))
    observed_artifact_sha256 = digest(raw)
    if observed_artifact_sha256 != expected_artifact_sha256:
        failures.add("board_artifact_hash")
    expected = expected_production_board(custody)
    if board != expected or raw != canonical_json_bytes(expected):
        failures.add("board_replay")
    if board.get("seed_commitment_sha256") != custody.seed_digests["board"]:
        failures.add("board_seed_commitment")
    expected_custody = {
        "freeze_commit": custody.provenance["freeze_commit"],
        **custody_block(custody),
    }
    if board.get("custody") != expected_custody:
        failures.add("board_custody")
    rows = board.get("rows") if isinstance(board.get("rows"), list) else []
    return {
        "admitted": not failures,
        "artifacts": {
            "board_rows_sha256": digest(canonical_json_bytes(rows)),
            "board_sha256": observed_artifact_sha256,
        },
        "custody": custody_block(custody),
        "failures": sorted(failures),
        "replay": {
            "case_count": len(rows),
            "expected_board_sha256": digest(canonical_json_bytes(expected)),
            "expected_rows_sha256": expected["rows_sha256"],
            "strata": dict(
                sorted(
                    Counter(
                        row.get("stratum") for row in rows if isinstance(row, dict)
                    ).items()
                )
            ),
        },
        "schema": BOARD_AUDIT_SCHEMA,
    }


def validate_board_audit_receipt(
    audit: Mapping[str, Any],
    *,
    board_sha256: str,
    board_rows_sha256: str,
    custody: Custody,
) -> list[str]:
    expected_keys = {"admitted", "artifacts", "custody", "failures", "replay", "schema"}
    failures = []
    if set(audit) != expected_keys or audit.get("schema") != BOARD_AUDIT_SCHEMA:
        failures.append("board_audit_schema")
    if audit.get("admitted") is not True or audit.get("failures") != []:
        failures.append("board_audit_admission")
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("board_sha256") != board_sha256:
        failures.append("board_audit_board_hash")
    elif artifacts.get("board_rows_sha256") != board_rows_sha256:
        failures.append("board_audit_rows_hash")
    if audit.get("custody") != custody_block(custody):
        failures.append("board_audit_custody")
    return failures


def expected_manifest(
    *,
    board: Mapping[str, Any],
    board_sha256: str,
    board_audit_sha256: str,
    generation_invocation_sha256: str,
    custody: Custody,
    programs: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    sham: Sequence[Mapping[str, Any]],
    mapping: Sequence[int],
    treatment_tokens: Mapping[str, Any],
    sham_tokens: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_raw = jsonl_bytes(treatment)
    sham_raw = jsonl_bytes(sham)
    strata = Counter(matching_key(program) for program in programs)
    treatment_updaters = [row for row in treatment if row["kind"] == "updater"]
    sham_updaters = [row for row in sham if row["kind"] == "updater"]
    return {
        "artifacts": {
            "board_audit_sha256": board_audit_sha256,
            "board_rows_sha256": board["rows_sha256"],
            "board_sha256": board_sha256,
            "sham_rows": len(sham),
            "sham_sha256": digest(sham_raw),
            "treatment_rows": len(treatment),
            "treatment_sha256": digest(treatment_raw),
        },
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "generation_invocation_sha256": generation_invocation_sha256,
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
        "programs_sha256": digest(canonical_json_bytes(programs)),
        "schema": MANIFEST_SCHEMA,
        "sham_contract": {
            "mapping_sha256": digest(canonical_json_bytes(list(mapping))),
            "minimum_stratum_size": min(strata.values()),
            "stratum_count": len(strata),
        },
        "token_accounting": {"sham": sham_tokens, "treatment": treatment_tokens},
        "token_parity": {
            "all_locked_fields_equal": treatment_tokens == sham_tokens,
            "updater_rows_byte_identical": jsonl_bytes(treatment_updaters)
            == jsonl_bytes(sham_updaters),
        },
    }


def audit_data_payloads(
    *,
    board: Mapping[str, Any],
    board_raw: bytes,
    board_sha256: str,
    board_audit: Mapping[str, Any],
    board_audit_raw: bytes,
    board_audit_sha256: str,
    generation_invocation_raw: bytes,
    generation_invocation_sha256: str,
    treatment: Sequence[Mapping[str, Any]],
    treatment_raw: bytes,
    treatment_sha256: str,
    sham: Sequence[Mapping[str, Any]],
    sham_raw: bytes,
    sham_sha256: str,
    manifest: Mapping[str, Any],
    manifest_raw: bytes,
    manifest_sha256: str,
    tokenizer: Any,
    custody: Custody,
) -> dict[str, Any]:
    supplied_hashes = {
        "board": require_hash(board_sha256, "expected board SHA-256"),
        "board_audit": require_hash(board_audit_sha256, "expected board audit SHA-256"),
        "generation_invocation": require_hash(
            generation_invocation_sha256, "expected generation invocation SHA-256"
        ),
        "manifest": require_hash(manifest_sha256, "expected manifest SHA-256"),
        "sham": require_hash(sham_sha256, "expected sham SHA-256"),
        "treatment": require_hash(treatment_sha256, "expected treatment SHA-256"),
    }
    observed_hashes = {
        "board": digest(board_raw),
        "board_audit": digest(board_audit_raw),
        "generation_invocation": digest(generation_invocation_raw),
        "manifest": digest(manifest_raw),
        "sham": digest(sham_raw),
        "treatment": digest(treatment_raw),
    }
    failures: set[str] = set()
    for name in supplied_hashes:
        if supplied_hashes[name] != observed_hashes[name]:
            failures.add(f"{name}_artifact_hash")
    expected_board = expected_production_board(custody)
    if board != expected_board or board_raw != canonical_json_bytes(expected_board):
        failures.add("board_replay")
    failures.update(board_failures(board))
    failures.update(
        validate_board_audit_receipt(
            board_audit,
            board_sha256=observed_hashes["board"],
            board_rows_sha256=str(board.get("rows_sha256", "")),
            custody=custody,
        )
    )
    (
        programs,
        expected_treatment,
        expected_sham,
        mapping,
        treatment_tokens,
        sham_tokens,
    ) = replay_data(
        board=expected_board,
        tokenizer=tokenizer,
        training_seed=custody.seed_integers["training"],
        observation_seed=custody.seed_integers["observation"],
        sham_seed=custody.seed_integers["sham"],
        geometry=PRODUCTION_GEOMETRY,
        id_prefix="rsp_c2_",
    )
    if treatment != expected_treatment or treatment_raw != jsonl_bytes(
        expected_treatment
    ):
        failures.add("treatment_replay")
    if sham != expected_sham or sham_raw != jsonl_bytes(expected_sham):
        failures.add("sham_replay")
    failures.update(row_schema_failures(treatment))
    failures.update(row_schema_failures(sham))
    failures.update(sham_failures(treatment, sham, programs))
    failures.update(
        updater_observation_failures(treatment, programs, expected_board["rows"])
    )
    failures.update(training_disjointness_failures(expected_board["rows"], programs))
    try:
        eos = tokenizer.token_to_id("<|endoftext|>")
        if eos is None:
            raise ValueError("missing EOS")
        actual_treatment_tokens = token_ledger(treatment, tokenizer, int(eos))
        actual_sham_tokens = token_ledger(sham, tokenizer, int(eos))
        if actual_treatment_tokens != actual_sham_tokens:
            failures.add("token_parity")
        if manifest.get("token_accounting") != {
            "sham": actual_sham_tokens,
            "treatment": actual_treatment_tokens,
        }:
            failures.add("token_accounting")
    except (KeyError, RuntimeError, TypeError, ValueError):
        failures.add("token_accounting")
    expected = expected_manifest(
        board=expected_board,
        board_sha256=observed_hashes["board"],
        board_audit_sha256=observed_hashes["board_audit"],
        generation_invocation_sha256=observed_hashes["generation_invocation"],
        custody=custody,
        programs=programs,
        treatment=expected_treatment,
        sham=expected_sham,
        mapping=mapping,
        treatment_tokens=treatment_tokens,
        sham_tokens=sham_tokens,
    )
    if set(manifest) != set(expected) or manifest.get("schema") != MANIFEST_SCHEMA:
        failures.add("manifest_schema")
    if manifest != expected or manifest_raw != canonical_json_bytes(expected):
        failures.add("manifest_replay")
    return {
        "admitted": not failures,
        "artifacts": {
            "board_audit_sha256": observed_hashes["board_audit"],
            "board_sha256": observed_hashes["board"],
            "manifest_sha256": observed_hashes["manifest"],
            "sham_sha256": observed_hashes["sham"],
            "treatment_sha256": observed_hashes["treatment"],
        },
        "custody": custody_block(custody),
        "failures": sorted(failures),
        "replay": {
            "program_count": len(programs),
            "sham_rows": len(sham),
            "treatment_rows": len(treatment),
        },
        "schema": DATA_AUDIT_SCHEMA,
    }


def validate_generation_invocation(
    invocation: Mapping[str, Any],
    *,
    phase: str,
    custody: Custody,
    output_paths: Sequence[str | Path],
    extra_inputs: Mapping[str, str],
) -> list[str]:
    expected = {
        "custody": {
            "freeze_commit": custody.provenance["freeze_commit"],
            "prerequisite_receipt_sha256": custody.prerequisite_sha256,
            "provenance_receipt_sha256": custody.provenance_sha256,
            "runtime_receipt_sha256": custody.runtime_receipt_sha256,
            "seed_receipt_sha256": custody.seed_receipt_sha256,
            "tokenizer_sha256": custody.tokenizer_sha256,
        },
        "extra_inputs": dict(sorted(extra_inputs.items())),
        "output_paths": [
            str(Path(path).resolve(strict=False)) for path in output_paths
        ],
        "phase": phase,
        "profile": PRODUCTION_PROFILE,
        "schema": GENERATION_INVOCATION_SCHEMA,
    }
    return [] if invocation == expected else ["generation_invocation"]


def exclusive_write(path: str | Path, payload: bytes) -> str:
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
                raise OSError("short write while creating immutable C2 audit output")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    info = destination.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise PermissionError("C2 audit output is not a regular read-only file")
    return digest(payload)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    return exclusive_write(path, canonical_json_bytes(payload))


def output_preflight(paths: Sequence[str | Path]) -> list[Path]:
    outputs = [Path(path).resolve(strict=False) for path in paths]
    if len(outputs) != len(set(outputs)):
        raise ValueError("all audit output paths must be distinct")
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to reuse audit output path: {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    return outputs


def audit_invocation(
    *,
    phase: str,
    custody: Custody,
    outputs: Sequence[Path],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "custody": custody_block(custody),
        "output_paths": [str(path) for path in outputs],
        "phase": phase,
        "profile": PRODUCTION_PROFILE,
        "schema": AUDIT_INVOCATION_SCHEMA,
    }


def custody_from_args(args: argparse.Namespace) -> Custody:
    return load_custody(
        prerequisite_path=args.prerequisite_receipt,
        prerequisite_sha256=args.prerequisite_receipt_sha256,
        seed_path=args.seed_receipt,
        seed_sha256=args.seed_receipt_sha256,
        provenance_path=args.provenance_receipt,
        provenance_sha256=args.provenance_receipt_sha256,
        tokenizer_path=args.tokenizer,
        runtime_path=args.runtime_receipt,
    )


def run_board_audit(args: argparse.Namespace) -> dict[str, Any]:
    outputs = output_preflight((args.audit_invocation_out, args.audit_out))
    custody = custody_from_args(args)
    expected_board_sha256 = require_hash(args.board_sha256, "expected board SHA-256")
    expected_generation_invocation_sha256 = require_hash(
        args.generation_invocation_sha256,
        "expected board generation invocation SHA-256",
    )
    generation_invocation, generation_invocation_raw = read_json_artifact(
        args.generation_invocation, "C2 board generation invocation"
    )
    if digest(generation_invocation_raw) != expected_generation_invocation_sha256:
        raise ValueError("board generation invocation SHA-256 mismatch")
    invocation = audit_invocation(
        phase="board",
        custody=custody,
        outputs=outputs,
        artifact_hashes={
            "board_sha256": expected_board_sha256,
            "generation_invocation_sha256": expected_generation_invocation_sha256,
        },
    )
    invocation_sha256 = write_json(outputs[0], invocation)
    board, board_raw = read_json_artifact(args.board, "C2 board")
    report = audit_board_payload(
        board,
        board_raw,
        expected_artifact_sha256=expected_board_sha256,
        custody=custody,
    )
    invocation_failures = validate_generation_invocation(
        generation_invocation,
        phase="board",
        custody=custody,
        output_paths=(args.generation_invocation, args.board),
        extra_inputs={},
    )
    if invocation_failures:
        report["failures"] = sorted(set(report["failures"]) | set(invocation_failures))
        report["admitted"] = False
    audit_sha256 = write_json(outputs[1], report)
    return {
        "admitted": report["admitted"],
        "audit_invocation_sha256": invocation_sha256,
        "audit_sha256": audit_sha256,
        "failures": report["failures"],
        "schema": BOARD_AUDIT_SCHEMA,
    }


def run_data_audit(args: argparse.Namespace) -> dict[str, Any]:
    outputs = output_preflight((args.audit_invocation_out, args.audit_out))
    custody = custody_from_args(args)
    exact_hashes = {
        "board_sha256": require_hash(args.board_sha256, "expected board SHA-256"),
        "board_audit_sha256": require_hash(
            args.board_audit_sha256, "expected board audit SHA-256"
        ),
        "generation_invocation_sha256": require_hash(
            args.generation_invocation_sha256,
            "expected data generation invocation SHA-256",
        ),
        "manifest_sha256": require_hash(
            args.manifest_sha256, "expected manifest SHA-256"
        ),
        "sham_sha256": require_hash(args.sham_sha256, "expected sham SHA-256"),
        "treatment_sha256": require_hash(
            args.treatment_sha256, "expected treatment SHA-256"
        ),
    }
    invocation_payload = audit_invocation(
        phase="data",
        custody=custody,
        outputs=outputs,
        artifact_hashes=exact_hashes,
    )
    audit_invocation_sha256 = write_json(outputs[0], invocation_payload)

    board, board_raw = read_json_artifact(args.board, "C2 board")
    board_audit, board_audit_raw = read_json_artifact(
        args.board_audit, "C2 board audit"
    )
    generation_invocation, generation_invocation_raw = read_json_artifact(
        args.generation_invocation, "C2 data generation invocation"
    )
    treatment, treatment_raw = read_jsonl_artifact(args.treatment, "C2 treatment data")
    sham, sham_raw = read_jsonl_artifact(args.sham, "C2 sham data")
    manifest, manifest_raw = read_json_artifact(args.manifest, "C2 generation manifest")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(custody.tokenizer_bytes.decode("utf-8"))
    report = audit_data_payloads(
        board=board,
        board_raw=board_raw,
        board_sha256=exact_hashes["board_sha256"],
        board_audit=board_audit,
        board_audit_raw=board_audit_raw,
        board_audit_sha256=exact_hashes["board_audit_sha256"],
        generation_invocation_raw=generation_invocation_raw,
        generation_invocation_sha256=exact_hashes["generation_invocation_sha256"],
        treatment=treatment,
        treatment_raw=treatment_raw,
        treatment_sha256=exact_hashes["treatment_sha256"],
        sham=sham,
        sham_raw=sham_raw,
        sham_sha256=exact_hashes["sham_sha256"],
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_sha256=exact_hashes["manifest_sha256"],
        tokenizer=tokenizer,
        custody=custody,
    )
    generation_failures = validate_generation_invocation(
        generation_invocation,
        phase="data",
        custody=custody,
        output_paths=(
            args.generation_invocation,
            args.treatment,
            args.sham,
            args.manifest,
        ),
        extra_inputs={
            "board_audit_sha256": exact_hashes["board_audit_sha256"],
            "board_sha256": exact_hashes["board_sha256"],
        },
    )
    if generation_failures:
        report["failures"] = sorted(set(report["failures"]) | set(generation_failures))
        report["admitted"] = False
    audit_sha256 = write_json(outputs[1], report)
    return {
        "admitted": report["admitted"],
        "audit_invocation_sha256": audit_invocation_sha256,
        "audit_sha256": audit_sha256,
        "failures": report["failures"],
        "schema": DATA_AUDIT_SCHEMA,
    }


def add_custody_arguments(parser: argparse.ArgumentParser) -> None:
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
    board = commands.add_parser("board", help="one-shot independent board admission")
    add_custody_arguments(board)
    board.add_argument("--generation-invocation", required=True)
    board.add_argument("--generation-invocation-sha256", required=True)
    board.add_argument("--board", required=True)
    board.add_argument("--board-sha256", required=True)
    board.add_argument("--audit-invocation-out", required=True)
    board.add_argument("--audit-out", required=True)
    data = commands.add_parser("data", help="one-shot independent data admission")
    add_custody_arguments(data)
    data.add_argument("--board", required=True)
    data.add_argument("--board-sha256", required=True)
    data.add_argument("--board-audit", required=True)
    data.add_argument("--board-audit-sha256", required=True)
    data.add_argument("--generation-invocation", required=True)
    data.add_argument("--generation-invocation-sha256", required=True)
    data.add_argument("--treatment", required=True)
    data.add_argument("--treatment-sha256", required=True)
    data.add_argument("--sham", required=True)
    data.add_argument("--sham-sha256", required=True)
    data.add_argument("--manifest", required=True)
    data.add_argument("--manifest-sha256", required=True)
    data.add_argument("--audit-invocation-out", required=True)
    data.add_argument("--audit-out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_board_audit(args) if args.command == "board" else run_data_audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
