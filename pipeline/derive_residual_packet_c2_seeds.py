#!/usr/bin/env python3
"""Derive RSP-C2 production seeds from immutable, externally verified receipts.

This module is intentionally offline. It does not contain HTTP, socket, or
beacon-retrieval code. Production use requires read-only canonical receipts for
the prerequisite pass, remote implementation push, and dual beacon validation,
plus the exact raw NIST Beacon 2.0 response those receipts bind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "rsp_c2_seed_receipt_v2"
PREREQUISITE_SCHEMA = "source_scheduled_reasoning_confirmation_pass_v1"
FREEZE_PUSH_SCHEMA = "rsp_c2_freeze_push_receipt_v1"
BEACON_VERIFICATION_SCHEMA = "rsp_c2_nist_beacon_verification_v1"
SEED_SCHEME = "sha256_domain_separated_nist_receipts_v2"
BASE_DOMAIN = b"SHOHIN-RSP-C2-SEED-BASE-v2\0"
SEED_DOMAIN = b"SHOHIN-RSP-C2-SEED-v2\0"
MINIMUM_BEACON_DELAY_MS = 3_600_000
MAX_PUSH_OBSERVATION_LAG_MS = 300_000
NIST_PERIOD_MS = 60_000
SEED_LABELS = ("board", "training", "observation", "sham", "fit-a", "fit-b")

HEX_64_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)
OUTPUT_VALUE_RE = re.compile(r"[0-9A-Fa-f]{128}\Z", re.ASCII)
UTC_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<millis>\d{3}))?Z\Z",
    re.ASCII,
)
REMOTE_URL_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*\.git\Z",
    re.ASCII,
)
BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,126}[A-Za-z0-9]\Z", re.ASCII)
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}\Z", re.ASCII)

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
FREEZE_PUSH_KEYS = {
    "branch",
    "commit_oid",
    "observed_at",
    "observer_id",
    "observer_implementation_sha256",
    "pushed_at",
    "ref",
    "remote_ref_evidence_sha256",
    "remote_ref_oid",
    "remote_url",
    "schema",
}
BEACON_VERIFICATION_KEYS = {
    "beacon_raw_sha256",
    "certificate_id",
    "certificate_sha256",
    "chain_index",
    "output_value_sha256",
    "period_ms",
    "pulse_index",
    "schema",
    "signature_value_sha256",
    "status_code",
    "time_stamp",
    "time_stamp_ms",
    "validators",
    "validators_agree",
}
BEACON_VALIDATOR_KEYS = {
    "certificate_valid",
    "chain_valid",
    "evidence_sha256",
    "implementation_sha256",
    "raw_beacon_sha256",
    "signature_valid",
    "validated_at",
    "validator_id",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


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


def require_canonical_payload(
    payload: Mapping[str, Any], raw: bytes, label: str
) -> None:
    if raw != canonical_json_bytes(payload):
        raise ValueError(f"{label} must use canonical JSON bytes")


def read_immutable_json(
    path: str | Path, label: str, *, require_canonical: bool
) -> tuple[Mapping[str, Any], bytes]:
    source = Path(path)
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if before.st_mode & 0o222:
        raise PermissionError(f"{label} must be read-only")
    raw = source.read_bytes()
    after = source.lstat()
    identity_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"{label} changed while being read")
    payload = parse_json_bytes(raw, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    if require_canonical:
        require_canonical_payload(payload, raw, label)
    return payload, raw


def require_lower_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical lowercase hexadecimal")
    return value


def require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical nonempty identifier")
    return value


def require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")


def parse_utc_timestamp_ms(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an RFC3339 UTC string")
    match = UTC_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} must use YYYY-MM-DDTHH:MM:SS[.mmm]Z")
    millis = match.group("millis") or "000"
    try:
        parsed = datetime.strptime(match.group("date"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{label} is not a valid UTC timestamp") from error
    return int(parsed.timestamp()) * 1000 + int(millis)


def canonical_utc_timestamp(timestamp_ms: int) -> str:
    seconds, millis = divmod(timestamp_ms, 1000)
    rendered = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    return f"{rendered}.{millis:03d}Z"


def require_canonical_timestamp(value: Any, label: str) -> tuple[str, int]:
    timestamp_ms = parse_utc_timestamp_ms(value, label)
    canonical = canonical_utc_timestamp(timestamp_ms)
    if value != canonical:
        raise ValueError(f"{label} must use canonical millisecond UTC form")
    return canonical, timestamp_ms


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def validate_prerequisite_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(payload, PREREQUISITE_KEYS, "prerequisite receipt")
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
    hash_keys = (
        "confirmation_contract_sha256",
        "confirmation_result_sha256",
        "independent_score_receipt_sha256",
        "independent_scorer_sha256",
        "primary_score_receipt_sha256",
        "primary_scorer_sha256",
    )
    normalized = dict(payload)
    for key in hash_keys:
        normalized[key] = require_lower_hex(payload[key], HEX_64_RE, key)
    if normalized["primary_scorer_sha256"] == normalized["independent_scorer_sha256"]:
        raise ValueError("primary and independent scorer implementations must differ")
    if (
        normalized["primary_score_receipt_sha256"]
        == normalized["independent_score_receipt_sha256"]
    ):
        raise ValueError("primary and independent score receipts must differ")
    return normalized


def validate_freeze_push_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(payload, FREEZE_PUSH_KEYS, "freeze push receipt")
    if payload["schema"] != FREEZE_PUSH_SCHEMA:
        raise ValueError("freeze push receipt schema mismatch")
    remote_url = payload["remote_url"]
    if not isinstance(remote_url, str) or REMOTE_URL_RE.fullmatch(remote_url) is None:
        raise ValueError("freeze remote_url must be a canonical HTTPS GitHub .git URL")
    branch = payload["branch"]
    if (
        not isinstance(branch, str)
        or BRANCH_RE.fullmatch(branch) is None
        or ".." in branch
        or "@{" in branch
        or "//" in branch
        or branch.endswith((".", "/", ".lock"))
    ):
        raise ValueError("freeze branch is not canonical")
    expected_ref = f"refs/heads/{branch}"
    if payload["ref"] != expected_ref:
        raise ValueError("freeze ref does not match branch")
    commit_oid = require_lower_hex(
        payload["commit_oid"], GIT_OID_RE, "freeze commit_oid"
    )
    remote_ref_oid = require_lower_hex(
        payload["remote_ref_oid"], GIT_OID_RE, "freeze remote_ref_oid"
    )
    if commit_oid != remote_ref_oid:
        raise ValueError("freeze commit_oid does not match remotely observed ref OID")
    pushed_at, pushed_at_ms = require_canonical_timestamp(
        payload["pushed_at"], "freeze pushed_at"
    )
    observed_at, observed_at_ms = require_canonical_timestamp(
        payload["observed_at"], "freeze observed_at"
    )
    observation_lag_ms = observed_at_ms - pushed_at_ms
    if not 0 <= observation_lag_ms <= MAX_PUSH_OBSERVATION_LAG_MS:
        raise ValueError(
            "freeze observed_at must be within five minutes after pushed_at"
        )
    return {
        "branch": branch,
        "commit_oid": commit_oid,
        "observed_at": observed_at,
        "observed_at_ms": observed_at_ms,
        "observer_id": require_identifier(payload["observer_id"], "freeze observer_id"),
        "observer_implementation_sha256": require_lower_hex(
            payload["observer_implementation_sha256"],
            HEX_64_RE,
            "freeze observer implementation SHA-256",
        ),
        "pushed_at": pushed_at,
        "pushed_at_ms": pushed_at_ms,
        "ref": expected_ref,
        "remote_ref_evidence_sha256": require_lower_hex(
            payload["remote_ref_evidence_sha256"],
            HEX_64_RE,
            "freeze remote-ref evidence SHA-256",
        ),
        "remote_ref_oid": remote_ref_oid,
        "remote_url": remote_url,
    }


def validate_beacon_pulse(
    payload: Mapping[str, Any], *, freeze_pushed_at_ms: int
) -> dict[str, Any]:
    if set(payload) != {"pulse"} or not isinstance(payload["pulse"], dict):
        raise ValueError("beacon JSON must contain exactly one pulse object")
    pulse = payload["pulse"]
    required = {
        "certificateId",
        "chainIndex",
        "outputValue",
        "period",
        "pulseIndex",
        "signatureValue",
        "statusCode",
        "timeStamp",
    }
    missing = sorted(required - set(pulse))
    if missing:
        raise ValueError(f"beacon pulse is missing fields: {missing}")
    period = _positive_integer(pulse["period"], "beacon period")
    if period != NIST_PERIOD_MS:
        raise ValueError(f"beacon period must be {NIST_PERIOD_MS} milliseconds")
    status_code = _nonnegative_integer(pulse["statusCode"], "beacon statusCode")
    if status_code != 0:
        raise ValueError("beacon statusCode must be integer zero")
    timestamp_ms = parse_utc_timestamp_ms(pulse["timeStamp"], "beacon timeStamp")
    target_ms = freeze_pushed_at_ms + MINIMUM_BEACON_DELAY_MS
    delta_ms = timestamp_ms - target_ms
    if not 0 <= delta_ms < period:
        raise ValueError(
            "beacon must be the first pulse at or after freeze pushed_at plus one hour"
        )
    output_value = pulse["outputValue"]
    if (
        not isinstance(output_value, str)
        or OUTPUT_VALUE_RE.fullmatch(output_value) is None
    ):
        raise ValueError(
            "beacon outputValue must contain exactly 512 bits of hexadecimal"
        )
    signature = pulse["signatureValue"]
    certificate = pulse["certificateId"]
    if not isinstance(signature, str) or not signature:
        raise ValueError("beacon signatureValue must be present")
    if not isinstance(certificate, str) or not certificate:
        raise ValueError("beacon certificateId must be present")
    return {
        "certificate_id": certificate,
        "chain_index": _positive_integer(pulse["chainIndex"], "beacon chainIndex"),
        "first_pulse_target_ms": target_ms,
        "output_value": output_value.lower(),
        "output_value_sha256": sha256_bytes(bytes.fromhex(output_value)),
        "period_ms": period,
        "pulse_index": _positive_integer(pulse["pulseIndex"], "beacon pulseIndex"),
        "signature_value_sha256": sha256_bytes(signature.encode("utf-8")),
        "status_code": status_code,
        "time_stamp": canonical_utc_timestamp(timestamp_ms),
        "time_stamp_ms": timestamp_ms,
    }


def validate_beacon_validator(
    payload: Mapping[str, Any], *, beacon_raw_sha256: str, beacon_timestamp_ms: int
) -> dict[str, Any]:
    require_exact_keys(payload, BEACON_VALIDATOR_KEYS, "beacon validator")
    for key in ("certificate_valid", "chain_valid", "signature_valid"):
        if payload[key] is not True:
            raise ValueError(f"beacon validator requires {key}=true")
    if payload["raw_beacon_sha256"] != beacon_raw_sha256:
        raise ValueError("beacon validator raw hash does not match beacon")
    validated_at, validated_at_ms = require_canonical_timestamp(
        payload["validated_at"], "beacon validator validated_at"
    )
    if validated_at_ms < beacon_timestamp_ms:
        raise ValueError("beacon validator timestamp predates the pulse")
    return {
        "certificate_valid": True,
        "chain_valid": True,
        "evidence_sha256": require_lower_hex(
            payload["evidence_sha256"], HEX_64_RE, "beacon validator evidence SHA-256"
        ),
        "implementation_sha256": require_lower_hex(
            payload["implementation_sha256"],
            HEX_64_RE,
            "beacon validator implementation SHA-256",
        ),
        "raw_beacon_sha256": beacon_raw_sha256,
        "signature_valid": True,
        "validated_at": validated_at,
        "validator_id": require_identifier(
            payload["validator_id"], "beacon validator_id"
        ),
    }


def validate_beacon_verification_receipt(
    payload: Mapping[str, Any], *, beacon: Mapping[str, Any], beacon_raw: bytes
) -> dict[str, Any]:
    require_exact_keys(payload, BEACON_VERIFICATION_KEYS, "beacon verification receipt")
    if payload["schema"] != BEACON_VERIFICATION_SCHEMA:
        raise ValueError("beacon verification receipt schema mismatch")
    if payload["validators_agree"] is not True:
        raise ValueError("beacon verification receipt requires validators_agree=true")
    beacon_raw_sha256 = sha256_bytes(beacon_raw)
    exact_bindings = {
        "beacon_raw_sha256": beacon_raw_sha256,
        "certificate_id": beacon["certificate_id"],
        "chain_index": beacon["chain_index"],
        "output_value_sha256": beacon["output_value_sha256"],
        "period_ms": beacon["period_ms"],
        "pulse_index": beacon["pulse_index"],
        "signature_value_sha256": beacon["signature_value_sha256"],
        "status_code": beacon["status_code"],
        "time_stamp": beacon["time_stamp"],
        "time_stamp_ms": beacon["time_stamp_ms"],
    }
    for key, expected in exact_bindings.items():
        if payload[key] != expected:
            raise ValueError(f"beacon verification {key} does not match raw pulse")
    certificate_sha256 = require_lower_hex(
        payload["certificate_sha256"], HEX_64_RE, "beacon certificate SHA-256"
    )
    validators_payload = payload["validators"]
    if not isinstance(validators_payload, list) or len(validators_payload) != 2:
        raise ValueError("beacon verification requires exactly two validators")
    validators = []
    for entry in validators_payload:
        if not isinstance(entry, dict):
            raise ValueError("beacon validator entry must be an object")
        validators.append(
            validate_beacon_validator(
                entry,
                beacon_raw_sha256=beacon_raw_sha256,
                beacon_timestamp_ms=beacon["time_stamp_ms"],
            )
        )
    distinct_fields = ("validator_id", "implementation_sha256", "evidence_sha256")
    for field in distinct_fields:
        if validators[0][field] == validators[1][field]:
            raise ValueError(f"beacon validators must have distinct {field}")
    return {
        **exact_bindings,
        "certificate_sha256": certificate_sha256,
        "validators": validators,
        "validators_agree": True,
    }


def _field(value: str | int) -> bytes:
    return str(value).encode("ascii") + b"\0"


def derive_seed_receipt(
    *,
    freeze_push_payload: Mapping[str, Any],
    freeze_push_raw: bytes,
    prerequisite_payload: Mapping[str, Any],
    prerequisite_raw: bytes,
    beacon_payload: Mapping[str, Any],
    beacon_raw: bytes,
    beacon_verification_payload: Mapping[str, Any],
    beacon_verification_raw: bytes,
) -> dict[str, Any]:
    require_canonical_payload(
        freeze_push_payload, freeze_push_raw, "freeze push receipt"
    )
    require_canonical_payload(
        prerequisite_payload, prerequisite_raw, "prerequisite pass receipt"
    )
    require_canonical_payload(
        beacon_verification_payload,
        beacon_verification_raw,
        "beacon verification receipt",
    )
    freeze = validate_freeze_push_receipt(freeze_push_payload)
    prerequisite = validate_prerequisite_receipt(prerequisite_payload)
    beacon = validate_beacon_pulse(
        beacon_payload, freeze_pushed_at_ms=freeze["pushed_at_ms"]
    )
    beacon_verification = validate_beacon_verification_receipt(
        beacon_verification_payload, beacon=beacon, beacon_raw=beacon_raw
    )
    freeze_push_raw_sha256 = sha256_bytes(freeze_push_raw)
    prerequisite_raw_sha256 = sha256_bytes(prerequisite_raw)
    beacon_raw_sha256 = sha256_bytes(beacon_raw)
    beacon_verification_raw_sha256 = sha256_bytes(beacon_verification_raw)

    base_preimage = b"".join(
        (
            BASE_DOMAIN,
            _field(freeze["commit_oid"]),
            _field(freeze["remote_url"]),
            _field(freeze["ref"]),
            _field(freeze["pushed_at"]),
            _field(freeze["observed_at"]),
            _field(freeze_push_raw_sha256),
            _field(prerequisite_raw_sha256),
            _field(prerequisite["confirmation_result_sha256"]),
            _field(beacon_raw_sha256),
            _field(beacon["chain_index"]),
            _field(beacon["pulse_index"]),
            _field(beacon["time_stamp_ms"]),
            _field(beacon["output_value"]),
            _field(beacon_verification_raw_sha256),
        )
    )
    base_digest = hashlib.sha256(base_preimage).digest()
    seeds: dict[str, dict[str, str]] = {}
    for label in SEED_LABELS:
        digest = hashlib.sha256(
            SEED_DOMAIN + base_digest + b"\0" + label.encode("ascii")
        ).digest()
        seeds[label] = {
            "integer_decimal": str(int.from_bytes(digest, "big")),
            "sha256": digest.hex(),
        }

    return {
        "base_commitment_sha256": base_digest.hex(),
        "beacon": {
            "certificate_id": beacon["certificate_id"],
            "chain_index": beacon["chain_index"],
            "first_pulse_target_ms": beacon["first_pulse_target_ms"],
            "output_value": beacon["output_value"],
            "output_value_sha256": beacon["output_value_sha256"],
            "period_ms": beacon["period_ms"],
            "pulse_index": beacon["pulse_index"],
            "raw_sha256": beacon_raw_sha256,
            "signature_value_sha256": beacon["signature_value_sha256"],
            "status_code": beacon["status_code"],
            "time_stamp": beacon["time_stamp"],
            "time_stamp_ms": beacon["time_stamp_ms"],
            "verification": {
                "certificate_sha256": beacon_verification["certificate_sha256"],
                "raw_sha256": beacon_verification_raw_sha256,
                "validators": beacon_verification["validators"],
                "validators_agree": True,
            },
        },
        "freeze": {
            "branch": freeze["branch"],
            "commit_oid": freeze["commit_oid"],
            "observed_at": freeze["observed_at"],
            "observer_id": freeze["observer_id"],
            "observer_implementation_sha256": freeze["observer_implementation_sha256"],
            "pushed_at": freeze["pushed_at"],
            "raw_sha256": freeze_push_raw_sha256,
            "ref": freeze["ref"],
            "remote_ref_evidence_sha256": freeze["remote_ref_evidence_sha256"],
            "remote_ref_oid": freeze["remote_ref_oid"],
            "remote_url": freeze["remote_url"],
        },
        "prerequisite": {
            "confirmation_contract_sha256": prerequisite[
                "confirmation_contract_sha256"
            ],
            "confirmation_result_sha256": prerequisite["confirmation_result_sha256"],
            "independent_score_receipt_sha256": prerequisite[
                "independent_score_receipt_sha256"
            ],
            "independent_scorer_sha256": prerequisite["independent_scorer_sha256"],
            "primary_score_receipt_sha256": prerequisite[
                "primary_score_receipt_sha256"
            ],
            "primary_scorer_sha256": prerequisite["primary_scorer_sha256"],
            "raw_sha256": prerequisite_raw_sha256,
        },
        "schema": RECEIPT_SCHEMA,
        "seed_labels": list(SEED_LABELS),
        "seed_scheme": SEED_SCHEME,
        "seeds": seeds,
    }


def write_exclusive_readonly(path: str | Path, payload: bytes) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating seed receipt")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    os.chmod(destination, 0o444)
    return sha256_bytes(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-push-receipt", required=True)
    parser.add_argument("--prerequisite-pass-receipt", required=True)
    parser.add_argument("--beacon-json", required=True)
    parser.add_argument("--beacon-verification-receipt", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    freeze_push, freeze_push_raw = read_immutable_json(
        args.freeze_push_receipt, "freeze push receipt", require_canonical=True
    )
    prerequisite, prerequisite_raw = read_immutable_json(
        args.prerequisite_pass_receipt,
        "prerequisite pass receipt",
        require_canonical=True,
    )
    beacon, beacon_raw = read_immutable_json(
        args.beacon_json, "NIST beacon pulse", require_canonical=False
    )
    beacon_verification, beacon_verification_raw = read_immutable_json(
        args.beacon_verification_receipt,
        "beacon verification receipt",
        require_canonical=True,
    )
    receipt = derive_seed_receipt(
        freeze_push_payload=freeze_push,
        freeze_push_raw=freeze_push_raw,
        prerequisite_payload=prerequisite,
        prerequisite_raw=prerequisite_raw,
        beacon_payload=beacon,
        beacon_raw=beacon_raw,
        beacon_verification_payload=beacon_verification,
        beacon_verification_raw=beacon_verification_raw,
    )
    payload = canonical_json_bytes(receipt)
    digest = write_exclusive_readonly(args.out, payload)
    print(
        json.dumps({"receipt_sha256": digest, "schema": RECEIPT_SCHEMA}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
