#!/usr/bin/env python3
"""Create and verify the pre-outcome CTAA bootstrap-seed commitment."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ctaa_evaluation_io import canonical_json, sha256_file, write_json_once


SCHEMA = "r12_ctaa_v2_bootstrap_seed_receipt_v3"
ASSESSMENT_SCHEMA = "r12_ctaa_v2_assessment_v3"
BEACON_SCHEMA = "r12_ctaa_v2_public_beacon_receipt_v3"
BEACON_VERIFICATION_SCHEMA = "r12_ctaa_v2_nist_beacon_verification_v2"
DERIVATION_SCHEME = "sha256_domain_separated_normalized_nist_beacon_v3"
DERIVATION_LABEL = "shohin-ctaa-v2-paired-hierarchical-bootstrap"
DOMAIN = b"SHOHIN-CTAA-V2-BOOTSTRAP-SEED-v3\0"
FREEZE_SIGNATURE_DOMAIN = b"SHOHIN-CTAA-V2-SOURCE-FREEZE-v1\0"
VALIDATOR_SIGNATURE_DOMAIN = b"SHOHIN-CTAA-V2-BEACON-VALIDATOR-v1\0"
MINIMUM_BEACON_DELAY_MS = 3_600_000
MAX_FREEZE_OBSERVATION_LAG_MS = 300_000
NIST_PERIOD_MS = 60_000

# The matching private key is retained outside the repository. Validator keys
# are committed by this root before the delayed beacon pulse exists.
CUSTODY_ROOT_PUBLIC_KEY_HEX = (
    "d2ea87fee19d8069e26de16fa668bedb3a3ac607a8a2755e72e4bc33db6c303d"
)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
OUTPUT_VALUE = re.compile(r"[0-9A-Fa-f]{128}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}\Z")
REMOTE_URL = re.compile(
    r"https://github\.com/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*\.git\Z"
)
GIT_REF = re.compile(r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{1,180}\Z")
UTC_TIMESTAMP = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<millis>\d{3}))?Z\Z"
)


def _hex(value: object, label: str, pattern: re.Pattern[str] = HEX64) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"CTAA bootstrap {label} differs")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"CTAA bootstrap {label} differs")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or int(value) < 1:
        raise ValueError(f"CTAA bootstrap {label} differs")
    return int(value)


def _timestamp_ms(value: object, label: str, *, canonical: bool) -> int:
    if not isinstance(value, str) or (match := UTC_TIMESTAMP.fullmatch(value)) is None:
        raise ValueError(f"CTAA bootstrap {label} timestamp differs")
    millis = match.group("millis") or "000"
    try:
        parsed = datetime.strptime(match.group("date"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"CTAA bootstrap {label} timestamp differs") from error
    result = int(parsed.timestamp()) * 1000 + int(millis)
    normalized = (
        datetime.fromtimestamp(result // 1000, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + f".{result % 1000:03d}Z"
    )
    if canonical and value != normalized:
        raise ValueError(f"CTAA bootstrap {label} timestamp is not canonical")
    return result


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"CTAA bootstrap duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CTAA bootstrap {label} JSON differs") from error
    if not isinstance(value, dict):
        raise ValueError(f"CTAA bootstrap {label} root differs")
    return value


def _verify_signature(
    public_key_hex: str, signature_hex: object, message: bytes
) -> None:
    public_key = bytes.fromhex(_hex(public_key_hex, "public key"))
    signature = bytes.fromhex(_hex(signature_hex, "signature", HEX128))
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError) as error:
        raise ValueError("CTAA bootstrap signature verification failed") from error


def _validate_source_freeze(
    value: object,
    *,
    source_commit: str,
    custody_root_public_key_hex: str,
) -> tuple[dict[str, object], int, dict[str, str]]:
    keys = {
        "schema",
        "commit_oid",
        "remote_url",
        "ref",
        "remote_ref_oid",
        "pushed_at",
        "observed_at",
        "observer_id",
        "observer_implementation_sha256",
        "remote_ref_evidence_sha256",
        "validator_public_keys",
        "signature",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("CTAA bootstrap source-freeze schema differs")
    if value.get("schema") != "r12_ctaa_v2_source_freeze_v1":
        raise ValueError("CTAA bootstrap source-freeze version differs")
    if (
        value.get("commit_oid") != source_commit
        or value.get("remote_ref_oid") != source_commit
    ):
        raise ValueError("CTAA bootstrap source-freeze commit differs")
    if (
        not isinstance(value.get("remote_url"), str)
        or REMOTE_URL.fullmatch(str(value["remote_url"])) is None
    ):
        raise ValueError("CTAA bootstrap source-freeze remote differs")
    if (
        not isinstance(value.get("ref"), str)
        or GIT_REF.fullmatch(str(value["ref"])) is None
    ):
        raise ValueError("CTAA bootstrap source-freeze ref differs")
    _identifier(value.get("observer_id"), "source-freeze observer")
    _hex(
        value.get("observer_implementation_sha256"),
        "source-freeze observer implementation",
    )
    _hex(value.get("remote_ref_evidence_sha256"), "source-freeze remote evidence")
    pushed_at_ms = _timestamp_ms(
        value.get("pushed_at"), "source-freeze pushed_at", canonical=True
    )
    observed_at_ms = _timestamp_ms(
        value.get("observed_at"), "source-freeze observed_at", canonical=True
    )
    if not 0 <= observed_at_ms - pushed_at_ms <= MAX_FREEZE_OBSERVATION_LAG_MS:
        raise ValueError("CTAA bootstrap source-freeze observation lag differs")
    validator_keys = value.get("validator_public_keys")
    if not isinstance(validator_keys, dict) or len(validator_keys) != 2:
        raise ValueError("CTAA bootstrap source-freeze validator keyset differs")
    normalized_keys: dict[str, str] = {}
    for name, public_key in validator_keys.items():
        identity = _identifier(name, "source-freeze validator")
        normalized_keys[identity] = _hex(public_key, "source-freeze validator key")
    if len(set(normalized_keys.values())) != 2:
        raise ValueError("CTAA bootstrap source-freeze validator keys repeat")
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    _verify_signature(
        custody_root_public_key_hex,
        value.get("signature"),
        FREEZE_SIGNATURE_DOMAIN + canonical_json(unsigned).encode("ascii"),
    )
    return dict(value), pushed_at_ms, normalized_keys


def _validator_statement(
    validator: Mapping[str, object], *, verification_bindings: Mapping[str, object]
) -> dict[str, object]:
    return {
        "beacon": dict(verification_bindings),
        "certificate_valid": validator["certificate_valid"],
        "chain_valid": validator["chain_valid"],
        "evidence_sha256": validator["evidence_sha256"],
        "implementation_sha256": validator["implementation_sha256"],
        "raw_beacon_sha256": validator["raw_beacon_sha256"],
        "signature_valid": validator["signature_valid"],
        "validated_at": validator["validated_at"],
        "validator_id": validator["validator_id"],
    }


def _validate_beacon(
    value: object,
    *,
    source_commit: str,
    custody_root_public_key_hex: str,
) -> tuple[dict[str, object], dict[str, object]]:
    keys = {
        "schema",
        "provider",
        "source_freeze",
        "raw_payload_base64",
        "raw_payload_sha256",
        "verification",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("schema") != BEACON_SCHEMA
    ):
        raise ValueError("CTAA bootstrap beacon schema differs")
    if value.get("provider") != "nist-beacon-2.0":
        raise ValueError("CTAA bootstrap beacon provider differs")
    freeze, freeze_ms, trusted_validators = _validate_source_freeze(
        value.get("source_freeze"),
        source_commit=source_commit,
        custody_root_public_key_hex=custody_root_public_key_hex,
    )
    encoded = value.get("raw_payload_base64")
    if not isinstance(encoded, str):
        raise ValueError("CTAA bootstrap beacon raw payload differs")
    try:
        raw_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError("CTAA bootstrap beacon raw payload differs") from error
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if value.get("raw_payload_sha256") != raw_sha256:
        raise ValueError("CTAA bootstrap beacon raw payload hash differs")
    raw_payload = _parse_json_bytes(raw_bytes, "beacon raw payload")
    if set(raw_payload) != {"pulse"} or not isinstance(raw_payload.get("pulse"), dict):
        raise ValueError("CTAA bootstrap beacon raw payload differs")
    pulse = raw_payload["pulse"]
    pulse_keys = {
        "certificateId",
        "chainIndex",
        "outputValue",
        "period",
        "pulseIndex",
        "signatureValue",
        "statusCode",
        "timeStamp",
    }
    if set(pulse) != pulse_keys:
        raise ValueError("CTAA bootstrap beacon pulse schema differs")
    period_ms = _positive_int(pulse.get("period"), "beacon period")
    if (
        period_ms != NIST_PERIOD_MS
        or type(pulse.get("statusCode")) is not int
        or pulse["statusCode"] != 0
    ):
        raise ValueError("CTAA bootstrap beacon status or period differs")
    pulse_ms = _timestamp_ms(pulse.get("timeStamp"), "beacon pulse", canonical=True)
    target_ms = freeze_ms + MINIMUM_BEACON_DELAY_MS
    if not target_ms <= pulse_ms < target_ms + period_ms:
        raise ValueError("CTAA bootstrap beacon is not the first delayed pulse")
    output_value = pulse.get("outputValue")
    if (
        not isinstance(output_value, str)
        or OUTPUT_VALUE.fullmatch(output_value) is None
    ):
        raise ValueError("CTAA bootstrap beacon output differs")
    certificate_id = pulse.get("certificateId")
    signature_value = pulse.get("signatureValue")
    if not isinstance(certificate_id, str) or not certificate_id:
        raise ValueError("CTAA bootstrap beacon certificate differs")
    if not isinstance(signature_value, str) or not signature_value:
        raise ValueError("CTAA bootstrap beacon signature differs")
    chain_index = _positive_int(pulse.get("chainIndex"), "beacon chain index")
    pulse_index = _positive_int(pulse.get("pulseIndex"), "beacon pulse index")
    bindings = {
        "beacon_raw_sha256": raw_sha256,
        "certificate_id": certificate_id,
        "chain_index": chain_index,
        "output_value_sha256": hashlib.sha256(bytes.fromhex(output_value)).hexdigest(),
        "period_ms": period_ms,
        "pulse_index": pulse_index,
        "signature_value_sha256": hashlib.sha256(signature_value.encode()).hexdigest(),
        "status_code": 0,
        "time_stamp": pulse["timeStamp"],
        "time_stamp_ms": pulse_ms,
    }
    verification = value.get("verification")
    verification_keys = {
        "schema",
        *bindings,
        "certificate_sha256",
        "validators",
        "validators_agree",
    }
    if (
        not isinstance(verification, dict)
        or set(verification) != verification_keys
        or verification.get("schema") != BEACON_VERIFICATION_SCHEMA
        or verification.get("validators_agree") is not True
        or any(verification.get(key) != expected for key, expected in bindings.items())
    ):
        raise ValueError("CTAA bootstrap beacon verification differs")
    _hex(verification.get("certificate_sha256"), "beacon certificate")
    validators = verification.get("validators")
    if not isinstance(validators, list) or len(validators) != 2:
        raise ValueError("CTAA bootstrap requires exactly two beacon validators")
    seen: set[str] = set()
    for validator in validators:
        validator_keys = {
            "certificate_valid",
            "chain_valid",
            "signature_valid",
            "validator_id",
            "implementation_sha256",
            "evidence_sha256",
            "raw_beacon_sha256",
            "validated_at",
            "attestation_signature",
        }
        if not isinstance(validator, dict) or set(validator) != validator_keys:
            raise ValueError("CTAA bootstrap beacon validator schema differs")
        identity = _identifier(validator.get("validator_id"), "beacon validator")
        if identity in seen or identity not in trusted_validators:
            raise ValueError("CTAA bootstrap beacon validator identity differs")
        seen.add(identity)
        if any(
            validator.get(key) is not True
            for key in ("certificate_valid", "chain_valid", "signature_valid")
        ):
            raise ValueError("CTAA bootstrap beacon cryptographic validation failed")
        _hex(validator.get("implementation_sha256"), "validator implementation")
        _hex(validator.get("evidence_sha256"), "validator evidence")
        if validator.get("raw_beacon_sha256") != raw_sha256:
            raise ValueError("CTAA bootstrap beacon validator raw binding differs")
        if (
            _timestamp_ms(validator.get("validated_at"), "validator", canonical=True)
            < pulse_ms
        ):
            raise ValueError("CTAA bootstrap beacon validator predates pulse")
        statement = _validator_statement(validator, verification_bindings=bindings)
        _verify_signature(
            trusted_validators[identity],
            validator.get("attestation_signature"),
            VALIDATOR_SIGNATURE_DOMAIN + canonical_json(statement).encode("ascii"),
        )
    if seen != set(trusted_validators):
        raise ValueError("CTAA bootstrap beacon validator coverage differs")
    commitment = {
        "provider": "nist-beacon-2.0",
        "source_commit": source_commit,
        "chain_index": chain_index,
        "pulse_index": pulse_index,
        "time_stamp_ms": pulse_ms,
        "output_value_hex": output_value.lower(),
    }
    return {**value, "source_freeze": freeze}, commitment


def _derivation_payload(
    *,
    source_commit: str,
    manifest_sha256: str,
    gate_source_sha256: str,
    statistics_source_sha256: str,
    beacon_commitment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "assessment_schema": ASSESSMENT_SCHEMA,
        "beacon_commitment": dict(beacon_commitment),
        "derivation_label": DERIVATION_LABEL,
        "gate_source_sha256": gate_source_sha256,
        "manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "statistics_source_sha256": statistics_source_sha256,
    }


def derive_seed(payload: Mapping[str, object]) -> tuple[int, str]:
    digest = hashlib.sha256(
        DOMAIN + canonical_json(dict(payload)).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex()


def build_receipt(
    *,
    source_commit: str,
    manifest_sha256: str,
    gate_source_sha256: str,
    statistics_source_sha256: str,
    beacon: Mapping[str, object],
    custody_root_public_key_hex: str | None = None,
) -> dict[str, object]:
    source_commit = _hex(source_commit, "source commit", GIT_OID)
    manifest_sha256 = _hex(manifest_sha256, "manifest")
    gate_source_sha256 = _hex(gate_source_sha256, "gate source")
    statistics_source_sha256 = _hex(statistics_source_sha256, "statistics source")
    custody_root_public_key_hex = (
        CUSTODY_ROOT_PUBLIC_KEY_HEX
        if custody_root_public_key_hex is None
        else custody_root_public_key_hex
    )
    _hex(custody_root_public_key_hex, "custody root public key")
    checked_beacon, beacon_commitment = _validate_beacon(
        dict(beacon),
        source_commit=source_commit,
        custody_root_public_key_hex=custody_root_public_key_hex,
    )
    payload = _derivation_payload(
        source_commit=source_commit,
        manifest_sha256=manifest_sha256,
        gate_source_sha256=gate_source_sha256,
        statistics_source_sha256=statistics_source_sha256,
        beacon_commitment=beacon_commitment,
    )
    seed, derivation = derive_seed(payload)
    return {
        "schema": SCHEMA,
        "derivation_scheme": DERIVATION_SCHEME,
        "beacon": checked_beacon,
        **payload,
        "bootstrap_seed": seed,
        "derivation_sha256": derivation,
        "oracle_access": 0,
    }


def validate_receipt(
    value: object,
    *,
    manifest_sha256: str,
    gate_source_sha256: str,
    statistics_source_sha256: str,
    custody_root_public_key_hex: str | None = None,
) -> dict[str, object]:
    keys = {
        "schema",
        "derivation_scheme",
        "assessment_schema",
        "beacon",
        "beacon_commitment",
        "derivation_label",
        "gate_source_sha256",
        "manifest_sha256",
        "source_commit",
        "statistics_source_sha256",
        "bootstrap_seed",
        "derivation_sha256",
        "oracle_access",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("CTAA bootstrap receipt schema differs")
    if (
        value.get("schema") != SCHEMA
        or value.get("derivation_scheme") != DERIVATION_SCHEME
        or value.get("assessment_schema") != ASSESSMENT_SCHEMA
        or value.get("derivation_label") != DERIVATION_LABEL
        or type(value.get("oracle_access")) is not int
        or value.get("oracle_access") != 0
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("gate_source_sha256") != gate_source_sha256
        or value.get("statistics_source_sha256") != statistics_source_sha256
    ):
        raise ValueError("CTAA bootstrap receipt binding differs")
    custody_root_public_key_hex = (
        CUSTODY_ROOT_PUBLIC_KEY_HEX
        if custody_root_public_key_hex is None
        else custody_root_public_key_hex
    )
    source_commit = _hex(value.get("source_commit"), "source commit", GIT_OID)
    _hex(value.get("derivation_sha256"), "derivation")
    checked_beacon, beacon_commitment = _validate_beacon(
        value.get("beacon"),
        source_commit=source_commit,
        custody_root_public_key_hex=custody_root_public_key_hex,
    )
    if value.get("beacon_commitment") != beacon_commitment:
        raise ValueError("CTAA bootstrap normalized beacon commitment differs")
    payload = _derivation_payload(
        source_commit=source_commit,
        manifest_sha256=manifest_sha256,
        gate_source_sha256=gate_source_sha256,
        statistics_source_sha256=statistics_source_sha256,
        beacon_commitment=beacon_commitment,
    )
    seed, derivation = derive_seed(payload)
    if (
        type(value.get("bootstrap_seed")) is not int
        or value.get("bootstrap_seed") != seed
    ):
        raise ValueError("CTAA bootstrap seed derivation differs")
    if value.get("derivation_sha256") != derivation:
        raise ValueError("CTAA bootstrap seed derivation differs")
    return {**value, "beacon": checked_beacon}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-source", type=Path, required=True)
    parser.add_argument("--statistics-source", type=Path, required=True)
    parser.add_argument("--beacon-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    beacon = _parse_json_bytes(args.beacon_receipt.read_bytes(), "beacon receipt")
    receipt = build_receipt(
        source_commit=args.source_commit,
        manifest_sha256=sha256_file(args.manifest),
        gate_source_sha256=sha256_file(args.gate_source),
        statistics_source_sha256=sha256_file(args.statistics_source),
        beacon=beacon,
    )
    digest = write_json_once(args.output, receipt)
    print(json.dumps({"bootstrap_seed": receipt["bootstrap_seed"], "sha256": digest}))


if __name__ == "__main__":
    main()
