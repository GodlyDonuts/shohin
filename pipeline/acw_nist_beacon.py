#!/usr/bin/env python3
"""Verify an archived NIST Beacon v2 pulse for ACW confirmation seeding.

This module implements the deployed NIST Beacon 2.0 beta cipher-suite-0
serialization. The deployed service uses four-byte length prefixes and hashes
the signature without a length prefix, which differs from the eight-byte
prefixes in draft NISTIR 8213. It does not treat HTTPS or a JSON payload hash
as pulse authentication: the pulse RSA signature, certificate identifier,
output hash, previous link, and previous precommitment reveal are replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BEACON_BASE_URL = "https://beacon.nist.gov/beacon/2.0"
PULSE_PROTOCOL = "R12-ACW-NIST-BEACON-PULSE-v1"
DERIVATION_DOMAIN = b"R12-ACW-NIST-CONFIRM-v1\x00"
HASH_BYTES = 64
SIGNATURE_BYTES = 512
PERIOD_MS = 60_000
TARGET_DELAY_PULSES = 60

_PULSE_KEYS = {
    "uri",
    "version",
    "cipherSuite",
    "period",
    "certificateId",
    "chainIndex",
    "pulseIndex",
    "timeStamp",
    "localRandomValue",
    "external",
    "listValues",
    "precommitmentValue",
    "statusCode",
    "signatureValue",
    "outputValue",
}
_EXTERNAL_KEYS = {"sourceId", "statusCode", "value"}
_LIST_TYPES = ("previous", "hour", "day", "month", "year")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _uint(value: Any, width: int, label: str) -> bytes:
    if not isinstance(value, int) or not 0 <= value < 1 << (8 * width):
        raise ValueError(f"{label} is not an unsigned {8 * width}-bit integer")
    return value.to_bytes(width, "big")


def _string(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a string")
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _fixed_hex(value: Any, size: int, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != 2 * size:
        raise ValueError(f"{label} does not contain {size} bytes")
    try:
        encoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    if len(encoded) != size:
        raise ValueError(f"{label} does not contain {size} bytes")
    return size.to_bytes(4, "big") + encoded


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("pulse timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("pulse timestamp is malformed") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise ValueError("pulse timestamp must have whole-second UTC precision")
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if value != canonical:
        raise ValueError("pulse timestamp spelling is not canonical")
    return parsed


def _list_values(pulse: dict[str, Any]) -> dict[str, str]:
    records = pulse.get("listValues")
    if not isinstance(records, list) or len(records) != len(_LIST_TYPES):
        raise ValueError("pulse listValues has the wrong cardinality")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"type", "uri", "value"}:
            raise ValueError("pulse listValues record has the wrong schema")
        kind = record["type"]
        if kind not in _LIST_TYPES or kind in result:
            raise ValueError("pulse listValues type is missing or duplicated")
        if not isinstance(record["uri"], str) or not record["uri"].startswith(
            BEACON_BASE_URL + "/"
        ):
            raise ValueError("pulse listValues URI is outside the NIST Beacon")
        _fixed_hex(record["value"], HASH_BYTES, f"listValues.{kind}")
        result[kind] = record["value"]
    if tuple(sorted(result, key=_LIST_TYPES.index)) != _LIST_TYPES:
        raise ValueError("pulse listValues type set is incomplete")
    return result


def pulse_signature_message(pulse: dict[str, Any]) -> bytes:
    """Serialize fields F1..F19 exactly as NISTIR 8213 Algorithm 2."""

    if not isinstance(pulse, dict) or set(pulse) != _PULSE_KEYS:
        raise ValueError("pulse has the wrong schema")
    if pulse["version"] != "2.0" or pulse["cipherSuite"] != 0:
        raise ValueError("only NIST Beacon v2 cipher suite 0 is accepted")
    if pulse["period"] != PERIOD_MS:
        raise ValueError("unexpected NIST Beacon pulse period")
    if not isinstance(pulse["uri"], str) or not pulse["uri"].startswith(
        BEACON_BASE_URL + "/chain/"
    ):
        raise ValueError("pulse URI is outside the canonical NIST chain API")
    _timestamp(pulse["timeStamp"])
    external = pulse["external"]
    if not isinstance(external, dict) or set(external) != _EXTERNAL_KEYS:
        raise ValueError("pulse external record has the wrong schema")
    listed = _list_values(pulse)
    fields = [
        _string(pulse["uri"], "uri"),
        _string(pulse["version"], "version"),
        _uint(pulse["cipherSuite"], 4, "cipherSuite"),
        _uint(pulse["period"], 4, "period"),
        _fixed_hex(pulse["certificateId"], HASH_BYTES, "certificateId"),
        _uint(pulse["chainIndex"], 8, "chainIndex"),
        _uint(pulse["pulseIndex"], 8, "pulseIndex"),
        _string(pulse["timeStamp"], "timeStamp"),
        _fixed_hex(pulse["localRandomValue"], HASH_BYTES, "localRandomValue"),
        _fixed_hex(external["sourceId"], HASH_BYTES, "external.sourceId"),
        _uint(external["statusCode"], 4, "external.statusCode"),
        _fixed_hex(external["value"], HASH_BYTES, "external.value"),
    ]
    fields.extend(
        _fixed_hex(listed[kind], HASH_BYTES, f"listValues.{kind}")
        for kind in _LIST_TYPES
    )
    fields.extend(
        [
            _fixed_hex(pulse["precommitmentValue"], HASH_BYTES, "precommitmentValue"),
            _uint(pulse["statusCode"], 4, "statusCode"),
        ]
    )
    return b"".join(fields)


def _certificate_der(certificate_pem: bytes) -> bytes:
    try:
        text = certificate_pem.decode("ascii")
        return ssl.PEM_cert_to_DER_cert(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            "NIST certificate is not one canonical PEM certificate"
        ) from error


def _verify_rsa_signature(
    message: bytes,
    signature: bytes,
    certificate_pem: bytes,
) -> None:
    with tempfile.TemporaryDirectory(prefix="acw-nist-") as temporary:
        root = Path(temporary)
        certificate = root / "certificate.pem"
        public_key = root / "public.pem"
        message_path = root / "message.bin"
        signature_path = root / "signature.bin"
        certificate.write_bytes(certificate_pem)
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        with public_key.open("wb") as public_handle:
            subprocess.run(
                ["openssl", "x509", "-in", str(certificate), "-pubkey", "-noout"],
                check=True,
                stdout=public_handle,
                stderr=subprocess.PIPE,
            )
        result = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha512",
                "-verify",
                str(public_key),
                "-signature",
                str(signature_path),
                str(message_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "Verified OK":
            raise ValueError("NIST pulse RSA signature verification failed")


def verify_pulse(
    pulse: dict[str, Any],
    certificate_pem: bytes,
    *,
    previous_pulse: dict[str, Any] | None = None,
    expected_chain_index: int | None = None,
    expected_pulse_index: int | None = None,
    expected_timestamp: str | None = None,
) -> dict[str, Any]:
    """Verify one pulse and optionally its immediate chain/precommitment link."""

    message = pulse_signature_message(pulse)
    certificate_der = _certificate_der(certificate_pem)
    certificate_id = hashlib.sha512(certificate_der).hexdigest().upper()
    if certificate_id != pulse["certificateId"].upper():
        raise ValueError("NIST certificate does not match pulse certificateId")
    signature = bytes.fromhex(pulse["signatureValue"])
    if len(signature) != SIGNATURE_BYTES:
        raise ValueError("NIST pulse signature has the wrong length")
    _verify_rsa_signature(message, signature, certificate_pem)
    output_preimage = message + signature
    observed_output = hashlib.sha512(output_preimage).hexdigest().upper()
    if observed_output != pulse["outputValue"].upper():
        raise ValueError("NIST pulse outputValue hash mismatch")
    if expected_chain_index is not None and pulse["chainIndex"] != expected_chain_index:
        raise ValueError("NIST pulse chain index differs from authorization")
    if expected_pulse_index is not None and pulse["pulseIndex"] != expected_pulse_index:
        raise ValueError("NIST pulse index differs from authorization")
    if expected_timestamp is not None and pulse["timeStamp"] != expected_timestamp:
        raise ValueError("NIST pulse timestamp differs from authorization")

    link = None
    if previous_pulse is not None:
        previous_message = pulse_signature_message(previous_pulse)
        del previous_message
        if pulse["chainIndex"] != previous_pulse["chainIndex"]:
            raise ValueError("NIST previous pulse is from another chain")
        if pulse["pulseIndex"] != previous_pulse["pulseIndex"] + 1:
            raise ValueError("NIST pulse indices are not consecutive")
        if _timestamp(pulse["timeStamp"]) != _timestamp(
            previous_pulse["timeStamp"]
        ) + timedelta(milliseconds=PERIOD_MS):
            raise ValueError(
                "NIST consecutive pulse timestamps are not one period apart"
            )
        previous_link = _list_values(pulse)["previous"].upper()
        if previous_link != previous_pulse["outputValue"].upper():
            raise ValueError("NIST previous output link mismatch")
        revealed = hashlib.sha512(bytes.fromhex(pulse["localRandomValue"])).hexdigest()
        if revealed.upper() != previous_pulse["precommitmentValue"].upper():
            raise ValueError("NIST previous precommitment was not revealed")
        if pulse["statusCode"] & 1:
            raise ValueError("NIST pulse marks its precommitment reveal invalid")
        link = {
            "previous_output_value": previous_pulse["outputValue"],
            "previous_precommitment_value": previous_pulse["precommitmentValue"],
        }

    pulse_hash = sha256_bytes(canonical_json_bytes(pulse))
    return {
        "protocol": PULSE_PROTOCOL,
        "chain_index": pulse["chainIndex"],
        "pulse_index": pulse["pulseIndex"],
        "timestamp": pulse["timeStamp"],
        "output_value": pulse["outputValue"],
        "pulse_payload_sha256": pulse_hash,
        "certificate_der_sha512": certificate_id.lower(),
        "signature_verified": True,
        "output_hash_verified": True,
        "previous_link": link,
    }


def derive_confirmation_seed(
    *,
    authorization_payload_sha256: str,
    pulse: dict[str, Any],
    index: int,
) -> bytes:
    if (
        not isinstance(authorization_payload_sha256, str)
        or len(authorization_payload_sha256) != 64
    ):
        raise ValueError("authorization payload SHA-256 is malformed")
    try:
        authorization_hash = bytes.fromhex(authorization_payload_sha256)
    except ValueError as error:
        raise ValueError("authorization payload SHA-256 is malformed") from error
    if index not in range(3):
        raise ValueError("confirmation seed index must be 0, 1, or 2")
    output = bytes.fromhex(pulse["outputValue"])
    material = (
        DERIVATION_DOMAIN
        + authorization_hash
        + _uint(pulse["chainIndex"], 8, "chainIndex")
        + _uint(pulse["pulseIndex"], 8, "pulseIndex")
        + output
        + bytes([index])
    )
    return hashlib.sha256(material).digest()


def fetch_json(url: str) -> dict[str, Any]:
    if not url.startswith(BEACON_BASE_URL + "/"):
        raise ValueError("refusing to fetch outside the NIST Beacon v2 API")
    request = urllib.request.Request(url, headers={"User-Agent": "shohin-acw/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"NIST Beacon returned HTTP {response.status}")
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("NIST Beacon JSON response is unexpectedly large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("NIST Beacon response is not a JSON object")
    return value


def fetch_certificate(certificate_id: str) -> bytes:
    if len(certificate_id) != 128:
        raise ValueError("NIST certificate ID is malformed")
    url = f"{BEACON_BASE_URL}/certificate/{certificate_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "shohin-acw/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(
                f"NIST certificate endpoint returned HTTP {response.status}"
            )
        value = response.read(1024 * 1024 + 1)
    if len(value) > 1024 * 1024:
        raise RuntimeError("NIST certificate response is unexpectedly large")
    return value


def fetch_pulse(chain_index: int, pulse_index: int) -> dict[str, Any]:
    value = fetch_json(f"{BEACON_BASE_URL}/chain/{chain_index}/pulse/{pulse_index}")
    if set(value) != {"pulse"} or not isinstance(value["pulse"], dict):
        raise ValueError("NIST Beacon pulse response has the wrong envelope")
    return value["pulse"]


def write_snapshot(
    *,
    chain_index: int,
    pulse_index: int,
    out: Path,
) -> dict[str, Any]:
    pulse = fetch_pulse(chain_index, pulse_index)
    previous = fetch_pulse(chain_index, pulse_index - 1)
    certificate = fetch_certificate(pulse["certificateId"])
    receipt = verify_pulse(
        pulse,
        certificate,
        previous_pulse=previous,
        expected_chain_index=chain_index,
        expected_pulse_index=pulse_index,
    )
    payload = {
        "protocol": PULSE_PROTOCOL,
        "pulse": pulse,
        "previous_pulse": previous,
        "certificate_pem": certificate.decode("ascii"),
        "verification": receipt,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    encoded = canonical_json_bytes(payload) + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-index", type=int, required=True)
    parser.add_argument("--pulse-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = write_snapshot(
        chain_index=args.chain_index,
        pulse_index=args.pulse_index,
        out=args.out,
    )
    print(
        json.dumps(
            {
                "pulse_index": payload["pulse"]["pulseIndex"],
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
