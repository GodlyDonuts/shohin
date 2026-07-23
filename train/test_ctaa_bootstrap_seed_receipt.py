from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import ctaa_bootstrap_seed_receipt as bootstrap
from ctaa_bootstrap_seed_receipt import build_receipt, validate_receipt
from ctaa_evaluation_io import canonical_json


ROOT_KEY = Ed25519PrivateKey.from_private_bytes(b"r" * 32)
VALIDATOR_KEYS = {
    "alpha": Ed25519PrivateKey.from_private_bytes(b"a" * 32),
    "beta": Ed25519PrivateKey.from_private_bytes(b"b" * 32),
}


def _public(key: Ed25519PrivateKey) -> str:
    return (
        key.public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )


TEST_ROOT_PUBLIC = _public(ROOT_KEY)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _timestamp(value_ms: int) -> str:
    return datetime.fromtimestamp(value_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )


def _beacon(
    *,
    observer_id: str = "test-observer",
    validator_order: tuple[str, str] = ("alpha", "beta"),
    validation_delay_ms: int = 0,
    output_value: str = "ab" * 64,
    raw_indent: int | None = None,
    status_code: object = 0,
    root_key: Ed25519PrivateKey = ROOT_KEY,
) -> dict[str, object]:
    source_commit = "a" * 40
    freeze_ms = 1_700_000_000_000
    pulse_ms = freeze_ms + 3_600_000
    pulse = {
        "certificateId": "test-certificate",
        "chainIndex": 4,
        "outputValue": output_value,
        "period": 60_000,
        "pulseIndex": 42,
        "signatureValue": "test-signature",
        "statusCode": status_code,
        "timeStamp": _timestamp(pulse_ms),
    }
    raw_payload = {"pulse": pulse}
    raw_bytes = (
        json.dumps(raw_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        if raw_indent is None
        else (json.dumps(raw_payload, sort_keys=True, indent=raw_indent) + "\n").encode(
            "ascii"
        )
    )
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    freeze_unsigned: dict[str, object] = {
        "schema": "r12_ctaa_v2_source_freeze_v1",
        "commit_oid": source_commit,
        "remote_url": "https://github.com/test-owner/test-repo.git",
        "ref": "refs/heads/ctaa-freeze",
        "remote_ref_oid": source_commit,
        "pushed_at": _timestamp(freeze_ms),
        "observed_at": _timestamp(freeze_ms + 1_000),
        "observer_id": observer_id,
        "observer_implementation_sha256": _digest("observer-implementation"),
        "remote_ref_evidence_sha256": _digest("remote-ref-evidence"),
        "validator_public_keys": {
            name: _public(key) for name, key in VALIDATOR_KEYS.items()
        },
    }
    freeze = {
        **freeze_unsigned,
        "signature": root_key.sign(
            bootstrap.FREEZE_SIGNATURE_DOMAIN
            + canonical_json(freeze_unsigned).encode("ascii")
        ).hex(),
    }
    bindings = {
        "beacon_raw_sha256": raw_sha256,
        "certificate_id": "test-certificate",
        "chain_index": 4,
        "output_value_sha256": hashlib.sha256(bytes.fromhex(output_value)).hexdigest(),
        "period_ms": 60_000,
        "pulse_index": 42,
        "signature_value_sha256": hashlib.sha256(b"test-signature").hexdigest(),
        "status_code": status_code,
        "time_stamp": _timestamp(pulse_ms),
        "time_stamp_ms": pulse_ms,
    }
    validators = []
    for name in validator_order:
        statement: dict[str, object] = {
            "beacon": bindings,
            "certificate_valid": True,
            "chain_valid": True,
            "evidence_sha256": _digest(f"evidence:{name}"),
            "implementation_sha256": _digest(f"impl:{name}"),
            "raw_beacon_sha256": raw_sha256,
            "signature_valid": True,
            "validated_at": _timestamp(pulse_ms + validation_delay_ms),
            "validator_id": name,
        }
        validators.append(
            {
                **{key: item for key, item in statement.items() if key != "beacon"},
                "attestation_signature": VALIDATOR_KEYS[name]
                .sign(
                    bootstrap.VALIDATOR_SIGNATURE_DOMAIN
                    + canonical_json(statement).encode("ascii")
                )
                .hex(),
            }
        )
    return {
        "schema": "r12_ctaa_v2_public_beacon_receipt_v3",
        "provider": "nist-beacon-2.0",
        "source_freeze": freeze,
        "raw_payload_base64": base64.b64encode(raw_bytes).decode("ascii"),
        "raw_payload_sha256": raw_sha256,
        "verification": {
            "schema": "r12_ctaa_v2_nist_beacon_verification_v2",
            **bindings,
            "certificate_sha256": _digest("certificate"),
            "validators": validators,
            "validators_agree": True,
        },
    }


def _receipt(
    beacon: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    bindings = {
        "manifest_sha256": _digest("manifest"),
        "gate_source_sha256": _digest("gate"),
        "statistics_source_sha256": _digest("statistics"),
    }
    return (
        build_receipt(
            source_commit="a" * 40,
            beacon=_beacon() if beacon is None else beacon,
            custody_root_public_key_hex=TEST_ROOT_PUBLIC,
            **bindings,
        ),
        bindings,
    )


def test_receipt_is_deterministic_and_validates_exact_bindings() -> None:
    receipt, bindings = _receipt()
    assert _receipt()[0] == receipt
    assert (
        validate_receipt(
            receipt,
            custody_root_public_key_hex=TEST_ROOT_PUBLIC,
            **bindings,
        )["bootstrap_seed"]
        == receipt["bootstrap_seed"]
    )


def test_validator_order_timing_observer_and_raw_format_cannot_grind_seed() -> None:
    variants = (
        _beacon(validator_order=("beta", "alpha")),
        _beacon(validation_delay_ms=2_000),
        _beacon(observer_id="other-observer"),
        _beacon(raw_indent=2),
        _beacon(output_value=("AB" * 64)),
    )
    seeds = {_receipt(beacon)[0]["bootstrap_seed"] for beacon in variants}
    seeds.add(_receipt()[0]["bootstrap_seed"])
    assert len(seeds) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("bootstrap_seed", 7),
        lambda value: value.__setitem__("source_commit", "b" * 40),
        lambda value: value["beacon"]["verification"]["validators"].pop(),
        lambda value: value["beacon"]["verification"]["validators"][0].__setitem__(
            "signature_valid", False
        ),
        lambda value: value["beacon"]["verification"]["validators"][0].__setitem__(
            "attestation_signature", "0" * 128
        ),
        lambda value: value.__setitem__("oracle_access", False),
    ),
)
def test_mutation_is_rejected(mutation) -> None:
    receipt, bindings = _receipt()
    hostile = deepcopy(receipt)
    mutation(hostile)
    with pytest.raises(ValueError):
        validate_receipt(
            hostile,
            custody_root_public_key_hex=TEST_ROOT_PUBLIC,
            **bindings,
        )


def test_self_consistent_attacker_root_is_rejected() -> None:
    attacker = Ed25519PrivateKey.from_private_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="signature"):
        _receipt(_beacon(root_key=attacker))


def test_boolean_status_code_is_rejected_even_when_signed() -> None:
    with pytest.raises(ValueError, match="status"):
        _receipt(_beacon(status_code=False))


def test_substituted_source_binding_is_rejected() -> None:
    receipt, bindings = _receipt()
    with pytest.raises(ValueError, match="binding"):
        validate_receipt(
            receipt,
            custody_root_public_key_hex=TEST_ROOT_PUBLIC,
            **{**bindings, "manifest_sha256": _digest("other")},
        )
