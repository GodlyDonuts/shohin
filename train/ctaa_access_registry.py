"""Externally verifiable append-only event registry for CTAA access custody.

The registry is a canonical-JSONL Ed25519-signed hash chain.  Its event state
machine separates oracle access from the assessment that can only exist after
that access:

* ``access_spend`` binds the manifest, board, run contract, runtime bundle,
  signed assessment claim, partition, and access ID.  It intentionally
  contains no assessment result hash.
* the immediately following ``assessment_commit`` closes that access ID and
  binds the resulting assessment hash;
* ``development_gate_commit`` binds a gate receipt for the latest closed
  development access and is required before any confirmation access.

The append API returns a signed head receipt.  Retaining that receipt outside
the registry filesystem and supplying it to ``verify_registry`` makes reset,
rollback, fork, truncation, and same-UID file replacement detectable.  This
module deliberately does not load keys or integrate with the CTAA assessor.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


ENTRY_SCHEMA = "r12_ctaa_access_registry_event_v5"
HEAD_RECEIPT_SCHEMA = "r12_ctaa_access_registry_head_receipt_v2"
GENESIS_PREVIOUS_HASH = "0" * 64
ACCESS_SPEND = "access_spend"
ASSESSMENT_COMMIT = "assessment_commit"
DEVELOPMENT_GATE_COMMIT = "development_gate_commit"
EVENT_TYPES = (ACCESS_SPEND, ASSESSMENT_COMMIT, DEVELOPMENT_GATE_COMMIT)
PARTITIONS = ("development", "confirmation")

_COMMON_PAYLOAD_KEYS = {
    "schema",
    "registry_id",
    "sequence",
    "previous_hash",
    "event_type",
    "event_id",
    "signing_public_key",
}
_EVENT_KEYS = {
    ACCESS_SPEND: {
        "access_id",
        "partition",
        "manifest_sha256",
        "board_sha256",
        "run_contract_sha256",
        "runtime_bundle_sha256",
        "assessment_claim_sha256",
        "bootstrap_seed_receipt_sha256",
        "bootstrap_seed",
        "statistical_gate_spec_file_sha256",
        "gate_spec_sha256",
    },
    ASSESSMENT_COMMIT: {
        "access_id",
        "assessment_sha256",
        "statistical_gate_spec_file_sha256",
        "gate_spec_sha256",
    },
    DEVELOPMENT_GATE_COMMIT: {
        "development_access_id",
        "development_gate_receipt_sha256",
    },
}
_ENTRY_KEYS = {"payload", "signature", "entry_hash"}
_RECEIPT_PAYLOAD_KEYS = {
    "schema",
    "registry_id",
    "sequence",
    "entry_count",
    "entry_hash",
    "event_type",
    "event_id",
    "event_payload_sha256",
    "signing_public_key",
}
_RECEIPT_KEYS = {"payload", "signature"}
_IDENTIFIER_LIMIT = 256

# Edwards25519 constants.  The explicit subgroup check rejects non-canonical,
# torsion, small-order, and mixed-order public keys before signature checking.
_FIELD_P = 2**255 - 19
_GROUP_L = 2**252 + 27742317777372353535851937790883648493
_CURVE_D = (-121665 * pow(121666, _FIELD_P - 2, _FIELD_P)) % _FIELD_P
_SQRT_M1 = pow(2, (_FIELD_P - 1) // 4, _FIELD_P)
_IDENTITY = (0, 1)


class RegistryVerificationError(ValueError):
    """The event chain, key, state transition, or retained receipt is invalid."""


class ConcurrentAppendError(RegistryVerificationError):
    """The registry head differs from the append caller's expected head."""


@dataclass(frozen=True)
class RegistryState:
    """Verified public state of an access registry."""

    registry_id: str
    entry_count: int
    head_sequence: int
    head_hash: str
    head_event_type: str
    head_event_id: str
    head_access_id: str
    signing_public_key: str
    open_access_id: str | None
    development_gate_access_id: str | None
    confirmation_started: bool


@dataclass(frozen=True)
class VerifiedRegistryEvent:
    """Immutable event view released only after full registry verification."""

    payload: Mapping[str, object]
    canonical_payload: bytes
    signature: str
    entry_hash: str


@dataclass
class _MachineState:
    open_access_id: str | None = None
    open_partition: str | None = None
    last_closed_development_access_id: str | None = None
    development_gate_access_id: str | None = None
    confirmation_started: bool = False


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a signature payload with one unambiguous JSON encoding."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RegistryVerificationError("value is not canonical JSON") from exc


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _IDENTIFIER_LIMIT
        or value.strip() != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise RegistryVerificationError(f"invalid {label}")
    return value


def _validate_hash(value: object, label: str) -> str:
    if not _is_lower_hex(value, 64):
        raise RegistryVerificationError(f"invalid {label}")
    return str(value)


def _point_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = left
    x2, y2 = right
    product = (_CURVE_D * x1 * x2 * y1 * y2) % _FIELD_P
    x_denominator = (1 + product) % _FIELD_P
    y_denominator = (1 - product) % _FIELD_P
    if x_denominator == 0 or y_denominator == 0:
        raise RegistryVerificationError("invalid Ed25519 public key point")
    x3 = (x1 * y2 + y1 * x2) * pow(x_denominator, _FIELD_P - 2, _FIELD_P)
    y3 = (y1 * y2 + x1 * x2) * pow(y_denominator, _FIELD_P - 2, _FIELD_P)
    return x3 % _FIELD_P, y3 % _FIELD_P


def _scalar_multiply(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


@lru_cache(maxsize=64)
def _decode_prime_order_public_key(raw_key: bytes) -> tuple[int, int]:
    if not isinstance(raw_key, bytes) or len(raw_key) != 32:
        raise RegistryVerificationError("Ed25519 public key must be 32 bytes")
    encoded = int.from_bytes(raw_key, "little")
    sign_bit = encoded >> 255
    y = encoded & ((1 << 255) - 1)
    if y >= _FIELD_P:
        raise RegistryVerificationError("non-canonical Ed25519 public key")
    y_squared = y * y % _FIELD_P
    denominator = (_CURVE_D * y_squared + 1) % _FIELD_P
    if denominator == 0:
        raise RegistryVerificationError("invalid Ed25519 public key point")
    x_squared = (y_squared - 1) * pow(denominator, _FIELD_P - 2, _FIELD_P) % _FIELD_P
    x = pow(x_squared, (_FIELD_P + 3) // 8, _FIELD_P)
    if x * x % _FIELD_P != x_squared:
        x = x * _SQRT_M1 % _FIELD_P
    if x * x % _FIELD_P != x_squared:
        raise RegistryVerificationError("Ed25519 public key is not on curve")
    if x == 0 and sign_bit:
        raise RegistryVerificationError("non-canonical Ed25519 public key sign")
    if (x & 1) != sign_bit:
        x = (-x) % _FIELD_P
    point = (x, y)
    if point == _IDENTITY or _scalar_multiply(point, _GROUP_L) != _IDENTITY:
        raise RegistryVerificationError(
            "Ed25519 public key is not in the prime-order subgroup"
        )
    return point


def _public_key_bytes(
    verification_key: bytes | Ed25519PublicKey,
) -> bytes:
    if isinstance(verification_key, Ed25519PublicKey):
        raw_key = verification_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    elif isinstance(verification_key, bytes):
        raw_key = verification_key
    else:
        raise RegistryVerificationError("unsupported Ed25519 public key type")
    _decode_prime_order_public_key(raw_key)
    return raw_key


def _signing_public_key(signing_key: Ed25519PrivateKey) -> bytes:
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise TypeError("signing_key must be an Ed25519PrivateKey")
    return _public_key_bytes(signing_key.public_key())


def _verify_signature(
    raw_key: bytes, signature_hex: object, payload: Mapping[str, object]
) -> None:
    if not _is_lower_hex(signature_hex, 128):
        raise RegistryVerificationError("malformed Ed25519 signature")
    try:
        Ed25519PublicKey.from_public_bytes(raw_key).verify(
            bytes.fromhex(str(signature_hex)), canonical_json_bytes(dict(payload))
        )
    except InvalidSignature as exc:
        raise RegistryVerificationError(
            "Ed25519 signature verification failed"
        ) from exc


def _validate_entry_payload(
    payload: object,
    *,
    expected_key_hex: str,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RegistryVerificationError("event payload is not an object")
    event_type = payload.get("event_type")
    if event_type not in EVENT_TYPES:
        raise RegistryVerificationError("invalid event type")
    expected_keys = _COMMON_PAYLOAD_KEYS | _EVENT_KEYS[str(event_type)]
    if set(payload) != expected_keys:
        raise RegistryVerificationError("event payload schema differs")
    if payload["schema"] != ENTRY_SCHEMA:
        raise RegistryVerificationError("event schema version differs")
    _validate_identifier(payload["registry_id"], "registry_id")
    _validate_identifier(payload["event_id"], "event_id")
    if type(payload["sequence"]) is not int or int(payload["sequence"]) < 0:
        raise RegistryVerificationError("invalid event sequence")
    _validate_hash(payload["previous_hash"], "previous hash")
    if payload["signing_public_key"] != expected_key_hex:
        raise RegistryVerificationError("event uses the wrong signing key")

    if event_type == ACCESS_SPEND:
        _validate_identifier(payload["access_id"], "access_id")
        if payload["partition"] not in PARTITIONS:
            raise RegistryVerificationError("invalid access partition")
        for key in (
            "manifest_sha256",
            "board_sha256",
            "run_contract_sha256",
            "runtime_bundle_sha256",
            "assessment_claim_sha256",
            "bootstrap_seed_receipt_sha256",
            "statistical_gate_spec_file_sha256",
            "gate_spec_sha256",
        ):
            _validate_hash(payload[key], key)
        if (
            type(payload["bootstrap_seed"]) is not int
            or int(payload["bootstrap_seed"]) < 0
        ):
            raise RegistryVerificationError("invalid bootstrap_seed")
        if "assessment_sha256" in payload:
            raise RegistryVerificationError("access_spend cannot bind an assessment")
    elif event_type == ASSESSMENT_COMMIT:
        _validate_identifier(payload["access_id"], "access_id")
        for key in (
            "assessment_sha256",
            "statistical_gate_spec_file_sha256",
            "gate_spec_sha256",
        ):
            _validate_hash(payload[key], key)
    else:
        _validate_identifier(payload["development_access_id"], "development_access_id")
        _validate_hash(
            payload["development_gate_receipt_sha256"],
            "development_gate_receipt_sha256",
        )
    return dict(payload)


def _entry_hash(payload: Mapping[str, object], signature_hex: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"payload": dict(payload), "signature": signature_hex})
    ).hexdigest()


def _make_signed_entry(
    payload: Mapping[str, object], signing_key: Ed25519PrivateKey
) -> dict[str, object]:
    signature = signing_key.sign(canonical_json_bytes(dict(payload))).hex()
    return {
        "payload": dict(payload),
        "signature": signature,
        "entry_hash": _entry_hash(payload, signature),
    }


def _verify_entry(
    value: object,
    *,
    raw_key: bytes,
    expected_key_hex: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ENTRY_KEYS:
        raise RegistryVerificationError("event record schema differs")
    payload = _validate_entry_payload(
        value["payload"], expected_key_hex=expected_key_hex
    )
    _verify_signature(raw_key, value["signature"], payload)
    expected_hash = _entry_hash(payload, str(value["signature"]))
    if value["entry_hash"] != expected_hash:
        raise RegistryVerificationError("event entry hash differs")
    return {**value, "payload": payload}


def _open_registry_parent(path: Path) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(os.path.sep, flags)
        try:
            for component in absolute.parent.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as error:
        raise RegistryVerificationError(
            "access registry parent cannot be opened safely"
        ) from error
    return absolute, descriptor


def _read_registry_bytes(path: Path) -> bytes:
    absolute, parent_descriptor = _open_registry_parent(path)
    try:
        parent_before = os.fstat(parent_descriptor)
        try:
            metadata = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            raise RegistryVerificationError("access registry does not exist") from None
        except OSError as error:
            raise RegistryVerificationError(
                "access registry cannot be opened safely"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise RegistryVerificationError(
                "access registry must be a single-link regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise RegistryVerificationError(
                "access registry cannot be opened safely"
            ) from error
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        != (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size)
        or (before.st_mtime_ns, before.st_ctime_ns)
        != (metadata.st_mtime_ns, metadata.st_ctime_ns)
        or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        or after.st_nlink != 1
    ):
        raise RegistryVerificationError("access registry changed while being read")
    return b"".join(chunks)


def _load_entries_with_raw(
    path: Path, raw_key: bytes
) -> tuple[list[dict[str, object]], bytes]:
    data = _read_registry_bytes(path)
    if not data or not data.endswith(b"\n"):
        raise RegistryVerificationError("access registry is empty or incomplete")
    expected_key_hex = raw_key.hex()
    entries: list[dict[str, object]] = []
    for line_number, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.endswith(b"\n") or line == b"\n":
            raise RegistryVerificationError("access registry contains a partial row")
        try:
            decoded = json.loads(line[:-1].decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryVerificationError(
                f"access registry row {line_number} is malformed"
            ) from exc
        if line != canonical_json_bytes(decoded) + b"\n":
            raise RegistryVerificationError(
                f"access registry row {line_number} is not canonical JSON"
            )
        entries.append(
            _verify_entry(
                decoded,
                raw_key=raw_key,
                expected_key_hex=expected_key_hex,
            )
        )
    return entries, data


def _load_entries(path: Path, raw_key: bytes) -> list[dict[str, object]]:
    entries, _ = _load_entries_with_raw(path, raw_key)
    return entries


def _event_fingerprint(payload: Mapping[str, object]) -> str:
    excluded = {
        "sequence",
        "previous_hash",
        "event_id",
        "signing_public_key",
    }
    return hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key not in excluded}
        )
    ).hexdigest()


def _validate_chain(entries: list[dict[str, object]]) -> _MachineState:
    registry_id: str | None = None
    previous_hash = GENESIS_PREVIOUS_HASH
    event_ids: set[str] = set()
    event_fingerprints: set[str] = set()
    access_ids: set[str] = set()
    access_bindings: set[tuple[str, str, str, str, str, str, int]] = set()
    closed_accesses: dict[str, str] = {}
    machine = _MachineState()

    for sequence, entry in enumerate(entries):
        payload = entry["payload"]
        assert isinstance(payload, dict)
        if payload["sequence"] != sequence:
            raise RegistryVerificationError("event sequence is not monotonic")
        if payload["previous_hash"] != previous_hash:
            raise RegistryVerificationError("event hash chain differs")
        if registry_id is None:
            registry_id = str(payload["registry_id"])
        elif payload["registry_id"] != registry_id:
            raise RegistryVerificationError("access registry identifier changed")

        event_id = str(payload["event_id"])
        if event_id in event_ids:
            raise RegistryVerificationError("duplicate event_id")
        event_ids.add(event_id)
        fingerprint = _event_fingerprint(payload)
        if fingerprint in event_fingerprints:
            raise RegistryVerificationError("duplicate event")
        event_fingerprints.add(fingerprint)

        event_type = str(payload["event_type"])
        if machine.open_access_id is not None and event_type != ASSESSMENT_COMMIT:
            if event_type == DEVELOPMENT_GATE_COMMIT:
                raise RegistryVerificationError(
                    "development gate cannot precede development assessment close"
                )
            raise RegistryVerificationError(
                "open access must be immediately closed by assessment_commit"
            )

        if event_type == ACCESS_SPEND:
            access_id = str(payload["access_id"])
            partition = str(payload["partition"])
            if access_id in access_ids:
                raise RegistryVerificationError("duplicate access_id")
            access_ids.add(access_id)
            binding = (
                partition,
                str(payload["manifest_sha256"]),
                str(payload["board_sha256"]),
                str(payload["run_contract_sha256"]),
                str(payload["runtime_bundle_sha256"]),
                str(payload["assessment_claim_sha256"]),
                str(payload["bootstrap_seed_receipt_sha256"]),
                int(payload["bootstrap_seed"]),
                str(payload["statistical_gate_spec_file_sha256"]),
                str(payload["gate_spec_sha256"]),
            )
            if binding in access_bindings:
                raise RegistryVerificationError("duplicate access_spend binding")
            access_bindings.add(binding)
            if partition == "development":
                if (
                    machine.development_gate_access_id is not None
                    or machine.confirmation_started
                ):
                    raise RegistryVerificationError(
                        "development access cannot follow development gate or confirmation"
                    )
            else:
                if machine.development_gate_access_id is None:
                    raise RegistryVerificationError(
                        "confirmation access requires development_gate_commit"
                    )
                machine.confirmation_started = True
            machine.open_access_id = access_id
            machine.open_partition = partition

        elif event_type == ASSESSMENT_COMMIT:
            access_id = str(payload["access_id"])
            if machine.open_access_id is None:
                raise RegistryVerificationError(
                    "assessment_commit has no open access_spend"
                )
            if access_id != machine.open_access_id:
                raise RegistryVerificationError(
                    "assessment_commit must immediately close the open access_id"
                )
            assert machine.open_partition is not None
            closed_accesses[access_id] = machine.open_partition
            if machine.open_partition == "development":
                machine.last_closed_development_access_id = access_id
            machine.open_access_id = None
            machine.open_partition = None

        else:
            development_access_id = str(payload["development_access_id"])
            if machine.open_access_id is not None:
                raise RegistryVerificationError(
                    "development gate cannot precede development assessment close"
                )
            if machine.development_gate_access_id is not None:
                raise RegistryVerificationError("duplicate development_gate_commit")
            if closed_accesses.get(development_access_id) != "development":
                raise RegistryVerificationError(
                    "development gate requires a closed development access"
                )
            if development_access_id != machine.last_closed_development_access_id:
                raise RegistryVerificationError(
                    "development gate must bind the latest closed development access"
                )
            machine.development_gate_access_id = development_access_id

        previous_hash = str(entry["entry_hash"])
    return machine


def _event_access_id(payload: Mapping[str, object]) -> str:
    if payload["event_type"] == DEVELOPMENT_GATE_COMMIT:
        return str(payload["development_access_id"])
    return str(payload["access_id"])


def _receipt_payload_for_entry(
    entry: Mapping[str, object], entry_count: int
) -> dict[str, object]:
    payload = entry["payload"]
    assert isinstance(payload, Mapping)
    return {
        "schema": HEAD_RECEIPT_SCHEMA,
        "registry_id": payload["registry_id"],
        "sequence": payload["sequence"],
        "entry_count": entry_count,
        "entry_hash": entry["entry_hash"],
        "event_type": payload["event_type"],
        "event_id": payload["event_id"],
        "event_payload_sha256": hashlib.sha256(
            canonical_json_bytes(dict(payload))
        ).hexdigest(),
        "signing_public_key": payload["signing_public_key"],
    }


def make_head_receipt(
    entry: Mapping[str, object],
    *,
    entry_count: int,
    signing_key: Ed25519PrivateKey,
) -> dict[str, object]:
    """Create a signed head receipt for storage outside the registry boundary."""

    raw_key = _signing_public_key(signing_key)
    public_key_hex = raw_key.hex()
    verified_entry = _verify_entry(
        dict(entry), raw_key=raw_key, expected_key_hex=public_key_hex
    )
    payload = _receipt_payload_for_entry(verified_entry, entry_count)
    if payload["sequence"] != entry_count - 1:
        raise RegistryVerificationError("receipt entry count differs")
    signature = signing_key.sign(canonical_json_bytes(payload)).hex()
    return {"payload": payload, "signature": signature}


def verify_head_receipt(
    receipt: object,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Strictly verify a signed externally retained head receipt."""

    raw_key = _public_key_bytes(verification_key)
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise RegistryVerificationError("head receipt schema differs")
    payload = receipt["payload"]
    if not isinstance(payload, dict) or set(payload) != _RECEIPT_PAYLOAD_KEYS:
        raise RegistryVerificationError("head receipt payload schema differs")
    if payload["schema"] != HEAD_RECEIPT_SCHEMA:
        raise RegistryVerificationError("head receipt version differs")
    _validate_identifier(payload["registry_id"], "receipt registry_id")
    _validate_identifier(payload["event_id"], "receipt event_id")
    if (
        type(payload["sequence"]) is not int
        or int(payload["sequence"]) < 0
        or type(payload["entry_count"]) is not int
        or payload["entry_count"] != payload["sequence"] + 1
    ):
        raise RegistryVerificationError("head receipt sequence differs")
    if payload["event_type"] not in EVENT_TYPES:
        raise RegistryVerificationError("head receipt event type differs")
    for key in ("entry_hash", "event_payload_sha256"):
        _validate_hash(payload[key], f"receipt {key}")
    if payload["signing_public_key"] != raw_key.hex():
        raise RegistryVerificationError("head receipt uses the wrong signing key")
    _verify_signature(raw_key, receipt["signature"], payload)
    return dict(payload)


def _state_from_entries(
    entries: list[dict[str, object]], machine: _MachineState
) -> RegistryState:
    head = entries[-1]
    payload = head["payload"]
    assert isinstance(payload, dict)
    return RegistryState(
        registry_id=str(payload["registry_id"]),
        entry_count=len(entries),
        head_sequence=int(payload["sequence"]),
        head_hash=str(head["entry_hash"]),
        head_event_type=str(payload["event_type"]),
        head_event_id=str(payload["event_id"]),
        head_access_id=_event_access_id(payload),
        signing_public_key=str(payload["signing_public_key"]),
        open_access_id=machine.open_access_id,
        development_gate_access_id=machine.development_gate_access_id,
        confirmation_started=machine.confirmation_started,
    )


def _verified_entries(
    path: Path,
    raw_key: bytes,
    *,
    expected_head_receipt: object | None = None,
    allow_extensions: bool = False,
) -> tuple[list[dict[str, object]], _MachineState]:
    entries = _load_entries(Path(path), raw_key)
    machine = _validate_chain(entries)
    if expected_head_receipt is not None:
        receipt = verify_head_receipt(expected_head_receipt, raw_key)
        receipt_count = int(receipt["entry_count"])
        if len(entries) < receipt_count:
            raise RegistryVerificationError(
                "registry was truncated below retained head"
            )
        expected_payload = _receipt_payload_for_entry(
            entries[receipt_count - 1], receipt_count
        )
        if receipt != expected_payload:
            raise RegistryVerificationError(
                "registry does not contain the externally retained head"
            )
        if not allow_extensions and len(entries) != receipt_count:
            raise RegistryVerificationError(
                "registry head differs from retained receipt"
            )
    return entries, machine


def verify_registry(
    path: Path,
    verification_key: bytes | Ed25519PublicKey,
    *,
    expected_head_receipt: object | None = None,
    allow_extensions: bool = False,
) -> RegistryState:
    """Verify all signatures, hashes, event transitions, and retained head.

    A supplied receipt must describe the exact current head by default.  Set
    ``allow_extensions`` only to prove that an older retained head remains an
    ancestor of a legitimately extended registry.
    """

    raw_key = _public_key_bytes(verification_key)
    entries, machine = _verified_entries(
        path,
        raw_key,
        expected_head_receipt=expected_head_receipt,
        allow_extensions=allow_extensions,
    )
    return _state_from_entries(entries, machine)


def verify_registry_events(
    path: Path,
    verification_key: bytes | Ed25519PublicKey,
    *,
    expected_head_receipt: object | None = None,
    allow_extensions: bool = False,
) -> tuple[VerifiedRegistryEvent, ...]:
    """Return immutable canonical events after complete registry verification.

    The payload mappings expose every event-specific binding for independent
    assessor or gate validation.  No event is returned unless the complete
    file, state machine, and optional externally held receipt all verify.
    """

    raw_key = _public_key_bytes(verification_key)
    entries, _ = _verified_entries(
        path,
        raw_key,
        expected_head_receipt=expected_head_receipt,
        allow_extensions=allow_extensions,
    )
    views = []
    for entry in entries:
        payload = entry["payload"]
        assert isinstance(payload, dict)
        views.append(
            VerifiedRegistryEvent(
                payload=MappingProxyType(dict(payload)),
                canonical_payload=canonical_json_bytes(payload),
                signature=str(entry["signature"]),
                entry_hash=str(entry["entry_hash"]),
            )
        )
    return tuple(views)


def serialize_head_receipt(receipt: Mapping[str, object]) -> bytes:
    """Return canonical bytes suitable for external receipt storage."""

    return canonical_json_bytes(dict(receipt)) + b"\n"


def _atomic_replace(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(
    path: Path,
    *,
    signing_key: Ed25519PrivateKey,
    registry_id: str,
    event_type: str,
    event_id: str,
    event_fields: Mapping[str, object],
    expected_previous_hash: str,
    expected_head_receipt: object | None,
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_key = _signing_public_key(signing_key)
    key_hex = raw_key.hex()
    _validate_identifier(registry_id, "registry_id")
    _validate_identifier(event_id, "event_id")
    _validate_hash(expected_previous_hash, "expected previous hash")

    lock_path = path.with_name(f".{path.name}.lock")
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        with os.fdopen(lock_descriptor, "r+b", closefd=True) as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if path.exists() or path.is_symlink():
                if (
                    not path.is_symlink()
                    and path.stat().st_size > 0
                    and expected_head_receipt is None
                ):
                    raise RegistryVerificationError(
                        "existing registry append requires externally retained head receipt"
                    )
                state = verify_registry(
                    path,
                    raw_key,
                    expected_head_receipt=expected_head_receipt,
                )
                entries, existing_bytes = _load_entries_with_raw(path, raw_key)
                current_hash = state.head_hash
                if state.registry_id != registry_id:
                    raise RegistryVerificationError(
                        "registry_id differs from existing chain"
                    )
            else:
                if expected_head_receipt is not None:
                    raise RegistryVerificationError(
                        "retained head supplied for a missing registry"
                    )
                entries = []
                current_hash = GENESIS_PREVIOUS_HASH
                existing_bytes = b""
            if current_hash != expected_previous_hash:
                raise ConcurrentAppendError(
                    "expected previous hash differs from current head"
                )

            payload = {
                "schema": ENTRY_SCHEMA,
                "registry_id": registry_id,
                "sequence": len(entries),
                "previous_hash": current_hash,
                "event_type": event_type,
                "event_id": event_id,
                "signing_public_key": key_hex,
                **dict(event_fields),
            }
            entry = _make_signed_entry(payload, signing_key)
            verified_entry = _verify_entry(
                entry, raw_key=raw_key, expected_key_hex=key_hex
            )
            candidate_entries = [*entries, verified_entry]
            _validate_chain(candidate_entries)
            _atomic_replace(
                path,
                existing_bytes + canonical_json_bytes(verified_entry) + b"\n",
            )
            receipt = make_head_receipt(
                verified_entry,
                entry_count=len(candidate_entries),
                signing_key=signing_key,
            )
            verified = verify_registry(path, raw_key, expected_head_receipt=receipt)
            if verified.head_hash != verified_entry["entry_hash"]:
                raise RegistryVerificationError(
                    "post-append registry verification differs"
                )
            return receipt
    except OSError as exc:
        raise RegistryVerificationError(
            "access registry lock or append failed"
        ) from exc


def append_access_spend(
    path: Path,
    *,
    signing_key: Ed25519PrivateKey,
    registry_id: str,
    event_id: str,
    access_id: str,
    partition: str,
    manifest_sha256: str,
    board_sha256: str,
    run_contract_sha256: str,
    runtime_bundle_sha256: str,
    assessment_claim_sha256: str,
    bootstrap_seed_receipt_sha256: str,
    bootstrap_seed: int,
    statistical_gate_spec_file_sha256: str,
    gate_spec_sha256: str,
    expected_previous_hash: str,
    expected_head_receipt: object | None = None,
) -> dict[str, object]:
    """Append an access spend without prematurely binding an assessment."""

    _validate_identifier(access_id, "access_id")
    if partition not in PARTITIONS:
        raise RegistryVerificationError("invalid access partition")
    for value, label in (
        (manifest_sha256, "manifest_sha256"),
        (board_sha256, "board_sha256"),
        (run_contract_sha256, "run_contract_sha256"),
        (runtime_bundle_sha256, "runtime_bundle_sha256"),
        (assessment_claim_sha256, "assessment_claim_sha256"),
        (bootstrap_seed_receipt_sha256, "bootstrap_seed_receipt_sha256"),
        (
            statistical_gate_spec_file_sha256,
            "statistical_gate_spec_file_sha256",
        ),
        (gate_spec_sha256, "gate_spec_sha256"),
    ):
        _validate_hash(value, label)
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise RegistryVerificationError("invalid bootstrap_seed")
    return _append_event(
        path,
        signing_key=signing_key,
        registry_id=registry_id,
        event_type=ACCESS_SPEND,
        event_id=event_id,
        event_fields={
            "access_id": access_id,
            "partition": partition,
            "manifest_sha256": manifest_sha256,
            "board_sha256": board_sha256,
            "run_contract_sha256": run_contract_sha256,
            "runtime_bundle_sha256": runtime_bundle_sha256,
            "assessment_claim_sha256": assessment_claim_sha256,
            "bootstrap_seed_receipt_sha256": bootstrap_seed_receipt_sha256,
            "bootstrap_seed": bootstrap_seed,
            "statistical_gate_spec_file_sha256": (
                statistical_gate_spec_file_sha256
            ),
            "gate_spec_sha256": gate_spec_sha256,
        },
        expected_previous_hash=expected_previous_hash,
        expected_head_receipt=expected_head_receipt,
    )


def append_assessment_commit(
    path: Path,
    *,
    signing_key: Ed25519PrivateKey,
    registry_id: str,
    event_id: str,
    access_id: str,
    assessment_sha256: str,
    statistical_gate_spec_file_sha256: str,
    gate_spec_sha256: str,
    expected_previous_hash: str,
    expected_head_receipt: object | None = None,
) -> dict[str, object]:
    """Immediately close the open access and bind its produced assessment."""

    _validate_identifier(access_id, "access_id")
    for value, label in (
        (assessment_sha256, "assessment_sha256"),
        (
            statistical_gate_spec_file_sha256,
            "statistical_gate_spec_file_sha256",
        ),
        (gate_spec_sha256, "gate_spec_sha256"),
    ):
        _validate_hash(value, label)
    return _append_event(
        path,
        signing_key=signing_key,
        registry_id=registry_id,
        event_type=ASSESSMENT_COMMIT,
        event_id=event_id,
        event_fields={
            "access_id": access_id,
            "assessment_sha256": assessment_sha256,
            "statistical_gate_spec_file_sha256": (
                statistical_gate_spec_file_sha256
            ),
            "gate_spec_sha256": gate_spec_sha256,
        },
        expected_previous_hash=expected_previous_hash,
        expected_head_receipt=expected_head_receipt,
    )


def append_development_gate_commit(
    path: Path,
    *,
    signing_key: Ed25519PrivateKey,
    registry_id: str,
    event_id: str,
    development_access_id: str,
    development_gate_receipt_sha256: str,
    expected_previous_hash: str,
    expected_head_receipt: object | None = None,
) -> dict[str, object]:
    """Bind the gate receipt for the latest closed development access."""

    _validate_identifier(development_access_id, "development_access_id")
    _validate_hash(development_gate_receipt_sha256, "development_gate_receipt_sha256")
    return _append_event(
        path,
        signing_key=signing_key,
        registry_id=registry_id,
        event_type=DEVELOPMENT_GATE_COMMIT,
        event_id=event_id,
        event_fields={
            "development_access_id": development_access_id,
            "development_gate_receipt_sha256": development_gate_receipt_sha256,
        },
        expected_previous_hash=expected_previous_hash,
        expected_head_receipt=expected_head_receipt,
    )
