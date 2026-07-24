"""Fail-closed EFC qualification resource and custody receipts.

This module only validates and serializes caller-supplied resource facts. It
does not inspect artifacts, read environment variables, or launch work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import re
from typing import Literal

from pipeline.episode_functor_wire_protocol import MACHINE_SIZE


RECEIPT_SCHEMA = "efc-qualification-resource-custody/v1"
TOTAL_PARAMETER_LIMIT_EXCLUSIVE = 200_000_000
PERSISTENT_BYTE_LIMIT = MACHINE_SIZE

Basis = Literal["forecast", "measured"]
ReceiptKind = Literal["forecast", "measured", "mixed"]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_FIELDS = frozenset({"board_sha256", "source_sha256", "config_sha256"})
_OBSERVATION_FIELDS = frozenset({"basis", "value"})
_LIMIT_FIELDS = frozenset({"persistent_bytes_max", "total_parameters_exclusive"})
_RECEIPT_FIELDS = frozenset(
    {
        "bindings",
        "limits",
        "receipt_kind",
        "receipt_sha256",
        "resources",
        "schema",
    }
)


class ResourceReceiptError(ValueError):
    """Raised when an EFC resource receipt fails closed."""


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest without reading from the filesystem."""
    if not isinstance(payload, bytes):
        raise TypeError("EFC digest payload must be bytes")
    return sha256(payload).hexdigest()


def _require_exact_dict(
    value: object,
    expected_keys: frozenset[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ResourceReceiptError(f"EFC {context} schema differs")
    return value


def _require_plain_int(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResourceReceiptError(f"EFC {context} must be an integer")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ResourceReceiptError(f"EFC {context} digest differs")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ResourceReceiptError("EFC receipt contains a duplicate key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ArtifactBindings:
    """Immutable digests for the exact board, source corpus, and config."""

    board_sha256: str
    source_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        for name in _BINDING_FIELDS:
            _require_sha256(getattr(self, name), context=name)

    def to_mapping(self) -> dict[str, str]:
        return {
            "board_sha256": self.board_sha256,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ArtifactBindings:
        mapping = _require_exact_dict(
            value,
            _BINDING_FIELDS,
            context="artifact binding",
        )
        return cls(
            board_sha256=_require_sha256(
                mapping["board_sha256"],
                context="board_sha256",
            ),
            source_sha256=_require_sha256(
                mapping["source_sha256"],
                context="source_sha256",
            ),
            config_sha256=_require_sha256(
                mapping["config_sha256"],
                context="config_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    """One exact resource value and whether it is forecast or measured."""

    value: int
    basis: Basis

    def __post_init__(self) -> None:
        value = _require_plain_int(self.value, context="resource value")
        if value < 0:
            raise ResourceReceiptError("EFC resource value is negative")
        if self.basis not in {"forecast", "measured"}:
            raise ResourceReceiptError("EFC resource basis differs")

    def to_mapping(self) -> dict[str, int | str]:
        return {"basis": self.basis, "value": self.value}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
    ) -> ResourceObservation:
        mapping = _require_exact_dict(
            value,
            _OBSERVATION_FIELDS,
            context=f"{context} observation",
        )
        raw_basis = mapping["basis"]
        if raw_basis not in {"forecast", "measured"}:
            raise ResourceReceiptError(f"EFC {context} basis differs")
        return cls(
            value=_require_plain_int(
                mapping["value"],
                context=f"{context} value",
            ),
            basis=raw_basis,
        )


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """The complete preregistered EFC qualification resource vector."""

    examples: ResourceObservation
    target_bits: ResourceObservation
    source_bytes: ResourceObservation
    oracle_calls: ResourceObservation
    updates: ResourceObservation
    trainable_parameters: ResourceObservation
    total_parameters: ResourceObservation
    optimizer_bytes: ResourceObservation
    compiler_flops: ResourceObservation | None
    compiler_time_ns: ResourceObservation | None
    persistent_bytes: ResourceObservation
    executor_flops_per_query: ResourceObservation

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {"compiler_flops", "compiler_time_ns"}:
                if value is not None and not isinstance(value, ResourceObservation):
                    raise ResourceReceiptError(f"EFC {item.name} observation differs")
            elif not isinstance(value, ResourceObservation):
                raise ResourceReceiptError(f"EFC {item.name} observation differs")

        if (self.compiler_flops is None) == (self.compiler_time_ns is None):
            raise ResourceReceiptError(
                "EFC compiler cost requires exactly one of FLOPs or time"
            )
        if (
            self.compiler_time_ns is not None
            and self.compiler_time_ns.basis != "measured"
        ):
            raise ResourceReceiptError("EFC compiler time must be explicitly measured")

        for name in (
            "examples",
            "target_bits",
            "source_bytes",
            "total_parameters",
            "persistent_bytes",
            "executor_flops_per_query",
        ):
            if getattr(self, name).value <= 0:
                raise ResourceReceiptError(f"EFC {name} must be positive")
        compiler_cost = self.compiler_flops or self.compiler_time_ns
        if compiler_cost is None or compiler_cost.value <= 0:
            raise ResourceReceiptError("EFC compiler cost must be positive")

        trainable = self.trainable_parameters.value
        total = self.total_parameters.value
        updates = self.updates.value
        optimizer_bytes = self.optimizer_bytes.value
        if trainable > total:
            raise ResourceReceiptError(
                "EFC trainable parameters exceed total parameters"
            )
        if total >= TOTAL_PARAMETER_LIMIT_EXCLUSIVE:
            raise ResourceReceiptError(
                "EFC total parameters leave the preregistered bound"
            )
        if updates > 0 and trainable == 0:
            raise ResourceReceiptError("EFC updates require trainable parameters")
        if optimizer_bytes > 0 and trainable == 0:
            raise ResourceReceiptError(
                "EFC optimizer state requires trainable parameters"
            )
        if self.persistent_bytes.value > PERSISTENT_BYTE_LIMIT:
            raise ResourceReceiptError(
                "EFC persistent state leaves the deployed machine bound"
            )

    @property
    def receipt_kind(self) -> ReceiptKind:
        bases = {observation.basis for observation in self.observations()}
        if bases == {"forecast"}:
            return "forecast"
        if bases == {"measured"}:
            return "measured"
        return "mixed"

    def observations(self) -> tuple[ResourceObservation, ...]:
        values: list[ResourceObservation] = []
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None:
                values.append(value)
        return tuple(values)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = None if value is None else value.to_mapping()
        return result

    @classmethod
    def from_mapping(cls, value: object) -> ResourceVector:
        expected = frozenset(item.name for item in fields(cls))
        mapping = _require_exact_dict(
            value,
            expected,
            context="resource vector",
        )
        parsed: dict[str, ResourceObservation | None] = {}
        for name in expected:
            raw = mapping[name]
            if name in {"compiler_flops", "compiler_time_ns"} and raw is None:
                parsed[name] = None
            else:
                parsed[name] = ResourceObservation.from_mapping(
                    raw,
                    context=name,
                )
        return cls(**parsed)


def _limits_mapping() -> dict[str, int]:
    return {
        "persistent_bytes_max": PERSISTENT_BYTE_LIMIT,
        "total_parameters_exclusive": TOTAL_PARAMETER_LIMIT_EXCLUSIVE,
    }


def _validate_limits(value: object) -> None:
    mapping = _require_exact_dict(
        value,
        _LIMIT_FIELDS,
        context="resource limit",
    )
    for key, expected in _limits_mapping().items():
        observed = _require_plain_int(
            mapping[key],
            context=f"{key} limit",
        )
        if observed != expected:
            raise ResourceReceiptError(f"EFC {key} limit differs")


@dataclass(frozen=True, slots=True)
class QualificationResourceReceipt:
    """Canonical, self-hashed custody receipt for one EFC qualification arm."""

    bindings: ArtifactBindings
    resources: ResourceVector
    receipt_sha256: str
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ResourceReceiptError("EFC resource receipt version differs")
        if not isinstance(self.bindings, ArtifactBindings):
            raise ResourceReceiptError("EFC resource receipt bindings differ")
        if not isinstance(self.resources, ResourceVector):
            raise ResourceReceiptError("EFC resource receipt vector differs")
        _require_sha256(
            self.receipt_sha256,
            context="receipt_sha256",
        )
        if self.receipt_sha256 != sha256_bytes(
            _canonical_json_bytes(self._unsigned_mapping())
        ):
            raise ResourceReceiptError("EFC resource receipt hash differs")

    @classmethod
    def create(
        cls,
        *,
        bindings: ArtifactBindings,
        resources: ResourceVector,
    ) -> QualificationResourceReceipt:
        if not isinstance(bindings, ArtifactBindings):
            raise ResourceReceiptError("EFC resource receipt bindings differ")
        if not isinstance(resources, ResourceVector):
            raise ResourceReceiptError("EFC resource receipt vector differs")
        unsigned = {
            "bindings": bindings.to_mapping(),
            "limits": _limits_mapping(),
            "receipt_kind": resources.receipt_kind,
            "resources": resources.to_mapping(),
            "schema": RECEIPT_SCHEMA,
        }
        return cls(
            bindings=bindings,
            resources=resources,
            receipt_sha256=sha256_bytes(_canonical_json_bytes(unsigned)),
        )

    @property
    def receipt_kind(self) -> ReceiptKind:
        return self.resources.receipt_kind

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            "bindings": self.bindings.to_mapping(),
            "limits": _limits_mapping(),
            "receipt_kind": self.receipt_kind,
            "resources": self.resources.to_mapping(),
            "schema": self.schema,
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        """Return the sole accepted canonical on-disk representation."""
        return _canonical_json_bytes(self.to_mapping()) + b"\n"

    def assert_bindings(self, expected: ArtifactBindings) -> None:
        if not isinstance(expected, ArtifactBindings):
            raise ResourceReceiptError("EFC expected bindings differ")
        if self.bindings != expected:
            raise ResourceReceiptError("EFC artifact custody binding differs")

    @classmethod
    def from_mapping(cls, value: object) -> QualificationResourceReceipt:
        mapping = _require_exact_dict(
            value,
            _RECEIPT_FIELDS,
            context="resource receipt",
        )
        if mapping["schema"] != RECEIPT_SCHEMA:
            raise ResourceReceiptError("EFC resource receipt version differs")
        _validate_limits(mapping["limits"])
        bindings = ArtifactBindings.from_mapping(mapping["bindings"])
        resources = ResourceVector.from_mapping(mapping["resources"])
        if mapping["receipt_kind"] != resources.receipt_kind:
            raise ResourceReceiptError("EFC resource receipt kind differs")
        return cls(
            bindings=bindings,
            resources=resources,
            receipt_sha256=_require_sha256(
                mapping["receipt_sha256"],
                context="receipt_sha256",
            ),
            schema=RECEIPT_SCHEMA,
        )

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes,
    ) -> QualificationResourceReceipt:
        if not isinstance(payload, bytes):
            raise TypeError("EFC receipt payload must be bytes")
        try:
            decoded = json.loads(
                payload.decode("ascii"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ResourceReceiptError(
                        f"EFC receipt contains non-finite value {value}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResourceReceiptError("EFC receipt JSON differs") from exc
        receipt = cls.from_mapping(decoded)
        if payload != receipt.to_json_bytes():
            raise ResourceReceiptError("EFC receipt is not canonical custody bytes")
        return receipt


def create_resource_receipt(
    *,
    bindings: ArtifactBindings,
    resources: ResourceVector,
) -> QualificationResourceReceipt:
    """Create a canonical receipt without launching or measuring any work."""
    return QualificationResourceReceipt.create(
        bindings=bindings,
        resources=resources,
    )


def load_resource_receipt(payload: bytes) -> QualificationResourceReceipt:
    """Validate canonical receipt bytes and return the frozen receipt."""
    return QualificationResourceReceipt.from_json_bytes(payload)


__all__ = [
    "ArtifactBindings",
    "PERSISTENT_BYTE_LIMIT",
    "QualificationResourceReceipt",
    "RECEIPT_SCHEMA",
    "ResourceObservation",
    "ResourceReceiptError",
    "ResourceVector",
    "TOTAL_PARAMETER_LIMIT_EXCLUSIVE",
    "create_resource_receipt",
    "load_resource_receipt",
    "sha256_bytes",
]
