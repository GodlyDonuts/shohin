"""Canonical architecture/objective receipts for HSC qualification arms.

This receipt deliberately excludes optimizer updates, measured runtime, and
dataset resources. Those belong to the complete qualification resource
receipt and must not be guessed before a fit is authorized. The purpose here
is narrower: bind the exact HSC mechanism, supervision contract, control
transform, and parameter arithmetic before any neural result is observed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import re

from episode_functor_capacity_lanes import HANKEL_SHIFT_MAXIMUM_EXPECTED
from episode_functor_hankel_completion import HankelShiftCompletionProjector
from episode_functor_learned_system import (
    GLOBAL_PARAMETER_LIMIT,
    PROTECTED_SHOHIN_PARAMETERS,
    PROTECTED_SHOHIN_SHA256,
)
from episode_functor_qualification_loss import EFCHankelQualificationLoss
from episode_functor_query_parser import NeuralOpaqueQueryParser
from episode_functor_witness_compiler import ProofCarryingWitnessCompiler
from pipeline.episode_functor_hankel_geometry import enumerate_action_words
from pipeline.episode_functor_machine_contract import MACHINE_SIZE


HANKEL_ARM_SCHEMA = "efc-hankel-arm-architecture-objective/v2"
SUPERVISION_CONTRACT = (
    "post-forward-source-sha256-fixed-prefix-categorical/v2"
)
SOURCE_BOUNDARY = "candidate-source-only-no-query-no-target/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ARM_NAMES = {
    ("hankel-shift", "prefix"): "hsc-prefix-treatment",
    ("hankel-shift", "random"): "hsc-position-scramble-control",
    ("hankel-shift", "commutative"): "hsc-stable-bag-control",
    ("direct-base", "prefix"): "hsc-dual-direct-decode-control",
}


class HankelArmReceiptError(ValueError):
    """An HSC architecture/objective receipt failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _float_hex(value: float, *, label: str) -> str:
    if not isinstance(value, float) or not math.isfinite(value):
        raise HankelArmReceiptError(f"{label} must be a finite float")
    return value.hex()


def _incidence_mapping(projector: HankelShiftCompletionProjector) -> dict[str, object]:
    incidence = projector.shift_incidence.detach().cpu()
    return {
        "dtype": "int64",
        "shape": list(incidence.shape),
        "values": incidence.tolist(),
    }


def _loss_mapping(objective: EFCHankelQualificationLoss) -> dict[str, str]:
    result = {
        field.name: _float_hex(
            getattr(objective.weights, field.name),
            label=f"base loss {field.name}",
        )
        for field in fields(objective.weights)
    }
    result.update(
        {
            f"hankel_{field.name}": _float_hex(
                getattr(objective.hankel_weights, field.name),
                label=f"Hankel loss {field.name}",
            )
            for field in fields(objective.hankel_weights)
        }
    )
    result["syndrome_margin"] = _float_hex(
        objective.syndrome_margin,
        label="syndrome margin",
    )
    result["state_separation_margin"] = _float_hex(
        objective.state_separation_margin,
        label="state separation margin",
    )
    return result


@dataclass(frozen=True, slots=True)
class HankelArmReceipt:
    """Self-hashed receipt for one isoparametric HSC arm."""

    arm_name: str
    decode_mode: str
    incidence_mode: str
    incidence_sha256: str
    random_seed_sha256: str
    max_depth: int
    word_count: int
    base_signature_cells_per_example: int
    derivative_signature_cells_per_example: int
    signature_target_bits_per_example: int
    temperature_hex: str
    objective_sha256: str
    source_compiler_parameters: int
    projector_parameters: int
    query_parser_parameters: int
    qualification_trainable_parameters: int
    complete_parameters: int
    headroom: int
    persistent_machine_bytes: int
    protected_checkpoint_sha256: str
    source_boundary: str
    supervision_contract: str
    receipt_sha256: str
    schema: str = HANKEL_ARM_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANKEL_ARM_SCHEMA
            or (self.decode_mode, self.incidence_mode) not in _ARM_NAMES
            or self.arm_name
            != _ARM_NAMES[(self.decode_mode, self.incidence_mode)]
            or self.source_boundary != SOURCE_BOUNDARY
            or self.supervision_contract != SUPERVISION_CONTRACT
        ):
            raise HankelArmReceiptError("HSC arm identity differs")
        for name in (
            "incidence_sha256",
            "random_seed_sha256",
            "objective_sha256",
            "protected_checkpoint_sha256",
            "receipt_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise HankelArmReceiptError(f"HSC {name} differs")
        if self.protected_checkpoint_sha256 != PROTECTED_SHOHIN_SHA256:
            raise HankelArmReceiptError("HSC protected checkpoint differs")
        if self.max_depth != 3 or self.word_count != 40:
            raise HankelArmReceiptError("HSC signature depth differs")
        if (
            self.base_signature_cells_per_example != 640
            or self.derivative_signature_cells_per_example != 1_920
            or self.signature_target_bits_per_example != 5_120
        ):
            raise HankelArmReceiptError("HSC signature target accounting differs")
        if self.temperature_hex != float(0.05).hex():
            raise HankelArmReceiptError("HSC temperature differs")
        expected = HANKEL_SHIFT_MAXIMUM_EXPECTED
        if (
            self.source_compiler_parameters != expected.compiler_parameters
            or self.projector_parameters != expected.projector_parameters
            or self.query_parser_parameters != expected.query_parameters
            or self.qualification_trainable_parameters
            != expected.compiler_parameters
            or self.complete_parameters != expected.complete_parameters
            or self.headroom != expected.headroom
            or self.persistent_machine_bytes != MACHINE_SIZE
            or self.complete_parameters
            != PROTECTED_SHOHIN_PARAMETERS
            + self.source_compiler_parameters
            + self.query_parser_parameters
            or self.headroom != GLOBAL_PARAMETER_LIMIT - self.complete_parameters
        ):
            raise HankelArmReceiptError("HSC parameter or state receipt differs")
        if self.receipt_sha256 != _sha256_json(self._unsigned_mapping()):
            raise HankelArmReceiptError("HSC arm receipt hash differs")

    def _unsigned_mapping(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "receipt_sha256"
        }

    def to_mapping(self) -> dict[str, object]:
        result = self._unsigned_mapping()
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_mapping()) + b"\n"

    def matched_resource_signature(self) -> dict[str, object]:
        """Return noncausal fields that must match across every control."""

        excluded = {
            "arm_name",
            "decode_mode",
            "incidence_mode",
            "incidence_sha256",
            "random_seed_sha256",
            "receipt_sha256",
        }
        return {
            key: value
            for key, value in self._unsigned_mapping().items()
            if key not in excluded
        }

    def assert_isoparametric(self, other: HankelArmReceipt) -> None:
        """Reject comparisons that change zero or multiple causal factors."""

        if (
            not isinstance(other, HankelArmReceipt)
            or self.matched_resource_signature()
            != other.matched_resource_signature()
        ):
            raise HankelArmReceiptError(
                "HSC treatment and control resources differ"
            )
        decode_changed = self.decode_mode != other.decode_mode
        incidence_changed = self.incidence_sha256 != other.incidence_sha256
        if decode_changed == incidence_changed:
            raise HankelArmReceiptError(
                "HSC comparison must change exactly one causal factor"
            )

    def assert_decode_control(self, other: HankelArmReceipt) -> None:
        """Require an incidence-identical arm differing only in decoding."""

        self.assert_isoparametric(other)
        if (
            self.decode_mode == other.decode_mode
            or self.incidence_mode != other.incidence_mode
            or self.incidence_sha256 != other.incidence_sha256
            or self.random_seed_sha256 != other.random_seed_sha256
        ):
            raise HankelArmReceiptError(
                "HSC decode control changes incidence or not decoding"
            )

    def assert_incidence_control(self, other: HankelArmReceipt) -> None:
        """Require a decoder-identical arm differing only in incidence."""

        self.assert_isoparametric(other)
        if (
            self.decode_mode != other.decode_mode
            or self.incidence_sha256 == other.incidence_sha256
        ):
            raise HankelArmReceiptError(
                "HSC incidence control changes decoding or not incidence"
            )


def create_hankel_arm_receipt(
    *,
    compiler: ProofCarryingWitnessCompiler,
    query_parser: NeuralOpaqueQueryParser,
    objective: EFCHankelQualificationLoss,
) -> HankelArmReceipt:
    """Bind an actual maximum-lane HSC constructor and objective."""

    if (
        not isinstance(compiler, ProofCarryingWitnessCompiler)
        or not isinstance(query_parser, NeuralOpaqueQueryParser)
        or not isinstance(objective, EFCHankelQualificationLoss)
        or not isinstance(compiler.projector, HankelShiftCompletionProjector)
    ):
        raise HankelArmReceiptError("HSC arm constructor differs")
    projector = compiler.projector
    identity = (projector.decode_mode, projector.incidence_mode)
    if identity not in _ARM_NAMES:
        raise HankelArmReceiptError("HSC incidence mode differs")
    words = enumerate_action_words(projector.max_depth)
    base_cells = 8 * len(words) * 2
    derivative_cells = 3 * base_cells
    unsigned = {
        "arm_name": _ARM_NAMES[identity],
        "decode_mode": projector.decode_mode,
        "incidence_mode": projector.incidence_mode,
        "incidence_sha256": _sha256_json(_incidence_mapping(projector)),
        "random_seed_sha256": sha256(projector.random_seed.encode("utf-8")).hexdigest(),
        "max_depth": projector.max_depth,
        "word_count": len(words),
        "base_signature_cells_per_example": base_cells,
        "derivative_signature_cells_per_example": derivative_cells,
        "signature_target_bits_per_example": 2 * (base_cells + derivative_cells),
        "temperature_hex": _float_hex(
            projector.temperature,
            label="HSC temperature",
        ),
        "objective_sha256": _sha256_json(_loss_mapping(objective)),
        "source_compiler_parameters": compiler.parameter_count(),
        "projector_parameters": projector.parameter_count(),
        "query_parser_parameters": query_parser.parameter_count(),
        "qualification_trainable_parameters": compiler.parameter_count(),
        "complete_parameters": (
            PROTECTED_SHOHIN_PARAMETERS
            + compiler.parameter_count()
            + query_parser.parameter_count()
        ),
        "headroom": (
            GLOBAL_PARAMETER_LIMIT
            - PROTECTED_SHOHIN_PARAMETERS
            - compiler.parameter_count()
            - query_parser.parameter_count()
        ),
        "persistent_machine_bytes": MACHINE_SIZE,
        "protected_checkpoint_sha256": PROTECTED_SHOHIN_SHA256,
        "source_boundary": SOURCE_BOUNDARY,
        "supervision_contract": SUPERVISION_CONTRACT,
        "schema": HANKEL_ARM_SCHEMA,
    }
    return HankelArmReceipt(
        **unsigned,
        receipt_sha256=_sha256_json(unsigned),
    )


__all__ = [
    "HANKEL_ARM_SCHEMA",
    "HankelArmReceipt",
    "HankelArmReceiptError",
    "SOURCE_BOUNDARY",
    "SUPERVISION_CONTRACT",
    "create_hankel_arm_receipt",
]
