"""Named, auditable capacity lanes for learned EFC no-host treatments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from episode_functor_learned_completion import (
    LearnedRelationalCompletionProjector,
)
from episode_functor_hankel_completion import (
    DirectDualCompletionControlProjector,
    HankelShiftCompletionProjector,
)
from episode_functor_learned_system import (
    GLOBAL_PARAMETER_LIMIT,
    PROTECTED_SHOHIN_PARAMETERS,
)
from episode_functor_query_parser import NeuralOpaqueQueryParser
from episode_functor_witness_compiler import ProofCarryingWitnessCompiler


class EFCCapacityError(ValueError):
    """A named capacity lane or its exact parameter receipt differs."""


@dataclass(frozen=True, slots=True)
class EFCCapacityLane:
    name: str
    compiler_width: int
    compiler_encoder_layers: int
    compiler_decoder_layers: int
    compiler_heads: int
    compiler_feedforward: int
    completer_width: int
    completer_iterations: int
    query_width: int
    query_layers: int
    query_heads: int
    query_feedforward: int
    expected_compiler_parameters: int
    expected_completer_parameters: int
    expected_query_parameters: int

    @property
    def expected_added_parameters(self) -> int:
        return (
            self.expected_compiler_parameters
            + self.expected_query_parameters
        )

    @property
    def expected_complete_parameters(self) -> int:
        return PROTECTED_SHOHIN_PARAMETERS + self.expected_added_parameters

    @property
    def expected_headroom(self) -> int:
        return GLOBAL_PARAMETER_LIMIT - self.expected_complete_parameters


@dataclass(frozen=True, slots=True)
class EFCCapacityReceipt:
    lane: str
    compiler_parameters: int
    completer_parameters: int
    query_parameters: int
    added_parameters: int
    complete_parameters: int
    headroom: int


@dataclass(frozen=True, slots=True)
class EFCHankelCapacityReceipt:
    lane: str
    incidence_mode: str
    decode_mode: str
    compiler_parameters: int
    projector_parameters: int
    query_parameters: int
    added_parameters: int
    complete_parameters: int
    headroom: int


MINIMAL_ATTRIBUTION_LANE: Final = EFCCapacityLane(
    name="minimal",
    compiler_width=192,
    compiler_encoder_layers=4,
    compiler_decoder_layers=2,
    compiler_heads=6,
    compiler_feedforward=768,
    completer_width=96,
    completer_iterations=4,
    query_width=128,
    query_layers=2,
    query_heads=4,
    query_feedforward=512,
    expected_compiler_parameters=3_821_202,
    expected_completer_parameters=225_410,
    expected_query_parameters=728_993,
)

WIDE_LANE: Final = EFCCapacityLane(
    name="wide",
    compiler_width=384,
    compiler_encoder_layers=8,
    compiler_decoder_layers=4,
    compiler_heads=12,
    compiler_feedforward=1_536,
    completer_width=512,
    completer_iterations=8,
    query_width=256,
    query_layers=4,
    query_heads=8,
    query_feedforward=1_024,
    expected_compiler_parameters=31_673_746,
    expected_completer_parameters=6_313_986,
    expected_query_parameters=3_951_521,
)

MAXIMUM_PREREG_LANE: Final = EFCCapacityLane(
    name="maximum",
    compiler_width=512,
    compiler_encoder_layers=8,
    compiler_decoder_layers=4,
    compiler_heads=16,
    compiler_feedforward=2_048,
    completer_width=640,
    completer_iterations=8,
    query_width=320,
    query_layers=4,
    query_heads=10,
    query_feedforward=1_280,
    expected_compiler_parameters=54_549_394,
    expected_completer_parameters=9_858_562,
    expected_query_parameters=6_003_489,
)

CAPACITY_LANES: Final = MappingProxyType(
    {
        lane.name: lane
        for lane in (
            MINIMAL_ATTRIBUTION_LANE,
            WIDE_LANE,
            MAXIMUM_PREREG_LANE,
        )
    }
)

HANKEL_SHIFT_MAXIMUM_EXPECTED: Final = EFCHankelCapacityReceipt(
    lane="maximum-hankel-shift",
    incidence_mode="prefix",
    decode_mode="hankel-shift",
    compiler_parameters=64_407_956,
    projector_parameters=19_717_124,
    query_parameters=6_003_489,
    added_parameters=70_411_445,
    complete_parameters=195_493_109,
    headroom=4_506_891,
)


def build_no_host_capacity_lane(
    name: str,
    *,
    external_feature_width: int,
) -> tuple[
    ProofCarryingWitnessCompiler,
    NeuralOpaqueQueryParser,
    EFCCapacityReceipt,
]:
    try:
        lane = CAPACITY_LANES[name]
    except KeyError as exc:
        raise EFCCapacityError(f"unknown EFC capacity lane: {name}") from exc
    if external_feature_width < 0:
        raise EFCCapacityError("external feature width is negative")

    completer = LearnedRelationalCompletionProjector(
        width=lane.completer_width,
        iterations=lane.completer_iterations,
    )
    compiler = ProofCarryingWitnessCompiler(
        width=lane.compiler_width,
        encoder_layers=lane.compiler_encoder_layers,
        decoder_layers=lane.compiler_decoder_layers,
        heads=lane.compiler_heads,
        feedforward=lane.compiler_feedforward,
        external_feature_width=external_feature_width,
        projector=completer,
    )
    query = NeuralOpaqueQueryParser(
        width=lane.query_width,
        layers=lane.query_layers,
        heads=lane.query_heads,
        feedforward=lane.query_feedforward,
        external_feature_width=external_feature_width,
    )
    receipt = EFCCapacityReceipt(
        lane=lane.name,
        compiler_parameters=compiler.parameter_count(),
        completer_parameters=completer.parameter_count(),
        query_parameters=query.parameter_count(),
        added_parameters=compiler.parameter_count() + query.parameter_count(),
        complete_parameters=(
            PROTECTED_SHOHIN_PARAMETERS
            + compiler.parameter_count()
            + query.parameter_count()
        ),
        headroom=(
            GLOBAL_PARAMETER_LIMIT
            - PROTECTED_SHOHIN_PARAMETERS
            - compiler.parameter_count()
            - query.parameter_count()
        ),
    )
    expected = EFCCapacityReceipt(
        lane=lane.name,
        compiler_parameters=lane.expected_compiler_parameters,
        completer_parameters=lane.expected_completer_parameters,
        query_parameters=lane.expected_query_parameters,
        added_parameters=lane.expected_added_parameters,
        complete_parameters=lane.expected_complete_parameters,
        headroom=lane.expected_headroom,
    )
    if receipt != expected or receipt.headroom <= 0:
        raise EFCCapacityError(
            f"EFC capacity receipt differs for lane {lane.name}"
        )
    return compiler, query, receipt


def build_hankel_shift_capacity_lane(
    *,
    external_feature_width: int,
    incidence_mode: str = "prefix",
    random_seed: str = "efc-hankel-random-control-v1",
    decode_mode: str = "hankel-shift",
) -> tuple[
    ProofCarryingWitnessCompiler,
    NeuralOpaqueQueryParser,
    EFCHankelCapacityReceipt,
]:
    """Build the preregistered maximum HSC arm or an isoparametric control."""

    if external_feature_width < 0:
        raise EFCCapacityError("external feature width is negative")
    if incidence_mode not in ("prefix", "random", "commutative"):
        raise EFCCapacityError(
            f"unknown Hankel incidence mode: {incidence_mode}"
        )
    if decode_mode not in ("hankel-shift", "direct-base"):
        raise EFCCapacityError(
            f"unknown Hankel decode mode: {decode_mode}"
        )
    lane = MAXIMUM_PREREG_LANE
    projector_type = (
        HankelShiftCompletionProjector
        if decode_mode == "hankel-shift"
        else DirectDualCompletionControlProjector
    )
    projector = projector_type(
        width=lane.completer_width,
        iterations=lane.completer_iterations,
        max_depth=3,
        incidence_mode=incidence_mode,
        random_seed=random_seed,
    )
    compiler = ProofCarryingWitnessCompiler(
        width=lane.compiler_width,
        encoder_layers=lane.compiler_encoder_layers,
        decoder_layers=lane.compiler_decoder_layers,
        heads=lane.compiler_heads,
        feedforward=lane.compiler_feedforward,
        external_feature_width=external_feature_width,
        projector=projector,
    )
    query = NeuralOpaqueQueryParser(
        width=lane.query_width,
        layers=lane.query_layers,
        heads=lane.query_heads,
        feedforward=lane.query_feedforward,
        external_feature_width=external_feature_width,
    )
    receipt = EFCHankelCapacityReceipt(
        lane=(
            "maximum-hankel-shift"
            if decode_mode == "hankel-shift"
            else "maximum-hankel-direct"
        ),
        incidence_mode=incidence_mode,
        decode_mode=decode_mode,
        compiler_parameters=compiler.parameter_count(),
        projector_parameters=projector.parameter_count(),
        query_parameters=query.parameter_count(),
        added_parameters=compiler.parameter_count() + query.parameter_count(),
        complete_parameters=(
            PROTECTED_SHOHIN_PARAMETERS
            + compiler.parameter_count()
            + query.parameter_count()
        ),
        headroom=(
            GLOBAL_PARAMETER_LIMIT
            - PROTECTED_SHOHIN_PARAMETERS
            - compiler.parameter_count()
            - query.parameter_count()
        ),
    )
    expected = EFCHankelCapacityReceipt(
        lane=receipt.lane,
        incidence_mode=incidence_mode,
        decode_mode=decode_mode,
        compiler_parameters=HANKEL_SHIFT_MAXIMUM_EXPECTED.compiler_parameters,
        projector_parameters=HANKEL_SHIFT_MAXIMUM_EXPECTED.projector_parameters,
        query_parameters=HANKEL_SHIFT_MAXIMUM_EXPECTED.query_parameters,
        added_parameters=HANKEL_SHIFT_MAXIMUM_EXPECTED.added_parameters,
        complete_parameters=HANKEL_SHIFT_MAXIMUM_EXPECTED.complete_parameters,
        headroom=HANKEL_SHIFT_MAXIMUM_EXPECTED.headroom,
    )
    if receipt != expected or receipt.headroom <= 0:
        raise EFCCapacityError(
            "EFC maximum Hankel-shift capacity receipt differs"
        )
    return compiler, query, receipt


__all__ = [
    "CAPACITY_LANES",
    "EFCCapacityError",
    "EFCCapacityLane",
    "EFCCapacityReceipt",
    "EFCHankelCapacityReceipt",
    "HANKEL_SHIFT_MAXIMUM_EXPECTED",
    "MAXIMUM_PREREG_LANE",
    "MINIMAL_ATTRIBUTION_LANE",
    "WIDE_LANE",
    "build_hankel_shift_capacity_lane",
    "build_no_host_capacity_lane",
]
