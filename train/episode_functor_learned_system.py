"""End-to-end attached and sealed learned EFC system.

The source compiler, query parser, and categorical executor are composed here
without exposing source bytes or transition tables to the late-query parser.
The sealed object contains only hard machine fields and exact copied keys.
This module defines mechanics and parameter accounting; it does not train a
model or make a reasoning claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from episode_functor_constrained_transport import hard_assign_keys
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import (
    HardFunctorKeys,
    HardFunctorMachine,
    HardFunctorQuery,
    FunctorRollout,
    execute_hard,
)
from episode_functor_query_parser import (
    NeuralOpaqueQueryParser,
    QueryParserOutput,
    QueryPointerBatch,
    bind_query_roles_to_hard_keys,
)
from episode_functor_shohin_trunk import (
    FrozenShohinTrunk,
    ShohinTrunkBatch,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    WitnessCompilerBatch,
    WitnessCompilerOutput,
)


PROTECTED_SHOHIN_PARAMETERS = 125_081_664
PROTECTED_SHOHIN_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
GLOBAL_PARAMETER_LIMIT = 200_000_000


class LearnedEFCError(ValueError):
    """The composed learned EFC boundary or parameter receipt failed."""


@dataclass(frozen=True, slots=True)
class SealedFunctorBatch:
    """The only source-derived object that may survive source deletion."""

    machine: HardFunctorMachine
    keys: HardFunctorKeys

    def __post_init__(self) -> None:
        if self.machine.batch_size != self.keys.batch_size:
            raise LearnedEFCError("sealed machine and key batches differ")
        expected = (
            (self.machine.state_active, PRIMARY_STATES),
            (self.machine.action_active, PRIMARY_ACTIONS),
            (self.machine.observer_active, PRIMARY_OBSERVERS),
        )
        for active, count in expected:
            if (
                not bool(active[:, :count].eq(1).all())
                or bool(active[:, count:].ne(0).any())
            ):
                raise LearnedEFCError(
                    "sealed machine is outside the primary active geometry"
                )
        for row in range(self.machine.batch_size):
            self.keys.validate_masks(self.machine, row)

    def deployed_wire(self, row: int) -> bytes:
        return self.machine.deployed_wire(self.keys, row)


@dataclass(frozen=True, slots=True)
class SealedEFCOutput:
    sealed: SealedFunctorBatch
    query_parse: QueryParserOutput
    query: HardFunctorQuery
    rollout: FunctorRollout


@dataclass(frozen=True, slots=True)
class EFCParameterReceipt:
    protected_shohin_reference: int
    protected_checkpoint_sha256: str
    integrated_shohin: int
    integrated_checkpoint_sha256: str
    checkpoint_verified: bool
    source_compiler: int
    query_parser: int
    added_total: int
    instantiated_total: int
    hypothetical_complete_total: int
    global_limit: int
    hypothetical_headroom: int
    integration_status: str

    def __post_init__(self) -> None:
        values = tuple(
            getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in (
                "checkpoint_verified",
                "integrated_checkpoint_sha256",
                "integration_status",
                "protected_checkpoint_sha256",
            )
        )
        if any(not isinstance(value, int) for value in values):
            raise LearnedEFCError("parameter receipt is not integral")
        if (
            self.added_total != self.source_compiler + self.query_parser
            or self.instantiated_total
            != self.integrated_shohin + self.added_total
            or self.hypothetical_complete_total
            != self.protected_shohin_reference + self.added_total
            or self.hypothetical_headroom
            != self.global_limit - self.hypothetical_complete_total
            or self.hypothetical_headroom <= 0
        ):
            raise LearnedEFCError("parameter receipt arithmetic differs")
        hashes_are_canonical = all(
            len(value) == 64
            and set(value).issubset(set("0123456789abcdef"))
            for value in (
                self.protected_checkpoint_sha256,
                self.integrated_checkpoint_sha256,
            )
        )
        if not hashes_are_canonical:
            raise LearnedEFCError(
                "parameter receipt checkpoint digest differs"
            )
        if type(self.checkpoint_verified) is not bool:
            raise LearnedEFCError(
                "parameter receipt checkpoint verification differs"
            )
        expected_status = "connected" if (
            self.integrated_shohin == self.protected_shohin_reference
            and self.checkpoint_verified
            and self.integrated_checkpoint_sha256
            == self.protected_checkpoint_sha256
        ) else "not_connected"
        if self.integration_status != expected_status:
            raise LearnedEFCError("parameter integration status differs")


class LearnedEFCSystem(nn.Module):
    """Proof-carrying source compiler plus opaque late-query parser."""

    def __init__(
        self,
        *,
        source_compiler: ProofCarryingWitnessCompiler | None = None,
        query_parser: NeuralOpaqueQueryParser | None = None,
        frozen_trunk: FrozenShohinTrunk | None = None,
    ) -> None:
        super().__init__()
        self.source_compiler = (
            ProofCarryingWitnessCompiler()
            if source_compiler is None
            else source_compiler
        )
        self.query_parser = (
            NeuralOpaqueQueryParser()
            if query_parser is None
            else query_parser
        )
        self.frozen_trunk = frozen_trunk
        expected_external_width = (
            0 if frozen_trunk is None else frozen_trunk.feature_width
        )
        if (
            self.source_compiler.external_feature_width
            != expected_external_width
            or self.query_parser.external_feature_width
            != expected_external_width
        ):
            raise LearnedEFCError(
                "compiler/parser frozen-feature widths differ from trunk"
            )

    def parameter_receipt(
        self,
        *,
        protected_shohin: int = PROTECTED_SHOHIN_PARAMETERS,
        protected_checkpoint_sha256: str = PROTECTED_SHOHIN_SHA256,
        global_limit: int = GLOBAL_PARAMETER_LIMIT,
    ) -> EFCParameterReceipt:
        source_parameters = self.source_compiler.parameter_count()
        query_parameters = self.query_parser.parameter_count()
        trunk_receipt = (
            None
            if self.frozen_trunk is None
            else self.frozen_trunk.parameter_receipt()
        )
        integrated_shohin = (
            0
            if trunk_receipt is None
            else trunk_receipt.parent_unique_parameters
        )
        if integrated_shohin not in (0, protected_shohin):
            raise LearnedEFCError(
                "integrated Shohin parameter count differs from reference"
            )
        return EFCParameterReceipt(
            protected_shohin_reference=protected_shohin,
            protected_checkpoint_sha256=protected_checkpoint_sha256,
            integrated_shohin=integrated_shohin,
            integrated_checkpoint_sha256=(
                "0" * 64
                if trunk_receipt is None
                else trunk_receipt.checkpoint_sha256
            ),
            checkpoint_verified=(
                False
                if trunk_receipt is None
                else trunk_receipt.checkpoint_verified
            ),
            source_compiler=source_parameters,
            query_parser=query_parameters,
            added_total=source_parameters + query_parameters,
            instantiated_total=(
                integrated_shohin
                + source_parameters
                + query_parameters
            ),
            hypothetical_complete_total=(
                protected_shohin + source_parameters + query_parameters
            ),
            global_limit=global_limit,
            hypothetical_headroom=(
                global_limit
                - protected_shohin
                - source_parameters
                - query_parameters
            ),
            integration_status=(
                "connected"
                if (
                    trunk_receipt is not None
                    and integrated_shohin == protected_shohin
                    and trunk_receipt.checkpoint_verified
                    and trunk_receipt.checkpoint_sha256
                    == protected_checkpoint_sha256
                )
                else "not_connected"
            ),
        )

    def _frozen_features(
        self,
        trunk_batch: ShohinTrunkBatch | None,
        *,
        byte_valid,
        label: str,
    ):
        if self.frozen_trunk is None:
            if trunk_batch is not None:
                raise LearnedEFCError(
                    f"standalone system received {label} trunk input"
                )
            return None
        if trunk_batch is None:
            raise LearnedEFCError(
                f"connected system is missing {label} trunk input"
            )
        features = self.frozen_trunk.encode_batch(trunk_batch)
        flattened = self.frozen_trunk.flatten_byte_features(features)
        if (
            flattened.shape[:2] != byte_valid.shape
            or features.byte_valid.device != byte_valid.device
            or not bool(torch.equal(features.byte_valid, byte_valid))
        ):
            raise LearnedEFCError(
                f"{label} trunk byte alignment differs from parser input"
            )
        return flattened

    def compile_source(
        self,
        source: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        trunk_batch: ShohinTrunkBatch | None = None,
    ) -> WitnessCompilerOutput:
        frozen_features = self._frozen_features(
            trunk_batch,
            byte_valid=source.pointer.byte_valid,
            label="source",
        )
        return self.source_compiler(
            source,
            straight_through=straight_through,
            frozen_byte_features=frozen_features,
        )

    def parse_query(
        self,
        query: QueryPointerBatch,
        sealed: SealedFunctorBatch,
        *,
        trunk_batch: ShohinTrunkBatch | None = None,
    ) -> QueryParserOutput:
        frozen_features = self._frozen_features(
            trunk_batch,
            byte_valid=query.byte_valid,
            label="query",
        )
        roles = self.query_parser.parse_roles(
            query,
            frozen_byte_features=frozen_features,
        )
        return bind_query_roles_to_hard_keys(
            role_occurrence_logits=roles.role_occurrence_logits,
            stop_position_logits=roles.stop_position_logits,
            query_occurrence_key_bytes=query.occurrence_key_bytes,
            query_occurrence_valid=query.occurrence_valid,
            sealed_keys=sealed.keys,
        )

    def forward(
        self,
        sealed: SealedFunctorBatch,
        query: QueryPointerBatch,
        *,
        trunk_batch: ShohinTrunkBatch | None = None,
    ) -> SealedEFCOutput:
        """Parse and execute a query that arrives after source sealing."""

        query_parse = self.parse_query(
            query,
            sealed,
            trunk_batch=trunk_batch,
        )
        hard_query = query_parse.query.harden(sealed.machine)
        rollout = execute_hard(sealed.machine, hard_query)
        return SealedEFCOutput(
            sealed=sealed,
            query_parse=query_parse,
            query=hard_query,
            rollout=rollout,
        )

    def seal(
        self,
        compilation: WitnessCompilerOutput,
    ) -> SealedFunctorBatch:
        machine = self.source_compiler.projector.hard_project(
            compilation.relation_evidence.transition_logits,
            compilation.relation_evidence.observer_logits,
        )
        keys = hard_assign_keys(
            slot_assignment_logits=compilation.key_assignment_logits,
            source_unique_key_bytes=compilation.unique_key_bytes,
            source_unique_key_valid=compilation.unique_key_valid,
        ).keys
        return SealedFunctorBatch(machine=machine, keys=keys)

    @staticmethod
    def harden_query(
        query_parse: QueryParserOutput,
        sealed: SealedFunctorBatch,
    ) -> HardFunctorQuery:
        return query_parse.query.harden(sealed.machine)

    @staticmethod
    def execute_sealed(
        sealed: SealedFunctorBatch,
        query: HardFunctorQuery,
    ) -> FunctorRollout:
        return execute_hard(sealed.machine, query)


__all__ = [
    "EFCParameterReceipt",
    "GLOBAL_PARAMETER_LIMIT",
    "LearnedEFCError",
    "LearnedEFCSystem",
    "PROTECTED_SHOHIN_PARAMETERS",
    "PROTECTED_SHOHIN_SHA256",
    "SealedEFCOutput",
    "SealedFunctorBatch",
]
