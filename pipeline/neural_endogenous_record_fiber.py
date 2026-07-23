"""Guaranteed-equivalence record-fiber decoder for neural ECCR.

The wrapped ECCR encoder emits anonymous, permutation-equivariant record
features. A shared head maps each record/record-anchor pair to five replicated
binary signature votes. Hard equivalence is equality of complete active
signature rows, so every decoded relation is an equivalence relation without
clustering, closure, search, retry, repair, or class labels.

Structural validity is guaranteed. Whether the relation preserves observations
and descends through every physical generator remains an explicit, falsifiable
validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import torch
import torch.nn.functional as functional
from pipeline.neural_endogenous_congruence import (
    MASKED_LOGIT,
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
    EquivalenceResiduals,
    NeuralEndogenousCongruence,
    NeuralEndogenousCongruenceConfig,
    NeuralEndogenousCongruenceOutput,
    _relation_residuals,
    _require_tensor_batch,
)
from pipeline.tensorize_endogenous_congruence import (
    N,
    EndogenousCongruenceTensors,
)
from torch import Tensor, nn


SIGNATURE_REPLICAS = 5
SIGNATURE_MAJORITY = 3


class NeuralEndogenousRecordFiberError(ValueError):
    """A record-fiber input or physical-law claim violates the contract."""


@dataclass(frozen=True)
class NeuralEndogenousRecordFiberConfig:
    """Architecture and differentiable-relaxation settings."""

    encoder_config: NeuralEndogenousCongruenceConfig = field(
        default_factory=NeuralEndogenousCongruenceConfig
    )
    vote_hidden_dim: int = 128
    soft_temperature: float = 1.0
    soft_majority_scale: float = 2.0
    parameter_cap: int = 10_000_000

    def __post_init__(self) -> None:
        if self.vote_hidden_dim <= 0:
            raise NeuralEndogenousRecordFiberError("vote_hidden_dim must be positive")
        if not isfinite(self.soft_temperature) or self.soft_temperature <= 0:
            raise NeuralEndogenousRecordFiberError(
                "soft_temperature must be finite and positive"
            )
        if not isfinite(self.soft_majority_scale) or self.soft_majority_scale <= 0:
            raise NeuralEndogenousRecordFiberError(
                "soft_majority_scale must be finite and positive"
            )
        if self.parameter_cap <= 0:
            raise NeuralEndogenousRecordFiberError("parameter_cap must be positive")


@dataclass(frozen=True)
class NeuralEndogenousRecordFiberParameterCount:
    """Auditable standalone and protected-system parameter ledger."""

    encoder: int
    vote_head: int
    total: int
    trainable: int
    cap: int
    protected_base: int
    complete_system: int
    system_cap: int
    headroom: int

    @property
    def under_cap(self) -> bool:
        return self.total <= self.cap

    @property
    def under_system_cap(self) -> bool:
        return self.complete_system < self.system_cap


@dataclass(frozen=True)
class RecordFiberHardDecoding:
    """Hard anonymous signatures and their guaranteed equivalence fibers."""

    signatures: Tensor
    equivalence: Tensor
    projector: Tensor
    record_mask: Tensor


@dataclass(frozen=True)
class RecordFiberPhysicalValidation:
    """Per-episode physical-law and exact-projector measurements."""

    residuals: EquivalenceResiduals
    observation_valid: Tensor
    descent_valid: Tensor
    projector_symmetric: Tensor
    projector_idempotent: Tensor
    projector_row_stochastic: Tensor

    @property
    def valid(self) -> Tensor:
        return (
            self.observation_valid
            & self.descent_valid
            & self.projector_symmetric
            & self.projector_idempotent
            & self.projector_row_stochastic
        )


@dataclass(frozen=True)
class NeuralEndogenousRecordFiberOutput:
    """Differentiable fiber proposal plus guaranteed hard decoding."""

    vote_logits: Tensor
    vote_probabilities: Tensor
    soft_signatures: Tensor
    soft_fiber_relation: Tensor
    soft_residuals: EquivalenceResiduals
    hard_residuals: EquivalenceResiduals
    hard: RecordFiberHardDecoding
    encoder_output: NeuralEndogenousCongruenceOutput


@dataclass(frozen=True)
class RecordFiberLoss:
    """Balanced differentiable supervision for anonymous relation fibers."""

    total: Tensor
    code: Tensor
    fiber: Tensor
    distance: Tensor
    margin: Tensor


class _VoteHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, SIGNATURE_REPLICAS),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def _require_record_mask(record_mask: Tensor) -> int:
    if type(record_mask) is not Tensor:
        raise NeuralEndogenousRecordFiberError("record_mask is not a Tensor")
    if record_mask.dtype != torch.bool:
        raise NeuralEndogenousRecordFiberError("record_mask must be boolean")
    if record_mask.ndim != 2 or record_mask.shape[1] != N:
        raise NeuralEndogenousRecordFiberError(
            f"record_mask must have shape [batch, {N}]"
        )
    if record_mask.shape[0] == 0:
        raise NeuralEndogenousRecordFiberError("record_mask batch is empty")
    if torch.any(record_mask.sum(dim=1) < 1):
        raise NeuralEndogenousRecordFiberError(
            "each record-fiber episode needs an active record"
        )
    return record_mask.shape[0]


def _pair_mask(record_mask: Tensor) -> Tensor:
    return record_mask[:, :, None] & record_mask[:, None, :]


def _active_identity(record_mask: Tensor) -> Tensor:
    identity = torch.eye(
        N,
        dtype=torch.bool,
        device=record_mask.device,
    )[None]
    return identity & _pair_mask(record_mask)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(value.dtype).unsqueeze(-1)
    return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _require_vote_logits(
    vote_logits: Tensor,
    record_mask: Tensor,
) -> int:
    batch_size = _require_record_mask(record_mask)
    if type(vote_logits) is not Tensor:
        raise NeuralEndogenousRecordFiberError("vote_logits is not a Tensor")
    if tuple(vote_logits.shape) != (
        batch_size,
        N,
        N,
        SIGNATURE_REPLICAS,
    ):
        raise NeuralEndogenousRecordFiberError(
            "vote_logits has invalid record-fiber geometry"
        )
    if not torch.is_floating_point(vote_logits):
        raise NeuralEndogenousRecordFiberError("vote_logits must be floating")
    if vote_logits.device != record_mask.device:
        raise NeuralEndogenousRecordFiberError(
            "vote_logits and record_mask are on different devices"
        )
    if not torch.all(torch.isfinite(vote_logits)):
        raise NeuralEndogenousRecordFiberError("vote_logits contains non-finite values")
    padding = ~_pair_mask(record_mask).unsqueeze(-1)
    if torch.any(
        vote_logits.masked_select(padding)
        != torch.as_tensor(
            MASKED_LOGIT,
            dtype=vote_logits.dtype,
            device=vote_logits.device,
        )
    ):
        raise NeuralEndogenousRecordFiberError(
            "vote_logits must be exactly masked in record padding"
        )
    return batch_size


def _relation_from_signatures(
    signatures: Tensor,
    record_mask: Tensor,
) -> Tensor:
    anchor_active = record_mask[:, None, None, :]
    matching_bits = (
        signatures[:, :, None, :] == signatures[:, None, :, :]
    ) | ~anchor_active
    return matching_bits.all(dim=-1) & _pair_mask(record_mask)


def _exact_projector(
    equivalence: Tensor,
    record_mask: Tensor,
) -> Tensor:
    relation = equivalence.to(torch.float32)
    relation = relation * _pair_mask(record_mask).to(torch.float32)
    class_size = relation.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return relation / class_size


def decode_record_fiber_vote_logits(
    vote_logits: Tensor,
    record_mask: Tensor,
    *,
    threshold: float = 0.0,
) -> RecordFiberHardDecoding:
    """Threshold five votes once and decode equality of complete signatures."""

    _require_vote_logits(vote_logits, record_mask)
    if not isfinite(threshold):
        raise NeuralEndogenousRecordFiberError("threshold must be finite")

    pair_mask = _pair_mask(record_mask)
    votes = vote_logits >= threshold
    signatures = votes.sum(dim=-1) >= SIGNATURE_MAJORITY
    signatures = (signatures & pair_mask) | _active_identity(record_mask)
    equivalence = _relation_from_signatures(signatures, record_mask)
    projector = _exact_projector(equivalence, record_mask)
    return RecordFiberHardDecoding(
        signatures=signatures,
        equivalence=equivalence,
        projector=projector,
        record_mask=record_mask,
    )


def _soft_record_fibers(
    vote_logits: Tensor,
    record_mask: Tensor,
    *,
    temperature: float,
    majority_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    pair_mask = _pair_mask(record_mask)
    vote_probabilities = torch.sigmoid(vote_logits / temperature)
    vote_probabilities = vote_probabilities * pair_mask.unsqueeze(-1).to(
        vote_logits.dtype
    )

    vote_evidence = torch.tanh(vote_logits / temperature).sum(dim=-1)
    soft_signatures = torch.sigmoid(majority_scale * vote_evidence)
    soft_signatures = soft_signatures * pair_mask.to(vote_logits.dtype)
    soft_signatures = torch.where(
        _active_identity(record_mask),
        torch.ones((), dtype=vote_logits.dtype, device=vote_logits.device),
        soft_signatures,
    )

    left = soft_signatures[:, :, None, :]
    right = soft_signatures[:, None, :, :]
    bit_match = left * right + (1.0 - left) * (1.0 - right)
    anchor_active = record_mask[:, None, None, :]
    bit_match = torch.where(
        anchor_active,
        bit_match,
        torch.ones((), dtype=vote_logits.dtype, device=vote_logits.device),
    )
    soft_fiber_relation = bit_match.clamp_min(torch.finfo(vote_logits.dtype).tiny).prod(
        dim=-1
    )
    soft_fiber_relation = soft_fiber_relation * pair_mask.to(vote_logits.dtype)
    soft_fiber_relation = torch.where(
        _active_identity(record_mask),
        torch.ones((), dtype=vote_logits.dtype, device=vote_logits.device),
        soft_fiber_relation,
    )
    return vote_probabilities, soft_signatures, soft_fiber_relation


def _validate_hard_decoding_contract(
    hard: RecordFiberHardDecoding,
    record_mask: Tensor,
) -> None:
    if type(hard) is not RecordFiberHardDecoding:
        raise NeuralEndogenousRecordFiberError(
            "hard decoding has an invalid container type"
        )
    batch_size = _require_record_mask(record_mask)
    if hard.record_mask.device != record_mask.device or not torch.equal(
        hard.record_mask,
        record_mask,
    ):
        raise NeuralEndogenousRecordFiberError("hard decoding record mask changed")
    for name, value, shape, dtype in (
        ("signatures", hard.signatures, (batch_size, N, N), torch.bool),
        ("equivalence", hard.equivalence, (batch_size, N, N), torch.bool),
        ("projector", hard.projector, (batch_size, N, N), torch.float32),
    ):
        if type(value) is not Tensor:
            raise NeuralEndogenousRecordFiberError(f"{name} is not a Tensor")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise NeuralEndogenousRecordFiberError(f"{name} has invalid shape or dtype")
        if value.device != record_mask.device:
            raise NeuralEndogenousRecordFiberError(f"{name} is on the wrong device")
    pair_mask = _pair_mask(record_mask)
    if torch.any(hard.signatures & ~pair_mask):
        raise NeuralEndogenousRecordFiberError("signatures enter record padding")
    if not torch.equal(
        hard.signatures & _active_identity(record_mask), _active_identity(record_mask)
    ):
        raise NeuralEndogenousRecordFiberError(
            "active diagonal signature bits are not forced"
        )
    expected_equivalence = _relation_from_signatures(
        hard.signatures,
        record_mask,
    )
    if not torch.equal(hard.equivalence, expected_equivalence):
        raise NeuralEndogenousRecordFiberError(
            "equivalence is not exact signature-row equality"
        )
    expected_projector = _exact_projector(expected_equivalence, record_mask)
    if not torch.equal(hard.projector, expected_projector):
        raise NeuralEndogenousRecordFiberError(
            "projector is not the exact normalized fiber relation"
        )


def measure_record_fiber_physical_laws(
    tensors: EndogenousCongruenceTensors,
    hard: RecordFiberHardDecoding,
) -> RecordFiberPhysicalValidation:
    """Measure causal descent, observation factorization, and projector laws."""

    _require_tensor_batch(tensors)
    _validate_hard_decoding_contract(hard, tensors.record_mask)
    residuals = _relation_residuals(
        tensors,
        hard.equivalence.to(torch.float32),
    )
    projector = hard.projector
    projected_twice = torch.einsum("bij,bjk->bik", projector, projector)
    transpose = projector.transpose(1, 2)
    expected_row_sum = tensors.record_mask.to(torch.float32)
    row_sum = projector.sum(dim=-1)
    return RecordFiberPhysicalValidation(
        residuals=residuals,
        observation_valid=residuals.observation == 0,
        descent_valid=residuals.descent == 0,
        projector_symmetric=torch.isclose(
            projector,
            transpose,
            rtol=0.0,
            atol=1e-6,
        ).all(dim=(1, 2)),
        projector_idempotent=torch.isclose(
            projected_twice,
            projector,
            rtol=0.0,
            atol=1e-6,
        ).all(dim=(1, 2)),
        projector_row_stochastic=torch.isclose(
            row_sum,
            expected_row_sum,
            rtol=0.0,
            atol=1e-6,
        ).all(dim=1),
    )


def validate_record_fiber_physical_laws(
    tensors: EndogenousCongruenceTensors,
    hard: RecordFiberHardDecoding,
) -> RecordFiberPhysicalValidation:
    """Fail closed when a valid equivalence is not a physical congruence."""

    validation = measure_record_fiber_physical_laws(tensors, hard)
    if not torch.all(validation.observation_valid):
        raise NeuralEndogenousRecordFiberError(
            "record fibers do not preserve observations"
        )
    if not torch.all(validation.descent_valid):
        raise NeuralEndogenousRecordFiberError(
            "record fibers do not descend through generators"
        )
    if not torch.all(
        validation.projector_symmetric
        & validation.projector_idempotent
        & validation.projector_row_stochastic
    ):
        raise NeuralEndogenousRecordFiberError(
            "record-fiber projector violates exact quotient laws"
        )
    return validation


class NeuralEndogenousRecordFiber(nn.Module):
    """Recoding-invariant ECCR encoder with a guaranteed-valid fiber decoder."""

    def __init__(
        self,
        config: NeuralEndogenousRecordFiberConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NeuralEndogenousRecordFiberConfig()
        self.encoder = NeuralEndogenousCongruence(self.config.encoder_config)
        hidden = self.config.encoder_config.hidden_dim
        self.vote_head = _VoteHead(4 * hidden, self.config.vote_hidden_dim)

        count = self.parameter_count()
        if not count.under_cap:
            raise NeuralEndogenousRecordFiberError(
                f"record-fiber model has {count.total} parameters, cap is {count.cap}"
            )
        if not count.under_system_cap:
            raise NeuralEndogenousRecordFiberError(
                f"complete system has {count.complete_system} parameters"
            )

    def parameter_count(self) -> NeuralEndogenousRecordFiberParameterCount:
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        vote_head = sum(parameter.numel() for parameter in self.vote_head.parameters())
        total = encoder + vote_head
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        complete_system = PROTECTED_BASE_PARAMETERS + total
        return NeuralEndogenousRecordFiberParameterCount(
            encoder=encoder,
            vote_head=vote_head,
            total=total,
            trainable=trainable,
            cap=self.config.parameter_cap,
            protected_base=PROTECTED_BASE_PARAMETERS,
            complete_system=complete_system,
            system_cap=SYSTEM_PARAMETER_CAP,
            headroom=SYSTEM_PARAMETER_CAP - complete_system,
        )

    def forward(
        self,
        tensors: EndogenousCongruenceTensors,
    ) -> NeuralEndogenousRecordFiberOutput:
        encoder_output = self.encoder(tensors)
        record = encoder_output.record_features
        record_mask = encoder_output.record_mask
        dtype = record.dtype
        device = record.device

        left = record[:, :, None, :].expand(-1, -1, N, -1)
        anchor = record[:, None, :, :].expand(-1, N, -1, -1)
        global_context = _masked_mean(record, record_mask)
        pair_features = torch.cat(
            (
                left + anchor,
                torch.abs(left - anchor),
                left * anchor,
                global_context[:, None, None, :].expand(-1, N, N, -1),
            ),
            dim=-1,
        )
        vote_logits = self.vote_head(pair_features)
        vote_logits = 0.5 * (vote_logits + vote_logits.transpose(1, 2))
        pair_mask = _pair_mask(record_mask)
        vote_logits = torch.where(
            pair_mask.unsqueeze(-1),
            vote_logits,
            torch.full(
                (),
                MASKED_LOGIT,
                dtype=dtype,
                device=device,
            ),
        )
        hard = decode_record_fiber_vote_logits(vote_logits, record_mask)
        vote_probabilities, soft_signatures, soft_fiber_relation = _soft_record_fibers(
            vote_logits,
            record_mask,
            temperature=self.config.soft_temperature,
            majority_scale=self.config.soft_majority_scale,
        )
        soft_residuals = _relation_residuals(tensors, soft_fiber_relation)
        hard_residuals = _relation_residuals(
            tensors,
            hard.equivalence.to(torch.float32),
        )
        return NeuralEndogenousRecordFiberOutput(
            vote_logits=vote_logits,
            vote_probabilities=vote_probabilities,
            soft_signatures=soft_signatures,
            soft_fiber_relation=soft_fiber_relation,
            soft_residuals=soft_residuals,
            hard_residuals=hard_residuals,
            hard=hard,
            encoder_output=encoder_output,
        )


def _masked_mean_loss(value: Tensor, mask: Tensor) -> Tensor:
    if torch.any(mask):
        return value.masked_select(mask).mean()
    return value.sum() * 0.0


def _require_target_equivalence(
    target: Tensor,
    record_mask: Tensor,
) -> None:
    batch_size = _require_record_mask(record_mask)
    if type(target) is not Tensor:
        raise NeuralEndogenousRecordFiberError("target_equivalence is not a Tensor")
    if (
        tuple(target.shape) != (batch_size, N, N)
        or target.dtype != torch.bool
        or target.device != record_mask.device
    ):
        raise NeuralEndogenousRecordFiberError(
            "target_equivalence has invalid shape, dtype, or device"
        )
    pair_mask = _pair_mask(record_mask)
    if torch.any(target & ~pair_mask):
        raise NeuralEndogenousRecordFiberError(
            "target_equivalence enters record padding"
        )
    if not torch.equal(
        target & _active_identity(record_mask), _active_identity(record_mask)
    ):
        raise NeuralEndogenousRecordFiberError("target_equivalence is not reflexive")
    if not torch.equal(target, target.transpose(1, 2)):
        raise NeuralEndogenousRecordFiberError("target_equivalence is not symmetric")
    composed = torch.einsum(
        "bij,bjk->bik",
        target.to(torch.float32),
        target.to(torch.float32),
    )
    if torch.any((composed > 0) & ~target & pair_mask):
        raise NeuralEndogenousRecordFiberError("target_equivalence is not transitive")


def record_fiber_loss(
    output: NeuralEndogenousRecordFiberOutput,
    target_equivalence: Tensor,
    *,
    margin: float = 1.0,
) -> RecordFiberLoss:
    """Train relation signatures without canonical class assignments."""

    if type(output) is not NeuralEndogenousRecordFiberOutput:
        raise NeuralEndogenousRecordFiberError("output has an invalid type")
    record_mask = output.hard.record_mask
    _require_target_equivalence(target_equivalence, record_mask)
    if not isfinite(margin) or margin <= 0:
        raise NeuralEndogenousRecordFiberError("margin must be finite and positive")

    pair_mask = _pair_mask(record_mask)
    target_float = target_equivalence.to(output.vote_logits.dtype)
    expanded_target = target_float.unsqueeze(-1).expand_as(output.vote_logits)
    expanded_mask = pair_mask.unsqueeze(-1).expand_as(output.vote_logits)
    code = functional.binary_cross_entropy_with_logits(
        output.vote_logits.masked_select(expanded_mask),
        expanded_target.masked_select(expanded_mask),
    )

    probabilities = output.soft_fiber_relation.clamp(
        min=torch.finfo(output.soft_fiber_relation.dtype).eps,
        max=1.0 - torch.finfo(output.soft_fiber_relation.dtype).eps,
    )
    positive = pair_mask & target_equivalence
    negative = pair_mask & ~target_equivalence
    positive_fiber = _masked_mean_loss(-torch.log(probabilities), positive)
    negative_fiber = _masked_mean_loss(-torch.log1p(-probabilities), negative)
    fiber = positive_fiber + negative_fiber

    soft = output.soft_signatures
    left = soft[:, :, None, :]
    right = soft[:, None, :, :]
    distance_value = (
        (left + right - 2.0 * left * right)
        * record_mask[:, None, None, :].to(soft.dtype)
    ).sum(dim=-1)
    positive_distance = _masked_mean_loss(distance_value, positive)
    negative_distance = _masked_mean_loss(
        functional.relu(2.0 - distance_value).square(),
        negative,
    )
    distance = positive_distance + negative_distance

    signed_target = 2.0 * expanded_target - 1.0
    signed_margin = signed_target * output.vote_logits
    margin_loss = functional.relu(margin - signed_margin).square()
    margin_term = margin_loss.masked_select(expanded_mask).mean()
    total = code + 0.5 * fiber + 0.25 * distance + 0.05 * margin_term
    return RecordFiberLoss(
        total=total,
        code=code,
        fiber=fiber,
        distance=distance,
        margin=margin_term,
    )


__all__ = [
    "SIGNATURE_MAJORITY",
    "SIGNATURE_REPLICAS",
    "NeuralEndogenousRecordFiber",
    "NeuralEndogenousRecordFiberConfig",
    "NeuralEndogenousRecordFiberError",
    "NeuralEndogenousRecordFiberOutput",
    "NeuralEndogenousRecordFiberParameterCount",
    "RecordFiberHardDecoding",
    "RecordFiberLoss",
    "RecordFiberPhysicalValidation",
    "decode_record_fiber_vote_logits",
    "measure_record_fiber_physical_laws",
    "record_fiber_loss",
    "validate_record_fiber_physical_laws",
]
