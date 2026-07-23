"""Bounded neural substrate for endogenous causal congruence.

The model consumes only the source-deleted physical tensor boundary. It emits
an anonymous same-class relation and equivariant entity features. The hard
decoder performs one threshold operation and rejects invalid relations; it
does not search for, refine, or repair a relation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import torch
from pipeline.tensorize_endogenous_congruence import (
    G,
    N,
    Q,
    EndogenousCongruenceTensors,
)
from torch import Tensor, nn


MASKED_LOGIT = -10_000.0
PROTECTED_BASE_PARAMETERS = 125_081_664
SYSTEM_PARAMETER_CAP = 200_000_000


class NeuralEndogenousCongruenceError(ValueError):
    """The neural input or proposed relation violates the frozen contract."""


@dataclass(frozen=True)
class NeuralEndogenousCongruenceConfig:
    """Architecture settings independent of episode-local entity counts."""

    hidden_dim: int = 192
    rounds: int = 4
    parameter_cap: int = 8_000_000

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise NeuralEndogenousCongruenceError("hidden_dim must be positive")
        if self.rounds <= 0:
            raise NeuralEndogenousCongruenceError("rounds must be positive")
        if self.parameter_cap <= 0:
            raise NeuralEndogenousCongruenceError("parameter_cap must be positive")


@dataclass(frozen=True)
class NeuralEndogenousCongruenceParameterCount:
    """Auditable standalone and complete-system parameter budget."""

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
class EquivalenceResiduals:
    """Differentiable physical-law residuals for a proposed relation."""

    descent: Tensor
    observation: Tensor


@dataclass(frozen=True)
class NeuralEndogenousCongruenceOutput:
    """Anonymous relation proposal and equivariant latent entity states."""

    same_class_logits: Tensor
    equivalence_mask: Tensor
    soft_equivalence: Tensor
    soft_projector: Tensor
    residuals: EquivalenceResiduals
    record_features: Tensor
    record_mask: Tensor
    generator_features: Tensor
    generator_mask: Tensor
    query_features: Tensor
    query_mask: Tensor


@dataclass(frozen=True)
class DecodedEndogenousCongruence:
    """A validated hard relation with no canonical class numbering."""

    equivalence: Tensor
    projector: Tensor
    residuals: EquivalenceResiduals
    record_mask: Tensor
    generator_mask: Tensor
    query_mask: Tensor


class _SharedMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def _masked_mean(value: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(value.dtype).unsqueeze(-1)
    total = (value * weights).sum(dim=dimension)
    count = weights.sum(dim=dimension).clamp_min(1.0)
    return total / count


def _normalized_observation(value: Tensor) -> Tensor:
    floating = value.to(torch.float32)
    signed_log = torch.sign(floating) * torch.log1p(torch.abs(floating))
    return torch.stack(
        (
            torch.tanh(signed_log / 16.0),
            (value == 0).to(torch.float32),
            torch.tanh(torch.abs(signed_log) / 16.0),
        ),
        dim=-1,
    )


def _require_tensor_batch(value: EndogenousCongruenceTensors) -> int:
    if type(value) is not EndogenousCongruenceTensors:
        raise NeuralEndogenousCongruenceError(
            "forward accepts only exact EndogenousCongruenceTensors"
        )
    expected = {
        "record_mask": ((N,), torch.bool),
        "generator_mask": ((G,), torch.bool),
        "query_mask": ((Q,), torch.bool),
        "record_equal": ((N, N), torch.bool),
        "generator_equal": ((G, G), torch.bool),
        "query_equal": ((Q, Q), torch.bool),
        "transition_mask": ((N, G), torch.bool),
        "transition_target": ((N, G, N), torch.bool),
        "observation_mask": ((N, Q), torch.bool),
        "observation_value": ((N, Q), torch.int64),
        "observation_equal": ((N, Q, N, Q), torch.bool),
    }
    batch_size: int | None = None
    device: torch.device | None = None
    for field in fields(value):
        tensor = getattr(value, field.name)
        if type(tensor) is not Tensor:
            raise NeuralEndogenousCongruenceError(f"{field.name} is not a Tensor")
        tail, dtype = expected[field.name]
        if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
            raise NeuralEndogenousCongruenceError(
                f"{field.name} has invalid shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise NeuralEndogenousCongruenceError(
                f"{field.name} has invalid dtype {tensor.dtype}"
            )
        if batch_size is None:
            batch_size = tensor.shape[0]
            device = tensor.device
        if tensor.shape[0] != batch_size or tensor.device != device:
            raise NeuralEndogenousCongruenceError(
                "all boundary tensors must share batch and device"
            )
    if batch_size is None or batch_size == 0:
        raise NeuralEndogenousCongruenceError("tensor batch is empty")
    if torch.any(value.record_mask.sum(dim=1) < 2):
        raise NeuralEndogenousCongruenceError(
            "each episode requires at least two active records"
        )
    if torch.any(value.generator_mask.sum(dim=1) < 1):
        raise NeuralEndogenousCongruenceError(
            "each episode requires at least one active generator"
        )
    if torch.any(value.query_mask.sum(dim=1) < 1):
        raise NeuralEndogenousCongruenceError(
            "each episode requires at least one active query"
        )

    record_pair = value.record_mask[:, :, None] & value.record_mask[:, None, :]
    generator_pair = value.generator_mask[:, :, None] & value.generator_mask[:, None, :]
    query_pair = value.query_mask[:, :, None] & value.query_mask[:, None, :]
    record_identity = torch.eye(N, dtype=torch.bool, device=device)[None]
    generator_identity = torch.eye(G, dtype=torch.bool, device=device)[None]
    query_identity = torch.eye(Q, dtype=torch.bool, device=device)[None]
    if not torch.equal(value.record_equal, record_pair & record_identity):
        raise NeuralEndogenousCongruenceError("record equality channel is invalid")
    if not torch.equal(value.generator_equal, generator_pair & generator_identity):
        raise NeuralEndogenousCongruenceError("generator equality channel is invalid")
    if not torch.equal(value.query_equal, query_pair & query_identity):
        raise NeuralEndogenousCongruenceError("query equality channel is invalid")
    expected_transition_mask = (
        value.record_mask[:, :, None] & value.generator_mask[:, None, :]
    )
    expected_observation_mask = (
        value.record_mask[:, :, None] & value.query_mask[:, None, :]
    )
    if not torch.equal(value.transition_mask, expected_transition_mask):
        raise NeuralEndogenousCongruenceError("transition mask is invalid")
    if not torch.equal(value.observation_mask, expected_observation_mask):
        raise NeuralEndogenousCongruenceError("observation mask is invalid")
    target_count = value.transition_target.sum(dim=-1)
    if not torch.equal(target_count, value.transition_mask.to(target_count.dtype)):
        raise NeuralEndogenousCongruenceError(
            "active transitions must have one target and padding must have none"
        )
    active_targets = value.transition_target & value.record_mask[:, None, None, :]
    if not torch.equal(active_targets, value.transition_target):
        raise NeuralEndogenousCongruenceError("transition targets enter padding")
    if torch.any(value.observation_value.masked_select(~value.observation_mask) != 0):
        raise NeuralEndogenousCongruenceError("observation padding is nonzero")
    active_values = value.observation_value
    expected_observation_equal = (
        active_values[:, :, :, None, None] == active_values[:, None, None, :, :]
    )
    observation_pair_mask = (
        value.observation_mask[:, :, :, None, None]
        & value.observation_mask[:, None, None, :, :]
    )
    expected_observation_equal &= observation_pair_mask
    if not torch.equal(value.observation_equal, expected_observation_equal):
        raise NeuralEndogenousCongruenceError("observation equality channel is invalid")
    return batch_size


def _relation_projector(relation: Tensor, record_mask: Tensor) -> Tensor:
    pair_mask = record_mask[:, :, None] & record_mask[:, None, :]
    masked = relation * pair_mask.to(relation.dtype)
    degree = masked.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return masked / degree


def _relation_residuals(
    tensors: EndogenousCongruenceTensors,
    relation: Tensor,
) -> EquivalenceResiduals:
    relation_float = relation.to(torch.float32)
    transition = tensors.transition_target.to(torch.float32)
    transported = torch.einsum(
        "biga,bac,bjgc->bgij",
        transition,
        relation_float,
        transition,
    )
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    descent_mask = tensors.generator_mask[:, :, None, None] & pair_mask[:, None, :, :]
    descent_terms = (
        relation_float[:, None, :, :]
        * (1.0 - transported)
        * descent_mask.to(torch.float32)
    )
    descent_denominator = descent_mask.sum(dim=(1, 2, 3)).clamp_min(1)
    descent = descent_terms.sum(dim=(1, 2, 3)) / descent_denominator

    same_observation = torch.einsum(
        "biqjr,bqr->bqij",
        tensors.observation_equal.to(torch.float32),
        tensors.query_equal.to(torch.float32),
    )
    observation_mask = tensors.query_mask[:, :, None, None] & pair_mask[:, None, :, :]
    observation_terms = (
        relation_float[:, None, :, :]
        * (1.0 - same_observation)
        * observation_mask.to(torch.float32)
    )
    observation_denominator = observation_mask.sum(dim=(1, 2, 3)).clamp_min(1)
    observation = observation_terms.sum(dim=(1, 2, 3)) / observation_denominator
    return EquivalenceResiduals(descent=descent, observation=observation)


class NeuralEndogenousCongruence(nn.Module):
    """Permutation-equivariant recurrent inducer over anonymous episodes."""

    def __init__(
        self,
        config: NeuralEndogenousCongruenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NeuralEndogenousCongruenceConfig()
        hidden = self.config.hidden_dim

        self.record_seed = nn.Parameter(torch.empty(hidden))
        self.generator_seed = nn.Parameter(torch.empty(hidden))
        self.query_seed = nn.Parameter(torch.empty(hidden))
        self.observation_encoder = _SharedMlp(3, hidden, hidden)
        self.transition_encoder = _SharedMlp(3 * hidden + 1, hidden, hidden)
        self.observation_edge = _SharedMlp(3 * hidden, hidden, hidden)
        self.record_message = _SharedMlp(4 * hidden, hidden, hidden)
        self.generator_message = _SharedMlp(2 * hidden, hidden, hidden)
        self.query_message = _SharedMlp(2 * hidden, hidden, hidden)
        self.record_update = nn.GRUCell(hidden, hidden)
        self.generator_update = nn.GRUCell(hidden, hidden)
        self.query_update = nn.GRUCell(hidden, hidden)
        self.record_norm = nn.LayerNorm(hidden)
        self.generator_norm = nn.LayerNorm(hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.pair_head = _SharedMlp(4 * hidden, hidden, 1)
        self._reset_parameters()

        count = self.parameter_count()
        if not count.under_cap:
            raise NeuralEndogenousCongruenceError(
                f"inducer has {count.total} parameters, cap is {count.cap}"
            )
        if not count.under_system_cap:
            raise NeuralEndogenousCongruenceError(
                f"complete system has {count.complete_system} parameters"
            )

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.record_seed, mean=0.0, std=0.02)
        nn.init.normal_(self.generator_seed, mean=0.0, std=0.02)
        nn.init.normal_(self.query_seed, mean=0.0, std=0.02)

    def parameter_count(self) -> NeuralEndogenousCongruenceParameterCount:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        complete_system = PROTECTED_BASE_PARAMETERS + total
        return NeuralEndogenousCongruenceParameterCount(
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
    ) -> NeuralEndogenousCongruenceOutput:
        batch_size = _require_tensor_batch(tensors)
        hidden = self.config.hidden_dim
        dtype = self.record_seed.dtype
        device = tensors.record_mask.device

        record = self.record_seed.view(1, 1, hidden).expand(batch_size, N, hidden)
        generator = self.generator_seed.view(1, 1, hidden).expand(
            batch_size,
            G,
            hidden,
        )
        query = self.query_seed.view(1, 1, hidden).expand(batch_size, Q, hidden)
        record = record * tensors.record_mask.to(dtype).unsqueeze(-1)
        generator = generator * tensors.generator_mask.to(dtype).unsqueeze(-1)
        query = query * tensors.query_mask.to(dtype).unsqueeze(-1)
        observation_features = self.observation_encoder(
            _normalized_observation(tensors.observation_value).to(dtype)
        )
        observation_features = observation_features * tensors.observation_mask.to(
            dtype
        ).unsqueeze(-1)
        record = record + _masked_mean(
            observation_features,
            tensors.observation_mask,
            2,
        )
        query = query + _masked_mean(
            observation_features.transpose(1, 2),
            tensors.observation_mask.transpose(1, 2),
            2,
        )

        transition = tensors.transition_target.to(dtype)
        for _ in range(self.config.rounds):
            target_record = torch.einsum("bign,bnh->bigh", transition, record)
            source_record = record[:, :, None, :].expand(-1, -1, G, -1)
            generator_edge = generator[:, None, :, :].expand(-1, N, -1, -1)
            fixed_point = torch.einsum(
                "bign,bin->big",
                transition,
                tensors.record_equal.to(dtype),
            ).unsqueeze(-1)
            transition_edge = self.transition_encoder(
                torch.cat(
                    (source_record, generator_edge, target_record, fixed_point),
                    dim=-1,
                )
            )
            transition_edge = transition_edge * tensors.transition_mask.to(
                dtype
            ).unsqueeze(-1)

            query_edge = self.observation_edge(
                torch.cat(
                    (
                        record[:, :, None, :].expand(-1, -1, Q, -1),
                        query[:, None, :, :].expand(-1, N, -1, -1),
                        observation_features,
                    ),
                    dim=-1,
                )
            )
            query_edge = query_edge * tensors.observation_mask.to(dtype).unsqueeze(-1)

            outgoing = _masked_mean(
                transition_edge,
                tensors.transition_mask,
                2,
            )
            incoming_total = torch.einsum(
                "bign,bigh->bnh",
                transition,
                transition_edge,
            )
            incoming_count = transition.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)
            incoming = incoming_total / incoming_count
            record_observation = _masked_mean(
                query_edge,
                tensors.observation_mask,
                2,
            )
            global_record = _masked_mean(record, tensors.record_mask, 1)
            record_input = self.record_message(
                torch.cat(
                    (
                        outgoing,
                        incoming,
                        record_observation,
                        global_record[:, None, :].expand(-1, N, -1),
                    ),
                    dim=-1,
                )
            )
            record = self.record_update(
                record_input.reshape(batch_size * N, hidden),
                record.reshape(batch_size * N, hidden),
            ).reshape(batch_size, N, hidden)
            record = self.record_norm(record)
            record = record * tensors.record_mask.to(dtype).unsqueeze(-1)

            generator_input = self.generator_message(
                torch.cat(
                    (
                        _masked_mean(
                            transition_edge.transpose(1, 2),
                            tensors.transition_mask.transpose(1, 2),
                            2,
                        ),
                        _masked_mean(record, tensors.record_mask, 1)[:, None, :].expand(
                            -1,
                            G,
                            -1,
                        ),
                    ),
                    dim=-1,
                )
            )
            generator = self.generator_update(
                generator_input.reshape(batch_size * G, hidden),
                generator.reshape(batch_size * G, hidden),
            ).reshape(batch_size, G, hidden)
            generator = self.generator_norm(generator)
            generator = generator * tensors.generator_mask.to(dtype).unsqueeze(-1)

            query_input = self.query_message(
                torch.cat(
                    (
                        _masked_mean(
                            query_edge.transpose(1, 2),
                            tensors.observation_mask.transpose(1, 2),
                            2,
                        ),
                        _masked_mean(record, tensors.record_mask, 1)[:, None, :].expand(
                            -1,
                            Q,
                            -1,
                        ),
                    ),
                    dim=-1,
                )
            )
            query = self.query_update(
                query_input.reshape(batch_size * Q, hidden),
                query.reshape(batch_size * Q, hidden),
            ).reshape(batch_size, Q, hidden)
            query = self.query_norm(query)
            query = query * tensors.query_mask.to(dtype).unsqueeze(-1)

        left = record[:, :, None, :].expand(-1, -1, N, -1)
        right = record[:, None, :, :].expand(-1, N, -1, -1)
        global_context = _masked_mean(record, tensors.record_mask, 1)
        pair_input = torch.cat(
            (
                left + right,
                torch.abs(left - right),
                left * right,
                global_context[:, None, None, :].expand(-1, N, N, -1),
            ),
            dim=-1,
        )
        same_class_logits = self.pair_head(pair_input).squeeze(-1)
        same_class_logits = 0.5 * (
            same_class_logits + same_class_logits.transpose(1, 2)
        )
        equivalence_mask = (
            tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
        )
        same_class_logits = torch.where(
            equivalence_mask,
            same_class_logits,
            torch.full(
                (),
                MASKED_LOGIT,
                dtype=dtype,
                device=device,
            ),
        )
        soft_equivalence = torch.sigmoid(same_class_logits)
        soft_equivalence = soft_equivalence * equivalence_mask.to(dtype)
        soft_projector = _relation_projector(
            soft_equivalence,
            tensors.record_mask,
        )
        residuals = _relation_residuals(tensors, soft_equivalence)
        return NeuralEndogenousCongruenceOutput(
            same_class_logits=same_class_logits,
            equivalence_mask=equivalence_mask,
            soft_equivalence=soft_equivalence,
            soft_projector=soft_projector,
            residuals=residuals,
            record_features=record,
            record_mask=tensors.record_mask,
            generator_features=generator,
            generator_mask=tensors.generator_mask,
            query_features=query,
            query_mask=tensors.query_mask,
        )


def decode_endogenous_congruence_logits(
    tensors: EndogenousCongruenceTensors,
    same_class_logits: Tensor,
    *,
    threshold: float = 0.0,
) -> DecodedEndogenousCongruence:
    """Threshold once and reject any relation that violates physical laws."""

    batch_size = _require_tensor_batch(tensors)
    if tuple(same_class_logits.shape) != (batch_size, N, N):
        raise NeuralEndogenousCongruenceError("same-class logits have invalid shape")
    if not torch.is_floating_point(same_class_logits):
        raise NeuralEndogenousCongruenceError("same-class logits must be floating")
    if same_class_logits.device != tensors.record_mask.device:
        raise NeuralEndogenousCongruenceError(
            "logits and tensors are on different devices"
        )
    if not torch.all(torch.isfinite(same_class_logits)):
        raise NeuralEndogenousCongruenceError("same-class logits are non-finite")
    if not isfinite(threshold):
        raise NeuralEndogenousCongruenceError("threshold must be finite")

    relation = same_class_logits >= threshold
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    if torch.any(relation & ~pair_mask):
        raise NeuralEndogenousCongruenceError("relation enters record padding")
    identity = torch.eye(
        N,
        dtype=torch.bool,
        device=relation.device,
    )[None]
    if not torch.equal(relation & identity, tensors.record_equal):
        raise NeuralEndogenousCongruenceError("relation is not reflexive")
    if not torch.equal(relation, relation.transpose(1, 2)):
        raise NeuralEndogenousCongruenceError("relation is not symmetric")
    composed = torch.einsum(
        "bij,bjk->bik",
        relation.to(torch.float32),
        relation.to(torch.float32),
    )
    if torch.any((composed > 0) & ~relation & pair_mask):
        raise NeuralEndogenousCongruenceError("relation is not transitive")

    residuals = _relation_residuals(tensors, relation.to(torch.float32))
    if torch.any(residuals.observation != 0):
        raise NeuralEndogenousCongruenceError("relation does not preserve observations")
    if torch.any(residuals.descent != 0):
        raise NeuralEndogenousCongruenceError(
            "relation is not compatible with generators"
        )
    projector = _relation_projector(relation.to(torch.float32), tensors.record_mask)
    projected_twice = torch.einsum("bij,bjk->bik", projector, projector)
    if not torch.allclose(projected_twice, projector, rtol=0.0, atol=1e-6):
        raise NeuralEndogenousCongruenceError("relation projector is not idempotent")
    return DecodedEndogenousCongruence(
        equivalence=relation,
        projector=projector,
        residuals=residuals,
        record_mask=tensors.record_mask,
        generator_mask=tensors.generator_mask,
        query_mask=tensors.query_mask,
    )


__all__ = [
    "MASKED_LOGIT",
    "PROTECTED_BASE_PARAMETERS",
    "SYSTEM_PARAMETER_CAP",
    "DecodedEndogenousCongruence",
    "EquivalenceResiduals",
    "NeuralEndogenousCongruence",
    "NeuralEndogenousCongruenceConfig",
    "NeuralEndogenousCongruenceError",
    "NeuralEndogenousCongruenceOutput",
    "NeuralEndogenousCongruenceParameterCount",
    "decode_endogenous_congruence_logits",
]
