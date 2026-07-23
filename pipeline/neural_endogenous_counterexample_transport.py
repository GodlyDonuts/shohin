"""Monotone counterexample transport over anonymous record pairs.

MCTFR maintains a learned state on every ordered record pair for exactly eight
tied rounds. Each round routes the state at ``(T_g(i), T_g(j))`` through the
physical one-hot transition tensor and aggregates the resulting generator
set with max and log-sum-exp channels. The first state coordinate is a
distinction channel whose update is nonnegative by construction.

The hard decoder concatenates immutable observation-equality fibers with
thresholded learned dynamical fibers and compares complete signature rows.
Consequently every hard output is an equivalence relation and preserves every
active observation query. Generator descent remains learned and falsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from pipeline.neural_endogenous_congruence import (
    MASKED_LOGIT,
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
    EquivalenceResiduals,
    _relation_residuals,
    _require_tensor_batch,
)
from pipeline.tensorize_endogenous_congruence import (
    G,
    N,
    Q,
    EndogenousCongruenceTensors,
)
from torch import Tensor, nn


MCTFR_ROUNDS = 8
MCTFR_PARAMETER_CAP = 24_000_000


class NeuralEndogenousCounterexampleTransportError(ValueError):
    """An MCTFR input or structural claim violates the frozen contract."""


@dataclass(frozen=True)
class NeuralEndogenousCounterexampleTransportConfig:
    """Architecture settings independent of episode-local entity counts."""

    hidden_dim: int = 192
    dynamical_bits: int = 4
    parameter_cap: int = MCTFR_PARAMETER_CAP
    soft_distance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 2:
            raise NeuralEndogenousCounterexampleTransportError(
                "hidden_dim must be at least two"
            )
        if self.dynamical_bits <= 0:
            raise NeuralEndogenousCounterexampleTransportError(
                "dynamical_bits must be positive"
            )
        if self.parameter_cap <= 0 or self.parameter_cap > MCTFR_PARAMETER_CAP:
            raise NeuralEndogenousCounterexampleTransportError(
                f"parameter_cap must be in [1, {MCTFR_PARAMETER_CAP}]"
            )
        if not isfinite(self.soft_distance_scale) or self.soft_distance_scale <= 0:
            raise NeuralEndogenousCounterexampleTransportError(
                "soft_distance_scale must be finite and positive"
            )


@dataclass(frozen=True)
class NeuralEndogenousCounterexampleTransportParameterCount:
    """Auditable standalone and protected-system parameter ledger."""

    total: int
    trainable: int
    cap: int
    protected_base: int
    complete_system: int
    system_cap: int
    headroom: int

    @property
    def under_cap(self) -> bool:
        return self.total < self.cap

    @property
    def under_system_cap(self) -> bool:
        return self.complete_system < self.system_cap


@dataclass(frozen=True)
class CounterexampleTransportHardDecoding:
    """One-threshold anonymous signatures and their guaranteed fibers."""

    observation_fibers: Tensor
    dynamical_fibers: Tensor
    signatures: Tensor
    signature_mask: Tensor
    equivalence: Tensor
    projector: Tensor
    record_mask: Tensor
    query_mask: Tensor


@dataclass(frozen=True)
class NeuralEndogenousCounterexampleTransportOutput:
    """Differentiable pair trajectory plus the single hard decoding."""

    pair_state_trace: tuple[Tensor, ...]
    dynamical_logits: Tensor
    dynamical_probabilities: Tensor
    soft_fiber_relation: Tensor
    soft_residuals: EquivalenceResiduals
    hard_residuals: EquivalenceResiduals
    hard: CounterexampleTransportHardDecoding

    @property
    def final_pair_state(self) -> Tensor:
        return self.pair_state_trace[-1]

    @property
    def distinction_trace(self) -> tuple[Tensor, ...]:
        return tuple(state[..., 0] for state in self.pair_state_trace)


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


def _pair_mask(record_mask: Tensor) -> Tensor:
    return record_mask[:, :, None] & record_mask[:, None, :]


def _active_identity(record_mask: Tensor) -> Tensor:
    identity = torch.eye(N, dtype=torch.bool, device=record_mask.device)[None]
    return identity & _pair_mask(record_mask)


def _within_query_observation_fibers(
    tensors: EndogenousCongruenceTensors,
) -> Tensor:
    # diagonal(q_left, q_right) has geometry [B, record, anchor, query].
    return tensors.observation_equal.diagonal(dim1=2, dim2=4).permute(0, 1, 3, 2)


def _stable_masked_max_logsumexp(
    value: Tensor,
    mask: Tensor,
    *,
    dimension: int,
) -> tuple[Tensor, Tensor]:
    """Permutation-invariant universal channels with no count-normalized mean."""

    if value.ndim != mask.ndim + 1 or value.shape[:-1] != mask.shape:
        raise NeuralEndogenousCounterexampleTransportError(
            "universal aggregation mask has invalid geometry"
        )
    if mask.dtype != torch.bool or mask.device != value.device:
        raise NeuralEndogenousCounterexampleTransportError(
            "universal aggregation mask has invalid dtype or device"
        )
    if torch.any(mask.sum(dim=dimension) == 0):
        raise NeuralEndogenousCounterexampleTransportError(
            "universal aggregation has an empty active set"
        )
    masked = value.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    maximum = masked.amax(dim=dimension)
    # Sorting makes the floating reduction order independent of entity order.
    ordered = masked.sort(dim=dimension).values
    logsumexp = torch.logsumexp(ordered, dim=dimension)
    return maximum, logsumexp


def gather_aligned_successor_pairs(
    pair_state: Tensor,
    transition_target: Tensor,
) -> Tensor:
    """Gather ``z[T_g(i), T_g(j)]`` for each same-generator edge."""

    if type(pair_state) is not Tensor or type(transition_target) is not Tensor:
        raise NeuralEndogenousCounterexampleTransportError(
            "successor gather accepts exact tensors"
        )
    if (
        pair_state.ndim != 4
        or pair_state.shape[1:3] != (N, N)
        or not torch.is_floating_point(pair_state)
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "pair_state has invalid geometry or dtype"
        )
    if (
        transition_target.shape != (pair_state.shape[0], N, G, N)
        or transition_target.dtype != torch.bool
        or transition_target.device != pair_state.device
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "transition_target has invalid geometry, dtype, or device"
        )
    transition = transition_target.to(pair_state.dtype)
    return torch.einsum(
        "biga,bach,bjgc->bijgh",
        transition,
        pair_state,
        transition,
    )


def _validate_observation_fibers(
    observation_fibers: Tensor,
    record_mask: Tensor,
    query_mask: Tensor,
) -> int:
    if type(record_mask) is not Tensor or type(query_mask) is not Tensor:
        raise NeuralEndogenousCounterexampleTransportError(
            "fiber masks are not tensors"
        )
    if (
        record_mask.ndim != 2
        or record_mask.shape[1] != N
        or record_mask.dtype != torch.bool
        or query_mask.ndim != 2
        or query_mask.shape != (record_mask.shape[0], Q)
        or query_mask.dtype != torch.bool
        or query_mask.device != record_mask.device
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "fiber masks have invalid geometry, dtype, or device"
        )
    batch_size = record_mask.shape[0]
    if batch_size == 0:
        raise NeuralEndogenousCounterexampleTransportError("fiber batch is empty")
    if (
        type(observation_fibers) is not Tensor
        or observation_fibers.shape != (batch_size, N, Q, N)
        or observation_fibers.dtype != torch.bool
        or observation_fibers.device != record_mask.device
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "observation_fibers have invalid geometry, dtype, or device"
        )
    active = (
        record_mask[:, :, None, None]
        & query_mask[:, None, :, None]
        & record_mask[:, None, None, :]
    )
    if torch.any(observation_fibers & ~active):
        raise NeuralEndogenousCounterexampleTransportError(
            "observation_fibers enter padding"
        )
    fibers_by_query = observation_fibers.permute(0, 2, 1, 3)
    pair_mask = _pair_mask(record_mask)
    query_pair_mask = query_mask[:, :, None, None] & pair_mask[:, None]
    identity = _active_identity(record_mask)
    if not torch.equal(
        fibers_by_query & identity[:, None],
        query_pair_mask & identity[:, None],
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "observation_fibers are not reflexive"
        )
    if not torch.equal(fibers_by_query, fibers_by_query.transpose(2, 3)):
        raise NeuralEndogenousCounterexampleTransportError(
            "observation_fibers are not symmetric"
        )
    composed = torch.einsum(
        "bqij,bqjk->bqik",
        fibers_by_query.to(torch.float32),
        fibers_by_query.to(torch.float32),
    )
    if torch.any((composed > 0) & ~fibers_by_query & query_pair_mask):
        raise NeuralEndogenousCounterexampleTransportError(
            "observation_fibers are not transitive"
        )
    return batch_size


def _validate_dynamical_logits(
    dynamical_logits: Tensor,
    record_mask: Tensor,
) -> None:
    pair_mask = _pair_mask(record_mask)
    if (
        type(dynamical_logits) is not Tensor
        or dynamical_logits.ndim != 4
        or dynamical_logits.shape[:3] != pair_mask.shape
        or dynamical_logits.shape[-1] < 1
        or not torch.is_floating_point(dynamical_logits)
        or dynamical_logits.device != record_mask.device
    ):
        raise NeuralEndogenousCounterexampleTransportError(
            "dynamical_logits have invalid geometry, dtype, or device"
        )
    if not torch.all(torch.isfinite(dynamical_logits)):
        raise NeuralEndogenousCounterexampleTransportError(
            "dynamical_logits contain non-finite values"
        )
    padding = ~pair_mask.unsqueeze(-1)
    expected = torch.as_tensor(
        MASKED_LOGIT,
        dtype=dynamical_logits.dtype,
        device=dynamical_logits.device,
    )
    if torch.any(dynamical_logits.masked_select(padding) != expected):
        raise NeuralEndogenousCounterexampleTransportError(
            "dynamical_logits must be exactly masked in padding"
        )


def _exact_projector(equivalence: Tensor, record_mask: Tensor) -> Tensor:
    relation = equivalence.to(torch.float32)
    relation *= _pair_mask(record_mask).to(torch.float32)
    return relation / relation.sum(dim=-1, keepdim=True).clamp_min(1.0)


def decode_counterexample_transport_fibers(
    observation_fibers: Tensor,
    dynamical_logits: Tensor,
    record_mask: Tensor,
    query_mask: Tensor,
    *,
    threshold: float = 0.0,
) -> CounterexampleTransportHardDecoding:
    """Threshold learned bits once, then decode equality of complete rows."""

    _validate_observation_fibers(
        observation_fibers,
        record_mask,
        query_mask,
    )
    _validate_dynamical_logits(dynamical_logits, record_mask)
    if not isfinite(threshold):
        raise NeuralEndogenousCounterexampleTransportError(
            "hard threshold must be finite"
        )

    pair_mask = _pair_mask(record_mask)
    dynamical_fibers = (dynamical_logits > threshold) & pair_mask.unsqueeze(-1)
    observation_signature = observation_fibers.reshape(
        observation_fibers.shape[0],
        N,
        Q * N,
    )
    dynamical_signature = dynamical_fibers.reshape(
        dynamical_fibers.shape[0],
        N,
        N * dynamical_fibers.shape[-1],
    )
    signatures = torch.cat((observation_signature, dynamical_signature), dim=-1)

    observation_signature_mask = (
        query_mask[:, :, None] & record_mask[:, None, :]
    ).reshape(record_mask.shape[0], Q * N)
    dynamical_signature_mask = (
        record_mask[:, :, None]
        .expand(-1, -1, dynamical_fibers.shape[-1])
        .reshape(record_mask.shape[0], -1)
    )
    signature_mask = torch.cat(
        (observation_signature_mask, dynamical_signature_mask),
        dim=-1,
    )
    signatures &= record_mask[:, :, None] & signature_mask[:, None, :]
    bit_equal = signatures[:, :, None, :] == signatures[:, None, :, :]
    equivalence = (bit_equal | ~signature_mask[:, None, None, :]).all(
        dim=-1
    ) & pair_mask
    return CounterexampleTransportHardDecoding(
        observation_fibers=observation_fibers,
        dynamical_fibers=dynamical_fibers,
        signatures=signatures,
        signature_mask=signature_mask,
        equivalence=equivalence,
        projector=_exact_projector(equivalence, record_mask),
        record_mask=record_mask,
        query_mask=query_mask,
    )


def _soft_fiber_relation(
    observation_fibers: Tensor,
    dynamical_probabilities: Tensor,
    record_mask: Tensor,
    query_mask: Tensor,
    *,
    distance_scale: float,
) -> Tensor:
    observation_signature = observation_fibers.reshape(
        observation_fibers.shape[0],
        N,
        Q * N,
    )
    observation_mask = (query_mask[:, :, None] & record_mask[:, None, :]).reshape(
        record_mask.shape[0], Q * N
    )
    observation_equal = (
        observation_signature[:, :, None, :] == observation_signature[:, None, :, :]
    )
    observation_compatible = (
        observation_equal | ~observation_mask[:, None, None, :]
    ).all(dim=-1)

    dynamic = dynamical_probabilities.reshape(
        dynamical_probabilities.shape[0],
        N,
        -1,
    )
    dynamic_mask = (
        record_mask[:, :, None]
        .expand(-1, -1, dynamical_probabilities.shape[-1])
        .reshape(record_mask.shape[0], -1)
    )
    difference = dynamic[:, :, None, :] - dynamic[:, None, :, :]
    distance = (
        difference.square() * dynamic_mask[:, None, None, :].to(dynamic.dtype)
    ).sum(dim=-1)
    relation = observation_compatible.to(dynamic.dtype) * torch.exp(
        -distance_scale * distance
    )
    return relation * _pair_mask(record_mask).to(dynamic.dtype)


class NeuralEndogenousCounterexampleTransport(nn.Module):
    """Eight-round tied monotone reactor over physical ordered-pair edges."""

    def __init__(
        self,
        config: NeuralEndogenousCounterexampleTransportConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NeuralEndogenousCounterexampleTransportConfig()
        hidden = self.config.hidden_dim
        auxiliary = hidden - 1

        self.query_encoder = _SharedMlp(2, hidden, hidden)
        self.initial_auxiliary = _SharedMlp(2 * hidden + 1, hidden, auxiliary)
        self.successor_encoder = _SharedMlp(hidden, hidden, hidden)
        self.distinction_increment = _SharedMlp(3 * hidden, hidden, 1)
        self.auxiliary_update = _SharedMlp(3 * hidden, hidden, auxiliary)
        self.auxiliary_norm = nn.LayerNorm(auxiliary)
        self.dynamical_head = _SharedMlp(
            hidden,
            hidden,
            self.config.dynamical_bits,
        )

        count = self.parameter_count()
        if not count.under_cap:
            raise NeuralEndogenousCounterexampleTransportError(
                f"MCTFR has {count.total} parameters, cap is {count.cap}"
            )
        if not count.under_system_cap:
            raise NeuralEndogenousCounterexampleTransportError(
                f"complete system has {count.complete_system} parameters"
            )

    def parameter_count(
        self,
    ) -> NeuralEndogenousCounterexampleTransportParameterCount:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        complete = PROTECTED_BASE_PARAMETERS + total
        return NeuralEndogenousCounterexampleTransportParameterCount(
            total=total,
            trainable=trainable,
            cap=self.config.parameter_cap,
            protected_base=PROTECTED_BASE_PARAMETERS,
            complete_system=complete,
            system_cap=SYSTEM_PARAMETER_CAP,
            headroom=SYSTEM_PARAMETER_CAP - complete,
        )

    def _initial_pair_state(
        self,
        tensors: EndogenousCongruenceTensors,
    ) -> tuple[Tensor, Tensor]:
        observation_fibers = _within_query_observation_fibers(tensors)
        equality = observation_fibers.permute(0, 1, 3, 2)
        token = torch.stack(
            (
                equality.to(torch.float32),
                (~equality).to(torch.float32),
            ),
            dim=-1,
        )
        encoded = self.query_encoder(token)
        query_mask = tensors.query_mask[:, None, None, :].expand(
            -1,
            N,
            N,
            -1,
        )
        maximum, logsumexp = _stable_masked_max_logsumexp(
            encoded,
            query_mask,
            dimension=3,
        )
        distinction = (
            ((~equality) & query_mask).any(dim=3).to(encoded.dtype).unsqueeze(-1)
        )
        auxiliary = self.initial_auxiliary(
            torch.cat((maximum, logsumexp, distinction), dim=-1)
        )
        state = torch.cat((distinction, auxiliary), dim=-1)
        state *= _pair_mask(tensors.record_mask).to(state.dtype).unsqueeze(-1)
        return state, observation_fibers

    def _reactor_round(
        self,
        pair_state: Tensor,
        tensors: EndogenousCongruenceTensors,
    ) -> Tensor:
        successors = gather_aligned_successor_pairs(
            pair_state,
            tensors.transition_target,
        )
        encoded_successors = self.successor_encoder(successors)
        generator_mask = tensors.generator_mask[:, None, None, :].expand(
            -1,
            N,
            N,
            -1,
        )
        maximum, logsumexp = _stable_masked_max_logsumexp(
            encoded_successors,
            generator_mask,
            dimension=3,
        )
        context = torch.cat((pair_state, maximum, logsumexp), dim=-1)
        pair_mask = _pair_mask(tensors.record_mask)
        off_diagonal = pair_mask & ~_active_identity(tensors.record_mask)

        increment = torch.nn.functional.softplus(self.distinction_increment(context))
        increment *= off_diagonal.to(increment.dtype).unsqueeze(-1)
        distinction = pair_state[..., :1] + increment

        auxiliary_delta = self.auxiliary_update(context)
        auxiliary = self.auxiliary_norm(pair_state[..., 1:] + auxiliary_delta)
        output = torch.cat((distinction, auxiliary), dim=-1)
        return output * pair_mask.to(output.dtype).unsqueeze(-1)

    def forward(
        self,
        tensors: EndogenousCongruenceTensors,
    ) -> NeuralEndogenousCounterexampleTransportOutput:
        """Run one fixed source-deleted MCTFR pass and one hard decode."""

        _require_tensor_batch(tensors)
        pair_state, observation_fibers = self._initial_pair_state(tensors)
        trace = [pair_state]
        for _ in range(MCTFR_ROUNDS):
            pair_state = self._reactor_round(pair_state, tensors)
            trace.append(pair_state)

        pair_mask = _pair_mask(tensors.record_mask)
        dynamical_logits = self.dynamical_head(pair_state)
        dynamical_logits = torch.where(
            pair_mask.unsqueeze(-1),
            dynamical_logits,
            torch.full(
                (),
                MASKED_LOGIT,
                dtype=dynamical_logits.dtype,
                device=dynamical_logits.device,
            ),
        )
        hard = decode_counterexample_transport_fibers(
            observation_fibers,
            dynamical_logits,
            tensors.record_mask,
            tensors.query_mask,
        )
        dynamical_probabilities = torch.sigmoid(dynamical_logits) * pair_mask.to(
            dynamical_logits.dtype
        ).unsqueeze(-1)
        soft_fiber_relation = _soft_fiber_relation(
            observation_fibers,
            dynamical_probabilities,
            tensors.record_mask,
            tensors.query_mask,
            distance_scale=self.config.soft_distance_scale,
        )
        return NeuralEndogenousCounterexampleTransportOutput(
            pair_state_trace=tuple(trace),
            dynamical_logits=dynamical_logits,
            dynamical_probabilities=dynamical_probabilities,
            soft_fiber_relation=soft_fiber_relation,
            soft_residuals=_relation_residuals(tensors, soft_fiber_relation),
            hard_residuals=_relation_residuals(
                tensors,
                hard.equivalence.to(torch.float32),
            ),
            hard=hard,
        )


__all__ = [
    "MCTFR_PARAMETER_CAP",
    "MCTFR_ROUNDS",
    "CounterexampleTransportHardDecoding",
    "NeuralEndogenousCounterexampleTransport",
    "NeuralEndogenousCounterexampleTransportConfig",
    "NeuralEndogenousCounterexampleTransportError",
    "NeuralEndogenousCounterexampleTransportOutput",
    "NeuralEndogenousCounterexampleTransportParameterCount",
    "decode_counterexample_transport_fibers",
    "gather_aligned_successor_pairs",
]
