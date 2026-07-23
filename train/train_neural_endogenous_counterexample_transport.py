#!/usr/bin/env python3
"""Train and assess source-deleted MCTFR congruence induction.

The neural boundary receives only ``EndogenousCongruenceTensors``. Packet
digests, target relations, split metadata, renderer morphisms, orbit identity,
family, motif, variant, and assessor results remain offline. Optimization uses
only final target-relation supervision; no intermediate partition,
counterexample path, distinction certificate, or oracle state is supplied.

The model performs one fixed eight-round pass and one frozen hard decode.
Assessment never repairs, closes, searches, retries, or reclusters that output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Literal

import torch
from pipeline.neural_endogenous_congruence import (
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
)
from pipeline.neural_endogenous_counterexample_transport import (
    MCTFR_ROUNDS,
    NeuralEndogenousCounterexampleTransport,
    NeuralEndogenousCounterexampleTransportConfig,
    NeuralEndogenousCounterexampleTransportOutput,
    gather_aligned_successor_pairs,
)
from pipeline.tensorize_endogenous_congruence import (
    N,
    EndogenousCongruenceTensors,
)
from torch import Tensor, nn
from torch.nn import functional as F

import train_neural_endogenous_congruence as audited_harness


REPORT_SCHEMA = "neural_endogenous_counterexample_transport_exploratory_v1"
CHECKPOINT_SCHEMA = "neural_endogenous_counterexample_transport_checkpoint_v1"
BUNDLE_SCHEMA = "neural_endogenous_counterexample_transport_bundle_v1"
HARD_THRESHOLD = 0.0
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_BASE_CHECKPOINT = REPOSITORY_ROOT / "train/flagship_out/ckpt_0300000.pt"
PROTECTED_BASE_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
OfflinePartition = audited_harness.OfflinePartition
ProceduralPartitions = audited_harness.ProceduralPartitions
OfflineBatch = audited_harness.OfflineBatch
load_procedural_partitions = audited_harness.load_procedural_partitions
subset_partition = audited_harness.subset_partition


class EndogenousCounterexampleTransportTrainingHarnessError(ValueError):
    """A score-bearing custody, objective, or artifact invariant failed."""


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen exploratory MCTFR optimizer settings."""

    updates: int = 800
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    active_bit_margin: float = 1.0
    descent_temperature: float = 0.1
    target_control: Literal["true", "shuffled"] = "true"
    seed: int = 2026072305
    log_interval: int = 25

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "updates must be positive"
            )
        if self.batch_size < 2:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "batch size must be at least two"
            )
        finite_positive = {
            "learning rate": self.learning_rate,
            "gradient clip": self.gradient_clip,
            "active-bit margin": self.active_bit_margin,
            "descent temperature": self.descent_temperature,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise EndogenousCounterexampleTransportTrainingHarnessError(
                    f"{name} must be finite and positive"
                )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "weight decay must be finite and nonnegative"
            )
        if self.target_control not in {"true", "shuffled"}:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "target control must be true or shuffled"
            )
        if self.log_interval <= 0:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "log interval must be positive"
            )


@dataclass(frozen=True)
class CounterexampleTransportLoss:
    """All preregistered MCTFR objective components."""

    total: Tensor
    balanced_fiber_relation: Tensor
    fiber: Tensor
    relation: Tensor
    max_descent: Tensor
    fixed_point: Tensor
    margin: Tensor


@dataclass(frozen=True)
class ObjectiveSnapshot:
    """Detached objective and physical residual audit."""

    total: float
    balanced_fiber_relation: float
    fiber: float
    relation: float
    max_descent: float
    fixed_point: float
    margin: float
    soft_descent_residual: float
    soft_observation_residual: float
    hard_descent_residual: float
    hard_observation_residual: float


@dataclass(frozen=True)
class HardExampleResult:
    """One model pass, one hard decode, and offline assessment only."""

    packet_sha256: str
    orbit_id: str
    variant: str
    family: str
    motif: str
    cell: str
    records: tuple[str, ...]
    equivalence_valid: bool
    observation_valid: bool
    descent_valid: bool
    physical_law_valid: bool
    projector_symmetric: bool
    projector_idempotent: bool
    projector_row_stochastic: bool
    exact_target_relation: bool
    coarsest_target_relation: bool
    false_splits: int
    false_collisions: int
    target_positive_pairs: int
    target_negative_pairs: int
    predicted_positive_pairs: int
    hard_descent_residual: float
    hard_observation_residual: float
    minimum_absolute_active_bit_margin: float
    equivalent_pairs: tuple[tuple[str, str], ...]
    elapsed_seconds: float

    @property
    def valid_decode(self) -> bool:
        """Compatibility with the audited renderer-orbit assessor."""

        return self.equivalence_valid


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _protected_base_receipt() -> dict[str, object]:
    if not PROTECTED_BASE_CHECKPOINT.is_file():
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "protected base checkpoint is missing"
        )
    actual = _sha256_file(PROTECTED_BASE_CHECKPOINT)
    if actual != PROTECTED_BASE_SHA256:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "protected base checkpoint hash drifted"
        )
    return {
        "path": str(PROTECTED_BASE_CHECKPOINT.relative_to(REPOSITORY_ROOT)),
        "sha256": actual,
        "parameters": PROTECTED_BASE_PARAMETERS,
        "integration_status": (
            "not integrated; complete-system count is hypothetical accounting"
        ),
    }


def _pair_mask(record_mask: Tensor) -> Tensor:
    return record_mask[:, :, None] & record_mask[:, None, :]


def _active_identity(record_mask: Tensor) -> Tensor:
    identity = torch.eye(
        N,
        dtype=torch.bool,
        device=record_mask.device,
    )[None]
    return identity & _pair_mask(record_mask)


def _prepare_batch(
    partition: OfflinePartition,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> OfflineBatch:
    """Reuse the audited SHA-256 and axis-receipt target join."""

    return audited_harness._prepare_batch(  # noqa: SLF001
        partition,
        indices,
        device=device,
    )


def _forward_tensor_only(
    model: nn.Module,
    tensors: EndogenousCongruenceTensors,
) -> NeuralEndogenousCounterexampleTransportOutput:
    """Cross the neural boundary with the exact physical tensor type only."""

    if type(tensors) is not EndogenousCongruenceTensors:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "model forward accepts only exact EndogenousCongruenceTensors"
        )
    output = model(tensors)
    if not isinstance(
        output,
        NeuralEndogenousCounterexampleTransportOutput,
    ):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "model did not return NeuralEndogenousCounterexampleTransportOutput"
        )
    return output


def _require_target_equivalence(
    target: Tensor,
    record_mask: Tensor,
) -> None:
    expected = (record_mask.shape[0], N, N)
    if (
        type(target) is not Tensor
        or tuple(target.shape) != expected
        or target.dtype != torch.bool
        or target.device != record_mask.device
    ):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "target relation has invalid geometry, dtype, or device"
        )
    pair_mask = _pair_mask(record_mask)
    if torch.any(target & ~pair_mask):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "target relation enters record padding"
        )
    identity = _active_identity(record_mask)
    if not torch.equal(target & identity, identity):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "target relation is not reflexive"
        )
    if not torch.equal(target, target.transpose(1, 2)):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "target relation is not symmetric"
        )
    composed = torch.einsum(
        "bij,bjk->bik",
        target.to(torch.float32),
        target.to(torch.float32),
    )
    if torch.any((composed > 0) & ~target & pair_mask):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "target relation is not transitive"
        )


def _balanced_binary_probability_loss(
    probability: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    epsilon = torch.finfo(probability.dtype).eps
    probability = probability.clamp(min=epsilon, max=1.0 - epsilon)
    positive = mask & target
    negative = mask & ~target
    terms: list[Tensor] = []
    if torch.any(positive):
        terms.append(-torch.log(probability.masked_select(positive)).mean())
    if torch.any(negative):
        terms.append(-torch.log1p(-probability.masked_select(negative)).mean())
    if not terms:
        return probability.sum() * 0.0
    return torch.stack(terms).mean()


def _balanced_fiber_loss(
    logits: Tensor,
    target: Tensor,
    pair_mask: Tensor,
) -> Tensor:
    expanded_mask = pair_mask.unsqueeze(-1).expand_as(logits)
    expanded_target = target.unsqueeze(-1).expand_as(logits)
    positive = expanded_mask & expanded_target
    negative = expanded_mask & ~expanded_target
    terms: list[Tensor] = []
    if torch.any(positive):
        terms.append(F.softplus(-logits.masked_select(positive)).mean())
    if torch.any(negative):
        terms.append(F.softplus(logits.masked_select(negative)).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _smooth_max_descent_hinge(
    relation: Tensor,
    tensors: EndogenousCongruenceTensors,
    *,
    temperature: float,
) -> Tensor:
    successor = gather_aligned_successor_pairs(
        relation.unsqueeze(-1),
        tensors.transition_target,
    ).squeeze(-1)
    violation = relation.unsqueeze(-1) - successor
    active = (
        _pair_mask(tensors.record_mask).unsqueeze(-1)
        & tensors.generator_mask[:, None, None, :]
    )
    episode_terms: list[Tensor] = []
    for episode in range(relation.shape[0]):
        values = violation[episode].masked_select(active[episode])
        if values.numel() == 0:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "descent objective has no active constraints"
            )
        worst_violation = values.amax()
        smooth_hinge = temperature * F.softplus(worst_violation / temperature)
        episode_terms.append(smooth_hinge.square())
    return torch.stack(episode_terms).mean()


def _fixed_point_loss(
    output: NeuralEndogenousCounterexampleTransportOutput,
    record_mask: Tensor,
) -> Tensor:
    if len(output.pair_state_trace) != MCTFR_ROUNDS + 1:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "MCTFR trace length differs from the frozen eight-round contract"
        )
    previous = output.penultimate_soft_fiber_relation
    final = output.soft_fiber_relation
    return (final - previous).square().masked_select(_pair_mask(record_mask)).mean()


def counterexample_transport_loss(
    output: NeuralEndogenousCounterexampleTransportOutput,
    tensors: EndogenousCongruenceTensors,
    target_equivalence: Tensor,
    *,
    active_bit_margin: float = 1.0,
    descent_temperature: float = 0.1,
) -> CounterexampleTransportLoss:
    """Apply only final-relation supervision plus physical regularizers."""

    if type(output) is not NeuralEndogenousCounterexampleTransportOutput:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "output has an invalid type"
        )
    if type(tensors) is not EndogenousCongruenceTensors:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "loss physical input has an invalid type"
        )
    if (
        not math.isfinite(active_bit_margin)
        or active_bit_margin <= 0
        or not math.isfinite(descent_temperature)
        or descent_temperature <= 0
    ):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "loss margins and temperatures must be finite and positive"
        )
    record_mask = tensors.record_mask
    _require_target_equivalence(target_equivalence, record_mask)
    pair_mask = _pair_mask(record_mask)

    fiber = _balanced_fiber_loss(
        output.dynamical_logits,
        target_equivalence,
        pair_mask,
    )
    relation = _balanced_binary_probability_loss(
        output.soft_fiber_relation,
        target_equivalence,
        pair_mask,
    )
    balanced = 0.5 * (fiber + relation)
    max_descent = _smooth_max_descent_hinge(
        output.soft_fiber_relation,
        tensors,
        temperature=descent_temperature,
    )
    fixed_point = _fixed_point_loss(output, record_mask)
    active_logits = output.dynamical_logits.masked_select(
        pair_mask.unsqueeze(-1).expand_as(output.dynamical_logits)
    )
    margin = F.relu(active_bit_margin - active_logits.abs()).square().mean()
    total = balanced + max_descent + 0.5 * fixed_point + 0.1 * margin
    if not torch.isfinite(total):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "MCTFR objective is non-finite"
        )
    return CounterexampleTransportLoss(
        total=total,
        balanced_fiber_relation=balanced,
        fiber=fiber,
        relation=relation,
        max_descent=max_descent,
        fixed_point=fixed_point,
        margin=margin,
    )


def _sample_unique_batch(
    partition: OfflinePartition,
    *,
    batch_size: int,
    sampler: random.Random,
) -> tuple[int, ...]:
    """Sample unique train packets using only partition length."""

    if partition.name != "train":
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "batch sampler accepts only the train partition"
        )
    if batch_size > len(partition.packets):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "batch size exceeds unique training packets"
        )
    indices = tuple(sampler.sample(range(len(partition.packets)), k=batch_size))
    if len(indices) != len(set(indices)):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "batch sampler did not produce unique packets"
        )
    return indices


def _shuffle_target_relations(
    target: Tensor,
    record_mask: Tensor,
) -> tuple[Tensor, int]:
    """Create one deterministic valid final-target negative control."""

    _require_target_equivalence(target, record_mask)
    shuffled = torch.zeros_like(target)
    changed = 0
    for episode in range(target.shape[0]):
        active = int(record_mask[episode].sum().item())
        source = target[episode, :active, :active]
        candidate = source
        anchor = next(
            (
                left
                for left in range(active)
                if int(source[left].sum().item()) > 1
                and int(source[left].sum().item()) < active
            ),
            None,
        )
        if anchor is not None:
            outside = next(
                right for right in range(active) if not bool(source[anchor, right])
            )
            order = list(range(active))
            order[anchor], order[outside] = order[outside], order[anchor]
            axis = torch.tensor(order, dtype=torch.long, device=target.device)
            candidate = source.index_select(0, axis).index_select(1, axis)
            if torch.equal(candidate, source):
                raise EndogenousCounterexampleTransportTrainingHarnessError(
                    "nondegenerate deterministic target control did not change"
                )
        shuffled[episode, :active, :active] = candidate
        changed += int(not torch.equal(candidate, source))
    _require_target_equivalence(shuffled, record_mask)
    return shuffled, changed


def _controlled_targets(
    target: Tensor,
    record_mask: Tensor,
    *,
    target_control: str,
) -> tuple[Tensor, int]:
    if target_control == "true":
        return target, 0
    if target_control == "shuffled":
        return _shuffle_target_relations(
            target,
            record_mask,
        )
    raise EndogenousCounterexampleTransportTrainingHarnessError(
        "unknown target control"
    )


def _objective_snapshot(
    objective: CounterexampleTransportLoss,
    output: NeuralEndogenousCounterexampleTransportOutput,
) -> ObjectiveSnapshot:
    return ObjectiveSnapshot(
        total=float(objective.total.detach().item()),
        balanced_fiber_relation=float(
            objective.balanced_fiber_relation.detach().item()
        ),
        fiber=float(objective.fiber.detach().item()),
        relation=float(objective.relation.detach().item()),
        max_descent=float(objective.max_descent.detach().item()),
        fixed_point=float(objective.fixed_point.detach().item()),
        margin=float(objective.margin.detach().item()),
        soft_descent_residual=float(output.soft_residuals.descent.mean().item()),
        soft_observation_residual=float(
            output.soft_residuals.observation.mean().item()
        ),
        hard_descent_residual=float(output.hard_residuals.descent.mean().item()),
        hard_observation_residual=float(
            output.hard_residuals.observation.mean().item()
        ),
    )


def _calibration_snapshot(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    indices: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
) -> ObjectiveSnapshot:
    was_training = model.training
    model.eval()
    batch = _prepare_batch(partition, indices, device=device)
    targets, _ = _controlled_targets(
        batch.same_class_target,
        batch.tensorization.tensors.record_mask,
        target_control=config.target_control,
    )
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        objective = counterexample_transport_loss(
            output,
            batch.tensorization.tensors,
            targets,
            active_bit_margin=config.active_bit_margin,
            descent_temperature=config.descent_temperature,
        )
        snapshot = _objective_snapshot(objective, output)
    if was_training:
        model.train()
    return snapshot


def train_counterexample_transport(
    model: NeuralEndogenousCounterexampleTransport,
    training: OfflinePartition,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    """Optimize an explicit train partition without accepting development."""

    if training.name != "train":
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "optimizer accepts only an explicit train partition"
        )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sampler = random.Random(config.seed)
    calibration_indices = _sample_unique_batch(
        training,
        batch_size=config.batch_size,
        sampler=random.Random(config.seed ^ 0xCA11B),
    )
    initial = _calibration_snapshot(
        model,
        training,
        indices=calibration_indices,
        config=config,
        device=device,
    )
    trace: list[dict[str, object]] = []
    optimizer_digests: set[str] = set()
    target_control_changed_examples = 0
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        indices = _sample_unique_batch(
            training,
            batch_size=config.batch_size,
            sampler=sampler,
        )
        batch = _prepare_batch(training, indices, device=device)
        targets, changed = _controlled_targets(
            batch.same_class_target,
            batch.tensorization.tensors.record_mask,
            target_control=config.target_control,
        )
        target_control_changed_examples += changed
        optimizer_digests.update(batch.packet_digests)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        objective = counterexample_transport_loss(
            output,
            batch.tensorization.tensors,
            targets,
            active_bit_margin=config.active_bit_margin,
            descent_temperature=config.descent_temperature,
        )
        objective.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                "gradient norm is non-finite"
            )
        optimizer.step()
        if update == 1 or update % config.log_interval == 0 or update == config.updates:
            item = {
                "update": update,
                **asdict(_objective_snapshot(objective, output)),
                "gradient_norm": float(gradient_norm.detach().item()),
                "target_control_changed_examples": changed,
            }
            trace.append(item)
            print(_canonical_json(item), flush=True)
    final = _calibration_snapshot(
        model,
        training,
        indices=calibration_indices,
        config=config,
        device=device,
    )
    return {
        "initial_calibration": asdict(initial),
        "final_calibration": asdict(final),
        "loss_decreased": final.total < initial.total,
        "optimizer_packet_digests": sorted(optimizer_digests),
        "optimizer_development_packet_digests": [],
        "unique_optimizer_packets": len(optimizer_digests),
        "target_control": config.target_control,
        "target_control_changed_examples": target_control_changed_examples,
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _target_relation_for_singleton(batch: OfflineBatch) -> Tensor:
    if batch.same_class_target.shape[0] != 1:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "hard assessment requires a singleton offline target batch"
        )
    return batch.same_class_target[0].detach().cpu()


def _equivalence_is_valid(relation: Tensor, record_mask: Tensor) -> bool:
    if (
        type(relation) is not Tensor
        or tuple(relation.shape) != (N, N)
        or relation.dtype != torch.bool
        or type(record_mask) is not Tensor
        or tuple(record_mask.shape) != (N,)
        or record_mask.dtype != torch.bool
    ):
        return False
    pair_mask = record_mask[:, None] & record_mask[None, :]
    if torch.any(relation & ~pair_mask):
        return False
    identity = torch.diag(record_mask)
    if not torch.equal(relation & identity, identity):
        return False
    if not torch.equal(relation, relation.transpose(0, 1)):
        return False
    composed = torch.einsum(
        "ij,jk->ik",
        relation.to(torch.float32),
        relation.to(torch.float32),
    )
    return not bool(torch.any((composed > 0) & ~relation & pair_mask))


def _projector_laws(
    projector: Tensor,
    record_mask: Tensor,
) -> tuple[bool, bool, bool]:
    projected_twice = projector @ projector
    symmetric = bool(
        torch.isclose(
            projector,
            projector.transpose(0, 1),
            rtol=0.0,
            atol=1e-6,
        ).all()
    )
    idempotent = bool(
        torch.isclose(
            projected_twice,
            projector,
            rtol=0.0,
            atol=1e-6,
        ).all()
    )
    stochastic = bool(
        torch.isclose(
            projector.sum(dim=-1),
            record_mask.to(torch.float32),
            rtol=0.0,
            atol=1e-6,
        ).all()
    )
    return symmetric, idempotent, stochastic


def assess_one_example(
    model: nn.Module,
    partition: OfflinePartition,
    index: int,
    *,
    device: torch.device,
) -> HardExampleResult:
    """Decode once, then assess target and physical laws offline."""

    started = time.perf_counter()
    batch = _prepare_batch(partition, (index,), device=device)
    metadata = partition.metadata[index]
    packet = partition.packets[index]
    target = _target_relation_for_singleton(batch)
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)

    predicted = output.hard.equivalence[0].detach().cpu()
    record_mask = output.hard.record_mask[0].detach().cpu()
    equivalence_valid = _equivalence_is_valid(predicted, record_mask)
    if not equivalence_valid:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "MCTFR hard decoder violated its equivalence guarantee"
        )
    observation_valid = bool(output.hard_residuals.observation[0].item() == 0)
    descent_valid = bool(output.hard_residuals.descent[0].item() == 0)
    projector_symmetric, projector_idempotent, projector_row_stochastic = (
        _projector_laws(
            output.hard.projector[0].detach().cpu(),
            record_mask,
        )
    )
    physical_valid = (
        equivalence_valid
        and observation_valid
        and descent_valid
        and projector_symmetric
        and projector_idempotent
        and projector_row_stochastic
    )
    active_pairs = record_mask[:, None] & record_mask[None, :]
    unordered = torch.triu(active_pairs, diagonal=1)
    target_positive = unordered & target
    target_negative = unordered & ~target
    predicted_positive = unordered & predicted
    false_splits = int((target_positive & ~predicted).sum().item())
    false_collisions = int((target_negative & predicted).sum().item())
    exact = torch.equal(predicted, target)
    active_logits = (
        output.dynamical_logits[0]
        .detach()
        .cpu()
        .masked_select(
            active_pairs.unsqueeze(-1).expand(
                N,
                N,
                output.dynamical_logits.shape[-1],
            )
        )
    )
    minimum_margin = (
        float(active_logits.abs().min().item()) if active_logits.numel() else 0.0
    )
    equivalent_pairs = tuple(
        (left, right)
        for left_index, left in enumerate(packet.records)
        for right_index, right in enumerate(packet.records)
        if bool(predicted[left_index, right_index])
    )
    return HardExampleResult(
        packet_sha256=metadata.packet_sha256,
        orbit_id=metadata.orbit_id,
        variant=metadata.variant,
        family=metadata.family,
        motif=metadata.motif,
        cell=metadata.cell,
        records=packet.records,
        equivalence_valid=True,
        observation_valid=observation_valid,
        descent_valid=descent_valid,
        physical_law_valid=physical_valid,
        projector_symmetric=projector_symmetric,
        projector_idempotent=projector_idempotent,
        projector_row_stochastic=projector_row_stochastic,
        exact_target_relation=exact,
        coarsest_target_relation=exact,
        false_splits=false_splits,
        false_collisions=false_collisions,
        target_positive_pairs=int(target_positive.sum().item()),
        target_negative_pairs=int(target_negative.sum().item()),
        predicted_positive_pairs=int(predicted_positive.sum().item()),
        hard_descent_residual=float(
            output.hard_residuals.descent[0].detach().cpu().item()
        ),
        hard_observation_residual=float(
            output.hard_residuals.observation[0].detach().cpu().item()
        ),
        minimum_absolute_active_bit_margin=minimum_margin,
        equivalent_pairs=equivalent_pairs,
        elapsed_seconds=time.perf_counter() - started,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_results(
    results: Sequence[HardExampleResult],
) -> dict[str, object]:
    examples = len(results)
    physical = [result for result in results if result.physical_law_valid]
    false_splits = sum(result.false_splits for result in results)
    false_collisions = sum(result.false_collisions for result in results)
    target_positive = sum(result.target_positive_pairs for result in results)
    target_negative = sum(result.target_negative_pairs for result in results)

    def rate(attribute: str) -> float | None:
        if not examples:
            return None
        return sum(bool(getattr(result, attribute)) for result in results) / examples

    return {
        "examples": examples,
        "equivalence_valid": sum(result.equivalence_valid for result in results),
        "equivalence_valid_rate": rate("equivalence_valid"),
        "observation_valid": sum(result.observation_valid for result in results),
        "observation_valid_rate": rate("observation_valid"),
        "descent_valid": sum(result.descent_valid for result in results),
        "descent_valid_rate": rate("descent_valid"),
        "physical_law_valid": sum(result.physical_law_valid for result in results),
        "physical_law_valid_rate": rate("physical_law_valid"),
        "projector_symmetric_rate": rate("projector_symmetric"),
        "projector_idempotent_rate": rate("projector_idempotent"),
        "projector_row_stochastic_rate": rate("projector_row_stochastic"),
        "exact_target_relations": sum(
            result.exact_target_relation for result in results
        ),
        "exact_target_relation_rate": rate("exact_target_relation"),
        "coarsest_target_relations": sum(
            result.coarsest_target_relation for result in results
        ),
        "coarsest_target_relation_rate": rate("coarsest_target_relation"),
        "conditional_exact_physically_valid": sum(
            result.exact_target_relation for result in physical
        ),
        "conditional_exact_physically_valid_rate": (
            sum(result.exact_target_relation for result in physical) / len(physical)
            if physical
            else None
        ),
        "false_splits": false_splits,
        "false_split_rate": (
            false_splits / target_positive if target_positive else 0.0
        ),
        "false_collisions": false_collisions,
        "false_collision_rate": (
            false_collisions / target_negative if target_negative else 0.0
        ),
        "target_positive_pairs": target_positive,
        "target_negative_pairs": target_negative,
        "mean_hard_descent_residual": _mean(
            [result.hard_descent_residual for result in results]
        ),
        "mean_hard_observation_residual": _mean(
            [result.hard_observation_residual for result in results]
        ),
        "minimum_absolute_active_bit_margin": (
            min(result.minimum_absolute_active_bit_margin for result in results)
            if results
            else None
        ),
    }


def _grouped_report(
    results: Sequence[HardExampleResult],
    attribute: Literal["family", "motif", "variant", "cell"],
) -> list[dict[str, object]]:
    groups: dict[str, list[HardExampleResult]] = defaultdict(list)
    for result in results:
        groups[str(getattr(result, attribute))].append(result)
    return [
        {attribute: key, **_summarize_results(groups[key])} for key in sorted(groups)
    ]


def evaluate_counterexample_transport(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Run exactly one built-in hard decode for every packet."""

    was_training = model.training
    model.eval()
    started = time.perf_counter()
    results = [
        assess_one_example(model, partition, index, device=device)
        for index in range(len(partition.packets))
    ]
    if was_training:
        model.train()
    return {
        "partition": partition.name,
        "manifest_sha256": partition.manifest_sha256,
        "hard_threshold": HARD_THRESHOLD,
        "hard_decoder": (
            "one threshold over learned bits, then complete signature-row "
            "equality with immutable observation fibers"
        ),
        "decode_count_per_example": 1,
        **_summarize_results(results),
        "families": _grouped_report(results, "family"),
        "motifs": _grouped_report(results, "motif"),
        "variants": _grouped_report(results, "variant"),
        "cells": _grouped_report(results, "cell"),
        "orbit_consistency": audited_harness._orbit_consistency_report(  # noqa: SLF001
            partition,
            results,
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "results": [asdict(result) for result in results],
    }


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_receipts() -> dict[str, str]:
    """Seal every repository module that can affect inputs, targets, or scores."""

    modules = {
        "audited_partition_label_harness": "train_neural_endogenous_congruence",
        "mctfr_model": ("pipeline.neural_endogenous_counterexample_transport"),
        "neural_eccr_encoder": "pipeline.neural_endogenous_congruence",
        "packet_tensorizer": "pipeline.tensorize_endogenous_congruence",
        "packet_mechanics": "pipeline.endogenous_congruence_board",
        "procedural_generator": ("pipeline.generate_endogenous_congruence_corpus"),
    }
    paths = {"trainer": Path(__file__).resolve()}
    for name, module_name in modules.items():
        module_file = inspect.getsourcefile(importlib.import_module(module_name))
        if module_file is None:
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                f"source path is unavailable for {module_name}"
            )
        paths[name] = Path(module_file).resolve()
    return {name: _sha256_file(path) for name, path in paths.items()}


def _parameter_ledger(
    model: NeuralEndogenousCounterexampleTransport,
) -> dict[str, object]:
    count = model.parameter_count()
    summary = {
        **asdict(count),
        "under_cap": count.under_cap,
        "under_system_cap": count.under_system_cap,
    }
    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "parameters": parameter.numel(),
            "trainable": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    ]
    total = sum(int(item["parameters"]) for item in parameters)
    trainable = sum(int(item["parameters"]) for item in parameters if item["trainable"])
    if total != summary["total"] or trainable != summary["trainable"]:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "parameter ledger does not reconcile"
        )
    if summary["protected_base"] != PROTECTED_BASE_PARAMETERS:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "protected base parameter receipt drifted"
        )
    if (
        summary["complete_system"] != PROTECTED_BASE_PARAMETERS + total
        or summary["complete_system"] >= SYSTEM_PARAMETER_CAP
        or not summary["under_system_cap"]
    ):
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "complete system is not strictly below 200M parameters"
        )
    return {
        "summary": summary,
        "parameters": parameters,
        "integration_status": (
            "standalone MCTFR; complete-system count is hypothetical accounting"
        ),
    }


def verify_artifact_bundle(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "bundle manifest is missing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "bundle manifest schema is invalid"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "bundle manifest has no artifacts"
        )
    for name, expected in artifacts.items():
        path = output_dir / name
        if (
            not isinstance(name, str)
            or "/" in name
            or not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or _sha256_file(path) != expected
        ):
            raise EndogenousCounterexampleTransportTrainingHarnessError(
                f"bundle artifact failed verification: {name}"
            )
    return manifest


def _publish_artifact_bundle(
    output_dir: Path,
    checkpoint: dict[str, object],
    report: dict[str, object],
) -> dict[str, object]:
    """Publish checkpoint and report together by one atomic directory rename."""

    if output_dir.exists():
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "output directory already exists; sealed runs are not overwritten"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.part-",
            dir=output_dir.parent,
        )
    )
    try:
        checkpoint_path = staging / "mctfr.pt"
        torch.save(checkpoint, checkpoint_path)
        report["checkpoint"] = {
            "path": str(output_dir / "mctfr.pt"),
            "sha256": _sha256_file(checkpoint_path),
        }
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "artifacts": {
                "mctfr.pt": _sha256_file(checkpoint_path),
                "report.json": _sha256_file(report_path),
            },
        }
        manifest_path = staging / "bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        for path in (checkpoint_path, report_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        staging_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        staging.replace(output_dir)
        parent_descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        for path in sorted(staging.glob("*")):
            if path.is_file():
                path.unlink()
        if staging.exists():
            staging.rmdir()
        raise
    verify_artifact_bundle(output_dir)
    return report


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Generate, train, hard-assess, seal, and atomically publish one run."""

    source_before = _source_receipts()
    protected_base = _protected_base_receipt()
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    partitions = load_procedural_partitions(
        seed=args.data_seed,
        train_packets=args.train_packets,
        development_packets=args.development_packets,
    )
    model_config = NeuralEndogenousCounterexampleTransportConfig(
        hidden_dim=args.hidden_dim,
        dynamical_bits=args.dynamical_bits,
        parameter_cap=args.parameter_cap,
        soft_distance_scale=args.soft_distance_scale,
    )
    model = NeuralEndogenousCounterexampleTransport(model_config).to(device)
    training_config = TrainingConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        active_bit_margin=args.active_bit_margin,
        descent_temperature=args.descent_temperature,
        target_control=args.target_control,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    training_report = train_counterexample_transport(
        model,
        partitions.train,
        config=training_config,
        device=device,
    )
    train_evaluation = evaluate_counterexample_transport(
        model,
        partitions.train,
        device=device,
    )
    development_evaluation = evaluate_counterexample_transport(
        model,
        partitions.development,
        device=device,
    )
    source_after = _source_receipts()
    if source_after != source_before:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "score-bearing source changed during generation, training, or scoring"
        )
    parameter_ledger = _parameter_ledger(model)
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    optimizer_digests = set(training_report["optimizer_packet_digests"])
    if development_digests & optimizer_digests:
        raise EndogenousCounterexampleTransportTrainingHarnessError(
            "development packet entered the optimizer"
        )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": asdict(model_config),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "seed": args.seed,
        "data_seed": args.data_seed,
        "source_payload_sha256": partitions.source_receipt["payload_sha256"],
        "source_sha256": source_before,
        "protected_base": protected_base,
        "parameter_ledger_summary": parameter_ledger["summary"],
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "claim_scope": (
            "sealed exploratory bounded quotient induction on a previously "
            "inspected tuning split; not confirmation, language reasoning, "
            "or a genuine-general-reasoning claim"
        ),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "device": str(device),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "source_receipt": partitions.source_receipt,
        "protected_base": protected_base,
        "source_sha256": {
            "before": source_before,
            "after": source_after,
            "unchanged": True,
        },
        "parameter_ledger": parameter_ledger,
        "training": training_report,
        "train_evaluation": train_evaluation,
        "development_evaluation": development_evaluation,
        "custody_boundary": {
            "model_forward_input": "EndogenousCongruenceTensors only",
            "axis_receipts_in_forward": False,
            "target_relations_in_forward": False,
            "partitions_in_forward": False,
            "orbit_mappings_in_forward": False,
            "family_motif_variant_in_forward": False,
            "assessor_outputs_in_forward": False,
            "label_join": (
                "audited offline final target relations by packet SHA-256, "
                "placed through tensorizer axis receipts"
            ),
            "optimizer_partition": "train only",
            "development_labels_in_optimizer": False,
            "intermediate_oracle_supervision": False,
            "objective": (
                "balanced final fiber/relation + smooth-max physical descent "
                "+ final-relation fixed point + active-bit margin"
            ),
            "hard_selection": (
                "one model-owned threshold at zero followed by complete "
                "signature-row equality"
            ),
            "physical_residuals": (
                "regularized and reported, never used to repair the hard output"
            ),
            "repair_search_retry": False,
        },
    }
    return _publish_artifact_bundle(Path(args.output_dir), checkpoint, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="train/neural_endogenous_counterexample_transport_exploratory",
    )
    parser.add_argument("--seed", type=int, default=2026072305)
    parser.add_argument("--data-seed", type=int, default=2026072306)
    parser.add_argument("--train-packets", type=int, default=256)
    parser.add_argument("--development-packets", type=int, default=64)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--active-bit-margin", type=float, default=1.0)
    parser.add_argument("--descent-temperature", type=float, default=0.1)
    parser.add_argument(
        "--target-control",
        choices=("true", "shuffled"),
        default="true",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dynamical-bits", type=int, default=1)
    parser.add_argument("--soft-distance-scale", type=float, default=1.0)
    parser.add_argument("--parameter-cap", type=int, default=24_000_000)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_training(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_exact_target_relation_rate": report["train_evaluation"][
                    "exact_target_relation_rate"
                ],
                "development_exact_target_relation_rate": report[
                    "development_evaluation"
                ]["exact_target_relation_rate"],
                "development_physical_law_valid_rate": report["development_evaluation"][
                    "physical_law_valid_rate"
                ],
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
