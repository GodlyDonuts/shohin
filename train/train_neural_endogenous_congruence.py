#!/usr/bin/env python3
"""Train and assess source-deleted neural endogenous congruence induction.

The model forward boundary receives only ``EndogenousCongruenceTensors``.
Packet identifiers, split metadata, target relations, renderer morphisms, and
all orbit provenance remain in this offline trainer/assessor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import tempfile
import time
from typing import Literal

import torch
import torch.nn.functional as F
from pipeline.endogenous_congruence_board import EndogenousCongruencePacket
from pipeline.generate_endogenous_congruence_corpus import (
    DEVELOPMENT_PARTITION,
    TRAIN_PARTITION,
    CorpusPacketMetadata,
    ProceduralEndogenousCongruenceCorpus,
    RendererLedger,
    TargetRelationLedger,
    generate_endogenous_congruence_corpus,
    packet_sha256,
)
from pipeline.neural_endogenous_congruence import (
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
    NeuralEndogenousCongruence,
    NeuralEndogenousCongruenceConfig,
    NeuralEndogenousCongruenceError,
    NeuralEndogenousCongruenceOutput,
    decode_endogenous_congruence_logits,
)
from pipeline.tensorize_endogenous_congruence import (
    N,
    EndogenousCongruenceTensors,
    TensorizedEndogenousCongruencePackets,
    tensorize_endogenous_congruence_packets,
)
from torch import Tensor, nn


REPORT_SCHEMA = "neural_endogenous_congruence_exploratory_v1"
CHECKPOINT_SCHEMA = "neural_endogenous_congruence_checkpoint_v1"


class EndogenousCongruenceTrainingHarnessError(ValueError):
    """A score-bearing custody, objective, or artifact invariant failed."""


@dataclass(frozen=True)
class OfflineExampleMetadata:
    """Assessor-only split and reporting fields."""

    packet_sha256: str
    partition: str
    orbit_id: str
    variant: str
    family: str
    motif: str
    physical_records: int
    generators: int
    query_ports: int

    @property
    def cell(self) -> str:
        return (
            f"N{self.physical_records}:G{self.generators}:Q{self.query_ports}:"
            f"{self.variant}"
        )


@dataclass(frozen=True)
class OfflinePartition:
    """A digest-aligned physical packet, label, and assessor partition."""

    name: Literal["train", "development"]
    packets: tuple[EndogenousCongruencePacket, ...]
    targets: tuple[TargetRelationLedger, ...]
    metadata: tuple[OfflineExampleMetadata, ...]
    renderers: tuple[RendererLedger, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.packets:
            raise EndogenousCongruenceTrainingHarnessError(
                f"{self.name} partition is empty"
            )
        cardinalities = {
            len(self.packets),
            len(self.targets),
            len(self.metadata),
            len(self.renderers),
        }
        if len(cardinalities) != 1:
            raise EndogenousCongruenceTrainingHarnessError(
                f"{self.name} partition ledgers have different cardinality"
            )
        packet_digests = tuple(packet_sha256(packet) for packet in self.packets)
        if len(packet_digests) != len(set(packet_digests)):
            raise EndogenousCongruenceTrainingHarnessError(
                f"{self.name} packet digest is duplicated"
            )
        ledgers = (
            tuple(item.packet_sha256 for item in self.targets),
            tuple(item.packet_sha256 for item in self.metadata),
            tuple(item.packet_sha256 for item in self.renderers),
        )
        if any(values != packet_digests for values in ledgers):
            raise EndogenousCongruenceTrainingHarnessError(
                f"{self.name} packet and offline ledgers are not digest aligned"
            )
        expected_partition = (
            TRAIN_PARTITION if self.name == "train" else DEVELOPMENT_PARTITION
        )
        if any(item.partition != expected_partition for item in self.metadata):
            raise EndogenousCongruenceTrainingHarnessError(
                f"{self.name} contains metadata from another partition"
            )


@dataclass(frozen=True)
class ProceduralPartitions:
    """Strict metadata-derived train/development split and corpus receipt."""

    train: OfflinePartition
    development: OfflinePartition
    source_receipt: dict[str, object]


@dataclass(frozen=True)
class TrainingConfig:
    """Exploratory optimizer settings."""

    updates: int = 800
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    diagonal_weight: float = 1.0
    descent_weight: float = 0.05
    observation_weight: float = 0.05
    seed: int = 2026072303
    log_interval: int = 25

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise EndogenousCongruenceTrainingHarnessError("updates must be positive")
        if self.batch_size <= 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "batch size must be positive"
            )
        if self.learning_rate <= 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "learning rate must be positive"
            )
        if self.weight_decay < 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "weight decay must be nonnegative"
            )
        if self.gradient_clip <= 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "gradient clip must be positive"
            )
        if self.diagonal_weight <= 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "diagonal weight must be positive"
            )
        if self.descent_weight < 0 or self.observation_weight < 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "residual weights must be nonnegative"
            )
        if self.log_interval <= 0:
            raise EndogenousCongruenceTrainingHarnessError(
                "log interval must be positive"
            )


@dataclass(frozen=True)
class OfflineBatch:
    """Tensor-only model input plus separately held offline supervision."""

    tensorization: TensorizedEndogenousCongruencePackets
    same_class_target: Tensor
    packet_digests: tuple[str, ...]


@dataclass(frozen=True)
class CompleteActivePairLoss:
    """Complete active-pair loss and detached audit statistics."""

    loss: Tensor
    off_diagonal_loss: Tensor
    diagonal_loss: Tensor
    descent_regularization: Tensor
    observation_regularization: Tensor
    active_pairs: int
    diagonal_pairs: int
    positive_off_diagonal_pairs: int
    negative_off_diagonal_pairs: int


@dataclass(frozen=True)
class HardExampleResult:
    """One threshold, one frozen decode, and no repair."""

    packet_sha256: str
    orbit_id: str
    variant: str
    family: str
    motif: str
    cell: str
    records: tuple[str, ...]
    exact_relation: bool
    valid_decode: bool
    coarsest_valid_relation: bool
    coarseness_recall: float | None
    coarseness_precision: float | None
    predicted_off_diagonal_pairs: int | None
    target_off_diagonal_pairs: int
    equivalent_pairs: tuple[tuple[str, str], ...] | None
    invalid_reason: str | None
    elapsed_seconds: float


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_metadata(
    packet: EndogenousCongruencePacket,
    value: CorpusPacketMetadata,
) -> OfflineExampleMetadata:
    return OfflineExampleMetadata(
        packet_sha256=value.packet_sha256,
        partition=value.partition,
        orbit_id=value.orbit_id,
        variant=value.variant,
        family=value.family,
        motif=value.motif,
        physical_records=len(packet.records),
        generators=len(packet.generators),
        query_ports=len(packet.query_ports),
    )


def _partition_generated_corpus(
    corpus: ProceduralEndogenousCongruenceCorpus,
) -> ProceduralPartitions:
    """Partition only by the corpus's assessor-side metadata ledger."""

    packets = corpus.packets
    packet_digests = tuple(packet_sha256(packet) for packet in packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise EndogenousCongruenceTrainingHarnessError(
            "procedural corpus contains duplicate packet digests"
        )
    expected = set(packet_digests)

    def unique_map(values: Sequence[object], name: str) -> dict[str, object]:
        output: dict[str, object] = {}
        for value in values:
            digest = str(getattr(value, "packet_sha256"))
            if digest in output:
                raise EndogenousCongruenceTrainingHarnessError(
                    f"{name} contains duplicate packet digest"
                )
            output[digest] = value
        if set(output) != expected:
            raise EndogenousCongruenceTrainingHarnessError(
                f"{name} does not cover the model packet manifest"
            )
        return output

    metadata_by_digest = unique_map(corpus.metadata, "metadata")
    targets_by_digest = unique_map(corpus.target_relations, "target relations")
    renderers_by_digest = unique_map(corpus.renderers, "renderers")
    allowed_partitions = {TRAIN_PARTITION, DEVELOPMENT_PARTITION}
    if {
        str(getattr(value, "partition")) for value in metadata_by_digest.values()
    } != allowed_partitions:
        raise EndogenousCongruenceTrainingHarnessError(
            "metadata does not contain exactly the frozen train/development partitions"
        )

    train_indices = tuple(
        index
        for index, digest in enumerate(packet_digests)
        if getattr(metadata_by_digest[digest], "partition") == TRAIN_PARTITION
    )
    development_indices = tuple(
        index
        for index, digest in enumerate(packet_digests)
        if getattr(metadata_by_digest[digest], "partition") == DEVELOPMENT_PARTITION
    )
    if not train_indices or not development_indices:
        raise EndogenousCongruenceTrainingHarnessError(
            "metadata-derived train/development partition is empty"
        )

    def build(
        name: Literal["train", "development"],
        indices: tuple[int, ...],
    ) -> OfflinePartition:
        subset_packets = tuple(packets[index] for index in indices)
        subset_digests = tuple(packet_digests[index] for index in indices)
        subset_metadata = tuple(
            _normalize_metadata(
                packets[index],
                metadata_by_digest[packet_digests[index]],  # type: ignore[arg-type]
            )
            for index in indices
        )
        return OfflinePartition(
            name=name,
            packets=subset_packets,
            targets=tuple(
                targets_by_digest[digest]
                for digest in subset_digests  # type: ignore[misc]
            ),
            metadata=subset_metadata,
            renderers=tuple(
                renderers_by_digest[digest]
                for digest in subset_digests  # type: ignore[misc]
            ),
            manifest_sha256=_sha256_json(
                {
                    "corpus_payload_sha256": corpus.manifest.payload_sha256,
                    "partition": name,
                    "packet_sha256": subset_digests,
                }
            ),
        )

    train = build("train", train_indices)
    development = build("development", development_indices)
    train_digests = {item.packet_sha256 for item in train.metadata}
    development_digests = {item.packet_sha256 for item in development.metadata}
    if train_digests & development_digests:
        raise EndogenousCongruenceTrainingHarnessError(
            "metadata-derived partitions overlap"
        )
    isolation = corpus.manifest.split_isolation
    if any(
        (
            isolation.exact_packet_overlap,
            isolation.latent_signature_overlap,
            isolation.action_signature_overlap,
            isolation.path_signature_overlap,
        )
    ):
        raise EndogenousCongruenceTrainingHarnessError(
            "corpus split-isolation receipt is not clean"
        )
    return ProceduralPartitions(
        train=train,
        development=development,
        source_receipt={
            "generator": "generate_endogenous_congruence_corpus",
            "seed": corpus.manifest.seed,
            "payload_sha256": corpus.manifest.payload_sha256,
            "packet_manifest_sha256": corpus.manifest.packet_manifest_sha256,
            "offline_ledger_sha256": corpus.manifest.offline_ledger_sha256,
            "semantics_sha256": corpus.manifest.semantics_sha256,
            "split_isolation": asdict(corpus.manifest.split_isolation),
            "train_examples": len(train.packets),
            "development_examples": len(development.packets),
            "packets_per_orbit": corpus.manifest.packets_per_orbit,
            "orbit_count": corpus.manifest.orbit_count,
        },
    )


def load_procedural_partitions(
    *,
    seed: int,
    train_packets: int = 256,
    development_packets: int = 64,
) -> ProceduralPartitions:
    """Generate the audited corpus in memory and derive strict partitions."""

    corpus = generate_endogenous_congruence_corpus(
        seed=seed,
        train_packets=train_packets,
        development_packets=development_packets,
    )
    return _partition_generated_corpus(corpus)


def subset_partition(
    partition: OfflinePartition,
    indices: Sequence[int],
) -> OfflinePartition:
    """Create a digest-aligned deterministic subset for tests or pilots."""

    resolved = tuple(int(index) for index in indices)
    if not resolved:
        raise EndogenousCongruenceTrainingHarnessError("partition subset is empty")
    if len(resolved) != len(set(resolved)):
        raise EndogenousCongruenceTrainingHarnessError(
            "partition subset repeats an index"
        )
    if any(index < 0 or index >= len(partition.packets) for index in resolved):
        raise EndogenousCongruenceTrainingHarnessError(
            "partition subset index is out of range"
        )
    return OfflinePartition(
        name=partition.name,
        packets=tuple(partition.packets[index] for index in resolved),
        targets=tuple(partition.targets[index] for index in resolved),
        metadata=tuple(partition.metadata[index] for index in resolved),
        renderers=tuple(partition.renderers[index] for index in resolved),
        manifest_sha256=_sha256_json(
            {
                "parent": partition.manifest_sha256,
                "indices": resolved,
                "packet_sha256": [
                    partition.metadata[index].packet_sha256 for index in resolved
                ],
            }
        ),
    )


def _same_class_labels_from_receipts(
    tensorization: TensorizedEndogenousCongruencePackets,
    packets: Sequence[EndogenousCongruencePacket],
    targets_by_digest: Mapping[str, TargetRelationLedger],
    *,
    device: torch.device,
) -> tuple[Tensor, tuple[str, ...]]:
    """Join labels by packet SHA-256, then place them using axis receipts."""

    if len(packets) != len(tensorization.receipts):
        raise EndogenousCongruenceTrainingHarnessError(
            "packet and tensorizer receipt cardinality differs"
        )
    labels = torch.zeros(
        (len(packets), N, N),
        dtype=torch.bool,
        device=device,
    )
    digests: list[str] = []
    for batch_index, (packet, receipt) in enumerate(
        zip(packets, tensorization.receipts, strict=True)
    ):
        digest = packet_sha256(packet)
        digests.append(digest)
        target = targets_by_digest.get(digest)
        if target is None or target.packet_sha256 != digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "target relation digest join failed"
            )
        if (
            receipt.record_ids != packet.records
            or receipt.generator_ids != packet.generators
            or receipt.query_ids != packet.query_ports
        ):
            raise EndogenousCongruenceTrainingHarnessError(
                "tensorizer axis receipt does not reconstruct packet axes"
            )
        flattened = tuple(record for block in target.blocks for record in block)
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            receipt.record_ids
        ):
            raise EndogenousCongruenceTrainingHarnessError(
                "target blocks do not partition the tensorizer record receipt"
            )
        record_index = {
            record: index for index, record in enumerate(receipt.record_ids)
        }
        for block in target.blocks:
            indices = [record_index[record] for record in block]
            for left in indices:
                for right in indices:
                    labels[batch_index, left, right] = True
        active = tensorization.tensors.record_mask[batch_index]
        pair_mask = active[:, None] & active[None, :]
        if torch.any(labels[batch_index] & ~pair_mask):
            raise EndogenousCongruenceTrainingHarnessError(
                "same-class label enters record padding"
            )
        if not torch.equal(
            labels[batch_index].diagonal(),
            active,
        ):
            raise EndogenousCongruenceTrainingHarnessError(
                "same-class label is not explicitly reflexive"
            )
    return labels, tuple(digests)


def _prepare_batch(
    partition: OfflinePartition,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> OfflineBatch:
    resolved = tuple(int(index) for index in indices)
    packets = tuple(partition.packets[index] for index in resolved)
    targets = tuple(partition.targets[index] for index in resolved)
    tensorization = tensorize_endogenous_congruence_packets(
        packets,
        device=device,
    )
    target_by_digest = {target.packet_sha256: target for target in targets}
    if len(target_by_digest) != len(targets):
        raise EndogenousCongruenceTrainingHarnessError(
            "batch target digest is duplicated"
        )
    labels, digests = _same_class_labels_from_receipts(
        tensorization,
        packets,
        target_by_digest,
        device=device,
    )
    expected = tuple(partition.metadata[index].packet_sha256 for index in resolved)
    if digests != expected:
        raise EndogenousCongruenceTrainingHarnessError(
            "batch packet SHA-256 order differs from metadata"
        )
    return OfflineBatch(
        tensorization=tensorization,
        same_class_target=labels,
        packet_digests=digests,
    )


def _forward_tensor_only(
    model: nn.Module,
    tensors: EndogenousCongruenceTensors,
) -> NeuralEndogenousCongruenceOutput:
    """Cross the neural boundary with source-deleted tensors and nothing else."""

    output = model(tensors)
    if not isinstance(output, NeuralEndogenousCongruenceOutput):
        raise EndogenousCongruenceTrainingHarnessError(
            "model did not return NeuralEndogenousCongruenceOutput"
        )
    return output


def complete_active_pair_loss(
    output: NeuralEndogenousCongruenceOutput,
    tensors: EndogenousCongruenceTensors,
    same_class_target: Tensor,
    *,
    diagonal_weight: float = 1.0,
    descent_weight: float = 0.05,
    observation_weight: float = 0.05,
) -> CompleteActivePairLoss:
    """Supervise every active pair with balanced off-diagonal classes.

    The two residual terms are necessary conditions for a causal congruence:
    related records must remain related after every generator, and they must
    have equal observations at every query port. They are soft regularizers,
    not target construction or hard-decoder repair.
    """

    logits = output.same_class_logits
    expected_shape = tensors.record_mask.shape[0], N, N
    if tuple(logits.shape) != expected_shape:
        raise EndogenousCongruenceTrainingHarnessError(
            "same-class logits have invalid shape"
        )
    if tuple(same_class_target.shape) != expected_shape:
        raise EndogenousCongruenceTrainingHarnessError(
            "same-class target has invalid shape"
        )
    if (
        same_class_target.dtype != torch.bool
        or same_class_target.device != logits.device
    ):
        raise EndogenousCongruenceTrainingHarnessError(
            "same-class target has invalid dtype or device"
        )
    pair_mask = tensors.record_mask[:, :, None] & tensors.record_mask[:, None, :]
    if not torch.equal(output.equivalence_mask, pair_mask):
        raise EndogenousCongruenceTrainingHarnessError(
            "model equivalence mask differs from the active-pair mask"
        )
    identity = tensors.record_equal
    if torch.any(same_class_target & ~pair_mask):
        raise EndogenousCongruenceTrainingHarnessError("target relation enters padding")
    if not torch.equal(same_class_target & identity, identity):
        raise EndogenousCongruenceTrainingHarnessError(
            "target relation is not reflexive on every active record"
        )

    off_diagonal = pair_mask & ~identity
    positive = off_diagonal & same_class_target
    negative = off_diagonal & ~same_class_target
    off_diagonal_terms: list[Tensor] = []
    if torch.any(positive):
        off_diagonal_terms.append(F.softplus(-logits[positive]).mean())
    if torch.any(negative):
        off_diagonal_terms.append(F.softplus(logits[negative]).mean())
    if off_diagonal_terms:
        off_diagonal_loss = torch.stack(off_diagonal_terms).mean()
    else:
        off_diagonal_loss = logits.sum() * 0.0
    diagonal_loss = F.softplus(-logits[identity]).mean()
    descent_regularization = output.residuals.descent.mean()
    observation_regularization = output.residuals.observation.mean()
    loss = (
        off_diagonal_loss
        + diagonal_weight * diagonal_loss
        + descent_weight * descent_regularization
        + observation_weight * observation_regularization
    )
    if not torch.isfinite(loss):
        raise EndogenousCongruenceTrainingHarnessError(
            "complete active-pair loss is non-finite"
        )
    return CompleteActivePairLoss(
        loss=loss,
        off_diagonal_loss=off_diagonal_loss.detach(),
        diagonal_loss=diagonal_loss.detach(),
        descent_regularization=descent_regularization.detach(),
        observation_regularization=observation_regularization.detach(),
        active_pairs=int(pair_mask.sum().item()),
        diagonal_pairs=int(identity.sum().item()),
        positive_off_diagonal_pairs=int(positive.sum().item()),
        negative_off_diagonal_pairs=int(negative.sum().item()),
    )


def _calibration_loss(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    indices: Sequence[int],
    config: TrainingConfig,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    batch = _prepare_batch(partition, indices, device=device)
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        loss = complete_active_pair_loss(
            output,
            batch.tensorization.tensors,
            batch.same_class_target,
            diagonal_weight=config.diagonal_weight,
            descent_weight=config.descent_weight,
            observation_weight=config.observation_weight,
        ).loss
    if was_training:
        model.train()
    return float(loss.item())


def train_inducer(
    model: NeuralEndogenousCongruence,
    training: OfflinePartition,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    """Optimize an explicit train partition; no development input is accepted."""

    if training.name != "train":
        raise EndogenousCongruenceTrainingHarnessError(
            "optimizer accepts only an explicit train partition"
        )
    if config.batch_size > len(training.packets):
        raise EndogenousCongruenceTrainingHarnessError(
            "batch size exceeds the number of unique training packets"
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
    calibration_indices = tuple(range(config.batch_size))
    initial_loss = _calibration_loss(
        model,
        training,
        indices=calibration_indices,
        config=config,
        device=device,
    )
    trace: list[dict[str, object]] = []
    optimizer_digests: set[str] = set()
    started = time.perf_counter()
    for update in range(1, config.updates + 1):
        indices = tuple(
            sampler.sample(range(len(training.packets)), k=config.batch_size)
        )
        batch = _prepare_batch(training, indices, device=device)
        optimizer_digests.update(batch.packet_digests)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        objective = complete_active_pair_loss(
            output,
            batch.tensorization.tensors,
            batch.same_class_target,
            diagonal_weight=config.diagonal_weight,
            descent_weight=config.descent_weight,
            observation_weight=config.observation_weight,
        )
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise EndogenousCongruenceTrainingHarnessError(
                "gradient norm is non-finite"
            )
        optimizer.step()
        if update == 1 or update % config.log_interval == 0 or update == config.updates:
            item = {
                "update": update,
                "loss": float(objective.loss.detach().item()),
                "off_diagonal_loss": float(objective.off_diagonal_loss.item()),
                "diagonal_loss": float(objective.diagonal_loss.item()),
                "descent_regularization": float(
                    objective.descent_regularization.item()
                ),
                "observation_regularization": float(
                    objective.observation_regularization.item()
                ),
                "gradient_norm": float(gradient_norm.detach().item()),
                "active_pairs": objective.active_pairs,
                "positive_off_diagonal_pairs": (objective.positive_off_diagonal_pairs),
                "negative_off_diagonal_pairs": (objective.negative_off_diagonal_pairs),
            }
            trace.append(item)
            print(_canonical_json(item), flush=True)
    final_loss = _calibration_loss(
        model,
        training,
        indices=calibration_indices,
        config=config,
        device=device,
    )
    development_digests: set[str] = set()
    return {
        "initial_calibration_loss": initial_loss,
        "final_calibration_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "optimizer_packet_digests": sorted(optimizer_digests),
        "optimizer_development_packet_digests": sorted(development_digests),
        "unique_optimizer_packets": len(optimizer_digests),
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _target_relation_for_singleton(batch: OfflineBatch) -> Tensor:
    if batch.same_class_target.shape[0] != 1:
        raise EndogenousCongruenceTrainingHarnessError(
            "hard assessment requires a singleton label batch"
        )
    return batch.same_class_target[0].detach().cpu()


def assess_one_example(
    model: nn.Module,
    partition: OfflinePartition,
    index: int,
    *,
    threshold: float,
    device: torch.device,
) -> HardExampleResult:
    """Threshold exactly once through the frozen decoder and never repair."""

    started = time.perf_counter()
    batch = _prepare_batch(partition, (index,), device=device)
    metadata = partition.metadata[index]
    packet = partition.packets[index]
    target = _target_relation_for_singleton(batch)
    target_off_diagonal = target & ~torch.eye(N, dtype=torch.bool)
    target_off_diagonal_pairs = int(target_off_diagonal.sum().item())
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        try:
            decoded = decode_endogenous_congruence_logits(
                batch.tensorization.tensors,
                output.same_class_logits,
                threshold=threshold,
            )
        except NeuralEndogenousCongruenceError as error:
            return HardExampleResult(
                packet_sha256=metadata.packet_sha256,
                orbit_id=metadata.orbit_id,
                variant=metadata.variant,
                family=metadata.family,
                motif=metadata.motif,
                cell=metadata.cell,
                records=packet.records,
                exact_relation=False,
                valid_decode=False,
                coarsest_valid_relation=False,
                coarseness_recall=None,
                coarseness_precision=None,
                predicted_off_diagonal_pairs=None,
                target_off_diagonal_pairs=target_off_diagonal_pairs,
                equivalent_pairs=None,
                invalid_reason=str(error),
                elapsed_seconds=time.perf_counter() - started,
            )
    predicted = decoded.equivalence[0].detach().cpu()
    exact = torch.equal(predicted, target)
    identity = torch.eye(N, dtype=torch.bool)
    predicted_off_diagonal = predicted & ~identity
    predicted_off_diagonal_pairs = int(predicted_off_diagonal.sum().item())
    overlap = int((predicted_off_diagonal & target_off_diagonal).sum().item())
    recall = overlap / target_off_diagonal_pairs if target_off_diagonal_pairs else 1.0
    precision = (
        overlap / predicted_off_diagonal_pairs
        if predicted_off_diagonal_pairs
        else (1.0 if target_off_diagonal_pairs == 0 else 0.0)
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
        exact_relation=exact,
        valid_decode=True,
        coarsest_valid_relation=exact,
        coarseness_recall=recall,
        coarseness_precision=precision,
        predicted_off_diagonal_pairs=predicted_off_diagonal_pairs,
        target_off_diagonal_pairs=target_off_diagonal_pairs,
        equivalent_pairs=equivalent_pairs,
        invalid_reason=None,
        elapsed_seconds=time.perf_counter() - started,
    )


def _mean_optional(values: Sequence[float | None]) -> float | None:
    resolved = [value for value in values if value is not None]
    return sum(resolved) / len(resolved) if resolved else None


def _summarize_results(
    results: Sequence[HardExampleResult],
) -> dict[str, object]:
    invalid = Counter(
        result.invalid_reason for result in results if result.invalid_reason is not None
    )
    return {
        "examples": len(results),
        "exact_relations": sum(result.exact_relation for result in results),
        "exact_relation_rate": (
            sum(result.exact_relation for result in results) / len(results)
            if results
            else None
        ),
        "valid_decodes": sum(result.valid_decode for result in results),
        "valid_decode_rate": (
            sum(result.valid_decode for result in results) / len(results)
            if results
            else None
        ),
        "coarsest_valid_relations": sum(
            result.coarsest_valid_relation for result in results
        ),
        "coarsest_valid_relation_rate": (
            sum(result.coarsest_valid_relation for result in results) / len(results)
            if results
            else None
        ),
        "mean_coarseness_recall": _mean_optional(
            [result.coarseness_recall for result in results]
        ),
        "mean_coarseness_precision": _mean_optional(
            [result.coarseness_precision for result in results]
        ),
        "invalid_reasons": dict(sorted(invalid.items())),
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


def _relation_set(result: HardExampleResult) -> set[tuple[str, str]] | None:
    if not result.valid_decode or result.equivalent_pairs is None:
        return None
    return set(result.equivalent_pairs)


def _mapped_relation_consistency(
    source: HardExampleResult,
    target: HardExampleResult,
    record_map: Mapping[str, str],
) -> bool:
    source_relation = _relation_set(source)
    target_relation = _relation_set(target)
    if source_relation is None or target_relation is None:
        return False
    if set(record_map) != set(source.records):
        raise EndogenousCongruenceTrainingHarnessError(
            "orbit record morphism is not total on the source presentation"
        )
    if not set(record_map.values()) <= set(target.records):
        raise EndogenousCongruenceTrainingHarnessError(
            "orbit record morphism leaves the target presentation"
        )
    return all(
        ((left, right) in source_relation)
        == ((record_map[left], record_map[right]) in target_relation)
        for left in source.records
        for right in source.records
    )


def _orbit_consistency_report(
    partition: OfflinePartition,
    results: Sequence[HardExampleResult],
) -> dict[str, object]:
    result_by_digest = {item.packet_sha256: item for item in results}
    metadata_by_digest = {item.packet_sha256: item for item in partition.metadata}
    renderer_by_digest = {item.packet_sha256: item for item in partition.renderers}
    orbit_variants: dict[str, dict[str, str]] = defaultdict(dict)
    for item in partition.metadata:
        orbit_variants[item.orbit_id][item.variant] = item.packet_sha256
    required = {
        "base",
        "opaque_reindex",
        "value_recode",
        "bisimilar_split",
        "bisimilar_merge",
        "path_collision",
        "path_collision_reindex",
    }
    orbit_rows: list[dict[str, object]] = []
    for orbit_id in sorted(orbit_variants):
        variants = orbit_variants[orbit_id]
        if not required <= set(variants):
            orbit_rows.append(
                {
                    "orbit_id": orbit_id,
                    "complete": False,
                    "reindex_consistent": None,
                    "recoding_consistent": None,
                    "split_consistent": None,
                    "merge_consistent": None,
                    "all_consistent": None,
                }
            )
            continue
        by_variant = {
            variant: result_by_digest[variants[variant]] for variant in required
        }
        base_digest = variants["base"]
        collision_digest = variants["path_collision"]

        reindex_renderer = renderer_by_digest[variants["opaque_reindex"]]
        if reindex_renderer.parent_packet_sha256 != base_digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "base reindex parent receipt is invalid"
            )
        base_reindex = _mapped_relation_consistency(
            by_variant["base"],
            by_variant["opaque_reindex"],
            dict(reindex_renderer.parent_record_map),
        )

        collision_renderer = renderer_by_digest[variants["path_collision_reindex"]]
        if collision_renderer.parent_packet_sha256 != collision_digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "collision reindex parent receipt is invalid"
            )
        collision_reindex = _mapped_relation_consistency(
            by_variant["path_collision"],
            by_variant["path_collision_reindex"],
            dict(collision_renderer.parent_record_map),
        )

        recode_renderer = renderer_by_digest[variants["value_recode"]]
        if recode_renderer.parent_packet_sha256 != base_digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "value-recode parent receipt is invalid"
            )
        recoding = _mapped_relation_consistency(
            by_variant["base"],
            by_variant["value_recode"],
            {record: record for record in by_variant["base"].records},
        )

        split_renderer = renderer_by_digest[variants["bisimilar_split"]]
        if split_renderer.parent_packet_sha256 != base_digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "split parent receipt is invalid"
            )
        split = _mapped_relation_consistency(
            by_variant["bisimilar_split"],
            by_variant["base"],
            dict(split_renderer.parent_record_map),
        )

        merge_renderer = renderer_by_digest[variants["bisimilar_merge"]]
        if merge_renderer.parent_packet_sha256 != base_digest:
            raise EndogenousCongruenceTrainingHarnessError(
                "merge parent receipt is invalid"
            )
        merge = _mapped_relation_consistency(
            by_variant["base"],
            by_variant["bisimilar_merge"],
            dict(merge_renderer.parent_record_map),
        )
        reindex = base_reindex and collision_reindex
        values = (reindex, recoding, split, merge)
        metadata = metadata_by_digest[base_digest]
        orbit_rows.append(
            {
                "orbit_id": orbit_id,
                "partition": metadata.partition,
                "family": metadata.family,
                "motif": metadata.motif,
                "complete": True,
                "base_reindex_consistent": base_reindex,
                "collision_reindex_consistent": collision_reindex,
                "reindex_consistent": reindex,
                "recoding_consistent": recoding,
                "split_consistent": split,
                "merge_consistent": merge,
                "all_consistent": all(values),
            }
        )
    complete = [row for row in orbit_rows if row["complete"]]

    def aggregate(field: str) -> dict[str, object]:
        passed = sum(bool(row[field]) for row in complete)
        return {
            "orbits": len(complete),
            "passed": passed,
            "rate": passed / len(complete) if complete else None,
        }

    return {
        "complete_orbits": len(complete),
        "incomplete_orbits": len(orbit_rows) - len(complete),
        "reindex": aggregate("reindex_consistent"),
        "recoding": aggregate("recoding_consistent"),
        "split": aggregate("split_consistent"),
        "merge": aggregate("merge_consistent"),
        "all": aggregate("all_consistent"),
        "orbits": orbit_rows,
    }


def evaluate_inducer(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    """Run one irreversible hard decode per packet and report crossed cells."""

    if not math.isfinite(threshold):
        raise EndogenousCongruenceTrainingHarnessError("hard threshold must be finite")
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    results = [
        assess_one_example(
            model,
            partition,
            index,
            threshold=threshold,
            device=device,
        )
        for index in range(len(partition.packets))
    ]
    if was_training:
        model.train()
    return {
        "partition": partition.name,
        "manifest_sha256": partition.manifest_sha256,
        "threshold": threshold,
        **_summarize_results(results),
        "families": _grouped_report(results, "family"),
        "motifs": _grouped_report(results, "motif"),
        "variants": _grouped_report(results, "variant"),
        "cells": _grouped_report(results, "cell"),
        "orbit_consistency": _orbit_consistency_report(partition, results),
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
    paths = {
        "trainer": Path(__file__).resolve(),
        "neural_inducer": Path(
            importlib.import_module("pipeline.neural_endogenous_congruence").__file__
        ).resolve(),
        "packet_tensorizer": Path(
            importlib.import_module("pipeline.tensorize_endogenous_congruence").__file__
        ).resolve(),
        "packet_mechanics": Path(
            importlib.import_module("pipeline.endogenous_congruence_board").__file__
        ).resolve(),
        "procedural_generator": Path(
            importlib.import_module(
                "pipeline.generate_endogenous_congruence_corpus"
            ).__file__
        ).resolve(),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _parameter_ledger(
    model: NeuralEndogenousCongruence,
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
        raise EndogenousCongruenceTrainingHarnessError(
            "parameter ledger does not reconcile"
        )
    if summary["protected_base"] != PROTECTED_BASE_PARAMETERS:
        raise EndogenousCongruenceTrainingHarnessError(
            "protected base parameter receipt drifted"
        )
    if (
        summary["complete_system"] != PROTECTED_BASE_PARAMETERS + total
        or summary["complete_system"] >= SYSTEM_PARAMETER_CAP
        or not summary["under_system_cap"]
    ):
        raise EndogenousCongruenceTrainingHarnessError(
            "complete system is not strictly below 200M parameters"
        )
    return {
        "summary": summary,
        "parameters": parameters,
    }


def _publish_artifact_bundle(
    output_dir: Path,
    checkpoint: dict[str, object],
    report: dict[str, object],
) -> dict[str, object]:
    """Publish checkpoint and report as one directory rename."""

    if output_dir.exists():
        raise EndogenousCongruenceTrainingHarnessError(
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
        checkpoint_path = staging / "inducer.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_receipt = {
            "path": str(output_dir / "inducer.pt"),
            "sha256": _sha256_file(checkpoint_path),
        }
        report["checkpoint"] = checkpoint_receipt
        report_path = staging / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        staging.replace(output_dir)
    except Exception:
        for path in sorted(staging.glob("*")):
            if path.is_file():
                path.unlink()
        if staging.exists():
            staging.rmdir()
        raise
    return report


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Generate, train, hard-assess, seal, and atomically publish one run."""

    source_before = _source_receipts()
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    partitions = load_procedural_partitions(
        seed=args.data_seed,
        train_packets=args.train_packets,
        development_packets=args.development_packets,
    )
    model_config = NeuralEndogenousCongruenceConfig(
        hidden_dim=args.hidden_dim,
        rounds=args.rounds,
        parameter_cap=args.parameter_cap,
    )
    model = NeuralEndogenousCongruence(model_config).to(device)
    training_config = TrainingConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        diagonal_weight=args.diagonal_weight,
        descent_weight=args.descent_weight,
        observation_weight=args.observation_weight,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    training_report = train_inducer(
        model,
        partitions.train,
        config=training_config,
        device=device,
    )
    train_evaluation = evaluate_inducer(
        model,
        partitions.train,
        threshold=args.threshold,
        device=device,
    )
    development_evaluation = evaluate_inducer(
        model,
        partitions.development,
        threshold=args.threshold,
        device=device,
    )
    source_after = _source_receipts()
    if source_after != source_before:
        raise EndogenousCongruenceTrainingHarnessError(
            "score-bearing source changed during generation, optimization, or scoring"
        )
    parameter_ledger = _parameter_ledger(model)
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
        "parameter_ledger_summary": parameter_ledger["summary"],
    }
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    optimizer_digests = set(training_report["optimizer_packet_digests"])
    if development_digests & optimizer_digests:
        raise EndogenousCongruenceTrainingHarnessError(
            "development packet entered the optimizer"
        )
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "claim_scope": (
            "sealed exploratory quotient-induction run; not a general-reasoning claim"
        ),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "device": str(device),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "source_receipt": partitions.source_receipt,
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
            "metadata_in_forward": False,
            "label_join": (
                "offline target_relations by packet SHA-256, placed by "
                "tensorizer axis receipts"
            ),
            "partition_source": "offline CorpusPacketMetadata.partition only",
            "objective": (
                "complete active ordered pairs; separately normalized positive "
                "and negative off-diagonals; explicit diagonal BCE"
            ),
            "regularization": (
                "necessary differentiable generator-descent and observation "
                "compatibility residuals only"
            ),
            "development_labels_in_optimizer": False,
            "hard_selection": (
                "one fixed threshold through decode_endogenous_congruence_logits; "
                "no repair, refinement, search, or retry"
            ),
        },
    }
    return _publish_artifact_bundle(Path(args.output_dir), checkpoint, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="train/neural_endogenous_congruence_exploratory",
    )
    parser.add_argument("--seed", type=int, default=2026072303)
    parser.add_argument("--data-seed", type=int, default=2026072304)
    parser.add_argument("--train-packets", type=int, default=256)
    parser.add_argument("--development-packets", type=int, default=64)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--diagonal-weight", type=float, default=1.0)
    parser.add_argument("--descent-weight", type=float, default=0.05)
    parser.add_argument("--observation-weight", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--parameter-cap", type=int, default=8_000_000)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_training(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_exact_relation_rate": report["train_evaluation"][
                    "exact_relation_rate"
                ],
                "development_exact_relation_rate": report["development_evaluation"][
                    "exact_relation_rate"
                ],
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
