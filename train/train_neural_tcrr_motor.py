#!/usr/bin/env python3
"""Train and assess the bounded one-step neural TCRR motor.

The neural forward boundary receives only source-deleted packet tensors. Exact
legal actions, variable bindings, and successor graphs remain in this offline
trainer/assessor and are never passed to ``NeuralTcrrMotor.forward``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import random
import time
from typing import Literal

import torch
import torch.nn.functional as F
from neural_tcrr_board import (
    ExpectedTransitionRecord,
    SourceDeletedPacket,
    packet_sha256,
)
from neural_tcrr_committer import (
    CHILD_ABSENT,
    CHILD_PRESENT,
    KIND_CONSTRUCTOR,
    KIND_VARIABLE,
    NODE_CLEAR,
    NODE_KEEP,
    NODE_WRITE,
    NeuralTcrrCommitError,
    NeuralTcrrGraphTensors,
    NeuralTcrrGraphTransaction,
    commit_neural_tcrr_graph,
    decode_neural_tcrr_graph_delta,
)
from neural_tcrr_motor import (
    NeuralTcrrGraphDelta,
    NeuralTcrrMotor,
    NeuralTcrrMotorConfig,
    NeuralTcrrMotorOutput,
)
from tensorize_neural_tcrr_packets import (
    A,
    C,
    D,
    GRAPH_KIND_COUNT,
    N,
    V,
    Y,
    NeuralTcrrPacketTensors,
    TensorizedNeuralTcrrPackets,
    tensorize_neural_tcrr_packets,
)
from tensorize_neural_tcrr_training import (
    PATH_STOP,
    PATH_WIDTH,
    NeuralTcrrTrainingTensors,
    TensorizedNeuralTcrrTraining,
    tensorize_neural_tcrr_training,
)
from torch import Tensor, nn


REPORT_SCHEMA = "neural_tcrr_one_step_motor_v1"
CHECKPOINT_SCHEMA = "neural_tcrr_one_step_motor_checkpoint_v1"
REDEX_PRESENT = 0
NO_REDEX = 1
CONTINUE = 0
HALT = 1
_GRAPH_FIELDS = (
    "active",
    "root",
    "kind",
    "type",
    "constructor",
    "variable",
    "children",
    "child_type",
    "child_mask",
    "capacity",
)


class NeuralTcrrTrainingHarnessError(ValueError):
    """Raised when a training or assessment custody invariant fails."""


@dataclass(frozen=True)
class ExampleCell:
    """Offline reporting metadata excluded from model-visible tensors."""

    packet_sha256: str
    family: str
    depth: int
    renderer: str
    composition: str


@dataclass(frozen=True)
class OfflinePartition:
    """Physically explicit train or development packet/label partition."""

    name: Literal["train", "development"]
    packets: tuple[SourceDeletedPacket, ...]
    records: tuple[ExpectedTransitionRecord, ...]
    metadata: tuple[ExampleCell, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.packets:
            raise NeuralTcrrTrainingHarnessError(f"{self.name} partition is empty")
        if not (len(self.packets) == len(self.records) == len(self.metadata)):
            raise NeuralTcrrTrainingHarnessError(
                f"{self.name} packet/label/metadata cardinality differs"
            )
        packet_digests = tuple(packet_sha256(packet) for packet in self.packets)
        record_digests = tuple(record.packet_sha256 for record in self.records)
        metadata_digests = tuple(item.packet_sha256 for item in self.metadata)
        if len(packet_digests) != len(set(packet_digests)):
            raise NeuralTcrrTrainingHarnessError(
                f"{self.name} packet digest is duplicated"
            )
        if packet_digests != record_digests or packet_digests != metadata_digests:
            raise NeuralTcrrTrainingHarnessError(
                f"{self.name} ledgers are not digest aligned"
            )


@dataclass(frozen=True)
class ProceduralPartitions:
    """Train/development split plus source custody receipt."""

    train: OfflinePartition
    development: OfflinePartition
    source_receipt: dict[str, object]


@dataclass(frozen=True)
class TrainingConfig:
    """Modest one-step optimization defaults, configurable for H100 runs."""

    updates: int = 400
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 2026072301
    log_interval: int = 25

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise NeuralTcrrTrainingHarnessError("updates must be positive")
        if self.batch_size <= 0:
            raise NeuralTcrrTrainingHarnessError("batch size must be positive")
        if self.learning_rate <= 0:
            raise NeuralTcrrTrainingHarnessError("learning rate must be positive")
        if self.weight_decay < 0:
            raise NeuralTcrrTrainingHarnessError("weight decay must be nonnegative")
        if self.gradient_clip <= 0:
            raise NeuralTcrrTrainingHarnessError("gradient clip must be positive")
        if self.log_interval <= 0:
            raise NeuralTcrrTrainingHarnessError("log interval must be positive")


@dataclass(frozen=True)
class OfflineBatch:
    """Packet tensors and separately held offline action tensors."""

    packet_tensorization: TensorizedNeuralTcrrPackets
    training_tensorization: TensorizedNeuralTcrrTraining
    packet_digests: tuple[str, ...]


@dataclass(frozen=True)
class GraphTransactionTarget:
    """One offline exact transaction in categorical index form."""

    node_operation: Tensor
    root_pointer: int
    node_kind: Tensor
    node_type_pointer: Tensor
    node_constructor_pointer: Tensor
    node_variable_pointer: Tensor
    child_pointer: Tensor
    child_presence: Tensor


@dataclass(frozen=True)
class CompleteSetLoss:
    """Differentiable legal-set loss and detached audit metrics."""

    loss: Tensor
    per_example_nll: Tensor
    mean_log_legal_mass: Tensor
    redex_examples: int
    no_redex_examples: int
    complete_legal_actions: int


@dataclass(frozen=True)
class HardExampleResult:
    """One single-shot decode, commit, and complete-set assessment."""

    packet_sha256: str
    exact: bool
    no_redex_target: bool
    predicted_no_redex: bool
    predicted_halt: bool
    commit_valid: bool
    invalid_reason: str | None
    matched_action_index: int | None
    predicted_rule: int | None
    predicted_path: tuple[int, ...]
    predicted_bindings: tuple[tuple[int, int], ...]
    elapsed_seconds: float


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata_value(value: object, name: str, fallback: object) -> object:
    return getattr(value, name, fallback)


def _normalize_metadata(value: object) -> ExampleCell:
    digest = str(getattr(value, "packet_sha256"))
    family = str(_metadata_value(value, "family", "unknown"))
    depth = int(
        _metadata_value(
            value,
            "depth",
            _metadata_value(value, "max_occurrence_depth", 0),
        )
    )
    renderer = str(_metadata_value(value, "renderer", "unknown"))
    if hasattr(value, "composition"):
        composition = str(getattr(value, "composition"))
    else:
        lane = str(_metadata_value(value, "grammar_lane", "unknown"))
        behavior = str(_metadata_value(value, "behavior", "unknown"))
        composition = f"{lane}:{behavior}"
    return ExampleCell(
        packet_sha256=digest,
        family=family,
        depth=depth,
        renderer=renderer,
        composition=composition,
    )


def _is_training_partition(value: object) -> bool:
    return str(value) in {"train", "local_transition_train"}


def _partition_generated_dataset(
    generated: object,
    *,
    source: str,
) -> ProceduralPartitions:
    packets = tuple(getattr(generated, "packets"))
    records = tuple(getattr(generated, "expected_records"))
    raw_metadata = tuple(getattr(generated, "metadata"))
    if not (len(packets) == len(records) == len(raw_metadata)):
        raise NeuralTcrrTrainingHarnessError(
            "procedural generator ledgers have different cardinality"
        )
    normalized = tuple(_normalize_metadata(value) for value in raw_metadata)
    train_indices = tuple(
        index
        for index, value in enumerate(raw_metadata)
        if _is_training_partition(getattr(value, "partition"))
    )
    development_indices = tuple(
        index for index in range(len(packets)) if index not in set(train_indices)
    )
    if not train_indices or not development_indices:
        raise NeuralTcrrTrainingHarnessError(
            "procedural generator did not produce both partitions"
        )
    payload_sha256 = str(getattr(getattr(generated, "manifest"), "payload_sha256"))

    def build(
        name: Literal["train", "development"],
        indices: tuple[int, ...],
    ) -> OfflinePartition:
        subset_packets = tuple(packets[index] for index in indices)
        subset_records = tuple(records[index] for index in indices)
        subset_metadata = tuple(normalized[index] for index in indices)
        manifest = _sha256_json(
            {
                "source_payload_sha256": payload_sha256,
                "partition": name,
                "packet_sha256": [packet_sha256(packet) for packet in subset_packets],
            }
        )
        return OfflinePartition(
            name=name,
            packets=subset_packets,
            records=subset_records,
            metadata=subset_metadata,
            manifest_sha256=manifest,
        )

    train = build("train", train_indices)
    development = build("development", development_indices)
    overlap = {item.packet_sha256 for item in train.metadata} & {
        item.packet_sha256 for item in development.metadata
    }
    if overlap:
        raise NeuralTcrrTrainingHarnessError("procedural split digest overlap")
    return ProceduralPartitions(
        train=train,
        development=development,
        source_receipt={
            "source": source,
            "generator_version": str(
                getattr(getattr(generated, "manifest"), "generator_version")
            ),
            "payload_sha256": payload_sha256,
            "train_examples": len(train.packets),
            "development_examples": len(development.packets),
        },
    )


def load_procedural_partitions(
    *,
    source: Literal["pilot", "corpus"],
    seed: int,
    train_packets: int = 96,
    development_packets: int = 32,
    maximum_attempts: int = 128,
) -> ProceduralPartitions:
    """Generate deterministic in-memory packet and offline label partitions."""

    if source == "pilot":
        module = importlib.import_module("generate_neural_tcrr_board")
        generated = module.generate_neural_tcrr_pilot(
            seed=seed,
            max_attempts=maximum_attempts,
        )
    elif source == "corpus":
        try:
            module = importlib.import_module("generate_neural_tcrr_corpus")
        except ModuleNotFoundError as exc:
            raise NeuralTcrrTrainingHarnessError(
                "procedural corpus API is not available"
            ) from exc
        generated = module.generate_neural_tcrr_corpus(
            seed=seed,
            train_packets=train_packets,
            development_packets=development_packets,
            maximum_attempts=maximum_attempts,
        )
    else:
        raise NeuralTcrrTrainingHarnessError(f"unknown source {source!r}")
    return _partition_generated_dataset(generated, source=source)


def subset_partition(
    partition: OfflinePartition,
    indices: Sequence[int],
) -> OfflinePartition:
    """Build a digest-aligned subset for deterministic smoke or pilot runs."""

    resolved = tuple(int(index) for index in indices)
    if not resolved:
        raise NeuralTcrrTrainingHarnessError("partition subset is empty")
    if any(index < 0 or index >= len(partition.packets) for index in resolved):
        raise NeuralTcrrTrainingHarnessError("partition subset index is out of range")
    packets = tuple(partition.packets[index] for index in resolved)
    records = tuple(partition.records[index] for index in resolved)
    metadata = tuple(partition.metadata[index] for index in resolved)
    return OfflinePartition(
        name=partition.name,
        packets=packets,
        records=records,
        metadata=metadata,
        manifest_sha256=_sha256_json(
            {
                "parent": partition.manifest_sha256,
                "indices": resolved,
                "digests": [packet_sha256(packet) for packet in packets],
            }
        ),
    )


def _prepare_batch(
    partition: OfflinePartition,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> OfflineBatch:
    packets = tuple(partition.packets[index] for index in indices)
    records = tuple(partition.records[index] for index in indices)
    packet_tensorization = tensorize_neural_tcrr_packets(
        packets,
        device=device,
    )
    training_tensorization = tensorize_neural_tcrr_training(
        packets,
        records,
        device=device,
    )
    packet_digests = tuple(
        receipt.packet_digest for receipt in packet_tensorization.receipts
    )
    training_digests = tuple(
        receipt.packet_digest for receipt in training_tensorization.training_receipts
    )
    if packet_digests != training_digests:
        raise NeuralTcrrTrainingHarnessError(
            "packet and offline training tensor receipts differ"
        )
    return OfflineBatch(
        packet_tensorization=packet_tensorization,
        training_tensorization=training_tensorization,
        packet_digests=packet_digests,
    )


def _forward_packet_only(
    model: nn.Module,
    packets: NeuralTcrrPacketTensors,
) -> NeuralTcrrMotorOutput:
    """Cross the model boundary with packet tensors and nothing else."""

    output = model(packets)
    if not isinstance(output, NeuralTcrrMotorOutput):
        raise NeuralTcrrTrainingHarnessError(
            "model did not return NeuralTcrrMotorOutput"
        )
    return output


def _one_hot_index(value: Tensor, *, location: str) -> int:
    positions = torch.nonzero(value, as_tuple=False).flatten()
    if positions.numel() != 1:
        raise NeuralTcrrTrainingHarnessError(f"{location} is not one-hot")
    return int(positions[0].item())


def _successor_field(
    labels: NeuralTcrrTrainingTensors,
    name: str,
    batch_index: int,
    action_index: int,
) -> Tensor:
    return getattr(labels, f"successor_{name}")[batch_index, action_index]


def derive_graph_transaction_target(
    packets: NeuralTcrrPacketTensors,
    labels: NeuralTcrrTrainingTensors | None,
    *,
    batch_index: int,
    action_index: int | None,
) -> GraphTransactionTarget:
    """Derive exact KEEP/WRITE/CLEAR labels from a successor graph offline."""

    device = packets.graph_active.device
    if action_index is None:
        successor = {
            "active": packets.graph_active[batch_index],
            "root": packets.graph_root[batch_index],
            "kind": packets.graph_kind[batch_index],
            "type": packets.graph_type[batch_index],
            "constructor": packets.graph_constructor[batch_index],
            "variable": packets.graph_variable[batch_index],
            "children": packets.graph_children[batch_index],
            "child_type": packets.graph_child_type[batch_index],
            "child_mask": packets.graph_child_mask[batch_index],
            "capacity": packets.graph_capacity[batch_index],
        }
    else:
        if labels is None:
            raise NeuralTcrrTrainingHarnessError(
                "successor labels are required for a legal action"
            )
        successor = {
            name: _successor_field(
                labels,
                name,
                batch_index,
                action_index,
            )
            for name in _GRAPH_FIELDS
        }
    if not torch.equal(
        packets.graph_capacity[batch_index],
        successor["capacity"],
    ):
        raise NeuralTcrrTrainingHarnessError(
            "successor changes the fixed reservoir capacity"
        )

    operation = torch.full(
        (N,),
        NODE_KEEP,
        dtype=torch.long,
        device=device,
    )
    kind = torch.full((N,), -1, dtype=torch.long, device=device)
    type_pointer = torch.full((N,), -1, dtype=torch.long, device=device)
    constructor_pointer = torch.full(
        (N,),
        -1,
        dtype=torch.long,
        device=device,
    )
    variable_pointer = torch.full(
        (N,),
        -1,
        dtype=torch.long,
        device=device,
    )
    child_pointer = torch.full(
        (N, A),
        -1,
        dtype=torch.long,
        device=device,
    )
    child_presence = torch.full(
        (N, A),
        -1,
        dtype=torch.long,
        device=device,
    )
    source_fields = {
        "active": packets.graph_active[batch_index],
        "kind": packets.graph_kind[batch_index],
        "type": packets.graph_type[batch_index],
        "constructor": packets.graph_constructor[batch_index],
        "variable": packets.graph_variable[batch_index],
        "children": packets.graph_children[batch_index],
        "child_type": packets.graph_child_type[batch_index],
        "child_mask": packets.graph_child_mask[batch_index],
    }
    for storage in range(N):
        exact = all(
            torch.equal(source_fields[name][storage], successor[name][storage])
            for name in source_fields
        )
        if exact:
            continue
        if not bool(successor["active"][storage].item()):
            operation[storage] = NODE_CLEAR
            continue
        operation[storage] = NODE_WRITE
        kind[storage] = _one_hot_index(
            successor["kind"][storage],
            location=f"successor storage={storage} kind",
        )
        type_pointer[storage] = _one_hot_index(
            successor["type"][storage],
            location=f"successor storage={storage} type",
        )
        if int(kind[storage].item()) == KIND_CONSTRUCTOR:
            constructor_pointer[storage] = _one_hot_index(
                successor["constructor"][storage],
                location=f"successor storage={storage} constructor",
            )
        elif int(kind[storage].item()) == KIND_VARIABLE:
            variable_pointer[storage] = _one_hot_index(
                successor["variable"][storage],
                location=f"successor storage={storage} variable",
            )
        else:
            raise NeuralTcrrTrainingHarnessError(
                "active successor has an empty graph kind"
            )
        for argument in range(A):
            present = bool(successor["child_mask"][storage, argument].item())
            child_presence[storage, argument] = (
                CHILD_PRESENT if present else CHILD_ABSENT
            )
            if present:
                child_pointer[storage, argument] = _one_hot_index(
                    successor["children"][storage, argument],
                    location=(f"successor storage={storage} argument={argument} child"),
                )
    return GraphTransactionTarget(
        node_operation=operation,
        root_pointer=_one_hot_index(successor["root"], location="successor root"),
        node_kind=kind,
        node_type_pointer=type_pointer,
        node_constructor_pointer=constructor_pointer,
        node_variable_pointer=variable_pointer,
        child_pointer=child_pointer,
        child_presence=child_presence,
    )


def graph_transaction_from_target(
    target: GraphTransactionTarget,
    *,
    device: torch.device,
) -> NeuralTcrrGraphTransaction:
    """Materialize one offline target as a committer transaction."""

    operation = F.one_hot(target.node_operation, num_classes=3).bool().unsqueeze(0)
    root = (
        F.one_hot(
            torch.tensor(target.root_pointer, device=device),
            num_classes=N + 1,
        )
        .bool()
        .unsqueeze(0)
    )
    kind = torch.zeros((1, N, GRAPH_KIND_COUNT), dtype=torch.bool, device=device)
    type_pointer = torch.zeros((1, N, Y), dtype=torch.bool, device=device)
    constructor = torch.zeros((1, N, C), dtype=torch.bool, device=device)
    variable = torch.zeros((1, N, V), dtype=torch.bool, device=device)
    child = torch.zeros((1, N, A, N + 1), dtype=torch.bool, device=device)
    presence = torch.zeros((1, N, A, 2), dtype=torch.bool, device=device)
    for storage in range(N):
        if int(target.node_operation[storage].item()) != NODE_WRITE:
            continue
        kind[0, storage, int(target.node_kind[storage].item())] = True
        type_pointer[
            0,
            storage,
            int(target.node_type_pointer[storage].item()),
        ] = True
        constructor_index = int(target.node_constructor_pointer[storage].item())
        variable_index = int(target.node_variable_pointer[storage].item())
        if constructor_index >= 0:
            constructor[0, storage, constructor_index] = True
        if variable_index >= 0:
            variable[0, storage, variable_index] = True
        for argument in range(A):
            presence_index = int(target.child_presence[storage, argument].item())
            presence[0, storage, argument, presence_index] = True
            pointer = int(target.child_pointer[storage, argument].item())
            child[0, storage, argument, pointer if pointer >= 0 else N] = True
    return NeuralTcrrGraphTransaction(
        node_operation=operation,
        root_pointer=root,
        node_kind=kind,
        node_type_pointer=type_pointer,
        node_constructor_pointer=constructor,
        node_variable_pointer=variable,
        child_pointer=child,
        child_presence=presence,
    )


def _masked_log_probability(
    logits: Tensor,
    mask: Tensor,
    target: int,
    *,
    location: str,
) -> Tensor:
    if target < 0 or target >= logits.shape[-1]:
        raise NeuralTcrrTrainingHarnessError(f"{location} target is out of range")
    if not bool(mask[target].item()):
        raise NeuralTcrrTrainingHarnessError(f"{location} target is masked")
    masked = logits.masked_fill(~mask, -torch.inf)
    return F.log_softmax(masked, dim=-1)[target]


def _plain_log_probability(logits: Tensor, target: int) -> Tensor:
    return F.log_softmax(logits, dim=-1)[target]


def _graph_transaction_log_probability(
    delta: NeuralTcrrGraphDelta,
    target: GraphTransactionTarget,
    *,
    batch_index: int,
) -> Tensor:
    score = _masked_log_probability(
        delta.root_pointer_logits[batch_index],
        delta.root_pointer_mask[batch_index],
        target.root_pointer,
        location="root",
    )
    for storage in range(N):
        operation = int(target.node_operation[storage].item())
        score = score + _masked_log_probability(
            delta.node_operation_logits[batch_index, storage],
            delta.node_operation_mask[batch_index, storage],
            operation,
            location=f"storage={storage} operation",
        )
        if operation != NODE_WRITE:
            continue
        kind = int(target.node_kind[storage].item())
        score = score + _masked_log_probability(
            delta.node_kind_logits[batch_index, storage],
            delta.node_kind_mask[batch_index, storage],
            kind,
            location=f"storage={storage} kind",
        )
        score = score + _masked_log_probability(
            delta.node_type_pointer_logits[batch_index, storage],
            delta.node_type_pointer_mask[batch_index, storage],
            int(target.node_type_pointer[storage].item()),
            location=f"storage={storage} type",
        )
        if kind == KIND_CONSTRUCTOR:
            score = score + _masked_log_probability(
                delta.node_constructor_pointer_logits[batch_index, storage],
                delta.node_constructor_pointer_mask[batch_index, storage],
                int(target.node_constructor_pointer[storage].item()),
                location=f"storage={storage} constructor",
            )
        elif kind == KIND_VARIABLE:
            score = score + _masked_log_probability(
                delta.node_variable_pointer_logits[batch_index, storage],
                delta.node_variable_pointer_mask[batch_index, storage],
                int(target.node_variable_pointer[storage].item()),
                location=f"storage={storage} variable",
            )
        for argument in range(A):
            presence = int(target.child_presence[storage, argument].item())
            score = score + _masked_log_probability(
                delta.child_presence_logits[batch_index, storage, argument],
                delta.child_presence_mask[batch_index, storage, argument],
                presence,
                location=f"storage={storage} argument={argument} presence",
            )
            if presence == CHILD_PRESENT:
                score = score + _masked_log_probability(
                    delta.child_pointer_logits[batch_index, storage, argument],
                    delta.child_pointer_mask[batch_index, storage, argument],
                    int(target.child_pointer[storage, argument].item()),
                    location=f"storage={storage} argument={argument} pointer",
                )
    return score


def _path_log_probability(
    output: NeuralTcrrMotorOutput,
    labels: NeuralTcrrTrainingTensors,
    *,
    batch_index: int,
    action_index: int,
) -> Tensor:
    score = output.path_logits.new_zeros(())
    stopped = False
    for depth in range(PATH_WIDTH):
        if not bool(labels.path_token_mask[batch_index, action_index, depth]):
            break
        target = int(labels.path_tokens[batch_index, action_index, depth].item())
        score = score + _masked_log_probability(
            output.path_logits[batch_index, depth],
            output.path_mask[batch_index, depth],
            target,
            location=f"path depth={depth}",
        )
        if target == PATH_STOP:
            stopped = True
            break
    if not stopped:
        raise NeuralTcrrTrainingHarnessError("legal path lacks terminal STOP")
    return score


def _binding_log_probability(
    output: NeuralTcrrMotorOutput,
    labels: NeuralTcrrTrainingTensors,
    *,
    batch_index: int,
    action_index: int,
    rule_index: int,
) -> Tensor:
    score = output.binding_logits.new_zeros(())
    target_mask = labels.variable_binding_mask[batch_index, action_index]
    model_required = output.binding_mask[
        batch_index,
        rule_index,
        :,
        :N,
    ].any(dim=-1)
    if not torch.equal(target_mask, model_required):
        raise NeuralTcrrTrainingHarnessError(
            "offline required bindings differ from packet-derived motor mask"
        )
    for variable in torch.nonzero(target_mask, as_tuple=False).flatten().tolist():
        target = _one_hot_index(
            labels.variable_binding[
                batch_index,
                action_index,
                variable,
            ],
            location=f"binding variable={variable}",
        )
        score = score + _masked_log_probability(
            output.binding_logits[batch_index, rule_index, variable],
            output.binding_mask[batch_index, rule_index, variable],
            target,
            location=f"binding variable={variable}",
        )
    return score


def complete_legal_set_loss(
    output: NeuralTcrrMotorOutput,
    packets: NeuralTcrrPacketTensors,
    offline: NeuralTcrrTrainingTensors,
) -> CompleteSetLoss:
    """Negative log probability mass over all complete legal actions."""

    batch_size = packets.graph_active.shape[0]
    if output.no_redex_logits.shape != (batch_size, 2):
        raise NeuralTcrrTrainingHarnessError("no-redex output shape differs")
    per_example = []
    redex_examples = 0
    no_redex_examples = 0
    complete_actions = 0
    for batch_index in range(batch_size):
        action_indices = (
            torch.nonzero(offline.action_mask[batch_index], as_tuple=False)
            .flatten()
            .tolist()
        )
        if not action_indices:
            no_redex_examples += 1
            identity = derive_graph_transaction_target(
                packets,
                None,
                batch_index=batch_index,
                action_index=None,
            )
            log_mass = (
                _plain_log_probability(
                    output.no_redex_logits[batch_index],
                    NO_REDEX,
                )
                + _plain_log_probability(output.halt_logits[batch_index], HALT)
                + _graph_transaction_log_probability(
                    output.graph_delta,
                    identity,
                    batch_index=batch_index,
                )
            )
        else:
            redex_examples += 1
            complete_actions += len(action_indices)
            status = _plain_log_probability(
                output.no_redex_logits[batch_index],
                REDEX_PRESENT,
            ) + _plain_log_probability(
                output.halt_logits[batch_index],
                CONTINUE,
            )
            action_scores = []
            for action_index in action_indices:
                rule_index = _one_hot_index(
                    offline.rule_pointer[batch_index, action_index],
                    location=f"action={action_index} rule",
                )
                target = derive_graph_transaction_target(
                    packets,
                    offline,
                    batch_index=batch_index,
                    action_index=action_index,
                )
                action_scores.append(
                    status
                    + _masked_log_probability(
                        output.rule_logits[batch_index],
                        output.rule_mask[batch_index],
                        rule_index,
                        location=f"action={action_index} rule",
                    )
                    + _path_log_probability(
                        output,
                        offline,
                        batch_index=batch_index,
                        action_index=action_index,
                    )
                    + _binding_log_probability(
                        output,
                        offline,
                        batch_index=batch_index,
                        action_index=action_index,
                        rule_index=rule_index,
                    )
                    + _graph_transaction_log_probability(
                        output.graph_delta,
                        target,
                        batch_index=batch_index,
                    )
                )
            log_mass = torch.logsumexp(torch.stack(action_scores), dim=0)
        if not bool(torch.isfinite(log_mass).item()):
            raise NeuralTcrrTrainingHarnessError(
                f"non-finite legal mass at batch index {batch_index}"
            )
        per_example.append(-log_mass)
    nll = torch.stack(per_example)
    return CompleteSetLoss(
        loss=nll.mean(),
        per_example_nll=nll,
        mean_log_legal_mass=-nll.mean().detach(),
        redex_examples=redex_examples,
        no_redex_examples=no_redex_examples,
        complete_legal_actions=complete_actions,
    )


def _decode_path(output: NeuralTcrrMotorOutput) -> tuple[int, ...]:
    path = []
    for depth in range(PATH_WIDTH):
        logits = output.path_logits[0, depth]
        mask = output.path_mask[0, depth]
        target = int(logits.masked_fill(~mask, -torch.inf).argmax().item())
        if target == PATH_STOP:
            return tuple(path)
        path.append(target)
    raise NeuralTcrrTrainingHarnessError("hard path failed to emit STOP")


def _decode_bindings(
    output: NeuralTcrrMotorOutput,
    *,
    rule_index: int,
) -> tuple[tuple[int, int], ...]:
    bindings = []
    for variable in range(V):
        mask = output.binding_mask[0, rule_index, variable]
        logits = output.binding_logits[0, rule_index, variable]
        selected = int(logits.masked_fill(~mask, -torch.inf).argmax().item())
        if selected < N:
            bindings.append((variable, selected))
    return tuple(bindings)


def _graph_mapping(value: NeuralTcrrGraphTensors) -> dict[str, Tensor]:
    return {
        "active": value.graph_active,
        "root": value.graph_root,
        "kind": value.graph_kind,
        "type": value.graph_type,
        "constructor": value.graph_constructor,
        "variable": value.graph_variable,
        "children": value.graph_children,
        "child_type": value.graph_child_type,
        "child_mask": value.graph_child_mask,
        "capacity": value.graph_capacity,
    }


def _committed_matches_successor(
    committed: NeuralTcrrGraphTensors,
    labels: NeuralTcrrTrainingTensors,
    *,
    action_index: int,
) -> bool:
    output = _graph_mapping(committed)
    return all(
        torch.equal(
            output[name][0],
            _successor_field(labels, name, 0, action_index),
        )
        for name in _GRAPH_FIELDS
    )


def _committed_matches_input(
    committed: NeuralTcrrGraphTensors,
    packets: NeuralTcrrPacketTensors,
) -> bool:
    output = _graph_mapping(committed)
    packet_fields = {
        "active": packets.graph_active,
        "root": packets.graph_root,
        "kind": packets.graph_kind,
        "type": packets.graph_type,
        "constructor": packets.graph_constructor,
        "variable": packets.graph_variable,
        "children": packets.graph_children,
        "child_type": packets.graph_child_type,
        "child_mask": packets.graph_child_mask,
        "capacity": packets.graph_capacity,
    }
    return all(
        torch.equal(output[name][0], packet_fields[name][0]) for name in _GRAPH_FIELDS
    )


def _label_path(
    labels: NeuralTcrrTrainingTensors,
    *,
    action_index: int,
) -> tuple[int, ...]:
    values = []
    for depth in range(PATH_WIDTH):
        if not bool(labels.path_token_mask[0, action_index, depth]):
            break
        token = int(labels.path_tokens[0, action_index, depth].item())
        if token == PATH_STOP:
            return tuple(values)
        values.append(token)
    raise NeuralTcrrTrainingHarnessError("offline action path lacks STOP")


def _label_bindings(
    labels: NeuralTcrrTrainingTensors,
    *,
    action_index: int,
) -> tuple[tuple[int, int], ...]:
    output = []
    for variable in (
        torch.nonzero(
            labels.variable_binding_mask[0, action_index],
            as_tuple=False,
        )
        .flatten()
        .tolist()
    ):
        storage = _one_hot_index(
            labels.variable_binding[0, action_index, variable],
            location=f"action={action_index} variable={variable}",
        )
        output.append((variable, storage))
    return tuple(output)


def _path_terminal_storage(
    packets: NeuralTcrrPacketTensors,
    path: tuple[int, ...],
) -> int | None:
    root = _one_hot_index(packets.graph_root[0], location="input root")
    if root == N:
        return None
    storage = root
    for argument in path:
        if argument < 0 or argument >= A:
            return None
        pointer = _one_hot_index(
            packets.graph_children[0, storage, argument],
            location=f"path storage={storage} argument={argument}",
        )
        if pointer == N:
            return None
        storage = pointer
    return storage


def assess_one_example(
    model: nn.Module,
    packet: SourceDeletedPacket,
    record: ExpectedTransitionRecord,
    *,
    device: torch.device,
) -> HardExampleResult:
    """Hard-decode once, commit once, and assess against the complete set."""

    started = time.perf_counter()
    if packet_sha256(packet) != record.packet_sha256:
        raise NeuralTcrrTrainingHarnessError("assessment packet/label digest differs")
    packet_tensorization = tensorize_neural_tcrr_packets((packet,), device=device)
    packets = packet_tensorization.tensors
    with torch.inference_mode():
        output = _forward_packet_only(model, packets)

    predicted_no_redex = int(output.no_redex_logits[0].argmax().item()) == NO_REDEX
    predicted_halt = int(output.halt_logits[0].argmax().item()) == HALT
    rule_mask = output.rule_mask[0]
    predicted_rule = int(
        output.rule_logits[0].masked_fill(~rule_mask, -torch.inf).argmax().item()
    )
    invalid_reason = None
    try:
        predicted_path = _decode_path(output)
        predicted_bindings = _decode_bindings(
            output,
            rule_index=predicted_rule,
        )
    except NeuralTcrrTrainingHarnessError as exc:
        predicted_path = ()
        predicted_bindings = ()
        invalid_reason = str(exc)

    committed = None
    try:
        transaction = decode_neural_tcrr_graph_delta(
            packets,
            output.graph_delta,
        )
        committed = commit_neural_tcrr_graph(packets, transaction)
    except NeuralTcrrCommitError as exc:
        invalid_reason = exc.reason_code

    offline = tensorize_neural_tcrr_training(
        (packet,),
        (record,),
        device=device,
    ).tensors
    action_indices = (
        torch.nonzero(offline.action_mask[0], as_tuple=False).flatten().tolist()
    )
    matched = None
    if not action_indices:
        exact = (
            predicted_no_redex
            and predicted_halt
            and invalid_reason is None
            and committed is not None
            and _committed_matches_input(committed, packets)
        )
    else:
        exact = False
        if (
            not predicted_no_redex
            and not predicted_halt
            and invalid_reason is None
            and committed is not None
        ):
            terminal_storage = _path_terminal_storage(packets, predicted_path)
            for action_index in action_indices:
                target_storage = _one_hot_index(
                    offline.target_storage_pointer[0, action_index],
                    location=f"action={action_index} target storage",
                )
                if (
                    predicted_rule
                    == _one_hot_index(
                        offline.rule_pointer[0, action_index],
                        location=f"action={action_index} rule",
                    )
                    and predicted_path
                    == _label_path(
                        offline,
                        action_index=action_index,
                    )
                    and terminal_storage == target_storage
                    and predicted_bindings
                    == _label_bindings(offline, action_index=action_index)
                    and _committed_matches_successor(
                        committed,
                        offline,
                        action_index=action_index,
                    )
                ):
                    matched = action_index
                    exact = True
                    break
    return HardExampleResult(
        packet_sha256=record.packet_sha256,
        exact=exact,
        no_redex_target=not action_indices,
        predicted_no_redex=predicted_no_redex,
        predicted_halt=predicted_halt,
        commit_valid=committed is not None,
        invalid_reason=invalid_reason,
        matched_action_index=matched,
        predicted_rule=predicted_rule,
        predicted_path=predicted_path,
        predicted_bindings=predicted_bindings,
        elapsed_seconds=time.perf_counter() - started,
    )


def _cell_report(
    results: Sequence[HardExampleResult],
    metadata: Sequence[ExampleCell],
) -> list[dict[str, object]]:
    cells: dict[
        tuple[str, int, str, str],
        dict[str, object],
    ] = {}
    for result, item in zip(results, metadata, strict=True):
        key = (item.family, item.depth, item.renderer, item.composition)
        cell = cells.setdefault(
            key,
            {
                "family": item.family,
                "depth": item.depth,
                "renderer": item.renderer,
                "composition": item.composition,
                "examples": 0,
                "exact": 0,
                "no_redex_examples": 0,
                "invalid_commits": 0,
            },
        )
        cell["examples"] = int(cell["examples"]) + 1
        cell["exact"] = int(cell["exact"]) + int(result.exact)
        cell["no_redex_examples"] = int(cell["no_redex_examples"]) + int(
            result.no_redex_target
        )
        cell["invalid_commits"] = int(cell["invalid_commits"]) + int(
            not result.commit_valid
        )
    output = []
    for key in sorted(cells):
        cell = cells[key]
        cell["exact_rate"] = int(cell["exact"]) / int(cell["examples"])
        output.append(cell)
    return output


def evaluate_motor(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Run one-shot hard assessment over an offline partition."""

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    results = [
        assess_one_example(
            model,
            packet,
            record,
            device=device,
        )
        for packet, record in zip(
            partition.packets,
            partition.records,
            strict=True,
        )
    ]
    if was_training:
        model.train()
    invalid = Counter(
        result.invalid_reason for result in results if result.invalid_reason is not None
    )
    exact = sum(result.exact for result in results)
    no_redex = [result for result in results if result.no_redex_target]
    redex = [result for result in results if not result.no_redex_target]
    return {
        "partition": partition.name,
        "manifest_sha256": partition.manifest_sha256,
        "examples": len(results),
        "exact": exact,
        "exact_rate": exact / len(results),
        "redex_examples": len(redex),
        "redex_exact": sum(result.exact for result in redex),
        "redex_exact_rate": (
            sum(result.exact for result in redex) / len(redex) if redex else None
        ),
        "no_redex_examples": len(no_redex),
        "no_redex_exact": sum(result.exact for result in no_redex),
        "no_redex_exact_rate": (
            sum(result.exact for result in no_redex) / len(no_redex)
            if no_redex
            else None
        ),
        "valid_commits": sum(result.commit_valid for result in results),
        "invalid_commit_reasons": dict(sorted(invalid.items())),
        "cells": _cell_report(results, partition.metadata),
        "elapsed_seconds": time.perf_counter() - started,
        "results": [asdict(result) for result in results],
    }


def _calibration_loss(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    indices: Sequence[int],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    batch = _prepare_batch(partition, indices, device=device)
    with torch.inference_mode():
        output = _forward_packet_only(model, batch.packet_tensorization.tensors)
        loss = complete_legal_set_loss(
            output,
            batch.packet_tensorization.tensors,
            batch.training_tensorization.tensors,
        ).loss
    if was_training:
        model.train()
    return float(loss.item())


def train_motor(
    model: NeuralTcrrMotor,
    training: OfflinePartition,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    """Optimize only a train partition; no development argument is accepted."""

    if training.name != "train":
        raise NeuralTcrrTrainingHarnessError(
            "optimizer accepts only an explicit train partition"
        )
    if config.batch_size > len(training.packets):
        raise NeuralTcrrTrainingHarnessError(
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
    calibration_indices = tuple(range(min(len(training.packets), config.batch_size)))
    initial_loss = _calibration_loss(
        model,
        training,
        indices=calibration_indices,
        device=device,
    )
    started = time.perf_counter()
    trace = []
    used_digests: set[str] = set()
    for update in range(1, config.updates + 1):
        indices = tuple(
            sampler.sample(
                range(len(training.packets)),
                k=config.batch_size,
            )
        )
        batch = _prepare_batch(training, indices, device=device)
        used_digests.update(batch.packet_digests)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_packet_only(
            model,
            batch.packet_tensorization.tensors,
        )
        objective = complete_legal_set_loss(
            output,
            batch.packet_tensorization.tensors,
            batch.training_tensorization.tensors,
        )
        if not bool(torch.isfinite(objective.loss).item()):
            raise NeuralTcrrTrainingHarnessError(f"non-finite loss at update {update}")
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise NeuralTcrrTrainingHarnessError(
                f"non-finite gradient at update {update}"
            )
        optimizer.step()
        if update == 1 or update == config.updates or update % config.log_interval == 0:
            trace.append(
                {
                    "update": update,
                    "loss": float(objective.loss.detach().item()),
                    "mean_log_legal_mass": float(objective.mean_log_legal_mass.item()),
                    "gradient_norm": float(gradient_norm.item()),
                    "redex_examples": objective.redex_examples,
                    "no_redex_examples": objective.no_redex_examples,
                    "complete_legal_actions": objective.complete_legal_actions,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    final_loss = _calibration_loss(
        model,
        training,
        indices=calibration_indices,
        device=device,
    )
    return {
        "examples": len(training.packets),
        "updates": config.updates,
        "batch_size": config.batch_size,
        "optimizer_examples": config.updates * config.batch_size,
        "optimizer_packet_digests": sorted(used_digests),
        "initial_calibration_loss": initial_loss,
        "final_calibration_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_receipts(*, source: Literal["pilot", "corpus"]) -> dict[str, str]:
    generator_module = (
        "generate_neural_tcrr_board"
        if source == "pilot"
        else "generate_neural_tcrr_corpus"
    )
    paths = {
        "trainer": Path(__file__).resolve(),
        "motor": Path(importlib.import_module("neural_tcrr_motor").__file__).resolve(),
        "committer": Path(
            importlib.import_module("neural_tcrr_committer").__file__
        ).resolve(),
        "packet_tensorizer": Path(
            importlib.import_module("tensorize_neural_tcrr_packets").__file__
        ).resolve(),
        "training_tensorizer": Path(
            importlib.import_module("tensorize_neural_tcrr_training").__file__
        ).resolve(),
        "packet_mechanics": Path(
            importlib.import_module("neural_tcrr_board").__file__
        ).resolve(),
        "rewrite_mechanics": Path(
            importlib.import_module("typed_critical_pair_rewrite_board").__file__
        ).resolve(),
        "procedural_generator": Path(
            importlib.import_module(generator_module).__file__
        ).resolve(),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Generate, train, hard-assess, and atomically publish one bounded run."""

    source_sha256 = _source_receipts(source=args.source)
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    partitions = load_procedural_partitions(
        source=args.source,
        seed=args.data_seed,
        train_packets=args.train_packets,
        development_packets=args.development_packets,
        maximum_attempts=args.maximum_attempts,
    )
    model_config = NeuralTcrrMotorConfig(
        hidden_dim=args.hidden_dim,
        entity_rounds=args.entity_rounds,
        term_rounds=args.term_rounds,
        graph_rounds=args.graph_rounds,
        max_arity=A,
        path_depth=D,
    )
    model = NeuralTcrrMotor(model_config).to(device)
    training_config = TrainingConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    training_report = train_motor(
        model,
        partitions.train,
        config=training_config,
        device=device,
    )
    train_evaluation = evaluate_motor(
        model,
        partitions.train,
        device=device,
    )
    development_evaluation = evaluate_motor(
        model,
        partitions.development,
        device=device,
    )
    if _source_receipts(source=args.source) != source_sha256:
        raise NeuralTcrrTrainingHarnessError(
            "score-bearing source changed during generation, training, or evaluation"
        )
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "motor.pt"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": asdict(model_config),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "seed": args.seed,
        "data_seed": args.data_seed,
        "source_payload_sha256": partitions.source_receipt["payload_sha256"],
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    parameter_ledger = asdict(model.parameter_count())
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "device": str(device),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "source_receipt": partitions.source_receipt,
        "parameter_ledger": parameter_ledger,
        "training": training_report,
        "train_evaluation": train_evaluation,
        "development_evaluation": development_evaluation,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "source_sha256": source_sha256,
        "custody_boundary": {
            "model_forward_input": "NeuralTcrrPacketTensors only",
            "labels_location": "offline trainer/assessor only",
            "objective": (
                "negative log probability mass over every complete legal action"
            ),
            "selection": "one hard decode and one atomic commit; no retry or repair",
            "development_labels_in_optimizer": False,
            "successors_in_forward": False,
        },
    }
    _atomic_json_save(report, output_dir / "report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="train/neural_tcrr_motor_pilot")
    parser.add_argument("--source", choices=("pilot", "corpus"), default="pilot")
    parser.add_argument("--seed", type=int, default=2026072301)
    parser.add_argument("--data-seed", type=int, default=2026072302)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--entity-rounds", type=int, default=2)
    parser.add_argument("--term-rounds", type=int, default=2)
    parser.add_argument("--graph-rounds", type=int, default=3)
    parser.add_argument("--train-packets", type=int, default=96)
    parser.add_argument("--development-packets", type=int, default=32)
    parser.add_argument("--maximum-attempts", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_training(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
