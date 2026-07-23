#!/usr/bin/env python3
"""Train and assess guaranteed-equivalence Record-Fiber ECCR induction.

The neural forward boundary receives only ``EndogenousCongruenceTensors``.
Packet identifiers, split metadata, target relations, renderer morphisms, and
all orbit provenance remain in the offline trainer and assessor. The hard
Record-Fiber relation is decoded once inside the model forward and is never
repaired, searched, clustered, retried, or refined.
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
from pathlib import Path
import random
import tempfile
import time
from typing import Literal

import torch
from pipeline.neural_endogenous_congruence import (
    PROTECTED_BASE_PARAMETERS,
    SYSTEM_PARAMETER_CAP,
    NeuralEndogenousCongruenceConfig,
)
from pipeline.neural_endogenous_record_fiber import (
    NeuralEndogenousRecordFiber,
    NeuralEndogenousRecordFiberConfig,
    NeuralEndogenousRecordFiberOutput,
    RecordFiberLoss,
    measure_record_fiber_physical_laws,
    record_fiber_loss,
)
from pipeline.tensorize_endogenous_congruence import (
    N,
    EndogenousCongruenceTensors,
)
from torch import Tensor, nn

import train_neural_endogenous_congruence as audited_harness


REPORT_SCHEMA = "neural_endogenous_record_fiber_exploratory_v1"
CHECKPOINT_SCHEMA = "neural_endogenous_record_fiber_checkpoint_v1"
HARD_THRESHOLD = 0.0

OfflineExampleMetadata = audited_harness.OfflineExampleMetadata
OfflinePartition = audited_harness.OfflinePartition
ProceduralPartitions = audited_harness.ProceduralPartitions
OfflineBatch = audited_harness.OfflineBatch
load_procedural_partitions = audited_harness.load_procedural_partitions
subset_partition = audited_harness.subset_partition


class EndogenousRecordFiberTrainingHarnessError(ValueError):
    """A score-bearing custody, objective, or artifact invariant failed."""


@dataclass(frozen=True)
class TrainingConfig:
    """Exploratory Record-Fiber optimizer settings."""

    updates: int = 800
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    margin: float = 1.0
    seed: int = 2026072305
    log_interval: int = 25

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise EndogenousRecordFiberTrainingHarnessError("updates must be positive")
        if self.batch_size <= 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "batch size must be positive"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "learning rate must be finite and positive"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "weight decay must be finite and nonnegative"
            )
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "gradient clip must be finite and positive"
            )
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "margin must be finite and positive"
            )
        if self.log_interval <= 0:
            raise EndogenousRecordFiberTrainingHarnessError(
                "log interval must be positive"
            )


@dataclass(frozen=True)
class ObjectiveSnapshot:
    """Detached objective components and physical residual measurements."""

    total: float
    code: float
    fiber: float
    distance: float
    margin: float
    soft_descent_residual: float
    soft_observation_residual: float
    hard_descent_residual: float
    hard_observation_residual: float


@dataclass(frozen=True)
class HardExampleResult:
    """One model forward, one built-in hard decode, and no repair."""

    packet_sha256: str
    orbit_id: str
    variant: str
    family: str
    motif: str
    cell: str
    records: tuple[str, ...]
    equivalence_valid: bool
    physical_law_valid: bool
    observation_valid: bool
    descent_valid: bool
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
    minimum_absolute_vote_margin: float
    equivalent_pairs: tuple[tuple[str, str], ...]
    elapsed_seconds: float

    @property
    def valid_decode(self) -> bool:
        """Compatibility with the audited orbit-consistency assessor."""

        return self.equivalence_valid


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _prepare_batch(
    partition: OfflinePartition,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> OfflineBatch:
    """Reuse the audited SHA-256 and axis-receipt label join."""

    return audited_harness._prepare_batch(  # noqa: SLF001
        partition,
        indices,
        device=device,
    )


def _forward_tensor_only(
    model: nn.Module,
    tensors: EndogenousCongruenceTensors,
) -> NeuralEndogenousRecordFiberOutput:
    """Cross the neural boundary with physical tensors and nothing else."""

    if type(tensors) is not EndogenousCongruenceTensors:
        raise EndogenousRecordFiberTrainingHarnessError(
            "model forward accepts only exact EndogenousCongruenceTensors"
        )
    output = model(tensors)
    if not isinstance(output, NeuralEndogenousRecordFiberOutput):
        raise EndogenousRecordFiberTrainingHarnessError(
            "model did not return NeuralEndogenousRecordFiberOutput"
        )
    return output


def _objective_snapshot(
    objective: RecordFiberLoss,
    output: NeuralEndogenousRecordFiberOutput,
) -> ObjectiveSnapshot:
    return ObjectiveSnapshot(
        total=float(objective.total.detach().item()),
        code=float(objective.code.detach().item()),
        fiber=float(objective.fiber.detach().item()),
        distance=float(objective.distance.detach().item()),
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
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        objective = record_fiber_loss(
            output,
            batch.same_class_target,
            margin=config.margin,
        )
        snapshot = _objective_snapshot(objective, output)
    if was_training:
        model.train()
    return snapshot


def train_record_fiber(
    model: NeuralEndogenousRecordFiber,
    training: OfflinePartition,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    """Optimize only an explicit train partition; development is not accepted."""

    if training.name != "train":
        raise EndogenousRecordFiberTrainingHarnessError(
            "optimizer accepts only an explicit train partition"
        )
    if config.batch_size > len(training.packets):
        raise EndogenousRecordFiberTrainingHarnessError(
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
    initial = _calibration_snapshot(
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
        objective = record_fiber_loss(
            output,
            batch.same_class_target,
            margin=config.margin,
        )
        objective.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise EndogenousRecordFiberTrainingHarnessError(
                "gradient norm is non-finite"
            )
        optimizer.step()
        if update == 1 or update % config.log_interval == 0 or update == config.updates:
            item = {
                "update": update,
                **asdict(_objective_snapshot(objective, output)),
                "gradient_norm": float(gradient_norm.detach().item()),
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
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _target_relation_for_singleton(batch: OfflineBatch) -> Tensor:
    if batch.same_class_target.shape[0] != 1:
        raise EndogenousRecordFiberTrainingHarnessError(
            "hard assessment requires a singleton label batch"
        )
    return batch.same_class_target[0].detach().cpu()


def _equivalence_is_valid(relation: Tensor, record_mask: Tensor) -> bool:
    """Independently verify the structural invariant without repairing it."""

    if (
        type(relation) is not Tensor
        or tuple(relation.shape) != (N, N)
        or relation.dtype != torch.bool
    ):
        return False
    if (
        type(record_mask) is not Tensor
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


def assess_one_example(
    model: nn.Module,
    partition: OfflinePartition,
    index: int,
    *,
    device: torch.device,
) -> HardExampleResult:
    """Use the model's one hard decode and separately assess physical truth."""

    started = time.perf_counter()
    batch = _prepare_batch(partition, (index,), device=device)
    metadata = partition.metadata[index]
    packet = partition.packets[index]
    target = _target_relation_for_singleton(batch)
    with torch.inference_mode():
        output = _forward_tensor_only(model, batch.tensorization.tensors)
        hard = output.hard
        physical = measure_record_fiber_physical_laws(
            batch.tensorization.tensors,
            hard,
        )

    predicted = hard.equivalence[0].detach().cpu()
    record_mask = hard.record_mask[0].detach().cpu()
    equivalence_valid = _equivalence_is_valid(predicted, record_mask)
    if not equivalence_valid:
        raise EndogenousRecordFiberTrainingHarnessError(
            "Record-Fiber hard decoder violated its equivalence guarantee"
        )
    active_pairs = record_mask[:, None] & record_mask[None, :]
    unordered = torch.triu(active_pairs, diagonal=1)
    target_positive = unordered & target
    target_negative = unordered & ~target
    predicted_positive = unordered & predicted
    false_splits = int((target_positive & ~predicted).sum().item())
    false_collisions = int((target_negative & predicted).sum().item())
    exact = torch.equal(predicted, target)
    active_vote_logits = output.vote_logits[0][
        active_pairs.to(output.vote_logits.device)
    ]
    minimum_margin = (
        float(active_vote_logits.abs().min().item())
        if active_vote_logits.numel()
        else 0.0
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
        physical_law_valid=bool(physical.valid[0].item()),
        observation_valid=bool(physical.observation_valid[0].item()),
        descent_valid=bool(physical.descent_valid[0].item()),
        projector_symmetric=bool(physical.projector_symmetric[0].item()),
        projector_idempotent=bool(physical.projector_idempotent[0].item()),
        projector_row_stochastic=bool(physical.projector_row_stochastic[0].item()),
        exact_target_relation=exact,
        coarsest_target_relation=exact,
        false_splits=false_splits,
        false_collisions=false_collisions,
        target_positive_pairs=int(target_positive.sum().item()),
        target_negative_pairs=int(target_negative.sum().item()),
        predicted_positive_pairs=int(predicted_positive.sum().item()),
        hard_descent_residual=float(physical.residuals.descent[0].item()),
        hard_observation_residual=float(physical.residuals.observation[0].item()),
        minimum_absolute_vote_margin=minimum_margin,
        equivalent_pairs=equivalent_pairs,
        elapsed_seconds=time.perf_counter() - started,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_results(
    results: Sequence[HardExampleResult],
) -> dict[str, object]:
    examples = len(results)
    false_splits = sum(result.false_splits for result in results)
    false_collisions = sum(result.false_collisions for result in results)
    target_positive = sum(result.target_positive_pairs for result in results)
    target_negative = sum(result.target_negative_pairs for result in results)
    return {
        "examples": examples,
        "equivalence_valid": sum(result.equivalence_valid for result in results),
        "equivalence_valid_rate": (
            sum(result.equivalence_valid for result in results) / examples
            if examples
            else None
        ),
        "physical_law_valid": sum(result.physical_law_valid for result in results),
        "physical_law_valid_rate": (
            sum(result.physical_law_valid for result in results) / examples
            if examples
            else None
        ),
        "observation_valid_rate": (
            sum(result.observation_valid for result in results) / examples
            if examples
            else None
        ),
        "descent_valid_rate": (
            sum(result.descent_valid for result in results) / examples
            if examples
            else None
        ),
        "projector_symmetric_rate": (
            sum(result.projector_symmetric for result in results) / examples
            if examples
            else None
        ),
        "projector_idempotent_rate": (
            sum(result.projector_idempotent for result in results) / examples
            if examples
            else None
        ),
        "projector_row_stochastic_rate": (
            sum(result.projector_row_stochastic for result in results) / examples
            if examples
            else None
        ),
        "exact_target_relations": sum(
            result.exact_target_relation for result in results
        ),
        "exact_target_relation_rate": (
            sum(result.exact_target_relation for result in results) / examples
            if examples
            else None
        ),
        "coarsest_target_relations": sum(
            result.coarsest_target_relation for result in results
        ),
        "coarsest_target_relation_rate": (
            sum(result.coarsest_target_relation for result in results) / examples
            if examples
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
        "minimum_absolute_vote_margin": (
            min(result.minimum_absolute_vote_margin for result in results)
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


def evaluate_record_fiber(
    model: nn.Module,
    partition: OfflinePartition,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Run exactly one built-in hard decode per packet and report all cells."""

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
        "hard_decoder": "record-fiber signature-row equality",
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
    """Seal every repository module that can affect packets, labels, or scores."""

    modules = {
        "audited_partition_label_harness": "train_neural_endogenous_congruence",
        "record_fiber_model": "pipeline.neural_endogenous_record_fiber",
        "neural_eccr_encoder": "pipeline.neural_endogenous_congruence",
        "packet_tensorizer": "pipeline.tensorize_endogenous_congruence",
        "packet_mechanics": "pipeline.endogenous_congruence_board",
        "procedural_generator": "pipeline.generate_endogenous_congruence_corpus",
    }
    paths = {"trainer": Path(__file__).resolve()}
    for name, module_name in modules.items():
        module_file = inspect.getsourcefile(importlib.import_module(module_name))
        if module_file is None:
            raise EndogenousRecordFiberTrainingHarnessError(
                f"source path is unavailable for {module_name}"
            )
        paths[name] = Path(module_file).resolve()
    return {name: _sha256_file(path) for name, path in paths.items()}


def _parameter_ledger(
    model: NeuralEndogenousRecordFiber,
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
        raise EndogenousRecordFiberTrainingHarnessError(
            "parameter ledger does not reconcile"
        )
    if summary["protected_base"] != PROTECTED_BASE_PARAMETERS:
        raise EndogenousRecordFiberTrainingHarnessError(
            "protected base parameter receipt drifted"
        )
    if (
        summary["complete_system"] != PROTECTED_BASE_PARAMETERS + total
        or summary["complete_system"] >= SYSTEM_PARAMETER_CAP
        or not summary["under_system_cap"]
    ):
        raise EndogenousRecordFiberTrainingHarnessError(
            "complete system is not strictly below 200M parameters"
        )
    return {"summary": summary, "parameters": parameters}


def _publish_artifact_bundle(
    output_dir: Path,
    checkpoint: dict[str, object],
    report: dict[str, object],
) -> dict[str, object]:
    """Publish checkpoint and report as one atomic directory rename."""

    if output_dir.exists():
        raise EndogenousRecordFiberTrainingHarnessError(
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
        checkpoint_path = staging / "record_fiber.pt"
        torch.save(checkpoint, checkpoint_path)
        report["checkpoint"] = {
            "path": str(output_dir / "record_fiber.pt"),
            "sha256": _sha256_file(checkpoint_path),
        }
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
    encoder_config = NeuralEndogenousCongruenceConfig(
        hidden_dim=args.hidden_dim,
        rounds=args.rounds,
        parameter_cap=args.encoder_parameter_cap,
    )
    model_config = NeuralEndogenousRecordFiberConfig(
        encoder_config=encoder_config,
        vote_hidden_dim=args.vote_hidden_dim,
        soft_temperature=args.soft_temperature,
        soft_majority_scale=args.soft_majority_scale,
        parameter_cap=args.parameter_cap,
    )
    model = NeuralEndogenousRecordFiber(model_config).to(device)
    training_config = TrainingConfig(
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        margin=args.margin,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    training_report = train_record_fiber(
        model,
        partitions.train,
        config=training_config,
        device=device,
    )
    train_evaluation = evaluate_record_fiber(
        model,
        partitions.train,
        device=device,
    )
    development_evaluation = evaluate_record_fiber(
        model,
        partitions.development,
        device=device,
    )
    source_after = _source_receipts()
    if source_after != source_before:
        raise EndogenousRecordFiberTrainingHarnessError(
            "score-bearing source changed during generation, optimization, or scoring"
        )
    parameter_ledger = _parameter_ledger(model)
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    optimizer_digests = set(training_report["optimizer_packet_digests"])
    if development_digests & optimizer_digests:
        raise EndogenousRecordFiberTrainingHarnessError(
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
        "parameter_ledger_summary": parameter_ledger["summary"],
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "claim_scope": (
            "sealed exploratory Record-Fiber quotient induction; guaranteed "
            "equivalence is structural and not a general-reasoning claim"
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
                "audited offline target relations by packet SHA-256, placed "
                "through tensorizer axis receipts"
            ),
            "partition_source": "offline CorpusPacketMetadata.partition only",
            "optimizer_partition": "train only",
            "development_labels_in_optimizer": False,
            "objective": (
                "record_fiber_loss: code + fiber + distance + margin; target "
                "relations are assessor-side supervision only"
            ),
            "physical_residuals": (
                "reported during optimization and hard evaluation; never used "
                "to repair a decoded relation"
            ),
            "hard_selection": (
                "one decoder invocation inside model forward; majority-vote "
                "anonymous signatures followed by exact row equality"
            ),
            "equivalence_validity": (
                "guaranteed by construction and independently asserted; "
                "physical-law validity and coarsest-target exactness are "
                "reported separately"
            ),
            "repair_search_retry": False,
        },
    }
    return _publish_artifact_bundle(Path(args.output_dir), checkpoint, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="train/neural_endogenous_record_fiber_exploratory",
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
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--encoder-parameter-cap", type=int, default=8_000_000)
    parser.add_argument("--vote-hidden-dim", type=int, default=128)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--soft-majority-scale", type=float, default=2.0)
    parser.add_argument("--parameter-cap", type=int, default=10_000_000)
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
