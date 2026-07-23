from __future__ import annotations

import dataclasses
from functools import lru_cache
import inspect
import json
from pathlib import Path

import pytest
import torch
from pipeline.neural_endogenous_congruence import (
    NeuralEndogenousCongruence,
    NeuralEndogenousCongruenceConfig,
    NeuralEndogenousCongruenceOutput,
)
from pipeline.tensorize_endogenous_congruence import (
    N,
    EndogenousCongruenceTensors,
)
from torch import nn

import train_neural_endogenous_congruence as harness


DATA_SEED = 2026072304


@lru_cache(maxsize=1)
def _partitions() -> harness.ProceduralPartitions:
    return harness.load_procedural_partitions(
        seed=DATA_SEED,
        train_packets=8,
        development_packets=8,
    )


def _tiny_model(seed: int = 17) -> NeuralEndogenousCongruence:
    torch.manual_seed(seed)
    return NeuralEndogenousCongruence(
        NeuralEndogenousCongruenceConfig(
            hidden_dim=8,
            rounds=1,
            parameter_cap=8_000_000,
        )
    )


class _StaticInducer(nn.Module):
    def __init__(self, output: NeuralEndogenousCongruenceOutput) -> None:
        super().__init__()
        self.output = output
        self.forward_calls = 0
        self.tensor_only = False

    def forward(
        self,
        tensors: EndogenousCongruenceTensors,
    ) -> NeuralEndogenousCongruenceOutput:
        self.forward_calls += 1
        self.tensor_only = type(tensors) is EndogenousCongruenceTensors
        return self.output


def _forced_output(
    partition: harness.OfflinePartition,
    index: int,
    *,
    exact: bool = True,
) -> NeuralEndogenousCongruenceOutput:
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        (index,),
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        output = _tiny_model()(batch.tensorization.tensors)
    relation = batch.same_class_target.clone()
    if not exact:
        relation[0, 0, 0] = False
    logits = torch.where(
        relation,
        torch.tensor(12.0),
        torch.tensor(-12.0),
    )
    return dataclasses.replace(output, same_class_logits=logits)


def _exact_result(
    partition: harness.OfflinePartition,
    index: int,
) -> harness.HardExampleResult:
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        (index,),
        device=torch.device("cpu"),
    )
    metadata = partition.metadata[index]
    packet = partition.packets[index]
    relation = batch.same_class_target[0]
    identity = torch.eye(N, dtype=torch.bool)
    off_diagonal = relation & ~identity
    pairs = tuple(
        (left, right)
        for left_index, left in enumerate(packet.records)
        for right_index, right in enumerate(packet.records)
        if bool(relation[left_index, right_index])
    )
    return harness.HardExampleResult(
        packet_sha256=metadata.packet_sha256,
        orbit_id=metadata.orbit_id,
        variant=metadata.variant,
        family=metadata.family,
        motif=metadata.motif,
        cell=metadata.cell,
        records=packet.records,
        exact_relation=True,
        valid_decode=True,
        coarsest_valid_relation=True,
        coarseness_recall=1.0,
        coarseness_precision=1.0,
        predicted_off_diagonal_pairs=int(off_diagonal.sum().item()),
        target_off_diagonal_pairs=int(off_diagonal.sum().item()),
        equivalent_pairs=pairs,
        invalid_reason=None,
        elapsed_seconds=0.0,
    )


def test_metadata_partition_and_digest_receipt_label_join_are_strict() -> None:
    partitions = _partitions()
    assert len(partitions.train.packets) == 8
    assert len(partitions.development.packets) == 8
    train_digests = {item.packet_sha256 for item in partitions.train.metadata}
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    assert not train_digests & development_digests
    assert all(
        item.partition == harness.TRAIN_PARTITION for item in partitions.train.metadata
    )
    assert all(
        item.partition == harness.DEVELOPMENT_PARTITION
        for item in partitions.development.metadata
    )

    batch = harness._prepare_batch(  # noqa: SLF001
        partitions.train,
        tuple(range(8)),
        device=torch.device("cpu"),
    )
    assert batch.packet_digests == tuple(
        item.packet_sha256 for item in partitions.train.metadata
    )
    assert batch.same_class_target.dtype == torch.bool
    assert torch.equal(
        batch.same_class_target.diagonal(dim1=1, dim2=2),
        batch.tensorization.tensors.record_mask,
    )
    assert all(
        receipt.record_ids == packet.records
        and receipt.generator_ids == packet.generators
        and receipt.query_ids == packet.query_ports
        for packet, receipt in zip(
            partitions.train.packets,
            batch.tensorization.receipts,
            strict=True,
        )
    )

    missing = {target.packet_sha256: target for target in partitions.train.targets[1:]}
    with pytest.raises(
        harness.EndogenousCongruenceTrainingHarnessError,
        match="digest join",
    ):
        harness._same_class_labels_from_receipts(  # noqa: SLF001
            batch.tensorization,
            partitions.train.packets,
            missing,
            device=torch.device("cpu"),
        )


def test_complete_active_pair_objective_is_balanced_and_differentiable() -> None:
    partition = _partitions().train
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        tuple(range(8)),
        device=torch.device("cpu"),
    )
    model = _tiny_model()
    output = model(batch.tensorization.tensors)
    output.same_class_logits.retain_grad()
    objective = harness.complete_active_pair_loss(
        output,
        batch.tensorization.tensors,
        batch.same_class_target,
        diagonal_weight=1.0,
        descent_weight=0.05,
        observation_weight=0.05,
    )
    expected_active = int(
        (
            batch.tensorization.tensors.record_mask[:, :, None]
            & batch.tensorization.tensors.record_mask[:, None, :]
        )
        .sum()
        .item()
    )
    assert objective.active_pairs == expected_active
    assert objective.diagonal_pairs == int(
        batch.tensorization.tensors.record_mask.sum().item()
    )
    assert objective.positive_off_diagonal_pairs > 0
    assert objective.negative_off_diagonal_pairs > 0
    assert (
        objective.diagonal_pairs
        + objective.positive_off_diagonal_pairs
        + objective.negative_off_diagonal_pairs
        == objective.active_pairs
    )
    objective.loss.backward()
    gradient = output.same_class_logits.grad
    assert gradient is not None
    active = output.equivalence_mask
    assert torch.all(torch.isfinite(gradient[active]))
    assert torch.all(gradient[batch.tensorization.tensors.record_equal] < 0)
    assert torch.any(gradient[active] != 0)


def test_optimizer_rejects_development_and_never_accepts_labels_in_forward() -> None:
    partitions = _partitions()
    config = harness.TrainingConfig(
        updates=1,
        batch_size=2,
        learning_rate=1e-3,
        log_interval=1,
    )
    with pytest.raises(
        harness.EndogenousCongruenceTrainingHarnessError,
        match="train partition",
    ):
        harness.train_inducer(
            _tiny_model(),
            partitions.development,
            config=config,
            device=torch.device("cpu"),
        )
    singleton = harness.subset_partition(partitions.train, (0,))
    with pytest.raises(
        harness.EndogenousCongruenceTrainingHarnessError,
        match="unique training packets",
    ):
        harness.train_inducer(
            _tiny_model(),
            singleton,
            config=config,
            device=torch.device("cpu"),
        )

    signature = inspect.signature(harness._forward_tensor_only)  # noqa: SLF001
    assert tuple(signature.parameters) == ("model", "tensors")
    source = inspect.getsource(harness._forward_tensor_only).lower()  # noqa: SLF001
    for forbidden in ("label", "target", "metadata", "receipt", "oracle"):
        assert forbidden not in source
    train_signature = inspect.signature(harness.train_inducer)
    assert "development" not in train_signature.parameters
    assert "evaluation" not in train_signature.parameters

    report = harness.train_inducer(
        _tiny_model(),
        harness.subset_partition(partitions.train, (0, 1)),
        config=config,
        device=torch.device("cpu"),
    )
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    assert not development_digests & set(report["optimizer_packet_digests"])
    assert report["optimizer_development_packet_digests"] == []


def test_hard_assessment_calls_decoder_once_and_never_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = harness.subset_partition(_partitions().development, (0,))
    original = harness.decode_endogenous_congruence_logits
    calls = 0

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        harness,
        "decode_endogenous_congruence_logits",
        counted_decode,
    )
    exact_model = _StaticInducer(_forced_output(partition, 0))
    exact = harness.assess_one_example(
        exact_model,
        partition,
        0,
        threshold=0.0,
        device=torch.device("cpu"),
    )
    assert calls == 1
    assert exact_model.tensor_only
    assert exact.exact_relation
    assert exact.valid_decode
    assert exact.coarsest_valid_relation
    assert exact.coarseness_recall == 1.0
    assert exact.coarseness_precision == 1.0

    invalid_model = _StaticInducer(_forced_output(partition, 0, exact=False))
    invalid = harness.assess_one_example(
        invalid_model,
        partition,
        0,
        threshold=0.0,
        device=torch.device("cpu"),
    )
    assert calls == 2
    assert not invalid.valid_decode
    assert not invalid.exact_relation
    assert invalid.equivalent_pairs is None
    assert invalid.invalid_reason == "relation is not reflexive"


def test_exact_orbits_report_recode_reindex_split_and_merge_consistency() -> None:
    for partition in (_partitions().train, _partitions().development):
        results = [
            _exact_result(partition, index) for index in range(len(partition.packets))
        ]
        report = harness._orbit_consistency_report(  # noqa: SLF001
            partition,
            results,
        )
        assert report["complete_orbits"] == 1
        assert report["incomplete_orbits"] == 0
        for key in ("reindex", "recoding", "split", "merge", "all"):
            assert report[key]["rate"] == 1.0
            assert report[key]["passed"] == 1
        row = report["orbits"][0]
        assert row["base_reindex_consistent"]
        assert row["collision_reindex_consistent"]
        assert row["all_consistent"]


def test_source_seal_parameter_ledger_and_atomic_tiny_run(
    tmp_path: Path,
) -> None:
    source = harness._source_receipts()  # noqa: SLF001
    assert set(source) == {
        "trainer",
        "neural_inducer",
        "packet_tensorizer",
        "packet_mechanics",
        "procedural_generator",
    }
    assert all(len(value) == 64 for value in source.values())
    output_dir = tmp_path / "sealed_eccr"
    args = harness.build_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--seed",
            "19",
            "--data-seed",
            str(DATA_SEED),
            "--train-packets",
            "8",
            "--development-packets",
            "8",
            "--updates",
            "1",
            "--batch-size",
            "2",
            "--log-interval",
            "1",
            "--hidden-dim",
            "8",
            "--rounds",
            "1",
            "--device",
            "cpu",
        ]
    )
    harness.run_training(args)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "inducer.pt"
    assert report_path.is_file()
    assert checkpoint_path.is_file()
    assert not list(tmp_path.glob(".sealed_eccr.part-*"))
    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == harness.REPORT_SCHEMA
    assert persisted["source_sha256"]["before"] == source
    assert persisted["source_sha256"]["after"] == source
    assert persisted["source_sha256"]["unchanged"]
    assert persisted["custody_boundary"]["development_labels_in_optimizer"] is False
    assert persisted["custody_boundary"]["model_forward_input"] == (
        "EndogenousCongruenceTensors only"
    )
    assert persisted["training"]["optimizer_development_packet_digests"] == []
    summary = persisted["parameter_ledger"]["summary"]
    assert summary["total"] == summary["trainable"]
    assert summary["complete_system"] == summary["protected_base"] + summary["total"]
    assert summary["complete_system"] < 200_000_000
    assert summary["under_system_cap"]
    assert (
        sum(item["parameters"] for item in persisted["parameter_ledger"]["parameters"])
        == summary["total"]
    )
    assert persisted["checkpoint"]["sha256"] == harness._sha256_file(  # noqa: SLF001
        checkpoint_path
    )
    for evaluation in (
        persisted["train_evaluation"],
        persisted["development_evaluation"],
    ):
        assert len(evaluation["families"]) >= 1
        assert len(evaluation["motifs"]) >= 1
        assert len(evaluation["variants"]) == 8
        assert len(evaluation["cells"]) == 8
        assert set(evaluation["orbit_consistency"]) >= {
            "reindex",
            "recoding",
            "split",
            "merge",
        }
    with pytest.raises(
        harness.EndogenousCongruenceTrainingHarnessError,
        match="not overwritten",
    ):
        harness.run_training(args)
