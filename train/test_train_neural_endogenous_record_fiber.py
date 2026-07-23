from __future__ import annotations

from functools import lru_cache
import inspect
import json
from pathlib import Path

import pytest
import torch
import pipeline.neural_endogenous_record_fiber as record_fiber_module
from pipeline.neural_endogenous_congruence import NeuralEndogenousCongruenceConfig
from pipeline.neural_endogenous_record_fiber import (
    NeuralEndogenousRecordFiber,
    NeuralEndogenousRecordFiberConfig,
    NeuralEndogenousRecordFiberOutput,
    record_fiber_loss,
)
from pipeline.tensorize_endogenous_congruence import (
    EndogenousCongruenceTensors,
)
from torch import nn

import train_neural_endogenous_record_fiber as harness


DATA_SEED = 2026072306


@lru_cache(maxsize=1)
def _partitions() -> harness.ProceduralPartitions:
    return harness.load_procedural_partitions(
        seed=DATA_SEED,
        train_packets=8,
        development_packets=8,
    )


def _tiny_model(seed: int = 23) -> NeuralEndogenousRecordFiber:
    torch.manual_seed(seed)
    return NeuralEndogenousRecordFiber(
        NeuralEndogenousRecordFiberConfig(
            encoder_config=NeuralEndogenousCongruenceConfig(
                hidden_dim=8,
                rounds=1,
                parameter_cap=8_000_000,
            ),
            vote_hidden_dim=8,
            parameter_cap=10_000_000,
        )
    )


class _StaticFiber(nn.Module):
    def __init__(self, output: NeuralEndogenousRecordFiberOutput) -> None:
        super().__init__()
        self.output = output
        self.forward_calls = 0
        self.tensor_only = False

    def forward(
        self,
        tensors: EndogenousCongruenceTensors,
    ) -> NeuralEndogenousRecordFiberOutput:
        self.forward_calls += 1
        self.tensor_only = type(tensors) is EndogenousCongruenceTensors
        return self.output


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
    relation = batch.same_class_target[0].cpu()
    active = batch.tensorization.tensors.record_mask[0].cpu()
    unordered = torch.triu(active[:, None] & active[None, :], diagonal=1)
    positive = unordered & relation
    negative = unordered & ~relation
    equivalent_pairs = tuple(
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
        equivalence_valid=True,
        physical_law_valid=True,
        observation_valid=True,
        descent_valid=True,
        projector_symmetric=True,
        projector_idempotent=True,
        projector_row_stochastic=True,
        exact_target_relation=True,
        coarsest_target_relation=True,
        false_splits=0,
        false_collisions=0,
        target_positive_pairs=int(positive.sum().item()),
        target_negative_pairs=int(negative.sum().item()),
        predicted_positive_pairs=int(positive.sum().item()),
        hard_descent_residual=0.0,
        hard_observation_residual=0.0,
        minimum_absolute_vote_margin=1.0,
        equivalent_pairs=equivalent_pairs,
        elapsed_seconds=0.0,
    )


def test_parser_defaults_match_frozen_exploratory_gate() -> None:
    args = harness.build_parser().parse_args([])
    assert args.train_packets == 256
    assert args.development_packets == 64
    assert args.rounds == 8
    assert args.updates == 800


def test_audited_digest_join_and_tensor_only_boundary_are_exact() -> None:
    partitions = _partitions()
    train_digests = {item.packet_sha256 for item in partitions.train.metadata}
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    assert not train_digests & development_digests
    batch = harness._prepare_batch(  # noqa: SLF001
        partitions.train,
        tuple(range(8)),
        device=torch.device("cpu"),
    )
    assert batch.packet_digests == tuple(
        item.packet_sha256 for item in partitions.train.metadata
    )
    assert torch.equal(
        batch.same_class_target.diagonal(dim1=1, dim2=2),
        batch.tensorization.tensors.record_mask,
    )
    output = _tiny_model()(batch.tensorization.tensors)
    static = _StaticFiber(output)
    returned = harness._forward_tensor_only(  # noqa: SLF001
        static,
        batch.tensorization.tensors,
    )
    assert returned is output
    assert static.forward_calls == 1
    assert static.tensor_only

    signature = inspect.signature(harness._forward_tensor_only)  # noqa: SLF001
    assert tuple(signature.parameters) == ("model", "tensors")
    source = inspect.getsource(harness._forward_tensor_only).lower()  # noqa: SLF001
    for forbidden in ("label", "target", "metadata", "receipt", "oracle"):
        assert forbidden not in source


def test_record_fiber_objective_reports_all_terms_and_has_gradients() -> None:
    batch = harness._prepare_batch(  # noqa: SLF001
        _partitions().train,
        (0, 1),
        device=torch.device("cpu"),
    )
    model = _tiny_model()
    output = model(batch.tensorization.tensors)
    output.vote_logits.retain_grad()
    objective = record_fiber_loss(output, batch.same_class_target, margin=1.0)
    for term in (
        objective.total,
        objective.code,
        objective.fiber,
        objective.distance,
        objective.margin,
    ):
        assert torch.isfinite(term)
    objective.total.backward()
    assert output.vote_logits.grad is not None
    active = (
        batch.tensorization.tensors.record_mask[:, :, None]
        & batch.tensorization.tensors.record_mask[:, None, :]
    ).unsqueeze(-1)
    assert torch.any(output.vote_logits.grad.masked_select(active) != 0)


def test_optimizer_rejects_development_and_never_accepts_it_in_api() -> None:
    partitions = _partitions()
    config = harness.TrainingConfig(
        updates=1,
        batch_size=2,
        learning_rate=1e-3,
        log_interval=1,
    )
    with pytest.raises(
        harness.EndogenousRecordFiberTrainingHarnessError,
        match="train partition",
    ):
        harness.train_record_fiber(
            _tiny_model(),
            partitions.development,
            config=config,
            device=torch.device("cpu"),
        )
    signature = inspect.signature(harness.train_record_fiber)
    assert "development" not in signature.parameters
    assert "evaluation" not in signature.parameters
    report = harness.train_record_fiber(
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
    for section in ("initial_calibration", "final_calibration"):
        assert set(report[section]) == {
            "total",
            "code",
            "fiber",
            "distance",
            "margin",
            "soft_descent_residual",
            "soft_observation_residual",
            "hard_descent_residual",
            "hard_observation_residual",
        }


def test_hard_assessment_decodes_once_and_separates_physical_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = harness.subset_partition(_partitions().development, (0,))
    original = record_fiber_module.decode_record_fiber_vote_logits
    calls = 0

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        record_fiber_module,
        "decode_record_fiber_vote_logits",
        counted_decode,
    )
    result = harness.assess_one_example(
        _tiny_model(),
        partition,
        0,
        device=torch.device("cpu"),
    )
    assert calls == 1
    assert result.equivalence_valid
    assert result.projector_symmetric
    assert result.projector_idempotent
    assert result.projector_row_stochastic
    assert isinstance(result.physical_law_valid, bool)
    assert result.coarsest_target_relation == result.exact_target_relation
    assert result.false_splits >= 0
    assert result.false_collisions >= 0


def test_exact_orbits_and_grouped_metrics_are_auditable() -> None:
    for partition in (_partitions().train, _partitions().development):
        results = [
            _exact_result(partition, index) for index in range(len(partition.packets))
        ]
        summary = harness._summarize_results(results)  # noqa: SLF001
        assert summary["equivalence_valid_rate"] == 1.0
        assert summary["physical_law_valid_rate"] == 1.0
        assert summary["exact_target_relation_rate"] == 1.0
        assert summary["coarsest_target_relation_rate"] == 1.0
        assert summary["false_splits"] == 0
        assert summary["false_collisions"] == 0
        report = harness.audited_harness._orbit_consistency_report(  # noqa: SLF001
            partition,
            results,
        )
        assert report["complete_orbits"] == 1
        for key in ("reindex", "recoding", "split", "merge", "all"):
            assert report[key]["rate"] == 1.0


def test_source_seals_parameter_ledger_and_atomic_tiny_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = harness._source_receipts()  # noqa: SLF001
    assert set(source) == {
        "trainer",
        "audited_partition_label_harness",
        "record_fiber_model",
        "neural_eccr_encoder",
        "packet_tensorizer",
        "packet_mechanics",
        "procedural_generator",
    }
    assert all(len(value) == 64 for value in source.values())
    monkeypatch.setattr(
        harness,
        "load_procedural_partitions",
        lambda **_: _partitions(),
    )
    output_dir = tmp_path / "sealed_record_fiber"
    args = harness.build_parser().parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--seed",
            "29",
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
            "--vote-hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )
    harness.run_training(args)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "record_fiber.pt"
    assert report_path.is_file()
    assert checkpoint_path.is_file()
    assert not list(tmp_path.glob(".sealed_record_fiber.part-*"))
    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == harness.REPORT_SCHEMA
    assert persisted["source_sha256"]["before"] == source
    assert persisted["source_sha256"]["after"] == source
    assert persisted["source_sha256"]["unchanged"]
    assert persisted["custody_boundary"]["development_labels_in_optimizer"] is False
    assert persisted["custody_boundary"]["model_forward_input"] == (
        "EndogenousCongruenceTensors only"
    )
    assert persisted["custody_boundary"]["repair_search_retry"] is False
    assert persisted["training"]["optimizer_development_packet_digests"] == []
    assert set(persisted["training"]["trace"][0]) >= {
        "total",
        "code",
        "fiber",
        "distance",
        "margin",
        "soft_descent_residual",
        "soft_observation_residual",
        "hard_descent_residual",
        "hard_observation_residual",
    }
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
        assert evaluation["decode_count_per_example"] == 1
        assert evaluation["equivalence_valid_rate"] == 1.0
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
        harness.EndogenousRecordFiberTrainingHarnessError,
        match="not overwritten",
    ):
        harness._publish_artifact_bundle(  # noqa: SLF001
            output_dir,
            {},
            {},
        )
