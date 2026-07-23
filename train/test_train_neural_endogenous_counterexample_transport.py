from __future__ import annotations

import inspect
import json
from pathlib import Path
import random

import pytest
import torch
from pipeline import neural_endogenous_counterexample_transport as transport_module
from pipeline.neural_endogenous_counterexample_transport import (
    NeuralEndogenousCounterexampleTransport,
    NeuralEndogenousCounterexampleTransportConfig,
    NeuralEndogenousCounterexampleTransportOutput,
)
from torch import nn

import train_neural_endogenous_counterexample_transport as harness


DATA_SEED = 2026072306


def _partitions() -> harness.ProceduralPartitions:
    return harness.load_procedural_partitions(
        seed=DATA_SEED,
        train_packets=8,
        development_packets=8,
    )


def _tiny_model(seed: int = 23) -> NeuralEndogenousCounterexampleTransport:
    torch.manual_seed(seed)
    return NeuralEndogenousCounterexampleTransport(
        NeuralEndogenousCounterexampleTransportConfig(
            hidden_dim=8,
            dynamical_bits=1,
            parameter_cap=24_000_000,
        )
    )


class _StaticTransport(nn.Module):
    def __init__(
        self,
        output: NeuralEndogenousCounterexampleTransportOutput,
    ) -> None:
        super().__init__()
        self.output = output
        self.forward_calls = 0
        self.tensor_only = False

    def forward(
        self,
        tensors: object,
    ) -> NeuralEndogenousCounterexampleTransportOutput:
        self.forward_calls += 1
        self.tensor_only = type(tensors).__name__ == ("EndogenousCongruenceTensors")
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
    target = batch.same_class_target[0]
    packet = partition.packets[index]
    metadata = partition.metadata[index]
    record_mask = batch.tensorization.tensors.record_mask[0]
    active_pairs = record_mask[:, None] & record_mask[None, :]
    unordered = torch.triu(active_pairs, diagonal=1)
    positive = unordered & target
    negative = unordered & ~target
    equivalent_pairs = tuple(
        (left, right)
        for left_index, left in enumerate(packet.records)
        for right_index, right in enumerate(packet.records)
        if bool(target[left_index, right_index])
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
        observation_valid=True,
        descent_valid=True,
        physical_law_valid=True,
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
        minimum_absolute_active_bit_margin=1.0,
        equivalent_pairs=equivalent_pairs,
        elapsed_seconds=0.0,
    )


def test_parser_defaults_match_frozen_mctfr_gate() -> None:
    args = harness.build_parser().parse_args([])
    assert args.seed == 2026072305
    assert args.data_seed == 2026072306
    assert args.train_packets == 256
    assert args.development_packets == 64
    assert args.updates == 800
    assert args.batch_size == 32
    assert args.hidden_dim == 192
    assert args.dynamical_bits == 1
    assert args.target_control == "true"


def test_digest_join_and_tensor_only_forward_boundary_are_exact() -> None:
    partitions = _partitions()
    train_digests = {item.packet_sha256 for item in partitions.train.metadata}
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    assert not train_digests & development_digests
    batch = harness._prepare_batch(  # noqa: SLF001
        partitions.train,
        (0, 1),
        device=torch.device("cpu"),
    )
    assert batch.packet_digests == tuple(
        partitions.train.metadata[index].packet_sha256 for index in (0, 1)
    )
    output = _tiny_model()(batch.tensorization.tensors)
    static = _StaticTransport(output)
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
    for forbidden in (
        "label",
        "target",
        "metadata",
        "partition",
        "receipt",
        "orbit",
        "family",
        "motif",
        "variant",
        "oracle",
        "assessor",
    ):
        assert forbidden not in source


def test_primary_sampler_is_unique_and_blind_to_assessor_metadata() -> None:
    partition = _partitions().train
    indices = harness._sample_unique_batch(  # noqa: SLF001
        partition,
        batch_size=8,
        sampler=random.Random(7),
    )
    assert len(indices) == 8
    assert len(set(indices)) == 8
    source = inspect.getsource(harness._sample_unique_batch).lower()  # noqa: SLF001
    for forbidden in ("metadata", "renderer", "orbit", "variant", "morphism"):
        assert forbidden not in source


def test_full_objective_has_every_term_and_reactor_gradients() -> None:
    partition = _partitions().train
    indices = harness._sample_unique_batch(  # noqa: SLF001
        partition,
        batch_size=8,
        sampler=random.Random(11),
    )
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        indices,
        device=torch.device("cpu"),
    )
    model = _tiny_model()
    output = model(batch.tensorization.tensors)
    output.dynamical_logits.retain_grad()
    objective = harness.counterexample_transport_loss(
        output,
        batch.tensorization.tensors,
        batch.same_class_target,
    )
    for term in (
        objective.total,
        objective.balanced_fiber_relation,
        objective.fiber,
        objective.relation,
        objective.max_descent,
        objective.fixed_point,
        objective.margin,
    ):
        assert torch.isfinite(term)
    objective.total.backward()
    assert output.dynamical_logits.grad is not None
    active = (
        batch.tensorization.tensors.record_mask[:, :, None]
        & batch.tensorization.tensors.record_mask[:, None, :]
    ).unsqueeze(-1)
    assert torch.any(output.dynamical_logits.grad.masked_select(active) != 0)
    gradient_groups = {
        name.split(".", maxsplit=1)[0]
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and torch.any(torch.isfinite(parameter.grad))
        and torch.any(parameter.grad != 0)
    }
    assert {
        "query_encoder",
        "initial_auxiliary",
        "successor_encoder",
        "distinction_increment",
        "auxiliary_update",
    } <= gradient_groups


def test_descent_loss_cannot_dilute_one_violation_with_safe_constraints() -> None:
    partition = _partitions().train
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        (0,),
        device=torch.device("cpu"),
    )
    tensors = batch.tensorization.tensors
    relation = torch.zeros(
        (1, transport_module.N, transport_module.N),
        dtype=torch.float32,
    )
    active = int(tensors.record_mask[0].sum().item())
    relation[0, :active, :active] = 0.1

    source = 0
    generator = 0
    successor = int(tensors.transition_target[0, source, generator].nonzero().item())
    relation[0, source, source] = 1.0
    relation[0, successor, successor] = 0.0

    loss = harness._smooth_max_descent_hinge(  # noqa: SLF001
        relation,
        tensors,
        temperature=0.1,
    )
    assert loss.item() > 0.9


def test_fixed_point_loss_uses_relation_proposals_not_raw_state_coordinate() -> None:
    partition = _partitions().train
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        (0, 1),
        device=torch.device("cpu"),
    )
    output = _tiny_model()(batch.tensorization.tensors)
    expected = (
        output.soft_fiber_relation - output.penultimate_soft_fiber_relation
    ).square()
    pair_mask = (
        batch.tensorization.tensors.record_mask[:, :, None]
        & batch.tensorization.tensors.record_mask[:, None, :]
    )
    torch.testing.assert_close(
        harness._fixed_point_loss(  # noqa: SLF001
            output,
            batch.tensorization.tensors.record_mask,
        ),
        expected.masked_select(pair_mask).mean(),
    )


def test_shuffled_target_control_is_valid_and_source_deleted() -> None:
    partition = _partitions().train
    batch = harness._prepare_batch(  # noqa: SLF001
        partition,
        tuple(range(8)),
        device=torch.device("cpu"),
    )
    shuffled, changed = harness._shuffle_target_relations(  # noqa: SLF001
        batch.same_class_target,
        batch.tensorization.tensors.record_mask,
    )
    repeated, repeated_changed = harness._shuffle_target_relations(  # noqa: SLF001
        batch.same_class_target,
        batch.tensorization.tensors.record_mask,
    )
    harness._require_target_equivalence(  # noqa: SLF001
        shuffled,
        batch.tensorization.tensors.record_mask,
    )
    assert torch.equal(shuffled, repeated)
    assert repeated_changed == changed
    assert changed >= 1
    assert torch.equal(
        shuffled.diagonal(dim1=1, dim2=2),
        batch.tensorization.tensors.record_mask,
    )
    control_source = inspect.getsource(
        harness._shuffle_target_relations  # noqa: SLF001
    ).lower()
    for forbidden in ("packet", "metadata", "family", "motif", "orbit"):
        assert forbidden not in control_source


def test_optimizer_rejects_development_and_never_accepts_it_in_api() -> None:
    partitions = _partitions()
    config = harness.TrainingConfig(
        updates=1,
        batch_size=2,
        learning_rate=1e-3,
        log_interval=1,
    )
    with pytest.raises(
        harness.EndogenousCounterexampleTransportTrainingHarnessError,
        match="train partition",
    ):
        harness.train_counterexample_transport(
            _tiny_model(),
            partitions.development,
            config=config,
            device=torch.device("cpu"),
        )
    signature = inspect.signature(harness.train_counterexample_transport)
    assert "development" not in signature.parameters
    assert "evaluation" not in signature.parameters
    optimizer_source = inspect.getsource(harness.train_counterexample_transport).lower()
    for forbidden in ("metadata", "renderer", "orbit", "variant", "morphism"):
        assert forbidden not in optimizer_source
    report = harness.train_counterexample_transport(
        _tiny_model(),
        partitions.train,
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
            "balanced_fiber_relation",
            "fiber",
            "relation",
            "max_descent",
            "fixed_point",
            "margin",
            "soft_descent_residual",
            "soft_observation_residual",
            "hard_descent_residual",
            "hard_observation_residual",
        }


def test_hard_assessment_decodes_once_and_separates_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = harness.subset_partition(_partitions().development, (0,))
    original = transport_module.decode_counterexample_transport_fibers
    calls = 0

    def counted_decode(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        transport_module,
        "decode_counterexample_transport_fibers",
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
    assert result.observation_valid
    assert result.projector_symmetric
    assert result.projector_idempotent
    assert result.projector_row_stochastic
    assert isinstance(result.descent_valid, bool)
    assert isinstance(result.physical_law_valid, bool)
    assert result.coarsest_target_relation == result.exact_target_relation
    assert result.false_splits >= 0
    assert result.false_collisions >= 0


def test_exact_orbits_grouped_metrics_and_conditional_exactness() -> None:
    for partition in (_partitions().train, _partitions().development):
        results = [
            _exact_result(partition, index) for index in range(len(partition.packets))
        ]
        summary = harness._summarize_results(results)  # noqa: SLF001
        assert summary["equivalence_valid_rate"] == 1.0
        assert summary["observation_valid_rate"] == 1.0
        assert summary["descent_valid_rate"] == 1.0
        assert summary["physical_law_valid_rate"] == 1.0
        assert summary["exact_target_relation_rate"] == 1.0
        assert summary["conditional_exact_physically_valid_rate"] == 1.0
        assert summary["false_splits"] == 0
        assert summary["false_collisions"] == 0
        orbit = harness.audited_harness._orbit_consistency_report(  # noqa: SLF001
            partition,
            results,
        )
        assert orbit["complete_orbits"] == 1
        for key in ("reindex", "recoding", "split", "merge", "all"):
            assert orbit[key]["rate"] == 1.0


def test_source_seals_parameter_ledger_and_atomic_tiny_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = harness._source_receipts()  # noqa: SLF001
    assert set(source) == {
        "trainer",
        "audited_partition_label_harness",
        "mctfr_model",
        "neural_eccr_encoder",
        "packet_tensorizer",
        "packet_mechanics",
        "procedural_generator",
    }
    assert all(len(value) == 64 for value in source.values())
    partitions = _partitions()
    monkeypatch.setattr(
        harness,
        "load_procedural_partitions",
        lambda **_: partitions,
    )
    protected_base = {
        "path": "train/flagship_out/ckpt_0300000.pt",
        "sha256": harness.PROTECTED_BASE_SHA256,
        "parameters": harness.PROTECTED_BASE_PARAMETERS,
        "integration_status": (
            "not integrated; complete-system count is hypothetical accounting"
        ),
    }
    monkeypatch.setattr(
        harness,
        "_protected_base_receipt",
        lambda: protected_base,
    )
    output_dir = tmp_path / "sealed_mctfr"
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
            "--dynamical-bits",
            "1",
            "--device",
            "cpu",
        ]
    )
    harness.run_training(args)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "mctfr.pt"
    manifest_path = output_dir / "bundle_manifest.json"
    assert report_path.is_file()
    assert checkpoint_path.is_file()
    assert manifest_path.is_file()
    assert not list(tmp_path.glob(".sealed_mctfr.part-*"))
    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == harness.REPORT_SCHEMA
    assert persisted["source_sha256"]["before"] == source
    assert persisted["source_sha256"]["after"] == source
    assert persisted["source_sha256"]["unchanged"]
    assert persisted["protected_base"] == protected_base
    custody = persisted["custody_boundary"]
    assert custody["model_forward_input"] == ("EndogenousCongruenceTensors only")
    assert custody["target_relations_in_forward"] is False
    assert custody["orbit_mappings_in_forward"] is False
    assert custody["intermediate_oracle_supervision"] is False
    assert custody["development_labels_in_optimizer"] is False
    assert custody["repair_search_retry"] is False
    assert persisted["training"]["optimizer_development_packet_digests"] == []
    assert set(persisted["training"]["trace"][0]) >= {
        "balanced_fiber_relation",
        "fiber",
        "relation",
        "max_descent",
        "fixed_point",
        "margin",
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
    manifest = harness.verify_artifact_bundle(output_dir)
    assert manifest["artifacts"]["mctfr.pt"] == harness._sha256_file(  # noqa: SLF001
        checkpoint_path
    )
    assert manifest["artifacts"]["report.json"] == harness._sha256_file(  # noqa: SLF001
        report_path
    )
    for evaluation in (
        persisted["train_evaluation"],
        persisted["development_evaluation"],
    ):
        assert evaluation["decode_count_per_example"] == 1
        assert evaluation["equivalence_valid_rate"] == 1.0
        assert evaluation["observation_valid_rate"] == 1.0
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
        harness.EndogenousCounterexampleTransportTrainingHarnessError,
        match="not overwritten",
    ):
        harness._publish_artifact_bundle(  # noqa: SLF001
            output_dir,
            {},
            {},
        )
    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(
        harness.EndogenousCounterexampleTransportTrainingHarnessError,
        match="failed verification",
    ):
        harness.verify_artifact_bundle(output_dir)


def test_shuffled_control_runs_without_changing_forward_surface() -> None:
    partition = _partitions().train
    config = harness.TrainingConfig(
        updates=1,
        batch_size=2,
        learning_rate=1e-3,
        target_control="shuffled",
        log_interval=1,
    )
    report = harness.train_counterexample_transport(
        _tiny_model(),
        partition,
        config=config,
        device=torch.device("cpu"),
    )
    assert report["target_control"] == "shuffled"
    assert report["optimizer_development_packet_digests"] == []
    source = inspect.getsource(
        transport_module.NeuralEndogenousCounterexampleTransport.forward
    ).lower()
    for forbidden in (
        "target",
        "partition",
        "renderer",
        "orbit",
        "family",
        "motif",
        "variant",
        "assessor",
    ):
        assert forbidden not in source
