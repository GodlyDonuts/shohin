from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from episode_functor_constrained_transport import (
    LawfulMachineProjector,
)
from episode_functor_qualification_loss import (
    EFCQualificationLoss,
    QualificationLossError,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    generate_pilot_rows,
)
from pipeline.episode_functor_qualification_supervisor import (
    collate_qualification_supervision,
)


def _fixture():
    rows = generate_pilot_rows(
        seed="efc-qualification-loss-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    rows = tuple(row for row in rows if row.split == "train")
    batch = collate_witness_sources(
        tuple(scan_witness_source(row.source) for row in rows)
    )
    supervisor = collate_qualification_supervision(rows)
    hashes = tuple(sha256(row.source).hexdigest() for row in rows)
    return rows, batch, supervisor, hashes


def test_qualification_loss_is_post_forward_hash_joined_and_finite() -> None:
    torch.manual_seed(101)
    _, batch, supervisor, hashes = _fixture()
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
    )
    output = compiler(batch, straight_through=True)
    objective = EFCQualificationLoss()
    losses = objective(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    losses.total.backward()
    assert float(losses.total.detach()) > 0.0
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.parameters()
    )
    metrics = objective.exact_metrics(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.rows == len(hashes)
    assert metrics.hidden_transition_cells == 3 * len(hashes)
    assert metrics.hidden_observer_cells == 2 * len(hashes)
    assert 0 <= metrics.exact_machines <= metrics.rows


def test_oracle_machine_logits_score_exact_completion_metrics() -> None:
    _, batch, supervisor, hashes = _fixture()
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
    )
    output = compiler(batch)
    transition_logits = torch.full(
        (len(hashes), 3, 8, 8),
        -20.0,
    )
    transition_logits.scatter_(
        -1,
        supervisor.transition_next[..., None],
        20.0,
    )
    observer_logits = torch.full(
        (len(hashes), 2, 8, 4),
        -20.0,
    )
    observer_logits.scatter_(
        -1,
        supervisor.observer_answer[..., None],
        20.0,
    )
    projection = LawfulMachineProjector(
        sinkhorn_iterations=16
    )(
        transition_logits,
        observer_logits,
        straight_through=True,
    )
    output = replace(output, projection=projection)
    metrics = EFCQualificationLoss().exact_metrics(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.exact_transition_cells == 24 * len(hashes)
    assert metrics.exact_observer_cells == 16 * len(hashes)
    assert metrics.exact_hidden_transition_cells == 3 * len(hashes)
    assert metrics.exact_hidden_observer_cells == 2 * len(hashes)


def test_qualification_loss_rejects_receipt_or_batch_mismatch() -> None:
    _, batch, supervisor, hashes = _fixture()
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
    )
    output = compiler(batch)
    objective = EFCQualificationLoss()
    bad_hashes = ("0" * 64, *hashes[1:])
    with pytest.raises(
        (QualificationLossError, ValueError),
        match="source receipts differ",
    ):
        objective(
            output,
            supervisor,
            candidate_source_sha256=bad_hashes,
        )
