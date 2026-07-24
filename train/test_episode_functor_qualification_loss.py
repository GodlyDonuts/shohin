from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from episode_functor_constrained_transport import (
    LawfulMachineProjector,
)
from episode_functor_hankel_completion import (
    HankelShiftCompletionProjector,
    project_behavioral_shifts,
)
from episode_functor_qualification_loss import (
    EFCHankelQualificationLoss,
    EFCQualificationLoss,
    HankelQualificationExactMetrics,
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
from pipeline.episode_functor_hankel_shift import (
    prefix_shift_incidence,
    random_shift_incidence,
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


def test_hankel_qualification_supervises_transient_codes_post_forward() -> None:
    torch.manual_seed(331)
    _, batch, supervisor, hashes = _fixture()
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=HankelShiftCompletionProjector(
            width=32,
            iterations=2,
            max_depth=2,
        ),
    )
    output = compiler(batch)
    objective = EFCHankelQualificationLoss()
    losses = objective(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert output.projector_auxiliary is not None
    assert float(losses.base_signature.detach()) > 0.0
    assert float(losses.derivative_signature.detach()) > 0.0
    assert float(losses.total.detach()) > float(losses.base.total.detach())
    losses.total.backward()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.parameters()
    )
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in compiler.parameters()
        if parameter.grad is not None
    ) > 0.0
    metrics = objective.exact_metrics(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert isinstance(metrics, HankelQualificationExactMetrics)
    assert metrics.base_signature_cells == len(hashes) * 8 * 13 * 2
    assert metrics.derivative_signature_cells == (
        len(hashes) * 3 * 8 * 13 * 2
    )


@pytest.mark.parametrize(
    ("incidence_mode", "expect_exact_derivative"),
    (("prefix", True), ("random", False)),
)
def test_hankel_arms_share_one_independent_prefix_target(
    incidence_mode: str,
    expect_exact_derivative: bool,
) -> None:
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
    transition = torch.nn.functional.one_hot(
        supervisor.transition_next,
        8,
    ).float()
    observer = torch.nn.functional.one_hot(
        supervisor.observer_answer,
        4,
    ).float()
    incidence = (
        prefix_shift_incidence(2)
        if incidence_mode == "prefix"
        else random_shift_incidence(2, seed="loss-control-v1")
    )
    details = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=2,
        incidence=torch.tensor(incidence),
        temperature=0.05,
    )
    output = replace(
        output,
        projection=details.projection,
        projector_auxiliary=details,
    )
    metrics = EFCHankelQualificationLoss().exact_metrics(
        output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.exact_base_signature_cells == metrics.base_signature_cells
    assert (
        metrics.exact_derivative_signature_cells
        == metrics.derivative_signature_cells
    ) is expect_exact_derivative
    assert metrics.exact_base_codebooks == len(hashes)
    assert (
        metrics.exact_derivative_codebooks == len(hashes)
    ) is expect_exact_derivative


def test_candidate_incidence_cannot_change_frozen_derivative_targets() -> None:
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
    transition = torch.nn.functional.one_hot(
        supervisor.transition_next,
        8,
    ).float()
    observer = torch.nn.functional.one_hot(
        supervisor.observer_answer,
        4,
    ).float()
    random_details = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=2,
        incidence=torch.tensor(
            random_shift_incidence(2, seed="loss-causal-mismatch-v1")
        ),
        temperature=0.05,
    )
    original_output = replace(
        output,
        projection=random_details.projection,
        projector_auxiliary=random_details,
    )
    changed_details = replace(
        random_details,
        shift_incidence=torch.tensor(prefix_shift_incidence(2)),
    )
    changed_output = replace(
        output,
        projection=changed_details.projection,
        projector_auxiliary=changed_details,
    )
    objective = EFCHankelQualificationLoss()
    original = objective.exact_metrics(
        original_output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    changed = objective.exact_metrics(
        changed_output,
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert (
        changed.exact_derivative_signature_cells
        == original.exact_derivative_signature_cells
    )


def test_output_source_receipt_rejects_cross_batch_substitution() -> None:
    _, batch, _, _ = _fixture()
    rows = generate_pilot_rows(
        seed="efc-qualification-loss-substitution-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    rows = tuple(row for row in rows if row.split == "train")
    other_batch = collate_witness_sources(
        tuple(scan_witness_source(row.source) for row in rows)
    )
    other_supervisor = collate_qualification_supervision(rows)
    other_hashes = tuple(sha256(row.source).hexdigest() for row in rows)
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
    )
    output = compiler(batch)
    assert output.source_sha256 != other_batch.source_sha256
    with pytest.raises(
        QualificationLossError,
        match="output source receipt",
    ):
        EFCQualificationLoss()(
            output,
            other_supervisor,
            candidate_source_sha256=other_hashes,
        )


def test_tied_machine_receives_no_exact_credit() -> None:
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
    tied = LawfulMachineProjector(sinkhorn_iterations=16)(
        torch.zeros((len(hashes), 3, 8, 8)),
        torch.zeros((len(hashes), 2, 8, 4)),
    )
    metrics = EFCQualificationLoss().exact_metrics(
        replace(output, projection=tied),
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.unhardenable_rows == len(hashes)
    assert metrics.exact_transition_cells == 0
    assert metrics.exact_observer_cells == 0
    assert metrics.exact_machines == 0


def test_tied_hankel_signatures_receive_no_exact_credit() -> None:
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
    transition = torch.nn.functional.one_hot(
        supervisor.transition_next,
        8,
    ).float()
    observer = torch.nn.functional.one_hot(
        supervisor.observer_answer,
        4,
    ).float()
    details = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=2,
        incidence=torch.tensor(prefix_shift_incidence(2)),
        temperature=0.05,
    )
    tied = replace(
        details,
        base_signatures=torch.full_like(details.base_signatures, 0.25),
        derivative_signatures=torch.full_like(
            details.derivative_signatures,
            0.25,
        ),
    )
    metrics = EFCHankelQualificationLoss().exact_metrics(
        replace(
            output,
            projection=tied.projection,
            projector_auxiliary=tied,
        ),
        supervisor,
        candidate_source_sha256=hashes,
    )
    assert metrics.exact_base_signature_cells == 0
    assert metrics.exact_derivative_signature_cells == 0
    assert (
        metrics.unhardenable_base_signature_cells
        == metrics.base_signature_cells
    )
    assert (
        metrics.unhardenable_derivative_signature_cells
        == metrics.derivative_signature_cells
    )
    assert metrics.exact_base_codebooks == 0
    assert metrics.exact_derivative_codebooks == 0


def test_hankel_objective_requires_noncollapsed_independent_branches() -> None:
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
    base_transition_logits = torch.zeros(
        (len(hashes), 3, 8, 8),
        requires_grad=True,
    )
    base_observer_logits = torch.zeros(
        (len(hashes), 2, 8, 4),
        requires_grad=True,
    )
    derivative_transition_logits = torch.zeros(
        (len(hashes), 3, 8, 8),
        requires_grad=True,
    )
    derivative_observer_logits = torch.zeros(
        (len(hashes), 2, 8, 4),
        requires_grad=True,
    )
    incidence = torch.tensor(prefix_shift_incidence(2))
    details = project_behavioral_shifts(
        base_transition=base_transition_logits.softmax(-1),
        base_observer=base_observer_logits.softmax(-1),
        derivative_transition=derivative_transition_logits.softmax(-1),
        derivative_observer=derivative_observer_logits.softmax(-1),
        max_depth=2,
        incidence=incidence,
        temperature=0.05,
    )
    losses = EFCHankelQualificationLoss()(
        replace(
            output,
            projection=details.projection,
            projector_auxiliary=details,
        ),
        supervisor,
        candidate_source_sha256=hashes,
    )
    losses.total.backward(retain_graph=True)
    collapsed_gradients = (
        base_transition_logits.grad,
        base_observer_logits.grad,
        derivative_transition_logits.grad,
        derivative_observer_logits.grad,
    )
    assert all(gradient is not None for gradient in collapsed_gradients)
    assert float(base_transition_logits.grad.abs().sum()) < 1e-6
    assert float(base_observer_logits.grad.abs().sum()) > 1e-6
    assert float(derivative_transition_logits.grad.abs().sum()) < 1e-6
    assert float(derivative_observer_logits.grad.abs().sum()) < 1e-6

    generator = torch.Generator().manual_seed(20260724)
    base_transition_logits = (
        0.1
        * torch.randn(
            (len(hashes), 3, 8, 8),
            generator=generator,
        )
    ).requires_grad_()
    base_observer_logits = (
        0.1
        * torch.randn(
            (len(hashes), 2, 8, 4),
            generator=generator,
        )
    ).requires_grad_()
    derivative_transition_logits = (
        0.1
        * torch.randn(
            (len(hashes), 3, 8, 8),
            generator=generator,
        )
    ).requires_grad_()
    derivative_observer_logits = (
        0.1
        * torch.randn(
        (len(hashes), 2, 8, 4),
        generator=generator,
        )
    ).requires_grad_()
    details = project_behavioral_shifts(
        base_transition=base_transition_logits.softmax(-1),
        base_observer=base_observer_logits.softmax(-1),
        derivative_transition=derivative_transition_logits.softmax(-1),
        derivative_observer=derivative_observer_logits.softmax(-1),
        max_depth=2,
        incidence=incidence,
        temperature=0.05,
    )
    losses = EFCHankelQualificationLoss()(
        replace(
            output,
            projection=details.projection,
            projector_auxiliary=details,
        ),
        supervisor,
        candidate_source_sha256=hashes,
    )
    losses.total.backward()
    for branch_logits in (
        base_transition_logits,
        base_observer_logits,
        derivative_transition_logits,
        derivative_observer_logits,
    ):
        assert branch_logits.grad is not None
        assert float(branch_logits.grad.abs().sum()) > 1e-6


def test_label_aware_structural_margins_shape_transient_codes() -> None:
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
    transition = torch.nn.functional.one_hot(
        supervisor.transition_next,
        8,
    ).float()
    observer = torch.nn.functional.one_hot(
        supervisor.observer_answer,
        4,
    ).float()
    details = project_behavioral_shifts(
        base_transition=transition,
        base_observer=observer,
        derivative_transition=transition,
        derivative_observer=observer,
        max_depth=2,
        incidence=torch.tensor(prefix_shift_incidence(2)),
        temperature=0.05,
    )
    collapsed_base = torch.full_like(
        details.base_signatures,
        0.25,
        requires_grad=True,
    )
    collapsed_derivative = torch.full_like(
        details.derivative_signatures,
        0.25,
        requires_grad=True,
    )
    collapsed = replace(
        details,
        base_signatures=collapsed_base,
        derivative_signatures=collapsed_derivative,
    )
    losses = EFCHankelQualificationLoss()(
        replace(
            output,
            projection=collapsed.projection,
            projector_auxiliary=collapsed,
        ),
        supervisor,
        candidate_source_sha256=hashes,
    )
    losses.syndrome_margin.backward(retain_graph=True)
    assert collapsed_derivative.grad is not None
    assert float(collapsed_derivative.grad.abs().sum()) > 0.0
    losses.state_separation.backward()
    assert collapsed_base.grad is not None
    assert float(collapsed_base.grad.abs().sum()) > 0.0
