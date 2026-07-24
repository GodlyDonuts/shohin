from __future__ import annotations

import pytest
import torch

from episode_functor_learned_system import LearnedEFCSystem
from episode_functor_hankel_completion import (
    HankelShiftCompletionProjector,
)
from episode_functor_query_parser import NeuralOpaqueQueryParser
from episode_functor_qualification_loss import (
    EFCHankelQualificationLoss,
    HankelQualificationExactMetrics,
)
from episode_functor_qualification_trainer import (
    EFCQualificationTrainer,
    QualificationTrainerError,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
)
from pipeline.episode_functor_identifiable_board import (
    generate_pilot_rows,
    project_candidate_sources,
)
from pipeline.episode_functor_qualification_boundary import (
    collate_candidate_sources,
)
from pipeline.episode_functor_qualification_custody import (
    QualificationCustodyError,
    create_qualification_split_custody,
)
from pipeline.episode_functor_qualification_supervisor import (
    collate_qualification_supervision,
)


class _Encoded:
    def __init__(self, payload: str) -> None:
        self.ids = list(payload.encode("ascii"))
        self.offsets = [
            (index, index + 1)
            for index in range(len(self.ids))
        ]


class _ByteTokenizer:
    def encode(self, payload: str) -> _Encoded:
        return _Encoded(payload)


def _fixture():
    rows = generate_pilot_rows(
        seed="efc-qualification-trainer-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    rows = tuple(row for row in rows if row.split == "train")
    candidate = collate_candidate_sources(
        project_candidate_sources(rows, split="train"),
        tokenizer=_ByteTokenizer(),
    )
    supervisor = collate_qualification_supervision(rows)
    custody = create_qualification_split_custody(rows, split="train")
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
            width=48,
            encoder_layers=1,
            decoder_layers=1,
            heads=3,
            feedforward=96,
            sinkhorn_iterations=16,
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
        ),
    )
    return candidate, supervisor, system, custody


def test_trainer_defaults_to_verified_trunk_requirement() -> None:
    _, _, system, custody = _fixture()
    with pytest.raises(
        QualificationTrainerError,
        match="verified protected trunk",
    ):
        EFCQualificationTrainer(
            system,
            training_custody=custody,
        )


def test_one_step_updates_only_source_compiler_and_materializes_state() -> None:
    torch.manual_seed(103)
    candidate, supervisor, system, custody = _fixture()
    trainer = EFCQualificationTrainer(
        system,
        training_custody=custody,
        require_verified_trunk=False,
    )
    source_before = tuple(
        parameter.detach().clone()
        for parameter in system.source_compiler.parameters()
    )
    query_before = tuple(
        parameter.detach().clone()
        for parameter in system.query_parser.parameters()
    )
    receipt = trainer.train_step(candidate, supervisor)
    assert receipt.loss > 0.0
    assert receipt.gradient_norm > 0.0
    assert receipt.trainable_parameters == (
        system.source_compiler.parameter_count()
    )
    assert receipt.optimizer_state_bytes > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            source_before,
            system.source_compiler.parameters(),
            strict=True,
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            query_before,
            system.query_parser.parameters(),
            strict=True,
        )
    )
    evaluated = trainer.evaluate(
        candidate,
        supervisor,
        custody=custody,
    )
    assert evaluated.split == "train"
    assert evaluated.exact_metrics.rows == supervisor.batch_size


def test_trainer_fails_before_update_on_source_receipt_mismatch() -> None:
    candidate, supervisor, system, custody = _fixture()
    trainer = EFCQualificationTrainer(
        system,
        training_custody=custody,
        require_verified_trunk=False,
    )
    changed = list(candidate.source_sha256)
    changed[0] = "0" * 64
    object.__setattr__(candidate, "source_sha256", tuple(changed))
    with pytest.raises(ValueError, match="leaves split custody"):
        trainer.train_step(candidate, supervisor)


def test_hankel_trainer_updates_both_code_branches_under_source_only_custody() -> None:
    torch.manual_seed(337)
    candidate, supervisor, _, custody = _fixture()
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
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
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
        ),
    )
    trainer = EFCQualificationTrainer(
        system,
        objective=EFCHankelQualificationLoss(),
        training_custody=custody,
        require_verified_trunk=False,
    )
    base_before = tuple(
        parameter.detach().clone()
        for parameter in system.source_compiler.projector.base.parameters()
    )
    derivative_before = tuple(
        parameter.detach().clone()
        for parameter in system.source_compiler.projector.derivative.parameters()
    )
    query_before = tuple(
        parameter.detach().clone()
        for parameter in system.query_parser.parameters()
    )
    receipt = trainer.train_step(candidate, supervisor)
    assert not system.source_compiler.training
    assert not system.query_parser.training
    assert isinstance(
        receipt.exact_metrics,
        HankelQualificationExactMetrics,
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            base_before,
            system.source_compiler.projector.base.parameters(),
            strict=True,
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            derivative_before,
            system.source_compiler.projector.derivative.parameters(),
            strict=True,
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            query_before,
            system.query_parser.parameters(),
            strict=True,
        )
    )


def test_hankel_projector_cannot_train_with_machine_only_objective() -> None:
    candidate, supervisor, _, custody = _fixture()
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
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
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
        ),
    )
    with pytest.raises(
        QualificationTrainerError,
        match="must be paired",
    ):
        EFCQualificationTrainer(
            system,
            training_custody=custody,
            require_verified_trunk=False,
        )


def test_hankel_trainer_rejects_incidence_mutation_before_update() -> None:
    candidate, supervisor, _, custody = _fixture()
    system = LearnedEFCSystem(
        source_compiler=ProofCarryingWitnessCompiler(
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
        ),
        query_parser=NeuralOpaqueQueryParser(
            width=48,
            layers=1,
            heads=3,
            feedforward=96,
            max_steps=8,
        ),
    )
    trainer = EFCQualificationTrainer(
        system,
        objective=EFCHankelQualificationLoss(),
        training_custody=custody,
        require_verified_trunk=False,
    )
    with torch.no_grad():
        system.source_compiler.projector.shift_incidence[0, 0] += 1
    with pytest.raises(
        QualificationTrainerError,
        match="incidence changed",
    ):
        trainer.train_step(candidate, supervisor)


def test_optimizer_rejects_nontrain_split_custody() -> None:
    _, _, system, _ = _fixture()
    rows = generate_pilot_rows(
        seed="efc-qualification-trainer-eval-custody-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    mechanics = create_qualification_split_custody(
        rows,
        split="mechanics",
    )
    with pytest.raises(
        QualificationCustodyError,
        match="restricted to the train split",
    ):
        EFCQualificationTrainer(
            system,
            training_custody=mechanics,
            require_verified_trunk=False,
        )
