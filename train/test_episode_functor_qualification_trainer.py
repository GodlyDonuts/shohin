from __future__ import annotations

import pytest
import torch

from episode_functor_learned_system import LearnedEFCSystem
from episode_functor_query_parser import NeuralOpaqueQueryParser
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
    return candidate, supervisor, system


def test_trainer_defaults_to_verified_trunk_requirement() -> None:
    _, _, system = _fixture()
    with pytest.raises(
        QualificationTrainerError,
        match="verified protected trunk",
    ):
        EFCQualificationTrainer(system)


def test_one_step_updates_only_source_compiler_and_materializes_state() -> None:
    torch.manual_seed(103)
    candidate, supervisor, system = _fixture()
    trainer = EFCQualificationTrainer(
        system,
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
    evaluated = trainer.evaluate(candidate, supervisor)
    assert evaluated.rows == supervisor.batch_size


def test_trainer_fails_before_update_on_source_receipt_mismatch() -> None:
    candidate, supervisor, system = _fixture()
    trainer = EFCQualificationTrainer(
        system,
        require_verified_trunk=False,
    )
    changed = list(candidate.source_sha256)
    changed[0] = "0" * 64
    object.__setattr__(candidate, "source_sha256", tuple(changed))
    with pytest.raises(ValueError, match="source receipts differ"):
        trainer.train_step(candidate, supervisor)
