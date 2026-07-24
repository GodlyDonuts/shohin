from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from episode_functor_constrained_transport import (  # noqa: E402
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_pointer_compiler import (  # noqa: E402
    MAX_KEY_OCCURRENCES,
    MAX_UNIQUE_KEYS,
)
from episode_functor_witness_compiler import (  # noqa: E402
    MAX_RECORDS,
    OCCURRENCE_ROLES,
    ProofCarryingWitnessCompiler,
    RECORD_OBSERVATION,
    RECORD_TRANSITION,
    assemble_relation_evidence,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    GrammarFactors,
    SOURCE_FACTOR_COMBINATIONS,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _source(values: tuple[int, int, int]) -> bytes:
    machine = generate_machine(
        seed="efc-witness-compiler-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-witness-compiler-test-v1",
        split="mechanics",
        index=0,
    )
    return encode_source(evidence, GrammarFactors(*values))


@pytest.mark.parametrize("values", SOURCE_FACTOR_COMBINATIONS)
def test_generic_record_candidates_cover_every_key_without_roles(
    values: tuple[int, int, int],
) -> None:
    scanned = scan_witness_source(_source(values))
    assert 1 <= len(scanned.record_spans) <= MAX_RECORDS
    assert len(scanned.occurrence_to_record) == 99
    assert len(scanned.pointer.unique_keys) == 13
    for occurrence, record in enumerate(scanned.occurrence_to_record):
        key_start, key_end = scanned.pointer.spans[occurrence]
        record_start, record_end = scanned.record_spans[record]
        assert record_start <= key_start < key_end <= record_end


def test_witness_batch_is_target_free_and_bounded() -> None:
    batch = collate_witness_sources(
        [
            scan_witness_source(_source((0, 0, 0))),
            scan_witness_source(_source((1, 1, 1))),
        ]
    )
    assert batch.record_bounds.shape == (2, MAX_RECORDS, 2)
    assert batch.record_valid.shape == (2, MAX_RECORDS)
    assert batch.occurrence_to_record.shape == (
        2,
        MAX_KEY_OCCURRENCES,
    )
    assert batch.pointer.unique_key_valid.sum(1).tolist() == [13, 13]


def _one_hot_logits(
    indices: torch.Tensor,
    classes: int,
) -> torch.Tensor:
    result = torch.full((*indices.shape, classes), -20.0)
    return result.scatter(
        -1,
        indices.long().unsqueeze(-1),
        20.0,
    )


def test_one_key_transport_causally_indexes_transition_axes() -> None:
    batch = 1
    record_type = _one_hot_logits(
        torch.tensor(
            [[RECORD_TRANSITION, RECORD_OBSERVATION] + [0] * 62]
        ),
        5,
    )
    occurrence_roles = _one_hot_logits(
        torch.tensor(
            [
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    *([6] * (MAX_KEY_OCCURRENCES - 5)),
                ]
            ]
        ),
        OCCURRENCE_ROLES,
    )
    answer = _one_hot_logits(
        torch.tensor([[0, 2] + [0] * 62]),
        PRIMARY_ANSWERS,
    )
    occurrence_valid = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.bool,
    )
    occurrence_valid[:, :5] = True
    occurrence_to_record = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.long,
    )
    occurrence_to_record[:, 3:5] = 1
    occurrence_to_unique = torch.zeros(
        (batch, MAX_KEY_OCCURRENCES),
        dtype=torch.long,
    )
    occurrence_to_unique[0, :5] = torch.tensor((8, 0, 1, 11, 2))
    source_valid = torch.zeros((batch, MAX_UNIQUE_KEYS), dtype=torch.bool)
    source_valid[:, :13] = True
    assignment = torch.full((batch, 32, MAX_UNIQUE_KEYS), -20.0)
    semantic_slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(16 + index for index in range(PRIMARY_ACTIONS))
        + tuple(24 + index for index in range(PRIMARY_OBSERVERS))
    )
    for unique, slot in enumerate(semantic_slots):
        assignment[0, slot, unique] = 20.0

    before = assemble_relation_evidence(
        record_type_logits=record_type,
        occurrence_role_logits=occurrence_roles,
        answer_logits=answer,
        occurrence_valid=occurrence_valid,
        occurrence_to_record=occurrence_to_record,
        occurrence_to_unique=occurrence_to_unique,
        source_unique_key_valid=source_valid,
        key_assignment_logits=assignment,
    )
    assert before.transition_logits[0].argmax().item() == (
        0 * 64 + 0 * 8 + 1
    )
    assert before.observer_logits[0].argmax().item() == (
        0 * 32 + 2 * 4 + 2
    )

    swapped = assignment.clone()
    swapped[0, 16, 8] = -20.0
    swapped[0, 17, 9] = -20.0
    swapped[0, 16, 9] = 20.0
    swapped[0, 17, 8] = 20.0
    after = assemble_relation_evidence(
        record_type_logits=record_type,
        occurrence_role_logits=occurrence_roles,
        answer_logits=answer,
        occurrence_valid=occurrence_valid,
        occurrence_to_record=occurrence_to_record,
        occurrence_to_unique=occurrence_to_unique,
        source_unique_key_valid=source_valid,
        key_assignment_logits=swapped,
    )
    assert after.transition_logits[0].argmax().item() == (
        1 * 64 + 0 * 8 + 1
    )


def test_witness_compiler_is_attached_small_and_law_projected() -> None:
    torch.manual_seed(53)
    batch = collate_witness_sources(
        [
            scan_witness_source(_source((0, 0, 0))),
            scan_witness_source(_source((1, 1, 1))),
        ]
    )
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
    )
    output = compiler(batch)
    assert output.relation_evidence.transition_logits.shape == (
        2,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    assert output.relation_evidence.observer_logits.shape == (
        2,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    assert output.key_assignment_logits.shape == (
        2,
        32,
        MAX_UNIQUE_KEYS,
    )
    assert compiler.parameter_count() < 1_000_000
    assert torch.allclose(
        output.projection.transition_transport.sum(-2),
        torch.ones(2, PRIMARY_ACTIONS, PRIMARY_STATES),
        atol=2e-4,
    )
    assert torch.allclose(
        output.projection.observer_transport.sum(-2),
        torch.full(
            (2, PRIMARY_OBSERVERS, PRIMARY_ANSWERS),
            2.0,
        ),
        atol=2e-4,
    )
    loss = sum(
        value.square().mean()
        for value in (
            output.projection.machine.action_next,
            output.projection.machine.observer_answer,
            output.key_assignment_logits,
            output.raw_key_assignment_logits,
            output.record_type_logits,
            output.occurrence_role_logits,
            output.answer_logits,
        )
    )
    loss.backward()
    missing = [
        name
        for name, parameter in compiler.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in compiler.parameters()
        if parameter.grad is not None
    )
