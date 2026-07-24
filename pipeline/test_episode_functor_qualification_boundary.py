from __future__ import annotations

from dataclasses import dataclass, fields

import pytest
import torch

from pipeline.episode_functor_identifiable_board import (
    CandidateSource,
    GrammarFactors,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)
from pipeline.episode_functor_qualification_boundary import (
    QualificationBoundaryError,
    collate_candidate_sources,
)


@dataclass(frozen=True)
class _Encoding:
    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]


class _ByteTokenizer:
    def encode(self, text: str) -> _Encoding:
        payload = text.encode("ascii")
        return _Encoding(
            ids=tuple(payload),
            offsets=tuple(
                (index, index + 1)
                for index in range(len(payload))
            ),
        )


class _GapTokenizer:
    def encode(self, text: str) -> _Encoding:
        payload = text.encode("ascii")
        return _Encoding(
            ids=(int(payload[0]),),
            offsets=((0, 1),),
        )


def _candidate() -> CandidateSource:
    machine = generate_machine(
        seed="efc-qualification-boundary-v1",
        split="train",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-qualification-boundary-v1",
        split="train",
        index=0,
    )
    return CandidateSource(
        encode_source(evidence, GrammarFactors(0, 0, 0))
    )


def test_candidate_batch_contains_only_source_derived_inputs() -> None:
    candidate = _candidate()
    batch = collate_candidate_sources(
        (candidate,),
        tokenizer=_ByteTokenizer(),
    )
    assert tuple(field.name for field in fields(batch)) == (
        "witness",
        "trunk",
        "source_sha256",
    )
    assert batch.trunk.payloads == (candidate.source,)
    assert torch.equal(
        batch.witness.pointer.byte_ids[0],
        batch.trunk.token_ids[0],
    )
    assert batch.trunk.token_valid.all()
    assert batch.witness.pointer.unique_key_valid.sum().item() == 13


def test_poisoned_candidate_type_and_bad_offsets_fail_closed() -> None:
    candidate = _candidate()

    @dataclass(frozen=True)
    class PoisonedCandidate:
        source: bytes
        family: str

    with pytest.raises(
        QualificationBoundaryError,
        match="exact CandidateSource",
    ):
        collate_candidate_sources(
            (PoisonedCandidate(candidate.source, "leak"),),
            tokenizer=_ByteTokenizer(),
        )
    with pytest.raises(
        QualificationBoundaryError,
        match="do not partition",
    ):
        collate_candidate_sources(
            (candidate,),
            tokenizer=_GapTokenizer(),
        )
