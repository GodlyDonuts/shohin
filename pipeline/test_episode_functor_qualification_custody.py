from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pipeline.episode_functor_identifiable_board import generate_pilot_rows
from pipeline.episode_functor_qualification_boundary import (
    collate_candidate_sources,
)
from pipeline.episode_functor_identifiable_board import (
    project_candidate_sources,
)
from pipeline.episode_functor_qualification_custody import (
    QualificationCustodyError,
    create_qualification_split_custody,
)
from pipeline.episode_functor_qualification_supervisor import (
    collate_qualification_supervision,
)


def _rows():
    return generate_pilot_rows(
        seed="efc-qualification-custody-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
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


def test_split_custody_is_canonical_and_train_access_is_explicit() -> None:
    train = create_qualification_split_custody(_rows(), split="train")
    mechanics = create_qualification_split_custody(
        _rows(),
        split="mechanics",
    )
    train.assert_training_split()
    with pytest.raises(
        QualificationCustodyError,
        match="restricted to the train split",
    ):
        mechanics.assert_training_split()
    assert train.receipt_sha256 != mechanics.receipt_sha256
    assert train.to_json_bytes().endswith(b"\n")


def test_split_custody_rejects_receipt_mutation() -> None:
    receipt = create_qualification_split_custody(
        _rows(),
        split="train",
    )
    with pytest.raises(
        QualificationCustodyError,
        match="receipt hash",
    ):
        replace(receipt, receipt_sha256="0" * 64)


def test_split_custody_binds_every_supervisor_target_tensor() -> None:
    rows = tuple(row for row in _rows() if row.split == "train")
    receipt = create_qualification_split_custody(rows, split="train")
    candidate = collate_candidate_sources(
        project_candidate_sources(rows, split="train"),
        tokenizer=_ByteTokenizer(),
    )
    supervisor = collate_qualification_supervision(rows)
    receipt.assert_batches(candidate, supervisor)

    for field_name in (
        "key_slot_to_unique",
        "record_type",
        "record_label_valid",
        "occurrence_role",
        "occurrence_label_valid",
        "record_answer",
        "answer_label_valid",
        "transition_next",
        "transition_exposed",
        "observer_answer",
        "observer_exposed",
    ):
        tensor = getattr(supervisor, field_name)
        changed = tensor.clone()
        flat = changed.reshape(-1)
        if changed.dtype == torch.bool:
            flat[0] = ~flat[0]
        elif field_name == "key_slot_to_unique":
            changed[:, :2] = changed[:, :2].flip(-1)
        elif field_name == "transition_next":
            flat[0] = (flat[0] + 1) % 8
        elif field_name in {"record_answer", "observer_answer"}:
            flat[0] = (flat[0] + 1) % 4
        else:
            flat[0] = flat[0] + 1
        tampered = replace(supervisor, **{field_name: changed})
        with pytest.raises(
            QualificationCustodyError,
            match="leaves split custody",
        ):
            receipt.assert_batches(candidate, tampered)
