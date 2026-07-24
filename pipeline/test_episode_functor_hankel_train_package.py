from __future__ import annotations

import json

import pytest

from pipeline.episode_functor_hankel_train_package import (
    HankelTrainPackageError,
    build_hankel_train_package,
    load_hankel_train_package,
)
from pipeline.episode_functor_identifiable_board import (
    generate_pilot_rows,
    project_candidate_sources,
)
from pipeline.episode_functor_qualification_boundary import (
    collate_candidate_sources,
    tokenizer_runtime_sha256,
)
from pipeline.episode_functor_qualification_custody import (
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
    def to_str(self) -> str:
        return '{"kind":"test-byte-tokenizer"}'

    def encode(self, payload: str) -> _Encoded:
        return _Encoded(payload)


def _fixture():
    rows = generate_pilot_rows(
        seed="efc-hankel-package-test-v1",
        counts={
            "train": 1,
            "mechanics": 1,
            "development": 1,
            "confirmation": 1,
        },
    )
    train = tuple(row for row in rows if row.split == "train")
    tokenizer = _ByteTokenizer()
    candidate = collate_candidate_sources(
        project_candidate_sources(train, split="train"),
        tokenizer=tokenizer,
        tokenizer_artifact_sha256="a" * 64,
        expected_tokenizer_runtime_sha256=tokenizer_runtime_sha256(
            tokenizer
        ),
    )
    supervisor = collate_qualification_supervision(train)
    custody = create_qualification_split_custody(
        train,
        split="train",
        candidate=candidate,
    )
    return train, tokenizer, candidate, supervisor, custody


def test_train_package_round_trips_by_recollating_raw_sources(tmp_path) -> None:
    train, tokenizer, candidate, supervisor, custody = _fixture()
    output = tmp_path / "package"
    receipt = build_hankel_train_package(
        output,
        sources=tuple(row.source for row in train),
        candidate=candidate,
        supervisor=supervisor,
        custody=custody,
    )
    loaded_candidate, loaded_supervisor, loaded_custody, loaded = (
        load_hankel_train_package(
            output,
            tokenizer=tokenizer,
            expected_package_sha256=receipt.package_sha256,
        )
    )
    assert loaded == receipt
    assert (
        loaded_candidate.candidate_input_manifest_sha256
        == candidate.candidate_input_manifest_sha256
    )
    assert loaded_supervisor.source_sha256 == supervisor.source_sha256
    assert loaded_custody == custody
    assert not any(
        name in path.name
        for path in output.rglob("*")
        for name in ("mechanics", "development", "confirmation")
    )


def test_train_package_rejects_mutation_and_overwrite(tmp_path) -> None:
    train, tokenizer, candidate, supervisor, custody = _fixture()
    output = tmp_path / "package"
    receipt = build_hankel_train_package(
        output,
        sources=tuple(row.source for row in train),
        candidate=candidate,
        supervisor=supervisor,
        custody=custody,
    )
    with pytest.raises(FileExistsError):
        build_hankel_train_package(
            output,
            sources=tuple(row.source for row in train),
            candidate=candidate,
            supervisor=supervisor,
            custody=custody,
        )
    manifest = output / "candidate_receipt.json"
    manifest.chmod(0o644)
    value = json.loads(manifest.read_text())
    value["source_sha256"][0] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(
        HankelTrainPackageError,
        match="file differs",
    ):
        load_hankel_train_package(
            output,
            tokenizer=tokenizer,
            expected_package_sha256=receipt.package_sha256,
        )


def test_train_package_rejects_unlisted_file(tmp_path) -> None:
    train, tokenizer, candidate, supervisor, custody = _fixture()
    output = tmp_path / "package"
    receipt = build_hankel_train_package(
        output,
        sources=tuple(row.source for row in train),
        candidate=candidate,
        supervisor=supervisor,
        custody=custody,
    )
    output.chmod(0o755)
    (output / "unlisted.txt").write_text(
        "not authorized",
        encoding="ascii",
    )
    with pytest.raises(
        HankelTrainPackageError,
        match="directory closure",
    ):
        load_hankel_train_package(
            output,
            tokenizer=tokenizer,
            expected_package_sha256=receipt.package_sha256,
        )
