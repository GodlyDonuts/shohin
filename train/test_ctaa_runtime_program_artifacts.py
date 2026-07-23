from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import pytest
from tokenizers import Tokenizer

import ctaa_runtime_program_artifacts as artifact_loader
from ctaa_run_contract import canonical_json
from ctaa_runtime_execution_projection import make_execution_projection
from ctaa_runtime_program_artifacts import (
    ProgramArtifactLoadError,
    load_runtime_program_artifacts,
)
from test_build_ctaa_runtime_intervention_plan import (
    _build,
    frozen_inputs as _frozen_inputs_fixture,  # noqa: F401
)


def _locked_write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    path.chmod(0o444)
    return path


def _rows_raw(rows: list[dict[str, object]]) -> bytes:
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode("ascii")


def _rebind_tokenizer(projection: dict[str, object], raw: bytes) -> dict[str, object]:
    changed = deepcopy(projection)
    changed["tokenizer_sha256"] = hashlib.sha256(raw).hexdigest()
    unsigned = {
        key: value for key, value in changed.items() if key != "projection_sha256"
    }
    changed["projection_sha256"] = hashlib.sha256(
        canonical_json(unsigned).encode("ascii")
    ).hexdigest()
    return changed


@pytest.fixture(scope="module")
def program_inputs(request: pytest.FixtureRequest, tmp_path_factory):
    frozen_inputs = request.getfixturevalue("_frozen_inputs_fixture")
    root = tmp_path_factory.mktemp("ctaa-program-artifacts")
    plan = _build(frozen_inputs, root / "runtime-plan.json")
    projection = make_execution_projection(plan)
    board_rows = {
        row["family_id"]: row
        for row in (
            json.loads(line)
            for line in Path(frozen_inputs["program_path"]).read_text().splitlines()
        )
    }
    selected_rows = [deepcopy(board_rows[anchor.family_id]) for anchor in plan.anchors]
    selected_path = _locked_write(
        root / "selected-programs.jsonl", _rows_raw(selected_rows)
    )
    return {
        "root": root,
        "projection": projection,
        "plan": plan,
        "rows": selected_rows,
        "program_path": selected_path,
        "tokenizer_path": Path(frozen_inputs["tokenizer_path"]),
    }


def _load(program_inputs, **overrides):
    return load_runtime_program_artifacts(
        projection=overrides.get("projection", program_inputs["projection"]),
        program_source_path=overrides.get(
            "program_source_path", program_inputs["program_path"]
        ),
        tokenizer_path=overrides.get(
            "tokenizer_path", program_inputs["tokenizer_path"]
        ),
    )


@pytest.fixture(scope="module")
def loaded(program_inputs):
    return _load(program_inputs)


def test_loader_builds_exact_immutable_registry_from_frozen_bytes(
    program_inputs, loaded
) -> None:
    projection = program_inputs["projection"]
    expected: list[str] = []
    for anchor in projection["anchors"]:
        digest = anchor["program_source_sha256"]
        if digest not in expected:
            expected.append(digest)
    for attempt in projection["attempts"]:
        if attempt["operation"] in artifact_loader._SOURCE_OPERATIONS:
            digest = attempt["resulting_program_source_sha256"]
            if digest not in expected:
                expected.append(digest)
    assert isinstance(loaded, MappingProxyType)
    assert list(loaded) == expected
    assert all(key == artifact.source_sha256 for key, artifact in loaded.items())

    parent = projection["anchors"][0]
    artifact = loaded[parent["program_source_sha256"]]
    tokenizer = Tokenizer.from_file(str(program_inputs["tokenizer_path"]))
    assert artifact.token_ids == tuple(tokenizer.encode(artifact.source.decode()).ids)
    with pytest.raises(TypeError):
        loaded["0" * 64] = artifact


def test_loader_uses_only_v2_minimal_anchor_bindings(program_inputs) -> None:
    projection = program_inputs["projection"]
    assert all(
        set(anchor) == {"anchor_id", "program_source_sha256", "packet_sha256"}
        and anchor["anchor_id"].startswith("oa")
        for anchor in projection["anchors"]
    )
    serialized = canonical_json(projection)
    for anchor in program_inputs["plan"].anchors:
        assert anchor.family_id not in serialized
        assert anchor.anchor_id not in serialized


def test_caller_supplied_token_substitution_is_rejected(
    tmp_path: Path, program_inputs
) -> None:
    rows = deepcopy(program_inputs["rows"])
    rows[0]["token_ids"] = [999999]
    path = _locked_write(tmp_path / "token-substitution.jsonl", _rows_raw(rows))
    with pytest.raises(ProgramArtifactLoadError, match="schema differs"):
        _load(program_inputs, program_source_path=path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_source_coverage_must_be_exact(
    tmp_path: Path, program_inputs, mutation: str
) -> None:
    rows = deepcopy(program_inputs["rows"])
    if mutation == "missing":
        rows.pop()
    else:
        rows.append(
            {
                "family_id": "Dhhh99999999",
                "program_source": rows[0]["program_source"],
            }
        )
    path = _locked_write(tmp_path / f"{mutation}.jsonl", _rows_raw(rows))
    with pytest.raises(ProgramArtifactLoadError, match="coverage differs"):
        _load(program_inputs, program_source_path=path)


def test_malformed_source_is_rejected(tmp_path: Path, program_inputs) -> None:
    path = _locked_write(
        tmp_path / "malformed.jsonl",
        b'{"family_id":"Dhhh00000000","family_id":"Dhhh00000000"}\n',
    )
    with pytest.raises(ProgramArtifactLoadError, match="duplicate keys|malformed"):
        _load(program_inputs, program_source_path=path)


def test_writable_source_is_rejected(tmp_path: Path, program_inputs) -> None:
    path = tmp_path / "writable.jsonl"
    path.write_bytes(Path(program_inputs["program_path"]).read_bytes())
    path.chmod(0o644)
    with pytest.raises(ProgramArtifactLoadError, match="immutable"):
        _load(program_inputs, program_source_path=path)


def test_writable_tokenizer_is_rejected(tmp_path: Path, program_inputs) -> None:
    path = tmp_path / "writable-tokenizer.json"
    path.write_bytes(Path(program_inputs["tokenizer_path"]).read_bytes())
    path.chmod(0o644)
    with pytest.raises(ProgramArtifactLoadError, match="immutable"):
        _load(program_inputs, tokenizer_path=path)


def test_symlink_source_is_rejected(tmp_path: Path, program_inputs) -> None:
    target = _locked_write(
        tmp_path / "target.jsonl", Path(program_inputs["program_path"]).read_bytes()
    )
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ProgramArtifactLoadError, match="symlinked"):
        _load(program_inputs, program_source_path=link)


def test_symlink_tokenizer_is_rejected(tmp_path: Path, program_inputs) -> None:
    link = tmp_path / "tokenizer-link.json"
    link.symlink_to(program_inputs["tokenizer_path"])
    with pytest.raises(ProgramArtifactLoadError, match="symlinked"):
        _load(program_inputs, tokenizer_path=link)


def test_hardlink_source_is_rejected(tmp_path: Path, program_inputs) -> None:
    source = _locked_write(
        tmp_path / "hardlink-source.jsonl",
        Path(program_inputs["program_path"]).read_bytes(),
    )
    link = tmp_path / "hardlink.jsonl"
    os.link(source, link)
    with pytest.raises(ProgramArtifactLoadError, match="single-link"):
        _load(program_inputs, program_source_path=link)


def test_hardlink_tokenizer_is_rejected(tmp_path: Path, program_inputs) -> None:
    source = _locked_write(
        tmp_path / "hardlink-tokenizer-source.json",
        Path(program_inputs["tokenizer_path"]).read_bytes(),
    )
    link = tmp_path / "hardlink-tokenizer.json"
    os.link(source, link)
    with pytest.raises(ProgramArtifactLoadError, match="single-link"):
        _load(program_inputs, tokenizer_path=link)


def test_tokenizer_byte_substitution_breaks_frozen_hash(
    tmp_path: Path, program_inputs
) -> None:
    raw = Path(program_inputs["tokenizer_path"]).read_bytes() + b"\n"
    path = _locked_write(tmp_path / "substituted-tokenizer.json", raw)
    with pytest.raises(ProgramArtifactLoadError, match="tokenizer source hash differs"):
        _load(program_inputs, tokenizer_path=path)


def test_hash_bound_malformed_tokenizer_is_rejected(
    tmp_path: Path, program_inputs
) -> None:
    raw = b'{"not":"a tokenizer"}'
    path = _locked_write(tmp_path / "malformed-tokenizer.json", raw)
    projection = _rebind_tokenizer(program_inputs["projection"], raw)
    with pytest.raises(ProgramArtifactLoadError, match="tokenizer source is malformed"):
        _load(program_inputs, tokenizer_path=path, projection=projection)


def test_parent_source_hash_substitution_is_rejected(
    tmp_path: Path, program_inputs
) -> None:
    rows = deepcopy(program_inputs["rows"])
    replacement = next(
        row["program_source"]
        for row in rows[1:]
        if row["program_source"] != rows[0]["program_source"]
    )
    rows[0]["program_source"] = replacement
    path = _locked_write(tmp_path / "source-swap.jsonl", _rows_raw(rows))
    with pytest.raises(ProgramArtifactLoadError, match="hash coverage differs"):
        _load(program_inputs, program_source_path=path)


@pytest.mark.parametrize("leaked_key", ["query_source", "answer"])
def test_query_and_answer_fields_are_rejected(
    tmp_path: Path, program_inputs, leaked_key: str
) -> None:
    rows = deepcopy(program_inputs["rows"])
    rows[0][leaked_key] = "sealed material"
    path = _locked_write(tmp_path / f"leak-{leaked_key}.jsonl", _rows_raw(rows))
    with pytest.raises(ProgramArtifactLoadError, match="schema differs"):
        _load(program_inputs, program_source_path=path)


def test_input_row_order_cannot_change_registry_order(
    tmp_path: Path, program_inputs, loaded
) -> None:
    path = _locked_write(
        tmp_path / "reversed.jsonl", _rows_raw(list(reversed(program_inputs["rows"])))
    )
    reordered = _load(program_inputs, program_source_path=path)
    assert list(reordered) == list(loaded)
    assert dict(reordered) == dict(loaded)


def test_legacy_family_labels_have_no_authority(
    tmp_path: Path, program_inputs, loaded
) -> None:
    rows = deepcopy(program_inputs["rows"])
    for index, row in enumerate(rows):
        row["family_id"] = f"ignored-label-{index:06d}"
    path = _locked_write(tmp_path / "relabeled.jsonl", _rows_raw(rows))
    relabeled = _load(program_inputs, program_source_path=path)
    assert dict(relabeled) == dict(loaded)


def test_nondeterministic_tokenizer_is_rejected(
    monkeypatch: pytest.MonkeyPatch, program_inputs
) -> None:
    class _Encoding:
        def __init__(self, token: int) -> None:
            self.ids = [token]

    class _NondeterministicTokenizer:
        calls = 0

        @classmethod
        def from_str(cls, raw: str):
            assert raw
            return cls()

        def encode(self, text: str):
            assert text
            type(self).calls += 1
            return _Encoding(type(self).calls % 2)

    monkeypatch.setattr(artifact_loader, "Tokenizer", _NondeterministicTokenizer)
    with pytest.raises(ProgramArtifactLoadError, match="nondeterministic"):
        _load(program_inputs)
