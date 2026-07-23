from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter

import pytest

from pipeline.build_ctaa_board_v2 import BoardSizes, build_board


TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reduced_board_writer_is_single_use_sealed_and_hash_complete(tmp_path: Path) -> None:
    output = tmp_path / "board"
    sizes = BoardSizes(
        atomic_contexts=1,
        closure_contexts=1,
        compiler_per_depth=18,
        long_per_class_depth_cell=288,
        diagnostics_per_class_depth=1,
        name_pool_per_axis=32,
    )
    report = build_board(991_027, output, TOKENIZER, sizes=sizes)
    assert report["all_gates_pass"]
    assert report["counts"]["train_atomic"] == 243
    assert report["counts"]["train_closure"] == 945
    assert report["counts"]["train_compiler"] == 144
    assert report["counts"]["development"] == 13_824
    assert report["counts"]["confirmation"] == 13_824
    assert report["counts"]["development_interventions"] == 5_202
    assert report["counts"]["confirmation_interventions"] == 5_202
    assert (output / "confirmation_program.jsonl").stat().st_mode & 0o777 == 0o400
    assert (output / "confirmation_query.jsonl").stat().st_mode & 0o777 == 0o400
    assert (output / "confirmation_oracle.jsonl").stat().st_mode & 0o777 == 0o400
    assert (output / "confirmation_intervention_program.jsonl").stat().st_mode & 0o777 == 0o400
    assert (output / "confirmation_intervention_query.jsonl").stat().st_mode & 0o777 == 0o400
    assert (output / "confirmation_intervention_oracle.jsonl").stat().st_mode & 0o777 == 0o400
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(
        sha256_file(output / name) == digest
        for name, digest in manifest["files"].items()
    )
    with (output / "train_compiler.jsonl").open() as handle:
        first = json.loads(next(handle))
    assert "prefix_states" not in first
    assert "terminal_state" not in first
    assert "answer" not in first
    development_oracle = [
        json.loads(line)
        for line in (output / "development_oracle.jsonl").read_text().splitlines()
    ]
    joint = Counter(
        (
            row["factorial_cell"],
            row["program_class"],
            row["depth"],
            row["renderer"],
            row["query_position"],
            tuple(row["initial_state"]),
        )
        for row in development_oracle
    )
    assert len(joint) == 8 * 3 * 2 * 16 * 18
    assert set(joint.values()) == {1}
    with pytest.raises(FileExistsError):
        build_board(991_027, output, TOKENIZER, sizes=sizes)
