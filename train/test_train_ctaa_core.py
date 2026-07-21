from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from train_ctaa_core import load_atomic, load_closure, train


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_train_only_loaders_reject_outcome_or_scored_fields(tmp_path: Path) -> None:
    atomic = tmp_path / "atomic.jsonl"
    write_rows(
        atomic,
        [{"action": [0, 1, 2], "state": [2, 1, 0], "context": 0, "output": [2, 1, 0]}],
    )
    assert load_atomic(atomic).action.shape == (1, 3)
    write_rows(
        atomic,
        [{"action": [0, 1, 2], "state": [2, 1, 0], "context": 0, "output": [2, 1, 0], "answer": 2}],
    )
    with pytest.raises(ValueError, match="schema"):
        load_atomic(atomic)

    closure = tmp_path / "closure.jsonl"
    write_rows(
        closure,
        [{
            "first": [0, 1, 2],
            "second": [1, 0, 2],
            "state": [2, 1, 0],
            "context": 0,
            "composed": [1, 0, 2],
            "output": [1, 2, 0],
        }],
    )
    assert load_closure(closure).first.shape == (1, 3)


def test_cpu_smoke_training_writes_weights_only_compatible_checkpoint(tmp_path: Path) -> None:
    atomic = tmp_path / "atomic.jsonl"
    closure = tmp_path / "closure.jsonl"
    output = tmp_path / "core.pt"
    write_rows(
        atomic,
        [
            {"action": [0, 1, 2], "state": [2, 1, 0], "context": index, "output": [2, 1, 0]}
            for index in range(4)
        ],
    )
    write_rows(
        closure,
        [
            {
                "first": [0, 1, 2],
                "second": [1, 0, 2],
                "state": [2, 1, 0],
                "context": index,
                "composed": [1, 0, 2],
                "output": [1, 2, 0],
            }
            for index in range(4)
        ],
    )
    report = train(
        arm="ctaa_closure",
        atomic_path=atomic,
        closure_path=closure,
        output=output,
        seed=7,
        updates=2,
        batch_size=2,
        learning_rate=1e-3,
        device_name="cpu",
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["schema"] == "ctaa_recurrent_core_v1"
    assert payload["training"]["development_access"] == 0
    assert payload["training"]["confirmation_access"] == 0
    assert report["checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        train(
            arm="ctaa_closure",
            atomic_path=atomic,
            closure_path=closure,
            output=output,
            seed=7,
            updates=1,
            batch_size=1,
            learning_rate=1e-3,
            device_name="cpu",
        )
