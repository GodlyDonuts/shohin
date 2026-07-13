#!/usr/bin/env python3
"""Small end-to-end generator contracts for digitwise recurrent data."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    train, heldout, report = directory / "train.jsonl", directory / "heldout.jsonl", directory / "report.json"
    subprocess.run([
        sys.executable, "pipeline/generate_digitwise_recurrent_v1.py",
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(report),
        "--train-episodes", "24", "--heldout-per-regime", "3", "--seed", "53",
    ], cwd=ROOT, check=True)
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    episodes = [json.loads(line) for line in heldout.read_text().splitlines()]
    summary = json.loads(report.read_text())
    assert summary["schema"] == "shohin-digitwise-recurrent-v1"
    assert summary["heldout_by_regime"] == {
        "fit_w4": 3, "fit_w6": 3, "value_ood_w4": 3, "value_ood_w6": 3, "width_ood_w8": 3,
    }
    assert {row["kind"] for row in rows} == {"transition", "digit", "final"}
    assert {row["training_group"] for row in rows} == {"digitwise_recurrent"}
    assert all(row["question"] == row["completion_prompt"] and row["response"] for row in rows)
    assert {episode["width"] for episode in episodes} == {4, 6, 8}
    assert all(episode["expected_answer"] != episode["counterfactual"]["expected_answer"] for episode in episodes)
print("digitwise recurrent generator checks: passed")
