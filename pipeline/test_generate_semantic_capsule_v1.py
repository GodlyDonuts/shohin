#!/usr/bin/env python3
"""Small deterministic contract tests for semantic-capsule generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return " ".join(text.lower().split())


def main():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, heldout = directory / "train.jsonl", directory / "heldout.jsonl"
        subprocess.run([
            sys.executable, "pipeline/generate_semantic_capsule_v1.py",
            "--train-out", str(train), "--heldout-out", str(heldout),
            "--train-per-domain", "4", "--heldout-per-domain", "3", "--seed", "17",
        ], cwd=ROOT, check=True)
        rows = [json.loads(line) for line in train.read_text().splitlines()]
        episodes = [json.loads(line) for line in heldout.read_text().splitlines()]
        assert len(rows) > 0 and len(episodes) == 9
        assert all(row["training_group"] == "semantic_capsule" for row in rows)
        assert all(row["response"].startswith("<think>") for row in rows)
        assert {row["mode"] for row in rows} == {"write", "update", "repair", "readout"}
        train_prompts = {normalized(row["question"]) for row in rows}
        eval_prompts = {normalized(row["initial"]["prompt"]) for row in episodes}
        assert not train_prompts & eval_prompts
        assert {episode["regime"] for episode in episodes} == {"semantic_len4", "semantic_len8", "semantic_len12"}
    print("semantic capsule generator tests passed")


if __name__ == "__main__":
    main()
