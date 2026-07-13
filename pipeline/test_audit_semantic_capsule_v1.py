#!/usr/bin/env python3
"""End-to-end small audit contract for semantic-capsule generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, heldout, report = directory / "train.jsonl", directory / "heldout.jsonl", directory / "report.json"
        subprocess.run([
            sys.executable, "pipeline/generate_semantic_capsule_v1.py",
            "--train-out", str(train), "--heldout-out", str(heldout),
            "--train-per-domain", "5", "--heldout-per-domain", "4", "--seed", "19",
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "pipeline/audit_semantic_capsule_v1.py",
            "--data", str(train), "--episodes", str(heldout), "--out", str(report),
        ], cwd=ROOT, check=True)
        payload = json.loads(report.read_text())
        assert payload["malformed_train_rows"] == 0
        assert payload["invalid_completion_prompts"] == 0
        assert payload["duplicate_normalized_train_questions"] == 0
        assert payload["invalid_heldout_episodes"] == 0
        assert payload["overlap"] == {"exact_prompt_hits": 0, "ngram13_hits": 0}
    print("semantic capsule audit tests passed")


if __name__ == "__main__":
    main()
