#!/usr/bin/env python3
"""Small deterministic admission test for the NRR corpus generator."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, heldout, report = directory / "train.jsonl", directory / "heldout.jsonl", directory / "report.json"
        subprocess.run([
            sys.executable, str(root / "generate_native_residual_relay_v1.py"),
            "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(report),
            "--train-count", "64", "--heldout-count", "32", "--seed", "17",
        ], check=True, capture_output=True, text=True)
        audit = json.loads(report.read_text())
        assert audit["train_rows"] == 64 and audit["heldout_rows"] == 32
        assert not any(audit[key] for key in (
            "duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits", "train_heldout_13gram_hits",
        ))
        row = json.loads(train.read_text().splitlines()[0])
        assert row["source"] not in row["suffix_prompt"]
        assert row["response"].startswith("answer=") and row["counterfactual_response"].startswith("answer=")
    print("native residual relay generator checks: passed")


if __name__ == "__main__":
    main()
