#!/usr/bin/env python3
"""Focused audit contract for latent-operator data."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from generate_latent_operator_v1 import build_rows, write_jsonl


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, heldout = directory / "train.jsonl", directory / "heldout.jsonl"
        report = directory / "audit.json"
        write_jsonl(train, build_rows(48, (1, 2, 3, 4), 23, False))
        write_jsonl(heldout, build_rows(36, (5, 6, 8), 24, True))
        subprocess.run([
            sys.executable, str(root / "audit_latent_operator_v1.py"),
            "--train", str(train), "--heldout", str(heldout), "--out", str(report),
        ], check=True)
        result = json.loads(report.read_text())
        assert result["invalid_train_rows"] == 0
        assert result["invalid_heldout_rows"] == 0
        assert result["overlap"] == {"exact_prompt_hits": 0, "ngram13_hits": 0}
    print("latent operator audit tests passed")


if __name__ == "__main__":
    main()
