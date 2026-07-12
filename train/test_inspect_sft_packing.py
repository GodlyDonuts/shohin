#!/usr/bin/env python3
"""Regression test for hash-bound SFT packing reports."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tokenizer = root / "artifacts" / "shohin-tok-32k.json"
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        data = temporary / "data.jsonl"
        data.write_text(
            json.dumps({
                "question": "What is 2 plus 2?", "response": "4",
                "training_group": "verifier_correct",
            }) + "\n" + json.dumps({
                "question": "What is 3 plus 3?", "response": "5",
                "training_group": "verifier_incorrect",
            }) + "\n"
        )
        report = temporary / "packing.json"
        subprocess.run([
            sys.executable, str(root / "train" / "inspect_sft_packing.py"),
            "--data", str(data), "--tokenizer", str(tokenizer), "--out", str(report),
            "--pack-len", "8",
        ], check=True, stdout=subprocess.DEVNULL)
        payload = json.loads(report.read_text())
        key = str(data.resolve())
        assert payload["data_sha256"][key] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert set(payload["group_counts"]) <= {"verifier_correct", "verifier_incorrect"}
    print("SFT packing hash checks: passed")


if __name__ == "__main__":
    main()
