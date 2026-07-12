#!/usr/bin/env python3
"""Small deterministic test for verifier-data balance and label separation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "rollouts.jsonl"
        rows = [
            {"question": "q1", "candidate": "72", "correct": True},
            {"question": "q1", "candidate": "71", "correct": False},
            {"question": "q2", "candidate": "5", "correct": True},
            {"question": "q2", "candidate": "6", "correct": False},
            {"question": "q2", "candidate": "7", "correct": False},
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        out = root / "verifier.jsonl"
        subprocess.run([
            sys.executable, "pipeline/build_verifier_dataset.py",
            "--input", str(source), "--out", str(out), "--negative-ratio", "1",
        ], check=True)
        built = [json.loads(line) for line in out.read_text().splitlines()]
        labels = {row["response"] for row in built}
        assert labels == {"<|correct|>", "<|incorrect|>"}
        assert len(built) == 4
        assert all("q" not in row["response"] for row in built)
    print("verifier dataset checks: passed")


if __name__ == "__main__":
    main()
