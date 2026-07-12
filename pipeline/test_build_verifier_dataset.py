#!/usr/bin/env python3
"""Small deterministic test for verifier-data balance and label separation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
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
            sys.executable, str(ROOT / "build_verifier_dataset.py"), "--input", str(source),
            "--out", str(out), "--negative-ratio", "1",
        ], check=True)
        built = [json.loads(line) for line in out.read_text().splitlines()]
        labels = {row["response"] for row in built}
        assert labels == {"<|correct|>", "<|incorrect|>"}
        assert len(built) == 4
        assert all("q" not in row["response"] for row in built)

        skewed = root / "skewed.jsonl"
        skewed.write_text("".join(
            json.dumps({"question": "q", "candidate": f"yes-{index}", "correct": True}) + "\n"
            for index in range(3)
        ) + json.dumps({"question": "q", "candidate": "no", "correct": False}) + "\n")
        balanced = root / "balanced.jsonl"
        subprocess.run([
            sys.executable, str(ROOT / "build_verifier_dataset.py"),
            "--input", str(skewed), "--out", str(balanced), "--balance-classes",
        ], check=True)
        balanced_rows = [json.loads(line) for line in balanced.read_text().splitlines()]
        assert len(balanced_rows) == 2
        assert sum(row["response"] == "<|correct|>" for row in balanced_rows) == 1

        duplicate = root / "duplicate.jsonl"
        duplicate.write_text("".join(json.dumps(row) + "\n" for row in [
            {"question": "q3", "candidate": "same", "correct": True},
            {"question": "q3", "candidate": "same", "correct": True},
            {"question": "q3", "candidate": "same!", "correct": True},
            {"question": "q3", "candidate": "different", "correct": False},
        ]))
        deduped = root / "deduped.jsonl"
        subprocess.run([
            sys.executable, str(ROOT / "build_verifier_dataset.py"),
            "--input", str(duplicate), "--out", str(deduped), "--balance-classes",
        ], check=True)
        deduped_rows = [json.loads(line) for line in deduped.read_text().splitlines()]
        assert len(deduped_rows) == 2
        assert len({row["question"] for row in deduped_rows}) == 2
    print("verifier dataset checks: passed")


if __name__ == "__main__":
    main()
