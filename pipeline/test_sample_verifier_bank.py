#!/usr/bin/env python3
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        source = temporary / "source.jsonl"
        source.write_text(
            json.dumps({"question": "Repeated prompt!", "answer": "1", "family": "a"}) + "\n" +
            json.dumps({"question": "repeated prompt", "answer": "1", "family": "b"}) + "\n" +
            json.dumps({"question": "Fresh A", "answer": "2", "family": "a"}) + "\n" +
            json.dumps({"question": "Fresh B", "answer": "3", "family": "b"}) + "\n"
        )
        out = temporary / "bank.jsonl"
        subprocess.run([
            sys.executable, str(root / "sample_verifier_bank.py"), "--input", str(source),
            "--out", str(out), "--per-family", "2", "--seed", "7",
        ], check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        questions = [row["question"].lower().replace("!", "") for row in rows]
        assert len(questions) == len(set(questions))
        assert len(rows) == 3
        expected = ["Repeated prompt!", "Fresh A", "Fresh B"]
        random.Random(7).shuffle(expected)
        assert [row["question"] for row in rows] == expected
    print("balanced verifier-bank dedup checks: passed")


if __name__ == "__main__":
    main()
