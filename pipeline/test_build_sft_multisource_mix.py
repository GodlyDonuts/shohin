#!/usr/bin/env python3
"""Regression tests for group-preserving multi-source SFT candidate freezing."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        broad, memory, evals = temporary / "broad.jsonl", temporary / "memory.jsonl", temporary / "evals.jsonl"
        write_rows(broad, [
            {"question": "What is two plus two?", "response": "4", "source": "math", "training_group": "math"},
            {"question": "Write a Python identity function", "response": "def identity(x):\n    return x", "source": "code", "training_group": "code"},
        ])
        write_rows(memory, [
            {"question": "What is two plus two?", "response": "wm:a=4;b=0", "source": "vrwm", "training_group": "vrwm"},
            {"question": "Update wm:a=3;b=1 by adding b to a", "response": "wm:a=4;b=1", "source": "vrwm", "training_group": "vrwm"},
        ])
        write_rows(evals, [
            {"question": "What is two plus two?", "answer": "4"},
        ])
        out, report = temporary / "mix.jsonl", temporary / "mix.report.json"
        subprocess.run([
            sys.executable, str(root / "build_sft_multisource_mix.py"),
            "--inputs", str(broad), str(memory), "--out", str(out), "--report", str(report),
            "--eval-glob", str(evals),
        ], check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        summary = json.loads(report.read_text())
        assert len(rows) == 2
        assert [row["training_group"] for row in rows] == ["code", "vrwm"]
        assert summary["duplicate_normalized_questions_dropped"] == 0
        assert summary["training_group_rows"] == {"code": 1, "vrwm": 1}
        assert summary["eval_filter"]["enabled"]
        assert summary["eval_filter"]["exact_prompt_drops"] == 2
    print("multi-source SFT mix builder: passed")


if __name__ == "__main__":
    main()
