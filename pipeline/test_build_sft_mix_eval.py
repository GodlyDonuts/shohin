#!/usr/bin/env python3
"""Regression: live eval prompt n-grams must gate SFT mixing without a pickle."""
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from build_sft_mix import training_group


def main():
    assert training_group("taco_verified_train") == "algorithmic_code"
    assert training_group("code_contests_train") == "code"
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        evals = temporary / "evals"
        evals.mkdir()
        eval_question = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        (evals / "heldout.jsonl").write_text(json.dumps({"question": eval_question}) + "\n")
        source = temporary / "source.jsonl"
        source.write_text(
            json.dumps({"question": eval_question + " fifteen", "response": "bad", "source": "hy3"}) + "\n" +
            json.dumps({"question": "safe fresh question", "response": "good", "source": "hy3"}) + "\n"
        )
        out = temporary / "mix.jsonl"
        report = temporary / "report.json"
        subprocess.run([
            sys.executable, str(root / "build_sft_mix.py"), "--inputs", str(source),
            "--out", str(out), "--report", str(report), "--eval-glob", str(evals / "*.jsonl"),
            "--decontam-grams", str(temporary / "missing.pkl"),
        ], check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        result = json.loads(report.read_text())
        assert [row["question"] for row in rows] == ["safe fresh question"]
        assert result["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        assert result["drops"]["eval_ngram"] == 1
        assert result["limits"]["direct_eval_gram_count"] == 2
    print("sft live-eval ngram mix gate: passed")


if __name__ == "__main__":
    main()
