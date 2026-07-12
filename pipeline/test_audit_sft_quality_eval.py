#!/usr/bin/env python3
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        evals = temporary / "evals"
        evals.mkdir()
        (evals / "gsm8k.jsonl").write_text(json.dumps({"question": "What is 2 plus 2?"}) + "\n")
        data = temporary / "data.jsonl"
        data.write_text(
            json.dumps({"question": "What is 2 plus 2?", "response": "The answer is 4."}) + "\n" +
            json.dumps({"question": "Compute 3 plus 3.", "response": "The answer is 6."}) + "\n"
        )
        report = temporary / "report.json"
        subprocess.run([sys.executable, str(root / "audit_sft_quality.py"), "--data", str(data),
                        "--evals", str(evals), "--out", str(report)], check=True, capture_output=True, text=True)
        result = json.loads(report.read_text())
        assert result["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert result["eval_overlap"]["exact_prompt_hits"] == 1
        assert result["eval_overlap"]["ngram_prompt_hits"] == 1
    print("sft eval-overlap audit checks: passed")


if __name__ == "__main__":
    main()
