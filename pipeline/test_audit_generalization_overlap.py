#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(data_rows):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        cases = temporary / "cases.py"
        cases.write_text("CASES = [{'id': 'apple_sum', 'question': 'Add three bright apples and two green pears.'}]\n")
        data = temporary / "data.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in data_rows))
        report = temporary / "report.json"
        subprocess.run([
            sys.executable, str(root / "audit_generalization_overlap.py"),
            "--data", str(data), "--case-source", str(cases), "--out", str(report), "--ngram", "3",
        ], check=True, capture_output=True, text=True)
        return json.loads(report.read_text())


def run_json(data_rows, regimes=()):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        cases = temporary / "cases.json"
        cases.write_text(json.dumps({"cases": [
            {"id": "apple_sum", "question": "Add three bright apples and two green pears.", "regime": "wording"},
            {"id": "orange_sum", "question": "Add four bright oranges and two green pears.", "regime": "value"},
        ]}))
        data = temporary / "data.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in data_rows))
        report = temporary / "report.json"
        command = [
            sys.executable, str(root / "audit_generalization_overlap.py"),
            "--data", str(data), "--cases-json", str(cases), "--out", str(report), "--ngram", "3",
        ]
        if regimes:
            command.extend(("--case-regimes", *regimes))
        subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(report.read_text())


def main():
    clean = run([{"question": "Compute a different triangle perimeter.", "response": "9"}])
    assert clean["exact_prompt_hits"] == 0
    assert clean["ngram_prompt_hits"] == 0
    leaked = run([{"question": "Add three bright apples and two green pears.", "response": "5"}])
    assert leaked["exact_prompt_hits"] == 1
    assert leaked["ngram_prompt_hits"] == 1
    assert leaked["examples"][0]["case_ids"] == ["apple_sum"]
    json_clean = run_json([{"question": "Compute a different triangle perimeter.", "response": "9"}])
    assert json_clean["exact_prompt_hits"] == 0
    assert json_clean["ngram_prompt_hits"] == 0
    regime_clean = run_json([{"question": "Add four bright oranges and two green pears.", "response": "6"}], ("wording",))
    assert regime_clean["cases"] == 1
    assert regime_clean["case_regimes"] == ["wording"]
    assert regime_clean["exact_prompt_hits"] == 0
    print("generalization interview overlap audit: passed")


if __name__ == "__main__":
    main()
