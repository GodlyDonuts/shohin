#!/usr/bin/env python3
"""Focused regression checks for the direct operator-trace source gate."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from audit_operator_trace_directness import audit


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def direct_row():
    return {
        "training_group": "operator_trace_contrast",
        "contract": "direct",
        "question": "Question: Start at 7, add 3, then multiply by 2. Answer:",
        "completion_prompt": "Question: Start at 7, add 3, then multiply by 2. Answer:",
        "response": "<think>plan=ADD,MULTIPLY; 7 + 3 = 10; 10 * 2 = 20</think>\nThe answer is 20.",
    }


def pair_row():
    row = direct_row()
    row["contract"] = "minimal_pair"
    row["question"] = "Problem A: add 3.\nProblem B: subtract 3.\nAnswer:"
    row["completion_prompt"] = row["question"]
    row["response"] = "<think>Problem A: 7 + 3 = 10; Problem B: 7 - 3 = 4</think>\nThe answers are A=10; B=4."
    return row


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        direct = tmp / "direct.jsonl"
        paired = tmp / "paired.jsonl"
        out = tmp / "out.json"
        write(direct, [direct_row()])
        write(paired, [direct_row(), pair_row()])

        report = audit(direct)
        assert report["operator_rows"] == 1
        assert report["contract_rows"] == {"direct": 1}
        assert report["pair_marker_rows"] == {}
        subprocess.run(
            [sys.executable, str(root / "audit_operator_trace_directness.py"), "--data", str(direct),
             "--out", str(out), "--require-only-contract", "direct"],
            check=True,
            capture_output=True,
        )
        rejected = subprocess.run(
            [sys.executable, str(root / "audit_operator_trace_directness.py"), "--data", str(paired),
             "--out", str(tmp / "paired.json"), "--require-only-contract", "direct"],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "non-direct contract" in rejected.stderr
    print("operator trace directness audit tests: passed")


if __name__ == "__main__":
    main()
