#!/usr/bin/env python3
"""Focused regression tests for the response-contract audit."""

import json
import tempfile
from pathlib import Path

from audit_response_contracts import audit


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_contract_rates_and_percentiles():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "mix.jsonl"
        write_rows(
            data,
            [
                {"training_group": "math", "response": "<think>2 + 1 = 3.</think>\nThe answer is 3."},
                {"training_group": "math", "response": "<think>4 + 1 = 5.</think>\nThe answer is 5."},
                {"training_group": "memory", "response": "wm:a=3;b=4"},
                {"training_group": "code", "response": "def f(x):\n    return x"},
            ],
        )
        report = audit(data)

    assert report["rows"] == 4
    assert report["malformed_rows"] == 0
    assert report["missing_response_rows"] == 0
    assert report["groups"]["math"]["think_marker_rate"] == 1.0
    assert report["groups"]["math"]["answer_marker_rate"] == 1.0
    assert report["groups"]["memory"]["state_marker_rate"] == 1.0
    assert report["groups"]["code"]["think_marker_rate"] == 0.0
    assert len(report["data_sha256"]) == 64


def test_malformed_and_missing_response_are_counted():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "broken.jsonl"
        data.write_text('{"training_group":"math","response":"ok"}\nnot-json\n{"training_group":"math"}\n')
        report = audit(data)

    assert report["rows"] == 1
    assert report["malformed_rows"] == 1
    assert report["missing_response_rows"] == 1


if __name__ == "__main__":
    test_contract_rates_and_percentiles()
    test_malformed_and_missing_response_are_counted()
    print("response-contract audit tests passed")
