#!/usr/bin/env python3
"""End-to-end contract checks for the retention decision script."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from assess_retention_sft_v1 import summary_for_manual


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def run_assessment(root: Path, candidate_manual: dict, candidate_visible_traces: int = 1) -> dict:
    raw_primitive = write_json(root / "raw_primitive.json", {"accuracy": 0.10})
    candidate_primitive = write_json(root / "candidate_primitive.json", {
        "accuracy": 0.25,
        "by_contract": {"answer": {"families": {
            "arithmetic": {"accuracy": 0.30},
            "base_conversion": {"accuracy": 0.20},
        }}},
    })
    raw_rg = write_json(root / "raw_rg.json", {"accuracy": 0.02})
    candidate_rg = write_json(root / "candidate_rg.json", {"accuracy": 0.05})
    manual = write_json(root / "manual.json", {"models": [
        {"checkpoint": "/x/best_step200000.pt", "summary": {"initial": 1, "verified_fact": 1}},
        {"checkpoint": "/x/sft_ep1.pt", "summary": candidate_manual},
    ]})
    deep = write_json(root / "deep.json", {"model": {"summary": {"initial": 1}}})
    raw_trace = write_json(root / "raw_trace.json", {"summary": {"correct_trace_and_final": 0}})
    candidate_trace = write_json(root / "candidate_trace.json", {
        "summary": {"correct_trace_and_final": candidate_visible_traces},
    })
    out = root / "assessment.json"
    script = Path(__file__).with_name("assess_retention_sft_v1.py")
    subprocess.run([
        sys.executable, str(script),
        "--raw-primitives", str(raw_primitive),
        "--candidate-primitives", str(candidate_primitive),
        "--raw-rg", str(raw_rg),
        "--candidate-rg", str(candidate_rg),
        "--manual", str(manual),
        "--deep", str(deep),
        "--raw-trace", str(raw_trace),
        "--candidate-trace", str(candidate_trace),
        "--out", str(out),
    ], check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


report = {"models": [{"checkpoint": "/x/best_step200000.pt", "summary": {"initial": 1}}]}
assert summary_for_manual(report, "best_step200000.pt") == {"initial": 1}
try:
    summary_for_manual(report, "sft_ep1.pt")
except ValueError:
    pass
else:
    raise AssertionError("missing checkpoint must fail")

with tempfile.TemporaryDirectory() as root:
    root_path = Path(root)
    admitted = run_assessment(root_path / "admitted", {"initial": 1, "verified_fact": 1})
    assert admitted["decision"] == "bounded_behavior_preserving_visible_reasoning_signal"
    rejected = run_assessment(root_path / "rejected", {"initial": 0, "verified_fact": 1})
    assert rejected["decision"] == "reject_retention_candidate"
    no_trace = run_assessment(root_path / "no_trace", {"initial": 1, "verified_fact": 1}, 0)
    assert no_trace["decision"] == "reject_retention_candidate"

print("Retention assessment checks: passed")
