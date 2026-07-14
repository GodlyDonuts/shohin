#!/usr/bin/env python3
"""Contract test for the post-freeze state-trace case generator."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


with tempfile.TemporaryDirectory() as root:
    out = Path(root) / "cases.json"
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("build_trace_state_ood_v1.py")),
        "--out", str(out), "--seed", "19", "--per-template", "2",
    ], check=True, stdout=subprocess.DEVNULL)
    payload = json.loads(out.read_text())
    cases = payload["cases"]
    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
    assert len({case["question"] for case in cases}) == 12
    assert {case["id"].split("_")[0] for case in cases} == {"mid", "high"}
    for case in cases:
        assert isinstance(case["answer"], int)
        assert len(case["markers"]) == 2
        assert "<think>" in case["question"]

print("trace state OOD builder checks: passed")
