#!/usr/bin/env python3
"""Exercise positive and rejection paths for the operator-trace assessment."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def factor(joint):
    rows = []
    for regime in ("wording", "value", "full"):
        for family in ("a", "b", "c"):
            for index in range(100):
                rows.append({"regime": regime, "family": family, "correct_trace_and_final": index < joint[regime]})
    return {"rows": rows}


def primitive(arithmetic, base):
    return {"summary": {"by_family": {"arithmetic": {"accuracy": arithmetic}, "base_conversion": {"accuracy": base}}}}


def primitive_by_contract(arithmetic, base):
    return {"by_contract": {"answer": {"families": {
        "arithmetic": {"accuracy": arithmetic},
        "base_conversion": {"accuracy": base},
    }}}}


def trace(joint):
    return {"summary": {"correct_trace_and_final": joint}}


with tempfile.TemporaryDirectory() as temporary:
    temporary = Path(temporary)
    values = {
        "baseline_factor": factor({"wording": 1, "value": 1, "full": 1}),
        "candidate_factor": factor({"wording": 16, "value": 16, "full": 11}),
        "baseline_primitives": primitive(0.07, 0.02),
        "candidate_primitives": primitive(0.10, 0.10),
        "baseline_rg": {"accuracy": 0.18},
        "candidate_rg": {"accuracy": 0.17},
        "manual": {"models": [
            {"checkpoint": "baseline", "summary": {"initial": 2, "verified_fact": 1}},
            {"checkpoint": "candidate", "summary": {"initial": 2, "verified_fact": 1}},
        ]},
        "baseline_fixed_trace": trace(3),
        "candidate_fixed_trace": trace(3),
    }
    arguments = []
    for name, value in values.items():
        path = temporary / (name + ".json")
        path.write_text(json.dumps(value))
        arguments.extend(("--" + name.replace("_", "-"), str(path)))
    accepted = temporary / "accepted.json"
    subprocess.run([sys.executable, str(ROOT / "assess_operator_trace_v2.py"), *arguments, "--out", str(accepted)], check=True, capture_output=True)
    assert json.loads(accepted.read_text())["decision"] == "bounded_operator_binding_transfer"
    values["candidate_primitives"] = primitive_by_contract(0.10, 0.10)
    path = temporary / "candidate_primitives.json"
    path.write_text(json.dumps(values["candidate_primitives"]))
    contract_accepted = temporary / "contract_accepted.json"
    subprocess.run([sys.executable, str(ROOT / "assess_operator_trace_v2.py"), *arguments, "--out", str(contract_accepted)], check=True, capture_output=True)
    assert json.loads(contract_accepted.read_text())["decision"] == "bounded_operator_binding_transfer"
    values["candidate_primitives"] = primitive(0.10, 0.10)
    values["manual"] = {"models": [
        {"checkpoint": "baseline", "summary": {"initial": 2, "verified_fact": 1}},
        {"checkpoint": "candidate", "summary": {"initial": 2, "verified_fact": 1}, "rows": [
            {"initial": {"response": "Problem A: 1 + 1 = 2"}},
        ]},
    ]}
    path = temporary / "manual.json"
    path.write_text(json.dumps(values["manual"]))
    pair_rejected = temporary / "pair_rejected.json"
    subprocess.run([sys.executable, str(ROOT / "assess_operator_trace_v2.py"), *arguments, "--out", str(pair_rejected)], check=True, capture_output=True)
    assert not json.loads(pair_rejected.read_text())["gates"]["paired_response_mode_absent"]
    assert json.loads(pair_rejected.read_text())["decision"] == "reject_operator_trace_candidate"
    values["candidate_primitives"] = primitive(0.09, 0.10)
    path = temporary / "candidate_primitives.json"
    path.write_text(json.dumps(values["candidate_primitives"]))
    rejected = temporary / "rejected.json"
    subprocess.run([sys.executable, str(ROOT / "assess_operator_trace_v2.py"), *arguments, "--out", str(rejected)], check=True, capture_output=True)
    assert json.loads(rejected.read_text())["decision"] == "reject_operator_trace_candidate"

print("operator-trace assessment checks: passed")
