#!/usr/bin/env python3
"""Focused policy checks for FQRB gate aggregation."""
from __future__ import annotations

from assess_finite_query_residual_basis_v1 import CONTROL_FIELDS, DIRECT_FIELDS, QUERY_KINDS, evaluate_report, manual_summary


def report(joint: int = 300, control: int = 25, direct: int = 350) -> dict:
    return {
        "audit": "finite_query_residual_basis_v1",
        "basis_summary": {"groups": 500, "joint_strict": joint, **{field: control for field in CONTROL_FIELDS}},
        "consumer_summary": {kind: {field: direct for field in DIRECT_FIELDS} for kind in QUERY_KINDS},
    }


assert evaluate_report(report())["bounded_basis_gate"] is True
assert evaluate_report(report(joint=299))["bounded_basis_gate"] is False
assert evaluate_report(report(control=26))["bounded_basis_gate"] is False
assert evaluate_report(report(direct=349))["bounded_basis_gate"] is False


def manual(initial: int, fact: int) -> dict:
    checkpoint = "/tmp/fqrb.pt"
    return {
        "audit": "manual_capability_probe_v1",
        "models": [
            {"checkpoint": "/tmp/raw.pt", "summary": {"initial": 1, "review": 0, "verified_fact": 1, "state_reuse": 0}},
            {"checkpoint": checkpoint, "summary": {"initial": initial, "review": 0, "verified_fact": fact, "state_reuse": 0}},
        ],
    }


assert manual_summary(manual(1, 1), "/tmp/fqrb.pt")["direct_decode_preserved"] is True
assert manual_summary(manual(0, 1), "/tmp/fqrb.pt")["direct_decode_preserved"] is False
assert manual_summary(manual(1, 0), "/tmp/fqrb.pt")["direct_decode_preserved"] is False
print("FQRB assessment checks: passed")
