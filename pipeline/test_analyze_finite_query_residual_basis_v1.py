#!/usr/bin/env python3
"""CPU-only contracts for FQRB failure classification."""
from __future__ import annotations

from analyze_finite_query_residual_basis_v1 import QUERY_KINDS, analyze


def report(value: int = 400, strict: int = 350, controls: int = 0) -> dict:
    return {
        "consumer_summary": {kind: {field: value for field in ("normal_correct", "paraphrase_correct", "counterfactual_correct")} for kind in QUERY_KINDS},
        "basis_summary": {
            "joint_strict": strict, "any_zero_recreates_normal": controls,
            "any_shuffle_recreates_normal": controls, "any_wrong_query_recreates_normal": controls,
        },
    }


passed = report()
assert analyze(passed, passed, passed, passed)["reasons"] == ["bounded_fqrb_gate_passed"]
failed_train = report(strict=299)
assert analyze(failed_train, passed, passed, passed)["reasons"] == ["no_in_distribution_multi_reader_primitive"]
failed_core = report(value=349)
assert analyze(passed, passed, failed_core, passed)["reasons"] == ["unseen_source_tuple_transport_failure"]
failed_combined = report(value=349)
assert analyze(passed, failed_combined, passed, passed)["reasons"] == ["combined_language_or_joint_surface_failure"]
failed_magnitude = report(value=349)
assert analyze(passed, passed, passed, failed_magnitude)["reasons"] == ["primary_magnitude_transport_failure"]
leaky = report(controls=26)
assert analyze(passed, leaky, passed, passed)["reasons"] == ["source_free_control_leakage"]
print("FQRB taxonomy checks: passed")
