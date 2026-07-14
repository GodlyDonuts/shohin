#!/usr/bin/env python3
"""CPU-only checks for the FQRB three-digit primary-state factor."""
from __future__ import annotations

from generate_finite_query_residual_basis_magnitude_factor_v1 import build
from generate_finite_query_residual_basis_v1 import QUERY_KINDS


rows = build(30, 17)
assert len(rows) == 30 * len(QUERY_KINDS)
assert {row["query_kind"] for row in rows} == set(QUERY_KINDS)
assert all(abs(row["state"][field]["primary"]) >= 100 for row in rows for field in ("base", "donor"))
assert all(abs(row["state"]["base"]["primary"] + row["delta"]) >= 100 for row in rows)
assert all(row["response"] != row["counterfactual_response"] for row in rows)
print("FQRB magnitude-factor generator checks: passed")
