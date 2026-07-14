#!/usr/bin/env python3
"""CPU-only checks for the FQRB unseen two-edit factor."""
from __future__ import annotations

from generate_finite_query_residual_basis_two_edit_factor_v1 import build
from generate_finite_query_residual_basis_v1 import QUERY_KINDS


rows = build(30, 19)
assert len(rows) == 30 * len(QUERY_KINDS)
assert {row["query_kind"] for row in rows} == set(QUERY_KINDS)
assert all(row["mode"] == "two_edit" and row["response"] != row["counterfactual_response"] for row in rows)
assert all(
    row["state"]["target"]["primary"] == row["state"]["donor"]["primary"] + row["primary_delta"] + row["secondary_delta"]
    for row in rows
)
print("FQRB two-edit factor generator checks: passed")
