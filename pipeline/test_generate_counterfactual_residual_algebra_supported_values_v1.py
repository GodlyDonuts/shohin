#!/usr/bin/env python3
"""Focused invariants for the CRA answer-support-matched value probe."""
from __future__ import annotations

import random

from generate_counterfactual_residual_algebra_supported_values_v1 import OUT_OF_RANGE, build, make_row, source_numbers


rows = build(80, 17)
assert len(rows) == 80
assert len({row["episode_id"] for row in rows}) == 80
for row in rows:
    assert row["split"] == "factor_values_answer_supported"
    assert row["query_kind"] == "difference"
    assert set(source_numbers(row)) <= set(OUT_OF_RANGE)
    assert all(-4 <= int(row[key].removeprefix("answer=")) <= 4 for key in ("response", "counterfactual_response"))
    assert row["response"] != row["counterfactual_response"]
assert make_row(random.Random(3), "smoke", 0)["state"]["donor"]["primary"] in OUT_OF_RANGE
print("CRA supported-value factor checks: passed")
