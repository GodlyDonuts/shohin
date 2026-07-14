#!/usr/bin/env python3
"""CPU-only checks for FQRB familiar-wording source-tuple factor generation."""
from __future__ import annotations

from generate_finite_query_residual_basis_core_factor_v1 import choose_unseen_groups
from generate_finite_query_residual_basis_v1 import QUERY_KINDS, TWO_DIGIT_VALUES, build, source_bundle_key


train = build(100, 7, "train", TWO_DIGIT_VALUES, 90)
factor = choose_unseen_groups(train, 25, 8)
assert len(factor) == 25 * len(QUERY_KINDS)
assert not ({source_bundle_key(row) for row in train} & {source_bundle_key(row) for row in factor})
assert {row["query_kind"] for row in factor} == set(QUERY_KINDS)
print("FQRB core-factor generator checks: passed")
