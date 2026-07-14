#!/usr/bin/env python3
"""Focused mechanical checks for the FQRB data contract."""
from __future__ import annotations

from generate_finite_query_residual_basis_v1 import QUERY_KINDS, TWO_DIGIT_VALUES, audit, build, consumer_support


train = build(30, 41, "train", TWO_DIGIT_VALUES, 90)
heldout = build(12, 42, "heldout", TWO_DIGIT_VALUES, 90, language_heldout=True)
assert len(train) == 30 * len(QUERY_KINDS)
assert len(heldout) == 12 * len(QUERY_KINDS)
for rows in (train, heldout):
    by_basis = {}
    for row in rows:
        by_basis.setdefault(row["basis_id"], []).append(row)
        assert row["response"] != row["counterfactual_response"]
    assert all({row["query_kind"] for row in group} == set(QUERY_KINDS) for group in by_basis.values())
report = audit(train, heldout)
assert not report["duplicate_train_prompts"]
assert not report["duplicate_heldout_prompts"]
assert not report["train_heldout_exact_prompt_hits"]
assert not report["train_heldout_13gram_hits"]
assert not report["train_heldout_exact_source_bundle_hits"]
assert len(set().union(*consumer_support().values())) == 13
print("FQRB generator checks: passed")
