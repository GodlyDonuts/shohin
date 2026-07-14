#!/usr/bin/env python3
"""CPU-only contracts for ephemeral-codebook FQRB generation."""
from __future__ import annotations

import random

from generate_ephemeral_codebook_fqrb_v1 import (
    CANONICAL_LABELS,
    audit,
    codebook_key,
    swapped_mapping,
    validate_row,
    wrap_groups,
)
from generate_finite_query_residual_basis_v1 import TWO_DIGIT_VALUES, build


train_base = build(12, 2026071418, "train", TWO_DIGIT_VALUES, 90)
held_base = build(4, 2026071419, "heldout", TWO_DIGIT_VALUES, 90, language_heldout=True)
train, train_codebooks = wrap_groups(train_base, random.Random(31), set())
heldout, _ = wrap_groups(held_base, random.Random(32), train_codebooks)
assert len(train) == 60 and len(heldout) == 20
assert not ({codebook_key(row["codebook"]) for row in train} & {codebook_key(row["codebook"]) for row in heldout})
for row in train + heldout:
    validate_row(row)
    assert row["response"] != row["counterfactual_response"]
    assert row["response"] != row["codebook_swap_response"]
mapping = train[0]["codebook"]
swapped, decoy = swapped_mapping(mapping, train[0]["semantic_response"], 0)
assert decoy != train[0]["semantic_response"]
assert swapped[train[0]["semantic_response"]] != mapping[train[0]["semantic_response"]]
report = audit(train, heldout)
for field in (
    "duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits",
    "train_heldout_exact_source_bundle_hits", "train_heldout_codebook_hits",
    "train_heldout_semantic_13gram_hits", "bad_train_group_cardinality", "bad_heldout_group_cardinality",
):
    assert report[field] == 0, (field, report[field])
assert len(CANONICAL_LABELS) == 13
print("ephemeral-codebook FQRB generator checks: passed")
