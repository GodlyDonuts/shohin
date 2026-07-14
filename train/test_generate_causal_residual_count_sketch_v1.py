#!/usr/bin/env python3
"""CPU-only contract checks for the gated CRCS curriculum builder."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from generate_causal_residual_count_sketch_v1 import (  # noqa: E402
    QUERY_KINDS,
    admitted_parent,
    audit,
    build_histories,
    validate_row,
)


with tempfile.TemporaryDirectory() as directory:
    good = Path(directory) / "good.json"
    rejected = Path(directory) / "rejected.json"
    good.write_text(json.dumps({"decision": "bounded_ecli_late_binding_candidate"}))
    rejected.write_text(json.dumps({"decision": "rejected"}))
    assert admitted_parent(good)["decision"] == "bounded_ecli_late_binding_candidate"
    try:
        admitted_parent(rejected)
    except SystemExit as error:
        assert "does not admit" in str(error)
    else:
        raise AssertionError("CRCS must reject an unadmitted ECLI parent")


train, train_codebooks = build_histories(12, (4,), 11, "train", False, set())
heldout, _ = build_histories(4, (8, 16), 12, "heldout", True, train_codebooks)
report = audit(train, heldout)
for key in (
    "bad_train_history_cardinality",
    "bad_heldout_history_cardinality",
    "train_heldout_exact_history_hits",
    "train_heldout_codebook_hits",
    "train_heldout_semantic_13gram_hits",
):
    assert report[key] == 0, (key, report)
assert report["train_event_counts"] == [4]
assert report["heldout_event_counts"] == [8, 16]

row = train[0]
validate_row(row)
assert len(row["events"]) == 4
assert row["event_ordinal"] < len(row["events"])
assert row["response"] != row["counterfactual_response"]
assert row["response"] != row["codebook_swap_response"]
assert set(row["events"][row["event_ordinal"]]["semantic_by_query"]) == set(QUERY_KINDS)
for event in row["events"]:
    for field in ("base_source", "edited_source", "donor_source"):
        assert event[field] not in row["suffix_prompt"]

print("CRCS curriculum checks: passed")
