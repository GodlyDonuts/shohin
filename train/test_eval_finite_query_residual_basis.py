#!/usr/bin/env python3
"""CPU-only checks for FQRB grouping and causal score semantics."""
from __future__ import annotations

from eval_finite_query_residual_basis import QUERY_KINDS, group_rows, score_result, shifted_group_keys, summarize_groups


def row(basis_id: str, kind: str) -> dict:
    return {
        "schema": "counterfactual_residual_algebra_v1", "split": "heldout", "basis_mode": "multi_consumer",
        "basis_id": basis_id, "episode_id": basis_id + ":" + kind, "query_kind": kind,
        "base_source": "base", "edited_source": "edited", "donor_source": "donor",
        "paraphrase_base_source": "pbase", "paraphrase_edited_source": "pedited", "paraphrase_donor_source": "pdonor",
        "response": "answer=" + kind, "counterfactual_response": "answer=other_" + kind,
    }


rows = [row(basis_id, kind) for basis_id in ("a", "b") for kind in QUERY_KINDS]
groups = group_rows(rows, "heldout", 0)
assert [basis_id for basis_id, _ in groups] == ["a", "b"]
assert shifted_group_keys([basis_id for basis_id, _ in groups]) == {"a": "b", "b": "a"}
results = []
for basis_id, group in groups:
    for item in group:
        result = {
            "basis_id": basis_id, "query_kind": item["query_kind"], "expected": item["response"],
            "counterfactual_expected": item["counterfactual_response"], "normal": item["response"],
            "paraphrase": item["response"], "counterfactual": item["counterfactual_response"],
            "zero": "wrong", "shuffled": "wrong", "wrong_query": "wrong",
        }
        results.append(score_result(result))
summary = summarize_groups(results, [basis_id for basis_id, _ in groups])
assert summary == {
    "groups": 2, "joint_normal": 2, "joint_paraphrase": 2, "joint_counterfactual": 2, "joint_strict": 2,
    "any_zero_recreates_normal": 0, "any_shuffle_recreates_normal": 0, "any_wrong_query_recreates_normal": 0,
}
results[0]["wrong_query"] = results[0]["expected"]
score_result(results[0])
assert summarize_groups(results, [basis_id for basis_id, _ in groups])["joint_strict"] == 1
print("FQRB evaluator checks: passed")
