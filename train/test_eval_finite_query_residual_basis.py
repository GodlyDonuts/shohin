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

codebook_results = []
for basis_id, group in groups:
    for item in group:
        result = {
            "basis_id": basis_id, "query_kind": item["query_kind"], "expected": item["response"],
            "counterfactual_expected": item["counterfactual_response"], "normal": item["response"],
            "paraphrase": item["response"], "counterfactual": item["counterfactual_response"],
            "zero": "wrong", "shuffled": "wrong", "wrong_query": "wrong",
            "codebook_swap": "code=swapped", "codebook_swap_expected": "code=swapped",
        }
        codebook_results.append(score_result(result))
codebook_summary = summarize_groups(codebook_results, [basis_id for basis_id, _ in groups])
assert codebook_summary["joint_codebook_swap"] == 2
assert codebook_summary["any_codebook_swap_recreates_normal"] == 0
assert all(result["strict_causal"] for result in codebook_results)
codebook_results[0]["codebook_swap"] = codebook_results[0]["expected"]
score_result(codebook_results[0])
assert not codebook_results[0]["strict_causal"]

two_edit_rows = [row("two", kind) for kind in QUERY_KINDS]
for item in two_edit_rows:
    item["basis_mode"] = "multi_consumer_two_edit"
    item["mode"] = "two_edit"
    item.pop("edited_source")
    item.pop("paraphrase_edited_source")
    item["primary_edited_source"] = "primary"
    item["secondary_edited_source"] = "secondary"
    item["paraphrase_primary_edited_source"] = "pprimary"
    item["paraphrase_secondary_edited_source"] = "psecondary"
assert len(group_rows(two_edit_rows, "heldout", 0)) == 1
print("FQRB evaluator checks: passed")
