#!/usr/bin/env python3
"""Contracts for regime-aware role-equivariant overlap classification."""

from audit_role_equivariant_full_text_v3 import audit_rows, grams, normalized


def main():
    allowed = "Question: In a workshop record copper has 3 parts and silver has 4 parts."
    forbidden = "Task: A harbor inventory record marked x lists 3 items under crates and 4 items under lanterns."
    exact = {}
    gram_index = {}
    for gram in grams(allowed, 5):
        gram_index.setdefault(gram, set()).add("latent_operator_eval_slices_v2_64.jsonl:fit_iid")
    for gram in grams(forbidden, 5):
        gram_index.setdefault(gram, set()).add("latent_operator_eval_slices_v2_64.jsonl:language_ood")
    rows = [
        {"question": allowed, "response": "The answer is 7.", "semantic_view": "anchor", "reference": "a"},
        {"question": allowed, "response": "The answer is 7.", "semantic_view": "paraphrase_a", "reference": "b"},
        {"question": forbidden, "response": "The answer is 7.", "semantic_view": "anchor", "reference": "c"},
        {"question": "Entirely unrelated sentence here.", "response": "The answer is 7.", "semantic_view": "paraphrase_b", "reference": "d"},
    ]
    report = audit_rows(rows, exact, gram_index, 5)
    assert normalized("A, B!") == "a b"
    assert report["allowed_same_surface_rows"] == 1
    assert report["forbidden_rows"] == 2
    assert report["allowed_by_view"] == {"anchor": 1}
    print("role-equivariant full-text audit tests passed")


if __name__ == "__main__":
    main()
