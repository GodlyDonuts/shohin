#!/usr/bin/env python3
"""Focused solver and split-contract tests for semantic bridge generation."""

from generate_semantic_bridge_v1 import FAMILIES, build_split, normalized_question, summary


def main():
    train = build_split(24, 17, heldout=False)
    heldout = build_split(8, 18, heldout=True, excluded={normalized_question(row["question"]) for row in train})
    assert len(train) == len(FAMILIES) * 24
    assert len(heldout) == len(FAMILIES) * 8
    assert not ({normalized_question(row["question"]) for row in train} & {normalized_question(row["question"]) for row in heldout})
    assert summary(train)["all_have_think"] and summary(train)["all_have_final"]
    for row in train + heldout:
        assert row["training_group"] == "semantic_bridge"
        assert row["source"] == "semantic_bridge_v1_train"
        assert row["family"] in FAMILIES
        assert row["answer"].lstrip("-").isdigit()
        assert "<think>" in row["response"] and "The answer is " + row["answer"] + "." in row["response"]
    print("semantic bridge generator tests passed")


if __name__ == "__main__":
    main()
