#!/usr/bin/env python3
"""Focused parser contracts for semantic-capsule rollouts."""

from eval_semantic_capsule import canonical_capsule, parse_capsule, parse_final


def main():
    keys = ("copper", "silver")
    expected = {"copper": 17, "silver": 23}
    assert canonical_capsule(expected, keys) == "capsule:copper=17;silver=23"
    assert parse_capsule("<think>x</think>\ncapsule:copper=17;silver=23", keys) == expected
    assert parse_capsule("capsule:silver=23;copper=17", keys) == expected
    assert parse_capsule("capsule:copper=17;copper=23", keys) is None
    assert parse_capsule("capsule:copper=17", keys) is None
    assert parse_final("The answer is 7. The answer is 9.") == 9
    print("semantic capsule evaluator tests passed")

if __name__ == "__main__":
    main()
