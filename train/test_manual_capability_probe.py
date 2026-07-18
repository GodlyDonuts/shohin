#!/usr/bin/env python3
"""Regression tests for transcript-first manual capability scoring."""

from manual_capability_probe import CASES, score


def case(case_id):
    return next(item for item in CASES if item["id"] == case_id)


def main():
    logic = case("logic")
    assert score(logic, "No, it is not possible.")
    assert score(logic, "Answer: no")
    assert not score(logic, "State: A zol is a mar. No mar is a tiv.")

    arithmetic = case("arithmetic")
    assert score(arithmetic, "427")
    assert score(arithmetic, "427\nExplanation follows")
    assert not score(arithmetic, "29 * 16 = 464; 464 - 37 = 427")

    sorted_case = case("sort_deduplicate")
    assert score(sorted_case, "[1, 3, 8, 13]")
    assert not score(sorted_case, "Input: [13, 3, 13, 8, 1, 8]")

    string_case = case("string_insert")
    assert score(string_case, "lanPQtern")
    assert score(string_case, "'lanPQtern'")
    assert not score(string_case, "The result is lanPQtern")

    print("manual capability scoring: passed (strict answer-position contract)")


if __name__ == "__main__":
    main()
