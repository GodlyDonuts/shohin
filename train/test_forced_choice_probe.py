#!/usr/bin/env python3
"""Unit checks for forced-choice diagnostic ranking."""
from forced_choice_probe import rank_candidates


def test_mean_probability_wins_before_length_bias():
    ranked = rank_candidates([
        {"candidate": "long", "mean_logprob": -1.0, "total_logprob": -10.0},
        {"candidate": "short", "mean_logprob": -1.1, "total_logprob": -1.1},
    ])
    assert [row["candidate"] for row in ranked] == ["long", "short"]


def test_total_breaks_equal_mean_ties():
    ranked = rank_candidates([
        {"candidate": "b", "mean_logprob": -1.0, "total_logprob": -3.0},
        {"candidate": "a", "mean_logprob": -1.0, "total_logprob": -2.0},
    ])
    assert [row["candidate"] for row in ranked] == ["a", "b"]


if __name__ == "__main__":
    test_mean_probability_wins_before_length_bias()
    test_total_breaks_equal_mean_ties()
    print("forced-choice probe checks: passed")
