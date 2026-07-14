#!/usr/bin/env python3
"""Tests for deterministic R6 qualitative trace selection."""

from inspect_future_effect_r6 import build_transcript


def arm(correct, answer):
    return {
        "answer": answer,
        "answer_correct": correct,
        "exact_program": correct,
        "opcodes": [0],
        "values": [3],
        "traces": [[{"selected_probe": 1, "predicted_effect": 3.0}]],
    }


def record(index, active, random, zero=False, shuffled=False):
    return {
        "index": index,
        "reference": "case-{}".format(index),
        "regime": "language",
        "answer": 3,
        "query_prediction": 0,
        "query_correct": True,
        "policies": {
            "active": arm(active, 3 if active else 2),
            "random": arm(random, 3 if random else 2),
            "zero": arm(zero, 3 if zero else 0),
            "shuffled": arm(shuffled, 3 if shuffled else 1),
            "oracle": arm(True, 3),
        },
    }


def main():
    report = {
        "protocol": "active_counterfactual_distinction_eval_r6",
        "adapter_sha256": "abc",
        "records": [
            record(0, True, False),
            record(1, False, True),
            record(2, False, False),
            record(3, True, True),
        ],
    }
    rows = [{"question": "Question {}".format(index)} for index in range(4)]
    transcript = build_transcript(report, rows, 4)
    assert [case["category"] for case in transcript["cases"]] == [
        "active_only_over_random_and_controls", "random_only", "both_wrong", "both_correct",
    ]
    assert transcript["cases"][0]["question"] == "Question 0"
    assert transcript["cases"][0]["active"]["traces"][0][0]["selected_probe"] == 1
    print("R6 qualitative trace selection: passed")


if __name__ == "__main__":
    main()
