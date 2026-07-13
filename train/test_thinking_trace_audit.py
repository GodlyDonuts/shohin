#!/usr/bin/env python3
"""Focused parser tests for the visible-thinking audit."""

from thinking_trace_audit import CASES, score_response, summarize


def main():
    case = CASES[0]
    correct = score_response(case, "<think>product=378</think>\nThe answer is 369.")
    assert correct["trace_present"]
    assert correct["trace_correct"]
    assert correct["answer_correct"]
    assert correct["correct_trace_and_final"]

    wrong_trace = score_response(case, "<think>product=377</think>\nThe answer is 369.")
    assert not wrong_trace["trace_correct"]
    assert wrong_trace["answer_correct"]
    assert not wrong_trace["correct_trace_and_final"]

    missing_tags = score_response(case, "product=378\nThe answer is 369.")
    assert not missing_tags["trace_present"]
    assert not missing_tags["correct_trace_and_final"]

    row = dict(CASES[0], **correct)
    summary = summarize([row])
    assert summary["cases"] == 1
    assert summary["correct_trace_and_final"] == 1
    print("thinking trace audit tests passed")


if __name__ == "__main__":
    main()
