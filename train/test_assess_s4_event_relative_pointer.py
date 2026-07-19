#!/usr/bin/env python3

import unittest

from assess_s4_event_relative_pointer import assess


FIELDS = (
    "valid", "count_exact", "program_exact", "query_exact", "state_exact",
    "answer_correct", "initial_exact",
)


def summary(value):
    return {field: {"accuracy": value, "correct": int(value * 1000)} for field in FIELDS}


def report(value):
    return {
        "overall": summary(value),
        "by_depth": {str(depth): summary(value) for depth in range(3, 9)},
        "gold_event_s3_sanity": True,
        "parameter_count": 135_000_000,
        "development_access": 1,
        "confirmation_access": 0,
    }


class S4EventRelativeAssessmentTest(unittest.TestCase):
    def test_pass_and_program_floor(self):
        self.assertTrue(assess(report(0.99), report(0.0))["all_gates_pass"])
        failed = assess(report(0.89), report(0.0))
        self.assertFalse(failed["all_gates_pass"])
        self.assertFalse(failed["gates"]["program_overall_at_least_95pct"])


if __name__ == "__main__":
    unittest.main()
