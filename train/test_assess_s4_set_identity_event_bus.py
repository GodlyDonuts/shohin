#!/usr/bin/env python3

import unittest

from assess_s4_set_identity_event_bus import assess


FIELDS = (
    "valid", "count_exact", "program_exact", "query_exact", "state_exact",
    "answer_correct", "initial_exact",
)


def summary(value):
    return {field: {"accuracy": value, "correct": int(value * 1000)} for field in FIELDS}


def report(value, deranged=0.0):
    return {
        "overall": summary(value),
        "by_depth": {str(depth): summary(value) for depth in range(3, 9)},
        "roster_deranged": {"overall": summary(deranged)},
        "gold_event_s3_sanity": True,
        "parameter_count": 135_000_000,
        "development_access": 1,
        "confirmation_access": 0,
    }


def baseline(value):
    return {
        "schema": "r12_s4_self_delimiting_event_tape_eval_v1",
        "strict_autonomous": {"overall": summary(value)},
    }


class S4SetIdentityAssessmentTest(unittest.TestCase):
    def test_pass_and_baseline_margin(self):
        self.assertTrue(assess(report(0.99), report(0.0), baseline(0.93))["all_gates_pass"])
        failed = assess(report(0.95), report(0.0), baseline(0.949))
        self.assertFalse(failed["all_gates_pass"])
        self.assertFalse(failed["gates"]["program_at_least_v1_plus_1pp"])


if __name__ == "__main__":
    unittest.main()
