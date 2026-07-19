#!/usr/bin/env python3

import unittest

from assess_s4_event_tape import assess


def summary(value):
    fields = (
        "valid", "count_exact", "program_exact", "query_exact", "state_exact",
        "answer_correct", "initial_exact", "all_kind_lexical_matched",
    )
    return {field: {"accuracy": value, "correct": int(value * 1000)} for field in fields}


def report(primary_value, strict_value=0.943, gold_value=0.946):
    primary = {
        "overall": summary(primary_value),
        "by_depth": {str(depth): summary(primary_value) for depth in range(3, 9)},
    }
    return {
        "pointer_anchored_v1_1": primary,
        "strict_autonomous": {"overall": summary(strict_value)},
        "gold_count_control": {"overall": summary(gold_value)},
        "gold_event_s3_sanity": True,
        "parameter_count": 133_689_935,
        "development_access": 1,
        "confirmation_access": 0,
    }


class S4AssessmentTest(unittest.TestCase):
    def test_all_frozen_gates_can_pass(self):
        result = assess(report(0.99), report(0.0))
        self.assertTrue(result["all_gates_pass"], result["gates"])
        self.assertEqual(result["decision"], "qualify_s4_v1_1_for_fresh_confirmation")

    def test_program_floor_is_binding(self):
        result = assess(report(0.949), report(0.0))
        self.assertFalse(result["all_gates_pass"])
        self.assertFalse(result["gates"]["treatment_program_overall_at_least_95pct"])


if __name__ == "__main__":
    unittest.main()
