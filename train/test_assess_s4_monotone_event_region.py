#!/usr/bin/env python3

import unittest

from assess_s4_monotone_event_region import assess


def summary(program=0.96, control=False):
    values = {
        field: {"accuracy": 0.99}
        for field in (
            "valid", "count_exact", "query_exact", "state_exact", "answer_correct",
            "initial_exact",
        )
    }
    values["program_exact"] = {"accuracy": 0.10 if control else program}
    return values


class AssessMonotoneEventRegionTest(unittest.TestCase):
    def test_all_frozen_gates_pass(self):
        treatment = {
            "overall": summary(),
            "by_depth": {str(depth): summary() for depth in range(3, 9)},
            "roster_deranged": {"overall": summary(control=True)},
            "event_region_deranged": {"overall": summary(control=True)},
            "gold_event_s3_sanity": True,
            "trainable_parameters": 0,
            "parameter_count": 134_000_000,
            "development_access": 1,
            "confirmation_access": 0,
        }
        baseline = {"strict_autonomous": {"overall": summary(program=0.93)}}
        result = assess(treatment, baseline)
        self.assertTrue(result["all_gates_pass"], result)

    def test_event_control_is_required(self):
        treatment = {
            "overall": summary(),
            "by_depth": {str(depth): summary() for depth in range(3, 9)},
            "roster_deranged": {"overall": summary(control=True)},
            "event_region_deranged": {"overall": summary(program=0.90)},
            "gold_event_s3_sanity": True,
            "trainable_parameters": 0,
            "parameter_count": 134_000_000,
            "development_access": 1,
            "confirmation_access": 0,
        }
        baseline = {"strict_autonomous": {"overall": summary(program=0.93)}}
        result = assess(treatment, baseline)
        self.assertFalse(result["all_gates_pass"])


if __name__ == "__main__":
    unittest.main()
