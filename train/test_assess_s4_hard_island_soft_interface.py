#!/usr/bin/env python3

import unittest

from assess_s4_hard_island_soft_interface import assess


def summary(program=0.97, control=False):
    values = {
        field: {"accuracy": 0.99}
        for field in (
            "valid", "count_exact", "query_exact", "state_exact", "answer_correct",
            "initial_exact",
        )
    }
    values["program_exact"] = {"accuracy": 0.10 if control else program}
    return values


class AssessHardIslandSoftInterfaceTest(unittest.TestCase):
    def test_frozen_gates_pass_and_control_is_required(self):
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
        self.assertTrue(assess(treatment, baseline)["all_gates_pass"])
        treatment["event_region_deranged"]["overall"] = summary(program=0.90)
        self.assertFalse(assess(treatment, baseline)["all_gates_pass"])


if __name__ == "__main__":
    unittest.main()
