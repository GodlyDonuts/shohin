#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import assess_referential_literal_pointer_factorized as assessment  # noqa: E402


def result(program=1.0):
    return {
        "overall": {
            "answer_accuracy": 1.0,
            "semantic_program_exact": program,
            "full_pointer_exact": 1.0,
            "kind_accuracy": 1.0,
            "initial_joint": 1.0,
        },
        "group_summary": {
            "canonical_paraphrase_both_exact": 512,
            "all_four_full_pointer_exact": 512,
        },
    }


class FactorizedAssessmentTest(unittest.TestCase):
    def test_tied_perfect_arms_fail_only_attribution(self):
        gates = assessment.primary_gate_results(result(), result())
        self.assertTrue(all(
            gate["pass"] for name, gate in gates.items()
            if name != "islands_program_advantage_over_ordinary"
        ))
        self.assertFalse(gates["islands_program_advantage_over_ordinary"]["pass"])

    def test_five_point_program_advantage_passes_attribution(self):
        gates = assessment.primary_gate_results(result(0.90), result(0.84))
        self.assertTrue(gates["islands_program_advantage_over_ordinary"]["pass"])


if __name__ == "__main__":
    unittest.main()
