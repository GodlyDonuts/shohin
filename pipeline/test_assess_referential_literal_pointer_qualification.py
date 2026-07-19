#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import assess_referential_literal_pointer_qualification as qualification  # noqa: E402


def result(value=1.0, quartets=2048):
    return {
        "overall": {
            "answer_accuracy": value,
            "semantic_program_exact": value,
            "full_pointer_exact": value,
            "kind_accuracy": value,
            "initial_joint": value,
        },
        "group_summary": {"all_four_full_pointer_exact": quartets},
    }


class ReferentialLiteralPointerQualificationAssessmentTest(unittest.TestCase):
    def test_perfect_result_passes(self):
        gates = qualification.assess(result())
        self.assertTrue(all(gate["pass"] for gate in gates.values()))

    def test_subfloor_program_fails(self):
        candidate = result()
        candidate["overall"]["semantic_program_exact"] = 0.989
        gates = qualification.assess(candidate)
        self.assertFalse(gates["semantic_program_exact"]["pass"])

    def test_quartet_floor_is_inclusive(self):
        gates = qualification.assess(result(quartets=2000))
        self.assertTrue(gates["all_four_exact"]["pass"])


if __name__ == "__main__":
    unittest.main()
