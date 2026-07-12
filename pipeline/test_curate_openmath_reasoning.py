#!/usr/bin/env python3
"""Non-network selection checks for OpenMathReasoning curation."""
import unittest

from curate_openmath_reasoning import answer_matches, clean_trace, parsed_rate, truthy


class OpenMathReasoningSelectionTests(unittest.TestCase):
    def test_normalizes_think_tags_and_verifies_latex_fraction(self):
        trace = clean_trace("<think>Compute carefully. The answer is \\frac{1}{2}.</think>")
        self.assertEqual(trace, "Compute carefully. The answer is \\frac{1}{2}.")
        self.assertTrue(answer_matches(trace, "1/2"))

    def test_parses_quality_and_kaggle_flags_conservatively(self):
        self.assertEqual(parsed_rate("0.875"), 0.875)
        self.assertIsNone(parsed_rate("unknown"))
        self.assertTrue(truthy(True))
        self.assertTrue(truthy("yes"))
        self.assertFalse(truthy("false"))


if __name__ == "__main__":
    unittest.main()
