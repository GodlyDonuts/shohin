#!/usr/bin/env python3
"""Non-network selection checks for OpenMathReasoning curation."""
import unittest

from curate_openmath_reasoning import (answer_matches, clean_trace, extract_final, parsed_rate,
                                       supervised_token_count, truthy)


class OpenMathReasoningSelectionTests(unittest.TestCase):
    def test_normalizes_think_tags_and_verifies_latex_fraction(self):
        trace = clean_trace("<think>Compute carefully. The answer is \\frac{1}{2}.</think>")
        self.assertEqual(trace, "Compute carefully. The answer is \\frac{1}{2}.")
        self.assertTrue(answer_matches(trace, "1/2"))
        self.assertEqual(extract_final("work \\boxed{a_{n}+1}"), "a_{n}+1")

    def test_parses_quality_and_kaggle_flags_conservatively(self):
        self.assertEqual(parsed_rate("0.875"), 0.875)
        self.assertIsNone(parsed_rate("unknown"))
        self.assertTrue(truthy(True))
        self.assertTrue(truthy("yes"))
        self.assertFalse(truthy("false"))
        self.assertFalse(truthy("0"))

    def test_combined_limit_matches_separate_prompt_completion_encoding(self):
        class StubTokenizer:
            class Encoded:
                def __init__(self, count):
                    self.ids = list(range(count))

            def encode(self, text):
                return self.Encoded(len(text.split()))

        count = supervised_token_count(StubTokenizer(), "one two", "three four five", eos_id=0)
        self.assertEqual(count, 8)  # prompt=4, completion=3, EOS=1


if __name__ == "__main__":
    unittest.main()
