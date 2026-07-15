import unittest

import probe_atomic_operation_formats as probe


class AtomicOperationFormatTests(unittest.TestCase):
    def test_prompts_are_answer_free(self):
        self.assertEqual(
            probe.format_prompt("question_answer", 12, "multiply", 7),
            "Question: Compute 12 times 7. Return only the final integer.\nAnswer:",
        )
        self.assertEqual(probe.format_prompt("bare_equation", 25, "remainder", 6), "25 % 6 =")
        self.assertEqual(
            probe.format_prompt("problem_work", 14, "subtract", 3),
            "Problem: Compute 14 minus 3.\nWork:",
        )

    def test_first_line_final(self):
        self.assertEqual(probe.parse_first_line_final(" 12 * 7 = 84\nAnswer: 84"), 84)
        self.assertEqual(probe.parse_first_line_final("\n  -11\nNext: 3"), -11)
        self.assertIsNone(probe.parse_first_line_final("No number here"))


if __name__ == "__main__":
    unittest.main()
