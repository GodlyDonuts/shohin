#!/usr/bin/env python3
"""Focused non-network tests for the TACO-derived code curator."""
import json
import unittest

from curate_taco_verified import normalized_question, parse_cases, python_solutions


class TacoCuratorTests(unittest.TestCase):
    def test_parse_stdio_cases(self):
        raw = json.dumps({"inputs": ["2\n", "4\n"], "outputs": ["4\n", "16\n"]})
        self.assertEqual(parse_cases(raw, max_tests=1, max_case_chars=16), [("2\n", "4\n")])
        self.assertEqual(
            parse_cases(raw, max_tests=0, max_case_chars=16),
            [("2\n", "4\n"), ("4\n", "16\n")],
        )

    def test_reject_function_contract(self):
        raw = json.dumps({"fn_name": "square", "inputs": ["2"], "outputs": ["4"]})
        self.assertIsNone(parse_cases(raw, max_tests=3, max_case_chars=16))

    def test_keep_only_syntax_valid_python(self):
        row = {"solutions": ["def bad(:\n", "print('ok')"]}
        self.assertEqual(list(python_solutions(row, 100)), ["print('ok')"])

    def test_normalized_question_matches_quality_gate_identity(self):
        self.assertEqual(
            normalized_question("Sort [3, 1]!"),
            normalized_question("sort 3 1"),
        )


if __name__ == "__main__":
    unittest.main()
