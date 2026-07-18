#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from rescore_ssc_first_integer import answer_appears, first_integer, last_integer, main


class FirstIntegerRescoreTest(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(first_integer("12 then 99"), 12)
        self.assertEqual(last_integer("12 then 99"), 99)
        self.assertTrue(answer_appears("oops 371 done", 371))
        self.assertFalse(answer_appears("370", 371))

    def test_cli_echo_frozen(self):
        payload = {
            "rows": [
                {
                    "id": "t0",
                    "family": "multiply_subtract",
                    "answer": 10,
                    "whole_problem_work": {
                        "correct": False,
                        "response": "5\n10\nQuestion: again",
                    },
                },
                {
                    "id": "t1",
                    "family": "modular_update",
                    "answer": 3,
                    "whole_problem_work": {
                        "correct": True,
                        "response": "3",
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.json"
            out = Path(td) / "out.json"
            src.write_text(json.dumps(payload))
            import sys

            sys.argv = ["rescore_ssc_first_integer.py", "--result", str(src), "--out", str(out)]
            main()
            summary = json.loads(out.read_text())["summary"]
            self.assertEqual(summary["totals"]["first_integer_correct"], 1)
            self.assertEqual(summary["totals"]["answer_appears_in_segment"], 2)
            self.assertEqual(summary["totals"]["frozen_last_integer_correct"], 1)


if __name__ == "__main__":
    unittest.main()
