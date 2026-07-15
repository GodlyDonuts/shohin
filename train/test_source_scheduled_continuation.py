import unittest

import probe_source_scheduled_continuation as ssc


class SourceScheduledContinuationTests(unittest.TestCase):
    def test_schedules(self):
        row = {
            "family": "sequential_state",
            "question": "Start at 11, add 7, multiply by 4, then subtract 13.",
        }
        self.assertEqual(
            ssc.parse_schedule(row),
            (11, [("add", 7), ("multiply", 4), ("subtract", 13)]),
        )

    def test_base_uses_horner_schedule(self):
        row = {"family": "base_conversion", "question": "Convert the base-7 numeral 352 to base 10."}
        self.assertEqual(
            ssc.parse_schedule(row),
            (3, [("multiply", 7), ("add", 5), ("multiply", 7), ("add", 2)]),
        )

    def test_first_integer(self):
        self.assertEqual(ssc.first_integer(" 59\nNext example: 12"), 59)
        self.assertIsNone(ssc.first_integer("no integer"))


if __name__ == "__main__":
    unittest.main()
