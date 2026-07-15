import json
import unittest

import assess_raw_continuation_confirmation as assessor
import probe_raw_continuation_confirmation as confirmation


class RawContinuationProbeTests(unittest.TestCase):
    def test_numbered_followup_is_not_part_of_first_answer(self):
        response = "11 + 7 = 18\n18 * 4 = 72\n72 - 13 = 59\n\nQuestion 2: unrelated 99"
        self.assertEqual(
            assessor.answer_segment(response),
            "11 + 7 = 18\n18 * 4 = 72\n72 - 13 = 59",
        )

    def test_assessor_scores_terminal_answer_and_intermediates(self):
        case = {"answer": 59, "required_intermediates": [18, 72]}
        scored = assessor.score("11 + 7 = 18; 18 * 4 = 72; 72 - 13 = 59", case)
        self.assertTrue(scored["final_correct"])
        self.assertTrue(scored["intermediates_present"])

    def test_confirmation_case_manifest_is_deterministic(self):
        first = confirmation.build_cases()
        second = confirmation.build_cases()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["id"] for row in first}), 20)
        json.dumps(first, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
