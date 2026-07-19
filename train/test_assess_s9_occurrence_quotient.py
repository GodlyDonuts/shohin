from __future__ import annotations

import unittest

from assess_s9_occurrence_quotient import assess


def passing_evaluation():
    depth = {str(value): {"accuracy": 0.90} for value in range(3, 9)}
    arm = {"state_accuracy": 0.90, "answer_accuracy": 0.91, "depth": depth, "total": 2048}
    low = {"state_accuracy": 0.10, "answer_accuracy": 0.20, "depth": depth, "total": 2048}
    return {
        "rows": 2048,
        "span": {"f1": 0.99, "class_exact_accuracy": 0.96},
        "graph": {
            "valid": 1900, "valid_accuracy": 0.93,
            "exact_accuracy": 0.92, "no_class_exact_accuracy": 0.86,
            "shuffled_exact_accuracy": 0.01,
        },
        "arms": {
            "treatment": arm, "no_class_message": arm,
            "reversed_links": low, "deranged_cards": low,
            "one_witness": low, "state_reset": low, "early_nil": low,
        },
        "invariance": {
            "eligible": 1900, "class_reindex": 1900,
            "relation_storage_reindex": 1900,
            "nonce_eligible": 1900, "nonce_identical": 1900,
        },
        "parameters": {"complete_system": 134580046},
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }


class S9AssessmentTest(unittest.TestCase):
    def test_passing_fixture_qualifies(self):
        result = assess(passing_evaluation())
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(
            result["decision"],
            "qualify_s9_occurrence_quotient_for_fresh_confirmation",
        )

    def test_no_class_attribution_gate_is_immutable(self):
        evaluation = passing_evaluation()
        evaluation["graph"]["no_class_exact_accuracy"] = 0.89
        result = assess(evaluation)
        self.assertFalse(result["gates"]["exact_graph_plus_5pp_over_no_class"])
        self.assertEqual(result["decision"], "reject_s9_occurrence_quotient_v1")


if __name__ == "__main__":
    unittest.main()
