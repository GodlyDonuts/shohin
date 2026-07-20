from __future__ import annotations

import unittest

from assess_s9_1_alpha_closed import assess


def passing_evaluation():
    depth = {str(value): {"accuracy": 0.96} for value in range(3, 9)}
    arm = {"state_accuracy": 0.96, "answer_accuracy": 0.96, "depth": depth, "total": 2048}
    low = {"state_accuracy": 0.10, "answer_accuracy": 0.10, "depth": depth, "total": 2048}
    fit = {"charged_views": 48_000, "updates": 750}
    return {
        "rows": 2048,
        "span": {"f1": 0.99, "class_exact_accuracy": 0.96},
        "graph": {
            "valid": 2000,
            "valid_accuracy": 2000 / 2048,
            "exact_accuracy": 0.96,
            "no_class_exact_accuracy": 0.80,
            "shuffled_exact_accuracy": 0.0,
            "uniform_exact": 0,
            "source_free_exact_accuracy": 0.0,
            "unconstrained_exact_accuracy": 0.94,
        },
        "arms": {
            "treatment": arm,
            "no_class_message": arm,
            "reversed_links": low,
            "deranged_cards": low,
            "one_witness": low,
            "state_reset": low,
            "early_nil": low,
        },
        "invariance": {
            "eligible": 2000,
            "class_reindex": 2000,
            "relation_storage_reindex": 2000,
            "nonce_eligible": 2000,
            "nonce_graph_identical": 2000,
            "nonce_state_identical": 2000,
            "nonce_answer_identical": 2000,
        },
        "parameters": {"complete_system": 134_580_264},
        "fit": {"treatment": fit, "no_class": fit, "shuffled": fit},
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }


class S91AssessmentTest(unittest.TestCase):
    def test_passing_fixture_qualifies(self):
        result = assess(passing_evaluation())
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(
            result["decision"],
            "qualify_s9_1_alpha_closed_for_fresh_confirmation",
        )

    def test_recode_eligibility_is_strict(self):
        evaluation = passing_evaluation()
        evaluation["invariance"]["nonce_eligible"] = 1999
        result = assess(evaluation)
        self.assertFalse(result["gates"]["operation_nonce_all_valid_eligible"])


if __name__ == "__main__":
    unittest.main()
