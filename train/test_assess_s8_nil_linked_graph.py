from __future__ import annotations

import copy
import unittest

from assess_s8_nil_linked_graph import assess


def _arm(value: float) -> dict[str, object]:
    return {
        "state_accuracy": value,
        "answer_accuracy": value,
        "depth_state": {
            str(depth): {"accuracy": value} for depth in range(3, 9)
        },
    }


class AssessS8NilLinkedGraphTest(unittest.TestCase):
    def passing(self) -> dict[str, object]:
        return {
            "graph": {
                "valid": 2028,
                "valid_accuracy": 0.99,
                "exact_accuracy": 0.95,
                "count_halt_accuracy": 0.99,
                "shuffled_exact_accuracy": 0.01,
            },
            "arms": {
                "treatment": _arm(0.95),
                "gold_graph": _arm(1.0),
                "ordinary_sequence_parser": _arm(0.96),
                "storage_order_shortcut": _arm(0.1),
                "reversed_links": _arm(0.2),
                "deranged_cards": _arm(0.1),
                "one_witness": _arm(0.2),
                "state_reset": _arm(0.4),
                "early_nil": _arm(0.2),
            },
            "invariance": {
                "graph_reindex_eligible": 2028,
                "graph_reindex_accuracy": 1.0,
                "operation_nonce_eligible": 2028,
                "operation_nonce_accuracy": 1.0,
            },
            "parameters": {
                "graph_compiler": 9_000_000,
                "complete_system": 135_000_000,
            },
            "training_contract": (
                "graph fields; zero final-state, answer, recurrent supervision"
            ),
            "development_accesses": 1,
            "confirmation_accesses": 0,
        }

    def test_passing_qualifies(self) -> None:
        result = assess(self.passing())
        self.assertEqual(
            result["decision"],
            "qualify_s8_nil_linked_law_graph_for_fresh_confirmation",
        )
        self.assertTrue(all(result["gates"].values()))

    def test_nonce_failure_rejects(self) -> None:
        evaluation = copy.deepcopy(self.passing())
        evaluation["invariance"]["operation_nonce_accuracy"] = 0.99
        result = assess(evaluation)
        self.assertEqual(result["decision"], "reject_s8_nil_linked_law_graph_v1")
        self.assertFalse(result["gates"]["operation_nonce_bit_identical"])


if __name__ == "__main__":
    unittest.main()
