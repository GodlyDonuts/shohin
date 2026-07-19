from __future__ import annotations

import copy
import unittest

from assess_s7_learned_cayley import assess


def _score(value: float) -> dict[str, object]:
    return {
        "state_accuracy": value,
        "answer_accuracy": value,
        "depth_state": {str(depth): {"accuracy": value} for depth in range(3, 9)},
    }


class AssessS7Test(unittest.TestCase):
    def passing_evaluation(self) -> dict[str, object]:
        return {
            "training_contract": (
                "treatment/false see 23 successor cells plus three zero anchors"
            ),
            "fit": {
                "treatment": {"successor_accuracy": 1.0, "zero_accuracy": 1.0},
                "false_generator": {"successor_accuracy": 1.0, "zero_accuracy": 1.0},
                "ordinary_transformer": {"atomic_train_accuracy": 1.0},
            },
            "atomic_development": {
                "treatment": {"accuracy": 1.0},
                "ordinary_transformer": {"accuracy": 0.2},
            },
            "arms": {
                "host": _score(1.0),
                "treatment": _score(1.0),
                "ordinary_transformer": _score(0.2),
                "stride_two_generator": _score(0.1),
                "deranged_card": _score(0.1),
                "one_witness": _score(0.3),
                "state_reset": _score(0.5),
            },
            "nonce_operation_invariance": {"all_rows_bit_identical": True},
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "parameters": {"treatment": 218, "whole_system": 133_695_087},
        }

    def test_all_passing_qualifies(self) -> None:
        result = assess(self.passing_evaluation())
        self.assertEqual(
            result["decision"],
            "qualify_s7_learned_cayley_for_fresh_confirmation",
        )
        self.assertTrue(all(result["gates"].values()))

    def test_weak_causal_control_rejects(self) -> None:
        evaluation = copy.deepcopy(self.passing_evaluation())
        evaluation["arms"]["stride_two_generator"] = _score(0.5)
        result = assess(evaluation)
        self.assertEqual(result["decision"], "reject_s7_learned_cayley_development")
        self.assertFalse(result["gates"]["treatment_beats_false_generator_by_60pp"])


if __name__ == "__main__":
    unittest.main()
