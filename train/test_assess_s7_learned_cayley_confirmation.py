from __future__ import annotations

import copy
import unittest

from assess_s7_learned_cayley_confirmation import assess_confirmation


def _score(value: float) -> dict[str, object]:
    return {
        "state_accuracy": value,
        "answer_accuracy": value,
        "depth_state": {str(depth): {"accuracy": value} for depth in range(3, 9)},
    }


class AssessS7ConfirmationTest(unittest.TestCase):
    def passing_evaluation(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": (
                "c26e2cb6ef54ff409b580b3828c6ace4369423cf67b11bd66d9af05c93db4607"
            ),
            "development_assessment_sha256": (
                "2ef4d5ee053d2bf599726aa8db6fa39305f4fc112c0a35af291fe6e109c8bbc4"
            ),
            "confirmation_sha256": (
                "c2eb8d5c5dd285dfcb60389c3067c4842e47872d64b5233681c32c8542434bc5"
            ),
            "training_contract": (
                "treatment/false see 23 successor cells plus three zero anchors"
            ),
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
            "confirmation_accesses": 1,
            "parameters": {"treatment": 218, "whole_system": 133_695_087},
        }

    def test_all_passing_confirms(self) -> None:
        result = assess_confirmation(self.passing_evaluation())
        self.assertEqual(
            result["decision"],
            "confirm_s7_learned_cayley_contextual_law_compilation",
        )
        self.assertTrue(all(result["gates"].values()))

    def test_hash_mismatch_rejects(self) -> None:
        evaluation = copy.deepcopy(self.passing_evaluation())
        evaluation["confirmation_sha256"] = "0" * 64
        result = assess_confirmation(evaluation)
        self.assertEqual(result["decision"], "reject_s7_learned_cayley_confirmation")
        self.assertFalse(result["gates"]["bound_confirmation_hash"])


if __name__ == "__main__":
    unittest.main()
