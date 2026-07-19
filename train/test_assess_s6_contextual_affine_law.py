import unittest

from assess_s6_contextual_affine_law import build_assessment


def _arm(accuracy: float) -> dict[str, object]:
    return {
        "state_accuracy": accuracy,
        "answer_accuracy": accuracy,
        "depth_state": {str(depth): {"accuracy": accuracy} for depth in range(3, 9)},
        "multi_law_state_accuracy": accuracy,
    }


class S6AssessmentTests(unittest.TestCase):
    def _evaluation(self) -> dict[str, object]:
        return {
            "schema": "r12_s6_contextual_affine_law_development_eval_v1",
            "fit": {
                "treatment": {"atomic_train_accuracy": 1.0},
                "law_id_control": {"atomic_train_accuracy": 1.0},
            },
            "atomic_development": {"accuracy": 0.99},
            "arms": {
                "host": _arm(1.0),
                "treatment": _arm(0.995),
                "deranged_card": _arm(0.2),
                "one_witness": _arm(0.4),
                "state_reset": _arm(0.5),
                "law_id": _arm(0.1),
            },
            "nonce_name_invariance": {
                "all_rows_bit_identical": True,
            },
            "scale_diagnostic": _arm(0.85),
            "parameters": {
                "treatment": 4_753_677,
                "whole_system": 138_448_546,
                "law_id_control": 4_780_301,
            },
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "training_contract": "atomic destination cells from train laws only; zero recurrent labels",
        }

    def test_qualifies_only_when_every_gate_passes(self):
        assessment = build_assessment(self._evaluation())
        self.assertEqual(assessment["decision"], "qualify_s6_for_one_confirmation")
        self.assertTrue(all(assessment["gates"].values()))

    def test_rejects_weak_unseen_law_accuracy(self):
        evaluation = self._evaluation()
        evaluation["atomic_development"]["accuracy"] = 0.5
        assessment = build_assessment(evaluation)
        self.assertEqual(
            assessment["decision"], "reject_s6_contextual_affine_law_development"
        )
        self.assertFalse(assessment["gates"]["heldout_atomic_destination_at_least_95pct"])


if __name__ == "__main__":
    unittest.main()
