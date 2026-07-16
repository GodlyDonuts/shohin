import unittest

import torch

from probe_counterfactual_cursor_readout import (
    CursorOnlyReadout,
    FrozenScoreCalibrator,
    JointCursorReadout,
    SourceOnlyReadout,
    decision,
    exact_full_vocabulary_loss,
    standardize,
)


class ReadoutMechanicsTests(unittest.TestCase):
    def test_joint_readout_selects_cursor_slice(self):
        model = JointCursorReadout(2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.weight[1, 3] = torch.tensor([2.0, -1.0])
        hidden = torch.tensor([[4.0, 1.0], [4.0, 1.0]])
        scores = model(hidden, torch.tensor([1, 2]))
        self.assertEqual(float(scores[0, 3].detach()), 7.0)
        self.assertTrue(bool(scores[1].eq(0).all()))

    def test_source_only_is_cursor_invariant(self):
        model = SourceOnlyReadout(3)
        hidden = torch.randn(4, 3)
        left = model(hidden, torch.tensor([0, 1, 2, 3]))
        right = model(hidden, torch.tensor([4, 3, 2, 1]))
        torch.testing.assert_close(left, right)

    def test_cursor_only_is_source_invariant(self):
        model = CursorOnlyReadout(3)
        with torch.no_grad():
            model.table.weight.copy_(torch.arange(25).reshape(5, 5))
        cursor = torch.tensor([0, 4])
        left = model(torch.randn(2, 3), cursor)
        right = model(torch.randn(2, 3), cursor)
        torch.testing.assert_close(left, right)

    def test_exact_valve_loss_matches_explicit_vocabulary(self):
        full = torch.tensor([
            [0.2, -0.4, 1.1, 0.7, -0.1, 0.5, -1.0],
            [-0.3, 0.8, 0.1, -0.2, 0.9, -0.7, 0.4],
        ])
        label_ids = torch.tensor([0, 2, 4, 5, 6])
        action = full.index_select(1, label_ids)
        masked = full.clone()
        masked[:, label_ids] = -torch.inf
        non_lse = torch.logsumexp(masked, dim=-1)
        delta = torch.tensor([
            [0.1, -0.2, 0.3, 0.4, -0.5],
            [-0.1, 0.2, -0.3, 0.5, 0.7],
        ])
        target = torch.tensor([2, 4])
        observed = exact_full_vocabulary_loss(action, non_lse, delta, target)
        explicit = full.clone()
        explicit[:, label_ids] += delta
        expected = torch.nn.functional.cross_entropy(
            explicit, label_ids.index_select(0, target),
        )
        torch.testing.assert_close(observed, expected)

    def test_standardization_uses_training_statistics(self):
        train = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        other = torch.tensor([[5.0, 10.0]])
        train_z, other_z, mean, std = standardize(train, other)
        torch.testing.assert_close(mean, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(std, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(train_z.mean(0), torch.zeros(2))
        torch.testing.assert_close(other_z, torch.tensor([[3.0, 3.0]]))

    def test_calibrator_has_positive_gain_and_common_bias(self):
        model = FrozenScoreCalibrator()
        scores = torch.tensor([[1.0, -2.0, 0.5, 3.0, -1.0]])
        delta = model(scores)
        self.assertGreater(float(model.alpha.detach()), 0.0)
        self.assertEqual(int(delta.argmax(dim=-1)), int(scores.argmax(dim=-1)))
        gaps = delta[0, 1:] - delta[0, :-1]
        torch.testing.assert_close(gaps, model.alpha * (scores[0, 1:] - scores[0, :-1]))

    def test_decision_applies_source_control_per_surface(self):
        def metric(value):
            return {"proportion": value, "numerator": int(value * 100), "denominator": 100}

        def arm(readout_cell, readout_groups=1.0, train_cell=1.0, full=1.0):
            return {
                "readout": {
                    "train": {"restricted": {"cell_accuracy": metric(train_cell)}},
                    "development": {"restricted": {
                        "cell_accuracy": metric(readout_cell),
                        "exact_five_action_groups": metric(readout_groups),
                    }},
                },
                "calibration": {"development": {"full_vocabulary": {
                    "cell_accuracy": metric(full),
                    "exact_five_action_groups": metric(full),
                    "per_renderer_cell_accuracy": {"6": metric(full), "7": metric(full)},
                }}},
            }

        arms = {
            "pre_joint": arm(0.99),
            "post_joint": arm(0.99),
            "pre_source_only": arm(0.50, full=0.20),
            "post_source_only": arm(0.20, full=0.20),
            "cursor_only": arm(0.40, full=0.20),
        }
        observed = decision(arms)
        self.assertFalse(
            observed["surfaces"]["pre"]["oracle_cursor_indexed_linear_separability"]
        )
        self.assertTrue(
            observed["surfaces"]["post"]["oracle_cursor_indexed_linear_separability"]
        )
        self.assertEqual(observed["admitted_surfaces"], ["post"])
        self.assertFalse(observed["actuation_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
