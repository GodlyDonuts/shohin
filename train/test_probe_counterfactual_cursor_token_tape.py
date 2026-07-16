import unittest

import torch

from probe_counterfactual_cursor_token_tape import (
    CursorOnlyTapeReadout,
    MeanJointTapeReadout,
    TokenTapeReadout,
    decision,
    derangement,
    position_tape,
    standardize_tapes,
    verify_implementation,
)


class TokenTapeMechanicsTests(unittest.TestCase):
    def test_attention_masks_padding_and_normalizes(self):
        model = TokenTapeReadout(3, cursor_query=True, cursor_value=False)
        tape = torch.randn(2, 4, 3)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        scores, attention = model.forward_with_attention(tape, mask, torch.tensor([0, 4]))
        self.assertEqual(scores.shape, (2, 5))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2))
        self.assertTrue(bool(attention[~mask].eq(0).all()))

    def test_shared_value_uses_cursor_only_for_query(self):
        model = TokenTapeReadout(576, cursor_query=True, cursor_value=False)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 5_765)
        self.assertFalse(hasattr(model, "weight"))

    def test_cursor_specific_parameter_count(self):
        model = TokenTapeReadout(576, cursor_query=True, cursor_value=True)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 17_305)

    def test_source_only_query_is_cursor_invariant(self):
        model = TokenTapeReadout(3, cursor_query=False, cursor_value=False)
        tape = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        left = model(tape, mask, torch.tensor([0, 1]))
        right = model(tape, mask, torch.tensor([3, 4]))
        torch.testing.assert_close(left, right)

    def test_mean_joint_uses_exact_mask(self):
        model = MeanJointTapeReadout(2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.weight[2, 4] = torch.tensor([1.0, 2.0])
        tape = torch.tensor([[[2.0, 1.0], [4.0, 3.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        scores = model(tape, mask, torch.tensor([2]))
        self.assertEqual(float(scores[0, 4].detach()), 7.0)

    def test_cursor_only_ignores_tape(self):
        model = CursorOnlyTapeReadout(3)
        cursor = torch.tensor([1, 4])
        left = model(torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool), cursor)
        right = model(torch.randn(2, 3, 3), torch.zeros(2, 3, dtype=torch.bool), cursor)
        torch.testing.assert_close(left, right)

    def test_tape_standardization_uses_only_valid_train_tokens(self):
        train = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])
        development = torch.tensor([[[5.0], [7.0], [9.0]]])
        train_z, dev_z, mean, std = standardize_tapes(train, mask, development)
        torch.testing.assert_close(mean, torch.tensor([2.0]))
        torch.testing.assert_close(std, torch.tensor([1.0]))
        torch.testing.assert_close(train_z[0, :2].mean(), torch.tensor(0.0))
        torch.testing.assert_close(dev_z[0, 0], torch.tensor([3.0]))

    def test_derangement_is_deterministic_and_has_no_fixed_points(self):
        left = derangement(100, 17, "cpu")
        right = derangement(100, 17, "cpu")
        torch.testing.assert_close(left, right)
        self.assertTrue(bool(left.ne(torch.arange(100)).all()))

    def test_position_tape_is_source_invariant(self):
        tape = position_tape(3, 7, 8, "cpu")
        self.assertEqual(tape.shape, (3, 7, 8))
        torch.testing.assert_close(tape[0], tape[2])

    def test_implementation_binding_rejects_malformed_commit(self):
        with self.assertRaisesRegex(ValueError, "implementation commit is malformed"):
            verify_implementation("not-a-commit")

    def test_decision_requires_controls_and_exact_groups(self):
        def metric(value):
            return {"proportion": value, "numerator": int(value * 100), "denominator": 100}

        def arm(train, development, groups):
            return {"readout": {
                "train": {"restricted": {"cell_accuracy": metric(train)}},
                "development": {"restricted": {
                    "cell_accuracy": metric(development),
                    "exact_five_action_groups": metric(groups),
                }},
            }}

        arms = {
            **{
                f"pre_final_shared_seed{index}": arm(1.0, 0.96, 0.92)
                for index in range(3)
            },
            **{
                f"pre_final_cursor_specific_seed{index}": arm(1.0, 0.99, 0.99)
                for index in range(3)
            },
            "embedding_shared": arm(1.0, 0.20, 0.0),
            "position_shared": arm(1.0, 0.20, 0.0),
            "source_deranged_shared": arm(1.0, 0.20, 0.0),
            "mean_joint": arm(1.0, 0.50, 0.0),
            "source_only_tape": arm(0.20, 0.20, 0.0),
            "cursor_only": arm(0.40, 0.40, 0.0),
        }
        for value in arms.values():
            restricted = value["readout"]["development"]["restricted"]
            restricted["per_renderer_cell_accuracy"] = {
                "6": metric(restricted["cell_accuracy"]["proportion"]),
                "7": metric(restricted["cell_accuracy"]["proportion"]),
            }
            restricted["per_cursor_cell_accuracy"] = {
                str(index): metric(restricted["cell_accuracy"]["proportion"])
                for index in range(5)
            }
        observed = decision(arms)
        self.assertEqual(observed["decision"], "deep_shared_attention_bottleneck_available")
        self.assertFalse(observed["reasoning_claim_authorized"])

    def test_decision_reports_optimization_inconclusive(self):
        def metric(value):
            return {"proportion": value, "numerator": 0, "denominator": 100}

        def arm(train, development):
            restricted = {
                "cell_accuracy": metric(development),
                "exact_five_action_groups": metric(0.0),
                "per_renderer_cell_accuracy": {"6": metric(development)},
                "per_cursor_cell_accuracy": {
                    str(index): metric(development) for index in range(5)
                },
            }
            return {"readout": {
                "train": {"restricted": {"cell_accuracy": metric(train)}},
                "development": {"restricted": restricted},
            }}

        arms = {
            **{f"pre_final_shared_seed{i}": arm(0.5, 0.2) for i in range(3)},
            **{
                f"pre_final_cursor_specific_seed{i}": arm(0.5, 0.2)
                for i in range(3)
            },
            "embedding_shared": arm(0.5, 0.2),
            "position_shared": arm(0.5, 0.2),
            "source_deranged_shared": arm(0.5, 0.2),
            "mean_joint": arm(0.5, 0.2),
            "source_only_tape": arm(0.2, 0.2),
            "cursor_only": arm(0.4, 0.4),
        }
        observed = decision(arms)
        self.assertEqual(observed["decision"], "token_tape_optimization_inconclusive")


if __name__ == "__main__":
    unittest.main()
