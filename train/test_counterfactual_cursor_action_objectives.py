import unittest

import torch

from counterfactual_cursor_action_objectives import (
    adjacent_equivariance_loss,
    cursor_interchange_loss,
    relation_losses,
    renderer_invariance_loss,
    restricted_action_logits,
)


class CounterfactualCursorActionObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.renderers = 3
        left_labels = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ])
        right_labels = left_labels.clone()
        right_labels[:, 1], right_labels[:, 2] = (
            left_labels[:, 2], left_labels[:, 1]
        )
        self.labels = torch.stack((left_labels, right_labels))
        logits = torch.full((2, self.renderers, 5, 5), -8.0)
        for side in range(2):
            for renderer in range(self.renderers):
                for cursor in range(5):
                    logits[side, renderer, cursor, self.labels[side, renderer, cursor]] = 8.0
        self.logits = logits

    def test_restricted_logits_preserve_frozen_order(self):
        full = torch.arange(16000, dtype=torch.float32).reshape(2, 8000)
        ids = torch.tensor([820, 5498, 4307, 7486, 2165])
        selected = restricted_action_logits(full, ids)
        self.assertEqual(
            selected.tolist(),
            [[820, 5498, 4307, 7486, 2165], [8820, 13498, 12307, 15486, 10165]],
        )
        with self.assertRaisesRegex(ValueError, "frozen"):
            restricted_action_logits(full, torch.tensor([1, 1, 2, 3, 4]))

    def test_exact_relation_has_near_zero_equivariance_and_invariance(self):
        adjacent, adjacent_count = adjacent_equivariance_loss(
            self.logits, swap_index=1, sham=False,
        )
        renderer, renderer_count = renderer_invariance_loss(self.logits, sham=False)
        self.assertLess(adjacent.item(), 1e-5)
        self.assertLess(renderer.item(), 1e-5)
        self.assertEqual(adjacent_count, 2 * self.renderers * 5 // 2)
        self.assertEqual(renderer_count, 2 * 3 * 5)

    def test_sham_relations_reject_exact_controller(self):
        adjacent, adjacent_count = adjacent_equivariance_loss(
            self.logits, swap_index=1, sham=True,
        )
        renderer, renderer_count = renderer_invariance_loss(self.logits, sham=True)
        self.assertGreater(adjacent.item(), 1.0)
        self.assertGreater(renderer.item(), 1.0)
        self.assertEqual(adjacent_count, self.renderers * 5)
        self.assertEqual(renderer_count, 2 * 3 * 5)

    def test_cursor_interchange_has_frozen_directed_pair_count(self):
        correct, correct_count = cursor_interchange_loss(
            self.logits, self.labels, sham=False, margin=1.0,
        )
        sham, sham_count = cursor_interchange_loss(
            self.logits, self.labels, sham=True, margin=1.0,
        )
        self.assertLess(correct.item(), 1e-4)
        self.assertGreater(sham.item(), 1.0)
        self.assertEqual(correct_count, 2 * self.renderers * 20)
        self.assertEqual(sham_count, correct_count)

    def test_combined_counts_and_validation(self):
        result = relation_losses(self.logits, self.labels, swap_index=1)
        self.assertEqual(result.cursor_pairs, 120)
        self.assertEqual(result.adjacent_pairs, 15)
        self.assertEqual(result.renderer_pairs, 30)
        self.assertLess(result.total().item(), 1e-4)
        broken = self.labels.clone()
        broken[0, 0, 1] = 0
        with self.assertRaisesRegex(ValueError, "one target"):
            relation_losses(self.logits, broken, swap_index=1)


if __name__ == "__main__":
    unittest.main()
