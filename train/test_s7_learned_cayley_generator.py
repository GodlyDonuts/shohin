from __future__ import annotations

import unittest

import torch

from s7_learned_cayley_generator import LearnedCayleyGenerator


class LearnedCayleyGeneratorTest(unittest.TestCase):
    def test_frozen_parameter_count(self) -> None:
        model = LearnedCayleyGenerator()
        self.assertEqual(model.num_params(), 218)
        self.assertEqual(model.total_system_params(), 133_695_087)
        self.assertLess(model.total_system_params(), 150_000_000)

    def test_discrete_views_follow_logits(self) -> None:
        model = LearnedCayleyGenerator((5,))
        successor = (2, 4, 1, 0, 3)
        with torch.no_grad():
            model.successor(5).fill_(-10.0)
            for source, destination in enumerate(successor):
                model.successor(5)[source, destination] = 10.0
            model.zero(5).fill_(-10.0)
            model.zero(5)[3] = 10.0
        self.assertEqual(model.discrete_successor(5), successor)
        self.assertEqual(model.discrete_zero(5), 3)


if __name__ == "__main__":
    unittest.main()
