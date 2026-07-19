import unittest

import torch

from s6_contextual_affine_law_inducer import (
    ContextualAffineLawInducer,
    LawIdMemorizer,
)


class ContextualAffineLawInducerTests(unittest.TestCase):
    def test_shape_mask_and_parameter_cap(self):
        model = ContextualAffineLawInducer(width=64, layers=2, heads=4, feedforward=128)
        modulus = torch.tensor([5, 7])
        logits = model(
            modulus,
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([4, 6]),
        )
        self.assertEqual(tuple(logits.shape), (2, 13))
        self.assertTrue(torch.isneginf(logits[0, 5:]).all())
        self.assertTrue(torch.isneginf(logits[1, 7:]).all())
        production = ContextualAffineLawInducer()
        self.assertLess(production.num_params(), 8_000_000)
        self.assertLess(production.total_system_params(), 150_000_000)

    def test_backward_is_finite(self):
        model = ContextualAffineLawInducer(width=32, layers=1, heads=4, feedforward=64)
        logits = model(
            torch.tensor([5, 5]),
            torch.tensor([1, 2]),
            torch.tensor([2, 4]),
            torch.tensor([3, 4]),
        )
        torch.nn.functional.cross_entropy(logits, torch.tensor([4, 1])).backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_law_id_control_has_favorable_parameter_count(self):
        treatment = ContextualAffineLawInducer(
            width=32, layers=1, heads=4, feedforward=64
        )
        control = LawIdMemorizer(
            train_law_count=103, width=32, layers=1, heads=4, feedforward=64
        )
        self.assertGreater(control.num_params(), treatment.num_params())
        logits = control(
            torch.tensor([5]),
            torch.tensor([control.oov_law_id]),
            torch.tensor([3]),
        )
        self.assertEqual(tuple(logits.shape), (1, 13))


if __name__ == "__main__":
    unittest.main()
