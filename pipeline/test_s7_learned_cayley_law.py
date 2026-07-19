from __future__ import annotations

import inspect
import unittest

from s6_contextual_affine_law import AffineLaw, apply_law
from s7_learned_cayley_law import (
    SymbolBinding,
    apply_compiled_law,
    compile_destination,
    stride_two_successor,
    validate_successor,
)


class LearnedCayleyLawTest(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = SymbolBinding(5, (3, 0, 4, 1, 2))

    def test_binding_roundtrip(self) -> None:
        for latent in range(5):
            self.assertEqual(self.binding.decode(self.binding.encode(latent)), latent)

    def test_successor_is_single_cycle(self) -> None:
        self.assertEqual(
            validate_successor(self.binding.successor, self.binding.zero_symbol),
            self.binding.successor,
        )

    def test_all_hidden_affine_cells_compile(self) -> None:
        for slope in range(1, 5):
            for intercept in range(5):
                law = AffineLaw(5, slope, intercept)
                y0, y1 = self.binding.card(law)
                for observed in range(5):
                    predicted = compile_destination(
                        self.binding.successor,
                        self.binding.zero_symbol,
                        y0,
                        y1,
                        observed,
                    )
                    self.assertEqual(predicted, self.binding.destination(law, observed))

    def test_recurrent_state_matches_hidden_oracle(self) -> None:
        state = (2, 4, 1, 0, 3)
        law = AffineLaw(5, 3, 2)
        identity = 4
        y0, y1 = self.binding.card(law)
        compiled = apply_compiled_law(
            state,
            identity,
            self.binding.successor,
            self.binding.zero_symbol,
            y0,
            y1,
        )
        source_observed = state.index(identity)
        expected_destination = self.binding.destination(law, source_observed)
        expected = list(state)
        expected.insert(expected_destination, expected.pop(source_observed))
        self.assertEqual(compiled, tuple(expected))

    def test_stride_two_is_distinct_complete_cycle(self) -> None:
        false_successor = stride_two_successor(
            self.binding.successor, self.binding.zero_symbol
        )
        self.assertNotEqual(false_successor, self.binding.successor)
        self.assertEqual(
            validate_successor(false_successor, self.binding.zero_symbol),
            false_successor,
        )

    def test_equal_witnesses_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compile_destination(
                self.binding.successor,
                self.binding.zero_symbol,
                2,
                2,
                1,
            )

    def test_compiler_source_has_no_field_solver(self) -> None:
        source = inspect.getsource(compile_destination)
        for forbidden in ("%", "infer_affine_law", ".destination(", "slope *"):
            self.assertNotIn(forbidden, source)

    def test_canonical_oracle_is_not_treatment_dependency(self) -> None:
        self.assertNotIn("apply_law", inspect.getsource(compile_destination))
        self.assertIsNotNone(apply_law)


if __name__ == "__main__":
    unittest.main()

