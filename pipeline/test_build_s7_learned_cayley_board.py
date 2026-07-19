from __future__ import annotations

import random
import unittest

from build_s7_learned_cayley_board import (
    _audit,
    _binding,
    build_atomic_development_rows,
    build_program_rows,
    build_training_rows,
    s7_law_pools,
)
from s7_learned_cayley_law import PRIMARY_MODULI


class BuildS7BoardTest(unittest.TestCase):
    def test_fresh_law_pools_are_disjoint(self) -> None:
        for modulus in PRIMARY_MODULI:
            pools = s7_law_pools(modulus)
            self.assertFalse(set(pools["train"]) & set(pools["development"]))
            self.assertFalse(set(pools["train"]) & set(pools["confirmation"]))
            self.assertFalse(
                set(pools["development"]) & set(pools["confirmation"])
            )

    def test_small_board_passes_audit(self) -> None:
        rng = random.Random(1234567)
        bindings = {modulus: _binding(rng, modulus) for modulus in PRIMARY_MODULI}
        generator, atomic_train = build_training_rows(bindings)
        atomic_dev = build_atomic_development_rows(bindings)
        development = build_program_rows(rng, bindings, "development", 36)
        confirmation = build_program_rows(rng, bindings, "confirmation", 36)
        audit = _audit(
            generator, atomic_train, atomic_dev, development, confirmation
        )
        self.assertEqual(audit["generator_training_rows"], 23)
        self.assertEqual(audit["zero_anchor_count"], 3)
        self.assertEqual(audit["development_rows"], 36)
        self.assertEqual(audit["confirmation_rows"], 36)
        self.assertEqual(audit["development_accesses"], 0)
        self.assertEqual(audit["confirmation_accesses"], 0)


if __name__ == "__main__":
    unittest.main()
