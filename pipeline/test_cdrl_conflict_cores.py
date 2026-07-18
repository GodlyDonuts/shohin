#!/usr/bin/env python3
"""Tests for R12 CDRL CPU core extraction mechanics."""

from __future__ import annotations

import unittest

from pipeline.cdrl_conflict_cores import (
    build_heisenberg_mechanics_board,
    build_free_word_negative_board,
    evaluate_mechanics_gates,
    extract_heisenberg_core,
    extract_register_core,
    extract_free_word_core,
    heisenberg_residual_key,
)


class HeisenbergCoreTests(unittest.TestCase):
    def test_padding_only_collapses_to_empty_core(self) -> None:
        extraction = extract_heisenberg_core(("P", "P", "P"), modulus=5)
        self.assertEqual(extraction.core, ())
        self.assertEqual(extraction.residual_key, (0, 0, 0))

    def test_padding_around_A_strips_to_A(self) -> None:
        extraction = extract_heisenberg_core(("P", "A", "P", "P"), modulus=5)
        self.assertEqual(extraction.core, ("A",))
        self.assertEqual(extraction.residual_key, heisenberg_residual_key(("A",), modulus=5))

    def test_core_preserves_noncommutative_AB(self) -> None:
        history = ("P", "A", "P", "B", "P")
        extraction = extract_heisenberg_core(history, modulus=5)
        self.assertEqual(extraction.core, ("A", "B"))
        self.assertEqual(
            heisenberg_residual_key(extraction.core, modulus=5),
            heisenberg_residual_key(history, modulus=5),
        )
        # Order matters: B then A is a different residual.
        self.assertNotEqual(
            heisenberg_residual_key(("A", "B"), modulus=5),
            heisenberg_residual_key(("B", "A"), modulus=5),
        )

    def test_lex_min_prefers_earlier_indices_at_equal_length(self) -> None:
        extraction = extract_heisenberg_core(("A", "P", "P"), modulus=5)
        self.assertEqual(extraction.core, ("A",))


class FreeWordNegativeTests(unittest.TestCase):
    def test_core_equals_history(self) -> None:
        history = ("A", "B", "C")
        extraction = extract_free_word_core(history)
        self.assertEqual(extraction.core, history)

    def test_repeats_are_essential(self) -> None:
        history = ("A", "A", "A")
        extraction = extract_free_word_core(history)
        self.assertEqual(extraction.core, history)


class RegisterOverwriteUnitTests(unittest.TestCase):
    def test_overwrite_keeps_last_write_only_if_sufficient(self) -> None:
        history = ("W0:1", "W0:2")
        extraction = extract_register_core(history, modulus=4, n_registers=3)
        self.assertEqual(extraction.core, ("W0:2",))
        self.assertNotEqual(extraction.core, history)


class MechanicsGateTests(unittest.TestCase):
    def test_all_cpu_gates_pass(self) -> None:
        report = evaluate_mechanics_gates()
        self.assertTrue(report["all_pass"], report)

    def test_boards_are_deterministic(self) -> None:
        a = build_heisenberg_mechanics_board(modulus=5)
        b = build_heisenberg_mechanics_board(modulus=5)
        self.assertEqual(a["board_sha256"], b["board_sha256"])
        c = build_free_word_negative_board()
        d = build_free_word_negative_board()
        self.assertEqual(c["board_sha256"], d["board_sha256"])

    def test_heisenberg_board_strips_all_padding(self) -> None:
        board = build_heisenberg_mechanics_board(modulus=5)
        for row in board["rows"]:
            self.assertNotIn("P", row["core"])


if __name__ == "__main__":
    unittest.main()
