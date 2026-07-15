#!/usr/bin/env python3
"""Exhaustive contracts for the DWEPR Stage-A residual oracle."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from wgrq_residual_oracle import (  # noqa: E402
    DWEPR,
    EVENT_ALPHABET,
    FLIP,
    ROTATE,
    answer_after_rotations,
    apply_event,
    apply_word,
    cancellation_gadget_report,
    cancellation_gadgets,
    canonical_access_word,
    canonical_code,
    canonical_representative,
    deserialize_edge_vector,
    edge_vector,
    event_counts,
    exhaustive_symbolic_check,
    flip,
    inverse_rotate,
    quotient_read,
    quotient_transition,
    read,
    residual_equivalent,
    rotate,
    run_stage_a_symbolic_gates,
    serialize_edge_vector,
    shortest_witness_depth,
    state_mask,
)


class DWEPRMechanicsTests(unittest.TestCase):
    def test_physical_orientation_and_reversibility(self) -> None:
        n = 4
        state = 0b1011
        self.assertEqual(rotate(state, n), 0b1101)
        self.assertEqual(inverse_rotate(rotate(state, n), n), state)
        self.assertEqual(flip(state, n), 0b1010)
        self.assertEqual(flip(flip(state, n), n), state)
        self.assertEqual(read(state, n), 0)

    def test_edge_code_is_exact_observable_quotient(self) -> None:
        for n in (3, 6):
            classes: dict[str, list[int]] = defaultdict(list)
            for state in range(1 << n):
                edges = edge_vector(state, n)
                self.assertEqual(sum(edges) % 2, 0)
                code = canonical_code(state, n)
                self.assertEqual(len(code), n - 1)
                self.assertEqual(deserialize_edge_vector(code), edges)
                self.assertEqual(canonical_code(canonical_representative(code), n), code)
                classes[code].append(state)
            self.assertEqual(len(classes), 1 << (n - 1))
            self.assertTrue(all(len(members) == 2 for members in classes.values()))
            self.assertTrue(
                all(members[0] ^ members[1] == state_mask(n) for members in classes.values())
            )

    def test_quotient_transitions_are_representative_independent(self) -> None:
        n = 6
        for state in range(1 << n):
            complement = state ^ state_mask(n)
            code = canonical_code(state, n)
            self.assertEqual(quotient_read(code), read(state, n))
            for event in EVENT_ALPHABET:
                expected = quotient_transition(code, event)
                self.assertEqual(canonical_code(apply_event(state, event, n), n), expected)
                self.assertEqual(canonical_code(apply_event(complement, event, n), n), expected)

    def test_shortest_witnesses_cover_the_tight_bound(self) -> None:
        for n in (3, 6):
            depths: set[int] = set()
            for left in range(1 << n):
                for right in range(1 << n):
                    witness = shortest_witness_depth(left, right, n)
                    if residual_equivalent(left, right, n):
                        self.assertIsNone(witness)
                        continue
                    self.assertIsNotNone(witness)
                    assert witness is not None
                    self.assertTrue(
                        all(
                            answer_after_rotations(left, earlier, n)
                            == answer_after_rotations(right, earlier, n)
                            for earlier in range(witness)
                        )
                    )
                    self.assertNotEqual(
                        answer_after_rotations(left, witness, n),
                        answer_after_rotations(right, witness, n),
                    )
                    depths.add(witness)
            self.assertEqual(depths, set(range(n - 1)))
            self.assertEqual(max(depths), n - 2)

    def test_canonical_access_words_reach_every_state(self) -> None:
        for n in (4, 6, 8):
            for state in range(1 << n):
                word = canonical_access_word(state, n)
                self.assertEqual(apply_word(0, word, n), state)
                self.assertEqual(event_counts(word)[ROTATE], n)
                self.assertEqual(event_counts(word)[FLIP], state.bit_count())
                self.assertLessEqual(len(word), 2 * n)

    def test_cancellation_gadgets_have_the_frozen_semantics(self) -> None:
        for n in (4, 6, 8):
            gadgets = cancellation_gadgets(n)
            identity = gadgets["ff_rn_identity"]
            nonidentity = gadgets["frf_rn_minus_1_nonidentity"]
            global_complement = gadgets["global_complement"]
            equal_count_identity = gadgets["equal_count_identity"]
            self.assertEqual(event_counts(identity), event_counts(nonidentity))
            self.assertEqual(event_counts(global_complement), event_counts(equal_count_identity))
            for state in range(1 << n):
                self.assertEqual(apply_word(state, identity, n), state)
                self.assertNotEqual(apply_word(state, nonidentity, n), state)
                self.assertNotEqual(
                    apply_word(state, gadgets["fr_order"], n),
                    apply_word(state, gadgets["rf_order"], n),
                )
                complement = apply_word(state, global_complement, n)
                self.assertEqual(complement, state ^ state_mask(n))
                self.assertEqual(canonical_code(complement, n), canonical_code(state, n))
                self.assertEqual(apply_word(state, equal_count_identity, n), state)
            self.assertTrue(cancellation_gadget_report(n)["passed"])

    def test_facade_and_validation(self) -> None:
        machine = DWEPR(4)
        state = machine.apply(machine.initial_state, "FRRF")
        self.assertEqual(machine.edges(state), edge_vector(state, 4))
        self.assertEqual(machine.code(state), canonical_code(state, 4))
        with self.assertRaises(ValueError):
            DWEPR(2)
        with self.assertRaises(ValueError):
            rotate(16, 4)
        with self.assertRaises(ValueError):
            apply_event(0, "X", 4)
        with self.assertRaises(ValueError):
            serialize_edge_vector((1, 0, 0))


class StageASymbolicGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_stage_a_symbolic_gates()

    def test_exact_exhaustive_scales_and_minimum_ledgers(self) -> None:
        self.assertTrue(self.report["passed"])
        by_n = {scale["n"]: scale for scale in self.report["scales"]}
        self.assertEqual(set(by_n), {3, 6})
        for n, scale in by_n.items():
            minimum = (n - 1) * (1 << (2 * n)) + 3 * (1 << n)
            self.assertEqual(scale["minimum_check_count"], minimum)
            self.assertEqual(scale["core_check_count"], minimum)
            self.assertGreaterEqual(scale["check_count"], minimum)
            self.assertEqual(scale["residual_classes"], 1 << (n - 1))
            self.assertEqual(scale["class_size"], 2)
            self.assertEqual(scale["maximum_shortest_witness_depth"], n - 2)

    def test_aggregate_check_ledger_is_additive(self) -> None:
        self.assertEqual(self.report["total_minimum_check_count"], 20_824)
        self.assertEqual(self.report["total_core_check_count"], 20_824)
        self.assertEqual(
            self.report["total_check_count"],
            sum(scale["check_count"] for scale in self.report["scales"]),
        )

    def test_single_scale_certificate_is_deterministic(self) -> None:
        first = exhaustive_symbolic_check(3)
        second = exhaustive_symbolic_check(3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
