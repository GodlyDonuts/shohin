import unittest

from s6_contextual_affine_law import (
    ADMITTED_MODULI,
    AffineLaw,
    apply_law,
    enumerate_laws,
    execute_program,
    infer_affine_law,
    one_witness_candidates,
    raw_split_laws,
    repaired_split_laws,
    split_laws,
    treatment_input,
)
from s6_contextual_affine_law_falsifier import build_report


class ContextualAffineLawTests(unittest.TestCase):
    def test_two_witnesses_identify_every_law(self):
        for modulus in ADMITTED_MODULI:
            for law in enumerate_laws(modulus):
                self.assertEqual(infer_affine_law(modulus, *law.card), law)

    def test_one_witness_leaves_every_slope_ambiguous(self):
        for modulus in ADMITTED_MODULI:
            for card_y0 in range(modulus):
                candidates = one_witness_candidates(modulus, card_y0)
                self.assertEqual(len(candidates), modulus - 1)
                self.assertEqual({law.intercept for law in candidates}, {card_y0})
                self.assertEqual(
                    {law.slope for law in candidates}, set(range(1, modulus))
                )

    def test_state_update_is_closed_and_places_target(self):
        for modulus in ADMITTED_MODULI:
            initial = tuple(range(modulus))
            for law in enumerate_laws(modulus):
                for identity in range(modulus):
                    updated = apply_law(initial, identity, law)
                    self.assertEqual(set(updated), set(initial))
                    self.assertEqual(updated.index(identity), law.destination(identity))

    def test_order_can_change_a_late_query(self):
        modulus = 5
        initial = tuple(range(modulus))
        left = AffineLaw(modulus, 1, 1)
        reflect = AffineLaw(modulus, modulus - 1, 2)
        forward = execute_program(initial, ((0, left), (1, reflect)))
        reverse = execute_program(initial, ((1, reflect), (0, left)))
        self.assertNotEqual(forward, reverse)
        self.assertTrue(any(a != b for a, b in zip(forward, reverse)))

    def test_splits_are_disjoint_and_nonempty(self):
        for modulus in ADMITTED_MODULI:
            split = split_laws(modulus)
            keys = {name: {law.key for law in laws} for name, laws in split.items()}
            self.assertTrue(all(keys.values()))
            self.assertFalse(keys["train"] & keys["development"])
            self.assertFalse(keys["train"] & keys["confirmation"])
            self.assertFalse(keys["development"] & keys["confirmation"])

    def test_repair_records_only_missing_coordinate_promotions(self):
        raw = raw_split_laws(5)
        self.assertNotIn(1, {law.card[1] for law in raw["train"]})
        repaired, promotions = repaired_split_laws(5)
        self.assertIn(1, {law.card[1] for law in repaired["train"]})
        self.assertEqual(
            promotions,
            ({
                "modulus": 5,
                "field": "card_y1",
                "value": 1,
                "law": "m5_a4_b2",
                "source": "confirmation",
                "destination": "train",
            },),
        )

    def test_treatment_input_exposes_only_card_and_query(self):
        fields = treatment_input(AffineLaw(5, 2, 3), 4)
        self.assertEqual(
            set(fields), {"modulus", "card_y0", "card_y1", "current_location"}
        )

    def test_complete_falsifier_passes(self):
        report = build_report()
        self.assertEqual(report["decision"], "pass_s6_cpu_mechanics")
        self.assertTrue(all(report["gates"].values()))


if __name__ == "__main__":
    unittest.main()
