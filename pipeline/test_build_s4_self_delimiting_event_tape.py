#!/usr/bin/env python3

import random
import unittest

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import (
    DEVELOPMENT_SPLIT,
    build_development,
    build_train,
    gold_events,
)
from semantic_compiler_falsifier import candidate_names


class SelfDelimitingEventTapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(cls.tokenizer, 100)
        cls.names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        cls.factors = list(factor_catalogue("known"))
        random.Random(7).shuffle(cls.factors)

    def test_train_rows_are_unpadded_and_self_counting(self):
        rows = build_train(8, 11, self.tokenizer, self.names, self.factors[:8])
        self.assertEqual({row["depth"] for row in rows}, {1, 2, 3, 4})
        for row in rows:
            self.assertNotIn("active_operations", row)
            self.assertEqual(len(gold_events(row)), row["depth"])
            self.assertLess(row["token_count"], 2048)

    def test_development_has_matched_twins_through_depth_eight(self):
        rows = build_development(
            6, 13, self.tokenizer, self.names, self.factors[:12],
        )
        self.assertEqual({row["depth"] for row in rows}, set(range(3, 9)))
        self.assertTrue(all(row["split"] == DEVELOPMENT_SPLIT for row in rows))
        for offset in range(0, len(rows), 4):
            canonical, paraphrase, order, binding = rows[offset:offset + 4]
            self.assertEqual(canonical["program"], paraphrase["program"])
            self.assertEqual(canonical["token_bag"], order["token_bag"])
            self.assertEqual(canonical["token_bag"], binding["token_bag"])
            self.assertNotEqual(canonical["answer"], order["answer"])
            self.assertNotEqual(canonical["answer"], binding["answer"])


if __name__ == "__main__":
    unittest.main()
