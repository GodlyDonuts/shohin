#!/usr/bin/env python3

import random
import unittest

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_development
from build_s4_set_identity_fresh_development import audit_fresh
from semantic_compiler_falsifier import candidate_names


class FreshS4SetIdentityDevelopmentTest(unittest.TestCase):
    def test_small_fresh_board_has_unique_roster_multisets(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        factors = list(factor_catalogue("known"))
        random.Random(7).shuffle(factors)
        rows = build_development(24, 13, tokenizer, names, factors[:48])
        public = {
            "questions": set(), "grams": set(), "names": set(), "factors": set(),
        }
        report = audit_fresh(
            rows,
            public,
            "train-sha",
            "report-sha",
            "artifacts/shohin-tok-32k.json",
            set(),
        )
        excluded_production_size_gates = {
            name: value for name, value in report["gates"].items()
            if name not in {
                "exactly_2048_rows", "exactly_512_matched_groups",
                "each_depth_at_least_300_rows",
            }
        }
        self.assertTrue(
            all(excluded_production_size_gates.values()), excluded_production_size_gates,
        )


if __name__ == "__main__":
    unittest.main()
