#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import build_referential_literal_pointer_factorized_corpus as corpus  # noqa: E402
import build_referential_literal_pointer_qualification as qualification  # noqa: E402
import semantic_compiler_falsifier as falsifier  # noqa: E402


class ReferentialLiteralPointerQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = ROOT / "artifacts" / "shohin-tok-32k.json"
        cls.tokenizer = Tokenizer.from_file(str(cls.tokenizer_path))
        names, _ = falsifier.candidate_names(cls.tokenizer, 500)
        cls.reference = (
            corpus.build_split("train", 48, 2201, cls.tokenizer, names[:220])
            + corpus.build_split(
                "development_compositional", 16, 2202, cls.tokenizer, names[220:300],
            )
            + corpus.build_split(
                "development_lexical_ood", 16, 2203, cls.tokenizer, names[300:380],
            )
        )
        excluded = {row["factor_signature"] for row in cls.reference}
        cls.rows = qualification.build_qualification(
            64, 9901, cls.tokenizer, names[380:500], excluded,
        )
        cls.report = qualification.audit_qualification(
            cls.rows, cls.reference, cls.tokenizer_path,
            ROOT / "pipeline" / "build_referential_literal_pointer_qualification.py",
        )

    def test_shape_and_schema(self):
        self.assertEqual(len(self.rows), 256)
        self.assertTrue(all(row["id"].startswith("RLPCQ-") for row in self.rows))
        self.assertTrue(all(row["split"] == "development_compositional" for row in self.rows))

    def test_public_overlaps_are_zero(self):
        self.assertEqual(self.report["public_overlap"], {
            "exact_prompts": 0,
            "word_13grams": 0,
            "entity_names": 0,
            "factor_combinations": 0,
        })

    def test_deterministic_gates_pass_and_shortcuts_remain_near_chance(self):
        deterministic = {
            name: passed
            for name, passed in self.report["structural_gates"].items()
            if name != "all_shortcut_ceilings_at_chance_plus_one_example"
        }
        self.assertTrue(all(deterministic.values()), deterministic)
        self.assertLessEqual(
            self.report["shortcut_ceilings"]["absolute_pointer_positions"]["accuracy"],
            0.36,
        )
        self.assertEqual(self.report["confirmation_access"], 0)


if __name__ == "__main__":
    unittest.main()
