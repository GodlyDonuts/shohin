#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import build_referential_literal_pointer_corpus as corpus  # noqa: E402
import semantic_compiler_falsifier as falsifier  # noqa: E402


class ReferentialLiteralPointerCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts" / "shohin-tok-32k.json"))
        names, _ = falsifier.candidate_names(cls.tokenizer, 180)
        cls.split_rows = {
            "train": corpus.build_split("train", 32, 101, cls.tokenizer, names[:120]),
            "development": corpus.build_split(
                "development", 16, 102, cls.tokenizer, names[120:180],
            ),
        }

    def test_split_shapes(self):
        self.assertEqual(len(self.split_rows["train"]), 128)
        self.assertEqual(len(self.split_rows["development"]), 64)

    def test_matched_surfaces_are_token_bag_identical(self):
        for rows in self.split_rows.values():
            groups = {}
            for row in rows:
                groups.setdefault(row["group"], {})[row["surface_type"]] = row
            for group in groups.values():
                self.assertEqual(group["canonical"]["token_bag"], group["order_twin"]["token_bag"])
                self.assertEqual(group["canonical"]["token_bag"], group["binding_twin"]["token_bag"])

    def test_names_and_renderers_are_split_disjoint(self):
        train_names = {name for row in self.split_rows["train"] for name in row["initial_order"]}
        development_names = {
            name for row in self.split_rows["development"] for name in row["initial_order"]
        }
        self.assertFalse(train_names & development_names)
        train_renderers = {row["renderer"] for row in self.split_rows["train"]}
        development_renderers = {row["renderer"] for row in self.split_rows["development"]}
        self.assertFalse(train_renderers & development_renderers)

    def test_every_target_span_is_nonempty(self):
        required = {
            "intro.entity0", "intro.entity1", "intro.entity2",
            "op0.kind", "op0.entity", "op0.literal",
            "op1.kind", "op1.entity", "op1.literal", "query.position",
        }
        for rows in self.split_rows.values():
            for row in rows:
                self.assertTrue(required.issubset(row["spans"]))
                self.assertTrue(all(row["spans"][label]["token_positions"] for label in required))

    def test_split_artifact_paths_are_distinct(self):
        paths = corpus.artifact_paths(Path("/tmp/rlpc"))
        self.assertEqual(len(set(paths.values())), 3)
        self.assertEqual(paths["train"].name, "train.jsonl")
        self.assertEqual(paths["development"].name, "development.jsonl")
        self.assertEqual(paths["confirmation"].name, "confirmation.jsonl")


if __name__ == "__main__":
    unittest.main()
