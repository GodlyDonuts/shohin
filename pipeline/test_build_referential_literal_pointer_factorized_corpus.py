#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import build_referential_literal_pointer_factorized_corpus as corpus  # noqa: E402
import semantic_compiler_falsifier as falsifier  # noqa: E402


class FactorizedReferentialLiteralPointerCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts" / "shohin-tok-32k.json"))
        names, _ = falsifier.candidate_names(cls.tokenizer, 360)
        cls.split_rows = {
            "train": corpus.build_split("train", 48, 1101, cls.tokenizer, names[:180]),
            "development_compositional": corpus.build_split(
                "development_compositional", 16, 1102, cls.tokenizer, names[180:240],
            ),
            "development_lexical_ood": corpus.build_split(
                "development_lexical_ood", 16, 1103, cls.tokenizer, names[240:300],
            ),
            "confirmation": corpus.build_split(
                "confirmation", 16, 1104, cls.tokenizer, names[300:360],
            ),
        }
        cls.report = corpus.audit_splits(
            cls.split_rows,
            ROOT / "artifacts" / "shohin-tok-32k.json",
            ROOT / "pipeline" / "build_referential_literal_pointer_factorized_corpus.py",
        )

    def test_split_shapes(self):
        self.assertEqual(len(self.split_rows["train"]), 192)
        self.assertEqual(len(self.split_rows["development_compositional"]), 64)
        self.assertEqual(len(self.split_rows["development_lexical_ood"]), 64)
        self.assertEqual(len(self.split_rows["confirmation"]), 64)

    def test_every_target_span_is_nonempty_and_executors_agree(self):
        for rows in self.split_rows.values():
            for row in rows:
                self.assertTrue(set(corpus.REQUIRED_SPANS).issubset(row["spans"]))
                self.assertTrue(all(
                    row["spans"][label]["token_positions"]
                    for label in corpus.REQUIRED_SPANS
                ))
                self.assertTrue(row["executor_agreement"])

    def test_quartet_invariants(self):
        for rows in self.split_rows.values():
            groups = {}
            for row in rows:
                groups.setdefault(row["group"], {})[row["surface_type"]] = row
            for group in groups.values():
                canonical = group["canonical"]
                paraphrase = group["paraphrase"]
                order_twin = group["order_twin"]
                binding_twin = group["binding_twin"]
                self.assertEqual(corpus.row_label(canonical), corpus.row_label(paraphrase))
                self.assertEqual(canonical["terminal_order"], paraphrase["terminal_order"])
                self.assertNotEqual(canonical["answer"], order_twin["answer"])
                self.assertNotEqual(canonical["answer"], binding_twin["answer"])
                self.assertEqual(canonical["token_bag"], order_twin["token_bag"])
                self.assertEqual(canonical["token_bag"], binding_twin["token_bag"])

    def test_compositional_atoms_are_seen_but_combinations_are_not(self):
        train_vocab = corpus.split_factor_vocab(self.split_rows["train"])
        comp_vocab = corpus.split_factor_vocab(
            self.split_rows["development_compositional"],
        )
        for field in corpus.FACTOR_FIELDS:
            self.assertTrue(set(comp_vocab[field]).issubset(train_vocab[field]))
        train_signatures = {row["factor_signature"] for row in self.split_rows["train"]}
        comp_signatures = {
            row["factor_signature"]
            for row in self.split_rows["development_compositional"]
        }
        self.assertFalse(train_signatures & comp_signatures)

    def test_lexical_ood_is_real(self):
        known = {
            corpus.direction_pairs(row["factors"]["lexicon"])[
                row["factors"]["direction_pair"]
            ]
            for split in ("train", "development_compositional", "confirmation")
            for row in self.split_rows[split]
        }
        lexical = {
            corpus.direction_pairs(row["factors"]["lexicon"])[
                row["factors"]["direction_pair"]
            ]
            for row in self.split_rows["development_lexical_ood"]
        }
        self.assertFalse(known & lexical)

    def test_names_are_split_disjoint(self):
        pools = {
            split: {
                name
                for row in rows
                for name in (*row["initial_order"], row["neutral_anchor"])
            }
            for split, rows in self.split_rows.items()
        }
        for left, right in __import__("itertools").combinations(pools, 2):
            self.assertFalse(pools[left] & pools[right])

    def test_audit_gates_pass(self):
        self.assertTrue(self.report["all_gates_pass"], self.report["structural_gates"])

    def test_artifact_paths_are_distinct(self):
        paths = corpus.artifact_paths(Path("/tmp/rlpc-factorized"))
        self.assertEqual(len(set(paths.values())), 4)
        self.assertEqual(paths["development_compositional"].name,
                         "development_compositional.jsonl")
        self.assertEqual(paths["development_lexical_ood"].name,
                         "development_lexical_ood.jsonl")


if __name__ == "__main__":
    unittest.main()
