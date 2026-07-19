#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "pipeline" / "semantic_compiler_falsifier.py"
SPEC = importlib.util.spec_from_file_location("semantic_compiler_falsifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SemanticCompilerFalsifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts" / "shohin-tok-32k.json"))
        cls.report = MODULE.build_board(cls.tokenizer)

    def test_all_frozen_gates_pass(self):
        self.assertTrue(self.report["all_gates_pass"], self.report["gates"])

    def test_exact_board_shape(self):
        self.assertEqual(self.report["quartets"], 32)
        self.assertEqual(self.report["surfaces"], 128)
        self.assertEqual(len(self.report["rows"]), 128)

    def test_matched_twins_have_equal_token_bags(self):
        by_quartet = {}
        for row in self.report["rows"]:
            by_quartet.setdefault(row["quartet"], {})[row["surface_type"]] = row
        for group in by_quartet.values():
            self.assertEqual(group["canonical"]["token_bag"], group["order_twin"]["token_bag"])
            self.assertEqual(group["canonical"]["token_bag"], group["binding_twin"]["token_bag"])

    def test_independent_executors_agree(self):
        for row in self.report["rows"]:
            program = tuple(
                MODULE.Operation(op["kind"], op["entity"], op["amount"])
                for op in row["program"]
            )
            initial = tuple(row["initial_order"])
            self.assertEqual(
                MODULE.apply_program_pop_insert(initial, program),
                MODULE.apply_program_adjacent_swaps(initial, program),
            )


if __name__ == "__main__":
    unittest.main()
