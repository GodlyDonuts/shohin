from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
from tokenizers import Tokenizer

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS
from s9_occurrence_quotient_compiler import (
    compile_row,
    decode_graph,
    emitted_spans_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "artifacts/r12/s8_1_nil_linked_graph_board_5943437777437228096"


class S9CompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/tokenizer/tokenizer.json"))
        cls.row = json.loads((BOARD / "development.jsonl").read_text().splitlines()[0])
        cls.example = compile_row(cls.row, cls.tokenizer)

    def test_proposals_cover_all_gold_spans(self):
        positive = [value for value in self.example.candidates if value.target]
        self.assertEqual(len(positive), len(self.row["spans"]))
        self.assertLessEqual(max(value.end - value.start + 1 for value in positive), 4)

    def test_oracle_candidate_logits_reconstruct_exact_graph(self):
        logits = torch.full((len(self.example.candidates), len(ROLE_LABELS)), -20.0)
        for index, candidate in enumerate(self.example.candidates):
            logits[index, candidate.target] = 20.0
        graph, spans = decode_graph(self.example, self.example.candidates, logits)
        self.assertEqual(graph.modulus, int(self.row["modulus"]))
        self.assertEqual(graph.initial_state, tuple(self.row["initial_state"]))
        self.assertEqual(graph.entry_node, int(self.row["entry_node"]))
        self.assertEqual(len(graph.nodes), int(self.row["depth"]))
        self.assertEqual(len(spans), len(self.row["spans"]))

    def test_none_logits_emit_no_graph(self):
        logits = torch.zeros((len(self.example.candidates), len(ROLE_INDEX)))
        with self.assertRaises(ValueError):
            emitted_spans_from_logits(self.example, self.example.candidates, logits)


if __name__ == "__main__":
    unittest.main()
