from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
from tokenizers import Tokenizer

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS, compile_row as compile_s8_row, recode_operation_ids
from s9_occurrence_quotient_compiler import all_candidates, compile_row, emitted_spans_from_logits
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key
from s9_1_alpha_closed_compiler import (
    aligned_positive_logits,
    orbit_consistency_loss,
    structured_decode_graph,
    structured_spans_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "artifacts/r12/s9_occurrence_quotient_board_7563652620455132721"


def oracle_logits(candidates):
    logits = torch.full((len(candidates), len(ROLE_LABELS)), -20.0)
    logits[:, ROLE_INDEX["none"]] = 0.0
    for index, candidate in enumerate(candidates):
        if candidate.target:
            logits[index, candidate.target] = 20.0
    return logits


class S91AlphaClosedCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/tokenizer/tokenizer.json"))
        cls.row = json.loads((BOARD / "development.jsonl").read_text().splitlines()[0])
        cls.example = compile_row(cls.row, cls.tokenizer)
        cls.candidates = all_candidates(cls.example)

    def test_oracle_structured_decode_is_exact(self):
        graph, spans, _ = structured_decode_graph(
            self.example, self.candidates, oracle_logits(self.candidates)
        )
        self.assertEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))
        self.assertEqual(len(spans), len(self.row["spans"]))

    def test_structured_decode_recovers_one_low_margin_required_child(self):
        logits = oracle_logits(self.candidates)
        target_index = next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.target == ROLE_INDEX["event.operation"]
        )
        logits[target_index, ROLE_INDEX["none"]] = 30.0
        with self.assertRaises(ValueError):
            emitted_spans_from_logits(self.example, self.candidates, logits)
        graph, _, _ = structured_decode_graph(self.example, self.candidates, logits)
        self.assertEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))

    def test_uniform_logits_do_not_create_anchors(self):
        logits = torch.zeros((len(self.candidates), len(ROLE_LABELS)))
        with self.assertRaises(ValueError):
            structured_spans_from_logits(self.example, self.candidates, logits)

    def test_wrong_high_margin_child_is_not_semantically_repaired(self):
        logits = oracle_logits(self.candidates)
        gold = next(
            candidate
            for candidate in self.candidates
            if candidate.target == ROLE_INDEX["event.operation"]
        )
        wrong_index = next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.target == ROLE_INDEX["none"]
            and candidate.start > gold.end
            and candidate.start < gold.start + 12
        )
        logits[wrong_index, ROLE_INDEX["event.operation"]] = 40.0
        try:
            graph, _, _ = structured_decode_graph(self.example, self.candidates, logits)
        except ValueError:
            return
        self.assertNotEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))

    def test_orbit_alignment_and_loss(self):
        recoded = compile_row(
            recode_operation_ids(
                compile_s8_row(self.row, self.tokenizer), self.tokenizer
            ).row,
            self.tokenizer,
        )
        recoded_candidates = all_candidates(recoded)
        first = oracle_logits(self.candidates).requires_grad_()
        second = oracle_logits(recoded_candidates).requires_grad_()
        first_gold, first_targets = aligned_positive_logits(
            [self.example], [self.candidates], first
        )
        second_gold, second_targets = aligned_positive_logits(
            [recoded], [recoded_candidates], second
        )
        loss = orbit_consistency_loss(
            first_gold, second_gold, first_targets, second_targets
        )
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
