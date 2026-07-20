from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer

import s9_2_global_anchor_compiler as s92
from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS
from s9_occurrence_quotient_compiler import (
    S9Example,
    SpanCandidate,
    all_candidates,
    compile_row,
)
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key
from s9_1_alpha_closed_compiler import structured_decode_graph


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "artifacts/r12/s9_1_alpha_closed_board_1370124171784245712"


def oracle_logits(candidates):
    logits = torch.full((len(candidates), len(ROLE_LABELS)), -20.0)
    logits[:, ROLE_INDEX["none"]] = 0.0
    for index, candidate in enumerate(candidates):
        if candidate.target:
            logits[index, candidate.target] = 20.0
    return logits


def synthetic_span(index: int, target: int = 0, text: str | None = None):
    return SpanCandidate(
        start=2 * index,
        end=2 * index,
        text=text or f"v{index}",
        char_start=3 * index,
        char_end=3 * index + 2,
        target=target,
    )


class S92GlobalAnchorCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(
            str(ROOT / "artifacts/tokenizer/tokenizer.json")
        )
        cls.row = json.loads((BOARD / "train.jsonl").read_text().splitlines()[0])
        cls.example = compile_row(cls.row, cls.tokenizer)
        cls.candidates = all_candidates(cls.example)

    def test_oracle_global_decode_is_exact_and_reports_counts(self):
        graph, spans, _, result = s92.global_structured_decode_graph(
            self.example,
            self.candidates,
            oracle_logits(self.candidates),
        )
        self.assertEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))
        self.assertEqual(len(spans), len(self.row["spans"]))
        self.assertEqual(result.modulus, self.row["modulus"])
        self.assertEqual(result.card_count, len(self.row["cards"]))
        self.assertEqual(result.event_count, self.row["depth"])
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.selected_counts["entry.tag"], 1)
        self.assertEqual(result.selected_counts["query.position"], 1)

    def test_global_search_recovers_one_required_low_margin_root(self):
        logits = oracle_logits(self.candidates)
        target_index = next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.target == ROLE_INDEX["entity.roster"]
        )
        logits[target_index, ROLE_INDEX["none"]] = 30.0
        with self.assertRaises(ValueError):
            structured_decode_graph(self.example, self.candidates, logits)
        graph, _, _, _ = s92.global_structured_decode_graph(
            self.example,
            self.candidates,
            logits,
        )
        self.assertEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))

    def test_global_order_and_cardinality_ignore_late_extra_anchor(self):
        logits = oracle_logits(self.candidates)
        query_end = max(
            int(value)
            for value in self.row["spans"]["query.position"]["token_positions"]
        )
        extra_index = next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.target == ROLE_INDEX["none"] and candidate.start > query_end
        )
        logits[extra_index, ROLE_INDEX["entity.roster"]] = 40.0
        with self.assertRaises(ValueError):
            structured_decode_graph(self.example, self.candidates, logits)
        graph, _, _, _ = s92.global_structured_decode_graph(
            self.example,
            self.candidates,
            logits,
        )
        self.assertEqual(semantic_key(graph), semantic_key(expected_graph(self.row)))

    def test_uniform_logits_abstain(self):
        logits = torch.zeros((len(self.candidates), len(ROLE_LABELS)))
        with self.assertRaisesRegex(ValueError, "did not beat none"):
            s92.global_anchor_assignment(self.candidates, logits)

    def test_optimizer_uses_neither_targets_metadata_nor_compiler(self):
        logits = oracle_logits(self.candidates)
        expected = s92.global_anchor_assignment(self.candidates, logits)
        target_free = tuple(replace(candidate, target=0) for candidate in self.candidates)
        with mock.patch.object(
            s92,
            "compile_quotient",
            side_effect=AssertionError("compiler entered optimization"),
        ):
            observed = s92.global_anchor_assignment(target_free, logits)
        self.assertEqual(observed, expected)

    def test_viterbi_ties_are_stable_and_choose_first_candidate(self):
        roles = (
            ("entity.roster",) * 5
            + ("position.roster",) * 5
            + ("state.entity",) * 5
            + ("card.operation",) * 2
            + ("entry.tag", "event.tag", "query.position")
        )
        candidates = [synthetic_span(index) for index in range(len(roles))]
        candidates.insert(
            1,
            SpanCandidate(
                start=0,
                end=0,
                text="alternate",
                char_start=0,
                char_end=2,
                target=0,
            ),
        )
        logits = torch.full((len(candidates), len(ROLE_LABELS)), -20.0)
        logits[:, ROLE_INDEX["none"]] = 0.0
        for index, role in enumerate(roles):
            candidate_index = index if index == 0 else index + 1
            logits[candidate_index, ROLE_INDEX[role]] = 5.0
        logits[1, ROLE_INDEX[roles[0]]] = 5.0

        first = s92.global_anchor_assignment(candidates, logits)
        second = s92.global_anchor_assignment(candidates, logits.clone())
        self.assertEqual(first, second)
        self.assertEqual(first.candidate_indices[0], 0)
        self.assertEqual((first.modulus, first.card_count, first.event_count), (5, 2, 1))
        self.assertAlmostEqual(first.score, 5.0 * len(roles))


class S92AlphaOrbitLossTest(unittest.TestCase):
    def setUp(self):
        self.candidates = tuple(
            [
                synthetic_span(0, ROLE_INDEX["entity.roster"]),
                synthetic_span(1, ROLE_INDEX["card.operation"]),
            ]
            + [synthetic_span(index) for index in range(2, 14)]
        )
        self.example = S9Example(
            ids=tuple(range(14)),
            offsets=tuple((index, index + 1) for index in range(14)),
            gold=(
                (0, 0, ROLE_INDEX["entity.roster"]),
                (2, 2, ROLE_INDEX["card.operation"]),
            ),
            row={},
        )
        logits = torch.zeros((len(self.candidates), len(ROLE_LABELS)))
        logits[0, ROLE_INDEX["entity.roster"]] = 4.0
        logits[1, ROLE_INDEX["card.operation"]] = 4.0
        for index in range(2, len(self.candidates)):
            logits[index, 1:] = torch.linspace(-2.0, 1.0, len(ROLE_LABELS) - 1)
            logits[index, 1:] += index / 100.0
        self.logits = logits

    def test_hard_negative_profiles_are_coordinate_and_order_free(self):
        permutation = [0, 1] + list(reversed(range(2, len(self.candidates))))
        recoded_candidates = tuple(self.candidates[index] for index in permutation)
        recoded_logits = self.logits[permutation]
        loss = s92.hard_negative_orbit_loss(
            [self.candidates],
            [recoded_candidates],
            self.logits,
            recoded_logits,
            top_k=4,
        )
        self.assertEqual(float(loss.item()), 0.0)

    def test_combined_orbit_loss_detects_hard_negative_and_has_finite_gradients(self):
        original = self.logits.clone().requires_grad_()
        recoded = self.logits.clone()
        recoded[2, ROLE_INDEX["entity.roster"]] = 12.0
        recoded = recoded.requires_grad_()
        loss = s92.alpha_orbit_consistency_loss(
            [self.example],
            [self.example],
            [self.candidates],
            [self.candidates],
            original,
            recoded,
            top_k=4,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertTrue(torch.isfinite(recoded.grad).all())
        self.assertGreater(float(original.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(recoded.grad.abs().sum().item()), 0.0)

    def test_combined_orbit_loss_is_zero_for_identical_views(self):
        loss = s92.alpha_orbit_consistency_loss(
            [self.example],
            [self.example],
            [self.candidates],
            [self.candidates],
            self.logits,
            self.logits.clone(),
            top_k=4,
        )
        self.assertEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
