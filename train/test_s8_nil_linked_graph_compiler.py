from __future__ import annotations

import hashlib
import json
import unittest

import torch

from build_s8_nil_linked_law_graph_board import render_source
from semantic_compiler_falsifier import attach_token_targets
from s8_nil_linked_graph_compiler import (
    ROLE_INDEX,
    ROLE_LABELS,
    S8GraphExample,
    _islands,
    compile_row,
    decode_graph,
    recode_operation_ids,
    reindex_graph,
    semantic_graph_key,
)
from s8_nil_linked_law_graph import (
    EventNode,
    LawCardNode,
    NIL,
    NilLinkedLawGraph,
    linked_path,
)


class _CharEncoding:
    def __init__(self, text: str) -> None:
        self.ids = [ord(value) for value in text]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _CharTokenizer:
    def encode(self, text: str) -> _CharEncoding:
        return _CharEncoding(text)


class S8NilLinkedGraphCompilerTest(unittest.TestCase):
    @staticmethod
    def _gold_decode_fixture() -> tuple[S8GraphExample, torch.Tensor, torch.Tensor]:
        ids: list[int] = []
        roles: list[int] = []
        ranks: list[int] = []

        def add(token: int, role: str = "none", rank: int = -100) -> None:
            ids.extend((token, 1))
            roles.extend((ROLE_INDEX[role], ROLE_INDEX["none"]))
            ranks.extend((rank, -100))

        for token in range(10, 15):
            add(token, "entity.roster")
        for token in range(20, 25):
            add(token, "position.roster")
        for token in (12, 10, 14, 11, 13):
            add(token, "state.entity")
        add(30, "card.operation")
        add(20, "card.y0")
        add(21, "card.y1")
        add(31, "card.operation")
        add(22, "card.y0")
        add(24, "card.y1")
        add(40, "entry.tag")
        # Storage order is node B, node C, node A; linked order is A, B, C.
        add(41, "event.tag", 1)
        add(31, "event.operation")
        add(11, "event.entity")
        add(42, "event.next")
        add(42, "event.tag", 2)
        add(30, "event.operation")
        add(12, "event.entity")
        add(2, "event.nil")
        add(40, "event.tag", 0)
        add(30, "event.operation")
        add(10, "event.entity")
        add(41, "event.next")
        add(22, "query.position")
        row = {
            "modulus": 5,
            "initial_state": [2, 0, 4, 1, 3],
            "cards": [
                {"operation": "op-a", "y0": 0, "y1": 1},
                {"operation": "op-b", "y0": 2, "y1": 4},
            ],
            "nodes": [
                {"tag": "B", "operation": "op-b", "identity": 1, "next_tag": "C"},
                {"tag": "C", "operation": "op-a", "identity": 2, "next_tag": None},
                {"tag": "A", "operation": "op-a", "identity": 0, "next_tag": "B"},
            ],
            "entry_node": 2,
            "query_position": 2,
        }
        example = S8GraphExample(
            ids=tuple(ids),
            roles=tuple(roles),
            ranks=tuple(ranks),
            role_positions={},
            operation_positions={},
            row=row,
        )
        role_logits = torch.full((len(ids), len(ROLE_LABELS)), -20.0)
        rank_logits = torch.full((len(ids), 8), -20.0)
        for index, role in enumerate(roles):
            role_logits[index, role] = 20.0
            if ranks[index] >= 0:
                rank_logits[index, ranks[index]] = 20.0
            else:
                rank_logits[index, 0] = 20.0
        return example, role_logits, rank_logits

    def test_islands_find_complete_runs(self) -> None:
        labels = [
            ROLE_INDEX["none"],
            ROLE_INDEX["event.tag"],
            ROLE_INDEX["event.tag"],
            ROLE_INDEX["none"],
            ROLE_INDEX["event.tag"],
        ]
        self.assertEqual(_islands(labels, "event.tag"), ((1, 2), (4,)))

    def test_reindex_preserves_path_semantics(self) -> None:
        graph = NilLinkedLawGraph(
            modulus=5,
            initial_state=(0, 1, 2, 3, 4),
            cards=(LawCardNode("op", 0, 1),),
            nodes=(EventNode(0, "op", 1), EventNode(1, "op", NIL)),
            entry_node=0,
            query_position=0,
        )
        renamed = reindex_graph(graph, (1, 0))
        self.assertEqual(linked_path(renamed), (1, 0))
        self.assertEqual(renamed.nodes[1].identity, 0)

    def test_gold_logits_decode_permuted_nil_linked_graph(self) -> None:
        example, role_logits, rank_logits = self._gold_decode_fixture()
        decoded = decode_graph(example, role_logits, rank_logits)
        self.assertEqual(decoded["treatment_path"], (2, 0, 1))
        self.assertEqual(decoded["ordinary_path"], (2, 0, 1))
        self.assertEqual(
            semantic_graph_key(decoded["graph"]),
            (
                5,
                (2, 0, 4, 1, 3),
                ((0, 1), (2, 4)),
                ((1, (2, 4), 1), (2, (0, 1), -1), (0, (0, 1), 0)),
                2,
                2,
            ),
        )

    def test_missing_entry_only_invalidates_linked_treatment(self) -> None:
        example, role_logits, rank_logits = self._gold_decode_fixture()
        entry = example.roles.index(ROLE_INDEX["entry.tag"])
        role_logits[entry].fill_(-20.0)
        role_logits[entry, ROLE_INDEX["none"]] = 20.0
        decoded = decode_graph(example, role_logits, rank_logits)
        self.assertIsNone(decoded["treatment_path"])
        self.assertEqual(decoded["ordinary_path"], (2, 0, 1))

    def test_source_nonce_recoding_handles_unequal_contextual_widths(self) -> None:
        tokenizer = _CharTokenizer()
        question, spans = render_source(
            entities=("e0", "e1", "e2", "e3", "e4"),
            positions=("p0", "p1", "p2", "p3", "p4"),
            initial_state=(2, 0, 4, 1, 3),
            cards=(("x", 0, 1), ("long", 2, 4)),
            nodes=(
                ("b", "long", 1, "c"),
                ("c", "x", 2, None),
                ("a", "x", 0, "b"),
            ),
            entry_tag="a",
            query_position=2,
            renderer="registry",
            card_order=(0, 1),
        )
        encoding, token_spans = attach_token_targets(
            question, spans, tokenizer
        )
        row = {
            "question": question,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(
                    encoding.ids, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "spans": token_spans,
            "execution_tags": ["a", "b", "c"],
            "cards": [
                {"operation": "x", "y0": 0, "y1": 1},
                {"operation": "long", "y0": 2, "y1": 4},
            ],
            "nodes": [
                {"tag": "b", "operation": "long", "identity": 1, "next_tag": "c"},
                {"tag": "c", "operation": "x", "identity": 2, "next_tag": None},
                {"tag": "a", "operation": "x", "identity": 0, "next_tag": "b"},
            ],
        }
        example = compile_row(row, tokenizer)
        recoded = recode_operation_ids(example, tokenizer)
        self.assertNotEqual(len(recoded.ids), len(example.ids))
        self.assertEqual(
            [card["operation"] for card in recoded.row["cards"]],
            ["long", "x"],
        )
        self.assertEqual(
            [node["operation"] for node in recoded.row["nodes"]],
            ["x", "long", "long"],
        )
        for label, span in recoded.row["spans"].items():
            if label.endswith("operation"):
                self.assertIn(span["text"], {"x", "long"})


if __name__ == "__main__":
    unittest.main()
