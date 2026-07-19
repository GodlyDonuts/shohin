from __future__ import annotations

import unittest

from s9_occurrence_quotient_falsifier import semantic_key
from s8_nil_linked_law_graph import EventNode, LawCardNode, NilLinkedLawGraph


class OccurrenceQuotientFalsifierTest(unittest.TestCase):
    def test_semantic_key_ignores_card_storage_order_only(self):
        first = NilLinkedLawGraph(
            modulus=5,
            initial_state=(0, 1, 2, 3, 4),
            cards=(LawCardNode("a", 0, 1), LawCardNode("b", 1, 2)),
            nodes=(EventNode(0, "a", -1),),
            entry_node=0,
            query_position=0,
        )
        second = replace_cards(first, tuple(reversed(first.cards)))
        self.assertEqual(semantic_key(first), semantic_key(second))


def replace_cards(graph, cards):
    return NilLinkedLawGraph(
        modulus=graph.modulus,
        initial_state=graph.initial_state,
        cards=cards,
        nodes=graph.nodes,
        entry_node=graph.entry_node,
        query_position=graph.query_position,
    )


if __name__ == "__main__":
    unittest.main()
