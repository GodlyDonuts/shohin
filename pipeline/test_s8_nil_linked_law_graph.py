from __future__ import annotations

import unittest

from s6_contextual_affine_law import AffineLaw
from s7_learned_cayley_law import SymbolBinding
from s8_nil_linked_law_graph import (
    NIL,
    EventNode,
    execute_graph,
    graph_from_ordered_events,
    linked_path,
    rewire_path,
)


class NilLinkedLawGraphTest(unittest.TestCase):
    def graph(self):
        binding = SymbolBinding(5, (2, 4, 1, 0, 3))
        graph = graph_from_ordered_events(
            modulus=5,
            initial_state=(3, 1, 4, 0, 2),
            cards={"amber": binding.card(AffineLaw(5, 2, 1))},
            events=((4, "amber"), (1, "amber"), (3, "amber")),
            storage_ids=(2, 0, 1),
            query_position=2,
        )
        return binding, graph

    def test_link_path_ignores_storage_order(self) -> None:
        binding, graph = self.graph()
        self.assertEqual(linked_path(graph), (2, 0, 1))
        linked = execute_graph(graph, binding.successor, binding.zero_symbol)
        storage = execute_graph(
            graph, binding.successor, binding.zero_symbol, storage_order=True
        )
        self.assertNotEqual(linked[0], storage[0])

    def test_rewire_is_valid_and_changes_order(self) -> None:
        binding, graph = self.graph()
        rewired = rewire_path(graph, tuple(reversed(linked_path(graph))))
        self.assertEqual(linked_path(rewired), (1, 0, 2))
        self.assertNotEqual(
            execute_graph(graph, binding.successor, binding.zero_symbol)[0],
            execute_graph(rewired, binding.successor, binding.zero_symbol)[0],
        )

    def test_cycle_is_rejected(self) -> None:
        _, graph = self.graph()
        nodes = list(graph.nodes)
        nodes[1] = EventNode(nodes[1].identity, nodes[1].operation, 2)
        broken = type(graph)(
            graph.modulus,
            graph.initial_state,
            graph.cards,
            tuple(nodes),
            graph.entry_node,
            graph.query_position,
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            linked_path(broken)

    def test_stranded_node_is_rejected(self) -> None:
        _, graph = self.graph()
        nodes = list(graph.nodes)
        nodes[0] = EventNode(nodes[0].identity, nodes[0].operation, NIL)
        broken = type(graph)(
            graph.modulus,
            graph.initial_state,
            graph.cards,
            tuple(nodes),
            graph.entry_node,
            graph.query_position,
        )
        with self.assertRaisesRegex(ValueError, "omits or strands"):
            linked_path(broken)


if __name__ == "__main__":
    unittest.main()
