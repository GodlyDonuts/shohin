from __future__ import annotations

import random
import unittest

from s9_occurrence_quotient import (
    compile_quotient,
    corrupt_first_relation_kind,
    merge_first_two_entities,
    permute_relation_storage,
    quotient_from_emitted_spans,
    reindex_classes,
    split_first_event_operation,
    unique_every_occurrence,
)


def _source():
    parts = []
    spans = {}

    def add(label, text):
        start = sum(len(value) for value in parts)
        parts.append(text)
        spans[label] = {"start": start, "end": start + len(text), "text": text}
        parts.append(" ")

    for index, value in enumerate(("e0", "e1", "e2", "e3", "e4")):
        add(f"entity.roster.{index}", value)
    for index, value in enumerate(("p0", "p1", "p2", "p3", "p4")):
        add(f"position.roster.{index}", value)
    for index, value in enumerate(("e2", "e0", "e4", "e1", "e3")):
        add(f"state.entity.{index}", value)
    add("card.0.operation", "op0")
    add("card.0.y0", "p0")
    add("card.0.y1", "p1")
    add("card.1.operation", "op1")
    add("card.1.y0", "p1")
    add("card.1.y1", "p2")
    add("entry.tag", "t0")
    add("event.0.tag", "t1")
    add("event.0.operation", "op1")
    add("event.0.entity", "e0")
    add("event.0.nil", "nil")
    add("event.1.tag", "t0")
    add("event.1.operation", "op0")
    add("event.1.entity", "e2")
    add("event.1.next", "t1")
    add("query.position", "p3")
    return "".join(parts), spans


class OccurrenceQuotientTest(unittest.TestCase):
    def setUp(self):
        question, spans = _source()
        self.quotient = quotient_from_emitted_spans(question, spans)

    def test_compiles_linked_graph(self):
        graph = compile_quotient(self.quotient)
        self.assertEqual(graph.initial_state, (2, 0, 4, 1, 3))
        self.assertEqual(graph.entry_node, 1)
        self.assertEqual(graph.nodes[1].next_node, 0)
        self.assertEqual(graph.nodes[0].next_node, -1)

    def test_class_and_relation_storage_are_invariant(self):
        expected = compile_quotient(self.quotient)
        classes = list(range(len(self.quotient.classes)))
        relations = list(range(len(self.quotient.relations)))
        random.Random(7).shuffle(classes)
        random.Random(11).shuffle(relations)
        self.assertEqual(compile_quotient(reindex_classes(self.quotient, classes)), expected)
        self.assertEqual(
            compile_quotient(permute_relation_storage(self.quotient, relations)), expected
        )

    def test_identity_and_relation_corruptions_reject(self):
        for value in (
            split_first_event_operation(self.quotient),
            merge_first_two_entities(self.quotient),
            unique_every_occurrence(self.quotient),
            corrupt_first_relation_kind(self.quotient),
        ):
            with self.assertRaises(ValueError):
                compile_quotient(value)


if __name__ == "__main__":
    unittest.main()
