#!/usr/bin/env python3

import unittest

from build_s4_structural_lexicon import build


class S4StructuralLexiconTest(unittest.TestCase):
    def test_training_patterns_are_collected_without_collision(self):
        row = {
            "split": "s4_event_tape_train",
            "spans": {
                "intro.entity0": {"token_ids": [10, 11]},
                "intro.entity1": {"token_ids": [12, 13]},
                "intro.entity2": {"token_ids": [14, 15]},
                "op0.kind": {"token_ids": [20]},
                "op0.literal": {"token_ids": [30]},
                "op1.kind": {"token_ids": [21]},
                "op1.literal": {"token_ids": [31]},
                "query.position": {"token_ids": [30]},
            },
            "program": [
                {"kind": "left", "amount": 1},
                {"kind": "right", "amount": 2},
            ],
            "query": {"position": 0},
        }
        result = build([row])
        self.assertEqual(result["entity_widths"], [2])
        self.assertEqual(result["event_references"], 2)
        self.assertEqual({record["value"] for record in result["kind_patterns"]}, {"left", "right"})


if __name__ == "__main__":
    unittest.main()
