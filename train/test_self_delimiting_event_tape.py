#!/usr/bin/env python3

import hashlib
import json
import unittest

import torch
from tokenizers import Tokenizer

from self_delimiting_event_tape import (
    KIND_TO_ID,
    ROLE_INDEX,
    build_kind_lexicon,
    compile_row,
    contiguous_runs,
    decode_example,
    execute_program,
)


class SelfDelimitingEventTapeModelTest(unittest.TestCase):
    def test_exact_s3_execution(self):
        state, answer = execute_program(
            (("right", 0, 2), ("left", 2, 1), ("right", 1, 1)), 1,
        )
        self.assertEqual(state, (2, 0, 1))
        self.assertEqual(answer, 0)

    def test_contiguous_runs(self):
        self.assertEqual(contiguous_runs([0, 2, 2, 0, 2], 2), ((1, 2), (4,)))

    def test_gold_logits_decode_without_depth_metadata(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        row = {
            "id": "tiny",
            "question": "Order aa, bb, cc. Move bb left by 1. Move cc right by 2. Slot 2?",
            "initial_order": ["aa", "bb", "cc"],
            "program": [
                {"kind": "left", "entity": "bb", "amount": 1},
                {"kind": "right", "entity": "cc", "amount": 2},
            ],
            "query": {"position": 1},
            "depth": 2,
            "surface_type": "test",
        }
        encoding = tokenizer.encode(row["question"])
        spans = {}
        for label, text, start in (
            ("intro.entity0", "aa", row["question"].index("aa")),
            ("intro.entity1", "bb", row["question"].index("bb")),
            ("intro.entity2", "cc", row["question"].index("cc")),
            ("op0.entity", "bb", row["question"].index("bb", 15)),
            ("op0.kind", "left", row["question"].index("left")),
            ("op0.literal", "1", row["question"].index("1")),
            ("op1.entity", "cc", row["question"].index("cc", 30)),
            ("op1.kind", "right", row["question"].index("right")),
            ("op1.literal", "2", row["question"].index("2")),
            ("query.position", "2", row["question"].rindex("2")),
        ):
            positions = [
                index for index, (left, right) in enumerate(encoding.offsets)
                if right > start and left < start + len(text)
            ]
            spans[label] = {"token_positions": positions}
        row["spans"] = spans
        row["token_ids_sha256"] = hashlib.sha256(
            json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        terminal, answer = execute_program((("left", 1, 1), ("right", 2, 2)), 1)
        row["answer"] = row["initial_order"][answer]
        example = compile_row(row, tokenizer)
        length = len(example.ids)
        role_logits = torch.full((1, length, len(ROLE_INDEX)), -10.0)
        for position, role in enumerate(example.roles):
            role_logits[0, position, role] = 10.0
        kind_logits = torch.zeros((1, length, 2))
        amount_logits = torch.zeros((1, length, 2))
        query_logits = torch.zeros((1, length, 3))
        for index, event in enumerate(example.event_positions):
            kind_logits[0, list(event["kind"]), example.kind_targets[index]] = 10
            amount_logits[0, list(event["literal"]), example.amount_targets[index]] = 10
        query_logits[0, list(example.query_positions), example.query_target] = 10
        decoded = decode_example(
            example,
            {
                "role_logits": role_logits,
                "kind_logits": kind_logits,
                "amount_logits": amount_logits,
                "query_logits": query_logits,
            },
            0,
            build_kind_lexicon([example]),
        )
        self.assertTrue(decoded["valid"])
        self.assertEqual(decoded["event_count"], 2)
        self.assertEqual(decoded["program"], example.program)
        self.assertEqual(decoded["final_state"], terminal)
        self.assertEqual(KIND_TO_ID[decoded["program"][0][0]], 0)


if __name__ == "__main__":
    unittest.main()
