#!/usr/bin/env python3

import unittest

import torch
from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_train
from s4_event_relative_pointer import decode_event_relative_example, pointer_span
from self_delimiting_event_tape import (
    ROLE_INDEX,
    build_kind_lexicon,
    compile_row,
)
from semantic_compiler_falsifier import candidate_names


class PerfectPointerParser:
    def __init__(self, example):
        self.example = example

    def event_pointer_scores(self, outputs, row, anchor_positions):
        del outputs, row
        event = next(
            event for event in self.example.event_positions
            if tuple(event["kind"]) == tuple(anchor_positions)
        )
        length = len(self.example.ids)

        def logits(target):
            values = torch.full((length,), -10.0)
            values[target] = 10.0
            return values

        return {
            "entity_start": logits(min(event["entity"])),
            "entity_end": logits(max(event["entity"])),
            "literal_start": logits(min(event["literal"])),
            "literal_end": logits(max(event["literal"])),
        }


class S4EventRelativePointerTest(unittest.TestCase):
    def test_pointer_span_rejects_reverse_boundaries(self):
        self.assertIsNone(pointer_span(torch.tensor([0.0, 2.0]), torch.tensor([2.0, 0.0]), 2))

    def test_perfect_source_pointers_decode_without_depth(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        row = build_train(2, 11, tokenizer, names, list(factor_catalogue("known"))[:2])[1]
        example = compile_row(row, tokenizer)
        length = len(example.ids)
        role_logits = torch.full((1, length, len(ROLE_INDEX)), -10.0)
        for position, role in enumerate(example.roles):
            role_logits[0, position, role] = 10.0
        intro_start = torch.full((1, length, 3), -10.0)
        intro_end = torch.full((1, length, 3), -10.0)
        for identity, positions in enumerate(example.intro_positions):
            intro_start[0, min(positions), identity] = 10.0
            intro_end[0, max(positions), identity] = 10.0
        query_start = torch.full((1, length), -10.0)
        query_end = torch.full((1, length), -10.0)
        query_start[0, min(example.query_positions)] = 10.0
        query_end[0, max(example.query_positions)] = 10.0
        amount_logits = torch.zeros((1, length, 2))
        query_logits = torch.zeros((1, length, 3))
        for index, event in enumerate(example.event_positions):
            amount_logits[0, list(event["literal"]), example.amount_targets[index]] = 10.0
        query_logits[0, list(example.query_positions), example.query_target] = 10.0
        decoded = decode_event_relative_example(
            PerfectPointerParser(example),
            example,
            {
                "role_logits": role_logits,
                "intro_start_logits": intro_start,
                "intro_end_logits": intro_end,
                "query_start_logits": query_start,
                "query_end_logits": query_end,
                "amount_logits": amount_logits,
                "query_logits": query_logits,
            },
            0,
            build_kind_lexicon([example]),
        )
        self.assertTrue(decoded["valid"], decoded)
        self.assertEqual(decoded["program"], example.program)
        self.assertEqual(decoded["final_state"], example.final_state)
        self.assertEqual(decoded["answer_identity"], example.answer_identity)


if __name__ == "__main__":
    unittest.main()
