#!/usr/bin/env python3

import hashlib
import json
import unittest

import torch
from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_train
from build_s4_structural_lexicon import build as build_structural_lexicon
from semantic_compiler_falsifier import candidate_names
from self_delimiting_event_tape import (
    KIND_TO_ID,
    ROLE_INDEX,
    build_kind_lexicon,
    compile_row,
    contiguous_runs,
    decode_example,
    decode_structural_example,
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
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        row = build_train(2, 11, tokenizer, names, list(factor_catalogue("known"))[:2])[1]
        encoding = tokenizer.encode(row["question"])
        self.assertEqual(
            row["token_ids_sha256"],
            hashlib.sha256(
                json.dumps(encoding.ids, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        example = compile_row(row, tokenizer)
        terminal, _ = execute_program(example.program, example.query_target)
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
        self.assertEqual(
            KIND_TO_ID[decoded["program"][0][0]], KIND_TO_ID[example.program[0][0]],
        )
        self.assertEqual(decoded["failure_reason"], "none")
        structural = decode_structural_example(
            example,
            {
                "role_logits": role_logits,
                "kind_logits": kind_logits,
                "amount_logits": amount_logits,
                "query_logits": query_logits,
            },
            0,
            build_structural_lexicon([row]),
        )
        self.assertTrue(structural["valid"], structural)
        self.assertEqual(structural["program"], example.program)


if __name__ == "__main__":
    unittest.main()
