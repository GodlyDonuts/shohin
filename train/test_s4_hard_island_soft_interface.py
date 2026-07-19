#!/usr/bin/env python3

import unittest

import torch
from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_train
from s4_hard_island_soft_interface import (
    decode_hard_island_soft_interface,
    select_role_island,
)
from self_delimiting_event_tape import ROLE_INDEX, build_kind_lexicon, compile_row
from semantic_compiler_falsifier import candidate_names


class FrozenParser:
    class Model:
        class Config:
            vocab_size = 32768

        cfg = Config()

    model = Model()


class HardIslandSoftInterfaceTest(unittest.TestCase):
    def test_margin_selects_complete_stronger_island(self):
        logits = torch.zeros((8, len(ROLE_INDEX)))
        role = ROLE_INDEX["event.entity"]
        logits[1, role] = 2.0
        logits[4:6, role] = 3.0
        island, count = select_role_island(logits, "event.entity", 0, 8)
        self.assertEqual(count, 2)
        self.assertEqual(island, (4, 5))

    def test_perfect_hard_islands_decode_and_controls_break(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        row = build_train(3, 43, tokenizer, names, list(factor_catalogue("known"))[:3])[2]
        example = compile_row(row, tokenizer)
        length = len(example.ids)
        role_logits = torch.full((1, length, len(ROLE_INDEX)), -20.0)
        for position, role in enumerate(example.roles):
            role_logits[0, position, role] = 20.0
        amount_logits = torch.zeros((1, length, 2))
        query_logits = torch.zeros((1, length, 3))
        for index, event in enumerate(example.event_positions):
            amount_logits[0, list(event["literal"]), example.amount_targets[index]] = 20.0
        query_logits[0, list(example.query_positions), example.query_target] = 20.0
        outputs = {
            "role_logits": role_logits,
            "amount_logits": amount_logits,
            "query_logits": query_logits,
        }
        valid = torch.ones(length, dtype=torch.bool)
        lexicon = build_kind_lexicon([example])
        decoded = decode_hard_island_soft_interface(
            FrozenParser(), example, outputs, 0, valid, lexicon,
        )
        self.assertTrue(decoded["valid"], decoded)
        self.assertEqual(decoded["program"], example.program)
        self.assertEqual(decoded["final_state"], example.final_state)
        roster = decode_hard_island_soft_interface(
            FrozenParser(),
            example,
            outputs,
            0,
            valid,
            lexicon,
            roster_permutation=(1, 2, 0),
        )
        event = decode_hard_island_soft_interface(
            FrozenParser(), example, outputs, 0, valid, lexicon, region_shift=1,
        )
        self.assertNotEqual(roster["program"], example.program)
        self.assertNotEqual(event["program"], example.program)


if __name__ == "__main__":
    unittest.main()
