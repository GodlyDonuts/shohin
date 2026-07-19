#!/usr/bin/env python3

import unittest

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_train
from referential_literal_pointer_compiler import KIND_TO_ID
from s4_hard_island_soft_interface import decode_hard_island_soft_interface
from s5_learned_generator_executor import (
    GeneratorFactoredS3Executor,
    decode_v5_program,
    unit_generator_examples,
)
from self_delimiting_event_tape import ROLE_INDEX, build_kind_lexicon, compile_row
from semantic_compiler_falsifier import candidate_names


class FrozenParser:
    class Model:
        class Config:
            vocab_size = 32768

        cfg = Config()

    model = Model()


def perfect_outputs(example):
    length = len(example.ids)
    role_logits = torch.full((1, length, len(ROLE_INDEX)), -20.0)
    for position, role in enumerate(example.roles):
        role_logits[0, position, role] = 20.0
    amount_logits = torch.zeros((1, length, 2))
    query_logits = torch.zeros((1, length, 3))
    for index, event in enumerate(example.event_positions):
        amount_logits[0, list(event["literal"]), example.amount_targets[index]] = 20.0
    query_logits[0, list(example.query_positions), example.query_target] = 20.0
    return {
        "role_logits": role_logits,
        "amount_logits": amount_logits,
        "query_logits": query_logits,
    }


class LearnedGeneratorExecutorTest(unittest.TestCase):
    def test_training_contract_contains_only_six_unit_generators(self):
        locations, directions, targets = unit_generator_examples()
        self.assertEqual(locations.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(directions.tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(targets.numel(), 6)
        self.assertTrue(((targets >= 0) & (targets < 6)).all())

    def test_tied_generator_learns_units_and_composes_unseen_amount_two(self):
        torch.manual_seed(7)
        executor = GeneratorFactoredS3Executor(width=32)
        optimizer = torch.optim.AdamW(executor.generator.parameters(), lr=1e-2)
        locations, directions, targets = unit_generator_examples()
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(executor.generator(locations, directions), targets)
            loss.backward()
            optimizer.step()
        self.assertTrue(executor.generator(locations, directions).argmax(-1).eq(targets).all())
        outputs = executor(
            identities=torch.tensor([[1, 1]]),
            directions=torch.tensor([[KIND_TO_ID["right"], KIND_TO_ID["left"]]]),
            amounts=torch.tensor([[2, 1]]),
            query_positions=torch.tensor([0]),
        )
        self.assertEqual(tuple(outputs["assignment"].argmax(-1)[0].tolist()), (0, 1, 2))
        self.assertEqual(int(outputs["answer_probabilities"].argmax(-1)[0]), 0)

    def test_program_only_decoder_matches_frozen_v5(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        row = build_train(3, 43, tokenizer, names, list(factor_catalogue("known"))[:3])[2]
        example = compile_row(row, tokenizer)
        outputs = perfect_outputs(example)
        valid = torch.ones(len(example.ids), dtype=torch.bool)
        lexicon = build_kind_lexicon([example])
        old = decode_hard_island_soft_interface(
            FrozenParser(), example, outputs, 0, valid, lexicon,
        )
        new = decode_v5_program(FrozenParser(), example, outputs, 0, valid, lexicon)
        self.assertTrue(new["valid"])
        self.assertEqual(new["program"], old["program"])
        self.assertEqual(new["query"], old["query"])
        self.assertNotIn("final_state", new)
        self.assertNotIn("answer_identity", new)


if __name__ == "__main__":
    unittest.main()
