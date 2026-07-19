#!/usr/bin/env python3

import unittest

import torch
from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import factor_catalogue
from build_s4_self_delimiting_event_tape import build_train
from s4_set_identity_event_bus import (
    carrier_logits,
    decode_set_identity_example,
    masked_distribution,
    roster_distributions,
    roster_recovery_exact,
    vocabulary_carrier,
)
from self_delimiting_event_tape import ROLE_INDEX, build_kind_lexicon, compile_row
from semantic_compiler_falsifier import candidate_names


class PerfectSetBus:
    class Model:
        class Config:
            vocab_size = 32768

        cfg = Config()

    model = Model()

    def __init__(self, example):
        self.example = example

    def event_membership_scores(self, outputs, row, anchor_positions):
        del outputs, row
        event = next(
            event for event in self.example.event_positions
            if tuple(event["kind"]) == tuple(anchor_positions)
        )
        length = len(self.example.ids)

        def logits(positions):
            value = torch.full((length,), -20.0)
            value[list(positions)] = 20.0
            return value

        return {"entity": logits(event["entity"]), "literal": logits(event["literal"])}


class SetIdentityEventBusTest(unittest.TestCase):
    def test_vocabulary_carrier_is_occurrence_invariant(self):
        ids = torch.tensor([7, 11, 7, 5])
        left = vocabulary_carrier(ids, torch.tensor([0.5, 0.5, 0.0, 0.0]), 16)
        right = vocabulary_carrier(ids, torch.tensor([0.5, 0.5, 0.0, 0.0]), 16)
        other = vocabulary_carrier(ids, torch.tensor([0.0, 0.0, 0.5, 0.5]), 16)
        roster = torch.stack((left, other))
        self.assertEqual(int(carrier_logits(right, roster).argmax()), 0)

    def test_masked_distribution_has_no_padding_mass(self):
        weights = masked_distribution(
            torch.tensor([0.0, 0.0, 100.0]), torch.tensor([True, True, False]),
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(float(weights[-1]), 0.0)

    def test_roster_distributions_slice_batch_padding(self):
        outputs = {"role_logits": torch.zeros((1, 5, len(ROLE_INDEX)))}
        distributions = roster_distributions(
            outputs, 0, torch.tensor([True, True, True]),
        )
        self.assertEqual(tuple(value.shape for value in distributions), ((3,), (3,), (3,)))

    def test_perfect_soft_sets_decode_without_boundaries_or_depth(self):
        tokenizer = Tokenizer.from_file("artifacts/shohin-tok-32k.json")
        atoms, _ = candidate_names(tokenizer, 100)
        names = ["{}-{}".format(atoms[index], atoms[index + 1]) for index in range(40)]
        row = build_train(2, 11, tokenizer, names, list(factor_catalogue("known"))[:2])[1]
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
        valid = torch.ones(length, dtype=torch.bool)
        outputs = {
            "role_logits": role_logits,
            "amount_logits": amount_logits,
            "query_logits": query_logits,
        }
        bus = PerfectSetBus(example)
        decoded = decode_set_identity_example(
            bus, example, outputs, 0, valid, build_kind_lexicon([example]),
        )
        self.assertTrue(decoded["valid"], decoded)
        self.assertEqual(decoded["program"], example.program)
        self.assertEqual(decoded["final_state"], example.final_state)
        self.assertEqual(decoded["answer_identity"], example.answer_identity)
        self.assertTrue(roster_recovery_exact(
            example, outputs, 0, valid, bus.model.cfg.vocab_size,
        ))
        deranged = decode_set_identity_example(
            bus,
            example,
            outputs,
            0,
            valid,
            build_kind_lexicon([example]),
            roster_permutation=(1, 2, 0),
        )
        self.assertNotEqual(deranged["program"], example.program)


if __name__ == "__main__":
    unittest.main()
