#!/usr/bin/env python3
"""Focused deterministic contracts for the RSP-C1 artifact generator."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

import generate_residual_packet_v1 as generator


TOKENIZER_PATH = Path(__file__).parents[1] / "artifacts" / "shohin-tok-32k.json"


def parse_update_prompt(prompt: str):
    prefix = "Packet:\n"
    marker = "\nObserved result: "
    suffix = "\nNext packet:"
    assert prompt.startswith(prefix) and prompt.endswith(suffix)
    packet_text, separator, observed = prompt[len(prefix) : -len(suffix)].rpartition(marker)
    assert separator
    packet = generator.protocol_module().parse_packet(packet_text)
    assert packet is not None
    return packet, int(observed)


class ResidualPacketGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        cls.board = generator.board_payload()
        cls.programs = generator.build_training_programs(cls.board["rows"], cls.tokenizer)
        cls.mapping = generator.build_sham_permutation(cls.programs)
        cls.treatment, cls.sham = generator.build_training_arms(
            cls.programs, cls.board["rows"], cls.mapping
        )

    def test_exact_repo_modules_ignore_nonpackage_train_collision(self):
        had_train = "train" in sys.modules
        previous_train = sys.modules.get("train")
        previous_protocol = generator._PROTOCOL
        previous_sft_encoding = generator._SFT_ENCODING
        sys.modules["train"] = types.ModuleType("train")
        generator._PROTOCOL = None
        generator._SFT_ENCODING = None
        try:
            protocol = generator.protocol_module()
            sft_encoding = generator.sft_encoding_module()
            self.assertEqual(Path(protocol.__file__).resolve(), generator.PROTOCOL_PATH)
            self.assertEqual(protocol.__source_sha256__, generator.EXPECTED_PROTOCOL_SHA256)
            self.assertEqual(Path(sft_encoding.__file__).resolve(), generator.SFT_ENCODING_PATH)
            self.assertEqual(
                sft_encoding.__source_sha256__, generator.EXPECTED_SFT_ENCODING_SHA256
            )
            eos_id = self.tokenizer.token_to_id("<|endoftext|>")
            accounting = generator.token_accounting(
                self.treatment[:4], self.tokenizer, eos_id
            )
            self.assertEqual(accounting["examples"], 4)
        finally:
            generator._PROTOCOL = previous_protocol
            generator._SFT_ENCODING = previous_sft_encoding
            if had_train:
                sys.modules["train"] = previous_train
            else:
                sys.modules.pop("train", None)

    def test_board_has_exact_frozen_strata_and_hashes(self):
        rows = self.board["rows"]
        self.assertEqual(len(rows), 256)
        self.assertEqual(
            Counter(row["stratum"] for row in rows),
            Counter({stratum: 64 for stratum in generator.STRATUM_ORDER}),
        )
        self.assertEqual(self.board["rows_sha256"], generator.EXPECTED_BOARD_ROWS_SHA256)
        self.assertEqual(
            generator.digest_bytes(generator.pretty_json_bytes(self.board)),
            generator.EXPECTED_BOARD_SHA256,
        )
        self.assertEqual(len({row["source"] for row in rows}), 256)
        self.assertEqual(len({row["answer"] for row in rows}), 256)
        self.assertEqual(len({tuple(row["trajectory"]) for row in rows}), 256)
        self.assertTrue(all(min(row["trajectory"]) > 0 for row in rows))

        held = Counter()
        for row in rows:
            kinds = generator.operation_types(row["operations"])
            held_bigrams = [
                pair for pair in zip(kinds, kinds[1:]) if pair in generator.HELD_OUT_BIGRAMS
            ]
            if row["stratum"] == "renderer_ood":
                self.assertEqual(len(row["operations"]), 3)
                self.assertEqual(row["template_id"], generator.reserved_template_id())
                self.assertFalse(held_bigrams)
            elif row["stratum"] == "value_ood":
                self.assertEqual(len(row["operations"]), 3)
                self.assertTrue(100 <= row["initial_state"] <= 299)
                self.assertFalse(held_bigrams)
            elif row["stratum"] == "order_ood":
                self.assertIn(len(row["operations"]), (3, 4))
                self.assertEqual(len(held_bigrams), 1)
                held[held_bigrams[0]] += 1
            else:
                self.assertEqual(len(row["operations"]), 5)
                self.assertFalse(held_bigrams)
        self.assertEqual(
            held,
            Counter({generator.HELD_OUT_BIGRAMS[0]: 32, generator.HELD_OUT_BIGRAMS[1]: 32}),
        )

    def test_training_counts_false_observations_and_no_completion_leakage(self):
        self.assertEqual(len(self.programs), 4096)
        self.assertEqual(
            Counter(len(program["operations"]) for program in self.programs),
            Counter(generator.TRAIN_LENGTH_COUNTS),
        )
        self.assertEqual(len(self.treatment), 16_384)
        self.assertEqual(Counter(row["kind"] for row in self.treatment), {"compiler": 4096, "updater": 12_288})
        eval_answers = {row["answer"] for row in self.board["rows"]}
        final_by_program = {
            program["id"]: program["final_answer"] for program in self.programs
        }
        for treatment_row, sham_row in zip(self.treatment, self.sham):
            response_values = set(generator.integer_occurrences(treatment_row["response"]))
            sham_values = set(generator.integer_occurrences(sham_row["response"]))
            self.assertFalse(response_values & eval_answers)
            self.assertFalse(sham_values & eval_answers)
            source_answer = final_by_program[treatment_row["program_id"]]
            self.assertNotIn(source_answer, response_values)
            self.assertNotIn(source_answer, sham_values)
            if treatment_row["kind"] == "updater":
                self.assertEqual(treatment_row, sham_row)
                packet, observed = parse_update_prompt(treatment_row["completion_prompt"])
                self.assertNotEqual(
                    observed,
                    generator.apply_operation(packet["state"], packet["plan"][0]),
                )
                self.assertNotIn(packet["state"], eval_answers)
                self.assertNotIn(observed, eval_answers)

    def test_sham_is_token_matched_seeded_derangement(self):
        self.assertEqual(len(self.mapping), 4096)
        self.assertEqual(len(set(self.mapping)), 4096)
        strata = Counter(generator.sham_stratum(program) for program in self.programs)
        self.assertGreaterEqual(min(strata.values()), 2)
        for index, donor in enumerate(self.mapping):
            recipient_program = self.programs[index]
            donor_program = self.programs[donor]
            self.assertNotEqual(index, donor)
            self.assertEqual(
                generator.sham_stratum(recipient_program),
                generator.sham_stratum(donor_program),
            )
            self.assertNotEqual(
                generator.semantic_signature(
                    recipient_program["initial_state"], recipient_program["operations"]
                ),
                generator.semantic_signature(
                    donor_program["initial_state"], donor_program["operations"]
                ),
            )
            self.assertNotEqual(recipient_program["trajectory"], donor_program["trajectory"])
            self.assertNotEqual(recipient_program["final_answer"], donor_program["final_answer"])
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.assertEqual(
            generator.token_accounting(self.treatment, self.tokenizer, eos_id),
            generator.token_accounting(self.sham, self.tokenizer, eos_id),
        )

    def test_immutable_writers_are_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "small.json"
            jsonl_path = Path(directory) / "small.jsonl"
            generator.write_immutable_json(json_path, {"seed": generator.BOARD_SEED})
            generator.write_immutable_jsonl(jsonl_path, [{"row": 1}])
            self.assertEqual(json_path.stat().st_mode & 0o222, 0)
            self.assertEqual(jsonl_path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                generator.write_immutable_json(json_path, {"seed": 0})
            os.chmod(json_path, 0o600)
            os.chmod(jsonl_path, 0o600)


if __name__ == "__main__":
    unittest.main()
