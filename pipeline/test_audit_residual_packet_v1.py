#!/usr/bin/env python3
"""Independent-admission and adversarial fixtures for RSP-C1 artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tokenizers import Tokenizer

import audit_residual_packet_v1 as auditor
import generate_residual_packet_v1 as generator


TOKENIZER_PATH = Path(__file__).parents[1] / "artifacts" / "shohin-tok-32k.json"


class ResidualPacketAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.board = cls.root / "board.json"
        cls.treatment = cls.root / "treatment.jsonl"
        cls.sham = cls.root / "sham.jsonl"
        cls.manifest = cls.root / "manifest.json"
        generator.write_board(cls.board)
        generator.generate_training_outputs(
            cls.board,
            TOKENIZER_PATH,
            cls.treatment,
            cls.sham,
            cls.manifest,
        )

    @classmethod
    def tearDownClass(cls):
        for path in cls.root.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o600)
        cls.temporary.cleanup()

    def test_valid_artifacts_admit_only_after_full_recomputation(self):
        report = auditor.audit_artifacts(
            self.board, TOKENIZER_PATH, self.treatment, self.sham, self.manifest
        )
        self.assertTrue(report["admitted"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["board"]["case_count"], 256)
        self.assertEqual(report["training"]["programs"], 4096)
        self.assertEqual(report["training"]["treatment_rows"], 16_384)
        self.assertEqual(report["training"]["sham_rows"], 16_384)
        self.assertEqual(report["row_metrics"]["correct_arithmetic_updater_transitions"], 0)
        self.assertEqual(report["row_metrics"]["reconstructed_sham_mapping_mismatches"], 0)
        self.assertEqual(report["row_metrics"]["evaluation_answer_response_occurrences"], 0)
        self.assertEqual(report["overlap"]["normalized_13_token_source_ngram_overlap"], 0)
        self.assertEqual(report["token_accounting"]["treatment"], report["token_accounting"]["sham"])

    def test_auditor_exact_loader_ignores_nonpackage_train_collision(self):
        had_train = "train" in sys.modules
        previous_train = sys.modules.get("train")
        previous_protocol = auditor._PROTOCOL
        previous_sft_encoding = auditor._SFT_ENCODING
        sys.modules["train"] = types.ModuleType("train")
        auditor._PROTOCOL = None
        auditor._SFT_ENCODING = None
        try:
            protocol = auditor.protocol_module()
            sft_encoding = auditor.sft_encoding_module()
            self.assertEqual(Path(protocol.__file__).resolve(), auditor.PROTOCOL_PATH)
            self.assertEqual(protocol.__source_sha256__, auditor.EXPECTED_PROTOCOL_SHA256)
            self.assertEqual(Path(sft_encoding.__file__).resolve(), auditor.SFT_ENCODING_PATH)
            self.assertEqual(
                sft_encoding.__source_sha256__, auditor.EXPECTED_SFT_ENCODING_SHA256
            )
        finally:
            auditor._PROTOCOL = previous_protocol
            auditor._SFT_ENCODING = previous_sft_encoding
            if had_train:
                sys.modules["train"] = previous_train
            else:
                sys.modules.pop("train", None)

    def test_rejects_arithmetic_correct_observation_even_with_rehashed_manifest(self):
        mutation = self.root / "mutation"
        mutation.mkdir()
        treatment_rows, _ = auditor.read_immutable_jsonl(self.treatment, "treatment")
        sham_rows, _ = auditor.read_immutable_jsonl(self.sham, "sham")
        protocol = auditor.protocol_module()
        row_index = next(
            index for index, row in enumerate(treatment_rows) if row["kind"] == "updater"
        )
        parsed = auditor.parse_updater_prompt(treatment_rows[row_index]["completion_prompt"])
        self.assertIsNotNone(parsed)
        packet_text, packet, _ = parsed
        correct = auditor.apply_operation(packet["state"], packet["plan"][0])
        prompt = protocol.update_prompt(packet_text, correct)
        response = protocol.expected_update(packet_text, correct)
        for rows in (treatment_rows, sham_rows):
            rows[row_index]["completion_prompt"] = prompt
            rows[row_index]["question"] = prompt
            rows[row_index]["response"] = response

        treatment_path = mutation / "treatment.jsonl"
        sham_path = mutation / "sham.jsonl"
        manifest_path = mutation / "manifest.json"
        generator.write_immutable_jsonl(treatment_path, treatment_rows)
        generator.write_immutable_jsonl(sham_path, sham_rows)
        manifest = json.loads(self.manifest.read_text())
        manifest["artifacts"]["treatment_sha256"] = generator.sha256_file(treatment_path)
        manifest["artifacts"]["sham_sha256"] = generator.sha256_file(sham_path)
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        manifest["token_accounting"] = {
            "treatment": auditor.token_accounting(treatment_rows, tokenizer, eos_id),
            "sham": auditor.token_accounting(sham_rows, tokenizer, eos_id),
        }
        generator.write_immutable_json(manifest_path, manifest)

        report = auditor.audit_artifacts(
            self.board, TOKENIZER_PATH, treatment_path, sham_path, manifest_path
        )
        self.assertFalse(report["admitted"])
        self.assertGreater(
            report["row_metrics"]["correct_arithmetic_updater_transitions"], 0
        )
        self.assertIn("correct_arithmetic_updater_transitions", report["failures"])
        self.assertIn("treatment_seed_reconstruction", report["failures"])
        self.assertIn("sham_seed_reconstruction", report["failures"])

    def test_rejects_writable_or_noncanonical_inputs_before_admission(self):
        writable = self.root / "writable-board.json"
        shutil.copyfile(self.board, writable)
        os.chmod(writable, 0o644)
        with self.assertRaisesRegex(PermissionError, "writable"):
            auditor.read_immutable_json(writable, "board")
        noncanonical = self.root / "noncanonical.jsonl"
        noncanonical.write_text('{"b": 2, "a": 1}\n')
        os.chmod(noncanonical, 0o444)
        with self.assertRaisesRegex(ValueError, "canonical JSONL"):
            auditor.read_immutable_jsonl(noncanonical, "training")


if __name__ == "__main__":
    unittest.main()
