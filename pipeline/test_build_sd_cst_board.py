#!/usr/bin/env python3

import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_sd_cst_board import (
    EVENT_COUNT,
    OPERATION_COUNT,
    SURFACE_TYPES,
    build_all,
)


class SDCSTBoardBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.development, cls.confirmation = build_all(
            train_rows=36,
            development_families=18,
            confirmation_families=18,
            seed=2026072002,
        )

    def test_every_record_has_withheld_query_and_interleaved_stop(self):
        for row in self.train + self.development + self.confirmation:
            self.assertEqual(len(row["program_text"].splitlines()), 9)
            self.assertEqual(len(row["late_query_text"].splitlines()), 1)
            self.assertNotIn("query", row["program_text"].lower())
            self.assertNotIn("position", row["program_text"].lower())
            self.assertEqual(row["program_text"].count("STOP"), 1)
            self.assertEqual(len(row["compiler_targets"]["event_slots"]), EVENT_COUNT)
            self.assertEqual(
                sum(
                    slot["kind"] == "stop"
                    for slot in row["compiler_targets"]["event_slots"]
                ),
                1,
            )
            self.assertEqual(
                sum(
                    slot["kind"] != "stop"
                    for slot in row["compiler_targets"]["event_slots"]
                ),
                OPERATION_COUNT,
            )
            self.assertIn(row["compiler_targets"]["halt_after"], range(1, 7))
            self.assertGreater(
                OPERATION_COUNT - row["compiler_targets"]["halt_after"], 0
            )
            self.assertFalse(
                row["model_input_contract"]["combined_prompt_is_model_input"]
            )

    def test_training_contains_compiler_fields_but_no_answer_state_or_trajectory(self):
        forbidden = {"oracle", "answer", "answer_role", "final_state_roles"}

        def keys(value):
            found = set()
            if isinstance(value, dict):
                for key, child in value.items():
                    found.add(key)
                    found.update(keys(child))
            elif isinstance(value, list):
                for child in value:
                    found.update(keys(child))
            return found

        for row in self.train:
            row_keys = keys(row)
            self.assertFalse(forbidden & row_keys)
            self.assertFalse(any("trajectory" in key for key in row_keys))
            self.assertNotIn("query_position", row["compiler_targets"])
            self.assertEqual(row["supervision"], "compiler_fields_only")

    def test_query_twins_share_program_bytes_and_change_only_late_query_semantics(self):
        for rows in (self.development, self.confirmation):
            families = {}
            for row in rows:
                families.setdefault(row["family_id"], {})[row["variant"]] = row
            for family in families.values():
                self.assertEqual(set(family), set(SURFACE_TYPES))
                canonical = family["canonical"]
                query_swap = family["query_swap"]
                self.assertEqual(
                    canonical["program_text"].encode(),
                    query_swap["program_text"].encode(),
                )
                self.assertNotEqual(
                    canonical["late_query_text"], query_swap["late_query_text"]
                )
                self.assertNotEqual(
                    canonical["oracle"]["answer_role"],
                    query_swap["oracle"]["answer_role"],
                )

    def test_cli_writes_private_sealed_file_without_reading_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "board"
            subprocess.run(
                [
                    sys.executable,
                    "pipeline/build_sd_cst_board.py",
                    "--out-dir",
                    str(out_dir),
                    "--seed",
                    "2026072003",
                    "--train-rows",
                    "18",
                    "--development-families",
                    "18",
                    "--confirmation-families",
                    "18",
                    "--test-only-unfrozen-source",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            sealed = out_dir / "confirmation.sealed.jsonl"
            self.assertTrue(sealed.exists())
            self.assertEqual(stat.S_IMODE(sealed.stat().st_mode), 0o600)
            self.assertTrue((out_dir / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
