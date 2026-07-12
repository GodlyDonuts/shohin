#!/usr/bin/env python3
"""Non-network integrity tests for interrupted TACO full-audit recovery."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from audit_taco_verified import read_completed_partial


class TacoAuditResumeTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {
            "17": {"problem_id": 17, "response": "print('ok')"},
        }

    def write_partial(self, row):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.jsonl.partial"
        path.write_text(json.dumps(row) + "\n")
        return path

    def test_accepts_matching_full_verified_row(self):
        path = self.write_partial({
            "problem_id": 17,
            "response": "print('ok')",
            "full_verified_cases": 4,
        })
        self.assertEqual(set(read_completed_partial(path, self.candidates)), {"17"})

    def test_rejects_changed_response(self):
        path = self.write_partial({
            "problem_id": 17,
            "response": "print('different')",
            "full_verified_cases": 4,
        })
        with self.assertRaises(SystemExit):
            read_completed_partial(path, self.candidates)

    def test_rejects_unverified_row(self):
        path = self.write_partial({
            "problem_id": 17,
            "response": "print('ok')",
            "full_verified_cases": 0,
        })
        with self.assertRaises(SystemExit):
            read_completed_partial(path, self.candidates)

    def test_partial_is_visible_after_flush(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "progress.jsonl.partial"
        with path.open("w") as output:
            output.write(json.dumps({"problem_id": 17}) + "\n")
            output.flush()
            os.fsync(output.fileno())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
