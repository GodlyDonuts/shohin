from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest

from eval_s9_2_global_anchor import (
    claim_development_access,
    development_access_ledger_path,
    verify_evaluation_bindings,
)
from s9_occurrence_quotient_compiler import sha256_file


class S92EvaluationCustodyTest(unittest.TestCase):
    def test_development_ledger_is_deterministic_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board_sha = "a" * 64
            path = development_access_ledger_path(root, board_sha)
            payload = {
                "schema": "r12_s9_2_development_access_v1",
                "board_report_sha256": board_sha,
                "development_accesses": 1,
            }
            claim_development_access(path, payload)
            self.assertEqual(
                json.loads(path.read_text()),
                payload,
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            with self.assertRaises(FileExistsError):
                claim_development_access(path, payload)

    def test_evaluation_bindings_require_base_tokenizer_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            tokenizer = root / "tokenizer.json"
            base.write_bytes(b"base")
            tokenizer.write_bytes(b"tokenizer")
            checkpoint = {
                "base_sha256": sha256_file(base),
                "tokenizer_sha256": sha256_file(tokenizer),
                "source_commit": "b" * 40,
            }
            report = {
                "tokenizer_sha256": sha256_file(tokenizer),
                "source_commit": "b" * 40,
            }
            verify_evaluation_bindings(checkpoint, report, base, tokenizer)
            base.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "base hash mismatch"):
                verify_evaluation_bindings(checkpoint, report, base, tokenizer)

    def test_tokenizer_must_match_both_checkpoint_and_board(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            tokenizer = root / "tokenizer.json"
            base.write_bytes(b"base")
            tokenizer.write_bytes(b"tokenizer")
            checkpoint = {
                "base_sha256": sha256_file(base),
                "tokenizer_sha256": sha256_file(tokenizer),
                "source_commit": "c" * 40,
            }
            report = {
                "tokenizer_sha256": "0" * 64,
                "source_commit": "c" * 40,
            }
            with self.assertRaisesRegex(ValueError, "tokenizer/board"):
                verify_evaluation_bindings(checkpoint, report, base, tokenizer)


if __name__ == "__main__":
    unittest.main()
