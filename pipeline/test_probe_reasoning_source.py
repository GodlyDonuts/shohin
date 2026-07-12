#!/usr/bin/env python3
"""Non-network tests for source-probe normalization and overlap reporting."""
import json
import tempfile
import unittest
from pathlib import Path

from probe_reasoning_source import describe_rows, load_eval_index, parse_card_frontmatter


class ReasoningSourceProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "eval.jsonl"
        path.write_text(json.dumps({
            "question": "What is the sum of one two three four five six seven eight nine ten eleven twelve thirteen?"
        }) + "\n")
        self.index = load_eval_index(self.directory.name)

    def test_reports_exact_and_ngram_overlap_without_retaining_rows(self):
        report = describe_rows([{
            "problem": "What is the sum of one two three four five six seven eight nine ten eleven twelve thirteen?",
            "solutions": ["91"],
            "metadata": {"source": "synthetic"},
        }], self.index)
        overlap = report["sample_eval_overlap"]
        self.assertEqual(overlap["exact_prompt_rows"], 1)
        self.assertEqual(overlap["eval_13gram_rows"], 1)
        self.assertEqual(overlap["exact_hits_by_field"], {"problem": 1})
        self.assertIn("solutions", report["top_level_fields"])

    def test_short_nonmatching_strings_do_not_trigger_overlap(self):
        report = describe_rows([{"prompt": "unrelated short prompt", "answer": "no"}], self.index)
        overlap = report["sample_eval_overlap"]
        self.assertEqual(overlap["exact_prompt_rows"], 0)
        self.assertEqual(overlap["eval_13gram_rows"], 0)

    def test_reads_only_flat_provenance_declarations_from_card(self):
        card = "---\nlicense: cc-by-4.0\nlanguage: en\npretty_name: Test Set\n---\n# body\n"
        self.assertEqual(parse_card_frontmatter(card), {
            "license": "cc-by-4.0",
            "language": "en",
            "pretty_name": "Test Set",
        })


if __name__ == "__main__":
    unittest.main()
