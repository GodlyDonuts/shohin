#!/usr/bin/env python3
"""Non-network tests for full prompt-and-completion overlap detection."""
import unittest

from audit_training_text_overlap import audit_rows, grams, normalized


class TrainingTextOverlapTests(unittest.TestCase):
    def setUp(self):
        self.prompt = "What is the sum of one two three four five six seven eight nine ten eleven twelve thirteen"
        self.exact = {normalized(self.prompt)}
        self.ngrams = set(grams(self.prompt, 13))

    def test_detects_overlap_in_completion_when_prompt_is_clean(self):
        result = audit_rows([{
            "question": "Solve a different problem",
            "response": "The source quoted: " + self.prompt,
            "completion_prompt": "Solve a different problem\nAnswer:",
            "source": "fixture",
        }], ("question", "response", "completion_prompt"), self.exact, self.ngrams, 13)
        self.assertEqual(result["overlap"]["exact_rows"], 0)
        self.assertEqual(result["overlap"]["ngram_rows"], 1)
        self.assertEqual(result["overlap"]["ngram_hits_by_field"], {"response": 1})

    def test_records_no_training_text_in_examples(self):
        result = audit_rows([{
            "question": self.prompt,
            "response": "42",
            "completion_prompt": self.prompt,
        }], ("question", "response", "completion_prompt"), self.exact, self.ngrams, 13)
        example = result["overlap"]["examples"][0]
        self.assertEqual(set(example), {"line", "field", "kind", "source", "training_group"})
        self.assertNotIn("question", example)


if __name__ == "__main__":
    unittest.main()
