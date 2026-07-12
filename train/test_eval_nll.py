import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_nll import batch_nll, parse_input_spec, text_rows, token_blocks


class EvalNllTest(unittest.TestCase):
    def test_parse_input_spec_requires_unique_label_shape_and_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitor.jsonl"
            path.write_text('{"text": "hello"}\n')
            label, parsed = parse_input_spec(f"english={path}")
            self.assertEqual(label, "english")
            self.assertEqual(parsed, path)
            with self.assertRaises(ValueError):
                parse_input_spec(str(path))
            with self.assertRaises(ValueError):
                parse_input_spec(f"bad label={path}")

    def test_text_rows_and_blocks_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitor.jsonl"
            path.write_text("".join([
                json.dumps({"text": "first"}) + "\n",
                json.dumps({"text": "second"}) + "\n",
            ]))
            self.assertEqual(list(text_rows(path, "text")), ["first", "second"])
            blocks = list(token_blocks([[1, 2, 0], [3, 4, 0]], seq_len=3, max_sequences=2))
            self.assertEqual(blocks, [[1, 2, 0, 3]])

    def test_text_rows_rejects_missing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitor.jsonl"
            path.write_text(json.dumps({"wrong": "field"}) + "\n")
            with self.assertRaises(ValueError):
                list(text_rows(path, "text"))

    def test_batch_nll_excludes_auxiliary_training_loss(self):
        class Model:
            def __call__(self, inputs):
                logits = torch.zeros((inputs.size(0), inputs.size(1), 2))
                return logits, torch.tensor(999.0)

        inputs = torch.tensor([[0, 1]])
        targets = torch.tensor([[1, 0]])
        self.assertAlmostEqual(float(batch_nll(Model(), inputs, targets, "cpu")), 0.693147, places=5)


if __name__ == "__main__":
    unittest.main()
