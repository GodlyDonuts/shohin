import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN.parent
PIPELINE = ROOT / "pipeline"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(PIPELINE))

import audit_counterfactual_cursor_action_canary as audit  # noqa: E402
import generate_counterfactual_cursor_action_canary as generate  # noqa: E402
from counterfactual_cursor_action_data import load_canary  # noqa: E402
from train_counterfactual_cursor_action import (  # noqa: E402
    ARMS,
    EPOCHS,
    compile_training_units,
    epoch_unit_orders,
    lr_scale,
)


TOKENIZER = ROOT / "artifacts/shohin-tok-32k.json"
EVALGRAMS = ROOT / "artifacts/evals/evalgrams.pkl"


class TrainCounterfactualCursorActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = generate.generate_document(TOKENIZER, bind_identity=False)
        cls.canary_bytes = json.dumps(
            cls.document, indent=2, sort_keys=True,
        ).encode("ascii") + b"\n"
        cls.canary_sha256 = hashlib.sha256(cls.canary_bytes).hexdigest()
        cls.report = audit.audit_document(
            cls.document, TOKENIZER, EVALGRAMS,
            canary_file_sha256=cls.canary_sha256,
        )

    def load(self):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        canary_path = root / "canary.json"
        audit_path = root / "audit.json"
        tokenizer_path = root / "tokenizer.json"
        canary_path.write_bytes(self.canary_bytes)
        audit_path.write_text(json.dumps(self.report), encoding="ascii")
        shutil.copyfile(TOKENIZER, tokenizer_path)
        for path in (canary_path, audit_path, tokenizer_path):
            path.chmod(0o444)
        return directory, load_canary(canary_path, audit_path, tokenizer_path)

    def test_compiled_units_have_exact_relation_geometry(self):
        directory, dataset = self.load()
        with directory:
            units = compile_training_units(dataset.split("train"))
        self.assertEqual(len(units), 288)
        self.assertTrue(all(unit.examples == 60 for unit in units))
        self.assertTrue(all(unit.renderer_count == 6 for unit in units))
        self.assertEqual(set(units[0].cursors), set(range(5)))
        self.assertEqual(len(units[0].source_indices), 60)
        self.assertEqual(len(units[0].cell_indices), 60)

    def test_epoch_orders_are_reproducible_complete_and_distinct(self):
        observed = epoch_unit_orders(288)
        self.assertEqual(observed, epoch_unit_orders(288))
        self.assertEqual(len(observed), EPOCHS)
        self.assertTrue(all(set(order) == set(range(288)) for order in observed))
        self.assertEqual(len(set(observed)), EPOCHS)

    def test_learning_rate_schedule_is_bounded(self):
        values = [lr_scale(update, 1152) for update in range(1152)]
        self.assertGreater(values[49], values[0])
        self.assertAlmostEqual(values[49], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(0.0 < value <= 1.0 for value in values))
        self.assertTrue(all(0.1 <= value <= 1.0 for value in values[49:]))

    def test_arm_order_and_treatment_name_are_frozen(self):
        self.assertEqual(ARMS, (
            "orbit_interchange", "ordinary_loss", "relation_sham",
            "source_only", "cursor_table", "text_cursor_lora",
        ))


if __name__ == "__main__":
    unittest.main()
