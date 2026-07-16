import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.acw_hidden_basis_training import (
    initialized_model_for_arm,
    load_public_training_data,
    write_checkpoint,
)
from pipeline.evaluate_acw_hidden_basis import (
    EVALUATION_PROTOCOL,
    compiled_sparse_report,
    evaluate_checkpoint,
    load_oracle_split,
    predict_public_queries,
)
from pipeline.generate_acw_hidden_basis import development_seed_material, generate_dataset


class ACWEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "dataset"
        generate_dataset(
            cls.root,
            development_seed_material(2026071601),
            seed_identity={"kind": "development", "seed": 2026071601},
            train_count=16,
            adaptation_count=8,
            evaluation_count=8,
            evaluation_depths=(8,),
        )
        cls.checkpoint = Path(cls.temporary.name) / "checkpoint.pt"
        data = load_public_training_data(cls.root, reject_oracle=False)
        model = initialized_model_for_arm("acw", 17)
        write_checkpoint(
            cls.checkpoint,
            model,
            arm="acw",
            seed=17,
            data=data,
            curriculum_sha256="0" * 64,
            training_report={"test_only": True},
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_split_loader_and_literal_predictions(self):
        split = load_oracle_split(self.root, "oracle/evaluation/depth_008")
        model = initialized_model_for_arm("acw", 17)
        predictions, packets = predict_public_queries(model, "acw", split.data, batch_size=4)
        self.assertEqual(predictions.shape, (8, 24))
        self.assertEqual(packets.shape, (8, 3))
        self.assertEqual(packets.dtype, torch.uint8)

    def test_untrained_checkpoint_runs_complete_causal_schema(self):
        report = evaluate_checkpoint(
            self.checkpoint,
            self.root,
            depths=(8,),
            new_reader_updates=1,
            batch_size=4,
            event_word_limit=4,
            allow_unbound=True,
        )
        self.assertEqual(report["protocol"], EVALUATION_PROTOCOL)
        self.assertIn("packet_interventions", report)
        self.assertIn("event_words", report)
        self.assertEqual(report["write_legality"]["illegal_writes"], 0)

    def test_compiled_sparse_realization_is_exact(self):
        report = compiled_sparse_report(self.root, (8,))
        self.assertEqual(report["depths"]["8"]["state_exactness"], 1.0)
        self.assertGreater(report["external_event_updates"], 0)


if __name__ == "__main__":
    unittest.main()
