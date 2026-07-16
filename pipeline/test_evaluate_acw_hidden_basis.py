import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pipeline.acw_hidden_basis_training import (
    canonical_json_bytes,
    file_sha256,
    initialized_model_for_arm,
    load_public_training_data,
    write_checkpoint,
)
from pipeline.evaluate_acw_hidden_basis import (
    EVALUATION_PROTOCOL,
    _apply_event,
    _word_pairs,
    compiled_sparse_report,
    evaluate_checkpoint,
    load_oracle_split,
    predict_public_queries,
)
from pipeline.generate_acw_hidden_basis import (
    development_seed_material,
    generate_dataset,
)


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
        predictions, packets = predict_public_queries(
            model, "acw", split.data, batch_size=4
        )
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

    def test_evaluator_replay_is_byte_stable(self):
        kwargs = dict(
            depths=(8,),
            new_reader_updates=1,
            batch_size=4,
            event_word_limit=4,
            allow_unbound=True,
        )
        first = evaluate_checkpoint(self.checkpoint, self.root, **kwargs)
        second = evaluate_checkpoint(self.checkpoint, self.root, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            canonical_json_bytes(first),
            canonical_json_bytes(second),
        )

    def test_compiled_sparse_realization_is_exact(self):
        report = compiled_sparse_report(self.root, (8,))
        self.assertEqual(report["depths"]["8"]["state_exactness"], 1.0)
        self.assertEqual(report["depths"]["8"]["transition_state_exactness"], 1.0)
        self.assertGreater(report["external_event_updates"], 0)

    def test_compiled_sparse_replays_events_instead_of_reading_final_state(self):
        mutated = Path(self.temporary.name) / "mutated_events"
        shutil.copytree(self.root, mutated)
        relative = "oracle/evaluation/depth_008/event_ids.npy"
        path = mutated / relative
        with path.open("rb") as handle:
            event_ids = np.load(handle, allow_pickle=False)
        event_ids[:] = 0
        with path.open("wb") as handle:
            np.save(handle, event_ids, allow_pickle=False)
        manifest_path = mutated / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["arrays"][relative] = {
            "bytes": path.stat().st_size,
            "dtype": str(event_ids.dtype),
            "shape": list(event_ids.shape),
            "sha256": file_sha256(path),
        }
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        report = compiled_sparse_report(mutated, (8,))
        self.assertLess(report["depths"]["8"]["transition_state_exactness"], 1.0)

    def test_event_word_witnesses_are_first_lexicographic_pairs(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        path = self.root / "oracle/domain/events.npy"
        with path.open("rb") as handle:
            events = np.load(handle, allow_pickle=False)
        self.assertEqual(
            file_sha256(path), manifest["arrays"]["oracle/domain/events.npy"]["sha256"]
        )
        state = (1, 2, 3)
        equivalent, non_equivalent = _word_pairs(state, events)
        words = [(a, b) for a in range(len(events)) for b in range(len(events))]
        outputs = {
            word: _apply_event(_apply_event(state, events[word[0]]), events[word[1]])
            for word in words
        }
        first_equal = None
        first_unequal = None
        for index, first in enumerate(words):
            for second in words[index + 1 :]:
                if first_equal is None and outputs[first] == outputs[second]:
                    first_equal = (first, second)
                if first_unequal is None and outputs[first] != outputs[second]:
                    first_unequal = (first, second)
                if first_equal is not None and first_unequal is not None:
                    break
            if first_equal is not None and first_unequal is not None:
                break
        self.assertEqual(equivalent[:2], first_equal)
        self.assertEqual(non_equivalent[:2], first_unequal)


if __name__ == "__main__":
    unittest.main()
