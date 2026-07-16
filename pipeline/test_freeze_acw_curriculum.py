import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.acw_hidden_basis_training import (
    file_sha256,
    load_curriculum,
    load_public_training_data,
)
from pipeline.freeze_acw_curriculum import (
    build_trainer_bundle,
    build_uniform_schedule,
    load_oracle_truth,
    run_pilot,
    select_refinement_round,
    validate_query_schedule,
)
from pipeline.generate_acw_hidden_basis import (
    development_seed_material,
    generate_dataset,
)


class FreezeCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "dataset"
        generate_dataset(
            cls.root,
            development_seed_material(2026071600),
            seed_identity={"kind": "pilot", "seed": 2026071600},
            train_count=16,
            adaptation_count=8,
            evaluation_count=8,
            evaluation_depths=(8,),
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_selection_uses_a_common_unused_separator(self):
        packets = np.zeros((4, 3), dtype=np.uint8)
        states = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int8)
        answers = np.tile(np.arange(24, dtype=np.int8) % 17, (4, 1))
        answers[:, 3] = np.arange(4)
        rows = [
            {"history_id": history_id, "query_id": query, "round": 0}
            for history_id in range(4)
            for query in (0, 1)
        ]
        additions, report = select_refinement_round(
            packets, states, answers, rows, round_index=1, max_groups=4,
        )
        self.assertEqual({row["query_id"] for row in additions}, {3})
        self.assertEqual(report["selected_witnesses"], 1)
        self.assertEqual(report["filler_histories"], 0)
        self.assertEqual(report["candidate_evaluations"], 22)

    def test_uniform_schedule_has_exact_multiplicity(self):
        data = load_public_training_data(self.root, reject_oracle=False)
        initial = [
            {"history_id": history_id, "query_id": int(query_id), "round": 0}
            for history_id in range(data.histories)
            for query_id in data.initial_queries[history_id]
        ]
        schedule = build_uniform_schedule(initial, data.histories, refinement_rounds=2)
        validate_query_schedule(
            schedule, data.histories, refinement_rounds=2, canonical=False,
        )
        self.assertEqual(len(schedule), 64)

    def test_small_pilot_is_deterministic(self):
        first = run_pilot(
            self.root,
            refinement_rounds=2,
            updates_per_round=1,
            final_updates=1,
            batch_size=8,
            max_groups=4,
            canonical=False,
        )
        second = run_pilot(
            self.root,
            refinement_rounds=2,
            updates_per_round=1,
            final_updates=1,
            batch_size=8,
            max_groups=4,
            canonical=False,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[2]["total_updates"], 4)

    def test_public_bundle_strips_oracle_and_binds_curriculum(self):
        data = load_public_training_data(self.root, reject_oracle=False)
        _, answers, _ = load_oracle_truth(self.root)
        initial = [
            {"history_id": history_id, "query_id": int(query_id), "round": 0}
            for history_id in range(data.histories)
            for query_id in data.initial_queries[history_id]
        ]
        schedule = build_uniform_schedule(initial, data.histories, refinement_rounds=2)
        schedule_path = Path(self.temporary.name) / "schedule.jsonl"
        schedule_path.write_text(
            "".join(
                __import__("json").dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in schedule
            )
        )
        out = Path(self.temporary.name) / "bundle"
        manifest = build_trainer_bundle(self.root, schedule_path, out, canonical=False)
        self.assertFalse((out / "oracle").exists())
        self.assertEqual(manifest["oracle_paths_exported"], 0)
        loaded = load_public_training_data(out, reject_oracle=True)
        self.assertEqual(loaded.bound_curriculum_sha256, file_sha256(out / "curriculum.jsonl"))
        curriculum = load_curriculum(out / "curriculum.jsonl")
        curriculum.validate(data.histories, canonical=False)
        for index in range(len(curriculum.history_ids)):
            history_id = int(curriculum.history_ids[index])
            query_id = int(curriculum.query_ids[index])
            self.assertEqual(int(curriculum.answers[index]), int(answers[history_id, query_id]))


if __name__ == "__main__":
    unittest.main()
