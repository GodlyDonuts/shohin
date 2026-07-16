import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.acw_hidden_basis_training import (
    ACW_SCIENTIFIC_PATHS,
    ARM_IDS,
    CONFIRMATION_COMMITMENTS,
    EXPECTED_PARAMETERS,
    Curriculum,
    direct_state_forward,
    forward_logits,
    expected_optimizer_seed,
    initialized_model_for_arm,
    initial_curriculum,
    load_public_training_data,
    load_direct_state_truth,
    model_for_arm,
    profile_answer_step_resources,
    profile_inference_resources,
    recurrent_state,
    scientific_identity,
    train_model,
    train_direct_state_model,
)
from pipeline.generate_acw_hidden_basis import (
    ACW_SCIENTIFIC_PATHS as GENERATOR_SCIENTIFIC_PATHS,
    development_seed_material,
    generate_dataset,
)


class PublicTrainerTests(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.data = load_public_training_data(self.root, reject_oracle=False)
        self.curriculum = initial_curriculum(self.data)

    def test_trainer_source_never_imports_generator(self):
        import pipeline.acw_hidden_basis_training as trainer

        source = inspect.getsource(trainer)
        self.assertNotIn("import pipeline.generate_acw_hidden_basis", source)
        self.assertNotIn("from pipeline.generate_acw_hidden_basis", source)
        self.assertEqual(set(ACW_SCIENTIFIC_PATHS), set(GENERATOR_SCIENTIFIC_PATHS))

    def test_scientific_identity_rejects_blob_drift_even_if_status_claims_clean(self):
        import pipeline.acw_hidden_basis_training as trainer

        responses = [
            SimpleNamespace(stdout="a" * 40),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout=b"different committed bytes"),
        ]
        with (
            patch.object(
                trainer,
                "ACW_SCIENTIFIC_PATHS",
                ("pipeline/testdata/acw_nist_beacon_snapshot.json",),
            ),
            patch.object(trainer.subprocess, "run", side_effect=responses),
            self.assertRaisesRegex(RuntimeError, "differs from HEAD"),
        ):
            scientific_identity(require_clean=True)

    def test_scientific_identity_rejects_unpushed_head(self):
        import pipeline.acw_hidden_basis_training as trainer

        relative = "pipeline/testdata/acw_nist_beacon_snapshot.json"
        payload = Path(relative).read_bytes()
        responses = [
            SimpleNamespace(stdout="a" * 40),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout=payload),
            SimpleNamespace(stdout=f"{'b' * 40}\trefs/heads/main\n"),
        ]
        with (
            patch.object(trainer, "ACW_SCIENTIFIC_PATHS", (relative,)),
            patch.object(trainer.subprocess, "run", side_effect=responses),
            self.assertRaisesRegex(RuntimeError, "must equal pushed origin/main"),
        ):
            scientific_identity(require_clean=True)

    def test_canonical_loader_rejects_visible_oracle(self):
        with self.assertRaises(RuntimeError):
            load_public_training_data(self.root, reject_oracle=True)

    def test_initial_curriculum_is_two_labels_per_history(self):
        self.curriculum.validate(self.data.histories, canonical=False)
        counts = torch.bincount(
            self.curriculum.history_ids,
            minlength=self.data.histories,
        )
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))

    def test_every_arm_executes_and_matches_parameters(self):
        histories = torch.tensor([0, 1, 2, 3])
        queries = torch.tensor([0, 1, 2, 3])
        for arm in ARM_IDS:
            model = model_for_arm(arm)
            self.assertEqual(
                sum(p.numel() for p in model.parameters()), EXPECTED_PARAMETERS[arm]
            )
            logits = forward_logits(
                model,
                arm,
                self.data,
                histories,
                queries,
                training=False,
                literal_symbols=arm
                in {"acw", "dense_categorical", "packet_token_transformer"},
            )
            self.assertEqual(logits.shape, (4, 17))

    def test_seeded_model_initialization_is_byte_stable(self):
        first = initialized_model_for_arm("acw", 123)
        second = initialized_model_for_arm("acw", 123)
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]))
        third = initialized_model_for_arm("acw", 124)
        self.assertTrue(
            any(
                not torch.equal(tensor, third.state_dict()[name])
                for name, tensor in first.state_dict().items()
            )
        )

    def test_optimizer_seed_is_bound_to_domain_identity(self):
        self.assertEqual(
            expected_optimizer_seed({"kind": "development", "seed": 2026071601}),
            2026071601,
        )
        confirmation = expected_optimizer_seed(
            {
                "kind": "confirmation",
                "index": 0,
                "commitment": CONFIRMATION_COMMITMENTS[0],
            }
        )
        self.assertEqual(
            confirmation,
            expected_optimizer_seed(
                {
                    "kind": "confirmation",
                    "index": 0,
                    "commitment": CONFIRMATION_COMMITMENTS[0],
                }
            ),
        )
        with self.assertRaises(ValueError):
            expected_optimizer_seed(
                {
                    "kind": "confirmation",
                    "index": 999,
                    "commitment": CONFIRMATION_COMMITMENTS[0],
                }
            )
        with self.assertRaises(ValueError):
            expected_optimizer_seed({"kind": "unknown"})

    def test_literal_acw_rollout_persists_uint8_only(self):
        model = model_for_arm("acw")
        state = recurrent_state(
            model,
            "acw",
            self.data,
            torch.tensor([0, 1, 2, 3]),
            training=False,
            literal_symbols=True,
        )
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.nelement() * state.element_size(), 12)

    def test_direct_state_diagnostic_executes_with_trajectory_supervision(self):
        truth = load_direct_state_truth(self.root)
        model = initialized_model_for_arm("acw", 11)
        logits, state_loss = direct_state_forward(
            model,
            self.data,
            truth,
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3]),
        )
        self.assertEqual(logits.shape, (4, 17))
        self.assertTrue(torch.isfinite(state_loss))
        report = train_direct_state_model(
            model,
            self.data,
            truth,
            self.curriculum,
            seed=11,
            updates_per_round=1,
            final_updates=1,
            batch_size=8,
            canonical=False,
        )
        self.assertEqual(report["updates"], 14)
        self.assertTrue(torch.isfinite(torch.tensor(report["state_loss_last"])))

    def test_small_training_schedule_is_exact(self):
        model = model_for_arm("acw")
        report = train_model(
            model,
            "acw",
            self.data,
            self.curriculum,
            seed=9,
            updates_per_round=1,
            final_updates=2,
            batch_size=8,
            canonical=False,
        )
        self.assertEqual(report["updates"], 15)
        self.assertEqual(report["labels"], 32)
        self.assertTrue(torch.isfinite(torch.tensor(report["loss_last"])))

    def test_resource_profiles_cover_training_inference_flops_time_and_memory(self):
        model = initialized_model_for_arm("packet_token_transformer", 19)
        training = profile_answer_step_resources(
            model,
            "packet_token_transformer",
            self.data,
            self.curriculum,
            batch_size=8,
        )
        inference = profile_inference_resources(
            model,
            "packet_token_transformer",
            self.data,
            self.curriculum,
            batch_size=8,
        )
        for report in (training, inference):
            self.assertGreater(report["wall_seconds"], 0)
            self.assertGreater(report["profiler_event_count"], 0)
            self.assertTrue(report["operator_inventory_complete"])
            self.assertTrue(report["operator_inventory"])
            self.assertEqual(
                sum(row["calls"] for row in report["operator_inventory"]),
                report["profiler_event_count"],
            )
            self.assertEqual(
                [
                    row["name"]
                    for row in report["operator_inventory"]
                    if row["operator_reported_flops"] == 0
                ],
                report["uncounted_operator_names"],
            )
            self.assertGreater(report["operator_reported_flops"], 0)
            self.assertGreaterEqual(report["largest_operator_allocation_bytes"], 0)
            self.assertGreater(report["process_peak_rss_bytes"], 0)

    def test_curriculum_rejects_duplicate_pair(self):
        bad = Curriculum(
            history_ids=torch.tensor([0, 0]),
            query_ids=torch.tensor([1, 1]),
            answers=torch.tensor([2, 2]),
            rounds=torch.tensor([0, 0]),
        )
        with self.assertRaises(ValueError):
            bad.validate(self.data.histories, canonical=False)

    def test_canonical_round_accounting_starts_with_two_labels(self):
        histories = 4096
        history_ids = []
        query_ids = []
        answers = []
        rounds = []
        for history_id in range(histories):
            for query_id in range(14):
                history_ids.append(history_id)
                query_ids.append(query_id)
                answers.append((history_id + query_id) % 17)
                rounds.append(0 if query_id < 2 else query_id - 1)
        curriculum = Curriculum(
            history_ids=torch.tensor(history_ids),
            query_ids=torch.tensor(query_ids),
            answers=torch.tensor(answers),
            rounds=torch.tensor(rounds),
        )
        curriculum.validate(histories, canonical=True)
        round_counts = torch.bincount(curriculum.rounds, minlength=13)
        self.assertEqual(int(round_counts[0]), 8192)
        self.assertTrue(torch.equal(round_counts[1:], torch.full((12,), 4096)))

        bad_rounds = curriculum.rounds.clone()
        bad_rounds[0] = 1
        bad = Curriculum(
            history_ids=curriculum.history_ids,
            query_ids=curriculum.query_ids,
            answers=curriculum.answers,
            rounds=bad_rounds,
        )
        with self.assertRaises(ValueError):
            bad.validate(histories, canonical=True)


if __name__ == "__main__":
    unittest.main()
