import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.acw_hidden_basis_training import (
    ACW_SCIENTIFIC_PATHS,
    ARM_IDS,
    EXPECTED_PARAMETERS,
    PILOT_SCIENTIFIC_PATHS,
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
        self.assertEqual(set(PILOT_SCIENTIFIC_PATHS), set(GENERATOR_SCIENTIFIC_PATHS))
        self.assertGreater(set(ACW_SCIENTIFIC_PATHS), set(PILOT_SCIENTIFIC_PATHS))

    def test_scientific_identity_rejects_blob_drift_even_if_status_claims_clean(self):
        import pipeline.acw_hidden_basis_training as trainer

        responses = [
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="/definitely/missing/grafts\n"),
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
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="/definitely/missing/grafts\n"),
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

    def test_scientific_identity_uses_the_fingerprinted_git_executable(self):
        import pipeline.acw_hidden_basis_training as trainer

        relative = "pipeline/testdata/acw_nist_beacon_snapshot.json"
        payload = Path(relative).read_bytes()
        responses = [
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout="/definitely/missing/grafts\n"),
            SimpleNamespace(stdout="a" * 40),
            SimpleNamespace(stdout=""),
            SimpleNamespace(stdout=payload),
        ]
        with (
            patch.object(trainer, "ACW_SCIENTIFIC_PATHS", (relative,)),
            patch.object(
                trainer.subprocess, "run", side_effect=responses
            ) as run_command,
        ):
            scientific_identity(require_clean=False)
        self.assertTrue(run_command.call_args_list)
        self.assertTrue(
            all(
                call.args[0][:2] == ["/usr/bin/git", "--no-replace-objects"]
                for call in run_command.call_args_list
            )
        )

    def test_scientific_identity_rejects_real_git_replacement_ref(self):
        import pipeline.acw_hidden_basis_training as trainer

        root = Path(self.temporary.name) / "git_replacement_repo"
        remote = Path(self.temporary.name) / "git_replacement_remote.git"
        pipeline = root / "pipeline"
        pipeline.mkdir(parents=True)
        scientific_file = pipeline / "scientific.py"

        def git(cwd, *args):
            return subprocess.run(
                ["/usr/bin/git", "--no-replace-objects", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )

        git(root, "init", "-b", "main")
        git(root, "config", "user.email", "acw-test@example.invalid")
        git(root, "config", "user.name", "ACW Test")
        scientific_file.write_text("VALUE = 'approved'\n")
        git(root, "add", "pipeline/scientific.py")
        git(root, "commit", "-m", "approved")
        approved = git(root, "rev-parse", "HEAD").stdout.strip()
        git(self.temporary.name, "init", "--bare", str(remote))
        git(root, "remote", "add", "origin", str(remote))
        git(root, "push", "-u", "origin", "main")

        scientific_file.write_text("VALUE = 'substituted'\n")
        git(root, "add", "pipeline/scientific.py")
        git(root, "commit", "-m", "substituted")
        replacement = git(root, "rev-parse", "HEAD").stdout.strip()
        git(root, "reset", "--hard", approved)
        git(root, "replace", approved, replacement)

        with (
            patch.object(trainer, "__file__", str(pipeline / "trainer.py")),
            patch.object(
                trainer,
                "ACW_SCIENTIFIC_PATHS",
                ("pipeline/scientific.py",),
            ),
            self.assertRaisesRegex(RuntimeError, "forbids Git replacements"),
        ):
            scientific_identity(require_clean=True)

    def test_activation_lineage_accepts_only_exact_s_a_e_f_g_chain(self):
        import pipeline.acw_hidden_basis_training as trainer

        root = Path(self.temporary.name) / "activation_lineage_repo"
        remote = Path(self.temporary.name) / "activation_lineage_remote.git"
        root.mkdir()

        def git(cwd, *args):
            return subprocess.run(
                ["/usr/bin/git", "--no-replace-objects", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )

        git(root, "init", "-b", "main")
        git(root, "config", "user.email", "acw-test@example.invalid")
        git(root, "config", "user.name", "ACW Test")
        (root / "scientific.txt").write_text("scientific\n")
        (root / "activation_a.txt").write_text("disabled-a\n")
        (root / "activation_b.txt").write_text("disabled-b\n")
        (root / "custody_a.txt").write_text("disabled-a\n")
        (root / "custody_b.txt").write_text("disabled-b\n")
        (root / "development_a.txt").write_text("disabled-a\n")
        (root / "development_b.txt").write_text("disabled-b\n")
        git(root, "add", ".")
        git(root, "commit", "-m", "S")
        scientific_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        (root / "registry.json").write_text("anchored\n")
        git(root, "add", "registry.json")
        git(root, "commit", "-m", "A")
        anchor_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        (root / "activation_a.txt").write_text("enabled-a\n")
        (root / "activation_b.txt").write_text("enabled-b\n")
        git(root, "add", "activation_a.txt", "activation_b.txt")
        git(root, "commit", "-m", "E")
        activation_commit = git(root, "rev-parse", "HEAD").stdout.strip()
        (root / "custody_a.txt").write_text("enabled-a\n")
        (root / "custody_b.txt").write_text("enabled-b\n")
        git(root, "add", "custody_a.txt", "custody_b.txt")
        git(root, "commit", "-m", "F")
        custody_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        additions = (
            "development_plan.json",
            "pipeline/build_acw_development_manifest.py",
            "pipeline/jobs/run_acw_development_stokes.sbatch",
            "pipeline/jobs/run_acw_terminal_monitor_stokes.sbatch",
            "pipeline/test_build_acw_development_manifest.py",
            "pipeline/test_acw_g_custody.py",
        )
        (root / "development_a.txt").write_text("enabled-a\n")
        (root / "development_b.txt").write_text("enabled-b\n")
        for relative in additions:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative}\n")
        git(root, "add", "development_a.txt", "development_b.txt", *additions)
        git(root, "commit", "-m", "G")
        development_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        offline_template = str(Path(self.temporary.name) / "acw_{commit8}.bundle")
        offline_bundle = Path(offline_template.format(commit8=development_commit[:8]))
        git(root, "bundle", "create", str(offline_bundle), "main")
        git(self.temporary.name, "init", "--bare", str(remote))
        git(root, "remote", "add", "origin", str(remote))
        git(root, "push", "-u", "origin", "main")

        with (
            patch.object(trainer, "PILOT_SCIENTIFIC_COMMIT", scientific_commit),
            patch.object(trainer, "PILOT_ANCHOR_COMMIT", anchor_commit),
            patch.object(trainer, "PILOT_EXECUTION_COMMIT", activation_commit),
            patch.object(trainer, "PILOT_CUSTODY_COMMIT", custody_commit),
            patch.object(trainer, "PILOT_REGISTRY_PATH", "registry.json"),
            patch.object(
                trainer,
                "PILOT_ACTIVATION_ALLOWLIST",
                ("activation_a.txt", "activation_b.txt"),
            ),
            patch.object(
                trainer,
                "PILOT_CUSTODY_ALLOWLIST",
                ("custody_a.txt", "custody_b.txt"),
            ),
            patch.object(
                trainer,
                "PILOT_DEVELOPMENT_ALLOWLIST",
                (
                    "development_a.txt",
                    "development_b.txt",
                    *additions,
                ),
            ),
            patch.object(trainer, "DEVELOPMENT_PLAN_PATH", "development_plan.json"),
            patch.object(trainer, "ACW_SCIENTIFIC_PATHS", ("scientific.txt",)),
            patch.object(trainer, "PILOT_CANONICAL_REMOTE_URL", str(remote)),
            patch.object(trainer, "PILOT_OFFLINE_BUNDLE_TEMPLATE", offline_template),
        ):
            self.assertEqual(
                trainer._require_activation_lineage(root), development_commit
            )

            git(root, "remote", "set-url", "origin", str(remote) + ".unapproved")
            with self.assertRaisesRegex(RuntimeError, "approved publication route"):
                trainer._require_activation_lineage(root)

            git(root, "remote", "set-url", "origin", str(offline_bundle))
            self.assertEqual(
                trainer._require_activation_lineage(root), development_commit
            )
            git(root, "remote", "set-url", "origin", str(remote))

            git(root, "checkout", "-B", "bad-development", custody_commit)
            (root / "development_a.txt").write_text("bad-a\n")
            (root / "development_b.txt").write_text("bad-b\n")
            for relative in additions:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"bad {relative}\n")
            (root / "extra.txt").write_text("not allowed\n")
            git(
                root,
                "add",
                "development_a.txt",
                "development_b.txt",
                *additions,
                "extra.txt",
            )
            git(root, "commit", "-m", "bad G")
            git(root, "push", "--force", "origin", "HEAD:main")
            with self.assertRaisesRegex(RuntimeError, "exact allowlist"):
                trainer._require_activation_lineage(root)

    def test_checkpoint_publication_is_immutable_and_refuses_symlinks(self):
        import pipeline.acw_hidden_basis_training as trainer

        root = Path(self.temporary.name) / "checkpoint_publication"
        root.mkdir()
        checkpoint = root / "candidate.pt"
        record = trainer.write_checkpoint(
            checkpoint,
            model_for_arm("acw"),
            arm="acw",
            seed=7,
            data=self.data,
            curriculum_sha256="a" * 64,
            training_report={"updates": 0},
            scientific_identity_record={"scientific_commit": "b" * 40},
        )
        self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o444)
        self.assertEqual(record["bytes"], checkpoint.stat().st_size)
        self.assertEqual(record["sha256"], trainer.file_sha256(checkpoint))
        with self.assertRaises(FileExistsError):
            trainer.write_checkpoint(
                checkpoint,
                model_for_arm("acw"),
                arm="acw",
                seed=7,
                data=self.data,
                curriculum_sha256="a" * 64,
                training_report={"updates": 0},
            )

        dangling = root / "dangling.pt"
        dangling.symlink_to(root / "missing.pt")
        with self.assertRaises(FileExistsError):
            trainer.write_checkpoint(
                dangling,
                model_for_arm("acw"),
                arm="acw",
                seed=7,
                data=self.data,
                curriculum_sha256="a" * 64,
                training_report={"updates": 0},
            )

        raced = root / "raced.pt"
        real_link = trainer.os.link

        def publish_collision(source, destination, *, follow_symlinks=True):
            Path(destination).write_bytes(b"concurrent-winner")
            return real_link(
                source,
                destination,
                follow_symlinks=follow_symlinks,
            )

        with (
            patch.object(trainer.os, "link", side_effect=publish_collision),
            self.assertRaises(FileExistsError),
        ):
            trainer.write_checkpoint(
                raced,
                model_for_arm("acw"),
                arm="acw",
                seed=7,
                data=self.data,
                curriculum_sha256="a" * 64,
                training_report={"updates": 0},
            )
        self.assertEqual(raced.read_bytes(), b"concurrent-winner")
        self.assertFalse((root / "raced.pt.tmp").exists())

        foreign = root / "foreign.pt"
        foreign_temporary = root / "foreign.pt.tmp"
        foreign_temporary.write_bytes(b"another-writer")
        with self.assertRaises(FileExistsError):
            trainer.write_checkpoint(
                foreign,
                model_for_arm("acw"),
                arm="acw",
                seed=7,
                data=self.data,
                curriculum_sha256="a" * 64,
                training_report={"updates": 0},
            )
        self.assertEqual(foreign_temporary.read_bytes(), b"another-writer")

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
        with self.assertRaises(ValueError):
            expected_optimizer_seed(
                {
                    "kind": "confirmation",
                    "index": 0,
                    "commitment": "f" * 64,
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
