import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pipeline.freeze_acw_curriculum as freezer
from pipeline.acw_hidden_basis_training import (
    file_sha256,
    load_curriculum,
    load_public_training_data,
)
from pipeline.freeze_acw_curriculum import (
    build_trainer_bundle,
    build_uniform_schedule,
    execute_pilot_replay,
    freeze_pilot_replays,
    load_oracle_truth,
    run_pilot,
    select_refinement_round,
    validate_query_schedule,
    verify_registered_dataset,
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

    def _verify_small_registered_dataset(
        self,
        root: Path,
        *,
        allowed_kinds: set[str],
    ) -> dict:
        with patch.multiple(
            freezer,
            GENERATOR_TRAIN_HISTORIES=16,
            GENERATOR_ADAPTATION_HISTORIES=8,
            GENERATOR_EVALUATION_HISTORIES=8,
            GENERATOR_EVALUATION_DEPTHS=(8,),
        ):
            return verify_registered_dataset(root, allowed_kinds=allowed_kinds)

    def _copy_dataset(self, name: str) -> Path:
        destination = Path(self.temporary.name) / name
        shutil.copytree(self.root, destination)
        return destination

    @staticmethod
    def _rewrite_manifest(root: Path, mutate) -> None:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        mutate(manifest)
        manifest.pop("payload_sha256", None)
        manifest["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(manifest)
        ).hexdigest()
        path.write_bytes(freezer.canonical_json_bytes(manifest) + b"\n")

    @classmethod
    def _rewrite_array(cls, root: Path, relative: str, array: np.ndarray) -> None:
        path = root / relative
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)

        def mutate(manifest):
            manifest["arrays"][relative] = {
                "bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": file_sha256(path),
            }

        cls._rewrite_manifest(root, mutate)

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
            packets,
            states,
            answers,
            rows,
            round_index=1,
            max_groups=4,
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
            schedule,
            data.histories,
            refinement_rounds=2,
            canonical=False,
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
        import pipeline.adjudicate_acw_hidden_basis as adjudicator
        import pipeline.acw_hidden_basis_training as trainer

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
                __import__("json").dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in schedule
            )
        )
        out = Path(self.temporary.name) / "bundle"
        manifest = build_trainer_bundle(self.root, schedule_path, out, canonical=False)
        self.assertFalse((out / "oracle").exists())
        self.assertEqual(manifest["oracle_paths_exported"], 0)
        self.assertEqual(manifest["protocol"], "R12-ACW-TRAINER-BUNDLE-v4")
        self.assertEqual(manifest["protocol"], trainer.BUNDLE_PROTOCOL)
        self.assertEqual(manifest["protocol"], adjudicator.TRAINER_BUNDLE_PROTOCOL)
        self.assertEqual(set(manifest), adjudicator.BUNDLE_KEYS)
        self.assertEqual(set(manifest), trainer.BUNDLE_KEYS)
        self.assertIsNone(manifest["pilot_artifacts"])
        self.assertEqual(
            manifest["files"]["curriculum.jsonl"]["sha256"],
            file_sha256(out / "curriculum.jsonl"),
        )
        curriculum = load_curriculum(out / "curriculum.jsonl")
        curriculum.validate(data.histories, canonical=False)
        for index in range(len(curriculum.history_ids)):
            history_id = int(curriculum.history_ids[index])
            query_id = int(curriculum.query_ids[index])
            self.assertEqual(
                int(curriculum.answers[index]), int(answers[history_id, query_id])
            )

    def test_registered_pilot_dataset_passes_full_deterministic_replay(self):
        record = self._verify_small_registered_dataset(
            self.root,
            allowed_kinds={"pilot"},
        )
        self.assertEqual(record["protocol"], "R12-ACW-DATA-REPLAY-v1")
        self.assertEqual(record["seed_identity"], {"kind": "pilot", "seed": 2026071600})
        self.assertGreater(record["public_arrays_verified"], 0)
        self.assertGreater(record["oracle_arrays_verified"], 0)
        self.assertEqual(
            record["source_manifest_payload_sha256"],
            record["regenerated_manifest_payload_sha256"],
        )

    def test_relabelled_pilot_as_development_fails_replay(self):
        root = self._copy_dataset("relabeled_development")
        material = development_seed_material(2026071601)

        def relabel(manifest):
            manifest["seed_identity"] = {"kind": "development", "seed": 2026071601}
            manifest["seed_fingerprint"] = hashlib.sha256(material).hexdigest()

        self._rewrite_manifest(root, relabel)
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            self._verify_small_registered_dataset(
                root,
                allowed_kinds={"development"},
            )

    def test_self_hashed_public_truth_leakage_fails_replay(self):
        root = self._copy_dataset("public_truth_leak")
        with (root / "public/train/source_features.npy").open("rb") as handle:
            source_features = np.load(handle, allow_pickle=False)
        with (root / "oracle/train/final_states.npy").open("rb") as handle:
            final_states = np.load(handle, allow_pickle=False)
        leaked = source_features.copy()
        leaked[:, :3] = final_states.astype(np.float32)
        self._rewrite_array(root, "public/train/source_features.npy", leaked)
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            self._verify_small_registered_dataset(root, allowed_kinds={"pilot"})

    def test_self_hashed_oracle_mutation_fails_replay(self):
        root = self._copy_dataset("oracle_mutation")
        with (root / "oracle/train/final_states.npy").open("rb") as handle:
            final_states = np.load(handle, allow_pickle=False)
        corrupted = final_states.copy()
        corrupted[0, 0] = (int(corrupted[0, 0]) + 1) % 17
        self._rewrite_array(root, "oracle/train/final_states.npy", corrupted)
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            self._verify_small_registered_dataset(root, allowed_kinds={"pilot"})

    def test_canonical_pilot_rejects_every_hyperparameter_override(self):
        overrides = {
            "seed": 1,
            "refinement_rounds": 1,
            "updates_per_round": 1,
            "final_updates": 1,
            "batch_size": 1,
            "max_groups": 1,
        }
        for name, value in overrides.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError,
                    "hyperparameters are frozen",
                ),
            ):
                run_pilot(self.root, canonical=True, **{name: value})

    def test_two_independent_replays_freeze_only_when_byte_identical(self):
        first = Path(self.temporary.name) / "replay_a"
        second = Path(self.temporary.name) / "replay_b"
        frozen = Path(self.temporary.name) / "frozen"
        # This test deliberately creates non-Slurm receipts. Do not let a real
        # scheduler allocation make the negative control environment-dependent.
        with patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
            execute_pilot_replay(
                self.root,
                first,
                replay_id="a",
                canonical=False,
                pilot_kwargs={
                    "refinement_rounds": 2,
                    "updates_per_round": 1,
                    "final_updates": 1,
                    "batch_size": 8,
                    "max_groups": 4,
                },
            )
            execute_pilot_replay(
                self.root,
                second,
                replay_id="b",
                canonical=False,
                pilot_kwargs={
                    "refinement_rounds": 2,
                    "updates_per_round": 1,
                    "final_updates": 1,
                    "batch_size": 8,
                    "max_groups": 4,
                },
            )
        comparison = freeze_pilot_replays(
            first,
            second,
            frozen,
            dataset_root=self.root,
            canonical=False,
        )
        self.assertTrue(comparison["reports_byte_identical"])
        self.assertTrue(comparison["schedules_byte_identical"])
        self.assertEqual(
            (first / "report.json").read_bytes(),
            (second / "report.json").read_bytes(),
        )
        first_execution = json.loads((first / "execution.json").read_text())
        second_execution = json.loads((second / "execution.json").read_text())
        self.assertNotEqual(
            first_execution["execution_nonce"],
            second_execution["execution_nonce"],
        )
        with self.assertRaisesRegex(ValueError, "Slurm execution evidence"):
            freezer._validate_execution(
                first,
                json.loads((first / "report.json").read_text()),
                replay_id="a",
                canonical=True,
            )
        self.assertTrue((frozen / "replay_comparison.json").is_file())

    def test_v4_structure_is_exercised_but_canonical_use_requires_git_anchor(self):
        import pipeline.acw_hidden_basis_training as trainer

        first = Path(self.temporary.name) / "bundle_replay_a"
        second = Path(self.temporary.name) / "bundle_replay_b"
        frozen = Path(self.temporary.name) / "bundle_frozen_pilot"
        pilot_kwargs = {
            "refinement_rounds": 2,
            "updates_per_round": 1,
            "final_updates": 1,
            "batch_size": 8,
            "max_groups": 4,
        }
        execute_pilot_replay(
            self.root,
            first,
            replay_id="a",
            canonical=False,
            pilot_kwargs=pilot_kwargs,
        )
        execute_pilot_replay(
            self.root,
            second,
            replay_id="b",
            canonical=False,
            pilot_kwargs=pilot_kwargs,
        )
        freeze_pilot_replays(
            first,
            second,
            frozen,
            dataset_root=self.root,
            canonical=False,
        )
        pilot_identity = {
            "scientific_commit": "a" * 40,
            "scientific_path_sha256": {"test-scientific-path": "b" * 64},
        }
        report_path = frozen / "report.json"
        report_path.chmod(0o644)
        pilot_report = json.loads(report_path.read_text())
        pilot_report["scientific_identity"] = pilot_identity
        pilot_report.pop("payload_sha256")
        pilot_report["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(pilot_report)
        ).hexdigest()
        report_path.write_bytes(freezer.canonical_json_bytes(pilot_report) + b"\n")
        comparison_path = frozen / "replay_comparison.json"
        comparison_path.chmod(0o644)
        comparison = json.loads(comparison_path.read_text())
        comparison["scientific_identity"] = pilot_identity
        comparison["common_files"]["report.json"] = {
            "bytes": report_path.stat().st_size,
            "sha256": file_sha256(report_path),
        }
        comparison["independent_recomputation_sha256"]["report.json"] = file_sha256(
            report_path
        )
        comparison.pop("payload_sha256")
        comparison["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(comparison)
        ).hexdigest()
        comparison_path.write_bytes(freezer.canonical_json_bytes(comparison) + b"\n")

        dataset = Path(self.temporary.name) / "bundle_development_dataset"
        generate_dataset(
            dataset,
            development_seed_material(2026071601),
            seed_identity={"kind": "development", "seed": 2026071601},
            train_count=16,
            adaptation_count=8,
            evaluation_count=8,
            evaluation_depths=(8,),
        )
        source_manifest = json.loads((dataset / "manifest.json").read_text())
        replay = {
            "protocol": "R12-ACW-DATA-REPLAY-v1",
            "seed_identity": source_manifest["seed_identity"],
            "seed_fingerprint": source_manifest["seed_fingerprint"],
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "regenerated_manifest_payload_sha256": source_manifest["payload_sha256"],
            "array_registry_sha256": hashlib.sha256(b"test-registry").hexdigest(),
            "arrays_verified": len(source_manifest["arrays"]),
            "public_arrays_verified": 7,
            "oracle_arrays_verified": len(source_manifest["arrays"]) - 7,
        }
        bundle = Path(self.temporary.name) / "canonical_v4_bundle"
        with self.assertRaisesRegex(RuntimeError, "external anchor"):
            build_trainer_bundle(
                dataset,
                frozen / "cgb_schedule.jsonl",
                bundle,
                canonical=True,
                pilot_report_path=frozen / "report.json",
            )
        with (
            patch.multiple(
                freezer,
                CANONICAL_HISTORIES=16,
                CANONICAL_LABELS=64,
                REFINEMENT_ROUNDS=2,
            ),
            patch.object(freezer, "load_pilot_report", return_value=pilot_report),
            patch.object(freezer, "verify_registered_dataset", return_value=replay),
            patch.object(freezer, "_require_committed_pilot_anchor", return_value=None),
        ):
            manifest = build_trainer_bundle(
                dataset,
                frozen / "cgb_schedule.jsonl",
                bundle,
                canonical=True,
                pilot_report_path=frozen / "report.json",
            )
        with self.assertRaisesRegex(RuntimeError, "external anchor"):
            trainer.validate_trainer_bundle_contract(bundle, manifest)
        summary = trainer._validate_unanchored_trainer_bundle_structure(
            bundle, manifest
        )
        self.assertEqual(summary["pilot_artifacts_opened"], 4)
        self.assertEqual(
            summary["query_schedule_sha256"],
            file_sha256(frozen / "cgb_schedule.jsonl"),
        )

        partial = dict(manifest)
        partial.pop("pilot_artifacts")
        with self.assertRaisesRegex(ValueError, "wrong exact schema"):
            trainer._validate_unanchored_trainer_bundle_structure(bundle, partial)

        forked = deepcopy(manifest)
        forked["query_schedule_sha256"] = "0" * 64
        payload = dict(forked)
        payload.pop("payload_sha256")
        forked["payload_sha256"] = hashlib.sha256(
            trainer.canonical_json_bytes(payload)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "curriculum-derived"):
            trainer._validate_unanchored_trainer_bundle_structure(bundle, forked)

        extra_array = deepcopy(manifest)
        extra_array["arrays"]["public/unregistered.npy"] = deepcopy(
            next(iter(extra_array["arrays"].values()))
        )
        payload = dict(extra_array)
        payload.pop("payload_sha256")
        extra_array["payload_sha256"] = hashlib.sha256(
            trainer.canonical_json_bytes(payload)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "array registry differs"):
            trainer._validate_unanchored_trainer_bundle_structure(bundle, extra_array)

    def test_replay_freeze_rejects_a_shopped_nonidentical_report(self):
        first = Path(self.temporary.name) / "mismatch_a"
        second = Path(self.temporary.name) / "mismatch_b"
        execute_pilot_replay(
            self.root,
            first,
            replay_id="a",
            canonical=False,
            pilot_kwargs={
                "refinement_rounds": 2,
                "updates_per_round": 1,
                "final_updates": 1,
                "batch_size": 8,
                "max_groups": 4,
            },
        )
        execute_pilot_replay(
            self.root,
            second,
            replay_id="b",
            canonical=False,
            pilot_kwargs={
                "seed": 2026071699,
                "refinement_rounds": 2,
                "updates_per_round": 1,
                "final_updates": 1,
                "batch_size": 8,
                "max_groups": 4,
            },
        )
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            freeze_pilot_replays(
                first,
                second,
                Path(self.temporary.name) / "mismatch_frozen",
                dataset_root=self.root,
                canonical=False,
            )

    def test_replay_freeze_rejects_two_identical_fabricated_executions(self):
        real = run_pilot(
            self.root,
            refinement_rounds=2,
            updates_per_round=1,
            final_updates=1,
            batch_size=8,
            max_groups=4,
            canonical=False,
        )
        fabricated = deepcopy(real)
        fabricated[2]["model_tensor_sha256"] = "0" * 64
        fabricated[2]["final_loss_last"] = 0.0
        first = Path(self.temporary.name) / "fabricated_a"
        second = Path(self.temporary.name) / "fabricated_b"
        kwargs = {
            "refinement_rounds": 2,
            "updates_per_round": 1,
            "final_updates": 1,
            "batch_size": 8,
            "max_groups": 4,
        }
        with patch.object(freezer, "run_pilot", return_value=fabricated):
            execute_pilot_replay(
                self.root,
                first,
                replay_id="a",
                canonical=False,
                pilot_kwargs=kwargs,
            )
            execute_pilot_replay(
                self.root,
                second,
                replay_id="b",
                canonical=False,
                pilot_kwargs=kwargs,
            )
        with self.assertRaisesRegex(ValueError, "independent recomputation"):
            freeze_pilot_replays(
                first,
                second,
                Path(self.temporary.name) / "fabricated_frozen",
                dataset_root=self.root,
                canonical=False,
            )

    def test_canonical_freeze_cannot_accept_caller_supplied_replays(self):
        with self.assertRaisesRegex(RuntimeError, "parent-owned child processes"):
            freeze_pilot_replays(
                Path(self.temporary.name) / "caller_a",
                Path(self.temporary.name) / "caller_b",
                Path(self.temporary.name) / "caller_out",
                dataset_root=self.root,
                canonical=True,
            )

    def test_live_child_binding_uses_parent_observed_processes(self):
        roots = (
            Path(self.temporary.name) / "live_child_a",
            Path(self.temporary.name) / "live_child_b",
        )
        for root in roots:
            root.mkdir()
            (root / "execution.json").write_text("{}\n")
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(2)
        ]
        now = time.time_ns()
        executions = tuple(
            {
                "process_id": process.pid,
                "finished_time_ns": now - 1,
                "slurm_job_id": "123",
            }
            for process in processes
        )
        children = [
            {
                "replay_id": replay_id,
                "command": process.args,
                "process": process,
                "release_fd": -1,
                "started_time_ns": now - 3,
                "ready_time_ns": now,
                "execution_sha256": file_sha256(root / "execution.json"),
            }
            for replay_id, process, root in zip(
                ("a", "b"), processes, roots, strict=True
            )
        ]
        try:
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
                records = freezer._validate_live_child_processes(
                    children,
                    executions,
                    roots,
                )
            self.assertEqual(
                [record["observed_process_id"] for record in records],
                [process.pid for process in processes],
            )
            executions[0]["process_id"] += 1
            with (
                patch.dict(os.environ, {"SLURM_JOB_ID": "123"}),
                self.assertRaisesRegex(RuntimeError, "parent-observed"),
            ):
                freezer._validate_live_child_processes(
                    children,
                    executions,
                    roots,
                )
        finally:
            for process in processes:
                process.terminate()
                process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()

    def test_canonical_execution_reconciles_live_slurm_allocation(self):
        snapshot = {
            "command": ["scontrol", "show", "job", "-o", "123"],
            "stdout": "JobId=123 JobState=RUNNING NumCPUs=4 NodeList=n1",
            "allocation": {
                "job_id": "123",
                "job_state": "RUNNING",
                "num_cpus": 4,
                "node_list": "n1",
            },
        }
        snapshot["stdout_sha256"] = hashlib.sha256(
            snapshot["stdout"].encode("utf-8")
        ).hexdigest()
        out = Path(self.temporary.name) / "live_slurm_replay"
        kwargs = {
            "refinement_rounds": 2,
            "updates_per_round": 1,
            "final_updates": 1,
            "batch_size": 8,
            "max_groups": 4,
        }
        environment = {
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_JOB_NODELIST": "n1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(freezer, "_slurm_snapshot", return_value=snapshot),
        ):
            execute_pilot_replay(
                self.root,
                out,
                replay_id="a",
                canonical=False,
                pilot_kwargs=kwargs,
            )
            report = json.loads((out / "report.json").read_text())
            execution = freezer._validate_execution(
                out,
                report,
                replay_id="a",
                canonical=True,
                require_live_scheduler=True,
            )
        self.assertEqual(execution["slurm_job_id"], "123")

        stale = deepcopy(snapshot)
        stale["allocation"] = dict(snapshot["allocation"], job_id="999")
        with (
            patch.dict(os.environ, environment),
            patch.object(freezer, "_slurm_snapshot", return_value=stale),
            self.assertRaisesRegex(ValueError, "live Slurm allocation"),
        ):
            freezer._validate_execution(
                out,
                report,
                replay_id="a",
                canonical=True,
                require_live_scheduler=True,
            )

    def test_canonical_runner_owns_both_children_and_consumer_reopen(self):
        root = Path(self.temporary.name) / "canonical_runner"
        paths = {
            freezer.CANONICAL_PILOT_DATASET: root / "dataset",
            freezer.CANONICAL_PILOT_REPLAY_A: root / "replay_a",
            freezer.CANONICAL_PILOT_REPLAY_B: root / "replay_b",
            freezer.CANONICAL_PILOT_OUTPUT: root / "out",
        }
        identity = {
            "scientific_commit": "a" * 40,
            "scientific_path_sha256": {"test": "b" * 64},
        }
        children = [
            {"replay_id": "a", "process": object()},
            {"replay_id": "b", "process": object()},
        ]

        def canonical_path(relative):
            return paths[relative]

        with (
            patch.object(freezer, "_canonical_path", side_effect=canonical_path),
            patch.object(freezer, "scientific_identity", return_value=identity),
            patch.object(freezer, "_regenerate_registered_dataset"),
            patch.object(freezer, "verify_registered_dataset"),
            patch.object(freezer, "_launch_held_replay", side_effect=children),
            patch.object(freezer, "_wait_for_held_replay") as wait_replay,
            patch.object(
                freezer,
                "freeze_pilot_replays",
                return_value={"payload_sha256": "c" * 64},
            ) as freeze_replays,
            patch.object(freezer, "load_pilot_report") as reopen,
            patch.object(freezer, "_cleanup_held_replays") as cleanup,
        ):
            comparison = freezer.run_canonical_pilot()
        self.assertEqual(comparison["payload_sha256"], "c" * 64)
        self.assertEqual(wait_replay.call_count, 2)
        self.assertEqual(
            freeze_replays.call_args.kwargs["canonical_children"], children
        )
        reopen.assert_called_once_with(
            paths[freezer.CANONICAL_PILOT_OUTPUT] / "report.json"
        )
        cleanup.assert_called_once_with(children)

    def test_canonical_scored_bundle_rejects_pilot_domain_and_unbound_schedule(self):
        data = load_public_training_data(self.root, reject_oracle=False)
        initial = [
            {"history_id": history_id, "query_id": int(query_id), "round": 0}
            for history_id in range(data.histories)
            for query_id in data.initial_queries[history_id]
        ]
        schedule = build_uniform_schedule(initial, data.histories, refinement_rounds=2)
        schedule_path = Path(self.temporary.name) / "unbound_schedule.jsonl"
        schedule_path.write_text(
            "".join(
                __import__("json").dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in schedule
            )
        )
        with (
            patch.object(freezer, "_require_committed_pilot_anchor", return_value=None),
            self.assertRaisesRegex(ValueError, "may not use a pilot"),
        ):
            build_trainer_bundle(
                self.root,
                schedule_path,
                Path(self.temporary.name) / "forbidden_canonical_bundle",
                canonical=True,
            )

    def test_stokes_job_executes_only_the_public_pilot_phase(self):
        source = Path("pipeline/jobs/run_acw_pilot_stokes.sbatch").read_text()
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith('"$PY" -m pipeline.')
        ]
        self.assertEqual(
            commands,
            [
                '"$PY" -m pipeline.freeze_acw_curriculum pilot-run',
                '"$PY" -m pipeline.freeze_acw_curriculum verify-pilot',
            ],
        )
        self.assertNotIn(" bundle", source)
        self.assertNotIn('"$PY" -m pipeline.acw_hidden_basis_training', source)
        self.assertNotIn('"$PY" -m pipeline.adjudicate_acw_hidden_basis', source)


if __name__ == "__main__":
    unittest.main()
