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
from types import SimpleNamespace
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

    @staticmethod
    def _canonical_environment(role: str, job_id: str) -> dict[str, str]:
        expected = freezer.CANONICAL_PILOT_ROLES[role]
        return {
            **freezer.CANONICAL_PILOT_STATIC_ENV,
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NAME": expected["job_name"],
            "SLURM_JOB_NODELIST": expected["node_list"],
            "SLURM_NODELIST": expected["node_list"],
            "SLURM_SUBMIT_DIR": freezer.CANONICAL_PILOT_BASE,
        }

    @staticmethod
    def _canonical_snapshot(role: str, job_id: str) -> dict:
        expected = freezer.CANONICAL_PILOT_ROLES[role]
        stdout_path = f"{expected['stdout_prefix']}{job_id}.out"
        stdout = " ".join(
            (
                f"JobId={job_id}",
                f"JobName={expected['job_name']}",
                "JobState=RUNNING",
                "NumCPUs=4",
                "NumNodes=1",
                f"NodeList={expected['node_list']}",
                f"BatchHost={expected['node_list']}",
                "Partition=normal",
                f"Command={expected['command']}",
                f"WorkDir={freezer.CANONICAL_PILOT_BASE}",
                f"StdOut={stdout_path}",
            )
        )
        return {
            "command": [
                freezer.CANONICAL_PILOT_SCONTROL,
                "show",
                "job",
                "-o",
                job_id,
            ],
            "stdout": stdout,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "allocation": {
                "batch_host": expected["node_list"],
                "command": expected["command"],
                "job_id": job_id,
                "job_name": expected["job_name"],
                "job_state": "RUNNING",
                "node_list": expected["node_list"],
                "num_cpus": 4,
                "num_nodes": 1,
                "partition": "normal",
                "stdout": stdout_path,
                "work_dir": freezer.CANONICAL_PILOT_BASE,
            },
        }

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

    def test_v4_structure_is_exercised_with_a_separate_git_anchor(self):
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
        bundle_sources = {
            "pilot/report.json": "anchor/report.json",
            "pilot/replay_comparison.json": "anchor/replay_comparison.json",
            "pilot/cgb_schedule.jsonl": "anchor/cgb_schedule.jsonl",
            "pilot/uniform_schedule.jsonl": "anchor/uniform_schedule.jsonl",
        }
        bundle_paths = {
            "pilot/report.json": report_path.resolve(),
            "pilot/replay_comparison.json": comparison_path.resolve(),
            "pilot/cgb_schedule.jsonl": (frozen / "cgb_schedule.jsonl").resolve(),
            "pilot/uniform_schedule.jsonl": (
                frozen / "uniform_schedule.jsonl"
            ).resolve(),
        }
        artifact_files = {
            source_relative: {
                "bytes": bundle_paths[bundle_relative].stat().st_size,
                "sha256": file_sha256(bundle_paths[bundle_relative]),
            }
            for bundle_relative, source_relative in bundle_sources.items()
        }
        anchor = {
            "activation_commit": "c" * 40,
            "anchor_commit": "d" * 40,
            "scientific_identity": pilot_identity,
            "activation_scientific_identity": {
                "scientific_commit": "c" * 40,
                "scientific_path_sha256": pilot_identity["scientific_path_sha256"],
            },
            "registry_raw_sha256": "e" * 64,
            "artifact_files": artifact_files,
            "bundle_sources": bundle_sources,
            "bundle_paths": bundle_paths,
            "pilot_report": pilot_report,
            "pilot_comparison": comparison,
        }
        with (
            patch.multiple(
                freezer,
                CANONICAL_HISTORIES=16,
                CANONICAL_LABELS=64,
                REFINEMENT_ROUNDS=2,
            ),
            patch.object(freezer, "verify_registered_dataset", return_value=replay),
            patch.object(
                freezer,
                "_require_committed_pilot_anchor",
                return_value=anchor,
            ),
        ):
            manifest = build_trainer_bundle(
                dataset,
                frozen / "cgb_schedule.jsonl",
                bundle,
                canonical=True,
                pilot_report_path=frozen / "report.json",
            )
        with patch.object(
            trainer,
            "load_committed_pilot_anchor",
            return_value=anchor,
        ):
            anchored_summary = trainer.validate_trainer_bundle_contract(
                bundle, manifest
            )
        self.assertEqual(anchored_summary["pilot_anchor_commit"], "d" * 40)
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
        snapshot = self._canonical_snapshot("producer", "123")
        out = Path(self.temporary.name) / "live_slurm_replay"
        kwargs = {
            "refinement_rounds": 2,
            "updates_per_round": 1,
            "final_updates": 1,
            "batch_size": 8,
            "max_groups": 4,
        }
        environment = self._canonical_environment("producer", "123")
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(freezer, "_slurm_snapshot", return_value=snapshot),
            patch.object(freezer, "CANONICAL_PILOT_RUNTIME", None),
            patch.object(
                freezer.socket,
                "getfqdn",
                return_value=freezer.CANONICAL_PILOT_ROLES["producer"]["hostname"],
            ),
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
            patch.dict(os.environ, environment, clear=True),
            patch.object(freezer, "_slurm_snapshot", return_value=stale),
            patch.object(freezer, "CANONICAL_PILOT_RUNTIME", None),
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
            freezer.CANONICAL_PILOT_VERIFICATION: root / "verification",
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
            patch.dict(
                os.environ,
                self._canonical_environment("producer", "123"),
                clear=True,
            ),
            patch.object(
                freezer,
                "require_canonical_pilot_runtime",
                return_value=freezer.CANONICAL_PILOT_RUNTIME,
            ),
            patch.object(freezer, "_canonical_path", side_effect=canonical_path),
            patch.object(freezer, "scientific_identity", return_value=identity),
            patch.object(
                freezer,
                "_slurm_snapshot",
                return_value=self._canonical_snapshot("producer", "123"),
            ),
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

    def test_runtime_warmup_loads_hostname_resolver_before_fingerprinting(self):
        with patch.object(
            freezer.socket,
            "getfqdn",
            return_value="test.example",
        ) as getfqdn:
            freezer._warm_pilot_runtime()
        getfqdn.assert_called_once_with()

    def test_canonical_runtime_and_namespace_are_literal_pins(self):
        self.assertEqual(freezer.PILOT_PROTOCOL, "R12-ACW-CGBR-PILOT-v6")
        self.assertEqual(
            freezer.PILOT_EXECUTION_PROTOCOL,
            "R12-ACW-PILOT-REPLAY-EXECUTION-v6",
        )
        self.assertEqual(
            freezer.PILOT_COMPARISON_PROTOCOL,
            "R12-ACW-PILOT-REPLAY-COMPARISON-v6",
        )
        self.assertEqual(
            freezer.PILOT_ORCHESTRATION_PROTOCOL,
            "R12-ACW-PILOT-ORCHESTRATION-v3",
        )
        self.assertEqual(
            freezer.PILOT_INDEPENDENT_VERIFICATION_PROTOCOL,
            "R12-ACW-PILOT-INDEPENDENT-VERIFICATION-v3",
        )
        self.assertEqual(
            freezer.PILOT_ARTIFACT_REGISTRY_PROTOCOL,
            "R12-ACW-PILOT-ARTIFACT-REGISTRY-v2",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_DATASET_PAYLOAD_SHA256,
            "3294a0d12d277f46ea8c0cbf50142be14816447c15bc3792f6e4df7e77e2ba33",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_DATASET,
            "artifacts/r12/acw_pilot_domain_v3_runtime_v2",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_REPLAY_A,
            "artifacts/r12/acw_cgbr_pilot_v6_replay_a",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_REPLAY_B,
            "artifacts/r12/acw_cgbr_pilot_v6_replay_b",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_OUTPUT,
            "artifacts/r12/acw_cgbr_pilot_v6",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_VERIFICATION,
            "artifacts/r12/acw_cgbr_pilot_v6_independent_verification",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_REGISTRY,
            "R12_ACW_PILOT_ARTIFACT_REGISTRY_V2.json",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_THREAD_ENV["ATEN_CPU_CAPABILITY"],
            "avx2",
        )
        self.assertEqual(
            freezer.CANONICAL_PILOT_THREAD_ENV["OPENBLAS_CORETYPE"],
            "Haswell",
        )
        self.assertTrue(freezer.CANONICAL_PILOT_RUNTIME["python_no_site"])
        self.assertTrue(freezer.CANONICAL_PILOT_RUNTIME["python_safe_path"])
        for key in (
            "code_trees",
            "external_executables",
            "generated_modules",
            "imported_external_code",
            "native_files",
            "python_startup",
        ):
            self.assertTrue(freezer.CANONICAL_PILOT_RUNTIME[key], key)
        self.assertEqual(len(freezer.CANONICAL_PILOT_RUNTIME["native_files"]), 93)
        self.assertEqual(
            hashlib.sha256(
                freezer.canonical_json_bytes(
                    freezer.CANONICAL_PILOT_RUNTIME["native_files"]
                )
            ).hexdigest(),
            freezer.CANONICAL_PILOT_RUNTIME["native_files_payload_sha256"],
        )
        environment = self._canonical_environment("producer", "123")
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                freezer, "_canonical_pilot_process_membership", return_value={}
            ),
            patch.object(
                freezer,
                "pilot_runtime_identity",
                return_value=freezer.CANONICAL_PILOT_RUNTIME,
            ) as runtime_identity,
        ):
            self.assertEqual(
                freezer.require_canonical_pilot_runtime(),
                freezer.CANONICAL_PILOT_RUNTIME,
            )
            self.assertEqual(
                freezer.require_canonical_pilot_runtime(),
                freezer.CANONICAL_PILOT_RUNTIME,
            )
            self.assertEqual(runtime_identity.call_count, 2)
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                freezer, "_canonical_pilot_process_membership", return_value={}
            ),
            patch.object(freezer, "pilot_runtime_identity", return_value={}),
            self.assertRaisesRegex(RuntimeError, "runtime identity mismatch"),
        ):
            freezer.require_canonical_pilot_runtime()
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "environment allowlist mismatch"),
        ):
            freezer.require_canonical_pilot_runtime()
        with (
            patch.dict(
                os.environ, {**environment, "LD_PRELOAD": "/tmp/inject.so"}, clear=True
            ),
            self.assertRaisesRegex(RuntimeError, "environment allowlist mismatch"),
        ):
            freezer.require_canonical_pilot_runtime()

    def test_runtime_code_tree_fingerprint_covers_every_file(self):
        root = Path(self.temporary.name) / "runtime_tree"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "module.py").write_text("VALUE = 1\n")
        (root / "pkg" / "weights.bin").write_bytes(b"weights")
        (root / "pkg" / "__pycache__").mkdir()
        (root / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"cache")
        first = freezer._tree_payload_summary(root.resolve())
        self.assertEqual(first["file_count"], 3)
        reference = hashlib.sha256()
        for path in sorted(path for path in root.rglob("*") if path.is_file()):
            record = {
                "bytes": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
            encoded = freezer.canonical_json_bytes(record)
            reference.update(len(encoded).to_bytes(8, "big"))
            reference.update(encoded)
        self.assertEqual(first["payload_sha256"], reference.hexdigest())
        (root / "pkg" / "module.py").write_text("VALUE = 2\n")
        second = freezer._tree_payload_summary(root.resolve())
        self.assertNotEqual(first["payload_sha256"], second["payload_sha256"])
        link = root / "pkg" / "alias.py"
        link.symlink_to(root / "pkg" / "module.py")
        with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
            freezer._tree_payload_summary(root.resolve())

    def test_runtime_code_tree_prunes_explicitly_excluded_top_level(self):
        root = Path(self.temporary.name) / "pruned_runtime_tree"
        root.mkdir()
        (root / "module.py").write_bytes(b"included")
        excluded = root / "site-packages"
        excluded.mkdir()
        (excluded / "large.bin").write_bytes(b"excluded")
        pruned = freezer._tree_payload_summary(
            root.resolve(),
            excluded_top_levels={"site-packages"},
        )
        (excluded / "large.bin").write_bytes(b"changed but still excluded")
        repeated = freezer._tree_payload_summary(
            root.resolve(),
            excluded_top_levels={"site-packages"},
        )
        self.assertEqual(pruned, repeated)
        self.assertEqual(pruned["file_count"], 1)

    def test_runtime_import_closure_does_not_hide_site_packages(self):
        root = Path(self.temporary.name) / "import_closure"
        stdlib = root / "python3.13"
        site_packages = stdlib / "site-packages"
        torch_root = site_packages / "torch"
        numpy_root = site_packages / "numpy"
        site_packages.mkdir(parents=True)
        torch_root.mkdir()
        numpy_root.mkdir()
        stdlib_module = stdlib / "stdlib_module.py"
        external_module = site_packages / "external_module.py"
        stdlib_module.write_bytes(b"stdlib")
        external_module.write_bytes(b"external")
        modules = {
            "stdlib_module": SimpleNamespace(__file__=str(stdlib_module)),
            "external_module": SimpleNamespace(__file__=str(external_module)),
        }
        with (
            patch.object(
                freezer,
                "CANONICAL_PILOT_CODE_TREES",
                {
                    "numpy": str(numpy_root),
                    "python_stdlib": str(stdlib),
                    "torch": str(torch_root),
                },
            ),
            patch.object(freezer, "ACW_SCIENTIFIC_PATHS", ()),
            patch.object(freezer.sys, "modules", modules),
        ):
            summary = freezer._pilot_imported_external_code_summary()
        self.assertEqual(summary["file_count"], 1)
        self.assertEqual(summary["total_bytes"], len(b"external"))

    def test_runtime_startup_fingerprint_covers_pth_execution_files(self):
        purelib = Path(self.temporary.name) / "startup_purelib"
        purelib.mkdir()
        startup = purelib / "runtime_hook.pth"
        startup.write_text("import runtime_hook\n")
        with (
            patch.object(freezer.sysconfig, "get_path", return_value=str(purelib)),
            patch.object(
                freezer.sys,
                "path",
                [str(purelib), str(purelib / "python313.zip")],
            ),
            self.assertRaisesRegex(RuntimeError, "archive import paths are forbidden"),
        ):
            freezer._pilot_python_startup_identity()
        with (
            patch.object(freezer.sysconfig, "get_path", return_value=str(purelib)),
            patch.object(freezer.sys, "path", [str(purelib)]),
        ):
            first = freezer._pilot_python_startup_identity()
            startup.write_text("import different_runtime_hook\n")
            second = freezer._pilot_python_startup_identity()
        self.assertNotEqual(
            first["startup_files_payload_sha256"],
            second["startup_files_payload_sha256"],
        )

    def test_runtime_external_executable_registry_hashes_bytes(self):
        self.assertIn("/usr/bin/git", freezer.CANONICAL_PILOT_EXTERNAL_EXECUTABLE_PATHS)
        self.assertIn("/bin/ps", freezer.CANONICAL_PILOT_EXTERNAL_EXECUTABLE_PATHS)
        executable = Path(self.temporary.name) / "runtime_tool"
        executable.write_bytes(b"version-one")
        with patch.object(
            freezer,
            "CANONICAL_PILOT_EXTERNAL_EXECUTABLE_PATHS",
            (str(executable),),
        ):
            first = freezer._pilot_external_executable_registry()
            executable.write_bytes(b"version-two")
            second = freezer._pilot_external_executable_registry()
        self.assertNotEqual(
            first[str(executable)]["sha256"],
            second[str(executable)]["sha256"],
        )

    def test_runtime_generated_module_is_path_independent_and_closed(self):
        root = Path(self.temporary.name) / "generated_module_root"
        root.mkdir()
        module_path = root / "_remote_module_non_scriptable.py"
        module_path.write_bytes(b"GENERATED = True\n")
        with (
            patch.object(
                freezer,
                "_DETACHED_TORCH_GENERATED_MODULES",
                {"_remote_module_non_scriptable": module_path},
            ),
            patch.object(freezer.sys, "path", ["/fixed/import/root"]),
        ):
            summary = freezer._pilot_generated_module_summary()
            self.assertEqual(
                summary,
                {
                    "_remote_module_non_scriptable": {
                        "bytes": module_path.stat().st_size,
                        "filename": module_path.name,
                        "sha256": file_sha256(module_path),
                    }
                },
            )
            (root / "injected.py").write_text("INJECTED = True\n")
            with self.assertRaisesRegex(RuntimeError, "import root is open"):
                freezer._pilot_generated_module_summary()

    def test_runtime_process_membership_binds_the_live_slurm_cgroup(self):
        job_id = "740149"
        user_id = freezer.CANONICAL_PILOT_UID
        task = f"/slurm/uid_{user_id}/job_{job_id}/step_batch/task_0"
        step = f"/slurm/uid_{user_id}/job_{job_id}/step_batch"
        cgroups = "\n".join(
            (
                f"9:memory:{task}",
                f"6:freezer:{step}",
                f"5:cpu,cpuacct:{task}",
                f"3:cpuset:{task}",
            )
        )
        status = "Cpus_allowed_list:\t9,11,15,19\nMems_allowed_list:\t0-1\n"
        record = freezer._validate_canonical_pilot_process_membership(
            cgroups,
            status,
            job_id=job_id,
            user_id=user_id,
        )
        self.assertEqual(record["cpu_list"], "9,11,15,19")
        with self.assertRaisesRegex(RuntimeError, "outside its Slurm cgroup"):
            freezer._validate_canonical_pilot_process_membership(
                cgroups.replace(f"job_{job_id}", "job_999999"),
                status,
                job_id=job_id,
                user_id=user_id,
            )
        with self.assertRaisesRegex(RuntimeError, "allocation differs"):
            freezer._validate_canonical_pilot_process_membership(
                cgroups,
                status.replace("9,11,15,19", "9,11,15"),
                job_id=job_id,
                user_id=user_id,
            )
        self.assertEqual(
            freezer._validate_canonical_pilot_batch_script(b"exact", b"exact"),
            hashlib.sha256(b"exact").hexdigest(),
        )
        with self.assertRaisesRegex(RuntimeError, "spooled batch script differs"):
            freezer._validate_canonical_pilot_batch_script(b"modified", b"exact")

    def test_canonical_path_rejects_dangling_symlink(self):
        root = Path(self.temporary.name) / "symlink_repo"
        pipeline = root / "pipeline"
        parent = root / "artifacts" / "r12"
        pipeline.mkdir(parents=True)
        parent.mkdir(parents=True)
        (parent / "redirect").symlink_to(
            root / "missing-target", target_is_directory=True
        )
        with (
            patch.object(freezer, "__file__", str(pipeline / "freeze.py")),
            self.assertRaisesRegex(ValueError, "contains a symlink"),
        ):
            freezer._canonical_path("artifacts/r12/redirect/result")

    def test_canonical_artifact_tree_rejects_leaf_symlink(self):
        root = Path(self.temporary.name) / "symlink_tree"
        root.mkdir()
        target = Path(self.temporary.name) / "target.json"
        target.write_text("{}")
        (root / "report.json").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "tree contains a symlink"):
            freezer._require_tree_without_symlinks(root)

    def test_canonical_pilot_rejects_unpinned_dataset_payload(self):
        with (
            patch.object(
                freezer,
                "require_canonical_pilot_runtime",
                return_value=freezer.CANONICAL_PILOT_RUNTIME,
            ),
            patch.object(freezer, "verify_registered_dataset", return_value={}),
            self.assertRaisesRegex(ValueError, "cross-runtime pin"),
        ):
            run_pilot(self.root, canonical=True)

    def test_registered_dataset_rejects_unmanifested_file(self):
        root = self._copy_dataset("unmanifested_file_dataset")
        extra = root / "notes.txt"
        extra.write_text("not part of the generator contract")
        with self.assertRaisesRegex(ValueError, "tree registry"):
            self._verify_small_registered_dataset(
                root,
                allowed_kinds={"pilot"},
            )

    def test_pilot_comparison_rejects_unknown_claim(self):
        first = Path(self.temporary.name) / "schema_first"
        second = Path(self.temporary.name) / "schema_second"
        frozen = Path(self.temporary.name) / "schema_frozen"
        kwargs = {
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
            pilot_kwargs=kwargs,
        )
        execute_pilot_replay(
            self.root,
            second,
            replay_id="b",
            canonical=False,
            pilot_kwargs=kwargs,
        )
        freezer.freeze_pilot_replays(
            first,
            second,
            frozen,
            dataset_root=self.root,
            canonical=False,
        )
        comparison_path = frozen / "replay_comparison.json"
        comparison = json.loads(comparison_path.read_text())
        comparison["different_node_verified"] = True
        comparison.pop("payload_sha256")
        comparison["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(comparison)
        ).hexdigest()
        comparison_path.chmod(0o644)
        comparison_path.write_bytes(freezer.canonical_json_bytes(comparison) + b"\n")
        report = json.loads((frozen / "report.json").read_text())
        with (
            patch.object(
                freezer,
                "_canonical_path",
                side_effect=lambda relative: (
                    frozen if relative == freezer.CANONICAL_PILOT_OUTPUT else self.root
                ),
            ),
            patch.object(freezer, "_validate_replay_report", return_value=report),
            self.assertRaisesRegex(ValueError, "comparison has the wrong schema"),
        ):
            freezer.load_pilot_report(frozen / "report.json")

    def test_independent_verifier_requires_different_job_and_node(self):
        producer = self._canonical_snapshot("producer", "10")
        verifier = self._canonical_snapshot("verifier", "11")
        producer_hostname = freezer.CANONICAL_PILOT_ROLES["producer"]["hostname"]
        verifier_hostname = freezer.CANONICAL_PILOT_ROLES["verifier"]["hostname"]
        with patch.dict(
            os.environ,
            self._canonical_environment("verifier", "11"),
            clear=True,
        ):
            freezer._validate_independent_verifier_allocation(
                producer,
                verifier,
                producer_hostname=producer_hostname,
                verifier_hostname=verifier_hostname,
            )
            for key, value in (
                ("job_id", "10"),
                ("node_list", "ec10"),
            ):
                same = deepcopy(verifier)
                same["allocation"][key] = value
                with self.assertRaisesRegex(ValueError, "canonical producer"):
                    freezer._validate_independent_verifier_allocation(
                        producer,
                        same,
                        producer_hostname=producer_hostname,
                        verifier_hostname=verifier_hostname,
                    )
            with self.assertRaisesRegex(ValueError, "canonical producer"):
                freezer._validate_independent_verifier_allocation(
                    producer,
                    verifier,
                    producer_hostname=producer_hostname,
                    verifier_hostname=producer_hostname,
                )

    def test_slurm_role_binding_rejects_manual_module_invocation(self):
        snapshot = self._canonical_snapshot("producer", "123")
        expected_command = freezer.CANONICAL_PILOT_ROLES["producer"]["command"]
        snapshot["stdout"] = snapshot["stdout"].replace(
            f"Command={expected_command}",
            "Command=/tmp/manual_probe.sh",
        )
        snapshot["stdout_sha256"] = hashlib.sha256(
            snapshot["stdout"].encode("utf-8")
        ).hexdigest()
        snapshot["allocation"]["command"] = "/tmp/manual_probe.sh"
        with self.assertRaisesRegex(ValueError, "differs from its role"):
            freezer._validate_slurm_snapshot_record(snapshot, role="producer")

    def test_independent_verifier_freezes_hash_bound_receipt(self):
        root = Path(self.temporary.name) / "independent_verifier"
        pilot = root / "pilot"
        verification = root / "verification"
        pilot.mkdir(parents=True)
        report = {
            "dataset_manifest_payload_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        report["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(
                {key: value for key, value in report.items() if key != "payload_sha256"}
            )
        ).hexdigest()
        (pilot / "report.json").write_bytes(
            freezer.canonical_json_bytes(report) + b"\n"
        )
        producer_snapshot = self._canonical_snapshot("producer", "10")
        comparison = {
            "orchestration": {
                "hostname": freezer.CANONICAL_PILOT_ROLES["producer"]["hostname"],
                "slurm_snapshot": producer_snapshot,
            }
        }
        comparison["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(comparison)
        ).hexdigest()
        (pilot / "replay_comparison.json").write_bytes(
            freezer.canonical_json_bytes(comparison) + b"\n"
        )
        verifier_snapshot = self._canonical_snapshot("verifier", "11")
        identity = {
            "scientific_commit": "c" * 40,
            "scientific_path_sha256": {"test": "d" * 64},
        }
        artifact_path = Path("pipeline/freeze_acw_curriculum.py")
        artifacts = {
            str(artifact_path): {
                "bytes": artifact_path.stat().st_size,
                "sha256": file_sha256(artifact_path),
            }
        }

        def canonical_path(relative):
            if relative == freezer.CANONICAL_PILOT_OUTPUT:
                return pilot
            if relative == freezer.CANONICAL_PILOT_VERIFICATION:
                return verification
            raise AssertionError(relative)

        with (
            patch.dict(
                os.environ,
                self._canonical_environment("verifier", "11"),
                clear=True,
            ),
            patch.object(freezer, "_canonical_path", side_effect=canonical_path),
            patch.object(
                freezer,
                "require_canonical_pilot_runtime",
                return_value=freezer.CANONICAL_PILOT_RUNTIME,
            ),
            patch.object(freezer, "scientific_identity", return_value=identity),
            patch.object(freezer, "_slurm_snapshot", return_value=verifier_snapshot),
            patch.object(freezer, "load_pilot_report", return_value=report),
            patch.object(
                freezer,
                "_canonical_pilot_artifact_registry",
                return_value=artifacts,
            ),
            patch.object(
                freezer.sys,
                "executable",
                freezer.CANONICAL_PILOT_RUNTIME["python_executable"],
            ),
            patch.object(
                freezer.socket,
                "getfqdn",
                return_value=freezer.CANONICAL_PILOT_ROLES["verifier"]["hostname"],
            ),
        ):
            receipt = freezer.verify_canonical_pilot_independently()
            receipt_bytes = freezer.canonical_json_bytes(receipt) + b"\n"
            (
                loaded_receipt,
                loaded_receipt_file,
                loaded_report,
                loaded_comparison,
                loaded_artifacts,
            ) = freezer.load_independent_pilot_verification(
                expected_receipt=receipt,
                expected_receipt_bytes=receipt_bytes,
            )
            substituted_receipt = {**receipt, "process_id": receipt["process_id"] + 1}
            substituted_receipt.pop("payload_sha256")
            substituted_receipt["payload_sha256"] = hashlib.sha256(
                freezer.canonical_json_bytes(substituted_receipt)
            ).hexdigest()
            receipt_path = verification / "verification.json"
            receipt_path.chmod(0o644)
            receipt_path.write_bytes(
                freezer.canonical_json_bytes(substituted_receipt) + b"\n"
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "differs from verifier-process bytes",
            ):
                freezer.load_independent_pilot_verification(
                    expected_receipt=receipt,
                    expected_receipt_bytes=receipt_bytes,
                )
            receipt_path.write_bytes(receipt_bytes)
            receipt_path.chmod(0o444)
        frozen = json.loads((verification / "verification.json").read_text())
        self.assertEqual(receipt, frozen)
        self.assertEqual(loaded_receipt, receipt)
        self.assertEqual(
            loaded_receipt_file,
            {
                "bytes": (verification / "verification.json").stat().st_size,
                "sha256": file_sha256(verification / "verification.json"),
            },
        )
        self.assertEqual(loaded_report, report)
        self.assertEqual(loaded_comparison, comparison)
        self.assertEqual(loaded_artifacts, artifacts)
        self.assertEqual(receipt["artifact_files"], artifacts)
        self.assertTrue(receipt["fresh_recomputation_complete"])
        self.assertEqual(
            receipt["producer"]["hostname"],
            freezer.CANONICAL_PILOT_ROLES["producer"]["hostname"],
        )
        self.assertEqual(
            receipt["verifier_slurm_snapshot_finish"]["allocation"]["node_list"],
            "ec52",
        )
        self.assertEqual(verification.stat().st_mode & 0o777, 0o555)
        self.assertEqual(
            (verification / "verification.json").stat().st_mode & 0o777,
            0o444,
        )

    def test_independent_receipt_requires_unique_canonical_json_bytes(self):
        payload = {"value": 1}
        record = {
            **payload,
            "payload_sha256": hashlib.sha256(
                freezer.canonical_json_bytes(payload)
            ).hexdigest(),
        }
        canonical = freezer.canonical_json_bytes(record) + b"\n"
        self.assertEqual(
            freezer._load_hash_bound_json_bytes(
                canonical,
                label="receipt",
                require_canonical_bytes=True,
            ),
            record,
        )
        duplicate = (
            b'{"payload_sha256":"'
            + record["payload_sha256"].encode("ascii")
            + b'","value":0,"value":1}\n'
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            freezer._load_hash_bound_json_bytes(
                duplicate,
                label="receipt",
                require_canonical_bytes=True,
            )
        noncanonical = (
            json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "not canonical JSON"):
            freezer._load_hash_bound_json_bytes(
                noncanonical,
                label="receipt",
                require_canonical_bytes=True,
            )

    def test_anchored_registry_adds_the_independent_receipt(self):
        root = Path(self.temporary.name) / "anchored_registry"
        pipeline = root / "pipeline"
        verification = root / freezer.CANONICAL_PILOT_VERIFICATION
        pipeline.mkdir(parents=True)
        verification.mkdir(parents=True)
        receipt_path = verification / "verification.json"
        receipt_path.write_text("{}\n")
        receipt_record = {
            "bytes": receipt_path.stat().st_size,
            "sha256": file_sha256(receipt_path),
        }
        base = {"artifacts/r12/base/file.bin": {"bytes": 1, "sha256": "a" * 64}}
        with (
            patch.object(freezer, "__file__", str(pipeline / "freeze.py")),
            patch.object(
                freezer,
                "_canonical_pilot_artifact_registry",
                return_value=base,
            ),
            patch.object(freezer, "CANONICAL_PILOT_ANCHORED_FILES", 2),
        ):
            anchored = freezer._canonical_pilot_anchored_artifact_registry(
                base,
                receipt_record,
            )
            with self.assertRaisesRegex(ValueError, "changed after receipt validation"):
                freezer._canonical_pilot_anchored_artifact_registry(
                    {
                        **base,
                        "artifacts/r12/injected.bin": {
                            "bytes": 1,
                            "sha256": "b" * 64,
                        },
                    },
                    receipt_record,
                )
            with self.assertRaisesRegex(
                ValueError,
                "verification receipt changed after semantic validation",
            ):
                freezer._canonical_pilot_anchored_artifact_registry(
                    base,
                    {"bytes": 0, "sha256": "c" * 64},
                )
        relative = f"{freezer.CANONICAL_PILOT_VERIFICATION}/verification.json"
        self.assertEqual(anchored[relative], receipt_record)
        self.assertEqual(len(anchored), 2)

    def test_registry_builder_is_receipt_gated_and_exact_schema(self):
        with self.assertRaises(TypeError):
            freezer.build_canonical_pilot_artifact_registry()
        root = Path(self.temporary.name) / "registry_builder"
        pilot = root / "pilot"
        verification = root / "verification"
        pilot.mkdir(parents=True)
        verification.mkdir()
        report = {
            "dataset_manifest_payload_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        comparison = {"payload_sha256": "c" * 64}
        receipt = {"test": "receipt"}
        receipt["payload_sha256"] = hashlib.sha256(
            freezer.canonical_json_bytes(receipt)
        ).hexdigest()
        receipt_bytes = freezer.canonical_json_bytes(receipt) + b"\n"
        (pilot / "report.json").write_bytes(
            freezer.canonical_json_bytes(report) + b"\n"
        )
        (pilot / "replay_comparison.json").write_bytes(
            freezer.canonical_json_bytes(comparison) + b"\n"
        )
        (verification / "verification.json").write_bytes(
            freezer.canonical_json_bytes(receipt) + b"\n"
        )
        receipt_file = {
            "bytes": (verification / "verification.json").stat().st_size,
            "sha256": file_sha256(verification / "verification.json"),
        }
        identity = {
            "scientific_commit": "e" * 40,
            "scientific_path_sha256": {"test": "f" * 64},
        }
        artifact_paths = (
            Path("AGENT_RUNBOOK.md"),
            Path("pipeline/freeze_acw_curriculum.py"),
        )
        anchored = {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        }
        registry_path = root / freezer.CANONICAL_PILOT_REGISTRY

        def canonical_path(relative):
            return {
                freezer.CANONICAL_PILOT_OUTPUT: pilot,
                freezer.CANONICAL_PILOT_VERIFICATION: verification,
                freezer.CANONICAL_PILOT_REGISTRY: registry_path,
            }[relative]

        with (
            patch.object(freezer, "_canonical_path", side_effect=canonical_path),
            patch.object(
                freezer,
                "load_independent_pilot_verification",
                return_value=(receipt, receipt_file, report, comparison, {}),
            ) as receipt_gate,
            patch.object(freezer, "scientific_identity", return_value=identity),
            patch.object(
                freezer,
                "_canonical_pilot_anchored_artifact_registry",
                return_value=anchored,
            ) as anchored_gate,
        ):
            registry = freezer.build_canonical_pilot_artifact_registry(
                expected_receipt=receipt,
                expected_receipt_bytes=receipt_bytes,
            )
            with self.assertRaisesRegex(ValueError, "differs from verified"):
                freezer._validate_pilot_artifact_registry_record(
                    {**registry, "unregistered_claim": True},
                    identity=identity,
                    receipt=receipt,
                    report=report,
                    comparison=comparison,
                    artifact_files=anchored,
                )
        receipt_gate.assert_called_once_with(
            expected_receipt=receipt,
            expected_receipt_bytes=receipt_bytes,
        )
        self.assertEqual(anchored_gate.call_count, 2)
        self.assertTrue(
            all(
                call.args == ({}, receipt_file) for call in anchored_gate.call_args_list
            )
        )
        self.assertEqual(registry["artifact_file_count"], 2)
        self.assertEqual(
            registry["activation_allowlist"],
            list(freezer.CANONICAL_PILOT_ACTIVATION_ALLOWLIST),
        )
        self.assertEqual(registry_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(
            registry_path.read_bytes(),
            freezer.canonical_json_bytes(registry) + b"\n",
        )

    def test_verifier_and_registry_builder_share_in_memory_receipt(self):
        receipt = {"payload_sha256": "a" * 64}
        registry = {"artifact_file_count": 81, "payload_sha256": "b" * 64}
        with (
            patch.object(
                freezer,
                "verify_canonical_pilot_independently",
                return_value=receipt,
            ) as verifier,
            patch.object(
                freezer,
                "build_canonical_pilot_artifact_registry",
                return_value=registry,
            ) as builder,
        ):
            actual_receipt, actual_registry = (
                freezer.verify_and_build_canonical_pilot_artifact_registry()
            )
        verifier.assert_called_once_with()
        builder.assert_called_once_with(
            expected_receipt=receipt,
            expected_receipt_bytes=freezer.canonical_json_bytes(receipt) + b"\n",
        )
        self.assertIs(actual_receipt, receipt)
        self.assertIs(actual_registry, registry)
        command_choices = freezer.build_parser()._subparsers._group_actions[0].choices
        self.assertNotIn("build-pilot-artifact-registry", command_choices)

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
        self.assertIn("#SBATCH --nodelist=ec51", source)
        self.assertIn("#SBATCH --export=NONE", source)
        for canonical_path in (
            freezer.CANONICAL_PILOT_DATASET,
            freezer.CANONICAL_PILOT_REPLAY_A,
            freezer.CANONICAL_PILOT_REPLAY_B,
            freezer.CANONICAL_PILOT_OUTPUT,
            freezer.CANONICAL_PILOT_VERIFICATION,
        ):
            self.assertIn(canonical_path, source)
        self.assertNotIn("acw_pilot_domain_v2", source)
        self.assertNotIn("acw_cgbr_pilot_v3", source)
        self.assertNotIn("acw_cgbr_pilot_v4", source)
        self.assertNotIn("acw_cgbr_pilot_v5", source)
        self.assertNotIn("acw_pilot_domain_v3_runtime_v1", source)
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(
                'run_python "$BASE/pipeline/freeze_acw_curriculum.py"'
            )
        ]
        self.assertEqual(
            commands,
            [
                'run_python "$BASE/pipeline/freeze_acw_curriculum.py" pilot-run',
                'run_python "$BASE/pipeline/freeze_acw_curriculum.py" verify-pilot',
            ],
        )
        self.assertIn("env -i", source)
        self.assertIn('"$PY" -S -P "$@"', source)
        self.assertIn(
            "PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13",
            source,
        )
        self.assertIn('[[ ! -e "$PYZIP" ]]', source)
        self.assertNotIn("${PY:-", source)
        for key, value in freezer.CANONICAL_PILOT_STATIC_ENV.items():
            self.assertIn(f"{key}={value} \\", source)
        for key in freezer.CANONICAL_PILOT_DYNAMIC_ENV_KEYS:
            self.assertIn(f'{key}="${key}" \\', source)
        self.assertNotIn(" bundle", source)
        self.assertNotIn('"$PY" -m pipeline.acw_hidden_basis_training', source)
        self.assertNotIn('"$PY" -m pipeline.adjudicate_acw_hidden_basis', source)

    def test_stokes_independent_verifier_has_one_fail_closed_command(self):
        source = Path("pipeline/jobs/verify_acw_pilot_stokes.sbatch").read_text()
        self.assertIn("#SBATCH --nodelist=ec52", source)
        self.assertIn("#SBATCH --export=NONE", source)
        self.assertIn(freezer.CANONICAL_PILOT_OUTPUT, source)
        self.assertIn(freezer.CANONICAL_PILOT_VERIFICATION, source)
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(
                'run_python "$BASE/pipeline/freeze_acw_curriculum.py"'
            )
        ]
        self.assertEqual(
            commands,
            [
                'run_python "$BASE/pipeline/freeze_acw_curriculum.py" '
                "verify-pilot-independent",
            ],
        )
        self.assertNotIn("build-pilot-artifact-registry", source)
        self.assertIn(freezer.CANONICAL_PILOT_REGISTRY, source)
        self.assertNotIn("acw_cgbr_pilot_v5", source)
        self.assertNotIn("R12_ACW_PILOT_ARTIFACT_REGISTRY.json", source)
        self.assertIn("env -i", source)
        self.assertIn('"$PY" -S -P "$@"', source)
        self.assertIn(
            "PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13",
            source,
        )
        self.assertIn('[[ ! -e "$PYZIP" ]]', source)
        for key, value in freezer.CANONICAL_PILOT_STATIC_ENV.items():
            self.assertIn(f"{key}={value} \\", source)
        for key in freezer.CANONICAL_PILOT_DYNAMIC_ENV_KEYS:
            self.assertIn(f'{key}="${key}" \\', source)
        self.assertNotIn(" pilot-run", source)
        self.assertNotIn(" bundle", source)


if __name__ == "__main__":
    unittest.main()
