import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import build_acw_development_manifest as builder
from pipeline.acw_hidden_basis_training import (
    DEVELOPMENT_PLAN_PATH,
    DEVELOPMENT_PLAN_RAW_SHA256,
    file_sha256,
)
from pipeline.adjudicate_acw_hidden_basis import (
    DEVELOPMENT_SEEDS,
    DEVELOPMENT_MANIFEST_PROTOCOL,
    DIRECT_STATE_ARM,
    DIRECT_STATE_MANIFEST_PROTOCOL,
    SCORED_ARMS,
)


def _attempt_id(index: int, arm: str) -> str:
    return f"{arm}__{DEVELOPMENT_SEEDS[index]}"


class ACWDevelopmentManifestBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "attempt"
        self.root.mkdir()
        repository = Path(__file__).resolve().parents[1]
        plan = self.root / "development_plan.json"
        shutil.copyfile(repository / DEVELOPMENT_PLAN_PATH, plan)
        plan.chmod(0o444)
        attempt_start = self.root / "attempt_start.json"
        attempt_start.write_bytes(b"{}\n")
        attempt_start.chmod(0o444)
        self._immutable_file(self.root / "attempt_claim.json", b"{}\n")
        self._immutable_file(self.root / "direct_refit_verification.json", b"{}\n")
        self._immutable_file(self.root / "final_refit_verification.json", b"{}\n")
        for role in builder.ROLES:
            self._immutable_file(self.root / builder.ROLE_START_FILES[role], b"{}\n")
            self._immutable_file(
                self.root / builder.ROLE_COMPLETION_FILES[role], b"{}\n"
            )
            self._immutable_file(
                self.root / builder.ROLE_ACCOUNTING_FILES[role], b"{}\n"
            )

    def tearDown(self) -> None:
        for path in sorted(
            self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif path.exists() and not path.is_symlink():
                path.chmod(0o600)
        self.root.chmod(0o700)
        self.temporary.cleanup()

    def _immutable_file(self, path: Path, raw: bytes = b"artifact\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o444)

    def _immutable_root(self, path: Path) -> None:
        self._immutable_file(path / "manifest.json", b"{}\n")
        path.chmod(0o555)

    def _monitor_binding(self) -> dict:
        return {
            "job_id": "740999",
            "job_name": builder.MONITOR_JOB_NAME,
            "node": builder.MONITOR_NODE,
            "cpus_per_task": "4",
            "dependency": None,
            "script": {"path": builder.MONITOR_SCRIPT, "sha256": "a" * 64},
            "spool_script_sha256": "a" * 64,
            "scontrol_snapshot_sha256": "b" * 64,
            "process_membership": {
                "cpu_list": "0-3",
                "memory_list": "0",
                "task_cgroup": "/test",
            },
            "runtime_identity_sha256": "c" * 64,
        }

    def _add_run(self, index: int, arm: str) -> None:
        dataset = self.root / "inputs" / "datasets" / f"development_{index}"
        family = "uniform" if arm == "uniform_query_acw" else "cgb"
        bundle = self.root / "inputs" / "bundles" / f"development_{index}_{family}"
        task = self.root / "runs" / f"{index:02d}_{arm}"
        if not dataset.exists():
            self._immutable_root(dataset)
        if not bundle.exists():
            self._immutable_root(bundle)
        self._immutable_file(task / "checkpoint.pt")
        self._immutable_file(task / "evaluation.json", b"{}\n")
        self._immutable_file(task / "replay.json", b"{}\n")
        receipt = builder._hash_bound(
            {
                "schema": "r12_acw_development_attempt_receipt_v1",
                "protocol": "R12-ACW-DEVELOPMENT-ATTEMPT-v1",
                "attempt_id": _attempt_id(index, arm),
                "artifact_root": str(self.root.resolve(strict=True)),
                "task_root": task.relative_to(self.root).as_posix(),
                "completed_once": True,
            }
        )
        self._immutable_file(
            task / "attempt.json", builder.canonical_json_bytes(receipt) + b"\n"
        )

    def test_committed_plan_copy_is_exact_and_hash_bound(self) -> None:
        reference = builder._plan_reference(self.root)
        self.assertEqual(reference["sha256"], DEVELOPMENT_PLAN_RAW_SHA256)
        self.assertEqual(
            file_sha256(self.root / "development_plan.json"),
            DEVELOPMENT_PLAN_RAW_SHA256,
        )

    def test_direct_manifest_has_exact_three_seed_matrix(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        manifest = builder.build_direct_manifest(self.root)
        self.assertEqual(manifest["protocol"], DIRECT_STATE_MANIFEST_PROTOCOL)
        self.assertEqual(len(manifest["reports"]), 3)
        self.assertEqual(
            [record["arm"] for record in manifest["reports"]],
            [DIRECT_STATE_ARM] * 3,
        )
        self.assertEqual(
            [record.get("attempt_id") for record in manifest["reports"]],
            [_attempt_id(index, DIRECT_STATE_ARM) for index in range(3)],
        )
        self.assertEqual(
            set(manifest.get("stage_receipts", {})),
            {"phase1_producer", "phase1_verifier"},
        )
        self.assertEqual(manifest["attempt_start"]["path"], "attempt_start.json")

    def test_development_manifest_has_exact_fixed_matrix_and_authorization(
        self,
    ) -> None:
        for arm in (*SCORED_ARMS, DIRECT_STATE_ARM):
            for index in range(3):
                self._add_run(index, arm)
        self._immutable_file(self.root / "phase2_authorization.json", b"{}\n")
        manifest = builder.build_development_manifest(self.root)
        self.assertEqual(manifest["protocol"], DEVELOPMENT_MANIFEST_PROTOCOL)
        self.assertEqual(len(manifest["reports"]), 27)
        self.assertEqual(
            [record["arm"] for record in manifest["reports"]],
            [arm for arm in (DIRECT_STATE_ARM, *SCORED_ARMS) for _ in range(3)],
        )
        expected_attempts = [
            _attempt_id(index, arm)
            for arm in (DIRECT_STATE_ARM, *SCORED_ARMS)
            for index in range(3)
        ]
        self.assertEqual(
            [record.get("attempt_id") for record in manifest["reports"]],
            expected_attempts,
        )
        self.assertEqual(len(set(expected_attempts)), 27)
        self.assertEqual(
            set(manifest.get("stage_receipts", {})),
            {
                "phase1_producer",
                "phase1_verifier",
                "phase2_producer",
                "phase2_verifier",
            },
        )
        self.assertEqual(
            manifest["phase2_authorization"]["path"],
            "phase2_authorization.json",
        )

    def test_hidden_or_extra_attempt_tree_is_rejected(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        self._immutable_file(
            self.root / "runs" / ".discarded_attempt" / "checkpoint.pt"
        )
        with self.assertRaisesRegex(ValueError, "attempt inventory|unexpected"):
            builder.build_direct_manifest(self.root)

    def test_missing_expected_attempt_is_rejected(self) -> None:
        for index in range(2):
            self._add_run(index, DIRECT_STATE_ARM)
        with self.assertRaises((FileNotFoundError, ValueError)):
            builder.build_direct_manifest(self.root)

    def test_off_root_attempt_via_intermediate_symlink_is_rejected(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        task = self.root / "runs" / f"00_{DIRECT_STATE_ARM}"
        external = Path(self.temporary.name) / "external_attempt"
        shutil.copytree(task, external)
        shutil.rmtree(task)
        task.symlink_to(external, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)):
            builder.build_direct_manifest(self.root)

    def test_off_root_attempt_receipt_cannot_be_silently_ignored(self) -> None:
        for index in range(3):
            self._add_run(index, DIRECT_STATE_ARM)
        task = self.root / "runs" / f"00_{DIRECT_STATE_ARM}"
        receipt = builder._hash_bound(
            {
                "schema": "r12_acw_run_attempt_v1",
                "attempt_id": _attempt_id(0, DIRECT_STATE_ARM),
                "artifact_root": str(Path(self.temporary.name) / "outside"),
            }
        )
        receipt_path = task / "attempt.json"
        receipt_path.chmod(0o600)
        receipt_path.write_bytes(builder.canonical_json_bytes(receipt) + b"\n")
        receipt_path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "off-root|artifact root|attempt"):
            builder.build_direct_manifest(self.root)

    def test_private_tree_comparison_opens_every_registered_file(self) -> None:
        producer = Path(self.temporary.name) / "producer_dataset"
        verifier = Path(self.temporary.name) / "verifier_dataset"
        manifest = builder._hash_bound({"arrays": {"array.bin": {}}})
        raw_manifest = builder.canonical_json_bytes(manifest) + b"\n"
        for tree in (producer, verifier):
            tree.mkdir()
            (tree / "manifest.json").write_bytes(raw_manifest)
            (tree / "array.bin").write_bytes(b"original")
        builder._require_registered_tree_bytes_equal(producer, verifier, kind="dataset")
        (verifier / "array.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "regeneration differs"):
            builder._require_registered_tree_bytes_equal(
                producer, verifier, kind="dataset"
            )

    def test_exclusive_writer_freezes_bytes_and_refuses_reuse(self) -> None:
        output = self.root / "manifest.json"
        payload = builder._hash_bound({"schema": "test"})
        digest = builder.write_exclusive(output, payload)
        self.assertEqual(digest, file_sha256(output))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(json.loads(output.read_bytes()), payload)
        with self.assertRaises(FileExistsError):
            builder.write_exclusive(output, payload)

    def test_atomic_publication_recovers_only_its_named_temporary(self) -> None:
        output = self.root / "atomic.bin"
        temporary = builder._protocol_publish_temp(output)
        temporary.write_bytes(b"truncated")
        temporary.chmod(0o444)
        raw = b"complete immutable bytes"
        real_link = os.link

        def checked_link(source: Path, destination: Path, **kwargs) -> None:
            self.assertFalse(os.path.lexists(destination))
            self.assertEqual(Path(source).read_bytes(), raw)
            real_link(source, destination, **kwargs)

        with mock.patch.object(builder.os, "link", side_effect=checked_link):
            digest = builder._atomic_publish_bytes(output, raw)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertFalse(os.path.lexists(temporary))
        with self.assertRaises(FileExistsError):
            builder._atomic_publish_bytes(output, b"replacement")
        self.assertEqual(output.read_bytes(), raw)

    def test_attempt_start_binds_the_precommitted_held_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            repository = home / "shohin_acw"
            root = repository / "artifacts" / "r12" / "acw_development_g1"
            (repository / "pipeline").mkdir(parents=True)
            (root / "runs").mkdir(parents=True)
            plan = json.loads(
                (
                    Path(__file__).resolve().parents[1] / DEVELOPMENT_PLAN_PATH
                ).read_bytes()
            )
            job_ids = ("740999", "741000", "741001", "741002")
            prior = None
            for stage, job_id in zip(plan["custody_stages"], job_ids, strict=True):
                stage["held_slurm_job_id"] = job_id
                stage["dependency"] = (
                    None
                    if prior is None
                    else {"type": "afterok", "held_slurm_job_id": prior}
                )
                prior = job_id
            plan["attempt_registry"]["held_slurm_job_id"] = job_ids[0]
            plan["accounting"]["monitor_stage"]["held_slurm_job_id"] = "741003"
            plan["accounting"]["monitor_stage"]["dependency"] = {
                "type": "afterok",
                "held_slurm_job_id": job_ids[-1],
            }
            plan["accounting"]["monitor_stage"]["script"]["sha256"] = file_sha256(
                builder.REPOSITORY / builder.MONITOR_SCRIPT
            )
            plan["ready_for_g_commit"] = True
            plan["input_table"] = builder.expected_input_table(plan)
            plan["attempt_table"] = builder.expected_attempt_table(plan)
            plan = builder._hash_bound(plan)
            plan_path = root / "development_plan.json"
            plan_path.write_bytes(builder.canonical_json_bytes(plan) + b"\n")
            plan_path.chmod(0o444)
            plan_sha256 = file_sha256(plan_path)
            environment = {
                "SLURM_JOB_ID": "740999",
                "SLURM_JOB_NAME": builder.ROLE_JOB_NAMES[builder.ROLE_PHASE1],
                "SLURM_JOB_NODELIST": "ec51",
                "SLURM_CPUS_PER_TASK": "4",
            }
            with (
                mock.patch.object(
                    builder,
                    "__file__",
                    str(repository / "pipeline" / "build_acw_development_manifest.py"),
                ),
                mock.patch.object(builder, "DEVELOPMENT_PLAN_RAW_SHA256", plan_sha256),
                mock.patch.object(
                    builder, "_validated_g_commit", return_value="d" * 40
                ),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                attempt = builder.build_attempt_start(root.resolve())
        self.assertEqual(attempt["scientific_commit"], "d" * 40)
        self.assertEqual(attempt["slurm"]["job_id"], "740999")
        self.assertTrue(attempt["created_before_scoring"])
        self.assertEqual(attempt["checkpoint_count_at_creation"], 0)

    def test_plan_drift_and_mutable_artifacts_fail_closed(self) -> None:
        plan = self.root / "development_plan.json"
        plan.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "immutable file"):
            builder._plan_reference(self.root)
        plan.chmod(0o600)
        plan.write_bytes(b"{}\n")
        plan.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "committed bytes"):
            builder._plan_reference(self.root)

        artifact = self.root / "mutable.json"
        artifact.write_bytes(b"{}\n")
        artifact.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "immutable file"):
            builder._reference(artifact, self.root)

    def test_private_closed_world_records_are_relative_to_private_root(self) -> None:
        private = Path(self.temporary.name) / "private"
        private.mkdir()
        artifact = private / "artifact.bin"
        artifact.write_bytes(b"private")
        artifact.chmod(0o444)
        with mock.patch.object(
            builder, "_expected_scan_files", return_value={artifact}
        ):
            summary = builder.closed_world_scan(
                self.root,
                "final",
                scan_root=private,
            )
        self.assertEqual(summary["files"][0]["path"], "artifact.bin")
        self.assertEqual(summary["file_count"], 1)

    def test_consumer_closed_world_scan_rejects_unlisted_live_file(self) -> None:
        scan_root = Path(self.temporary.name) / "handoff"
        scan_root.mkdir()
        expected = set()
        for name in ("a.bin", "b.bin", "omitted-hidden.bin"):
            path = scan_root / name
            path.write_bytes(name.encode("ascii"))
            path.chmod(0o444)
            if name != "omitted-hidden.bin":
                expected.add(path)
        with (
            mock.patch.object(builder, "_expected_scan_files", return_value=expected),
            self.assertRaisesRegex(ValueError, "extra=.*omitted-hidden"),
        ):
            builder.closed_world_scan(
                self.root,
                "phase1",
                scan_root=scan_root,
            )

    def test_baseline_publication_runs_only_after_terminal_validation(self) -> None:
        events: list[str] = []

        def validated(_root: Path) -> dict:
            events.append("validated")
            return {}

        def prepared(_root: Path) -> dict:
            events.append("frozen")
            return {
                "baseline_sha256": "b" * 64,
                "baseline_payload_sha256": "a" * 64,
                "checkpoint_sha256": "c" * 64,
            }

        with (
            mock.patch.object(
                builder, "validate_terminal_prerequisites", side_effect=validated
            ),
            mock.patch.object(
                builder,
                "_prepare_development_baseline_outputs",
                side_effect=prepared,
            ),
        ):
            result = builder.publish_development_baseline_after_terminal_validation(
                self.root
            )
        self.assertEqual(events, ["validated", "frozen"])
        self.assertEqual(result["baseline_sha256"], "b" * 64)
        self.assertEqual(result["checkpoint_sha256"], "c" * 64)

    def test_baseline_preparation_recovers_from_checkpoint_only(self) -> None:
        checkpoint = self.root / "best_development_checkpoint.pt"
        checkpoint.write_bytes(b"candidate weights")
        checkpoint.chmod(0o444)
        checkpoint_sha256 = file_sha256(checkpoint)
        baseline_path = self.root / "development_baseline.json"
        for destination in (checkpoint, baseline_path):
            temporary = builder._protocol_publish_temp(destination)
            temporary.write_bytes(b"interrupted")
            temporary.chmod(0o444)
        written_payload: dict = {}

        def frozen(_manifest: Path, staged: Path) -> dict:
            staged.write_bytes(b"candidate weights")
            staged.chmod(0o444)
            return builder._hash_bound(
                {
                    "copied_checkpoint": {
                        "path": str(staged.resolve(strict=True)),
                        "sha256": checkpoint_sha256,
                        "bytes": staged.stat().st_size,
                        "mode": "0444",
                    }
                }
            )

        def write_baseline(path: Path, payload: dict) -> str:
            written_payload.update(payload)
            raw = builder.canonical_json_bytes(payload) + b"\n"
            path.write_bytes(raw)
            path.chmod(0o444)
            return hashlib.sha256(raw).hexdigest()

        def validated(path: Path) -> dict:
            return {
                "record": {
                    "sha256": file_sha256(path),
                    "payload_sha256": written_payload["payload_sha256"],
                },
                "copied_checkpoint": written_payload["copied_checkpoint"],
            }

        with (
            mock.patch.object(
                builder, "freeze_development_baseline", side_effect=frozen
            ),
            mock.patch.object(
                builder,
                "write_immutable_development_baseline",
                side_effect=write_baseline,
            ),
            mock.patch.object(
                builder,
                "_validate_frozen_development_baseline",
                side_effect=validated,
            ),
        ):
            first = builder._prepare_development_baseline_outputs(self.root)
            second = builder._prepare_development_baseline_outputs(self.root)
        self.assertEqual(checkpoint.read_bytes(), b"candidate weights")
        self.assertEqual(
            written_payload["copied_checkpoint"]["path"],
            str(checkpoint.resolve(strict=True)),
        )
        self.assertEqual(first, second)
        self.assertFalse(os.path.lexists(builder._protocol_publish_temp(checkpoint)))
        self.assertFalse(os.path.lexists(builder._protocol_publish_temp(baseline_path)))

    def test_terminal_envelope_narrows_and_embargoes_claim(self) -> None:
        plan = {"execution_parent_commit": "d" * 40}
        reference = {"path": "artifact", "sha256": "a" * 64}
        semantic = {
            "record": reference,
            "development_manifest": reference,
            "source_checkpoint": reference,
            "copied_checkpoint": reference,
            "selection": {},
        }
        with (
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(
                builder,
                "_require_canonical_monitor_runtime",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(builder, "_validate_all_stage_receipts_and_accounting"),
            mock.patch.object(
                builder,
                "_validate_frozen_development_baseline",
                return_value=semantic,
            ),
            mock.patch.object(builder, "_reference", return_value=reference),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(builder, "_freeze_tree"),
            mock.patch.object(builder, "closed_world_scan", return_value={}),
            mock.patch.object(builder, "_validated_g_commit", return_value="e" * 40),
        ):
            envelope = builder.build_terminal_accounting(self.root)
        self.assertTrue(envelope["all_four_jobs_terminal_and_step_free"])
        self.assertTrue(envelope["exact_registered_root_verified"])
        self.assertFalse(envelope["same_uid_external_compute_excluded"])
        self.assertFalse(envelope["ordinary_batch_children_independently_attested"])
        self.assertTrue(
            envelope["external_sha256_anchor_required_before_performance_claim"]
        )
        self.assertFalse(envelope["performance_claim_ready"])

    def test_terminal_accounting_polls_until_exact_rows_stabilize(self) -> None:
        stage = {
            "held_slurm_job_id": "740999",
            "job_name": "stage",
            "expected_node": "ec51",
        }
        plan = {"accounting": {"poll_timeout_seconds": 180}}
        expected = [{"job_id_raw": "740999"}]
        completed = mock.Mock(stdout="rows")
        with (
            mock.patch.object(builder.subprocess, "run", return_value=completed) as run,
            mock.patch.object(
                builder,
                "_validated_terminal_rows_for_stage",
                side_effect=(ValueError("not terminal"), expected),
            ),
            mock.patch.object(builder.time, "sleep") as sleep,
        ):
            observed = builder._query_terminal_rows_for_stage(plan, stage)
        self.assertEqual(observed, expected)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once()

    def test_terminal_receipt_validator_recomputes_and_cannot_self_authorize(
        self,
    ) -> None:
        receipt = Path(self.temporary.name) / "terminal.json"
        fresh = builder._hash_bound(
            {
                "schema": "r12_acw_development_terminal_accounting_v1",
                "monitor": self._monitor_binding(),
                "performance_claim_ready": False,
            }
        )
        receipt.write_bytes(builder.canonical_json_bytes(fresh) + b"\n")
        receipt.chmod(0o444)
        canonical_receipt = receipt.resolve(strict=True)
        with (
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", canonical_receipt),
            mock.patch.object(builder, "build_terminal_accounting", return_value=fresh),
        ):
            validated = builder.validate_terminal_accounting(
                self.root, canonical_receipt
            )
        self.assertFalse(validated["performance_claim_ready"])

        forged = dict(fresh)
        forged["performance_claim_ready"] = True
        forged = builder._hash_bound(forged)
        receipt.chmod(0o600)
        receipt.write_bytes(builder.canonical_json_bytes(forged) + b"\n")
        receipt.chmod(0o444)
        with (
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", canonical_receipt),
            mock.patch.object(builder, "build_terminal_accounting", return_value=fresh),
        ):
            with self.assertRaisesRegex(ValueError, "differs|self-authorize"):
                builder.validate_terminal_accounting(self.root, canonical_receipt)

    def test_monitor_anchor_requires_completed_j5_and_binds_receipt_log(self) -> None:
        base = Path(self.temporary.name) / "base"
        log_root = base / "logs"
        log_root.mkdir(parents=True)
        receipt_path = Path(self.temporary.name) / "terminal.json"
        receipt_path.write_bytes(b"{}\n")
        receipt_path.chmod(0o444)
        reference = {"path": "artifact", "sha256": "a" * 64}
        receipt = {
            "schema": "r12_acw_development_terminal_accounting_v1",
            "protocol": "R12-ACW-DEVELOPMENT-TERMINAL-ACCOUNTING-v1",
            "development_plan": reference,
            "scientific_commit": "d" * 40,
            "monitor": self._monitor_binding(),
            "stages": {},
            "development_manifest": reference,
            "development_baseline": reference,
            "baseline_checkpoint": reference,
            "semantic_baseline_validation": {},
            "closed_world": {},
            "all_four_jobs_terminal_and_step_free": True,
            "exact_registered_root_verified": True,
            "same_uid_external_compute_excluded": False,
            "ordinary_batch_children_independently_attested": False,
            "claim_limited_to_exact_final_rooted_files_and_slurm_rows": True,
            "resource_values_are_diagnostic_only": True,
            "required_before_any_performance_claim": True,
            "external_sha256_anchor_required_before_performance_claim": True,
            "performance_claim_ready": False,
            "confirmation_authorized": False,
            "promotion_authorized": False,
            "payload_sha256": "e" * 64,
        }
        plan = {
            "accounting": {
                "terminal_receipt": str(receipt_path),
                "monitor_stage": {"held_slurm_job_id": "740999"},
            }
        }
        log_path = log_root / "acw_development_monitor_740999.out"
        log_path.write_bytes(
            (
                f"[acw-development-monitor] complete root={self.root} "
                "performance_claim_ready=0\n"
            ).encode("ascii")
        )
        rows = [{"job_id_raw": "740999", "state": "COMPLETED"}]
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(builder, "BASE", base),
            mock.patch.object(builder, "TERMINAL_ACCOUNTING_PATH", receipt_path),
            mock.patch.object(builder, "_plan_for_root", return_value=plan),
            mock.patch.object(builder, "validate_plan", return_value=plan),
            mock.patch.object(builder, "_validated_g_commit", return_value="d" * 40),
            mock.patch.object(builder, "_plan_reference", return_value=reference),
            mock.patch.object(
                builder, "_load_canonical_json", return_value=(receipt, b"{}\n")
            ),
            mock.patch.object(
                builder,
                "_validate_recorded_monitor_binding",
                return_value=self._monitor_binding(),
            ),
            mock.patch.object(
                builder, "_query_monitor_terminal_rows", return_value=rows
            ),
            mock.patch.object(builder, "_reference", return_value=reference),
        ):
            envelope = builder.build_monitor_anchor_ready_envelope(self.root)
        self.assertEqual(envelope["monitor_terminal_rows"], rows)
        self.assertTrue(envelope["monitor_completed_zero_exit_and_step_free"])
        self.assertFalse(envelope["performance_claim_ready"])
        self.assertEqual(envelope["monitor_log"]["sha256"], file_sha256(log_path))


if __name__ == "__main__":
    unittest.main()
