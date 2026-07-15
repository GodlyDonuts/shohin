#!/usr/bin/env python3
"""Focused tests for the frozen WGRQ Stage-A scorer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from pipeline import score_wgrq_falsifier_v1 as scorer
except ImportError:
    import score_wgrq_falsifier_v1 as scorer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def symbolic_audit(*, reported_pass: bool = True) -> dict:
    scales = {}
    for n in scorer.SYMBOLIC_SCALES:
        scales[str(n)] = {
            "check_count": scorer.minimum_symbolic_checks(n),
            "quotient_class_count": 2 ** (n - 1),
            "quotient_class_size": 2,
            "maximum_shortest_witness_depth": n - 2,
            "gates": {gate: True for gate in scorer.SYMBOLIC_SCALE_GATES},
        }
    return {
        "schema": scorer.SYMBOLIC_SCHEMA,
        "passed": reported_pass,
        "scales": scales,
        "cancellation_controls": {gate: True for gate in scorer.CANCELLATION_GATES},
        "generation_controls": {gate: True for gate in scorer.GENERATION_GATES},
    }


def protocol_gates() -> dict:
    return {gate: True for gate in scorer.PROTOCOL_GATES}


def build_manifest(directory: Path, *, episodes_per_stratum: int) -> tuple[Path, dict]:
    symbolic_path = directory / "symbolic.json"
    write_json(symbolic_path, symbolic_audit())
    fits = []
    for arm in scorer.ARMS:
        for seed in scorer.SEEDS:
            checkpoint_path = directory / f"checkpoint-{arm}-{seed}.bin"
            checkpoint_path.write_bytes(f"checkpoint:{arm}:{seed}\n".encode("ascii"))
            checkpoint_hash = sha256(checkpoint_path)
            exact = int(arm in ("wgrq_shortest", "privileged_edge"))
            evaluation = {
                "schema": scorer.EVALUATION_SCHEMA,
                "arm": arm,
                "seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "protocol_gates": protocol_gates(),
                "strata": {
                    stratum: {
                        "episodes": [
                            {
                                "committed_episode_id": f"{stratum}:{index:04d}",
                                "episode_exact": exact,
                            }
                            for index in range(episodes_per_stratum)
                        ]
                    }
                    for stratum in scorer.STRATA
                },
            }
            evaluation_path = directory / f"evaluation-{arm}-{seed}.json"
            write_json(evaluation_path, evaluation)
            fits.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": {
                        "path": checkpoint_path.name,
                        "sha256": checkpoint_hash,
                    },
                    "evaluation": {
                        "path": evaluation_path.name,
                        "sha256": sha256(evaluation_path),
                    },
                }
            )
    manifest = {
        "schema": scorer.MANIFEST_SCHEMA,
        "symbolic_audit": {"path": symbolic_path.name, "sha256": sha256(symbolic_path)},
        "fits": fits,
    }
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


class WgrqScorerTests(unittest.TestCase):
    def test_frozen_contract_constants(self) -> None:
        self.assertEqual(len(scorer.SEEDS), 12)
        self.assertEqual(len(scorer.ARMS), 5)
        self.assertEqual(scorer.EXPECTED_FITS, 60)
        self.assertEqual(scorer.EPISODES_PER_STRATUM, 1_024)
        self.assertEqual(scorer.BOOTSTRAP_REPLICATES, 20_000)
        self.assertEqual(scorer.PAIRED_SEED_REQUIRED, 10)
        parser_dests = {action.dest for action in scorer.build_parser()._actions}
        self.assertEqual(parser_dests, {"help", "manifest", "out"})

    def test_deterministic_bootstrap_is_arm_paired(self) -> None:
        values = np.zeros((len(scorer.STRATA), len(scorer.ARMS), 3, 5), dtype=np.uint8)
        values[:, scorer.ARMS.index("wgrq_shortest")] = 1
        values[:, scorer.ARMS.index("active_answer_only")] = 1
        values[:, scorer.ARMS.index("privileged_edge")] = 1
        first = scorer.paired_two_way_bootstrap(values, replicates=257, rng_seed=12345)
        second = scorer.paired_two_way_bootstrap(values, replicates=257, rng_seed=12345)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 257)
        self.assertEqual(first["lower_order_statistic_index_zero_based"], 12)
        self.assertAlmostEqual(first["simultaneous_g_lower_bound"], -0.05)
        self.assertEqual(
            first["resample_counts_sha256"], second["resample_counts_sha256"]
        )
        self.assertEqual(first["replicate_g_sha256"], second["replicate_g_sha256"])
        self.assertEqual(
            first["resample_counts_sha256"],
            "3db9f16255051234a263c6173adc7f7d7440d5e65809de965f10218a4ee824df",
        )
        self.assertEqual(
            first["replicate_g_sha256"],
            "8b64a02af6c8173a8bc8a527d184c438c93261a569f6d917bf9076daa276f3ed",
        )

    def test_point_gates_include_privileged_and_ten_of_twelve_rule(self) -> None:
        episodes = 100
        values = np.zeros(
            (len(scorer.STRATA), len(scorer.ARMS), len(scorer.SEEDS), episodes),
            dtype=np.uint8,
        )
        values[:, scorer.ARMS.index("wgrq_shortest")] = 1
        values[:, scorer.ARMS.index("privileged_edge")] = 1
        active = scorer.ARMS.index("active_answer_only")
        full = scorer.STRATA.index("full_ood")
        values[full, active] = 1
        values[full, active, :10, :6] = 0

        result = scorer.compute_point_results(values)
        self.assertTrue(result["privileged_ceiling"]["passed"])
        self.assertEqual(result["full_ood_paired_seed_rule"]["wins"], 10)
        self.assertTrue(result["full_ood_paired_seed_rule"]["passed"])

        values[full, active, 9, :6] = 1
        result = scorer.compute_point_results(values)
        self.assertEqual(result["full_ood_paired_seed_rule"]["wins"], 9)
        self.assertFalse(result["full_ood_paired_seed_rule"]["passed"])

        values[:, scorer.ARMS.index("privileged_edge"), :, :2] = 0
        result = scorer.compute_point_results(values)
        self.assertFalse(result["privileged_ceiling"]["passed"])

    def test_complete_sixty_fit_manifest_scores_go_with_20000_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, _ = build_manifest(directory, episodes_per_stratum=2)
            with mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "GO")
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(decision["verification"]["fits_verified"], 60)
        self.assertEqual(decision["verification"]["checkpoint_hashes_verified"], 60)
        self.assertEqual(decision["verification"]["evaluation_hashes_verified"], 60)
        self.assertTrue(decision["verification"]["paired_seed_episode_matrix_complete"])
        self.assertEqual(decision["bootstrap"]["replicates"], 20_000)
        self.assertEqual(
            decision["bootstrap"]["lower_order_statistic_index_zero_based"], 999
        )
        self.assertGreater(decision["bootstrap"]["simultaneous_g_lower_bound"], 0.0)
        self.assertEqual(
            decision["point_results"]["full_ood_paired_seed_rule"]["wins"], 12
        )
        self.assertFalse(decision["score_dependent_fallback_used"])

    def test_checkpoint_hash_corruption_rejects_without_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, manifest = build_manifest(directory, episodes_per_stratum=2)
            checkpoint_path = directory / manifest["fits"][0]["checkpoint"]["path"]
            checkpoint_path.write_bytes(b"corrupted after manifest freeze\n")
            with mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("artifact_hash_mismatch", decision["reasons"])
        self.assertEqual(decision["bootstrap"]["status"], "not_run_evidence_failure")
        self.assertEqual(decision["bootstrap"]["replicates"], 0)
        self.assertFalse(decision["bootstrap"]["score_dependent_fallback_used"])

    def test_evaluation_hash_corruption_rejects_without_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, manifest = build_manifest(directory, episodes_per_stratum=2)
            evaluation_path = directory / manifest["fits"][0]["evaluation"]["path"]
            evaluation_path.write_bytes(evaluation_path.read_bytes() + b" ")
            with mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("artifact_hash_mismatch", decision["reasons"])
        self.assertEqual(decision["bootstrap"]["status"], "not_run_evidence_failure")

    def test_committed_episode_id_mismatch_rejects_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, manifest = build_manifest(directory, episodes_per_stratum=2)
            fit = manifest["fits"][-1]
            evaluation_path = directory / fit["evaluation"]["path"]
            evaluation = json.loads(evaluation_path.read_text())
            evaluation["strata"]["full_ood"]["episodes"][0]["committed_episode_id"] = (
                "full_ood:different-commitment"
            )
            write_json(evaluation_path, evaluation)
            fit["evaluation"]["sha256"] = sha256(evaluation_path)
            write_json(manifest_path, manifest)
            with mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("committed_episode_pairing_mismatch", decision["reasons"])
        self.assertEqual(decision["bootstrap"]["replicates"], 0)

    def test_symbolic_false_is_an_absolute_gate(self) -> None:
        audit = symbolic_audit()
        audit["scales"]["6"]["gates"]["canonical_serialization_exact"] = False
        result = scorer.verify_symbolic_audit(audit)
        self.assertFalse(result["passed"])
        self.assertIn("n6:canonical_serialization_exact", result["failures"])

    def test_symbolic_failure_does_not_change_bootstrap_method(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, manifest = build_manifest(directory, episodes_per_stratum=2)
            symbolic_path = directory / manifest["symbolic_audit"]["path"]
            failed_audit = symbolic_audit()
            failed_audit["generation_controls"][
                "unavoidable_parity_obstruction_reported"
            ] = False
            write_json(symbolic_path, failed_audit)
            manifest["symbolic_audit"]["sha256"] = sha256(symbolic_path)
            write_json(manifest_path, manifest)
            with mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("symbolic_gate_failed", decision["reasons"])
        self.assertEqual(decision["bootstrap"]["replicates"], 20_000)
        self.assertFalse(decision["score_dependent_fallback_used"])

    def test_unexpected_scoring_failure_rejects_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, _ = build_manifest(directory, episodes_per_stratum=2)
            with (
                mock.patch.object(scorer, "EPISODES_PER_STRATUM", 2),
                mock.patch.object(
                    scorer,
                    "paired_two_way_bootstrap",
                    side_effect=RuntimeError("forced failure"),
                ),
            ):
                decision = scorer.score_manifest(manifest_path)
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertIn("scoring_execution_failed", decision["reasons"])
        self.assertEqual(decision["bootstrap"]["replicates"], 0)
        self.assertFalse(decision["bootstrap"]["score_dependent_fallback_used"])

    def test_immutable_decision_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            destination = Path(raw_directory) / "decision.json"
            payload = {"schema": scorer.DECISION_SCHEMA, "decision": "NO_GO"}
            expected = hashlib.sha256(scorer._json_bytes(payload)).hexdigest()
            actual = scorer.write_immutable_json(destination, payload)
            original = destination.read_bytes()
            self.assertEqual(actual, expected)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o444)
            with self.assertRaises(FileExistsError):
                scorer.write_immutable_json(destination, {"decision": "GO"})
            self.assertEqual(destination.read_bytes(), original)
            self.assertFalse(os.access(destination, os.W_OK))


if __name__ == "__main__":
    unittest.main()
