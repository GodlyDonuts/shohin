#!/usr/bin/env python3
"""Adversarial tests for the independent PCPT v3 verifier."""

from __future__ import annotations

import ast
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from pipeline import audit_post_commit_packet_transport_v3 as audit
from pipeline import post_commit_packet_transport_falsifier as parent


NONCE_HEX = "17" * 32


def _fixture_identity() -> dict[str, Any]:
    paths = {
        path: hashlib.sha256(f"fixture:{path}".encode("utf-8")).hexdigest()
        for path in audit.SCIENTIFIC_PATHS
    }
    paths["pipeline/audit_post_commit_packet_transport_v3.py"] = audit.sha256_bytes(
        Path(audit.__file__).read_bytes()
    )
    return {
        "scientific_commit": "0" * 40,
        "scientific_paths": paths,
        "scientific_source_tree_sha256": audit.sha256_bytes(
            audit.canonical_json_bytes(paths)
        ),
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.11.0",
            "platform": "independent-audit-fixture",
        },
        "role_environment_policy": copy.deepcopy(audit.ENVIRONMENT_POLICY),
        "role_environment_policy_sha256": audit.sha256_bytes(
            audit.canonical_json_bytes(audit.ENVIRONMENT_POLICY)
        ),
        "verified_against_head": True,
    }


def _fixture_command() -> audit.CommandContract:
    return audit.CommandContract(
        audit.SANDBOX_EXEC,
        audit.normalized_sandbox_profile(),
        audit.ROLE_PYTHON,
        str(Path(audit.__file__).resolve().parent / "post_commit_packet_transport_roles.py"),
    )


def _pending_core(report: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(report)
    result["deterministic_replay"] = {
        "status": "pending_second_full_run",
        "first_core_payload_sha256": None,
        "second_core_payload_sha256": None,
        "second_core_report": None,
    }
    result["gates"]["full_deterministic_replay_byte_identical"] = False
    result["pass"] = False
    result.pop("payload_sha256", None)
    result["payload_sha256"] = audit.sha256_bytes(
        audit.canonical_json_bytes(result)
    )
    return result


def _finalize_for_mutation_tests(core: dict[str, Any]) -> dict[str, Any]:
    pending = _pending_core(core)
    final = copy.deepcopy(pending)
    final["deterministic_replay"] = {
        "status": "confirmed_byte_identical",
        "first_core_payload_sha256": pending["payload_sha256"],
        "second_core_payload_sha256": pending["payload_sha256"],
        "second_core_report": copy.deepcopy(pending),
    }
    final["gates"]["full_deterministic_replay_byte_identical"] = True
    final["pass"] = all(final["gates"].values())
    final.pop("payload_sha256", None)
    final["payload_sha256"] = audit.sha256_bytes(
        audit.canonical_json_bytes(final)
    )
    return final


class IndependentPacketTransportV3AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = _fixture_identity()
        cls.command = _fixture_command()
        core = audit._expected_core(cls.identity, cls.command, NONCE_HEX)
        cls.report = _finalize_for_mutation_tests(core)
        cls.summary = audit.verify_artifact(cls.report, verify_git=False)

    def _forge(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        core = _pending_core(self.report)
        mutator(core)
        return _finalize_for_mutation_tests(core)

    def _assert_rejected(
        self,
        mutator: Callable[[dict[str, Any]], None],
        message: str | None = None,
    ) -> None:
        forged = self._forge(mutator)
        with self.assertRaises(audit.AuditFailure) as caught:
            audit.verify_artifact(forged, verify_git=False)
        if message is not None:
            self.assertIn(message, str(caught.exception))

    def test_reference_report_reconstructs_every_role(self) -> None:
        self.assertEqual(self.summary["status"], "evidence_reconstructed")
        self.assertTrue(self.summary["evidence_reconstruction_pass"])
        self.assertEqual(self.summary["role_invocations_per_core"], 337)
        self.assertEqual(
            self.report["role_invocation_counts"], audit.EXPECTED_ROLE_COUNTS
        )
        second = self.report["deterministic_replay"]["second_core_report"]
        self.assertEqual(len(second["role_invocations"]), 337)

    def test_independent_publisher_writes_and_binds_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            artifact = root / "artifact.json"
            receipt = root / "receipt.json"
            candidate.write_bytes(audit.canonical_json_bytes(self.report))
            result = audit.publish_verified_artifact(
                candidate,
                artifact,
                receipt,
                verify_git=False,
                require_canonical=False,
            )
            self.assertEqual(result["status"], "evidence_reconstructed_and_published")
            self.assertEqual(result["artifact_sha256"], audit.sha256_bytes(artifact.read_bytes()))
            self.assertEqual(result["verifier_source_sha256"], result["verifier_scientific_path_sha256"])
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o444)
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o444)
            replayed = audit.verify_published_artifact_and_receipt(
                artifact,
                receipt,
                verify_git=False,
                require_canonical=False,
            )
            self.assertEqual(replayed, result)
            with self.assertRaises(audit.AuditFailure):
                audit.publish_verified_artifact(
                    candidate,
                    artifact,
                    receipt,
                    verify_git=False,
                    require_canonical=False,
                )

    def test_rejects_self_rehashed_publication_receipt_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            artifact = root / "artifact.json"
            receipt = root / "receipt.json"
            candidate.write_bytes(audit.canonical_json_bytes(self.report))
            valid = audit.publish_verified_artifact(
                candidate,
                artifact,
                receipt,
                verify_git=False,
                require_canonical=False,
            )
            mutations = {
                "artifact_sha256": "0" * 64,
                "artifact_bytes": valid["artifact_bytes"] + 1,
                "receipt_payload_sha256": "f" * 64,
            }
            for field, replacement in mutations.items():
                with self.subTest(field=field):
                    forged = copy.deepcopy(valid)
                    forged[field] = replacement
                    if field != "receipt_payload_sha256":
                        forged.pop("receipt_payload_sha256")
                        forged["receipt_payload_sha256"] = audit.sha256_bytes(
                            audit.canonical_json_bytes(forged)
                        )
                    receipt.chmod(0o600)
                    receipt.write_bytes(audit.canonical_json_bytes(forged))
                    receipt.chmod(0o444)
                    with self.assertRaises(audit.AuditFailure):
                        audit.verify_published_artifact_and_receipt(
                            artifact,
                            receipt,
                            verify_git=False,
                            require_canonical=False,
                        )

    def test_auditor_has_no_implementation_import(self) -> None:
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        forbidden = {
            "pipeline.post_commit_packet_transport_falsifier",
            "pipeline.post_commit_packet_transport_roles",
        }
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertTrue(forbidden.isdisjoint(imported), imported & forbidden)

    def test_cross_implementation_public_contract_constants_match(self) -> None:
        self.assertEqual(parent.AUDIT_ID, audit.AUDIT_ID)
        self.assertEqual(parent.PROTOCOL_ID, audit.PROTOCOL_ID)
        self.assertEqual(parent.SCHEMA_VERSION, audit.SCHEMA_VERSION)
        self.assertEqual(parent.SCIENTIFIC_PATHS, audit.SCIENTIFIC_PATHS)
        self.assertEqual(parent.REPORT_FIELDS, audit.REPORT_FIELDS)
        self.assertEqual(parent.FROZEN_GATE_NAMES, audit.GATE_NAMES)
        self.assertEqual(parent.EXPECTED_PUBLIC_LAYOUT, audit.EXPECTED_PUBLIC_LAYOUT)
        self.assertEqual(
            parent.EXPECTED_DECISIVE_LAYOUT, audit.EXPECTED_DECISIVE_LAYOUT
        )
        self.assertEqual(parent.EXPECTED_ROLE_COUNTS, audit.EXPECTED_ROLE_COUNTS)
        self.assertEqual(
            parent.SANDBOX_PROBE_CHECK_NAMES, audit.SANDBOX_PROBE_CHECKS
        )
        self.assertEqual(
            parent.normalized_sandbox_profile(), audit.normalized_sandbox_profile()
        )
        self.assertEqual(parent.CLAIM_BOUNDARY, audit.CLAIM_BOUNDARY)

    def test_rejects_self_rehashed_forged_exit_code(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            updater = next(
                row
                for row in core["role_invocations"]
                if row["role"] == "updater" and row["exit_code"] == 0
            )
            updater["exit_code"] = 99

        self._assert_rejected(mutate, "exit_code")

    def test_rejects_self_rehashed_role_path_forgery(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            for row in core["role_invocations"]:
                row["command"][6] = (
                    "/tmp/forged/pipeline/post_commit_packet_transport_roles.py"
                )

        forged = self._forge(mutate)
        with self.assertRaises(audit.AuditFailure) as caught:
            audit.verify_artifact(forged, verify_git=False)
        self.assertIn("frozen role executable", str(caught.exception))

    def test_nonce_seed_uses_frozen_concatenation_contract(self) -> None:
        manifest = "a" * 64
        expected = int.from_bytes(
            hashlib.sha256(
                audit.PRIMARY_CHALLENGE_DOMAIN.encode("ascii")
                + manifest.encode("ascii")
                + bytes.fromhex(NONCE_HEX)
            ).digest()[:8],
            "big",
        )
        self.assertEqual(
            audit._derive_nonce_seed(
                manifest, NONCE_HEX, audit.PRIMARY_CHALLENGE_DOMAIN
            ),
            expected,
        )

    def test_rejects_self_rehashed_extra_cwd_file(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            updater = next(
                row
                for row in core["role_invocations"]
                if row["role"] == "updater" and row["exit_code"] == 0
            )
            updater["cwd_regular_files_before"].append("secret_source.jsonl")
            updater["cwd_regular_files_after"].append("secret_source.jsonl")
            updater["cwd_regular_files_before"].sort()
            updater["cwd_regular_files_after"].sort()

        self._assert_rejected(mutate, "cwd_regular_files_after")

    def test_rejects_self_rehashed_bogus_packet_hash(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            updater = next(
                row
                for row in core["role_invocations"]
                if row["role"] == "updater" and row["exit_code"] == 0
            )
            updater["file_outputs"][0]["sha256"] = "0" * 64

        self._assert_rejected(mutate, "sha256")

    def test_rejects_self_rehashed_identity_recoding(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            core["public_results"][0]["output_permutation"] = list(range(audit.MODULUS))

        self._assert_rejected(mutate, "output_permutation")

    def test_rejects_self_rehashed_missing_evidence_field(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            del core["role_invocations"][0]["stdout_sha256"]

        self._assert_rejected(mutate, "schema")

    def test_rejects_self_rehashed_replay_evidence_erasure(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            core["role_invocations"] = []
            core["role_invocation_counts"] = {}

        self._assert_rejected(mutate, "nonempty")

    def test_rejects_self_rehashed_sandbox_profile_forgery(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            core["role_invocations"][0]["sandbox_profile_sha256"] = "0" * 64

        self._assert_rejected(mutate, "sandbox_profile_sha256")

    def test_rejects_self_rehashed_sandbox_probe_forgery(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            core["sandbox_probe"]["checks"]["parent_read_blocked"] = False
            checks = core["sandbox_probe"]["checks"]
            core["sandbox_probe"]["stdout_sha256"] = audit.sha256_bytes(
                audit.canonical_json_bytes(checks)
            )

        self._assert_rejected(mutate, "failed confinement")

    def test_rejects_self_rehashed_nonce_binding_forgery(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            replacement = "23" * 32
            core["config"]["challenge_nonce_hex"] = replacement
            proof = core["phase_two"]["phase_order_proof"]
            proof["challenge_nonce_hex"] = replacement
            proof["challenge_nonce_commitment_sha256"] = audit.sha256_bytes(
                (replacement + "\n").encode("ascii")
            )

        self._assert_rejected(mutate, "challenge_seed")

    def test_rejects_absent_sandbox_probe(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            del core["sandbox_probe"]

        self._assert_rejected(mutate, "schema mismatch")

    def test_rejects_absent_nonce(self) -> None:
        def mutate(core: dict[str, Any]) -> None:
            del core["config"]["challenge_nonce_hex"]

        self._assert_rejected(mutate, "challenge nonce")


if __name__ == "__main__":
    unittest.main()
