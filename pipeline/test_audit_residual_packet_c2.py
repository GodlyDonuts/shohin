#!/usr/bin/env python3
"""Adversarial toy-only tests for independent RSP-C2 admission."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import audit_residual_packet_c2 as audit
import generate_residual_packet_c2 as generate
import test_generate_residual_packet_c2 as generator_fixtures


TOY_LABEL = "TOY_ONLY_NEVER_PRODUCTION_AUDITOR_FIXTURE_B"


class ResidualPacketC2AuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate.build_toy_fixture_bundle(TOY_LABEL)

    def audit(self, bundle: dict) -> dict:
        return audit.audit_toy_fixture_bundle(bundle, TOY_LABEL)

    def test_independent_replay_admits_the_untampered_toy_bundle(self):
        report = self.audit(copy.deepcopy(self.bundle))
        self.assertTrue(report["admitted"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["schema"], "toy_only_rsp_c2_admission_audit_fixture_v1")

    def test_rejects_board_schema_and_canonical_row_hash_tampering(self):
        tampered = copy.deepcopy(self.bundle)
        tampered["board"]["schema"] = "toy_only_wrong_board_schema"
        tampered["board"]["rows"][0]["answer"] += 1
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("board_schema", report["failures"])
        self.assertIn("board_rows_hash", report["failures"])
        self.assertIn("board_replay", report["failures"])
        self.assertIn("board_answer", report["failures"])

    def test_rejects_board_training_semantic_overlap(self):
        tampered = copy.deepcopy(self.bundle)
        _, _, _, _, programs = audit.expected_toy_bundle(
            TOY_LABEL,
            tokenizer=audit.ToyHashTokenizer(),
            per_stratum=4,
            length_counts={2: 4, 3: 8, 4: 4},
        )
        program = programs[0]
        row = tampered["board"]["rows"][0]
        row.update(
            {
                "answer": program["final_answer"],
                "initial_state": program["initial_state"],
                "operations": copy.deepcopy(program["operations"]),
                "packet": program["packet"],
                "source": program["source"],
                "template_id": program["template_id"],
                "trajectory": copy.deepcopy(program["trajectory"]),
            }
        )
        tampered["board"]["rows_sha256"] = audit.digest(
            audit.canonical_json_bytes(tampered["board"]["rows"])
        )
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("board_training_disjointness", report["failures"])

    def test_rejects_token_parity_and_manifest_hash_mismatch(self):
        tampered = copy.deepcopy(self.bundle)
        compiler = next(row for row in tampered["sham"] if row["kind"] == "compiler")
        compiler["response"] += " TOY_ONLY_EXTRA_TOKEN"
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("token_parity", report["failures"])
        self.assertIn("sham_hash", report["failures"])
        self.assertIn("sham_replay", report["failures"])

    def test_rejects_self_mapped_or_nonpermutation_sham(self):
        tampered = copy.deepcopy(self.bundle)
        treatment_compiler = next(
            row for row in tampered["treatment"] if row["kind"] == "compiler"
        )
        sham_compiler = next(
            row for row in tampered["sham"] if row["kind"] == "compiler"
        )
        sham_compiler["response"] = treatment_compiler["response"]
        sham_compiler["response_program_id"] = treatment_compiler["program_id"]
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("sham_derangement", report["failures"])

    def test_rejects_nonidentical_updater_rows(self):
        tampered = copy.deepcopy(self.bundle)
        updater = next(row for row in tampered["sham"] if row["kind"] == "updater")
        updater["response"] += " TOY_ONLY_MUTATION"
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("updater_byte_parity", report["failures"])

    def test_rejects_mathematically_correct_updater_observation(self):
        tampered = copy.deepcopy(self.bundle)
        index = next(
            index
            for index, row in enumerate(tampered["treatment"])
            if row["kind"] == "updater"
        )
        row = tampered["treatment"][index]
        match = audit.UPDATER_PROMPT_RE.fullmatch(row["completion_prompt"])
        self.assertIsNotNone(match)
        state, operations = audit.parse_packet(match.group("packet"))
        correct = audit.execute(state, operations[0])
        mutated_prompt = row["completion_prompt"].replace(
            f"Observation: {match.group('observed')}", f"Observation: {correct}"
        )
        for arm in (tampered["treatment"], tampered["sham"]):
            arm[index]["completion_prompt"] = mutated_prompt
            arm[index]["question"] = mutated_prompt
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("updater_observation_contract", report["failures"])

    def test_rejects_row_schema_extensions(self):
        tampered = copy.deepcopy(self.bundle)
        tampered["treatment"][0]["toy_unregistered_field"] = TOY_LABEL
        report = self.audit(tampered)
        self.assertFalse(report["admitted"])
        self.assertIn("row_schema", report["failures"])

    def test_json_decoders_reject_duplicate_keys_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit.decode_json(b'{"fixture":"A","fixture":"B"}\n', TOY_LABEL)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audit.decode_json(b'{"fixture":NaN}\n', TOY_LABEL)

    def test_independent_seed_receipt_replay_rejects_freeze_substitution(self):
        receipt, _, _ = generator_fixtures.toy_seed_receipt()
        integers, commitments = audit.inspect_seed_receipt(receipt)
        self.assertEqual(set(integers), set(audit.SEED_LABELS))
        self.assertEqual(len(set(commitments.values())), len(audit.SEED_LABELS))
        tampered = copy.deepcopy(receipt)
        tampered["freeze"]["remote_url"] = "https://github.com/example/other-audit.git"
        with self.assertRaisesRegex(ValueError, "does not replay"):
            audit.inspect_seed_receipt(tampered)

    def test_exclusive_audit_writer_is_read_only_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / f"{TOY_LABEL}_audit.json"
            payload = audit.canonical_json_bytes(
                {"fixture_label": TOY_LABEL, "schema": "toy_only_audit_write_v1"}
            )
            observed = audit.exclusive_write(destination, payload)
            self.assertEqual(observed, hashlib.sha256(payload).hexdigest())
            self.assertEqual(destination.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                audit.exclusive_write(destination, payload)

    def test_production_auditor_has_no_generator_import(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        self.assertNotIn("generate_residual_packet_c2", imports)
        self.assertFalse(any("residual_packet_v1" in name for name in imports))
        self.assertFalse(
            {name.split(".", 1)[0] for name in imports}
            & {"http", "requests", "socket", "urllib", "urllib3"}
        )

    def test_production_cli_requires_exact_receipt_and_artifact_hashes(self):
        parser = audit.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        board_options = {
            option
            for action in subparsers.choices["board"]._actions
            for option in action.option_strings
        }
        data_options = {
            option
            for action in subparsers.choices["data"]._actions
            for option in action.option_strings
        }
        for options in (board_options, data_options):
            self.assertIn("--prerequisite-receipt-sha256", options)
            self.assertIn("--seed-receipt-sha256", options)
            self.assertIn("--provenance-receipt-sha256", options)
            self.assertNotIn("--seed", options)
        for option in (
            "--treatment-sha256",
            "--sham-sha256",
            "--manifest-sha256",
            "--board-audit-sha256",
        ):
            self.assertIn(option, data_options)

    def test_toy_report_is_json_serializable_without_artifact_content(self):
        report = self.audit(copy.deepcopy(self.bundle))
        rendered = json.dumps(report, sort_keys=True)
        self.assertIn(TOY_LABEL, rendered)
        self.assertNotIn(self.bundle["board"]["rows"][0]["source"], rendered)


if __name__ == "__main__":
    unittest.main()
