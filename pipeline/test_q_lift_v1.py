#!/usr/bin/env python3
"""Adversarial CPU-only tests for the frozen Q-LIFT v1 package."""

from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import audit_q_lift_v1 as auditor
import generate_q_lift_v1 as generator


class QLiftV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = generator.board_payload()

    def test_board_is_deterministic_and_frozen(self):
        second = generator.board_payload()
        self.assertEqual(self.board, second)
        self.assertEqual(self.board["schema"], generator.SCHEMA)
        self.assertEqual(
            generator.digest_bytes(generator.pretty_json_bytes(self.board)),
            generator.EXPECTED_BOARD_SHA256,
        )
        self.assertEqual(self.board["content_sha256"], generator.EXPECTED_CONTENT_SHA256)
        generator.assert_frozen_board(self.board)
        self.assertEqual(len(self.board["cases"]), 32)
        self.assertEqual(
            Counter(case["length"] for case in self.board["cases"]),
            Counter({4: 8, 8: 8, 16: 8, 32: 8}),
        )

    def test_exact_quotient_lower_bound_and_closure(self):
        self.assertEqual(generator.FIELD ** generator.STATE_DIM, 289)
        self.assertEqual(generator.state_bits(), 9)
        self.assertEqual(self.board["config"]["state_fixed_bits"], 9)
        for case in self.board["cases"]:
            projection = case["task"]["projection"]
            world = generator.final_world(case["context"])
            state = generator.fold_state(projection, case["context"])
            self.assertEqual(state, generator.matrix_vector(projection, world))
            self.assertEqual(state, case["state"])
            self.assertEqual(generator.state_code(state), case["state_code"])
            for event in case["context"]["events"] + [case["in_family_query"]["future_event"]]:
                self.assertEqual(
                    generator.matrix_multiply(projection, event["matrix"]),
                    generator.matrix_multiply(event["state_matrix"], projection),
                )
                self.assertEqual(
                    generator.matrix_vector(projection, event["bias"]),
                    event["state_bias"],
                )
            self.assertEqual(
                generator.answer_in_family(case["state"], case["in_family_query"]),
                case["in_family_query"]["answer"],
            )

    def test_archive_is_reversible_and_all_retrieval_channels_are_charged(self):
        for case in self.board["cases"]:
            archive = case["archive"]
            self.assertEqual(generator.restore_context(archive), case["context"])
            address_bits = max(1, (archive["packet_count"] - 1).bit_length())
            self.assertEqual(archive["address_bits_per_read"], address_bits)
            self.assertEqual(
                archive["full_retrieval_bits"],
                archive["payload_bits"] + address_bits * archive["packet_count"],
            )
            self.assertEqual(case["accounting"]["in_family_retrieval_bits"], 0)
            self.assertEqual(
                generator.answer_out_of_family(case["archive"], case["out_of_family_query"]),
                case["out_of_family_query"]["answer"],
            )

    def test_merge_and_split_witnesses_are_causal(self):
        for control in self.board["controls"]["merge"]:
            self.assertNotEqual(control["left_context"], control["right_context"])
            self.assertEqual(control["left_state"], control["right_state"])
            self.assertEqual(
                generator.dot(control["coefficients"], control["left_state"]),
                control["shared_answer"],
            )
            self.assertEqual(
                generator.dot(control["coefficients"], control["right_state"]),
                control["shared_answer"],
            )
        for control in self.board["controls"]["split"]:
            self.assertNotEqual(control["left_state"], control["right_state"])
            self.assertNotEqual(control["left_answer"], control["right_answer"])
            self.assertEqual(
                generator.dot(control["coefficients"], control["left_state"]),
                control["left_answer"],
            )
            self.assertEqual(
                generator.dot(control["coefficients"], control["right_state"]),
                control["right_answer"],
            )

    def test_capacity_matched_prompt_prefix_copy_has_exact_collisions(self):
        self.assertEqual(
            (generator.FIELD ** generator.COPY_COORDINATES - 1).bit_length(),
            generator.state_bits(),
        )
        for control in self.board["controls"]["copy"]:
            left = control["left_context"]["initial"]
            right = control["right_context"]["initial"]
            self.assertEqual(left[: generator.COPY_COORDINATES], right[: generator.COPY_COORDINATES])
            self.assertEqual(control["copied_prefix"], left[: generator.COPY_COORDINATES])
            self.assertEqual(control["copied_prefix_bits"], generator.state_bits())
            self.assertNotEqual(control["left_answer"], control["right_answer"])
            self.assertNotEqual(control["left_state"], control["right_state"])
        metrics = self.board["reference_metrics"]
        self.assertEqual(metrics["copy_witness_collisions"], generator.PAIR_COUNT)
        self.assertLess(metrics["copy_prefix_zero_fill"]["correct"], 8)

    def test_archive_swap_and_state_swap_follow_only_the_declared_channel(self):
        for control in self.board["controls"]["swap"]:
            # Archive swap: the sealed in-family answer follows the retained
            # state even when the attached cold archive belongs to its donor.
            left_root = generator.dot(control["coefficients"], control["left_state"])
            self.assertEqual(left_root, control["left_answer"])
            self.assertEqual(
                generator.restore_context(control["right_archive"]),
                control["right_context"],
            )

            # State swap: with the left archive held fixed, the sealed answer
            # follows the substituted state, not the archive.
            swapped_root = generator.dot(control["coefficients"], control["right_state"])
            self.assertEqual(swapped_root, control["right_answer"])
            self.assertNotEqual(swapped_root, left_root)

            # Once retrieval is explicitly allowed, the out-of-family answer
            # follows the archive and not the retained state.
            query = {"coefficients": control["out_coefficients"]}
            self.assertEqual(
                generator.answer_out_of_family(control["left_archive"], query),
                control["left_out_answer"],
            )
            self.assertEqual(
                generator.answer_out_of_family(control["right_archive"], query),
                control["right_out_answer"],
            )
            self.assertNotEqual(control["left_out_answer"], control["right_out_answer"])

    def test_index_control_exhausts_every_seven_bit_collision(self):
        control = self.board["controls"]["index"]
        self.assertEqual(control["n"], 8)
        self.assertEqual(control["exact_fixed_state_lower_bound_bits"], 8)
        self.assertEqual(control["underbudget_bits"], 7)
        self.assertEqual(len(control["witnesses"]), 128)
        prefixes = set()
        for witness in control["witnesses"]:
            self.assertEqual(witness["left"][:-1], witness["right"][:-1])
            self.assertNotEqual(
                witness["left"][witness["query"]],
                witness["right"][witness["query"]],
            )
            prefixes.add(tuple(witness["copied_prefix"]))
        self.assertEqual(len(prefixes), 128)
        self.assertEqual(control["prefix_baseline_average_error"]["7"], 1 / 16)
        self.assertEqual(control["prefix_baseline_average_error"]["8"], 0)

    def test_reference_controls_do_not_create_a_capability_claim(self):
        metrics = self.board["reference_metrics"]
        self.assertEqual(metrics["analytic_quotient"], {"correct": 32, "retrieval_bits": 0, "total": 32})
        self.assertEqual(metrics["retrieval_only"]["correct"], 32)
        self.assertGreater(metrics["retrieval_only"]["retrieval_bits"], 0)
        self.assertLess(metrics["sham_projection"]["correct"], 8)
        self.assertEqual(
            self.board["claim_status"],
            "cpu_falsifier_only_no_capability_or_novelty_claim",
        )

    def test_independent_auditor_admits_exact_board_and_rejects_tampering(self):
        source = inspect.getsource(auditor)
        self.assertNotIn("import generate_q_lift_v1", source)
        self.assertNotIn("from generate_q_lift_v1", source)
        with tempfile.TemporaryDirectory() as directory:
            board_path = Path(directory) / "board.json"
            generator.write_board(board_path)
            report = auditor.audit_board(board_path)
            self.assertTrue(report["admitted"])
            self.assertFalse(report["model_fit"])
            self.assertFalse(report["gpu_used"])

            tampered = json.loads(board_path.read_text("ascii"))
            tampered["cases"][0]["in_family_query"]["answer"] = (
                tampered["cases"][0]["in_family_query"]["answer"] + 1
            ) % generator.FIELD
            tampered_path = Path(directory) / "tampered.json"
            tampered_path.write_bytes(generator.pretty_json_bytes(tampered))
            os.chmod(tampered_path, 0o444)
            with self.assertRaises(ValueError):
                auditor.audit_board(tampered_path)

    def test_generator_and_auditor_outputs_are_exclusive_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            board_path = Path(directory) / "board.json"
            audit_path = Path(directory) / "audit.json"
            generator.write_board(board_path)
            self.assertEqual(board_path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                generator.write_board(board_path)
            auditor.write_audit(board_path, audit_path)
            self.assertEqual(audit_path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                auditor.write_audit(board_path, audit_path)

    def test_package_has_no_training_or_accelerator_dependency(self):
        for module in (generator, auditor):
            tree = ast.parse(inspect.getsource(module))
            imports = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imports.update(
                node.module.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertFalse(imports & {"torch", "tensorflow", "jax", "subprocess"})


if __name__ == "__main__":
    unittest.main()
