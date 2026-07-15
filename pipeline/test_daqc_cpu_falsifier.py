#!/usr/bin/env python3
"""Exhaustive tests for the frozen DAQC D34 CPU falsifier."""

from __future__ import annotations

import ast
import copy
import json
import os
import stat
import tempfile
import unittest
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from pipeline import daqc_cpu_falsifier as daqc


class ActionAlgebraTests(unittest.TestCase):
    def test_all_34_actions_round_trip_and_are_behaviorally_distinct(self) -> None:
        actions = daqc.action_space()
        self.assertEqual(len(actions), 34)
        self.assertEqual(len(set(actions)), 34)
        signatures = set()
        for index, action in enumerate(actions):
            self.assertEqual(daqc.encode_action(action), index)
            self.assertEqual(daqc.decode_action(index), action)
            self.assertEqual(len(daqc.code_bits(index)), 6)
            signature = daqc.behavior_signature(action)
            self.assertEqual(len(signature), 17)
            signatures.add(signature)
        self.assertEqual(len(signatures), 34)

    def test_full_composition_table_closure_and_associativity(self) -> None:
        actions = daqc.action_space()
        action_set = set(actions)
        for left in actions:
            for right in actions:
                composed = left.followed_by(right)
                self.assertIn(composed, action_set)
                for value in range(17):
                    self.assertEqual(
                        composed.apply(value), right.apply(left.apply(value))
                    )
        for first in actions:
            for second in actions:
                for third in actions:
                    self.assertEqual(
                        first.followed_by(second).followed_by(third),
                        first.followed_by(second.followed_by(third)),
                    )

    def test_presentation_relations_and_noncommutative_witness(self) -> None:
        audit = daqc.relation_audit()
        self.assertTrue(audit["pass"])
        for relation in audit["relations"].values():
            self.assertTrue(relation["all_17_inputs_equal"])
            self.assertEqual(relation["left_code"], relation["right_code"])
        witness = audit["noncommutative_translation_reflection_witness"]
        self.assertEqual(witness["left_word"], "TN")
        self.assertEqual(witness["right_word"], "NT")
        self.assertNotEqual(witness["left_output"], witness["right_output"])
        for value in range(17):
            self.assertEqual(daqc.sequential_execute("T" * 17, value), value)
            self.assertEqual(daqc.sequential_execute("NN", value), value)
            self.assertEqual(
                daqc.sequential_execute("NTN", value),
                daqc.sequential_execute("T" * 16, value),
            )


class FrozenBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = daqc.generate_board()

    def test_score_blind_board_exhausts_actions_variants_and_late_inputs(self) -> None:
        cases = self.board["cases"]
        self.assertEqual(len(cases), 34 * 4)
        by_action: dict[int, list[dict[str, object]]] = {}
        for case in cases:
            by_action.setdefault(case["action_index"], []).append(case)
            self.assertEqual(sorted(case["late_inputs"]), list(range(17)))
            self.assertEqual(len(case["expected_outputs"]), 17)
            self.assertEqual(daqc.word_action(case["word"]), daqc.decode_action(case["action_index"]))
            for late_input, expected in zip(
                case["late_inputs"], case["expected_outputs"], strict=True
            ):
                self.assertEqual(
                    daqc.sequential_execute(case["word"], late_input), expected
                )
        self.assertEqual(set(by_action), set(range(34)))
        for action_index, action_cases in by_action.items():
            self.assertEqual(len(action_cases), 4)
            self.assertEqual(
                {case["variant"] for case in action_cases}, set(daqc.VARIANT_NAMES)
            )
            signatures = {
                tuple(case["expected_outputs"][case["late_inputs"].index(value)] for value in range(17))
                for case in action_cases
            }
            self.assertEqual(len(signatures), 1, action_index)

    def test_source_equivalence_classes_are_exact_and_cross_class_collision_free(self) -> None:
        signatures_by_action: dict[int, set[tuple[int, ...]]] = {}
        words = set()
        for case in self.board["cases"]:
            words.add(case["word"])
            signature = tuple(
                daqc.sequential_execute(case["word"], value) for value in range(17)
            )
            signatures_by_action.setdefault(case["action_index"], set()).add(signature)
        self.assertEqual(len(words), 136)
        self.assertTrue(all(len(values) == 1 for values in signatures_by_action.values()))
        class_signatures = {next(iter(values)) for values in signatures_by_action.values()}
        self.assertEqual(len(class_signatures), 34)
        self.assertTrue(daqc.collision_audit()["pass"])

    def test_source_commitment_excludes_late_inputs_but_board_hash_detects_them(self) -> None:
        original_source = daqc.source_commitment_sha256(self.board["cases"])
        original_board = daqc.board_sha256(self.board)
        mutated = copy.deepcopy(self.board)
        mutated["cases"][0]["late_inputs"][0], mutated["cases"][0]["late_inputs"][1] = (
            mutated["cases"][0]["late_inputs"][1],
            mutated["cases"][0]["late_inputs"][0],
        )
        self.assertEqual(
            daqc.source_commitment_sha256(mutated["cases"]), original_source
        )
        self.assertNotEqual(daqc.board_sha256(mutated), original_board)
        with self.assertRaises(daqc.AuditError):
            daqc.audit_board(mutated, require_frozen_hashes=False)

    def test_source_commitment_detects_every_source_field_class(self) -> None:
        original = daqc.source_commitment_sha256(self.board["cases"])
        for key, replacement in (
            ("case_id", "tampered-case"),
            ("action_index", 1),
            ("variant", "tampered-variant"),
            ("word", "N"),
            ("word_length", 999),
            ("source_sha256", "0" * 64),
            ("committed_code_bits", "111111"),
        ):
            mutated = copy.deepcopy(self.board["cases"])
            mutated[0][key] = replacement
            self.assertNotEqual(daqc.source_commitment_sha256(mutated), original, key)

    def test_all_favorable_controls_are_exact_on_every_cell(self) -> None:
        audit = daqc.controls_audit(self.board)
        self.assertEqual(audit["total_cells_per_control"], 136 * 17)
        self.assertTrue(audit["all_exact"])
        self.assertEqual(
            audit["correct_cells"],
            {
                "exact_fst": 136 * 17,
                "sequential": 136 * 17,
                "balanced_tree": 136 * 17,
                "direct": 136 * 17,
            },
        )
        for case in self.board["cases"]:
            self.assertEqual(daqc.fst_compile(case["word"]), case["action_index"])
            self.assertEqual(
                daqc.encode_action(daqc.tree_compile(case["word"])),
                case["action_index"],
            )

    def test_post_commitment_code_has_no_source_and_executes_all_cells(self) -> None:
        field_names = {field.name for field in fields(daqc.SealedCode)}
        self.assertTrue(daqc.sealed_code_has_no_source_fields())
        self.assertEqual(field_names, {"code_index"})
        commitment = self.board["source_commitment_sha256"]
        for case in self.board["cases"]:
            envelope = daqc.seal_source_case(case, commitment)
            self.assertEqual(envelope.source_commitment_sha256, commitment)
            for late_input, expected in zip(
                case["late_inputs"], case["expected_outputs"], strict=True
            ):
                self.assertEqual(envelope.code.execute(late_input), expected)

    def test_code_interchange_all_ordered_action_pairs_and_inputs(self) -> None:
        audit = daqc.code_interchange_audit(self.board)
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["ordered_distinct_action_pairs"], 34 * 33)
        self.assertEqual(audit["checked_late_input_cells"], 34 * 33 * 17)
        representatives = {
            case["action_index"]: daqc.seal_source_case(
                case, self.board["source_commitment_sha256"]
            )
            for case in reversed(self.board["cases"])
        }
        for recipient_index in range(34):
            for donor_index in range(34):
                if recipient_index == donor_index:
                    continue
                swapped = daqc.interchange_code(
                    representatives[recipient_index], representatives[donor_index]
                )
                self.assertEqual(swapped.case_id, representatives[recipient_index].case_id)
                self.assertEqual(swapped.code.code_index, donor_index)
                self.assertEqual(
                    swapped.code_provenance_case_id,
                    representatives[donor_index].code_provenance_case_id,
                )
                donor = daqc.decode_action(donor_index)
                self.assertTrue(
                    all(
                        swapped.code.execute(value) == donor.apply(value)
                        for value in range(17)
                    )
                )

    def test_five_bit_collision_and_six_bit_exact_control(self) -> None:
        five = daqc.finite_code_capacity_audit(5)
        six = daqc.finite_code_capacity_audit(6)
        self.assertEqual(five["required_bits"], 6)
        self.assertFalse(five["injective_possible"])
        self.assertEqual(len(five["collision_witnesses"]), 2)
        for witness in five["collision_witnesses"]:
            self.assertNotEqual(witness["left_output"], witness["right_output"])
        self.assertTrue(six["injective_possible"])
        self.assertEqual(six["collision_witnesses"], [])

    def test_free_binary_semigroup_has_linear_bit_no_go(self) -> None:
        audit = daqc.free_semigroup_linear_bit_no_go(2, 128)
        self.assertTrue(audit["pass"])
        self.assertEqual(len(audit["rows"]), 128)
        for row in audit["rows"]:
            self.assertEqual(
                row["minimum_injective_bits"], row["word_length"]
            )
            self.assertEqual(
                row["distinct_semantics"], 1 << row["word_length"]
            )

    def test_reliability_formulas_are_exact_not_floating_point(self) -> None:
        p = Fraction(99, 100)
        c = Fraction(999, 1000)
        self.assertEqual(daqc.serial_reliability(p, 64), p**64)
        self.assertEqual(daqc.compiled_reliability(c, p, 1), c * p)
        self.assertEqual(daqc.tree_reliability(p, 64), p**63)
        self.assertTrue(daqc.reliability_audit()["pass"])

    def test_resource_vector_is_complete_and_exactly_named(self) -> None:
        vector = daqc.resource_vector(self.board)
        self.assertEqual(tuple(vector), daqc.RESOURCE_VECTOR_FIELDS)
        self.assertEqual(vector["parameters"], 0)
        self.assertEqual(vector["retained_bits"]["per_sealed_code"], 6)
        self.assertEqual(vector["retained_bits"]["retained_source_after_sealing"], 0)
        self.assertEqual(vector["training_examples"], 0)
        self.assertEqual(vector["oracle_calls"], 0)
        self.assertEqual(vector["training_flops"], 0)
        self.assertEqual(vector["inference_flops"], 0)
        self.assertTrue(vector["external_execution"]["used"])

    def test_generation_is_byte_deterministic_and_hashes_are_frozen(self) -> None:
        first = daqc.board_bytes(daqc.generate_board())
        second = daqc.board_bytes(daqc.generate_board())
        self.assertEqual(first, second)
        self.assertEqual(daqc.sha256_bytes(first), daqc.CANONICAL_BOARD_SHA256)
        self.assertEqual(
            self.board["source_commitment_sha256"],
            daqc.CANONICAL_SOURCE_SHA256,
        )
        report = daqc.audit_board(self.board)
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["gates"].values()))

    def test_immutable_file_round_trip_refuses_overwrite_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "daqc_d34_v1.json"
            result = daqc.write_immutable_board(path)
            self.assertEqual(result["mode"], 0o444)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertFalse(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            self.assertTrue(daqc.audit_board_file(path)["pass"])
            with self.assertRaises(FileExistsError):
                daqc.write_immutable_board(path)

            os.chmod(path, 0o644)
            with self.assertRaises(daqc.AuditError):
                daqc.audit_board_file(path)
            parsed = json.loads(path.read_bytes())
            parsed["cases"][0]["expected_outputs"][0] ^= 1
            path.write_text(json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaises(daqc.AuditError):
                daqc.audit_board_file(path)

    def test_module_imports_only_cpu_standard_library(self) -> None:
        module_path = Path(daqc.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertFalse({"torch", "tensorflow", "jax", "numpy"} & imported_roots)


if __name__ == "__main__":
    unittest.main()
