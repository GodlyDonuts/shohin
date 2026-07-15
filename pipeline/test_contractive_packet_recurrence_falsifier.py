#!/usr/bin/env python3
"""Exhaustive tests for the frozen contractive-packet CPU falsifier."""

from __future__ import annotations

import ast
import copy
from dataclasses import fields
from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from pipeline import contractive_packet_recurrence_falsifier as cpr


class AlgebraAndStateTests(unittest.TestCase):
    def test_d14_is_closed_associative_and_noncommutative(self) -> None:
        actions = cpr.action_space()
        self.assertEqual(len(actions), 14)
        self.assertEqual(len(set(actions)), 14)
        for index, action in enumerate(actions):
            self.assertEqual(cpr.encode_action(action), index)
            self.assertEqual(cpr.decode_action(index), action)
        for first in actions:
            for second in actions:
                composed = first.followed_by(second)
                self.assertIn(composed, actions)
                for value in range(7):
                    self.assertEqual(
                        composed.apply(value), second.apply(first.apply(value))
                    )
                for third in actions:
                    self.assertEqual(
                        first.followed_by(second).followed_by(third),
                        first.followed_by(second.followed_by(third)),
                    )
        audit = cpr.relation_audit()
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["noncommutative_witness"]["TN_at_0"], 6)
        self.assertEqual(audit["noncommutative_witness"]["NT_at_0"], 1)

    def test_all_residual_words_states_and_transitions_are_closed(self) -> None:
        self.assertEqual(len(cpr.RESIDUAL_WORDS), 127)
        self.assertEqual(cpr.SEMANTIC_STATE_COUNT, 889)
        seen = set()
        for word in cpr.RESIDUAL_WORDS:
            self.assertLessEqual(len(word), 6)
            for value in range(7):
                state = cpr.encode_semantic_state(value, word)
                seen.add(state)
                self.assertEqual(cpr.decode_semantic_state(state), (value, word))
                successor = cpr.semantic_transition(state)
                next_value, next_word = cpr.decode_semantic_state(successor)
                if word:
                    self.assertEqual(next_word, word[1:])
                    self.assertEqual(
                        next_value, cpr.generator_action(word[0]).apply(value)
                    )
                else:
                    self.assertEqual(successor, state)
        self.assertEqual(seen, set(range(889)))

    def test_serial_tree_and_action_fsm_agree_for_every_source_and_value(self) -> None:
        for word in cpr.RESIDUAL_WORDS:
            action = cpr.compile_action(word)
            self.assertEqual(cpr.tree_compile_action(word), action)
            self.assertEqual(cpr.decode_action(cpr.fsm_compile_action(word)), action)
            for value in range(7):
                self.assertEqual(cpr.serial_execute(word, value), action.apply(value))


class FrozenBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = cpr.generate_board()
        cls.report = cpr.audit_board(cls.board)

    def test_board_is_exhaustive_deterministic_and_frozen(self) -> None:
        self.assertEqual(len(self.board["cases"]), 889)
        self.assertEqual(len(cpr.board_bytes(self.board)), 174208)
        self.assertEqual(
            self.board["source_commitment_sha256"], cpr.FROZEN_SOURCE_SHA256
        )
        self.assertEqual(cpr.board_sha256(self.board), cpr.FROZEN_BOARD_SHA256)
        self.assertEqual(cpr.audit_report_sha256(self.report), cpr.FROZEN_AUDIT_SHA256)
        self.assertEqual(cpr.generate_board(), self.board)
        keys = {(case["source"], case["initial_value"]) for case in self.board["cases"]}
        self.assertEqual(
            keys,
            {(word, value) for word in cpr.RESIDUAL_WORDS for value in range(7)},
        )

    def test_all_favorable_controls_and_every_local_transition_are_exact(self) -> None:
        controls = self.report["controls"]
        self.assertTrue(controls["all_exact"])
        self.assertEqual(
            controls["correct_final_cells"],
            {
                "action_fsm": 889,
                "balanced_tree": 889,
                "coded_recurrence": 889,
                "residual_fsm": 889,
                "serial": 889,
            },
        )
        self.assertEqual(controls["local_transition_correct"], 8988)
        self.assertEqual(controls["local_transition_checks"], 8988)

    def test_board_trajectories_are_complete_and_end_at_the_answer(self) -> None:
        transition_cells = 0
        for case in self.board["cases"]:
            trajectory = case["trajectory_state_indices"]
            self.assertEqual(len(trajectory), case["source_length"] + 1)
            self.assertEqual(trajectory[0], case["initial_state_index"])
            final_value, final_residual = cpr.decode_semantic_state(trajectory[-1])
            self.assertEqual(final_residual, "")
            self.assertEqual(final_value, case["expected_final"])
            transition_cells += len(trajectory) - 1
        self.assertEqual(transition_cells, 4494)

    def test_source_commitment_and_board_hash_cover_distinct_fields(self) -> None:
        source_hash = cpr.source_commitment_sha256(self.board["cases"])
        board_hash = cpr.board_sha256(self.board)
        expected_only = copy.deepcopy(self.board)
        expected_only["cases"][1]["expected_final"] ^= 1
        self.assertEqual(
            cpr.source_commitment_sha256(expected_only["cases"]), source_hash
        )
        self.assertNotEqual(cpr.board_sha256(expected_only), board_hash)
        with self.assertRaises(cpr.AuditError):
            cpr.audit_board(expected_only, require_frozen_hashes=False)

        source_mutation = copy.deepcopy(self.board)
        source_mutation["cases"][1]["source"] = "N"
        self.assertNotEqual(
            cpr.source_commitment_sha256(source_mutation["cases"]), source_hash
        )


class ContractionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = cpr.generate_board()
        cls.report = cpr.audit_board(cls.board)

    def test_every_radius_two_corruption_is_reset_and_radius_three_fails(self) -> None:
        corruption = self.report["corruption"]
        self.assertTrue(corruption["pass"])
        self.assertEqual(
            corruption["expected_subsets_by_weight"],
            {"0": 889, "1": 4445, "2": 8890, "3": 8890},
        )
        self.assertEqual(
            corruption["coherent_correct_by_weight"],
            {"0": 889, "1": 4445, "2": 8890, "3": 0},
        )
        self.assertEqual(corruption["coherent_wrong_donor_at_weight_3"], 8890)
        self.assertEqual(corruption["invalid_rejections_at_weight_3"], 8890)

    def test_projection_rejects_ties_and_invalid_majorities(self) -> None:
        with self.assertRaises(cpr.ProjectionError):
            cpr.project_packet(cpr.SealedPacket((0, 0, 1, 1, cpr.INVALID_LANE)))
        with self.assertRaises(cpr.ProjectionError):
            cpr.project_packet(
                cpr.SealedPacket((cpr.INVALID_LANE,) * cpr.REPETITION_LANES)
            )
        self.assertEqual(
            cpr.decode_valid_packet(
                cpr.project_packet(cpr.SealedPacket((0, 0, 0, 1, 2)))
            ),
            0,
        )

    def test_all_wrong_valid_codewords_are_fixed_points_not_contracted(self) -> None:
        no_go = self.report["valid_codeword_no_go"]
        expected = 889 * 888
        self.assertTrue(no_go["pass"])
        self.assertEqual(no_go["checked_ordered_distinct_pairs"], expected)
        self.assertEqual(no_go["distance_preserved_pairs"], expected)
        self.assertEqual(no_go["compiler_channel_swaps_follow_donor"], expected)
        self.assertEqual(no_go["transition_channel_swaps_follow_donor"], expected)
        self.assertEqual(no_go["projection_channel_swaps_follow_donor"], expected)

    def test_local_contraction_and_global_no_go_witness_are_both_exact(self) -> None:
        theorem = self.report["theorem"]
        self.assertTrue(theorem["pass"])
        self.assertEqual(
            theorem["local_basin_checks"], theorem["local_zero_after_projection"]
        )
        witness = theorem["global_strict_contraction_impossible_witness"]
        self.assertEqual(witness["distance_before"], 5)
        self.assertEqual(witness["distance_after"], 5)
        self.assertTrue(witness["ratio_is_one"])

    def test_semantic_error_reliability_still_multiplies_with_depth(self) -> None:
        rows = self.report["theorem"]["semantic_error_exact_sequence_success"]
        for row in rows:
            expected = Fraction(99, 100) ** row["depth"]
            self.assertEqual(row["numerator"], expected.numerator)
            self.assertEqual(row["denominator"], expected.denominator)
        self.assertGreater(
            Fraction(rows[0]["numerator"], rows[0]["denominator"]),
            Fraction(rows[-1]["numerator"], rows[-1]["denominator"]),
        )


class ChannelAndCollapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = cpr.generate_board()
        cls.report = cpr.audit_board(cls.board)

    def test_sealed_packet_has_no_external_source_but_embeds_residual_source(
        self,
    ) -> None:
        deletion = self.report["source_deletion_and_rescues"]
        self.assertEqual({field.name for field in fields(cpr.SealedPacket)}, {"lanes"})
        self.assertEqual(deletion["packet_fields"], ["lanes"])
        self.assertEqual(deletion["source_deleted_external_cases"], 889)
        self.assertEqual(deletion["embedded_source_cases"], 889)
        packet = cpr.compile_residual_packet(3, "TNT")
        value, residual = cpr.decode_semantic_state(cpr.decode_valid_packet(packet))
        self.assertEqual((value, residual), (3, "TNT"))

    def test_semantic_errors_survive_unless_duplicate_or_source_replay_is_used(
        self,
    ) -> None:
        deletion = self.report["source_deletion_and_rescues"]
        self.assertTrue(deletion["pass"])
        self.assertEqual(deletion["semantic_error_survives_projection"], 4494)
        self.assertEqual(deletion["duplicate_transition_rescues"], 4494)
        self.assertEqual(deletion["source_replay_rescues"], 4494)
        self.assertEqual(deletion["source_replay_symbols_read"], 14322)

    def test_coded_recurrence_collapses_exactly_to_a_smaller_residual_fsm(self) -> None:
        collapse = self.report["classical_collapse"]
        self.assertTrue(collapse["pass"])
        self.assertFalse(collapse["noncollapsed_interface_survives"])
        self.assertEqual(collapse["behaviorally_equivalent_semantic_states"], 889)
        self.assertEqual(collapse["coded_active_bits"], 50)
        self.assertEqual(collapse["fsm_active_bits"], 10)
        self.assertEqual(collapse["coded_depth_per_max_case"], 12)
        self.assertEqual(collapse["fsm_depth_per_max_case"], 6)

    def test_resource_ledger_has_all_channels_and_exact_totals(self) -> None:
        ledger = self.board["resource_ledger"]
        self.assertEqual(
            set(ledger),
            {
                "action_fsm",
                "balanced_tree",
                "coded_recurrence",
                "duplicate_verified_coded",
                "residual_fsm",
                "serial",
                "source_replay_rescue",
            },
        )
        for entry in ledger.values():
            self.assertEqual(set(entry), set(cpr.RESOURCE_SECTIONS))
            self.assertTrue(
                all(value == 0 for value in entry["external_resources"].values())
            )
        coded = ledger["coded_recurrence"]
        self.assertEqual(coded["compiler_channel"]["source_symbols_read"], 4494)
        self.assertEqual(coded["transition_channel"]["lane_updates"], 22470)
        self.assertEqual(coded["projection_channel"]["calls"], 4494)
        self.assertEqual(coded["projection_channel"]["packet_lane_reads"], 22470)
        self.assertEqual(coded["source_channel"]["postseal_external_reads"], 0)
        self.assertEqual(coded["source_channel"]["embedded_residual_symbols_max"], 6)
        verified = ledger["duplicate_verified_coded"]
        self.assertEqual(
            verified["transition_channel"]["duplicate_verifier_updates"], 4494
        )
        replay = ledger["source_replay_rescue"]
        self.assertEqual(replay["source_channel"]["replay_source_symbols_read"], 14322)


class FileAndStaticContractTests(unittest.TestCase):
    def test_immutable_round_trip_refuses_overwrite_writable_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "cpr_board.json"
            result = cpr.write_immutable_board(path)
            self.assertEqual(result["mode"], 0o444)
            self.assertEqual(result["sha256"], cpr.FROZEN_BOARD_SHA256)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertTrue(cpr.audit_board_file(path)["pass"])
            with self.assertRaises(FileExistsError):
                cpr.write_immutable_board(path)

            os.chmod(path, 0o644)
            with self.assertRaises(cpr.AuditError):
                cpr.audit_board_file(path)

            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(cpr.AuditError):
                cpr.audit_board_file(link)

    def test_file_audit_rejects_noncanonical_and_tampered_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "board.json"
            board = cpr.generate_board()
            path.write_text(json.dumps(board, indent=2), encoding="ascii")
            os.chmod(path, 0o444)
            with self.assertRaises(cpr.AuditError):
                cpr.audit_board_file(path)

            os.chmod(path, 0o644)
            payload = cpr.board_bytes(board).replace(
                b'"expected_final":0', b'"expected_final":1', 1
            )
            path.write_bytes(payload)
            os.chmod(path, 0o444)
            with self.assertRaises(cpr.AuditError):
                cpr.audit_board_file(path)

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_nonascii(self) -> None:
        with self.assertRaises(cpr.AuditError):
            cpr.strict_json_loads(b'{"x":1,"x":2}\n')
        with self.assertRaises(cpr.AuditError):
            cpr.strict_json_loads(b'{"x":NaN}\n')
        with self.assertRaises(cpr.AuditError):
            cpr.strict_json_loads('{"x":"é"}\n'.encode("utf-8"))

    def test_module_is_stdlib_cpu_only_and_has_no_training_surface(self) -> None:
        module_path = Path(cpr.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        allowed = set(sys.stdlib_module_names) | {"__future__"}
        self.assertEqual(imported_roots - allowed, set())
        source = module_path.read_text(encoding="utf-8").lower()
        self.assertNotIn("cuda", source)
        self.assertNotIn("checkpoint", source)
        self.assertNotIn("optimizer", source)

    def test_preregistration_binds_all_frozen_hashes_and_no_go(self) -> None:
        document = (
            Path(cpr.__file__).parents[1]
            / "R12_CONTRACTIVE_PACKET_RECURRENCE_PREREG.md"
        ).read_text(encoding="ascii")
        self.assertIn(cpr.PROTOCOL_ID, document)
        self.assertIn(cpr.FROZEN_SOURCE_SHA256, document)
        self.assertIn(cpr.FROZEN_BOARD_SHA256, document)
        self.assertIn(cpr.FROZEN_AUDIT_SHA256, document)
        self.assertIn("NO-GO as a new reasoning primitive", document)


if __name__ == "__main__":
    unittest.main()
