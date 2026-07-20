#!/usr/bin/env python3

import copy
import unittest

from audit_sd_cst_board import audit_board, simulate_adjacent_swaps
from build_sd_cst_board import build_all


class SDCSTBoardAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.development, cls.confirmation = build_all(
            train_rows=36,
            development_families=18,
            confirmation_families=18,
            seed=2026072004,
        )

    def test_fresh_board_passes_every_gate(self):
        report = audit_board(self.train, self.development, self.confirmation)
        self.assertTrue(report["all_gates_pass"], report["violations"])
        self.assertEqual(report["family_size"], 8)
        self.assertTrue(report["gates"]["answer_and_query_roles_balanced"])
        self.assertTrue(
            report["gates"][
                "query_twin_program_bytes_identical_and_answers_separate"
            ]
        )
        for fields in report["cross_split_overlap_counts"].values():
            self.assertFalse(any(fields.values()))

    def test_independent_simulator_halts_before_valid_suffix(self):
        canonical = next(
            row for row in self.development if row["variant"] == "canonical"
        )
        targets = canonical["compiler_targets"]
        program = tuple(
            (
                int(item["entity_role"]),
                str(item["direction"]),
                int(item["amount"]),
            )
            for item in targets["event_slots"] if item["kind"] != "stop"
        )
        initial = tuple(targets["initial_order_roles"])
        halted, trajectory = simulate_adjacent_swaps(
            initial, program, targets["halt_after"]
        )
        full, _ = simulate_adjacent_swaps(initial, program, 7)
        query = canonical["late_query_target"]["position"]
        self.assertEqual(tuple(canonical["oracle"]["final_state_roles"]), halted)
        self.assertEqual(len(trajectory), targets["halt_after"] + 1)
        self.assertNotEqual(halted[query], full[query])

    def test_audit_rejects_query_leak_and_nonidentical_query_twin(self):
        leaked = copy.deepcopy(self.development)
        leaked[0]["program_text"] += "\nMirexo query: report position 1."
        report = audit_board(self.train, leaked, self.confirmation)
        self.assertFalse(
            report["gates"]["late_query_withheld_until_after_program"]
        )

        broken = copy.deepcopy(self.development)
        twin = next(row for row in broken if row["variant"] == "query_swap")
        twin["program_text"] += " "
        report = audit_board(self.train, broken, self.confirmation)
        self.assertFalse(
            report["gates"][
                "query_twin_program_bytes_identical_and_answers_separate"
            ]
        )

    def test_audit_rejects_training_evidence_and_active_prefix_drift(self):
        leaked_train = copy.deepcopy(self.train)
        leaked_train[0]["oracle"] = {"answer_role": 0}
        report = audit_board(leaked_train, self.development, self.confirmation)
        self.assertFalse(report["gates"]["training_evidence_excluded"])

        broken = copy.deepcopy(self.development)
        suffix = next(row for row in broken if row["variant"] == "post_halt_suffix")
        canonical = next(
            row
            for row in broken
            if row["family_id"] == suffix["family_id"]
            and row["variant"] == "canonical"
        )
        suffix["compiler_targets"]["event_slots"][0] = copy.deepcopy(
            canonical["compiler_targets"]["event_slots"][1]
        )
        report = audit_board(self.train, broken, self.confirmation)
        self.assertFalse(report["gates"]["paired_family_semantics_exact"])

        bad_category = copy.deepcopy(self.development)
        operation = next(
            slot
            for slot in bad_category[0]["compiler_targets"]["event_slots"]
            if slot["kind"] != "stop"
        )
        operation["kind_id"] = 1 - operation["kind_id"]
        report = audit_board(self.train, bad_category, self.confirmation)
        self.assertFalse(report["gates"]["rendered_text_matches_compiler_fields"])


if __name__ == "__main__":
    unittest.main()
