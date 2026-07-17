#!/usr/bin/env python3
"""Exhaustive tests for the frozen R12 MDRT CPU mechanics falsifier."""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from pathlib import Path
import unittest

from pipeline import mdrt_cpu_falsifier as mdrt


class FiniteBoardTests(unittest.TestCase):
    def test_f17_actions_are_noncommutative(self) -> None:
        self.assertEqual(mdrt.execute_program(0, "AB"), 2)
        self.assertEqual(mdrt.execute_program(0, "BA"), 1)
        self.assertNotEqual(
            mdrt.execute_program(7, "AB"),
            mdrt.execute_program(7, "BA"),
        )

    def test_every_runtime_state_round_trips_in_canonical_order(self) -> None:
        states = mdrt.all_states()
        self.assertEqual(len(states), 8_688)
        self.assertEqual(mdrt.STATE_BITS, 14)
        for state_id, state in enumerate(states):
            self.assertEqual(mdrt.encode_state_id(state), state_id)
            self.assertEqual(mdrt.decode_state_id(state_id), state)
            self.assertEqual(
                mdrt.decode_state_code(mdrt.state_code_vector(state)), state
            )

    def test_board_counts_partitions_and_commitment_are_frozen(self) -> None:
        board = mdrt.board_summary()
        self.assertEqual(board["source_case_count"], 8_670)
        self.assertEqual(board["runtime_state_count"], 8_688)
        self.assertEqual(board["state_action_cell_count"], 26_064)
        self.assertEqual(
            board["source_trajectory_step_count_including_halt"], 69_632
        )
        self.assertEqual(
            board["partition_counts"],
            {
                "train_named_mechanics": 510,
                "development_named_mechanics": 1_632,
                "evaluation_named_mechanics": 6_528,
            },
        )
        self.assertEqual(
            board["source_commitment_sha256"], mdrt.FROZEN_SOURCE_SHA256
        )
        self.assertEqual(len({case.case_id for case in mdrt.source_cases()}), 8_670)

    def test_task_transition_is_total_and_wrong_actions_fail_closed(self) -> None:
        state = mdrt.RuntimeState(3, "AB")
        self.assertEqual(
            mdrt.task_transition(state, "A"), mdrt.RuntimeState(4, "B")
        )
        self.assertEqual(mdrt.task_transition(state, "B"), mdrt.SINK_STATE)
        self.assertEqual(mdrt.task_transition(state, mdrt.HALT), mdrt.SINK_STATE)
        terminal = mdrt.RuntimeState(9, "")
        self.assertEqual(mdrt.task_transition(terminal, mdrt.HALT), terminal)
        self.assertEqual(mdrt.task_transition(terminal, "A"), mdrt.SINK_STATE)
        for state in mdrt.all_states():
            for action in mdrt.ACTIONS:
                self.assertIn(mdrt.task_transition(state, action), mdrt.all_states())


class MixedDifferenceTests(unittest.TestCase):
    def test_four_term_subtraction_cancels_all_main_effects(self) -> None:
        states = (
            mdrt.RuntimeState(0, ""),
            mdrt.RuntimeState(3, "AB"),
            mdrt.RuntimeState(16, "B" * 8),
            mdrt.SINK_STATE,
        )
        for state in states:
            for action in mdrt.ACTIONS:
                expected = mdrt.state_code_vector(
                    mdrt.task_transition(state, action)
                )
                self.assertEqual(
                    mdrt.mixed_difference(mdrt.ARM_POSITIVE, state, action),
                    expected,
                )
                self.assertEqual(
                    mdrt.mixed_difference(mdrt.ARM_ZERO, state, action),
                    mdrt.ZERO_VECTOR,
                )

    def test_exhaustive_positive_and_zero_cell_gates(self) -> None:
        positive = mdrt.mixed_cell_audit(mdrt.ARM_POSITIVE)
        zero = mdrt.mixed_cell_audit(mdrt.ARM_ZERO)
        self.assertEqual(
            positive,
            {
                "arm": mdrt.ARM_POSITIVE,
                "total_cells": 26_064,
                "exact_successors": 26_064,
                "valid_decodes": 26_064,
                "zero_vectors": 0,
            },
        )
        self.assertEqual(
            zero,
            {
                "arm": mdrt.ARM_ZERO,
                "total_cells": 26_064,
                "exact_successors": 0,
                "valid_decodes": 0,
                "zero_vectors": 26_064,
            },
        )

    def test_decode_fails_closed_on_zero_or_corrupted_vectors(self) -> None:
        with self.assertRaises(mdrt.DecodeError):
            mdrt.decode_state_code(mdrt.ZERO_VECTOR)
        vector = list(mdrt.state_code_vector(mdrt.RuntimeState(4, "BA")))
        vector[2] = (vector[2] + 1) % mdrt.VECTOR_MODULUS
        with self.assertRaises(mdrt.DecodeError):
            mdrt.decode_state_code(tuple(vector))


class TrajectoryControlTests(unittest.TestCase):
    def test_positive_and_zero_trajectory_contracts(self) -> None:
        positive = mdrt.trajectory_audit(mdrt.ARM_POSITIVE)
        zero = mdrt.trajectory_audit(mdrt.ARM_ZERO)
        self.assertEqual(positive["exact_trajectories"], 8_670)
        self.assertEqual(positive["exact_final_values"], 8_670)
        self.assertEqual(positive["halted"], 8_670)
        self.assertEqual(positive["all_decodes_valid"], 8_670)
        self.assertEqual(zero["exact_trajectories"], 0)
        self.assertEqual(zero["exact_final_values"], 0)
        self.assertEqual(zero["halted"], 0)
        self.assertEqual(zero["all_decodes_valid"], 0)

    def test_state_depth_shortcut_hits_exact_frozen_ceiling(self) -> None:
        shortcut = mdrt.trajectory_audit(mdrt.ARM_SHORTCUT)
        self.assertEqual(shortcut["exact_trajectories"], 136)
        self.assertEqual(shortcut["halted"], 8_670)
        for length, row in shortcut["by_length"].items():
            self.assertEqual(row["cases"], 17 * (1 << length))
            self.assertEqual(row["exact_trajectories"], 17)

    def test_one_rollout_consumes_one_symbol_then_halts_autonomously(self) -> None:
        sealed = mdrt.compile_source(5, "ABB")
        rollout = mdrt.rollout_from_sealed(sealed, mdrt.ARM_POSITIVE)
        self.assertEqual(rollout.actions, ("A", "B", "B", mdrt.HALT))
        self.assertEqual(
            [len(mdrt.decode_state_id(state_id).remaining) for state_id in rollout.state_ids],
            [3, 2, 1, 0, 0],
        )
        final_state = mdrt.decode_state_id(rollout.final_state_id)
        self.assertEqual(final_state.value, mdrt.execute_program(5, "ABB"))
        self.assertTrue(rollout.halted)


class MooreCollapseTests(unittest.TestCase):
    def test_task_and_positive_preserve_complete_quotient(self) -> None:
        task = mdrt.moore_minimization_audit(mdrt.MACHINE_TASK)
        positive = mdrt.moore_minimization_audit(mdrt.ARM_POSITIVE)
        self.assertEqual(task["class_count"], 8_688)
        self.assertEqual(task["singleton_classes"], 8_688)
        self.assertEqual(positive["class_count"], 8_688)
        self.assertEqual(positive["singleton_classes"], 8_688)

    def test_negative_and_shortcut_collapse_to_smaller_machines(self) -> None:
        zero = mdrt.moore_minimization_audit(mdrt.ARM_ZERO)
        shortcut = mdrt.moore_minimization_audit(mdrt.ARM_SHORTCUT)
        self.assertEqual(zero["class_count"], 52)
        self.assertEqual(shortcut["class_count"], 154)
        self.assertLess(zero["class_count"], mdrt.STATE_COUNT)
        self.assertLess(shortcut["class_count"], mdrt.STATE_COUNT)


class ErasureAndResourceTests(unittest.TestCase):
    def test_runtime_surface_contains_only_the_sealed_baton(self) -> None:
        self.assertEqual([field.name for field in fields(mdrt.SealedState)], ["state_id"])
        self.assertEqual(
            list(inspect.signature(mdrt.rollout_from_sealed).parameters),
            ["sealed", "arm"],
        )

    def test_donor_interchange_follows_donor_not_recipient(self) -> None:
        recipient = mdrt.compile_source(0, "A")
        donor = mdrt.compile_source(7, "BB")
        transplanted = mdrt.interchange_sealed_state(recipient, donor)
        recipient_rollout = mdrt.rollout_from_sealed(recipient, mdrt.ARM_POSITIVE)
        donor_rollout = mdrt.rollout_from_sealed(donor, mdrt.ARM_POSITIVE)
        transplanted_rollout = mdrt.rollout_from_sealed(
            transplanted, mdrt.ARM_POSITIVE
        )
        self.assertEqual(
            mdrt.rollout_bytes(transplanted_rollout),
            mdrt.rollout_bytes(donor_rollout),
        )
        self.assertNotEqual(
            mdrt.rollout_bytes(transplanted_rollout),
            mdrt.rollout_bytes(recipient_rollout),
        )

    def test_exhaustive_erasure_and_donor_audit(self) -> None:
        audit = mdrt.erasure_and_donor_audit()
        self.assertTrue(audit["structural_source_free"])
        self.assertEqual(audit["source_mutation_bit_identical"], 8_670)
        self.assertEqual(audit["stale_state_bit_identical"], 8_670)
        self.assertEqual(audit["donor_following"], 8_670)

    def test_allocated_resources_are_matched_and_padding_is_visible(self) -> None:
        audit = mdrt.matched_resource_audit()
        self.assertTrue(audit["allocated_budgets_identical"])
        budgets = [
            audit["arms"][arm]["allocated"] for arm in mdrt.EXECUTABLE_ARMS
        ]
        self.assertTrue(all(budget == budgets[0] for budget in budgets[1:]))
        self.assertEqual(budgets[0]["allocated_persistent_state_bits"], 14)
        self.assertEqual(budgets[0]["allocated_transient_vector_bits"], 102)
        self.assertEqual(budgets[0]["allocated_fixed_tail_table_entries"], 26_064)
        self.assertEqual(budgets[0]["allocated_fixed_tail_table_bits"], 364_896)
        self.assertEqual(budgets[0]["source_bytes_retained_after_compile"], 0)
        self.assertEqual(budgets[0]["external_execution_calls"], 0)
        shortcut_used = audit["arms"][mdrt.ARM_SHORTCUT]["utilized"]
        self.assertEqual(shortcut_used["utilized_persistent_state_bits"], 9)
        self.assertEqual(shortcut_used["semantic_tail_calls_per_transition"], 0)


class FullContractTests(unittest.TestCase):
    def test_full_audit_is_deterministic_and_all_frozen_gates_hold(self) -> None:
        first = mdrt.run_audit()
        second = mdrt.run_audit()
        self.assertEqual(mdrt.canonical_json_bytes(first), mdrt.canonical_json_bytes(second))
        self.assertTrue(first["mechanics_contract_satisfied"])
        self.assertTrue(all(first["gates"].values()))
        self.assertIn("no neural learnability", first["claim_boundary"])

    def test_module_imports_no_accelerator_network_or_subprocess_stack(self) -> None:
        path = Path(mdrt.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden = {
            "numpy",
            "requests",
            "socket",
            "subprocess",
            "tensorflow",
            "torch",
        }
        self.assertTrue(forbidden.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
