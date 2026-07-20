#!/usr/bin/env python3

import unittest

from sd_cst_cpu_falsifier import run_falsifier


class SDCSTCPUFalsifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_falsifier(2026072005)

    def test_all_cpu_mechanics_gates_pass(self):
        self.assertEqual(
            self.report["decision"], "admit_sd_cst_board_mechanics"
        )
        self.assertTrue(all(self.report["gates"].values()), self.report["gates"])
        cells = self.report["atomic_simulator_cells"]
        self.assertEqual(cells["exact"], cells["total"])

    def test_registered_shortcuts_fail(self):
        controls = self.report["negative_controls"]
        self.assertEqual(controls["execute_through_stop_accuracy"], 0.0)
        self.assertEqual(
            controls["storage_order_as_semantic_order_accuracy"], 0.0
        )
        self.assertEqual(controls["event_bag_ignoring_order_accuracy"], 0.0)
        self.assertEqual(controls["ignore_stop_shift_accuracy"], 0.0)
        self.assertEqual(controls["ignore_late_query_swap_accuracy"], 0.0)
        self.assertEqual(controls["execute_post_halt_suffix_accuracy"], 0.0)
        self.assertAlmostEqual(
            controls["program_and_query_length_depth_accuracy"], 1.0 / 6.0
        )
        self.assertLess(controls["reset_state_each_step_accuracy"], 0.75)

    def test_adversarial_mutations_are_detected(self):
        self.assertTrue(all(self.report["mutation_checks"].values()))
        self.assertTrue(
            self.report["gates"]["board_mechanics_use_only_standard_library"]
        )


if __name__ == "__main__":
    unittest.main()
