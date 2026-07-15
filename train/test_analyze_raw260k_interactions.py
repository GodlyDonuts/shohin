import tempfile
import unittest
from pathlib import Path

import analyze_raw260k_interactions as analysis


ROOT = Path(__file__).resolve().parents[1]


class Raw260kInteractionAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analysis.analyze(ROOT)

    def test_frozen_hashes_and_bindings(self):
        self.assertEqual(len(self.report["artifact_hashes"]), 11)
        self.assertEqual(
            self.report["bindings"]["checkpoint_sha256"],
            "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d",
        )
        self.assertEqual(
            self.report["bindings"]["tokenizer_sha256"],
            "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4",
        )
        self.assertEqual(
            self.report["bindings"]["continuation_case_payload_sha256"],
            "3bae0add841e403d01251ae6e6ff110f3c6a07324b28de1b671a59f012071f7c",
        )

    def test_manual_and_corrected_continuation_scores(self):
        manual = self.report["manual"]
        self.assertEqual(manual["strict_scores"]["initial"]["correct"], 1)
        self.assertEqual(manual["strict_scores"]["review"]["correct"], 0)
        self.assertEqual(manual["strict_scores"]["verified_fact"]["correct"], 1)
        self.assertEqual(manual["strict_scores"]["state_reuse"]["correct"], 0)
        self.assertEqual(manual["compact_state_exact_prefix"]["correct"], 0)

        continuation = self.report["continuation"]
        self.assertEqual(
            continuation["embedded_v1_direct_final_before_parser_repair"], 1
        )
        self.assertEqual(continuation["corrected_v2_direct_final"], 4)
        self.assertEqual(
            continuation["overall"]["worked_completion"]["final_correct"]["correct"], 8
        )
        sequential = continuation["by_family"]["sequential_state"]
        self.assertEqual(sequential["direct_qa"]["intermediates_present"]["correct"], 5)
        self.assertEqual(sequential["direct_qa"]["final_correct"]["correct"], 4)
        self.assertEqual(sequential["direct_qa"]["termination_only_failures"], 1)
        self.assertEqual(sequential["worked_completion"]["final_correct"]["correct"], 5)

    def test_renderer_and_recurrence_separate_from_parser(self):
        ssc = self.report["ssc_next_state"]
        self.assertEqual(ssc["parse_success"]["correct"], 55)
        self.assertEqual(ssc["input_plus_one"]["correct"], 43)
        self.assertEqual(ssc["local_operation_correct"]["correct"], 1)
        self.assertEqual(ssc["full_chain_correct"]["correct"], 0)

        problem_work = self.report["atomic_formats"]["problem_work"]
        self.assertEqual(problem_work["parse_success"]["correct"], 110)
        self.assertEqual(problem_work["atomic_gold_state"]["correct"], 44)
        self.assertEqual(problem_work["first_transition"]["correct"], 18)
        self.assertEqual(problem_work["chained_local_operation"]["correct"], 44)
        self.assertEqual(problem_work["full_chain"]["correct"], 10)
        self.assertEqual(
            problem_work["local_gold_cross_tabulation"],
            {
                "local_false_gold_false": 11,
                "local_true_gold_false": 4,
                "local_true_gold_true": 40,
            },
        )
        self.assertEqual(
            problem_work["by_family"]["sequential_state"]["full_chain"]["correct"], 5
        )
        self.assertEqual(
            problem_work["by_operation"]["remainder"]["atomic_gold_state"]["correct"], 2
        )

        renderer = self.report["renderer_interchange"]
        self.assertEqual(renderer["local_wins"], 6)
        self.assertEqual(renderer["source_wins"], 0)
        self.assertAlmostEqual(renderer["minimum_absolute_margin"], 0.793863810133189)

    def test_packet_compiler_updater_and_halt_are_distinct(self):
        packet = self.report["packet_interface"]
        self.assertEqual(packet["compiler_arithmetic_complete"]["correct"], 2)
        self.assertEqual(packet["compiler_packet_form"]["correct"], 0)
        self.assertEqual(packet["updater_exact_next_packet"]["correct"], 0)
        self.assertEqual(packet["updater_repeats_observed_result"]["correct"], 2)
        self.assertEqual(packet["halt_first_integer_correct"]["correct"], 1)
        self.assertEqual(packet["halt_exact_integer_only"]["correct"], 0)

    def test_updater_subskills_fail_independently(self):
        updater = self.report["updater_subskills"]
        self.assertEqual(updater["model_calls"], 12)
        self.assertEqual(updater["copy_state"], {"correct": 0, "total": 3, "rate": 0.0})
        self.assertEqual(
            updater["delete_head"], {"correct": 0, "total": 3, "rate": 0.0}
        )
        self.assertEqual(
            updater["joint_natural"], {"correct": 0, "total": 3, "rate": 0.0}
        )
        self.assertEqual(
            updater["joint_packet"], {"correct": 0, "total": 3, "rate": 0.0}
        )
        self.assertEqual(
            updater["strict_joint"], {"correct": 0, "total": 6, "rate": 0.0}
        )
        self.assertTrue(updater["all_calls_hit_max_new"])

    def test_fresh_confirmation_file_is_a_board_not_a_transcript(self):
        board = self.report["fresh_confirmation_board"]
        self.assertEqual(board["cases"], 256)
        self.assertEqual(board["families"], 64)
        self.assertEqual(board["rows_with_responses"], 0)

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "artifact.json"
            path.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                analysis.read_hashed_json(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
