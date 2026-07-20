from __future__ import annotations

from pathlib import Path
import unittest

from s9_2_global_anchor_falsifier import (
    MIN_EXHAUSTIVE_CASES,
    run,
    run_reduced_exhaustive,
)


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "artifacts/r12/s9_occurrence_quotient_board_7563652620455132721"
TOKENIZER = ROOT / "artifacts/tokenizer/tokenizer.json"


class S92GlobalAnchorFalsifierTest(unittest.TestCase):
    def test_reduced_viterbi_matches_independent_exhaustive_search(self):
        result = run_reduced_exhaustive(seed=0x592A, cases=256)
        self.assertEqual(result["exact_matches"], result["cases"])
        self.assertLess(result["max_score_absolute_error"], 1e-4)

    def test_full_closed_board_falsifier_passes_every_preregistered_gate(self):
        result = run(
            BOARD,
            TOKENIZER,
            seed=0x592A,
            exhaustive_cases=MIN_EXHAUSTIVE_CASES,
        )
        self.assertEqual(result["rows"], 2_048)
        self.assertEqual(result["closed_split"], "development.jsonl")
        self.assertEqual(result["resource_boundary"]["confirmation_rows_read"], 0)
        self.assertEqual(
            result["decision"], "admit_s9_2_global_anchor_mechanics"
        )
        self.assertTrue(all(result["gates"].values()), result["gates"])


if __name__ == "__main__":
    unittest.main()
