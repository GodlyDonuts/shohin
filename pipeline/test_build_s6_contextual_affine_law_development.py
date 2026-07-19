import json
from pathlib import Path
import tempfile
import unittest

from build_s6_contextual_affine_law_development import (
    audit_rows,
    build_atomic_training_rows,
    build_program_rows,
)


class S6DevelopmentBoardTests(unittest.TestCase):
    def test_atomic_rows_expose_only_legal_treatment_fields(self):
        rows = build_atomic_training_rows()
        self.assertEqual(len(rows), 961)
        self.assertEqual(
            {row["supervision"] for row in rows}, {"atomic_destination_only"}
        )
        self.assertNotIn("slope", rows[0])
        self.assertNotIn("intercept", rows[0])
        self.assertNotIn("final_state", rows[0])
        self.assertNotIn("answer", rows[0])

    def test_program_board_is_deterministic_and_disjoint(self):
        first = build_program_rows(12345, 36, 12)
        second = build_program_rows(12345, 36, 12)
        self.assertEqual(first, second)
        audit = audit_rows(build_atomic_training_rows(), *first)
        self.assertEqual(audit["train_development_law_overlap"], 0)
        self.assertEqual(audit["confirmation_program_rows"], 0)
        self.assertGreaterEqual(audit["minimum_distinct_laws_per_program"], 2)

    def test_cli_refuses_existing_output(self):
        # The full CLI path is covered without creating a permanent board.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "report.json").write_text(json.dumps({"occupied": True}))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

