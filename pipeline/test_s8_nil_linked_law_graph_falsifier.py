from __future__ import annotations

import unittest
from unittest.mock import patch

from s8_nil_linked_law_graph_falsifier import run_falsifier


class S8NilLinkedLawGraphFalsifierTest(unittest.TestCase):
    def test_small_falsifier_passes(self) -> None:
        with (
            patch(
                "s8_nil_linked_law_graph_falsifier.MODULUS_BINDING_COUNTS",
                {5: 120, 7: 4},
            ),
            patch(
                "s8_nil_linked_law_graph_falsifier.PROGRAMS_PER_BINDING",
                2,
            ),
        ):
            report = run_falsifier(17)
        self.assertEqual(
            report["decision"],
            "admit_s8_nil_linked_law_graph_preregistration",
        )
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["scores"]["treatment"]["state_accuracy"], 1.0)

    def test_is_deterministic(self) -> None:
        with (
            patch(
                "s8_nil_linked_law_graph_falsifier.MODULUS_BINDING_COUNTS",
                {5: 120},
            ),
            patch(
                "s8_nil_linked_law_graph_falsifier.PROGRAMS_PER_BINDING",
                1,
            ),
        ):
            first = run_falsifier(23)
            second = run_falsifier(23)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
