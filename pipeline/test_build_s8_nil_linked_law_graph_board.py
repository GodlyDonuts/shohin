from __future__ import annotations

import unittest

from build_s8_nil_linked_law_graph_board import law_pools


class BuildS8NilLinkedLawGraphBoardTest(unittest.TestCase):
    def test_law_pools_are_disjoint_and_complete(self) -> None:
        for modulus in (5, 7, 11):
            pools = law_pools(modulus)
            sets = {name: set(values) for name, values in pools.items()}
            self.assertFalse(sets["train"] & sets["development"])
            self.assertFalse(sets["train"] & sets["confirmation"])
            self.assertFalse(sets["development"] & sets["confirmation"])
            self.assertEqual(
                len(set.union(*sets.values())),
                modulus * (modulus - 1),
            )
            self.assertGreaterEqual(min(map(len, pools.values())), 2)


if __name__ == "__main__":
    unittest.main()
