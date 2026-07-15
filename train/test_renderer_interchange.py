import unittest

import probe_renderer_interchange as probe


class RendererInterchangeTests(unittest.TestCase):
    def test_board_is_exactly_balanced(self):
        cells = probe.frozen_cells()
        self.assertEqual(len(cells), 12)
        self.assertEqual(sum(cell["crossed"] for cell in cells), 6)
        for operation in ("add", "multiply", "subtract"):
            selected = [cell for cell in cells if cell["operation"] == operation]
            self.assertEqual(len(selected), 4)
            self.assertEqual(sum(cell["crossed"] for cell in selected), 2)

    def test_crossed_candidates_disagree(self):
        for cell in probe.frozen_cells():
            self.assertEqual(
                cell["source_result"] == cell["local_result"],
                not cell["crossed"],
            )
            self.assertNotIn(str(cell["source_result"]), cell["prompt"])
            if cell["crossed"]:
                self.assertNotIn(str(cell["local_result"]), cell["prompt"])


if __name__ == "__main__":
    unittest.main()
