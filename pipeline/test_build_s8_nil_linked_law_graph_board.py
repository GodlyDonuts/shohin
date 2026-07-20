from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from build_s8_nil_linked_law_graph_board import (
    law_pools,
    verified_head_source_commit,
)


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

    def test_board_commit_must_be_current_clean_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(
                ("git", "config", "user.email", "test@example.com"),
                cwd=root,
                check=True,
            )
            subprocess.run(
                ("git", "config", "user.name", "Board Test"),
                cwd=root,
                check=True,
            )
            source = root / "source.py"
            source.write_text("VALUE = 1\n")
            subprocess.run(("git", "add", "source.py"), cwd=root, check=True)
            subprocess.run(
                ("git", "commit", "-qm", "freeze"),
                cwd=root,
                check=True,
            )
            commit = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(verified_head_source_commit(commit, root), commit)
            source.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                verified_head_source_commit(commit, root)


if __name__ == "__main__":
    unittest.main()
