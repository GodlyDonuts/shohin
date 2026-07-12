import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_nll_monitor import monitor_paths


class BuildNllMonitorTest(unittest.TestCase):
    def test_monitor_stays_outside_eval_decontamination_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifacts" / "monitors" / "english.jsonl"
            path, manifest = monitor_paths(output)
            self.assertEqual(path, output)
            self.assertEqual(manifest.name, "english.jsonl.manifest.json")
            with self.assertRaises(ValueError):
                monitor_paths(Path(tmp) / "artifacts" / "evals" / "english.jsonl")


if __name__ == "__main__":
    unittest.main()
