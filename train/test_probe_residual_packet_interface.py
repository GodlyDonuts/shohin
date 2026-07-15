import os
import tempfile
import unittest
from pathlib import Path

from probe_residual_packet_interface import PROBES, SCHEMA, write_immutable_json


class ResidualPacketProbeTests(unittest.TestCase):
    def test_probe_contract_is_fixed_and_source_separated(self):
        self.assertEqual(SCHEMA, "raw_residual_packet_interface_probe_v1")
        self.assertEqual(len(PROBES), 5)
        self.assertEqual(len({row["id"] for row in PROBES}), 5)
        self.assertEqual(
            [row["kind"] for row in PROBES],
            ["compiler", "compiler", "updater", "updater", "halt"],
        )
        for row in PROBES[:2]:
            self.assertIn("Problem:", row["prompt"])
        for row in PROBES[2:]:
            self.assertNotIn("Problem:", row["prompt"])
            self.assertIn("Observed result:", row["prompt"])

    def test_output_is_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            digest = write_immutable_json(path, {"schema": SCHEMA})
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                write_immutable_json(path, {"schema": SCHEMA})
            os.chmod(path, 0o600)


if __name__ == "__main__":
    unittest.main()
