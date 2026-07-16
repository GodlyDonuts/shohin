#!/usr/bin/env python3
"""Exhaustive tests for the R12 post-commit interface falsifier."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from pipeline import post_commit_interface_falsifier as pcif


class AlgebraTests(unittest.TestCase):
    def test_generated_updates_are_invertible_and_composition_matches(self) -> None:
        bundle = pcif.generate_challenges()
        sample = bundle["public"][0]
        total = pcif.compose_affine(sample.updates)
        self.assertTrue(pcif.is_invertible(total.matrix))
        for source in ((0, 0, 0, 0), (1, 2, 3, 4), (16, 15, 14, 13)):
            sequential = source
            for update in sample.updates:
                sequential = update.apply(sequential)
            self.assertEqual(total.apply(source), sequential)

    def test_public_and_decisive_effective_functionals_have_frozen_spans(self) -> None:
        bundle = pcif.generate_challenges()
        for challenge in bundle["public"]:
            coefficient, _ = pcif.effective_functional(challenge)
            self.assertEqual(coefficient[2:], (0, 0))
        for challenge in bundle["decisive"]:
            coefficient, _ = pcif.effective_functional(challenge)
            self.assertNotEqual(coefficient[2:], (0, 0))


class PacketTests(unittest.TestCase):
    def test_equal_width_packets_and_source_free_schema(self) -> None:
        state = (3, 5, 7, 11)
        complete = pcif.state_packet(state)
        motor = pcif.motor_packet(state)
        self.assertEqual(len(complete.values), 4)
        self.assertEqual(len(motor.values), 4)
        self.assertEqual(motor.values, (3, 5, 0, 0))
        self.assertTrue(pcif.sealed_packet_has_no_source_fields())
        self.assertEqual(pcif.validate_serialized_packet(complete.serialized()), complete)
        with self.assertRaises(pcif.AuditError):
            pcif.validate_serialized_packet(
                {"values": [3, 5, 0, 0], "source_id": "forbidden"}
            )

    def test_phase_one_hashes_are_challenge_seed_independent(self) -> None:
        first = pcif.phase_one_hashes()
        default = pcif.build_report(pcif.CHALLENGE_SEED)["phase_one"]
        alternate = pcif.build_report(pcif.ALTERNATE_CHALLENGE_SEED)["phase_one"]
        self.assertEqual(first, default)
        self.assertEqual(first, alternate)
        self.assertEqual(first["source_count"], 17**4)


class FrozenReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = pcif.build_report()

    def test_every_frozen_gate_passes(self) -> None:
        pcif.verify_report(self.report)
        self.assertTrue(self.report["pass"])
        self.assertTrue(all(self.report["gates"].values()))

    def test_public_controls_are_exact_for_both_packets(self) -> None:
        expected = 17**4
        self.assertEqual(len(self.report["public_results"]), len(pcif.DEPTHS))
        for row in self.report["public_results"]:
            self.assertEqual(row["state_correct"], expected)
            self.assertEqual(row["motor_correct"], expected)
            self.assertEqual(row["state_recoded_correct"], expected)
            self.assertEqual(row["motor_recoded_correct"], expected)
            self.assertFalse(row["decisive_outside_public_span"])

    def test_decisive_cells_are_exact_state_and_one_over_17_motor(self) -> None:
        expected_state = 17**4
        expected_motor = 17**3
        self.assertEqual(
            len(self.report["decisive_results"]),
            len(pcif.DEPTHS) * len(pcif.DECISIVE_KINDS),
        )
        for row in self.report["decisive_results"]:
            self.assertTrue(row["decisive_outside_public_span"])
            self.assertEqual(row["state_correct"], expected_state)
            self.assertEqual(row["state_recoded_correct"], expected_state)
            self.assertEqual(row["motor_correct"], expected_motor)
            self.assertEqual(row["motor_recoded_correct"], expected_motor)
            witness = row["collision_witness"]
            self.assertEqual(
                witness["shared_motor_packet"],
                pcif.motor_packet(tuple(witness["left_source"])).serialized(),
            )
            self.assertEqual(
                pcif.motor_packet(tuple(witness["left_source"])),
                pcif.motor_packet(tuple(witness["right_source"])),
            )
            self.assertNotEqual(witness["left_answer"], witness["right_answer"])
            self.assertNotEqual(witness["left_recoded"], witness["right_recoded"])

    def test_derangement_and_decoys(self) -> None:
        permutation = self.report["phase_two"]["output_permutation"]
        self.assertEqual(sorted(permutation), list(range(17)))
        self.assertTrue(all(index != value for index, value in enumerate(permutation)))
        self.assertTrue(self.report["source_pointer_decoy"]["rejected"])
        horizon = self.report["horizon_decoy"]
        self.assertTrue(horizon["passes_through_8"])
        self.assertTrue(horizon["rejected_at_9"])
        self.assertEqual(len(horizon["rows"]), 15)
        for row in horizon["rows"]:
            expected = 17**4 if row["depth"] <= 8 else 0
            self.assertEqual(row["correct"], expected)
            self.assertEqual(row["total_sources"], 17**4)

    def test_packet_reader_has_no_source_argument(self) -> None:
        coefficient = (2, 3, 5, 7)
        packet = pcif.SealedPacket((11, 13, 0, 0))
        self.assertEqual(
            pcif.packet_reader(packet, coefficient, 4),
            (2 * 11 + 3 * 13 + 4) % 17,
        )

    def test_generation_is_byte_deterministic(self) -> None:
        self.assertEqual(pcif.report_bytes(), pcif.report_bytes())
        default = pcif.build_report(pcif.CHALLENGE_SEED)
        alternate = pcif.build_report(pcif.ALTERNATE_CHALLENGE_SEED)
        self.assertEqual(default["phase_one"], alternate["phase_one"])
        self.assertNotEqual(
            default["phase_two"]["challenge_payload_sha256"],
            alternate["phase_two"]["challenge_payload_sha256"],
        )

    def test_immutable_writer_and_payload_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            pcif.write_immutable_report(output, self.report)
            self.assertTrue(output.is_file())
            mode = stat.S_IMODE(output.stat().st_mode)
            self.assertEqual(mode, 0o444)
            loaded = json.loads(output.read_text())
            pcif.verify_report(loaded)
            with self.assertRaises(FileExistsError):
                pcif.write_immutable_report(output, self.report)


if __name__ == "__main__":
    unittest.main()
