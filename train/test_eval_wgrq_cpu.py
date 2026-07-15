#!/usr/bin/env python3
"""Focused contracts for WGRQ Stage-A confirmation generation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "train", ROOT / "pipeline"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_wgrq_cpu import (  # noqa: E402
    ANSWERS_PER_EPISODE,
    BRANCHES_PER_HISTORY,
    CHECKPOINT_SCHEMA,
    CONFIRMATION_SCHEMA,
    HISTORY_ROLES,
    PACKET_BITS,
    READER_TASK_KEYS,
    REPORT_SCHEMA,
    STRATA,
    WRITER_TASK_KEYS,
    OracleFacade,
    WriterReaderBoundary,
    _binary_mask,
    _build_checkpoint_hash_document,
    _build_confirmation_document,
    _build_evaluation_report,
    _expected_checkpoint_pairs,
    _make_confirmation_episode,
    _packet_bytes,
    _training_model_state_hash,
    _validate_checkpoint_hash_document,
    _validate_confirmation_document,
    _validate_final_checkpoint,
    canonical_json_bytes,
    checkpoint_weights_wire,
    generate_confirmation,
    score_episode,
    sha256_bytes,
)
from score_wgrq_falsifier_v1 import _parse_evaluation  # noqa: E402
from wgrq_state_machine import HardBitDWEPRLearner  # noqa: E402


class CheckpointFreezeTests(unittest.TestCase):
    def test_hash_freeze_is_exact_and_detects_checkpoint_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"final checkpoint")
            pair = ("WGRQ-shortest", 17011)
            document = _build_checkpoint_hash_document(
                [{"arm": pair[0], "seed": pair[1], "path": str(checkpoint)}],
                frozenset((pair,)),
            )
            self.assertEqual(document["schema"], CHECKPOINT_SCHEMA)
            self.assertTrue(document["frozen"])
            indexed = _validate_checkpoint_hash_document(
                document,
                frozenset((pair,)),
                verify_files=True,
            )
            self.assertEqual(indexed[pair]["sha256"], sha256_bytes(b"final checkpoint"))
            with self.assertRaises(ValueError):
                _build_checkpoint_hash_document(
                    [{"arm": pair[0], "seed": pair[1], "path": str(checkpoint)}],
                    _expected_checkpoint_pairs(),
                )
            checkpoint.write_bytes(b"changed after freeze")
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                _validate_checkpoint_hash_document(
                    document,
                    frozenset((pair,)),
                    verify_files=True,
                )

    def test_confirmation_refuses_an_incomplete_checkpoint_grid_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            pair = ("WGRQ-shortest", 17011)
            reduced = _build_checkpoint_hash_document(
                [{"arm": pair[0], "seed": pair[1], "path": str(checkpoint)}],
                frozenset((pair,)),
            )
            manifest = root / "hashes.json"
            manifest.write_text(json.dumps(reduced))
            output = root / "confirmation.json"
            with self.assertRaisesRegex(ValueError, "exact checkpoint grid"):
                generate_confirmation(manifest, output)
            self.assertFalse(output.exists())

    def test_final_checkpoint_contract_binds_weights_and_training_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            torch.manual_seed(23)
            model = HardBitDWEPRLearner()
            metadata = {
                "protocol": "wgrq_dwepr_cpu_stage_a_v1",
                "arm": "WGRQ-shortest",
                "seed": 17011,
                "paired_seeds": [
                    17011, 27103, 38119, 49201, 50311, 61403,
                    72503, 83609, 94709, 105019, 116027, 127031,
                ],
                "transcript_sha256": "a" * 64,
                "audit_report_sha256": "b" * 64,
                "episodes": 18_432,
                "ordinary_answer_calls": 589_824,
                "batch_size": 64,
                "epochs": 4,
                "updates": 1_152,
                "warmup_updates": 64,
                "parameter_count": 5_136,
                "parameter_dtype": "torch.float32",
                "packet_bits": 15,
                "all_loss_terms_eager": True,
                "final_model_sha256": _training_model_state_hash(model),
            }
            torch.save({"model_state": model.state_dict(), "wgrq_cpu": metadata}, path)
            contract = _validate_final_checkpoint(path, "WGRQ-shortest", 17011)
            self.assertEqual(contract["transcript_sha256"], "a" * 64)
            self.assertEqual(contract["final_model_sha256"], metadata["final_model_sha256"])
            metadata["final_model_sha256"] = "c" * 64
            torch.save({"model_state": model.state_dict(), "wgrq_cpu": metadata}, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_final_checkpoint(path, "WGRQ-shortest", 17011)


class ConfirmationGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            pair = ("WGRQ-shortest", 17011)
            cls.hash_document = _build_checkpoint_hash_document(
                [{"arm": pair[0], "seed": pair[1], "path": str(checkpoint)}],
                frozenset((pair,)),
            )
        cls.document = _build_confirmation_document(
            cls.hash_document,
            episodes_per_stratum=15,
        )

    def test_three_ood_strata_have_exact_episode_geometry_and_call_ledger(self):
        document = self.document
        self.assertEqual(document["schema"], CONFIRMATION_SCHEMA)
        self.assertEqual(document["score_unit"], "episode_exact")
        self.assertEqual(document["ordinary_oracle_answer_calls"], 3 * 15 * ANSWERS_PER_EPISODE)
        self.assertEqual(document["oracle_call_ledger"]["equivalence_oracle_calls"], 0)
        self.assertEqual(document["oracle_call_ledger"]["witness_oracle_calls"], 0)
        self.assertEqual(len(document["episodes"]), 45)
        _validate_confirmation_document(
            document,
            self.hash_document,
            expected_episodes_per_stratum=15,
        )
        counts = {stratum.name: 0 for stratum in STRATA}
        for episode in document["episodes"]:
            counts[episode["stratum"]] += 1
            self.assertEqual(len(episode["histories"]), 4)
            self.assertEqual([row["role"] for row in episode["histories"]], list(HISTORY_ROLES))
            self.assertEqual(len(episode["continuations"]), BRANCHES_PER_HISTORY)
            self.assertEqual(episode["ordinary_oracle_answer_calls"], ANSWERS_PER_EPISODE)
            self.assertNotIn("endpoint", canonical_json_bytes(episode).decode("ascii"))
            for history in episode["histories"]:
                self.assertEqual(len(history["expected_answers"]), BRANCHES_PER_HISTORY)
        self.assertEqual(set(counts.values()), {15})
        full = [episode for episode in document["episodes"] if episode["stratum"] == "full_ood"]
        self.assertIn(14, {episode["non_equivalent_witness_depth"] for episode in full})
        self.assertTrue(
            all(
                len(history["source_events"]) > 8 * episode["n"]
                for episode in document["episodes"]
                if episode["stratum"] in {"length_ood", "full_ood"}
                for history in episode["histories"]
            )
        )

    def test_generation_and_serialization_are_byte_deterministic(self):
        oracle = OracleFacade()
        first = _make_confirmation_episode(STRATA[2], 0, oracle)
        second = _make_confirmation_episode(STRATA[2], 0, oracle)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(first["non_equivalent_witness_depth"], 14)
        witness_positions = [
            index
            for index, continuation in enumerate(first["continuations"])
            if len(continuation) == 14
        ]
        self.assertTrue(witness_positions)
        self.assertTrue(
            all(
                first["histories"][2]["expected_answers"][index]
                != first["histories"][3]["expected_answers"][index]
                for index in witness_positions
            )
        )

    def test_relations_are_derived_without_equivalence_or_witness_oracles(self):
        shared = OracleFacade()

        class PublicAnswersOnly:
            def state_after(self, n, events):
                return shared.state_after(n, events)

            def answer_after(self, state, continuation, n):
                return shared.answer_after(state, continuation, n)

            def __getattr__(self, name):
                raise AssertionError("unexpected relational oracle call: {}".format(name))

        episode = _make_confirmation_episode(STRATA[1], 7, PublicAnswersOnly())
        self.assertEqual(episode["ordinary_oracle_answer_calls"], ANSWERS_PER_EPISODE)


class ProcessBoundaryTests(unittest.TestCase):
    def test_writer_and_reader_are_fresh_disjoint_children_with_allowlisted_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(17)
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "model_state": HardBitDWEPRLearner().state_dict(),
                    "wgrq_cpu": {
                        "arm": "WGRQ-shortest",
                        "seed": 17011,
                        "packet_bits": PACKET_BITS,
                    },
                },
                checkpoint,
            )
            weights, digest = checkpoint_weights_wire(checkpoint, "WGRQ-shortest", 17011)
            with WriterReaderBoundary(weights, digest) as boundary:
                writer = boundary.writer.request(
                    {
                        "role": "write",
                        "scale_mask": _binary_mask(8),
                        "source_events": [0, 1, 0, 0, 1],
                    }
                )
                packet = tuple(writer["packet"])
                self.assertEqual(len(packet), PACKET_BITS)
                self.assertEqual(writer["input_keys"], sorted(WRITER_TASK_KEYS))
                readers = [
                    boundary.reader.request(
                        {
                            "role": "read",
                            "scale_mask": _binary_mask(8),
                            "packet": list(packet),
                            "continuation": [0] * (index % 8),
                        }
                    )
                    for index in range(BRANCHES_PER_HISTORY)
                ]
                self.assertEqual(len({reader["pid"] for reader in readers}), BRANCHES_PER_HISTORY)
                self.assertTrue(all(writer["pid"] != reader["pid"] for reader in readers))
                self.assertTrue(all(writer["broker_pid"] != reader["broker_pid"] for reader in readers))
                self.assertTrue(all(reader["input_keys"] == sorted(READER_TASK_KEYS) for reader in readers))
                self.assertTrue(all(reader["packet_reuse_identical"] for reader in readers))
                self.assertTrue(all(reader["cross_branch_memory_absent"] for reader in readers))
                self.assertTrue(all(reader["oracle_modules_absent"] for reader in readers))
                self.assertTrue(all(reader["forbidden_reader_inputs_absent"] for reader in readers))
                self.assertTrue(all(reader["model_buffers_absent"] for reader in readers))
                self.assertTrue(all(reader["rng_state_unchanged"] for reader in readers))
                self.assertTrue(
                    all(
                        reader["packet_sha256_after"] == sha256_bytes(_packet_bytes(packet))
                        for reader in readers
                    )
                )
                leaking_task = {
                    "role": "read",
                    "scale_mask": _binary_mask(8),
                    "packet": list(packet),
                    "continuation": [],
                    "source_events": [],
                }
                with self.assertRaisesRegex(RuntimeError, "allowlisted"):
                    boundary.reader.request(leaking_task)


def fake_process_requests(episode, *, wrong_history: int | None = None):
    expected = [list(history["expected_answers"]) for history in episode["histories"]]
    writer_index = 0
    reader_indices = [0] * len(expected)

    def writer(task):
        nonlocal writer_index
        index = writer_index
        writer_index += 1
        packet = [(index >> bit) & 1 for bit in range(2)] + [0] * (PACKET_BITS - 2)
        digest = sha256_bytes(_packet_bytes(packet))
        return {
            "role": "writer_result",
            "pid": 100 + index,
            "ppid": 10,
            "broker_pid": 10,
            "input_keys": sorted(task),
            "packet": packet,
            "packet_sha256": digest,
            "exactly_15_serialized_bits": True,
            "masked_bits_zero": True,
            "model_weights_unchanged": True,
        }

    def reader(task):
        packet = task["packet"]
        index = int(packet[0]) + 2 * int(packet[1])
        branch_index = reader_indices[index]
        reader_indices[index] += 1
        prediction = int(expected[index][branch_index])
        if wrong_history == index and branch_index == 0:
            prediction ^= 1
        digest = sha256_bytes(_packet_bytes(packet))
        return {
            "role": "reader_result",
            "pid": 200 + index * BRANCHES_PER_HISTORY + branch_index,
            "ppid": 20,
            "broker_pid": 20,
            "input_keys": sorted(task),
            "prediction": prediction,
            "packet_sha256_before": digest,
            "packet_sha256_after": digest,
            "packet_reuse_identical": True,
            "exactly_15_serialized_bits": True,
            "masked_bits_zero": True,
            "forbidden_reader_inputs_absent": True,
            "oracle_modules_absent": True,
            "model_weights_unchanged": True,
            "model_buffers_absent": True,
            "rng_state_unchanged": True,
            "cross_branch_memory_absent": True,
        }

    return writer, reader


class EpisodeScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = _make_confirmation_episode(STRATA[0], 3, OracleFacade())

    def test_all_normal_interchange_donor_and_integrity_checks_form_one_score(self):
        writer, reader = fake_process_requests(self.episode)
        result = score_episode(self.episode, writer, reader)
        self.assertTrue(result["episode_exact"])
        self.assertEqual(result["failed_checks"], [])
        self.assertTrue(result["checks"]["equivalent_interchange"])
        self.assertTrue(result["checks"]["non_equivalent_donor_follow"])
        self.assertNotIn("probe_accuracy", json.dumps(result, sort_keys=True))

    def test_one_wrong_branch_fails_the_whole_episode(self):
        writer, reader = fake_process_requests(self.episode, wrong_history=0)
        result = score_episode(self.episode, writer, reader)
        self.assertFalse(result["episode_exact"])
        self.assertIn("normal_reads", result["failed_checks"])

    def test_report_matches_the_existing_stage_a_scorer_contract(self):
        checks = {
            "normal_reads": True,
            "equivalent_interchange": True,
            "non_equivalent_donor_follow": True,
            "writer_reader_process_boundary": True,
            "exactly_15_serialized_bits": True,
            "masked_bits_zero": True,
            "packet_byte_reuse": True,
            "writer_input_allowlist": True,
            "reader_input_allowlist": True,
            "source_oracle_cache_leak_absent": True,
            "cross_branch_memory_absent": True,
            "ordinary_answer_geometry": True,
        }
        results = [
            {
                "id": "{}-{:04d}".format(stratum.name, index),
                "stratum": stratum.name,
                "episode_exact": True,
                "checks": checks,
                "failed_checks": [],
            }
            for stratum in STRATA
            for index in range(1024)
        ]
        report = _build_evaluation_report(
            results,
            arm="WGRQ-shortest",
            seed=17011,
            checkpoint_path="/frozen/checkpoint.pt",
            checkpoint_sha256="a" * 64,
            checkpoint_hashes_document_sha256="b" * 64,
            confirmation_document_sha256="c" * 64,
            fixed_weights_transport_sha256="d" * 64,
        )
        self.assertEqual(report["schema"], REPORT_SCHEMA)
        parsed, failures = _parse_evaluation(
            report,
            expected_arm="wgrq_shortest",
            expected_seed=17011,
            checkpoint_sha256="a" * 64,
        )
        self.assertEqual(failures, [])
        self.assertTrue(all(len(rows) == 1024 for rows in parsed.values()))
        self.assertTrue(all(all(value == 1 for value in rows.values()) for rows in parsed.values()))


if __name__ == "__main__":
    unittest.main()
