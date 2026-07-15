#!/usr/bin/env python3
"""Focused clean-admission and corruption rejection tests for WGRQ Stage A."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import audit_wgrq_falsifier_v1 as auditor  # noqa: E402
import generate_wgrq_falsifier_v1 as generator  # noqa: E402
import score_wgrq_falsifier_v1 as scorer  # noqa: E402


PREVIEW_EPISODES_PER_CELL = 64
TEST_CONTRACT = auditor.AuditContract(episodes_per_cell=PREVIEW_EPISODES_PER_CELL)


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(generator.canonical_json_bytes(row) for row in rows)


def build_preview_report(
    episodes: list[dict],
    ledger: list[dict],
    prf_ledger: dict,
) -> dict:
    symbolic = generator.run_stage_a_symbolic_gates()
    generation_contract = copy.deepcopy(generator.generation_contract(symbolic))
    generation_contract["episodes_per_cell"] = TEST_CONTRACT.episodes_per_cell
    generation_contract["total_episodes"] = TEST_CONTRACT.total_episodes
    generation_contract["total_ordinary_one_bit_read_calls"] = TEST_CONTRACT.total_calls

    cells = []
    for cell_index, (n, length_band) in enumerate(TEST_CONTRACT.cells):
        stats = generator._new_cell_stats(n, length_band)
        for local_index in range(TEST_CONTRACT.episodes_per_cell):
            global_index = cell_index * TEST_CONTRACT.episodes_per_cell + local_index
            generator._update_cell_stats(
                stats,
                episodes[global_index],
                ledger[
                    global_index * auditor.ORDINARY_CALLS_PER_EPISODE : (
                        global_index + 1
                    )
                    * auditor.ORDINARY_CALLS_PER_EPISODE
                ],
            )
        cells.append(generator._finalize_cell_stats(stats))

    transcript_payload = jsonl_bytes(episodes)
    ledger_payload = jsonl_bytes(ledger)
    parity_episodes = sum(cell["parity_obstruction_episodes"] for cell in cells)
    return {
        "schema": generator.REPORT_SCHEMA,
        "passed": True,
        "generation_contract": generation_contract,
        "symbolic_gates": symbolic,
        "cells": cells,
        "totals": {
            "cells": len(TEST_CONTRACT.cells),
            "episodes": len(episodes),
            "histories": len(episodes) * auditor.HISTORIES_PER_EPISODE,
            "ordinary_one_bit_read_calls": len(ledger),
            "returned_answer_bits": len(ledger),
            "batches": len(episodes) // auditor.FROZEN_BATCH_SIZE,
        },
        "frozen_call_ledger": {
            "schema": generator.LEDGER_SCHEMA,
            "first_call_id": 0,
            "last_call_id": len(ledger) - 1,
            "rows": len(ledger),
            "one_bit_read_calls": len(ledger),
            "model_dependent_calls": 0,
            "equivalence_oracle_calls": 0,
            "counterexample_oracle_calls": 0,
        },
        "balance": {
            "pair_labels_per_episode": {"equivalent": 1, "non_equivalent": 1},
            "depth_stratification_rule": "floor/ceiling balanced then fixed-PRF shuffled",
            "gadget_stratification_rule": "balanced within depth over every feasible gadget",
            "maximum_depth_flip_parity_obstruction": {
                "affected_episodes": parity_episodes,
                "reported_separately": True,
                "reason": auditor.PARITY_REASON,
            },
        },
        "prf_ledger": prf_ledger,
        "artifacts": {
            "transcript": {
                "schema": generator.SCHEMA,
                "rows": len(episodes),
                "bytes": len(transcript_payload),
                "sha256": auditor.sha256_bytes(transcript_payload),
            },
            "ordinary_call_ledger": {
                "schema": generator.LEDGER_SCHEMA,
                "rows": len(ledger),
                "bytes": len(ledger_payload),
                "sha256": auditor.sha256_bytes(ledger_payload),
            },
        },
        "hashes": {
            "generation_contract_sha256": auditor.sha256_bytes(
                generator.canonical_json_bytes(generation_contract)
            ),
            "transcript_sha256": auditor.sha256_bytes(transcript_payload),
            "ordinary_call_ledger_sha256": auditor.sha256_bytes(ledger_payload),
        },
    }


class WgrqAuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        episodes, ledger, prf = generator.generate_preview(
            episodes_per_cell=PREVIEW_EPISODES_PER_CELL
        )
        cls.base_episodes = episodes
        cls.base_ledger = ledger
        cls.base_report = build_preview_report(episodes, ledger, prf)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.transcript = self.root / "transcript.jsonl"
        self.ledger = self.root / "ledger.jsonl"
        self.generation_report = self.root / "generation-report.json"
        self.episodes_data = copy.deepcopy(self.base_episodes)
        self.ledger_data = copy.deepcopy(self.base_ledger)
        self.report_data = copy.deepcopy(self.base_report)
        self.write_bundle()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_bundle(self) -> None:
        transcript_payload = jsonl_bytes(self.episodes_data)
        ledger_payload = jsonl_bytes(self.ledger_data)
        self.transcript.write_bytes(transcript_payload)
        self.ledger.write_bytes(ledger_payload)
        self.report_data["artifacts"]["transcript"].update(
            {
                "rows": len(self.episodes_data),
                "bytes": len(transcript_payload),
                "sha256": auditor.sha256_bytes(transcript_payload),
            }
        )
        self.report_data["artifacts"]["ordinary_call_ledger"].update(
            {
                "rows": len(self.ledger_data),
                "bytes": len(ledger_payload),
                "sha256": auditor.sha256_bytes(ledger_payload),
            }
        )
        self.report_data["hashes"]["transcript_sha256"] = auditor.sha256_bytes(
            transcript_payload
        )
        self.report_data["hashes"]["ordinary_call_ledger_sha256"] = (
            auditor.sha256_bytes(ledger_payload)
        )
        self.generation_report.write_bytes(
            generator.canonical_json_bytes(self.report_data, pretty=True)
        )

    def audit(self, contract: auditor.AuditContract = TEST_CONTRACT) -> dict:
        return auditor.audit_bundle(
            transcript_path=self.transcript,
            ledger_path=self.ledger,
            generation_report_path=self.generation_report,
            contract=contract,
        )

    def assert_rejected(self, category: str) -> dict:
        report = self.audit()
        self.assertFalse(report["all_checks_pass"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["errors"][0]["category"], category, report["errors"])
        return report

    def test_clean_generator_preview_passes_end_to_end(self) -> None:
        report = self.audit()
        self.assertTrue(report["all_checks_pass"], report["errors"])
        self.assertEqual(report["transcript"]["rows"], 384)
        self.assertEqual(report["ordinary_call_ledger"]["rows"], 12_288)
        self.assertEqual(report["expected_ordinary_one_bit_answer_calls"], 12_288)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(scorer.verify_symbolic_audit(report)["passed"])
        self.assertNotIn("endpoint", json.dumps(report["cells"], sort_keys=True))

    def test_rebound_wrong_ordinary_answer_is_rejected(self) -> None:
        probe = self.episodes_data[0]["histories"][0]["probes"][0]
        probe["answer"] ^= 1
        self.ledger_data[0]["answer"] = probe["answer"]
        self.write_bundle()
        self.assert_rejected("ordinary_answers")

    def test_rebound_relation_and_witness_forgery_is_rejected(self) -> None:
        cases = (
            ("matrix", "relation"),
            ("witness", "witness"),
        )
        for name, category in cases:
            with self.subTest(name=name):
                self.episodes_data = copy.deepcopy(self.base_episodes)
                self.ledger_data = copy.deepcopy(self.base_ledger)
                self.report_data = copy.deepcopy(self.base_report)
                if name == "matrix":
                    self.episodes_data[0]["equivalence_label_matrix"][0][1] = 0
                else:
                    self.episodes_data[0]["first_distinguishing_witness_mask"] = [0] * 8
                self.write_bundle()
                self.assert_rejected(category)

    def test_hidden_physical_state_field_is_rejected(self) -> None:
        self.episodes_data[0]["physical_state"] = 0
        self.write_bundle()
        self.assert_rejected("hidden_state")

    def test_rebound_history_hash_tamper_is_rejected(self) -> None:
        forged = "0" * 64
        self.episodes_data[0]["histories"][0]["history_sha256"] = forged
        for call in self.ledger_data[:8]:
            call["history_sha256"] = forged
        self.write_bundle()
        self.assert_rejected("hash")

    def test_duplicate_source_history_is_rejected_with_matching_ledger(self) -> None:
        episode = self.episodes_data[0]
        duplicate = copy.deepcopy(episode["histories"][0])
        duplicate["history_index"] = 1
        duplicate["history_id"] = "{}:h1".format(episode["episode_id"])
        duplicate["role"] = "equivalent_b"
        first_call = episode["oracle_call_span"]["first_call_id"] + 8
        for probe_index, probe in enumerate(duplicate["probes"]):
            probe["oracle_call_id"] = first_call + probe_index
        episode["histories"][1] = duplicate
        for probe_index in range(8):
            source = copy.deepcopy(self.ledger_data[probe_index])
            source["call_id"] = first_call + probe_index
            source["history_index"] = 1
            self.ledger_data[8 + probe_index] = source
        self.write_bundle()
        self.assert_rejected("duplicate")

    def test_duplicate_or_trailing_call_rows_are_rejected(self) -> None:
        self.ledger_data[1] = copy.deepcopy(self.ledger_data[0])
        self.write_bundle()
        self.assert_rejected("call_ledger")

        self.episodes_data = copy.deepcopy(self.base_episodes)
        self.ledger_data = copy.deepcopy(self.base_ledger)
        self.report_data = copy.deepcopy(self.base_report)
        trailing = copy.deepcopy(self.ledger_data[-1])
        trailing["call_id"] += 1
        self.ledger_data.append(trailing)
        self.write_bundle()
        self.assert_rejected("overlap")

    def test_partial_artifacts_are_rejected_after_hash_rebinding(self) -> None:
        self.episodes_data.pop()
        del self.ledger_data[-auditor.ORDINARY_CALLS_PER_EPISODE :]
        self.write_bundle()
        self.assert_rejected("partial")

    def test_shortest_depth_rebalance_is_detected_from_valid_rows(self) -> None:
        first_cell = self.episodes_data[:PREVIEW_EPISODES_PER_CELL]
        target_index = next(
            index
            for index, episode in enumerate(first_cell)
            if episode["pairs"]["non_equivalent"]["shortest_witness_depth"] == 0
        )
        source = next(
            episode
            for episode in first_cell
            if episode["pairs"]["non_equivalent"]["shortest_witness_depth"] == 1
        )
        replacement = copy.deepcopy(source)
        new_id = auditor.episode_id(4, "le_2n", target_index)
        replacement["episode_id"] = new_id
        replacement["global_episode_index"] = target_index
        replacement["batch_index"] = target_index // auditor.FROZEN_BATCH_SIZE
        replacement["batch_offset"] = target_index % auditor.FROZEN_BATCH_SIZE
        replacement["cell"]["cell_episode_index"] = target_index
        first_call = target_index * auditor.ORDINARY_CALLS_PER_EPISODE
        for history_index, history in enumerate(replacement["histories"]):
            history["history_id"] = "{}:h{}".format(new_id, history_index)
            for probe_index, probe in enumerate(history["probes"]):
                probe["oracle_call_id"] = (
                    first_call
                    + history_index * auditor.PROBES_PER_HISTORY
                    + probe_index
                )
        replacement["oracle_call_span"] = {
            "first_call_id": first_call,
            "last_call_id": first_call + auditor.ORDINARY_CALLS_PER_EPISODE - 1,
            "ordinary_one_bit_read_calls": auditor.ORDINARY_CALLS_PER_EPISODE,
        }
        replay = auditor.audit_episode(
            replacement,
            expected_global_index=target_index,
            contract=TEST_CONTRACT,
        )
        self.episodes_data[target_index] = replacement
        self.ledger_data[
            first_call : first_call + auditor.ORDINARY_CALLS_PER_EPISODE
        ] = list(replay.expected_ledger)
        self.write_bundle()
        self.assert_rejected("strata")

    def test_cancellation_parity_flag_tamper_is_rejected(self) -> None:
        tight = next(
            episode
            for episode in self.episodes_data
            if episode["pairs"]["non_equivalent"]["shortest_witness_depth"]
            == episode["cell"]["n"] - 2
        )
        tight["balance"]["maximum_depth_flip_parity_obstruction"] = False
        self.write_bundle()
        self.assert_rejected("balance")

    def test_generation_report_hash_count_ledger_and_prf_tampering_is_rejected(
        self,
    ) -> None:
        cases = (
            ("hash", "hash"),
            ("cell", "strata"),
            ("ledger", "call_ledger"),
            ("prf", "prf"),
        )
        for name, category in cases:
            with self.subTest(name=name):
                self.episodes_data = copy.deepcopy(self.base_episodes)
                self.ledger_data = copy.deepcopy(self.base_ledger)
                self.report_data = copy.deepcopy(self.base_report)
                if name == "hash":
                    self.report_data["hashes"]["transcript_sha256"] = "0" * 64
                elif name == "cell":
                    self.report_data["cells"][0]["episodes"] += 1
                elif name == "ledger":
                    self.report_data["frozen_call_ledger"][
                        "equivalence_oracle_calls"
                    ] = 1
                else:
                    self.report_data["prf_ledger"]["seed_ascii"] = "forged"
                self.generation_report.write_bytes(
                    generator.canonical_json_bytes(self.report_data, pretty=True)
                )
                self.assert_rejected(category)

    def test_noncanonical_and_default_partial_inputs_fail_closed(self) -> None:
        lines = self.transcript.read_bytes().splitlines(keepends=True)
        lines[0] = lines[0][:-1] + b" \n"
        payload = b"".join(lines)
        self.transcript.write_bytes(payload)
        self.report_data["artifacts"]["transcript"].update(
            {
                "bytes": len(payload),
                "sha256": auditor.sha256_bytes(payload),
            }
        )
        self.report_data["hashes"]["transcript_sha256"] = auditor.sha256_bytes(payload)
        self.generation_report.write_bytes(
            generator.canonical_json_bytes(self.report_data, pretty=True)
        )
        self.assert_rejected("canonical_bytes")

        self.episodes_data = copy.deepcopy(self.base_episodes)
        self.ledger_data = copy.deepcopy(self.base_ledger)
        self.report_data = copy.deepcopy(self.base_report)
        self.write_bundle()
        default_report = self.audit(auditor.AuditContract())
        self.assertFalse(default_report["all_checks_pass"])

    def test_cli_surface_is_fixed_and_rejects_reduced_fixture(self) -> None:
        parser_dests = {action.dest for action in auditor.build_parser()._actions}
        self.assertEqual(
            parser_dests,
            {"help", "transcript", "ledger", "generation_report", "out"},
        )
        output = self.root / "audit.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "pipeline" / "audit_wgrq_falsifier_v1.py"),
                "--transcript",
                str(self.transcript),
                "--ledger",
                str(self.ledger),
                "--generation-report",
                str(self.generation_report),
                "--out",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads(output.read_text())["all_checks_pass"])

    def test_symbolic_minimum_counts_are_exact(self) -> None:
        symbolic = auditor.symbolic_gate_audit()
        self.assertEqual(symbolic["3"]["check_count"], 152)
        self.assertEqual(symbolic["6"]["check_count"], 20_672)
        self.assertTrue(all(item["passed"] for item in symbolic.values()))


if __name__ == "__main__":
    unittest.main()
