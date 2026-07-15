#!/usr/bin/env python3
"""Focused immutability and information-ledger tests for WGRQ generation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from generate_wgrq_falsifier_v1 import (  # noqa: E402
    EPISODES_PER_CELL,
    FROZEN_BATCH_SIZE,
    FROZEN_GENERATION_CONTRACT_SHA256,
    FROZEN_ORDINARY_CALL_LEDGER_SHA256,
    FROZEN_PRF_SEED,
    FROZEN_REPORT_SHA256,
    FROZEN_TRANSCRIPT_SHA256,
    HISTORIES_PER_EPISODE,
    LENGTH_BANDS,
    ORDINARY_CALLS_PER_EPISODE,
    PROBES_PER_HISTORY,
    TOTAL_EPISODES,
    TOTAL_ORDINARY_CALLS,
    TRAINING_SCALES,
    FixedPRF,
    canonical_json_bytes,
    cell_schedule,
    endpoint_quad_bank,
    feasible_gadget_names,
    generate_preview,
    generation_contract,
    preflight_immutable_outputs,
    prf_block,
    probe_rotation_bank,
    records_sha256,
    source_gadgets,
)
from wgrq_residual_oracle import (  # noqa: E402
    FLIP,
    ROTATE,
    apply_word,
    event_counts,
    shortest_witness_depth,
)


class FixedPRFTests(unittest.TestCase):
    def test_exact_counter_block_formula(self) -> None:
        expected = hashlib.sha256(
            b"seed" + b"\x00" + b"domain" + (7).to_bytes(8, "big")
        ).hexdigest()
        self.assertEqual(expected, "9e4bbde0ac4474bc7788ce98a3e89c57857a50ab6145c89c7b998fdc7be982de")
        self.assertEqual(prf_block(b"seed", "domain", 7).hex(), expected)

    def test_domain_counters_and_selection_are_reproducible(self) -> None:
        left = FixedPRF()
        right = FixedPRF()
        left_values = [left.randbelow(17, "a") for _ in range(20)]
        right_values = [right.randbelow(17, "a") for _ in range(20)]
        self.assertEqual(left_values, right_values)
        self.assertEqual(left.snapshot(), right.snapshot())
        self.assertEqual(left.counters, {"a": 20})

    def test_rejection_path_is_only_unbiased_bank_selection(self) -> None:
        class ScriptedPRF(FixedPRF):
            def __init__(self) -> None:
                super().__init__(FROZEN_PRF_SEED)
                self.blocks = [(1 << 256) - 1, 0]

            def next_block(self, domain: str) -> bytes:
                self.blocks_used += 1
                return self.blocks.pop(0).to_bytes(32, "big")

        prf = ScriptedPRF()
        self.assertEqual(prf.randbelow(10, "scripted"), 0)
        self.assertEqual(prf.rejected_blocks, 1)
        self.assertEqual(prf.blocks_used, 2)

    def test_invalid_prf_inputs_fail_closed(self) -> None:
        with self.assertRaises(UnicodeEncodeError):
            prf_block(b"seed", "not-ascii-\N{SNOWMAN}", 0)
        with self.assertRaises(ValueError):
            prf_block(b"seed", "domain", -1)
        with self.assertRaises(ValueError):
            FixedPRF().randbelow(0, "domain")


class FrozenScheduleTests(unittest.TestCase):
    def test_probe_banks_are_exact(self) -> None:
        self.assertEqual(probe_rotation_bank(4), (0, 1, 2, 3, 0, 1, 2, 3))
        self.assertEqual(probe_rotation_bank(6), (0, 1, 2, 3, 4, 5, 0, 1))
        self.assertEqual(probe_rotation_bank(8), (0, 1, 2, 3, 4, 5, 6, 7))
        with self.assertRaises(ValueError):
            probe_rotation_bank(16)

    def test_full_cell_schedules_are_depth_and_gadget_balanced(self) -> None:
        prf = FixedPRF()
        for n in TRAINING_SCALES:
            for length_band in LENGTH_BANDS:
                schedule = cell_schedule(n, length_band, EPISODES_PER_CELL, prf)
                self.assertEqual(len(schedule), EPISODES_PER_CELL)
                depth_counts = Counter(depth for depth, _ in schedule)
                self.assertEqual(set(depth_counts), set(range(n - 1)))
                self.assertLessEqual(max(depth_counts.values()) - min(depth_counts.values()), 1)
                gadgets_by_depth: dict[int, Counter[str]] = defaultdict(Counter)
                for depth, gadget in schedule:
                    self.assertIn(gadget, feasible_gadget_names(n, length_band, depth))
                    gadgets_by_depth[depth][gadget] += 1
                for depth, counts in gadgets_by_depth.items():
                    self.assertEqual(set(counts), set(feasible_gadget_names(n, length_band, depth)))
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_direct_endpoint_banks_have_the_declared_witnesses(self) -> None:
        for n in TRAINING_SCALES:
            for length_band in LENGTH_BANDS:
                for depth in range(n - 1):
                    for gadget_name in feasible_gadget_names(n, length_band, depth):
                        gadget = source_gadgets(n, length_band)[gadget_name]
                        rotations = event_counts(gadget)[ROTATE] % n
                        quad = endpoint_quad_bank(n, depth, rotations)[0]
                        transformed = [apply_word(state, gadget, n) for state in quad]
                        self.assertEqual(shortest_witness_depth(transformed[0], transformed[1], n), None)
                        self.assertEqual(shortest_witness_depth(transformed[2], transformed[3], n), depth)
                        if depth < n - 2:
                            self.assertEqual(quad[2].bit_count(), quad[3].bit_count())
                        else:
                            self.assertEqual(abs(quad[2].bit_count() - quad[3].bit_count()), 1)

    def test_production_cardinalities_are_frozen(self) -> None:
        self.assertEqual(TOTAL_EPISODES, 18_432)
        self.assertEqual(ORDINARY_CALLS_PER_EPISODE, 32)
        self.assertEqual(TOTAL_ORDINARY_CALLS, 589_824)
        self.assertEqual(TOTAL_EPISODES % FROZEN_BATCH_SIZE, 0)
        self.assertEqual(len(FROZEN_GENERATION_CONTRACT_SHA256), 64)
        self.assertEqual(len(FROZEN_TRANSCRIPT_SHA256), 64)
        self.assertEqual(len(FROZEN_ORDINARY_CALL_LEDGER_SHA256), 64)
        self.assertEqual(len(FROZEN_REPORT_SHA256), 64)


class EpisodeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes, cls.ledger, cls.prf_report = generate_preview(episodes_per_cell=12)

    def test_preview_hashes_freeze_order_and_content(self) -> None:
        self.assertEqual(len(self.episodes), 72)
        self.assertEqual(len(self.ledger), 2_304)
        self.assertEqual(
            records_sha256(self.episodes),
            "813697787b527ded1eb0a520e25e98ddf88341c5b77ba3e5da9b352a03fa638a",
        )
        self.assertEqual(
            records_sha256(self.ledger),
            "3a30bb8c9818ba7c19157f51f136f0772bee320dd5c24edda4065e386d3618ee",
        )
        second_episodes, second_ledger, second_prf = generate_preview(episodes_per_cell=12)
        self.assertEqual(self.episodes, second_episodes)
        self.assertEqual(self.ledger, second_ledger)
        self.assertEqual(self.prf_report, second_prf)

    def test_every_episode_has_four_histories_and_exactly_32_public_calls(self) -> None:
        for episode in self.episodes:
            n = episode["cell"]["n"]
            self.assertEqual(len(episode["histories"]), HISTORIES_PER_EPISODE)
            self.assertEqual(episode["probe_rotations"], list(probe_rotation_bank(n)))
            self.assertEqual(
                episode["oracle_call_span"]["ordinary_one_bit_read_calls"],
                ORDINARY_CALLS_PER_EPISODE,
            )
            history_hashes = {history["history_sha256"] for history in episode["histories"]}
            self.assertEqual(len(history_hashes), HISTORIES_PER_EPISODE)
            for history in episode["histories"]:
                self.assertEqual(len(history["probes"]), PROBES_PER_HISTORY)
                self.assertEqual(
                    history["canonical_edge_bits_from_public_answers"],
                    [history["probes"][index]["answer"] for index in range(n - 1)],
                )

    def test_labels_and_masks_are_functions_of_answers(self) -> None:
        for episode in self.episodes:
            n = episode["cell"]["n"]
            signatures = [
                tuple(history["probes"][index]["answer"] for index in range(n - 1))
                for history in episode["histories"]
            ]
            expected_matrix = [
                [int(left == right) for right in signatures]
                for left in signatures
            ]
            self.assertEqual(episode["equivalence_label_matrix"], expected_matrix)
            self.assertEqual(expected_matrix[0][1], 1)
            self.assertEqual(expected_matrix[2][3], 0)
            differing = [
                index
                for index in range(PROBES_PER_HISTORY)
                if episode["histories"][2]["probes"][index]["answer"]
                != episode["histories"][3]["probes"][index]["answer"]
            ]
            first = differing[0]
            mask = episode["first_distinguishing_witness_mask"]
            self.assertEqual(sum(mask), 1)
            self.assertEqual(mask[first], 1)
            self.assertEqual(
                episode["probe_rotations"][first],
                episode["pairs"]["non_equivalent"]["shortest_witness_depth"],
            )
            self.assertEqual(sum(episode["uniform_probe_mask"]), 1)
            self.assertEqual(episode["uniform_probe_mask"][episode["uniform_probe_index"]], 1)

    def test_call_ledger_is_contiguous_and_links_to_transcript(self) -> None:
        by_call = {call["call_id"]: call for call in self.ledger}
        self.assertEqual(set(by_call), set(range(len(self.ledger))))
        for episode in self.episodes:
            for history_index, history in enumerate(episode["histories"]):
                for probe in history["probes"]:
                    call = by_call[probe["oracle_call_id"]]
                    self.assertEqual(call["episode_id"], episode["episode_id"])
                    self.assertEqual(call["history_index"], history_index)
                    self.assertEqual(call["history_sha256"], history["history_sha256"])
                    self.assertEqual(call["probe_index"], probe["probe_index"])
                    self.assertEqual(call["continuation_rotations"], probe["continuation_rotations"])
                    self.assertEqual(call["answer"], probe["answer"])
                    self.assertEqual(call["call_kind"], "READ")
                    self.assertEqual(call["returned_bits"], 1)

    def test_balance_and_maximum_depth_parity_obstruction(self) -> None:
        seen_depths: dict[int, set[int]] = defaultdict(set)
        seen_obstruction = set()
        for episode in self.episodes:
            n = episode["cell"]["n"]
            depth = episode["pairs"]["non_equivalent"]["shortest_witness_depth"]
            seen_depths[n].add(depth)
            counts = [history["event_counts"] for history in episode["histories"]]
            lengths = [history["source_length"] for history in episode["histories"]]
            if depth == n - 2:
                seen_obstruction.add(n)
                self.assertTrue(episode["balance"]["maximum_depth_flip_parity_obstruction"])
                self.assertNotEqual(counts[2][FLIP] % 2, counts[3][FLIP] % 2)
                self.assertFalse(episode["balance"]["all_event_counts_matched"])
                self.assertFalse(episode["balance"]["all_source_lengths_matched"])
            else:
                self.assertTrue(episode["balance"]["all_event_counts_matched"])
                self.assertTrue(episode["balance"]["all_source_lengths_matched"])
                self.assertEqual(len({(count[ROTATE], count[FLIP]) for count in counts}), 1)
                self.assertEqual(len(set(lengths)), 1)
            ceiling = episode["cell"]["source_length_ceiling"]
            self.assertLessEqual(max(lengths), ceiling)
            if episode["cell"]["length_band"] == "le_8n":
                self.assertGreater(max(lengths), 2 * n)
        self.assertEqual(seen_obstruction, set(TRAINING_SCALES))
        self.assertEqual(seen_depths, {n: set(range(n - 1)) for n in TRAINING_SCALES})

    def test_no_hidden_physical_state_ids_are_serialized(self) -> None:
        forbidden = {"physical_state", "endpoint_state", "residual_class_id", "hidden_state_id"}

        def walk(value: object) -> None:
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        for episode in self.episodes:
            walk(episode)


class ArtifactImmutabilityTests(unittest.TestCase):
    def test_preflight_refuses_existing_or_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (root / "transcript.jsonl", root / "ledger.jsonl", root / "report.json")
            self.assertEqual(preflight_immutable_outputs(paths), paths)
            paths[0].write_text("occupied\n")
            with self.assertRaises(FileExistsError):
                preflight_immutable_outputs(paths)
            paths[0].unlink()
            Path(str(paths[1]) + ".partial").write_text("partial\n")
            with self.assertRaises(FileExistsError):
                preflight_immutable_outputs(paths)

    def test_preflight_requires_distinct_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.jsonl"
            with self.assertRaises(ValueError):
                preflight_immutable_outputs((path, path, Path(directory) / "report.json"))

    def test_report_contract_has_no_runtime_or_path_fields(self) -> None:
        symbolic_stub = {
            "schema": "dwepr_stage_a_symbolic_gates_v1",
            "scales": [{"n": 3}, {"n": 6}],
        }
        contract = generation_contract(symbolic_stub)
        rendered = canonical_json_bytes(contract)
        self.assertNotIn(b"generated_at", rendered)
        self.assertNotIn(b"transcript_out", rendered)
        self.assertEqual(contract["total_episodes"], 18_432)
        self.assertEqual(contract["total_ordinary_one_bit_read_calls"], 589_824)


if __name__ == "__main__":
    unittest.main()
