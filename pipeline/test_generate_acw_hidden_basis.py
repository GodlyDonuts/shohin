import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.generate_acw_hidden_basis import (
    CONFIRMATION_COMMITMENTS,
    CONFIRMATION_PROTOCOL_STATUS,
    Domain,
    apply_event,
    build_domain,
    confirmation_commitment,
    determinant_mod17,
    development_seed_material,
    file_sha256,
    generate_dataset,
    generate_histories,
    query_answers,
    render_source,
    split_name,
    state_bucket,
    validate_seed_identity,
)


class HiddenBasisGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.seed = development_seed_material(2026071601)
        self.domain = build_domain(self.seed)

    def test_domain_contract(self):
        self.assertNotEqual(determinant_mod17(self.domain.basis), 0)
        self.assertEqual(self.domain.event_features.shape, (48, 96))
        self.assertEqual(self.domain.events.shape, (48, 5))
        self.assertEqual(
            [int((self.domain.events[:, 0] == address).sum()) for address in range(3)],
            [16, 16, 16],
        )
        self.assertNotEqual(determinant_mod17(self.domain.query_coefficients[:3]), 0)
        public = {
            tuple(int(value) for value in row) for row in self.domain.query_coefficients
        }
        new = {
            tuple(int(value) for value in row)
            for row in self.domain.new_query_coefficients
        }
        self.assertFalse(public & new)

    def test_source_rendering_hides_basis_but_is_deterministic(self):
        state = np.asarray([1, 2, 3], dtype=np.int8)
        first = render_source(self.domain, state)
        second = render_source(self.domain, state)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (96,))
        self.assertEqual(first.dtype, np.float32)

    def test_float_features_have_cross_runtime_golden_hashes(self):
        self.assertEqual(
            hashlib.sha256(self.domain.event_features.tobytes()).hexdigest(),
            "74a5816300a6855cdea73ea9dd990b2b18ffe8a2f887cd568e19e12da0430900",
        )
        all_sources = np.stack(
            [
                render_source(self.domain, np.asarray(state, dtype=np.int8))
                for state in itertools.product(range(17), repeat=3)
            ]
        ).astype(np.float32)
        self.assertEqual(
            hashlib.sha256(all_sources.tobytes()).hexdigest(),
            "70acbc51b12d61bcaf6c654843e86efdfc39c39b931f986c548677ada4440991",
        )

    def test_event_and_query_semantics(self):
        state = np.asarray([4, 5, 6], dtype=np.int8)
        event = np.asarray([1, 2, 3, 4, 5], dtype=np.int8)
        observed = apply_event(state, event)
        self.assertEqual(int(observed[1]), (3 * 5 + 4 * 6 + 5) % 17)
        self.assertEqual(int(observed[0]), 4)
        self.assertEqual(int(observed[2]), 6)
        answers = query_answers(
            state,
            self.domain.query_coefficients,
            self.domain.query_offsets,
            self.domain.query_permutations,
        )
        self.assertEqual(answers.shape, (24,))
        self.assertTrue(np.all((answers >= 0) & (answers < 17)))

    def test_history_endpoints_obey_split_and_visits_are_counted(self):
        histories = generate_histories(
            self.domain,
            self.seed,
            count=24,
            target_split="evaluation",
            depths=(8,),
            label="unit-history",
        )
        self.assertEqual(histories.event_ids.shape, (24, 8))
        self.assertEqual(histories.trajectory_states.shape, (24, 9, 3))
        self.assertEqual(histories.public_answers.shape, (24, 24))
        for index, length in enumerate(histories.lengths):
            self.assertTrue(
                np.array_equal(
                    histories.trajectory_states[index, 0],
                    histories.source_states[index],
                )
            )
            self.assertTrue(
                np.array_equal(
                    histories.trajectory_states[index, int(length)],
                    histories.final_states[index],
                )
            )
        for state in histories.final_states:
            self.assertEqual(split_name(state_bucket(self.seed, state)), "evaluation")
        self.assertEqual(sum(histories.visited_buckets.values()), 24 * 9)

    def test_training_depths_use_balanced_accepted_quotas(self):
        histories = generate_histories(
            self.domain,
            self.seed,
            count=101,
            target_split="train",
            depths=range(9),
            label="balanced-depth-unit",
        )
        counts = list(histories.depth_counts.values())
        self.assertEqual(sum(counts), 101)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_small_dataset_is_deterministic_and_excludes_seed_preimage(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            kwargs = dict(
                seed_identity={"kind": "development", "seed": 2026071601},
                train_count=16,
                adaptation_count=8,
                evaluation_count=8,
                evaluation_depths=(8, 16),
            )
            first_manifest = generate_dataset(first, self.seed, **kwargs)
            second_manifest = generate_dataset(second, self.seed, **kwargs)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                file_sha256(first / "manifest.json"),
                file_sha256(second / "manifest.json"),
            )
            self.assertNotIn(self.seed.hex(), (first / "manifest.json").read_text())
            self.assertEqual(
                first_manifest["event_address_counts"], {"0": 16, "1": 16, "2": 16}
            )

    def test_confirmation_commitment_contract(self):
        seed = bytes(range(32))
        observed = confirmation_commitment(seed)
        self.assertEqual(len(observed), 64)
        self.assertEqual(len(CONFIRMATION_COMMITMENTS), 3)
        with self.assertRaises(ValueError):
            confirmation_commitment(b"too short")

    def test_confirmation_generation_is_disabled_until_future_beacon_opening(self):
        seed = bytes(range(32))
        identity = {
            "kind": "confirmation",
            "index": 0,
            "commitment": confirmation_commitment(seed),
        }
        with self.assertRaisesRegex(RuntimeError, CONFIRMATION_PROTOCOL_STATUS):
            validate_seed_identity(seed, identity)

    def test_seed_identity_must_match_seed_material(self):
        validate_seed_identity(
            self.seed,
            {"kind": "development", "seed": 2026071601},
        )
        with self.assertRaises(ValueError):
            validate_seed_identity(
                self.seed,
                {"kind": "development", "seed": 2026071602},
            )
        unregistered = development_seed_material(123)
        with self.assertRaises(ValueError):
            validate_seed_identity(
                unregistered,
                {"kind": "development", "seed": 123},
            )

    def test_state_bucket_matches_frozen_seed_plus_state_contract(self):
        state = np.asarray([3, 4, 5], dtype=np.int8)
        expected = (
            int.from_bytes(
                hashlib.sha256(self.seed + bytes([3, 4, 5])).digest()[:8],
                "big",
            )
            % 100
        )
        self.assertEqual(state_bucket(self.seed, state), expected)

    def test_domain_dataclass_is_not_public_seed_storage(self):
        self.assertIsInstance(self.domain, Domain)
        self.assertEqual(len(self.domain.seed_fingerprint), 64)
        self.assertFalse(hasattr(self.domain, "seed_material"))


if __name__ == "__main__":
    unittest.main()
