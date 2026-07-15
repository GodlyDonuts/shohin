#!/usr/bin/env python3
"""Focused CPU contracts for the five frozen WGRQ training arms."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "train", ROOT / "pipeline"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import generate_wgrq_falsifier_v1 as generator  # noqa: E402
from train_wgrq_cpu import (  # noqa: E402
    ACTIVE_ANSWER_ONLY,
    ARM_COEFFICIENTS,
    ARM_NAMES,
    BATCHES_PER_EPOCH,
    BATCH_SIZE,
    BETAS,
    EPOCHS,
    EPSILON,
    FROZEN_SEEDS,
    GENERATOR_REPORT_SCHEMA,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MATRIX_WEIGHT_DECAY,
    PRIVILEGED_EDGE,
    TOTAL_UPDATES,
    TRAIN_ANSWER_CALLS,
    TRAIN_EPISODES,
    UNIFORM_WITNESS,
    WARMUP_UPDATES,
    WGRQ_SHORTEST,
    WGRQBatch,
    WGRQForwardPass,
    _sham_permutation,
    bind_frozen_inputs,
    build_optimizer,
    compute_all_loss_terms,
    frozen_training_batches,
    hash_model_state,
    learning_rate_for_update,
    normalize_episode,
    run_training_update,
    save_frozen_checkpoint,
)
from wgrq_state_machine import HardBitDWEPRLearner, make_scale_mask  # noqa: E402


def producer_episode():
    prf = generator.FixedPRF()
    depth = 0
    gadget = generator.feasible_gadget_names(4, "le_2n", depth)[0]
    row, _ = generator.make_episode(
        n=4,
        length_band="le_2n",
        cell_episode_index=0,
        global_episode_index=0,
        depth=depth,
        gadget_name=gadget,
        prf=prf,
    )
    return row


def synthetic_batch(batch_size: int = 2) -> WGRQBatch:
    source_events = torch.tensor(
        [[[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]]] * batch_size,
        dtype=torch.long,
    )
    source_mask = torch.ones_like(source_events, dtype=torch.bool)
    probe_events = torch.zeros(batch_size, 8, 3, dtype=torch.long)
    probe_mask = torch.zeros_like(probe_events, dtype=torch.bool)
    for probe in range(8):
        probe_mask[:, probe, : probe % 4] = True
    answers = torch.tensor(
        [[
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1],
        ]] * batch_size,
        dtype=torch.float32,
    )
    scale_mask = make_scale_mask(4).repeat(batch_size, 1)
    canonical = torch.zeros(batch_size, 4, 15)
    witness = torch.zeros(batch_size, 8)
    witness[:, 0] = 1
    uniform = torch.arange(batch_size, dtype=torch.long) % 8
    if batch_size == 2:
        sham = torch.tensor([[1, 1], [0, 0]], dtype=torch.long)
    else:
        permutation = torch.arange(batch_size).roll(-1)
        sham = torch.stack((permutation, permutation), dim=-1)
    return WGRQBatch(
        source_events=source_events,
        source_mask=source_mask,
        probe_events=probe_events,
        probe_mask=probe_mask,
        answers=answers,
        scale_mask=scale_mask,
        canonical_codes=canonical,
        shortest_witness_mask=witness,
        uniform_probe_index=uniform,
        sham_permutation=sham,
    )


class ProducerContractTests(unittest.TestCase):
    def test_rich_producer_episode_is_consumed_without_hidden_targets(self):
        row = producer_episode()
        episode = normalize_episode(row, 0)
        self.assertEqual(episode.uniform_probe_index, row["uniform_probe_index"])
        self.assertEqual(episode.batch_index, 0)
        self.assertEqual(episode.batch_offset, 0)
        for history, public in zip(episode.histories, row["histories"]):
            self.assertEqual(
                list(history.canonical_code[: episode.n - 1]),
                public["canonical_edge_bits_from_public_answers"],
            )
        tampered = json.loads(json.dumps(row))
        tampered["uniform_probe_mask"].reverse()
        with self.assertRaisesRegex(ValueError, "uniform"):
            normalize_episode(tampered, 0)

    def test_sham_permutation_is_deterministic_batch_local_and_has_no_fixed_points(self):
        base = normalize_episode(producer_episode(), 0)
        episodes = []
        for index in range(BATCH_SIZE * 2):
            digest = hashlib.sha256(str(index).encode("ascii")).hexdigest()
            episodes.append(dataclasses.replace(
                base,
                episode_id=f"episode-{index}",
                record_sha256=digest,
                batch_index=index // BATCH_SIZE,
                batch_offset=index % BATCH_SIZE,
            ))
        first = _sham_permutation(episodes, 1, b"test-domain")
        second = _sham_permutation(episodes, 1, b"test-domain")
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(len(episodes))))
        for index, partner in enumerate(first):
            self.assertNotEqual(index, partner)
            self.assertEqual(episodes[index].batch_index, episodes[partner].batch_index)


class FrozenScheduleTests(unittest.TestCase):
    def test_all_frozen_counts_seeds_and_learning_rate_endpoints(self):
        self.assertEqual(BATCH_SIZE, 64)
        self.assertEqual(EPOCHS, 4)
        self.assertEqual(BATCHES_PER_EPOCH, 288)
        self.assertEqual(TOTAL_UPDATES, 1_152)
        self.assertEqual(WARMUP_UPDATES, 64)
        self.assertEqual(TRAIN_EPISODES, 18_432)
        self.assertEqual(TRAIN_ANSWER_CALLS, 589_824)
        self.assertEqual(len(FROZEN_SEEDS), 12)
        self.assertEqual(learning_rate_for_update(0), LEARNING_RATE / WARMUP_UPDATES)
        self.assertEqual(learning_rate_for_update(WARMUP_UPDATES - 1), LEARNING_RATE)
        self.assertEqual(learning_rate_for_update(TOTAL_UPDATES - 1), 0.0)
        self.assertLess(learning_rate_for_update(100), LEARNING_RATE)
        with self.assertRaises(ValueError):
            learning_rate_for_update(TOTAL_UPDATES)

    def test_seeded_order_shuffles_only_frozen_batch_blocks(self):
        first = frozen_training_batches(FROZEN_SEEDS[0])
        repeat = frozen_training_batches(FROZEN_SEEDS[0])
        other = frozen_training_batches(FROZEN_SEEDS[1])
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), TOTAL_UPDATES)
        for batch in first:
            self.assertEqual(len(batch), BATCH_SIZE)
            self.assertEqual(batch, tuple(range(batch[0], batch[0] + BATCH_SIZE)))
            self.assertEqual(batch[0] % BATCH_SIZE, 0)
        first_epoch = [index for batch in first[:BATCHES_PER_EPOCH] for index in batch]
        self.assertEqual(sorted(first_epoch), list(range(TRAIN_EPISODES)))


class OptimizerAndLossTests(unittest.TestCase):
    def test_adamw_contract_decays_matrices_only(self):
        model = HardBitDWEPRLearner()
        optimizer = build_optimizer(model)
        self.assertEqual(tuple(optimizer.defaults["betas"]), BETAS)
        self.assertEqual(optimizer.defaults["eps"], EPSILON)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [MATRIX_WEIGHT_DECAY, 0.0])
        matrix_ids = {id(parameter) for parameter in model.parameters() if parameter.ndim == 2}
        vector_ids = {id(parameter) for parameter in model.parameters() if parameter.ndim == 1}
        self.assertEqual({id(parameter) for parameter in optimizer.param_groups[0]["params"]}, matrix_ids)
        self.assertEqual({id(parameter) for parameter in optimizer.param_groups[1]["params"]}, vector_ids)

    def test_every_arm_builds_every_term_and_uses_only_its_frozen_coefficients(self):
        batch = synthetic_batch()
        logits = torch.linspace(-1.5, 1.5, 2 * 4 * 8).reshape(2, 4, 8).requires_grad_()
        edge_logits = torch.linspace(-0.8, 0.8, 2 * 4 * 15).reshape(2, 4, 15).requires_grad_()
        forward = WGRQForwardPass(
            probe_logits=logits,
            final_source_probabilities=torch.sigmoid(edge_logits),
            commitment=torch.tensor(0.2, requires_grad=True),
        )
        expected_keys = {
            "total",
            "answer",
            "equivalent_js",
            "separation_shortest",
            "separation_uniform",
            "relation_shortest",
            "relation_uniform",
            "sham_equivalent_js",
            "sham_separation",
            "relation_sham",
            "commitment",
            "privileged_edge",
        }
        for arm in ARM_NAMES:
            terms = compute_all_loss_terms(forward, batch, arm)
            self.assertEqual(set(terms), expected_keys)
            expected = sum(
                terms[name] * coefficient
                for name, coefficient in ARM_COEFFICIENTS[arm].items()
            )
            self.assertTrue(torch.allclose(terms["total"], expected))
            self.assertTrue(all(bool(torch.isfinite(value)) for value in terms.values()))

        answer_only = compute_all_loss_terms(forward, batch, ACTIVE_ANSWER_ONLY)
        edge_gradient = torch.autograd.grad(
            answer_only["total"], edge_logits, allow_unused=True, retain_graph=True
        )[0]
        self.assertIsNone(edge_gradient)
        privileged = compute_all_loss_terms(forward, batch, PRIVILEGED_EDGE)
        privileged_gradient = torch.autograd.grad(privileged["total"], edge_logits)[0]
        self.assertGreater(float(privileged_gradient.abs().sum()), 0.0)
        self.assertEqual(WGRQ_SHORTEST, "WGRQ-shortest")
        self.assertIn(UNIFORM_WITNESS, ARM_NAMES)

    def test_one_real_update_uses_schedule_clipping_and_changes_weights(self):
        torch.manual_seed(FROZEN_SEEDS[0])
        model = HardBitDWEPRLearner()
        optimizer = build_optimizer(model)
        before = hash_model_state(model)
        stats = run_training_update(model, optimizer, synthetic_batch(), WGRQ_SHORTEST, 0)
        self.assertNotEqual(hash_model_state(model), before)
        self.assertEqual(stats["learning_rate"], LEARNING_RATE / WARMUP_UPDATES)
        self.assertTrue(math.isfinite(stats["total"]))
        self.assertTrue(math.isfinite(stats["gradient_norm"]))
        self.assertEqual(GRADIENT_CLIP, 1.0)


class HashAndCheckpointTests(unittest.TestCase):
    def test_input_and_report_hashes_are_bound_and_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = root / "train.jsonl"
            report_path = root / "report.json"
            transcript.write_bytes(b"{}\n")
            transcript_hash = hashlib.sha256(transcript.read_bytes()).hexdigest()
            report = {
                "schema": GENERATOR_REPORT_SCHEMA,
                "passed": True,
                "artifacts": {
                    "transcript": {"sha256": transcript_hash, "rows": TRAIN_EPISODES},
                },
                "totals": {"ordinary_one_bit_read_calls": TRAIN_ANSWER_CALLS},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            binding = bind_frozen_inputs(transcript, report_path)
            self.assertEqual(binding.transcript_sha256, transcript_hash)
            transcript.write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "transcript changed"):
                binding.assert_unchanged()
            transcript.write_bytes(b"{}\n")
            binding = bind_frozen_inputs(transcript, report_path)
            report["passed"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "audit report changed"):
                binding.assert_unchanged()

    def test_checkpoint_contains_fixed_weights_and_hash_metadata_without_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            model = HardBitDWEPRLearner()
            metadata = {
                "arm": WGRQ_SHORTEST,
                "seed": FROZEN_SEEDS[0],
                "packet_bits": 15,
                "transcript_sha256": "a" * 64,
                "audit_report_sha256": "b" * 64,
            }
            digest = save_frozen_checkpoint(output, model, metadata)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            try:
                checkpoint = torch.load(output, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(output, map_location="cpu")
            self.assertEqual(set(checkpoint), {"model_state", "wgrq_cpu"})
            self.assertEqual(checkpoint["wgrq_cpu"], metadata)
            self.assertNotIn(str(Path(directory)), json.dumps(checkpoint["wgrq_cpu"], sort_keys=True))
            restored = HardBitDWEPRLearner()
            restored.load_state_dict(checkpoint["model_state"], strict=True)
            self.assertEqual(hash_model_state(restored), hash_model_state(model))


if __name__ == "__main__":
    unittest.main()
