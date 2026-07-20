from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS
from s9_occurrence_quotient_compiler import S9Example, SpanCandidate
from train_s9_2_global_anchor import (
    ARM_SPECS,
    BATCH_SIZE,
    CHARGED_VIEWS,
    CHECKPOINT_SCHEMA,
    COMPLETE_SYSTEM_PARAMETERS,
    NEGATIVE_CANDIDATES,
    ORBIT_WEIGHT,
    PAIRED_SOURCE_ROWS,
    UPDATES,
    PairedOrbitSelection,
    checkpoint_arm_payload,
    mask_gold_span_tokens,
    paired_consistent_shuffle,
    paired_source_indices,
    select_orbit_loss,
    split_paired_orbit_selection,
    verify_runtime_source,
)


def example(ids, gold) -> S9Example:
    return S9Example(
        ids=tuple(ids),
        offsets=tuple((index, index + 1) for index in range(len(ids))),
        gold=tuple(gold),
        row={},
    )


def candidate(position: int, target: int = 0) -> SpanCandidate:
    return SpanCandidate(
        start=position,
        end=position,
        text=str(position),
        char_start=position,
        char_end=position + 1,
        target=target,
    )


class S92TrainerCustodyTest(unittest.TestCase):
    def test_frozen_budget_and_checkpoint_contract(self):
        self.assertEqual(PAIRED_SOURCE_ROWS, 24_000)
        self.assertEqual(CHARGED_VIEWS, 48_000)
        self.assertEqual(BATCH_SIZE, 64)
        self.assertEqual(UPDATES, 750)
        self.assertEqual(NEGATIVE_CANDIDATES, 128)
        self.assertEqual(ORBIT_WEIGHT, 0.25)
        self.assertEqual(COMPLETE_SYSTEM_PARAMETERS, 134_580_264)
        self.assertEqual(CHECKPOINT_SCHEMA, "r12_s9_2_global_anchor_checkpoint_v1")

    def test_five_frozen_arm_specs(self):
        values = {
            spec.name: (
                spec.orbit_mode,
                spec.class_messages,
                spec.mask_gold_tokens,
            )
            for spec in ARM_SPECS
        }
        self.assertEqual(
            values,
            {
                "treatment": ("full", True, False),
                "positive_orbit_only": ("positive", True, False),
                "no_class": ("full", False, False),
                "shuffled": ("full", True, False),
                "layout_only": ("full", False, True),
            },
        )

    def test_evaluator_facing_checkpoint_arm_schema(self):
        arm_results = {
            spec.name: {
                "fit": {"name": spec.name},
                "adapter_state": {"weight": spec.name},
            }
            for spec in ARM_SPECS
        }
        payload = checkpoint_arm_payload(arm_results)
        self.assertEqual(
            set(payload["fit"]),
            {"treatment", "positive_only", "no_class", "shuffled", "layout"},
        )
        for name in payload["fit"]:
            self.assertIn(f"{name}_adapter_state", payload)
        incomplete = dict(arm_results)
        incomplete.pop("layout_only")
        with self.assertRaises(ValueError):
            checkpoint_arm_payload(incomplete)

    def test_paired_source_indices_are_unique_deterministic_and_bounded(self):
        first = paired_source_indices(20, 41, unique_sources=12)
        second = paired_source_indices(20, 41, unique_sources=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertTrue(all(0 <= index < 20 for index in first))
        with self.assertRaises(ValueError):
            paired_source_indices(11, 41, unique_sources=12)

    def test_paired_shuffle_reuses_one_per_source_permutation(self):
        labels = (
            ROLE_INDEX["entity.roster"],
            ROLE_INDEX["card.operation"],
            ROLE_INDEX["event.tag"],
            ROLE_INDEX["query.position"],
        )
        first = example(range(6), [(i, i, label) for i, label in enumerate(labels)])
        second = example(
            range(7), [(i + 1, i + 1, label) for i, label in enumerate(labels)]
        )
        shuffled_first, shuffled_second = paired_consistent_shuffle(
            [first], [second], 99
        )
        self.assertEqual(
            tuple(target for _, _, target in shuffled_first[0].gold),
            tuple(target for _, _, target in shuffled_second[0].gold),
        )
        self.assertCountEqual(
            [target for _, _, target in shuffled_first[0].gold], labels
        )

    def test_layout_mask_is_immutable_and_masks_only_gold_positions(self):
        first = example(
            [10, 11, 12, 13, 14, 15],
            [(1, 2, 1), (4, 4, 2)],
        )
        second = example([20, 21, 22, 23], [(0, 0, 1), (2, 3, 2)])
        ids = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 0, 0],
            ]
        )
        original = ids.clone()
        masked = mask_gold_span_tokens(ids, [first, second])
        self.assertTrue(torch.equal(ids, original))
        self.assertEqual(
            masked.tolist(),
            [
                [10, 0, 0, 13, 0, 15],
                [0, 21, 0, 0, 0, 0],
            ],
        )
        with self.assertRaises(ValueError):
            mask_gold_span_tokens(ids, [first, second], mask_token_id=7)

    def test_runtime_source_is_bound_to_committed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(
                ("git", "config", "user.email", "test@example.com"),
                cwd=root,
                check=True,
            )
            subprocess.run(
                ("git", "config", "user.name", "S9.2 Test"),
                cwd=root,
                check=True,
            )
            source = root / "scientific.py"
            source.write_text("VALUE = 1\n")
            subprocess.run(("git", "add", "scientific.py"), cwd=root, check=True)
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
            self.assertEqual(
                verify_runtime_source(root, commit, ("scientific.py",)),
                commit,
            )
            source.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(RuntimeError, "runtime bytes differ"):
                verify_runtime_source(root, commit, ("scientific.py",))

    def test_pair_split_handles_different_candidate_counts(self):
        first = example([1, 2], [(0, 0, 1)])
        second = example([1, 2, 3], [(1, 1, 1)])
        first_candidates = (candidate(0, 1), candidate(1))
        second_candidates = (
            candidate(0),
            candidate(1, 1),
            candidate(2),
        )
        logits = torch.arange(5 * len(ROLE_LABELS), dtype=torch.float32).reshape(
            5, len(ROLE_LABELS)
        )
        selection = split_paired_orbit_selection(
            [first, second],
            [first_candidates, second_candidates],
            logits,
            1,
        )
        self.assertEqual(selection.original_logits.shape[0], 2)
        self.assertEqual(selection.recoded_logits.shape[0], 3)
        self.assertTrue(torch.equal(selection.original_logits, logits[:2]))
        self.assertTrue(torch.equal(selection.recoded_logits, logits[2:]))


class S92OrbitSelectionTest(unittest.TestCase):
    def setUp(self):
        self.gold = (
            (0, 0, ROLE_INDEX["entity.roster"]),
            (1, 1, ROLE_INDEX["card.operation"]),
        )
        self.first = example(range(14), self.gold)
        self.second = example(range(14), self.gold)
        self.candidates = tuple(
            candidate(index, self.gold[index][2] if index < 2 else 0)
            for index in range(14)
        )
        logits = torch.zeros((14, len(ROLE_LABELS)), dtype=torch.float32)
        logits[0, ROLE_INDEX["entity.roster"]] = 4.0
        logits[1, ROLE_INDEX["card.operation"]] = 4.0
        for index in range(2, 14):
            logits[index, 1:] = (
                torch.linspace(-2.0, 1.0, len(ROLE_LABELS) - 1) + index / 100.0
            )
        self.logits = logits

    def selection(self, first_logits, second_logits) -> PairedOrbitSelection:
        return PairedOrbitSelection(
            original_examples=(self.first,),
            recoded_examples=(self.second,),
            original_candidate_rows=(self.candidates,),
            recoded_candidate_rows=(self.candidates,),
            original_logits=first_logits,
            recoded_logits=second_logits,
        )

    def test_positive_mode_is_exact_s9_1_loss(self):
        second = self.logits.clone()
        second[2, ROLE_INDEX["entity.roster"]] = 20.0
        loss = select_orbit_loss(
            self.selection(self.logits, second), "positive", top_k=8
        )
        self.assertEqual(float(loss.item()), 0.0)

    def test_full_mode_detects_hard_negative_change_with_finite_gradients(self):
        first = self.logits.clone().requires_grad_()
        second = self.logits.clone()
        second[2, ROLE_INDEX["entity.roster"]] = 20.0
        second.requires_grad_()
        loss = select_orbit_loss(self.selection(first, second), "full", top_k=8)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_unknown_orbit_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            select_orbit_loss(
                self.selection(self.logits, self.logits.clone()),
                "unknown",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
