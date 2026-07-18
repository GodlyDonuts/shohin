#!/usr/bin/env python3
"""CPU contracts for the grammar-gated causal result-digit motor.

Runnable as ``python -m unittest train.test_causal_result_digit_motor`` from
the repository root, or as ``python -m unittest test_causal_result_digit_motor``
from inside ``train/``.
"""

from __future__ import annotations

import collections
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import torch  # noqa: E402
import causal_result_digit_motor as digit_motor  # noqa: E402

from causal_result_digit_motor import (  # noqa: E402
    BASE_PARAM_COUNT,
    DigitMotor,
    DigitRouter,
    DIGIT_COUNT,
    EXPECTED_MOTOR_SOURCE_MANIFEST_SHA256,
    MAX_TOTAL_PARAMS,
    MOTOR_HIDDEN,
    MOTOR_MID,
    apply_motor_logits,
    full_vocab_motor_loss,
    is_digit_site,
    parameter_budget,
    permuted_control_labels,
    prepare_single_output,
    tensor_state_sha256,
    unique_model_parameter_count,
    validate_motor_bundle,
)
from digitwise_protocol import apply_microstep, initial_state, microstep_prompt  # noqa: E402
from probe_digitwise_workspace import field_prefix  # noqa: E402


class DigitMotorParameterCountTest(unittest.TestCase):
    def test_parameter_count_matches_budget(self):
        motor = DigitMotor(576, hidden=8, mid=8)
        # 576*8+8 + 8*8+8 + 8*10+10 = 4616+72+90 = 4778
        self.assertEqual(motor.parameter_count(), 4778)
        wide = DigitMotor(576, hidden=MOTOR_HIDDEN, mid=MOTOR_MID)
        self.assertLess(BASE_PARAM_COUNT + wide.parameter_count(), MAX_TOTAL_PARAMS)
        self.assertGreater(wide.parameter_count(), 19_000_000)
        self.assertEqual(motor.up.out_features, DIGIT_COUNT)

    def test_exact_150m_ceiling_is_rejected(self):
        motor_count = DigitMotor(576, hidden=8, mid=8).parameter_count()
        with mock.patch.object(
            digit_motor, "MAX_TOTAL_PARAMS", BASE_PARAM_COUNT + motor_count
        ):
            with self.assertRaisesRegex(ValueError, "parameter budget exceeded"):
                DigitMotor(576, hidden=8, mid=8)

    def test_budget_reports_exact_deployed_total(self):
        wide = DigitMotor(576, hidden=MOTOR_HIDDEN, mid=MOTOR_MID)
        budget = parameter_budget(wide.parameter_count())
        self.assertEqual(budget["base_parameters"], 125_081_664)
        self.assertEqual(budget["motor_parameters"], 19_185_674)
        self.assertEqual(budget["total_parameters"], 144_267_338)
        self.assertEqual(budget["strict_cap"], 150_000_000)
        self.assertEqual(budget["remaining_addable_parameters"], 5_732_661)

    def test_unique_count_handles_tied_parameters_once(self):
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(7, 3)
                self.head = torch.nn.Linear(3, 7, bias=False)
                self.head.weight = self.embedding.weight

        self.assertEqual(unique_model_parameter_count(Tied()), 21)


class OutputCustodyTest(unittest.TestCase):
    def test_eval_output_can_share_directory_with_training_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("training report\n")
            output = prepare_single_output(root / "eval.json")
            self.assertEqual(output, root / "eval.json")
            output.write_text("evaluation\n")
            with self.assertRaisesRegex(FileExistsError, "refusing existing output"):
                prepare_single_output(output)


class EvaluatorPatchCustodyTest(unittest.TestCase):
    @staticmethod
    def _bundle(source_contract):
        state = {"weight": torch.tensor([1.0])}
        state_hash = tensor_state_sha256(state)
        return {
            "audit": "causal_result_digit_motor_fit_v1",
            "deployment_logit_dtype": "torch.bfloat16",
            "source_contract": source_contract,
            "digit_ids": list(range(DIGIT_COUNT)),
            "treatment": state,
            "shuffled": state,
            "treatment_state_sha256": state_hash,
            "shuffled_state_sha256": state_hash,
        }

    def test_only_exact_bound_training_manifest_can_cross_evaluator_patch(self):
        current = {"git_commit": None, "manifest_sha256": "current"}
        training = {
            "git_commit": None,
            "manifest_sha256": EXPECTED_MOTOR_SOURCE_MANIFEST_SHA256,
        }
        bundle = self._bundle(training)
        with self.assertRaisesRegex(ValueError, "source contract mismatch"):
            validate_motor_bundle(bundle, {}, current)
        self.assertEqual(
            validate_motor_bundle(bundle, {}, current, allow_evaluator_patch=True),
            training,
        )
        forged = self._bundle(
            {"git_commit": None, "manifest_sha256": "forged-training-source"}
        )
        with self.assertRaisesRegex(ValueError, "source contract mismatch"):
            validate_motor_bundle(forged, {}, current, allow_evaluator_patch=True)


class SiteDetectionTest(unittest.TestCase):
    def _synthetic_prefix(self):
        state = initial_state("add", 9999, 1, 4)
        prompt = microstep_prompt(state, style="core")
        next_state = apply_microstep(state)
        prefix, target = field_prefix(prompt, next_state, "digit")
        return state, prompt, prefix[len(prompt) :], target

    def test_site_detected_on_synthetic_canonical_prefix(self):
        _state, prompt, response_prefix, target = self._synthetic_prefix()
        self.assertEqual(response_prefix, "dws:op=add;w=4;p=1;c=1;a=9999;b=1000;r=")
        self.assertEqual(target, "0")
        self.assertTrue(is_digit_site(prompt, response_prefix))

    def test_site_rejects_wrong_context_op_position_and_extra_digit(self):
        _state, prompt, response_prefix, _target = self._synthetic_prefix()
        self.assertFalse(is_digit_site("Question: add\nAnswer:", response_prefix))
        self.assertFalse(
            is_digit_site(prompt, "dws:op=sub;w=4;p=1;c=1;a=9999;b=1000;r=")
        )
        self.assertFalse(
            is_digit_site(prompt, "dws:op=add;w=4;p=999;c=1;a=9999;b=1000;r=")
        )
        self.assertFalse(
            is_digit_site(prompt, "dws:op=add;w=4;p=1;c=1;a=9999;b=1000;r=0")
        )
        self.assertFalse(
            is_digit_site(prompt, "dws:op=add;w=4;p=1;c=1;a=0000;b=1000;r=")
        )
        self.assertFalse(is_digit_site(prompt, response_prefix + "\n"))


class ApplyMotorLogitsTest(unittest.TestCase):
    def test_gate_off_is_bit_exact_and_gate_on_touches_only_ten_digit_ids(self):
        torch.manual_seed(3)
        motor = DigitMotor(6, hidden=2)
        with torch.no_grad():
            motor.up.weight.fill_(0.25)
            motor.up.bias.copy_(torch.arange(DIGIT_COUNT, dtype=torch.float32))
        digit_ids = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        logits = torch.randn(3, 31)
        hidden = torch.randn(3, 6)

        off = apply_motor_logits(logits, hidden, motor, digit_ids, False)
        self.assertTrue(torch.equal(off, logits))

        off_absent_motor = apply_motor_logits(logits, hidden, None, digit_ids, True)
        self.assertTrue(torch.equal(off_absent_motor, logits))

        on = apply_motor_logits(logits, hidden, motor, digit_ids, True)
        untouched = [i for i in range(logits.shape[-1]) if i not in digit_ids]
        self.assertTrue(torch.equal(on[:, untouched], logits[:, untouched]))
        self.assertFalse(torch.equal(on[:, digit_ids], logits[:, digit_ids]))

        dead = DigitMotor(6, hidden=2)
        dead_on = apply_motor_logits(logits, hidden, dead, digit_ids, True)
        self.assertTrue(torch.equal(dead_on, logits))

    def test_invalid_digit_id_lists_raise(self):
        motor = DigitMotor(4, hidden=2)
        logits = torch.randn(2, 40)
        hidden = torch.randn(2, 4)
        with self.assertRaises(ValueError):
            apply_motor_logits(logits, hidden, motor, list(range(9)), True)
        with self.assertRaises(ValueError):
            apply_motor_logits(logits, hidden, motor, [0] * DIGIT_COUNT, True)
        with self.assertRaises(ValueError):
            apply_motor_logits(logits, hidden, motor, list(range(-1, 9)), True)


class FullVocabMotorLossTest(unittest.TestCase):
    def test_full_vocab_loss_matches_dense_cross_entropy(self):
        torch.manual_seed(4)
        motor = DigitMotor(5, hidden=3)
        hidden = torch.randn(9, 5)
        logits = torch.randn(9, 41)
        targets = torch.randint(0, DIGIT_COUNT, (9,))
        digit_ids = [1, 4, 6, 9, 12, 15, 20, 25, 30, 37]
        keep = torch.ones(logits.shape[-1], dtype=torch.bool)
        keep[digit_ids] = False
        compact = full_vocab_motor_loss(
            hidden,
            logits[:, digit_ids],
            torch.logsumexp(logits[:, keep], dim=-1),
            targets,
            motor,
        )
        dense = apply_motor_logits(logits, hidden, motor, digit_ids, True)
        dense_targets = torch.as_tensor([digit_ids[int(t)] for t in targets])
        expected = torch.nn.functional.cross_entropy(dense, dense_targets)
        torch.testing.assert_close(compact, expected)


class DigitRouterTest(unittest.TestCase):
    def test_router_fires_once_and_reports_missed_site(self):
        state = initial_state("add", 9999, 1, 4)
        prompt = microstep_prompt(state, style="core")
        site = "dws:op=add;w=4;p=1;c=1;a=9999;b=1000;r="
        router = DigitRouter(prompt, motor_present=True)
        self.assertFalse(router.observe(""))
        self.assertTrue(router.observe(site))
        self.assertFalse(router.observe(site))
        self.assertEqual(router.site_count, 1)
        self.assertEqual(router.fire_count, 1)

        missed = DigitRouter(prompt, motor_present=True)
        self.assertFalse(missed.observe(site + "0"))
        self.assertEqual(missed.site_count, 0)
        self.assertEqual(missed.fire_count, 0)

        no_motor = DigitRouter(prompt, motor_present=False)
        self.assertFalse(no_motor.observe(site))
        self.assertEqual(no_motor.site_count, 1)
        self.assertEqual(no_motor.fire_count, 0)


class PermutedControlLabelsTest(unittest.TestCase):
    def test_shuffled_control_preserves_nuisance_counts(self):
        rows = []
        for operation in ("add", "sub"):
            for target in range(DIGIT_COUNT):
                for index in range(30):
                    rows.append(
                        {
                            "operation": operation,
                            "width": 4,
                            "position": 1,
                            "style": "core",
                            "current_carry": index % 2,
                            "target": target,
                        }
                    )
        labels, report = permuted_control_labels(rows, seed=77)
        self.assertGreaterEqual(report["changed"], len(rows) // 3)
        self.assertEqual(len(labels), len(rows))
        before, after = (
            collections.defaultdict(collections.Counter),
            collections.defaultdict(collections.Counter),
        )
        for row, label in zip(rows, labels):
            key = (
                row["operation"],
                row["width"],
                row["position"],
                row["style"],
                row["current_carry"],
            )
            before[key][row["target"]] += 1
            after[key][label] += 1
        self.assertEqual(before, after)


class RealBoardBalanceTest(unittest.TestCase):
    def test_real_board_is_balanced_over_operation_style_and_target(self):
        tokenizer_path = ROOT / "artifacts" / "shohin-tok-32k.json"
        episodes_path = (
            ROOT / "artifacts" / "evals" / "digitwise_recurrent_v2_heldout.jsonl"
        )
        if not tokenizer_path.exists() or not episodes_path.exists():
            self.skipTest("frozen tokenizer/episodes artifacts are not present locally")
        from tokenizers import Tokenizer

        from causal_result_digit_motor import generate_fit_rows

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        rows, report = generate_fit_rows(tokenizer, episodes_path.read_text(), quota=1)
        self.assertEqual(report["rows"], 40)
        counts = collections.Counter(
            (row["operation"], row["style"], row["target"]) for row in rows
        )
        self.assertEqual(set(counts.values()), {1})
        for row in rows:
            self.assertEqual(
                row["prefix_ids"][: len(row["prompt_ids"])], row["prompt_ids"]
            )


if __name__ == "__main__":
    unittest.main()
