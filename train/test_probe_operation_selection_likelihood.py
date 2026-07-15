#!/usr/bin/env python3
"""CPU-only tests for the frozen operation-selection likelihood diagnostic."""

from __future__ import annotations

import copy
import inspect
import json
import math
import tempfile
import unittest
from unittest import mock
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

import probe_operation_selection_likelihood as probe


TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
SOURCE = ROOT / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json"
TOKENIZER = ROOT / "artifacts/shohin-tok-32k.json"
FROZEN_COMMIT = "a" * 40
EVIDENCE_COMMIT = "b" * 40
PRESCORE_RECEIPT_SHA256 = "c" * 64


class FrozenFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.source_rows, cls.source_transitions, cls.source_sha256 = (
            probe.load_frozen_source(SOURCE)
        )
        cls.subset_rows, cls.subset_transitions = probe.select_frozen_subset(
            cls.source_rows
        )
        cls.frozen = probe.build_frozen_transitions(cls.subset_rows)
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))
        cls.prepared = probe.prepare_transitions(cls.frozen, cls.tokenizer)

    @staticmethod
    def synthetic_logits(item, prompt):
        gold_index = probe.OPERATIONS.index(item.frozen.gold_operation)
        wrong_index = (gold_index + 1) % len(probe.OPERATIONS)
        logits = [0.0] * len(probe.OPERATIONS)
        if prompt.arm == probe.FULL_SOURCE_CURSOR:
            logits[gold_index] = 4.0
        elif prompt.arm == probe.RESIDUAL_SUFFIX_HEAD:
            if item.frozen.index % 2 == 0:
                logits[gold_index] = 4.0
            else:
                logits[wrong_index] = 4.0
        elif item.frozen.gold_operation in {"add", "remainder"}:
            logits[gold_index] = 4.0
        elif item.frozen.row_id == "multiply_subtract_000" and item.frozen.index == 0:
            logits[gold_index] = 4.0
            logits[wrong_index] = 4.0
        else:
            logits[wrong_index] = 4.0
        return logits

    @classmethod
    def make_rows(cls):
        return probe.evaluate_prepared_transitions(
            cls.prepared, cls.synthetic_logits
        )

    @classmethod
    def result_arguments(cls):
        return {
            "source_path": SOURCE,
            "checkpoint_path": ROOT / "never_loaded_raw260k.pt",
            "tokenizer_path": TOKENIZER,
            "source_sha256": probe.EXPECTED_SOURCE_SHA256,
            "checkpoint_sha256": probe.EXPECTED_CHECKPOINT_SHA256,
            "tokenizer_sha256": probe.EXPECTED_TOKENIZER_SHA256,
            "candidate_sha256": probe.EXPECTED_CANDIDATE_MANIFEST_SHA256,
            "prompt_sha256": probe.EXPECTED_PROMPT_MANIFEST_SHA256,
            "tokenized_sha256": probe.EXPECTED_TOKENIZED_MANIFEST_SHA256,
            "implementation_hashes": {
                "preregistration": "1" * 64,
                "evaluator": "2" * 64,
                "tests": "3" * 64,
                "job": "4" * 64,
                "model_loader": "5" * 64,
                "inherited_operation_cursor_contract": "6" * 64,
                "inherited_operation_cursor_geometry": "7" * 64,
            },
            "frozen_commit": FROZEN_COMMIT,
            "evidence_commit": EVIDENCE_COMMIT,
            "prescore_receipt_sha256": PRESCORE_RECEIPT_SHA256,
            "device": {
                "type": "cuda",
                "index": 0,
                "name": "synthetic H100 metadata only",
                "compute_capability": "9.0",
                "visible_device_count": 1,
                "torch_version": "test",
                "cuda_runtime": "test",
                "autocast": False,
                "candidate_logit_dtype": "float32",
            },
            "prepared": cls.prepared,
        }


class SourceAndManifestTests(FrozenFixture):
    def test_frozen_source_subset_and_transition_geometry(self) -> None:
        self.assertEqual(self.source_sha256, probe.EXPECTED_SOURCE_SHA256)
        self.assertEqual(self.source_transitions, 704)
        self.assertEqual(self.subset_transitions, 176)
        self.assertEqual(len(self.source_rows), 256)
        self.assertEqual(len(self.subset_rows), 64)
        self.assertEqual(len(self.frozen), 176)
        self.assertEqual(
            probe.digest_rows(self.subset_rows), probe.EXPECTED_SUBSET_ROWS_SHA256
        )
        self.assertEqual(
            [row["id"] for row in self.subset_rows],
            [
                f"{family}_{index:03d}"
                for family in probe.FAMILIES
                for index in range(16)
            ],
        )
        self.assertEqual(
            Counter(item.family for item in self.frozen),
            probe.EXPECTED_TRANSITIONS_BY_FAMILY,
        )
        self.assertEqual(
            Counter(item.index for item in self.frozen),
            probe.EXPECTED_TRANSITIONS_BY_INDEX,
        )
        self.assertEqual(
            Counter(item.gold_operation for item in self.frozen),
            probe.EXPECTED_TRANSITIONS_BY_OPERATION,
        )

    def test_source_json_and_file_custody_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            probe.strict_json_loads(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            probe.strict_json_loads(b'{"a":NaN}\n')

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writable = root / "source.json"
            writable.write_bytes(SOURCE.read_bytes())
            writable.chmod(0o600)
            with self.assertRaisesRegex(PermissionError, "no write bits"):
                probe.load_frozen_source(writable)

            tampered = root / "tampered.json"
            payload = bytearray(SOURCE.read_bytes())
            payload[-2] = ord(" ")
            tampered.write_bytes(payload)
            tampered.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "hash"):
                probe.load_frozen_source(tampered)

    def test_candidate_prompt_and_tokenized_manifests(self) -> None:
        self.assertEqual(
            probe.candidate_manifest_sha256(),
            probe.EXPECTED_CANDIDATE_MANIFEST_SHA256,
        )
        self.assertEqual(
            probe.prompt_manifest_sha256(self.frozen),
            probe.EXPECTED_PROMPT_MANIFEST_SHA256,
        )
        self.assertEqual(
            probe.tokenized_manifest_sha256(self.prepared),
            probe.EXPECTED_TOKENIZED_MANIFEST_SHA256,
        )
        self.assertEqual(
            probe.verify_untokenized_manifests(self.frozen),
            (
                probe.EXPECTED_CANDIDATE_MANIFEST_SHA256,
                probe.EXPECTED_PROMPT_MANIFEST_SHA256,
            ),
        )
        self.assertEqual(
            probe.verify_tokenized_manifest(self.prepared),
            probe.EXPECTED_TOKENIZED_MANIFEST_SHA256,
        )

    def test_real_tokenizer_candidate_ids_and_every_boundary(self) -> None:
        expected = {
            "add": (" add", 820),
            "subtract": (" subtract", 5498),
            "multiply": (" multiply", 4307),
            "remainder": (" remainder", 7486),
        }
        self.assertEqual(
            probe.hash_regular_file(TOKENIZER), probe.EXPECTED_TOKENIZER_SHA256
        )
        for candidate in probe.CANDIDATES:
            text, token_id = expected[candidate.operation]
            self.assertEqual((candidate.text, candidate.token_id), (text, token_id))
            self.assertEqual(self.tokenizer.encode(text).ids, [token_id])
            self.assertEqual(self.tokenizer.decode([token_id]), text)

        boundary_count = 0
        for item in self.prepared:
            for prompt in item.prompts:
                for candidate in probe.CANDIDATES:
                    combined = self.tokenizer.encode(prompt.text + candidate.text).ids
                    self.assertEqual(
                        combined[: len(prompt.token_ids)], list(prompt.token_ids)
                    )
                    self.assertEqual(combined[len(prompt.token_ids) :], [candidate.token_id])
                    boundary_count += 1
        self.assertEqual(boundary_count, 2112)
        self.assertEqual(
            probe.prompt_token_counts(self.prepared),
            (probe.EXPECTED_PROMPT_TOKENS_BY_ARM, 33160, 79),
        )

    def test_prompt_templates_and_arm_exposure_are_exact(self) -> None:
        first = self.frozen[0]
        self.assertEqual(first.row_id, "multiply_subtract_000")
        self.assertEqual(first.index, 0)
        self.assertEqual(first.gold_operation, "multiply")
        self.assertEqual(
            first.prompt_for(probe.FULL_SOURCE_CURSOR),
            "Task: Select the operation at the supplied cursor.\n"
            f"Source: {first.question}\n"
            "Step index (zero-based): 0\n"
            "Candidate operations: add, subtract, multiply, remainder.\n"
            "The operation at that cursor is",
        )
        self.assertEqual(
            first.prompt_for(probe.RESIDUAL_SUFFIX_HEAD),
            "Task: Select the first operation in the supplied residual suffix.\n"
            f"Residual suffix (read-only JSON): {probe.render_residual_suffix(first.residual_suffix)}\n"
            "Candidate operations: add, subtract, multiply, remainder.\n"
            "The residual head operation is",
        )
        self.assertEqual(
            first.prompt_for(probe.RESIDUAL_SUFFIX_ORACLE_STATE),
            "Task: Select the first operation in the supplied residual suffix.\n"
            f"Current state (oracle-supplied for this arm): {first.current_state}\n"
            f"Residual suffix (read-only JSON): {probe.render_residual_suffix(first.residual_suffix)}\n"
            "Candidate operations: add, subtract, multiply, remainder.\n"
            "The residual head operation is",
        )
        for item in self.frozen:
            full = item.prompt_for(probe.FULL_SOURCE_CURSOR)
            suffix = item.prompt_for(probe.RESIDUAL_SUFFIX_HEAD)
            state = item.prompt_for(probe.RESIDUAL_SUFFIX_ORACLE_STATE)
            self.assertIn(item.question, full)
            self.assertIn(f"Step index (zero-based): {item.index}\n", full)
            self.assertNotIn("Residual suffix", full)
            self.assertNotIn("Current state", suffix)
            self.assertIn("Residual suffix (read-only JSON):", suffix)
            self.assertIn(f"Current state (oracle-supplied for this arm): {item.current_state}", state)
            for prompt in (full, suffix, state):
                self.assertTrue(prompt.endswith("is"))
                self.assertNotIn("verifier", prompt.lower())
                self.assertNotIn("score", prompt.lower())


class ScoringAndAccountingTests(FrozenFixture):
    def test_candidate_scoring_correct_wrong_and_tied(self) -> None:
        correct = probe.score_candidate_logits("add", [4.0, 3.0, 2.0, 1.0])
        self.assertTrue(correct["unique_top1"])
        self.assertTrue(correct["correct"])
        self.assertEqual(correct["prediction"], "add")
        self.assertEqual(correct["gold_logit_margin_to_best_incorrect"], 1.0)
        self.assertAlmostEqual(
            sum(row["restricted_probability"] for row in correct["candidates"]),
            1.0,
            places=14,
        )

        wrong = probe.score_candidate_logits("add", [1.0, 2.0, 4.0, 3.0])
        self.assertTrue(wrong["unique_top1"])
        self.assertFalse(wrong["correct"])
        self.assertEqual(wrong["prediction"], "multiply")
        self.assertEqual(wrong["gold_logit_margin_to_best_incorrect"], -3.0)

        tied = probe.score_candidate_logits("add", [4.0, 4.0, 0.0, 0.0])
        self.assertFalse(tied["unique_top1"])
        self.assertFalse(tied["correct"])
        self.assertIsNone(tied["prediction"])
        self.assertEqual(tied["top_operations"], ["add", "subtract"])

    def test_candidate_scoring_rejects_bad_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            probe.score_candidate_logits("add", [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            probe.score_candidate_logits("add", [1.0, math.nan, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "numeric"):
            probe.score_candidate_logits("add", [1.0, True, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "gold operation"):
            probe.score_candidate_logits("divide", [1.0, 2.0, 3.0, 4.0])

    def test_evaluation_call_order_resources_and_grouped_accuracy(self) -> None:
        call_order = []

        def scorer(item, prompt):
            call_order.append((item.frozen.row_id, item.frozen.index, prompt.arm))
            return self.synthetic_logits(item, prompt)

        rows, observed = probe.evaluate_prepared_transitions(self.prepared, scorer)
        self.assertEqual(len(rows), 64)
        self.assertEqual(observed, Counter({arm: 176 for arm in probe.ARMS}))
        self.assertEqual(len(call_order), 528)
        self.assertEqual(
            call_order[:3],
            [
                ("multiply_subtract_000", 0, probe.FULL_SOURCE_CURSOR),
                ("multiply_subtract_000", 0, probe.RESIDUAL_SUFFIX_HEAD),
                (
                    "multiply_subtract_000",
                    0,
                    probe.RESIDUAL_SUFFIX_ORACLE_STATE,
                ),
            ],
        )
        probe.audit_execution(rows, self.prepared, observed)

        summary = probe.build_summary(rows)
        self.assertEqual(
            summary["by_arm"][probe.FULL_SOURCE_CURSOR]["accuracy"],
            {"numerator": 176, "denominator": 176},
        )
        self.assertEqual(
            summary["by_arm"][probe.RESIDUAL_SUFFIX_HEAD]["accuracy"],
            {"numerator": 96, "denominator": 176},
        )
        self.assertEqual(
            summary["by_arm"][probe.RESIDUAL_SUFFIX_ORACLE_STATE]["accuracy"],
            {"numerator": 80, "denominator": 176},
        )
        self.assertEqual(
            summary["by_arm"][probe.RESIDUAL_SUFFIX_ORACLE_STATE]["ties"],
            {"numerator": 1, "denominator": 176},
        )
        self.assertEqual(
            summary["by_operation"]["add"]["by_arm"]
            [probe.RESIDUAL_SUFFIX_ORACLE_STATE]["accuracy"],
            {"numerator": 64, "denominator": 64},
        )
        self.assertEqual(
            summary["by_operation"]["multiply"]["by_arm"]
            [probe.RESIDUAL_SUFFIX_ORACLE_STATE]["accuracy"],
            {"numerator": 0, "denominator": 64},
        )
        self.assertEqual(
            {name: value["transition_count"] for name, value in summary["by_family"].items()},
            probe.EXPECTED_TRANSITIONS_BY_FAMILY,
        )
        self.assertEqual(
            {int(name): value["transition_count"] for name, value in summary["by_step_index"].items()},
            probe.EXPECTED_TRANSITIONS_BY_INDEX,
        )

        ledger = probe.resource_ledger(self.prepared)
        self.assertEqual(ledger["model_forward_calls"], 528)
        self.assertEqual(ledger["candidate_logit_values_scored"], 2112)
        self.assertEqual(ledger["model_input_token_positions"], 33160)
        self.assertEqual(ledger["candidate_tokens_appended_to_model_input"], 0)
        self.assertEqual(ledger["implementation_hash_passes"], 7)
        self.assertEqual(ledger["quarantine_result_files_created"], 1)
        self.assertEqual(ledger["preserved_result_copies"], 2)
        self.assertEqual(ledger["read_only_receipt_files"], 1)
        self.assertEqual(ledger["authenticated_prescore_remote_verifications"], 1)
        self.assertEqual(ledger["read_only_git_bundles"], 1)
        self.assertEqual(ledger["temporary_bare_git_repositories"], 1)
        self.assertEqual(ledger["private_runtime_directories"], 1)
        self.assertEqual(ledger["kernel_sealed_runtime_snapshots"], 2)
        self.assertEqual(ledger["kernel_sealed_implementation_memfds_created"], 14)
        self.assertEqual(ledger["h100_preflight_allocations"], 2)
        self.assertEqual(ledger["temporary_prescore_receipt_files"], 1)
        for zero_field in (
            "generated_tokens",
            "sampled_tokens",
            "training_tokens",
            "retries",
            "repairs",
            "searches",
            "threshold_searches",
            "verifier_feedback_calls",
            "external_generation_calls",
            "mutable_sidecars",
        ):
            self.assertEqual(ledger[zero_field], 0)

    def test_next_operation_scorer_uses_one_forward_for_four_logits(self) -> None:
        prompt = self.prepared[0].prompt_for(probe.FULL_SOURCE_CURSOR)

        class TinySyntheticModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, tokens):
                self.calls += 1
                logits = torch.zeros(
                    (1, tokens.shape[1], 8000), dtype=torch.float32
                )
                for offset, candidate in enumerate(probe.CANDIDATES):
                    logits[0, -1, candidate.token_id] = float(offset + 1)
                return logits, None

        model = TinySyntheticModel()
        values = probe.next_operation_candidate_logits(model, prompt, "cpu")
        self.assertEqual(values, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(model.calls, 1)
        source = inspect.getsource(probe.next_operation_candidate_logits)
        self.assertEqual(source.count("model(tokens)"), 1)
        self.assertNotIn("candidate.text", source)

    def test_attested_model_source_bypasses_import_caches(self) -> None:
        source = b"class GPT: pass\nclass GPTConfig: pass\n"
        gpt, config = probe.model_classes_from_source(source, "attested_model.py")
        self.assertEqual(gpt.__name__, "GPT")
        self.assertEqual(config.__name__, "GPTConfig")
        self.assertEqual(gpt.__module__, "_shohin_attested_operation_likelihood_model")
        self.assertNotIn(
            "_shohin_attested_operation_likelihood_model", __import__("sys").modules
        )
        implementation = inspect.getsource(probe.load_model)
        self.assertNotIn("from model import", implementation)
        self.assertNotIn("from train.model import", implementation)


class ResultCustodyTests(FrozenFixture):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        rows, observed = cls.make_rows()
        cls.result_args = cls.result_arguments()
        cls.result = probe.build_result(
            **cls.result_args,
            rows=rows,
            observed_calls=observed,
        )

    def assert_result_rejected(self, value) -> None:
        with self.assertRaises(ValueError):
            probe.audit_preserved_result(value, **self.result_args)

    def test_repository_root_uses_sentinel_for_shallow_sealed_file(self) -> None:
        self.assertEqual(
            probe.repository_root("/proc/self/fd/4", sealed_runtime=True),
            Path("/"),
        )
        self.assertEqual(
            probe.repository_root(str(TRAIN_DIR / "probe.py"), sealed_runtime=False),
            ROOT,
        )

    def test_complete_result_audits(self) -> None:
        self.assertTrue(
            probe.audit_preserved_result(self.result, **self.result_args)
        )
        self.assertEqual(self.result["schema"], probe.RESULT_SCHEMA)
        self.assertEqual(
            self.result["summary"]["by_arm"][probe.FULL_SOURCE_CURSOR]["accuracy"],
            {"numerator": 176, "denominator": 176},
        )

    def test_result_mutations_are_rejected(self) -> None:
        mutations = []

        value = copy.deepcopy(self.result)
        value["schema"] = "changed"
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["id"] = "changed"
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["steps"][0][probe.FULL_SOURCE_CURSOR]["prompt"] += " "
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["steps"][0][probe.FULL_SOURCE_CURSOR]["prompt_token_ids"][0] += 1
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["steps"][0][probe.FULL_SOURCE_CURSOR]["candidates"][0]["token_id"] += 1
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["steps"][0][probe.FULL_SOURCE_CURSOR]["candidates"][0]["logit"] += 0.25
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["rows"][0]["steps"][0][probe.FULL_SOURCE_CURSOR]["prediction"] = None
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["summary"]["by_arm"][probe.FULL_SOURCE_CURSOR]["accuracy"]["numerator"] -= 1
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["resource_ledger"]["model_forward_calls"] -= 1
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["bindings"]["checkpoint_sha256"] = "0" * 64
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["bindings"]["frozen_commit"] = "0" * 40
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["bindings"]["evidence_commit"] = "0" * 40
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["bindings"]["prescore_receipt_sha256"] = "0" * 64
        mutations.append(value)

        value = copy.deepcopy(self.result)
        value["integrity"]["exclusive_read_only_output"] = False
        mutations.append(value)

        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.assert_result_rejected(mutation)

    def test_checkpoint_metadata_binding_rejects_drift(self) -> None:
        payload = {
            "step": 260000,
            "cfg": {
                "seq_len": 2048,
                "n_layer": 30,
                "d_model": 576,
                "n_loop": 1,
            },
            "model": {},
        }
        self.assertEqual(probe.validate_checkpoint_metadata(payload), payload["cfg"])
        for key, changed in (
            ("step", 259999),
            ("seq_len", 1024),
            ("n_layer", 29),
            ("d_model", 512),
            ("n_loop", 2),
        ):
            mutated = copy.deepcopy(payload)
            if key == "step":
                mutated[key] = changed
            else:
                mutated["cfg"][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_checkpoint_metadata(mutated)
        boolean_step = copy.deepcopy(payload)
        boolean_step["step"] = True
        with self.assertRaises(ValueError):
            probe.validate_checkpoint_metadata(boolean_step)

    def test_immutable_output_is_exclusive_ascii_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / f"{probe.OUTPUT_PREFIX}unit.json"
            digest = probe.write_immutable_json(output, {"schema": "unit", "x": 1})
            payload = output.read_bytes()
            self.assertEqual(digest, probe.sha256_bytes(payload))
            self.assertEqual(json.loads(payload), {"schema": "unit", "x": 1})
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertEqual(output.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                probe.write_immutable_json(output, {"schema": "changed"})

            wrong_name = root / "result.json"
            with self.assertRaisesRegex(ValueError, "filename"):
                probe.write_immutable_json(wrong_name, {"schema": "unit"})
            nonfinite = root / f"{probe.OUTPUT_PREFIX}nonfinite.json"
            with self.assertRaises(ValueError):
                probe.write_immutable_json(nonfinite, {"x": math.nan})
            self.assertFalse(nonfinite.exists())

    def test_implementation_manifest_covers_exact_runtime_surface(self) -> None:
        paths = probe.implementation_source_paths()
        self.assertEqual(
            set(paths),
            {
                "preregistration",
                "evaluator",
                "tests",
                "job",
                "model_loader",
                "inherited_operation_cursor_contract",
                "inherited_operation_cursor_geometry",
            },
        )
        for path in paths.values():
            self.assertTrue(path.is_file(), path)
        hashes = probe.hash_implementation(paths)
        self.assertEqual(set(hashes), set(paths))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertEqual(len(probe.implementation_manifest_sha256(hashes)), 64)


class WrapperContractTests(unittest.TestCase):
    def test_isolated_h100_wrapper_is_frozen_and_non_training(self) -> None:
        wrapper_path = TRAIN_DIR / "jobs/probe_operation_selection_likelihood.sbatch"
        wrapper = wrapper_path.read_text(encoding="ascii")
        for required in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:nvidia_h100_pcie:1",
            "#SBATCH --exclude=evc34",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=64G",
            "#SBATCH --time=02:00:00",
            "set -euo pipefail",
            "export PATH=/usr/bin:/bin",
            "export GIT_NO_REPLACE_OBJECTS=1",
            "unset GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "BASE=/lustre/fs1/home/sa305415/shohin",
            "PY=$BASE/miniforge3/bin/python",
            "export PYTHONSAFEPATH=1",
            "unset LD_PRELOAD PYTHONHOME PYTHONPATH",
            "export PYTHONOPTIMIZE=0",
            "EXPECTED_CKPT=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d",
            "EXPECTED_TOKENIZER=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4",
            "EXPECTED_SOURCE=19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474",
            "raw260k_operation_selection_likelihood_*.json",
            "probe_operation_selection_likelihood.py",
            'git --no-replace-objects --git-dir="$EVIDENCE_GIT_DIR" bundle verify "$EVIDENCE_BUNDLE"',
            'git init --bare --quiet "$EVIDENCE_GIT_DIR"',
            'git --no-replace-objects --git-dir="$EVIDENCE_GIT_DIR" -c pack.threads=1 -c index.threads=1 fetch --quiet "$EVIDENCE_BUNDLE" refs/heads/main:refs/heads/main',
            '"--no-replace-objects"',
            "create_memfd",
            "libc.memfd_create",
            'getattr(fcntl, "F_ADD_SEALS", 1033)',
            'getattr(fcntl, "F_SEAL_WRITE", 0x0008)',
            "R12_OPERATION_SELECTION_PRESCORE_RECEIPT.json",
            "authenticated_gh_api_commit_lookup_and_origin_main_match",
            ".operation_selection_quarantine.",
            "full_preserved_result_audit",
            "score_printed_by_wrapper",
            "score_released",
            "MIRROR_ROOT",
            "RECEIPT",
            "SHOHIN_SEALED_IMPLEMENTATION_PATHS",
            "AUDITED_RESULT_SHA256",
            "PUBLISH_COMPLETE=0",
            "cleanup_publication",
            'RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}',
            "cleanup_runtime",
            "fsync_parent(receipt_path)",
            '"audited_result_sha256": audited_sha256',
            '--frozen-commit "$FROZEN_COMMIT"',
            '--evidence-commit "$EVIDENCE_COMMIT"',
            '--prescore-receipt-sha256 "$PRESCORE_RECEIPT_SHA256"',
        ):
            self.assertIn(required, wrapper)

        executable = "\n".join(
            line for line in wrapper.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("sbatch ", executable)
        self.assertNotIn("train.py", executable)
        self.assertNotIn("sft.py", executable)
        self.assertNotIn("optimizer", executable.lower())
        self.assertNotIn("--device", executable)
        self.assertEqual(executable.count("run_frozen_probe \\\n"), 2)
        self.assertEqual(executable.count('--audit-result "$QUARANTINE_OUT"'), 1)
        self.assertNotIn('"summary": result["summary"]', executable)
        self.assertNotIn("assert value.get", executable)
        self.assertNotIn("SLURM_TMPDIR is required", executable)
        self.assertLess(wrapper.index("receipt_fd ="), wrapper.index("mirror_sha256 ="))
        self.assertLess(wrapper.index("receipt_fd ="), wrapper.index("published_sha256 ="))

    def test_procfd_reader_duplicates_and_requires_full_seals(self) -> None:
        with tempfile.TemporaryFile() as source:
            source.write(b"sealed-runtime")
            source.flush()
            source.seek(0)
            path = f"/proc/self/fd/{source.fileno()}"
            with mock.patch.object(
                probe.fcntl, "fcntl", return_value=probe.REQUIRED_MEMFD_SEALS
            ):
                self.assertEqual(probe.read_regular_file_bytes(path), b"sealed-runtime")
                self.assertEqual(
                    probe.hash_regular_file(path),
                    probe.sha256_bytes(b"sealed-runtime"),
                )
            with mock.patch.object(probe.fcntl, "fcntl", return_value=0):
                with self.assertRaisesRegex(PermissionError, "not fully sealed"):
                    probe.read_regular_file_bytes(path)


if __name__ == "__main__":
    unittest.main()
