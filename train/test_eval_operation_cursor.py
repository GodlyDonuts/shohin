import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

import eval_operation_cursor as evaluate


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json"
TOKENIZER_PATH = ROOT / "artifacts/shohin-tok-32k.json"
JOB_PATH = ROOT / "train/jobs/eval_operation_cursor.sbatch"


def load_source_and_subset():
    source, rows, transitions, source_hash = evaluate.load_frozen_source(SOURCE_PATH)
    subset, subset_transitions = evaluate.select_frozen_subset(rows)
    return source, rows, transitions, source_hash, subset, subset_transitions


def decoded_record(prompt, response):
    return {
        "prompt": prompt,
        "max_new": evaluate.MAX_NEW,
        "response": response,
        "prompt_token_count": 7,
        "untruncated_prompt_token_count": 7,
        "prompt_truncated": False,
        "sampled_token_count": 2,
        "decoded_token_count": 1,
        "stop_reason": "eos",
    }


def gold_results(subset):
    results = []
    for row in subset:
        current_state, schedule = evaluate.reconstruct_schedule(row)
        steps = []
        for index, (operation, operand) in enumerate(schedule):
            residual = schedule[index:]
            next_state = evaluate.apply_operation(current_state, operation, operand)
            selector_response = json.dumps(
                {"operation": operation, "operand": operand}, separators=(",", ":")
            )
            state_response = json.dumps(
                {
                    "operation": operation,
                    "operand": operand,
                    "next_state": next_state,
                },
                separators=(",", ":"),
            )
            source_record = decoded_record(
                evaluate.source_step_prompt(row["question"], index), selector_response
            )
            suffix_record = decoded_record(
                evaluate.residual_suffix_prompt(residual), selector_response
            )
            state_record = decoded_record(
                evaluate.residual_state_prompt(current_state, residual), state_response
            )
            steps.append(
                {
                    "index": index,
                    evaluate.SOURCE_STEP_SELECTOR: evaluate.score_call_record(
                        source_record, operation, operand
                    ),
                    evaluate.RESIDUAL_SUFFIX_SELECTOR: evaluate.score_call_record(
                        suffix_record, operation, operand
                    ),
                    evaluate.RESIDUAL_SUFFIX_STATE_UPDATE: evaluate.score_call_record(
                        state_record,
                        operation,
                        operand,
                        expected_next_state=next_state,
                        include_next_state=True,
                    ),
                }
            )
            current_state = next_state
        results.append({"id": row["id"], "family": row["family"], "steps": steps})
    return results


def expected_call_counts():
    return Counter({arm: evaluate.DIAGNOSTIC_TRANSITION_COUNT for arm in evaluate.ARMS})


class FrozenSourceAndSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.source,
            cls.rows,
            cls.source_transitions,
            cls.source_hash,
            cls.subset,
            cls.subset_transitions,
        ) = load_source_and_subset()

    def test_source_artifact_and_first_16_per_family_are_exact(self):
        self.assertEqual(self.source_hash, evaluate.EXPECTED_SOURCE_SHA256)
        self.assertEqual(self.source_transitions, evaluate.SOURCE_TRANSITION_COUNT)
        self.assertEqual(len(self.subset), evaluate.DIAGNOSTIC_CASE_COUNT)
        self.assertEqual(self.subset_transitions, evaluate.DIAGNOSTIC_TRANSITION_COUNT)
        self.assertEqual(
            evaluate.digest_rows(self.subset), evaluate.EXPECTED_SUBSET_ROWS_SHA256
        )
        self.assertEqual(SOURCE_PATH.stat().st_mode & 0o222, 0)
        self.assertEqual(
            [row["id"] for row in self.subset],
            [
                f"{family}_{index:03d}"
                for family in evaluate.FAMILIES
                for index in range(evaluate.DIAGNOSTIC_PER_FAMILY)
            ],
        )

    def test_subset_transition_geometry_is_reconstructed(self):
        family_transitions = Counter()
        operation_counts = Counter()
        step_index_counts = Counter()
        for row in self.subset:
            start, schedule = evaluate.reconstruct_schedule(row)
            self.assertEqual(start, row["initial_state"])
            family_transitions[row["family"]] += len(schedule)
            for index, (operation, operand) in enumerate(schedule):
                self.assertIs(type(operand), int)
                operation_counts[operation] += 1
                step_index_counts[index] += 1
        self.assertEqual(
            family_transitions,
            Counter(
                {
                    "multiply_subtract": 32,
                    "base_conversion": 64,
                    "sequential_state": 48,
                    "modular_update": 32,
                }
            ),
        )
        self.assertEqual(
            operation_counts,
            Counter({"multiply": 64, "subtract": 32, "add": 64, "remainder": 16}),
        )
        self.assertEqual(step_index_counts, Counter({0: 64, 1: 64, 2: 32, 3: 16}))

    def test_source_and_tokenizer_are_bound_to_known_hashes(self):
        self.assertEqual(
            evaluate.sha256_file(SOURCE_PATH), evaluate.EXPECTED_SOURCE_SHA256
        )
        self.assertEqual(
            evaluate.sha256_file(TOKENIZER_PATH), evaluate.EXPECTED_TOKENIZER_SHA256
        )

    def test_source_reconstruction_rejects_schedule_tampering(self):
        row = copy.deepcopy(self.rows[0])
        row["schedule"][0][1] += 1
        with self.assertRaisesRegex(
            ValueError, "question and source schedule disagree"
        ):
            evaluate.reconstruct_schedule(row)

        source = copy.deepcopy(self.source)
        source["rows"][0]["schedule"][0][1] += 1
        source["cases_sha256"] = evaluate.digest_rows(source["rows"])
        with self.assertRaisesRegex(ValueError, "wrong frozen source rows hash"):
            evaluate.audit_source(source)

    def test_subset_selection_rejects_reordering(self):
        rows = copy.deepcopy(self.rows)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "first 16 per family"):
            evaluate.select_frozen_subset(rows)


class PromptAndParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, _, _, _, cls.subset, _ = load_source_and_subset()

    def test_prompt_snapshots_and_exposure_boundaries(self):
        row = self.subset[0]
        state, schedule = evaluate.reconstruct_schedule(row)
        self.assertEqual(
            evaluate.source_step_prompt(row["question"], 0),
            "Task: Select the operation and operand at the supplied cursor.\n"
            "Source: Compute 81 times 5, then subtract 34.\n"
            "Step index (zero-based): 0\n"
            "Output schema: operation is one of add, subtract, multiply, remainder; "
            "operand is an integer.\n"
            "Return only one JSON object with exactly the keys operation and operand, "
            "and no other text.\n"
            "JSON:",
        )
        self.assertEqual(
            evaluate.render_residual_suffix(schedule),
            '[["multiply",5],["subtract",34]]',
        )

        for row in self.subset:
            current_state, schedule = evaluate.reconstruct_schedule(row)
            for index, (operation, operand) in enumerate(schedule):
                residual = schedule[index:]
                source_prompt = evaluate.source_step_prompt(row["question"], index)
                suffix_prompt = evaluate.residual_suffix_prompt(residual)
                state_prompt = evaluate.residual_state_prompt(current_state, residual)

                self.assertIn(row["question"], source_prompt)
                self.assertIn(f"Step index (zero-based): {index}", source_prompt)
                self.assertNotIn("Residual suffix", source_prompt)
                self.assertNotIn("Current state", source_prompt)

                self.assertNotIn(row["question"], suffix_prompt)
                self.assertNotIn("Step index", suffix_prompt)
                self.assertNotIn("Current state", suffix_prompt)
                self.assertIn(evaluate.render_residual_suffix(residual), suffix_prompt)

                self.assertNotIn(row["question"], state_prompt)
                self.assertNotIn("Step index", state_prompt)
                self.assertIn(
                    f"Current state (oracle-supplied for this arm): {current_state}",
                    state_prompt,
                )
                self.assertIn(evaluate.render_residual_suffix(residual), state_prompt)
                for prompt in (source_prompt, suffix_prompt, state_prompt):
                    self.assertNotIn("expected next state", prompt.lower())
                    self.assertNotIn("final answer", prompt.lower())
                    self.assertNotIn("verifier", prompt.lower())

                source_numbers = [
                    int(value) for value in re.findall(r"-?\d+", row["question"])
                ]
                self.assertEqual(
                    [int(value) for value in re.findall(r"-?\d+", source_prompt)],
                    [*source_numbers, index],
                )
                residual_operands = [operand for _, operand in residual]
                self.assertEqual(
                    [int(value) for value in re.findall(r"-?\d+", suffix_prompt)],
                    residual_operands,
                )
                self.assertEqual(
                    [int(value) for value in re.findall(r"-?\d+", state_prompt)],
                    [current_state, *residual_operands],
                )
                current_state = evaluate.apply_operation(
                    current_state, operation, operand
                )

    def test_strict_selector_parser_accepts_only_exact_schema(self):
        valid = '  {"operand":5,"operation":"multiply"}\n'
        self.assertEqual(
            evaluate.parse_structured_response(valid),
            ({"operation": "multiply", "operand": 5}, None),
        )
        invalid = {
            "": "empty_response",
            "not json": "invalid_json",
            '{"operation":"add","operand":1} trailing': "invalid_json",
            '["add",1]': "not_object",
            '{"operation":"add","operand":1,"extra":0}': "wrong_keys",
            '{"operation":"add","operation":"subtract","operand":1}': "duplicate_key",
            '{"operation":1,"operand":1}': "operation_not_string",
            '{"operation":"divide","operand":1}': "operation_not_allowed",
            '{"operation":"add","operand":true}': "operand_not_integer",
            '{"operation":"add","operand":1.0}': "operand_not_integer",
            '{"operation":"add","operand":NaN}': "invalid_json_constant",
        }
        for response, error in invalid.items():
            with self.subTest(response=response):
                self.assertEqual(
                    evaluate.parse_structured_response(response), (None, error)
                )

    def test_strict_state_parser_requires_integer_next_state(self):
        valid = '{"operation":"add","operand":2,"next_state":5}'
        self.assertEqual(
            evaluate.parse_structured_response(valid, include_next_state=True),
            ({"operation": "add", "operand": 2, "next_state": 5}, None),
        )
        self.assertEqual(
            evaluate.parse_structured_response(
                '{"operation":"add","operand":2}', include_next_state=True
            ),
            (None, "wrong_keys"),
        )
        self.assertEqual(
            evaluate.parse_structured_response(
                '{"operation":"add","operand":2,"next_state":"5"}',
                include_next_state=True,
            ),
            (None, "next_state_not_integer"),
        )

    def test_greedy_decode_uses_only_eos_or_frozen_cap(self):
        class Encoding:
            ids = [0]

        class FakeTokenizer:
            pieces = {1: "answer ", 2: "7; ", 3: "continue", 4: ""}

            def encode(self, text):
                return Encoding()

            def token_to_id(self, token):
                return 4

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(self.pieces[token] for token in token_ids)

        class Config:
            seq_len = 64

        class FakeModel:
            cfg = Config()

            def __init__(self):
                self.tokens = iter((1, 2, 3, 4))

            def __call__(self, inputs, **kwargs):
                token = next(self.tokens)
                logits = torch.full((1, inputs.shape[1], 5), -1000.0)
                logits[:, -1, token] = 1.0
                return logits, object()

        completion = evaluate.greedy_completion(
            FakeModel(), FakeTokenizer(), "prompt", "cpu", evaluate.MAX_NEW
        )
        self.assertEqual(completion["response"], "answer 7; continue")
        self.assertEqual(completion["sampled_token_count"], 4)
        self.assertEqual(completion["decoded_token_count"], 3)
        self.assertEqual(completion["stop_reason"], "eos")


class ExecutionSummaryAndTamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.source,
            _,
            _,
            _,
            cls.subset,
            cls.transitions,
        ) = load_source_and_subset()
        cls.results = gold_results(cls.subset)

    def test_all_malformed_responses_still_issue_exactly_528_calls(self):
        malformed_completion = {
            "response": "not-json",
            "prompt_token_count": 1,
            "untruncated_prompt_token_count": 1,
            "prompt_truncated": False,
            "sampled_token_count": evaluate.MAX_NEW,
            "decoded_token_count": evaluate.MAX_NEW,
            "stop_reason": "max_new",
        }
        with mock.patch.object(
            evaluate, "greedy_completion", return_value=malformed_completion
        ) as completion:
            results, observed = evaluate.evaluate_rows(
                object(), object(), "cpu", self.subset
            )
        self.assertEqual(completion.call_count, evaluate.EXPECTED_MODEL_CALLS)
        self.assertEqual(observed, expected_call_counts())
        ledger, _ = evaluate.audit_execution(results, self.subset, observed)
        self.assertEqual(ledger["model_calls"], evaluate.EXPECTED_MODEL_CALLS)
        self.assertEqual(ledger["calls_not_issued_after_parse_failure"], 0)
        self.assertEqual(
            sum(
                step[arm]["parse_success"]
                for row in results
                for step in row["steps"]
                for arm in evaluate.ARMS
            ),
            0,
        )

    def test_execution_audit_and_exact_summaries_reconstruct_every_score(self):
        ledger, integrity = evaluate.audit_execution(
            self.results, self.subset, expected_call_counts()
        )
        self.assertEqual(ledger["model_calls"], evaluate.EXPECTED_MODEL_CALLS)
        self.assertEqual(ledger["prompt_token_count"], 7 * 528)
        self.assertEqual(ledger["sampled_token_count"], 2 * 528)
        self.assertTrue(all(integrity.values()))

        summary = evaluate.build_summary(self.results, self.subset)
        self.assertEqual(summary["case_count"], 64)
        self.assertEqual(summary["transition_count"], 176)
        self.assertEqual(summary["model_calls"], 528)
        for arm in evaluate.ARMS:
            self.assertEqual(
                summary["by_arm"][arm]["selection_correct"],
                {"numerator": 176, "denominator": 176},
            )
        self.assertEqual(
            summary["by_arm"][evaluate.RESIDUAL_SUFFIX_STATE_UPDATE]["joint_correct"],
            {"numerator": 176, "denominator": 176},
        )
        self.assertEqual(
            {
                operation: item["transition_count"]
                for operation, item in summary["by_operation"].items()
            },
            {"add": 64, "subtract": 32, "multiply": 64, "remainder": 16},
        )
        self.assertEqual(
            {
                index: item["transition_count"]
                for index, item in summary["by_step_index"].items()
            },
            {"0": 64, "1": 64, "2": 32, "3": 16},
        )
        paired = summary["paired_comparisons"]
        for cells in paired.values():
            self.assertEqual(cells["both_correct"], 176)
            self.assertEqual(cells["left_only_correct"], 0)
            self.assertEqual(cells["right_only_correct"], 0)
            self.assertEqual(cells["neither_correct"], 0)

    def test_execution_audit_rejects_transcript_and_accounting_tampering(self):
        def prompt_tamper(rows):
            rows[0]["steps"][0][evaluate.SOURCE_STEP_SELECTOR]["prompt"] += " changed"

        def response_tamper(rows):
            rows[0]["steps"][0][evaluate.RESIDUAL_SUFFIX_SELECTOR]["response"] = "{}"

        def parsed_tamper(rows):
            rows[0]["steps"][0][evaluate.SOURCE_STEP_SELECTOR]["parsed"]["operand"] += 1

        def score_tamper(rows):
            rows[0]["steps"][0][evaluate.SOURCE_STEP_SELECTOR]["selection_correct"] = (
                False
            )

        def token_tamper(rows):
            rows[0]["steps"][0][evaluate.SOURCE_STEP_SELECTOR][
                "sampled_token_count"
            ] = 3

        def index_tamper(rows):
            rows[0]["steps"][0]["index"] = 1

        def schema_tamper(rows):
            rows[0]["steps"][0].pop(evaluate.RESIDUAL_SUFFIX_SELECTOR)

        def missing_step(rows):
            rows[0]["steps"].pop()

        def row_order(rows):
            rows[0], rows[1] = rows[1], rows[0]

        mutations = {
            "prompt": prompt_tamper,
            "response": response_tamper,
            "parsed": parsed_tamper,
            "score": score_tamper,
            "token": token_tamper,
            "index": index_tamper,
            "schema": schema_tamper,
            "missing_step": missing_step,
            "row_order": row_order,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(self.results)
                mutate(rows)
                with self.assertRaises(ValueError):
                    evaluate.audit_execution(rows, self.subset, expected_call_counts())

        wrong_counts = expected_call_counts()
        wrong_counts[evaluate.SOURCE_STEP_SELECTOR] -= 1
        with self.assertRaisesRegex(ValueError, "observed model calls"):
            evaluate.audit_execution(self.results, self.subset, wrong_counts)

    def test_preserved_result_rejects_hash_summary_and_scope_tampering(self):
        input_hashes = {
            "checkpoint": evaluate.EXPECTED_CHECKPOINT_SHA256,
            "tokenizer": evaluate.EXPECTED_TOKENIZER_SHA256,
            "source": evaluate.EXPECTED_SOURCE_SHA256,
        }
        code_hashes = evaluate.hash_paths(evaluate.diagnostic_source_paths())
        result = evaluate.build_result(
            self.source,
            self.subset,
            self.transitions,
            evaluate.EXPECTED_CHECKPOINT_STEP,
            input_hashes,
            code_hashes,
            "cpu",
            copy.deepcopy(self.results),
            expected_call_counts(),
        )
        self.assertTrue(
            evaluate.audit_preserved_result(
                result,
                self.source,
                self.subset,
                self.transitions,
                input_hashes,
                code_hashes,
                "cpu",
            )
        )
        self.assertNotIn("gates", result)
        self.assertNotIn("advance", result)
        self.assertNotIn("answer", result["rows"][0])
        self.assertNotIn("schedule", result["rows"][0])

        mutations = []
        bad = copy.deepcopy(result)
        bad["summary"]["by_arm"][evaluate.SOURCE_STEP_SELECTOR]["selection_correct"][
            "numerator"
        ] -= 1
        mutations.append(bad)
        bad = copy.deepcopy(result)
        bad["input_sha256"]["checkpoint"] = "0" * 64
        mutations.append(bad)
        bad = copy.deepcopy(result)
        bad["diagnostic_scope"]["reasoning_claim"] = "pass"
        mutations.append(bad)
        bad = copy.deepcopy(result)
        bad["advance"] = True
        mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):
                evaluate.audit_preserved_result(
                    bad,
                    self.source,
                    self.subset,
                    self.transitions,
                    input_hashes,
                    code_hashes,
                    "cpu",
                )


class OutputAndJobTests(unittest.TestCase):
    def test_immutable_writer_is_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            expected_payload = b'{\n  "ok": true\n}\n'
            self.assertEqual(
                evaluate.write_immutable_json(path, {"ok": True}),
                hashlib.sha256(expected_payload).hexdigest(),
            )
            self.assertEqual(path.read_bytes(), expected_payload)
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                evaluate.write_immutable_json(path, {"ok": True})

            link = Path(directory) / "broken.json"
            os.symlink(Path(directory) / "missing-target", link)
            with self.assertRaises(FileExistsError):
                evaluate.write_immutable_json(link, {"ok": True})

    def test_code_manifest_covers_all_new_files_and_model_loader(self):
        paths = evaluate.diagnostic_source_paths()
        self.assertEqual(
            set(paths), {"contract", "evaluator", "tests", "job", "model_loader"}
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))
        self.assertEqual(
            evaluate.hash_paths(paths),
            {name: evaluate.sha256_file(path) for name, path in paths.items()},
        )

    def test_batch_wrapper_is_hash_bound_output_limited_and_non_training(self):
        text = JOB_PATH.read_text()
        self.assertIn(evaluate.EXPECTED_CHECKPOINT_SHA256, text)
        self.assertIn(evaluate.EXPECTED_TOKENIZER_SHA256, text)
        self.assertIn(evaluate.EXPECTED_SOURCE_SHA256, text)
        self.assertIn(
            "SOURCE=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json",
            text,
        )
        self.assertIn("raw260k_operation_cursor_*.json", text)
        self.assertIn("artifacts/eval_history", text)
        self.assertIn("--device cuda", text)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
        self.assertIn('model_calls") != 528', text)
        self.assertNotIn("train.py", text)
        self.assertNotIn("sft.py", text)
        self.assertFalse(
            any(
                line.strip().lower().startswith("sbatch ")
                for line in text.splitlines()
                if not line.lstrip().startswith("#")
            )
        )


if __name__ == "__main__":
    unittest.main()
