import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

import eval_scheduled_reasoning_confirmation as evaluate
import generate_scheduled_reasoning_confirmation as generate


ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json"


def load_board():
    return json.loads(BOARD_PATH.read_text())


def call_record(prompt, response, max_new):
    return {
        "prompt": prompt,
        "max_new": max_new,
        "response": response,
        "prompt_token_count": 1,
        "untruncated_prompt_token_count": 1,
        "prompt_truncated": False,
        "sampled_token_count": 2,
        "decoded_token_count": 1,
        "stop_reason": "eos",
    }


class FrozenBoardTests(unittest.TestCase):
    def test_frozen_board_is_exact_deterministic_and_balanced(self):
        first = generate.build_rows()
        second = generate.build_rows()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual(
            Counter(row["family"] for row in first),
            Counter({family: 64 for family in generate.FAMILIES}),
        )
        self.assertEqual(generate.digest_rows(first), generate.EXPECTED_CASES_SHA256)

        board = load_board()
        self.assertEqual(board["rows"], first)
        self.assertEqual(
            hashlib.sha256(BOARD_PATH.read_bytes()).hexdigest(),
            generate.EXPECTED_BOARD_SHA256,
        )
        self.assertEqual(BOARD_PATH.stat().st_mode & 0o222, 0)
        _, transitions = evaluate.audit_board(board)
        self.assertEqual(transitions, 704)
        loaded, rows, transitions, board_hash = evaluate.load_frozen_board(BOARD_PATH)
        self.assertEqual(loaded, board)
        self.assertEqual(rows, first)
        self.assertEqual(transitions, 704)
        self.assertEqual(board_hash, evaluate.EXPECTED_BOARD_SHA256)

    def test_generator_payload_reproduces_frozen_artifact_hash(self):
        rows = generate.build_rows()
        board = {
            "schema": generate.SCHEMA,
            "seed": generate.SEED,
            "per_family": generate.PER_FAMILY,
            "case_count": len(rows),
            "family_order": list(generate.FAMILIES),
            "cases_sha256": generate.digest_rows(rows),
            "rows": rows,
        }
        payload = (json.dumps(board, indent=2, sort_keys=True) + "\n").encode("ascii")
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), generate.EXPECTED_BOARD_SHA256
        )
        self.assertEqual(payload, BOARD_PATH.read_bytes())

    def test_implementation_manifest_covers_confirmation_runtime(self):
        paths = evaluate.confirmation_source_paths()
        self.assertEqual(
            set(paths), {"contract", "generator", "evaluator", "job", "model_loader"}
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))
        self.assertEqual(
            evaluate.hash_paths(paths),
            {name: evaluate.sha256_file(path) for name, path in paths.items()},
        )

    def test_audit_rejects_self_rehashed_board_substitute(self):
        board = copy.deepcopy(load_board())
        row = board["rows"][0]
        row["question"] = "Compute 82 times 5, then subtract 34."
        row["initial_state"] = 82
        row["answer"] = 376
        board["cases_sha256"] = evaluate.digest_rows(board["rows"])
        with self.assertRaisesRegex(ValueError, "wrong frozen cases hash"):
            evaluate.audit_board(board)

    def test_audit_requires_case_count_metadata(self):
        board = load_board()
        del board["case_count"]
        with self.assertRaisesRegex(ValueError, "case_count"):
            evaluate.audit_board(board)

    def test_independent_parser_enforces_question_shape_and_ranges(self):
        with self.assertRaisesRegex(ValueError, "unparsed base question"):
            evaluate._parse_question(
                "base_conversion", "Convert the base-8 numeral 1234 to base 10."
            )
        _, _, details = evaluate._parse_question(
            "multiply_subtract", "Compute 100 times 5, then subtract 2."
        )
        with self.assertRaisesRegex(ValueError, "frozen multiply_subtract ranges"):
            evaluate._validate_row_ranges("multiply_subtract", 0, details, 498)


class ParserAndDecodeTests(unittest.TestCase):
    def test_full_and_first_line_integer_policy(self):
        segment, predicted = evaluate.parse_full_response(
            " 19+6=25; 25*3=75; 75-11=64\nProblem: next"
        )
        self.assertEqual(predicted, 64)
        self.assertNotIn("next", segment)
        self.assertEqual(evaluate.parse_full_response(" 6,999\nQuestion 2: x")[1], 6999)
        self.assertEqual(evaluate.parse_first_line_final(" 6,999\n2"), 6999)
        self.assertIsNone(evaluate.parse_first_line_final(" malformed 6,99\n999"))
        self.assertIsNone(evaluate.parse_first_line_final(" value 1.5\n5"))

    def test_generated_header_at_start_cannot_supply_a_score(self):
        for response in ("Question: Compute 1 plus 2. Answer: 3", "Problem: 8 minus 1"):
            segment, predicted = evaluate.parse_full_response(response)
            self.assertEqual(segment, "")
            self.assertIsNone(predicted)

    def test_scheduled_parser_carries_comma_integer_and_stops_on_failure(self):
        prompts = []
        responses = iter(("6,999\nignored 1", "7,000"))

        def fake_call(model, tokenizer, prompt, device, max_new):
            prompts.append(prompt)
            return {"prompt": prompt, "response": next(responses)}

        with mock.patch.object(evaluate, "call", side_effect=fake_call):
            answer, steps = evaluate.run_scheduled(
                object(), object(), "cpu", 1, [("add", 0), ("add", 1)], 48
            )
        self.assertEqual(answer, 7000)
        self.assertEqual(len(steps), 2)
        self.assertEqual(prompts[1], evaluate.format_atomic_prompt(6999, "add", 1))
        self.assertNotIn("local_expected_state", steps[0])

        with mock.patch.object(
            evaluate,
            "call",
            return_value={"prompt": "unused", "response": "no integer here"},
        ) as patched:
            answer, steps = evaluate.run_scheduled(
                object(), object(), "cpu", 1, [("add", 1), ("multiply", 2)], 48
            )
        self.assertIsNone(answer)
        self.assertEqual(len(steps), 1)
        patched.assert_called_once()

    def test_greedy_decode_does_not_stop_on_answer_phrase(self):
        class Encoding:
            ids = [0]

        class FakeTokenizer:
            pieces = {
                1: "The answer is ",
                2: "7. ",
                3: "Actually 8.",
            }

            def encode(self, text):
                return Encoding()

            def token_to_id(self, token):
                return 4

            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(self.pieces[token] for token in token_ids)

        class Config:
            seq_len = 32

        class FakeModel:
            cfg = Config()

            def __init__(self):
                self.tokens = iter((1, 2, 3, 4))

            def __call__(self, inputs, **kwargs):
                next_token = next(self.tokens)
                logits = torch.full((1, inputs.shape[1], 5), -1000.0)
                logits[:, -1, next_token] = 1.0
                return logits, object()

        completion = evaluate.greedy_completion(
            FakeModel(), FakeTokenizer(), "prompt", "cpu", max_new=8
        )
        self.assertEqual(completion["response"], "The answer is 7. Actually 8.")
        self.assertEqual(completion["sampled_token_count"], 4)
        self.assertEqual(completion["decoded_token_count"], 3)
        self.assertEqual(completion["stop_reason"], "eos")


class StatisticsAndIntegrityTests(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(evaluate.exact_mcnemar_p(0, 0), 1.0)
        expected = 508 / (2**22)
        self.assertEqual(evaluate.exact_mcnemar_p(20, 2), expected)
        self.assertEqual(evaluate.exact_mcnemar_p(2, 20), expected)
        self.assertEqual(evaluate.exact_mcnemar_p(3, 3), 1.0)
        with self.assertRaises(ValueError):
            evaluate.exact_mcnemar_p(-1, 2)

    def test_all_locked_gate_boundaries(self):
        summary = {
            "case_count": 256,
            "scheduled_correct": 90,
            "direct_correct": 64,
            "atomic_correct": 493,
            "atomic_total": 704,
            "mcnemar_exact_p": 0.009,
            "by_family": {
                "multiply_subtract": {
                    "count": 64,
                    "scheduled_correct": 15,
                    "direct_correct": 13,
                },
                "base_conversion": {
                    "count": 64,
                    "scheduled_correct": 15,
                    "direct_correct": 13,
                },
                "sequential_state": {
                    "count": 64,
                    "scheduled_correct": 45,
                    "direct_correct": 25,
                },
                "modular_update": {
                    "count": 64,
                    "scheduled_correct": 15,
                    "direct_correct": 13,
                },
            },
        }
        gates, advance = evaluate.decide(summary)
        self.assertTrue(advance)
        self.assertTrue(all(gates.values()))

        mutations = (
            ("scheduled_absolute", lambda value: value.update(scheduled_correct=89)),
            ("scheduled_advantage", lambda value: value.update(direct_correct=65)),
            ("paired_significance", lambda value: value.update(mcnemar_exact_p=0.01)),
            (
                "family_nonregression",
                lambda value: value["by_family"]["multiply_subtract"].update(
                    scheduled_correct=12
                ),
            ),
            (
                "sequential_absolute",
                lambda value: value["by_family"]["sequential_state"].update(
                    scheduled_correct=44
                ),
            ),
            ("atomic_ceiling", lambda value: value.update(atomic_correct=492)),
        )
        for gate_name, mutate in mutations:
            value = copy.deepcopy(summary)
            mutate(value)
            gates, advance = evaluate.decide(value)
            self.assertFalse(gates[gate_name], gate_name)
            self.assertFalse(advance, gate_name)

    def test_execution_audit_counts_parse_failure_without_missing_transcript(self):
        row = load_board()["rows"][0]
        start, schedule = evaluate.parse_schedule(row)

        direct = call_record(
            evaluate.direct_prompt(row["question"]),
            str(row["answer"]),
            evaluate.MAX_NEW_FULL,
        )
        direct.update(
            {
                "answer_segment": str(row["answer"]),
                "predicted_answer": row["answer"],
                "correct": True,
            }
        )
        whole = call_record(
            evaluate.whole_prompt(row["question"]),
            str(row["answer"]),
            evaluate.MAX_NEW_FULL,
        )
        whole.update(
            {
                "answer_segment": str(row["answer"]),
                "predicted_answer": row["answer"],
                "correct": True,
            }
        )

        atomic = []
        true_state = start
        for index, (operation, operand) in enumerate(schedule):
            expected = evaluate.apply_operation(true_state, operation, operand)
            record = call_record(
                evaluate.format_atomic_prompt(true_state, operation, operand),
                str(expected),
                evaluate.MAX_NEW_ATOMIC,
            )
            record.update(
                {
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": true_state,
                    "expected_state": expected,
                    "predicted_state": expected,
                    "correct": True,
                }
            )
            atomic.append(record)
            true_state = expected

        scheduled_step = call_record(
            evaluate.format_atomic_prompt(start, *schedule[0]),
            "no integer",
            evaluate.MAX_NEW_ATOMIC,
        )
        scheduled_step.update(
            {
                "index": 0,
                "operation": schedule[0][0],
                "operand": schedule[0][1],
                "input_state": start,
                "predicted_state": None,
            }
        )
        result = {
            "id": row["id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "question": row["question"],
            "answer": row["answer"],
            "direct": direct,
            "whole_problem_work": whole,
            "atomic_oracle_state": atomic,
            "source_scheduled": {
                "predicted_answer": None,
                "correct": False,
                "steps": [scheduled_step],
            },
        }
        observed = Counter(
            {
                "direct_qa": 1,
                "whole_problem_work": 1,
                "atomic_oracle_state": 2,
                "source_scheduled": 1,
            }
        )
        ledger, integrity = evaluate.audit_execution(
            [result], [row], len(schedule), observed
        )
        self.assertEqual(ledger["model_calls"], 5)
        self.assertEqual(ledger["maximum_model_calls_without_parse_failures"], 6)
        self.assertEqual(ledger["scheduled_calls_not_issued_after_parse_failure"], 1)
        self.assertTrue(all(integrity.values()))

        extra = copy.deepcopy(result)
        extra["source_scheduled"]["steps"].append(copy.deepcopy(scheduled_step))
        with self.assertRaisesRegex(ValueError, "continued after parse failure"):
            evaluate.audit_execution([extra], [row], len(schedule), observed)

        with self.assertRaisesRegex(ValueError, "observed model calls"):
            evaluate.audit_execution(
                [result],
                [row],
                len(schedule),
                Counter({**observed, "source_scheduled": 2}),
            )

        wrong_cap = copy.deepcopy(result)
        wrong_cap["source_scheduled"]["steps"][0]["max_new"] = 47
        with self.assertRaisesRegex(ValueError, "decode cap deviation"):
            evaluate.audit_execution([wrong_cap], [row], len(schedule), observed)

    def test_immutable_writer_creates_read_only_output_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            expected = hashlib.sha256(b'{\n  "ok": true\n}\n').hexdigest()
            self.assertEqual(
                evaluate.write_immutable_json(path, {"ok": True}), expected
            )
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                evaluate.write_immutable_json(path, {"ok": True})


if __name__ == "__main__":
    unittest.main()
