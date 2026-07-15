import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

import assess_scheduled_reasoning_confirmation as assess

ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json"


def token_count(prompt):
    return len(prompt.encode("utf-8"))


def call_record(prompt, response, max_new):
    count = token_count(prompt)
    return {
        "prompt": prompt,
        "max_new": max_new,
        "response": response,
        "prompt_token_count": count,
        "untruncated_prompt_token_count": count,
        "prompt_truncated": False,
        "sampled_token_count": 2,
        "decoded_token_count": 1,
        "stop_reason": "eos",
    }


def load_board():
    return assess.strict_json_loads(BOARD_PATH.read_bytes(), "test board")


def build_valid_result(board):
    rows = []
    arm_records = {
        "direct_qa": [],
        "whole_problem_work": [],
        "atomic_oracle_state": [],
        "source_scheduled": [],
    }
    family_atomic_totals = Counter()
    for row in board["rows"]:
        direct = call_record(
            assess.direct_prompt(row["question"]),
            "no integer here",
            assess.MAX_NEW_FULL,
        )
        direct.update(
            {
                "answer_segment": "no integer here",
                "predicted_answer": None,
                "correct": False,
            }
        )
        arm_records["direct_qa"].append(direct)

        whole = call_record(
            assess.whole_prompt(row["question"]),
            str(row["answer"]),
            assess.MAX_NEW_FULL,
        )
        whole.update(
            {
                "answer_segment": str(row["answer"]),
                "predicted_answer": row["answer"],
                "correct": True,
            }
        )
        arm_records["whole_problem_work"].append(whole)

        schedule = [(step[0], step[1]) for step in row["schedule"]]
        atomic = []
        state = row["initial_state"]
        for index, (operation, operand) in enumerate(schedule):
            expected = assess.apply_operation(state, operation, operand)
            record = call_record(
                assess.format_atomic_prompt(state, operation, operand),
                str(expected),
                assess.MAX_NEW_ATOMIC,
            )
            record.update(
                {
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": state,
                    "expected_state": expected,
                    "predicted_state": expected,
                    "correct": True,
                }
            )
            atomic.append(record)
            arm_records["atomic_oracle_state"].append(record)
            state = expected
            family_atomic_totals[row["family"]] += 1

        scheduled_steps = []
        state = row["initial_state"]
        for index, (operation, operand) in enumerate(schedule):
            predicted = assess.apply_operation(state, operation, operand)
            record = call_record(
                assess.format_atomic_prompt(state, operation, operand),
                str(predicted),
                assess.MAX_NEW_ATOMIC,
            )
            record.update(
                {
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": state,
                    "predicted_state": predicted,
                }
            )
            scheduled_steps.append(record)
            arm_records["source_scheduled"].append(record)
            state = predicted

        rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "stratum": row["stratum"],
                "question": row["question"],
                "answer": row["answer"],
                "direct": direct,
                "whole_problem_work": whole,
                "atomic_oracle_state": atomic,
                "source_scheduled": {
                    "predicted_answer": state,
                    "correct": True,
                    "steps": scheduled_steps,
                },
            }
        )

    by_family = {
        family: {
            "count": 64,
            "direct_correct": 0,
            "whole_correct": 64,
            "scheduled_correct": 64,
            "atomic_correct": family_atomic_totals[family],
            "atomic_total": family_atomic_totals[family],
        }
        for family in assess.FAMILIES
    }
    mcnemar = Fraction(1, 2**255)
    summary = {
        "case_count": 256,
        "transition_count": 704,
        "direct_correct": 0,
        "whole_correct": 256,
        "scheduled_correct": 256,
        "atomic_correct": 704,
        "atomic_total": 704,
        "scheduler_only_correct": 256,
        "direct_only_correct": 0,
        "mcnemar_exact_p": float(mcnemar),
        "by_family": by_family,
    }
    by_arm = {
        arm: {
            "model_calls": len(records),
            "prompt_token_count": sum(
                record["prompt_token_count"] for record in records
            ),
            "sampled_token_count": sum(
                record["sampled_token_count"] for record in records
            ),
            "decoded_token_count": sum(
                record["decoded_token_count"] for record in records
            ),
        }
        for arm, records in arm_records.items()
    }
    ledger = {
        "model_calls": sum(item["model_calls"] for item in by_arm.values()),
        "maximum_model_calls_without_parse_failures": 1920,
        "prompt_token_count": sum(
            item["prompt_token_count"] for item in by_arm.values()
        ),
        "sampled_token_count": sum(
            item["sampled_token_count"] for item in by_arm.values()
        ),
        "decoded_token_count": sum(
            item["decoded_token_count"] for item in by_arm.values()
        ),
        "by_arm": by_arm,
        "scheduled_parse_failure_chains": 0,
        "scheduled_calls_not_issued_after_parse_failure": 0,
        "early_stop_policy": "eos_or_frozen_token_cap_or_context_limit_only",
        "external_schedule_parser": True,
        "external_horner_schedule": True,
        "gold_intermediates_in_scheduled_arm": 0,
        "retries": 0,
        "repair_calls": 0,
        "search_calls": 0,
        "verifier_feedback_calls": 0,
    }
    gates = {key: True for key in assess.GATE_KEYS}
    integrity = {key: True for key in assess.INTEGRITY_KEYS}
    return {
        "schema": assess.RESULT_SCHEMA,
        "board": "frozen/source_scheduled_reasoning_confirmation_v1.json",
        "board_sha256": assess.EXPECTED_BOARD_SHA256,
        "cases_sha256": assess.EXPECTED_CASES_SHA256,
        "checkpoint_step": assess.EXPECTED_CHECKPOINT_STEP,
        "checkpoint_sha256": assess.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": assess.EXPECTED_TOKENIZER_SHA256,
        "implementation_sha256": dict(assess.EXPECTED_IMPLEMENTATION_SHA256),
        "device": "cuda",
        "max_new_full": assess.MAX_NEW_FULL,
        "max_new_atomic": assess.MAX_NEW_ATOMIC,
        "resource_ledger": ledger,
        "summary": summary,
        "gates": gates,
        "integrity_gates": integrity,
        "advance_to_internalization": True,
        "rows": rows,
    }


class IndependentAssessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = load_board()
        cls.valid = build_valid_result(cls.board)

    def test_valid_raw_records_recompute_exact_evidence(self):
        output = assess.assess_payload(self.valid, self.board, token_count)
        self.assertTrue(output["advance_to_internalization"])
        self.assertEqual(output["summary"]["scheduled_correct"], 256)
        self.assertEqual(output["summary"]["direct_correct"], 0)
        self.assertEqual(
            output["summary"]["mcnemar_exact_two_sided"],
            {"numerator": 1, "denominator": 2**255},
        )
        self.assertEqual(output["resource_ledger"]["model_calls"], 1920)

    def test_hardcoded_implementation_hashes_match_admitted_files(self):
        self.assertEqual(
            assess.hash_paths(assess.implementation_paths()),
            assess.EXPECTED_IMPLEMENTATION_SHA256,
        )
        self.assertTrue(
            all(
                path.is_relative_to(assess.FROZEN_IMPLEMENTATION_ROOT)
                for path in assess.implementation_paths().values()
            )
        )
        self.assertNotEqual(
            assess.implementation_paths()["model_loader"], ROOT / "train/model.py"
        )
        source = Path(assess.__file__).read_text()
        self.assertNotIn("import eval_scheduled_reasoning_confirmation", source)

    def test_rejects_self_rehashed_summary_and_boolean_tampering(self):
        attacks = []
        summary = copy.deepcopy(self.valid)
        summary["summary"]["scheduled_correct"] = 0
        summary["gates"]["scheduled_absolute"] = False
        summary["advance_to_internalization"] = False
        attacks.append(summary)

        row_boolean = copy.deepcopy(self.valid)
        row_boolean["rows"][0]["direct"]["correct"] = True
        attacks.append(row_boolean)

        advance = copy.deepcopy(self.valid)
        advance["advance_to_internalization"] = False
        attacks.append(advance)

        for attack in attacks:
            with self.subTest(kind=len(json.dumps(attack))):
                payload = (json.dumps(attack, sort_keys=True) + "\n").encode()
                self.assertEqual(len(hashlib.sha256(payload).hexdigest()), 64)
                with self.assertRaises(assess.AssessmentError):
                    assess.assess_payload(attack, self.board, token_count)

    def test_rejects_missing_calls(self):
        missing_atomic = copy.deepcopy(self.valid)
        missing_atomic["rows"][0]["atomic_oracle_state"].pop()
        with self.assertRaisesRegex(assess.AssessmentError, "atomic call count"):
            assess.assess_payload(missing_atomic, self.board, token_count)

        missing_scheduled = copy.deepcopy(self.valid)
        missing_scheduled["rows"][0]["source_scheduled"]["steps"].pop()
        with self.assertRaisesRegex(
            assess.AssessmentError, "ended without parse failure"
        ):
            assess.assess_payload(missing_scheduled, self.board, token_count)

    def test_rejects_wrong_prompt_even_with_recounted_tokens(self):
        tampered = copy.deepcopy(self.valid)
        call = tampered["rows"][0]["direct"]
        call["prompt"] += " altered"
        call["prompt_token_count"] = token_count(call["prompt"])
        call["untruncated_prompt_token_count"] = token_count(call["prompt"])
        with self.assertRaisesRegex(assess.AssessmentError, "wrong prompt renderer"):
            assess.assess_payload(tampered, self.board, token_count)

    def test_rejects_wrong_frozen_or_implementation_hashes(self):
        mutations = (
            ("board", lambda value: value.update(board_sha256="0" * 64)),
            ("checkpoint", lambda value: value.update(checkpoint_sha256="0" * 64)),
            ("tokenizer", lambda value: value.update(tokenizer_sha256="0" * 64)),
            (
                "evaluator",
                lambda value: value["implementation_sha256"].update(evaluator="0" * 64),
            ),
        )
        for name, mutate in mutations:
            tampered = copy.deepcopy(self.valid)
            mutate(tampered)
            with self.subTest(name=name), self.assertRaises(assess.AssessmentError):
                assess.assess_payload(tampered, self.board, token_count)

    def test_rejects_token_accounting_tampering(self):
        tampered = copy.deepcopy(self.valid)
        tampered["rows"][0]["direct"]["prompt_token_count"] += 1
        with self.assertRaisesRegex(assess.AssessmentError, "frozen tokenizer"):
            assess.assess_payload(tampered, self.board, token_count)

    def test_duplicate_json_keys_fail_closed(self):
        with self.assertRaisesRegex(assess.AssessmentError, "duplicate JSON key"):
            assess.strict_json_loads(b'{"x":1,"x":2}', "duplicate fixture")

    def test_immutable_writer_is_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assessment.json"
            digest = assess.write_immutable_json(path, {"ok": True})
            self.assertEqual(
                digest, hashlib.sha256(b'{\n  "ok": true\n}\n').hexdigest()
            )
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                assess.write_immutable_json(path, {"ok": True})


if __name__ == "__main__":
    unittest.main()
