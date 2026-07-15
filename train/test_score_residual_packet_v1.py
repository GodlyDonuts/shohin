#!/usr/bin/env python3
"""Independent replay and evidence tests for the RSP-C1 scorer."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import eval_residual_packet_v1 as evaluate
    import score_residual_packet_v1 as scorer
except ModuleNotFoundError:
    from train import eval_residual_packet_v1 as evaluate
    from train import score_residual_packet_v1 as scorer

from pipeline import generate_residual_packet_v1 as generate

protocol = scorer.ADMITTED_PROTOCOL


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class CharacterTokenizer:
    def encode(self, text):
        return Encoding([ord(character) + 1 for character in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token - 1) for token in ids)

    def token_to_id(self, token):
        return 0 if token == "<|endoftext|>" else None


class Endpoint:
    def __init__(self, name, digest, responder):
        self.name = name
        self.checkpoint_path = f"/{name}.pt"
        self.checkpoint_sha256 = digest
        self.checkpoint_step = 260000
        self.responder = responder

    def complete(self, prompt, max_new):
        response = self.responder(prompt)
        ids = [ord(character) + 1 for character in response]
        if len(ids) > max_new:
            raise AssertionError(f"test response exceeds cap: {len(ids)} > {max_new}")
        return {
            "response": response,
            "prompt_token_count": len(prompt),
            "sampled_token_ids": ids,
            "sampled_token_count": len(ids),
            "decoded_token_ids": ids,
            "decoded_token_count": len(ids),
            "stop_reason": "context_limit",
        }


def executor_response(prompt):
    import re

    patterns = (
        (r"Problem: Compute (-?\d+) plus (\d+)\.\nWork:\Z", lambda a, b: a + b),
        (r"Problem: Compute (-?\d+) times (\d+)\.\nWork:\Z", lambda a, b: a * b),
        (r"Problem: Compute (-?\d+) minus (\d+)\.\nWork:\Z", lambda a, b: a - b),
    )
    for pattern, function in patterns:
        match = re.fullmatch(pattern, prompt)
        if match:
            return str(function(int(match.group(1)), int(match.group(2))))
    raise AssertionError(f"unknown executor prompt: {prompt!r}")


def controller_responder(compiler_responses):
    def respond(prompt):
        if prompt.startswith("Problem: "):
            return compiler_responses[prompt]
        prefix = "Packet:\n"
        marker = "\nObserved result: "
        suffix = "\nNext packet:"
        packet, observed = prompt[len(prefix) : -len(suffix)].split(marker)
        return protocol.expected_update(packet, int(observed))

    return respond


class IndependentParserTests(unittest.TestCase):
    def test_frozen_hashes_and_exact_protocol_path(self):
        self.assertEqual(
            scorer.EXPECTED_BOARD_SHA256,
            "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7",
        )
        self.assertEqual(
            scorer.EXPECTED_BOARD_ROWS_SHA256,
            "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e",
        )
        self.assertEqual(
            scorer.EXPECTED_PROTOCOL_SHA256,
            "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2",
        )
        self.assertEqual(Path(protocol.__file__).resolve(), scorer.PROTOCOL_PATH)
        self.assertEqual(
            scorer.sha256_file(scorer.PROTOCOL_PATH),
            scorer.EXPECTED_PROTOCOL_SHA256,
        )

    def test_packet_and_executor_parsers_are_strict(self):
        packet = "State: 10\nPlan: add 2; multiply 3"
        self.assertEqual(
            scorer.parse_packet(packet),
            (10, (("add", 2), ("multiply", 3))),
        )
        self.assertIsNone(scorer.parse_packet(packet + "\u00a0"))
        self.assertIsNone(scorer.parse_packet("State: 010\nPlan: add 2"))
        self.assertEqual(scorer.parse_executor_result("step 1 then 9\n100"), 9)
        self.assertIsNone(scorer.parse_executor_result("value 1.5\n5"))

    def test_exact_mcnemar_preserves_rational_probability(self):
        result = scorer.exact_mcnemar(20, 2)
        self.assertEqual(result["numerator"], 127)
        self.assertEqual(result["denominator"], 2**20)
        self.assertAlmostEqual(result["p"], 508 / 2**22)
        self.assertEqual(scorer.exact_mcnemar(0, 0)["p"], 1.0)


class BoardAndEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_rows = generate.build_board_rows()
        cls.board = generate.board_payload(cls.raw_rows)
        cls.rows, cls.rows_hash = scorer.audit_board(cls.board)

    def test_board_is_independently_replayed(self):
        self.assertEqual(len(self.rows), 256)
        self.assertEqual(self.rows_hash, self.board["rows_sha256"])
        self.assertEqual(
            {row["stratum"] for row in self.rows}, set(scorer.STRATA)
        )

    def test_self_rehashed_arithmetic_tamper_is_rejected(self):
        board = copy.deepcopy(self.board)
        board["rows"][0]["answer"] += 1
        board["rows_sha256"] = hashlib.sha256(
            scorer.canonical_json_bytes(board["rows"])
        ).hexdigest()
        with self.assertRaisesRegex(scorer.EvidenceError, "renderer_ood_000"):
            scorer.audit_board(board)

    def test_manifest_hash_mismatch_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            board_path = directory / "board.json"
            board_path.write_text(json.dumps(self.board))
            missing = {"path": "unused", "sha256": "0" * 64}
            board_digest = hashlib.sha256(board_path.read_bytes()).hexdigest()
            protocol_path = Path(protocol.__file__).resolve()
            evaluator_path = Path(evaluate.__file__).resolve()
            manifest = {
                "schema": scorer.MANIFEST_SCHEMA,
                "frozen": True,
                "preregistration": {
                    "path": board_path.name,
                    "sha256": board_digest,
                },
                "protocol": {
                    "path": str(protocol_path),
                    "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                },
                "evaluator": {
                    "path": str(evaluator_path),
                    "sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
                },
                "board": {"path": board_path.name, "sha256": "f" * 64},
                "tokenizer": missing,
                "raw_executor_checkpoint": missing,
                "treatment_data": missing,
                "sham_data": missing,
                "training_manifest": missing,
                "admission_audit": missing,
                "prerequisite_confirmation": missing,
                "runs": [],
            }
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            result = scorer.score_manifest(manifest_path)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertIn("artifact_hash_mismatch", result["reasons"])

    def test_prerequisite_is_recomputed_from_raw_calls(self):
        confirmation_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts/evals/source_scheduled_reasoning_confirmation_v1.json"
        )
        board = json.loads(confirmation_path.read_text())

        def call(prompt, response, max_new):
            return {
                "prompt": prompt,
                "max_new": max_new,
                "response": response,
                "prompt_token_count": 1,
                "untruncated_prompt_token_count": 1,
                "prompt_truncated": False,
                "sampled_token_count": 1,
                "decoded_token_count": 1,
                "stop_reason": "context_limit",
            }

        result_rows = []
        calls = {arm: [] for arm in (
            "direct_qa",
            "whole_problem_work",
            "atomic_oracle_state",
            "source_scheduled",
        )}
        by_family = {}
        for family in ("multiply_subtract", "base_conversion", "sequential_state", "modular_update"):
            family_rows = [row for row in board["rows"] if row["family"] == family]
            atomic_total = sum(len(row["schedule"]) for row in family_rows)
            by_family[family] = {
                "count": 64,
                "direct_correct": 0,
                "whole_correct": 64,
                "scheduled_correct": 64,
                "atomic_correct": atomic_total,
                "atomic_total": atomic_total,
            }

        for row in board["rows"]:
            start, schedule = scorer._prerequisite_schedule(
                row["family"], row["question"]
            )
            direct = call(
                f"Question: {row['question']} Return only the final integer.\nAnswer:",
                "no integer",
                128,
            )
            whole = call(
                f"Problem: {row['question']}\nWork:", str(row["answer"]), 128
            )
            calls["direct_qa"].append(direct)
            calls["whole_problem_work"].append(whole)
            atomic = []
            scheduled = []
            state = start
            for operation, operand in schedule:
                input_state = state
                state = scorer._prerequisite_apply(state, operation, operand)
                atomic_call = call(
                    scorer._prerequisite_prompt(input_state, operation, operand),
                    str(state),
                    48,
                )
                atomic.append(atomic_call)
                calls["atomic_oracle_state"].append(atomic_call)
            state = start
            for operation, operand in schedule:
                scheduled_call = call(
                    scorer._prerequisite_prompt(state, operation, operand),
                    str(scorer._prerequisite_apply(state, operation, operand)),
                    48,
                )
                state = scorer._prerequisite_apply(state, operation, operand)
                scheduled.append(scheduled_call)
                calls["source_scheduled"].append(scheduled_call)
            result_rows.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "direct": direct,
                    "whole_problem_work": whole,
                    "atomic_oracle_state": atomic,
                    "source_scheduled": {"steps": scheduled},
                }
            )

        mc = scorer.exact_mcnemar(256, 0)
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
            "mcnemar_exact_p": mc["p"],
            "by_family": by_family,
        }
        gates = {
            "scheduled_absolute": True,
            "scheduled_advantage": True,
            "paired_significance": True,
            "family_nonregression": True,
            "sequential_absolute": True,
            "atomic_ceiling": True,
        }
        prerequisite = {
            "board_sha256": scorer.PREREQUISITE_BOARD_SHA256,
            "cases_sha256": scorer.PREREQUISITE_CASES_SHA256,
            "checkpoint_sha256": scorer.RAW_EXECUTOR_SHA256,
            "tokenizer_sha256": scorer.TOKENIZER_SHA256,
            "rows": result_rows,
            "summary": summary,
            "gates": gates,
            "resource_ledger": {
                "by_arm": {
                    arm: {
                        "model_calls": len(records),
                        "prompt_token_count": len(records),
                        "sampled_token_count": len(records),
                        "decoded_token_count": len(records),
                    }
                    for arm, records in calls.items()
                }
            },
            "integrity_gates": {"raw_transcripts_replayed": True},
            "advance_to_internalization": True,
        }
        verified = scorer._prerequisite_summary(prerequisite)
        self.assertTrue(verified["passed"])
        tampered = copy.deepcopy(prerequisite)
        tampered["rows"][0]["source_scheduled"]["steps"][0]["response"] = "bad"
        with self.assertRaises(scorer.EvidenceError):
            scorer._prerequisite_summary(tampered)


class FullReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_rows = generate.build_board_rows()
        board = generate.board_payload(raw_rows)
        cls.rows, _ = scorer.audit_board(board)
        compiler_responses = {
            protocol.compiler_prompt(row["source"]): row["packet"] for row in raw_rows
        }
        controller = controller_responder(compiler_responses)
        treatment_hash = "1" * 64
        sham_hash = "2" * 64
        executor_hash = "3" * 64
        treatment = Endpoint("treatment", treatment_hash, controller)
        sham = Endpoint("sham", sham_hash, controller)
        executor = Endpoint("raw_260k_executor", executor_hash, executor_response)
        cls.hashes = {
            "board": "4" * 64,
            "tokenizer": "5" * 64,
            "raw_260k_executor": executor_hash,
            "treatment_checkpoint": treatment_hash,
            "sham_checkpoint": sham_hash,
        }
        cls.checkpoint_hashes = {
            "treatment": treatment_hash,
            "sham": sham_hash,
            "raw_260k_executor": executor_hash,
        }
        cls.tokenizer = CharacterTokenizer()
        cls.transcript = evaluate.acquire_transcript(
            board,
            raw_rows,
            treatment,
            sham,
            executor,
            scorer.FIT_SEEDS[0],
            cls.hashes,
        )

    def test_full_raw_transcript_replays_without_trusting_metrics(self):
        result = scorer.score_transcript(
            self.transcript,
            self.rows,
            self.tokenizer,
            scorer.FIT_SEEDS[0],
            self.hashes,
            self.checkpoint_hashes,
        )
        treatment = result["arms"]["treatment"]
        self.assertEqual(result["external_scheduler_gold"], 256)
        self.assertEqual(treatment["compile_exact"], 256)
        self.assertEqual(treatment["strict_closed_loop"], 256)
        self.assertEqual(treatment["oracle_packet_loop"], 256)
        self.assertEqual(treatment["packet_swap_follow"], 64)
        self.assertTrue(result["gates"]["packet_swap"])
        self.assertFalse(result["gates"]["treatment_sham_compilation_gap"])
        self.assertFalse(result["gates"]["treatment_sham_strict_gap"])
        self.assertFalse(result["passed"])

    def test_call_token_tamper_is_rejected(self):
        transcript = copy.deepcopy(self.transcript)
        call = transcript["external_scheduler"][0]["runtime"]["steps"][0]
        call["prompt_token_count"] += 1
        with self.assertRaises(scorer.EvidenceError) as captured:
            scorer.score_transcript(
                transcript,
                self.rows,
                self.tokenizer,
                scorer.FIT_SEEDS[0],
                self.hashes,
                self.checkpoint_hashes,
            )
        self.assertEqual(captured.exception.code, "prompt_token_mismatch")

    def test_any_raw_boolean_is_rejected(self):
        transcript = copy.deepcopy(self.transcript)
        transcript["trusted_success"] = True
        with self.assertRaisesRegex(scorer.EvidenceError, "trusted_success"):
            scorer.score_transcript(
                transcript,
                self.rows,
                self.tokenizer,
                scorer.FIT_SEEDS[0],
                self.hashes,
                self.checkpoint_hashes,
            )


if __name__ == "__main__":
    unittest.main()
