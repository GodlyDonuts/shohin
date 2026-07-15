#!/usr/bin/env python3
"""Focused tests for raw RSP-C1 transcript acquisition."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

try:
    import eval_residual_packet_v1 as evaluate
except ModuleNotFoundError:
    from train import eval_residual_packet_v1 as evaluate

protocol = evaluate.PROTOCOL


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
    def __init__(self, name, responder):
        self.name = name
        self.responder = responder
        self.checkpoint_path = f"/{name}.pt"
        self.checkpoint_sha256 = name[0] * 64
        self.checkpoint_step = 260000

    def complete(self, prompt, max_new):
        response = self.responder(prompt)
        decoded = [ord(character) + 1 for character in response]
        return {
            "response": response,
            "prompt_token_count": len(prompt),
            "sampled_token_ids": decoded,
            "sampled_token_count": len(decoded),
            "decoded_token_ids": decoded,
            "decoded_token_count": len(decoded),
            "stop_reason": "context_limit",
        }


def arithmetic_responder(prompt):
    matchers = (
        (r"Problem: Compute (-?\d+) plus (\d+)\.\nWork:\Z", lambda a, b: a + b),
        (r"Problem: Compute (-?\d+) times (\d+)\.\nWork:\Z", lambda a, b: a * b),
        (r"Problem: Compute (-?\d+) minus (\d+)\.\nWork:\Z", lambda a, b: a - b),
    )
    import re

    for pattern, operation in matchers:
        match = re.fullmatch(pattern, prompt)
        if match:
            return str(operation(int(match.group(1)), int(match.group(2))))
    raise AssertionError(f"unexpected executor prompt: {prompt!r}")


def updater_responder(prompt):
    prefix = "Packet:\n"
    marker = "\nObserved result: "
    suffix = "\nNext packet:"
    if not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise AssertionError(f"unexpected updater prompt: {prompt!r}")
    packet, observed = prompt[len(prefix) : -len(suffix)].split(marker)
    return protocol.expected_update(packet, int(observed))


class PacketGrammarTests(unittest.TestCase):
    def test_frozen_hashes_and_exact_protocol_path(self):
        self.assertEqual(
            evaluate.EXPECTED_BOARD_SHA256,
            "ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7",
        )
        self.assertEqual(
            evaluate.EXPECTED_BOARD_ROWS_SHA256,
            "fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e",
        )
        self.assertEqual(
            evaluate.EXPECTED_PROTOCOL_SHA256,
            "e011e8389d51188553d9fb0392ec892a5107249cc23c82cdba4df216e6db2ce2",
        )
        self.assertEqual(
            Path(protocol.__file__).resolve(), evaluate.PROTOCOL_PATH
        )
        self.assertEqual(
            evaluate.sha256_file(evaluate.PROTOCOL_PATH),
            evaluate.EXPECTED_PROTOCOL_SHA256,
        )

    def test_parser_matches_shared_strict_grammar(self):
        packet = "State: -10\nPlan: add 2; multiply 3"
        self.assertEqual(
            evaluate.parse_packet(" \t" + packet + "\r\n"),
            (-10, (("add", 2), ("multiply", 3))),
        )
        malformed = (
            "State: 01\nPlan: add 2",
            "State: 1\r\nPlan: add 2",
            "State: 1\nPlan: add 2\nExtra: x",
            "State: 1\nPlan: add 2\u00a0",
            "Answer: 3",
        )
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(evaluate.parse_packet(text))

    def test_renderers_are_byte_identical_to_protocol(self):
        plan = (("add", 2), ("multiply", 3))
        packet = protocol.canonical_packet(10, plan)
        self.assertEqual(evaluate.format_packet(10, plan), packet)
        self.assertEqual(evaluate.compiler_prompt("source"), protocol.compiler_prompt("source"))
        self.assertEqual(evaluate.updater_prompt(packet, 12), protocol.update_prompt(packet, 12))
        self.assertEqual(
            evaluate.executor_prompt(10, "add", 2),
            protocol.format_atomic_prompt(10, "add", 2),
        )


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.executor = Endpoint("raw_260k_executor", arithmetic_responder)
        self.controller = Endpoint("treatment", updater_responder)

    def test_source_blind_loop_executes_and_halts_exactly(self):
        recorder = evaluate.CallRecorder()
        result = evaluate.run_source_blind_loop(
            "State: 10\nPlan: add 2; multiply 3",
            self.controller,
            self.executor,
            recorder,
            "oracle_packet_loop",
        )
        self.assertEqual(result["termination"], "answer")
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(
            result["steps"][1]["executor"]["prompt"],
            "Problem: Compute 12 times 3.\nWork:",
        )
        self.assertEqual([record["call_index"] for record in recorder.records], [0, 1, 2, 3])

    def test_invalid_update_stops_without_later_calls(self):
        controller = Endpoint("treatment", lambda prompt: "State: 12\nPlan: add 2; multiply 3")
        recorder = evaluate.CallRecorder()
        result = evaluate.run_source_blind_loop(
            "State: 10\nPlan: add 2; multiply 3",
            controller,
            self.executor,
            recorder,
            "strict_closed_loop",
        )
        self.assertEqual(result["termination"], "updater_output_invalid")
        self.assertEqual(len(recorder.records), 2)

    def test_external_scheduler_carries_only_executor_output(self):
        responses = iter(("999", "1001"))
        executor = Endpoint("raw_260k_executor", lambda prompt: next(responses))
        recorder = evaluate.CallRecorder()
        runtime = evaluate.run_external_scheduler(
            10, (("add", 2), ("subtract", 3)), executor, recorder
        )
        self.assertEqual(runtime["termination"], "complete")
        self.assertEqual(
            recorder.records[1]["prompt"],
            "Problem: Compute 999 minus 3.\nWork:",
        )

    def test_compiled_wrapper_deletes_source_before_blind_boundary(self):
        compiler = Endpoint("treatment", lambda prompt: "State: 10\nPlan: add 2")
        recorder = evaluate.CallRecorder()
        captured = {}

        def blind(packet, controller, executor, recorder, arm):
            captured["packet"] = packet
            captured["arguments"] = (controller, executor, recorder, arm)
            return {"termination": "initial_packet_invalid", "steps": []}

        with mock.patch.object(evaluate, "run_source_blind_loop", side_effect=blind):
            result = evaluate.run_compiled_case(
                "secret source", 1, compiler, self.executor, recorder
            )
        self.assertEqual(captured["packet"], "State: 10\nPlan: add 2")
        self.assertNotIn("secret source", repr(captured["arguments"]))
        self.assertEqual(result["compiler"]["prompt"], evaluate.compiler_prompt("secret source"))


class DecodeAndDiagnosticsTests(unittest.TestCase):
    def test_greedy_completion_starts_with_fresh_cache_every_call(self):
        class FakeModel:
            cfg = SimpleNamespace(seq_len=32)

            def __init__(self):
                self.prefill_cache_arguments = []
                self.tokens = iter((1, 0, 2, 0))

            def __call__(self, inputs, **kwargs):
                if inputs.shape[1] > 1:
                    self.prefill_cache_arguments.append(kwargs.get("cache"))
                token = next(self.tokens)
                logits = torch.full((1, inputs.shape[1], 4), -1000.0)
                logits[:, -1, token] = 1.0
                return logits, object()

        tokenizer = CharacterTokenizer()
        model = FakeModel()
        first = evaluate.greedy_completion(model, tokenizer, "ab", "cpu", 4)
        second = evaluate.greedy_completion(model, tokenizer, "cd", "cpu", 4)
        self.assertEqual(first["stop_reason"], "eos")
        self.assertEqual(second["stop_reason"], "eos")
        self.assertEqual(model.prefill_cache_arguments, [None, None])

    def test_teacher_cases_are_false_arithmetic_and_swaps_are_exactly_64(self):
        rows = []
        for stratum in evaluate.STRATA:
            for index in range(64):
                rows.append(
                    {
                        "id": f"{stratum}_{index:03d}",
                        "stratum": stratum,
                        "source": f"source {stratum} {index}",
                        "initial_state": 10 + index,
                        "operations": [["add", 2], ["multiply", 3], ["subtract", 4]],
                        "answer": 10000 + len(rows),
                    }
                )
        cases = evaluate.build_teacher_forced_cases(rows)
        self.assertEqual(len(cases), 256 * 3)
        by_id = {row["id"]: row for row in rows}
        for case in cases:
            state, plan = evaluate.parse_packet(case["packet"])
            operation, operand = plan[0]
            self.assertNotEqual(
                case["observed"], evaluate._apply(state, operation, operand)
            )
            self.assertNotIn(case["observed"], {row["answer"] for row in rows})
            self.assertEqual(len(plan), 3 - case["step_index"])
            self.assertIn(case["id"], by_id)
        swaps = evaluate.build_packet_swaps(rows)
        self.assertEqual(len(swaps), 64)
        self.assertTrue(all(original["id"] != donor["id"] for original, donor in swaps))

    def test_raw_output_rejects_booleans_and_is_immutable(self):
        with self.assertRaisesRegex(ValueError, "forbidden boolean"):
            evaluate.assert_no_booleans({"trusted_success": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            digest = evaluate.write_immutable_json(path, {"schema": "raw", "count": 0})
            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                evaluate.write_immutable_json(path, {"schema": "raw", "count": 0})
            os.chmod(path, 0o600)


if __name__ == "__main__":
    unittest.main()
