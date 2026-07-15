#!/usr/bin/env python3
"""Pure tests for the frozen raw-260k updater likelihood diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import probe_updater_candidate_likelihood as probe
except ModuleNotFoundError:
    from train import probe_updater_candidate_likelihood as probe


class FrozenCandidateContractTests(unittest.TestCase):
    def test_source_and_candidate_contract_is_exact(self):
        self.assertEqual(len(probe.EXPECTED_SOURCE_PROMPTS), 12)
        self.assertEqual(
            [row.id for row in probe.EXPECTED_SOURCE_PROMPTS],
            [
                "copy_25",
                "copy_75",
                "copy_64",
                "delete_a",
                "delete_b",
                "delete_c",
                "joint_a",
                "joint_b",
                "joint_c",
                "packet_a",
                "packet_b",
                "packet_c",
            ],
        )
        self.assertEqual(sum(row.prompt_token_count for row in probe.EXPECTED_SOURCE_PROMPTS), 368)
        self.assertEqual(len(probe.FROZEN_CASES), 6)
        self.assertEqual(
            [case.id for case in probe.FROZEN_CASES],
            ["joint_a", "joint_b", "joint_c", "packet_a", "packet_b", "packet_c"],
        )
        for case in probe.FROZEN_CASES:
            self.assertEqual(
                tuple(candidate.candidate_class for candidate in case.candidates),
                probe.CANDIDATE_CLASS_ORDER,
            )
            self.assertEqual(len({candidate.text for candidate in case.candidates}), 5)
            for candidate in case.candidates:
                self.assertTrue(candidate.text.startswith("\n"))
                self.assertFalse(candidate.text.endswith("\n"))

        self.assertEqual(
            probe.FROZEN_CASES[0].candidates[0].text,
            "\nState: 25\nPlan: multiply 3; subtract 11",
        )
        self.assertEqual(
            probe.FROZEN_CASES[1].candidates[2].text,
            "\n27 * 4 = 108",
        )
        self.assertEqual(
            probe.FROZEN_CASES[2].candidates[3].text,
            "\n37 + 9 = 46\n46 * 2 = 92",
        )
        self.assertEqual(
            probe.candidate_manifest_sha256(),
            probe.EXPECTED_CANDIDATE_MANIFEST_SHA256,
        )
        self.assertEqual(
            probe.verify_candidate_manifest(),
            probe.EXPECTED_CANDIDATE_MANIFEST_SHA256,
        )

    def test_real_source_artifact_is_read_only_and_bound(self):
        value, digest = probe.load_bound_source(probe.DEFAULT_SOURCE)
        self.assertEqual(digest, probe.EXPECTED_SOURCE_SHA256)
        self.assertEqual(value["checkpoint_sha256"], probe.EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(value["tokenizer_sha256"], probe.EXPECTED_TOKENIZER_SHA256)
        self.assertEqual(value["checkpoint_step"], 260000)

    def test_bound_tokenized_manifest_and_exact_token_ledger(self):
        source, _ = probe.load_bound_source(probe.DEFAULT_SOURCE)
        tokenizer = probe.Tokenizer.from_file(
            str(probe.ROOT / "artifacts/shohin-tok-32k.json")
        )
        prepared, eos_token_id = probe.prepare_cases(source, tokenizer)
        self.assertEqual(eos_token_id, 0)
        self.assertEqual(
            probe.verify_tokenized_manifest(prepared, eos_token_id),
            probe.EXPECTED_TOKENIZED_MANIFEST_SHA256,
        )
        self.assertEqual(
            sum(
                len(candidate.token_ids)
                for case in prepared
                for candidate in case.candidates
            ),
            518,
        )
        self.assertEqual(
            sum(len(case.prompt_token_ids) * len(case.candidates) for case in prepared),
            1080,
        )

    def test_source_byte_tamper_and_binding_tamper_are_rejected(self):
        source_payload = probe.DEFAULT_SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "source.json"
            copied.write_bytes(source_payload)
            os.chmod(copied, 0o444)
            probe.load_bound_source(copied)

            os.chmod(copied, 0o600)
            tampered = source_payload.replace(b'"device": "mps"', b'"device": "cpu"')
            self.assertNotEqual(tampered, source_payload)
            copied.write_bytes(tampered)
            os.chmod(copied, 0o444)
            with self.assertRaisesRegex(ValueError, "source artifact SHA-256 mismatch"):
                probe.load_bound_source(copied)
            os.chmod(copied, 0o600)

        value = json.loads(source_payload)
        value["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256 binding differs"):
            probe.validate_source_artifact(value)

    def test_hash_helper_rejects_file_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bound.bin"
            path.write_bytes(b"frozen\n")
            expected = hashlib.sha256(b"frozen\n").hexdigest()
            self.assertEqual(probe.require_file_sha256(path, expected, "toy"), expected)
            path.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "toy SHA-256 mismatch"):
                probe.require_file_sha256(path, expected, "toy")


def make_toy_prepared_cases() -> tuple[probe.PreparedCase, ...]:
    prepared = []
    for case_index, case in enumerate(probe.FROZEN_CASES):
        candidates = []
        for candidate_index, candidate in enumerate(case.candidates):
            token_count = 2
            if case.id == "joint_b" and candidate.candidate_class == probe.CORRECT_CLASS:
                token_count = 3
            elif case.id == "joint_b" and candidate.candidate_class == "unchanged_source_packet":
                token_count = 1
            token_ids = tuple(
                1000 + case_index * 100 + candidate_index * 10 + offset
                for offset in range(token_count)
            )
            candidates.append(
                probe.PreparedCandidate(
                    candidate.candidate_class,
                    candidate.text,
                    token_ids,
                )
            )
        prepared.append(
            probe.PreparedCase(
                case.id,
                case.kind,
                f"toy prompt {case.id}",
                (10, 11, 12),
                tuple(candidates),
                0,
            )
        )
    return tuple(prepared)


def toy_score(
    case: probe.PreparedCase, candidate: probe.PreparedCandidate
) -> probe.TeacherForcedScore:
    correct = candidate.candidate_class == probe.CORRECT_CLASS
    mean = -0.1 if correct else -1.0
    eos_log_likelihood = -0.05 if correct else -1.0
    eos_rank = 1
    eos_unique = True
    top_id = case.eos_token_id
    top_log_likelihood = eos_log_likelihood

    if case.id == "joint_a" and correct:
        eos_log_likelihood = -2.0
        eos_rank = 3
        eos_unique = False
        top_id = 99
        top_log_likelihood = -0.1
    elif case.id == "joint_b":
        if correct:
            mean = -0.2
        elif candidate.candidate_class == "unchanged_source_packet":
            mean = -0.5
    elif case.id == "joint_c":
        if correct:
            mean = -1.0
        elif candidate.candidate_class == "reordered_tail":
            mean = -0.1
    elif case.id == "packet_b":
        if correct:
            mean = -0.1
            eos_log_likelihood = -3.0
            top_log_likelihood = -3.0
        elif candidate.candidate_class == "unchanged_source_packet":
            mean = -0.15
            eos_log_likelihood = -0.01
            top_log_likelihood = -0.01

    return probe.TeacherForcedScore(
        candidate_token_log_likelihoods=tuple(
            mean for _ in candidate.token_ids
        ),
        eos_log_likelihood=eos_log_likelihood,
        eos_rank=eos_rank,
        eos_is_unique_top1=eos_unique,
        next_token_top_id=top_id,
        next_token_top_log_likelihood=top_log_likelihood,
    )


class ToyScorerDecisionTests(unittest.TestCase):
    def test_toy_scorer_separates_preference_decoding_and_termination(self):
        result = probe.score_prepared_board(make_toy_prepared_cases(), toy_score)
        diagnoses = {row["id"]: row["diagnosis"] for row in result["rows"]}
        self.assertEqual(diagnoses["joint_a"], probe.DIAGNOSIS_TERMINATION)
        self.assertEqual(diagnoses["joint_b"], probe.DIAGNOSIS_DECODING)
        self.assertEqual(diagnoses["joint_c"], probe.DIAGNOSIS_NOT_PREFERRED)
        self.assertEqual(diagnoses["packet_a"], probe.DIAGNOSIS_PREFERRED)
        self.assertEqual(diagnoses["packet_b"], probe.DIAGNOSIS_JOINT)
        self.assertEqual(diagnoses["packet_c"], probe.DIAGNOSIS_PREFERRED)
        self.assertEqual(
            result["summary"]["overall_diagnosis"],
            "correct_update_not_consistently_preferred",
        )
        self.assertEqual(result["resource_ledger"]["candidate_sequence_evaluations"], 30)
        self.assertEqual(result["resource_ledger"]["model_forward_calls"], 30)
        self.assertEqual(result["resource_ledger"]["generated_tokens"], 0)
        self.assertEqual(result["resource_ledger"]["retries"], 0)
        self.assertEqual(result["resource_ledger"]["candidate_searches"], 0)

    def test_all_normalized_wins_with_bad_eos_reports_latent_termination_loss(self):
        def termination_scorer(case, candidate):
            correct = candidate.candidate_class == probe.CORRECT_CLASS
            mean = -0.1 if correct else -2.0
            eos = -2.0 if correct else -1.0
            return probe.TeacherForcedScore(
                candidate_token_log_likelihoods=tuple(mean for _ in candidate.token_ids),
                eos_log_likelihood=eos,
                eos_rank=2 if correct else 1,
                eos_is_unique_top1=not correct,
                next_token_top_id=99 if correct else case.eos_token_id,
                next_token_top_log_likelihood=-0.1 if correct else eos,
            )

        result = probe.score_prepared_board(
            make_toy_prepared_cases(), termination_scorer
        )
        self.assertEqual(
            result["summary"]["overall_diagnosis"],
            "correct_update_likelihood_preferred_but_decoding_or_termination_loses",
        )
        self.assertTrue(
            all(row["diagnosis"] == probe.DIAGNOSIS_TERMINATION for row in result["rows"])
        )

    def test_eos_tied_for_top_is_a_termination_loss_not_an_error(self):
        def tied_eos_scorer(case, candidate):
            correct = candidate.candidate_class == probe.CORRECT_CLASS
            mean = -0.1 if correct else -2.0
            eos = -0.5 if correct else -1.0
            return probe.TeacherForcedScore(
                candidate_token_log_likelihoods=tuple(mean for _ in candidate.token_ids),
                eos_log_likelihood=eos,
                eos_rank=1,
                eos_is_unique_top1=not correct,
                next_token_top_id=99 if correct else case.eos_token_id,
                next_token_top_log_likelihood=eos,
            )

        result = probe.score_prepared_board(
            make_toy_prepared_cases(), tied_eos_scorer
        )
        self.assertTrue(
            all(row["diagnosis"] == probe.DIAGNOSIS_TERMINATION for row in result["rows"])
        )

    def test_nonfinite_toy_score_is_rejected(self):
        def invalid_scorer(case, candidate):
            return probe.TeacherForcedScore(
                candidate_token_log_likelihoods=tuple(
                    float("nan") for _ in candidate.token_ids
                ),
                eos_log_likelihood=-1.0,
                eos_rank=1,
                eos_is_unique_top1=True,
                next_token_top_id=case.eos_token_id,
                next_token_top_log_likelihood=-1.0,
            )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            probe.score_prepared_board(make_toy_prepared_cases(), invalid_scorer)


class TeacherForcedAlignmentTests(unittest.TestCase):
    def test_candidate_and_immediate_eos_positions_are_aligned(self):
        class ToyModel:
            def __init__(self):
                self.logits = None

            def __call__(self, tokens):
                self.asserted_tokens = tokens.detach().cpu().tolist()
                logits = probe.torch.zeros(
                    (1, tokens.shape[1], 8),
                    dtype=probe.torch.float32,
                    device=tokens.device,
                )
                logits[0, 1, 3] = 5.0
                logits[0, 2, 4] = 4.0
                logits[0, 3, 0] = 6.0
                self.logits = logits.detach().cpu()
                return logits, None

        candidate = probe.PreparedCandidate("toy", "xy", (3, 4))
        case = probe.PreparedCase(
            "toy",
            "toy",
            "prompt",
            (1, 2),
            (candidate,),
            0,
        )
        model = ToyModel()
        score = probe.teacher_forced_score(model, case, candidate, "cpu")
        self.assertEqual(model.asserted_tokens, [[1, 2, 3, 4]])
        expected_first = float(
            probe.torch.log_softmax(model.logits[0, 1], dim=-1)[3].item()
        )
        expected_second = float(
            probe.torch.log_softmax(model.logits[0, 2], dim=-1)[4].item()
        )
        expected_eos = float(
            probe.torch.log_softmax(model.logits[0, 3], dim=-1)[0].item()
        )
        self.assertAlmostEqual(score.candidate_token_log_likelihoods[0], expected_first)
        self.assertAlmostEqual(score.candidate_token_log_likelihoods[1], expected_second)
        self.assertAlmostEqual(score.eos_log_likelihood, expected_eos)
        self.assertEqual(score.eos_rank, 1)
        self.assertTrue(score.eos_is_unique_top1)
        self.assertEqual(score.next_token_top_id, 0)


class ImmutableOutputTests(unittest.TestCase):
    def test_output_is_exclusive_read_only_and_ascii_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            digest = probe.write_immutable_json(
                path,
                {"schema": probe.SCHEMA, "status": "toy"},
            )
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            path.read_bytes().decode("ascii")
            with self.assertRaises(FileExistsError):
                probe.write_immutable_json(path, {"schema": "replacement"})
            os.chmod(path, 0o600)


if __name__ == "__main__":
    unittest.main()
