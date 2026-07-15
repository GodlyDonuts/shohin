#!/usr/bin/env python3
"""Frozen teacher-forced likelihood diagnostic for the raw-260k updater.

The probe reads the existing twelve-prompt updater transcript, verifies its
byte hash and bound checkpoint/tokenizer hashes, then scores five predeclared
candidate continuations on only the six joint/packet prompts. It performs no
generation, candidate construction from model output, search, retry, or
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from model import GPT, GPTConfig
except ModuleNotFoundError:  # Allows `python3 -m unittest train.test_...`.
    from train.model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts/eval_history/raw260k_updater_subskill_probe_20260715_mps.json"
)

SCHEMA = "raw260k_updater_candidate_likelihood_v1"
SOURCE_SCHEMA = "raw260k_updater_subskill_probe_v1"
CANDIDATE_MANIFEST_SCHEMA = "raw260k_updater_candidate_manifest_v1"
TOKENIZED_MANIFEST_SCHEMA = "raw260k_updater_tokenized_candidate_manifest_v1"

EXPECTED_SOURCE_SHA256 = (
    "4505602994a0e337b99359e580a6f2f04fad4d365b2dac59f4c339fac13a7593"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
)
EXPECTED_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
EXPECTED_CHECKPOINT_STEP = 260000
EXPECTED_CONTEXT_LENGTH = 2048
EXPECTED_SOURCE_MODEL_CALLS = 12
EOS_TOKEN = "<|endoftext|>"

# These two digests are mirrored in R12_UPDATER_CANDIDATE_LIKELIHOOD_PREREG.md.
# They bind the literal strings below and their tokenization before model load.
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "0e01fc54abfe63dcfd063fa6d5a1e4ed46b57aef617580d2cc286db839b3ba98"
)
EXPECTED_TOKENIZED_MANIFEST_SHA256 = (
    "13528bacb21d8bca006b434283830d9a0790225dd9a7ddcdcbc8f02e7bac8a99"
)

CORRECT_CLASS = "correct_residual_packet_state_tail"
CANDIDATE_CLASS_ORDER = (
    CORRECT_CLASS,
    "unchanged_source_packet",
    "consumed_head_replay",
    "arithmetic_execution_continuation",
    "reordered_tail",
)

DIAGNOSIS_NOT_PREFERRED = "correct_update_not_preferred"
DIAGNOSIS_DECODING = "correct_update_latent_but_loses_sequence_decoding"
DIAGNOSIS_TERMINATION = "correct_update_latent_but_loses_termination"
DIAGNOSIS_JOINT = "correct_update_latent_but_loses_joint_eos_likelihood"
DIAGNOSIS_PREFERRED = "correct_update_and_termination_preferred"


@dataclass(frozen=True)
class SourcePromptSpec:
    id: str
    kind: str
    prompt: str
    prompt_token_count: int


@dataclass(frozen=True)
class CandidateSpec:
    candidate_class: str
    text: str


@dataclass(frozen=True)
class FrozenCaseSpec:
    id: str
    kind: str
    candidates: tuple[CandidateSpec, ...]


@dataclass(frozen=True)
class PreparedCandidate:
    candidate_class: str
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCase:
    id: str
    kind: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    candidates: tuple[PreparedCandidate, ...]
    eos_token_id: int


@dataclass(frozen=True)
class TeacherForcedScore:
    candidate_token_log_likelihoods: tuple[float, ...]
    eos_log_likelihood: float
    eos_rank: int
    eos_is_unique_top1: bool
    next_token_top_id: int
    next_token_top_log_likelihood: float


EXPECTED_SOURCE_PROMPTS = (
    SourcePromptSpec(
        "copy_25",
        "copy_state",
        "Problem: The observed result is 25. Write the new state.\nWork:",
        18,
    ),
    SourcePromptSpec(
        "copy_75",
        "copy_state",
        "Problem: The observed result is 75. Write the new state.\nWork:",
        18,
    ),
    SourcePromptSpec(
        "copy_64",
        "copy_state",
        "Problem: The observed result is 64. Write the new state.\nWork:",
        18,
    ),
    SourcePromptSpec(
        "delete_a",
        "delete_head",
        "Problem: The plan is add 6; multiply 3; subtract 11. The first operation "
        "is complete. Write only the remaining operations.\nWork:",
        33,
    ),
    SourcePromptSpec(
        "delete_b",
        "delete_head",
        "Problem: The plan is multiply 4; subtract 13; add 8. The first operation "
        "is complete. Write only the remaining operations.\nWork:",
        33,
    ),
    SourcePromptSpec(
        "delete_c",
        "delete_head",
        "Problem: The plan is subtract 5; add 9; multiply 2. The first operation "
        "is complete. Write only the remaining operations.\nWork:",
        32,
    ),
    SourcePromptSpec(
        "joint_a",
        "joint_natural",
        "Problem: The plan was add 6; multiply 3; subtract 11. The first operation "
        "produced 25. Write the new state and remaining plan.\nWork:",
        37,
    ),
    SourcePromptSpec(
        "joint_b",
        "joint_natural",
        "Problem: The plan was multiply 4; subtract 13; add 8. The first operation "
        "produced 108. Write the new state and remaining plan.\nWork:",
        38,
    ),
    SourcePromptSpec(
        "joint_c",
        "joint_natural",
        "Problem: The plan was subtract 5; add 9; multiply 2. The first operation "
        "produced 37. Write the new state and remaining plan.\nWork:",
        36,
    ),
    SourcePromptSpec(
        "packet_a",
        "joint_packet",
        "Packet:\nState: 19\nPlan: add 6; multiply 3; subtract 11\n"
        "Observed result: 25\nNext packet:",
        35,
    ),
    SourcePromptSpec(
        "packet_b",
        "joint_packet",
        "Packet:\nState: 27\nPlan: multiply 4; subtract 13; add 8\n"
        "Observed result: 108\nNext packet:",
        36,
    ),
    SourcePromptSpec(
        "packet_c",
        "joint_packet",
        "Packet:\nState: 42\nPlan: subtract 5; add 9; multiply 2\n"
        "Observed result: 37\nNext packet:",
        34,
    ),
)


def _candidate_set(
    correct: str,
    unchanged: str,
    consumed_replay: str,
    arithmetic_continuation: str,
    reordered: str,
) -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(CORRECT_CLASS, correct),
        CandidateSpec("unchanged_source_packet", unchanged),
        CandidateSpec("consumed_head_replay", consumed_replay),
        CandidateSpec("arithmetic_execution_continuation", arithmetic_continuation),
        CandidateSpec("reordered_tail", reordered),
    )


# Literal candidate bytes. Do not derive these with arithmetic or packet helpers.
_CANDIDATES_A = _candidate_set(
    "\nState: 25\nPlan: multiply 3; subtract 11",
    "\nState: 19\nPlan: add 6; multiply 3; subtract 11",
    "\n19 + 6 = 25",
    "\n25 * 3 = 75\n75 - 11 = 64",
    "\nState: 25\nPlan: subtract 11; multiply 3",
)
_CANDIDATES_B = _candidate_set(
    "\nState: 108\nPlan: subtract 13; add 8",
    "\nState: 27\nPlan: multiply 4; subtract 13; add 8",
    "\n27 * 4 = 108",
    "\n108 - 13 = 95\n95 + 8 = 103",
    "\nState: 108\nPlan: add 8; subtract 13",
)
_CANDIDATES_C = _candidate_set(
    "\nState: 37\nPlan: add 9; multiply 2",
    "\nState: 42\nPlan: subtract 5; add 9; multiply 2",
    "\n42 - 5 = 37",
    "\n37 + 9 = 46\n46 * 2 = 92",
    "\nState: 37\nPlan: multiply 2; add 9",
)

FROZEN_CASES = (
    FrozenCaseSpec("joint_a", "joint_natural", _CANDIDATES_A),
    FrozenCaseSpec("joint_b", "joint_natural", _CANDIDATES_B),
    FrozenCaseSpec("joint_c", "joint_natural", _CANDIDATES_C),
    FrozenCaseSpec("packet_a", "joint_packet", _CANDIDATES_A),
    FrozenCaseSpec("packet_b", "joint_packet", _CANDIDATES_B),
    FrozenCaseSpec("packet_c", "joint_packet", _CANDIDATES_C),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def validate_source_artifact(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "source artifact must be a JSON object")
    expected_metadata = {
        "schema": SOURCE_SCHEMA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "model_calls": EXPECTED_SOURCE_MODEL_CALLS,
        "max_new": 40,
        "claim_status": "exploratory_causal_diagnostic_only",
    }
    for key, expected in expected_metadata.items():
        _require(value.get(key) == expected, f"source artifact {key} binding differs")

    rows = value.get("rows")
    _require(isinstance(rows, list), "source artifact rows must be a list")
    _require(len(rows) == len(EXPECTED_SOURCE_PROMPTS), "source prompt count differs")
    for row, expected in zip(rows, EXPECTED_SOURCE_PROMPTS, strict=True):
        _require(isinstance(row, dict), f"source row {expected.id} is not an object")
        _require(row.get("id") == expected.id, f"source row id differs at {expected.id}")
        _require(
            row.get("kind") == expected.kind,
            f"source row kind differs at {expected.id}",
        )
        _require(
            row.get("prompt") == expected.prompt,
            f"source prompt bytes differ at {expected.id}",
        )
        _require(
            row.get("prompt_token_count") == expected.prompt_token_count,
            f"source prompt token count differs at {expected.id}",
        )
        _require(
            row.get("untruncated_prompt_token_count") == expected.prompt_token_count,
            f"source untruncated prompt count differs at {expected.id}",
        )
        _require(
            row.get("prompt_truncated") is False,
            f"source prompt is truncated at {expected.id}",
        )
    return value


def load_bound_source(path: str | Path) -> tuple[dict[str, Any], str]:
    source_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("source artifact must be a regular file")
        if metadata.st_mode & 0o222:
            raise PermissionError("source artifact must have no write bits")
        payload = source.read()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "source artifact SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_sha256}"
        )
    value = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    return validate_source_artifact(value), actual_sha256


def require_file_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def candidate_manifest_value() -> dict[str, Any]:
    prompts = {row.id: row for row in EXPECTED_SOURCE_PROMPTS}
    return {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "source_artifact_sha256": EXPECTED_SOURCE_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "eos_token": EOS_TOKEN,
        "candidate_class_order": list(CANDIDATE_CLASS_ORDER),
        "cases": [
            {
                "id": case.id,
                "kind": case.kind,
                "prompt": prompts[case.id].prompt,
                "candidates": [
                    {
                        "candidate_class": candidate.candidate_class,
                        "text": candidate.text,
                    }
                    for candidate in case.candidates
                ],
            }
            for case in FROZEN_CASES
        ],
    }


def candidate_manifest_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(candidate_manifest_value()))


def verify_candidate_manifest() -> str:
    actual = candidate_manifest_sha256()
    if actual != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise ValueError(
            "candidate manifest SHA-256 mismatch: "
            f"expected {EXPECTED_CANDIDATE_MANIFEST_SHA256}, got {actual}"
        )
    return actual


def prepare_cases(
    source_artifact: Mapping[str, Any], tokenizer: Any
) -> tuple[tuple[PreparedCase, ...], int]:
    source_rows = {row["id"]: row for row in source_artifact["rows"]}
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    _require(eos_token_id is not None, f"tokenizer is missing EOS token {EOS_TOKEN!r}")

    expected_prompts = {row.id: row for row in EXPECTED_SOURCE_PROMPTS}
    prepared_cases = []
    for case in FROZEN_CASES:
        prompt = source_rows[case.id]["prompt"]
        expected = expected_prompts[case.id]
        _require(prompt == expected.prompt, f"prepared prompt differs at {case.id}")
        prompt_token_ids = tuple(tokenizer.encode(prompt).ids)
        _require(bool(prompt_token_ids), f"prompt tokenized empty at {case.id}")
        _require(
            len(prompt_token_ids) == expected.prompt_token_count,
            f"prompt tokenization drift at {case.id}",
        )

        prepared_candidates = []
        for candidate in case.candidates:
            full_token_ids = tuple(tokenizer.encode(prompt + candidate.text).ids)
            _require(
                full_token_ids[: len(prompt_token_ids)] == prompt_token_ids,
                f"candidate boundary retokenized prompt at {case.id}/{candidate.candidate_class}",
            )
            candidate_token_ids = full_token_ids[len(prompt_token_ids) :]
            _require(
                bool(candidate_token_ids),
                f"candidate tokenized empty at {case.id}/{candidate.candidate_class}",
            )
            _require(
                len(full_token_ids) <= EXPECTED_CONTEXT_LENGTH,
                f"candidate exceeds frozen context at {case.id}/{candidate.candidate_class}",
            )
            prepared_candidates.append(
                PreparedCandidate(
                    candidate.candidate_class,
                    candidate.text,
                    candidate_token_ids,
                )
            )
        prepared_cases.append(
            PreparedCase(
                case.id,
                case.kind,
                prompt,
                prompt_token_ids,
                tuple(prepared_candidates),
                int(eos_token_id),
            )
        )
    return tuple(prepared_cases), int(eos_token_id)


def tokenized_manifest_value(
    prepared_cases: Sequence[PreparedCase], eos_token_id: int
) -> dict[str, Any]:
    return {
        "schema": TOKENIZED_MANIFEST_SCHEMA,
        "candidate_manifest_sha256": candidate_manifest_sha256(),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "eos_token": EOS_TOKEN,
        "eos_token_id": eos_token_id,
        "cases": [
            {
                "id": case.id,
                "kind": case.kind,
                "prompt_token_ids": list(case.prompt_token_ids),
                "candidates": [
                    {
                        "candidate_class": candidate.candidate_class,
                        "candidate_token_ids": list(candidate.token_ids),
                    }
                    for candidate in case.candidates
                ],
            }
            for case in prepared_cases
        ],
    }


def tokenized_manifest_sha256(
    prepared_cases: Sequence[PreparedCase], eos_token_id: int
) -> str:
    return sha256_bytes(
        canonical_json_bytes(tokenized_manifest_value(prepared_cases, eos_token_id))
    )


def verify_tokenized_manifest(
    prepared_cases: Sequence[PreparedCase], eos_token_id: int
) -> str:
    actual = tokenized_manifest_sha256(prepared_cases, eos_token_id)
    if actual != EXPECTED_TOKENIZED_MANIFEST_SHA256:
        raise ValueError(
            "tokenized candidate manifest SHA-256 mismatch: "
            f"expected {EXPECTED_TOKENIZED_MANIFEST_SHA256}, got {actual}"
        )
    return actual


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_teacher_forced_score(
    score: TeacherForcedScore, expected_candidate_tokens: int
) -> TeacherForcedScore:
    _require(
        isinstance(score, TeacherForcedScore),
        "scorer must return TeacherForcedScore",
    )
    _require(
        len(score.candidate_token_log_likelihoods) == expected_candidate_tokens,
        "scorer candidate token count differs",
    )
    for index, value in enumerate(score.candidate_token_log_likelihoods):
        _finite_float(value, f"candidate token log likelihood {index}")
    eos_log_likelihood = _finite_float(
        score.eos_log_likelihood, "EOS log likelihood"
    )
    top_log_likelihood = _finite_float(
        score.next_token_top_log_likelihood, "top-token log likelihood"
    )
    _require(
        isinstance(score.eos_rank, int)
        and not isinstance(score.eos_rank, bool)
        and score.eos_rank >= 1,
        "EOS rank must be a positive integer",
    )
    _require(
        isinstance(score.eos_is_unique_top1, bool),
        "EOS unique-top1 flag must be boolean",
    )
    _require(
        isinstance(score.next_token_top_id, int)
        and not isinstance(score.next_token_top_id, bool)
        and score.next_token_top_id >= 0,
        "top-token id must be a nonnegative integer",
    )
    _require(
        top_log_likelihood + 1e-7 >= eos_log_likelihood,
        "top-token log likelihood is below EOS",
    )
    if score.eos_is_unique_top1:
        _require(
            score.eos_rank == 1
            and abs(top_log_likelihood - eos_log_likelihood) <= 1e-7,
            "unique-top1 EOS must have rank one and match the top likelihood",
        )
    elif score.eos_rank > 1:
        _require(
            top_log_likelihood > eos_log_likelihood,
            "non-top1 EOS must be below the top likelihood",
        )
    return score


def _ranked_classes(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    order = {name: index for index, name in enumerate(CANDIDATE_CLASS_ORDER)}
    return [
        row["candidate_class"]
        for row in sorted(
            rows,
            key=lambda row: (-float(row[field]), order[row["candidate_class"]]),
        )
    ]


def _unique_winner(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    best = max(float(row[field]) for row in rows)
    winners = [row["candidate_class"] for row in rows if float(row[field]) == best]
    return winners[0] if len(winners) == 1 else None


def diagnose_scored_case(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class = {row["candidate_class"]: row for row in scored}
    _require(set(by_class) == set(CANDIDATE_CLASS_ORDER), "scored classes differ")
    normalized_winner = _unique_winner(scored, "normalized_log_likelihood")
    total_winner = _unique_winner(scored, "total_log_likelihood")
    joint_winner = _unique_winner(scored, "candidate_plus_eos_total_log_likelihood")
    correct = by_class[CORRECT_CLASS]

    normalized_preferred = normalized_winner == CORRECT_CLASS
    total_preferred = total_winner == CORRECT_CLASS
    joint_preferred = joint_winner == CORRECT_CLASS
    eos_preferred = bool(correct["eos"]["is_unique_top1"])

    if not normalized_preferred:
        diagnosis = DIAGNOSIS_NOT_PREFERRED
    elif not total_preferred:
        diagnosis = DIAGNOSIS_DECODING
    elif not eos_preferred:
        diagnosis = DIAGNOSIS_TERMINATION
    elif not joint_preferred:
        diagnosis = DIAGNOSIS_JOINT
    else:
        diagnosis = DIAGNOSIS_PREFERRED

    return {
        "diagnosis": diagnosis,
        "correct_unique_top1_normalized": normalized_preferred,
        "correct_unique_top1_total": total_preferred,
        "correct_unique_top1_candidate_plus_eos": joint_preferred,
        "eos_unique_top1_after_correct": eos_preferred,
        "normalized_winner": normalized_winner,
        "total_winner": total_winner,
        "candidate_plus_eos_winner": joint_winner,
        "rankings": {
            "normalized_log_likelihood": _ranked_classes(
                scored, "normalized_log_likelihood"
            ),
            "total_log_likelihood": _ranked_classes(scored, "total_log_likelihood"),
            "candidate_plus_eos_total_log_likelihood": _ranked_classes(
                scored, "candidate_plus_eos_total_log_likelihood"
            ),
        },
    }


def score_prepared_board(
    prepared_cases: Sequence[PreparedCase],
    scorer: Callable[[PreparedCase, PreparedCandidate], TeacherForcedScore],
) -> dict[str, Any]:
    _require(len(prepared_cases) == 6, "prepared case count must be six")
    rows = []
    for case in prepared_cases:
        _require(len(case.candidates) == 5, f"candidate count differs at {case.id}")
        scored_candidates = []
        for candidate in case.candidates:
            raw_score = validate_teacher_forced_score(
                scorer(case, candidate), len(candidate.token_ids)
            )
            if raw_score.eos_is_unique_top1:
                _require(
                    raw_score.next_token_top_id == case.eos_token_id,
                    "unique-top1 EOS does not match the bound EOS token id",
                )
            token_values = [
                float(value) for value in raw_score.candidate_token_log_likelihoods
            ]
            total = sum(token_values)
            normalized = total / len(token_values)
            eos = float(raw_score.eos_log_likelihood)
            scored_candidates.append(
                {
                    "candidate_class": candidate.candidate_class,
                    "text": candidate.text,
                    "candidate_token_ids": list(candidate.token_ids),
                    "candidate_token_log_likelihoods": token_values,
                    "candidate_token_count": len(candidate.token_ids),
                    "total_log_likelihood": total,
                    "normalized_log_likelihood": normalized,
                    "eos": {
                        "token": EOS_TOKEN,
                        "token_id": case.eos_token_id,
                        "log_likelihood": eos,
                        "rank": raw_score.eos_rank,
                        "is_unique_top1": raw_score.eos_is_unique_top1,
                        "margin_to_top1": eos
                        - float(raw_score.next_token_top_log_likelihood),
                        "next_token_top_id": raw_score.next_token_top_id,
                        "next_token_top_log_likelihood": float(
                            raw_score.next_token_top_log_likelihood
                        ),
                    },
                    "candidate_plus_eos_total_log_likelihood": total + eos,
                    "candidate_plus_eos_normalized_log_likelihood": (
                        total + eos
                    )
                    / (len(candidate.token_ids) + 1),
                }
            )
        case_diagnosis = diagnose_scored_case(scored_candidates)
        rows.append(
            {
                "id": case.id,
                "kind": case.kind,
                "prompt": case.prompt,
                "prompt_token_ids": list(case.prompt_token_ids),
                "prompt_token_count": len(case.prompt_token_ids),
                "candidates": scored_candidates,
                **case_diagnosis,
            }
        )

    diagnosis_counts = {
        diagnosis: sum(row["diagnosis"] == diagnosis for row in rows)
        for diagnosis in (
            DIAGNOSIS_NOT_PREFERRED,
            DIAGNOSIS_DECODING,
            DIAGNOSIS_TERMINATION,
            DIAGNOSIS_JOINT,
            DIAGNOSIS_PREFERRED,
        )
    }
    if diagnosis_counts[DIAGNOSIS_NOT_PREFERRED]:
        overall = "correct_update_not_consistently_preferred"
    elif diagnosis_counts[DIAGNOSIS_PREFERRED] == len(rows):
        overall = "correct_update_and_termination_preferred"
    else:
        overall = (
            "correct_update_likelihood_preferred_but_decoding_or_termination_loses"
        )

    prompt_tokens = sum(
        len(case.prompt_token_ids) * len(case.candidates) for case in prepared_cases
    )
    candidate_tokens = sum(
        len(candidate.token_ids)
        for case in prepared_cases
        for candidate in case.candidates
    )
    candidate_evaluations = sum(len(case.candidates) for case in prepared_cases)
    resource_ledger = {
        "source_prompt_rows_read": len(EXPECTED_SOURCE_PROMPTS),
        "scored_prompt_rows": len(prepared_cases),
        "candidate_classes_per_prompt": len(CANDIDATE_CLASS_ORDER),
        "candidate_sequence_evaluations": candidate_evaluations,
        "model_forward_calls": candidate_evaluations,
        "prompt_tokens_replayed": prompt_tokens,
        "supervised_candidate_tokens": candidate_tokens,
        "supervised_eos_tokens": candidate_evaluations,
        "teacher_forced_target_tokens": candidate_tokens + candidate_evaluations,
        "forward_token_positions": prompt_tokens + candidate_tokens,
        "generated_tokens": 0,
        "sampled_tokens": 0,
        "training_tokens": 0,
        "retries": 0,
        "repairs": 0,
        "candidate_searches": 0,
        "threshold_searches": 0,
        "verifier_feedback_calls": 0,
        "external_generation_calls": 0,
    }
    summary = {
        "overall_diagnosis": overall,
        "diagnosis_counts": diagnosis_counts,
        "correct_unique_top1_normalized": sum(
            row["correct_unique_top1_normalized"] for row in rows
        ),
        "correct_unique_top1_total": sum(
            row["correct_unique_top1_total"] for row in rows
        ),
        "correct_unique_top1_candidate_plus_eos": sum(
            row["correct_unique_top1_candidate_plus_eos"] for row in rows
        ),
        "eos_unique_top1_after_correct": sum(
            row["eos_unique_top1_after_correct"] for row in rows
        ),
        "cases": len(rows),
    }
    return {"rows": rows, "summary": summary, "resource_ledger": resource_ledger}


@torch.inference_mode()
def teacher_forced_score(
    model: Any,
    case: PreparedCase,
    candidate: PreparedCandidate,
    device: str,
) -> TeacherForcedScore:
    input_token_ids = case.prompt_token_ids + candidate.token_ids
    target_token_ids = candidate.token_ids + (case.eos_token_id,)
    tokens = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    logits, _ = model(tokens)
    start = len(case.prompt_token_ids) - 1
    selected_logits = logits[0, start : start + len(target_token_ids)].float()
    if selected_logits.shape[0] != len(target_token_ids):
        raise ValueError("model returned too few positions for candidate plus EOS")
    log_probabilities = F.log_softmax(selected_logits, dim=-1)
    targets = torch.tensor(target_token_ids, dtype=torch.long, device=device)
    selected = log_probabilities.gather(1, targets[:, None]).squeeze(1)

    candidate_values = tuple(float(value) for value in selected[:-1].cpu().tolist())
    eos_distribution = log_probabilities[-1]
    eos_log_likelihood = float(selected[-1].item())
    top_log_likelihood, top_id = torch.max(eos_distribution, dim=0)
    eos_rank = 1 + int((eos_distribution > selected[-1]).sum().item())
    top_ties = int((eos_distribution == top_log_likelihood).sum().item())
    eos_is_unique_top1 = (
        eos_rank == 1 and int(top_id.item()) == case.eos_token_id and top_ties == 1
    )
    return TeacherForcedScore(
        candidate_token_log_likelihoods=candidate_values,
        eos_log_likelihood=eos_log_likelihood,
        eos_rank=eos_rank,
        eos_is_unique_top1=eos_is_unique_top1,
        next_token_top_id=int(top_id.item()),
        next_token_top_log_likelihood=float(top_log_likelihood.item()),
    )


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cuda", "mps"}:
        raise ValueError("device must be auto, cuda, or mps")
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            raise RuntimeError("neither CUDA nor MPS is available")
    else:
        device = requested
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    try:
        probe = torch.zeros(1, device=device)
        del probe
    except Exception as error:
        raise RuntimeError(f"{device} allocation preflight failed") from error
    return device


def load_model(path: str | Path, device: str) -> tuple[int, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "checkpoint must be a dictionary")
    _require(
        checkpoint.get("step") == EXPECTED_CHECKPOINT_STEP,
        "checkpoint step differs from frozen raw-260k step",
    )
    config = checkpoint.get("cfg")
    _require(isinstance(config, dict), "checkpoint config is missing")
    _require(
        config.get("seq_len") == EXPECTED_CONTEXT_LENGTH,
        "checkpoint context length differs",
    )
    _require(isinstance(checkpoint.get("model"), dict), "checkpoint model is missing")
    model = GPT(GPTConfig(**config))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    step = int(checkpoint["step"])
    del checkpoint
    return step, model


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    if destination.stat().st_mode & 0o222:
        raise PermissionError("diagnostic output remained writable")
    try:
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError:
        pass
    return sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    # Candidate text and token IDs are fully verified before the model is loaded.
    source_artifact, source_sha256 = load_bound_source(DEFAULT_SOURCE)
    candidate_sha256 = verify_candidate_manifest()
    checkpoint_sha256 = require_file_sha256(
        args.ckpt, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
    )
    tokenizer_sha256 = require_file_sha256(
        args.tokenizer, EXPECTED_TOKENIZER_SHA256, "tokenizer"
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prepared_cases, eos_token_id = prepare_cases(source_artifact, tokenizer)
    tokenized_sha256 = verify_tokenized_manifest(prepared_cases, eos_token_id)

    device = resolve_device(args.device)
    checkpoint_step, model = load_model(args.ckpt, device)
    scored = score_prepared_board(
        prepared_cases,
        lambda case, candidate: teacher_forced_score(
            model, case, candidate, device
        ),
    )
    scored["resource_ledger"].update(
        {
            "source_artifact_reads": 1,
            "checkpoint_hash_passes": 1,
            "tokenizer_hash_passes": 1,
            "tokenizer_loads": 1,
            "device_preflight_allocations": 1,
            "model_loads": 1,
            "candidates_frozen_before_model_load": True,
        }
    )

    result = {
        "schema": SCHEMA,
        "claim_status": "preregistered_teacher_forced_diagnostic_only",
        "bindings": {
            "source_artifact": str(DEFAULT_SOURCE.resolve()),
            "source_artifact_sha256": source_sha256,
            "candidate_manifest_sha256": candidate_sha256,
            "tokenized_candidate_manifest_sha256": tokenized_sha256,
            "checkpoint": str(Path(args.ckpt).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": checkpoint_step,
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "tokenizer_sha256": tokenizer_sha256,
            "eos_token": EOS_TOKEN,
            "eos_token_id": eos_token_id,
        },
        "device": device,
        "decision_rule": {
            "primary_content_metric": "candidate_token_normalized_log_likelihood",
            "sequence_metric": "candidate_token_total_log_likelihood",
            "termination_metric": "eos_unique_top1_immediately_after_candidate",
            "complete_sequence_metric": "candidate_plus_eos_total_log_likelihood",
            "ties": "not_unique_top1",
            "latent_label_scope": (
                "Operational fixed-candidate preference only; it is not proof of a "
                "latent state, causal mechanism, or free-running ability."
            ),
        },
        "summary": scored["summary"],
        "resource_ledger": scored["resource_ledger"],
        "rows": scored["rows"],
    }
    digest = write_immutable_json(output, result)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": digest,
                "overall_diagnosis": result["summary"]["overall_diagnosis"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
