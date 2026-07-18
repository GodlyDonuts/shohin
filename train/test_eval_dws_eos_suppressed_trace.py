#!/usr/bin/env python3
"""CPU contracts for the development-only EOS-suppressed DWS screen."""

from __future__ import annotations

import ast
import array
import base64
import copy
import fcntl
import hashlib
import inspect
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import textwrap
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer as RealTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_dws_eos_suppressed_trace as screen  # noqa: E402

screen.torch = torch
screen.F = torch.nn.functional
screen.Tokenizer = RealTokenizer
TEST_AUTHORITY_PRIVATE_KEY = bytes(range(32))
screen.ALLOW_TEST_AUTHORITY = True
screen.TEST_AUTHORITY_PUBLIC_KEY_HEX = screen.signing_key_record(
    TEST_AUTHORITY_PRIVATE_KEY
)["public_key_hex"]


def _test_rename_noreplace_at(
    old_directory_fd: int,
    old_name: str,
    new_directory_fd: int,
    new_name: str,
) -> None:
    try:
        os.stat(new_name, dir_fd=new_directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(new_name)
    os.rename(
        old_name,
        new_name,
        src_dir_fd=old_directory_fd,
        dst_dir_fd=new_directory_fd,
    )


def _noop_publication_stage(_stage: str) -> None:
    return None


class TinyEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class TinyTokenizer:
    def __init__(self) -> None:
        fixed = {";": 39, "x": 100, "\n": 211, " ": 233}
        self._char_to_id = dict(fixed)
        self._id_to_char = {value: key for key, value in fixed.items()}
        self._next = 1
        self.decode_calls = 0

    def _id_for(self, character: str) -> int:
        if character in self._char_to_id:
            return self._char_to_id[character]
        while self._next in {0, 39, 100, 211, 233} or self._next in self._id_to_char:
            self._next += 1
        token_id = self._next
        self._next += 1
        self._char_to_id[character] = token_id
        self._id_to_char[token_id] = character
        return token_id

    def encode(self, text: str) -> TinyEncoding:
        return TinyEncoding([self._id_for(character) for character in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        self.decode_calls += 1
        parts = []
        for token_id in ids:
            if token_id == 0:
                if not skip_special_tokens:
                    parts.append("<|endoftext|>")
                continue
            parts.append(self._id_to_char.get(token_id, f"<{token_id}>"))
        return "".join(parts)

    def token_to_id(self, token: str) -> int | None:
        return 0 if token == "<|endoftext|>" else self._char_to_id.get(token)


class TinyModel:
    def __init__(
        self, raw_choices: list[int], runner_up: int = 7, vocabulary: int = 256
    ) -> None:
        self.cfg = SimpleNamespace(seq_len=4096)
        self.raw_choices = raw_choices
        self.runner_up = runner_up
        self.vocabulary = vocabulary
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        idx: torch.Tensor,
        cache: object | None = None,
        pos: int = 0,
        return_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        assert return_cache is True
        step = len(self.calls)
        raw_choice = self.raw_choices[min(step, len(self.raw_choices) - 1)]
        logits = torch.full((1, idx.shape[1], self.vocabulary), -20.0)
        logits[0, -1, self.runner_up] = 9.0
        logits[0, -1, raw_choice] = 10.0
        self.calls.append(
            {
                "cache_was_none": cache is None,
                "pos": pos,
                "input": idx.detach().cpu().tolist(),
            }
        )
        return logits, {"step": step}


def test_eos_mask_is_the_only_logit_intervention() -> None:
    logits = torch.linspace(-3.0, 3.0, 256)
    logits[0] = 20.0
    logits[17] = 19.0
    logits_before = logits.clone()
    decision = screen.apply_decode_intervention(
        logits, screen.DecodeMode.EOS_MASKED_ARGMAX, 0
    )
    assert torch.isneginf(decision.logits_after_intervention[0])
    assert torch.equal(decision.logits_after_intervention[1:], logits[1:])
    assert decision.raw_argmax_id == 0
    assert decision.selected_token_id == 17
    assert decision.override_applied is True
    assert torch.equal(logits, logits_before)

    for mode, expected in (
        (screen.DecodeMode.EOS_TO_LF, 211),
        (screen.DecodeMode.EOS_TO_SPACE, 233),
        (screen.DecodeMode.EOS_TO_SEMICOLON, 39),
        (screen.DecodeMode.EOS_TO_NONFORMAT_X, 100),
    ):
        forced = screen.apply_decode_intervention(logits, mode, 0)
        assert torch.equal(forced.logits_after_intervention, logits)
        assert forced.selected_token_id == expected
        assert forced.override_applied is True


@pytest.mark.parametrize(
    "mode,selected",
    [
        (screen.DecodeMode.EOS_MASKED_ARGMAX, 7),
        (screen.DecodeMode.EOS_TO_LF, 211),
        (screen.DecodeMode.EOS_TO_SPACE, 233),
        (screen.DecodeMode.EOS_TO_SEMICOLON, 39),
        (screen.DecodeMode.EOS_TO_NONFORMAT_X, 100),
    ],
)
def test_fixed_budget_has_no_content_dependent_stop(
    mode: screen.DecodeMode, selected: int
) -> None:
    model = TinyModel([0])
    raw = screen.decode_cached_greedy(
        screen.DecodeRequest(
            model=model,
            prompt_token_ids=(12, 13),
            device="cpu",
            max_new_tokens=5,
            mode=mode,
            eos_token_id=0,
        )
    )
    assert raw.generated_token_ids == (selected,) * 5
    assert raw.prompt_token_count == 2
    assert raw.stop_reason == "fixed_budget"
    assert [event.generated_index for event in raw.eos_events] == list(range(5))
    assert len(model.calls) == 5
    assert model.calls[0]["cache_was_none"] is True
    assert all(call["cache_was_none"] is False for call in model.calls[1:])
    if mode == screen.DecodeMode.EOS_MASKED_ARGMAX:
        assert raw.eos_mask_applied_positions == tuple(range(5))
    else:
        assert raw.eos_mask_applied_positions == ()

    ordinary_model = TinyModel([0])
    ordinary = screen.decode_cached_greedy(
        screen.DecodeRequest(
            model=ordinary_model,
            prompt_token_ids=(12, 13),
            device="cpu",
            max_new_tokens=5,
            mode=screen.DecodeMode.ORDINARY_EOS_STOP,
            eos_token_id=0,
        )
    )
    assert ordinary.generated_token_ids == (0,)
    assert ordinary.prompt_token_count == 2
    assert ordinary.stop_reason == "eos"
    assert len(ordinary_model.calls) == 1


def test_decode_has_no_tokenizer_gold_or_oracle_access() -> None:
    request_fields = set(screen.DecodeRequest.__dataclass_fields__)
    assert request_fields == {
        "model",
        "prompt_token_ids",
        "device",
        "max_new_tokens",
        "mode",
        "eos_token_id",
    }
    source = textwrap.dedent(inspect.getsource(screen.decode_cached_greedy))
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(
        {
            "parse_dws_line",
            "apply_microstep_posthoc",
            "reconstruct_oracle_posthoc",
            "state_answer_posthoc",
            "decode_text_posthoc",
        }
    )
    lowered = source.lower()
    assert "gold" not in lowered
    assert "oracle" not in lowered
    assert "answer" not in lowered
    assert "tokenizer" not in lowered


def test_parser_and_trace_scorer_handle_malformed_repeated_and_post_terminal() -> None:
    initial = screen.parse_dws_line("dws:op=add;w=2;p=0;c=0;a=21;b=90;r=00;z=0")
    assert initial is not None
    oracle = screen.reconstruct_oracle_posthoc(initial)
    first, terminal = [screen.canonical_dws_state(state) for state in oracle]

    valid = screen.score_trace_posthoc(
        f"{first}\n{terminal}\nanswer=21", initial, oracle
    )
    assert valid["longest_exact_prefix"] == 2
    assert valid["full_exact_trace_through_first_terminal"] is True
    assert valid["exact_final_tape"] is True
    assert valid["terminal_answer_exact"] is True
    assert valid["generated_answer_exact"] is True
    assert valid["response_grammar_valid"] is True

    malformed = "dws:op=add;w=2;p=2;c=0;a=21;b=90;r=1;z=1"
    hostile = screen.score_trace_posthoc(
        f"{first}\n{malformed}\n{first}\n{terminal}\n{terminal}\nanswer=21",
        initial,
        oracle,
    )
    assert hostile["longest_exact_prefix"] == 1
    assert hostile["malformed_dws_line_indices"] == [1]
    assert hostile["repeated_dws_line_indices"] == [2, 4]
    assert hostile["post_terminal_dws_line_indices"] == [4]
    assert hostile["full_exact_trace_through_first_terminal"] is False
    assert hostile["exact_final_tape"] is True
    assert hostile["response_grammar_valid"] is False

    post_terminal = screen.score_trace_posthoc(
        f"{first}\n{terminal}\n{terminal}", initial, oracle
    )
    assert post_terminal["longest_exact_prefix"] == 2
    assert post_terminal["full_exact_trace_through_first_terminal"] is True
    assert post_terminal["post_terminal_dws_line_indices"] == [2]
    assert post_terminal["response_grammar_valid"] is False


def test_history_counterfactuals_and_compound_fresh_reencoding_are_separate() -> None:
    tokenizer = TinyTokenizer()
    initial = screen.parse_dws_line("dws:op=add;w=3;p=0;c=0;a=867;b=965;r=000;z=0")
    assert initial is not None
    emitted = screen.apply_microstep_posthoc(initial)
    emitted_line = screen.canonical_dws_state(emitted)
    emitted_ids = tuple(tokenizer.encode(emitted_line).ids)
    branches = screen.build_history_branches_posthoc(emitted, emitted_ids, tokenizer)
    assert branches["carry_flip"]["state"]["c"] == 1 - emitted["c"]
    assert branches["written_result_r0_flip"]["state"]["r"][1:] == emitted["r"][1:]
    operand = branches["active_operand_digit_perturbation"]
    assert operand["metadata"]["position"] == emitted["p"]
    assert (
        screen.apply_microstep_posthoc(operand["state"])["r"][emitted["p"]]
        != screen.apply_microstep_posthoc(emitted)["r"][emitted["p"]]
    )
    destroyed = branches["equal_token_length_destroyed_history"]
    assert destroyed["history_token_count"] == len(emitted_ids)
    assert destroyed["history_token_ids"] == (100,) * len(emitted_ids)

    full_history_prompt = screen.render_initial_prompt_bytes(
        screen.canonical_dws_state(initial)
    )
    fresh_prompt = screen.render_core_prompt_bytes(emitted_line)
    assert screen.canonical_dws_state(initial).encode() in full_history_prompt
    assert screen.canonical_dws_state(initial).encode() not in fresh_prompt
    assert emitted_line.encode() in fresh_prompt
    assert fresh_prompt.startswith(screen.PROMPT_PREFIX.encode())
    assert fresh_prompt.endswith(screen.PROMPT_SUFFIX.encode())


def test_external_reencoding_requests_execute_exact_fresh_core_prompts() -> None:
    tokenizer = TinyTokenizer()
    model = TinyModel([0])
    initial = screen.parse_dws_line("dws:op=add;w=3;p=0;c=0;a=867;b=965;r=000;z=0")
    assert initial is not None
    initial_line = screen.canonical_dws_state(initial)
    prompt_ids = tuple(
        tokenizer.encode(screen.render_initial_prompt_bytes(initial_line).decode()).ids
    )
    emitted = screen.apply_microstep_posthoc(initial)
    emitted_line = screen.canonical_dws_state(emitted)
    emitted_ids = tuple(tokenizer.encode(emitted_line).ids)
    ordinary_raw = screen.RawDecode(
        mode=screen.DecodeMode.ORDINARY_EOS_STOP.value,
        prompt_token_ids=prompt_ids,
        prompt_token_count=len(prompt_ids),
        generated_token_ids=(*emitted_ids, 0),
        stop_reason="eos",
        eos_mask_applied_positions=(),
        eos_events=(
            screen.RawEosEvent(
                generated_index=len(emitted_ids),
                absolute_token_position=len(prompt_ids) + len(emitted_ids),
                eos_logit=10.0,
                non_eos_argmax_token_id=211,
                non_eos_argmax_logit=9.0,
                replacement_token_id=None,
                replacement_raw_logit=None,
            ),
        ),
    )
    ordinary_report = screen.raw_decode_report_posthoc(
        ordinary_raw, tokenizer, initial, screen.reconstruct_oracle_posthoc(initial)
    )

    full_history, fresh, branches, availability = screen.prepare_field_requests_posthoc(
        model, "cpu", prompt_ids, ordinary_report, tokenizer
    )
    assert availability["available"] is True
    assert branches is not None
    assert fresh["intact"].prompt_token_ids == tuple(
        tokenizer.encode(screen.render_core_prompt_bytes(emitted_line).decode()).ids
    )
    assert fresh["intact"].mode is screen.DecodeMode.ORDINARY_EOS_STOP
    assert fresh["intact"].max_new_tokens == screen.MAX_NEW_TOKENS
    assert full_history[screen.DecodeMode.EOS_TO_LF.value][
        "intact"
    ].prompt_token_ids == (*prompt_ids, *emitted_ids, 211)
    decoded_fresh = tokenizer.decode(list(fresh["intact"].prompt_token_ids))
    assert initial_line not in decoded_fresh
    assert emitted_line in decoded_fresh


def test_spurious_counterfactual_accuracy_does_not_pass_paired_carry_veto() -> None:
    emitted = screen.parse_dws_line("dws:op=add;w=2;p=1;c=1;a=86;b=96;r=70;z=0")
    assert emitted is not None
    flipped = dict(emitted)
    flipped["c"] = 0
    intact_target = screen.apply_microstep_posthoc(emitted)
    counterfactual_target = screen.apply_microstep_posthoc(flipped)

    # A default-carry-zero output can be counterfactually exact without responding.
    default_output = counterfactual_target
    score = screen.paired_endpoint_score(
        "active_result_digit",
        emitted,
        {"field": "c"},
        intact_target,
        counterfactual_target,
        default_output,
        default_output,
    )
    assert default_output == counterfactual_target
    assert score["counterfactual_target_exact"] is True
    assert score["output_changed"] is False
    assert score["paired_target_switch_exact"] is False

    exact_pair = screen.paired_endpoint_score(
        "active_result_digit",
        emitted,
        {"field": "c"},
        intact_target,
        counterfactual_target,
        intact_target,
        counterfactual_target,
    )
    assert exact_pair["output_changed"] is True
    assert exact_pair["paired_target_switch_exact"] is True
    records = [{"carry_target_switch_exact": True} for _ in range(100)]
    assert screen.carry_target_switch_veto(records) is False
    records[-1]["carry_target_switch_exact"] = False
    assert screen.carry_target_switch_veto(records) is True
    assert "noncompensatory carry-use veto" in screen.CLAIM_BOUNDARY
    assert "compound intervention" in screen.CLAIM_BOUNDARY
    assert "does not isolate a stale-source mechanism" in screen.CLAIM_BOUNDARY


def test_compound_fresh_reencoding_scores_without_mechanism_claim() -> None:
    tokenizer = TinyTokenizer()
    initial = screen.parse_dws_line("dws:op=add;w=3;p=0;c=0;a=867;b=965;r=000;z=0")
    assert initial is not None
    emitted = screen.apply_microstep_posthoc(initial)
    emitted_ids = tuple(tokenizer.encode(screen.canonical_dws_state(emitted)).ids)
    branches = screen.build_history_branches_posthoc(emitted, emitted_ids, tokenizer)
    reports = {}
    for name in screen.FRESH_REENCODING_BRANCHES:
        target = screen.apply_microstep_posthoc(branches[name]["state"])
        reports[name] = {
            "observed_first_state": screen.canonical_dws_state(target),
            "full_target_exact": True,
        }
    oracle = screen.reconstruct_oracle_posthoc(initial)
    detail = screen.score_fresh_reencoding_posthoc(
        reports,
        branches,
        emitted,
        oracle,
        {"carry_target_switch_exact": False},
    )
    assert detail["external_reencoding"] is True
    assert detail["adjacent_transition_of_emitted_state_exact"] is True
    assert detail["carry_full_target_exact"] is True
    assert detail["carry_output_switch"] is True
    assert detail["carry_target_switch_exact"] is True
    assert detail["carry_target_switch_recovery_vs_full_history_lf"] is True


def test_frozen_hashes_selection_and_prompt_contract() -> None:
    screen.validate_static_contract()
    heldout = screen.read_verified_bytes(
        screen.HELDOUT_PATH, screen.EXPECTED_SHA256["heldout"]
    )
    selected = screen.select_cases(screen.parse_heldout_bytes(heldout))
    assert len(selected) == 100
    assert Counter((row["split"], row["operation"]) for row in selected) == {
        (regime, operation): 10
        for regime in screen.EXPECTED_REGIMES
        for operation in screen.OPERATIONS
    }
    ordered = ("\n".join(row["id"] for row in selected) + "\n").encode()
    assert hashlib.sha256(ordered).hexdigest() == screen.ORDERED_CASE_IDS_SHA256
    assert set(screen.REPLICATION_CASE_IDS).issubset({row["id"] for row in selected})
    assert (
        hashlib.sha256(
            ("\n".join(screen.REPLICATION_CASE_IDS) + "\n").encode()
        ).hexdigest()
        == screen.REPLICATION_CASE_IDS_SHA256
    )


def test_exact_heldout_board_binding_rejects_case_and_oracle_substitution(
    tmp_path: Path,
) -> None:
    heldout_bytes = screen.read_verified_bytes(
        screen.HELDOUT_PATH, screen.EXPECTED_SHA256["heldout"]
    )
    selected = screen.select_cases(screen.parse_heldout_bytes(heldout_bytes))
    case_identities = [
        {
            "case_id": row["id"],
            "split": row["split"],
            "operation": row["operation"],
            "width": row["width"],
            "initial_state": row["initial_state"],
            "selection_sha256": screen.selection_digest(row["id"]),
        }
        for row in selected
    ]
    for index, (case, row) in enumerate(zip(case_identities, selected, strict=True)):
        screen.validate_report_case_heldout_identity(case, row, f"cases[{index}]")

    substituted_case = copy.deepcopy(case_identities[0])
    substituted_case["initial_state"] = selected[1]["initial_state"]
    with pytest.raises(screen.ContractError, match="frozen heldout identity"):
        screen.validate_report_case_heldout_identity(
            substituted_case, selected[0], "cases[0]"
        )

    initial = screen.parse_dws_line(selected[0]["initial_state"])
    assert initial is not None
    oracle = screen.reconstruct_oracle_posthoc(initial)
    binding = screen.validate_heldout_oracle_posthoc(selected[0], oracle)
    assert (
        binding["row_sha256"]
        == hashlib.sha256(screen.stable_json_bytes(selected[0])).hexdigest()
    )
    substituted_oracle = copy.deepcopy(selected[0])
    substituted_oracle["expected_answer"] += 1
    with pytest.raises(screen.ContractError, match="expected-answer mismatch"):
        screen.validate_heldout_oracle_posthoc(substituted_oracle, oracle)
    substituted_oracle = copy.deepcopy(selected[0])
    substituted_oracle["expected_states"][0] = selected[1]["expected_states"][0]
    with pytest.raises(screen.ContractError, match="expected-state mismatch"):
        screen.validate_heldout_oracle_posthoc(substituted_oracle, oracle)

    hostile_path = tmp_path / "heldout-substitution.jsonl"
    hostile_path.write_bytes(heldout_bytes + b"\n")
    with pytest.raises(screen.ContractError, match="hash mismatch"):
        screen.read_verified_bytes(hostile_path, screen.EXPECTED_SHA256["heldout"])


_SYNTHETIC_CASES: list[dict[str, object]] | None = None


def _synthetic_full_schema_cases() -> list[dict[str, object]]:
    global _SYNTHETIC_CASES
    if _SYNTHETIC_CASES is not None:
        return copy.deepcopy(_SYNTHETIC_CASES)
    tokenizer = RealTokenizer.from_str(
        screen.read_verified_bytes(
            screen.TOKENIZER_PATH, screen.EXPECTED_SHA256["tokenizer"]
        ).decode("utf-8")
    )
    heldout = screen.parse_heldout_bytes(
        screen.read_verified_bytes(
            screen.HELDOUT_PATH, screen.EXPECTED_SHA256["heldout"]
        )
    )
    cases = []
    for row in screen.select_cases(heldout):
        initial_state = screen.parse_dws_line(row["initial_state"])
        assert initial_state is not None
        oracle = screen.reconstruct_oracle_posthoc(initial_state)
        prompt_bytes = screen.render_initial_prompt_bytes(row["initial_state"])
        prompt_ids = tuple(tokenizer.encode(prompt_bytes.decode("ascii")).ids)
        ordinary_content = (100,)
        ordinary_event = screen.RawEosEvent(
            generated_index=len(ordinary_content),
            absolute_token_position=len(prompt_ids) + len(ordinary_content),
            eos_logit=10.0,
            non_eos_argmax_token_id=211,
            non_eos_argmax_logit=9.0,
            replacement_token_id=None,
            replacement_raw_logit=None,
        )
        primary = {
            screen.DecodeMode.ORDINARY_EOS_STOP.value: (
                screen.raw_decode_report_posthoc(
                    screen.RawDecode(
                        mode=screen.DecodeMode.ORDINARY_EOS_STOP.value,
                        prompt_token_ids=prompt_ids,
                        prompt_token_count=len(prompt_ids),
                        generated_token_ids=(*ordinary_content, screen.EOS_TOKEN_ID),
                        stop_reason="eos",
                        eos_mask_applied_positions=(),
                        eos_events=(ordinary_event,),
                    ),
                    tokenizer,
                    initial_state,
                    oracle,
                )
            )
        }
        filler = (100,) * screen.MAX_NEW_TOKENS
        for arm in screen.FIXED_BUDGET_ARMS:
            primary[arm] = screen.raw_decode_report_posthoc(
                screen.RawDecode(
                    mode=arm,
                    prompt_token_ids=prompt_ids,
                    prompt_token_count=len(prompt_ids),
                    generated_token_ids=filler,
                    stop_reason="fixed_budget",
                    eos_mask_applied_positions=(
                        tuple(range(screen.MAX_NEW_TOKENS))
                        if arm == screen.DecodeMode.EOS_MASKED_ARGMAX.value
                        else ()
                    ),
                    eos_events=(),
                ),
                tokenizer,
                initial_state,
                oracle,
            )
        cases.append(
            {
                "case_id": row["id"],
                "split": row["split"],
                "operation": row["operation"],
                "width": row["width"],
                "selection_sha256": screen.selection_digest(row["id"]),
                "initial_state": row["initial_state"],
                "prompt": {
                    "utf8": prompt_bytes.decode("ascii"),
                    "byte_count": len(prompt_bytes),
                    "sha256": screen.sha256_bytes(prompt_bytes),
                    "token_ids": list(prompt_ids),
                    "token_count": len(prompt_ids),
                    "token_ids_sha256": screen.token_ids_sha256(prompt_ids),
                },
                "oracle": {
                    "states": [screen.canonical_dws_state(state) for state in oracle],
                    "trace_length": len(oracle),
                    "first_state_carry": oracle[0]["c"],
                    "final_tape": {"r": oracle[-1]["r"], "c": oracle[-1]["c"]},
                    "answer": screen.state_answer_posthoc(oracle[-1]),
                    "heldout_binding": screen.validate_heldout_oracle_posthoc(
                        row, oracle
                    ),
                },
                "primary_arms": primary,
                "field_screen": screen.unavailable_field_screen(
                    "ordinary_first_response_is_not_one_nonterminal_step"
                ),
            }
        )
    _SYNTHETIC_CASES = cases
    return copy.deepcopy(cases)


def _runtime_observation(phase: str, python_path: str) -> dict[str, object]:
    records = [
        {
            "path": path,
            "sha256": digest * 64,
            "device": 1,
            "inode": index,
            "size": 1,
            "mapping_identity": None,
        }
        for index, (path, digest) in enumerate(
            (
                ("/test/ld-linux.so", "1"),
                ("/test/libc.so.6", "2"),
                ("/test/libcuda.so.1", "3"),
            ),
            1,
        )
    ]
    return {
        "schema": screen.RUNTIME_OBSERVATION_SCHEMA,
        "phase": phase,
        "coverage": "loaded_file_snapshot_not_complete_immutable_executed_runtime_seal",
        "python_executable": python_path,
        "platform": {"system": "test", "release": "test", "machine": "test"},
        "libc_version": ["test-libc", "1"],
        "ld_preload": None,
        "dyld_insert_libraries": None,
        "ld_library_path": None,
        "mapping_source_sha256": "a" * 64,
        "loaded_files": records,
        "loaded_files_sha256": screen.sha256_bytes(screen.stable_json_bytes(records)),
        "shared_objects": records,
        "shared_objects_sha256": screen.sha256_bytes(screen.stable_json_bytes(records)),
        "libc_objects": [records[1]],
        "loader_objects": [records[0]],
        "cuda_objects": [records[2]],
        "cuda_driver_version": 12000,
        "cuda_runtime_version": "12.test",
        "cuda_visible_devices": "0",
        "cuda_device_name": screen.REQUIRED_CUDA_DEVICE_NAME,
        "cuda_device_capability": list(screen.REQUIRED_CUDA_DEVICE_CAPABILITY),
        "cuda_device_total_memory_bytes": screen.REQUIRED_CUDA_MEMORY_MIN_BYTES,
        "cuda_device_uuid": "GPU-test",
        "nvidia_driver_file": {
            "path": "/proc/driver/nvidia/version",
            "sha256": "b" * 64,
            "text_sha256": "c" * 64,
        },
    }


def _slurm_tres() -> dict[str, str]:
    return {
        "billing": str(screen.SLURM_CPUS_PER_TASK),
        "cpu": str(screen.SLURM_CPUS_PER_TASK),
        "gres/gpu": str(screen.SLURM_GPU_COUNT),
        screen.SLURM_TYPED_GPU_TRES: str(screen.SLURM_GPU_COUNT),
        "mem": screen.SLURM_MEMORY,
        "node": str(screen.SLURM_NODE_COUNT),
    }


def _slurm_identity(
    command: str,
    command_sha256: str,
    *,
    job_id: str = "12345",
) -> dict[str, object]:
    identity = {
        "job_id": job_id,
        "job_name": "shohin-r12-dws-eos-dev",
        "job_state": "RUNNING",
        "cluster_name": "test-cluster",
        "user_name": "test-user",
        "user_uid": os.getuid(),
        "batch_flag": 1,
        "command": command,
        "command_sha256": command_sha256,
        "batch_host": "test-host",
        "node_list": "test-host",
        "observed_hostname": "test-host",
        "partition": screen.SLURM_PARTITION,
        "num_nodes": screen.SLURM_NODE_COUNT,
        "num_cpus": screen.SLURM_CPUS_PER_TASK,
        "num_tasks": screen.SLURM_TASK_COUNT,
        "cpus_per_task": screen.SLURM_CPUS_PER_TASK,
        "min_cpus_node": screen.SLURM_CPUS_PER_TASK,
        "min_memory_node": screen.SLURM_MEMORY,
        "memory_bytes": screen.SLURM_MEMORY_BYTES,
        "time_limit": screen.SLURM_TIME_LIMIT,
        "time_limit_seconds": screen.SLURM_TIME_LIMIT_SECONDS,
        "requeue": 0,
        "gres": screen.SLURM_GRES,
        "gpu_type": screen.SLURM_GPU_TYPE,
        "gpu_count": screen.SLURM_GPU_COUNT,
        "req_tres": _slurm_tres(),
        "alloc_tres": _slurm_tres(),
        "tres_per_node": {screen.SLURM_TYPED_GPU_TRES: str(screen.SLURM_GPU_COUNT)},
        "job_record_sha256": "e" * 64,
        "cluster_config_sha256": "f" * 64,
        "sacct_identity": {
            "job_id_raw": job_id,
            "job_name": "shohin-r12-dws-eos-dev",
            "partition": screen.SLURM_PARTITION,
            "state": "RUNNING",
            "alloc_cpus": screen.SLURM_CPUS_PER_TASK,
            "req_memory": screen.SLURM_MEMORY,
            "time_limit": screen.SLURM_TIME_LIMIT,
            "req_tres": _slurm_tres(),
            "alloc_tres": _slurm_tres(),
            "node_list": "test-host",
            "record_sha256": "9" * 64,
        },
        "gpu_binding": {
            "cgroup_version": "v1_devices_controller",
            "cgroup_path": f"/slurm/uid_{os.getuid()}/job_{job_id}/step_batch",
            "devices_list_path": "/sys/fs/cgroup/devices/test/devices.list",
            "devices_list_sha256": "8" * 64,
            "allowed_device_rules": ["c 195:0 rwm", "c 195:255 rwm"],
            "allocated_gpu_device": "/dev/nvidia0",
            "pci_bus_id": "00000000:3b:00.0",
            "pci_sysfs_path": "/sys/bus/pci/devices/0000:3b:00.0",
            "pci_vendor_id": "0x10de",
            "pci_device_id": "0x2331",
            "pci_class_id": "0x030200",
            "pci_identity_sha256": "5" * 64,
            "gpu_uuid": "GPU-test",
            "gpu_name": screen.REQUIRED_CUDA_DEVICE_NAME,
            "nvidia_smi_index": 0,
            "nvidia_smi_minor_number": 0,
            "gpu_minor": 0,
            "gpu_major": 195,
            "nvidia_control_devices": [
                {"name": "nvidiactl", "major": 195, "minor": 255}
            ],
            "concrete_physical_gpu_permissions": [{"major": 195, "minor": 0}],
            "mig_mode": "Disabled",
            "mig_devices_present": False,
            "nvidia_smi_query_sha256": "7" * 64,
            "nvidia_smi_list_sha256": "6" * 64,
            "cuda_visible_devices": "0",
            "slurm_job_gpus": "0",
            "selector_mapping": {
                "cuda_uuid": "GPU-test",
                "slurm_uuid": "GPU-test",
            },
        },
    }
    screen._validate_slurm_identity(identity)
    return identity


def _synthetic_linux_qualification_receipt(
    evaluator_sha256: str,
) -> dict[str, object]:
    source = {
        "sha256": evaluator_sha256,
        "byte_count": 1,
        "descriptor_kind": "sealed_memfd",
        "seals": screen._required_memfd_seals(),
    }
    stage = "after_complete_receipt_write"
    private_key = screen._linux_qualification_private_key(stage, evaluator_sha256)
    signing_key = screen.signing_key_record(private_key)
    broker_request = screen._linux_qualification_broker_request(
        stage, evaluator_sha256, signing_key, process_id=4202
    )
    broker_request_sha256 = screen.sha256_bytes(
        screen.stable_json_bytes(broker_request)
    )
    report = {
        "schema": screen.LINUX_QUALIFICATION_REPORT_SCHEMA,
        "stage": stage,
        "evaluator_source": source,
        "broker_request_sha256": broker_request_sha256,
        "brokered_signing_key": signing_key,
        "authority_boundary": screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
    }
    report_payload = screen.stable_json_bytes(report)
    marker_body = {
        "schema": screen.LINUX_QUALIFICATION_MARKER_SCHEMA,
        "stage": stage,
        "report_name": screen.LINUX_QUALIFICATION_REPORT_NAME,
        "report_sha256": screen.sha256_bytes(report_payload),
        "evaluator_sha256": evaluator_sha256,
        "broker_request_sha256": broker_request_sha256,
        "brokered_public_key_hex": signing_key["public_key_hex"],
        "authority_boundary": screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
    }
    marker = {
        **marker_body,
        "signature_hex": screen._ed25519_sign(
            private_key, screen.stable_json_bytes(marker_body)
        ).hex(),
    }
    marker_payload = screen.stable_json_bytes(marker)
    report_inode = {
        "device": 1,
        "inode": 101,
        "uid": os.getuid(),
        "mode": 0o444,
        "nlink": 1,
        "size": len(report_payload),
    }
    marker_inode = {
        "device": 1,
        "inode": 102,
        "uid": os.getuid(),
        "mode": 0o444,
        "nlink": 1,
        "size": len(marker_payload),
    }
    receipt_body = {
        "schema": screen.LINUX_QUALIFICATION_RECEIPT_SCHEMA,
        "status": screen.LINUX_QUALIFICATION_RECEIPT_STATUS,
        "stage": stage,
        "report_name": screen.LINUX_QUALIFICATION_REPORT_NAME,
        "marker_name": screen.LINUX_QUALIFICATION_MARKER_NAME,
        "receipt_name": screen.LINUX_QUALIFICATION_RECEIPT_NAME,
        "evaluator_source": source,
        "report_sha256": screen.sha256_bytes(report_payload),
        "marker_sha256": screen.sha256_bytes(marker_payload),
        "report_inode": report_inode,
        "marker_inode": marker_inode,
        "receipt_inode": {"device": 1, "inode": 103, "uid": os.getuid()},
        "durability_checks": {
            key: True for key in screen.LINUX_QUALIFICATION_RECEIPT_CHECK_KEYS
        },
        "broker_request_sha256": broker_request_sha256,
        "brokered_public_key_hex": signing_key["public_key_hex"],
        "authority_boundary": screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
    }
    receipt = {
        **receipt_body,
        "signature_hex": screen._ed25519_sign(
            private_key, screen.stable_json_bytes(receipt_body)
        ).hex(),
    }
    receipt_payload = screen.stable_json_bytes(receipt)
    cleanup_authorization = {
        "schema": "r12_linux_smoke_stale_cleanup_authorization_v1",
        "authority_boundary": screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
        "payload_sha256": "1" * 64,
        "ephemeral_public_key_hex": signing_key["public_key_hex"],
        "signature_hex": "2" * 128,
        "signature_verified": True,
    }
    cases = []
    for index, (case_stage, exit_code, replay) in enumerate(
        screen.LINUX_RECEIPT_CRASH_STAGES
    ):
        case_key = screen.signing_key_record(
            screen._linux_qualification_private_key(case_stage, evaluator_sha256)
        )
        case_request = screen._linux_qualification_broker_request(
            case_stage,
            evaluator_sha256,
            case_key,
            process_id=4200 + index,
        )
        cases.append(
            {
                "stage": case_stage,
                "child_exit_code": exit_code,
                "expected_replay": replay,
                "observed_replay": replay,
                "independent_report_marker_replay": "validated",
                "receipt_size": (0, 1, len(receipt_payload))[index],
                "evaluator_source": source,
                "broker_request_sha256": screen.sha256_bytes(
                    screen.stable_json_bytes(case_request)
                ),
            }
        )
    result = {
        "schema": screen.LINUX_QUALIFICATION_SCHEMA,
        "scientific_decode_executed": False,
        "gpu_required": False,
        "filesystem": "lustre",
        "evaluator_sha256": evaluator_sha256,
        "checks": list(screen.LINUX_QUALIFICATION_REQUIRED_CHECKS),
        "qualification_evidence": {
            "delegated_key_broker_transfer": {
                "authority_boundary": screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY,
                "request_sha256": broker_request_sha256,
                "request_process_id": 4202,
                "brokered_signing_key": signing_key,
                "scm_rights_descriptor_received": True,
            },
            "publisher_lease": {
                "lease": {
                    "accepted_name": "qualification-output.json",
                    "authorization_nonce": "a" * 64,
                    "directory_device": 1,
                    "directory_inode": 100,
                    "job_id": "1",
                    "owner_pid": 4199,
                },
                "concurrent_attempt": {
                    "child_pid": 4200,
                    "result": "exclusive_flock_rejected_concurrent_process",
                },
            },
            "signed_stale_cleanup": {
                "authorization": cleanup_authorization,
                "source_name": ".qualification-stale",
                "quarantine_name": ".qualification-stale-retained",
                "source_inode": 104,
                "quarantine_inode": 104,
                "pathname_unlink_used": False,
            },
            "foreign_inode_substitution": {
                "authorization": cleanup_authorization,
                "substituted_path": ".qualification-foreign-retained",
                "foreign_inode": 105,
                "foreign_payload_sha256": "3" * 64,
                "foreign_inode_preserved": True,
            },
            "held_evaluator_path_substitution": {
                "source_name": ".linux-smoke-held-evaluator-source.py",
                "retained_name": ".linux-smoke-held-evaluator-retained.py",
                "replacement_name": ".linux-smoke-held-evaluator-source.py",
                "source_inode": 106,
                "retained_inode": 106,
                "replacement_inode": 107,
                "held_sha256": evaluator_sha256,
                "retained_sha256": evaluator_sha256,
                "replacement_sha256": screen.sha256_bytes(b"hostile replacement"),
                "substitution_before_first_child": True,
                "original_inode_retained": True,
            },
            "directory_path_substitution": {
                "held_device": 1,
                "held_inode": 100,
                "replacement_rejected": True,
            },
        },
        "receipt_crash_cases": cases,
        "status": screen.LINUX_QUALIFICATION_STATUS,
        "claim_boundary": screen.LINUX_QUALIFICATION_CLAIM_BOUNDARY,
    }
    accepted_publication = {
        "broker_request": broker_request,
        "report": report,
        "marker": marker,
        "receipt": receipt,
    }
    return screen.build_replayable_linux_qualification_receipt(
        result, accepted_publication, private_key
    )


def _full_schema_fixture(tmp_path: Path) -> dict[str, object]:
    output_directory = (tmp_path / "accepted-output").resolve()
    output_directory.mkdir(mode=0o700)
    output_directory.chmod(0o700)
    directory_fd, output_record = screen.open_owned_output_directory(output_directory)
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").resolve()
    wrapper_sha256 = screen.sha256_regular_file(wrapper)
    python_path = str(Path(sys.executable).resolve())

    def executable_record(path: str, digit: str) -> dict[str, str]:
        return {"path": path, "sha256": digit * 64, "version": "test-1"}

    def package_record(name: str, digit: str) -> dict[str, object]:
        root = f"/test/{name}-site"
        installation_root = "/test"
        files = sorted(
            [
                {
                    "relative_path": f"{name}/__init__.py",
                    "path": f"{root}/{name}/__init__.py",
                    "sha256": digit * 64,
                    "byte_count": 10,
                    "device": 1,
                    "inode": int(digit, 16),
                    "uid": os.getuid(),
                    "mode": 0o644,
                    "nlink": 1,
                },
                {
                    "relative_path": f"{name}-test.dist-info/RECORD",
                    "path": f"{root}/{name}-test.dist-info/RECORD",
                    "sha256": digit * 64,
                    "byte_count": 20,
                    "device": 1,
                    "inode": int(digit, 16) + 10,
                    "uid": os.getuid(),
                    "mode": 0o644,
                    "nlink": 1,
                },
            ],
            key=lambda entry: entry["relative_path"],
        )
        record = {
            "distribution_name": name,
            "version": "test-1",
            "distribution_root": root,
            "installation_root": installation_root,
            "module_path": f"{root}/{name}/__init__.py",
            "record_path": f"{root}/{name}-test.dist-info/RECORD",
            "file_count": len(files),
            "files": files,
        }
        record["closure_sha256"] = screen.sha256_bytes(
            screen.stable_json_bytes(
                {
                    "distribution_name": name,
                    "distribution_root": root,
                    "installation_root": installation_root,
                    "files": files,
                    "version": "test-1",
                }
            )
        )
        return record

    python_startup = {
        "mode": screen.PYTHON_STARTUP_MODE,
        "flags": {
            "isolated": True,
            "no_site": True,
            "no_user_site": True,
            "ignore_environment": True,
            "dont_write_bytecode": True,
            "safe_path": True,
        },
        "startup_environment": {
            name: None for name in screen.PYTHON_STARTUP_ENVIRONMENT_KEYS
        },
        "site_modules_loaded": [],
        "processed_pth_files": [],
        "module_origins": [
            {
                "name": "os",
                "origin": "/test/python-stdlib/os.py",
                "file": "/test/python-stdlib/os.py",
                "cached": None,
            }
        ],
        "components": [
            {
                "reference": "/test/python-stdlib/os.py",
                "kind": "regular_file",
                "path": "/test/python-stdlib/os.py",
                "sha256": "9" * 64,
                "byte_count": 1,
                "device": 1,
                "inode": 9,
                "uid": os.getuid(),
                "mode": 0o444,
                "nlink": 1,
                "seals": None,
            }
        ],
        "search_path": [
            {
                "path": "/test/python-stdlib",
                "kind": "nonwritable_directory",
                "device": 1,
                "inode": 10,
                "uid": os.getuid(),
                "mode": 0o555,
                "ancestor": None,
                "sha256": None,
                "byte_count": None,
            }
        ],
    }
    python_startup["closure_sha256"] = screen.sha256_bytes(
        screen.stable_json_bytes(python_startup)
    )
    runtime = {
        "schema": screen.RUNTIME_IDENTITY_SCHEMA,
        "python": executable_record(python_path, "1"),
        "git": executable_record("/usr/bin/git", "2"),
        "scontrol": executable_record("/usr/bin/scontrol", "3"),
        "sacct": executable_record("/usr/bin/sacct", "7"),
        "nvidia_smi": executable_record("/usr/bin/nvidia-smi", "8"),
        "python_startup": python_startup,
        "packages": {
            "torch": package_record("torch", "4"),
            "tokenizers": package_record("tokenizers", "5"),
        },
        "backend": {
            "device": screen.DEVICE,
            "precision": screen.PRECISION,
            "sdpa_backend": screen.SDPA_BACKEND,
            "cublas_workspace_config": screen.CUBLAS_WORKSPACE_CONFIG,
            "deterministic_algorithms": True,
            "ld_preload": None,
            "dyld_insert_libraries": None,
            "ld_library_path": None,
            "preauthorization_native_libraries": {
                "platform": sys.platform,
                "source": (
                    "proc_self_maps_bound_device_inode_and_path"
                    if sys.platform.startswith("linux")
                    else "non_linux_development_only_no_proc_self_maps"
                ),
                "files": (
                    [
                        {
                            "path": "/test/libpython.so",
                            "sha256": "6" * 64,
                            "byte_count": 30,
                            "device": 1,
                            "inode": 30,
                            "uid": os.getuid(),
                            "mode": 0o755,
                            "nlink": 1,
                            "mapped_device_major": os.major(1),
                            "mapped_device_minor": os.minor(1),
                            "mapped_inode": 30,
                        }
                    ]
                    if sys.platform.startswith("linux")
                    else []
                ),
                "closure_sha256": "",
            },
            "coverage_claim": (
                "sealed_python_startup_full_record_distributions_and_"
                "mapped_device_inode_bound_native_maps"
            ),
        },
    }
    runtime["backend"]["preauthorization_native_libraries"]["closure_sha256"] = (
        screen.sha256_bytes(
            screen.stable_json_bytes(
                runtime["backend"]["preauthorization_native_libraries"]["files"]
            )
        )
    )
    source_files = {
        path: screen.sha256_regular_file(ROOT / path)
        for path in screen.RUNTIME_SOURCE_PATHS
    }
    manifest = {
        "schema": screen.RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "source_root": str((tmp_path / "reviewed-source").resolve()),
        "source_commit": "1" * 40,
        "files": source_files,
        "runtime": runtime,
    }
    manifest_path = (tmp_path / "external-runtime-manifest.json").resolve()
    manifest_bytes = screen.stable_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o444)
    manifest_info = manifest_path.lstat()
    manifest_file = {
        "device": manifest_info.st_dev,
        "inode": manifest_info.st_ino,
        "uid": manifest_info.st_uid,
        "mode": stat.S_IMODE(manifest_info.st_mode),
        "size": manifest_info.st_size,
    }
    generator_private_key = os.urandom(32)
    verifier_private_key = os.urandom(32)
    delegated_marker_private_key = os.urandom(32)
    while (
        len(
            {
                generator_private_key,
                verifier_private_key,
                delegated_marker_private_key,
            }
        )
        != 3
    ):
        verifier_private_key = os.urandom(32)
        delegated_marker_private_key = os.urandom(32)
    nonce = "a" * 64
    accepted_name = "accepted.json"
    slurm_identity = _slurm_identity(str(wrapper), wrapper_sha256)
    delegated_marker_record = screen.signing_key_record(delegated_marker_private_key)
    authority_key_path = (tmp_path / "test-only-authority.pub").resolve()
    authority_key_bytes = (screen.TEST_AUTHORITY_PUBLIC_KEY_HEX + "\n").encode()
    authority_key_path.write_bytes(authority_key_bytes)
    authority_key_path.chmod(0o444)
    authorization_payload = {
        "schema": screen.RUN_AUTHORIZATION_SCHEMA,
        "authority_scope": screen.TEST_AUTHORITY_SCOPE,
        "authority_key_id": "r12-test-authority-no-production-authority",
        "authority_public_key_sha256": screen.sha256_bytes(
            bytes.fromhex(screen.TEST_AUTHORITY_PUBLIC_KEY_HEX)
        ),
        "authorization_sequence": 1,
        "authorization_nonce": "a" * 64,
        "issued_at_utc": "2025-12-31T23:59:00+00:00",
        "not_before_utc": "2026-01-01T00:00:00+00:00",
        "expires_at_utc": "2026-01-02T00:00:00+00:00",
        "source_commit": manifest["source_commit"],
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": screen.sha256_bytes(manifest_bytes),
        "output_path": str(output_directory / accepted_name),
        "output_directory": output_record,
        "accepted_name": accepted_name,
        "frozen_inputs": {
            "checkpoint_path": str(screen.CHECKPOINT_PATH),
            "checkpoint_sha256": screen.EXPECTED_SHA256["checkpoint"],
            "tokenizer_path": str(screen.TOKENIZER_PATH),
            "tokenizer_sha256": screen.EXPECTED_SHA256["tokenizer"],
            "heldout_path": str(screen.HELDOUT_PATH),
            "heldout_sha256": screen.EXPECTED_SHA256["heldout"],
            "prereg_path": str(screen.PREREG_PATH),
            "prereg_sha256": source_files["R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md"],
        },
        "ordered_case_ids_sha256": screen.ORDERED_CASE_IDS_SHA256,
        "slurm_allocation": screen.slurm_authorization_projection(slurm_identity),
        "delegated_marker_public_key_hex": delegated_marker_record["public_key_hex"],
        "delegated_marker_private_key_sha256": delegated_marker_record[
            "private_key_sha256"
        ],
        "delegated_publication_scopes": list(screen.DELEGATED_PUBLICATION_SCOPES),
        "linux_qualification_receipt": _synthetic_linux_qualification_receipt(
            source_files["train/eval_dws_eos_suppressed_trace.py"]
        ),
        "stale_cleanup_entries": [],
    }
    authorization = screen.sign_run_authorization(
        authorization_payload, TEST_AUTHORITY_PRIVATE_KEY
    )
    authorization_path = (tmp_path / "test-only-run-authorization.json").resolve()
    authorization_bytes = screen.stable_json_bytes(authorization)
    authorization_path.write_bytes(authorization_bytes)
    authorization_path.chmod(0o444)
    context = {
        "schema": screen.WRAPPER_ACCEPTANCE_SCHEMA,
        "publication_state": screen.PRIVATE_CANDIDATE_STATE,
        "slurm_identity": slurm_identity,
        "wrapper_sha256": wrapper_sha256,
        "source_manifest_sha256": screen.sha256_bytes(manifest_bytes),
        "runtime_identity_sha256": screen.sha256_bytes(
            screen.stable_json_bytes(runtime)
        ),
        "nonce": nonce,
        "output_directory": output_record,
        "candidate_name": f".{accepted_name}.r12-candidate-12345-{nonce}",
        "accepted_name": accepted_name,
        "production_authority_key_file": screen.external_file_record(
            authority_key_path,
            authority_key_bytes,
            authority_key_path.lstat(),
        ),
        "run_authorization_file": screen.external_file_record(
            authorization_path,
            authorization_bytes,
            authorization_path.lstat(),
        ),
        "run_authorization_sha256": screen.sha256_bytes(authorization_bytes),
        "run_authorization": authorization,
        "generator_signing_key": screen.signing_key_record(generator_private_key),
        "verifier_signing_key": screen.signing_key_record(verifier_private_key),
        "delegated_marker_signing_key": delegated_marker_record,
        "sealed_generator": {
            "evaluator_sha256": source_files["train/eval_dws_eos_suppressed_trace.py"],
            "model_sha256": source_files["train/model.py"],
        },
    }
    cases = _synthetic_full_schema_cases()
    aggregate = screen.aggregate_report(cases)

    def source_record(digest: str) -> dict[str, object]:
        return {
            "descriptor_kind": "sealed_memfd",
            "sha256": digest,
            "byte_count": 1,
            "seals": screen._required_memfd_seals(),
        }

    report_body = {
        "schema": screen.OUTPUT_SCHEMA,
        "protocol": screen.PROTOCOL_ID,
        "development_only": True,
        "claim_boundary": screen.CLAIM_BOUNDARY,
        "frozen_contract": screen.frozen_contract(),
        "execution": {
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "finished_at_utc": "2026-01-01T00:01:00+00:00",
            "input_paths": screen.frozen_input_paths(),
            "verified_input_sha256": dict(screen.EXPECTED_SHA256),
            "checkpoint_step": screen.EXPECTED_CHECKPOINT_STEP,
            "ordered_case_ids": [case["case_id"] for case in cases],
            "runtime_source_manifest": {
                "schema": screen.RUNTIME_SOURCE_MANIFEST_SCHEMA,
                "path": str(manifest_path),
                "sha256": screen.sha256_bytes(manifest_bytes),
                "manifest_file": manifest_file,
                "source_root": manifest["source_root"],
                "source_commit": manifest["source_commit"],
                "files": source_files,
                "runtime": runtime,
                "git_status": "clean",
                "git_show_byte_equality": True,
            },
            "source_execution": {
                "mode": screen.SOURCE_EXECUTION_MODE,
                "python_startup_mode": screen.PYTHON_STARTUP_MODE,
                "evaluator": source_record(
                    source_files["train/eval_dws_eos_suppressed_trace.py"]
                ),
                "model": source_record(source_files["train/model.py"]),
            },
            "runtime_observation": _runtime_observation(
                "generator_post_decode", python_path
            ),
            "device": screen.DEVICE,
            "precision": screen.PRECISION,
            "python": "3.test",
            "torch": "test",
            "cuda_runtime": "12.test",
            "cuda_visible_devices": "0",
            "device_name": screen.REQUIRED_CUDA_DEVICE_NAME,
            "device_capability": list(screen.REQUIRED_CUDA_DEVICE_CAPABILITY),
            "device_total_memory_bytes": screen.REQUIRED_CUDA_MEMORY_MIN_BYTES,
            "device_uuid": "GPU-test",
            "visible_cuda_device_count": 1,
            "cublas_workspace_config": screen.CUBLAS_WORKSPACE_CONFIG,
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "cuda_matmul_tf32_allowed": False,
            "cudnn_tf32_allowed": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
            "sdpa_backend": screen.SDPA_BACKEND,
            "sdpa_math_enabled": True,
            "sdpa_flash_enabled": False,
            "sdpa_mem_efficient_enabled": False,
            "sdpa_cudnn_enabled": False,
            "sdpa_bf16_probe_bitwise_equal": True,
            "seed": 0,
        },
        "aggregate": aggregate,
        "cases": cases,
        "adjudication": {
            "field_screen_execution": "development_go",
            "full_state_recurrence": "no_go",
            "carry_target_switch_noncompensatory_veto": aggregate[
                "carry_target_switch_global_veto"
            ],
            "compound_fresh_reencoding_screen_pass": aggregate[
                "fresh_latest_reencoding"
            ]["compound_fresh_reencoding_screen_pass"],
            "promotion_authorized": False,
        },
        "wrapper_acceptance": context,
    }
    report = screen.attach_generator_attestation(
        report_body, context, generator_private_key
    )
    screen.validate_report_schema(report, context, live_custody=False)
    return {
        "directory_fd": directory_fd,
        "output_directory": output_directory,
        "context": context,
        "report": report,
        "manifest_bytes": manifest_bytes,
        "manifest_info": manifest_info,
        "generator_private_key": generator_private_key,
        "verifier_private_key": verifier_private_key,
        "delegated_marker_private_key": delegated_marker_private_key,
    }


def _resign_fixture_report(
    report: dict[str, object], fixture: dict[str, object]
) -> dict[str, object]:
    context = fixture["context"]
    generator_private_key = fixture["generator_private_key"]
    assert isinstance(context, dict)
    assert isinstance(generator_private_key, bytes)
    body = {
        key: value for key, value in report.items() if key != "generator_attestation"
    }
    return screen.attach_generator_attestation(body, context, generator_private_key)


def test_external_authorization_is_not_self_rooted_and_binds_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_schema_fixture(tmp_path)
    context = fixture["context"]
    assert isinstance(context, dict)
    authorization = context["run_authorization"]
    assert isinstance(authorization, dict)
    screen.validate_run_authorization(
        authorization,
        expected_slurm=context["slurm_identity"],
        expected_output_directory=context["output_directory"],
        expected_accepted_name=context["accepted_name"],
        expected_source_manifest_sha256=context["source_manifest_sha256"],
        expected_delegated_marker_key=context["delegated_marker_signing_key"],
        require_current=False,
    )
    with pytest.raises(screen.ContractError, match="not currently valid"):
        screen.validate_run_authorization(
            authorization,
            expected_slurm=context["slurm_identity"],
            expected_output_directory=context["output_directory"],
            expected_accepted_name=context["accepted_name"],
            expected_source_manifest_sha256=context["source_manifest_sha256"],
            expected_delegated_marker_key=context["delegated_marker_signing_key"],
            require_current=True,
            now=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    with monkeypatch.context() as scoped:
        scoped.setattr(screen, "ALLOW_TEST_AUTHORITY", False)
        with pytest.raises(screen.ContractError, match="production authority"):
            screen.validate_run_authorization(
                authorization,
                expected_slurm=context["slurm_identity"],
                expected_output_directory=context["output_directory"],
                expected_accepted_name=context["accepted_name"],
                expected_source_manifest_sha256=context["source_manifest_sha256"],
                expected_delegated_marker_key=context["delegated_marker_signing_key"],
                require_current=False,
            )

    forged_payload = screen.run_authorization_payload(authorization)
    forged_payload["authority_scope"] = screen.PRODUCTION_AUTHORITY_SCOPE
    forged_payload["authority_key_id"] = screen.PRODUCTION_AUTHORITY_KEY_ID
    forged_payload["authority_public_key_sha256"] = (
        screen.PRODUCTION_AUTHORITY_PUBLIC_KEY_SHA256
    )
    forged = screen.sign_run_authorization(forged_payload, TEST_AUTHORITY_PRIVATE_KEY)
    with pytest.raises(screen.ContractError, match="signature does not verify"):
        screen.validate_run_authorization(
            forged,
            expected_slurm=context["slurm_identity"],
            expected_output_directory=context["output_directory"],
            expected_accepted_name=context["accepted_name"],
            expected_source_manifest_sha256=context["source_manifest_sha256"],
            expected_delegated_marker_key=context["delegated_marker_signing_key"],
            require_current=False,
        )

    redirected_payload = screen.run_authorization_payload(authorization)
    redirected_payload["output_path"] = "/tmp/unauthorized-output.json"
    redirected = screen.sign_run_authorization(
        redirected_payload, TEST_AUTHORITY_PRIVATE_KEY
    )
    with pytest.raises(screen.ContractError, match="cross-binding"):
        screen.validate_run_authorization(
            redirected,
            expected_slurm=context["slurm_identity"],
            expected_output_directory=context["output_directory"],
            expected_accepted_name=context["accepted_name"],
            expected_source_manifest_sha256=context["source_manifest_sha256"],
            expected_delegated_marker_key=context["delegated_marker_signing_key"],
            require_current=False,
        )
    assert authorization["delegated_publication_scopes"] == list(
        screen.DELEGATED_PUBLICATION_SCOPES
    )
    narrowed_payload = screen.run_authorization_payload(authorization)
    narrowed_payload["delegated_publication_scopes"] = [
        screen.DELEGATED_PUBLICATION_SCOPES[0]
    ]
    narrowed = screen.sign_run_authorization(
        narrowed_payload, TEST_AUTHORITY_PRIVATE_KEY
    )
    with pytest.raises(screen.ContractError, match="cross-binding"):
        screen.validate_run_authorization(
            narrowed,
            expected_slurm=context["slurm_identity"],
            expected_output_directory=context["output_directory"],
            expected_accepted_name=context["accepted_name"],
            expected_source_manifest_sha256=context["source_manifest_sha256"],
            expected_delegated_marker_key=context["delegated_marker_signing_key"],
            require_current=False,
        )
    os.close(fixture["directory_fd"])


def test_authorization_and_report_bind_replayable_linux_qualification_receipt(
    tmp_path: Path,
) -> None:
    fixture = _full_schema_fixture(tmp_path)
    context = fixture["context"]
    report = fixture["report"]
    assert isinstance(context, dict)
    assert isinstance(report, dict)
    authorization = context["run_authorization"]
    assert isinstance(authorization, dict)
    qualification_receipt = authorization["linux_qualification_receipt"]
    assert isinstance(qualification_receipt, dict)
    try:
        screen.validate_replayable_linux_qualification_receipt(qualification_receipt)
        assert (
            report["wrapper_acceptance"]["run_authorization"][
                "linux_qualification_receipt"
            ]
            == qualification_receipt
        )

        missing = copy.deepcopy(authorization)
        missing.pop("linux_qualification_receipt")
        with pytest.raises(screen.ContractError, match="schema keys differ"):
            screen.validate_run_authorization(
                missing,
                expected_slurm=context["slurm_identity"],
                expected_output_directory=context["output_directory"],
                expected_accepted_name=context["accepted_name"],
                expected_source_manifest_sha256=context["source_manifest_sha256"],
                expected_delegated_marker_key=context["delegated_marker_signing_key"],
                require_current=False,
            )

        hostile_payload = screen.run_authorization_payload(authorization)
        hostile_receipt = copy.deepcopy(qualification_receipt)
        hostile_receipt["accepted_publication"]["receipt"]["durability_checks"][
            "publication_parent_fsync_complete"
        ] = False
        evaluator_sha256 = hostile_receipt["qualification_result"]["evaluator_sha256"]
        qualification_private_key = screen._linux_qualification_private_key(
            "after_complete_receipt_write", evaluator_sha256
        )
        outer_payload = {
            key: value
            for key, value in hostile_receipt.items()
            if key != "signature_hex"
        }
        hostile_receipt["signature_hex"] = screen._ed25519_sign(
            qualification_private_key, screen.stable_json_bytes(outer_payload)
        ).hex()
        hostile_payload["linux_qualification_receipt"] = hostile_receipt
        hostile_authorization = screen.sign_run_authorization(
            hostile_payload, TEST_AUTHORITY_PRIVATE_KEY
        )
        with pytest.raises(screen.ContractError, match="durability checks differ"):
            screen.validate_run_authorization(
                hostile_authorization,
                expected_slurm=context["slurm_identity"],
                expected_output_directory=context["output_directory"],
                expected_accepted_name=context["accepted_name"],
                expected_source_manifest_sha256=context["source_manifest_sha256"],
                expected_delegated_marker_key=context["delegated_marker_signing_key"],
                require_current=False,
            )
    finally:
        os.close(fixture["directory_fd"])


def test_authorization_verification_does_not_import_third_party_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_schema_fixture(tmp_path)
    context = fixture["context"]
    assert isinstance(context, dict)
    monkeypatch.setattr(
        screen.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("unexpected import")),
    )
    try:
        screen.validate_run_authorization(
            context["run_authorization"],
            expected_slurm=context["slurm_identity"],
            expected_output_directory=context["output_directory"],
            expected_accepted_name=context["accepted_name"],
            expected_source_manifest_sha256=context["source_manifest_sha256"],
            expected_delegated_marker_key=context["delegated_marker_signing_key"],
            require_current=False,
        )
    finally:
        os.close(fixture["directory_fd"])


def test_ed25519_rejects_every_canonical_small_order_encoding() -> None:
    small_order_encodings = (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000080",
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39cc3c6fc1d095dba7a037",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39cc3c6fc1d095dba7a0b7",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85",
    )
    for encoded_hex in small_order_encodings:
        encoded = bytes.fromhex(encoded_hex)
        with pytest.raises(ValueError):
            screen._ed25519_decode_point(encoded)
        assert not screen._ed25519_verify(encoded, encoded + bytes(32), b"forged")


def test_ed25519_rejects_alternate_identity_and_noncanonical_field_encodings() -> None:
    canonical_identity = (1).to_bytes(32, "little")
    alternate_zero_sign_identity = bytearray(canonical_identity)
    alternate_zero_sign_identity[31] |= 0x80
    alternate_field_identity = (screen._ED25519_Q + 1).to_bytes(32, "little")
    alternate_field_and_sign_identity = bytearray(alternate_field_identity)
    alternate_field_and_sign_identity[31] |= 0x80
    noncanonical_field_encodings = (
        screen._ED25519_Q.to_bytes(32, "little"),
        ((1 << 255) - 1).to_bytes(32, "little"),
    )
    with pytest.raises(ValueError, match="identity point"):
        screen._ed25519_decode_point(canonical_identity)
    with pytest.raises(ValueError, match="noncanonical zero sign"):
        screen._ed25519_decode_point(bytes(alternate_zero_sign_identity))
    for encoded in (alternate_field_identity, bytes(alternate_field_and_sign_identity)):
        with pytest.raises(ValueError, match="noncanonical"):
            screen._ed25519_decode_point(encoded)
    for encoded in noncanonical_field_encodings:
        with pytest.raises(ValueError, match="noncanonical"):
            screen._ed25519_decode_point(encoded)


def test_ed25519_scalar_encoding_bounds_are_strict() -> None:
    assert screen._ed25519_decode_scalar(
        (screen._ED25519_L - 1).to_bytes(32, "little")
    ) == (screen._ED25519_L - 1)
    for scalar in (screen._ED25519_L, screen._ED25519_L + 1, (1 << 256) - 1):
        encoded = scalar.to_bytes(32, "little")
        with pytest.raises(ValueError, match="out of range"):
            screen._ed25519_decode_scalar(encoded)

    private_key = bytes(range(32))
    message = b"canonical-scalar-boundary"
    public_key = screen._ed25519_public_key(private_key)
    signature = screen._ed25519_sign(private_key, message)
    assert screen._ed25519_verify(public_key, signature, message)
    assert not screen._ed25519_verify(
        public_key,
        signature[:32] + screen._ED25519_L.to_bytes(32, "little"),
        message,
    )


def test_delegated_key_arrives_only_over_post_exec_broker_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = bytes(reversed(range(32)))
    expected = screen.signing_key_record(key)
    request = screen.build_delegated_key_broker_request(
        authorization_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        wrapper_sha256="4" * 64,
        runtime_identity={
            "python": {"path": sys.executable, "sha256": "5" * 64},
            "python_startup": {"mode": screen.PYTHON_STARTUP_MODE},
        },
        expected_signing_key=expected,
    )
    key_path = tmp_path / "broker-key.bin"
    key_path.write_bytes(key)
    key_fd = os.open(key_path, os.O_RDONLY)
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    server_errors: list[BaseException] = []

    def fake_read_sealed(descriptor: int, observed: dict[str, object]) -> bytes:
        assert observed == expected
        assert os.pread(descriptor, 33, 0) == key
        return key

    monkeypatch.setattr(screen, "read_sealed_signing_key", fake_read_sealed)

    def serve() -> None:
        try:
            request_payload = server.recv(1 << 20)
            assert request_payload == screen.stable_json_bytes(request)
            response = screen.stable_json_bytes(
                {
                    "schema": screen.DELEGATED_KEY_BROKER_RESPONSE_SCHEMA,
                    "request_sha256": screen.sha256_bytes(request_payload),
                }
            )
            descriptors = array.array("i", [key_fd])
            server.sendmsg(
                [response],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())],
            )
        except BaseException as error:
            server_errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    received_fd = None
    try:
        received_fd, received_key = screen.receive_delegated_signing_key_from_broker(
            client.fileno(), request, expected
        )
        assert received_key == key
        assert not os.get_inheritable(received_fd)
    finally:
        thread.join(timeout=5)
        if received_fd is not None:
            os.close(received_fd)
        client.close()
        server.close()
        os.close(key_fd)
    assert not thread.is_alive()
    assert server_errors == []


def test_linux_broker_python_identity_is_canonical_under_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_python = Path(sys.executable).resolve(strict=True)
    symlink_python = tmp_path / "python-symlink"
    symlink_python.symlink_to(resolved_python)
    signing_key = screen.signing_key_record(bytes(range(32)))

    monkeypatch.setattr(screen.sys, "executable", str(symlink_python))
    symlink_request = screen._linux_qualification_broker_request(
        "after_complete_receipt_write",
        "a" * 64,
        signing_key,
        process_id=1234,
    )
    monkeypatch.setattr(screen.sys, "executable", str(resolved_python))
    resolved_request = screen._linux_qualification_broker_request(
        "after_complete_receipt_write",
        "a" * 64,
        signing_key,
        process_id=1234,
    )

    assert symlink_request == resolved_request
    assert symlink_request["python_executable"] == {
        "path": str(resolved_python),
        "sha256": screen.sha256_regular_file(resolved_python),
    }
    assert symlink_request["python_executable"]["sha256"] != screen.sha256_bytes(
        os.fsencode(symlink_python)
    )


def test_broker_socket_identity_is_revalidated_after_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_key = screen.signing_key_record(bytes(range(32)))
    request = screen.build_delegated_key_broker_request(
        authorization_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        wrapper_sha256="4" * 64,
        runtime_identity={
            "python": {"path": sys.executable},
            "python_startup": {"mode": screen.PYTHON_STARTUP_MODE},
        },
        expected_signing_key=expected_key,
    )
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    request_payload = screen.stable_json_bytes(request)
    response = screen.stable_json_bytes(
        {
            "schema": screen.DELEGATED_KEY_BROKER_RESPONSE_SCHEMA,
            "request_sha256": screen.sha256_bytes(request_payload),
        }
    )
    server_errors: list[BaseException] = []

    def serve() -> None:
        try:
            assert server.recv(1 << 20) == request_payload
            server.send(response)
        except BaseException as error:
            server_errors.append(error)

    real_fstat = screen.os.fstat
    broker_inode = real_fstat(client.fileno()).st_ino
    broker_observations = 0

    def substitute_last_socket_stat(descriptor: int):
        nonlocal broker_observations
        observed = real_fstat(descriptor)
        if stat.S_ISSOCK(observed.st_mode) and observed.st_ino == broker_inode:
            broker_observations += 1
            if broker_observations == 3:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino + 1,
                    st_mode=observed.st_mode,
                    st_nlink=observed.st_nlink,
                    st_uid=observed.st_uid,
                )
        return observed

    thread = threading.Thread(target=serve)
    thread.start()
    monkeypatch.setattr(screen.os, "fstat", substitute_last_socket_stat)
    try:
        with pytest.raises(screen.ContractError, match="changed during use"):
            screen.receive_delegated_signing_key_from_broker(
                client.fileno(), request, expected_key
            )
    finally:
        thread.join(timeout=5)
        client.close()
        server.close()
    assert not thread.is_alive()
    assert server_errors == []


def test_held_descriptor_rejects_mode_link_and_same_inode_byte_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "held-publication.json"
    original_payload = b'{"held":true}\n'
    path.write_bytes(original_payload)
    path.chmod(0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        _, expected_info = screen._read_exact_descriptor_bytes(
            descriptor, "test held descriptor", expected_payload=original_payload
        )
        path.chmod(0o644)
        with pytest.raises(screen.ContractError, match="identity differs"):
            screen._read_exact_descriptor_bytes(
                descriptor,
                "test held descriptor",
                expected_info=expected_info,
                expected_payload=original_payload,
            )
        path.chmod(0o444)

        alias = tmp_path / "held-publication.alias"
        os.link(path, alias)
        with pytest.raises(screen.ContractError, match="identity differs"):
            screen._read_exact_descriptor_bytes(
                descriptor,
                "test held descriptor",
                expected_info=expected_info,
                expected_payload=original_payload,
            )
        alias.unlink()

        replacement_payload = b'{"held":null}\n'
        assert len(replacement_payload) == len(original_payload)
        path.chmod(0o644)
        writer = os.open(path, os.O_WRONLY)
        try:
            assert os.pwrite(writer, replacement_payload, 0) == len(replacement_payload)
            os.fsync(writer)
        finally:
            os.close(writer)
        path.chmod(0o444)
        os.utime(path, ns=(expected_info.st_atime_ns, expected_info.st_mtime_ns))
        with pytest.raises(screen.ContractError, match="bytes differ"):
            screen._read_exact_descriptor_bytes(
                descriptor,
                "test held descriptor",
                expected_info=expected_info,
                expected_payload=original_payload,
            )
    finally:
        os.close(descriptor)


def test_rollback_quarantine_preserves_swap_between_validation_and_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd, _directory_record = screen.open_owned_output_directory(tmp_path)
    source_name = "accepted.json"
    displaced_name = "accepted.original"
    original_payload = b"original"
    foreign_payload = b"foreign!"
    source_path = tmp_path / source_name
    source_path.write_bytes(original_payload)
    source_path.chmod(0o444)
    descriptor = os.open(source_path, os.O_RDONLY)
    expected_info = os.fstat(descriptor)
    nonce = "a" * 64

    def swap_during_rename(
        old_directory_fd: int,
        old_name: str,
        new_directory_fd: int,
        new_name: str,
    ) -> None:
        assert old_name == source_name
        os.rename(
            old_name,
            displaced_name,
            src_dir_fd=old_directory_fd,
            dst_dir_fd=old_directory_fd,
        )
        foreign_descriptor = os.open(
            old_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=old_directory_fd,
        )
        try:
            assert os.write(foreign_descriptor, foreign_payload) == len(foreign_payload)
            os.fsync(foreign_descriptor)
        finally:
            os.close(foreign_descriptor)
        _test_rename_noreplace_at(
            old_directory_fd, old_name, new_directory_fd, new_name
        )

    monkeypatch.setattr(screen, "rename_noreplace_at", swap_during_rename)
    try:
        quarantined, quarantine_name = screen._quarantine_held_entry_if_exact(
            directory_fd,
            source_name,
            nonce,
            source_name,
            descriptor,
            expected_info,
            original_payload,
            "adversarial rollback",
        )
        assert not quarantined
        assert quarantine_name is not None
        assert (tmp_path / displaced_name).read_bytes() == original_payload
        assert (tmp_path / quarantine_name).read_bytes() == foreign_payload
    finally:
        os.close(descriptor)
        os.close(directory_fd)


def _fake_distribution_fixture(
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict[str, Path]]:
    root = (tmp_path / "site-packages").resolve()
    package = root / "demo"
    metadata = root / "demo-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    paths = {
        "module": package / "__init__.py",
        "transitive_python": package / "worker.py",
        "native": package / "_native.so",
        "record": metadata / "RECORD",
    }
    paths["module"].write_bytes(b"from .worker import VALUE\n")
    paths["transitive_python"].write_bytes(b"VALUE = 7\n")
    paths["native"].write_bytes(b"ELF-test-native-bytes\n")
    relative_paths = {
        "module": "demo/__init__.py",
        "transitive_python": "demo/worker.py",
        "native": "demo/_native.so",
        "record": "demo-1.0.dist-info/RECORD",
    }
    rows = []
    for key in ("module", "transitive_python", "native"):
        payload = paths[key].read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        rows.append(
            f"{relative_paths[key]},sha256={digest.decode('ascii')},{len(payload)}"
        )
    rows.append(f"{relative_paths['record']},,")
    paths["record"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    distribution = SimpleNamespace(
        files=[Path(value) for value in relative_paths.values()],
        version="1.0",
        locate_file=lambda value: root / str(value),
    )
    return distribution, paths


def test_distribution_identity_binds_transitive_python_and_native_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    distribution, paths = _fake_distribution_fixture(tmp_path)
    monkeypatch.setattr(
        screen.importlib.metadata,
        "distribution",
        lambda name: distribution if name == "demo" else None,
    )
    monkeypatch.setattr(screen.sys, "prefix", str(tmp_path.resolve()))
    identity = screen._distribution_identity("demo", "demo/__init__.py")
    assert identity["file_count"] == 4
    assert [entry["relative_path"] for entry in identity["files"]] == sorted(
        entry["relative_path"] for entry in identity["files"]
    )
    original_python = paths["transitive_python"].read_bytes()
    paths["transitive_python"].write_bytes(b"VALUE = 8\n")
    with pytest.raises(screen.ContractError, match="RECORD SHA-256"):
        screen._distribution_identity("demo", "demo/__init__.py")
    paths["transitive_python"].write_bytes(original_python)
    paths["native"].write_bytes(b"ELF-substituted-native\n")
    with pytest.raises(screen.ContractError, match="RECORD SHA-256|RECORD size"):
        screen._distribution_identity("demo", "demo/__init__.py")
    activation = inspect.getsource(screen.activate_pinned_runtime_packages)
    assert (
        activation.index("_rehash_manifest_distribution")
        < activation.index("sys.path.append")
        < activation.index('importlib.import_module("torch")')
    )


def test_distribution_identity_rejects_record_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    distribution, paths = _fake_distribution_fixture(tmp_path)
    distribution.files.append(Path("../../escape.py"))
    paths["record"].write_text(
        paths["record"].read_text(encoding="utf-8") + "../../escape.py,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        screen.importlib.metadata, "distribution", lambda _name: distribution
    )
    monkeypatch.setattr(screen.sys, "prefix", str(tmp_path.resolve()))
    with pytest.raises(screen.ContractError, match="escapes"):
        screen._distribution_identity("demo", "demo/__init__.py")


def test_preauthorization_native_closure_detects_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = (tmp_path / "libauthority-support.so").resolve()
    native.write_bytes(b"native-authority-support-v1\n")
    native.chmod(0o444)
    mapped_info = native.stat()
    maps = (
        "1000-2000 r-xp 00000000 "
        f"{os.major(mapped_info.st_dev):x}:{os.minor(mapped_info.st_dev):x} "
        f"{mapped_info.st_ino} {native}\n"
    ).encode()
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if str(path) == "/proc/self/maps":
            return maps
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(screen.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    first = screen._preauthorization_native_library_identity()
    native.chmod(0o644)
    native.write_bytes(b"native-authority-support-v2\n")
    native.chmod(0o444)
    second = screen._preauthorization_native_library_identity()
    assert first["closure_sha256"] != second["closure_sha256"]
    native.unlink()
    native.write_bytes(b"replacement-path-inode\n")
    native.chmod(0o444)
    with pytest.raises(screen.ContractError, match="mapped device/inode"):
        screen._preauthorization_native_library_identity()


def test_isolated_cli_disables_site_pth_and_customization_startup(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "sitecustomize-ran"
    customization = tmp_path / "sitecustomize.py"
    customization.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    )
    (tmp_path / "hostile.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('pth')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-W",
            "error",
            str(ROOT / "train/eval_dws_eos_suppressed_trace.py"),
            "--help",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    assert not marker.exists()


def test_later_runtime_snapshot_binds_mapped_device_inode(
    tmp_path: Path,
) -> None:
    mapped = (tmp_path / "later-loaded-object.so").resolve()
    mapped.write_bytes(b"mapped-v1\n")
    mapped.chmod(0o444)
    info = mapped.stat()
    binding = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    record = screen._mapped_regular_file_closure_record(
        mapped, binding, label="later runtime mapping"
    )
    assert record["mapped_inode"] == info.st_ino
    assert "_mapped_regular_file_closure_record" in inspect.getsource(
        screen.observe_executed_runtime
    ) or "_mapped_regular_file_closure_record" in inspect.getsource(
        screen._loaded_runtime_paths
    )
    mapped.unlink()
    mapped.write_bytes(b"mapped-path-replacement\n")
    mapped.chmod(0o444)
    with pytest.raises(screen.ContractError, match="mapped device/inode"):
        screen._mapped_regular_file_closure_record(
            mapped, binding, label="later runtime mapping"
        )


def test_field_screen_missing_work_requires_independently_unavailable_boundary(
    tmp_path: Path,
) -> None:
    fixture = _full_schema_fixture(tmp_path)
    report = copy.deepcopy(fixture["report"])
    case = report["cases"][0]
    prompt_ids = tuple(case["prompt"]["token_ids"])
    initial = screen.parse_dws_line(case["initial_state"])
    assert initial is not None
    oracle = screen.reconstruct_oracle_posthoc(initial)
    emitted_line = screen.canonical_dws_state(oracle[0])
    tokenizer = RealTokenizer.from_str(
        screen.read_verified_bytes(
            screen.TOKENIZER_PATH, screen.EXPECTED_SHA256["tokenizer"]
        ).decode("utf-8")
    )
    emitted_ids = tuple(tokenizer.encode(emitted_line).ids)
    raw = screen.RawDecode(
        mode=screen.DecodeMode.ORDINARY_EOS_STOP.value,
        prompt_token_ids=prompt_ids,
        prompt_token_count=len(prompt_ids),
        generated_token_ids=(*emitted_ids, screen.EOS_TOKEN_ID),
        stop_reason="eos",
        eos_mask_applied_positions=(),
        eos_events=(
            screen.RawEosEvent(
                generated_index=len(emitted_ids),
                absolute_token_position=len(prompt_ids) + len(emitted_ids),
                eos_logit=10.0,
                non_eos_argmax_token_id=211,
                non_eos_argmax_logit=9.0,
                replacement_token_id=None,
                replacement_raw_logit=None,
            ),
        ),
    )
    case["primary_arms"][screen.DecodeMode.ORDINARY_EOS_STOP.value] = (
        screen.raw_decode_report_posthoc(raw, tokenizer, initial, oracle)
    )
    report["aggregate"] = screen.aggregate_report(report["cases"])
    report = _resign_fixture_report(report, fixture)
    try:
        with pytest.raises(screen.ContractError, match="work is missing"):
            screen.validate_report_schema(
                report, fixture["context"], live_custody=False
            )
    finally:
        os.close(fixture["directory_fd"])


def test_recursive_exact_types_and_prompt_token_sequences_are_fail_closed() -> None:
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class IntSubclass(int):
        pass

    for hostile in (
        DictSubclass({"ok": True}),
        {"items": ListSubclass([1])},
        {"count": IntSubclass(1)},
        {"number": float("nan")},
        {"number": float("inf")},
    ):
        with pytest.raises(screen.ContractError):
            screen._require_plain_json_tree(hostile, "hostile")

    tokenizer = TinyTokenizer()
    initial = screen.parse_dws_line("dws:op=add;w=1;p=0;c=0;a=1;b=2;r=0;z=0")
    assert initial is not None
    oracle = screen.reconstruct_oracle_posthoc(initial)
    raw = screen.decode_cached_greedy(
        screen.DecodeRequest(
            model=TinyModel([0]),
            prompt_token_ids=(12, 13, 14),
            device="cpu",
            max_new_tokens=screen.MAX_NEW_TOKENS,
            mode=screen.DecodeMode.EOS_MASKED_ARGMAX,
            eos_token_id=0,
        )
    )
    record = screen.raw_decode_report_posthoc(raw, tokenizer, initial, oracle)
    substituted = copy.deepcopy(record)
    substituted["decode_prompt_token_ids"] = [12, 99, 14]
    substituted["decode_prompt_token_ids_sha256"] = screen.token_ids_sha256(
        substituted["decode_prompt_token_ids"]
    )
    with pytest.raises(screen.ContractError, match="token sequence"):
        screen._validate_decode_common(
            substituted,
            "arm",
            expected_mode=screen.DecodeMode.EOS_MASKED_ARGMAX.value,
            expected_prompt_token_ids=(12, 13, 14),
            tokenizer=tokenizer,
        )
    wrong_bool = copy.deepcopy(record)
    wrong_bool["generated_token_count"] = True
    with pytest.raises(screen.ContractError, match="nonnegative integer"):
        screen._validate_decode_common(wrong_bool, "arm", tokenizer=tokenizer)
    wrong_float_alias = copy.deepcopy(record)
    wrong_float_alias["eos_events"][0]["eos_logit"] = 10
    with pytest.raises(screen.ContractError, match="finite float"):
        screen._validate_decode_common(wrong_float_alias, "arm", tokenizer=tokenizer)


def test_full_report_replay_rejects_int_float_aggregate_alias(tmp_path: Path) -> None:
    fixture = _full_schema_fixture(tmp_path)
    report = copy.deepcopy(fixture["report"])
    report["aggregate"]["field_by_clock"]["eos_masked_argmax"]["nominal_second_exact"][
        "rate"
    ] = 0
    report = _resign_fixture_report(report, fixture)
    try:
        with pytest.raises(screen.ContractError, match="aggregate"):
            screen.validate_report_schema(
                report, fixture["context"], live_custody=False
            )
    finally:
        os.close(fixture["directory_fd"])


def test_linux_qualification_surface_is_explicit_and_non_scientific() -> None:
    source = inspect.getsource(screen.run_linux_publication_qualification)
    worker = inspect.getsource(screen._run_linux_receipt_crash_worker)
    subprocess_runner = inspect.getsource(screen._run_linux_receipt_crash_subprocess)
    replay = inspect.getsource(screen._read_and_validate_linux_qualification_receipt)
    assert "rename_noreplace_at" in source
    assert "os.memfd_create" in source
    assert "libc.prctl" in source
    assert "directory_fsync_failure_rollback" in source
    assert "acquire_publisher_lease" in source
    assert "_linux_smoke_concurrent_publisher_lease" in source
    assert "_linux_smoke_signed_cleanup_authorization" in source
    assert "cleanup_stale_publication_entries" in source
    assert "foreign_inode_after_quarantine_rename_preserved" in source
    assert "held_directory_pathname_substitution_rejected" in source
    assert "_run_linux_receipt_crash_consistency_cases" in source
    assert "_linux_qualification_broker_exchange" in source
    assert "receive_delegated_signing_key_from_broker" in inspect.getsource(
        screen._linux_qualification_broker_exchange
    )
    assert "publish_accepted_bundle_exclusive" in worker
    assert "qualification_contract=contract" in worker
    assert "during_partial_receipt_write" in worker
    assert "os._exit" in worker
    publisher = inspect.getsource(screen.publish_accepted_bundle_exclusive)
    assert "_open_durable_o_sync_receipt_slot" in publisher
    assert "_write_complete_o_sync_receipt" in publisher
    assert "_read_and_validate_held_publication_triplet" in publisher
    assert "subprocess.Popen" in subprocess_runner
    assert '"-S"' in subprocess_runner
    assert "-B" in subprocess_runner
    assert "-c" in subprocess_runner
    assert "HELD_EVALUATOR_DESCRIPTOR_BOOTSTRAP" in subprocess_runner
    assert "_held_descriptor_execution_path" in subprocess_runner
    assert "pass_fds" in subprocess_runner
    assert "no valid complete O_SYNC receipt" in replay
    assert "_read_and_validate_held_publication_triplet" in replay
    assert "symlink_rejected" in source
    assert "hardlink_rejected" in source
    assert '"lustre"' in source
    assert "torch" not in source
    assert '"linux-smoke"' in inspect.getsource(screen.main)


def test_linux_smoke_wires_first_child_evaluator_substitution_into_receipt() -> None:
    source = inspect.getsource(screen.run_linux_publication_qualification)
    assert "first_child_before_spawn=substitute_evaluator_path" in source
    assert "evaluator_path=evaluator_probe_path" in source
    assert "held_evaluator_path_substitution" in source
    assert "held_evaluator_pathname_substitution_exercised" in source

    evaluator_sha256 = screen.sha256_regular_file(Path(screen.__file__).resolve())
    receipt = _synthetic_linux_qualification_receipt(evaluator_sha256)
    screen.validate_replayable_linux_qualification_receipt(receipt)
    evidence = receipt["qualification_result"]["qualification_evidence"][
        "held_evaluator_path_substitution"
    ]
    assert evidence["substitution_before_first_child"] is True
    assert evidence["source_inode"] == evidence["retained_inode"]
    assert evidence["replacement_inode"] != evidence["retained_inode"]
    assert evidence["retained_sha256"] == evaluator_sha256
    assert evidence["replacement_sha256"] != evaluator_sha256


@pytest.mark.skipif(
    type(getattr(os, "O_SYNC", None)) is not int or getattr(os, "O_SYNC", 0) == 0,
    reason="requires O_SYNC",
)
def test_linux_receipt_crash_subprocess_replay_accepts_only_complete_receipt(
    tmp_path: Path,
) -> None:
    transaction_path = (tmp_path / "linux-receipt-crash-cases").resolve()
    transaction_path.mkdir(mode=0o700)
    transaction_path.chmod(0o700)
    transaction_fd, transaction_record = screen.open_owned_output_directory(
        transaction_path
    )
    evaluator_sha256 = screen.sha256_regular_file(Path(screen.__file__).resolve())
    try:
        cases, accepted_publication = screen._run_linux_receipt_crash_consistency_cases(
            transaction_fd, transaction_path, evaluator_sha256
        )
        assert [
            (case["stage"], case["child_exit_code"], case["observed_replay"])
            for case in cases
        ] == [
            (stage, exit_code, expected_replay)
            for stage, exit_code, expected_replay in screen.LINUX_RECEIPT_CRASH_STAGES
        ]
        receipt_sizes = {case["stage"]: case["receipt_size"] for case in cases}
        assert receipt_sizes["before_publication_parent_fsync"] == 0
        assert 0 < receipt_sizes["during_partial_receipt_write"]
        assert (
            receipt_sizes["during_partial_receipt_write"]
            < receipt_sizes["after_complete_receipt_write"]
        )
        assert accepted_publication["receipt"]["status"] == (
            screen.LINUX_QUALIFICATION_RECEIPT_STATUS
        )
        assert os.listdir(transaction_fd) == []
        screen.validate_output_directory_fd(
            transaction_fd, transaction_record, require_path_identity=True
        )
    finally:
        os.close(transaction_fd)


@pytest.mark.skipif(
    type(getattr(os, "O_SYNC", None)) is not int or getattr(os, "O_SYNC", 0) == 0,
    reason="requires O_SYNC",
)
def test_linux_receipt_child_executes_held_bytes_across_path_substitution(
    tmp_path: Path,
) -> None:
    evaluator_copy = (tmp_path / "evaluator-copy.py").resolve()
    evaluator_copy.write_bytes(Path(screen.__file__).resolve().read_bytes())
    evaluator_copy.chmod(0o444)
    expected_hash = screen.sha256_regular_file(evaluator_copy)
    held_fd, held_source = screen._create_held_evaluator_image(
        evaluator_copy, expected_hash
    )
    scenario = (tmp_path / "held-child-scenario").resolve()
    scenario.mkdir(mode=0o700)
    scenario.chmod(0o700)

    def substitute_path() -> None:
        evaluator_copy.chmod(0o644)
        evaluator_copy.write_text("raise SystemExit('substituted pathname')\n")
        evaluator_copy.chmod(0o444)

    try:
        exit_code, broker_process_id = screen._run_linux_receipt_crash_subprocess(
            scenario,
            "after_complete_receipt_write",
            held_source,
            held_fd,
            before_spawn=substitute_path,
        )
        assert exit_code == 93
        assert screen.sha256_regular_file(evaluator_copy) != expected_hash
        scenario_fd, record = screen.open_owned_output_directory(scenario)
        try:
            receipt = screen._read_and_validate_linux_qualification_receipt(
                scenario_fd,
                record,
                "after_complete_receipt_write",
                held_source,
                broker_process_id,
            )
            assert receipt["evaluator_source"] == held_source
            assert (
                receipt["authority_boundary"]
                == screen.LINUX_QUALIFICATION_AUTHORITY_BOUNDARY
            )
        finally:
            os.close(scenario_fd)
    finally:
        os.close(held_fd)


def test_strict_file_hash_schema_and_refuse_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    assert screen.read_verified_bytes(frozen, digest) == b"frozen"
    frozen.write_bytes(b"changed")
    with pytest.raises(screen.ContractError, match="hash mismatch"):
        screen.read_verified_bytes(frozen, digest)

    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    symlink = tmp_path / "link.bin"
    symlink.symlink_to(target)
    with pytest.raises(screen.ContractError, match="regular single-link"):
        screen.read_verified_bytes(symlink, hashlib.sha256(b"target").hexdigest())
    wrong_mode_directory = tmp_path / "wrong-mode-output"
    wrong_mode_directory.mkdir(mode=0o755)
    wrong_mode_directory.chmod(0o755)
    with pytest.raises(screen.ContractError, match="current-UID-owned mode 0700"):
        screen.open_owned_output_directory(wrong_mode_directory)

    malformed_report = {key: None for key in screen.TOP_LEVEL_KEYS}
    malformed_report["extra"] = None
    with pytest.raises(screen.ContractError, match="schema keys differ"):
        screen.validate_report_schema(malformed_report)
    missing = {key: None for key in screen.TOP_LEVEL_KEYS if key != "cases"}
    with pytest.raises(screen.ContractError, match="schema keys differ"):
        screen.validate_report_schema(missing)
    nested = {key: None for key in screen.PRIMARY_RECORD_KEYS}
    nested["extra"] = None
    with pytest.raises(screen.ContractError, match="schema keys differ"):
        screen._require_exact_keys(nested, screen.PRIMARY_RECORD_KEYS, "arm")

    monkeypatch.setattr(
        screen, "validate_report_schema", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    tmp_path.chmod(0o700)
    directory_fd, output_record = screen.open_owned_output_directory(tmp_path)
    candidate_name = ".report.json.r12-candidate-1-test"
    context = {
        "output_directory": output_record,
        "candidate_name": candidate_name,
        "nonce": "a" * 64,
    }
    first_fd = None
    verifier_fd = None
    try:
        first_hash, first_fd, verifier_fd, creation_info = (
            screen.publish_private_candidate_exclusive(
                directory_fd, candidate_name, {"ok": True}, context
            )
        )
        assert first_hash == hashlib.sha256(b'{"ok":true}\n').hexdigest()
        published_stat = os.stat(
            candidate_name, dir_fd=directory_fd, follow_symlinks=False
        )
        assert screen._file_identity(os.fstat(first_fd)) == screen._file_identity(
            creation_info
        )
        assert screen._file_identity(published_stat) == screen._file_identity(
            creation_info
        )
        assert screen._file_identity(os.fstat(verifier_fd)) == screen._file_identity(
            creation_info
        )
        assert fcntl.fcntl(verifier_fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert stat.S_IMODE(published_stat.st_mode) == 0o400
        assert published_stat.st_nlink == 1
        assert (
            screen.verify_private_candidate(directory_fd, candidate_name, context)
            == first_hash
        )
        with pytest.raises(FileExistsError):
            screen.publish_private_candidate_exclusive(
                directory_fd, candidate_name, {"ok": False}, context
            )
        assert (tmp_path / candidate_name).read_bytes() == b'{"ok":true}\n'
    finally:
        if verifier_fd is not None:
            os.close(verifier_fd)
        if first_fd is not None:
            os.close(first_fd)
        os.close(directory_fd)


def test_candidate_creation_descriptor_survives_pathname_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        screen, "validate_report_schema", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    tmp_path.chmod(0o700)
    directory_fd, output_record = screen.open_owned_output_directory(tmp_path)
    candidate_name = ".candidate-retained-descriptor"
    context = {
        "output_directory": output_record,
        "candidate_name": candidate_name,
        "accepted_name": "accepted.json",
        "nonce": "b" * 64,
    }
    candidate_fd = None
    verifier_fd = None
    try:
        candidate_sha256, candidate_fd, verifier_fd, creation_info = (
            screen.publish_private_candidate_exclusive(
                directory_fd, candidate_name, {"original": True}, context
            )
        )
        assert fcntl.fcntl(verifier_fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        with pytest.raises(OSError):
            os.write(verifier_fd, b"forbidden verifier write")
        os.rename(
            candidate_name,
            ".candidate-retained-original",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replacement_fd = os.open(
            candidate_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
            dir_fd=directory_fd,
        )
        try:
            os.write(replacement_fd, b'{"hostile":true}\n')
            os.fsync(replacement_fd)
        finally:
            os.close(replacement_fd)
        os.fsync(directory_fd)

        payload, held_info = screen.read_private_candidate_descriptor(candidate_fd)
        assert payload == b'{"original":true}\n'
        assert screen.sha256_bytes(payload) == candidate_sha256
        assert screen._file_identity(held_info) == screen._file_identity(creation_info)
        assert screen._file_identity(
            os.stat(candidate_name, dir_fd=directory_fd, follow_symlinks=False)
        ) != screen._file_identity(creation_info)
        with pytest.raises(screen.ContractError, match="does not name the held inode"):
            screen._read_held_directory_entry_descriptor(
                directory_fd,
                candidate_name,
                candidate_fd,
                "candidate substitution regression",
                expected_info=creation_info,
                expected_payload=payload,
            )
    finally:
        if verifier_fd is not None:
            os.close(verifier_fd)
        if candidate_fd is not None:
            os.close(candidate_fd)
        os.close(directory_fd)


def test_evaluator_has_no_canonical_acceptance_publication_path(tmp_path: Path) -> None:
    main_source = inspect.getsource(screen.main)
    assert '"generate-sealed-report"' in main_source
    assert '"verify-private-candidate"' in main_source
    assert '"replay-accepted-bundle"' in main_source
    assert 'parser.add_argument("--out"' not in main_source
    assert "publish_private_candidate_exclusive" not in main_source
    assert "publish_accepted_bundle_exclusive" not in main_source
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    try:
        screen._validate_wrapper_acceptance(context, context)
        hostile = copy.deepcopy(context)
        hostile["candidate_name"] = hostile["accepted_name"]
        with pytest.raises(screen.ContractError, match="candidate/final name"):
            screen._validate_wrapper_acceptance(hostile, hostile)
    finally:
        os.close(directory_fd)


def test_source_execution_contract_requires_exact_sealed_memfd_records() -> None:
    custody = {
        "files": {
            "train/eval_dws_eos_suppressed_trace.py": "1" * 64,
            "train/model.py": "2" * 64,
        }
    }
    record = {
        "mode": screen.SOURCE_EXECUTION_MODE,
        "python_startup_mode": screen.PYTHON_STARTUP_MODE,
        "evaluator": {
            "descriptor_kind": "sealed_memfd",
            "sha256": "1" * 64,
            "byte_count": 10,
            "seals": screen.REQUIRED_MEMFD_SEALS,
        },
        "model": {
            "descriptor_kind": "sealed_memfd",
            "sha256": "2" * 64,
            "byte_count": 20,
            "seals": screen.REQUIRED_MEMFD_SEALS,
        },
    }
    screen._validate_source_execution(record, custody, live_custody=False)
    hostile = copy.deepcopy(record)
    hostile["model"]["descriptor_kind"] = "pathname"
    with pytest.raises(screen.ContractError, match="sealed source descriptor"):
        screen._validate_source_execution(hostile, custody, live_custody=False)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _sealed_runtime_source_fixture(
    tmp_path: Path,
) -> tuple[Path, str, Path, str, Path, Path, Path]:
    source_root = (tmp_path / "reviewed-source").resolve()
    source_root.mkdir()
    for index, relative_path in enumerate(screen.RUNTIME_SOURCE_PATHS):
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"runtime-source-{index}\n".encode())
    _git(source_root, "init", "-q")
    _git(source_root, "config", "user.name", "R12 Test")
    _git(source_root, "config", "user.email", "r12-test@example.invalid")
    _git(source_root, "add", "--", *screen.RUNTIME_SOURCE_PATHS)
    _git(source_root, "commit", "-q", "-m", "sealed runtime sources")
    source_commit = _git(source_root, "rev-parse", "HEAD")
    files = {
        relative_path: hashlib.sha256(
            (source_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in screen.RUNTIME_SOURCE_PATHS
    }
    python_bin = Path(sys.executable).resolve()
    git_location = shutil.which("git")
    assert git_location is not None
    git_bin = Path(git_location).resolve()
    scontrol_bin = (tmp_path / "scontrol-test-stub").resolve()
    scontrol_bin.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "--version" ]; then\n'
        "  echo 'slurm-test-stub 1'\n"
        "  exit 0\n"
        "fi\n"
        "echo 'JobId=1 JobName=shohin-r12-dws-eos-dev JobState=RUNNING'\n"
    )
    scontrol_bin.chmod(0o755)
    runtime = screen.observe_runtime_identity(
        python_bin, git_bin, scontrol_bin, scontrol_bin, scontrol_bin
    )
    manifest = {
        "schema": screen.RUNTIME_SOURCE_MANIFEST_SCHEMA,
        "source_root": str(source_root),
        "source_commit": source_commit,
        "files": files,
        "runtime": runtime,
    }
    manifest_path = tmp_path / "sealed-runtime-sources.json"
    manifest_bytes = screen.stable_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o444)
    return (
        source_root,
        source_commit,
        manifest_path,
        hashlib.sha256(manifest_bytes).hexdigest(),
        python_bin,
        git_bin,
        scontrol_bin,
    )


def test_runtime_source_manifest_binds_clean_commit_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    (
        source_root,
        source_commit,
        manifest_path,
        manifest_hash,
        python_bin,
        git_bin,
        scontrol_bin,
    ) = _sealed_runtime_source_fixture(tmp_path)
    custody = screen.verify_runtime_source_manifest(
        source_root,
        source_commit,
        manifest_path,
        manifest_hash,
        python_bin,
        git_bin,
        scontrol_bin,
        scontrol_bin,
        scontrol_bin,
    )
    assert custody["source_commit"] == source_commit
    assert custody["files"] == {
        relative_path: hashlib.sha256(
            (source_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in screen.RUNTIME_SOURCE_PATHS
    }
    assert custody["git_status"] == "clean"
    assert custody["git_show_byte_equality"] is True
    round_trip = screen._parse_json_object_bytes(
        screen.stable_json_bytes(custody), "custody round trip"
    )
    with pytest.raises(screen.ContractError, match="Python startup mode"):
        screen._validate_runtime_custody(round_trip)

    manifest_descriptor = os.open(
        manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    moved_manifest = manifest_path.with_name("moved-runtime-manifest.json")
    try:
        original_manifest_bytes = manifest_path.read_bytes()
        manifest_path.rename(moved_manifest)
        manifest_path.write_bytes(original_manifest_bytes)
        manifest_path.chmod(0o444)
        with pytest.raises(screen.ContractError, match="held runtime-source manifest"):
            screen.verify_runtime_source_manifest(
                source_root,
                source_commit,
                manifest_path,
                manifest_hash,
                python_bin,
                git_bin,
                scontrol_bin,
                scontrol_bin,
                scontrol_bin,
                manifest_descriptor=manifest_descriptor,
            )
    finally:
        os.close(manifest_descriptor)
        manifest_path.unlink(missing_ok=True)
        moved_manifest.rename(manifest_path)

    evaluator = source_root / "train/eval_dws_eos_suppressed_trace.py"
    evaluator.write_bytes(evaluator.read_bytes() + b"tamper\n")
    with pytest.raises(screen.ContractError, match="Git state is not clean"):
        screen.verify_runtime_source_manifest(
            source_root,
            source_commit,
            manifest_path,
            manifest_hash,
            python_bin,
            git_bin,
            scontrol_bin,
            scontrol_bin,
            scontrol_bin,
        )


def test_runtime_git_commands_neutralize_fsmonitor_hooks_and_config_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        source_root,
        source_commit,
        manifest_path,
        manifest_hash,
        python_bin,
        git_bin,
        scontrol_bin,
    ) = _sealed_runtime_source_fixture(tmp_path)
    marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "hostile-fsmonitor.sh"
    hook.write_text(f"#!/bin/sh\nprintf ran > {marker}\nexit 0\n")
    hook.chmod(0o755)
    _git(source_root, "config", "core.fsmonitor", str(hook))
    global_config = tmp_path / "hostile-global-gitconfig"
    global_config.write_text(f"[core]\n\tfsmonitor = {hook}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(hook))

    custody = screen.verify_runtime_source_manifest(
        source_root,
        source_commit,
        manifest_path,
        manifest_hash,
        python_bin,
        git_bin,
        scontrol_bin,
        scontrol_bin,
        scontrol_bin,
    )
    assert custody["git_status"] == "clean"
    assert not marker.exists()


def test_runtime_source_manifest_rejects_manifest_tamper_and_write_bits(
    tmp_path: Path,
) -> None:
    (
        source_root,
        source_commit,
        manifest_path,
        manifest_hash,
        python_bin,
        git_bin,
        scontrol_bin,
    ) = _sealed_runtime_source_fixture(tmp_path)
    manifest_path.chmod(0o644)
    with pytest.raises(screen.ContractError, match="no write permission"):
        screen.verify_runtime_source_manifest(
            source_root,
            source_commit,
            manifest_path,
            manifest_hash,
            python_bin,
            git_bin,
            scontrol_bin,
            scontrol_bin,
            scontrol_bin,
        )
    manifest_path.chmod(0o444)
    with pytest.raises(screen.ContractError, match="hash mismatch"):
        screen.verify_runtime_source_manifest(
            source_root,
            source_commit,
            manifest_path,
            "0" * 64,
            python_bin,
            git_bin,
            scontrol_bin,
            scontrol_bin,
            scontrol_bin,
        )

    manifest = json.loads(manifest_path.read_bytes())
    manifest["runtime"]["backend"]["sdpa_backend"] = "flash"
    hostile_bytes = screen.stable_json_bytes(manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(hostile_bytes)
    manifest_path.chmod(0o444)
    with pytest.raises(screen.ContractError, match="runtime/package/backend"):
        screen.verify_runtime_source_manifest(
            source_root,
            source_commit,
            manifest_path,
            hashlib.sha256(hostile_bytes).hexdigest(),
            python_bin,
            git_bin,
            scontrol_bin,
            scontrol_bin,
            scontrol_bin,
        )


def test_decode_report_replay_rejects_mode_count_mask_position_and_stop_tamper() -> (
    None
):
    tokenizer = TinyTokenizer()
    initial = screen.parse_dws_line("dws:op=add;w=1;p=0;c=0;a=1;b=2;r=0;z=0")
    assert initial is not None
    oracle = screen.reconstruct_oracle_posthoc(initial)
    raw = screen.decode_cached_greedy(
        screen.DecodeRequest(
            model=TinyModel([0]),
            prompt_token_ids=(12, 13, 14),
            device="cpu",
            max_new_tokens=screen.MAX_NEW_TOKENS,
            mode=screen.DecodeMode.EOS_MASKED_ARGMAX,
            eos_token_id=0,
        )
    )
    record = screen.raw_decode_report_posthoc(raw, tokenizer, initial, oracle)
    screen._validate_decode_common(
        record,
        "arm",
        expected_mode=screen.DecodeMode.EOS_MASKED_ARGMAX.value,
        expected_prompt_token_count=3,
        tokenizer=tokenizer,
    )

    mutations = []
    wrong_mode = copy.deepcopy(record)
    wrong_mode["mode"] = screen.DecodeMode.EOS_TO_LF.value
    mutations.append(wrong_mode)
    wrong_content = copy.deepcopy(record)
    wrong_content["content_token_count"] -= 1
    mutations.append(wrong_content)
    wrong_mask = copy.deepcopy(record)
    wrong_mask["eos_mask_applied_positions"].pop()
    mutations.append(wrong_mask)
    wrong_absolute = copy.deepcopy(record)
    wrong_absolute["eos_events"][0]["absolute_token_position"] += 1
    mutations.append(wrong_absolute)
    wrong_stop = copy.deepcopy(record)
    wrong_stop["stop_reason"] = "max_new_tokens"
    mutations.append(wrong_stop)
    wrong_replacement = copy.deepcopy(record)
    wrong_replacement["eos_events"][0]["replacement_token_id"] = 211
    mutations.append(wrong_replacement)
    wrong_text = copy.deepcopy(record)
    wrong_text["response_text"] = "internally rehashed but not token-derived"
    wrong_text["response_sha256"] = hashlib.sha256(
        wrong_text["response_text"].encode()
    ).hexdigest()
    mutations.append(wrong_text)
    for mutation in mutations:
        with pytest.raises(screen.ContractError):
            screen._validate_decode_common(
                mutation,
                "arm",
                expected_mode=screen.DecodeMode.EOS_MASKED_ARGMAX.value,
                expected_prompt_token_count=3,
                tokenizer=tokenizer,
            )


def _publish_candidate_and_build_verifier_receipt(
    fixture: dict[str, object],
) -> dict[str, object]:
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    report = fixture["report"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    assert isinstance(report, dict)
    (
        _candidate_sha256,
        candidate_fd,
        candidate_verifier_fd,
        candidate_creation_info,
    ) = screen.publish_private_candidate_exclusive(
        directory_fd, context["candidate_name"], report, context
    )
    try:
        candidate_payload, candidate_info = screen.read_private_candidate_descriptor(
            candidate_verifier_fd
        )
        assert screen._file_identity(candidate_info) == screen._file_identity(
            candidate_creation_info
        )
        verifier_receipt = screen.build_independent_verifier_receipt(
            candidate_payload,
            candidate_info,
            report,
            _runtime_observation(
                "independent_validator", str(Path(sys.executable).resolve())
            ),
            {"paths": screen.frozen_input_paths(), "sha256": screen.EXPECTED_SHA256},
            fixture["verifier_private_key"],
        )
    finally:
        os.close(candidate_verifier_fd)
        os.close(candidate_fd)
    screen._validate_independent_verifier_receipt(verifier_receipt, report)
    return verifier_receipt


def test_candidate_publication_move_then_failure_quarantines_exact_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    report = fixture["report"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    assert isinstance(report, dict)

    def move_then_fail(
        old_directory_fd: int,
        old_name: str,
        new_directory_fd: int,
        new_name: str,
    ) -> None:
        os.rename(
            old_name,
            new_name,
            src_dir_fd=old_directory_fd,
            dst_dir_fd=new_directory_fd,
        )
        raise OSError("injected post-rename candidate failure")

    monkeypatch.setattr(screen, "rename_noreplace_at", move_then_fail)
    try:
        with pytest.raises(OSError, match="post-rename candidate failure"):
            screen.publish_private_candidate_exclusive(
                directory_fd, context["candidate_name"], report, context
            )
        assert (
            screen._entry_stat_or_none(directory_fd, context["candidate_name"]) is None
        )
        assert not any(".tmp-" in name for name in os.listdir(directory_fd))
        rollback_names = [
            name
            for name in os.listdir(directory_fd)
            if ".r12-rollback-quarantine-" in name
        ]
        assert len(rollback_names) == 1
        assert (tmp_path / "accepted-output" / rollback_names[0]).read_bytes() == (
            screen.stable_json_bytes(report)
        )
    finally:
        os.close(directory_fd)


def test_generator_and_independent_verifier_signatures_reject_self_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    report = fixture["report"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(report, dict)
    assert isinstance(context, dict)
    try:
        hostile_report = copy.deepcopy(report)
        hostile_report["execution"]["device_name"] = "self-rehashed-hostile-device"
        hostile_body = {
            key: value
            for key, value in hostile_report.items()
            if key != "generator_attestation"
        }
        hostile_report["generator_attestation"]["report_body_sha256"] = (
            screen.sha256_bytes(screen.stable_json_bytes(hostile_body))
        )
        with pytest.raises(screen.ContractError, match="attestation does not verify"):
            screen.validate_report_schema(hostile_report, context, live_custody=False)

        verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)
        hostile_receipt = copy.deepcopy(verifier_receipt)
        hostile_receipt["validation_runtime_observation"]["phase"] = "forged-phase"
        hostile_receipt["validation_runtime_observation_sha256"] = screen.sha256_bytes(
            screen.stable_json_bytes(hostile_receipt["validation_runtime_observation"])
        )
        with pytest.raises(screen.ContractError, match="signature does not verify"):
            screen._validate_independent_verifier_receipt(hostile_receipt, report)
    finally:
        os.close(directory_fd)


def _wrapper_heredoc(text: str, marker: str) -> str:
    return text.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0]


def test_slurm_identity_is_exactly_parsed_and_command_hash_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").resolve()
    uid = os.getuid()
    job_output = (
        f"JobId=42 JobName=shohin-r12-dws-eos-dev JobState=RUNNING "
        f"UserId=test-user({uid}) BatchFlag=1 Command={wrapper} "
        "BatchHost=node-a NodeList=node-a Partition=normal NumNodes=1 NumCPUs=4 NumTasks=1 "
        "CPUs/Task=4 MinCPUsNode=4 MinMemoryNode=64G TimeLimit=08:00:00 "
        "Requeue=0 "
        "ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,"
        "gres/gpu:nvidia_h100_pcie=1 "
        "AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,"
        "gres/gpu:nvidia_h100_pcie=1 "
        "TresPerNode=gres/gpu:nvidia_h100_pcie=1 "
        "Gres=gpu:nvidia_h100_pcie:1 OtherField=ignored\n"
    ).encode()
    config_output = b"ClusterName = exact-test-cluster\nSlurmctldHost=node-a\n"
    tres = "cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1"
    sacct_output = (
        f"42|shohin-r12-dws-eos-dev|normal|RUNNING|4|64G|08:00:00|"
        f"{tres}|{tres}|node-a\n"
    ).encode()

    def exact_command(
        _executable: str,
        *arguments: str,
        executable_descriptor: int | None = None,
    ) -> bytes:
        del executable_descriptor
        if arguments[:3] == ("show", "job", "-o"):
            return job_output
        if arguments[:2] == ("show", "config"):
            return config_output
        return sacct_output

    monkeypatch.setattr(screen, "_run_exact_command", exact_command)
    monkeypatch.setattr(screen.socket, "gethostname", lambda: "node-a")
    monkeypatch.setattr(
        screen,
        "observe_gpu_allocation_binding",
        lambda *_args, **_kwargs: _slurm_identity(
            str(wrapper), screen.sha256_regular_file(wrapper), job_id="42"
        )["gpu_binding"],
    )
    monkeypatch.setenv("SLURM_JOB_NAME", "hostile-env-name")
    observed = screen.observe_slurm_identity(
        "/reviewed/scontrol", "/reviewed/sacct", "/reviewed/nvidia-smi", "42"
    )
    assert observed["cluster_name"] == "exact-test-cluster"
    assert observed["command"] == str(wrapper)
    assert observed["command_sha256"] == screen.sha256_regular_file(wrapper)
    assert observed["partition"] == "normal"
    assert observed["batch_host"] == observed["node_list"] == "node-a"
    assert observed["observed_hostname"] == "node-a"
    assert observed["num_cpus"] == 4
    assert observed["memory_bytes"] == 64 * 1024**3
    assert observed["time_limit_seconds"] == 8 * 60 * 60
    assert observed["gres"] == "gpu:nvidia_h100_pcie:1"
    assert observed["tres_per_node"] == {"gres/gpu:nvidia_h100_pcie": "1"}
    assert observed["requeue"] == 0
    assert "hostile-env-name" not in json.dumps(observed)

    duplicate = job_output.rstrip() + b" JobId=999\n"

    def duplicate_command(
        _executable: str,
        *_arguments: str,
        executable_descriptor: int | None = None,
    ) -> bytes:
        del executable_descriptor
        return duplicate

    monkeypatch.setattr(screen, "_run_exact_command", duplicate_command)
    with pytest.raises(screen.ContractError, match="duplicate"):
        screen.observe_slurm_identity(
            "/reviewed/scontrol", "/reviewed/sacct", "/reviewed/nvidia-smi", "42"
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("Partition=normal", "Partition=debug"),
        ("NumCPUs=4", "NumCPUs=8"),
        ("MinMemoryNode=64G", "MinMemoryNode=32G"),
        ("TimeLimit=08:00:00", "TimeLimit=04:00:00"),
        ("Requeue=0", "Requeue=1"),
        (
            "gres/gpu:nvidia_h100_pcie=1",
            "gres/gpu:nvidia_a100=1",
        ),
    ),
)
def test_slurm_identity_rejects_allocation_contract_tamper(
    monkeypatch: pytest.MonkeyPatch, field: str, replacement: str
) -> None:
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").resolve()
    uid = os.getuid()
    base = (
        f"JobId=42 JobName=shohin-r12-dws-eos-dev JobState=RUNNING "
        f"UserId=test-user({uid}) BatchFlag=1 Command={wrapper} BatchHost=node-a "
        "NodeList=node-a "
        "Partition=normal NumNodes=1 NumCPUs=4 NumTasks=1 CPUs/Task=4 "
        "MinCPUsNode=4 MinMemoryNode=64G TimeLimit=08:00:00 Requeue=0 "
        "ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,"
        "gres/gpu:nvidia_h100_pcie=1 "
        "AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,"
        "gres/gpu:nvidia_h100_pcie=1 "
        "TresPerNode=gres/gpu:nvidia_h100_pcie=1 "
        "Gres=gpu:nvidia_h100_pcie:1\n"
    )
    hostile = base.replace(field, replacement)
    tres = "cpu=4,mem=64G,node=1,billing=4,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1"
    sacct_output = (
        f"42|shohin-r12-dws-eos-dev|normal|RUNNING|4|64G|08:00:00|"
        f"{tres}|{tres}|node-a\n"
    ).encode()

    def exact_command(
        _executable: str,
        *arguments: str,
        executable_descriptor: int | None = None,
    ) -> bytes:
        del executable_descriptor
        if arguments[:3] == ("show", "job", "-o"):
            return hostile.encode()
        if arguments[:2] == ("show", "config"):
            return b"ClusterName = exact-test-cluster\n"
        return sacct_output

    monkeypatch.setattr(screen, "_run_exact_command", exact_command)
    monkeypatch.setattr(screen.socket, "gethostname", lambda: "node-a")
    monkeypatch.setattr(
        screen,
        "observe_gpu_allocation_binding",
        lambda *_args, **_kwargs: _slurm_identity(
            str(wrapper), screen.sha256_regular_file(wrapper), job_id="42"
        )["gpu_binding"],
    )
    with pytest.raises(screen.ContractError, match="Slurm|TRES|allocation"):
        screen.observe_slurm_identity(
            "/reviewed/scontrol", "/reviewed/sacct", "/reviewed/nvidia-smi", "42"
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("batch_host", "node-b"),
        ("node_list", "node-b"),
        ("observed_hostname", "node-b"),
        ("node_list", "node-[01-02]"),
        ("node_list", "node-a,node-b"),
        ("node_list", "node-a.example"),
    ),
)
def test_slurm_identity_rejects_node_cross_binding_and_ambiguity(
    field: str, replacement: str
) -> None:
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").resolve()
    identity = _slurm_identity(str(wrapper), screen.sha256_regular_file(wrapper))
    identity[field] = replacement
    with pytest.raises(screen.ContractError, match="node|Node"):
        screen._validate_slurm_identity(identity)


def test_slurm_identity_rejects_sacct_node_disagreement() -> None:
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").resolve()
    identity = _slurm_identity(str(wrapper), screen.sha256_regular_file(wrapper))
    identity["sacct_identity"]["node_list"] = "node-b"
    with pytest.raises(screen.ContractError, match="sacct identity"):
        screen._validate_slurm_identity(identity)


def test_gpu_binding_binds_smi_minor_and_rejects_absent_node_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("5:devices:/slurm/uid_1/job_42/step_batch\n")
    cgroup_root = tmp_path / "cgroup-devices"
    devices_path = cgroup_root / "slurm/uid_1/job_42/step_batch"
    devices_path.mkdir(parents=True)
    devices_file = devices_path / "devices.list"
    null_info = Path("/dev/null").stat()
    major = os.major(null_info.st_rdev)
    minor = os.minor(null_info.st_rdev)
    devices_file.write_text(f"c {major}:{minor} rwm\n")
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    (dev_root / f"nvidia{minor}").symlink_to("/dev/null")
    pci_root = tmp_path / "pci-devices"
    pci_device = pci_root / "0000:3b:00.0"
    pci_device.mkdir(parents=True)
    (pci_device / "vendor").write_text("0x10de\n")
    (pci_device / "device").write_text("0x2331\n")
    (pci_device / "class").write_text("0x030200\n")
    query = (
        f"7, {minor}, NVIDIA H100 PCIe, GPU-exact, 00000000:3B:00.0, Disabled\n"
    ).encode()
    listing = b"GPU 7: NVIDIA H100 PCIe (UUID: GPU-exact)\n"

    def exact_command(
        _executable: str,
        *arguments: str,
        executable_descriptor: int | None = None,
    ) -> bytes:
        del executable_descriptor
        return listing if arguments == ("-L",) else query

    monkeypatch.setattr(screen, "_run_exact_command", exact_command)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("SLURM_JOB_GPUS", "GPU-exact")
    binding = screen.observe_gpu_allocation_binding(
        "/reviewed/nvidia-smi",
        "42",
        proc_self_cgroup=proc_cgroup,
        cgroup_devices_root=cgroup_root,
        dev_root=dev_root,
        pci_devices_root=pci_root,
    )
    assert binding["gpu_uuid"] == "GPU-exact"
    assert binding["pci_bus_id"] == "00000000:3b:00.0"
    assert binding["nvidia_smi_index"] == 7
    assert binding["nvidia_smi_minor_number"] == minor
    assert (binding["gpu_major"], binding["gpu_minor"]) == (major, minor)
    assert binding["nvidia_control_devices"] == []

    absent_extra_minor = minor + 100
    original_lstat = Path.lstat

    def canonical_control_lstat(path: Path) -> os.stat_result | SimpleNamespace:
        if path == dev_root / "nvidiactl":
            return SimpleNamespace(
                st_mode=stat.S_IFCHR | 0o666,
                st_rdev=os.makedev(major, 255),
            )
        return original_lstat(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", canonical_control_lstat)
        devices_file.write_text(f"c {major}:{minor} rwm\nc {major}:255 rwm\n")
        control_binding = screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )
        assert control_binding["nvidia_control_devices"] == [
            {"name": "nvidiactl", "major": major, "minor": 255}
        ]
        assert control_binding["concrete_physical_gpu_permissions"] == [
            {"major": major, "minor": minor}
        ]

    def physical_gpu_alias_lstat(path: Path) -> os.stat_result | SimpleNamespace:
        if path == dev_root / "nvidiactl":
            return SimpleNamespace(
                st_mode=stat.S_IFCHR | 0o666,
                st_rdev=os.makedev(major, minor),
            )
        return original_lstat(path)

    devices_file.write_text(f"c {major}:{minor} rwm\n")
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", physical_gpu_alias_lstat)
        with pytest.raises(screen.ContractError, match="collides with a physical-GPU"):
            screen.observe_gpu_allocation_binding(
                "/reviewed/nvidia-smi",
                "42",
                proc_self_cgroup=proc_cgroup,
                cgroup_devices_root=cgroup_root,
                dev_root=dev_root,
                pci_devices_root=pci_root,
            )

    devices_file.write_text(
        f"c {major}:{minor} rwm\nc {major}:{absent_extra_minor} rwm\n"
    )
    assert not (dev_root / f"nvidia{absent_extra_minor}").exists()
    with pytest.raises(screen.ContractError, match="extra concrete physical-GPU minor"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )

    devices_file.write_text(f"c {major}:{minor} rwm\nc {major}:{minor} rwm\n")
    with pytest.raises(screen.ContractError, match="duplicated"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )

    devices_file.write_text("c 195:99 rwm\n")
    with pytest.raises(screen.ContractError, match="major/minor"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )
    devices_file.write_text(f"c {major}:{minor} rwm\n")
    monkeypatch.setenv("SLURM_JOB_GPUS", "1")
    with pytest.raises(screen.ContractError, match="does not resolve"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )
    monkeypatch.setenv("SLURM_JOB_GPUS", "GPU-exact")
    hostile_control_path = dev_root / "nvidiactl"
    hostile_control_path.write_text("not a device\n")
    with pytest.raises(screen.ContractError, match="not a character device"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )
    hostile_control_path.unlink()
    proc_cgroup.write_text("0::/slurm/uid_1/job_42/step_batch\n")
    with pytest.raises(screen.ContractError, match="cgroup-v1"):
        screen.observe_gpu_allocation_binding(
            "/reviewed/nvidia-smi",
            "42",
            proc_self_cgroup=proc_cgroup,
            cgroup_devices_root=cgroup_root,
            dev_root=dev_root,
            pci_devices_root=pci_root,
        )


@pytest.mark.parametrize(
    "control_name",
    ("nvidiactl", "nvidia-modeset", "nvidia-uvm", "nvidia-uvm-tools"),
)
def test_nvidia_control_device_cannot_alias_physical_gpu_number(
    control_name: str,
) -> None:
    with pytest.raises(screen.ContractError, match="collides with a physical-GPU"):
        screen._validate_nvidia_control_device_numbers({control_name: (195, 0)}, 195, 0)


def test_nvidia_control_device_numbers_are_exact_and_canonical() -> None:
    screen._validate_nvidia_control_device_numbers(
        {
            "nvidia-modeset": (195, 254),
            "nvidiactl": (195, 255),
            "nvidia-uvm": (511, 0),
            "nvidia-uvm-tools": (511, 1),
        },
        195,
        7,
    )
    with pytest.raises(screen.ContractError, match="noncanonical identity"):
        screen._validate_nvidia_control_device_numbers(
            {"nvidiactl": (511, 255)}, 195, 7
        )
    with pytest.raises(screen.ContractError, match="inconsistent majors"):
        screen._validate_nvidia_control_device_numbers(
            {"nvidia-uvm": (510, 0), "nvidia-uvm-tools": (511, 1)}, 195, 7
        )
    with pytest.raises(screen.ContractError, match="alias one device number"):
        screen._validate_nvidia_control_device_numbers(
            {"nvidia-uvm": (511, 0), "nvidiactl": (511, 0)}, 195, 7
        )


def test_nvidia_smi_inventory_rejects_mig_and_identity_tamper() -> None:
    with pytest.raises(screen.ContractError, match="full H100 PCIe identity differs"):
        screen._parse_nvidia_smi_gpu(
            b"7, 3, NVIDIA H100 PCIe, GPU-exact, 00000000:3B:00.0, Enabled\n"
        )
    with pytest.raises(screen.ContractError, match="full H100 PCIe identity differs"):
        screen._parse_nvidia_smi_gpu(
            b"7, x, NVIDIA H100 PCIe, GPU-exact, 00000000:3B:00.0, Disabled\n"
        )


@pytest.mark.parametrize(
    ("name", "properties_name", "capability", "memory"),
    (
        ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB", (8, 0), 80 << 30),
        ("NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3", (9, 0), 80 << 30),
        ("NVIDIA H100 PCIe", "NVIDIA H100 PCIe", (8, 9), 80 << 30),
        ("NVIDIA H100 PCIe", "NVIDIA H100 PCIe", (9, 0), 40 << 30),
    ),
)
def test_h100_pcie_device_identity_is_exact_and_fail_closed(
    name: str,
    properties_name: str,
    capability: tuple[int, int],
    memory: int,
) -> None:
    with pytest.raises(screen.ContractError, match="full NVIDIA H100 PCIe"):
        screen._validate_h100_pcie_device_identity(
            name, properties_name, capability, memory, "GPU-test"
        )
    screen._validate_h100_pcie_device_identity(
        screen.REQUIRED_CUDA_DEVICE_NAME,
        screen.REQUIRED_CUDA_DEVICE_NAME,
        screen.REQUIRED_CUDA_DEVICE_CAPABILITY,
        screen.REQUIRED_CUDA_MEMORY_MIN_BYTES,
        "GPU-test",
    )


def test_wrapper_freezes_source_runtime_acceptance_and_determinism_contract() -> None:
    wrapper = ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch"
    text = wrapper.read_text()
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    compile(
        _wrapper_heredoc(text, "PY_ORCHESTRATE"),
        f"{wrapper}:PY_ORCHESTRATE",
        "exec",
    )
    assert "SOURCE_ROOT_REVIEWED" in text
    assert screen.RUNTIME_SOURCE_MANIFEST_SCHEMA in text
    assert 'f"{source_commit}:{relative_path}"' in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --export=NONE" in text
    assert text.startswith("#!/bin/bash -p\n")
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert (
        "unset BASH_ENV ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE" in text
    )
    assert 'exec 8<"$PYTHON_BIN"' in text
    assert 'exec 9<"$GIT_BIN"' in text
    assert 'exec 7<"$SCONTROL_BIN"' in text
    assert 'exec 6<"$SOURCE_MANIFEST"' in text
    assert 'exec 12<"$PRODUCTION_AUTHORITY_PUBLIC_KEY"' not in text
    assert 'exec 13<"$RUN_AUTHORIZATION"' not in text
    assert text.index("for environment_name in LD_PRELOAD") < text.index('exec 5<"$0"')
    assert text.index("unset BASH_ENV ENV PYTHONHOME") < text.index('exec 5<"$0"')
    assert 'exec "/proc/self/fd/8" -I -S -B' in text
    assert "os.memfd_create" in text
    assert "PR_SET_DUMPABLE" in text and "PR_GET_DUMPABLE" in text
    assert "actual Slurm-spooled wrapper bytes differ" in text
    assert "generate-sealed-report" in text
    assert "verify-private-candidate" in text
    assert text.index("verify-private-candidate") < text.index(
        "publish_accepted_bundle_exclusive"
    )
    assert all(
        key in text
        for key in (
            "generator_key_fd",
            "verifier_key_fd",
            "delegated_marker_key_fd",
        )
    )
    assert "PRODUCTION_AUTHORITY_PUBLIC_KEY" in text
    assert "RUN_AUTHORIZATION_SHA256" in text
    assert "DELEGATED_MARKER_KEY_BROKER_FD" in text
    assert "DELEGATED_MARKER_KEY_FD" not in text
    assert "receive_delegated_signing_key_from_broker" in text
    assert "SCM_RIGHTS" in inspect.getsource(
        screen.receive_delegated_signing_key_from_broker
    )
    assert "SACCT_BIN" in text and "NVIDIA_SMI_BIN" in text
    assert "SLURM_JOB_NAME" not in text
    assert "SLURM_CLUSTER_NAME" not in text
    assert "failure_injector=final_verification" in text
    orchestration = _wrapper_heredoc(text, "PY_ORCHESTRATE")
    assert orchestration.index("observed_runtime = module.observe_runtime_identity") < (
        orchestration.index("authority_key_fd, authority_key_bytes")
    )
    assert orchestration.index("module.validate_run_authorization") < (
        orchestration.index("delegated marker key broker is not an inherited socket")
    )
    assert "os.unlink(candidate_name" not in orchestration
    assert "candidate_verifier_fd" in orchestration
    assert "str(candidate_verifier_fd)" in orchestration
    assert "candidate_fd = os.open" not in orchestration
    assert orchestration.index("_read_held_directory_entry_descriptor") < (
        orchestration.index("_quarantine_held_entry_if_exact")
    )
    assert orchestration.index("_quarantine_held_entry_if_exact") < (
        orchestration.index("publish_accepted_bundle_exclusive")
    )
    assert "wrapper private candidate final quarantine hold" in orchestration
    assert "LD_AUDIT" in text and "GLIBC_TUNABLES" in text
    child_environment = orchestration.split("child_environment =", 1)[1].split(
        "common_arguments =", 1
    )[0]
    assert "LD_LIBRARY_PATH" not in child_environment
    assert text.index("acquire_publisher_lease") < text.index(
        "cleanup_stale_publication_entries"
    )
    assert text.index("validate_run_authorization") < text.index(
        "activate_pinned_runtime_packages"
    )
    assert text.index("run authorization source/manifest/prereg binding differs") < (
        text.index("cleanup_stale_publication_entries")
    )
    assert text.index("cleanup_stale_publication_entries") < text.index(
        "receive_delegated_signing_key_from_broker"
    )
    assert "core.fsmonitor=false" in text
    assert "core.hooksPath=/dev/null" in text
    assert "commit_marker_sha256" in text
    assert "durable_acceptance_receipt_sha256" in text
    publication = inspect.getsource(screen.publish_accepted_bundle_exclusive)
    assert publication.index('failure_injector("after_final_readback")') < (
        publication.index("commit_marker, commit_payload = build_marker_artifact")
    )
    assert publication.index(
        "commit_marker, commit_payload = build_marker_artifact"
    ) < publication.index("before_commit_marker_rename")
    assert publication.index("before_commit_marker_parent_fsync") < publication.index(
        'failure_injector("after_commit_marker_parent_fsync")'
    )
    assert publication.index("_fsync_held_canonical_publication") < (
        publication.index("durable_receipt, durable_receipt_payload =")
    )
    receipt_builder_index = publication.index(
        "durable_receipt, durable_receipt_payload ="
    )
    assert receipt_builder_index < publication.index(
        "_write_complete_o_sync_receipt", receipt_builder_index
    )
    assert publication.index("_write_complete_o_sync_receipt") < (
        publication.index("after_durable_acceptance_receipt_sync_write")
    )
    receipt_slot = inspect.getsource(screen._open_durable_o_sync_receipt_slot)
    receipt_writer = inspect.getsource(screen._write_complete_o_sync_receipt)
    assert 'getattr(os, "O_SYNC", None)' in receipt_slot
    assert receipt_writer.index("_write_all(descriptor, payload)") < (
        receipt_writer.index("after_sync_write()")
    )
    assert receipt_writer.index("after_sync_write()") < receipt_writer.index(
        "os.fsync(descriptor)"
    )
    assert receipt_writer.index("os.fsync(descriptor)") < receipt_writer.index(
        "_read_exact_descriptor_bytes"
    )
    preflight = inspect.getsource(screen._device_preflight)
    assert "torch.use_deterministic_algorithms(True, warn_only=False)" in preflight
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in preflight
    assert "torch.backends.cudnn.allow_tf32 = False" in preflight
    assert "F.scaled_dot_product_attention" in preflight
    assert "_validate_h100_pcie_device_identity" in preflight
    main_source = inspect.getsource(screen.main)
    assert "with pinned_math_sdpa()" in main_source
    autocast_body = main_source.split("with autocast:", 1)[1].split(
        "_finish_field_scores", 1
    )[0]
    assert "_run_primary_decodes" in autocast_body
    assert "_run_field_decodes" in autocast_body
    assert "_run_fresh_reencoding_decodes" in autocast_body


def test_full_schema_acceptance_is_failure_atomic_durable_and_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    report = fixture["report"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    assert isinstance(report, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)

    def fail_after_final_readback(stage: str) -> None:
        if stage == "after_final_readback":
            raise OSError("injected post-rename final-verification failure")

    try:
        with pytest.raises(OSError, match="post-rename final-verification"):
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                report,
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=fail_after_final_readback,
            )
        assert (
            screen._entry_stat_or_none(directory_fd, context["accepted_name"]) is None
        )
        marker_name = screen.acceptance_commit_marker_name(context["accepted_name"])
        durable_receipt_name = screen.durable_acceptance_receipt_name(
            context["accepted_name"]
        )
        assert screen._entry_stat_or_none(directory_fd, marker_name) is None
        assert screen._entry_stat_or_none(directory_fd, durable_receipt_name) is None
        assert not any(
            ".r12-report-" in name or ".r12-commit-" in name
            for name in os.listdir(directory_fd)
        )

        payload, marker_payload, receipt_payload, receipt = (
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                report,
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=_noop_publication_stage,
            )
        )
        accepted_info = os.stat(
            context["accepted_name"], dir_fd=directory_fd, follow_symlinks=False
        )
        assert stat.S_IMODE(accepted_info.st_mode) == 0o444
        assert accepted_info.st_nlink == 1
        marker_info = os.stat(marker_name, dir_fd=directory_fd, follow_symlinks=False)
        receipt_info = os.stat(
            durable_receipt_name, dir_fd=directory_fd, follow_symlinks=False
        )
        assert stat.S_IMODE(marker_info.st_mode) == 0o444
        assert marker_info.st_nlink == 1
        assert stat.S_IMODE(receipt_info.st_mode) == 0o444
        assert receipt_info.st_nlink == 1
        assert receipt["final_inode"] == screen._published_file_record(accepted_info)
        assert receipt["commit_marker_inode"] == screen._published_file_record(
            marker_info
        )
        assert receipt["durable_acceptance_receipt_inode"] == {
            "device": receipt_info.st_dev,
            "inode": receipt_info.st_ino,
            "uid": receipt_info.st_uid,
        }
        bundle = json.loads(payload)
        marker = json.loads(marker_payload)
        assert "acceptance_receipt" not in bundle
        assert marker["run_authorization_sha256"] == context["run_authorization_sha256"]
        assert (
            marker["authorization_sequence"]
            == context["run_authorization"]["authorization_sequence"]
        )
        assert (
            marker["delegated_marker_public_key_hex"]
            == context["run_authorization"]["delegated_marker_public_key_hex"]
        )
        assert marker["authority_key_id"] == (
            "r12-test-authority-no-production-authority"
        )
        screen.validate_acceptance_commit_marker(
            bundle,
            accepted_info,
            marker,
            marker_info,
            fixture["manifest_bytes"],
            fixture["manifest_info"],
        )
        screen.validate_durable_acceptance_receipt(
            bundle,
            accepted_info,
            marker,
            marker_info,
            json.loads(receipt_payload),
            receipt_info,
            fixture["manifest_bytes"],
            fixture["manifest_info"],
        )
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                report,
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=_noop_publication_stage,
            )
        assert (
            screen.read_directory_entry_bytes(
                directory_fd, context["accepted_name"], "accepted"
            )[0]
            == payload
        )

        tampered = copy.deepcopy(marker)
        tampered["report_sha256"] = "0" * 64
        with pytest.raises(screen.ContractError, match="cross-binding"):
            screen.validate_acceptance_commit_marker(
                bundle,
                accepted_info,
                tampered,
                marker_info,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
        self_rehashed_marker = copy.deepcopy(marker)
        self_rehashed_marker["committed_at_utc"] = "2026-01-01T09:09:09+00:00"
        with pytest.raises(screen.ContractError, match="signature does not verify"):
            screen.validate_acceptance_commit_marker(
                bundle,
                accepted_info,
                self_rehashed_marker,
                marker_info,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
        tampered_receipt = json.loads(receipt_payload)
        tampered_receipt["commit_marker_sha256"] = "0" * 64
        with pytest.raises(screen.ContractError, match="cross-binding"):
            screen.validate_durable_acceptance_receipt(
                bundle,
                accepted_info,
                marker,
                marker_info,
                tampered_receipt,
                receipt_info,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
    finally:
        os.close(directory_fd)


def test_final_canonical_held_descriptor_rejects_post_marker_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)
    marker_name = screen.acceptance_commit_marker_name(context["accepted_name"])
    displaced_marker = f"{marker_name}.hostile-displaced"

    def replace_after_first_final_reopen(stage: str) -> None:
        if stage != "after_final_canonical_path_validation":
            return
        os.rename(
            marker_name,
            displaced_marker,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        descriptor = os.open(
            marker_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=directory_fd,
        )
        os.write(descriptor, b"{}\n")
        os.fsync(descriptor)
        os.close(descriptor)
        os.fsync(directory_fd)

    try:
        with pytest.raises(screen.ContractError, match="inode-substituted"):
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=replace_after_first_final_reopen,
            )
        assert (
            screen._entry_stat_or_none(directory_fd, context["accepted_name"]) is None
        )
        assert screen._entry_stat_or_none(directory_fd, marker_name) is not None
        assert screen._entry_stat_or_none(directory_fd, displaced_marker) is not None
    finally:
        for name in (marker_name, displaced_marker):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
        os.close(directory_fd)


@pytest.mark.parametrize(
    ("artifact", "mutation_stage"),
    (
        ("report", "after_final_canonical_path_validation"),
        ("marker", "after_final_canonical_path_validation"),
        ("receipt", "after_final_acceptance_replay"),
    ),
)
def test_publication_rejects_same_inode_byte_mutation_through_final_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    mutation_stage: str,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)
    names = {
        "report": context["accepted_name"],
        "marker": screen.acceptance_commit_marker_name(context["accepted_name"]),
        "receipt": screen.durable_acceptance_receipt_name(context["accepted_name"]),
    }
    mutated_inode: int | None = None

    def mutate_same_inode(stage: str) -> None:
        nonlocal mutated_inode
        if stage != mutation_stage or mutated_inode is not None:
            return
        name = names[artifact]
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        os.chmod(name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, os.O_RDWR, dir_fd=directory_fd)
        try:
            original = os.pread(descriptor, before.st_size, 0)
            assert original
            replacement = bytes([original[0] ^ 1]) + original[1:]
            assert os.pwrite(descriptor, replacement, 0) == len(replacement)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(name, 0o444, dir_fd=directory_fd, follow_symlinks=False)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        assert (after.st_dev, after.st_ino, after.st_size) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
        )
        mutated_inode = after.st_ino

    try:
        with pytest.raises(screen.ContractError, match="byte-mutated"):
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=mutate_same_inode,
            )
        assert mutated_inode is not None
        retained = os.stat(names[artifact], dir_fd=directory_fd, follow_symlinks=False)
        assert retained.st_ino == mutated_inode
    finally:
        for name in os.listdir(directory_fd):
            try:
                os.chmod(name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            os.unlink(name, dir_fd=directory_fd)
        os.close(directory_fd)


def test_restart_replay_rejects_same_inode_mutation_before_final_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)
    screen.publish_accepted_bundle_exclusive(
        directory_fd,
        fixture["report"],
        verifier_receipt,
        fixture["manifest_bytes"],
        fixture["manifest_info"],
        fixture["delegated_marker_private_key"],
        failure_injector=_noop_publication_stage,
    )
    accepted_name = context["accepted_name"]
    before = os.stat(accepted_name, dir_fd=directory_fd, follow_symlinks=False)
    original_reader = screen._read_held_directory_entry_descriptor
    mutated = False

    def mutate_after_receipt_read(*args, **kwargs):
        nonlocal mutated
        result = original_reader(*args, **kwargs)
        if (
            kwargs.get("expected_payload") is None
            and args[3] == "publication receipt replay"
        ):
            os.chmod(
                accepted_name,
                0o600,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            descriptor = os.open(accepted_name, os.O_RDWR, dir_fd=directory_fd)
            try:
                payload = os.pread(descriptor, before.st_size, 0)
                assert os.pwrite(descriptor, bytes([payload[0] ^ 1]) + payload[1:], 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(
                accepted_name,
                0o444,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            mutated = True
        return result

    monkeypatch.setattr(
        screen, "_read_held_directory_entry_descriptor", mutate_after_receipt_read
    )
    try:
        with pytest.raises(screen.ContractError, match="final hold"):
            screen.read_and_validate_accepted_publication(
                directory_fd,
                accepted_name,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
        assert mutated
        after = os.stat(accepted_name, dir_fd=directory_fd, follow_symlinks=False)
        assert after.st_ino == before.st_ino
    finally:
        monkeypatch.setattr(
            screen,
            "_read_held_directory_entry_descriptor",
            original_reader,
        )
        for name in os.listdir(directory_fd):
            os.chmod(name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
            os.unlink(name, dir_fd=directory_fd)
        os.close(directory_fd)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires hard-exit fork semantics")
@pytest.mark.parametrize(
    "interruption_stage", ("after_report_parent_fsync", "after_final_readback")
)
def test_hard_interruption_before_commit_marker_is_not_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_stage: str,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)

    child = os.fork()
    if child == 0:

        def hard_exit(stage: str) -> None:
            if stage == interruption_stage:
                os._exit(73)

        try:
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=hard_exit,
            )
        except BaseException:
            os._exit(74)
        os._exit(75)

    try:
        _, status_code = os.waitpid(child, 0)
        assert os.WIFEXITED(status_code)
        assert os.WEXITSTATUS(status_code) == 73
        assert screen._entry_stat_or_none(directory_fd, context["accepted_name"])
        marker_name = screen.acceptance_commit_marker_name(context["accepted_name"])
        assert screen._entry_stat_or_none(directory_fd, marker_name) is None
        with pytest.raises(
            screen.ContractError, match="no durable post-publication commit marker"
        ):
            screen.read_and_validate_accepted_publication(
                directory_fd,
                context["accepted_name"],
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
    finally:
        os.close(directory_fd)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires hard-exit fork semantics")
def test_hard_interruption_at_pre_parent_fsync_pair_is_not_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)

    child = os.fork()
    if child == 0:

        def hard_exit(stage: str) -> None:
            if stage == "before_commit_marker_parent_fsync":
                os._exit(76)

        try:
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=hard_exit,
            )
        except BaseException:
            os._exit(77)
        os._exit(78)

    try:
        _, status_code = os.waitpid(child, 0)
        assert os.WIFEXITED(status_code)
        assert os.WEXITSTATUS(status_code) == 76
        marker_name = screen.acceptance_commit_marker_name(context["accepted_name"])
        receipt_name = screen.durable_acceptance_receipt_name(context["accepted_name"])
        assert screen._entry_stat_or_none(directory_fd, context["accepted_name"])
        assert screen._entry_stat_or_none(directory_fd, marker_name)
        receipt_info = screen._entry_stat_or_none(directory_fd, receipt_name)
        assert receipt_info is not None and receipt_info.st_size == 0
        with pytest.raises(
            screen.ContractError,
            match="no valid durable post-fsync acceptance receipt",
        ):
            screen.read_and_validate_accepted_publication(
                directory_fd,
                context["accepted_name"],
                fixture["manifest_bytes"],
                fixture["manifest_info"],
            )
    finally:
        os.close(directory_fd)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires hard-exit fork semantics")
def test_hard_interruption_after_o_sync_receipt_write_remains_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)

    child = os.fork()
    if child == 0:

        def hard_exit(stage: str) -> None:
            if stage == "after_durable_acceptance_receipt_sync_write":
                os._exit(79)

        try:
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=hard_exit,
            )
        except BaseException:
            os._exit(80)
        os._exit(81)

    try:
        _, status_code = os.waitpid(child, 0)
        assert os.WIFEXITED(status_code)
        assert os.WEXITSTATUS(status_code) == 79
        publication = screen.read_and_validate_accepted_publication(
            directory_fd,
            context["accepted_name"],
            fixture["manifest_bytes"],
            fixture["manifest_info"],
        )
        durable_receipt = publication[-1]
        assert durable_receipt["status"] == (
            "wrapper_durable_post_fsync_acceptance_complete"
        )
        assert all(durable_receipt["durability_checks"].values())
    finally:
        os.close(directory_fd)


def test_full_schema_acceptance_pathname_substitution_rolls_back_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    verifier_receipt = _publish_candidate_and_build_verifier_receipt(fixture)
    output_directory = fixture["output_directory"]
    assert isinstance(output_directory, Path)
    moved_directory = (tmp_path / "moved-output-inode").resolve()

    def substitute_path(stage: str) -> None:
        if stage != "after_report_rename":
            return
        output_directory.rename(moved_directory)
        output_directory.mkdir(mode=0o700)
        output_directory.chmod(0o700)

    try:
        with pytest.raises(screen.ContractError, match="pathname no longer names"):
            screen.publish_accepted_bundle_exclusive(
                directory_fd,
                fixture["report"],
                verifier_receipt,
                fixture["manifest_bytes"],
                fixture["manifest_info"],
                fixture["delegated_marker_private_key"],
                failure_injector=substitute_path,
            )
        assert not (moved_directory / context["accepted_name"]).exists()
        assert not (output_directory / context["accepted_name"]).exists()
        assert not any(
            ".r12-report-" in path.name or ".r12-commit-" in path.name
            for path in moved_directory.iterdir()
        )
    finally:
        os.close(directory_fd)


def _authorize_stale_cleanup(
    fixture: dict[str, object], entries: list[dict[str, object]]
) -> dict[str, object]:
    context = fixture["context"]
    assert isinstance(context, dict)
    payload = screen.run_authorization_payload(context["run_authorization"])
    payload["stale_cleanup_entries"] = entries
    authorization = screen.sign_run_authorization(payload, TEST_AUTHORITY_PRIVATE_KEY)
    screen.validate_run_authorization(
        authorization,
        expected_slurm=context["slurm_identity"],
        expected_output_directory=context["output_directory"],
        expected_accepted_name=context["accepted_name"],
        expected_source_manifest_sha256=context["source_manifest_sha256"],
        expected_delegated_marker_key=context["delegated_marker_signing_key"],
        require_current=False,
    )
    return authorization


def test_stale_cleanup_requires_external_exact_inode_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    accepted_name = context["accepted_name"]
    job_id = context["slurm_identity"]["job_id"]
    nonce = context["run_authorization"]["authorization_nonce"]
    stale_names = (
        f".{accepted_name}.r12-candidate-{job_id}-{nonce}",
        f"..{accepted_name}.r12-candidate-{job_id}-{nonce}.tmp-{nonce}-{'b' * 32}",
        f".{accepted_name}.r12-report-{nonce}-{'d' * 32}",
        (
            f".{screen.acceptance_commit_marker_name(accepted_name)}.r12-commit-"
            f"{nonce}-{'d' * 32}"
        ),
    )
    unrelated = "unrelated.keep"
    lease = screen.acquire_publisher_lease(
        directory_fd,
        context["output_directory"],
        accepted_name,
        job_id,
        nonce,
    )
    try:
        for name in (*stale_names, unrelated):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            os.write(descriptor, name.encode("ascii"))
            os.close(descriptor)
        entries = sorted(
            (
                screen._stale_cleanup_record(
                    directory_fd, name, "test stale cleanup entry"
                )[0]
                for name in stale_names
            ),
            key=lambda entry: entry["name"],
        )
        authorization = _authorize_stale_cleanup(fixture, entries)
        quarantined = screen.cleanup_stale_publication_entries(
            directory_fd, accepted_name, authorization, lease
        )
        assert len(quarantined) == len(stale_names)
        assert all(name not in os.listdir(directory_fd) for name in stale_names)
        assert sorted(os.listdir(directory_fd)) == sorted((unrelated, *quarantined))
        for source_entry, quarantine_name in zip(entries, quarantined):
            quarantine_entry = screen._stale_cleanup_record(
                directory_fd, quarantine_name, "test stale quarantine"
            )[0]
            assert {**quarantine_entry, "name": source_entry["name"]} == source_entry
    finally:
        screen.release_publisher_lease(directory_fd, lease)
        os.close(directory_fd)


def test_publisher_lease_rejects_concurrent_live_owner(tmp_path: Path) -> None:
    fixture = _full_schema_fixture(tmp_path)
    first_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(first_fd, int)
    assert isinstance(context, dict)
    second_fd, _record = screen.open_owned_output_directory(fixture["output_directory"])
    lease = screen.acquire_publisher_lease(
        first_fd,
        context["output_directory"],
        context["accepted_name"],
        context["slurm_identity"]["job_id"],
        context["run_authorization"]["authorization_nonce"],
    )
    try:
        concurrent = screen._linux_smoke_concurrent_publisher_lease(
            fixture["output_directory"],
            first_fd,
            context["output_directory"],
            context["accepted_name"],
            context["slurm_identity"]["job_id"],
            context["run_authorization"]["authorization_nonce"],
        )
        assert concurrent["result"] == ("exclusive_flock_rejected_concurrent_process")
        with pytest.raises(screen.ContractError, match="live publisher"):
            screen.acquire_publisher_lease(
                second_fd,
                context["output_directory"],
                context["accepted_name"],
                context["slurm_identity"]["job_id"],
                context["run_authorization"]["authorization_nonce"],
            )
    finally:
        screen.release_publisher_lease(first_fd, lease)
        os.close(second_fd)
        os.close(first_fd)


def test_stale_cleanup_preserves_substituted_foreign_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    assert isinstance(directory_fd, int)
    assert isinstance(context, dict)
    accepted_name = context["accepted_name"]
    job_id = context["slurm_identity"]["job_id"]
    nonce = context["run_authorization"]["authorization_nonce"]
    name = f".{accepted_name}.r12-candidate-{job_id}-{nonce}"
    path = fixture["output_directory"] / name
    path.write_bytes(b"authorized stale bytes")
    authorized_record = screen._stale_cleanup_record(
        directory_fd, name, "authorized stale cleanup entry"
    )[0]
    authorization = _authorize_stale_cleanup(fixture, [authorized_record])
    path.unlink()
    path.write_bytes(b"foreign replacement must survive")
    lease = screen.acquire_publisher_lease(
        directory_fd, context["output_directory"], accepted_name, job_id, nonce
    )
    try:
        with pytest.raises(screen.ContractError, match="inode or bytes differ"):
            screen.cleanup_stale_publication_entries(
                directory_fd, accepted_name, authorization, lease
            )
        assert path.read_bytes() == b"foreign replacement must survive"
    finally:
        screen.release_publisher_lease(directory_fd, lease)
        os.close(directory_fd)


def test_stale_cleanup_never_unlinks_foreign_inode_substituted_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    accepted_name = context["accepted_name"]
    job_id = context["slurm_identity"]["job_id"]
    nonce = context["run_authorization"]["authorization_nonce"]
    name = f".{accepted_name}.r12-candidate-{job_id}-{nonce}"
    source_path = fixture["output_directory"] / name
    source_path.write_bytes(b"authorized-stale-inode\n")
    entry = screen._stale_cleanup_record(
        directory_fd, name, "authorized stale cleanup entry"
    )[0]
    authorization = _authorize_stale_cleanup(fixture, [entry])
    lease = screen.acquire_publisher_lease(
        directory_fd, context["output_directory"], accepted_name, job_id, nonce
    )
    displaced_name = ".authorized-inode-preserved-for-review"
    substituted_name = None

    def substitute(stage: str, _source: str, destination: str) -> None:
        nonlocal substituted_name
        assert stage == "after_quarantine_rename"
        os.rename(
            destination,
            displaced_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        os.write(descriptor, b"foreign-inode-must-survive\n")
        os.close(descriptor)
        substituted_name = destination

    try:
        with pytest.raises(screen.ContractError, match="preserved"):
            screen.cleanup_stale_publication_entries(
                directory_fd,
                accepted_name,
                authorization,
                lease,
                failure_injector=substitute,
            )
        assert substituted_name is not None
        assert (fixture["output_directory"] / substituted_name).read_bytes() == (
            b"foreign-inode-must-survive\n"
        )
        assert (fixture["output_directory"] / displaced_name).read_bytes() == (
            b"authorized-stale-inode\n"
        )
    finally:
        screen.release_publisher_lease(directory_fd, lease)
        os.close(directory_fd)


def test_stale_cleanup_recovers_hard_death_after_quarantine_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    accepted_name = context["accepted_name"]
    job_id = context["slurm_identity"]["job_id"]
    nonce = context["run_authorization"]["authorization_nonce"]
    name = f".{accepted_name}.r12-report-{nonce}-{'e' * 32}"
    path = fixture["output_directory"] / name
    path.write_bytes(b"hard-death-stale-inode\n")
    entry = screen._stale_cleanup_record(
        directory_fd, name, "hard-death stale cleanup entry"
    )[0]
    authorization = _authorize_stale_cleanup(fixture, [entry])
    quarantine_name = screen._stale_quarantine_name(accepted_name, job_id, nonce, entry)
    child_pid = os.fork()
    if child_pid == 0:
        child_fd = None
        try:
            os.close(directory_fd)
            child_fd, child_record = screen.open_owned_output_directory(
                fixture["output_directory"]
            )
            if child_record != context["output_directory"]:
                os._exit(89)
            child_lease = screen.acquire_publisher_lease(
                child_fd,
                context["output_directory"],
                accepted_name,
                job_id,
                nonce,
            )
            screen.cleanup_stale_publication_entries(
                child_fd,
                accepted_name,
                authorization,
                child_lease,
                failure_injector=lambda *_args: os._exit(88),
            )
            os._exit(89)
        except BaseException:
            os._exit(89)
    waited_pid, child_status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.WIFEXITED(child_status)
    assert os.WEXITSTATUS(child_status) == 88

    assert not path.exists()
    quarantine_path = fixture["output_directory"] / quarantine_name
    assert quarantine_path.read_bytes() == b"hard-death-stale-inode\n"
    quarantine_entry = screen._stale_cleanup_record(
        directory_fd, quarantine_name, "recovered stale quarantine"
    )[0]
    recovery_authorization = _authorize_stale_cleanup(fixture, [quarantine_entry])
    recovery_lease = screen.acquire_publisher_lease(
        directory_fd, context["output_directory"], accepted_name, job_id, nonce
    )
    try:
        assert screen.cleanup_stale_publication_entries(
            directory_fd,
            accepted_name,
            recovery_authorization,
            recovery_lease,
        ) == [quarantine_name]
        assert quarantine_path.read_bytes() == b"hard-death-stale-inode\n"
    finally:
        screen.release_publisher_lease(directory_fd, recovery_lease)
        os.close(directory_fd)


def test_restart_discovers_signed_quarantine_from_prior_job_and_nonce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "rename_noreplace_at", _test_rename_noreplace_at)
    fixture = _full_schema_fixture(tmp_path)
    directory_fd = fixture["directory_fd"]
    context = fixture["context"]
    accepted_name = context["accepted_name"]
    current_job_id = context["slurm_identity"]["job_id"]
    current_nonce = context["run_authorization"]["authorization_nonce"]
    prior_job_id = "99999"
    prior_nonce = "b" * 64
    prior_name = f".{accepted_name}.r12-candidate-{prior_job_id}-{prior_nonce}"
    prior_path = fixture["output_directory"] / prior_name
    prior_path.write_bytes(b"prior-run-quarantine\n")
    prior_entry = screen._stale_cleanup_record(
        directory_fd, prior_name, "prior-run stale entry"
    )[0]
    quarantine_name = screen._stale_quarantine_name(
        accepted_name, prior_job_id, prior_nonce, prior_entry
    )
    os.rename(
        prior_name,
        quarantine_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    quarantine_entry = screen._stale_cleanup_record(
        directory_fd, quarantine_name, "prior-run quarantine"
    )[0]
    rollback_source_name = "prior-rollback-source"
    rollback_payload = b"prior-run-rollback-quarantine\n"
    rollback_path = fixture["output_directory"] / rollback_source_name
    rollback_path.write_bytes(rollback_payload)
    rollback_path.chmod(0o444)
    rollback_info = rollback_path.stat()
    rollback_name = screen._rollback_quarantine_name(
        accepted_name,
        prior_nonce,
        rollback_source_name,
        rollback_info,
        rollback_payload,
    )
    os.rename(
        rollback_source_name,
        rollback_name,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    rollback_entry = screen._stale_cleanup_record(
        directory_fd, rollback_name, "prior-run rollback quarantine"
    )[0]
    entries = sorted(
        (quarantine_entry, rollback_entry), key=lambda entry: entry["name"]
    )
    authorization = _authorize_stale_cleanup(fixture, entries)
    lease = screen.acquire_publisher_lease(
        directory_fd,
        context["output_directory"],
        accepted_name,
        current_job_id,
        current_nonce,
    )
    try:
        observed_quarantines = screen.cleanup_stale_publication_entries(
            directory_fd, accepted_name, authorization, lease
        )
        terminal_rollback_name = screen._stale_quarantine_name(
            accepted_name,
            current_job_id,
            current_nonce,
            rollback_entry,
        )
        expected_by_source = {
            quarantine_name: quarantine_name,
            rollback_name: terminal_rollback_name,
        }
        assert observed_quarantines == [
            expected_by_source[entry["name"]] for entry in entries
        ]
        assert (fixture["output_directory"] / quarantine_name).read_bytes() == (
            b"prior-run-quarantine\n"
        )
        assert not (fixture["output_directory"] / rollback_name).exists()
        assert (
            fixture["output_directory"] / terminal_rollback_name
        ).read_bytes() == rollback_payload
    finally:
        screen.release_publisher_lease(directory_fd, lease)
        os.close(directory_fd)


def test_compound_fresh_gates_are_width_separated_and_nonpromotional() -> None:
    gates = screen.FRESH_REENCODING_GATES
    assert gates["carry_paired_switch_each_width_min"] == 0.70
    assert gates["paired_recovery_vs_full_history_lf_each_width_min"] == 0.30
    contract = screen.frozen_contract()
    assert contract["fresh_latest_reencoding"]["resource_boundary"] == (
        "compound_context_removal_position_reset_and_surface_canonicalization"
    )
    assert contract["posthoc_kv_slicing_negative_control"]["status"] == (
        "descriptive_supplied_development_negative_not_reexecuted"
    )
    assert contract["external_controller_ceiling"]["status"] == (
        "descriptive_not_executed"
    )
    assert "No promotion" in screen.CLAIM_BOUNDARY


def test_candidate_retirement_contract_is_rename_only_retained_evidence() -> None:
    publication = screen.frozen_contract()["publication"]
    assert publication["candidate_creation_descriptor_retained_through_consumption"]
    assert publication["candidate_readonly_verifier_descriptor_bound_before_rename"]
    assert publication["candidate_consumption_rename_only_retained_quarantine"]
    assert publication["post_rename_failure_retains_exact_inode_in_quarantine"]
    assert "temp_unlink_then_parent_fsync" not in publication
    assert "post_rename_failure_removes_exact_inode_and_fsyncs_parent" not in (
        publication
    )

    publisher = inspect.getsource(screen.publish_private_candidate_exclusive)
    quarantine = inspect.getsource(screen._quarantine_held_entry_if_exact)
    wrapper = (ROOT / "train/jobs/eval_dws_eos_suppressed_trace.sbatch").read_text()
    prereg = (ROOT / "R12_DWS_EOS_SUPPRESSED_TRACE_PREREG.md").read_text()
    assert "os.unlink" not in publisher
    assert "os.unlink" not in quarantine
    assert "os.unlink(candidate_name" not in wrapper
    assert "unlinks the now-verified private candidate" not in prereg
    assert "retained rollback quarantine" in " ".join(prereg.split())
