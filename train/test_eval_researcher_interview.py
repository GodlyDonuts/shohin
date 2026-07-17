#!/usr/bin/env python3
"""Deterministic scoring tests for the quarantined Researcher Interview v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_researcher_interview import (
    evaluate_bank,
    load_bank,
    score_turn,
    write_json_no_replace,
)


BANK_PATH = (
    Path(__file__).parents[1] / "artifacts" / "evals" / "researcher_interview_v1.json"
)


def bank():
    return load_bank(BANK_PATH)


def test_frozen_bank_contract_and_quarantine():
    frozen = bank()
    assert [case["id"] for case in frozen["cases"]] == [
        f"RIV{i:02d}" for i in range(1, 21)
    ]
    assert sum(len(case["turns"]) for case in frozen["cases"]) == 21
    assert sum(case["parity"] == "odd" for case in frozen["cases"]) == 10
    assert frozen["quarantine"]["training_use"] == "forbidden"
    assert frozen["freshness_audit"]["exact_normalized_13_word_hits"] == 0


def test_all_gold_outputs_score_perfect_and_riv20_deletes_source():
    frozen = bank()
    outputs = {
        turn["prompt"]: turn["expected_output"]
        for case in frozen["cases"]
        for turn in case["turns"]
        if case["id"] != "RIV20" or turn["id"] == "writer"
    }
    reader = frozen["cases"][-1]["turns"][1]
    reader_prompt = f"{reader['gold_capsule']}\n{reader['prompt']}"
    outputs[reader_prompt] = reader["expected_output"]
    seen = []

    def ask(prompt):
        seen.append(prompt)
        return outputs[prompt]

    result = evaluate_bank(frozen, ask)
    assert result["summary"]["overall"]["normalized_exact_syntax_correct"] == 20
    assert result["summary"]["overall"]["semantic_state_correct"] == 20
    assert result["summary"]["odd_local"]["semantic_state_correct"] == 10
    assert result["summary"]["even_composition"]["semantic_state_correct"] == 10
    assert result["riv20"]["writer"]["semantic_state_correct"]
    assert result["riv20"]["reader_gold_capsule"]["semantic_state_correct"]
    assert result["riv20"]["end_to_end"]["semantic_state_correct"]
    source_deleted = [prompt for prompt in seen if "RIV20-B" in prompt]
    assert len(source_deleted) == 2
    assert all(
        prompt.startswith("memo{r=225;seal=KITE}\nRIV20-B") for prompt in source_deleted
    )
    assert all("Begin with r=31" not in prompt for prompt in source_deleted)


def test_semantics_divergence_termination_and_unchanged_corruption_are_separate():
    frozen = bank()
    riv01 = frozen["cases"][0]["turns"][0]
    scored = score_turn(riv01, "Work follows.\nquill = 85\nDone.")
    assert scored["semantic_state_correct"]
    assert not scored["normalized_exact_syntax_correct"]
    assert scored["extra_token_termination_failure"]
    assert not scored["format_only_failure"]

    formatted_only = score_turn(riv01, "quill = 85")
    assert formatted_only["semantic_state_correct"]
    assert formatted_only["format_only_failure"]
    assert not formatted_only["extra_token_termination_failure"]

    riv02 = frozen["cases"][1]["turns"][0]
    divergent = score_turn(riv02, "q0=58;q1=84;q2=336;q3=303")
    assert divergent["first_divergent_transition"]["index"] == 1
    assert divergent["first_divergent_transition"]["name"] == "add_27"
    assert divergent["observable_transitions"][0]["correct"]

    riv11 = frozen["cases"][10]["turns"][0]
    corrupted = score_turn(riv11, "tiles=ZNMLQ")
    assert corrupted["unchanged_field_check"]["scorable"]
    assert corrupted["unchanged_field_check"]["corrupted"]
    assert corrupted["unchanged_field_check"]["details"][0]["field"] == "tiles"


def test_riv20_separates_writer_reader_and_end_to_end_when_writer_is_malformed():
    frozen = bank()
    writer_prompt = frozen["cases"][-1]["turns"][0]["prompt"]
    reader = frozen["cases"][-1]["turns"][1]
    gold_reader_prompt = f"{reader['gold_capsule']}\n{reader['prompt']}"

    def ask(prompt):
        if prompt == writer_prompt:
            return "I cannot produce a capsule."
        if prompt == gold_reader_prompt:
            return "r=194"
        case = next(
            case
            for case in frozen["cases"]
            if case["id"] != "RIV20" and case["turns"][0]["prompt"] == prompt
        )
        return case["turns"][0]["expected_output"]

    result = evaluate_bank(frozen, ask)
    assert not result["riv20"]["writer"]["semantic_state_correct"]
    assert result["riv20"]["reader_gold_capsule"]["semantic_state_correct"]
    assert not result["riv20"]["end_to_end"]["semantic_state_correct"]
    riv20 = result["rows"][-1]
    assert riv20["reader_model_capsule"] is None
    assert riv20["source_deleted_contract"][
        "end_to_end_reader_skipped_without_parseable_capsule"
    ]


def test_atomic_json_publication_refuses_existing_path(tmp_path):
    output = tmp_path / "result.json"
    write_json_no_replace(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_json_no_replace(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
