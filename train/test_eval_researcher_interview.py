#!/usr/bin/env python3
"""Deterministic scoring tests for the quarantined Researcher Interview v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_researcher_interview import (
    evaluate_bank,
    load_bank,
    load_comparison_manifest,
    present_prompt,
    score_turn,
    write_json_no_replace,
)


BANK_PATH = (
    Path(__file__).parents[1] / "artifacts" / "evals" / "researcher_interview_v1.json"
)
COMPARISON_PATH = (
    Path(__file__).parents[1]
    / "artifacts"
    / "evals"
    / "researcher_interview_comparison_v1.json"
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


def test_comparison_manifest_binds_matched_parent_and_descriptive_role(tmp_path):
    manifest, treatment = load_comparison_manifest(
        COMPARISON_PATH,
        bank_sha256="a72387a0a72418f119bf35791032bb889266b3f9ba8a3b728fc7f6978c0d4f8d",
        presentation="qa",
        role="drs_r3_treatment",
        checkpoint_sha256="d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459",
    )
    assert treatment["lineage_parent_sha256"] == manifest["matched_pair"][
        "common_parent_sha256"
    ]
    assert not manifest["matched_pair"]["lineage_evidence"][
        "checkpoint_self_attests_parent"
    ]

    _, descriptive = load_comparison_manifest(
        COMPARISON_PATH,
        bank_sha256=manifest["bank_sha256"],
        presentation="qa",
        role="raw300k_descriptive",
        checkpoint_sha256="211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6",
    )
    assert descriptive["comparison_scope"] == "descriptive_only"

    with pytest.raises(ValueError, match="lineage parent"):
        bad = json.loads(COMPARISON_PATH.read_text())
        bad["entries"]["drs_r3_treatment"]["lineage_parent_sha256"] = "0" * 64
        path = tmp_path / "invalid-comparison-manifest.json"
        try:
            path.write_text(json.dumps(bad))
            load_comparison_manifest(
                path,
                bank_sha256=manifest["bank_sha256"],
                presentation="qa",
                role="drs_r3_treatment",
                checkpoint_sha256=treatment["checkpoint_sha256"],
            )
        finally:
            path.unlink(missing_ok=True)


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
    malformed_writer = "\nI cannot produce a capsule. \n"
    malformed_reader_prompt = malformed_writer + "\n" + reader["prompt"]
    seen = []

    def ask(prompt):
        seen.append(prompt)
        if prompt == writer_prompt:
            return malformed_writer
        if prompt == gold_reader_prompt:
            return "r=194"
        if prompt == malformed_reader_prompt:
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
    assert riv20["reader_model_capsule"]["semantic_state_correct"]
    assert malformed_reader_prompt in seen
    assert not riv20["source_deleted_contract"][
        "end_to_end_reader_skipped_without_parseable_capsule"
    ]
    assert not riv20["source_deleted_contract"][
        "parser_sanitized_model_transcript"
    ]
    assert (
        riv20["source_deleted_contract"]["model_authored_transcript"]
        == malformed_writer
    )


def test_riv20_source_leaking_writer_cannot_earn_semantic_end_to_end_success():
    frozen = bank()
    writer_turn, reader = frozen["cases"][-1]["turns"]
    leaked = (
        "Begin with r=31. Add 14, then multiply by 5. "
        "memo{r=225;seal=KITE}"
    )
    gold_reader_prompt = f"{reader['gold_capsule']}\n{reader['prompt']}"
    leaked_reader_prompt = f"{leaked}\n{reader['prompt']}"

    def ask(prompt):
        if prompt == writer_turn["prompt"]:
            return leaked
        if prompt in {gold_reader_prompt, leaked_reader_prompt}:
            return "r=194"
        case = next(
            case
            for case in frozen["cases"]
            if case["id"] != "RIV20" and case["turns"][0]["prompt"] == prompt
        )
        return case["turns"][0]["expected_output"]

    riv20 = evaluate_bank(frozen, ask)["rows"][-1]
    assert riv20["writer"]["semantic_state_correct"]
    assert not riv20["writer"]["normalized_exact_syntax_correct"]
    assert riv20["reader_model_capsule"]["semantic_state_correct"]
    assert not riv20["semantic_state_correct"]
    assert riv20["first_divergent_transition"]["phase"] == "writer_commit"
    assert not riv20["source_deleted_contract"]["writer_commit_valid"]


def test_prompt_presentation_is_explicit_and_deterministic():
    assert present_prompt("work", "raw") == "work"
    assert present_prompt("work", "qa") == "Question: work\nAnswer:"
    with pytest.raises(ValueError):
        present_prompt("work", "chat")


def test_atomic_json_publication_refuses_existing_path(tmp_path):
    output = tmp_path / "result.json"
    write_json_no_replace(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_json_no_replace(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}
