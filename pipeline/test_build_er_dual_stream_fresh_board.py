from __future__ import annotations

from pathlib import Path

import pytest

from build_er_dual_stream_fresh_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    NEUTRAL_TOKEN,
    TRAIN_SPLIT,
    build_board,
    validate_row,
    write_board,
)
from er_dual_stream_fresh_renderers import independently_execute, parse_rendered_row


FAMILIES = {
    TRAIN_SPLIT: 48,
    DEVELOPMENT_SPLIT: 24,
    CONFIRMATION_SPLIT: 24,
}


def test_fixture_board_passes_all_gates_and_public_roundtrip() -> None:
    splits, report = build_board(seed=71_903, families=FAMILIES)
    assert report["all_gates_pass"] is True
    assert report["rows"] == {
        TRAIN_SPLIT: 192,
        DEVELOPMENT_SPLIT: 96,
        CONFIRMATION_SPLIT: 96,
    }
    assert report["controls"]["distractor_swap_rate"] == 1.0
    assert report["maximum_program_bytes"] <= 640
    assert report["maximum_line_bytes"] <= 144
    assert not set(report["renderers"]["train"]) & set(report["renderers"]["scored"])
    for rows in splits.values():
        for row in rows:
            validate_row(row)
            parsed = parse_rendered_row(row)
            executed = independently_execute(row)
            assert len(parsed["distractors"]) == 6 - row["compiler_targets"]["rule_count"]
            assert executed["rule_relations"] == {
                item["slot"]: tuple(item["relation"])
                for item in row["compiler_targets"]["rule_cards"]
                if item["active"]
            }


def test_ordinal_candidate_counts_are_exact_despite_distractors() -> None:
    splits, _ = build_board(seed=84_117, families=FAMILIES)
    for rows in splits.values():
        for row in rows:
            target = row["compiler_targets"]
            cardinality = target["cardinality"]
            source = row["program_text"]
            semantic_lines = [
                source[start:end] for start, end in target["line_ranges"]
            ]
            assert len(NEUTRAL_TOKEN.findall(semantic_lines[0])) == 2 * cardinality
            for slot in range(4):
                expected = 2 * cardinality + 1 if slot < target["rule_count"] else 1
                assert len(NEUTRAL_TOKEN.findall(semantic_lines[slot + 1])) == expected
            event_counts = [
                len(NEUTRAL_TOKEN.findall(line)) for line in semantic_lines[5:]
            ]
            assert sum(event_counts) == 13
            assert sum(count == 2 for count in event_counts) <= 1
            assert len(NEUTRAL_TOKEN.findall(row["late_query_text"])) == 1


def test_fixture_board_is_deterministic() -> None:
    first = build_board(seed=91_441, families=FAMILIES)
    second = build_board(seed=91_441, families=FAMILIES)
    assert first == second


def test_confirmation_is_sealed_and_existing_output_is_refused(tmp_path: Path) -> None:
    splits, report = build_board(seed=97_633, families=FAMILIES)
    output = tmp_path / "board"
    final = write_board(output, splits, report)
    assert final["all_gates_pass"] is True
    assert (output / "confirmation.jsonl").stat().st_mode & 0o777 == 0o600
    assert len((output / "train.jsonl").read_text().splitlines()) == 192
    with pytest.raises(FileExistsError):
        write_board(output, splits, report)


def test_training_rows_never_contain_outcomes() -> None:
    splits, _ = build_board(seed=105_019, families=FAMILIES)
    for row in splits[TRAIN_SPLIT]:
        assert row["oracle"] is None
        assert row["supervision"] == "compiler_fields_only"
        assert "final_state" not in row["compiler_targets"]
        assert "answer_role" not in row["compiler_targets"]
