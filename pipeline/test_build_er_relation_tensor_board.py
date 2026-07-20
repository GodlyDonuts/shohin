from __future__ import annotations

from pathlib import Path

import pytest

from build_er_relation_tensor_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
    validate_row,
    write_board,
)
from er_relation_tensor_renderers import independently_execute, parse_rendered_row


FAMILIES = {
    TRAIN_SPLIT: 48,
    DEVELOPMENT_SPLIT: 24,
    CONFIRMATION_SPLIT: 24,
}


def test_fixture_board_passes_every_gate_and_independent_roundtrip() -> None:
    splits, report = build_board(seed=44_901, families=FAMILIES)
    assert report["all_gates_pass"] is True
    assert report["rows"] == {
        TRAIN_SPLIT: 192,
        DEVELOPMENT_SPLIT: 96,
        CONFIRMATION_SPLIT: 96,
    }
    assert report["non_bijective_family_rate"] >= 0.90
    assert report["maximum_program_bytes"] <= 640
    assert report["maximum_line_bytes"] <= 144
    assert report["controls"]["family_deranged_state_rate"] < 0.40
    assert report["controls"]["equality_ablated_state_rate"] < 0.40
    for rows in splits.values():
        for row in rows:
            validate_row(row)
            parsed = parse_rendered_row(row)
            executed = independently_execute(row)
            assert parsed["cardinality"] == row["compiler_targets"]["cardinality"]
            if row["oracle"] is not None:
                bindings = tuple(row["compiler_targets"]["entity_bindings"])
                roles = [bindings.index(value) for value in executed["final_state"]]
                assert roles == row["oracle"]["final_state"]


def test_fixture_board_is_deterministic() -> None:
    first = build_board(seed=72_111, families=FAMILIES)
    second = build_board(seed=72_111, families=FAMILIES)
    assert first == second


def test_confirmation_is_sealed_and_existing_output_is_refused(tmp_path: Path) -> None:
    splits, report = build_board(seed=81_337, families=FAMILIES)
    output = tmp_path / "board"
    final = write_board(output, splits, report)
    assert final["all_gates_pass"] is True
    assert (output / "confirmation.jsonl").stat().st_mode & 0o777 == 0o600
    assert len((output / "train.jsonl").read_text().splitlines()) == 192
    with pytest.raises(FileExistsError):
        write_board(output, splits, report)


def test_train_rows_never_contain_outcome_oracle() -> None:
    splits, _ = build_board(seed=99_031, families=FAMILIES)
    for row in splits[TRAIN_SPLIT]:
        assert row["oracle"] is None
        assert row["supervision"] == "compiler_fields_only"
        assert "final_state" not in row["compiler_targets"]
        assert "answer_role" not in row["compiler_targets"]
