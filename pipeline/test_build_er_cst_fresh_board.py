from __future__ import annotations

import json
from pathlib import Path

from build_er_cst_fresh_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
    sha256_bytes,
    write_board,
)


SMALL_FAMILIES = {
    TRAIN_SPLIT: 24,
    DEVELOPMENT_SPLIT: 24,
    CONFIRMATION_SPLIT: 24,
}


def test_small_board_is_deterministic_and_passes_every_gate() -> None:
    first_splits, first_report = build_board(seed=123_456, families=SMALL_FAMILIES)
    second_splits, second_report = build_board(seed=123_456, families=SMALL_FAMILIES)
    assert first_splits == second_splits
    assert first_report == second_report
    assert first_report["all_gates_pass"] is True
    assert all(first_report["gates"].values())
    assert all(value == 0 for overlap in first_report["overlap"].values() for value in overlap.values())


def test_training_rows_contain_no_answer_state_or_trajectory() -> None:
    splits, _ = build_board(seed=654_321, families=SMALL_FAMILIES)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in splits[TRAIN_SPLIT])
    assert '"oracle"' not in payload
    assert "final_state" not in payload
    assert "answer_role" not in payload
    assert "trajectory_roles" not in payload
    assert all(row["supervision"] == "compiler_fields_only" for row in splits[TRAIN_SPLIT])


def test_depth_eight_has_eight_updates_followed_by_explicit_halt() -> None:
    splits, _ = build_board(seed=111_222, families=SMALL_FAMILIES)
    rows = [
        row
        for row in splits[DEVELOPMENT_SPLIT]
        if int(row["compiler_targets"]["depth"]) == 8
    ]
    assert rows
    for row in rows:
        events = sorted(row["compiler_targets"]["events"], key=lambda item: int(item["slot"]))
        assert [bool(item["halt"]) for item in events] == [False] * 8 + [True]
        assert len(row["oracle"]["trajectory_roles"]) == 9


def test_board_writer_hashes_and_seals_confirmation(tmp_path: Path) -> None:
    splits, report = build_board(seed=333_444, families=SMALL_FAMILIES)
    output = tmp_path / "board"
    value = write_board(
        output=output,
        source_commit="0" * 40,
        splits=splits,
        report=report,
    )
    assert value["all_gates_pass"] is True
    for filename in ("train.jsonl", "development.jsonl", "confirmation.jsonl"):
        payload = (output / filename).read_bytes()
        assert value["files"][filename]["sha256"] == sha256_bytes(payload)
    assert (output / "confirmation.jsonl").stat().st_mode & 0o777 == 0o600
