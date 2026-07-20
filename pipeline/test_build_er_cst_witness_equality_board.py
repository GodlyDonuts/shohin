from __future__ import annotations

from build_er_cst_witness_equality_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    PROTOCOL,
    TRAIN_SPLIT,
    build_board,
)


SMALL_FAMILIES = {
    TRAIN_SPLIT: 8,
    DEVELOPMENT_SPLIT: 8,
    CONFIRMATION_SPLIT: 8,
}


def test_fresh_witness_board_exposes_only_occurrence_spans() -> None:
    splits, report = build_board(seed=2_604_721, families=SMALL_FAMILIES)
    assert report["all_gates_pass"] is True
    assert report["gates"]["all_witness_ranges_exact"] is True
    for split, rows in splits.items():
        for row in rows:
            assert row["protocol"] == PROTOCOL
            target = row["compiler_targets"]
            assert len(target["witness_before_ranges"]) == 3
            assert len(target["witness_after_ranges"]) == 3
            assert all(len(value) == 3 for value in target["witness_before_ranges"])
            assert all(len(value) == 3 for value in target["witness_after_ranges"])
            if split == TRAIN_SPLIT:
                assert "oracle" not in row


def test_witness_spans_follow_random_physical_storage() -> None:
    splits, _ = build_board(seed=3_197_411, families=SMALL_FAMILIES)
    row = splits[TRAIN_SPLIT][0]
    target = row["compiler_targets"]
    source = row["program_text"].encode("utf-8")
    for rule, before, after in zip(
        sorted(target["rule_cards"], key=lambda item: int(item["slot"])),
        target["witness_before_ranges"],
        target["witness_after_ranges"],
        strict=True,
    ):
        assert [source[start:end].decode() for start, end in before] == rule["before"]
        assert [source[start:end].decode() for start, end in after] == rule["after"]
