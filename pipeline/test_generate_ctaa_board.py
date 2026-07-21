from __future__ import annotations

from collections import Counter

from pipeline.generate_ctaa_board import (
    MAX_STEPS,
    RENDERERS,
    SCORED_DEPTHS,
    SPLIT_COUNTS,
    STOP_ID,
    TRAIN_DEPTHS,
    build_rows,
    make_family,
    render_row,
    semantic_splits,
)
from pipeline.ctaa_name_pool import build_name_pools
from pathlib import Path


DRY_SEED = 1729
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def test_semantic_maps_are_exhaustive_disjoint_and_rank_stratified() -> None:
    splits = semantic_splits()
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 9,
        "development": 9,
        "confirmation": 9,
    }
    assert len(set().union(*map(set, splits.values()))) == 27
    for split, actions in splits.items():
        assert Counter(len(set(action)) for action in actions) == SPLIT_COUNTS[split]
        for coordinate in range(3):
            assert Counter(action[coordinate] for action in actions) == {0: 3, 1: 3, 2: 3}


def test_training_rows_contain_compiler_labels_but_no_outcomes() -> None:
    rows = build_rows(DRY_SEED, "train", 8)
    assert len(rows) == 32
    assert {row.renderer for row in rows} <= set(RENDERERS["train"])
    for row in rows:
        record = row.training_record()
        assert row.terminal_state is None and row.answer is None
        assert "terminal_state" not in record and "answer" not in record
        assert "QUERY" not in row.program_source
        assert row.query_source not in row.program_source
        assert len(row.schedule) == MAX_STEPS
        assert row.schedule.count(STOP_ID) == 1


def test_scored_depths_are_unseen_and_stop_suffix_is_absorbing() -> None:
    train_depths = {make_family(DRY_SEED, "train", index).depth for index in range(16)}
    scored = [make_family(DRY_SEED, "development", index) for index in range(8)]
    assert train_depths == set(TRAIN_DEPTHS)
    assert {family.depth for family in scored} == set(SCORED_DEPTHS)
    assert train_depths.isdisjoint(SCORED_DEPTHS)
    for family in scored:
        states = family.execute()
        stop = family.depth
        assert all(state == states[stop] for state in states[stop:])


def test_program_classes_separate_raw_depth_from_causal_depth() -> None:
    families = [make_family(DRY_SEED, "development", index) for index in range(96)]
    classes = Counter(family.program_class for family in families)
    assert classes == {"persistent": 32, "mixed_copy": 32, "absorbing": 32}
    for family in families:
        if family.program_class == "persistent":
            assert family.causal_depth == family.depth
            assert len(set(family.composite)) == 3
        elif family.program_class == "mixed_copy":
            assert family.causal_depth >= 4
            assert len(set(family.composite)) == 2
        else:
            assert family.causal_depth < family.depth
            assert len(set(family.composite)) == 1


def test_alpha_names_and_rule_storage_do_not_change_private_targets() -> None:
    family = make_family(DRY_SEED, "development", 0)
    first = render_row(DRY_SEED, family, 0)
    renamed = render_row(DRY_SEED, family, 1)
    reordered = render_row(
        DRY_SEED,
        family,
        0,
        reverse_rule_storage=True,
    )
    assert first.program_source != renamed.program_source
    assert first.program_source != reordered.program_source
    for row in (renamed, reordered):
        assert row.action_cards == first.action_cards
        assert row.initial_state == first.initial_state
        assert row.schedule == first.schedule
        assert row.query_position == first.query_position
        assert row.terminal_state == first.terminal_state
        assert row.answer == first.answer


def test_split_sources_and_renderers_are_disjoint_in_dry_build() -> None:
    boards = {
        split: build_rows(DRY_SEED, split, 12)
        for split in ("train", "development", "confirmation")
    }
    sources = {
        split: {(row.program_source, row.query_source) for row in rows}
        for split, rows in boards.items()
    }
    assert sources["train"].isdisjoint(sources["development"])
    assert sources["train"].isdisjoint(sources["confirmation"])
    assert sources["development"].isdisjoint(sources["confirmation"])
    for split, rows in boards.items():
        assert {row.renderer for row in rows} <= set(RENDERERS[split])
    assert all(len(values) == 16 for values in RENDERERS.values())


def test_production_name_pool_changes_surface_without_changing_packet() -> None:
    pools = build_name_pools(TOKENIZER, per_split=16)
    family = make_family(DRY_SEED, "development", 2)
    dry = render_row(DRY_SEED, family, 0)
    admitted = render_row(DRY_SEED, family, 0, name_pools=pools)
    assert dry.program_source != admitted.program_source
    assert admitted.action_cards == dry.action_cards
    assert admitted.initial_state == dry.initial_state
    assert admitted.schedule == dry.schedule
