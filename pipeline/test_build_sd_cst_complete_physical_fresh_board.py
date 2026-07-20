from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))

from build_sd_cst_board import build_all  # noqa: E402
from build_sd_cst_complete_physical_fresh_board import (  # noqa: E402
    audit_fresh_board,
    rekey_families,
)
from projected_sd_cst_fresh import parse_projected_row  # noqa: E402
from sd_cst_complete_physical_fresh_renderers import (  # noqa: E402
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    expand_rows,
)


def _small_board():
    train, development, confirmation = build_all(
        train_rows=12,
        development_families=6,
        confirmation_families=6,
        seed=9182,
    )
    development = [row for row in development if row["variant"] == "canonical"]
    confirmation = [row for row in confirmation if row["variant"] == "canonical"]
    train = rekey_families(train, seed=9182, namespace="train", forbidden=set())
    used = {
        binding["entity"]
        for row in train
        for binding in row["compiler_targets"]["entity_bindings"]
    }
    development = rekey_families(
        development, seed=9182, namespace="development", forbidden=used
    )
    used.update(
        binding["entity"]
        for row in development
        for binding in row["compiler_targets"]["entity_bindings"]
    )
    confirmation = rekey_families(
        confirmation, seed=9182, namespace="confirmation", forbidden=used
    )
    return (
        expand_rows(train, TRAIN_RENDERERS),
        expand_rows(development, SCORED_RENDERERS),
        expand_rows(confirmation, SCORED_RENDERERS),
    )


def test_small_board_semantics_and_orbits() -> None:
    train, development, confirmation = _small_board()
    report = audit_fresh_board(train, development, confirmation, [])
    gates = report["gates"]
    assert gates["all_rows_semantically_exact"]
    assert gates["all_families_have_four_views"]
    assert gates["train_and_scored_renderer_orbits_disjoint"]
    assert gates["development_confirmation_renderer_orbits_equal"]
    assert gates["training_oracles_absent"]
    assert gates["evaluation_oracles_present"]
    assert gates["opaque_names_fixed_and_globally_unique"]
    for split, rows in (
        ("sd_cst_train", train),
        ("sd_cst_development", development),
        ("sd_cst_confirmation", confirmation),
    ):
        assert all(parse_projected_row(row, split) for row in rows)


def test_row_corruption_fails_semantic_audit() -> None:
    train, development, confirmation = _small_board()
    train[0]["program_text"] = train[0]["program_text"].replace("HALT", "WAIT")
    report = audit_fresh_board(train, development, confirmation, [])
    assert not report["gates"]["all_rows_semantically_exact"]
