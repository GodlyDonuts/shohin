from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from build_sd_cst_board import build_all  # noqa: E402
from build_sd_cst_complete_physical_fresh_board import (  # noqa: E402
    audit_fresh_board,
)
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


def test_row_corruption_fails_semantic_audit() -> None:
    train, development, confirmation = _small_board()
    train[0]["program_text"] = train[0]["program_text"].replace("HALT", "WAIT")
    report = audit_fresh_board(train, development, confirmation, [])
    assert not report["gates"]["all_rows_semantically_exact"]
