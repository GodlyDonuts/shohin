from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from build_sd_cst_board import build_all  # noqa: E402
from sd_cst_complete_physical_fresh_renderers import (  # noqa: E402
    RENDERERS,
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    render_row,
)


def _row():
    train, _, _ = build_all(
        train_rows=6,
        development_families=6,
        confirmation_families=6,
        seed=715,
    )
    return train[0]


def test_renderer_orbit_has_balanced_disjoint_parity() -> None:
    assert len(RENDERERS) == 8
    assert len(TRAIN_RENDERERS) == len(SCORED_RENDERERS) == 4
    assert set(TRAIN_RENDERERS).isdisjoint(SCORED_RENDERERS)
    assert {item.declaration for item in TRAIN_RENDERERS} == {0, 1}
    assert {item.event for item in TRAIN_RENDERERS} == {0, 1}
    assert {item.query for item in TRAIN_RENDERERS} == {0, 1}


@pytest.mark.parametrize("renderer", RENDERERS)
def test_rendered_row_has_exact_physical_shape(renderer) -> None:
    row = render_row(_row(), renderer, row_id="x", family_id="f")
    lines = row["program_text"].splitlines()
    assert len(lines) == 9
    assert sum("HALT" in line for line in lines) == 1
    assert max(len(line.encode("utf-8")) for line in lines) < 144
    assert len(row["late_query_text"].encode("utf-8")) < 144
    assert row["template_id"] == renderer.name
    start, end = row["late_query_target"]["byte_span"]
    assert row["late_query_text"].encode("utf-8")[start:end] in {b"1", b"2", b"3"}


def test_fresh_vocabulary_excludes_consumed_renderer_phrases() -> None:
    text = "\n".join(
        render_row(_row(), renderer, row_id="x", family_id="f")["program_text"]
        for renderer in RENDERERS
    ).lower()
    for phrase in ("bindings:", "registry:", "event ", "action ", "move ", "send "):
        assert phrase not in text
