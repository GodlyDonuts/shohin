from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from sd_cst_renderer_orbit import (  # noqa: E402
    HELD_OUT_RENDERERS,
    RENDERER_ORBIT,
    TRAIN_RENDERERS,
    orbit_atom_coverage,
    render_row,
)

sys.path.insert(0, str(ROOT / "train"))

from build_sd_cst_board import Operation, make_row  # noqa: E402
from projected_sd_cst_fresh import parse_projected_row  # noqa: E402


def _source_row() -> dict[str, object]:
    program = (
        Operation(0, "right", 1),
        Operation(2, "left", 1),
        Operation(1, "right", 2),
        Operation(0, "left", 1),
        Operation(2, "right", 2),
        Operation(1, "left", 1),
        Operation(0, "right", 2),
    )
    return make_row(
        split="sd_cst_train",
        row_id="orbit-source",
        variant="source",
        names_by_role=("aaaa-11111111", "bbbb-22222222", "cccc-33333333"),
        initial_order=(2, 0, 1),
        program=program,
        halt_after=4,
        query_position=2,
        storage_order=(4, 1, 8, 2, 7, 3, 6, 5),
        template="direct",
    )


def test_even_odd_renderer_orbits_are_combinatorial_holdouts() -> None:
    assert len(RENDERER_ORBIT) == 8
    assert len(TRAIN_RENDERERS) == len(HELD_OUT_RENDERERS) == 4
    assert set(TRAIN_RENDERERS).isdisjoint(HELD_OUT_RENDERERS)
    expected = {"declaration": {0, 1}, "event": {0, 1}, "query": {0, 1}}
    assert orbit_atom_coverage(TRAIN_RENDERERS) == expected
    assert orbit_atom_coverage(HELD_OUT_RENDERERS) == expected


def test_all_renderers_preserve_program_targets_and_query_span() -> None:
    source = _source_row()
    expected = parse_projected_row(source, "sd_cst_train")
    for renderer in RENDERER_ORBIT:
        rendered = render_row(source, renderer)
        parsed = parse_projected_row(rendered, "sd_cst_train")
        assert parsed.initial_state == expected.initial_state
        assert parsed.event_kind == expected.event_kind
        assert parsed.event_identity == expected.event_identity
        assert parsed.amount == expected.amount
        assert parsed.query_position == expected.query_position
        start, end = rendered["late_query_target"]["byte_span"]
        assert rendered["late_query_text"].encode()[start:end] == b"3"
        assert len(rendered["program_text"].splitlines()) == 9


def test_renderer_text_does_not_expose_factor_ids() -> None:
    source = _source_row()
    for renderer in RENDERER_ORBIT:
        rendered = render_row(source, renderer)
        text = rendered["program_text"] + "\n" + rendered["late_query_text"]
        assert "orbit" not in text.lower()
        assert renderer.name not in text
