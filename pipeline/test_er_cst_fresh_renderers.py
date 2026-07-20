from __future__ import annotations

import copy

import pytest

from build_er_cst_fresh_board import (
    TRAIN_SPLIT,
    _make_family,
    _row_exact,
)
from er_cst_fresh_renderers import (
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    parse_rendered_row,
    render_row,
)


def test_renderer_orbits_are_disjoint_and_factor_balanced() -> None:
    assert {item.name for item in TRAIN_RENDERERS}.isdisjoint(
        {item.name for item in SCORED_RENDERERS}
    )
    for renderers in (TRAIN_RENDERERS, SCORED_RENDERERS):
        for factor in range(4):
            assert sorted(item.as_tuple()[factor] for item in renderers) == [0, 0, 1, 1]


def test_every_renderer_round_trips_a_depth_eight_family() -> None:
    base = _make_family(71, TRAIN_SPLIT, 7)
    for index, renderer in enumerate(TRAIN_RENDERERS + SCORED_RENDERERS):
        order = list(range(13))
        order = order[index:] + order[:index]
        row = render_row(
            base,
            renderer,
            storage_order=order,
            row_id=f"row-{index}",
            family_id="family",
        )
        parsed = parse_rendered_row(row)
        assert parsed["events"].index(None) == 8
        assert len(row["program_text"].splitlines()) == 13
        assert _row_exact(row)


def test_renderer_parser_rejects_surface_corruption() -> None:
    base = _make_family(73, TRAIN_SPLIT, 0)
    row = render_row(
        base,
        TRAIN_RENDERERS[0],
        storage_order=list(range(13)),
        row_id="row",
        family_id="family",
    )
    changed = copy.deepcopy(row)
    changed["program_text"] = str(changed["program_text"]).replace(" HALT", " BROKEN")
    with pytest.raises(ValueError, match="HALT"):
        parse_rendered_row(changed)
    assert not _row_exact(changed)


def test_storage_order_must_be_a_complete_permutation() -> None:
    base = _make_family(79, TRAIN_SPLIT, 0)
    with pytest.raises(ValueError, match="storage"):
        render_row(
            base,
            TRAIN_RENDERERS[0],
            storage_order=[0] * 13,
            row_id="row",
            family_id="family",
        )
